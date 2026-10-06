import json
import os
import re
import time
import hashlib
import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from openai import APIConnectionError, APIStatusError, OpenAI

ROOT = Path(__file__).resolve().parent
TYPES = {"bc2gm": ["GENE"], "bc5cdr": ["CHEMICAL", "DISEASE"], "ncbi": ["DISEASE"]}


def tokenize(sentence):
    """Keep original character offsets; models select token IDs instead of counting chars."""
    return [{"id": i, "text": m.group(), "start": m.start(), "end": m.end()}
            for i, m in enumerate(re.finditer(r"\w+|[^\w\s]", sentence))]


def unique_entities(entities):
    merged = {}
    for entity in entities:
        key = (entity["start"], entity["end"], entity["type"])
        old = merged.get(key, {})
        merged[key] = {**entity, "sources": sorted(set(
            old.get("sources", []) + entity.get("sources", [])))}
    return list(merged.values())


def locate_entity(item, sentence, tokens, types):
    """Locate exact quotes, never fuzzy-match or guess a repeated occurrence."""
    a, b = item.get("start_token"), item.get("end_token")
    text, label = item.get("text"), item.get("type")
    if not isinstance(text, str) or not text.strip() or label not in types:
        raise ValueError("Each entity needs nonempty exact text and an allowed type.")
    matches = [m.start() for m in re.finditer(f"(?={re.escape(text)})", sentence)]
    occurrence = item.get("occurrence")
    if occurrence is not None:
        if type(occurrence) is not int or not 0 <= occurrence < len(matches):
            raise ValueError("occurrence must index an exact quote in the sentence, starting at 0.")
        start = matches[occurrence]
    elif len(matches) == 1:
        start = matches[0]
    elif type(a) is int and type(b) is int and 0 <= a < b <= len(tokens):
        enclosed = [p for p in matches if tokens[a]["start"] <= p
                    and p + len(text) <= tokens[b - 1]["end"]]
        if len(enclosed) != 1:
            raise ValueError("Repeated quotes require an unambiguous occurrence or token interval.")
        start = enclosed[0]
    else:
        raise ValueError("Quote must exist exactly; repeated quotes require occurrence.")
    end = start + len(text)
    a = next(t["id"] for t in tokens if t["start"] <= start < t["end"])
    b = next(t["id"] + 1 for t in tokens if t["start"] < end <= t["end"])
    return {"text": text, "type": label, "start": start, "end": end,
            "start_token": a, "end_token": b, "occurrence": matches.index(start)}


def entity_list(data, sentence, tokens, types):
    if not isinstance(data, dict) or not isinstance(data.get("entities"), list):
        raise ValueError('Return an object with an "entities" list.')
    if any(not isinstance(item, dict) for item in data["entities"]):
        raise ValueError("Every entity must be an object.")
    return unique_entities([locate_entity(item, sentence, tokens, types)
                            for item in data["entities"]])


def decisions(data, count):
    items = data.get("decisions") if isinstance(data, dict) else None
    if not isinstance(items, list) or any(not isinstance(x, dict) for x in items):
        raise ValueError('Return an object with a "decisions" list.')
    ids = [x.get("id") for x in items]
    if any(type(i) is not int for i in ids) or sorted(ids) != list(range(count)):
        raise ValueError("Return exactly one decision for every supplied candidate ID.")
    if any(not isinstance(x.get("reason"), str) or not x["reason"].strip() for x in items):
        raise ValueError("Each decision needs one short reason.")
    return sorted(items, key=lambda x: x["id"])


def check_evidence(item, payload, config):
    """Validate evidence references for edits; this does not prove semantic correctness."""
    quote = item.get("evidence_text")
    ids = item.get("example_ids", [])
    allowed = {e["id"] for e in payload.get("training_examples", [])}
    rule = item.get("rule_id")
    if not isinstance(quote, str) or not quote.strip() or quote not in payload["sentence"]:
        raise ValueError("An edit needs evidence_text quoted exactly from the sentence.")
    if not isinstance(ids, list) or any(not isinstance(i, str) or i not in allowed for i in ids):
        raise ValueError("example_ids must refer to supplied training examples.")
    if rule is not None and (type(rule) is not int or not 0 <= rule < config["_rule_count"]):
        raise ValueError("rule_id must refer to a supplied task rule.")
    if not ids and rule is None:
        raise ValueError("An edit needs at least one example_id or rule_id.")


def retrieval_terms(text):
    words = re.findall(r"\w+", text.casefold())
    return Counter(words + [a + " " + b for a, b in zip(words, words[1:])])


def training_example(row, index):
    sentence = row["sentence"]
    entities = []
    for item in row["entities"]:
        start, end = item["pos"]
        text = sentence[start:end]
        if text != item["name"] or not text:
            raise ValueError("Invalid training annotation")
        positions = [m.start() for m in re.finditer(f"(?={re.escape(text)})", sentence)]
        entities.append({"text": text, "type": item["type"], "occurrence": positions.index(start)})
    return {"id": f"train:{index}", "sentence": sentence, "entities": entities}


@lru_cache(maxsize=3)
def build_retriever(path, digest):
    """Small in-memory BM25 index: training text only, no vector service."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Training file changed while preparing retrieval")
    examples, postings, lengths = {}, defaultdict(list), {}
    for i, row in enumerate(json.loads(raw)):
        if len(tokenize(row["sentence"])) > 220:
            continue
        example = training_example(row, i)
        terms = retrieval_terms(row["sentence"])
        if not terms:
            continue
        examples[i], lengths[i] = example, sum(terms.values())
        for word, count in terms.items():
            postings[word].append((i, count))
    if not examples:
        raise ValueError("Training retrieval pool is empty")
    return examples, postings, lengths, sum(lengths.values()) / len(lengths)


def retrieve_examples(sentence, candidates, config):
    k = config["retrieval_k"]
    if not k:
        return []
    examples, postings, lengths, average = config["_retriever"]
    query = retrieval_terms(sentence + " " + " ".join(e["text"] for e in candidates for _ in range(2)))
    scores = defaultdict(float)
    for term, weight in query.items():
        hits = postings.get(term, [])
        idf = math.log(1 + (len(examples) - len(hits) + 0.5) / (len(hits) + 0.5))
        for i, frequency in hits:
            scores[i] += min(weight, 3) * idf * frequency * 2.2 / (
                frequency + 1.2 * (0.25 + 0.75 * lengths[i] / average))
    normalize = lambda text: " ".join(text.casefold().split())
    seen, ranked = {normalize(sentence)}, []
    for i in sorted(scores, key=lambda i: (-scores[i], i)):
        text = normalize(examples[i]["sentence"])
        if text not in seen:
            ranked.append(i)
            seen.add(text)
    selected = ranked[:k]
    # One relevant negative helps define the annotation scope; never force unrelated examples.
    if k > 1 and selected and all(examples[i]["entities"] for i in selected):
        negative = next((i for i in ranked if not examples[i]["entities"]
                         and scores[i] >= scores[ranked[0]] * 0.5), None)
        if negative is not None:
            selected[-1] = negative
    return [examples[i] for i in selected]


def prepare_config(config):
    config = dict(config)
    if config.get("dataset") not in TYPES:
        raise ValueError("dataset must be bc2gm, bc5cdr or ncbi")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", config["model_name"]):
        raise ValueError("model_name must be safe for an output directory name")
    limit = config.get("max_attempts", 3)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("max_attempts must be null (unlimited) or a positive integer")
    config.setdefault("retrieval_k", 4)
    if type(config["retrieval_k"]) is not int or not 0 <= config["retrieval_k"] <= 8:
        raise ValueError("retrieval_k must be an integer from 0 to 8")
    rules_path = ROOT / "prompts" / "annotation_rules" / f'{config["dataset"]}.json'
    rules = json.loads(rules_path.read_text())
    sources = {e["source"] for e in rules["examples"]}
    if len(sources) != 1:
        raise ValueError("Task must identify one training split")
    train_path = (ROOT / sources.pop()).resolve()
    if train_path.name != "train.json" or not train_path.is_relative_to((ROOT / "bioDataset").resolve()):
        raise ValueError("Retrieval accepts only bioDataset/*/train.json")
    config["training_file"] = str(train_path.relative_to(ROOT))
    config["training_sha256"] = hashlib.sha256(train_path.read_bytes()).hexdigest()
    config["_retriever"] = build_retriever(str(train_path), config["training_sha256"]) if config["retrieval_k"] else None
    config["_rule_count"] = len(rules["rules"])
    return config


def make_client(config):
    key = config.get("api_keys") or os.getenv(config.get("api_key_env", ""))
    if not key:
        raise ValueError("Set api_keys in the config or the configured API key environment variable.")
    return OpenAI(api_key=key, base_url=config["base_url"],
                  timeout=config.get("timeout", 120), max_retries=0)


def call_agent(stage, payload, config, client, validate):
    """Retry until validated success when max_attempts is None; Ctrl+C can cancel."""
    messages = [{"role": "system", "content": config["_prompts"][stage]},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    trace = {"stage": stage, "status": "failed", "attempts": [],
             "training_examples": payload.get("training_examples", [])}
    limit = config.get("max_attempts", 3)
    attempt = 0
    while limit is None or attempt < limit:
        if config.get("_progress"):
            config["_progress"](stage, f"第 {attempt + 1} 次请求" if limit is None else f"请求 {attempt + 1}/{limit}")
        time.sleep(config.get("sleep_seconds", 0))
        started = time.monotonic()
        entry = {"attempt": attempt + 1, "raw_response": None, "error": None}
        retryable = True
        retry_after = min(config.get("retry_delay", 2) * 2 ** min(attempt, 10), 30)
        value = None
        try:
            response = client.chat.completions.create(
                model=config["model_name"], messages=messages,
                temperature=config.get("temperature", 1))
            entry["raw_response"] = response.choices[0].message.content
            entry["usage"] = response.usage.model_dump() if response.usage else None
            if response.choices[0].finish_reason == "length":
                raise ValueError("Response was truncated; return a shorter complete JSON object.")
            raw = entry["raw_response"]
            if not isinstance(raw, str):
                raise ValueError("Expected JSON text in the response.")
            value = validate(json.loads(raw))
            trace["status"] = "ok"
        except APIStatusError as exc:
            entry["error"] = f"HTTP {exc.status_code}"
            secret = config.get("api_keys") or os.getenv(config.get("api_key_env", ""))
            detail = exc.response.text
            entry["error_detail"] = (detail.replace(secret, "[REDACTED]") if secret else detail)[:2000]
            trace["fatal"] = limit is not None and exc.status_code in (400, 401, 403, 404)
            retryable = limit is None or exc.status_code in (408, 409, 429) or exc.status_code >= 500
            try:
                retry_after = max(retry_after, min(float(exc.response.headers.get("retry-after", 0)), 120))
            except ValueError:
                pass
        except APIConnectionError:
            entry["error"] = "APIConnectionError/timeout"
        except (ValueError, TypeError, KeyError, IndexError, StopIteration) as exc:
            entry["error"] = "Invalid response: " + str(exc)[:300]
            previous = entry["raw_response"]
            messages = messages[:2] + ([{"role": "assistant", "content": previous[:16000]}]
                                       if isinstance(previous, str) and previous else []) + [{"role": "user", "content":
                "The previous response was invalid: " + entry["error"] +
                " Return complete corrected JSON; preserve valid entities and fix the reported defect."}]
        entry["elapsed_seconds"] = round(time.monotonic() - started, 3)
        trace["attempts"].append(entry)
        if config.get("_attempt_logger"):
            config["_attempt_logger"].info("stage=%s attempt=%s", stage, json.dumps(entry, ensure_ascii=False))
        if trace["status"] == "ok":
            return value, trace
        if not retryable:
            break
        attempt += 1
        if limit is None or attempt < limit:
            if config.get("_progress"):
                config["_progress"](stage, f"等待重试 {retry_after:g}s")
            time.sleep(retry_after)
    return None, trace
