import json
import os
import re
import time
import hashlib
from pathlib import Path
from openai import APIConnectionError, APIStatusError, OpenAI

ROOT = Path(__file__).resolve().parent
TYPES = {"bc2gm": ["GENE"], "bc5cdr": ["CHEMICAL", "DISEASE"], "ncbi": ["DISEASE"]}


def tokenize(sentence):
    """Keep original character offsets; models select token IDs instead of counting chars."""
    return [{"id": i, "text": m.group(), "start": m.start(), "end": m.end()}
            for i, m in enumerate(re.finditer(r"\w+|[^\w\s]", sentence))]


def unique_entities(entities):
    return list({(e["start"], e["end"], e["type"]): e for e in entities}.values())


def locate_entity(item, sentence, tokens, types):
    """A quoted string must occur uniquely inside the selected token interval."""
    a, b = item.get("start_token"), item.get("end_token")
    text, label = item.get("text"), item.get("type")
    if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(tokens):
        raise ValueError("Use valid integer start_token and exclusive end_token IDs.")
    if not isinstance(text, str) or not text.strip() or label not in types:
        raise ValueError("Each entity needs nonempty exact text and an allowed type.")
    left, right = tokens[a]["start"], tokens[b - 1]["end"]
    matches = list(re.finditer(f"(?={re.escape(text)})", sentence[left:right]))
    if len(matches) != 1:
        raise ValueError("Quoted text must occur exactly once within its token interval.")
    start = left + matches[0].start()
    end = start + len(text)
    if start >= tokens[a]["end"] or end <= tokens[b - 1]["start"]:
        raise ValueError("Token IDs must tightly enclose the quoted entity.")
    return {"text": text, "type": label, "start": start, "end": end,
            "start_token": a, "end_token": b}


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


def prepare_config(config):
    config = dict(config)
    if config.get("dataset") not in TYPES:
        raise ValueError("dataset must be bc2gm, bc5cdr or ncbi")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", config["model_name"]):
        raise ValueError("model_name must be safe for an output directory name")
    if type(config.get("max_attempts", 3)) is not int or config.get("max_attempts", 3) < 1:
        raise ValueError("max_attempts must be a positive integer")
    rules_path = ROOT / "prompts" / "annotation_rules" / f'{config["dataset"]}.json'
    rules = json.loads(rules_path.read_text())
    examples = []
    for example in rules["examples"]:
        sentence = example["sentence"]
        tokens = tokenize(sentence)
        entities = []
        for item in example["entities"]:
            start, end = item["pos"]
            assert sentence[start:end] == item["name"], "Invalid training example"
            entities.append({"text": item["name"], "type": item["type"],
                             "start_token": next(t["id"] for t in tokens if t["start"] <= start < t["end"]),
                             "end_token": next(t["id"] + 1 for t in tokens if t["start"] < end <= t["end"])})
        examples.append({"sentence": sentence, "tokens": [{"id": t["id"], "text": t["text"]} for t in tokens],
                         "entities": entities})
    context = json.dumps({"allowed_types": TYPES[config["dataset"]], "rules": rules["rules"],
                          "training_examples": examples}, ensure_ascii=False)
    shared = (ROOT / "prompts" / "shared.txt").read_text() + "\n" + context
    config["_prompts"] = {stage: shared + "\n" + (ROOT / "prompts" / f"{stage}.txt").read_text()
                          for stage in ("extraction", "discovery", "boundary", "verification")}
    config["_prompt_hash"] = hashlib.sha256(json.dumps(config["_prompts"], sort_keys=True).encode()).hexdigest()
    return config


def make_client(config):
    key = config.get("api_keys") or os.getenv(config.get("api_key_env", ""))
    if not key:
        raise ValueError("Set api_keys in the config or the configured API key environment variable.")
    return OpenAI(api_key=key, base_url=config["base_url"],
                  timeout=config.get("timeout", 120), max_retries=0)


def call_agent(stage, payload, config, client, validate):
    """Return validated data and an auditable trace; exhausted attempts return None."""
    messages = [{"role": "system", "content": config["_prompts"][stage]},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    trace = {"stage": stage, "status": "failed", "attempts": []}
    for attempt in range(config.get("max_attempts", 3)):
        time.sleep(config.get("sleep_seconds", 0))
        started = time.monotonic()
        entry = {"attempt": attempt + 1, "raw_response": None, "error": None}
        retryable = True
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
            trace["fatal"] = exc.status_code in (400, 401, 403, 404)
            retryable = exc.status_code in (408, 409, 429) or exc.status_code >= 500
        except APIConnectionError:
            entry["error"] = "APIConnectionError/timeout"
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            entry["error"] = "Invalid response: " + str(exc)[:300]
            messages = messages[:2] + [{"role": "user", "content":
                "The previous response was invalid: " + entry["error"] +
                " Return the complete corrected JSON using the original sentence and token IDs."}]
        entry["elapsed_seconds"] = round(time.monotonic() - started, 3)
        trace["attempts"].append(entry)
        if trace["status"] == "ok":
            return value, trace
        if not retryable:
            break
        if attempt + 1 < config.get("max_attempts", 3):
            time.sleep(min(config.get("retry_delay", 2) * 2 ** attempt, 30))
    return None, trace
