"""Dataset configuration, prompt assembly and training-only BM25 retrieval."""

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TYPES = {"bc2gm": ["GENE"], "bc5cdr": ["CHEMICAL", "DISEASE"], "ncbi": ["DISEASE"]}
VARIANTS = {
    "single_expert": ("single_expert",),
    "candidate_union": ("experts", "candidate_pool"),
    "vote_only": ("experts", "candidate_pool", "votes", "tally"),
    "without_voting": ("experts", "candidate_pool", "aggregation"),
    "full": ("experts", "candidate_pool", "votes", "tally", "aggregation"),
    "generic_ensemble": ("generic_experts", "candidate_pool", "generic_votes", "tally", "aggregation"),
    "without_retrieval": ("experts", "candidate_pool", "votes", "tally", "aggregation"),
}
_roles = json.loads((ROOT / "expert_agent/prompts/roles.json").read_text())
ROLES, GENERIC, PRIMARY = (_roles[key] for key in ("roles", "generic", "primary"))


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


def prepare_config(config):
    config = {"max_attempts": None, **config}
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
    config["agent_system"] = "expert_agent"
    config["system_version"] = "expert_v1"
    config.setdefault("overlap_policy", "forbid")
    if config["overlap_policy"] not in ("forbid", "allow_nested"):
        raise ValueError("overlap_policy must be forbid or allow_nested")
    config.setdefault("primary_expert", PRIMARY[config["dataset"]])
    if config["primary_expert"] not in ROLES:
        raise ValueError("Unknown primary_expert")
    shared = (ROOT / "prompts" / "shared.txt").read_text() + "\n" + json.dumps({
        "allowed_types": TYPES[config["dataset"]],
        "rules": [{"id": i, "text": r} for i, r in enumerate(rules["rules"])],
        "overlap_policy": config["overlap_policy"]}, ensure_ascii=False)
    directory = ROOT / "expert_agent" / "prompts"
    templates = {mode: (directory / f"{mode}.txt").read_text() for mode in ("extract", "vote", "aggregation")}
    prompts = {}
    for role, description in ROLES.items():
        for mode in ("extract", "vote"):
            template = templates[mode]
            prompts[f"{mode}_{role}"] = shared + "\n" + description + "\n" + template
            prompts[f"{mode}_generic_{role}"] = shared + "\n" + GENERIC + "\n" + template
    prompts["aggregation"] = shared + "\n" + templates["aggregation"]
    config["_prompts"] = prompts
    config["_prompt_hash"] = hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest()
    return config


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
