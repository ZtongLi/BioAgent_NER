"""Expert extraction, independent voting and constrained aggregation."""

import copy
import json

from data import TYPES, ROLES, VARIANTS, prepare_config, tokenize, unique_entities, retrieve_examples
from tool import make_client, call_agent, entity_list, decisions, check_evidence, evidence

def validate_response(stage, data, payload, config, tokens):
    if stage.startswith("extract_"):
        entities = entity_list(data, payload["sentence"], tokens, TYPES[config["dataset"]])
        for item in data["entities"]:
            evidence(item, payload)
        return entities
    items = decisions(data, len(payload["candidates"]))
    if stage.startswith("vote_"):
        for item in items:
            if item.get("vote") not in ("support", "oppose", "abstain"):
                raise ValueError("vote must be support, oppose or abstain.")
            evidence(item, payload)
        return items
    kept = set()
    locks = {int(k): v for k, v in payload["locked_decisions"].items()}
    for item in items:
        action = item.get("action")
        if action not in ("keep", "drop"):
            raise ValueError("action must be keep or drop; do not create or edit candidates.")
        if item["id"] in locks and action != locks[item["id"]]:
            raise ValueError("An unambiguous majority decision is locked.")
        check_evidence(item, payload, config)
        if action == "keep":
            kept.add(item["id"])
    if any(a in kept and b in kept for a, b in payload["conflict_pairs"]):
        raise ValueError("Cannot retain mutually incompatible candidates.")
    return items


def conflict_pairs(candidates, overlap_policy):
    if overlap_policy not in ("forbid", "allow_nested"):
        raise ValueError("overlap_policy must be forbid or allow_nested")
    pairs = []
    for i, a in enumerate(candidates):
        for b in candidates[i + 1:]:
            overlap = max(a["start"], b["start"]) < min(a["end"], b["end"])
            same_span = (a["start"], a["end"]) == (b["start"], b["end"])
            nested = ((a["start"] <= b["start"] and b["end"] <= a["end"])
                      or (b["start"] <= a["start"] and a["end"] <= b["end"]))
            if overlap and (overlap_policy == "forbid" or same_span or not nested):
                pairs.append([a["id"], b["id"]])
    return pairs


def tally(candidates, ballots):
    if len(ballots) != 3:
        raise ValueError("Exactly three successful expert ballots are required.")
    expected = list(range(len(candidates)))
    for items in ballots.values():
        if sorted(x["id"] for x in items) != expected:
            raise ValueError("Each expert must vote exactly once on every candidate.")
    counts = []
    for candidate in candidates:
        votes = [next(x["vote"] for x in items if x["id"] == candidate["id"])
                 for items in ballots.values()]
        if any(v not in ("support", "oppose", "abstain") for v in votes):
            raise ValueError("Invalid vote")
        count = {v: votes.count(v) for v in ("support", "oppose", "abstain")}
        status = "accepted" if count["support"] >= 2 else "rejected" if count["oppose"] >= 2 else "unresolved"
        counts.append({"id": candidate["id"], **count, "status": status})
    return counts


def constraints(counts, pairs):
    """A majority-supported candidate stays locked unless another non-rejected one conflicts."""
    status = {x["id"]: x["status"] for x in counts}
    contested = {i for pair in pairs if all(status[i] != "rejected" for i in pair) for i in pair}
    return {i: "drop" if value == "rejected" else "keep"
            for i, value in status.items() if value == "rejected" or (value == "accepted" and i not in contested)}


def vote_selection(counts, pairs):
    accepted = {x["id"] for x in counts if x["status"] == "accepted"}
    conflicting = {i for pair in pairs if all(i in accepted for i in pair) for i in pair}
    return accepted - conflicting


def run_pipeline(sentence, config, client, variant="full", cache=None):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown expert variant: {variant}")
    config = {**config, "retrieval_k": 0} if variant == "without_retrieval" else config
    cache = {} if cache is None else cache
    tokens = tokenize(sentence)
    stages, entities, ballots = [], [], {}
    candidates, counts, pairs = [], [], []

    def result():
        success = all(s["status"] != "failed" for s in stages)
        return {"pred_entities": copy.deepcopy(entities) if success else [], "stages": stages,
                "fully_executed": success, "candidates": candidates,
                "ballots": ballots, "vote_counts": counts, "conflict_pairs": pairs}

    def invoke(stage, payload):
        key = json.dumps([stage, config["_prompt_hash"], payload], sort_keys=True, ensure_ascii=False)
        hit = key in cache
        if hit:
            output, trace = copy.deepcopy(cache[key])
        else:
            output, trace = call_agent(stage, payload, config, client,
                                       lambda data: validate_response(stage, data, payload, config, tokens))
            cache[key] = copy.deepcopy((output, trace))
        trace.update(cache_hit=hit, input_entities=copy.deepcopy(entities))
        stages.append(trace)
        return output, trace

    base = {"sentence": sentence, "tokens": [{"id": t["id"], "text": t["text"]} for t in tokens],
            "training_examples": retrieve_examples(sentence, [], config)}
    roles = [config["primary_expert"]] if variant == "single_expert" else list(ROLES)
    prefix = "generic_" if variant == "generic_ensemble" else ""
    for role in roles:
        stage = f"extract_{prefix}{role}"
        output, trace = invoke(stage, base)
        if output is not None:
            entities = unique_entities(entities + [{**e, "sources": [role]} for e in output])
        trace["output_entities"] = copy.deepcopy(entities)
        if output is None:
            return result()
    candidates = [{"id": i, **e} for i, e in enumerate(sorted(entities, key=lambda x: (x["start"], x["end"], x["type"])))]
    pairs = conflict_pairs(candidates, config["overlap_policy"])
    if variant in ("single_expert", "candidate_union"):
        return result()
    if not candidates:
        stages.append({"stage": "empty_pool", "status": "skipped_empty", "attempts": [],
                       "cache_hit": False, "input_entities": [], "output_entities": []})
        return result()
    # The ballot input contains neither provenance nor prior responses/votes.
    payload = {**base, "candidates": [{k: v for k, v in c.items() if k != "sources"} for c in candidates],
               "conflict_pairs": pairs}
    if variant != "without_voting":
        for role in ROLES:
            stage = f"vote_{prefix}{role}"
            output, trace = invoke(stage, payload)
            trace["output_entities"] = copy.deepcopy(entities)
            if output is None:
                return result()
            ballots[role] = output
        counts = tally(candidates, ballots)
        selected = vote_selection(counts, pairs)
        previous = copy.deepcopy(entities)
        entities = [{k: v for k, v in c.items() if k != "id"} for c in candidates if c["id"] in selected]
        stages.append({"stage": "tally", "status": "ok", "attempts": [], "cache_hit": False,
                       "input_entities": previous, "output_entities": copy.deepcopy(entities), "counts": counts})
        if variant == "vote_only":
            return result()
    summary_payload = {**payload, "vote_counts": counts, "ballots": ballots,
                       "locked_decisions": constraints(counts, pairs) if counts else {}}
    output, trace = invoke("aggregation", summary_payload)
    if output is not None:
        selected = {x["id"] for x in output if x["action"] == "keep"}
        entities = [{k: v for k, v in c.items() if k != "id"} for c in candidates if c["id"] in selected]
        trace["decisions"] = output
    trace["output_entities"] = copy.deepcopy(entities)
    return result()


def predict_sentence(sentence, config, variant="full"):
    variants = list(VARIANTS) if variant == "all" else ["single_expert", "full"] if variant == "compare" else [variant]
    if any(name not in VARIANTS for name in variants):
        raise ValueError("Unknown expert ablation variant")
    config = prepare_config(config)
    cache = {}
    with make_client(config) as client:
        return {name: run_pipeline(sentence, config, client, name, cache) for name in variants}


def print_summary(results):
    for row in results:
        scores = " ".join(f"{key}={row[key]:.4f}" if row[key] is not None else f"{key}=NA"
                          for key in ("precision", "recall", "f1"))
        print(f"{row['variant']} {scores} results={row['result_dir']}", flush=True)
