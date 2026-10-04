"""Expert extraction -> candidate pool -> independent ballots -> aggregation."""

import copy
import hashlib
import json

from tool import ROOT, TYPES, prepare_config as prepare_base, retrieve_examples, tokenize, unique_entities
from expert_agent import aggregation, experts, voting

RESULT_ROOT = ROOT / "result" / "expert_agent"
VARIANTS = {
    "single_expert": ("single_expert",),
    "candidate_union": ("experts", "candidate_pool"),
    "vote_only": ("experts", "candidate_pool", "votes", "tally"),
    "without_voting": ("experts", "candidate_pool", "aggregation"),
    "full": ("experts", "candidate_pool", "votes", "tally", "aggregation"),
    "generic_ensemble": ("generic_experts", "candidate_pool", "generic_votes", "tally", "aggregation"),
    "without_retrieval": ("experts", "candidate_pool", "votes", "tally", "aggregation"),
}


def prepare_config(config):
    config = {"max_attempts": None, **config}
    config = prepare_base(config)
    config["agent_system"] = "expert_agent"
    config["system_version"] = "expert_v1"
    config.setdefault("overlap_policy", "forbid")
    voting.conflict_pairs([], config["overlap_policy"])
    config.setdefault("primary_expert", experts.PRIMARY[config["dataset"]])
    if config["primary_expert"] not in experts.ROLES:
        raise ValueError("Unknown primary_expert")
    # Reuse only the shared task contract, never the old stage prompts.
    rules = json.loads((ROOT / "prompts" / "annotation_rules" / f'{config["dataset"]}.json').read_text())
    shared = (ROOT / "prompts" / "shared.txt").read_text() + "\n" + json.dumps({
        "allowed_types": TYPES[config["dataset"]],
        "rules": [{"id": i, "text": r} for i, r in enumerate(rules["rules"])],
        "overlap_policy": config["overlap_policy"]}, ensure_ascii=False)
    directory = ROOT / "expert_agent" / "prompts"
    prompts = {}
    for role, description in experts.ROLES.items():
        for mode in ("extract", "vote"):
            template = (directory / f"{mode}.txt").read_text()
            prompts[f"{mode}_{role}"] = shared + "\n" + description + "\n" + template
            prompts[f"{mode}_generic_{role}"] = shared + "\n" + experts.GENERIC + "\n" + template
    prompts["aggregation"] = shared + "\n" + (directory / "aggregation.txt").read_text()
    config["_prompts"] = prompts
    config["_prompt_hash"] = hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest()
    return config


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

    def invoke(stage, payload, callback):
        key = json.dumps([stage, config["_prompt_hash"], payload], sort_keys=True, ensure_ascii=False)
        hit = key in cache
        if config.get("_progress"):
            config["_progress"](stage, "缓存" if hit else "运行中")
        if hit:
            output, trace = copy.deepcopy(cache[key])
        else:
            output, trace = callback()
            cache[key] = copy.deepcopy((output, trace))
        trace.update(cache_hit=hit, input_entities=copy.deepcopy(entities))
        stages.append(trace)
        return output, trace

    base = {"sentence": sentence, "tokens": [{"id": t["id"], "text": t["text"]} for t in tokens],
            "training_examples": retrieve_examples(sentence, [], config)}
    roles = [config["primary_expert"]] if variant == "single_expert" else list(experts.ROLES)
    prefix = "generic_" if variant == "generic_ensemble" else ""
    for role in roles:
        stage = f"extract_{prefix}{role}"
        output, trace = invoke(stage, base, lambda: experts.extract(stage, base, tokens, config, client))
        if output is not None:
            entities = unique_entities(entities + [{**e, "sources": [role]} for e in output])
        trace["output_entities"] = copy.deepcopy(entities)
        if output is None:
            return result()
    candidates = [{"id": i, **e} for i, e in enumerate(sorted(entities, key=lambda x: (x["start"], x["end"], x["type"])))]
    pairs = voting.conflict_pairs(candidates, config["overlap_policy"])
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
        for role in experts.ROLES:
            stage = f"vote_{prefix}{role}"
            output, trace = invoke(stage, payload, lambda: experts.vote(stage, payload, config, client))
            trace["output_entities"] = copy.deepcopy(entities)
            if output is None:
                return result()
            ballots[role] = output
        counts = voting.tally(candidates, ballots)
        selected = voting.vote_selection(counts, pairs)
        previous = copy.deepcopy(entities)
        entities = [{k: v for k, v in c.items() if k != "id"} for c in candidates if c["id"] in selected]
        stages.append({"stage": "tally", "status": "ok", "attempts": [], "cache_hit": False,
                       "input_entities": previous, "output_entities": copy.deepcopy(entities), "counts": counts})
        if variant == "vote_only":
            return result()
    summary_payload = {**payload, "vote_counts": counts, "ballots": ballots,
                       "locked_decisions": voting.constraints(counts, pairs) if counts else {}}
    output, trace = invoke("aggregation", summary_payload, lambda: aggregation.run(summary_payload, config, client))
    if output is not None:
        selected = {x["id"] for x in output if x["action"] == "keep"}
        entities = [{k: v for k, v in c.items() if k != "id"} for c in candidates if c["id"] in selected]
        trace["decisions"] = output
    trace["output_entities"] = copy.deepcopy(entities)
    return result()
