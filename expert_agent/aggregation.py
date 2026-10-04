"""Constrain the summarizer to existing candidates and enforce majority locks."""

from tool import call_agent, check_evidence, decisions


def run(payload, config, client, stage="aggregation"):
    def validate(data):
        items = decisions(data, len(payload["candidates"]))
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
    return call_agent(stage, payload, config, client, validate)
