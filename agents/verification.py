from tool import TYPES, call_agent, check_evidence, decisions, unique_entities


def run(payload, tokens, config, client):
    candidates = payload["candidates"]

    def validate(data):
        kept = []
        for item in decisions(data, len(candidates)):
            action = item.get("action")
            old = candidates[item["id"]]
            if action not in ("keep", "drop", "retype"):
                raise ValueError("Verification action must be keep, drop or retype")
            if action != "keep":
                check_evidence(item, payload, config)
            if action == "drop":
                continue
            entity = {k: v for k, v in old.items() if k != "id"}
            if action == "retype":
                if item.get("type") not in TYPES[config["dataset"]]:
                    raise ValueError("Retyped entity needs an allowed label")
                entity["type"] = item["type"]
            kept.append(entity)
        return unique_entities(kept)

    return call_agent("verification", payload, config, client, validate)
