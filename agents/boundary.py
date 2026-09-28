from tool import call_agent, check_evidence, decisions, locate_entity, unique_entities


def run(payload, tokens, config, client):
    candidates = payload["candidates"]

    def validate(data):
        corrected = []
        for item in decisions(data, len(candidates)):
            old = candidates[item["id"]]
            if item.get("action") == "keep":
                corrected.append({k: v for k, v in old.items() if k != "id"})
                continue
            if item.get("action") != "revise":
                raise ValueError("Boundary action must be keep or revise")
            check_evidence(item, payload, config)
            new = locate_entity({**item, "type": old["type"]}, payload["sentence"], tokens, [old["type"]])
            if new["start"] >= old["end"] or old["start"] >= new["end"]:
                raise ValueError("A boundary correction must overlap its original mention.")
            corrected.append({**new, "sources": old.get("sources", []),
                              "original_text": old["text"], "boundary_reason": item["reason"]})
        return unique_entities(corrected)

    return call_agent("boundary", payload, config, client, validate)
