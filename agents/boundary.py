from tool import call_agent, decisions, locate_entity, unique_entities


def run(payload, tokens, config, client):
    candidates = payload["candidates"]

    def validate(data):
        corrected = []
        for item in decisions(data, len(candidates)):
            old = candidates[item["id"]]
            new = locate_entity({**item, "type": old["type"]}, payload["sentence"], tokens, [old["type"]])
            if new["start"] >= old["end"] or old["start"] >= new["end"]:
                raise ValueError("A boundary correction must overlap its original mention.")
            corrected.append(new)
        return unique_entities(corrected)

    return call_agent("boundary", payload, config, client, validate)
