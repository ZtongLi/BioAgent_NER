from tool import call_agent, decisions


def run(payload, tokens, config, client):
    candidates = payload["candidates"]

    def validate(data):
        kept = []
        for item in decisions(data, len(candidates)):
            if type(item.get("keep")) is not bool:
                raise ValueError("keep must be a JSON boolean.")
            if item["keep"]:
                kept.append({k: v for k, v in candidates[item["id"]].items() if k != "id"})
        return kept

    return call_agent("verification", payload, config, client, validate)
