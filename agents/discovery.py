from tool import TYPES, call_agent, entity_list, unique_entities


def run(payload, tokens, config, client):
    def validate(data):
        additions = entity_list(data, payload["sentence"], tokens, TYPES[config["dataset"]])
        current = [{k: v for k, v in e.items() if k != "id"} for e in payload["candidates"]]
        return unique_entities(current + additions)
    return call_agent("discovery", payload, config, client, validate)
