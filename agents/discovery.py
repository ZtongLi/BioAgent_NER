from tool import TYPES, call_agent, entity_list


def run(payload, tokens, config, client):
    def validate(data):
        return entity_list(data, payload["sentence"], tokens, TYPES[config["dataset"]])
    return call_agent("discovery", payload, config, client, validate)
