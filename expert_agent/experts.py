"""Role prompts and validated expert responses; no evaluation labels enter here."""

from tool import TYPES, call_agent, decisions, entity_list

ROLES = {
    "molecular": "Molecular Biology Expert: emphasize genes, proteins and molecular context.",
    "clinical": "Clinical Medicine Expert: emphasize diseases, disorders and clinical context.",
    "pharmacology": "Pharmacology Expert: emphasize drugs, chemicals and pharmacological context.",
}
GENERIC = "Biomedical NER annotator: examine all allowed target types using the task rules."
PRIMARY = {"bc2gm": "molecular", "bc5cdr": "pharmacology", "ncbi": "clinical"}


def evidence(item, payload):
    quote = item.get("evidence_text")
    if not isinstance(quote, str) or not quote.strip() or quote not in payload["sentence"]:
        raise ValueError("evidence_text must be a nonempty exact quote from the sentence.")
    if not isinstance(item.get("reason"), str) or not item["reason"].strip():
        raise ValueError("Provide a short specific reason.")


def extract(stage, payload, tokens, config, client):
    def validate(data):
        entities = entity_list(data, payload["sentence"], tokens, TYPES[config["dataset"]])
        for item in data["entities"]:
            evidence(item, payload)
        return entities
    return call_agent(stage, payload, config, client, validate)


def vote(stage, payload, config, client):
    def validate(data):
        items = decisions(data, len(payload["candidates"]))
        for item in items:
            if item.get("vote") not in ("support", "oppose", "abstain"):
                raise ValueError("vote must be support, oppose or abstain.")
            evidence(item, payload)
        return items
    return call_agent(stage, payload, config, client, validate)
