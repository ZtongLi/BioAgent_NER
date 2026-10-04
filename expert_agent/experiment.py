"""Reuse the established experiment writer with a separate expert backend/root."""

from experiment import run_experiment as run_shared
from tool import make_client
from expert_agent import pipeline


def run_experiment(config, variant="full", limit=None, client=None):
    return run_shared(config, variant, limit, client, backend=pipeline)


def predict_sentence(sentence, config, variant="full"):
    variants = list(pipeline.VARIANTS) if variant == "all" else ["single_expert", "full"] if variant == "compare" else [variant]
    if any(name not in pipeline.VARIANTS for name in variants):
        raise ValueError("Unknown expert ablation variant")
    config = pipeline.prepare_config(config)
    cache = {}
    with make_client(config) as client:
        return {name: pipeline.run_pipeline(sentence, config, client, name, cache) for name in variants}
