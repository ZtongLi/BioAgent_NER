"""Small experiment runner: module switches, exact-span evaluation and saved traces."""

import copy
import csv
import hashlib
import json
import logging
import time
from contextlib import ExitStack, nullcontext
from datetime import datetime
from pathlib import Path

from agents import extraction, discovery, boundary, verification
from tool import ROOT, make_client, prepare_config, tokenize

MODULES = {"extraction": extraction, "discovery": discovery,
           "boundary": boundary, "verification": verification}
VARIANTS = {
    "extract_only": (),
    "full": ("discovery", "boundary", "verification"),
    "without_discovery": ("boundary", "verification"),
    "without_boundary": ("discovery", "verification"),
    "without_verification": ("discovery", "boundary"),
}
RESULT_ROOT = ROOT / "result" / "ner_agent"


def run_pipeline(sentence, config, client, variant="full", cache=None):
    """Only inference inputs enter agents. The per-sentence cache pairs ablations."""
    if cache is None:
        cache = {}
    tokens = tokenize(sentence)
    entities, stages = [], []
    for stage in ("extraction", *VARIANTS[variant]):
        previous = copy.deepcopy(entities)
        payload = {"sentence": sentence,
                   "tokens": [{"id": t["id"], "text": t["text"]} for t in tokens],
                   "candidates": [{"id": i, **e} for i, e in enumerate(entities)]}
        key = json.dumps([stage, payload], ensure_ascii=False, sort_keys=True)
        cache_hit = key in cache
        if cache_hit:
            output, trace = copy.deepcopy(cache[key])
        elif stage in ("boundary", "verification") and not entities:
            output, trace = [], {"stage": stage, "status": "skipped_empty", "attempts": []}
        else:
            output, trace = MODULES[stage].run(payload, tokens, config, client)
            cache[key] = copy.deepcopy((output, trace))
        if output is not None:
            entities = output
        trace.update({"cache_hit": cache_hit, "input_entities": previous,
                      "output_entities": copy.deepcopy(entities)})
        stages.append(trace)
        if trace.get("fatal") or (stage == "extraction" and trace["status"] == "failed"):
            break
    return {"pred_entities": entities, "stages": stages,
            "fully_executed": all(s["status"] != "failed" for s in stages)}


def span_set(entities):
    return {(*e["pos"], e["type"]) if "pos" in e else (e["start"], e["end"], e["type"])
            for e in entities or []}


def evaluate_ner(records):
    """All supplied samples count, including request/format failures as empty or fallback."""
    tp = fp = fn = 0
    effects = {}
    for row in records:
        gold, pred = span_set(row["gold_entities"]), span_set(row["pred_entities"])
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
        for stage in row.get("stages", []):
            before, after = span_set(stage["input_entities"]), span_set(stage["output_entities"])
            added, removed = after - before, before - after
            totals = effects.setdefault(stage["stage"], dict.fromkeys(
                ["samples", "failed_samples", "true_added", "false_added", "true_removed", "false_removed"], 0))
            for key, value in {"samples": 1, "failed_samples": int(stage["status"] == "failed"),
                               "true_added": len(added & gold), "false_added": len(added - gold),
                               "true_removed": len(removed & gold), "false_removed": len(removed - gold)}.items():
                totals[key] += value
    return {"precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "num_samples": len(records),
            "failed_samples": sum(not r.get("fully_executed", True) for r in records),
            "stage_effects": effects}


def save_results(run_dir, config, variant, records):
    """Finalize completed runs; the runner has already flushed each prediction and log."""
    metrics = evaluate_ner(records)
    stages = [s for r in records for s in r["stages"]]
    metrics.update({"status": "complete", "run_id": run_dir.name, "dataset": config["dataset"],
                    "model_name": config["model_name"], "variant": variant,
                    "prompt_sha256": config["_prompt_hash"],
                    "api_calls": sum(len(s["attempts"]) for s in stages if not s["cache_hit"]),
                    "logical_attempts": sum(len(s["attempts"]) for s in stages),
                    "cache_hits": sum(s["cache_hit"] for s in stages),
                    "duration_seconds": round(sum(r["elapsed_seconds"] for r in records), 3)})
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    summary = RESULT_ROOT / "summary_prf1.csv"
    columns = [k for k in metrics if k != "stage_effects"]
    new_file = not summary.exists()
    with summary.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if new_file:
            writer.writeheader()
        writer.writerow({k: metrics[k] for k in columns})
    return {"result_dir": str(run_dir), **metrics}


def run_experiment(config, variant="full", limit=None, client=None):
    config = prepare_config(config)
    variants = list(VARIANTS) if variant == "all" else [variant]
    if any(name not in VARIANTS for name in variants):
        raise ValueError("Unknown ablation variant")
    count = config.get("max_loop", 500) if limit is None else limit
    if type(count) is not int or count < 1:
        raise ValueError("Sample limit must be a positive integer")
    data_path = (ROOT / config["test_file_path"]).resolve()
    if not data_path.is_relative_to((ROOT / "bioDataset").resolve()):
        raise ValueError("Evaluation data must come from bioDataset")
    data_bytes = data_path.read_bytes()
    data = json.loads(data_bytes)[:count]
    if not data:
        raise ValueError("Evaluation dataset is empty")
    snapshot = {k: v for k, v in config.items()
                if not k.startswith("_") and not any(word in k.lower() for word in ("key", "secret", "token"))}
    snapshot.update({"requested_limit": count, "actual_samples": len(data),
                     "input_sha256": hashlib.sha256(data_bytes).hexdigest(),
                     "prompt_sha256": config["_prompt_hash"], "prompts": config["_prompts"]})
    files = [ROOT / "main.py", ROOT / "experiment.py", ROOT / "tool.py", *sorted((ROOT / "agents").glob("*.py"))]
    snapshot["code_sha256"] = hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    runs = {}
    with ExitStack() as stack:
        api = stack.enter_context(nullcontext(client) if client is not None else make_client(config))
        for name in variants:
            directory = RESULT_ROOT / config["dataset"] / config["model_name"] / name / run_id
            directory.mkdir(parents=True, exist_ok=False)
            (directory / "run_config.json").write_text(json.dumps(
                {**snapshot, "variant": name, "modules": ["extraction", *VARIANTS[name]]}, ensure_ascii=False, indent=2))
            output = stack.enter_context((directory / "predictions.jsonl").open("w", encoding="utf-8"))
            handler = logging.FileHandler(directory / "run.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger = logging.getLogger(f"ner_agent.{run_id}.{name}")
            logger.setLevel(logging.INFO)
            logger.propagate = False
            logger.addHandler(handler)
            stack.callback(handler.close)
            stack.callback(logger.removeHandler, handler)
            runs[name] = {"directory": directory, "output": output, "logger": logger, "records": []}
        for index, sample in enumerate(data):
            cache = {}
            for name, run in runs.items():
                started = time.monotonic()
                prediction = run_pipeline(sample["sentence"], config, api, name, cache)
                record = {"sample_id": index, "sentence": sample["sentence"], **prediction,
                          "gold_entities": sample["entities"],
                          "elapsed_seconds": round(time.monotonic() - started, 3)}
                run["output"].write(json.dumps(record, ensure_ascii=False) + "\n")
                run["output"].flush()
                run["records"].append(record)
                for stage in record["stages"]:
                    run["logger"].info("sample=%s stage=%s trace=%s", index, stage["stage"], json.dumps(stage, ensure_ascii=False))
                print(f"[{index + 1}/{len(data)}] {config['dataset']} {config['model_name']} {name} "
                      f"entities={len(record['pred_entities'])} complete={record['fully_executed']}", flush=True)
                if any(s.get("fatal") for s in record["stages"]):
                    raise RuntimeError(f"API configuration rejected (HTTP 400/401/403/404); inspect {run['directory'] / 'run.log'}")
        return [save_results(run["directory"], config, name, run["records"]) for name, run in runs.items()]


def predict_sentence(sentence, config, variant="full"):
    config = prepare_config(config)
    variants = list(VARIANTS) if variant == "all" else [variant]
    cache = {}
    with make_client(config) as client:
        return {name: run_pipeline(sentence, config, client, name, cache) for name in variants}
