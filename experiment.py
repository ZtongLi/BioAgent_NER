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
from tempfile import NamedTemporaryFile

from agents import extraction, discovery, boundary, verification
from tool import ROOT, make_client, prepare_config, retrieve_examples, tokenize, unique_entities

MODULES = {"extraction": extraction, "discovery": discovery,
           "boundary": boundary, "verification": verification}
VARIANTS = {
    "extract_only": (),
    "full": ("discovery", "boundary", "verification"),
    "without_discovery": ("boundary", "verification"),
    "without_boundary": ("discovery", "verification"),
    "without_verification": ("discovery", "boundary"),
    "without_retrieval": ("discovery", "boundary", "verification"),
}
RESULT_ROOT = ROOT / "result" / "ner_agent"


def run_pipeline(sentence, config, client, variant="full", cache=None):
    """Only inference inputs enter agents. The per-sentence cache pairs ablations."""
    if cache is None:
        cache = {}
    if variant == "without_retrieval":
        config = {**config, "retrieval_k": 0}
    tokens = tokenize(sentence)
    entities, stages = [], []
    for stage in ("extraction", *VARIANTS[variant]):
        previous = copy.deepcopy(entities)
        candidates = [] if stage in ("extraction", "discovery") else entities
        payload = {"sentence": sentence,
                   "tokens": [{"id": t["id"], "text": t["text"]} for t in tokens],
                   "candidates": [{"id": i, **e} for i, e in enumerate(candidates)],
                   "training_examples": retrieve_examples(sentence, candidates, config)}
        key = json.dumps([stage, config["_prompt_hash"], payload], ensure_ascii=False, sort_keys=True)
        cache_hit = key in cache
        if cache_hit:
            output, trace = copy.deepcopy(cache[key])
        elif stage in ("boundary", "verification") and not entities:
            output, trace = [], {"stage": stage, "status": "skipped_empty", "attempts": []}
        else:
            output, trace = MODULES[stage].run(payload, tokens, config, client)
            cache[key] = copy.deepcopy((output, trace))
        if output is not None:
            if stage in ("extraction", "discovery"):
                output = [{**e, "sources": [stage]} for e in output]
            entities = unique_entities(entities + output) if stage == "discovery" else output
        trace.update({"cache_hit": cache_hit, "input_entities": previous,
                      "output_entities": copy.deepcopy(entities)})
        stages.append(trace)
        if trace.get("fatal"):
            break
    return {"pred_entities": entities, "stages": stages,
            "fully_executed": all(s["status"] != "failed" for s in stages)}


def span_set(entities):
    return {(*e["pos"], e["type"]) if "pos" in e else (e["start"], e["end"], e["type"])
            for e in entities or []}


def successful_record(row):
    stages = row.get("stages", [])
    return row.get("fully_executed") is True and bool(stages) and all(
        s["status"] in ("ok", "skipped_empty") for s in stages)


def evaluate_ner(records):
    """Score fully successful samples; failed requests/parsing are neither FP nor FN."""
    tp = fp = fn = evaluated = 0
    effects = {}
    for row in records:
        stages = row.get("stages", [])
        success = successful_record(row)
        if success:
            evaluated += 1
            gold, pred = span_set(row["gold_entities"]), span_set(row["pred_entities"])
            tp += len(gold & pred)
            fp += len(pred - gold)
            fn += len(gold - pred)
        for stage in stages:
            totals = effects.setdefault(stage["stage"], dict.fromkeys(
                ["samples", "failed_samples", "evaluated_samples", "true_added", "false_added",
                 "true_removed", "false_removed"], 0))
            totals["samples"] += 1
            totals["failed_samples"] += int(stage["status"] == "failed")
            if not success:
                continue
            totals["evaluated_samples"] += 1
            before, after = span_set(stage["input_entities"]), span_set(stage["output_entities"])
            added, removed = after - before, before - after
            for key, value in {"true_added": len(added & gold), "false_added": len(added - gold),
                               "true_removed": len(removed & gold), "false_removed": len(removed - gold)}.items():
                totals[key] += value
    empty_score = 0.0 if evaluated else None
    return {"precision": tp / (tp + fp) if tp + fp else empty_score,
            "recall": tp / (tp + fn) if tp + fn else empty_score,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else empty_score,
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "num_samples": len(records),
            "failed_samples": len(records) - evaluated,
            "evaluated_samples": evaluated,
            "success_rate": evaluated / len(records) if records else None,
            "evaluation_scope": "successful_samples_only",
            "stage_effects": effects}


def append_summary(summary, metrics):
    """Keep existing columns and label historical rows without changing their scores."""
    rows, columns = [], []
    if summary.exists() and summary.stat().st_size:
        with summary.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            columns, rows = list(reader.fieldnames), list(reader)
    for row in rows:
        if not row.get("evaluation_scope"):
            row["evaluation_scope"] = "all_samples"
            row["evaluated_samples"] = row["num_samples"]
    current = {k: v for k, v in metrics.items() if k != "stage_effects"}
    columns += [k for k in current if k not in columns]
    with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=summary.parent,
                            prefix=summary.name + ".", suffix=".tmp", delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows([*rows, current])
    Path(f.name).replace(summary)


def save_results(run_dir, config, variant, records, paired_ids=None, reference=None, result_root=None):
    """Finalize completed runs; the runner has already flushed each prediction and log."""
    metrics = evaluate_ner(records)
    if paired_ids is not None:
        paired = evaluate_ner([r for r in records if r["sample_id"] in paired_ids])
        metrics.update({"paired_samples": paired["evaluated_samples"],
                        **{f"paired_{k}": paired[k] for k in ("precision", "recall", "f1")}})
        if reference is not None and reference["f1"]:
            gain = paired["f1"] / reference["f1"] - 1
            metrics.update({"paired_f1_relative_gain": gain,
                            "paired_precision_delta": paired["precision"] - reference["precision"],
                            "paired_recall_delta": paired["recall"] - reference["recall"],
                            "target_20pct_met": gain >= 0.2 - 1e-12
                            and paired["precision"] >= reference["precision"]
                            and paired["recall"] >= reference["recall"]})
    stages = [s for r in records for s in r["stages"]]
    metrics.update({"status": "complete", "run_id": run_dir.name, "dataset": config["dataset"],
                    "system_version": config["system_version"],
                    "model_name": config["model_name"], "variant": variant,
                    "retrieval_k": 0 if variant == "without_retrieval" else config["retrieval_k"],
                    "training_sha256": config["training_sha256"],
                    "prompt_sha256": config["_prompt_hash"],
                    "api_calls": sum(len(s["attempts"]) for s in stages if not s["cache_hit"]),
                    "logical_attempts": sum(len(s["attempts"]) for s in stages),
                    "cache_hits": sum(s["cache_hit"] for s in stages),
                    "duration_seconds": round(sum(r["elapsed_seconds"] for r in records), 3)})
    if config.get("agent_system") == "expert_agent":
        for prefix, selected in (("actual", [s for s in stages if not s["cache_hit"]]), ("logical", stages)):
            usages = [a["usage"] for s in selected for a in s["attempts"] if a.get("usage")]
            metrics[f"{prefix}_total_tokens"] = sum(u.get("total_tokens", 0) or 0 for u in usages)
            metrics[f"{prefix}_usage_reported_attempts"] = len(usages)
        metrics.update(expert_diagnostics(records))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    append_summary((result_root or RESULT_ROOT) / "summary_prf1.csv", metrics)
    return {"result_dir": str(run_dir), **metrics}


def expert_diagnostics(records):
    """Gold-dependent diagnostics live only in the offline evaluator."""
    totals = dict.fromkeys(("unique_expert_true_candidates", "unique_expert_true_retained",
                           "vote_true_rejected", "vote_false_accepted",
                           "aggregation_true_added", "aggregation_false_added",
                           "aggregation_true_removed", "aggregation_false_removed"), 0)
    for row in records:
        if not successful_record(row):
            continue
        gold, final = span_set(row["gold_entities"]), span_set(row["pred_entities"])
        candidates = {c["id"]: (c["start"], c["end"], c["type"]) for c in row.get("candidates", [])}
        for c in row.get("candidates", []):
            if len(c.get("sources", [])) == 1 and candidates[c["id"]] in gold:
                totals["unique_expert_true_candidates"] += 1
                totals["unique_expert_true_retained"] += int(candidates[c["id"]] in final)
        for count in row.get("vote_counts", []):
            truth = candidates[count["id"]] in gold
            totals["vote_true_rejected"] += int(truth and count["status"] == "rejected")
            totals["vote_false_accepted"] += int(not truth and count["status"] == "accepted")
        for stage in row["stages"]:
            if stage["stage"] != "aggregation":
                continue
            before, after = span_set(stage["input_entities"]), span_set(stage["output_entities"])
            for action, changed in (("added", after - before), ("removed", before - after)):
                totals[f"aggregation_true_{action}"] += len(changed & gold)
                totals[f"aggregation_false_{action}"] += len(changed - gold)
    count = totals["unique_expert_true_candidates"]
    totals["unique_expert_true_retention_rate"] = totals["unique_expert_true_retained"] / count if count else None
    return totals


def run_experiment(config, variant="full", limit=None, client=None, backend=None):
    prepare = backend.prepare_config if backend else prepare_config
    pipeline = backend.run_pipeline if backend else run_pipeline
    available = backend.VARIANTS if backend else VARIANTS
    result_root = backend.RESULT_ROOT if backend else RESULT_ROOT
    baseline = "single_expert" if backend else "extract_only"
    config = prepare(config)
    variants = list(available) if variant == "all" else [baseline, "full"] if variant == "compare" else [variant]
    if any(name not in available for name in variants):
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
                     "evaluation_scope": "successful_samples_only",
                     "input_sha256": hashlib.sha256(data_bytes).hexdigest(),
                     "prompt_sha256": config["_prompt_hash"], "prompts": config["_prompts"]})
    files = [ROOT / "main.py", ROOT / "experiment.py", ROOT / "tool.py", *sorted((ROOT / "agents").glob("*.py"))]
    if backend:
        files += sorted((ROOT / "expert_agent").glob("*.py"))
    snapshot["code_sha256"] = hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    progress = None
    if backend:
        from expert_agent.console import Progress
        progress = Progress(config, variants, len(data), result_root, run_id)
        config["_progress"] = progress.stage
    runs = {}
    with ExitStack() as stack:
        api = stack.enter_context(nullcontext(client) if client is not None else make_client(config))
        for name in variants:
            directory = result_root / config["dataset"] / config["model_name"] / name / run_id
            directory.mkdir(parents=True, exist_ok=False)
            (directory / "run_config.json").write_text(json.dumps(
                {**snapshot, "variant": name, "retrieval_k": 0 if name == "without_retrieval" else config["retrieval_k"],
                 "modules": list(available[name]) if backend else ["extraction", *available[name]]}, ensure_ascii=False, indent=2))
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
                if progress:
                    progress.begin(index + 1, name)
                    config["_attempt_logger"] = run["logger"]
                    run["logger"].info("sample=%s begin", index)
                prediction = pipeline(sample["sentence"], config, api, name, cache)
                record = {"sample_id": index, "sentence": sample["sentence"], **prediction,
                          "gold_entities": sample["entities"],
                          "elapsed_seconds": round(time.monotonic() - started, 3)}
                run["output"].write(json.dumps(record, ensure_ascii=False) + "\n")
                run["output"].flush()
                run["records"].append(record)
                for stage in record["stages"]:
                    run["logger"].info("sample=%s stage=%s trace=%s", index, stage["stage"], json.dumps(stage, ensure_ascii=False))
                if progress:
                    progress.finish(record)
                else:
                    print(f"[{index + 1}/{len(data)}] {config['dataset']} {config['model_name']} {name} "
                          f"entities={len(record['pred_entities'])} complete={record['fully_executed']}", flush=True)
                if any(s.get("fatal") for s in record["stages"]):
                    raise RuntimeError(f"API configuration rejected (HTTP 400/401/403/404); inspect {run['directory'] / 'run.log'}")
        paired_ids = None
        if len(runs) > 1:
            paired_ids = set.intersection(*[
                {r["sample_id"] for r in run["records"] if successful_record(r)} for run in runs.values()])
            if not progress:
                print(f"Paired evaluation: {len(paired_ids)}/{len(data)} samples succeeded in every variant", flush=True)
        reference = evaluate_ner([r for r in runs[baseline]["records"] if r["sample_id"] in paired_ids]) \
            if paired_ids is not None and baseline in runs else None
        return [save_results(run["directory"], config, name, run["records"], paired_ids, reference, result_root)
                for name, run in runs.items()]


def predict_sentence(sentence, config, variant="full"):
    config = prepare_config(config)
    variants = list(VARIANTS) if variant == "all" else ["extract_only", "full"] if variant == "compare" else [variant]
    cache = {}
    with make_client(config) as client:
        return {name: run_pipeline(sentence, config, client, name, cache) for name in variants}
