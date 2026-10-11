"""Entity-level evaluation, run metrics and backward-compatible CSV reports."""

import csv
import fcntl
import hashlib
import json
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from data import ROOT
from tool import precision_recall_f1, span_set, successful_record

RESULT_ROOT = ROOT / "result" / "expert_agent"
FIELDS = ["model_name", "variant", "precision", "recall", "f1", "dataset",
          "experiment_name", "completed_at", "evaluated_samples", "num_samples",
          "true_positive", "false_positive", "false_negative", "evaluation_scope",
          "config_path", "test_file_path", "input_sha256", "prediction_path", "log_path"]


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
    precision, recall, f1 = precision_recall_f1(tp, fp, fn, empty_score=empty_score)
    return {"precision": precision, "recall": recall, "f1": f1,
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "num_samples": len(records),
            "failed_samples": len(records) - evaluated,
            "evaluated_samples": evaluated,
            "success_rate": evaluated / len(records) if records else None,
            "evaluation_scope": "successful_samples_only",
            "stage_effects": effects}


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


def write_report(config, variant, metrics, prediction_path, log_path):
    """Write only finished run metrics. Re-exporting the same prediction replaces its row."""
    path = ROOT / config["summary_csv"]
    prediction_path = Path(prediction_path).resolve()
    row = {**metrics, "dataset": config["dataset"], "model_name": config["model_name"],
           "variant": variant,
           "experiment_name": f'{config["dataset"]}_{config["model_name"]}_{variant}_{Path(config["test_file_path"]).stem}',
           "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
           "config_path": config.get("_config_path", ""),
           "test_file_path": config["test_file_path"],
           "input_sha256": config.get("_input_sha256") or hashlib.sha256(
               (ROOT / config["test_file_path"]).read_bytes()).hexdigest(),
           "prediction_path": str(prediction_path), "log_path": str(Path(log_path).resolve())}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        reader = csv.DictReader(f)
        if reader.fieldnames and reader.fieldnames != FIELDS:
            raise ValueError(f"Unexpected report columns: {path}")
        rows = [r for r in reader if r["prediction_path"] != str(prediction_path)]
        f.seek(0)
        f.truncate()
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        order = {v: i for i, v in enumerate(("llm_baseline", "full", "single_expert",
                 "without_retrieval", "without_voting", "candidate_union", "vote_only", "generic_ensemble"))}
        writer.writerows(sorted([*rows, row], key=lambda r: (
            r["model_name"], r["dataset"], order.get(r["variant"], 99),
            r["variant"], r.get("completed_at", ""))))
        f.flush()


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
    if config.get("summary_csv"):
        write_report(config, variant, metrics, run_dir / "predictions.jsonl", run_dir / "run.log")
    else:
        append_summary((result_root or RESULT_ROOT) / "summary_prf1.csv", metrics)
    return {"result_dir": str(run_dir), **metrics}
