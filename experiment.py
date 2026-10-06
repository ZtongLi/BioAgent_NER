"""Expert experiment runner: progress, variant pairing, traces and result persistence."""

import csv
import fcntl
import hashlib
import json
import logging
import time
from contextlib import ExitStack, nullcontext
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from tool import ROOT, make_client
from expert_agent import pipeline
from expert_agent.console import Progress
from evaluation import evaluate_ner, expert_diagnostics, successful_record, span_set

RESULT_ROOT = pipeline.RESULT_ROOT


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
        from reporting import write_report
        write_report(config, variant, metrics, run_dir / "predictions.jsonl", run_dir / "run.log")
    else:
        append_summary((result_root or RESULT_ROOT) / "summary_prf1.csv", metrics)
    return {"result_dir": str(run_dir), **metrics}



def run_experiment(config, variant="full", limit=None, client=None, resume=False):
    if resume and variant != "full":
        raise ValueError("--resume currently supports --variant full only")
    available = pipeline.VARIANTS
    result_root = ROOT / config["result_root"] if config.get("result_root") else pipeline.RESULT_ROOT
    baseline = "single_expert"
    config = pipeline.prepare_config(config)
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
    config["_input_sha256"] = hashlib.sha256(data_bytes).hexdigest()
    data = json.loads(data_bytes)[:count]
    if not data:
        raise ValueError("Evaluation dataset is empty")
    snapshot = {k: v for k, v in config.items()
                if not k.startswith("_") and not any(word in k.lower() for word in ("key", "secret", "token"))}
    snapshot.update({"requested_limit": count, "actual_samples": len(data),
                     "evaluation_scope": "successful_samples_only",
                     "input_sha256": hashlib.sha256(data_bytes).hexdigest(),
                     "prompt_sha256": config["_prompt_hash"], "prompts": config["_prompts"]})
    files = [ROOT / name for name in ("main.py", "experiment.py", "evaluation.py", "tool.py")]
    files += sorted((ROOT / "expert_agent").glob("*.py"))
    snapshot["code_sha256"] = hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()
    resumed_dir = None
    if resume:
        from expert_agent.checkpoint import find_checkpoint, read_checkpoint
        resumed_dir = find_checkpoint(result_root / config["dataset"] / config["model_name"] / "full", snapshot)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if resumed_dir:
        run_id = resumed_dir.name
    progress = Progress(config, variants, len(data), result_root, run_id)
    config["_progress"] = progress.stage
    runs = {}
    with ExitStack() as stack:
        for name in variants:
            directory = result_root / config["dataset"] / config["model_name"] / name / run_id
            directory.mkdir(parents=True, exist_ok=bool(resumed_dir))
            lock = stack.enter_context((directory / ".run.lock").open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"Experiment already running: {directory}") from None
            if (directory / "shards" / "manifest.json").exists():
                raise RuntimeError("This run is sharded; resume with run_full_shard.py --part prefix|middle|tail")
            records = []
            if resumed_dir:
                records, raw, retained = read_checkpoint(directory, data)
                if raw != retained:
                    backup = directory / f"predictions.jsonl.bak.{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                    backup.write_bytes(raw)
                    (directory / "predictions.jsonl").write_bytes(retained)
                    progress.line(f"末条未完成记录已备份：{backup}")
                progress.line(f"已恢复 {len(records)}/{len(data)} 条；" +
                              (f"从第 {len(records) + 1} 条继续" if len(records) < len(data) else "全部完成，无需请求"))
            else:
                (directory / "run_config.json").write_text(json.dumps(
                    {**snapshot, "variant": name, "retrieval_k": 0 if name == "without_retrieval" else config["retrieval_k"],
                     "modules": list(available[name])}, ensure_ascii=False, indent=2))
            output = stack.enter_context((directory / "predictions.jsonl").open("a" if resumed_dir else "w", encoding="utf-8"))
            handler = logging.FileHandler(directory / "run.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger = logging.getLogger(f"ner_agent.{run_id}.{name}")
            logger.setLevel(logging.INFO)
            logger.propagate = False
            logger.addHandler(handler)
            stack.callback(handler.close)
            stack.callback(logger.removeHandler, handler)
            if resumed_dir:
                logger.info("Resumed saved_samples=%s current_code_sha256=%s", len(records), snapshot["code_sha256"])
            runs[name] = {"directory": directory, "output": output, "logger": logger, "records": records}
        needs_requests = any(len(run["records"]) < len(data) for run in runs.values())
        api = stack.enter_context(nullcontext(client) if client is not None or not needs_requests else make_client(config))
        for index, sample in enumerate(data):
            cache = {}
            for name, run in runs.items():
                if index < len(run["records"]):
                    continue
                started = time.monotonic()
                progress.begin(index + 1, name)
                config["_attempt_logger"] = run["logger"]
                run["logger"].info("sample=%s begin", index)
                prediction = pipeline.run_pipeline(sample["sentence"], config, api, name, cache)
                record = {"sample_id": index, "sentence": sample["sentence"], **prediction,
                          "gold_entities": sample["entities"],
                          "elapsed_seconds": round(time.monotonic() - started, 3)}
                run["output"].write(json.dumps(record, ensure_ascii=False) + "\n")
                run["output"].flush()
                run["records"].append(record)
                for stage in record["stages"]:
                    run["logger"].info("sample=%s stage=%s trace=%s", index, stage["stage"], json.dumps(stage, ensure_ascii=False))
                progress.finish(record)
                if any(s.get("fatal") for s in record["stages"]):
                    raise RuntimeError(f"API configuration rejected (HTTP 400/401/403/404); inspect {run['directory'] / 'run.log'}")
        paired_ids = None
        if len(runs) > 1:
            paired_ids = set.intersection(*[
                {r["sample_id"] for r in run["records"] if successful_record(r)} for run in runs.values()])
        reference = evaluate_ner([r for r in runs[baseline]["records"] if r["sample_id"] in paired_ids]) \
            if paired_ids is not None and baseline in runs else None
        return [save_results(run["directory"], config, name, run["records"], paired_ids, reference, result_root)
                for name, run in runs.items()]


def predict_sentence(sentence, config, variant="full"):
    variants = list(pipeline.VARIANTS) if variant == "all" else ["single_expert", "full"] if variant == "compare" else [variant]
    if any(name not in pipeline.VARIANTS for name in variants):
        raise ValueError("Unknown expert ablation variant")
    config = pipeline.prepare_config(config)
    cache = {}
    with make_client(config) as client:
        return {name: pipeline.run_pipeline(sentence, config, client, name, cache) for name in variants}
