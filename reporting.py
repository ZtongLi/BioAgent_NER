"""Shared run report; process locking permits independent terminal runs on macOS/Linux."""

import csv
import fcntl
import hashlib
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIELDS = ["model_name", "variant", "precision", "recall", "f1", "dataset",
          "experiment_name", "completed_at", "evaluated_samples", "num_samples",
          "true_positive", "false_positive", "false_negative", "evaluation_scope",
          "config_path", "test_file_path", "input_sha256", "prediction_path", "log_path"]


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
    print(f"汇总 CSV：{path}", flush=True)
