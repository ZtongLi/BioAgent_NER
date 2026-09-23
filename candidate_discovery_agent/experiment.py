"""Previous single-agent runner, retained for existing candidate configurations."""

import argparse
import csv
import json
import logging
import time
from pathlib import Path

from .tool import candidate_discovery_agent, evaluate_ner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "result" / "candidate_discovery_agent"


def save_results(config, records):
    """Write each response immediately, then append the completed run's metrics."""
    output_path = (PROJECT_ROOT / config["save_file_path"]).resolve()
    log_path = (PROJECT_ROOT / config["log_file_path"]).resolve()
    if not output_path.is_relative_to(RESULT_ROOT.resolve()) or not log_path.is_relative_to(RESULT_ROOT.resolve()):
        raise ValueError("Candidate Discovery Agent output paths must be inside result/candidate_discovery_agent")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    completed = []
    with output_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            completed.append(row)
            logging.info("Sample %s raw_response=%r", row["sample_id"], row["raw_response"])
            print(f"[{len(completed)}/{config['max_loop']}] {config['experiment_name']}", flush=True)

    metrics = evaluate_ner(completed)
    summary_path = RESULT_ROOT / config["dataset"] / "summary_prf1.csv"
    new_file = not summary_path.exists()
    with summary_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["experiment_name", "dataset", "model_name", "max_loop", "precision", "recall", "f1"])
        writer.writerow([config["experiment_name"], config["dataset"], config["model_name"],
                         len(completed), *(f"{metrics[key]:.4f}" for key in ("precision", "recall", "f1"))])
    return output_path, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Candidate Discovery Agent experiment.")
    parser.add_argument("--config", required=True, help="Experiment config JSON path")
    parser.add_argument("--sentence", help="Run one sentence without writing experiment files")
    return parser.parse_args()


def run_experiment(config):
    data_path = PROJECT_ROOT / config["test_file_path"]
    data = json.loads(data_path.read_text(encoding="utf-8"))[:config["max_loop"]]

    def responses():
        for index, sample in enumerate(data):
            raw = candidate_discovery_agent(sample["sentence"], config)
            try:
                prediction = json.loads(raw) if raw is not None else None
            except json.JSONDecodeError:
                prediction = []
            yield {"sample_id": index, "sentence": sample["sentence"],
                   "gold_entities": sample["entities"], "raw_response": raw,
                   "pred_entities": prediction}
            if index + 1 < len(data):
                time.sleep(config.get("sleep_seconds", 0))

    output_path, metrics = save_results(config, responses())
    print(json.dumps({"output": str(output_path), **metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    args = parse_args()
    path = (PROJECT_ROOT / args.config).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if args.sentence:
        print(candidate_discovery_agent(args.sentence, config))
    else:
        run_experiment(config)
