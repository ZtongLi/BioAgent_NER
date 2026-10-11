"""Command-line experiments with checkpoint validation and resumable outputs."""

import argparse
import fcntl
import hashlib
import json
import logging
import time
from contextlib import ExitStack, nullcontext
from datetime import datetime
from pathlib import Path

from data import ROOT, VARIANTS, prepare_config
from tool import make_client, successful_record
from agent import run_pipeline, predict_sentence, print_summary
from reporting import RESULT_ROOT, evaluate_ner, save_results

OPERATIONAL = {"code_sha256", "result_root", "summary_csv", "requested_limit",
               "timeout", "retry_delay", "sleep_seconds", "max_attempts", "test_file_path"}


def find_checkpoint(parent, snapshot):
    for directory in sorted(parent.iterdir(), reverse=True) if parent.exists() else []:
        path = directory / "run_config.json"
        if not path.is_file():
            continue
        old = json.loads(path.read_text())
        if old.get("input_sha256") != snapshot["input_sha256"]:
            continue
        current = {**snapshot, "variant": "full", "modules": ["experts", "candidate_pool", "votes", "tally", "aggregation"]}
        for key in set(old) | set(current):
            if key not in OPERATIONAL and old.get(key) != current.get(key):
                raise ValueError(f"Cannot resume {directory}: configuration mismatch: {key}")
        return directory
    return None


def read_checkpoint(directory, data, start=0):
    path = directory / "predictions.jsonl"
    raw = path.read_bytes() if path.exists() else b""
    records, retained = [], []
    lines = raw.splitlines(keepends=True)
    for i, line in enumerate(lines):
        # A killed write can leave a partial final line; preserve it in a backup.
        if not line.endswith(b"\n") and i == len(lines) - 1:
            break
        row = json.loads(line)
        if i >= len(data) or row.get("sample_id") != i + start:
            raise ValueError(f"Cannot resume: noncontiguous sample {i}")
        if row.get("sentence") != data[i]["sentence"] or row.get("gold_entities") != data[i]["entities"]:
            raise ValueError(f"Cannot resume: input mismatch at sample {i}")
        if not successful_record(row):
            if i != len(lines) - 1:
                raise ValueError(f"Cannot resume: unsuccessful non-final sample {i}")
            break
        for entity in row["pred_entities"]:
            entity_start, entity_end = entity.get("start"), entity.get("end")
            if type(entity_start) is not int or type(entity_end) is not int or not 0 <= entity_start < entity_end <= len(row["sentence"]):
                raise ValueError(f"Cannot resume: invalid predicted span at sample {i}")
        records.append(row)
        retained.append(line)
    return records, raw, b"".join(retained)


def run_experiment(config, variant="full", limit=None, client=None, resume=False):
    if resume and variant != "full":
        raise ValueError("--resume currently supports --variant full only")
    result_root = ROOT / config["result_root"] if config.get("result_root") else RESULT_ROOT
    baseline = "single_expert"
    config = prepare_config(config)
    variants = list(VARIANTS) if variant == "all" else [baseline, "full"] if variant == "compare" else [variant]
    if any(name not in VARIANTS for name in variants):
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
                     "input_sha256": config["_input_sha256"],
                     "prompt_sha256": config["_prompt_hash"], "prompts": config["_prompts"]})
    files = [ROOT / name for name in ("main.py", "agent.py", "data.py", "tool.py", "reporting.py")]
    snapshot["code_sha256"] = hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()
    resumed_dir = None
    if resume:
        resumed_dir = find_checkpoint(result_root / config["dataset"] / config["model_name"] / "full", snapshot)
    run_id = resumed_dir.name if resumed_dir else datetime.now().strftime("%Y%m%d_%H%M%S_%f")
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
                raise RuntimeError("This run is sharded; start a new run without --resume to preserve existing shard records")
            records = []
            if resumed_dir:
                records, raw, retained = read_checkpoint(directory, data)
                if raw != retained:
                    backup = directory / f"predictions.jsonl.bak.{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
                    backup.write_bytes(raw)
                    (directory / "predictions.jsonl").write_bytes(retained)
            else:
                (directory / "run_config.json").write_text(json.dumps(
                    {**snapshot, "variant": name, "retrieval_k": 0 if name == "without_retrieval" else config["retrieval_k"],
                     "modules": list(VARIANTS[name])}, ensure_ascii=False, indent=2))
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
                config["_attempt_logger"] = run["logger"]
                run["logger"].info("sample=%s begin", index)
                prediction = run_pipeline(sample["sentence"], config, api, name, cache)
                record = {"sample_id": index, "sentence": sample["sentence"], **prediction,
                          "gold_entities": sample["entities"],
                          "elapsed_seconds": round(time.monotonic() - started, 3)}
                run["output"].write(json.dumps(record, ensure_ascii=False) + "\n")
                run["output"].flush()
                run["records"].append(record)
                for stage in record["stages"]:
                    run["logger"].info("sample=%s stage=%s trace=%s", index, stage["stage"], json.dumps(stage, ensure_ascii=False))
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


def main():
    parser = argparse.ArgumentParser(description="Run biomedical expert-agent experiments.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", default="full", choices=[
        "full", "without_retrieval", "compare", "all",
        "single_expert", "candidate_union", "vote_only", "without_voting", "generic_ensemble"])
    parser.add_argument("--limit", type=int, help="Override the number of evaluation samples")
    parser.add_argument("--resume", action="store_true", help="full 从最近一次相同输入的实验续跑；无记录则新建")
    parser.add_argument("--sentence", help="Predict one sentence without saving results")
    parser.add_argument("--verbose", action="store_true", help="Print full expert experiment JSON instead of the compact summary")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    config["_config_path"] = args.config
    if config.get("agent_system") != "expert_agent":
        parser.error("main.py requires an expert_agent config; run baseline configs with baseline/llm.py")
    result = (predict_sentence(args.sentence, config, args.variant) if args.sentence is not None
              else run_experiment(config, args.variant, args.limit, resume=args.resume))
    if args.verbose or args.sentence is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_summary(result)


if __name__ == "__main__":
    main()
