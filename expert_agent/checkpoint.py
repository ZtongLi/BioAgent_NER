"""Validate sentence-level full checkpoints before appending predictions."""

import json
from pathlib import Path

from evaluation import successful_record

# Operational settings may change between sessions; inference settings must match.
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
    records, retained = [], b""
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
        retained += line
    return records, raw, retained
