"""Shared supervised data, alignment and strict entity-level result helpers."""
import hashlib
import json
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from data import TYPES as ENTITY_TYPES
from reporting import evaluate_ner, write_report

ROOT = Path(__file__).resolve().parents[2]


def project_path(value):
    return (ROOT / value).resolve()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def track(tracker, values, step=None):
    """Tracking must not alter training or entity scoring when logging fails."""
    if tracker is not None:
        try:
            tracker.log(values, step=step)
        except Exception as exc:
            logging.warning("SwanLab logging failed: %s", type(exc).__name__)


@contextmanager
def tracking_session(config_path, mode="disabled"):
    if mode == "disabled":
        yield None
        return
    if mode not in ("online", "offline"):
        raise ValueError("SwanLab mode must be disabled, online or offline")
    import swanlab
    config = json.loads(project_path(config_path).read_text(encoding="utf-8"))
    swanlab.init(project="MyBioAgent-supervised", mode=mode,
                 name=f"{config['model_name']}_{config['dataset']}_seed{config['seed']}",
                 group=config["model_name"], tags=[config["dataset"], "supervised"],
                 config=config, log_dir=str(project_path(config["prediction_path"]).parent / "swanlog"))
    state, error = "success", None
    try:
        yield swanlab
    except KeyboardInterrupt:
        state = "aborted"
        raise
    except BaseException as exc:
        state, error = "crashed", type(exc).__name__
        raise
    finally:
        try:
            swanlab.finish(state=state, error=error)
        except Exception as exc:
            logging.warning("SwanLab finish failed: %s", type(exc).__name__)

def label_schema(dataset):
    return ["O"] + [f"{prefix}-{kind}" for kind in ENTITY_TYPES[dataset]
                    for prefix in ("B", "I")]


def load_data(path, dataset):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected a nonempty sample list: {path}")
    for index, row in enumerate(data):
        sentence = row["sentence"]
        previous_end = 0
        for entity in sorted(row["entities"], key=lambda e: e["pos"]):
            a, b = entity["pos"]
            if (type(a) is not int or type(b) is not int or
                    not previous_end <= a < b <= len(sentence) or
                    entity["type"] not in ENTITY_TYPES[dataset] or
                    sentence[a:b] != entity["name"]):
                raise ValueError(f"Invalid/overlapping annotation: {path}, sample {index}: {entity}")
            previous_end = b
    return data


def character_bio(offsets, entities, labels):
    """Strict alignment: never silently round or discard a gold boundary."""
    tags = [0] * len(offsets)
    for entity in entities:
        start, end = entity["pos"]
        indices = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
        if (not indices or offsets[indices[0]][0] != start or
                offsets[indices[-1]][1] != end):
            raise ValueError(f"Gold boundary cannot be represented by tokenizer: {entity}")
        for n, i in enumerate(indices):
            if tags[i]:
                raise ValueError("Overlapping entity labels")
            tags[i] = labels.index(f"{'B' if n == 0 else 'I'}-{entity['type']}")
    return tags


def encode_sentence(tokenizer, sentence, max_length, stride):
    """Return full offsets and windows with exact global token indices."""
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer with offset_mapping is required")
    capacity = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if not 0 <= stride < capacity:
        raise ValueError("stride must be smaller than the window's content capacity")
    full = tokenizer(sentence, add_special_tokens=False, truncation=False,
                     return_offsets_mapping=True)
    offsets = [tuple(x) for x in full["offset_mapping"]]
    if not offsets or any(a >= b for a, b in offsets):
        raise ValueError("Sentence must have nonempty, valid token offsets")
    if len(set(offsets)) != len(offsets):
        raise ValueError("Ambiguous duplicate token offsets")
    lookup = {offset: i for i, offset in enumerate(offsets)}
    encoded = tokenizer(sentence, max_length=max_length, stride=stride, truncation=True,
                        return_overflowing_tokens=True, return_offsets_mapping=True,
                        return_special_tokens_mask=True)
    windows, covered = [], set()
    for w, ids in enumerate(encoded["input_ids"]):
        indices = []
        for token, offset, special in zip(ids, encoded["offset_mapping"][w],
                                          encoded["special_tokens_mask"][w]):
            i = -1 if special else lookup[tuple(offset)]
            if i >= 0:
                if full["input_ids"][i] != token:
                    raise ValueError("Window token differs from full sentence token")
                covered.add(i)
            indices.append(i)
        inputs = {k: encoded[k][w] for k in tokenizer.model_input_names if k in encoded}
        windows.append({"inputs": inputs, "indices": indices})
    if covered != set(range(len(offsets))):
        raise ValueError("Sliding windows did not cover the full sentence")
    return offsets, windows


def merge_windows(windows, logits, token_count):
    """Mean logits per global token; one BIO sequence across all windows."""
    import torch
    if len(windows) != len(logits):
        raise ValueError("Window/logit count mismatch")
    total = torch.zeros(token_count, logits[0].shape[-1])
    counts = torch.zeros(token_count, 1)
    for window, scores in zip(windows, logits):
        for j, i in enumerate(window["indices"]):
            if i >= 0:
                total[i] += scores[j].float().cpu()
                counts[i] += 1
    if (counts == 0).any() or not torch.isfinite(total).all():
        raise ValueError("Missing token predictions or non-finite logits")
    return (total / counts).argmax(dim=-1).tolist()


def decode_bio(sentence, offsets, tags, labels):
    """An orphan/type-changing I starts a new entity; no gold-dependent repair."""
    if len(offsets) != len(tags):
        raise ValueError("Token/tag count mismatch")
    entities, current = [], None
    for (start, end), tag in zip(offsets, tags):
        label = labels[tag]
        if label == "O":
            current = None
            continue
        prefix, kind = label.split("-", 1)
        if prefix == "B" or current is None or current["type"] != kind:
            current = {"start": start, "end": end, "type": kind}
            entities.append(current)
        else:
            current["end"] = end
    for entity in entities:
        entity["text"] = sentence[entity["start"]:entity["end"]]
    return entities


def evaluation_record(index, sample, prediction, dataset, model_name, stage="token_classification"):
    return {"sample_id": index, "dataset": dataset, "model_name": model_name,
            "sentence": sample["sentence"], "gold_entities": sample["entities"],
            "pred_entities": prediction, "fully_executed": True,
            "evaluation_scope": "successful_samples_only",
            "stages": [{"stage": stage, "status": "ok",
                        "input_entities": [], "output_entities": prediction}]}


def finish_run(config, records, expected_samples, metadata, log_path):
    """Publish only complete runs through the existing evaluator and reporter."""
    if (len(records) != expected_samples or
            [r["sample_id"] for r in records] != list(range(expected_samples))):
        raise RuntimeError("Missing or duplicate test samples; refusing to publish metrics")
    metrics = evaluate_ner(records)
    if metrics["evaluated_samples"] != expected_samples:
        raise RuntimeError("Test run is incomplete; refusing to publish metrics")
    prediction = project_path(config["prediction_path"])
    if prediction.exists():
        raise FileExistsError(f"Refusing to overwrite predictions: {prediction}")
    temporary = prediction.with_suffix(prediction.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(prediction)
    save_json(prediction.parent / "metrics.json", metrics)
    config["_input_sha256"] = hashlib.sha256(project_path(config["test_file_path"]).read_bytes()).hexdigest()
    metadata.update(status="complete", completed_at=datetime.now().astimezone().isoformat(),
                    test_sha256=config["_input_sha256"])
    save_json(prediction.parent / "run_config.json", metadata)
    save_json(project_path(config["checkpoint_dir"]) / "run_config.json", metadata)
    write_report(config, "supervised_baseline", metrics, prediction, log_path)
    return metrics
