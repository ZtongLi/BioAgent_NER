"""Shared BIO token-classification fine-tuning for BioBERT and PubMedBERT."""
import argparse
import hashlib
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.finetune.utils import (character_bio, decode_bio, encode_sentence,
                                    evaluation_record, label_schema, load_data,
                                    merge_windows, project_path, save_json, finish_run,
                                    tracking_session, track)
from reporting import evaluate_ner


def prepare_samples(data, tokenizer, config, labels, training=False):
    prepared = []
    for index, sample in enumerate(data):
        try:
            offsets, windows = encode_sentence(tokenizer, sample["sentence"],
                                               config["max_length"], config["stride"])
            if training:
                tags = character_bio(offsets, sample["entities"], labels)
                seen = set()
                for window in windows:
                    # Each source token contributes loss once, despite window overlap.
                    target = []
                    for i in window["indices"]:
                        target.append(tags[i] if i >= 0 and i not in seen else -100)
                        if i >= 0:
                            seen.add(i)
                    window["inputs"]["labels"] = target
            prepared.append((offsets, windows))
        except (ValueError, KeyError) as exc:
            raise ValueError(f"Token alignment failed at sample_id={index}: {exc}") from exc
    return prepared


def predict(model, tokenizer, data, prepared, labels, config, device, split):
    import torch
    model.eval()
    records = []
    with torch.inference_mode():
        for index, (sample, (offsets, windows)) in enumerate(zip(data, prepared)):
            scores = []
            for start in range(0, len(windows), config["batch_size"]):
                batch = tokenizer.pad([w["inputs"] for w in windows[start:start + config["batch_size"]]],
                                      padding=True, return_tensors="pt")
                outputs = model(**{k: v.to(device) for k, v in batch.items()}).logits
                scores.extend(outputs.detach().cpu())
            tags = merge_windows(windows, scores, len(offsets))
            entities = decode_bio(sample["sentence"], offsets, tags, labels)
            records.append(evaluation_record(index, sample, entities,
                                             config["dataset"], config["model_name"]))
            if (index + 1) % 100 == 0 or index + 1 == len(data):
                logging.info("%s prediction %d/%d", split, index + 1, len(data))
    return records


def run(config_path, swanlab_mode="disabled"):
    with tracking_session(config_path, swanlab_mode) as tracker:
        return _run(config_path, tracker)


def _run(config_path, tracker):
    import torch
    import transformers
    from torch.utils.data import DataLoader
    from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification

    config_path = project_path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {"dataset", "model_name", "pretrained_model_path", "train_file_path",
                "dev_file_path", "test_file_path", "checkpoint_dir", "prediction_path",
                "summary_csv", "seed", "max_length", "stride", "learning_rate", "epochs", "batch_size"}
    if required - config.keys():
        raise ValueError(f"Missing config fields: {sorted(required - config.keys())}")
    for key in ("max_length", "epochs", "batch_size"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["seed"]) is not int or type(config["stride"]) is not int:
        raise ValueError("seed and stride must be integers")
    if not 0 < config["learning_rate"] < 1:
        raise ValueError("learning_rate must be between 0 and 1")
    labels = label_schema(config["dataset"])
    pretrained = project_path(config["pretrained_model_path"])
    checkpoint = project_path(config["checkpoint_dir"])
    prediction = project_path(config["prediction_path"])
    if not pretrained.is_dir():
        raise FileNotFoundError(f"Local pretrained model missing: {pretrained}. Download on the server first.")
    if checkpoint.exists() or prediction.exists():
        raise FileExistsError("Checkpoint/output already exists. Use new paths for a new run; never overwrite a prior run.")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=True, local_files_only=True)
    tokenizer.padding_side = "right"
    model = AutoModelForTokenClassification.from_pretrained(
        pretrained, local_files_only=True, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id={v: i for i, v in enumerate(labels)})
    if config["max_length"] > model.config.max_position_embeddings:
        raise ValueError("max_length exceeds model position capacity")
    paths = {split: project_path(config[f"{split}_file_path"]) for split in ("train", "dev", "test")}
    if len(set(paths.values())) != 3:
        raise ValueError("train/dev/test must be distinct input files")
    train = load_data(paths["train"], config["dataset"])
    dev = load_data(paths["dev"], config["dataset"])
    # Test data is not loaded or scored until model selection is complete.
    if not paths["test"].is_file():
        raise FileNotFoundError(paths["test"])
    print("Checking training alignment and preparing sliding windows...", flush=True)
    training = prepare_samples(train, tokenizer, config, labels, training=True)
    validation = prepare_samples(dev, tokenizer, config, labels)
    checkpoint.mkdir(parents=True, exist_ok=False)
    prediction.parent.mkdir(parents=True, exist_ok=True)
    log_path = prediction.parent / "run.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)])
    config["_config_path"] = str(config_path)
    metadata = {"config": config, "started_at": datetime.now().astimezone().isoformat(),
                "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
                "device": str(device), "labels": labels,
                "train_sha256": hashlib.sha256(paths["train"].read_bytes()).hexdigest(),
                "dev_sha256": hashlib.sha256(paths["dev"].read_bytes()).hexdigest(),
                "window_merge": "mean_logits_per_global_token",
                "training_overlap": "first_window_owns_loss",
                "bio_decode": "orphan_I_starts_entity", "status": "training"}
    save_json(checkpoint / "run_config.json", metadata)
    features = [w["inputs"] for _, windows in training for w in windows]
    collator = DataCollatorForTokenClassification(tokenizer, padding=True)
    loader = DataLoader(features, batch_size=config["batch_size"], shuffle=True,
                        collate_fn=collator, generator=torch.Generator().manual_seed(config["seed"]))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)
    best_f1, best_epoch, history = -1.0, None, []
    logging.info("Device=%s train=%d dev=%d windows=%d", device, len(train), len(dev), len(features))
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        loss_total = 0.0
        for step, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; experiment stopped without a test score")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_total += loss.item()
            if step % 20 == 0 or step == len(loader):
                logging.info("Epoch %d/%d step %d/%d loss=%.6f", epoch, config["epochs"],
                             step, len(loader), loss_total / step)
                track(tracker, {"train/loss": loss_total / step, "train/epoch": epoch,
                                "train/learning_rate": config["learning_rate"]},
                      step=(epoch - 1) * len(loader) + step)
        metrics = evaluate_ner(predict(model, tokenizer, dev, validation, labels, config, device, "dev"))
        history.append({"epoch": epoch, "train_loss": loss_total / len(loader), "dev": metrics})
        logging.info("Epoch %d dev P=%.6f R=%.6f F1=%.6f", epoch,
                     metrics["precision"], metrics["recall"], metrics["f1"])
        if metrics["f1"] > best_f1:
            best_f1, best_epoch = metrics["f1"], epoch
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            logging.info("Saved best checkpoint: epoch %d", epoch)
        save_json(checkpoint / "training_history.json", history)
        track(tracker, {**{f"dev/{k}": metrics[k] for k in ("precision", "recall", "f1")},
                        "dev/epoch": epoch, "dev/best_f1": best_f1, "dev/best_epoch": best_epoch},
              step=epoch * len(loader))
    metadata.update(status="selected", best_epoch=best_epoch, best_dev_f1=best_f1)
    save_json(checkpoint / "run_config.json", metadata)
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = AutoModelForTokenClassification.from_pretrained(checkpoint, local_files_only=True).to(device)
    test = load_data(paths["test"], config["dataset"])
    test_windows = prepare_samples(test, tokenizer, config, labels)
    records = predict(model, tokenizer, test, test_windows, labels, config, device, "test")
    metrics = finish_run(config, records, len(test), metadata, log_path)
    track(tracker, {f"test/{k}": metrics[k] for k in ("precision", "recall", "f1")},
          step=config["epochs"] * len(loader) + 1)
    logging.info("Test P=%.6f R=%.6f F1=%.6f", metrics["precision"], metrics["recall"], metrics["f1"])
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--swanlab", choices=["disabled", "offline", "online"], default="disabled")
    args = parser.parse_args()
    run(args.config, args.swanlab)
