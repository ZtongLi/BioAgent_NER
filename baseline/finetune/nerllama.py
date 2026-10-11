"""Project NERLlama3.1 adaptation: offline Llama-3.1-8B-Instruct, NF4 QLoRA."""
import argparse
import gc
import hashlib
import json
import logging
import math
import re
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline.finetune.utils import (ENTITY_TYPES, evaluation_record, finish_run,
                                    load_data, project_path, save_json, tracking_session, track)
from reporting import evaluate_ner
from tool import RESPONSE_PARSER_VERSION, unwrap_json_fence

PROTOCOL_VERSION = "nerllama31_exact_quote_occurrence_v1"


def exact_starts(sentence, text):
    return [m.start() for m in re.finditer(f"(?={re.escape(text)})", sentence)]


def instruction(sentence, dataset):
    """Only sentence and allowed labels enter the inference prompt."""
    return [{"role": "system", "content": (
        "Perform biomedical named entity recognition. Return only a JSON list of objects "
        'with fields {"text": "exact quote", "type": "label", "occurrence": 0}. '
        "Allowed labels: " + ", ".join(ENTITY_TYPES[dataset]) + ". "
        "Copy entity text exactly, preserving case, spaces and punctuation. "
        "occurrence is the zero-based index among ALL exact substring matches of text "
        "in the sentence, including matches inside longer words. It is not a character "
        "offset or token index. Use one object per entity occurrence. "
        "If there are no entities, return []. No commentary or Markdown." )},
        {"role": "user", "content": "Sentence:\n" + sentence}]


def target_json(sample):
    """Training targets only; never called on dev/test by the runner."""
    targets = []
    for entity in sorted(sample["entities"], key=lambda e: e["pos"]):
        text = sample["sentence"][entity["pos"][0]:entity["pos"][1]]
        occurrence = exact_starts(sample["sentence"], text).index(entity["pos"][0])
        targets.append({"text": text, "type": entity["type"], "occurrence": occurrence})
    return json.dumps(targets, ensure_ascii=False)


def parse_entities(raw, sentence, dataset):
    """Reject the whole invalid response; no fuzzy matches or gold-based repair."""
    entities = json.loads(unwrap_json_fence(raw))
    if not isinstance(entities, list):
        raise ValueError("Expected a JSON list")
    spans = set()
    for entity in entities:
        if not isinstance(entity, dict):
            raise ValueError("Each entity must be an object")
        text, kind, occurrence = entity.get("text"), entity.get("type"), entity.get("occurrence")
        if not isinstance(text, str) or not text.strip() or kind not in ENTITY_TYPES[dataset]:
            raise ValueError("Expected nonempty exact text and a valid entity type")
        starts = exact_starts(sentence, text)
        if type(occurrence) is not int or not 0 <= occurrence < len(starts):
            raise ValueError(f"Entity {text!r}: occurrence must be an integer in {list(range(len(starts)))}")
        start = starts[occurrence]
        spans.add((start, start + len(text), kind))
    return [{"start": a, "end": b, "type": kind, "text": sentence[a:b]}
            for a, b, kind in sorted(spans)]


def chat_ids(tokenizer, messages, generation_prompt):
    # Render once, then tokenize without adding a second BOS/EOS.
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=generation_prompt)
    return tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]


def generation_ids(tokenizer, messages, config):
    ids = chat_ids(tokenizer, messages, True)
    if len(ids) + config["max_new_tokens"] > config["max_length"]:
        raise ValueError("Full prompt plus generation reserve exceeds max_length; no text was truncated")
    return ids


def training_feature(sample, tokenizer, config):
    messages = instruction(sample["sentence"], config["dataset"])
    prompt = generation_ids(tokenizer, messages, config)
    complete = chat_ids(tokenizer, messages + [{"role": "assistant", "content": target_json(sample)}], False)
    if complete[:len(prompt)] != prompt:
        raise ValueError("Chat template's assistant prefix is inconsistent; cannot mask prompt loss safely")
    answer_length = len(complete) - len(prompt)
    if not 0 < answer_length <= config["max_new_tokens"] or len(complete) > config["max_length"]:
        raise ValueError("Full training target exceeds generation/context budget; no target was truncated")
    return {"input_ids": complete, "attention_mask": [1] * len(complete),
            "labels": [-100] * len(prompt) + complete[len(prompt):]}


def load_qlora(config, tokenizer, adapter_path=None):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("NERLlama3.1 4-bit QLoRA requires a CUDA GPU; no CPU/full-precision fallback")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    base = AutoModelForCausalLM.from_pretrained(
        project_path(config["pretrained_model_path"]), local_files_only=True,
        quantization_config=quantization, torch_dtype=dtype,
        device_map={"": torch.cuda.current_device()})
    if base.config.model_type != "llama" or not getattr(base, "is_loaded_in_4bit", False):
        raise ValueError("Expected a 4-bit Llama base model")
    if config["max_length"] > base.config.max_position_embeddings:
        raise ValueError("max_length exceeds model context capacity")
    base.config.pad_token_id = tokenizer.pad_token_id
    if adapter_path is not None:
        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False, local_files_only=True)
    else:
        base.config.use_cache = False
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                              gradient_checkpointing_kwargs={"use_reentrant": False})
        model = get_peft_model(base, LoraConfig(
            task_type="CAUSAL_LM", r=config["lora_r"], lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"], target_modules="all-linear", bias="none"))
        model.print_trainable_parameters()
    return model, dtype


def generate_answer(model, tokenizer, ids, config):
    import torch
    # Stop on either EOS or Llama's assistant end-of-turn token.
    stops = [tokenizer.eos_token_id]
    if "<|eot_id|>" in tokenizer.get_vocab():
        stops.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
    stops = sorted({i for i in stops if i is not None})
    if not stops:
        raise ValueError("Tokenizer has no end-of-turn token")
    device = model.get_input_embeddings().weight.device
    inputs = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        result = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                max_new_tokens=config["max_new_tokens"], do_sample=False,
                                num_beams=1, use_cache=True, eos_token_id=stops,
                                pad_token_id=tokenizer.pad_token_id)
    generated = result[0, len(ids):].tolist()
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    finished = bool(generated) and generated[-1] in stops
    return raw, finished


def predict(model, tokenizer, data, config, split, trace_path):
    model.eval()
    records = []
    with trace_path.open("a", encoding="utf-8") as trace:
        for index, sample in enumerate(data):
            original = instruction(sample["sentence"], config["dataset"])
            messages = original
            for attempt in range(1, config["max_attempts"] + 1):
                # An overlong input stops the run; never trim the source sentence.
                ids = generation_ids(tokenizer, messages, config)
                raw, finished = generate_answer(model, tokenizer, ids, config)
                error = None
                try:
                    if not finished:
                        raise ValueError("Response truncated before end-of-turn")
                    entities = parse_entities(raw, sample["sentence"], config["dataset"])
                except (ValueError, TypeError) as exc:
                    error = str(exc)
                trace.write(json.dumps({"split": split, "sample_id": index, "attempt": attempt,
                                        "raw_response": raw, "error": error}, ensure_ascii=False) + "\n")
                trace.flush()
                if error is None:
                    record = evaluation_record(index, sample, entities, config["dataset"],
                                               config["model_name"], stage="instruction_ner")
                    record.update(attempts=attempt, response_format=PROTOCOL_VERSION,
                                  response_parser_version=RESPONSE_PARSER_VERSION)
                    records.append(record)
                    break
                logging.warning("%s sample=%d attempt=%d invalid: %s", split, index, attempt, error)
                if attempt == config["max_attempts"]:
                    raise RuntimeError(f"{split} sample {index} remains invalid; no partial PRF will be published")
                messages = original + [{"role": "assistant", "content": raw},
                    {"role": "user", "content": "The response is invalid: " + error +
                     ". Return the complete corrected JSON list. Preserve valid entities. "
                     "Use exact text and valid zero-based occurrence indices."}]
            if (index + 1) % 20 == 0 or index + 1 == len(data):
                logging.info("%s prediction %d/%d", split, index + 1, len(data))
    return records


def train_epoch(model, loader, optimizer, scaler, dtype, config, epoch, tracker=None):
    import torch
    model.train()
    model.config.use_cache = False
    device = model.get_input_embeddings().weight.device
    accumulation = config["gradient_accumulation_steps"]
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for step, batch in enumerate(loader, 1):
        # Normalize the final short accumulation group correctly.
        group_start = ((step - 1) // accumulation) * accumulation
        group_size = min(accumulation, len(loader) - group_start)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss; stopping without test metrics")
        total += loss.item()
        scaler.scale(loss / group_size).backward()
        if step % accumulation == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        if step % 20 == 0 or step == len(loader):
            logging.info("Epoch %d/%d step %d/%d loss=%.6f", epoch, config["epochs"],
                         step, len(loader), total / step)
            track(tracker, {"train/loss": total / step, "train/epoch": epoch,
                            "train/learning_rate": config["learning_rate"]},
                  step=(epoch - 1) * len(loader) + step)
    return total / len(loader)


def validate_config(config):
    required = {"dataset", "model_name", "pretrained_model_path", "train_file_path", "dev_file_path",
                "test_file_path", "checkpoint_dir", "prediction_path", "summary_csv", "seed",
                "max_length", "max_new_tokens", "learning_rate", "epochs", "batch_size",
                "gradient_accumulation_steps", "lora_r", "lora_alpha", "lora_dropout", "max_attempts"}
    if required - config.keys():
        raise ValueError(f"Missing fields: {sorted(required - config.keys())}")
    if config["dataset"] not in ENTITY_TYPES:
        raise ValueError("Unsupported dataset")
    for key in ("max_length", "max_new_tokens", "epochs", "batch_size", "gradient_accumulation_steps",
                "lora_r", "lora_alpha", "max_attempts"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (type(config["seed"]) is not int or not 0 < config["learning_rate"] < 1 or
            not 0 <= config["lora_dropout"] < 1 or config["max_new_tokens"] >= config["max_length"]):
        raise ValueError("Invalid seed, learning rate, dropout or context budget")


def run(config_path, swanlab_mode="disabled"):
    with tracking_session(config_path, swanlab_mode) as tracker:
        return _run(config_path, tracker)


def _run(config_path, tracker):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, DataCollatorForTokenClassification, set_seed

    config_path = project_path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    pretrained = project_path(config["pretrained_model_path"])
    checkpoint = project_path(config["checkpoint_dir"])
    prediction = project_path(config["prediction_path"])
    if not pretrained.is_dir():
        raise FileNotFoundError(f"Local Llama-3.1-8B-Instruct model missing: {pretrained}")
    if checkpoint.exists() or prediction.exists():
        raise FileExistsError("Checkpoint/output exists. Use new paths; prior experiments are not overwritten")
    paths = {s: project_path(config[f"{s}_file_path"]) for s in ("train", "dev", "test")}
    if len(set(paths.values())) != 3 or not all(p.is_file() for p in paths.values()):
        raise ValueError("train/dev/test must be three existing distinct files")
    set_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    tokenizer = AutoTokenizer.from_pretrained(pretrained, local_files_only=True, use_fast=True)
    if not tokenizer.chat_template or tokenizer.eos_token_id is None:
        raise ValueError("Use the Instruct tokenizer with its original chat template and EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train = load_data(paths["train"], config["dataset"])
    dev = load_data(paths["dev"], config["dataset"])
    features = []
    for i, sample in enumerate(train):
        try:
            features.append(training_feature(sample, tokenizer, config))
        except ValueError as exc:
            raise ValueError(f"Training sample {i}: {exc}") from exc
    for i, sample in enumerate(dev):
        try:
            generation_ids(tokenizer, instruction(sample["sentence"], config["dataset"]), config)
        except ValueError as exc:
            raise ValueError(f"Dev sample {i}: {exc}") from exc
    model, dtype = load_qlora(config, tokenizer)
    checkpoint.mkdir(parents=True, exist_ok=False)
    prediction.parent.mkdir(parents=True, exist_ok=True)
    log_path = prediction.parent / "run.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", force=True,
                        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)])
    config["_config_path"] = str(config_path)
    metadata = {"config": config, "status": "training", "started_at": datetime.now().astimezone().isoformat(),
                "protocol_version": PROTOCOL_VERSION, "implementation": "project_adaptation",
                "quantization": "4bit_nf4_double_quant", "compute_dtype": str(dtype),
                "lora_targets": "all-linear", "loss": "assistant_only",
                "decoding": "greedy_with_format_correction", "prompt": instruction("{sentence}", config["dataset"]),
                "versions": {p: version(p) for p in ("torch", "transformers", "peft", "accelerate", "bitsandbytes")},
                "train_sha256": hashlib.sha256(paths["train"].read_bytes()).hexdigest(),
                "dev_sha256": hashlib.sha256(paths["dev"].read_bytes()).hexdigest()}
    save_json(checkpoint / "run_config.json", metadata)
    loader = DataLoader(features, batch_size=config["batch_size"], shuffle=True,
                        collate_fn=DataCollatorForTokenClassification(tokenizer, padding=True),
                        generator=torch.Generator().manual_seed(config["seed"]))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config["learning_rate"], weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    trace_path = prediction.parent / "generation_attempts.jsonl"
    best_f1, best_epoch, history = -1.0, None, []
    logging.info("train=%d dev=%d optimizer_steps/epoch=%d", len(train), len(dev),
                 math.ceil(len(loader) / config["gradient_accumulation_steps"]))
    for epoch in range(1, config["epochs"] + 1):
        loss = train_epoch(model, loader, optimizer, scaler, dtype, config, epoch, tracker)
        records = predict(model, tokenizer, dev, config, f"dev_epoch_{epoch}", trace_path)
        metrics = evaluate_ner(records)
        if metrics["evaluated_samples"] != len(dev):
            raise RuntimeError("Cannot select a checkpoint using incomplete dev predictions")
        history.append({"epoch": epoch, "train_loss": loss, "dev": metrics})
        logging.info("Epoch %d dev P=%.6f R=%.6f F1=%.6f", epoch,
                     metrics["precision"], metrics["recall"], metrics["f1"])
        if metrics["f1"] > best_f1:
            best_f1, best_epoch = metrics["f1"], epoch
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            logging.info("Saved best LoRA adapter: epoch %d", epoch)
        save_json(checkpoint / "training_history.json", history)
        track(tracker, {**{f"dev/{k}": metrics[k] for k in ("precision", "recall", "f1")},
                        "dev/epoch": epoch, "dev/best_f1": best_f1, "dev/best_epoch": best_epoch},
              step=epoch * len(loader))
    metadata.update(status="selected", best_epoch=best_epoch, best_dev_f1=best_f1)
    save_json(checkpoint / "run_config.json", metadata)
    # Free the training base before reloading the selected adapter for final evaluation.
    del optimizer, scaler, model
    gc.collect()
    torch.cuda.empty_cache()
    model, _ = load_qlora(config, tokenizer, adapter_path=checkpoint)
    test = load_data(paths["test"], config["dataset"])
    records = predict(model, tokenizer, test, config, "test", trace_path)
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
