import argparse
import csv
import json
import logging
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTITY_TYPES = {
    "bc2gm": ["GENE"],
    "bc5cdr": ["CHEMICAL", "DISEASE"],
    "ncbi": ["DISEASE"],
}


def setup_logging(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run biomedical NER evaluation with an explicit LLM config and output file."
    )
    parser.add_argument("--config", required=True, help="配置文件路径，例如 config/baseline/bc2gm/bc2gm_deepseek.json")
    parser.add_argument("--output", help="预测结果 jsonl 保存路径；默认使用配置中的 save_file_path")
    parser.add_argument("--resume", action="store_true", help="校验已有有效结果并从下一条续跑；文件不存在时新建")
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def backup_existing_output(output_path):
    if not output_path.exists() or output_path.stat().st_size == 0:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = output_path.with_name(f"{output_path.name}.bak.{timestamp}")
    shutil.copy2(output_path, backup_path)
    return backup_path


def build_prompt(sentence, dataset):
    entity_types = ENTITY_TYPES[dataset]
    quoted_types = ", ".join(f'"{entity_type}"' for entity_type in entity_types)
    type_instruction = (
        f'The only valid entity type is: {quoted_types}. '
        f'For each entity, the "type" value must be exactly {quoted_types}.'
        if len(entity_types) == 1
        else f'The only valid entity types are: {quoted_types}. '
        f'For each entity, the "type" value must be exactly one of these labels.'
    )
    return f"""You are performing biomedical named entity recognition.

Identify all entities that belong to the valid entity type labels.
{type_instruction}

Return only a JSON list. Each item must have this schema:
{{"text": "exact entity text", "type": "entity type", "occurrence": 0}}
Copy text exactly, including case, spaces and punctuation.
occurrence is the zero-based index among ALL exact substring matches of this text,
counted left to right (including matches inside longer words). Do not count character offsets.
Return a separate item for each occurrence of an entity.
If the same text appears twice, the only valid indices are integer 0 and integer 1,
not 1 and 2, not a string, and not a list.
Do not output combined labels such as "CHEMICAL, DISEASE".
Do not wrap the JSON in markdown.
If there is no entity, return [].

Sentence:
{sentence}
"""


def get_api_key(config):
    if config.get("api_key_env"):
        api_key = os.getenv(config["api_key_env"])
        if not api_key:
            raise ValueError(f"Environment variable {config['api_key_env']} is not set.")
        return api_key
    return config["api_keys"]


def call_llm(prompt, config):
    with OpenAI(
        api_key=get_api_key(config),
        base_url=config["base_url"],
        timeout=config.get("timeout", 60),
        max_retries=0,
    ) as client:
        response = client.chat.completions.create(
            model=config["model_name"],
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
        )

    if response.choices[0].finish_reason == "length":
        raise ValueError("Response was truncated")
    return response.choices[0].message.content


def calculate_metrics(true_positive, predicted_total, gold_total):
    precision = true_positive / predicted_total if predicted_total else 0
    recall = true_positive / gold_total if gold_total else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return precision, recall, f1


def normalize_pred_entities(pred_entities, sentence, dataset=None):
    """Validate the whole response; resolve exact quotes without guessing occurrences."""
    if not isinstance(pred_entities, list):
        raise ValueError("Expected a JSON list")
    allowed = ENTITY_TYPES[dataset] if dataset else {t for ts in ENTITY_TYPES.values() for t in ts}
    normalized = set()
    for entity in pred_entities:
        if not isinstance(entity, dict):
            raise ValueError("Every entity must be an object")
        text, label = entity.get("text"), entity.get("type")
        if not isinstance(text, str) or not text.strip() or label not in allowed:
            raise ValueError("Expected nonempty exact text and an allowed entity type")
        start, end = entity.get("start"), entity.get("end")
        matches = [m.start() for m in re.finditer(f"(?={re.escape(text)})", sentence)]
        if "occurrence" in entity:
            occurrence = entity["occurrence"]
            if type(occurrence) is not int or not 0 <= occurrence < len(matches):
                raise ValueError(f"Entity {text!r}: received occurrence={occurrence!r} "
                                 f"({type(occurrence).__name__}); allowed integer indices="
                                 f"{list(range(len(matches)))}. Use one item per occurrence.")
            start = matches[occurrence]
            end = start + len(text)
        elif not (type(start) is int and type(end) is int
                and 0 <= start < end <= len(sentence) and sentence[start:end] == text):
            matches = [m.start() for m in re.finditer(f"(?={re.escape(text)})", sentence)]
            if len(matches) != 1:
                raise ValueError(f"Entity {text!r}: found {len(matches)} exact matches; "
                                 "copy exact text and supply a valid occurrence index")
            start = matches[0]
            end = start + len(text)
        normalized.add((start, end, label))
    return normalized


def correction_matches(answer, sentence):
    """Provide locations for model-proposed quotes only, without consulting gold labels."""
    try:
        entities = json.loads(answer)
    except (ValueError, TypeError):
        return []
    if not isinstance(entities, list):
        return []
    quotes = dict.fromkeys(e['text'] for e in entities if isinstance(e, dict)
                           and isinstance(e.get('text'), str) and e['text'])
    return [{"text": text, "matches": [
        {"occurrence": i, "start": m.start(), "end": m.start() + len(text),
         "context": sentence[max(0, m.start() - 20):m.start() + len(text) + 20]}
        for i, m in enumerate(re.finditer(f"(?={re.escape(text)})", sentence))]}
        for text in quotes]


def load_resume(path, data, config):
    """Read-only validation: never skip failures, mismatched data, or damaged records."""
    if not path.exists():
        return 0, 0, 0, 0
    raw = path.read_text(encoding="utf-8")
    if raw and not raw.endswith("\n"):
        raise ValueError("Resume file has an incomplete final line; original file left untouched")
    tp = predicted = gold_count = count = 0
    for i, line in enumerate(raw.splitlines()):
        row = json.loads(line)
        if i >= len(data) or row.get("sample_id") != i or row.get("fully_executed") is not True:
            raise ValueError(f"Cannot resume: invalid or noncontiguous sample {i}")
        sample = data[i]
        if row.get("sentence") != sample["sentence"]:
            raise ValueError(f"Cannot resume: sentence mismatch at sample {i}")
        for key in ("dataset", "model_name"):
            if key in row and row[key] != config[key]:
                raise ValueError(f"Cannot resume: {key} mismatch at sample {i}")
        if row.get("evaluation_scope") != "successful_samples_only":
            raise ValueError(f"Cannot resume legacy failure-as-empty metrics at sample {i}")
        gold = {(e["pos"][0], e["pos"][1], e["type"].upper()) for e in sample["entities"]}
        if {tuple(e) for e in row["gold_entities"]} != gold:
            raise ValueError(f"Cannot resume: gold mismatch at sample {i}")
        pred = set()
        for span in row["pred_entities"]:
            if (not isinstance(span, list) or len(span) != 3
                    or type(span[0]) is not int or type(span[1]) is not int
                    or not 0 <= span[0] < span[1] <= len(sample["sentence"])
                    or span[2] not in ENTITY_TYPES[config["dataset"]]):
                raise ValueError(f"Cannot resume: invalid predicted span at sample {i}")
            pred.add(tuple(span))
        tp += len(pred & gold)
        predicted += len(pred)
        gold_count += len(gold)
        count += 1
    return count, tp, predicted, gold_count


def save_summary(config, precision, recall, f1):
    summary_path = PROJECT_ROOT / "result" / "baseline" / config["dataset"] / "summary_prf1.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = summary_path.exists()

    with open(summary_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["experiment_name", "dataset", "model_name", "max_loop", "precision", "recall", "f1"])
        writer.writerow([
            config["experiment_name"], config["dataset"], config["model_name"],
            config["max_loop"], f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}",
        ])


class LLMExperiment:
    """Manage paths, predictions, and metrics for one evaluation run."""

    def __init__(self, config_path, output_path=None, resume=False):
        self.resume = resume
        self.config_path = resolve_project_path(config_path)
        self.config = load_config(self.config_path)
        self.output_path = resolve_project_path(output_path or self.config["save_file_path"])

    def _predict(self, sentence):
        # No failed attempt reaches the metric counters or becomes an empty prediction.
        prompt = build_prompt(sentence, self.config["dataset"])
        attempt = 0
        while True:
            attempt += 1
            answer = None
            try:
                answer = call_llm(prompt, self.config)
                entities = json.loads(answer)
                result = normalize_pred_entities(entities, sentence, self.config["dataset"])
            except (ValueError, TypeError, IndexError) as exc:
                logging.warning("Invalid response, attempt=%s: %s; raw_response=%r", attempt, exc, answer)
                prompt = build_prompt(sentence, self.config["dataset"]) + (
                    "\nYour previous response was invalid: " + str(exc)[:300]
                    + "\nPrevious response (data to correct): " + str(answer)[:16000]
                    + "\nExact matches computed from the input sentence: "
                    + json.dumps(correction_matches(answer, sentence), ensure_ascii=False)
                    + "\nReturn the complete corrected JSON list, retaining all valid entities. "
                    "Select occurrence indices from the table according to the sentence; "
                    "the table does not imply all matches must be entities.")
            except Exception as exc:
                # Includes API errors/timeouts. Do not log credential-bearing error bodies.
                logging.warning("Request failed, attempt=%s: %s", attempt, type(exc).__name__)
            else:
                self.last_attempts = attempt
                time.sleep(self.config.get("sleep_seconds", 0))
                return result
            delay = min(self.config.get("retry_delay", 2) * 2 ** min(attempt - 1, 10), 30)
            print(f"  第 {attempt} 次未通过；{delay}s 后重试，暂不计入 PRF；详见日志", flush=True)
            time.sleep(delay)

    def run(self):
        config = self.config
        # Fail fast on missing local credentials instead of retrying a setup error forever.
        get_api_key(config)
        setup_logging(PROJECT_ROOT / config["log_file_path"])
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path = None

        logging.info("Config file: %s", self.config_path)
        logging.info("Output jsonl: %s", self.output_path)
        if backup_path:
            logging.info("Backed up previous output to: %s", backup_path)
        logging.info("Experiment: %s", config["experiment_name"])
        logging.info("Model: %s", config["model_name"])

        print(f"Config file: {self.config_path}")
        print(f"Output jsonl: {self.output_path}")
        if backup_path:
            print(f"Previous output backup: {backup_path}")

        with open(PROJECT_ROOT / config["test_file_path"], "r", encoding="utf-8") as f:
            data = json.load(f)

        data = data[:config["max_loop"]]
        completed, true_positive, predicted_total, gold_total = (
            load_resume(self.output_path, data, config) if self.resume else (0, 0, 0, 0))
        if not self.resume:
            backup_path = backup_existing_output(self.output_path)
            if backup_path:
                print(f"Previous output backup: {backup_path}")
        print(f"已完成 {completed}/{len(data)} 条；从第 {completed + 1} 条继续" if completed < len(data)
              else f"已完成全部 {completed} 条，无需请求", flush=True)
        with open(self.output_path, "a" if self.resume else "w", encoding="utf-8") as f:
            for sample_id in range(completed, len(data)):
                sample = data[sample_id]
                sentence = sample["sentence"]
                gold_entities = {
                    (entity["pos"][0], entity["pos"][1], entity["type"].upper())
                    for entity in sample["entities"]
                }
                pred_entities = self._predict(sentence)
                true_positive += len(gold_entities & pred_entities)
                predicted_total += len(pred_entities)
                gold_total += len(gold_entities)

                f.write(json.dumps({
                    "sample_id": sample_id,
                    "dataset": config["dataset"],
                    "model_name": config["model_name"],
                    "response_format": "exact_text_occurrence_v1",
                    "fully_executed": True,
                    "attempts": self.last_attempts,
                    "evaluation_scope": "successful_samples_only",
                    "sentence": sentence,
                    "gold_entities": list(gold_entities),
                    "pred_entities": list(pred_entities),
                }, ensure_ascii=False) + "\n")
                f.flush()
                p, r, score = calculate_metrics(true_positive, predicted_total, gold_total)
                print(f"[{sample_id + 1}/{min(len(data), config['max_loop'])}] "
                      f"P={p:.4f} R={r:.4f} F1={score:.4f}", flush=True)

        precision, recall, f1 = calculate_metrics(true_positive, predicted_total, gold_total)
        metrics = (("Precision", precision), ("Recall", recall), ("F1", f1))
        for name, value in metrics:
            logging.info("%s: %.4f", name, value)
        save_summary(config, precision, recall, f1)

        print(f"Experiment: {config['experiment_name']}")
        print(f"Model: {config['model_name']}")
        for name, value in metrics:
            print(f"{name}: {value:.4f}")


def main():
    args = parse_args()
    LLMExperiment(args.config, args.output, resume=args.resume).run()


if __name__ == "__main__":
    main()
