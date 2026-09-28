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
{{"text": "entity text", "type": "entity type", "start": 0, "end": 0}}
The start and end values are zero-based character offsets in the sentence (start inclusive, end exclusive).
The first character of the sentence has index 0.
Return a separate item for each occurrence of an entity.
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
    client = OpenAI(
        api_key=get_api_key(config),
        base_url=config["base_url"],
        timeout=config.get("timeout", 60),
    )

    response = client.chat.completions.create(
        model=config["model_name"],
        messages=[{"role": "user", "content": prompt}],
        temperature=1,
    )

    return response.choices[0].message.content


def calculate_metrics(true_positive, predicted_total, gold_total):
    precision = true_positive / predicted_total if predicted_total else 0
    recall = true_positive / gold_total if gold_total else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return precision, recall, f1


def normalize_pred_entities(pred_entities, sentence):
    """Resolve predictions to mention spans; unmatched text remains a false positive."""
    normalized = set()
    unresolved = []
    if not isinstance(pred_entities, list):
        logging.warning("LLM 返回 JSON 不是 list：%s", pred_entities)
        return normalized

    for index, entity in enumerate(pred_entities):
        if not isinstance(entity, dict):
            logging.warning("跳过非 dict 实体：%s", entity)
            continue

        text = entity.get("text") or entity.get("name")
        entity_type = entity.get("type")
        if not text or not entity_type:
            logging.warning("跳过字段不完整的实体：%s", entity)
            continue

        text = str(text)
        entity_type = str(entity_type).upper()
        start, end = entity.get("start"), entity.get("end")
        if start is not None or end is not None:
            if (isinstance(start, int) and not isinstance(start, bool)
                    and isinstance(end, int) and not isinstance(end, bool)
                    and 0 <= start < end <= len(sentence)
                    and sentence[start:end].casefold() == text.casefold()):
                normalized.add((start, end, entity_type))
            elif (isinstance(start, int) and not isinstance(start, bool)
                    and isinstance(end, int) and not isinstance(end, bool)
                    and 1 <= start < end <= len(sentence) + 1
                    and sentence[start - 1:end - 1].casefold() == text.casefold()):
                logging.info("将 1-based 实体位置转换为 0-based：%s", entity)
                normalized.add((start - 1, end - 1, entity_type))
            else:
                unresolved.append((index, text, entity_type, start))
            continue
        unresolved.append((index, text, entity_type, None))

    # 位置不正确或未给位置时，用实体文本定位；多次出现时选最接近模型位置的一处。
    for index, text, entity_type, start in unresolved:
        candidates = [
            (match.start(), match.end(), entity_type)
            for match in re.finditer(re.escape(text), sentence, flags=re.IGNORECASE)
            if (match.start(), match.end(), entity_type) not in normalized
        ]
        if candidates:
            candidate = min(candidates, key=lambda span: abs(span[0] - start)) if isinstance(start, int) and not isinstance(start, bool) else candidates[0]
            logging.info("按实体文本定位：%s -> %s", text, candidate)
            normalized.add(candidate)
        else:
            # 无法定位的预测仍计入分母，不能当作没有预测。
            logging.warning("实体文本无法在句子中定位：%s", text)
            normalized.add((-index - 1, -index - 1, entity_type))

    return normalized


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

    def __init__(self, config_path, output_path=None):
        self.config_path = resolve_project_path(config_path)
        self.config = load_config(self.config_path)
        self.output_path = resolve_project_path(output_path or self.config["save_file_path"])

    def _predict(self, sentence):
        try:
            answer = call_llm(build_prompt(sentence, self.config["dataset"]), self.config)
        except Exception as e:
            logging.warning("LLM 调用失败：%s", e)
            answer = "[]"

        time.sleep(self.config["sleep_seconds"])
        try:
            entities = json.loads(answer)
        except json.JSONDecodeError:
            logging.warning("LLM 返回的不是合法 JSON：%s", answer)
            entities = []
        return normalize_pred_entities(entities, sentence)

    def run(self):
        config = self.config
        setup_logging(PROJECT_ROOT / config["log_file_path"])
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path = backup_existing_output(self.output_path)

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

        true_positive = predicted_total = gold_total = 0
        with open(self.output_path, "w", encoding="utf-8") as f:
            for sample in data[:config["max_loop"]]:
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
                    "sentence": sentence,
                    "gold_entities": list(gold_entities),
                    "pred_entities": list(pred_entities),
                }, ensure_ascii=False) + "\n")
                f.flush()

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
    LLMExperiment(args.config, args.output).run()


if __name__ == "__main__":
    main()
