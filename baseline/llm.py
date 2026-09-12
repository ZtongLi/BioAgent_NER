import argparse
import json
import logging
import time
import csv
import shutil
from pathlib import Path
from datetime import datetime
from openai import OpenAI


# 项目路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# 不同数据集对应的实体类型，用于构建 prompt。
ENTITY_TYPES = {
    "bc2gm": "Gene/Protein",
    "bc5cdr": "Chemical, Disease",
    "ncbi_disease": "Disease",
}


def setup_logging(log_path):
    # 初始化日志，把运行过程和最终指标写入当前实验的日志文件。
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
    parser.add_argument(
        "--config",
        required=True,
        help="配置文件路径，例如 config/bc2gm_deepseek.json",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="预测结果 jsonl 保存路径，例如 result/bc2gm_deepseek_predictions.jsonl",
    )
    return parser.parse_args()


def load_config(config_path):
    # 从实验配置文件中读取数据集、输入输出路径、模型名等参数。
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
    # 根据当前数据集的实体类型和输入句子，构建发给 LLM 的 NER 提示词。
    entity_types = ENTITY_TYPES[dataset]
    return f"""You are performing biomedical named entity recognition.

Identify all {entity_types} entities in the following biomedical sentence.

Return only a JSON list. Each item must have this schema:
{{"text": "entity text", "type": "entity type"}}
Do not wrap the JSON in markdown.
If there is no entity, return [].

Sentence:
{sentence}
"""


def call_llm(prompt, config):
    # LLM 调用入口：使用 config 中的 model_name/api_keys/base_url 请求模型。
    client = OpenAI(
        api_key=config["api_keys"],
        base_url=config["base_url"],
        timeout=config.get("timeout", 60),
    )

    response = client.chat.completions.create(
        model=config["model_name"],
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=1,
    )

    return response.choices[0].message.content


def calculate_accuracy(correct, total):
    # 根据预测完全正确的样本数和总样本数，计算样本级准确率。
    return correct / total if total else 0


def save_summary(config, accuracy):
    # 把当前实验的关键信息和最终结果追加保存到 result/summary.csv。
    summary_path = PROJECT_ROOT / "result" / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = summary_path.exists()

    with open(summary_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["experiment_name", "dataset", "model_name", "max_loop", "accuracy"])
        writer.writerow([
            config["experiment_name"],
            config["dataset"],
            config["model_name"],
            config["max_loop"],
            f"{accuracy:.4f}",
        ])


def main():
    args = parse_args()
    config_path = resolve_project_path(args.config)
    output_path = resolve_project_path(args.output)

    # 读取配置并取出主流程需要的参数。
    config = load_config(config_path)

    experiment_name = config["experiment_name"]
    dataset = config["dataset"]
    test_file_path = PROJECT_ROOT / config["test_file_path"]
    save_file_path = output_path
    log_file_path = PROJECT_ROOT / config["log_file_path"]
    max_loop = config["max_loop"]
    sleep_seconds = config["sleep_seconds"]

    setup_logging(log_file_path)
    save_file_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = backup_existing_output(save_file_path)

    logging.info("Config file: %s", config_path)
    logging.info("Output jsonl: %s", save_file_path)
    if backup_path:
        logging.info("Backed up previous output to: %s", backup_path)
    logging.info("Experiment: %s", experiment_name)
    logging.info("Model: %s", config["model_name"])

    print(f"Config file: {config_path}")
    print(f"Output jsonl: {save_file_path}")
    if backup_path:
        print(f"Previous output backup: {backup_path}")

    with open(test_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 统计预测完全正确的样本数。
    correct = 0
    total = 0

    with open(save_file_path, "w", encoding="utf-8") as f:
        for sample in data[:max_loop]:
            # 读取单条样本，并把金标实体整理成集合，方便和预测结果做精确匹配。
            sentence = sample["sentence"]
            gold_entities = {
                (entity["name"].lower(), entity["type"].upper())
                for entity in sample["entities"]
            }

            # 构建 prompt，调用 LLM，并解析模型返回的 JSON 实体列表。
            prompt = build_prompt(sentence, dataset)
            try:
                answer = call_llm(prompt, config)
            except Exception as e:
                logging.warning("LLM 调用失败：%s", e)
                answer = "[]"

            time.sleep(sleep_seconds)

            try:
                pred_entities = json.loads(answer)
            except json.JSONDecodeError:
                logging.warning("LLM 返回的不是合法 JSON：%s", answer)
                pred_entities = []

            pred_entities = {
                (entity["text"].lower(), entity["type"].upper())
                for entity in pred_entities
            }

            # 样本级评估：整条样本的预测实体集合和金标实体集合完全一致，才算正确。
            total += 1
            if gold_entities == pred_entities:
                correct += 1

            # 保存当前样本的句子、金标实体和预测实体，方便后续查看错误案例。
            f.write(json.dumps({
                "sentence": sentence,
                "gold_entities": list(gold_entities),
                "pred_entities": list(pred_entities),
            }, ensure_ascii=False) + "\n")
            f.flush()

    # 计算并写入最终评估结果。
    accuracy = calculate_accuracy(correct, total)
    logging.info("Sample-level Accuracy: %.4f", accuracy)
    save_summary(config, accuracy)

    print(f"Experiment: {experiment_name}")
    print(f"Model: {config['model_name']}")
    print(f"Sample-level Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
