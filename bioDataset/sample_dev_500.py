# 用来随机筛选dev的500条样本
"""
Randomly sample 500 records from a JSON dev file.

Run:
    python3 sample_dev_500.py

Then enter the source dev file path when prompted. By default, the sampled
file is written next to the source file as dev.sampled_500.json.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


SAMPLE_SIZE = 500


def read_json_array(path: Path) -> list:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON array.")

    return data


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}.sampled_{SAMPLE_SIZE}{input_path.suffix}")


def ask_path(prompt: str) -> str:
    return input(prompt).strip().strip('"').strip("'")


def main() -> None:
    source_text = ask_path("请输入要删减的 dev 文件路径: ")
    if not source_text:
        raise SystemExit("未输入文件路径，已退出。")

    source_path = Path(source_text).expanduser().resolve()
    if not source_path.exists():
        raise SystemExit(f"文件不存在: {source_path}")
    if not source_path.is_file():
        raise SystemExit(f"路径不是文件: {source_path}")

    data = read_json_array(source_path)
    if len(data) < SAMPLE_SIZE:
        raise SystemExit(
            f"数据量只有 {len(data)} 条，少于 {SAMPLE_SIZE} 条，无法抽样。"
        )

    output_text = ask_path(
        f"请输入输出文件路径，直接回车则保存为 {default_output_path(source_path)}: "
    )
    output_path = (
        Path(output_text).expanduser().resolve()
        if output_text
        else default_output_path(source_path)
    )

    sampled = random.sample(data, SAMPLE_SIZE)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"已从 {source_path} 随机筛选 {SAMPLE_SIZE} 条数据。")
    print(f"新 dev 文件已保存到: {output_path}")


if __name__ == "__main__":
    main()
