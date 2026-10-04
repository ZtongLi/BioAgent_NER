"""Compact terminal reporting; full traces remain in the result files."""

import sys
import time


class Progress:
    def __init__(self, config, variants, total, result_root, run_id):
        self.stream = sys.stderr
        self.live = self.stream.isatty()
        self.started = time.monotonic()
        self.total = total
        self.index = 0
        self.variant = ""
        self.line(f"配置：{config.get('_config_path', '内存配置')}")
        self.line(f"模型：{config['model_name']} | 数据集：{config['dataset']} | 样本：{total}")
        self.line(f"方案：{' / '.join(variants)} | 检索：{config['retrieval_k']} 条"
                  + ("（without_retrieval 为 0）" if 'without_retrieval' in variants else "")
                  + f" | 温度：{config.get('temperature', 1)} | "
                  + ("重试：直到有效返回（无上限）" if config.get('max_attempts', 3) is None
                     else f"最多尝试：{config.get('max_attempts', 3)} 次"))
        self.line(f"结果：{result_root / config['dataset'] / config['model_name']}/<方案>/{run_id}/")

    def line(self, message):
        if self.live:
            self.stream.write("\r\033[2K")
        print(message, file=self.stream, flush=True)

    def begin(self, index, variant):
        self.index, self.variant = index, variant
        self.stage("准备输入", "")

    def stage(self, name, status):
        if not self.live:
            return
        elapsed = time.monotonic() - self.started
        self.stream.write(f"\r\033[2K[{self.index}/{self.total}] {self.variant} | {name} {status} | 累计 {elapsed:.0f}s")
        self.stream.flush()

    def finish(self, record):
        state = "完成" if record['fully_executed'] else "失败"
        self.line(f"[{self.index}/{self.total}] {self.variant:<18} {state} | "
                  f"实体 {len(record['pred_entities'])} | {record['elapsed_seconds']:.1f}s")


def print_summary(results):
    print("\n运行结束")
    print(f"{'方案':<20} {'P':>7} {'R':>7} {'F1':>7}  成功/总数  请求(实际/逻辑)")
    for row in results:
        values = ['—' if row[k] is None else f"{row[k]:.3f}" for k in ('precision', 'recall', 'f1')]
        print(f"{row['variant']:<22} {values[0]:>7} {values[1]:>7} {values[2]:>7}  "
              f"{row['evaluated_samples']}/{row['num_samples']}        {row['api_calls']}/{row['logical_attempts']}")
    if results and 'paired_samples' in results[0]:
        print(f"共同成功样本：{results[0]['paired_samples']}/{results[0]['num_samples']}")
        for row in results:
            value = row.get('paired_f1')
            print(f"  {row['variant']} 配对 F1：{'—' if value is None else f'{value:.3f}'}")
    print("完整指标及逐句记录已保存到上方结果目录。")
