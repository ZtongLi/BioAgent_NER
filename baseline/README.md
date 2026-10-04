# LLM baseline 运行与评估规则

在项目根目录另开终端，依次运行三个 DeepSeek 配置：

```bash
for dataset in bc2gm bc5cdr ncbi; do
  python3 -u baseline/llm.py --config "config/baseline/$dataset/${dataset}_deepseek.json" || break
done
```

baseline 结果与 expert agent 目录独立，不修改 expert agent 的进度。
并发调用同一 API 服务可能共享限流和额度，增加等待时间。
已有 baseline 预测文件在重跑前自动备份；使用 `--resume` 可校验已有有效样本并追加续跑；文件不存在时从头开始。
续跑会恢复已有 TP、预测总数和金标准总数，避免重复请求或重复计分。
不兼容的数据、旧版失败当空预测的记录或损坏末行会报错，原文件不变。

## PRF 规则

与 expert agent 一致，按实体 `(start, end, type)` 去重后严格匹配，
在有效样本上汇总 TP、FP、FN，再计算 micro PRF：

- P = TP / (TP + FP)
- R = TP / (TP + FN)
- F1 = 2TP / (2TP + FP + FN)

有有效样本时，分母为 0 的指标记 0。
API 错误、超时、截断、非法 JSON、非列表、实体字段或标签无效、
无法在原句精确定位的文本，以及无法确定出现位置的重复文本，均为无效响应。
整条响应中任何实体无效，就重试当前样本；不得跳过该实体后计算指标。
在返回有效结果前，该样本的预测和金标准都不计入 PRF，也不写入成功预测记录。
持续重试直到有效，可用 Ctrl+C 中断；重试等待逐步增加，最长 30 秒。
有效的空列表 `[]` 是正常预测，照常计算漏检，不能因与金标准不符而重试。
唯一的精确文本可用于修正偏移；新版提示要求提供 `occurrence`（精确子串匹配的从 0 开始出现序号），由程序计算偏移；
兼容旧版有效偏移。重复文本必须提供有效出现序号或偏移以确定具体位置。
校验仅依赖原句和标签，不读取金标准；有效但识别错误的结果仍计 FP/FN。

## 从当前结果继续

```bash
for dataset in bc2gm bc5cdr ncbi; do
  python3 -u baseline/llm.py --config "config/baseline/$dataset/${dataset}_deepseek.json" --resume || break
done
```

此次修复前已完成样本使用旧字符偏移提示，续跑样本使用出现序号提示。
PRF 计算规则相同，但整次实验包含两种输出提示版本；新记录以
`response_format=exact_text_occurrence_v1` 标记，正式比较时应披露这一差异。
