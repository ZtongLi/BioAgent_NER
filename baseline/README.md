# Baseline 运行与评估规则

## 三种 supervised baseline

BioBERT / PubMedBERT 共用 `finetune/bert_ner.py`；NERLlama3.1 使用 `finetune/nerllama.py`。
三者共用 `finetune/utils.py` 的数据校验与 evaluator/reporting 输出适配。
JSON 配置决定模型和数据集，三个模型乘三个数据集共九个实验。
原始参数与最佳 checkpoint 放在 `baseline/model/`，准备方法见 [模型目录说明](model/README.md)。
模型参数仅在服务器准备，本地与 Git 中只保留模型目录 README。

从项目根目录启动九次实验（每行是独立实验；多个 GPU 可分别设置 CUDA_VISIBLE_DEVICES）：

```bash
python -u baseline/finetune/bert_ner.py --config config/baseline/bc2gm/bc2gm_biobert.json
python -u baseline/finetune/bert_ner.py --config config/baseline/bc5cdr/bc5cdr_biobert.json
python -u baseline/finetune/bert_ner.py --config config/baseline/ncbi/ncbi_biobert.json
python -u baseline/finetune/bert_ner.py --config config/baseline/bc2gm/bc2gm_pubmedbert.json
python -u baseline/finetune/bert_ner.py --config config/baseline/bc5cdr/bc5cdr_pubmedbert.json
python -u baseline/finetune/bert_ner.py --config config/baseline/ncbi/ncbi_pubmedbert.json
python -u baseline/finetune/nerllama.py --config config/baseline/bc2gm/bc2gm_nerllama31.json
python -u baseline/finetune/nerllama.py --config config/baseline/bc5cdr/bc5cdr_nerllama31.json
python -u baseline/finetune/nerllama.py --config config/baseline/ncbi/ncbi_nerllama31.json
```

BERT 两种模型共同设置：seed=42、max_length=512、stride=128、learning_rate=2e-5、epochs=5、batch_size=16。
使用 AdamW（weight_decay=0.01）、固定学习率、梯度裁剪 1.0、全精度训练；优先 CUDA，否则 CPU。
训练终端显示 epoch、step、loss；dev/test 推理显示完成进度，dev 每轮输出严格实体 PRF。
每轮用完整 dev 选模，F1 相同时保留较早 epoch。测试集只在选模完成后加载、预测和评分一次。
训练只读取完整 train；dev 不参与梯度更新；所有空实体句子保留。
两种 backbone 的 tokenizer 不同，但训练/解码/评估代码和六份配置的超参数相同。

字符实体通过 fast tokenizer 的 offset_mapping 映射到 BIO，所有 WordPiece 参与训练。
无法精确表达的训练边界会带样本 ID 报错；不会扩大/缩小标注或跳过该样本。
长句滑动窗口覆盖全部 token；训练中每个 token 仅在首次出现的窗口计算损失，重叠副本标 -100。
预测时对同一原句 token 的多窗口 logits 取均值，再对整句进行一次 argmax/BIO 解码。
孤立 I 或类型变化的 I 按新实体起点处理，此规则不依赖 gold。
实体输出用原句切片填写 text，字符范围始终为 [start, end)。

### NERLlama3.1：Llama-3.1-8B-Instruct + QLoRA

本项目按指定 backbone 和训练方式实现 NERLlama3.1 适配 baseline；
不声明与某个外部 NERLlama 论文、模板或公开 checkpoint 完全相同。
采用原始 Llama-3.1-8B-Instruct，在完整 train 上训练各数据集独立的 LoRA adapter。
NF4 4-bit + double quantization；冻结量化 backbone，只训练 all-linear LoRA 参数。
使用 PEFT 的 prepare_model_for_kbit_training 和 gradient checkpointing。
需要支持 bitsandbytes 的 NVIDIA CUDA 环境；没有 CPU 或全精度替代运行路径。
计算 dtype 优先 BF16，不支持则 FP16 + GradScaler，实际 dtype 写入运行记录。

默认 seed=42、epochs=3、learning_rate=2e-4、batch_size=1、梯度累积=16、
LoRA r=16、alpha=32、dropout=0.05；AdamW、固定学习率、weight_decay=0.01、梯度裁剪=1.0。
使用原模型 chat template；仅 assistant 答案及结束 token 计算损失，system/user/padding 为 -100。
不同于 BERT 的 512-token 窗口，Llama 使用完整句子，max_length=4096、max_new_tokens=1536。
训练目标或推理完整输入超过预算会报出样本位置并停止，不截断原句或丢弃后半段标注。

训练答案为 JSON 列表：`[{"text":"原文精确引用","type":"GENE","occurrence":0}]`。
出现序号基于原句所有精确子串匹配（含长词内部的匹配）从 0 计数，支持重复实体文本。
dev/test 提示只包含句子与允许类型；gold 只用于推理完成后的离线计分。
使用 greedy generation，无随机采样；生成到 EOS/EOT 后才视为完整输出。
解析复用项目的外层 JSON 围栏处理，再严格校验整条响应；不能模糊匹配、改大小写或丢弃坏实体。
按 `(start,end,type)` 去重后生成原句字符实体，`text` 必须等于 `sentence[start:end]`。

无效格式/文本/occurrence/截断会携带原句与错误反馈纠正，最多 max_attempts=3 次；
纠错不读取 gold。有效的 `[]` 或识别错误的有效响应直接计分，不因得分低而重试。
仍然无效则中止当前实验，不用部分 dev 选模、不跳过测试样本、不发布部分样本 PRF。
原始响应和错误保存为 `generation_attempts.jsonl`（含 dev epoch、sample_id、attempt）。
每轮必须完整完成 dev 才参与最佳 checkpoint 选择；最终重新加载 base + 最佳 adapter 测试。
“最终测试一次”指只对选定模型进行一轮测试集评估，格式纠错可能使某条样本生成多次。

### 共同结果接口

每条 `predictions.jsonl` 记录包含 sample_id、sentence、gold_entities、pred_entities、
fully_executed，以及一个成功的 token_classification（BERT）或 instruction_ner（Llama）stage。
gold 使用原始 pos/type 字典；prediction 使用 start/end/type/text 字典，
可直接传给共用的 `tool.evaluate_ner`，无需转换三元组。
有效的空预测照常产生 FN；程序错误停止运行，不将错误样本当作空预测或删去后发布分数。

全部测试样本完成后，结果保存在：

```text
result/test_500/baseline/<dataset>/<BioBERT或PubMedBERT或NERLlama3.1>/
├── predictions.jsonl
├── metrics.json
├── run_config.json
└── run.log
```

随后通过 `tool.write_report`，以 `supervised_baseline` 类型写入
`result/test_500/test_500_summary_prf1.csv`。指标是现有严格字符实体级 micro PRF，
不使用 token-level F1，也不使用 seqeval。
supervised baseline 使用训练集标注进行参数更新，应与 LLM 提示推理的训练条件分别说明。

当前最小实现不提供训练断点续训。已有 checkpoint 或预测文件时拒绝覆盖；
中断后如需重新实验，先归档旧文件，或为新实验配置新的 checkpoint_dir 和 prediction_path。
每次训练的依赖版本、数据哈希、最佳 epoch 和完整 dev 曲线均有记录。

### 服务器最小上传清单

上传 `baseline/finetune/` 三个 Python 文件、`baseline/model/README.md`、九份 JSON 配置、
三个数据目录各自的 `train.json`、`dev.json`、`test.sampled_500.json`，以及根目录的
`data.py`、`tool.py`、`reporting.py`、`expert_agent/prompts/roles.json`、`requirements-finetune.txt`。
共用评分和报表只依赖标准库；OpenAI SDK 仅在实际调用 LLM API 时导入。
保留原项目目录层级；BC2GM 的数据目录实际叫 `bioDataset/bc2gm1/`。
真实模型参数到服务器再准备。无需上传 LLM API 密钥或原有结果 CSV。
服务器会创建自己的结果表，下载后按实验记录合并，避免覆盖本地已有总表。

### SwanLab 记录训练进程

两个入口都支持 `--swanlab online|offline|disabled`，默认 disabled。
安装更新后的 requirements-finetune.txt 后，服务器执行一次 `swanlab login`，
在交互提示中输入 SwanLab API key（不放进实验 JSON）。

```bash
python -u baseline/finetune/bert_ner.py --config config/baseline/bc2gm/bc2gm_biobert.json --swanlab online
python -u baseline/finetune/nerllama.py --config config/baseline/bc2gm/bc2gm_nerllama31.json --swanlab online
```

实验集中在 `MyBioAgent-supervised` 项目中，按模型分组，名称包含模型、数据集和 seed。
记录 train/loss、learning_rate、epoch、dev/precision、recall、f1、最佳 dev F1/epoch，
以及最终 test/precision、recall、f1。PRF 数值范围 0–1。
横轴 step 为训练 micro-batch 序号；训练 loss 是该轮截至当前的 batch 平均。
失败或 Ctrl+C 分别将运行标记为 crashed/aborted，不作为完成实验。
指标记录异常不会改变梯度、预测或本地 PRF 结果。
离线模式 `--swanlab offline` 不需要登录，记录位于各实验结果目录的 `swanlog/`。
这是日志记录功能，不提供训练断点恢复。

### 单张 RTX 4090D 的并行安排

多个终端会在同一 GPU 上分别加载模型、激活与优化器状态。
四位 QLoRA 仅缩小 backbone 权重，不代表训练总显存只有 4-bit 权重大小。
建议先试两组 BERT 并发，观察 `nvidia-smi` 的显存、GPU 利用率以及合计训练速度；
NERLlama 三个数据集先依次单独运行。显存充足只是并发的必要条件，并不保证总时间缩短。
所有任务 `CUDA_VISIBLE_DEVICES=0` 仍然共享同一张卡，不会自动隔离资源。
九份配置的 checkpoint/预测目录互不相同，汇总 CSV 通过原有文件锁串行写入。
如果需要比较并发与单任务速度，保持 batch_size、梯度累积与其他训练超参数不变。

实现参考：[PEFT 量化训练](https://huggingface.co/docs/peft/developer_guides/quantization)、
[Transformers chat template](https://huggingface.co/docs/transformers/chat_templating)、
[SwanLab init](https://docs.swanlab.cn/api/py-init.html)。

## LLM baseline

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
