# MyBioAgent 项目架构

项目面向生物医学命名实体识别，包含 Expert Agent 和 LLM Baseline 两套实验流程，支持 BC2GM、BC5CDR、NCBI 三个数据集。

## 目录结构

```text
MyBioAgent/
├── main.py                          # Expert Agent 命令行入口
├── run_expert_experiments.sh         # Expert Agent 批量实验入口
├── experiment.py                    # 专家流程复用的运行器、实体级评估与结果保存
├── tool.py                          # 模型请求、重试、实体定位、校验与 BM25 检索
│
├── expert_agent/                    # 多专家方案
│   ├── experiment.py                # 将专家流程接入根目录运行器
│   ├── pipeline.py                  # 流程编排、配置准备与消融变体
│   ├── experts.py                   # 专家角色、独立抽取与投票调用
│   ├── voting.py                    # 票数统计、候选冲突与决策约束
│   ├── aggregation.py               # 汇总决策与输出校验
│   ├── console.py                   # 终端进度与实验摘要
│   ├── prompts/                     # 抽取、投票、汇总提示词
│   └── DESIGN.md                    # 方案设计说明
│
├── baseline/                        # 单次 LLM 抽取基线
│   ├── llm.py                       # 独立入口、提示构建、校验重试、续跑与评估
│   └── README.md                    # Baseline 说明
│
├── agents/                          # 根目录运行器仍导入的兼容模块
├── prompts/                         # 共享提示与配置准备所需模板
│   ├── shared.txt                   # 共享任务约定
│   └── annotation_rules/            # 数据集标注规则与训练数据来源
│
├── config/
│   ├── expert_agent/                # 专家配置：数据集 / 模型配置文件
│   ├── baseline/                    # 基线配置：数据集 / 模型配置文件
│   ├── bc2gm/                       # 历史 Baseline 配置
│   └── baseline_expert_comparison.csv
│
├── bioDataset/
│   ├── bc2gm1/                      # BC2GM 数据
│   ├── bc5cdr/                      # BC5CDR 数据
│   ├── ncbi/                        # NCBI 数据
│   └── sample_dev_500.py             # 开发集固定子集采样
│
├── result/
│   ├── expert_agent/                # 专家结果：数据集 / 模型 / 变体 / 运行时间
│   └── baseline/                    # 基线结果：数据集 / 模型
│
├── tests/                           # 专家流程、评估及 Baseline 续跑等离线测试
└── docs/archive/                    # 本地历史分析文档，已加入 Git 忽略规则
```

仅展示主要结构，省略缓存、凭据、部分辅助文件和历史结果文件。

## Expert Agent 流程

```text
配置 + 输入句子
      ↓
任务规则 + 训练集 BM25 示例检索
      ↓
分子生物学 / 临床医学 / 药理学专家独立抽取
      ↓
实体定位与候选并集
      ↓
专家独立投票 → 程序计票与冲突约束
      ↓
汇总 Agent 决策与校验
      ↓
实体级评估 → 预测、日志、配置快照与指标保存
```

`main.py` 调用 `expert_agent/experiment.py`，后者复用根目录 `experiment.py` 的运行与评估逻辑，并由 `expert_agent/pipeline.py` 提供专家流程。

`pipeline.py` 定义完整流程及消融变体：`full`、`single_expert`、`candidate_union`、`vote_only`、`without_voting`、`generic_ensemble`、`without_retrieval`。

## LLM Baseline 流程

```text
配置 + 输入句子
      ↓
单次 LLM 实体抽取
      ↓
响应校验与实体定位；无效时重试
      ↓
实体级评估 → 逐条预测、日志与指标保存
```

`baseline/llm.py` 是独立入口，包含自己的运行器和评估实现，支持读取已有有效预测进行续跑，不经过 Expert Agent 的投票与汇总模块。

## 数据与共享依赖

三个数据集目录保存训练集、开发集、测试集及固定开发集子集。专家检索读取训练集，金标准实体用于离线评估。

`tool.py` 提供专家方案的共享能力；检索在进程内构建 BM25 索引。`agents/` 是根目录运行器的兼容导入依赖，`prompts/` 中的阶段模板仍被配置准备过程读取；实际专家提示位于 `expert_agent/prompts/`。

两套方案最终都按句内去重后的 `(start, end, type)` 实体集合统计 TP、FP、FN，并计算 micro Precision、Recall、F1。当前实现仅对有效结果计分。
