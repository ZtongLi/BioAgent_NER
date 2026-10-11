# 服务器模型目录

Git 和本地工程仅保留此 README。此目录下其余所有内容已被 `.gitignore` 忽略。
训练程序只加载本地路径，设置 `local_files_only=True`，不会自动下载模型。

```text
baseline/model/
├── README.md
├── pretrained/
│   ├── biobert/
│   ├── pubmedbert/
│   └── llama31-8b/
└── checkpoints/
    ├── biobert/{bc2gm,bc5cdr,ncbi}/
    ├── pubmedbert/{bc2gm,bc5cdr,ncbi}/
    └── nerllama31/{bc2gm,bc5cdr,ncbi}/
```

`pretrained` 是原始预训练参数及 tokenizer；`checkpoints` 是用完整 train 训练、
按完整 dev 的严格实体级 F1 选择的最佳模型。每个 checkpoint 目录同时记录
训练配置、依赖版本、数据哈希、每轮 dev 指标和最佳 epoch。无需提前创建 checkpoint 叶子目录。
BERT checkpoint 是完整微调参数；NERLlama checkpoint 是最佳 LoRA adapter 及 tokenizer，
需要配合 `pretrained/llama31-8b` 的原始 base model 使用，并非独立的完整 8B 模型。

仅在服务器上，从项目根目录安装依赖：

```bash
python3 -m venv .venv-finetune
source .venv-finetune/bin/activate
python -m pip install -r requirements-finetune.txt
```

根据服务器驱动选择合适的 PyTorch CUDA 安装方式，确认 `torch.cuda.is_available()`。
以下下载命令仅供服务器执行，本次工程实现没有执行下载：

```bash
export HF_ENDPOINT=https://hf-mirror.com

python - <<'PY'
import os
from huggingface_hub import snapshot_download
for repo, destination in [
    ('dmis-lab/biobert-base-cased-v1.2', 'baseline/model/pretrained/biobert'),
    ('microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
     'baseline/model/pretrained/pubmedbert'),
    ('meta-llama/Llama-3.1-8B-Instruct', 'baseline/model/pretrained/llama31-8b'),
]:
    snapshot_download(repo_id=repo, local_dir=destination,
                      token=os.environ.get('HF_TOKEN') if repo.startswith('meta-llama/') else False,
                      allow_patterns=['*.json', '*.txt', '*.jinja', '*.model', '*.safetensors', 'pytorch_model.bin'])
PY
```

`huggingface_hub` 是 transformers 的依赖，无需另加训练框架。
应下载原始预训练 checkpoint，不使用已经在 NER 数据上微调过的第三方 checkpoint。
正式复现时记录下载仓库的 revision，并固定相同模型文件和依赖环境。
国内镜像通过 `HF_ENDPOINT` 配置，需要在启动 Python 前设置；无需更改训练 JSON 中的本地模型路径。
下载 Llama 前需在 Hugging Face 官网取得该模型访问权限，并在服务器设置对应的只读 Token：

```bash
export HF_TOKEN='你的只读HF令牌'
```

镜像不提供 Hugging Face 账号登录/许可申请；授权仍在官网办理。
模型全部下载成功后再启动训练，训练仍只读取本地参数。
镜像使用方法见 [HF-Mirror](https://hf-mirror.com/)。
保留 tokenizer 的 chat template、特殊 token、模型配置与所有权重分片/索引。
使用 Transformers 格式原始 Instruct 权重，不使用 GGUF、第三方 NER adapter 或已经量化的替代模型。
程序加载时用 bitsandbytes NF4 量化，不会修改下载的原始模型。

模型来源：
- [BioBERT v1.2](https://huggingface.co/dmis-lab/biobert-base-cased-v1.2)
- [PubMedBERT / BiomedBERT abstracts + full text](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)
- [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

微软已将 PubMedBERT 重命名为 BiomedBERT；本项目实验名称和本地目录仍使用 PubMedBERT。
预训练模型本身采用各自的 tokenizer（BioBERT cased、PubMedBERT uncased），
原始输入句子与评分字符偏移保持不变。
