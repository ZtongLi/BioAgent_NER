import json
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. 读取知识库
# ==========================================

with open(
    "example/st1_normalization.json",
    "r",
    encoding="utf-8"
) as f:
    knowledge = json.load(f)


print("Number of knowledge items:", len(knowledge))


# ==========================================
# 2. 加载 Embedding 模型
# ==========================================

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)


# ==========================================
# 3. 构造 Embedding 文本
# ==========================================

texts = []

for item in knowledge:

    text = (
        f"Name: {item['name']}\n"
        f"Aliases: {', '.join(item['aliases'])}\n"
        f"Description: {item['description']}"
    )

    texts.append(text)


# ==========================================
# 4. 生成 Embedding
# ==========================================

embeddings = model.encode(
    texts,
    normalize_embeddings=True
)


# ==========================================
# 5. 检查结果
# ==========================================

print("Embedding generated successfully!")
print("Number of embeddings:", len(embeddings))
print("Embedding shape:", embeddings.shape)

print("First embedding:")
print(embeddings[0][:10])