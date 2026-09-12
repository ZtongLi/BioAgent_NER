import json
import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. 读取处理后的知识
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

print("Embedding shape:", embeddings.shape)


# ==========================================
# 5. 创建 Chroma 数据库
# ==========================================

client = chromadb.PersistentClient(
    path="./vector_db/chroma"
)


# ==========================================
# 6. 创建 Collection
# ==========================================

collection = client.get_or_create_collection(
    name="biomedical_knowledge"
)


# ==========================================
# 7. 写入 Chroma
# ==========================================

collection.add(
    ids=[
        item["id"]
        for item in knowledge
    ],

    embeddings=embeddings.tolist(),

    documents=texts,

    metadatas=[
        {
            "type": item["type"],
            "name": item["name"],
            "source": item["source"],
            "source_id": item["source_id"]
        }
        for item in knowledge
    ]
)


print("Knowledge successfully added to Chroma!")
print("Total documents:", collection.count())