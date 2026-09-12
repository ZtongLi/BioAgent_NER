import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. 加载 Embedding 模型
# ==========================================

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)


# ==========================================
# 2. 连接已有的 Chroma 数据库
# ==========================================

client = chromadb.PersistentClient(
    path="./vector_db/chroma"
)


# ==========================================
# 3. 获取已有的知识集合
# ==========================================

collection = client.get_collection(
    name="biomedical_knowledge"
)


# ==========================================
# 4. 准备用户 Query
# ==========================================

query = "Which gene has the aliases A1B and ABG?"


# ==========================================
# 5. 将 Query 转换成 Embedding
# ==========================================

query_embedding = model.encode(
    query,
    normalize_embeddings=True
)


# ==========================================
# 6. 在 Chroma 中进行相似度检索
# ==========================================

results = collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=1
)


# ==========================================
# 7. 打印检索结果
# ==========================================

print("\n===== Retrieval Result =====")

print("Query:")
print(query)

print("\nRetrieved document:")
print(results["documents"][0][0])

print("\nMetadata:")
print(results["metadatas"][0][0])

print("\nID:")
print(results["ids"][0][0])