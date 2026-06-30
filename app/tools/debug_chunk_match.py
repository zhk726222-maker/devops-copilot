from app.core.config import client, EMBEDDING_MODEL
from app.tools.chunker import chunk_all_docs
import numpy as np

def get_embedding(text: str):
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 找出所有包含 restartPolicy 的chunk
chunks = chunk_all_docs()
target_chunks = [c for c in chunks if "restartPolicy" in c["text"]]

print(f"找到 {len(target_chunks)} 个包含 restartPolicy 的chunk\n")

query = "Pod的重启策略是什么"
query_emb = get_embedding(query)

for c in target_chunks:
    chunk_emb = get_embedding(c["text"])
    sim = cosine_sim(query_emb, chunk_emb)
    print(f"chunk id: {c['id']}")
    print(f"相似度: {sim:.4f}")
    print(f"内容片段: {c['text'][:150]}")
    print("---")
    