import numpy as np
from app.core.config import client, EMBEDDING_MODEL

def get_embedding(text: str):
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

text1 = "Kubernetes Pod的重启策略包括Always、OnFailure和Never三种"
text2 = "今天北京的天气晴朗,适合出门散步"

emb1 = get_embedding(text1)
emb2 = get_embedding(text2)

print(f"向量维度: {len(emb1)}")
print(f"emb1前5个值: {emb1[:5]}")
print(f"emb2前5个值: {emb2[:5]}")
print(f"两段完全不相关文本的余弦相似度: {cosine_sim(emb1, emb2):.4f}")