import chromadb
from app.core.config import client, EMBEDDING_MODEL
from app.tools.chunker import chunk_all_docs

# Chroma的持久化客户端:数据会存在本地文件夹里,重启程序也不会丢
chroma_client = chromadb.PersistentClient(path="data/chroma_db")

# 创建(或获取已存在的)一个集合(collection),
# 集合可以理解成向量数据库里的"一张表",我们的知识库chunk都存这里
collection = chroma_client.get_or_create_collection(
    name="k8s_knowledge",
    metadata={"hnsw:space": "cosine"},
)

def get_embedding(text: str) -> list[float]:
    """调用智谱embedding接口,把一段文字转成向量"""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding

def build_index():
    """把所有chunk转成向量,批量写入Chroma"""
    chunks = chunk_all_docs()
    print(f"准备写入 {len(chunks)} 个chunk到向量库...")

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk["text"])
        collection.add(
            ids=[chunk["id"]],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{"source": chunk["source"]}],
        )
        if (i + 1) % 10 == 0:
            print(f"已写入 {i + 1}/{len(chunks)}")

    print(f"\n入库完成,向量库中共有 {collection.count()} 条记录")

def search(query: str, top_k: int = 3) -> list[dict]:
    """输入问题,返回最相关的top_k个chunk"""
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i],
        })
    return hits

if __name__ == "__main__":
    build_index()

    print("\n--- 测试检索 ---")
    test_query = "Pod的重启策略是什么"
    results = search(test_query, top_k=3)
    for i, hit in enumerate(results):
        print(f"\n结果{i+1} (来源: {hit['source']}, 距离: {hit['distance']:.4f})")
        print(hit["text"][:200])