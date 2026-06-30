import chromadb
from app.core.config import client, EMBEDDING_MODEL
from app.tools.chunker import chunk_all_docs

# Chroma的持久化客户端:数据会存在本地文件夹里,重启程序也不会丢
chroma_client = chromadb.PersistentClient(path="data/chroma_db")

# 创建(或获取已存在的)一个集合(collection),
# 集合可以理解成向量数据库里的"一张表",我们的知识库chunk都存这里
collection = chroma_client.get_or_create_collection(name="k8s_knowledge")

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

if __name__ == "__main__":
    build_index()