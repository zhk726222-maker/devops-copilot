import os
import tiktoken

# 用于精确计算token数量的编码器,cl100k_base是GPT系列常用编码,
# 国产模型没有完全对应的tokenizer时,用它做近似估算也够用
encoding = tiktoken.get_encoding("cl100k_base")

CHUNK_SIZE = 500       # 每个chunk目标token数
CHUNK_OVERLAP = 50     # 相邻chunk之间重叠的token数

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """把长文本按token数量切块,块与块之间保留overlap个token重叠"""
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        # 下一块的起点往回退overlap个token,形成重叠
        start = end - overlap
    return chunks

def chunk_all_docs(raw_dir: str = "data/raw") -> list[dict]:
    """读取raw_dir下所有txt文件,切块后返回一个列表,
    每个元素包含chunk文本和它的来源信息(方便后面溯源)"""
    all_chunks = []
    for filename in os.listdir(raw_dir):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(raw_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = split_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{filename}_{i}",
                "text": chunk,
                "source": filename,
            })
        print(f"{filename}: 切出 {len(chunks)} 个chunk")
    return all_chunks

if __name__ == "__main__":
    chunks = chunk_all_docs()
    print(f"\n总共切出 {len(chunks)} 个chunk")
    print(f"\n示例chunk内容:\n{chunks[0]['text'][:200]}...")