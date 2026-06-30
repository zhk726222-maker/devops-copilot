import re
from rank_bm25 import BM25Okapi
from app.tools.chunker import chunk_all_docs

def tokenize(text: str) -> list[str]:
    """分词:先拆分驼峰命名(如restartPolicy -> restart Policy),
    再转小写,提取连续字母数字片段作为词"""
    # 在小写字母后紧跟大写字母的位置插入空格,拆开驼峰词
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

# 模块级缓存,避免每次搜索都重新切块、重新建索引
_chunks = None
_bm25 = None

def _build_index():
    global _chunks, _bm25
    if _bm25 is not None:
        return
    _chunks = chunk_all_docs()
    corpus_tokens = [tokenize(c["text"]) for c in _chunks]
    _bm25 = BM25Okapi(corpus_tokens)

def search_bm25(query: str, top_k: int = 3) -> list[dict]:
    """输入问题,返回BM25打分最高的top_k个chunk"""
    _build_index()
    query_tokens = tokenize(query)
    scores = _bm25.get_scores(query_tokens)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in ranked_idx:
        results.append({
            "id": _chunks[idx]["id"],
            "text": _chunks[idx]["text"],
            "source": _chunks[idx]["source"],
            "score": scores[idx],
        })
    return results

if __name__ == "__main__":
    test_query = "What is the restart policy for a Pod"
    results = search_bm25(test_query, top_k=3)
    for i, r in enumerate(results):
        contains_target = "restartPolicy" in r["text"]
        print(f"\n结果{i+1} (chunk id: {r['id']}, 来源: {r['source']}, BM25分数: {r['score']:.4f}, 包含restartPolicy: {contains_target})")
        print(r["text"][:200])