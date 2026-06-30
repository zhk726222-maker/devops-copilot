from app.tools.vectorstore import search as vector_search
from app.tools.bm25_search import search_bm25

def reciprocal_rank_fusion(vector_results: list[dict], bm25_results: list[dict], k: int = 60) -> list[dict]:
    """
    RRF融合算法:对每个chunk,根据它在两路检索结果里的排名计算融合分数,
    分数公式: 1/(k + rank),rank从1开始,k是平滑常数(业界常用60)
    """
    scores = {}
    chunk_data = {}

    for rank, item in enumerate(vector_results, start=1):
        cid = item.get("id") or f"{item['source']}_{item['text'][:20]}"
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
        chunk_data[cid] = item

    for rank, item in enumerate(bm25_results, start=1):
        cid = item["id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
        chunk_data[cid] = item

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"id": cid, "rrf_score": score, **chunk_data[cid]} for cid, score in ranked]

def hybrid_search(query: str, top_k: int = 3, candidate_k: int = 10) -> list[dict]:
    """混合检索:分别从向量检索和BM25检索拿候选结果,RRF融合后取top_k"""
    vector_results = vector_search(query, top_k=candidate_k)
    bm25_results = search_bm25(query, top_k=candidate_k)
    fused = reciprocal_rank_fusion(vector_results, bm25_results)
    return fused[:top_k]

if __name__ == "__main__":
    test_query = "What is the restart policy for a Pod"
    results = hybrid_search(test_query, top_k=3)
    for i, r in enumerate(results):
        print(f"\n结果{i+1} (chunk id: {r['id']}, RRF分数: {r['rrf_score']:.4f})")
        print(r["text"][:200])