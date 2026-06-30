from app.tools.vectorstore import search as vector_search
from app.tools.bm25_search import search_bm25
from app.tools.hybrid_search import hybrid_search
from app.tools.reranker import rerank

# 测试集:每条包含查询问题、目标关键词(用来判断检索结果是否命中相关内容)
TEST_SET = [
    {"query": "If a container in a Pod crashes, will it come back automatically", "keyword": "restartPolicy"},
    {"query": "How can other Pods inside the cluster reach a service without exposing it externally", "keyword": "ClusterIP"},
    {"query": "What happens to the underlying storage after I delete a PersistentVolumeClaim", "keyword": "Retain"},
    {"query": "Which component decides which node a new Pod should run on", "keyword": "kube-scheduler"},
    {"query": "How long has this container orchestration project existed since its origin at Google", "keyword": "Google open sourced"},
    {"query": "How does traffic get distributed across multiple backend Pods", "keyword": "load balanc"},
    {"query": "What do you call a helper container that runs alongside the main application container", "keyword": "sidecar"},
    {"query": "What access mode only allows a single node to mount the volume for writing", "keyword": "ReadWriteOnce"},
]


def reciprocal_rank(results: list[dict], keyword: str) -> float:
    """找到结果列表里第一个包含关键词的位置,返回1/排名;如果完全没命中返回0"""
    for rank, r in enumerate(results, start=1):
        if keyword.lower() in r["text"].lower():
            return 1 / rank
    return 0.0

def evaluate_vector(top_k: int = 5) -> float:
    scores = []
    for item in TEST_SET:
        results = vector_search(item["query"], top_k=top_k)
        scores.append(reciprocal_rank(results, item["keyword"]))
    return sum(scores) / len(scores)

def evaluate_bm25(top_k: int = 5) -> float:
    scores = []
    for item in TEST_SET:
        results = search_bm25(item["query"], top_k=top_k)
        scores.append(reciprocal_rank(results, item["keyword"]))
    return sum(scores) / len(scores)

def evaluate_hybrid_rerank(top_k: int = 5) -> float:
    scores = []
    for item in TEST_SET:
        candidates = hybrid_search(item["query"], top_k=10, candidate_k=10)
        reranked = rerank(item["query"], candidates, top_k=top_k)
        scores.append(reciprocal_rank(reranked, item["keyword"]))
    return sum(scores) / len(scores)

def diagnose():
    print("\n=== 逐题诊断:对比向量检索 vs 混合检索+Rerank ===\n")
    for item in TEST_SET:
        v_results = vector_search(item["query"], top_k=5)
        v_rr = reciprocal_rank(v_results, item["keyword"])

        candidates = hybrid_search(item["query"], top_k=10, candidate_k=10)
        h_results = rerank(item["query"], candidates, top_k=5)
        h_rr = reciprocal_rank(h_results, item["keyword"])

        flag = "⚠️ 混合更差" if h_rr < v_rr else ("✓ 混合更好" if h_rr > v_rr else "= 持平")
        print(f"[{flag}] 向量RR={v_rr:.2f} 混合RR={h_rr:.2f} | {item['query'][:60]}")
        if h_rr < v_rr:
            print(f"   混合检索前2名内容:")
            for rank, r in enumerate(h_results[:2], start=1):
                hit_mark = "✓命中" if item["keyword"].lower() in r["text"].lower() else "✗未命中"
                print(f"   第{rank}名[{hit_mark}] rerank_score={r.get('rerank_score', 0):.2f}: {r['text'][:100]}")

if __name__ == "__main__":
    print("=== 检索效果评估 (MRR) ===\n")

    vector_mrr = evaluate_vector()
    print(f"纯向量检索:        {vector_mrr:.4f}")

    bm25_mrr = evaluate_bm25()
    print(f"纯BM25关键词检索:  {bm25_mrr:.4f}")

    hybrid_mrr = evaluate_hybrid_rerank()
    print(f"混合检索+Rerank:   {hybrid_mrr:.4f}")
    diagnose()

