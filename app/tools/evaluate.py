from app.tools.vectorstore import search as vector_search
from app.tools.bm25_search import search_bm25
from app.tools.hybrid_search import hybrid_search
from app.tools.reranker import rerank

# 测试集:每条包含查询问题、目标关键词(用来判断检索结果是否命中相关内容)
TEST_SET = [
    {"query": "How do I revert a Deployment to its last working version after a bad update", "keyword": "rollback"},
    {"query": "How does a Deployment know which Pods belong to it without manually labeling them", "keyword": "ownerReferences"},
    {"query": "Can a DaemonSet Pod be removed from a node when resources are running low", "keyword": "eviction"},
    {"query": "How many Pods can a Job run at the same time", "keyword": "parallelism"},
    {"query": "What is the simplest way to expose a Service on a static port across every node", "keyword": "NodePort"},
    {"query": "If I don't define any rules, can Pods talk to each other freely by default", "keyword": "default deny"},
    {"query": "What kind of role binding lets a user access resources across the entire cluster, not just one namespace", "keyword": "ClusterRoleBinding"},
    {"query": "How do I set a default resource request for containers that don't specify one themselves", "keyword": "LimitRange"},
    {"query": "What taint effect would actually kick existing Pods off a node, not just block new ones", "keyword": "NoExecute"},
    {"query": "How do I tell the autoscaler what CPU percentage to aim for when scaling Pods", "keyword": "target utilization"},
    {"query": "After how many consecutive failed checks does Kubernetes consider a container actually unhealthy", "keyword": "failureThreshold"},
    {"query": "How do I decode the actual value stored inside a Kubernetes Secret", "keyword": "base64 -d"},
    {"query": "What reclaim policy automatically removes the storage volume once it's no longer needed", "keyword": "Delete"},
    {"query": "Why do Pods in a StatefulSet have predictable numbered names like web-0, web-1", "keyword": "ordinal"},
    {"query": "How do I make a ConfigMap's data available as files inside my container", "keyword": "volumeMounts"},
    {"query": "Which namespace holds the core system components that Kubernetes itself relies on", "keyword": "kube-system"},
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
    for i, item in enumerate(TEST_SET, start=1):
        print(f"  正在评估第 {i}/{len(TEST_SET)} 题...")
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

