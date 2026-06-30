import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# 模块级缓存,模型只在第一次调用时加载一次,避免每次重排序都重新加载浪费时间
_tokenizer = None
_model = None
_device = "cpu"  # 评估等批量任务用CPU跑,避免长任务触发GPU不稳定;后续微调部署阶段单独处理

def _load_model():
    global _tokenizer, _model
    if _model is not None:
        return
    print(f"正在加载Rerank模型到 {_device} ...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    _model.to(_device)
    _model.eval()
    print("Rerank模型加载完成")

def rerank(query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    """
    对候选chunk按和query的相关性重新打分排序
    candidates: 形如 [{"id": ..., "text": ..., ...}, ...] 的候选列表(来自混合检索)
    """
    _load_model()

    pairs = [[query, c["text"]] for c in candidates]
    with torch.no_grad():
        inputs = _tokenizer(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt", return_token_type_ids=False).to(_device)
        scores = _model(**inputs).logits.view(-1).float()

    scored = list(zip(candidates, scores.tolist()))
    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    for candidate, score in scored[:top_k]:
        results.append({**candidate, "rerank_score": score})
    return results

if __name__ == "__main__":
    from app.tools.hybrid_search import hybrid_search

    test_query = "What is the restart policy for a Pod"
    candidates = hybrid_search(test_query, top_k=10, candidate_k=10)
    print(f"混合检索拿到 {len(candidates)} 个候选,开始Rerank...\n")

    reranked = rerank(test_query, candidates, top_k=3)
    for i, r in enumerate(reranked):
        print(f"\n结果{i+1} (chunk id: {r['id']}, Rerank分数: {r['rerank_score']:.4f})")
        print(r["text"][:200])