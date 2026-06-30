from app.core.config import client, DEFAULT_MODEL
from app.tools.hybrid_search import hybrid_search
from app.tools.reranker import rerank

RAG_PROMPT_TEMPLATE = """你是一个Kubernetes技术助手,请基于下面提供的参考资料回答用户的问题。

要求:
1. 优先使用参考资料中的信息回答,不要凭空编造
2. 如果参考资料中没有足够信息回答问题,请明确说明"知识库中没有找到相关信息",不要瞎猜
3. 回答末尾用[来源: 文件名]的格式标注信息来自哪个文档

参考资料:
{context}

用户问题:{query}
"""

def build_context(chunks: list[dict]) -> str:
    """把检索到的chunk拼接成prompt里的参考资料部分,每段标注来源"""
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[资料{i}, 来源: {c['source']}]\n{c['text']}")
    return "\n\n".join(parts)

def rag_answer(query: str, top_k: int = 3) -> dict:
    """完整RAG流程:检索 -> Rerank -> 拼接prompt -> 生成回答"""
    candidates = hybrid_search(query, top_k=10, candidate_k=10)
    top_chunks = rerank(query, candidates, top_k=top_k)

    context = build_context(top_chunks)
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, query=query)

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": [{"source": c["source"], "id": c["id"]} for c in top_chunks],
    }