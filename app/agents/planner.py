from app.core.config import client, DEFAULT_MODEL
from app.agents.rag_agent import rag_answer
from app.agents.nl2sql_agent import nl2sql_answer
from app.agents.tool_agent import tool_agent_answer

ROUTING_PROMPT = """你是一个任务路由助手,需要判断用户的问题应该交给哪个子系统处理。

三个可选的子系统:
1. RAG - 适合回答Kubernetes概念、原理、用法相关的知识性问题,比如"什么是ConfigMap""Pod重启策略是什么"
2. SQL - 适合回答需要查询具体运维数据的问题,比如"哪些服务在告警""谁部署了xx服务""现在谁在值班"
3. TOOL - 适合需要真正执行一个动作的请求,比如"重启xx服务""查一下xx服务的日志""把xx服务扩容到N个副本"

用户问题:{query}

只回答一个词:RAG 或者 SQL 或者 TOOL,不要任何解释。
"""

def route(query: str) -> str:
    """判断问题应该走RAG、SQL还是TOOL路径"""
    prompt = ROUTING_PROMPT.format(query=query)
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    decision = response.choices[0].message.content.strip().upper()

    # 防御性处理:按优先级依次判断,模型偶尔可能输出带标点或多余文字的内容
    if "TOOL" in decision:
        return "TOOL"
    if "SQL" in decision:
        return "SQL"
    return "RAG"  # 默认兜底走RAG

def planner_answer(query: str) -> dict:
    """完整流程:路由判断 -> 分发给对应子Agent -> 返回结果(附带路由信息方便调试)"""
    decision = route(query)

    if decision == "TOOL":
        result = tool_agent_answer(query)
        result["routed_to"] = "Tool Agent"
    elif decision == "SQL":
        result = nl2sql_answer(query)
        result["routed_to"] = "NL2SQL Agent"
    else:
        result = rag_answer(query)
        result["routed_to"] = "RAG Agent"

    return result

if __name__ == "__main__":
    test_queries = [
        "Pod的重启策略有哪几种",
        "现在有哪些服务在告警",
        "谁在负责值班",
        "帮我重启一下payment-api",
        "查一下notification-worker最近的日志",
    ]
    for q in test_queries:
        print(f"\n{'='*50}")
        print(f"用户问题: {q}")
        result = planner_answer(q)
        print(f"路由到: {result['routed_to']}")
        print(f"回答: {result['answer']}")