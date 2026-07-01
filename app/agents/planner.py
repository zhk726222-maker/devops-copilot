import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.agents.rag_agent import rag_answer
from app.agents.nl2sql_agent import nl2sql_answer
from app.agents.tool_agent import tool_agent_answer

MERGED_MODEL_PATH = "data/models/planner-merged"
SYSTEM_PROMPT = "你是一个任务路由助手。判断用户的问题应该交给哪个子系统处理:RAG(知识问答)、SQL(数据查询)、TOOL(执行操作)。只回答一个词。"

# 模块级缓存:模型只在第一次调用时加载,后续复用
_tokenizer = None
_model = None

def _load_model():
    global _tokenizer, _model
    if _model is not None:
        return
    print("正在加载本地Planner路由模型...")
    _tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        MERGED_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    _model.eval()
    print("Planner模型加载完成")

def route(query: str) -> str:
    """用本地微调模型做路由判断,返回 RAG / SQL / TOOL"""
    _load_model()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(text, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    decision = _tokenizer.decode(new_tokens, skip_special_tokens=True).strip().upper()

    # 防御性处理,按优先级匹配
    if "TOOL" in decision:
        return "TOOL"
    if "SQL" in decision:
        return "SQL"
    return "RAG"

def planner_answer(query: str) -> dict:
    """完整流程:本地模型路由判断 -> 分发给对应子Agent -> 返回结果"""
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
        "帮我重启一下payment-api",
    ]
    for q in test_queries:
        print(f"\n{'='*50}")
        print(f"用户问题: {q}")
        result = planner_answer(q)
        print(f"路由到: {result['routed_to']}")
        print(f"回答: {result['answer']}")