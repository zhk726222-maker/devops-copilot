import json
from app.core.config import client, DEFAULT_MODEL
from app.tools.ops_tools import restart_service, get_logs, get_resource_usage, scale_service

# 工具名到真实Python函数的映射,后面拿到模型的调用指令后,通过这个字典找到对应函数去执行
TOOL_REGISTRY = {
    "restart_service": restart_service,
    "get_logs": get_logs,
    "get_resource_usage": get_resource_usage,
    "scale_service": scale_service,
}

# 按照Function Calling要求的JSON Schema格式,描述每个工具的功能和参数
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "restart_service",
            "description": "重启指定的服务",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string", "description": "要重启的服务名称"}
                },
                "required": ["service_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_logs",
            "description": "获取指定服务最近的运行日志",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string", "description": "服务名称"},
                    "lines": {"type": "integer", "description": "要获取的日志行数,默认20"},
                },
                "required": ["service_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_resource_usage",
            "description": "查看指定服务当前的CPU、内存占用和运行副本数",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string", "description": "服务名称"}
                },
                "required": ["service_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scale_service",
            "description": "将指定服务扩容或缩容到目标副本数",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string", "description": "服务名称"},
                    "replicas": {"type": "integer", "description": "目标副本数"},
                },
                "required": ["service_name", "replicas"],
            },
        },
    },
]

def tool_agent_answer(query: str) -> dict:
    """完整流程: 模型判断调用哪个工具 -> 真实执行 -> 模型基于结果生成回复"""
    messages = [{"role": "user", "content": query}]

    # 第一轮调用:让模型判断要不要调用工具、调用哪个
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        tools=TOOLS_SCHEMA,
    )
    message = response.choices[0].message

    if not message.tool_calls:
        # 模型判断不需要调用任何工具,直接返回它的文字回复
        return {"answer": message.content, "tool_called": None}

    tool_call = message.tool_calls[0]
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)

    # 真正执行对应的工具函数
    tool_function = TOOL_REGISTRY[tool_name]
    tool_result = tool_function(**tool_args)

    # 第二轮调用:把工具执行的真实结果交给模型,让它组织成自然语言回复
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
        }],
    })
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(tool_result, ensure_ascii=False),
    })

    final_response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
    )

    return {
        "answer": final_response.choices[0].message.content,
        "tool_called": tool_name,
        "tool_args": tool_args,
        "tool_result": tool_result,
    }

if __name__ == "__main__":
    test_queries = [
        "帮我重启一下payment-api这个服务",
        "查一下user-auth现在的CPU和内存占用情况",
    ]
    for q in test_queries:
        print(f"\n{'='*50}")
        print(f"用户问题: {q}")
        result = tool_agent_answer(q)
        print(f"调用的工具: {result['tool_called']}")
        print(f"回答: {result['answer']}")