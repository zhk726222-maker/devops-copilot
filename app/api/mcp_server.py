"""
MCP(Model Context Protocol)协议规范的运维工具服务
手动实现协议格式,不依赖mcp SDK,避免依赖冲突
提供标准化的工具发现和调用接口
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any
from app.tools.ops_tools import (
    restart_service,
    get_logs,
    get_resource_usage,
    scale_service,
)

router = APIRouter(prefix="/mcp", tags=["MCP Tools"])

# ===== MCP工具注册表 =====
# 按照MCP协议规范定义每个工具的元数据
MCP_TOOLS = [
    {
        "name": "restart_service",
        "description": "重启指定的Kubernetes服务",
        "inputSchema": {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "description": "要重启的服务名称,如 payment-api"
                }
            },
            "required": ["service_name"],
        },
    },
    {
        "name": "get_logs",
        "description": "获取指定服务最近的运行日志",
        "inputSchema": {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "description": "服务名称"
                },
                "lines": {
                    "type": "integer",
                    "description": "要获取的日志行数,默认20",
                    "default": 20,
                },
            },
            "required": ["service_name"],
        },
    },
    {
        "name": "get_resource_usage",
        "description": "查看指定服务当前的CPU、内存占用和运行副本数",
        "inputSchema": {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "description": "服务名称"
                }
            },
            "required": ["service_name"],
        },
    },
    {
        "name": "scale_service",
        "description": "将指定服务扩容或缩容到目标副本数(安全范围1-20)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "service_name": {
                    "type": "string",
                    "description": "服务名称"
                },
                "replicas": {
                    "type": "integer",
                    "description": "目标副本数,必须在1-20之间"
                },
            },
            "required": ["service_name", "replicas"],
        },
    },
]

# 工具名到执行函数的映射
TOOL_REGISTRY = {
    "restart_service": restart_service,
    "get_logs": get_logs,
    "get_resource_usage": get_resource_usage,
    "scale_service": scale_service,
}

# ===== MCP协议接口 =====

@router.get("/tools/list")
def list_tools():
    """
    MCP工具发现接口:返回所有可用工具的元数据
    对应MCP协议的 tools/list 方法
    """
    return {
        "tools": MCP_TOOLS,
        "total": len(MCP_TOOLS),
    }

class ToolCallRequest(BaseModel):
    name: str
    arguments: dict[str, Any]

@router.post("/tools/call")
def call_tool(request: ToolCallRequest):
    """
    MCP工具调用接口:执行指定工具并返回结果
    对应MCP协议的 tools/call 方法
    """
    if request.name not in TOOL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"工具 '{request.name}' 不存在,可用工具: {list(TOOL_REGISTRY.keys())}"
        )

    tool_fn = TOOL_REGISTRY[request.name]
    try:
        result = tool_fn(**request.arguments)
        return {
            "content": [
                {
                    "type": "text",
                    "text": str(result),
                }
            ],
            "isError": False,
        }
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"参数错误: {str(e)}")
    except Exception as e:
        return {
            "content": [
                {
                    "type": "text",
                    "text": str(e),
                }
            ],
            "isError": True,
        }