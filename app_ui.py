"""
DevOps Copilot - Streamlit前端Demo
运行方式: streamlit run app_ui.py
"""
import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="DevOps Copilot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 DevOps Copilot")
st.caption("基于Multi-Agent架构的Kubernetes运维智能助手 | RAG · NL2SQL · Tool Calling · QLoRA微调Planner")

# 侧边栏:系统说明
with st.sidebar:
    st.header("📖 使用说明")
    st.markdown("""
    **支持三类问题:**

    🔍 **知识问答 (RAG)**
    - 什么是Kubernetes的滚动更新
    - Pod和容器有什么区别
    - RBAC的ClusterRole和Role有什么区别

    📊 **数据查询 (NL2SQL)**
    - 现在有哪些服务在告警
    - 过去7天哪些服务出现过critical告警
    - 现在谁在值班

    ⚙️ **运维操作 (Tool)**
    - 帮我重启一下payment-api
    - 查一下billing-service的资源占用
    - 把order-service扩容到3个副本
    """)

    st.divider()
    st.header("🔧 系统状态")
    try:
        resp = requests.get(f"{API_BASE}/", timeout=3)
        if resp.status_code == 200:
            st.success("后端服务正常运行")
        else:
            st.error("后端服务异常")
    except Exception:
        st.error("无法连接后端服务,请先运行 .\\start.ps1")

    st.divider()
    st.header("📊 技术栈")
    st.markdown("""
    - **Planner**: QLoRA微调 Qwen2.5-1.5B
    - **RAG**: 混合检索 + BGE Rerank
    - **NL2SQL**: Schema-aware SQL生成
    - **Tools**: Function Calling + MCP协议
    - **后端**: FastAPI + GLM-4.6
    """)

# 主界面:对话
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "routed_to" in msg:
            st.caption(f"路由到: {msg['routed_to']}")

# 输入框
if query := st.chat_input("请输入你的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 调用后端API
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/chat",
                    json={"query": query},
                    timeout=120,
                )
                result = resp.json()
                answer = result.get("answer", "抱歉,没有获取到回答")
                routed_to = result.get("routed_to", "未知")

                st.markdown(answer)
                st.caption(f"路由到: {routed_to}")

                # 如果是SQL路由,额外显示执行的SQL语句
                if "sql" in result:
                    with st.expander("查看执行的SQL"):
                        st.code(result["sql"], language="sql")

                # 如果是Tool路由,额外显示工具调用信息
                if "tool_called" in result and result["tool_called"]:
                    with st.expander("查看工具调用详情"):
                        st.json({
                            "工具名": result.get("tool_called"),
                            "参数": result.get("tool_args"),
                        })

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "routed_to": routed_to,
                })

            except requests.exceptions.Timeout:
                st.error("请求超时,请稍后重试")
            except Exception as e:
                st.error(f"请求失败: {str(e)}")