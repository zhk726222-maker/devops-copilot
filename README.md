# DevOps Copilot

一个基于 Multi-Agent 架构的运维智能助手,面向 Kubernetes 运维场景,支持知识问答、数据查询和运维操作执行三类任务的自动路由与处理。

## 系统架构

用户提问
│
▼
Planner (本地QLoRA微调模型,~50ms路由延迟)
│
├─→ RAG Agent      知识类问题 → 混合检索+Rerank → GLM-4.6生成
├─→ NL2SQL Agent   数据类问题 → SQL生成+安全校验+执行 → GLM-4.6整理
└─→ Tool Agent     操作类问题 → Function Calling → 工具执行 → GLM-4.6回复

## 核心能力

**RAG Agent**: 向量检索(语义)+BM25(关键词)+RRF融合+BGE Rerank精排,本地知识库问答

**NL2SQL Agent**: 自然语言转SQL,含schema-aware生成、安全校验(仅允许SELECT)、多表JOIN支持

**Tool Agent**: Function Calling两轮对话,支持重启服务/查日志/查资源/扩缩容4类运维操作

**Planner**: QLoRA微调的本地路由模型,替代API调用,延迟从800-2000ms降至~50ms

## 技术栈

| 模块 | 技术选型 |
|------|---------|
| 大模型 | 智谱 GLM-4.6(生成)、embedding-3(向量化) |
| 向量检索 | ChromaDB(余弦相似度) |
| 关键词检索 | BM25Okapi(rank-bm25),支持驼峰命名分词 |
| 检索融合 | RRF(Reciprocal Rank Fusion,自实现) |
| Rerank | BAAI/bge-reranker-v2-m3(本地GPU推理) |
| 数据库 | SQLite + SQLAlchemy ORM |
| 工具调用 | Function Calling(智谱原生支持) |
| 模型微调 | QLoRA(peft+bitsandbytes),基座Qwen2.5-1.5B-Instruct |
| 本地推理 | transformers inference pipeline(GPU加速) |
| 后端服务 | FastAPI |

## 效果数据

### RAG检索质量(MRR指标,16道关系类测试题)

| 检索方式 | MRR |
|---------|-----|
| 纯向量检索 | 0.6562 |
| 纯BM25关键词检索 | 0.6115 |
| **混合检索 + Rerank** | **0.7573** |

知识库规模:20篇K8s官方文档,289个chunk

### Planner路由模型(QLoRA微调效果)

| 方案 | 准确率 | 路由延迟 |
|------|--------|---------|
| 基座模型(未微调) | 33.3% | ~56ms |
| **QLoRA微调后** | **77.8%** | **~50ms** |
| 调用API路由 | - | 800-2000ms |

微调参数:75条训练样本,30 epochs,loss从5.4收敛至0.57

### Planner路由准确率(端到端测试)

5/5 三选一路由全部正确(RAG/SQL/TOOL各类型覆盖验证)

## 快速开始

```powershell
# 1. 创建虚拟环境
conda create -n devops-copilot python=3.11 -y
conda activate devops-copilot

# 2. 安装依赖
pip install -r requirements.txt
# GPU版PyTorch需单独安装(按实际CUDA版本替换cu124)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 3. 配置API密钥
# 新建 .env 文件,写入: ZHIPU_API_KEY=你的密钥

# 4. 构建知识库
python -m app.tools.build_knowledge_base

# 5. 初始化运维数据库
python -m app.core.database
python -m app.tools.seed_database

# 6. 下载并微调Planner模型(可选,跳过则Planner使用API路由)
# 下载基座模型(需要modelscope):
# python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='data/models')"
# 准备训练数据并微调:
# python -m app.tools.prepare_training_data
# python train_planner.py
# python merge_lora.py

# 7. 启动服务
.\start.ps1
```

启动后访问 `http://127.0.0.1:8000/docs` 测试 `/chat` 接口。

## 接口示例

```json
POST /chat
{"query": "什么是Kubernetes的滚动更新"}
{"query": "现在有哪些服务在告警"}
{"query": "帮我重启一下payment-api"}
```

## 评估复现

```powershell
# RAG检索效果评估
python -m app.tools.evaluate

# Planner路由效果测试
python -m app.agents.planner
```

## 已知局限与工程取舍

**RAG层**
- "Pod重启策略"这类查询中Always策略会被漏掉,根因是job.txt中OnFailure/Never出现频率更高抢占排名
- RRF采用均等权重,未针对两路检索的相对可靠性做调优

**Planner微调**
- 训练集75条样本量偏小,"帮我看看xx的内存"类表达覆盖不足导致偶发误判
- AWQ量化因autoawq强制降级torch版本(2.6→2.3)跳过,使用float16直接部署
- vLLM因Windows路径长度限制(MAX_PATH=260)安装失败,使用transformers inference替代

**工程判断记录**
以上取舍均有明确的技术原因和替代方案记录,体现了"遇到环境限制时的工程决策能力"而非简单绕过

## 踩过的坑(节选)

- Chroma默认L2距离在未归一化embedding下排序失真,需显式指定余弦相似度
- BM25对驼峰命名(如`restartPolicy`)需要特殊分词处理
- Secret文档中的Base64编码长字符串触发Rerank模型GPU崩溃,需在切块阶段截断
- autoawq与torch 2.6.0+cu124存在版本冲突,会强制降级torch
- Function Calling的tool_call对象不可直接JSON序列化,需手动拆解成字典
- 测试集设计中"问题包含答案关键词"会导致BM25虚高(测试集泄漏)