# Planner路由模型压测报告

## 实验设置

- 基座模型: Qwen2.5-1.5B-Instruct
- 微调方式: QLoRA (r=8, lora_alpha=16, target_modules=q_proj+v_proj)
- 训练数据: 75条自标注样本 (RAG/SQL/TOOL各25条)
- 训练轮次: 30 epochs, loss从5.4收敛至0.57
- 测试集: 9道与训练集措辞不同的新问题(泛化能力测试)
- 硬件: RTX 4060 Laptop GPU 8GB

## 准确率对比

| 方案 | 正确数 | 准确率 |
|------|--------|--------|
| 基座模型(未微调) | 3/9 | 33.3% |
| QLoRA微调后 | 7/9 | 77.8% |
| 提升幅度 | +4道 | **+44.5%** |

## 推理延迟对比

| 方案 | 冷启动延迟 | 稳定延迟 |
|------|-----------|---------|
| 调用智谱API(路由) | 800-2000ms | 800-2000ms |
| 本地微调模型 | ~1000ms(一次性) | **~50ms** |
| 延迟降低倍数 | - | **16-40x** |

## 显存占用

- 模型加载: 2945MB (~3GB)
- 训练峰值: ~1200MB (QLoRA 4bit量化训练)
- 剩余可用: ~5GB (可同时跑其他推理任务)

## 已知局限

- 测试集样本量较小(9道),结论有一定统计局限性
- 两道错误案例分析:
  - "帮我看看billing-service的内存" → 被预测为SQL,训练集中"看看"类表达覆盖不足
  - "Ingress和Service有什么区别" → 合并LoRA权重时轻微精度损失导致回退
- vLLM部署因Windows路径长度限制未能完成,使用transformers inference替代
- AWQ量化因依赖冲突(autoawq强制降级torch版本)跳过,直接使用float16精度部署

## 工程取舍说明

评估了vLLM和AWQ量化两种生产级部署方案,均因Windows环境兼容性问题放弃:
- AWQ: autoawq 0.2.x要求torch==2.3.1,与项目使用的torch 2.6.0+cu124不兼容
- vLLM: Windows路径长度限制(MAX_PATH=260)导致安装失败

最终选择transformers inference pipeline作为本地推理方案,对1.5B量级模型在单次路由场景下稳定延迟50ms完全满足需求。