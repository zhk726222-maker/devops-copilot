"""
Planner本地推理服务
用微调后合并的模型替代API调用做路由判断
对比项: API调用延迟 vs 本地模型推理延迟
"""
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MERGED_MODEL_PATH = "data/models/planner-merged"

print("正在加载微调后的Planner模型...")
start = time.time()

tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
)
model.eval()
load_time = time.time() - start
print(f"模型加载完成,耗时: {load_time:.1f}秒")
print(f"显存占用: {torch.cuda.memory_allocated()/1024**2:.1f}MB\n")

SYSTEM_PROMPT = "你是一个任务路由助手。判断用户的问题应该交给哪个子系统处理:RAG(知识问答)、SQL(数据查询)、TOOL(执行操作)。只回答一个词。"

def local_route(query: str) -> tuple[str, float]:
    """用本地微调模型做路由判断,返回(路由结果, 推理延迟ms)"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.time() - start) * 1000

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().upper()

    if "TOOL" in prediction:
        result = "TOOL"
    elif "SQL" in prediction:
        result = "SQL"
    else:
        result = "RAG"

    return result, latency_ms

# ===== 压测:对比本地模型 vs API调用的延迟 =====
TEST_QUERIES = [
    ("K8s里的Service是什么", "RAG"),
    ("NotificationWorker最近有没有报错", "SQL"),
    ("把payment-api重启一下", "TOOL"),
    ("Pod的几种重启策略分别是什么意思", "RAG"),
    ("现在值班的工程师是谁", "SQL"),
    ("帮我看看billing-service的内存", "TOOL"),
    ("Ingress和Service有什么区别", "RAG"),
    ("哪些服务部署失败过", "SQL"),
    ("把order-service扩到3个副本", "TOOL"),
]

print("===== 本地模型推理压测 =====\n")
latencies = []
correct = 0

for query, expected in TEST_QUERIES:
    result, latency = local_route(query)
    latencies.append(latency)
    is_correct = result == expected
    if is_correct:
        correct += 1
    status = "✓" if is_correct else "✗"
    print(f"{status} [{expected}→{result}] {latency:.0f}ms | {query}")

avg_latency = sum(latencies) / len(latencies)
p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

print(f"\n准确率: {correct}/{len(TEST_QUERIES)} ({correct/len(TEST_QUERIES)*100:.1f}%)")
print(f"平均延迟: {avg_latency:.0f}ms")
print(f"P95延迟: {p95_latency:.0f}ms")
print(f"\n对比参考: 调用智谱API做路由判断通常需要800-2000ms(含网络延迟)")
print(f"本地模型推理延迟降低约: {2000/avg_latency:.1f}x - {800/avg_latency:.1f}x")