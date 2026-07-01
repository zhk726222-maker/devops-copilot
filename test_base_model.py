import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "data/models/Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True
)
model.eval()

SYSTEM = "你是一个任务路由助手。判断用户的问题应该交给哪个子系统处理:RAG(知识问答)、SQL(数据查询)、TOOL(执行操作)。只回答一个词。"
TEST_CASES = [
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

correct = 0
latencies = []
for query, expected in TEST_CASES:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": query}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=5, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    latency = (time.time() - start) * 1000
    latencies.append(latency)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().upper()
    is_correct = expected in prediction
    if is_correct:
        correct += 1
    status = "✓" if is_correct else "✗"
    print(f"{status} [{expected}] {latency:.0f}ms 预测: {prediction[:15]!r} | {query}")

print(f"\n基座模型准确率: {correct}/9 ({correct/9*100:.1f}%)")
print(f"平均延迟(含冷启动): {sum(latencies)/len(latencies):.0f}ms")
print(f"稳定延迟(去掉第一次): {sum(latencies[1:])/len(latencies[1:]):.0f}ms")