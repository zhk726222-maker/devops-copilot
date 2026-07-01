
"""
验证微调后的LoRA模型路由效果
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_PATH = "data/models/Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "data/checkpoints/planner-lora-final"

print("正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# 加载LoRA适配器,叠加在基座模型上
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()
print("模型加载完成\n")

# 测试用例,刻意选一些和训练集措辞不同的新表达方式,测试泛化能力
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

SYSTEM_PROMPT = "你是一个任务路由助手。判断用户的问题应该交给哪个子系统处理:RAG(知识问答)、SQL(数据查询)、TOOL(执行操作)。只回答一个词。"

correct = 0
print("===== 路由效果验证 =====\n")

for query, expected in TEST_CASES:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,       # 路由结果只需要一个词,最多5个token足够
            do_sample=False,        # 不随机采样,用贪婪解码保证结果稳定可复现
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 只取新生成的部分(去掉输入的prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip().upper()

    # 宽松匹配:只要输出里包含正确标签词就算对
    is_correct = expected in prediction
    if is_correct:
        correct += 1

    status = "✓" if is_correct else "✗"
    print(f"{status} [{expected}] 预测: {prediction[:20]!r} | {query}")

print(f"\n准确率: {correct}/{len(TEST_CASES)} ({correct/len(TEST_CASES)*100:.1f}%)")