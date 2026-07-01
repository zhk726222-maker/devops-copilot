"""
Planner路由模型 QLoRA 微调脚本
基座模型: Qwen2.5-1.5B-Instruct
任务: 三分类路由(RAG / SQL / TOOL)
显存需求: ~3GB,适合RTX 4060 8G
"""
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# 关闭不必要的日志噪音
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ===== 路径配置 =====
MODEL_PATH = "data/models/Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_DATA_PATH = "data/training/routing_train.jsonl"
OUTPUT_DIR = "data/checkpoints/planner-lora"

# ===== 第一步:加载tokenizer =====
print("正在加载tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print(f"tokenizer加载完成,词表大小: {tokenizer.vocab_size}")

# ===== 第二步:配置4bit量化加载(这是QLoRA的核心——用4bit压缩加载原始模型,节省显存) =====
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                          # 用4bit精度加载模型权重
    bnb_4bit_use_double_quant=True,             # 双重量化,进一步压缩显存
    bnb_4bit_quant_type="nf4",                 # NF4量化类型,QLoRA论文推荐
    bnb_4bit_compute_dtype=torch.bfloat16,     # 计算时用bfloat16,平衡精度和速度
)

# ===== 第三步:加载模型 =====
print("正在加载基座模型(4bit量化)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",           # 自动把模型分配到可用设备(GPU优先)
    trust_remote_code=True,
)
model.config.use_cache = False   # 训练时关闭KV cache(推理时才需要,训练时关闭节省显存)
print("基座模型加载完成")

# ===== 第四步:配置LoRA适配器 =====
# LoRA的思路是:不直接修改原模型的权重矩阵W,而是在旁边加两个小矩阵A和B
# W_new = W + A*B,只训练A和B,W保持冻结
# r是这两个小矩阵的"秩"(rank),r越大表达能力越强但显存占用也越大
lora_config = LoraConfig(
    r=8,                        # LoRA秩,8是路由分类任务的常用值,够用且省显存
    lora_alpha=16,              # 缩放系数,通常设为r的2倍
    target_modules=["q_proj", "v_proj"],  # 在哪些层加LoRA,注意力层的Q和V矩阵是常用选择
    lora_dropout=0.05,          # 防止过拟合的dropout比例
    bias="none",
    task_type="CAUSAL_LM",      # 因果语言模型(GPT类模型的任务类型)
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 打印可训练参数量,应该只有总参数的1-2%

# ===== 第五步:加载训练数据 =====
print("正在加载训练数据...")
dataset = load_dataset("json", data_files=TRAIN_DATA_PATH, split="train")
print(f"训练样本数: {len(dataset)}")

def format_conversation(example):
    """把对话格式转成模型需要的文本格式"""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = dataset.map(format_conversation)

# ===== 第六步:配置训练参数 =====
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=30,             # 从5增加到30,小数据集需要更多轮次收敛
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=5,
    save_strategy="epoch",
    save_total_limit=3,              # 加这一行:只保留最近3个checkpoint,避免占满磁盘
    optim="paged_adamw_8bit",
    report_to="none",
)

# ===== 第七步:创建训练器并开始训练 =====
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=256,              # 路由任务输入输出都很短,256足够
)

print("\n===== 开始训练 =====")
print(f"训练设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"显存使用(训练前): {torch.cuda.memory_allocated()/1024**2:.1f}MB")

trainer.train()

print("\n===== 训练完成 =====")

# ===== 第八步:保存LoRA适配器权重 =====
# 注意:我们只保存LoRA适配器(A和B两个小矩阵),不保存完整模型
# 因为原始模型权重没有改变,只需要保存新增的这一小部分参数
LORA_SAVE_PATH = "data/checkpoints/planner-lora-final"
model.save_pretrained(LORA_SAVE_PATH)
tokenizer.save_pretrained(LORA_SAVE_PATH)
print(f"LoRA适配器已保存到: {LORA_SAVE_PATH}")
print(f"显存使用(训练后): {torch.cuda.memory_allocated()/1024**2:.1f}MB")