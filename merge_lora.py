"""
把LoRA适配器权重合并进基座模型,生成完整模型文件供vLLM加载
注意:合并时需要用float16精度加载基座模型(不能用4bit量化),大约需要3-4GB显存
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_PATH = "data/models/Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = "data/checkpoints/planner-lora-final"
MERGED_PATH = "data/models/planner-merged"

print("正在加载基座模型(float16精度)...")
print(f"当前显存使用: {torch.cuda.memory_allocated()/1024**2:.1f}MB")

# 合并LoRA时必须用float16加载,不能用4bit量化
# 因为量化加载的权重是整数格式,无法和LoRA的浮点权重做数学合并
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cpu",    # 合并操作在CPU上做,避免GPU显存不够
    trust_remote_code=True,
)
print(f"基座模型加载完成")

print("正在加载LoRA适配器...")
model = PeftModel.from_pretrained(model, LORA_PATH)

print("正在合并LoRA权重...")
model = model.merge_and_unload()  # 把LoRA的A*B矩阵加回原始权重W,得到完整模型

print(f"正在保存合并后的模型到 {MERGED_PATH} ...")
model.save_pretrained(MERGED_PATH)

# 同时保存tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.save_pretrained(MERGED_PATH)

print("合并完成!")
print(f"合并后模型保存在: {MERGED_PATH}")