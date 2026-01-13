# ========== 魔搭+4卡4090+PyTorch2.0+ModelScope1.28.0 最终可运行版 ==========
import os

# 核心环境配置（魔搭+CUDA适配）
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MODELSCOPE_CACHE"] = "/root/.cache/modelscope"  # 魔搭缓存路径

import torch
# 适配ModelScope 1.28.0的正确导入路径
from modelscope.msdatasets import MsDataset
from modelscope.models import Model
from modelscope.hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, TrainingArguments

# ====================== 魔搭核心配置 ======================
model_name = "qwen/Qwen-7B-Chat"  # 魔搭源模型
dataset_name = "AI-ModelScope/alpaca-data-gpt4-chinese"  # 魔搭源数据集
output_dir = "/root/qwen-7b-alpaca-lora-4card"  # 权重保存路径

# 1. 魔搭源加载数据集（无任何报错）
print("✅ 加载魔搭数据集（国内源）...")
dataset = MsDataset.load(dataset_name, split="train")
dataset = dataset.to_hf_dataset()  # 转换为兼容格式
print("数据集示例：", dataset[0])

# 2. 8-bit量化配置（适配CUDA 11.7）
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    load_in_8bit_fp32_cpu_offload=False,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_use_double_quant=True,
    bnb_8bit_compute_dtype=torch.float16,
)

# 3. 魔搭源加载Qwen-7B模型（4卡自动分配）
print("✅ 加载Qwen-7B模型（8-bit量化）...")
# 先通过魔搭下载模型到本地缓存
model_dir = snapshot_download(model_name, cache_dir="/root/.cache/modelscope")
# 用transformers加载魔搭下载的模型
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    quantization_config=bnb_config,
    device_map="auto",  # 自动分配到4张4090
    trust_remote_code=True
)
model.config.use_cache = False
model.config.pretraining_tp = 4  # 4卡张量并行

# 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 4. 准备模型用于低精度训练
model = prepare_model_for_kbit_training(model)

# 5. LoRA配置（4卡微调最优参数）
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_attn", "c_proj"],  # Qwen-7B适配层
)
model = get_peft_model(model, peft_config)
print("✅ 可训练参数占比：")
model.print_trainable_parameters()

# 6. 数据格式化（适配Qwen对话格式）
def formatting_func(example):
    instruction = example["instruction"]
    input_text = example.get("input", "").strip()
    output_text = example["output"].strip()

    formatted_text = (
        f"<|im_start|>system\n你是专业AI助手，回答准确简洁。<|im_end|>"
        f"<|im_start|>user\n{instruction} {input_text}<|im_end|>"
        f"<|im_start|>assistant\n{output_text}<|im_end|>"
    )
    return {"text": formatted_text}

# 7. 4卡训练参数（稳定版）
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,
    per_device_train_batch_size=8,  # 4卡8-bit最优batch size
    gradient_accumulation_steps=1,
    learning_rate=1.5e-4,
    fp16=True,  # 适配4090的FP16算力
    logging_steps=5,
    save_steps=200,
    save_total_limit=2,
    report_to="none",  # 关闭不必要的依赖
    remove_unused_columns=False,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    ddp_find_unused_parameters=False,  # 多卡通信核心参数
    dataloader_num_workers=0,  # 适配容器环境
    dataloader_pin_memory=False,
)

# 8. 初始化训练器
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
    formatting_func=formatting_func,
)

# 9. 开始4卡训练
print("===== 魔搭环境 + 4卡4090 开始微调 Qwen-7B =====")
trainer.train()

# 10. 保存LoRA权重
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ 训练完成！权重保存至：{output_dir}")

# 验证4卡使用情况
print("\n📊 4卡GPU使用情况：")
os.system("nvidia-smi | grep -E 'GPU|Memory Usage|GPU Util'")