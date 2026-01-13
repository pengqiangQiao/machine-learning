import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ====================== 关键修改：数据集名称 ======================
model_name = "qwen/Qwen-7B-Chat-Int4"
dataset_name = "AI-ModelScope/alpaca-data-gpt4-chinese"  # 改为你指定的数据集
new_model = "qwen-7b-alpaca-lora"

# 1. 加载数据集（魔搭源，自动下载）
print("正在加载数据集...")
# 加载训练集，可指定split="train[:10%]"先跑10%数据测试
dataset = load_dataset(dataset_name, split="train")
# 查看单条数据，确认格式（新手必做）
print("数据集示例：", dataset[0])

# 2. 4-bit量化配置（适配4090单卡）
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # 最优4bit量化方式
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # 二次量化，进一步降显存
)

# 3. 加载Qwen模型和Tokenizer
print("正在加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # 自动分配到4090
    trust_remote_code=True,  # Qwen必须加
)
model.config.use_cache = False  # 训练时关闭缓存，避免报错
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Qwen默认无pad_token，用eos_token替代
tokenizer.padding_side = "right"  # 右填充，避免推理时警告

# 4. 准备模型用于k-bit训练
model = prepare_model_for_kbit_training(model)

# 5. LoRA配置（平衡效果和显存）
peft_config = LoraConfig(
    r=32,  # 4090单卡建议32，比64显存占用低，效果也够用
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["c_attn", "c_proj"],  # Qwen-7B的核心LoRA模块
)
model = get_peft_model(model, peft_config)
# 打印可训练参数（仅占总参数0.2%左右，显存友好）
print("可训练参数占比：")
model.print_trainable_parameters()


# 6. 核心：Alpaca数据转Qwen格式（无需修改）
def formatting_func(example):
    instruction = example["instruction"]
    input_text = example.get("input", "").strip()
    output_text = example["output"].strip()

    # 组合用户输入
    if input_text:
        user_prompt = f"以下是任务的说明：{instruction}\n以下是任务的输入：{input_text}"
    else:
        user_prompt = f"以下是任务的说明：{instruction}"

    # Qwen官方Chat格式（必须严格遵循）
    formatted_text = (
        f"<|im_start|>system\n你是一个专业、乐于助人的AI助手。<|im_end|>"
        f"<|im_start|>user\n{user_prompt}<|im_end|>"
        f"<|im_start|>assistant\n{output_text}<|im_end|>"
    )
    return {"text": formatted_text}


# 7. 训练参数（4090单卡最优配置）
training_args = TrainingArguments(
    output_dir="./qwen-finetune-results",  # 结果保存路径
    num_train_epochs=2,  # 2轮足够，避免过拟合
    per_device_train_batch_size=8,  # 4090 Int4可设8，显存足够
    gradient_accumulation_steps=1,
    learning_rate=1.5e-4,  # 针对7B-Int4优化的学习率
    fp16=True,  # 必须开启，降显存
    logging_steps=5,  # 每5步打印一次loss，方便监控
    save_steps=100,  # 每100步保存一次
    save_total_limit=2,  # 只保留最新2个模型，节省空间
    report_to="tensorboard",  # 生成loss曲线
    remove_unused_columns=False,  # 关键：保留所有字段，避免格式化函数报错
    lr_scheduler_type="cosine",  # 余弦学习率，训练更稳定
    warmup_ratio=0.05,  # 预热学习率，防止初期loss波动
)

# 8. 初始化训练器
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",  # 对应格式化后的字段
    max_seq_length=512,  # 限制序列长度，4090单卡建议512
    tokenizer=tokenizer,
    args=training_args,
    formatting_func=formatting_func,  # 应用数据格式化
)

# 9. 开始训练
print("===== 开始微调 =====")
trainer.train()

# 10. 保存LoRA权重（仅几十MB，可下载到本地）
trainer.model.save_pretrained(new_model)
print(f"微调完成！LoRA权重保存在：{new_model}")