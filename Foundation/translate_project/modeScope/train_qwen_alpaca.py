# train_qwen_scale.py
import os
import argparse
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    LOCAL_MODEL_PATH = "/root/.cache/modelscope/hub/models/Qwen/Qwen-1_8B-Chat"
    LOCAL_DATA_PATH = "/root/autodl-tmp/firefly/firefly-train-1.1M.jsonl"

    # === Tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_PATH,
        trust_remote_code=True,
        padding_side='right'
    )
    IM_END_ID = 151645
    IM_START_ID = 151643
    tokenizer.pad_token_id = IM_END_ID
    tokenizer.eos_token_id = IM_END_ID
    tokenizer.bos_token_id = IM_START_ID
    tokenizer.pad_token = "<|im_end|>"
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.bos_token = "<|im_start|>"

    print(f"✅ pad_token_id: {tokenizer.pad_token_id}")
    print(f"✅ eos_token_id: {tokenizer.eos_token_id}")

    # === 关键修复：强制使用 float32，彻底禁用 bf16/fp16 自动行为 ===
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float32,  # 强制 float32
    )
    model.enable_input_require_grads()

    # === LoRA 配置 ===
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", "w1", "w2"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # === 加载数据 ===
    data = []
    with open(LOCAL_DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.num_samples:
                break
            item = json.loads(line.strip())
            data.append({"input": item["input"], "target": item["target"]})
    dataset = Dataset.from_list(data)

    # === 预处理函数 ===
    def preprocess(example):
        input_text = f"<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n"
        target_text = f"{example['target']}<|im_end|>"
        full_text = input_text + target_text
        tokenized = tokenizer(full_text, max_length=512, truncation=True, add_special_tokens=False)
        input_ids = tokenized["input_ids"]
        source_len = len(tokenizer(input_text, add_special_tokens=False)["input_ids"])
        labels = [-100] * source_len + input_ids[source_len:]
        return {
            "input_ids": input_ids[:512],
            "labels": labels[:512]
        }

    tokenized_ds = dataset.map(preprocess, remove_columns=dataset.column_names)

    # === 训练参数：全程 float32（fp16=False），避免 4090 上的 bfloat16 陷阱 ===
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,      # ← 关键：关闭 fp16（4090 不需要，且可避免 dtype 混乱）
        bf16=False,      # 显式关闭
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        # 可选：如果你后续想用 TensorBoard，可设 report_to="tensorboard"
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        padding=True,
        max_length=512,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )

    print(f"🚀 开始训练（{args.num_samples} samples）...")
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"🎉 训练完成！模型已保存至: {args.output_dir}")

if __name__ == "__main__":
    main()