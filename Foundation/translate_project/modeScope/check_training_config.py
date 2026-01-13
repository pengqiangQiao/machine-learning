#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微调脚本配置合理性检查器
用于在训练前静态分析 TrainingArguments + 模型加载配置是否与当前 GPU 兼容
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig


def check_gpu_compatibility():
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，跳过 GPU 检查")
        return "cpu"

    gpu_name = torch.cuda.get_device_name(0)
    compute_capability = torch.cuda.get_device_capability()
    cc = f"{compute_capability[0]}.{compute_capability[1]}"

    print(f"✅ GPU: {gpu_name} (Compute Capability: {cc})")

    # RTX 30/40 系列：不原生支持 bfloat16
    if "RTX 30" in gpu_name or "RTX 40" in gpu_name:
        return "consumer_gpu"  # 消费级卡
    elif "A100" in gpu_name or "H100" in gpu_name:
        return "datacenter_gpu"  # 数据中心卡
    else:
        return "unknown"


def validate_training_args(fp16: bool, bf16: bool, model_torch_dtype: str):
    gpu_type = check_gpu_compatibility()

    print("\n🔍 分析训练配置...")
    print(f"   fp16={fp16}, bf16={bf16}, model_torch_dtype={model_torch_dtype}")

    # 规则 1：消费级 GPU（如 4090）禁止使用 bf16
    if gpu_type == "consumer_gpu":
        if bf16:
            print("   ❌ 危险！RTX 4090 不原生支持 bfloat16，开启 bf16=True 极易导致数值溢出！")
            print("   💡 建议：bf16=False, fp16=False（用 float32 最稳）")
            return False
        if fp16:
            print("   ⚠️  注意：RTX 4090 支持 fp16，但 Qwen 旧版 modeling 可能有 dtype bug")
            print("   💡 推荐：若遇 masked_fill 报错，请关闭 fp16")

    # 规则 2：model_torch_dtype 应与混合精度一致
    if model_torch_dtype == "bfloat16" and not bf16:
        print("   ⚠️  模型加载为 bfloat16，但训练未启用 bf16，可能浪费精度或出错")

    if model_torch_dtype == "float16" and not fp16:
        print("   ⚠️  模型加载为 float16，但训练未启用 fp16，可能影响性能")

    print("   ✅ 配置基本合理（仍需结合模型实现验证）")
    return True


def check_qwen_target_modules(model_path: str):
    """检查 LoRA target_modules 是否匹配 Qwen 结构"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="auto"
        )

        # 获取一个层的命名示例
        layer = model.model.layers[0]
        names = [name for name, _ in layer.named_modules()]

        # Qwen 常见投影层名称
        expected = {"c_attn", "c_proj", "w1", "w2"}
        found = set()
        for n in names:
            if any(e in n for e in expected):
                found.add(n.split('.')[-1])

        if expected.issubset(found):
            print(f"   ✅ Qwen 目标模块检测正常: {sorted(found)}")
        else:
            print(f"   ⚠️  Qwen 模块名可能不匹配！检测到: {sorted(found)}, 期望包含: {sorted(expected)}")

    except Exception as e:
        print(f"   ❌ 模型结构检查失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="微调配置合理性检查")
    parser.add_argument("--model_path", type=str, required=True, help="本地模型路径")
    parser.add_argument("--fp16", action="store_true", help="是否启用 fp16")
    parser.add_argument("--bf16", action="store_true", help="是否启用 bf16")
    parser.add_argument("--model_torch_dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"],
                        help="模型加载时的 torch_dtype")

    args = parser.parse_args()

    print("=" * 60)
    print("🛠️  微调脚本配置合理性检查器（专为 Qwen + RTX 4090 优化）")
    print("=" * 60)

    validate_training_args(args.fp16, args.bf16, args.model_torch_dtype)
    check_qwen_target_modules(args.model_path)

    print("\n💡 使用建议：")
    print("   - RTX 4090 用户：优先使用 float32（fp16=False, bf16=False）")
    print("   - 若坚持用 fp16，请确保 Qwen modeling 文件已修复 dtype bug")


if __name__ == "__main__":
    main()