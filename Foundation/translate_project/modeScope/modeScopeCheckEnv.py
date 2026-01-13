#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import subprocess
import importlib

def check_python():
    print("🔍 检查 Python 版本...")
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"   Python 版本: {sys.version}")
    if version not in ["3.9", "3.10"]:
        print("   ⚠️  警告：推荐使用 Python 3.9 或 3.10，当前版本可能有兼容性风险。")
    else:
        print("   ✅ Python 版本符合推荐。")

def check_package(pkg_name, expected_version=None, optional=False):
    try:
        pkg = importlib.import_module(pkg_name)
        if hasattr(pkg, '__version__'):
            ver = pkg.__version__
            if expected_version and ver != expected_version:
                print(f"   ⚠️  {pkg_name} 版本不匹配！期望 {expected_version}，实际 {ver}")
            else:
                print(f"   ✅ {pkg_name} == {ver}")
        else:
            print(f"   ✅ {pkg_name} 已安装（无 __version__ 属性）")
    except ImportError:
        if optional:
            print(f"   ℹ️  {pkg_name} 未安装（可选组件）")
        else:
            print(f"   ❌ {pkg_name} 未安装！")

def check_torch_cuda():
    print("\n🔍 检查 PyTorch 和 CUDA...")
    try:
        import torch
        print(f"   PyTorch 版本: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"   CUDA 可用: {cuda_available}")
        if cuda_available:
            print(f"   CUDA 版本 (PyTorch编译): {torch.version.cuda}")
            print(f"   GPU 数量: {torch.cuda.device_count()}")
            print(f"   当前 GPU: {torch.cuda.get_device_name(0)}")
            # 测试张量计算
            x = torch.randn(2, 3).to('cuda')
            y = torch.randn(3, 2).to('cuda')
            z = torch.mm(x, y)
            print("   ✅ GPU 张量计算测试通过！")
        else:
            print("   ❌ CUDA 不可用！请确认 PyTorch 安装了 CUDA 版本（非 CPU 版）")
    except Exception as e:
        print(f"   ❌ PyTorch 检查失败: {e}")

def main():
    print("=" * 50)
    print("魔搭 (ModelScope) + RTX 4090 环境检查工具")
    print("=" * 50)

    check_python()

    print("\n🔍 检查核心依赖包...")
    check_package("modelscope", expected_version="1.29.1")
    check_package("transformers", expected_version="4.36.2")
    check_package("datasets", expected_version="2.16.0")
    check_package("peft", expected_version="0.7.1")
    check_package("accelerate", expected_version="0.27.2")
    check_package("gradio", expected_version="3.46.0")
    check_package("numpy", expected_version="1.23.5")

    # 可选但常用
    check_package("sentencepiece", optional=True)
    check_package("tokenizers", optional=True)
    check_package("PIL", optional=True)  # Pillow
    check_package("cv2", optional=True)  # opencv-python

    check_torch_cuda()

    print("\n✅ 环境检查完毕！")
    print("如有 ❌ 项，请根据提示重新安装对应包。")

if __name__ == "__main__":
    main()