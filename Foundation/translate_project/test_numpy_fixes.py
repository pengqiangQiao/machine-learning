"""
测试 NumPy 2.0 兼容性修复
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import math

print("="*60)
print("测试 NumPy 版本和兼容性修复")
print("="*60)

print(f"\nNumPy 版本: {np.__version__}")

# 测试 1: np.trapz vs np.trapezoid
print("\n测试 1: 梯形积分函数")
x = np.linspace(0, 1, 100)
y = x**2

try:
    result = np.trapezoid(y, x)
    print(f"  np.trapezoid: 成功 (结果: {result:.6f})")
except AttributeError:
    try:
        result = np.trapz(y, x)
        print(f"  np.trapz: 成功 (结果: {result:.6f})")
    except AttributeError:
        print("  错误: 两个函数都不可用")

# 测试 2: math.factorial vs np.math.factorial
print("\n测试 2: 阶乘函数")
n = 5

# 使用 math.factorial (推荐)
result_math = math.factorial(n)
print(f"  math.factorial({n}): 成功 (结果: {result_math})")

# 测试 np.math.factorial (已弃用)
try:
    result_np = np.math.factorial(n)
    print(f"  np.math.factorial({n}): 仍然可用 (结果: {result_np})")
except AttributeError:
    print(f"  np.math.factorial({n}): 已移除 (这是预期的)")

# 测试 3: 验证修复后的代码
print("\n测试 3: 验证修复后的代码片段")

# 模拟 ml_math_tutorial.py 中的代码
try:
    a, b = 0, 1
    n = 1000
    x = np.linspace(a, b, n)
    y = x**2
    
    # 使用兼容性代码
    try:
        numerical = np.trapezoid(y, x)
    except AttributeError:
        numerical = np.trapz(y, x)
    
    print(f"  积分计算: 成功 (结果: {numerical:.6f})")
except Exception as e:
    print(f"  积分计算: 失败 ({e})")

# 模拟 ml_math_advanced.py 中的代码
try:
    x = 0.5
    n_terms = 10
    exp_approx = sum([x**n / math.factorial(n) for n in range(n_terms)])
    print(f"  泰勒级数: 成功 (e^{x} ≈ {exp_approx:.6f})")
except Exception as e:
    print(f"  泰勒级数: 失败 ({e})")

print("\n" + "="*60)
print("所有测试完成！")
print("="*60)