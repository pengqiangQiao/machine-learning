"""
测试所有文件的 NumPy 2.0 兼容性
Test NumPy 2.0 compatibility for all files
"""

import sys
import importlib.util

def test_import(module_name, file_path):
    """测试导入模块"""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"✓ {module_name}: 导入成功")
        return True
    except Exception as e:
        print(f"✗ {module_name}: 导入失败")
        print(f"  错误: {str(e)}")
        return False

def main():
    """测试所有机器学习相关的 Python 文件"""
    print("="*60)
    print("测试 NumPy 2.0 兼容性")
    print("="*60)
    
    test_files = [
        ("ml_math_tutorial", "ml_math_tutorial.py"),
        ("ml_math_advanced", "ml_math_advanced.py"),
        ("ml_math_foundations", "ml_math_foundations.py"),
        ("ml_data_preprocessing", "ml_data_preprocessing.py"),
        ("ml_linear_regression", "ml_linear_regression.py"),
        ("ml_logistic_regression", "ml_logistic_regression.py"),
        ("ml_clustering", "ml_clustering.py"),
        ("ml_decision_tree", "ml_decision_tree.py"),
        ("ml_neural_network", "ml_neural_network.py"),
        ("ml_optimization", "ml_optimization.py"),
        ("ml_model_evaluation", "ml_model_evaluation.py"),
        ("ml_advanced_algorithms", "ml_advanced_algorithms.py"),
        ("ml_deep_learning", "ml_deep_learning.py"),
        ("ml_probabilistic_graphical_models", "ml_probabilistic_graphical_models.py"),
    ]
    
    results = []
    for module_name, file_path in test_files:
        result = test_import(module_name, file_path)
        results.append((module_name, result))
    
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    success_count = sum(1 for _, result in results if result)
    total_count = len(results)
    
    print(f"\n成功: {success_count}/{total_count}")
    print(f"失败: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n✓ 所有文件都兼容 NumPy 2.0！")
        return 0
    else:
        print("\n✗ 部分文件存在兼容性问题，请检查上述错误信息。")
        return 1

if __name__ == "__main__":
    sys.exit(main())