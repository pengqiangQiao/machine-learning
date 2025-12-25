# -*- coding: utf-8 -*-
"""
批量测试所有机器学习相关的 Python 文件
Batch test all machine learning related Python files
"""

import sys
import subprocess
import os

# 确保标准输出使用 UTF-8 编码
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

def test_file(file_path, timeout=30):
    """测试单个文件"""
    print(f"\n{'='*60}")
    print(f"测试文件: {file_path}")
    print('='*60)
    
    # 使用虚拟环境的 Python
    python_exe = os.path.join('.venv', 'Scripts', 'python.exe')
    if not os.path.exists(python_exe):
        python_exe = 'python'
    
    try:
        # 运行文件，设置超时
        result = subprocess.run(
            [python_exe, file_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='replace',
            env={**os.environ, 'MPLBACKEND': 'Agg'}  # 使用非交互式后端
        )
        
        # 检查是否有错误
        if result.returncode != 0:
            print(f"[FAILED] 文件执行失败，退出码: {result.returncode}")
            if result.stderr:
                print("\n错误输出:")
                print(result.stderr)
            return False, result.stderr
        else:
            # 检查输出中是否有错误关键词
            error_keywords = ['Error', 'Exception', 'Traceback', 'AttributeError']
            has_error = any(keyword in result.stderr for keyword in error_keywords)
            
            if has_error:
                print("[WARNING] 文件执行完成但有警告/错误")
                print("\n标准错误输出:")
                print(result.stderr[:500])  # 只显示前500字符
                return True, result.stderr
            else:
                print("[OK] 文件执行成功")
                return True, None
    
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] 文件执行超时 (>{timeout}秒)")
        return False, "Timeout"
    except Exception as e:
        print(f"[ERROR] 测试过程出错: {str(e)}")
        return False, str(e)

def main():
    """主函数"""
    print("="*60)
    print("批量测试所有机器学习相关的 Python 文件")
    print("="*60)
    
    # 要测试的文件列表
    test_files = [
        "ml_math_tutorial.py",
        "ml_math_advanced.py",
        "ml_math_foundations.py",
        "ml_data_preprocessing.py",
        "ml_linear_regression.py",
        "ml_logistic_regression.py",
        "ml_clustering.py",
        "ml_decision_tree.py",
        "ml_neural_network.py",
        "ml_optimization.py",
        "ml_model_evaluation.py",
        "ml_advanced_algorithms.py",
        "ml_deep_learning.py",
        "ml_probabilistic_graphical_models.py",
        "ml_advanced_topics.py",
    ]
    
    results = {}
    errors = {}
    
    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"\n[SKIP] 文件不存在: {file_path}")
            results[file_path] = 'skip'
            continue
        
        # 对于可能有可视化的文件，增加超时时间
        timeout = 120 if file_path in ['ml_linear_regression.py', 'ml_clustering.py'] else 60
        success, error_msg = test_file(file_path, timeout=timeout)
        results[file_path] = 'success' if success else 'failed'
        if error_msg:
            errors[file_path] = error_msg
    
    # 打印汇总
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    success_count = sum(1 for v in results.values() if v == 'success')
    failed_count = sum(1 for v in results.values() if v == 'failed')
    skip_count = sum(1 for v in results.values() if v == 'skip')
    
    print(f"\n总计: {len(test_files)} 个文件")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"跳过: {skip_count}")
    
    if failed_count > 0:
        print("\n失败的文件:")
        for file_path, status in results.items():
            if status == 'failed':
                print(f"  - {file_path}")
                if file_path in errors:
                    # 只显示错误的前几行
                    error_lines = errors[file_path].split('\n')[:5]
                    for line in error_lines:
                        if line.strip():
                            print(f"    {line}")
    
    return 0 if failed_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())