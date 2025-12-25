# -*- coding: utf-8 -*-
"""
Matplotlib中文字体配置模块
Matplotlib Chinese Font Configuration Module

解决matplotlib显示中文时出现方框或乱码的问题
Solves the issue of Chinese characters appearing as boxes or garbled text in matplotlib

使用方法 / Usage:
    在需要显示中文的脚本开头导入：
    Import at the beginning of scripts that need to display Chinese:
    
    from ml_font_config import setup_chinese_font
    setup_chinese_font()
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import platform
import os
import sys

# 确保标准输出使用 UTF-8 编码
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python 3.6 及更早版本
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')


def setup_chinese_font(verbose=True):
    """
    配置matplotlib使用中文字体
    Configure matplotlib to use Chinese fonts
    
    Java对应：
    Java中使用Swing/AWT时需要设置字体：
    Font font = new Font("Microsoft YaHei", Font.PLAIN, 12);
    或者
    Font font = new Font("SimHei", Font.PLAIN, 12);
    
    该函数会根据操作系统自动选择合适的中文字体：
    - Windows: Microsoft YaHei (微软雅黑) 或 SimHei (黑体)
    - macOS: PingFang SC (苹方) 或 STHeiti (华文黑体)
    - Linux: WenQuanYi Micro Hei (文泉驿微米黑) 或 Noto Sans CJK
    """
    
    system = platform.system()
    
    # 根据操作系统选择字体
    if system == 'Windows':
        # Windows系统常用中文字体
        fonts = [
            'Microsoft YaHei',  # 微软雅黑
            'SimHei',           # 黑体
            'SimSun',           # 宋体
            'KaiTi',            # 楷体
        ]
    elif system == 'Darwin':  # macOS
        # macOS系统常用中文字体
        fonts = [
            'PingFang SC',      # 苹方-简
            'Heiti SC',         # 黑体-简
            'STHeiti',          # 华文黑体
            'STSong',           # 华文宋体
        ]
    else:  # Linux
        # Linux系统常用中文字体
        fonts = [
            'WenQuanYi Micro Hei',  # 文泉驿微米黑
            'WenQuanYi Zen Hei',    # 文泉驿正黑
            'Noto Sans CJK SC',     # 思源黑体
            'Droid Sans Fallback',  # Droid备用字体
        ]
    
    # 尝试设置字体
    font_set = False
    for font in fonts:
        try:
            # 设置全局字体
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            
            # 测试字体是否可用（简单测试，不创建图形）
            try:
                test_fig = plt.figure()
                test_ax = test_fig.add_subplot(111)
                test_ax.text(0.5, 0.5, '测试中文', fontsize=12)
                plt.close(test_fig)
            except:
                pass  # 即使测试失败也继续，因为字体可能已经设置成功
            
            if verbose:
                print(f"[OK] 成功设置中文字体: {font}")
            font_set = True
            break
        except Exception as e:
            continue
    
    if not font_set:
        if verbose:
            print("[WARNING] 警告: 未找到合适的中文字体，尝试使用项目自带字体...")
        # 尝试使用项目中的字体文件
        try:
            font_path = os.path.join(os.path.dirname(__file__), 'ttf', 'SourceHanSerif-VF.ttf.ttc')
            if os.path.exists(font_path):
                # 使用自定义字体
                font_prop = FontProperties(fname=font_path)
                plt.rcParams['font.family'] = font_prop.get_name()
                plt.rcParams['axes.unicode_minus'] = False
                if verbose:
                    print(f"[OK] 使用项目字体: {font_path}")
                font_set = True
        except Exception as e:
            pass
    
    if not font_set and verbose:
        print("[WARNING] 警告: 无法设置中文字体，中文可能显示为方框")
        print("解决方案:")
        print("1. Windows: 确保系统已安装微软雅黑或黑体")
        print("2. macOS: 系统自带中文字体")
        print("3. Linux: 安装中文字体包")
        print("   Ubuntu/Debian: sudo apt-get install fonts-wqy-microhei")
        print("   CentOS/RHEL: sudo yum install wqy-microhei-fonts")
        print("4. 或者将中文字体文件放到项目的 ttf/ 目录下")
    
    return font_set


def get_chinese_font():
    """
    获取中文字体属性对象
    Get Chinese font properties object
    
    Returns:
        FontProperties对象，可用于单独设置某个文本的字体
        FontProperties object that can be used to set font for individual text
    
    使用示例 / Usage example:
        font = get_chinese_font()
        plt.title('标题', fontproperties=font)
    """
    system = platform.system()
    
    if system == 'Windows':
        font_name = 'Microsoft YaHei'
    elif system == 'Darwin':
        font_name = 'PingFang SC'
    else:
        font_name = 'WenQuanYi Micro Hei'
    
    try:
        return FontProperties(family=font_name)
    except:
        # 如果失败，尝试使用项目字体
        font_path = os.path.join(os.path.dirname(__file__), 'ttf', 'SourceHanSerif-VF.ttf.ttc')
        if os.path.exists(font_path):
            return FontProperties(fname=font_path)
        else:
            return FontProperties()


def demo():
    """
    演示中文字体配置效果
    Demonstrate Chinese font configuration
    """
    setup_chinese_font()
    
    # 创建测试图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 测试1: 折线图
    axes[0, 0].plot([1, 2, 3, 4], [1, 4, 2, 3])
    axes[0, 0].set_title('折线图测试')
    axes[0, 0].set_xlabel('横轴标签')
    axes[0, 0].set_ylabel('纵轴标签')
    axes[0, 0].grid(True)
    
    # 测试2: 柱状图
    axes[0, 1].bar(['类别A', '类别B', '类别C'], [10, 20, 15])
    axes[0, 1].set_title('柱状图测试')
    axes[0, 1].set_ylabel('数值')
    
    # 测试3: 散点图
    axes[1, 0].scatter([1, 2, 3, 4], [2, 3, 1, 4])
    axes[1, 0].set_title('散点图测试')
    axes[1, 0].set_xlabel('特征1')
    axes[1, 0].set_ylabel('特征2')
    axes[1, 0].grid(True)
    
    # 测试4: 文本显示
    axes[1, 1].text(0.5, 0.5, '中文显示测试\n汉字、标点、数字123\n特殊符号：±×÷',
                   ha='center', va='center', fontsize=14)
    axes[1, 1].set_title('文本显示测试')
    axes[1, 1].axis('off')
    
    plt.suptitle('Matplotlib中文字体配置测试', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\n如果上面的图表中文显示正常，说明字体配置成功！")
    print("If Chinese characters display correctly above, font configuration is successful!")


if __name__ == '__main__':
    """
    直接运行此脚本可以测试中文字体配置
    Run this script directly to test Chinese font configuration
    """
    print("=" * 60)
    print("Matplotlib中文字体配置测试")
    print("Matplotlib Chinese Font Configuration Test")
    print("=" * 60)
    demo()