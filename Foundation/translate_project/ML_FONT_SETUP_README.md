# Matplotlib 中文字体配置说明

## 问题描述

在使用 matplotlib 绘制包含中文的图表时，可能会出现以下警告：

```
UserWarning: Glyph 39044 (\N{CJK UNIFIED IDEOGRAPH-9884}) missing from font(s) DejaVu Sans.
```

这是因为 matplotlib 默认使用的字体不支持中文字符。

## 解决方案

本项目已经添加了 `ml_font_config.py` 模块来自动配置中文字体支持。

### 方法1：自动配置（推荐）

所有机器学习脚本已经自动导入字体配置：

```python
from ml_font_config import setup_chinese_font
setup_chinese_font()
```

直接运行脚本即可，字体配置会自动完成。

### 方法2：手动配置

如果需要在其他脚本中使用中文字体，只需在导入 matplotlib 后添加：

```python
import matplotlib.pyplot as plt
from ml_font_config import setup_chinese_font

# 配置中文字体
setup_chinese_font()

# 然后正常使用 matplotlib
plt.plot([1, 2, 3], [1, 2, 3])
plt.title('中文标题')
plt.show()
```

### 方法3：测试字体配置

运行字体配置模块来测试是否正常工作：

```bash
python ml_font_config.py
```

这将显示一个包含中文的测试图表。

## 支持的操作系统

### Windows
- 自动使用：微软雅黑 (Microsoft YaHei) 或 黑体 (SimHei)
- 系统自带，无需额外安装

### macOS
- 自动使用：苹方 (PingFang SC) 或 黑体 (Heiti SC)
- 系统自带，无需额外安装

### Linux
- 自动使用：文泉驿微米黑 (WenQuanYi Micro Hei) 或 思源黑体 (Noto Sans CJK)
- 如果没有安装，请运行：

**Ubuntu/Debian:**
```bash
sudo apt-get install fonts-wqy-microhei
```

**CentOS/RHEL:**
```bash
sudo yum install wqy-microhei-fonts
```

**Arch Linux:**
```bash
sudo pacman -S wqy-microhei
```

## 使用项目自带字体

如果系统没有合适的中文字体，可以使用项目 `ttf/` 目录下的字体文件：

1. 确保 `ttf/SourceHanSerif-VF.ttf.ttc` 文件存在
2. 字体配置模块会自动检测并使用该字体

## 已更新的文件

以下文件已经添加了中文字体支持：

- ✅ `ml_linear_regression.py` - 线性回归
- ✅ `ml_logistic_regression.py` - 逻辑回归
- ✅ `ml_decision_tree.py` - 决策树
- ✅ `ml_neural_network.py` - 神经网络
- ✅ `ml_clustering.py` - 聚类算法
- ✅ `ml_model_evaluation.py` - 模型评估

## 依赖包

确保已安装以下依赖：

```bash
pip install -r ml_requirements.txt
```

主要依赖：
- matplotlib >= 3.4.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- seaborn >= 0.11.0

## 常见问题

### Q: 仍然显示方框或乱码？

A: 尝试以下步骤：
1. 确认系统已安装中文字体
2. 清除 matplotlib 缓存：
   ```bash
   rm -rf ~/.matplotlib
   ```
3. 重新运行脚本

### Q: 如何指定特定字体？

A: 修改 `ml_font_config.py` 中的字体列表，将你想要的字体放在列表最前面。

### Q: 负号显示为方框？

A: 这个问题已经在配置中解决：
```python
plt.rcParams['axes.unicode_minus'] = False
```

## Java 开发者注意事项

如果你是从 Java 转到 Python 的开发者：

- **Java**: 使用 `Font font = new Font("Microsoft YaHei", Font.PLAIN, 12);`
- **Python**: 使用 `setup_chinese_font()` 自动配置

Java 中使用 JFreeChart 或 Swing 绘图时需要手动设置字体，而 Python 的这个配置模块会自动处理。

## 更多信息

- Matplotlib 官方文档：https://matplotlib.org/
- 字体配置文档：https://matplotlib.org/stable/tutorials/text/text_props.html