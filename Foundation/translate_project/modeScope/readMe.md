# 卸载 PyTorch 生态
pip uninstall torch torchvision torchaudio -y

# 卸载 HuggingFace 和魔搭生态
pip uninstall transformers tokenizers sentencepiece datasets modelscope -y

#一般步骤是先执行以上一个，再执行requirement.xml

# 安装 PyTorch 2.1.0 及相关包（CUDA 12.1） 一般不需要自己安装,在autoDL里已经装好了
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 检查pytorch版本号
pip show torch torchvision


# 清除旧包
pip cache purge

数据集合直接用魔搭的git下载方便
git clone https://www.modelscope.cn/datasets/wyj123456/firefly.git

# 1. 安装 Git LFS
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs


# 2. 拉取真实数据
cd ～/autodl-tmp/firefly-train-1.1M
git lfs pull

# 3. 验证
wc -l firefly-train-1.1M.jsonl   # 应该有约 110 万行
head firefly-train-1.1M.jsonl 






你好！我正在 RTX 4090 单卡 上微调 Qwen-1.8B-Chat（ModelScope 版），为确保训练脚本 100% 安全且适配我的本地路径，我按以下步骤提供信息：

📁【关键路径】
- 模型本地路径：/root/.cache/modelscope/hub/models/Qwen/Qwen-1_8B-Chat
- 数据集本地路径：/data/my_finetune_data/train.jsonl

1️⃣ 【环境检查结果】
（粘贴 collect_env.sh 的输出）

（粘贴 env_check.py 的输出）

2️⃣ 【检查器代码】
这是我准备使用的配置检查脚本 check_training_config.py：
（粘贴 check_training_config.py 的完整代码）

3️⃣ 【数据样例】
我的数据格式如下（JSONL，每行一个样本）：
{"input": "中国的首都是哪里？", "target": "中国的首都是北京。"}
{"instruction": "写一首诗", "input": "", "output": "春风又绿江南岸..."}

4️⃣ 【请求】
请根据以上信息：
- 告诉我应该如何运行 check_training_config.py（即完整的命令行）
- 后续我会运行它并反馈结果，再请你生成最终的微调训练脚本（需正确加载上述模型和数据路径）

谢谢！