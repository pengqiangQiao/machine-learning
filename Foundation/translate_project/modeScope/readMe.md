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






echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

echo -e "\n=== PyTorch & CUDA ==="
python -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, available: {torch.cuda.is_available()}')"

echo -e "\n=== Key Libraries ==="
pip list | grep -E "transformers|peft|accelerate|datasets|modelscope" | sort

echo -e "\n=== Model Cache Paths ==="
if [ -d ～/.cache/modelscope/hub ]; then
    echo "ModelScope cache exists (likely using魔搭 models)"
    ls ～/.cache/modelscope/hub/ | head -n 3
fi
if [ -d ～/.cache/huggingface/hub ]; then
    echo "HuggingFace cache exists"
    ls ～/.cache/huggingface/hub/ | head -n 3
fi

echo -e "\n=== Data Sample (please provide manually) ==="
echo "👉 请手动贴 1-2 行你的 JSONL 数据，例如："
echo '{"input": "...", "target": "..."}'