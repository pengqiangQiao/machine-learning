#!/bin/bash
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

echo -e "\n=== PyTorch & CUDA ==="
python -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, available: {torch.cuda.is_available()}')"

echo -e "\n=== Key Libraries ==="
pip list | grep -E "transformers|peft|accelerate|datasets|modelscope" | sort

echo -e "\n=== Model Cache Paths ==="
if [ -d ～/.cache/modelscope/hub ]; then
    echo "ModelScope cache exists"
    ls ～/.cache/modelscope/hub/ | head -n 3
fi
if [ -d ～/.cache/huggingface/hub ]; then
    echo "HuggingFace cache exists"
    ls ～/.cache/huggingface/hub/ | head -n 3
fi

echo -e "\n=== Data Sample ==="
echo "（请手动补充 1-2 行你的数据）"