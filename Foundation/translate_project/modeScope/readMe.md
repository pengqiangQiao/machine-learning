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