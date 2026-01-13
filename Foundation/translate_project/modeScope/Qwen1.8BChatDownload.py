from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen-1_8B-Chat')
print(f"模型已下载至: {model_dir}")