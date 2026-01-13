# download_firefly.py
from modelscope import snapshot_download
import os

# 下载 wyj123456/firefly 到本地目录
local_dir = "./firefly_wyj"
if not os.path.exists(local_dir):
    print("正在从魔搭下载数据集...")
    local_dir = snapshot_download('wyj123456/firefly', cache_dir="./")
else:
    print(f"数据集已存在: {local_dir}")

print(f"数据集路径: {local_dir}")