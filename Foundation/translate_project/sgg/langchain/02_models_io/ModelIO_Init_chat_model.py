# 1.导入依赖
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv(encoding='utf-8')


# 2.实例化模型
model = init_chat_model(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API"),
    base_url="https://api.deepseek.com"
)

# 3.调用模型
print(model.invoke("你是谁").content)