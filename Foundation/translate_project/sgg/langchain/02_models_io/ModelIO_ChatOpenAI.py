from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv(encoding='utf-8')
chatLLM = ChatOpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",  # 此处以qwen-plus为例，您可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    # other params...
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你是谁？"}]

response = chatLLM.invoke(messages)
print(response)
print("*" * 50)
print(response.content)