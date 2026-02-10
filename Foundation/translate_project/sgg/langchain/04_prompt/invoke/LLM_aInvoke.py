# 1.导入依赖
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
import asyncio
from langchain.messages import HumanMessage,SystemMessage



#通过 python-dotenv 库读取 env 文件中的环境变量，并加载到当前运行的环境中
load_dotenv()

# 2.实例化模型
model = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

async def main():
    # 异步调用一条请求
    response = await model.ainvoke("解释一下LangChain是什么，简洁回答100字以内")
    print(f"响应类型：{type(response)}")
    print(response.content_blocks)

# 4.运行异步函数
if __name__ == "__main__":
    asyncio.run(main())

'''
LangChain 提供 ainvoke() 异步调用接口，用于在 异步环境（async/await） 中高效并行地执行模型推理。
它的核心作用是：让你同时调用多个模型请求而不阻塞主线程 —— 特别适合大批量请求或 Web 服务场景（如 FastAPI）
'''
