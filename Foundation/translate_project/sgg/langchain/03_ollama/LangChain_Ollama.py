# pip install -qU langchain-ollama
# pip install -U ollama

from langchain_ollama import ChatOllama

# 设置本地模型，不使用深度思考
model = ChatOllama(base_url="http://localhost:11434", model="qwen2.5:latest", reasoning=False)
# 打印结果，
print(model.invoke("什么是LangChain，100字以内回答"))

