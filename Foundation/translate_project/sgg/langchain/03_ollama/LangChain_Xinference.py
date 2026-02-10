# 安装依赖（只需一次）
# pip install -U langchain-openai

from langchain_openai import ChatOpenAI
from pydantic import SecretStr

# 替换为你的实际 Model UID（从 Web UI 复制！）
MODEL_UID = "Qwen3-0.6B"  # ←←← 重要！

model = ChatOpenAI(
    base_url="https://u862956-a2e0-caf8526b.westc.gpuhub.com:8443/v1",
    api_key=SecretStr("sk-no-key-needed"),  # Xinference 不验证 key，但 LangChain 要求非空
    model=MODEL_UID,             # 必须是 UID，不是模型名
    temperature=0.0,
)

response = model.invoke("什么是LangChain？请用500字以内回答。")
print(response.content)
response = model.invoke("你是qWen3-0.6B吗？请用500字以内回答。")
print(response.content)