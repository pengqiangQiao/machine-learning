import os
from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv


load_dotenv(encoding='utf-8')
# 初始化 deepseek
# 给学生们看看ChatDeepSeek类的源码，解释为什么不写调用地址,chat_modesl.py源码第176行
model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("DEEPSEEK_API"),
)

# 打印结果
print(model.invoke("什么是LangChain?100字以内回答，简洁"))