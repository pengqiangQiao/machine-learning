from langchain.chat_models import init_chat_model
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler # 1. 导入回调基类
import redis
from loguru import logger
import os
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional  # 2. 导入类型提示

load_dotenv()

# --- 配置部分 ---
REDIS_URL = "redis://:YourStrongPassword123!@localhost:6379"
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


# --- 1. 定义自定义日志回调处理器 ---
class LoggingCallbackHandler(BaseCallbackHandler):
    """自定义回调：记录输入 Prompt 和输出 Response"""

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """当 LLM 开始运行时触发：记录最终发送给模型的完整提示词"""
        logger.info("=" * 30 + " [LLM 请求开始] " + "=" * 30)
        for i, prompt in enumerate(prompts):
            # prompts 是一个列表，通常包含拼接好 history 后的完整内容
            logger.info(f"[Prompt 片段 {i + 1}]:\n{prompt}")
        logger.info("-" * 80)

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """当 LLM 结束时触发：记录模型返回的内容"""
        # response.generations[0][0].text 是具体的回复内容
        if hasattr(response, 'generations') and response.generations:
            content = response.generations[0][0].text
            logger.info(f"[AI 回答]:\n{content}")
        logger.info("=" * 30 + " [LLM 请求结束] " + "=" * 30 + "\n")

    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        """记录错误"""
        logger.error(f"[发生错误]: {str(error)}")


# --- 模型与链设置 ---
llm = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # 可选：也可以在这里直接加 callbacks=[LoggingCallbackHandler()]
    # 但为了灵活控制（比如生产环境关闭日志），建议在 invoke 时动态添加
)

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder("history"),
    ("human", "{question}")
])


def get_session_history(session_id: str) -> RedisChatMessageHistory:
    history = RedisChatMessageHistory(
        session_id=session_id,
        url=REDIS_URL,
    )
    return history


chain = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

config_base = RunnableConfig(configurable={"session_id": "user-001"})

# --- 主循环 ---
print("开始对话（输入 'quit' 退出）")
while True:
    question = input("\n输入问题：")
    if question.lower() in ['quit', 'exit', 'q']:
        break

    # 【关键修改点】
    # 构建一个新的 config，将自定义的 handler 注入进去
    # 这样每次 invoke 都会触发 on_llm_start 和 on_llm_end
    config_with_logging = {
        **config_base,
        "callbacks": [LoggingCallbackHandler()]
    }

    try:
        # 传入带有 callbacks 的 config
        response = chain.invoke({"question": question}, config=config_with_logging)

        # 原有的日志（保留作为简单记录）
        logger.info(f"简略记录 - AI回答:{response.content}")

        # 强制写入磁盘 (根据你的需求保留)
        redis_client.save()

    except Exception as e:
        logger.error(f"调用出错: {e}")