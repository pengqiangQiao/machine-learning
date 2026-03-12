# https://bailian.console.aliyun.com/cn-beijing/?productCode=p_efm&tab=doc#/doc/?type=model&url=2842587
import os
import dashscope
from http import HTTPStatus
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 1. 全局配置 API Key（官方推荐方式）
dashscope.api_key = os.getenv("QWEN_API_KEY")


def get_text_embedding(input_text):
    """获取文本的 embedding 向量"""
    try:
        # 2. 按官方推荐格式传入输入（列表形式，支持多文本）
        resp = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input=[input_text],  # 改为列表，兼容多文本场景
            # 可选：指定输出维度（如 128/256/512/1024，默认 1024）
            # dimension=1024
        )

        # 3. 状态码判断 + 提取核心向量结果
        if resp.status_code == HTTPStatus.OK:
            # 提取 embedding 向量（核心数据）
            embedding = resp.output["embeddings"][0]["embedding"]
            return embedding
        else:
            print(f"API 调用失败：{resp.code} - {resp.message}")
            return None

    # 4. 捕获各类异常（网络、密钥、参数等）
    except Exception as e:
        print(f"调用出错：{str(e)}")
        return None


# 测试调用
if __name__ == "__main__":
    input_text = "衣服的质量杠杠的"
    embedding = get_text_embedding(input_text)

    if embedding:
        print(f"文本：{input_text}")
        print(f"Embedding 向量长度：{len(embedding)}")  # 默认 1024 维
        print(f"向量：{embedding}")
        print(f"前 10 维向量：{embedding[:10]}")