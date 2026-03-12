# pip install unstructured
# pip install docx2txt
# pip install python-docx
from langchain.chat_models  import  init_chat_model
import os

from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.prompts import PromptTemplate
from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import Redis
import os
from dotenv import load_dotenv
load_dotenv()
# 没有使用RAG，直接查询大模型，出现歧义，上课时先给学生演示before情况，没有用RAG
'''llm = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("aliQwen-api"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
response=llm.invoke("00000是什么意思")
print(response.content)'''

llm = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=os.getenv("QWEN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

prompt_template = """
    请使用以下提供的文本内容来回答问题。仅使用提供的文本信息，
    如果文本中没有相关信息，请回答"抱歉，提供的文本中没有这个信息"。

    文本内容：
    {context}

    问题：{question}

    回答：
    "
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# 1. 初始化阿里千问 Embedding 模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",  # 支持 v1 或 v2
    dashscope_api_key=os.getenv("QWEN_API_KEY")  # 从环境变量读取
)

# 4. 加载文档
# 4.1 TextLoader 无法处理 .docx 格式文件，专门用于加载纯文本文件的（如 .txt）
# loader = TextLoader("alibaba-more.docx", encoding="utf-8")

# 4.2 LangChain提供了Docx2txtLoader专门用于加载.docx文件，先通过pip install docx2txt
loader = Docx2txtLoader("alibaba-java.docx")  # 直接传入文件路径即可
documents = loader.load()

# 5. 分割文档
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, length_function=len)
texts = text_splitter.split_documents(documents)

print(f"文档个数:{len(texts)}")

# 6. 创建向量存储
# 连接到 Redis 并存入向量（自动调用 embeddings 嵌入）
vector_store = Redis.from_documents(
    documents=documents,
    embedding=embeddings,
    redis_url="redis://:YourStrongPassword123!@localhost:6379",  # 替换为你的 Redis 地址
    index_name="my_index3",  # 向量索引名称
)

retriever = vector_store.as_retriever(search_kwargs={"k": 2})
# context 的值是检索到的文档片段，会和透传的问题一起填充 Prompt
# 8. 创建Runnable链
rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
)

# 9. 提问
question = "00000和A0001分别是什么意思"
result = rag_chain.invoke(question)
print("\n问题:", question)
print("\n回答:", result.content)