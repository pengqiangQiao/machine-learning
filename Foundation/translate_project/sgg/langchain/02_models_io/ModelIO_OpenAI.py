# Please install OpenAI SDK first: `pip install openai`
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(encoding='utf-8')

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API"),
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello,你是谁"},
    ],
    stream=False
)

print(response.choices[0].message.content)