
# 方式1：外部加载Prompt,将 prompt 保存为 JSON
from langchain_core.prompts import load_prompt

template = load_prompt("prompt.json", encoding="utf-8")
print(template.format(name="张三", what="搞笑的"))
# 请张三讲一个搞笑的的故事

