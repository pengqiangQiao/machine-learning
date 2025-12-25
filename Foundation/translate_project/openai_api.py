import os
from openai import OpenAI
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

gpt_api_key = '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'  # GPT
gpt_base_url = "https://api.openai.com/v1"  # GPT API请求地址
def run(key_, model_name="gpt-3.5-turbo", messages=None):
    client = OpenAI(
        base_url=gpt_base_url,
        api_key=key_,
    )
    if not messages or messages == []:
        messages = [{"role": "user", "content": "你好"}]
    res = client.chat.completions.create(
        messages=messages,
        model=model_name
    )
    return res

def get_res(query, model_name="gpt-3.5-turbo", history=None):
    if not history or history == []:
        history = [{"role": "system", "content": "You are a helpful assistant."}]
    history.append({"role": "user", "content": query})
    response_ = run(gpt_api_key, model_name=model_name, messages=history)
    # print(response_.choices[0].message.content)
    # print(type(response_.choices[0].message.content))
    return response_


if __name__ == '__main__':
    query = "树上有7只鸟,地上1只鸟,我开枪打死7只,树上还有几只？"
    message = [{"role": "system", "content": "You are a helpful assistant."}]
    print((get_res(query, "gpt-3.5-turbo",message)).choices[0].message.content)