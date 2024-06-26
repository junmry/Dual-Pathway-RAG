from openai import OpenAI
import json
import requests
from dotenv import load_dotenv, find_dotenv
import os

def get_completion(prompt :str, model :str, temperature=0.1,api_key=None, secret_key=None, max_tokens=2048):
    # 调用大模型获取回复，支持gpt和文心一言
    # arguments:
    # prompt: 输入提示
    # model：模型名
    # temperature: 温度系数
    # api_key：如名
    # secret_key, access_token：调用文心系列模型需要
    # return: 模型返回，字符串
    if model in ["gpt-3.5-turbo"]:
        return get_completion_gpt(prompt, api_key, model, temperature)
    elif model in ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"]:
        return get_completion_wenxin(prompt, model, temperature, api_key, secret_key, max_tokens)
    else:
        return "不正确的模型"


def get_completion_gpt(prompt: str, api_key: str, model: str, temperature: float, ):
    if api_key == None:
        api_key = parse_llm_api_key("openai")
    api_base = 'https://key.wenwen-ai.com/v1'
    openai_client = OpenAI(api_key=api_key, base_url=api_base)
    '''
    prompt: 对应的提示词
    model: 调用的模型，默认为 gpt-3.5-turbo(ChatGPT)
    '''
    message = [
        {"role": "user", "content": prompt}
    ]
    # 调用 OpenAI 的 ChatCompletion 接口
    completion = openai_client.chat.completions.create(
        model=model,
        messages=message,
        temperature=temperature
    )

    return completion.choices[0].message.content


def get_access_token(api_key:str, secret_key : str):
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """
    # 指定网址
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
    # 设置 POST 访问
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    # 通过 POST 访问获取账户对应的 access_token
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")

def get_completion_wenxin(prompt : str, model : str, temperature : float, api_key:str, secret_key : str):
    # 封装百度文心原生接口
    if api_key == None or secret_key == None:
        api_key, secret_key = parse_llm_api_key("wenxin")
    # 获取access_token
    access_token = get_access_token(api_key, secret_key)
    # 调用接口
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
    # 配置 POST 参数
    payload = json.dumps({
        "messages": [
            {
                "role": "user",# user prompt
                "content": "{}".format(prompt)# 输入的 prompt
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }
    # 发起请求
    response = requests.request("POST", url, headers=headers, data=payload)
    # 返回的是一个 Json 字符串
    js = json.loads(response.text)
    return js["result"]

def parse_llm_api_key(model:str, env_file:dict()=None):
    """
    通过 model 和 env_file 的来解析平台参数
    """   
    if env_file == None:
        _ = load_dotenv(find_dotenv())
        env_file = os.environ
    if model == "openai":
        return env_file["OPENAI_API_KEY"]
    elif model == "wenxin":
        return env_file["wenxin_api_key"], env_file["wenxin_secret_key"]
    else:
        raise ValueError(f"model{model} not support!!!")
