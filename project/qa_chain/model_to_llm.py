from llm.wenxin_llm import Wenxin_LLM
from llm.call_llm import parse_llm_api_key
from langchain.chat_models import ChatOpenAI
import os
os.environ["OPENAI_BASE_URL"] = 'https://key.wenwen-ai.com/v1'

def model_to_llm(model:str=None, temperature:float=0.0, api_key:str=None, Wenxin_secret_key:str=None):
        """
        百度：model,temperature,api_key,api_secret
        OpenAI：model,temperature,api_key
        """
        if model in ["gpt-3.5-turbo"]:
            if api_key == None:
                api_key = parse_llm_api_key("openai")
            llm = ChatOpenAI(model_name = model, temperature = temperature , openai_api_key = api_key)
        elif model in ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"]:
            if api_key == None or Wenxin_secret_key == None:
                api_key, Wenxin_secret_key = parse_llm_api_key("wenxin")
            llm = Wenxin_LLM(model=model, temperature = temperature, api_key=api_key, secret_key=Wenxin_secret_key)
        else:
            raise ValueError(f"model{model} not support!!!")
        return llm