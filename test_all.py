from llm.call_llm import get_completion
from dotenv import find_dotenv, load_dotenv
import os

# 读取本地/项目的环境变量。

# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中
_ = load_dotenv(find_dotenv())

# 获取环境变量 OPENAI_API_KEY
openai_api_key = os.environ["OPENAI_API_KEY"]

#wenxin_api_key = os.environ["wenxin_api_key"]
#wenxin_secret_key = os.environ["wenxin_secret_key"]

print(get_completion(prompt="你是谁", model="gpt-3.5-turbo"))

#get_completion("你是谁",model="ERNIE-Bot-turbo", api_key=wenxin_api_key, secret_key=wenxin_secret_key)

