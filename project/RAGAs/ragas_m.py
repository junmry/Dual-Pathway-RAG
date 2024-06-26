from dotenv import find_dotenv, load_dotenv
import os
_ = load_dotenv(find_dotenv())
api_key = os.environ["OPENAI_API_KEY"]
os.environ["OPENAI_BASE_URL"] = 'https://key.wenwen-ai.com/v1'

#----------------------------------------------------------------

# 定义 Embeddings
from embedding.call_embedding import get_embedding

embeddings = get_embedding(embedding="m3e")

persist_directory = '../../data_base/vector_db/chroma'

from langchain.vectorstores import Chroma

#   #加载数据库
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)


##rag+kg
from openai import OpenAI

def rag_kg_gpt(question, api_key, g_result, sim_docs):

     openai_client = OpenAI(api_key=api_key)

     message = [
         {"role": "system", "content": "That's the problem "+ question},
         {"role": "user", "content": "Please answer the questions based on the following searches.A short and precise overview"},
         {"role": "user", "content": g_result},#使用参数从 Cypher QA 链返回中间步骤,作为提示词
         {"role": "user", "content": sim_docs}
     ]

     completion = openai_client.chat.completions.create(
         model="gpt-4",
         messages=message,
         temperature=0,
     )

     # 调用 OpenAI 的 ChatCompletion 接口
     return completion.choices[0].message.content


##test
g_result = "In Ms. Mercier's opinion, AI technologies can play a key role in detecting and predicting new cybersecurity threats, enabling automated response and remediation, and preventing future attacks through continuous learning."
question = "In Ms. Mercier's opinion, what key role can AI technologies play in responding to new cybersecurity threats?"
final_content = "In her opinion, Ms. Mercier believes that AI technologies can play a crucial role in responding to new cybersecurity threats by proactively identifying and neutralizing them before they can cause harm. AI can analyze patterns and predict potential threats faster than traditional methods, allowing for a more efficient and dynamic defense mechanism against evolving cyber threats."
sim_docs = vectordb.similarity_search(question, k=2)
for i, sim_doc in enumerate(sim_docs):
    final_content += sim_doc.page_content

result = rag_kg_gpt(question, api_key,g_result,final_content)
print(result)
#print(result)