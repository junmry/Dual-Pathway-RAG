from langchain.vectorstores import Chroma
import openai
from dotenv import load_dotenv, find_dotenv
import os

import warnings
warnings.filterwarnings("ignore")

_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
os.environ["OPENAI_BASE_URL"] = 'https://key.wenwen-ai.com/v1'
# 定义 Embeddings
from embedding.call_embedding import get_embedding

embedding = get_embedding(embedding="m3e")

# 向量数据库持久化路径
persist_directory = '../../data_base/vector_db/chroma'

# 加载数据库
vectordb = Chroma(
    persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
    embedding_function=embedding
)

print(f"向量库中存储的数量：{vectordb._collection.count()}")

#初始化的 Prompt 创建一个基于模板的检索链
from langchain.prompts import PromptTemplate

template_v1 = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
{context}
问题: {question}
有用的回答:"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template_v1)

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature = 0 )

qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})


question = "强化学习的定义是什么"
result = qa_chain({"query": question})
print(result["result"])

#提升直观回答质量

template_v2 = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。你应该使答案尽可能详细具体，但不要偏题。如果答案比较长，请酌情进行分段，以提高答案的阅读体验。
{context}
问题: {question}
有用的回答:"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template_v2)
qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

question = "强化学习的定义是什么"
result = qa_chain({"query": question})
print(result["result"])

#标明知识来源，提高可信度

template_v3 = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。你应该使答案尽可能详细具体，但不要偏题。如果答案比较长，请酌情进行分段，以提高答案的阅读体验。
如果答案有几点，你应该分点标号回答，让答案清晰具体。
请你附上回答的来源原文，以保证回答的正确性。
{context}
问题: {question}
有用的回答:"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template_v3)
qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})

question = "强化学习的定义是什么"
result = qa_chain({"query": question})
print(result["result"])