from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdf 文档路径
loader = PyMuPDFLoader("../../data_base/knowledge_db/小程序工程师.pdf")

# 调用 PyMuPDFLoader Class 的函数 load 对 pdf 文件进行加载
pages = loader.load()

page = pages[1]
# 知识库中单段文本长度
CHUNK_SIZE = 500
# 知识库中相邻文本重合长度
OVERLAP_SIZE = 50

# 使用递归字符文本分割器

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP_SIZE
)
text_splitter.split_text(page.page_content[0:1000])

split_docs = text_splitter.split_documents(pages)

# 定义 Embeddings
from embedding.call_embedding import get_embedding
embeddings = get_embedding(embedding="m3e")
persist_directory = '../../data_base/vector_db/chroma'
from langchain.vectorstores import Chroma
#构建向量数据库
vectordb = Chroma.from_documents(
       documents=split_docs[:1000],
       embedding=embeddings,
      persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
)
vectordb.persist()#持久化
#加载数据库
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

print(f"向量库中存储的数量：{vectordb._collection.count()}")
##
question="微信小程序是什么？"

sim_docs = vectordb.similarity_search(question,k=5)
for i, sim_doc in enumerate(sim_docs):
     print(f"检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")

mmr_docs = vectordb.max_marginal_relevance_search(question,k=3)
for i, sim_doc in enumerate(mmr_docs):
    print(f"MMR 检索到的第{i}个内容: \n{sim_doc.page_content[:200]}", end="\n--------------\n")


# final_content=""
#
# for i, sim_doc in enumerate(sim_docs):
#     final_content += sim_doc.page_content
#
# from openai import OpenAI
# from dotenv import find_dotenv, load_dotenv
# import os
# _ = load_dotenv(find_dotenv())
# api_key = os.environ["OPENAI_API_KEY"]
# os.environ["OPENAI_BASE_URL"] = 'https://key.wenwen-ai.com/v1'
#
# def kg_gpt(question, api_key, sim_docs):
#
#     openai_client = OpenAI(api_key=api_key)
#
#     message = [
#         {"role": "system", "content": "That's the problem " + question},
#         {"role": "user", "content": "Please answer the questions based on the following searches,A short and precise"},
#         {"role": "user", "content": sim_docs}
#     ]
#
#     completion = openai_client.chat.completions.create(
#         model="gpt-4",
#         messages=message,
#         temperature=0,
#     )
#
#     # 调用 OpenAI 的 ChatCompletion 接口
#     return completion.choices[0].message.content
#
# result = kg_gpt(question, api_key, final_content)
# print(result)
