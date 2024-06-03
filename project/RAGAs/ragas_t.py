from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#加载 PDF
loaders_chinese = [
    PyMuPDFLoader("../../data_base/knowledge_db/conference/test_conf.pdf")
]
docs = []
for loader in loaders_chinese:
    docs.extend(loader.load())

# 切分文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)
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
#   #加载数据库
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

