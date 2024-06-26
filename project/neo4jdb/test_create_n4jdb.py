from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from dotenv import find_dotenv, load_dotenv
import os
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
# Warning control
import warnings
warnings.filterwarnings("ignore")

_ = load_dotenv(find_dotenv())

#加载 PDF
loaders = [
    PyMuPDFLoader("../../data_base/knowledge_db/conference/test_conf.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# 切分文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)

#print(f"切分后的文件数量：{len(split_docs)}")

#print(f"切分后的字符数：{sum([len(doc.page_content) for doc in split_docs])}")

# API调用参数
api_key = os.environ["OPENAI_API_KEY"]
os.environ["OPENAI_BASE_URL"] = 'https://key.wenwen-ai.com/v1'
model = "gpt-3.5-turbo"

os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "18851806367"

graph = Neo4jGraph()
#实例化

llm = ChatOpenAI(temperature=0, openai_api_key=api_key )

llm_transformer = LLMGraphTransformer(llm=llm)

documents = split_docs

graph_documents_filtered = llm_transformer.convert_to_graph_documents(documents)
print(f"Nodes:{graph_documents_filtered[0].nodes}")
print(f"Relationships:{graph_documents_filtered[0].relationships}")

graph.add_graph_documents(graph_documents_filtered)

graph.refresh_schema()
