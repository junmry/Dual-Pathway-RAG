import os
import sys

# 将当前文件的上一级目录添加到系统路径中，以便于导入上层目录的模块
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 导入临时文件处理模块、文档加载器模块、文本分割器模块、PDF文档加载器模块以及向量数据库模块
import tempfile
from embedding.call_embedding import get_embedding
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import Chroma

# 定义默认的数据库路径和持久化路径
DEFAULT_DB_PATH = "../../data_base/knowledge_db"
DEFAULT_PERSIST_PATH = "../database/vector_data_base"

# 定义一个函数，用于获取指定目录下的所有文件
def getFiles(dir_path):
    # 初始化一个空列表来存储文件路径
    file_list = []
    # 遍历指定目录下的所有文件和子目录
    for filepath, dirnames, filenames in os.walk(dir_path):
        # 遍历当前目录下的所有文件
        for filename in filenames:
            # 将文件的完整路径添加到列表中
            file_list.append(os.path.join(filepath, filename))
    # 返回包含所有文件路径的列表
    return file_list

# 定义一个函数，用于根据文件类型加载不同的文档加载器
def fileLoader(file, loaders):
    # 如果传入的是一个临时文件对象，则获取其文件名
    if isinstance(file, tempfile._TemporaryFileWrapper):
        file = file.name
    # 如果传入的不是一个文件（可能是一个目录），则遍历该目录下的所有文件，并递归调用fileLoader函数
    if not os.path.isfile(file):
        [fileLoader(os.path.join(file, f), loaders) for f in os.listdir(file)]
        return
    # 获取文件的扩展名，以确定文件类型
    file_type = file.split('.')[-1]
    # 根据文件类型添加相应的文档加载器到loaders列表中
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file))
    elif file_type == 'md':
        loaders.append(UnstructuredMarkdownLoader(file))
    elif file_type == 'txt':
        loaders.append(UnstructuredFileLoader(file))
    # 返回loaders列表
    return

# 定义一个函数，用于创建数据库信息
def createDatabaseInfo(files=DEFAULT_DB_PATH, embeddings="m3e", persist_directory=DEFAULT_PERSIST_PATH):
    # 调用createDatabase函数创建数据库，并返回创建的数据库对象
    vectordb = createDatabase(files, persist_directory, embeddings)
    return ""

# 定义一个函数，用于创建数据库并返回数据库对象
def createDatabase(files=DEFAULT_DB_PATH, persist_directory=DEFAULT_PERSIST_PATH, embeddings="m3e"):
    # 如果传入的文件列表为空，则返回错误信息
    if files == None:
        return "can't load empty file"
    # 如果传入的不是列表类型，则将其转换为列表
    if type(files) != list:
        files = [files]
    # 初始化一个空列表来存储文档加载器
    loaders = []
    # 遍历所有文件，并调用fileLoader函数加载文档
    [fileLoader(file, loaders) for file in files]
    # 初始化一个空列表来存储加载的文档
    docs = []
    # 遍历所有文档加载器，并将加载的文档添加到docs列表中
    for loader in loaders:
        if loader is not None:
            docs.extend(loader.load())
    # 使用文本分割器对文档进行切分
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=150)
    # 对前10个文档进行切分，并获取切分后的文档列表
    split_docs = text_splitter.split_documents(docs[:10])
    # 定义持久化路径
    # persist_directory = '../../data_base/vector_db/chroma'
    # 如果传入的embeddings是字符串类型，则获取相应的嵌入函数
    if type(embeddings) == str:
        embeddings = get_embedding(embedding=embeddings)
    # 使用Chroma模块从文档创建向量数据库，并指定嵌入函数和持久化路径
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory  # 指定向量数据库的存储路径
    )
    # 将向量数据库持久化到磁盘
    vectordb.persist()
    # 返回创建的向量数据库对象
    return vectordb

# 定义一个函数，用于持久化知识数据库
def presitKnowledgeDatabase(vectordb):
    # 调用向量数据库的persist方法进行持久化
    vectordb.persist()

# 定义一个函数，用于加载知识数据库
def loadKnowledgeDatabase(path, embeddings):
    # 使用Chroma模块加载持久化后的向量数据库，并指定持久化路径和嵌入函数
    vectordb = Chroma(
        persist_directory=path,
        embedding_function=embeddings
    )
    # 返回加载的向量数据库对象
    return vectordb


