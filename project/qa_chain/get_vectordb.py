import os
from database.create_db import createDatabase,loadKnowledgeDatabase
from embedding.call_embedding import get_embedding

def get_vectordb(file_path:str=None, persist_path:str=None, embedding = "m3e"):

    embedding = get_embedding(embedding=embedding)
    if os.path.exists(persist_path):  #持久化目录存在
        contents = os.listdir(persist_path)
        if len(contents) == 0:  #但是下面为空
            #print("目录为空")
            vectordb = createDatabase(file_path, persist_path, embedding)
            #presit_knowledge_db(vectordb)
            vectordb = loadKnowledgeDatabase(persist_path, embedding)
        else:
            #print("目录不为空")
            vectordb = loadKnowledgeDatabase(persist_path, embedding)
    else: #目录不存在，从头开始创建向量数据库
        vectordb = createDatabase(file_path, persist_path, embedding)
        #presit_knowledge_db(vectordb)
        vectordb = loadKnowledgeDatabase(persist_path, embedding)

    return vectordb