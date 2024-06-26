from langchain.chains import ConversationalRetrievalChain

from qa_chain.model_to_llm import model_to_llm
from qa_chain.get_vectordb import get_vectordb


class ChatQuestionChainSelf:

    def __init__(self,model:str, temperature:float=0.0, top_k:int=4, chat_history:list=[], file_path:str=None, persist_path:str=None,api_key:str=None,Wenxin_secret_key:str=None, embedding = "m3e"):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.chat_history = chat_history
        #self.history_len = history_len
        self.file_path = file_path
        self.persist_path = persist_path
        self.api_key = api_key
        self.Wenxin_secret_key = Wenxin_secret_key
        self.embedding = embedding
        self.vectordb = get_vectordb(self.file_path, self.persist_path, self.embedding)
        
    
    def clear_history(self):
        "清空历史记录"
        return self.chat_history.clear()

    
    def change_history_length(self,history_len:int=1):

        n = len(self.chat_history)
        return self.chat_history[n-history_len:]

 
    def answer(self, question:str=None,temperature = None, top_k = 4):
        
        if len(question) == 0:
            return "", self.chat_history
        
        if len(question) == 0:
            return ""
        
        if temperature == None:
            temperature = self.temperature

        llm = model_to_llm(self.model, temperature, self.api_key, self.Wenxin_secret_key)

        #self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        retriever = self.vectordb.as_retriever(search_type="similarity",
                                        search_kwargs={'k': top_k})  #默认similarity，k=4

        qa = ConversationalRetrievalChain.from_llm(
            llm = llm,
            retriever = retriever,
            #memory = self.memory,
            return_source_documents=True
        )

        result = qa({"question": question,"chat_history": self.chat_history})       #result里有question、chat_history、answer
        answer =  result['answer']
        self.chat_history.append((question,answer)) #更新历史记录

        return self.chat_history  #返回本次回答和更新后的历史记录



    
        
        
















