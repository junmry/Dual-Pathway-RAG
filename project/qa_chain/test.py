from qa_chain.ChatQuestionChainSelf import ChatQuestionChainSelf #带历史记录的问答链
from qa_chain.questionAnswerChainSelf import questionAnswerChainSelf       #不带历史记录的问答链
from llm.call_llm import parse_llm_api_key
model:str = "gpt-3.5-turbo"
temperature:float=0.0
top_k:int=4
chat_history:list=[]
file_path:str = "../../data_base/knowledge_db"
persist_path:str = "../../data_base/vector_db/chroma"
api_key = parse_llm_api_key("openai")
api_secret:str=None
embedding = "m3e"

# 不带历史记录的问答链
qa_chain = questionAnswerChainSelf(model=model, temperature=temperature, top_k=top_k, file_path=file_path, persist_path=persist_path, api_key=api_key, embedding = embedding)
#print(qa_chain)

question = "李白是谁？"
answer = qa_chain.answer(question)
print(answer)
#answer_h = qa_chain_h.answer(question)
#print(answer_h)





