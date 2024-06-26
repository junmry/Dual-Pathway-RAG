from embedding.Custom_LangChain_Embeddings import CustomLangChainEmbeddings

def get_embedding(embedding: str):

    if embedding == 'm3e':
        return  CustomLangChainEmbeddings()
    else:
        raise ValueError(f"embedding {embedding} not support ")
