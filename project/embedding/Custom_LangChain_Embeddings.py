from typing import List
from langchain.embeddings.base import Embeddings

class CustomLangChainEmbeddings(Embeddings):
    def __init__(self, model_name='Jerry0/m3e-base'):
        super().__init__()
        self.model_name = model_name
        self.model_dir = None  # 模型目录，可以在此处设置或在 _load_model 方法中设置
        self.model = None  # 在 _load_model 方法中加载模型

    def _load_model(self):
        """
        加载 SentenceTransformer 模型。
        """
        from sentence_transformers import SentenceTransformer
        from modelscope import snapshot_download

        self.model_dir = snapshot_download(self.model_name)
        self.model = SentenceTransformer(self.model_dir)

    def _embed(self, text: str) -> List[float]:
        """
        嵌入单个字符串。

        Args:
        text (str): 要嵌入的输入字符串。

        Returns:
        List[float]: 输入字符串的嵌入。
        """
        if self.model is None:
            self._load_model()
        return self.model.encode([text])[0].tolist()  # 转换为列表

    def embed_query(self, query: str) -> List[float]:
        """
        嵌入单个查询字符串。

        Args:
        query (str): 要嵌入的查询字符串。

        Returns:
        List[float]: 查询字符串的嵌入。
        """
        return self._embed(query)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        嵌入字符串文档列表。

        Args:
        documents (List[str]): 要嵌入的文档字符串列表。

        Returns:
        List[List[float]]: 每个文档的嵌入列表。
        """
        if self.model is None:
            self._load_model()
        return [self._embed(doc) for doc in documents]


# 示例用法：
if __name__ == "__main__":
    # 创建 CustomLangChainEmbeddings 的实例
    custom_embeddings = CustomLangChainEmbeddings()

    # 要嵌入的句子
    sentences = [
        '* Moka 此文本嵌入模型由 MokaAI 训练并开源，训练脚本使用 uniem',
        '* Massive 此文本嵌入模型通过**千万级**的中文句对数据集进行训练',
        '* Mixed 此文本嵌入模型支持中英双语的同质文本相似度计算，异质文本检索等功能，未来还会支持代码检索，ALL in one'
    ]

    # 嵌入查询
    query = '这个模型的性能如何？'
    query_embedding = custom_embeddings.embed_query(query)
    print("查询嵌入:", query_embedding)

    # 嵌入文档
    documents_embeddings = custom_embeddings.embed_documents(sentences)
    print("文档嵌入:")
    for i, embedding in enumerate(documents_embeddings):
        print("句子", i + 1, "嵌入:", embedding)

