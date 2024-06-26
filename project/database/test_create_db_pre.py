from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 创建一个 PyMuPDFLoader Class 实例，输入为待加载的 pdf 文档路径
loader = PyMuPDFLoader("../../data_base/knowledge_db/pumkin_book/pumpkin_book.pdf")

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
print(f"切分后的文件数量：{len(split_docs)}")

print(f"切分后的字符数：{sum([len(doc.page_content) for doc in split_docs])}")
import numpy as np
from embedding.call_embedding import get_embedding

embedding = get_embedding(embedding="m3e")

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

query1 = "土豆"
query2 = "马铃薯"
query3 = "苹果"

# 通过对应的 embedding 类生成 query 的 embedding。
emb1 = embedding.embed_query(query1)
emb2 = embedding.embed_query(query2)
emb3 = embedding.embed_query(query3)

# 将返回结果转成 numpy 的格式，便于后续计算
emb1 = np.array(emb1)
emb2 = np.array(emb2)
emb3 = np.array(emb3)

print(f"查询词 '{query1}' 生成的 embedding 长度为 {len(emb1)}, 其值为：{emb1[:]}")


print(f"查询词 '{query1}' 和 '{query2}' 的向量点积为：{np.dot(emb1, emb2)}")
print(f"查询词 '{query1}' 和 '{query3}' 的向量点积为：{np.dot(emb1, emb3)}")
print(f"查询词 '{query2}' 和 '{query3}' 的向量点积为：{np.dot(emb2, emb3)}")

cos_sim_12 = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))
cos_sim_13 = cosine_similarity(emb1.reshape(1, -1), emb3.reshape(1, -1))
cos_sim_23 = cosine_similarity(emb2.reshape(1, -1), emb3.reshape(1, -1))

print(f"查询词 '{query1}' 和 '{query2}' 的余弦相似度为：{cos_sim_12}")
print(f"查询词 '{query1}' 和 '{query3}' 的余弦相似度为：{cos_sim_13}")
print(f"查询词 '{query2}' 和 '{query3}' 的余弦相似度为：{cos_sim_23}")

