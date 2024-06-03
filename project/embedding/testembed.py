from sentence_transformers import SentenceTransformer
from modelscope import snapshot_download
import numpy as np
model_dir = snapshot_download('Jerry0/m3e-base')

model = SentenceTransformer(model_dir)


sentences = [
    '* Moka 此文本嵌入模型由 MokaAI 训练并开源，训练脚本使用 uniem',
    '* Massive 此文本嵌入模型通过**千万级**的中文句对数据集进行训练',
    '* Mixed 此文本嵌入模型支持中英双语的同质文本相似度计算，异质文本检索等功能，未来还会支持代码检索，ALL in one'
]

embeddings = model.encode(sentences)
embeddings_l2_norm = np.linalg.norm(embeddings, axis=1)

print(embeddings_l2_norm)



