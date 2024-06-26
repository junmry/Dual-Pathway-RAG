import numpy as np
from openai import OpenAI
from scipy.spatial.distance import cosine
import numpy as np

from dotenv import find_dotenv, load_dotenv
import os
_ = load_dotenv(find_dotenv())
api_key = os.environ["OPENAI_API_KEY"]


def evaluate_faithfulness(question, context, answer):
    """评估答案对给定上下文的忠实度"""
    # 从答案中提取陈述句
    api_base = 'https://key.wenwen-ai.com/v1'
    openai_client = OpenAI(api_key=api_key, base_url=api_base)
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"给定一个问题和答案,从答案中的每个句子创建一个或多个陈述。\n问题: {question}\n答案: {answer}"}
        ],
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    statements = response.choices[0].message.content.split("\n")
    # 验证每个陈述是否能从上下文中推导得出
    supported_statements = 0
    for statement in statements:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"考虑给定的上下文和以下陈述,判断该陈述是否能从上下文中推导得出。在做出判断(Yes/No)之前,请简要解释一下该陈述。\n上下文: {context}\n陈述: {statement}"}
            ],
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        verdict = response.choices[0].message.content.strip().split("\n")[-1]
        if verdict == "Yes":
            supported_statements += 1

    # 计算忠实度得分
    faithfulness_score = supported_statements / len(statements)
    return faithfulness_score

import numpy as np
import openai

# 设定OpenAI API密钥
openai.api_key = ''



# 生成潜在问题函数
def generate_potential_questions(answer, n=5):
    # 使用GPT模型生成基于答案的潜在问题
    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt = f"生成潜在问题：\n答案：'{answer}'\n生成{n}个潜在问题：\n1.",
        temperature=0.7,
        max_tokens=50 * n,
        n=n,
        stop="\n"
    )
    potential_questions = [choice['text'].strip() for choice in response['choices']]
    return potential_questions

from embedding.Custom_LangChain_Embeddings import CustomLangChainEmbeddings
custom_embeddings = CustomLangChainEmbeddings()

# 计算相似度函数（使用自定义文本嵌入模型）
def calculate_similarity(original_question, potential_questions):
    similarities = []
    # 原始问题的嵌入
    original_embedding = custom_embeddings.embed_query(original_question)

    for question in potential_questions:
        # 潜在问题的嵌入
        potential_embedding = custom_embeddings.embed_query(question)

        # 计算余弦相似度
        similarity = np.dot(original_embedding, potential_embedding) / (
                    np.linalg.norm(original_embedding) * np.linalg.norm(potential_embedding))
        similarities.append(similarity)
    return similarities


# 计算答案相关性得分函数（保持不变）
def compute_answer_relevance(original_question, answer, n=5):
    # 生成潜在问题
    potential_questions = generate_potential_questions(answer, n)

    # 计算相似度
    similarities = calculate_similarity(original_question, potential_questions)

    # 计算答案相关性得分
    answer_relevance_score = sum(similarities) / len(similarities)
    return answer_relevance_score


# 示例使用（保持不变）
original_question = "What is the capital of France?"
answer = "Paris"
answer_relevance_score = compute_answer_relevance(original_question, answer)
print("Answer Relevance Score:", answer_relevance_score)


def evaluate_context_relevance(question, context):
    """评估上下文对给定问题的相关性"""
    # 从上下文中提取相关句子
    api_base = 'https://key.wenwen-ai.com/v1'
    openai_client = OpenAI(api_key=api_key, base_url=api_base)
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"请从提供的上下文中提取可能有助于回答以下问题的相关句子。如果没有找到相关句子,或者认为该问题无法从给定的上下文中回答,请返回'Insufficient Information'。在提取候选句子时,不允许对上下文中的句子进行任何修改。\n问题: {question}\n上下文: {context}"}
        ],
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    relevant_sentences = response.choices[0].message.content.split("\n")

    if relevant_sentences[0] == "Insufficient Information":
        return 0

    # 计算上下文相关性得分
    context_relevance_score = len(relevant_sentences) / len(context.split("\n"))
    return context_relevance_score

question = "When was the Chimnabai Clock Tower completed, and who was it named after?"
context = """The Chimnabai Clock Tower, also known as the Raopura Tower, is a clock tower situated in the Raopura area of Vadodara, Gujarat, India. It was completed in 1896 and named in memory of Chimnabai I (1864–1885), a queen and the first wife of Sayajirao Gaekwad III of Baroda State. It was built in Indo-Saracenic architecture style."""
answer = "The Chimnabai Clock Tower was completed in 1896 and named after Chimnabai I, a queen and the first wife of Sayajirao Gaekwad III of Baroda State."

faithfulness_score = evaluate_faithfulness(question, context, answer)
context_relevance_score = evaluate_context_relevance(question, context)

print(f"Faithfulness score: {faithfulness_score:.2f}")
print(f"Context relevance score: {context_relevance_score:.2f}")