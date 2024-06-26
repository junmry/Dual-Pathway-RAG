# 导入必要的库

import sys
import os                # 用于操作系统相关的操作，例如读取环境变量

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import io                # 用于处理流式数据（例如文件流）
import gradio as gr
from dotenv import load_dotenv, find_dotenv
from llm.call_llm import get_completion
from database.create_db import createDatabaseInfo
from qa_chain.ChatQuestionChainSelf import ChatQuestionChainSelf
from qa_chain.questionAnswerChainSelf import questionAnswerChainSelf

# 导入 dotenv 库的函数
# dotenv 允许您从 .env 文件中读取环境变量
# 这在开发时特别有用，可以避免将敏感信息（如API密钥）硬编码到代码中

# 寻找 .env 文件并加载它的内容
# 这允许您使用 os.environ 来读取在 .env 文件中设置的环境变量
_ = load_dotenv(find_dotenv())
LLM_MODEL_DICT = {
    "openai": ["gpt-3.5-turbo", "gpt-4"],
    "wenxin": ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"],
}

INIT_LLM = 'gpt-3.5-turbo'
LLM_MODEL_LIST = sum(list(LLM_MODEL_DICT.values()),[])
EMBEDDING_MODEL_LIST = ['m3e']
INIT_EMBEDDING_MODEL = "m3e"
DEFAULT_DB_PATH = "../../data_base/knowledge_db"
DEFAULT_PERSIST_PATH = "../../data_base/vector_db/chroma"



def get_model_by_platform(platform):
    return LLM_MODEL_DICT.get(platform, "")

class Model_center():
    """
    存储问答 Chain 的对象
    - ChatQuestionChainSelf: 带历史记录的问答链。
    - qa_chain_self: 不带历史记录的问答链。
    """
    def __init__(self):
        self.chat_qa_chain_self = {}
        self.qa_chain_self = {}

    def ChatQuestionChainSelf(self, question: str, chat_history: list = [], model: str = "gpt-3.5-turbo", embedding: str = "m3e", temperature: float = 0.0, top_k: int = 4, file_path: str = DEFAULT_DB_PATH, persist_path: str = DEFAULT_PERSIST_PATH):
        """
        调用带历史记录的问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            if (model, embedding) not in self.chat_qa_chain_self:
                self.chat_qa_chain_self[(model, embedding)] = ChatQuestionChainSelf(model=model, temperature=temperature,
                                                                                    top_k=top_k, chat_history=chat_history, file_path=file_path, persist_path=persist_path, embedding=embedding)
            chain = self.chat_qa_chain_self[(model, embedding)]
            return "", chain.answer(question=question, temperature=temperature, top_k=top_k)
        except Exception as e:
            return e, chat_history

    def questionAnswerChainSelf(self, question: str, chat_history: list = [], model: str = "gpt-3.5-turbo", embedding="m3e", temperature: float = 0.0, top_k: int = 4, file_path: str = DEFAULT_DB_PATH, persist_path: str = DEFAULT_PERSIST_PATH):
        """
        调用不带历史记录的问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            if (model, embedding) not in self.qa_chain_self:
                self.qa_chain_self[(model, embedding)] = questionAnswerChainSelf(model=model, temperature=temperature,
                                                                                 top_k=top_k, file_path=file_path, persist_path=persist_path, embedding=embedding)
            chain = self.qa_chain_self[(model, embedding)]
            chat_history.append(
                (question, chain.answer(question, temperature, top_k)))
            return "", chat_history
        except Exception as e:
            return e, chat_history

    def clear_history(self):
        if len(self.chat_qa_chain_self) > 0:
            for chain in self.chat_qa_chain_self.values():
                chain.clear_history()


def format_chat_prompt(message, chat_history):
    """
    该函数用于格式化聊天 prompt。

    参数:
    message: 当前的用户消息。
    chat_history: 聊天历史记录。

    返回:
    prompt: 格式化后的 prompt。
    """
    # 初始化一个空字符串，用于存放格式化后的聊天 prompt。
    prompt = ""
    # 遍历聊天历史记录。
    for turn in chat_history:
        # 从聊天记录中提取用户和机器人的消息。
        user_message, bot_message = turn
        # 更新 prompt，加入用户和机器人的消息。
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    # 将当前的用户消息也加入到 prompt中，并预留一个位置给机器人的回复。
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    # 返回格式化后的 prompt。
    return prompt



def respond(message, chat_history, llm, history_len=3, temperature=0.1, max_tokens=2048):
    """
    该函数用于生成机器人的回复。
    参数:
    message: 当前的用户消息。
    chat_history: 聊天历史记录。
    返回:
    "": 空字符串表示没有内容需要显示在界面上，可以替换为真正的机器人回复。
    chat_history: 更新后的聊天历史记录
    """
    if message == None or len(message) < 1:
            return "", chat_history
    try:
        # 限制 history 的记忆长度
        chat_history = chat_history[-history_len:] if history_len > 0 else []
        # 调用上面的函数，将用户的消息和聊天历史记录格式化为一个 prompt。
        formatted_prompt = format_chat_prompt(message, chat_history)
        # 使用llm对象的predict方法生成机器人的回复（注意：llm对象在此代码中并未定义）。
        bot_message = get_completion(
            formatted_prompt, llm, temperature=temperature, max_tokens=max_tokens)
        # 将用户的消息和机器人的回复加入到聊天历史记录中。
        chat_history.append((message, bot_message))
        # 返回一个空字符串和更新后的聊天历史记录（这里的空字符串可以替换为真正的机器人回复，如果需要显示在界面上）。
        return "", chat_history
    except Exception as e:
        return e, chat_history


model_center = Model_center()

# 使用Gradio Blocks和主题设置
with gr.Blocks(theme=gr.themes.Default(primary_hue="red", secondary_hue="pink")) as demo:
    # 头部区域
    with gr.Row():
        gr.Markdown("""<h1><center>展示DEMO</center></h1>
            <center>多源检索</center>
            """)

    # 主要内容区域
    with gr.Row():
        # 左侧列(聊天机器人和输入框)
        with gr.Column():
            chatbot = gr.Chatbot(height=400, show_copy_button=True, show_share_button=True)
            msg = gr.Textbox(label="提示/问题")

            # 按钮行
            with gr.Row():
                db_with_his_btn = gr.Button("带历史记录检索知识库")
                db_wo_his_btn = gr.Button("不带历史记录检索知识库")
                llm_btn = gr.Button("只与大语言模型对话")

            # 清除按钮
            clear = gr.ClearButton(components=[chatbot], value="清除对话记录")

        # 右侧列(文件上传、初始化数据库和调参部分)
        with gr.Column():
            # 文件上传区域
            file = gr.File(label='选择知识库文件夹',
                           file_count='directory',
                           file_types=['.txt', '.md', '.docx', '.pdf'])

            # 初始化数据库按钮
            init_db = gr.Button("知识库文件向量化")

            # 调参部分
            with gr.Accordion("参数设置", open=False):
                with gr.Row():
                    temperature = gr.Slider(0, 1,     # 创建一个滑块组件,范围从0到1
                                            value=0.01,  # 设置默认值为0.01
                                            step=0.01,  # 设置滑动步长为0.01
                                            label="语言模型温度",
                                            interactive=True)# 设置滑块为交互式的
                    top_k = gr.Slider(1, 10,
                                      value=3,
                                      step=1,
                                      label="向量数据库搜索top k",
                                      interactive=True)
                    history_len = gr.Slider(0, 5,
                                            value=3,
                                            step=1,
                                            label="历史记录长度",
                                            interactive=True)

            with gr.Accordion("模型选择"):
                with gr.Row():

                    llm = gr.Dropdown(LLM_MODEL_LIST,
                                      label="大语言模型",
                                      value=INIT_LLM,
                                      interactive=True)

                    embeddings = gr.Dropdown(EMBEDDING_MODEL_LIST,
                                             label="嵌入模型",
                                             value=INIT_EMBEDDING_MODEL)

    # 事件处理
    init_db.click(
        fn=createDatabaseInfo,
        inputs=[file, embeddings],
        outputs=[msg])
    db_with_his_btn.click(
        fn=model_center.ChatQuestionChainSelf,
        inputs=[msg, chatbot, llm, embeddings, temperature, top_k, history_len],
        outputs=[msg, chatbot])
    db_wo_his_btn.click(
        fn=model_center.questionAnswerChainSelf,
        inputs=[msg, chatbot, llm, embeddings, temperature, top_k],
        outputs=[msg, chatbot])
    llm_btn.click(
        fn=respond,
        inputs=[msg, chatbot, llm, history_len, temperature], outputs=[msg, chatbot],
        show_progress="minimal")
    msg.submit(
        fn=respond,
        inputs=[msg, chatbot, llm, history_len, temperature],
        outputs=[msg, chatbot], show_progress="hidden")
    clear.click(fn=model_center.clear_history)



# threads to consume the request
gr.close_all()
# 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
# demo.launch(share=True, server_port=int(os.environ['PORT1']))
# 直接启动
demo.launch()
