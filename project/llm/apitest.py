from openai import OpenAI

def get_completion_gpt(api_key, prompt, model, temperature=0):
    api_base = 'https://key.wenwen-ai.com/v1'
    openai_client = OpenAI(api_key=api_key, base_url=api_base)

    '''
    prompt: 对应的提示词
    model: 调用的模型，默认为 gpt-3.5-turbo(ChatGPT)
    '''
    message = [
        {"role": "user", "content": prompt}
    ]

    completion = openai_client.chat.completions.create(
        model=model,
        messages=message,
        temperature=temperature
    )

    # 调用 OpenAI 的 ChatCompletion 接口
    return completion.choices[0].message.content

# 使用示例
api_key = "sk-CQabJqvQqI4kR30DCb337e880cD44627Ac689f533d5fC824"
prompt = "Hello!"
model = "gpt-3.5-turbo"

result = get_completion_gpt(api_key, prompt, model)
print(result)
