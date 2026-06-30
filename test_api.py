import os
from dotenv import load_dotenv
from zhipuai import ZhipuAI

# 从 .env 文件读取你的API密钥
load_dotenv()
api_key = os.getenv("ZHIPU_API_KEY")

# 初始化客户端
client = ZhipuAI(api_key=api_key)

# 发一条最简单的对话请求
response = client.chat.completions.create(
    model="glm-5.2",
    messages=[
        {"role": "user", "content": "用一句话介绍一下你自己，你是什么模型型号glm-5.2吗"}
    ]
)

print(response.choices[0].message.content)