import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from zhipuai import ZhipuAI

load_dotenv()
api_key = os.getenv("ZHIPU_API_KEY")
client = ZhipuAI(api_key=api_key)

app = FastAPI(title="DevOps Copilot")

# 定义请求体的格式:用户必须传一个叫 query 的字符串字段
class ChatRequest(BaseModel):
    query: str

@app.get("/")
def health_check():
    return {"status": "ok", "message": "DevOps Copilot 服务已启动"}

@app.post("/chat")
def chat(request: ChatRequest):
    response = client.chat.completions.create(
        model="glm-4.6",
        messages=[
            {"role": "user", "content": request.query}
        ]
    )
    answer = response.choices[0].message.content
    return {"answer": answer}