from fastapi import FastAPI
from pydantic import BaseModel
from app.core.config import client, DEFAULT_MODEL

app = FastAPI(title="DevOps Copilot")

class ChatRequest(BaseModel):
    query: str

@app.get("/")
def health_check():
    return {"status": "ok", "message": "DevOps Copilot 服务已启动"}

@app.post("/chat")
def chat(request: ChatRequest):
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "user", "content": request.query}
        ]
    )
    answer = response.choices[0].message.content
    return {"answer": answer}