from fastapi import FastAPI
from pydantic import BaseModel
from app.agents.planner import planner_answer

app = FastAPI(title="DevOps Copilot")

class ChatRequest(BaseModel):
    query: str

@app.get("/")
def health_check():
    return {"status": "ok", "message": "DevOps Copilot 服务已启动"}

@app.post("/chat")
def chat(request: ChatRequest):
    result = planner_answer(request.query)
    return result