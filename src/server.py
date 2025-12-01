from fastapi import FastAPI
from pydantic import BaseModel
from main import build_app
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage

from datetime import datetime
from pathlib import Path
import json
import uuid

from utils import extract_event

# LOG PATH
LOG_DIR = Path("messages")
LOG_FILE = LOG_DIR / "logs.txt"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def log_raw_event(event):
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(str(event)+"\n")


agent_app = build_app()
THREAD_ID = "agent_1"

class ChatRequest(BaseModel):
    user_input: str

class ChatResponse(BaseModel):
    answer: str
    trace: list

api = FastAPI()
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@api.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    events = agent_app.stream(
        {
            "messages": [HumanMessage(content=request.user_input)],
        },
        config={
            "configurable": {"thread_id": THREAD_ID},
        }
    )

    trace_steps = []
    final_answer = ""

    for event in events:
        # Log the event to file
        log_raw_event(event)

        node_name, payload = extract_event(event)
        trace_steps.append(node_name)
        last_message = payload["messages"][-1]
        if isinstance(last_message, AIMessage):
            final_answer = last_message.content

    return ChatResponse(
        answer=final_answer,
        trace=trace_steps
    )