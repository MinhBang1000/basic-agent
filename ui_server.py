from fastapi import FastAPI
from pydantic import BaseModel
from ui_agent import build_app
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage

agent_app = build_app()
THREAD_ID = "web-thread"

class ChatRequest(BaseModel):
    user_input: str

class ChatResponse(BaseModel):
    answer: str
    trace: list

api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "*",  # you can remove "*" later if you want to lock it down
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@api.post("/chat", response_model = ChatResponse)
def chat(req: ChatRequest):
    events = agent_app.stream(
        {"messages": [HumanMessage(content=req.user_input)]},
        config={"configurable": {"thread_id": THREAD_ID}}
    )

    final_answer = ""
    trace_steps = []

    for event in events:
        node_name, payload = next(iter(event.items()))
        trace_steps.append(node_name)

        last_msg = payload["messages"][-1]
        if isinstance(last_msg, AIMessage):
            final_answer = last_msg.content

    return ChatResponse(
        answer=final_answer,
        trace=trace_steps
    )