from fastapi import FastAPI
from pydantic import BaseModel
from main import build_app
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage

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
        node_name, payload = next(iter(event.items()))
        trace_steps.append(node_name)
        last_message = payload["messages"][-1]
        if isinstance(last_message, AIMessage):
            final_answer = last_message.content

    return ChatResponse(
        answer=final_answer,
        trace=trace_steps
    )