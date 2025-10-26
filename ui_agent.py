from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

# RAG imported packages
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


from typing import Any, List, Dict
import uuid
import json
from dotenv import load_dotenv

def build_app():
    # ===== your current setup code starts here =====

    # load .env
    load_dotenv()

    # ---------- RAG setup ----------
    with open("docs/ai_security.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text(raw_text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    chroma_db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # ---------- tools ----------
    @tool("get_weather", description="To get weather in a city.")
    def get_weather(city: str):
        return f"It is always sunny in {city}."

    @tool("add", description="To add two numbers. Both of them are integers.")
    def add(a: int, b: int):
        return a + b

    TOOLS = [get_weather, add]
    TOOLS_BY_NAME = {t.name: t for t in TOOLS}

    # ---------- model / system prompt ----------
    llm = init_chat_model(
        model="google_genai:gemini-2.0-flash",
        temperature=0.6,
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    SYSTEM_PROMPT = SystemMessage(content=(
        "You are an integrated assistant. "
        "Be concise, cite tool results explicitly, and never fabricate tool outputs."
    ))

    # ---------- node defs ----------
    def retrieve_context(state: MessagesState):
        last_user_message = None
        for m in reversed(state["messages"]):
            if isinstance(m, HumanMessage):
                last_user_message = m.content
                break
        docs = chroma_db.similarity_search(last_user_message, k=2)
        context = "\n \n".join([d.page_content for d in docs])
        return {
            "messages": [SystemMessage(content=f"Relevant context: {context}")]
        }

    def call_model(state: MessagesState):
        messages: List = [SYSTEM_PROMPT] + state["messages"]
        ai: AIMessage = llm_with_tools.invoke(messages)
        return {"messages": [ai]}

    def should_call_tools(state: MessagesState):
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage):
                if m.tool_calls:
                    return "tool_calls"
                break
        return "no_tools"

    def call_tool(state: MessagesState):
        latest_ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
        tool_calls = latest_ai.tool_calls
        tool_messages: List[ToolMessage] = []

        for c in tool_calls:
            tool_name = c.get("name")
            tool_args = c.get("args", {})
            tool_call_id = c.get("id", str(uuid.uuid4()))

            tool_fn = TOOLS_BY_NAME.get(tool_name)
            if tool_fn is None:
                result = f"[ToolError] Unknown tool: {tool_name}"
            else:
                try:
                    result = tool_fn.invoke(tool_args)
                except Exception as e:
                    result = f"[ToolError] {type(e).__name__}: {e}"

            tool_messages.append(
                ToolMessage(
                    content=json.dumps({"result": result}, ensure_ascii=False),
                    name=tool_name,
                    tool_call_id=tool_call_id,
                    id=str(uuid.uuid4()),
                )
            )

        return {"messages": tool_messages}

    def final_model(state: MessagesState):
        final_messages: List = [SYSTEM_PROMPT] + state["messages"]
        ai: AIMessage = llm_with_tools.invoke(final_messages)
        return {"messages": [ai]}

    # ---------- graph wiring ----------
    graph = StateGraph(MessagesState)
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("model", call_model)
    graph.add_node("tools", call_tool)
    graph.add_node("final", final_model)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "model")

    graph.add_conditional_edges(
        "model",
        should_call_tools,
        {
            "tool_calls": "tools",
            "no_tools": END,
        },
    )

    graph.add_edge("tools", "final")
    graph.add_edge("final", END)

    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)

    return app

# keep CLI mode working for terminal
if __name__ == "__main__":
    app = build_app()
    print("Agent ready. Type 'exit' to quit.")
    thread_id = "demo-thread"
    while True:
        user = input("\nYou: ").strip()
        if user.lower() == "exit":
            break
        events = app.stream(
            {"messages": [HumanMessage(content=user)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        final_state = None
        for event in events:
            node_name, payload = next(iter(event.items()))
            if node_name in ("model", "final"):
                msg = payload["messages"][-1]
                print(f"\n[{node_name.upper()}] ({msg.type}) -> {msg.content}")
            elif node_name == "tools":
                msg = payload["messages"][-1]
                print(f"\n[TOOLS] ({msg.name}) -> {msg.content}")
            final_state = payload

        if final_state:
            last = final_state["messages"][-1]
            if isinstance(last, AIMessage):
                print("\n--- response_metadata ---")
                print(json.dumps(last.response_metadata or {}, indent=2))
