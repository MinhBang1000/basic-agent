from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from config import SYSTEM_PROMPT, LLM
from tools import TOOLS, TOOLS_BY_NAME
from rag import setup_rag
from graph_nodes import retrieve_context, call_model, call_tool, should_call_tools, final_model
from utils import extract_event

def build_app():
    # rag define
    chroma_db = setup_rag()

    # llm with tools define
    llm_with_tools = LLM.bind_tools(TOOLS)

    # graph define
    graph = StateGraph(MessagesState)
    graph.add_node("retrieve", lambda s: retrieve_context(s, chroma_db))
    graph.add_node("agent", lambda s: call_model(s, llm_with_tools))
    graph.add_node("tools", lambda s: call_tool(s, TOOLS_BY_NAME))
    graph.add_node("final", lambda s: final_model(s, llm_with_tools))

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "agent")
    graph.add_conditional_edges(
        "agent", 
        should_call_tools,
        {
            "tool_calls": "tools",
            "no_tools": END
        }
    )
    graph.add_edge("tools", "agent")
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    app = build_app()
    thread_id = "demo-thread"
    while True:
        user = input("\nYou: ").strip()
        if user.lower() == "exit":
            break
        events = app.stream(
            {"messages": [HumanMessage(content=user)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        for ev in events:
            node_name, payload = extract_event(ev)
            if node_name in ["model", "final"]:
                msg = payload["messages"][-1]
                print(f"[{node_name.upper()}] -> {msg.content}")