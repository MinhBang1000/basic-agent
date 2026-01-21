from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from config import LLM
from tools import TOOLS, TOOLS_BY_NAME
from graph_nodes import call_model, call_tool, should_call_tools, retrieve_context
from utils import extract_event
import constraints

def build_app(chroma_db):
    # number of chunk
    k = 5

    # LLM with tools
    llm_with_tools = LLM.bind_tools(TOOLS)

    graph = StateGraph(MessagesState)

    # ---------- Nodes ----------
    graph.add_node(
        constraints.NODE_RETRIEVE,
        lambda s: retrieve_context(s, chroma_db, k)
    )

    graph.add_node(
        constraints.NODE_AGENT,
        lambda s: call_model(s, llm_with_tools)
    )

    graph.add_node(
        constraints.NODE_TOOLS,
        lambda s: call_tool(s, TOOLS_BY_NAME)
    )

    # ---------- Edges ----------
    graph.add_edge(START, constraints.NODE_RETRIEVE)
    graph.add_edge(constraints.NODE_RETRIEVE, constraints.NODE_AGENT)

    graph.add_conditional_edges(
        constraints.NODE_AGENT,
        should_call_tools,
        {
            "tool_calls": constraints.NODE_TOOLS,
            "no_tools": END,
        },
    )

    graph.add_edge(constraints.NODE_TOOLS, constraints.NODE_AGENT)

    # ---------- Checkpoint ----------
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