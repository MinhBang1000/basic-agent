from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from typing import Any, List, Dict
import uuid
import json
from dotenv import load_dotenv

load_dotenv()


# -----------------------
# Tools
# -----------------------

@tool("get_weather", description="To get weather in a city.")
def get_weather(city: str):
    # f-string so `city` is actually inserted
    return f"It is always sunny in {city}."

@tool("add", description="To add two numbers. Both of them are integers.")
def add(a: int, b: int):
    return a + b


# List of tool objects (langchain_core.tools.Tool)
TOOLS = [get_weather, add]

# Fast lookup by name, e.g. "get_weather" -> get_weather tool object
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# -----------------------
# LLM
# -----------------------

llm = init_chat_model(
    model="google_genai:gemini-2.0-flash",  # adjust to a model you actually have access to
    temperature=0.6,
)

# Bind tools so the model is ALLOWED to ask for tool calls
llm_with_tools = llm.bind_tools(TOOLS)


# -----------------------
# System prompt
# -----------------------

SYSTEM_PROMPT = SystemMessage(content=(
    "You are an integrated assistant. "
    "Be concise, cite tool results explicitly, and never fabricate tool outputs."
))


# -----------------------
# Node 1: model call
# -----------------------

def call_model(state: MessagesState):
    # Prepend system instruction to the running chat state
    messages: List = [SYSTEM_PROMPT] + state["messages"]

    # Ask model. Because llm_with_tools is bound, the AIMessage may contain tool_calls.
    ai: AIMessage = llm_with_tools.invoke(messages)

    # We append the AIMessage into the state["messages"]
    return {"messages": [ai]}


# -----------------------
# Router: decide if we need tools
# -----------------------

def should_call_tools(state: MessagesState):
    # Look at the latest AIMessage and see if it requested any tool calls.
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage):
            if m.tool_calls:
                return "tool_calls"  # <-- must match mapping below
            break
    return "no_tools"


# -----------------------
# Node 2: actually execute tools
# -----------------------

def call_tool(state: MessagesState):
    # Get the most recent AIMessage (the one that asked for tools)
    latest_ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))

    # The model encodes tool call requests in AIMessage.additional_kwargs["tool_calls"]
    tool_calls = latest_ai.tool_calls

    tool_messages: List[ToolMessage] = []

    for c in tool_calls:
        tool_name = c.get("name")
        # depending on provider version this may be "arguments" or "args"
        tool_args = c.get("args", {})
        tool_call_id = c.get("id", str(uuid.uuid4()))

        tool_fn = TOOLS_BY_NAME.get(tool_name)

        if tool_fn is None:
            result = f"[ToolError] Unknown tool: {tool_name}"
        else:
            try:
                # tool_fn.invoke(...) accepts a dict of kwargs
                result = tool_fn.invoke(tool_args)
            except Exception as e:
                result = f"[ToolError] {type(e).__name__}: {e}"

        # Build a ToolMessage to feed back into the graph state
        tool_messages.append(
            ToolMessage(
                content=json.dumps({"result": result}, ensure_ascii=False),
                name=tool_name,
                tool_call_id=tool_call_id,
                id=str(uuid.uuid4()),
            )
        )

    # Return all tool outputs so they get merged into state["messages"]
    return {"messages": tool_messages}


# -----------------------
# Node 3: final model call
# -----------------------

def final_model(state: MessagesState):
    # Now that tools have answered, ask the LLM again WITHOUT binding tools.
    # This should produce the final user-facing answer.
    final_messages: List = [SYSTEM_PROMPT] + state["messages"]
    ai: AIMessage = llm_with_tools.invoke(final_messages)
    return {"messages": [ai]}


# -----------------------
# Graph wiring
# -----------------------

graph = StateGraph(MessagesState)

graph.add_node("model", call_model)   # Node 1
graph.add_node("tools", call_tool)    # Node 2
graph.add_node("final", final_model)  # Node 3

# START -> model
graph.add_edge(START, "model")

# If model asked for tools, go to "tools"; else go to "final"
graph.add_conditional_edges(
    "model",
    should_call_tools,
    {
        "tool_calls": "tools",
        "no_tools": END,
    },
)

# After running tools, always go to final
graph.add_edge("tools", "final")

# final -> END
graph.add_edge("final", END)


# -----------------------
# Memory + compile
# -----------------------

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)


# -----------------------
# CLI loop to test
# -----------------------

if __name__ == "__main__":
    print("Agent ready. Type 'exit' to quit.")
    thread_id = "demo-thread"  # in a real app this is per user/session

    while True:
        user = input("\nYou: ").strip()
        if user.lower() == "exit":
            break

        # .stream() yields events node-by-node so you can watch the reasoning
        events = app.stream(
            {"messages": [HumanMessage(content=user)]},
            config={"configurable": {"thread_id": thread_id}},
        )

        final_state = None

        for event in events:
            node_name, payload = next(iter(event.items()))
            if node_name in ("model", "final"):
                msg = payload["messages"][-1]  # AIMessage
                print(f"\n[{node_name.upper()}] ({msg.type}) -> {msg.content}")
            elif node_name == "tools":
                msg = payload["messages"][-1]  # ToolMessage
                print(f"\n[TOOLS] ({msg.name}) -> {msg.content}")
            final_state = payload

        # Optional debug info about the final AIMessage
        if final_state:
            last = final_state["messages"][-1]
            if isinstance(last, AIMessage):
                print("\n--- response_metadata ---")
                print(json.dumps(last.response_metadata or {}, indent=2))
