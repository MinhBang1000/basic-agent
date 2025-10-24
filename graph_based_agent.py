from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny, 26°C."

def llm_node(state: MessagesState):
    user_input = state["messages"][-1].content
    tool_call_id = str(uuid.uuid4())
    return {
        "messages": [
            AIMessage(
                content=f"I'll check the weather for Taipei.",
                additional_kwargs = {
                    "tool_calls": [
                        {
                            "name": "get_weather",
                            "arguments": {"city":"Taipei"},
                            "id": tool_call_id
                        }
                    ]
                },
                response_metadata = {
                    "model": "mock-llm",
                    "finish_reason" : "tool_calls",
                    "token_usage": {"input":10, "output": 20}
                },
                id=str(uuid.uuid4())
            )
        ]
    }


def tool_node(state: MessagesState):
    ai_message = next(m for m in state["messages"] if isinstance(m, AIMessage))
    tool_call = ai_message.additional_kwargs["tool_calls"][0]
    result = get_weather(tool_call["arguments"]["city"])
    return {
        "messages": [
            ToolMessage(
                content=result,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                id=str(uuid.uuid4())
            )
        ]
    }

def final_llm(state: MessagesState):
    tool_output = [m for m in state["messages"] if isinstance(m, ToolMessage)][-1].content
    return {
        "messages": [
            AIMessage(
                content=f"{tool_output}. Have a nice day!",
                response_metadata={"finish_reason": "stop", "model": "mock-llm"},
                id=str(uuid.uuid4())
            )
        ]
    }

# Build the graph
graph = StateGraph(MessagesState)
graph.add_node("llm_node", llm_node)
graph.add_node("tool_node", tool_node)
graph.add_node("final_llm", final_llm)
graph.add_edge(START, "llm_node")
graph.add_edge("llm_node","tool_node")
graph.add_edge("tool_node", "final_llm")
graph.add_edge("final_llm", END)
graph = graph.compile()

# --- Run ---
result = graph.invoke({"messages": [HumanMessage(content="What’s the weather in Taipei?")]})
for msg in result["messages"]:
    print(f"\n[{msg.__class__.__name__}] {msg.content}")
    print("role/type:", msg.type)
    print("id:", msg.id)
    print("additional_kwargs:", msg.additional_kwargs)
    print("response_metadata:", msg.response_metadata)

