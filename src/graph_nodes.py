from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, SystemMessage
from config import SYSTEM_PROMPT
from typing import List, Dict, Any
import uuid, json

def retrieve_context(state, chroma_db):
    last_user_message = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    docs = chroma_db.similarity_search(last_user_message, k = 2)
    context = "\n\n".join([d.page_content for d in docs])
    return {
        "messages": [
            SystemMessage(content=f"Relevant context:\n{context}")
        ]
    }

def call_model(state, llm_with_tools):
    messages: List = [SYSTEM_PROMPT] + state["messages"]
    ai: AIMessage = llm_with_tools.invoke(messages)
    return {"messages": [ai]}

def should_call_tools(state):
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage):
                if m.tool_calls:
                    return "tool_calls"
                break
        return "no_tools"

def call_tool(state, tools_by_name):
    latest_ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
    tool_calls = latest_ai.tool_calls
    tool_messages: List[ToolMessage] = []

    for c in tool_calls:
        tool_name = c.get("name")
        tool_args = c.get("args", {})
        tool_call_id = c.get("id", str(uuid.uuid4()))

        tool_fn = tools_by_name.get(tool_name)
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

def final_model(state, llm_with_tools):
    final_messages: List = [SYSTEM_PROMPT] + state["messages"]
    ai: AIMessage = llm_with_tools.invoke(final_messages)
    return {"messages": [ai]}