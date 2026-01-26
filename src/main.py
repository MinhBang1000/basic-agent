from langgraph.graph import MessagesState, START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from config import LLM
from graph_nodes import call_model, call_tool, should_call_tools, retrieve_context
from rag import setup_rag
from pathlib import Path
from utils import extract_event
import constraints
import time
import sys
import os



ENV_PATH = ".env"
# ---------------------------
# Optional logging (same style as server.py)
# ---------------------------
LOG_DIR = Path("messages")
LOG_FILE = LOG_DIR / "logs.txt"
LOG_DIR.mkdir(parents=True, exist_ok=True)

def log_raw_event(event):
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(str(event) + "\n")

def smooth_print(text: str, delay: float = 0.015):
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    print()  # newline

def build_app(chroma_db, k: int = 5):
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


def run_cli(mode: int = 2, thread_id: str = "agent_1", k: int = 5, enable_log: bool = False):
    chroma_db = setup_rag(mode)
    app = build_app(chroma_db, k=k)

    print("\n=== LangGraph Agent CLI ===")
    print(f"MODE={mode} | THREAD_ID={thread_id} | k={k}")
    print("Type 'exit' to quit.\n")

    while True:
        user = input("You: ").strip()
        if not user:
            continue
        if user.lower() == "exit":
            smooth_print("Goodbye, my friend. Take care and see you next time.")
            break

        events = app.stream(
            {"messages": [HumanMessage(content=user)]},
            config={"configurable": {"thread_id": thread_id}},
        )

        trace_steps = []
        final_answer = ""

        for ev in events:
            if enable_log:
                log_raw_event(ev)

            node_name, payload = extract_event(ev)
            msg = payload["messages"][-1]

            # trace: tool name nếu là ToolMessage, còn lại là node_name
            if isinstance(msg, ToolMessage):
                trace_steps.append(f"TOOL:{msg.name}")
            else:
                trace_steps.append(node_name)

            if isinstance(msg, AIMessage):
                final_answer = msg.content or final_answer

        if enable_log:
            log_raw_event(f"\ntrace: {trace_steps}\n---\n")

        print("\n[ANSWER]")
        smooth_print(final_answer, delay=0.02)
        print("\n[TRACE]")
        print(" -> ".join(trace_steps))
        print("")

def _set_env_var_in_file(env_path: str, key: str, value: str) -> None:
    """
    Update or insert KEY=VALUE in .env file.
    Keeps other variables intact.
    """
    lines = []
    found = False

    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith(f"{key}="):
                    lines.append(f"{key}={value}\n")
                    found = True
                else:
                    # keep original line, but normalize newline
                    lines.append(line if line.endswith("\n") else line + "\n")

    if not found:
        lines.append(f"{key}={value}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def pick_mode_menu() -> int:
    """
    MODE meanings:
      1 = benign
      2 = poisoned_as
      3 = tool_injection
    """
    print("\nSelect RAG corpus:")
    print("  1) benign")
    print("  2) poisoned_as")
    print("  3) tool_injection")

    while True:
        choice = input("Enter choice (1/2/3): ").strip()

        if choice in ("1", "2", "3"):
            _set_env_var_in_file(ENV_PATH, "MODE", choice)
            print(f"[ENV] MODE={choice} written to .env")
            return int(choice)

        print("Invalid choice. Please enter 1, 2, or 3.\n")

if __name__ == "__main__":
    # Minimal CLI settings (you can still hardcode these)
    THREAD_ID = "agent_1"
    K = 5
    ENABLE_LOG = True

    mode = pick_mode_menu()

    # Ensure the pick mode menu will be effective
    from tools import TOOLS, TOOLS_BY_NAME

    label = {1: "benign", 2: "poisoned_as", 3: "tool_injection"}.get(mode, "unknown")
    print(f"\n✅ Using MODE={mode} ({label})\n")

    run_cli(mode=mode, thread_id=THREAD_ID, k=K, enable_log=ENABLE_LOG)