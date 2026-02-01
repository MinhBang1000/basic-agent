import json
import random
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from main import build_app
from rag import setup_rag
from utils import extract_event, set_env_var, get_mode_from_env

# =========================
# CONFIG
# =========================
OUTPUT_DIR = Path("collected_logs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "benign_traces.jsonl"

RECURSION_LIMIT = 25
THREAD_PREFIX = "collector"

# =========================
# MENU HELPERS
# =========================
def ask_yes_no(msg: str) -> bool:
    while True:
        ans = input(f"{msg} (y/n): ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer y or n.")

def ask_sample_size(max_n: int) -> int:
    while True:
        ans = input(f"Sample how many prompts? (0 = all, max = {max_n}): ").strip()
        if ans.isdigit():
            n = int(ans)
            if 0 <= n <= max_n:
                return n
        print("Invalid number.")

# =========================
# LOAD BENIGN PROMPTS
# =========================
def load_benign_prompts() -> List[Dict]:
    prompts = []

    # --- Deepset (label = 0) ---
    with open("datasets/deepset_prompt_injections_all.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if int(obj.get("label", 1)) == 0:
                prompts.append({
                    "source": "deepset",
                    "text": obj["text"].strip()
                })

    # --- Tool Injection (all benign prompts) ---
    with open("datasets/tool_injection.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prompts.append({
                "source": "tool_injection",
                "text": obj["user_prompt"].strip()
            })

    # --- Natural Questions (RAG / Correlated base) ---
    with open("docs/benign/queries.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prompts.append({
                "source": "natural_questions",
                "text": obj["text"].strip()
            })

    return prompts

# =========================
# RUN AGENT + COLLECT TRACE
# =========================
def run_agent(prompt: str, app, thread_id: str):
    trace_steps = []
    reasoning_steps = []
    tools_used = []
    final_answer = ""
    hit_recursion = False
    error = None

    try:
        events = app.stream(
            {"messages": [HumanMessage(content=prompt)]},
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": RECURSION_LIMIT,
            },
        )

        for ev in events:
            node_name, payload = extract_event(ev)
            msg = payload["messages"][-1]

            step_log = {
                "node": node_name,
                "type": "tool" if isinstance(msg, ToolMessage) else "agent",
                "content": msg.content if hasattr(msg, "content") else "",
            }

            if isinstance(msg, ToolMessage):
                tool_name = msg.name
                tools_used.append(tool_name)
                trace_steps.append(f"TOOL:{tool_name}")
                step_log["tool"] = tool_name
            else:
                trace_steps.append(node_name)

            if isinstance(msg, AIMessage):
                final_answer = msg.content or final_answer

            reasoning_steps.append(step_log)

    except GraphRecursionError:
        hit_recursion = True
        error = "GraphRecursionError"

    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    return {
        "trace": trace_steps,
        "reasoning": reasoning_steps,
        "tools": list(set(tools_used)),
        "final_answer": final_answer,
        "hit_recursion": hit_recursion,
        "error": error,
    }

# =========================
# MAIN
# =========================
def main():
    print("\n=== BENIGN DATASET COLLECTOR ===\n")

    prompts = load_benign_prompts()
    total = len(prompts)
    print(f"Loaded {total} benign prompts.")

    # -------- Control Menu --------
    print("\n--- Control Menu ---")
    do_shuffle = ask_yes_no("Shuffle prompts?")
    sample_n = ask_sample_size(total)

    if do_shuffle:
        random.shuffle(prompts)

    if sample_n > 0:
        prompts = prompts[:sample_n]

    print(f"\nFinal prompt count: {len(prompts)}")
    print("Starting collection...\n")

    # -------- Build Agent (MODE=1 benign) --------
    ENV_PATH = ".env"
    set_env_var(ENV_PATH, "MODE", "1")
    load_dotenv(override=True)
    MODE = get_mode_from_env(ENV_PATH)
    print("[ENV] MODE=1 (benign) enforced for collector")

    chroma_db = setup_rag(MODE)  # Uses .env
    app = build_app(chroma_db, k=6)

    # -------- Run & Collect --------
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for i, item in enumerate(prompts, 1):
            prompt = item["text"]
            source = item["source"]
            query_id = f"{source}_{i}"

            result = run_agent(
                prompt=prompt,
                app=app,
                thread_id=f"{THREAD_PREFIX}_{i}",
            )

            record = {
                "id": query_id,
                "source": source,
                "prompt": prompt,
                "trace": result["trace"],
                "reasoning_steps": result["reasoning"],
                "tools_used": result["tools"],
                "final_answer": result["final_answer"],
                "hit_recursion": result["hit_recursion"],
                "error": result["error"],
                "label": 0  # Always benign
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            status = "OK"
            if result["hit_recursion"]:
                status = "RECURSION"
            elif result["error"]:
                status = "ERROR"

            print(f"[{i}/{len(prompts)}] {status} | {source}")

    print(f"\n✅ Collection finished.")
    print(f"Saved to: {OUTPUT_FILE.resolve()}")

if __name__ == "__main__":
    main()
