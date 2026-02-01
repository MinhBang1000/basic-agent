import json

# ========== INPUT FILES ==========
DEEPSET_FILE = "datasets/deepset_prompt_injections_all.jsonl"
TOOL_FILE = "datasets/tool_injection.jsonl"
NQ_FILE = "docs/benign/queries.jsonl"

# ========== OUTPUT FILE ==========
OUTPUT_FILE = "datasets/benign_user_prompts.jsonl"

benign_prompts = []

# --- Collect from Deepset (label: 0 = benign) ---
with open(DEEPSET_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("label") == 0:
            benign_prompts.append({
                "source": "deepset",
                "text": obj["text"].strip()
            })

# --- Collect from Tool Injection ---
with open(TOOL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        benign_prompts.append({
            "source": "tool_injection",
            "text": obj["user_prompt"].strip()
        })

# --- Collect from Natural Questions (RAG / Correlated) ---
with open(NQ_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        benign_prompts.append({
            "source": "natural_questions",
            "text": obj["text"].strip()
        })

# --- Save merged output ---
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for obj in benign_prompts:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"✅ Total benign prompts collected: {len(benign_prompts)}")
