# Agent Security Testbed for RAG & Tool-Using Agents

## Overview

This repository implements a **research-oriented agent testbed** for studying **security vulnerabilities and defenses in tool-using, RAG-enabled AI agents**.

The project focuses on **misleading retrieved context**, **tool-use hijacking**, and **agent behavior deviation**, rather than classical QA accuracy.
It supports **dataset generation, controlled poisoning, full-corpus evaluation**, and **multi-layer defense experimentation**.

The system is designed to:

* Build **benign and poisoned knowledge bases**
* Execute agents on **large Chroma vector stores**
* Log **step-by-step agent message traces**
* Evaluate **attack success at the behavior level (ASR)**

This codebase directly supports **thesis-scale experiments** on AI agent security.

---

## Research Scope

This project studies **agent misbehavior**, not traditional RAG QA errors.

### Threats considered

* Misleading retrieved context (AS / AK-style poisoning)
* Tool-use steering without system-prompt control
* Prompt injection via retrieved knowledge
* Extra or unintended agent actions

### Threats NOT assumed

* No system prompt manipulation
* No access to ground-truth relevance labels (qrels)
* No multi-document poisoning per query
* No model weight access

---

## Repository Structure

```text
basic_agent/
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
│
├── datasets/
│   ├── nq.zip
│   └── nq/
│
├── src/
│   ├── .env
│   ├── main.py                 # Entry point (agent execution)
│   ├── server.py               # Optional server / API entry
│   ├── config.py               # Global configuration
│   ├── constraints.py          # Security constraints & policies
│   ├── credentials.json        # Local credentials (DO NOT COMMIT)
│   ├── utils.py
│   ├── tools.py                # Tool definitions
│   ├── graph_nodes.py          # Agent graph node logic
│   ├── rag.py                  # RAG + Chroma setup (mode-based)
│   │
│   ├── build_index.py           # Build Chroma vector DB
│   ├── generate_poisoned_as.py  # Generate AS poisoned corpus
│   ├── generate_poisoned_as_ak.py
│   ├── eval_full_as.py          # Full-corpus AS evaluation
│   ├── eval_range_as.py         # k-range evaluation
│   │
│   ├── chroma_db/
│   │   ├── benign/              # Benign Chroma DB
│   │   ├── poisoned_as/         # AS-poisoned Chroma DB
│   │   ├── poisoned_ak/         # AK-poisoned Chroma DB
│   │   ├── _mini_as_m_queries/  # Mini test DB
│   │   └── _one_test_as_mini/
│   │
│   ├── docs/
│   │   ├── benign/
│   │   │   ├── corpus.jsonl
│   │   │   └── queries.jsonl
│   │   ├── poisoned_as/
│   │   │   ├── corpus.jsonl
│   │   │   └── queries.jsonl
│   │   ├── poisoned_ak/
│   │   │   ├── corpus.jsonl
│   │   │   └── queries.jsonl
│   │   └── _one_test/
│   │
│   ├── eval_logs/
│   │   ├── eval_benign_k4.*
│   │   ├── eval_poisoned_as_k4.*
│   │   ├── eval_poisoned_ak_k4.*
│   │   └── summary_k4.json
│   │
│   ├── eval_out/
│   │   └── eval_poisoned_as.jsonl
│   │
│   ├── messages/
│   │   └── logs.txt             # Raw agent message traces
│   │
│   └── uploads/                 # Tool test files
│
└── __pycache__/
```

---

## Key Components

### 1. Knowledge Bases (Corpus)

Located in `src/docs/`:

* `benign/` – clean corpus + queries
* `poisoned_as/` – benign + AS poisoned docs
* `poisoned_ak/` – benign + AK poisoned docs

Each corpus is stored as `jsonl` and later indexed into Chroma.

---

### 2. Vector Databases (Chroma)

Located in `src/chroma_db/`:

* `benign/` – baseline knowledge base
* `poisoned_as/` – AS-poisoned DB
* `poisoned_ak/` – AK-poisoned DB

All evaluation is performed on **full databases**, not curated mini sets.

---

### 3. Agent & RAG Logic

* `rag.py`

  * `setup_rag(mode=1)` → benign DB
  * `setup_rag(mode=2)` → poisoned AS DB
  * `setup_rag(mode=3)` → poisoned AK DB

* `graph_nodes.py`

  * Agent reasoning steps
  * Tool invocation logic
  * Message logging hooks

---

### 4. Dataset Generation

* `generate_poisoned_as.py`

  * Injects **one AS poison per query**
  * No system prompt control
  * Poison derived only from LLM interaction

* `generate_poisoned_as_ak.py`

  * Extends AS → AK (knowledge-style rewrite)

---

### 5. Evaluation Scripts

* `eval_full_as.py`

  * Full-corpus evaluation on selected queries
  * Computes:

    * **RHR** – Retrieval Hit Rate
    * **ASR** – Attack Success Rate (behavior-level)
    * **TSR (behavioral)** – agent stays on-task

* `eval_range_as.py`

  * Evaluate ASR over different top-k values

Evaluation outputs:

* CSV / JSONL logs per query
* Summary JSON

---

## Metrics Definition

This project prioritizes **agent behavior**, not factual correctness.

* **AS (Attack Success)**
  Agent exhibits attacker-intended behavior (e.g., tool-use steering).

* **ASR (Attack Success Rate)**
  Percentage of queries with successful behavioral deviation.

* **RHR (Retrieval Hit Rate)**
  Poisoned document appears in retrieved context.

* **TSR (Behavioral Task Success Rate)**
  Agent completes the original task without deviation.

---

## Environment Setup

### Requirements

* Python 3.8+
* pip
* Windows / Linux / macOS (tested on Windows + PowerShell)

### Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Variables

Create `src/.env`:

```env
OPENAI_API_KEY=your_api_key_here
```

---

## Typical Workflow

```text
1. Prepare benign corpus + queries
2. Build benign Chroma DB
3. Generate AS / AK poisoned corpora
4. Build poisoned Chroma DBs
5. Run full-corpus evaluation
6. Collect ASR / RHR / TSR
7. Log agent traces for defense research
```

---

## Research Notes

* Retrieved context is treated as **untrusted input**
* Tool usage is considered **high-risk behavior**
* Message logs are first-class research artifacts
* The design supports **multi-layer defenses**:

  * Layer 1: Anomaly detection on benign traces
  * Layer 2: Supervised attack detection
  * Layer 3: LLM-based behavioral auditor

---

## License

This repository is intended for **academic research and thesis work**.
Add a license file before public release.
