# Project README

Short description
- This repository provides several small agent examples that use LangChain-style chat agents. The primary entrypoint shown here is main.py, which demonstrates creating an agent with a simple tool function.

Repository files
- .env — environment variables
- .gitignore
- main.py — example entrypoint; contains `main.main` and `main.get_weather`
- basic_agent.py
- graph_based_agent.py
- integrated_graph_agent.py
- rag_agent.py
- real_agent.py
- chroma.sqlite3
- 751a7066-3958-47ff-a8f5-9ab62fcde293
- ai_security.txt

Requirements
- Python 3.8+
- pip
- Project dependencies (LangChain and any chat model SDKs referenced by the code). Install typical requirements with pip.

Quick setup
1. Create and activate a virtual environment:
    ````sh
    python -m venv .venv
    source .venv/bin/activate  # or .venv\Scripts\activate on Windows
    ````
2. Install dependencies (example):
    ````sh
    pip install -r requirements.txt
    ````

Environment
- Put API keys and other secrets into .env. The example main.py calls load_dotenv() to load environment variables.

Run the example
- Start the interactive agent demo in main.py:
    ````sh
    python main.py
    ````
- Type prompts when asked. Enter "exit" to quit.

Notes and gotchas
- The example tool function in main.py is `main.get_weather`. It currently returns a literal string with a small bug: it uses "f{city}" instead of an f-string interpolation. Fix by making it an f-string:
    ````py
    # filepath: main.py
    # ...existing code...
    def get_weather(city: str) -> str:
        return f"The weather in the {city} city is Sunny"
    # ...existing code...
    ````
- The example uses `init_chat_model` and `create_agent` from external libraries (LangChain). Check those APIs and the required model provider credentials.

Security
- See ai_security.txt for security considerations and handling of API keys and data.

If you want, I can:
- Add a requirements.txt
- Fix the f-string bug in main.py
- Add usage examples for the other agent scripts

The structure:
project-name/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app instance and router inclusion
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── endpoints.py     # FastAPI routers (@app.post, @app.get)
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py        # Settings, API keys (loaded from .env)
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── agent_schemas.py # Pydantic models for request/response
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   └── agent_service.py # Business logic (where FastAPI calls LangGraph)
│   │
│   ├── graphs/
│   │   ├── __init__.py
│   │   ├── state.py         # LangGraph state definition (TypedDict)
│   │   ├── nodes.py         # Node functions (call models, tools, etc.)
│   │   ├── tools/           # Directory for custom tools
│   │   │   ├── __init__.py
│   │   │   └── web_search.py
│   │   └── workflow.py      # Graph assembly (add_nodes, add_edges, compile)
│   │
│   └── tools/
│       ├── __init__.py
│       └── ...              # (Alternative location for tools)
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py          # Tests for FastAPI endpoints
│   └── test_graph.py        # Tests for the LangGraph logic
│
├── .env                     # Secrets and environment variables (NEVER commit)
├── .gitignore               # Files/dirs to ignore (e.g., __pycache__, .env, venv/)
├── README.md                # Project description and setup instructions
└── requirements.txt         # Project dependencies (fastapi, uvicorn, langgraph, langchain)