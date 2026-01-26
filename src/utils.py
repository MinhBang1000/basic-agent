from typing import Any, Tuple
from dotenv import load_dotenv
import os

def extract_event(event: dict[str, Any]) -> Tuple[str, Any]:
    """
    Extract the single (key, value) pair from a LangGraph event dictionary.

    Each LangGraph event is expected to be a one-key dictionary, e.g.:
        {"model": {"messages": [...]}}

    Returns:
        A tuple (node_name, payload), where:
            node_name (str): The name of the executed node (e.g., "model", "tools", etc.)
            payload (Any):   The node's output dictionary (e.g., {"messages": [...]})

    Raises:
        ValueError: If the event is not a dictionary or is empty.
        TypeError:  If the event is not iterable as expected.
        RuntimeError: If multiple items exist (unexpected in LangGraph events).
    """
    try:
        if not isinstance(event, dict):
            raise TypeError(f"Expected dict, got {type(event).__name__}")

        if len(event) == 0:
            raise ValueError("Event dictionary is empty.")

        if len(event) > 1:
            raise RuntimeError(
                f"Expected a single key in event, but got {len(event)} keys: {list(event.keys())}"
            )

        # Safely extract first (and only) item
        node_name, payload = next(iter(event.items()))
        return node_name, payload

    except Exception as e:
        # You can log this instead of printing in production
        raise RuntimeError(f"Failed to extract event: {e}") from e

def get_mode_from_env(env_path: str = ".env") -> int:
    """
    Read MODE from .env as an integer.

    MODE meanings:
      1 = benign
      2 = poisoned_as
      3 = tool_injection

    Returns:
        int: MODE value

    Raises:
        RuntimeError: if MODE is missing or invalid
    """
    load_dotenv(env_path, override=False)

    raw = os.getenv("MODE")
    if raw is None:
        raise RuntimeError("MODE is not set in .env")

    try:
        mode = int(raw)
    except ValueError:
        raise RuntimeError(f"MODE must be an integer, got: {raw}")

    if mode not in (1, 2, 3):
        raise RuntimeError(f"Invalid MODE={mode}. Expected 1, 2, or 3.")

    return mode

