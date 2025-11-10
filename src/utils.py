from typing import Any, Tuple

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
