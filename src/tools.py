from langchain_core.tools import tool

@tool("get_weather", description="Get weather in a city")
def get_weather(city: str):
    return f"It is always sunny in {city}."

@tool("add", description="Add two integers.")
def add(a: int, b: int):
    return a+b

TOOLS = [get_weather, add]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}