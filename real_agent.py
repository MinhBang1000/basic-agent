from pydantic import BaseModel
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv

load_dotenv()

# Step 1: Define the system prompt
system_prompt = "You are the supporter of a bussiness man who are really busy.Please do everything carefully because your boss has no time to fix."

# Step 2: 
# Create tools
@tool("get_user_location", description="Return the user's location")
def get_user_location(user_id: str) -> str:
    return "Hsinchu" if user_id == "1" else "Taipei"

@tool("get_weather", description="Return the weather at the location we already known")
def get_weather(city: str) -> str:
    return f"It's always sunny in the {city} city"

# Define Response Format, Context
class Context(BaseModel):
    user_id: str


class ResponseFormat(BaseModel):
    bunny_response: str
    weather_condition: str | None = None

def main():
    # Step 3: Configre your model
    llm = init_chat_model(
        model="google_genai:gemini-2.0-flash",
        temperature=0.6
    )

    # Step 4: Add memory
    checkpointer = MemorySaver()

    # Step 5: Configure agent and run
    agent = create_react_agent(
        model=llm,
        prompt=system_prompt,
        tools = [get_user_location, get_weather],
        context_schema = Context,
        # response_format = ResponseFormat,
        checkpointer = checkpointer
    )

    while True:
        user_prompt = input("Input: ")
        if user_prompt=="exit":
            break
        thread_id = "1"
        user_id = "1"
        config = {"configurable": {"thread_id": thread_id}}
        response = agent.invoke(
            {"messages": [{"role": "user", "content": f"{user_prompt}"}]},
            config = config,
            context = Context(user_id=user_id)
        )
        # print(response["structured_response"].bunny_response)
        # print(response["structured_response"].weather_condition)

        # print(f"Output: {str(response['messages'][-1].content)}")
        # print(f"Output {str(response)}")
        # print("")

        messages = response["messages"]
        for item in messages:
            print(f'[{item.type}]: {item.content}')
        print("")

if __name__ == "__main__":
    main()