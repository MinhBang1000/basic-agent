from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

def get_weather(city: str) -> str:
    """Return the weather of a given city"""
    return f"The weather in the {city} city is Sunny"

def main():

    llm = init_chat_model(
        model="google_genai:gemini-2.0-flash",
        temperature=0.3
    )

    agent = create_agent(
        model=llm,
        tools=[get_weather],
        system_prompt="You are the good supporter"
    )

    while True:
        prompt = input("Input: ")

        if prompt == "exit":
            break

        result = agent.invoke({
            "messages": [
                {"role":"user", "content": prompt}
            ]
        })

        print("Output: " + str(result["messages"][-1].content))

if __name__ == "__main__":
    main()