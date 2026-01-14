# it is the amplified version of simple agent
# where we are adding more tools to the agent, memory
from dataclasses import dataclass
import requests
from dotenv import load_dotenv

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model

load_dotenv()

@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float
    

@tool 
def get_weather(city: str) -> str:
    """Get the weather for a given city"""
    try:
        response = requests.get(f"https://wttr.in/{city}?format=j1")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return f"Error getting weather: {e}"

@tool
def locate_user(config: RunnableConfig) -> str:
    """Look up a user city based on the context"""
    user_id = config.get("configurable", {}).get("user_id")
    match user_id:
        case "abc111":
            return "New York"
        case "def222":
            return "Los Angeles"
        case "ghi333":
            return "Amsterdam"
        case "jkl444":
            return "Paris"
        case _:
            return "Unknown"

# Initialize model
# Using Groq (Llama 3.3) which works well with tools and structured output
model = init_chat_model(
    model="mistral-medium-2508", 
    model_provider="mistralai",
    temperature=0.1,
)

checkpointer = InMemorySaver()

agent = create_react_agent(
    model=model,
    tools=[get_weather, locate_user],
    prompt=(
        "You are a helpful assistant that can get the weather for a given city. "
    ),
    response_format=ResponseFormat,
    checkpointer=checkpointer,
)

config = {
    'configurable': {
        'thread_id': '1',
        'user_id': 'ghi333'
    }
}

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is the weather in my location?"}
    ]
}, config=config)

structured = response.get('structured_response')

if structured:
    print("-" * 20)
    print("Structured Response:")
    print(f"Summary: {structured['summary']}")
    print(f"Temp C: {structured['temperature_celsius']}")
    print(f"Temp F: {structured['temperature_fahrenheit']}")
    print(f"Humidity: {structured['humidity']}")
else:
    print("No structured response returned.")
    # Fallback to last message if structured response failed
    print(response["messages"][-1].content)


responseX = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is the weather in my location?"}
    ]
}, 
    config=config,
    context= Context(user_id="ghi333")
)

print(responseX)
