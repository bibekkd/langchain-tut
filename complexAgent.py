# it is the amplified version of simple agent
# where we are adding more tools to the agent, memory
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

@dataclass
class context:
    user_id: str

@dataclass
class ResponseFormat:
    summary: str,
    temperature_celsius: int,
    temperature_fahrenheit: int,
    humidity: float,
    

@tool
def get_weather(city: str) -> str:
    """Get the weather for a given city"""
    response = requests.get(f"https://wttr.in/{city}?format=j1")
    return response.json()

# Use Groq's Llama model (fast & generous free tier)
llm = ChatGroq(model="llama-3.3-70b-versatile")

agent = create_react_agent(
    model=llm,
    tools=[get_weather],
    prompt="You are a helpful assistant that can get the weather for a given city.",
)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is the weather in New York?"}
    ]
})

print(response)
print(response["messages"][-1].content)

