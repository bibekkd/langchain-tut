from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="mistral-medium-2508",
    model_provider="mistralai",
    temperature=0.1,
)

# conversation = [
#     SystemMessage("You are a helpful assistant for question regarding programming."),
#     HumanMessage("Hello, what is langchain , tell me in detail."),
#     AIMessage("Langchain is a framework for building language models."),
#     HumanMessage("Can you tell me more about it?")
# ]


# response = model.invoke(conversation)

# response = model.invoke("Hello, what is langchain , tell me in detail.")

# showing content in normal manner

# print(response)
# print(f"\n \n \n {response.content}")

# showing content in streaming manner
for chunk in model.stream("Hello, what is langchain , tell me in detail."):
    print(chunk.content, end="", flush=True)

