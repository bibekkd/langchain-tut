from base64 import b64encode
from langchain.chat_models.base import init_chat_model
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("pixtral-12b-2409", model_provider="mistralai")

# Read and encode the image
with open('image.jpg', 'rb') as image_file:
    base64_image = b64encode(image_file.read()).decode('utf-8')

message = HumanMessage(
    content=[
        {"type": "text", "text": "Describe the content of this image"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        }
    ]
)

response = model.invoke([message])
print(response.content)

# you can two types image input 
# 1. base64 encoded image
# 2. image url
# #         {
#             "type": "image_url",
#             "image_url": {"url": "https://i.ytimg.com/vi/90uYEDEStQI/hq720.jpg?sqp=-oaymwEhCK4FEIIDSFryq4qpAxMIARUAAAAAGAElAADIQj0AgKJD&rs=AOn4CLDRnJYUbwVyW1wBmxM_TbUpsfD12g"}
#         },