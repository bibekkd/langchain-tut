from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

embeddings = MistralAIEmbeddings(model="mistral-embed")

texts = [
    "Apple makes very good computers",
    "Apple makes good phones",
    "I am fan of macbooks",
    "I like apples as a fruit too",
    "I like iPhones",
    "I like oranges",
    "I like Samsung phones",
    "I am fond of lenovo thinkpads"
]

vectorstore = FAISS.from_texts(texts, embeddings)

print(vectorstore.similarity_search("Pineapple is my favorite fruit", k=7))
print(vectorstore.similarity_search("I samsung ultrabook", k=7))
