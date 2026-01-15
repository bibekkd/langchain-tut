from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool
from langchain.agents import create_agent
from langchain_groq import ChatGroq

load_dotenv()

embeddings = MistralAIEmbeddings(model="mistral-embed")

texts = [
    'I love apples',
    'I enjoy oranges',
    'I think pears tastes very good.',
    'I dislike bananas',
    'I hate mangoes',
    'I love linux',
    'I like macos',
    'I hate windows'
]

vectorstore = FAISS.from_texts(texts, embeddings)

print(vectorstore.similarity_search("what fruit do the person like", k=7))
print(vectorstore.similarity_search("what os do the person dislike", k=7))


retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

retriever_tool = create_retriever_tool(retriever, "kb_search", "Search the small product/fruit knowledge base for relevant information")

# Create Groq model instance
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

agent = create_agent(
    model=llm,
    tools=[retriever_tool],
    system_prompt="You are a helpful assistant for question about macos, linux, windows, apple, google, samsung, etc. First call the kb_search tool to get relevant information from the knowledge base. Then answer succinctly and concisely. You may have to use kb_search tool multiple times to get the answer."
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "what os do the person dislike and what fruit do the person like"
        }
    ]
})

print(result)
print(result['messages'][-1].content)