from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_groq import ChatGroq

@dataclass
class Context:
    user_role: str

def get_system_prompt(user_role: str) -> str:
    """Generate system prompt based on user role"""
    base_prompt = "You are a helpful and very concise assistant."
    
    match user_role:
        case "expert":
            return f'{base_prompt} Provide a detailed technical response.'
        case "beginner":
            return f'{base_prompt} Keep your response simple and easy to understand.'
        case "child":
            return f'{base_prompt} Explain everything in a way such that you are talking to a child.'
        case _:
            return base_prompt

def create_role_based_agent(user_role: str):
    """Create an agent with a role-specific system prompt"""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )
    
    agent = create_agent(
        model=llm,
        tools=[],
        system_prompt=get_system_prompt(user_role)
    )
    
    return agent

def ask_question(user_role: str, question: str):
    """Ask a question with a specific user role"""
    agent = create_role_based_agent(user_role)
    
    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    })
    
    print(f"\n{'='*60}")
    print(f"User Role: {user_role.upper()}")
    print(f"Question: {question}")
    print(f"{'='*60}")
    print(response['messages'][-1].content)
    print(f"{'='*60}\n")

# Example: Ask the same question with different roles
question = "Explain PCA"

print("\n" + "="*60)
print("DEMONSTRATING DYNAMIC PROMPTS WITH DIFFERENT USER ROLES")
print("="*60)

# Beginner explanation
ask_question("beginner", question)

# Expert explanation
ask_question("expert", question)

# Child explanation
ask_question("child", question)