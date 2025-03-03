from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the LLM
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.9)

# Create chat template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful storyteller who creates engaging stories on specific topics"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])

def load_chat_history():
    chat_history = []
    try:
        with open('chat_history.txt', 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                if i + 1 < len(lines):
                    chat_history.append(("human", lines[i].strip().replace("User: ", "")))
                    chat_history.append(("assistant", lines[i+1].strip().replace("Assistant: ", "")))
    except FileNotFoundError:
        print("No chat history file found. Starting fresh conversation.")
    return chat_history

def save_interaction(query, response):
    with open('chat_history.txt', 'a') as f:
        f.write(f"User: {query}\n")
        f.write(f"Assistant: {response}\n")

def chat():
    # Load existing chat history
    chat_history = load_chat_history()
    
    print("Welcome to the storyteller AI! Type 'exit' to end the conversation.")
    
    while True:
        # Get user input
        user_query = input("\nYou: ")
        
        # Check if user wants to exit
        if user_query.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye! Thank you for chatting.")
            break
        
        # Create and invoke the chain
        chain = chat_template | llm
        response = chain.invoke({
            "chat_history": chat_history,
            "query": user_query
        })
        
        # Print the response
        print(f"\nAI: {response.content}")
        
        # Add the interaction to chat history
        chat_history.append(("human", user_query))
        chat_history.append(("assistant", response.content))
        
        # Save the interaction
        save_interaction(user_query, response.content)

if __name__ == "__main__":
    # Clear terminal for better user experience
    os.system('cls' if os.name == 'nt' else 'clear')
    chat()