from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Define the ChatGroq object

chat_groq = ChatGroq(model="deepseek-r1-distill-llama-70b" , temperature=0.5, max_tokens=100)

results = chat_groq.invoke("What is the capital of France?")

print(results.content)