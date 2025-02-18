from langchain_google_genai import GoogleGenerativeAI as GoogleGenAI
from dotenv import load_dotenv

load_dotenv()

# Create a new instance of the GoogleGenAI class

genai = GoogleGenAI(model="gemini-1.5-pro")

results = genai.invoke("Hello, my name is Muhammad Abdullah. I am an AI/ML Developer. I am currently working on a project that involves natural language processing. I am using a language model to generate text based on the input I provide. I am excited to see what I can create with this technology.")

print(results)