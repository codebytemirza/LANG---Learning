from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os


load_dotenv()
os.environ["HF_HOME"] = "E:/LANG - Learning/HF_CACHE"


embedings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "Muhammad Abdullah"

vector = embedings.embed_query(text=text)

print(str(vector))