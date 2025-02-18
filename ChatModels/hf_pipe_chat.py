from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['HF_HOME'] = "E:/LANG - Learning/HF_CACHE"

# Load the pipeline
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task="text-generation",
    pipeline_kwargs=dict(
        temprature=0.9,
        max_length=100
    )
)

chatllm = ChatHuggingFace(llm=llm)

result = chatllm.invoke("Hello, how are you?")

print(result.content)