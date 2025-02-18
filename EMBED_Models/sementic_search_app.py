from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os 
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
import numpy as np

os.environ["HF_HOME"] = "E:/LANG - Learning/HF_CACHE"
load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

doc = [
    "1. Imran Khan - Legendary all-rounder and captain who led Pakistan to its first World Cup victory in 1992.",
    "2. Wasim Akram - One of the greatest fast bowlers in cricket history, known as the 'Sultan of Swing'.",
    "3. Waqar Younis - Master of reverse swing and one of the most feared fast bowlers of his time.",
    "4. Inzamam-ul-Haq - Stylish batsman with exceptional timing and one of Pakistan's highest run-scorers.",
    "5. Shahid Afridi - Explosive all-rounder known for his big hitting and record-breaking fastest century."
]

async def main():
    # Your search query
    query = "tell me about Imran Khan"
    
    # Get embeddings
    doc_embed = await embeddings.aembed_documents(doc)
    query_embed = embeddings.embed_query(query)
    
    # Calculate similarities
    similarities = cosine_similarity([query_embed], doc_embed)[0]
    
    # Get top match
    top_indices = np.argsort(similarities)[::-1][:1]
    
    print(f"\nQuery: {query}")
    print("\nTop Matches:")
    print("-" * 50)
    
    # Print results
    for idx in top_indices:
        score = similarities[idx] * 100
        print(f"Similarity: {score:.2f}%")
        print(f"Document: {doc[idx]}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())