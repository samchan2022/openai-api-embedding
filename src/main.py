import os
import openai
import numpy as np
from dotenv import load_dotenv
from utils import get_embeddings, cosine_similarity, recommend

# Load environment variables from .env file
load_dotenv()

# Set your API key and API URL
openai.api_key = os.getenv('OPENAI_API_KEY')
api_url = os.getenv('OPENAI_API_URL')

# Sample texts to get embeddings
texts = [
    "I love playing football.",
    "Soccer is my favorite sport.",
    "I enjoy reading books.",
    "Cooking is a great way to relax."
]

# Get embeddings for the sample texts
embeddings = get_embeddings(texts, api_url)

# Input text for which to find similar recommendations
input_text = "I love reading novels."

# Get recommendations
recommended_texts = recommend(input_text, texts, embeddings, api_url, top_k=2)

# Print the results
print("Input text:", input_text)
print("Recommended texts:", recommended_texts)

