"""
Provides utils to access openai api
"""

import os
import openai
from dotenv import load_dotenv

# from utils import get_embeddings, cosine_similarity, recommend
# from .utils.utils import get_embeddings,  recommend
# from openai.client import get_embeddings, recommend
from .openai.client import OpenAIClient

# Load environment variables from .env file
load_dotenv()

# Set your API key and API URL
api_url = os.getenv("OPENAI_API_URL")

# Sample texts to get embeddings
texts = [
    "I love playing football.",
    "Soccer is my favorite sport.",
    "I enjoy reading books.",
    "Cooking is a great way to relax.",
]

# Get embeddings for the sample texts
embeddings = get_embeddings(texts, api_url)

# Input text for which to find similar recommendations
INPUT_TEXT = "I love reading novels."

# Get recommendations
recommended_texts = recommend(INPUT_TEXT, texts, embeddings, api_url, top_k=2)

# Print the results
print("Input text:", INPUT_TEXT)
print("Recommended texts:", recommended_texts)
