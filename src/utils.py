import openai
import numpy as np
from numpy.linalg import norm


def get_embeddings(texts, api_url):
    """
    Get embeddings for a list of texts using OpenAI's embedding model.

    Parameters:
        texts (list of str): List of texts to embed.
        api_url (str): URL of the OpenAI embedding API.

    Returns:
        list of list of float: List of embeddings for the input texts.
    """
    response = openai.Embedding.create(
        input=texts,
        model="text-embedding-ada-002",  # Specify the embedding model
        api_base=api_url,
    )
    embeddings = [result["embedding"] for result in response["data"]]
    return embeddings


def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.

    Parameters:
        vec1 (list of float): First vector.
        vec2 (list of float): Second vector.

    Returns:
        float: Cosine similarity between the vectors.
    """
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def recommend(input_text, texts, embeddings, api_url, top_k=1):
    """
    Recommend texts similar to the input text.

    Parameters:
        input_text (str): The text for which to find recommendations.
        texts (list of str): List of original texts.
        embeddings (list of list of float): List of embeddings corresponding to the texts.
        api_url (str): URL of the OpenAI embedding API.
        top_k (int): Number of top similar texts to return.

    Returns:
        list of str: List of recommended texts.
    """
    # Get embedding for the input text
    input_embedding = get_embeddings([input_text], api_url)[0]

    # Calculate similarities
    similarities = [cosine_similarity(input_embedding, emb) for emb in embeddings]

    # Get the indices of the top_k similar texts
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]

    # Return the most similar texts
    return [texts[i] for i in top_k_indices]
