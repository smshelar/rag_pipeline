from langchain_community.embeddings.ollama import OllamaEmbeddings


# import os
# api_key = os.environ['OPENAI_API_KEY']

def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings
