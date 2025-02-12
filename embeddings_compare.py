import numpy as np
from langchain.vectorstores import Chroma
from embedding_function import get_embedding_function
from sklearn.metrics.pairwise import cosine_similarity

CHROMA_PATH = "chroma"


def load_chroma_db():
    """Load the existing ChromaDB"""
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())


def get_embedding(text):
    """Generate embedding for a given text."""
    embedding_model = get_embedding_function()
    return embedding_model.embed_query(text)


def compare_documents(query_1: str, query_2: str):
    """Compare two document chunks based on their embeddings."""
    db = load_chroma_db()

    # Retrieve embeddings from ChromaDB
    results_1 = db.similarity_search(query_1, k=1)
    results_2 = db.similarity_search(query_2, k=1)

    if not results_1 or not results_2:
        print("❌ Could not retrieve one or both document embeddings.")
        return None

    embedding_1 = get_embedding(results_1[0].page_content)
    embedding_2 = get_embedding(results_2[0].page_content)

    # Compute cosine similarity
    similarity = cosine_similarity([embedding_1], [embedding_2])[0][0]

    return {
        "query_1": query_1,
        "query_2": query_2,
        "similarity_score": similarity,
        "source_1": results_1[0].metadata,
        "source_2": results_2[0].metadata,
    }


if __name__ == "__main__":
    query_1 = "Analyze the potential environmental impact of the proposed development, specifically focusing on the impact on the Douglas River?"
    query_2 = "increased flood risk"

    comparison_result = compare_documents(query_1, query_2)

    if comparison_result:
        print("\n🔍 **Comparison Results**:")
        print(f"📄 Query 1: {comparison_result['query_1']}")
        print(f"📄 Query 2: {comparison_result['query_2']}")
        print(f"✅ Similarity Score: {comparison_result['similarity_score']:.4f}")
        print(f"📌 Source 1: {comparison_result['source_1']}")
        print(f"📌 Source 2: {comparison_result['source_2']}")