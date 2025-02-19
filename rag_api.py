from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import shutil
from langchain.vectorstores import Chroma
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from embedding_function import get_embedding_function

# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "data"

# Initialize FastAPI app
app = FastAPI()

# Load the vector database
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


# -------------------------------
# 📌 API Models
# -------------------------------
class QueryRequest(BaseModel):
    query_text: str


class CompareRequest(BaseModel):
    query_1: str
    query_2: str


class PopulateDBRequest(BaseModel):
    reset: bool = False


# -------------------------------
# 📌 Database Population
# -------------------------------
def clear_database():
    """Delete existing ChromaDB directory."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def load_documents():
    """Load PDFs from the data directory."""
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def populate_database(reset=False):
    """Populate ChromaDB with documents."""
    if reset:
        print("✨ Clearing and Resetting Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    db.add_documents(chunks)
    db.persist()
    return {"message": f"Database populated with {len(chunks)} chunks"}


@app.post("/populate/")
async def populate_db(request: PopulateDBRequest):
    """Endpoint to populate the database with documents."""
    return populate_database(reset=request.reset)


# -------------------------------
# 📌 Query Processing (RAG)
# -------------------------------
@app.post("/query/")
async def query_rag(request: QueryRequest):
    """Search for relevant documents and generate a response."""
    # Search the DB
    results = db.similarity_search_with_score(request.query_text, k=5)

    if not results:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=request.query_text)

    print(f"Generated prompt: {prompt}")

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", "Unknown") for doc, _ in results]

    return {
        "response": response_text,
        "sources": sources
    }


# -------------------------------
# 📌 Compare Embeddings
# -------------------------------
def get_embedding(text):
    """Generate embedding for a given text."""
    embedding_model = get_embedding_function()
    return embedding_model.embed_query(text)


@app.post("/compare/")
async def compare_documents(request: CompareRequest):
    """Compare two documents based on embeddings."""

    results_1 = db.similarity_search(request.query_1, k=1)
    results_2 = db.similarity_search(request.query_2, k=1)

    if not results_1 or not results_2:
        raise HTTPException(status_code=404, detail="One or both queries did not match any document.")

    # Generate embeddings for comparison
    embedding_1 = get_embedding(results_1[0].page_content)
    embedding_2 = get_embedding(results_2[0].page_content)

    # Compute cosine similarity
    similarity = cosine_similarity([embedding_1], [embedding_2])[0][0]

    return {
        "query_1": request.query_1,
        "query_2": request.query_2,
        "similarity_score": round(similarity, 4),
        "source_1": results_1[0].metadata,
        "source_2": results_2[0].metadata,
    }

# -------------------------------
# ✅ Run FastAPI Server
# -------------------------------
# Start the server: uvicorn rag_api:app --host 0.0.0.0 --port 8000 --reload
