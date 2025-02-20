import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from embedding_function import get_embedding_function
__import__("pysqtite3")
import sys
sys.modules['sqlite3']= sys.modules.pop( 'pysqlite3')
from langchain.vectorstores.chroma import Chroma
from tqdm import tqdm
import streamlit as st

CHROMA_PATH = "chroma"
DATA_PATH = "docs"


def main():
    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents(file_dir=DATA_PATH):
    # Load PDFs from the specified directory
    document_loader = PyPDFDirectoryLoader(file_dir)
    return document_loader.load()


def split_documents(documents: list[Document]):
    # Split documents into chunks to optimize embeddings
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document], streamlit_flag=False):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Assign unique IDs to document chunks
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Retrieve existing document IDs from the database
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Filter out chunks that are already in the database
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        if streamlit_flag:
            st.toast(f"⚙️ Adding new documents: {new_chunks}", icon="⏳")
        else:
            print(f"👉 Adding new documents: {len(new_chunks)}")

        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        # Add new chunks to the database
        for chunk in tqdm(new_chunks, desc="Creating embeddings"):
            db.add_documents([chunk], ids=[chunk.metadata["id"]])
        db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index
    # Generate unique IDs for each document chunk based on file and page number
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        # Increment index if the page is the same as the last one
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    # Remove the existing database if it exists
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
