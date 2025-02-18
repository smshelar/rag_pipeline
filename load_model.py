from create_db import *


def clear_database():
    """Delete existing ChromaDB directory."""
    print("Clearing Database...")
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


def get_list_of_documents():
    file_list = os.listdir(DATA_PATH)
    return file_list


def database_pipeline(file_list):
    # Create (or update) the data store.
    documents = load_documents(file_list)
    chunks = split_documents(documents)
    add_to_chroma(chunks, True)
