import pytest
from create_db import main
from query import query_rag
from langchain_community.llms.ollama import Ollama
from langchain.vectorstores import Chroma

EVAL_PROMPT = "Evaluate the response: Expected: {expected_response}, Actual: {actual_response}"  # Define the prompt


@pytest.fixture(scope="module")
def db():
    """Initialize and populate the vector database before running tests."""
    print("Setting up test environment...")
    main()


def test_database_population():
    """Ensure documents are successfully added to the vector database."""
    db = Chroma(persist_directory="chroma")  # Load DB
    assert db._collection.count() > 0, "Database should contain at least one document."


def test_query_response():
    """Ensure the query function returns relevant results."""
    query_1 = "What is the company name?"
    response_1 = query_rag(query_1)

    assert isinstance(response_1, str) and len(response_1) > 0, "Response should be a non-empty string."
    assert query_and_validate(query_1, "ConocoPhillips"), "Query validation failed."

    query_2 = "This document speaks about US Mayor office?"
    response_2 = query_rag(query_2)

    assert isinstance(response_2, str) and len(response_2) > 0, "Response should be a non-empty string."
    assert query_and_validate(query_2, "ConocoPhillips"), "Query validation failed."


def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="mistral")
    evaluation_results_str = model.invoke(prompt)  # Use `generate()` instead of `invoke()`
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True
    elif "false" in evaluation_results_str_cleaned:
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False
    else:
        raise ValueError("Invalid evaluation result. Cannot determine if 'true' or 'false'.")


@pytest.fixture(scope="module", autouse=True)
def cleanup():
    """Cleanup resources after testing."""
    yield
    print("Cleaning up test environment...")
