# 🚀 RAG-based Document Search API

This repository contains a **Retrieval-Augmented Generation (RAG) API** that allows querying a vector database (**ChromaDB**) to retrieve relevant document chunks and generate responses using **Ollama (Mistral)**.  

---

## 📌 Features
✅ **FastAPI-based API** for document retrieval and comparison  
✅ **Embeddings with Ollama (`nomic-embed-text`)** for vector search  
✅ **PDF document loading and chunking** with `PyPDFDirectoryLoader`  
✅ **Cosine similarity-based document comparison**  
✅ **Automated testing with `pytest`**  

---

## 🛠️ Installation

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/smshelar/rag_pipeline.git
cd your-repo
```

### 2️⃣ **Create a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### 3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4️⃣ **Start the API**
```bash
uvicorn rag_api:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🚀 API Endpoints

### 🔍 **1. Query Documents (RAG)**
**Endpoint:**  
```http
POST /query/
```
**Request Body:**
```json
{
  "query_text": "What is the company name?"
}
```
**Response:**
```json
{
  "response": "The company name is ConocoPhillips.",
  "sources": ["document_1.pdf(page_num:chunk_num)", 
              "document_2.pdf(page_num:chunk_num)", 
              "document_3.pdf(page_num:chunk_num)"]
}
```

---

### 🔄 **2. Compare Two Documents**
**Endpoint:**  
```http
POST /compare/
```
**Request Body:**
```json
{
  "query_1": "Impact of climate change",
  "query_2": "Rising sea levels"
}
```
**Response:**
```json
{
  "query_1": "Impact of climate change",
  "query_2": "Rising sea levels",
  "similarity_score": 0.87,
  "source_1": "doc1.pdf",
  "source_2": "doc2.pdf"
}
```

---

### 📂 **3. Populate the Database**
**Endpoint:**  
```http
POST /populate/
```
**Request Body:**
```json
{
  "reset": true
}
```
**Response:**
```json
{
  "message": "Database populated with 100 chunks"
}
```

---

## 🧪 Running Tests
Run all tests using:
```bash
pytest test.py
```

---

## 🏗️ Folder Structure

```
📁 your-repo
│-- 📂 data/                  # Directory for PDFs
│-- 📂 chroma/                # ChromaDB storage
│-- 📜 embedding_function.py   # Ollama embedding function
│-- 📜 query.py                # Query processing
│-- 📜 compare_embeddings.py    # Document similarity comparison
│-- 📜 load_model.py            # Data pipeline for ChromaDB
│-- 📜 rag_api.py               # FastAPI server
│-- 📜 test.py                  # Pytest-based tests
│-- 📜 requirements.txt         # Dependencies
│-- 📜 README.md                # Project Documentation
```

---

## 🔗 References
- [LangChain Docs](https://python.langchain.com/)
- [Ollama Models](https://ollama.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [ChromaDB](https://github.com/chroma-core/chroma)

---

## 🎯 Future Enhancements
- 🔹 **Dockerization for deployment**  
- 🔹 **Support for more document formats (TXT, DOCX)**  
- 🔹 **Advanced ranking using LLM-generated summaries**  

---
🚀 **Developed with ❤️ using Python, LangChain & FastAPI**



### 🔥 **Want Any Custom Changes?**
- ✅ Add **Docker setup**?  
- ✅ Include **environment variables (`.env`)**?  
- ✅ Create a **GitHub Actions CI/CD pipeline**?  

Let me know what you need! 🚀🔥
