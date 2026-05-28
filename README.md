# 📚 Enterprise RAG Chatbot with Evaluation Harness

A production-grade Retrieval-Augmented Generation (RAG) system built using LangChain, ChromaDB, Ollama, Phi-3, and Streamlit.

This project enables intelligent PDF-based question answering using semantic retrieval and local LLM inference while also integrating an evaluation harness to measure retrieval quality and hallucination detection.

---

# 🚀 Key Features

## 📄 Intelligent PDF Question Answering

* Upload PDFs dynamically
* Ask natural language questions
* Context-aware semantic retrieval
* Local LLM-powered responses

---

## 🧠 Advanced RAG Pipeline

* Recursive text chunking
* Vector embeddings using HuggingFace
* ChromaDB vector storage
* MMR-based retrieval
* Context injection prompting
* Retrieval-Augmented Generation architecture

---

## 📊 Evaluation Harness (Major Differentiator)

Integrated evaluation pipeline for measuring RAG quality using:

* Faithfulness
* Answer Relevancy
* Context Precision
* Hallucination Detection

This simulates real-world enterprise AI evaluation workflows.

---

# 🛠️ Tech Stack

## AI / LLM

* LangChain
* Ollama
* Phi-3
* HuggingFace Embeddings
* RAG Architecture

---

## Vector Database

* ChromaDB

---

## Frontend

* Streamlit

---

## Backend / Processing

* Python
* Recursive Text Splitters
* Semantic Search
* Prompt Engineering

---

## Evaluation

* RAGAS
* Retrieval Evaluation
* Hallucination Analysis

---

# 🏗️ System Architecture

```text
User Query
    ↓
PDF Upload
    ↓
Document Loader
    ↓
Text Chunking
    ↓
Embedding Generation
    ↓
Chroma Vector Database
    ↓
Retriever (MMR Search)
    ↓
LLM (Phi-3 via Ollama)
    ↓
Generated Answer
    ↓
Evaluation Harness
```

---

# 📂 Project Structure

```bash
production-rag-eval/

│
├── backend/
│
├── frontend/
│   └── app.py
│
├── evaluation/
│   └── evaluation.py
│
├── docs/
│
├── chroma_db/
│
├── requirements.txt
│
└── README.md
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/production-rag-eval.git
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

---

## 3️⃣ Activate Environment

### Windows

```bash
venv\Scripts\activate
```

---

## 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🤖 Install Ollama

Download Ollama:

https://ollama.com/download

Pull Phi-3 model:

```bash
ollama pull phi3
```

Run model:

```bash
ollama run phi3
```

---

# ▶️ Run Frontend Application

Open another terminal:

```bash
cd frontend
streamlit run app.py
```

---

# 📊 Run Evaluation Harness

Open separate terminal:

```bash
cd evaluation
python evaluation.py
```

---

# 📈 Evaluation Metrics

| Metric            | Purpose                     |
| ----------------- | --------------------------- |
| Faithfulness      | Checks hallucination level  |
| Answer Relevancy  | Measures answer quality     |
| Context Precision | Evaluates retrieval quality |

---

# 🔥 Enterprise-Level Concepts Implemented

* Production-style RAG architecture
* Local LLM deployment
* Semantic vector search
* Retrieval optimization using MMR
* Evaluation harness separation
* Context-aware prompting
* Hallucination analysis
* Vector database management

---

# 📸 Demo

Upload PDFs and ask intelligent questions using local LLM inference with enterprise RAG architecture.

---

# 👨‍💻 Author

Purvisha Desani

M.Sc. Software Engineering
Hochschule Heilbronn, Germany

---

# ⭐ Future Improvements

* Multi-PDF support
* Conversation memory
* Hybrid search
* Re-ranking models
* Docker deployment
* Kubernetes scaling
* Authentication layer
* API deployment

---
