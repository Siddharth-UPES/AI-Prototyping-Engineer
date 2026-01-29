# Task 4 – AI System Architecture  


## 📌 Problem Statement
Design a **production-ready AI assistant** for enterprise internal use that allows employees to query uploaded PDFs or internal documents and get answers strictly grounded in the source material.  

The system must address:  
- Secure and efficient **data ingestion**  
- High-performance **semantic search**  
- **LLM orchestration** for accurate, context-aware answers  
- **Cost control** and optional API usage  
- **Monitoring, feedback, and evaluation** for real-world deployment  

---


## 📌 Overview


This task involves designing an AI assistant for internal enterprise use.  
The architecture emphasizes **reliability, transparency, and maintainability**, combining **document retrieval**, **LLM generation**, and **feedback loops** to ensure high-quality, source-grounded responses.

---

## 🔧 System Components

### 1. Data Ingestion Pipeline
- **Purpose:** Collect and process documents from various sources.  
- **Inputs:** PDFs, internal docs, emails, tickets, or CRM databases.  
- **Processing Steps:**  
  - Extract text  
  - Clean & transform  
  - Store chunks with metadata

### 2. Vector Database (FAISS)
- **Role:** Store embeddings for semantic search.  
- **Why FAISS?**  
  - Local, high-performance, and cost-efficient  
  - Supports cosine similarity and large-scale retrieval  
- **Stored Data:**  
  - Chunk embeddings  
  - Metadata (page numbers, document info)

### 3. LLM Orchestration
- **Models Supported:**  
  - Local: `FLAN-T5`  
  - API: OpenAI GPT (optional)
- **Responsibilities:**  
  - Inject context from retrieved chunks  
  - Use last 4 chat turns for memory  
  - Generate strict, grounded answers

### 4. Context Retrieval
- **Process:**  
  - User question → embedding → FAISS Top-K search  
  - Threshold filter to discard low-relevance chunks (`SIM_THRESHOLD = 0.35`)
- **Outcome:** Only high-confidence chunks are used for generation

### 5. Answer Generation
- **Prompt Engineering:**  
  - Injects retrieved context and chat memory  
  - Forces LLM to respond only using document information  
  - Fallback response:  
    > "The answer is not available in the document."
- **Output:** Clear, technical, multi-sentence answers with source snippet references

### 6. Feedback Loop & Memory
- **Memory:** Last 4 turns of chat are used for context continuity  
- **Feedback Loop:** Users can mark answers as Helpful / Not Helpful  
- **Purpose:** Improves response quality and mimics real enterprise assistant behavior

### 7. Cost Control & Monitoring
- **Cost Control:**  
  - Use local FLAN-T5 by default  
  - Optional OpenAI API  
  - Index persistence avoids repeated embedding computation
- **Monitoring & Evaluation:**  
  - Track query analytics, similarity scores, drift detection  
  - Maintain feedback log for quality evaluation

### 8. Deployment & UI
- **Interface:** Streamlit  
- **Features:**  
  - PDF upload  
  - Interactive Q&A  
  - Display top retrieved snippets and best source  
  - Optional document summarization

---

## 🔁 End-to-End Workflow

User → Upload PDF → Text Extraction → Chunking → Embeddings (MiniLM) → FAISS Vector Search → Context Injection → LLM Generation → Answer + Source Reference  

---

## 🖼️ System Diagram

![Enterprise AI System Architecture](AI TASK.png)

---

## 🎯 Key Design Choices

| Component         | Choice              | Reason                                         |
|------------------|------------------|-----------------------------------------------|
| LLM              | FLAN-T5 / OpenAI | Open-source, controllable, scalable          |
| Embeddings       | MiniLM           | Lightweight and fast                          |
| Vector DB        | FAISS             | Local, free, high-performance                 |
| Retrieval        | Top-K + Threshold | Reduces hallucinations, improves relevance    |
| UI               | Streamlit         | Rapid prototyping and interactive interface   |
| Memory & Feedback| Last 4 turns + feedback buttons | Conversational continuity, improvement loop |
| Cost Control     | Local model + optional API | Reduce dependency on paid APIs             |

---

## 📦 Deliverables

- Fully functional AI PDF Question Answering system  
- Modular, readable Python code  
- Enterprise-ready architecture design  
- Monitoring, feedback loop, and cost-control features  
- Assessment-ready README with diagram and trade-offs  

## 🚀 Deployment & Live Demo

- **Live Demo:** [Streamlit Cloud Link](https://ml-engineer-assessment-mt5l9ai3jmk82s2hqdndym.streamlit.app/)  
- **Local Run Instructions:**
```bash
git clone <repo-url>
cd task4_ai_pdf_qa
pip install -r requirements.txt
streamlit run app.py



  
