## 📌 Task Overview

This repository contains my submission for the **Practical Assessment – AI Prototyping Engineer**.  
The project focuses on building a **production-grade LLM-powered AI system** with strong emphasis on  
**retrieval grounding, hallucination control, modular design, and enterprise readiness**.

Each task is implemented with clear design justification and working code.

---

## 🧠 Task 1: LLM-Powered AI Prototype (Mandatory)

### Prototype Implemented
**Chat with PDFs (RAG-based Question Answering System)**

### Key Capabilities
- PDF ingestion and text extraction using PyPDF  
- Text chunking with overlap for semantic continuity  
- Dense embeddings using SentenceTransformers (MiniLM)  
- Vector storage and retrieval using FAISS  
- Context-aware answer generation using FLAN-T5 (local) or OpenAI (optional)  
- Interactive user interface built with Streamlit  

### Deliverables
- Fully working end-to-end prototype  
- Modular and readable code (`app.py`)  
- Clear explanation of design choices  

📄 **Detailed implementation and explanation available in this repository**

---

## 🛡️ Task 2: Hallucination & Quality Control

### Problem Addressed
LLMs may generate **confident but incorrect answers** when context is weak or missing.

### Guardrails Implemented
- **Similarity Threshold Guardrail**  
  Stops answer generation if retrieved chunks are not relevant enough.
- **Source-Grounded Prompt Constraint**  
  Forces the model to answer strictly from retrieved document content.

### Outcome
- Prevented hallucinated answers  
- Improved trust, transparency, and answer reliability  

📄 **Guardrail logic implemented directly in the RAG pipeline**

---

## ⚡ Task 3: Rapid Iteration Challenge

### Advanced Capability Added
**Chat Memory + Feedback Loop**

### Why This Choice
- Enables conversational, multi-turn Q&A  
- Mimics real enterprise AI assistant behavior  
- Allows qualitative evaluation of responses  

### Implementation
- Last 4 chat turns injected into the prompt  
- User feedback collected using Streamlit session state  

### Trade-offs
- Increased prompt length  
- Slight increase in response latency  

📄 **Memory and feedback logic implemented in `app.py`**

---

## 🏢 Task 4: AI System Architecture (Enterprise Design)

### System Designed
**Enterprise-Ready Internal AI Assistant**

### Architecture Covers
- Secure document ingestion  
- Embedding and vector database selection (FAISS)  
- Modular LLM orchestration (local + API-based)  
- Cost control via local inference and index persistence  
- Monitoring via feedback and similarity scores  
- Future-ready retraining and scalability strategy  

📄 **Architecture explanation included in README with system flow**

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- SentenceTransformers  
- FAISS  
- Hugging Face Transformers (FLAN-T5)  
- OpenAI API (optional)  
- NumPy, PyPDF  

---

## 🗂️ Notes

- Each task maps directly to assessment requirements  
- Code is modular, readable, and reproducible  
- The system avoids black-box AutoML tools  
- Designed with real-world deployment constraints in mind  

---

## 👤 Author

**Siddharth Kumar**  
MCA (AI & ML)  
AI Prototyping Engineer | RAG Systems | LLM Applications


