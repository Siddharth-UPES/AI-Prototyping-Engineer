# 📘 AI PDF Question Answering System (RAG Pipeline)

## Overview

This project implements an **AI-powered PDF Question Answering system** using a **Retrieval Augmented Generation (RAG)** pipeline.

Users can upload a PDF, ask natural language questions, and receive answers that are **strictly grounded in the document content**.  
The system retrieves relevant text chunks using **FAISS**, generates answers using **FLAN-T5 (local)** or **OpenAI (optional)**, highlights the most reliable source snippet, maintains chat memory, and collects user feedback.

The solution is built primarily using **open-source technologies**, making it cost-efficient and suitable for **enterprise internal deployments**.

---

## ✨ Key Features

- PDF ingestion and text extraction  
- Chunking with overlap for semantic continuity  
- SentenceTransformer embeddings (MiniLM)  
- FAISS vector database (local)  
- Answer generation using FLAN-T5 (local) or OpenAI (optional)  
- Best-snippet selection using similarity score  
- Chat memory (multi-turn Q&A)  
- Feedback loop (Helpful / Not Helpful)  
- Similarity-threshold guardrails to reduce hallucinations  
- Local index persistence  
- Optional document summarization  

---

## 🏗️ High-Level Architecture

### Document Processing Flow
User
→ Streamlit UI
→ PDF Upload
→ Text Extraction (PyPDF)
→ Chunking
→ Embeddings (MiniLM)
→ FAISS Vector Database


### Question Answering Flow
User Question
→ Question Embedding
→ FAISS Similarity Search (Top-K + Threshold)
→ Context + Chat Memory Injection
→ LLM (FLAN-T5 / OpenAI)
→ Answer + Best Snippet + Feedback



---

## 🧠 TASK 1 – LLM Powered Prototype

### Prototype Chosen
**Chat with PDFs**

### Components

#### LLM
- `google/flan-t5-small` (local, default)
- OpenAI GPT models (optional)

#### RAG Stack
- SentenceTransformers (`all-MiniLM-L6-v2`)
- FAISS (`IndexFlatIP` with cosine similarity)

#### Chunking Strategy
- Chunk size: `500`
- Overlap: `100`

#### Prompt Engineering
- Retrieved context injected into the prompt
- Page references included
- Strict answer constraints enforced
- Chat history appended for continuity

#### User Interface
- Streamlit

### Design Choices

- **MiniLM**: Lightweight, fast, CPU-friendly embeddings  
- **FAISS**: Free, local, high-performance vector search  
- **Streamlit**: Rapid prototyping with minimal boilerplate  
- **FLAN-T5**: Fully open-source and controllable generation model  

---

## 🛡️ TASK 2 – Hallucination & Quality Control

### Causes of Hallucination

1. Weak similarity matches during retrieval  
2. LLM prior knowledge overriding document content  
3. Missing information in the uploaded PDF  
4. Over-short or unconstrained answers  

---

### Guardrails Implemented

#### Guardrail 1 – Similarity Threshold

Low-relevance chunks are discarded before answer generation.

### Behavior

- Chunks below the similarity threshold are ignored  
- If no chunk crosses the threshold, answer generation is skipped  
- The user is asked to rephrase the question  

This mechanism prevents **low-confidence, irrelevant, or fabricated answers**.

---

## 🧱 Guardrail 2 – Source-Grounded Prompt Constraint

The prompt strictly enforces **document-only answering**.

### Prompt Rule

> If the answer is not present, say:  
> **"The answer is not available in the document."**

This ensures the model does **not hallucinate** when relevant information is missing from the PDF.

---

## 📊 Improved Response Behavior

### Without Guardrails

- The LLM may generate a **confident but incorrect explanation**

### With Guardrails Enabled

- **"The answer is not available in the document."**

Responses are **transparent, grounded, and source-aware**.

⚡ TASK 3 – Rapid Iteration Challenge
Advanced Capability Added
Chat Memory + Feedback Loop
Why This Choice
Improves conversational continuity
Enables follow-up questions
Mimics real enterprise assistant behavior
Implementation
Last 4 chat turns injected into the prompt
User feedback stored using Streamlit session state
Trade-offs
Increased prompt length
Slight latency increase
Limitations
Memory is session-based (not persistent across restarts)


🏢 TASK 4 – AI System Architecture (Enterprise Design)
Architecture Overview
User
 │
 ▼
Streamlit UI
 │
 ▼
PDF Ingestion & Parsing
 │
 ▼
Chunking Engine
 │
 ▼
Embedding Model (MiniLM)
 │
 ▼
FAISS Vector Database
 │
 ▼
Top-K Retrieval + Thresholding
 │
 ▼
Prompt Builder (Context + Memory)
 │
 ▼
LLM (FLAN-T5 / OpenAI)
 │
 ▼
Answer + Source Attribution

Enterprise Considerations
Data Ingestion
Secure PDF upload
Chunk-level metadata with page references
Vector Database Choice
FAISS for local, cost-efficient deployment
Easily replaceable with Pinecone / Weaviate
LLM Orchestration
Modular generation layer
Supports open-source and API-based models
Cost Control
Local embeddings and indexing
Optional OpenAI usage
Index persistence avoids re-embedding
Monitoring & Evaluation
User feedback collection
Similarity score inspection
Source attribution for trust validation
