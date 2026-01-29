# AI PDF Question Answering System (RAG Pipeline)

## Overview

This project implements an AI-powered PDF Question Answering system using a Retrieval Augmented Generation (RAG) pipeline.

Users can upload a PDF, ask natural language questions, and receive answers grounded strictly in the document. The system retrieves relevant chunks using FAISS, generates answers using FLAN-T5 (or optional OpenAI), highlights the most reliable source snippet, maintains chat memory, and collects user feedback.

The stack is primarily open-source, making it cost-efficient and suitable for enterprise internal deployments.

---

## Key Features

• PDF ingestion and text extraction  
• Chunking with overlap for semantic continuity  
• SentenceTransformer embeddings (MiniLM)  
• FAISS vector database (local)  
• Answer generation using FLAN-T5 (local) or OpenAI (optional)  
• Best-snippet selection using similarity score  
• Chat memory (multi-turn Q&A)  
• Feedback loop (Helpful / Not Helpful)  
• Similarity threshold guardrails to reduce hallucinations  
• Local index persistence  
• Optional document summarization  

---

## High-Level Architecture

User → Streamlit UI  
→ PDF Upload  
→ Text Extraction (PyPDF)  
→ Chunking  
→ Embeddings (MiniLM)  
→ FAISS Vector DB  

When a question is asked:

Question → Embedding  
→ FAISS Similarity Search (Top-K + Threshold)  
→ Context + Chat Memory Injection  
→ LLM (FLAN-T5 / OpenAI)  
→ Answer + Best Snippet + Feedback  

---

## TASK 1 – LLM Powered Prototype

### Prototype Chosen
Chat with PDFs

### Components

LLM  
• google/flan-t5-small (local, default)  
• OpenAI (optional)

RAG  
• SentenceTransformers (all-MiniLM-L6-v2)  
• FAISS (IndexFlatIP)

Chunking  
• Chunk size: 500  
• Overlap: 100  

Prompt Engineering  
• Retrieved context injected  
• Page references included  
• Technical constraints applied  
• Chat history appended  

UI  
• Streamlit

### Design Choices

• MiniLM: lightweight, fast, CPU-friendly  
• FAISS: free, local, high-performance vector search  
• Streamlit: rapid prototyping with minimal boilerplate  
• FLAN-T5: fully open-source generation model  

---

## TASK 2 – Hallucination & Quality Control

### Causes of Hallucination

1. Weak similarity matches  
2. LLM prior knowledge overriding document content  
3. Missing information in PDF  
4. Over-short or unconstrained answers  

---

### Guardrails Implemented

#### Guardrail 1 – Similarity Threshold

Weak chunks are discarded:

```python
SIM_THRESHOLD = 0.35
