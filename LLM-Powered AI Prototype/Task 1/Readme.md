# TASK 1 – LLM-Powered AI Prototype  
## Chat with PDFs (RAG-Based System)

---

## Overview

This task implements a **working LLM-powered AI prototype** using a  
**Retrieval Augmented Generation (RAG)** architecture.

The chosen prototype is **Chat with PDFs**, where users can upload a PDF document,  
ask natural language questions, and receive answers that are **strictly grounded in the document content**.

The system combines **document retrieval** with **LLM-based answer generation** to ensure accuracy,  
reduce hallucinations, and maintain transparency.

---

## Prototype Selected

**Chat with PDFs**

Other options provided in the task:
- Resume screening assistant  
- Internal knowledge bot  
- AI form auto-filler  

---

## System Components

### Large Language Model (LLM)

- **Local (Default):** `google/flan-t5-small`
- **Optional:** OpenAI GPT models  

**Why this choice?**
- FLAN-T5 is open-source and cost-efficient  
- Suitable for controlled, document-grounded generation  
- OpenAI option allows easy scalability if required  

---

### Retrieval Augmented Generation (RAG)

- **Embedding Model:** `all-MiniLM-L6-v2`
- **Vector Database:** FAISS (local, in-memory + persisted)

**Why RAG?**
- Prevents hallucinations by grounding answers in documents  
- Enables scalable and efficient document search  
- Separates retrieval from generation for a modular architecture  

---

### Chunking Strategy

- **Chunk size:** 500 characters  
- **Overlap:** 100 characters  

**Why chunking with overlap?**
- Preserves semantic continuity across chunks  
- Improves retrieval accuracy for long documents  
- Prevents loss of context at chunk boundaries  

---

### Prompt Engineering

The prompt enforces **strict document grounding**:

- Retrieved context is injected directly into the prompt  
- Clear instructions restrict the model from using external knowledge  
- A fixed fallback response is used when the answer is missing  

**Example Constraint:**
> *If the answer is not present, say:*  
> **"The answer is not available in the document."**

This ensures reliable, transparent, and trustworthy responses.

---

###  User Interface

- **Framework:** Streamlit  

**Why Streamlit?**
- Rapid prototyping with minimal boilerplate  
- Interactive UI for PDF upload and Q&A  
- Ideal for demos and academic evaluation  

---

## End-to-End Workflow

User  
→ Upload PDF  
→ Text Extraction  
→ Chunking  
→ Embeddings (MiniLM)  
→ FAISS Vector Search  
→ Context Injection  
→ LLM Generation  
→ Answer with Source Reference  

---

##  Design Choices Summary

| Component | Choice | Reason |
|---------|-------|--------|
| LLM | FLAN-T5 | Open-source, controllable |
| Embeddings | MiniLM | Lightweight, fast |
| Vector DB | FAISS | Free, local, high-performance |
| UI | Streamlit | Rapid development |
| Architecture | RAG | Hallucination control |

---

##  Deliverables

- Fully functional LLM-powered prototype  
- Modular and readable Python code  
- Clear explanation of design decisions  
- Assessment-ready documentation  


