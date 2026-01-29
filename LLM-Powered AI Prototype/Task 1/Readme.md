# TASK 1 – LLM-Powered AI Prototype  
## Chat with PDFs (RAG-Based System)

---

## 📌 Overview

This task implements a **working LLM-powered AI prototype** using a  
**Retrieval Augmented Generation (RAG)** architecture.

The chosen prototype is **Chat with PDFs**, where users can upload a PDF document,
ask natural language questions, and receive answers that are **strictly grounded in the document content**.

The system combines **document retrieval** with **LLM-based answer generation** to ensure accuracy,
reduce hallucinations, and maintain transparency.

---

## ✅ Prototype Selected

**Chat with PDFs**

Other options provided in the task:
- Resume screening assistant  
- Internal knowledge bot  
- AI form auto-filler  

---

## ⚙️ System Components

### 🧠 Large Language Model (LLM)

- **Local (Default):** `google/flan-t5-small`
- **Optional:** OpenAI GPT models

**Why this choice?**
- FLAN-T5 is open-source and cost-efficient  
- Suitable for controlled, document-grounded generation  
- OpenAI option allows easy scalability if needed  

---

### 🔍 Retrieval Augmented Generation (RAG)

- **Embedding Model:** `all-MiniLM-L6-v2`
- **Vector Database:** FAISS (local, in-memory + persisted)

**Why RAG?**
- Prevents hallucinations by grounding answers in documents  
- Enables scalable document search  
- Separates retrieval from generation for modular design  

---

### ✂️ Chunking Strategy

- **Chunk size:** 500 characters  
- **Overlap:** 100 characters  

**Why chunking with overlap?**
- Preserves semantic continuity across chunks  
- Improves retrieval accuracy for long documents  
- Avoids loss of context at chunk boundaries  

---

### 🧾 Prompt Engineering

The prompt enforces **strict document grounding**:

- Retrieved context is injected into the prompt  
- Clear instructions prevent use of external knowledge  
- Fixed fallback response when information is missing  

**Example Constraint:**
> *If the answer is not present, say:*  
> **"The answer is not available in the document."**

This ensures reliable and transparent responses.

---

### 🖥️ User Interface

- **Framework:** Streamlit  

**Why Streamlit?**
- Rapid prototyping with minimal boilerplate  
- Interactive UI for file upload and Q&A  
- Ideal for demos and assessment evaluation  

---

## 🔁 End-to-End Workflow
####User
→ Upload PDF
→ Text Extraction
→ Chunking
→ Embeddings (MiniLM)
→ FAISS Vector Search
→ Context Injection
→ LLM Generation
→ Answer with Source Reference
