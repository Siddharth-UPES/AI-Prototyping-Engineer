# TASK 2 – Hallucination & Quality Control

---

##  Problem Statement

Large Language Models (LLMs) are powerful, fluent, and confident.  
That’s the danger.

They can generate **answers that sound correct but are factually wrong**,  
especially when:
- Context is missing  
- The question goes beyond the document  
- The model relies on learned patterns instead of evidence  

This behavior is known as **hallucination**, and controlling it is critical in any real-world AI system.

---

##  Causes of Hallucination in This System

In a **Chat with PDFs (RAG-based system)**, hallucinations can arise due to:

### 1️ Missing or Weak Retrieval Context
- Relevant information is not retrieved from the vector database
- Poor similarity match between query and document chunks

### 2️ Overconfidence of the LLM
- LLM attempts to “fill the gaps” using its pretrained knowledge
- Generates plausible but unsupported answers

### 3️ Ambiguous or Out-of-Scope Questions
- User asks questions not covered in the uploaded PDF
- Model responds instead of refusing

### 4️ Long or Noisy Documents
- Important facts buried deep inside the document
- Retrieval misses key sections

---

##  Guardrails Implemented

To control hallucinations and improve answer quality, the following guardrails are implemented:

---

###  Guardrail 1: Source-Grounded Answers (RAG Enforcement)

**What is done:**
- The LLM receives only the retrieved document chunks as context
- External or general knowledge is explicitly disallowed

**Prompt Constraint Example:**
> *Answer the question using only the provided context.*  
> *If the answer is not present, say:*  
> **"The answer is not available in the document."**

**Impact:**
- Prevents the model from inventing facts
- Ensures transparency and trust

---

###  Guardrail 2: Confidence / Similarity Threshold

**What is done:**
- FAISS similarity score is checked before generation
- If retrieval confidence is below a threshold, the system does not query the LLM

**Fallback Response:**
> **"The answer is not available in the document."**

**Impact:**
- Stops low-quality or irrelevant answers
- Reduces false positives and misleading outputs

---

###  (Optional) Guardrail 3: Prompt-Level Refusal Policy

**What is done:**
- Prompt explicitly instructs the model to refuse guessing
- Encourages factual silence over fabricated confidence

**Impact:**
- Aligns model behavior with real-world QA ethics
- Improves reliability for academic and professional use

---

##  Examples of Improved Responses

###  Before Guardrails (Hallucination)

**User Question:**  
> What is the author’s conclusion about climate change?

**Model Answer:**  
> The author strongly believes climate change is caused by human activities and suggests policy reforms.

 *Issue:*  
The document never mentioned climate change.

---

###  After Guardrails (Controlled Response)

**Improved Answer:**  
> **The answer is not available in the document.**

✔️ Honest  
✔️ Transparent  
✔️ Trustworthy  

---

###  Before Guardrails

**User Question:**  
> What algorithm does the system use for ranking pages?

**Model Answer:**  
> The system uses PageRank and HITS algorithms.

 *Issue:*  
Not mentioned in the PDF.

---

###  After Guardrails

**Improved Answer:**  
> **The answer is not available in the document.**

No lies. No fluff. Just facts—or silence.

---

##  Summary

| Issue | Solution |
|-----|---------|
| Confident wrong answers | Source-grounded RAG |
| Missing context | Similarity threshold |
| Out-of-scope queries | Prompt refusal policy |
| Overgeneration | Strict fallback response |

---

##  Key Takeaway

A good LLM talks well.  
A **reliable LLM knows when to shut up**.

By combining **RAG**, **confidence checks**, and **prompt constraints**,  
this system prioritizes **truth over fluency**—which is exactly what real AI needs.

