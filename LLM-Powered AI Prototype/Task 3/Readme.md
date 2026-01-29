# TASK 3 – Rapid Iteration Challenge  
## Advanced Capability: **Multi-Document Reasoning**

---

## 🎯 Goal

Enhance the existing **LLM-powered Chat with PDFs (RAG system)** by adding  
**Multi-Document Reasoning** capability.

This allows the system to:
- Accept **multiple PDFs**
- Retrieve information **across documents**
- Generate answers that synthesize knowledge from **more than one source**

---

## 🚀 Advanced Capability Chosen

### ✅ Multi-Document Reasoning

Instead of limiting answers to a single PDF, the system now reasons over  
**multiple documents simultaneously** using a shared vector space.

---

## 🧠 Why This Capability Was Chosen

Let’s be real — in the real world, answers don’t live in one file.

### Practical Reasons:
- Enterprise data is **distributed** (reports, policies, manuals)
- Academic research spans **multiple papers**
- Users expect **cross-document insights**, not isolated answers

### Technical Value:
- Tests the **true strength of RAG**
- Pushes retrieval + reasoning, not just generation
- Directly relevant to **knowledge bots** and **enterprise AI**

---

## ⚙️ How It Works (High-Level)

1. User uploads **multiple PDFs**
2. All documents are:
   - Extracted
   - Chunked
   - Embedded into a **single FAISS index**
3. User asks a question
4. Retriever fetches **top-k chunks across all documents**
5. Context is merged and passed to the LLM
6. LLM generates a **document-grounded answer**

---

## 🔄 Example Scenario

**Documents Uploaded:**
- Company Policy PDF
- Technical Architecture PDF

**User Question:**
> What security rules apply to the system architecture?

**System Behavior:**
- Retrieves security rules from Policy PDF
- Retrieves architecture details from Tech PDF
- Combines both to produce a **single grounded answer**

---

## ⚖️ Trade-Offs

| Trade-Off | Explanation |
|--------|------------|
| Increased retrieval complexity | More documents = larger vector space |
| Context window limits | Too many chunks can overflow LLM context |
| Slight latency increase | More embeddings + retrieval time |

**Reality check:**  
Power comes with cost. We manage it, not avoid it.

---

## 🚧 Limitations

### 1️⃣ Context Window Constraints
- LLMs can only process limited tokens
- Requires careful **top-k tuning**

### 2️⃣ Source Attribution Complexity
- Harder to clearly reference *which document said what*
- Needs future improvement (citations per chunk)

### 3️⃣ Scaling Challenges
- Large document collections require:
  - Persistent vector DB
  - Sharding or metadata filtering

---

## 🔮 Future Improvements

- Per-document metadata filtering
- Document-level confidence scoring
- Citation-aware responses
- Hybrid search (keyword + semantic)

---

## 🧠 Key Insight

Single-document QA is a demo.  
**Multi-document reasoning is production.**

This enhancement moves the system from a **toy chatbot**  
to a **real knowledge assistant** capable of handling real-world complexity.

---

## 👤 Author

**Siddharth Kumar**  
MCA (AI & ML)  
LLM & RAG Systems Engineer

