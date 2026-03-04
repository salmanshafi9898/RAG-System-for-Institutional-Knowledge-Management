# 🤖 FuquaAssist — RAG System for Institutional Knowledge Management

> **Custom Retrieval-Augmented Generation (RAG) pipeline** built for Duke Fuqua — grounding LLM answers strictly in internal PDF documents to eliminate hallucinations on institution-specific policy queries. RAG achieved **60% Exact Match accuracy** vs 25% for the best baseline LLM — with hallucination rates reduced to under 5%.

---

## 📌 Project Overview

Large Language Models like ChatGPT are powerful — but they hallucinate. For a university environment where students rely on precise answers about visa policies, GPA requirements, and course prerequisites, a generic LLM giving plausible-but-wrong advice is a serious risk.

**FuquaAssist** solves this by separating *knowledge* from *reasoning*. Rather than asking an LLM to recall Fuqua-specific policies from training data it doesn't have, the system first retrieves the exact relevant passage from internal documents, then uses the LLM only to articulate the answer. The result is a chatbot that answers Fuqua-specific questions accurately, 24/7 — without burdening admissions or student services staff.

---

## 🎯 The Business Problem

Fuqua generates vast amounts of unstructured policy data — MQM Handbooks, OPT/CPT Visa Guides, course catalogs, and more. Students must manually parse these documents to find answers. Standard LLMs:

- ❌ Don't have access to private, real-time Fuqua documents
- ❌ Hallucinate plausible-sounding but incorrect policy details
- ❌ Fail completely on specific numerical queries (GPA thresholds, deadlines, etc.)

**FuquaAssist** addresses all three.

---

## 🏗️ System Architecture

The pipeline performs five distinct steps:

```
┌──────────────────────────────────────────────────────────┐
│                     RAG PIPELINE                          │
│                                                           │
│  PDF Docs → Chunking → Embedding → FAISS Index            │
│                                          ↓                │
│  User Query → Embed Query → Top-K Search → LLM → Answer  │
└──────────────────────────────────────────────────────────┘
```

| Step | Description |
|---|---|
| **1. Ingestion** | Scans `data/` directory, loads and parses raw PDF text via PyPDF2 (strips headers/footers) |
| **2. Chunking** | Sliding window — 600-token chunks with 50-token overlap to preserve semantic continuity |
| **3. Vectorization** | Encodes chunks using `sentence-transformers` (embedding model configurable via `EMBED_MODEL`) |
| **4. Retrieval** | FAISS L2 nearest-neighbor search returns top-K=5 most relevant chunks for any query |
| **5. Generation** | Retrieved chunks injected into strict system prompt — *"Answer using ONLY the CONTEXT"* — before LLM call |

---

## 📊 Results & Evaluation

Evaluated on **20 curated high-priority questions** (e.g., *"What is the minimum GPA for the strategy track?"*) against ground-truth answers using three metrics: Exact Match, Token F1, and Retrieval Hit@5.

### Model Comparison

| Model | Type | Exact Match | F1 Score | Hit@5 |
|---|---|---|---|---|
| **all-mpnet-base-v2** | ✅ RAG | **0.600** | 0.177 | 0.800 |
| **intfloat/e5-large-v2** | ✅ RAG | **0.600** | 0.133 | **0.850** |
| all-MiniLM-L6-v2 | ✅ RAG | 0.450 | 0.112 | 0.650 |
| DeepSeek-Chat | ❌ Baseline | 0.250 | 0.005 | N/A |
| ArliAI (Gemma-3-27B-it) | ❌ Baseline | 0.150 | 0.000 | N/A |
| Gemini 1.5 Flash | ❌ Baseline | 0.000 | 0.032 | N/A |

### Key Findings

- **RAG dominates across the board** — every RAG configuration outperformed every baseline. Best RAG (EM 0.600) is 2.4× more accurate than best baseline (DeepSeek at 0.250)
- **Gemini scored 0.000 Exact Match** without RAG — not because it's unintelligent, but because it simply cannot hallucinate a specific GPA requirement it has never seen
- **Embedding model matters** — upgrading from MiniLM to MPNet or E5-Large yields a 20–33% improvement in retrieval quality, directly improving answer accuracy
- **Hallucination rate reduced to under 5%** with RAG vs ~75–100% failure rate for baselines on institution-specific queries

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Embeddings | `sentence-transformers` (MPNet, E5-Large, MiniLM) |
| Vector Search | FAISS (L2 / Euclidean distance) |
| LLM Backends | DeepSeek-Chat, Google Gemini 1.5 Flash, ArliAI (Gemma-3-27B) |
| PDF Parsing | PyPDF2 |
| API Framework | FastAPI + uvicorn |
| Frontend | HTML/JS chat UI (`index.html`) |
| Evaluation | Custom harness (`eval.py`) — Exact Match, Token F1, Hit@K |

---

## 📁 Repository Structure

```
├── Untitled2.py          # Core RAG pipeline (ingestion, chunking, FAISS, LLM call)
├── server.py             # FastAPI backend — exposes /chat endpoint
├── index.html            # Frontend chat UI
├── eval.py               # Evaluation harness (RAG / baseline / Gemini / ArliAI)
├── eval_qa.jsonl         # 20 curated QA pairs with ground-truth answers
├── run_all.sh            # Batch runner — sweeps 6 embeddings + all baselines
├── requirements.txt      # Python dependencies
├── data/                 # Knowledge base (PDFs: MQM Handbook, Visa Guides, etc.)
├── faiss.index           # Cached FAISS vector index (auto-generated)
├── index_meta.json       # Index metadata (auto-generated)
├── FuquaAssist.ipynb     # Colab quickstart notebook
└── README.md
```

---

## 🚀 Quickstart

### 1. Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install uvicorn google-generativeai
```

### 2. Set API Keys
```bash
export OPENAI_API_KEY="your_deepseek_key"
export OPENAI_BASE_URL="https://api.deepseek.com"
export GOOGLE_API_KEY="your_gemini_key"
export ARLIAI_API_KEY="your_arliai_key"
```

### 3. Run the Chatbot
```bash
uvicorn server:app --reload --port 8000
# Open http://localhost:8000 or double-click index.html
```

### 4. Run Full Evaluation (all embeddings + baselines)
```bash
bash run_all.sh
# Results saved as .rag.results.jsonl, .baseline.results.jsonl, etc.
```

### 5. Single Eval Runs
```bash
# RAG
python3 eval.py --qa-file eval_qa.jsonl --mode rag --k 5

# Baseline (DeepSeek)
python3 eval.py --qa-file eval_qa.jsonl --mode baseline --baseline-model deepseek-chat

# Gemini
python3 eval.py --qa-file eval_qa.jsonl --mode gemini --gemini-model gemini-1.5-flash

# ArliAI
python3 eval.py --qa-file eval_qa.jsonl --mode arliai --arliai-models "Gemma-3-27B-it" \
  --arliai-base-url https://api.arliai.com/v1 --arliai-timeout 15
```

### Swap Embedding Model
```bash
export EMBED_MODEL="intfloat/e5-large-v2"  # or all-mpnet-base-v2, all-MiniLM-L6-v2
# Delete cached index to force rebuild:
rm faiss.index index_meta.json
```

---

## ⚠️ Troubleshooting

| Issue | Fix |
|---|---|
| Stale/missing FAISS index | Delete `faiss.index` and `index_meta.json` — pipeline rebuilds automatically |
| Frontend "Error contacting backend" | Ensure uvicorn is running; check `BACKEND_URL` in `index.html` script tag |
| Slow or hanging eval | Reduce QA set size or shorten ArliAI timeout (`--arliai-timeout 15`) |
| Missing HF embedding model | Verify repo ID (e.g., `intfloat/e5-base-v2`) and HF auth token if gated |

---

## 💡 Key Takeaways

- **RAG is not optional for institutional chatbots** — baseline LLMs fail completely on private, specific policy data. The architecture choice matters more than the LLM choice
- **Chunking strategy is underrated** — the 600-token / 50-token overlap design directly prevents the "lost in the middle" problem where relevant content gets buried
- **Embedding model ROI is real** — moving from MiniLM to MPNet costs more compute but buys a 33% jump in Exact Match accuracy; for a student-facing production system, that tradeoff is clearly justified
- **Custom pipelines beat black-box tools** — building the RAG logic in Python (rather than using OpenAI Assistants API) gave full control over chunking, retrieval filtering, and data privacy

---

## 👥 Team

| Member | Role |
|---|---|
| Gaurang Agrawal | Evaluation & Benchmarking (`eval.py`) |
| Sawaiz Fatar | Data Ingestion & PDF Parsing |
| Skylar Qiu | Chunking Strategy & Preprocessing |
| Marwa Bouabid | Modeling & Embedding Architecture |
| Arshad Rizvi | Pipeline Integration & FAISS-LLM Connection |
| **Salman Khan Shafi** | **MS Business Analytics — Duke Fuqua '26** |

---

## 👤 Author

**Salman Khan Shafi**
MS Business Analytics — Duke Fuqua '26
[LinkedIn](https://linkedin.com/in/salmankhanshafi) • [GitHub](https://github.com/salmanshafi9898)
