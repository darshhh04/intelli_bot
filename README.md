# 🤖 IntelliBot — Context-Aware RAG Chatbot

A chatbot that answers questions exclusively from uploaded documents using
Retrieval-Augmented Generation (RAG) — grounded, cited answers with no
hallucination on out-of-scope questions.

**Live demo:** https://intellibot-darshan.streamlit.app/

Built for the Code A Nova Internship Program 2026 — AI Track, Assignment 2.

---

## Architecture

```text
PDF/TXT files
      ↓
PyMuPDF (text extraction)
      ↓
RecursiveCharacterTextSplitter (chunking, 500 chars, 50 overlap)
      ↓
SentenceTransformers — all-MiniLM-L6-v2 (embeddings, 384-dim)
      ↓
ChromaDB (persistent vector store, cosine similarity)
      ↓
Top-K retrieval (K=3) → Gemini 3.5 Flash (grounded generation)
      ↓
Streamlit chat UI (session-based conversation memory)
```

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.13+ |
| LLM | Gemini 3.5 Flash |
| Embeddings | SentenceTransformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB (persistent, cosine similarity) |
| Document Parsing | PyMuPDF (fitz) |
| Chat Memory | Session-based list, passed per-request |
| UI | Streamlit |
| Evaluation | LLM-as-judge (faithfulness + relevancy scoring) |

## Features

- 📄 PDF/TXT document ingestion with graceful error handling
- ✂️ Recursive text chunking for coherent, precise retrieval
- 🔍 Top-K cosine similarity search over persistent embeddings
- 🚫 Strict grounding — refuses to answer from outside the knowledge base
- 💬 Multi-turn conversation memory for natural follow-up questions
- ⚡ Instant canned responses for greetings/thanks/goodbyes (skips LLM call entirely)
- 📤 Live document upload directly from the UI
- 🎨 Clean, minimal Streamlit interface

## Setup (local)

```bash
git clone https://github.com/<your-username>/intelli_bot.git
cd intelli_bot

python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

Create `.env` in the project root:

```env
GEMINI_API_KEY=your_key_here
```

Get a free Gemini API key at:
https://aistudio.google.com/apikey

Add PDF/TXT files to `data/`, then build the knowledge base:

```bash
python -m src.vector_store
```

Run the app:

```bash
streamlit run app.py
```

## Evaluation

Run the evaluation script to score the RAG pipeline:

```bash
python -m src.evaluate
```

The evaluation uses an **LLM-as-judge** approach. For each test question,
the bot's retrieved context and generated answer are evaluated on a scale
of 0.0–1.0 for:

- **Faithfulness** — whether the answer's claims are supported by the retrieved context.
- **Answer relevancy** — whether the answer directly addresses the question asked.

### Evaluation Report

| Question | Faithfulness | Relevancy | Notes |
|---|---:|---:|---|
| What is IntelliBot's tech stack? | 1.00 | 0.95 | Fully grounded, correct sources cited |
| What UI framework does the project use? | 1.00 | 0.98 | Direct match from document |
| What does the evaluation framework use? | 0.95 | 0.90 | Minor phrasing difference from context |
| What LLM provider does the project use? | 1.00 | 0.92 | Correct, well-cited |
| What are the deliverables in this project? | 0.90 | 0.85 | Answer slightly incomplete vs. full list |
| What should I eat today? | 1.00 | 1.00 | Correctly refused — no hallucination |

**Average — Faithfulness: 0.98 | Relevancy: 0.93**

> **Evaluation note:** The table above is an LLM-based evaluation and is **not a Ragas evaluation**. Ragas was attempted during development but could not be successfully run in the current environment, so an LLM-as-judge approach was used instead to evaluate faithfulness and answer relevancy. This provides a practical evaluation of the RAG pipeline, but it should not be described as official Ragas output.

> **LLM provider note:** Gemini AI was used for generation and evaluation instead of OpenAI because Gemini provides a free API tier suitable for this project and its development/testing requirements.

### Hallucination Testing

Manually verified:

| Test Case | Result |
|---|---|
| Question answerable from documents | ✅ Grounded, correct, cited source |
| Question outside document scope (e.g. general trivia) | ✅ Correctly refuses: "I don't have that information in the provided documents" |
| Follow-up question relying on conversation memory | ✅ Correctly resolved using prior context |
| Greeting / thanks / goodbye | ✅ Instant canned reply, no document context pulled in |
| Malformed/unreadable PDF | ✅ Skipped gracefully, no crash |

## Known Limitations

- No re-ranking after retrieval — relies purely on top-K cosine similarity
- Chat history is per-session only, not persisted across app restarts
- Vector store rebuilds on Streamlit Cloud redeploys (free-tier filesystem is ephemeral)

## Project Structure

```text
intelli_bot/
│
├── app.py                 # Streamlit UI
│
├── src/
│   ├── config.py          # Environment/secrets loading
│   ├── loader.py          # PDF/TXT ingestion
│   ├── chunker.py         # Text splitting
│   ├── embedder.py        # SentenceTransformer embeddings
│   ├── vector_store.py    # ChromaDB persistence + search
│   ├── rag.py             # Retrieval + Gemini generation + memory
│   └── evaluate.py        # LLM-as-judge evaluation
│
├── data/                  # Sample knowledge base documents
│
└── requirements.txt
```
