# 🧠 AI Knowledge Auditor Pro (v5.0)

_A lightning-fast Streamlit app that **audits LLM answers** against trusted sources and tells you — in seconds — whether to trust, verify, or toss them._

---

## 🚀  What it does

1. **Ingest any PDF** *or* flip to **real-time Web Search mode** when no document is loaded.  
2. **Chunk & embed** the reference text with Sentence-Transformers (all-mpnet-base-v2) and index it in FAISS for millisecond retrieval.  
3. Run an **enhanced audit pipeline** that blends  
   - semantic similarity  
   - lexical overlap  
   - named‑entity alignment (spaCy)  
   - answer‑specificity & completeness heuristics  
   - retrieval quality  
4. **Non‑linear calibration** → one intuitive **Trust Score (0–100 %)** with “Trusted / Caution / Suspicious” bands.  
5. Surface a **highlighted evidence snippet**, a factor‑by‑factor score breakdown, and an optional entity‑level diff.  
6. **Batch mode & Analytics**: audit thousands of Q‑A pairs at once and track aggregate performance over time.

---

## 🏆  Why it’s different

| ⚙️ Feature | How it’s done |
|-----------|---------------|
| **Hybrid retrieval** | PDF‑backed FAISS *or* Tavily Web Search – same pipeline, no code changes |
| **Multi‑factor scoring** | Tunable weights + non‑linear scaling curves (`sigmoid`, `power`, `threshold`) |
| **Entity‑aware** | spaCy `en_core_web_sm` + fuzzy matching boosts factual alignment |
| **Scale‑ready batch runner** | Threaded executor with graceful error capture and CSV/Markdown export |
| **Live analytics** | Plotly dashboards (time‑series, distribution, model vs. source comparisons) |
| **Streamlit UX polish** | Dark‑friendly CSS, metric cards, tabbed workflow, debug sidebar |

---

## 🌐  Live links

|  | Version | Link |
|---|---|---|
| 🔹 **MVP (v1)** | <https://ai-knowledge-auditor.streamlit.app/> |
| 🔸 **Pro (v5.0) – source** | <https://github.com/Pranav-here/llm-auditor-pro> |
| 🔹 **MVP code** | <https://github.com/Pranav-here/ai-knowledge-auditor> |

> **Heads‑up:** the public demo runs on free resources, so cold‑starts may take ~20 s.

---

## 🧩  High‑level architecture
```text
┌──────────┐ ┌──────────────┐
│  Streamlit Front‑end      │
│  – tabs (Audit/Batch/…)   │
└──────┬───┘      │   user inputs
       ▼
┌─────────────────────────────┐
│ smart_chunk_text()          │  PDF ingestion (PyMuPDF) → sentence‑aware chunking
└──────┬──────────────────────┘
       ▼ embeddings (Sentence‑Transformers)
┌─────────────┐    ┌────────────────┐
│  FAISS idx  │←── │  Tavily Search │  (fallback)
└──────┬──────┘    └────────────────┘
       ▼ top‑K chunks
┌─────────────────────────────┐
│ enhanced_audit()            │  multi‑factor scoring + calibration
└──────┬──────────────────────┘
       ▼
┌──────────────┐
│ Trust Score  │ + evidence / report / analytics
└──────────────┘
```

---

## 🔬  Scoring recipe (default **standard** mode)

Weight | Metric | How it’s computed
---|---|---
0.25 | Semantic similarity | combined (0.7 × question + 0.3 × answer) vs. chunk
0.20 | Lexical overlap | n‑gram phrase intersection with length‑aware penalty
0.25 | Entity matching | recall + precision of spaCy entities (fuzzy)
0.20 | Specificity | numeric / proper‑noun density ratio
0.10 | Completeness | keyword coverage + length heuristic

Raw score → sigmoid scaling → retrieval quality boost ⇒ **Trust Score**.

Switch to **strict** / **lenient** modes to tweak the weights + curve on the fly.

---

## 🗂  Key modules

| Path | Purpose |
|---|---|
| `app.py` | Streamlit UI & session orchestration |
| `core/loader.py` | PDF extraction, smart chunking, entity pull |
| `core/vector_store.py` | FAISS index helpers + hybrid search |
| `core/auditor.py` | All scoring logic & explanation generator |
| `core/websearch.py` | Tavily API wrapper + temp index builder |
| `core/batch_processor.py` | CSV batch runner, threading, exports |

---

## 🛣️  Roadmap

- 🏗 **Continuous chat auditing** (no page refresh)  
- 📦 **Docker & HuggingFace Spaces** deployment  
- 🔐 **Source‑chain of custody** (hash PDFs, log web snippets)  
- 🧪 **Unit & fuzz tests** for every scoring sub‑module  

---

## 🙏  Acknowledgements
- **Sentence‑Transformers** – amazing OSS embeddings  
- **spaCy** – robust NER out‑of‑the‑box  
- **Tavily Search** – fast, reliable web snippets  
- Streamlit, Plotly, FAISS, NLTK, PyMuPDF… you make data apps fun.

---

**Made with ☕ and too many tweak cycles by [Pranav Kuchibhotla](https://pranavkuchibhotla.com).**
