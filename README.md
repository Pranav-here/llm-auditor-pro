# AI Knowledge Auditor Pro

A lightweight **Streamlit** application for objectively assessing whether an AI‑generated answer is supported by evidence found in a reference PDF.

🔗 **Live Demo**: [https://llm-auditor-pro.streamlit.app](https://llm-auditor-pro.streamlit.app)

---

## 1  Project Scope

The project implements an end‑to‑end verification pipeline that
1. **Ingests PDF documents** and performs sentence‑aware chunking;
2. **Builds a FAISS vector index** using `all‑mpnet‑base‑v2` sentence embeddings;
3. **Retrieves candidate passages** via a hybrid query that blends the user’s question and the model’s answer;
4. **Ranks passages** with a cross‑encoder similarity check;
5. **Calculates a multi‑factor trust score** using four signals:
   - Semantic similarity
   - Textual overlap (n‑gram, keyphrase, sequence ratio)
   - Named‑entity alignment (spaCy NER)
   - Basic factual consistency (numeric/context agreement)
6. **Explains the verdict** by highlighting matching sentences and scoring components;
7. **Aggregates results** in a single‑audit view, a batch processor, and a live analytics dashboard.

All computational steps run locally; no external services are required once the models are downloaded.

---

## 2  Key Features (Implemented)

| Module | Description |
| ------ | ----------- |
| **Document Loader** | Extracts text from PDFs using *PyMuPDF* (fitz) and cleans layout artifacts. |
| **Smart Chunker** | Splits text at sentence boundaries; retains configurable overlap for context preservation. |
| **Vector Store** | Normalised cosine‑similarity index built with **FAISS**; enables millisecond retrieval. |
| **Hybrid Search** | Weights the question (70 %) and the answer (30 %) to form a single query vector. |
| **Auditor Core** | Combines retrieval, scoring, explanation generation, and sentence highlighting. |
| **Batch Processor** | Processes CSV files of Q&A pairs concurrently; outputs a Markdown or CSV report. |
| **Streamlit UI** | Wide‑layout interface with tabbed navigation, Plotly charts, and adjustable thresholds. |

---

## 3  System Architecture

```
┌────────────┐      ┌───────────────┐      ┌───────────────┐
│   PDF in   │──▶──│   Chunker &   │──▶──│   Embeddings   │
└────────────┘      │  Entity NER   │      └───────────────┘
        │            └───────────────┘            │
        ▼                                         ▼
   FAISS Index  ◀── Hybrid Search ◀──  Question + Answer
        │                                         │
        ▼                                         ▼
Candidate Passages ─▶ Scoring & Highlighting ─▶ Trust Report
```

---

## 4  Usage

```bash
# Launch the application
streamlit run app.py
```

1. Upload a PDF (≤ 200 MB).
2. Enter the question posed to the LLM and the model’s answer.
3. Review the trust score, highlighted evidence, and factor breakdown.
4. (Optional) Switch to **Batch Processing** to audit many Q&A pairs at once.

---

## 5  Example Output

<!-- Replace with annotated screenshot or animated GIF illustrating a single audit. -->
*(See the `app test/` folder for working examples and benchmark screenshots.)*

---

## 6  Performance Snapshot *(local workstation, RTX 3060)*

| Task | Size | Time |
| ---- | ---- | ---- |
| Index build | 200‑page PDF | ≈ 15 s |
| Single audit | avg. 500 tokens answer | < 4 s |
| Batch (100 pairs) | same doc | ≈ 70 s |

---

## 7  Roadmap

- [ ] Cross‑encoder factuality classifier (planned)
- [ ] Hugging Face Spaces demo (planned)
- [ ] Pluggable vector databases (Milvus / Qdrant)

---

## 8  License

Released under the MIT License.

---

## 9  Acknowledgements

Built with **Sentence‑Transformers**, **FAISS**, **spaCy**, **PyMuPDF**, **Plotly**, and **Streamlit**.


