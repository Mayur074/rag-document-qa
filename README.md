# 📚 RAG Document Q&A System

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline that answers questions grounded in your own documents using **LangChain, ChromaDB, Hugging Face embeddings, and Claude AI**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange)
![Claude](https://img.shields.io/badge/Claude-Sonnet-purple)
![CI](https://github.com/yourgithub/rag-document-qa/actions/workflows/ci.yml/badge.svg)

---

## Architecture

```
Documents (PDF/TXT)
       │
       ▼
  Document Loader (LangChain)
       │
       ▼
  Text Splitter (RecursiveCharacterTextSplitter)
  chunk_size=1000, overlap=200
       │
       ▼
  HuggingFace Embeddings (all-MiniLM-L6-v2)
       │
       ▼
  ChromaDB Vector Store (persistent, local)
       │
  ┌────┴────────────────────────┐
  │   MMR Retrieval (k=5)       │
  └────────────────┬────────────┘
                   │
                   ▼
          Claude Sonnet (LLM)
          RetrievalQA Chain
                   │
                   ▼
            Answer + Sources
                   │
                   ▼
         LLM-as-Judge Evaluation
    (Relevance / Faithfulness / Context Precision)
```

## Features

- **Multi-format ingestion** — PDF and TXT files
- **Persistent vector store** — ChromaDB stores embeddings locally across sessions
- **MMR retrieval** — Maximal Marginal Relevance reduces redundant context
- **LLM-as-judge evaluation** — Claude scores every answer on 3 dimensions
- **Streamlit UI** — upload docs, ask questions, see sources and eval scores
- **GitHub Actions CI** — automated testing on every push

## Quickstart

```bash
git clone https://github.com/yourgithub/rag-document-qa.git
cd rag-document-qa
pip install -r requirements.txt

export ANTHROPIC_API_KEY="sk-ant-..."
streamlit run app.py
```

## Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| Answer Relevance | Does the answer address the question? |
| Faithfulness | Is the answer grounded in retrieved context? |
| Context Precision | Are the retrieved chunks relevant to the question? |

Each metric scored 1–5 by Claude-as-judge. Overall = average of three.

## Project Structure

```
rag-document-qa/
├── app.py                    # Streamlit UI
├── src/
│   ├── rag_pipeline.py       # Core RAG pipeline
│   └── evaluator.py          # LLM-as-judge evaluation
├── data/
│   └── sample_docs/          # Sample documents for testing
├── tests/
│   └── test_rag_pipeline.py  # Unit tests
├── .github/workflows/ci.yml  # GitHub Actions CI
└── requirements.txt
```

## Author

**Mayur Sangle** — M.S. Data Science, University of Maryland
[LinkedIn](https://linkedin.com/in/mayur-sangle-04m) • [GitHub](https://github.com/Mayur074)
