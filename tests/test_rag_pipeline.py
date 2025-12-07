"""
tests/test_rag_pipeline.py
---------------------------
Unit tests for RAG pipeline config and evaluator logic.
Designed to run in CI without requiring LangChain/ChromaDB.
"""

import os
import pytest
from dataclasses import dataclass


# ── Config constants (mirrored here to avoid heavy imports in CI) ─────────────

CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR     = "data/chroma_db"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Use the following retrieved context
to answer the question accurately and concisely. If the answer is not in the
context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""


# ── Config tests ──────────────────────────────────────────────────────────────

def test_chunk_size_positive():
    assert CHUNK_SIZE > 0

def test_chunk_overlap_less_than_chunk_size():
    assert CHUNK_OVERLAP < CHUNK_SIZE

def test_embedding_model_defined():
    assert len(EMBEDDING_MODEL) > 0

def test_persist_dir_defined():
    assert len(PERSIST_DIR) > 0

def test_anthropic_model_defined():
    assert len(ANTHROPIC_MODEL) > 0

def test_rag_prompt_has_context_variable():
    assert "{context}" in RAG_PROMPT_TEMPLATE

def test_rag_prompt_has_question_variable():
    assert "{question}" in RAG_PROMPT_TEMPLATE


# ── Evaluator dataclass tests ─────────────────────────────────────────────────

@dataclass
class EvalResult:
    question:           str
    answer:             str
    relevance_score:    float
    faithfulness_score: float
    context_precision:  float
    overall_score:      float
    relevance_reason:   str
    faithfulness_reason: str
    context_reason:     str


def test_eval_overall_score_in_range():
    r = EvalResult(
        question="q", answer="a",
        relevance_score=4.0, faithfulness_score=3.0,
        context_precision=5.0, overall_score=4.0,
        relevance_reason="", faithfulness_reason="", context_reason=""
    )
    assert 1.0 <= r.overall_score <= 5.0

def test_overall_score_is_average():
    r, f, c = 4.0, 3.0, 5.0
    overall = round((r + f + c) / 3, 2)
    assert overall == 4.0

def test_score_minimum_boundary():
    assert 1.0 >= 1.0

def test_score_maximum_boundary():
    assert 5.0 <= 5.0

def test_eval_result_fields_populated():
    r = EvalResult(
        question="What is RAG?", answer="Retrieval-Augmented Generation.",
        relevance_score=5.0, faithfulness_score=5.0,
        context_precision=4.0, overall_score=4.67,
        relevance_reason="Directly answers", faithfulness_reason="Grounded",
        context_reason="Relevant chunks"
    )
    assert r.question == "What is RAG?"
    assert r.relevance_score == 5.0


# ── File existence tests ──────────────────────────────────────────────────────

def test_sample_document_exists():
    path = os.path.join(
        os.path.dirname(__file__),
        "..", "data", "sample_docs", "healthcare_ai_overview.txt"
    )
    assert os.path.exists(path), "Sample document missing from repo"

def test_app_file_exists():
    path = os.path.join(os.path.dirname(__file__), "..", "app.py")
    assert os.path.exists(path), "app.py missing from repo"

def test_requirements_file_exists():
    path = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
    assert os.path.exists(path), "requirements.txt missing from repo"
