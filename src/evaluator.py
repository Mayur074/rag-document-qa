"""
src/evaluator.py
-----------------
Evaluates RAG pipeline quality using three metrics:
  1. Answer Relevance   — does the answer address the question?
  2. Faithfulness       — is the answer grounded in retrieved context?
  3. Context Precision  — how relevant are the retrieved chunks?

Uses Claude as the judge (LLM-as-evaluator pattern).
"""

from __future__ import annotations
import os
import json
import logging
from dataclasses import dataclass, asdict

from anthropic import Anthropic

logger = logging.getLogger(__name__)
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

EVAL_SYSTEM = """You are a strict RAG evaluation judge. Respond ONLY with valid JSON.
No preamble, no markdown, no explanation outside the JSON object."""

RELEVANCE_PROMPT = """Rate how well the answer addresses the question.
Score 1-5 where 5 = perfectly addresses the question.

Question: {question}
Answer: {answer}

Respond with: {{"score": <1-5>, "reason": "<one sentence>"}}"""

FAITHFULNESS_PROMPT = """Rate whether the answer is fully supported by the context.
Score 1-5 where 5 = every claim is directly supported by context, 1 = claims not in context.

Context: {context}
Answer: {answer}

Respond with: {{"score": <1-5>, "reason": "<one sentence>"}}"""

CONTEXT_PRECISION_PROMPT = """Rate how relevant the retrieved context is to the question.
Score 1-5 where 5 = all retrieved chunks are highly relevant.

Question: {question}
Context: {context}

Respond with: {{"score": <1-5>, "reason": "<one sentence>"}}"""


@dataclass
class EvalResult:
    question:          str
    answer:            str
    relevance_score:   float
    faithfulness_score: float
    context_precision: float
    overall_score:     float
    relevance_reason:  str
    faithfulness_reason: str
    context_reason:    str


def _judge(prompt: str) -> dict:
    """Call Claude as judge and parse JSON response."""
    try:
        msg = client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 200,
            system     = EVAL_SYSTEM,
            messages   = [{"role": "user", "content": prompt}],
        )
        return json.loads(msg.content[0].text)
    except Exception as e:
        logger.warning(f"Judge call failed: {e}")
        return {"score": 0, "reason": "evaluation failed"}


def evaluate(question: str, answer: str, source_documents: list) -> EvalResult:
    """
    Evaluate a single RAG response across three dimensions.
    """
    context = "\n\n".join(
        doc.page_content[:500] for doc in source_documents[:3]
    )

    rel  = _judge(RELEVANCE_PROMPT.format(question=question, answer=answer))
    fth  = _judge(FAITHFULNESS_PROMPT.format(context=context, answer=answer))
    ctp  = _judge(CONTEXT_PRECISION_PROMPT.format(question=question, context=context))

    r_score = float(rel.get("score", 0))
    f_score = float(fth.get("score", 0))
    c_score = float(ctp.get("score", 0))
    overall = round((r_score + f_score + c_score) / 3, 2)

    return EvalResult(
        question           = question,
        answer             = answer,
        relevance_score    = r_score,
        faithfulness_score = f_score,
        context_precision  = c_score,
        overall_score      = overall,
        relevance_reason   = rel.get("reason", ""),
        faithfulness_reason= fth.get("reason", ""),
        context_reason     = ctp.get("reason", ""),
    )


def batch_evaluate(qa_pairs: list[dict], pipeline) -> list[EvalResult]:
    """
    Evaluate a list of {question, expected_answer} pairs.
    qa_pairs: [{"question": "...", "expected_answer": "..."}, ...]
    """
    results = []
    for pair in qa_pairs:
        q = pair["question"]
        logger.info(f"Evaluating: {q[:60]}...")
        try:
            response = pipeline.query(q)
            result   = evaluate(q, response["answer"], response["source_documents"])
            results.append(result)
        except Exception as e:
            logger.error(f"Evaluation failed for '{q}': {e}")
    return results


def print_eval_report(results: list[EvalResult]):
    """Print a formatted evaluation report to console."""
    if not results:
        print("No results to report.")
        return

    print("\n" + "="*60)
    print("RAG EVALUATION REPORT")
    print("="*60)

    for i, r in enumerate(results, 1):
        print(f"\n[Q{i}] {r.question[:80]}")
        print(f"  Relevance:        {r.relevance_score}/5  — {r.relevance_reason}")
        print(f"  Faithfulness:     {r.faithfulness_score}/5  — {r.faithfulness_reason}")
        print(f"  Context Precision:{r.context_precision}/5  — {r.context_reason}")
        print(f"  Overall:          {r.overall_score}/5")

    avg_overall = sum(r.overall_score for r in results) / len(results)
    avg_rel     = sum(r.relevance_score for r in results) / len(results)
    avg_fth     = sum(r.faithfulness_score for r in results) / len(results)
    avg_ctp     = sum(r.context_precision for r in results) / len(results)

    print("\n" + "-"*60)
    print("AGGREGATE SCORES")
    print(f"  Avg Relevance:         {avg_rel:.2f}/5")
    print(f"  Avg Faithfulness:      {avg_fth:.2f}/5")
    print(f"  Avg Context Precision: {avg_ctp:.2f}/5")
    print(f"  Avg Overall:           {avg_overall:.2f}/5")
    print("="*60)
