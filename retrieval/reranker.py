"""Cross-encoder reranking via Cohere's Rerank API. Ported verbatim from
enterprise-agentic-rag — reranking candidate passages for precision is
entirely domain-agnostic; the medical-specific re-weighting happens
AFTER this, in retrieval/evidence_ranker.py.

Why reranking at all: Reciprocal Rank Fusion (hybrid_retriever.py) only
looks at each retriever's RANK POSITION, never actual semantic relevance.
A reranker scores each (query, candidate) pair directly.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cohere

from config import COHERE_API_KEY
from retry_utils import retry_cohere_call

logger = logging.getLogger(__name__)

_RERANK_MODEL = "rerank-v3.5"


@retry_cohere_call
def _call_rerank(client: cohere.ClientV2, query: str, documents: List[str], top_n: int):
    return client.rerank(model=_RERANK_MODEL, query=query, documents=documents, top_n=top_n)


def rerank(query: str, documents: List[str], top_n: int) -> Optional[List[Tuple[int, float]]]:
    """Returns (index_into_documents, relevance_score) pairs, best-first.
    Returns None if reranking isn't available or fails for ANY reason — the
    caller falls back to its pre-rerank ordering instead of breaking
    retrieval outright."""
    if not COHERE_API_KEY or not documents:
        return None
    try:
        client = cohere.ClientV2(api_key=COHERE_API_KEY)
        response = _call_rerank(client, query, documents, top_n)
        return [(result.index, result.relevance_score) for result in response.results]
    except Exception:
        logger.exception("rerank_failed")
        return None
