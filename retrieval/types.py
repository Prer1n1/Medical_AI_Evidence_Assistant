"""Shared retrieval types — split into their own module so
hybrid_retriever.py, evidence_ranker.py, confidence.py, and agent/state.py
can all depend on RetrievedChunk without any of them having to import
FROM hybrid_retriever.py (which imports evidence_ranker.py, and would
otherwise create a circular import)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    chunk_id: str
    content: str
    metadata: dict
    score: float
