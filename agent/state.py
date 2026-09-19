"""Shared state that flows through the LangGraph workflow. Simplified from
enterprise-agentic-rag's AgentState: no allowed_categories/access_restricted
(access control was deliberately dropped from this project's scope — see
docs/design-decisions.md, "Scope"). New: `confidence`, populated by
synthesize_node from retrieval/confidence.py — the per-answer trust signal
this project's evidence layer exists to produce."""

from __future__ import annotations

import operator
from typing import Annotated, List, TypedDict

from retrieval.types import RetrievedChunk


class AgentState(TypedDict):
    query: str
    categories: List[str]  # decided by the planner node
    # Annotated with operator.add: when multiple retrieve_node branches run
    # in PARALLEL (one per category), LangGraph concatenates their results
    # into this list rather than one branch overwriting another.
    retrieved_chunks: Annotated[List[RetrievedChunk], operator.add]
    answer: str
    citations: List[dict]
    confidence: dict  # {"label": str, "score": float, "reasoning": str}
