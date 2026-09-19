"""Confidence scoring — no equivalent in enterprise-agentic-rag. Answers a
question evidence ranking alone doesn't: not "how good is each individual
chunk" but "how much should the FINAL ANSWER, as a whole, be trusted."

Three signals, each independently inspectable (this is the "explainable"
half of the resume claim — a caller can see WHY a confidence label was
assigned, not just the label):
  - evidence strength: relevance-weighted average of the cited chunks'
    evidence levels (retrieval/evidence_ranker.py's weights)
  - source agreement: how many INDEPENDENT sources (distinct doc_id, so
    two chunks from the same paper don't count as two agreeing sources)
    support the answer — a claim backed by one case report is weaker than
    the same claim showing up in a guideline AND a trial
  - retrieval strength: the top chunk's own relevance score — if even the
    best match was a weak one, nothing downstream should be reported as
    confident
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ingestion.metadata_extractor import EVIDENCE_LEVEL_WEIGHTS
from retrieval.types import RetrievedChunk

HIGH_THRESHOLD = 0.7
MODERATE_THRESHOLD = 0.45


@dataclass
class ConfidenceResult:
    label: str  # "High" | "Moderate" | "Low"
    score: float
    reasoning: str
    independent_source_count: int


def _evidence_weight(chunk: RetrievedChunk) -> float:
    return EVIDENCE_LEVEL_WEIGHTS.get(
        chunk.metadata.get("evidence_level", ""), EVIDENCE_LEVEL_WEIGHTS["Unknown"]
    )


def compute_confidence(chunks: List[RetrievedChunk]) -> ConfidenceResult:
    if not chunks:
        return ConfidenceResult(
            label="Low", score=0.0, reasoning="No evidence retrieved.", independent_source_count=0
        )

    # relevance_score is set by evidence_ranker.apply_evidence_ranking
    # (the pre-evidence-weighting score); falls back to .score for chunks
    # that bypassed evidence ranking (shouldn't happen in the normal
    # agent flow, but keeps this function safe to call standalone/in tests).
    weights = [c.metadata.get("relevance_score", c.score) for c in chunks]
    total_weight = sum(weights) or 1.0
    evidence_strength = sum(_evidence_weight(c) * w for c, w in zip(chunks, weights)) / total_weight

    independent_sources = {c.metadata.get("doc_id") or c.metadata.get("source") for c in chunks}
    independent_source_count = len(independent_sources)
    # Diminishing returns, capped at 1.0: agreement across 2 independent
    # sources matters a lot more than the jump from 4 to 5.
    agreement_strength = min(1.0, 0.5 + 0.15 * (independent_source_count - 1))

    top_relevance = max(weights)
    # Normalize: RRF-fused scores and Cohere relevance scores are both
    # roughly 0-1 in the ranges this project's retrieval actually produces,
    # but clip defensively rather than assume that always holds.
    retrieval_strength = max(0.0, min(1.0, top_relevance))

    score = 0.45 * evidence_strength + 0.35 * agreement_strength + 0.20 * retrieval_strength

    if score >= HIGH_THRESHOLD:
        label = "High"
    elif score >= MODERATE_THRESHOLD:
        label = "Moderate"
    else:
        label = "Low"

    reasoning = (
        f"evidence strength {evidence_strength:.2f} (weighted avg. evidence level of cited sources), "
        f"{independent_source_count} independent source(s) agreeing, "
        f"top retrieval relevance {top_relevance:.2f}"
    )
    return ConfidenceResult(
        label=label, score=round(score, 3), reasoning=reasoning,
        independent_source_count=independent_source_count,
    )
