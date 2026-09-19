"""Evidence ranking — no equivalent in enterprise-agentic-rag. This is
what "evidence ranking" in the resume bullet actually does: re-weight
retrieved chunks by where they sit on the clinical evidence hierarchy
(ingestion/metadata_extractor.EVIDENCE_LEVEL_WEIGHTS), not just by how
semantically close they are to the query.

Deliberately a SEPARATE pass after RRF fusion + Cohere reranking, not
folded into either: RRF/reranking answer "how relevant is this passage to
the query", evidence ranking answers a different question entirely — "how
much should this passage be trusted, independent of relevance". Keeping
them separate means each stays individually explainable (a chunk's
citation can report BOTH its relevance score and its evidence level,
rather than one opaque blended number with no way to say which part came
from where — this is also what makes evidence-aware retrieval
"explainable", the other half of that resume claim).
"""

from __future__ import annotations

from dataclasses import replace
from typing import List

from ingestion.metadata_extractor import EVIDENCE_LEVEL_WEIGHTS
from retrieval.types import RetrievedChunk

# How much evidence level can discount an otherwise-relevant chunk's
# score. alpha=0.5 means the weakest evidence level (weight ~0.3-0.4)
# roughly halves the relevance score, never zeroes it out entirely — a
# highly relevant case report should still be able to outrank a barely
# relevant systematic review, just not a comparably relevant one. Tunable,
# documented rather than a hidden magic number.
EVIDENCE_WEIGHT_ALPHA = 0.5


def evidence_weight(evidence_level: str) -> float:
    return EVIDENCE_LEVEL_WEIGHTS.get(evidence_level, EVIDENCE_LEVEL_WEIGHTS["Unknown"])


def apply_evidence_ranking(
    chunks: List[RetrievedChunk], alpha: float = EVIDENCE_WEIGHT_ALPHA
) -> List[RetrievedChunk]:
    """Returns a NEW list re-sorted by relevance-times-evidence-weight,
    best first. Does not mutate the input chunks — .score on the returned
    objects is a fresh combined value; the pre-evidence-ranking relevance
    score is preserved separately in metadata (see retrieval/confidence.py,
    which needs both signals independently rather than one already-blended
    number)."""
    ranked = []
    for chunk in chunks:
        level = chunk.metadata.get("evidence_level", "")
        weight = evidence_weight(level)
        combined_score = chunk.score * (alpha + (1 - alpha) * weight)
        ranked.append(
            replace(
                chunk,
                score=combined_score,
                metadata={**chunk.metadata, "relevance_score": chunk.score},
            )
        )
    ranked.sort(key=lambda c: c.score, reverse=True)
    return ranked
