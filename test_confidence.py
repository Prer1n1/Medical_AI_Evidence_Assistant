"""Free/offline, pure functions."""

from __future__ import annotations

from retrieval.confidence import compute_confidence
from retrieval.types import RetrievedChunk


def _chunk(chunk_id, doc_id, evidence_level, relevance_score):
    return RetrievedChunk(
        chunk_id=chunk_id, content="...", score=relevance_score,
        metadata={"evidence_level": evidence_level, "doc_id": doc_id, "relevance_score": relevance_score},
    )


def test_no_evidence_is_low_confidence():
    result = compute_confidence([])
    assert result.label == "Low"
    assert result.independent_source_count == 0
    print("OK: empty evidence set is Low confidence")


def test_single_weak_source_is_lower_confidence_than_multiple_strong():
    single_case_report = [_chunk("a", "doc1", "Case Report / Case Series", 0.5)]
    multiple_strong_agreeing = [
        _chunk("b", "doc2", "Systematic Review / Meta-Analysis", 0.9),
        _chunk("c", "doc3", "Randomized Controlled Trial", 0.85),
    ]
    low = compute_confidence(single_case_report)
    high = compute_confidence(multiple_strong_agreeing)
    assert high.score > low.score
    assert high.independent_source_count == 2
    assert low.independent_source_count == 1
    print(f"OK: single weak source ({low.label}, {low.score}) scores lower than multiple strong agreeing sources ({high.label}, {high.score})")


def test_same_source_multiple_chunks_counts_once():
    """Two chunks from the SAME paper shouldn't count as two independent
    sources agreeing — that's not evidence agreement, it's one source
    cited twice."""
    chunks = [
        _chunk("a", "doc1", "Randomized Controlled Trial", 0.8),
        _chunk("b", "doc1", "Randomized Controlled Trial", 0.75),
    ]
    result = compute_confidence(chunks)
    assert result.independent_source_count == 1
    print("OK: chunks from the same doc_id count as one independent source")


def test_high_confidence_achievable():
    chunks = [
        _chunk("a", "doc1", "Systematic Review / Meta-Analysis", 0.9),
        _chunk("b", "doc2", "Randomized Controlled Trial", 0.85),
        _chunk("c", "doc3", "Randomized Controlled Trial", 0.8),
    ]
    result = compute_confidence(chunks)
    assert result.label == "High", f"expected High, got {result.label} ({result.score})"
    print(f"OK: strong multi-source agreement reaches High confidence ({result.score})")


if __name__ == "__main__":
    test_no_evidence_is_low_confidence()
    test_single_weak_source_is_lower_confidence_than_multiple_strong()
    test_same_source_multiple_chunks_counts_once()
    test_high_confidence_achievable()
    print("All test_confidence.py checks passed.")
