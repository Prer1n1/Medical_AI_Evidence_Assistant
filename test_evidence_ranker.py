"""Free/offline, pure functions — no storage, no API calls."""

from __future__ import annotations

from retrieval.evidence_ranker import apply_evidence_ranking, evidence_weight
from retrieval.types import RetrievedChunk


def test_evidence_weight_ordering():
    assert evidence_weight("Systematic Review / Meta-Analysis") > evidence_weight("Randomized Controlled Trial")
    assert evidence_weight("Randomized Controlled Trial") > evidence_weight("Case Report / Case Series")
    assert evidence_weight("totally unknown level") == evidence_weight("Unknown")
    print("OK: evidence weight ordering matches the clinical hierarchy")


def test_high_evidence_can_overtake_lower_at_similar_relevance():
    weak_but_high_evidence = RetrievedChunk(
        chunk_id="a", content="...", score=0.8,
        metadata={"evidence_level": "Systematic Review / Meta-Analysis"},
    )
    strong_but_low_evidence = RetrievedChunk(
        chunk_id="b", content="...", score=0.82,
        metadata={"evidence_level": "Case Report / Case Series"},
    )
    ranked = apply_evidence_ranking([strong_but_low_evidence, weak_but_high_evidence])
    assert ranked[0].chunk_id == "a", "a comparably-relevant systematic review should outrank a case report"
    print("OK: evidence ranking reorders comparably-relevant chunks by evidence tier")


def test_much_more_relevant_low_evidence_still_wins():
    """alpha=0.5 caps the discount — evidence level should never be able
    to fully override a MUCH more relevant match."""
    highly_relevant_low_evidence = RetrievedChunk(
        chunk_id="a", content="...", score=0.95,
        metadata={"evidence_level": "Case Report / Case Series"},
    )
    barely_relevant_high_evidence = RetrievedChunk(
        chunk_id="b", content="...", score=0.15,
        metadata={"evidence_level": "Systematic Review / Meta-Analysis"},
    )
    ranked = apply_evidence_ranking([barely_relevant_high_evidence, highly_relevant_low_evidence])
    assert ranked[0].chunk_id == "a"
    print("OK: evidence weighting doesn't override a much more relevant match")


def test_relevance_score_preserved_in_metadata():
    chunk = RetrievedChunk(chunk_id="a", content="...", score=0.6, metadata={"evidence_level": "Cohort Study"})
    ranked = apply_evidence_ranking([chunk])
    assert ranked[0].metadata["relevance_score"] == 0.6
    assert ranked[0].score != 0.6  # combined score, not the raw relevance score
    print("OK: pre-evidence-ranking relevance score preserved separately in metadata")


if __name__ == "__main__":
    test_evidence_weight_ordering()
    test_high_evidence_can_overtake_lower_at_similar_relevance()
    test_much_more_relevant_low_evidence_still_wins()
    test_relevance_score_preserved_in_metadata()
    print("All test_evidence_ranker.py checks passed.")
