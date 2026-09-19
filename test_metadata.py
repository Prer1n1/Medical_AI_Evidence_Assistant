"""Part 1 (free/offline): the keyword fallback classifiers for category
and evidence_level. Part 2 (gated on OPENAI_API_KEY, real LLM calls):
the LLM-primary classifiers, verified against real WHO guideline text.
"""

from __future__ import annotations

import os

import config  # noqa: F401 — import side effect: load_dotenv(), so OPENAI_API_KEY from .env reaches os.environ
from ingestion.metadata_extractor import (
    CATEGORIES,
    EVIDENCE_LEVELS,
    classify_category,
    classify_evidence_level,
    stable_doc_id,
)


def test_keyword_category_fallback():
    assert classify_category("Patient presents with chest pain and coronary artery disease.", use_llm=False) == "Cardiology"
    assert classify_category("The patient was started on chemotherapy for stage III carcinoma.", use_llm=False) == "Oncology"
    assert classify_category("General front matter with no clinical content at all.", use_llm=False) == "General"
    print("OK: keyword category fallback")


def test_keyword_evidence_level_fallback():
    assert classify_evidence_level("This systematic review and meta-analysis pooled 40 trials.", use_llm=False) == "Systematic Review / Meta-Analysis"
    assert classify_evidence_level("A randomized controlled trial of drug X versus placebo.", use_llm=False) == "Randomized Controlled Trial"
    assert classify_evidence_level("We report a single case report of an unusual presentation.", use_llm=False) == "Case Report / Case Series"
    assert classify_evidence_level("No matching keyword at all here.", use_llm=False) == "Expert Opinion / Clinical Guideline"
    print("OK: keyword evidence-level fallback")


def test_hint_overrides_classification():
    """Ground-truth structured hints (PubMed's PublicationType, via the
    connector) win outright — never overridden by re-classifying free
    text, same principle the enterprise project applied to CSV hints."""
    assert classify_category("irrelevant text", hint="Oncology", use_llm=False) == "Oncology"
    assert classify_evidence_level("irrelevant text", hint="Randomized Controlled Trial", use_llm=False) == "Randomized Controlled Trial"
    # An unknown hint is ignored, falls through to classification
    assert classify_category("chest pain and coronary disease", hint="NotARealCategory", use_llm=False) == "Cardiology"
    print("OK: structured hints override classification")


def test_stable_doc_id_deterministic():
    assert stable_doc_id("pubmed:12345") == stable_doc_id("pubmed:12345")
    assert stable_doc_id("pubmed:12345") != stable_doc_id("pubmed:54321")
    print("OK: stable_doc_id is deterministic")


def test_live_llm_classification():
    if not os.getenv("OPENAI_API_KEY"):
        print("SKIPPED (no OPENAI_API_KEY): live LLM classification")
        return

    category = classify_category(
        "WHO recommends initiation of pharmacological antihypertensive treatment for adults "
        "with confirmed hypertension and systolic blood pressure >=140 mmHg."
    )
    # Real finding from live verification: this excerpt is genuinely
    # ambiguous between Cardiology (hypertension is a cardiac condition)
    # and Pharmacology (the sentence itself is about drug-treatment
    # initiation) — the LLM classifier picked Pharmacology, a legitimate
    # reading, not a wrong one. Documented in docs/design-decisions.md
    # ("Category taxonomy overlap") rather than forcing the test to assert
    # one "correct" answer to an inherently ambiguous excerpt.
    assert category in {"Cardiology", "Pharmacology"}, f"expected Cardiology or Pharmacology, got {category}"

    level = classify_evidence_level(
        "This systematic review and meta-analysis of randomized controlled trials found that "
        "antihypertensive treatment reduces cardiovascular events."
    )
    assert level == "Systematic Review / Meta-Analysis", f"expected Systematic Review / Meta-Analysis, got {level}"
    print("OK: live LLM classification against real guideline text")


if __name__ == "__main__":
    assert set(CATEGORIES) >= {"Cardiology", "Oncology", "General"}
    assert "Randomized Controlled Trial" in EVIDENCE_LEVELS
    test_keyword_category_fallback()
    test_keyword_evidence_level_fallback()
    test_hint_overrides_classification()
    test_stable_doc_id_deterministic()
    test_live_llm_classification()
    print("All test_metadata.py checks passed.")
