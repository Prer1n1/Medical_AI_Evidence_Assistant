"""Metadata extractor — enriches what loaders already capture (source,
title, author, section/page) with information no loader can infer purely
from file structure: a stable document ID, word count, a clinical
category, and — the field with no equivalent in a generic enterprise RAG
pipeline — an **evidence level**.

Evidence level is what "evidence ranking" (the resume bullet) actually
means in practice: not every retrieved passage is equally trustworthy. A
systematic review of 40 RCTs and one physician's case report can both
mention the same drug, but they don't deserve equal weight in an answer.
This module tags every chunk with where it sits on the evidence hierarchy,
so retrieval/evidence_ranker.py and retrieval/confidence.py have something
concrete to weight by.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from retry_utils import retry_openai_call

from .schema import Document

# --- Clinical category -------------------------------------------------
# Keyword matching is cheap, deterministic, explainable — kept as the
# FALLBACK (used only if the LLM call fails), same "LLM primary, keyword
# safety net" architecture the enterprise project settled on after finding
# real-world vocabulary a hand-picked keyword list doesn't generalize to.
_CATEGORY_KEYWORDS = {
    "Cardiology": {"cardiac", "heart", "coronary", "hypertension", "arrhythmia", "myocardial"},
    "Oncology": {"cancer", "tumor", "tumour", "chemotherapy", "oncology", "malignant", "carcinoma"},
    "Infectious Disease": {"infection", "bacterial", "viral", "antibiotic", "sepsis", "pathogen"},
    "Pharmacology": {"dosage", "drug interaction", "pharmacokinetics", "contraindication", "adverse effect"},
    "Emergency Medicine": {"emergency", "triage", "trauma", "resuscitation", "acute"},
}
CATEGORIES = list(_CATEGORY_KEYWORDS.keys()) + ["General"]

# --- Evidence level ------------------------------------------------------
# Ordered strongest -> weakest, per the standard clinical evidence
# hierarchy (systematic reviews of RCTs sit above individual RCTs, which
# sit above observational designs, which sit above anecdote/opinion).
# EVIDENCE_LEVEL_WEIGHTS is what retrieval/evidence_ranker.py and
# retrieval/confidence.py actually multiply scores by.
EVIDENCE_LEVELS = [
    "Systematic Review / Meta-Analysis",
    "Randomized Controlled Trial",
    "Cohort Study",
    "Case-Control Study",
    "Case Report / Case Series",
    "Expert Opinion / Clinical Guideline",
]
EVIDENCE_LEVEL_WEIGHTS = {
    "Systematic Review / Meta-Analysis": 1.0,
    "Randomized Controlled Trial": 0.9,
    "Cohort Study": 0.7,
    "Case-Control Study": 0.6,
    "Case Report / Case Series": 0.4,
    "Expert Opinion / Clinical Guideline": 0.5,
    "Unknown": 0.3,
}
_DEFAULT_EVIDENCE_LEVEL = "Expert Opinion / Clinical Guideline"

# PubMed's own PublicationType field (see ingestion/connectors/pubmed.py)
# is ground-truth structured data when it's present — same "trust
# authoritative structured metadata over inferred classification"
# principle the enterprise project applied to CSV department columns.
PUBMED_PUBLICATION_TYPE_MAP = {
    "systematic review": "Systematic Review / Meta-Analysis",
    "meta-analysis": "Systematic Review / Meta-Analysis",
    "randomized controlled trial": "Randomized Controlled Trial",
    "controlled clinical trial": "Randomized Controlled Trial",
    "observational study": "Cohort Study",
    "comparative study": "Cohort Study",
    "multicenter study": "Cohort Study",
    "case reports": "Case Report / Case Series",
    "practice guideline": "Expert Opinion / Clinical Guideline",
    "guideline": "Expert Opinion / Clinical Guideline",
    "consensus development conference": "Expert Opinion / Clinical Guideline",
}

_WORD_RE = re.compile(r"[a-zA-Z]{3,}")
_CLASSIFIER_MODEL = "gpt-4o-mini"
_MAX_CHARS_FOR_CLASSIFICATION = 3000

_EVIDENCE_KEYWORDS = {
    "Systematic Review / Meta-Analysis": {"systematic review", "meta-analysis", "meta analysis"},
    "Randomized Controlled Trial": {"randomized controlled trial", "randomised controlled trial", "double-blind"},
    "Cohort Study": {"cohort study", "prospective cohort", "retrospective cohort", "observational study"},
    "Case-Control Study": {"case-control", "case control study"},
    "Case Report / Case Series": {"case report", "case series"},
    "Expert Opinion / Clinical Guideline": {"clinical guideline", "practice guideline", "expert consensus", "recommendation"},
}


class CategoryDecision(BaseModel):
    category: str = Field(
        description=(
            f"The single best-fitting clinical category for this document excerpt. "
            f"Must be exactly one of: {CATEGORIES}. Use 'General' only if none of "
            f"the specific categories genuinely fit."
        )
    )


class EvidenceLevelDecision(BaseModel):
    evidence_level: str = Field(
        description=(
            f"The single best-fitting evidence level for this document excerpt, "
            f"per the standard clinical evidence hierarchy. Must be exactly one "
            f"of: {EVIDENCE_LEVELS}. If the excerpt is a clinical guideline, "
            f"protocol, or expert recommendation with no described study design, "
            f"use 'Expert Opinion / Clinical Guideline'."
        )
    )


def stable_doc_id(source: str) -> str:
    """Stable ID for the ORIGINAL source (file path or 'pubmed:<pmid>') —
    every page/section/table/figure loaded from the same source shares
    this ID, so they can be traced back to one document for citation
    grouping or dedup."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _keyword_classify_category(text: str) -> str:
    lowered = text.lower()
    scores = {
        category: sum(1 for kw in keywords if kw in lowered)
        for category, keywords in _CATEGORY_KEYWORDS.items()
    }
    best_category, best_score = max(scores.items(), key=lambda kv: kv[1])
    return best_category if best_score > 0 else "General"


def _keyword_classify_evidence_level(text: str) -> str:
    lowered = text.lower()
    for level, keywords in _EVIDENCE_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return level
    return _DEFAULT_EVIDENCE_LEVEL


@retry_openai_call
def _invoke_structured(structured_llm, prompt: str):
    return structured_llm.invoke(prompt)


def _llm_classify_category(text: str, llm=None) -> Optional[str]:
    try:
        llm = llm or ChatOpenAI(model=_CLASSIFIER_MODEL, temperature=0)
        structured_llm = llm.with_structured_output(CategoryDecision)
        decision = _invoke_structured(
            structured_llm,
            f"Categories: {CATEGORIES}\n\nDocument excerpt:\n"
            f"{text[:_MAX_CHARS_FOR_CLASSIFICATION]}\n\nWhich single category best fits this content?",
        )
        return decision.category if decision.category in CATEGORIES else None
    except Exception:
        return None


def _llm_classify_evidence_level(text: str, llm=None) -> Optional[str]:
    try:
        llm = llm or ChatOpenAI(model=_CLASSIFIER_MODEL, temperature=0)
        structured_llm = llm.with_structured_output(EvidenceLevelDecision)
        decision = _invoke_structured(
            structured_llm,
            f"Evidence levels: {EVIDENCE_LEVELS}\n\nDocument excerpt:\n"
            f"{text[:_MAX_CHARS_FOR_CLASSIFICATION]}\n\n"
            f"Which single evidence level best describes this content's study design "
            f"or nature (guideline vs. trial vs. case report, etc.)?",
        )
        return decision.evidence_level if decision.evidence_level in EVIDENCE_LEVELS else None
    except Exception:
        return None


def classify_category(text: str, hint: Optional[str] = None, llm=None, use_llm: bool = True) -> str:
    if hint and hint in CATEGORIES:
        return hint
    if use_llm:
        result = _llm_classify_category(text, llm=llm)
        if result:
            return result
    return _keyword_classify_category(text)


def classify_evidence_level(
    text: str, hint: Optional[str] = None, llm=None, use_llm: bool = True
) -> str:
    """hint: ground-truth evidence level already known from structured
    source data (PubMed's PublicationType field, via
    PUBMED_PUBLICATION_TYPE_MAP in the connector) — trusted outright,
    never overridden by classifying free text, same principle the
    enterprise project applied to its CSV department-column hint."""
    if hint and hint in EVIDENCE_LEVELS:
        return hint
    if use_llm:
        result = _llm_classify_evidence_level(text, llm=llm)
        if result:
            return result
    return _keyword_classify_evidence_level(text)


def enrich(document: Document, llm=None, use_llm: bool = True) -> Document:
    """Adds doc_id, word_count, category, and evidence_level into
    metadata.extra. Mutates and returns the same Document."""
    extra = document.metadata.extra
    extra.setdefault("doc_id", stable_doc_id(document.metadata.source))
    extra["word_count"] = len(_WORD_RE.findall(document.content))
    extra["category"] = classify_category(
        document.content, hint=extra.get("category_hint"), llm=llm, use_llm=use_llm
    )
    extra["evidence_level"] = classify_evidence_level(
        document.content, hint=extra.get("evidence_level_hint"), llm=llm, use_llm=use_llm
    )
    return document


def enrich_all(documents: List[Document], llm=None, use_llm: bool = True) -> List[Document]:
    return [enrich(doc, llm=llm, use_llm=use_llm) for doc in documents]
