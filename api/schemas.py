"""Request/response models for the REST API — kept separate from internal
dataclasses so internal refactors don't silently change the public API
shape. New fields vs. enterprise-agentic-rag: Citation carries
evidence_level/guideline_name/pmid/journal/doi; QueryResponse carries a
`confidence` block; a new PubMedIngestRequest/Response pair for the
PubMed connector.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)


class Citation(BaseModel):
    source: str
    section: Optional[str] = None
    category: Optional[str] = None
    evidence_level: Optional[str] = None
    guideline_name: Optional[str] = None
    effective_date: Optional[str] = None
    pmid: Optional[str] = None
    journal: Optional[str] = None
    doi: Optional[str] = None


class Confidence(BaseModel):
    label: str
    score: float
    reasoning: str
    independent_source_count: int


class QueryResponse(BaseModel):
    answer: str
    categories_queried: List[str]
    citations: List[Citation]
    confidence: Confidence


class IngestRequest(BaseModel):
    directory: str = "sample_data"
    doc_subtype: str = "clinical_guideline"


class IngestResponse(BaseModel):
    ingested_files: int
    skipped_unchanged: int
    deleted_files: int
    chunks_stored: int
    superseded_guideline_versions: List[str] = Field(
        default_factory=list,
        description="Sources marked is_current=false because a newer version of the same guideline was just ingested.",
    )


class PubMedIngestRequest(BaseModel):
    query: str = Field(..., min_length=1, description="A PubMed search query, e.g. 'hypertension AND RCT'.")
    retmax: int = Field(default=20, ge=1, le=100)


class PubMedIngestResponse(BaseModel):
    query: str
    records_fetched: int
    chunks_stored: int


class HealthResponse(BaseModel):
    status: str
    chunk_store_reachable: bool
    vector_store_reachable: bool
    chunk_count: int
