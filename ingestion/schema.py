"""Common document schema every format-specific loader normalizes into.

Why this exists: PDF/DOCX/HTML each have wildly different native
structures (pages, paragraphs+headings, DOM tags) — and medical PDFs add
tables and figures on top of that. Every downstream component (chunking,
metadata, retrieval) should only ever see ONE shape, not four+. This is
what makes the pipeline format-agnostic past this point, and lets the
PubMed connector (ingestion/connectors/pubmed.py) hand off abstracts/full
text through the exact same pipeline as an uploaded PDF.

Ported near-verbatim from the enterprise-agentic-rag project — this shape
and canonical_source()'s cross-platform identity fix are format/domain
agnostic, no medical-specific change needed here. Medical-specific fields
(evidence_level, guideline_name, pmid, ...) live in DocumentMetadata.extra,
same as the enterprise project's category/language/doc_id fields did.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union


def canonical_source(file_path: Union[str, Path]) -> str:
    """The identity string used for a source file EVERYWHERE it matters
    for correctness: metadata.source, the incremental tracker's keys, and
    chunk_id derivation. Always forward-slash (POSIX), regardless of OS —
    a Windows dev machine ingesting, then a Linux Docker container
    re-ingesting the same file, must agree this is the SAME file. Only
    used for IDENTITY strings — actually opening a file still uses the
    native str(file_path)."""
    return Path(file_path).as_posix()


@dataclass
class DocumentMetadata:
    source: str                    # file path (or a synthetic "pubmed:<pmid>" for PubMed records)
    doc_type: str                  # "pdf" | "docx" | "html" | "pubmed"
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[str] = None
    section: Optional[str] = None  # heading/section name, or "Table N" / "Figure N"
    page_number: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    """One loaded unit of content (a PDF page, a DOCX section, a PubMed
    abstract, a table, an image caption) before chunking splits it further."""

    content: str
    metadata: DocumentMetadata

    def content_hash(self) -> str:
        """SHA-256 of the content — used later by the incremental
        ingestion tracker to detect whether a document actually changed."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()
