"""Ingestion pipeline — wires the pieces together: tracker (skip
unchanged) -> loaders (format -> common schema) -> image captioning
(optional) -> metadata extractor (category + evidence level) -> chunking.

Deliberately does NOT port the enterprise project's prompt-injection
defense or PII redaction stages — those are enterprise-security features
with no equivalent claim in the "Medical AI Evidence Assistant" scope this
project targets (see docs/design-decisions.md, "Scope"). What IS new here
that the enterprise pipeline has no equivalent of: `doc_subtype` tagging
(clinical_guideline vs. treatment_protocol vs. research_paper — the
research_paper case comes from ingestion/connectors/pubmed.py, which
builds Documents directly rather than going through this file-based path)
and optional image captioning for extracted figures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

from .chunking import Chunk, chunk_documents
from .guideline_versioning import derive_effective_date_hint, derive_guideline_name
from .image_captioner import caption_image
from .loaders import load_document
from .metadata_extractor import enrich_all
from .schema import canonical_source
from .tracker import IngestionTracker

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm"}
DEFAULT_DOC_SUBTYPE = "clinical_guideline"


@dataclass
class IngestionResult:
    chunks: List[Chunk] = field(default_factory=list)
    ingested_files: List[str] = field(default_factory=list)
    skipped_unchanged: List[str] = field(default_factory=list)
    deleted_files: List[str] = field(default_factory=list)
    # Real Path objects (tracker.mark_ingested() needs to reopen the file to
    # hash it), NOT yet marked in the tracker — marking is the CALLER's job,
    # done only after chunks are actually persisted to both stores. See
    # docs/design-decisions.md ("Reliability") for the bug this ordering
    # avoids (ported reasoning from the enterprise project, still applies
    # here unchanged).
    pending_mark: List[Path] = field(default_factory=list)


def ingest_directory(
    directory: Union[str, Path],
    tracker: IngestionTracker = None,
    embeddings=None,
    classifier_llm=None,
    use_llm_classifier: bool = True,
    caption_llm=None,
    use_image_captioning: bool = True,
    doc_subtype: str = DEFAULT_DOC_SUBTYPE,
) -> IngestionResult:
    directory = Path(directory)
    tracker = tracker or IngestionTracker()

    all_files = [p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    result = IngestionResult(deleted_files=tracker.find_deleted(all_files, root=directory))

    docs_to_chunk = []
    files_pending_mark = []

    for file_path in all_files:
        decision = tracker.check(file_path)
        if not decision.should_ingest:
            result.skipped_unchanged.append(canonical_source(file_path))
            continue

        documents = load_document(file_path)
        for doc in documents:
            doc.metadata.extra.setdefault("doc_subtype", doc_subtype)
            # Versioning only makes sense for guidelines/protocols, which
            # get periodically revised — a research paper is immutable
            # once published, so PubMed-sourced Documents never set these
            # (see ingestion/connectors/pubmed.py).
            if doc.metadata.extra["doc_subtype"] in ("clinical_guideline", "treatment_protocol"):
                doc.metadata.extra.setdefault("guideline_name", derive_guideline_name(file_path.stem))
                doc.metadata.extra.setdefault("effective_date", derive_effective_date_hint(file_path.stem))

        if use_image_captioning:
            for doc in documents:
                if not doc.metadata.extra.get("is_image"):
                    continue
                caption = caption_image(doc.metadata.extra["image_path"], llm=caption_llm)
                if caption:
                    doc.content = caption

        docs_to_chunk.extend(
            enrich_all(documents, llm=classifier_llm, use_llm=use_llm_classifier)
        )
        files_pending_mark.append(file_path)

    if docs_to_chunk:
        result.chunks = chunk_documents(docs_to_chunk, embeddings=embeddings)

    for file_path in files_pending_mark:
        result.pending_mark.append(file_path)
        result.ingested_files.append(canonical_source(file_path))

    return result
