"""Vector store wrapper — Chroma, persisted locally to disk. Same choice
as enterprise-agentic-rag (metadata-filtered similarity search in one
component) but with a different embedding_function: a LOCAL biomedical
sentence-transformer instead of OpenAI's general-purpose embeddings.

Why local biomedical embeddings, not OpenAI text-embedding-3: this
project's resume claim is specifically "biomedical embeddings" — a model
actually pretrained/fine-tuned on biomedical text (PubMedBERT, further
tuned on MS MARCO for retrieval) captures domain vocabulary ("MI" as
myocardial infarction, drug-class relationships, dosage units) that a
general-purpose embedding model has no particular advantage on. Real
tradeoff, named honestly: CPU inference is slower per-document than an
API call and the first run downloads the model (~400MB, one-time,
huggingface's cache), but it's free per-call and needs no API key —
embeddings are the one call site in this entire project that never touches
OpenAI at all. See docs/design-decisions.md ("Embeddings").
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from langchain_chroma import Chroma
from langchain_core.documents import Document as LangchainDocument
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBEDDING_MODEL
from ingestion.chunking import Chunk
from storage.chunk_store import compute_chunk_id

DEFAULT_PERSIST_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "medical_evidence"

_embeddings_singleton: Optional[HuggingFaceEmbeddings] = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Cached — loading the model checkpoint is the slow part (first call
    only; the file is disk-cached by huggingface afterward), so every
    caller in the process shares ONE loaded model instead of re-loading it
    per HybridRetriever/ChunkStore construction."""
    global _embeddings_singleton
    if _embeddings_singleton is None:
        _embeddings_singleton = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_singleton


def get_vector_store(embeddings=None, persist_directory: Union[str, Path] = DEFAULT_PERSIST_DIR) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=str(persist_directory),
    )


def _to_langchain_document(chunk: Chunk) -> LangchainDocument:
    m = chunk.metadata
    # Chroma metadata values must be str/int/float/bool — flatten or drop
    # anything nested. is_current stored as int (0/1): Chroma's `where`
    # filter compares equality on the stored type, and bool True == 1 in
    # Python but Chroma's filter engine wants an exact type match.
    metadata = {
        "chunk_id": compute_chunk_id(chunk),
        "source": m.source,
        "doc_type": m.doc_type,
        "doc_subtype": m.extra.get("doc_subtype") or "",
        "title": m.title or "",
        "section": m.section or "",
        "page_number": m.page_number or 0,
        "doc_id": m.extra.get("doc_id", ""),
        "category": m.extra.get("category", ""),
        "evidence_level": m.extra.get("evidence_level", ""),
        "guideline_name": m.extra.get("guideline_name") or "",
        "effective_date": m.extra.get("effective_date") or "",
        "is_current": 1,
        "pmid": m.extra.get("pmid") or "",
        "journal": m.extra.get("journal") or "",
        "doi": m.extra.get("doi") or "",
    }
    return LangchainDocument(page_content=chunk.content, metadata=metadata)


def add_chunks(vector_store: Chroma, chunks: List[Chunk]) -> List[str]:
    if not chunks:
        return []
    ids = [compute_chunk_id(c) for c in chunks]
    docs = [_to_langchain_document(c) for c in chunks]
    vector_store.add_documents(documents=docs, ids=ids)
    return ids


def delete_by_source(vector_store: Chroma, source: str) -> None:
    vector_store.delete(where={"source": source})


def set_current(vector_store: Chroma, source: str, is_current: bool) -> int:
    """Flips is_current on every chunk of one source, in place — no
    higher-level langchain API does a metadata-only update by filter, so
    this reaches into Chroma's own collection object directly (the one
    low-level escape hatch used in this whole storage layer, and
    deliberately confined to this one function). Mirrors
    ChunkStore.set_current(); called together by
    ingestion.guideline_versioning.apply_versioning() so both stores
    agree about which version is current."""
    existing = vector_store.get(where={"source": source})
    ids = existing.get("ids", [])
    if not ids:
        return 0
    updated_metadatas = [{**meta, "is_current": 1 if is_current else 0} for meta in existing["metadatas"]]
    vector_store._collection.update(ids=ids, metadatas=updated_metadatas)
    return len(ids)


def similarity_search(
    vector_store: Chroma,
    query: str,
    k: int = 5,
    filter: Optional[dict] = None,
) -> List[LangchainDocument]:
    return vector_store.similarity_search(query, k=k, filter=filter)
