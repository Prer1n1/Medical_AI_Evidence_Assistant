"""Chunking — hierarchical hybrid strategy, ported near-verbatim from
enterprise-agentic-rag. The mechanism (structure-aware boundaries from the
loaders -> semantic split for prose -> row/table batching for tabular
content) is domain-agnostic; the only real change here is which
embeddings object callers pass in (storage/vector_store.py's local
biomedical HuggingFaceEmbeddings instead of OpenAIEmbeddings) — this
module doesn't care, it just calls .embed_documents() on whatever it's
given.

1. Structure-aware boundaries already exist: loaders already split content
   by page/section/table/figure, so that work is done before this module runs.
2. Prose Documents get SEMANTIC chunking: split into sentences, embed each
   one, and cut where consecutive sentences suddenly diverge in meaning —
   not at a fixed token count.
3. Tabular Documents (dosage tables, DOCX/HTML tables) get BATCHED, not
   semantically chunked. Image Documents (figure captions) are short
   enough to pass through as single chunks untouched by either path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import groupby
from typing import List, Optional

import numpy as np

from .schema import Document, DocumentMetadata

MAX_TABLE_CHUNK_CHARS = 1500
SEMANTIC_BREAKPOINT_PERCENTILE = 90  # cut at the sharpest 10% of meaning "jumps"
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    content: str
    metadata: DocumentMetadata
    chunk_index: int


def _is_tabular(document: Document) -> bool:
    return document.metadata.extra.get("is_table", False)


def _is_image(document: Document) -> bool:
    return document.metadata.extra.get("is_image", False)


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]


def _cosine_distance(a: List[float], b: List[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    similarity = np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    return 1 - similarity


def _embed_documents(embeddings, sentences: List[str]) -> List[List[float]]:
    """No retry_openai_call wrapper here anymore — the default embeddings
    object is now a local sentence-transformers model (see
    storage/vector_store.py), which has no network call to retry. Callers
    that DO pass an OpenAI-backed embeddings object are still free to; this
    function just doesn't assume one specific provider's transient-error
    types the way the enterprise version could."""
    return embeddings.embed_documents(sentences)


def semantic_split(
    text: str,
    embeddings,
    breakpoint_percentile: int = SEMANTIC_BREAKPOINT_PERCENTILE,
) -> List[str]:
    """Embed every sentence, measure cosine distance between each
    consecutive pair, and cut wherever that distance is in the top
    (100 - breakpoint_percentile)% — an unusually large jump in meaning,
    signaling a real topic shift rather than just a new sentence."""
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [text] if text.strip() else []

    vectors = _embed_documents(embeddings, sentences)
    distances = [_cosine_distance(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
    threshold = np.percentile(distances, breakpoint_percentile)

    pieces: List[str] = []
    current = [sentences[0]]
    for i, dist in enumerate(distances):
        if dist > threshold:
            pieces.append(" ".join(current))
            current = [sentences[i + 1]]
        else:
            current.append(sentences[i + 1])
    pieces.append(" ".join(current))
    return pieces


def _chunk_prose(documents: List[Document], embeddings) -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc in documents:
        for piece in semantic_split(doc.content, embeddings):
            chunks.append(Chunk(content=piece, metadata=doc.metadata, chunk_index=len(chunks)))
    return chunks


def _chunk_images(documents: List[Document]) -> List[Chunk]:
    """A figure caption is already a single short, self-contained unit —
    semantic-splitting a one-paragraph caption would be pointless work,
    and batching figures together like table rows would lose the
    per-figure image_path each chunk needs to stay individually
    retrievable and displayable."""
    return [Chunk(content=doc.content, metadata=doc.metadata, chunk_index=0) for doc in documents]


def _combine_batch(docs: List[Document]) -> Chunk:
    combined = "\n---\n".join(d.content for d in docs)
    first, last = docs[0], docs[-1]
    section = (
        first.metadata.section
        if len(docs) == 1
        else f"{first.metadata.section} .. {last.metadata.section}"
    )
    metadata = DocumentMetadata(
        source=first.metadata.source,
        doc_type=first.metadata.doc_type,
        title=first.metadata.title,
        author=first.metadata.author,
        created_date=first.metadata.created_date,
        section=section,
        extra={**first.metadata.extra, "batched_rows": len(docs)},
    )
    return Chunk(content=combined, metadata=metadata, chunk_index=0)


def _source_category_key(document: Document):
    return document.metadata.source, document.metadata.extra.get("category", "")


def _chunk_tabular(documents: List[Document], max_chars: int = MAX_TABLE_CHUNK_CHARS) -> List[Chunk]:
    """Groups consecutive table rows from the same (source, category) up
    to a size cap — never one chunk per row, never a whole large table in
    one chunk."""
    chunks: List[Chunk] = []
    sorted_docs = sorted(documents, key=_source_category_key)
    for _, group_iter in groupby(sorted_docs, key=_source_category_key):
        group = list(group_iter)
        buffer: List[Document] = []
        buffer_len = 0
        for doc in group:
            if buffer and buffer_len + len(doc.content) > max_chars:
                chunks.append(_combine_batch(buffer))
                buffer, buffer_len = [], 0
            buffer.append(doc)
            buffer_len += len(doc.content)
        if buffer:
            chunks.append(_combine_batch(buffer))

    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i
    return chunks


def chunk_documents(
    documents: List[Document],
    embeddings: Optional[object] = None,
    max_table_chunk_chars: int = MAX_TABLE_CHUNK_CHARS,
) -> List[Chunk]:
    if embeddings is None:
        from storage.vector_store import get_embeddings

        embeddings = get_embeddings()

    tabular = [d for d in documents if _is_tabular(d)]
    images = [d for d in documents if _is_image(d)]
    prose = [d for d in documents if not _is_tabular(d) and not _is_image(d)]
    combined = (
        _chunk_tabular(tabular, max_table_chunk_chars)
        + _chunk_images(images)
        + _chunk_prose(prose, embeddings)
    )

    # Reindex once, across the full combined list — each helper indexes
    # its own output from 0 independently, so chunk_index isn't unique
    # across the concatenated list until this runs (compute_chunk_id uses
    # it, and a duplicate index paired with a duplicate section string
    # would collide two different chunks onto the same chunk_id).
    for i, chunk in enumerate(combined):
        chunk.chunk_index = i
    return combined
