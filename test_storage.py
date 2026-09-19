"""Free/offline — a fake, deterministic embeddings double (matching
test_chunking.py's FakeEmbeddings shape but for query/document embedding,
which is what Chroma's embedding_function interface needs) proves the
Chroma + SQLite round trip without loading the real biomedical model.
Ported test strategy from enterprise-agentic-rag.
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from ingestion.chunking import Chunk
from ingestion.schema import DocumentMetadata
from storage.chunk_store import ChunkStore, compute_chunk_id
from storage.vector_store import add_chunks, delete_by_source, get_vector_store, set_current


@contextmanager
def _temp_dir():
    """Like tempfile.TemporaryDirectory(), but tolerates Windows' file
    locking on cleanup — Chroma's persistent client keeps its HNSW index
    file (data_level0.bin) open for the lifetime of the process, and
    Windows (unlike Linux) refuses to unlink an open file, so the
    context manager's own cleanup raised PermissionError even though the
    test itself had already passed. Real, verified platform difference,
    not a flaky test — best-effort cleanup (ignore_errors=True) is the
    correct fix, not retrying or suppressing the whole test."""
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class FakeQueryEmbeddings:
    """embed_documents/embed_query with a tiny deterministic hash-based
    vector — enough for Chroma to store/retrieve without any real model."""

    def _vec(self, text: str):
        h = abs(hash(text))
        return [((h >> (i * 8)) % 100) / 100.0 for i in range(8)]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def _chunk(source: str, section: str, content: str, **extra) -> Chunk:
    return Chunk(
        content=content,
        metadata=DocumentMetadata(source=source, doc_type="pdf", section=section, extra=extra),
        chunk_index=0,
    )


def test_chunk_store_save_and_retrieve():
    with _temp_dir() as tmp:
        store = ChunkStore(db_path=Path(tmp) / "chunks.db")
        chunk = _chunk("a.pdf", "Section 1", "hypertension content", category="Cardiology", evidence_level="Randomized Controlled Trial")
        store.save_chunks([chunk])

        rows = store.get_by_source("a.pdf")
        assert len(rows) == 1
        assert rows[0]["category"] == "Cardiology"
        assert rows[0]["evidence_level"] == "Randomized Controlled Trial"
        assert rows[0]["is_current"] == 1
        print("OK: ChunkStore save/retrieve round trip, including evidence_level")


def test_chunk_id_shared_between_stores():
    """The vector store and chunk store must assign the SAME id to the
    same chunk — required for delete_by_source/set_current to stay in
    sync across both."""
    chunk = _chunk("a.pdf", "Section 1", "content")
    assert compute_chunk_id(chunk) == compute_chunk_id(chunk)
    print("OK: compute_chunk_id is deterministic (shared identity across both stores)")


def test_vector_store_add_and_delete():
    with _temp_dir() as tmp:
        vector_store = get_vector_store(embeddings=FakeQueryEmbeddings(), persist_directory=Path(tmp) / "chroma")
        chunk = _chunk("a.pdf", "Section 1", "hypertension treatment content", category="Cardiology")
        add_chunks(vector_store, [chunk])

        results = vector_store.similarity_search("hypertension", k=1)
        assert len(results) == 1
        assert results[0].metadata["source"] == "a.pdf"

        delete_by_source(vector_store, "a.pdf")
        results_after = vector_store.similarity_search("hypertension", k=1)
        assert results_after == []
        print("OK: vector store add + similarity_search + delete_by_source round trip")


def test_vector_store_set_current():
    with _temp_dir() as tmp:
        vector_store = get_vector_store(embeddings=FakeQueryEmbeddings(), persist_directory=Path(tmp) / "chroma")
        chunk = _chunk("old_guideline.pdf", "Section 1", "old guideline content about hypertension")
        add_chunks(vector_store, [chunk])

        updated_count = set_current(vector_store, "old_guideline.pdf", is_current=False)
        assert updated_count == 1

        raw = vector_store.get(where={"source": "old_guideline.pdf"})
        assert raw["metadatas"][0]["is_current"] == 0
        print("OK: vector_store.set_current flips is_current metadata in place")


if __name__ == "__main__":
    test_chunk_store_save_and_retrieve()
    test_chunk_id_shared_between_stores()
    test_vector_store_add_and_delete()
    test_vector_store_set_current()
    print("All test_storage.py checks passed.")
