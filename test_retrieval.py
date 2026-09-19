"""Live — real local biomedical embeddings (free, but a real model/CPU
cost, not mocked) and real LLM classification during ingestion
(OPENAI_API_KEY required for the category/evidence-level classifier to
run at full quality, not just its keyword fallback). Ingests the real WHO
hypertension guideline PDF into temporary Chroma + SQLite storage, then
exercises HybridRetriever end-to-end: hybrid fusion, evidence ranking, and
the only_current filter.
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import config  # noqa: F401 — loads .env
from ingestion.pipeline import ingest_directory
from ingestion.tracker import IngestionTracker
from retrieval.hybrid_retriever import HybridRetriever
from storage.chunk_store import ChunkStore
from storage.vector_store import add_chunks, get_embeddings, get_vector_store, set_current


@contextmanager
def _temp_dir():
    """See test_storage.py's _temp_dir for why plain
    tempfile.TemporaryDirectory() fails here on Windows (Chroma keeps its
    index file open)."""
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)

GUIDELINES_DIR = Path("sample_data/guidelines")


def _build_retriever(tmp: Path):
    embeddings = get_embeddings()
    chunk_store = ChunkStore(db_path=tmp / "chunks.db")
    vector_store = get_vector_store(embeddings=embeddings, persist_directory=tmp / "chroma")
    tracker = IngestionTracker(db_path=tmp / "manifest.db")

    result = ingest_directory(GUIDELINES_DIR, tracker=tracker, embeddings=embeddings, use_image_captioning=False)
    chunk_store.save_chunks(result.chunks)
    add_chunks(vector_store, result.chunks)
    for file_path in result.pending_mark:
        tracker.mark_ingested(file_path)

    return HybridRetriever(vector_store, chunk_store), chunk_store, vector_store


def test_hybrid_retrieve_real_corpus():
    with _temp_dir() as tmp:
        retriever, _, _ = _build_retriever(Path(tmp))
        results = retriever.retrieve("target blood pressure for patients with cardiovascular disease", top_k=5)
        assert len(results) > 0, "expected real results from the ingested guideline"
        assert any("130" in r.content for r in results), "expected the actual <130 mmHg target to surface"
        assert all(r.metadata.get("evidence_level") for r in results)
        print(f"OK: hybrid retrieval returned {len(results)} results, including the real BP target passage")


def test_only_current_filter_excludes_superseded():
    with _temp_dir() as tmp:
        tmp_path = Path(tmp)
        retriever, chunk_store, vector_store = _build_retriever(tmp_path)

        source = "sample_data/guidelines/who_hypertension_guideline_2021.pdf"
        chunk_store.set_current(source, is_current=False)
        set_current(vector_store, source, is_current=False)
        retriever = HybridRetriever(vector_store, chunk_store)  # rebuild — BM25 index is an in-memory snapshot

        current_results = retriever.retrieve("target blood pressure", top_k=5, only_current=True)
        assert current_results == [], "superseded chunks should be excluded when only_current=True"

        all_results = retriever.retrieve("target blood pressure", top_k=5, only_current=False)
        assert len(all_results) > 0, "the same chunks should still be reachable with only_current=False"
        print("OK: only_current filter correctly excludes/includes superseded chunks")


if __name__ == "__main__":
    test_hybrid_retrieve_real_corpus()
    test_only_current_filter_excludes_superseded()
    print("All test_retrieval.py checks passed.")
