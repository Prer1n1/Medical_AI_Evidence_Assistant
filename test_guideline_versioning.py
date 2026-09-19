"""Free/offline — no API key needed. Pure filename-heuristic functions,
plus the cross-document supersession logic against a real (temp-file)
ChunkStore. vector_store=None exercises apply_versioning's "no vector
store to update" path so this test doesn't need to load the embedding
model."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ingestion.chunking import Chunk
from ingestion.guideline_versioning import apply_versioning, derive_effective_date_hint, derive_guideline_name
from ingestion.schema import DocumentMetadata
from storage.chunk_store import ChunkStore


def test_derive_guideline_name_collapses_versions():
    assert derive_guideline_name("hypertension_guideline_v1") == derive_guideline_name("hypertension_guideline_v2")
    assert derive_guideline_name("who-hypertension-2021") == derive_guideline_name("who-hypertension-2023")
    assert derive_guideline_name("hypertension_guideline_v1") == "hypertension guideline"
    print("OK: derive_guideline_name collapses version/year suffixes")


def test_derive_guideline_name_distinguishes_unrelated_guidelines():
    assert derive_guideline_name("hypertension_guideline_v1") != derive_guideline_name("sepsis_protocol_v1")
    print("OK: unrelated guidelines stay distinct")


def test_derive_effective_date_hint_extracts_year():
    assert derive_effective_date_hint("who-hypertension-2021") == "2021"
    assert derive_effective_date_hint("no_year_here") is None
    print("OK: effective date hint extraction")


def _make_chunk(source: str, guideline_name: str) -> Chunk:
    return Chunk(
        content="some guideline content",
        metadata=DocumentMetadata(
            source=source, doc_type="pdf", section="Section 1",
            extra={"doc_subtype": "clinical_guideline", "guideline_name": guideline_name, "evidence_level": "Expert Opinion / Clinical Guideline"},
        ),
        chunk_index=0,
    )


def test_apply_versioning_marks_old_version_superseded():
    with tempfile.TemporaryDirectory() as tmp:
        chunk_store = ChunkStore(db_path=Path(tmp) / "chunks.db")

        old_chunk = _make_chunk("guidelines/hypertension_v1.pdf", "hypertension guideline")
        chunk_store.save_chunks([old_chunk])
        assert chunk_store.get_by_source("guidelines/hypertension_v1.pdf")[0]["is_current"] == 1

        new_chunk = _make_chunk("guidelines/hypertension_v2.pdf", "hypertension guideline")
        chunk_store.save_chunks([new_chunk])

        superseded = apply_versioning(chunk_store, vector_store=None, ingested_sources=["guidelines/hypertension_v2.pdf"])
        assert superseded == ["guidelines/hypertension_v1.pdf"]
        assert chunk_store.get_by_source("guidelines/hypertension_v1.pdf")[0]["is_current"] == 0
        assert chunk_store.get_by_source("guidelines/hypertension_v2.pdf")[0]["is_current"] == 1
        print("OK: re-ingesting a newer guideline version marks the old one is_current=0")


def test_apply_versioning_no_op_for_unrelated_guideline():
    with tempfile.TemporaryDirectory() as tmp:
        chunk_store = ChunkStore(db_path=Path(tmp) / "chunks.db")
        chunk_store.save_chunks([_make_chunk("guidelines/sepsis_v1.pdf", "sepsis protocol")])
        chunk_store.save_chunks([_make_chunk("guidelines/hypertension_v1.pdf", "hypertension guideline")])

        superseded = apply_versioning(chunk_store, vector_store=None, ingested_sources=["guidelines/hypertension_v1.pdf"])
        assert superseded == [], "unrelated guideline must not be marked superseded"
        assert chunk_store.get_by_source("guidelines/sepsis_v1.pdf")[0]["is_current"] == 1
        print("OK: unrelated guidelines are never marked superseded by each other")


if __name__ == "__main__":
    test_derive_guideline_name_collapses_versions()
    test_derive_guideline_name_distinguishes_unrelated_guidelines()
    test_derive_effective_date_hint_extracts_year()
    test_apply_versioning_marks_old_version_superseded()
    test_apply_versioning_no_op_for_unrelated_guideline()
    print("All test_guideline_versioning.py checks passed.")
