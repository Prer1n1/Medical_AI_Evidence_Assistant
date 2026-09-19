"""Free/offline — use_llm_classifier=False and use_image_captioning=False
keep this deterministic and free (keyword-fallback category/evidence
classification, no captioning calls). Uses the real sample_data PDFs but
a FakeEmbeddings double so semantic chunking doesn't need the real model."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ingestion.pipeline import ingest_directory
from ingestion.tracker import IngestionTracker
from test_chunking import FakeEmbeddings

GUIDELINES_DIR = Path("sample_data/guidelines")


def test_ingest_directory_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        tracker = IngestionTracker(db_path=Path(tmp) / "manifest.db")
        result = ingest_directory(
            GUIDELINES_DIR,
            tracker=tracker,
            embeddings=FakeEmbeddings(),
            use_llm_classifier=False,
            use_image_captioning=False,
            doc_subtype="clinical_guideline",
        )
        assert len(result.ingested_files) == 1
        assert len(result.chunks) > 20, f"expected many chunks from a 47-page guideline, got {len(result.chunks)}"
        assert all(c.metadata.extra.get("doc_subtype") == "clinical_guideline" for c in result.chunks)
        # derive_guideline_name() deliberately strips the trailing year
        # (see ingestion/guideline_versioning.py) so different years of
        # the same guideline collapse to one grouping key.
        assert all(c.metadata.extra.get("guideline_name") == "who hypertension guideline" for c in result.chunks)
        assert all(c.metadata.extra.get("effective_date") == "2021" for c in result.chunks)
        assert all(c.metadata.extra.get("evidence_level") for c in result.chunks)
        print(f"OK: first ingest produced {len(result.chunks)} chunks, all correctly tagged")

        # ingest_directory() deliberately does NOT call mark_ingested()
        # itself — that's the caller's job, done only after chunks are
        # actually persisted (see ingestion/pipeline.py's docstring on the
        # real bug this ordering avoids). This test has no real storage to
        # persist to, but still must mark the tracker itself, exactly like
        # main.py/api.py do, or the second ingest below will incorrectly
        # see every file as still "new".
        for file_path in result.pending_mark:
            tracker.mark_ingested(file_path)

        # Second run: nothing changed -> should skip entirely.
        result2 = ingest_directory(
            GUIDELINES_DIR, tracker=tracker, embeddings=FakeEmbeddings(),
            use_llm_classifier=False, use_image_captioning=False,
        )
        assert len(result2.ingested_files) == 0
        assert len(result2.skipped_unchanged) == 1
        assert result2.chunks == []
        print("OK: second ingest (nothing changed) skips the file entirely")


def test_uncaptioned_images_keep_placeholder_when_captioning_disabled():
    with tempfile.TemporaryDirectory() as tmp:
        tracker = IngestionTracker(db_path=Path(tmp) / "manifest.db")
        result = ingest_directory(
            GUIDELINES_DIR, tracker=tracker, embeddings=FakeEmbeddings(),
            use_llm_classifier=False, use_image_captioning=False,
        )
        image_chunks = [c for c in result.chunks if c.metadata.extra.get("is_image")]
        assert len(image_chunks) >= 1
        assert all("not yet captioned" in c.content for c in image_chunks)
        print(f"OK: {len(image_chunks)} image chunk(s) kept placeholder text with captioning disabled")


if __name__ == "__main__":
    test_ingest_directory_end_to_end()
    test_uncaptioned_images_keep_placeholder_when_captioning_disabled()
    print("All test_pipeline.py checks passed.")
