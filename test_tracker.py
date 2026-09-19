"""Free/offline. Content-hash incremental tracker — ported logic from
enterprise-agentic-rag, verified fresh here since the module itself was
re-typed (not literally copy-pasted) for this project."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ingestion.tracker import IngestionTracker


def test_new_changed_unchanged_deleted_cycle():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        db_path = tmp_path / "manifest.db"
        file_a = tmp_path / "a.pdf"
        file_b = tmp_path / "b.pdf"
        file_a.write_text("original content")
        file_b.write_text("other file, unrelated")

        tracker = IngestionTracker(db_path=db_path)

        decision = tracker.check(file_a)
        assert decision.reason == "new"
        tracker.mark_ingested(file_a)
        tracker.mark_ingested(file_b)

        decision = tracker.check(file_a)
        assert decision.reason == "unchanged"

        file_a.write_text("changed content")
        decision = tracker.check(file_a)
        assert decision.reason == "changed"
        tracker.mark_ingested(file_a)

        # b unaffected by a's edit
        assert tracker.check(file_b).reason == "unchanged"

        # file_a removed from the corpus listing -> reported as deleted,
        # scoped to this root so file_b (still present) isn't affected
        deleted = tracker.find_deleted([file_b], root=tmp_path)
        assert str(file_a.as_posix()) in deleted
        assert str(file_b.as_posix()) not in deleted
        print("OK: new -> unchanged -> changed -> deleted cycle")


def test_find_deleted_scoped_to_root():
    """Regression pin for the multi-root bug the enterprise project found
    (ingesting one corpus root wrongly marked every file in ANOTHER root
    as deleted) — same tracker code, same risk here since /ingest and
    /documents/upload can point at different directories."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        root_a = tmp_path / "root_a"
        root_b = tmp_path / "root_b"
        root_a.mkdir()
        root_b.mkdir()
        file_a = root_a / "x.pdf"
        file_b = root_b / "y.pdf"
        file_a.write_text("a")
        file_b.write_text("b")

        tracker = IngestionTracker(db_path=tmp_path / "manifest.db")
        tracker.mark_ingested(file_a)
        tracker.mark_ingested(file_b)

        # Ingesting root_a alone (root_b untouched) must not report
        # root_b's file as deleted.
        deleted = tracker.find_deleted([file_a], root=root_a)
        assert deleted == [], f"root_b's file wrongly reported deleted: {deleted}"
        print("OK: find_deleted scoped correctly, no cross-root false positives")


if __name__ == "__main__":
    test_new_changed_unchanged_deleted_cycle()
    test_find_deleted_scoped_to_root()
    print("All test_tracker.py checks passed.")
