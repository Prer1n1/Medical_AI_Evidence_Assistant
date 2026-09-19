"""Live network calls to NCBI E-utilities — no API key/secret required
(NCBI_API_KEY is optional, only raises the rate limit), so this runs
unconditionally in CI's free-tests job, same bucket as the offline tests,
just needing network access rather than a billed API key. See
docs/design-decisions.md ("PubMed connector") for why this distinction
(no secret vs. no network) is worth drawing explicitly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import config  # noqa: F401 — loads .env (NCBI_EMAIL) into os.environ
from ingestion.connectors.pubmed import (
    PubMedSyncTracker,
    fetch_pubmed_records,
    record_to_document,
    search_pubmed,
    sync_topic,
)


def test_search_and_fetch_real_records():
    pmids = search_pubmed("hypertension randomized controlled trial", retmax=3)
    assert len(pmids) > 0, "expected at least one real PMID"

    records = fetch_pubmed_records(pmids)
    assert len(records) == len(pmids)
    for record in records:
        assert record.pmid
        assert record.title
        print(f"  PMID {record.pmid}: {record.title[:70]}")
    print(f"OK: fetched {len(records)} real PubMed records")


def test_record_to_document_shape():
    pmids = search_pubmed("hypertension treatment", retmax=1)
    records = fetch_pubmed_records(pmids)
    doc = record_to_document(records[0])
    assert doc.metadata.source == f"pubmed:{records[0].pmid}"
    assert doc.metadata.doc_type == "pubmed"
    assert doc.metadata.extra["doc_subtype"] == "research_paper"
    assert doc.metadata.extra["pmid"] == records[0].pmid
    print("OK: record_to_document produces a correctly-shaped Document")


def test_incremental_sync_skips_already_synced():
    with tempfile.TemporaryDirectory() as tmp:
        tracker = PubMedSyncTracker(db_path=Path(tmp) / "pubmed.db")
        query = "aspirin cardiovascular prevention"

        first = sync_topic(query, retmax=3, tracker=tracker)
        assert len(first) > 0, "expected real records on first sync"

        second = sync_topic(query, retmax=3, tracker=tracker)
        assert second == [], "re-syncing the identical query should fetch nothing new"
        print(f"OK: incremental sync fetched {len(first)} new records, then correctly skipped all on re-sync")


if __name__ == "__main__":
    test_search_and_fetch_real_records()
    test_record_to_document_shape()
    test_incremental_sync_skips_already_synced()
    print("All test_pubmed.py checks passed.")
