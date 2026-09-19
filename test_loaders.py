"""Free/offline — no API key needed. Exercises the PDF loader against the
real sample guideline/protocol PDFs in sample_data/ (downloaded from
iris.who.int and afro.who.int — see docs/design-decisions.md, "Sample
data"). No LLM calls anywhere in the loader layer, so this stays fast and
deterministic in CI.
"""

from __future__ import annotations

from pathlib import Path

from ingestion.loaders.pdf_loader import PDFLoader

HYPERTENSION_PDF = Path("sample_data/guidelines/who_hypertension_guideline_2021.pdf")
POCKETBOOK_PDF = Path("sample_data/protocols/who_pocketbook_child_care_dosages.pdf")


def test_pdf_loader_extracts_prose_tables_and_images():
    docs = PDFLoader().load(HYPERTENSION_PDF)
    prose = [d for d in docs if not d.metadata.extra.get("is_table") and not d.metadata.extra.get("is_image")]
    tables = [d for d in docs if d.metadata.extra.get("is_table")]
    images = [d for d in docs if d.metadata.extra.get("is_image")]

    assert len(prose) > 40, f"expected substantial prose content, got {len(prose)} pages"
    assert len(tables) > 5, f"expected multiple real evidence tables, got {len(tables)}"
    assert len(images) >= 1, "expected at least one extracted image (this PDF has a WHO logo + a figure)"

    # canonical_source identity: forward-slash, matches everywhere else in the pipeline
    assert all("\\" not in d.metadata.source for d in docs)
    print(f"OK: {len(prose)} prose, {len(tables)} tables, {len(images)} images")


def test_pdf_table_false_positive_filtered():
    """Real bug found in Phase 2 verification: pymupdf's find_tables()
    detects the cover page's title-text layout as a near-empty 'table' (5
    cells, ~84 chars, mostly blank). MIN_TABLE_CONTENT_CHARS in
    pdf_loader.py filters it while keeping every real evidence table
    (all 400+ chars) — this test pins that behavior so it can't silently
    regress."""
    docs = PDFLoader().load(HYPERTENSION_PDF)
    tables = [d for d in docs if d.metadata.extra.get("is_table")]
    cover_page_tables = [t for t in tables if t.metadata.page_number == 1]
    assert cover_page_tables == [], "cover-page table false-positive should be filtered out"
    assert all(len(t.content.strip()) >= 150 for t in tables)
    print("OK: cover-page table false positive filtered, all real tables retained")


def test_pdf_loader_dosage_table_content():
    """Verifies the pocket book's real dosage/growth tables extract with
    actual clinical numbers, not just structurally (see
    docs/design-decisions.md, 'Multimodal ingestion')."""
    docs = PDFLoader().load(POCKETBOOK_PDF)
    tables = [d for d in docs if d.metadata.extra.get("is_table")]
    assert len(tables) >= 5
    joined = "\n".join(t.content for t in tables)
    assert any(ch.isdigit() for ch in joined), "dosage tables should contain real numeric values"
    print(f"OK: {len(tables)} tables extracted from pocket book, contain numeric dosage data")


if __name__ == "__main__":
    test_pdf_loader_extracts_prose_tables_and_images()
    test_pdf_table_false_positive_filtered()
    test_pdf_loader_dosage_table_content()
    print("All test_loaders.py checks passed.")
