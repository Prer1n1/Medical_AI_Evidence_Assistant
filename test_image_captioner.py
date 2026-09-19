"""Part 1 (free/offline): no OPENAI_API_KEY -> caption_image() returns
None gracefully. Part 2 (gated on OPENAI_API_KEY, real GPT-4o-mini vision
call): captions one of the real images extracted from the WHO hypertension
guideline PDF (see test_loaders.py) and checks the result is a real,
non-trivial description."""

from __future__ import annotations

import os

import config  # noqa: F401 — loads .env
from ingestion.image_captioner import caption_image
from ingestion.loaders.pdf_loader import PDFLoader

HYPERTENSION_PDF = "sample_data/guidelines/who_hypertension_guideline_2021.pdf"


def test_no_api_key_returns_none(monkeypatch=None):
    import ingestion.image_captioner as captioner_module

    original_key = captioner_module.OPENAI_API_KEY
    captioner_module.OPENAI_API_KEY = None
    try:
        result = caption_image("nonexistent.png")
    finally:
        captioner_module.OPENAI_API_KEY = original_key
    assert result is None
    print("OK: no OPENAI_API_KEY -> caption_image() returns None")


def test_missing_file_returns_none():
    if not os.getenv("OPENAI_API_KEY"):
        print("SKIPPED (no OPENAI_API_KEY): missing-file path only reachable past the key check")
        return
    result = caption_image("this_file_does_not_exist.png")
    assert result is None
    print("OK: missing image file -> caption_image() returns None")


def test_live_captioning_real_extracted_image():
    if not os.getenv("OPENAI_API_KEY"):
        print("SKIPPED (no OPENAI_API_KEY): live image captioning")
        return
    docs = PDFLoader().load(HYPERTENSION_PDF)
    images = [d for d in docs if d.metadata.extra.get("is_image")]
    assert images, "expected at least one extracted image to caption"

    caption = caption_image(images[0].metadata.extra["image_path"])
    assert caption is not None
    assert len(caption) > 20, f"expected a real descriptive caption, got: {caption!r}"
    print(f"OK: live caption for {images[0].metadata.extra['image_path']}: {caption[:100]}...")


if __name__ == "__main__":
    test_no_api_key_returns_none()
    test_missing_file_returns_none()
    test_live_captioning_real_extracted_image()
    print("All test_image_captioner.py checks passed.")
