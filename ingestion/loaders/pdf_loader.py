"""PDF loader — pymupdf (fitz), not pypdf.

Rebuilt from the enterprise-agentic-rag version specifically for medical
PDFs, which are far more table- and figure-heavy than typical office
documents (dosage tables, treatment algorithms, diagnostic flowcharts).
pypdf only extracts flat page text; pymupdf additionally exposes
`page.find_tables()` (structured table detection) and `page.get_images()`
(embedded raster images) — both needed for the "multimodal: PDFs, tables,
image-rich clinical references" requirement.

Produces THREE kinds of Document per PDF, same "keep tables/images as
their own Document, never flattened into prose" discipline the enterprise
loaders used for DOCX/HTML tables:
  - prose: one Document per page (page text with any detected table
    regions removed, so table content isn't duplicated as prose)
  - tables: one Document per detected table, pipe-joined rows, `is_table`
  - images: one Document per extracted figure, `is_image` + a path to the
    saved image file. Content starts as a placeholder — a real caption is
    filled in later by ingestion/image_captioner.py (optional, degrades
    gracefully without an API key), which is what makes the figure
    text-searchable at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import pymupdf

from ..schema import Document, DocumentMetadata, canonical_source
from .base import BaseLoader

# Project-root-relative storage location for extracted figures — gitignored
# (regenerable from the source PDFs), same treatment as chroma_db/chunk_store.db.
IMAGE_DIR = Path(__file__).resolve().parent.parent.parent / "storage" / "images"

# Skip small embedded images (logos, bullet icons, letterhead graphics) —
# a real clinical figure (dosage chart, diagnostic algorithm, scan image)
# is virtually never this small. Heuristic, not a hard science; documented
# rather than silently guessed at.
MIN_IMAGE_DIMENSION = 150

# See the false-positive note where this is used, in _load_from()'s table loop.
MIN_TABLE_CONTENT_CHARS = 150

UNCAPTIONED_PLACEHOLDER = "[Image extracted from page {page}, not yet captioned]"


class PDFLoader(BaseLoader):
    doc_type = "pdf"

    def load(self, file_path: Union[str, Path]) -> List[Document]:
        file_path = Path(file_path)
        source = canonical_source(file_path)
        doc = pymupdf.open(str(file_path))
        try:
            return self._load_from(doc, file_path, source)
        finally:
            doc.close()

    def _load_from(self, doc: "pymupdf.Document", file_path: Path, source: str) -> List[Document]:
        info = doc.metadata or {}
        title = (info.get("title") or "").strip() or file_path.stem
        author = (info.get("author") or "").strip() or None
        created_date = (info.get("creationDate") or "").strip() or None

        documents: List[Document] = []
        image_out_dir = IMAGE_DIR / file_path.stem
        image_counter = 0

        for page_index, page in enumerate(doc):
            page_number = page_index + 1

            # Tables first — find_tables() gives us the table's bbox, which
            # we use to exclude that region from the plain page-text
            # extraction below so table content isn't duplicated as prose.
            table_finder = page.find_tables()
            for table_index, table in enumerate(table_finder.tables, start=1):
                rows = table.extract()
                rows_text = [
                    " | ".join((cell or "").strip() for cell in row)
                    for row in rows
                ]
                table_content = "\n".join(r for r in rows_text if r.strip())
                # find_tables() has a real, verified false-positive mode:
                # a cover/title page's decorative text layout gets detected
                # as a "table" with mostly-empty cells (e.g. one real cell
                # of title text plus four empty ones). MIN_TABLE_CONTENT_CHARS
                # filters those out while keeping legitimate sparse tables
                # (a real PICO/evidence table with some intentionally blank
                # cells still clears this bar easily — verified against the
                # WHO hypertension guideline's own real tables, all 400+
                # chars, vs. its cover page's false positive at 84 chars).
                if len(table_content.strip()) < MIN_TABLE_CONTENT_CHARS:
                    continue
                documents.append(
                    Document(
                        content=table_content,
                        metadata=DocumentMetadata(
                            source=source,
                            doc_type=self.doc_type,
                            title=title,
                            author=author,
                            created_date=created_date,
                            section=f"Table {table_index} (page {page_number})",
                            page_number=page_number,
                            extra={"is_table": True},
                        ),
                    )
                )

            # Normalize to pymupdf.Rect regardless of whether .bbox comes
            # back as a Rect or a plain (x0, y0, x1, y1) tuple across
            # pymupdf versions — Rect(...) accepts either.
            table_bboxes = [pymupdf.Rect(t.bbox) for t in table_finder.tables]
            text = page.get_text("text", clip=None).strip()
            if table_bboxes:
                # Re-extract excluding table regions so their rows aren't
                # ALSO captured as unstructured prose (double-counting the
                # same content in two different Document shapes).
                text = _text_excluding_tables(page, table_bboxes).strip()
            if text:
                documents.append(
                    Document(
                        content=text,
                        metadata=DocumentMetadata(
                            source=source,
                            doc_type=self.doc_type,
                            title=title,
                            author=author,
                            created_date=created_date,
                            page_number=page_number,
                        ),
                    )
                )

            for img_index, img in enumerate(page.get_images(full=True), start=1):
                xref = img[0]
                width, height = img[2], img[3]
                if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
                    continue
                try:
                    base_image = doc.extract_image(xref)
                except Exception:
                    continue  # a malformed/unsupported embedded image — skip, don't abort the whole page
                image_counter += 1
                image_out_dir.mkdir(parents=True, exist_ok=True)
                image_path = image_out_dir / f"p{page_number}_img{img_index}.{base_image['ext']}"
                image_path.write_bytes(base_image["image"])
                documents.append(
                    Document(
                        content=UNCAPTIONED_PLACEHOLDER.format(page=page_number),
                        metadata=DocumentMetadata(
                            source=source,
                            doc_type=self.doc_type,
                            title=title,
                            author=author,
                            created_date=created_date,
                            section=f"Figure {image_counter} (page {page_number})",
                            page_number=page_number,
                            extra={"is_image": True, "image_path": str(image_path)},
                        ),
                    )
                )

        return documents


def _text_excluding_tables(page: "pymupdf.Page", table_bboxes: list) -> str:
    """Reassembles page text from text blocks whose bounding box doesn't
    fall inside a detected table region — avoids table rows appearing
    twice (once as a structured table Document, once as raw prose)."""
    blocks = page.get_text("blocks")
    kept = []
    for block in blocks:
        bx0, by0, bx1, by1 = block[:4]
        block_text = block[4]
        inside_table = any(
            bx0 >= tb.x0 - 1 and by0 >= tb.y0 - 1 and bx1 <= tb.x1 + 1 and by1 <= tb.y1 + 1
            for tb in table_bboxes
        )
        if not inside_table:
            kept.append(block_text)
    return "\n".join(kept)
