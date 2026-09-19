"""HTML loader — one Document per heading-delimited section, plus one
Document per <table>. Ported verbatim from enterprise-agentic-rag: some
public health guidance (WHO/CDC web pages saved locally) is HTML rather
than PDF, and this loader's structure is format-generic.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

from bs4 import BeautifulSoup

from ..schema import Document, DocumentMetadata, canonical_source
from .base import BaseLoader

_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_TEXT_TAGS = {"p", "li"}


class HTMLLoader(BaseLoader):
    doc_type = "html"

    def load(self, file_path: Union[str, Path]) -> List[Document]:
        file_path = Path(file_path)
        html = file_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "lxml")

        for tag in soup(["script", "style"]):
            tag.decompose()

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else file_path.stem

        documents: List[Document] = []
        current_section = "Document Start"
        buffer: List[str] = []

        def flush() -> None:
            text = "\n".join(buffer).strip()
            if text:
                documents.append(
                    Document(
                        content=text,
                        metadata=DocumentMetadata(
                            source=canonical_source(file_path),
                            doc_type=self.doc_type,
                            title=title,
                            section=current_section,
                        ),
                    )
                )

        body = soup.body or soup
        for el in body.find_all(list(_HEADING_TAGS | _TEXT_TAGS | {"table"})):
            if el.find_parent("table"):
                continue

            if el.name in _HEADING_TAGS:
                flush()
                buffer = []
                current_section = el.get_text(strip=True) or current_section

            elif el.name == "table":
                flush()
                buffer = []

                rows_text = []
                for row in el.find_all("tr"):
                    cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
                    if cells:
                        rows_text.append(" | ".join(cells))
                table_content = "\n".join(rows_text)
                if table_content:
                    documents.append(
                        Document(
                            content=table_content,
                            metadata=DocumentMetadata(
                                source=canonical_source(file_path),
                                doc_type=self.doc_type,
                                title=title,
                                section=current_section,
                                extra={"is_table": True},
                            ),
                        )
                    )

            else:
                text = el.get_text(strip=True)
                if text:
                    buffer.append(text)

        flush()
        return documents
