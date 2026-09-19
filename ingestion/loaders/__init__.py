"""Format dispatch — maps a file's extension to its loader, so callers
never need an if/elif ladder of their own."""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

from ..schema import Document
from .base import BaseLoader
from .docx_loader import DOCXLoader
from .html_loader import HTMLLoader
from .pdf_loader import PDFLoader

_LOADERS: dict[str, BaseLoader] = {
    ".pdf": PDFLoader(),
    ".docx": DOCXLoader(),
    ".html": HTMLLoader(),
    ".htm": HTMLLoader(),
}


def load_document(file_path: Union[str, Path]) -> List[Document]:
    file_path = Path(file_path)
    loader = _LOADERS.get(file_path.suffix.lower())
    if loader is None:
        raise ValueError(f"No loader registered for extension '{file_path.suffix}': {file_path}")
    return loader.load(file_path)
