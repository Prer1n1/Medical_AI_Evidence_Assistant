"""Chunk store — the source-of-truth record for every chunk's text and
metadata (SQLite). Ported from enterprise-agentic-rag with new medical
columns: evidence_level (what retrieval/evidence_ranker.py and
retrieval/confidence.py weight by), and the guideline-versioning triplet
(guideline_name, effective_date, is_current) that has no equivalent in
the enterprise version at all — see docs/design-decisions.md
("Guideline Versioning").
"""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

from ingestion.chunking import Chunk

DEFAULT_DB_PATH = Path(__file__).parent / "chunk_store.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    doc_id TEXT,
    doc_type TEXT,
    doc_subtype TEXT,
    section TEXT,
    page_number INTEGER,
    category TEXT,
    evidence_level TEXT,
    guideline_name TEXT,
    effective_date TEXT,
    is_current INTEGER NOT NULL DEFAULT 1,
    pmid TEXT,
    journal TEXT,
    doi TEXT,
    chunk_index INTEGER,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
CREATE INDEX IF NOT EXISTS idx_chunks_guideline_name ON chunks(guideline_name);
"""


def compute_chunk_id(chunk: Chunk) -> str:
    """Deterministic, not random — re-ingesting the same source/section/
    index combo produces the same ID, so an UPSERT naturally replaces the
    old record. The vector store uses the SAME function, keeping both
    stores in sync."""
    raw = f"{chunk.metadata.source}::{chunk.metadata.section}::{chunk.chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class ChunkStore:
    def __init__(self, db_path: Union[str, Path] = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.executescript(_SCHEMA)
        return conn

    def save_chunks(self, chunks: List[Chunk]) -> List[str]:
        """Upserts chunks, returns their chunk_ids. New chunks always
        insert as is_current=1 — an older version of the SAME guideline
        is marked is_current=0 separately, by
        ingestion.guideline_versioning.apply_versioning(), not here (this
        method has no cross-document knowledge to make that call itself)."""
        now = datetime.now(timezone.utc).isoformat()
        chunk_ids = []
        with closing(self._connect()) as conn:
            for chunk in chunks:
                chunk_id = compute_chunk_id(chunk)
                chunk_ids.append(chunk_id)
                m = chunk.metadata
                conn.execute(
                    """INSERT INTO chunks (chunk_id, source, doc_id, doc_type, doc_subtype, section,
                                            page_number, category, evidence_level, guideline_name,
                                            effective_date, is_current, pmid, journal, doi,
                                            chunk_index, content, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(chunk_id) DO UPDATE SET
                           content = excluded.content,
                           is_current = 1,
                           created_at = excluded.created_at""",
                    (
                        chunk_id, m.source, m.extra.get("doc_id"), m.doc_type, m.extra.get("doc_subtype"),
                        m.section, m.page_number, m.extra.get("category"), m.extra.get("evidence_level"),
                        m.extra.get("guideline_name"), m.extra.get("effective_date"),
                        m.extra.get("pmid"), m.extra.get("journal"), m.extra.get("doi"),
                        chunk.chunk_index, chunk.content, now,
                    ),
                )
            conn.commit()
        return chunk_ids

    def delete_by_source(self, source: str) -> int:
        with closing(self._connect()) as conn:
            cur = conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
            conn.commit()
            return cur.rowcount

    def set_current(self, source: str, is_current: bool) -> int:
        """Flips is_current for every chunk of one source — used by
        ingestion.guideline_versioning.apply_versioning() to mark a
        superseded guideline's chunks without deleting them (deleting
        would lose the ability to ever answer "what did the OLD guideline
        say" — traceability the versioning feature exists to preserve)."""
        with closing(self._connect()) as conn:
            cur = conn.execute(
                "UPDATE chunks SET is_current = ? WHERE source = ?", (1 if is_current else 0, source)
            )
            conn.commit()
            return cur.rowcount

    def get_all(self) -> List[dict]:
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM chunks").fetchall()
            return [dict(r) for r in rows]

    def get_by_source(self, source: str) -> List[dict]:
        with closing(self._connect()) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM chunks WHERE source = ? ORDER BY chunk_index", (source,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_sources_by_guideline_name(self, guideline_name: str, exclude_source: Optional[str] = None) -> List[str]:
        """Every DISTINCT source currently tagged with this guideline_name
        — how ingestion.guideline_versioning finds prior versions of a
        guideline being re-ingested under a new source path/filename."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT DISTINCT source FROM chunks WHERE guideline_name = ? AND source != ?",
                (guideline_name, exclude_source or ""),
            ).fetchall()
            return [row[0] for row in rows]
