"""PubMed connector — NCBI E-utilities (esearch + efetch), the piece with
no equivalent anywhere in the enterprise-agentic-rag project. Its job
mirrors the shape of that project's Google Drive connector: authenticate
(here, just an identifying email — no OAuth), list/search, fetch, hand the
result to the SAME downstream pipeline everything else goes through. The
difference is PubMed records aren't files on a filesystem, so this
connector builds Documents directly instead of writing to a local cache
directory for the file-based loaders to pick up.

Two E-utilities calls per sync:
  esearch — query -> list of PMIDs
  efetch  — PMIDs -> full XML records (title, abstract, journal, pub date,
            publication types, DOI, authors) in ONE batched request, not
            one request per PMID — both fewer round trips and friendlier
            to NCBI's rate limit (3 req/sec without a key, 10 req/sec with
            one; see config.NCBI_API_KEY).

Incremental sync, by PMID: PubMedSyncTracker (SQLite) records which PMIDs
have already been fetched for a given topic query, so re-running the same
search doesn't re-fetch/re-embed unchanged records — the PubMed analogue
of ingestion/tracker.py's content-hash tracking for local files.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

import requests
from lxml import etree

from config import NCBI_API_KEY, NCBI_EMAIL
from ingestion.metadata_extractor import PUBMED_PUBLICATION_TYPE_MAP
from ingestion.schema import Document, DocumentMetadata
from retry_utils import NCBITransientError, retry_ncbi_call

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_TOOL_NAME = "medical-ai-evidence-assistant"
_REQUEST_TIMEOUT = 30

DEFAULT_TRACKER_DB_PATH = Path(__file__).parent.parent.parent / "storage" / "pubmed_synced.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pubmed_synced (
    pmid TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    fetched_at TEXT NOT NULL
)
"""


@dataclass
class PubMedRecord:
    pmid: str
    title: str
    abstract: str
    journal: Optional[str]
    pub_date: Optional[str]
    publication_types: List[str] = field(default_factory=list)
    doi: Optional[str] = None
    authors: List[str] = field(default_factory=list)


class PubMedSyncTracker:
    def __init__(self, db_path: Union[str, Path] = DEFAULT_TRACKER_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(_SCHEMA)
        return conn

    def already_synced(self, pmids: List[str]) -> set:
        if not pmids:
            return set()
        with closing(self._connect()) as conn:
            placeholders = ",".join("?" * len(pmids))
            rows = conn.execute(
                f"SELECT pmid FROM pubmed_synced WHERE pmid IN ({placeholders})", pmids
            ).fetchall()
            return {row[0] for row in rows}

    def mark_synced(self, pmids: List[str], query: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as conn:
            conn.executemany(
                """INSERT INTO pubmed_synced (pmid, query, fetched_at) VALUES (?, ?, ?)
                   ON CONFLICT(pmid) DO UPDATE SET query = excluded.query, fetched_at = excluded.fetched_at""",
                [(pmid, query, now) for pmid in pmids],
            )
            conn.commit()


def _base_params() -> dict:
    params = {"tool": _TOOL_NAME}
    if NCBI_EMAIL:
        params["email"] = NCBI_EMAIL
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    return params


@retry_ncbi_call
def _get(url: str, params: dict) -> requests.Response:
    try:
        response = requests.get(url, params=params, timeout=_REQUEST_TIMEOUT)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        raise  # already a type retry_ncbi_call retries on
    if response.status_code >= 500:
        raise NCBITransientError(f"NCBI returned {response.status_code} for {url}")
    response.raise_for_status()  # a 4xx here is a real, non-retryable error (bad query, etc.)
    return response


def search_pubmed(query: str, retmax: int = 20) -> List[str]:
    """Returns a list of PMIDs matching the query, best-relevance first
    (esearch's default sort)."""
    params = {**_base_params(), "db": "pubmed", "term": query, "retmax": retmax, "retmode": "json"}
    response = _get(f"{_EUTILS_BASE}/esearch.fcgi", params)
    return response.json().get("esearchresult", {}).get("idlist", [])


def _text(el, path: str) -> Optional[str]:
    found = el.find(path)
    return found.text.strip() if found is not None and found.text else None


def _parse_abstract(article_el) -> str:
    parts = article_el.findall("Abstract/AbstractText")
    pieces = []
    for part in parts:
        label = part.get("Label")
        text = "".join(part.itertext()).strip()
        if not text:
            continue
        pieces.append(f"{label}: {text}" if label else text)
    return "\n\n".join(pieces)


def _parse_pub_date(article_el) -> Optional[str]:
    pub_date_el = article_el.find("Journal/JournalIssue/PubDate")
    if pub_date_el is None:
        return None
    medline_date = _text(pub_date_el, "MedlineDate")
    if medline_date:
        return medline_date
    year = _text(pub_date_el, "Year")
    month = _text(pub_date_el, "Month")
    day = _text(pub_date_el, "Day")
    return " ".join(p for p in (year, month, day) if p) or None


def _parse_authors(article_el) -> List[str]:
    authors = []
    for author_el in article_el.findall("AuthorList/Author"):
        last = _text(author_el, "LastName")
        initials = _text(author_el, "Initials")
        if last:
            authors.append(f"{last} {initials}" if initials else last)
    return authors


def _parse_doi(pubmed_article_el) -> Optional[str]:
    for id_el in pubmed_article_el.findall("PubmedData/ArticleIdList/ArticleId"):
        if id_el.get("IdType") == "doi" and id_el.text:
            return id_el.text.strip()
    return None


def fetch_pubmed_records(pmids: List[str]) -> List[PubMedRecord]:
    """One batched efetch call for all PMIDs, not one call per ID."""
    if not pmids:
        return []
    params = {
        **_base_params(),
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    response = _get(f"{_EUTILS_BASE}/efetch.fcgi", params)
    root = etree.fromstring(response.content)

    records = []
    for pubmed_article in root.findall("PubmedArticle"):
        citation = pubmed_article.find("MedlineCitation")
        if citation is None:
            continue
        article = citation.find("Article")
        pmid = _text(citation, "PMID")
        title = _text(article, "ArticleTitle") or "(no title)"
        abstract = _parse_abstract(article) if article is not None else ""
        journal = _text(article, "Journal/Title") if article is not None else None
        pub_types = [
            pt.text.strip() for pt in article.findall("PublicationTypeList/PublicationType")
            if pt.text
        ] if article is not None else []

        records.append(
            PubMedRecord(
                pmid=pmid,
                title=title,
                abstract=abstract,
                journal=journal,
                pub_date=_parse_pub_date(article) if article is not None else None,
                publication_types=pub_types,
                doi=_parse_doi(pubmed_article),
                authors=_parse_authors(article) if article is not None else [],
            )
        )
    return records


def _evidence_level_hint(publication_types: List[str]) -> Optional[str]:
    """PubMed's PublicationType list is ground-truth structured metadata —
    trusted outright by metadata_extractor.classify_evidence_level's hint
    parameter, never overridden by re-classifying the abstract text. First
    matching type wins (a record can carry several types; e.g. both
    "Journal Article" and "Randomized Controlled Trial" — the latter is
    what actually signals evidence level, so PUBMED_PUBLICATION_TYPE_MAP
    only has entries for the types that DO signal it)."""
    for pt in publication_types:
        mapped = PUBMED_PUBLICATION_TYPE_MAP.get(pt.strip().lower())
        if mapped:
            return mapped
    return None


def record_to_document(record: PubMedRecord) -> Document:
    content = f"{record.title}\n\n{record.abstract}" if record.abstract else record.title
    author_str = ", ".join(record.authors[:3]) + (" et al." if len(record.authors) > 3 else "")
    return Document(
        content=content,
        metadata=DocumentMetadata(
            source=f"pubmed:{record.pmid}",
            doc_type="pubmed",
            title=record.title,
            author=author_str or None,
            created_date=record.pub_date,
            extra={
                "doc_subtype": "research_paper",
                "pmid": record.pmid,
                "journal": record.journal,
                "doi": record.doi,
                "publication_types": record.publication_types,
                "evidence_level_hint": _evidence_level_hint(record.publication_types),
            },
        ),
    )


def sync_topic(
    query: str,
    retmax: int = 20,
    tracker: Optional[PubMedSyncTracker] = None,
    force: bool = False,
) -> List[Document]:
    """Searches PubMed for `query`, fetches full records for any PMIDs not
    already synced (unless force=True), marks them synced, and returns
    ready-to-enrich Documents — the PubMed equivalent of a file-based
    loader's output, meant to be handed to
    ingestion.metadata_extractor.enrich_all() then
    ingestion.chunking.chunk_documents() exactly like locally-loaded
    Documents are."""
    tracker = tracker or PubMedSyncTracker()
    pmids = search_pubmed(query, retmax=retmax)
    new_pmids = pmids if force else [p for p in pmids if p not in tracker.already_synced(pmids)]
    if not new_pmids:
        return []

    records = fetch_pubmed_records(new_pmids)
    tracker.mark_synced([r.pmid for r in records], query)
    return [record_to_document(r) for r in records]
