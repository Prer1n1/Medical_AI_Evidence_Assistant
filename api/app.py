"""FastAPI REST layer — HTTP wrapper around the same ingestion/agent
pipeline main.py's CLI drives. Structure ported from enterprise-agentic-rag
(sync handlers, lifespan-built AppState, structured logging, API-key auth)
minus access control/PII redaction/prompt-injection defense (dropped from
scope — see docs/design-decisions.md, "Scope"). New: POST /ingest/pubmed,
guideline-versioning applied after every directory ingest, confidence in
every /query response.

Run with: uvicorn api.app:app --reload
"""

from __future__ import annotations

import logging
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from agent.graph import build_agent_graph
from api.schemas import (
    Citation,
    Confidence,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    PubMedIngestRequest,
    PubMedIngestResponse,
    QueryRequest,
    QueryResponse,
)
from api.security import require_api_key
from config import API_KEY
from ingestion.connectors.pubmed import PubMedSyncTracker, sync_topic
from ingestion.guideline_versioning import apply_versioning
from ingestion.metadata_extractor import enrich_all
from ingestion.pipeline import SUPPORTED_EXTENSIONS, IngestionResult, ingest_directory
from ingestion.chunking import chunk_documents
from ingestion.tracker import IngestionTracker
from logging_config import configure_logging
from retrieval.hybrid_retriever import HybridRetriever
from storage.chunk_store import ChunkStore
from storage.vector_store import add_chunks, delete_by_source, get_embeddings, get_vector_store

configure_logging()
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")


class AppState:
    """Built once at startup, reused across requests — the BM25 index in
    particular loads every chunk into memory, wasteful to rebuild per
    request."""

    def __init__(self) -> None:
        self.embeddings = get_embeddings()
        self.chunk_store = ChunkStore()
        self.vector_store = get_vector_store(embeddings=self.embeddings)
        self.retriever = HybridRetriever(self.vector_store, self.chunk_store)
        self.graph = build_agent_graph(self.retriever)

    def rebuild_retriever(self) -> None:
        """Call after ingestion changes the corpus — the BM25 index is an
        in-memory snapshot, invisible to new/deleted/superseded chunks
        until this runs."""
        self.retriever = HybridRetriever(self.vector_store, self.chunk_store)
        self.graph = build_agent_graph(self.retriever)


_state_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_KEY:
        raise RuntimeError(
            "API_KEY not set — copy .env.example to .env and set one "
            '(generate with: python -c "import secrets; print(secrets.token_urlsafe(32))")'
        )
    logger.info("startup_begin")
    app.state.rag = AppState()
    logger.info("startup_complete", extra={"chunk_count": len(app.state.rag.chunk_store.get_all())})
    yield
    logger.info("shutdown")


app = FastAPI(title="Medical AI Evidence Assistant", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Deliberately NOT behind require_api_key — an orchestrator's
    liveness/readiness probe needs to reach this without a secret."""
    state: AppState = app.state.rag
    chunk_store_ok = True
    chunk_count = 0
    try:
        chunk_count = len(state.chunk_store.get_all())
    except Exception:
        chunk_store_ok = False

    vector_store_ok = True
    try:
        state.vector_store.similarity_search("health check", k=1)
    except Exception:
        vector_store_ok = False

    status = "ok" if (chunk_store_ok and vector_store_ok) else "degraded"
    return HealthResponse(
        status=status,
        chunk_store_reachable=chunk_store_ok,
        vector_store_reachable=vector_store_ok,
        chunk_count=chunk_count,
    )


def _ingest_and_persist(directory: Path, doc_subtype: str) -> tuple[IngestionResult, list[str]]:
    """Shared by /ingest and /documents/upload. Returns (result,
    superseded_sources) — superseded_sources comes from
    ingestion.guideline_versioning.apply_versioning(), run AFTER
    persistence, same "only act once storage actually reflects the new
    chunks" discipline as the mark_ingested ordering below."""
    logger.info("ingestion_begin", extra={"directory": str(directory), "doc_subtype": doc_subtype})
    tracker = IngestionTracker()
    superseded: list[str] = []
    with _state_lock:
        state: AppState = app.state.rag
        result = ingest_directory(
            directory, tracker=tracker, embeddings=state.embeddings, doc_subtype=doc_subtype
        )

        for source in result.ingested_files:
            state.chunk_store.delete_by_source(source)
            delete_by_source(state.vector_store, source)

        if result.chunks:
            state.chunk_store.save_chunks(result.chunks)
            add_chunks(state.vector_store, result.chunks)

        for deleted_source in result.deleted_files:
            state.chunk_store.delete_by_source(deleted_source)
            delete_by_source(state.vector_store, deleted_source)

        for file_path in result.pending_mark:
            tracker.mark_ingested(file_path)

        if result.ingested_files:
            superseded = apply_versioning(state.chunk_store, state.vector_store, result.ingested_files)

        if result.ingested_files or result.deleted_files or superseded:
            state.rebuild_retriever()

    logger.info(
        "ingestion_complete",
        extra={
            "directory": str(directory),
            "ingested_files": len(result.ingested_files),
            "skipped_unchanged": len(result.skipped_unchanged),
            "deleted_files": len(result.deleted_files),
            "chunks_stored": len(result.chunks),
            "superseded_guideline_versions": superseded,
        },
    )
    return result, superseded


def _to_ingest_response(result: IngestionResult, superseded: list[str]) -> IngestResponse:
    return IngestResponse(
        ingested_files=len(result.ingested_files),
        skipped_unchanged=len(result.skipped_unchanged),
        deleted_files=len(result.deleted_files),
        chunks_stored=len(result.chunks),
        superseded_guideline_versions=superseded,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest, _: None = Depends(require_api_key)) -> IngestResponse:
    directory = Path(request.directory)
    if not directory.exists():
        raise HTTPException(status_code=400, detail=f"Directory not found: {directory}")

    result, superseded = _ingest_and_persist(directory, request.doc_subtype)
    return _to_ingest_response(result, superseded)


@app.post("/documents/upload", response_model=IngestResponse)
def upload_document(
    file: UploadFile = File(...),
    doc_subtype: str = "clinical_guideline",
    _: None = Depends(require_api_key),
) -> IngestResponse:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # NOT "filename" — collides with a reserved attribute on Python's own
    # LogRecord (verified bug in the enterprise project this was ported
    # from; avoided here from the start).
    logger.info("document_uploaded", extra={"uploaded_filename": file.filename})

    result, superseded = _ingest_and_persist(UPLOAD_DIR, doc_subtype)
    return _to_ingest_response(result, superseded)


@app.post("/ingest/pubmed", response_model=PubMedIngestResponse)
def ingest_pubmed(request: PubMedIngestRequest, _: None = Depends(require_api_key)) -> PubMedIngestResponse:
    """Searches PubMed for `request.query`, fetches any not-yet-synced
    records, and runs them through the SAME enrich -> chunk -> persist
    path as file-based ingestion — just entering it after
    connectors/pubmed.py's Documents instead of after a loader's."""
    logger.info("pubmed_ingest_begin", extra={"query": request.query, "retmax": request.retmax})
    with _state_lock:
        state: AppState = app.state.rag
        documents = sync_topic(request.query, retmax=request.retmax, tracker=PubMedSyncTracker())
        chunks_stored = 0
        if documents:
            enriched = enrich_all(documents)
            chunks = chunk_documents(enriched, embeddings=state.embeddings)
            if chunks:
                state.chunk_store.save_chunks(chunks)
                add_chunks(state.vector_store, chunks)
                chunks_stored = len(chunks)
                state.rebuild_retriever()

    logger.info(
        "pubmed_ingest_complete",
        extra={"query": request.query, "records_fetched": len(documents), "chunks_stored": chunks_stored},
    )
    return PubMedIngestResponse(query=request.query, records_fetched=len(documents), chunks_stored=chunks_stored)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, _: None = Depends(require_api_key)) -> QueryResponse:
    with _state_lock:
        graph = app.state.rag.graph

    logger.info("query_received", extra={"question_preview": request.question[:80]})

    try:
        result = graph.invoke({"query": request.question})
    except Exception:
        logger.exception("query_failed")
        raise HTTPException(status_code=500, detail="Agent execution failed") from None

    logger.info(
        "query_answered",
        extra={
            "categories_queried": result["categories"],
            "citation_count": len(result["citations"]),
            "confidence": result["confidence"]["label"],
        },
    )
    return QueryResponse(
        answer=result["answer"],
        categories_queried=result["categories"],
        citations=[Citation(**c) for c in result["citations"]],
        confidence=Confidence(**result["confidence"]),
    )
