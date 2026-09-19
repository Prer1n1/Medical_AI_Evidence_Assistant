# Medical AI Evidence Assistant

A medical evidence assistant that retrieves grounded, cited answers from research papers, hospital guidelines, and treatment protocols — built with LangGraph, FastAPI, and locally-run biomedical embeddings.

An LLM router decides which clinical categories are relevant to a question, queries them **in parallel** via a LangGraph workflow, then synthesizes one grounded answer with medical-format citations (PMID/journal or guideline name+version), an **evidence level** per source, and an overall **confidence score** — all explainable, not a single opaque relevance number.

Sibling project: [`enterprise-agentic-rag`](../enterprise-agentic-rag) — shares its proven storage/retrieval/agent/API architecture; this project's ingestion and evidence layer are rebuilt from scratch for medical documents. See [docs/design-decisions.md](docs/design-decisions.md) for the full reasoning log, including what's ported, what's new, and real bugs found while building it.

## Architecture

```
Sources: PDF/DOCX/HTML guidelines & protocols  +  PubMed (live E-utilities API)
        |
   [Ingestion]  pymupdf (text/tables/images) -> optional GPT-4o-mini image captioning
                -> medical category + evidence-level classification (LLM + keyword fallback)
                -> guideline-name/version extraction -> hierarchical/semantic chunking
                -> incremental tracker (files) / PMID tracker (PubMed)
        |
   [Storage]    Chroma (local biomedical embeddings: PubMedBERT-based sentence-transformer)
                +  SQLite (chunk text, evidence level, guideline version/is_current)
        |
   [Retrieval]  hybrid: dense (Chroma) + sparse (BM25) fused with RRF
                -> Cohere Rerank (optional) -> evidence-level re-weighting
                -> only_current filter (excludes superseded guideline versions)
        |
   [Agent]      LangGraph: plan (route to clinical categories) -> parallel retrieve
                -> synthesize (grounded, medically-cited, confidence-scored,
                   clinical-safety-framed — never prescriptive)
        |
   [Evaluation] RAGAS: faithfulness/relevancy/context precision
                + evidence-level distribution check
        |
   [API]        FastAPI: GET /health, POST /ingest, POST /documents/upload,
                POST /ingest/pubmed, POST /query
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env            # then fill in:
                                 #   OPENAI_API_KEY  (classification, routing, synthesis, optional image captions)
                                 #   API_KEY         (generate: python -c "import secrets; print(secrets.token_urlsafe(32))")
                                 #   NCBI_EMAIL      (identifies you to NCBI's E-utilities — required by their usage policy, not a secret)
                                 #   COHERE_API_KEY  (optional — reranking; falls back to plain RRF order without it)
```

Embeddings need no API key — `EMBEDDING_MODEL` (default `pritamdeka/S-PubMedBert-MS-MARCO`) downloads once (~430MB) and runs locally on CPU.

## Running it

```bash
# Ingest local guideline/protocol documents
python main.py ingest sample_data/guidelines --doc-subtype clinical_guideline
python main.py ingest sample_data/protocols --doc-subtype treatment_protocol

# Search + ingest PubMed records for a topic (incremental — re-running skips already-synced PMIDs)
python main.py ingest-pubmed "hypertension randomized controlled trial" --retmax 20

# Ask one question / interactive chat
python main.py ask "What target systolic blood pressure does WHO recommend for a patient with known cardiovascular disease?"
python main.py chat

# RAGAS evaluation over the curated eval set
python main.py evaluate
```

Or as an HTTP API:

```bash
uvicorn api.app:app --reload

curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/ingest -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"directory": "sample_data/guidelines", "doc_subtype": "clinical_guideline"}'
curl -X POST http://127.0.0.1:8000/ingest/pubmed -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"query": "hypertension randomized controlled trial", "retmax": 20}'
curl -X POST http://127.0.0.1:8000/query -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" -d '{"question": "What target blood pressure does WHO recommend for a patient with known cardiovascular disease?"}'
```

Swagger UI: `http://127.0.0.1:8000/docs`.

### Docker

```bash
docker compose up --build
```

## Project structure

```
ingestion/    loaders (pymupdf: text/tables/images) -> medical metadata extraction
              (category + evidence level) -> guideline versioning -> chunking
              connectors/pubmed.py — NCBI E-utilities, incremental by PMID
storage/      Chroma (local biomedical embeddings) + SQLite chunk store
retrieval/    BM25 + hybrid retriever (RRF) + Cohere rerank + evidence ranking + confidence scoring
agent/        LangGraph state, planner (clinical-category router), nodes, graph
evaluation/   curated eval set + RAGAS scoring + evidence-level distribution
api/          FastAPI app + request/response schemas
main.py       CLI: ingest / ingest-pubmed / ask / chat / evaluate
docs/         design-decisions.md — the full reasoning log
```

## Testing

Standalone `test_*.py` scripts, runnable individually, free/no-secret vs. live (billed) split — see `.github/workflows/tests.yml`.

```bash
# free / no secret needed
python test_loaders.py
python test_metadata.py
python test_chunking.py
python test_tracker.py
python test_pipeline.py
python test_storage.py
python test_guideline_versioning.py
python test_evidence_ranker.py
python test_confidence.py
python test_reranker.py
python test_image_captioner.py
python test_pubmed.py          # real network calls to NCBI, no secret required

# live — real OpenAI calls + real embedding model, needs OPENAI_API_KEY in .env
python test_retrieval.py
python test_agent.py
```

## Status

Built and verified end-to-end against real data: real WHO guideline/protocol PDFs (table + image extraction), live PubMed fetches, live LLM category/evidence-level classification, live Cohere reranking, local biomedical embeddings, guideline versioning, evidence ranking, confidence scoring, and a full LangGraph agent run producing grounded, medically-cited, confidence-scored, clinically-safety-framed answers.

This is a tested prototype demonstrating the architecture end-to-end — not a hardened clinical deployment. See [docs/design-decisions.md](docs/design-decisions.md) for what's in scope, what's deliberately out of scope (relative to the sibling enterprise project), and every real bug found while building it.
