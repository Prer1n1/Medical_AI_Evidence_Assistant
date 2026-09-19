"""Central place every component reads config/secrets from. Loads a
local .env (gitignored) so real API keys never get committed."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Local biomedical embedding model (sentence-transformers, via
# langchain-huggingface) — runs on CPU, no API key, no per-call cost.
# PubMedBERT-based, fine-tuned for retrieval (MS MARCO). See
# storage/vector_store.py and docs/design-decisions.md ("Embeddings") for
# why this was chosen over OpenAI's general-purpose embeddings.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "pritamdeka/S-PubMedBert-MS-MARCO")

# API layer auth — a shared secret every request to api/app.py must present
# via the X-API-Key header (except /health). See api/security.py.
API_KEY = os.getenv("API_KEY")

# PubMed connector (ingestion/connectors/pubmed.py) — NCBI's E-utilities
# usage policy asks every caller to identify itself via an email param
# (not a secret). NCBI_API_KEY is optional and raises the rate limit from
# 3 req/sec to 10 req/sec; unset means the connector still works, just
# slower. Same "degrade, don't break" pattern as everywhere else.
NCBI_EMAIL = os.getenv("NCBI_EMAIL")
NCBI_API_KEY = os.getenv("NCBI_API_KEY")

# LangSmith tracing: setting these three env vars is the ENTIRE integration —
# LangChain/LangGraph auto-instrument every LLM call once they're present, no
# code changes needed anywhere else in the project. Get a free key at
# smith.langchain.com. If LANGSMITH_API_KEY is unset, tracing is simply off —
# nothing breaks, it degrades silently to "no tracing."
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "medical-ai-evidence-assistant")

if LANGSMITH_API_KEY:
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", LANGSMITH_API_KEY)
    os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT)

# Cohere Rerank — reranks retrieval candidates for precision (see
# retrieval/reranker.py). Unset means retrieval silently falls back to
# plain RRF ordering, same degrade-silently pattern as LangSmith above.
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
