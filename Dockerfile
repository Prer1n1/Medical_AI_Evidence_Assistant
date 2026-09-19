# Slim, not full — this image only needs Python + pip installs.
FROM python:3.11-slim

WORKDIR /app

# Dependencies before source code — its own cached layer. Changing
# agent/nodes.py later invalidates layers after this point, not this one,
# so a code-only change doesn't reinstall ~25 packages (torch,
# sentence-transformers, chromadb, langchain, ragas...) on every rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the biomedical embedding model INTO the image at build
# time, not on first request at runtime — otherwise the container's first
# real ingestion/query pays a ~400MB download + load penalty inside the
# request path instead of during `docker build`. No API key needed for
# this (it's a public HuggingFace model). Copies only the files
# get_embeddings() actually imports (verified against their real import
# statements, not guessed) — this layer stays cached across changes to
# unrelated files like api/app.py or agent/nodes.py.
COPY config.py .
COPY ingestion/__init__.py ingestion/schema.py ingestion/chunking.py ingestion/
COPY storage/__init__.py storage/chunk_store.py storage/vector_store.py storage/
RUN python -c "from storage.vector_store import get_embeddings; get_embeddings()"

COPY . .

EXPOSE 8000

# Hits the real /health endpoint (checks both stores), not a fake liveness
# ping — same reasoning as enterprise-agentic-rag's Dockerfile.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
