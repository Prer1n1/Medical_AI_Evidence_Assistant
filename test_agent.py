"""Live — the full LangGraph pipeline end-to-end: real local embeddings,
real LLM routing/classification/synthesis (OPENAI_API_KEY required). Runs
real cross-source questions against BOTH the WHO hypertension guideline
AND a small real excerpt of the WHO pocket book of hospital care for
children.

Ingests ONCE (module-level, not per-test) and shares the built graph
across all three assertions — a real earlier version of this file called
_build_agent_over_real_corpus() inside EACH test function, which
re-ingested the corpus 3 times over. Caught before it ran to completion:
harmless for the (already-small) guidelines PDF, but the first version of
this test pointed at the FULL ~250-page pocket book directory, which would
have meant ~1000+ real LLM classification calls, three times over, on
every CI run. Fixed by (a) ingesting once, shared across tests, and
(b) replacing the full pocket book with a small real excerpt
(sample_data/protocols_excerpt/) for automated runs — see
docs/design-decisions.md, "Evaluation / test corpus sizing".
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import config  # noqa: F401 — loads .env
from agent.graph import build_agent_graph
from ingestion.pipeline import ingest_directory
from ingestion.tracker import IngestionTracker
from retrieval.hybrid_retriever import HybridRetriever
from storage.chunk_store import ChunkStore
from storage.vector_store import add_chunks, get_embeddings, get_vector_store


@contextmanager
def _temp_dir():
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _build_agent_over_real_corpus(tmp: Path):
    embeddings = get_embeddings()
    chunk_store = ChunkStore(db_path=tmp / "chunks.db")
    vector_store = get_vector_store(embeddings=embeddings, persist_directory=tmp / "chroma")
    tracker = IngestionTracker(db_path=tmp / "manifest.db")

    for directory, doc_subtype in [
        (Path("sample_data/guidelines"), "clinical_guideline"),
        (Path("sample_data/protocols_excerpt"), "treatment_protocol"),
    ]:
        result = ingest_directory(
            directory, tracker=tracker, embeddings=embeddings, doc_subtype=doc_subtype,
            use_image_captioning=False,
        )
        if result.chunks:
            chunk_store.save_chunks(result.chunks)
            add_chunks(vector_store, result.chunks)
        for file_path in result.pending_mark:
            tracker.mark_ingested(file_path)

    retriever = HybridRetriever(vector_store, chunk_store)
    return build_agent_graph(retriever)


def test_focused_question_answers_grounded_with_citations(graph):
    result = graph.invoke({"query": "What target systolic blood pressure does WHO recommend for a patient with known cardiovascular disease?"})

    assert "130" in result["answer"], f"expected the real <130 mmHg target in the answer, got: {result['answer']}"
    assert len(result["citations"]) > 0
    assert result["confidence"]["label"] in {"High", "Moderate", "Low"}
    assert "consult a qualified clinician" in result["answer"].lower(), "clinical-safety disclaimer must be present"
    print(f"OK: focused question answered with citations and confidence={result['confidence']['label']}")
    print(f"    answer: {result['answer'][:200]}...")


def test_pediatric_dosage_question(graph):
    result = graph.invoke({"query": "What oral antibiotic should a child be switched to once they improve from pneumonia treatment?"})

    assert "amoxicillin" in result["answer"].lower(), f"expected amoxicillin in the answer, got: {result['answer']}"
    assert len(result["citations"]) > 0
    print(f"OK: pediatric dosage question answered correctly, confidence={result['confidence']['label']}")


def test_no_relevant_evidence_is_honest_not_fabricated(graph):
    result = graph.invoke({"query": "What is the recommended surgical technique for laparoscopic appendectomy?"})

    # The corpus has no appendectomy content at all — a good answer either
    # says it found nothing relevant, or clearly signals low confidence
    # rather than fabricating a specific technique.
    low_confidence_or_honest_gap = (
        result["confidence"]["label"] == "Low"
        or "couldn't find" in result["answer"].lower()
        or "no evidence" in result["answer"].lower()
    )
    assert low_confidence_or_honest_gap, f"expected an honest gap signal, got: {result['answer']} (confidence={result['confidence']})"
    print("OK: out-of-corpus question correctly signaled as low-confidence/no-evidence rather than fabricated")


if __name__ == "__main__":
    with _temp_dir() as tmp:
        graph = _build_agent_over_real_corpus(Path(tmp))
        test_focused_question_answers_grounded_with_citations(graph)
        test_pediatric_dosage_question(graph)
        test_no_relevant_evidence_is_honest_not_fabricated(graph)
    print("All test_agent.py checks passed.")
