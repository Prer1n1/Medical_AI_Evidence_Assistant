"""Part 1 (free/offline): the no-COHERE_API_KEY fallback, plus a
monkeypatched fake rerank call. Part 2 (gated on COHERE_API_KEY, real
Cohere call): verifies real reranking actually reorders candidates.
Ported pattern from enterprise-agentic-rag — reranking mechanics are
entirely domain-agnostic.
"""

from __future__ import annotations

import os

import config  # noqa: F401 — loads .env (COHERE_API_KEY) into os.environ
import retrieval.reranker as reranker_module


def test_no_api_key_returns_none():
    # Saves and restores the module-level key itself — this script has no
    # pytest fixture to undo the patch automatically. A real bug was found
    # here during verification: an earlier version of this test used a
    # hand-rolled monkeypatch shim with no teardown, which left
    # COHERE_API_KEY permanently None for every test that ran after it in
    # the same process — including the live-rerank test below, which then
    # silently "passed" as a skip-like None instead of actually failing
    # loudly. Fixed by making every mutating test responsible for its own
    # restore, verified by re-running the live test afterward and
    # confirming it exercises the real API again.
    original_key = reranker_module.COHERE_API_KEY
    reranker_module.COHERE_API_KEY = None
    try:
        result = reranker_module.rerank("query", ["doc a", "doc b"], top_n=2)
    finally:
        reranker_module.COHERE_API_KEY = original_key
    assert result is None
    print("OK: no COHERE_API_KEY -> rerank() returns None (caller falls back to RRF order)")


def test_monkeypatched_rerank_reorders():
    class FakeResult:
        def __init__(self, index, relevance_score):
            self.index = index
            self.relevance_score = relevance_score

    class FakeResponse:
        results = [FakeResult(1, 0.9), FakeResult(0, 0.1)]

    def fake_call_rerank(client, query, documents, top_n):
        return FakeResponse()

    original_call = reranker_module._call_rerank
    original_key = reranker_module.COHERE_API_KEY
    reranker_module._call_rerank = fake_call_rerank
    reranker_module.COHERE_API_KEY = "fake-key-for-offline-test"
    try:
        result = reranker_module.rerank("query", ["irrelevant doc", "relevant doc"], top_n=2)
    finally:
        reranker_module._call_rerank = original_call
        reranker_module.COHERE_API_KEY = original_key

    assert result == [(1, 0.9), (0, 0.1)]
    print("OK: monkeypatched rerank correctly reorders by relevance_score")


def test_live_rerank_reorders_real_candidates():
    if not os.getenv("COHERE_API_KEY"):
        print("SKIPPED (no COHERE_API_KEY): live Cohere rerank")
        return
    documents = [
        "The patient reported mild headache and no other symptoms.",
        "WHO recommends a target systolic blood pressure below 130 mmHg for patients with known cardiovascular disease.",
        "The hospital cafeteria menu changes weekly on Mondays.",
    ]
    result = reranker_module.rerank("target blood pressure for cardiovascular disease patients", documents, top_n=3)
    assert result is not None
    top_index, top_score = result[0]
    assert top_index == 1, f"expected the hypertension-target document to rank first, got index {top_index}"
    print(f"OK: live Cohere rerank correctly ranked the clinically relevant document first (score {top_score:.2f})")


if __name__ == "__main__":
    test_no_api_key_returns_none()
    test_monkeypatched_rerank_reorders()
    test_live_rerank_reorders_real_candidates()
    print("All test_reranker.py checks passed.")
