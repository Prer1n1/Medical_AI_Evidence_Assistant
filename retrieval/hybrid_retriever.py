"""Hybrid retriever — fuses dense (Chroma) and sparse (BM25) rankings via
Reciprocal Rank Fusion, same "rank position, not raw score" reasoning as
enterprise-agentic-rag. New here: an evidence-ranking pass after reranking
(retrieval/evidence_ranker.py) and an only_current filter (excludes
superseded guideline versions by default, see
ingestion/guideline_versioning.py) applied to BOTH retrievers.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from retrieval.bm25_retriever import BM25Retriever
from retrieval.evidence_ranker import apply_evidence_ranking
from retrieval.reranker import rerank
from retrieval.types import RetrievedChunk
from storage.chunk_store import ChunkStore
from storage.vector_store import similarity_search

RRF_K = 60  # standard constant from the original RRF paper (Cormack et al.)

_METADATA_FIELDS = (
    "source", "doc_type", "doc_subtype", "section", "page_number", "category",
    "evidence_level", "guideline_name", "effective_date", "doc_id",
    "pmid", "journal", "doi",
)


def reciprocal_rank_fusion(rankings: List[List[str]], k: int = RRF_K) -> Dict[str, float]:
    scores: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1 / (k + rank)
    return scores


def _dense_filter(category: Optional[str], only_current: bool) -> Optional[dict]:
    conditions = []
    if category:
        conditions.append({"category": category})
    if only_current:
        conditions.append({"is_current": 1})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}  # Chroma requires $and to combine multiple equality filters


class HybridRetriever:
    def __init__(self, vector_store, chunk_store: ChunkStore):
        self.vector_store = vector_store
        self.chunk_store = chunk_store
        self.bm25 = BM25Retriever(chunk_store)
        self._rows_by_id = {r["chunk_id"]: r for r in self.bm25.rows}

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        fetch_k: int = 20,
        category: str = None,
        use_reranker: bool = True,
        use_evidence_ranking: bool = True,
        only_current: bool = True,
    ) -> List[RetrievedChunk]:
        """category=None searches the whole corpus; only_current=True (the
        default) excludes chunks from a guideline version that's been
        superseded by a newer one. use_reranker=False/use_evidence_ranking=False
        skip those passes for tests that need to stay free/offline/deterministic."""
        dense_docs = similarity_search(
            self.vector_store, query, k=fetch_k, filter=_dense_filter(category, only_current)
        )
        dense_ranking = [d.metadata["chunk_id"] for d in dense_docs if d.metadata.get("chunk_id")]

        sparse_results = self.bm25.search(query, k=fetch_k, category=category, only_current=only_current)
        sparse_ranking = [chunk_id for chunk_id, _ in sparse_results]

        fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking])
        candidate_ids = sorted(fused, key=fused.get, reverse=True)[:fetch_k]

        candidates = []
        for chunk_id in candidate_ids:
            row = self._rows_by_id.get(chunk_id)
            if row is None:
                continue
            candidates.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    content=row["content"],
                    metadata={key: row[key] for key in _METADATA_FIELDS},
                    score=fused[chunk_id],
                )
            )

        if use_reranker and candidates:
            reranked = rerank(query, [c.content for c in candidates], top_n=min(fetch_k, len(candidates)))
            if reranked is not None:
                results = []
                for index, relevance_score in reranked:
                    chunk = candidates[index]
                    chunk.score = relevance_score
                    results.append(chunk)
                candidates = results
        # else: no reranker configured/failed — candidates stay in RRF order

        if use_evidence_ranking:
            candidates = apply_evidence_ranking(candidates)

        return candidates[:top_k]
