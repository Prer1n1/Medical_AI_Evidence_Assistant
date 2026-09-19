"""Sparse / keyword retrieval — BM25 over the chunk store's persisted
text. Catches exact term matches (drug names, PMIDs, dosage numbers) that
a semantic embedding can blur together; dense embedding search catches
paraphrases/synonyms pure keyword matching misses. Ported from
enterprise-agentic-rag — the fusion argument is domain-agnostic. New here:
`only_current` defaults to excluding chunks superseded by a newer
guideline version (see ingestion/guideline_versioning.py) — a stale
recommendation shouldn't win a keyword match just because it used the
same terminology as the current one.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from rank_bm25 import BM25Okapi

from storage.chunk_store import ChunkStore

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(self, chunk_store: ChunkStore):
        self.rows = chunk_store.get_all()
        corpus = [_tokenize(r["content"]) for r in self.rows]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(
        self,
        query: str,
        k: int = 10,
        category: Optional[str] = None,
        only_current: bool = True,
    ) -> List[Tuple[str, float]]:
        """Returns [(chunk_id, bm25_score), ...] ranked best first.

        category filters AFTER scoring against the full corpus (keeps
        BM25's IDF statistics computed over the real corpus size, not a
        skewed per-category subset). only_current does the same
        post-scoring filter for is_current, for the same reason.
        """
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.rows, scores), key=lambda pair: pair[1], reverse=True)
        if category:
            ranked = [pair for pair in ranked if pair[0]["category"] == category]
        if only_current:
            ranked = [pair for pair in ranked if pair[0]["is_current"]]
        return [(row["chunk_id"], score) for row, score in ranked[:k]]
