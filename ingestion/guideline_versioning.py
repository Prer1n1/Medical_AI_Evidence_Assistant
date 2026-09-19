"""Guideline versioning — no equivalent in enterprise-agentic-rag at all.
A hospital guideline gets revised periodically (a 2021 hypertension
guideline superseded by a 2023 update); re-ingesting the new PDF shouldn't
silently mix current and outdated recommendations in retrieval, but it
also shouldn't DELETE the old guideline's chunks outright — a clinician or
auditor may legitimately need to see what the prior guidance said.

Two pieces:
  1. derive_guideline_name() — groups different files/versions of "the
     same guideline" together, from the filename alone (no LLM call: this
     needs to run on every ingested file, and a regex covers the common
     naming conventions real guideline PDFs actually ship with —
     "hypertension_guideline_v2.pdf", "who-hypertension-2021.pdf" — well
     enough to be worth the zero cost/latency over an LLM call per file).
  2. apply_versioning() — after new chunks are persisted, finds any OTHER
     source previously ingested under the same guideline_name and marks
     it is_current=False in both stores (via ChunkStore.set_current /
     storage.vector_store.set_current), so retrieval defaults to only the
     current version (see retrieval/hybrid_retriever.py's is_current
     filter) while the old version's chunks remain queryable by anyone who
     explicitly asks about prior guidance.
"""

from __future__ import annotations

import re
from typing import List, Optional

# Strips a trailing version/date token so "x_guideline_v2.pdf" and
# "x-guideline-2023.pdf" collapse to the same base name as
# "x_guideline.pdf". Matches (case-insensitive): "_v1", "-v2", "(v3)",
# a bare 4-digit year, or a year in parens, at the END of the stem.
_VERSION_SUFFIX_RE = re.compile(
    r"[\s_\-\(]*(v(?:ersion)?[\s_\-]?\d+|20\d{2})\)?\s*$", re.IGNORECASE
)


def derive_guideline_name(file_stem: str) -> str:
    """Normalized grouping key for 'the same guideline, any version'.
    Lowercased, version/year suffix stripped, separators collapsed to a
    single space. Documented limitation: this is a filename heuristic, not
    content analysis — two guidelines that happen to share a stripped stem
    but are genuinely unrelated would incorrectly be treated as versions
    of each other. Acceptable for a project at this scale; a production
    system would corroborate with a title/topic similarity check before
    superseding anything automatically."""
    base = _VERSION_SUFFIX_RE.sub("", file_stem).strip()
    normalized = re.sub(r"[\s_\-]+", " ", base).strip().lower()
    return normalized or file_stem.lower()


def derive_effective_date_hint(file_stem: str) -> Optional[str]:
    """Pulls a bare 4-digit year out of the filename, if present — used
    only as a display/sort hint (effective_date), not as the versioning
    key itself (that's guideline_name)."""
    match = re.search(r"20\d{2}", file_stem)
    return match.group(0) if match else None


def apply_versioning(chunk_store, vector_store, ingested_sources: List[str]) -> List[str]:
    """For each newly-persisted source that carries a guideline_name,
    finds every OTHER source sharing that name and marks it superseded in
    both stores. Must run AFTER save_chunks()/add_chunks() have already
    persisted the new chunks (so get_sources_by_guideline_name can see the
    new source's guideline_name to know which siblings to look for) — the
    same "only act after persistence actually succeeded" discipline
    ingestion/pipeline.py's pending_mark ordering uses. Returns the list of
    source paths that were just marked superseded, for logging."""
    superseded: List[str] = []
    seen_guideline_names = set()

    for source in ingested_sources:
        rows = chunk_store.get_by_source(source)
        if not rows:
            continue
        guideline_name = rows[0]["guideline_name"]
        if not guideline_name or guideline_name in seen_guideline_names:
            continue
        seen_guideline_names.add(guideline_name)

        other_sources = chunk_store.get_sources_by_guideline_name(guideline_name, exclude_source=source)
        for other_source in other_sources:
            if other_source in ingested_sources:
                # Both were just (re-)ingested in this same batch — an
                # actual re-ingestion of the identical file, not a new
                # version superseding an old one. Nothing to supersede.
                continue
            chunk_store.set_current(other_source, is_current=False)
            vector_store_updated = vector_store is not None
            if vector_store_updated:
                from storage.vector_store import set_current as set_current_vector

                set_current_vector(vector_store, other_source, is_current=False)
            superseded.append(other_source)

    return superseded
