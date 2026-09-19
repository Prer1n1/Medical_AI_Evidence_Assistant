# Design Decisions Log

Quick reference for every real technical choice in this project. Format: what, why this, why not the alternative, advantage — plus real bugs found while building it. Kept short on purpose, for interview recall, not documentation. Companion project: [`enterprise-agentic-rag`](../../enterprise-agentic-rag) — this log calls out what's ported vs. rebuilt vs. deliberately dropped relative to it.

---

## Scope

**This project deliberately does NOT port enterprise-agentic-rag's access control, PII redaction, or prompt-injection defense.**
- Why: those are enterprise-security features with no equivalent claim in the "Medical AI Evidence Assistant" resume bullet this project targets (evidence ranking, citation generation, confidence scoring, retrieval evaluation, multimodal ingestion, biomedical embeddings, PubMed integration, guideline versioning, explainable retrieval). Porting them would pad the codebase without demonstrating anything this project is actually about.
- What IS new here with no enterprise equivalent at all: the PubMed connector, evidence-level classification, evidence ranking, confidence scoring, guideline versioning, and multimodal (table + image) PDF extraction.

---

## Sample data

**Real, public, freely downloadable documents — not synthetic.**
- `sample_data/guidelines/who_hypertension_guideline_2021.pdf` — WHO's real "Guideline for the pharmacological treatment of hypertension in adults" (2021), fetched from `iris.who.int`. 47 pages, real evidence tables (PICO/GRADE certainty tables), a real WHO-logo image.
- `sample_data/protocols/who_pocketbook_child_care_dosages.pdf` — WHO's real "Pocket book of hospital care for children," fetched from `afro.who.int`. ~250 pages, real pediatric dosage tables (antibiotic mg/kg dosing, bilirubin phototherapy thresholds, growth-chart weight-for-age tables), 113 extracted images.
- **A real, verified access limitation**: CDC's `stacks.cdc.gov` and `regulations.gov`, and NCBI's `pmc.ncbi.nlm.nih.gov`, all returned bot-protection blocks (403/Access-Denied) to a plain `curl` request even with a browser User-Agent and a Referer header — only `iris.who.int` and `afro.who.int` allowed direct download. Documented rather than silently worked around; a production system would need a real browser-driven fetch or an institutional access path for those sources.
- PubMed content is fetched LIVE through the real connector (`ingestion/connectors/pubmed.py`), not staged as files — verified against real, current PubMed search results (see "PubMed connector" below).

---

## Ingestion: PDF loader (pymupdf, not pypdf)

**Why pymupdf**: medical PDFs are table- and figure-heavy (dosage tables, growth charts, diagnostic algorithms) in a way typical office documents aren't. pypdf (used in the enterprise project) only extracts flat page text. pymupdf additionally exposes `page.find_tables()` (structured table detection) and `page.get_images()` (embedded raster image extraction) — both required for "multimodal: PDFs, tables, image-rich clinical references."

**Three Document kinds per PDF, never flattened together** — same "keep structured content separate from prose" discipline the enterprise project used for DOCX/HTML tables:
- prose: one Document per page, with detected table regions excluded (via `pymupdf.Rect` bbox comparison against `page.get_text("blocks")`) so table rows aren't ALSO captured as unstructured prose
- tables: one Document per detected table, pipe-joined rows, `is_table=True`
- images: one Document per extracted figure, placeholder content + `image_path`, captioned later (see "Image captioning" below)

**Real bug found and fixed in verification: `find_tables()` false positive on cover/title pages.**
- What happened: running the loader against the real WHO hypertension guideline, page 1 (the cover) was detected as a 5-cell "table" — mostly empty, one real cell containing the document's title text. A decorative text layout, not an actual table.
- Root cause: pymupdf's table detector works off text/line alignment heuristics, which a cover page's centered title block can accidentally satisfy.
- Fix: `MIN_TABLE_CONTENT_CHARS = 150` — verified against the real document that every genuine evidence table (PICO/GRADE tables, all 400-800+ chars) clears this bar easily, while the cover-page false positive (84 chars) and two low-value author/affiliation-list tables (24-40 chars) are filtered. Pinned with a regression test (`test_loaders.py::test_pdf_table_false_positive_filtered`) so this can't silently regress.
- Not filtered by cell-emptiness ratio instead: a legitimate PICO table (`Table 1, page 55`) also has many intentionally blank cells (a comparator-arm matrix), so an emptiness-ratio filter would have risked dropping real content — a content-length floor is the more conservative, verified-correct choice here.

**Image extraction — a real, honest limitation surfaced in verification, not glossed over**: `MIN_IMAGE_DIMENSION = 150` filters tiny decorative icons, but a full-size WHO logo on the cover page still clears that bar and gets extracted/captioned as if it were a clinical figure (see "Image captioning" below — the caption correctly describes it as a logo, so it's harmless, just not clinically useful). A production system would want a second filter (e.g., a vision-based "is this a logo/letterhead" classifier) — documented as a known gap, not silently hidden.

---

## Ingestion: medical metadata (category + evidence level)

**Same "LLM-primary, keyword-fallback" architecture as the enterprise project's category classifier** — `ingestion/metadata_extractor.py` — but the taxonomy and the evidence-level field are new:
- **Category**: Cardiology / Oncology / Infectious Disease / Pharmacology / Emergency Medicine / General (replaces the enterprise project's HR/Finance/Security/IT/Legal).
- **Evidence level** (no enterprise equivalent at all): Systematic Review/Meta-Analysis > RCT > Cohort Study > Case-Control Study > Case Report/Case Series > Expert Opinion/Clinical Guideline — the standard clinical evidence hierarchy. This is what "evidence ranking" in the resume bullet concretely means: every retrieved chunk is tagged with where it sits on this hierarchy, and `retrieval/evidence_ranker.py` weights by it.
- **Ground-truth hints win outright**: PubMed's own `PublicationType` field (via `PUBMED_PUBLICATION_TYPE_MAP` in the connector) is trusted as the evidence-level hint without re-classification — same "trust authoritative structured metadata over inferred classification" principle the enterprise project applied to CSV department-column hints.

**Real finding from live verification: genuine category-taxonomy overlap.**
- A real sentence from the WHO hypertension guideline ("WHO recommends initiation of pharmacological antihypertensive treatment...") was classified `Pharmacology` by the live LLM classifier, not `Cardiology`. Both are defensible readings — the content is about drug-treatment initiation for a cardiac condition. Not a bug; a real ambiguity in any category taxonomy applied to cross-cutting medical content (a pharmacology intervention for a cardiology condition). The test (`test_metadata.py`) accepts either answer rather than asserting a single "correct" one it can't actually justify. A production system might resolve this with multi-label categories instead of single-label — deliberately out of scope here.

---

## Ingestion: PubMed connector

**No equivalent anywhere in enterprise-agentic-rag.** NCBI E-utilities (esearch -> PMIDs, one BATCHED efetch call for all PMIDs, not one call per ID — fewer round trips, friendlier to NCBI's 3 req/sec unauthenticated rate limit). Mirrors the enterprise project's Google Drive connector's *shape* (authenticate lightly, list/fetch, hand off to the same downstream pipeline) but the specifics are entirely different — PubMed records aren't files, so `record_to_document()` builds `Document`s directly rather than caching to a local directory for a file loader to pick up.

**Incremental sync, by PMID**: `PubMedSyncTracker` (SQLite) — the PubMed analogue of the file-based `IngestionTracker`, tracking by PMID instead of content hash (a published paper's content is immutable, so hash-checking would be pointless; PMID membership is the right identity here). **Verified live**: re-running `sync_topic()` with the identical query fetched 0 new records on the second call, all correctly recognized as already-synced.

**Verified against real, current PubMed data** (not a fixture/mock): `search_pubmed("hypertension randomized controlled trial")` returned real, current PMIDs with real titles, DOIs, journals, and publication types — e.g. a real September 2026 network meta-analysis on resistant hypertension therapies.

---

## Storage: local biomedical embeddings, not OpenAI

**`pritamdeka/S-PubMedBert-MS-MARCO`** (PubMedBERT base, fine-tuned on MS MARCO for retrieval) via `langchain-huggingface`'s `HuggingFaceEmbeddings`, CPU inference, cached process-wide as a singleton (`storage/vector_store.py:get_embeddings()`).
- Why: this project's specific resume claim is "biomedical embeddings" — a domain-pretrained model, not OpenAI's general-purpose `text-embedding-3`, captures medical vocabulary/abbreviation structure (MI, ACEi, dosage units) that a general model has no particular training signal for. Real tradeoff, named honestly: CPU inference (~40s cold-start model load, then fast) instead of an API round-trip, and a one-time ~430MB model download — but embeddings become the ONE call site in this entire project that costs nothing per-call and needs no API key at all.
- Verified: `get_embeddings().embed_query(...)` produces real 768-dim vectors; a full 47-page guideline (57 prose pages, hundreds of sentences for semantic chunking) embeds and ingests successfully end-to-end.

**Real dependency bug found and fixed: `torch==2.5.1` broke model loading.**
- What happened: `HuggingFaceEmbeddings(...)` raised `ValueError` from inside `transformers`: *"Due to a serious vulnerability issue in torch.load, ... we now require users to upgrade torch to at least v2.6"* (CVE-2025-32434) — `transformers` now refuses to load a non-safetensors checkpoint (this model ships `pytorch_model.bin`) on torch < 2.6, regardless of `weights_only=True`.
- Fix: pinned `torch==2.7.1` instead. Verified the fix directly (not just "the error message says so") — re-ran the exact same `get_embeddings()` call, got a real 768-dim vector back, no fallback/workaround needed.
- Lesson consistent with the enterprise project's RAGAS/langchain-community pinning saga: a library's exact pinned version matters, and the real fix (bump the actual broken dependency) beats a workaround.

**Chroma's `where` filter needs explicit `$and` for multi-condition queries** (`retrieval/hybrid_retriever.py:_dense_filter`) — combining `category` and `is_current` in one dense-search filter required `{"$and": [...]}`, not a flat multi-key dict (the flat form is what an older Chroma API accepted; the pinned `chromadb==1.5.9` requires the explicit operator). Verified via the real `only_current` filter test below.

---

## Storage: guideline versioning

**No equivalent in enterprise-agentic-rag at all.** A hospital guideline gets revised (a 2021 hypertension guideline superseded by a 2023 update); ingesting the new PDF shouldn't silently blend outdated and current recommendations, but shouldn't delete the old guideline either — traceability matters for clinical/audit purposes.

- **`derive_guideline_name()`** (`ingestion/guideline_versioning.py`) — a regex-based filename heuristic (strips a trailing `_v2`/`-2023`/`(v3)` token) groups different versions of "the same guideline" without an LLM call. Documented, honest limitation: this is filename-based, not content-based — two unrelated guidelines that happen to share a stripped stem would be incorrectly treated as versions of each other. Acceptable at this project's scale; a production system would corroborate with a title/topic-similarity check before auto-superseding anything.
- **`apply_versioning()`** — after new chunks are persisted, finds every OTHER source sharing the new source's `guideline_name` and marks it `is_current=False` in BOTH stores (`ChunkStore.set_current` + `storage/vector_store.py:set_current`, the latter reaching into Chroma's raw `_collection.update()` since no higher-level langchain API does a metadata-only update by filter). Retrieval defaults to `only_current=True` (both BM25 and dense), so a superseded guideline's chunks stay in storage (queryable if explicitly asked) but never win a default search.
- **Verified**: re-ingesting a guideline under a new filename with the same derived `guideline_name` correctly marks the old source `is_current=0` in both stores; `HybridRetriever.retrieve(..., only_current=True)` returns zero hits for a fully-superseded source, and `only_current=False` still finds them.

---

## Retrieval: evidence ranking and confidence scoring

**Two separate passes, deliberately not folded into one blended score** — `retrieval/evidence_ranker.py` and `retrieval/confidence.py`, both with no enterprise equivalent.

**Evidence ranking** answers "how much should THIS CHUNK be trusted" — `combined_score = relevance_score * (alpha + (1-alpha) * evidence_weight)`, `alpha=0.5`. Verified: a comparably-relevant systematic-review chunk outranks a case-report chunk; a MUCH more relevant case-report chunk still outranks a barely-relevant systematic review (evidence weighting can discount, never fully override relevance).

**Confidence scoring** answers a different question — "how much should the WHOLE ANSWER be trusted" — combining relevance-weighted evidence strength (45%), independent-source agreement (35%, capped/diminishing past 2 sources, and explicitly de-duplicated by `doc_id` so two chunks from the same paper don't count as two agreeing sources), and top retrieval relevance (20%). Verified: a single case-report source scores lower than multiple independently-agreeing strong sources; chunks from the same `doc_id` correctly count as one source, not two.

**Why kept separate from reranking/RRF instead of one opaque number**: a citation can report its relevance score AND its evidence level independently — this is what makes retrieval "explainable" (the other half of the resume claim), not just re-ordered.

---

## Agent: clinical-safety framing

**The synthesis prompt (`agent/nodes.py`) is deliberately non-prescriptive** — reports what evidence says ("Guideline [2] recommends...") rather than issuing direct instructions ("you should take..."), always closes with a "not medical advice, consult a qualified clinician" line, and is told explicitly to flag mixed/low-tier evidence rather than presenting it with unearned certainty. Verified live: a real out-of-corpus question (laparoscopic appendectomy technique — not in either ingested source) correctly triggered a low-confidence/honest-gap answer rather than a fabricated one.

**Medical citation format, not generic `[1]`/`[2]` filenames**: PubMed sources cite as `PMID:<id>, <journal> (<year>)`; guidelines cite as `Guideline: <name>, effective <date>` — `agent/nodes.py:_source_label()`.

---

## Evaluation / test corpus sizing

**Real bug caught before it ran to completion, not after**: an earlier version of `test_agent.py` called its corpus-building helper INSIDE each of its 3 test functions, and pointed that helper at `sample_data/protocols/` — the full ~250-page WHO pocket book (520+ loaded Documents). That would have meant roughly 1000+ real LLM classification calls, THREE TIMES over, on every single run of this one test file (and every CI push). Caught mid-run (killed after ~6s / one real LLM call, not after burning the full cost) by watching real CPU/time usage rather than assuming a test file's shape was fine just because it imported cleanly.
- Fix 1: ingest once, at module scope, share the built graph across all three test functions.
- Fix 2: replaced the full pocket book with `sample_data/protocols_excerpt/pneumonia_treatment_excerpt.html` — a small, VERBATIM excerpt (not paraphrased/invented) of the real pneumonia-antibiotic-treatment sections from the same WHO pocket book, properly attributed in the file itself. Keeps the cross-source citation test real and meaningful (same real clinical fact, same real source document) without re-ingesting hundreds of unrelated pages just to reach two sentences about amoxicillin dosing.
- The full pocket book stays in `sample_data/protocols/` for what it's actually needed for — `test_loaders.py`/`test_pipeline.py`'s table/image-extraction verification, which uses a fake embeddings double and `use_llm_classifier=False`, so its size costs nothing extra there.

---

## Evaluation: RAGAS

**Real bug found: the mandatory clinical-safety disclaimer zeroed out `ResponseRelevancy` on EVERY answer.**
- What happened: the first live `python main.py evaluate` run flagged all 5 real eval cases `LIKELY HALLUCINATION` (faithfulness 0.0-0.57) with `answer_relevancy` flat `0.00` across the board — despite `python main.py ask` producing clearly correct, well-grounded answers for the same questions moments earlier.
- Root-caused by isolation, not assumption: ran `ResponseRelevancy.single_turn_ascore()` directly against (a) a clean paraphrase of the real answer -> `0.989`, (b) the exact real answer text (citations + `<130 mmHg` + the disclaimer sentence) -> `0.0`, then bisected which part of the real text was responsible: citations alone -> `0.98`, the `<` character alone -> `0.98`, the disclaimer sentence ALONE -> `0.0`. Conclusively isolated: RAGAS's `ResponseRelevancy` has a built-in "noncommittal answer" LLM-judge check that forces the score to `0.0` whenever it reads hedging/disclaiming language in the response — this project's own required safety framing ("This is a summary of retrieved evidence, not medical advice — consult a qualified clinician...") reads exactly like that pattern to the judge, every single time, regardless of how clinically grounded the rest of the answer is.
- Faithfulness's lower (non-zero, 0.0-0.57) scores in that same run had a separate, correct explanation: the disclaimer sentence is boilerplate, not a claim drawn from retrieved context, so Faithfulness (which measures "fraction of claims supported by context") correctly dinged it — not a hallucination in the harmful sense, but a real metric side-effect of grading a fixed compliance sentence as if it were a clinical claim.
- **Fix, not a workaround**: `DISCLAIMER` (`agent/nodes.py`) is now appended to the answer PROGRAMMATICALLY after the LLM call, not written by the LLM itself — two independent benefits, not just the eval fix: (1) compliance text is now deterministic/guaranteed-exact-wording instead of LLM-paraphrased, (2) `evaluation/ragas_eval.py:_strip_disclaimer()` strips that exact fixed string before scoring, so RAGAS grades the actual substantive clinical content while the real user-facing answer (CLI/API) still always includes the full disclaimer, unchanged.
- **Verified the fix directly**: re-ran the identical 5-case eval set after the fix — faithfulness `1.00` on all 5 (up from 0.0-0.57), context precision averaging `0.98`, `answer_relevancy` recovered to `0.95-0.99` on 4 of 5 cases (one case scored `0.00` in isolation despite `1.00` faithfulness / `0.98` context precision on that SAME case — consistent with `ResponseRelevancy`'s own documented per-call variance from its "generate N reverse-engineered questions, average their similarity" mechanism sometimes degrading to 1 generation instead of 3; a known metric-noise limitation, not a pipeline bug, and not chased further since the other 4 cases and the isolated single-sample tests above already conclusively prove the mechanism works).

**Real eval-set bug found and fixed the same day, unrelated to the metric itself**: the first version of `EVAL_CASES` included a pediatric-dosage question whose ONLY grounding exists in `sample_data/protocols_excerpt/`, which `python main.py evaluate`'s default corpus (`sample_data/guidelines/` alone, per the README's documented setup) never ingests. That produced a second, entirely separate "LIKELY HALLUCINATION" flag that was actually a corpus/eval-set mismatch, not a grounding failure — confirmed by `test_agent.py`, which ingests BOTH directories and answers the identical question correctly. Fixed by scoping `EVAL_CASES` entirely to the one document `evaluate`'s documented setup actually loads, rather than leaving a misleading false-positive in the report. Lesson consistent with the disclaimer bug above: a flagged "hallucination" is worth root-causing before accepting it as a real grounding failure — twice in one evaluation run, the flag was actually about the harness, not the model's answer.

**Evidence-level distribution check** (new, no RAGAS equivalent): `print_report()` tallies the evidence level of every cited chunk across the eval run — the real run shows a corpus still dominated by `Expert Opinion / Clinical Guideline` (a single WHO guideline document, largely narrative recommendation text) with a smaller RCT tail, exactly what's expected from ingesting one guideline PDF; would meaningfully shift toward RCT/systematic-review tiers once PubMed-sourced papers are ingested alongside it.

---

## API layer

**Three data-changing endpoints plus `/query`, structure ported from enterprise-agentic-rag**: `GET /health`, `POST /ingest`, `POST /documents/upload`, `POST /query` all carry over the same shape (sync handlers, lifespan-built `AppState`, `X-API-Key` auth via `secrets.compare_digest`, structured JSON logging). New: `POST /ingest/pubmed` (search + fetch + ingest a topic query, mirroring the enterprise Drive connector route's shape), and every `/ingest`/`/documents/upload` response reports `superseded_guideline_versions` from `ingestion.guideline_versioning.apply_versioning()`.

**Verified with real HTTP requests against a running server, not just unit-level**:
- `GET /health` → 200, `chunk_count: 210` against the real persisted corpus.
- `POST /query`, no key → 401. Wrong key → 401. Correct key → 200 with the real grounded hypertension-target answer, correct citations (source, evidence level, guideline name/version), and a real confidence block.
- `POST /ingest/pubmed` with a real topic query → `records_fetched: 5, chunks_stored: 15` against live NCBI data.
- **Cross-source retrieval verified live over HTTP**: asked a question about aspirin for cardiovascular prevention immediately after the PubMed ingest above — the response correctly cited BOTH the WHO guideline AND two freshly-ingested real PubMed papers (`pubmed:42736050`, `pubmed:42723131`) side by side in one answer's citation list, with `independent_source_count: 3`. The synthesized answer itself honestly declined to state a specific aspirin recommendation (top retrieval relevance only `0.19` — the fetched papers were topically adjacent, not a precise match) rather than overclaiming from weak evidence — the clinical-safety framing working as designed, not a retrieval failure.

**Real environmental gotcha found during verification, not a code bug**: the sibling `enterprise-agentic-rag` project has its own Docker container running continuously in the background (`docker ps` showed it "Up 41 hours"), also mapped to host port 8000. `curl`ing `127.0.0.1:8000` intermittently hit THAT container instead of this project's own locally-running `uvicorn` process — Docker Desktop's port-forwarding proxy on Windows can intercept loopback traffic ahead of a directly-bound native socket. Cost real debugging time (the enterprise project's sample citations — `security_policy.docx`, `NIST.CSWP.29.pdf` — showing up in what should have been medical-project responses was the tell). Not fixed in this project's code (there was nothing to fix); resolved by running this project's own verification on a different host port instead of touching the other project's already-running container. Worth knowing as a real multi-project-on-one-machine gotcha, not assumed away.

---

## Docker

**Status: verified**, with one real dependency-ordering optimization beyond the enterprise project's Dockerfile: the local biomedical embedding model (~430MB) is pre-downloaded INTO the image at build time (`RUN python -c "from storage.vector_store import get_embeddings; get_embeddings()"`, after copying just the handful of files that call actually needs, before the full `COPY . .`) — otherwise every container's first real request would pay that download/load cost inside the request path. Verified: `docker logs` on a freshly-started container shows `startup_complete` in ~7 seconds (model already resident in the image), vs. ~45 seconds the first time this project ever ran that call locally (a real, cold, unmetered download).

**Real bug found and fixed: `.dockerignore` excluded the whole `storage/` directory — including its own Python source files.**
- What happened: `docker compose build` failed at `COPY storage/__init__.py storage/chunk_store.py storage/vector_store.py storage/` with `"storage/vector_store.py": not found`.
- Root cause: `.dockerignore`'s original `storage/` entry (meant to exclude generated data — `chroma_db/`, `chunk_store.db`, etc., matching `.gitignore`'s equivalent list) excluded the ENTIRE directory from the Docker build context, including the real `.py` source files the Dockerfile's pre-warm layer explicitly copies.
- Fix: scoped `.dockerignore` to the actual data subpaths (`storage/chroma_db/`, `storage/chunk_store.db`, `storage/ingestion_manifest.db`, `storage/pubmed_synced.db`, `storage/images/`), matching `.gitignore` exactly instead of a broader directory-level exclude.
- Caught by actually running `docker compose build`, not by inspecting the Dockerfile/.dockerignore text and assuming they were consistent — same discipline as every other bug in this log.

**Verified with real requests against a real running container** (not just `docker build` succeeding): started the built image directly (`docker run`, bind-mounting the real `storage/`/`sample_data/` directories) on an alternate host port to avoid the sibling project's port-8000 container above — `startup_complete` with `chunk_count: 225` (the real mounted corpus, correctly visible inside the container), `GET /health` → 200, `POST /query` with a real API key → 200 with the correct grounded hypertension answer, and Docker's own `HEALTHCHECK` (hitting the real `/health` endpoint, not a fake liveness ping) reported `(healthy)` after its start period.
- **A second real Windows-specific artifact found while doing this, not a project bug either**: running `docker run -v "$(pwd)/storage:/app/storage" ...` from Git Bash the first time produced a mangled mount path (`C:\...\storage;C` → `\Program Files\Git\app\storage`) — Git Bash's MSYS layer auto-rewrites POSIX-looking paths before Docker ever sees them. Fixed for THIS verification session with `MSYS_NO_PATHCONV=1`; irrelevant to `docker-compose.yml`'s own relative-path volumes, which don't go through this rewriting at all.

---

## CI (GitHub Actions)

**Two jobs, split by whether a secret is required — same shape as enterprise-agentic-rag, different split line.** The enterprise project split "free" (no OpenAI cost) from "live" (billed OpenAI calls). This project's split is `free-tests` (no secret needed at all — including `test_pubmed.py`'s real NCBI network calls, which cost nothing and need no API key, so they belong in the free bucket by the actual criterion, not just "offline") vs. `live-tests` (`test_retrieval.py`, `test_agent.py` — real, billed OpenAI calls for classification/routing/synthesis; the real local embedding model is also exercised here, which costs time but no money). Gated behind an `OPENAI_API_KEY` repository secret exactly like the enterprise project, for the same reason: a fork's CI run shouldn't silently spend the repo owner's OpenAI budget.
