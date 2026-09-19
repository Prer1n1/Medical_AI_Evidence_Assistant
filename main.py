"""Command-line entry point.

    python main.py ingest [directory] [--doc-subtype clinical_guideline|treatment_protocol]
    python main.py ingest-pubmed "query" [--retmax 20]
    python main.py ask "question"
    python main.py chat
    python main.py evaluate
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agent.graph import build_agent_graph
from config import API_KEY
from ingestion.chunking import chunk_documents
from ingestion.connectors.pubmed import PubMedSyncTracker, sync_topic
from ingestion.guideline_versioning import apply_versioning
from ingestion.metadata_extractor import enrich_all
from ingestion.pipeline import IngestionResult, ingest_directory
from ingestion.tracker import IngestionTracker
from retrieval.hybrid_retriever import HybridRetriever
from storage.chunk_store import ChunkStore
from storage.vector_store import add_chunks, delete_by_source, get_embeddings, get_vector_store


def _require_api_key() -> None:
    if not API_KEY:
        print("ERROR: API_KEY not found. Copy .env.example to .env and set one.")
        raise SystemExit(1)


def _ingest_and_persist(directory: Path, chunk_store: ChunkStore, vector_store, embeddings, doc_subtype: str) -> IngestionResult:
    tracker = IngestionTracker()
    result = ingest_directory(directory, tracker=tracker, embeddings=embeddings, doc_subtype=doc_subtype)

    for source in result.ingested_files:
        chunk_store.delete_by_source(source)
        delete_by_source(vector_store, source)

    if result.chunks:
        chunk_store.save_chunks(result.chunks)
        add_chunks(vector_store, result.chunks)

    for deleted_source in result.deleted_files:
        chunk_store.delete_by_source(deleted_source)
        delete_by_source(vector_store, deleted_source)

    for file_path in result.pending_mark:
        tracker.mark_ingested(file_path)

    if result.ingested_files:
        superseded = apply_versioning(chunk_store, vector_store, result.ingested_files)
        if superseded:
            print(f"Superseded (older guideline version): {len(superseded)} file(s)")
            for source in superseded:
                print(f"  - {source}")

    return result


def _print_ingest_result(result: IngestionResult) -> None:
    print(f"Ingested (new/changed): {len(result.ingested_files)} file(s)")
    print(f"Skipped (unchanged):    {len(result.skipped_unchanged)} file(s)")
    print(f"Purged (deleted):       {len(result.deleted_files)} file(s)")
    print(f"Chunks stored:          {len(result.chunks)}")


def cmd_ingest(args: argparse.Namespace) -> None:
    embeddings = get_embeddings()
    chunk_store = ChunkStore()
    vector_store = get_vector_store(embeddings=embeddings)

    directory = Path(args.directory)
    print(f"Ingesting: {directory} (doc_subtype={args.doc_subtype})")
    result = _ingest_and_persist(directory, chunk_store, vector_store, embeddings, args.doc_subtype)
    _print_ingest_result(result)


def cmd_ingest_pubmed(args: argparse.Namespace) -> None:
    embeddings = get_embeddings()
    chunk_store = ChunkStore()
    vector_store = get_vector_store(embeddings=embeddings)

    print(f"Searching PubMed: {args.query!r} (retmax={args.retmax})")
    documents = sync_topic(args.query, retmax=args.retmax, tracker=PubMedSyncTracker())
    print(f"New records fetched: {len(documents)}")
    if not documents:
        return

    enriched = enrich_all(documents)
    chunks = chunk_documents(enriched, embeddings=embeddings)
    if chunks:
        chunk_store.save_chunks(chunks)
        add_chunks(vector_store, chunks)
    print(f"Chunks stored: {len(chunks)}")


def _build_agent():
    embeddings = get_embeddings()
    chunk_store = ChunkStore()
    vector_store = get_vector_store(embeddings=embeddings)
    retriever = HybridRetriever(vector_store, chunk_store)
    return build_agent_graph(retriever)


def _print_answer(result: dict) -> None:
    print(f"\nCategories queried: {result['categories']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nConfidence: {result['confidence']['label']} ({result['confidence']['score']}) — {result['confidence']['reasoning']}")
    print("\nCitations:")
    for c in result["citations"]:
        label = f"PMID:{c['pmid']}" if c.get("pmid") else (c.get("guideline_name") or Path(c["source"]).name)
        print(f"  - {label} [{c['evidence_level']}] ({c['category']})")


def cmd_ask(args: argparse.Namespace) -> None:
    graph = _build_agent()
    result = graph.invoke({"query": args.question})
    _print_answer(result)


def cmd_chat(args: argparse.Namespace) -> None:
    graph = _build_agent()
    print("Medical AI Evidence Assistant — type a question, or 'exit' to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        result = graph.invoke({"query": question})
        _print_answer(result)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from evaluation.ragas_eval import print_report, run_evaluation

    embeddings = get_embeddings()
    chunk_store = ChunkStore()
    vector_store = get_vector_store(embeddings=embeddings)
    retriever = HybridRetriever(vector_store, chunk_store)

    print("Running evaluation (RAGAS: faithfulness, relevancy, context precision)...")
    results = run_evaluation(retriever)
    print_report(results)


def main() -> None:
    _require_api_key()
    parser = argparse.ArgumentParser(description="Medical AI Evidence Assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest guideline/protocol documents into persistent storage")
    ingest_parser.add_argument("directory", nargs="?", default="sample_data")
    ingest_parser.add_argument("--doc-subtype", dest="doc_subtype", default="clinical_guideline",
                                choices=["clinical_guideline", "treatment_protocol"])
    ingest_parser.set_defaults(func=cmd_ingest)

    pubmed_parser = subparsers.add_parser("ingest-pubmed", help="Search + ingest PubMed records for a topic")
    pubmed_parser.add_argument("query")
    pubmed_parser.add_argument("--retmax", type=int, default=20)
    pubmed_parser.set_defaults(func=cmd_ingest_pubmed)

    ask_parser = subparsers.add_parser("ask", help="Ask one question")
    ask_parser.add_argument("question")
    ask_parser.set_defaults(func=cmd_ask)

    chat_parser = subparsers.add_parser("chat", help="Interactive Q&A loop")
    chat_parser.set_defaults(func=cmd_chat)

    evaluate_parser = subparsers.add_parser("evaluate", help="Run RAGAS evaluation over the curated eval set")
    evaluate_parser.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
