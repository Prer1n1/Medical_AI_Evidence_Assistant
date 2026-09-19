"""Runs the full agent pipeline over the curated eval set and scores it
with real RAGAS metrics. Ported structure from enterprise-agentic-rag —
Faithfulness (doubles as hallucination detection), ResponseRelevancy, and
LLMContextPrecisionWithoutReference are all domain-agnostic quality
metrics. New here: evidence_level_distribution — not a RAGAS metric, a
project-specific check that retrieval isn't systematically starving
higher-tier evidence (see docs/design-decisions.md, "Evaluation").

API note carried over from the enterprise project: ragas's
llm_factory/embedding_factory (the path its own deprecation warnings
recommend) throws AttributeError with the Faithfulness/ResponseRelevancy
metric classes in the pinned ragas==0.3.9. LangchainLLMWrapper /
LangchainEmbeddingsWrapper are used here because they're what actually
works, despite being flagged deprecated — verify, don't trust the warning
text.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List

from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy

from agent.graph import build_agent_graph
from agent.nodes import DISCLAIMER
from evaluation.eval_dataset import EVAL_CASES
from retrieval.hybrid_retriever import HybridRetriever
from storage.vector_store import get_embeddings


def _strip_disclaimer(answer: str) -> str:
    """RAGAS's ResponseRelevancy metric scores the fixed clinical-safety
    disclaimer (agent.nodes.DISCLAIMER) as a "noncommittal answer" and
    forces the WHOLE score to 0.0 — verified in isolation: the identical
    answer scored 0.98 without this sentence, 0.0 with it, regardless of
    how clinically grounded the substantive content actually is (see
    docs/design-decisions.md, "Evaluation"). The disclaimer is fixed
    boilerplate appended after synthesis (agent/nodes.py), not part of the
    model's actual clinical answer, so scoring metrics against it isn't
    evaluating anything RAGAS is meant to measure — stripped here for
    scoring only; the full answer (with disclaimer) is still what
    print_report() shows and what a real caller receives."""
    return answer.replace(f"\n\n{DISCLAIMER}", "").strip()

# Below this faithfulness score, flag the answer as a likely hallucination.
HALLUCINATION_THRESHOLD = 0.7


@dataclass
class EvalRunResult:
    question: str
    answer: str
    reference_answer: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    likely_hallucination: bool
    evidence_levels_cited: List[str]


def run_evaluation(retriever: HybridRetriever) -> List[EvalRunResult]:
    graph = build_agent_graph(retriever)
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    agent_outputs = []
    samples = []
    for case in EVAL_CASES:
        result = graph.invoke({"query": case.question})

        seen = set()
        contexts = []
        evidence_levels = []
        for chunk in result["retrieved_chunks"]:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                contexts.append(chunk.content)
                evidence_levels.append(chunk.metadata.get("evidence_level", "Unknown"))

        agent_outputs.append((case, result["answer"], evidence_levels))
        samples.append(
            SingleTurnSample(
                user_input=case.question,
                retrieved_contexts=contexts,
                response=_strip_disclaimer(result["answer"]),
                reference=case.reference_answer,
            )
        )

    dataset = EvaluationDataset(samples=samples)
    ragas_result = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithoutReference()],
        llm=llm,
        embeddings=embeddings,
    )
    scores = ragas_result.to_pandas()

    results = []
    for i, (case, answer, evidence_levels) in enumerate(agent_outputs):
        row = scores.iloc[i]
        faithfulness_score = float(row["faithfulness"])
        results.append(
            EvalRunResult(
                question=case.question,
                answer=answer,
                reference_answer=case.reference_answer,
                faithfulness=faithfulness_score,
                answer_relevancy=float(row["answer_relevancy"]),
                context_precision=float(row["llm_context_precision_without_reference"]),
                likely_hallucination=faithfulness_score < HALLUCINATION_THRESHOLD,
                evidence_levels_cited=evidence_levels,
            )
        )
    return results


def print_report(results: List[EvalRunResult]) -> None:
    print(f"{'Question':<50} {'Faith':>6} {'Relev':>6} {'CtxPrec':>8}  Flag")
    print("-" * 85)
    for r in results:
        flag = "LIKELY HALLUCINATION" if r.likely_hallucination else ""
        print(f"{r.question[:48]:<50} {r.faithfulness:>6.2f} {r.answer_relevancy:>6.2f} {r.context_precision:>8.2f}  {flag}")

    n = len(results)
    avg_faith = sum(r.faithfulness for r in results) / n
    avg_relev = sum(r.answer_relevancy for r in results) / n
    avg_prec = sum(r.context_precision for r in results) / n
    print("-" * 85)
    print(f"{'AVERAGE':<50} {avg_faith:>6.2f} {avg_relev:>6.2f} {avg_prec:>8.2f}")

    all_levels = [level for r in results for level in r.evidence_levels_cited]
    print("\nEvidence level distribution across all cited context:")
    for level, count in Counter(all_levels).most_common():
        print(f"  {level:<40} {count}")
