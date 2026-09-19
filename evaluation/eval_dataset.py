"""Curated evaluation set — hand-picked (question, reference_answer) pairs,
ALL grounded in the real WHO hypertension guideline specifically (see
docs/design-decisions.md, "Sample data"). Deliberately scoped to the one
document `python main.py evaluate` actually has loaded once a user follows
the README's documented setup (`python main.py ingest sample_data/guidelines`)
— an earlier version of this set included a pediatric-dosage question whose
answer only exists in sample_data/protocols_excerpt/, which `evaluate`
never ingests by default. That produced a misleading "LIKELY HALLUCINATION"
flag that was really a corpus/eval-set mismatch, not a real grounding
failure (verified: the exact same question scored perfectly once answered
against a corpus that actually contained the pediatric excerpt — see
test_agent.py). Kept small and hand-curated for the same reason as
enterprise-agentic-rag's set: every RAGAS metric here is an LLM judgment
call per sample, so a bigger set costs real API spend without adding much
signal at this corpus size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class EvalCase:
    question: str
    reference_answer: str  # ground truth, taken directly from the source documents


EVAL_CASES: List[EvalCase] = [
    EvalCase(
        question="At what systolic blood pressure does WHO recommend starting pharmacological treatment for a confirmed hypertension diagnosis?",
        reference_answer="WHO recommends initiating pharmacological antihypertensive treatment for adults with a confirmed diagnosis of hypertension and a systolic blood pressure of 140 mmHg or higher (or diastolic blood pressure of 90 mmHg or higher).",
    ),
    EvalCase(
        question="What target systolic blood pressure does WHO recommend for a hypertensive patient with known cardiovascular disease?",
        reference_answer="WHO recommends a target systolic blood pressure treatment goal of below 130 mmHg for patients with hypertension and known cardiovascular disease — a strong recommendation based on moderate-certainty evidence.",
    ),
    EvalCase(
        question="How often should a patient be followed up after starting or changing antihypertensive medication?",
        reference_answer="WHO suggests a monthly follow-up after initiation of, or a change to, antihypertensive medications.",
    ),
    EvalCase(
        question="When starting pharmacological therapy for hypertension, should WHO's laboratory-testing recommendation ever delay treatment?",
        reference_answer="No — WHO suggests obtaining tests to screen for comorbidities and secondary hypertension when starting pharmacological therapy, but only when testing does not delay or impede initiation of treatment.",
    ),
    EvalCase(
        question="Should WHO recommend pharmacological antihypertensive treatment for a diabetic patient without cardiovascular disease but with a systolic blood pressure of 135 mmHg?",
        reference_answer="Yes — WHO suggests pharmacological antihypertensive treatment for individuals without cardiovascular disease but with high cardiovascular risk, diabetes mellitus, or chronic kidney disease, and a systolic blood pressure of 130-139 mmHg (a conditional recommendation).",
    ),
]
