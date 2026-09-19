"""Shared retry policies for transient third-party API failures (OpenAI,
Cohere, NCBI E-utilities).

Uses `tenacity` (a real, well-known retry library) instead of a hand-rolled
retry loop.

Retrying is scoped to genuinely TRANSIENT errors per provider (rate
limits, connection drops, timeouts, 5xx server errors) — retrying an
AuthenticationError or a BadRequestError would just burn several seconds
waiting for an error that will never go away no matter how many times
it's retried.
"""

from __future__ import annotations

import logging

import cohere.errors
import openai
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_TRANSIENT_OPENAI_ERRORS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)

_TRANSIENT_COHERE_ERRORS = (
    cohere.errors.GatewayTimeoutError,
    cohere.errors.InternalServerError,
    cohere.errors.ServiceUnavailableError,
    cohere.errors.TooManyRequestsError,
)


class NCBITransientError(Exception):
    """Raised deliberately by ingestion/connectors/pubmed.py for a 5xx
    response or a connection-level failure from NCBI's E-utilities — a
    distinct type so retrying doesn't accidentally cover an actual 4xx
    (bad query syntax, invalid PMID) that no amount of retrying would fix."""


_TRANSIENT_NCBI_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    NCBITransientError,
)


def _log_retry(retry_state) -> None:
    logger.warning(
        "api_call_retry",
        extra={
            "function": retry_state.fn.__name__ if retry_state.fn else None,
            "attempt": retry_state.attempt_number,
            "error": str(retry_state.outcome.exception()),
        },
    )


# max_attempts=3, exponential backoff starting at 1s (1s, 2s, then gives up):
# enough to ride out a brief network blip or a rate-limit window without
# making a user-facing request hang for a long time waiting on retries that
# were never going to succeed.
retry_openai_call = retry(
    retry=retry_if_exception_type(_TRANSIENT_OPENAI_ERRORS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
    before_sleep=_log_retry,
)

# Same policy shape as retry_openai_call, just scoped to Cohere's own
# transient exception types.
retry_cohere_call = retry(
    retry=retry_if_exception_type(_TRANSIENT_COHERE_ERRORS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
    before_sleep=_log_retry,
)

# More attempts and a longer max wait than the OpenAI/Cohere policies —
# real finding from GitHub Actions CI: NCBI's per-IP rate limit (429) is
# hit far more easily from a shared CI runner IP (many unrelated repos'
# jobs share Microsoft/GitHub's IP ranges) than from a home connection,
# and NCBI's rate-limit window didn't clear within 3 attempts / 10s max
# backoff. 5 attempts up to 30s gives real headroom to actually recover
# instead of just retrying into the same still-rate-limited window.
retry_ncbi_call = retry(
    retry=retry_if_exception_type(_TRANSIENT_NCBI_ERRORS),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
    before_sleep=_log_retry,
)
