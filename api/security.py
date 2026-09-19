"""API-key authentication for the FastAPI layer. Ported verbatim from
enterprise-agentic-rag, minus role/category scoping (access control was
deliberately dropped from this project's scope — see
docs/design-decisions.md, "Scope"): one shared admin key, no KeyScope.
"""

from __future__ import annotations

import logging
import secrets
from typing import Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from config import API_KEY

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(request: Request, provided: Optional[str] = Security(_api_key_header)) -> None:
    if not provided or not secrets.compare_digest(provided, API_KEY or ""):
        logger.warning(
            "auth_failed",
            extra={"path": request.url.path, "client": request.client.host if request.client else None},
        )
        raise HTTPException(status_code=401, detail="Missing or invalid API key (X-API-Key header)")
