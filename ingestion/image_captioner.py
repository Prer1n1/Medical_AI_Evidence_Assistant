"""Image captioning — the piece that actually makes an extracted clinical
figure (dosage chart, diagnostic algorithm, imaging scan) retrievable at
all. A raster image has no text of its own to embed or BM25-index; without
a caption, ingestion/loaders/pdf_loader.py's placeholder text
("[Image extracted from page N, not yet captioned]") is all retrieval
would ever see, which means the figure could never surface for a real
clinical question about its content.

Uses GPT-4o-mini's vision input (OpenAI's ChatOpenAI with an image_url
content block) — no new dependency, reuses the same OPENAI_API_KEY every
other LLM call in this project already needs. Optional and degrades
gracefully, same "an enhancement, never a hard dependency" pattern as
Cohere reranking: if OPENAI_API_KEY is unset or the call fails for any
reason, the placeholder text is left as-is rather than blocking ingestion.
The image file itself is still saved and still linked from the chunk's
metadata either way — captioning only affects whether it's *searchable*.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY
from retry_utils import retry_openai_call

logger = logging.getLogger(__name__)

_CAPTION_MODEL = "gpt-4o-mini"
_CAPTION_PROMPT = (
    "This image was extracted from a clinical/medical document (a research "
    "paper, hospital guideline, or treatment protocol). Write a single, "
    "dense, factual caption (2-4 sentences) describing exactly what it "
    "shows — e.g. a dosage table, a diagnostic decision tree, an anatomical "
    "diagram, a lab scan, a trial outcome chart. Be specific about any "
    "labels, values, or steps visible. This caption is the ONLY text "
    "representation of the image that will ever be searchable, so include "
    "every clinically meaningful detail you can actually see — do not "
    "guess at anything not visible in the image."
)

_EXT_TO_MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}


@retry_openai_call
def _invoke_caption(llm: ChatOpenAI, image_b64: str, mime: str) -> str:
    message = HumanMessage(
        content=[
            {"type": "text", "text": _CAPTION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
        ]
    )
    response = llm.invoke([message])
    return response.content


def caption_image(image_path: str, llm: Optional[ChatOpenAI] = None) -> Optional[str]:
    """Returns a caption string, or None if captioning is unavailable/
    fails for any reason — caller keeps the placeholder text in that case."""
    if not OPENAI_API_KEY:
        return None
    path = Path(image_path)
    ext = path.suffix.lstrip(".").lower()
    mime = _EXT_TO_MIME.get(ext)
    if mime is None or not path.exists():
        return None
    try:
        image_b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        llm = llm or ChatOpenAI(model=_CAPTION_MODEL, temperature=0)
        caption = _invoke_caption(llm, image_b64, mime)
        return caption.strip() or None
    except Exception:
        logger.exception("image_caption_failed", extra={"image_path": str(image_path)})
        return None
