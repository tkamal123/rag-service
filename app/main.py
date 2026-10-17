"""FastAPI service: /health and /query."""
import logging
import secrets
import time
from collections import defaultdict, deque
from functools import lru_cache

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from app.config import get_settings
from app.guardrails import GuardrailViolation, validate_input
from app.rag import RAGPipeline, VectorStore, make_llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("api")

app = FastAPI(title="Production RAG Service", version="1.0.0")


# ------------------------------------------------------------------ schemas
class QueryRequest(BaseModel):
    question: str = Field(..., examples=["How long do customers have to request a refund?"])


class SourceOut(BaseModel):
    source: str
    chunk_id: str
    distance: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    refused: bool
    reason: str | None = None
    pii_redacted: list[str] = []
    sources: list[SourceOut]
    latency_ms: int


# ------------------------------------------------------------------ dependencies
@lru_cache
def get_pipeline() -> RAGPipeline:
    settings = get_settings()
    return RAGPipeline(VectorStore(settings), make_llm(settings), settings)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = get_settings().api_key
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key.")


_hits: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request) -> None:
    """Sliding-window limiter, per client IP, per instance (use Redis if you scale out)."""
    limit = get_settings().rate_limit_per_min
    client = request.client.host if request.client else "unknown"
    now, window = time.monotonic(), _hits[client]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")
    window.append(now)


# ------------------------------------------------------------------ routes
@app.get("/health")
def health(pipeline: RAGPipeline = Depends(get_pipeline)) -> dict:
    llm = pipeline.llm
    return {
        "status": "ok",
        "indexed_chunks": pipeline.store.count(),
        "llm_provider": pipeline.settings.llm_provider,
        "llm_model": pipeline.settings.llm_model,
        # False means the API key env var for the chosen provider is not set.
        "llm_configured": getattr(llm, "configured", True),
    }


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key), Depends(rate_limit)])
def query(body: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> QueryResponse:
    settings = get_settings()
    try:
        question = validate_input(body.question, settings.max_input_chars)
    except GuardrailViolation as exc:
        log.warning("blocked request: %s", exc.reason)
        raise HTTPException(status_code=400, detail=exc.reason)

    start = time.perf_counter()
    try:
        result = pipeline.answer(question)
    except Exception:  # LLM/network failure: don't leak internals to the caller
        log.exception("pipeline failure")
        raise HTTPException(status_code=502, detail="Upstream model error. Please retry.")
    latency = int((time.perf_counter() - start) * 1000)

    log.info("query ok refused=%s sources=%d latency_ms=%d", result.refused, len(result.sources), latency)
    return QueryResponse(
        answer=result.answer,
        refused=result.refused,
        reason=result.reason,
        pii_redacted=result.pii_redacted,
        sources=[SourceOut(**vars(s)) for s in result.sources],
        latency_ms=latency,
    )
