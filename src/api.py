"""HTTP API over the RAG pipeline.

    uvicorn api:app --reload --app-dir src

Then open http://localhost:8000/docs for the generated schema and a query form.
"""

import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import APIError, APITimeoutError, RateLimitError
from pydantic import BaseModel, Field

import config
from generate import answer, answer_stream
from retrieve import get_collection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

MAX_QUESTION_CHARS = 1000


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=MAX_QUESTION_CHARS,
        description="Natural-language question about the indexed documents.",
    )
    k: int = Field(
        default=config.TOP_K,
        ge=1,
        le=20,
        description="Number of chunks to retrieve. Bounded — an unbounded k is a "
                    "way to run up someone else's token bill.",
    )


class SourceOut(BaseModel):
    n: int = Field(..., description="Citation marker used in the answer text.")
    source: str
    page: int
    distance: float = Field(..., description="Cosine distance. Lower is closer.")
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    latency_ms: int
    cost_usd: float


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
    embed_model: str
    chat_model: str


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the Chroma collection once, at startup.

    retrieve.get_collection is lru_cached, so calling it here warms the cache
    and every later request reuses the handle. Opening it per request would
    add tens of milliseconds to every call and re-read the index from disk —
    a small mistake that is very visible under any concurrency.
    """
    try:
        count = get_collection().count()
        log.info("collection ready: %d chunks", count)
        if count == 0:
            log.warning("index is EMPTY — run python src/ingest.py")
    except Exception as exc:
        # Do not crash the process: /health should be able to report the
        # problem rather than the container just dying at boot.
        log.error("could not open collection: %s", exc)

    yield
    log.info("shutting down")


app = FastAPI(
    title="Domain RAG Assistant",
    description="Question answering over EU regulatory documents, with citations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten before any deployment that matters
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness plus enough state to diagnose a broken deployment.

    Reports the model names because the most confusing possible failure is an
    index built with one embedding model being queried with another — the
    results are quietly wrong rather than erroring.
    """
    try:
        count = get_collection().count()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"index unavailable: {exc}")

    return HealthResponse(
        status="ok" if count else "empty_index",
        chunks_indexed=count,
        embed_model=config.EMBED_MODEL,
        chat_model=config.CHAT_MODEL,
    )


def _guard_index() -> None:
    try:
        if get_collection().count() == 0:
            raise HTTPException(
                status_code=503,
                detail="No documents indexed. Run ingest.py before querying.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"index unavailable: {exc}")


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    _guard_index()
    started = time.perf_counter()

    try:
        result = answer(request.question, k=request.k)
    except APITimeoutError:
        raise HTTPException(status_code=504, detail="Upstream model timed out.")
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limited upstream. Retry shortly.")
    except APIError as exc:
        # Log the detail, return something generic — upstream errors can carry
        # internal information that should not reach a public client.
        log.error("upstream API error: %s", exc)
        raise HTTPException(status_code=502, detail="Upstream model error.")

    latency_ms = int((time.perf_counter() - started) * 1000)
    log.info("query k=%d %dms $%.5f | %s",
             request.k, latency_ms, result.cost_usd, request.question[:80])

    return QueryResponse(
        answer=result.text,
        sources=[
            SourceOut(n=s.n, source=s.source, page=s.page,
                      distance=s.distance, text=s.text)
            for s in result.sources
        ],
        latency_ms=latency_ms,
        cost_usd=round(result.cost_usd, 6),
    )


@app.post("/query/stream")
def query_stream(request: QueryRequest) -> StreamingResponse:
    """Server-sent events: one `sources` event, then `token` events, then `done`.

    Sources are sent first so the UI can render citations while the answer is
    still being written.
    """
    _guard_index()

    def events():
        started = time.perf_counter()
        try:
            for sources, token in answer_stream(request.question, k=request.k):
                if sources is not None:
                    payload = [
                        {"n": s.n, "source": s.source, "page": s.page,
                         "distance": round(s.distance, 4), "text": s.text}
                        for s in sources
                    ]
                    yield f"event: sources\ndata: {json.dumps(payload)}\n\n"
                if token:
                    yield f"event: token\ndata: {json.dumps(token)}\n\n"

            elapsed = int((time.perf_counter() - started) * 1000)
            yield f"event: done\ndata: {json.dumps({'latency_ms': elapsed})}\n\n"

        except Exception as exc:
            # The response has already started, so a normal HTTP error code is
            # no longer available — the error has to travel inside the stream.
            log.error("stream failed: %s", exc)
            yield f"event: error\ndata: {json.dumps({'detail': 'generation failed'})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )