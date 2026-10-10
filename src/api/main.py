import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.api.cache import cached_understand, peek_cached
from src.api.examples import get_examples
from src.api.limits import DailyCapExceeded, check_and_increment_daily_cap, limiter
from src.api.schemas import SearchRequest
from src.data.schemas import QueryUnderstandingOutput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "http://localhost:5173")

app = FastAPI(title="Graphazon Query Understanding API")
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests - try an example query below, or wait a minute."},
    )


@app.exception_handler(DailyCapExceeded)
async def daily_cap_handler(request: Request, exc: DailyCapExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "detail": (
                f"This demo has hit its daily limit of {exc.limit} live searches. "
                "Try an example query below, or come back tomorrow."
            )
        },
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/examples", response_model=list[QueryUnderstandingOutput])
def examples() -> list[QueryUnderstandingOutput]:
    return get_examples()


@app.post("/search", response_model=QueryUnderstandingOutput)
@limiter.limit("3/minute")
def search(request: Request, body: SearchRequest) -> QueryUnderstandingOutput:
    cached = peek_cached(body.query, body.locale)
    if cached is not None:
        logger.info("search cache hit (query=%r, locale=%s)", body.query, body.locale)
        return cached

    # Only an actual LLM call counts against the daily cap - cache hits are free.
    check_and_increment_daily_cap()
    result, _ = cached_understand(body.query, body.locale)
    logger.info("search cache miss, called LLM (query=%r, locale=%s)", body.query, body.locale)
    return result
