from __future__ import annotations

import logging
import threading
from typing import List

from src.data.schemas import QueryUnderstandingOutput
from src.query_understanding.interface import understand_query

logger = logging.getLogger(__name__)

# Real ESCI dataset queries, spanning the four query_understanding intents.
EXAMPLE_QUERIES = [
    {"query_id": "ex-iphone-case", "query": "apple case for iphone x", "locale": "us"},
    {"query_id": "ex-cabinet-lock", "query": "alternative locks for cabinet", "locale": "us"},
    {"query_id": "ex-f150-part", "query": "05 f150 rack and pinion without inner tie rod", "locale": "us"},
    {"query_id": "ex-hip-bag", "query": "hip bag", "locale": "us"},
]

_lock = threading.Lock()
_example_cache: dict[str, QueryUnderstandingOutput] = {}


def get_examples() -> List[QueryUnderstandingOutput]:
    """Computes each example's real result once per container lifetime, then serves it
    from memory. No request ever triggers more than one LLM call per example, and repeat
    visitors cost nothing."""
    results = []
    for ex in EXAMPLE_QUERIES:
        with _lock:
            cached = _example_cache.get(ex["query_id"])
        if cached is None:
            logger.info("Warming example cache for query_id=%s", ex["query_id"])
            cached = understand_query(**ex)
            with _lock:
                _example_cache[ex["query_id"]] = cached
        results.append(cached)
    return results
