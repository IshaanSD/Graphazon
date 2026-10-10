from __future__ import annotations

import hashlib
import logging
import threading
from typing import Optional, Tuple

from src.data.schemas import QueryUnderstandingOutput
from src.query_understanding.interface import understand_query

logger = logging.getLogger(__name__)

_MAX_ENTRIES = 256
_lock = threading.Lock()
_cache: dict[tuple[str, str], QueryUnderstandingOutput] = {}


def normalize_query(query: str) -> str:
    return " ".join(query.strip().lower().split())


def peek_cached(query: str, locale: str) -> Optional[QueryUnderstandingOutput]:
    """Returns the cached result if present, without ever calling the LLM."""
    key = (normalize_query(query), locale)
    with _lock:
        return _cache.get(key)


def cached_understand(query: str, locale: str) -> Tuple[QueryUnderstandingOutput, bool]:
    """Returns (result, was_cache_hit). Calls the LLM only on a cache miss."""
    key = (normalize_query(query), locale)
    with _lock:
        cached = _cache.get(key)
    if cached is not None:
        return cached, True

    query_id = hashlib.sha1(f"{key[0]}|{key[1]}".encode()).hexdigest()[:12]
    logger.info("Cache miss, calling LLM (query_id=%s, query=%r, locale=%s)", query_id, key[0], key[1])
    result = understand_query(query_id=query_id, query=key[0], locale=key[1])

    with _lock:
        if len(_cache) >= _MAX_ENTRIES:
            _cache.pop(next(iter(_cache)))
        _cache[key] = result
    return result, False
