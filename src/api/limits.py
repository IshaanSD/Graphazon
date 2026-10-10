from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone

from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

DAILY_LLM_CALL_LIMIT = int(os.getenv("DAILY_LLM_CALL_LIMIT", "30"))

_lock = threading.Lock()
_counter_date: str | None = None
_counter_count = 0


class DailyCapExceeded(Exception):
    def __init__(self, limit: int):
        self.limit = limit
        super().__init__(f"Daily LLM call limit of {limit} reached")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def check_and_increment_daily_cap() -> None:
    """Raises DailyCapExceeded if today's live-LLM-call budget is used up."""
    global _counter_date, _counter_count
    today = _today()
    with _lock:
        if _counter_date != today:
            _counter_date = today
            _counter_count = 0
        if _counter_count >= DAILY_LLM_CALL_LIMIT:
            raise DailyCapExceeded(DAILY_LLM_CALL_LIMIT)
        _counter_count += 1
        logger.info("Daily LLM call count: %d/%d", _counter_count, DAILY_LLM_CALL_LIMIT)
