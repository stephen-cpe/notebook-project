"""Resilient execution — retry with backoff, pause-don't-fail on rate limits.

Ported from pdf2md ``resilience.py``: transient cloud failures (HTTP 429
with ``Retry-After``, 5xx, timeouts, connection errors) are retried with
exponential backoff; a ``PauseJob`` is raised when the service asks us to
back off longer than we are willing to wait, so the caller can mark the
job paused/resumable instead of failed. Client errors (4xx) propagate
immediately — retrying a bad key or bad payload never heals.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class PauseJob(Exception):  # noqa: N818
    """Raised when the job should pause (resumable) instead of failing."""

    def __init__(self, message: str, retry_after: float = 0.0) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _retry_after_seconds(exc: Exception, default: float = 0.0) -> float:
    """Best-effort extraction of a Retry-After delay from an exception."""
    m = re.search(r"retry-?after[:\s]+(\d+)", str(exc).lower())
    if m:
        try:
            return float(m.group(1))
        except (TypeError, ValueError):
            return default
    if "429" in str(exc):
        return default
    return 0.0


def resilient(
    fn: Callable[[], Any],
    *,
    max_attempts: int = 3,
    base_delay: float = 2.0,
    max_retry_after: float = 120.0,
) -> Any:  # noqa: ANN401
    """Call ``fn`` with exponential backoff on transient failures.

    - 429 / 5xx / timeout / connection errors: retry up to ``max_attempts``.
    - 429 with ``Retry-After`` > ``max_retry_after``: raise ``PauseJob``.
    - Other exceptions (incl. 4xx): propagate immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except PauseJob:
            raise
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc).lower()
            is_rate_limit = "429" in msg or "rate-limit" in msg or "rate limit" in msg
            is_transient = is_rate_limit or any(
                s in msg for s in ("timeout", "timed out", "connection", "500", "502", "503", "504")
            )
            if not is_transient:
                raise
            retry_after = _retry_after_seconds(exc)
            if retry_after > max_retry_after:
                raise PauseJob(
                    f"Service asked to back off for {retry_after:.0f}s; pausing job.",
                    retry_after=retry_after,
                ) from exc
            if attempt >= max_attempts:
                break
            delay = max(base_delay * (2 ** (attempt - 1)), retry_after)
            logger.warning(
                "Transient failure (attempt %d/%d): %s; retrying in %.1fs",
                attempt,
                max_attempts,
                exc,
                delay,
            )
            time.sleep(delay)
    raise last_exc if last_exc is not None else RuntimeError("resilient() failed without exception")
