"""In-memory background-task tracker — singleflight + bell feed.

Complements ``jobs.py`` generation tokens: while tokens discard stale
*results*, this tracker discards stale *launches* (two tabs starting the
same audio job) and feeds a header bell (running → ready/failed).

Singleflight: ``claim_job(key)`` returns True for the oldest claimant;
losers return False and should skip launching. ``finish_job`` / ``fail_job``
record terminal state; ``list_jobs`` returns newest-first for the bell.
Entries expire after 2h so the dict never grows unboundedly. Thread-safe.
"""

from __future__ import annotations

import threading
import time
from typing import Any

_lock = threading.Lock()
_tasks: dict[str, dict[str, Any]] = {}


def _prune(now: float | None = None) -> None:
    now = now if now is not None else time.time()
    expired = [k for k, v in _tasks.items() if now - float(v.get("updated_at", 0)) > 7200]
    for k in expired:
        _tasks.pop(k, None)


def claim_job(key: str, label: str = "", notebook_id: int = 0) -> bool:
    """Claim ``key`` for a new job. True = oldest claimant, proceed."""
    now = time.time()
    with _lock:
        _prune(now)
        existing = _tasks.get(key)
        if existing is not None and existing.get("status") == "running":
            return False
        _tasks[key] = {
            "key": key,
            "label": label,
            "notebook_id": notebook_id,
            "status": "running",
            "progress": 0,
            "error": "",
            "created_at": now,
            "updated_at": now,
            "read": False,
        }
        return True


def update_progress(key: str, progress: int, label: str = "") -> None:
    """Update a running job's progress percent (0-100)."""
    with _lock:
        task = _tasks.get(key)
        if task is None:
            return
        task["progress"] = max(0, min(100, int(progress)))
        if label:
            task["label"] = label
        task["updated_at"] = time.time()


def finish_job(key: str, label: str = "") -> None:
    """Mark a job ready (bell badge)."""
    with _lock:
        task = _tasks.get(key)
        if task is None:
            return
        task["status"] = "ready"
        task["progress"] = 100
        if label:
            task["label"] = label
        task["updated_at"] = time.time()
        task["read"] = False


def fail_job(key: str, error: str = "") -> None:
    """Mark a job failed (bell badge, error retained server-side)."""
    with _lock:
        task = _tasks.get(key)
        if task is None:
            return
        task["status"] = "failed"
        task["error"] = error[:300]
        task["updated_at"] = time.time()
        task["read"] = False


def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    """Return newest-first job dicts (copy, capped)."""
    with _lock:
        _prune()
        ordered = sorted(_tasks.values(), key=lambda t: t.get("updated_at", 0), reverse=True)
        return [dict(t) for t in ordered[:limit]]


def unread_count() -> int:
    """Return the number of unread terminal jobs."""
    with _lock:
        return sum(
            1
            for t in _tasks.values()
            if t.get("status") in ("ready", "failed") and not t.get("read")
        )


def mark_read(key: str) -> None:
    """Mark one job read."""
    with _lock:
        task = _tasks.get(key)
        if task is not None:
            task["read"] = True


def mark_all_read() -> None:
    """Mark all terminal jobs read."""
    with _lock:
        for task in _tasks.values():
            if task.get("status") in ("ready", "failed"):
                task["read"] = True


def reset_tracker() -> None:
    """Clear all jobs (used by tests)."""
    with _lock:
        _tasks.clear()
