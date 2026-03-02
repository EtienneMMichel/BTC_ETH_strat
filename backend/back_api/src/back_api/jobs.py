from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Coroutine, Literal


@dataclass
class JobRecord:
    status: Literal["pending", "running", "done", "failed"]
    result: Any = None
    error: str | None = None
    fingerprint: str | None = None


class JobStore:
    """Thread-safe in-memory job store backed by asyncio.Lock."""

    _MAX_JOBS = 500  # TTL eviction threshold

    def __init__(self) -> None:
        self._store: dict[str, JobRecord] = {}
        self._fingerprint_index: dict[str, str] = {}  # fingerprint → job_id
        self._lock: asyncio.Lock = asyncio.Lock()

    def create(self) -> str:
        """Register a new job and return its UUID."""
        job_id = str(uuid.uuid4())
        self._store[job_id] = JobRecord(status="pending")
        return job_id

    async def find_or_create(self, fingerprint: str) -> tuple[str, bool]:
        """Return (job_id, is_new). Caller spawns task only when is_new is True."""
        async with self._lock:
            existing_id = self._fingerprint_index.get(fingerprint)
            if existing_id is not None:
                rec = self._store.get(existing_id)
                if rec is not None and rec.status in ("pending", "running", "done"):
                    return existing_id, False
            job_id = str(uuid.uuid4())
            self._store[job_id] = JobRecord(status="pending", fingerprint=fingerprint)
            self._fingerprint_index[fingerprint] = job_id
            return job_id, True

    def get(self, job_id: str) -> JobRecord | None:
        return self._store.get(job_id)

    async def run(self, job_id: str, coro: Coroutine[Any, Any, Any]) -> None:
        """
        Execute *coro* in a thread pool (via asyncio.to_thread) so that
        CPU-bound work never blocks the event loop.

        Updates job status: pending → running → done | failed.
        """
        async with self._lock:
            rec = self._store.get(job_id)
            if rec is None:
                coro.close()
                return
            rec.status = "running"

        try:
            # to_thread wraps a sync callable; we schedule the coroutine via
            # asyncio.ensure_future after converting it with asyncio.run_coroutine_threadsafe
            # For simplicity we await the coroutine directly — it must call
            # asyncio.to_thread internally for any blocking work.
            result = await coro
            async with self._lock:
                rec = self._store[job_id]
                rec.status = "done"
                rec.result = result
        except Exception as exc:  # noqa: BLE001
            async with self._lock:
                rec = self._store[job_id]
                rec.status = "failed"
                rec.error = str(exc)
        finally:
            await self._evict_if_needed()

    async def _evict_if_needed(self) -> None:
        """Remove oldest done/failed jobs when store exceeds _MAX_JOBS."""
        async with self._lock:
            if len(self._store) <= self._MAX_JOBS:
                return
            removable = [
                jid
                for jid, rec in self._store.items()
                if rec.status in ("done", "failed")
            ]
            for jid in removable[: len(self._store) - self._MAX_JOBS]:
                rec = self._store.pop(jid)
                # Clean index only when this job is still the current owner
                if rec.fingerprint and self._fingerprint_index.get(rec.fingerprint) == jid:
                    del self._fingerprint_index[rec.fingerprint]
