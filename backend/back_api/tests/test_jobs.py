"""Unit tests for JobStore: create, get, find_or_create, run, eviction."""
from __future__ import annotations

import pytest

from back_api.jobs import JobStore


@pytest.fixture
def store() -> JobStore:
    return JobStore()


# ---------------------------------------------------------------------------
# create / get
# ---------------------------------------------------------------------------

class TestCreateGet:
    def test_create_registers_pending(self, store):
        job_id = store.create()
        rec = store.get(job_id)
        assert rec is not None
        assert rec.status == "pending"
        assert rec.fingerprint is None

    def test_get_unknown_returns_none(self, store):
        assert store.get("does-not-exist") is None

    def test_get_returns_correct_record(self, store):
        id_a = store.create()
        id_b = store.create()
        assert store.get(id_a).status == "pending"
        assert store.get(id_b).status == "pending"
        assert id_a != id_b


# ---------------------------------------------------------------------------
# find_or_create
# ---------------------------------------------------------------------------

class TestFindOrCreate:
    async def test_new_fingerprint_is_new(self, store):
        job_id, is_new = await store.find_or_create("fp-A")
        assert is_new is True
        assert store.get(job_id) is not None
        assert store.get(job_id).fingerprint == "fp-A"

    async def test_pending_job_reused(self, store):
        id1, _ = await store.find_or_create("fp-B")
        id2, is_new = await store.find_or_create("fp-B")
        assert is_new is False
        assert id1 == id2

    async def test_running_job_reused(self, store):
        id1, _ = await store.find_or_create("fp-C")
        store.get(id1).status = "running"
        id2, is_new = await store.find_or_create("fp-C")
        assert is_new is False
        assert id1 == id2

    async def test_done_job_reused(self, store):
        id1, _ = await store.find_or_create("fp-D")
        store.get(id1).status = "done"
        id2, is_new = await store.find_or_create("fp-D")
        assert is_new is False
        assert id1 == id2

    async def test_failed_job_creates_new(self, store):
        id1, _ = await store.find_or_create("fp-E")
        store.get(id1).status = "failed"
        id2, is_new = await store.find_or_create("fp-E")
        assert is_new is True
        assert id1 != id2
        # Index now points to the new job
        assert store._fingerprint_index.get("fp-E") == id2

    async def test_different_fingerprints_independent(self, store):
        id1, new1 = await store.find_or_create("fp-X")
        id2, new2 = await store.find_or_create("fp-Y")
        assert id1 != id2
        assert new1 is True
        assert new2 is True


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

class TestRun:
    async def test_success_transitions_to_done(self, store):
        job_id = store.create()

        async def coro():
            return {"answer": 42}

        await store.run(job_id, coro())
        rec = store.get(job_id)
        assert rec.status == "done"
        assert rec.result == {"answer": 42}
        assert rec.error is None

    async def test_exception_transitions_to_failed(self, store):
        job_id = store.create()

        async def failing():
            raise ValueError("boom")

        await store.run(job_id, failing())
        rec = store.get(job_id)
        assert rec.status == "failed"
        assert rec.result is None
        assert "boom" in rec.error

    async def test_unknown_job_is_noop(self, store):
        async def coro():
            return 1

        # Must not raise
        await store.run("nonexistent", coro())


# ---------------------------------------------------------------------------
# eviction
# ---------------------------------------------------------------------------

class TestEviction:
    async def test_eviction_removes_done_jobs_when_over_limit(self, store):
        store._MAX_JOBS = 3
        for _ in range(4):
            jid = store.create()
            store.get(jid).status = "done"

        extra = store.create()

        async def coro():
            return 1

        await store.run(extra, coro())
        assert len(store._store) <= 3

    async def test_eviction_cleans_fingerprint_index(self, store):
        store._MAX_JOBS = 2
        fp = "evict-me"
        job_id, _ = await store.find_or_create(fp)
        store.get(job_id).status = "done"

        jid2 = store.create()
        store.get(jid2).status = "done"

        extra = store.create()

        async def coro():
            return 1

        await store.run(extra, coro())

        # The fingerprinted job was evicted → index entry removed
        if job_id not in store._store:
            assert store._fingerprint_index.get(fp) != job_id

    async def test_eviction_preserves_index_on_retry(self, store):
        """When a retry replaces a fingerprint's index entry, evicting the
        first (failed) job must NOT delete the index entry for the retry."""
        store._MAX_JOBS = 2
        fp = "retry-fp"

        id1, _ = await store.find_or_create(fp)
        store.get(id1).status = "failed"

        # Retry → index now points to id2
        id2, _ = await store.find_or_create(fp)
        assert store._fingerprint_index.get(fp) == id2

        # Fill store to trigger eviction of id1 (failed)
        jid3 = store.create()
        store.get(jid3).status = "done"

        extra = store.create()

        async def coro():
            return 1

        await store.run(extra, coro())

        # id1 may be evicted, but the index for fp must still point to id2
        assert store._fingerprint_index.get(fp) == id2

    async def test_pending_jobs_not_evicted(self, store):
        store._MAX_JOBS = 2
        # Fill with pending jobs (should not be evicted)
        id_p1 = store.create()
        id_p2 = store.create()
        id_done = store.create()
        store.get(id_done).status = "done"

        extra = store.create()

        async def coro():
            return 1

        await store.run(extra, coro())

        # Pending jobs must survive
        assert store.get(id_p1) is not None
        assert store.get(id_p2) is not None
