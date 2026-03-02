# CLAUDE.md — `backend/back_api/` subagent briefing

## Role

You are building and maintaining the **FastAPI layer** that exposes `backend/core` (the quant
library) to external consumers (frontend, notebooks, CI pipelines).

All heavy computation (walk-forward backtest, GARCH fits, evaluation metrics) is CPU-bound and
dispatched via `asyncio.to_thread()` so the event loop is never blocked.  Jobs are tracked in an
in-memory `JobStore`; clients submit a job and poll for results.

---

## Architecture overview

```
FastAPI app (main.py)
├── GET  /health                          ← liveness check
├── POST /backtest/run  →  {job_id}       ← submit backtest job
├── GET  /backtest/{job_id}               ← poll status / fetch result
├── POST /evaluation/vol-comparison       ← submit vol comparison job
├── POST /evaluation/var-test             ← submit VaR test job
├── POST /evaluation/dm-test              ← submit DM test job
└── GET  /evaluation/{job_id}             ← poll status / fetch result
```

**Single-worker model** — job store is in-memory (not shared across processes).
Scale horizontally only with an external store (Redis / Postgres) if needed.

---

## Module layout

```
backend/back_api/
├── pyproject.toml            # Poetry project; core is a path dep
├── CLAUDE.md                 # this file
└── src/back_api/
    ├── __init__.py
    ├── main.py               # FastAPI app + lifespan
    ├── config.py             # Settings via pydantic-settings + BACK_API_ env vars
    ├── jobs.py               # async JobStore (UUID → JobRecord)
    ├── schemas/
    │   ├── backtest.py       # BacktestRequest, BacktestResultSchema, JobStatusResponse
    │   └── evaluation.py     # VolComparisonRequest, VarTestRequest, DMTestRequest, …
    └── routers/
        ├── backtest.py       # POST /run, GET /{job_id}
        └── evaluation.py     # POST /vol-comparison, /var-test, /dm-test, GET /{job_id}
```

---

## Core library dependency

The `core` package lives at `../core` and is installed as a **path dependency** (develop mode):

```toml
[tool.poetry.dependencies]
core = {path = "../core", develop = true}
```

Import pattern:
```python
from core.backtest.data.loader import load_assets
from core.backtest.engine.runner import WalkForwardRunner
from core.evaluation.reports.compare import comparison_table, diebold_mariano
from core.evaluation.metrics.risk import kupiec_test, christoffersen_test
from core.strats.momentum.strategy import MomentumStrategy
from core.strats.mean_reversion.strategy import MeanReversionStrategy
from core.strats.orchestrator.regime import OrchestratorStrategy
```

**Always use lazy imports inside sync worker functions** (not at module top level) so that heavy
model dependencies (arch, filterpy, statsmodels) are not loaded until a request triggers them.

---

## Job queue pattern

Every endpoint that does >100 ms of work must use the job queue:

```python
@router.post("/some-endpoint")
async def some_endpoint(req: SomeRequest, request: Request) -> dict[str, str]:
    store: JobStore = request.app.state.job_store
    job_id = store.create()
    asyncio.create_task(
        store.run(job_id, asyncio.to_thread(_run_sync_worker, req))
    )
    return {"job_id": job_id}
```

The sync worker (`_run_sync_worker`) must be a **plain function** (not async) — it runs in a
thread pool.  Never call `asyncio.run()` or create new event loops inside a worker.

---

## Configuration (env vars)

| Variable | Default | Description |
|---|---|---|
| `BACK_API_DATA_DIR` | `backend/data/raw` | Root directory of raw parquet data (contains BTCUSDT/, ETHUSDT/, … sub-dirs) |
| `BACK_API_HOST` | `0.0.0.0` | Uvicorn bind host |
| `BACK_API_PORT` | `8000` | Uvicorn bind port |
| `BACK_API_WORKERS` | `1` | Number of uvicorn workers (keep 1 for in-memory store) |

---

## Server optimisation rules

1. **Never block the event loop.** Wrap all CPU work in `asyncio.to_thread()`.
2. **Cache `Settings`** with `@lru_cache` — load once, reuse on every request.
3. **Load data once per job**, not once per request sub-step.
4. **Job store is bounded** — eviction kicks in at 500 entries (done/failed jobs removed first).
5. **GZip middleware** compresses large JSON responses (equity curves, signal tables).
6. **Single worker** — `BACK_API_WORKERS=1` because the in-memory job store is not
   process-safe.  If you need multiple workers, replace `JobStore` with a Redis-backed store.

---

## Adding new endpoints

1. Add a Pydantic model to the appropriate `schemas/` file.
2. Write a `_run_*_sync()` plain function with lazy imports from `core`.
3. Wire it up in the router with `asyncio.create_task(store.run(...))`.
4. Register the router in `main.py` if it is a new router file.

---

## Commands

```bash
cd backend/back_api

# Install (first time or after pyproject.toml changes)
python -m poetry install

# Run development server (auto-reload)
python -m poetry run uvicorn back_api.main:app --reload --host 0.0.0.0 --port 8000

# Run production server (single worker, in-memory store)
python -m poetry run uvicorn back_api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

## Quick smoke test

```bash
curl http://localhost:8000/health
# → {"status":"ok"}

curl -X POST http://localhost:8000/evaluation/dm-test \
  -H "Content-Type: application/json" \
  -d '{"forecast_a":[1,2,3],"forecast_b":[1.1,2.1,3.1],"realised":[1,2,3],"index":["2024-01-01","2024-01-02","2024-01-03"]}'
# → {"job_id":"<uuid>"}

curl http://localhost:8000/evaluation/<uuid>
# → {"job_id":"...","status":"done","result":{"dm_stat":...,"p_value":...},...}
```
