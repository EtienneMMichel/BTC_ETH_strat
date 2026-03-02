from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py lives at backend/back_api/src/back_api/config.py
# parents[4] is the repo root regardless of the working directory.
_REPO_ROOT = Path(__file__).parents[4]
_DEFAULT_DATA_DIR = _REPO_ROOT / "backend" / "data" / "raw"
_DEFAULT_SAVED_RUNS_DIR = _REPO_ROOT / "backend" / "data" / "saved_runs"


class Settings(BaseSettings):
    data_dir: Path = _DEFAULT_DATA_DIR
    saved_runs_dir: Path = _DEFAULT_SAVED_RUNS_DIR
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    model_config = SettingsConfigDict(env_prefix="BACK_API_")


@lru_cache
def get_settings() -> Settings:
    return Settings()
