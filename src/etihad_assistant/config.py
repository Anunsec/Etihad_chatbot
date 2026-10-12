"""Environment-backed settings and project-relative data locations."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and an optional .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_timeout_seconds: float = 45.0
    openai_max_retries: int = 2

    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "etihad-local-mvp"

    jwt_secret: str = "development-only-change-me-32-bytes-minimum"
    access_token_minutes: int = 480
    database_path: Path = Field(default=Path("data/app.db"))
    langgraph_checkpoint_path: Path | None = None
    audit_log_path: Path = Field(default=Path("logs/audit.jsonl"))
    policy_corpus_path: Path = Field(
        default=Path("data/corpus/etihad_hub_policy_library.json")
    )
    rag_backend: Literal["chroma", "bm25"] = "chroma"
    chroma_persist_directory: Path = Field(default=Path("data/chroma"))
    redis_url: str | None = None
    redis_cache_ttl_seconds: int = Field(default=604800, ge=60)
    redis_require_tls: bool = True
    chat_history_messages: int = Field(default=8, ge=0, le=20)

    demo_user_username: str = "user"
    demo_user_password: str = "user-change-me"
    demo_admin_username: str = "admin"
    demo_admin_password: str = "admin-change-me"

    aviationstack_api_key: str | None = None
    aviationstack_timeout_seconds: float = 10.0
    enable_live_flight_data: bool = True

    def absolute_path(self, value: Path) -> Path:
        """Resolve relative data paths from the repository root, not the shell cwd."""
        return value if value.is_absolute() else PROJECT_ROOT / value

    @property
    def db_file(self) -> Path:
        return self.absolute_path(self.database_path)

    @property
    def checkpoint_db_file(self) -> Path:
        """Keep LangGraph checkpoints separate from application tables by default."""
        if self.langgraph_checkpoint_path is not None:
            return self.absolute_path(self.langgraph_checkpoint_path)
        database = self.db_file
        return database.with_name(f"{database.stem}.checkpoints.db")

    @property
    def audit_file(self) -> Path:
        return self.absolute_path(self.audit_log_path)

    @property
    def corpus_file(self) -> Path:
        return self.absolute_path(self.policy_corpus_path)

    @property
    def chroma_dir(self) -> Path:
        return self.absolute_path(self.chroma_persist_directory)


@lru_cache
def get_settings() -> Settings:
    """Reuse one parsed settings object per process."""
    return Settings()
