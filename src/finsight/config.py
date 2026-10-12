"""Central, typed configuration. Every knob is an environment variable (12-factor)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderName = Literal["offline", "openai", "azure_openai", "anthropic", "openai_compatible"]
EmbeddingProviderName = Literal["hash", "fastembed", "openai", "azure_openai"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- runtime --------------------------------------------------------
    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    data_dir: Path = Path("./data")

    # ---- security / access control --------------------------------------
    # Comma separated "key:role" pairs, roles = viewer | admin. Empty => auth disabled (dev only).
    api_keys: str = ""
    rate_limit_per_minute: int = 60
    cors_origins: str = "*"

    # ---- LLM --------------------------------------------------------------
    llm_provider: LLMProviderName = "offline"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1200
    llm_timeout_s: float = 60.0
    openai_api_key: str | None = None
    openai_base_url: str | None = None  # openai_compatible: Ollama, vLLM, Gemini, Groq...
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str = "2024-10-21"
    azure_openai_deployment: str | None = None
    anthropic_api_key: str | None = None

    # ---- embeddings / retrieval -------------------------------------------
    embedding_provider: EmbeddingProviderName = "hash"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker: Literal["none", "cross-encoder"] = "none"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    qdrant_url: str | None = None  # http://qdrant:6333 ; None => embedded mode
    # on-disk embedded mode; ":memory:" => in-memory; default (no url, no path) => <data_dir>/qdrant
    qdrant_path: str | None = None
    qdrant_api_key: str | None = None
    collection: str = "sec_filings"
    retrieval_candidates: int = 30
    rerank_candidates: int = 20
    top_k: int = 6
    chunk_tokens: int = 450
    chunk_overlap_tokens: int = 60

    # ---- guardrails ----------------------------------------------------------
    guardrail_mode: Literal["block", "flag"] = "block"
    injection_threshold: float = 0.6
    min_groundedness: float = 0.55
    mask_pii: bool = True

    # ---- prompts -------------------------------------------------------------
    rag_prompt: str = "rag_answer@v2"
    agent_prompt: str = "agent_system@v1"

    # ---- SEC EDGAR -------------------------------------------------------------
    # SEC requires a descriptive User-Agent with a contact e-mail.
    sec_user_agent: str = "CATsFi FinSight research contact@example.com"
    sec_rps: float = 8.0

    # ---- caching -----------------------------------------------------------------
    answer_cache_ttl_s: int = 600
    answer_cache_size: int = 512

    # ---- observability -------------------------------------------------------------
    otel_service_name: str = "catsfi-finsight"
    otel_exporter_otlp_endpoint: str | None = None
    otel_console: bool = False

    agent_max_steps: int = 6

    # Comma separated tickers indexed in the background at startup when the index is empty (demo deploys).
    seed_tickers: str = ""
    seed_forms: str = "10-K"

    @field_validator("data_dir")
    @classmethod
    def _mk(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def api_key_roles(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for pair in filter(None, (p.strip() for p in self.api_keys.split(","))):
            key, _, role = pair.partition(":")
            out[key] = role or "viewer"
        return out

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key_roles)


@lru_cache
def get_settings() -> Settings:
    return Settings()


__all__ = ["Settings", "get_settings"]
