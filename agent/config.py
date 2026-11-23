"""Central configuration. Values come from environment variables (or a .env file)
so the same code runs in dev, eval replay, and the demo without edits."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _env(name: str, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


@dataclass
class Config:
    # --- LLM ---------------------------------------------------------------
    model: str = field(default_factory=lambda: _env("AGENT_MODEL", "claude-sonnet-5-5"))
    max_tokens: int = field(default_factory=lambda: _env("AGENT_MAX_TOKENS", 2048))
    # USD per million tokens. Fill in from the provider's current pricing page.
    price_in_per_mtok: float = field(default_factory=lambda: _env("PRICE_IN_PER_MTOK", 0.0))
    price_out_per_mtok: float = field(default_factory=lambda: _env("PRICE_OUT_PER_MTOK", 0.0))

    # --- Loop limits (M2 design: max 8 tool calls, 90 s per question) -------
    max_tool_calls: int = field(default_factory=lambda: _env("MAX_TOOL_CALLS", 8))
    max_seconds: float = field(default_factory=lambda: _env("MAX_SECONDS", 90.0))
    tool_failures_before_disable: int = 2  # circuit breaker per tool

    # --- Router: retries / timeouts / cache --------------------------------
    max_attempts: int = field(default_factory=lambda: _env("MAX_ATTEMPTS", 3))
    backoff_base_s: float = field(default_factory=lambda: _env("BACKOFF_BASE_S", 1.0))
    web_timeout_s: float = 20.0
    retrieval_timeout_s: float = 15.0
    sandbox_timeout_s: float = 10.0
    # off | readwrite | replay   (replay = cache only, used for eval runs)
    cache_mode: str = field(default_factory=lambda: _env("CACHE_MODE", "readwrite"))
    cache_dir: Path = field(default_factory=lambda: Path(_env("CACHE_DIR", ".cache/tools")))

    # --- Web search ---------------------------------------------------------
    tavily_api_key: str = field(default_factory=lambda: _env("TAVILY_API_KEY", ""))
    brave_api_key: str = field(default_factory=lambda: _env("BRAVE_API_KEY", ""))  # fallback
    web_max_results: int = 5
    web_fetch_pages: bool = field(default_factory=lambda: _env("WEB_FETCH_PAGES", True))

    # --- Paper retrieval ----------------------------------------------------
    index_dir: Path = field(default_factory=lambda: Path(_env("INDEX_DIR", "data/index")))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    retrieval_top_k: int = 5
    retrieval_candidates: int = 20  # per retriever before fusion
    min_relevance: float = field(default_factory=lambda: _env("MIN_RELEVANCE", 0.35))

    # --- Sandbox ------------------------------------------------------------
    # docker (isolated, required for demos/eval) | local (dev only, NOT a security boundary)
    sandbox_mode: str = field(default_factory=lambda: _env("SANDBOX_MODE", "docker"))
    sandbox_image: str = field(default_factory=lambda: _env("SANDBOX_IMAGE", "research-agent-sandbox"))
    sandbox_memory: str = "256m"

    # --- Validator ----------------------------------------------------------
    max_evidence_chars: int = 2500
    conflict_tolerance: float = 0.05  # numbers from two sources differing >5% = conflict

    # --- Output -------------------------------------------------------------
    trace_dir: Path = field(default_factory=lambda: Path(_env("TRACE_DIR", "traces")))

    def cost_usd(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.price_in_per_mtok + tokens_out * self.price_out_per_mtok) / 1e6
