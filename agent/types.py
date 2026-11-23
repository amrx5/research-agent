"""Shared data types passed between router, validator, executor and synthesizer."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    INVALID_ARGS = "INVALID_ARGS"
    DUPLICATE_CALL = "DUPLICATE_CALL"
    TOOL_DISABLED = "TOOL_DISABLED"
    STEP_LIMIT = "STEP_LIMIT"
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    HTTP_ERROR = "HTTP_ERROR"
    CACHE_MISS = "CACHE_MISS"
    SCHEMA_ERROR = "SCHEMA_ERROR"
    EMPTY_RESULT = "EMPTY_RESULT"
    LOW_RELEVANCE = "LOW_RELEVANCE"
    CODE_ERROR = "CODE_ERROR"
    IMPLAUSIBLE_VALUE = "IMPLAUSIBLE_VALUE"
    INTERNAL = "INTERNAL"


# What the executor tells the LLM to do after each kind of failure.
RECOVERY_HINTS: dict[ErrorCode, str] = {
    ErrorCode.UNKNOWN_TOOL: "Use one of the listed tools.",
    ErrorCode.INVALID_ARGS: "Fix the arguments to match the tool schema and call again.",
    ErrorCode.DUPLICATE_CALL: "You already made this exact call; use its result or change the query.",
    ErrorCode.TOOL_DISABLED: "This tool kept failing and is disabled for this question; use another tool or report the gap.",
    ErrorCode.STEP_LIMIT: "Tool budget used up. Answer with the evidence you have and state what is missing.",
    ErrorCode.TIMEOUT: "The tool timed out after retries. Try a different tool or a simpler request.",
    ErrorCode.RATE_LIMITED: "The service is rate limited. Try a different tool or report the gap.",
    ErrorCode.HTTP_ERROR: "The service failed. Try a different tool or report the gap.",
    ErrorCode.CACHE_MISS: "No cached result exists for this call in replay mode. Try a different query or tool.",
    ErrorCode.SCHEMA_ERROR: "The tool returned malformed data. Rephrase or use a different tool.",
    ErrorCode.EMPTY_RESULT: "Nothing found. Try a broader or rephrased query, or a different tool.",
    ErrorCode.LOW_RELEVANCE: "Only weak matches. Rephrase with specific names/terms, or try web search.",
    ErrorCode.CODE_ERROR: "The code failed. Read the error, fix the code, and run it again.",
    ErrorCode.IMPLAUSIBLE_VALUE: "The result looks wrong (NaN/inf or out of range). Check units and inputs.",
    ErrorCode.INTERNAL: "Unexpected error. Try a different tool or report the gap.",
}


class ToolError(Exception):
    """Raised by a tool implementation. `retryable` tells the router whether to back off and retry."""

    def __init__(self, code: ErrorCode, message: str = "", retryable: bool = False):
        super().__init__(message or code.value)
        self.code = code
        self.message = message or code.value
        self.retryable = retryable


@dataclass
class Evidence:
    id: str  # e.g. W1 (web), P2 (paper), C1 (code)
    tool: str
    source: str  # URL, "paper_id p.<page>", or "python"
    title: str
    content: str
    meta: dict[str, Any] = field(default_factory=dict)

    def as_prompt(self) -> str:
        return f"[{self.id}] {self.title} ({self.source})\n{self.content}"


@dataclass
class ToolOutcome:
    tool: str
    args: dict[str, Any]
    ok: bool
    evidence: list[Evidence] = field(default_factory=list)
    error: ErrorCode | None = None
    message: str = ""
    warnings: list[str] = field(default_factory=list)
    attempts: int = 0
    latency_s: float = 0.0
    from_cache: bool = False

    def as_tool_result_text(self) -> str:
        """Text the LLM sees as the tool result."""
        if self.ok:
            parts = [e.as_prompt() for e in self.evidence]
            if self.warnings:
                parts.append("Validator notes: " + "; ".join(self.warnings))
            return "\n\n".join(parts)
        hint = RECOVERY_HINTS.get(self.error, "")
        return f"ERROR {self.error.value}: {self.message}\nSuggested next step: {hint}"


@dataclass
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0

    def add(self, other: "Usage") -> None:
        self.tokens_in += other.tokens_in
        self.tokens_out += other.tokens_out
