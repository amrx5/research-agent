"""Validator: every tool result passes through here before the LLM sees it.

Checks, in order:
  1. schema        - result is a list of items with text content and a source
  2. code result   - python: non-zero exit -> CODE_ERROR, no output -> EMPTY_RESULT, NaN/inf -> IMPLAUSIBLE_VALUE
  3. empty         - nothing usable -> EMPTY_RESULT
  4. relevance     - drop items under the relevance threshold; all dropped -> LOW_RELEVANCE
  5. injection     - strip instruction-like lines from web/document text (kept as data, never obeyed)
  6. hygiene       - de-duplicate, truncate long content
The cross-source numeric check lives in `numbers_conflict()` and is applied by the synthesizer.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .types import ErrorCode

INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?)",
    r"disregard (all |the )?(previous|prior|above)",
    r"you are now\b",
    r"new instructions?:",
    r"\bsystem prompt\b",
    r"^\s*(system|assistant)\s*:",
    r"<\|.*?\|>",
    r"do not (tell|inform) the user",
    r"call the \w+ tool",
]
INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.I | re.M)
NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
BAD_NUMBER_RE = re.compile(r"\b(nan|inf|-inf|infinity)\b", re.I)


@dataclass
class ValidationResult:
    ok: bool
    items: list[dict[str, Any]] = field(default_factory=list)
    error: ErrorCode | None = None
    message: str = ""
    warnings: list[str] = field(default_factory=list)


def extract_numbers(text: str) -> list[float]:
    out = []
    for m in NUMBER_RE.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def numbers_conflict(a: float, b: float, tolerance: float = 0.05) -> bool:
    """True if two reported values for the same quantity differ by more than `tolerance` (relative)."""
    if a == b:
        return False
    scale = max(abs(a), abs(b))
    return scale > 0 and abs(a - b) / scale > tolerance


def strip_injection(text: str) -> tuple[str, int]:
    kept, removed = [], 0
    for line in text.splitlines():
        if INJECTION_RE.search(line):
            removed += 1
        else:
            kept.append(line)
    return "\n".join(kept), removed


class Validator:
    def __init__(self, config: Config):
        self.cfg = config

    def validate(self, tool_name: str, raw: Any) -> ValidationResult:
        # 1. schema
        if not isinstance(raw, list):
            return ValidationResult(False, error=ErrorCode.SCHEMA_ERROR, message=f"expected a list of results, got {type(raw).__name__}")
        items, malformed = [], 0
        for it in raw:
            if isinstance(it, dict) and isinstance(it.get("content"), str) and isinstance(it.get("source"), str):
                items.append(dict(it, meta=dict(it.get("meta") or {})))
            else:
                malformed += 1
        if raw and not items:
            return ValidationResult(False, error=ErrorCode.SCHEMA_ERROR, message=f"all {malformed} results were malformed")
        warnings = [f"dropped {malformed} malformed result(s)"] if malformed else []

        # 2. code results
        if tool_name == "python":
            return self._validate_code(items, warnings)

        # 3. empty
        items = [it for it in items if it["content"].strip()]
        if not items:
            return ValidationResult(False, error=ErrorCode.EMPTY_RESULT, message="the tool returned no results")

        # 4. relevance (only where the tool reports a score)
        if tool_name == "paper_search":
            scored = [it for it in items if it.get("relevance") is not None]
            kept = [it for it in items if it.get("relevance") is None or it["relevance"] >= self.cfg.min_relevance]
            if not kept:
                best = max(it["relevance"] for it in scored)
                return ValidationResult(
                    False,
                    error=ErrorCode.LOW_RELEVANCE,
                    message=f"best passage relevance {best:.2f} is below threshold {self.cfg.min_relevance:.2f}",
                )
            if len(kept) < len(items):
                warnings.append(f"dropped {len(items) - len(kept)} weakly relevant passage(s)")
            items = kept

        # 5. injection + 6. hygiene
        seen, clean = set(), []
        for it in items:
            text, removed = strip_injection(it["content"])
            if removed:
                warnings.append(f"removed {removed} instruction-like line(s) from {it['source']}")
            text = text.strip()
            if len(text) > self.cfg.max_evidence_chars:
                text = text[: self.cfg.max_evidence_chars] + " [...truncated]"
            key = hashlib.sha1((it["source"] + text[:500]).encode()).hexdigest()
            if not text or key in seen:
                continue
            seen.add(key)
            clean.append({**it, "content": text})
        if not clean:
            return ValidationResult(False, error=ErrorCode.EMPTY_RESULT, message="no usable content after cleaning", warnings=warnings)
        return ValidationResult(True, items=clean, warnings=warnings)

    def _validate_code(self, items: list[dict], warnings: list[str]) -> ValidationResult:
        if not items:
            return ValidationResult(False, error=ErrorCode.SCHEMA_ERROR, message="sandbox returned nothing")
        it = items[0]
        meta = it["meta"]
        if meta.get("exit_code", 0) != 0:
            return ValidationResult(False, error=ErrorCode.CODE_ERROR, message=meta.get("stderr", "") or "non-zero exit code")
        out = it["content"].strip()
        if not out:
            return ValidationResult(False, error=ErrorCode.EMPTY_RESULT, message="the code printed nothing; print() the result")
        if BAD_NUMBER_RE.search(out):
            return ValidationResult(False, error=ErrorCode.IMPLAUSIBLE_VALUE, message=f"output contains NaN/inf: {out[:200]}")
        if any(math.isinf(x) for x in extract_numbers(out)):
            return ValidationResult(False, error=ErrorCode.IMPLAUSIBLE_VALUE, message="output overflowed to infinity")
        if len(out) > self.cfg.max_evidence_chars:
            out = out[: self.cfg.max_evidence_chars] + " [...truncated]"
        return ValidationResult(True, items=[{**it, "content": out}], warnings=warnings)
