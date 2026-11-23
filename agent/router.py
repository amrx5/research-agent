"""Tool router: argument validation, de-duplication, caching, timeouts, retries with
exponential backoff, a per-tool circuit breaker, and the hand-off to the Validator.

The optional `fault_hook(tool_name, args, raw) -> raw` lets the eval harness inject
failures (timeouts, empty results, corrupted numbers) without touching tool code.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import time
from typing import Any, Callable

from pydantic import ValidationError

from .config import Config
from .tools.base import Tool
from .trace import Trace
from .types import ErrorCode, Evidence, ToolError, ToolOutcome
from .validator import Validator

FaultHook = Callable[[str, dict, Any], Any]


class EvidenceStore:
    """Holds validated evidence for one question and hands out stable IDs (W1, P1, C1...)."""

    def __init__(self):
        self.items: list[Evidence] = []
        self._counters: dict[str, int] = {}

    def add(self, prefix: str, tool: str, item: dict) -> Evidence:
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        ev = Evidence(
            id=f"{prefix}{self._counters[prefix]}",
            tool=tool,
            source=item["source"],
            title=item.get("title") or item["source"],
            content=item["content"],
            meta={**item.get("meta", {}), "relevance": item.get("relevance")},
        )
        self.items.append(ev)
        return ev

    def get(self, ev_id: str) -> Evidence | None:
        return next((e for e in self.items if e.id == ev_id), None)

    @property
    def ids(self) -> set[str]:
        return {e.id for e in self.items}


class ToolRouter:
    def __init__(
        self,
        tools: list[Tool],
        config: Config,
        validator: Validator | None = None,
        trace: Trace | None = None,
        fault_hook: FaultHook | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.tools = {t.name: t for t in tools}
        self.cfg = config
        self.validator = validator or Validator(config)
        self.trace = trace
        self.fault_hook = fault_hook
        self.sleep = sleep
        self._pool = cf.ThreadPoolExecutor(max_workers=4)
        self.reset()

    def reset(self) -> None:
        """Per-question state."""
        self._seen_calls: set[str] = set()
        self._consecutive_failures: dict[str, int] = {}
        self.disabled: set[str] = set()

    def specs(self) -> list[dict]:
        return [t.spec() for t in self.tools.values()]

    # ------------------------------------------------------------------ main
    def call(self, name: str, args: dict, store: EvidenceStore) -> ToolOutcome:
        start = time.monotonic()
        outcome = self._call(name, args, store)
        outcome.latency_s = round(time.monotonic() - start, 3)
        self._update_breaker(outcome)
        if self.trace:
            self.trace.log(
                "tool_result",
                tool=name,
                args=args,
                ok=outcome.ok,
                error=outcome.error.value if outcome.error else None,
                message=outcome.message[:300],
                evidence_ids=[e.id for e in outcome.evidence],
                warnings=outcome.warnings,
                attempts=outcome.attempts,
                latency_s=outcome.latency_s,
                from_cache=outcome.from_cache,
            )
        return outcome

    def _call(self, name: str, args: dict, store: EvidenceStore) -> ToolOutcome:
        tool = self.tools.get(name)
        if tool is None:
            return ToolOutcome(name, args, False, error=ErrorCode.UNKNOWN_TOOL, message=f"no tool named '{name}'; available: {sorted(self.tools)}")
        if name in self.disabled:
            return ToolOutcome(name, args, False, error=ErrorCode.TOOL_DISABLED, message=f"{name} failed repeatedly")
        try:
            parsed = tool.Args.model_validate(args)
        except ValidationError as exc:
            msg = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            return ToolOutcome(name, args, False, error=ErrorCode.INVALID_ARGS, message=msg)

        canonical = parsed.model_dump()
        key = hashlib.sha256(f"{name}:{json.dumps(canonical, sort_keys=True)}".encode()).hexdigest()[:24]
        if key in self._seen_calls:
            return ToolOutcome(name, canonical, False, error=ErrorCode.DUPLICATE_CALL, message="identical call already made for this question")
        self._seen_calls.add(key)

        raw, attempts, from_cache, err = self._run_with_retries(tool, parsed, canonical, key)
        if err is not None:
            return ToolOutcome(name, canonical, False, error=err.code, message=err.message, attempts=attempts)

        if self.fault_hook:
            try:
                raw = self.fault_hook(name, canonical, raw)
            except ToolError as exc:
                return ToolOutcome(name, canonical, False, error=exc.code, message=exc.message, attempts=attempts)

        result = self.validator.validate(name, raw)
        if not result.ok:
            return ToolOutcome(name, canonical, False, error=result.error, message=result.message,
                               warnings=result.warnings, attempts=attempts, from_cache=from_cache)
        evidence = [store.add(tool.evidence_prefix, name, it) for it in result.items]
        return ToolOutcome(name, canonical, True, evidence=evidence, warnings=result.warnings,
                           attempts=attempts, from_cache=from_cache)

    # ------------------------------------------------------------- helpers
    def _run_with_retries(self, tool: Tool, parsed, canonical: dict, key: str):
        use_cache = tool.cacheable and self.cfg.cache_mode != "off"
        cache_file = self.cfg.cache_dir / tool.name / f"{key}.json"
        if use_cache and cache_file.exists():
            return json.loads(cache_file.read_text())["raw"], 0, True, None
        if use_cache and self.cfg.cache_mode == "replay":
            return None, 0, False, ToolError(ErrorCode.CACHE_MISS, "replay mode and no cached result for this call")

        last: ToolError | None = None
        for attempt in range(1, self.cfg.max_attempts + 1):
            future = self._pool.submit(tool.run, parsed)
            try:
                raw = future.result(timeout=tool.timeout_s)
                if use_cache:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps({"tool": tool.name, "args": canonical, "raw": raw}))
                return raw, attempt, False, None
            except cf.TimeoutError:
                future.cancel()
                last = ToolError(ErrorCode.TIMEOUT, f"no response within {tool.timeout_s:.0f} s", retryable=True)
            except ToolError as exc:
                last = exc
            except Exception as exc:  # noqa: BLE001 - a bug in a tool must not crash the agent
                last = ToolError(ErrorCode.INTERNAL, f"{type(exc).__name__}: {exc}", retryable=False)
            if not last.retryable or attempt == self.cfg.max_attempts:
                return None, attempt, False, last
            self.sleep(self.cfg.backoff_base_s * 2 ** (attempt - 1))
        return None, self.cfg.max_attempts, False, last

    def _update_breaker(self, outcome: ToolOutcome) -> None:
        # Only infrastructure failures count; "nothing found" or a bad query is the model's to fix.
        infra = {ErrorCode.TIMEOUT, ErrorCode.RATE_LIMITED, ErrorCode.HTTP_ERROR, ErrorCode.INTERNAL, ErrorCode.SCHEMA_ERROR}
        name = outcome.tool
        if outcome.ok:
            self._consecutive_failures[name] = 0
        elif outcome.error in infra:
            n = self._consecutive_failures.get(name, 0) + 1
            self._consecutive_failures[name] = n
            if n >= self.cfg.tool_failures_before_disable:
                self.disabled.add(name)
