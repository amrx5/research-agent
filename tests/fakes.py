"""Test doubles: a scripted LLM and configurable fake tools, so the whole agent loop
runs offline and deterministically (no API keys, no network, no Docker)."""
from __future__ import annotations

import itertools
from typing import Any, Callable

from pydantic import BaseModel, Field

from agent.llm import LLMResponse, LLMToolCall
from agent.tools.base import Tool
from agent.types import Usage

_ids = itertools.count(1)


def text(t: str) -> LLMResponse:
    return LLMResponse(text=t, raw_content=[{"type": "text", "text": t}], usage=Usage(100, 20))


def call(name: str, **inp) -> LLMResponse:
    tid = f"toolu_{next(_ids)}"
    return LLMResponse(
        text="",
        tool_calls=[LLMToolCall(tid, name, inp)],
        raw_content=[{"type": "tool_use", "id": tid, "name": name, "input": inp}],
        stop_reason="tool_use",
        usage=Usage(150, 30),
    )


Scripted = LLMResponse | Callable[[str, list, Any], LLMResponse]


class ScriptedLLM:
    """Routes each chat() to the planner / executor / synthesizer script by system prompt."""

    def __init__(self, plan: list[Scripted] = (), execute: list[Scripted] = (), synth: list[Scripted] = ()):
        self.queues = {"plan": list(plan), "execute": list(execute), "synth": list(synth)}
        self.calls: list[tuple[str, list]] = []

    def chat(self, system, messages, tools=None, max_tokens=None) -> LLMResponse:
        role = "plan" if system.startswith("You plan") else "synth" if system.startswith("You write the final answer") else "execute"
        self.calls.append((role, [m for m in messages]))
        q = self.queues[role]
        if not q:
            return text("DONE: nothing more") if role == "execute" else text("{}")
        item = q.pop(0)
        return item(system, messages, tools) if callable(item) else item


class QueryArgs(BaseModel):
    query: str = Field(..., min_length=2)


class CodeArgs(BaseModel):
    code: str = Field(..., min_length=1)


class FakeTool(Tool):
    """Returns (or raises) the next item from `results` on each run()."""

    def __init__(self, name: str, prefix: str, results: list, args_model=QueryArgs, timeout_s: float = 2.0, cacheable=False):
        self.name = name
        self.evidence_prefix = prefix
        self.description = f"fake {name}"
        self.Args = args_model
        self.results = list(results)
        self.timeout_s = timeout_s
        self.cacheable = cacheable
        self.runs: list[BaseModel] = []

    def run(self, args):
        self.runs.append(args)
        item = self.results.pop(0) if self.results else []
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(args)
        return item


def doc(source: str, content: str, title: str = "", relevance: float | None = None, **meta) -> dict:
    return {"source": source, "title": title or source, "content": content, "relevance": relevance, "meta": meta}


def code_result(stdout: str, code: str = "print(1)", exit_code: int = 0, stderr: str = "") -> list[dict]:
    return [{"source": "python", "title": "Python calculation", "content": stdout, "relevance": None,
             "meta": {"code": code, "exit_code": exit_code, "stderr": stderr}}]
