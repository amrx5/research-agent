"""Thin LLM adapter. The rest of the code only sees `LLM.chat()` and `LLMResponse`,
so switching provider (or plugging in a scripted fake for tests) is a one-class change.

Messages use the Anthropic Messages format as the internal canonical format:
  {"role": "user"|"assistant", "content": str | list[block]}
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .types import Usage


@dataclass
class LLMToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    raw_content: list[dict[str, Any]] = field(default_factory=list)  # appended to history as-is
    stop_reason: str = ""
    usage: Usage = field(default_factory=Usage)


class LLM(Protocol):
    def chat(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...


class AnthropicLLM:
    """Claude via the Anthropic Messages API with native tool use."""

    def __init__(self, model: str, max_tokens: int = 2048, api_key: str | None = None):
        import anthropic  # imported lazily so tests run without the SDK

        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def chat(self, system, messages, tools=None, max_tokens=None) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            system=system,
            messages=messages,
        )
        if tools:
            kwargs["tools"] = tools
        resp = self.client.messages.create(**kwargs)

        text_parts, calls, raw = [], [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
                raw.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                calls.append(LLMToolCall(id=block.id, name=block.name, input=dict(block.input)))
                raw.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=calls,
            raw_content=raw,
            stop_reason=resp.stop_reason or "",
            usage=Usage(resp.usage.input_tokens, resp.usage.output_tokens),
        )


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of an LLM reply (tolerates ```json fences and chatter)."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fence.group(1)] if fence else []
    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None
