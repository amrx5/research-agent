"""Tool interface. A tool declares a name, an LLM-facing description, a Pydantic
argument model, and a `run()` that returns raw items. It does NOT validate its own
output; that is the Validator's job, so every tool is checked the same way."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel

# Raw item every tool returns (validated later):
#   {"source": str, "title": str, "content": str, "relevance": float | None, "meta": dict}
RawItem = dict[str, Any]


class Tool(ABC):
    name: ClassVar[str]
    evidence_prefix: ClassVar[str]  # W / P / C
    description: ClassVar[str]  # this text IS the selection logic the LLM reads: keep it precise
    Args: ClassVar[type[BaseModel]]
    timeout_s: float = 15.0
    cacheable: bool = False

    def spec(self) -> dict[str, Any]:
        """Tool definition in the Anthropic `tools=[...]` format."""
        schema = self.Args.model_json_schema()
        schema.pop("title", None)
        return {"name": self.name, "description": self.description, "input_schema": schema}

    @abstractmethod
    def run(self, args: BaseModel) -> list[RawItem]:
        """Execute the tool. Raise `ToolError` for failures the router should know about."""
