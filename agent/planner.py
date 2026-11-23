"""Planner: one cheap LLM call that drafts an ordered tool plan. The plan is a HINT for
the executor (not a script), which keeps routing flexible while giving the model a
head start on multi-step questions."""
from __future__ import annotations

from dataclasses import dataclass, field

from .llm import LLM, extract_json
from .trace import Trace
from .types import Usage

PLANNER_SYSTEM = """You plan how to answer a technical question with tools.
Available tools:
{tool_list}

Reply with ONLY a JSON object:
{{"needs_tools": true|false,
  "steps": [{{"tool": "<tool name>", "purpose": "<what this step finds or computes>"}}],
  "notes": "<one line: what makes this question tricky, if anything>"}}

Rules:
- 0 to 4 steps, in the order they depend on each other.
- Any arithmetic, even simple, is a python step after the steps that find its inputs.
- Facts that may have changed recently need web_search. Content of a specific paper in the corpus needs paper_search.
- needs_tools is false only for pure definitions or reasoning that needs no facts or numbers."""


@dataclass
class PlanStep:
    tool: str
    purpose: str


@dataclass
class Plan:
    needs_tools: bool = True
    steps: list[PlanStep] = field(default_factory=list)
    notes: str = ""

    def as_hint(self) -> str:
        if not self.steps:
            return "Suggested plan: none (answer may not need tools)." if not self.needs_tools else ""
        lines = [f"{i}. {s.tool}: {s.purpose}" for i, s in enumerate(self.steps, 1)]
        return "Suggested plan (adapt it if results show a better route):\n" + "\n".join(lines)


class Planner:
    def __init__(self, llm: LLM, tool_specs: list[dict]):
        self.llm = llm
        self.tool_names = {t["name"] for t in tool_specs}
        tool_list = "\n".join(f"- {t['name']}: {t['description']}" for t in tool_specs)
        self.system = PLANNER_SYSTEM.format(tool_list=tool_list)

    def plan(self, question: str, usage: Usage, trace: Trace | None = None) -> Plan:
        resp = self.llm.chat(self.system, [{"role": "user", "content": question}], max_tokens=500)
        usage.add(resp.usage)
        data = extract_json(resp.text) or {}
        steps = [
            PlanStep(str(s.get("tool")), str(s.get("purpose", "")))
            for s in data.get("steps", [])
            if isinstance(s, dict) and s.get("tool") in self.tool_names
        ][:4]
        plan = Plan(needs_tools=bool(data.get("needs_tools", True)), steps=steps, notes=str(data.get("notes", "")))
        if trace:
            trace.log("plan", parsed=bool(data), needs_tools=plan.needs_tools,
                      steps=[s.__dict__ for s in plan.steps], notes=plan.notes)
        return plan
