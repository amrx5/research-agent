"""Synthesizer: writes the final answer from validated evidence only, then checks it.

Post-checks (deterministic, in code):
  - every cited ID exists in the evidence store
  - every claim containing a number cites at least one evidence ID
    -> if not, regenerate once with feedback; anything still uncited is marked UNVERIFIED
  - cross-source check: two claims about the same quantity from different sources that
    differ by more than the tolerance are reported as a conflict (both kept, both cited)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .config import Config
from .executor import ExecutionResult
from .llm import LLM, extract_json
from .trace import Trace
from .types import Usage
from .validator import extract_numbers, numbers_conflict

SYNTH_SYSTEM = """You write the final answer to a technical question using ONLY the evidence provided.
Each evidence item has an ID like [W1] (web), [P1] (paper), [C1] (python calculation).

Reply with ONLY a JSON object:
{
  "status": "answered" | "partial" | "insufficient_evidence",
  "answer": "<direct answer in 1-3 sentences, with numbers and units>",
  "claims": [
    {"text": "<one factual claim>", "citations": ["P1"], "quantity": "<short name of the measured quantity, or null>", "value": <number or null>}
  ],
  "conflicts": ["<sources that disagree, citing both IDs>"],
  "gaps": ["<what could not be found or verified>"]
}

Rules:
- Every claim that contains a number MUST cite the evidence it comes from. Computed numbers cite the [C*] item.
- Do not add facts from your own memory. If the evidence does not support an answer, use status "insufficient_evidence" and explain the gap.
- Use "partial" when some parts are answered and others are not.
- If sources disagree, report both values with their citations instead of picking one."""

DIGIT_RE = re.compile(r"\d")


@dataclass
class Claim:
    text: str
    citations: list[str]
    quantity: str | None = None
    value: float | None = None
    unverified: bool = False


@dataclass
class AgentAnswer:
    question: str
    status: str
    answer: str
    claims: list[Claim] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    evidence: list[Any] = field(default_factory=list)  # Evidence used
    tools_called: list[str] = field(default_factory=list)
    stop_reason: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    trace_path: str | None = None

    def to_markdown(self) -> str:
        cited = {c for cl in self.claims for c in cl.citations}
        out = [f"**Answer ({self.status.replace('_', ' ')}):** {self.answer}", ""]
        if self.claims:
            out.append("**Supporting claims**")
            for cl in self.claims:
                tags = " ".join(f"[{c}]" for c in cl.citations)
                flag = " _(unverified)_" if cl.unverified else ""
                out.append(f"- {cl.text} {tags}{flag}".rstrip())
            out.append("")
        if self.conflicts:
            out += ["**Conflicting sources**"] + [f"- {c}" for c in self.conflicts] + [""]
        if self.gaps:
            out += ["**Gaps**"] + [f"- {g}" for g in self.gaps] + [""]
        used = [e for e in self.evidence if e.id in cited]
        if used:
            out.append("**Sources**")
            for e in used:
                out.append(f"- [{e.id}] {e.title} - {e.source}")
            out.append("")
        calcs = [e for e in used if e.tool == "python"]
        for e in calcs:
            out += [f"**Calculation [{e.id}]**", "```python", e.meta.get("code", "").strip(), "```",
                    f"Output: `{e.content[:300]}`", ""]
        return "\n".join(out).strip()


class Synthesizer:
    def __init__(self, llm: LLM, config: Config):
        self.llm = llm
        self.cfg = config

    def synthesize(self, question: str, ex: ExecutionResult, usage: Usage, trace: Trace) -> AgentAnswer:
        prompt = self._prompt(question, ex)
        messages = [{"role": "user", "content": prompt}]
        data = self._ask(messages, usage)
        claims, problems = self._check(data, ex) if data else ([], ["reply was not valid JSON"])

        if problems:  # one regeneration with concrete feedback
            trace.log("synth_retry", problems=problems)
            messages += [
                {"role": "assistant", "content": str(data) if data else "(invalid)"},
                {"role": "user", "content": "Fix these problems and reply with the full JSON again:\n- " + "\n- ".join(problems)},
            ]
            data2 = self._ask(messages, usage)
            if data2:
                data = data2
                claims, problems = self._check(data, ex)
            for cl in claims:  # still-uncited numeric claims are flagged, not silently kept
                if DIGIT_RE.search(cl.text) and not cl.citations:
                    cl.unverified = True

        if not data:
            data = {"status": "insufficient_evidence", "answer": "The agent could not produce a valid answer.", "gaps": ["synthesis failed"]}

        conflicts = list(data.get("conflicts") or []) + self._numeric_conflicts(claims, ex)
        gaps = list(data.get("gaps") or [])
        # Record tools that were unavailable and never recovered. "Nothing found" results are
        # left to the model's own gaps list, since another tool may have found the answer.
        infra = {"TIMEOUT", "RATE_LIMITED", "HTTP_ERROR", "INTERNAL", "SCHEMA_ERROR", "TOOL_DISABLED", "CACHE_MISS"}
        tools_ok = {o.tool for o in ex.outcomes if o.ok}
        unrecovered = {o.tool: o for o in ex.failures if o.tool not in tools_ok and o.error.value in infra}
        gaps += [f"{o.tool} unavailable ({o.error.value}): {o.message[:120]}" for o in unrecovered.values()]
        if ex.stop_reason in {"step_limit", "time_limit"}:
            gaps.append(f"stopped early ({ex.stop_reason.replace('_', ' ')})")

        status = str(data.get("status", "partial"))
        if status not in {"answered", "partial", "insufficient_evidence"}:
            status = "partial"
        if status == "answered" and any(cl.unverified for cl in claims):
            status = "partial"

        ans = AgentAnswer(question=question, status=status, answer=str(data.get("answer", "")).strip(),
                          claims=claims, conflicts=conflicts, gaps=gaps, evidence=list(ex.store.items),
                          stop_reason=ex.stop_reason)
        trace.log("synthesis", status=status, answer=ans.answer, n_claims=len(claims),
                  unverified=sum(c.unverified for c in claims), conflicts=conflicts, gaps=gaps)
        return ans

    # ------------------------------------------------------------- helpers
    def _prompt(self, question: str, ex: ExecutionResult) -> str:
        ev = "\n\n".join(e.as_prompt() for e in ex.store.items) or "(no evidence was collected)"
        fails = "\n".join(f"- {o.tool}({o.args}): {o.error.value} {o.message[:150]}" for o in ex.failures) or "- none"
        return (f"Question: {question}\n\nEVIDENCE:\n{ev}\n\nTOOL FAILURES:\n{fails}\n\n"
                f"Researcher's closing note: {ex.final_note or '(none)'}")

    def _ask(self, messages, usage: Usage) -> dict | None:
        resp = self.llm.chat(SYNTH_SYSTEM, messages, max_tokens=self.cfg.max_tokens)
        usage.add(resp.usage)
        return extract_json(resp.text)

    def _check(self, data: dict, ex: ExecutionResult) -> tuple[list[Claim], list[str]]:
        valid_ids, claims, problems = ex.store.ids, [], []
        for raw in data.get("claims") or []:
            if not isinstance(raw, dict) or not raw.get("text"):
                continue
            cites = [str(c).strip("[] ") for c in (raw.get("citations") or [])]
            bad = [c for c in cites if c not in valid_ids]
            if bad:
                problems.append(f"claim '{raw['text'][:60]}' cites unknown evidence {bad}")
            cites = [c for c in cites if c in valid_ids]
            value = raw.get("value")
            try:
                value = float(value) if value is not None else None
            except (TypeError, ValueError):
                value = None
            cl = Claim(text=str(raw["text"]), citations=cites, quantity=raw.get("quantity") or None, value=value)
            if DIGIT_RE.search(cl.text) and not cites:
                problems.append(f"claim '{cl.text[:60]}' has a number but no valid citation")
            claims.append(cl)
        if data.get("status") == "answered" and not claims and DIGIT_RE.search(str(data.get("answer", ""))):
            problems.append("the answer contains numbers but lists no cited claims")
        return claims, problems

    def _numeric_conflicts(self, claims: list[Claim], ex: ExecutionResult) -> list[str]:
        groups: dict[str, list[Claim]] = {}
        for cl in claims:
            if cl.quantity and cl.citations:
                val = cl.value if cl.value is not None else next(iter(extract_numbers(cl.text)), None)
                if val is not None:
                    cl.value = val
                    groups.setdefault(cl.quantity.strip().lower(), []).append(cl)
        out = []
        for qty, cls in groups.items():
            for i in range(len(cls)):
                for j in range(i + 1, len(cls)):
                    a, b = cls[i], cls[j]
                    src_a = {ex.store.get(c).source for c in a.citations if ex.store.get(c)}
                    src_b = {ex.store.get(c).source for c in b.citations if ex.store.get(c)}
                    if src_a != src_b and numbers_conflict(a.value, b.value, self.cfg.conflict_tolerance):
                        out.append(f"{qty}: {a.value:g} [{', '.join(a.citations)}] vs {b.value:g} [{', '.join(b.citations)}]")
        return out
