"""ResearchAgent: wires planner -> executor -> synthesizer for one question."""
from __future__ import annotations

import time

from .config import Config
from .executor import Executor
from .llm import LLM, AnthropicLLM
from .planner import Planner
from .router import FaultHook, ToolRouter
from .synthesizer import AgentAnswer, Synthesizer
from .tools import Tool, default_tools
from .trace import Trace
from .types import Usage


class ResearchAgent:
    def __init__(
        self,
        config: Config | None = None,
        llm: LLM | None = None,
        tools: list[Tool] | None = None,
        fault_hook: FaultHook | None = None,
        use_planner: bool = True,
    ):
        self.cfg = config or Config()
        self.llm = llm or AnthropicLLM(self.cfg.model, self.cfg.max_tokens)
        self.router = ToolRouter(tools if tools is not None else default_tools(self.cfg), self.cfg, fault_hook=fault_hook)
        self.planner = Planner(self.llm, self.router.specs()) if use_planner else None
        self.executor = Executor(self.llm, self.router, self.cfg)
        self.synthesizer = Synthesizer(self.llm, self.cfg)

    def run(self, question: str, question_id: str = "q") -> AgentAnswer:
        t0 = time.monotonic()
        trace = Trace(self.cfg.trace_dir, question_id)
        usage = Usage()
        self.router.reset()
        self.router.trace = trace
        trace.log("question", id=question_id, question=question, model=self.cfg.model)

        plan = self.planner.plan(question, usage, trace) if self.planner else None
        ex = self.executor.run(question, plan, usage, trace)
        answer = self.synthesizer.synthesize(question, ex, usage, trace)

        answer.tools_called = trace.tools_called()
        answer.tokens_in, answer.tokens_out = usage.tokens_in, usage.tokens_out
        answer.cost_usd = round(self.cfg.cost_usd(usage.tokens_in, usage.tokens_out), 5)
        answer.latency_s = round(time.monotonic() - t0, 2)
        answer.trace_path = str(trace.path) if trace.path else None
        trace.log("final", status=answer.status, latency_s=answer.latency_s, tokens_in=usage.tokens_in,
                  tokens_out=usage.tokens_out, cost_usd=answer.cost_usd, tools_called=answer.tools_called)
        return answer
