"""End-to-end tests of the loop with a scripted LLM and fake tools."""
import json

from fakes import CodeArgs, FakeTool, ScriptedLLM, call, code_result, doc, text

from agent.agent import ResearchAgent

PAPER = doc("vaswani2017 p.8", "The big transformer model achieves 28.4 BLEU on EN-DE and 41.8 BLEU on EN-FR.",
            title="Attention Is All You Need (2017) - Results", relevance=0.82)
PLAN = text(json.dumps({"needs_tools": True, "steps": [{"tool": "paper_search", "purpose": "find BLEU"},
                                                        {"tool": "python", "purpose": "percent"}]}))
GOOD_SYNTH = text(json.dumps({
    "status": "answered",
    "answer": "EN-FR (41.8 BLEU) is 47.18% higher than EN-DE (28.4 BLEU).",
    "claims": [
        {"text": "Big Transformer: 28.4 BLEU on EN-DE", "citations": ["P1"], "quantity": "bleu en-de", "value": 28.4},
        {"text": "Big Transformer: 41.8 BLEU on EN-FR", "citations": ["P1"], "quantity": "bleu en-fr", "value": 41.8},
        {"text": "Difference is 47.18%", "citations": ["C1"], "quantity": None, "value": 47.18},
    ],
    "conflicts": [], "gaps": [],
}))


def tools(paper_results, code_results=None, web_results=None):
    return [
        FakeTool("web_search", "W", web_results or []),
        FakeTool("paper_search", "P", paper_results),
        FakeTool("python", "C", code_results or [code_result("47.18", code="print(round((41.8-28.4)/28.4*100,2))")], args_model=CodeArgs),
    ]


def test_happy_path_multi_tool(cfg):
    llm = ScriptedLLM(
        plan=[PLAN],
        execute=[call("paper_search", query="Transformer BLEU WMT 2014"),
                 call("python", code="print(round((41.8-28.4)/28.4*100,2))  # P1"),
                 text("DONE: found both scores and computed the difference")],
        synth=[GOOD_SYNTH],
    )
    agent = ResearchAgent(cfg, llm=llm, tools=tools([[PAPER]]))
    ans = agent.run("How much higher is EN-FR than EN-DE BLEU?", "t1")
    assert ans.status == "answered"
    assert ans.tools_called == ["paper_search", "python"]
    assert {c for cl in ans.claims for c in cl.citations} == {"P1", "C1"}
    md = ans.to_markdown()
    assert "[P1]" in md and "```python" in md and "47.18" in md
    assert ans.cost_usd > 0 and ans.trace_path
    events = [json.loads(line)["event"] for line in open(ans.trace_path)]
    assert events[0] == "question" and "plan" in events and events[-1] == "final"


def test_recovers_from_empty_result_by_switching_tool(cfg):
    web = [doc("https://example.org/bleu", "Transformer big: 28.4 BLEU EN-DE, 41.8 EN-FR")]
    seen_error = {}

    def after_error(system, messages, tools_):
        # the executor must have passed the typed error note back to the model
        last = messages[-1]["content"][0]
        seen_error["text"] = last["content"]
        return call("web_search", query="Transformer big BLEU WMT14")

    llm = ScriptedLLM(
        plan=[PLAN],
        execute=[call("paper_search", query="Transformer BLEU"), after_error, call("python", code="print(47.18)"),
                 text("DONE")],
        synth=[text(json.dumps({"status": "answered", "answer": "47.18% higher",
                                "claims": [{"text": "28.4 and 41.8 BLEU", "citations": ["W1"]},
                                           {"text": "47.18% higher", "citations": ["C1"]}]}))],
    )
    agent = ResearchAgent(cfg, llm=llm, tools=tools([[]], web_results=[web]))
    ans = agent.run("q", "t2")
    assert "ERROR EMPTY_RESULT" in seen_error["text"] and "Suggested next step" in seen_error["text"]
    assert ans.tools_called == ["paper_search", "web_search", "python"]
    assert ans.status == "answered" and not ans.gaps  # recovered failure is not a gap


def test_uncited_number_regenerated_then_flagged(cfg):
    bad = text(json.dumps({"status": "answered", "answer": "It is 99%.",
                           "claims": [{"text": "The value is 99%", "citations": []}]}))
    llm = ScriptedLLM(plan=[PLAN], execute=[call("paper_search", query="BLEU"), text("DONE")], synth=[bad, bad])
    ans = ResearchAgent(cfg, llm=llm, tools=tools([[PAPER]])).run("q", "t3")
    assert ans.claims[0].unverified and ans.status == "partial"
    assert sum(1 for role, _ in llm.calls if role == "synth") == 2


def test_unknown_citation_is_removed(cfg):
    s = text(json.dumps({"status": "answered", "answer": "28.4 BLEU",
                         "claims": [{"text": "28.4 BLEU", "citations": ["P1", "W9"]}]}))
    llm = ScriptedLLM(plan=[PLAN], execute=[call("paper_search", query="BLEU"), text("DONE")], synth=[s, s])
    ans = ResearchAgent(cfg, llm=llm, tools=tools([[PAPER]])).run("q", "t4")
    assert ans.claims[0].citations == ["P1"]


def test_step_limit_stops_loop(cfg):
    cfg.max_tool_calls = 3
    llm = ScriptedLLM(plan=[PLAN], execute=[call("paper_search", query=f"query {i}") for i in range(10)],
                      synth=[text(json.dumps({"status": "partial", "answer": "partial", "claims": []}))])
    ans = ResearchAgent(cfg, llm=llm, tools=tools([[PAPER]] * 10)).run("q", "t5")
    assert len(ans.tools_called) == 3 and ans.stop_reason == "step_limit"
    assert any("step limit" in g for g in ans.gaps)


def test_numeric_conflict_between_sources(cfg):
    web = [doc("https://blog.example/bleu", "Transformer gets 35.0 BLEU on EN-DE")]
    s = text(json.dumps({"status": "answered", "answer": "Sources disagree: 28.4 vs 35.0",
                         "claims": [{"text": "28.4 BLEU EN-DE", "citations": ["P1"], "quantity": "BLEU EN-DE", "value": 28.4},
                                    {"text": "35.0 BLEU EN-DE", "citations": ["W1"], "quantity": "bleu en-de", "value": 35.0}]}))
    llm = ScriptedLLM(plan=[PLAN], execute=[call("paper_search", query="BLEU"), call("web_search", query="BLEU EN-DE"),
                                            text("DONE")], synth=[s])
    ans = ResearchAgent(cfg, llm=llm, tools=tools([[PAPER]], web_results=[web])).run("q", "t6")
    assert any("28.4" in c and "35" in c for c in ans.conflicts)


def test_unrecovered_tool_failure_reported_as_gap(cfg):
    from agent.types import ErrorCode, ToolError

    cfg.max_attempts = 1
    err = ToolError(ErrorCode.HTTP_ERROR, "503", retryable=True)
    llm = ScriptedLLM(plan=[PLAN], execute=[call("web_search", query="latest SOTA"), text("DONE: web down")],
                      synth=[text(json.dumps({"status": "insufficient_evidence", "answer": "Could not find it.", "claims": []}))])
    ans = ResearchAgent(cfg, llm=llm, tools=tools([], web_results=[err])).run("q", "t7")
    assert ans.status == "insufficient_evidence"
    assert any("web_search unavailable" in g for g in ans.gaps)
