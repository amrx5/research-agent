import json

from fakes import CodeArgs, FakeTool, ScriptedLLM, call, doc, text

from agent.agent import ResearchAgent
from agent.synthesizer import AgentAnswer
from eval.faults import FaultInjector
from eval.run_eval import grade, tool_f1


def test_grade_numeric_and_unanswerable():
    q = {"answerable": True, "answer_type": "numeric", "reference": 47.18, "tolerance": 0.02}
    assert grade(q, AgentAnswer("q", "answered", "It is 47.2% higher."))
    assert not grade(q, AgentAnswer("q", "answered", "It is 52% higher."))
    assert grade({"answerable": False}, AgentAnswer("q", "insufficient_evidence", "Not reported."))
    assert not grade({"answerable": False}, AgentAnswer("q", "answered", "0.42"))
    assert grade({"answer_type": "text"}, AgentAnswer("q", "answered", "x")) is None


def test_tool_f1():
    assert tool_f1(["paper_search", "python"], ["paper_search", "python"]) == 1.0
    assert round(tool_f1(["web_search", "paper_search", "python"], ["paper_search", "python"]), 2) == 0.8
    assert tool_f1(["web_search"], ["python"]) == 0.0


def test_injected_prompt_injection_is_filtered_end_to_end(cfg):
    inj = FaultInjector("injection", tools=("paper_search",))
    inj.new_question()
    seen = {}

    def capture(system, messages, tools):
        seen["tool_result"] = messages[-1]["content"][0]["content"]
        return text("DONE")

    llm = ScriptedLLM(execute=[call("paper_search", query="BLEU"), capture],
                      synth=[text(json.dumps({"status": "answered", "answer": "28.4", "claims": [{"text": "28.4 BLEU", "citations": ["P1"]}]}))])
    tools = [FakeTool("paper_search", "P", [[doc("p1", "BLEU is 28.4.", relevance=0.9)]]),
             FakeTool("python", "C", [], args_model=CodeArgs)]
    ResearchAgent(cfg, llm=llm, tools=tools, fault_hook=inj, use_planner=False).run("q", "e1")
    assert inj.injected == ["injection"]
    assert "28.4" in seen["tool_result"] and "Ignore all previous" not in seen["tool_result"]
    assert "instruction-like" in seen["tool_result"]
