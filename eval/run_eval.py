"""Evaluation harness: runs a question set and scores the M2 success criteria.

    python -m eval.run_eval --questions eval/questions.jsonl --split dev
    python -m eval.run_eval --mode llm_only            # baseline 1: no tools
    python -m eval.run_eval --mode all_tools           # baseline 2: always call every lookup tool
    python -m eval.run_eval --faults mixed             # failure-handling run
    CACHE_MODE=replay python -m eval.run_eval ...      # rerun from cached search results

Question file (JSONL), one object per line:
  {"id": "q01", "split": "dev", "question": "...", "answerable": true,
   "answer_type": "numeric" | "text", "reference": 47.18, "tolerance": 0.02,
   "expected_tools": ["paper_search", "python"]}

Numeric answers are graded automatically; text answers get correct=None and are left
for two team members to grade in the CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path

from agent.agent import ResearchAgent
from agent.config import Config
from agent.executor import ExecutionResult
from agent.router import EvidenceStore
from agent.synthesizer import AgentAnswer
from agent.trace import Trace
from agent.types import Usage
from agent.validator import extract_numbers

from .faults import FaultInjector

TARGETS = {
    "answer_accuracy": ">= 0.70",
    "tool_f1": ">= 0.80",
    "failure_handling": ">= 0.80",
    "fabricated_on_unanswerable": "= 0",
    "latency_median_s": "<= 30",
    "latency_p95_s": "<= 60",
    "cost_median_usd": "<= 0.10",
    "calls_vs_expected": "<= 1.5",
}


def load_questions(path: Path, split: str | None, limit: int | None) -> list[dict]:
    qs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if split:
        qs = [q for q in qs if q.get("split") == split]
    return qs[:limit] if limit else qs


def grade(q: dict, ans: AgentAnswer) -> bool | None:
    if not q.get("answerable", True):
        return ans.status == "insufficient_evidence"
    if ans.status == "insufficient_evidence":
        return False
    if q.get("answer_type") == "numeric" and q.get("reference") is not None:
        ref, tol = float(q["reference"]), float(q.get("tolerance", 0.02))
        return any(abs(x - ref) <= tol * max(abs(ref), 1e-9) for x in extract_numbers(ans.answer))
    return None  # manual grading


def tool_f1(called: list[str], expected: list[str]) -> float:
    c, e = set(called), set(expected)
    if not c and not e:
        return 1.0
    if not c or not e:
        return 0.0
    tp = len(c & e)
    if tp == 0:
        return 0.0
    p, r = tp / len(c), tp / len(e)
    return 2 * p * r / (p + r)


def run_llm_only(agent: ResearchAgent, q: dict) -> AgentAnswer:
    usage = Usage()
    resp = agent.llm.chat("Answer the technical question concisely, with numbers and units where relevant.",
                          [{"role": "user", "content": q["question"]}], max_tokens=400)
    usage.add(resp.usage)
    a = AgentAnswer(question=q["question"], status="answered", answer=resp.text.strip())
    a.tokens_in, a.tokens_out = usage.tokens_in, usage.tokens_out
    return a


def run_all_tools(agent: ResearchAgent, q: dict) -> AgentAnswer:
    """Naive baseline: query every lookup tool with the raw question, then synthesize."""
    trace = Trace(agent.cfg.trace_dir, f"{q['id']}_alltools")
    agent.router.reset()
    agent.router.trace = trace
    usage, store = Usage(), EvidenceStore()
    ex = ExecutionResult(store=store)
    for name in ("web_search", "paper_search"):
        if name in agent.router.tools:
            ex.outcomes.append(agent.router.call(name, {"query": q["question"][:300]}, store))
    a = agent.synthesizer.synthesize(q["question"], ex, usage, trace)
    a.tools_called = trace.tools_called()
    a.tokens_in, a.tokens_out = usage.tokens_in, usage.tokens_out
    return a


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=Path("eval/questions.jsonl"))
    ap.add_argument("--split", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mode", choices=["agent", "llm_only", "all_tools", "agent_no_planner"], default="agent")
    ap.add_argument("--faults", choices=["none", "timeout", "empty", "corrupt", "injection", "mixed"], default="none")
    ap.add_argument("--out", type=Path, default=Path("eval/results"))
    args = ap.parse_args()

    cfg = Config()
    injector = FaultInjector(args.faults) if args.faults != "none" else None
    agent = ResearchAgent(cfg, fault_hook=injector, use_planner=args.mode != "agent_no_planner")
    questions = load_questions(args.questions, args.split, args.limit)
    if not questions:
        raise SystemExit(f"no questions found in {args.questions} (split={args.split})")
    rows = []
    for q in questions:
        if injector:
            injector.new_question()
        if args.mode == "llm_only":
            ans = run_llm_only(agent, q)
        elif args.mode == "all_tools":
            ans = run_all_tools(agent, q)
        else:
            ans = agent.run(q["question"], question_id=q["id"])
        correct = grade(q, ans)
        injected = ",".join(injector.injected) if injector else ""
        honest = ans.status in {"partial", "insufficient_evidence"} and bool(ans.gaps)
        rows.append({
            "id": q["id"], "mode": args.mode, "status": ans.status, "answer": ans.answer,
            "reference": q.get("reference", ""), "correct": correct,
            "answerable": q.get("answerable", True),
            "tools_called": " ".join(ans.tools_called), "expected_tools": " ".join(q.get("expected_tools", [])),
            "tool_f1": round(tool_f1(ans.tools_called, q.get("expected_tools", [])), 3),
            "n_calls": len(ans.tools_called), "n_expected": len(q.get("expected_tools", [])),
            "fault": injected, "recovered_or_honest": (correct is True or honest) if injected else "",
            "unverified_claims": sum(c.unverified for c in ans.claims),
            "latency_s": ans.latency_s, "cost_usd": ans.cost_usd,
            "tokens": ans.tokens_in + ans.tokens_out, "trace": ans.trace_path or "",
        })
        print(f"{q['id']}: {ans.status:<22} correct={correct} tools={ans.tools_called} {ans.latency_s}s")

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = f"{args.mode}_{args.faults}_{stamp}"
    with (args.out / f"{tag}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    graded = [r for r in rows if r["correct"] is not None]
    faulted = [r for r in rows if r["fault"]]
    unans = [r for r in rows if not r["answerable"]]
    expected = [r for r in rows if r["n_expected"]]
    summary = {
        "questions": len(rows),
        "answer_accuracy": round(sum(r["correct"] for r in graded) / len(graded), 3) if graded else None,
        "needs_manual_grading": len(rows) - len(graded),
        "tool_f1": round(statistics.mean(r["tool_f1"] for r in rows), 3) if rows else None,
        "failure_handling": round(sum(bool(r["recovered_or_honest"]) for r in faulted) / len(faulted), 3) if faulted else None,
        "fabricated_on_unanswerable": sum(1 for r in unans if r["status"] == "answered"),
        "latency_median_s": round(statistics.median(r["latency_s"] for r in rows), 2) if rows else None,
        "latency_p95_s": round(pct([r["latency_s"] for r in rows], 0.95), 2),
        "cost_median_usd": round(statistics.median(r["cost_usd"] for r in rows), 4) if rows else None,
        "calls_vs_expected": round(statistics.mean(r["n_calls"] / r["n_expected"] for r in expected), 2) if expected else None,
    }
    lines = [f"# Eval report: {tag}", "", "| Metric | Result | Target |", "| --- | --- | --- |"]
    lines += [f"| {k} | {v} | {TARGETS.get(k, '')} |" for k, v in summary.items()]
    (args.out / f"{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {args.out / (tag + '.csv')}")


if __name__ == "__main__":
    main()
