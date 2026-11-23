"""Command line entry point.

    python -m agent.cli ask "your question"
    python -m agent.cli ask "..." --no-planner --json
"""
from __future__ import annotations

import json
from dataclasses import asdict

import typer

from .agent import ResearchAgent
from .config import Config

app = typer.Typer(add_completion=False, help="Multi-tool research assistant agent")


@app.command()
def ask(
    question: str,
    no_planner: bool = typer.Option(False, help="Skip the planning step (for ablation)."),
    as_json: bool = typer.Option(False, "--json", help="Print the full answer object as JSON."),
):
    agent = ResearchAgent(Config(), use_planner=not no_planner)
    ans = agent.run(question, question_id="cli")
    if as_json:
        d = asdict(ans)
        d["evidence"] = [e.__dict__ for e in ans.evidence]
        typer.echo(json.dumps(d, indent=2, default=str))
        return
    typer.echo(ans.to_markdown())
    typer.echo(f"\n---\ntools: {' -> '.join(ans.tools_called) or 'none'} | {ans.latency_s}s | "
               f"{ans.tokens_in}+{ans.tokens_out} tokens | ${ans.cost_usd} | trace: {ans.trace_path}")


@app.command()
def tools():
    """List the tools and the descriptions the model sees."""
    from .tools import default_tools

    for t in default_tools(Config()):
        typer.echo(f"{t.name}\n  {t.description}\n")


if __name__ == "__main__":
    app()
