"""Multi-tool research assistant agent (M2 design: plan -> act -> validate -> synthesize)."""
from .agent import ResearchAgent
from .config import Config
from .synthesizer import AgentAnswer

__all__ = ["ResearchAgent", "Config", "AgentAnswer"]
