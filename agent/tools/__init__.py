from .base import RawItem, Tool

__all__ = ["RawItem", "Tool", "default_tools"]


def default_tools(config) -> list[Tool]:
    """Build the three production tools. Imports are local so a missing optional
    dependency only breaks the tool that needs it."""
    from .paper_retrieval import PaperSearchTool
    from .python_sandbox import PythonSandboxTool
    from .web_search import WebSearchTool

    return [WebSearchTool(config), PaperSearchTool(config), PythonSandboxTool(config)]
