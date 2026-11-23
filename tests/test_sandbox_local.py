"""Local-mode sandbox tests (Docker mode is exercised manually / in CI with Docker)."""
import pytest

from agent.tools.python_sandbox import PythonArgs, PythonSandboxTool
from agent.types import ErrorCode, ToolError
from agent.validator import Validator


def test_runs_code(cfg):
    out = PythonSandboxTool(cfg).run(PythonArgs(code="print(round((41.8-28.4)/28.4*100, 2))"))
    assert out[0]["content"] == "47.18" and out[0]["meta"]["exit_code"] == 0


def test_error_becomes_code_error(cfg):
    raw = PythonSandboxTool(cfg).run(PythonArgs(code="1/0"))
    r = Validator(cfg).validate("python", raw)
    assert r.error == ErrorCode.CODE_ERROR and "ZeroDivisionError" in r.message


def test_infinite_loop_times_out(cfg):
    cfg.sandbox_timeout_s = 1.0
    with pytest.raises(ToolError) as e:
        PythonSandboxTool(cfg).run(PythonArgs(code="while True: pass"))
    assert e.value.code == ErrorCode.TIMEOUT
