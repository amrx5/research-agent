"""Python execution for calculations.

docker mode (default): fresh container per call, no network, CPU/memory/pid limits,
read-only filesystem, non-root user. This is the mode for demos and evaluation.

local mode: a resource-limited subprocess for quick development only. It is NOT a
security boundary; never use it on untrusted questions.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import uuid

from pydantic import BaseModel, Field

from ..config import Config
from ..types import ErrorCode, ToolError
from .base import RawItem, Tool


class PythonArgs(BaseModel):
    code: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="Self-contained Python 3 code. print() every result you need, with units. numpy, sympy, pandas available. No network or files.",
    )


class PythonSandboxTool(Tool):
    name = "python"
    evidence_prefix = "C"
    description = (
        "Run Python code in an isolated sandbox and return what it prints. "
        "USE FOR: any arithmetic, percentages, unit conversion, statistics or formula evaluation, even simple ones; "
        "put the input numbers (with their evidence IDs in comments) directly in the code. "
        "DO NOT USE FOR: looking up facts (it has no internet and no files)."
    )
    Args = PythonArgs

    def __init__(self, config: Config):
        self.cfg = config
        self.timeout_s = config.sandbox_timeout_s + 5  # router timeout slightly above the sandbox's own
        if config.sandbox_mode not in {"docker", "local"}:
            raise ValueError("SANDBOX_MODE must be 'docker' or 'local'")

    def run(self, args: PythonArgs) -> list[RawItem]:
        if self.cfg.sandbox_mode == "docker":
            proc = self._run_docker(args.code)
        else:
            proc = self._run_local(args.code)
        return [
            {
                "source": "python",
                "title": "Python calculation",
                "content": proc.stdout.strip(),
                "relevance": None,
                "meta": {
                    "code": args.code,
                    "exit_code": proc.returncode,
                    "stderr": proc.stderr.strip()[-1500:],
                },
            }
        ]

    def _run_docker(self, code: str) -> subprocess.CompletedProcess:
        name = f"ra-sbx-{uuid.uuid4().hex[:10]}"
        cmd = [
            "docker", "run", "--rm", "-i", "--name", name,
            "--network", "none",
            "--memory", self.cfg.sandbox_memory, "--cpus", "1", "--pids-limit", "64",
            "--read-only", "--tmpfs", "/tmp:size=16m",
            "--user", "65534:65534",
            "--security-opt", "no-new-privileges",
            self.cfg.sandbox_image, "python", "-I", "-",
        ]
        try:
            return subprocess.run(cmd, input=code, capture_output=True, text=True, timeout=self.cfg.sandbox_timeout_s)
        except subprocess.TimeoutExpired as exc:
            subprocess.run(["docker", "kill", name], capture_output=True)
            raise ToolError(ErrorCode.TIMEOUT, f"code ran longer than {self.cfg.sandbox_timeout_s:.0f} s (infinite loop?)") from exc
        except FileNotFoundError as exc:
            raise ToolError(ErrorCode.INTERNAL, "docker not found; build the sandbox image or set SANDBOX_MODE=local for dev") from exc

    def _run_local(self, code: str) -> subprocess.CompletedProcess:
        def limits():  # applied in the child process (Unix only)
            import resource

            cpu = int(self.cfg.sandbox_timeout_s)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            resource.setrlimit(resource.RLIMIT_AS, (512 * 2**20, 512 * 2**20))

        with tempfile.TemporaryDirectory() as tmp:
            try:
                return subprocess.run(
                    [sys.executable, "-I", "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=self.cfg.sandbox_timeout_s,
                    cwd=tmp,
                    env={"PATH": os.environ.get("PATH", ""), "HOME": tmp},
                    preexec_fn=limits if os.name == "posix" else None,
                )
            except subprocess.TimeoutExpired as exc:
                raise ToolError(ErrorCode.TIMEOUT, f"code ran longer than {self.cfg.sandbox_timeout_s:.0f} s (infinite loop?)") from exc
