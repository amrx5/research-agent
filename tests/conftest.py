import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.config import Config  # noqa: E402


@pytest.fixture
def cfg(tmp_path) -> Config:
    c = Config()
    c.cache_mode = "off"
    c.cache_dir = tmp_path / "cache"
    c.trace_dir = tmp_path / "traces"
    c.backoff_base_s = 0.0
    c.sandbox_mode = "local"
    c.sandbox_timeout_s = 3.0
    c.price_in_per_mtok, c.price_out_per_mtok = 3.0, 15.0
    return c
