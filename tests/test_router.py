import time

from fakes import FakeTool, doc

from agent.router import EvidenceStore, ToolRouter
from agent.types import ErrorCode, ToolError


def make(cfg, *tools, **kw):
    return ToolRouter(list(tools), cfg, sleep=lambda s: None, **kw), EvidenceStore()


def test_unknown_tool_and_invalid_args(cfg):
    r, s = make(cfg, FakeTool("web_search", "W", []))
    assert r.call("nope", {}, s).error == ErrorCode.UNKNOWN_TOOL
    assert r.call("web_search", {"query": ""}, s).error == ErrorCode.INVALID_ARGS


def test_success_assigns_ids(cfg):
    r, s = make(cfg, FakeTool("web_search", "W", [[doc("u1", "a"), doc("u2", "b")]]))
    out = r.call("web_search", {"query": "x y"}, s)
    assert out.ok and [e.id for e in out.evidence] == ["W1", "W2"]


def test_duplicate_call_blocked(cfg):
    r, s = make(cfg, FakeTool("web_search", "W", [[doc("u1", "a")], [doc("u1", "a")]]))
    assert r.call("web_search", {"query": "same"}, s).ok
    assert r.call("web_search", {"query": "same"}, s).error == ErrorCode.DUPLICATE_CALL


def test_retry_then_success(cfg):
    t = FakeTool("web_search", "W", [ToolError(ErrorCode.RATE_LIMITED, "429", retryable=True), [doc("u1", "a")]])
    r, s = make(cfg, t)
    out = r.call("web_search", {"query": "x y"}, s)
    assert out.ok and out.attempts == 2


def test_non_retryable_fails_fast(cfg):
    t = FakeTool("web_search", "W", [ToolError(ErrorCode.HTTP_ERROR, "401", retryable=False), [doc("u1", "a")]])
    r, s = make(cfg, t)
    out = r.call("web_search", {"query": "x y"}, s)
    assert out.error == ErrorCode.HTTP_ERROR and out.attempts == 1


def test_timeout(cfg):
    cfg.max_attempts = 2
    slow = FakeTool("web_search", "W", [lambda a: time.sleep(0.5) or [doc("u", "x")]] * 2, timeout_s=0.1)
    r, s = make(cfg, slow)
    out = r.call("web_search", {"query": "x y"}, s)
    assert out.error == ErrorCode.TIMEOUT and out.attempts == 2


def test_circuit_breaker_disables_tool(cfg):
    cfg.max_attempts = 1
    err = ToolError(ErrorCode.HTTP_ERROR, "500", retryable=True)
    r, s = make(cfg, FakeTool("web_search", "W", [err, err, [doc("u", "x")]]))
    r.call("web_search", {"query": "a1"}, s)
    r.call("web_search", {"query": "a2"}, s)
    assert r.call("web_search", {"query": "a3"}, s).error == ErrorCode.TOOL_DISABLED


def test_cache_readwrite_then_replay(cfg):
    cfg.cache_mode = "readwrite"
    t = FakeTool("web_search", "W", [[doc("u1", "cached text")]], cacheable=True)
    r, s = make(cfg, t)
    assert r.call("web_search", {"query": "x y"}, s).ok
    cfg.cache_mode = "replay"
    r.reset()
    out = r.call("web_search", {"query": "x y"}, s)
    assert out.ok and out.from_cache and len(t.runs) == 1
    assert r.call("web_search", {"query": "never seen"}, s).error == ErrorCode.CACHE_MISS


def test_fault_hook(cfg):
    r, s = make(cfg, FakeTool("web_search", "W", [[doc("u1", "a")]]), fault_hook=lambda n, a, raw: [])
    assert r.call("web_search", {"query": "x y"}, s).error == ErrorCode.EMPTY_RESULT
