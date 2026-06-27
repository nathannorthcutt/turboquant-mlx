"""Tests for the prefill-keepalive feature (turboquant-serve).

Covers:
1. Flag extraction: `--prefill-keepalive` / `--prefill-keepalive-interval`
   peeled off argv, the rest forwarded; off by default.
2. `_KeepaliveSSEWriter`: passes all bytes through unchanged; mirrors an
   mlx_lm `: keepalive` SSE comment as a real `data:` chunk; throttles to the
   configured interval; ignores non-keepalive writes; never raises if the
   chunk builder fails.
3. Integration: `_patch_prefill_keepalive` wraps `APIHandler.handle_completion`
   and is a no-op for non-streaming requests.
"""

import io
import json

from turboquant_mlx.serve import (
    _extract_prefill_keepalive_args,
    _KeepaliveSSEWriter,
    _patch_prefill_keepalive,
)


# ── Flag extraction ─────────────────────────────────────────────────────────


def test_extract_no_flag_returns_none():
    cfg, remaining = _extract_prefill_keepalive_args(["--model", "foo", "--port", "8080"])
    assert cfg is None
    assert remaining == ["--model", "foo", "--port", "8080"]


def test_extract_keepalive_flag_default_interval():
    cfg, remaining = _extract_prefill_keepalive_args(["--model", "foo", "--prefill-keepalive"])
    assert cfg == {"interval": 10.0}
    assert remaining == ["--model", "foo"]


def test_extract_keepalive_custom_interval():
    cfg, remaining = _extract_prefill_keepalive_args(
        ["--prefill-keepalive", "--prefill-keepalive-interval", "3.5", "--model", "foo"]
    )
    assert cfg == {"interval": 3.5}
    assert remaining == ["--model", "foo"]


def test_extract_interval_without_flag_is_noop():
    # Interval alone (no --prefill-keepalive) must NOT enable the feature.
    cfg, remaining = _extract_prefill_keepalive_args(
        ["--prefill-keepalive-interval", "5", "--model", "foo"]
    )
    assert cfg is None
    assert remaining == ["--model", "foo"]


# ── _KeepaliveSSEWriter ─────────────────────────────────────────────────────


class _FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _chunk():
    return {"choices": [{"index": 0, "finish_reason": None,
                         "delta": {"role": "assistant"}}]}


def test_passthrough_and_mirror_on_keepalive():
    buf = io.BytesIO()
    clk = _FakeClock()
    w = _KeepaliveSSEWriter(buf, make_chunk=_chunk, interval=10.0, clock=clk)

    # A normal data write: passed through, no mirror.
    w.write(b'data: {"x":1}\n\n')
    assert buf.getvalue() == b'data: {"x":1}\n\n'

    # A keepalive comment: passed through AND mirrored as a real data chunk.
    buf.seek(0); buf.truncate()
    w.write(b": keepalive 512/21976\n\n")
    out = buf.getvalue().decode()
    assert out.startswith(": keepalive 512/21976\n\n")
    assert "data: " in out
    mirrored = out.split("data: ", 1)[1].strip()
    assert json.loads(mirrored) == _chunk()


def test_throttle_within_interval():
    buf = io.BytesIO()
    clk = _FakeClock()
    w = _KeepaliveSSEWriter(buf, make_chunk=_chunk, interval=10.0, clock=clk)

    w.write(b": keepalive 1/100\n\n")          # t=100 -> mirrors (last was 0)
    first = buf.getvalue().count(b"data: ")
    assert first == 1

    clk.t = 105.0                               # +5s < interval -> no mirror
    w.write(b": keepalive 2/100\n\n")
    assert buf.getvalue().count(b"data: ") == 1

    clk.t = 111.0                               # +11s >= interval -> mirror again
    w.write(b": keepalive 3/100\n\n")
    assert buf.getvalue().count(b"data: ") == 2


def test_non_keepalive_never_mirrors():
    buf = io.BytesIO()
    w = _KeepaliveSSEWriter(buf, make_chunk=_chunk, interval=0.0, clock=_FakeClock())
    w.write(b"data: real token\n\n")
    w.write(b": some other comment\n\n")
    assert buf.getvalue().count(b'"role": "assistant"') == 0


def test_chunk_builder_failure_is_swallowed():
    buf = io.BytesIO()

    def _boom():
        raise RuntimeError("no handler state yet")

    w = _KeepaliveSSEWriter(buf, make_chunk=_boom, interval=0.0, clock=_FakeClock())
    # Must not raise, and the original comment still went through.
    w.write(b": keepalive 1/10\n\n")
    assert buf.getvalue() == b": keepalive 1/10\n\n"


def test_getattr_delegates_to_wrapped():
    buf = io.BytesIO()
    w = _KeepaliveSSEWriter(buf, make_chunk=_chunk, interval=0.0)
    # e.g. .getvalue() is not defined on the writer -> delegates to BytesIO.
    w.write(b"abc")
    assert w.getvalue() == b"abc"


# ── Integration with the real APIHandler ────────────────────────────────────


def test_patch_is_noop_for_non_streaming():
    from mlx_lm.server import APIHandler

    orig = APIHandler.handle_completion
    try:
        _patch_prefill_keepalive(interval=10.0)
        patched = APIHandler.handle_completion
        assert patched is not orig

        # Build a bare handler shell (skip __init__) and drive the patched
        # method on a non-streaming "request": it must call straight through
        # to the original without touching wfile.
        h = APIHandler.__new__(APIHandler)
        h.stream = False
        called = {}

        def _fake_orig(self, request, stop_words):
            called["hit"] = (request, stop_words)
            return "ok"

        # Temporarily point the captured original at our spy by re-patching.
        APIHandler.handle_completion = orig  # restore
        _orig_ref = []
        _orig_ref.append(orig)
        # Re-apply patch so it wraps our spy.
        APIHandler.handle_completion = _fake_orig
        _patch_prefill_keepalive(interval=10.0)
        assert APIHandler.handle_completion(h, "req", ["stop"]) == "ok"
        assert called["hit"] == ("req", ["stop"])
    finally:
        APIHandler.handle_completion = orig
