"""Actual image-sized authored CLI frames; no network or model calls."""
import json
import os
import subprocess
import sys

import pytest

from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.cli_runtime.cli_metadata import get_last_cli_observation
from ipfs_accelerate_py.cli_runtime.grok_stream_capture import (
    MAX_FRAME_BYTES, MAX_EXPLICIT_FRAME_BYTES, GrokStreamParser,
    capture_grok_stream, validate_max_frame_bytes,
)


def command(tmp_path, *, body_bytes=320_000):
    script = tmp_path / "authored_image_cli.py"
    script.write_text("import json\n"
        f"print(json.dumps({{'type':'tool_call_update','status':'completed','rawOutput': 'private-image:' + 'A'*{body_bytes}}}), flush=True)\n"
        "print(json.dumps({'type':'text','data':'observed result'}), flush=True)\n"
        "print(json.dumps({'type':'end','stopReason':'end_turn'}), flush=True)\n")
    return [sys.executable, str(script)]


def test_image_sized_frame_requires_explicit_budget_and_keeps_private_payload_out(tmp_path):
    argv = command(tmp_path)
    ordinary = capture_grok_stream(argv, env=os.environ.copy(), timeout=5)
    assert ordinary.progress["oversized_frames"] == 1
    assert ordinary.progress["stream_tainted"] is True
    assert ordinary.progress["response_usable"] is False
    assert ordinary.response_text == ""
    image = capture_grok_stream(argv, env=os.environ.copy(), timeout=5, max_frame_bytes=512 * 1024)
    assert image.progress["oversized_frames"] == 0
    assert image.progress["response_usable"] is True
    assert image.response_text == "observed result"
    assert "private-image" not in json.dumps(image.progress)
    assert "private-image" not in image.stdout
    assert len(json.dumps(image.progress)) < 6000


def test_explicit_budget_remains_a_hard_bound(tmp_path):
    result = capture_grok_stream(command(tmp_path, body_bytes=600_000), env=os.environ.copy(),
                                timeout=5, max_frame_bytes=512 * 1024)
    assert result.progress["oversized_frames"] == 1
    assert result.progress["stream_tainted"] is True
    assert result.progress["response_usable"] is False


@pytest.mark.parametrize("value", [0, -1, True, False, None, 1.0, "524288", float("inf"), MAX_EXPLICIT_FRAME_BYTES + 1])
def test_invalid_budget_refused_before_child(monkeypatch, value):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("child started"))
    with pytest.raises(ValueError):
        capture_grok_stream(["authored"], env={}, timeout=1, max_frame_bytes=value)
    with pytest.raises(ValueError):
        GrokStreamParser(max_frame_bytes=value)


def test_public_bounds_have_unchanged_default_and_finite_explicit_ceiling():
    assert MAX_FRAME_BYTES == 256 * 1024
    assert MAX_EXPLICIT_FRAME_BYTES == 8 * 1024 * 1024
    assert validate_max_frame_bytes(MAX_EXPLICIT_FRAME_BYTES) == MAX_EXPLICIT_FRAME_BYTES


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(llm_router, "find_grok_cli", lambda: "/fixture/grok")
    monkeypatch.setattr(llm_router, "_cli_available", lambda _: True)
    return llm_router._get_grok_cli_provider()


def test_router_forwards_explicit_image_budget_through_real_capture(provider, tmp_path):
    result = provider.generate("private prompt", timeout=5, grok_cli_cmd=command(tmp_path),
                               grok_stream_progress=True, grok_stream_max_frame_bytes=512 * 1024)
    assert result == "observed result"
    progress = get_last_cli_observation("grok_cli")["native_grok_progress"]
    assert progress["response_usable"] is True and progress["oversized_frames"] == 0


@pytest.mark.parametrize("options", [
    {"grok_stream_max_frame_bytes": 512 * 1024},
    {"grok_stream_max_frame_bytes": True, "grok_stream_progress": True},
    {"grok_stream_max_frame_bytes": MAX_EXPLICIT_FRAME_BYTES + 1, "grok_stream_progress": True},
    {"grok_stream_max_frame_bytes": None, "grok_stream_progress": True},
])
def test_router_rejects_invalid_or_nonstreaming_opt_in_before_spawn(provider, monkeypatch, options):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("child started"))
    with pytest.raises(ValueError):
        provider.generate("private prompt", **options)
