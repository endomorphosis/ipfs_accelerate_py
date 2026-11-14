"""Real worker/MCP source selection, with an explicitly authored provider.

Count complete pipe frames and the rendered initial prompt in one bounded local
exchange. Provider history, reasoning and billed usage are not measured here.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_context_delivery_observation as delivery
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_policy as policy
from ipfs_accelerate_py.agent_supervisor.runtime.local_prompt_tokenizer import load_owner_prompt_tokenizer
from test.api.test_local_prompt_tokenizer import make_binding
from test.api import test_repository_source_mcp as protocol
from test.api.test_repository_source_router import route, _run


class RecordingPipe:
    """Observe the actual pipe bytes while preserving the ordinary client."""

    def __init__(self, stream, direction, frames):
        self.stream, self.direction, self.frames = stream, direction, frames

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def write(self, raw):
        count = self.stream.write(raw)
        assert count == len(raw)
        self.frames.append((self.direction, bytes(raw)))
        return count

    def readline(self, *args):
        raw = self.stream.readline(*args)
        if raw:
            self.frames.append((self.direction, bytes(raw)))
        return raw


def authored_source():
    def neighbors(start, stop):
        return "".join(
            f"def fixture_{i}(value):\n    return value + {i}\n\n" for i in range(start, stop))
    prefix = neighbors(0, 32)
    selected = "def square(value):\n    # Retained selected implementation.\n    return value * value\n"
    raw = (prefix + selected + "\n" + neighbors(32, 64)).encode()
    start = prefix.count("\n") + 1
    return raw, start, start + selected.count("\n") - 1


def exercise(binding, *, mode, bounds):
    """The authored action sequence is identical except for the read selector."""
    arguments = dict(path="module.py", representation="original")
    if mode == "line_range":
        arguments.update(start_line=bounds[0], end_line=bounds[1])
    with protocol.real_client(Path(binding["config_path"]), binding["config_sha256"]) as client:
        discovered = client.request("tools/list")
        assert any(tool["name"] == "repository_read" for tool in discovered["result"]["tools"])
        status, failed = client.call("repository_status")
        assert not failed and status["generation"] == 1
        candidates, failed = client.call("repository_context_query", generation=1, query="square", limit=1)
        assert not failed and candidates["candidates"]
        before, failed = client.call("repository_read", generation=1, **arguments)
        assert not failed and "return value * value\n" in before["content"]
        edited, failed = client.call("repository_replace_symbol", generation=1, path="module.py",
            symbol="square", representation="original", expected_sha256=before["source_sha256"],
            replacement="def square(value):\n    # Retained selected implementation.\n    return value * value + 1\n")
        assert not failed and edited["canonical_changed"] and edited["generation"] == 2
        stale, failed = client.call("repository_read", generation=1, **arguments)
        assert failed and stale == {"error": "stale_generation"}
        after, failed = client.call("repository_read", generation=2, **arguments)
        assert not failed and "return value * value + 1\n" in after["content"]
        assert after["source_sha256"] == edited["source_sha256"] != before["source_sha256"]
        if mode == "line_range":
            assert before["scope"] == after["scope"] == "line_range"
            assert before["omitted_lines"] > 0 and after["omitted_lines"] > 0
        else:
            assert "scope" not in before and "scope" not in after


def assert_retained_read_scope(receipt, mode):
    binding = receipt["repository_source_policy"]
    usage = binding["usage"]
    assert usage["successful_reads"] == 2 and usage["errors"] == 1
    assert usage["token_selection"]["read_count"] == 2
    assert usage["retrieval_observation"]["successful_queries"] == 1
    observed = usage["read_observation"]
    assert observed["complete"] is True
    assert observed["read_calls"] == 3 and observed["failed_reads"] == 1
    assert observed["generation_bindings_observed"] == 2
    assert observed["line_range_reads"] == (2 if mode == "line_range" else 0)
    assert observed["full_file_reads"] == (2 if mode == "full_file" else 0)
    assert usage["store_cleanup_complete"] is True
    assert not Path(binding["invocation_dir"]).exists()
    collected = delivery.collect(report={"provider_invocations": [receipt],
                                         "unreceipted_provider_attempt": False})
    retained = collected["repository_source_mcp"]
    assert retained["read_scope_complete"] is True
    assert retained["read_scope_counts"]["line_range_reads"] == observed["line_range_reads"]
    assert retained["observations"][0]["read_observation"] == observed
    assert retained["observations"][0]["token_selection_summary_valid"] is True
    assert retained["observations"][0]["retrieval_observation_valid"] is True
    assert collected["provider_usage_modified"] is False
    assert collected["total_token_savings_measured"] is False


def test_complete_authored_worker_exchange_compares_full_and_line_reads(
        route, tmp_path, monkeypatch, record_property):
    workspace, state = route
    raw, start, end = authored_source()
    artifact, digest, _ = make_binding(tmp_path, provider="codex_cli", model="pinned")
    tokenizer = load_owner_prompt_tokenizer(artifact=artifact, expected_sha256=digest,
                                           provider="codex_cli", model="pinned")
    frame_sets = []
    original_client = protocol.Client

    class ObservedClient(original_client):
        def __init__(self, process):
            frames = []
            frame_sets.append(frames)
            process.stdin = RecordingPipe(process.stdin, "request", frames)
            process.stdout = RecordingPipe(process.stdout, "reply", frames)
            super().__init__(process)

    monkeypatch.setattr(protocol, "Client", ObservedClient)
    results = {}
    for mode in ("full_file", "line_range"):
        (workspace / "module.py").write_bytes(raw)
        state["callback"] = lambda binding, mode=mode: exercise(binding, mode=mode, bounds=(start, end))
        _, receipt = _run(repository_source_policy=policy.POLICIES[1],
            prompt_tokenizer_artifact=artifact, prompt_tokenizer_sha256=digest)
        assert_retained_read_scope(receipt, mode)
        assert (workspace / "module.py").read_bytes() == raw.replace(
            b"return value * value\n", b"return value * value + 1\n")
        checked = subprocess.run([sys.executable, "-B", "-c",
            "from module import square, fixture_0, fixture_63; assert square(3) == 10; "
            "assert fixture_0(2) == 2; assert fixture_63(2) == 65"],
            cwd=workspace, capture_output=True, timeout=10)
        assert checked.returncode == 0, checked.stderr
        assert len(frame_sets) == len(results) + 1
        frames = frame_sets[-1]
        requests = [json.loads(frame) for direction, frame in frames if direction == "request"]
        assert requests[0]["method"] == "initialize"
        assert requests[1]["method"] == "notifications/initialized"
        assert any(request["method"] == "tools/list" for request in requests)
        replies = [json.loads(frame) for direction, frame in frames if direction == "reply"]
        assert len(requests) == 9 and len(replies) == 8
        assert sum(reply.get("result", {}).get("isError", False) for reply in replies) == 1
        prompt = state["seen"][-1][0]
        assert "start_line" in prompt and "end_line" in prompt
        results[mode] = {
            "requests": len(requests), "replies": len(replies),
            "request_bytes": sum(len(frame) for direction, frame in frames if direction == "request"),
            "reply_bytes": sum(len(frame) for direction, frame in frames if direction == "reply"),
            "rendered_prompt_bytes": len(prompt.encode()),
            "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_and_wire_local_tokens": tokenizer.count(prompt) + sum(
                tokenizer.count(frame.decode()) for _, frame in frames),
            "fixture_validation_passed": True,
            "canonical_result_sha256": hashlib.sha256((workspace / "module.py").read_bytes()).hexdigest(),
        }
        results[mode]["prompt_and_wire_bytes"] = sum(results[mode][key] for key in (
            "request_bytes", "reply_bytes", "rendered_prompt_bytes"))
    full, window = results["full_file"], results["line_range"]
    assert full["canonical_result_sha256"] == window["canonical_result_sha256"]
    assert full["rendered_prompt_sha256"] == window["rendered_prompt_sha256"]
    assert full["requests"] == window["requests"] and full["replies"] == window["replies"]
    assert full["prompt_and_wire_bytes"] > window["prompt_and_wire_bytes"]
    assert full["prompt_and_wire_local_tokens"] > window["prompt_and_wire_local_tokens"]
    record_property("source_working_set_session_comparison", json.dumps({
        "scope": "authored_worker_initial_prompt_and_complete_MCP_pipe_exchange",
        "arms": results, "provider_responses": "authored fixtures",
        "tokenizer_binding": tokenizer.binding(),
        "tool_discovery_and_guidance_included": True, "failed_read_included": True,
        "provider_internal_history_measured": False, "provider_usage_measured": False,
        "official_task_success_measured": False, "whole_task_token_savings_measured": False,
    }, sort_keys=True))


def test_failed_authored_provider_retains_real_range_use_and_cleanup(route, tmp_path, capsys):
    workspace, state = route
    raw, start, end = authored_source()
    (workspace / "module.py").write_bytes(raw)
    artifact, digest, _ = make_binding(tmp_path, provider="codex_cli", model="pinned")

    def failing_worker(binding):
        exercise(binding, mode="line_range", bounds=(start, end))
        raise TimeoutError("authored provider failed after actual range use")

    state["callback"] = failing_worker
    with pytest.raises(TimeoutError, match="authored provider failed"):
        _run(repository_source_policy=policy.POLICIES[1],
             prompt_tokenizer_artifact=artifact, prompt_tokenizer_sha256=digest)
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    receipt = next(row for row in rows if row.get("schema") == "router-implementation-invocation@1")
    assert receipt["status"] == "failed" and receipt["failure_phase"] == "provider_invocation"
    assert_retained_read_scope(receipt, "line_range")
