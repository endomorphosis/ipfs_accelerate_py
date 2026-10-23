"""Capacity telemetry must not interpret task output or usage tokens as errors."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner import GROK_QUOTA_RECEIPT_SCHEMA
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as M


QUOTA = "API error (status 402 Payment Required): Grok Build usage balance exhausted"
COMMAND = [sys.executable, "-m", "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner",
           "--model", "grok-4.7"]


@pytest.fixture
def daemon(tmp_path):
    board = tmp_path / "tasks.todo.md"
    board.write_text("# Synthetic capacity characterization\n")
    return M.PortalImplementationDaemon(
        todo_path=board, state_path=tmp_path / "state" / "tasks.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl", repo_root=tmp_path,
        worktree_root=tmp_path / "worktrees",
    )


def classify(daemon, path: Path, text: str):
    path.write_text(text)
    path.chmod(0o600)
    return daemon._provider_capacity_failure_from_log(path, command=COMMAND, returncode=1)


@pytest.mark.parametrize("frame", (
    {"type": "usage", "opaque_usage_token": "synthetic-XAI" + "x" * 500 + "402"},
    {"type": "usage", "error": {"message": QUOTA}},
    {"type": "thought", "data": "The fixture must handle " + QUOTA},
    {"type": "tool_call", "rawInput": {"content": QUOTA}},
    {"type": "tool_call_update", "status": "completed", "rawOutput": QUOTA},
    {"type": "unknown_future_frame", "message": QUOTA},
    {"type": "end", "message": "Source fixture says " + QUOTA},
    {"type": "error", "diagnostic_context": {"source_text": QUOTA}, "message": "max turns reached"},
))
def test_non_error_stream_payloads_cannot_manufacture_quota(daemon, tmp_path, frame):
    # This matches the actual failure shape: an encoded usage payload happens
    # to contain XAI then402, followed by an ordinary configured turn-budget
    # terminal. It must remain a spent failed attempt, without a capacity hold.
    text = "\n".join(map(json.dumps, (frame, {"type": "max_turns_reached", "num_turns": 32}, {"type": "end"})))
    failure = classify(daemon, tmp_path / "attempt.log", text + "\nError: max turns reached\n")
    assert failure["exhausted"] is False
    assert not failure.get("retained_callback_required")
    assert failure.get("reason") == ""


@pytest.mark.parametrize("frame", (
    {"type": "error", "message": QUOTA},
    {"type": "error", "error": {"message": QUOTA}},
    {"type": "error", "code": "usage_limit_reached"},
    {"schema": GROK_QUOTA_RECEIPT_SCHEMA, "provider": "grok_cli",
     "failure_kind": "quota_or_balance_exhausted", "message": "Grok Build usage balance exhausted"},
))
def test_genuine_error_diagnostics_remain_non_refundable_after_dispatch(daemon, tmp_path, frame):
    text = json.dumps({"type": "tool_call_update", "status": "completed"}) + "\n" + json.dumps(frame) + "\n"
    failure = classify(daemon, tmp_path / "quota.log", text)
    assert failure["exhausted"] is True
    assert failure["pre_dispatch_refusal_verified"] is False
    assert failure["retained_callback_required"] is True
    assert failure["execution_observation"]["task_execution_observed"] is True


def test_plain_provider_stderr_still_classifies_without_refund(daemon, tmp_path):
    failure = classify(daemon, tmp_path / "stderr.log", QUOTA + "\n")
    assert failure["exhausted"] and failure["retained_callback_required"]
    assert failure["pre_dispatch_refusal_verified"] is False


def test_partial_first_tail_frame_cannot_become_plaintext_quota(daemon, tmp_path, monkeypatch):
    monkeypatch.setattr(M, "PROVIDER_CAPACITY_LOG_TAIL_BYTES", 200)
    text = json.dumps({"type": "usage", "opaque_usage_token": "z" * 1000 + "XAI" + "z" * 20 + "402" + "z" * 20})
    text += "\n" + json.dumps({"type": "max_turns_reached", "num_turns": 32}) + "\nError: max turns reached\n"
    failure = classify(daemon, tmp_path / "bounded.log", text)
    assert failure["exhausted"] is False


@pytest.mark.parametrize("text", (
    '{"type":"usage","opaque_usage_token":"XAI-402\n',
    '{"type":"error","message":"' + QUOTA + '"\n',
    json.dumps([QUOTA]) + "\n",
    json.dumps(QUOTA) + "\n",
    "Command: grok --synthetic-example '" + QUOTA + "'\n",
))
def test_malformed_or_metadata_frames_are_not_plain_error_evidence(daemon, tmp_path, text):
    assert classify(daemon, tmp_path / "malformed.log", text)["exhausted"] is False


def test_capacity_evidence_contains_only_error_diagnostics(daemon, tmp_path):
    text = json.dumps({"type": "tool_call", "rawInput": {"content": "source-only " + QUOTA}}) + "\n"
    text += json.dumps({"type": "usage", "opaque_usage_token": "XAI-x-402"}) + "\n"
    text += json.dumps({"type": "error", "message": QUOTA}) + "\n"
    failure = classify(daemon, tmp_path / "evidence.log", text)
    assert failure["exhausted"] is True
    assert failure["evidence"] == [QUOTA]


def test_protected_route_keeps_its_original_typed_audit(daemon, tmp_path, monkeypatch):
    path = tmp_path / "protected.log"
    text = json.dumps({"type": "usage", "opaque_usage_token": "XAI-x-402"}) + "\n"
    path.write_text(text)
    path.chmod(0o600)
    command = [*COMMAND, "--agent-implementation-route-json", '{"signed":"fixture"}']
    seen = []
    result = {"exhausted": False, "reason": "protected-native-audit"}

    def audit(**kwargs):
        seen.append(kwargs)
        return result

    monkeypatch.setattr(daemon, "_protected_provider_effect_audit", audit)
    monkeypatch.setattr(M, "_grok_capacity_diagnostic_text", lambda *_a: pytest.fail("legacy filter touched protected route"))
    assert daemon._provider_capacity_failure_from_log(path, command=command, returncode=1) is result
    assert seen == [{"repo_root": daemon.repo_root, "command_items": command, "receipt_text": text, "returncode": 1}]
