"""Retained provider commands fence rollback without granting new execution."""

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
)


@pytest.mark.parametrize("model", ["gpt-5.6-terra", "gpt-6.1-sol", "unrecognized-model"])
@pytest.mark.parametrize("legacy_preflight", [False, True])
def test_damaged_retained_command_preserves_model_recovery_fence(
    tmp_path, monkeypatch, model, legacy_preflight,
):
    daemon = object.__new__(PortalImplementationDaemon)
    daemon.state_path = tmp_path / "missing-state.json"
    # Model migration must preserve recovery custody even if a route flag is
    # missing from the retained event. Its orphaned JSON is diagnostic only.
    command = [
        "python", "-m", "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
        "--grok-failure-receipt-nonce", "a" * 64,
        "--codex-fallback-command-json",
        json.dumps(["codex", "exec", "-m", model]),
        json.dumps({"fallback_reasoning_effort": "high"}),
    ]
    if legacy_preflight:
        command.append("--canonical-legacy-preflight-route")
    monkeypatch.setattr(daemon, "_protected_attempt_event_latch", lambda **_: None)
    monkeypatch.setattr(daemon, "_protected_attempt_latched", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(daemon, "_iter_events", lambda: [{
        "type": "implementation_started", "task_id": "T-1", "attempt": 1,
        "command": command,
    }])

    protected, reservation = daemon._scoped_attempt_recovery_reservation(
        task_id="T-1", attempt=1, task_revision_cid="task:fixture",
    )

    assert protected is (not legacy_preflight and model in {"gpt-5.6-terra", "gpt-6.1-sol"})
    assert reservation is None  # Damaged retained evidence never admits execution.
