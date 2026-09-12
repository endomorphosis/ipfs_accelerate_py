"""Native protected-marker producer/consumer compatibility and custody."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_agent_supervisor_implementation_protected_paths import (
    _persist_live_shaped_interrupted_attempt,
    _protected_git_worktree_daemon,
    _protected_git_worktree_supervisor,
    _stub_supervisor_maintenance_tail,
    _task,
)

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
)


def _native_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    daemon, repo, workspace, _ = _protected_git_worktree_daemon(tmp_path)
    task, lock, claim = _persist_live_shaped_interrupted_attempt(
        daemon,
        workspace=workspace,
    )
    monkeypatch.setattr(PortalImplementationDaemon, "_load_tasks", lambda _self: [task])
    supervisor = _protected_git_worktree_supervisor(daemon, repo)
    _stub_supervisor_maintenance_tail(supervisor, monkeypatch)
    marker = daemon._implementation_protected_active_snapshot_path()
    return daemon, supervisor, task, workspace, lock, claim, marker


@pytest.mark.parametrize("envelope", ["current_native", "exact_legacy"])
def test_native_marker_recovery_accepts_only_bound_known_envelopes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    envelope: str,
) -> None:
    daemon, supervisor, task, workspace, lock, claim, marker = _native_attempt(
        tmp_path, monkeypatch
    )
    payload = json.loads(marker.read_bytes())
    identity = daemon._identity_for_task(task)
    assert payload["canonical_task_key"] == identity.canonical_task_key
    assert payload["canonical_task_cid"] == identity.canonical_task_cid
    if envelope == "exact_legacy":
        # The baseline consumer explicitly accepted this exact eight-field v1
        # shape. Its independent state/claim/lifecycle checks remain required.
        del payload["canonical_task_key"]
        del payload["canonical_task_cid"]
        marker.write_text(json.dumps(payload), encoding="utf-8")
    result = supervisor.run_once(include_refill=False)
    assert "maintenance_blocked" not in result
    recovery = result["interrupted_implementation_reconciliation"]
    assert recovery["reconciled"] is True
    assert recovery["receipt_archived"] is True
    assert not marker.exists() and not lock.exists() and not claim.exists()
    lifecycle = daemon.worktree_lifecycle.load_workspace(workspace)
    assert lifecycle is not None and lifecycle.is_terminal
    assert lifecycle.terminal_reason == "controlled_shutdown_quiesced_owner"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_key",
        "missing_cid",
        "wrong_key",
        "wrong_cid",
        "cross_task_pair",
        "empty_key",
        "non_string_cid",
        "extra_authority",
        "legacy_extra_authority",
    ],
)
def test_native_marker_identity_refusal_retains_original_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    daemon, supervisor, _current_task, workspace, lock, claim, marker = _native_attempt(
        tmp_path, monkeypatch
    )
    payload = json.loads(marker.read_bytes())
    if mutation == "missing_key":
        del payload["canonical_task_key"]
    elif mutation == "missing_cid":
        del payload["canonical_task_cid"]
    elif mutation == "wrong_key":
        payload["canonical_task_key"] += ":other"
    elif mutation == "wrong_cid":
        payload["canonical_task_cid"] = "sha256:" + "1" * 64
    elif mutation == "cross_task_pair":
        other = daemon._identity_for_task(_task(outputs=["src/different.py"]))
        assert other.canonical_task_key != payload["canonical_task_key"]
        assert other.canonical_task_cid != payload["canonical_task_cid"]
        payload["canonical_task_key"] = other.canonical_task_key
        payload["canonical_task_cid"] = other.canonical_task_cid
    elif mutation == "empty_key":
        payload["canonical_task_key"] = ""
    elif mutation == "non_string_cid":
        payload["canonical_task_cid"] = {"cid": payload["canonical_task_cid"]}
    elif mutation == "extra_authority":
        payload["callback_settlement_authority"] = True
    elif mutation == "legacy_extra_authority":
        del payload["canonical_task_key"]
        del payload["canonical_task_cid"]
        payload["callback_settlement_authority"] = True
    marker.write_text(json.dumps(payload), encoding="utf-8")
    lifecycle = daemon.worktree_lifecycle.workspace_path_for(workspace)
    retained = {
        path: path.read_bytes()
        for path in (marker, daemon.state_path, claim, lifecycle)
    }
    result = supervisor.run_once(include_refill=False)
    assert result["maintenance_blocked"] is True
    recovery = result["interrupted_implementation_reconciliation"]
    assert recovery["blocked"] is True and recovery["reconciled"] is False
    assert "state/marker identity differs" in recovery["error"]
    assert lock.exists()
    assert {path: path.read_bytes() for path in retained} == retained
    assert not (
        daemon.state_path.parent / "interrupted-implementation-recovery.json"
    ).exists()
    events = daemon._iter_merge_lifecycle_events(require_canonical_raw=True)
    assert not any(
        event.get("type") == "implementation_shutdown_recovery_closed"
        for event in events
    )
