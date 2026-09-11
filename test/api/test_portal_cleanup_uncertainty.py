from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTaskState,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime import (
    ProcessGroupCleanupUnverified,
)


def _daemon(tmp_path: Path, *, ephemeral: bool) -> PortalImplementationDaemon:
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (
        ["init", "-q"],
        ["config", "user.email", "test@example.invalid"],
        ["config", "user.name", "Cleanup Fixture"],
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    (repo / "README.md").write_text("fixture\n")
    (repo / "todo.md").write_text(
        "# Tasks\n\n## ACCEL-001 Preserve uncertain provider custody\n\n"
        "- Status: todo\n- Completion: manual\n- Priority: P0\n- Track: ops\n"
        "- Depends on:\n- Outputs:\n- Validation:\n"
        "- Acceptance: Preserve unknown process-group custody.\n"
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-qm", "fixture"], cwd=repo, check=True, capture_output=True
    )
    return PortalImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=tmp_path / "state" / "task_state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        implement=True,
        implementation_command="python",
        use_ephemeral_worktree=ephemeral,
        worktree_root=tmp_path / "worktrees" if ephemeral else repo,
        worktree_pool_enabled=False,
    )


def _install_fixture_gate(daemon, monkeypatch):
    """Supply disposable typed analytical receipts; retain real gate/handoff checks."""
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
        content_identity,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_disposition import (
        ImplementationForestRoots,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.pre_implementation_provider_gate import (
        evaluate_provider_gate,
    )
    from test.api.test_agent_supervisor_residual_provider_invocation import _seal

    def gate(*, task, attempt, worktree_path):
        tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=worktree_path, text=True
        ).strip()
        roots = ImplementationForestRoots(
            repository_id="repository:cleanup-fixture",
            repository_forest_cid=content_identity({"tree": tree}),
            git_tree_id=tree,
            policy_root=content_identity({"fixture": "policy"}),
            dirty_overlay_cid=content_identity({"fixture": "overlay"}),
            capability_catalog_root=content_identity({"fixture": "capabilities"}),
            configuration_root=content_identity({"fixture": "config"}),
        )
        packet = _seal(
            task_id=task.task_id,
            repository_id=roots.repository_id,
            tree_id=tree,
            write_paths=("README.md",),
        )
        task_cid = daemon._canonical_ref(task)
        receipts = {}
        for kind in ("planner", "doctor", "obligation", "logic", "repair"):
            body = {
                "schema": "ipfs_accelerate_py/agent-supervisor/authority-receipt@1",
                "receipt_kind": kind,
                "task_cid": task_cid,
                "repository_forest_cid": roots.repository_forest_cid,
            }
            receipts[kind] = {**body, "content_id": content_identity(body)}
        ids = {k: v["content_id"] for k, v in receipts.items()}
        decision = evaluate_provider_gate(
            task_cid=task_cid,
            forest_roots=roots,
            attempt=attempt,
            residual_packet_cid=packet.packet_id,
            authority_receipt_cids=ids,
            obligation_graph_cid=ids["obligation"],
            plan_cid=ids["planner"],
            doctor_cid=ids["doctor"],
            authority_receipt_resolver=lambda cid: next(
                (v for v in receipts.values() if v["content_id"] == cid), None
            ),
        )
        assert decision.provider_authorized and not decision.skip_provider
        return {
            "skip_provider": False,
            "provider_authorized": True,
            "disposition": decision.disposition.value,
            "reason_code": decision.reason_code,
            "receipt_cid": decision.receipt_cid,
            "event": decision.to_event_payload(task_id=task.task_id, attempt=attempt),
            "_decision": decision,
            "_residual_packet": packet,
        }

    monkeypatch.setattr(daemon, "_evaluate_pre_implementation_provider_gate", gate)


@pytest.mark.parametrize("ephemeral", [True])
def test_public_portal_pass_preserves_cleanup_uncertainty(
    tmp_path, monkeypatch, ephemeral
):
    daemon = _daemon(tmp_path, ephemeral=ephemeral)
    _install_fixture_gate(daemon, monkeypatch)
    captured = {}
    released = []
    reconciled = []
    original_reconcile = daemon._reconcile_unselected_implementation_dispatch_intents

    def reconcile(*args, **kwargs):
        reconciled.append(kwargs.get("selected"))
        return original_reconcile(*args, **kwargs)

    monkeypatch.setattr(
        daemon, "_reconcile_unselected_implementation_dispatch_intents", reconcile
    )
    for name in (
        "_release_implementation_lock",
        "_release_implementation_task_claim",
        "_release_implementation_resource_claims",
    ):
        original = getattr(daemon, name)

        def release(*args, _name=name, _original=original, **kwargs):
            released.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(daemon, name, release)

    def provider(*args, **kwargs):
        state = PortalTaskState.load(daemon.state_path)
        state.active_provider_runner = {"fixture_retained_process_group": "unknown"}
        state.save(daemon.state_path)
        captured["runner"] = dict(state.active_provider_runner)
        captured["workspace"] = Path(kwargs["cwd"])
        captured["lifecycle"] = daemon._active_worktree_lifecycle
        lock_path = daemon._implementation_lock_path()
        lock_metadata = json.loads(lock_path.read_bytes())
        claim_path = daemon._implementation_task_claim_path(
            lock_metadata["task_id"],
            canonical_task_cid=lock_metadata["canonical_task_cid"],
        )
        captured["owned_files"] = {
            path: (path.stat().st_ino, path.read_bytes())
            for path in (lock_path, claim_path)
        }
        raise ProcessGroupCleanupUnverified("process_group_probe_unavailable")

    monkeypatch.setattr(module, "run_process_group_stream", provider)
    raised = None
    try:
        daemon.run_once()
    except ProcessGroupCleanupUnverified as exc:
        raised = exc

    assert captured, "fixture must reach the actual provider caller"
    assert isinstance(raised, ProcessGroupCleanupUnverified)
    state = PortalTaskState.load(daemon.state_path)
    assert state.implementation_in_progress is True
    assert state.active_provider_runner == captured["runner"]
    assert state.last_implementation_finished_at == ""
    assert released == []
    for path, (inode, body) in captured["owned_files"].items():
        assert path.stat().st_ino == inode
        assert path.read_bytes() == body
    assert reconciled and reconciled[-1] is not None
    assert captured["workspace"].is_dir()
    if ephemeral:
        assert daemon._active_worktree_lifecycle == captured["lifecycle"]
        assert (
            daemon.worktree_lifecycle.load_workspace(str(captured["workspace"]))
            is not None
        )
    events = [json.loads(line) for line in daemon.events_path.read_text().splitlines()]
    assert not any(
        item["type"]
        in {"implementation_finished", "task_completed", "implementation_exception"}
        for item in events
    )


@pytest.mark.parametrize("ephemeral", [True])
@pytest.mark.parametrize("outcome", ["exception", "failure", "success"])
def test_public_portal_pass_retains_ordinary_finish_semantics(
    tmp_path, monkeypatch, ephemeral, outcome
):
    daemon = _daemon(tmp_path, ephemeral=ephemeral)
    _install_fixture_gate(daemon, monkeypatch)
    calls = []

    def provider(command, **kwargs):
        calls.append(kwargs["cwd"])
        if outcome == "exception":
            raise RuntimeError("ordinary fixture error")
        return subprocess.CompletedProcess(command, int(outcome == "failure"))

    monkeypatch.setattr(module, "run_process_group_stream", provider)
    result = daemon.run_once()
    assert calls, "fixture must reach the actual provider caller"
    # A successful provider exit with no patch still reaches the existing
    # proposal gate. It does not acquire task-completion authority.
    assert result["implementation_result"]["returncode"] == (
        78 if outcome == "success" else 1
    )
    state = PortalTaskState.load(daemon.state_path)
    assert state.implementation_in_progress is False
    assert state.active_provider_runner == {}
    assert state.last_implementation_finished_at
    assert daemon._active_worktree_lifecycle is None
    if outcome == "success":
        events = [
            json.loads(line) for line in daemon.events_path.read_text().splitlines()
        ]
        assert any(
            item.get("type") == "implementation_proposal_rejected"
            or (
                item.get("accepted") is False
                and "empty_patch" in item.get("reason_codes", [])
            )
            for item in events
        )


def test_non_ephemeral_dispatch_keeps_existing_admission_denial(tmp_path, monkeypatch):
    daemon = _daemon(tmp_path, ephemeral=False)
    monkeypatch.setattr(
        module,
        "run_process_group_stream",
        lambda *a, **k: pytest.fail("direct dispatch is forbidden"),
    )
    result = daemon.run_once()["implementation_result"]
    assert result["reason"] == "residual_provider_requires_isolated_fenced_worktree"
    assert result["provider_dispatched"] is False
