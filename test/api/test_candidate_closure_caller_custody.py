"""Actual Portal public-call cleanup boundary for an incomplete candidate handoff."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.candidate_rejection_closure import (
    CandidateClosureObservationUnknown,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTaskState,
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


@pytest.mark.parametrize("mode", ["unknown", "diagnostic_append_failure", "ordinary"])
def test_public_portal_keeps_candidate_handoff_custody(tmp_path, monkeypatch, mode):
    unknown = mode != "ordinary"
    daemon = _daemon(tmp_path, ephemeral=True)
    _install_fixture_gate(daemon, monkeypatch)
    captured = {}
    released = []
    selected = []
    reconcile = daemon._reconcile_unselected_implementation_dispatch_intents

    def observe_reconcile(*args, **kwargs):
        selected.append(kwargs.get("selected"))
        return reconcile(*args, **kwargs)

    monkeypatch.setattr(
        daemon,
        "_reconcile_unselected_implementation_dispatch_intents",
        observe_reconcile,
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

    def provider(command, **kwargs):
        captured["provider_called"] = True
        (Path(kwargs["cwd"]) / "candidate.txt").write_text(
            "rejected fixture candidate\n"
        )
        return subprocess.CompletedProcess(command, 0)

    if mode == "diagnostic_append_failure":
        record = daemon._record_event

        def fail_diagnostic(event, *args, **kwargs):
            if event == "candidate_rejection_cleanup_unverified":
                raise OSError("fixture diagnostic append unavailable")
            return record(event, *args, **kwargs)

        monkeypatch.setattr(daemon, "_record_event", fail_diagnostic)

    preserve = daemon._preserve_failed_validation_worktree

    def incomplete(workspace, *_a, **_k):
        captured["workspace"] = Path(workspace)
        captured["lifecycle"] = daemon._active_worktree_lifecycle
        captured["state"] = PortalTaskState.load(daemon.state_path)
        if unknown:
            return preserve(
                workspace,
                *_a,
                **{
                    **_k,
                    "candidate_cleanup_required": True,
                    "candidate_cleanup_evidence": None,
                },
            )
        raise RuntimeError("ordinary preservation fixture error")

    # Upstream analytical receipts are disposable fixture authority. The real
    # residual gate, sealed invocation checks and public Portal caller remain.
    monkeypatch.setattr(module, "run_process_group_stream", provider)
    monkeypatch.setattr(daemon, "_preserve_failed_validation_worktree", incomplete)
    raised = None
    try:
        daemon.run_once()
    except CandidateClosureObservationUnknown as exc:
        raised = exc
    assert captured.get("provider_called"), "actual provider caller must be reached"
    assert "workspace" in captured, (
        "actual validation/preservation caller must be reached"
    )
    after = PortalTaskState.load(daemon.state_path)
    if not unknown:
        assert raised is None
        assert after.implementation_in_progress is False
        assert after.last_implementation_finished_at
        assert released and selected[-1] is None
        assert daemon._active_worktree_lifecycle is None
        return
    assert raised is not None
    if mode == "diagnostic_append_failure":
        assert isinstance(raised.__cause__, OSError)
    assert after.implementation_in_progress is True
    assert after.active_phase == captured["state"].active_phase
    assert after.active_provider_runner == captured["state"].active_provider_runner
    assert not after.last_implementation_finished_at
    assert released == []
    assert selected and selected[-1] is not None
    assert captured["workspace"].is_dir()
    assert daemon._active_worktree_lifecycle == captured["lifecycle"]
    assert (
        daemon.worktree_lifecycle.load_workspace(captured["workspace"])
        == captured["lifecycle"]
    )
    events = [json.loads(line) for line in daemon.events_path.read_text().splitlines()]
    assert not any(
        event["type"]
        in {"implementation_finished", "task_completed", "implementation_exception"}
        for event in events
    )
