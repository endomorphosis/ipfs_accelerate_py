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


@pytest.mark.parametrize("ephemeral", [False, True])
def test_public_portal_pass_preserves_cleanup_uncertainty(
    tmp_path, monkeypatch, ephemeral
):
    daemon = _daemon(tmp_path, ephemeral=ephemeral)
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


@pytest.mark.parametrize("ephemeral", [False, True])
@pytest.mark.parametrize("outcome", ["exception", "failure", "success"])
def test_public_portal_pass_retains_ordinary_finish_semantics(
    tmp_path, monkeypatch, ephemeral, outcome
):
    daemon = _daemon(tmp_path, ephemeral=ephemeral)
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


@pytest.mark.parametrize("uncertain", [False, True])
def test_inline_rescue_preserves_typed_unknown_instead_of_terminal_failure(
    tmp_path, monkeypatch, uncertain
):
    from ipfs_accelerate_py.agent_supervisor.validation import (
        implementation_auto_rescue as rescue,
    )

    daemon = _daemon(tmp_path, ephemeral=False)
    task = daemon._load_tasks()[0]
    state = PortalTaskState(
        implementation_in_progress=True,
        active_provider_runner={"fixture_group": "unknown"},
    )
    state.save(daemon.state_path)
    calls = []
    monkeypatch.setattr(
        rescue,
        "plan_automatic_implementation_rescue",
        lambda **kwargs: rescue.AutoRescuePlan(
            action=rescue.AutoRescueAction.INLINE_PROVIDER_RESCUE, reason="fixture"
        ),
    )

    def provider(*args, **kwargs):
        calls.append(True)
        if uncertain:
            raise ProcessGroupCleanupUnverified("process_group_probe_unavailable")
        raise RuntimeError("ordinary fixture error")

    monkeypatch.setattr(module, "run_process_group_stream", provider)
    arguments = {
        "task": task,
        "attempt": 1,
        "workspace_path": daemon.repo_root,
        "branch_name": "fixture",
        "baseline_ref": "HEAD",
        "validation_result": {"passed": False},
        "log_path": tmp_path / "rescue.log",
        "state": state,
        "command": ["python"],
        "base_prompt": "fixture",
    }
    if uncertain:
        with pytest.raises(ProcessGroupCleanupUnverified):
            daemon._automatic_implementation_rescue(**arguments)
    else:
        result = daemon._automatic_implementation_rescue(**arguments)
        assert result["passed"] is False
    assert calls == [True]
    persisted = PortalTaskState.load(daemon.state_path)
    assert persisted.implementation_in_progress is True
    assert persisted.active_provider_runner == {"fixture_group": "unknown"}
    events = [json.loads(line) for line in daemon.events_path.read_text().splitlines()]
    assert (
        any(
            event["type"] == "implementation_auto_rescue_provider_failed"
            for event in events
        )
        is not uncertain
    )
