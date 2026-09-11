"""Retained SAWM-016 sequence-12 failure, with provider contents sanitized."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.grok_cli_runner import (
    build_grok_failure_receipt,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    provider_capacity_custody as custody,
)
from ipfs_accelerate_py.agent_supervisor.runtime.provider_failure_policy import (
    render_grok_failure_receipt,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    WorktreeLifecycleError,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
    PortalTaskState,
)

# Historical sequence 12 / f5ffee52a7f1c93dd86064de9c2cd9b101b81437650b09bcbdaee257740ab162.
# Keep only execution and quota facts; tool content, repository paths, usage
# details and identifiers are intentionally omitted from the fixture.
QUOTA = "API error (status 402 Payment Required): Grok Build usage balance exhausted"
POST_DISPATCH = (
    "\n".join(
        json.dumps(item)
        for item in (
            {
                "type": "tool_call_update",
                "status": "completed",
                "toolCallId": "sanitized-read-1",
            },
            {
                "type": "tool_call_update",
                "status": "completed",
                "toolCallId": "sanitized-read-2",
            },
            {
                "type": "error",
                "message": QUOTA,
                "num_turns": 16,
                "modelUsage": {"grok-4.6-build": {"modelCalls": 16}},
            },
        )
    )
    + "\n"
)
COMMAND = [
    "/usr/bin/python3",
    "-m",
    "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
    "--model",
    "grok-4.6",
    "--grok-failure-receipt-nonce",
    "a" * 64,
]


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()


def make_daemon(tmp_path: Path, *, pool: bool = False):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Capacity Test")
    git(repo, "config", "user.email", "capacity@example.invalid")
    board = repo / "tasks.todo.md"
    board.write_text("# Tasks\n")
    (repo / "candidate.py").write_text("VALUE = 'baseline'\n")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "baseline")
    daemon = PortalImplementationDaemon(
        todo_path=board,
        state_path=repo / "state" / "task.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## SAWM-",
        implement=True,
        use_ephemeral_worktree=True,
        worktree_root=repo / "worktrees",
        worktree_pool_enabled=pool,
    )
    task = PortalTask(
        task_id="SAWM-016",
        title="Retain partial work",
        status="ready",
        completion="manual",
        priority="P0",
        track="agent",
        outputs=["candidate.py"],
    )
    daemon._register_task_identities([task])
    return daemon, task


def private_log(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o600)
    return path


def pre_dispatch() -> str:
    receipt = build_grok_failure_receipt(
        probe_stderr_text=QUOTA,
        nonce="a" * 64,
        model="grok-4.6",
        probe_returncode=1,
        primary_dispatched=False,
    )
    return QUOTA + "\n" + render_grok_failure_receipt(receipt) + "\n"


def test_retained_sequence12_is_not_a_refundable_capacity_refusal(tmp_path):
    daemon, _ = make_daemon(tmp_path)
    log = private_log(tmp_path / "post.log", POST_DISPATCH)
    failure = daemon._provider_capacity_failure_from_log(
        log, command=COMMAND, returncode=1
    )
    assert failure["exhausted"] is True
    assert failure.get("pre_dispatch_refusal_verified") is False
    assert failure.get("retained_callback_required") is True
    assert failure["execution_observation"]["model_calls"] == 16
    assert failure["execution_observation"]["tool_event_count"] == 2


@pytest.mark.parametrize(
    "mutation",
    ["none", "wrong_nonce", "duplicate", "positive_work", "truncated", "fallback"],
)
def test_only_complete_bound_no_task_dispatch_proof_refunds(tmp_path, mutation):
    daemon, task = make_daemon(tmp_path)
    text = pre_dispatch()
    command = list(COMMAND)
    if mutation == "wrong_nonce":
        command[-1] = "b" * 64
    elif mutation == "duplicate":
        text += text
    elif mutation == "positive_work":
        text = POST_DISPATCH + text
    elif mutation == "truncated":
        text = (
            POST_DISPATCH
            + "x" * (module.PROVIDER_CAPACITY_LOG_TAIL_BYTES + 1)
            + "\n"
            + text
        )
    elif mutation == "fallback":
        command += ["--codex-fallback-command-json", '["codex"]']
    log = private_log(tmp_path / "quota.log", text)
    failure = daemon._provider_capacity_failure_from_log(
        log, command=command, returncode=1
    )
    assert failure.get("pre_dispatch_refusal_verified") is (mutation == "none")
    if mutation == "none":
        state = PortalTaskState()
        daemon._mark_implementation_started(
            state, task=task, attempt=1, started_at=module.utc_now(), log_path=log
        )
        result = daemon._record_provider_capacity_deferral(
            task=task,
            state=state,
            attempt=1,
            started_at=module.utc_now(),
            returncode=1,
            log_path=log,
            failure=failure,
            command=command,
        )
        assert result["attempt_consumed"] is False
        assert not state.implementation_attempts
        assert daemon._retained_provider_capacity_deferral() is None


@pytest.mark.parametrize("pool", [False, True])
@pytest.mark.parametrize("observed_work", [False, True])
def test_actual_callback_keeps_workspace_budget_log_and_blocks_replay(
    tmp_path, monkeypatch, pool, observed_work
):
    daemon, task = make_daemon(tmp_path, pool=pool)
    evidence = POST_DISPATCH if observed_work else QUOTA + "\n"
    script = (
        "from pathlib import Path; "
        "Path('candidate.py').write_text(\"VALUE = 'partial'\\n\"); "
        f"print({evidence!r}, end=''); raise SystemExit(1)"
    )
    command = [sys.executable, "-c", script]
    monkeypatch.setattr(
        daemon, "_build_implementation_command", lambda *_a, **_k: command
    )
    monkeypatch.setattr(
        daemon, "_decision_runtime_mutation", lambda _kind, _body, action: action()
    )
    monkeypatch.setattr(
        daemon,
        "_evaluate_pre_implementation_provider_gate",
        lambda **_k: {"provider_authorized": True, "skip_provider": False},
    )
    state = PortalTaskState()
    log = daemon.implementation_log_dir / "dispatch-a.log"
    result = daemon._run_implementation_in_ephemeral_worktree(
        task=task,
        state=state,
        attempt=1,
        started_at=module.utc_now(),
        log_path=log,
        prompt="isolated test",
    )
    assert result.get("reason") == custody.REASON, result
    assert result["attempt_consumed"] is True
    workspace = Path(result["worktree_path"])
    assert (workspace / "candidate.py").read_text() == "VALUE = 'partial'\n"
    assert evidence in log.read_text()
    assert result["execution_observation"]["task_execution_observed"] is observed_work
    assert result["callback_outcome"] == "unknown"
    assert result["settlement_authority"] is False
    assert state.implementation_attempts[task.task_id] == 1
    original = {p: p.read_bytes() for p in (log, workspace / "candidate.py")}
    quarantine = daemon.worktree_lifecycle.load_quarantine(workspace)
    assert quarantine["terminalized"] is False
    assert quarantine["cleanup_allowed"] is False
    captured = daemon.worktree_lifecycle.load_workspace(workspace)
    assert captured is not None and not captured.is_terminal
    cleanup = daemon._cleanup_merged_worktree(
        workspace, result["branch"], reusable=False
    )
    assert cleanup["cleaned"] is False
    assert (
        daemon.worktree_lifecycle.compare_and_delete(
            workspace,
            expected_fence=captured.fence,
            lease_id=captured.lease_id,
        )
        is False
    )
    with pytest.raises(WorktreeLifecycleError):
        daemon.worktree_lifecycle.mark_terminal(
            workspace,
            lease_id=captured.lease_id,
            expected_fence=captured.fence,
        )
    monkeypatch.setattr(
        daemon, "_run_once", lambda: pytest.fail("retained callback was replayed")
    )
    assert daemon.run_once()["implementation_result"]["reason"] == custody.REASON
    restarted = PortalImplementationDaemon(
        todo_path=daemon.todo_path,
        state_path=daemon.state_path,
        strategy_path=daemon.strategy_path,
        events_path=daemon.events_path,
        repo_root=daemon.repo_root,
        worktree_root=daemon.worktree_root,
    )
    monkeypatch.setattr(
        restarted, "_run_once", lambda: pytest.fail("restart replayed callback")
    )
    assert restarted.run_once()["implementation_result"]["retry_authorized"] is False
    assert all(path.read_bytes() == raw for path, raw in original.items())


def test_audit_failure_after_native_retention_does_not_replay(tmp_path, monkeypatch):
    daemon, task = make_daemon(tmp_path)
    workspace = daemon.repo_root / "partial"
    workspace.mkdir()
    (workspace / "candidate.py").write_text("partial candidate\n")
    record = daemon.worktree_lifecycle.begin_preparing(
        workspace_path=workspace,
        task_id=task.task_id,
        canonical_task_cid=daemon._canonical_ref(task),
        attempt=1,
        lane_id="lane-3",
        state_dir=str(daemon.state_path.parent),
        branch="partial",
        merge_target="main",
    )
    daemon._active_worktree_lifecycle = record
    log = private_log(tmp_path / "dispatch.log", POST_DISPATCH)
    failure = daemon._provider_capacity_failure_from_log(
        log, command=COMMAND, returncode=1
    )
    monkeypatch.setattr(
        daemon,
        "_record_event",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("audit failed")),
    )
    with pytest.raises(custody.RetentionError, match="custody publication"):
        daemon._record_provider_capacity_deferral(
            task=task,
            state=PortalTaskState(),
            attempt=1,
            started_at=module.utc_now(),
            returncode=1,
            log_path=log,
            failure=failure,
            worktree_path=workspace,
            branch_name="partial",
            command=COMMAND,
        )
    assert daemon.worktree_lifecycle.load_quarantine(workspace) is not None
    monkeypatch.setattr(
        daemon, "_run_once", lambda: pytest.fail("audit error permitted replay")
    )
    assert daemon.run_once()["implementation_result"]["retry_authorized"] is False
    assert (workspace / "candidate.py").read_text() == "partial candidate\n"


def test_direct_refund_flag_cannot_replace_typed_no_dispatch_proof(tmp_path):
    daemon, task = make_daemon(tmp_path)
    log = private_log(tmp_path / "dispatch.log", POST_DISPATCH)
    result = daemon._record_provider_capacity_deferral(
        task=task,
        state=PortalTaskState(),
        attempt=1,
        started_at=module.utc_now(),
        returncode=1,
        log_path=log,
        command=COMMAND,
        failure={"exhausted": True, "pre_dispatch_refusal_verified": True},
    )
    assert result["attempt_consumed"] is True
    assert result["reason"] == custody.REASON
    assert (
        custody.read(custody.path_for(daemon.state_path))["receipt_id"]
        == result["receipt_id"]
    )


def test_real_portal_bridge_retains_exact_callback_on_the_next_pass(
    tmp_path, monkeypatch
):
    from test.api.test_agent_supervisor_database_portal_bridge import (
        _TaskSource,
        _attempt,
        _record,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
        DatabasePortalExecutionBridge,
        DatabasePortalBridgeDeferred,
    )

    seed, _ = make_daemon(tmp_path)
    record = _record()
    record.task_alias = "SAWM-016"
    record.dependencies = ()
    record.outputs = ({"path": "candidate.py"},)
    record.validations = ()
    record.body["read_scope"] = ["candidate.py"]
    record.body["write_scope"] = ["candidate.py"]
    attempt = replace(_attempt(), task_alias="SAWM-016")
    dispatches = []

    def factory(paths, alias):
        daemon = PortalImplementationDaemon(
            todo_path=paths.task_projection,
            state_path=paths.state,
            strategy_path=paths.strategy,
            events_path=paths.events,
            repo_root=seed.repo_root,
            worktree_root=seed.worktree_root,
            task_header_prefix="## SAWM-",
            implement=True,
            use_ephemeral_worktree=True,
            worktree_pool_enabled=False,
        )
        [task] = daemon._load_tasks()
        command = [
            sys.executable,
            "-c",
            f"print({POST_DISPATCH!r}); raise SystemExit(1)",
        ]
        monkeypatch.setattr(
            daemon, "_build_implementation_command", lambda *_a, **_k: command
        )
        monkeypatch.setattr(
            daemon, "_decision_runtime_mutation", lambda _kind, _body, action: action()
        )
        monkeypatch.setattr(
            daemon,
            "_evaluate_pre_implementation_provider_gate",
            lambda **_k: {"provider_authorized": True, "skip_provider": False},
        )

        def actual_pass():
            dispatches.append(alias)
            result = daemon._run_implementation_in_ephemeral_worktree(
                task=task,
                state=PortalTaskState(),
                attempt=1,
                started_at=module.utc_now(),
                log_path=daemon.implementation_log_dir / "bridge-dispatch.log",
                prompt="test",
            )
            assert (
                result["database_attempt_authority"]["database_attempt_id"]
                == attempt.attempt_id
            )
            assert result["implementation_start"]["event_id"]
            return {"implementation_result": result}

        monkeypatch.setattr(daemon, "_run_once", actual_pass)
        return daemon

    bridge = DatabasePortalExecutionBridge(
        task_source=_TaskSource(record),
        attempt_root=seed.repo_root / "attempts",
        portal_factory=factory,
        repo_root=seed.repo_root,
        board_namespace="capacity-test",
        task_header_prefix="## SAWM-",
    )
    for _ in range(2):
        with pytest.raises(DatabasePortalBridgeDeferred, match=custody.REASON):
            bridge.run_provider(attempt)
    assert dispatches == ["SAWM-016"]
    assert "- Status: ready" in bridge._paths(attempt).task_projection.read_text()


@pytest.mark.parametrize("mutation", ["fence", "lease", "owner", "index"])
def test_current_owner_quarantine_rejects_crossed_native_custody(tmp_path, mutation):
    daemon, task = make_daemon(tmp_path)
    workspace = daemon.repo_root / "candidate"
    record = daemon.worktree_lifecycle.begin_preparing(
        workspace_path=workspace,
        task_id=task.task_id,
        canonical_task_cid=daemon._canonical_ref(task),
        attempt=1,
        lane_id="lane-3",
        state_dir=str(daemon.state_path.parent),
        branch="candidate",
        merge_target="main",
    )
    expected = record
    if mutation == "fence":
        expected = replace(record, fence=record.fence + 1)
    elif mutation == "lease":
        expected = replace(record, lease_id="foreign")
    elif mutation == "owner":
        expected = replace(
            record,
            owner=replace(
                record.owner, start_time_ticks=record.owner.start_time_ticks + 1
            ),
        )
    else:
        index = daemon.worktree_lifecycle.task_index_path_for(
            canonical_task_cid=record.canonical_task_cid,
            task_id=record.task_id,
            attempt=record.attempt,
        )
        payload = json.loads(index.read_text())
        payload["lease_id"] = "foreign"
        index.write_text(json.dumps(payload))
    with pytest.raises(WorktreeLifecycleError):
        daemon.worktree_lifecycle.quarantine_current_owner(
            expected,
            fence_authority={"observation": "unknown"},
            reason=custody.REASON,
        )
    assert daemon.worktree_lifecycle.load_quarantine(workspace) is None


def test_unreadable_retained_marker_never_releases_callback(tmp_path, monkeypatch):
    daemon, _ = make_daemon(tmp_path)
    path = custody.path_for(daemon.state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("partial publication")
    monkeypatch.setattr(
        daemon, "_run_once", lambda: pytest.fail("invalid marker was ignored")
    )
    result = daemon.run_once()["implementation_result"]
    assert result["reason"] == custody.REASON
    assert result["retry_authorized"] is False


def test_outer_portal_does_not_turn_retention_io_failure_into_terminal_failure(
    tmp_path, monkeypatch
):
    daemon, task = make_daemon(tmp_path)
    state = PortalTaskState()
    monkeypatch.setattr(daemon, "_require_primary_provider_readiness", lambda *_a: None)
    monkeypatch.setattr(daemon, "_build_implementation_prompt", lambda *_a: "test")
    monkeypatch.setattr(
        daemon,
        "_persist_implementation_context_receipt",
        lambda *_a: tmp_path / "context.json",
    )

    def incomplete_publication(**kwargs):
        daemon._mark_implementation_started(
            kwargs["state"],
            task=task,
            attempt=kwargs["attempt"],
            started_at=kwargs["started_at"],
            log_path=kwargs["log_path"],
        )
        raise custody.RetentionError("custody publication failed")

    monkeypatch.setattr(
        daemon, "_run_implementation_in_ephemeral_worktree", incomplete_publication
    )
    monkeypatch.setattr(
        daemon,
        "_record_task_queue_outcome",
        lambda *_a, **_k: pytest.fail("invented terminal failure"),
    )
    with pytest.raises(custody.RetentionError):
        daemon._run_implementation(task, state)
    assert state.implementation_in_progress is True
    assert state.implementation_attempts[task.task_id] == 1
