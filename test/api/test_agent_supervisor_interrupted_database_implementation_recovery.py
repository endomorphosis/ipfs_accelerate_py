from __future__ import annotations

import json
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    serialized_lock_update,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    supervisor as portal_supervisor,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalExecutionBridge,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTaskState,
)
from ipfs_accelerate_py.agent_supervisor.worktree_lifecycle import (
    ProcessBirthIdentity,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return result.stdout.strip()


def _timestamp(offset_seconds: int = 0) -> str:
    value = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    return value.isoformat().replace("+00:00", "Z")


def _events(daemon: PortalImplementationDaemon) -> list[dict[str, Any]]:
    if not daemon.events_path.exists():
        return []
    return [
        json.loads(line)
        for line in daemon.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _restart_daemon(
    daemon: PortalImplementationDaemon,
) -> PortalImplementationDaemon:
    return PortalImplementationDaemon(
        todo_path=daemon.todo_path,
        state_path=daemon.state_path,
        strategy_path=daemon.strategy_path,
        events_path=daemon.events_path,
        repo_root=daemon.repo_root,
        worktree_root=daemon.worktree_root,
        merge_target_branch="main",
        task_header_prefix="## PCTDD-",
        implementation_protected_paths=(),
        implement=False,
    )


def _seed_interrupted_implementation(
    tmp_path: Path,
) -> tuple[PortalImplementationDaemon, dict[str, Any], Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    todo_path = repo / "tasks.todo.md"
    todo_path.write_text(
        """# Tasks

## PCTDD-034 Define aggregate test-pass statement

- Status: todo
- Completion: auto
- Priority: P1
- Track: implementation
- Outputs: src/aggregate_statement.py
- Validation: python -m pytest -q focused.py
""",
        encoding="utf-8",
    )
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", "README.md", "tasks.todo.md")
    _git(repo, "commit", "-m", "base")

    worktree_root = tmp_path / "worktrees"
    workspace = worktree_root / "pctdd-034-attempt-1"
    branch = "implementation/pctdd-034-attempt-1"
    worktree_root.mkdir()
    _git(repo, "worktree", "add", "-b", branch, str(workspace), "HEAD")
    dirty_candidate = workspace / "src" / "aggregate_statement.py"
    dirty_candidate.parent.mkdir()
    dirty_candidate.write_text("CANDIDATE = True\n", encoding="utf-8")

    state_dir = tmp_path / "state"
    daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task-state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        worktree_root=worktree_root,
        merge_target_branch="main",
        task_header_prefix="## PCTDD-",
        implementation_protected_paths=(),
        implement=False,
    )
    task = daemon._load_tasks()[0]
    identity = daemon._identity_for_task(task)
    attempt = 1
    started_at = _timestamp(-10)
    clear_epoch = _timestamp(-1)
    dead_owner = ProcessBirthIdentity(
        pid=2**30 - 73,
        start_time_ticks=1,
        boot_id="dead-owner",
    )
    preparing = daemon.worktree_lifecycle.begin_preparing(
        task_id=task.task_id,
        canonical_task_cid=identity.canonical_task_cid,
        attempt=attempt,
        lane_id="lane-3",
        workspace_path=workspace,
        branch=branch,
        merge_target="main",
        state_dir=str(state_dir.resolve()),
        owner=dead_owner,
    )
    active = daemon.worktree_lifecycle.mark_active(
        workspace,
        lease_id=preparing.lease_id,
        expected_fence=preparing.fence,
    )
    terminal = daemon.worktree_lifecycle.mark_terminal(
        workspace,
        lease_id=active.lease_id,
        expected_fence=active.fence,
        reason="controlled_restart_dead_owner",
    )

    state = PortalTaskState(
        heartbeat_at=clear_epoch,
        last_progress_at=clear_epoch,
        task_statuses={task.task_id: "todo"},
        task_identities={task.task_id: identity.to_dict()},
        implementation_attempts={task.task_id: attempt},
        implementation_attempts_by_cid={
            identity.canonical_task_cid: attempt
        },
        last_implementation_task_id=task.task_id,
        last_implementation_task_key=identity.canonical_task_key,
        last_implementation_task_cid=identity.canonical_task_cid,
        last_implementation_started_at=started_at,
        last_implementation_worktree_path=str(workspace),
        last_implementation_branch=branch,
    )
    state.save(daemon.state_path)

    claim = daemon._build_implementation_task_claim_metadata(
        task,
        attempt,
        started_at,
    )
    claim.update(
        {
            "pid": dead_owner.pid,
            "owner_process_birth": dead_owner.to_dict(),
        }
    )
    claim_path = daemon._implementation_task_claim_path(
        task.task_id,
        canonical_task_cid=identity.canonical_task_cid,
    )
    daemon_module.write_json_atomic(claim_path, claim)
    daemon._record_event(
        "implementation_started",
        {
            "task_id": task.task_id,
            "canonical_task_key": identity.canonical_task_key,
            "canonical_task_cid": identity.canonical_task_cid,
            "board_namespace": identity.board_namespace,
            "attempt": attempt,
            "worktree_path": str(workspace),
            "branch": branch,
        },
    )

    portal = {
        "reconciled": False,
        "blocked": True,
        "reason": "task_claim_reconciliation_blocked",
        "reconciled_at": clear_epoch,
        "protected_path_reconciliation": {
            "blocked": False,
            "reason": "crash_reconciliation_unchanged",
            "task_id": task.task_id,
            "attempt": attempt,
            "workspace_path": str(workspace),
        },
        "worktree_lifecycle_reconciliation": {
            "blocked": False,
            "reconciled": True,
            "state": "terminal",
            "task_id": task.task_id,
            "canonical_task_cid": identity.canonical_task_cid,
            "attempt": attempt,
            "workspace_path": str(workspace),
            "record_id": terminal.record_id,
            "fence": terminal.fence,
        },
        "task_claim_reconciliation": {
            "blocked": True,
            "reconciled": False,
            "reason": "canonical_task_not_terminal",
            "observed_task_status": "todo",
            "task_id": task.task_id,
            "canonical_task_cid": identity.canonical_task_cid,
        },
        "attempt_recovery": {
            "consumed": False,
            "attempt": attempt,
            "task_id": task.task_id,
            "canonical_task_cid": identity.canonical_task_cid,
            "previous_display_count": attempt,
            "previous_cid_count": attempt,
        },
    }
    receipt = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "database-portal-attempt-reconciliation@1"
        ),
        "stage": "blocked",
        "binding_id": "sha256:" + "b" * 64,
        "task_alias": task.task_id,
        "blocked": True,
        "reconciled": False,
        "reason": "nested_portal_attempt_reconciliation_blocked",
        "terminal_provider_evidence": False,
        "provider_runner_reconciliation_authority": (
            "ordinary_provider_runner_fence"
        ),
        "nested_state": {
            "active": True,
            "active_phase": "implementing",
            "active_task_id": task.task_id,
            "active_attempt": attempt,
            "active_worktree_path": str(workspace),
            "active_branch": branch,
            "state_path": str(daemon.state_path),
        },
        "provider_runner_fence": {
            "applicable": True,
            "fenced": True,
            "safe_to_restart": True,
            "reason": "ordinary_provider_runner_exact_birth_fenced",
            "pid": dead_owner.pid,
        },
        "portal_reconciliation": portal,
    }
    receipt["receipt_id"] = daemon._database_recovery_sha256(receipt)
    evidence = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "database-portal-interrupted-implementation-retry@1"
        ),
        "binding_id": receipt["binding_id"],
        "reconciliation_receipt": receipt,
    }
    evidence["evidence_id"] = daemon._database_recovery_sha256(evidence)
    return daemon, evidence, claim_path, dirty_candidate


def _mutate_first_clear_authority(
    receipt: dict[str, Any],
    mutation: str,
) -> None:
    portal = receipt["portal_reconciliation"]
    if mutation == "receipt_blocked_false":
        receipt["blocked"] = False
    elif mutation == "receipt_reconciled_true":
        receipt["reconciled"] = True
    elif mutation == "portal_blocked_false":
        portal["blocked"] = False
    elif mutation == "portal_reconciled_true":
        portal["reconciled"] = True
    elif mutation == "protected_blocked_true":
        portal["protected_path_reconciliation"]["blocked"] = True
    elif mutation == "lifecycle_blocked_true":
        portal["worktree_lifecycle_reconciliation"]["blocked"] = True
    elif mutation == "lifecycle_reconciled_false":
        portal["worktree_lifecycle_reconciliation"]["reconciled"] = False
    elif mutation == "claim_blocked_false":
        portal["task_claim_reconciliation"]["blocked"] = False
    elif mutation == "claim_reconciled_true":
        portal["task_claim_reconciliation"]["reconciled"] = True
    elif mutation == "claim_task_id_wrong":
        portal["task_claim_reconciliation"]["task_id"] = "PCTDD-WRONG"
    elif mutation == "protected_attempt_bool":
        portal["protected_path_reconciliation"]["attempt"] = True
    elif mutation == "lifecycle_attempt_float":
        lifecycle = portal["worktree_lifecycle_reconciliation"]
        lifecycle["attempt"] = float(lifecycle["attempt"])
    elif mutation == "lifecycle_fence_float":
        lifecycle = portal["worktree_lifecycle_reconciliation"]
        lifecycle["fence"] = float(lifecycle["fence"])
    elif mutation == "recovery_attempt_bool":
        portal["attempt_recovery"]["attempt"] = True
    elif mutation == "recovery_display_float":
        recovery = portal["attempt_recovery"]
        recovery["previous_display_count"] = float(
            recovery["previous_display_count"]
        )
    elif mutation == "recovery_cid_bool":
        portal["attempt_recovery"]["previous_cid_count"] = True
    else:  # pragma: no cover - closed test vocabulary
        raise AssertionError(mutation)


def test_interrupted_implementation_releases_once_and_is_idempotent(
    tmp_path: Path,
) -> None:
    daemon, evidence, claim_path, dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )
    state_before = PortalTaskState.load(daemon.state_path)

    first = daemon.reconcile_interrupted_database_implementation_attempt(
        evidence
    )
    second = daemon.reconcile_interrupted_database_implementation_attempt(
        evidence
    )

    assert first["reconciled"] is True, first
    assert second["reconciled"] is True, second
    assert first["provider_dispatched"] is False
    assert first["implementation_dispatched"] is False
    assert first["acceptance_inferred"] is False
    assert first["task_claim_reconciliation"]["receipt_id"] == (
        second["task_claim_reconciliation"]["receipt_id"]
    )
    assert not claim_path.exists()
    assert dirty_candidate.read_text(encoding="utf-8") == "CANDIDATE = True\n"
    state = PortalTaskState.load(daemon.state_path)
    assert "PCTDD-034" not in state.implementation_attempts
    assert (
        state_before.last_implementation_task_cid
        not in state.implementation_attempts_by_cid
    )
    assert state.last_implementation_task_id == (
        state_before.last_implementation_task_id
    )
    assert state.last_implementation_task_key == (
        state_before.last_implementation_task_key
    )
    assert state.last_implementation_task_cid == (
        state_before.last_implementation_task_cid
    )
    assert state.last_implementation_started_at == (
        state_before.last_implementation_started_at
    )
    assert state.last_implementation_worktree_path == (
        state_before.last_implementation_worktree_path
    )
    assert state.last_implementation_branch == (
        state_before.last_implementation_branch
    )
    assert state.last_implementation_finished_at == ""
    assert state.last_implementation_returncode is None
    events = _events(daemon)
    assert sum(
        event["type"] == "implementation_state_recovered"
        for event in events
    ) == 1
    assert sum(
        event["type"] == "implementation_task_claim_released"
        for event in events
    ) == 1
    assert not any(event["type"] == "task_completed" for event in events)
    assert not any(
        event["type"]
        in {
            "implementation_finished",
            "implementation_candidate_merged",
            "merge_finished",
        }
        for event in events
    )


def test_interrupted_release_replays_after_empty_identity_cache_restart(
    tmp_path: Path,
) -> None:
    daemon, evidence, _claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )
    daemon._register_task_identities(daemon._load_tasks())
    assert daemon._task_identity_by_display_id
    first = daemon.reconcile_interrupted_database_implementation_attempt(
        evidence
    )
    assert first["reconciled"] is True, first
    restarted = _restart_daemon(daemon)
    assert not restarted._task_identity_by_display_id

    replay = restarted.reconcile_interrupted_database_implementation_attempt(
        evidence
    )

    assert replay["reconciled"] is True, replay
    assert replay["blocked"] is False


def test_interrupted_writer_revalidates_release_event_before_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, evidence, _claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )
    original_record = daemon._record_event
    original_iter = daemon._iter_merge_lifecycle_events
    emitted = {"release": False}

    def emit_then_corrupt_projection(
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        original_record(event_type, payload)
        if event_type == "implementation_task_claim_released":
            emitted["release"] = True

    def projected_events() -> list[dict[str, Any]]:
        current = [dict(event) for event in original_iter()]
        if emitted["release"]:
            for event in current:
                if event.get("type") == "implementation_task_claim_released":
                    event["timestamp"] = "2000-01-01T00:00:00Z"
        return current

    monkeypatch.setattr(daemon, "_record_event", emit_then_corrupt_projection)
    monkeypatch.setattr(
        daemon,
        "_iter_merge_lifecycle_events",
        projected_events,
    )

    rejected = daemon.reconcile_interrupted_database_implementation_attempt(
        evidence
    )

    assert rejected["blocked"] is True
    assert rejected["reconciled"] is False
    claim_release = rejected["task_claim_reconciliation"]
    assert claim_release["reason"] == "task_claim_release_event_not_durable"


@pytest.mark.parametrize(
    "mutation",
    (
        "receipt_blocked_false",
        "receipt_reconciled_true",
        "portal_blocked_false",
        "portal_reconciled_true",
        "protected_blocked_true",
        "lifecycle_blocked_true",
        "lifecycle_reconciled_false",
        "claim_blocked_false",
        "claim_reconciled_true",
        "claim_task_id_wrong",
        "protected_attempt_bool",
        "lifecycle_attempt_float",
        "lifecycle_fence_float",
        "recovery_attempt_bool",
        "recovery_display_float",
        "recovery_cid_bool",
    ),
)
def test_interrupted_implementation_rejects_malformed_authority(
    tmp_path: Path,
    mutation: str,
) -> None:
    daemon, evidence, claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )
    candidate = json.loads(json.dumps(evidence))
    receipt = candidate["reconciliation_receipt"]
    _mutate_first_clear_authority(receipt, mutation)
    receipt.pop("receipt_id")
    receipt["receipt_id"] = daemon._database_recovery_sha256(receipt)
    candidate.pop("evidence_id")
    candidate["evidence_id"] = daemon._database_recovery_sha256(candidate)

    result = daemon.reconcile_interrupted_database_implementation_attempt(
        candidate
    )

    assert result["blocked"] is True
    expected_reason = (
        "interrupted_implementation_receipt_invalid"
        if mutation
        in {
            "receipt_blocked_false",
            "receipt_reconciled_true",
            "portal_blocked_false",
            "portal_reconciled_true",
        }
        else "interrupted_implementation_receipt_authority_invalid"
    )
    assert result["reason"] == expected_reason
    assert claim_path.exists()
    assert not any(
        event["type"] == "interrupted_implementation_retry_prepared"
        for event in _events(daemon)
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "receipt_blocked_false",
        "receipt_reconciled_true",
        "portal_blocked_false",
        "portal_reconciled_true",
        "protected_blocked_true",
        "lifecycle_blocked_true",
        "lifecycle_reconciled_false",
        "claim_blocked_false",
        "claim_reconciled_true",
        "claim_task_id_wrong",
    ),
)
def test_bridge_selector_rejects_rehashed_malformed_first_clear(
    tmp_path: Path,
    mutation: str,
) -> None:
    _daemon, evidence, _claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )
    attempt = SimpleNamespace(
        attempt_id="database-attempt:selector",
        claim_id="database-claim:selector",
        task_cid="database-task:selector",
        task_alias="PCTDD-034",
        attempt_number=1,
        owner_session_id="database-session:selector",
        fencing_token=7,
        fence_epoch=3,
    )
    bridge = DatabasePortalExecutionBridge(
        task_source=object(),
        attempt_root=tmp_path / "database-attempts",
        portal_factory=lambda _paths, _alias: None,
    )
    paths = bridge._paths(attempt)
    receipt = json.loads(json.dumps(evidence["reconciliation_receipt"]))
    receipt.pop("receipt_id")
    receipt["trigger"] = "database_daemon_startup"
    receipt["reconciled_at"] = _timestamp(-1)
    receipt["nested_state"]["state_path"] = str(paths.state)
    _mutate_first_clear_authority(receipt, mutation)
    bridge.persist_reconciliation_receipt(attempt, receipt)

    assert bridge._interrupted_implementation_retry_evidence(
        attempt,
        {
            "binding_id": receipt["binding_id"],
            "task_alias": attempt.task_alias,
        },
    ) is None


def test_interrupted_claim_release_replay_rejects_numeric_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, evidence, claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )
    original = daemon_module.write_json_atomic
    injected = {"raised": False}

    def crash_after_unlink(path: Path, payload: Any) -> None:
        if (
            isinstance(payload, dict)
            and payload.get("phase") == "released"
            and not injected["raised"]
        ):
            injected["raised"] = True
            raise RuntimeError("crash after claim unlink")
        original(path, payload)

    monkeypatch.setattr(daemon_module, "write_json_atomic", crash_after_unlink)
    with pytest.raises(RuntimeError, match="crash after claim unlink"):
        daemon.reconcile_interrupted_database_implementation_attempt(evidence)
    assert not claim_path.exists()
    monkeypatch.setattr(daemon_module, "write_json_atomic", original)
    events = _events(daemon)
    preparation = next(
        event
        for event in events
        if event["type"] == "interrupted_implementation_retry_prepared"
    )
    recovery_event = next(
        event
        for event in events
        if event["type"] == "implementation_state_recovered"
    )
    restarted = _restart_daemon(daemon)
    aliased_preparation = {**preparation, "attempt": True}
    rejected_preparation = (
        restarted._replay_interrupted_implementation_claim_release(
            preparation=aliased_preparation,
            recovery_event=recovery_event,
        )
    )
    assert rejected_preparation["blocked"] is True
    assert rejected_preparation["reason"] == (
        "interrupted_claim_release_preparation_invalid"
    )

    receipt_path = restarted._task_claim_release_receipt_path(
        canonical_task_cid=preparation["canonical_task_cid"],
        attempt=preparation["attempt"],
        lease_id=preparation["claim_lease_id"],
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["released_unfinished_attempt"]["released_from"] = True
    basis = {
        key: value
        for key, value in receipt.items()
        if key
        not in {
            "operation_id",
            "phase",
            "prepared_at",
            "released_at",
            "receipt_id",
        }
    }
    receipt["operation_id"] = content_identity(basis)
    daemon_module.write_json_atomic(receipt_path, receipt)
    rejected_receipt = (
        restarted._replay_interrupted_implementation_claim_release(
            preparation=preparation,
            recovery_event=recovery_event,
        )
    )
    assert rejected_receipt["blocked"] is True
    assert rejected_receipt["reason"] == (
        "interrupted_claim_release_receipt_invalid"
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "operation",
        "claim_path",
        "claim_owner_pid",
        "claim_legacy_worktree_root_missing",
        "lifecycle_state",
        "released_start_event_id",
        "released_claim_started_at",
    ),
)
def test_interrupted_claim_release_replay_rejects_rehashed_basis_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    daemon, evidence, claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )
    original = daemon_module.write_json_atomic
    injected = {"raised": False}

    def crash_after_unlink(path: Path, payload: Any) -> None:
        if (
            isinstance(payload, dict)
            and payload.get("phase") == "released"
            and not injected["raised"]
        ):
            injected["raised"] = True
            raise RuntimeError("crash after claim unlink")
        original(path, payload)

    monkeypatch.setattr(daemon_module, "write_json_atomic", crash_after_unlink)
    with pytest.raises(RuntimeError, match="crash after claim unlink"):
        daemon.reconcile_interrupted_database_implementation_attempt(evidence)
    assert not claim_path.exists()
    monkeypatch.setattr(daemon_module, "write_json_atomic", original)
    events = _events(daemon)
    preparation = next(
        event
        for event in events
        if event["type"] == "interrupted_implementation_retry_prepared"
    )
    recovery_event = next(
        event
        for event in events
        if event["type"] == "implementation_state_recovered"
    )
    restarted = _restart_daemon(daemon)
    receipt_path = restarted._task_claim_release_receipt_path(
        canonical_task_cid=preparation["canonical_task_cid"],
        attempt=preparation["attempt"],
        lease_id=preparation["claim_lease_id"],
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if mutation == "operation":
        receipt["operation"] = "malicious-authority"
    elif mutation == "claim_path":
        receipt["claim"]["path"] = "/wrong"
    elif mutation == "claim_owner_pid":
        receipt["claim"]["owner_pid"] += 1
    elif mutation == "claim_legacy_worktree_root_missing":
        receipt["claim"]["legacy_worktree_root_missing"] = not receipt[
            "claim"
        ]["legacy_worktree_root_missing"]
    elif mutation == "lifecycle_state":
        receipt["worktree_lifecycle"]["state"] = "active"
    elif mutation == "released_start_event_id":
        receipt["released_unfinished_attempt"]["start_event_id"] = (
            "sha256:" + "f" * 64
        )
    elif mutation == "released_claim_started_at":
        receipt["released_unfinished_attempt"]["claim_started_at"] = (
            "2020-01-01T00:00:00Z"
        )
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(mutation)
    basis = {
        key: value
        for key, value in receipt.items()
        if key
        not in {
            "operation_id",
            "phase",
            "prepared_at",
            "released_at",
            "receipt_id",
        }
    }
    receipt["operation_id"] = content_identity(basis)
    daemon_module.write_json_atomic(receipt_path, receipt)

    rejected = restarted._replay_interrupted_implementation_claim_release(
        preparation=preparation,
        recovery_event=recovery_event,
    )

    assert rejected["blocked"] is True
    assert rejected["reconciled"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        "task_id",
        "canonical_task_cid",
        "attempt_bool",
        "reason",
        "attempt_recovery_released_false",
    ),
)
def test_interrupted_replay_rejects_malformed_recovery_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    daemon, evidence, claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )
    original = daemon_module.write_json_atomic
    injected = {"raised": False}

    def crash_after_unlink(path: Path, payload: Any) -> None:
        if (
            isinstance(payload, dict)
            and payload.get("phase") == "released"
            and not injected["raised"]
        ):
            injected["raised"] = True
            raise RuntimeError("crash after claim unlink")
        original(path, payload)

    monkeypatch.setattr(daemon_module, "write_json_atomic", crash_after_unlink)
    with pytest.raises(RuntimeError, match="crash after claim unlink"):
        daemon.reconcile_interrupted_database_implementation_attempt(evidence)
    assert not claim_path.exists()
    monkeypatch.setattr(daemon_module, "write_json_atomic", original)
    events = _events(daemon)
    preparation = next(
        event
        for event in events
        if event["type"] == "interrupted_implementation_retry_prepared"
    )
    recovery_event = dict(
        next(
            event
            for event in events
            if event["type"] == "implementation_state_recovered"
        )
    )
    recovery_event["attempt_recovery"] = dict(
        recovery_event["attempt_recovery"]
    )
    if mutation == "task_id":
        recovery_event["task_id"] = "PCTDD-WRONG"
    elif mutation == "canonical_task_cid":
        recovery_event["canonical_task_cid"] = "wrong:cid"
    elif mutation == "attempt_bool":
        recovery_event["attempt"] = True
    elif mutation == "reason":
        recovery_event["reason"] = "wrong-reason"
    elif mutation == "attempt_recovery_released_false":
        recovery_event["attempt_recovery"]["released"] = False
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(mutation)
    restarted = _restart_daemon(daemon)

    rejected = restarted._replay_interrupted_implementation_claim_release(
        preparation=preparation,
        recovery_event=recovery_event,
    )

    assert rejected["blocked"] is True
    assert rejected["reason"] == (
        "interrupted_claim_release_recovery_event_invalid"
    )


def test_interrupted_selector_rejects_malformed_recovery_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, evidence, claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )
    original = daemon_module.write_json_atomic
    injected = {"raised": False}

    def crash_after_unlink(path: Path, payload: Any) -> None:
        if (
            isinstance(payload, dict)
            and payload.get("phase") == "released"
            and not injected["raised"]
        ):
            injected["raised"] = True
            raise RuntimeError("crash after claim unlink")
        original(path, payload)

    monkeypatch.setattr(daemon_module, "write_json_atomic", crash_after_unlink)
    with pytest.raises(RuntimeError, match="crash after claim unlink"):
        daemon.reconcile_interrupted_database_implementation_attempt(evidence)
    assert not claim_path.exists()
    monkeypatch.setattr(daemon_module, "write_json_atomic", original)
    events = _events(daemon)
    malformed = []
    for event in events:
        candidate = dict(event)
        if candidate.get("type") == "implementation_state_recovered":
            candidate["attempt"] = True
        malformed.append(candidate)
    restarted = _restart_daemon(daemon)
    monkeypatch.setattr(
        restarted,
        "_iter_merge_lifecycle_events",
        lambda: list(malformed),
    )

    rejected = restarted.reconcile_interrupted_database_implementation_attempt(
        evidence
    )

    assert rejected["blocked"] is True
    assert rejected["reason"] == "interrupted_implementation_recovery_invalid"


@pytest.mark.parametrize(
    "mutation",
    ("attempt_bool", "unknown_field", "forged_task_source"),
)
def test_interrupted_retry_preparation_rejects_noncanonical_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    daemon, evidence, _claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )

    def crash_after_preparation(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("crash after retry preparation")

    monkeypatch.setattr(
        daemon,
        "_release_unfinished_active_attempt",
        crash_after_preparation,
    )
    with pytest.raises(RuntimeError, match="crash after retry preparation"):
        daemon.reconcile_interrupted_database_implementation_attempt(evidence)
    malformed = []
    for event in _events(daemon):
        candidate = dict(event)
        if candidate.get("type") == "interrupted_implementation_retry_prepared":
            if mutation == "attempt_bool":
                candidate["attempt"] = True
            elif mutation == "unknown_field":
                candidate["completion_authorized"] = True
            elif mutation == "forged_task_source":
                candidate["task_source_identity"] = {"forged": True}
            else:  # pragma: no cover - closed parametrization
                raise AssertionError(mutation)
        malformed.append(candidate)
    restarted = _restart_daemon(daemon)
    monkeypatch.setattr(
        restarted,
        "_iter_merge_lifecycle_events",
        lambda: list(malformed),
    )

    rejected = restarted.reconcile_interrupted_database_implementation_attempt(
        evidence
    )

    assert rejected["blocked"] is True
    assert rejected["reason"] == "interrupted_implementation_preparation_invalid"


@pytest.mark.parametrize("mutation", ("task_id", "task_cid", "attempt_bool"))
def test_interrupted_retry_preparation_quarantines_malformed_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    daemon, evidence, _claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )

    def crash_after_preparation(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("crash after retry preparation")

    monkeypatch.setattr(
        daemon,
        "_release_unfinished_active_attempt",
        crash_after_preparation,
    )
    with pytest.raises(RuntimeError, match="crash after retry preparation"):
        daemon.reconcile_interrupted_database_implementation_attempt(evidence)
    events = _events(daemon)
    sibling = dict(
        next(
            event
            for event in events
            if event["type"] == "interrupted_implementation_retry_prepared"
        )
    )
    sibling["event_id"] = "sha256:" + "c" * 64
    if mutation == "task_id":
        sibling["task_id"] = "PCTDD-WRONG"
    elif mutation == "task_cid":
        sibling["canonical_task_cid"] = "sha256:" + "b" * 64
    elif mutation == "attempt_bool":
        sibling["attempt"] = True
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(mutation)
    restarted = _restart_daemon(daemon)
    monkeypatch.setattr(
        restarted,
        "_iter_merge_lifecycle_events",
        lambda: [*events, sibling],
    )

    rejected = restarted.reconcile_interrupted_database_implementation_attempt(
        evidence
    )

    assert rejected["blocked"] is True
    assert rejected["reason"] == "interrupted_implementation_preparation_ambiguous"


def test_interrupted_claim_release_rejects_malformed_existing_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, evidence, _claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )
    completed = daemon.reconcile_interrupted_database_implementation_attempt(
        evidence
    )
    assert completed["reconciled"] is True
    events = _events(daemon)
    preparation = next(
        event
        for event in events
        if event["type"] == "interrupted_implementation_retry_prepared"
    )
    recovery_event = next(
        event
        for event in events
        if event["type"] == "implementation_state_recovered"
    )
    malformed = []
    for event in events:
        candidate = dict(event)
        if candidate.get("type") == "implementation_task_claim_released":
            candidate["task_id"] = "PCTDD-WRONG"
        malformed.append(candidate)
    restarted = _restart_daemon(daemon)
    monkeypatch.setattr(
        restarted,
        "_iter_merge_lifecycle_events",
        lambda: list(malformed),
    )

    rejected = restarted._replay_interrupted_implementation_claim_release(
        preparation=preparation,
        recovery_event=recovery_event,
    )

    assert rejected["blocked"] is True
    assert rejected["reason"] == "interrupted_claim_release_event_conflict"


def test_interrupted_claim_release_rejects_early_existing_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, evidence, _claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )
    completed = daemon.reconcile_interrupted_database_implementation_attempt(
        evidence
    )
    assert completed["reconciled"] is True
    events = _events(daemon)
    preparation = next(
        event
        for event in events
        if event["type"] == "interrupted_implementation_retry_prepared"
    )
    recovery_event = next(
        event
        for event in events
        if event["type"] == "implementation_state_recovered"
    )
    early = []
    for event in events:
        candidate = dict(event)
        if candidate.get("type") == "implementation_task_claim_released":
            candidate["timestamp"] = "2000-01-01T00:00:00Z"
        early.append(candidate)
    restarted = _restart_daemon(daemon)
    monkeypatch.setattr(
        restarted,
        "_iter_merge_lifecycle_events",
        lambda: list(early),
    )

    rejected = restarted._replay_interrupted_implementation_claim_release(
        preparation=preparation,
        recovery_event=recovery_event,
    )

    assert rejected["blocked"] is True
    assert rejected["reason"] == "interrupted_claim_release_event_conflict"


def test_interrupted_claim_release_revalidates_new_event_before_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, evidence, claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )
    original_record = daemon._record_event

    def crash_before_release_event(
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        if event_type == "implementation_task_claim_released":
            raise RuntimeError("crash before claim release event")
        original_record(event_type, payload)

    monkeypatch.setattr(daemon, "_record_event", crash_before_release_event)
    with pytest.raises(RuntimeError, match="before claim release event"):
        daemon.reconcile_interrupted_database_implementation_attempt(evidence)
    assert not claim_path.exists()
    events = _events(daemon)
    preparation = next(
        event
        for event in events
        if event["type"] == "interrupted_implementation_retry_prepared"
    )
    recovery_event = next(
        event
        for event in events
        if event["type"] == "implementation_state_recovered"
    )
    restarted = _restart_daemon(daemon)
    original_restarted_record = restarted._record_event
    original_iter = restarted._iter_merge_lifecycle_events
    emitted = {"release": False}

    def emit_then_corrupt_projection(
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        original_restarted_record(event_type, payload)
        if event_type == "implementation_task_claim_released":
            emitted["release"] = True

    def projected_events() -> list[dict[str, Any]]:
        current = [dict(event) for event in original_iter()]
        if emitted["release"]:
            for event in current:
                if event.get("type") == "implementation_task_claim_released":
                    event["timestamp"] = "2000-01-01T00:00:00Z"
        return current

    monkeypatch.setattr(restarted, "_record_event", emit_then_corrupt_projection)
    monkeypatch.setattr(
        restarted,
        "_iter_merge_lifecycle_events",
        projected_events,
    )

    rejected = restarted._replay_interrupted_implementation_claim_release(
        preparation=preparation,
        recovery_event=recovery_event,
    )

    assert rejected["blocked"] is True
    assert rejected["reason"] == "interrupted_claim_release_event_not_durable"


@pytest.mark.parametrize("attempt_value", (True, 1.0, "1", None))
def test_interrupted_claim_release_rejects_malformed_sibling_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attempt_value: Any,
) -> None:
    daemon, evidence, _claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )
    completed = daemon.reconcile_interrupted_database_implementation_attempt(
        evidence
    )
    assert completed["reconciled"] is True
    events = _events(daemon)
    preparation = next(
        event
        for event in events
        if event["type"] == "interrupted_implementation_retry_prepared"
    )
    recovery_event = next(
        event
        for event in events
        if event["type"] == "implementation_state_recovered"
    )
    released = next(
        event
        for event in events
        if event["type"] == "implementation_task_claim_released"
    )
    sibling = dict(released)
    sibling["event_id"] = "sha256:" + "e" * 64
    sibling["receipt_id"] = "sha256:" + "d" * 64
    if attempt_value is None:
        sibling.pop("attempt")
    else:
        sibling["attempt"] = attempt_value
    restarted = _restart_daemon(daemon)
    monkeypatch.setattr(
        restarted,
        "_iter_merge_lifecycle_events",
        lambda: [*events, sibling],
    )

    rejected = restarted._replay_interrupted_implementation_claim_release(
        preparation=preparation,
        recovery_event=recovery_event,
    )

    assert rejected["blocked"] is True
    assert rejected["reason"] == "interrupted_claim_release_event_ambiguous"


def test_interrupted_claim_release_replay_missing_lifecycle_is_typed_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, evidence, claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )
    original = daemon_module.write_json_atomic
    injected = {"raised": False}

    def crash_after_unlink(path: Path, payload: Any) -> None:
        if (
            isinstance(payload, dict)
            and payload.get("phase") == "released"
            and not injected["raised"]
        ):
            injected["raised"] = True
            raise RuntimeError("crash after claim unlink")
        original(path, payload)

    monkeypatch.setattr(daemon_module, "write_json_atomic", crash_after_unlink)
    with pytest.raises(RuntimeError, match="crash after claim unlink"):
        daemon.reconcile_interrupted_database_implementation_attempt(evidence)
    assert not claim_path.exists()
    monkeypatch.setattr(daemon_module, "write_json_atomic", original)
    events = _events(daemon)
    preparation = dict(
        next(
            event
            for event in events
            if event["type"] == "interrupted_implementation_retry_prepared"
        )
    )
    recovery_event = next(
        event
        for event in events
        if event["type"] == "implementation_state_recovered"
    )
    preparation["workspace_path"] = str(tmp_path / "missing-workspace")

    rejected = _restart_daemon(
        daemon
    )._replay_interrupted_implementation_claim_release(
        preparation=preparation,
        recovery_event=recovery_event,
    )

    assert rejected["blocked"] is True
    assert rejected["reconciled"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        "preparation_after_recovery",
        "preparation_timestamp_future",
        "preparation_release_epoch_future",
        "receipt_prepared_before_start",
    ),
)
def test_interrupted_claim_release_rejects_contradictory_event_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    daemon, evidence, claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )
    original = daemon_module.write_json_atomic
    injected = {"raised": False}

    def crash_after_unlink(path: Path, payload: Any) -> None:
        if (
            isinstance(payload, dict)
            and payload.get("phase") == "released"
            and not injected["raised"]
        ):
            injected["raised"] = True
            raise RuntimeError("crash after claim unlink")
        original(path, payload)

    monkeypatch.setattr(daemon_module, "write_json_atomic", crash_after_unlink)
    with pytest.raises(RuntimeError, match="crash after claim unlink"):
        daemon.reconcile_interrupted_database_implementation_attempt(evidence)
    assert not claim_path.exists()
    monkeypatch.setattr(daemon_module, "write_json_atomic", original)
    events = _events(daemon)
    preparation = dict(
        next(
            event
            for event in events
            if event["type"] == "interrupted_implementation_retry_prepared"
        )
    )
    recovery_event = dict(
        next(
            event
            for event in events
            if event["type"] == "implementation_state_recovered"
        )
    )
    restarted = _restart_daemon(daemon)
    if mutation == "preparation_after_recovery":
        preparation["sequence"] = recovery_event["sequence"] + 1
    elif mutation == "preparation_timestamp_future":
        preparation["timestamp"] = "2099-01-01T00:00:00Z"
    elif mutation == "preparation_release_epoch_future":
        preparation["release_epoch"] = "2099-01-01T00:00:00Z"
        retry_basis = {
            field: preparation[field]
            for field in daemon_module._INTERRUPTED_RETRY_PREPARATION_FIELDS
        }
        retry_id = content_identity(retry_basis)
        preparation["interrupted_retry_id"] = retry_id
        recovery_event["interrupted_retry_id"] = retry_id
        receipt_path = restarted._task_claim_release_receipt_path(
            canonical_task_cid=preparation["canonical_task_cid"],
            attempt=preparation["attempt"],
            lease_id=preparation["claim_lease_id"],
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["released_unfinished_retry_id"] = retry_id
        basis = {
            key: value
            for key, value in receipt.items()
            if key
            not in {
                "operation_id",
                "phase",
                "prepared_at",
                "released_at",
                "receipt_id",
            }
        }
        receipt["operation_id"] = content_identity(basis)
        daemon_module.write_json_atomic(receipt_path, receipt)
    elif mutation == "receipt_prepared_before_start":
        receipt_path = restarted._task_claim_release_receipt_path(
            canonical_task_cid=preparation["canonical_task_cid"],
            attempt=preparation["attempt"],
            lease_id=preparation["claim_lease_id"],
        )
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["prepared_at"] = "2000-01-01T00:00:00Z"
        daemon_module.write_json_atomic(receipt_path, receipt)
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(mutation)

    rejected = restarted._replay_interrupted_implementation_claim_release(
        preparation=preparation,
        recovery_event=recovery_event,
    )

    assert rejected["blocked"] is True
    assert rejected["reconciled"] is False


def test_interrupted_claim_release_replays_legacy_missing_worktree_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, evidence, claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    claim.pop("worktree_root", None)
    daemon_module.write_json_atomic(claim_path, claim)
    original = daemon_module.write_json_atomic
    injected = {"raised": False}

    def crash_after_unlink(path: Path, payload: Any) -> None:
        if (
            isinstance(payload, dict)
            and payload.get("phase") == "released"
            and not injected["raised"]
        ):
            injected["raised"] = True
            raise RuntimeError("crash after claim unlink")
        original(path, payload)

    monkeypatch.setattr(daemon_module, "write_json_atomic", crash_after_unlink)
    with pytest.raises(RuntimeError, match="crash after claim unlink"):
        daemon.reconcile_interrupted_database_implementation_attempt(evidence)
    assert not claim_path.exists()
    monkeypatch.setattr(daemon_module, "write_json_atomic", original)

    replay = _restart_daemon(
        daemon
    ).reconcile_interrupted_database_implementation_attempt(evidence)

    assert replay["reconciled"] is True, replay
    assert replay["blocked"] is False


def test_interrupted_implementation_replays_prepared_claim_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, evidence, claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )
    original = daemon_module.write_json_atomic
    injected = {"raised": False}

    def crash_after_unlink(path: Path, payload: Any) -> None:
        if (
            isinstance(payload, dict)
            and payload.get("phase") == "released"
            and not injected["raised"]
        ):
            injected["raised"] = True
            raise RuntimeError("crash after claim unlink")
        original(path, payload)

    monkeypatch.setattr(daemon_module, "write_json_atomic", crash_after_unlink)
    with pytest.raises(RuntimeError, match="crash after claim unlink"):
        daemon.reconcile_interrupted_database_implementation_attempt(evidence)
    assert not claim_path.exists()
    monkeypatch.setattr(daemon_module, "write_json_atomic", original)

    replay = _restart_daemon(
        daemon
    ).reconcile_interrupted_database_implementation_attempt(
        evidence
    )

    assert replay["reconciled"] is True, replay
    events = _events(daemon)
    assert sum(
        event["type"] == "implementation_state_recovered"
        for event in events
    ) == 1


def test_interrupted_implementation_replays_released_receipt_event_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, evidence, claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )
    original_record = daemon._record_event

    def crash_before_release_event(
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        if event_type == "implementation_task_claim_released":
            raise RuntimeError("crash before claim release event")
        original_record(event_type, payload)

    monkeypatch.setattr(daemon, "_record_event", crash_before_release_event)
    with pytest.raises(RuntimeError, match="before claim release event"):
        daemon.reconcile_interrupted_database_implementation_attempt(evidence)
    assert not claim_path.exists()
    assert not any(
        event["type"] == "implementation_task_claim_released"
        for event in _events(daemon)
    )

    replay = _restart_daemon(
        daemon
    ).reconcile_interrupted_database_implementation_attempt(evidence)

    assert replay["reconciled"] is True, replay
    events = _events(daemon)
    assert sum(
        event["type"] == "implementation_state_recovered"
        for event in events
    ) == 1
    assert sum(
        event["type"] == "implementation_task_claim_released"
        for event in events
    ) == 1


def test_interrupted_implementation_replays_preparation_before_decrement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, evidence, claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )

    def crash_after_preparation(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("crash before counter decrement")

    monkeypatch.setattr(
        daemon,
        "_release_unfinished_active_attempt",
        crash_after_preparation,
    )
    with pytest.raises(RuntimeError, match="before counter decrement"):
        daemon.reconcile_interrupted_database_implementation_attempt(evidence)
    assert claim_path.exists()
    state = PortalTaskState.load(daemon.state_path)
    assert state.implementation_attempts["PCTDD-034"] == 1

    replay = _restart_daemon(
        daemon
    ).reconcile_interrupted_database_implementation_attempt(evidence)

    assert replay["reconciled"] is True, replay
    assert not claim_path.exists()
    events = _events(daemon)
    assert sum(
        event["type"] == "implementation_state_recovered"
        for event in events
    ) == 1


def test_interrupted_implementation_replays_decrement_before_recovery_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, evidence, claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )
    original_record = daemon._record_event

    def crash_before_recovery_event(
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        if event_type == "implementation_state_recovered":
            raise RuntimeError("crash after counter decrement")
        original_record(event_type, payload)

    monkeypatch.setattr(daemon, "_record_event", crash_before_recovery_event)
    with pytest.raises(RuntimeError, match="after counter decrement"):
        daemon.reconcile_interrupted_database_implementation_attempt(evidence)
    state = PortalTaskState.load(daemon.state_path)
    assert "PCTDD-034" not in state.implementation_attempts
    assert claim_path.exists()
    assert not any(
        event["type"] == "implementation_state_recovered"
        for event in _events(daemon)
    )

    replay = _restart_daemon(
        daemon
    ).reconcile_interrupted_database_implementation_attempt(evidence)

    assert replay["reconciled"] is True, replay
    assert not claim_path.exists()
    events = _events(daemon)
    assert sum(
        event["type"] == "implementation_state_recovered"
        for event in events
    ) == 1


def test_interrupted_implementation_holds_claim_guard_through_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, evidence, claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )
    original_release = daemon._release_unfinished_active_attempt
    replacement_results: list[str] = []

    def race_replacement(*args: Any, **kwargs: Any) -> dict[str, Any]:
        def replace() -> None:
            try:
                with serialized_lock_update(
                    claim_path,
                    timeout_seconds=0.05,
                ):
                    replacement_results.append("acquired")
            except TimeoutError:
                replacement_results.append("fenced")

        thread = threading.Thread(target=replace)
        thread.start()
        thread.join(timeout=1)
        assert not thread.is_alive()
        return original_release(*args, **kwargs)

    monkeypatch.setattr(
        daemon,
        "_release_unfinished_active_attempt",
        race_replacement,
    )

    result = daemon.reconcile_interrupted_database_implementation_attempt(
        evidence
    )

    assert result["reconciled"] is True, result
    assert replacement_results == ["fenced"]
    events = _events(daemon)
    assert sum(
        event["type"] == "implementation_task_claim_released"
        for event in events
    ) == 1


def test_interrupted_implementation_rejects_replaced_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    daemon, evidence, claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )
    original_release = daemon._release_unfinished_active_attempt

    def crash_after_preparation(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("crash after retry preparation")

    monkeypatch.setattr(
        daemon,
        "_release_unfinished_active_attempt",
        crash_after_preparation,
    )
    with pytest.raises(RuntimeError, match="crash after retry preparation"):
        daemon.reconcile_interrupted_database_implementation_attempt(evidence)
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    claim["lease_id"] = "replacement-lease"
    daemon_module.write_json_atomic(claim_path, claim)
    monkeypatch.setattr(
        daemon,
        "_release_unfinished_active_attempt",
        original_release,
    )

    result = daemon.reconcile_interrupted_database_implementation_attempt(
        evidence
    )

    assert result["blocked"] is True
    assert result["reason"] == "interrupted_implementation_claim_changed"
    state = PortalTaskState.load(daemon.state_path)
    assert state.implementation_attempts["PCTDD-034"] == 1
    assert not any(
        event["type"] == "implementation_state_recovered"
        for event in _events(daemon)
    )


def test_interrupted_implementation_rejects_truncated_state_identity(
    tmp_path: Path,
) -> None:
    daemon, evidence, _claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )
    raw = json.loads(daemon.state_path.read_text(encoding="utf-8"))
    raw["task_identities"]["PCTDD-034"].pop("display_task_id")
    daemon_module.write_json_atomic(daemon.state_path, raw)

    result = daemon.reconcile_interrupted_database_implementation_attempt(
        evidence
    )

    assert result["blocked"] is True
    assert result["reason"] == (
        "interrupted_implementation_task_identity_changed"
    )
    state = PortalTaskState.load(daemon.state_path)
    assert state.implementation_attempts["PCTDD-034"] == 1
    assert not any(
        event["type"] == "interrupted_implementation_retry_prepared"
        for event in _events(daemon)
    )


@pytest.mark.parametrize("invalid_count", [0, False, "0"])
def test_interrupted_implementation_rejects_noncanonical_zero_counters(
    tmp_path: Path,
    invalid_count: Any,
) -> None:
    daemon, evidence, _claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )
    raw = json.loads(daemon.state_path.read_text(encoding="utf-8"))
    task_cid = raw["last_implementation_task_cid"]
    raw["implementation_attempts"]["PCTDD-034"] = invalid_count
    raw["implementation_attempts_by_cid"][task_cid] = invalid_count
    daemon_module.write_json_atomic(daemon.state_path, raw)

    result = daemon.reconcile_interrupted_database_implementation_attempt(
        evidence
    )

    assert result["blocked"] is True
    assert result["reason"] == (
        "interrupted_implementation_attempt_count_invalid"
    )
    assert not any(
        event["type"] == "interrupted_implementation_retry_prepared"
        for event in _events(daemon)
    )


def test_interrupted_implementation_rejects_terminal_completion(
    tmp_path: Path,
) -> None:
    daemon, evidence, _claim_path, _dirty_candidate = (
        _seed_interrupted_implementation(tmp_path)
    )
    receipt = evidence["reconciliation_receipt"]
    nested = receipt["nested_state"]
    daemon._record_event(
        "task_completed",
        {
            "task_id": nested["active_task_id"],
            "canonical_task_cid": receipt["portal_reconciliation"][
                "task_claim_reconciliation"
            ]["canonical_task_cid"],
            "completion_receipt_id": content_identity({"completed": True}),
        },
    )

    result = daemon.reconcile_interrupted_database_implementation_attempt(
        evidence
    )

    assert result["blocked"] is True
    assert result["reason"] == "interrupted_implementation_terminal_present"
    state = PortalTaskState.load(daemon.state_path)
    assert state.implementation_attempts["PCTDD-034"] == 1


def test_historical_bridge_routes_only_to_interrupted_retry_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = SimpleNamespace(
        attempt_id="outer-attempt",
        claim_id="outer-claim",
        task_cid="outer-historical-cid",
        task_alias="PCTDD-034",
        attempt_number=1,
        owner_session_id="outer-owner",
        fencing_token=7,
        fence_epoch=3,
        lease_id="outer-lease",
        body={
            "control_claim": {
                "revision": 1,
                "execution_spec_cid": "exec-spec",
                "validation_spec_cid": "validation-spec",
            }
        },
    )
    received: list[dict[str, Any]] = []

    class RetryOnlyPortal:
        def reconcile_quiesced_active_attempt(self) -> dict[str, Any]:
            raise AssertionError("ordinary reconciliation was selected")

        def reconcile_interrupted_database_implementation_attempt(
            self,
            evidence: dict[str, Any],
        ) -> dict[str, Any]:
            received.append(evidence)
            return {
                "reconciled": True,
                "blocked": False,
                "reason": "interrupted_implementation_recovered_for_retry",
                "provider_dispatched": False,
                "implementation_dispatched": False,
                "acceptance_inferred": False,
            }

        def reconcile_provider_forbidden_terminal_result(
            self,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            raise AssertionError("historical retry entered terminal recovery")

        def close_event_runtime(self) -> None:
            return None

    bridge = DatabasePortalExecutionBridge(
        task_source=object(),
        attempt_root=tmp_path / "attempts",
        portal_factory=lambda _paths, _alias: RetryOnlyPortal(),
    )
    bridge.attempt_root.mkdir()
    paths = bridge._paths(attempt)
    paths.root.mkdir()
    paths.binding.write_text("{}\n", encoding="utf-8")
    paths.task_projection.write_text("projection\n", encoding="utf-8")
    paths.state.write_text("{}\n", encoding="utf-8")
    state_digest = "sha256:" + "7" * 64
    workspace = str(tmp_path / "retained-candidate")
    binding = {
        "attempt_id": attempt.attempt_id,
        "claim_id": attempt.claim_id,
        "task_cid": attempt.task_cid,
        "task_alias": attempt.task_alias,
        "task_revision": 1,
        "fencing_token": attempt.fencing_token,
        "fence_epoch": attempt.fence_epoch,
        "lease_id": attempt.lease_id,
        "projection_immutable_digest": "sha256:" + "8" * 64,
        "binding_id": "sha256:" + "9" * 64,
    }
    nested = {
        "present": True,
        "state_path": str(paths.state),
        "state_digest": state_digest,
        "active": True,
        "active_phase": "implementing",
        "active_task_id": attempt.task_alias,
        "active_attempt": 1,
        "active_worktree_path": workspace,
        "active_branch": "implementation/pctdd-034-attempt-1",
    }
    bridge.persist_reconciliation_receipt(
        attempt,
        {
            "stage": "blocked",
            "trigger": "supervisor_signal_shutdown",
            "reconciled_at": _timestamp(-1),
            "reconciled": False,
            "blocked": True,
            "reason": "nested_portal_attempt_reconciliation_blocked",
            "binding_id": binding["binding_id"],
            "task_alias": attempt.task_alias,
            "nested_state": nested,
            "provider_runner_fence": {
                "applicable": True,
                "fenced": True,
                "safe_to_restart": True,
                "reason": "ordinary_provider_runner_exact_birth_fenced",
                "pid": 2**30 - 73,
            },
            "provider_runner_reconciliation_authority": (
                "ordinary_provider_runner_fence"
            ),
            "portal_reconciliation": {
                "blocked": True,
                "reconciled": False,
                "reason": "task_claim_reconciliation_blocked",
                "protected_path_reconciliation": {
                    "blocked": False,
                    "reason": "crash_reconciliation_unchanged",
                    "task_id": attempt.task_alias,
                    "attempt": 1,
                    "workspace_path": workspace,
                },
                "worktree_lifecycle_reconciliation": {
                    "blocked": False,
                    "reconciled": True,
                    "state": "terminal",
                    "task_id": attempt.task_alias,
                    "attempt": 1,
                    "workspace_path": workspace,
                    "record_id": "lifecycle-record",
                    "fence": 5,
                },
                "task_claim_reconciliation": {
                    "blocked": True,
                    "reconciled": False,
                    "reason": "canonical_task_not_terminal",
                    "observed_task_status": "todo",
                    "task_id": attempt.task_alias,
                    "canonical_task_cid": "nested-current-cid",
                },
                "attempt_recovery": {
                    "consumed": False,
                    "attempt": 1,
                    "task_id": attempt.task_alias,
                    "canonical_task_cid": "nested-current-cid",
                    "previous_display_count": 1,
                    "previous_cid_count": 1,
                },
            },
            "terminal_provider_evidence": False,
        },
    )
    current_binding = {**binding, "binding_id": "sha256:" + "a" * 64}
    monkeypatch.setattr(bridge, "_record_for_attempt", lambda *_args: object())
    monkeypatch.setattr(bridge, "_render_projection", lambda *_args: "current\n")
    monkeypatch.setattr(bridge, "_binding", lambda *_args: current_binding)
    monkeypatch.setattr(bridge, "_read_binding", lambda _path: dict(binding))
    monkeypatch.setattr(bridge, "_verify_binding_identity", lambda _value: None)
    monkeypatch.setattr(
        bridge,
        "_verify_projection",
        lambda *_args: "projection\n",
    )
    monkeypatch.setattr(
        bridge,
        "_projection_task_identity",
        lambda *_args: {
            "task_id": attempt.task_alias,
            "canonical_task_key": "task/v1/current",
            "canonical_task_cid": "nested-current-cid",
            "board_namespace": "pctdd",
        },
    )
    monkeypatch.setattr(
        bridge,
        "_strict_state_record",
        lambda _path: ({"active_phase": "implementing"}, state_digest),
    )
    monkeypatch.setattr(
        bridge,
        "_verify_nested_state_identity",
        lambda *_args, **_kwargs: dict(nested),
    )
    bridge._binding_lookup = lambda _attempt: {
        **binding,
        "stage": "portal_entered",
    }
    monkeypatch.setattr(
        bridge,
        "recover_provider_result",
        lambda _attempt: pytest.fail("historical retry queried provider result"),
    )
    monkeypatch.setattr(
        portal_supervisor,
        "fence_ordinary_provider_runner",
        lambda *_args, **_kwargs: {
            "applicable": True,
            "fenced": True,
            "safe_to_restart": True,
            "reason": "ordinary_provider_runner_exact_birth_fenced",
            "pid": 2**30 - 73,
        },
    )

    result = bridge.reconcile_quiesced_attempt(attempt)

    assert result["reconciled"] is True, result
    assert result["historical_binding"] is True
    assert result["terminal_provider_evidence"] is False
    assert len(received) == 1
    assert received[0]["schema"].endswith(
        "database-portal-interrupted-implementation-retry@1"
    )
