"""Regressions for bounded ordinary-task stale-replay recovery.

The recovery is intentionally narrower than a general retry-budget reset:

* a ready ordinary task must have a complete final-attempt event segment that
  proves its accepted proposal was rejected unchanged by same-attempt rescue
  solely because the proposal receipt had already been consumed;
* one durable fingerprint grants exactly one fresh attempt for each recovery
  runtime revision; and
* same-attempt rescue revalidation may replay the exact live accepted proposal,
  but may not widen that exception across tasks or repository trees.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.event_log import (
    append_jsonl_event,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
    PortalTaskState,
    parse_task_file,
)

TASK_PREFIX = "## TEST-"
TASK_ID = "TEST-301"
MAX_ATTEMPTS = 3
REVISION_A = "ordinary-stale-replay-recovery-a"
REVISION_B = "ordinary-stale-replay-recovery-b"
PROPOSAL_ID = "2099ef810bde44023020329be8d2badac1fc062760f6b833881244833aa475f9"
REPOSITORY_TREE_ID = "ddebe2cc99e0c9c3b58196eda2112e238de065f0"
CHANGED_PATHS = (
    "ipfs_kit_py/ipfs_kit_py/kernel_vfs/wal_recovery.py",
    "ipfs_kit_py/tests/kernel_vfs/wal/test_mount_recovery.py",
)
BRANCH = "implementation/test-301-attempt-3"
WORKTREE = "/tmp/test-301-attempt-3"


def _ordinary_task_board(path: Path) -> None:
    path.write_text(
        """# Tasks

## TEST-301 Recover WAL-backed mounts

- Status: todo
- Completion: auto
- Priority: P0
- Track: vfs
- Outputs: ipfs_kit_py/ipfs_kit_py/kernel_vfs/wal_recovery.py, ipfs_kit_py/tests/kernel_vfs/wal/test_mount_recovery.py
- Validation: python -m pytest -q ipfs_kit_py/tests/kernel_vfs/wal/test_mount_recovery.py
- Acceptance: WAL-backed mounts recover without losing committed mutations.
""",
        encoding="utf-8",
    )


def _daemon(repo: Path, todo_path: Path) -> PortalImplementationDaemon:
    state_dir = repo / "state"
    return PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix=TASK_PREFIX,
        max_task_attempts=MAX_ATTEMPTS,
        worktree_pool_enabled=False,
    )


def _event(
    sequence: int,
    event_type: str,
    task: PortalTask,
    *,
    canonical_task_key: str,
    canonical_task_cid: str,
    **payload: object,
) -> dict[str, object]:
    return {
        "type": event_type,
        "timestamp": f"2026-08-10T12:{sequence:02d}:00+00:00",
        "sequence": sequence,
        "event_id": f"sha256:{sequence:064x}",
        "previous_event_id": (
            f"sha256:{sequence - 1:064x}" if sequence > 1 else ""
        ),
        "stream_id": "stream:test-stale-replay",
        "snapshot_id": "snapshot:test-stale-replay",
        "task_id": task.task_id,
        "canonical_task_key": canonical_task_key,
        "canonical_task_cid": canonical_task_cid,
        **payload,
    }


def _stale_replay_attempt_events(
    task: PortalTask,
    *,
    canonical_task_key: str,
    canonical_task_cid: str,
) -> list[dict[str, object]]:
    accepted_gate = {
        "attempted": True,
        "accepted": True,
        "reason_codes": [],
        "proposal_id": PROPOSAL_ID,
        "policy_id": "sha256:accepted-policy",
        "receipt_id": "sha256:accepted-receipt",
        "repository_tree_id": REPOSITORY_TREE_ID,
        "changed_paths": list(CHANGED_PATHS),
    }
    rejected_gate = {
        "attempted": True,
        "accepted": False,
        "reason_codes": ["stale_proposal_replay"],
        "proposal_id": PROPOSAL_ID,
        # A same-body replay is validated under a consumed-proposal policy, so
        # production policy/receipt IDs legitimately differ from admission.
        "policy_id": "sha256:consumed-policy",
        "receipt_id": "sha256:rejected-receipt",
        "repository_tree_id": REPOSITORY_TREE_ID,
        "changed_paths": list(CHANGED_PATHS),
    }
    terminal_validation = {
        "attempted": True,
        "passed": False,
        "returncode": 1,
        "reason": "proposal_gate_failed",
        "error": "proposal_validation_failed",
        "proposal_gate": deepcopy(rejected_gate),
        "results": [],
    }
    return [
        _event(
            1,
            "implementation_started",
            task,
            canonical_task_key=canonical_task_key,
            canonical_task_cid=canonical_task_cid,
            attempt=MAX_ATTEMPTS,
            branch=BRANCH,
            worktree_path=WORKTREE,
            baseline_ref=REPOSITORY_TREE_ID,
        ),
        _event(
            2,
            "implementation_proposal_validated",
            task,
            canonical_task_key=canonical_task_key,
            canonical_task_cid=canonical_task_cid,
            **accepted_gate,
        ),
        _event(
            3,
            "implementation_failure_reviewed",
            task,
            canonical_task_key=canonical_task_key,
            canonical_task_cid=canonical_task_cid,
            attempt=MAX_ATTEMPTS,
            decision="guide_rescue",
            reason_codes=["validation_command_failed"],
            worktree_path=WORKTREE,
        ),
        _event(
            4,
            "implementation_auto_rescue_provider_started",
            task,
            canonical_task_key=canonical_task_key,
            canonical_task_cid=canonical_task_cid,
            attempt=MAX_ATTEMPTS,
            plan={
                "action": "inline_provider_rescue",
                "reason": "validation_failed_provider_rescue",
                "reason_codes": ["validation_command_failed"],
            },
            failed_commands=[task.validation[0]],
        ),
        _event(
            5,
            "implementation_proposal_rejected",
            task,
            canonical_task_key=canonical_task_key,
            canonical_task_cid=canonical_task_cid,
            **rejected_gate,
        ),
        _event(
            6,
            "failed_validation_worktree_preserved",
            task,
            canonical_task_key=canonical_task_key,
            canonical_task_cid=canonical_task_cid,
            attempt=MAX_ATTEMPTS,
            branch=BRANCH,
            worktree_path=WORKTREE,
            preserved=True,
            rescue_branch="rescue/test-301-attempt-3-failed-validation",
            implementation_commit=(
                "073254e5c6651919a93556ac639a0ba0025de1e4"
            ),
            validation_result=deepcopy(terminal_validation),
        ),
        _event(
            7,
            "implementation_finished",
            task,
            canonical_task_key=canonical_task_key,
            canonical_task_cid=canonical_task_cid,
            attempt=MAX_ATTEMPTS,
            branch=BRANCH,
            worktree_path=WORKTREE,
            attempt_consumed=True,
            provider_dispatched=True,
            returncode=78,
            validation_result=deepcopy(terminal_validation),
            merge_result={
                "attempted": False,
                "merged": False,
                "reason": "not_attempted",
            },
        ),
    ]


def _write_events(path: Path, events: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    path.with_name(f"{path.name}.manifest.json").unlink(missing_ok=True)
    reserved = {
        "type",
        "stream_id",
        "snapshot_id",
        "sequence",
        "event_id",
        "previous_event_id",
    }
    for event in events:
        append_jsonl_event(
            path,
            str(event["type"]),
            {
                key: value
                for key, value in event.items()
                if key not in reserved
            },
        )


def _exhausted_ordinary_task(
    tmp_path: Path,
) -> tuple[
    PortalImplementationDaemon,
    PortalTask,
    PortalTaskState,
    str,
    str,
]:
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "tasks.todo.md"
    _ordinary_task_board(todo_path)
    daemon = _daemon(repo, todo_path)
    task = parse_task_file(todo_path, TASK_PREFIX)[0]
    daemon._register_task_identities((task,))
    identity = daemon._identity_for_task(task)
    state = PortalTaskState(
        task_identities={task.task_id: identity.to_dict()},
        implementation_attempts={task.task_id: MAX_ATTEMPTS},
        implementation_attempts_by_cid={
            identity.canonical_task_cid: MAX_ATTEMPTS
        },
        stale_proposal_replay_rearm_receipts={},
    )
    state.save(daemon.state_path)
    daemon.strategy_path.parent.mkdir(parents=True, exist_ok=True)
    daemon.strategy_path.write_text(
        json.dumps(
            {
                "blocked_tasks": ["TEST-999"],
                "focus_tracks": ["vfs"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    daemon.task_queue.register_task(identity).record_failure(
        "proposal_gate_failed"
    )
    daemon.task_queue.save()
    _write_events(
        daemon.events_path,
        _stale_replay_attempt_events(
            task,
            canonical_task_key=identity.canonical_task_key,
            canonical_task_cid=identity.canonical_task_cid,
        ),
    )
    return (
        daemon,
        task,
        state,
        identity.canonical_task_key,
        identity.canonical_task_cid,
    )


def _restore_exhausted_attempt(
    daemon: PortalImplementationDaemon,
    task: PortalTask,
    canonical_task_cid: str,
) -> PortalTaskState:
    state = PortalTaskState.load(daemon.state_path)
    state.implementation_attempts[task.task_id] = MAX_ATTEMPTS
    state.implementation_attempts_by_cid[
        canonical_task_cid
    ] = MAX_ATTEMPTS
    state.save(daemon.state_path)
    daemon.task_queue.record_failure(
        canonical_task_cid,
        "proposal_gate_failed",
    )
    daemon.task_queue.save()
    return state


def test_ordinary_stale_replay_rearm_persists_once_per_revision_aba(
    tmp_path: Path,
) -> None:
    """A/B revisions each grant one attempt; returning to A grants none."""

    daemon, task, state, _task_key, task_cid = _exhausted_ordinary_task(
        tmp_path
    )
    todo_before = daemon.todo_path.read_bytes()
    strategy_before = daemon.strategy_path.read_bytes()

    first = daemon._rearm_attempt_limited_stale_proposal_tasks(
        state,
        (task,),
        {task.task_id: "ready"},
        recovery_revision=REVISION_A,
    )
    persisted_a = PortalTaskState.load(daemon.state_path)
    restarted_a = _daemon(daemon.repo_root, daemon.todo_path)
    restarted_a._register_task_identities((task,))

    assert len(first) == 1
    assert first[0]["task_id"] == task.task_id
    assert first[0]["canonical_task_cid"] == task_cid
    assert first[0]["recovery_revision"] == REVISION_A
    assert first[0]["rearm_fingerprint"]
    assert persisted_a.implementation_attempts[task.task_id] == (
        MAX_ATTEMPTS - 1
    )
    assert persisted_a.implementation_attempts_by_cid[task_cid] == (
        MAX_ATTEMPTS - 1
    )
    assert persisted_a.stale_proposal_replay_rearm_receipts == {
        task_cid: [first[0]["rearm_fingerprint"]]
    }
    assert restarted_a.task_queue.is_cooled_down(task_cid) is False
    assert daemon.todo_path.read_bytes() == todo_before
    assert daemon.strategy_path.read_bytes() == strategy_before
    assert any(
        event.get("type") == "ordinary_task_stale_proposal_replay_rearmed"
        for event in daemon._iter_events()
    )

    # Model the single newly granted attempt failing, then restart. The same
    # runtime revision must not reset the task a second time.
    exhausted_a = _restore_exhausted_attempt(
        restarted_a,
        task,
        task_cid,
    )
    repeated_a = restarted_a._rearm_attempt_limited_stale_proposal_tasks(
        exhausted_a,
        (task,),
        {task.task_id: "ready"},
        recovery_revision=REVISION_A,
    )
    still_exhausted_a = PortalTaskState.load(restarted_a.state_path)

    assert repeated_a == []
    assert still_exhausted_a.implementation_attempts[task.task_id] == (
        MAX_ATTEMPTS
    )
    assert still_exhausted_a.implementation_attempts_by_cid[task_cid] == (
        MAX_ATTEMPTS
    )

    second = restarted_a._rearm_attempt_limited_stale_proposal_tasks(
        still_exhausted_a,
        (task,),
        {task.task_id: "ready"},
        recovery_revision=REVISION_B,
    )
    persisted_b = PortalTaskState.load(restarted_a.state_path)

    assert len(second) == 1
    assert second[0]["recovery_revision"] == REVISION_B
    assert second[0]["rearm_fingerprint"] != first[0]["rearm_fingerprint"]
    assert persisted_b.stale_proposal_replay_rearm_receipts == {
        task_cid: [
            first[0]["rearm_fingerprint"],
            second[0]["rearm_fingerprint"],
        ]
    }
    assert persisted_b.implementation_attempts[task.task_id] == (
        MAX_ATTEMPTS - 1
    )
    assert persisted_b.implementation_attempts_by_cid[task_cid] == (
        MAX_ATTEMPTS - 1
    )

    returned_daemon = _daemon(daemon.repo_root, daemon.todo_path)
    returned_daemon._register_task_identities((task,))
    exhausted_b = _restore_exhausted_attempt(
        returned_daemon,
        task,
        task_cid,
    )
    returned_to_a = (
        returned_daemon._rearm_attempt_limited_stale_proposal_tasks(
            exhausted_b,
            (task,),
            {task.task_id: "ready"},
            recovery_revision=REVISION_A,
        )
    )
    final_state = PortalTaskState.load(returned_daemon.state_path)

    assert returned_to_a == []
    assert final_state.implementation_attempts[task.task_id] == MAX_ATTEMPTS
    assert final_state.implementation_attempts_by_cid[task_cid] == (
        MAX_ATTEMPTS
    )
    assert final_state.stale_proposal_replay_rearm_receipts == (
        persisted_b.stale_proposal_replay_rearm_receipts
    )
    assert daemon.todo_path.read_bytes() == todo_before
    assert daemon.strategy_path.read_bytes() == strategy_before


def test_daemon_pass_rearms_before_attempt_limit_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live selection path can dispatch the one recovered attempt."""

    daemon, task, _state, _task_key, task_cid = _exhausted_ordinary_task(
        tmp_path
    )
    monkeypatch.setattr(
        daemon,
        "_retry_budget_repair_runtime_revision",
        lambda: REVISION_A,
    )

    result = daemon.run_once()
    persisted = PortalTaskState.load(daemon.state_path)

    assert result["active_task_id"] == task.task_id
    assert result["attempt_limited_task_ids"] == []
    assert result["stale_proposal_replay_rearms"][0]["task_id"] == (
        task.task_id
    )
    assert persisted.implementation_attempts[task.task_id] == (
        MAX_ATTEMPTS - 1
    )
    assert persisted.implementation_attempts_by_cid[task_cid] == (
        MAX_ATTEMPTS - 1
    )
    assert persisted.active_task_id == task.task_id
    assert persisted.selection_idle_reason == ""


@pytest.mark.parametrize(
    "invalid_proof",
    (
        "mixed_rejection",
        "proposal_mismatch",
        "accepted_event_missing",
        "rescue_event_missing",
        "preservation_event_missing",
        "attempt_not_consumed",
        "canonical_identity_mismatch",
        "task_active",
    ),
)
def test_ordinary_stale_replay_rearm_fails_closed_without_exact_proof(
    tmp_path: Path,
    invalid_proof: str,
) -> None:
    """Partial, mixed, or cross-identity evidence never earns new budget."""

    daemon, task, state, task_key, task_cid = _exhausted_ordinary_task(
        tmp_path
    )
    events = _stale_replay_attempt_events(
        task,
        canonical_task_key=task_key,
        canonical_task_cid=task_cid,
    )
    if invalid_proof == "mixed_rejection":
        rejected = next(
            event
            for event in events
            if event["type"] == "implementation_proposal_rejected"
        )
        rejected["reason_codes"] = [
            "stale_proposal_replay",
            "path_outside_scope",
        ]
    elif invalid_proof == "proposal_mismatch":
        rejected = next(
            event
            for event in events
            if event["type"] == "implementation_proposal_rejected"
        )
        rejected["proposal_id"] = "different-proposal"
    elif invalid_proof == "accepted_event_missing":
        events = [
            event
            for event in events
            if event["type"] != "implementation_proposal_validated"
        ]
    elif invalid_proof == "rescue_event_missing":
        events = [
            event
            for event in events
            if event["type"]
            != "implementation_auto_rescue_provider_started"
        ]
    elif invalid_proof == "preservation_event_missing":
        events = [
            event
            for event in events
            if event["type"] != "failed_validation_worktree_preserved"
        ]
    elif invalid_proof == "attempt_not_consumed":
        finished = next(
            event
            for event in events
            if event["type"] == "implementation_finished"
        )
        finished["attempt_consumed"] = False
    elif invalid_proof == "canonical_identity_mismatch":
        rejected = next(
            event
            for event in events
            if event["type"] == "implementation_proposal_rejected"
        )
        rejected["canonical_task_cid"] = "baguqeera-wrong-task"
    elif invalid_proof == "task_active":
        state.implementation_in_progress = True
        state.active_task_id = task.task_id
        state.active_task_cid = task_cid
    _write_events(daemon.events_path, events)
    state.save(daemon.state_path)
    todo_before = daemon.todo_path.read_bytes()
    strategy_before = daemon.strategy_path.read_bytes()

    rearmed = daemon._rearm_attempt_limited_stale_proposal_tasks(
        state,
        (task,),
        {task.task_id: "ready"},
        recovery_revision=REVISION_A,
    )
    persisted = PortalTaskState.load(daemon.state_path)

    assert rearmed == []
    assert persisted.implementation_attempts[task.task_id] == MAX_ATTEMPTS
    assert persisted.implementation_attempts_by_cid[task_cid] == MAX_ATTEMPTS
    assert persisted.stale_proposal_replay_rearm_receipts == {}
    assert daemon.task_queue.is_cooled_down(task_cid) is True
    assert daemon.todo_path.read_bytes() == todo_before
    assert daemon.strategy_path.read_bytes() == strategy_before


def _live_proposal_validation(
    *,
    task_id: str = TASK_ID,
    repository_tree_id: str = REPOSITORY_TREE_ID,
    proposal_id: str = PROPOSAL_ID,
    accepted: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        accepted=accepted,
        proposal=SimpleNamespace(
            task_id=task_id,
            repository_tree_id=repository_tree_id,
            proposal_id=proposal_id,
        ),
    )


def test_same_attempt_replayable_ids_require_exact_live_accepted_proposal(
    tmp_path: Path,
) -> None:
    """The replay exception is bound to the live task/tree proposal object."""

    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = _daemon(repo, repo / "tasks.todo.md")
    seed_id = "seed-proposal"
    validation_result = {
        "proposal_validation": _live_proposal_validation(),
        # A compact gate alone is not the authority; the live object above is.
        "proposal_gate": {
            "accepted": True,
            "proposal_id": PROPOSAL_ID,
            "repository_tree_id": REPOSITORY_TREE_ID,
        },
    }

    replayable = daemon._same_attempt_replayable_proposal_ids(
        validation_result,
        task_id=TASK_ID,
        repository_tree_id=REPOSITORY_TREE_ID,
        seed_proposal_ids=(seed_id, PROPOSAL_ID, seed_id),
    )

    assert replayable == tuple(sorted((seed_id, PROPOSAL_ID)))

    invalid_live_proposals = (
        _live_proposal_validation(accepted=False),
        _live_proposal_validation(task_id="TEST-OTHER"),
        _live_proposal_validation(repository_tree_id="different-tree"),
        _live_proposal_validation(proposal_id=""),
    )
    for live_proposal in invalid_live_proposals:
        assert daemon._same_attempt_replayable_proposal_ids(
            {"proposal_validation": live_proposal},
            task_id=TASK_ID,
            repository_tree_id=REPOSITORY_TREE_ID,
            seed_proposal_ids=(seed_id,),
        ) == (seed_id,)
    assert daemon._same_attempt_replayable_proposal_ids(
        {
            "proposal_gate": {
                "accepted": True,
                "proposal_id": PROPOSAL_ID,
                "repository_tree_id": REPOSITORY_TREE_ID,
            }
        },
        task_id=TASK_ID,
        repository_tree_id=REPOSITORY_TREE_ID,
        seed_proposal_ids=(seed_id,),
    ) == (seed_id,)


def test_auto_rescue_forwards_same_attempt_accepted_proposal_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage rescue revalidation receives both seed and live proposal IDs."""

    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = _daemon(repo, repo / "tasks.todo.md")
    task = PortalTask(
        task_id=TASK_ID,
        title="Recover WAL-backed mounts",
        status="todo",
        completion="auto",
        priority="P0",
        track="vfs",
        outputs=[CHANGED_PATHS[0]],
        validation=["python -m pytest -q test_wal_recovery.py"],
    )
    workspace = repo / "worktree"
    workspace.mkdir()
    log_path = repo / "state" / "implementation.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("validation failed\n", encoding="utf-8")
    live_proposal = _live_proposal_validation()
    captured_replayable_ids: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        daemon,
        "_expected_outputs_present_on_disk",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        daemon,
        "_dirty_in_scope_declared_output_paths",
        lambda *_args, **_kwargs: (CHANGED_PATHS[0],),
    )
    monkeypatch.setattr(
        daemon,
        "_stage_declared_candidate_outputs",
        lambda *_args, **_kwargs: (CHANGED_PATHS[0],),
    )
    monkeypatch.setattr(daemon, "_record_event", lambda *_args, **_kwargs: None)

    def revalidate(*_args: object, **kwargs: object) -> dict[str, object]:
        captured_replayable_ids.append(
            tuple(kwargs["replayable_consumed_proposal_ids"])
        )
        return {
            "attempted": True,
            "passed": True,
            "returncode": 0,
            "proposal_validation": live_proposal,
            "proposal_gate": {
                "accepted": True,
                "proposal_id": PROPOSAL_ID,
                "repository_tree_id": REPOSITORY_TREE_ID,
            },
        }

    monkeypatch.setattr(
        daemon,
        "_run_validation_with_candidate_binding",
        revalidate,
    )
    monkeypatch.setattr(
        daemon,
        "_apply_implementation_failure_review",
        lambda **kwargs: dict(kwargs["validation_result"]),
    )

    result = daemon._automatic_implementation_rescue(
        task=task,
        attempt=MAX_ATTEMPTS,
        workspace_path=workspace,
        branch_name=BRANCH,
        baseline_ref=REPOSITORY_TREE_ID,
        validation_result={
            "attempted": True,
            "passed": False,
            "returncode": 1,
            "reason": "proposal_gate_failed",
            "error": "proposal_validation_failed",
            "finding_codes": [
                "empty_patch",
                "expected_output_ignored_or_unstaged",
            ],
            "proposal_validation": live_proposal,
            "proposal_gate": {
                "accepted": True,
                "proposal_id": PROPOSAL_ID,
                "repository_tree_id": REPOSITORY_TREE_ID,
            },
            "failure_review": {
                "decision": "guide_rescue",
                "reason_codes": [
                    "proposal_gate_failed",
                    "empty_or_no_change",
                ],
                "finding_codes": [
                    "empty_patch",
                    "expected_output_ignored_or_unstaged",
                ],
            },
        },
        log_path=log_path,
        state=None,
        command=(),
        allow_provider_rescue=False,
        replayable_consumed_proposal_ids=("seed-proposal",),
    )

    assert result["passed"] is True
    assert captured_replayable_ids == [
        tuple(sorted(("seed-proposal", PROPOSAL_ID))),
    ]
