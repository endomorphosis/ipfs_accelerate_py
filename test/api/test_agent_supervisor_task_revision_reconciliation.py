from __future__ import annotations

from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTask,
    PortalTaskState,
)


def test_revised_task_ignores_prior_revision_merge_and_finish_history(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = PortalImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        implement=False,
    )
    current = PortalTask(
        task_id="ACCEL-001",
        title="Revised contract",
        status="todo",
        completion="auto",
        priority="P1",
        track="ops",
        outputs=["feature.py"],
        acceptance="Strict current revision acceptance.",
    )
    prior = replace(
        current,
        title="Prior contract",
        acceptance="Superseded acceptance.",
    )
    current_cid = daemon._canonical_ref(current)
    prior_cid = daemon._canonical_ref(prior)
    assert current_cid != prior_cid

    prior_event = {
        "type": "implementation_finished",
        "task_id": current.task_id,
        "canonical_task_cid": prior_cid,
        "attempt": 1,
        "implementation_commit": "a" * 40,
        "returncode": 0,
        "validation_result": {"attempted": True, "passed": True},
        "merge_result": {
            "attempted": True,
            "merged": False,
            "queued": True,
            "request_id": "prior-request",
            "canonical_task_cid": prior_cid,
            "completion_task_cids": {current.task_id: prior_cid},
            "reason": "merge_failed",
        },
        "cleanup_result": {"cleaned": True},
    }
    monkeypatch.setattr(daemon, "_load_tasks", lambda: [current])
    monkeypatch.setattr(daemon, "_iter_events", lambda: [prior_event])
    monkeypatch.setattr(
        daemon,
        "_iter_merge_lifecycle_events",
        lambda: [prior_event],
    )
    monkeypatch.setattr(daemon, "_main_branch_name", lambda: "main")
    monkeypatch.setattr(daemon, "_git_ref_is_ancestor", lambda *_: False)

    assert daemon._latest_implementation_finished_by_task() == {}
    assert daemon._pending_queued_merge_task_ids(
        {current.task_id: prior_event},
    ) == set()
    assert daemon._quarantined_queued_merge_task_ids(
        {current.task_id: prior_event},
    ) == set()
    assert daemon._failed_merge_candidates() == []
    assert daemon._unresolved_merge_failures_by_task() == {}
    assert (
        daemon._task_attempt_has_implementation_finish(
            current.task_id,
            1,
            canonical_task_cid=current_cid,
        )
        is False
    )
    assert not daemon._has_unresolved_merge_failure(
        current,
        PortalTaskState(
            last_implementation_task_id=current.task_id,
            last_implementation_task_cid=prior_cid,
            last_implementation_commit="a" * 40,
            last_merge_returncode=1,
        ),
    )

    current_event = {
        **prior_event,
        "canonical_task_cid": current_cid,
        "merge_result": {
            **prior_event["merge_result"],
            "canonical_task_cid": current_cid,
            "completion_task_cids": {current.task_id: current_cid},
        },
    }
    monkeypatch.setattr(daemon, "_iter_events", lambda: [current_event])
    monkeypatch.setattr(
        daemon,
        "_iter_merge_lifecycle_events",
        lambda: [current_event],
    )
    assert daemon._latest_implementation_finished_by_task() == {
        current.task_id: current_event,
    }
    assert daemon._pending_queued_merge_task_ids() == {current.task_id}
    assert daemon._failed_merge_candidates() == [current_event]
    assert set(daemon._unresolved_merge_failures_by_task()) == {
        current.task_id,
    }
    assert daemon._task_attempt_has_implementation_finish(
        current.task_id,
        1,
        canonical_task_cid=current_cid,
    )

    contradictory_event = {
        **current_event,
        "task_cid": prior_cid,
    }
    monkeypatch.setattr(
        daemon,
        "_iter_events",
        lambda: [contradictory_event],
    )
    monkeypatch.setattr(
        daemon,
        "_iter_merge_lifecycle_events",
        lambda: [contradictory_event],
    )
    assert daemon._latest_implementation_finished_by_task() == {}
    assert daemon._failed_merge_candidates() == []


def test_reconciled_event_from_prior_cid_cannot_cancel_current_candidate(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = PortalImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        implement=False,
    )
    task = PortalTask(
        task_id="ACCEL-001",
        title="Current task",
        status="todo",
        completion="auto",
        priority="P1",
        track="ops",
        outputs=["feature.py"],
    )
    current_cid = daemon._canonical_ref(task)
    implementation_commit = "b" * 40
    finished = {
        "type": "implementation_finished",
        "task_id": task.task_id,
        "canonical_task_cid": current_cid,
        "implementation_commit": implementation_commit,
        "validation_result": {"attempted": True, "passed": True},
        "merge_result": {
            "attempted": True,
            "merged": False,
            "canonical_task_cid": current_cid,
            "reason": "merge_failed",
        },
        "cleanup_result": {"cleaned": True},
    }
    stale_reconciled = {
        "type": "merge_reconciled",
        "task_id": task.task_id,
        "canonical_task_cid": "baguqeera-prior-revision",
        "implementation_commit": implementation_commit,
        "resolved": True,
    }
    monkeypatch.setattr(daemon, "_load_tasks", lambda: [task])
    monkeypatch.setattr(
        daemon,
        "_iter_merge_lifecycle_events",
        lambda: [finished, stale_reconciled],
    )
    monkeypatch.setattr(daemon, "_main_branch_name", lambda: "main")
    monkeypatch.setattr(daemon, "_git_ref_is_ancestor", lambda *_: False)

    assert daemon._failed_merge_candidates() == [finished]


def test_completed_current_revision_retains_completion_persistence_recovery(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = PortalImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        implement=False,
    )
    task = PortalTask(
        task_id="ACCEL-001",
        title="Landed task",
        status="completed",
        completion="auto",
        priority="P1",
        track="ops",
        outputs=["feature.py"],
    )
    task_cid = daemon._canonical_ref(task)
    implementation_commit = "c" * 40
    finished = {
        "type": "implementation_finished",
        "task_id": task.task_id,
        "canonical_task_cid": task_cid,
        "implementation_commit": implementation_commit,
        "validation_result": {"attempted": True, "passed": True},
        "merge_result": {
            "attempted": True,
            "merged": False,
            "canonical_task_cid": task_cid,
            "reason": "post_merge_integration_commit_unproven",
        },
        "cleanup_result": {"cleaned": True},
    }
    persistence_failure = {
        "type": "merge_reconciled",
        "task_id": task.task_id,
        "canonical_task_cid": task_cid,
        "implementation_commit": implementation_commit,
        "resolved": False,
        "reason": "completion_persistence_failed",
    }
    monkeypatch.setattr(daemon, "_load_tasks", lambda: [task])
    monkeypatch.setattr(
        daemon,
        "_iter_merge_lifecycle_events",
        lambda: [finished, persistence_failure],
    )
    monkeypatch.setattr(daemon, "_main_branch_name", lambda: "main")
    monkeypatch.setattr(daemon, "_git_ref_is_ancestor", lambda *_: True)

    candidates = daemon._failed_merge_candidates(
        skip_task_ids={task.task_id},
    )
    assert candidates == [
        {
            **finished,
            "completion_persistence_recovery": {
                "event_id": "",
                "timestamp": "",
                "reason": "completion_persistence_failed",
                "task_id": task.task_id,
                "implementation_commit": implementation_commit,
                "landed_commit": "",
                "landed_ref_source": "",
                "merge_commit": "",
                "cleanup_cleaned": False,
                "completion_task_cids": {},
                "integration_commit_proof": {
                    "passed": None,
                    "implementation_commit": "",
                    "integration_ref": "",
                    "integration_commit": "",
                    "target_branch": "",
                },
            },
        }
    ]


def test_old_display_id_merge_row_does_not_suppress_current_selection(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = PortalImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        implement=False,
    )
    task = PortalTask(
        task_id="ACCEL-001",
        title="Current task",
        status="todo",
        completion="auto",
        priority="P1",
        track="ops",
        outputs=["feature.py"],
    )
    current_cid = daemon._canonical_ref(task)
    checked_identities: list[str] = []

    class QueueProbe:
        @staticmethod
        def has_pending_for_task(identity):
            checked_identities.append(identity)
            return identity == task.task_id

    daemon.merge_queue = QueueProbe()
    selected = daemon._select_next_task(
        [task],
        {task.task_id: "ready"},
        {},
        {},
        {},
    )

    assert selected == task
    assert checked_identities == [current_cid]


def test_finish_receipt_requires_the_expected_task_revision(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    daemon = PortalImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
        implement=False,
    )
    daemon._record_event(
        "implementation_finished",
        {
            "task_id": "ACCEL-001",
            "canonical_task_cid": "baguqeera-prior-revision",
            "attempt": 1,
        },
    )
    assert daemon._task_attempt_has_implementation_finish(
        "ACCEL-001",
        1,
    )
    assert not daemon._task_attempt_has_implementation_finish(
        "ACCEL-001",
        1,
        canonical_task_cid="baguqeera-current-revision",
    )
