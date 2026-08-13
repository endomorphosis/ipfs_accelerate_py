from __future__ import annotations

import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoImplementationDaemon,
    TodoTaskState,
)


def _daemon(tmp_path: Path) -> TodoImplementationDaemon:
    repo = tmp_path / "repo"
    repo.mkdir()
    return TodoImplementationDaemon(
        todo_path=repo / "todo.md",
        state_path=repo / "state" / "task_state.json",
        strategy_path=repo / "state" / "strategy.json",
        events_path=repo / "state" / "events.jsonl",
        repo_root=repo,
    )


def test_quarantined_queued_merge_is_ignored_after_resolved_reconcile(tmp_path):
    daemon = _daemon(tmp_path)
    finished = {
        "IPS-001": {
            "type": "implementation_finished",
            "task_id": "IPS-001",
            "implementation_commit": "8391986ca97964361063d8af79c4ad4e30849008",
            "merge_result": {
                "queued": True,
                "request_id": "req-ips-001",
                "reason": "merge_queued",
            },
        }
    }

    class _Queue:
        def get(self, request_id):
            assert request_id == "req-ips-001"
            return type(
                "Request",
                (),
                {
                    "status": "quarantined",
                    "failure_reason": "merge_branch_candidate_mismatch",
                },
            )()

    daemon.merge_queue = _Queue()
    daemon._latest_implementation_finished_by_task = lambda: finished  # type: ignore[method-assign]
    daemon._implementation_commit_was_reconciled = (  # type: ignore[method-assign]
        lambda task_id, commit: False
    )

    assert daemon._quarantined_queued_merge_task_ids() == {"IPS-001"}

    daemon._implementation_commit_was_reconciled = (  # type: ignore[method-assign]
        lambda task_id, commit: True
    )
    assert daemon._quarantined_queued_merge_task_ids() == set()
    daemon._git_ref_is_ancestor = lambda ancestor, descendant: False  # type: ignore[method-assign]
    daemon._main_branch_name = lambda: "main"  # type: ignore[method-assign]
    assert daemon._pending_queued_merge_task_ids() == set()


def test_implementation_daemon_releases_stale_quarantined_inventory_merge(tmp_path):
    daemon = _daemon(tmp_path)
    finished = {
        "IPS-003": {
            "type": "implementation_finished",
            "task_id": "IPS-003",
            "attempt": 3,
            "branch": "implementation/ips-003-stale",
            "implementation_commit": "1d13a5582e167a0a77844ec66b32941a94aecbbf",
            "merge_result": {
                "queued": True,
                "request_id": "req-ips-003",
                "branch": "implementation/ips-003-stale",
                "reason": "merge_queued",
            },
        }
    }

    class _Queue:
        def get(self, request_id):
            assert request_id == "req-ips-003"
            return type(
                "Request",
                (),
                {
                    "status": "quarantined",
                    "failure_reason": "merge_branch_candidate_mismatch",
                },
            )()

    daemon.merge_queue = _Queue()
    daemon._queued_merge_candidates = lambda: [  # type: ignore[method-assign]
        {
            "task_id": "IPS-003",
            "attempt": 3,
            "branch": "implementation/ips-003-stale",
            "implementation_commit": "1d13a5582e167a0a77844ec66b32941a94aecbbf",
            "request_id": "req-ips-003",
            "merge_result": finished["IPS-003"]["merge_result"],
        }
    ]
    daemon._implementation_commit_was_reconciled = (  # type: ignore[method-assign]
        lambda task_id, commit: False
    )
    daemon._git_ref_is_ancestor = lambda ancestor, descendant: False  # type: ignore[method-assign]
    daemon._main_branch_name = lambda: "main"  # type: ignore[method-assign]
    daemon._inventory_task_passes_published_gate = lambda task_id: False  # type: ignore[method-assign]

    result = daemon._release_stale_quarantined_merges()

    assert len(result) == 1
    assert result[0]["task_id"] == "IPS-003"
    assert result[0]["resolved"] is True
    assert result[0]["reason"] == "stale_quarantined_merge"
    assert result[0]["implementation_commit"] == (
        "1d13a5582e167a0a77844ec66b32941a94aecbbf"
    )
    events = [
        json.loads(line)
        for line in daemon.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["type"] == "merge_reconciled"
    assert events[-1]["reason"] == "stale_quarantined_merge"
    assert events[-1]["resolved"] is True


def test_release_skips_ancestor_inventory_merge_waiting_on_status_commit(tmp_path):
    daemon = _daemon(tmp_path)
    finished = {
        "IPS-002": {
            "type": "implementation_finished",
            "task_id": "IPS-002",
            "attempt": 4,
            "branch": "implementation/ips-002-landed",
            "implementation_commit": "531ce91c0323deadbeefdeadbeefdeadbeefde",
            "merge_result": {
                "queued": True,
                "request_id": "req-ips-002",
                "reason": "merge_queued",
            },
        }
    }

    class _Queue:
        def get(self, request_id):
            return type(
                "Request",
                (),
                {
                    "status": "quarantined",
                    "failure_reason": "inventory_published_gate_not_satisfied",
                },
            )()

    daemon.merge_queue = _Queue()
    daemon._queued_merge_candidates = lambda: [  # type: ignore[method-assign]
        {
            "task_id": "IPS-002",
            "attempt": 4,
            "branch": "implementation/ips-002-landed",
            "implementation_commit": "531ce91c0323deadbeefdeadbeefdeadbeefde",
            "request_id": "req-ips-002",
            "merge_result": finished["IPS-002"]["merge_result"],
        }
    ]
    daemon._implementation_commit_was_reconciled = (  # type: ignore[method-assign]
        lambda task_id, commit: False
    )
    daemon._git_ref_is_ancestor = lambda ancestor, descendant: True  # type: ignore[method-assign]
    daemon._main_branch_name = lambda: "main"  # type: ignore[method-assign]
    daemon._inventory_task_passes_published_gate = lambda task_id: False  # type: ignore[method-assign]

    assert daemon._release_stale_quarantined_merges() == []


def test_has_unresolved_merge_failure_ignores_reconciled_last_implementation(
    tmp_path,
):
    daemon = _daemon(tmp_path)
    task = PortalTask(
        task_id="IPS-001",
        title="Inventory accelerate",
        status="todo",
        completion="manual",
        priority="P0",
        track="inventory",
    )
    previous = TodoTaskState(
        last_implementation_task_id="IPS-001",
        last_implementation_commit="8391986ca97964361063d8af79c4ad4e30849008",
        last_merge_returncode=1,
    )
    daemon._git_ref_is_ancestor = lambda ancestor, descendant: False  # type: ignore[method-assign]
    daemon._main_branch_name = lambda: "main"  # type: ignore[method-assign]
    daemon._implementation_commit_was_reconciled = (  # type: ignore[method-assign]
        lambda task_id, commit: False
    )
    assert daemon._has_unresolved_merge_failure(task, previous) is True
    daemon._implementation_commit_was_reconciled = (  # type: ignore[method-assign]
        lambda task_id, commit: True
    )
    assert daemon._has_unresolved_merge_failure(task, previous) is False


def test_select_next_task_ignores_pending_merge_after_reconcile(tmp_path):
    daemon = _daemon(tmp_path)
    task = PortalTask(
        task_id="IPS-002",
        title="Inventory datasets",
        status="ready",
        completion="manual",
        priority="P0",
        track="inventory",
    )

    class _Queue:
        def has_pending_for_task(self, task_id, commit_sha=None):
            return True

    daemon.merge_queue = _Queue()
    daemon._queued_merge_candidates = lambda: [  # type: ignore[method-assign]
        {
            "task_id": "IPS-002",
            "implementation_commit": "531ce91c0323deadbeefdeadbeefdeadbeefde",
        }
    ]
    daemon._implementation_commit_was_reconciled = (  # type: ignore[method-assign]
        lambda task_id, commit: True
    )
    daemon._canonical_ref = lambda item: "cid-ips-002"  # type: ignore[method-assign]
    selected = daemon._select_next_task(
        [task],
        {"IPS-002": "ready"},
        {"focus_tracks": [], "deprioritized_tasks": [], "blocked_tasks": []},
        {},
        {},
    )
    assert selected is task


def test_pending_merge_ignores_reconciled_commit_after_later_failed_finish(
    tmp_path,
):
    daemon = _daemon(tmp_path)
    task = PortalTask(
        task_id="IPS-002",
        title="Inventory datasets",
        status="ready",
        completion="manual",
        priority="P0",
        track="inventory",
    )

    class _Queue:
        def has_pending_for_task(self, task_id, commit_sha=None):
            return True

    daemon.merge_queue = _Queue()
    daemon._queued_merge_candidates = lambda: [  # type: ignore[method-assign]
        {
            "task_id": "IPS-002",
            "implementation_commit": "531ce91c0323deadbeefdeadbeefdeadbeefde",
            "request_id": "req-ips-002",
        }
    ]
    daemon._latest_implementation_finished_by_task = lambda: {  # type: ignore[method-assign]
        "IPS-002": {
            "task_id": "IPS-002",
            "implementation_commit": "2357f2d06ae7deadbeefdeadbeefdeadbeef",
            "merge_result": {"reason": "not_attempted"},
            "returncode": 78,
        }
    }
    daemon._implementation_commit_was_reconciled = (  # type: ignore[method-assign]
        lambda task_id, commit: commit.startswith("531ce91c")
    )
    daemon._canonical_ref = lambda item: "cid-ips-002"  # type: ignore[method-assign]
    assert daemon._task_has_blocking_pending_merge(task) is False


def test_release_cancels_pending_request_for_reconciled_commit(tmp_path):
    daemon = _daemon(tmp_path)
    cancelled: list[str] = []

    class _Queue:
        def get(self, request_id):
            return type("Request", (), {"status": "pending", "failure_reason": ""})()

        def cancel(self, request_id, reason="cancelled"):
            cancelled.append(f"{request_id}:{reason}")
            return type("Request", (), {"status": "cancelled"})()

    daemon.merge_queue = _Queue()
    daemon._queued_merge_candidates = lambda: [  # type: ignore[method-assign]
        {
            "task_id": "IPS-002",
            "attempt": 4,
            "branch": "implementation/ips-002",
            "implementation_commit": "531ce91c0323deadbeefdeadbeefdeadbeefde",
            "request_id": "req-ips-002",
            "merge_result": {"queued": True, "request_id": "req-ips-002"},
        }
    ]
    daemon._implementation_commit_was_reconciled = (  # type: ignore[method-assign]
        lambda task_id, commit: True
    )
    daemon._main_branch_name = lambda: "main"  # type: ignore[method-assign]

    result = daemon._release_stale_quarantined_merges()

    assert cancelled == ["req-ips-002:stale_quarantined_merge"]
    assert result[0]["cancelled_reconciled_pending"] is True
    assert result[0]["task_id"] == "IPS-002"


def test_proposal_gate_failure_is_rearmable_evidence(tmp_path):
    daemon = _daemon(tmp_path)
    task = PortalTask(
        task_id="IPS-001",
        title="Inventory accelerate",
        status="ready",
        completion="manual",
        priority="P0",
        track="inventory",
    )
    daemon._latest_implementation_finished_by_task = lambda: {  # type: ignore[method-assign]
        "IPS-001": {
            "task_id": "IPS-001",
            "implementation_commit": "6c5c3f1d7240deadbeef",
            "returncode": 78,
            "validation_result": {
                "attempted": False,
                "passed": False,
                "reason": "proposal_gate_failed",
            },
            "merge_result": {"merged": False, "reason": "not_attempted"},
        }
    }
    assert daemon._task_has_validation_command_failure_evidence(task) is True


def test_rearm_clears_exhausted_inventory_attempts_after_proposal_gate(
    tmp_path,
):
    daemon = _daemon(tmp_path)
    daemon.max_task_attempts = 5
    task = PortalTask(
        task_id="IPS-001",
        title="Inventory accelerate",
        status="ready",
        completion="manual",
        priority="P0",
        track="inventory",
    )
    identity = daemon._identity_for_task(task)
    state = TodoTaskState(
        implementation_attempts={"IPS-001": 5},
        implementation_attempts_by_cid={identity.canonical_task_cid: 5},
    )
    daemon._latest_implementation_failure_is_rearmable = (  # type: ignore[method-assign]
        lambda item: True
    )
    daemon.task_queue.reset_retry_state = lambda cid: False  # type: ignore[method-assign]
    rearmed = daemon._rearm_attempt_limited_obsoleted_validation_tasks(
        state,
        [task],
        {"IPS-001": "ready"},
        recovery_revision="rev-unstuck",
    )
    assert len(rearmed) == 1
    assert rearmed[0]["task_id"] == "IPS-001"
    assert "IPS-001" not in state.implementation_attempts
    assert identity.canonical_task_cid not in state.implementation_attempts_by_cid
