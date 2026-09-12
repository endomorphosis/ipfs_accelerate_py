"""Real projection/event/queue tests for legacy fresh-attempt recovery evidence."""
from __future__ import annotations

import copy
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueue
from ipfs_accelerate_py.agent_supervisor.runtime.event_log import append_jsonl_event
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalBridgeError, DatabasePortalExecutionBridge,
    verify_database_portal_attempt_projection_identity,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseTaskAttempt
from ipfs_accelerate_py.agent_supervisor.todo_daemon.legacy_verification_retry import (
    REASON, inspect_legacy_verification_retry, hold_legacy_retry_queue_absence,
)


@pytest.fixture
def history(tmp_path):
    attempt = DatabaseTaskAttempt(
        attempt_id="attempt:one", claim_id="claim:one", task_cid="task:one",
        task_alias="TEST-041", attempt_number=1, owner_session_id="session:one",
        fencing_token=7, fence_epoch=3, lease_id="lease:one", committed_phase="claimed",
        status="running", started_at_ms=1,
    )
    record = SimpleNamespace(task_cid=attempt.task_cid, task_alias=attempt.task_alias,
        goal_cid="goal:one", plan_cid="plan:one", revision=3, priority="P2", dependencies=(),
        outputs=({"path": "result.py"},), validations=({"argv": ["python3", "-m", "pytest"]},),
        acceptance=({"criterion": "validation passes"},), body={"objective": "test revision CAS",
        "completion": "auto", "write_scope": ["result.py"]})
    source = SimpleNamespace(get_task=lambda _: record, snapshot=lambda: SimpleNamespace(repository_tree_id="tree:one"))
    bridge = DatabasePortalExecutionBridge(task_source=source, attempt_root=tmp_path / "attempts",
                                            portal_factory=lambda *_: None)
    paths, _ = bridge._ensure_attempt_projection(attempt, record)
    identity = verify_database_portal_attempt_projection_identity(paths.task_projection)
    workspace = tmp_path / "worktrees/retained"
    workspace.mkdir(parents=True)
    (workspace / "result.py").write_text("# unverified work must survive\n")
    common = {"task_id": record.task_alias, "canonical_task_cid": identity["portal_canonical_task_cid"],
              "canonical_task_key": identity["portal_canonical_task_key"], "attempt": 1}
    candidate = {**common, "worktree_path": str(workspace), "branch": "implementation/test-041-attempt-1",
                 "baseline_ref": "a" * 40}
    retained = {**candidate, "cleanup_result": {"retained": True, "cleaned": False,
        "reason": "verification_deferred_checkout_lease_active"}, "commit_result": {"committed": False,
        "reason": "verification_deferred_checkout_lease_active"}, "implementation_commit": "",
        "retained_candidate_receipt": None}
    finished = {**copy.deepcopy(retained), "reason": REASON, "returncode": 1,
        "provider_dispatched": True, "attempt_consumed": False, "deferred": True,
        "merge_result": {"merged": False, "reason": "not_attempted"},
        "board_completion": {"complete": False, "pending_merge": False},
        "failed_preservation_result": {"retained": True, "preserved": False, "retained_candidate_receipt": None}}
    events = [("implementation_started", candidate),
        (REASON, {**common, "reason": REASON, "workspace_path": str(workspace),
                  "lock": {"acquired": False, "reason": "lock_exists"}}),
        ("protected_path_verification_deferred_worktree_retained", retained),
        ("implementation_finished", finished), ("daemon_pass", {})]
    terminal = {key: identity[key] for key in ("attempt_id", "claim_id", "lease_id", "owner_session_id",
        "attempt_number", "fencing_token", "fence_epoch")}
    terminal.update(operation="database_portal_terminal_failure", reason=REASON, retryable=False,
                    execution_phase="failed", control_expected_status="in_progress", control_expected_revision=3)
    row = {"task_alias": record.task_alias, "task_cid": record.task_cid, "goal_cid": record.goal_cid,
           "plan_cid": record.plan_cid, "status": "blocked", "revision": 4,
           "body_json": json.dumps({**record.body, "completion_receipt": terminal})}
    return SimpleNamespace(row=row, paths=paths, events=events, workspace=workspace,
                           attempt_root=tmp_path / "attempts", worktrees=tmp_path / "worktrees")


def inspect(history):
    for kind, body in history.events:
        append_jsonl_event(history.paths.events, kind, body)
    return inspect_existing(history)


def inspect_existing(history):
    return inspect_legacy_verification_retry(history.row, task_projection=history.paths.task_projection,
        allowed_attempt_root=history.attempt_root, retained_worktree_root=history.worktrees,
        expected_task_revision=4)


def test_finished_legacy_provider_proposes_only_fresh_validation_and_preserves_bytes(history):
    original_row = copy.deepcopy(history.row)
    result = inspect(history)
    event_bytes = history.paths.events.read_bytes()
    assert result["fresh_attempt_number"] == 2
    assert result["attempt_refunded"] is False
    assert result["require_fresh_portal_revalidation"] is True
    assert result["retained_candidate_admitted"] is False
    assert result["retry_authorized"] is False
    assert history.row == original_row
    assert history.workspace.joinpath("result.py").read_text() == "# unverified work must survive\n"
    assert inspect_existing(history) == result
    assert history.paths.events.read_bytes() == event_bytes


@pytest.mark.parametrize("field,value", [
    ("claim_id", "claim:other"), ("attempt_number", True), ("fencing_token", 8),
    ("reason", "pending_merge_timeout"), ("control_expected_revision", True),
    ("execution_phase", "running"), ("retryable", True),
])
def test_terminal_receipt_drift_never_admits_retry(history, field, value):
    body = json.loads(history.row["body_json"])
    body["completion_receipt"][field] = value
    history.row["body_json"] = json.dumps(body)
    with pytest.raises(DatabasePortalBridgeError):
        inspect(history)


@pytest.mark.parametrize("mutation", ["restarted", "queued", "callback", "fingerprint", "provider_unknown",
                                        "timeout", "committed", "merged", "foreign_task", "foreign_workspace"])
def test_existing_effects_or_uncertain_lifecycle_need_separate_recovery(history, mutation):
    finished = history.events[3][1]
    if mutation == "restarted": history.events.append(history.events[0])
    elif mutation == "queued": history.events.insert(1, ("merge_candidate_enqueued", {"request_id": "request:one"}))
    elif mutation == "callback": history.events.append(("merge_reconciled", {}))
    elif mutation == "fingerprint": finished["failed_preservation_result"]["retained_candidate_receipt"] = {}
    elif mutation == "provider_unknown": finished["provider_dispatched"] = None
    elif mutation == "timeout": finished["timeout_result"] = {"timed_out": True}
    elif mutation == "committed": finished["implementation_commit"] = "b" * 40
    elif mutation == "merged": finished["merge_result"] = {"merged": True}
    elif mutation == "foreign_task": finished["canonical_task_cid"] = "task:foreign"
    else: finished["worktree_path"] = str(history.workspace.parent / "foreign")
    with pytest.raises(DatabasePortalBridgeError):
        inspect(history)


@pytest.mark.parametrize("artifact", ["events", "projection", "binding"])
def test_changed_historical_artifact_is_not_reconstructed_as_original(history, artifact):
    inspect(history)
    if artifact == "events":
        history.paths.events.write_bytes(history.paths.events.read_bytes().replace(b"lock_exists", b"lock_changed"))
    elif artifact == "projection":
        history.paths.task_projection.write_text(history.paths.task_projection.read_text() + "\nChanged contract\n")
    else:
        binding = history.paths.task_projection.with_name("database-attempt-binding.json")
        data = json.loads(binding.read_text()); data["claim_id"] = "claim:foreign"
        binding.write_text(json.dumps(data))
    with pytest.raises(DatabasePortalBridgeError):
        inspect_existing(history)


def queue(tmp_path):
    return MergeQueue(tmp_path / "queue", target_repository_id="repository:test", target_branch="main")


def guard(q, evidence):
    return hold_legacy_retry_queue_absence(queue_dir=q.queue_dir, target_repository_id="repository:test",
                                         target_branch="main", evidence=evidence)


def test_queue_guard_is_observation_only_and_does_not_block_on_unrelated_candidate(history, tmp_path):
    evidence = inspect(history)
    q = queue(tmp_path)
    request = q.enqueue(branch_name="implementation/other", task_id="TEST-002", canonical_task_cid="task:other",
              commit_sha="c" * 40, metadata={"completion_task_cids": {"TEST-002": "task:other"}})
    original = q.database_path.read_bytes()
    with guard(q, evidence) as receipt:
        assert receipt["matching_queue_rows"] == 0 and receipt["guard_retained"] is True
        assert receipt["queue_settlement"]["settled"] is False
        assert receipt["retry_authorized"] is False
    assert q.database_path.read_bytes() == original
    assert q.get(request.request_id).status == "pending"


@pytest.mark.parametrize("matching", ["alias", "canonical", "secondary"])
def test_any_task_queue_binding_refuses_fresh_retry(history, tmp_path, matching):
    evidence = inspect(history)
    q = queue(tmp_path)
    alias, cid = ("TEST-041", "task:one") if matching == "alias" else ("TEST-002", "task:other")
    if matching == "canonical": cid = "task:one"
    bindings = {alias: cid}
    if matching == "secondary": bindings["TEST-041"] = "task:one"
    q.enqueue(branch_name="implementation/prior", task_id=alias, canonical_task_cid=cid,
              commit_sha="c" * 40, metadata={"completion_task_cids": bindings})
    with pytest.raises(DatabasePortalBridgeError, match="existing task merge history"):
        with guard(q, evidence): pytest.fail("must not yield retry evidence")


def test_forged_evidence_cannot_select_another_queue_task(history, tmp_path):
    evidence = inspect(history); evidence["task_cid"] = "task:foreign"
    with pytest.raises(DatabasePortalBridgeError, match="verified evidence"):
        with guard(queue(tmp_path), evidence): pytest.fail("must not yield")


def test_queue_guard_retains_exclusive_writer_flock_until_caller_finishes(history, tmp_path):
    evidence = inspect(history)
    q = queue(tmp_path)
    with (q.queue_dir / ".merge_queue.duckdb.lock").open("rb") as contender:
        with guard(q, evidence):
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(contender, fcntl.LOCK_UN)


@pytest.mark.parametrize("status", ["completed", "quarantined", "cancelled"])
def test_even_terminal_candidate_history_requires_reconciliation(history, tmp_path, status):
    evidence = inspect(history)
    q = queue(tmp_path)
    q.enqueue(branch_name="implementation/prior", task_id="TEST-041", canonical_task_cid="task:one",
              commit_sha="c" * 40, metadata={"completion_task_cids": {"TEST-041": "task:one"}})
    # Construct a disposable historical row; production code never mutates it.
    with q._connect() as connection:
        connection.execute("UPDATE merge_requests SET status = ?", (status,))
    with pytest.raises(DatabasePortalBridgeError, match="existing task merge history"):
        with guard(q, evidence): pytest.fail("terminal history must not be silently abandoned")
