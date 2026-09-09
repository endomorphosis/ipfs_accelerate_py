"""Missing auxiliary keys cannot substitute for exact completion evidence."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.event_log import append_jsonl_event
from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalBridgeError,
    DatabasePortalExecutionBridge,
)


@pytest.mark.parametrize(
    "tamper",
    [
        "",
        "nonempty_key",
        "foreign_launch",
        "foreign_worktree",
        "foreign_task",
        "foreign_candidate",
        "foreign_request",
        "missing_result",
        "missing_validation",
        "dirty_candidate",
        "wrong_recovery",
        "duplicate_result",
    ],
)
def test_only_exact_legacy_merge_observations_are_ignored(tmp_path, tamper):
    paths = SimpleNamespace(events=tmp_path / "events.jsonl")
    identity = {
        "task_id": "TASK-1",
        "canonical_task_cid": "task:one",
        "canonical_task_key": "key:one",
        "branch": "implementation/retained",
        "baseline_ref": "a" * 40,
        "implementation_commit": "b" * 40,
        "attempt": 1,
    }
    worktree = str(tmp_path / "retained")
    recovery = "sha256:" + "d" * 64
    start = {
        **identity,
        "worktree_path": worktree,
        "recovery_key": recovery,
        "provider_dispatched": False,
    }
    enqueue = {
        **identity,
        "request_id": "request:one",
        "worktree_path": worktree,
        "completion_task_cids": {"TASK-1": "task:one"},
        "queued": True,
        "merged": False,
    }
    merge = {
        **identity,
        "request_id": "request:one",
        "queued": False,
        "merged": True,
        "merge_commit": "c" * 40,
    }
    queued = {
        **identity,
        "merge_result": {**merge, "queued": True, "merged": False},
        "provider_dispatched": False,
        "attempt_consumed": False,
    }
    queued.pop("canonical_task_key")
    reconciled = {
        **identity,
        "request_id": "request:one",
        "resolved": True,
        "completion_task_cids": {"TASK-1": "task:one"},
        "merge_result": merge,
    }
    reconciled.pop("canonical_task_key")
    final = {
        **identity,
        "worktree_path": worktree,
        "recovery_key": recovery,
        "returncode": 0,
        "provider_dispatched": False,
        "merge_result": deepcopy(merge),
        "validation_result": {
            "passed": True,
            "candidate_binding": {
                "verified": True,
                "expected_fingerprint": "sha256:" + "f" * 64,
                "current_fingerprint": "sha256:" + "f" * 64,
                "validated_workspace": {
                    "verified": True,
                    "head": "b" * 40,
                    "branch": identity["branch"],
                    "status_clean": True,
                },
            },
        },
    }
    if tamper == "nonempty_key":
        queued["canonical_task_key"] = "foreign"
    if tamper == "foreign_worktree":
        enqueue["worktree_path"] = str(tmp_path / "other")
    if tamper == "foreign_task":
        enqueue["canonical_task_cid"] = "task:foreign"
    if tamper == "foreign_candidate":
        enqueue["implementation_commit"] = "e" * 40
    if tamper == "foreign_request":
        enqueue["request_id"] = "request:other"
    if tamper == "dirty_candidate":
        final["validation_result"]["candidate_binding"]["validated_workspace"][
            "status_clean"
        ] = False
    if tamper == "wrong_recovery":
        start["recovery_key"] = "sha256:" + "e" * 64
    if tamper != "missing_validation":
        append_jsonl_event(
            paths.events, "worktree_reconciliation_validation_started", start
        )
    append_jsonl_event(paths.events, "merge_candidate_enqueued", enqueue)
    events = DatabasePortalExecutionBridge._verified_event_chain(paths)
    queued["merge_queue_synchronous_source"] = {
        "merge_candidate_enqueued_event_id": events[-1]["event_id"]
    }
    append_jsonl_event(paths.events, "worktree_reconciliation_candidate_queued", queued)
    append_jsonl_event(paths.events, "merge_reconciled", reconciled)
    events = DatabasePortalExecutionBridge._verified_event_chain(paths)
    final["merge_result"]["merge_reconciliation_receipt"] = {
        "recorded": True,
        "event_id": events[-1]["event_id"],
    }
    if tamper != "missing_result":
        append_jsonl_event(paths.events, "implementation_finished", final)
    if tamper == "duplicate_result":
        append_jsonl_event(paths.events, "implementation_finished", final)
    append_jsonl_event(paths.events, "task_completed", identity)
    if tamper == "foreign_launch":
        # A separately verified chain is injected at the read seam so this
        # test exercises correlation as well as the hash-chain tests elsewhere.
        events = DatabasePortalExecutionBridge._verified_event_chain(paths)
        events[0]["stream_id"] = "event-log:foreign-launch"

        class Reader(DatabasePortalExecutionBridge):
            @classmethod
            def _verified_event_chain(cls, paths):
                return events
    else:
        Reader = DatabasePortalExecutionBridge
    if tamper:
        with pytest.raises(DatabasePortalBridgeError):
            Reader._completion_event_evidence(
                paths,
                alias="TASK-1",
                task_cid="task:one",
                completion_task_key="key:one",
            )
    else:
        proof = Reader._completion_event_evidence(
            paths, alias="TASK-1", task_cid="task:one", completion_task_key="key:one"
        )
        assert proof["implementation_commit"] == "b" * 40
        assert proof["completion_source_event_type"] == "implementation_finished"
        assert (
            proof["completion_source_event_id"]
            == DatabasePortalExecutionBridge._verified_event_chain(paths)[-2][
                "event_id"
            ]
        )
