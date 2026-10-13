"""Observe the native queued/reconciled event shape after bound completion."""
from copy import deepcopy
import json
import os
from types import MethodType, SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_output_observation as observation
from benchmarks.agent_supervisor.container_coding.test_terminal_output_observation import (
    authored, native_completed, observe, private_json, seal, sha,
)


@pytest.fixture
def queued(native_completed):
    f, task = native_completed
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalImplementationDaemon
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import _accepted_source_events

    # Use the actual producer's identity enrichment and event writer. The native
    # completion CAS fixture supplies the real database_complete receipt. The
    # queue event deliberately has no finalized lifecycle: that is the native
    # shape when merge_result.queued is true.
    path = f.directory / "portal-events.jsonl"
    path.unlink()
    producer = SimpleNamespace(events_path=path,
        _task_source_identity_record=lambda: None, _invalidate_event_cache=lambda: None,
        _task_identity_by_display_id={f.scope["task_alias"]: SimpleNamespace(
            canonical_task_key="authored:task", canonical_task_cid=f.scope["task_cid"],
            board_namespace="authored")},
        _implementation_finished_event_payload=PortalImplementationDaemon._implementation_finished_event_payload)
    producer._project_dependency_preflight_event_payload = MethodType(
        PortalImplementationDaemon._project_dependency_preflight_event_payload, producer)
    producer._record_event = MethodType(PortalImplementationDaemon._record_event, producer)
    finished = dict(f.event)
    finished.pop("lifecycle_finalize")
    finished.update(returncode=0, implementation_commit="c" * 40,
        board_completion={"complete": False, "pending_merge": True,
                          "reason": "merge_queued_awaiting_integration"},
        merge_result={"queued": True})
    previous = os.umask(0o022)
    try:
        producer._record_event("implementation_finished", finished)
        producer._record_event("merge_reconciled", dict(task_id=f.scope["task_alias"], attempt=1,
            implementation_commit="c" * 40, resolved=True,
            cleanup_result={"cleaned": True, "worktree_path": str(f.workspace)}))
    finally:
        os.umask(previous)
    f.events, f.events_sha256 = _accepted_source_events(path)
    f.events_sha256 = f.events_sha256.removeprefix("sha256:")
    f.events = [dict(row) for row in f.events]
    assert "lifecycle_finalize" not in f.events[0]
    assert f.events[1]["canonical_task_key"] == f.events[0]["canonical_task_key"]
    assert path.stat().st_uid == f.directory.stat().st_uid
    return f


def rewrite(f):
    (f.directory / "portal-events.jsonl").write_text("".join(
        json.dumps(row, sort_keys=True) + "\n" for row in f.events))


@pytest.mark.parametrize("arm", ["full", "no-index"])
@pytest.mark.parametrize("child", [False, True])
def test_native_queued_reconciliation_observes_current_declared_bytes(queued, arm, child):
    queued.report["arm"] = arm
    result = (observation.collect(native_state=queued.native, workspace_root=queued.root,
        repository=queued.app, scope=queued.scope, report=queued.report, timeout_seconds=5)
        if child else observe(queued))
    assert result["status"] == "observed"
    assert result["allocation_lifecycle"] == "owner_finished_and_reconciled_events"
    assert result["owner_events_sha256"] == queued.events_sha256
    assert result["reconciliation_event_sha256"] == sha(json.dumps(queued.events[1],
        sort_keys=True, separators=(",", ":"), allow_nan=False))
    assert result["outputs"][0]["candidate"]["sha256"] == sha(b"authored changed candidate\n")
    assert result["outputs"][0]["canonical"]["sha256"] == sha(b"authored baseline\n")
    assert result["patch_application_established"] is False
    assert result["writer_identity_established"] is False
    assert result["completion_authority"] is False
    assert "PRIVATE" not in json.dumps(result)


def test_cleaned_queued_allocation_retains_canonical_fact_and_unknown_candidate(queued):
    (queued.workspace / "eigen.py").unlink()
    queued.workspace.rmdir()
    result = observe(queued)
    assert result["status"] == "partial"
    assert result["outputs"][0]["candidate"]["status"] == "unavailable"
    assert result["outputs"][0]["candidate"]["exists"] is None
    assert result["outputs"][0]["canonical"]["sha256"] == sha(b"authored baseline\n")


@pytest.mark.parametrize("mutation", ["missing", "wrong_type", "unresolved", "duplicate", "predecessor",
    "task_id", "task_cid", "canonical_task_cid", "canonical_task_key", "attempt", "bool_attempt",
    "implementation_commit", "baseline_ref", "worktree_path", "cleanup_worktree", "cleanup_shape", "cleanup_not_cleaned",
    "finished_baseline", "finished_root", "finished_commit", "finished_not_queued", "finished_board",
    "finished_board_integer", "explicit_failed_finalization", "binding_changed"])
def test_queued_reconciliation_refuses_wrong_stale_or_ambiguous_evidence(queued, mutation, monkeypatch):
    event, row = queued.events
    if mutation == "missing": queued.events.pop()
    elif mutation == "wrong_type": row["type"] = "cleanup_finished"
    elif mutation == "unresolved": row["resolved"] = False
    elif mutation == "duplicate": queued.events.append(deepcopy(row))
    elif mutation == "predecessor": queued.events.reverse()
    elif mutation in {"task_id", "task_cid", "canonical_task_cid", "canonical_task_key"}:
        row[mutation] = "foreign"
    elif mutation == "attempt": row["attempt"] = 2
    elif mutation == "bool_attempt": row["attempt"] = True
    elif mutation == "implementation_commit": row["implementation_commit"] = "d" * 40
    elif mutation == "baseline_ref": row["baseline_ref"] = "b" * 40
    elif mutation == "worktree_path": row["worktree_path"] = str(queued.app)
    elif mutation == "cleanup_worktree": row["cleanup_result"]["worktree_path"] = str(queued.app)
    elif mutation == "cleanup_shape": row["cleanup_result"] = []
    elif mutation == "cleanup_not_cleaned": row["cleanup_result"]["cleaned"] = False
    elif mutation == "finished_baseline": event["baseline_ref"] = "b" * 40
    elif mutation == "finished_root": event["worktree_path"] = str(queued.app)
    elif mutation == "finished_commit": event["implementation_commit"] = row["implementation_commit"] = "main"
    elif mutation == "finished_not_queued": event["merge_result"]["queued"] = False
    elif mutation == "finished_board": event["board_completion"]["complete"] = True
    elif mutation == "finished_board_integer": event["board_completion"]["complete"] = 0
    elif mutation == "explicit_failed_finalization": event["lifecycle_finalize"] = {"finalized": False}
    else:
        queued.binding["task_cid"] = "foreign"
        private_json(queued.directory / "database-attempt-binding.json", seal(queued.binding))
    rewrite(queued)
    monkeypatch.setattr(observation, "_file", lambda *args: pytest.fail("refused event must not select output files"))
    result = observe(queued)
    assert result["status"] == "unavailable"
    assert result["outputs"] == []


def test_queued_reconciliation_does_not_upgrade_failed_settlement(queued, monkeypatch):
    from benchmarks.agent_supervisor.container_coding.test_terminal_output_observation import sha
    terminal = queued.report["native_progress"]["latest"]["task"]
    terminal.update(status="blocked")
    terminal["completion_receipt"].update(operation="database_task_claim_failure",
        failure_kind="terminal_portal_bridge_error", automatic_retry_admitted=False,
        settlement_id_sha256=sha("settlement:authored"))
    queued.report["task_state"]["status"] = "blocked"
    monkeypatch.setattr(observation, "_file", lambda *args: pytest.fail("failed settlement cannot use queue reconciliation"))
    assert observe(queued)["status"] == "unavailable"
