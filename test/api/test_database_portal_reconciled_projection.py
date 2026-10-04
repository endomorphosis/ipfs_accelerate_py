"""Constructed filesystem tests; no provider, native store or real oracle."""
import json
import os
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon import database_portal_bridge as mod
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon, parse_task_file,
)


def fixture(tmp_path):
    attempt = SimpleNamespace(attempt_id="attempt:fixture", claim_id="claim:fixture",
        task_cid="task:fixture", task_alias="FIX-001", fencing_token=7,
        fence_epoch=3, lease_id="lease:fixture")
    record = SimpleNamespace(task_cid=attempt.task_cid, task_alias=attempt.task_alias,
        goal_cid="goal:fixture", plan_cid="plan:fixture", revision=4, priority="P0",
        dependencies=(), outputs=({"path": "out.json"},), validations=(),
        acceptance=({"criterion": "Retain real test output"},),
        body={"objective": "Constructed completion", "completion": "auto"})
    source = SimpleNamespace(get_task=lambda cid: record if cid == attempt.task_cid else None)
    bridge = mod.DatabasePortalExecutionBridge(task_source=source,
        attempt_root=tmp_path / "outside_git", portal_factory=lambda *_: None, task_header_prefix="FIX-")
    paths, binding = bridge._ensure_attempt_projection(attempt, record)
    daemon = object.__new__(PortalImplementationDaemon)
    daemon._runtime_wake_coordinator = None
    daemon.todo_path, daemon.state_path, daemon.events_path = paths.task_projection, paths.state, paths.events
    text = paths.task_projection.read_text().replace("- Status: ready", "- Status: completed")
    paths.task_projection.write_text(text)
    identity = daemon._identity_for_task(parse_task_file(paths.task_projection, task_header_prefix="FIX-")[0])
    cids = {attempt.task_alias: identity.canonical_task_cid}
    update = {"updated": True, "path": str(paths.task_projection),
        "updated_task_ids": [attempt.task_alias],
        "completion_receipts": [{"task_id": attempt.task_alias,
            "canonical_task_cid": identity.canonical_task_cid, "status": "succeeded"}],
        "commit_result": {"committed": False, "reason": "not_in_git_repo", "path": str(paths.task_projection)}}
    callback = lambda u, c: bridge._reconciled_projection_snapshot(
        daemon=daemon, paths=paths, binding=binding, todo_update=u, completion_task_cids=c)
    return bridge, attempt, paths, binding, daemon, cids, update, callback


def test_exact_outside_git_completion_has_fsynced_evidence(tmp_path):
    bridge, attempt, paths, binding, daemon, cids, update, callback = fixture(tmp_path)
    assert daemon._reconciled_completion_persisted(update, cids)["passed"] is False
    daemon._database_portal_reconciled_snapshot = callback
    result = daemon._reconciled_completion_persisted(update, cids)
    assert result["passed"] is True and result["durable_update"] is True
    proof = result["database_projection_snapshot"]
    assert proof["projection_authority"] is False
    assert proof["binding_id"] == binding["binding_id"]
    assert proof["completion_task_cids"] == cids
    # Durability evidence alone never bypasses the outer event requirement.
    with pytest.raises(mod.DatabasePortalBridgeError, match="matching durable"):
        bridge._acceptance_receipt(attempt=attempt, paths=paths, binding=binding, summaries=[])


@pytest.mark.parametrize("mutation", ["status", "criterion", "binding", "cid", "result_path", "daemon_path", "symlink"])
def test_changed_identity_or_scope_is_rejected(tmp_path, mutation):
    bridge, attempt, paths, binding, daemon, cids, update, callback = fixture(tmp_path)
    daemon._database_portal_reconciled_snapshot = callback
    if mutation in {"status", "criterion"}:
        text = paths.task_projection.read_text()
        text = (text.replace("- Status: completed", "- Status: ready") if mutation == "status"
                else text.replace("Retain real test output", "Different criterion"))
        paths.task_projection.write_text(text)
    elif mutation == "binding":
        data = json.loads(paths.binding.read_text()); data["attempt_id"] = "attempt:other"
        paths.binding.write_text(json.dumps(data))
    elif mutation == "cid": cids = {"FIX-001": "task:wrong"}
    elif mutation == "result_path": update = {**update, "path": str(tmp_path / "other")}
    elif mutation == "daemon_path": daemon.events_path = tmp_path / "other-events"
    else:
        target = tmp_path / "copy"; target.write_bytes(paths.task_projection.read_bytes())
        paths.task_projection.unlink(); paths.task_projection.symlink_to(target)
    assert daemon._reconciled_completion_persisted(update, cids)["passed"] is False


@pytest.mark.parametrize("mutation", ["fsync_failure", "same_inode_rewrite", "replacement"])
def test_durability_or_concurrent_change_is_rejected(tmp_path, monkeypatch, mutation):
    bridge, attempt, paths, binding, daemon, cids, update, callback = fixture(tmp_path)
    daemon._database_portal_reconciled_snapshot = callback
    original = mod.os.fsync; fired = False
    def fsync(fd):
        nonlocal fired
        if not fired and os.fstat(fd).st_ino == paths.task_projection.stat().st_ino:
            fired = True
            if mutation == "fsync_failure": raise OSError("constructed fsync failure")
            if mutation == "same_inode_rewrite":
                paths.task_projection.write_text(paths.task_projection.read_text().replace("completed", "ready"))
            else:
                replacement = paths.root / "replacement"
                replacement.write_bytes(paths.task_projection.read_bytes()); os.replace(replacement, paths.task_projection)
        return original(fd)
    monkeypatch.setattr(mod.os, "fsync", fsync)
    assert daemon._reconciled_completion_persisted(update, cids)["passed"] is False
    assert fired


def test_completed_status_still_needs_matching_member_receipt(tmp_path):
    bridge, attempt, paths, binding, daemon, cids, update, callback = fixture(tmp_path)
    daemon._database_portal_reconciled_snapshot = callback
    update["completion_receipts"][0]["canonical_task_cid"] = "task:wrong"
    assert daemon._reconciled_completion_persisted(update, cids)["passed"] is False


def test_same_completed_projection_restart_requires_fresh_proof(tmp_path):
    bridge, attempt, paths, binding, daemon, cids, update, callback = fixture(tmp_path)
    daemon._database_portal_reconciled_snapshot = callback
    update.update(updated=False, reason="already_completed",
                  updated_task_ids=[], already_completed_task_ids=["FIX-001"])
    update.pop("commit_result")
    assert daemon._reconciled_completion_persisted(update, cids)["passed"] is True
    paths.task_projection.write_text(paths.task_projection.read_text().replace("completed", "ready"))
    assert daemon._reconciled_completion_persisted(update, cids)["passed"] is False


def test_bridge_installs_callback_and_keeps_completion_event_boundary(tmp_path):
    bridge, attempt, paths, binding, daemon, cids, update, callback = fixture(tmp_path)
    calls = []
    def run_once():
        result = daemon._reconciled_completion_persisted(update, cids); calls.append(result)
        assert result["passed"] is True
        # Constructed event producer; no real model/merge execution is claimed.
        with paths.events.open("w") as handle:
            json.dump({"type": "task_completed", "task_id": "FIX-001"}, handle)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        return {"completed_count": 1}
    daemon.run_once = run_once
    bridge.portal_factory = lambda *_: daemon
    receipt = bridge.run_provider(attempt)
    assert len(calls) == 1 and receipt["accepted"] is True
    assert receipt["completion_authority"] == "DatabaseImplementationDaemon"
