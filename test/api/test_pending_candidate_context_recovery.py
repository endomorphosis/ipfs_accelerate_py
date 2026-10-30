"""Real Native preservation; synthetic stopped ownership, no model dispatch.

The normal SDK fixture models the historical event projection. The isolated
compatibility driver instead loads actual unmodified6af and sets the explicit
fixture flag, proving the missing cache enrichment without any event patch.
"""
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as native
from ipfs_accelerate_py.agent_supervisor.todo_daemon import pending_task_claim_recovery as legacy
from ipfs_accelerate_py.agent_supervisor.todo_daemon import pending_candidate_context_recovery as bridge
from test.api.test_pending_task_claim_recovery import pending

pytestmark = pytest.mark.parametrize("pending", ["candidate"], indirect=True)


@pytest.fixture
def historical_helper(tmp_path, monkeypatch):
    """This historical bridge always exercises its actual frozen70 dependency."""
    if hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest() == bridge.LEGACY_SHA256:
        return legacy
    source = Path(legacy.__file__).resolve().parents[3]
    root = tmp_path / "frozen-sdk70"
    subprocess.run(["git", "clone", "--shared", "--no-checkout", str(source), str(root)], check=True, capture_output=True)
    subprocess.run(["git", "sparse-checkout", "init", "--no-cone"], cwd=root, check=True)
    relative = "ipfs_accelerate_py/agent_supervisor/todo_daemon/pending_task_claim_recovery.py"
    subprocess.run(["git", "sparse-checkout", "set", "--no-cone", "/" + relative], cwd=root, check=True)
    subprocess.run(["git", "checkout", "--detach", bridge.LEGACY_COMMIT], cwd=root, check=True, capture_output=True)
    name = legacy.__name__; parent = sys.modules[legacy.__package__]
    spec = importlib.util.spec_from_file_location(name, root / relative)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(parent, "pending_task_claim_recovery", module)
    monkeypatch.setattr(bridge, "legacy", module)
    monkeypatch.setattr(sys.modules[__name__], "legacy", module)
    return module


@pytest.fixture
def checkpoint(pending, monkeypatch, historical_helper):
    daemon, task, workspace, claim = pending
    daemon._task_identity_by_display_id.clear()
    request = legacy.review_recovery(daemon, task_id=task.task_id)
    actual = daemon._record_event
    if os.environ.get("TEST_ACTUAL_LEGACY_NATIVE") == "1":
        path = Path(native.__file__).resolve(); root = path.parents[3]
        assert path.read_bytes() == subprocess.check_output(["git", "show",
            "6af2ffd429eb0902f66064a55ac76f18c54498d2:" + path.relative_to(root).as_posix()], cwd=root)
        assert "result.update(canonical_task_cid=identity.canonical_task_cid" not in path.read_text()
    else:
        # Historical-only fixture projection, not production guard weakening.
        def historical_projection(kind, value, **kwargs):
            if kind == "implementation_pending_candidate_preserved":
                value = {k: v for k, v in value.items() if k not in bridge._IDENTITY}
            return actual(kind, value, **kwargs)
        monkeypatch.setattr(daemon, "_record_event", historical_projection)
    result = legacy.reconcile_recovery(daemon, request=request)
    monkeypatch.setattr(daemon, "_record_event", actual)
    assert result["blocked"] and result["reason"] == "native_preservation_event_required", result
    assert not workspace.exists() and claim.exists()
    rows = legacy._event_rows(legacy._read(daemon.events_path))
    original = next(row for row in rows if row["type"] == "implementation_pending_candidate_preserved")
    assert not any(k in original for k in bridge._IDENTITY)
    assert daemon.worktree_lifecycle.load_workspace(workspace) is not None
    return daemon, task, workspace, claim, request, original


def test_current_context_projection_is_honest_idempotent_and_old70_resumes_unchanged(checkpoint):
    daemon, task, workspace, claim, old_request, original = checkpoint
    state = asdict(native.PortalTaskState.load(daemon.state_path))
    path = legacy._candidate_receipt_path(daemon, task.task_id)
    snapshots = {p: p.read_bytes() for p in (path, claim, daemon.todo_path, daemon.strategy_path,
        daemon.worktree_lifecycle.workspace_path_for(workspace))}
    history = daemon.events_path.read_bytes()
    request = bridge.review(daemon, task_id=task.task_id)
    assert daemon.events_path.read_bytes() == history
    result = bridge.reconcile(daemon, request=request)
    assert result["reverified"] and not result["blocked"], result
    assert all(result[k] is False for k in legacy._FALSE)
    assert all(p.read_bytes() == raw for p, raw in snapshots.items())
    assert asdict(native.PortalTaskState.load(daemon.state_path)) == state
    assert daemon.events_path.read_bytes().startswith(history)
    rows = legacy._event_rows(legacy._read(daemon.events_path))
    projection = rows[-1]
    assert projection["event_id"] != original["event_id"]
    assert projection["canonical_context_reverification"]["original_event_id"] == original["event_id"]
    assert projection["canonical_context_reverification"]["new_preservation_performed"] is False
    assert all(projection[k] == request[k] for k in bridge._IDENTITY)
    projected_history = daemon.events_path.read_bytes()
    assert bridge.reconcile(daemon, request=request) == result
    assert daemon.events_path.read_bytes() == projected_history
    assert legacy.review_recovery(daemon, task_id=task.task_id) == old_request
    resumed = legacy.reconcile_recovery(daemon, request=old_request)
    assert resumed["reconciled"] and not claim.exists(), resumed
    assert asdict(native.PortalTaskState.load(daemon.state_path)) == state
    assert legacy._sha(legacy._read(Path(legacy.__file__))) == bridge.LEGACY_SHA256
    assert not any(row["type"] in legacy._OUTCOMES for row in daemon._iter_events())


@pytest.mark.parametrize("damage", ["claim", "state", "ref", "strategy", "protected", "lifecycle", "later_proposal", "review"])
def test_changed_custody_or_proposal_refuses_context_projection_without_mutation(checkpoint, damage):
    daemon, task, workspace, claim, _, original = checkpoint
    request = bridge.review(daemon, task_id=task.task_id)
    if damage == "claim":
        value = json.loads(claim.read_text()); value["lease_id"] = "replacement"; claim.write_text(json.dumps(value))
    elif damage == "state":
        state = native.PortalTaskState.load(daemon.state_path); state.implementation_attempts[task.task_id] = 2; state.save(daemon.state_path)
    elif damage == "ref":
        subprocess.run(["git", "branch", "-f", original["rescue_branch"], request["preserved"]["commit"] + "^"], cwd=daemon.repo_root, check=True)
    elif damage == "strategy": daemon.strategy_path.write_text('{"blocked_tasks":["held"]}')
    elif damage == "protected": daemon._implementation_protected_incident_path().write_text("{}")
    elif damage == "lifecycle":
        path = daemon.worktree_lifecycle.workspace_path_for(workspace); value = json.loads(path.read_text()); value["fence"] += 1; path.write_text(json.dumps(value))
    elif damage == "later_proposal":
        daemon._record_event("implementation_proposal_validated", {"task_id": task.task_id,
            "canonical_task_cid": request["canonical_task_cid"], "canonical_task_key": request["canonical_task_key"],
            "attempt": 1, "accepted": True, "attempted": True, "reason_codes": [], "proposal_id": "f" * 64})
    else: request["bridge_source"]["sha256"] = "0" * 64
    history = daemon.events_path.read_bytes(); raw = claim.read_bytes()
    state = asdict(native.PortalTaskState.load(daemon.state_path))
    result = bridge.reconcile(daemon, request=request)
    assert result["blocked"] and not result["reverified"], result
    assert daemon.events_path.read_bytes() == history and claim.read_bytes() == raw
    assert asdict(native.PortalTaskState.load(daemon.state_path)) == state


def test_unrelated_event_cannot_impersonate_the_context_projection(checkpoint):
    daemon, task, _, claim, _, _ = checkpoint
    request = bridge.review(daemon, task_id=task.task_id)
    daemon._record_event("unrelated_diagnostic", {"canonical_context_reverification": {"review": request}})
    history = daemon.events_path.read_bytes()
    result = bridge.reconcile(daemon, request=request)
    assert result["blocked"] and result["reason"] == "canonical_context_projection_unproved"
    assert claim.exists() and daemon.events_path.read_bytes() == history


def test_unrelated_nonmapping_context_metadata_is_not_a_projection(checkpoint):
    daemon, task, _, claim, _, _ = checkpoint
    daemon._record_event("unrelated_diagnostic", {"canonical_context_reverification": None})
    request = bridge.review(daemon, task_id=task.task_id)
    result = bridge.reconcile(daemon, request=request)
    assert result["reverified"] and not result["blocked"] and claim.exists(), result
