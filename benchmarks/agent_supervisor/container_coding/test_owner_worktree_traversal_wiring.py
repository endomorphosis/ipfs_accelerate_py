"""Exact generated owner staging selects explicit traversal before provisioning.

Helpers are authored ordering seams here. Real permission changes and trusted
scope refusal are checked in the separate traversal helper module.
"""
import time

import pytest

from benchmarks.agent_supervisor.container_coding.test_owner_writable_worktree_wiring import namespace, setup_code
from ipfs_accelerate_py.agent_supervisor.runtime import owner_worktree_traversal as traversal
from ipfs_accelerate_py.agent_supervisor.runtime import owner_writable_worktree as provisioning


FIELDS = ("worktree_traversal_anchor", "worktree_traversal_directories", "worker_shared_group")


@pytest.mark.parametrize("seconds", [None, 30])
def test_explicit_generated_traversal_precedes_workspace_provisioning_with_same_original_deadline(
        tmp_path, monkeypatch, seconds):
    inherited = None if seconds is None else time.monotonic_ns() + seconds * 1_000_000_000
    state = namespace(tmp_path, monkeypatch, inherited=inherited)
    state["boundary"].update(worktree_traversal_anchor=str(tmp_path),
        worktree_traversal_directories=[str(state["cwd"].parent)], worker_shared_group=1000)
    calls = []
    observation = {"authored_traversal_observation": True}
    def publish(**kwargs):
        assert "worker_arguments" not in state and "owner_workspace_provisioning" not in state
        calls.append(("traversal", kwargs))
        return observation
    def provision(**kwargs):
        assert state["owner_worktree_traversal"] is observation
        assert "worker_arguments" not in state
        calls.append(("workspace", kwargs))
        return {"authored_workspace_observation": True}
    monkeypatch.setattr(traversal, "provision_owner_worktree_traversal", publish)
    monkeypatch.setattr(provisioning, "provision_owner_writable_worktree", provision)
    exec(setup_code(), state)
    expected = dict(boundary=state["root"] / "container-boundary.json", workspace=state["cwd"],
        absolute_deadline_ns=state["owner_workspace_deadline_ns"])
    assert calls == [("traversal", expected), ("workspace", expected)]
    assert state["owner_worktree_traversal"] is observation


def test_absent_traversal_fields_keep_existing_generated_workspace_route(tmp_path, monkeypatch):
    state = namespace(tmp_path, monkeypatch, inherited=None)
    calls = []
    monkeypatch.setattr(traversal, "provision_owner_worktree_traversal", lambda **_kwargs:
        pytest.fail("default wrapper selected traversal publication"))
    monkeypatch.setattr(provisioning, "provision_owner_writable_worktree", lambda **kwargs:
        calls.append(kwargs) or {"authored_workspace_observation": True})
    exec(setup_code(), state)
    assert len(calls) == 1
    assert state["owner_worktree_traversal"] is None


@pytest.mark.parametrize("field", FIELDS)
def test_any_partial_optin_reaches_fail_closed_helper_before_workspace_or_worker_setup(
        tmp_path, monkeypatch, field):
    state = namespace(tmp_path, monkeypatch, inherited=time.monotonic_ns() + 30_000_000_000)
    state["boundary"][field] = "authored incomplete traversal field"
    original = traversal.OwnerWorktreeTraversalError("authored incomplete traversal policy")
    calls = []
    def refuse(**kwargs):
        calls.append(kwargs)
        raise original
    monkeypatch.setattr(traversal, "provision_owner_worktree_traversal", refuse)
    monkeypatch.setattr(provisioning, "provision_owner_writable_worktree", lambda **_kwargs:
        pytest.fail("partial traversal policy reached workspace provisioning"))
    with pytest.raises(traversal.OwnerWorktreeTraversalError) as error:
        exec(setup_code(), state)
    assert error.value is original and len(calls) == 1
    assert "worker_arguments" not in state and "owner_workspace_provisioning" not in state
    assert "owner_worker_attempt" not in state and "worker_source_broker" not in state


def test_failed_workspace_provisioning_preserves_completed_traversal_observation_without_worker_setup(
        tmp_path, monkeypatch):
    state = namespace(tmp_path, monkeypatch, inherited=None)
    state["boundary"].update({field: "authored selected field" for field in FIELDS})
    observation = {"authored_traversal_observation": True}
    original = ValueError("authored workspace provisioning failure")
    monkeypatch.setattr(traversal, "provision_owner_worktree_traversal", lambda **_kwargs: observation)
    def refuse(**kwargs):
        assert state["owner_worktree_traversal"] is observation
        raise original
    monkeypatch.setattr(provisioning, "provision_owner_writable_worktree", refuse)
    with pytest.raises(ValueError) as error:
        exec(setup_code(), state)
    assert error.value is original
    assert state["owner_worktree_traversal"] is observation
    assert "worker_arguments" not in state and "owner_worker_attempt" not in state
