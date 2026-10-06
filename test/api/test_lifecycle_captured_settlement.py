"""A lifecycle path or reused lease/fence cannot replace captured ownership."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    OwnershipError, WorkspaceLifecycleState,
)
from test.api.test_captured_lifecycle_boundaries import custody


def _bytes(store, record):
    return (store.workspace_path_for(record.workspace_path).read_bytes(),
            store.task_index_path_for(task_id=record.task_id,
                canonical_task_cid=record.canonical_task_cid, attempt=record.attempt).read_bytes())


def _replace_owner(store, captured):
    terminal = store.mark_terminal(captured.workspace_path, lease_id=captured.lease_id,
                                   expected_fence=captured.fence)
    assert store.compare_and_delete(terminal.workspace_path, lease_id=terminal.lease_id,
                                    expected_fence=terminal.fence)
    peer = store.begin_preparing(task_id="PEER", canonical_task_cid="cid:peer", attempt=2,
        lane_id="peer-lane", workspace_path=captured.workspace_path,
        branch="implementation/peer", merge_target=captured.merge_target, lease_id=captured.lease_id)
    while peer.fence < captured.fence:
        peer = store.transition(peer.workspace_path, peer.state, lease_id=peer.lease_id,
                                expected_fence=peer.fence)
    assert peer.fence == captured.fence and peer.lease_id == captured.lease_id
    return peer


@pytest.mark.parametrize("terminal_first", [False, True])
def test_missing_captured_record_never_reports_finalization(custody, terminal_first):
    daemon, store, record = custody
    if terminal_first:
        record = store.mark_terminal(record.workspace_path, lease_id=record.lease_id,
                                     expected_fence=record.fence)
        daemon._active_worktree_lifecycle = record
    assert store.compare_and_delete(record.workspace_path, lease_id=record.lease_id,
                                    expected_fence=record.fence)
    assert daemon._finalize_exact_worktree_lifecycle(record, reason="fixture")["finalized"] is False
    assert daemon._active_worktree_lifecycle == record


def test_exact_present_terminal_capture_requires_actual_compare_delete(custody):
    daemon, store, record = custody
    terminal = store.mark_terminal(record.workspace_path, lease_id=record.lease_id,
                                   expected_fence=record.fence)
    daemon._active_worktree_lifecycle = terminal
    result = daemon._finalize_exact_worktree_lifecycle(terminal, reason="fixture")
    assert result["finalized"] is True
    assert store.load_workspace(record.workspace_path) is None
    assert daemon._active_worktree_lifecycle is None
    assert daemon._finalize_exact_worktree_lifecycle(terminal, reason="fixture")["finalized"] is False


@pytest.mark.parametrize("field", ["lane_id", "branch", "owner", "created_at"])
@pytest.mark.parametrize("boundary", ["transition", "delete", "active", "rebind", "same_path_rebind"])
def test_full_record_cas_rejects_same_record_id_lease_and_fence(custody, field, boundary):
    daemon, store, record = custody
    if boundary == "delete":
        record = store.mark_terminal(record.workspace_path, lease_id=record.lease_id,
                                     expected_fence=record.fence)
    value = {"lane_id": "peer-lane", "branch": "implementation/peer",
             "owner": replace(record.owner, start_time_ticks=record.owner.start_time_ticks + 1),
             "created_at": record.created_at + 1}[field]
    peer = replace(record, **{field: value})
    assert peer.record_id == record.record_id and peer.lease_id == record.lease_id and peer.fence == record.fence
    store.workspace_path_for(peer.workspace_path).write_text(json.dumps(peer.to_dict()))
    before = _bytes(store, peer)
    if boundary == "transition":
        with pytest.raises(OwnershipError, match="captured record"):
            store.mark_terminal(record.workspace_path, lease_id=record.lease_id,
                expected_fence=record.fence, expected_record=record)
    elif boundary == "delete":
        assert store.compare_and_delete(record.workspace_path, lease_id=record.lease_id,
            expected_fence=record.fence, expected_record=record) is False
    elif boundary == "active":
        with pytest.raises(OwnershipError, match="captured record"):
            store.mark_active(record.workspace_path, lease_id=record.lease_id,
                expected_fence=record.fence, expected_record=record)
    else:
        destination = (record.workspace_path if boundary == "same_path_rebind"
                       else str(Path(record.workspace_path).with_name("pooled")))
        with pytest.raises(OwnershipError, match="captured record"):
            store.rebind_workspace(record.workspace_path, destination,
                lease_id=record.lease_id, expected_fence=record.fence, expected_record=record)
        if boundary == "rebind":
            assert store.load_workspace(destination) is None
    assert _bytes(store, peer) == before


def test_replacement_between_terminal_cas_and_delete_is_retained(custody, monkeypatch):
    daemon, store, record = custody
    original = store.compare_and_delete
    seen = {}
    def replace_before_delete(workspace, **kwargs):
        terminal = store.load_workspace(workspace)
        assert original(workspace, lease_id=terminal.lease_id, expected_fence=terminal.fence)
        peer = store.begin_preparing(task_id="PEER", canonical_task_cid="cid:peer", attempt=2,
            lane_id="peer", workspace_path=workspace, branch="implementation/peer",
            merge_target="main", lease_id=terminal.lease_id)
        while peer.fence < terminal.fence:
            peer = store.transition(workspace, peer.state, lease_id=peer.lease_id, expected_fence=peer.fence)
        seen.update(record=peer, before=_bytes(store, peer))
        return original(workspace, **kwargs)
    monkeypatch.setattr(store, "compare_and_delete", replace_before_delete)
    assert daemon._finalize_exact_worktree_lifecycle(record, reason="fixture")["finalized"] is False
    assert store.load_workspace(record.workspace_path) == seen["record"]
    assert _bytes(store, seen["record"]) == seen["before"]
    assert daemon._active_worktree_lifecycle == record


def test_clearing_active_capture_never_clears_reused_tokens_from_another_owner(custody, monkeypatch):
    daemon, store, record = custody
    peer = replace(record, lane_id="peer", owner=replace(record.owner, start_time_ticks=record.owner.start_time_ticks + 1))
    original = store.compare_and_delete
    def replace_active(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is True
        daemon._active_worktree_lifecycle = peer
        return result
    monkeypatch.setattr(store, "compare_and_delete", replace_active)
    assert daemon._finalize_exact_worktree_lifecycle(record, reason="fixture")["finalized"] is True
    assert daemon._active_worktree_lifecycle == peer


@pytest.mark.parametrize("pooled", [False, True])
def test_native_failure_cannot_settle_after_peer_replaces_captured_workspace(tmp_path, monkeypatch, pooled):
    from test.api.test_native_provider_failure_settlement import _native_chain
    daemon, bridge, portals = _native_chain(tmp_path, monkeypatch, pooled=pooled)
    factory = bridge.portal_factory
    seen = {}
    def guarded_factory(paths, alias):
        native = factory(paths, alias)
        original = native._finalize_exact_worktree_lifecycle
        def replace_before_finalize(record, **kwargs):
            if not seen:
                peer = _replace_owner(native.worktree_lifecycle, record)
                seen.update(record=peer, before=_bytes(native.worktree_lifecycle, peer))
            return original(record, **kwargs)
        monkeypatch.setattr(native, "_finalize_exact_worktree_lifecycle", replace_before_finalize)
        return native
    bridge.portal_factory = guarded_factory
    try:
        result = daemon.run_once()
        assert result["implementation_result"]["reason"] == "provider_callback_outcome_unknown"
        attempt = daemon.get_attempt(result["attempt_id"])
        assert attempt.status == "running"
        assert daemon.coordinator.get_task_claim(attempt.claim_id).state.value == "accepted"
        assert len(portals) == 1
        store = portals[0].worktree_lifecycle
        assert store.load_workspace(seen["record"].workspace_path) == seen["record"]
        assert _bytes(store, seen["record"]) == seen["before"]
        receipt = daemon.provider_invocation_recorded(attempt.attempt_id,
            idempotency_key=f"provider:{attempt.attempt_id}")
        assert receipt["callback_state"] == "started_outcome_unknown"
    finally:
        daemon.close()


def test_exact_capture_can_move_to_pool_activate_and_finalize(custody):
    daemon, store, record = custody
    destination = Path(record.workspace_path).with_name("pooled")
    rebound = daemon._sync_worktree_lifecycle_workspace(record, destination)
    assert rebound.workspace_path == str(destination)
    assert store.load_workspace(record.workspace_path) is None
    assert daemon._active_worktree_lifecycle == rebound
    active = store.mark_active(rebound.workspace_path, lease_id=rebound.lease_id,
                              expected_fence=rebound.fence, expected_record=rebound)
    daemon._active_worktree_lifecycle = active
    assert active.state is WorkspaceLifecycleState.ACTIVE
    assert daemon._finalize_worktree_lifecycle(destination)["finalized"] is True
    assert store.load_workspace(destination) is None
    assert daemon._active_worktree_lifecycle is None


def test_daemon_pool_rebind_does_not_adopt_reused_owner_tokens(custody, monkeypatch):
    daemon, store, record = custody
    destination = Path(record.workspace_path).with_name("pooled")
    original = store.rebind_workspace
    seen = {}
    def replace_before_rebind(*args, **kwargs):
        peer = _replace_owner(store, record)
        seen.update(peer=peer, before=_bytes(store, peer))
        return original(*args, **kwargs)
    monkeypatch.setattr(store, "rebind_workspace", replace_before_rebind)
    with pytest.raises(OwnershipError, match="captured record"):
        daemon._sync_worktree_lifecycle_workspace(record, destination)
    assert daemon._active_worktree_lifecycle == record
    assert store.load_workspace(record.workspace_path) == seen["peer"]
    assert _bytes(store, seen["peer"]) == seen["before"]
    assert store.load_workspace(destination) is None


@pytest.mark.parametrize("pooled", [False, True])
def test_native_provider_is_not_started_after_replacement_before_activation(tmp_path, monkeypatch, pooled):
    from test.api.test_native_provider_failure_settlement import _native_chain
    daemon, bridge, portals = _native_chain(tmp_path, monkeypatch, pooled=pooled)
    script = tmp_path / "router_timeout.py"
    marker = tmp_path / "provider-started"
    script.write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('started')\n" + script.read_text())
    factory = bridge.portal_factory
    seen = {}
    def guarded_factory(paths, alias):
        native = factory(paths, alias)
        original = native.worktree_lifecycle.mark_active
        def replace_before_active(workspace, **kwargs):
            record = native._active_worktree_lifecycle
            peer = _replace_owner(native.worktree_lifecycle, record)
            seen.update(capture=record, peer=peer, before=_bytes(native.worktree_lifecycle, peer))
            return original(workspace, **kwargs)
        monkeypatch.setattr(native.worktree_lifecycle, "mark_active", replace_before_active)
        return native
    bridge.portal_factory = guarded_factory
    try:
        daemon.run_once()
        assert len(portals) == 1 and seen
        assert marker.exists() is False
        native = portals[0]
        assert native._active_worktree_lifecycle == seen["capture"]
        assert native.worktree_lifecycle.load_workspace(seen["peer"].workspace_path) == seen["peer"]
        assert _bytes(native.worktree_lifecycle, seen["peer"]) == seen["before"]
    finally:
        daemon.close()
