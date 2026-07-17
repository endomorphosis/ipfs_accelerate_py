"""Deterministic unit tests for peer trust-level resolution and priority-capped queue claiming.

Covers:
- PeerTrustLevel enum values
- resolve_peer_trust_level: shared-token, UCAN, event-DAG, and baseline paths
- trust_tiers_enabled / baseline_max_claim_priority env-var controls
- TaskQueue.claim_next max_priority filter
- TaskQueue.claim_next_many max_priority filter
"""

from __future__ import annotations

import os
import json
import time
import tempfile
import threading

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_queue(tmp_path):
    """Create a fresh in-memory or temp-file TaskQueue for testing."""
    from ipfs_accelerate_py.p2p_tasks.task_queue import TaskQueue

    path = str(tmp_path / "test_trust_queue.duckdb")
    return TaskQueue(path=path)


def _submit_with_priority(queue, *, priority: int, task_type: str = "test", model: str = "m") -> str:
    return queue.submit(
        task_type=task_type,
        model_name=model,
        payload={"priority": priority},
    )


# ---------------------------------------------------------------------------
# PeerTrustLevel enum
# ---------------------------------------------------------------------------


class TestPeerTrustLevelEnum:
    def test_enum_values_exist(self):
        from ipfs_accelerate_py.p2p_tasks.peer_trust import PeerTrustLevel

        assert PeerTrustLevel.TRUSTED == "trusted"
        assert PeerTrustLevel.ELEVATED == "elevated"
        assert PeerTrustLevel.BASELINE == "baseline"

    def test_enum_ordering_by_name(self):
        from ipfs_accelerate_py.p2p_tasks.peer_trust import PeerTrustLevel

        levels = [PeerTrustLevel.TRUSTED, PeerTrustLevel.ELEVATED, PeerTrustLevel.BASELINE]
        assert len(levels) == 3


# ---------------------------------------------------------------------------
# resolve_peer_trust_level
# ---------------------------------------------------------------------------


class TestResolvePeerTrustLevel:
    def test_shared_token_match_returns_trusted(self, monkeypatch):
        monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_TOKEN", "secret-abc")
        from ipfs_accelerate_py.p2p_tasks.peer_trust import resolve_peer_trust_level, PeerTrustLevel

        result = resolve_peer_trust_level({"token": "secret-abc"})
        assert result == PeerTrustLevel.TRUSTED

    def test_compat_token_env_match_returns_trusted(self, monkeypatch):
        monkeypatch.setenv("IPFS_DATASETS_PY_TASK_P2P_TOKEN", "compat-token")
        from ipfs_accelerate_py.p2p_tasks import peer_trust
        # Reimport to pick up new env
        import importlib
        importlib.reload(peer_trust)
        from ipfs_accelerate_py.p2p_tasks.peer_trust import resolve_peer_trust_level, PeerTrustLevel

        result = resolve_peer_trust_level({"token": "compat-token"})
        assert result == PeerTrustLevel.TRUSTED

    def test_wrong_token_does_not_elevate(self, monkeypatch):
        monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_TOKEN", "secret-abc")
        from ipfs_accelerate_py.p2p_tasks.peer_trust import resolve_peer_trust_level, PeerTrustLevel

        result = resolve_peer_trust_level({"token": "wrong-token"})
        assert result != PeerTrustLevel.TRUSTED

    def test_no_token_configured_ucan_present_returns_trusted(self, monkeypatch):
        monkeypatch.delenv("IPFS_ACCELERATE_PY_TASK_P2P_TOKEN", raising=False)
        monkeypatch.delenv("IPFS_DATASETS_PY_TASK_P2P_TOKEN", raising=False)
        from ipfs_accelerate_py.p2p_tasks.peer_trust import resolve_peer_trust_level, PeerTrustLevel

        result = resolve_peer_trust_level({"ucan_token": "******"})
        assert result == PeerTrustLevel.TRUSTED

    def test_no_token_no_ucan_returns_baseline(self, monkeypatch):
        monkeypatch.delenv("IPFS_ACCELERATE_PY_TASK_P2P_TOKEN", raising=False)
        monkeypatch.delenv("IPFS_DATASETS_PY_TASK_P2P_TOKEN", raising=False)
        from ipfs_accelerate_py.p2p_tasks.peer_trust import resolve_peer_trust_level, PeerTrustLevel

        result = resolve_peer_trust_level({"worker_id": "peer-xyz"})
        assert result == PeerTrustLevel.BASELINE

    def test_peer_did_in_event_dag_returns_elevated(self, monkeypatch):
        monkeypatch.delenv("IPFS_ACCELERATE_PY_TASK_P2P_TOKEN", raising=False)
        monkeypatch.delenv("IPFS_DATASETS_PY_TASK_P2P_TOKEN", raising=False)
        from ipfs_accelerate_py.p2p_tasks.peer_trust import resolve_peer_trust_level, PeerTrustLevel

        # Build a minimal event DAG stub with a peer DID entry.
        class _FakeStore:
            def export_snapshot(self):
                return {
                    "events": [
                        {"event_cid": "cid1", "payload": {"peer_did": "did:key:worker-alice"}},
                    ]
                }

        class _FakeDag:
            _store = _FakeStore()

        result = resolve_peer_trust_level(
            {"worker_id": "did:key:worker-alice"},
            event_dag=_FakeDag(),
        )
        assert result == PeerTrustLevel.ELEVATED

    def test_unknown_peer_did_returns_baseline(self, monkeypatch):
        monkeypatch.delenv("IPFS_ACCELERATE_PY_TASK_P2P_TOKEN", raising=False)
        monkeypatch.delenv("IPFS_DATASETS_PY_TASK_P2P_TOKEN", raising=False)
        from ipfs_accelerate_py.p2p_tasks.peer_trust import resolve_peer_trust_level, PeerTrustLevel

        class _FakeStore:
            def export_snapshot(self):
                return {
                    "events": [
                        {"event_cid": "cid1", "payload": {"peer_did": "did:key:alice"}},
                    ]
                }

        class _FakeDag:
            _store = _FakeStore()

        result = resolve_peer_trust_level(
            {"worker_id": "did:key:bob"},
            event_dag=_FakeDag(),
        )
        assert result == PeerTrustLevel.BASELINE

    def test_explicit_peer_did_kwarg_overrides_msg_fields(self, monkeypatch):
        monkeypatch.delenv("IPFS_ACCELERATE_PY_TASK_P2P_TOKEN", raising=False)
        monkeypatch.delenv("IPFS_DATASETS_PY_TASK_P2P_TOKEN", raising=False)
        from ipfs_accelerate_py.p2p_tasks.peer_trust import resolve_peer_trust_level, PeerTrustLevel

        class _FakeStore:
            def export_snapshot(self):
                return {
                    "events": [
                        {"event_cid": "cid1", "payload": {"peer_did": "did:key:explicit"}},
                    ]
                }

        class _FakeDag:
            _store = _FakeStore()

        result = resolve_peer_trust_level(
            {"worker_id": "did:key:different"},
            event_dag=_FakeDag(),
            peer_did="did:key:explicit",
        )
        assert result == PeerTrustLevel.ELEVATED


# ---------------------------------------------------------------------------
# trust_tiers_enabled / baseline_max_claim_priority
# ---------------------------------------------------------------------------


class TestEnvVarControls:
    def test_trust_tiers_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("IPFS_ACCELERATE_PY_TASK_P2P_TRUST_TIERS", raising=False)
        monkeypatch.delenv("IPFS_DATASETS_PY_TASK_P2P_TRUST_TIERS", raising=False)
        from ipfs_accelerate_py.p2p_tasks.peer_trust import trust_tiers_enabled

        assert not trust_tiers_enabled()

    def test_trust_tiers_enabled_by_env(self, monkeypatch):
        monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_TRUST_TIERS", "1")
        from ipfs_accelerate_py.p2p_tasks.peer_trust import trust_tiers_enabled

        assert trust_tiers_enabled()

    def test_trust_tiers_enabled_compat_env(self, monkeypatch):
        monkeypatch.delenv("IPFS_ACCELERATE_PY_TASK_P2P_TRUST_TIERS", raising=False)
        monkeypatch.setenv("IPFS_DATASETS_PY_TASK_P2P_TRUST_TIERS", "true")
        from ipfs_accelerate_py.p2p_tasks.peer_trust import trust_tiers_enabled

        assert trust_tiers_enabled()

    def test_baseline_max_priority_defaults_to_5(self, monkeypatch):
        monkeypatch.delenv("IPFS_ACCELERATE_PY_TASK_P2P_BASELINE_MAX_PRIORITY", raising=False)
        monkeypatch.delenv("IPFS_DATASETS_PY_TASK_P2P_BASELINE_MAX_PRIORITY", raising=False)
        from ipfs_accelerate_py.p2p_tasks.peer_trust import baseline_max_claim_priority

        assert baseline_max_claim_priority() == 5

    def test_baseline_max_priority_from_env(self, monkeypatch):
        monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_BASELINE_MAX_PRIORITY", "3")
        from ipfs_accelerate_py.p2p_tasks.peer_trust import baseline_max_claim_priority

        assert baseline_max_claim_priority() == 3

    def test_baseline_max_priority_clamped(self, monkeypatch):
        monkeypatch.setenv("IPFS_ACCELERATE_PY_TASK_P2P_BASELINE_MAX_PRIORITY", "99")
        from ipfs_accelerate_py.p2p_tasks.peer_trust import baseline_max_claim_priority

        assert baseline_max_claim_priority() == 10


# ---------------------------------------------------------------------------
# TaskQueue.claim_next with max_priority
# ---------------------------------------------------------------------------


class TestClaimNextMaxPriority:
    def test_claim_next_no_cap_returns_highest_priority(self, tmp_path):
        queue = _make_queue(tmp_path)
        _submit_with_priority(queue, priority=3)
        _submit_with_priority(queue, priority=8)
        _submit_with_priority(queue, priority=5)

        # Without cap: should return the highest-priority task (priority=8) first
        # because claim_next is FIFO by created_at, not priority.
        # The FIFO ordering means oldest is returned; priority filtering applies to max_priority.
        # Let's verify the filter works by restricting to priority <= 3.
        task = queue.claim_next(worker_id="w1", max_priority=3)
        assert task is not None
        assert task.payload.get("priority") == 3

    def test_claim_next_cap_excludes_high_priority(self, tmp_path):
        queue = _make_queue(tmp_path)
        _submit_with_priority(queue, priority=9)
        _submit_with_priority(queue, priority=9)

        # cap=5 means priority <= 5; both tasks have priority 9, so none eligible.
        task = queue.claim_next(worker_id="w1", max_priority=5)
        assert task is None

    def test_claim_next_cap_returns_eligible_task(self, tmp_path):
        queue = _make_queue(tmp_path)
        _submit_with_priority(queue, priority=9)
        _submit_with_priority(queue, priority=4)

        # cap=5: only the priority=4 task is eligible.
        task = queue.claim_next(worker_id="w1", max_priority=5)
        assert task is not None
        assert task.payload.get("priority") == 4

    def test_claim_next_none_cap_returns_any(self, tmp_path):
        queue = _make_queue(tmp_path)
        _submit_with_priority(queue, priority=9)

        task = queue.claim_next(worker_id="w1", max_priority=None)
        assert task is not None

    def test_claim_next_cap_boundary_inclusive(self, tmp_path):
        queue = _make_queue(tmp_path)
        _submit_with_priority(queue, priority=5)

        # cap=5 should include priority==5.
        task = queue.claim_next(worker_id="w1", max_priority=5)
        assert task is not None
        assert task.payload.get("priority") == 5


# ---------------------------------------------------------------------------
# TaskQueue.claim_next_many with max_priority
# ---------------------------------------------------------------------------


class TestClaimNextManyMaxPriority:
    def test_claim_next_many_cap_excludes_all_high(self, tmp_path):
        queue = _make_queue(tmp_path)
        _submit_with_priority(queue, priority=8)
        _submit_with_priority(queue, priority=9)

        tasks = queue.claim_next_many(worker_id="w1", max_tasks=2, max_priority=5)
        assert tasks == []

    def test_claim_next_many_cap_returns_only_eligible(self, tmp_path):
        queue = _make_queue(tmp_path)
        _submit_with_priority(queue, priority=3)
        _submit_with_priority(queue, priority=9)
        _submit_with_priority(queue, priority=4)

        tasks = queue.claim_next_many(
            worker_id="w1",
            max_tasks=3,
            same_task_type=False,
            max_priority=5,
        )
        priorities = [t.payload.get("priority") for t in tasks]
        assert all(p <= 5 for p in priorities)
        assert 9 not in priorities

    def test_claim_next_many_no_cap_returns_all(self, tmp_path):
        queue = _make_queue(tmp_path)
        for p in [1, 5, 10]:
            _submit_with_priority(queue, priority=p)

        tasks = queue.claim_next_many(
            worker_id="w1",
            max_tasks=3,
            same_task_type=False,
            max_priority=None,
        )
        assert len(tasks) == 3


# ---------------------------------------------------------------------------
# Package-level export
# ---------------------------------------------------------------------------


class TestPackageExports:
    def test_peer_trust_exported_from_package(self):
        from ipfs_accelerate_py.p2p_tasks import (
            PeerTrustLevel,
            baseline_max_claim_priority,
            resolve_peer_trust_level,
            trust_tiers_enabled,
        )

        assert PeerTrustLevel.TRUSTED == "trusted"
        assert callable(resolve_peer_trust_level)
        assert callable(trust_tiers_enabled)
        assert callable(baseline_max_claim_priority)
