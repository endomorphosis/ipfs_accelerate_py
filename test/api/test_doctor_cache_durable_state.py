"""Real DuckDB restart, migration and race controls for Doctor denial state."""
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.doctor_cache_state import DoctorCacheStateError
from ipfs_accelerate_py.agent_supervisor.proof.doctor_proof_cache import DoctorProofCacheGate
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import FormalVerificationCache
from test.api.test_agent_supervisor_doctor_proof_cache import _key, _receipt, _identity


def _restart(tmp_path, key, operation="lookup"):
    inputs = tmp_path / "restart-input.json"
    inputs.write_text(json.dumps(key.to_dict()))
    script = """
import json,sys
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.proof.doctor_proof_cache import DoctorProofCacheGate,DoctorProofCacheKey
gate=DoctorProofCacheGate(Path(sys.argv[1]))
key=DoctorProofCacheKey.from_dict(json.loads(Path(sys.argv[2]).read_text()))
if sys.argv[3]=='invalidate':
    rows=gate.invalidate_semantic_root(root_field='forest',root_cid=key.forest.cid)
    print(json.dumps({'count':len(rows)}))
else:
    value=gate.revalidate_for_commit(key)
    print(json.dumps({'status':value.disposition.value,'hit':value.hit}))
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path / "cache"), str(inputs), operation],
                            env=dict(os.environ), text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("operation,expected", [("invalidate", "tombstoned"), ("quarantine", "quarantined")])
def test_new_process_cannot_revive_positive_after_durable_refusal(tmp_path, operation, expected):
    gate = DoctorProofCacheGate(tmp_path / "cache")
    key = _key()
    assert gate.put(key, _receipt()).stored
    if operation == "invalidate":
        gate.invalidate_semantic_root(root_field="forest", root_cid=key.forest.cid)
    else:
        gate.quarantine(key)
    assert _restart(tmp_path, key) == {"status": expected, "hit": False}
    assert not DoctorProofCacheGate(tmp_path / "cache").put(key, _receipt()).stored


def test_new_process_rebuilds_roots_then_invalidates_and_existing_gate_observes(tmp_path):
    gate = DoctorProofCacheGate(tmp_path / "cache")
    key = _key()
    assert gate.put(key, _receipt()).stored
    assert _restart(tmp_path, key, "invalidate") == {"count": 1}
    assert gate.revalidate_for_render(key).disposition.value == "tombstoned"


def test_unseen_old_root_revocation_blocks_late_put_but_not_new_root(tmp_path):
    gate = DoctorProofCacheGate(tmp_path / "cache")
    key = _key()
    assert gate.invalidate_semantic_root(root_field="forest", root_cid=key.forest.cid) == ()
    reopened = DoctorProofCacheGate(tmp_path / "cache")
    assert not reopened.put(key, _receipt()).stored
    assert reopened.is_tombstoned(key)
    successor = _key(forest=_identity("forest", "forest-1", 2))
    assert reopened.put(successor, _receipt()).stored
    assert reopened.lookup(successor).hit


def test_pre_control_domain_migrates_roots_without_authority_upgrade(tmp_path):
    cache = FormalVerificationCache(tmp_path / "cache")
    key = _key()
    assert cache.put(key.to_formal_key(), _receipt()).stored
    gate = DoctorProofCacheGate(cache=cache)
    tombstones = gate.invalidate_semantic_root(root_field="forest", root_cid=key.forest.cid)
    assert [item.key_id for item in tombstones] == [key.key_id]
    assert _restart(tmp_path, key) == {"status": "tombstoned", "hit": False}


def test_restart_detects_equivocation_and_retains_original_positive(tmp_path):
    gate = DoctorProofCacheGate(tmp_path / "cache")
    key = _key(); receipt = _receipt()
    assert gate.put(key, receipt).stored
    restarted = DoctorProofCacheGate(tmp_path / "cache")
    assert not restarted.put(key, replace(receipt, attempt_id="different-attempt")).stored
    assert _restart(tmp_path, key) == {"status": "quarantined", "hit": False}
    assert gate.authoritative_cache.lookup(key.to_formal_key()).receipt.receipt_id == receipt.receipt_id


@pytest.mark.parametrize("during", ["lookup", "put"])
def test_invalidation_between_formal_operation_and_return_refuses(tmp_path, monkeypatch, during):
    gate = DoctorProofCacheGate(tmp_path / "cache")
    key = _key()
    assert gate.put(key, _receipt()).stored
    other = DoctorProofCacheGate(tmp_path / "cache")
    original = getattr(gate.authoritative_cache, during)
    def interleave(*args, **kwargs):
        result = original(*args, **kwargs)
        other.invalidate_semantic_root(root_field="forest", root_cid=key.forest.cid)
        return result
    monkeypatch.setattr(gate.authoritative_cache, during, interleave)
    result = gate.lookup(key) if during == "lookup" else gate.put(key, _receipt())
    assert not (result.hit if during == "lookup" else result.stored)
    assert gate.is_tombstoned(key)


@pytest.mark.parametrize("damage", ["key", "roots", "schema"])
def test_corrupt_durable_state_fails_closed(tmp_path, damage):
    gate = DoctorProofCacheGate(tmp_path / "cache")
    key = _key()
    assert gate.put(key, _receipt()).stored
    cx = gate.authoritative_cache._connect()
    try:
        if damage == "key":
            cx.execute("UPDATE doctor_cache_control_keys SET key_json='{}'")
        elif damage == "roots":
            cx.execute("DELETE FROM doctor_cache_control_roots WHERE root_field='forest'")
        else:
            cx.execute("UPDATE doctor_cache_control_meta SET schema_version=2")
    finally:
        cx.close()
    with pytest.raises(DoctorCacheStateError):
        gate.lookup(key)
    if damage == "schema":
        with pytest.raises(DoctorCacheStateError):
            DoctorProofCacheGate(tmp_path / "cache")


def test_concurrent_distinct_receipts_cannot_both_escape_equivocation(tmp_path):
    first = DoctorProofCacheGate(tmp_path / "cache")
    second = DoctorProofCacheGate(tmp_path / "cache")
    start = threading.Barrier(2)
    key = _key(); receipt = _receipt()
    def store(gate, candidate):
        start.wait(timeout=10)
        return gate.put(key, candidate).stored
    with ThreadPoolExecutor(max_workers=2) as workers:
        outcomes = [workers.submit(store, first, receipt),
                    workers.submit(store, second, replace(receipt, attempt_id="concurrent-other"))]
        assert sum(item.result(timeout=30) for item in outcomes) <= 1
    assert first.lookup(key).disposition.value == "quarantined"
    assert _restart(tmp_path, key) == {"status": "quarantined", "hit": False}


def test_quack_snapshot_cannot_substitute_for_current_owner_transaction(tmp_path, monkeypatch):
    cache = FormalVerificationCache(tmp_path / "cache")
    connect = cache._connect
    def snapshot():
        connection = connect()
        connection._transport_mode = "quack"
        return connection
    monkeypatch.setattr(cache, "_connect", snapshot)
    with pytest.raises(DoctorCacheStateError, match="typed owner command"):
        DoctorProofCacheGate(cache=cache)


def test_tombstone_query_does_not_persist_private_key_preimages(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.analysis.content_identity_bridge import identify_strict_artifact
    from ipfs_accelerate_py.agent_supervisor.proof.doctor_proof_cache import DoctorIdentityBinding
    gate = DoctorProofCacheGate(tmp_path / "cache")
    binding = DoctorIdentityBinding.from_identity(identify_strict_artifact({"password": "placeholder"}),
                                                 logical_id="private-fields")
    with pytest.raises(DoctorCacheStateError, match="private Doctor key"):
        gate.is_tombstoned(_key(goal=binding))
    cx = gate.authoritative_cache._connect()
    try:
        assert cx.execute("SELECT count(*) FROM doctor_cache_control_keys").fetchone()[0] == 0
    finally:
        cx.close()
