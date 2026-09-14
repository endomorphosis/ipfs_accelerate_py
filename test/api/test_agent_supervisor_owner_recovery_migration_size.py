"""Migration history may span wire frames without enlarging worker requests."""
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.merge import owner_recovery_runtime as recovery
from test.api.test_agent_supervisor_owner_merge_queue import owner
from test.api.test_agent_supervisor_owner_recovery_runtime import SCOPE


def test_migration_hash_preserves_existing_canonical_identity():
    payload = {"unicode": "\u00e9\U0001f986", "ordered": {"z": 2, "a": 1}, "values": [None, False, 1.25]}
    assert recovery._migration_payload_cid(payload) == recovery._cid(payload)


@pytest.mark.parametrize("payload", [{"x": float("nan")}, {"x": object()}])
def test_migration_hash_rejects_non_json_history(payload):
    with pytest.raises(recovery.OwnerRecoveryRuntimeError):
        recovery._migration_payload_cid(payload)


def test_migration_hash_enforces_aggregate_byte_limit(monkeypatch):
    monkeypatch.setattr(recovery, "MAX_PRESERVED_HISTORY_BYTES", 64)
    with pytest.raises(recovery.OwnerRecoveryRuntimeError, match="byte bound"):
        recovery._migration_payload_cid({"receipt": "x" * 64})


def test_multi_frame_local_receipt_migration_and_exact_replay(owner):
    gateway, connection, bind, *_ = owner
    bind()
    imports = []
    for index in range(40):
        body = {"schema": "preserved-receipt@1", "index": index, "evidence": "x" * 16384}
        imports.append({"receipt_key": f"receipt:{index}", "revision": 1,
                        "receipt_cid": recovery._cid(body), "receipt": body})
    assert len(json.dumps(imports).encode()) > recovery.MAX_JSON_BYTES
    args = dict(expected_identity=dict(gateway.identity), repository_id="repo:one",
                target_branch="main", migration_id="migration:multi-frame",
                scope_bindings=[SCOPE], receipt_imports=imports)
    assert gateway.provision_legacy_merge_recovery_schema(**args)["replayed"] is False
    assert gateway.provision_legacy_merge_recovery_schema(**args)["replayed"] is True
    gateway.bind_legacy_merge_recovery_service(expected_identity=dict(gateway.identity),
                                             repository_id="repo:one", target_branch="main")
    rows = connection.execute(
        "SELECT receipt_key,receipt_cid,receipt_json FROM legacy_merge_recovery_receipt_versions"
    ).fetchall()
    expected = {r["receipt_key"]: r for r in imports}
    assert len(rows) == len(imports)
    for row in rows:
        key, cid, raw = (row[index] for index in range(3))
        assert cid == expected[key]["receipt_cid"]
        assert json.loads(raw) == expected[key]["receipt"]
    # Mutation under the same migration identity must still be rejected.
    altered = [dict(r) for r in imports]
    altered[0]["receipt"] = {"changed": True}
    altered[0]["receipt_cid"] = recovery._cid(altered[0]["receipt"])
    with pytest.raises(recovery.OwnerRecoveryRuntimeError):
        gateway.provision_legacy_merge_recovery_schema(**{**args, "receipt_imports": altered})


def test_individual_receipt_still_requires_wire_bound():
    with pytest.raises(recovery.OwnerRecoveryRuntimeError, match="exceeds bound"):
        recovery._cid({"receipt": "x" * recovery.MAX_JSON_BYTES})
