import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon.work_packet import parse_work_packet, WorkPacketError, REQUIRED

def _packet(**overrides):
    payload = {name: f"sha256:{name}" if name.endswith("cid") or name.endswith("_cid") else name for name in REQUIRED}
    payload.update({"scope": "owned", "effects": ("write",), "resource_vector": {"cpu_ms": 1}, "repository_capabilities": ("embedded",), "mode": "embedded-one-writer"})
    payload.update(overrides)
    return payload

def test_roundtrip_and_required_fields() -> None:
    first = parse_work_packet(_packet())
    second = parse_work_packet(dict(first))
    assert first["packet_cid"] == second["packet_cid"]
    assert set(REQUIRED) <= set(first)

def test_forbidden_and_sentinel_rejected() -> None:
    with pytest.raises(WorkPacketError, match="forbidden"):
        parse_work_packet(_packet(duckdb_path="/tmp/db"))
    with pytest.raises(WorkPacketError, match="sentinel"):
        parse_work_packet(_packet(plan_cid="REBIND_REQUIRED_BY_LGSWF-005"))
    with pytest.raises(WorkPacketError, match="1.5"):
        parse_work_packet(_packet(mode="duckdb-1.5"))
