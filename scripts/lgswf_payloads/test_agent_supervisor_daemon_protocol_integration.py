"""LGSWF-072 packet/checkpoint integration checks."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.todo_daemon.checkpoints import stale_stop
from ipfs_accelerate_py.agent_supervisor.todo_daemon.work_packet import REQUIRED, parse_work_packet


def test_admitted_packet_and_stale_stop_are_wired() -> None:
    payload = {
        name: f"sha256:{name}" if name.endswith("cid") or name.endswith("_cid") else name
        for name in REQUIRED
    }
    payload.update(
        {
            "scope": "owned",
            "effects": ("write",),
            "resource_vector": {"cpu_ms": 1},
            "repository_capabilities": ("embedded",),
            "mode": "embedded-one-writer",
        }
    )
    packet = parse_work_packet(payload)
    assert packet["mode"] == "embedded-one-writer"
    assert stale_stop("stale-fence")["effect_after"] is False
