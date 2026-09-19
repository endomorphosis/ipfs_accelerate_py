"""SAWM-033 causal-event federation."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.runtime.program_world_causal_federation import (
    ProgramWorldCausalFederationAdapter,
    compute_affected_supervisor_wakes,
    publish_program_world_event,
)


def test_publish_is_idempotent_and_wakes_only_subscribers() -> None:
    adapter = ProgramWorldCausalFederationAdapter()
    adapter.subscribe("invalidation", ("sawm-lane-0", "doep-lane-1"))
    first = publish_program_world_event(
        {"event_id": "e1", "topic": "invalidation", "payload_cid": "bafy-1"},
        adapter=adapter,
    )
    second = publish_program_world_event(
        {"event_id": "e1", "topic": "invalidation", "payload_cid": "bafy-1"},
        adapter=adapter,
    )
    assert first.event_id == second.event_id
    plan = compute_affected_supervisor_wakes("e1", adapter=adapter)
    assert plan.full_scan is False
    assert plan.idempotent is True
    assert plan.supervisor_ids == ("sawm-lane-0", "doep-lane-1")
