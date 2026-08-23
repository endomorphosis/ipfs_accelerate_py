from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.residual_intelligence.drift import (
    DriftDisposition,
    DriftEvent,
    ExpertDriftMonitor,
    ExpertState,
)


def test_stale_experts_are_unroutable_and_false_accept_revokes() -> None:
    monitor = ExpertDriftMonitor()
    state, disposition = monitor.apply(DriftEvent(expert_id="exp:1", signal="false_accept", current=True))
    assert state is ExpertState.REVOKED
    assert disposition is DriftDisposition.REVOKE
    assert monitor.routable(state) is False
    shadow, wider = monitor.apply(
        DriftEvent(expert_id="exp:1", signal="calibration_group_change", current=True)
    )
    assert shadow is ExpertState.SHADOW
    assert wider is DriftDisposition.WIDER_ABSTENTION
