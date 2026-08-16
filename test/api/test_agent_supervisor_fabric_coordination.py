import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_fabric import issue_fence, SupervisorFabricError

def test_fence_requires_capability_and_rejects_stale_epoch() -> None:
    fence = issue_fence({"supervisor_id": "S1", "capability": "dispatch", "epoch": 2})
    assert fence["fenced"] is True
    with pytest.raises(SupervisorFabricError, match="stale"):
        issue_fence({"supervisor_id": "S1", "capability": "dispatch", "stale_epoch": True})
