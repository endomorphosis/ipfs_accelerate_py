import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.stage_backpressure import admit_stage, preempt

def test_independent_stage_limits() -> None:
    assert admit_stage("implement", 1, 2)["admitted"] is True
    assert admit_stage("validate", 2, 2)["admitted"] is False

def test_unsafe_preemption_rejected() -> None:
    with pytest.raises(Exception, match="unsafe"):
        preempt({"has_external_effect": True, "compensatable": False})
    assert preempt({"has_external_effect": False})["safe"] is True
