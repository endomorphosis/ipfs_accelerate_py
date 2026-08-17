import pytest
from ipfs_accelerate_py.agent_supervisor.planning.plan_doctor import diagnose, PlanDoctorError
def test_doctor_is_proposal_only():
    assert diagnose({"finding": "gap"})["accepted"] is False
    with pytest.raises(PlanDoctorError):
        diagnose({"mutate_accepted": True})
