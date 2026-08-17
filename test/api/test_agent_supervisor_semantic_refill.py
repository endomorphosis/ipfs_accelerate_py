import pytest
from ipfs_accelerate_py.agent_supervisor.task_sources.semantic_refill import propose_refill, RefillError
def test_bounded_refill_is_proposal_only():
    assert propose_refill({"bound": 1, "max_bound": 4})["accepted"] is False
    with pytest.raises(RefillError):
        propose_refill({"rewrite_accepted": True})
