import pytest
from ipfs_accelerate_py.agent_supervisor.semantic_state.work_loop import run_provisional_loop, WorkLoopError
def test_provisional_loop_does_not_publish():
    assert run_provisional_loop({})["canonical"] is False
    with pytest.raises(WorkLoopError):
        run_provisional_loop({"publish_canonical": True})
