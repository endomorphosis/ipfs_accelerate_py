import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.logic_governed_fabric import converge, ConvergenceError
def test_bounded_fixed_point():
    assert converge(["a", "a"], bound=3)["terminal"] == "a"
    with pytest.raises(ConvergenceError):
        converge(["a","b","c","d"], bound=2)
