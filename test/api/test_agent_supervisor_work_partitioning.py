import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.work_partitioning import partition_tasks, steal, PartitionError

def test_deterministic_partition_and_eligible_steal() -> None:
    parts = partition_tasks(["T2", "T1", "T3"], ["S1", "S2"])
    assert parts["S1"][0] == "T1"
    moved = steal(parts["S1"], parts["S2"], eligible=True)
    assert moved["stolen"] == "T3"
    with pytest.raises(PartitionError):
        steal(("T1",), (), eligible=False)
