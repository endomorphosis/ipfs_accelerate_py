from ipfs_accelerate_py.agent_supervisor.runtime.resource_admission import admit_frontier

def test_frontier_respects_integer_capacity() -> None:
    result = admit_frontier(
        [
            {"task_id": "A", "demand": {"cpu": 2}},
            {"task_id": "B", "demand": {"cpu": 2}},
        ],
        capacity={"cpu": 3},
    )
    assert result["accepted"] == ("A",)
    assert result["rejected"][0]["task_id"] == "B"
    assert result["dispatched"] is False
