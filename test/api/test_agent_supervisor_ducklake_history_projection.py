from ipfs_accelerate_py.agent_supervisor.integrations.ducklake_history_projection import project_history
def test_projection_is_non_authoritative():
    assert project_history({"receipt": True})["authoritative"] is False
