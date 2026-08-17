from ipfs_accelerate_py.agent_supervisor.semantic_state.harness import run_closed_loop
def test_closed_loop_keeps_authorities_separate():
    result = run_closed_loop({"provisional": {}, "canonical": {"accepted_merge": True, "fresh_rescan": True}})
    assert result["provisional"]["canonical"] is False
    assert result["canonical"]["canonical"] is True
