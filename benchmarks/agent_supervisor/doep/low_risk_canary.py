"""Hermetic add low-risk canary cohort. Live execution is optional; completion is never admitted."""
def describe() -> dict:
    return {"schema": "ipfs_accelerate_py/agent-supervisor/doep-harness@1", "harness": "low_risk_canary", "live": False, "completion_authority": False, "reason": "board_dispatcher_broken; hermetic description only"}
