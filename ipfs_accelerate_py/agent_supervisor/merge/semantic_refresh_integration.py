"""Merge-side closed-loop refresh adapter."""
from ipfs_accelerate_py.agent_supervisor.semantic_state.harness import run_closed_loop
def integrate_refresh(record):
    return run_closed_loop(record)
