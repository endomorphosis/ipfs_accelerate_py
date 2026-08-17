"""Closed-loop semantic refresh harness."""
from types import MappingProxyType
from ipfs_accelerate_py.agent_supervisor.semantic_state.work_loop import run_provisional_loop
from ipfs_accelerate_py.agent_supervisor.semantic_state.post_merge_refresh import refresh_canonical
def run_closed_loop(record):
    prov = run_provisional_loop(record.get("provisional") or {})
    can = refresh_canonical(record.get("canonical") or {"accepted_merge": True, "fresh_rescan": True})
    return MappingProxyType({"provisional": prov, "canonical": can})
