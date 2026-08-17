"""Highest-level LGSWF machine output entrypoint."""
from ipfs_accelerate_py.agent_supervisor.runtime.decision_receipts import emit_decision
def main(record):
    return emit_decision(record)
