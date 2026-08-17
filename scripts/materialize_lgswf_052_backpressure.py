#!/usr/bin/env python3
from pathlib import Path
from lgswf_emit_copy_writer import copy_pairs, emit
PAIRS = (
    ("scripts/lgswf_payloads/stage_backpressure.py",
     "ipfs_accelerate_py/agent_supervisor/runtime/stage_backpressure.py"),
    ("scripts/lgswf_payloads/test_agent_supervisor_stage_backpressure.py",
     "test/api/test_agent_supervisor_stage_backpressure.py"),
)
if __name__ == "__main__":
    emit(copy_pairs(PAIRS, dest=Path.cwd()))
