#!/usr/bin/env python3
from pathlib import Path
from lgswf_emit_copy_writer import copy_pairs, emit
PAIRS = (
    ("scripts/lgswf_payloads/work_partitioning.py",
     "ipfs_accelerate_py/agent_supervisor/runtime/work_partitioning.py"),
    ("scripts/lgswf_payloads/test_agent_supervisor_work_partitioning.py",
     "test/api/test_agent_supervisor_work_partitioning.py"),
)
if __name__ == "__main__":
    emit(copy_pairs(PAIRS, dest=Path.cwd()))
