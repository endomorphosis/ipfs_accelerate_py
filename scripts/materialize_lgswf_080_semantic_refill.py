#!/usr/bin/env python3
from pathlib import Path
from lgswf_emit_copy_writer import copy_pairs, emit
PAIRS = (
    ("scripts/lgswf_payloads/semantic_refill.py", "ipfs_accelerate_py/agent_supervisor/task_sources/semantic_refill.py"),
    ("scripts/lgswf_payloads/test_agent_supervisor_semantic_refill.py", "test/api/test_agent_supervisor_semantic_refill.py"),
)
if __name__ == "__main__":
    emit(copy_pairs(PAIRS, dest=Path.cwd()))
