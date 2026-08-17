#!/usr/bin/env python3
from pathlib import Path
from lgswf_emit_copy_writer import copy_pairs, emit
PAIRS = (
    ("scripts/lgswf_payloads/work_loop.py", "ipfs_accelerate_py/agent_supervisor/semantic_state/work_loop.py"),
    ("scripts/lgswf_payloads/test_semantic_work_loop.py", "test/api/semantic_state/test_semantic_work_loop.py"),
)
if __name__ == "__main__":
    emit(copy_pairs(PAIRS, dest=Path.cwd()))
