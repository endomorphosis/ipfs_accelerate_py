#!/usr/bin/env python3
from pathlib import Path
from lgswf_emit_copy_writer import copy_pairs, emit
PAIRS = (
    ("scripts/lgswf_payloads/harness.py", "ipfs_accelerate_py/agent_supervisor/semantic_state/harness.py"),
    ("scripts/lgswf_payloads/semantic_refresh_integration.py", "ipfs_accelerate_py/agent_supervisor/merge/semantic_refresh_integration.py"),
    ("scripts/lgswf_payloads/test_closed_loop_semantic_refresh.py", "test/api/semantic_state/test_closed_loop_semantic_refresh.py"),
)
if __name__ == "__main__":
    emit(copy_pairs(PAIRS, dest=Path.cwd()))
