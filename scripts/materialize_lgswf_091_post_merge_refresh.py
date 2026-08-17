#!/usr/bin/env python3
from pathlib import Path
from lgswf_emit_copy_writer import copy_pairs, emit
PAIRS = (
    ("scripts/lgswf_payloads/post_merge_refresh.py", "ipfs_accelerate_py/agent_supervisor/semantic_state/post_merge_refresh.py"),
    ("scripts/lgswf_payloads/test_post_merge_semantic_refresh.py", "test/api/semantic_state/test_post_merge_semantic_refresh.py"),
)
if __name__ == "__main__":
    emit(copy_pairs(PAIRS, dest=Path.cwd()))
