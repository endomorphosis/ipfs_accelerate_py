#!/usr/bin/env python3
from pathlib import Path
from lgswf_emit_copy_writer import copy_pairs, emit
PAIRS = (
    ("scripts/lgswf_payloads/work_packet.py", "ipfs_accelerate_py/agent_supervisor/todo_daemon/work_packet.py"),
    ("scripts/lgswf_payloads/test_agent_supervisor_semantic_work_packet.py", "test/api/test_agent_supervisor_semantic_work_packet.py"),
)
if __name__ == "__main__":
    emit(copy_pairs(PAIRS, dest=Path.cwd()))
