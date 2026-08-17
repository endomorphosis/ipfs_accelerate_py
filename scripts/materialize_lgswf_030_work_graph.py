#!/usr/bin/env python3
from pathlib import Path
from lgswf_copy_owned import copy_owned, emit
OWNED = (
    "ipfs_accelerate_py/agent_supervisor/planning/semantic_work_graph.py",
    "test/api/test_agent_supervisor_semantic_work_graph.py",
)
if __name__ == "__main__":
    emit(copy_owned(OWNED, dest=Path.cwd()))
