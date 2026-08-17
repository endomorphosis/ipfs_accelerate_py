#!/usr/bin/env python3
from pathlib import Path
from lgswf_copy_owned import copy_owned, emit
OWNED = (
    "ipfs_accelerate_py/agent_supervisor/planning/work_graph_metrics.py",
    "test/api/test_agent_supervisor_work_graph_metrics.py",
)
if __name__ == "__main__":
    emit(copy_owned(OWNED, dest=Path.cwd()))
