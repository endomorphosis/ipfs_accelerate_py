#!/usr/bin/env python3
from pathlib import Path
from lgswf_emit_copy_writer import copy_pairs, emit
PAIRS = (
    ("scripts/lgswf_payloads/lgswf_plan_doctor.py",
     "ipfs_accelerate_py/agent_supervisor/planning/plan_doctor.py"),
    ("scripts/lgswf_payloads/test_agent_supervisor_plan_doctor.py",
     "test/api/test_agent_supervisor_plan_doctor.py"),
)
if __name__ == "__main__":
    emit(copy_pairs(PAIRS, dest=Path.cwd()))
