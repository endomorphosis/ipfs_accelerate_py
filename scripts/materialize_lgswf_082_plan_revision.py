#!/usr/bin/env python3
from pathlib import Path
from lgswf_emit_copy_writer import copy_pairs, emit
# owned: plan_revision_store.py, plan_revision_contracts.py, test
# only add the test if store already exists
PAIRS = (
    ("scripts/lgswf_payloads/test_agent_supervisor_lgswf_plan_revision.py",
     "test/api/test_agent_supervisor_lgswf_plan_revision.py"),
)
if __name__ == "__main__":
    emit(copy_pairs(PAIRS, dest=Path.cwd()))
