#!/usr/bin/env python3
from pathlib import Path
from lgswf_emit_copy_writer import copy_pairs, emit
PAIRS = (
    ("scripts/lgswf_payloads/decision_receipts.py", "ipfs_accelerate_py/agent_supervisor/runtime/decision_receipts.py"),
    ("scripts/lgswf_payloads/logic_governed_fabric_entrypoint.py", "ipfs_accelerate_py/agent_supervisor/entrypoints/logic_governed_fabric.py"),
    ("scripts/lgswf_payloads/ducklake_history_projection.py", "ipfs_accelerate_py/agent_supervisor/integrations/ducklake_history_projection.py"),
    ("scripts/lgswf_payloads/test_agent_supervisor_logic_governed_observability.py", "test/api/test_agent_supervisor_logic_governed_observability.py"),
    ("scripts/lgswf_payloads/test_agent_supervisor_ducklake_history_projection.py", "test/api/test_agent_supervisor_ducklake_history_projection.py"),
)
if __name__ == "__main__":
    emit(copy_pairs(PAIRS, dest=Path.cwd()))
