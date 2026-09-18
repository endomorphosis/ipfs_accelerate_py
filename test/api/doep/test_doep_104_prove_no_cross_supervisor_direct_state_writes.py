"""Current-tree checks for DOEP-104. DuckDB terminalization is forbidden."""
from pathlib import Path
import json

ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = (
    "test/api/doep/test_cross_supervisor_isolation.py",
    "test/api/doep/test_doep_104_prove_no_cross_supervisor_direct_state_writes.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-104.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-104.json",
)

def test_declared_outputs_exist():
    for rel in OUTPUTS:
        assert (ACCELERATE_ROOT / rel).is_file(), rel

def test_receipt_is_not_completion_authority():
    receipt = json.loads((ACCELERATE_ROOT / OUTPUTS[-1]).read_text())
    assert receipt["completion_authoritative"] is False
    assert receipt["task_id"] == "DOEP-104"
    assert receipt["worker_completion_insufficient"] is True

def test_isolation_module_forbids_direct_store_writes():
    from test.api.doep.test_cross_supervisor_isolation import FORBIDDEN
    assert "direct DuckDB write" in FORBIDDEN

