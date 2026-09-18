"""Current-tree checks for DOEP-117. DuckDB terminalization is forbidden."""
from pathlib import Path
import json

ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = (
    "benchmarks/agent_supervisor/doep/low_risk_canary.py",
    "test/api/doep/test_doep_117_add_low_risk_canary_cohort.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-117.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-117.json",
)

def test_declared_outputs_exist():
    for rel in OUTPUTS:
        assert (ACCELERATE_ROOT / rel).is_file(), rel

def test_receipt_is_not_completion_authority():
    receipt = json.loads((ACCELERATE_ROOT / OUTPUTS[-1]).read_text())
    assert receipt["completion_authoritative"] is False
    assert receipt["task_id"] == "DOEP-117"
    assert receipt["worker_completion_insufficient"] is True

def test_harness_is_not_live_completion():
    from benchmarks.agent_supervisor.doep.low_risk_canary import describe
    payload = describe()
    assert payload["completion_authority"] is False
    assert payload["live"] is False

