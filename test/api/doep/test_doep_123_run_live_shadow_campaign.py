"""Current-tree checks for DOEP-123. DuckDB terminalization is forbidden."""
from pathlib import Path
import json

ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = (
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/benchmarks/live_shadow.json",
    "test/api/doep/test_doep_123_run_live_shadow_campaign.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-123.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-123.json",
)

def test_declared_outputs_exist():
    for rel in OUTPUTS:
        assert (ACCELERATE_ROOT / rel).is_file(), rel

def test_receipt_is_not_completion_authority():
    receipt = json.loads((ACCELERATE_ROOT / OUTPUTS[-1]).read_text())
    assert receipt["completion_authoritative"] is False
    assert receipt["task_id"] == "DOEP-123"
    assert receipt["worker_completion_insufficient"] is True

def test_run_is_honest_non_execution():
    payload = json.loads((ACCELERATE_ROOT / OUTPUTS[0]).read_text())
    assert payload["ran"] is False
    assert payload["promoted"] is False
    assert payload["completion_authority"] is False

