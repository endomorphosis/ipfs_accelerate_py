"""Current-tree checks for DOEP-125. DuckDB terminalization is forbidden."""
from pathlib import Path
import json

ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = (
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/release/promotion_decision.json",
    "test/api/doep/test_doep_125_produce_promotion_or_honest_non_promotion_receipt.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-125.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-125.json",
)

def test_declared_outputs_exist():
    for rel in OUTPUTS:
        assert (ACCELERATE_ROOT / rel).is_file(), rel

def test_receipt_is_not_completion_authority():
    receipt = json.loads((ACCELERATE_ROOT / OUTPUTS[-1]).read_text())
    assert receipt["completion_authoritative"] is False
    assert receipt["task_id"] == "DOEP-125"
    assert receipt["worker_completion_insufficient"] is True

def test_honest_non_promotion():
    payload = json.loads((ACCELERATE_ROOT / OUTPUTS[0]).read_text())
    assert payload["promoted"] is False
    assert payload["completion_authority"] is False

