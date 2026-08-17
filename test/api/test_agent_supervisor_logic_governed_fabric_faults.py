import json
from pathlib import Path
def test_fault_matrix_has_26_scenarios():
    root = Path(__file__).resolve().parents[2]
    data = json.loads((root/"data/agent_supervisor/logic_governed_semantic_work_fabric/qualification/fault-results.json").read_text())
    assert data["schema"] == "lgswf/fault-results@1"
    assert data["scenarios"] == 26
    assert data["fail_closed"] is True
