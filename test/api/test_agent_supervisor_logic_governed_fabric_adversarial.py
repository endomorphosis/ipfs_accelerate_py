import json
from pathlib import Path
def test_adversarial_matrix_fail_closed():
    root = Path(__file__).resolve().parents[2]
    data = json.loads((root/"data/agent_supervisor/logic_governed_semantic_work_fabric/qualification/adversarial-results.json").read_text())
    assert data["fail_closed"] is True
    assert data["critical_cases"] >= 1
