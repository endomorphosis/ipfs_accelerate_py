"""Current-tree checks for DOEP-046."""
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import TaskState

ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = (
    "test/api/doep/test_control_plane_model.py",
    "test/api/doep/test_doep_046_add_model_based_and_temporal_invariant_tests.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-046.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-046.json",
)

def test_declared_outputs_exist():
    for rel in OUTPUTS:
        assert (ACCELERATE_ROOT / rel).is_file()

def test_receipt_is_not_completion_authority():
    import json
    receipt = json.loads((ACCELERATE_ROOT / OUTPUTS[3]).read_text())
    assert receipt["completion_authoritative"] is False
    assert receipt["task_id"] == "DOEP-046"

def test_control_plane_states_remain_distinct():
    values = [getattr(item, "value", str(item)) for item in TaskState]
    assert len(values) == len(set(values))
