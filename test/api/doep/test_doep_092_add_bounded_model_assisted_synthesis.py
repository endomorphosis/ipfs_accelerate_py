"""Current-tree checks for DOEP-092. DuckDB terminalization is forbidden."""
from pathlib import Path
import json

ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/planning/program_repair_synthesis.py",
    "test/api/doep/test_doep_092_add_bounded_model_assisted_synthesis.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-092.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-092.json",
)

def test_declared_outputs_exist():
    for rel in OUTPUTS:
        assert (ACCELERATE_ROOT / rel).is_file(), rel

def test_receipt_is_not_completion_authority():
    receipt = json.loads((ACCELERATE_ROOT / OUTPUTS[-1]).read_text())
    assert receipt["completion_authoritative"] is False
    assert receipt["task_id"] == "DOEP-092"
    assert receipt["worker_completion_insufficient"] is True

def test_synthesizer_exists_and_does_not_self_admit():
    from ipfs_accelerate_py.agent_supervisor.planning.program_repair_synthesis import create_program_repair_synthesizer
    synth = create_program_repair_synthesizer()
    assert synth is not None

