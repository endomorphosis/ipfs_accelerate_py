"""Current-tree checks for DOEP-094. DuckDB terminalization is forbidden."""
from pathlib import Path
import json

ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/post_merge_validation.py",
    "test/api/doep/test_doep_094_add_proof_test_based_merge_admission.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-094.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-094.json",
)

def test_declared_outputs_exist():
    for rel in OUTPUTS:
        assert (ACCELERATE_ROOT / rel).is_file(), rel

def test_receipt_is_not_completion_authority():
    receipt = json.loads((ACCELERATE_ROOT / OUTPUTS[-1]).read_text())
    assert receipt["completion_authoritative"] is False
    assert receipt["task_id"] == "DOEP-094"
    assert receipt["worker_completion_insufficient"] is True

def test_post_merge_validation_builder_exists():
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.post_merge_validation import build_post_merge_validation_evidence, verify_post_merge_validation_evidence
    assert callable(build_post_merge_validation_evidence)
    assert callable(verify_post_merge_validation_evidence)

