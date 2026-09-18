"""Current-tree checks for DOEP-093. DuckDB terminalization is forbidden."""
from pathlib import Path
import json

ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/validation/scope_adjudication.py",
    "test/api/doep/test_doep_093_add_patch_scope_and_semantic_nonempty_validation.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-093.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-093.json",
)

def test_declared_outputs_exist():
    for rel in OUTPUTS:
        assert (ACCELERATE_ROOT / rel).is_file(), rel

def test_receipt_is_not_completion_authority():
    receipt = json.loads((ACCELERATE_ROOT / OUTPUTS[-1]).read_text())
    assert receipt["completion_authoritative"] is False
    assert receipt["task_id"] == "DOEP-093"
    assert receipt["worker_completion_insufficient"] is True

def test_scope_adjudication_is_callable():
    from ipfs_accelerate_py.agent_supervisor.validation.scope_adjudication import adjudicate_scope_expansion
    assert callable(adjudicate_scope_expansion)

