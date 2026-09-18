"""Current-tree checks for DOEP-111. DuckDB terminalization is forbidden."""
from pathlib import Path
import json

ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = (
    "benchmarks/agent_supervisor/doep/baseline.py",
    "test/api/doep/test_doep_111_build_codex_primed_baseline_harness.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-111.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-111.json",
)

def test_declared_outputs_exist():
    for rel in OUTPUTS:
        assert (ACCELERATE_ROOT / rel).is_file(), rel

def test_receipt_is_not_completion_authority():
    receipt = json.loads((ACCELERATE_ROOT / OUTPUTS[-1]).read_text())
    assert receipt["completion_authoritative"] is False
    assert receipt["task_id"] == "DOEP-111"
    assert receipt["worker_completion_insufficient"] is True

def test_harness_is_not_live_completion():
    from benchmarks.agent_supervisor.doep.baseline import describe
    payload = describe()
    assert payload["completion_authority"] is False
    assert payload["live"] is False

