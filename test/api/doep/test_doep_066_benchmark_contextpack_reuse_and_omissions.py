"""Current-tree checks for DOEP-066. DuckDB terminalization is forbidden."""
from pathlib import Path
import json

ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = (
    "benchmarks/agent_supervisor/doep/context_pack.py",
    "test/api/doep/test_doep_066_benchmark_contextpack_reuse_and_omissions.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-066.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-066.json",
)

def test_declared_outputs_exist():
    for rel in OUTPUTS:
        assert (ACCELERATE_ROOT / rel).is_file(), rel

def test_receipt_is_not_completion_authority():
    receipt = json.loads((ACCELERATE_ROOT / OUTPUTS[-1]).read_text())
    assert receipt["completion_authoritative"] is False
    assert receipt["task_id"] == "DOEP-066"
    assert receipt["worker_completion_insufficient"] is True

def test_benchmark_does_not_admit_completion():
    from benchmarks.agent_supervisor.doep.context_pack import run_reuse_omission_benchmark
    result = run_reuse_omission_benchmark()
    assert result["completion_authority"] is False

