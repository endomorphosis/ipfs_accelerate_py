"""Current-tree checks for DOEP-110. DuckDB terminalization is forbidden."""
from pathlib import Path
import json

ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/runtime/benchmark_telemetry.py",
    "test/api/doep/test_doep_110_instrument_end_to_end_token_and_compute_telemetry.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-110.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-110.json",
)

def test_declared_outputs_exist():
    for rel in OUTPUTS:
        assert (ACCELERATE_ROOT / rel).is_file(), rel

def test_receipt_is_not_completion_authority():
    receipt = json.loads((ACCELERATE_ROOT / OUTPUTS[-1]).read_text())
    assert receipt["completion_authoritative"] is False
    assert receipt["task_id"] == "DOEP-110"
    assert receipt["worker_completion_insufficient"] is True

def test_telemetry_session_exists():
    from ipfs_accelerate_py.agent_supervisor.runtime.benchmark_telemetry import BenchmarkTelemetrySession
    assert BenchmarkTelemetrySession is not None

