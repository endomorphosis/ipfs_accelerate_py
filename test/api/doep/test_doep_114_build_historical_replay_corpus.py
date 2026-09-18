"""Current-tree checks for DOEP-114. DuckDB terminalization is forbidden."""
from pathlib import Path
import json

ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = (
    "test/fixtures/agent_supervisor_doep/historical_replays.json",
    "test/api/doep/test_doep_114_build_historical_replay_corpus.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-114.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-114.json",
)

def test_declared_outputs_exist():
    for rel in OUTPUTS:
        assert (ACCELERATE_ROOT / rel).is_file(), rel

def test_receipt_is_not_completion_authority():
    receipt = json.loads((ACCELERATE_ROOT / OUTPUTS[-1]).read_text())
    assert receipt["completion_authoritative"] is False
    assert receipt["task_id"] == "DOEP-114"
    assert receipt["worker_completion_insufficient"] is True

def test_fixture_does_not_admit_completion():
    payload = json.loads((ACCELERATE_ROOT / "test/fixtures/agent_supervisor_doep/historical_replays.json").read_text())
    assert payload["completion_authority"] is False

