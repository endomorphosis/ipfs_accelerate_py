"""Current-tree checks for DOEP-065 ContextPack invalidation."""
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
    ContextBudget,
    ContextCapsule,
    invalidate_context_capsule,
)

ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/context/context_contracts.py",
    "test/api/doep/test_doep_065_add_contextpack_invalidation.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-065.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-065.json",
)

def _capsule(tree_id="tree:a"):
    return ContextCapsule(
        repository_id="repo:doep",
        tree_id=tree_id,
        objective_id="DOEP-G070",
        objective_revision="sha256:obj",
        policy_id="policy:supervisor",
        policy_revision="sha256:pol",
        caller="supervisor:doep-065",
        stage="planning",
        budget=ContextBudget(max_input_tokens=220, reserved_output_tokens=40, reserved_tool_tokens=10, max_items=16, max_item_bytes=16384, max_serialized_bytes=262144),
        goal={"id": "g"},
        authority={"mode": "proposal"},
        scope={"paths": ["src"]},
        acceptance={"criteria": ["fresh"]},
    )

def test_declared_outputs_exist():
    for rel in OUTPUTS:
        assert (ACCELERATE_ROOT / rel).is_file()

def test_same_tree_stays_valid():
    result = invalidate_context_capsule(_capsule("tree:a"), tree_id="tree:a", reason="check")
    assert result["valid"] is True
    assert result["completion_authority"] is False

def test_changed_tree_invalidates_without_completing():
    result = invalidate_context_capsule(_capsule("tree:a"), tree_id="tree:b", reason="tree_mismatch")
    assert result["stale"] is True
    assert result["valid"] is False
    assert result["completion_authoritative"] is False
