"""Independent current-tree checks for the DOEP-022 repository capability seam."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.formal_plan_compiler import compile_formal_plan
from ipfs_accelerate_py.agent_supervisor.planning.formal_plan_context import (
    FormalPlanContextError,
    FormalPlanContextLimits,
    RepositoryCapabilityAnalysis,
    RepositoryCapabilityStatus,
    analyze_repository_capabilities,
    build_formal_plan_context_capsule,
)
from ipfs_accelerate_py.agent_supervisor.planning.formal_plan_validator import validate_formal_plan


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
CONTEXT_PATH = ACCELERATE_ROOT / "ipfs_accelerate_py/agent_supervisor/planning/formal_plan_context.py"
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = ACCELERATE_ROOT / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-022.json"
RECEIPT_PATH = ACCELERATE_ROOT / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-022.json"
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/planning/formal_plan_context.py",
    "test/api/doep/test_doep_022_add_repository_capability_analysis.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-022.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-022.json",
)
TASK_CID = "sha256:5422b21bd2848b09884adfa8204f170313ac5bc484bee0dd0e758d8f943f36a0"
PLAN_CID = "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"


def _source() -> dict[str, Any]:
    return {
        "repository_tree_id": "tree:doep-022",
        "objectives": [{"goal_cid": "goal:analysis", "owner_actor_id": "supervisor"}],
        "tasks": [{"task_cid": "task:analysis", "goal_cid": "goal:analysis", "actor_id": "agent:analysis", "acceptance_criteria": ["focused tests pass"]}],
        "policies": [{"policy_cid": "policy:analysis"}],
    }


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _capsule(*, capabilities: list[dict[str, Any]] | None = None, limit: int = 32):
    compilation = compile_formal_plan(_source())
    assert compilation.plan is not None
    validation = validate_formal_plan(compilation.plan, compilation.formulas)
    return build_formal_plan_context_capsule(
        compilation,
        validation,
        task_id="task:analysis",
        limits=FormalPlanContextLimits(max_capabilities=limit),
        repository_id="repository:accelerate",
        repository_analysis_source_cid="receipt:inventory",
        repository_capabilities=capabilities or [],
    )


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_repository_capabilities_are_canonical_bounded_and_advisory() -> None:
    capsule = _capsule(
        capabilities=[
            {"capability_id": "tests", "available": False, "path": "test/api", "metadata": {"stderr": "private", "kind": "pytest"}},
            {"capability_id": "ast", "status": "available", "path": "src/../ipfs_accelerate_py", "evidence_cid": "evidence:ast"},
            {"capability_id": "proof", "status": "unknown"},
        ],
        limit=2,
    )
    analysis = capsule.repository_analysis
    assert analysis.repository_tree_cid == "tree:doep-022"
    assert analysis.repository_id == "repository:accelerate"
    assert analysis.source_cid == "receipt:inventory"
    assert analysis.truncated is True and analysis.omitted_capabilities == 1
    assert [item.capability_id for item in analysis.capabilities] == ["ast", "proof"]
    assert analysis.to_dict()["authority"] == "advisory_only"
    assert capsule.omitted["repository_capabilities"] == 1
    assert RepositoryCapabilityAnalysis.from_dict(analysis.to_dict()).analysis_cid == analysis.analysis_cid
    assert capsule.from_json(capsule.to_json()).repository_analysis.to_dict() == analysis.to_dict()


def test_capability_analysis_fails_closed_for_authority_and_tree_drift() -> None:
    with pytest.raises(FormalPlanContextError):
        analyze_repository_capabilities(
            "tree:doep-022",
            [{"capability_id": "unsafe", "details": {"completion_authoritative": True}}],
        )
    with pytest.raises(FormalPlanContextError):
        RepositoryCapabilityAnalysis.from_dict(
            {"repository_tree_cid": "tree:doep-022", "authority": "admitted", "capabilities": []}
        )
    capsule = _capsule()
    payload = capsule.to_dict()
    payload["repository_analysis"]["repository_tree_cid"] = "tree:other"
    with pytest.raises(FormalPlanContextError):
        capsule.from_dict(payload)
    assert RepositoryCapabilityStatus.AVAILABLE.value == "available"


def test_manifest_and_candidate_receipt_bind_exact_current_tree_outputs() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in ((manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"), (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1")):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-022"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(CONTEXT_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
