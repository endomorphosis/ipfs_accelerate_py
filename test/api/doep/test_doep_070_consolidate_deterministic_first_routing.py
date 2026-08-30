"""Independent current-tree checks for the DOEP-070 routing seam."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.verification.contracts import ModelRoute
from ipfs_accelerate_py.agent_supervisor.verification.model_route import (
    AnalysisKind,
    CounterexampleQuality,
    ModelRouteFacts,
    ModelRoutePlanner,
    RiskLevel,
    default_inventory,
    decide_model_route,
    policy_cid_for,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
ROUTE_PATH = ACCELERATE_ROOT / "ipfs_accelerate_py/agent_supervisor/verification/model_route.py"
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = ACCELERATE_ROOT / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-070.json"
RECEIPT_PATH = ACCELERATE_ROOT / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-070.json"
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/verification/model_route.py",
    "test/api/doep/test_doep_070_consolidate_deterministic_first_routing.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-070.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-070.json",
)
TASK_CID = "sha256:a00298c9d0d6c3f176c9eb359e340adf971cff6072737601f9173558e14e0eea"
PLAN_CID = "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
BASE_REPOSITORIES = {
    "ipfs_accelerate_py": {"commit": "87715e9295626e7918f7fc8a7b1a1531ab04208f", "tree": "1c9a399cc7a599d5904e5be2ae58c6be3650cff7"},
    "ipfs_datasets_py": {"commit": "3668b8857a9aa7b1a3c847be12725b5cd057d2e7", "tree": "456e09b51d6a07a3a5873436df24054768195320"},
    "ipfs_kit_py": {"commit": "b6c65ba732733d7e33852713ba18aa3b12235668", "tree": "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2"},
    "lift_coding": {"commit": "bb8869ed72eb7002434345d9969efee729c4f7f6", "tree": "99e85bfe584b7688ffbeff86da1e612dd6893a42"},
}


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _facts(**updates: object) -> ModelRouteFacts:
    values: dict[str, object] = {
        "context_token_estimate": 2_048,
        "analysis_kind": AnalysisKind.LOCALIZED_EXACT,
        "risk_level": RiskLevel.LOW,
        "changed_file_count": 1,
        "dependency_cone_size": 2,
        "opaque_dependency_count": 0,
        "counterexample_quality": CounterexampleQuality.MINIMIZED,
        "environment_reproducible": True,
    }
    values.update(updates)
    return ModelRouteFacts(**values)


def _route(**updates: object) -> ModelRoute:
    return decide_model_route(
        _facts(**updates),
        available_models=default_inventory(),
        policy={"policy_cid": policy_cid_for("doep-070")},
    ).route


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_existing_canonical_planner_uses_the_shared_route_contract_and_ladder() -> None:
    assert ModelRoutePlanner.__module__ == "ipfs_accelerate_py.agent_supervisor.verification.model_route"
    assert _route(analysis_kind=AnalysisKind.MECHANICAL_IMPORT, counterexample_quality=CounterexampleQuality.NONE) is ModelRoute.DETERMINISTIC_ONLY
    assert _route() is ModelRoute.SMALL_LOCAL_MODEL
    assert _route(analysis_kind=AnalysisKind.MULTI_FILE_SYNTHESIS, changed_file_count=4, dependency_cone_size=12) is ModelRoute.MEDIUM_MODEL
    assert _route(analysis_kind=AnalysisKind.AMBIGUOUS) is ModelRoute.FRONTIER_MODEL
    assert _route(analysis_kind=AnalysisKind.MECHANICAL_RENAME, unresolved_authority=True) is ModelRoute.HUMAN_REVIEW_REQUIRED


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in ((manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"), (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1")):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-070"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(ROUTE_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == "pending_independent_fenced_supervisor"
