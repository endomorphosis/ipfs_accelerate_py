"""Independent current-tree checks for the DOEP-027 plan validation seam."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ipfs_accelerate_py.agent_supervisor.planning.formal_plan_compiler import (
    CompilationStatus,
    compile_formal_plan,
)
from ipfs_accelerate_py.agent_supervisor.planning.formal_plan_validator import (
    CANONICAL_PLAN_COMPLETENESS_WITNESS,
    CANONICAL_PLAN_VALIDATOR,
    FormalPlanValidator,
    PlanCheckKind,
    PlanCompletenessWitness,
    PlanValidationResult,
    PlanValidationStatus,
    ValidationBounds,
    check_formal_plan,
    plan_completeness_witness,
    validate_formal_plan,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
VALIDATOR_PATH = (
    ACCELERATE_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "planning"
    / "formal_plan_validator.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "outputs"
    / "DOEP-027.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts"
    / "agent_supervisor_direct_objective_event_driven_planning"
    / "receipts"
    / "DOEP-027.json"
)

OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/planning/formal_plan_validator.py",
    "test/api/doep/test_doep_027_add_plan_validation_and_completeness_witness.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-027.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-027.json",
)
TASK_CID = "sha256:8f3052cd9bdbcddaa8c1b416c84f2a088ec7999d3754003a1288226d28f96685"
PLAN_CID = "sha256:6c197a4b92682b3b813656123e09956846dc4f5abadf417f37fb7cc0133ddba4"
BASE_REPOSITORIES = {
    "ipfs_accelerate_py": {
        "commit": "87715e9295626e7918f7fc8a7b1a1531ab04208f",
        "tree": "1c9a399cc7a599d5904e5be2ae58c6be3650cff7",
    },
    "ipfs_datasets_py": {
        "commit": "3668b8857a9aa7b1a3c847be12725b5cd057d2e7",
        "tree": "456e09b51d6a07a3a5873436df24054768195320",
    },
    "ipfs_kit_py": {
        "commit": "b6c65ba732733d7e33852713ba18aa3b12235668",
        "tree": "14da7d92e130b7ba3523d0d6741a3ef7ef1e1bc2",
    },
    "lift_coding": {
        "commit": "bb8869ed72eb7002434345d9969efee729c4f7f6",
        "tree": "99e85bfe584b7688ffbeff86da1e612dd6893a42",
    },
}


def _source() -> dict[str, Any]:
    return {
        "repository_tree_id": "tree:doep-027",
        "objectives": [
            {
                "goal_id": "G",
                "goal_cid": "goal:doep-027",
                "owner_actor_id": "supervisor",
                "acceptance_criteria": ["plan validation completeness witness binds"],
            }
        ],
        "tasks": [
            {
                "task_id": "A",
                "task_cid": "task:doep-027",
                "goal_id": "G",
                "actor_id": "agent:doep-027",
                "resource_needs": ["cpu"],
                "acceptance_criteria": ["focused tests pass"],
                "validation_commands": [
                    "pytest test/api/doep/test_doep_027_add_plan_validation_and_completeness_witness.py"
                ],
                "lease": {
                    "lease_cid": "lease:doep-027",
                    "holder_id": "agent:doep-027",
                    "fencing_token": 27,
                },
            }
        ],
        "ast": [
            {
                "symbol_cid": "symbol:doep-027",
                "tree_cid": "tree:doep-027",
                "task_cid": "task:doep-027",
            }
        ],
        "policies": [
            {
                "policy_cid": "policy:doep-027",
                "minimum_code_assurance": "candidate",
                "fallback_check_ids": ["policy-review"],
            }
        ],
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _compiled():
    result = compile_formal_plan(_source())
    assert result.status is CompilationStatus.COMPILED
    assert result.plan is not None
    return result.plan, result.formulas


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_canonical_validator_emits_a_plan_completeness_witness() -> None:
    plan, formulas = _compiled()

    via_function = validate_formal_plan(plan, formulas)
    via_instance = FormalPlanValidator().validate(plan, formulas)
    witness = via_function.completeness_witness

    assert CANONICAL_PLAN_VALIDATOR == "FormalPlanValidator@1"
    assert CANONICAL_PLAN_COMPLETENESS_WITNESS == "PlanCompletenessWitness@1"
    assert validate_formal_plan is check_formal_plan
    assert (
        validate_formal_plan.__module__
        == "ipfs_accelerate_py.agent_supervisor.planning.formal_plan_validator"
    )
    assert via_function.to_dict() == via_instance.to_dict()
    assert via_function.status is PlanValidationStatus.CONSISTENT
    assert via_function.plan_check_only
    assert witness is not None
    assert witness.complete is True
    assert witness.truncated is False
    assert witness.omitted_reference_ids == ()
    assert witness.plan_check_only is True
    assert witness.plan_id == via_function.plan_id
    assert witness.bounds_id == via_function.bounds.bounds_id
    assert witness.formula_ids == via_function.formula_ids
    assert witness.assumption_ids == via_function.assumption_ids
    assert set(witness.checks_performed) >= {
        PlanCheckKind.DEPENDENCY_READINESS,
        PlanCheckKind.ACTOR_AUTHORITY,
        PlanCheckKind.UNIQUE_LEASE,
        PlanCheckKind.FENCING,
        PlanCheckKind.REQUIRED_EVIDENCE,
        PlanCheckKind.LEGAL_TRANSITION,
        PlanCheckKind.EVENTUAL_TERMINAL,
        PlanCheckKind.FORBIDDEN_MERGE_STATE,
        PlanCheckKind.DEONTIC_CONSISTENCY,
        PlanCheckKind.TEMPORAL_FORMULA,
    }
    assert plan.plan_id in witness.included_reference_ids
    assert PlanCompletenessWitness.from_dict(witness.to_dict()).witness_id == witness.witness_id
    assert plan_completeness_witness(via_function).witness_id == witness.witness_id
    restored = PlanValidationResult.from_dict(via_function.to_record())
    assert restored.validation_id == via_function.validation_id
    assert restored.completeness_witness is not None
    assert restored.completeness_witness.witness_id == witness.witness_id

    truncated = validate_formal_plan(
        plan, formulas, bounds=ValidationBounds(max_trace_steps=2)
    )
    assert truncated.status is PlanValidationStatus.INCOMPLETE
    assert truncated.completeness_witness is not None
    assert truncated.completeness_witness.complete is False
    assert truncated.completeness_witness.truncated is True
    assert truncated.completeness_witness.truncated_dimensions == ("trace_steps",)
    assert "trace_steps" in truncated.completeness_witness.omitted_reference_ids

    timed_out = validate_formal_plan(plan, formulas, bounds=ValidationBounds(timeout_ms=0))
    assert timed_out.status is PlanValidationStatus.TIMED_OUT
    assert timed_out.completeness_witness is not None
    assert timed_out.completeness_witness.complete is False
    assert "timeout" in timed_out.completeness_witness.omitted_reference_ids


def test_manifest_and_candidate_receipt_bind_the_exact_current_tree_outputs() -> None:
    manifest = _load_json(OUTPUT_PATH)
    receipt = _load_json(RECEIPT_PATH)

    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-027"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["plan_revision"] == "DOEP-PLAN-V5"
        assert payload["board_namespace"] == (
            "agent-supervisor-direct-objective-and-event-driven-planning-v1"
        )
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True

    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["canonical_extension"]["module"] == (
        "ipfs_accelerate_py.agent_supervisor.planning.formal_plan_validator"
    )
    assert manifest["canonical_extension"]["entrypoint"] == "validate_formal_plan"
    assert manifest["canonical_extension"]["carrier"] == "PlanCompletenessWitness"
    assert manifest["canonical_extension"]["binding"] == (
        "PlanValidationResult.completeness_witness"
    )
    assert manifest["canonical_extension"]["identifier"] == CANONICAL_PLAN_VALIDATOR
    assert manifest["canonical_extension"]["authority"] == "plan_check_only"
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["expected_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["write_scope"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["base_repositories"] == BASE_REPOSITORIES
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(VALIDATOR_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == (
        "pending_independent_fenced_supervisor"
    )
    assert receipt["required_evidence"]["source_commit_tree_gitlinks"] == BASE_REPOSITORIES
