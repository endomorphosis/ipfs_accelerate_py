"""Independent current-tree checks for DOEP-074 escalation policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.verification.contracts import ModelRoute
from ipfs_accelerate_py.agent_supervisor.verification.model_route import (
    DETERMINISTIC_FIRST_ROUTE_LADDER,
    MODEL_ESCALATION_LADDER,
    MODEL_ESCALATION_POLICY_INTERFACE,
    MODEL_ESCALATION_POLICY_SCHEMA,
    REASON_REQUIRED_TIER_UNAVAILABLE,
    REASON_SMALLER_ROUTE_FAILED,
    AnalysisKind,
    CounterexampleQuality,
    ModelRouteFacts,
    ModelRoutePolicy,
    PriorRepairAttempt,
    RiskLevel,
    apply_escalation_policy,
    decide_model_route,
    default_inventory,
    escalate_failed_routes,
    failed_model_routes,
    policy_cid_for,
    select_required_route,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
ROUTE_PATH = ACCELERATE_ROOT / "ipfs_accelerate_py/agent_supervisor/verification/model_route.py"
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-074.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-074.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/verification/model_route.py",
    "test/api/doep/test_doep_074_add_small_medium_frontier_escalation_policy.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-074.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-074.json",
)
TASK_CID = "sha256:91a296e2a05606ec13bef31e2c0a503acee771e5714c8f173eb3d156f86f8b8f"
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
_ROUTE_ORDER = {
    ModelRoute.DETERMINISTIC_ONLY: 0,
    ModelRoute.SMALL_LOCAL_MODEL: 1,
    ModelRoute.MEDIUM_MODEL: 2,
    ModelRoute.FRONTIER_MODEL: 3,
    ModelRoute.HUMAN_REVIEW_REQUIRED: 4,
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
        "exact_contract_available": True,
        "environment_reproducible": True,
    }
    values.update(updates)
    return ModelRouteFacts(**values)


def _policy() -> dict[str, str]:
    return {"policy_cid": policy_cid_for("doep-074")}


def _decide(facts: ModelRouteFacts, *, prior: list[PriorRepairAttempt] | tuple = (), inventory=None):
    return apply_escalation_policy(
        facts,
        prior_attempts=prior,
        available_models=inventory if inventory is not None else default_inventory(),
        policy=_policy(),
    )


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_escalation_policy_is_named_on_the_canonical_model_route_module() -> None:
    assert apply_escalation_policy.__module__ == (
        "ipfs_accelerate_py.agent_supervisor.verification.model_route"
    )
    assert escalate_failed_routes.__module__ == apply_escalation_policy.__module__
    assert DETERMINISTIC_FIRST_ROUTE_LADDER == (
        "receipt",
        "static",
        "tests",
        "proof",
        "small",
        "medium",
        "frontier",
        "human",
    )
    assert MODEL_ESCALATION_LADDER == (
        ModelRoute.SMALL_LOCAL_MODEL,
        ModelRoute.MEDIUM_MODEL,
        ModelRoute.FRONTIER_MODEL,
        ModelRoute.HUMAN_REVIEW_REQUIRED,
    )
    assert MODEL_ESCALATION_POLICY_INTERFACE == "ModelEscalationPolicy@1"
    assert MODEL_ESCALATION_POLICY_SCHEMA.endswith("verification-model-escalation-policy@1")


def test_failed_smaller_model_routes_escalate_monotonically() -> None:
    localized = _facts()
    small_failed = _decide(
        localized,
        prior=[PriorRepairAttempt(route=ModelRoute.SMALL_LOCAL_MODEL, failed=True)],
    )
    assert small_failed.route is ModelRoute.MEDIUM_MODEL
    assert REASON_SMALLER_ROUTE_FAILED in small_failed.decisive_reason_codes
    assert _ROUTE_ORDER[small_failed.route] > _ROUTE_ORDER[ModelRoute.SMALL_LOCAL_MODEL]

    medium_failed = _decide(
        _facts(
            analysis_kind=AnalysisKind.MULTI_FILE_SYNTHESIS,
            changed_file_count=4,
            dependency_cone_size=12,
            counterexample_quality=CounterexampleQuality.NONE,
        ),
        prior=[
            PriorRepairAttempt(route=ModelRoute.SMALL_LOCAL_MODEL, failed=True),
            PriorRepairAttempt(route=ModelRoute.MEDIUM_MODEL, failed=True),
        ],
    )
    assert medium_failed.route is ModelRoute.FRONTIER_MODEL
    assert REASON_SMALLER_ROUTE_FAILED in medium_failed.decisive_reason_codes

    frontier_failed = _decide(
        _facts(analysis_kind=AnalysisKind.AMBIGUOUS),
        prior=[PriorRepairAttempt(route=ModelRoute.FRONTIER_MODEL, failed=True)],
    )
    assert frontier_failed.route is ModelRoute.HUMAN_REVIEW_REQUIRED
    assert frontier_failed.requires_human_review

    failed = failed_model_routes(
        (
            PriorRepairAttempt(route=ModelRoute.SMALL_LOCAL_MODEL, failed=True),
            PriorRepairAttempt(route=ModelRoute.MEDIUM_MODEL, failed=False),
        )
    )
    assert failed == frozenset({ModelRoute.SMALL_LOCAL_MODEL})
    escalated, reasons = escalate_failed_routes(ModelRoute.SMALL_LOCAL_MODEL, (
        PriorRepairAttempt(route=ModelRoute.SMALL_LOCAL_MODEL, failed=True),
    ))
    assert escalated is ModelRoute.MEDIUM_MODEL
    assert REASON_SMALLER_ROUTE_FAILED in reasons


def test_unavailable_required_tier_never_downgrades() -> None:
    ambiguous = _decide(
        _facts(analysis_kind=AnalysisKind.AMBIGUOUS),
        inventory=default_inventory(small=True, medium=False, frontier=False),
    )
    assert ambiguous.route is ModelRoute.HUMAN_REVIEW_REQUIRED
    assert REASON_REQUIRED_TIER_UNAVAILABLE in ambiguous.decisive_reason_codes

    missing_small = _decide(
        localized := _facts(),
        inventory=default_inventory(small=False, medium=True, frontier=True),
    )
    assert localized.analysis_kind is AnalysisKind.LOCALIZED_EXACT
    assert missing_small.route is ModelRoute.HUMAN_REVIEW_REQUIRED
    assert REASON_REQUIRED_TIER_UNAVAILABLE in missing_small.decisive_reason_codes
    assert missing_small.route is not ModelRoute.DETERMINISTIC_ONLY


def test_apply_escalation_policy_agrees_with_decide_and_keeps_classifier() -> None:
    facts = _facts(analysis_kind=AnalysisKind.MULTI_FILE_SYNTHESIS, changed_file_count=5, dependency_cone_size=10)
    prior = (PriorRepairAttempt(route=ModelRoute.SMALL_LOCAL_MODEL, failed=True),)
    via_policy = apply_escalation_policy(
        facts,
        prior_attempts=prior,
        available_models=default_inventory(),
        policy=_policy(),
    )
    via_decide = decide_model_route(
        facts,
        prior_attempts=prior,
        available_models=default_inventory(),
        policy=_policy(),
    )
    assert via_policy.to_record() == via_decide.to_record()
    required, _reasons = select_required_route(
        facts,
        prior,
        ModelRoutePolicy.from_value(_policy()),
    )
    assert required in {
        ModelRoute.MEDIUM_MODEL,
        ModelRoute.FRONTIER_MODEL,
        ModelRoute.HUMAN_REVIEW_REQUIRED,
    }
    assert select_required_route.__module__.endswith("verification.model_route")


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-074"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["canonical_extension"]["entrypoint"] == "apply_escalation_policy"
    assert manifest["canonical_extension"]["carrier"] == "ModelRouteDecision"
    assert manifest["canonical_extension"]["module"].endswith("verification.model_route")
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(ROUTE_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == (
        "pending_independent_fenced_supervisor"
    )
