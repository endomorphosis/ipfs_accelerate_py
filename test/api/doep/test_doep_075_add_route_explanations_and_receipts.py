"""Independent current-tree checks for DOEP-075 route explanations and receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.decision_receipts import (
    DECISION_RECEIPT_SCHEMA,
    DETERMINISTIC_FIRST_ROUTE_LADDER,
    ROUTE_EXPLANATION_INTERFACE,
    ROUTE_EXPLANATION_RECEIPT_INTERFACE,
    ROUTE_EXPLANATION_RECEIPT_SCHEMA,
    ROUTE_EXPLANATION_SCHEMA,
    ROUTE_LADDER_RUNGS,
    DecisionReceiptError,
    RouteExplanationReceipt,
    emit_decision,
    emit_route_explanation_receipt,
    explain_model_route,
)
from ipfs_accelerate_py.agent_supervisor.verification.contracts import ModelRoute
from ipfs_accelerate_py.agent_supervisor.verification.model_route import (
    REASON_LOCALIZED_EXACT_COUNTEREXAMPLE,
    REASON_REQUIRED_TIER_UNAVAILABLE,
    REASON_SMALLER_ROUTE_FAILED,
    REASON_UNRESOLVED_AUTHORITY,
    AnalysisKind,
    CounterexampleQuality,
    ModelRouteFacts,
    PriorRepairAttempt,
    RiskLevel,
    apply_escalation_policy,
    decide_model_route,
    default_inventory,
    policy_cid_for,
)


ACCELERATE_ROOT = Path(__file__).resolve().parents[3]
RECEIPTS_PATH = (
    ACCELERATE_ROOT / "ipfs_accelerate_py/agent_supervisor/runtime/decision_receipts.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-075.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-075.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/runtime/decision_receipts.py",
    "test/api/doep/test_doep_075_add_route_explanations_and_receipts.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-075.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-075.json",
)
TASK_CID = "sha256:27ae4096dd8a6cdfe8d9662ee88eb01c510ebc043da79c441cdbbaa7a2904050"
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
    return {"policy_cid": policy_cid_for("doep-075")}


def _decide(facts: ModelRouteFacts, *, prior=(), inventory=None):
    return apply_escalation_policy(
        facts,
        prior_attempts=prior,
        available_models=inventory if inventory is not None else default_inventory(),
        policy=_policy(),
    )


def test_declared_outputs_exist() -> None:
    for relative in OWNER_RELATIVE_OUTPUTS:
        assert (ACCELERATE_ROOT / relative).is_file(), f"missing declared output: {relative}"


def test_route_explanation_lives_on_canonical_decision_receipts_module() -> None:
    assert explain_model_route.__module__.endswith("runtime.decision_receipts")
    assert emit_route_explanation_receipt.__module__.endswith("runtime.decision_receipts")
    assert emit_decision.__module__.endswith("runtime.decision_receipts")
    assert ROUTE_EXPLANATION_INTERFACE == "RouteExplanation@1"
    assert ROUTE_EXPLANATION_RECEIPT_INTERFACE == "RouteExplanationReceipt@1"
    assert ROUTE_EXPLANATION_SCHEMA.endswith("route-explanation@1")
    assert ROUTE_EXPLANATION_RECEIPT_SCHEMA.endswith("route-explanation-receipt@1")
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
    assert ROUTE_LADDER_RUNGS[ModelRoute.SMALL_LOCAL_MODEL] == "small"
    assert ROUTE_LADDER_RUNGS[ModelRoute.HUMAN_REVIEW_REQUIRED] == "human"


def test_lgswf_emit_decision_envelope_remains_compatible() -> None:
    receipt = emit_decision({"decision": "select", "metrics": {"n": 1}})
    assert receipt["schema"] == DECISION_RECEIPT_SCHEMA
    assert receipt["decision"] == "select"
    assert receipt["metrics"] == {"n": 1}
    assert receipt["authoritative"] is False
    assert receipt["completion_authoritative"] is False
    with pytest.raises(TypeError):
        receipt["decision"] = "mutated"  # type: ignore[index]


def test_explanation_projects_canonical_model_route_decision_without_reclassifying() -> None:
    localized = _decide(_facts())
    assert localized.route is ModelRoute.SMALL_LOCAL_MODEL
    explanation = explain_model_route(localized)
    receipt = emit_route_explanation_receipt(localized)

    assert explanation.route is localized.route
    assert explanation.decision_id == localized.decision_id
    assert explanation.policy_cid == localized.policy_cid
    assert explanation.decisive_reason_codes == localized.decisive_reason_codes
    assert explanation.requires_human_review is localized.requires_human_review
    assert explanation.ladder_rung == "small"
    assert explanation.escalated is False
    assert REASON_LOCALIZED_EXACT_COUNTEREXAMPLE in explanation.decisive_reason_codes
    assert REASON_LOCALIZED_EXACT_COUNTEREXAMPLE in explanation.explanation
    assert receipt["schema"] == ROUTE_EXPLANATION_RECEIPT_SCHEMA
    assert receipt["decision_id"] == localized.decision_id
    assert receipt["route"] == localized.route.value
    assert receipt["authoritative"] is False
    assert receipt["completion_authoritative"] is False
    assert "provider" not in receipt
    assert "vendor" not in receipt

    # Same decision => same receipt identity; different reasons => different identity.
    again = emit_route_explanation_receipt(localized)
    assert again["receipt_id"] == receipt["receipt_id"]

    escalated = _decide(
        _facts(),
        prior=[PriorRepairAttempt(route=ModelRoute.SMALL_LOCAL_MODEL, failed=True)],
    )
    assert escalated.route is ModelRoute.MEDIUM_MODEL
    escalated_explanation = explain_model_route(escalated)
    assert escalated_explanation.ladder_rung == "medium"
    assert escalated_explanation.escalated is True
    assert REASON_SMALLER_ROUTE_FAILED in escalated_explanation.decisive_reason_codes
    assert REASON_SMALLER_ROUTE_FAILED in escalated_explanation.explanation
    escalated_receipt = emit_route_explanation_receipt(escalated)
    assert escalated_receipt["receipt_id"] != receipt["receipt_id"]

    unavailable = _decide(
        _facts(analysis_kind=AnalysisKind.AMBIGUOUS),
        inventory=default_inventory(small=True, medium=False, frontier=False),
    )
    assert unavailable.route is ModelRoute.HUMAN_REVIEW_REQUIRED
    human_explanation = explain_model_route(unavailable)
    assert human_explanation.ladder_rung == "human"
    assert human_explanation.requires_human_review is True
    assert REASON_REQUIRED_TIER_UNAVAILABLE in human_explanation.decisive_reason_codes

    gated = _decide(_facts(unresolved_authority=True))
    assert gated.route is ModelRoute.HUMAN_REVIEW_REQUIRED
    gated_explanation = explain_model_route(gated)
    assert REASON_UNRESOLVED_AUTHORITY in gated_explanation.decisive_reason_codes
    assert gated_explanation.ladder_rung == "human"

    # Projection agrees whether the caller used decide_model_route or apply_escalation_policy.
    via_decide = decide_model_route(
        _facts(),
        available_models=default_inventory(),
        policy=_policy(),
    )
    assert explain_model_route(via_decide).to_dict() == explain_model_route(localized).to_dict()


def test_emit_decision_nests_route_explanation_for_model_route_decisions() -> None:
    decision = _decide(_facts())
    nested = emit_decision({"decision": decision, "metrics": {"calls": 0}})
    assert nested["schema"] == DECISION_RECEIPT_SCHEMA
    assert nested["route_explanation"]["schema"] == ROUTE_EXPLANATION_SCHEMA
    assert nested["route_receipt"]["schema"] == ROUTE_EXPLANATION_RECEIPT_SCHEMA
    assert nested["decision_id"] == decision.decision_id
    assert nested["model_route_decision"]["decision_id"] == decision.decision_id
    assert nested["route_receipt"]["decision_id"] == decision.decision_id
    round_trip = RouteExplanationReceipt.from_dict(dict(nested["route_receipt"]))
    assert round_trip.receipt_id == nested["route_receipt"]["receipt_id"]


def test_competing_routing_objects_and_provider_identity_fail_closed() -> None:
    class RoutingDecision:
        pass

    with pytest.raises(DecisionReceiptError, match="competing semantic_state"):
        explain_model_route(RoutingDecision())  # type: ignore[arg-type]

    decision = _decide(_facts())
    with pytest.raises(DecisionReceiptError, match="provider/vendor identity"):
        emit_route_explanation_receipt(decision, metrics={"provider": "openai"})
    with pytest.raises(DecisionReceiptError, match="never authoritative"):
        emit_decision({"decision": "select", "authoritative": True})


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-075"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["canonical_extension"]["module"].endswith("runtime.decision_receipts")
    assert manifest["canonical_extension"]["entrypoint"] == "emit_route_explanation_receipt"
    assert manifest["canonical_extension"]["carrier"] == "RouteExplanationReceipt"
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(RECEIPTS_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == (
        "pending_independent_fenced_supervisor"
    )
