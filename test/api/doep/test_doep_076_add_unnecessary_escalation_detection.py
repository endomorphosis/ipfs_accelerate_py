"""Independent current-tree checks for DOEP-076 unnecessary-escalation detection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.decision_receipts import (
    explain_model_route,
    emit_route_explanation_receipt,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement.supervisor_efficiency_metrics import (
    REASON_JUSTIFIED_FAILURE_ESCALATION,
    REASON_LOWER_ROUTE_SUCCEEDED,
    REASON_OBSERVED_ABOVE_JUSTIFIED_WITHOUT_FAILURE,
    REASON_REQUIRED_TIER_UNAVAILABLE_EXCUSED,
    REASON_ROUTE_MATCHES_JUSTIFIED,
    REASON_SKIPPED_DETERMINISTIC_ELIGIBILITY,
    UNNECESSARY_ESCALATION_INTERFACE,
    UNNECESSARY_ESCALATION_OBSERVATION_SCHEMA,
    UNNECESSARY_ESCALATION_REPORT_SCHEMA,
    EfficiencyValidationError,
    UnnecessaryEscalationObservation,
    UnnecessaryEscalationReport,
    aggregate_unnecessary_escalation_observations,
    detect_unnecessary_escalation,
)
from ipfs_accelerate_py.agent_supervisor.verification.contracts import (
    ModelRoute,
    ModelRouteDecision,
)
from ipfs_accelerate_py.agent_supervisor.verification.model_route import (
    REASON_REQUIRED_TIER_UNAVAILABLE,
    REASON_SMALLER_ROUTE_FAILED,
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
METRICS_PATH = (
    ACCELERATE_ROOT
    / "ipfs_accelerate_py/agent_supervisor/self_improvement/supervisor_efficiency_metrics.py"
)
TEST_PATH = Path(__file__).resolve()
OUTPUT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-076.json"
)
RECEIPT_PATH = (
    ACCELERATE_ROOT
    / "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-076.json"
)
OWNER_RELATIVE_OUTPUTS = (
    "ipfs_accelerate_py/agent_supervisor/self_improvement/supervisor_efficiency_metrics.py",
    "test/api/doep/test_doep_076_add_unnecessary_escalation_detection.py",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/outputs/DOEP-076.json",
    "artifacts/agent_supervisor_direct_objective_event_driven_planning/receipts/DOEP-076.json",
)
TASK_CID = "sha256:553a7e4096927c36e06bd991387a0dd76921a9cfa350e1439c43287cb7b089cf"
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
    return {"policy_cid": policy_cid_for("doep-076")}


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


def test_detection_lives_on_canonical_efficiency_metrics_module() -> None:
    assert detect_unnecessary_escalation.__module__.endswith(
        "self_improvement.supervisor_efficiency_metrics"
    )
    assert aggregate_unnecessary_escalation_observations.__module__ == (
        detect_unnecessary_escalation.__module__
    )
    assert UnnecessaryEscalationObservation.SCHEMA == (
        UNNECESSARY_ESCALATION_OBSERVATION_SCHEMA
    )
    assert UnnecessaryEscalationReport.SCHEMA == UNNECESSARY_ESCALATION_REPORT_SCHEMA
    assert UNNECESSARY_ESCALATION_INTERFACE == "UnnecessaryEscalationDetection@1"
    assert UNNECESSARY_ESCALATION_OBSERVATION_SCHEMA.endswith(
        "unnecessary-escalation-observation@1"
    )


def test_matching_route_is_not_unnecessary_and_uses_canonical_classifier() -> None:
    localized = _decide(_facts())
    assert localized.route is ModelRoute.SMALL_LOCAL_MODEL
    observation = detect_unnecessary_escalation(
        localized,
        facts=_facts(),
        policy=_policy(),
    )
    assert observation.unnecessary is False
    assert REASON_ROUTE_MATCHES_JUSTIFIED in observation.reason_codes
    assert observation.observed_route == ModelRoute.SMALL_LOCAL_MODEL.value
    assert observation.justified_route == ModelRoute.SMALL_LOCAL_MODEL.value
    assert observation.ladder_rung_observed == "small"
    payload = observation.to_dict()
    assert payload["authoritative"] is False
    assert payload["completion_authoritative"] is False
    assert payload["schema"] == UNNECESSARY_ESCALATION_OBSERVATION_SCHEMA
    assert UnnecessaryEscalationObservation.from_dict(payload).content_id == (
        observation.content_id
    )

    # agree whether the caller started from decide_model_route or apply_escalation_policy
    via_decide = decide_model_route(
        _facts(),
        available_models=default_inventory(),
        policy=_policy(),
    )
    again = detect_unnecessary_escalation(via_decide, facts=_facts(), policy=_policy())
    assert again.to_dict() == observation.to_dict()


def test_failure_escalation_is_necessary_unless_lower_route_succeeded() -> None:
    escalated = _decide(
        _facts(),
        prior=[PriorRepairAttempt(route=ModelRoute.SMALL_LOCAL_MODEL, failed=True)],
    )
    assert escalated.route is ModelRoute.MEDIUM_MODEL
    assert REASON_SMALLER_ROUTE_FAILED in escalated.decisive_reason_codes
    explanation = explain_model_route(escalated)
    assert explanation.escalated is True

    necessary = detect_unnecessary_escalation(
        explanation,
        facts=_facts(),
        policy=_policy(),
    )
    assert necessary.unnecessary is False
    assert necessary.escalated is True
    assert REASON_JUSTIFIED_FAILURE_ESCALATION in necessary.reason_codes
    assert necessary.rank_delta > 0

    wasted = detect_unnecessary_escalation(
        escalated,
        facts=_facts(),
        policy=_policy(),
        lower_route_succeeded=True,
    )
    assert wasted.unnecessary is True
    assert REASON_LOWER_ROUTE_SUCCEEDED in wasted.reason_codes
    assert wasted.decision_id == necessary.decision_id
    assert wasted.content_id != necessary.content_id


def test_over_routing_without_failure_marker_is_unnecessary() -> None:
    over_routed = ModelRouteDecision(
        route=ModelRoute.FRONTIER_MODEL,
        considered_routes=(
            ModelRoute.DETERMINISTIC_ONLY,
            ModelRoute.SMALL_LOCAL_MODEL,
            ModelRoute.MEDIUM_MODEL,
            ModelRoute.FRONTIER_MODEL,
            ModelRoute.HUMAN_REVIEW_REQUIRED,
        ),
        decisive_reason_codes=("ambiguous_work",),
        required_capabilities=("bounded_context", "frontier_reasoning"),
        context_token_estimate=2_048,
        policy_cid=policy_cid_for("doep-076"),
    )
    observation = detect_unnecessary_escalation(
        over_routed,
        facts=_facts(),
        policy=_policy(),
    )
    assert observation.unnecessary is True
    assert REASON_OBSERVED_ABOVE_JUSTIFIED_WITHOUT_FAILURE in observation.reason_codes
    assert observation.observed_route == ModelRoute.FRONTIER_MODEL.value
    assert observation.justified_route == ModelRoute.SMALL_LOCAL_MODEL.value
    assert REASON_SMALLER_ROUTE_FAILED not in observation.decisive_reason_codes

    skipped = ModelRouteDecision(
        route=ModelRoute.SMALL_LOCAL_MODEL,
        considered_routes=(
            ModelRoute.DETERMINISTIC_ONLY,
            ModelRoute.SMALL_LOCAL_MODEL,
            ModelRoute.MEDIUM_MODEL,
            ModelRoute.FRONTIER_MODEL,
            ModelRoute.HUMAN_REVIEW_REQUIRED,
        ),
        decisive_reason_codes=("localized_exact_counterexample",),
        required_capabilities=("bounded_context", "local_execution"),
        context_token_estimate=2_048,
        policy_cid=policy_cid_for("doep-076"),
    )
    mechanical = _facts(
        analysis_kind=AnalysisKind.MECHANICAL_IMPORT,
        counterexample_quality=CounterexampleQuality.NONE,
    )
    skipped_obs = detect_unnecessary_escalation(
        skipped,
        facts=mechanical,
        policy=_policy(),
    )
    assert skipped_obs.unnecessary is True
    assert REASON_SKIPPED_DETERMINISTIC_ELIGIBILITY in skipped_obs.reason_codes


def test_unavailable_required_tier_is_excused_not_unnecessary() -> None:
    unavailable = _decide(
        _facts(analysis_kind=AnalysisKind.AMBIGUOUS),
        inventory=default_inventory(small=True, medium=False, frontier=False),
    )
    assert unavailable.route is ModelRoute.HUMAN_REVIEW_REQUIRED
    assert REASON_REQUIRED_TIER_UNAVAILABLE in unavailable.decisive_reason_codes
    observation = detect_unnecessary_escalation(
        unavailable,
        facts=_facts(analysis_kind=AnalysisKind.AMBIGUOUS),
        available_models=default_inventory(small=True, medium=False, frontier=False),
        policy=_policy(),
    )
    assert observation.unnecessary is False
    assert REASON_REQUIRED_TIER_UNAVAILABLE_EXCUSED in observation.reason_codes


def test_route_explanation_receipt_binding_and_aggregate_report() -> None:
    localized = _decide(_facts())
    receipt = emit_route_explanation_receipt(localized)
    observation = detect_unnecessary_escalation(
        dict(receipt),
        justified_route=ModelRoute.SMALL_LOCAL_MODEL,
        route_explanation_receipt_id=str(receipt["receipt_id"]),
    )
    assert observation.unnecessary is False
    assert observation.route_explanation_receipt_id == receipt["receipt_id"]

    over_routed = ModelRouteDecision(
        route=ModelRoute.FRONTIER_MODEL,
        considered_routes=(
            ModelRoute.SMALL_LOCAL_MODEL,
            ModelRoute.MEDIUM_MODEL,
            ModelRoute.FRONTIER_MODEL,
        ),
        decisive_reason_codes=("ambiguous_work",),
        required_capabilities=("bounded_context", "frontier_reasoning"),
        context_token_estimate=2_048,
        policy_cid=policy_cid_for("doep-076"),
    )
    unnecessary = detect_unnecessary_escalation(
        over_routed,
        justified_route=ModelRoute.SMALL_LOCAL_MODEL,
    )
    report = aggregate_unnecessary_escalation_observations(
        [observation, unnecessary]
    )
    assert report.observation_count == 2
    assert report.unnecessary_count == 1
    assert report.unnecessary_ratio.numerator == 1
    assert report.unnecessary_ratio.denominator == 2
    assert report.to_dict()["schema"] == UNNECESSARY_ESCALATION_REPORT_SCHEMA
    assert report.to_dict()["authoritative"] is False
    assert UnnecessaryEscalationReport.from_dict(report.to_dict()).content_id == (
        report.content_id
    )


def test_competing_routing_objects_and_authoritative_claims_fail_closed() -> None:
    class RoutingDecision:
        pass

    with pytest.raises(EfficiencyValidationError, match="competing routers"):
        detect_unnecessary_escalation(
            RoutingDecision(),
            justified_route=ModelRoute.SMALL_LOCAL_MODEL,
        )

    localized = _decide(_facts())
    observation = detect_unnecessary_escalation(
        localized,
        justified_route=ModelRoute.SMALL_LOCAL_MODEL,
    )
    payload = observation.to_dict()
    payload["authoritative"] = True
    with pytest.raises(EfficiencyValidationError, match="never authoritative"):
        UnnecessaryEscalationObservation.from_dict(payload)

    with pytest.raises(EfficiencyValidationError, match="facts or justified_route"):
        detect_unnecessary_escalation(localized)


def test_manifest_and_candidate_receipt_bind_current_tree_evidence() -> None:
    manifest = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    for payload, schema in (
        (manifest, "ipfs_accelerate_py/agent-supervisor/doep-task-output@1"),
        (receipt, "ipfs_accelerate_py/agent-supervisor/doep-task-receipt@1"),
    ):
        assert payload["schema"] == schema
        assert payload["task_id"] == "DOEP-076"
        assert payload["task_cid"] == TASK_CID
        assert payload["plan_cid"] == PLAN_CID
        assert payload["completion_authoritative"] is False
        assert payload["worker_completion_insufficient"] is True
        assert payload["no_competing_subsystem_created"] is True
    assert manifest["primary_output"] == OWNER_RELATIVE_OUTPUTS[0]
    assert manifest["declared_outputs"] == list(OWNER_RELATIVE_OUTPUTS)
    assert manifest["base_repositories"] == BASE_REPOSITORIES
    assert manifest["canonical_extension"]["module"].endswith(
        "self_improvement.supervisor_efficiency_metrics"
    )
    assert manifest["canonical_extension"]["entrypoint"] == (
        "detect_unnecessary_escalation"
    )
    assert manifest["canonical_extension"]["carrier"] == (
        "UnnecessaryEscalationObservation"
    )
    assert receipt["changed_paths"] == list(OWNER_RELATIVE_OUTPUTS)
    assert receipt["outputs_present"] == {path: True for path in OWNER_RELATIVE_OUTPUTS}
    assert receipt["path_digests"] == {
        OWNER_RELATIVE_OUTPUTS[0]: _sha256_file(METRICS_PATH),
        OWNER_RELATIVE_OUTPUTS[1]: _sha256_file(TEST_PATH),
        OWNER_RELATIVE_OUTPUTS[2]: _sha256_file(OUTPUT_PATH),
    }
    assert receipt["required_evidence"]["verifier_admission"] == (
        "pending_independent_fenced_supervisor"
    )
