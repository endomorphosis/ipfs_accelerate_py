from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest
from ipfs_accelerate_py.proof_context.benchmarks.configurations_ab import (
    METRIC_NAMES,
    PairIdentity,
    SemanticContextPack,
    TaskAgentView,
    estimate_context_tokens,
)
from ipfs_accelerate_py.proof_context.benchmarks.configurations_cd import (
    C_STAGES,
    CONFIGURATION_C_CID,
    CONFIGURATION_D_CID,
    RUNNER_DESCRIPTOR_CID,
    AssuranceDecision,
    CDExecutionObservation,
    CDExecutionRequest,
    ConfigurationCDError,
    ContextDecision,
    DispositionDecision,
    ExecutionPortUnavailable,
    HiddenDataDenied,
    RouteDecision,
    RouteExecutionPermit,
    SealDecision,
    StageEvidence,
    VerificationDecision,
    configuration_cid,
    configuration_descriptor,
    run_configuration,
    run_paired_cd,
    runner_descriptor,
)
from ipfs_accelerate_py.proof_context.lifecycle import STAGES as GOVERNED_LIFECYCLE_STAGES
from ipfs_accelerate_py.utils.cid_utils import cid_for_bytes, cid_for_obj


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode(), codec="raw")


def _identity(**changes: Any) -> PairIdentity:
    values: dict[str, Any] = {
        "corpus_manifest_cid": _cid("corpus"),
        "task_record_cid": _cid("task"),
        "visible_projection_cid": _cid("visible"),
        "repository_state_cid": _cid("repository"),
        "environment_cid": _cid("environment"),
        "task_id": "typed-001",
        "provider_id": "provider/local",
        "model_id": "routed-model",
        "model_revision": "routed-model@immutable-2026-08-24",
        "seed": 60060,
        "attempt": 1,
    }
    values.update(changes)
    return PairIdentity(**values)


def _task(**changes: Any) -> TaskAgentView:
    values: dict[str, Any] = {
        "objective": "Correct the visible converter without broadening its public API.",
        "owned_paths": ("src/converter.py", "tests/test_converter.py"),
        "routine_localized": True,
        "risk_class": "routine",
    }
    values.update(changes)
    return TaskAgentView(**values)


def _pack(**changes: Any) -> SemanticContextPack:
    rendered = "converter signature, public tests, and integer-to-string contract"
    values: dict[str, Any] = {
        "pack_cid": _cid("context-pack"),
        "visible_projection_cid": _cid("visible"),
        "rendered_context": rendered,
        "declared_tokens": estimate_context_tokens(rendered),
        "exact_source_tokens": 5,
        "capsule_tokens": 4,
        "fallback_count": 1,
    }
    values.update(changes)
    return SemanticContextPack(**values)


def _permit(configuration_id: str, **changes: Any) -> RouteExecutionPermit:
    values: dict[str, Any] = {
        "permit_cid": _cid(f"permit-{configuration_id}"),
        "route_policy_cid": _cid("route-policy"),
        "configuration_cid": configuration_cid(configuration_id),
        "corpus_manifest_cid": _cid("corpus"),
        "task_record_cid": _cid("task"),
        "visible_projection_cid": _cid("visible"),
        "repository_state_cid": _cid("repository"),
        "environment_cid": _cid("environment"),
        "provider_id": "provider/local",
        "model_id": "routed-model",
        "model_revision": "routed-model@immutable-2026-08-24",
        "task_id": "typed-001",
        "seed": 60060,
        "attempt": 1,
        "provenance": "replayed",
        "available": True,
        "live_execution_eligible": False,
        "reason": "reviewed-deterministic-replay-fixture",
    }
    values.update(changes)
    return RouteExecutionPermit(**values)


def _context_decision(configuration_id: str, **changes: Any) -> ContextDecision:
    values: dict[str, Any] = {
        "status": "succeeded",
        "provenance": "replayed",
        "evidence_cid": _cid(f"context-evidence-{configuration_id}"),
        "pack_cid": _cid("context-pack"),
        "visible_projection_cid": _cid("visible"),
        "initial_sufficient": None if configuration_id == "C" else False,
        "sufficient_after_expansion": None if configuration_id == "C" else True,
        "fallback_count": 1,
        "expansion_count": 0 if configuration_id == "C" else 1,
        "expansion_tokens": 0 if configuration_id == "C" else 8,
    }
    values.update(changes)
    return ContextDecision(**values)


def _route(**changes: Any) -> RouteDecision:
    values: dict[str, Any] = {
        "status": "succeeded",
        "provenance": "replayed",
        "evidence_cid": _cid("route-evidence"),
        "route_policy_cid": _cid("route-policy"),
        "route": "local",
        "decision_kind": "selected",
        "provider_id": "provider/local",
        "model_id": "routed-model",
        "model_revision": "routed-model@immutable-2026-08-24",
        "previous_route": None,
        "reason": "frozen-route-policy-selected-exact-local-model",
    }
    values.update(changes)
    return RouteDecision(**values)


def _verification(configuration_id: str, **changes: Any) -> VerificationDecision:
    is_c = configuration_id == "C"
    values: dict[str, Any] = {
        "status": "succeeded",
        "provenance": "replayed",
        "evidence_cid": _cid(f"verification-evidence-{configuration_id}"),
        "proposal_cid": _cid("proposal"),
        "verification_plan_cid": _cid(f"verification-plan-{configuration_id}"),
        "reuse_outcome": "hit" if is_c else "miss",
        "reuse_receipt_cid": _cid("reuse-receipt") if is_c else None,
        "selected_test_count": 4,
        "selected_test_pass_count": 4,
        "proof_selected_count": 1,
        "proof_executed_count": 0 if is_c else 1,
        "proof_pass_count": 0 if is_c else 1,
        "proof_fail_count": 0,
        "full_fallback_used": False,
        "hidden_scoring_after_proposal": True,
        "full_test_count": 12,
        "full_test_pass_count": 12,
        "hidden_test_total_count": 4,
        "hidden_test_pass_count": 4,
        "regression_count": 0,
        "critical_regression_count": 0,
        "out_of_scope_edit_count": 0,
        "semantic_outcome_match": True,
        "verification_cost_micros": 50,
        "proof_cost_micros": 2 if is_c else 10,
    }
    values.update(changes)
    return VerificationDecision(**values)


def _assurance(**changes: Any) -> AssuranceDecision:
    values: dict[str, Any] = {
        "status": "succeeded",
        "provenance": "replayed",
        "evidence_cid": _cid("assurance-evidence"),
        "accepted": True,
        "self_approved": False,
        "hidden_benchmark_exposed": False,
        "mutant_count": 4,
        "mutant_detected_count": 4,
        "omission_mutant_count": 1,
        "omission_mutant_detected_count": 1,
        "vacuity_mutant_count": 1,
        "vacuity_mutant_detected_count": 1,
        "context_expansion_mutant_count": 1,
        "context_expansion_mutant_detected_count": 1,
        "critical_mutant_accepted_count": 0,
        "sample_count": 4,
        "failure_count": 0,
        "assurance_cost_micros": 70,
    }
    values.update(changes)
    return AssuranceDecision(**values)


def _seal(**changes: Any) -> SealDecision:
    values: dict[str, Any] = {
        "status": "succeeded",
        "provenance": "replayed",
        "evidence_cid": _cid("seal-evidence"),
        "seal_cid": _cid("incremental-seal"),
        "parent_proposal_cid": _cid("proposal"),
        "incremental": True,
    }
    values.update(changes)
    return SealDecision(**values)


def _disposition(**changes: Any) -> DispositionDecision:
    values: dict[str, Any] = {
        "status": "succeeded",
        "provenance": "replayed",
        "evidence_cid": _cid("disposition-evidence"),
        "human_review_required": False,
        "human_review_performed": False,
        "human_review_correct": None,
        "autonomous_accept": False,
        "human_cost_micros": 0,
    }
    values.update(changes)
    return DispositionDecision(**values)


def _trace(
    configuration_id: str,
    *,
    provenance: str = "replayed",
    through: str | None = None,
    last_status: str = "succeeded",
    identity: PairIdentity | None = None,
) -> tuple[StageEvidence, ...]:
    stages = C_STAGES if configuration_id == "C" else GOVERNED_LIFECYCLE_STAGES
    if through is not None:
        stages = stages[: stages.index(through) + 1]
    bound_identity = identity or _identity()
    return tuple(
        StageEvidence(
            stage=stage,
            status=last_status if index == len(stages) - 1 else "succeeded",
            provenance=provenance,
            evidence_cid=_cid(f"{configuration_id}-{stage}-{provenance}"),
            configuration_cid=configuration_cid(configuration_id),
            identity=bound_identity,
        )
        for index, stage in enumerate(stages)
    )


def _observation(configuration_id: str, **changes: Any) -> CDExecutionObservation:
    is_c = configuration_id == "C"
    values: dict[str, Any] = {
        "configuration_id": configuration_id,
        "status": "succeeded",
        "provenance": "replayed",
        "context": _context_decision(configuration_id),
        "route": _route(),
        "verification": _verification(configuration_id),
        "assurance": None if is_c else _assurance(),
        "seal": None if is_c else _seal(),
        "disposition": None if is_c else _disposition(),
        "stage_trace": _trace(configuration_id),
        "provider_status": "succeeded",
        "provider_evidence_cid": _cid(f"provider-evidence-{configuration_id}"),
        "proposal_cid": _cid("proposal"),
        "provider_call_count": 1,
        "provider_input_tokens": 90,
        "provider_output_tokens": 20,
        "provider_cached_input_tokens": 0,
        "inference_cost_micros": 300,
        "failure_cost_micros": 0,
        "reason": "reviewed-deterministic-successful-observation",
    }
    values.update(changes)
    return CDExecutionObservation(**values)


class RecordingPort:
    def __init__(
        self,
        observation: CDExecutionObservation | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.observation = observation
        self.error = error
        self.requests: list[CDExecutionRequest] = []

    def execute(self, request: CDExecutionRequest) -> CDExecutionObservation:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        if self.observation is None:
            raise AssertionError("fixture requires an observation or error")
        return self.observation


def _run(configuration_id: str, observation: CDExecutionObservation | None = None):
    port = RecordingPort(observation or _observation(configuration_id))
    run = run_configuration(
        configuration_id=configuration_id,
        identity=_identity(),
        task=_task(),
        context=_pack(),
        permit=_permit(configuration_id),
        port=port,
    )
    return run, port


def test_frozen_descriptors_and_exact_pcce060_cids() -> None:
    assert CONFIGURATION_C_CID == "baguqeeraqrsu4psh7r6ehgwt5ebcqvjqckgyrwdgl2twxcku7ihknoxrzz5q"
    assert CONFIGURATION_D_CID == "baguqeeranxarzlueyd5lsmabqgke7bt7qvarxaevuazzces6gf7cozkdmxxq"
    assert configuration_cid("C") == cid_for_obj(configuration_descriptor("C"), codec="dag-json")
    assert configuration_cid("D") == cid_for_obj(configuration_descriptor("D"), codec="dag-json")


def test_only_frozen_c_to_d_governance_controls_change() -> None:
    arm_c = configuration_descriptor("C")
    arm_d = configuration_descriptor("D")
    changed = sorted(key for key in arm_c if key != "configuration_id" and arm_c[key] != arm_d[key])
    assert changed == [
        "assurance_enabled",
        "context_expansion_enabled",
        "human_escalation_enabled",
        "incremental_seal_enabled",
        "sufficiency_enabled",
    ]
    assert arm_c["routing_enabled"] is arm_d["routing_enabled"] is True
    assert arm_c["incremental_verification_enabled"] is True
    assert arm_c["proof_reuse_enabled"] is True


def test_descriptor_copies_and_observations_are_frozen() -> None:
    descriptor = configuration_descriptor("C")
    descriptor["routing_enabled"] = False
    assert configuration_descriptor("C")["routing_enabled"] is True
    observation = _observation("C")
    with pytest.raises(FrozenInstanceError):
        observation.status = "stale"


def test_module_is_provider_neutral_and_has_no_direct_io_or_network() -> None:
    path = Path("ipfs_accelerate_py/proof_context/benchmarks/configurations_cd.py")
    source = path.read_text()
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not ({"requests", "urllib", "httpx", "socket", "subprocess"} & imports)
    assert "open(" not in source
    assert "provider/openai" not in source
    assert "provider/anthropic" not in source


def test_runner_descriptor_binds_policies_and_stages() -> None:
    descriptor = runner_descriptor()
    assert descriptor["identity_profile"] == "software-contract-cid-profile-v1"
    assert descriptor["configuration_c_stages"] == list(C_STAGES)
    assert descriptor["configuration_d_stages"] == list(GOVERNED_LIFECYCLE_STAGES)
    assert RUNNER_DESCRIPTOR_CID == cid_for_obj(descriptor, codec="dag-json")


@pytest.mark.parametrize("configuration_id", ("C", "D"))
def test_pcce056_no_go_or_inexact_permit_never_dispatches(configuration_id: str) -> None:
    port = RecordingPort(_observation(configuration_id))
    run = run_configuration(
        configuration_id=configuration_id,
        identity=_identity(),
        task=_task(),
        context=_pack(),
        permit=_permit(
            configuration_id,
            provenance="live",
            available=False,
            live_execution_eligible=False,
            reason="pcce-056-no-go",
        ),
        port=port,
    )
    assert run.raw_result["terminal_status"] == "unavailable"
    assert run.raw_result["provenance"] == "live"
    assert run.raw_result["metrics"]["provider_call_count"] == 0
    assert run.raw_result["metrics"]["inference_cost_micros"] is None
    assert not port.requests


def test_permit_binds_task_seed_and_attempt_before_dispatch() -> None:
    for change in ({"task_id": "other-task"}, {"seed": 60061}, {"attempt": 2}):
        port = RecordingPort(_observation("C"))
        run = run_configuration(
            configuration_id="C",
            identity=_identity(),
            task=_task(),
            context=_pack(),
            permit=_permit("C", **change),
            port=port,
        )
        assert run.raw_result["terminal_status"] == "unavailable"
        assert run.raw_result["metrics"]["provider_call_count"] == 0
        assert not port.requests


def test_wrong_visible_projection_is_rejected_before_port_dispatch() -> None:
    port = RecordingPort(_observation("C"))
    with pytest.raises(HiddenDataDenied, match="different visible projection"):
        run_configuration(
            configuration_id="C",
            identity=_identity(),
            task=_task(),
            context=_pack(visible_projection_cid=_cid("other-visible")),
            permit=_permit("C"),
            port=port,
        )
    assert not port.requests


@pytest.mark.parametrize("configuration_id", ("C", "D"))
def test_forged_hidden_context_is_revalidated_before_dispatch(configuration_id: str) -> None:
    context = _pack()
    hidden = "hidden-tests/test_secret.py\nSECRET_HIDDEN_FIXTURE = True\n"
    object.__setattr__(context, "rendered_context", hidden)
    object.__setattr__(context, "declared_tokens", estimate_context_tokens(hidden))
    port = RecordingPort(_observation(configuration_id))
    with pytest.raises(ValueError, match="hidden or evaluator namespace"):
        run_configuration(
            configuration_id=configuration_id,
            identity=_identity(),
            task=_task(),
            context=context,
            permit=_permit(configuration_id),
            port=port,
        )
    assert not port.requests


def test_configuration_c_records_route_reuse_incremental_verification_and_cost() -> None:
    run, port = _run("C")
    raw = run.raw_result
    assert raw["terminal_status"] == "succeeded"
    assert raw["provenance"] == "replayed"
    assert raw["metrics"]["route_local_count"] == 1
    assert raw["metrics"]["provider_call_count"] == 1
    assert raw["metrics"]["selected_test_count"] == 4
    assert raw["metrics"]["verification_reuse_hit_count"] == 1
    assert raw["metrics"]["proof_executed_count"] == 0
    assert raw["metrics"]["verification_full_fallback_count"] == 0
    assert raw["metrics"]["accepted_patch_count"] == 1
    assert raw["metrics"]["total_cost_micros"] == 352
    assert run.audit["route"] == "local"
    assert run.audit["reuse_outcome"] == "hit"
    assert run.audit["full_fallback_used"] is False
    assert len(port.requests) == 1


def test_configuration_c_miss_executes_fresh_proof() -> None:
    verification = _verification(
        "C",
        reuse_outcome="miss",
        reuse_receipt_cid=None,
        proof_executed_count=1,
        proof_pass_count=1,
        proof_cost_micros=10,
    )
    run, _ = _run("C", _observation("C", verification=verification))
    assert run.raw_result["terminal_status"] == "succeeded"
    assert run.raw_result["metrics"]["verification_reuse_miss_count"] == 1
    assert run.raw_result["metrics"]["proof_executed_count"] == 1


def test_full_verification_fallback_is_never_silent() -> None:
    verification = replace(
        _verification("C"),
        reuse_outcome="miss",
        reuse_receipt_cid=None,
        selected_test_count=0,
        selected_test_pass_count=0,
        proof_selected_count=0,
        proof_executed_count=0,
        proof_pass_count=0,
        full_fallback_used=True,
        proof_cost_micros=0,
    )
    run, _ = _run("C", _observation("C", verification=verification))
    assert run.raw_result["terminal_status"] == "succeeded"
    assert run.raw_result["metrics"]["verification_full_fallback_count"] == 1
    assert run.audit["full_fallback_used"] is True


def test_zero_incremental_selection_without_explicit_fallback_is_invalid() -> None:
    verification = replace(
        _verification("C"),
        reuse_outcome="miss",
        reuse_receipt_cid=None,
        selected_test_count=0,
        selected_test_pass_count=0,
        proof_selected_count=0,
        proof_executed_count=0,
        proof_pass_count=0,
        proof_cost_micros=0,
    )
    run, _ = _run("C", _observation("C", verification=verification))
    assert run.raw_result["terminal_status"] == "invalid"
    assert run.audit["defect"] == "successful-run-silently-omits-incremental-verification"
    assert run.audit["accepted"] is False


def test_reuse_hit_requires_selected_nonexecuted_proof_work() -> None:
    with pytest.raises(ConfigurationCDError, match="selected proof not freshly executed"):
        replace(
            _verification("C"),
            proof_selected_count=0,
            proof_executed_count=0,
            proof_pass_count=0,
        )


@pytest.mark.parametrize(
    ("decision_kind", "previous_route", "expected_escalation"),
    (("selected", None, 1), ("escalated", "local", 1)),
)
def test_frontier_route_is_explicit_never_a_silent_fallback(
    decision_kind: str,
    previous_route: str | None,
    expected_escalation: int,
) -> None:
    identity = _identity(
        provider_id="provider/frontier",
        model_id="frontier-model",
        model_revision="frontier-model@immutable-2026-08-24",
    )
    route = _route(
        route="frontier",
        decision_kind=decision_kind,
        previous_route=previous_route,
        provider_id=identity.provider_id,
        model_id=identity.model_id,
        model_revision=identity.model_revision,
    )
    observation = _observation(
        "C",
        route=route,
        stage_trace=_trace("C", identity=identity),
    )
    permit = _permit(
        "C",
        provider_id=identity.provider_id,
        model_id=identity.model_id,
        model_revision=identity.model_revision,
    )
    run = run_configuration(
        configuration_id="C",
        identity=identity,
        task=_task(),
        context=_pack(),
        permit=permit,
        port=RecordingPort(observation),
    )
    assert run.raw_result["terminal_status"] == "succeeded"
    assert run.raw_result["metrics"]["route_frontier_count"] == 1
    assert run.raw_result["metrics"]["frontier_escalation_count"] == expected_escalation
    assert run.audit["route_decision_kind"] == decision_kind


def test_unavailable_route_never_calls_provider_or_claims_cost() -> None:
    observation = _observation(
        "C",
        status="unavailable",
        route=_route(
            status="unavailable",
            route="unavailable",
            decision_kind="unavailable",
            reason="no-eligible-exact-route",
        ),
        verification=None,
        stage_trace=_trace("C", through="route", last_status="unavailable"),
        provider_status=None,
        provider_evidence_cid=None,
        proposal_cid=None,
        provider_call_count=0,
        provider_input_tokens=None,
        provider_output_tokens=None,
        provider_cached_input_tokens=None,
        inference_cost_micros=None,
        failure_cost_micros=None,
    )
    run, _ = _run("C", observation)
    assert run.raw_result["terminal_status"] == "unavailable"
    assert run.raw_result["metrics"]["route_unavailable_count"] == 1
    assert run.raw_result["metrics"]["provider_call_count"] == 0
    assert run.raw_result["metrics"]["total_cost_micros"] is None


def test_simulated_execution_is_forced_to_simulated_and_never_accepted() -> None:
    observation = _observation(
        "C",
        provenance="simulated",
        context=replace(_context_decision("C"), provenance="simulated"),
        route=replace(_route(), provenance="simulated"),
        verification=replace(_verification("C"), provenance="simulated"),
        stage_trace=_trace("C", provenance="simulated"),
    )
    run = run_configuration(
        configuration_id="C",
        identity=_identity(),
        task=_task(),
        context=_pack(),
        permit=_permit("C", provenance="simulated"),
        port=RecordingPort(observation),
    )
    assert run.raw_result["terminal_status"] == "simulated"
    assert run.raw_result["provenance"] == "simulated"
    assert run.raw_result["metrics"]["accepted_patch_count"] == 0
    assert run.raw_result["metrics"]["simulated_success_accepted_count"] == 0
    assert run.audit["accepted"] is False


def test_replayed_authority_cannot_be_relabelled_live() -> None:
    observation = _observation(
        "C",
        provenance="live",
        context=replace(_context_decision("C"), provenance="live"),
        route=replace(_route(), provenance="live"),
        verification=replace(_verification("C"), provenance="live"),
        stage_trace=_trace("C", provenance="live"),
    )
    run, _ = _run("C", observation)
    assert run.raw_result["terminal_status"] == "invalid"
    assert run.raw_result["provenance"] == "replayed"
    assert run.audit["defect"] == "observation-permit-provenance-mismatch"
    assert run.audit["accepted"] is False


def test_nested_simulation_watermarks_the_entire_result() -> None:
    observation = _observation(
        "C",
        verification=replace(_verification("C"), provenance="simulated"),
    )
    run, _ = _run("C", observation)
    assert run.raw_result["terminal_status"] == "simulated"
    assert run.raw_result["provenance"] == "simulated"
    assert run.raw_result["metrics"]["simulated_success_accepted_count"] == 0
    assert run.audit["accepted"] is False


def test_stale_context_is_rejected_and_never_routed() -> None:
    observation = _observation(
        "C",
        status="stale",
        context=replace(_context_decision("C"), status="stale"),
        route=None,
        verification=None,
        stage_trace=_trace("C", through="context-pack", last_status="stale"),
        provider_status=None,
        provider_evidence_cid=None,
        proposal_cid=None,
        provider_call_count=0,
        provider_input_tokens=None,
        provider_output_tokens=None,
        provider_cached_input_tokens=None,
        inference_cost_micros=None,
        failure_cost_micros=None,
    )
    run, _ = _run("C", observation)
    assert run.raw_result["terminal_status"] == "stale"
    assert run.raw_result["metrics"]["stale_capsule_rejected_count"] == 1
    assert run.raw_result["metrics"]["stale_capsule_accepted_count"] == 0
    assert run.raw_result["metrics"]["provider_call_count"] == 0


def test_stale_reuse_is_rejected_while_provider_cost_remains_visible() -> None:
    verification = _verification(
        "C",
        status="stale",
        reuse_outcome="stale",
        reuse_receipt_cid=None,
    )
    observation = _observation(
        "C",
        status="stale",
        verification=verification,
        stage_trace=_trace("C", through="incremental-verify", last_status="stale"),
    )
    run, _ = _run("C", observation)
    assert run.raw_result["terminal_status"] == "stale"
    assert run.raw_result["metrics"]["stale_proof_rejected_count"] == 1
    assert run.raw_result["metrics"]["stale_proof_accepted_count"] == 0
    assert run.raw_result["metrics"]["inference_cost_micros"] == 300
    assert run.raw_result["metrics"]["accepted_patch_count"] == 0


@pytest.mark.parametrize(
    ("error", "terminal"),
    (
        (ExecutionPortUnavailable("PCCE-067 execution authority unavailable"), "unavailable"),
        (TimeoutError("governed port timed out"), "infrastructure_failure"),
    ),
)
def test_port_exceptions_return_raw_rows_without_false_cost_zero(
    error: Exception,
    terminal: str,
) -> None:
    port = RecordingPort(error=error)
    run = run_configuration(
        configuration_id="D",
        identity=_identity(),
        task=_task(),
        context=_pack(),
        permit=_permit("D"),
        port=port,
    )
    assert run.raw_result["terminal_status"] == terminal
    assert run.raw_result["metrics"]["provider_call_count"] is None
    assert run.raw_result["metrics"]["inference_cost_micros"] is None
    assert run.raw_result["metrics"]["total_cost_micros"] is None
    assert "total_cost_micros" in run.raw_result["missingness"]
    assert len(port.requests) == 1


def test_stage_identity_forgery_returns_invalid_row_and_retains_observed_cost() -> None:
    forged = _trace("C", identity=_identity(environment_cid=_cid("wrong-environment")))
    run, _ = _run("C", _observation("C", stage_trace=forged))
    assert run.raw_result["terminal_status"] == "invalid"
    assert run.audit["defect"] == "lifecycle-stage-identity-mismatch"
    assert run.raw_result["metrics"]["inference_cost_micros"] == 300


def test_mutated_observation_is_revalidated_and_cannot_inject_metrics() -> None:
    observation = _observation("C")

    class MutatingPort:
        def execute(self, request: CDExecutionRequest) -> CDExecutionObservation:
            object.__setattr__(observation, "provider_call_count", -1)
            return observation

    run = run_configuration(
        configuration_id="C",
        identity=_identity(),
        task=_task(),
        context=_pack(),
        permit=_permit("C"),
        port=MutatingPort(),
    )
    assert run.raw_result["terminal_status"] == "invalid"
    assert run.raw_result["metrics"]["provider_call_count"] is None
    assert run.raw_result["metrics"]["total_cost_micros"] is None
    assert run.audit["accepted"] is False


def test_reordered_stage_trace_is_invalid_never_partially_admitted() -> None:
    stages = list(_trace("D"))
    stages[4], stages[5] = stages[5], stages[4]
    run, _ = _run("D", _observation("D", stage_trace=tuple(stages)))
    assert run.raw_result["terminal_status"] == "invalid"
    assert run.audit["accepted"] is False
    assert run.audit["defect"] == "lifecycle-stage-order-or-prefix-invalid"


def test_provider_failure_retains_complete_observed_failed_attempt_cost() -> None:
    observation = _observation(
        "C",
        status="timeout",
        verification=None,
        stage_trace=_trace("C", through="proposal", last_status="timeout"),
        provider_status="timeout",
        proposal_cid=None,
        provider_input_tokens=80,
        provider_output_tokens=0,
        inference_cost_micros=200,
        failure_cost_micros=35,
    )
    run, _ = _run("C", observation)
    assert run.raw_result["terminal_status"] == "timeout"
    assert run.raw_result["metrics"]["total_cost_micros"] == 235
    assert run.raw_result["metrics"]["failed_attempt_cost_micros"] == 235


def test_configuration_d_enforces_complete_governed_lifecycle_and_measurements() -> None:
    run, port = _run("D")
    raw = run.raw_result
    assert raw["terminal_status"] == "succeeded"
    assert tuple(stage["stage"] for stage in run.audit["lifecycle_trace"]) == (
        GOVERNED_LIFECYCLE_STAGES
    )
    assert raw["metrics"]["context_expansion_count"] == 1
    assert raw["metrics"]["context_expansion_tokens"] == 8
    assert raw["metrics"]["verification_reuse_miss_count"] == 1
    assert raw["metrics"]["assurance_mutant_count"] == 4
    assert raw["metrics"]["assurance_mutant_survivor_count"] == 0
    assert raw["metrics"]["critical_mutant_accepted_count"] == 0
    assert raw["metrics"]["accepted_patch_count"] == 1
    assert raw["metrics"]["total_cost_micros"] == 430
    assert len(port.requests) == 1


def test_configuration_d_does_not_expand_already_sufficient_context() -> None:
    context = _context_decision(
        "D",
        initial_sufficient=True,
        sufficient_after_expansion=True,
        expansion_count=0,
        expansion_tokens=0,
    )
    run, _ = _run("D", _observation("D", context=context))
    assert run.raw_result["terminal_status"] == "succeeded"
    assert run.raw_result["metrics"]["context_expansion_count"] == 0


def test_configuration_d_missing_lifecycle_stage_is_invalid() -> None:
    run, _ = _run("D", _observation("D", stage_trace=_trace("D")[:-1]))
    assert run.raw_result["terminal_status"] == "invalid"
    assert run.audit["defect"] == "successful-observation-omits-required-lifecycle-stage"


def test_configuration_d_insufficient_context_without_expansion_stops_before_route() -> None:
    context = _context_decision(
        "D",
        status="context_insufficient",
        initial_sufficient=False,
        sufficient_after_expansion=False,
        expansion_count=0,
        expansion_tokens=0,
    )
    observation = _observation(
        "D",
        status="context_insufficient",
        context=context,
        route=None,
        verification=None,
        assurance=None,
        seal=None,
        disposition=None,
        stage_trace=_trace("D", through="sufficiency", last_status="context_insufficient"),
        provider_status=None,
        provider_evidence_cid=None,
        proposal_cid=None,
        provider_call_count=0,
        provider_input_tokens=None,
        provider_output_tokens=None,
        provider_cached_input_tokens=None,
        inference_cost_micros=None,
        failure_cost_micros=None,
    )
    run, _ = _run("D", observation)
    assert run.raw_result["terminal_status"] == "context_insufficient"
    assert run.raw_result["metrics"]["provider_call_count"] == 0
    assert run.audit["accepted"] is False


def test_configuration_d_assurance_failure_never_seals_or_accepts() -> None:
    assurance = _assurance(
        status="assurance_failed",
        accepted=False,
        mutant_detected_count=3,
        failure_count=1,
    )
    observation = _observation(
        "D",
        status="assurance_failed",
        assurance=assurance,
        seal=None,
        disposition=None,
        stage_trace=_trace("D", through="assurance", last_status="assurance_failed"),
    )
    run, _ = _run("D", observation)
    assert run.raw_result["terminal_status"] == "assurance_failed"
    assert run.raw_result["metrics"]["assurance_mutant_survivor_count"] == 1
    assert run.raw_result["metrics"]["assurance_failure_count"] == 1
    assert run.raw_result["metrics"]["accepted_patch_count"] == 0
    assert run.raw_result["metrics"]["total_cost_micros"] == 430
    assert run.raw_result["metrics"]["failed_attempt_cost_micros"] == 430
    assert run.audit["accepted"] is False


def test_configuration_d_zero_sample_assurance_cannot_be_accepted() -> None:
    run, _ = _run("D", _observation("D", assurance=replace(_assurance(), sample_count=0)))
    assert run.raw_result["terminal_status"] == "invalid"
    assert run.audit["defect"] == "configuration-d-requires-observed-assurance-sampling"
    assert run.audit["accepted"] is False


def test_stale_incremental_seal_is_never_accepted() -> None:
    observation = _observation(
        "D",
        status="stale",
        seal=replace(_seal(), status="stale", seal_cid=None),
        disposition=None,
        stage_trace=_trace("D", through="seal", last_status="stale"),
    )
    run, _ = _run("D", observation)
    assert run.raw_result["terminal_status"] == "stale"
    assert run.raw_result["metrics"]["accepted_patch_count"] == 0
    assert run.audit["accepted"] is False


def test_wrong_seal_parent_is_invalid_and_cannot_accept() -> None:
    observation = _observation(
        "D",
        seal=replace(_seal(), parent_proposal_cid=_cid("other-proposal")),
    )
    run, _ = _run("D", observation)
    assert run.raw_result["terminal_status"] == "invalid"
    assert run.audit["defect"] == "seal-parent-proposal-mismatch"
    assert run.audit["accepted"] is False


def test_self_approved_or_hidden_exposed_assurance_is_invalid() -> None:
    for assurance in (
        replace(_assurance(), self_approved=True),
        replace(_assurance(), hidden_benchmark_exposed=True),
    ):
        run, _ = _run("D", _observation("D", assurance=assurance))
        assert run.raw_result["terminal_status"] == "invalid"
        assert run.audit["defect"] == "assurance-is-self-approved-or-hidden-exposed"
        assert run.audit["accepted"] is False


def test_required_human_review_remains_visible_and_unaccepted_until_performed() -> None:
    disposition = _disposition(
        status="human_review_required",
        human_review_required=True,
        human_review_performed=False,
        human_review_correct=None,
        human_cost_micros=None,
    )
    observation = _observation(
        "D",
        status="human_review_required",
        disposition=disposition,
        stage_trace=_trace("D", last_status="human_review_required"),
    )
    run, _ = _run("D", observation)
    assert run.raw_result["terminal_status"] == "human_review_required"
    assert run.raw_result["metrics"]["human_review_required_count"] == 1
    assert run.raw_result["metrics"]["human_cost_micros"] is None
    assert run.audit["accepted"] is False


def test_human_review_case_cannot_be_autonomously_accepted() -> None:
    disposition = _disposition(
        human_review_required=True,
        human_review_performed=True,
        human_review_correct=True,
        autonomous_accept=True,
        human_cost_micros=40,
    )
    run, _ = _run("D", _observation("D", disposition=disposition))
    assert run.raw_result["terminal_status"] == "invalid"
    assert run.raw_result["metrics"]["negative_review_autonomous_accept_count"] == 1
    assert run.audit["accepted"] is False


def test_incorrect_human_review_cannot_accept_a_patch() -> None:
    disposition = _disposition(
        human_review_required=True,
        human_review_performed=True,
        human_review_correct=False,
        human_cost_micros=40,
    )
    run, _ = _run("D", _observation("D", disposition=disposition))
    assert run.raw_result["terminal_status"] == "invalid"
    assert run.raw_result["metrics"]["human_review_correct_count"] == 0
    assert run.audit["defect"] == "configuration-d-cannot-accept-an-incorrect-human-review"
    assert run.audit["accepted"] is False


def test_configuration_c_rejects_d_only_governance_observation() -> None:
    run, _ = _run("C", _observation("C", assurance=_assurance()))
    assert run.raw_result["terminal_status"] == "invalid"
    assert run.audit["defect"] == "configuration-c-observed-d-only-governance"


def test_pairing_rejects_identity_drift_before_either_port() -> None:
    port_c = RecordingPort(_observation("C"))
    port_d = RecordingPort(_observation("D"))
    with pytest.raises(ConfigurationCDError, match="identities differ"):
        run_paired_cd(
            identity_c=_identity(),
            identity_d=_identity(seed=60061),
            task_c=_task(),
            task_d=_task(),
            context_c=_pack(),
            context_d=_pack(),
            permit_c=_permit("C"),
            permit_d=_permit("D"),
            port_c=port_c,
            port_d=port_d,
        )
    assert not port_c.requests
    assert not port_d.requests


def test_pairing_rejects_swapped_configuration_permits_before_either_port() -> None:
    port_c = RecordingPort(_observation("C"))
    port_d = RecordingPort(_observation("D"))
    with pytest.raises(ConfigurationCDError, match="exact C/D configurations"):
        run_paired_cd(
            identity_c=_identity(),
            identity_d=_identity(),
            task_c=_task(),
            task_d=_task(),
            context_c=_pack(),
            context_d=_pack(),
            permit_c=replace(_permit("C"), configuration_cid=CONFIGURATION_D_CID),
            permit_d=replace(_permit("D"), configuration_cid=CONFIGURATION_C_CID),
            port_c=port_c,
            port_d=port_d,
        )
    assert not port_c.requests
    assert not port_d.requests


def test_paired_cd_run_is_deterministic_and_treatment_bound() -> None:
    kwargs = {
        "identity_c": _identity(),
        "identity_d": _identity(),
        "task_c": _task(),
        "task_d": _task(),
        "context_c": _pack(),
        "context_d": _pack(),
        "permit_c": _permit("C"),
        "permit_d": _permit("D"),
    }
    first = run_paired_cd(
        **kwargs,
        port_c=RecordingPort(_observation("C")),
        port_d=RecordingPort(_observation("D")),
    )
    second = run_paired_cd(
        **kwargs,
        port_c=RecordingPort(_observation("C")),
        port_d=RecordingPort(_observation("D")),
    )
    assert first.pairing_cid == second.pairing_cid
    assert first.arm_c.result_cid == second.arm_c.result_cid
    assert first.arm_d.result_cid == second.arm_d.result_cid
    assert first.pairing_record["only_treatment_differences"] == (
        "assurance_enabled",
        "context_expansion_enabled",
        "human_escalation_enabled",
        "incremental_seal_enabled",
        "sufficiency_enabled",
    )


@pytest.mark.parametrize("configuration_id", ("C", "D"))
def test_raw_results_have_exact_78_metrics_and_missingness(configuration_id: str) -> None:
    run, _ = _run(configuration_id)
    raw = run.raw_result
    assert tuple(raw["metrics"]) == METRIC_NAMES
    assert len(raw["metrics"]) == 78
    assert set(raw["missingness"]) == {
        name for name, value in raw["metrics"].items() if value is None
    }
    assert raw["configuration_cid"] == configuration_cid(configuration_id)
    assert run.result_cid == cid_for_obj(run.as_dict(), codec="dag-json")


def test_unknown_configuration_and_malformed_route_fail_closed() -> None:
    with pytest.raises(ConfigurationCDError, match="only frozen configurations"):
        configuration_descriptor("E")
    with pytest.raises(ConfigurationCDError, match="explicit escalation"):
        _route(decision_kind="escalated", previous_route=None)
    with pytest.raises(ConfigurationCDError, match="human decisions"):
        _route(route="local", decision_kind="human")
    with pytest.raises(ConfigurationCDError, match="unavailable decisions"):
        _route(route="local", decision_kind="unavailable")
