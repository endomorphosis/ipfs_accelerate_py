from __future__ import annotations

from dataclasses import replace
import json
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.prompt_workflow import (
    IncidentKind,
    ProgrammaticRecoveryExhaustionReceipt,
    PromptWorkflowBudget,
    RecordStatus,
    RecoveryAttempt,
    RecoveryAttemptOutcome,
    RescueAction,
    RescueOperation,
    RescuePlan,
    SupervisorIncident,
    prompt_workflow_cid,
)
from ipfs_accelerate_py.agent_supervisor.rescue_planner import (
    DEFAULT_RESCUE_OPERATION_CATALOG,
    RESCUE_PLAN_RESPONSE_NAME,
    RescueGuidanceStep,
    RescuePlanner,
    RescuePlannerPolicy,
    RescuePlannerState,
    RescuePlannerValidationError,
    RescuePlanningDisposition,
    RescuePlanningRequest,
    build_rescue_prompt,
    parse_rescue_plan,
)


NOW_MS = 10_000


def _cid(name: str) -> str:
    return prompt_workflow_cid({"fixture": name})


def _budget(**changes: int) -> PromptWorkflowBudget:
    values = {
        "max_prompt_tokens": 16_384,
        "max_provider_tokens": 4_096,
        "max_latency_ms": 60_000,
        "max_rescue_actions": 4,
    }
    values.update(changes)
    return PromptWorkflowBudget(**values)


def _incident(
    *,
    fingerprint: str = "a",
    target_ids: tuple[str, ...] = ("lane:implementation",),
    cooldown_key: str = "rescue/implementation",
    repository_root_cid: str | None = None,
    run_cid: str | None = None,
    policy_root: str | None = None,
    evidence_cids: tuple[str, ...] | None = None,
    observed_at_ms: int = NOW_MS,
) -> SupervisorIncident:
    return SupervisorIncident(
        repository_root="/workspace/repository",
        state_root="/workspace/repository/state/supervisor",
        repository_root_cid=repository_root_cid or _cid("repository"),
        policy_root=policy_root or _cid("policy"),
        run_cid=run_cid or _cid("run"),
        kind=IncidentKind.STALE_HEARTBEAT,
        failure_fingerprint="sha256:" + fingerprint * 64,
        target_ids=target_ids,
        evidence_cids=evidence_cids or (_cid("health"),),
        health={"heartbeat_state": "stale"},
        cooldown_key=cooldown_key,
        observed_at_ms=observed_at_ms,
        updated_at_ms=observed_at_ms,
    )


def _exhaustion(
    incident: SupervisorIncident,
    *,
    circuit_open: bool = False,
    created_at_ms: int = NOW_MS,
    status: RecordStatus = RecordStatus.QUARANTINED,
) -> ProgrammaticRecoveryExhaustionReceipt:
    return ProgrammaticRecoveryExhaustionReceipt(
        incident_cid=incident.incident_cid,
        repository_root_cid=incident.repository_root_cid,
        policy_root=incident.policy_root,
        run_cid=incident.run_cid,
        attempts=(
            RecoveryAttempt(
                operation=RescueOperation.RESTART_LANE,
                target_id=incident.target_ids[0],
                attempt=1,
                outcome=RecoveryAttemptOutcome.FAILED,
                receipt_cid=_cid("restart-failed-" + incident.failure_fingerprint),
                failure_fingerprint="sha256:" + "b" * 64,
            ),
        ),
        inapplicable_operations=(RescueOperation.REPAIR_ORPHANED_LOCK,),
        exhaustion_reason="bounded deterministic recovery exhausted",
        budget=_budget(),
        circuit_open=circuit_open,
        status=status,
        created_at_ms=created_at_ms,
        updated_at_ms=created_at_ms,
    )


def _request(
    incident: SupervisorIncident | None = None,
    exhaustion: ProgrammaticRecoveryExhaustionReceipt | None = None,
    **changes: Any,
) -> RescuePlanningRequest:
    current_incident = incident or _incident()
    current_exhaustion = exhaustion or _exhaustion(current_incident)
    values: dict[str, Any] = {
        "incident": current_incident,
        "exhaustion_receipt": current_exhaustion,
        "diagnostics": {
            "heartbeat_state": "stale",
            "failed_attempt_count": 2,
            "health_reference": current_incident.evidence_cids[0],
        },
        "evidence_redacted": True,
        "current_repository_root_cid": current_incident.repository_root_cid,
        "current_run_cid": current_incident.run_cid,
        "current_policy_root": current_incident.policy_root,
        "evidence_reference_cids": current_incident.evidence_cids,
        "max_provider_tokens": 1_024,
        "timeout_ms": 10_000,
        "max_cost_microunits": 20_000,
        "now_ms": NOW_MS,
    }
    values.update(changes)
    return RescuePlanningRequest(**values)


def _policy(**changes: Any) -> RescuePlannerPolicy:
    values: dict[str, Any] = {
        "provider": "test-provider",
        "model": "test-model",
        "cooldown_ms": 1_000,
        "max_prompt_tokens": 16_384,
        "max_provider_tokens": 2_048,
        "max_cost_microunits": 50_000,
        "cost_per_1k_tokens_microunits": 1_000,
    }
    values.update(changes)
    return RescuePlannerPolicy.permit(**values)


def _valid_plan(request: RescuePlanningRequest) -> RescuePlan:
    spec = DEFAULT_RESCUE_OPERATION_CATALOG[RescueOperation.RESTART_LANE]
    action = RescueAction(
        operation=RescueOperation.RESTART_LANE,
        target_id="lane:implementation",
        parameters={"grace_period_ms": 1_000},
        precondition_cids=(
            request.incident.incident_cid,
            request.exhaustion_receipt.receipt_cid,
            request.evidence_reference_cids[0],
        ),
        expected_effects=spec.expected_effects,
        success_test=spec.success_test,
        stop_condition=spec.stop_condition,
    )
    return RescuePlan(
        incident_cid=request.incident.incident_cid,
        exhaustion_receipt_cid=request.exhaustion_receipt.receipt_cid,
        repository_root_cid=request.current_repository_root_cid,
        run_cid=request.current_run_cid,
        policy_root=request.current_policy_root,
        actions=(action,),
        rationale_reference_cids=request.evidence_reference_cids,
        unresolved_risks=("The lane may remain unhealthy after one restart.",),
        max_actions=2,
    )


def _valid_response(request: RescuePlanningRequest) -> str:
    # RescuePlan's canonical nested action includes a content_id. The strict
    # parser accepts it only when the contract validates the claimed identity.
    return json.dumps(_valid_plan(request).to_dict(), sort_keys=True)


def _raw_response(request: RescuePlanningRequest) -> dict[str, Any]:
    return json.loads(_valid_response(request))


def test_provider_is_disabled_by_default_and_guidance_has_no_effects() -> None:
    calls = 0

    def provider(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return "{}"

    result = RescuePlanner(provider=provider).plan(_request())

    assert result.disposition is RescuePlanningDisposition.NO_PLAN
    assert result.reason_code == "policy_denied"
    assert result.guidance is not None
    assert result.guidance.next_steps == (RescueGuidanceStep.OPERATOR_REVIEW,)
    assert result.effects == ()
    assert result.guidance.effects == ()
    assert calls == 0


@pytest.mark.parametrize(
    ("request_change", "exhaustion_change", "reason_code"),
    [
        ({"evidence_redacted": False}, {}, "unredacted_evidence"),
        ({"diagnostics": {"api_key": "redacted"}}, {}, "unredacted_evidence"),
        (
            {"current_repository_root_cid": _cid("stale-repository")},
            {},
            "stale_roots",
        ),
        ({"max_provider_tokens": 9_999}, {}, "token_budget_denied"),
        ({"timeout_ms": 999_999}, {}, "time_budget_denied"),
        ({"max_cost_microunits": 999_999}, {}, "cost_budget_denied"),
        ({"now_ms": 1_000_000}, {"created_at_ms": 1}, "stale_exhaustion"),
        ({}, {"circuit_open": True}, "programmatic_circuit_open"),
        ({}, {"status": RecordStatus.FAILED}, "exhaustion_not_terminal"),
    ],
)
def test_all_pre_provider_gates_fail_closed_without_calling_provider(
    request_change: dict[str, Any],
    exhaustion_change: dict[str, Any],
    reason_code: str,
) -> None:
    incident = _incident()
    exhaustion = _exhaustion(incident, **exhaustion_change)
    request = _request(incident, exhaustion, **request_change)
    calls = 0

    def provider(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return _valid_response(request)

    result = RescuePlanner(_policy(), provider=provider).plan(request)

    assert result.reason_code == reason_code
    assert not result.provider_invoked
    assert result.plan is None
    assert result.effects == ()
    assert calls == 0
    if reason_code == "programmatic_circuit_open":
        assert result.quarantine_required


def test_exhaustion_must_be_bound_to_the_exact_current_incident() -> None:
    incident = _incident(fingerprint="a")
    foreign = _incident(fingerprint="c")
    request = _request(incident, _exhaustion(foreign))
    calls: list[str] = []

    result = RescuePlanner(
        _policy(), provider=lambda prompt: calls.append(prompt) or "{}"
    ).plan(request)

    assert result.reason_code == "exhaustion_mismatch"
    assert calls == []


def test_prompt_contains_only_bounded_references_roots_diagnostics_and_catalog() -> None:
    request = _request()
    prompt = build_rescue_prompt(request, policy=_policy())
    payload = json.loads(prompt)

    assert set(payload) == {
        "instruction",
        "incident_reference",
        "exhaustion_reference",
        "exact_roots",
        "bounded_redacted_diagnostics",
        "evidence_reference_cids",
        "closed_operation_catalog",
        "limits",
        "response_schema",
    }
    assert payload["incident_reference"]["incident_cid"] == (
        request.incident.incident_cid
    )
    assert payload["exhaustion_reference"]["exhaustion_receipt_cid"] == (
        request.exhaustion_receipt.receipt_cid
    )
    assert payload["exact_roots"] == {
        "repository_root": request.incident.repository_root,
        "state_root": request.incident.state_root,
        "repository_root_cid": request.current_repository_root_cid,
        "run_cid": request.current_run_cid,
        "policy_root": request.current_policy_root,
    }
    assert "health" not in payload["incident_reference"]
    assert "attempts" not in payload["exhaustion_reference"]
    assert "exhaustion_reason" not in payload["exhaustion_reference"]
    assert payload["response_schema"]["title"] == RESCUE_PLAN_RESPONSE_NAME
    assert set(payload["closed_operation_catalog"]) == {
        item.value for item in _policy().allowed_operations
    }
    assert len(prompt.encode("utf-8")) < _policy().max_prompt_tokens * 4


def test_valid_plan_is_proposal_only_and_identical_incident_is_reused() -> None:
    request = _request()
    calls: list[dict[str, Any]] = []

    def provider(prompt: str) -> str:
        calls.append(json.loads(prompt))
        return _valid_response(request)

    planner = RescuePlanner(_policy(), provider=provider, clock_ms=lambda: NOW_MS)
    first = planner.plan(request)
    second = planner.plan(request)

    assert first.disposition is RescuePlanningDisposition.PROPOSED
    assert first.provider_invoked
    assert first.plan is not None
    assert first.plan.status is RecordStatus.PROPOSED
    assert first.plan.incident_cid == request.incident.incident_cid
    assert first.plan.actions[0].operation is RescueOperation.RESTART_LANE
    assert first.effects == ()
    assert second.disposition is RescuePlanningDisposition.REUSED
    assert second.reused
    assert not second.provider_invoked
    assert second.plan is first.plan
    assert second.effects == ()
    assert len(calls) == 1


def test_cooldown_suppresses_a_changed_incident_with_the_same_key() -> None:
    first_request = _request(_incident(fingerprint="a"))
    second_request = _request(_incident(fingerprint="c"))
    calls = 0

    def provider(prompt: str) -> str:
        nonlocal calls
        calls += 1
        request = first_request if calls == 1 else second_request
        return _valid_response(request)

    planner = RescuePlanner(_policy(cooldown_ms=5_000), provider=provider)
    assert planner.plan(first_request).proposed
    result = planner.plan(second_request)

    assert result.reason_code == "cooldown_active"
    assert result.guidance is not None
    assert result.guidance.retry_after_ms == 5_000
    assert result.guidance.next_steps[0] is RescueGuidanceStep.WAIT_FOR_COOLDOWN
    assert calls == 1


def test_unavailable_provider_returns_typed_no_plan_and_identical_call_is_suppressed() -> None:
    request = _request()
    calls = 0

    def unavailable(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        raise OSError("provider offline")

    planner = RescuePlanner(_policy(), provider=unavailable, clock_ms=lambda: NOW_MS)
    first = planner.plan(request)
    second = planner.plan(request)

    assert first.reason_code == "provider_unavailable"
    assert first.provider_invoked
    assert first.guidance is not None
    assert first.guidance.next_steps[0] is (
        RescueGuidanceStep.RETRY_PROVIDER_AFTER_BACKOFF
    )
    assert first.effects == ()
    assert second.reason_code == "identical_incident_circuit_break"
    assert not second.provider_invoked
    assert second.effects == ()
    assert calls == 1


def test_repeated_provider_failures_open_the_shared_circuit() -> None:
    first = _request(_incident(fingerprint="a"))
    second = _request(_incident(fingerprint="c"))
    calls = 0

    def malformed(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return "not-json"

    planner = RescuePlanner(
        _policy(cooldown_ms=0, circuit_breaker_failures=2),
        provider=malformed,
        state=RescuePlannerState(),
        clock_ms=lambda: NOW_MS,
    )
    first_result = planner.plan(first)
    second_result = planner.plan(second)

    assert first_result.disposition is RescuePlanningDisposition.NO_PLAN
    assert first_result.reason_code == "provider_malformed_json"
    assert second_result.disposition is RescuePlanningDisposition.QUARANTINE
    assert second_result.reason_code == "planner_circuit_open"
    assert second_result.effects == ()
    assert calls == 2


def test_prompt_cost_is_checked_before_provider_invocation() -> None:
    request = _request(max_cost_microunits=1)
    calls = 0

    def provider(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        return "{}"

    result = RescuePlanner(
        _policy(cost_per_1k_tokens_microunits=100_000),
        provider=provider,
    ).plan(request)

    assert result.reason_code == "cost_budget_denied"
    assert result.estimated_cost_microunits > 1
    assert not result.provider_invoked
    assert calls == 0


def test_provider_elapsed_time_and_output_bytes_are_fail_closed() -> None:
    request = _request(timeout_ms=10)
    ticks = iter((NOW_MS, NOW_MS + 11))
    timed = RescuePlanner(
        _policy(max_response_bytes=1_024),
        provider=lambda _prompt: _valid_response(request),
        clock_ms=lambda: next(ticks),
    ).plan(request)

    assert timed.reason_code == "provider_time_over_budget"
    assert timed.plan is None
    assert timed.effects == ()

    oversized = RescuePlanner(
        _policy(max_response_bytes=64),
        provider=lambda _prompt: "{" + (" " * 100) + "}",
        clock_ms=lambda: NOW_MS,
    ).plan(_request())
    assert oversized.reason_code == "provider_response_over_budget"
    assert oversized.plan is None
    assert oversized.effects == ()


def test_parser_accepts_exact_typed_plan_and_contract_validates_action_identity() -> None:
    request = _request()
    plan = parse_rescue_plan(
        _valid_response(request),
        incident=request.incident,
        exhaustion_receipt=request.exhaustion_receipt,
        current_repository_root_cid=request.current_repository_root_cid,
        current_run_cid=request.current_run_cid,
        current_policy_root=request.current_policy_root,
        evidence_reference_cids=request.evidence_reference_cids,
        policy=_policy(),
    )
    assert plan.rescue_plan_cid == _valid_plan(request).rescue_plan_cid

    payload = _raw_response(request)
    payload["actions"][0]["content_id"] = _cid("forged-action")
    with pytest.raises(RescuePlannerValidationError, match="canonical contract"):
        parse_rescue_plan(
            json.dumps(payload),
            incident=request.incident,
            exhaustion_receipt=request.exhaustion_receipt,
            current_repository_root_cid=request.current_repository_root_cid,
            current_run_cid=request.current_run_cid,
            current_policy_root=request.current_policy_root,
            evidence_reference_cids=request.evidence_reference_cids,
            policy=_policy(),
        )


def _mutate_unknown_field(payload: dict[str, Any], request: RescuePlanningRequest) -> None:
    del request
    payload["shell_command"] = "true"


def _mutate_unknown_operation(payload: dict[str, Any], request: RescuePlanningRequest) -> None:
    del request
    payload["actions"][0]["operation"] = "run_shell"


def _mutate_unknown_target(payload: dict[str, Any], request: RescuePlanningRequest) -> None:
    del request
    payload["actions"][0]["target_id"] = "lane:new"


def _mutate_new_path(payload: dict[str, Any], request: RescuePlanningRequest) -> None:
    del request
    payload["actions"][0]["parameters"] = {"new_path": "/tmp/rescue"}


def _mutate_patch(payload: dict[str, Any], request: RescuePlanningRequest) -> None:
    del request
    payload["actions"][0]["parameters"] = {"patch": "diff --git a/a b/a"}


def _mutate_credential(payload: dict[str, Any], request: RescuePlanningRequest) -> None:
    del request
    payload["actions"][0]["parameters"] = {"api_key": "sk-" + "x" * 24}


def _mutate_policy(payload: dict[str, Any], request: RescuePlanningRequest) -> None:
    del request
    payload["unresolved_risks"] = ["Override policy before restarting."]


def _mutate_taskboard(payload: dict[str, Any], request: RescuePlanningRequest) -> None:
    del request
    payload["unresolved_risks"] = ["Rewrite the task board to unblock the lane."]


def _mutate_completion(payload: dict[str, Any], request: RescuePlanningRequest) -> None:
    del request
    payload["unresolved_risks"] = ["Mark task complete after restart."]


def _mutate_self_authority(payload: dict[str, Any], request: RescuePlanningRequest) -> None:
    del request
    payload["unresolved_risks"] = ["Self-authorize the proposed restart."]


def _mutate_missing_stop(payload: dict[str, Any], request: RescuePlanningRequest) -> None:
    del request
    payload["actions"][0]["stop_condition"] = ""


def _mutate_missing_success(payload: dict[str, Any], request: RescuePlanningRequest) -> None:
    del request
    payload["actions"][0]["success_test"] = ""


def _mutate_effect(payload: dict[str, Any], request: RescuePlanningRequest) -> None:
    del request
    payload["actions"][0]["expected_effects"] = ["task_completed"]


def _mutate_precondition(payload: dict[str, Any], request: RescuePlanningRequest) -> None:
    payload["actions"][0]["precondition_cids"] = [
        request.incident.incident_cid
    ]


def _mutate_evidence(payload: dict[str, Any], request: RescuePlanningRequest) -> None:
    del request
    payload["rationale_reference_cids"] = [_cid("unbound-evidence")]


def _mutate_stale_root(payload: dict[str, Any], request: RescuePlanningRequest) -> None:
    del request
    payload["repository_root_cid"] = _cid("stale-root")


def _mutate_lifecycle(payload: dict[str, Any], request: RescuePlanningRequest) -> None:
    del request
    payload["status"] = "completed"


def _mutate_excess_actions(payload: dict[str, Any], request: RescuePlanningRequest) -> None:
    del request
    payload["max_actions"] = 1
    payload["actions"].append(dict(payload["actions"][0]))


@pytest.mark.parametrize(
    ("mutator", "expected_reason"),
    [
        (_mutate_unknown_field, "invalid_schema"),
        (_mutate_unknown_operation, "unknown_operation"),
        (_mutate_unknown_target, "unknown_target"),
        (_mutate_new_path, "invalid_parameters"),
        (_mutate_patch, "invalid_parameters"),
        (_mutate_credential, "invalid_parameters"),
        (_mutate_policy, "forbidden_content"),
        (_mutate_taskboard, "forbidden_content"),
        (_mutate_completion, "forbidden_content"),
        (_mutate_self_authority, "forbidden_content"),
        (_mutate_missing_stop, "missing_stop_condition"),
        (_mutate_missing_success, "missing_success_condition"),
        (_mutate_effect, "invalid_expected_effects"),
        (_mutate_precondition, "invalid_preconditions"),
        (_mutate_evidence, "invalid_evidence_references"),
        (_mutate_stale_root, "stale_roots"),
        (_mutate_lifecycle, "self_authorization"),
        (_mutate_excess_actions, "excess_actions"),
    ],
)
def test_strict_parser_rejects_open_unsafe_stale_or_self_authorizing_plans(
    mutator: Any,
    expected_reason: str,
) -> None:
    request = _request()
    payload = _raw_response(request)
    mutator(payload, request)

    with pytest.raises(RescuePlannerValidationError) as raised:
        parse_rescue_plan(
            json.dumps(payload),
            incident=request.incident,
            exhaustion_receipt=request.exhaustion_receipt,
            current_repository_root_cid=request.current_repository_root_cid,
            current_run_cid=request.current_run_cid,
            current_policy_root=request.current_policy_root,
            evidence_reference_cids=request.evidence_reference_cids,
            policy=_policy(),
        )

    assert raised.value.reason_code == expected_reason


def test_operation_parameters_targets_and_catalog_conditions_are_typed() -> None:
    request = _request()
    payload = _raw_response(request)
    payload["actions"][0]["parameters"]["grace_period_ms"] = "1000"

    with pytest.raises(RescuePlannerValidationError) as raised:
        parse_rescue_plan(
            json.dumps(payload),
            incident=request.incident,
            exhaustion_receipt=request.exhaustion_receipt,
            current_repository_root_cid=request.current_repository_root_cid,
            current_run_cid=request.current_run_cid,
            current_policy_root=request.current_policy_root,
            evidence_reference_cids=request.evidence_reference_cids,
            policy=_policy(),
        )
    assert raised.value.reason_code == "invalid_parameters"

    objective = _incident(target_ids=("objective:asi-g460",))
    objective_request = _request(objective)
    objective_payload = _raw_response(request)
    objective_payload["incident_cid"] = objective.incident_cid
    objective_payload["exhaustion_receipt_cid"] = (
        objective_request.exhaustion_receipt.receipt_cid
    )
    objective_payload["repository_root_cid"] = objective.repository_root_cid
    objective_payload["run_cid"] = objective.run_cid
    objective_payload["policy_root"] = objective.policy_root
    objective_payload["actions"][0]["target_id"] = "objective:asi-g460"
    objective_payload["actions"][0]["precondition_cids"] = [
        objective.incident_cid,
        objective_request.exhaustion_receipt.receipt_cid,
    ]
    objective_payload["rationale_reference_cids"] = list(
        objective.evidence_cids
    )
    objective_payload["actions"][0].pop("content_id", None)
    with pytest.raises(RescuePlannerValidationError) as target_error:
        parse_rescue_plan(
            json.dumps(objective_payload),
            incident=objective,
            exhaustion_receipt=objective_request.exhaustion_receipt,
            current_repository_root_cid=objective.repository_root_cid,
            current_run_cid=objective.run_cid,
            current_policy_root=objective.policy_root,
            evidence_reference_cids=objective.evidence_cids,
            policy=_policy(),
        )
    assert target_error.value.reason_code == "invalid_target_type"
