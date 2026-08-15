"""VGO-051: affected-check planner with uncertainty fallback.

Acceptance coverage:

* local style/component changes never select unrelated screenshots
* uncertain confidence, missing/stale/opaque edges, dynamic behavior,
  shared tokens, and prior failures broaden verification predictably
* action-binding changes include policy, interaction, and host checks
* commands come only from the host registry; browser/proposal argv,
  shell, and fallback-suppression fields reject
* a failed required check blocks acceptance and is recorded on the
  execution receipt
* closed wire inputs reject unknown fields, nulls, and wrong containers
"""

from __future__ import annotations

from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.gui_optimizer.check_plan import (
    ACCEPTANCE_POLICY,
    ACTION_REQUIRED_CHECKS,
    CHECK_REGISTRY,
    CheckFamily,
    CheckPlanReasonCode,
    CheckRiskClass,
    CheckStatus,
    DIRECT_FAMILIES,
    EvaluatorRiskClassification,
    FAMILY_ORDER,
    FALLBACK_EXPANSION_CHECKS,
    GUI_AFFECTED_CHECK_PLANNER_INTERFACE,
    GUI_CHECK_EXECUTION_RECEIPT_INTERFACE,
    GUI_CHECK_PLAN_INTERFACE,
    GuiAffectedCheckPlanner,
    GuiCheckPlanError,
    GuiCheckPlanRequest,
    HOST_PYTHON_EXECUTABLE,
    HOST_VALIDATION_PATH,
    HostCheckResult,
    HostCheckRunner,
    PlanDisposition,
    REGISTERED_CHECK_IDS,
    default_affected_check_planner,
    registry_argv,
    sealed_check_environment,
)
from ipfs_datasets_py.logic.gui_optimizer.models import UiInvalidationPlan


AFFECTED_SHOT = "screenshot:goal-form-desktop"
UNRELATED_SHOT = "screenshot:legal-assistant-wide"
GLOBAL_SHOTS = [AFFECTED_SHOT, UNRELATED_SHOT, "screenshot:other-app-mobile"]


def _invalidation(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "plan_id": "invalidate:label-form",
        "change_set_id": "changeset:label-form",
        "reasons": ["component_changed"],
        "affected_component_ids": ["comp:goal-form"],
        "affected_scenario_ids": ["scenario:keyboard-only"],
        "affected_check_ids": ["check:direct-tests"],
        "fallback_triggered": False,
        "fallback_explanation": "",
        "interface": "UiInvalidationPlan@1",
        "schema_version": "ui-invalidation-plan/v1",
        "confidence": "exact",
    }
    payload.update(overrides)
    return payload


def _risk(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "graph_confidence": "exact",
        "risk_class": "hard",
        "dynamic_behavior": False,
        "shared_tokens": False,
        "critical_evidence_unknown": False,
        "hard_gate_regression": False,
        "hard_gate_families": [],
        "failed_check_ids": [],
    }
    payload.update(overrides)
    return payload


def _request(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "invalidation": _invalidation(),
        "evaluator_risk": _risk(),
        "change_kinds": ["component_implementation"],
        "affected_screenshot_ids": [AFFECTED_SHOT],
        "known_screenshot_ids": list(GLOBAL_SHOTS),
        "unrelated_screenshot_ids": [UNRELATED_SHOT],
        "affected_component_ids": ["comp:goal-form"],
        "application_id": "app:agent-supervisor",
        "screen_id": "screen:agent-supervisor",
    }
    payload.update(overrides)
    return payload


def _plan(**overrides: Any):
    return default_affected_check_planner().plan(_request(**overrides))


def _scripted_runner(
    outcomes: dict[str, int],
) -> HostCheckRunner:
    results = {}
    for check_id, code in outcomes.items():
        results[check_id] = HostCheckResult(
            check_id=check_id,
            argv=registry_argv(check_id),
            returncode=code,
            stdout="ok" if code == 0 else "",
            stderr="" if code == 0 else f"{check_id} failed",
        )
    return HostCheckRunner(scripted_results=results)


def _pass_all(plan) -> HostCheckRunner:
    return _scripted_runner({entry.check_id: 0 for entry in plan.entries})


# ---------------------------------------------------------------------------
# Package / interface surface
# ---------------------------------------------------------------------------


def test_planner_exports_declared_interfaces() -> None:
    planner = default_affected_check_planner()
    assert planner.interface == GUI_AFFECTED_CHECK_PLANNER_INTERFACE
    assert GUI_AFFECTED_CHECK_PLANNER_INTERFACE == "GuiAffectedCheckPlanner@1"
    assert GUI_CHECK_PLAN_INTERFACE == "GuiCheckPlan@1"
    assert GUI_CHECK_EXECUTION_RECEIPT_INTERFACE == "GuiCheckExecutionReceipt@1"
    assert HOST_PYTHON_EXECUTABLE == "/usr/bin/python3.12"
    assert HOST_VALIDATION_PATH == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    env = sealed_check_environment()
    assert env["PATH"] == HOST_VALIDATION_PATH
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert FAMILY_ORDER[0] is CheckFamily.UNIT
    assert FAMILY_ORDER[1] is CheckFamily.COMPONENT
    assert FAMILY_ORDER[2] is CheckFamily.SCENARIO
    assert CheckFamily.POLICY in FAMILY_ORDER
    assert CheckFamily.HOST in FAMILY_ORDER
    assert CheckFamily.BROWSER in FAMILY_ORDER
    assert CheckFamily.BUILD in FAMILY_ORDER
    assert CheckFamily.FALLBACK in FAMILY_ORDER
    assert "check:policy" in REGISTERED_CHECK_IDS
    assert "check:broader-screen-fallback" in REGISTERED_CHECK_IDS
    assert ACCEPTANCE_POLICY == "block_on_required_failure"


def test_registry_argv_is_host_fixed_and_unique() -> None:
    seen: set[tuple[str, ...]] = set()
    for check_id, entry in CHECK_REGISTRY.items():
        argv = registry_argv(check_id)
        assert argv[0] == HOST_PYTHON_EXECUTABLE
        assert argv[-1] == check_id
        assert "-I" in argv
        assert argv == entry.argv
        assert argv not in seen
        seen.add(argv)
    with pytest.raises(GuiCheckPlanError) as caught:
        registry_argv("check:rm-rf")
    assert caught.value.reason_code == CheckPlanReasonCode.UNKNOWN_CHECK_ID.value


# ---------------------------------------------------------------------------
# Local-change precision
# ---------------------------------------------------------------------------


def test_local_component_change_avoids_unrelated_screenshots() -> None:
    plan = _plan()
    assert AFFECTED_SHOT in plan.screenshot_ids
    assert UNRELATED_SHOT not in plan.screenshot_ids
    assert "screenshot:other-app-mobile" not in plan.screenshot_ids
    assert CheckPlanReasonCode.UNRELATED_SCREENSHOT_EXCLUDED.value in plan.reason_codes
    for entry in plan.entries:
        assert UNRELATED_SHOT not in entry.screenshot_ids
        assert "screenshot:other-app-mobile" not in entry.screenshot_ids
    assert "check:broader-screen-fallback" not in plan.selected_check_ids
    assert plan.fallback_triggered is False
    encoded = plan.to_dict()
    assert encoded["interface"] == GUI_CHECK_PLAN_INTERFACE
    assert UNRELATED_SHOT not in encoded["screenshot_ids"]


def test_local_style_change_does_not_invalidate_global_screenshots() -> None:
    plan = _plan(
        invalidation=_invalidation(
            reasons=["style_changed"],
            affected_check_ids=["check:dependent-screenshots"],
            confidence="exact",
        ),
        change_kinds=["css_design_token"],
        evaluator_risk=_risk(graph_confidence="exact"),
        affected_screenshot_ids=[AFFECTED_SHOT],
        known_screenshot_ids=list(GLOBAL_SHOTS),
        unrelated_screenshot_ids=[UNRELATED_SHOT, "screenshot:other-app-mobile"],
    )
    assert plan.screenshot_ids == (AFFECTED_SHOT,)
    assert "check:dependent-screenshots" in plan.selected_check_ids
    assert "check:contrast" in plan.selected_check_ids
    assert "check:responsive" in plan.selected_check_ids
    assert "check:broader-screen-fallback" not in plan.selected_check_ids
    for entry in plan.entries:
        if entry.check_id == "check:dependent-screenshots":
            assert entry.screenshot_ids == (AFFECTED_SHOT,)
    assert UNRELATED_SHOT not in plan.screenshot_ids


def test_unknown_screenshot_ownership_does_not_select_global_inventory() -> None:
    plan = _plan(
        affected_screenshot_ids=[],
        known_screenshot_ids=list(GLOBAL_SHOTS),
        unrelated_screenshot_ids=[],
    )
    assert plan.screenshot_ids == ()
    assert CheckPlanReasonCode.LOCAL_SCREENSHOT_PRECISION.value in plan.reason_codes
    for entry in plan.entries:
        assert entry.screenshot_ids == ()


# ---------------------------------------------------------------------------
# Action changes require policy / interaction / host
# ---------------------------------------------------------------------------


def test_action_change_includes_policy_interaction_and_host() -> None:
    plan = _plan(
        invalidation=_invalidation(
            reasons=["action_changed"],
            affected_check_ids=["check:invocation-tests"],
            confidence="exact",
        ),
        change_kinds=["action_binding"],
    )
    for check_id in ACTION_REQUIRED_CHECKS:
        assert check_id in plan.selected_check_ids
    assert "check:confirmation" in plan.selected_check_ids
    assert "check:invocation-tests" in plan.selected_check_ids
    assert CheckPlanReasonCode.ACTION_POLICY_REQUIRED.value in plan.reason_codes
    required = set(plan.required_check_ids)
    assert "check:policy" in required
    assert "check:interaction" in required
    assert "check:host-boundary" in required
    families = [CHECK_REGISTRY[item].family for item in plan.selected_check_ids]
    first_expansion = min(
        index
        for index, family in enumerate(families)
        if family not in DIRECT_FAMILIES
    )
    last_direct = max(
        (index for index, family in enumerate(families) if family in DIRECT_FAMILIES),
        default=-1,
    )
    assert last_direct < first_expansion


# ---------------------------------------------------------------------------
# Uncertainty fallback
# ---------------------------------------------------------------------------


def test_exact_confidence_does_not_force_fallback() -> None:
    plan = _plan()
    assert plan.fallback_triggered is False
    assert plan.confidence == "exact"
    assert "check:broader-screen-fallback" not in plan.selected_check_ids
    assert "No uncertainty" in plan.fallback_explanation


@pytest.mark.parametrize(
    "confidence",
    ["heuristic", "conservative", "opaque"],
)
def test_uncertain_invalidation_confidence_expands_fallback(confidence: str) -> None:
    plan = _plan(
        invalidation=_invalidation(confidence=confidence),
        evaluator_risk=_risk(graph_confidence="exact"),
    )
    assert plan.fallback_triggered is True
    assert "check:broader-screen-fallback" in plan.selected_check_ids
    assert "check:broader-screen-fallback" in plan.required_check_ids
    assert CheckPlanReasonCode.UNCERTAIN_GRAPH_CONFIDENCE.value in plan.reason_codes
    assert CheckPlanReasonCode.FALLBACK_EXPANDED.value in plan.reason_codes or (
        CheckPlanReasonCode.UNCERTAIN_GRAPH_CONFIDENCE.value in plan.uncertainty_reasons
    )
    assert UNRELATED_SHOT not in plan.screenshot_ids


def test_uncertain_graph_confidence_from_evaluator_expands_fallback() -> None:
    plan = _plan(evaluator_risk=_risk(graph_confidence="heuristic"))
    assert plan.fallback_triggered is True
    assert "check:broader-screen-fallback" in plan.selected_check_ids
    for check_id in (
        "check:policy",
        "check:host-boundary",
        "check:interaction",
        "check:formal",
    ):
        assert check_id in plan.selected_check_ids
    assert plan.confidence in {"heuristic", "conservative", "opaque"}


@pytest.mark.parametrize(
    ("reason", "code"),
    [
        ("missing_edge", CheckPlanReasonCode.MISSING_EDGE.value),
        ("stale_edge", CheckPlanReasonCode.STALE_EDGE.value),
        ("opaque_edge", CheckPlanReasonCode.OPAQUE_EDGE.value),
    ],
)
def test_uncertain_edge_reasons_expand_fallback(reason: str, code: str) -> None:
    plan = _plan(
        invalidation=_invalidation(
            reasons=["component_changed", reason],
            confidence="exact",
        )
    )
    assert plan.fallback_triggered is True
    assert code in plan.uncertainty_reasons
    assert "check:broader-screen-fallback" in plan.selected_check_ids


def test_dynamic_behavior_adds_interaction_host_and_fallback() -> None:
    plan = _plan(evaluator_risk=_risk(dynamic_behavior=True))
    assert plan.fallback_triggered is True
    assert CheckPlanReasonCode.DYNAMIC_BEHAVIOR.value in plan.uncertainty_reasons
    assert "check:interaction" in plan.selected_check_ids
    assert "check:host-boundary" in plan.selected_check_ids
    assert "check:interaction-scenarios" in plan.selected_check_ids
    assert "check:broader-screen-fallback" in plan.selected_check_ids


def test_shared_tokens_broaden_visual_checks_without_global_screenshots() -> None:
    plan = _plan(
        invalidation=_invalidation(
            reasons=["style_changed"],
            affected_check_ids=["check:dependent-screenshots"],
        ),
        change_kinds=["css_design_token"],
        evaluator_risk=_risk(shared_tokens=True),
        affected_screenshot_ids=[AFFECTED_SHOT],
        known_screenshot_ids=list(GLOBAL_SHOTS),
        unrelated_screenshot_ids=[UNRELATED_SHOT],
    )
    assert plan.fallback_triggered is True
    assert CheckPlanReasonCode.SHARED_TOKENS.value in plan.uncertainty_reasons
    assert "check:contrast" in plan.selected_check_ids
    assert "check:responsive" in plan.selected_check_ids
    assert "check:broader-screen-fallback" in plan.selected_check_ids
    assert plan.screenshot_ids == (AFFECTED_SHOT,)
    assert UNRELATED_SHOT not in plan.screenshot_ids


def test_prior_failures_and_hard_gate_regression_expand_fallback() -> None:
    failed = _plan(evaluator_risk=_risk(failed_check_ids=["check:direct-tests"]))
    assert failed.fallback_triggered is True
    assert CheckPlanReasonCode.PRIOR_FAILURE.value in failed.uncertainty_reasons
    regression = _plan(evaluator_risk=_risk(hard_gate_regression=True))
    assert regression.fallback_triggered is True
    unknown = _plan(evaluator_risk=_risk(critical_evidence_unknown=True))
    assert unknown.fallback_triggered is True


def test_invalidation_fallback_flag_cannot_be_ignored() -> None:
    plan = _plan(
        invalidation=_invalidation(
            fallback_triggered=True,
            fallback_explanation="opaque edge on styled_by",
            confidence="exact",
        )
    )
    assert plan.fallback_triggered is True
    assert "check:broader-screen-fallback" in plan.selected_check_ids
    assert "opaque edge" in plan.fallback_explanation


def test_direct_checks_precede_expansion_families() -> None:
    plan = _plan(
        invalidation=_invalidation(confidence="heuristic"),
        evaluator_risk=_risk(graph_confidence="heuristic"),
        change_kinds=["component_implementation"],
    )
    families = [CHECK_REGISTRY[item].family for item in plan.selected_check_ids]
    last_direct = -1
    first_expansion = None
    for index, family in enumerate(families):
        if family in DIRECT_FAMILIES:
            last_direct = index
        elif first_expansion is None:
            first_expansion = index
    assert first_expansion is not None
    assert last_direct < first_expansion
    assert plan.selected_check_ids == tuple(sorted(
        plan.selected_check_ids,
        key=lambda item: (FAMILY_ORDER.index(CHECK_REGISTRY[item].family), item),
    ))


# ---------------------------------------------------------------------------
# Command allowlist / injection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "command",
        "commands",
        "argv",
        "shell",
        "executable",
        "subprocess",
        "process_command",
        "cwd",
    ],
)
def test_request_rejects_command_injection_fields(field: str) -> None:
    payload = _request()
    payload[field] = ["rm", "-rf", "/"]
    with pytest.raises(GuiCheckPlanError) as caught:
        GuiCheckPlanRequest.from_mapping(payload)
    assert (
        caught.value.reason_code
        == CheckPlanReasonCode.COMMAND_STRING_FORBIDDEN.value
    )


@pytest.mark.parametrize(
    "field",
    ["suppress_fallback", "skip_fallback", "disable_fallback", "no_fallback"],
)
def test_request_rejects_fallback_suppression(field: str) -> None:
    payload = _request()
    payload[field] = True
    with pytest.raises(GuiCheckPlanError) as caught:
        default_affected_check_planner().plan(payload)
    assert (
        caught.value.reason_code
        == CheckPlanReasonCode.FALLBACK_SUPPRESSION_FORBIDDEN.value
    )


def test_evaluator_risk_cannot_suppress_fallback() -> None:
    payload = _request()
    payload["evaluator_risk"] = _risk()
    payload["evaluator_risk"]["suppress_fallback"] = True
    with pytest.raises(GuiCheckPlanError) as caught:
        GuiCheckPlanRequest.from_mapping(payload)
    assert (
        caught.value.reason_code
        == CheckPlanReasonCode.FALLBACK_SUPPRESSION_FORBIDDEN.value
    )


def test_proposal_cannot_inject_commands() -> None:
    payload = _request()
    payload["proposal"] = {
        "proposal_id": "proposal:label-form",
        "application_id": "app:agent-supervisor",
        "screen_id": "screen:agent-supervisor",
        "objective": "label",
        "command": "pytest --capture=no",
    }
    with pytest.raises(GuiCheckPlanError) as caught:
        GuiCheckPlanRequest.from_mapping(payload)
    assert (
        caught.value.reason_code
        == CheckPlanReasonCode.COMMAND_STRING_FORBIDDEN.value
    )


def test_browser_input_cannot_select_host_commands() -> None:
    payload = _request()
    payload["browser_input"] = {
        "fixture_only": True,
        "payload": {},
        "selected_commands": ["pytest -k injected"],
        "selected_executables": [],
        "selected_host_paths": [],
    }
    with pytest.raises(GuiCheckPlanError) as caught:
        GuiCheckPlanRequest.from_mapping(payload)
    assert (
        caught.value.reason_code
        == CheckPlanReasonCode.BROWSER_COMMAND_FORBIDDEN.value
    )


def test_host_runner_rejects_unregistered_and_injected_argv() -> None:
    runner = HostCheckRunner()
    with pytest.raises(GuiCheckPlanError) as injected:
        runner.validate_argv(("/bin/sh", "-c", "rm -rf /"))
    assert (
        injected.value.reason_code
        == CheckPlanReasonCode.COMMAND_STRING_FORBIDDEN.value
    )
    with pytest.raises(GuiCheckPlanError) as unknown:
        runner.validate_argv(
            (HOST_PYTHON_EXECUTABLE, "-I", "-B", "-c", "raise SystemExit(0)")
        )
    assert (
        unknown.value.reason_code
        == CheckPlanReasonCode.COMMAND_STRING_FORBIDDEN.value
    )
    with pytest.raises(GuiCheckPlanError) as executable:
        HostCheckRunner(executable="/usr/bin/python3")
    assert (
        executable.value.reason_code
        == CheckPlanReasonCode.COMMAND_STRING_FORBIDDEN.value
    )
    runner.validate_argv(registry_argv("check:policy"))


# ---------------------------------------------------------------------------
# Execution receipts
# ---------------------------------------------------------------------------


def test_executed_checks_record_registry_argv_and_allow_acceptance() -> None:
    planner = default_affected_check_planner()
    plan = planner.plan(_request())
    receipt = planner.execute(plan, runner=_pass_all(plan))
    assert receipt.disposition is PlanDisposition.EXECUTED
    assert receipt.acceptance_blocked is False
    assert receipt.failed_required_check_ids == ()
    assert list(receipt.executed_check_ids) == list(plan.selected_check_ids)
    assert CheckPlanReasonCode.ACCEPTANCE_ALLOWED.value in receipt.reason_codes
    encoded = receipt.to_dict()
    assert encoded["interface"] == GUI_CHECK_EXECUTION_RECEIPT_INTERFACE
    assert encoded["acceptance_blocked"] is False
    for result in receipt.check_results:
        assert result.argv == registry_argv(result.check_id)
        assert result.status is CheckStatus.PASSED
        assert result.argv[0] == HOST_PYTHON_EXECUTABLE


def test_failed_required_check_blocks_acceptance() -> None:
    planner = default_affected_check_planner()
    plan = planner.plan(_request())
    outcomes = {entry.check_id: 0 for entry in plan.entries}
    outcomes["check:direct-tests"] = 1
    receipt = planner.execute(plan, runner=_scripted_runner(outcomes))
    assert receipt.acceptance_blocked is True
    assert receipt.disposition is PlanDisposition.BLOCKED
    assert "check:direct-tests" in receipt.failed_required_check_ids
    assert CheckPlanReasonCode.REQUIRED_CHECK_FAILED.value in receipt.reason_codes
    assert CheckPlanReasonCode.ACCEPTANCE_BLOCKED.value in receipt.reason_codes
    assert receipt.to_dict()["blocked"] is True


def test_required_failure_expands_fallback_when_plan_was_precise() -> None:
    planner = default_affected_check_planner()
    plan = planner.plan(_request())
    assert plan.fallback_triggered is False
    outcomes = {entry.check_id: 0 for entry in plan.entries}
    outcomes["check:direct-tests"] = 1
    for check_id in FALLBACK_EXPANSION_CHECKS:
        outcomes.setdefault(check_id, 0)
    receipt = planner.execute(plan, runner=_scripted_runner(outcomes))
    assert receipt.acceptance_blocked is True
    assert receipt.fallback_applied is True
    assert "check:broader-screen-fallback" in receipt.executed_check_ids
    assert CheckPlanReasonCode.FALLBACK_EXPANDED.value in receipt.reason_codes


def test_failed_optional_check_does_not_block_when_no_required_failed() -> None:
    planner = default_affected_check_planner()
    plan = planner.plan(
        _request(
            invalidation=_invalidation(
                reasons=["style_changed"],
                affected_check_ids=["check:dependent-screenshots"],
            ),
            change_kinds=["css_design_token"],
        )
    )
    optional = next(entry for entry in plan.entries if not entry.required)
    outcomes = {entry.check_id: 0 for entry in plan.entries}
    outcomes[optional.check_id] = 1
    receipt = planner.execute(plan, runner=_scripted_runner(outcomes))
    assert optional.check_id not in plan.required_check_ids
    assert receipt.acceptance_blocked is False
    assert receipt.disposition is PlanDisposition.EXECUTED


def test_unavailable_required_check_is_fail_closed() -> None:
    planner = default_affected_check_planner()
    plan = planner.plan(_request())
    outcomes = {entry.check_id: 0 for entry in plan.entries}
    outcomes["check:direct-tests"] = 127
    receipt = planner.execute(plan, runner=_scripted_runner(outcomes))
    assert receipt.acceptance_blocked is True
    assert "check:direct-tests" in receipt.failed_required_check_ids


# ---------------------------------------------------------------------------
# Closed schema / interop
# ---------------------------------------------------------------------------


def test_closed_request_rejects_unknown_fields_nulls_and_tuples() -> None:
    with pytest.raises(GuiCheckPlanError) as unknown:
        GuiCheckPlanRequest.from_mapping(_request(extra_field="nope"))
    assert unknown.value.reason_code == CheckPlanReasonCode.UNKNOWN_FIELD.value
    with pytest.raises(GuiCheckPlanError) as null_invalidation:
        GuiCheckPlanRequest.from_mapping(_request(invalidation=None))
    assert (
        null_invalidation.value.reason_code
        == CheckPlanReasonCode.INVALID_CHECK_PLAN_INPUT.value
    )
    with pytest.raises(GuiCheckPlanError) as tuple_kinds:
        GuiCheckPlanRequest.from_mapping(
            _request(change_kinds=("component_implementation",))
        )
    assert (
        tuple_kinds.value.reason_code
        == CheckPlanReasonCode.INVALID_COLLECTION_TYPE.value
    )
    with pytest.raises(GuiCheckPlanError) as missing:
        GuiCheckPlanRequest.from_mapping({"evaluator_risk": _risk()})
    assert (
        missing.value.reason_code
        == CheckPlanReasonCode.MISSING_INVALIDATION_RECORD.value
    )


def test_unknown_invalidation_check_id_rejects() -> None:
    with pytest.raises(GuiCheckPlanError) as caught:
        _plan(
            invalidation=_invalidation(affected_check_ids=["check:not-registered"])
        )
    assert caught.value.reason_code == CheckPlanReasonCode.UNKNOWN_CHECK_ID.value


def test_datasets_invalidation_plan_interop() -> None:
    invalidation = UiInvalidationPlan.from_dict(
        _invalidation(
            reasons=["action_changed"],
            affected_check_ids=["check:invocation-tests"],
        )
    )
    plan = default_affected_check_planner().plan(
        {
            "invalidation": invalidation,
            "evaluator_risk": _risk(),
            "change_kinds": ["action_binding"],
            "affected_screenshot_ids": [AFFECTED_SHOT],
        }
    )
    assert plan.invalidation_plan_id == invalidation.plan_id
    assert "check:policy" in plan.selected_check_ids
    assert "check:interaction" in plan.selected_check_ids
    assert "check:host-boundary" in plan.selected_check_ids


def test_identical_inputs_produce_identical_plan_identity() -> None:
    first = _plan()
    second = _plan()
    assert first.plan_id == second.plan_id
    assert first.plan_id.startswith("checkplan:")
    third = _plan(evaluator_risk=_risk(graph_confidence="opaque"))
    assert third.plan_id != first.plan_id


def test_empty_affected_checks_are_derived_from_change_kinds() -> None:
    plan = _plan(
        invalidation=_invalidation(affected_check_ids=[]),
        change_kinds=["state_machine"],
    )
    assert "check:reachability" in plan.selected_check_ids
    assert "check:outcome" in plan.selected_check_ids
    assert "check:formal" in plan.selected_check_ids
    assert "check:interaction-scenarios" in plan.selected_check_ids


def test_hard_gate_families_select_required_checks() -> None:
    plan = _plan(
        evaluator_risk=_risk(hard_gate_families=["policy", "confirmation"]),
        change_kinds=["component_implementation"],
    )
    assert "check:policy" in plan.selected_check_ids
    assert "check:confirmation" in plan.selected_check_ids
    assert "check:policy" in plan.required_check_ids
    assert "check:confirmation" in plan.required_check_ids


def test_evaluator_risk_rejects_unknown_family_and_check() -> None:
    with pytest.raises(GuiCheckPlanError):
        EvaluatorRiskClassification.from_mapping(
            _risk(hard_gate_families=["aesthetic"])
        )
    with pytest.raises(GuiCheckPlanError) as failed:
        EvaluatorRiskClassification.from_mapping(
            _risk(failed_check_ids=["check:not-a-check"])
        )
    assert failed.value.reason_code == CheckPlanReasonCode.UNKNOWN_CHECK_ID.value
