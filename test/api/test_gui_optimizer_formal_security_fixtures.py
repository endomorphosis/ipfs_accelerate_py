"""VGO-071: formal, form, modal, policy, and security fixtures.

Acceptance coverage:

* every required failure is detected
* accessibility and confirmation/security regressions block automatic acceptance
* visibility/enabled state never authorizes
* unsupported proof claims remain unknown/review-required
* fixtures may model attacks but never execute production tools, credentials,
  remote scripts, arbitrary HTML, paths or commands
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.gui_optimizer.authority import (
    AuthorityVerdict,
)
from ipfs_accelerate_py.agent_supervisor.gui_optimizer.patch_scope import (
    GUI_PATCH_SCOPE_DECISION_INTERFACE,
    GuiPatchScopeDecision,
    default_patch_scope_gate,
)
from ipfs_datasets_py.logic.gui_optimizer.formal_adapter import (
    UiAsyncEffectPremise,
    UiConstraintSourceBinding,
)
from ipfs_datasets_py.logic.gui_optimizer.invariants import (
    ENGINE_AUTHORIZES_ACTIONS,
    FORBIDDEN_CLAIM_KINDS,
    FULL_ACCESSIBILITY_PROOF,
    FULL_SECURITY_PROOF,
    UI_INVARIANT_WORLD_INTERFACE,
    UI_INVARIANT_WORLD_SCHEMA,
    UiActionRuntimeObservation,
    UiConfirmationObservation,
    UiDomNodeObservation,
    UiFormInputObservation,
    UiFormSubmissionObservation,
    UiImageKind,
    UiInvariantAcceptanceOutcome,
    UiInvariantVerdict,
    UiInvariantWorld,
    UiModalFocusObservation,
    UiPolicyObservation,
    UiPresentationObservation,
    UiValidationErrorObservation,
    create_ui_invariant_engine,
)
from ipfs_datasets_py.logic.gui_optimizer.models import (
    SourceSpan,
    UiActionBinding,
    UiConstraintReceipt,
    UiEventDefinition,
    UiStateDefinition,
    UiTransitionDefinition,
)
from ipfs_datasets_py.logic.gui_optimizer.schema import (
    SOURCE_SPAN_INTERFACE,
    SOURCE_SPAN_SCHEMA,
    UI_ACTION_BINDING_INTERFACE,
    UI_ACTION_BINDING_SCHEMA,
    UI_CONSTRAINT_RECEIPT_INTERFACE,
    UI_CONSTRAINT_RECEIPT_SCHEMA,
    UI_EVENT_DEFINITION_INTERFACE,
    UI_EVENT_DEFINITION_SCHEMA,
    UI_STATE_DEFINITION_INTERFACE,
    UI_STATE_DEFINITION_SCHEMA,
    UI_TRANSITION_DEFINITION_INTERFACE,
    UI_TRANSITION_DEFINITION_SCHEMA,
    AnalysisClassification,
    ConstraintCheckStatus,
    VerificationStatus,
)

GUI_FORMAL_SECURITY_FIXTURE_SUITE_INTERFACE = "GuiFormalSecurityFixtureSuite@1"
GUI_FORMAL_SECURITY_FIXTURE_SUITE_SCHEMA = "gui-formal-security-fixture-suite/v1"
FIXTURE_RELATIVE_PATH = "test/fixtures/gui_optimizer/formal-security-cases.json"
FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "gui_optimizer"
    / "formal-security-cases.json"
)

APP = "app:agent-supervisor"
SCREEN = "screen:agent-supervisor"
MACHINE = "machine:agent-supervisor"
REVISION = "deadbeef"
EMPTY_DIGEST = (
    "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
IN_SCOPE = "swissknife/web/js/apps/agent-supervisor.js"

REQUIRED_FAILURE_IDS = (
    "unlabeled_input",
    "missing_error_association",
    "inaccessible_custom_control",
    "broken_modal_focus",
    "duplicate_ids",
    "missing_async_failure",
    "unconfirmed_destruction",
    "disabled_dispatch",
    "stale_policy",
    "confirmation_mismatch",
)
REQUIRED_FAILURE_TO_PROPERTY = {
    "unlabeled_input": "form_accessible_names",
    "missing_error_association": "form_error_association",
    "inaccessible_custom_control": "keyboard_activation",
    "broken_modal_focus": "modal_focus_lifecycle",
    "duplicate_ids": "unique_dom_ids",
    "missing_async_failure": "async_effect_completeness",
    "unconfirmed_destruction": "confirmation_bound_action",
    "disabled_dispatch": "no_hidden_dispatch",
    "stale_policy": "stale_policy_cannot_authorize",
    "confirmation_mismatch": "confirmation_bound_action",
}
ACCESSIBILITY_FAMILIES = frozenset({"form_integrity", "structure_accessibility"})
CONFIRMATION_SECURITY_PROPERTIES = frozenset(
    {
        "confirmation_bound_action",
        "no_hidden_dispatch",
        "stale_policy_cannot_authorize",
        "policy_not_browser_authoritative",
        "presentation_no_credentials",
    }
)
SUITE_FIELDS = frozenset(
    {
        "application_id",
        "cases",
        "conflict_policy",
        "forbidden_claim_kinds",
        "interface",
        "required_failure_ids",
        "schema_version",
        "screen_id",
        "suite_id",
        "task_id",
    }
)
CASE_FIELDS = frozenset(
    {
        "authorizes",
        "case_id",
        "claimed_proof_kinds",
        "expected_acceptance_outcome",
        "expected_decision",
        "expected_reason_codes",
        "expected_subject_ids",
        "expected_verdict",
        "family",
        "kind",
        "machine_only",
        "property_kind",
        "required_failure",
        "scope_request",
        "source_bindings",
        "title",
        "world_overrides",
    }
)
CASE_KINDS = frozenset(
    {"invariant_failure", "invariant_unknown", "invariant_pass", "patch_scope"}
)
DECISIONS = frozenset({"accepted", "rejected", "review_required"})
_VALIDATION_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
_FORBIDDEN_EXECUTION_MARKERS = (
    "<script",
    "javascript:",
    "eval(",
    "subprocess",
    "/bin/sh",
    "curl ",
    "wget ",
)


def load_suite() -> dict[str, Any]:
    raw = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise AssertionError("formal-security-cases.json must be a JSON object")
    return raw


def _reject_unknown(payload: Mapping[str, Any], allowed: frozenset[str], label: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise AssertionError(f"unknown {label} field(s): {', '.join(unknown)}")


def _state(
    state_id: str,
    kind: str,
    *,
    is_initial: bool = False,
    is_terminal: bool = False,
) -> UiStateDefinition:
    return UiStateDefinition(
        state_id=state_id,
        kind=kind,
        screen_id=SCREEN,
        label=kind,
        is_initial=is_initial,
        is_terminal=is_terminal,
        description="",
        interface=UI_STATE_DEFINITION_INTERFACE,
        schema_version=UI_STATE_DEFINITION_SCHEMA,
    )


def _event(event_id: str, kind: str, name: str | None = None) -> UiEventDefinition:
    return UiEventDefinition(
        event_id=event_id,
        kind=kind,
        name=name or kind,
        description="",
        interface=UI_EVENT_DEFINITION_INTERFACE,
        schema_version=UI_EVENT_DEFINITION_SCHEMA,
    )


def _transition(
    transition_id: str,
    from_state_id: str,
    to_state_id: str,
    event_id: str,
) -> UiTransitionDefinition:
    return UiTransitionDefinition(
        transition_id=transition_id,
        from_state_id=from_state_id,
        to_state_id=to_state_id,
        event_id=event_id,
        guard="",
        effect_ids=[],
        is_noop=False,
        interface=UI_TRANSITION_DEFINITION_INTERFACE,
        schema_version=UI_TRANSITION_DEFINITION_SCHEMA,
    )


def _binding(
    action_id: str,
    *,
    method: str = "method:dispatch",
    schema_id: str = "schema:dispatch",
    is_destructive: bool = False,
    requires_confirmation: bool = False,
    confirmation_id: str = "",
    component_id: str = "comp:goal-form",
) -> UiActionBinding:
    return UiActionBinding(
        action_id=action_id,
        method=method,
        schema_id=schema_id,
        requires_confirmation=requires_confirmation,
        confirmation_id=confirmation_id,
        policy_id="policy:host",
        depends_on_schema=True,
        is_destructive=is_destructive,
        component_id=component_id,
        interface=UI_ACTION_BINDING_INTERFACE,
        schema_version=UI_ACTION_BINDING_SCHEMA,
    )


def _runtime(
    action_id: str,
    *,
    method: str = "method:dispatch",
    schema_id: str = "schema:dispatch",
    visibility: str = "enabled",
    deontic: str = "permitted",
    resolution: str = "exact",
    target_count: int = 1,
    is_dispatchable: bool = True,
    has_hidden_dispatch_path: bool = False,
    runtime_reevaluated: bool = True,
    policy_fresh: bool = True,
    browser_policy_authoritative_claim: bool = False,
) -> UiActionRuntimeObservation:
    return UiActionRuntimeObservation(
        action_id=action_id,
        current_method=method,
        current_schema_id=schema_id,
        current_argument_digest=EMPTY_DIGEST,
        presentation_visibility=visibility,
        deontic_status=deontic,
        resolution=resolution,
        target_count=target_count,
        is_dispatchable=is_dispatchable,
        has_hidden_dispatch_path=has_hidden_dispatch_path,
        runtime_reevaluated=runtime_reevaluated,
        policy_fresh=policy_fresh,
        browser_policy_authoritative_claim=browser_policy_authoritative_claim,
    )


def _source_binding_from_dict(payload: Mapping[str, Any]) -> UiConstraintSourceBinding:
    return UiConstraintSourceBinding.from_dict(payload)


def _healthy_machine() -> dict[str, object]:
    states = (
        _state("state:initial", "initial", is_initial=True),
        _state("state:ready", "ready"),
        _state("state:loading", "loading"),
        _state("state:success", "success", is_terminal=True),
        _state("state:failure", "failure"),
        _state("state:recovery", "recovery"),
    )
    events = (
        _event("event:load", "click", "load"),
        _event("event:network_success", "network_success"),
        _event("event:network_failure", "network_failure"),
        _event("event:retry", "click", "retry"),
    )
    transitions = (
        _transition("t:init-ready", "state:initial", "state:ready", "event:load"),
        _transition("t:ready-loading", "state:ready", "state:loading", "event:load"),
        _transition(
            "t:loading-success",
            "state:loading",
            "state:success",
            "event:network_success",
        ),
        _transition(
            "t:loading-failure",
            "state:loading",
            "state:failure",
            "event:network_failure",
        ),
        _transition(
            "t:failure-recovery",
            "state:failure",
            "state:recovery",
            "event:retry",
        ),
        _transition("t:recovery-ready", "state:recovery", "state:ready", "event:load"),
    )
    return {
        "states": states,
        "events": events,
        "transitions": transitions,
        "initial_state_id": "state:initial",
    }


def _default_source_binding() -> UiConstraintSourceBinding:
    return UiConstraintSourceBinding(
        binding_id="binding:machine",
        subject_id=MACHINE,
        source_span=SourceSpan(
            path=IN_SCOPE,
            start_line=1,
            start_column=0,
            end_line=10,
            end_column=1,
            interface=SOURCE_SPAN_INTERFACE,
            schema_version=SOURCE_SPAN_SCHEMA,
        ),
        evidence="state-machine wire record",
    )


def _healthy_world(**overrides: object) -> UiInvariantWorld:
    machine = _healthy_machine()
    payload: dict[str, object] = {
        "application_id": APP,
        "screen_id": SCREEN,
        "machine_id": MACHINE,
        "repository_revision": REVISION,
        "initial_state_id": machine["initial_state_id"],
        "states": machine["states"],
        "events": machine["events"],
        "transitions": machine["transitions"],
        "analysis_classification": AnalysisClassification.EXACT,
        "async_effects": (
            UiAsyncEffectPremise(
                effect_id="effect:load",
                has_loading=True,
                has_success=True,
                has_failure=True,
            ),
        ),
        "required_action_ids": ("action:dispatch",),
        "action_state_ids": {
            "action:dispatch": "state:ready",
            "action:delete": "state:ready",
        },
        "action_bindings": (
            _binding("action:dispatch"),
            _binding(
                "action:delete",
                method="method:delete",
                schema_id="schema:delete",
                is_destructive=True,
                requires_confirmation=True,
                confirmation_id="confirm:delete",
            ),
        ),
        "confirmations": (
            UiConfirmationObservation(
                confirmation_id="confirm:delete",
                action_id="action:delete",
                argument_digest=EMPTY_DIGEST,
                granted=False,
                policy_decision_id="policy-decision:1",
            ),
        ),
        "runtime_observations": (
            _runtime("action:dispatch"),
            _runtime(
                "action:delete",
                method="method:delete",
                schema_id="schema:delete",
            ),
        ),
        "form_inputs": (
            UiFormInputObservation(
                input_id="input:goal",
                accessible_name="Goal",
                required=True,
                exposes_required_state=True,
                associated_error_ids=("error:goal-empty",),
            ),
        ),
        "validation_errors": (
            UiValidationErrorObservation(
                error_id="error:goal-empty",
                field_id="input:goal",
                message="Goal is required",
            ),
        ),
        "form_submission": UiFormSubmissionObservation(
            discards_validation_failure=False,
            success_follows_confirmed_effect=True,
        ),
        "modal_focus": (
            UiModalFocusObservation(
                modal_id="modal:confirm",
                opens_moves_focus_inside=True,
                tab_contained=True,
                escape_or_cancel_defined=True,
                close_restores_focus=True,
                hidden_not_focusable=True,
            ),
        ),
        "dom_nodes": (
            UiDomNodeObservation(
                node_id="node:heading",
                dom_id="heading-main",
                role="heading",
                heading_level=1,
                accessible_name="Agent Supervisor",
            ),
            UiDomNodeObservation(
                node_id="node:subheading",
                dom_id="heading-goals",
                role="heading",
                heading_level=2,
                accessible_name="Goals",
            ),
            UiDomNodeObservation(
                node_id="node:submit",
                dom_id="submit-goal",
                role="button",
                interactive=True,
                native_control=True,
                accessible_name="Submit goal",
                has_keyboard_activation=True,
            ),
            UiDomNodeObservation(
                node_id="node:toggle",
                dom_id="custom-toggle",
                role="switch",
                interactive=True,
                native_control=False,
                accessible_name="Compact layout",
                has_keyboard_activation=True,
            ),
            UiDomNodeObservation(
                node_id="node:chart",
                dom_id="status-chart",
                role="img",
                image_kind=UiImageKind.MEANINGFUL,
                has_text_alternative=True,
                accessible_name="Lane status",
            ),
            UiDomNodeObservation(
                node_id="node:decor",
                dom_id="spacer-mark",
                role="presentation",
                image_kind=UiImageKind.DECORATIVE,
                decorative_hidden=True,
            ),
        ),
        "presentation_components": (
            UiPresentationObservation(
                component_id="comp:goal-form",
                is_presentation=True,
                accesses_credentials=False,
            ),
        ),
        "policy": UiPolicyObservation(
            browser_policy_authoritative=False,
            host_authorization_authoritative=True,
        ),
        "source_bindings": (_default_source_binding(),),
        "unresolved": (),
        "interface": UI_INVARIANT_WORLD_INTERFACE,
        "schema_version": UI_INVARIANT_WORLD_SCHEMA,
    }
    payload.update(overrides)
    return UiInvariantWorld(**payload)  # type: ignore[arg-type]


def _machine_only_world(
    source_bindings: tuple[UiConstraintSourceBinding, ...],
) -> UiInvariantWorld:
    machine = _healthy_machine()
    return UiInvariantWorld(
        application_id=APP,
        screen_id=SCREEN,
        machine_id=MACHINE,
        repository_revision=REVISION,
        initial_state_id=str(machine["initial_state_id"]),
        states=machine["states"],  # type: ignore[arg-type]
        events=machine["events"],  # type: ignore[arg-type]
        transitions=machine["transitions"],  # type: ignore[arg-type]
        source_bindings=source_bindings,
        interface=UI_INVARIANT_WORLD_INTERFACE,
        schema_version=UI_INVARIANT_WORLD_SCHEMA,
    )


def world_from_case(case: Mapping[str, Any]) -> UiInvariantWorld:
    bindings = tuple(
        _source_binding_from_dict(item)
        for item in case.get("source_bindings") or ()
    )
    if case.get("machine_only"):
        return _machine_only_world(bindings or (_default_source_binding(),))
    overrides: dict[str, object] = {}
    raw_overrides = case.get("world_overrides") or {}
    if not isinstance(raw_overrides, Mapping):
        raise AssertionError(f"{case['case_id']} world_overrides must be a mapping")
    for key, value in raw_overrides.items():
        overrides[key] = value
    if bindings:
        overrides["source_bindings"] = bindings
    return _healthy_world(**overrides)


@pytest.fixture(scope="module")
def suite() -> dict[str, Any]:
    payload = load_suite()
    _reject_unknown(payload, SUITE_FIELDS, "GuiFormalSecurityFixtureSuite")
    return payload


@pytest.fixture(scope="module")
def cases(suite: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw_cases = suite["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise AssertionError("suite.cases must be a non-empty JSON array")
    decoded: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            raise AssertionError(f"cases[{index}] must be a JSON object")
        _reject_unknown(item, CASE_FIELDS, f"cases[{index}]")
        case_id = item.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise AssertionError(f"cases[{index}].case_id must be a non-empty string")
        if case_id in seen:
            raise AssertionError(f"duplicate case_id {case_id!r}")
        seen.add(case_id)
        kind = item.get("kind")
        if kind not in CASE_KINDS:
            raise AssertionError(f"{case_id} has unknown kind {kind!r}")
        decision = item.get("expected_decision")
        if decision not in DECISIONS:
            raise AssertionError(f"{case_id} has unknown expected_decision {decision!r}")
        if item.get("authorizes") is not False:
            raise AssertionError(f"{case_id} must declare authorizes=false")
        decoded.append(item)
    return tuple(decoded)


def _invariant_cases(cases: tuple[dict[str, Any], ...]) -> Iterator[dict[str, Any]]:
    for case in cases:
        if case["kind"] != "patch_scope":
            yield case


def _patch_cases(cases: tuple[dict[str, Any], ...]) -> Iterator[dict[str, Any]]:
    for case in cases:
        if case["kind"] == "patch_scope":
            yield case


def _check_world(case: Mapping[str, Any]):
    return create_ui_invariant_engine().check(world_from_case(case))


def _evaluate_scope(case: Mapping[str, Any]) -> GuiPatchScopeDecision:
    request = case.get("scope_request")
    if not isinstance(request, Mapping):
        raise AssertionError(f"{case['case_id']} is missing scope_request")
    return default_patch_scope_gate().evaluate_request(request)


def test_suite_interface_and_declared_outputs(suite: dict[str, Any]) -> None:
    assert FIXTURE_PATH.is_file()
    assert FIXTURE_PATH.as_posix().endswith(FIXTURE_RELATIVE_PATH)
    assert suite["interface"] == GUI_FORMAL_SECURITY_FIXTURE_SUITE_INTERFACE
    assert suite["schema_version"] == GUI_FORMAL_SECURITY_FIXTURE_SUITE_SCHEMA
    assert suite["suite_id"] == "suite:vgo-071-formal-security"
    assert suite["task_id"] == "VGO-071"
    assert suite["application_id"] == APP
    assert suite["screen_id"] == SCREEN
    assert suite["required_failure_ids"] == list(REQUIRED_FAILURE_IDS)
    assert set(suite["forbidden_claim_kinds"]) == set(FORBIDDEN_CLAIM_KINDS)
    assert "never execute" in suite["conflict_policy"]
    assert FULL_ACCESSIBILITY_PROOF is False
    assert FULL_SECURITY_PROOF is False
    assert ENGINE_AUTHORIZES_ACTIONS is False
    assert GUI_PATCH_SCOPE_DECISION_INTERFACE == "GuiPatchScopeDecision@1"
    assert UI_CONSTRAINT_RECEIPT_INTERFACE == "UiConstraintReceipt@1"


def test_required_failures_are_declared_exactly_once(
    cases: tuple[dict[str, Any], ...]
) -> None:
    observed = [
        case["required_failure"]
        for case in cases
        if case.get("required_failure")
    ]
    assert observed == list(REQUIRED_FAILURE_IDS)
    for case in cases:
        required = case.get("required_failure") or ""
        if not required:
            continue
        assert case["kind"] == "invariant_failure"
        assert case["property_kind"] == REQUIRED_FAILURE_TO_PROPERTY[required]
        assert case["expected_verdict"] == "fail"
        assert case["expected_decision"] == "rejected"
        assert case["expected_acceptance_outcome"] == "block_automatic"


@pytest.mark.parametrize("failure_id", REQUIRED_FAILURE_IDS)
def test_required_failure_is_detected(
    cases: tuple[dict[str, Any], ...], failure_id: str
) -> None:
    case = next(item for item in cases if item.get("required_failure") == failure_id)
    report = _check_world(case)
    property_kind = case["property_kind"]
    result = report.result_for(property_kind)
    assert result.verdict is UiInvariantVerdict.FAIL
    assert result.status is ConstraintCheckStatus.VIOLATED
    assert result.violation is not None
    assert result.violation.property_kind == property_kind
    expected_subjects = tuple(case["expected_subject_ids"])
    assert expected_subjects
    assert set(expected_subjects) <= set(result.violation.subject_ids)
    assert result.rule.check_id in report.receipt.violated_check_ids
    assert report.may_auto_accept is False
    assert report.acceptance_outcome is UiInvariantAcceptanceOutcome.BLOCK_AUTOMATIC
    assert report.authorizes is False
    assert report.verification_status is VerificationStatus.INVALID
    assert isinstance(report.receipt, UiConstraintReceipt)
    assert report.receipt.interface == UI_CONSTRAINT_RECEIPT_INTERFACE
    assert report.receipt.schema_version == UI_CONSTRAINT_RECEIPT_SCHEMA


def test_every_required_failure_has_counterexample_and_source_binding(
    cases: tuple[dict[str, Any], ...]
) -> None:
    for case in cases:
        if not case.get("required_failure"):
            continue
        report = _check_world(case)
        result = report.result_for(case["property_kind"])
        assert result.violation is not None
        assert result.violation.message
        assert result.violation.subject_ids
        bindings = case["source_bindings"]
        assert isinstance(bindings, list) and bindings
        for binding in bindings:
            assert binding["binding_id"]
            assert binding["subject_id"]
            span = binding.get("source_span") or {}
            assert span.get("path") == IN_SCOPE


def test_accessibility_and_security_regressions_block_automatic_acceptance(
    cases: tuple[dict[str, Any], ...]
) -> None:
    seen_a11y = False
    seen_security = False
    for case in _invariant_cases(cases):
        if case["expected_verdict"] != "fail":
            continue
        report = _check_world(case)
        assert report.may_auto_accept is False
        assert report.acceptance_outcome is UiInvariantAcceptanceOutcome.BLOCK_AUTOMATIC
        assert report.authorizes is False
        if case["family"] in ACCESSIBILITY_FAMILIES:
            seen_a11y = True
        if case["property_kind"] in CONFIRMATION_SECURITY_PROPERTIES:
            seen_security = True
    assert seen_a11y
    assert seen_security


def test_visibility_and_enabled_state_never_authorize(
    cases: tuple[dict[str, Any], ...]
) -> None:
    visibility = next(
        case
        for case in cases
        if case["case_id"] == "case:visibility-enabled-never-authorizes"
    )
    world = world_from_case(visibility)
    enabled = [
        item
        for item in world.runtime_observations
        if item.presentation_visibility.value == "enabled"
    ]
    assert enabled
    assert all(item.is_dispatchable for item in enabled)
    report = create_ui_invariant_engine().check(world)
    stale = report.result_for("stale_policy_cannot_authorize")
    assert stale.verdict is UiInvariantVerdict.FAIL
    assert report.authorizes is False
    assert report.may_auto_accept is False
    for case in _invariant_cases(cases):
        checked = _check_world(case)
        assert checked.authorizes is False
        assert ENGINE_AUTHORIZES_ACTIONS is False


def test_healthy_baseline_can_auto_accept_but_still_does_not_authorize(
    cases: tuple[dict[str, Any], ...]
) -> None:
    case = next(item for item in cases if item["case_id"] == "case:healthy-baseline")
    report = _check_world(case)
    assert all(item.verdict is UiInvariantVerdict.PASS for item in report.check_results)
    assert report.may_auto_accept is True
    assert report.acceptance_outcome is UiInvariantAcceptanceOutcome.ALLOW_AUTOMATIC
    assert report.authorizes is False
    assert report.full_accessibility_proof is False
    assert report.full_security_proof is False
    assert report.forbidden_claims_rejected is True
    assert report.verification_status is VerificationStatus.STRUCTURALLY_VALID
    assert report.receipt.violated_check_ids == ()
    assert report.receipt.unsupported_check_ids == ()


def test_unsupported_proof_claims_remain_unknown_or_review_required(
    cases: tuple[dict[str, Any], ...]
) -> None:
    case = next(
        item for item in cases if item["case_id"] == "case:unsupported-proof-claims"
    )
    report = _check_world(case)
    assert report.may_auto_accept is False
    assert report.acceptance_outcome is UiInvariantAcceptanceOutcome.BLOCK_AUTOMATIC
    assert report.authorizes is False
    assert report.verification_status is VerificationStatus.UNVERIFIED
    assert report.unsupported_markers
    claimed = set(case["claimed_proof_kinds"])
    assert claimed == set(FORBIDDEN_CLAIM_KINDS)
    for kind in claimed:
        with pytest.raises(KeyError):
            report.result_for(kind)
    observation_unknown = [
        item for item in report.check_results if item.rule.requires_observations
    ]
    assert observation_unknown
    assert all(item.verdict is UiInvariantVerdict.UNKNOWN for item in observation_unknown)
    heuristic = next(
        item
        for item in cases
        if item["case_id"] == "case:heuristic-analysis-blocks-accept"
    )
    heuristic_report = _check_world(heuristic)
    assert heuristic_report.may_auto_accept is False
    assert heuristic_report.authorizes is False
    assert heuristic_report.verification_status is not VerificationStatus.VERIFIED


def test_patch_scope_modeled_attacks_are_not_executed(
    cases: tuple[dict[str, Any], ...]
) -> None:
    modeled = list(_patch_cases(cases))
    assert modeled
    for case in modeled:
        blob = json.dumps(case["scope_request"], sort_keys=True)
        for marker in _FORBIDDEN_EXECUTION_MARKERS:
            assert marker not in blob.lower()
        assert "modeled-attack" in blob
        decision = _evaluate_scope(case)
        assert isinstance(decision, GuiPatchScopeDecision)
        assert decision.interface == GUI_PATCH_SCOPE_DECISION_INTERFACE
        encoded = decision.to_dict()
        assert encoded["interface"] == GUI_PATCH_SCOPE_DECISION_INTERFACE
        expected = case["expected_decision"]
        if expected == "rejected":
            assert decision.verdict is AuthorityVerdict.REJECT
            assert decision.rejected
            assert not decision.allowed
        elif expected == "review_required":
            assert decision.verdict is AuthorityVerdict.REQUIRE_HUMAN_REVIEW
            assert decision.requires_human_review
            assert not decision.allowed
        else:
            raise AssertionError(f"unexpected patch decision {expected!r}")
        for code in case["expected_reason_codes"]:
            assert code in decision.reason_codes
        assert decision.allowed is False


def test_fixture_worlds_are_closed_and_source_grounded(
    cases: tuple[dict[str, Any], ...]
) -> None:
    for case in _invariant_cases(cases):
        world = world_from_case(case)
        assert world.interface == UI_INVARIANT_WORLD_INTERFACE
        assert world.application_id == APP
        assert world.screen_id == SCREEN
        assert world.source_bindings
        for binding in world.source_bindings:
            assert binding.binding_id
            assert binding.subject_id
            if binding.source_span is not None:
                assert not binding.source_span.path.startswith("/")
                assert ".." not in binding.source_span.path.split("/")


def test_validation_environment_does_not_claim_absent_provers() -> None:
    assert _VALIDATION_PATH == "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    assert FULL_SECURITY_PROOF is False
    assert FULL_ACCESSIBILITY_PROOF is False
    assert "aesthetic_optimality" in FORBIDDEN_CLAIM_KINDS
    for kind in (
        "beauty",
        "complete_accessibility",
        "complete_security",
        "unbounded_correctness",
    ):
        assert kind in FORBIDDEN_CLAIM_KINDS
    assert FIXTURE_PATH.is_file()
    assert FIXTURE_PATH.name == "formal-security-cases.json"
