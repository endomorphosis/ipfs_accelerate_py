"""VGO-045: provider-neutral patch proposal interface tests.

Acceptance coverage (compact, no oversized fixtures):

* deterministic label, deprecated-prop, design-token, ARIA, exact-route,
  and exact action-binding migrations are mechanical and rerun-identical
* opaque, ambiguous, policy-bound, security, repeated-failure, and
  constraint-conflict requests escalate without a fabricated patch
* provider absence or exception cannot broaden scope or invent a patch
* declared method/tier are recorded without a vendor
* closed wire inputs reject unknown fields, nulls, and wrong containers
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.gui_optimizer.proposal import (
    DETERMINISTIC_GUI_TRANSFORMATION_INTERFACE,
    GUI_PATCH_PROPOSER_INTERFACE,
    HUMAN_GUI_REVIEW_REQUEST_INTERFACE,
    DeterministicGuiTransformation,
    EscalationKind,
    GuiPatchProposer,
    GuiProposalError,
    HumanGuiReviewRequest,
    ProposalDisposition,
    ProposalReasonCode,
    ProposalRoute,
    TransformationKind,
    default_gui_patch_proposer,
)
from ipfs_accelerate_py.agent_supervisor.gui_optimizer.patch_scope import (
    GUI_IMPROVEMENT_PROPOSAL_INTERFACE,
)
from ipfs_datasets_py.logic.gui_optimizer.models import GuiImprovementProposal

IN_SCOPE = "swissknife/web/js/apps/agent-supervisor.js"
SOURCE = (
    "const deprecatedTitle = title;\n"
    "const color = var(--old-token);\n"
    "const route = '/legacy/inbox';\n"
    "button.setAttribute('aria-labelledby', 'missing');\n"
    "<label>Goal</label>\n"
    "dispatch({ action: 'old.action', schema: 'old.schema' });\n"
)
MODULE_PATH = Path(__file__).resolve().parents[2] / (
    "ipfs_accelerate_py/agent_supervisor/gui_optimizer/proposal.py"
)


def _source(**overrides: Any) -> dict[str, Any]:
    payload = {
        "path": IN_SCOPE,
        "content": SOURCE,
        "component_id": "comp:goal-form",
        "editable": True,
    }
    payload.update(overrides)
    return payload


def _pack(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "pack_id": "pack:agent-supervisor-goal",
        "application_id": "app:agent-supervisor",
        "screen_id": "screen:agent-supervisor",
        "objective": "Repair the goal form label.",
        "raw_sources": [_source()],
        "analysis_classification": "exact",
        "verification_status": "unverified",
        "escalation_conditions": [],
        "formal_invariant_failures": [],
        "acceptance_criteria": ["crit:goal-label"],
    }
    payload.update(overrides)
    return payload


def _transform(**overrides: Any) -> dict[str, Any]:
    payload = {
        "kind": "label",
        "path": IN_SCOPE,
        "find": "<label>Goal</label>",
        "replace": "<label for=\"goal\">Goal</label>",
        "expected_count": 1,
        "interface": DETERMINISTIC_GUI_TRANSFORMATION_INTERFACE,
        "schema_version": (
            "ipfs_accelerate_py/agent-supervisor/gui-optimizer/"
            "deterministic-transformation@1"
        ),
    }
    payload.update(overrides)
    return payload


def _request(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "request_id": "req:goal-label",
        "route_kind": "deterministic_transform",
        "context_pack": _pack(),
        "transformations": [_transform()],
        "intended_file_paths": [IN_SCOPE],
        "intended_component_ids": ["comp:goal-form"],
        "acceptance_criteria": ["Goal input has one accessible name."],
        "objective": "Ensure the goal form has an accessible name.",
        "application_id": "app:agent-supervisor",
        "screen_id": "screen:agent-supervisor",
        "analysis_classification": "exact",
        "verification_status": "unverified",
        "prior_failure_count": 0,
        "expected_test_ids": ["test:goal-form-a11y"],
        "expected_screenshot_ids": ["screenshot:keyboard-desktop"],
        "state_effect_ids": ["state:ready"],
    }
    payload.update(overrides)
    return payload


def _propose(**overrides: Any):
    proposer = default_gui_patch_proposer()
    return proposer.propose(_request(**overrides))


def test_proposer_exports_declared_interfaces() -> None:
    proposer = default_gui_patch_proposer()
    assert proposer.interface == GUI_PATCH_PROPOSER_INTERFACE
    assert GUI_PATCH_PROPOSER_INTERFACE == "GuiPatchProposer@1"
    assert DETERMINISTIC_GUI_TRANSFORMATION_INTERFACE == (
        "DeterministicGuiTransformation@1"
    )
    assert HUMAN_GUI_REVIEW_REQUEST_INTERFACE == "HumanGuiReviewRequest@1"
    assert ProposalRoute.DETERMINISTIC_TRANSFORM.value == "deterministic_transform"
    assert TransformationKind.LABEL.value == "label"


def test_module_does_not_import_model_routing() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    forbidden = {
        "llm_router",
        "model_router",
        "model_routing",
        "semantic_index",
        "semantic_capsule",
        "proof_cache",
    }
    assert imported.isdisjoint(forbidden)
    assert "proposal" not in imported or "llm_router" not in MODULE_PATH.read_text(
        encoding="utf-8"
    )
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "llm_router" not in source
    assert "model_routing" not in source


@pytest.mark.parametrize(
    ("kind", "find", "replace"),
    [
        ("label", "<label>Goal</label>", "<label for=\"goal\">Goal</label>"),
        ("deprecated_prop", "deprecatedTitle", "accessibleName"),
        ("design_token", "var(--old-token)", "var(--color-text)"),
        ("aria_reference", "'missing'", "'goal-label'"),
        ("exact_route", "'/legacy/inbox'", "'/agent-supervisor/inbox'"),
        (
            "exact_action_binding",
            "action: 'old.action', schema: 'old.schema'",
            "action: 'inbox.dispatch', schema: 'InboxDispatch@1'",
        ),
    ],
)
def test_deterministic_migrations_are_mechanical(kind: str, find: str, replace: str) -> None:
    first = _propose(transformations=[_transform(kind=kind, find=find, replace=replace)])
    second = _propose(transformations=[_transform(kind=kind, find=find, replace=replace)])
    assert first.proposed
    assert first.disposition is ProposalDisposition.PROPOSE
    assert first.vendor == ""
    assert first.declared_tier == "deterministic"
    assert first.patch_text
    assert find not in first.patch_text.split("+++", 1)[-1] or replace in first.patch_text
    assert replace in first.patch_text
    assert first.to_dict() == second.to_dict()
    decoded = GuiImprovementProposal.from_dict(dict(first.proposal or {}))
    assert decoded.interface == GUI_IMPROVEMENT_PROPOSAL_INTERFACE
    assert decoded.route_kind.value == "deterministic_transform"
    assert decoded.intended_file_paths == (IN_SCOPE,)
    assert decoded.decision.value == "pending"


def test_deterministic_records_method_without_vendor() -> None:
    result = _propose()
    assert result.declared_method == "exact_label_substitution"
    assert result.declared_tier == "deterministic"
    assert result.vendor == ""
    assert "openai" not in result.to_dict()["declared_method"]
    encoded = result.to_dict()
    assert encoded["vendor"] == ""
    assert ProposalReasonCode.DETERMINISTIC_TRANSFORM.value in result.reason_codes


def test_opaque_context_escalates_without_patch() -> None:
    result = _propose(analysis_classification="opaque")
    assert result.escalated
    assert result.patch_text == ""
    assert result.proposal is None
    assert result.review_request is not None
    assert result.review_request.escalation_kind is EscalationKind.OPAQUE
    assert ProposalReasonCode.OPAQUE_CONTEXT.value in result.reason_codes
    assert isinstance(result.review_request, HumanGuiReviewRequest)


def test_stale_verification_is_opaque() -> None:
    result = _propose(verification_status="stale")
    assert result.escalated
    assert result.review_request is not None
    assert result.review_request.escalation_kind is EscalationKind.OPAQUE


def test_ambiguous_zero_and_multiple_matches_escalate() -> None:
    missing = _propose(
        transformations=[_transform(find="<label>Missing</label>", replace="<label>X</label>")]
    )
    assert missing.escalated
    assert missing.patch_text == ""
    assert ProposalReasonCode.AMBIGUOUS_TRANSFORM.value in missing.reason_codes
    doubled = SOURCE.replace("<label>Goal</label>", "<label>Goal</label>\n<label>Goal</label>")
    multi = _propose(context_pack=_pack(raw_sources=[_source(content=doubled)]))
    assert multi.escalated
    assert multi.proposal is None


def test_policy_bound_escalates_unless_exact_action_binding() -> None:
    blocked = _propose(policy_bound=True)
    assert blocked.escalated
    assert blocked.patch_text == ""
    assert ProposalReasonCode.POLICY_BOUND.value in blocked.reason_codes
    allowed = _propose(
        policy_bound=True,
        transformations=[
            _transform(
                kind="exact_action_binding",
                find="action: 'old.action', schema: 'old.schema'",
                replace="action: 'inbox.dispatch', schema: 'InboxDispatch@1'",
            )
        ],
    )
    assert allowed.proposed
    assert "inbox.dispatch" in allowed.patch_text


def test_security_sensitive_escalates_without_patch() -> None:
    result = _propose(security_sensitive=True)
    assert result.escalated
    assert result.patch_text == ""
    assert ProposalReasonCode.SECURITY_REGRESSION.value in result.reason_codes


def test_credential_selector_in_transform_rejects() -> None:
    with pytest.raises(GuiProposalError) as caught:
        _propose(
            transformations=[
                _transform(find="const color", replace="const api_key = 'secret'")
            ]
        )
    assert caught.value.reason_code in {
        ProposalReasonCode.BROWSER_CREDENTIAL_FORBIDDEN.value,
        ProposalReasonCode.SECURITY_REGRESSION.value,
    }


def test_repeated_failures_escalate() -> None:
    result = _propose(prior_failure_count=2)
    assert result.escalated
    assert result.patch_text == ""
    assert ProposalReasonCode.REPEATED_FAILURE.value in result.reason_codes


def test_constraint_conflict_escalates() -> None:
    result = _propose(
        context_pack=_pack(formal_invariant_failures=["fail:modal-focus"])
    )
    assert result.escalated
    assert result.proposal is None
    assert ProposalReasonCode.CONSTRAINT_CONFLICT.value in result.reason_codes


def test_human_route_emits_review_request() -> None:
    result = _propose(route_kind="human_review", transformations=[])
    assert result.escalated
    assert result.declared_tier == "human"
    assert result.declared_method == "human_review"
    assert result.review_request is not None
    assert result.review_request.interface == HUMAN_GUI_REVIEW_REQUEST_INTERFACE
    assert result.review_request.route_kind is ProposalRoute.HUMAN_REVIEW
    assert result.patch_text == ""


class _RecordingProvider:
    def __init__(self, result: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._result = result
        self._error = error

    def propose(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(dict(request))
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


def test_provider_absence_cannot_fabricate_patch() -> None:
    proposer = GuiPatchProposer(provider=None)
    result = proposer.propose(_request(route_kind="small_local_model", transformations=[]))
    assert result.escalated
    assert result.patch_text == ""
    assert result.proposal is None
    assert result.declared_tier == "small_local"
    assert ProposalReasonCode.PROVIDER_ABSENT.value in result.reason_codes


@pytest.mark.parametrize("route", ["medium_model", "frontier_model"])
def test_missing_provider_on_every_model_route(route: str) -> None:
    result = default_gui_patch_proposer().propose(
        _request(route_kind=route, transformations=[])
    )
    assert result.escalated
    assert result.proposal is None
    assert result.review_request is not None
    assert result.review_request.escalation_kind is EscalationKind.PROVIDER_ABSENT


def test_provider_exception_cannot_fabricate_patch() -> None:
    provider = _RecordingProvider(error=RuntimeError("backend down"))
    proposer = GuiPatchProposer(provider=provider)
    result = proposer.propose(_request(route_kind="medium_model", transformations=[]))
    assert result.escalated
    assert result.patch_text == ""
    assert ProposalReasonCode.PROVIDER_EXCEPTION.value in result.reason_codes
    assert provider.calls[0]["route_kind"] == "medium_model"
    assert "vendor" not in provider.calls[0]


def test_injected_provider_records_declared_tier_without_vendor() -> None:
    provider = _RecordingProvider(
        result={
            "proposal": {
                "intended_file_paths": [IN_SCOPE],
            },
            "patch_text": "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n",
        }
    )
    proposer = GuiPatchProposer(provider=provider)
    result = proposer.propose(
        _request(
            route_kind="frontier_model",
            declared_method="injected_provider",
            declared_tier="frontier",
            transformations=[],
        )
    )
    assert result.proposed
    assert result.declared_method == "injected_provider"
    assert result.declared_tier == "frontier"
    assert result.vendor == ""
    assert result.route_kind is ProposalRoute.FRONTIER_MODEL
    decoded = GuiImprovementProposal.from_dict(dict(result.proposal or {}))
    assert decoded.route_kind.value == "frontier_model"


def test_provider_cannot_broaden_scope() -> None:
    provider = _RecordingProvider(
        result={
            "proposal": {
                "intended_file_paths": [
                    IN_SCOPE,
                    "swissknife/web/js/apps/legal-assistant.js",
                ]
            }
        }
    )
    proposer = GuiPatchProposer(provider=provider)
    with pytest.raises(GuiProposalError) as caught:
        proposer.propose(_request(route_kind="small_local_model", transformations=[]))
    assert caught.value.reason_code == ProposalReasonCode.SCOPE_BROADENED.value


def test_provider_vendor_field_is_unknown() -> None:
    provider = _RecordingProvider(
        result={"proposal": {"intended_file_paths": [IN_SCOPE]}, "vendor": "openai"}
    )
    proposer = GuiPatchProposer(provider=provider)
    with pytest.raises(GuiProposalError) as caught:
        proposer.propose(_request(route_kind="small_local_model", transformations=[]))
    assert caught.value.reason_code == ProposalReasonCode.UNKNOWN_FIELD.value


def test_declared_vendor_method_is_rejected() -> None:
    with pytest.raises(GuiProposalError) as caught:
        _propose(declared_method="openai-gpt4")
    assert caught.value.reason_code == ProposalReasonCode.VENDOR_FORBIDDEN.value


def test_tier_must_match_route() -> None:
    with pytest.raises(GuiProposalError):
        _propose(declared_tier="frontier")


def test_closed_wire_rejects_unknown_null_and_wrong_container() -> None:
    with pytest.raises(GuiProposalError) as unknown:
        _propose(vendor="x")
    assert unknown.value.reason_code == ProposalReasonCode.UNKNOWN_FIELD.value
    with pytest.raises(GuiProposalError) as null_field:
        payload = _request()
        payload["objective"] = None
        default_gui_patch_proposer().propose(payload)
    assert "null" in str(null_field.value)
    with pytest.raises(GuiProposalError) as tuple_paths:
        _propose(intended_file_paths=(IN_SCOPE,))
    assert tuple_paths.value.reason_code == ProposalReasonCode.INVALID_COLLECTION_TYPE.value
    with pytest.raises(GuiProposalError) as enum_route:
        payload = _request()
        payload["route_kind"] = ProposalRoute.DETERMINISTIC_TRANSFORM
        default_gui_patch_proposer().propose(payload)
    assert "string" in str(enum_route.value)


def test_undeclared_transform_path_cannot_broaden_scope() -> None:
    with pytest.raises(GuiProposalError) as caught:
        _propose(
            transformations=[
                _transform(path="swissknife/web/js/apps/legal-assistant.js")
            ]
        )
    assert caught.value.reason_code == ProposalReasonCode.SCOPE_BROADENED.value


def test_deterministic_does_not_need_a_provider() -> None:
    proposer = GuiPatchProposer(provider=None)
    result = proposer.propose(_request())
    assert result.proposed
    assert result.patch_text
    again = proposer.propose(_request())
    assert again.to_dict() == result.to_dict()


def test_typed_transformation_round_trip() -> None:
    item = DeterministicGuiTransformation.from_mapping(_transform())
    assert item.kind is TransformationKind.LABEL
    encoded = item.to_dict()
    assert DeterministicGuiTransformation.from_mapping(encoded).to_dict() == encoded
