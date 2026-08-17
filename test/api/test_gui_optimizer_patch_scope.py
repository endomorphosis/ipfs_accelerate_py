"""VGO-043: bounded patch-scope gate tests.

Acceptance coverage:

* undeclared files reject with stable reason codes
* unrelated applications reject
* backend authorization and credential changes require review
* disabled security and arbitrary HTML reject
* deleted tests require review (or reject when undeclared)
* unverified action-binding edits require review
* verified action-contract evidence admits a declared binding edit
* file/line/hunk limits reject
* generated, unresolved, and traversal paths reject
* malformed diffs reject
* missing invalidation records reject
* closed wire inputs reject unknown fields, nulls, and wrong containers
* computed declaration/kind facts override caller claims
* a scope declaration alone never verifies a binding
"""

from __future__ import annotations

from collections import UserDict
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.gui_optimizer.authority import (
    AuthorityEvidence,
    AuthorityEvidenceKind,
    AuthorityVerdict,
    ForbiddenChangeKind,
)
from ipfs_accelerate_py.agent_supervisor.gui_optimizer.patch_scope import (
    ABSOLUTE_MAX_FILES,
    DEFAULT_MAX_CHANGED_LINES,
    DEFAULT_MAX_FILES,
    GUI_IMPROVEMENT_PROPOSAL_INTERFACE,
    GUI_IMPROVEMENT_PROPOSAL_SCHEMA,
    GUI_PATCH_SCOPE_DECISION_INTERFACE,
    GUI_PATCH_SCOPE_GATE_INTERFACE,
    GuiImprovementProposalView,
    GuiPatchScopeError,
    GuiPatchScopeGate,
    PatchHunk,
    PatchOperation,
    PatchScopeInvalidationRecord,
    PatchScopeLimits,
    PatchScopeObservation,
    PatchScopeReasonCode,
    application_slug,
    default_patch_scope_gate,
    default_patch_scope_limits,
    is_screenshot_path,
    is_test_path,
    parse_unified_diff,
    path_implies_unrelated_application,
)
from ipfs_datasets_py.logic.gui_optimizer.models import (
    GuiImprovementProposal,
    UiInvalidationPlan,
)

IN_SCOPE = "swissknife/web/js/apps/agent-supervisor.js"
IN_SCOPE_TEST = "swissknife/test/browser/agent-supervisor-console-gateway.test.ts"
OTHER_APP = "swissknife/web/js/apps/legal-assistant.js"
OUT_OF_ROOTS = (
    "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/"
    "control/authorization_logic.py"
)
GENERATED = "swissknife/web/js/apps/node_modules/evil/index.js"
DIGEST_A = "sha256:" + ("a" * 64)
DIGEST_B = "sha256:" + ("b" * 64)


def _proposal(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "proposal_id": "proposal:label-form",
        "application_id": "app:agent-supervisor",
        "screen_id": "screen:agent-supervisor",
        "objective": "Ensure the goal form has an accessible name.",
        "intended_file_paths": [IN_SCOPE],
        "intended_component_ids": ["comp:goal-form"],
        "acceptance_criteria": ["Goal input has one accessible name."],
        "expected_test_ids": ["test:goal-form-a11y"],
        "expected_screenshot_ids": ["screenshot:keyboard-desktop"],
        "state_effect_ids": ["state:ready"],
        "visual_effect_summary": "Adds the declared visible label.",
        "route_kind": "deterministic_transform",
        "context_pack_id": "pack:label-form",
        "decision": "pending",
        "analysis_classification": "exact",
        "verification_status": "unverified",
        "interface": GUI_IMPROVEMENT_PROPOSAL_INTERFACE,
        "schema_version": GUI_IMPROVEMENT_PROPOSAL_SCHEMA,
    }
    payload.update(overrides)
    return payload


def _hunk(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "path": IN_SCOPE,
        "operation": "modify",
        "added_lines": 3,
        "deleted_lines": 1,
        "change_kinds": [],
        "content_markers": [],
        "diff_text": "+ <label for=\"goal\">Goal</label>",
    }
    payload.update(overrides)
    return payload


def _observation(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "hunks": [_hunk()],
        "touched_component_ids": ["comp:goal-form"],
        "touched_state_effect_ids": ["state:ready"],
        "touched_test_ids": [],
        "touched_screenshot_ids": [],
        "application_ids": ["app:agent-supervisor"],
        "action_binding_ids": [],
        "action_contract_evidence": [],
        "visual_effect_observed": True,
        "unresolved_paths": [],
    }
    payload.update(overrides)
    return payload


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


def _evaluate(**overrides: Any):
    gate = default_patch_scope_gate()
    request = {
        "proposal": _proposal(),
        "observation": _observation(),
        "invalidation": _invalidation(),
    }
    request.update(overrides)
    return gate.evaluate_request(request)


def _bound_contract(
    *,
    action_id: str = "action:dispatch",
    argument_digest: str = DIGEST_A,
) -> dict[str, Any]:
    return {
        "kind": AuthorityEvidenceKind.CONTRACT_VERIFICATION.value,
        "valid": True,
        "evidence_id": "contract-1",
        "binds_action_id": action_id,
        "binds_argument_digest": argument_digest,
    }


# ---------------------------------------------------------------------------
# Package / interface surface
# ---------------------------------------------------------------------------


def test_gate_exports_declared_interfaces() -> None:
    gate = default_patch_scope_gate()
    assert gate.interface == GUI_PATCH_SCOPE_GATE_INTERFACE
    assert GUI_PATCH_SCOPE_GATE_INTERFACE == "GuiPatchScopeGate@1"
    assert GUI_PATCH_SCOPE_DECISION_INTERFACE == "GuiPatchScopeDecision@1"
    assert GUI_IMPROVEMENT_PROPOSAL_INTERFACE == "GuiImprovementProposal@1"
    limits = default_patch_scope_limits()
    assert limits.max_files == DEFAULT_MAX_FILES
    assert limits.max_changed_lines == DEFAULT_MAX_CHANGED_LINES
    assert is_test_path(IN_SCOPE_TEST)
    assert is_screenshot_path(
        "swissknife/test/fixtures/gui-optimizer/screenshot-desktop.png"
    )
    assert application_slug("app:agent-supervisor") == "agent-supervisor"
    assert path_implies_unrelated_application(OTHER_APP, "app:agent-supervisor")
    assert not path_implies_unrelated_application(IN_SCOPE, "app:agent-supervisor")


def test_in_scope_declared_patch_is_allowed() -> None:
    decision = _evaluate()
    assert decision.verdict is AuthorityVerdict.ALLOW
    assert decision.allowed
    assert not decision.rejected
    assert PatchScopeReasonCode.ALLOWED.value in decision.reason_codes
    assert decision.declared_paths == (IN_SCOPE,)
    assert decision.observed_paths == (IN_SCOPE,)
    assert decision.undeclared_paths == ()
    encoded = decision.to_dict()
    assert encoded["interface"] == GUI_PATCH_SCOPE_DECISION_INTERFACE
    assert encoded["allowed"] is True
    authority = decision.as_authority_decision()
    assert authority.allowed


def test_datasets_proposal_and_invalidation_interop() -> None:
    proposal = GuiImprovementProposal.from_dict(_proposal())
    invalidation = UiInvalidationPlan.from_dict(_invalidation())
    gate = default_patch_scope_gate()
    decision = gate.evaluate(
        proposal,
        _observation(),
        invalidation=invalidation,
    )
    assert decision.allowed
    view = GuiImprovementProposalView.from_any(proposal)
    assert view.intended_file_paths == (IN_SCOPE,)


# ---------------------------------------------------------------------------
# Undeclared / unrelated / out-of-scope
# ---------------------------------------------------------------------------


def test_undeclared_file_rejects() -> None:
    decision = _evaluate(
        observation=_observation(
            hunks=[
                _hunk(),
                _hunk(path="swissknife/web/js/apps/agent-supervisor-extra.js"),
            ]
        )
    )
    assert decision.rejected
    assert PatchScopeReasonCode.UNDECLARED_FILE.value in decision.reason_codes
    assert PatchScopeReasonCode.UNDECLARED_PATH.value in decision.reason_codes
    assert (
        "swissknife/web/js/apps/agent-supervisor-extra.js"
        in decision.undeclared_paths
    )


def test_caller_cannot_mark_undeclared_path_as_declared() -> None:
    decision = _evaluate(
        observation=_observation(
            hunks=[_hunk(path=OTHER_APP, change_kinds=[])]
        )
    )
    assert decision.rejected
    assert PatchScopeReasonCode.UNDECLARED_FILE.value in decision.reason_codes
    assert PatchScopeReasonCode.UNRELATED_APPLICATION.value in decision.reason_codes


def test_unrelated_application_id_rejects() -> None:
    decision = _evaluate(
        observation=_observation(application_ids=["app:legal-assistant"])
    )
    assert decision.rejected
    assert PatchScopeReasonCode.UNRELATED_APPLICATION.value in decision.reason_codes


def test_path_outside_allowed_roots_rejects() -> None:
    decision = _evaluate(
        proposal=_proposal(intended_file_paths=[OUT_OF_ROOTS]),
        observation=_observation(hunks=[_hunk(path=OUT_OF_ROOTS)]),
    )
    assert decision.rejected
    assert (
        PatchScopeReasonCode.PATH_OUTSIDE_ALLOWED_ROOTS.value
        in decision.reason_codes
    )


def test_generated_vendor_path_rejects() -> None:
    decision = _evaluate(
        proposal=_proposal(intended_file_paths=[GENERATED]),
        observation=_observation(hunks=[_hunk(path=GENERATED)]),
    )
    assert decision.rejected
    assert PatchScopeReasonCode.GENERATED_PATH.value in decision.reason_codes
    assert (
        PatchScopeReasonCode.PATH_FORBIDDEN_SEGMENT.value in decision.reason_codes
    )


def test_unresolved_paths_reject() -> None:
    decision = _evaluate(observation=_observation(unresolved_paths=["???"]))
    assert decision.rejected
    assert PatchScopeReasonCode.UNRESOLVED_PATH.value in decision.reason_codes


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "../secrets.env",
        "swissknife/web/js/apps/../../etc/passwd",
        "C:\\Windows\\System32\\cmd.exe",
    ],
)
def test_absolute_or_traversal_paths_reject(path: str) -> None:
    gate = default_patch_scope_gate()
    with pytest.raises(GuiPatchScopeError) as exc_info:
        gate.evaluate(
            _proposal(intended_file_paths=[IN_SCOPE]),
            _observation(hunks=[_hunk(path=path)]),
            invalidation=_invalidation(),
        )
    assert exc_info.value.reason_code in {
        PatchScopeReasonCode.PATH_ABSOLUTE_OR_TRAVERSAL.value,
        PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value,
    }


def test_undeclared_component_and_state_reject() -> None:
    decision = _evaluate(
        observation=_observation(
            touched_component_ids=["comp:other"],
            touched_state_effect_ids=["state:other"],
        )
    )
    assert decision.rejected
    assert PatchScopeReasonCode.UNDECLARED_COMPONENT.value in decision.reason_codes
    assert (
        PatchScopeReasonCode.UNDECLARED_STATE_EFFECT.value in decision.reason_codes
    )


def test_undeclared_test_and_screenshot_reject() -> None:
    decision = _evaluate(
        observation=_observation(
            touched_test_ids=["test:undeclared"],
            touched_screenshot_ids=["screenshot:undeclared"],
        )
    )
    assert decision.rejected
    assert PatchScopeReasonCode.UNDECLARED_TEST.value in decision.reason_codes
    assert (
        PatchScopeReasonCode.UNDECLARED_SCREENSHOT.value in decision.reason_codes
    )


def test_visual_effect_without_summary_rejects() -> None:
    decision = _evaluate(
        proposal=_proposal(visual_effect_summary=""),
        observation=_observation(visual_effect_observed=True),
    )
    assert decision.rejected
    assert (
        PatchScopeReasonCode.UNDECLARED_VISUAL_EFFECT.value in decision.reason_codes
    )


# ---------------------------------------------------------------------------
# Forbidden mutations
# ---------------------------------------------------------------------------


def test_backend_authorization_requires_review() -> None:
    decision = _evaluate(
        observation=_observation(
            hunks=[
                _hunk(
                    change_kinds=["backend_authorization"],
                    diff_text="+ // skip_authorization for fixture",
                )
            ]
        )
    )
    assert decision.requires_human_review
    assert (
        PatchScopeReasonCode.BACKEND_AUTHORIZATION.value in decision.reason_codes
    )
    assert (
        PatchScopeReasonCode.SENSITIVE_CHANGE_REQUIRES_REVIEW.value
        in decision.reason_codes
    )


def test_credential_change_requires_review() -> None:
    decision = _evaluate(
        observation=_observation(
            hunks=[
                _hunk(
                    change_kinds=["credentials"],
                    content_markers=["api_key"],
                    diff_text="+ const api_key = 'fixture'",
                )
            ]
        )
    )
    assert decision.requires_human_review
    assert PatchScopeReasonCode.CREDENTIALS.value in decision.reason_codes


def test_disabled_security_rejects() -> None:
    decision = _evaluate(
        observation=_observation(
            hunks=[
                _hunk(
                    change_kinds=["disabled_security_check"],
                    diff_text="+ requires_confirmation = false",
                )
            ]
        )
    )
    assert decision.rejected
    assert (
        PatchScopeReasonCode.DISABLED_SECURITY_CHECK.value in decision.reason_codes
    )
    assert PatchScopeReasonCode.FORBIDDEN_MUTATION.value in decision.reason_codes


def test_arbitrary_html_execution_rejects() -> None:
    decision = _evaluate(
        observation=_observation(
            hunks=[
                _hunk(
                    diff_text="+ root.innerHTML = request.body",
                    content_markers=["innerHTML"],
                )
            ]
        )
    )
    assert decision.rejected
    assert (
        PatchScopeReasonCode.ARBITRARY_HTML_EXECUTION.value
        in decision.reason_codes
    )


def test_deleted_test_requires_review() -> None:
    decision = _evaluate(
        proposal=_proposal(
            intended_file_paths=[IN_SCOPE, IN_SCOPE_TEST],
            expected_test_ids=["test:goal-form-a11y"],
        ),
        observation=_observation(
            hunks=[
                _hunk(),
                _hunk(
                    path=IN_SCOPE_TEST,
                    operation="delete",
                    added_lines=0,
                    deleted_lines=12,
                    diff_text="- def test_goal_form_name():\n-     assert True",
                ),
            ],
            touched_test_ids=["test:goal-form-a11y"],
        ),
    )
    assert decision.requires_human_review
    assert PatchScopeReasonCode.DELETED_TEST.value in decision.reason_codes
    assert (
        PatchScopeReasonCode.SENSITIVE_CHANGE_REQUIRES_CONTRACT.value
        in decision.reason_codes
    )


def test_deleted_test_without_declaration_rejects() -> None:
    decision = _evaluate(
        proposal=_proposal(
            intended_file_paths=[IN_SCOPE, IN_SCOPE_TEST],
            expected_test_ids=[],
        ),
        observation=_observation(
            hunks=[
                _hunk(
                    path=IN_SCOPE_TEST,
                    operation="delete",
                    added_lines=0,
                    deleted_lines=4,
                    diff_text="- it('keeps the label', () => {})",
                )
            ],
            visual_effect_observed=False,
        ),
    )
    assert decision.rejected
    assert PatchScopeReasonCode.UNDECLARED_TEST.value in decision.reason_codes
    assert PatchScopeReasonCode.DELETED_TEST.value in decision.reason_codes


def test_unverified_binding_requires_review() -> None:
    decision = _evaluate(
        observation=_observation(
            hunks=[
                _hunk(
                    change_kinds=["unverified_action_binding"],
                    diff_text="+ action_binding: dispatch",
                )
            ],
            action_binding_ids=["action:dispatch"],
            action_argument_digest=DIGEST_A,
        )
    )
    assert decision.requires_human_review
    assert (
        PatchScopeReasonCode.UNVERIFIED_ACTION_BINDING.value
        in decision.reason_codes
    )
    assert (
        PatchScopeReasonCode.MISSING_ACTION_CONTRACT_EVIDENCE.value
        in decision.reason_codes
    )


def test_scope_declaration_cannot_verify_binding() -> None:
    decision = _evaluate(
        observation=_observation(
            hunks=[_hunk(change_kinds=["unverified_action_binding"])],
            action_binding_ids=["action:dispatch"],
            action_argument_digest=DIGEST_A,
            action_contract_evidence=[
                {
                    "kind": AuthorityEvidenceKind.SCOPE_DECLARATION.value,
                    "valid": True,
                    "evidence_id": "scope-1",
                    "binds_action_id": "action:dispatch",
                    "binds_argument_digest": DIGEST_A,
                }
            ],
        )
    )
    assert decision.requires_human_review
    assert (
        PatchScopeReasonCode.SCOPE_DECLARATION_NOT_AUTHORITY.value
        in decision.reason_codes
    )
    assert (
        PatchScopeReasonCode.UNVERIFIED_ACTION_BINDING.value
        in decision.reason_codes
    )


def test_verified_binding_with_contract_allows() -> None:
    decision = _evaluate(
        observation=_observation(
            hunks=[
                _hunk(
                    change_kinds=["unverified_action_binding"],
                    diff_text="+ action_binding: dispatch",
                )
            ],
            action_binding_ids=["action:dispatch"],
            action_argument_digest=DIGEST_A,
            action_contract_evidence=[_bound_contract()],
        )
    )
    assert decision.allowed
    assert (
        PatchScopeReasonCode.UNVERIFIED_ACTION_BINDING.value
        not in decision.reason_codes
    )


def test_binding_digest_mismatch_requires_review() -> None:
    decision = _evaluate(
        observation=_observation(
            hunks=[_hunk(change_kinds=["unverified_action_binding"])],
            action_binding_ids=["action:dispatch"],
            action_argument_digest=DIGEST_A,
            action_contract_evidence=[_bound_contract(argument_digest=DIGEST_B)],
        )
    )
    assert decision.requires_human_review
    assert (
        PatchScopeReasonCode.EVIDENCE_BINDING_MISMATCH.value
        in decision.reason_codes
    )


# ---------------------------------------------------------------------------
# File / line / hunk limits
# ---------------------------------------------------------------------------


def test_file_limit_exceeded_rejects() -> None:
    paths = [
        f"swissknife/web/js/apps/agent-supervisor.part{index}.js"
        for index in range(DEFAULT_MAX_FILES + 1)
    ]
    decision = _evaluate(
        proposal=_proposal(intended_file_paths=paths),
        observation=_observation(
            hunks=[_hunk(path=path) for path in paths],
            visual_effect_observed=False,
        ),
    )
    assert decision.rejected
    assert PatchScopeReasonCode.FILE_LIMIT_EXCEEDED.value in decision.reason_codes


def test_line_limit_exceeded_rejects() -> None:
    decision = _evaluate(
        observation=_observation(
            hunks=[
                _hunk(
                    added_lines=DEFAULT_MAX_CHANGED_LINES,
                    deleted_lines=1,
                )
            ]
        )
    )
    assert decision.rejected
    assert PatchScopeReasonCode.LINE_LIMIT_EXCEEDED.value in decision.reason_codes


def test_limits_cannot_exceed_absolute_caps() -> None:
    with pytest.raises(GuiPatchScopeError) as exc_info:
        PatchScopeLimits(max_files=ABSOLUTE_MAX_FILES + 1)
    assert (
        exc_info.value.reason_code
        == PatchScopeReasonCode.INVALID_PATCH_SCOPE_INPUT.value
    )


def test_limits_cannot_be_zero_or_negative() -> None:
    with pytest.raises(GuiPatchScopeError):
        PatchScopeLimits(max_files=0)
    with pytest.raises(GuiPatchScopeError):
        PatchScopeLimits(max_changed_lines=-1)


def test_tighter_limits_are_enforced() -> None:
    gate = GuiPatchScopeGate(limits=PatchScopeLimits(max_files=1, max_changed_lines=4))
    decision = gate.evaluate(
        _proposal(intended_file_paths=[IN_SCOPE, IN_SCOPE_TEST]),
        _observation(
            hunks=[
                _hunk(),
                _hunk(path=IN_SCOPE_TEST, added_lines=2, deleted_lines=0),
            ]
        ),
        invalidation=_invalidation(),
    )
    assert decision.rejected
    assert PatchScopeReasonCode.FILE_LIMIT_EXCEEDED.value in decision.reason_codes


def test_require_invalidation_cannot_be_disabled() -> None:
    with pytest.raises(GuiPatchScopeError):
        GuiPatchScopeGate(require_invalidation=False)


# ---------------------------------------------------------------------------
# Diff semantics and unified-diff fixtures
# ---------------------------------------------------------------------------


def test_create_hunk_cannot_delete_lines() -> None:
    with pytest.raises(GuiPatchScopeError) as exc_info:
        PatchHunk(
            path=IN_SCOPE,
            operation=PatchOperation.CREATE,
            added_lines=2,
            deleted_lines=1,
        )
    assert (
        exc_info.value.reason_code
        == PatchScopeReasonCode.DIFF_SEMANTICS_INVALID.value
    )


def test_rename_requires_old_path() -> None:
    with pytest.raises(GuiPatchScopeError):
        PatchHunk(
            path=IN_SCOPE,
            operation=PatchOperation.RENAME,
            added_lines=1,
            deleted_lines=1,
        )


def test_parse_unified_diff_out_of_scope_file() -> None:
    diff = (
        f"--- a/{IN_SCOPE}\n"
        f"+++ b/{IN_SCOPE}\n"
        "@@ -1,1 +1,2 @@\n"
        " label\n"
        "+name\n"
        f"--- a/{OTHER_APP}\n"
        f"+++ b/{OTHER_APP}\n"
        "@@ -1,1 +1,2 @@\n"
        " other\n"
        "+app\n"
    )
    hunks = parse_unified_diff(diff)
    assert len(hunks) == 2
    decision = _evaluate(
        observation=PatchScopeObservation(
            hunks=hunks,
            application_ids=("app:agent-supervisor",),
        )
    )
    assert decision.rejected
    assert PatchScopeReasonCode.UNDECLARED_FILE.value in decision.reason_codes
    assert PatchScopeReasonCode.UNRELATED_APPLICATION.value in decision.reason_codes


def test_observation_accepts_unified_diff_text() -> None:
    diff = (
        f"--- a/{IN_SCOPE}\n"
        f"+++ b/{IN_SCOPE}\n"
        "@@ -10,1 +10,2 @@\n"
        " keep\n"
        "+label\n"
    )
    observation = PatchScopeObservation.from_mapping(
        {
            "diff_text": diff,
            "touched_component_ids": ["comp:goal-form"],
            "touched_state_effect_ids": ["state:ready"],
            "application_ids": ["app:agent-supervisor"],
            "visual_effect_observed": True,
        }
    )
    decision = default_patch_scope_gate().evaluate(
        _proposal(),
        observation,
        invalidation=_invalidation(),
    )
    assert decision.allowed


def test_malformed_unified_diff_rejects() -> None:
    with pytest.raises(GuiPatchScopeError) as exc_info:
        parse_unified_diff("+++ b/only-one-side\n@@ broken")
    assert (
        exc_info.value.reason_code
        == PatchScopeReasonCode.DIFF_SEMANTICS_INVALID.value
    )


def test_empty_hunks_reject() -> None:
    with pytest.raises(GuiPatchScopeError):
        PatchScopeObservation(hunks=())


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------


def test_missing_invalidation_record_rejects() -> None:
    gate = default_patch_scope_gate()
    decision = gate.evaluate(_proposal(), _observation(), invalidation=None)
    assert decision.rejected
    assert (
        PatchScopeReasonCode.MISSING_INVALIDATION_RECORD.value
        in decision.reason_codes
    )


def test_invalidation_coverage_gap_rejects() -> None:
    decision = _evaluate(
        invalidation=_invalidation(affected_component_ids=["comp:other-only"])
    )
    assert decision.rejected
    assert (
        PatchScopeReasonCode.INVALIDATION_COVERAGE_GAP.value
        in decision.reason_codes
    )


def test_invalidation_fallback_covers_unknown_components() -> None:
    decision = _evaluate(
        invalidation=_invalidation(
            affected_component_ids=[],
            fallback_triggered=True,
            fallback_explanation="opaque edge expanded the plan",
            reasons=["opaque_edge", "fallback_expansion"],
        )
    )
    assert decision.allowed


# ---------------------------------------------------------------------------
# Closed schema / typing
# ---------------------------------------------------------------------------


def test_unknown_proposal_field_rejects() -> None:
    with pytest.raises(GuiPatchScopeError) as exc_info:
        GuiImprovementProposalView.from_mapping(_proposal(extra="nope"))
    assert exc_info.value.reason_code == PatchScopeReasonCode.UNKNOWN_FIELD.value


def test_tuple_is_not_a_json_array_on_the_wire() -> None:
    with pytest.raises(GuiPatchScopeError) as exc_info:
        GuiImprovementProposalView.from_mapping(
            _proposal(intended_file_paths=(IN_SCOPE,))
        )
    assert (
        exc_info.value.reason_code
        == PatchScopeReasonCode.INVALID_COLLECTION_TYPE.value
    )


def test_present_null_collection_rejects() -> None:
    with pytest.raises(GuiPatchScopeError) as exc_info:
        PatchScopeObservation.from_mapping(
            {
                "hunks": [_hunk()],
                "touched_component_ids": None,
            }
        )
    assert (
        exc_info.value.reason_code
        == PatchScopeReasonCode.INVALID_COLLECTION_TYPE.value
    )


def test_userdict_is_not_a_json_object() -> None:
    with pytest.raises(GuiPatchScopeError) as exc_info:
        GuiImprovementProposalView.from_mapping(UserDict(_proposal()))
    assert (
        exc_info.value.reason_code
        == PatchScopeReasonCode.INVALID_COLLECTION_TYPE.value
    )


def test_unknown_evaluate_request_field_rejects() -> None:
    gate = default_patch_scope_gate()
    with pytest.raises(GuiPatchScopeError) as exc_info:
        gate.evaluate_request(
            {
                "proposal": _proposal(),
                "observation": _observation(),
                "invalidation": _invalidation(),
                "execute": True,
            }
        )
    assert exc_info.value.reason_code == PatchScopeReasonCode.UNKNOWN_FIELD.value


def test_empty_intended_files_reject_at_decode() -> None:
    with pytest.raises(GuiPatchScopeError) as exc_info:
        GuiImprovementProposalView.from_mapping(
            _proposal(intended_file_paths=[])
        )
    assert (
        exc_info.value.reason_code
        == PatchScopeReasonCode.MISSING_PROPOSAL_DECLARATION.value
    )


# ---------------------------------------------------------------------------
# Compact fixture matrix (evidence subset)
# ---------------------------------------------------------------------------


SCOPE_FIXTURE_CASES = (
    {
        "id": "undeclared-file",
        "observation": {
            "hunks": [_hunk(path="swissknife/web/js/apps/agent-supervisor-extra.js")]
        },
        "verdict": AuthorityVerdict.REJECT,
        "reason": PatchScopeReasonCode.UNDECLARED_FILE.value,
    },
    {
        "id": "unrelated-application",
        "observation": {"hunks": [_hunk(path=OTHER_APP)]},
        "proposal": {"intended_file_paths": [OTHER_APP]},
        "verdict": AuthorityVerdict.REJECT,
        "reason": PatchScopeReasonCode.UNRELATED_APPLICATION.value,
    },
    {
        "id": "backend-authorization",
        "observation": {
            "hunks": [_hunk(change_kinds=["backend_authorization"])]
        },
        "verdict": AuthorityVerdict.REQUIRE_HUMAN_REVIEW,
        "reason": PatchScopeReasonCode.BACKEND_AUTHORIZATION.value,
    },
    {
        "id": "credentials",
        "observation": {"hunks": [_hunk(content_markers=["client_secret"])]},
        "verdict": AuthorityVerdict.REQUIRE_HUMAN_REVIEW,
        "reason": PatchScopeReasonCode.CREDENTIALS.value,
    },
    {
        "id": "disabled-security",
        "observation": {"hunks": [_hunk(content_markers=["sandbox=false"])]},
        "verdict": AuthorityVerdict.REJECT,
        "reason": PatchScopeReasonCode.DISABLED_SECURITY_CHECK.value,
    },
    {
        "id": "arbitrary-html",
        "observation": {
            "hunks": [_hunk(diff_text="+ el.innerHTML = payload")]
        },
        "verdict": AuthorityVerdict.REJECT,
        "reason": PatchScopeReasonCode.ARBITRARY_HTML_EXECUTION.value,
    },
    {
        "id": "deleted-test",
        "proposal": {
            "intended_file_paths": [IN_SCOPE, IN_SCOPE_TEST],
        },
        "observation": {
            "hunks": [
                _hunk(
                    path=IN_SCOPE_TEST,
                    operation="delete",
                    added_lines=0,
                    deleted_lines=3,
                    diff_text="- def test_label():\n-     return True",
                )
            ],
            "touched_test_ids": ["test:goal-form-a11y"],
            "visual_effect_observed": False,
        },
        "verdict": AuthorityVerdict.REQUIRE_HUMAN_REVIEW,
        "reason": PatchScopeReasonCode.DELETED_TEST.value,
    },
    {
        "id": "file-limit",
        "proposal": {
            "intended_file_paths": [
                f"swissknife/web/js/apps/agent-supervisor.part{index}.js"
                for index in range(DEFAULT_MAX_FILES + 1)
            ]
        },
        "observation": {
            "hunks": [
                _hunk(path=f"swissknife/web/js/apps/agent-supervisor.part{index}.js")
                for index in range(DEFAULT_MAX_FILES + 1)
            ],
            "visual_effect_observed": False,
        },
        "verdict": AuthorityVerdict.REJECT,
        "reason": PatchScopeReasonCode.FILE_LIMIT_EXCEEDED.value,
    },
    {
        "id": "line-limit",
        "observation": {
            "hunks": [
                _hunk(
                    added_lines=DEFAULT_MAX_CHANGED_LINES + 1,
                    deleted_lines=0,
                )
            ]
        },
        "verdict": AuthorityVerdict.REJECT,
        "reason": PatchScopeReasonCode.LINE_LIMIT_EXCEEDED.value,
    },
    {
        "id": "unverified-binding",
        "observation": {
            "hunks": [_hunk(change_kinds=["unverified_action_binding"])],
            "action_binding_ids": ["action:dispatch"],
        },
        "verdict": AuthorityVerdict.REQUIRE_HUMAN_REVIEW,
        "reason": PatchScopeReasonCode.UNVERIFIED_ACTION_BINDING.value,
    },
)


def test_scope_fixture_case_ids_are_unique() -> None:
    ids = [case["id"] for case in SCOPE_FIXTURE_CASES]
    assert len(ids) == len(set(ids))
    assert len(SCOPE_FIXTURE_CASES) == 10


@pytest.mark.parametrize(
    "case", SCOPE_FIXTURE_CASES, ids=lambda case: case["id"]
)
def test_scope_fixture_matrix(case: dict[str, Any]) -> None:
    proposal = _proposal(**case.get("proposal", {}))
    observation = _observation(**case["observation"])
    decision = _evaluate(proposal=proposal, observation=observation)
    assert decision.verdict is case["verdict"]
    assert case["reason"] in decision.reason_codes


def test_typed_hunk_constructor_round_trip() -> None:
    hunk = PatchHunk(
        path=IN_SCOPE,
        operation=PatchOperation.MODIFY,
        added_lines=2,
        deleted_lines=1,
        change_kinds=(ForbiddenChangeKind.DELETED_TEST,),
        content_markers=("def test_",),
        diff_text="- def test_old():\n+ label",
    )
    assert hunk.changed_lines == 3
    assert hunk.observed_paths == (IN_SCOPE,)
    assert ForbiddenChangeKind.DELETED_TEST in hunk.inferred_kinds()


def test_invalidation_record_rejects_unknown_reason() -> None:
    with pytest.raises(GuiPatchScopeError):
        PatchScopeInvalidationRecord.from_mapping(
            _invalidation(reasons=["not_a_reason"])
        )


def test_authority_evidence_instances_are_accepted() -> None:
    evidence = AuthorityEvidence(
        kind=AuthorityEvidenceKind.CONTRACT_VERIFICATION,
        valid=True,
        evidence_id="contract-typed",
        binds_action_id="action:dispatch",
        binds_argument_digest=DIGEST_A,
    )
    observation = PatchScopeObservation(
        hunks=(
            PatchHunk(
                path=IN_SCOPE,
                added_lines=1,
                deleted_lines=0,
                change_kinds=(ForbiddenChangeKind.UNVERIFIED_ACTION_BINDING,),
                diff_text="+ action_binding",
            ),
        ),
        touched_component_ids=("comp:goal-form",),
        action_binding_ids=("action:dispatch",),
        action_argument_digest=DIGEST_A,
        action_contract_evidence=(evidence,),
        visual_effect_observed=True,
    )
    decision = default_patch_scope_gate().evaluate(
        _proposal(),
        observation,
        invalidation=_invalidation(),
    )
    assert decision.allowed
