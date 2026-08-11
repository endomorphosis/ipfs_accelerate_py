"""VGO-009: patch and browser-host security authority tests.

Acceptance coverage:

* UI state cannot synthesize authorization
* browser content cannot select host paths or commands
* sensitive changes require contract verification or human review
* missing/invalid authority evidence rejects safely

Also exercises host-boundary, forbidden-path, stale-policy, and
exact-confirmation doctrine used by later patch-scope gates.
"""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.gui_optimizer import (
    ALWAYS_HUMAN_REVIEW_KINDS,
    DEFAULT_ALLOWED_ROOTS,
    FORBIDDEN_BROWSER_PAYLOAD_KEYS,
    GUI_ACCEPTANCE_AUTHORITY_INTERFACE,
    GUI_HOST_BOUNDARY_POLICY_INTERFACE,
    GUI_OPTIMIZER_OWNED_MODULES,
    GUI_PATCH_AUTHORITY_INTERFACE,
    AcceptanceAuthorityRequest,
    AuthorityEvidence,
    AuthorityEvidenceKind,
    AuthorityReasonCode,
    AuthorityVerdict,
    BrowserHostInput,
    ForbiddenChangeKind,
    GuiAcceptanceAuthority,
    GuiAuthorityError,
    GuiHostBoundaryPolicy,
    GuiOptimizerSecurityAuthority,
    GuiPatchAuthority,
    PatchPathClaim,
    SENSITIVE_CHANGE_KINDS,
    default_security_authority,
    path_under_allowed_roots,
)
from ipfs_accelerate_py.agent_supervisor.gui_optimizer.authority import (
    GUI_PATCH_AUTHORITY_SCHEMA,
)


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------


def test_package_exports_authority_interfaces() -> None:
    assert GUI_PATCH_AUTHORITY_INTERFACE == "GuiPatchAuthority@1"
    assert GUI_HOST_BOUNDARY_POLICY_INTERFACE == "GuiHostBoundaryPolicy@1"
    assert GUI_ACCEPTANCE_AUTHORITY_INTERFACE == "GuiAcceptanceAuthority@1"
    assert GUI_OPTIMIZER_OWNED_MODULES == ("authority",)
    assert "swissknife/web/js/apps/" in DEFAULT_ALLOWED_ROOTS
    assert "host_path" in FORBIDDEN_BROWSER_PAYLOAD_KEYS
    assert "process_command" in FORBIDDEN_BROWSER_PAYLOAD_KEYS


def test_default_security_authority_is_fail_closed_wrapper() -> None:
    authority = default_security_authority()
    assert isinstance(authority, GuiOptimizerSecurityAuthority)
    assert isinstance(authority.patch, GuiPatchAuthority)
    assert isinstance(authority.host_boundary, GuiHostBoundaryPolicy)
    assert isinstance(authority.acceptance, GuiAcceptanceAuthority)


# ---------------------------------------------------------------------------
# GuiPatchAuthority@1 — allowed roots and forbidden change kinds
# ---------------------------------------------------------------------------


def test_allowed_gui_path_is_permitted() -> None:
    authority = GuiPatchAuthority()
    decision = authority.evaluate_path(
        "swissknife/web/js/apps/agent-supervisor.js"
    )
    assert decision.verdict is AuthorityVerdict.ALLOW
    assert AuthorityReasonCode.ALLOWED.value in decision.reason_codes
    assert decision.allowed


def test_path_outside_allowed_roots_is_rejected() -> None:
    authority = GuiPatchAuthority()
    decision = authority.evaluate_path(
        "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/control/authorization_logic.py"
    )
    assert decision.rejected
    assert (
        AuthorityReasonCode.PATH_OUTSIDE_ALLOWED_ROOTS.value
        in decision.reason_codes
    )


@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "../secrets.env",
        "swissknife/web/js/apps/../../etc/passwd",
        "C:\\Windows\\System32\\cmd.exe",
        "",
    ],
)
def test_absolute_or_traversal_paths_reject(path: str) -> None:
    authority = GuiPatchAuthority()
    decision = authority.evaluate_path(path)
    assert decision.rejected
    assert (
        AuthorityReasonCode.PATH_ABSOLUTE_OR_TRAVERSAL.value
        in decision.reason_codes
        or AuthorityReasonCode.PATH_OUTSIDE_ALLOWED_ROOTS.value
        in decision.reason_codes
    )


def test_forbidden_path_segments_reject() -> None:
    authority = GuiPatchAuthority()
    decision = authority.evaluate_path(
        "swissknife/web/js/apps/node_modules/evil/index.js"
    )
    assert decision.rejected
    assert (
        AuthorityReasonCode.PATH_FORBIDDEN_SEGMENT.value in decision.reason_codes
    )


def test_undeclared_path_rejects() -> None:
    authority = GuiPatchAuthority()
    decision = authority.evaluate_path(
        "swissknife/web/js/apps/agent-supervisor.js",
        declared=False,
    )
    assert decision.rejected
    assert AuthorityReasonCode.UNDECLARED_PATH.value in decision.reason_codes


def test_sensitive_change_kinds_require_review_or_contract() -> None:
    authority = GuiPatchAuthority()
    review = authority.evaluate_change_kinds(
        [ForbiddenChangeKind.BACKEND_AUTHORIZATION]
    )
    assert review.requires_human_review
    assert (
        AuthorityReasonCode.SENSITIVE_CHANGE_REQUIRES_REVIEW.value
        in review.reason_codes
    )

    contract = authority.evaluate_change_kinds(
        [ForbiddenChangeKind.UNVERIFIED_ACTION_BINDING]
    )
    assert contract.requires_human_review
    assert (
        AuthorityReasonCode.SENSITIVE_CHANGE_REQUIRES_CONTRACT.value
        in contract.reason_codes
    )
    assert ForbiddenChangeKind.BACKEND_AUTHORIZATION in ALWAYS_HUMAN_REVIEW_KINDS
    assert ForbiddenChangeKind.DELETED_TEST in SENSITIVE_CHANGE_KINDS


def test_evaluate_claims_batches_path_and_kind_gates() -> None:
    authority = GuiPatchAuthority()
    allowed = authority.evaluate_claims(
        [
            PatchPathClaim(
                path="swissknife/web/js/apps/agent-supervisor.js",
                declared=True,
            )
        ]
    )
    assert allowed.allowed

    out_of_scope = authority.evaluate_claims(
        [
            {
                "path": "config/secrets.yaml",
                "declared": True,
                "change_kinds": [],
            }
        ]
    )
    assert out_of_scope.rejected
    assert (
        AuthorityReasonCode.PATH_OUTSIDE_ALLOWED_ROOTS.value
        in out_of_scope.reason_codes
    )

    sensitive = authority.evaluate_claims(
        [
            PatchPathClaim(
                path="swissknife/web/js/apps/agent-supervisor.js",
                change_kinds=(ForbiddenChangeKind.CREDENTIALS,),
            )
        ]
    )
    assert sensitive.requires_human_review


def test_empty_claims_reject_safely() -> None:
    authority = GuiPatchAuthority()
    decision = authority.evaluate_claims([])
    assert decision.rejected
    assert (
        AuthorityReasonCode.MISSING_AUTHORITY_EVIDENCE.value
        in decision.reason_codes
    )


def test_path_under_allowed_roots_helper() -> None:
    assert path_under_allowed_roots(
        "external/ipfs_accelerate/ipfs_accelerate_py/agent_supervisor/gui_optimizer/authority.py"
    )
    assert not path_under_allowed_roots("README.md")


# ---------------------------------------------------------------------------
# GuiHostBoundaryPolicy@1 — browser cannot select host paths/commands
# ---------------------------------------------------------------------------


def test_fixture_only_browser_payload_is_allowed() -> None:
    policy = GuiHostBoundaryPolicy()
    decision = policy.evaluate(
        BrowserHostInput(
            payload={"view": "queue", "scenario": "empty"},
            fixture_only=True,
        )
    )
    assert decision.allowed


def test_browser_selected_host_paths_reject() -> None:
    policy = GuiHostBoundaryPolicy()
    decision = policy.evaluate(
        BrowserHostInput(
            payload={"view": "queue"},
            selected_host_paths=("/home/user/secret.key",),
        )
    )
    assert decision.rejected
    assert (
        AuthorityReasonCode.BROWSER_HOST_PATH_FORBIDDEN.value
        in decision.reason_codes
    )


def test_browser_selected_commands_reject() -> None:
    policy = GuiHostBoundaryPolicy()
    decision = policy.evaluate(
        {
            "payload": {"view": "queue"},
            "selected_commands": ["rm -rf /"],
        }
    )
    assert decision.rejected
    assert (
        AuthorityReasonCode.BROWSER_COMMAND_FORBIDDEN.value
        in decision.reason_codes
    )


def test_forbidden_browser_payload_keys_reject() -> None:
    policy = GuiHostBoundaryPolicy()
    for key in (
        "host_path",
        "process_command",
        "authorization",
        "backend_credentials",
        "api_key",
    ):
        decision = policy.evaluate(
            BrowserHostInput(payload={key: "/tmp/x" if "path" in key else "x"})
        )
        assert decision.rejected, key
        assert decision.reason_codes


def test_nested_forbidden_payload_keys_reject() -> None:
    policy = GuiHostBoundaryPolicy()
    decision = policy.evaluate(
        BrowserHostInput(
            payload={"tool": {"options": {"filesystem_path": "/var/lib/data"}}}
        )
    )
    assert decision.rejected
    assert (
        AuthorityReasonCode.BROWSER_HOST_PATH_FORBIDDEN.value
        in decision.reason_codes
    )


def test_embedded_absolute_path_string_rejects() -> None:
    policy = GuiHostBoundaryPolicy()
    decision = policy.evaluate(
        BrowserHostInput(payload={"note": "/home/operator/.ssh/id_rsa"})
    )
    assert decision.rejected
    assert (
        AuthorityReasonCode.BROWSER_HOST_PATH_FORBIDDEN.value
        in decision.reason_codes
    )


def test_embedded_command_string_rejects() -> None:
    policy = GuiHostBoundaryPolicy()
    decision = policy.evaluate(
        BrowserHostInput(payload={"hint": "python3 -c 'import os; os.system(1)'"})
    )
    assert decision.rejected
    assert (
        AuthorityReasonCode.BROWSER_COMMAND_FORBIDDEN.value
        in decision.reason_codes
    )


def test_production_inputs_reject_fixture_only_doctrine() -> None:
    policy = GuiHostBoundaryPolicy()
    decision = policy.evaluate(
        BrowserHostInput(
            payload={"view": "queue"},
            uses_production_credentials=True,
        )
    )
    assert decision.rejected
    assert (
        AuthorityReasonCode.BROWSER_PRODUCTION_INPUT_FORBIDDEN.value
        in decision.reason_codes
    )

    non_fixture = policy.evaluate(
        BrowserHostInput(payload={"view": "queue"}, fixture_only=False)
    )
    assert non_fixture.rejected
    assert (
        AuthorityReasonCode.FIXTURE_ONLY_VIOLATION.value
        in non_fixture.reason_codes
    )


# ---------------------------------------------------------------------------
# GuiAcceptanceAuthority@1 — evidence, confirmation, UI non-authority
# ---------------------------------------------------------------------------


def test_ui_state_cannot_synthesize_authorization() -> None:
    acceptance = GuiAcceptanceAuthority()
    decision = acceptance.evaluate(
        AcceptanceAuthorityRequest(
            intended_action_id="dispatch_task",
            ui_visible=True,
            ui_enabled=True,
            evidence=(),
        )
    )
    assert decision.rejected
    assert (
        AuthorityReasonCode.UI_STATE_NOT_AUTHORIZATION.value
        in decision.reason_codes
    )


def test_browser_policy_is_never_authoritative() -> None:
    acceptance = GuiAcceptanceAuthority()
    decision = acceptance.evaluate(
        {
            "intended_action_id": "dispatch_task",
            "browser_policy_outcome": "allow",
            "browser_policy_authoritative_claim": True,
            "policy_decision_id": "policy-1",
            "policy_fresh": True,
        }
    )
    assert decision.rejected
    assert (
        AuthorityReasonCode.BROWSER_POLICY_NOT_AUTHORITATIVE.value
        in decision.reason_codes
    )


def test_stale_policy_decision_rejects() -> None:
    acceptance = GuiAcceptanceAuthority()
    decision = acceptance.evaluate(
        AcceptanceAuthorityRequest(
            intended_action_id="dispatch_task",
            policy_decision_id="policy-stale",
            policy_fresh=False,
        )
    )
    assert decision.rejected
    assert (
        AuthorityReasonCode.STALE_POLICY_DECISION.value in decision.reason_codes
    )


def test_exact_confirmation_binding_required() -> None:
    acceptance = GuiAcceptanceAuthority()
    missing = acceptance.evaluate(
        AcceptanceAuthorityRequest(
            intended_action_id="delete_goal",
            intended_argument_digest="args:abc",
            confirmation_required=True,
            confirmation_granted=False,
            policy_decision_id="policy-ok",
            policy_fresh=True,
        )
    )
    assert missing.rejected
    assert (
        AuthorityReasonCode.CONFIRMATION_REQUIRED.value in missing.reason_codes
    )

    mismatched = acceptance.evaluate(
        AcceptanceAuthorityRequest(
            intended_action_id="delete_goal",
            intended_argument_digest="args:abc",
            confirmation_required=True,
            confirmation_granted=True,
            confirmation_action_id="delete_goal",
            confirmation_argument_digest="args:OTHER",
            policy_decision_id="policy-ok",
            policy_fresh=True,
        )
    )
    assert mismatched.rejected
    assert (
        AuthorityReasonCode.CONFIRMATION_BINDING_MISMATCH.value
        in mismatched.reason_codes
    )

    exact = acceptance.evaluate(
        AcceptanceAuthorityRequest(
            intended_action_id="delete_goal",
            intended_argument_digest="args:abc",
            confirmation_required=True,
            confirmation_granted=True,
            confirmation_action_id="delete_goal",
            confirmation_argument_digest="args:abc",
            policy_decision_id="policy-ok",
            policy_fresh=True,
        )
    )
    assert exact.allowed


def test_missing_authority_evidence_rejects_safely() -> None:
    acceptance = GuiAcceptanceAuthority()
    decision = acceptance.evaluate(
        AcceptanceAuthorityRequest(intended_action_id="dispatch_task")
    )
    assert decision.rejected
    assert (
        AuthorityReasonCode.MISSING_AUTHORITY_EVIDENCE.value
        in decision.reason_codes
    )


def test_invalid_authority_evidence_rejects_safely() -> None:
    acceptance = GuiAcceptanceAuthority()
    decision = acceptance.evaluate(
        AcceptanceAuthorityRequest(
            intended_action_id="dispatch_task",
            evidence=(
                AuthorityEvidence(
                    kind=AuthorityEvidenceKind.CONTRACT_VERIFICATION,
                    valid=False,
                    evidence_id="broken-receipt",
                ),
            ),
        )
    )
    assert decision.rejected
    assert (
        AuthorityReasonCode.INVALID_AUTHORITY_EVIDENCE.value
        in decision.reason_codes
    )


def test_sensitive_changes_require_contract_or_human_review() -> None:
    acceptance = GuiAcceptanceAuthority()
    needs_review = acceptance.evaluate(
        AcceptanceAuthorityRequest(
            intended_action_id="edit_binding",
            policy_decision_id="policy-ok",
            policy_fresh=True,
            change_kinds=(ForbiddenChangeKind.UNVERIFIED_ACTION_BINDING,),
        )
    )
    assert needs_review.requires_human_review
    assert (
        AuthorityReasonCode.SENSITIVE_CHANGE_REQUIRES_CONTRACT.value
        in needs_review.reason_codes
    )

    with_contract = acceptance.evaluate(
        AcceptanceAuthorityRequest(
            intended_action_id="edit_binding",
            policy_decision_id="policy-ok",
            policy_fresh=True,
            change_kinds=(ForbiddenChangeKind.UNVERIFIED_ACTION_BINDING,),
            evidence=(
                AuthorityEvidence(
                    kind=AuthorityEvidenceKind.CONTRACT_VERIFICATION,
                    valid=True,
                    evidence_id="contract-1",
                ),
            ),
        )
    )
    assert with_contract.allowed

    credentials = acceptance.evaluate(
        AcceptanceAuthorityRequest(
            intended_action_id="rotate_secret",
            policy_decision_id="policy-ok",
            policy_fresh=True,
            change_kinds=(ForbiddenChangeKind.CREDENTIALS,),
            evidence=(
                AuthorityEvidence(
                    kind=AuthorityEvidenceKind.CONTRACT_VERIFICATION,
                    valid=True,
                    evidence_id="contract-2",
                ),
            ),
        )
    )
    assert credentials.requires_human_review
    assert (
        AuthorityReasonCode.SENSITIVE_CHANGE_REQUIRES_REVIEW.value
        in credentials.reason_codes
    )

    with_human = acceptance.evaluate(
        AcceptanceAuthorityRequest(
            intended_action_id="rotate_secret",
            policy_decision_id="policy-ok",
            policy_fresh=True,
            change_kinds=(ForbiddenChangeKind.CREDENTIALS,),
            evidence=(
                AuthorityEvidence(
                    kind=AuthorityEvidenceKind.HUMAN_REVIEW,
                    valid=True,
                    evidence_id="review-1",
                ),
            ),
        )
    )
    assert with_human.allowed


def test_accessibility_and_security_regressions_block_acceptance() -> None:
    acceptance = GuiAcceptanceAuthority()
    a11y = acceptance.evaluate(
        AcceptanceAuthorityRequest(
            intended_action_id="style_tweak",
            policy_decision_id="policy-ok",
            policy_fresh=True,
            accessibility_regression=True,
        )
    )
    assert a11y.rejected
    assert (
        AuthorityReasonCode.ACCESSIBILITY_REGRESSION.value in a11y.reason_codes
    )

    security = acceptance.evaluate(
        AcceptanceAuthorityRequest(
            intended_action_id="style_tweak",
            policy_decision_id="policy-ok",
            policy_fresh=True,
            security_regression=True,
        )
    )
    assert security.rejected
    assert (
        AuthorityReasonCode.SECURITY_REGRESSION.value in security.reason_codes
    )


def test_fresh_host_policy_evidence_allows_acceptance() -> None:
    acceptance = GuiAcceptanceAuthority()
    decision = acceptance.evaluate(
        AcceptanceAuthorityRequest(
            intended_action_id="dispatch_task",
            intended_argument_digest="args:1",
            ui_visible=True,
            ui_enabled=True,
            browser_policy_outcome="allow",
            browser_policy_authoritative_claim=False,
            policy_decision_id="policy-fresh",
            policy_fresh=True,
            evidence=(
                AuthorityEvidence(
                    kind=AuthorityEvidenceKind.HOST_POLICY_REEVALUATION,
                    valid=True,
                    evidence_id="host-reeval-1",
                    policy_decision_id="policy-fresh",
                    policy_fresh=True,
                ),
            ),
        )
    )
    assert decision.allowed
    assert decision.to_dict()["schema"]
    assert decision.to_dict()["allowed"] is True


# ---------------------------------------------------------------------------
# Combined facade and decision serialization
# ---------------------------------------------------------------------------


def test_combined_authority_rejects_host_path_before_acceptance() -> None:
    authority = default_security_authority()
    decision = authority.evaluate_proposal(
        claims=[
            PatchPathClaim(
                path="swissknife/web/js/apps/agent-supervisor.js",
            )
        ],
        browser_input=BrowserHostInput(
            payload={"host_path": "/tmp/x"},
        ),
        acceptance=AcceptanceAuthorityRequest(
            intended_action_id="dispatch_task",
            policy_decision_id="policy-ok",
            policy_fresh=True,
        ),
    )
    assert decision.rejected
    assert (
        AuthorityReasonCode.BROWSER_HOST_PATH_FORBIDDEN.value
        in decision.reason_codes
    )


def test_combined_authority_allows_clean_proposal() -> None:
    authority = default_security_authority()
    decision = authority.evaluate_proposal(
        claims=[
            {
                "path": "swissknife/web/js/apps/agent-supervisor.js",
                "declared": True,
                "change_kinds": [],
            }
        ],
        browser_input={
            "payload": {"view": "queue", "scenario": "ready"},
            "fixture_only": True,
        },
        acceptance={
            "intended_action_id": "rerender_preserve_focus",
            "policy_decision_id": "policy-ok",
            "policy_fresh": True,
            "evidence": [
                {
                    "kind": "scope_declaration",
                    "valid": True,
                    "evidence_id": "scope-1",
                }
            ],
        },
    )
    assert decision.allowed


def test_combined_authority_surfaces_sensitive_patch_review() -> None:
    authority = default_security_authority()
    decision = authority.evaluate_proposal(
        claims=[
            PatchPathClaim(
                path="swissknife/web/js/apps/agent-supervisor.js",
                change_kinds=(ForbiddenChangeKind.CONFIRMATION_WEAKENING,),
            )
        ],
        acceptance={
            "intended_action_id": "weaken_confirm",
            "policy_decision_id": "policy-ok",
            "policy_fresh": True,
        },
    )
    assert decision.requires_human_review
    assert (
        AuthorityReasonCode.SENSITIVE_CHANGE_REQUIRES_REVIEW.value
        in decision.reason_codes
        or ForbiddenChangeKind.CONFIRMATION_WEAKENING.value
        in decision.reason_codes
    )


def test_authority_decision_to_dict_is_stable() -> None:
    decision = GuiPatchAuthority().evaluate_path(
        "swissknife/web/js/apps/agent-supervisor.js"
    )
    payload = decision.to_dict()
    assert payload["interface"] == GUI_PATCH_AUTHORITY_INTERFACE
    assert payload["schema"] == GUI_PATCH_AUTHORITY_SCHEMA or payload["schema"]
    assert payload["verdict"] == "allow"
    assert payload["reason_codes"] == sorted(payload["reason_codes"])


def test_malformed_inputs_raise_gui_authority_error() -> None:
    with pytest.raises(GuiAuthorityError):
        GuiPatchAuthority(allowed_roots=())
    with pytest.raises(GuiAuthorityError):
        PatchPathClaim(path="/absolute")
    with pytest.raises(GuiAuthorityError):
        GuiPatchAuthority().evaluate_claims("not-a-sequence")  # type: ignore[arg-type]
    with pytest.raises(GuiAuthorityError):
        AuthorityEvidence(kind="not-a-real-kind", valid=True)  # type: ignore[arg-type]
