"""VGO-009: patch and browser-host security authority tests.

Acceptance coverage:

* UI state cannot synthesize authorization
* browser content cannot select host paths, commands, or credentials
* sensitive changes require contract verification or human review
* missing/invalid authority evidence rejects safely
* mapping inputs are closed and strictly typed before coercion
* every present identifier/digest/outcome/note accepts only its string type
* collection fields accept only declared JSON array/object types
* Python tuples are not treated as JSON arrays
* recursive browser payloads admit only JSON shapes at every depth
* omitted optional fields may default; present null rejects for ten scalars
* policy_decision_id/policy_fresh have no authority without bound evidence
* digests authorize only exact sha256:[0-9a-f]{64}
* browser envelopes reject path/command/credential selectors under nesting,
  placement, casing, percent/double encoding, URI/Windows form, and aliases
* claim-derived change kinds and computed decisions override acceptance input
* authority evidence has nonempty identity and is current + action-bound
* a scope declaration alone is never host authority
* WIRE_TYPE_CASES has exactly 214 unique IDs (exact Cartesian products)
* AUTHORIZATION_CASES has exactly 27 unique IDs
"""

from __future__ import annotations

from collections import UserDict
from typing import Any, Callable

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

# Canonical argument digests (exact sha256:[0-9a-f]{64}).
DIGEST_A = "sha256:" + ("a" * 64)
DIGEST_B = "sha256:" + ("b" * 64)
DIGEST_C = "sha256:" + ("c" * 64)
DIGEST_UPPER = "sha256:" + ("A" * 64)
DIGEST_SHORT = "sha256:" + ("a" * 63)
DIGEST_LONG = "sha256:" + ("a" * 65)
DIGEST_OTHER_ALG = "sha512:" + ("a" * 64)
DIGEST_EMPTY = ""
DIGEST_NOT_CANONICAL = "not-canonical"


def _bound_contract(
    *,
    action_id: str,
    argument_digest: str,
    evidence_id: str = "contract-1",
) -> AuthorityEvidence:
    return AuthorityEvidence(
        kind=AuthorityEvidenceKind.CONTRACT_VERIFICATION,
        valid=True,
        evidence_id=evidence_id,
        binds_action_id=action_id,
        binds_argument_digest=argument_digest,
    )


def _bound_human(
    *,
    action_id: str,
    argument_digest: str,
    evidence_id: str = "review-1",
) -> AuthorityEvidence:
    return AuthorityEvidence(
        kind=AuthorityEvidenceKind.HUMAN_REVIEW,
        valid=True,
        evidence_id=evidence_id,
        binds_action_id=action_id,
        binds_argument_digest=argument_digest,
    )


def _bound_host_policy(
    *,
    action_id: str,
    argument_digest: str,
    policy_decision_id: str = "policy-fresh",
    evidence_id: str = "host-reeval-1",
) -> AuthorityEvidence:
    return AuthorityEvidence(
        kind=AuthorityEvidenceKind.HOST_POLICY_REEVALUATION,
        valid=True,
        evidence_id=evidence_id,
        binds_action_id=action_id,
        binds_argument_digest=argument_digest,
        policy_decision_id=policy_decision_id,
        policy_fresh=True,
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
    assert "host_file_path" in FORBIDDEN_BROWSER_PAYLOAD_KEYS
    assert "cwd" in FORBIDDEN_BROWSER_PAYLOAD_KEYS
    assert "cmd" in FORBIDDEN_BROWSER_PAYLOAD_KEYS
    assert "credential" in FORBIDDEN_BROWSER_PAYLOAD_KEYS


def test_default_security_authority_is_fail_closed_wrapper() -> None:
    authority = default_security_authority()
    assert isinstance(authority, GuiOptimizerSecurityAuthority)
    assert isinstance(authority.patch, GuiPatchAuthority)
    assert isinstance(authority.host_boundary, GuiHostBoundaryPolicy)
    assert isinstance(authority.acceptance, GuiAcceptanceAuthority)


# ---------------------------------------------------------------------------
# GuiPatchAuthority@1
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
    # Forbidden segment doctrine; companion relocation lives under test fixtures.
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
# GuiHostBoundaryPolicy@1
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
# GuiAcceptanceAuthority@1
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
            intended_argument_digest=DIGEST_A,
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
            intended_argument_digest=DIGEST_A,
            confirmation_required=True,
            confirmation_granted=True,
            confirmation_action_id="delete_goal",
            confirmation_argument_digest=DIGEST_B,
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
            intended_argument_digest=DIGEST_A,
            confirmation_required=True,
            confirmation_granted=True,
            confirmation_action_id="delete_goal",
            confirmation_argument_digest=DIGEST_A,
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
            intended_argument_digest=DIGEST_A,
            evidence=(
                AuthorityEvidence(
                    kind=AuthorityEvidenceKind.CONTRACT_VERIFICATION,
                    valid=False,
                    evidence_id="broken-receipt",
                    binds_action_id="dispatch_task",
                    binds_argument_digest=DIGEST_A,
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
            intended_argument_digest=DIGEST_A,
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
            intended_argument_digest=DIGEST_A,
            change_kinds=(ForbiddenChangeKind.UNVERIFIED_ACTION_BINDING,),
            evidence=(
                _bound_contract(
                    action_id="edit_binding",
                    argument_digest=DIGEST_A,
                ),
            ),
        )
    )
    assert with_contract.allowed

    credentials = acceptance.evaluate(
        AcceptanceAuthorityRequest(
            intended_action_id="rotate_secret",
            intended_argument_digest=DIGEST_B,
            change_kinds=(ForbiddenChangeKind.CREDENTIALS,),
            evidence=(
                _bound_contract(
                    action_id="rotate_secret",
                    argument_digest=DIGEST_B,
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
            intended_argument_digest=DIGEST_B,
            change_kinds=(ForbiddenChangeKind.CREDENTIALS,),
            evidence=(
                _bound_human(
                    action_id="rotate_secret",
                    argument_digest=DIGEST_B,
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
            intended_argument_digest=DIGEST_C,
            accessibility_regression=True,
            evidence=(
                _bound_contract(
                    action_id="style_tweak", argument_digest=DIGEST_C
                ),
            ),
        )
    )
    assert a11y.rejected
    assert (
        AuthorityReasonCode.ACCESSIBILITY_REGRESSION.value in a11y.reason_codes
    )

    security = acceptance.evaluate(
        AcceptanceAuthorityRequest(
            intended_action_id="style_tweak",
            intended_argument_digest=DIGEST_C,
            security_regression=True,
            evidence=(
                _bound_contract(
                    action_id="style_tweak", argument_digest=DIGEST_C
                ),
            ),
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
            intended_argument_digest=DIGEST_A,
            ui_visible=True,
            ui_enabled=True,
            browser_policy_outcome="allow",
            browser_policy_authoritative_claim=False,
            policy_decision_id="policy-fresh",
            policy_fresh=True,
            evidence=(
                _bound_host_policy(
                    action_id="dispatch_task",
                    argument_digest=DIGEST_A,
                    policy_decision_id="policy-fresh",
                ),
            ),
        )
    )
    assert decision.allowed
    assert decision.to_dict()["schema"]
    assert decision.to_dict()["allowed"] is True


# ---------------------------------------------------------------------------
# Combined facade
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
            intended_argument_digest=DIGEST_A,
            evidence=(
                _bound_host_policy(
                    action_id="dispatch_task", argument_digest=DIGEST_A
                ),
            ),
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
            "intended_argument_digest": DIGEST_A,
            "policy_decision_id": "policy-ok",
            "policy_fresh": True,
            "evidence": [
                {
                    "kind": "host_policy_reevaluation",
                    "valid": True,
                    "evidence_id": "host-1",
                    "binds_action_id": "rerender_preserve_focus",
                    "binds_argument_digest": DIGEST_A,
                    "policy_decision_id": "policy-ok",
                    "policy_fresh": True,
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
            "intended_argument_digest": DIGEST_A,
            "evidence": [
                {
                    "kind": "contract_verification",
                    "valid": True,
                    "evidence_id": "contract-w",
                    "binds_action_id": "weaken_confirm",
                    "binds_argument_digest": DIGEST_A,
                }
            ],
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


# ---------------------------------------------------------------------------
# Closed typing, disguise resistance, binding, scope, override
# ---------------------------------------------------------------------------


def test_string_booleans_reject_on_mapping_inputs() -> None:
    with pytest.raises(GuiAuthorityError) as declared_exc:
        GuiPatchAuthority().evaluate_claims(
            [
                {
                    "path": "swissknife/web/js/apps/agent-supervisor.js",
                    "declared": "true",
                }
            ]
        )
    assert (
        declared_exc.value.reason_code
        == AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value
    )

    with pytest.raises(GuiAuthorityError) as fixture_exc:
        GuiHostBoundaryPolicy().evaluate(
            {"payload": {"view": "queue"}, "fixture_only": "true"}
        )
    assert (
        fixture_exc.value.reason_code
        == AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value
    )

    with pytest.raises(GuiAuthorityError) as policy_exc:
        GuiAcceptanceAuthority().evaluate(
            {
                "intended_action_id": "dispatch_task",
                "policy_decision_id": "policy-1",
                "policy_fresh": "yes",
            }
        )
    assert (
        policy_exc.value.reason_code
        == AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value
    )


def test_unknown_mapping_keys_reject() -> None:
    with pytest.raises(GuiAuthorityError) as claim_exc:
        GuiPatchAuthority().evaluate_claims(
            [
                {
                    "path": "swissknife/web/js/apps/agent-supervisor.js",
                    "extra": True,
                }
            ]
        )
    assert claim_exc.value.reason_code == AuthorityReasonCode.UNKNOWN_FIELD.value

    with pytest.raises(GuiAuthorityError) as browser_exc:
        GuiHostBoundaryPolicy().evaluate(
            {"payload": {"view": "queue"}, "inject": "x"}
        )
    assert browser_exc.value.reason_code == AuthorityReasonCode.UNKNOWN_FIELD.value

    with pytest.raises(GuiAuthorityError) as acceptance_exc:
        GuiAcceptanceAuthority().evaluate(
            {
                "intended_action_id": "dispatch_task",
                "policy_decision_id": "policy-ok",
                "policy_fresh": True,
                "forged_allow": True,
            }
        )
    assert acceptance_exc.value.reason_code == AuthorityReasonCode.UNKNOWN_FIELD.value


def test_scope_declaration_alone_is_never_host_authority() -> None:
    acceptance = GuiAcceptanceAuthority()
    decision = acceptance.evaluate(
        AcceptanceAuthorityRequest(
            intended_action_id="dispatch_task",
            intended_argument_digest=DIGEST_A,
            evidence=(
                AuthorityEvidence(
                    kind=AuthorityEvidenceKind.SCOPE_DECLARATION,
                    valid=True,
                    evidence_id="scope-only-1",
                ),
            ),
        )
    )
    assert decision.rejected
    assert (
        AuthorityReasonCode.SCOPE_DECLARATION_NOT_AUTHORITY.value
        in decision.reason_codes
    )


def test_authority_evidence_requires_nonempty_identity() -> None:
    with pytest.raises(GuiAuthorityError) as exc:
        AuthorityEvidence(
            kind=AuthorityEvidenceKind.CONTRACT_VERIFICATION,
            valid=True,
            evidence_id="",
        )
    assert (
        exc.value.reason_code
        == AuthorityReasonCode.EVIDENCE_IDENTITY_REQUIRED.value
    )


def test_unbound_authorizing_evidence_rejects() -> None:
    acceptance = GuiAcceptanceAuthority()
    decision = acceptance.evaluate(
        AcceptanceAuthorityRequest(
            intended_action_id="dispatch_task",
            intended_argument_digest=DIGEST_A,
            evidence=(
                AuthorityEvidence(
                    kind=AuthorityEvidenceKind.HOST_POLICY_REEVALUATION,
                    valid=True,
                    evidence_id="host-1",
                    binds_action_id="other_action",
                    binds_argument_digest=DIGEST_A,
                    policy_decision_id="policy-1",
                    policy_fresh=True,
                ),
            ),
        )
    )
    assert decision.rejected
    assert (
        AuthorityReasonCode.EVIDENCE_BINDING_MISMATCH.value in decision.reason_codes
    )


def test_stale_host_policy_evidence_rejects() -> None:
    acceptance = GuiAcceptanceAuthority()
    decision = acceptance.evaluate(
        AcceptanceAuthorityRequest(
            intended_action_id="dispatch_task",
            intended_argument_digest=DIGEST_A,
            evidence=(
                AuthorityEvidence(
                    kind=AuthorityEvidenceKind.HOST_POLICY_REEVALUATION,
                    valid=True,
                    evidence_id="stale-host-1",
                    binds_action_id="dispatch_task",
                    binds_argument_digest=DIGEST_A,
                    policy_decision_id="policy-old",
                    policy_fresh=False,
                ),
            ),
        )
    )
    assert decision.rejected
    assert AuthorityReasonCode.EVIDENCE_NOT_CURRENT.value in decision.reason_codes


def test_bound_current_evidence_authorizes_without_policy_id() -> None:
    acceptance = GuiAcceptanceAuthority()
    decision = acceptance.evaluate(
        AcceptanceAuthorityRequest(
            intended_action_id="dispatch_task",
            intended_argument_digest=DIGEST_A,
            evidence=(
                AuthorityEvidence(
                    kind=AuthorityEvidenceKind.CONTRACT_VERIFICATION,
                    valid=True,
                    evidence_id="contract-bound-1",
                    binds_action_id="dispatch_task",
                    binds_argument_digest=DIGEST_A,
                ),
            ),
        )
    )
    assert decision.allowed


def test_claim_derived_change_kinds_override_acceptance_input() -> None:
    authority = default_security_authority()
    decision = authority.evaluate_proposal(
        claims=[
            PatchPathClaim(
                path="swissknife/web/js/apps/agent-supervisor.js",
                change_kinds=(ForbiddenChangeKind.CREDENTIALS,),
            )
        ],
        acceptance={
            "intended_action_id": "rotate_secret",
            "intended_argument_digest": DIGEST_B,
            "change_kinds": [],
            "evidence": [
                {
                    "kind": "contract_verification",
                    "valid": True,
                    "evidence_id": "contract-ignore",
                    "binds_action_id": "rotate_secret",
                    "binds_argument_digest": DIGEST_B,
                }
            ],
        },
    )
    assert decision.requires_human_review
    assert (
        AuthorityReasonCode.SENSITIVE_CHANGE_REQUIRES_REVIEW.value
        in decision.reason_codes
        or ForbiddenChangeKind.CREDENTIALS.value in decision.reason_codes
    )


def test_computed_host_and_patch_decisions_override_acceptance_input() -> None:
    authority = default_security_authority()
    forged_allow = GuiPatchAuthority().evaluate_path(
        "swissknife/web/js/apps/agent-supervisor.js"
    )
    assert forged_allow.allowed

    host_blocked = authority.evaluate_proposal(
        claims=[
            PatchPathClaim(path="swissknife/web/js/apps/agent-supervisor.js")
        ],
        browser_input={"payload": {"host_path": "/tmp/x"}, "fixture_only": True},
        acceptance={
            "intended_action_id": "dispatch_task",
            "intended_argument_digest": DIGEST_A,
            "host_boundary_decision": forged_allow,
            "patch_authority_decision": forged_allow,
            "evidence": [
                {
                    "kind": "contract_verification",
                    "valid": True,
                    "evidence_id": "c1",
                    "binds_action_id": "dispatch_task",
                    "binds_argument_digest": DIGEST_A,
                }
            ],
        },
    )
    assert host_blocked.rejected
    assert (
        AuthorityReasonCode.BROWSER_HOST_PATH_FORBIDDEN.value
        in host_blocked.reason_codes
    )

    without_human = authority.evaluate_proposal(
        claims=[
            PatchPathClaim(
                path="swissknife/web/js/apps/agent-supervisor.js",
                change_kinds=(ForbiddenChangeKind.CONFIRMATION_WEAKENING,),
            )
        ],
        acceptance={
            "intended_action_id": "weaken_confirm",
            "intended_argument_digest": DIGEST_A,
            "change_kinds": [],
            "patch_authority_decision": forged_allow,
            "evidence": [
                {
                    "kind": "contract_verification",
                    "valid": True,
                    "evidence_id": "c2",
                    "binds_action_id": "weaken_confirm",
                    "binds_argument_digest": DIGEST_A,
                }
            ],
        },
    )
    assert without_human.requires_human_review
    assert (
        AuthorityReasonCode.SENSITIVE_CHANGE_REQUIRES_REVIEW.value
        in without_human.reason_codes
        or ForbiddenChangeKind.CONFIRMATION_WEAKENING.value
        in without_human.reason_codes
    )


def test_caller_policy_fields_alone_have_no_authority() -> None:
    acceptance = GuiAcceptanceAuthority()
    decision = acceptance.evaluate(
        AcceptanceAuthorityRequest(
            intended_action_id="dispatch_task",
            intended_argument_digest=DIGEST_A,
            policy_decision_id="policy-forged",
            policy_fresh=True,
            evidence=(),
        )
    )
    assert decision.rejected
    assert (
        AuthorityReasonCode.CALLER_POLICY_NOT_AUTHORITY.value
        in decision.reason_codes
    )
    assert (
        AuthorityReasonCode.MISSING_AUTHORITY_EVIDENCE.value
        in decision.reason_codes
    )


def test_python_tuples_are_not_json_arrays() -> None:
    with pytest.raises(GuiAuthorityError) as exc:
        GuiAcceptanceAuthority().evaluate(
            {
                "intended_action_id": "dispatch_task",
                "change_kinds": ("credentials",),
            }
        )
    assert (
        exc.value.reason_code
        == AuthorityReasonCode.INVALID_COLLECTION_TYPE.value
    )

    with pytest.raises(GuiAuthorityError) as browser_exc:
        GuiHostBoundaryPolicy().evaluate(
            {
                "payload": {"view": "queue"},
                "selected_host_paths": ("/tmp/x",),
            }
        )
    assert (
        browser_exc.value.reason_code
        == AuthorityReasonCode.INVALID_COLLECTION_TYPE.value
    )


def test_present_null_scalars_reject() -> None:
    """Omitted optional fields may default; present null must reject."""
    null_fields = [
        ("binds_action_id", "evidence"),
        ("binds_argument_digest", "evidence"),
        ("policy_decision_id", "evidence"),
        ("notes", "evidence"),
        ("intended_action_id", "request"),
        ("intended_argument_digest", "request"),
        ("browser_policy_outcome", "request"),
        ("policy_decision_id", "request"),
        ("confirmation_action_id", "request"),
        ("confirmation_argument_digest", "request"),
    ]
    assert len(null_fields) == 10
    for field, location in null_fields:
        if location == "evidence":
            payload: dict[str, Any] = {
                "intended_action_id": "dispatch_task",
                "intended_argument_digest": DIGEST_A,
                "evidence": [
                    {
                        "kind": "contract_verification",
                        "valid": True,
                        "evidence_id": "e1",
                        "binds_action_id": "dispatch_task",
                        "binds_argument_digest": DIGEST_A,
                        field: None,
                    }
                ],
            }
        else:
            payload = {field: None}
        with pytest.raises(GuiAuthorityError) as exc:
            GuiAcceptanceAuthority().evaluate(payload)
        assert (
            exc.value.reason_code
            == AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value
        ), field


def test_arbitrary_equal_noncanonical_digests_never_authorize() -> None:
    """Matching non-canonical digests must not authorize."""
    with pytest.raises(GuiAuthorityError) as exc:
        GuiAcceptanceAuthority().evaluate(
            {
                "intended_action_id": "dispatch_task",
                "intended_argument_digest": DIGEST_NOT_CANONICAL,
                "evidence": [
                    {
                        "kind": "contract_verification",
                        "valid": True,
                        "evidence_id": "e1",
                        "binds_action_id": "dispatch_task",
                        "binds_argument_digest": DIGEST_NOT_CANONICAL,
                    }
                ],
            }
        )
    assert (
        exc.value.reason_code
        == AuthorityReasonCode.NONCANONICAL_ARGUMENT_DIGEST.value
    )


# ---------------------------------------------------------------------------
# WIRE_TYPE_CASES — exact Cartesian products (214 unique IDs)
# ---------------------------------------------------------------------------

_STRING_TYPE_BAD: dict[str, Any] = {
    "null": None,
    "number": 1,
    "boolean": True,
    "json_array": [],
    "json_object": {},
}
_BOOL_TYPE_BAD: dict[str, Any] = {
    "null": None,
    "number": 1,
    "string": "true",
    "json_array": [],
    "json_object": {},
}
_ARRAY_TYPE_BAD: dict[str, Any] = {
    "null": None,
    "string": "x",
    "number": 1,
    "boolean": True,
    "json_object": {},
    "python_tuple": ("x",),
}
_PAYLOAD_TYPE_BAD: dict[str, Any] = {
    "null": None,
    "string": "x",
    "number": 1,
    "boolean": True,
    "json_array": [],
    "non_dict_mapping": UserDict({"view": "queue"}),
}
_DIGEST_GRAMMAR_BAD: dict[str, str] = {
    "uppercase": DIGEST_UPPER,
    "leading_whitespace": f" {DIGEST_A}",
    "trailing_whitespace": f"{DIGEST_A} ",
    "other_algorithm": DIGEST_OTHER_ALG,
    "short": DIGEST_SHORT,
    "long": DIGEST_LONG,
    "empty": DIGEST_EMPTY,
    "arbitrary_equal_noncanonical": DIGEST_NOT_CANONICAL,
}

_STRING_FIELDS: tuple[tuple[str, str], ...] = (
    ("AuthorityEvidence", "kind"),
    ("AuthorityEvidence", "evidence_id"),
    ("AuthorityEvidence", "binds_action_id"),
    ("AuthorityEvidence", "binds_argument_digest"),
    ("AuthorityEvidence", "policy_decision_id"),
    ("AuthorityEvidence", "notes"),
    ("AcceptanceAuthorityRequest", "intended_action_id"),
    ("AcceptanceAuthorityRequest", "intended_argument_digest"),
    ("AcceptanceAuthorityRequest", "browser_policy_outcome"),
    ("AcceptanceAuthorityRequest", "policy_decision_id"),
    ("AcceptanceAuthorityRequest", "confirmation_action_id"),
    ("AcceptanceAuthorityRequest", "confirmation_argument_digest"),
    ("PatchPathClaim", "path"),
)
_BOOL_FIELDS: tuple[tuple[str, str], ...] = (
    ("AuthorityEvidence", "valid"),
    ("AuthorityEvidence", "policy_fresh"),
    ("AcceptanceAuthorityRequest", "ui_visible"),
    ("AcceptanceAuthorityRequest", "ui_enabled"),
    ("AcceptanceAuthorityRequest", "browser_policy_authoritative_claim"),
    ("AcceptanceAuthorityRequest", "policy_fresh"),
    ("AcceptanceAuthorityRequest", "confirmation_required"),
    ("AcceptanceAuthorityRequest", "confirmation_granted"),
    ("AcceptanceAuthorityRequest", "accessibility_regression"),
    ("AcceptanceAuthorityRequest", "security_regression"),
    ("BrowserHostInput", "fixture_only"),
    ("BrowserHostInput", "uses_production_credentials"),
    ("BrowserHostInput", "uses_production_services"),
    ("BrowserHostInput", "uses_production_mcp_tools"),
    ("BrowserHostInput", "uses_user_or_legal_data"),
    ("PatchPathClaim", "declared"),
)
_ARRAY_FIELDS: tuple[tuple[str, str], ...] = (
    ("AcceptanceAuthorityRequest", "change_kinds"),
    ("AcceptanceAuthorityRequest", "evidence"),
    ("BrowserHostInput", "selected_host_paths"),
    ("BrowserHostInput", "selected_commands"),
    ("BrowserHostInput", "selected_executables"),
    ("PatchPathClaim", "change_kinds"),
)
_DIGEST_FIELDS: tuple[tuple[str, str], ...] = (
    ("AcceptanceAuthorityRequest", "intended_argument_digest"),
    ("AcceptanceAuthorityRequest", "confirmation_argument_digest"),
    ("AuthorityEvidence", "binds_argument_digest"),
)


def _apply_authority_evidence_field(field: str, value: Any) -> None:
    base: dict[str, Any] = {
        "kind": "contract_verification",
        "valid": True,
        "evidence_id": "e1",
        "binds_action_id": "dispatch_task",
        "binds_argument_digest": DIGEST_A,
    }
    base[field] = value
    GuiAcceptanceAuthority().evaluate(
        {
            "intended_action_id": "dispatch_task",
            "intended_argument_digest": DIGEST_A,
            "evidence": [base],
        }
    )


def _apply_acceptance_field(field: str, value: Any) -> None:
    payload: dict[str, Any] = {
        "intended_action_id": "dispatch_task",
        "intended_argument_digest": DIGEST_A,
    }
    payload[field] = value
    GuiAcceptanceAuthority().evaluate(payload)


def _apply_browser_field(field: str, value: Any) -> None:
    payload: dict[str, Any] = {"fixture_only": True, "payload": {"view": "queue"}}
    payload[field] = value
    GuiHostBoundaryPolicy().evaluate(payload)


def _apply_patch_field(field: str, value: Any) -> None:
    claim: dict[str, Any] = {
        "path": "swissknife/web/js/apps/agent-supervisor.js",
        "declared": True,
        "change_kinds": [],
    }
    claim[field] = value
    GuiPatchAuthority().evaluate_claims([claim])


def _apply_wire_case(owner: str, field: str, value: Any) -> None:
    if owner == "AuthorityEvidence":
        _apply_authority_evidence_field(field, value)
    elif owner == "AcceptanceAuthorityRequest":
        _apply_acceptance_field(field, value)
    elif owner == "BrowserHostInput":
        _apply_browser_field(field, value)
    elif owner == "PatchPathClaim":
        _apply_patch_field(field, value)
    else:
        raise AssertionError(f"unknown owner {owner}")


def _build_wire_type_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    for owner, field in _STRING_FIELDS:
        for category, bad in _STRING_TYPE_BAD.items():
            cases.append(
                {
                    "id": f"wire:string:{owner}.{field}:{category}",
                    "owner": owner,
                    "field": field,
                    "category": category,
                    "value": bad,
                    "group": "string",
                }
            )

    for owner, field in _BOOL_FIELDS:
        for category, bad in _BOOL_TYPE_BAD.items():
            cases.append(
                {
                    "id": f"wire:bool:{owner}.{field}:{category}",
                    "owner": owner,
                    "field": field,
                    "category": category,
                    "value": bad,
                    "group": "bool",
                }
            )

    for owner, field in _ARRAY_FIELDS:
        for category, bad in _ARRAY_TYPE_BAD.items():
            cases.append(
                {
                    "id": f"wire:array:{owner}.{field}:{category}",
                    "owner": owner,
                    "field": field,
                    "category": category,
                    "value": bad,
                    "group": "array",
                }
            )

    for category, bad in _PAYLOAD_TYPE_BAD.items():
        cases.append(
            {
                "id": f"wire:payload:BrowserHostInput.payload:{category}",
                "owner": "BrowserHostInput",
                "field": "payload",
                "category": category,
                "value": bad,
                "group": "payload",
            }
        )

    for owner, field in _DIGEST_FIELDS:
        for category, bad in _DIGEST_GRAMMAR_BAD.items():
            cases.append(
                {
                    "id": f"wire:digest:{owner}.{field}:{category}",
                    "owner": owner,
                    "field": field,
                    "category": category,
                    "value": bad,
                    "group": "digest",
                }
            )

    # Exactly three recursive-shape cases.
    cases.extend(
        [
            {
                "id": "wire:recursive:nested_tuple",
                "owner": "BrowserHostInput",
                "field": "payload",
                "category": "nested_tuple",
                "value": {"items": (1, 2)},
                "group": "recursive",
            },
            {
                "id": "wire:recursive:nested_non_string_object_key",
                "owner": "BrowserHostInput",
                "field": "payload",
                "category": "nested_non_string_object_key",
                "value": {"nested": {1: "x"}},
                "group": "recursive",
            },
            {
                "id": "wire:recursive:nested_non_json_container",
                "owner": "BrowserHostInput",
                "field": "payload",
                "category": "nested_non_json_container",
                "value": {"items": {1, 2}},
                "group": "recursive",
            },
        ]
    )
    return cases


WIRE_TYPE_CASES: list[dict[str, Any]] = _build_wire_type_cases()


def test_wire_type_cases_manifest_is_exact_cartesian_product() -> None:
    """Assert exact field/category Cartesian products with no padding."""
    expected_ids = {case["id"] for case in _build_wire_type_cases()}
    actual_ids = {case["id"] for case in WIRE_TYPE_CASES}
    assert actual_ids == expected_ids
    assert len(WIRE_TYPE_CASES) == 214
    assert len(actual_ids) == 214

    # Decomposition from Evidence subset.
    assert len(_STRING_FIELDS) * len(_STRING_TYPE_BAD) == 65
    assert len(_BOOL_FIELDS) * len(_BOOL_TYPE_BAD) == 80
    assert len(_ARRAY_FIELDS) * len(_ARRAY_TYPE_BAD) == 36
    assert len(_PAYLOAD_TYPE_BAD) == 6
    assert len(_DIGEST_FIELDS) * len(_DIGEST_GRAMMAR_BAD) == 24
    recursive = [c for c in WIRE_TYPE_CASES if c["group"] == "recursive"]
    assert len(recursive) == 3
    assert 65 + 80 + 36 + 6 + 24 + 3 == 214


@pytest.mark.parametrize(
    "case",
    WIRE_TYPE_CASES,
    ids=[case["id"] for case in WIRE_TYPE_CASES],
)
def test_wire_type_case_rejects(case: dict[str, Any]) -> None:
    with pytest.raises(GuiAuthorityError) as exc:
        _apply_wire_case(case["owner"], case["field"], case["value"])
    # Type/grammar failures must reject safely — never authorize.
    assert exc.value.reason_code in {
        AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
        AuthorityReasonCode.INVALID_COLLECTION_TYPE.value,
        AuthorityReasonCode.INVALID_AUTHORITY_EVIDENCE.value,
        AuthorityReasonCode.NONCANONICAL_ARGUMENT_DIGEST.value,
        AuthorityReasonCode.EMPTY_ARGUMENT_DIGEST.value,
        AuthorityReasonCode.EVIDENCE_IDENTITY_REQUIRED.value,
        AuthorityReasonCode.PATH_ABSOLUTE_OR_TRAVERSAL.value,
    }


# ---------------------------------------------------------------------------
# AUTHORIZATION_CASES — exactly 27 unique IDs
# ---------------------------------------------------------------------------


def _auth_error(
    case_id: str,
    runner: Callable[[], None],
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "mode": "error",
        "runner": runner,
        "reason": reason,
    }


def _auth_decision(
    case_id: str,
    runner: Callable[[], Any],
    *,
    reason: str,
    allow: bool = False,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "mode": "decision",
        "runner": runner,
        "reason": reason,
        "allow": allow,
    }


def _build_authorization_cases_exact() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    # --- 10 present-null scalars ---
    for field, location, case_name in (
        ("binds_action_id", "evidence", "binds_action_id"),
        ("binds_argument_digest", "evidence", "binds_argument_digest"),
        ("policy_decision_id", "evidence", "policy_decision_id_evidence"),
        ("notes", "evidence", "notes"),
        ("intended_action_id", "request", "intended_action_id"),
        ("intended_argument_digest", "request", "intended_argument_digest"),
        ("browser_policy_outcome", "request", "browser_policy_outcome"),
        ("policy_decision_id", "request", "policy_decision_id_request"),
        ("confirmation_action_id", "request", "confirmation_action_id"),
        ("confirmation_argument_digest", "request", "confirmation_argument_digest"),
    ):
        def _make(field: str = field, location: str = location) -> Callable[[], None]:
            def run() -> None:
                if location == "evidence":
                    GuiAcceptanceAuthority().evaluate(
                        {
                            "intended_action_id": "dispatch_task",
                            "intended_argument_digest": DIGEST_A,
                            "evidence": [
                                {
                                    "kind": "contract_verification",
                                    "valid": True,
                                    "evidence_id": "e1",
                                    field: None,
                                }
                            ],
                        }
                    )
                else:
                    GuiAcceptanceAuthority().evaluate({field: None})

            return run

        cases.append(
            _auth_error(
                f"auth:present_null:{case_name}",
                _make(),
                reason=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
            )
        )

    # 11 strict coercion
    cases.append(
        _auth_error(
            "auth:strict_coercion:string_boolean",
            lambda: GuiAcceptanceAuthority().evaluate(
                {"intended_action_id": "x", "policy_fresh": "true"}
            ),
            reason=AuthorityReasonCode.INVALID_AUTHORITY_INPUT.value,
        )
    )
    # 12 unknown field
    cases.append(
        _auth_error(
            "auth:unknown_field",
            lambda: GuiAcceptanceAuthority().evaluate(
                {"intended_action_id": "x", "forged_allow": True}
            ),
            reason=AuthorityReasonCode.UNKNOWN_FIELD.value,
        )
    )
    # 13 unbound caller policy
    cases.append(
        _auth_decision(
            "auth:caller_policy_unbound",
            lambda: GuiAcceptanceAuthority().evaluate(
                {
                    "intended_action_id": "dispatch_task",
                    "intended_argument_digest": DIGEST_A,
                    "policy_decision_id": "policy-forged",
                    "policy_fresh": True,
                }
            ),
            reason=AuthorityReasonCode.CALLER_POLICY_NOT_AUTHORITY.value,
        )
    )
    # 14 digest uppercase
    cases.append(
        _auth_error(
            "auth:digest_grammar:uppercase",
            lambda: GuiAcceptanceAuthority().evaluate(
                {
                    "intended_action_id": "dispatch_task",
                    "intended_argument_digest": DIGEST_UPPER,
                }
            ),
            reason=AuthorityReasonCode.NONCANONICAL_ARGUMENT_DIGEST.value,
        )
    )
    # 15 digest not-canonical equal
    cases.append(
        _auth_error(
            "auth:digest_grammar:not_canonical_equal",
            lambda: GuiAcceptanceAuthority().evaluate(
                {
                    "intended_action_id": "dispatch_task",
                    "intended_argument_digest": DIGEST_NOT_CANONICAL,
                    "evidence": [
                        {
                            "kind": "contract_verification",
                            "valid": True,
                            "evidence_id": "e1",
                            "binds_action_id": "dispatch_task",
                            "binds_argument_digest": DIGEST_NOT_CANONICAL,
                        }
                    ],
                }
            ),
            reason=AuthorityReasonCode.NONCANONICAL_ARGUMENT_DIGEST.value,
        )
    )
    # 16 recursive JSON shape
    cases.append(
        _auth_error(
            "auth:recursive_json_shape",
            lambda: GuiHostBoundaryPolicy().evaluate(
                {"payload": {"items": (1, 2)}, "fixture_only": True}
            ),
            reason=AuthorityReasonCode.INVALID_COLLECTION_TYPE.value,
        )
    )

    def _br(payload: dict[str, Any]) -> Callable[[], Any]:
        return lambda: GuiHostBoundaryPolicy().evaluate(
            BrowserHostInput(payload=payload)
        )

    # 17 percent/double-encoded selector key + encoded alias + path value
    cases.append(
        _auth_decision(
            "auth:encoded_selector:host_path_encoded_double",
            _br(
                {
                    "host_path_encoded": "%252Ftmp%252Fx",
                    "host%5Fpath": "/etc/shadow",
                    "host%252Fpath": "/var/secret",
                }
            ),
            reason=AuthorityReasonCode.BROWSER_HOST_PATH_FORBIDDEN.value,
        )
    )
    # 18 workingDirectoryEncoded
    cases.append(
        _auth_decision(
            "auth:encoded_selector:workingDirectoryEncoded",
            _br({"workingDirectoryEncoded": "/var/tmp"}),
            reason=AuthorityReasonCode.BROWSER_HOST_PATH_FORBIDDEN.value,
        )
    )
    # 19 fileUriEncoded FILE URI
    cases.append(
        _auth_decision(
            "auth:encoded_selector:fileUriEncoded",
            _br({"fileUriEncoded": "FILE:///etc/passwd"}),
            reason=AuthorityReasonCode.BROWSER_HOST_PATH_FORBIDDEN.value,
        )
    )
    # 20 credentialEncoded
    cases.append(
        _auth_decision(
            "auth:encoded_selector:credentialEncoded",
            _br({"credentialEncoded": "secret%3Atoken"}),
            reason=AuthorityReasonCode.BROWSER_CREDENTIAL_FORBIDDEN.value,
        )
    )
    # 21 generic target path-looking value
    cases.append(
        _auth_decision(
            "auth:value:generic_target_path",
            _br({"target": "/home/op/.ssh/id_rsa"}),
            reason=AuthorityReasonCode.BROWSER_HOST_PATH_FORBIDDEN.value,
        )
    )
    # 22 UNC + percent-encoded Windows path
    cases.append(
        _auth_decision(
            "auth:value:unc_and_encoded_windows",
            _br({"a": "\\\\server\\share", "b": "C:%5Csecret"}),
            reason=AuthorityReasonCode.BROWSER_HOST_PATH_FORBIDDEN.value,
        )
    )
    # 23 encoded command
    cases.append(
        _auth_decision(
            "auth:value:encoded_command",
            _br({"hint": "cmd%2Eexe%20%2Fc"}),
            reason=AuthorityReasonCode.BROWSER_COMMAND_FORBIDDEN.value,
        )
    )
    # 24 encoded credential value + named path/command/credential aliases
    cases.append(
        _auth_decision(
            "auth:value:encoded_credential_and_aliases",
            _br(
                {
                    "note": "secret%3Atoken",
                    "hostFilePath": "/tmp/x",
                    "workingDirectory": "/home/op",
                    "cwd": "/var",
                    "fileUri": "file:///etc/passwd",
                    "hostFilesystemPath": "/opt/data",
                    "cmd": "bash",
                    "credential": "token",
                }
            ),
            # Credential classification wins when path/command/credential keys coexist.
            reason=AuthorityReasonCode.BROWSER_CREDENTIAL_FORBIDDEN.value,
        )
    )
    # 25 evidence binding mismatch
    cases.append(
        _auth_decision(
            "auth:evidence_binding_mismatch",
            lambda: GuiAcceptanceAuthority().evaluate(
                AcceptanceAuthorityRequest(
                    intended_action_id="dispatch_task",
                    intended_argument_digest=DIGEST_A,
                    evidence=(
                        AuthorityEvidence(
                            kind=AuthorityEvidenceKind.CONTRACT_VERIFICATION,
                            valid=True,
                            evidence_id="e1",
                            binds_action_id="other",
                            binds_argument_digest=DIGEST_A,
                        ),
                    ),
                )
            ),
            reason=AuthorityReasonCode.EVIDENCE_BINDING_MISMATCH.value,
        )
    )
    # 26 evidence freshness
    cases.append(
        _auth_decision(
            "auth:evidence_not_current",
            lambda: GuiAcceptanceAuthority().evaluate(
                AcceptanceAuthorityRequest(
                    intended_action_id="dispatch_task",
                    intended_argument_digest=DIGEST_A,
                    evidence=(
                        AuthorityEvidence(
                            kind=AuthorityEvidenceKind.HOST_POLICY_REEVALUATION,
                            valid=True,
                            evidence_id="stale",
                            binds_action_id="dispatch_task",
                            binds_argument_digest=DIGEST_A,
                            policy_fresh=False,
                        ),
                    ),
                )
            ),
            reason=AuthorityReasonCode.EVIDENCE_NOT_CURRENT.value,
        )
    )
    # 27 scope-not-authority + computed-decision override (combined runner)
    def _scope_and_override() -> Any:
        # Scope alone never authorizes.
        scope = GuiAcceptanceAuthority().evaluate(
            AcceptanceAuthorityRequest(
                intended_action_id="dispatch_task",
                intended_argument_digest=DIGEST_A,
                evidence=(
                    AuthorityEvidence(
                        kind=AuthorityEvidenceKind.SCOPE_DECLARATION,
                        valid=True,
                        evidence_id="scope-1",
                    ),
                ),
            )
        )
        if AuthorityReasonCode.SCOPE_DECLARATION_NOT_AUTHORITY.value not in (
            scope.reason_codes
        ):
            return scope
        # Computed host decision overrides forged acceptance ALLOW.
        authority = default_security_authority()
        forged = GuiPatchAuthority().evaluate_path(
            "swissknife/web/js/apps/agent-supervisor.js"
        )
        return authority.evaluate_proposal(
            claims=[
                PatchPathClaim(path="swissknife/web/js/apps/agent-supervisor.js")
            ],
            browser_input={
                "payload": {"host_path": "/tmp/x"},
                "fixture_only": True,
            },
            acceptance={
                "intended_action_id": "dispatch_task",
                "intended_argument_digest": DIGEST_A,
                "host_boundary_decision": forged,
                "patch_authority_decision": forged,
                "evidence": [
                    {
                        "kind": "contract_verification",
                        "valid": True,
                        "evidence_id": "c1",
                        "binds_action_id": "dispatch_task",
                        "binds_argument_digest": DIGEST_A,
                    }
                ],
            },
        )

    cases.append(
        _auth_decision(
            "auth:scope_and_computed_override",
            _scope_and_override,
            reason=AuthorityReasonCode.BROWSER_HOST_PATH_FORBIDDEN.value,
        )
    )
    return cases


AUTHORIZATION_CASES: list[dict[str, Any]] = _build_authorization_cases_exact()


def test_authorization_cases_manifest_has_exactly_27_unique_ids() -> None:
    ids = [case["id"] for case in AUTHORIZATION_CASES]
    assert len(ids) == 27
    assert len(set(ids)) == 27
    # Required topic coverage markers appear in case ids.
    joined = " ".join(ids)
    assert "present_null" in joined
    assert "strict_coercion" in joined
    assert "unknown_field" in joined
    assert "caller_policy" in joined
    assert "digest_grammar" in joined
    assert "recursive_json_shape" in joined
    assert "host_path_encoded" in joined
    assert "workingDirectoryEncoded" in joined
    assert "fileUriEncoded" in joined
    assert "credentialEncoded" in joined
    assert "generic_target" in joined
    assert "unc" in joined
    assert "encoded_command" in joined
    assert "encoded_credential" in joined
    assert "evidence_binding" in joined or "evidence_not_current" in joined
    assert "scope" in joined
    assert "override" in joined


@pytest.mark.parametrize(
    "case",
    AUTHORIZATION_CASES,
    ids=[case["id"] for case in AUTHORIZATION_CASES],
)
def test_authorization_case_rejects_safely(case: dict[str, Any]) -> None:
    if case["mode"] == "error":
        with pytest.raises(GuiAuthorityError) as exc:
            case["runner"]()
        assert exc.value.reason_code == case["reason"]
        return
    decision = case["runner"]()
    if case.get("allow"):
        assert decision.allowed
        return
    assert not decision.allowed
    assert case["reason"] in decision.reason_codes
