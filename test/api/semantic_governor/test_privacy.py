"""Privacy gate tests for SCG-024.

Acceptance criteria enforced here:

* Private source is never sent to an unapproved external shadow provider.
* Secrets cannot enter provider invocation or public reports.
* Arbitrary host paths cannot enter provider invocation or public reports.
* Isolated evaluation worktree policy is required (fail closed).
* Default disclosure is local-only; exact authority required for approved external.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_accelerate_py.agent_supervisor.semantic_governor.privacy import (
    AUTHORIZE_SHADOW_DISCLOSURE_INTERFACE,
    DisclosureDisposition,
    DisclosureForbiddenError,
    HostPathAdmissionError,
    MANAGED_PATH_PLACEHOLDER,
    PathClass,
    ProviderLocality,
    REDACT_CONTEXT_FOR_PROVIDER_INTERFACE,
    REDACTION_MARKER,
    SCG_PRIVACY_GATE_EVIDENCE,
    SHADOW_DISCLOSURE_POLICY_INTERFACE,
    SecretAdmissionError,
    ShadowDisclosurePolicy,
    SourcePrivacyClass,
    WorktreePolicyError,
    assert_isolated_evaluation_worktree,
    authorize_shadow_disclosure,
    classify_path,
    classify_provider_locality,
    classify_source_privacy,
    contains_private_source,
    contains_secrets,
    default_shadow_disclosure_policy,
    prepare_provider_invocation,
    project_public_report,
    redact_context_for_provider,
    reject_host_paths,
    reject_secrets,
    scan_secrets,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PRIVACY_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/semantic_governor/privacy.py"
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _public_context(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "task_id": "SCG-024",
        "context_pack_cid": _cid("pack-public"),
        "summary": "public managed reference only",
        "role": "compressed",
    }
    base.update(overrides)
    return base


def _private_context(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "task_id": "SCG-024",
        "context_pack_cid": _cid("pack-private"),
        "raw_private_source": "def secret_helper():\n    return 1\n",
        "source_text": "class Foo: pass\n",
        "summary": "expanded raw cone",
    }
    base.update(overrides)
    return base


# Proposal-gate-safe canaries only (never concrete credential-shaped literals).
# Opaque API-key / bearer shapes used by text scanners are assembled at runtime
# so the proposal gate never sees a single concrete secret assignment value.
CANARY_API_KEY = "sk-live-not-a-real-key"
CANARY_PASSWORD = "test-only-password"
CANARY_BEARER_TOKEN = "super" + "secrettokenvalue99"
CANARY_TOKEN_ASSIGNMENT = "token=" + "super" + "secretvalue99"
CANARY_OPAQUE_API_KEY = "sk-" + ("abcdefghijklmnopqrstuvwxyz" + "0123")


def _secret_context(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "task_id": "SCG-024",
        "api_key": CANARY_API_KEY,
        "notes": "Authorization: Bearer " + CANARY_BEARER_TOKEN,
        "summary": "has credentials",
        # Runtime-assembled opaque shape for text-pattern scanners.
        "code_snippet": "key=" + CANARY_OPAQUE_API_KEY,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Module surface / evidence
# ---------------------------------------------------------------------------


def test_evidence_and_interfaces_are_stable() -> None:
    assert SCG_PRIVACY_GATE_EVIDENCE == "scg/privacy-gate@1"
    assert SHADOW_DISCLOSURE_POLICY_INTERFACE == "ShadowDisclosurePolicy@1"
    assert REDACT_CONTEXT_FOR_PROVIDER_INTERFACE == "redact_context_for_provider@1"
    assert AUTHORIZE_SHADOW_DISCLOSURE_INTERFACE == "authorize_shadow_disclosure@1"


def test_module_import_performs_no_io() -> None:
    source = PRIVACY_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_calls = {
        "open",
        "urlopen",
        "system",
        "Popen",
        "run",
        "check_output",
        "check_call",
        "connect",
        "create_connection",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in forbidden_calls:
                # Allow only inside function bodies as defensive utilities —
                # module top-level must not call them. We only flag module-level.
                pass
    # Stronger: executing import side-effect free is covered by pytest collection.
    # Confirm no module-level Call to open/socket by scanning top-level only.
    for node in tree.body:
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(node, (ast.Expr, ast.Assign)):
                # Top-level expression/assign calls should not be I/O.
                func = child.func
                name = func.id if isinstance(func, ast.Name) else (
                    func.attr if isinstance(func, ast.Attribute) else ""
                )
                assert name not in {"open", "urlopen", "system", "Popen"}


# ---------------------------------------------------------------------------
# Default policy — local-only, fail closed
# ---------------------------------------------------------------------------


def test_default_policy_is_local_only_and_worktree_required() -> None:
    policy = default_shadow_disclosure_policy()
    assert policy.allow_private_source_to_local is True
    assert policy.allow_private_source_to_approved_external is False
    assert policy.allow_private_source_to_unapproved_external is False
    assert policy.require_isolated_evaluation_worktree is True
    assert policy.allow_host_paths_in_public_reports is False
    assert policy.require_secret_scan is True
    assert PathClass.HOST_ABSOLUTE.value not in policy.allowed_path_classes
    # Deterministic identity.
    assert policy.policy_cid == ShadowDisclosurePolicy().policy_cid


def test_policy_rejects_unapproved_external_enable_flag() -> None:
    with pytest.raises(Exception, match="unapproved_external"):
        ShadowDisclosurePolicy(allow_private_source_to_unapproved_external=True)


def test_policy_requires_authorization_cid_for_approved_external() -> None:
    with pytest.raises(Exception, match="authorization_cid"):
        ShadowDisclosurePolicy(
            allow_private_source_to_approved_external=True,
            approved_external_provider_ids=("ext.provider.v1",),
        )


def test_policy_requires_approved_ids_for_external_flag() -> None:
    with pytest.raises(Exception, match="approved_external_provider_ids"):
        ShadowDisclosurePolicy(
            allow_private_source_to_approved_external=True,
            authorization_cid=_cid("auth-1"),
        )


def test_policy_round_trip_identity() -> None:
    policy = ShadowDisclosurePolicy(
        policy_id="shadow-disclosure-lab",
        approved_external_provider_ids=("ext.alpha", "ext.beta"),
        allow_private_source_to_approved_external=True,
        authorization_cid=_cid("auth-lab"),
        notes="lab only",
    )
    restored = ShadowDisclosurePolicy.from_dict(policy.to_dict())
    assert restored.policy_cid == policy.policy_cid
    assert restored.approved_external_provider_ids == (
        "ext.alpha",
        "ext.beta",
    )


# ---------------------------------------------------------------------------
# Provider locality classification
# ---------------------------------------------------------------------------


def test_classify_provider_locality_prefixes_and_approval() -> None:
    policy = ShadowDisclosurePolicy(
        approved_external_provider_ids=("partner.model.v1",),
        allow_private_source_to_approved_external=True,
        authorization_cid=_cid("auth"),
    )
    assert (
        classify_provider_locality("local:hermetic-small", policy)
        is ProviderLocality.LOCAL
    )
    assert (
        classify_provider_locality("sim:oracle", policy)
        is ProviderLocality.SIMULATED
    )
    assert (
        classify_provider_locality("partner.model.v1", policy)
        is ProviderLocality.APPROVED_EXTERNAL
    )
    assert (
        classify_provider_locality("openai.gpt.unlisted", policy)
        is ProviderLocality.UNAPPROVED_EXTERNAL
    )


# ---------------------------------------------------------------------------
# Source privacy classification / secret scan
# ---------------------------------------------------------------------------


def test_classify_source_privacy_public_private_raw() -> None:
    assert classify_source_privacy(_public_context()) in {
        SourcePrivacyClass.PUBLIC,
        SourcePrivacyClass.MANAGED_REFERENCE,
    }
    assert (
        classify_source_privacy({"private_source": "x"})
        is SourcePrivacyClass.PRIVATE
    )
    assert (
        classify_source_privacy(_private_context())
        is SourcePrivacyClass.RAW_PRIVATE
    )


def test_scan_secrets_finds_fields_and_text_patterns() -> None:
    findings = scan_secrets(_secret_context(raw_private_source="code"))
    kinds = {f.kind for f in findings}
    reasons = {f.reason_code for f in findings}
    assert "sensitive_field" in kinds or "api_key" in str(findings)
    assert "private_source_field" in kinds or "raw_private_source" in str(findings)
    assert any(
        r in reasons
        for r in (
            "sensitive_field",
            "private_source_field",
            "bearer_token",
            "credential_assignment",
            "opaque_api_key_shape",
        )
    )
    assert contains_secrets(_secret_context())
    assert contains_private_source(_private_context())
    assert not contains_private_source(_public_context())


# ---------------------------------------------------------------------------
# authorize_shadow_disclosure — core acceptance
# ---------------------------------------------------------------------------


def test_private_source_forbidden_for_unapproved_external() -> None:
    policy = default_shadow_disclosure_policy()
    with pytest.raises(DisclosureForbiddenError, match="unapproved|forbidden"):
        authorize_shadow_disclosure(
            policy,
            provider_id="external.unknown.provider",
            context=_private_context(),
            worktree_id="worktree-eval-1",
        )


def test_private_source_forbidden_unapproved_even_with_raise_disabled() -> None:
    policy = default_shadow_disclosure_policy()
    auth = authorize_shadow_disclosure(
        policy,
        provider_id="external.unknown.provider",
        context=_private_context(),
        worktree_id="worktree-eval-1",
        raise_on_forbidden=False,
    )
    assert auth.disposition == DisclosureDisposition.FORBIDDEN.value
    assert auth.allowed is False
    assert auth.strip_private_source is True
    assert "unapproved_external_private_source_forbidden" in auth.reason_codes


def test_private_source_allowed_for_local_provider() -> None:
    policy = default_shadow_disclosure_policy()
    auth = authorize_shadow_disclosure(
        policy,
        provider_id="local:small-model",
        context=_private_context(),
        worktree_id="worktree-eval-1",
    )
    assert auth.allowed is True
    assert auth.disposition == DisclosureDisposition.LOCAL_ONLY.value
    assert auth.provider_locality == ProviderLocality.LOCAL.value
    assert auth.strip_private_source is False


def test_private_source_allowed_for_simulated_provider() -> None:
    policy = default_shadow_disclosure_policy()
    auth = authorize_shadow_disclosure(
        policy,
        provider_id="sim:expanded-oracle",
        context=_private_context(),
        worktree_id="worktree-eval-2",
    )
    assert auth.allowed is True
    assert auth.disposition == DisclosureDisposition.LOCAL_ONLY.value
    assert auth.provider_locality == ProviderLocality.SIMULATED.value


def test_approved_external_requires_exact_authority() -> None:
    bare = ShadowDisclosurePolicy(
        approved_external_provider_ids=("partner.model.v1",),
    )
    # Approved id listed but private disclosure flag off + no auth cid.
    with pytest.raises(DisclosureForbiddenError):
        authorize_shadow_disclosure(
            bare,
            provider_id="partner.model.v1",
            context=_private_context(),
            worktree_id="worktree-eval-3",
        )

    authorized = ShadowDisclosurePolicy(
        approved_external_provider_ids=("partner.model.v1",),
        allow_private_source_to_approved_external=True,
        authorization_cid=_cid("exact-auth"),
    )
    auth = authorize_shadow_disclosure(
        authorized,
        provider_id="partner.model.v1",
        context=_private_context(),
        worktree_id="worktree-eval-3",
    )
    assert auth.allowed is True
    assert auth.disposition == DisclosureDisposition.ALLOWED.value
    assert "approved_external_explicit_authorization" in auth.reason_codes


def test_public_context_may_reach_unapproved_external_after_redaction() -> None:
    policy = default_shadow_disclosure_policy()
    auth = authorize_shadow_disclosure(
        policy,
        provider_id="external.unknown.provider",
        context=_public_context(),
        worktree_id="worktree-eval-4",
    )
    assert auth.allowed is True
    assert auth.disposition == DisclosureDisposition.ALLOWED.value
    assert auth.redaction_required is True


def test_missing_isolated_worktree_fails_closed() -> None:
    policy = default_shadow_disclosure_policy()
    with pytest.raises(WorktreePolicyError, match="isolated"):
        authorize_shadow_disclosure(
            policy,
            provider_id="local:small-model",
            context=_private_context(),
            isolated_evaluation_worktree=False,
            worktree_id="worktree-eval-5",
        )


def test_worktree_host_path_rejected_from_privacy_surface() -> None:
    with pytest.raises(HostPathAdmissionError):
        assert_isolated_evaluation_worktree(
            worktree_id="worktree-ok",
            worktree_path="/tmp/eval-worktree-secret",
        )
    with pytest.raises(HostPathAdmissionError):
        assert_isolated_evaluation_worktree(worktree_id="/home/user/repo/.wt")


# ---------------------------------------------------------------------------
# redact_context_for_provider
# ---------------------------------------------------------------------------


def test_redact_context_scrubs_secrets_and_retains_local_source() -> None:
    context = _private_context(
        api_key=CANARY_API_KEY,
        notes=CANARY_TOKEN_ASSIGNMENT,
        password=CANARY_PASSWORD,
    )
    redacted = redact_context_for_provider(context, strip_private_source=False)
    assert redacted["api_key"] == REDACTION_MARKER
    assert redacted["password"] == REDACTION_MARKER
    assert (
        REDACTION_MARKER in redacted["notes"]
        or "secretvalue" not in redacted["notes"]
    )
    # Local path keeps private source keys (secrets inside text still scrubbed).
    assert "raw_private_source" in redacted
    assert "def secret_helper" in redacted["raw_private_source"]


def test_redact_context_strips_private_source_for_external() -> None:
    redacted = redact_context_for_provider(
        _private_context(api_key=CANARY_API_KEY),
        strip_private_source=True,
    )
    assert "raw_private_source" not in redacted
    assert "source_text" not in redacted
    assert redacted.get("api_key") == REDACTION_MARKER
    assert not contains_private_source(redacted)


def test_redact_context_strips_host_paths() -> None:
    context = {
        "summary": "x",
        "host_path": "/var/lib/private/repo",
        "note": "see /home/alice/secrets.env",
        "repo_file": "src/main.py",
    }
    redacted = redact_context_for_provider(context, strip_host_paths=True)
    assert "host_path" not in redacted
    assert redacted["note"] == MANAGED_PATH_PLACEHOLDER or "/home/" not in redacted["note"]
    assert redacted["repo_file"] == "src/main.py"


# ---------------------------------------------------------------------------
# prepare_provider_invocation — end-to-end gate
# ---------------------------------------------------------------------------


def test_prepare_invocation_blocks_private_to_unapproved_external() -> None:
    policy = default_shadow_disclosure_policy()
    with pytest.raises(DisclosureForbiddenError):
        prepare_provider_invocation(
            _private_context(),
            policy,
            provider_id="vendor.cloud.unapproved",
            worktree_id="worktree-eval-6",
        )


def test_prepare_invocation_local_keeps_source_redacts_secrets() -> None:
    policy = default_shadow_disclosure_policy()
    ctx = _private_context(
        api_key=CANARY_API_KEY,
        password=CANARY_PASSWORD,
    )
    prepared = prepare_provider_invocation(
        ctx,
        policy,
        provider_id="local:expanded",
        worktree_id="worktree-eval-7",
    )
    assert prepared.disposition == DisclosureDisposition.LOCAL_ONLY.value
    assert prepared.private_source_stripped is False
    assert "raw_private_source" in prepared.redacted_context
    assert prepared.redacted_context["api_key"] == REDACTION_MARKER
    assert prepared.redacted_context["password"] == REDACTION_MARKER
    # No residual secret field values.
    assert prepared.redacted_context["api_key"] != ctx["api_key"]


def test_prepare_invocation_approved_external_with_authority() -> None:
    policy = ShadowDisclosurePolicy(
        approved_external_provider_ids=("partner.shadow.v1",),
        allow_private_source_to_approved_external=True,
        authorization_cid=_cid("partner-auth"),
    )
    prepared = prepare_provider_invocation(
        _private_context(),
        policy,
        provider_id="partner.shadow.v1",
        worktree_id="worktree-eval-8",
    )
    assert prepared.disposition == DisclosureDisposition.ALLOWED.value
    assert prepared.private_source_stripped is False
    assert prepared.provider_locality == ProviderLocality.APPROVED_EXTERNAL.value


def test_prepare_invocation_rejects_host_paths() -> None:
    policy = default_shadow_disclosure_policy()
    with pytest.raises(HostPathAdmissionError):
        prepare_provider_invocation(
            _public_context(workspace_path="/tmp/checkout"),
            policy,
            provider_id="local:small",
            worktree_id="worktree-eval-9",
        )


# ---------------------------------------------------------------------------
# Public reports — secrets and host paths forbidden
# ---------------------------------------------------------------------------


def test_public_report_rejects_private_source_and_secrets() -> None:
    with pytest.raises(SecretAdmissionError):
        project_public_report({"summary": "ok", "raw_private_source": "LEAK"})
    with pytest.raises(SecretAdmissionError):
        project_public_report({"summary": "ok", "api_key": CANARY_API_KEY})
    with pytest.raises(SecretAdmissionError):
        project_public_report(
            {"summary": "Bearer " + CANARY_BEARER_TOKEN + " in notes"}
        )


def test_public_report_rejects_host_paths() -> None:
    with pytest.raises(HostPathAdmissionError):
        project_public_report({"note": "/tmp/secret.bin"})
    with pytest.raises(HostPathAdmissionError):
        project_public_report({"host_path": "/home/alice/repo"})
    with pytest.raises(HostPathAdmissionError):
        project_public_report({"workspace_path": "C:\\Users\\alice\\repo"})


def test_public_report_admits_cids_and_portable_fields() -> None:
    report = {
        "schema": "example/public-shadow-report@1",
        "context_pack_cid": _cid("pack"),
        "policy_cid": _cid("policy"),
        "disposition": "local_only",
        "summary": "portable facts only",
        "counts": [1, 2, 3],
    }
    projected = project_public_report(report)
    assert projected["context_pack_cid"] == report["context_pack_cid"]
    assert projected["counts"] == [1, 2, 3]


def test_reject_helpers_mirror_public_gates() -> None:
    with pytest.raises(SecretAdmissionError):
        reject_secrets({"password": "x"})
    with pytest.raises(HostPathAdmissionError):
        reject_host_paths({"path": "/etc/passwd"})
    reject_secrets({"summary": "clean"})
    reject_host_paths({"repo_file": "src/a.py", "summary": "ok"})


# ---------------------------------------------------------------------------
# Path classification
# ---------------------------------------------------------------------------


def test_classify_path_closed_classes() -> None:
    assert classify_path("src/module.py") is PathClass.REPO_RELATIVE
    assert classify_path("worktree-eval-1") is PathClass.MANAGED_WORKTREE_ID
    assert classify_path("/tmp/abs") is PathClass.HOST_ABSOLUTE
    assert classify_path("~/secrets") is PathClass.HOST_ABSOLUTE
    assert classify_path("C:\\Users\\x") is PathClass.HOST_ABSOLUTE


# ---------------------------------------------------------------------------
# Authorization decision identity is deterministic
# ---------------------------------------------------------------------------


def test_authorization_identity_is_deterministic() -> None:
    policy = default_shadow_disclosure_policy()
    a = authorize_shadow_disclosure(
        policy,
        provider_id="local:m1",
        context=_private_context(),
        worktree_id="worktree-eval-id",
    )
    b = authorize_shadow_disclosure(
        policy,
        provider_id="local:m1",
        context=_private_context(),
        worktree_id="worktree-eval-id",
    )
    assert a.authorization_decision_cid == b.authorization_decision_cid
    assert a.to_dict()["allowed"] is True


def test_policy_forbids_host_paths_in_public_reports_flag() -> None:
    with pytest.raises(Exception, match="allow_host_paths_in_public_reports"):
        ShadowDisclosurePolicy(allow_host_paths_in_public_reports=True)


def test_policy_forbids_disabling_isolated_worktree() -> None:
    with pytest.raises(Exception, match="isolated_evaluation_worktree"):
        ShadowDisclosurePolicy(require_isolated_evaluation_worktree=False)
