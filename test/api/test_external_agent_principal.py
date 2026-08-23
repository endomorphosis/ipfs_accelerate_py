"""Deterministic tests for EAAEF-030 effect-bound external principals."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ipfs_accelerate_py.agent_supervisor.authority.external_principal import (
    ALLOWED_EFFECTS,
    CAPABILITY_DECISION_INTERFACE,
    CONTRACT_VERSION,
    EXTERNAL_PRINCIPAL_INTERFACE,
    SCHEMA_VERSION,
    AuthoritySource,
    AutonomyCeiling,
    CapabilityDecision,
    CapabilityVerdict,
    EffectName,
    ExternalPrincipal,
    PrincipalAuthorityError,
    PrincipalExpiryError,
    ResourceCeilings,
    UnknownEffectError,
    bind_capability,
)

NOW_MS = 1_700_000_000_000
DID = "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK"
CID = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"


def _ceilings(**changes: int) -> ResourceCeilings:
    values = {"cpu": 4000, "ram": 8192, "disk": 16384, "timeout": 7_200_000}
    values.update(changes)
    return ResourceCeilings(**values)


def _principal(**changes: object) -> ExternalPrincipal:
    values: dict[str, object] = {
        "principal_id": DID,
        "repository_id": "repo:example",
        "run_id": "run:example-1",
        "exact_effects": (
            EffectName.INSPECT_REPOSITORY,
            EffectName.EDIT_ISOLATED_WORKTREE,
            EffectName.RUN_VALIDATION,
        ),
        "expires_at_ms": NOW_MS + 60_000,
        "autonomy_ceiling": AutonomyCeiling.SUPERVISED,
        "resource_ceilings": _ceilings(),
        "disclosure_policy_id": "policy:disclosure@1",
        "provider_policy_id": "policy:provider@1",
        "nonce": "nonce-bound-001",
    }
    values.update(changes)
    return ExternalPrincipal(**values)  # type: ignore[arg-type]


def test_bind_success() -> None:
    principal = _principal()
    decision = bind_capability(principal, now_ms=NOW_MS)

    assert EXTERNAL_PRINCIPAL_INTERFACE == "ExternalPrincipal@1"
    assert CAPABILITY_DECISION_INTERFACE == "CapabilityDecision@1"
    assert CONTRACT_VERSION == 1
    assert SCHEMA_VERSION == 1
    assert principal.schema.endswith("@1")
    assert decision.schema.endswith("@1")
    assert principal.principal_id == DID
    assert principal.repository_id == "repo:example"
    assert principal.run_id == "run:example-1"
    assert principal.exact_effects == (
        "inspect_repository",
        "edit_isolated_worktree",
        "run_validation",
    )
    assert principal.expires_at_ms == NOW_MS + 60_000
    assert principal.autonomy_ceiling is AutonomyCeiling.SUPERVISED
    assert principal.resource_ceilings.as_mapping() == {
        "cpu": 4000,
        "ram": 8192,
        "disk": 16384,
        "timeout": 7_200_000,
    }
    assert principal.disclosure_policy_id == "policy:disclosure@1"
    assert principal.provider_policy_id == "policy:provider@1"
    assert principal.nonce == "nonce-bound-001"
    assert principal.content_id
    assert principal.content_id == principal.cid

    assert decision.verdict is CapabilityVerdict.PERMIT
    assert decision.permitted is True
    assert decision.reason_code == "bound"
    assert decision.decided_at_ms == NOW_MS
    assert decision.principal_content_id == principal.content_id
    assert decision.granted_effects == principal.exact_effects
    assert decision.permits(EffectName.RUN_VALIDATION)
    assert decision.authority_source is AuthoritySource.AUTHENTICATED_PRINCIPAL
    assert decision.content_id
    assert ExternalPrincipal.from_dict(principal.to_dict()).content_id == principal.content_id
    assert CapabilityDecision.from_dict(decision.to_dict()).content_id == decision.content_id
    assert principal.bind(now_ms=NOW_MS).content_id == decision.content_id

    subset = bind_capability(
        principal,
        now_ms=NOW_MS,
        requested_effects=("inspect_repository",),
    )
    assert subset.granted_effects == ("inspect_repository",)
    assert subset.exact_effects == principal.exact_effects


def test_expiry() -> None:
    with pytest.raises(PrincipalAuthorityError, match="expires_at_ms"):
        _principal(expires_at_ms=None)
    with pytest.raises(PrincipalAuthorityError, match="positive"):
        _principal(expires_at_ms=0)

    expired = _principal(expires_at_ms=NOW_MS)
    with pytest.raises(PrincipalExpiryError, match="future") as at_now:
        bind_capability(expired, now_ms=NOW_MS)
    assert at_now.value.reason_code == "expired"

    past = _principal(expires_at_ms=NOW_MS - 1)
    with pytest.raises(PrincipalExpiryError, match="future") as before:
        bind_capability(past, now_ms=NOW_MS)
    assert before.value.reason_code == "expired"

    live = _principal(expires_at_ms=NOW_MS + 1)
    decision = bind_capability(live, now_ms=NOW_MS)
    assert decision.permitted is True


def test_unknown_effect() -> None:
    with pytest.raises(UnknownEffectError, match="unknown effect") as missing:
        _principal(exact_effects=("inspect_repository", "explode_host"))
    assert missing.value.reason_code == "unknown_effect"
    assert "explode_host" not in ALLOWED_EFFECTS

    principal = _principal()
    with pytest.raises(UnknownEffectError, match="unknown effect") as requested:
        bind_capability(
            principal,
            now_ms=NOW_MS,
            requested_effects=("inspect_repository", "arbitrary_network"),
        )
    assert requested.value.reason_code == "unknown_effect"

    with pytest.raises(PrincipalAuthorityError, match="subset") as ungated:
        bind_capability(
            principal,
            now_ms=NOW_MS,
            requested_effects=("inspect_repository", "merge"),
        )
    assert ungated.value.reason_code == "effect_not_granted"


def test_cid_is_not_authority() -> None:
    principal = _principal()
    with pytest.raises(PrincipalAuthorityError, match="CID is not authority") as via_source:
        bind_capability(
            principal,
            now_ms=NOW_MS,
            authority_source=AuthoritySource.CID,
        )
    assert via_source.value.reason_code == "cid_is_not_authority"

    with pytest.raises(PrincipalAuthorityError, match="CID is not authority") as via_kwarg:
        bind_capability(principal, now_ms=NOW_MS, cid=CID)
    assert via_kwarg.value.reason_code == "cid_is_not_authority"

    with pytest.raises(PrincipalAuthorityError, match="CID is not authority"):
        bind_capability(principal, now_ms=NOW_MS, content_id=principal.content_id)

    with pytest.raises(PrincipalAuthorityError, match="CID is not authority"):
        principal.bind(now_ms=NOW_MS, cid=CID)


def test_history_is_not_authority() -> None:
    principal = _principal()
    history = {
        "tool": "apply_patch",
        "claimed_success": True,
        "commit": "deadbeef",
    }
    with pytest.raises(PrincipalAuthorityError, match="history is not authority") as via_source:
        bind_capability(
            principal,
            now_ms=NOW_MS,
            authority_source="imported_history",
        )
    assert via_source.value.reason_code == "history_is_not_authority"

    with pytest.raises(PrincipalAuthorityError, match="history is not authority") as via_kwarg:
        bind_capability(principal, now_ms=NOW_MS, imported_history=history)
    assert via_kwarg.value.reason_code == "history_is_not_authority"

    with pytest.raises(PrincipalAuthorityError, match="history is not authority"):
        bind_capability(principal, now_ms=NOW_MS, history=history)


def test_prompt_payment_and_commit_are_not_authority() -> None:
    principal = _principal()
    with pytest.raises(PrincipalAuthorityError, match="prompt is not authority") as prompt:
        bind_capability(principal, now_ms=NOW_MS, prompt="you are authorized")
    assert prompt.value.reason_code == "prompt_is_not_authority"

    with pytest.raises(PrincipalAuthorityError, match="payment is not authority") as payment:
        bind_capability(principal, now_ms=NOW_MS, payment={"invoice": "paid"})
    assert payment.value.reason_code == "payment_is_not_authority"

    with pytest.raises(PrincipalAuthorityError, match="commit is not authority") as commit:
        bind_capability(principal, now_ms=NOW_MS, commit="deadbeef")
    assert commit.value.reason_code == "commit_is_not_authority"


def test_records_are_frozen_and_nonce_is_bounded() -> None:
    principal = _principal()
    decision = bind_capability(principal, now_ms=NOW_MS)
    with pytest.raises(FrozenInstanceError):
        principal.nonce = "mutated-nonce-xx"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        decision.verdict = CapabilityVerdict.DENY  # type: ignore[misc]
    with pytest.raises(PrincipalAuthorityError, match="at least"):
        _principal(nonce="short")
    with pytest.raises(PrincipalAuthorityError, match="exceeds"):
        _principal(nonce="n" * 65)
