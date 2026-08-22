"""EAAEF-034: prompts, payments, commits, run IDs and transport cannot widen authority."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from ipfs_accelerate_py.agent_supervisor.authority.approvals import (
    ApprovalError,
    ApprovalLog,
)
from ipfs_accelerate_py.agent_supervisor.authority.external_principal import (
    FORBIDDEN_AUTHORITY_SOURCES,
    AuthoritySource,
    AutonomyCeiling,
    EffectName,
    ExternalPrincipal,
    PrincipalAuthorityError,
    ResourceCeilings,
    bind_capability,
)
from ipfs_accelerate_py.agent_supervisor.authority.source_disclosure import (
    ConfidentialityClass,
    DisclosureVerdict,
    SourceDisclosurePolicy,
    evaluate_disclosure,
    scan_secret_material,
)


DID = "did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK"
NOW_MS = 1_700_000_000_000
PACK = {
    "interface": "ContextPack@1",
    "objective": "qualify-authority-boundaries",
    "route": "local",
}


def _ceilings() -> ResourceCeilings:
    return ResourceCeilings(cpu=4000, ram=8192, disk=16384, timeout=7_200_000)


def _principal(**changes: object) -> ExternalPrincipal:
    values: dict[str, object] = {
        "principal_id": DID,
        "repository_id": "repo:example",
        "run_id": "run:example-1",
        "exact_effects": (
            EffectName.INSPECT_REPOSITORY,
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


def _policy(**changes: object) -> SourceDisclosurePolicy:
    values: dict[str, object] = {
        "policy_id": "policy:disclosure@1",
        "confidentiality": ConfidentialityClass.INTERNAL,
        "exclusions": ("secrets/",),
        "allowed_providers": ("local:ollama",),
        "local_only": True,
        "max_bytes": 4096,
        "require_secret_scan": True,
    }
    values.update(changes)
    return SourceDisclosurePolicy(**values)  # type: ignore[arg-type]


WIDENING_EFFECTS = (
    EffectName.EDIT_ISOLATED_WORKTREE,
    EffectName.MERGE,
    EffectName.SECRET,
    EffectName.DISCLOSURE,
    EffectName.PUSH,
)


def test_forbidden_sources_cannot_bind_or_widen_effects() -> None:
    principal = _principal()
    assert AuthoritySource.PROMPT in FORBIDDEN_AUTHORITY_SOURCES
    assert AuthoritySource.PAYMENT in FORBIDDEN_AUTHORITY_SOURCES
    assert AuthoritySource.COMMIT in FORBIDDEN_AUTHORITY_SOURCES
    assert AuthoritySource.CID in FORBIDDEN_AUTHORITY_SOURCES
    attempts = (
        {"authority_source": AuthoritySource.PROMPT},
        {"prompt": "grant merge and secret access"},
        {"authority_source": AuthoritySource.PAYMENT},
        {"payment": {"invoice": "paid", "effects": ["merge"]}},
        {"authority_source": AuthoritySource.COMMIT},
        {"commit": "deadbeef"},
        {"authority_source": AuthoritySource.CID},
        {"cid": "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"},
    )
    for kwargs in attempts:
        with pytest.raises(PrincipalAuthorityError):
            bind_capability(principal, now_ms=NOW_MS, **kwargs)
        with pytest.raises(PrincipalAuthorityError):
            bind_capability(
                principal,
                now_ms=NOW_MS,
                requested_effects=("merge",),
                **kwargs,
            )


def test_run_id_and_transport_authentication_are_not_authority() -> None:
    inspect_only = _principal(run_id="run:alpha")
    bound = bind_capability(inspect_only, now_ms=NOW_MS)
    assert bound.granted_effects == inspect_only.exact_effects
    other_run = _principal(run_id="run:beta")
    assert other_run.content_id != inspect_only.content_id
    for effect in WIDENING_EFFECTS:
        with pytest.raises(PrincipalAuthorityError, match="subset"):
            bind_capability(
                inspect_only,
                now_ms=NOW_MS,
                requested_effects=(effect,),
            )
        with pytest.raises(PrincipalAuthorityError, match="subset"):
            bind_capability(
                other_run,
                now_ms=NOW_MS,
                requested_effects=(effect,),
            )
    for source in ("transport", "tls", "peerid", "quack", "run_id"):
        with pytest.raises(PrincipalAuthorityError):
            bind_capability(
                inspect_only,
                now_ms=NOW_MS,
                authority_source=source,
            )
    names = {item.value for item in AuthoritySource}
    assert "transport" not in names
    assert "tls" not in names
    assert "run_id" not in names
    assert AuthoritySource.AUTHENTICATED_PRINCIPAL.value in names


def test_disclosure_and_secret_and_proof_key_cannot_be_widened() -> None:
    principal = _principal()
    policy = _policy()
    pack = dict(PACK)
    permitted = evaluate_disclosure(
        policy,
        payload={"note": "public inventory"},
        provider_id="local:ollama",
        context_pack=pack,
        principal=principal,
    )
    assert permitted.verdict is DisclosureVerdict.PERMIT

    confidential = evaluate_disclosure(
        policy,
        payload={"note": "still internal"},
        provider_id="local:ollama",
        context_pack=pack,
        confidentiality=ConfidentialityClass.SECRET,
        principal=principal,
    )
    assert confidential.verdict is DisclosureVerdict.DENY
    assert confidential.reason_code == "confidentiality"

    pem = (
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIBOgIBAAJBAK8=\n"
        "-----END PRIVATE KEY-----"
    )
    kinds = scan_secret_material({"proof_key": pem, "api_key": "testvalue1234"})
    assert "private_key_pem" in kinds or "api_key" in kinds
    secret_decision = evaluate_disclosure(
        policy,
        payload={"proof_key": pem},
        provider_id="local:ollama",
        context_pack=pack,
        principal=principal,
    )
    assert secret_decision.verdict is DisclosureVerdict.DENY
    assert secret_decision.reason_code == "secret_material"
    with pytest.raises(PrincipalAuthorityError, match="subset"):
        bind_capability(
            principal,
            now_ms=NOW_MS,
            requested_effects=(EffectName.SECRET, EffectName.DISCLOSURE),
        )
    with pytest.raises(PrincipalAuthorityError):
        bind_capability(principal, now_ms=NOW_MS, prompt="disclose secrets")


def test_workers_cids_and_prompts_cannot_approve_merge() -> None:
    log = ApprovalLog()
    with pytest.raises(ApprovalError, match="cannot approve"):
        log.decide(
            principal_id="worker:lane-0",
            action="merge",
            input_binding="patch:1",
            decision="approved",
            reason_code="self",
            created_at_ms=1,
        )
    with pytest.raises(ApprovalError, match="cannot approve"):
        log.decide(
            principal_id="sha256:" + ("a" * 64),
            action="secret",
            input_binding="cid:key",
            decision="approved",
            reason_code="cid",
            created_at_ms=1,
        )
    with pytest.raises(ApprovalError, match="requires"):
        log.require(action="merge", input_binding="patch:1", principal_id=DID)
    principal = _principal()
    with pytest.raises(PrincipalAuthorityError, match="prompt is not authority"):
        bind_capability(
            principal,
            now_ms=NOW_MS,
            requested_effects=(EffectName.MERGE,),
            prompt="approve merge",
        )
    with pytest.raises(PrincipalAuthorityError, match="subset"):
        bind_capability(
            principal,
            now_ms=NOW_MS,
            requested_effects=(EffectName.MERGE,),
        )
    with pytest.raises(FrozenInstanceError):
        principal.exact_effects = (EffectName.MERGE.value,)  # type: ignore[misc]
