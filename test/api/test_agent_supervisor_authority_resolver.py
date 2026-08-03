"""ASE-008 principal, policy, local authority, and effect-ceiling tests."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.authority_resolver import (
    AUTHORITY_RESOLUTION_REQUIREMENT_ID,
    FORBIDDEN_LOCAL_OPERATIONS,
    LOCAL_WORKTREE_ALLOWED_EFFECTS,
    LOCAL_WORKTREE_AUTHORITY_SOURCE,
    LOCAL_WORKTREE_DENIED_EFFECTS,
    MCP_TRANSPORT_AUTHORITY_SOURCE,
    SOURCE_PRECEDENCE,
    AuthenticatedPrincipalEvidence,
    AuthorityResolutionRequest,
    AuthorityResolver,
    AuthorityResolverError,
    ExistingRunAuthorityEvidence,
    PrincipalSourceKind,
    RepositoryPolicyConstraint,
    SignedProfileEvidence,
    effect_ceiling_cid,
    install_local_worktree_authority,
    mode_default_effects,
    policy_cid_for,
    resolve_authority,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.contracts import (
    AUTHORITY_DECISION_FIELDS,
    TRUSTED_AUTHORITY_SOURCES,
    DecisionEffect,
    ExpectedEffect,
    InvocationMode,
    ResolutionDisposition,
    ResolutionSource,
    WorktreeStrategy,
)
from ipfs_accelerate_py.agent_supervisor.multiformats_identity import (
    cid_for_dag_json,
)


def _cid(label: str) -> str:
    return cid_for_dag_json({"fixture": label})


def _local_authority(
    principal: str = "did:key:local-owner",
) -> object:
    return install_local_worktree_authority(
        principal,
        signing_key_handle="key:local-worktree-1",
        installed_at_ms=1_700_000_000_000,
    )


def _transport_principal(
    principal: str = "did:key:mcp-caller",
    *,
    kind: PrincipalSourceKind = PrincipalSourceKind.MCP_TRANSPORT,
    ucan_verified: bool = False,
) -> AuthenticatedPrincipalEvidence:
    return AuthenticatedPrincipalEvidence(
        principal_ref=principal,
        source=ResolutionSource.AUTHENTICATED_TRANSPORT,
        evidence_cid=_cid(f"transport-{principal}"),
        kind=kind,
        transport="mcp",
        signature_verified=True,
        ucan_verified=ucan_verified,
    )


def _signed_profile(
    principal: str = "did:key:local-owner",
    *,
    effects: tuple[ExpectedEffect, ...] | None = None,
) -> SignedProfileEvidence:
    allowed = effects or LOCAL_WORKTREE_ALLOWED_EFFECTS
    return SignedProfileEvidence(
        profile_name="local-worktree",
        profile_cid=_cid("signed-profile"),
        policy_cid=policy_cid_for("policy:signed-local-worktree@1"),
        principal_ref=principal,
        authority_source_ref="authority:signed-profile",
        allowed_effects=allowed,
        evidence_cid=_cid("signed-profile-evidence"),
        signature_verified=True,
        worktree_strategy=WorktreeStrategy.ISOLATED,
    )


def test_requirement_id_is_stable() -> None:
    assert AUTHORITY_RESOLUTION_REQUIREMENT_ID.startswith("requirement:")
    assert "authority-resolution" in AUTHORITY_RESOLUTION_REQUIREMENT_ID


def test_source_precedence_matches_plan_ladder() -> None:
    assert (
        SOURCE_PRECEDENCE[ResolutionSource.EXISTING_RUN]
        > SOURCE_PRECEDENCE[ResolutionSource.AUTHENTICATED_TRANSPORT]
        > SOURCE_PRECEDENCE[ResolutionSource.SIGNED_PROFILE]
        > SOURCE_PRECEDENCE[ResolutionSource.REPOSITORY_HINT]
        > SOURCE_PRECEDENCE[ResolutionSource.DISCOVERY]
        > SOURCE_PRECEDENCE[ResolutionSource.BUILTIN_DEFAULT]
    )


def test_local_principal_binding_after_explicit_install() -> None:
    authority = _local_authority()
    result = resolve_authority(
        AuthorityResolutionRequest(
            mode=InvocationMode.WORKTREE,
            local_worktree_authority=authority,
        )
    )
    assert result.authorized
    assert result.requirement_id == AUTHORITY_RESOLUTION_REQUIREMENT_ID
    assert result.principal.bound
    assert result.principal.principal_ref == "did:key:local-owner"
    assert result.principal.source is ResolutionSource.SIGNED_PROFILE
    assert result.principal.kind is PrincipalSourceKind.LOCAL_WORKTREE_INSTALL
    assert result.authority_source_ref == LOCAL_WORKTREE_AUTHORITY_SOURCE
    assert result.local_worktree is not None
    assert (
        result.local_worktree.installation_receipt_cid
        == authority.installation_receipt_cid
    )


def test_mcp_principal_binding_from_authenticated_transport() -> None:
    principal = _transport_principal()
    profile = _signed_profile(principal="did:key:mcp-caller")
    result = resolve_authority(
        AuthorityResolutionRequest(
            mode=InvocationMode.WORKTREE,
            authenticated_principal=principal,
            signed_profile=profile,
        )
    )
    assert result.authorized
    assert result.principal.principal_ref == "did:key:mcp-caller"
    assert result.principal.source is ResolutionSource.AUTHENTICATED_TRANSPORT
    assert result.principal.kind is PrincipalSourceKind.MCP_TRANSPORT
    assert result.principal.transport == "mcp"


def test_mcp_plus_requires_verified_ucan() -> None:
    with pytest.raises(AuthorityResolverError, match="UCAN"):
        resolve_authority(
            AuthorityResolutionRequest(
                authenticated_principal=AuthenticatedPrincipalEvidence(
                    principal_ref="did:key:mcp-plus",
                    source=ResolutionSource.AUTHENTICATED_TRANSPORT,
                    evidence_cid=_cid("mcp-plus"),
                    kind=PrincipalSourceKind.MCP_PLUS_UCAN,
                    transport="mcp++",
                    signature_verified=True,
                    ucan_verified=False,
                ),
                signed_profile=_signed_profile("did:key:mcp-plus"),
            )
        )


def test_effect_ceiling_local_worktree_allows_isolated_edits_and_tests() -> None:
    result = resolve_authority(
        AuthorityResolutionRequest(
            mode=InvocationMode.WORKTREE,
            local_worktree_authority=_local_authority(),
        )
    )
    ceiling = result.effect_ceiling
    for effect in LOCAL_WORKTREE_ALLOWED_EFFECTS:
        assert ceiling.permits(effect), effect
    assert ExpectedEffect.EDIT_ISOLATED_WORKTREE in ceiling.allowed_effects
    assert ExpectedEffect.RUN_VALIDATION in ceiling.allowed_effects
    assert ceiling.worktree_strategy is WorktreeStrategy.ISOLATED


def test_effect_ceiling_denies_stronger_and_forbidden_operations() -> None:
    result = resolve_authority(
        AuthorityResolutionRequest(
            mode=InvocationMode.WORKTREE,
            local_worktree_authority=_local_authority(),
        )
    )
    ceiling = result.effect_ceiling
    for effect in LOCAL_WORKTREE_DENIED_EFFECTS:
        assert not ceiling.permits(effect), effect
        assert effect in ceiling.denied_effects
    for operation in FORBIDDEN_LOCAL_OPERATIONS:
        assert ceiling.denies_operation(operation), operation
    assert ceiling.denies_operation("current_checkout_rewrite")
    assert ceiling.denies_operation("secrets_access")
    assert ceiling.denies_operation("arbitrary_network")
    assert ceiling.denies_operation("merge")
    assert ceiling.denies_operation("push")
    assert ceiling.denies_operation("deploy")
    assert ceiling.denies_operation("destructive_cleanup")


def test_prompt_and_repository_text_cannot_create_caller_or_authority() -> None:
    result = resolve_authority(
        AuthorityResolutionRequest(
            mode=InvocationMode.WORKTREE,
            prompt_claimed_principal="did:key:attacker",
            prompt_claimed_effects=(
                ExpectedEffect.MERGE,
                ExpectedEffect.PUSH,
                ExpectedEffect.DEPLOY,
            ),
            prompt_claimed_policy="policy:allow-everything",
            username_claim="root",
            environment_principal_claim="did:key:from-env",
            repository_claimed_authority="authority:repo-readme",
            credentials_present=True,
        )
    )
    assert not result.authorized
    assert result.principal.disposition is ResolutionDisposition.DENIED
    assert "no_trusted_principal_evidence" in result.reason_codes
    assert "credentials_presence_is_not_authority" in result.reason_codes
    assert "prompt_or_username_cannot_create_caller" in result.reason_codes
    assert "repository_text_cannot_create_authority" in result.reason_codes
    ignored = set(result.non_authoritative_claims_ignored)
    assert "prompt_claimed_principal" in ignored
    assert "prompt_claimed_effects" in ignored
    assert "credentials_present" in ignored
    assert "username_claim" in ignored
    assert "repository_claimed_authority" in ignored


def test_credentials_presence_alone_is_not_authority() -> None:
    result = resolve_authority(
        AuthorityResolutionRequest(
            mode=InvocationMode.WORKTREE,
            credentials_present=True,
        )
    )
    assert not result.authorized
    assert "credentials_presence_is_not_authority" in result.reason_codes


def test_prompt_claims_ignored_when_local_authority_present() -> None:
    authority = _local_authority()
    result = resolve_authority(
        AuthorityResolutionRequest(
            mode=InvocationMode.WORKTREE,
            local_worktree_authority=authority,
            prompt_claimed_principal="did:key:attacker",
            prompt_claimed_effects=(ExpectedEffect.MERGE, ExpectedEffect.PUSH),
            username_claim="root",
            credentials_present=True,
            repository_claimed_authority="authority:repo-policy",
        )
    )
    assert result.authorized
    assert result.principal.principal_ref == "did:key:local-owner"
    assert not result.effect_ceiling.permits(ExpectedEffect.MERGE)
    assert not result.effect_ceiling.permits(ExpectedEffect.PUSH)
    principal_decision = result.decision_for("principal")
    rejected = [
        candidate
        for candidate in principal_decision.candidates
        if candidate.rejection_reason
    ]
    assert rejected
    assert any(
        "prompt" in candidate.rejection_reason
        or "username" in candidate.rejection_reason
        for candidate in rejected
    )


def test_repository_policy_only_narrows_never_creates_authority() -> None:
    # Repository policy alone cannot authorize.
    alone = resolve_authority(
        AuthorityResolutionRequest(
            mode=InvocationMode.WORKTREE,
            repository_policy_constraint=RepositoryPolicyConstraint(
                policy_cid=_cid("repo-policy"),
                evidence_cid=_cid("repo-policy-evidence"),
                allowed_effects=(
                    ExpectedEffect.INSPECT_REPOSITORY,
                    ExpectedEffect.MERGE,
                ),
            ),
        )
    )
    assert not alone.authorized

    # With local authority, repository policy intersects (narrows) only.
    authority = _local_authority()
    narrowed = resolve_authority(
        AuthorityResolutionRequest(
            mode=InvocationMode.WORKTREE,
            local_worktree_authority=authority,
            repository_policy_constraint=RepositoryPolicyConstraint(
                policy_cid=_cid("repo-policy"),
                evidence_cid=_cid("repo-policy-evidence"),
                allowed_effects=(
                    ExpectedEffect.INSPECT_REPOSITORY,
                    ExpectedEffect.RUN_VALIDATION,
                    ExpectedEffect.MERGE,
                ),
                denied_effects=(ExpectedEffect.LAUNCH_LOCAL_PROCESS,),
            ),
        )
    )
    assert narrowed.authorized
    assert ExpectedEffect.INSPECT_REPOSITORY in narrowed.effect_ceiling.allowed_effects
    assert ExpectedEffect.RUN_VALIDATION in narrowed.effect_ceiling.allowed_effects
    assert ExpectedEffect.MERGE not in narrowed.effect_ceiling.allowed_effects
    assert (
        ExpectedEffect.LAUNCH_LOCAL_PROCESS
        not in narrowed.effect_ceiling.allowed_effects
    )
    assert (
        _cid("repo-policy")
        in narrowed.policy.constraint_policy_cids
    )
    policy_decision = narrowed.decision_for("policy")
    assert any(
        candidate.rejection_reason
        == "repository_policy_is_constraint_not_authority"
        for candidate in policy_decision.candidates
    )


def test_lower_precedence_sources_only_narrow_effects() -> None:
    wide = LOCAL_WORKTREE_ALLOWED_EFFECTS
    profile = _signed_profile(effects=wide)
    # Requested narrowing + repository constraint both subtract.
    result = resolve_authority(
        AuthorityResolutionRequest(
            mode=InvocationMode.WORKTREE,
            signed_profile=profile,
            repository_policy_constraint=RepositoryPolicyConstraint(
                policy_cid=_cid("repo-constraint"),
                evidence_cid=_cid("repo-constraint-evidence"),
                denied_effects=(ExpectedEffect.EDIT_ISOLATED_WORKTREE,),
            ),
            requested_effect_narrowing=(
                ExpectedEffect.INSPECT_REPOSITORY,
                ExpectedEffect.WRITE_SUPERVISOR_STATE,
                ExpectedEffect.EDIT_ISOLATED_WORKTREE,
                ExpectedEffect.RUN_VALIDATION,
            ),
        )
    )
    assert result.authorized
    allowed = set(result.effect_ceiling.allowed_effects)
    assert ExpectedEffect.INSPECT_REPOSITORY in allowed
    assert ExpectedEffect.WRITE_SUPERVISOR_STATE in allowed
    assert ExpectedEffect.RUN_VALIDATION in allowed
    # Repository denial removes edit even though request asked for it.
    assert ExpectedEffect.EDIT_ISOLATED_WORKTREE not in allowed
    # Profile had create/launch, but requested narrowing removed them.
    assert ExpectedEffect.CREATE_ISOLATED_WORKTREE not in allowed
    assert ExpectedEffect.LAUNCH_LOCAL_PROCESS not in allowed
    # Stronger effects remain impossible.
    assert ExpectedEffect.PUSH not in allowed


def test_existing_run_outpaces_signed_profile_for_principal() -> None:
    run = ExistingRunAuthorityEvidence(
        run_id="run:fixture-1",
        principal_ref="did:key:run-owner",
        policy_cid=policy_cid_for("policy:run@1"),
        authority_source_ref="authority:existing-run",
        allowed_effects=(
            ExpectedEffect.INSPECT_REPOSITORY,
            ExpectedEffect.WRITE_SUPERVISOR_STATE,
            ExpectedEffect.RUN_VALIDATION,
        ),
        evidence_cid=_cid("existing-run"),
        worktree_strategy=WorktreeStrategy.ISOLATED,
    )
    profile = _signed_profile(principal="did:key:run-owner")
    result = resolve_authority(
        AuthorityResolutionRequest(
            mode=InvocationMode.WORKTREE,
            existing_run=run,
            signed_profile=profile,
        )
    )
    assert result.authorized
    assert result.principal.source is ResolutionSource.EXISTING_RUN
    assert result.principal.principal_ref == "did:key:run-owner"
    # Intersection with run effects removes worktree edit/create/launch.
    assert ExpectedEffect.EDIT_ISOLATED_WORKTREE not in (
        result.effect_ceiling.allowed_effects
    )


def test_conflicting_trusted_principals_fail_closed() -> None:
    result = resolve_authority(
        AuthorityResolutionRequest(
            mode=InvocationMode.WORKTREE,
            existing_run=ExistingRunAuthorityEvidence(
                run_id="run:a",
                principal_ref="did:key:alice",
                policy_cid=policy_cid_for("policy:a@1"),
                authority_source_ref="authority:existing-run",
                allowed_effects=LOCAL_WORKTREE_ALLOWED_EFFECTS,
                evidence_cid=_cid("run-a"),
            ),
            signed_profile=_signed_profile(principal="did:key:bob"),
        )
    )
    assert not result.authorized
    assert "conflicting_trusted_principals" in result.reason_codes


def test_transport_alone_without_effect_grant_denies_worktree_mode() -> None:
    result = resolve_authority(
        AuthorityResolutionRequest(
            mode=InvocationMode.WORKTREE,
            authenticated_principal=_transport_principal(),
        )
    )
    assert not result.authorized
    assert "no_trusted_policy_evidence" in result.reason_codes or (
        "no_trusted_effect_grant" in result.reason_codes
    )


def test_preview_mode_with_transport_principal_is_inspect_only() -> None:
    result = resolve_authority(
        AuthorityResolutionRequest(
            mode=InvocationMode.PREVIEW,
            authenticated_principal=_transport_principal(),
        )
    )
    assert result.authorized
    assert result.effect_ceiling.allowed_effects == (
        ExpectedEffect.INSPECT_REPOSITORY,
    )
    assert not result.effect_ceiling.permits(ExpectedEffect.EDIT_ISOLATED_WORKTREE)


def test_authority_decisions_are_contract_valid_and_require_authority() -> None:
    result = resolve_authority(
        AuthorityResolutionRequest(
            mode=InvocationMode.WORKTREE,
            local_worktree_authority=_local_authority(),
        )
    )
    fields = {decision.field_name for decision in result.decisions}
    assert fields == set(AUTHORITY_DECISION_FIELDS)
    for decision in result.decisions:
        assert decision.effect is DecisionEffect.REQUIRES_AUTHORITY
        assert decision.selected_source in TRUSTED_AUTHORITY_SOURCES
        assert not decision.override_accepted
        # Round-trip through contract serialization.
        restored = type(decision).from_dict(decision.to_dict())
        assert restored.content_id == decision.content_id


def test_decision_reference_is_content_addressed_and_stable() -> None:
    request = AuthorityResolutionRequest(
        mode=InvocationMode.WORKTREE,
        local_worktree_authority=_local_authority(),
    )
    first = resolve_authority(request)
    second = resolve_authority(request)
    assert first.decision_reference_cid == second.decision_reference_cid
    assert first.effect_ceiling.ceiling_cid == second.effect_ceiling.ceiling_cid
    assert first.decision_reference_cid.startswith("baguqeer")


def test_install_local_worktree_is_receipt_bound_and_deterministic() -> None:
    a = install_local_worktree_authority(
        "did:key:local-owner",
        signing_key_handle="key:local-1",
        installed_at_ms=42,
    )
    b = install_local_worktree_authority(
        "did:key:local-owner",
        signing_key_handle="key:local-1",
        installed_at_ms=42,
    )
    assert a.installation_receipt_cid == b.installation_receipt_cid
    assert a.effect_ceiling_cid == b.effect_ceiling_cid
    assert a.verified
    assert a.worktree_strategy is WorktreeStrategy.ISOLATED
    assert set(a.allowed_effects) == set(LOCAL_WORKTREE_ALLOWED_EFFECTS)
    for effect in LOCAL_WORKTREE_DENIED_EFFECTS:
        assert not a.permits(effect)


def test_install_rejects_stronger_effects() -> None:
    with pytest.raises(AuthorityResolverError, match="stronger effects"):
        install_local_worktree_authority(
            "did:key:local-owner",
            signing_key_handle="key:local-1",
            allowed_effects=(
                ExpectedEffect.INSPECT_REPOSITORY,
                ExpectedEffect.MERGE,
            ),
        )


def test_local_worktree_authority_rejects_current_checkout_strategy() -> None:
    base = _local_authority()
    with pytest.raises(AuthorityResolverError, match="current checkout"):
        type(base)(
            principal_ref=base.principal_ref,
            installation_receipt_cid=base.installation_receipt_cid,
            signing_key_handle=base.signing_key_handle,
            policy_cid=base.policy_cid,
            worktree_strategy=WorktreeStrategy.CURRENT_CHECKOUT,
        )


def test_unsigned_profile_cannot_supply_authority() -> None:
    with pytest.raises(AuthorityResolverError, match="unsigned|unverified"):
        SignedProfileEvidence(
            profile_name="forged",
            profile_cid=_cid("forged-profile"),
            policy_cid=_cid("forged-policy"),
            principal_ref="did:key:forged",
            authority_source_ref="authority:forged",
            allowed_effects=LOCAL_WORKTREE_ALLOWED_EFFECTS,
            evidence_cid=_cid("forged-evidence"),
            signature_verified=False,
        )


def test_unverified_principal_evidence_rejected() -> None:
    with pytest.raises(AuthorityResolverError, match="signature or UCAN"):
        AuthenticatedPrincipalEvidence(
            principal_ref="did:key:x",
            source=ResolutionSource.AUTHENTICATED_TRANSPORT,
            evidence_cid=_cid("x"),
            kind=PrincipalSourceKind.MCP_TRANSPORT,
            signature_verified=False,
            ucan_verified=False,
        )


def test_untrusted_resolution_source_rejected_for_principal_evidence() -> None:
    with pytest.raises(AuthorityResolverError, match="trusted authority source"):
        AuthenticatedPrincipalEvidence(
            principal_ref="did:key:x",
            source=ResolutionSource.REPOSITORY_HINT,
            evidence_cid=_cid("x"),
            kind=PrincipalSourceKind.MCP_TRANSPORT,
            signature_verified=True,
        )


def test_resolver_reuses_installed_local_authority_without_repeated_flags() -> None:
    resolver = AuthorityResolver()
    installed = resolver.install_local_worktree(
        "did:key:local-owner",
        signing_key_handle="key:local-worktree-1",
        installed_at_ms=99,
    )
    # Subsequent prompt-only style resolve uses installed authority only.
    result = resolver.resolve(mode=InvocationMode.WORKTREE)
    assert result.authorized
    assert result.local_worktree is not None
    assert (
        result.local_worktree.installation_receipt_cid
        == installed.installation_receipt_cid
    )
    assert result.principal.principal_ref == "did:key:local-owner"
    # Still denies stronger effects after reuse.
    assert not result.effect_ceiling.permits(ExpectedEffect.DEPLOY)


def test_mode_default_effects_are_conservative() -> None:
    assert mode_default_effects(InvocationMode.PREVIEW) == (
        ExpectedEffect.INSPECT_REPOSITORY,
    )
    assert set(mode_default_effects(InvocationMode.WORKTREE)) == set(
        LOCAL_WORKTREE_ALLOWED_EFFECTS
    )


def test_effect_ceiling_cid_changes_when_effects_change() -> None:
    a = effect_ceiling_cid((ExpectedEffect.INSPECT_REPOSITORY,))
    b = effect_ceiling_cid(
        (
            ExpectedEffect.INSPECT_REPOSITORY,
            ExpectedEffect.RUN_VALIDATION,
        )
    )
    assert a != b
    assert a == effect_ceiling_cid((ExpectedEffect.INSPECT_REPOSITORY,))


def test_requested_narrowing_cannot_widen_beyond_authority() -> None:
    result = resolve_authority(
        AuthorityResolutionRequest(
            mode=InvocationMode.WORKTREE,
            local_worktree_authority=_local_authority(),
            requested_effect_narrowing=(
                ExpectedEffect.INSPECT_REPOSITORY,
                ExpectedEffect.MERGE,
                ExpectedEffect.PUSH,
            ),
        )
    )
    assert result.authorized
    assert result.effect_ceiling.allowed_effects == (
        ExpectedEffect.INSPECT_REPOSITORY,
    )
    assert not result.effect_ceiling.permits(ExpectedEffect.MERGE)


def test_authority_source_decision_records_repository_forgery() -> None:
    result = resolve_authority(
        AuthorityResolutionRequest(
            mode=InvocationMode.WORKTREE,
            local_worktree_authority=_local_authority(),
            repository_claimed_authority="forged-local-root",
        )
    )
    decision = result.decision_for("authority_source")
    assert decision.selected_value == LOCAL_WORKTREE_AUTHORITY_SOURCE
    assert any(
        candidate.rejection_reason
        == "repository_text_cannot_create_authority"
        for candidate in decision.candidates
    )


def test_resolution_to_dict_is_json_friendly() -> None:
    result = resolve_authority(
        AuthorityResolutionRequest(
            mode=InvocationMode.WORKTREE,
            local_worktree_authority=_local_authority(),
        )
    )
    payload = result.to_dict()
    assert payload["authorized"] is True
    assert payload["requirement_id"] == AUTHORITY_RESOLUTION_REQUIREMENT_ID
    assert payload["principal"]["bound"] is True
    assert payload["effect_ceiling"]["ceiling_cid"]
    assert payload["local_worktree"]["profile_name"] == "local-worktree"
    assert MCP_TRANSPORT_AUTHORITY_SOURCE.startswith("authority:")


def test_denied_resolution_marks_all_authority_fields_denied() -> None:
    result = resolve_authority(AuthorityResolutionRequest())
    assert not result.authorized
    for field_name in AUTHORITY_DECISION_FIELDS:
        decision = result.decision_for(field_name)
        assert decision.disposition is ResolutionDisposition.DENIED
        assert decision.selected_value == ""
        assert decision.reason_codes
