"""ASE-010 profile precedence and complete target-resolution composition tests."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.authority_resolver import (
    LOCAL_WORKTREE_ALLOWED_EFFECTS,
    AuthorityResolutionRequest,
    install_local_worktree_authority,
    resolve_authority,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.capability_resolver import (
    FALLBACK_PROVIDER,
    PREFERRED_PROVIDER,
    CapabilityEvidence,
    PreferredProviderCapability,
    ProviderCapabilityEvidence,
    ResourceSampleEvidence,
    TopologyEvidence,
    TopologyMode,
    ValidationPolicyEvidence,
    resolve_capabilities,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.contracts import (
    REQUIRED_TARGET_DECISION_FIELDS,
    DecisionEffect,
    ExpectedEffect,
    InvocationBudget,
    InvocationMode,
    OutputMode,
    ResolutionDisposition,
    ResolutionSource,
    RevalidationRule,
    SupervisorInvocationRequest,
    TargetCandidate,
    TargetInferenceDecision,
    WorktreeStrategy,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.objective_resolver import (
    ObjectiveResolutionEvidence,
    resolve_objectives,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.profile_resolver import (
    BUILTIN_PROFILE_LOCAL_WORKTREE,
    BUILTIN_PROFILE_PREVIEW,
    PROFILE_OWNED_FIELDS,
    RESOLVED_SUPERVISOR_PROFILE_REQUIREMENT_ID,
    SOURCE_PRECEDENCE,
    CanonicalRequestBinding,
    ProfileCompositionRequest,
    ProfileResolverError,
    ProfileSourceKind,
    ProfileSourceLayer,
    SupervisorProfileResolver,
    build_target_resolution_receipt,
    builtin_profile_for_mode,
    resolve_supervisor_profile,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.state_resolver import (
    StateResolutionEvidence,
    resolve_state,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.target_resolver import (
    RepositoryTargetBinding,
    RepositoryTargetResolution,
)
from ipfs_accelerate_py.agent_supervisor.multiformats_identity import (
    cid_for_bytes,
    cid_for_dag_json,
)

REPO_ROOT = "/home/dev/src/project"
PRINCIPAL = "did:key:local-owner"
PROMPT = "Improve validation-cache correctness without leaking this body."


def _cid(label: str) -> str:
    return cid_for_dag_json({"fixture": label})


def _prompt_cid(text: str = PROMPT) -> str:
    return cid_for_bytes(text.encode("utf-8"))


def _invocation(**overrides: object) -> SupervisorInvocationRequest:
    values: dict[str, object] = {
        "prompt_cid": _prompt_cid(),
        "prompt_ref": "prompt-broker:fixture",
        "mode": InvocationMode.WORKTREE,
        "budget": InvocationBudget(
            max_prompt_bytes=16_384,
            max_actions=64,
            max_lanes=4,
            timeout_ms=3_600_000,
            max_result_bytes=1024 * 1024,
        ),
        "profile_hint": "local-worktree",
    }
    values.update(overrides)
    return SupervisorInvocationRequest(**values)  # type: ignore[arg-type]


def _repository(
    *,
    unique: bool = True,
    root: str = REPO_ROOT,
) -> RepositoryTargetResolution:
    values = {
        "repository_root": root,
        "repository_id": "repository:sha256:primary",
        "checkout_id": "checkout-main",
        "scope": root,
        "tree_id": _cid("tree"),
        "dirty_overlay": _cid("dirty"),
        "submodules": _cid("submodules"),
        "nested_repositories": _cid("nested"),
    }
    decisions: list[TargetInferenceDecision] = []
    if unique:
        for name, value in values.items():
            candidate = TargetCandidate(
                field_name=name,
                value=value,
                source=ResolutionSource.DISCOVERY,
                source_precedence=SOURCE_PRECEDENCE[ResolutionSource.DISCOVERY],
                evidence_cid=_cid(f"repo-{name}"),
            )
            decisions.append(
                TargetInferenceDecision(
                    field_name=name,
                    disposition=ResolutionDisposition.UNIQUE,
                    selected_value=value,
                    selected_source=ResolutionSource.DISCOVERY,
                    source_precedence=SOURCE_PRECEDENCE[
                        ResolutionSource.DISCOVERY
                    ],
                    evidence_cid=_cid(f"repo-{name}"),
                    candidates=(candidate,),
                    reason_codes=(),
                    effect=DecisionEffect.IDENTITY_ONLY,
                    override_accepted=False,
                    fresh_until_ms=0,
                    revalidation_rule=RevalidationRule.BEFORE_MUTATION,
                )
            )
        binding = RepositoryTargetBinding(
            repository_root=root,
            repository_id=values["repository_id"],
            checkout_id=values["checkout_id"],
            scope_path=root,
            tree_id=values["tree_id"],
            dirty_overlay_cid=values["dirty_overlay"],
            submodule_population_cid=values["submodules"],
            nested_repository_population_cid=values["nested_repositories"],
            head_commit="abc123def456",
            head_tree=values["tree_id"],
            dirty=False,
            descriptor_cid=_cid("descriptor"),
            selected_source=ResolutionSource.DISCOVERY,
            alias="primary",
        )
        return RepositoryTargetResolution(
            decisions=tuple(decisions),
            evidence_cid=_cid("repository-evidence"),
            binding=binding,
            unresolved_fields=(),
            reason_codes=(),
            candidates_considered=(),
        )

    # Ambiguous multi-root preview: no binding, unresolved identity fields.
    root_a = f"{root}-a"
    root_b = f"{root}-b"
    for name in values:
        if name == "repository_root":
            candidates = (
                TargetCandidate(
                    field_name=name,
                    value=root_a,
                    source=ResolutionSource.DISCOVERY,
                    source_precedence=SOURCE_PRECEDENCE[
                        ResolutionSource.DISCOVERY
                    ],
                    evidence_cid=_cid("root-a"),
                    confidence_ppm=500_000,
                ),
                TargetCandidate(
                    field_name=name,
                    value=root_b,
                    source=ResolutionSource.DISCOVERY,
                    source_precedence=SOURCE_PRECEDENCE[
                        ResolutionSource.DISCOVERY
                    ],
                    evidence_cid=_cid("root-b"),
                    confidence_ppm=500_000,
                ),
            )
            decisions.append(
                TargetInferenceDecision(
                    field_name=name,
                    disposition=ResolutionDisposition.AMBIGUOUS,
                    selected_value="",
                    selected_source=ResolutionSource.DISCOVERY,
                    source_precedence=SOURCE_PRECEDENCE[
                        ResolutionSource.DISCOVERY
                    ],
                    evidence_cid=_cid("repo-ambiguous"),
                    candidates=candidates,
                    reason_codes=("multiple_equal_rank_roots",),
                    effect=DecisionEffect.IDENTITY_ONLY,
                    override_accepted=False,
                    fresh_until_ms=0,
                    revalidation_rule=RevalidationRule.BEFORE_MUTATION,
                )
            )
        else:
            decisions.append(
                TargetInferenceDecision(
                    field_name=name,
                    disposition=ResolutionDisposition.UNAVAILABLE,
                    selected_value="",
                    selected_source=ResolutionSource.DISCOVERY,
                    source_precedence=SOURCE_PRECEDENCE[
                        ResolutionSource.DISCOVERY
                    ],
                    evidence_cid=_cid(f"repo-{name}-unavail"),
                    candidates=(),
                    reason_codes=("repository_root_unresolved",),
                    effect=DecisionEffect.IDENTITY_ONLY,
                    override_accepted=False,
                    fresh_until_ms=0,
                    revalidation_rule=RevalidationRule.BEFORE_MUTATION,
                )
            )
    return RepositoryTargetResolution(
        decisions=tuple(decisions),
        evidence_cid=_cid("repository-ambiguous-evidence"),
        binding=None,
        unresolved_fields=tuple(sorted(values)),
        reason_codes=("multiple_equal_rank_roots",),
        candidates_considered=(),
    )


def _state(repository_id: str = "repository:sha256:primary"):
    return resolve_state(
        StateResolutionEvidence(
            repository_id=repository_id,
            repository_root=REPO_ROOT,
            checkout_id="checkout-main",
            home_directory="/home/dev",
            environ={},
        )
    )


def _objective(state_root: str, run_namespace: str):
    return resolve_objectives(
        ObjectiveResolutionEvidence(
            repository_root=REPO_ROOT,
            state_root=state_root,
            prompt_cid=_prompt_cid(),
            repository_id="repository:sha256:primary",
            run_namespace=run_namespace,
            duckdb_available=True,
        )
    )


def _authority(principal: str = PRINCIPAL):
    return resolve_authority(
        AuthorityResolutionRequest(
            mode=InvocationMode.WORKTREE,
            local_worktree_authority=install_local_worktree_authority(
                principal,
                signing_key_handle="key:local-worktree-1",
                installed_at_ms=1_700_000_000_000,
            ),
        )
    )


def _provider(provider_id: str) -> ProviderCapabilityEvidence:
    return ProviderCapabilityEvidence(
        provider_id=provider_id,
        capability=PreferredProviderCapability.AVAILABLE,
        policy_allowed=True,
        healthy=True,
        authenticated=True,
        observed_capability_cid=_cid(f"{provider_id}-capability"),
        usage_evidence_cid=_cid(f"{provider_id}-usage"),
        budget_cid=_cid(f"{provider_id}-budget"),
        max_concurrency=4,
        request_headroom=10,
    )


def _capability(state_root: str, *, principal: str = PRINCIPAL):
    return resolve_capabilities(
        CapabilityEvidence(
            providers={
                PREFERRED_PROVIDER: _provider(PREFERRED_PROVIDER),
                FALLBACK_PROVIDER: _provider(FALLBACK_PROVIDER),
            },
            resources=ResourceSampleEvidence(
                ready_width=4,
                host_worker_limit=8,
                host_available_workers=6,
                max_processes=8,
                max_validation_workers=4,
                cpu_millis=8_000,
                memory_bytes=8 * 1024**3,
                provider_request_limit=100,
                deadline_ms=3_600_000,
                lane_labels=("alpha", "beta"),
            ),
            validation=ValidationPolicyEvidence(
                allowlisted_argv=(("python", "-m", "pytest", "-q"),),
                policy_cid=_cid("validation-policy"),
            ),
            topology=TopologyEvidence(
                distributed_capable=False,
                shard_count=1,
                owner_principal_ref=principal,
                state_root=state_root,
                database_relative_path="coordination.duckdb",
                coordinator_cid=_cid("coordinator"),
                lease_namespace="repo-run",
                fencing_generation=3,
                ipfs_publish_capable=True,
                parquet_capable=True,
                preferred_mode=TopologyMode.LOCAL,
                ipfs_backend_handle="ipfs-kit:development",
            ),
            task_revision_cid=_cid("task-revision"),
            attempt_cid=_cid("attempt"),
            worktree_cid=_cid("worktree"),
        )
    )


def _request(
    *,
    unique_repository: bool = True,
    authority: object | None = None,
    profile_layers: tuple[ProfileSourceLayer, ...] = (),
    canonical_request: CanonicalRequestBinding | None = None,
    invocation: SupervisorInvocationRequest | None = None,
    merge_target_hint: str = "",
    worktree_strategy_override: WorktreeStrategy | None = None,
) -> ProfileCompositionRequest:
    repository = _repository(unique=unique_repository)
    state = _state()
    objective = _objective(state.state_root, state.run_namespace)
    auth = authority if authority is not None else _authority()
    capability = _capability(state.state_root)
    inv = invocation or _invocation(
        canonical_request_cid=(
            canonical_request.request_cid if canonical_request is not None else ""
        )
    )
    return ProfileCompositionRequest(
        invocation=inv,
        repository=repository,
        state=state,
        objective=objective,
        authority=auth,  # type: ignore[arg-type]
        capability=capability,
        profile_layers=profile_layers,
        canonical_request=canonical_request,
        resolved_at_ms=1_700_000_000_000,
        fresh_until_ms=1_700_000_060_000,
        merge_target_hint=merge_target_hint,
        worktree_strategy_override=worktree_strategy_override,
    )


def test_requirement_id_is_stable() -> None:
    assert RESOLVED_SUPERVISOR_PROFILE_REQUIREMENT_ID.startswith("requirement:")
    assert "resolved-supervisor-profile" in RESOLVED_SUPERVISOR_PROFILE_REQUIREMENT_ID


def test_source_precedence_ladder_lower_wins() -> None:
    assert (
        SOURCE_PRECEDENCE[ResolutionSource.CANONICAL_REQUEST]
        < SOURCE_PRECEDENCE[ResolutionSource.EXPLICIT_OVERRIDE]
        < SOURCE_PRECEDENCE[ResolutionSource.EXISTING_RUN]
        < SOURCE_PRECEDENCE[ResolutionSource.AUTHENTICATED_TRANSPORT]
        < SOURCE_PRECEDENCE[ResolutionSource.SIGNED_PROFILE]
        < SOURCE_PRECEDENCE[ResolutionSource.REPOSITORY_HINT]
        < SOURCE_PRECEDENCE[ResolutionSource.DISCOVERY]
        < SOURCE_PRECEDENCE[ResolutionSource.BUILTIN_DEFAULT]
    )


def test_builtin_profile_for_mode() -> None:
    assert builtin_profile_for_mode(InvocationMode.PREVIEW) == BUILTIN_PROFILE_PREVIEW
    assert (
        builtin_profile_for_mode(InvocationMode.WORKTREE)
        == BUILTIN_PROFILE_LOCAL_WORKTREE
    )


def test_happy_path_composes_complete_receipt_and_profile() -> None:
    resolution = resolve_supervisor_profile(_request())

    assert resolution.requirement_id == RESOLVED_SUPERVISOR_PROFILE_REQUIREMENT_ID
    assert not resolution.inference_disabled
    assert not resolution.effects_blocked
    assert not resolution.safe_preview
    assert resolution.profile is not None
    assert resolution.receipt.is_authorization is False
    assert resolution.authorizes_effects is False

    names = {item.field_name for item in resolution.decisions}
    assert names == set(REQUIRED_TARGET_DECISION_FIELDS)
    assert set(PROFILE_OWNED_FIELDS).issubset(names)
    assert not resolution.receipt.unresolved_fields

    profile = resolution.profile
    assert profile.profile_name in {
        BUILTIN_PROFILE_LOCAL_WORKTREE,
        "local-worktree",
    }
    assert profile.repository_root == REPO_ROOT
    assert profile.state_root == resolution.receipt.state_root
    assert profile.worktree_strategy is WorktreeStrategy.ISOLATED
    assert ExpectedEffect.EDIT_ISOLATED_WORKTREE in profile.expected_effects
    assert ExpectedEffect.MERGE not in profile.expected_effects
    assert ExpectedEffect.PUSH not in profile.expected_effects
    assert profile.target_resolution_receipt_cid == resolution.receipt.content_id
    assert profile.provider_route.preferred_provider == PREFERRED_PROVIDER
    assert "python" in profile.supervisor_argv
    assert profile.coordination_shard.writable is True
    assert profile.principal_ref == PRINCIPAL


def test_composition_is_deterministic() -> None:
    request = _request()
    first = resolve_supervisor_profile(request)
    second = resolve_supervisor_profile(request)

    assert first.receipt.content_id == second.receipt.content_id
    assert first.profile is not None and second.profile is not None
    assert first.profile.content_id == second.profile.content_id
    assert first.to_dict()["profile_source_cid"] == second.to_dict()[
        "profile_source_cid"
    ]


def test_signed_profile_beats_repository_hint_and_defaults() -> None:
    signed = ProfileSourceLayer(
        kind=ProfileSourceKind.SIGNED_PROFILE,
        evidence_cid=_cid("signed-profile"),
        profile_name="local-worktree",
        allowed_effects=LOCAL_WORKTREE_ALLOWED_EFFECTS,
        worktree_strategy=WorktreeStrategy.ISOLATED,
        merge_target="release",
        max_lanes=2,
        signature_verified=True,
        reviewed=True,
    )
    repo_hint = ProfileSourceLayer(
        kind=ProfileSourceKind.REPOSITORY_HINT,
        evidence_cid=_cid("repo-hint"),
        profile_name="ci-worker",
        allowed_effects=(
            *LOCAL_WORKTREE_ALLOWED_EFFECTS,
            ExpectedEffect.MERGE,
            ExpectedEffect.PUSH,
        ),
        worktree_strategy=WorktreeStrategy.ISOLATED,
        merge_target="main",
        max_lanes=8,
        reviewed=True,
    )
    resolution = resolve_supervisor_profile(
        _request(profile_layers=(signed, repo_hint))
    )

    assert resolution.profile is not None
    assert resolution.receipt.merge_target == "release"
    assert resolution.receipt.lane_ceiling == 2
    assert ExpectedEffect.MERGE not in resolution.expected_effects
    assert ExpectedEffect.PUSH not in resolution.expected_effects
    assert "lower_source_effect_widen_ignored" in resolution.reason_codes or any(
        "widen" in code for code in resolution.reason_codes
    )


def test_existing_run_binding_beats_signed_profile() -> None:
    """Run bindings (EXISTING_RUN) outrank signed profiles on the ladder."""

    run_binding = ProfileSourceLayer(
        kind=ProfileSourceKind.EXISTING_RUN,
        evidence_cid=_cid("run-binding"),
        profile_name="local-worktree",
        allowed_effects=LOCAL_WORKTREE_ALLOWED_EFFECTS,
        worktree_strategy=WorktreeStrategy.ISOLATED,
        merge_target="run-bound-branch",
        max_lanes=1,
        reviewed=True,
        signature_verified=True,
    )
    signed = ProfileSourceLayer(
        kind=ProfileSourceKind.SIGNED_PROFILE,
        evidence_cid=_cid("signed-lower"),
        profile_name="ci-worker",
        allowed_effects=LOCAL_WORKTREE_ALLOWED_EFFECTS,
        worktree_strategy=WorktreeStrategy.ISOLATED,
        merge_target="signed-branch",
        max_lanes=4,
        signature_verified=True,
    )
    resolution = resolve_supervisor_profile(
        _request(profile_layers=(run_binding, signed))
    )

    assert resolution.receipt.merge_target == "run-bound-branch"
    assert resolution.receipt.lane_ceiling == 1
    merge_decision = resolution.decision_for("merge_target")
    assert merge_decision.selected_source is ResolutionSource.EXISTING_RUN


def test_authenticated_server_policy_beats_repository_hint() -> None:
    """Authenticated/server policy outranks reviewed repository hints."""

    server_policy = ProfileSourceLayer(
        kind=ProfileSourceKind.AUTHENTICATED_SERVER_POLICY,
        evidence_cid=_cid("server-policy"),
        profile_name="local-worktree",
        allowed_effects=LOCAL_WORKTREE_ALLOWED_EFFECTS,
        worktree_strategy=WorktreeStrategy.ISOLATED,
        merge_target="policy-branch",
        max_lanes=2,
        signature_verified=True,
        reviewed=True,
    )
    repo_hint = ProfileSourceLayer(
        kind=ProfileSourceKind.REPOSITORY_HINT,
        evidence_cid=_cid("repo-hint-lower"),
        profile_name="ci-worker",
        allowed_effects=LOCAL_WORKTREE_ALLOWED_EFFECTS,
        worktree_strategy=WorktreeStrategy.ISOLATED,
        merge_target="repo-branch",
        max_lanes=8,
        reviewed=True,
    )
    resolution = resolve_supervisor_profile(
        _request(profile_layers=(server_policy, repo_hint))
    )

    assert resolution.receipt.merge_target == "policy-branch"
    assert resolution.receipt.lane_ceiling == 2
    merge_decision = resolution.decision_for("merge_target")
    assert merge_decision.selected_source is ResolutionSource.AUTHENTICATED_TRANSPORT


def test_lower_source_cannot_widen_worktree_or_lanes() -> None:
    narrow = ProfileSourceLayer(
        kind=ProfileSourceKind.SIGNED_PROFILE,
        evidence_cid=_cid("signed-narrow"),
        profile_name="preview",
        allowed_effects=(ExpectedEffect.INSPECT_REPOSITORY,),
        worktree_strategy=WorktreeStrategy.NONE,
        max_lanes=1,
        signature_verified=True,
    )
    widen = ProfileSourceLayer(
        kind=ProfileSourceKind.REPOSITORY_HINT,
        evidence_cid=_cid("repo-widen"),
        profile_name="local-worktree",
        allowed_effects=LOCAL_WORKTREE_ALLOWED_EFFECTS,
        worktree_strategy=WorktreeStrategy.ISOLATED,
        max_lanes=16,
        reviewed=True,
    )
    resolution = resolve_supervisor_profile(
        _request(profile_layers=(narrow, widen))
    )

    assert resolution.receipt.worktree_strategy is WorktreeStrategy.NONE
    assert resolution.receipt.lane_ceiling == 1
    assert resolution.effects_blocked is True
    assert resolution.safe_preview is True
    assert resolution.profile is not None
    assert resolution.profile.mode is InvocationMode.PREVIEW
    assert resolution.profile.coordination_shard.writable is False
    assert resolution.profile.replication.ipfs_publish is False


def test_material_ambiguity_blocks_effects_preserves_safe_preview_receipt() -> None:
    resolution = resolve_supervisor_profile(_request(unique_repository=False))

    assert resolution.effects_blocked is True
    assert resolution.safe_preview is True
    assert "repository_root" in resolution.receipt.unresolved_fields
    assert resolution.receipt.repository_root == ""
    assert resolution.receipt.worktree_strategy is WorktreeStrategy.NONE
    assert resolution.receipt.coordination_shard.writable is False
    assert resolution.receipt.replication.ipfs_publish is False
    assert resolution.receipt.replication.pin is False
    assert resolution.receipt.is_authorization is False
    # Profile may be withheld without a unique repository root.
    assert resolution.profile is None or resolution.profile.mode is InvocationMode.PREVIEW


def test_authority_denial_blocks_effects_and_keeps_receipt() -> None:
    denied = resolve_authority(
        AuthorityResolutionRequest(
            mode=InvocationMode.WORKTREE,
            prompt_claimed_principal="did:key:attacker",
            credentials_present=True,
        )
    )
    assert not denied.authorized

    resolution = resolve_supervisor_profile(_request(authority=denied))

    assert resolution.effects_blocked is True
    assert resolution.safe_preview is True
    assert resolution.receipt.worktree_strategy is WorktreeStrategy.NONE
    assert resolution.receipt.coordination_shard.writable is False
    assert resolution.receipt.is_authorization is False


def test_explicit_override_merge_target() -> None:
    resolution = resolve_supervisor_profile(
        _request(merge_target_hint="feature/ase-010")
    )
    assert resolution.receipt.merge_target == "feature/ase-010"
    assert resolution.decision_for("merge_target").selected_value == "feature/ase-010"
    assert (
        resolution.decision_for("merge_target").selected_source
        is ResolutionSource.EXPLICIT_OVERRIDE
    )


def test_canonical_request_disables_inference() -> None:
    base = resolve_supervisor_profile(_request())
    assert base.profile is not None

    authority_fields = {
        "policy",
        "principal",
        "authority_source",
        "effect_ceiling",
    }
    canonical_decisions = []
    for item in base.decisions:
        if item.field_name in authority_fields:
            source = ResolutionSource.EXISTING_RUN
            effect = DecisionEffect.REQUIRES_AUTHORITY
            reasons = ("canonical_authority_binding",)
        else:
            source = ResolutionSource.CANONICAL_REQUEST
            effect = item.effect
            reasons = ("canonical_request_field",)
        candidate = TargetCandidate(
            field_name=item.field_name,
            value=item.selected_value,
            source=source,
            source_precedence=SOURCE_PRECEDENCE[source],
            evidence_cid=_cid(f"canonical-{item.field_name}"),
        )
        canonical_decisions.append(
            TargetInferenceDecision(
                field_name=item.field_name,
                disposition=ResolutionDisposition.UNIQUE,
                selected_value=item.selected_value,
                selected_source=source,
                source_precedence=SOURCE_PRECEDENCE[source],
                evidence_cid=_cid(f"canonical-{item.field_name}"),
                candidates=(candidate,),
                reason_codes=reasons,
                effect=effect,
                override_accepted=False,
                fresh_until_ms=0,
                revalidation_rule=RevalidationRule.IMMUTABLE,
            )
        )

    request_cid = _cid("canonical-request")
    canonical = CanonicalRequestBinding(
        request_cid=request_cid,
        decisions=tuple(canonical_decisions),
        provider_route=base.receipt.provider_route,
        resource_budget=base.profile.resource_budget,
        coordination_shard=base.receipt.coordination_shard,
        replication=base.receipt.replication,
        task_source_kind=base.receipt.task_source_kind,
        objective_revision_cid=base.receipt.objective_revision_cid,
        task_source_revision_cid=base.receipt.task_source_revision_cid,
        markdown_path=base.receipt.markdown_path,
        duckdb_path=base.receipt.duckdb_path,
        profile_name=base.profile.profile_name,
        expected_effects=base.profile.expected_effects,
        environment_names=base.profile.environment_names,
        credential_handles=base.profile.credential_handles,
        lifecycle_health_contract_cid=base.profile.lifecycle_health_contract_cid,
        supervisor_argv=base.profile.supervisor_argv,
        daemon_argv=base.profile.daemon_argv,
        task_source_path=base.profile.task_source_path,
        mode=InvocationMode.WORKTREE,
    )
    resolution = resolve_supervisor_profile(
        _request(
            canonical_request=canonical,
            invocation=_invocation(canonical_request_cid=request_cid),
        )
    )

    assert resolution.inference_disabled is True
    assert resolution.profile is not None
    assert "inference_disabled" in resolution.reason_codes
    assert resolution.receipt.content_id
    non_authority = [
        item
        for item in resolution.decisions
        if item.field_name not in authority_fields
    ]
    assert all(
        item.selected_source is ResolutionSource.CANONICAL_REQUEST
        for item in non_authority
    )


def test_canonical_binding_rejects_incomplete_decisions() -> None:
    with pytest.raises(ProfileResolverError, match="missing"):
        CanonicalRequestBinding(
            request_cid=_cid("incomplete"),
            decisions=(),
        )


def test_unsigned_signed_profile_layer_rejected() -> None:
    with pytest.raises(ProfileResolverError, match="signature_verified"):
        ProfileSourceLayer(
            kind=ProfileSourceKind.SIGNED_PROFILE,
            evidence_cid=_cid("unsigned"),
            profile_name="local-worktree",
            signature_verified=False,
        )


def test_unreviewed_repository_hint_rejected() -> None:
    with pytest.raises(ProfileResolverError, match="reviewed"):
        ProfileSourceLayer(
            kind=ProfileSourceKind.REPOSITORY_HINT,
            evidence_cid=_cid("unreviewed"),
            profile_name="local-worktree",
            reviewed=False,
        )


def test_server_policy_requires_verification_or_review() -> None:
    with pytest.raises(ProfileResolverError, match="verification or review"):
        ProfileSourceLayer(
            kind=ProfileSourceKind.AUTHENTICATED_SERVER_POLICY,
            evidence_cid=_cid("unverified-policy"),
            profile_name="local-worktree",
            reviewed=False,
            signature_verified=False,
        )


def test_receipt_builder_round_trip() -> None:
    resolution = resolve_supervisor_profile(_request())
    rebuilt = build_target_resolution_receipt(
        invocation=_invocation(),
        decisions=resolution.decisions,
        repository_root=resolution.receipt.repository_root,
        repository_id=resolution.receipt.repository_id,
        checkout_id=resolution.receipt.checkout_id,
        scope_path=resolution.receipt.scope_path,
        head_tree_cid=resolution.receipt.head_tree_cid,
        dirty_overlay_cid=resolution.receipt.dirty_overlay_cid,
        submodule_population_cid=resolution.receipt.submodule_population_cid,
        nested_repository_population_cid=(
            resolution.receipt.nested_repository_population_cid
        ),
        state_root=resolution.receipt.state_root,
        run_namespace=resolution.receipt.run_namespace,
        objective_cid=resolution.receipt.objective_cid,
        objective_revision_cid=resolution.receipt.objective_revision_cid,
        plan_cid=resolution.receipt.plan_cid,
        task_source_cid=resolution.receipt.task_source_cid,
        task_source_revision_cid=resolution.receipt.task_source_revision_cid,
        task_source_kind=resolution.receipt.task_source_kind,
        policy_cid=resolution.receipt.policy_cid,
        principal_ref=resolution.receipt.principal_ref,
        authority_source_ref=resolution.receipt.authority_source_ref,
        effect_ceiling_cid=resolution.receipt.effect_ceiling_cid,
        output_mode=resolution.receipt.output_mode,
        markdown_path=resolution.receipt.markdown_path,
        duckdb_path=resolution.receipt.duckdb_path,
        provider_route=resolution.receipt.provider_route,
        capability_report_cid=resolution.receipt.capability_report_cid,
        resource_budget_cid=resolution.receipt.resource_budget_cid,
        lane_ceiling=resolution.receipt.lane_ceiling,
        merge_target=resolution.receipt.merge_target,
        worktree_strategy=resolution.receipt.worktree_strategy,
        validation_profile_cid=resolution.receipt.validation_profile_cid,
        coordination_shard=resolution.receipt.coordination_shard,
        replication=resolution.receipt.replication,
        configuration_root_cid=resolution.receipt.configuration_root_cid,
        capability_catalog_cid=resolution.receipt.capability_catalog_cid,
        resolved_at_ms=resolution.receipt.resolved_at_ms,
        fresh_until_ms=resolution.receipt.fresh_until_ms,
    )
    # Same projections with a different invocation CID produce a different
    # receipt identity; structural fields still validate.
    assert rebuilt.repository_root == resolution.receipt.repository_root
    assert rebuilt.is_authorization is False
    assert set(item.field_name for item in rebuilt.decisions) == set(
        REQUIRED_TARGET_DECISION_FIELDS
    )


def test_resolver_class_matches_module_function() -> None:
    request = _request()
    via_class = SupervisorProfileResolver().resolve(request)
    via_fn = resolve_supervisor_profile(request)
    assert via_class.receipt.content_id == via_fn.receipt.content_id
    assert via_class.profile is not None and via_fn.profile is not None
    assert via_class.profile.content_id == via_fn.profile.content_id


def test_precedence_trace_covers_required_fields() -> None:
    resolution = resolve_supervisor_profile(_request())
    traced = {item.field_name for item in resolution.precedence_trace}
    assert set(REQUIRED_TARGET_DECISION_FIELDS).issubset(traced)
    assert "profile_name" in traced
    assert "expected_effects" in traced


def test_profile_cid_changes_when_lane_ceiling_changes() -> None:
    base = resolve_supervisor_profile(_request())
    narrow = resolve_supervisor_profile(
        _request(
            profile_layers=(
                ProfileSourceLayer(
                    kind=ProfileSourceKind.SIGNED_PROFILE,
                    evidence_cid=_cid("lanes-2"),
                    profile_name="local-worktree",
                    allowed_effects=LOCAL_WORKTREE_ALLOWED_EFFECTS,
                    worktree_strategy=WorktreeStrategy.ISOLATED,
                    max_lanes=2,
                    signature_verified=True,
                ),
            )
        )
    )
    assert base.profile is not None and narrow.profile is not None
    assert base.profile.content_id != narrow.profile.content_id
    assert narrow.receipt.lane_ceiling == 2


def test_missing_canonical_binding_when_invocation_claims_cid() -> None:
    with pytest.raises(ProfileResolverError, match="CanonicalRequestBinding"):
        ProfileCompositionRequest(
            invocation=_invocation(canonical_request_cid=_cid("claimed")),
            repository=_repository(),
            state=_state(),
            objective=_objective(_state().state_root, _state().run_namespace),
            authority=_authority(),
            capability=_capability(_state().state_root),
        )


def test_current_checkout_strategy_denied_from_layers() -> None:
    layer = ProfileSourceLayer(
        kind=ProfileSourceKind.SIGNED_PROFILE,
        evidence_cid=_cid("current-checkout"),
        profile_name="local-worktree",
        allowed_effects=LOCAL_WORKTREE_ALLOWED_EFFECTS,
        worktree_strategy=WorktreeStrategy.CURRENT_CHECKOUT,
        signature_verified=True,
    )
    resolution = resolve_supervisor_profile(_request(profile_layers=(layer,)))
    assert resolution.receipt.worktree_strategy is not WorktreeStrategy.CURRENT_CHECKOUT
    assert "current_checkout_rewrite_denied" in resolution.reason_codes
