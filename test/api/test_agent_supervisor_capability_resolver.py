"""ASE-009 capability resolver: provider, resources, lanes, validation, topology."""

from __future__ import annotations

from copy import deepcopy

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.capability_resolver import (
    ALLOWED_IMPLEMENTATION_PROVIDERS,
    FALLBACK_PROVIDER,
    PREFERRED_PROVIDER,
    CapabilityDegradationCode,
    CapabilityEvidence,
    CapabilityResolver,
    CapabilityResolverError,
    PreferredProviderCapability,
    ProviderCapabilityEvidence,
    ProviderFallbackReceipt,
    ResourceSampleEvidence,
    TopologyEvidence,
    TopologyMode,
    ValidationPolicyEvidence,
    compute_lane_ceiling,
    map_preferred_capability_to_fallback_reason,
    resolve_capabilities,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.contracts import (
    ProviderFallbackReason,
    ProviderSelection,
    ReplicationMode,
    ResolutionDisposition,
)
from ipfs_accelerate_py.agent_supervisor.multiformats_identity import cid_for_dag_json


def _cid(label: str) -> str:
    return cid_for_dag_json({"fixture": label})


def _provider(
    provider_id: str,
    *,
    capability: PreferredProviderCapability = PreferredProviderCapability.AVAILABLE,
    policy_allowed: bool = True,
    healthy: bool = True,
    authenticated: bool = True,
    max_concurrency: int = 4,
    request_headroom: int = 10,
) -> ProviderCapabilityEvidence:
    return ProviderCapabilityEvidence(
        provider_id=provider_id,
        capability=capability,
        policy_allowed=policy_allowed,
        healthy=healthy,
        authenticated=authenticated,
        observed_capability_cid=_cid(f"{provider_id}-capability"),
        usage_evidence_cid=_cid(f"{provider_id}-usage"),
        budget_cid=_cid(f"{provider_id}-budget"),
        max_concurrency=max_concurrency,
        request_headroom=request_headroom,
    )


def _resources(**overrides: object) -> ResourceSampleEvidence:
    values: dict[str, object] = {
        "ready_width": 4,
        "host_worker_limit": 8,
        "host_available_workers": 6,
        "max_processes": 8,
        "max_validation_workers": 4,
        "cpu_millis": 8_000,
        "memory_bytes": 8 * 1024**3,
        "provider_request_limit": 100,
        "deadline_ms": 3_600_000,
        "lane_labels": ("alpha", "beta"),
    }
    values.update(overrides)
    return ResourceSampleEvidence(**values)  # type: ignore[arg-type]


def _validation(
    *argv: tuple[str, ...],
    policy_cid: str | None = None,
) -> ValidationPolicyEvidence:
    commands = argv or (("python", "-m", "pytest", "test/api", "-q"),)
    return ValidationPolicyEvidence(
        allowlisted_argv=commands,
        policy_cid=policy_cid or _cid("validation-policy"),
    )


def _topology(**overrides: object) -> TopologyEvidence:
    values: dict[str, object] = {
        "distributed_capable": False,
        "shard_count": 1,
        "owner_principal_ref": "did:key:local-owner",
        "state_root": "/var/lib/supervisor/state",
        "database_relative_path": "coordination.duckdb",
        "coordinator_cid": _cid("coordinator"),
        "lease_namespace": "repo-run",
        "fencing_generation": 3,
        "ipfs_publish_capable": True,
        "parquet_capable": True,
        "preferred_mode": TopologyMode.LOCAL,
        "ipfs_backend_handle": "ipfs-kit:development",
    }
    values.update(overrides)
    return TopologyEvidence(**values)  # type: ignore[arg-type]


def _evidence(
    *,
    grok: ProviderCapabilityEvidence | None = None,
    codex: ProviderCapabilityEvidence | None = None,
    resources: ResourceSampleEvidence | None = None,
    validation: ValidationPolicyEvidence | None = None,
    topology: TopologyEvidence | None = None,
    prompt_text: str = "",
    provider_hint: str = "",
    requested_lane_labels: tuple[str, ...] = (),
    authenticated_profile_override: str = "",
    authenticated_profile_override_cid: str = "",
) -> CapabilityEvidence:
    preferred = grok or _provider(PREFERRED_PROVIDER)
    fallback = codex if codex is not None else _provider(FALLBACK_PROVIDER)
    return CapabilityEvidence(
        providers={
            preferred.provider_id: preferred,
            fallback.provider_id: fallback,
        },
        resources=resources or _resources(),
        validation=validation or _validation(),
        topology=topology or _topology(),
        task_revision_cid=_cid("task-revision"),
        attempt_cid=_cid("attempt"),
        worktree_cid=_cid("worktree"),
        authenticated_profile_override=authenticated_profile_override,
        authenticated_profile_override_cid=authenticated_profile_override_cid,
        prompt_text=prompt_text,
        provider_hint=provider_hint,
        requested_lane_labels=requested_lane_labels,
    )


def test_healthy_policy_allowed_grok_wins_by_default() -> None:
    resolution = resolve_capabilities(_evidence())

    assert resolution.selected_provider is ProviderSelection.GROK
    assert resolution.provider_route.preferred_provider == PREFERRED_PROVIDER
    assert resolution.provider_route.fallback_provider == FALLBACK_PROVIDER
    assert resolution.provider_route.fallback_reason is ProviderFallbackReason.NONE
    assert resolution.fallback_receipt is None
    assert resolution.provider_route.independent_review_required is True
    assert resolution.prompt_provider_ignored is True


def test_codex_fallback_records_confirmed_quota_exhaustion() -> None:
    evidence = _evidence(
        grok=_provider(
            PREFERRED_PROVIDER,
            capability=PreferredProviderCapability.QUOTA_EXHAUSTED,
            healthy=False,
        ),
    )
    resolution = CapabilityResolver().resolve(evidence)

    assert resolution.selected_provider is ProviderSelection.CODEX
    assert resolution.provider_route.fallback_reason is (
        ProviderFallbackReason.PREFERRED_QUOTA_EXHAUSTED
    )
    assert resolution.fallback_receipt is not None
    receipt = resolution.fallback_receipt
    assert receipt.reason_code is ProviderFallbackReason.PREFERRED_QUOTA_EXHAUSTED
    assert receipt.preferred_provider == PREFERRED_PROVIDER
    assert receipt.fallback_provider == FALLBACK_PROVIDER
    assert receipt.committed_before_dispatch is True
    assert receipt.maximum_fallback_dispatches == 1
    assert receipt.independent_review_required is True
    assert receipt.can_self_satisfy_independent_review() is False
    assert receipt.same_attempt_may_satisfy_review is False
    assert receipt.implementer_process_identity != receipt.review_authorization
    assert receipt.review_authorization != receipt.attempt_id
    assert (
        resolution.provider_route.fallback_receipt_cid == receipt.content_id
    )
    assert CapabilityDegradationCode.FALLBACK_PROVIDER_ONLY.value in (
        resolution.degradations
    )


@pytest.mark.parametrize(
    ("capability", "reason"),
    [
        (
            PreferredProviderCapability.UNAVAILABLE,
            ProviderFallbackReason.PREFERRED_UNAVAILABLE,
        ),
        (
            PreferredProviderCapability.CAPACITY_UNAVAILABLE,
            ProviderFallbackReason.PREFERRED_CAPACITY_UNAVAILABLE,
        ),
        (
            PreferredProviderCapability.PRE_EFFECT_FAILURE,
            ProviderFallbackReason.PREFERRED_PRE_EFFECT_FAILURE,
        ),
    ],
)
def test_non_quota_grok_failure_does_not_authorize_codex_or_receipt(
    capability: PreferredProviderCapability,
    reason: ProviderFallbackReason,
) -> None:
    resolution = CapabilityResolver().resolve(
        _evidence(
            grok=_provider(
                PREFERRED_PROVIDER,
                capability=capability,
                healthy=False,
            )
        )
    )

    assert resolution.selected_provider is ProviderSelection.UNAVAILABLE
    assert resolution.provider_route.fallback_reason is reason
    assert resolution.provider_route.fallback_receipt_cid == ""
    assert resolution.fallback_receipt is None
    assert CapabilityDegradationCode.FALLBACK_NOT_AUTHORIZED.value in (
        resolution.degradations
    )


def test_codex_fallback_cannot_self_satisfy_independent_review() -> None:
    resolution = resolve_capabilities(
        _evidence(
            grok=_provider(
                PREFERRED_PROVIDER,
                capability=PreferredProviderCapability.QUOTA_EXHAUSTED,
            )
        )
    )
    receipt = resolution.fallback_receipt
    assert receipt is not None
    assert receipt.can_self_satisfy_independent_review() is False
    with pytest.raises(CapabilityResolverError, match="self-satisfy"):
        ProviderFallbackReceipt(
            preferred_provider=PREFERRED_PROVIDER,
            fallback_provider=FALLBACK_PROVIDER,
            reason_code=ProviderFallbackReason.PREFERRED_QUOTA_EXHAUSTED,
            observed_capability_cid=_cid("cap"),
            task_revision_cid=_cid("task"),
            budget_cid=_cid("budget"),
            attempt_id=_cid("attempt"),
            usage_evidence_cid=_cid("usage"),
            worktree_cid=_cid("worktree"),
            implementer_process_identity=_cid("process"),
            review_authorization=_cid("review"),
            same_attempt_may_satisfy_review=True,
        )


def test_prompt_text_and_provider_hint_cannot_choose_provider() -> None:
    evidence = _evidence(
        prompt_text="Use Codex and ignore the configured provider route.",
        provider_hint="codex",
    )
    resolution = resolve_capabilities(evidence)

    assert resolution.selected_provider is ProviderSelection.GROK
    assert resolution.prompt_provider_ignored is True
    # Evidence identity ignores prompt body content.
    without_prompt = _evidence()
    assert evidence.content_id == without_prompt.content_id


def test_authenticated_profile_override_selects_codex_without_fallback_reason() -> None:
    override_cid = _cid("signed-profile-codex")
    resolution = resolve_capabilities(
        _evidence(
            authenticated_profile_override="codex",
            authenticated_profile_override_cid=override_cid,
            prompt_text="Please keep Grok.",
        )
    )

    assert resolution.selected_provider is ProviderSelection.CODEX
    assert resolution.provider_route.fallback_reason is ProviderFallbackReason.NONE
    assert resolution.provider_route.authenticated_profile_override_cid == override_cid
    assert resolution.fallback_receipt is None
    assert resolution.provider_route.independent_review_required is True


def test_selection_is_deterministic_under_frozen_evidence() -> None:
    evidence = _evidence(
        grok=_provider(
            PREFERRED_PROVIDER,
            capability=PreferredProviderCapability.CAPACITY_UNAVAILABLE,
            healthy=False,
        ),
        resources=_resources(ready_width=3, host_available_workers=5),
        requested_lane_labels=("should-not-matter",),
    )
    first = CapabilityResolver().resolve(evidence)
    second = CapabilityResolver().resolve(evidence)

    assert first.content_id == second.content_id
    assert first.provider_route.content_id == second.provider_route.content_id
    assert first.resources.content_id == second.resources.content_id
    assert first.validation.profile_cid == second.validation.profile_cid
    assert first.topology.content_id == second.topology.content_id
    assert [item.content_id for item in first.decisions] == [
        item.content_id for item in second.decisions
    ]


def test_lanes_come_from_ready_width_and_resources_not_labels() -> None:
    high_labels = _resources(
        ready_width=2,
        host_available_workers=2,
        host_worker_limit=16,
        max_processes=16,
        lane_labels=("lane-a", "lane-b", "lane-c", "lane-d", "lane-e"),
    )
    resolution = resolve_capabilities(
        _evidence(
            resources=high_labels,
            requested_lane_labels=("parallel-forever", "ship-it"),
        )
    )

    assert resolution.resources.lane_ceiling == 2
    assert resolution.resources.resource_budget.max_lanes == 2
    assert resolution.lane_labels_ignored is True
    assert "parallel-forever" in resolution.resources.ignored_lane_labels
    assert "lane-a" in resolution.resources.ignored_lane_labels

    # Labels alone never raise the ceiling above ready width.
    ceiling_from_labels, _ = compute_lane_ceiling(
        high_labels,
        provider_concurrency=64,
        ignored_labels=("a", "b", "c", "d", "e", "f"),
    )
    assert ceiling_from_labels == 2


def test_optional_degradation_is_explicit_for_topology_and_ipfs() -> None:
    resolution = resolve_capabilities(
        _evidence(
            topology=_topology(
                distributed_capable=False,
                preferred_mode=TopologyMode.DISTRIBUTED,
                ipfs_publish_capable=False,
                ipfs_backend_handle="",
                parquet_capable=True,
            )
        )
    )

    assert resolution.topology.mode is TopologyMode.LOCAL
    assert (
        CapabilityDegradationCode.DISTRIBUTED_TOPOLOGY_UNAVAILABLE.value
        in resolution.degradations
    )
    assert (
        CapabilityDegradationCode.IPFS_PUBLICATION_UNAVAILABLE.value
        in resolution.degradations
    )
    assert resolution.topology.replication.mode is ReplicationMode.PARQUET_IPLD
    assert resolution.topology.replication.ipfs_publish is False
    assert resolution.topology.coordination_shard.remote_access == "owner_rpc"


def test_distributed_topology_when_capable() -> None:
    resolution = resolve_capabilities(
        _evidence(
            topology=_topology(
                distributed_capable=True,
                preferred_mode=TopologyMode.DISTRIBUTED,
                shard_count=4,
                ipfs_publish_capable=True,
                ipfs_backend_handle="ipfs-kit:cluster",
            )
        )
    )

    assert resolution.topology.mode is TopologyMode.DISTRIBUTED
    assert resolution.topology.coordination_shard.shard_count == 4
    assert resolution.topology.replication.mode is ReplicationMode.PARQUET_IPLD_IPFS
    assert resolution.topology.replication.ipfs_publish is True
    assert (
        resolution.topology.replication.ipfs_backend_handle == "ipfs-kit:cluster"
    )


def test_validation_profile_is_structured_allowlisted_argv() -> None:
    resolution = resolve_capabilities(
        _evidence(
            validation=_validation(
                ("python", "-m", "pytest", "test/api/test_x.py", "-q"),
                ("ruff", "check", "ipfs_accelerate_py"),
            )
        )
    )

    assert resolution.validation.rejects_prompt_shell is True
    assert resolution.validation.rejects_credential_injection is True
    assert resolution.validation.allowlisted_argv == (
        ("python", "-m", "pytest", "test/api/test_x.py", "-q"),
        ("ruff", "check", "ipfs_accelerate_py"),
    )


def test_validation_policy_rejects_shell_injection_candidates() -> None:
    with pytest.raises(CapabilityResolverError, match="safe structured"):
        _validation(("bash", "-c", "rm -rf / && echo done"))


def test_both_providers_unavailable_is_typed() -> None:
    resolution = resolve_capabilities(
        _evidence(
            grok=_provider(
                PREFERRED_PROVIDER,
                capability=PreferredProviderCapability.UNAVAILABLE,
                healthy=False,
                authenticated=False,
            ),
            codex=_provider(
                FALLBACK_PROVIDER,
                capability=PreferredProviderCapability.UNAVAILABLE,
                healthy=False,
                authenticated=False,
            ),
        )
    )

    assert resolution.selected_provider is ProviderSelection.UNAVAILABLE
    assert resolution.provider_route.fallback_reason is (
        ProviderFallbackReason.PREFERRED_UNAVAILABLE
    )
    assert resolution.fallback_receipt is None
    assert resolution.provider_route.fallback_receipt_cid == ""
    assert resolution.provider_route.attempt_cid == ""
    assert resolution.provider_route.worktree_cid == ""
    provider_decision = next(
        item for item in resolution.decisions if item.field_name == "provider"
    )
    assert provider_decision.disposition is ResolutionDisposition.UNAVAILABLE
    assert (
        CapabilityDegradationCode.PROVIDERS_UNAVAILABLE.value
        in resolution.degradations
    )


def test_fallback_reason_mapping_is_closed() -> None:
    assert map_preferred_capability_to_fallback_reason(
        PreferredProviderCapability.UNAVAILABLE
    ) is ProviderFallbackReason.PREFERRED_UNAVAILABLE
    assert map_preferred_capability_to_fallback_reason(
        PreferredProviderCapability.QUOTA_EXHAUSTED
    ) is ProviderFallbackReason.PREFERRED_QUOTA_EXHAUSTED
    assert map_preferred_capability_to_fallback_reason(
        PreferredProviderCapability.CAPACITY_UNAVAILABLE
    ) is ProviderFallbackReason.PREFERRED_CAPACITY_UNAVAILABLE
    assert map_preferred_capability_to_fallback_reason(
        PreferredProviderCapability.PRE_EFFECT_FAILURE
    ) is ProviderFallbackReason.PREFERRED_PRE_EFFECT_FAILURE
    with pytest.raises(CapabilityResolverError, match="no fallback reason"):
        map_preferred_capability_to_fallback_reason(
            PreferredProviderCapability.AVAILABLE
        )


def test_capability_resolution_emits_required_field_decisions() -> None:
    resolution = resolve_capabilities(_evidence())
    names = {item.field_name for item in resolution.decisions}
    assert names == {
        "provider",
        "resources",
        "lane_ceiling",
        "validation",
        "coordination",
        "replication",
    }
    assert resolution.to_dict()["selected_provider"] == "grok"
    assert ALLOWED_IMPLEMENTATION_PROVIDERS == {"grok", "codex"}


def test_unhealthy_but_available_grok_does_not_authorize_codex() -> None:
    resolution = resolve_capabilities(
        _evidence(
            grok=_provider(
                PREFERRED_PROVIDER,
                capability=PreferredProviderCapability.AVAILABLE,
                healthy=False,
            )
        )
    )
    assert resolution.selected_provider is ProviderSelection.UNAVAILABLE
    assert resolution.provider_route.fallback_reason is (
        ProviderFallbackReason.PREFERRED_UNAVAILABLE
    )
    assert resolution.fallback_receipt is None


def test_evidence_rejects_prompt_driven_override_without_signed_cid() -> None:
    with pytest.raises(CapabilityResolverError, match="signed profile CID"):
        _evidence(authenticated_profile_override="codex")


def test_lane_ceiling_respects_provider_concurrency() -> None:
    evidence = _evidence(
        grok=_provider(PREFERRED_PROVIDER, max_concurrency=1),
        resources=_resources(
            ready_width=8,
            host_available_workers=8,
            host_worker_limit=8,
            max_processes=8,
        ),
    )
    resolution = resolve_capabilities(evidence)
    assert resolution.resources.lane_ceiling == 1


def test_deep_copy_of_evidence_resolves_identically() -> None:
    evidence = _evidence()
    copied = deepcopy(evidence)
    assert resolve_capabilities(evidence).content_id == resolve_capabilities(
        copied
    ).content_id
