"""ASE3-028: sole router-owned implementation-provider decision."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.control.capability_resolver import (
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
from ipfs_accelerate_py.agent_supervisor.contracts.execution import ProviderSelection
from ipfs_accelerate_py.agent_supervisor.contracts.provider_capacity import (
    NON_AUTHORITATIVE_CAPACITY_SCHEMA,
    NonAuthoritativeProviderCapacityObservation,
)
from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    cid_for_dag_json,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_provider_auto import (
    AutoProviderDecision,
    BackendObservation,
    select_implementation_provider,
)
from ipfs_accelerate_py.llm_router import (
    ROUTER_OWNED_COMPATIBILITY_CANONICAL,
    ROUTER_OWNED_COMPATIBILITY_LEGACY_AUTO,
    ROUTER_OWNED_PROVIDER_DECISION_SCHEMA,
    LegacyAutoProviderCompatibilityAdapter,
    RouterOwnedProviderDecision,
    RouterOwnedProviderObservation,
    RouterOwnedProviderPolicyContext,
    RouterOwnedProviderReason,
    decide_router_owned_implementation_provider,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LLM_ROUTER = REPO_ROOT / "ipfs_accelerate_py" / "llm_router.py"
PROVIDER_AUTO = (
    REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "todo_daemon"
    / "implementation_provider_auto.py"
)
CAPABILITY_RESOLVER = (
    REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "control"
    / "capability_resolver.py"
)


def _cid(label: str) -> str:
    return cid_for_dag_json({"fixture": label})


def _obs(
    provider_id: str,
    *,
    ready: bool = True,
    authenticated: bool = True,
    binary_available: bool = True,
    hard_quota_exhausted: bool = False,
    capacity_latched: bool = False,
    request_headroom: int | None = 10,
) -> RouterOwnedProviderObservation:
    return RouterOwnedProviderObservation(
        provider_id=provider_id,
        ready=ready,
        authenticated=authenticated,
        binary_available=binary_available,
        hard_quota_exhausted=hard_quota_exhausted,
        capacity_latched=capacity_latched,
        request_headroom=request_headroom,
        source="test",
    )


def _backend(
    provider_id: str,
    *,
    ready: bool = True,
    authenticated: bool = True,
    binary_available: bool = True,
    hard_quota_exhausted: bool = False,
    capacity_latched: bool = False,
    request_headroom: int | None = 10,
) -> BackendObservation:
    return BackendObservation(
        provider_id=provider_id,
        ready=ready,
        authenticated=authenticated,
        binary_available=binary_available,
        hard_quota_exhausted=hard_quota_exhausted,
        capacity_latched=capacity_latched,
        request_headroom=request_headroom,
        source="test",
    )


def _provider(
    provider_id: str,
    *,
    capability: PreferredProviderCapability = PreferredProviderCapability.AVAILABLE,
    policy_allowed: bool = True,
    healthy: bool = True,
    authenticated: bool = True,
    request_headroom: int = 10,
    max_concurrency: int = 4,
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


def _evidence(
    *,
    grok: ProviderCapabilityEvidence | None = None,
    codex: ProviderCapabilityEvidence | None = None,
) -> CapabilityEvidence:
    preferred = grok or _provider(PREFERRED_PROVIDER)
    fallback = codex if codex is not None else _provider(FALLBACK_PROVIDER)
    return CapabilityEvidence(
        providers={
            preferred.provider_id: preferred,
            fallback.provider_id: fallback,
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
            allowlisted_argv=(("python", "-m", "pytest", "test/api", "-q"),),
            policy_cid=_cid("validation-policy"),
        ),
        topology=TopologyEvidence(
            distributed_capable=False,
            shard_count=1,
            owner_principal_ref="did:key:local-owner",
            state_root="/var/lib/supervisor/state",
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


def test_router_decision_prefers_ready_grok_and_is_content_addressed() -> None:
    decision = decide_router_owned_implementation_provider(
        (
            _obs("grok"),
            _obs("codex"),
            _obs("claude"),
        )
    )
    assert isinstance(decision, RouterOwnedProviderDecision)
    assert decision.schema == ROUTER_OWNED_PROVIDER_DECISION_SCHEMA
    assert decision.selected_provider == "grok"
    assert decision.authorized is True
    assert RouterOwnedProviderReason.PREFERRED_READY in decision.reason_codes
    assert decision.decision_cid.startswith("sha256:")
    again = decide_router_owned_implementation_provider(
        (
            _obs("grok"),
            _obs("codex"),
            _obs("claude"),
        )
    )
    assert again.decision_cid == decision.decision_cid


def test_router_decision_opens_secondary_only_after_preferred_hard_quota() -> None:
    denied = decide_router_owned_implementation_provider(
        (
            _obs(
                "grok",
                ready=False,
                capacity_latched=True,
                hard_quota_exhausted=False,
            ),
            _obs("codex"),
        )
    )
    assert denied.authorized is False
    assert denied.decision == "backoff"
    assert (
        RouterOwnedProviderReason.PREFERRED_TRANSIENT_CAPACITY
        in denied.reason_codes
    )

    allowed = decide_router_owned_implementation_provider(
        (
            _obs(
                "grok",
                ready=False,
                hard_quota_exhausted=True,
            ),
            _obs("codex"),
            _obs("claude", request_headroom=99),
        )
    )
    assert allowed.authorized is True
    assert allowed.selected_provider == "codex"
    assert RouterOwnedProviderReason.FALLBACK_AFTER_QUOTA in allowed.reason_codes


def test_identical_observations_yield_same_decision_cid_through_both_callers() -> None:
    observations = (
        _backend("grok", ready=True),
        _backend("codex", ready=True),
    )
    auto = select_implementation_provider(observations)
    assert auto.decision is AutoProviderDecision.GROK

    router_from_auto = decide_router_owned_implementation_provider(
        tuple(
            RouterOwnedProviderObservation(
                provider_id=item.provider_id,
                ready=item.ready,
                authenticated=item.authenticated,
                binary_available=item.binary_available,
                hard_quota_exhausted=item.hard_quota_exhausted,
                capacity_latched=item.capacity_latched,
                request_headroom=item.request_headroom,
                source=item.source,
                reason_codes=item.reason_codes,
            )
            for item in observations
        ),
        preferred_provider=PREFERRED_PROVIDER,
        fallback_provider=FALLBACK_PROVIDER,
        secondary_providers=(FALLBACK_PROVIDER,),
        compatibility_mode=ROUTER_OWNED_COMPATIBILITY_LEGACY_AUTO,
    )

    capability_router = decide_router_owned_implementation_provider(
        (
            RouterOwnedProviderObservation(
                provider_id="grok",
                ready=True,
                authenticated=True,
                binary_available=True,
                hard_quota_exhausted=False,
                capacity_latched=False,
                request_headroom=10,
                source="test",
            ),
            RouterOwnedProviderObservation(
                provider_id="codex",
                ready=True,
                authenticated=True,
                binary_available=True,
                hard_quota_exhausted=False,
                capacity_latched=False,
                request_headroom=10,
                source="test",
            ),
        ),
        preferred_provider=PREFERRED_PROVIDER,
        fallback_provider=FALLBACK_PROVIDER,
        secondary_providers=(FALLBACK_PROVIDER,),
        compatibility_mode=ROUTER_OWNED_COMPATIBILITY_LEGACY_AUTO,
    )
    assert router_from_auto.decision_cid == capability_router.decision_cid

    adapted = LegacyAutoProviderCompatibilityAdapter.to_selection_fields(
        router_from_auto
    )
    assert adapted["selected_provider"] == auto.selected_provider
    assert adapted["decision"] == auto.decision.value


def test_capability_resolver_consumes_router_decision_for_quota_fallback() -> None:
    evidence = _evidence(
        grok=_provider(
            PREFERRED_PROVIDER,
            capability=PreferredProviderCapability.QUOTA_EXHAUSTED,
            request_headroom=0,
        ),
        codex=_provider(FALLBACK_PROVIDER),
    )
    resolution = resolve_capabilities(evidence)
    assert resolution.selected_provider is ProviderSelection.CODEX
    assert resolution.fallback_receipt is not None
    assert any(
        str(code).startswith("sha256:")
        for code in resolution.decisions[0].reason_codes
    )


def test_neutral_capacity_dto_cannot_authorize_dispatch() -> None:
    observation = NonAuthoritativeProviderCapacityObservation(
        schema=NON_AUTHORITATIVE_CAPACITY_SCHEMA,
        observed_at_ms=1,
        available_worker_capacity=8,
        worker_limit=8,
        details={"provider_id": "grok", "healthy": True},
    )
    assert observation.available_worker_capacity == 8
    with pytest.raises(TypeError):
        # Capacity DTO is not a RouterOwnedProviderDecision and has no
        # authorization surface; misuse must not silently become policy.
        decide_router_owned_implementation_provider(observation)  # type: ignore[arg-type]


def test_callers_do_not_retain_independent_rank_or_allow_deny_tables() -> None:
    forbidden_names = {
        "_PROVIDER_PREFERENCE_RANK",
        "_soft_rank_key",
        "_best_secondary",
    }
    auto_tree = ast.parse(PROVIDER_AUTO.read_text(encoding="utf-8"))
    auto_defs = {
        node.name
        for node in ast.walk(auto_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    auto_assigns = {
        target.id
        for node in ast.walk(auto_tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert not (forbidden_names & (auto_defs | auto_assigns))

    auto_src = PROVIDER_AUTO.read_text(encoding="utf-8")
    assert "decide_router_owned_implementation_provider" in auto_src
    assert "LegacyAutoProviderCompatibilityAdapter" in auto_src

    capability_src = CAPABILITY_RESOLVER.read_text(encoding="utf-8")
    assert "decide_router_owned_implementation_provider" in capability_src
    assert "_PROVIDER_PREFERENCE_RANK" not in capability_src

    router_src = LLM_ROUTER.read_text(encoding="utf-8")
    assert "def decide_router_owned_implementation_provider" in router_src
    assert "class RouterOwnedProviderDecision" in router_src


def test_policy_context_is_part_of_decision_identity() -> None:
    observations = (
        _obs("grok", ready=False, hard_quota_exhausted=True),
        _obs("codex"),
    )
    a = decide_router_owned_implementation_provider(
        observations,
        policy_context=RouterOwnedProviderPolicyContext(
            secondary_providers=("codex",),
            compatibility_mode=ROUTER_OWNED_COMPATIBILITY_CANONICAL,
        ),
    )
    b = decide_router_owned_implementation_provider(
        observations,
        policy_context=RouterOwnedProviderPolicyContext(
            secondary_providers=("codex", "claude"),
            compatibility_mode=ROUTER_OWNED_COMPATIBILITY_CANONICAL,
        ),
    )
    assert a.selected_provider == "codex"
    assert b.selected_provider == "codex"
    assert a.decision_cid != b.decision_cid
