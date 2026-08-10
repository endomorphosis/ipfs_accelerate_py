"""Provider, resource, lane, validation, and topology resolution (ASE-009).

This module is the leaf capability resolver for prompt-only entrypoints.  It
compiles frozen health, budget, ready-width, validation-policy, and topology
evidence into typed, deterministic selection records without executing
providers, tests, or launch argv.

Design rules enforced here:

- selection is deterministic under identical frozen evidence;
- healthy, policy-allowed Grok is preferred over Codex;
- Codex fallback is authorized only by confirmed Grok quota exhaustion and
  always requires an independent reviewer (the implementer cannot self-attest);
- prompt text and untrusted provider/lane labels cannot choose a provider or
  raise lane width;
- optional degradation is explicit rather than silent.

Evidence is expected to be produced by capability/usage gateways,
:class:`~ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler.ResourceScheduler`
safe ready-width samples, and reviewed validation policy.  This module only
consumes that evidence.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Final, Iterable

from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
    cid_for_dag_json,
)

from ..contracts.execution import (
    DEFAULT_PARQUET_PARTITIONS,
    MAX_LANES,
    CoordinationShardBinding,
    DecisionEffect,
    EntrypointContractError,
    ProviderFallbackReason,
    ProviderRouteProvenance,
    ProviderSelection,
    ReplicationBinding,
    ReplicationMode,
    ResolutionDisposition,
    ResolutionSource,
    ResourceBudget,
    RevalidationRule,
    TargetCandidate,
    TargetInferenceDecision,
)

SCHEMA_PREFIX: Final = "ipfs_accelerate_py/agent-supervisor/entrypoints"
PROVIDER_FALLBACK_RECEIPT_SCHEMA: Final = (
    f"{SCHEMA_PREFIX}/provider-fallback-receipt@1"
)
RESOURCE_ENVELOPE_SCHEMA: Final = f"{SCHEMA_PREFIX}/resource-envelope@1"
VALIDATION_PROFILE_SCHEMA: Final = f"{SCHEMA_PREFIX}/validation-profile@1"
DEPLOYMENT_TOPOLOGY_SCHEMA: Final = f"{SCHEMA_PREFIX}/deployment-topology@1"
CAPABILITY_EVIDENCE_SCHEMA: Final = f"{SCHEMA_PREFIX}/capability-evidence@1"
CAPABILITY_RESOLUTION_SCHEMA: Final = (
    f"{SCHEMA_PREFIX}/capability-resolution@1"
)

PREFERRED_PROVIDER: Final = "grok"
FALLBACK_PROVIDER: Final = "codex"
ALLOWED_IMPLEMENTATION_PROVIDERS: Final[frozenset[str]] = frozenset(
    {PREFERRED_PROVIDER, FALLBACK_PROVIDER}
)
MAXIMUM_FALLBACK_DISPATCHES: Final = 1

CAPABILITY_FIELD_NAMES: Final[tuple[str, ...]] = (
    "provider",
    "resources",
    "lane_ceiling",
    "validation",
    "coordination",
    "replication",
)

_SHELL_INJECTION_MARKERS: Final[tuple[str, ...]] = (
    "|",
    "&&",
    "||",
    ";",
    "`",
    "$(",
    "${",
    "\n",
    "\r",
    ">",
    "<",
)
_FORBIDDEN_VALIDATION_FLAGS: Final[tuple[str, ...]] = (
    "--prompt",
    "--authorization",
    "--api-key",
    "--apikey",
    "--password",
    "--private-key",
    "--secret",
    "--token",
    "--ucan",
)


class CapabilityResolverError(EntrypointContractError):
    """Raised when capability evidence is malformed or non-authoritative."""


class PreferredProviderCapability(str, Enum):
    """Closed set of preferred-provider health/budget observations."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    QUOTA_EXHAUSTED = "quota_exhausted"
    CAPACITY_UNAVAILABLE = "capacity_unavailable"
    PRE_EFFECT_FAILURE = "pre_effect_failure"


class TopologyMode(str, Enum):
    """Resolved deployment topology for coordination and workers."""

    LOCAL = "local"
    DISTRIBUTED = "distributed"


class CapabilityDegradationCode(str, Enum):
    """Explicit optional-capability degradation markers."""

    NONE = "none"
    PREFERRED_PROVIDER_DEGRADED = "preferred_provider_degraded"
    FALLBACK_PROVIDER_ONLY = "fallback_provider_only"
    FALLBACK_NOT_AUTHORIZED = "fallback_not_authorized"
    PROVIDERS_UNAVAILABLE = "providers_unavailable"
    IPFS_PUBLICATION_UNAVAILABLE = "ipfs_publication_unavailable"
    DISTRIBUTED_TOPOLOGY_UNAVAILABLE = "distributed_topology_unavailable"
    VALIDATION_CANDIDATES_FILTERED = "validation_candidates_filtered"
    LANE_WIDTH_CONSTRAINED = "lane_width_constrained"


_CAPABILITY_TO_FALLBACK_REASON: Final[
    Mapping[PreferredProviderCapability, ProviderFallbackReason]
] = {
    PreferredProviderCapability.UNAVAILABLE: (
        ProviderFallbackReason.PREFERRED_UNAVAILABLE
    ),
    PreferredProviderCapability.QUOTA_EXHAUSTED: (
        ProviderFallbackReason.PREFERRED_QUOTA_EXHAUSTED
    ),
    PreferredProviderCapability.CAPACITY_UNAVAILABLE: (
        ProviderFallbackReason.PREFERRED_CAPACITY_UNAVAILABLE
    ),
    PreferredProviderCapability.PRE_EFFECT_FAILURE: (
        ProviderFallbackReason.PREFERRED_PRE_EFFECT_FAILURE
    ),
}


def _cid(label: str, payload: Mapping[str, Any] | None = None) -> str:
    body: dict[str, Any] = {"label": label}
    if payload is not None:
        body["payload"] = dict(payload)
    return cid_for_dag_json(body)


def _require_cid(value: str, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise CapabilityResolverError(f"{name} is required")
    # Accept either a real multiformats CID or a content-addressed token.
    if not re.fullmatch(r"[A-Za-z0-9:._+/-]{8,}", text):
        raise CapabilityResolverError(f"{name} is not a valid identity")
    return text


def _non_negative_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CapabilityResolverError(f"{name} must be an integer")
    if value < 0:
        raise CapabilityResolverError(f"{name} must be non-negative")
    if maximum is not None and value > maximum:
        raise CapabilityResolverError(f"{name} exceeds maximum {maximum}")
    return value


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    number = _non_negative_int(value, name, maximum=maximum)
    if number < 1:
        raise CapabilityResolverError(f"{name} must be at least 1")
    return number


def _token(value: Any, name: str) -> str:
    text = str(value or "").strip().lower()
    if not text or not re.fullmatch(r"[a-z0-9][a-z0-9._:-]*", text):
        raise CapabilityResolverError(f"{name} must be a closed token")
    return text


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise CapabilityResolverError(f"{name} must be a boolean")
    return value


def _sorted_unique(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(item) for item in items if str(item)}))


def _argv_is_safe(argv: Sequence[str]) -> bool:
    if not argv:
        return False
    if not all(isinstance(item, str) and item for item in argv):
        return False
    joined = " ".join(argv)
    lowered = joined.lower()
    if any(marker in joined for marker in _SHELL_INJECTION_MARKERS):
        return False
    if any(marker in lowered for marker in _FORBIDDEN_VALIDATION_FLAGS):
        return False
    # Reject prompt-body injection vectors in argv tokens.
    if "prompt" in lowered and any(
        token.lower().startswith("--prompt") for token in argv
    ):
        return False
    return True


@dataclass(frozen=True)
class ProviderCapabilityEvidence:
    """Frozen observation of one implementation provider's readiness."""

    provider_id: str
    capability: PreferredProviderCapability
    policy_allowed: bool
    healthy: bool
    authenticated: bool
    observed_capability_cid: str
    usage_evidence_cid: str
    budget_cid: str
    max_concurrency: int = 1
    request_headroom: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "provider_id", _token(self.provider_id, "provider_id")
        )
        if self.provider_id not in ALLOWED_IMPLEMENTATION_PROVIDERS:
            raise CapabilityResolverError(
                f"unsupported implementation provider {self.provider_id!r}"
            )
        capability = self.capability
        if not isinstance(capability, PreferredProviderCapability):
            try:
                capability = PreferredProviderCapability(
                    str(capability).strip().lower()
                )
            except ValueError as exc:
                raise CapabilityResolverError(
                    f"unknown preferred capability {self.capability!r}"
                ) from exc
            object.__setattr__(self, "capability", capability)
        object.__setattr__(
            self, "policy_allowed", _bool(self.policy_allowed, "policy_allowed")
        )
        object.__setattr__(self, "healthy", _bool(self.healthy, "healthy"))
        object.__setattr__(
            self, "authenticated", _bool(self.authenticated, "authenticated")
        )
        object.__setattr__(
            self,
            "observed_capability_cid",
            _require_cid(self.observed_capability_cid, "observed_capability_cid"),
        )
        object.__setattr__(
            self,
            "usage_evidence_cid",
            _require_cid(self.usage_evidence_cid, "usage_evidence_cid"),
        )
        object.__setattr__(
            self, "budget_cid", _require_cid(self.budget_cid, "budget_cid")
        )
        object.__setattr__(
            self,
            "max_concurrency",
            _non_negative_int(self.max_concurrency, "max_concurrency", maximum=MAX_LANES),
        )
        object.__setattr__(
            self,
            "request_headroom",
            _non_negative_int(
                self.request_headroom, "request_headroom", maximum=10**9
            ),
        )

    @property
    def ready(self) -> bool:
        return (
            self.policy_allowed
            and self.healthy
            and self.authenticated
            and self.capability is PreferredProviderCapability.AVAILABLE
            and self.request_headroom > 0
            and self.max_concurrency > 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "capability": self.capability.value,
            "policy_allowed": self.policy_allowed,
            "healthy": self.healthy,
            "authenticated": self.authenticated,
            "observed_capability_cid": self.observed_capability_cid,
            "usage_evidence_cid": self.usage_evidence_cid,
            "budget_cid": self.budget_cid,
            "max_concurrency": self.max_concurrency,
            "request_headroom": self.request_headroom,
            "ready": self.ready,
        }


@dataclass(frozen=True)
class ResourceSampleEvidence:
    """Frozen host/conflict ready-width sample used for lane ceilings.

    ``ready_width`` is the conflict-safe ready concurrency computed by the
    resource/conflict scheduler.  ``lane_labels`` are accepted only so callers
    can prove they are ignored.
    """

    ready_width: int
    host_worker_limit: int
    host_available_workers: int
    max_processes: int
    max_validation_workers: int
    cpu_millis: int
    memory_bytes: int
    provider_request_limit: int
    deadline_ms: int
    lane_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "ready_width",
            "host_worker_limit",
            "host_available_workers",
            "max_processes",
            "max_validation_workers",
            "cpu_millis",
            "memory_bytes",
            "provider_request_limit",
            "deadline_ms",
        ):
            maximum = MAX_LANES if name in {
                "ready_width",
                "host_worker_limit",
                "host_available_workers",
                "max_processes",
                "max_validation_workers",
            } else None
            if name in {"cpu_millis", "memory_bytes", "provider_request_limit", "deadline_ms"}:
                maximum = 10**12 if name != "deadline_ms" else 7 * 24 * 60 * 60 * 1000
            object.__setattr__(
                self,
                name,
                _non_negative_int(getattr(self, name), name, maximum=maximum),
            )
        labels = tuple(str(item) for item in self.lane_labels)
        object.__setattr__(self, "lane_labels", labels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready_width": self.ready_width,
            "host_worker_limit": self.host_worker_limit,
            "host_available_workers": self.host_available_workers,
            "max_processes": self.max_processes,
            "max_validation_workers": self.max_validation_workers,
            "cpu_millis": self.cpu_millis,
            "memory_bytes": self.memory_bytes,
            "provider_request_limit": self.provider_request_limit,
            "deadline_ms": self.deadline_ms,
            "lane_labels": list(self.lane_labels),
        }


@dataclass(frozen=True)
class ValidationPolicyEvidence:
    """Reviewed structured validation candidates (never prompt shell)."""

    allowlisted_argv: tuple[tuple[str, ...], ...]
    policy_cid: str

    def __post_init__(self) -> None:
        if isinstance(self.allowlisted_argv, (str, bytes)) or not isinstance(
            self.allowlisted_argv, Sequence
        ):
            raise CapabilityResolverError("allowlisted_argv must be a sequence")
        normalized: list[tuple[str, ...]] = []
        for index, argv in enumerate(self.allowlisted_argv):
            if isinstance(argv, str) or not isinstance(argv, Sequence):
                raise CapabilityResolverError(
                    f"allowlisted_argv[{index}] must be an argv sequence"
                )
            tokens = tuple(str(item) for item in argv)
            if not _argv_is_safe(tokens):
                raise CapabilityResolverError(
                    f"allowlisted_argv[{index}] is not a safe structured command"
                )
            normalized.append(tokens)
        object.__setattr__(self, "allowlisted_argv", tuple(normalized))
        object.__setattr__(
            self, "policy_cid", _require_cid(self.policy_cid, "policy_cid")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowlisted_argv": [list(item) for item in self.allowlisted_argv],
            "policy_cid": self.policy_cid,
        }


@dataclass(frozen=True)
class TopologyEvidence:
    """Frozen deployment topology and coordination ownership evidence."""

    distributed_capable: bool
    shard_count: int
    owner_principal_ref: str
    state_root: str
    database_relative_path: str
    coordinator_cid: str
    lease_namespace: str
    fencing_generation: int
    ipfs_publish_capable: bool
    parquet_capable: bool
    preferred_mode: TopologyMode = TopologyMode.LOCAL
    ipfs_backend_handle: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "distributed_capable",
            _bool(self.distributed_capable, "distributed_capable"),
        )
        object.__setattr__(
            self,
            "shard_count",
            _positive_int(self.shard_count, "shard_count", maximum=1024),
        )
        object.__setattr__(
            self,
            "owner_principal_ref",
            str(self.owner_principal_ref or "").strip(),
        )
        if not self.owner_principal_ref:
            raise CapabilityResolverError("owner_principal_ref is required")
        state_root = str(self.state_root or "").strip()
        if not state_root.startswith("/"):
            raise CapabilityResolverError("state_root must be absolute")
        object.__setattr__(self, "state_root", state_root)
        relative = str(self.database_relative_path or "").strip().lstrip("/")
        if not relative or ".." in relative.split("/"):
            raise CapabilityResolverError(
                "database_relative_path must be a contained relative path"
            )
        object.__setattr__(self, "database_relative_path", relative)
        object.__setattr__(
            self,
            "coordinator_cid",
            _require_cid(self.coordinator_cid, "coordinator_cid"),
        )
        object.__setattr__(
            self,
            "lease_namespace",
            _token(self.lease_namespace, "lease_namespace"),
        )
        object.__setattr__(
            self,
            "fencing_generation",
            _non_negative_int(self.fencing_generation, "fencing_generation"),
        )
        object.__setattr__(
            self,
            "ipfs_publish_capable",
            _bool(self.ipfs_publish_capable, "ipfs_publish_capable"),
        )
        object.__setattr__(
            self,
            "parquet_capable",
            _bool(self.parquet_capable, "parquet_capable"),
        )
        mode = self.preferred_mode
        if not isinstance(mode, TopologyMode):
            try:
                mode = TopologyMode(str(mode).strip().lower())
            except ValueError as exc:
                raise CapabilityResolverError(
                    f"unknown topology mode {self.preferred_mode!r}"
                ) from exc
            object.__setattr__(self, "preferred_mode", mode)
        handle = str(self.ipfs_backend_handle or "").strip()
        if self.ipfs_publish_capable and not handle:
            raise CapabilityResolverError(
                "ipfs_publish_capable requires ipfs_backend_handle"
            )
        object.__setattr__(self, "ipfs_backend_handle", handle)

    def to_dict(self) -> dict[str, Any]:
        return {
            "distributed_capable": self.distributed_capable,
            "shard_count": self.shard_count,
            "owner_principal_ref": self.owner_principal_ref,
            "state_root": self.state_root,
            "database_relative_path": self.database_relative_path,
            "coordinator_cid": self.coordinator_cid,
            "lease_namespace": self.lease_namespace,
            "fencing_generation": self.fencing_generation,
            "ipfs_publish_capable": self.ipfs_publish_capable,
            "parquet_capable": self.parquet_capable,
            "preferred_mode": self.preferred_mode.value,
            "ipfs_backend_handle": self.ipfs_backend_handle,
        }


@dataclass(frozen=True)
class CapabilityEvidence:
    """Complete frozen evidence bundle for deterministic capability resolution."""

    providers: Mapping[str, ProviderCapabilityEvidence]
    resources: ResourceSampleEvidence
    validation: ValidationPolicyEvidence
    topology: TopologyEvidence
    task_revision_cid: str
    attempt_cid: str
    worktree_cid: str
    authenticated_profile_override: str = ""
    authenticated_profile_override_cid: str = ""
    prompt_text: str = ""
    provider_hint: str = ""
    requested_lane_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.providers, Mapping) or not self.providers:
            raise CapabilityResolverError("providers evidence is required")
        normalized: dict[str, ProviderCapabilityEvidence] = {}
        for key, value in self.providers.items():
            provider_id = _token(key, "providers key")
            if not isinstance(value, ProviderCapabilityEvidence):
                raise CapabilityResolverError(
                    f"providers[{provider_id!r}] must be ProviderCapabilityEvidence"
                )
            if value.provider_id != provider_id:
                raise CapabilityResolverError(
                    "providers map keys must match provider_id"
                )
            normalized[provider_id] = value
        if PREFERRED_PROVIDER not in normalized:
            raise CapabilityResolverError("Grok provider evidence is required")
        object.__setattr__(self, "providers", dict(normalized))
        if not isinstance(self.resources, ResourceSampleEvidence):
            raise CapabilityResolverError(
                "resources must be ResourceSampleEvidence"
            )
        if not isinstance(self.validation, ValidationPolicyEvidence):
            raise CapabilityResolverError(
                "validation must be ValidationPolicyEvidence"
            )
        if not isinstance(self.topology, TopologyEvidence):
            raise CapabilityResolverError(
                "topology must be TopologyEvidence"
            )
        object.__setattr__(
            self,
            "task_revision_cid",
            _require_cid(self.task_revision_cid, "task_revision_cid"),
        )
        object.__setattr__(
            self, "attempt_cid", _require_cid(self.attempt_cid, "attempt_cid")
        )
        object.__setattr__(
            self, "worktree_cid", _require_cid(self.worktree_cid, "worktree_cid")
        )
        override = str(self.authenticated_profile_override or "").strip().lower()
        override_cid = str(self.authenticated_profile_override_cid or "").strip()
        if override:
            if override not in ALLOWED_IMPLEMENTATION_PROVIDERS:
                raise CapabilityResolverError(
                    "authenticated profile override must be an allowed provider"
                )
            if not override_cid:
                raise CapabilityResolverError(
                    "authenticated profile override requires a signed profile CID"
                )
            object.__setattr__(
                self,
                "authenticated_profile_override_cid",
                _require_cid(override_cid, "authenticated_profile_override_cid"),
            )
        elif override_cid:
            raise CapabilityResolverError(
                "authenticated_profile_override_cid requires an override provider"
            )
        object.__setattr__(self, "authenticated_profile_override", override)
        object.__setattr__(self, "prompt_text", str(self.prompt_text or ""))
        object.__setattr__(
            self, "provider_hint", str(self.provider_hint or "").strip().lower()
        )
        object.__setattr__(
            self,
            "requested_lane_labels",
            tuple(str(item) for item in self.requested_lane_labels),
        )

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CAPABILITY_EVIDENCE_SCHEMA,
            "providers": {
                key: value.to_dict() for key, value in sorted(self.providers.items())
            },
            "resources": self.resources.to_dict(),
            "validation": self.validation.to_dict(),
            "topology": self.topology.to_dict(),
            "task_revision_cid": self.task_revision_cid,
            "attempt_cid": self.attempt_cid,
            "worktree_cid": self.worktree_cid,
            "authenticated_profile_override": self.authenticated_profile_override,
            "authenticated_profile_override_cid": (
                self.authenticated_profile_override_cid
            ),
            # Prompt text and untrusted provider_hint are deliberately omitted
            # from the evidence identity so they cannot influence selection.
            "requested_lane_labels": list(self.requested_lane_labels),
        }


@dataclass(frozen=True)
class ProviderFallbackReceipt:
    """Typed pre-effect Codex fallback receipt committed before dispatch.

    The receipt is content-addressed and records that the implementer attempt
    cannot self-satisfy an independent review obligation.
    """

    SCHEMA: ClassVar[str] = PROVIDER_FALLBACK_RECEIPT_SCHEMA

    preferred_provider: str
    fallback_provider: str
    reason_code: ProviderFallbackReason
    observed_capability_cid: str
    task_revision_cid: str
    budget_cid: str
    attempt_id: str
    usage_evidence_cid: str
    worktree_cid: str
    implementer_process_identity: str
    review_authorization: str
    maximum_fallback_dispatches: int = MAXIMUM_FALLBACK_DISPATCHES
    independent_review_required: bool = True
    same_attempt_may_satisfy_review: bool = False
    committed_before_dispatch: bool = True

    def __post_init__(self) -> None:
        preferred = _token(self.preferred_provider, "preferred_provider")
        fallback = _token(self.fallback_provider, "fallback_provider")
        if preferred != PREFERRED_PROVIDER or fallback != FALLBACK_PROVIDER:
            raise CapabilityResolverError(
                "fallback receipt must record the Grok then Codex route"
            )
        object.__setattr__(self, "preferred_provider", preferred)
        object.__setattr__(self, "fallback_provider", fallback)
        reason = self.reason_code
        if not isinstance(reason, ProviderFallbackReason):
            try:
                reason = ProviderFallbackReason(str(reason))
            except ValueError as exc:
                raise CapabilityResolverError(
                    f"unknown fallback reason {self.reason_code!r}"
                ) from exc
            object.__setattr__(self, "reason_code", reason)
        if reason is ProviderFallbackReason.NONE:
            raise CapabilityResolverError(
                "fallback receipt requires a typed non-none reason"
            )
        if reason is not ProviderFallbackReason.PREFERRED_QUOTA_EXHAUSTED:
            raise CapabilityResolverError(
                "Codex fallback receipt requires confirmed Grok quota exhaustion"
            )
        for name in (
            "observed_capability_cid",
            "task_revision_cid",
            "budget_cid",
            "attempt_id",
            "usage_evidence_cid",
            "worktree_cid",
            "implementer_process_identity",
            "review_authorization",
        ):
            object.__setattr__(
                self, name, _require_cid(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "maximum_fallback_dispatches",
            _positive_int(
                self.maximum_fallback_dispatches,
                "maximum_fallback_dispatches",
                maximum=MAXIMUM_FALLBACK_DISPATCHES,
            ),
        )
        for name in (
            "independent_review_required",
            "same_attempt_may_satisfy_review",
            "committed_before_dispatch",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        if not self.independent_review_required:
            raise CapabilityResolverError(
                "Codex fallback always requires an independent review"
            )
        if self.same_attempt_may_satisfy_review:
            raise CapabilityResolverError(
                "a fallback implementer cannot self-satisfy independent review"
            )
        if not self.committed_before_dispatch:
            raise CapabilityResolverError(
                "fallback receipt must commit before fallback dispatch"
            )
        if self.implementer_process_identity == self.review_authorization:
            raise CapabilityResolverError(
                "implementer process identity and review authorization must differ"
            )
        if self.implementer_process_identity == self.attempt_id:
            # Same-attempt self-review is forbidden; process identity must be
            # distinct from the review authorization, and the attempt itself
            # is never a valid review authorization.
            pass
        if self.review_authorization == self.attempt_id:
            raise CapabilityResolverError(
                "review authorization cannot equal the implementation attempt"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "preferred_provider": self.preferred_provider,
            "fallback_provider": self.fallback_provider,
            "reason_code": self.reason_code.value,
            "observed_capability_cid": self.observed_capability_cid,
            "task_revision_cid": self.task_revision_cid,
            "budget_cid": self.budget_cid,
            "attempt_id": self.attempt_id,
            "usage_evidence_cid": self.usage_evidence_cid,
            "worktree_cid": self.worktree_cid,
            "implementer_process_identity": self.implementer_process_identity,
            "review_authorization": self.review_authorization,
            "maximum_fallback_dispatches": self.maximum_fallback_dispatches,
            "independent_review_required": self.independent_review_required,
            "same_attempt_may_satisfy_review": (
                self.same_attempt_may_satisfy_review
            ),
            "committed_before_dispatch": self.committed_before_dispatch,
        }

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self._payload())

    def can_self_satisfy_independent_review(self) -> bool:
        """Codex implementation attempts never self-attest review."""

        return False

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["content_id"] = self.content_id
        return payload


@dataclass(frozen=True)
class ResourceEnvelope:
    """Resolved host/provider resource ceilings and lane width."""

    SCHEMA: ClassVar[str] = RESOURCE_ENVELOPE_SCHEMA

    resource_budget: ResourceBudget
    ready_width: int
    lane_ceiling: int
    evidence_cid: str
    ignored_lane_labels: tuple[str, ...] = ()
    degradations: tuple[str, ...] = ()
    source: ResolutionSource = ResolutionSource.DISCOVERY

    def __post_init__(self) -> None:
        if not isinstance(self.resource_budget, ResourceBudget):
            raise CapabilityResolverError(
                "resource_budget must be a ResourceBudget"
            )
        object.__setattr__(
            self,
            "ready_width",
            _non_negative_int(self.ready_width, "ready_width", maximum=MAX_LANES),
        )
        object.__setattr__(
            self,
            "lane_ceiling",
            _positive_int(self.lane_ceiling, "lane_ceiling", maximum=MAX_LANES),
        )
        if self.lane_ceiling > self.resource_budget.max_lanes:
            raise CapabilityResolverError(
                "lane_ceiling cannot exceed resource budget max_lanes"
            )
        if self.lane_ceiling > self.resource_budget.max_processes:
            raise CapabilityResolverError(
                "lane_ceiling cannot exceed max_processes"
            )
        object.__setattr__(
            self, "evidence_cid", _require_cid(self.evidence_cid, "evidence_cid")
        )
        object.__setattr__(
            self,
            "ignored_lane_labels",
            tuple(str(item) for item in self.ignored_lane_labels),
        )
        object.__setattr__(
            self, "degradations", _sorted_unique(self.degradations)
        )
        if not isinstance(self.source, ResolutionSource):
            try:
                object.__setattr__(
                    self,
                    "source",
                    ResolutionSource(str(self.source)),
                )
            except ValueError as exc:
                raise CapabilityResolverError(
                    f"unknown resource source {self.source!r}"
                ) from exc

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "resource_budget": self.resource_budget.to_dict(),
            "ready_width": self.ready_width,
            "lane_ceiling": self.lane_ceiling,
            "evidence_cid": self.evidence_cid,
            "ignored_lane_labels": list(self.ignored_lane_labels),
            "degradations": list(self.degradations),
            "source": self.source.value,
        }

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["content_id"] = self.content_id
        return payload


@dataclass(frozen=True)
class ValidationProfile:
    """Structured allowlisted validation argv policy for a resolved target."""

    SCHEMA: ClassVar[str] = VALIDATION_PROFILE_SCHEMA

    profile_cid: str
    policy_cid: str
    allowlisted_argv: tuple[tuple[str, ...], ...]
    rejects_prompt_shell: bool = True
    rejects_credential_injection: bool = True
    degradations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "profile_cid", _require_cid(self.profile_cid, "profile_cid")
        )
        object.__setattr__(
            self, "policy_cid", _require_cid(self.policy_cid, "policy_cid")
        )
        if isinstance(self.allowlisted_argv, (str, bytes)) or not isinstance(
            self.allowlisted_argv, Sequence
        ):
            raise CapabilityResolverError("allowlisted_argv must be a sequence")
        normalized: list[tuple[str, ...]] = []
        for argv in self.allowlisted_argv:
            tokens = tuple(str(item) for item in argv)
            if not _argv_is_safe(tokens):
                raise CapabilityResolverError(
                    "validation profile contains unsafe argv"
                )
            normalized.append(tokens)
        object.__setattr__(self, "allowlisted_argv", tuple(normalized))
        object.__setattr__(
            self,
            "rejects_prompt_shell",
            _bool(self.rejects_prompt_shell, "rejects_prompt_shell"),
        )
        object.__setattr__(
            self,
            "rejects_credential_injection",
            _bool(
                self.rejects_credential_injection,
                "rejects_credential_injection",
            ),
        )
        if not self.rejects_prompt_shell or not self.rejects_credential_injection:
            raise CapabilityResolverError(
                "validation profile must reject prompt shell and credentials"
            )
        object.__setattr__(
            self, "degradations", _sorted_unique(self.degradations)
        )

    @property
    def content_id(self) -> str:
        return self.profile_cid

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "profile_cid": self.profile_cid,
            "policy_cid": self.policy_cid,
            "allowlisted_argv": [list(item) for item in self.allowlisted_argv],
            "rejects_prompt_shell": self.rejects_prompt_shell,
            "rejects_credential_injection": self.rejects_credential_injection,
            "degradations": list(self.degradations),
        }


@dataclass(frozen=True)
class DeploymentTopology:
    """Local or distributed coordination/replication topology."""

    SCHEMA: ClassVar[str] = DEPLOYMENT_TOPOLOGY_SCHEMA

    mode: TopologyMode
    coordination_shard: CoordinationShardBinding
    replication: ReplicationBinding
    degradations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        mode = self.mode
        if not isinstance(mode, TopologyMode):
            try:
                mode = TopologyMode(str(mode).strip().lower())
            except ValueError as exc:
                raise CapabilityResolverError(
                    f"unknown topology mode {self.mode!r}"
                ) from exc
            object.__setattr__(self, "mode", mode)
        if not isinstance(self.coordination_shard, CoordinationShardBinding):
            raise CapabilityResolverError(
                "coordination_shard must be CoordinationShardBinding"
            )
        if not isinstance(self.replication, ReplicationBinding):
            raise CapabilityResolverError(
                "replication must be ReplicationBinding"
            )
        if self.coordination_shard.remote_access != "owner_rpc":
            raise CapabilityResolverError(
                "coordination topology requires owner_rpc remote access"
            )
        object.__setattr__(
            self, "degradations", _sorted_unique(self.degradations)
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "mode": self.mode.value,
            "coordination_shard": self.coordination_shard.to_dict(),
            "replication": self.replication.to_dict(),
            "degradations": list(self.degradations),
        }

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["content_id"] = self.content_id
        return payload


@dataclass(frozen=True)
class CapabilityResolution:
    """Complete capability resolution for profile composition."""

    SCHEMA: ClassVar[str] = CAPABILITY_RESOLUTION_SCHEMA

    provider_route: ProviderRouteProvenance
    fallback_receipt: ProviderFallbackReceipt | None
    resources: ResourceEnvelope
    validation: ValidationProfile
    topology: DeploymentTopology
    decisions: tuple[TargetInferenceDecision, ...]
    degradations: tuple[str, ...]
    evidence_cid: str
    selected_provider: ProviderSelection
    prompt_provider_ignored: bool
    lane_labels_ignored: bool

    def __post_init__(self) -> None:
        if not isinstance(self.provider_route, ProviderRouteProvenance):
            raise CapabilityResolverError(
                "provider_route must be ProviderRouteProvenance"
            )
        if self.fallback_receipt is not None and not isinstance(
            self.fallback_receipt, ProviderFallbackReceipt
        ):
            raise CapabilityResolverError(
                "fallback_receipt must be ProviderFallbackReceipt or None"
            )
        if not isinstance(self.resources, ResourceEnvelope):
            raise CapabilityResolverError(
                "resources must be ResourceEnvelope"
            )
        if not isinstance(self.validation, ValidationProfile):
            raise CapabilityResolverError(
                "validation must be ValidationProfile"
            )
        if not isinstance(self.topology, DeploymentTopology):
            raise CapabilityResolverError(
                "topology must be DeploymentTopology"
            )
        if not isinstance(self.decisions, Sequence):
            raise CapabilityResolverError("decisions must be a sequence")
        decisions = tuple(self.decisions)
        names = {item.field_name for item in decisions}
        missing = [name for name in CAPABILITY_FIELD_NAMES if name not in names]
        if missing:
            raise CapabilityResolverError(
                f"capability resolution missing decisions: {missing}"
            )
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(
            self, "degradations", _sorted_unique(self.degradations)
        )
        object.__setattr__(
            self, "evidence_cid", _require_cid(self.evidence_cid, "evidence_cid")
        )
        selected = self.selected_provider
        if not isinstance(selected, ProviderSelection):
            try:
                selected = ProviderSelection(str(selected))
            except ValueError as exc:
                raise CapabilityResolverError(
                    f"unknown selected provider {self.selected_provider!r}"
                ) from exc
            object.__setattr__(self, "selected_provider", selected)
        if selected is not self.provider_route.selected_provider:
            raise CapabilityResolverError(
                "selected_provider must match provider_route.selected_provider"
            )
        object.__setattr__(
            self,
            "prompt_provider_ignored",
            _bool(self.prompt_provider_ignored, "prompt_provider_ignored"),
        )
        object.__setattr__(
            self,
            "lane_labels_ignored",
            _bool(self.lane_labels_ignored, "lane_labels_ignored"),
        )
        if (
            selected is ProviderSelection.CODEX
            and self.provider_route.fallback_reason
            is not ProviderFallbackReason.NONE
        ):
            if self.fallback_receipt is None:
                raise CapabilityResolverError(
                    "Codex fallback requires a committed fallback receipt"
                )
            if not self.provider_route.independent_review_required:
                raise CapabilityResolverError(
                    "Codex fallback requires independent_review_required"
                )
            if self.fallback_receipt.can_self_satisfy_independent_review():
                raise CapabilityResolverError(
                    "fallback receipt cannot self-satisfy independent review"
                )
        elif self.fallback_receipt is not None:
            raise CapabilityResolverError(
                "only an authorized Codex quota fallback may carry a fallback receipt"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "provider_route": self.provider_route.to_dict(),
            "fallback_receipt": (
                None
                if self.fallback_receipt is None
                else self.fallback_receipt.to_dict()
            ),
            "resources": self.resources.to_dict(),
            "validation": self.validation.to_dict(),
            "topology": self.topology.to_dict(),
            "decisions": [item.to_dict() for item in self.decisions],
            "degradations": list(self.degradations),
            "evidence_cid": self.evidence_cid,
            "selected_provider": self.selected_provider.value,
            "prompt_provider_ignored": self.prompt_provider_ignored,
            "lane_labels_ignored": self.lane_labels_ignored,
        }

    @property
    def content_id(self) -> str:
        return cid_for_dag_json(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["content_id"] = self.content_id
        return payload


def map_preferred_capability_to_fallback_reason(
    capability: PreferredProviderCapability,
) -> ProviderFallbackReason:
    """Map a non-available preferred-provider state to a typed fallback reason."""

    if not isinstance(capability, PreferredProviderCapability):
        capability = PreferredProviderCapability(str(capability).strip().lower())
    if capability is PreferredProviderCapability.AVAILABLE:
        raise CapabilityResolverError(
            "available preferred capability has no fallback reason"
        )
    return _CAPABILITY_TO_FALLBACK_REASON[capability]


def compute_lane_ceiling(
    resources: ResourceSampleEvidence,
    *,
    provider_concurrency: int,
    ignored_labels: Sequence[str] = (),
) -> tuple[int, tuple[str, ...]]:
    """Compute lane ceiling from safe ready width and resources, not labels.

    Labels are accepted only to prove they do not influence the result.
    """

    _ = tuple(ignored_labels)  # intentional no-op: labels are non-authoritative
    candidates = [
        value
        for value in (
            resources.ready_width,
            resources.host_available_workers,
            resources.host_worker_limit,
            resources.max_processes,
            provider_concurrency if provider_concurrency > 0 else MAX_LANES,
            MAX_LANES,
        )
        if value > 0
    ]
    if not candidates:
        return 1, (CapabilityDegradationCode.LANE_WIDTH_CONSTRAINED.value,)
    ceiling = min(candidates)
    ceiling = max(1, min(ceiling, MAX_LANES, resources.max_processes or ceiling))
    degradations: list[str] = []
    if resources.ready_width > 0 and ceiling < resources.ready_width:
        degradations.append(CapabilityDegradationCode.LANE_WIDTH_CONSTRAINED.value)
    if resources.host_available_workers == 0 or resources.ready_width == 0:
        degradations.append(CapabilityDegradationCode.LANE_WIDTH_CONSTRAINED.value)
    return ceiling, tuple(sorted(set(degradations)))


def _candidate(
    *,
    field_name: str,
    value: str,
    source: ResolutionSource,
    precedence: int,
    evidence_cid: str,
    confidence_ppm: int = 1_000_000,
    rejection_reason: str = "",
) -> TargetCandidate:
    return TargetCandidate(
        field_name=field_name,
        value=value,
        source=source,
        source_precedence=precedence,
        evidence_cid=evidence_cid,
        confidence_ppm=confidence_ppm,
        rejection_reason=rejection_reason,
    )


def _decision(
    *,
    field_name: str,
    disposition: ResolutionDisposition,
    selected_value: str,
    selected_source: ResolutionSource,
    source_precedence: int,
    evidence_cid: str,
    candidates: Sequence[TargetCandidate],
    reason_codes: Sequence[str],
    effect: DecisionEffect,
    override_accepted: bool = False,
) -> TargetInferenceDecision:
    return TargetInferenceDecision(
        field_name=field_name,
        disposition=disposition,
        selected_value=selected_value,
        selected_source=selected_source,
        source_precedence=source_precedence,
        evidence_cid=evidence_cid,
        candidates=tuple(candidates),
        reason_codes=tuple(reason_codes),
        effect=effect,
        override_accepted=override_accepted,
        fresh_until_ms=0,
        revalidation_rule=RevalidationRule.BEFORE_MUTATION,
    )


@dataclass(frozen=True)
class CapabilityResolver:
    """Deterministic provider/resource/lane/validation/topology resolver."""

    def resolve(self, evidence: CapabilityEvidence) -> CapabilityResolution:
        if not isinstance(evidence, CapabilityEvidence):
            raise CapabilityResolverError(
                "resolve requires frozen CapabilityEvidence"
            )
        evidence_cid = evidence.content_id
        degradations: list[str] = []

        provider_route, fallback_receipt, provider_decision, provider_degradations = (
            self._resolve_provider(evidence, evidence_cid=evidence_cid)
        )
        degradations.extend(provider_degradations)

        resources, resource_decision, lane_decision, resource_degradations = (
            self._resolve_resources(
                evidence,
                selected=provider_route.selected_provider,
                evidence_cid=evidence_cid,
            )
        )
        degradations.extend(resource_degradations)

        validation, validation_decision, validation_degradations = (
            self._resolve_validation(evidence, evidence_cid=evidence_cid)
        )
        degradations.extend(validation_degradations)

        topology, coordination_decision, replication_decision, topology_degradations = (
            self._resolve_topology(evidence, evidence_cid=evidence_cid)
        )
        degradations.extend(topology_degradations)

        selected = provider_route.selected_provider
        return CapabilityResolution(
            provider_route=provider_route,
            fallback_receipt=fallback_receipt,
            resources=resources,
            validation=validation,
            topology=topology,
            decisions=(
                provider_decision,
                resource_decision,
                lane_decision,
                validation_decision,
                coordination_decision,
                replication_decision,
            ),
            degradations=tuple(sorted(set(degradations))),
            evidence_cid=evidence_cid,
            selected_provider=selected,
            prompt_provider_ignored=True,
            lane_labels_ignored=True,
        )

    def _resolve_provider(
        self,
        evidence: CapabilityEvidence,
        *,
        evidence_cid: str,
    ) -> tuple[
        ProviderRouteProvenance,
        ProviderFallbackReceipt | None,
        TargetInferenceDecision,
        tuple[str, ...],
    ]:
        grok = evidence.providers[PREFERRED_PROVIDER]
        codex = evidence.providers.get(FALLBACK_PROVIDER)
        degradations: list[str] = []

        # Prompt text and untrusted provider_hint are non-authoritative.
        _ = evidence.prompt_text
        _ = evidence.provider_hint

        override = evidence.authenticated_profile_override
        override_cid = evidence.authenticated_profile_override_cid
        if override:
            selected = ProviderSelection(override)
            selected_evidence = evidence.providers[override]
            if not selected_evidence.policy_allowed:
                raise CapabilityResolverError(
                    "authenticated profile override is not policy-allowed"
                )
            route = ProviderRouteProvenance(
                preferred_provider=PREFERRED_PROVIDER,
                fallback_provider=FALLBACK_PROVIDER,
                selected_provider=selected,
                fallback_reason=ProviderFallbackReason.NONE,
                fallback_receipt_cid="",
                observed_capability_cid=selected_evidence.observed_capability_cid,
                usage_evidence_cid=selected_evidence.usage_evidence_cid,
                budget_cid=selected_evidence.budget_cid,
                task_revision_cid=evidence.task_revision_cid,
                attempt_cid=evidence.attempt_cid,
                worktree_cid=evidence.worktree_cid,
                authenticated_profile_override_cid=override_cid,
                maximum_fallback_dispatches=MAXIMUM_FALLBACK_DISPATCHES,
                independent_review_required=True,
            )
            decision = _decision(
                field_name="provider",
                disposition=ResolutionDisposition.UNIQUE,
                selected_value=selected.value,
                selected_source=ResolutionSource.SIGNED_PROFILE,
                source_precedence=40,
                evidence_cid=evidence_cid,
                candidates=(
                    _candidate(
                        field_name="provider",
                        value=selected.value,
                        source=ResolutionSource.SIGNED_PROFILE,
                        precedence=40,
                        evidence_cid=override_cid,
                    ),
                    _candidate(
                        field_name="provider",
                        value=PREFERRED_PROVIDER,
                        source=ResolutionSource.BUILTIN_DEFAULT,
                        precedence=90,
                        evidence_cid=grok.observed_capability_cid,
                        rejection_reason="signed_profile_override_selected",
                    ),
                ),
                reason_codes=("authenticated_profile_override",),
                effect=DecisionEffect.CONFIGURATION,
                override_accepted=False,
            )
            return route, None, decision, tuple(degradations)

        # ASE3-028: sole provider-policy decision lives in llm_router.
        # This method only normalizes non-authoritative observations and
        # adapts the returned RouterOwnedProviderDecision into route receipts.
        from ipfs_accelerate_py.llm_router import (
            ROUTER_OWNED_COMPATIBILITY_CANONICAL,
            RouterOwnedProviderObservation,
            RouterOwnedProviderReason,
            decide_router_owned_implementation_provider,
        )

        def _observation_for(
            provider: ProviderCapabilityEvidence,
        ) -> RouterOwnedProviderObservation:
            hard_quota = (
                provider.capability is PreferredProviderCapability.QUOTA_EXHAUSTED
            )
            capacity_latched = (
                provider.capability
                is PreferredProviderCapability.CAPACITY_UNAVAILABLE
            )
            # Expand readiness without re-authorizing: capability AVAILABLE is
            # an observation of preferred health, not a dispatch grant.
            ready = (
                provider.policy_allowed
                and provider.healthy
                and provider.authenticated
                and provider.capability is PreferredProviderCapability.AVAILABLE
                and provider.request_headroom > 0
                and provider.max_concurrency > 0
            )
            return RouterOwnedProviderObservation(
                provider_id=provider.provider_id,
                ready=ready,
                authenticated=provider.authenticated,
                binary_available=True,
                hard_quota_exhausted=hard_quota,
                capacity_latched=capacity_latched,
                request_headroom=provider.request_headroom,
                source="capability_evidence",
                reason_codes=(provider.capability.value,),
            )

        router_observations = [_observation_for(grok)]
        if codex is not None:
            router_observations.append(_observation_for(codex))

        router_decision = decide_router_owned_implementation_provider(
            router_observations,
            preferred_provider=PREFERRED_PROVIDER,
            fallback_provider=FALLBACK_PROVIDER,
            secondary_providers=(FALLBACK_PROVIDER,),
            global_capacity_latched=False,
            allow_secondary_without_preferred_quota=False,
            compatibility_mode=ROUTER_OWNED_COMPATIBILITY_CANONICAL,
        )

        selected_provider = str(router_decision.selected_provider or "")
        reason_codes = set(router_decision.reason_codes)

        # Map router classification of preferred state onto the protected
        # fallback-reason vocabulary without re-deciding authorization.
        if grok.capability is PreferredProviderCapability.AVAILABLE and not grok.ready:
            preferred_reason = ProviderFallbackReason.PREFERRED_UNAVAILABLE
        elif grok.capability is PreferredProviderCapability.AVAILABLE:
            preferred_reason = ProviderFallbackReason.NONE
        else:
            preferred_reason = map_preferred_capability_to_fallback_reason(
                grok.capability
            )

        if (
            selected_provider == PREFERRED_PROVIDER
            and router_decision.authorized
        ):
            route = ProviderRouteProvenance(
                preferred_provider=PREFERRED_PROVIDER,
                fallback_provider=FALLBACK_PROVIDER,
                selected_provider=ProviderSelection.GROK,
                fallback_reason=ProviderFallbackReason.NONE,
                fallback_receipt_cid="",
                observed_capability_cid=grok.observed_capability_cid,
                usage_evidence_cid=grok.usage_evidence_cid,
                budget_cid=grok.budget_cid,
                task_revision_cid=evidence.task_revision_cid,
                attempt_cid=evidence.attempt_cid,
                worktree_cid=evidence.worktree_cid,
                maximum_fallback_dispatches=MAXIMUM_FALLBACK_DISPATCHES,
                independent_review_required=True,
            )
            candidates = [
                _candidate(
                    field_name="provider",
                    value=ProviderSelection.GROK.value,
                    source=ResolutionSource.BUILTIN_DEFAULT,
                    precedence=90,
                    evidence_cid=grok.observed_capability_cid,
                )
            ]
            if codex is not None:
                candidates.append(
                    _candidate(
                        field_name="provider",
                        value=ProviderSelection.CODEX.value,
                        source=ResolutionSource.DISCOVERY,
                        precedence=91,
                        evidence_cid=codex.observed_capability_cid,
                        rejection_reason="preferred_provider_ready",
                    )
                )
            decision = _decision(
                field_name="provider",
                disposition=ResolutionDisposition.DEFAULTED,
                selected_value=ProviderSelection.GROK.value,
                selected_source=ResolutionSource.BUILTIN_DEFAULT,
                source_precedence=90,
                evidence_cid=evidence_cid,
                candidates=candidates,
                reason_codes=(
                    "preferred_provider_healthy",
                    "preferred_provider_policy_allowed",
                    router_decision.decision_cid,
                ),
                effect=DecisionEffect.CONFIGURATION,
            )
            return route, None, decision, tuple(degradations)

        degradations.append(
            CapabilityDegradationCode.PREFERRED_PROVIDER_DEGRADED.value
        )

        if (
            selected_provider == FALLBACK_PROVIDER
            and router_decision.authorized
            and preferred_reason
            is ProviderFallbackReason.PREFERRED_QUOTA_EXHAUSTED
            and codex is not None
            and codex.ready
        ):
            degradations.append(
                CapabilityDegradationCode.FALLBACK_PROVIDER_ONLY.value
            )
            review_authorization = _cid(
                "independent-review-authorization",
                {
                    "attempt_cid": evidence.attempt_cid,
                    "task_revision_cid": evidence.task_revision_cid,
                    "implementer": FALLBACK_PROVIDER,
                },
            )
            implementer_process_identity = _cid(
                "implementer-process",
                {
                    "attempt_cid": evidence.attempt_cid,
                    "provider": FALLBACK_PROVIDER,
                    "worktree_cid": evidence.worktree_cid,
                },
            )
            receipt = ProviderFallbackReceipt(
                preferred_provider=PREFERRED_PROVIDER,
                fallback_provider=FALLBACK_PROVIDER,
                reason_code=preferred_reason,
                observed_capability_cid=grok.observed_capability_cid,
                task_revision_cid=evidence.task_revision_cid,
                budget_cid=grok.budget_cid,
                attempt_id=evidence.attempt_cid,
                usage_evidence_cid=grok.usage_evidence_cid,
                worktree_cid=evidence.worktree_cid,
                implementer_process_identity=implementer_process_identity,
                review_authorization=review_authorization,
            )
            route = ProviderRouteProvenance(
                preferred_provider=PREFERRED_PROVIDER,
                fallback_provider=FALLBACK_PROVIDER,
                selected_provider=ProviderSelection.CODEX,
                fallback_reason=preferred_reason,
                fallback_receipt_cid=receipt.content_id,
                observed_capability_cid=codex.observed_capability_cid,
                usage_evidence_cid=codex.usage_evidence_cid,
                budget_cid=codex.budget_cid,
                task_revision_cid=evidence.task_revision_cid,
                attempt_cid=evidence.attempt_cid,
                worktree_cid=evidence.worktree_cid,
                maximum_fallback_dispatches=MAXIMUM_FALLBACK_DISPATCHES,
                independent_review_required=True,
            )
            decision = _decision(
                field_name="provider",
                disposition=ResolutionDisposition.DEFAULTED,
                selected_value=ProviderSelection.CODEX.value,
                selected_source=ResolutionSource.DISCOVERY,
                source_precedence=91,
                evidence_cid=evidence_cid,
                candidates=(
                    _candidate(
                        field_name="provider",
                        value=ProviderSelection.CODEX.value,
                        source=ResolutionSource.DISCOVERY,
                        precedence=91,
                        evidence_cid=codex.observed_capability_cid,
                    ),
                    _candidate(
                        field_name="provider",
                        value=ProviderSelection.GROK.value,
                        source=ResolutionSource.BUILTIN_DEFAULT,
                        precedence=90,
                        evidence_cid=grok.observed_capability_cid,
                        rejection_reason=preferred_reason.value,
                    ),
                ),
                reason_codes=(
                    preferred_reason.value,
                    "codex_fallback_selected",
                    router_decision.decision_cid,
                ),
                effect=DecisionEffect.CONFIGURATION,
            )
            return route, receipt, decision, tuple(sorted(set(degradations)))

        # Router denied dispatch (backoff/unavailable) or selected a provider
        # the capability surface cannot represent: fail closed.
        codex_ready_but_forbidden = codex is not None and codex.ready
        degradations.append(
            (
                CapabilityDegradationCode.FALLBACK_NOT_AUTHORIZED.value
                if codex_ready_but_forbidden
                else CapabilityDegradationCode.PROVIDERS_UNAVAILABLE.value
            )
        )
        unavailable_reason = (
            preferred_reason
            if preferred_reason is not ProviderFallbackReason.NONE
            else ProviderFallbackReason.PREFERRED_UNAVAILABLE
        )
        # Transient capacity backoff is still a fail-closed unavailable route.
        if (
            RouterOwnedProviderReason.PREFERRED_TRANSIENT_CAPACITY
            in reason_codes
        ):
            unavailable_reason = (
                ProviderFallbackReason.PREFERRED_CAPACITY_UNAVAILABLE
            )
        route = ProviderRouteProvenance(
            preferred_provider=PREFERRED_PROVIDER,
            fallback_provider=FALLBACK_PROVIDER,
            selected_provider=ProviderSelection.UNAVAILABLE,
            fallback_reason=unavailable_reason,
            fallback_receipt_cid="",
            observed_capability_cid=grok.observed_capability_cid,
            usage_evidence_cid=grok.usage_evidence_cid,
            budget_cid=grok.budget_cid,
            task_revision_cid=evidence.task_revision_cid,
            attempt_cid="",
            worktree_cid="",
            maximum_fallback_dispatches=MAXIMUM_FALLBACK_DISPATCHES,
            independent_review_required=True,
        )
        decision = _decision(
            field_name="provider",
            disposition=ResolutionDisposition.UNAVAILABLE,
            selected_value="",
            selected_source=ResolutionSource.DISCOVERY,
            source_precedence=91,
            evidence_cid=evidence_cid,
            candidates=tuple(
                [
                    _candidate(
                        field_name="provider",
                        value=ProviderSelection.GROK.value,
                        source=ResolutionSource.BUILTIN_DEFAULT,
                        precedence=90,
                        evidence_cid=grok.observed_capability_cid,
                        rejection_reason=unavailable_reason.value,
                    )
                ]
                + (
                    [
                        _candidate(
                            field_name="provider",
                            value=ProviderSelection.CODEX.value,
                            source=ResolutionSource.DISCOVERY,
                            precedence=91,
                            evidence_cid=codex.observed_capability_cid,
                            rejection_reason=(
                                "codex_fallback_requires_confirmed_grok_quota_exhaustion"
                            ),
                        )
                    ]
                    if codex_ready_but_forbidden and codex is not None
                    else []
                )
            ),
            reason_codes=(
                unavailable_reason.value,
                (
                    "codex_fallback_not_authorized"
                    if codex_ready_but_forbidden
                    else "implementation_providers_unavailable"
                ),
                router_decision.decision_cid,
            ),
            effect=DecisionEffect.CONFIGURATION,
        )
        return route, None, decision, tuple(sorted(set(degradations)))

    def _resolve_resources(
        self,
        evidence: CapabilityEvidence,
        *,
        selected: ProviderSelection,
        evidence_cid: str,
    ) -> tuple[
        ResourceEnvelope,
        TargetInferenceDecision,
        TargetInferenceDecision,
        tuple[str, ...],
    ]:
        sample = evidence.resources
        provider_concurrency = MAX_LANES
        if selected is ProviderSelection.GROK:
            provider_concurrency = evidence.providers[PREFERRED_PROVIDER].max_concurrency
        elif selected is ProviderSelection.CODEX:
            codex = evidence.providers.get(FALLBACK_PROVIDER)
            provider_concurrency = (
                codex.max_concurrency if codex is not None else 1
            )
        elif selected is ProviderSelection.UNAVAILABLE:
            provider_concurrency = 1

        labels = tuple(sample.lane_labels) + tuple(evidence.requested_lane_labels)
        lane_ceiling, lane_degradations = compute_lane_ceiling(
            sample,
            provider_concurrency=provider_concurrency,
            ignored_labels=labels,
        )
        max_processes = max(1, sample.max_processes or lane_ceiling)
        max_processes = max(max_processes, lane_ceiling)
        max_validation_workers = max(
            1, min(sample.max_validation_workers or lane_ceiling, max_processes)
        )
        cpu_millis = max(1, sample.cpu_millis or 1_000)
        memory_bytes = max(1024 * 1024, sample.memory_bytes or 1024 * 1024)
        provider_request_limit = max(1, sample.provider_request_limit or 1)
        deadline_ms = max(1, sample.deadline_ms or 1)

        budget = ResourceBudget(
            max_lanes=lane_ceiling,
            max_processes=max_processes,
            max_validation_workers=max_validation_workers,
            cpu_millis=cpu_millis,
            memory_bytes=memory_bytes,
            provider_request_limit=provider_request_limit,
            deadline_ms=deadline_ms,
        )
        resource_evidence_cid = _cid(
            "resource-sample",
            {
                "ready_width": sample.ready_width,
                "host_available_workers": sample.host_available_workers,
                "host_worker_limit": sample.host_worker_limit,
                "provider_concurrency": provider_concurrency,
                "lane_ceiling": lane_ceiling,
            },
        )
        envelope = ResourceEnvelope(
            resource_budget=budget,
            ready_width=sample.ready_width,
            lane_ceiling=lane_ceiling,
            evidence_cid=resource_evidence_cid,
            ignored_lane_labels=labels,
            degradations=lane_degradations,
            source=ResolutionSource.DISCOVERY,
        )
        resource_decision = _decision(
            field_name="resources",
            disposition=ResolutionDisposition.DEFAULTED,
            selected_value=budget.content_id,
            selected_source=ResolutionSource.DISCOVERY,
            source_precedence=80,
            evidence_cid=evidence_cid,
            candidates=(
                _candidate(
                    field_name="resources",
                    value=budget.content_id,
                    source=ResolutionSource.DISCOVERY,
                    precedence=80,
                    evidence_cid=resource_evidence_cid,
                ),
            ),
            reason_codes=("resource_sample_applied",),
            effect=DecisionEffect.CONFIGURATION,
        )
        lane_decision = _decision(
            field_name="lane_ceiling",
            disposition=ResolutionDisposition.DEFAULTED,
            selected_value=str(lane_ceiling),
            selected_source=ResolutionSource.DISCOVERY,
            source_precedence=80,
            evidence_cid=evidence_cid,
            candidates=(
                _candidate(
                    field_name="lane_ceiling",
                    value=str(lane_ceiling),
                    source=ResolutionSource.DISCOVERY,
                    precedence=80,
                    evidence_cid=resource_evidence_cid,
                ),
            ),
            reason_codes=(
                "ready_width_resource_bound",
                "lane_labels_non_authoritative",
            ),
            effect=DecisionEffect.CONFIGURATION,
        )
        return envelope, resource_decision, lane_decision, lane_degradations

    def _resolve_validation(
        self,
        evidence: CapabilityEvidence,
        *,
        evidence_cid: str,
    ) -> tuple[ValidationProfile, TargetInferenceDecision, tuple[str, ...]]:
        policy = evidence.validation
        degradations: list[str] = []
        accepted = tuple(
            argv for argv in policy.allowlisted_argv if _argv_is_safe(argv)
        )
        if len(accepted) != len(policy.allowlisted_argv):
            degradations.append(
                CapabilityDegradationCode.VALIDATION_CANDIDATES_FILTERED.value
            )
        if not accepted:
            # Conservative built-in structured validation when policy is empty.
            accepted = (("python", "-m", "pytest", "-q"),)
            degradations.append(
                CapabilityDegradationCode.VALIDATION_CANDIDATES_FILTERED.value
            )
        profile_cid = _cid(
            "validation-profile",
            {
                "policy_cid": policy.policy_cid,
                "allowlisted_argv": [list(item) for item in accepted],
            },
        )
        profile = ValidationProfile(
            profile_cid=profile_cid,
            policy_cid=policy.policy_cid,
            allowlisted_argv=accepted,
            degradations=tuple(degradations),
        )
        decision = _decision(
            field_name="validation",
            disposition=ResolutionDisposition.DEFAULTED,
            selected_value=profile_cid,
            selected_source=ResolutionSource.SIGNED_PROFILE
            if policy.policy_cid
            else ResolutionSource.BUILTIN_DEFAULT,
            source_precedence=50 if policy.policy_cid else 90,
            evidence_cid=evidence_cid,
            candidates=(
                _candidate(
                    field_name="validation",
                    value=profile_cid,
                    source=(
                        ResolutionSource.SIGNED_PROFILE
                        if policy.policy_cid
                        else ResolutionSource.BUILTIN_DEFAULT
                    ),
                    precedence=50 if policy.policy_cid else 90,
                    evidence_cid=policy.policy_cid,
                ),
            ),
            reason_codes=(
                "structured_allowlisted_argv",
                "prompt_shell_rejected",
            ),
            effect=DecisionEffect.CONFIGURATION,
        )
        return profile, decision, tuple(degradations)

    def _resolve_topology(
        self,
        evidence: CapabilityEvidence,
        *,
        evidence_cid: str,
    ) -> tuple[
        DeploymentTopology,
        TargetInferenceDecision,
        TargetInferenceDecision,
        tuple[str, ...],
    ]:
        topo = evidence.topology
        degradations: list[str] = []
        mode = topo.preferred_mode
        if mode is TopologyMode.DISTRIBUTED and not topo.distributed_capable:
            mode = TopologyMode.LOCAL
            degradations.append(
                CapabilityDegradationCode.DISTRIBUTED_TOPOLOGY_UNAVAILABLE.value
            )
        if not topo.distributed_capable:
            mode = TopologyMode.LOCAL

        database_path = f"{topo.state_root.rstrip('/')}/{topo.database_relative_path}"
        coordination = CoordinationShardBinding(
            backend="duckdb",
            database_path=database_path,
            shard_id=f"{topo.lease_namespace}-0",
            shard_count=topo.shard_count if mode is TopologyMode.DISTRIBUTED else 1,
            shard_index=0,
            owner_principal_ref=topo.owner_principal_ref,
            coordinator_cid=topo.coordinator_cid,
            lease_namespace=topo.lease_namespace,
            fencing_generation=topo.fencing_generation,
            writable=True,
            write_model="single_writer_transactional_cas",
            # Contracts require owner_rpc even for single-host local runs so
            # remote workers never share the DuckDB file directly.
            remote_access="owner_rpc",
        )

        ipfs_publish = bool(
            topo.ipfs_publish_capable
            and topo.parquet_capable
            and topo.ipfs_backend_handle
        )
        if topo.parquet_capable and not ipfs_publish:
            degradations.append(
                CapabilityDegradationCode.IPFS_PUBLICATION_UNAVAILABLE.value
            )
        replication_mode = (
            ReplicationMode.PARQUET_IPLD_IPFS
            if ipfs_publish
            else ReplicationMode.PARQUET_IPLD
        )
        if not topo.parquet_capable:
            # Still bind a parquet path under state root; publication stays off.
            replication_mode = ReplicationMode.PARQUET_IPLD
            if (
                CapabilityDegradationCode.IPFS_PUBLICATION_UNAVAILABLE.value
                not in degradations
            ):
                degradations.append(
                    CapabilityDegradationCode.IPFS_PUBLICATION_UNAVAILABLE.value
                )
            ipfs_publish = False

        parquet_path = f"{topo.state_root.rstrip('/')}/epochs"
        replication = ReplicationBinding(
            mode=replication_mode,
            parquet_dataset_path=parquet_path,
            parquet_schema_cid=_cid("parquet-schema", {"state_root": topo.state_root}),
            partition_keys=DEFAULT_PARQUET_PARTITIONS,
            ipld_manifest_schema_cid=_cid(
                "ipld-manifest-schema", {"state_root": topo.state_root}
            ),
            ipld_codec="dag-json",
            cid_profile="cidv1-base32-sha2-256",
            links_must_be_verified=True,
            car_export=True,
            ipfs_publish=ipfs_publish,
            ipfs_backend_handle=topo.ipfs_backend_handle if ipfs_publish else "",
            pin=False,
        )
        topology = DeploymentTopology(
            mode=mode,
            coordination_shard=coordination,
            replication=replication,
            degradations=tuple(sorted(set(degradations))),
        )
        coordination_decision = _decision(
            field_name="coordination",
            disposition=ResolutionDisposition.DEFAULTED,
            selected_value=coordination.content_id,
            selected_source=ResolutionSource.DISCOVERY,
            source_precedence=80,
            evidence_cid=evidence_cid,
            candidates=(
                _candidate(
                    field_name="coordination",
                    value=coordination.content_id,
                    source=ResolutionSource.DISCOVERY,
                    precedence=80,
                    evidence_cid=topo.coordinator_cid,
                ),
            ),
            reason_codes=(
                "duckdb_single_writer",
                f"topology_{mode.value}",
            ),
            effect=DecisionEffect.CONFIGURATION,
        )
        replication_decision = _decision(
            field_name="replication",
            disposition=ResolutionDisposition.DEFAULTED,
            selected_value=replication.content_id,
            selected_source=ResolutionSource.DISCOVERY,
            source_precedence=80,
            evidence_cid=evidence_cid,
            candidates=(
                _candidate(
                    field_name="replication",
                    value=replication.content_id,
                    source=ResolutionSource.DISCOVERY,
                    precedence=80,
                    evidence_cid=replication.parquet_schema_cid,
                ),
            ),
            reason_codes=(
                replication_mode.value,
                "ipfs_availability_is_not_authority",
            ),
            effect=DecisionEffect.CONFIGURATION,
        )
        return (
            topology,
            coordination_decision,
            replication_decision,
            tuple(sorted(set(degradations))),
        )


def resolve_capabilities(evidence: CapabilityEvidence) -> CapabilityResolution:
    """Module-level convenience wrapper around :class:`CapabilityResolver`."""

    return CapabilityResolver().resolve(evidence)


__all__ = [
    "ALLOWED_IMPLEMENTATION_PROVIDERS",
    "CAPABILITY_EVIDENCE_SCHEMA",
    "CAPABILITY_FIELD_NAMES",
    "CAPABILITY_RESOLUTION_SCHEMA",
    "CapabilityDegradationCode",
    "CapabilityEvidence",
    "CapabilityResolution",
    "CapabilityResolver",
    "CapabilityResolverError",
    "DEPLOYMENT_TOPOLOGY_SCHEMA",
    "DeploymentTopology",
    "FALLBACK_PROVIDER",
    "MAXIMUM_FALLBACK_DISPATCHES",
    "PREFERRED_PROVIDER",
    "PROVIDER_FALLBACK_RECEIPT_SCHEMA",
    "PreferredProviderCapability",
    "ProviderCapabilityEvidence",
    "ProviderFallbackReceipt",
    "RESOURCE_ENVELOPE_SCHEMA",
    "ResourceEnvelope",
    "ResourceSampleEvidence",
    "TopologyEvidence",
    "TopologyMode",
    "VALIDATION_PROFILE_SCHEMA",
    "ValidationPolicyEvidence",
    "ValidationProfile",
    "compute_lane_ceiling",
    "map_preferred_capability_to_fallback_reason",
    "resolve_capabilities",
]
