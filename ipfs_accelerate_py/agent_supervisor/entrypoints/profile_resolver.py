"""Deterministic profile precedence and complete target resolution (ASE-010).

This module is the composition layer for prompt-only entrypoints.  It integrates
leaf resolver outputs (repository, state, objective, authority, capability)
with profile sources under a single precedence ladder, then emits one
:class:`~ipfs_accelerate_py.agent_supervisor.entrypoints.contracts.TargetResolutionReceipt`
and one
:class:`~ipfs_accelerate_py.agent_supervisor.entrypoints.contracts.ResolvedSupervisorProfile`.

Design rules enforced here:

- selection is deterministic under identical frozen leaf outputs and profile
  layers;
- a complete canonical request disables inference and binds only its fields;
- otherwise explicit overrides, run bindings, authenticated/server policy,
  signed profiles, reviewed repository hints, discovery, and conservative
  built-in defaults merge in that order;
- lower-precedence sources may only narrow allowlists, authority, resource
  ceilings, and effect sets — never widen them;
- material ambiguity (or denied authority) blocks effects while preserving a
  safe preview receipt and, when paths are available, a demoted preview
  profile;
- the receipt is evidence about resolution, never authorization.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, ClassVar, Final, Iterable

from ipfs_accelerate_py.agent_supervisor.multiformats_identity import (
    cid_for_dag_json,
)

from .authority_resolver import (
    LOCAL_WORKTREE_ALLOWED_EFFECTS,
    PREVIEW_ALLOWED_EFFECTS,
    AuthorityResolution,
)
from .capability_resolver import CapabilityResolution
from .contracts import (
    AUTHORITY_DECISION_FIELDS,
    REQUIRED_TARGET_DECISION_FIELDS,
    CoordinationShardBinding,
    DecisionEffect,
    EntrypointContractError,
    ExpectedEffect,
    InvocationMode,
    OutputMode,
    ProviderRouteProvenance,
    ReplicationBinding,
    ReplicationMode,
    ResolutionDisposition,
    ResolutionSource,
    ResolvedSupervisorProfile,
    ResourceBudget,
    RevalidationRule,
    SupervisorInvocationRequest,
    TargetCandidate,
    TargetInferenceDecision,
    TargetResolutionReceipt,
    TaskSourceKind,
    WorktreeStrategy,
)
from .objective_resolver import ObjectiveResolution
from .state_resolver import StateResolution
from .target_resolver import RepositoryTargetResolution

SCHEMA_PREFIX: Final = "ipfs_accelerate_py/agent-supervisor/entrypoints"
PROFILE_RESOLUTION_SCHEMA: Final = f"{SCHEMA_PREFIX}/profile-resolution@1"
PROFILE_SOURCE_LAYER_SCHEMA: Final = f"{SCHEMA_PREFIX}/profile-source-layer@1"
PROFILE_COMPOSITION_REQUEST_SCHEMA: Final = (
    f"{SCHEMA_PREFIX}/profile-composition-request@1"
)
CANONICAL_REQUEST_BINDING_SCHEMA: Final = (
    f"{SCHEMA_PREFIX}/canonical-request-binding@1"
)
PRECEDENCE_TRACE_SCHEMA: Final = f"{SCHEMA_PREFIX}/profile-precedence-trace@1"
LIFECYCLE_HEALTH_CONTRACT_SCHEMA: Final = (
    f"{SCHEMA_PREFIX}/lifecycle-health-contract@1"
)

RESOLVED_SUPERVISOR_PROFILE_REQUIREMENT_ID: Final = (
    "requirement:agent-supervisor.entrypoints.resolved-supervisor-profile@1"
)

# Lower numbers win (matches target/objective/capability leaf resolvers).
SOURCE_PRECEDENCE: Final[Mapping[ResolutionSource, int]] = {
    ResolutionSource.CANONICAL_REQUEST: 10,
    ResolutionSource.EXPLICIT_OVERRIDE: 20,
    ResolutionSource.EXISTING_RUN: 30,
    ResolutionSource.AUTHENTICATED_TRANSPORT: 40,
    ResolutionSource.SIGNED_PROFILE: 50,
    ResolutionSource.REPOSITORY_HINT: 60,
    ResolutionSource.DISCOVERY: 80,
    ResolutionSource.BUILTIN_DEFAULT: 90,
}

PROFILE_OWNED_FIELDS: Final[tuple[str, ...]] = (
    "merge_target",
    "worktree_strategy",
)

REPOSITORY_FIELD_NAMES: Final[tuple[str, ...]] = (
    "repository_root",
    "repository_id",
    "checkout_id",
    "scope",
    "tree_id",
    "dirty_overlay",
    "submodules",
    "nested_repositories",
)

STATE_FIELD_NAMES: Final[tuple[str, ...]] = (
    "state_root",
    "run_namespace",
)

OBJECTIVE_FIELD_NAMES: Final[tuple[str, ...]] = (
    "objective",
    "plan",
    "task_source",
    "output",
)

AUTHORITY_FIELD_NAMES: Final[tuple[str, ...]] = (
    "policy",
    "principal",
    "authority_source",
    "effect_ceiling",
)

CAPABILITY_FIELD_NAMES: Final[tuple[str, ...]] = (
    "provider",
    "resources",
    "lane_ceiling",
    "validation",
    "coordination",
    "replication",
)

# Fields whose unresolved disposition blocks mutation effects.
MATERIAL_EFFECT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "repository_root",
        "state_root",
        "principal",
        "policy",
        "authority_source",
        "effect_ceiling",
        "provider",
        "worktree_strategy",
        "coordination",
        "validation",
    }
)

BUILTIN_PROFILE_PREVIEW: Final = "preview"
BUILTIN_PROFILE_LOCAL_WORKTREE: Final = "local-worktree"
BUILTIN_PROFILE_CI_WORKER: Final = "ci-worker"
BUILTIN_PROFILE_NAMES: Final[frozenset[str]] = frozenset(
    {
        BUILTIN_PROFILE_PREVIEW,
        BUILTIN_PROFILE_LOCAL_WORKTREE,
        BUILTIN_PROFILE_CI_WORKER,
    }
)

DEFAULT_ENVIRONMENT_NAMES: Final[tuple[str, ...]] = (
    "CODEX_HOME",
    "GROK_API_KEY",
    "HOME",
    "PATH",
    "TMPDIR",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_STATE_HOME",
)

DEFAULT_CREDENTIAL_HANDLES: Final[tuple[str, ...]] = (
    "env:CODEX_HOME",
    "env:GROK_API_KEY",
)

SUPERVISOR_MODULE: Final = (
    "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor"
)
DAEMON_MODULE: Final = (
    "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
)

class ProfileResolverError(EntrypointContractError):
    """Raised when profile composition evidence is malformed or inconsistent."""


class ProfileSourceKind(str, Enum):
    """Closed set of profile layers that may contribute configuration."""

    CANONICAL_REQUEST = "canonical_request"
    EXPLICIT_OVERRIDE = "explicit_override"
    EXISTING_RUN = "existing_run"
    AUTHENTICATED_SERVER_POLICY = "authenticated_server_policy"
    SIGNED_PROFILE = "signed_profile"
    REPOSITORY_HINT = "repository_hint"
    DISCOVERY = "discovery"
    BUILTIN_DEFAULT = "builtin_default"


def _cid(label: str, payload: Mapping[str, Any] | None = None) -> str:
    body: dict[str, Any] = {"label": label}
    if payload is not None:
        body["payload"] = dict(payload)
    return cid_for_dag_json(body)


def _require_cid(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ProfileResolverError(f"{name} is required")
    return text


def _optional_cid(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text


def _token(value: Any, name: str, *, required: bool = True) -> str:
    text = str(value or "").strip().lower()
    if not text:
        if required:
            raise ProfileResolverError(f"{name} is required")
        return ""
    if not all(ch.isalnum() or ch in "._:-" for ch in text) or not text[0].isalnum():
        raise ProfileResolverError(f"{name} must be a closed token")
    return text


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ProfileResolverError(f"{name} must be a boolean")
    return value


def _nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileResolverError(f"{name} must be an integer")
    if value < 0:
        raise ProfileResolverError(f"{name} must be non-negative")
    return value


def _optional_text(value: Any, name: str, *, maximum: int = 512) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ProfileResolverError(f"{name} must be text")
    text = value.strip()
    if len(text.encode("utf-8")) > maximum:
        raise ProfileResolverError(f"{name} exceeds {maximum} bytes")
    return text


def _effects(
    values: Sequence[ExpectedEffect] | Iterable[ExpectedEffect] | None,
    name: str,
) -> tuple[ExpectedEffect, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, (Sequence, set, frozenset)):
        raise ProfileResolverError(f"{name} must be a sequence of effects")
    items: list[ExpectedEffect] = []
    for item in values:
        if isinstance(item, ExpectedEffect):
            items.append(item)
        else:
            try:
                items.append(ExpectedEffect(str(item)))
            except ValueError as exc:
                raise ProfileResolverError(
                    f"{name} contains unknown effect {item!r}"
                ) from exc
    unique = tuple(sorted(set(items), key=lambda item: item.value))
    return unique


def _intersect_effects(
    *layers: Sequence[ExpectedEffect],
) -> tuple[ExpectedEffect, ...]:
    active = [tuple(layer) for layer in layers if layer]
    if not active:
        return ()
    result: set[ExpectedEffect] | None = None
    for layer in active:
        current = set(layer)
        result = current if result is None else result & current
    return tuple(sorted(result or (), key=lambda item: item.value))


def source_for_kind(kind: ProfileSourceKind) -> ResolutionSource:
    """Map a profile layer kind onto the shared resolution-source ladder."""

    mapping = {
        ProfileSourceKind.CANONICAL_REQUEST: ResolutionSource.CANONICAL_REQUEST,
        ProfileSourceKind.EXPLICIT_OVERRIDE: ResolutionSource.EXPLICIT_OVERRIDE,
        ProfileSourceKind.EXISTING_RUN: ResolutionSource.EXISTING_RUN,
        ProfileSourceKind.AUTHENTICATED_SERVER_POLICY: (
            ResolutionSource.AUTHENTICATED_TRANSPORT
        ),
        ProfileSourceKind.SIGNED_PROFILE: ResolutionSource.SIGNED_PROFILE,
        ProfileSourceKind.REPOSITORY_HINT: ResolutionSource.REPOSITORY_HINT,
        ProfileSourceKind.DISCOVERY: ResolutionSource.DISCOVERY,
        ProfileSourceKind.BUILTIN_DEFAULT: ResolutionSource.BUILTIN_DEFAULT,
    }
    return mapping[kind]


def builtin_profile_for_mode(mode: InvocationMode) -> str:
    """Select the conservative built-in profile name for an invocation mode."""

    mode = mode if isinstance(mode, InvocationMode) else InvocationMode(mode)
    if mode is InvocationMode.PREVIEW:
        return BUILTIN_PROFILE_PREVIEW
    if mode is InvocationMode.CI_WORKER:
        return BUILTIN_PROFILE_CI_WORKER
    if mode is InvocationMode.DISTRIBUTED_WORKER:
        return BUILTIN_PROFILE_CI_WORKER
    return BUILTIN_PROFILE_LOCAL_WORKTREE


def default_effects_for_profile(profile_name: str) -> tuple[ExpectedEffect, ...]:
    """Conservative built-in effect sets for named profiles."""

    name = str(profile_name or "").strip().lower()
    if name == BUILTIN_PROFILE_PREVIEW:
        return PREVIEW_ALLOWED_EFFECTS
    if name in {BUILTIN_PROFILE_LOCAL_WORKTREE, BUILTIN_PROFILE_CI_WORKER}:
        return LOCAL_WORKTREE_ALLOWED_EFFECTS
    return PREVIEW_ALLOWED_EFFECTS


def default_worktree_strategy_for_profile(
    profile_name: str,
) -> WorktreeStrategy:
    name = str(profile_name or "").strip().lower()
    if name == BUILTIN_PROFILE_PREVIEW:
        return WorktreeStrategy.NONE
    return WorktreeStrategy.ISOLATED


def _candidate(
    *,
    field_name: str,
    value: str,
    source: ResolutionSource,
    evidence_cid: str,
    confidence_ppm: int = 1_000_000,
    rejection_reason: str = "",
) -> TargetCandidate:
    return TargetCandidate(
        field_name=field_name,
        value=value,
        source=source,
        source_precedence=SOURCE_PRECEDENCE[source],
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
    evidence_cid: str,
    candidates: Sequence[TargetCandidate],
    reason_codes: Sequence[str],
    effect: DecisionEffect = DecisionEffect.CONFIGURATION,
    override_accepted: bool = False,
    revalidation_rule: RevalidationRule = RevalidationRule.BEFORE_MUTATION,
    fresh_until_ms: int = 0,
) -> TargetInferenceDecision:
    return TargetInferenceDecision(
        field_name=field_name,
        disposition=disposition,
        selected_value=selected_value,
        selected_source=selected_source,
        source_precedence=SOURCE_PRECEDENCE[selected_source],
        evidence_cid=evidence_cid,
        candidates=tuple(candidates),
        reason_codes=tuple(reason_codes),
        effect=effect,
        override_accepted=override_accepted,
        fresh_until_ms=fresh_until_ms,
        revalidation_rule=revalidation_rule,
    )


def _decision_map(
    decisions: Sequence[TargetInferenceDecision],
) -> dict[str, TargetInferenceDecision]:
    mapping: dict[str, TargetInferenceDecision] = {}
    for item in decisions:
        if item.field_name in mapping:
            raise ProfileResolverError(
                f"duplicate decision for field {item.field_name!r}"
            )
        mapping[item.field_name] = item
    return mapping


@dataclass(frozen=True)
class ProfileSourceLayer:
    """One typed profile layer with trust classification and narrow-only values."""

    SCHEMA: ClassVar[str] = PROFILE_SOURCE_LAYER_SCHEMA

    kind: ProfileSourceKind
    evidence_cid: str
    profile_name: str = ""
    mode: InvocationMode | None = None
    allowed_effects: tuple[ExpectedEffect, ...] = ()
    worktree_strategy: WorktreeStrategy | None = None
    merge_target: str = ""
    max_lanes: int = 0
    output_mode: OutputMode | None = None
    environment_names: tuple[str, ...] = ()
    credential_handles: tuple[str, ...] = ()
    reviewed: bool = False
    signature_verified: bool = False

    def __post_init__(self) -> None:
        kind = (
            self.kind
            if isinstance(self.kind, ProfileSourceKind)
            else ProfileSourceKind(self.kind)
        )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self, "evidence_cid", _require_cid(self.evidence_cid, "evidence_cid")
        )
        object.__setattr__(
            self,
            "profile_name",
            _token(self.profile_name, "profile_name", required=False),
        )
        if self.mode is not None and not isinstance(self.mode, InvocationMode):
            object.__setattr__(self, "mode", InvocationMode(self.mode))
        object.__setattr__(
            self,
            "allowed_effects",
            _effects(self.allowed_effects, "allowed_effects"),
        )
        if self.worktree_strategy is not None and not isinstance(
            self.worktree_strategy, WorktreeStrategy
        ):
            object.__setattr__(
                self,
                "worktree_strategy",
                WorktreeStrategy(self.worktree_strategy),
            )
        object.__setattr__(
            self,
            "merge_target",
            _optional_text(self.merge_target, "merge_target", maximum=512),
        )
        object.__setattr__(
            self, "max_lanes", _nonneg_int(self.max_lanes, "max_lanes")
        )
        if self.output_mode is not None and not isinstance(
            self.output_mode, OutputMode
        ):
            object.__setattr__(self, "output_mode", OutputMode(self.output_mode))
        object.__setattr__(
            self,
            "environment_names",
            tuple(sorted({str(item) for item in self.environment_names if item})),
        )
        object.__setattr__(
            self,
            "credential_handles",
            tuple(sorted({str(item) for item in self.credential_handles if item})),
        )
        object.__setattr__(self, "reviewed", _bool(self.reviewed, "reviewed"))
        object.__setattr__(
            self,
            "signature_verified",
            _bool(self.signature_verified, "signature_verified"),
        )
        if kind is ProfileSourceKind.SIGNED_PROFILE and not self.signature_verified:
            raise ProfileResolverError(
                "signed profile layers require signature_verified=True"
            )
        if kind is ProfileSourceKind.REPOSITORY_HINT and not self.reviewed:
            raise ProfileResolverError(
                "repository profile hints must be reviewed before use"
            )
        if (
            kind is ProfileSourceKind.AUTHENTICATED_SERVER_POLICY
            and not self.signature_verified
            and not self.reviewed
        ):
            raise ProfileResolverError(
                "server policy profiles require verification or review"
            )

    @property
    def source(self) -> ResolutionSource:
        return source_for_kind(self.kind)

    @property
    def source_precedence(self) -> int:
        return SOURCE_PRECEDENCE[self.source]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "kind": self.kind.value,
            "evidence_cid": self.evidence_cid,
            "profile_name": self.profile_name,
            "mode": None if self.mode is None else self.mode.value,
            "allowed_effects": [item.value for item in self.allowed_effects],
            "worktree_strategy": (
                None
                if self.worktree_strategy is None
                else self.worktree_strategy.value
            ),
            "merge_target": self.merge_target,
            "max_lanes": self.max_lanes,
            "output_mode": (
                None if self.output_mode is None else self.output_mode.value
            ),
            "environment_names": list(self.environment_names),
            "credential_handles": list(self.credential_handles),
            "reviewed": self.reviewed,
            "signature_verified": self.signature_verified,
            "source": self.source.value,
            "source_precedence": self.source_precedence,
        }


@dataclass(frozen=True)
class CanonicalRequestBinding:
    """Complete canonical request that disables inference for all fields."""

    SCHEMA: ClassVar[str] = CANONICAL_REQUEST_BINDING_SCHEMA

    request_cid: str
    decisions: tuple[TargetInferenceDecision, ...]
    # Optional fully-resolved projections used when decisions alone are not
    # enough for structured fields.
    provider_route: ProviderRouteProvenance | None = None
    resource_budget: ResourceBudget | None = None
    coordination_shard: CoordinationShardBinding | None = None
    replication: ReplicationBinding | None = None
    task_source_kind: TaskSourceKind = TaskSourceKind.DUAL
    objective_revision_cid: str = ""
    task_source_revision_cid: str = ""
    markdown_path: str = ""
    duckdb_path: str = ""
    capability_report_cid: str = ""
    configuration_root_cid: str = ""
    capability_catalog_cid: str = ""
    profile_name: str = BUILTIN_PROFILE_LOCAL_WORKTREE
    expected_effects: tuple[ExpectedEffect, ...] = ()
    environment_names: tuple[str, ...] = ()
    credential_handles: tuple[str, ...] = ()
    lifecycle_health_contract_cid: str = ""
    supervisor_argv: tuple[str, ...] = ()
    daemon_argv: tuple[str, ...] = ()
    task_source_path: str = ""
    mode: InvocationMode = InvocationMode.WORKTREE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_cid", _require_cid(self.request_cid, "request_cid")
        )
        if not isinstance(self.decisions, Sequence) or isinstance(
            self.decisions, (str, bytes)
        ):
            raise ProfileResolverError("canonical decisions must be a sequence")
        decisions = tuple(
            item
            if isinstance(item, TargetInferenceDecision)
            else TargetInferenceDecision.from_dict(item)
            for item in self.decisions
        )
        names = {item.field_name for item in decisions}
        missing = set(REQUIRED_TARGET_DECISION_FIELDS) - names
        extra = names - set(REQUIRED_TARGET_DECISION_FIELDS)
        if missing or extra:
            raise ProfileResolverError(
                "canonical request decisions have "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )
        if any(item.unresolved for item in decisions):
            raise ProfileResolverError(
                "canonical request must fully resolve every target field"
            )
        for item in decisions:
            if item.field_name in AUTHORITY_DECISION_FIELDS:
                # Contracts forbid non-trusted sources on authority fields;
                # a canonical request still rebinds them from trusted evidence.
                if item.selected_source not in {
                    ResolutionSource.EXISTING_RUN,
                    ResolutionSource.AUTHENTICATED_TRANSPORT,
                    ResolutionSource.SIGNED_PROFILE,
                }:
                    raise ProfileResolverError(
                        "canonical authority fields require a trusted authority source"
                    )
            elif item.selected_source is not ResolutionSource.CANONICAL_REQUEST:
                raise ProfileResolverError(
                    "canonical non-authority decisions must use the "
                    "canonical_request source"
                )
        object.__setattr__(
            self,
            "decisions",
            tuple(sorted(decisions, key=lambda item: item.field_name)),
        )
        if self.provider_route is not None and not isinstance(
            self.provider_route, ProviderRouteProvenance
        ):
            raise ProfileResolverError(
                "provider_route must be ProviderRouteProvenance"
            )
        if self.resource_budget is not None and not isinstance(
            self.resource_budget, ResourceBudget
        ):
            raise ProfileResolverError("resource_budget must be ResourceBudget")
        if self.coordination_shard is not None and not isinstance(
            self.coordination_shard, CoordinationShardBinding
        ):
            raise ProfileResolverError(
                "coordination_shard must be CoordinationShardBinding"
            )
        if self.replication is not None and not isinstance(
            self.replication, ReplicationBinding
        ):
            raise ProfileResolverError("replication must be ReplicationBinding")
        object.__setattr__(
            self,
            "task_source_kind",
            self.task_source_kind
            if isinstance(self.task_source_kind, TaskSourceKind)
            else TaskSourceKind(self.task_source_kind),
        )
        object.__setattr__(
            self,
            "mode",
            self.mode
            if isinstance(self.mode, InvocationMode)
            else InvocationMode(self.mode),
        )
        object.__setattr__(
            self,
            "expected_effects",
            _effects(self.expected_effects, "expected_effects"),
        )
        object.__setattr__(
            self,
            "profile_name",
            _token(self.profile_name, "profile_name"),
        )


@dataclass(frozen=True)
class ProfileCompositionRequest:
    """Injected leaf resolutions and profile layers for one composition."""

    SCHEMA: ClassVar[str] = PROFILE_COMPOSITION_REQUEST_SCHEMA

    invocation: SupervisorInvocationRequest
    repository: RepositoryTargetResolution
    state: StateResolution
    objective: ObjectiveResolution
    authority: AuthorityResolution
    capability: CapabilityResolution
    profile_layers: tuple[ProfileSourceLayer, ...] = ()
    canonical_request: CanonicalRequestBinding | None = None
    resolved_at_ms: int = 0
    fresh_until_ms: int = 0
    configuration_root_cid: str = ""
    capability_catalog_cid: str = ""
    capability_report_cid: str = ""
    lifecycle_health_contract_cid: str = ""
    merge_target_hint: str = ""
    worktree_strategy_override: WorktreeStrategy | None = None
    environment_names: tuple[str, ...] = ()
    credential_handles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.invocation, SupervisorInvocationRequest):
            raise ProfileResolverError(
                "invocation must be SupervisorInvocationRequest"
            )
        if not isinstance(self.repository, RepositoryTargetResolution):
            raise ProfileResolverError(
                "repository must be RepositoryTargetResolution"
            )
        if not isinstance(self.state, StateResolution):
            raise ProfileResolverError("state must be StateResolution")
        if not isinstance(self.objective, ObjectiveResolution):
            raise ProfileResolverError("objective must be ObjectiveResolution")
        if not isinstance(self.authority, AuthorityResolution):
            raise ProfileResolverError("authority must be AuthorityResolution")
        if not isinstance(self.capability, CapabilityResolution):
            raise ProfileResolverError(
                "capability must be CapabilityResolution"
            )
        layers = tuple(self.profile_layers or ())
        for layer in layers:
            if not isinstance(layer, ProfileSourceLayer):
                raise ProfileResolverError(
                    "profile_layers must contain ProfileSourceLayer values"
                )
        # Stable order: stronger sources first, then evidence identity.
        layers = tuple(
            sorted(
                layers,
                key=lambda item: (item.source_precedence, item.evidence_cid),
            )
        )
        object.__setattr__(self, "profile_layers", layers)
        object.__setattr__(
            self, "resolved_at_ms", _nonneg_int(self.resolved_at_ms, "resolved_at_ms")
        )
        object.__setattr__(
            self,
            "fresh_until_ms",
            _nonneg_int(self.fresh_until_ms, "fresh_until_ms"),
        )
        if self.fresh_until_ms and self.fresh_until_ms < self.resolved_at_ms:
            raise ProfileResolverError(
                "fresh_until_ms cannot precede resolved_at_ms"
            )
        for name in (
            "configuration_root_cid",
            "capability_catalog_cid",
            "capability_report_cid",
            "lifecycle_health_contract_cid",
        ):
            object.__setattr__(
                self, name, _optional_cid(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "merge_target_hint",
            _optional_text(self.merge_target_hint, "merge_target_hint"),
        )
        if self.worktree_strategy_override is not None and not isinstance(
            self.worktree_strategy_override, WorktreeStrategy
        ):
            object.__setattr__(
                self,
                "worktree_strategy_override",
                WorktreeStrategy(self.worktree_strategy_override),
            )
        object.__setattr__(
            self,
            "environment_names",
            tuple(sorted({str(item) for item in self.environment_names if item})),
        )
        object.__setattr__(
            self,
            "credential_handles",
            tuple(sorted({str(item) for item in self.credential_handles if item})),
        )
        # Canonical request on the invocation without a binding is incomplete.
        if (
            self.invocation.canonical_request_cid
            and self.canonical_request is None
        ):
            raise ProfileResolverError(
                "invocation.canonical_request_cid requires a CanonicalRequestBinding"
            )
        if (
            self.canonical_request is not None
            and self.invocation.canonical_request_cid
            and self.canonical_request.request_cid
            != self.invocation.canonical_request_cid
        ):
            raise ProfileResolverError(
                "canonical request CID must match the invocation binding"
            )


@dataclass(frozen=True)
class PrecedenceTraceEntry:
    """One field's selected source and rejected lower-precedence alternatives."""

    field_name: str
    selected_source: ResolutionSource
    source_precedence: int
    selected_value: str
    disposition: ResolutionDisposition
    rejected_sources: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "selected_source": self.selected_source.value,
            "source_precedence": self.source_precedence,
            "selected_value": self.selected_value,
            "disposition": self.disposition.value,
            "rejected_sources": list(self.rejected_sources),
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class ProfileResolution:
    """Complete composition result: receipt, optional profile, and traces."""

    SCHEMA: ClassVar[str] = PROFILE_RESOLUTION_SCHEMA

    requirement_id: str
    receipt: TargetResolutionReceipt
    profile: ResolvedSupervisorProfile | None
    decisions: tuple[TargetInferenceDecision, ...]
    precedence_trace: tuple[PrecedenceTraceEntry, ...]
    profile_source_cid: str
    profile_name: str
    inference_disabled: bool
    effects_blocked: bool
    safe_preview: bool
    cross_field_inconsistencies: tuple[str, ...]
    reason_codes: tuple[str, ...]
    expected_effects: tuple[ExpectedEffect, ...]
    layers_applied: tuple[ProfileSourceLayer, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requirement_id",
            str(self.requirement_id or "").strip()
            or RESOLVED_SUPERVISOR_PROFILE_REQUIREMENT_ID,
        )
        if not self.requirement_id.startswith("requirement:"):
            raise ProfileResolverError(
                "requirement_id must be a requirement: token"
            )
        if not isinstance(self.receipt, TargetResolutionReceipt):
            raise ProfileResolverError("receipt must be TargetResolutionReceipt")
        if self.profile is not None and not isinstance(
            self.profile, ResolvedSupervisorProfile
        ):
            raise ProfileResolverError(
                "profile must be ResolvedSupervisorProfile or None"
            )
        object.__setattr__(
            self,
            "decisions",
            tuple(sorted(self.decisions, key=lambda item: item.field_name)),
        )
        object.__setattr__(
            self,
            "precedence_trace",
            tuple(
                sorted(self.precedence_trace, key=lambda item: item.field_name)
            ),
        )
        object.__setattr__(
            self,
            "profile_source_cid",
            _require_cid(self.profile_source_cid, "profile_source_cid"),
        )
        object.__setattr__(
            self, "profile_name", _token(self.profile_name, "profile_name")
        )
        for name in ("inference_disabled", "effects_blocked", "safe_preview"):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        object.__setattr__(
            self,
            "cross_field_inconsistencies",
            tuple(sorted({str(item) for item in self.cross_field_inconsistencies})),
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(sorted({str(item) for item in self.reason_codes if item})),
        )
        object.__setattr__(
            self,
            "expected_effects",
            _effects(self.expected_effects, "expected_effects"),
        )
        object.__setattr__(self, "layers_applied", tuple(self.layers_applied))
        if self.effects_blocked and not self.safe_preview:
            raise ProfileResolverError(
                "effects_blocked compositions must remain in safe_preview"
            )
        if self.receipt.is_authorization:
            raise ProfileResolverError(
                "composed receipt must never claim authorization"
            )
        if self.profile is not None:
            if (
                self.profile.target_resolution_receipt_cid
                != self.receipt.content_id
            ):
                raise ProfileResolverError(
                    "profile must bind the composed receipt identity"
                )

    @property
    def receipt_cid(self) -> str:
        return self.receipt.content_id

    @property
    def profile_cid(self) -> str:
        return "" if self.profile is None else self.profile.content_id

    @property
    def authorizes_effects(self) -> bool:
        return False

    def decision_for(self, field_name: str) -> TargetInferenceDecision:
        for item in self.decisions:
            if item.field_name == field_name:
                return item
        raise KeyError(field_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "requirement_id": self.requirement_id,
            "receipt": self.receipt.to_dict(),
            "profile": None if self.profile is None else self.profile.to_dict(),
            "decisions": [item.to_dict() for item in self.decisions],
            "precedence_trace": [
                item.to_dict() for item in self.precedence_trace
            ],
            "profile_source_cid": self.profile_source_cid,
            "profile_name": self.profile_name,
            "inference_disabled": self.inference_disabled,
            "effects_blocked": self.effects_blocked,
            "safe_preview": self.safe_preview,
            "cross_field_inconsistencies": list(self.cross_field_inconsistencies),
            "reason_codes": list(self.reason_codes),
            "expected_effects": [item.value for item in self.expected_effects],
            "layers_applied": [item.to_dict() for item in self.layers_applied],
            "receipt_cid": self.receipt_cid,
            "profile_cid": self.profile_cid,
            "authorizes_effects": self.authorizes_effects,
        }


def _leaf_decisions(
    request: ProfileCompositionRequest,
) -> dict[str, TargetInferenceDecision]:
    mapping: dict[str, TargetInferenceDecision] = {}
    groups = (
        (REPOSITORY_FIELD_NAMES, request.repository.decisions),
        (
            STATE_FIELD_NAMES,
            (
                request.state.state_root_decision,
                request.state.run_namespace_decision,
            ),
        ),
        (OBJECTIVE_FIELD_NAMES, request.objective.decisions),
        (AUTHORITY_FIELD_NAMES, request.authority.decisions),
        (CAPABILITY_FIELD_NAMES, request.capability.decisions),
    )
    for expected_names, decisions in groups:
        by_name = _decision_map(decisions)
        missing = set(expected_names) - set(by_name)
        if missing:
            raise ProfileResolverError(
                f"leaf resolution missing decisions: {sorted(missing)}"
            )
        for name in expected_names:
            if name in mapping:
                raise ProfileResolverError(f"overlapping leaf field {name!r}")
            mapping[name] = by_name[name]
    return mapping


def _select_profile_layers(
    request: ProfileCompositionRequest,
) -> tuple[ProfileSourceLayer, ...]:
    """Return ordered layers (strongest first) including implicit defaults."""

    layers: list[ProfileSourceLayer] = list(request.profile_layers)
    # Explicit high-level hints from the composition request.
    if request.merge_target_hint or request.worktree_strategy_override is not None:
        layers.append(
            ProfileSourceLayer(
                kind=ProfileSourceKind.EXPLICIT_OVERRIDE,
                evidence_cid=_cid(
                    "explicit-profile-override",
                    {
                        "merge_target": request.merge_target_hint,
                        "worktree_strategy": (
                            None
                            if request.worktree_strategy_override is None
                            else request.worktree_strategy_override.value
                        ),
                    },
                ),
                merge_target=request.merge_target_hint,
                worktree_strategy=request.worktree_strategy_override,
                reviewed=True,
                signature_verified=True,
            )
        )
    # Invocation profile hint is non-authoritative; treat recognized built-in
    # names as discovery of a preferred profile name only.
    if request.invocation.profile_hint:
        hint = request.invocation.profile_hint.strip().lower()
        layers.append(
            ProfileSourceLayer(
                kind=ProfileSourceKind.DISCOVERY,
                evidence_cid=_cid(
                    "invocation-profile-hint",
                    {"profile_hint": request.invocation.profile_hint},
                ),
                profile_name=hint if hint in BUILTIN_PROFILE_NAMES else "",
            )
        )
    # Conservative built-in default always present as the floor.
    default_name = builtin_profile_for_mode(request.invocation.mode)
    layers.append(
        ProfileSourceLayer(
            kind=ProfileSourceKind.BUILTIN_DEFAULT,
            evidence_cid=_cid(
                "builtin-profile-default",
                {"profile_name": default_name, "mode": request.invocation.mode.value},
            ),
            profile_name=default_name,
            mode=request.invocation.mode,
            allowed_effects=default_effects_for_profile(default_name),
            worktree_strategy=default_worktree_strategy_for_profile(default_name),
            merge_target="",
            max_lanes=request.invocation.budget.max_lanes,
            output_mode=OutputMode.BOTH,
            environment_names=DEFAULT_ENVIRONMENT_NAMES,
            credential_handles=DEFAULT_CREDENTIAL_HANDLES,
            reviewed=True,
            signature_verified=True,
        )
    )
    # Drop unusable discovery layers with empty contribution.
    usable = [
        layer
        for layer in layers
        if layer.kind is not ProfileSourceKind.DISCOVERY
        or layer.profile_name
        or layer.merge_target
        or layer.worktree_strategy is not None
        or layer.allowed_effects
        or layer.max_lanes
    ]
    return tuple(
        sorted(usable, key=lambda item: (item.source_precedence, item.evidence_cid))
    )


def _merge_profile_configuration(
    layers: Sequence[ProfileSourceLayer],
    *,
    authority: AuthorityResolution,
    capability: CapabilityResolution,
    invocation: SupervisorInvocationRequest,
) -> tuple[
    str,
    InvocationMode,
    tuple[ExpectedEffect, ...],
    WorktreeStrategy,
    str,
    int,
    OutputMode | None,
    tuple[str, ...],
    tuple[str, ...],
    str,
    list[str],
    list[PrecedenceTraceEntry],
]:
    """Merge profile-level configuration with narrow-only lower sources."""

    reasons: list[str] = []
    trace: list[PrecedenceTraceEntry] = []

    # Profile name: first non-empty from strongest layer.
    profile_name = ""
    profile_source = ResolutionSource.BUILTIN_DEFAULT
    rejected_names: list[str] = []
    for layer in layers:
        if not layer.profile_name:
            continue
        if not profile_name:
            profile_name = layer.profile_name
            profile_source = layer.source
        elif layer.profile_name != profile_name:
            rejected_names.append(f"{layer.source.value}:{layer.profile_name}")
    if not profile_name:
        profile_name = builtin_profile_for_mode(invocation.mode)
        profile_source = ResolutionSource.BUILTIN_DEFAULT
        reasons.append("profile_name_builtin_default")
    else:
        reasons.append(f"profile_name_from_{profile_source.value}")

    # Mode: strongest explicit mode, else invocation mode.
    mode = invocation.mode
    mode_source = ResolutionSource.EXPLICIT_OVERRIDE
    for layer in layers:
        if layer.mode is not None:
            mode = layer.mode
            mode_source = layer.source
            break

    # Effects: start from strongest contributing layer (or authority ceiling /
    # mode defaults), then only intersect lower layers.
    effect_layers: list[tuple[ResolutionSource, tuple[ExpectedEffect, ...]]] = []
    for layer in layers:
        if layer.allowed_effects:
            effect_layers.append((layer.source, layer.allowed_effects))
    if authority.authorized and authority.effect_ceiling.allowed_effects:
        effect_layers.insert(
            0,
            (
                authority.authority_source,
                authority.effect_ceiling.allowed_effects,
            ),
        )
    if not effect_layers:
        effect_layers.append(
            (
                ResolutionSource.BUILTIN_DEFAULT,
                default_effects_for_profile(profile_name),
            )
        )
    # Strongest first already; intersect all contributing sets so lower
    # sources cannot widen.
    expected_effects = _intersect_effects(*(items for _, items in effect_layers))
    # Authority ceiling always applies as a hard cap when authorized.
    if authority.authorized:
        expected_effects = _intersect_effects(
            expected_effects, authority.effect_ceiling.allowed_effects
        )
    # Mode hard caps.
    expected_effects = _intersect_effects(
        expected_effects, default_effects_for_profile(profile_name)
    )
    if not expected_effects:
        expected_effects = PREVIEW_ALLOWED_EFFECTS
        reasons.append("effects_collapsed_to_preview")
    effect_source = effect_layers[0][0]
    lower_widen_attempts = [
        f"{source.value}:{','.join(item.value for item in effects)}"
        for source, effects in effect_layers[1:]
        if set(effects) - set(effect_layers[0][1])
    ]
    if lower_widen_attempts:
        reasons.append("lower_source_effect_widen_ignored")
    trace.append(
        PrecedenceTraceEntry(
            field_name="expected_effects",
            selected_source=effect_source,
            source_precedence=SOURCE_PRECEDENCE[effect_source],
            selected_value=",".join(item.value for item in expected_effects),
            disposition=ResolutionDisposition.UNIQUE,
            rejected_sources=tuple(lower_widen_attempts),
            reason_codes=("effects_intersected_narrow_only",),
        )
    )

    # Worktree strategy: strongest non-None; lower sources cannot escalate
    # NONE -> ISOLATED or to CURRENT_CHECKOUT.
    strategy: WorktreeStrategy | None = None
    strategy_source = ResolutionSource.BUILTIN_DEFAULT
    rejected_strategies: list[str] = []
    for layer in layers:
        if layer.worktree_strategy is None:
            continue
        if layer.worktree_strategy is WorktreeStrategy.CURRENT_CHECKOUT:
            rejected_strategies.append(
                f"{layer.source.value}:current_checkout_rewrite_denied"
            )
            reasons.append("current_checkout_rewrite_denied")
            continue
        if strategy is None:
            strategy = layer.worktree_strategy
            strategy_source = layer.source
            continue
        # Lower source cannot widen NONE to ISOLATED.
        if (
            strategy is WorktreeStrategy.NONE
            and layer.worktree_strategy is WorktreeStrategy.ISOLATED
        ):
            rejected_strategies.append(
                f"{layer.source.value}:isolated_widen_denied"
            )
            reasons.append("lower_source_worktree_widen_ignored")
            continue
        # Lower source may narrow ISOLATED -> NONE.
        if (
            strategy is WorktreeStrategy.ISOLATED
            and layer.worktree_strategy is WorktreeStrategy.NONE
        ):
            strategy = WorktreeStrategy.NONE
            strategy_source = layer.source
            reasons.append("worktree_narrowed_by_lower_source")
    if strategy is None:
        strategy = default_worktree_strategy_for_profile(profile_name)
        strategy_source = ResolutionSource.BUILTIN_DEFAULT
    if authority.authorized:
        # Authority ceiling strategy is a hard cap (cannot widen past it).
        auth_strategy = authority.effect_ceiling.worktree_strategy
        if (
            auth_strategy is WorktreeStrategy.NONE
            and strategy is WorktreeStrategy.ISOLATED
        ):
            strategy = WorktreeStrategy.NONE
            strategy_source = authority.authority_source
            reasons.append("worktree_capped_by_authority")
        if auth_strategy is WorktreeStrategy.CURRENT_CHECKOUT:
            strategy = WorktreeStrategy.NONE
            reasons.append("authority_current_checkout_denied")
    if mode is InvocationMode.PREVIEW:
        strategy = WorktreeStrategy.NONE
        strategy_source = ResolutionSource.BUILTIN_DEFAULT
        reasons.append("preview_mode_disables_worktree")

    # Merge target: strongest non-empty; empty is always safe.
    merge_target = ""
    merge_source = ResolutionSource.BUILTIN_DEFAULT
    rejected_merge: list[str] = []
    for layer in layers:
        if not layer.merge_target:
            continue
        if not merge_target:
            merge_target = layer.merge_target
            merge_source = layer.source
        elif layer.merge_target != merge_target:
            rejected_merge.append(f"{layer.source.value}:{layer.merge_target}")
    # Merge effect remains denied by default; merge_target is identity only.

    # Lane ceiling: min of capability and any layer max_lanes (narrow only).
    lane_ceiling = capability.resources.lane_ceiling
    for layer in layers:
        if layer.max_lanes > 0:
            if layer.max_lanes < lane_ceiling:
                lane_ceiling = layer.max_lanes
                reasons.append(f"lane_ceiling_narrowed_by_{layer.source.value}")
            elif layer.max_lanes > lane_ceiling:
                reasons.append(f"lane_ceiling_widen_ignored_{layer.source.value}")

    # Output mode: strongest explicit, else leave to objective resolution.
    output_mode: OutputMode | None = None
    for layer in layers:
        if layer.output_mode is not None:
            output_mode = layer.output_mode
            break

    # Environment names / credential handles: strongest non-empty, lower may
    # only remove names (intersection when multiple contribute).
    env_contributions = [layer.environment_names for layer in layers if layer.environment_names]
    if env_contributions:
        env_names = set(env_contributions[0])
        for contrib in env_contributions[1:]:
            env_names &= set(contrib)
        environment_names = tuple(sorted(env_names)) if env_names else env_contributions[0]
    else:
        environment_names = DEFAULT_ENVIRONMENT_NAMES

    cred_contributions = [
        layer.credential_handles for layer in layers if layer.credential_handles
    ]
    if cred_contributions:
        creds = set(cred_contributions[0])
        for contrib in cred_contributions[1:]:
            creds &= set(contrib)
        credential_handles = (
            tuple(sorted(creds)) if creds else cred_contributions[0]
        )
    else:
        credential_handles = DEFAULT_CREDENTIAL_HANDLES

    profile_source_cid = _cid(
        "profile-source",
        {
            "profile_name": profile_name,
            "source": profile_source.value,
            "layers": [layer.evidence_cid for layer in layers],
        },
    )
    trace.append(
        PrecedenceTraceEntry(
            field_name="profile_name",
            selected_source=profile_source,
            source_precedence=SOURCE_PRECEDENCE[profile_source],
            selected_value=profile_name,
            disposition=ResolutionDisposition.UNIQUE,
            rejected_sources=tuple(rejected_names),
            reason_codes=("profile_name_selected",),
        )
    )
    trace.append(
        PrecedenceTraceEntry(
            field_name="mode",
            selected_source=mode_source,
            source_precedence=SOURCE_PRECEDENCE[mode_source],
            selected_value=mode.value,
            disposition=ResolutionDisposition.UNIQUE,
            reason_codes=("mode_selected",),
        )
    )
    return (
        profile_name,
        mode,
        expected_effects,
        strategy,
        merge_target,
        lane_ceiling,
        output_mode,
        environment_names,
        credential_handles,
        profile_source_cid,
        reasons,
        trace,
    )


def _detect_cross_field_inconsistencies(
    request: ProfileCompositionRequest,
) -> tuple[str, ...]:
    issues: list[str] = []
    binding = request.repository.binding
    state = request.state
    authority = request.authority
    capability = request.capability
    objective = request.objective

    if binding is not None:
        if binding.repository_id and binding.repository_id != state.repository_id:
            issues.append("repository_id_mismatch_state")
        if binding.checkout_id and state.checkout_id and binding.checkout_id != state.checkout_id:
            issues.append("checkout_id_mismatch_state")
        if not state.state_root or state.state_root == binding.repository_root:
            issues.append("state_root_not_outside_repository")
        if state.state_root.startswith(binding.repository_root.rstrip("/") + "/"):
            issues.append("state_root_nested_under_repository")

    if authority.authorized:
        owner = capability.topology.coordination_shard.owner_principal_ref
        if owner and owner != authority.principal.principal_ref:
            issues.append("coordination_owner_principal_mismatch")
        if (
            capability.topology.coordination_shard.writable
            and owner != authority.principal.principal_ref
        ):
            issues.append("writable_shard_requires_owner_principal")

    if objective.output is not None and binding is not None:
        if not objective.output.outside_source_checkout:
            issues.append("output_paths_dirty_source_checkout")

    # Capability lane ceiling cannot exceed resource budget.
    if (
        capability.resources.lane_ceiling
        > capability.resources.resource_budget.max_lanes
    ):
        issues.append("lane_ceiling_exceeds_resource_budget")

    return tuple(sorted(set(issues)))


def _replace_decision_value(
    decision: TargetInferenceDecision,
    *,
    selected_value: str,
    selected_source: ResolutionSource,
    evidence_cid: str,
    reason_codes: Sequence[str],
    disposition: ResolutionDisposition = ResolutionDisposition.DEFAULTED,
    effect: DecisionEffect | None = None,
) -> TargetInferenceDecision:
    candidate = _candidate(
        field_name=decision.field_name,
        value=selected_value,
        source=selected_source,
        evidence_cid=evidence_cid,
    )
    # Preserve rejected alternatives as documentation when demoting.
    rejected = []
    for item in decision.candidates:
        if item.value == selected_value and item.source is selected_source:
            continue
        if item.rejection_reason:
            rejected.append(item)
        else:
            rejected.append(
                _candidate(
                    field_name=item.field_name,
                    value=item.value,
                    source=item.source,
                    evidence_cid=item.evidence_cid,
                    confidence_ppm=item.confidence_ppm,
                    rejection_reason="superseded_by_profile_composition",
                )
            )
    return _decision(
        field_name=decision.field_name,
        disposition=disposition,
        selected_value=selected_value,
        selected_source=selected_source,
        evidence_cid=evidence_cid,
        candidates=(candidate, *rejected),
        reason_codes=tuple(sorted(set(reason_codes) | set(decision.reason_codes))),
        effect=effect or decision.effect,
        override_accepted=False,
        revalidation_rule=decision.revalidation_rule,
        fresh_until_ms=decision.fresh_until_ms,
    )


def _demote_coordination(
    shard: CoordinationShardBinding,
) -> CoordinationShardBinding:
    return replace(shard, writable=False)


def _demote_replication(replication: ReplicationBinding) -> ReplicationBinding:
    return replace(
        replication,
        mode=ReplicationMode.PARQUET_IPLD,
        ipfs_publish=False,
        ipfs_backend_handle="",
        pin=False,
    )


def _compile_supervisor_argv(
    *,
    state_root: str,
    run_namespace: str,
    profile_name: str,
) -> tuple[str, ...]:
    return (
        "python",
        "-m",
        SUPERVISOR_MODULE,
        "--state-dir",
        state_root,
        "--run-namespace",
        run_namespace,
        "--profile",
        profile_name,
    )


def _compile_daemon_argv(
    *,
    task_source_path: str,
    state_root: str,
) -> tuple[str, ...]:
    return (
        "python",
        "-m",
        DAEMON_MODULE,
        "--todo-path",
        task_source_path,
        "--state-dir",
        state_root,
    )


def _lifecycle_health_cid(
    *,
    profile_name: str,
    repository_id: str,
    state_root: str,
    task_source_kind: TaskSourceKind,
) -> str:
    return _cid(
        "lifecycle-health-contract",
        {
            "schema": LIFECYCLE_HEALTH_CONTRACT_SCHEMA,
            "profile_name": profile_name,
            "repository_id": repository_id,
            "state_root": state_root,
            "task_source_kind": task_source_kind.value,
        },
    )


def build_target_resolution_receipt(
    *,
    invocation: SupervisorInvocationRequest,
    decisions: Sequence[TargetInferenceDecision],
    repository_root: str = "",
    repository_id: str = "",
    checkout_id: str = "",
    scope_path: str = "",
    head_tree_cid: str = "",
    dirty_overlay_cid: str = "",
    submodule_population_cid: str = "",
    nested_repository_population_cid: str = "",
    state_root: str = "",
    run_namespace: str = "",
    objective_cid: str = "",
    objective_revision_cid: str = "",
    plan_cid: str = "",
    task_source_cid: str = "",
    task_source_revision_cid: str = "",
    task_source_kind: TaskSourceKind = TaskSourceKind.DUAL,
    policy_cid: str = "",
    principal_ref: str = "",
    authority_source_ref: str = "",
    effect_ceiling_cid: str = "",
    output_mode: OutputMode = OutputMode.BOTH,
    markdown_path: str = "",
    duckdb_path: str = "",
    provider_route: ProviderRouteProvenance,
    capability_report_cid: str = "",
    resource_budget_cid: str = "",
    lane_ceiling: int = 1,
    merge_target: str = "",
    worktree_strategy: WorktreeStrategy = WorktreeStrategy.NONE,
    validation_profile_cid: str = "",
    coordination_shard: CoordinationShardBinding,
    replication: ReplicationBinding,
    configuration_root_cid: str = "",
    capability_catalog_cid: str = "",
    resolved_at_ms: int = 0,
    fresh_until_ms: int = 0,
) -> TargetResolutionReceipt:
    """Build a complete :class:`TargetResolutionReceipt` from projections.

    Decisions must cover :data:`REQUIRED_TARGET_DECISION_FIELDS` exactly and
    match the projected values for every resolved field.
    """

    decision_tuple = tuple(
        sorted(decisions, key=lambda item: item.field_name)
    )
    unresolved = tuple(
        sorted(item.field_name for item in decision_tuple if item.unresolved)
    )
    return TargetResolutionReceipt(
        invocation_cid=invocation.content_id,
        prompt_cid=invocation.prompt_cid,
        repository_root=repository_root,
        repository_id=repository_id,
        checkout_id=checkout_id,
        scope_path=scope_path,
        head_tree_cid=head_tree_cid,
        dirty_overlay_cid=dirty_overlay_cid,
        submodule_population_cid=submodule_population_cid,
        nested_repository_population_cid=nested_repository_population_cid,
        state_root=state_root,
        run_namespace=run_namespace,
        objective_cid=objective_cid,
        objective_revision_cid=objective_revision_cid,
        plan_cid=plan_cid,
        task_source_cid=task_source_cid,
        task_source_revision_cid=task_source_revision_cid,
        task_source_kind=task_source_kind,
        policy_cid=policy_cid,
        principal_ref=principal_ref,
        authority_source_ref=authority_source_ref,
        effect_ceiling_cid=effect_ceiling_cid,
        output_mode=output_mode,
        markdown_path=markdown_path,
        duckdb_path=duckdb_path,
        provider_route=provider_route,
        capability_report_cid=capability_report_cid
        or _cid("capability-report", {"invocation": invocation.content_id}),
        resource_budget_cid=resource_budget_cid,
        lane_ceiling=lane_ceiling,
        merge_target=merge_target,
        worktree_strategy=worktree_strategy,
        validation_profile_cid=validation_profile_cid,
        coordination_shard=coordination_shard,
        replication=replication,
        configuration_root_cid=configuration_root_cid
        or _cid("configuration-root", {"invocation": invocation.content_id}),
        capability_catalog_cid=capability_catalog_cid
        or _cid("capability-catalog", {"invocation": invocation.content_id}),
        decisions=decision_tuple,
        unresolved_fields=unresolved,
        resolved_at_ms=resolved_at_ms,
        fresh_until_ms=fresh_until_ms,
        is_authorization=False,
    )


class SupervisorProfileResolver:
    """Compose leaf resolutions into one receipt and supervisor profile."""

    requirement_id: Final = RESOLVED_SUPERVISOR_PROFILE_REQUIREMENT_ID

    def resolve(self, request: ProfileCompositionRequest) -> ProfileResolution:
        if not isinstance(request, ProfileCompositionRequest):
            raise ProfileResolverError(
                "request must be a ProfileCompositionRequest"
            )

        if request.canonical_request is not None:
            return self._resolve_canonical(request)

        return self._resolve_inferred(request)

    def build_target_resolution_receipt(
        self,
        **values: Any,
    ) -> TargetResolutionReceipt:
        return build_target_resolution_receipt(**values)

    def _resolve_canonical(
        self, request: ProfileCompositionRequest
    ) -> ProfileResolution:
        canonical = request.canonical_request
        assert canonical is not None
        decisions = list(canonical.decisions)
        by_name = _decision_map(decisions)

        def value(name: str) -> str:
            return by_name[name].selected_value

        provider_route = (
            canonical.provider_route
            if canonical.provider_route is not None
            else request.capability.provider_route
        )
        resource_budget = (
            canonical.resource_budget
            if canonical.resource_budget is not None
            else request.capability.resources.resource_budget
        )
        coordination = (
            canonical.coordination_shard
            if canonical.coordination_shard is not None
            else request.capability.topology.coordination_shard
        )
        replication = (
            canonical.replication
            if canonical.replication is not None
            else request.capability.topology.replication
        )
        # Ensure structured projections match decisions.
        if value("provider") != provider_route.content_id:
            raise ProfileResolverError(
                "canonical provider decision does not match provider_route"
            )
        if value("resources") != resource_budget.content_id:
            raise ProfileResolverError(
                "canonical resources decision does not match resource_budget"
            )
        if value("coordination") != coordination.content_id:
            raise ProfileResolverError(
                "canonical coordination decision does not match binding"
            )
        if value("replication") != replication.content_id:
            raise ProfileResolverError(
                "canonical replication decision does not match binding"
            )
        if value("lane_ceiling") != str(
            int(value("lane_ceiling")) if value("lane_ceiling").isdigit() else -1
        ):
            # lane_ceiling decision is a decimal string.
            pass
        lane_ceiling = int(value("lane_ceiling"))
        output_mode = OutputMode(value("output"))
        worktree_strategy = WorktreeStrategy(value("worktree_strategy"))

        receipt = build_target_resolution_receipt(
            invocation=request.invocation,
            decisions=decisions,
            repository_root=value("repository_root"),
            repository_id=value("repository_id"),
            checkout_id=value("checkout_id"),
            scope_path=value("scope"),
            head_tree_cid=value("tree_id"),
            dirty_overlay_cid=value("dirty_overlay"),
            submodule_population_cid=value("submodules"),
            nested_repository_population_cid=value("nested_repositories"),
            state_root=value("state_root"),
            run_namespace=value("run_namespace"),
            objective_cid=value("objective"),
            objective_revision_cid=canonical.objective_revision_cid
            or _cid("canonical-objective-revision"),
            plan_cid=value("plan"),
            task_source_cid=value("task_source"),
            task_source_revision_cid=canonical.task_source_revision_cid
            or _cid("canonical-task-source-revision"),
            task_source_kind=canonical.task_source_kind,
            policy_cid=value("policy"),
            principal_ref=value("principal"),
            authority_source_ref=value("authority_source"),
            effect_ceiling_cid=value("effect_ceiling"),
            output_mode=output_mode,
            markdown_path=canonical.markdown_path,
            duckdb_path=canonical.duckdb_path,
            provider_route=provider_route,
            capability_report_cid=canonical.capability_report_cid,
            resource_budget_cid=resource_budget.content_id,
            lane_ceiling=lane_ceiling,
            merge_target=value("merge_target"),
            worktree_strategy=worktree_strategy,
            validation_profile_cid=value("validation"),
            coordination_shard=coordination,
            replication=replication,
            configuration_root_cid=canonical.configuration_root_cid,
            capability_catalog_cid=canonical.capability_catalog_cid,
            resolved_at_ms=request.resolved_at_ms,
            fresh_until_ms=request.fresh_until_ms,
        )

        expected_effects = canonical.expected_effects or default_effects_for_profile(
            canonical.profile_name
        )
        task_source_path = (
            canonical.task_source_path
            or canonical.duckdb_path
            or canonical.markdown_path
            or f"{value('state_root').rstrip('/')}/tasks.duckdb"
        )
        lifecycle_cid = (
            canonical.lifecycle_health_contract_cid
            or _lifecycle_health_cid(
                profile_name=canonical.profile_name,
                repository_id=value("repository_id"),
                state_root=value("state_root"),
                task_source_kind=canonical.task_source_kind,
            )
        )
        profile = ResolvedSupervisorProfile(
            profile_name=canonical.profile_name,
            profile_source_cid=_cid(
                "canonical-profile-source", {"request_cid": canonical.request_cid}
            ),
            target_resolution_receipt_cid=receipt.content_id,
            mode=canonical.mode,
            repository_root=value("repository_root"),
            state_root=value("state_root"),
            run_namespace=value("run_namespace"),
            policy_cid=value("policy"),
            principal_ref=value("principal"),
            effect_ceiling_cid=value("effect_ceiling"),
            task_source_kind=canonical.task_source_kind,
            task_source_path=task_source_path,
            task_source_cid=value("task_source"),
            output_mode=output_mode,
            markdown_path=canonical.markdown_path,
            duckdb_path=canonical.duckdb_path,
            provider_route=provider_route,
            resource_budget=resource_budget,
            validation_profile_cid=value("validation"),
            lifecycle_health_contract_cid=lifecycle_cid,
            coordination_shard=coordination,
            replication=replication,
            supervisor_argv=canonical.supervisor_argv
            or _compile_supervisor_argv(
                state_root=value("state_root"),
                run_namespace=value("run_namespace"),
                profile_name=canonical.profile_name,
            ),
            daemon_argv=canonical.daemon_argv
            or _compile_daemon_argv(
                task_source_path=task_source_path,
                state_root=value("state_root"),
            ),
            environment_names=canonical.environment_names
            or DEFAULT_ENVIRONMENT_NAMES,
            credential_handles=canonical.credential_handles
            or DEFAULT_CREDENTIAL_HANDLES,
            expected_effects=expected_effects,
            worktree_strategy=worktree_strategy,
            merge_target=value("merge_target"),
        )
        trace = tuple(
            PrecedenceTraceEntry(
                field_name=item.field_name,
                selected_source=item.selected_source,
                source_precedence=item.source_precedence,
                selected_value=item.selected_value,
                disposition=item.disposition,
                reason_codes=("canonical_request_disables_inference",),
            )
            for item in decisions
        )
        return ProfileResolution(
            requirement_id=self.requirement_id,
            receipt=receipt,
            profile=profile,
            decisions=tuple(decisions),
            precedence_trace=trace,
            profile_source_cid=profile.profile_source_cid,
            profile_name=canonical.profile_name,
            inference_disabled=True,
            effects_blocked=False,
            safe_preview=canonical.mode is InvocationMode.PREVIEW,
            cross_field_inconsistencies=(),
            reason_codes=("canonical_request_applied", "inference_disabled"),
            expected_effects=expected_effects,
            layers_applied=(),
        )

    def _resolve_inferred(
        self, request: ProfileCompositionRequest
    ) -> ProfileResolution:
        leaf = _leaf_decisions(request)
        layers = _select_profile_layers(request)
        (
            profile_name,
            mode,
            expected_effects,
            worktree_strategy,
            merge_target,
            lane_ceiling,
            profile_output_mode,
            environment_names,
            credential_handles,
            profile_source_cid,
            merge_reasons,
            merge_trace,
        ) = _merge_profile_configuration(
            layers,
            authority=request.authority,
            capability=request.capability,
            invocation=request.invocation,
        )
        if request.environment_names:
            environment_names = tuple(
                sorted(set(environment_names) & set(request.environment_names))
            ) or environment_names
        if request.credential_handles:
            credential_handles = tuple(
                sorted(set(credential_handles) & set(request.credential_handles))
            ) or credential_handles

        inconsistencies = _detect_cross_field_inconsistencies(request)
        reasons = list(merge_reasons)

        # Profile-owned decisions.
        evidence_cid = _cid(
            "profile-owned-fields",
            {
                "profile_name": profile_name,
                "merge_target": merge_target,
                "worktree_strategy": worktree_strategy.value,
            },
        )
        # Empty selected_value is invalid for selected dispositions.  Use a
        # closed sentinel for "no merge target" when empty.
        if merge_target:
            merge_source = next(
                (
                    layer.source
                    for layer in layers
                    if layer.merge_target == merge_target
                ),
                ResolutionSource.BUILTIN_DEFAULT,
            )
            merge_decision = _decision(
                field_name="merge_target",
                disposition=ResolutionDisposition.DEFAULTED,
                selected_value=merge_target,
                selected_source=merge_source,
                evidence_cid=evidence_cid,
                candidates=(
                    _candidate(
                        field_name="merge_target",
                        value=merge_target,
                        source=merge_source,
                        evidence_cid=evidence_cid,
                    ),
                ),
                reason_codes=(
                    "merge_target_identity_only",
                    "merge_effect_denied_by_default",
                ),
                effect=DecisionEffect.IDENTITY_ONLY,
            )
        else:
            merge_decision = _decision(
                field_name="merge_target",
                disposition=ResolutionDisposition.DEFAULTED,
                selected_value="none",
                selected_source=ResolutionSource.BUILTIN_DEFAULT,
                evidence_cid=evidence_cid,
                candidates=(
                    _candidate(
                        field_name="merge_target",
                        value="none",
                        source=ResolutionSource.BUILTIN_DEFAULT,
                        evidence_cid=evidence_cid,
                    ),
                ),
                reason_codes=(
                    "merge_target_unset",
                    "merge_effect_denied_by_default",
                ),
                effect=DecisionEffect.IDENTITY_ONLY,
            )

        worktree_source = next(
            (
                layer.source
                for layer in layers
                if layer.worktree_strategy is worktree_strategy
            ),
            ResolutionSource.BUILTIN_DEFAULT,
        )
        worktree_decision = _decision(
            field_name="worktree_strategy",
            disposition=ResolutionDisposition.DEFAULTED,
            selected_value=worktree_strategy.value,
            selected_source=worktree_source,
            evidence_cid=evidence_cid,
            candidates=(
                _candidate(
                    field_name="worktree_strategy",
                    value=worktree_strategy.value,
                    source=worktree_source,
                    evidence_cid=evidence_cid,
                ),
            ),
            reason_codes=("worktree_strategy_composed",),
            effect=DecisionEffect.CONFIGURATION,
        )

        decisions: dict[str, TargetInferenceDecision] = dict(leaf)
        decisions["merge_target"] = merge_decision
        decisions["worktree_strategy"] = worktree_decision

        # Optionally override output mode from profile layer when objective
        # defaulted and a stronger profile layer sets output.
        if (
            profile_output_mode is not None
            and decisions["output"].selected_source
            in {
                ResolutionSource.BUILTIN_DEFAULT,
                ResolutionSource.DISCOVERY,
                ResolutionSource.REPOSITORY_HINT,
            }
            and not decisions["output"].unresolved
        ):
            out_evidence = _cid(
                "profile-output-mode",
                {"output_mode": profile_output_mode.value},
            )
            decisions["output"] = _replace_decision_value(
                decisions["output"],
                selected_value=profile_output_mode.value,
                selected_source=ResolutionSource.SIGNED_PROFILE
                if any(
                    layer.output_mode is profile_output_mode
                    and layer.kind is ProfileSourceKind.SIGNED_PROFILE
                    for layer in layers
                )
                else ResolutionSource.BUILTIN_DEFAULT,
                evidence_cid=out_evidence,
                reason_codes=("output_mode_from_profile_layer",),
            )

        # Lane ceiling may be narrowed by profile layers.
        if (
            not decisions["lane_ceiling"].unresolved
            and decisions["lane_ceiling"].selected_value != str(lane_ceiling)
        ):
            decisions["lane_ceiling"] = _replace_decision_value(
                decisions["lane_ceiling"],
                selected_value=str(lane_ceiling),
                selected_source=ResolutionSource.SIGNED_PROFILE
                if any(layer.max_lanes == lane_ceiling for layer in layers)
                else ResolutionSource.DISCOVERY,
                evidence_cid=_cid(
                    "profile-lane-ceiling", {"lane_ceiling": lane_ceiling}
                ),
                reason_codes=("lane_ceiling_narrowed_by_profile",),
            )

        unresolved = {
            name for name, item in decisions.items() if item.unresolved
        }
        material_unresolved = unresolved & MATERIAL_EFFECT_FIELDS
        authority_denied = not request.authority.authorized
        mutation_effects = {
            ExpectedEffect.CREATE_ISOLATED_WORKTREE,
            ExpectedEffect.EDIT_ISOLATED_WORKTREE,
            ExpectedEffect.RUN_VALIDATION,
            ExpectedEffect.LAUNCH_LOCAL_PROCESS,
            ExpectedEffect.WRITE_SUPERVISOR_STATE,
        }
        has_mutation_effects = bool(set(expected_effects) & mutation_effects)
        # Mutation is blocked when identity/authority is incomplete, the mode is
        # preview, the effect set is inspect-only, or worktrees are disabled.
        effects_blocked = bool(
            material_unresolved
            or authority_denied
            or inconsistencies
            or mode is InvocationMode.PREVIEW
            or not has_mutation_effects
            or worktree_strategy is WorktreeStrategy.NONE
        )
        safe_preview = effects_blocked

        # Project leaf bindings.
        binding = request.repository.binding
        repository_root = binding.repository_root if binding is not None else ""
        repository_id = binding.repository_id if binding is not None else ""
        checkout_id = binding.checkout_id if binding is not None else ""
        scope_path = binding.scope_path if binding is not None else ""
        head_tree_cid = binding.tree_id if binding is not None else ""
        dirty_overlay_cid = binding.dirty_overlay_cid if binding is not None else ""
        submodule_population_cid = (
            binding.submodule_population_cid if binding is not None else ""
        )
        nested_repository_population_cid = (
            binding.nested_repository_population_cid
            if binding is not None
            else ""
        )
        # Unresolved repository fields must use empty projections.
        for field_name, attr in (
            ("repository_root", "repository_root"),
            ("repository_id", "repository_id"),
            ("checkout_id", "checkout_id"),
            ("scope", "scope_path"),
            ("tree_id", "head_tree_cid"),
            ("dirty_overlay", "dirty_overlay_cid"),
            ("submodules", "submodule_population_cid"),
            ("nested_repositories", "nested_repository_population_cid"),
        ):
            if field_name in unresolved:
                if field_name == "repository_root":
                    repository_root = ""
                elif field_name == "repository_id":
                    repository_id = ""
                elif field_name == "checkout_id":
                    checkout_id = ""
                elif field_name == "scope":
                    scope_path = ""
                elif field_name == "tree_id":
                    head_tree_cid = ""
                elif field_name == "dirty_overlay":
                    dirty_overlay_cid = ""
                elif field_name == "submodules":
                    submodule_population_cid = ""
                elif field_name == "nested_repositories":
                    nested_repository_population_cid = ""

        state_root = (
            ""
            if "state_root" in unresolved
            else request.state.state_root
        )
        run_namespace = (
            ""
            if "run_namespace" in unresolved
            else request.state.run_namespace
        )

        objective = request.objective.objective
        objective_cid = (
            ""
            if "objective" in unresolved or objective is None
            else objective.objective_cid
        )
        objective_revision_cid = (
            ""
            if objective is None
            else objective.objective_revision_cid
        )
        plan_cid = (
            ""
            if "plan" in unresolved or objective is None
            else objective.plan_cid
        )
        task_source = request.objective.task_source
        task_source_cid = (
            ""
            if "task_source" in unresolved or task_source is None
            else task_source.task_source_cid
        )
        task_source_revision_cid = (
            ""
            if task_source is None
            else task_source.task_source_revision_cid
        )
        task_source_kind = (
            task_source.kind if task_source is not None else TaskSourceKind.DUAL
        )
        task_source_path = (
            task_source.path
            if task_source is not None
            else (f"{state_root.rstrip('/')}/tasks.duckdb" if state_root else "")
        )

        output = request.objective.output
        if "output" in unresolved or output is None:
            output_mode = OutputMode.BOTH
            markdown_path = ""
            duckdb_path = ""
        else:
            output_mode = (
                OutputMode(decisions["output"].selected_value)
                if not decisions["output"].unresolved
                else output.output_mode
            )
            markdown_path = output.markdown_path
            duckdb_path = output.duckdb_path

        # Authority projections.
        if "policy" in unresolved or not request.authority.policy.selected:
            policy_cid = ""
        else:
            policy_cid = request.authority.policy.policy_cid
        if "principal" in unresolved or not request.authority.principal.bound:
            principal_ref = ""
        else:
            principal_ref = request.authority.principal.principal_ref
        if "authority_source" in unresolved:
            authority_source_ref = ""
        else:
            authority_source_ref = request.authority.authority_source_ref
        if "effect_ceiling" in unresolved:
            effect_ceiling_cid = ""
        else:
            effect_ceiling_cid = request.authority.effect_ceiling.ceiling_cid

        provider_route = request.capability.provider_route
        resource_budget = request.capability.resources.resource_budget
        # Receipt projects provider as the route content identity.  Leaf
        # capability decisions may select the provider token (e.g. "grok");
        # normalize to the route CID before composing the receipt.
        if (
            not decisions["provider"].unresolved
            and decisions["provider"].selected_value != provider_route.content_id
        ):
            decisions["provider"] = _replace_decision_value(
                decisions["provider"],
                selected_value=provider_route.content_id,
                selected_source=decisions["provider"].selected_source,
                evidence_cid=decisions["provider"].evidence_cid,
                reason_codes=("provider_route_content_identity",),
            )
        # If lane ceiling was narrowed, rebuild a capped budget when needed.
        if resource_budget.max_lanes > lane_ceiling:
            resource_budget = replace(resource_budget, max_lanes=lane_ceiling)
            decisions["resources"] = _replace_decision_value(
                decisions["resources"],
                selected_value=resource_budget.content_id,
                selected_source=decisions["resources"].selected_source,
                evidence_cid=_cid(
                    "resource-budget-narrowed",
                    {"max_lanes": lane_ceiling},
                ),
                reason_codes=("resource_budget_narrowed_with_lanes",),
            )
        # Lane ceiling decision must match the integer projection.
        if (
            not decisions["lane_ceiling"].unresolved
            and decisions["lane_ceiling"].selected_value != str(lane_ceiling)
        ):
            decisions["lane_ceiling"] = _replace_decision_value(
                decisions["lane_ceiling"],
                selected_value=str(lane_ceiling),
                selected_source=decisions["lane_ceiling"].selected_source,
                evidence_cid=_cid(
                    "lane-ceiling-aligned", {"lane_ceiling": lane_ceiling}
                ),
                reason_codes=("lane_ceiling_aligned",),
            )

        validation_profile_cid = request.capability.validation.profile_cid
        coordination = request.capability.topology.coordination_shard
        replication = request.capability.topology.replication

        if safe_preview:
            reasons.append("safe_preview_effects_blocked")
            if material_unresolved:
                reasons.append("material_field_unresolved")
            if authority_denied:
                reasons.append("authority_denied")
            if inconsistencies:
                reasons.append("cross_field_inconsistency")
            worktree_strategy = WorktreeStrategy.NONE
            decisions["worktree_strategy"] = _replace_decision_value(
                decisions["worktree_strategy"],
                selected_value=WorktreeStrategy.NONE.value,
                selected_source=ResolutionSource.BUILTIN_DEFAULT,
                evidence_cid=_cid("safe-preview-worktree"),
                reason_codes=("safe_preview_disables_worktree",),
            )
            if coordination.writable:
                coordination = _demote_coordination(coordination)
                decisions["coordination"] = _replace_decision_value(
                    decisions["coordination"],
                    selected_value=coordination.content_id,
                    selected_source=ResolutionSource.BUILTIN_DEFAULT,
                    evidence_cid=_cid("safe-preview-coordination"),
                    reason_codes=("safe_preview_disables_writable_coordination",),
                )
            if (
                replication.ipfs_publish
                or replication.pin
                or replication.mode is ReplicationMode.PARQUET_IPLD_IPFS
            ):
                replication = _demote_replication(replication)
                decisions["replication"] = _replace_decision_value(
                    decisions["replication"],
                    selected_value=replication.content_id,
                    selected_source=ResolutionSource.BUILTIN_DEFAULT,
                    evidence_cid=_cid("safe-preview-replication"),
                    reason_codes=("safe_preview_disables_ipfs_publication",),
                )
            expected_effects = _intersect_effects(
                expected_effects, PREVIEW_ALLOWED_EFFECTS
            ) or PREVIEW_ALLOWED_EFFECTS
            mode = InvocationMode.PREVIEW
            profile_name = BUILTIN_PROFILE_PREVIEW
            # Writable shard still requires principal == owner when writable;
            # after demotion writable is false so principal may be empty.
            if not principal_ref and coordination.writable:
                coordination = _demote_coordination(coordination)

        # Ensure coordination owner matches principal when writable.
        if (
            coordination.writable
            and principal_ref
            and coordination.owner_principal_ref != principal_ref
        ):
            coordination = replace(
                coordination, owner_principal_ref=principal_ref, writable=False
            )
            decisions["coordination"] = _replace_decision_value(
                decisions["coordination"],
                selected_value=coordination.content_id,
                selected_source=ResolutionSource.BUILTIN_DEFAULT,
                evidence_cid=_cid("coordination-owner-mismatch-demoted"),
                reason_codes=("coordination_owner_mismatch_demoted",),
            )
            reasons.append("coordination_owner_mismatch_demoted")

        # Output paths must exist for selected modes.
        if output_mode in {OutputMode.MARKDOWN, OutputMode.BOTH} and not markdown_path:
            if state_root:
                markdown_path = f"{state_root.rstrip('/')}/projections/tasks.md"
            else:
                # Fall back to duckdb-only if we cannot place markdown.
                if duckdb_path:
                    output_mode = OutputMode.DUCKDB
                    decisions["output"] = _replace_decision_value(
                        decisions["output"],
                        selected_value=output_mode.value,
                        selected_source=ResolutionSource.BUILTIN_DEFAULT,
                        evidence_cid=_cid("output-fallback-duckdb"),
                        reason_codes=("markdown_path_unavailable",),
                    )
        if output_mode in {OutputMode.DUCKDB, OutputMode.BOTH} and not duckdb_path:
            if state_root:
                duckdb_path = f"{state_root.rstrip('/')}/projections/tasks.duckdb"
            else:
                if markdown_path:
                    output_mode = OutputMode.MARKDOWN
                    decisions["output"] = _replace_decision_value(
                        decisions["output"],
                        selected_value=output_mode.value,
                        selected_source=ResolutionSource.BUILTIN_DEFAULT,
                        evidence_cid=_cid("output-fallback-markdown"),
                        reason_codes=("duckdb_path_unavailable",),
                    )

        # Align output decision selected value with final mode when resolved.
        if not decisions["output"].unresolved and decisions["output"].selected_value != output_mode.value:
            decisions["output"] = _replace_decision_value(
                decisions["output"],
                selected_value=output_mode.value,
                selected_source=decisions["output"].selected_source,
                evidence_cid=_cid("output-mode-aligned", {"mode": output_mode.value}),
                reason_codes=("output_mode_aligned",),
            )

        decision_tuple = tuple(
            decisions[name] for name in sorted(REQUIRED_TARGET_DECISION_FIELDS)
        )
        # Recompute unresolved after demotions.
        unresolved_fields = tuple(
            sorted(item.field_name for item in decision_tuple if item.unresolved)
        )

        # Closed sentinel "none" when no merge target is configured.  Receipt and
        # decision must match for every resolved field.
        receipt_merge_target = decisions["merge_target"].selected_value

        receipt = build_target_resolution_receipt(
            invocation=request.invocation,
            decisions=decision_tuple,
            repository_root=repository_root,
            repository_id=repository_id,
            checkout_id=checkout_id,
            scope_path=scope_path,
            head_tree_cid=head_tree_cid,
            dirty_overlay_cid=dirty_overlay_cid,
            submodule_population_cid=submodule_population_cid,
            nested_repository_population_cid=nested_repository_population_cid,
            state_root=state_root,
            run_namespace=run_namespace,
            objective_cid=objective_cid,
            objective_revision_cid=objective_revision_cid,
            plan_cid=plan_cid,
            task_source_cid=task_source_cid,
            task_source_revision_cid=task_source_revision_cid,
            task_source_kind=task_source_kind,
            policy_cid=policy_cid,
            principal_ref=principal_ref,
            authority_source_ref=authority_source_ref,
            effect_ceiling_cid=effect_ceiling_cid,
            output_mode=output_mode,
            markdown_path=markdown_path,
            duckdb_path=duckdb_path,
            provider_route=provider_route,
            capability_report_cid=request.capability_report_cid
            or request.capability.evidence_cid,
            resource_budget_cid=resource_budget.content_id,
            lane_ceiling=lane_ceiling,
            merge_target=receipt_merge_target,
            worktree_strategy=worktree_strategy,
            validation_profile_cid=validation_profile_cid,
            coordination_shard=coordination,
            replication=replication,
            configuration_root_cid=request.configuration_root_cid,
            capability_catalog_cid=request.capability_catalog_cid,
            resolved_at_ms=request.resolved_at_ms,
            fresh_until_ms=request.fresh_until_ms,
        )

        profile: ResolvedSupervisorProfile | None = None
        can_build_profile = bool(
            repository_root
            and state_root
            and run_namespace
            and principal_ref
            and policy_cid
            and effect_ceiling_cid
            and task_source_cid
            and task_source_path
        )
        # Safe preview may still build a profile when identity is complete,
        # even if effects are blocked (demoted worktree/coordination).
        if can_build_profile and "principal" not in unresolved_fields:
            lifecycle_cid = request.lifecycle_health_contract_cid or _lifecycle_health_cid(
                profile_name=profile_name,
                repository_id=repository_id,
                state_root=state_root,
                task_source_kind=task_source_kind,
            )
            # Preview profile mode when effects blocked.
            profile_mode = (
                InvocationMode.PREVIEW if safe_preview else mode
            )
            if safe_preview:
                profile_name = BUILTIN_PROFILE_PREVIEW
            profile = ResolvedSupervisorProfile(
                profile_name=profile_name,
                profile_source_cid=profile_source_cid,
                target_resolution_receipt_cid=receipt.content_id,
                mode=profile_mode,
                repository_root=repository_root,
                state_root=state_root,
                run_namespace=run_namespace,
                policy_cid=policy_cid,
                principal_ref=principal_ref,
                effect_ceiling_cid=effect_ceiling_cid,
                task_source_kind=task_source_kind,
                task_source_path=task_source_path,
                task_source_cid=task_source_cid,
                output_mode=output_mode,
                markdown_path=markdown_path,
                duckdb_path=duckdb_path,
                provider_route=provider_route,
                resource_budget=resource_budget,
                validation_profile_cid=validation_profile_cid,
                lifecycle_health_contract_cid=lifecycle_cid,
                coordination_shard=coordination,
                replication=replication,
                supervisor_argv=_compile_supervisor_argv(
                    state_root=state_root,
                    run_namespace=run_namespace,
                    profile_name=profile_name,
                ),
                daemon_argv=_compile_daemon_argv(
                    task_source_path=task_source_path,
                    state_root=state_root,
                ),
                environment_names=environment_names or DEFAULT_ENVIRONMENT_NAMES,
                credential_handles=credential_handles or DEFAULT_CREDENTIAL_HANDLES,
                expected_effects=expected_effects,
                worktree_strategy=worktree_strategy,
                merge_target=(
                    ""
                    if receipt_merge_target == "none"
                    else receipt_merge_target
                ),
            )
        else:
            reasons.append("profile_withheld_incomplete_identity")

        # Precedence trace for every required field.
        trace_entries: list[PrecedenceTraceEntry] = list(merge_trace)
        for item in decision_tuple:
            rejected = tuple(
                f"{cand.source.value}:{cand.value}"
                for cand in item.candidates
                if cand.rejection_reason
            )
            trace_entries.append(
                PrecedenceTraceEntry(
                    field_name=item.field_name,
                    selected_source=item.selected_source,
                    source_precedence=item.source_precedence,
                    selected_value=item.selected_value,
                    disposition=item.disposition,
                    rejected_sources=rejected,
                    reason_codes=item.reason_codes,
                )
            )

        return ProfileResolution(
            requirement_id=self.requirement_id,
            receipt=receipt,
            profile=profile,
            decisions=decision_tuple,
            precedence_trace=tuple(trace_entries),
            profile_source_cid=profile_source_cid,
            profile_name=profile_name
            if profile is not None
            else builtin_profile_for_mode(InvocationMode.PREVIEW),
            inference_disabled=False,
            effects_blocked=effects_blocked,
            safe_preview=safe_preview,
            cross_field_inconsistencies=inconsistencies,
            reason_codes=tuple(reasons),
            expected_effects=expected_effects,
            layers_applied=layers,
        )


def resolve_supervisor_profile(
    request: ProfileCompositionRequest,
) -> ProfileResolution:
    """Module-level convenience wrapper around :class:`SupervisorProfileResolver`."""

    return SupervisorProfileResolver().resolve(request)


def resolve_profile(
    request: ProfileCompositionRequest,
) -> ProfileResolution:
    """Alias for :func:`resolve_supervisor_profile`."""

    return resolve_supervisor_profile(request)


__all__ = [
    "AUTHORITY_FIELD_NAMES",
    "BUILTIN_PROFILE_CI_WORKER",
    "BUILTIN_PROFILE_LOCAL_WORKTREE",
    "BUILTIN_PROFILE_NAMES",
    "BUILTIN_PROFILE_PREVIEW",
    "CANONICAL_REQUEST_BINDING_SCHEMA",
    "CAPABILITY_FIELD_NAMES",
    "CanonicalRequestBinding",
    "DEFAULT_CREDENTIAL_HANDLES",
    "DEFAULT_ENVIRONMENT_NAMES",
    "MATERIAL_EFFECT_FIELDS",
    "OBJECTIVE_FIELD_NAMES",
    "PROFILE_COMPOSITION_REQUEST_SCHEMA",
    "PROFILE_OWNED_FIELDS",
    "PROFILE_RESOLUTION_SCHEMA",
    "PROFILE_SOURCE_LAYER_SCHEMA",
    "PRECEDENCE_TRACE_SCHEMA",
    "ProfileCompositionRequest",
    "ProfileResolution",
    "ProfileResolverError",
    "ProfileSourceKind",
    "ProfileSourceLayer",
    "PrecedenceTraceEntry",
    "REPOSITORY_FIELD_NAMES",
    "RESOLVED_SUPERVISOR_PROFILE_REQUIREMENT_ID",
    "SOURCE_PRECEDENCE",
    "STATE_FIELD_NAMES",
    "SupervisorProfileResolver",
    "build_target_resolution_receipt",
    "builtin_profile_for_mode",
    "default_effects_for_profile",
    "default_worktree_strategy_for_profile",
    "resolve_profile",
    "resolve_supervisor_profile",
    "source_for_kind",
]
