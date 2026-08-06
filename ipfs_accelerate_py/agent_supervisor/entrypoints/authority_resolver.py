"""Principal, policy, local-authority, and effect-ceiling resolution.

ASE-008 / ASE-G033: bind authenticated local or transport identity, select only
trusted policies/profiles, install explicit local worktree signing authority,
and derive exact maximum effects.  Prompt text, repository prose, usernames,
environment claims, and mere credential presence never create a caller or
authority.  Lower-precedence sources may only narrow allowlists and ceilings.

This module is provider-free: import and pure resolution perform no I/O,
repository scans, network calls, or process starts.  Callers inject verified
evidence; the resolver records decisions as entrypoint contracts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final

from ..core.multiformats_identity import cid_for_dag_json
from .contracts import (
    AUTHORITY_DECISION_FIELDS,
    TRUSTED_AUTHORITY_SOURCES,
    DecisionEffect,
    ExpectedEffect,
    InvocationMode,
    ResolutionDisposition,
    ResolutionSource,
    RevalidationRule,
    TargetCandidate,
    TargetInferenceDecision,
    WorktreeStrategy,
)

AUTHORITY_RESOLUTION_REQUIREMENT_ID: Final[str] = (
    "requirement:agent-supervisor.entrypoints.authority-resolution@1"
)

AUTHORITY_RESOLUTION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/entrypoints/authority-resolution@1"
)

LOCAL_WORKTREE_PROFILE_NAME: Final[str] = "local-worktree"
LOCAL_WORKTREE_AUTHORITY_SOURCE: Final[str] = (
    "authority:local-worktree-profile"
)
LOCAL_WORKTREE_POLICY_NAME: Final[str] = "policy:local-worktree@1"
PREVIEW_POLICY_NAME: Final[str] = "policy:preview@1"
MCP_TRANSPORT_AUTHORITY_SOURCE: Final[str] = "authority:authenticated-transport"
EXISTING_RUN_AUTHORITY_SOURCE: Final[str] = "authority:existing-run"
SIGNED_PROFILE_AUTHORITY_SOURCE: Final[str] = "authority:signed-profile"

# Higher numbers win.  Matches the plan precedence ladder.
SOURCE_PRECEDENCE: Final[Mapping[ResolutionSource, int]] = {
    ResolutionSource.CANONICAL_REQUEST: 100,
    ResolutionSource.EXPLICIT_OVERRIDE: 90,
    ResolutionSource.EXISTING_RUN: 80,
    ResolutionSource.AUTHENTICATED_TRANSPORT: 70,
    ResolutionSource.SIGNED_PROFILE: 60,
    ResolutionSource.REPOSITORY_HINT: 40,
    ResolutionSource.DISCOVERY: 30,
    ResolutionSource.BUILTIN_DEFAULT: 10,
}

# Safe local-worktree mutation surface after explicit setup.
LOCAL_WORKTREE_ALLOWED_EFFECTS: Final[tuple[ExpectedEffect, ...]] = (
    ExpectedEffect.INSPECT_REPOSITORY,
    ExpectedEffect.WRITE_SUPERVISOR_STATE,
    ExpectedEffect.CREATE_ISOLATED_WORKTREE,
    ExpectedEffect.EDIT_ISOLATED_WORKTREE,
    ExpectedEffect.RUN_VALIDATION,
    ExpectedEffect.LAUNCH_LOCAL_PROCESS,
)

# Always denied under local-worktree authority (and by default generally).
LOCAL_WORKTREE_DENIED_EFFECTS: Final[tuple[ExpectedEffect, ...]] = (
    ExpectedEffect.MERGE,
    ExpectedEffect.PUSH,
    ExpectedEffect.DEPLOY,
    ExpectedEffect.DESTRUCTIVE_CLEANUP,
)

PREVIEW_ALLOWED_EFFECTS: Final[tuple[ExpectedEffect, ...]] = (
    ExpectedEffect.INSPECT_REPOSITORY,
)

# Operations without a dedicated ExpectedEffect enum member, still fail-closed.
FORBIDDEN_LOCAL_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        "current_checkout_rewrite",
        "merge",
        "push",
        "deploy",
        "secrets_access",
        "arbitrary_network",
        "destructive_cleanup",
    }
)

ALL_EXPECTED_EFFECTS: Final[frozenset[ExpectedEffect]] = frozenset(
    ExpectedEffect
)


class AuthorityResolverError(ValueError):
    """Malformed, untrusted, or inconsistent authority-resolution input."""


class PrincipalSourceKind(str, Enum):
    """How a principal was authenticated (never prompt/repository text)."""

    LOCAL_OS_KEY = "local_os_key"
    LOCAL_WORKTREE_INSTALL = "local_worktree_install"
    MCP_TRANSPORT = "mcp_transport"
    MCP_PLUS_UCAN = "mcp_plus_ucan"
    EXISTING_RUN = "existing_run"
    SIGNED_PROFILE = "signed_profile"


def _require_text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise AuthorityResolverError(f"{name} must be text")
    if "\x00" in value:
        raise AuthorityResolverError(f"{name} contains a NUL byte")
    if required and not value.strip():
        raise AuthorityResolverError(f"{name} must not be empty")
    return value.strip() if isinstance(value, str) else value


def _require_cid(value: Any, name: str, *, required: bool = True) -> str:
    text = _require_text(value, name, required=required)
    if not text:
        return ""
    # Accept opaque content identities; full CID validation happens when the
    # value is embedded in TargetInferenceDecision.
    if (
        not text.startswith("baguqeer")
        and not text.startswith("bafy")
        and len(text) < 8
    ):
        raise AuthorityResolverError(f"{name} is not a content identity")
    return text


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise AuthorityResolverError(f"{name} must be a boolean")
    return value


def _require_nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuthorityResolverError(f"{name} must be an integer")
    if value < 0:
        raise AuthorityResolverError(f"{name} must be non-negative")
    return value


def _effects(
    value: Any,
    name: str,
    *,
    required: bool = False,
) -> tuple[ExpectedEffect, ...]:
    if value is None:
        if required:
            raise AuthorityResolverError(f"{name} is required")
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AuthorityResolverError(f"{name} must be a sequence of effects")
    items: list[ExpectedEffect] = []
    seen: set[ExpectedEffect] = set()
    for item in value:
        effect = (
            item
            if isinstance(item, ExpectedEffect)
            else ExpectedEffect(str(item))
        )
        if effect in seen:
            continue
        seen.add(effect)
        items.append(effect)
    return tuple(sorted(items, key=lambda item: item.value))


def _effect_set(effects: Iterable[ExpectedEffect]) -> frozenset[ExpectedEffect]:
    return frozenset(effects)


def _sorted_effects(
    effects: Iterable[ExpectedEffect],
) -> tuple[ExpectedEffect, ...]:
    return tuple(sorted(_effect_set(effects), key=lambda item: item.value))


def _intersect_effects(
    *layers: Iterable[ExpectedEffect] | None,
) -> tuple[ExpectedEffect, ...]:
    active = [layer for layer in layers if layer is not None]
    if not active:
        return ()
    result: frozenset[ExpectedEffect] | None = None
    for layer in active:
        current = _effect_set(layer)
        result = current if result is None else result & current
    return _sorted_effects(result or ())


def effect_ceiling_cid(
    allowed_effects: Sequence[ExpectedEffect],
    *,
    denied_effects: Sequence[ExpectedEffect] = (),
    forbidden_operations: Iterable[str] = (),
    worktree_strategy: WorktreeStrategy = WorktreeStrategy.ISOLATED,
    profile_name: str = "",
) -> str:
    """Content-address an exact effect ceiling."""

    payload = {
        "schema": f"{AUTHORITY_RESOLUTION_SCHEMA}/effect-ceiling@1",
        "allowed_effects": [item.value for item in _sorted_effects(allowed_effects)],
        "denied_effects": [item.value for item in _sorted_effects(denied_effects)],
        "forbidden_operations": sorted(set(forbidden_operations)),
        "worktree_strategy": worktree_strategy.value,
        "profile_name": profile_name,
    }
    return cid_for_dag_json(payload)


def policy_cid_for(name: str, *, revision: str = "1") -> str:
    """Deterministic content identity for a named built-in/trusted policy."""

    return cid_for_dag_json(
        {
            "schema": f"{AUTHORITY_RESOLUTION_SCHEMA}/policy@1",
            "name": _require_text(name, "policy_name"),
            "revision": _require_text(revision, "revision"),
        }
    )


def mode_default_effects(mode: InvocationMode) -> tuple[ExpectedEffect, ...]:
    """Conservative built-in ceilings by invocation mode (never authority)."""

    mode = mode if isinstance(mode, InvocationMode) else InvocationMode(mode)
    if mode is InvocationMode.PREVIEW:
        return PREVIEW_ALLOWED_EFFECTS
    if mode in {
        InvocationMode.WORKTREE,
        InvocationMode.CI_WORKER,
        InvocationMode.DISTRIBUTED_WORKER,
    }:
        return LOCAL_WORKTREE_ALLOWED_EFFECTS
    raise AuthorityResolverError(f"unsupported invocation mode: {mode!r}")


@dataclass(frozen=True)
class AuthenticatedPrincipalEvidence:
    """Verified transport or local principal; never prompt/username text."""

    principal_ref: str
    source: ResolutionSource
    evidence_cid: str
    kind: PrincipalSourceKind
    transport: str = ""
    audience: str = ""
    signature_verified: bool = True
    ucan_verified: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "principal_ref",
            _require_text(self.principal_ref, "principal_ref"),
        )
        source = (
            self.source
            if isinstance(self.source, ResolutionSource)
            else ResolutionSource(self.source)
        )
        if source not in TRUSTED_AUTHORITY_SOURCES:
            raise AuthorityResolverError(
                "authenticated principal source must be a trusted authority source"
            )
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self, "evidence_cid", _require_cid(self.evidence_cid, "evidence_cid")
        )
        kind = (
            self.kind
            if isinstance(self.kind, PrincipalSourceKind)
            else PrincipalSourceKind(self.kind)
        )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self, "transport", _require_text(self.transport, "transport", required=False)
        )
        object.__setattr__(
            self, "audience", _require_text(self.audience, "audience", required=False)
        )
        object.__setattr__(
            self,
            "signature_verified",
            _require_bool(self.signature_verified, "signature_verified"),
        )
        object.__setattr__(
            self,
            "ucan_verified",
            _require_bool(self.ucan_verified, "ucan_verified"),
        )
        if not self.signature_verified and not self.ucan_verified:
            raise AuthorityResolverError(
                "principal evidence requires signature or UCAN verification"
            )


@dataclass(frozen=True)
class SignedProfileEvidence:
    """A signed/content-addressed supervisor profile used as authority."""

    profile_name: str
    profile_cid: str
    policy_cid: str
    principal_ref: str
    authority_source_ref: str
    allowed_effects: tuple[ExpectedEffect, ...]
    evidence_cid: str
    signature_verified: bool = True
    worktree_strategy: WorktreeStrategy = WorktreeStrategy.ISOLATED

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "profile_name",
            _require_text(self.profile_name, "profile_name"),
        )
        for name in ("profile_cid", "policy_cid", "evidence_cid"):
            object.__setattr__(
                self, name, _require_cid(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "principal_ref",
            _require_text(self.principal_ref, "principal_ref"),
        )
        object.__setattr__(
            self,
            "authority_source_ref",
            _require_text(self.authority_source_ref, "authority_source_ref"),
        )
        object.__setattr__(
            self,
            "allowed_effects",
            _effects(self.allowed_effects, "allowed_effects", required=True),
        )
        object.__setattr__(
            self,
            "signature_verified",
            _require_bool(self.signature_verified, "signature_verified"),
        )
        strategy = (
            self.worktree_strategy
            if isinstance(self.worktree_strategy, WorktreeStrategy)
            else WorktreeStrategy(self.worktree_strategy)
        )
        object.__setattr__(self, "worktree_strategy", strategy)
        if not self.signature_verified:
            raise AuthorityResolverError(
                "unsigned or unverified profiles cannot supply authority"
            )


@dataclass(frozen=True)
class ExistingRunAuthorityEvidence:
    """Authority rebound from an already-admitted compatible run."""

    run_id: str
    principal_ref: str
    policy_cid: str
    authority_source_ref: str
    allowed_effects: tuple[ExpectedEffect, ...]
    evidence_cid: str
    effect_ceiling_cid: str = ""
    worktree_strategy: WorktreeStrategy = WorktreeStrategy.ISOLATED

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_text(self.run_id, "run_id"))
        object.__setattr__(
            self,
            "principal_ref",
            _require_text(self.principal_ref, "principal_ref"),
        )
        object.__setattr__(
            self, "policy_cid", _require_cid(self.policy_cid, "policy_cid")
        )
        object.__setattr__(
            self,
            "authority_source_ref",
            _require_text(self.authority_source_ref, "authority_source_ref"),
        )
        object.__setattr__(
            self,
            "allowed_effects",
            _effects(self.allowed_effects, "allowed_effects", required=True),
        )
        object.__setattr__(
            self, "evidence_cid", _require_cid(self.evidence_cid, "evidence_cid")
        )
        object.__setattr__(
            self,
            "effect_ceiling_cid",
            _require_cid(
                self.effect_ceiling_cid, "effect_ceiling_cid", required=False
            ),
        )
        strategy = (
            self.worktree_strategy
            if isinstance(self.worktree_strategy, WorktreeStrategy)
            else WorktreeStrategy(self.worktree_strategy)
        )
        object.__setattr__(self, "worktree_strategy", strategy)


@dataclass(frozen=True)
class RepositoryPolicyConstraint:
    """Repository policy is a constraint only; it never creates authority."""

    policy_cid: str
    evidence_cid: str
    allowed_effects: tuple[ExpectedEffect, ...] | None = None
    denied_effects: tuple[ExpectedEffect, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_cid", _require_cid(self.policy_cid, "policy_cid")
        )
        object.__setattr__(
            self, "evidence_cid", _require_cid(self.evidence_cid, "evidence_cid")
        )
        if self.allowed_effects is not None:
            object.__setattr__(
                self,
                "allowed_effects",
                _effects(self.allowed_effects, "allowed_effects"),
            )
        object.__setattr__(
            self,
            "denied_effects",
            _effects(self.denied_effects, "denied_effects"),
        )


@dataclass(frozen=True)
class LocalWorktreeAuthority:
    """Explicitly installed local signing authority for bounded worktree work.

    Presence of credentials or repository text does not create this record.
    Operators must call :func:`install_local_worktree_authority` (or supply an
    equivalent verified installation receipt) once; subsequent prompt-only
    invocations may reuse it without repeated flags.
    """

    principal_ref: str
    installation_receipt_cid: str
    signing_key_handle: str
    policy_cid: str
    allowed_effects: tuple[ExpectedEffect, ...] = LOCAL_WORKTREE_ALLOWED_EFFECTS
    denied_effects: tuple[ExpectedEffect, ...] = LOCAL_WORKTREE_DENIED_EFFECTS
    forbidden_operations: frozenset[str] = field(
        default_factory=lambda: FORBIDDEN_LOCAL_OPERATIONS
    )
    worktree_strategy: WorktreeStrategy = WorktreeStrategy.ISOLATED
    profile_name: str = LOCAL_WORKTREE_PROFILE_NAME
    authority_source_ref: str = LOCAL_WORKTREE_AUTHORITY_SOURCE
    installed_at_ms: int = 0
    verified: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "principal_ref",
            _require_text(self.principal_ref, "principal_ref"),
        )
        object.__setattr__(
            self,
            "installation_receipt_cid",
            _require_cid(
                self.installation_receipt_cid, "installation_receipt_cid"
            ),
        )
        object.__setattr__(
            self,
            "signing_key_handle",
            _require_text(self.signing_key_handle, "signing_key_handle"),
        )
        object.__setattr__(
            self, "policy_cid", _require_cid(self.policy_cid, "policy_cid")
        )
        object.__setattr__(
            self,
            "allowed_effects",
            _effects(self.allowed_effects, "allowed_effects", required=True),
        )
        object.__setattr__(
            self,
            "denied_effects",
            _effects(self.denied_effects, "denied_effects"),
        )
        ops = frozenset(
            _require_text(item, "forbidden_operations[]")
            for item in self.forbidden_operations
        )
        object.__setattr__(self, "forbidden_operations", ops)
        strategy = (
            self.worktree_strategy
            if isinstance(self.worktree_strategy, WorktreeStrategy)
            else WorktreeStrategy(self.worktree_strategy)
        )
        if strategy is WorktreeStrategy.CURRENT_CHECKOUT:
            raise AuthorityResolverError(
                "local worktree authority cannot rewrite the current checkout"
            )
        object.__setattr__(self, "worktree_strategy", strategy)
        object.__setattr__(
            self,
            "profile_name",
            _require_text(self.profile_name, "profile_name"),
        )
        object.__setattr__(
            self,
            "authority_source_ref",
            _require_text(self.authority_source_ref, "authority_source_ref"),
        )
        object.__setattr__(
            self,
            "installed_at_ms",
            _require_nonneg_int(self.installed_at_ms, "installed_at_ms"),
        )
        object.__setattr__(
            self, "verified", _require_bool(self.verified, "verified")
        )
        forbidden = set(self.allowed_effects) & set(LOCAL_WORKTREE_DENIED_EFFECTS)
        if forbidden:
            raise AuthorityResolverError(
                "local worktree authority cannot allow stronger effects: "
                + ", ".join(sorted(item.value for item in forbidden))
            )
        if ExpectedEffect.INSPECT_REPOSITORY not in self.allowed_effects:
            raise AuthorityResolverError(
                "local worktree authority must allow inspect_repository"
            )

    @property
    def evidence_cid(self) -> str:
        return self.installation_receipt_cid

    @property
    def effect_ceiling_cid(self) -> str:
        return effect_ceiling_cid(
            self.allowed_effects,
            denied_effects=self.denied_effects,
            forbidden_operations=self.forbidden_operations,
            worktree_strategy=self.worktree_strategy,
            profile_name=self.profile_name,
        )

    def permits(self, effect: ExpectedEffect | str) -> bool:
        effect = (
            effect
            if isinstance(effect, ExpectedEffect)
            else ExpectedEffect(str(effect))
        )
        return effect in self.allowed_effects and effect not in self.denied_effects

    def denies_operation(self, operation: str) -> bool:
        return operation in self.forbidden_operations

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": f"{AUTHORITY_RESOLUTION_SCHEMA}/local-worktree-authority@1",
            "principal_ref": self.principal_ref,
            "installation_receipt_cid": self.installation_receipt_cid,
            "signing_key_handle": self.signing_key_handle,
            "policy_cid": self.policy_cid,
            "allowed_effects": [item.value for item in self.allowed_effects],
            "denied_effects": [item.value for item in self.denied_effects],
            "forbidden_operations": sorted(self.forbidden_operations),
            "worktree_strategy": self.worktree_strategy.value,
            "profile_name": self.profile_name,
            "authority_source_ref": self.authority_source_ref,
            "installed_at_ms": self.installed_at_ms,
            "verified": self.verified,
            "effect_ceiling_cid": self.effect_ceiling_cid,
        }


def install_local_worktree_authority(
    principal_ref: str,
    *,
    signing_key_handle: str,
    installed_at_ms: int = 0,
    allowed_effects: Sequence[ExpectedEffect] | None = None,
) -> LocalWorktreeAuthority:
    """Explicit one-time setup for bounded local worktree authority.

    This is the only local path that creates mutation authority.  Credential
    presence, environment variables, and repository files do not substitute.
    """

    principal_ref = _require_text(principal_ref, "principal_ref")
    signing_key_handle = _require_text(signing_key_handle, "signing_key_handle")
    if allowed_effects is None:
        effects = LOCAL_WORKTREE_ALLOWED_EFFECTS
    else:
        effects = _effects(allowed_effects, "allowed_effects", required=True)
        stronger = set(effects) & set(LOCAL_WORKTREE_DENIED_EFFECTS)
        if stronger:
            raise AuthorityResolverError(
                "local worktree authority cannot allow stronger effects: "
                + ", ".join(sorted(item.value for item in stronger))
            )
        unknown = set(effects) - set(LOCAL_WORKTREE_ALLOWED_EFFECTS)
        if unknown:
            raise AuthorityResolverError(
                "local worktree authority cannot allow stronger effects: "
                + ", ".join(sorted(item.value for item in unknown))
            )
        # Explicit installs may only narrow the built-in local-worktree ceiling.
        effects = _intersect_effects(effects, LOCAL_WORKTREE_ALLOWED_EFFECTS)
    policy = policy_cid_for(LOCAL_WORKTREE_POLICY_NAME)
    receipt_cid = cid_for_dag_json(
        {
            "schema": f"{AUTHORITY_RESOLUTION_SCHEMA}/local-worktree-install@1",
            "principal_ref": principal_ref,
            "signing_key_handle": signing_key_handle,
            "policy_cid": policy,
            "allowed_effects": [item.value for item in effects],
            "denied_effects": [
                item.value for item in LOCAL_WORKTREE_DENIED_EFFECTS
            ],
            "forbidden_operations": sorted(FORBIDDEN_LOCAL_OPERATIONS),
            "worktree_strategy": WorktreeStrategy.ISOLATED.value,
            "installed_at_ms": _require_nonneg_int(
                installed_at_ms, "installed_at_ms"
            ),
            "requirement_id": AUTHORITY_RESOLUTION_REQUIREMENT_ID,
        }
    )
    return LocalWorktreeAuthority(
        principal_ref=principal_ref,
        installation_receipt_cid=receipt_cid,
        signing_key_handle=signing_key_handle,
        policy_cid=policy,
        allowed_effects=effects,
        denied_effects=LOCAL_WORKTREE_DENIED_EFFECTS,
        forbidden_operations=FORBIDDEN_LOCAL_OPERATIONS,
        worktree_strategy=WorktreeStrategy.ISOLATED,
        profile_name=LOCAL_WORKTREE_PROFILE_NAME,
        authority_source_ref=LOCAL_WORKTREE_AUTHORITY_SOURCE,
        installed_at_ms=installed_at_ms,
        verified=True,
    )


@dataclass(frozen=True)
class PrincipalBinding:
    """Resolved caller identity from a trusted source, or a typed denial."""

    disposition: ResolutionDisposition
    principal_ref: str
    source: ResolutionSource
    source_precedence: int
    evidence_cid: str
    kind: PrincipalSourceKind | None
    reason_codes: tuple[str, ...] = ()
    transport: str = ""

    @property
    def bound(self) -> bool:
        return self.disposition in {
            ResolutionDisposition.UNIQUE,
            ResolutionDisposition.DEFAULTED,
        } and bool(self.principal_ref)

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "principal_ref": self.principal_ref,
            "source": self.source.value,
            "source_precedence": self.source_precedence,
            "evidence_cid": self.evidence_cid,
            "kind": None if self.kind is None else self.kind.value,
            "reason_codes": list(self.reason_codes),
            "transport": self.transport,
            "bound": self.bound,
        }


@dataclass(frozen=True)
class PolicySelection:
    """Selected trusted policy plus any repository constraint CIDs."""

    disposition: ResolutionDisposition
    policy_cid: str
    source: ResolutionSource
    source_precedence: int
    evidence_cid: str
    is_authority: bool
    constraint_policy_cids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    @property
    def selected(self) -> bool:
        return self.disposition in {
            ResolutionDisposition.UNIQUE,
            ResolutionDisposition.DEFAULTED,
        } and bool(self.policy_cid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "policy_cid": self.policy_cid,
            "source": self.source.value,
            "source_precedence": self.source_precedence,
            "evidence_cid": self.evidence_cid,
            "is_authority": self.is_authority,
            "constraint_policy_cids": list(self.constraint_policy_cids),
            "reason_codes": list(self.reason_codes),
            "selected": self.selected,
        }


@dataclass(frozen=True)
class EffectCeiling:
    """Exact maximum effects an invocation may attempt."""

    disposition: ResolutionDisposition
    allowed_effects: tuple[ExpectedEffect, ...]
    denied_effects: tuple[ExpectedEffect, ...]
    forbidden_operations: frozenset[str]
    worktree_strategy: WorktreeStrategy
    source: ResolutionSource
    source_precedence: int
    evidence_cid: str
    ceiling_cid: str
    profile_name: str = ""
    reason_codes: tuple[str, ...] = ()

    def permits(self, effect: ExpectedEffect | str) -> bool:
        effect = (
            effect
            if isinstance(effect, ExpectedEffect)
            else ExpectedEffect(str(effect))
        )
        return (
            self.disposition
            in {ResolutionDisposition.UNIQUE, ResolutionDisposition.DEFAULTED}
            and effect in self.allowed_effects
            and effect not in self.denied_effects
        )

    def denies_operation(self, operation: str) -> bool:
        return operation in self.forbidden_operations

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition.value,
            "allowed_effects": [item.value for item in self.allowed_effects],
            "denied_effects": [item.value for item in self.denied_effects],
            "forbidden_operations": sorted(self.forbidden_operations),
            "worktree_strategy": self.worktree_strategy.value,
            "source": self.source.value,
            "source_precedence": self.source_precedence,
            "evidence_cid": self.evidence_cid,
            "ceiling_cid": self.ceiling_cid,
            "profile_name": self.profile_name,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class AuthorityResolution:
    """Complete authority-resolution result for one invocation."""

    requirement_id: str
    mode: InvocationMode
    principal: PrincipalBinding
    policy: PolicySelection
    authority_source_ref: str
    authority_source: ResolutionSource
    authority_source_precedence: int
    authority_source_evidence_cid: str
    authority_source_disposition: ResolutionDisposition
    effect_ceiling: EffectCeiling
    local_worktree: LocalWorktreeAuthority | None
    decisions: tuple[TargetInferenceDecision, ...]
    decision_reference_cid: str
    non_authoritative_claims_ignored: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()

    @property
    def authorized(self) -> bool:
        return (
            self.principal.bound
            and self.policy.selected
            and self.authority_source_disposition
            in {
                ResolutionDisposition.UNIQUE,
                ResolutionDisposition.DEFAULTED,
            }
            and self.effect_ceiling.disposition
            in {
                ResolutionDisposition.UNIQUE,
                ResolutionDisposition.DEFAULTED,
            }
            and bool(self.authority_source_ref)
        )

    def decision_for(self, field_name: str) -> TargetInferenceDecision:
        for decision in self.decisions:
            if decision.field_name == field_name:
                return decision
        raise KeyError(field_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": AUTHORITY_RESOLUTION_SCHEMA,
            "requirement_id": self.requirement_id,
            "mode": self.mode.value,
            "principal": self.principal.to_dict(),
            "policy": self.policy.to_dict(),
            "authority_source_ref": self.authority_source_ref,
            "authority_source": self.authority_source.value,
            "authority_source_precedence": self.authority_source_precedence,
            "authority_source_evidence_cid": self.authority_source_evidence_cid,
            "authority_source_disposition": (
                self.authority_source_disposition.value
            ),
            "effect_ceiling": self.effect_ceiling.to_dict(),
            "local_worktree": (
                None
                if self.local_worktree is None
                else self.local_worktree.to_dict()
            ),
            "decisions": [item.to_dict() for item in self.decisions],
            "decision_reference_cid": self.decision_reference_cid,
            "non_authoritative_claims_ignored": list(
                self.non_authoritative_claims_ignored
            ),
            "reason_codes": list(self.reason_codes),
            "authorized": self.authorized,
        }


@dataclass(frozen=True)
class AuthorityResolutionRequest:
    """Injected evidence for authority resolution (never raw prompt bodies)."""

    mode: InvocationMode = InvocationMode.WORKTREE
    authenticated_principal: AuthenticatedPrincipalEvidence | None = None
    signed_profile: SignedProfileEvidence | None = None
    existing_run: ExistingRunAuthorityEvidence | None = None
    local_worktree_authority: LocalWorktreeAuthority | None = None
    repository_policy_constraint: RepositoryPolicyConstraint | None = None
    requested_effect_narrowing: tuple[ExpectedEffect, ...] | None = None
    # Non-authoritative claims — recorded as ignored, never bind.
    prompt_claimed_principal: str = ""
    prompt_claimed_effects: tuple[ExpectedEffect, ...] = ()
    prompt_claimed_policy: str = ""
    username_claim: str = ""
    environment_principal_claim: str = ""
    credentials_present: bool = False
    repository_claimed_authority: str = ""
    fresh_until_ms: int = 0

    def __post_init__(self) -> None:
        mode = (
            self.mode
            if isinstance(self.mode, InvocationMode)
            else InvocationMode(self.mode)
        )
        object.__setattr__(self, "mode", mode)
        if self.requested_effect_narrowing is not None:
            object.__setattr__(
                self,
                "requested_effect_narrowing",
                _effects(
                    self.requested_effect_narrowing,
                    "requested_effect_narrowing",
                ),
            )
        object.__setattr__(
            self,
            "prompt_claimed_effects",
            _effects(self.prompt_claimed_effects, "prompt_claimed_effects"),
        )
        for name in (
            "prompt_claimed_principal",
            "prompt_claimed_policy",
            "username_claim",
            "environment_principal_claim",
            "repository_claimed_authority",
        ):
            object.__setattr__(
                self,
                name,
                _require_text(getattr(self, name), name, required=False),
            )
        object.__setattr__(
            self,
            "credentials_present",
            _require_bool(self.credentials_present, "credentials_present"),
        )
        object.__setattr__(
            self,
            "fresh_until_ms",
            _require_nonneg_int(self.fresh_until_ms, "fresh_until_ms"),
        )


def _empty_evidence_cid(label: str) -> str:
    return cid_for_dag_json(
        {
            "schema": f"{AUTHORITY_RESOLUTION_SCHEMA}/empty-evidence@1",
            "label": label,
            "requirement_id": AUTHORITY_RESOLUTION_REQUIREMENT_ID,
        }
    )


def _authority_decision(
    field_name: str,
    *,
    disposition: ResolutionDisposition,
    selected_value: str,
    source: ResolutionSource,
    evidence_cid: str,
    reason_codes: Sequence[str] = (),
    alternatives: Sequence[tuple[str, ResolutionSource, str, str]] = (),
    fresh_until_ms: int = 0,
) -> TargetInferenceDecision:
    if field_name not in AUTHORITY_DECISION_FIELDS:
        raise AuthorityResolverError(
            f"{field_name} is not an authority decision field"
        )
    precedence = SOURCE_PRECEDENCE[source]
    selected = disposition in {
        ResolutionDisposition.UNIQUE,
        ResolutionDisposition.DEFAULTED,
    }
    candidates: list[TargetCandidate] = []
    if selected:
        candidates.append(
            TargetCandidate(
                field_name=field_name,
                value=selected_value,
                source=source,
                source_precedence=precedence,
                evidence_cid=evidence_cid,
            )
        )
    for value, alt_source, alt_evidence, rejection in alternatives:
        candidates.append(
            TargetCandidate(
                field_name=field_name,
                value=value,
                source=alt_source,
                source_precedence=SOURCE_PRECEDENCE[alt_source],
                evidence_cid=alt_evidence,
                rejection_reason=rejection,
            )
        )
    if not selected and not candidates and alternatives:
        # Unresolved with only rejected candidates already handled above.
        pass
    revalidation = (
        RevalidationRule.IMMUTABLE
        if fresh_until_ms == 0
        else RevalidationRule.BEFORE_MUTATION
    )
    return TargetInferenceDecision(
        field_name=field_name,
        disposition=disposition,
        selected_value=selected_value if selected else "",
        selected_source=source,
        source_precedence=precedence,
        evidence_cid=evidence_cid,
        candidates=tuple(candidates),
        reason_codes=tuple(reason_codes),
        effect=DecisionEffect.REQUIRES_AUTHORITY,
        override_accepted=False,
        fresh_until_ms=fresh_until_ms if revalidation is not RevalidationRule.IMMUTABLE else 0,
        revalidation_rule=revalidation,
    )


@dataclass(frozen=True)
class _TrustedAuthorityLayer:
    source: ResolutionSource
    principal_ref: str
    policy_cid: str
    authority_source_ref: str
    allowed_effects: tuple[ExpectedEffect, ...]
    evidence_cid: str
    kind: PrincipalSourceKind
    worktree_strategy: WorktreeStrategy
    transport: str = ""
    profile_name: str = ""
    forbidden_operations: frozenset[str] = field(
        default_factory=lambda: FORBIDDEN_LOCAL_OPERATIONS
    )


def _collect_trusted_layers(
    request: AuthorityResolutionRequest,
) -> list[_TrustedAuthorityLayer]:
    layers: list[_TrustedAuthorityLayer] = []

    if request.existing_run is not None:
        run = request.existing_run
        layers.append(
            _TrustedAuthorityLayer(
                source=ResolutionSource.EXISTING_RUN,
                principal_ref=run.principal_ref,
                policy_cid=run.policy_cid,
                authority_source_ref=run.authority_source_ref
                or EXISTING_RUN_AUTHORITY_SOURCE,
                allowed_effects=run.allowed_effects,
                evidence_cid=run.evidence_cid,
                kind=PrincipalSourceKind.EXISTING_RUN,
                worktree_strategy=run.worktree_strategy,
                profile_name="",
            )
        )

    if request.authenticated_principal is not None:
        principal = request.authenticated_principal
        kind = principal.kind
        if kind is PrincipalSourceKind.MCP_PLUS_UCAN and not principal.ucan_verified:
            raise AuthorityResolverError(
                "MCP++ principal evidence requires verified UCAN attenuation"
            )
        # Transport alone does not grant a mutation ceiling; it only binds
        # identity.  Effects come from an accompanying signed profile, existing
        # run, local worktree install, or mode-default after identity binds
        # and are intersected later.
        layers.append(
            _TrustedAuthorityLayer(
                source=ResolutionSource.AUTHENTICATED_TRANSPORT,
                principal_ref=principal.principal_ref,
                policy_cid="",
                authority_source_ref=MCP_TRANSPORT_AUTHORITY_SOURCE,
                allowed_effects=(),
                evidence_cid=principal.evidence_cid,
                kind=kind,
                worktree_strategy=WorktreeStrategy.ISOLATED,
                transport=principal.transport,
            )
        )

    if request.signed_profile is not None:
        profile = request.signed_profile
        layers.append(
            _TrustedAuthorityLayer(
                source=ResolutionSource.SIGNED_PROFILE,
                principal_ref=profile.principal_ref,
                policy_cid=profile.policy_cid,
                authority_source_ref=profile.authority_source_ref
                or SIGNED_PROFILE_AUTHORITY_SOURCE,
                allowed_effects=profile.allowed_effects,
                evidence_cid=profile.evidence_cid,
                kind=PrincipalSourceKind.SIGNED_PROFILE,
                worktree_strategy=profile.worktree_strategy,
                profile_name=profile.profile_name,
            )
        )

    if request.local_worktree_authority is not None:
        local = request.local_worktree_authority
        if not local.verified:
            raise AuthorityResolverError(
                "local worktree authority installation is not verified"
            )
        layers.append(
            _TrustedAuthorityLayer(
                source=ResolutionSource.SIGNED_PROFILE,
                principal_ref=local.principal_ref,
                policy_cid=local.policy_cid,
                authority_source_ref=local.authority_source_ref,
                allowed_effects=local.allowed_effects,
                evidence_cid=local.installation_receipt_cid,
                kind=PrincipalSourceKind.LOCAL_WORKTREE_INSTALL,
                worktree_strategy=local.worktree_strategy,
                profile_name=local.profile_name,
                forbidden_operations=local.forbidden_operations,
            )
        )

    layers.sort(
        key=lambda item: (-SOURCE_PRECEDENCE[item.source], item.evidence_cid)
    )
    return layers


def _ignored_non_authoritative_claims(
    request: AuthorityResolutionRequest,
) -> tuple[str, ...]:
    ignored: list[str] = []
    if request.prompt_claimed_principal:
        ignored.append("prompt_claimed_principal")
    if request.prompt_claimed_effects:
        ignored.append("prompt_claimed_effects")
    if request.prompt_claimed_policy:
        ignored.append("prompt_claimed_policy")
    if request.username_claim:
        ignored.append("username_claim")
    if request.environment_principal_claim:
        ignored.append("environment_principal_claim")
    if request.credentials_present:
        ignored.append("credentials_present")
    if request.repository_claimed_authority:
        ignored.append("repository_claimed_authority")
    if request.repository_policy_constraint is not None:
        # Present as a constraint, not as authority creation.
        ignored.append("repository_policy_as_authority")
    return tuple(ignored)


def _denied_resolution(
    request: AuthorityResolutionRequest,
    *,
    reasons: Sequence[str],
    ignored: Sequence[str],
) -> AuthorityResolution:
    empty = _empty_evidence_cid("denied")
    source = ResolutionSource.AUTHENTICATED_TRANSPORT
    principal = PrincipalBinding(
        disposition=ResolutionDisposition.DENIED,
        principal_ref="",
        source=source,
        source_precedence=SOURCE_PRECEDENCE[source],
        evidence_cid=empty,
        kind=None,
        reason_codes=tuple(reasons),
    )
    policy = PolicySelection(
        disposition=ResolutionDisposition.DENIED,
        policy_cid="",
        source=source,
        source_precedence=SOURCE_PRECEDENCE[source],
        evidence_cid=empty,
        is_authority=False,
        reason_codes=tuple(reasons),
    )
    ceiling = EffectCeiling(
        disposition=ResolutionDisposition.DENIED,
        allowed_effects=(),
        denied_effects=_sorted_effects(ALL_EXPECTED_EFFECTS),
        forbidden_operations=FORBIDDEN_LOCAL_OPERATIONS,
        worktree_strategy=WorktreeStrategy.NONE,
        source=source,
        source_precedence=SOURCE_PRECEDENCE[source],
        evidence_cid=empty,
        ceiling_cid=effect_ceiling_cid(
            (),
            denied_effects=ALL_EXPECTED_EFFECTS,
            forbidden_operations=FORBIDDEN_LOCAL_OPERATIONS,
            worktree_strategy=WorktreeStrategy.NONE,
        ),
        reason_codes=tuple(reasons),
    )
    decisions = (
        _authority_decision(
            "principal",
            disposition=ResolutionDisposition.DENIED,
            selected_value="",
            source=source,
            evidence_cid=empty,
            reason_codes=reasons,
            fresh_until_ms=request.fresh_until_ms,
        ),
        _authority_decision(
            "policy",
            disposition=ResolutionDisposition.DENIED,
            selected_value="",
            source=source,
            evidence_cid=empty,
            reason_codes=reasons,
            fresh_until_ms=request.fresh_until_ms,
        ),
        _authority_decision(
            "authority_source",
            disposition=ResolutionDisposition.DENIED,
            selected_value="",
            source=source,
            evidence_cid=empty,
            reason_codes=reasons,
            fresh_until_ms=request.fresh_until_ms,
        ),
        _authority_decision(
            "effect_ceiling",
            disposition=ResolutionDisposition.DENIED,
            selected_value="",
            source=source,
            evidence_cid=empty,
            reason_codes=reasons,
            fresh_until_ms=request.fresh_until_ms,
        ),
    )
    reference = cid_for_dag_json(
        {
            "schema": f"{AUTHORITY_RESOLUTION_SCHEMA}/decision-reference@1",
            "requirement_id": AUTHORITY_RESOLUTION_REQUIREMENT_ID,
            "authorized": False,
            "reason_codes": list(reasons),
            "decision_cids": [item.content_id for item in decisions],
        }
    )
    return AuthorityResolution(
        requirement_id=AUTHORITY_RESOLUTION_REQUIREMENT_ID,
        mode=request.mode,
        principal=principal,
        policy=policy,
        authority_source_ref="",
        authority_source=source,
        authority_source_precedence=SOURCE_PRECEDENCE[source],
        authority_source_evidence_cid=empty,
        authority_source_disposition=ResolutionDisposition.DENIED,
        effect_ceiling=ceiling,
        local_worktree=None,
        decisions=decisions,
        decision_reference_cid=reference,
        non_authoritative_claims_ignored=tuple(ignored),
        reason_codes=tuple(reasons),
    )


def resolve_authority(
    request: AuthorityResolutionRequest | None = None,
    **values: Any,
) -> AuthorityResolution:
    """Resolve principal, policy, authority source, and effect ceiling.

    Returns a typed denial when no trusted principal/authority evidence is
    present.  Non-authoritative claims are listed but never bind.
    """

    if request is None:
        request = AuthorityResolutionRequest(**values)
    elif values:
        raise AuthorityResolverError(
            "pass either an AuthorityResolutionRequest or keyword fields"
        )

    ignored = _ignored_non_authoritative_claims(request)
    layers = _collect_trusted_layers(request)
    if not layers:
        reasons = ["no_trusted_principal_evidence"]
        if request.credentials_present:
            reasons.append("credentials_presence_is_not_authority")
        if request.prompt_claimed_principal or request.username_claim:
            reasons.append("prompt_or_username_cannot_create_caller")
        if request.repository_claimed_authority:
            reasons.append("repository_text_cannot_create_authority")
        return _denied_resolution(
            request, reasons=tuple(reasons), ignored=ignored
        )

    # Principal: highest-precedence trusted layer wins; conflicting principals
    # at the same precedence fail closed.
    top_rank = SOURCE_PRECEDENCE[layers[0].source]
    top = [layer for layer in layers if SOURCE_PRECEDENCE[layer.source] == top_rank]
    principal_refs = {layer.principal_ref for layer in top}
    if len(principal_refs) != 1:
        return _denied_resolution(
            request,
            reasons=("ambiguous_trusted_principal",),
            ignored=ignored,
        )
    principal_layer = top[0]

    # All trusted layers that name a principal must agree when present; lower
    # layers may only restate the same principal (cannot introduce another).
    for layer in layers[1:]:
        if layer.principal_ref != principal_layer.principal_ref:
            return _denied_resolution(
                request,
                reasons=("conflicting_trusted_principals",),
                ignored=ignored,
            )

    principal = PrincipalBinding(
        disposition=ResolutionDisposition.UNIQUE,
        principal_ref=principal_layer.principal_ref,
        source=principal_layer.source,
        source_precedence=SOURCE_PRECEDENCE[principal_layer.source],
        evidence_cid=principal_layer.evidence_cid,
        kind=principal_layer.kind,
        transport=principal_layer.transport,
        reason_codes=("trusted_principal_bound",),
    )

    # Policy: first trusted layer with a policy CID; repository constraint
    # never becomes the authority policy.
    policy_layer = next((layer for layer in layers if layer.policy_cid), None)
    if policy_layer is None:
        # Transport-only identity with no profile/run/local authority cannot
        # invent a policy.  Preview mode may use a built-in preview policy only
        # when a trusted principal is already bound and effects stay inspect-only.
        if request.mode is InvocationMode.PREVIEW:
            policy_cid = policy_cid_for(PREVIEW_POLICY_NAME)
            policy_source = principal_layer.source
            policy_evidence = principal_layer.evidence_cid
            policy_reasons = ("preview_policy_under_trusted_principal",)
        else:
            return _denied_resolution(
                request,
                reasons=("no_trusted_policy_evidence",),
                ignored=ignored,
            )
    else:
        policy_cid = policy_layer.policy_cid
        policy_source = policy_layer.source
        policy_evidence = policy_layer.evidence_cid
        policy_reasons = ("trusted_policy_selected",)

    constraint_cids: tuple[str, ...] = ()
    if request.repository_policy_constraint is not None:
        constraint_cids = (request.repository_policy_constraint.policy_cid,)

    policy = PolicySelection(
        disposition=ResolutionDisposition.UNIQUE,
        policy_cid=policy_cid,
        source=policy_source,
        source_precedence=SOURCE_PRECEDENCE[policy_source],
        evidence_cid=policy_evidence,
        is_authority=True,
        constraint_policy_cids=constraint_cids,
        reason_codes=policy_reasons,
    )

    # Authority source reference: prefer the highest-precedence layer that
    # actually supplies mutation/profile authority (not transport-only).
    authority_layer = next(
        (
            layer
            for layer in layers
            if layer.allowed_effects or layer.policy_cid
        ),
        principal_layer,
    )
    authority_source_ref = authority_layer.authority_source_ref
    authority_source = authority_layer.source
    authority_evidence = authority_layer.evidence_cid

    # Effect ceiling: start from mode defaults, then intersect every trusted
    # layer that contributes an effect set.  Empty transport-only layers do not
    # widen.  Repository constraints and requested narrowings only subtract.
    # Prompt-claimed effects are ignored (never added).
    mode_ceiling = mode_default_effects(request.mode)
    contributing = [
        layer.allowed_effects for layer in layers if layer.allowed_effects
    ]
    if not contributing:
        # Trusted principal without an effect grant: preview may inspect only;
        # mutation modes remain denied for the ceiling field.
        if request.mode is InvocationMode.PREVIEW:
            contributing = [PREVIEW_ALLOWED_EFFECTS]
        else:
            return _denied_resolution(
                request,
                reasons=("no_trusted_effect_grant",),
                ignored=ignored,
            )

    allowed = _intersect_effects(mode_ceiling, *contributing)
    if request.repository_policy_constraint is not None:
        constraint = request.repository_policy_constraint
        if constraint.allowed_effects is not None:
            allowed = _intersect_effects(allowed, constraint.allowed_effects)
        if constraint.denied_effects:
            allowed = _sorted_effects(
                effect
                for effect in allowed
                if effect not in constraint.denied_effects
            )
    if request.requested_effect_narrowing is not None:
        allowed = _intersect_effects(allowed, request.requested_effect_narrowing)

    # Hard deny stronger effects under local-worktree-class authority.
    denied = _sorted_effects(
        set(LOCAL_WORKTREE_DENIED_EFFECTS) | (ALL_EXPECTED_EFFECTS - set(allowed))
    )
    # Local worktree install may contribute extra forbidden operations.
    forbidden: set[str] = set(FORBIDDEN_LOCAL_OPERATIONS)
    strategy = WorktreeStrategy.ISOLATED
    profile_name = ""
    for layer in layers:
        forbidden |= set(layer.forbidden_operations)
        if layer.worktree_strategy is WorktreeStrategy.CURRENT_CHECKOUT:
            return _denied_resolution(
                request,
                reasons=("current_checkout_rewrite_denied",),
                ignored=ignored,
            )
        if layer.profile_name:
            profile_name = layer.profile_name
        strategy = layer.worktree_strategy

    if strategy is WorktreeStrategy.CURRENT_CHECKOUT:
        return _denied_resolution(
            request,
            reasons=("current_checkout_rewrite_denied",),
            ignored=ignored,
        )

    if not allowed:
        return _denied_resolution(
            request,
            reasons=("effect_ceiling_empty_after_narrowing",),
            ignored=ignored,
        )

    ceiling_cid = effect_ceiling_cid(
        allowed,
        denied_effects=denied,
        forbidden_operations=forbidden,
        worktree_strategy=strategy,
        profile_name=profile_name,
    )
    ceiling = EffectCeiling(
        disposition=ResolutionDisposition.UNIQUE,
        allowed_effects=allowed,
        denied_effects=denied,
        forbidden_operations=frozenset(forbidden),
        worktree_strategy=strategy,
        source=authority_source,
        source_precedence=SOURCE_PRECEDENCE[authority_source],
        evidence_cid=authority_evidence,
        ceiling_cid=ceiling_cid,
        profile_name=profile_name,
        reason_codes=("effect_ceiling_derived", "lower_sources_only_narrow"),
    )

    # Rejected alternatives document non-authoritative forgery attempts.
    principal_alts: list[tuple[str, ResolutionSource, str, str]] = []
    if request.prompt_claimed_principal:
        principal_alts.append(
            (
                request.prompt_claimed_principal
                if request.prompt_claimed_principal.startswith("did:")
                or request.prompt_claimed_principal.startswith("principal:")
                else "principal:prompt-claim",
                ResolutionSource.REPOSITORY_HINT,
                _empty_evidence_cid("prompt-principal-claim"),
                "prompt_text_cannot_create_caller",
            )
        )
    if request.username_claim:
        principal_alts.append(
            (
                f"principal:username-{request.username_claim}",
                ResolutionSource.DISCOVERY,
                _empty_evidence_cid("username-claim"),
                "username_is_not_authenticated_principal",
            )
        )
    if request.environment_principal_claim:
        principal_alts.append(
            (
                request.environment_principal_claim
                if ":" in request.environment_principal_claim
                else f"principal:env-{request.environment_principal_claim}",
                ResolutionSource.DISCOVERY,
                _empty_evidence_cid("environment-principal-claim"),
                "environment_claim_is_not_authority",
            )
        )

    policy_alts: list[tuple[str, ResolutionSource, str, str]] = []
    if request.repository_policy_constraint is not None:
        policy_alts.append(
            (
                request.repository_policy_constraint.policy_cid,
                ResolutionSource.REPOSITORY_HINT,
                request.repository_policy_constraint.evidence_cid,
                "repository_policy_is_constraint_not_authority",
            )
        )
    if request.prompt_claimed_policy:
        claimed = request.prompt_claimed_policy
        policy_alts.append(
            (
                claimed if claimed.startswith("baguqeer") else policy_cid_for(claimed),
                ResolutionSource.REPOSITORY_HINT,
                _empty_evidence_cid("prompt-policy-claim"),
                "prompt_cannot_select_policy",
            )
        )

    effect_alts: list[tuple[str, ResolutionSource, str, str]] = []
    if request.prompt_claimed_effects:
        forged_cid = effect_ceiling_cid(
            request.prompt_claimed_effects,
            profile_name="prompt-forgery",
        )
        effect_alts.append(
            (
                forged_cid,
                ResolutionSource.REPOSITORY_HINT,
                _empty_evidence_cid("prompt-effect-claim"),
                "prompt_cannot_widen_effect_ceiling",
            )
        )

    authority_alts: list[tuple[str, ResolutionSource, str, str]] = []
    if request.repository_claimed_authority:
        claimed_authority = request.repository_claimed_authority
        if ":" not in claimed_authority:
            claimed_authority = f"authority:{claimed_authority}"
        authority_alts.append(
            (
                claimed_authority,
                ResolutionSource.REPOSITORY_HINT,
                _empty_evidence_cid("repository-authority-claim"),
                "repository_text_cannot_create_authority",
            )
        )

    decisions = (
        _authority_decision(
            "principal",
            disposition=ResolutionDisposition.UNIQUE,
            selected_value=principal.principal_ref,
            source=principal.source,
            evidence_cid=principal.evidence_cid,
            reason_codes=principal.reason_codes,
            alternatives=principal_alts,
            fresh_until_ms=request.fresh_until_ms,
        ),
        _authority_decision(
            "policy",
            disposition=ResolutionDisposition.UNIQUE,
            selected_value=policy.policy_cid,
            source=policy.source,
            evidence_cid=policy.evidence_cid,
            reason_codes=policy.reason_codes,
            alternatives=policy_alts,
            fresh_until_ms=request.fresh_until_ms,
        ),
        _authority_decision(
            "authority_source",
            disposition=ResolutionDisposition.UNIQUE,
            selected_value=authority_source_ref,
            source=authority_source,
            evidence_cid=authority_evidence,
            reason_codes=("trusted_authority_source_selected",),
            alternatives=authority_alts,
            fresh_until_ms=request.fresh_until_ms,
        ),
        _authority_decision(
            "effect_ceiling",
            disposition=ResolutionDisposition.UNIQUE,
            selected_value=ceiling_cid,
            source=authority_source,
            evidence_cid=authority_evidence,
            reason_codes=ceiling.reason_codes,
            alternatives=effect_alts,
            fresh_until_ms=request.fresh_until_ms,
        ),
    )

    reference = cid_for_dag_json(
        {
            "schema": f"{AUTHORITY_RESOLUTION_SCHEMA}/decision-reference@1",
            "requirement_id": AUTHORITY_RESOLUTION_REQUIREMENT_ID,
            "authorized": True,
            "principal_ref": principal.principal_ref,
            "policy_cid": policy.policy_cid,
            "authority_source_ref": authority_source_ref,
            "effect_ceiling_cid": ceiling_cid,
            "decision_cids": [item.content_id for item in decisions],
            "non_authoritative_claims_ignored": list(ignored),
        }
    )

    local = request.local_worktree_authority
    return AuthorityResolution(
        requirement_id=AUTHORITY_RESOLUTION_REQUIREMENT_ID,
        mode=request.mode,
        principal=principal,
        policy=policy,
        authority_source_ref=authority_source_ref,
        authority_source=authority_source,
        authority_source_precedence=SOURCE_PRECEDENCE[authority_source],
        authority_source_evidence_cid=authority_evidence,
        authority_source_disposition=ResolutionDisposition.UNIQUE,
        effect_ceiling=ceiling,
        local_worktree=local,
        decisions=decisions,
        decision_reference_cid=reference,
        non_authoritative_claims_ignored=tuple(ignored),
        reason_codes=("authority_resolved",),
    )


class AuthorityResolver:
    """Stateful facade that reuses an installed local worktree authority."""

    def __init__(
        self,
        *,
        local_worktree_authority: LocalWorktreeAuthority | None = None,
    ) -> None:
        self._local_worktree_authority = local_worktree_authority

    @property
    def local_worktree_authority(self) -> LocalWorktreeAuthority | None:
        return self._local_worktree_authority

    def install_local_worktree(
        self,
        principal_ref: str,
        *,
        signing_key_handle: str,
        installed_at_ms: int = 0,
        allowed_effects: Sequence[ExpectedEffect] | None = None,
    ) -> LocalWorktreeAuthority:
        authority = install_local_worktree_authority(
            principal_ref,
            signing_key_handle=signing_key_handle,
            installed_at_ms=installed_at_ms,
            allowed_effects=allowed_effects,
        )
        self._local_worktree_authority = authority
        return authority

    def resolve(
        self,
        request: AuthorityResolutionRequest | None = None,
        **values: Any,
    ) -> AuthorityResolution:
        if request is None:
            request = AuthorityResolutionRequest(**values)
        if (
            request.local_worktree_authority is None
            and self._local_worktree_authority is not None
        ):
            request = AuthorityResolutionRequest(
                mode=request.mode,
                authenticated_principal=request.authenticated_principal,
                signed_profile=request.signed_profile,
                existing_run=request.existing_run,
                local_worktree_authority=self._local_worktree_authority,
                repository_policy_constraint=request.repository_policy_constraint,
                requested_effect_narrowing=request.requested_effect_narrowing,
                prompt_claimed_principal=request.prompt_claimed_principal,
                prompt_claimed_effects=request.prompt_claimed_effects,
                prompt_claimed_policy=request.prompt_claimed_policy,
                username_claim=request.username_claim,
                environment_principal_claim=request.environment_principal_claim,
                credentials_present=request.credentials_present,
                repository_claimed_authority=request.repository_claimed_authority,
                fresh_until_ms=request.fresh_until_ms,
            )
        return resolve_authority(request)


__all__ = (
    "AUTHORITY_RESOLUTION_REQUIREMENT_ID",
    "AUTHORITY_RESOLUTION_SCHEMA",
    "EXISTING_RUN_AUTHORITY_SOURCE",
    "FORBIDDEN_LOCAL_OPERATIONS",
    "LOCAL_WORKTREE_ALLOWED_EFFECTS",
    "LOCAL_WORKTREE_AUTHORITY_SOURCE",
    "LOCAL_WORKTREE_DENIED_EFFECTS",
    "LOCAL_WORKTREE_POLICY_NAME",
    "LOCAL_WORKTREE_PROFILE_NAME",
    "MCP_TRANSPORT_AUTHORITY_SOURCE",
    "PREVIEW_ALLOWED_EFFECTS",
    "PREVIEW_POLICY_NAME",
    "SIGNED_PROFILE_AUTHORITY_SOURCE",
    "SOURCE_PRECEDENCE",
    "AuthenticatedPrincipalEvidence",
    "AuthorityResolution",
    "AuthorityResolutionRequest",
    "AuthorityResolver",
    "AuthorityResolverError",
    "EffectCeiling",
    "ExistingRunAuthorityEvidence",
    "LocalWorktreeAuthority",
    "PolicySelection",
    "PrincipalBinding",
    "PrincipalSourceKind",
    "RepositoryPolicyConstraint",
    "SignedProfileEvidence",
    "effect_ceiling_cid",
    "install_local_worktree_authority",
    "mode_default_effects",
    "policy_cid_for",
    "resolve_authority",
)
