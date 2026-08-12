"""Resolve prompt-only invocations from one frozen trusted context.

The prompt is only hashed.  It cannot select a repository, principal, policy,
provider, validation command, or authority.  ``CanonicalResolutionPipeline``
is the single place where a complete launch context is composed before an
entrypoint may cause effects.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .context_adapters import (
    LOCAL_ADAPTER_BINDING_SCHEMA,
    LOCAL_ADAPTER_BINDING_TTL_NS,
    CanonicalResolutionCore,
    FrozenInvocationContext,
    FrozenMapping,
    InvocationContext,
    InvocationContextError,
    LocalInvocationContextFactory,
    MCPInvocationContextFactory,
    ResolutionField,
    _freeze,
    _thaw,
    _verified_installed_profile,
)
from .context_adapters import _canonical as _context_canonical

PROMPT_FORBIDDEN_FIELDS = frozenset({"allowlist", "caller", "policy", "provider", "validation_argv", "authority"})
REQUIRED_LAUNCH_FIELDS = ("repository", "state", "profile", "run", "objective", "task_source", "resources", "validation", "topology")
_STALE = frozenset({"stale", "expired", "unverified"})


class AmbientInferenceError(Exception):
    """Base error for prompt-only resolution."""


class MaterialAmbiguityError(AmbientInferenceError):
    """Effects were requested with unresolved, stale, or ambiguous evidence."""


class PromptContaminationError(AmbientInferenceError):
    """Prompt-derived data attempted to populate authority-bearing input."""


def _canonical(value: Any) -> str:
    """Use the context encoder for receipts as well as context CIDs.

    Keeping a second, ``default=str`` encoder here used to make equal mixed
    key/set inputs either raise in ``json.dumps`` or produce transport-local
    identities.  A receipt is part of the trust boundary, so it gets the same
    closed encoder as the immutable core.
    """
    return _context_canonical(_freeze(value))


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResolutionContinuation:
    """One machine-readable continuation returned before any denied launch."""

    kind: str
    fields: tuple[str, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", str(self.kind))
        object.__setattr__(self, "fields", tuple(sorted({str(item) for item in self.fields})))
        object.__setattr__(self, "reason", str(self.reason))

    @property
    def type(self) -> str:
        return self.kind

    def as_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "fields": list(self.fields), "reason": self.reason}


@dataclass(frozen=True)
class AmbientEvidence:
    """Compatibility input view; it never authenticates facts by itself."""

    cwd: str
    profile_path: Optional[str] = None
    profile_signed: bool = False
    server_authenticated: bool = False
    server_context: Optional[Mapping[str, Any]] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Freeze external mappings at the evidence boundary as well.  This
        # avoids a mutable server-context changing a later receipt.
        object.__setattr__(self, "cwd", os.path.abspath(self.cwd))
        object.__setattr__(self, "profile_path", os.path.abspath(self.profile_path) if self.profile_path else None)
        object.__setattr__(self, "profile_signed", self.profile_signed is True)
        object.__setattr__(self, "server_authenticated", self.server_authenticated is True)
        object.__setattr__(self, "server_context", _freeze(self.server_context) if self.server_context is not None else None)
        object.__setattr__(self, "extra", _freeze(self.extra))

    def fingerprint(self) -> str:
        payload = {"cwd": self.cwd, "profile_path": self.profile_path, "profile_signed": self.profile_signed,
            "server_authenticated": self.server_authenticated, "server_context": _thaw(self.server_context), "extra": _thaw(self.extra)}
        return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()

    def is_sufficient_for_prompt_only(self) -> bool:
        try:
            context = self.invocation_context()
            return (
                context.transport == "local"
                and context.authenticated
                and context.field("repository").value is not None
                and context.field("profile").freshness == "fresh"
            )
        except InvocationContextError:
            return False

    def invocation_context(self) -> InvocationContext:
        if self.server_context is not None:
            context = self.server_context
            target = context.get("target")
            values: dict[str, Any] = {}
            # Only documented server-owned facts are admitted.  In particular
            # a client cannot smuggle a filesystem path under another key.
            for key in ("state", "profile", "run", "objective", "task_source", "resources", "validation", "topology"):
                if key in context:
                    values[key] = context[key]
            # Alias maps and capability snapshots belong to server process
            # configuration, never the request-shaped ambient payload.
            return MCPInvocationContextFactory().create(target_alias=target,
                authenticated=self.server_authenticated and context.get("authenticated") is True,
                values=values)
        try:
            return LocalInvocationContextFactory().create(cwd=self.cwd, profile_path=self.profile_path, profile_signed=self.profile_signed,
                values={"default_target": self.extra.get("default_target")} if self.extra.get("default_target") is not None else None)
        except InvocationContextError:
            # Legacy preview receipts remain available outside a worktree, but
            # are tagged unverified so a complete production pipeline cannot
            # turn them into effects.
            values = {"repository": ResolutionField(value=self.extra.get("default_target") or self.cwd,
                                                      source="legacy_ambient_preview", freshness="unverified"),
                      "profile": ResolutionField(value=self.profile_path,
                                                   source="unverified_profile_candidate", freshness="unverified")}
            return InvocationContext("local", False, values,
                                     {"repository": "legacy_preview", "profile": "none"})


@dataclass(frozen=True)
class ResolutionReceipt:
    evidence_fingerprint: str
    resolved: bool
    launch_authorized: bool
    target: Optional[str] = None
    profile: Optional[str] = None
    reason: Optional[str] = None
    prompt_hash: Optional[str] = None
    policy: Optional[Mapping[str, Any]] = None
    provider: Optional[str] = None
    caller: Optional[str] = None
    allowlist: Optional[Sequence[str]] = None
    authority: Optional[Mapping[str, Any]] = None
    validation_argv: Optional[Sequence[str]] = None
    context_cid: Optional[str] = None
    field_receipts: Mapping[str, Mapping[str, Any]] = field(default_factory=FrozenMapping)
    continuation: Optional[ResolutionContinuation] = None
    bindings_authoritative: bool = False
    untrusted_bindings: Mapping[str, Any] = field(default_factory=FrozenMapping)

    def __post_init__(self) -> None:
        # These legacy-named slots used to copy caller input onto the launch
        # receipt.  Keep the attributes null so downstream code cannot mistake
        # them for resolver-verified authority.  Direct constructors get the
        # same treatment as the public service and retain any presentation
        # data only in an explicitly untrusted envelope.
        untrusted = dict(self.untrusted_bindings)
        for name in PROMPT_FORBIDDEN_FIELDS:
            value = getattr(self, name)
            if value is not None:
                untrusted.setdefault(name, value)
            object.__setattr__(self, name, None)
        object.__setattr__(self, "untrusted_bindings", _freeze(untrusted))
        object.__setattr__(self, "field_receipts", _freeze(self.field_receipts))
        # There is no constructor input that can promote presentation metadata
        # into verified resolver facts.
        object.__setattr__(self, "bindings_authoritative", False)
        if isinstance(self.continuation, str):
            object.__setattr__(self, "continuation", ResolutionContinuation(self.continuation, reason=self.reason or ""))

    def to_dict(self) -> dict[str, Any]:
        return {"evidence_fingerprint": self.evidence_fingerprint, "resolved": self.resolved,
            "launch_authorized": self.launch_authorized, "target": self.target, "profile": self.profile,
            "reason": self.reason, "prompt_hash": self.prompt_hash, "policy": _thaw(self.policy),
            "provider": self.provider, "caller": self.caller, "allowlist": _thaw(self.allowlist),
            "authority": _thaw(self.authority), "validation_argv": _thaw(self.validation_argv),
            "context_cid": self.context_cid, "field_receipts": _thaw(self.field_receipts),
            "continuation": self.continuation.as_dict() if self.continuation else None,
            "bindings_authoritative": self.bindings_authoritative,
            "untrusted_bindings": _thaw(self.untrusted_bindings)}

    def identity(self) -> str:
        # Presentation-only caller metadata must not perturb the launch-gate
        # receipt identity.  Authority and identity are derived solely from the
        # canonical resolver outputs and continuation.
        identity_payload = self.to_dict()
        identity_payload.pop("untrusted_bindings", None)
        return hashlib.sha256(_canonical(identity_payload).encode("utf-8")).hexdigest()

    @property
    def cid(self) -> str:
        """Content identity of this frozen receipt."""
        return "sha256:" + self.identity()

    @property
    def receipt_cid(self) -> str:
        return self.cid


def _default_profile_search_paths(cwd: str) -> list[str]:
    home = Path.home()
    return [str(Path(cwd) / ".agent-supervisor" / "profile.signed.json"), str(Path(cwd) / "profile.signed.json"),
            str(home / ".agent-supervisor" / "profile.signed.json"), str(home / ".config" / "agent-supervisor" / "profile.signed.json")]


def _looks_signed(path: str) -> bool:
    """Legacy discovery predicate, deliberately never an authentication gate."""
    candidate = Path(path)
    return candidate.is_file() and not candidate.is_symlink()


def collect_ambient_evidence(*, cwd: Optional[str] = None, profile_path: Optional[str] = None,
    profile_signed: Optional[bool] = None, server_context: Optional[Mapping[str, Any]] = None,
    server_authenticated: Optional[bool] = None, profile_search_paths: Optional[Sequence[str]] = None,
    extra: Optional[Mapping[str, Any]] = None) -> AmbientEvidence:
    resolved_cwd = os.path.abspath(cwd or os.getcwd())
    found = os.path.abspath(profile_path) if profile_path else None
    if found is None:
        for candidate in profile_search_paths or _default_profile_search_paths(resolved_cwd):
            probe = Path(candidate)
            if probe.is_file() and not probe.is_symlink():
                found = str(probe.resolve())
                break
    # Preserve the caller's legacy metadata in the fingerprint, but no code
    # may turn it into profile authority.  LocalInvocationContextFactory asks
    # the owning lifecycle to verify the installed profile instead.
    signed = profile_signed is True if profile_signed is not None else False
    # Copying is insufficient: AmbientEvidence deep-freezes it in __post_init__.
    context = server_context if server_context is not None else None
    authenticated = server_authenticated is True if server_authenticated is not None else bool(context and context.get("authenticated") is True)
    return AmbientEvidence(resolved_cwd, found, signed, authenticated, context, extra or {})


def sanitize_prompt_bindings(prompt: str, bindings: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    for key in (bindings or {}):
        if str(key).lower() in PROMPT_FORBIDDEN_FIELDS:
            raise PromptContaminationError(f"prompt bindings must not populate forbidden field: {key}")
    lowered = prompt.lower()
    for name in PROMPT_FORBIDDEN_FIELDS:
        if any(marker in lowered for marker in (f'"{name}":', f"'{name}':", f"{name}=", f"{name}:")):
            raise PromptContaminationError(f"prompt text must not populate forbidden field: {name}")
    return dict(bindings or {})


LeafResolver = Callable[..., Any]


class _FrozenList(tuple):
    """Immutable sequence that preserves legacy list comparison behavior."""

    def __eq__(self, other: object) -> bool:
        return list(self) == list(other) if isinstance(other, (list, tuple)) else False


class CanonicalResolutionPipeline:
    """Compose leaf resolutions in dependency order from frozen context facts.

    Resolvers receive ``(context, already_resolved)`` (or just the latter for a
    one-argument resolver) and may return a ``ResolutionField``, a receipt-like
    mapping, or a plain value.  They cannot alter previous fields.
    """

    required_fields = REQUIRED_LAUNCH_FIELDS

    def __init__(self, resolvers: Optional[Mapping[str, LeafResolver]] = None,
                 required_fields: Sequence[str] = REQUIRED_LAUNCH_FIELDS,
                 *, verify_prefilled: bool = False,
                 resolver_factory: Optional[Callable[[str], Mapping[str, LeafResolver]]] = None,
                 require_authenticated_context: bool = True) -> None:
        names = tuple(str(name) for name in required_fields)
        if len(set(names)) != len(names) or not names or names[0] != "repository":
            raise ValueError("resolution pipeline requires ordered fields beginning with repository")
        self._resolvers = dict(resolvers or {})
        self._resolver_factory = resolver_factory
        self.required_fields = names
        self.verify_prefilled = verify_prefilled
        self.require_authenticated_context = require_authenticated_context

    @staticmethod
    def _as_field(value: Any, *, source: str = "leaf_resolver") -> ResolutionField:
        if isinstance(value, ResolutionField):
            return ResolutionField(**value.as_dict())
        if isinstance(value, Mapping) and {"value", "source"}.intersection(value):
            return ResolutionField(value=value.get("value"), source=value.get("source", source), freshness=value.get("freshness", "fresh"),
                alternatives=tuple(value.get("alternatives", ())), confidence=value.get("confidence", "high"))
        return ResolutionField(value=value, source=source)

    @staticmethod
    def _call(resolver: LeafResolver, context: InvocationContext, resolved: Mapping[str, ResolutionField]) -> Any:
        try:
            parameters = inspect.signature(resolver).parameters
        except (TypeError, ValueError):
            return resolver(context, resolved)
        if len(parameters) <= 1:
            return resolver(resolved)
        return resolver(context, resolved)

    def resolve_fields(self, context: InvocationContext, *, prompt_cid: str = "") -> tuple[FrozenMapping, Optional[ResolutionContinuation]]:
        if not isinstance(context, InvocationContext):
            raise TypeError("canonical resolution requires an InvocationContext")
        resolvers = (dict(self._resolver_factory(prompt_cid))
                     if self._resolver_factory is not None else self._resolvers)
        resolved: dict[str, ResolutionField] = {}
        failures: list[tuple[str, str]] = []
        for name in self.required_fields:
            field = context.field(name)
            # Production resolvers are verification gates.  A value collected
            # by an adapter is only a candidate; it cannot bypass its leaf
            # resolver simply by being prefilled in the context.
            if name in resolvers and (self.verify_prefilled or field.value is None):
                try:
                    field = self._as_field(self._call(resolvers[name], context, FrozenMapping(resolved)))
                except Exception as exc:  # Leaf failures are a denial, never a launch escape.
                    field = ResolutionField(value=None, source="leaf_resolver", freshness="unverified", confidence="none")
                    failures.append((name, f"resolver failed: {type(exc).__name__}"))
            resolved[name] = field
            if (self.required_fields == REQUIRED_LAUNCH_FIELDS and name == "repository" and context.transport == "local"
                    and field.source != "verified_git_worktree" and not field.source.startswith("target_resolver:")):
                failures.append((name, "unverified Git worktree"))
            elif field.freshness in _STALE or field.confidence == "none":
                failures.append((name, "stale evidence"))
            elif field.value is None:
                failures.append((name, "multiple candidates" if field.alternatives else "zero candidates"))
        if self.require_authenticated_context and not context.authenticated:
            failures.insert(0, ("context", "unauthenticated invocation context"))
        if not failures:
            return FrozenMapping(resolved), None
        names = tuple(name for name, _ in failures)
        details = "; ".join(f"{name}: {reason}" for name, reason in failures)
        if any(reason == "multiple candidates" for _, reason in failures): kind = "multiple_evidence"
        elif any("failed" in reason for _, reason in failures): kind = "resolver_failure"
        elif any(reason == "stale evidence" for _, reason in failures): kind = "stale_evidence"
        elif failures[0][0] == "context": kind = "unauthenticated_context"
        else: kind = "zero_evidence"
        return FrozenMapping(resolved), ResolutionContinuation(kind, names, details)


class ProductionCanonicalResolverFactory:
    """Build the only launch-capable resolver composition.

    The factory is deliberately thin: every accepted value is projected from
    a public resolver receipt.  Context fields are candidates only.  Missing
    provider/lifecycle evidence is represented by conservative typed evidence
    and still passed through the owning resolver, so the call graph is complete
    without inventing authority.
    """

    required_fields = REQUIRED_LAUNCH_FIELDS

    def __init__(
        self,
        *,
        capability_evidence: Optional[Mapping[str, Any]] = None,
        python_allowlisted_roots: Sequence[str] = (),
        mcp_repository_aliases: Optional[Mapping[str, str]] = None,
        mcpplusplus_issuer_public_keys: Optional[Mapping[str, str]] = None,
        mcpplusplus_revoked_proof_cids: Sequence[str] = (),
    ) -> None:
        """Capture host-owned evidence outside the request DTO.

        An :class:`InvocationContext` is caller-constructible, so none of its
        booleans, allowlists, alias maps, keys, or capability mappings can be
        authority.  The long-lived entrypoint constructs this factory from its
        own configuration and passes request candidates separately.  The
        default is deliberately unconfigured and therefore denies Python/MCP
        effects while retaining the independently verifiable, adapter-bound
        local lifecycle.
        """
        if capability_evidence is not None and not isinstance(
            capability_evidence, Mapping
        ):
            raise InvocationContextError(
                "production capability evidence must be a mapping"
            )
        self._configured_capability_evidence = (
            _freeze(capability_evidence)
            if capability_evidence is not None else None
        )
        self._python_allowlisted_roots = self._normalize_roots(
            python_allowlisted_roots,
            label="Python allowlist",
        )
        aliases: dict[str, str] = {}
        for alias, raw_path in (mcp_repository_aliases or {}).items():
            if (
                not isinstance(alias, str)
                or re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]{0,127}", alias) is None
            ):
                raise InvocationContextError("production MCP alias is invalid")
            roots = self._normalize_roots((raw_path,), label="MCP alias")
            aliases[alias] = roots[0]
        self._mcp_repository_aliases = FrozenMapping(aliases)
        self._mcpplusplus_issuer_public_keys = FrozenMapping(
            (str(issuer), str(key))
            for issuer, key in (mcpplusplus_issuer_public_keys or {}).items()
        )
        self._mcpplusplus_revoked_proof_cids = tuple(sorted({
            str(item) for item in mcpplusplus_revoked_proof_cids if str(item)
        }))

    @staticmethod
    def _normalize_roots(values: Sequence[str], *, label: str) -> tuple[str, ...]:
        roots: list[str] = []
        for value in values:
            try:
                absolute = Path(os.path.abspath(os.fspath(value)))
                resolved = absolute.resolve(strict=True)
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise InvocationContextError(f"{label} root is unavailable") from exc
            if absolute != resolved or not resolved.is_dir():
                raise InvocationContextError(
                    f"{label} roots must be real non-symlinked directories"
                )
            roots.append(str(resolved))
        return tuple(sorted(set(roots)))

    def pipeline(self) -> CanonicalResolutionPipeline:
        return CanonicalResolutionPipeline(
            required_fields=self.required_fields,
            verify_prefilled=True,
            resolver_factory=self._resolvers,
            # Authentication is established by the profile/authority leaf,
            # never by the public envelope's caller-set boolean.
            require_authenticated_context=False,
        )

    @staticmethod
    def _unavailable(name: str, reason: str, *, alternatives: Sequence[str] = ()) -> ResolutionField:
        return ResolutionField(
            value=None,
            source=f"{name}_resolver:{reason}",
            freshness="fresh",
            confidence="high",
            alternatives=tuple(alternatives),
        )

    @staticmethod
    def _value(context: InvocationContext, name: str) -> Any:
        return _thaw(context.field(name).value)

    @staticmethod
    def _cid(label: str, value: Any) -> str:
        from ipfs_accelerate_py.agent_supervisor.multiformats_identity import cid_for_dag_json

        return cid_for_dag_json({"kind": label, "value": value})

    def _capability_evidence(self, context: InvocationContext, *, state_root: str) -> tuple[Any, bool]:
        """Rehydrate the public evidence contract or a fail-closed observation."""
        from .capability_resolver import (
            CAPABILITY_EVIDENCE_SCHEMA,
            PREFERRED_PROVIDER,
            CapabilityEvidence,
            PreferredProviderCapability,
            ProviderCapabilityEvidence,
            ResourceSampleEvidence,
            TopologyEvidence,
            ValidationPolicyEvidence,
        )

        raw = (
            _thaw(self._configured_capability_evidence)
            if self._configured_capability_evidence is not None else None
        )
        if isinstance(raw, Mapping):
            try:
                providers: dict[str, ProviderCapabilityEvidence] = {}
                for provider_id, item in raw.get("providers", {}).items():
                    if not isinstance(item, Mapping):
                        raise TypeError("provider evidence must be a mapping")
                    providers[str(provider_id)] = ProviderCapabilityEvidence(**{
                        name: item[name] for name in (
                            "provider_id", "capability", "policy_allowed", "healthy",
                            "authenticated", "observed_capability_cid", "usage_evidence_cid",
                            "budget_cid", "max_concurrency", "request_headroom",
                        )
                    })
                resources_raw = raw["resources"]
                validation_raw = raw["validation"]
                topology_raw = raw["topology"]
                if not all(isinstance(item, Mapping) for item in
                           (resources_raw, validation_raw, topology_raw)):
                    raise TypeError("capability leaf evidence must be mappings")
                resources = ResourceSampleEvidence(**{
                    name: resources_raw[name] for name in (
                        "ready_width", "host_worker_limit", "host_available_workers",
                        "max_processes", "max_validation_workers", "cpu_millis",
                        "memory_bytes", "provider_request_limit", "deadline_ms",
                        "lane_labels",
                    )
                })
                validation = ValidationPolicyEvidence(
                    allowlisted_argv=tuple(tuple(argv) for argv in validation_raw["allowlisted_argv"]),
                    policy_cid=validation_raw["policy_cid"],
                )
                topology = TopologyEvidence(**{
                    name: topology_raw[name] for name in (
                        "distributed_capable", "shard_count", "owner_principal_ref",
                        "state_root", "database_relative_path", "coordinator_cid",
                        "lease_namespace", "fencing_generation", "ipfs_publish_capable",
                        "parquet_capable", "preferred_mode", "ipfs_backend_handle",
                    )
                })
                evidence = CapabilityEvidence(
                    providers=providers,
                    resources=resources,
                    validation=validation,
                    topology=topology,
                    task_revision_cid=raw["task_revision_cid"],
                    attempt_cid=raw["attempt_cid"],
                    worktree_cid=raw["worktree_cid"],
                    authenticated_profile_override=raw.get("authenticated_profile_override", ""),
                    authenticated_profile_override_cid=raw.get("authenticated_profile_override_cid", ""),
                )
                if raw.get("schema") != CAPABILITY_EVIDENCE_SCHEMA:
                    raise ValueError("capability evidence schema mismatch")
                if (raw.get("content_id") is not None
                        and raw.get("content_id") != evidence.content_id):
                    raise ValueError("capability evidence content identity mismatch")
                # Topology is state-bound.  A valid-looking receipt for another
                # run must not cross the join.
                if evidence.topology.state_root != state_root:
                    raise ValueError("capability topology is bound to another state root")
                return evidence, True
            except (KeyError, TypeError, ValueError):
                pass

        missing = self._cid("missing-capability-evidence", context.cid)
        provider = ProviderCapabilityEvidence(
            provider_id=PREFERRED_PROVIDER,
            capability=PreferredProviderCapability.UNAVAILABLE,
            policy_allowed=False,
            healthy=False,
            authenticated=False,
            observed_capability_cid=missing,
            usage_evidence_cid=missing,
            budget_cid=missing,
            max_concurrency=0,
            request_headroom=0,
        )
        return CapabilityEvidence(
            providers={PREFERRED_PROVIDER: provider},
            resources=ResourceSampleEvidence(
                ready_width=0, host_worker_limit=0, host_available_workers=0,
                max_processes=0, max_validation_workers=0, cpu_millis=0,
                memory_bytes=0, provider_request_limit=0, deadline_ms=1,
            ),
            validation=ValidationPolicyEvidence(allowlisted_argv=(), policy_cid=missing),
            topology=TopologyEvidence(
                distributed_capable=False, shard_count=1,
                owner_principal_ref="unresolved-principal", state_root=state_root,
                database_relative_path="coordination.duckdb", coordinator_cid=missing,
                lease_namespace="unresolved", fencing_generation=0,
                ipfs_publish_capable=False, parquet_capable=False,
            ),
            task_revision_cid=missing,
            attempt_cid=missing,
            worktree_cid=missing,
        ), False

    def _verified_mcpplusplus_evidence(
        self,
        context: InvocationContext,
        *,
        repository_id: str,
    ) -> str | None:
        """Revalidate a raw request chain against host-owned trust anchors."""
        if context.transport != "mcp++":
            return None
        alias = self._value(context, "repository_alias")
        raw_chain = self._value(context, "ucan_delegation_chain")
        key_map = _thaw(self._mcpplusplus_issuer_public_keys)
        configured_repository = (
            self._mcp_repository_aliases.get(alias)
            if isinstance(alias, str) else None
        )
        if (
            not isinstance(alias, str)
            or configured_repository is None
            or not isinstance(raw_chain, list)
            or not raw_chain
            or not all(isinstance(item, Mapping) for item in raw_chain)
            or not isinstance(key_map, Mapping)
            or not key_map
        ):
            return None

        try:
            from ipfs_accelerate_py.mcp_server.mcplusplus.delegation import (
                parse_delegation_chain,
                validate_raw_delegation_chain,
            )

            parsed = parse_delegation_chain(raw_chain)
            verdict = validate_raw_delegation_chain(
                raw_chain=raw_chain,
                resource=alias,
                ability="agent-supervisor/invoke",
                actor=alias,
                require_signatures=True,
                issuer_public_keys=dict(key_map),
                revoked_proof_cids=self._mcpplusplus_revoked_proof_cids,
            )
            leaf_caps = parsed[-1].capabilities if parsed else ()
            cryptographically_signed = bool(parsed) and all(
                bool(key_map.get(item.issuer))
                and (
                    item.signature.startswith("ed25519:")
                    or item.signature.startswith("ed25519-hex:")
                    or item.signature.startswith("hex:")
                    or bool(re.fullmatch(r"[0-9A-Fa-f]{128}", item.signature))
                )
                for item in parsed
            )
            exactly_attenuated = bool(leaf_caps) and all(
                cap.resource == alias
                and cap.ability == "agent-supervisor/invoke"
                and "*" not in (cap.resource, cap.ability)
                for cap in leaf_caps
            )
            if not (
                verdict.allowed
                and cryptographically_signed
                and exactly_attenuated
            ):
                raise ValueError("MCP++ delegation is not exactly verified")
        except (ImportError, TypeError, ValueError):
            return None

        return self._cid(
            "verified-mcpplusplus-delegation",
            {
                "alias": alias,
                "repository_id": repository_id,
                "proof_lineage": list(verdict.proof_lineage),
            },
        )

    def _verified_local_adapter_evidence(
        self,
        context: InvocationContext,
        *,
        repository_id: str,
        repository_root: str,
        checkout_id: str,
    ) -> Any | None:
        """Verify an exact local-adapter binding against the current lifecycle."""
        if context.transport != "local" or context.adapter_receipt is None:
            return None
        raw = _thaw(context.adapter_receipt)
        if not isinstance(raw, Mapping):
            return None
        expected_keys = {
            "schema", "transport", "core_cid", "repository_id",
            "repository_root", "checkout_id", "profile_id", "profile_cid",
            "identity_did", "lifecycle_generation", "lifecycle_anchor_id",
            "issued_at_ns", "expires_at_ns", "nonce", "signature",
        }
        if set(raw) != expected_keys:
            return None
        signature = raw.get("signature")
        body = {name: raw[name] for name in expected_keys - {"signature"}}
        generation = body.get("lifecycle_generation")
        issued_at_ns = body.get("issued_at_ns")
        expires_at_ns = body.get("expires_at_ns")
        nonce = body.get("nonce")
        now_ns = time.time_ns()
        if (
            body.get("schema") != LOCAL_ADAPTER_BINDING_SCHEMA
            or body.get("transport") != "local"
            or body.get("core_cid") != context.core_cid
            or body.get("repository_id") != repository_id
            or body.get("repository_root") != repository_root
            or body.get("checkout_id") != checkout_id
            or not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation < 1
            or not isinstance(issued_at_ns, int)
            or isinstance(issued_at_ns, bool)
            or not isinstance(expires_at_ns, int)
            or isinstance(expires_at_ns, bool)
            or expires_at_ns <= issued_at_ns
            or expires_at_ns - issued_at_ns > LOCAL_ADAPTER_BINDING_TTL_NS
            or issued_at_ns > now_ns + 5_000_000_000
            or expires_at_ns < now_ns
            or not isinstance(nonce, str)
            or re.fullmatch(r"[0-9a-f]{32}", nonce) is None
            or not isinstance(signature, str)
            or not signature
        ):
            return None
        installed = _verified_installed_profile(repository_id)
        if installed is None:
            return None
        repository_field = context.field("repository")
        profile_field = context.field("profile")
        if (
            repository_field.value != repository_root
            or repository_field.freshness != "fresh"
            or profile_field.value != getattr(installed, "content_id", None)
            or profile_field.freshness != "fresh"
            or body.get("profile_id") != getattr(installed, "profile_id", None)
            or body.get("profile_cid") != getattr(installed, "content_id", None)
            or body.get("identity_did") != getattr(installed, "identity_did", None)
            or generation != getattr(installed, "lifecycle_generation", None)
            or body.get("lifecycle_anchor_id")
            != getattr(installed, "lifecycle_anchor_id", None)
        ):
            return None
        try:
            from .local_profile import verify_did_key_signature

            verify_did_key_signature(
                identity_did=installed.identity_did,
                payload=body,
                signature=signature,
            )
        except (AttributeError, ImportError, TypeError, ValueError):
            return None
        return installed

    def _authority(
        self,
        context: InvocationContext,
        *,
        repository_id: str,
        repository_root: str | None = None,
        checkout_id: str | None = None,
    ) -> tuple[Any, Any | None]:
        """Resolve authority only from lifecycle/transport verification output."""
        from .authority_resolver import (
            LOCAL_WORKTREE_ALLOWED_EFFECTS,
            LOCAL_WORKTREE_POLICY_NAME,
            SIGNED_PROFILE_AUTHORITY_SOURCE,
            AuthenticatedPrincipalEvidence,
            AuthorityResolutionRequest,
            PrincipalSourceKind,
            SignedProfileEvidence,
            policy_cid_for,
            resolve_authority,
        )
        from .contracts import InvocationMode, ResolutionSource

        if context.transport == "local":
            # Repeat the signed adapter-binding and current lifecycle checks at
            # the authority leaf so rotation/revocation between leaf calls
            # fails closed.  The public transport label and boolean alone have
            # no authority.
            installed = (
                self._verified_local_adapter_evidence(
                    context,
                    repository_id=repository_id,
                    repository_root=repository_root,
                    checkout_id=checkout_id,
                )
                if repository_root is not None and checkout_id is not None
                else None
            )
            if installed is None:
                return resolve_authority(AuthorityResolutionRequest(mode=InvocationMode.WORKTREE)), None
            installed_receipt_cid = self._cid(
                "verified-local-profile",
                {"lifecycle_content_id": installed.content_id, "profile": installed.to_dict()},
            )
            signed = SignedProfileEvidence(
                profile_name="local-worktree",
                profile_cid=installed_receipt_cid,
                policy_cid=policy_cid_for(LOCAL_WORKTREE_POLICY_NAME),
                principal_ref=f"local-profile:{installed.profile_id}",
                authority_source_ref=SIGNED_PROFILE_AUTHORITY_SOURCE,
                allowed_effects=LOCAL_WORKTREE_ALLOWED_EFFECTS,
                evidence_cid=installed_receipt_cid,
                signature_verified=True,
            )
            return resolve_authority(AuthorityResolutionRequest(
                mode=InvocationMode.WORKTREE,
                signed_profile=signed,
            )), installed

        # Python and ordinary MCP have no independently verifiable principal
        # in this surface.  Their request booleans are metadata only.
        evidence_cid = self._verified_mcpplusplus_evidence(
            context,
            repository_id=repository_id,
        )
        if evidence_cid is None:
            return resolve_authority(AuthorityResolutionRequest(mode=InvocationMode.WORKTREE)), None
        principal = AuthenticatedPrincipalEvidence(
            principal_ref=f"mcp++:{repository_id}",
            source=ResolutionSource.AUTHENTICATED_TRANSPORT,
            evidence_cid=evidence_cid,
            kind=PrincipalSourceKind.MCP_PLUS_UCAN,
            transport="mcp++",
            audience=repository_id,
            signature_verified=True,
            ucan_verified=True,
        )
        return resolve_authority(AuthorityResolutionRequest(
            mode=InvocationMode.WORKTREE,
            authenticated_principal=principal,
        )), None

    def _resolvers(self, prompt_cid: str) -> Mapping[str, LeafResolver]:
        if not prompt_cid:
            from ipfs_accelerate_py.agent_supervisor.multiformats_identity import cid_for_bytes

            prompt_cid = cid_for_bytes(b"")
        cache: dict[str, Any] = {}

        def repository(context: InvocationContext, _: Mapping[str, ResolutionField]) -> ResolutionField:
            from .target_resolver import RepositoryTargetEvidence, resolve_repository_target

            candidate = context.field("repository")
            path: str | None = None
            roots: tuple[str, ...] = ()
            if context.transport == "local":
                # The path is only a discovery candidate.  A real Git receipt
                # and repository-bound installed profile are both required
                # below before this leaf publishes a value.
                if isinstance(candidate.value, str):
                    path = candidate.value
                    roots = (candidate.value,)
            elif context.transport == "python":
                if (
                    isinstance(candidate.value, str)
                    and candidate.value in self._python_allowlisted_roots
                ):
                    path = candidate.value
                    roots = self._python_allowlisted_roots
            elif context.transport in {"mcp", "mcp++"}:
                alias = self._value(context, "repository_alias")
                configured = (
                    self._mcp_repository_aliases.get(alias)
                    if isinstance(alias, str) else None
                )
                if isinstance(configured, str):
                    path = configured
                    roots = (configured,)
            if path is None or not roots:
                return self._unavailable(
                    "target",
                    "resolver_owned_repository_scope_required",
                    alternatives=candidate.alternatives,
                )
            resolution = resolve_repository_target(RepositoryTargetEvidence(
                cwd=path, allowlisted_roots=roots,
            ))
            if not resolution.unique or resolution.binding is None:
                alternatives = tuple(item.root_path for item in resolution.candidates_considered if not item.rejection_reason)
                return self._unavailable("target", "repository_target_not_unique", alternatives=alternatives)
            if context.transport == "local":
                installed = self._verified_local_adapter_evidence(
                    context,
                    repository_id=resolution.binding.repository_id,
                    repository_root=resolution.binding.repository_root,
                    checkout_id=resolution.binding.checkout_id,
                )
                if installed is None:
                    return self._unavailable(
                        "target", "verified_local_adapter_receipt_required",
                    )
            elif context.transport == "mcp++":
                if self._verified_mcpplusplus_evidence(
                    context,
                    repository_id=resolution.binding.repository_id,
                ) is None:
                    return self._unavailable(
                        "target", "verified_mcpplusplus_delegation_required",
                    )
            else:
                # No OS-key or signed server-transport evidence is defined by
                # this entrypoint yet.  A boolean cannot stand in for one.
                return self._unavailable(
                    "target", "verified_transport_principal_required",
                )
            cache["target"] = resolution
            return ResolutionField(resolution.binding.repository_root,
                source=f"target_resolver:{resolution.content_id}", freshness="fresh")

        def state(_: InvocationContext, resolved: Mapping[str, ResolutionField]) -> ResolutionField:
            from .state_resolver import StateResolutionEvidence, resolve_state

            target = cache.get("target")
            if target is None or target.binding is None:
                return self._unavailable("state", "repository_receipt_required")
            resolution = resolve_state(StateResolutionEvidence(
                repository_id=target.binding.repository_id,
                repository_root=target.binding.repository_root,
                checkout_id=target.binding.checkout_id,
            ))
            cache["state"] = resolution
            return ResolutionField(resolution.state_root,
                source=f"state_resolver:{resolution.content_id}", freshness="fresh")

        def compose(context: InvocationContext) -> None:
            """Run every remaining public leaf exactly once for this invocation."""
            if cache.get("composed"):
                return
            target, state_resolution = cache.get("target"), cache.get("state")
            if target is None or state_resolution is None or target.binding is None:
                return
            from .capability_resolver import ProviderSelection, resolve_capabilities
            from .contracts import (
                InvocationBudget,
                InvocationMode,
                SupervisorInvocationRequest,
                WorktreeStrategy,
            )
            from .objective_resolver import ObjectiveResolutionEvidence, resolve_objectives
            from .profile_resolver import (
                ProfileCompositionRequest,
                ProfileSourceKind,
                ProfileSourceLayer,
                resolve_supervisor_profile,
            )
            from .state_resolver import RunCandidateResolutionRequest, resolve_run_candidates

            run_resolution = resolve_run_candidates(RunCandidateResolutionRequest(
                repository_id=target.binding.repository_id,
                checkout_id=target.binding.checkout_id,
                run_namespace=state_resolution.run_namespace,
            ))
            cache["run"] = run_resolution
            objective_resolution = resolve_objectives(ObjectiveResolutionEvidence(
                repository_root=target.binding.repository_root,
                repository_id=target.binding.repository_id,
                state_root=state_resolution.state_root,
                run_namespace=state_resolution.run_namespace,
                prompt_cid=prompt_cid,
            ))
            cache["objective"] = objective_resolution
            capability_evidence, capability_verified = self._capability_evidence(
                context, state_root=state_resolution.state_root,
            )
            capability_resolution = resolve_capabilities(capability_evidence)
            cache["capability"] = capability_resolution
            cache["capability_verified"] = (
                capability_verified
                and capability_resolution.selected_provider is not ProviderSelection.UNAVAILABLE
            )
            authority, installed = self._authority(
                context,
                repository_id=target.binding.repository_id,
                repository_root=target.binding.repository_root,
                checkout_id=target.binding.checkout_id,
            )
            cache["authority"] = authority
            layers: tuple[Any, ...] = ()
            if installed is not None and authority.authorized:
                layers = (ProfileSourceLayer(
                    kind=ProfileSourceKind.SIGNED_PROFILE,
                    evidence_cid=authority.authority_source_evidence_cid,
                    profile_name="local-worktree",
                    allowed_effects=authority.effect_ceiling.allowed_effects,
                    worktree_strategy=WorktreeStrategy.ISOLATED,
                    max_lanes=max(1, capability_resolution.resources.lane_ceiling),
                    signature_verified=True,
                    reviewed=True,
                ),)
            invocation = SupervisorInvocationRequest(
                prompt_cid=prompt_cid,
                prompt_ref="prompt-broker:in-memory",
                mode=InvocationMode.WORKTREE,
                budget=InvocationBudget(),
            )
            profile_resolution = resolve_supervisor_profile(ProfileCompositionRequest(
                invocation=invocation,
                repository=target,
                state=state_resolution,
                objective=objective_resolution,
                authority=authority,
                capability=capability_resolution,
                profile_layers=layers,
            ))
            cache["profile"] = profile_resolution
            cache["composed"] = True

        def profile(context: InvocationContext, __: Mapping[str, ResolutionField]) -> ResolutionField:
            compose(context)
            resolution = cache.get("profile")
            authority = cache.get("authority")
            if resolution is None:
                return self._unavailable("profile", "state_receipt_required")
            if (authority is None or not authority.authorized
                    or not cache.get("capability_verified")
                    or resolution.profile is None or resolution.effects_blocked):
                return self._unavailable("profile", "verified_profile_or_authority_required")
            return ResolutionField(
                resolution.profile.content_id,
                source=f"profile_resolver:{resolution.receipt.content_id}",
            )

        def run(context: InvocationContext, __: Mapping[str, ResolutionField]) -> ResolutionField:
            compose(context)
            resolution = cache.get("run")
            if resolution is None:
                return self._unavailable("run", "state_receipt_required")
            return ResolutionField({
                "action": resolution.action.value,
                "selected_run_id": resolution.selected_run_id,
                "run_namespace": resolution.target_run_namespace,
                "receipt_cid": resolution.content_id,
            }, source=f"run_resolver:{resolution.content_id}")

        def objective(context: InvocationContext, __: Mapping[str, ResolutionField]) -> ResolutionField:
            compose(context)
            resolution = cache.get("objective")
            if resolution is None or resolution.objective is None:
                alternatives = () if resolution is None else tuple(
                    item.objective_cid for item in resolution.objective_candidates_considered
                )
                return self._unavailable("objective", "objective_not_unique", alternatives=alternatives)
            return ResolutionField(
                resolution.objective.objective_cid,
                source=f"objective_resolver:{resolution.content_id}",
            )

        def task_source(context: InvocationContext, __: Mapping[str, ResolutionField]) -> ResolutionField:
            compose(context)
            resolution = cache.get("objective")
            if resolution is None or resolution.task_source is None:
                alternatives = () if resolution is None else tuple(
                    item.task_source_cid for item in resolution.task_source_candidates_considered
                )
                return self._unavailable("task_source", "task_source_not_unique", alternatives=alternatives)
            return ResolutionField(
                resolution.task_source.task_source_cid,
                source=f"task_source_resolver:{resolution.content_id}",
            )

        def capability(name: str) -> LeafResolver:
            def resolve(context: InvocationContext, __: Mapping[str, ResolutionField]) -> ResolutionField:
                compose(context)
                resolution = cache.get("capability")
                if resolution is None:
                    return self._unavailable(name, "state_receipt_required")
                if not cache.get("capability_verified"):
                    return self._unavailable(name, "verified_capability_evidence_required")
                value = {
                    "resources": resolution.resources.to_dict,
                    "validation": resolution.validation.to_dict,
                    "topology": resolution.topology.to_dict,
                }[name]()
                return ResolutionField(
                    value,
                    source=f"capability_resolver:{resolution.content_id}:{name}",
                )
            return resolve

        return {
            "repository": repository, "state": state, "run": run,
            "objective": objective, "task_source": task_source,
            "resources": capability("resources"), "validation": capability("validation"),
            "topology": capability("topology"), "profile": profile,
        }


class SupervisorResolutionService:
    """Deterministically resolve a launch gate from the canonical pipeline."""

    # Defaults are launch-capable, hence must never regress to the former
    # repository-only compatibility pipeline.  A caller that needs a partial
    # preview can still pass an explicitly constructed limited pipeline.
    def __init__(self, pipeline: Optional[CanonicalResolutionPipeline] = None) -> None:
        self.pipeline = pipeline or ProductionCanonicalResolverFactory().pipeline()

    def resolve(self, prompt: str, context: InvocationContext, *, trusted_bindings: Optional[Mapping[str, Any]] = None,
                explicit_target: Optional[str] = None, explicit_profile: Optional[str] = None) -> ResolutionReceipt:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not isinstance(context, InvocationContext):
            raise TypeError("resolution requires a frozen InvocationContext")
        trusted = dict(trusted_bindings or {})
        from ipfs_accelerate_py.agent_supervisor.multiformats_identity import cid_for_bytes

        fields, continuation = self.pipeline.resolve_fields(
            context,
            prompt_cid=cid_for_bytes(prompt.encode("utf-8")),
        )
        repository = fields.get("repository", ResolutionField())
        profile = fields.get("profile", context.field("profile"))
        extra_reasons: list[str] = []
        if explicit_target is not None:
            observed_target = repository.value
            if observed_target is None:
                observed_target = context.field("repository").value
            if observed_target is None or str(explicit_target) != str(observed_target):
                extra_reasons.append("repository: conflicting explicit target")
        if explicit_profile is not None:
            observed_profile = profile.value
            if observed_profile is None:
                observed_profile = context.field("profile").value
            if (observed_profile is None
                    or os.path.abspath(str(explicit_profile)) != os.path.abspath(str(observed_profile))):
                extra_reasons.append("profile: conflicting explicit profile")
        if extra_reasons:
            continuation = ResolutionContinuation("conflicting_evidence", ("repository", "profile"), "; ".join(extra_reasons))
        # Public ``trusted_bindings`` is a legacy metadata carrier.  Never add
        # those values to the verified leaf receipt map or authority-shaped
        # top-level slots: callers can invoke this method directly, and the
        # name does not confer authority.
        field_receipts: dict[str, Any] = {
            name: item.as_dict() for name, item in fields.items()
        }
        allowed = continuation is None
        reason = "prompt-only resolution from trusted context" if allowed else continuation.reason
        target_value = (repository.value if isinstance(repository.value, str)
                        else (_canonical(repository.value) if repository.value is not None else None))
        profile_value = (profile.value if isinstance(profile.value, str)
                         else (_canonical(profile.value) if profile.value is not None else None))
        return ResolutionReceipt(context.cid, allowed, allowed,
            target=target_value,
            profile=profile_value, reason=reason, prompt_hash=_hash_prompt(prompt),
            context_cid=context.cid, field_receipts=field_receipts,
            continuation=continuation, untrusted_bindings=trusted)


def resolve_prompt_only(prompt: str, evidence: AmbientEvidence, *, trusted_bindings: Optional[Mapping[str, Any]] = None,
    prompt_bindings: Optional[Mapping[str, Any]] = None, target: Optional[str] = None, profile: Optional[str] = None,
    require_no_low_level_flags: bool = True) -> ResolutionReceipt:
    del require_no_low_level_flags
    sanitize_prompt_bindings(prompt, prompt_bindings)
    try:
        context = evidence.invocation_context()
    except InvocationContextError as exc:
        # Collection failures are normalized to the same typed continuation as
        # every other ambiguity and never become an exception-to-launch path.
        continuation = ResolutionContinuation("invalid_trusted_context", ("context",), str(exc))
        return ResolutionReceipt(evidence.fingerprint(), False, False, reason=continuation.reason, prompt_hash=_hash_prompt(prompt),
            context_cid=None, continuation=continuation)
    return SupervisorResolutionService().resolve(prompt, context, trusted_bindings=trusted_bindings, explicit_target=target, explicit_profile=profile)


def launch_if_authorized(receipt: ResolutionReceipt) -> ResolutionReceipt:
    if not receipt.launch_authorized:
        raise MaterialAmbiguityError(receipt.reason or "launch denied")
    return receipt


def orchestrate(prompt: str, *, cwd: Optional[str] = None, profile_path: Optional[str] = None,
    profile_signed: Optional[bool] = None, server_context: Optional[Mapping[str, Any]] = None,
    server_authenticated: Optional[bool] = None, trusted_bindings: Optional[Mapping[str, Any]] = None,
    prompt_bindings: Optional[Mapping[str, Any]] = None, target: Optional[str] = None, profile: Optional[str] = None,
    launch: bool = False) -> ResolutionReceipt:
    receipt = resolve_prompt_only(prompt, collect_ambient_evidence(cwd=cwd, profile_path=profile_path, profile_signed=profile_signed,
        server_context=server_context, server_authenticated=server_authenticated), trusted_bindings=trusted_bindings,
        prompt_bindings=prompt_bindings, target=target, profile=profile)
    return launch_if_authorized(receipt) if launch else receipt


__all__ = ["AmbientEvidence", "AmbientInferenceError", "CanonicalResolutionCore", "CanonicalResolutionPipeline", "FrozenInvocationContext", "LeafResolver", "MaterialAmbiguityError",
           "PROMPT_FORBIDDEN_FIELDS", "PromptContaminationError", "REQUIRED_LAUNCH_FIELDS", "ResolutionContinuation", "ResolutionReceipt",
           "ProductionCanonicalResolverFactory", "SupervisorResolutionService", "collect_ambient_evidence", "launch_if_authorized", "orchestrate", "resolve_prompt_only",
           "sanitize_prompt_bindings"]
