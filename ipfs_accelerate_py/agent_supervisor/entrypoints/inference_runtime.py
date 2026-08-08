"""Resolve prompt-only invocations from one frozen trusted context.

The prompt is only hashed.  It cannot select a repository, principal, policy,
provider, validation command, or authority.  ``CanonicalResolutionPipeline``
is the single place where a complete launch context is composed before an
entrypoint may cause effects.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from .context_adapters import (
    CanonicalResolutionCore,
    FrozenMapping,
    FrozenInvocationContext,
    InvocationContext,
    InvocationContextError,
    LocalInvocationContextFactory,
    MCPInvocationContextFactory,
    ResolutionField,
    _freeze,
    _thaw,
)

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
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


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
        if self.server_context is not None:
            return self.server_authenticated and self.server_context.get("authenticated") is True
        # Compatibility evidence can be inspected before the production
        # pipeline runs.  ``invocation_context`` remains the authoritative
        # local-worktree/profile gate.
        return bool(self.cwd and self.profile_path and self.profile_signed and _looks_signed(self.profile_path))

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
            return MCPInvocationContextFactory().create(target_alias=target, authenticated=self.server_authenticated and context.get("authenticated") is True,
                values=values)
        try:
            return LocalInvocationContextFactory().create(cwd=self.cwd, profile_path=self.profile_path, profile_signed=self.profile_signed,
                values={"default_target": self.extra.get("default_target")} if self.extra.get("default_target") is not None else None)
        except InvocationContextError:
            # Legacy preview receipts remain available outside a worktree, but
            # are tagged unverified so a complete production pipeline cannot
            # turn them into effects.
            signed = bool(self.profile_path and self.profile_signed and _looks_signed(self.profile_path))
            values = {"repository": ResolutionField(value=self.extra.get("default_target") or self.cwd, source="legacy_ambient_preview", freshness="fresh" if signed else "unverified"),
                      "profile": ResolutionField(value=self.profile_path, source="verified_installed_profile" if signed else "unverified_profile", freshness="fresh" if signed else "unverified")}
            return InvocationContext("local", signed, values, {"repository": "legacy_preview", "profile": "legacy_signed_profile" if signed else "none"})


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", _freeze(self.policy) if self.policy is not None else None)
        object.__setattr__(self, "authority", _freeze(self.authority) if self.authority is not None else None)
        object.__setattr__(self, "allowlist", _FrozenList(_freeze(item) for item in self.allowlist) if self.allowlist is not None else None)
        object.__setattr__(self, "validation_argv", _FrozenList(_freeze(item) for item in self.validation_argv) if self.validation_argv is not None else None)
        object.__setattr__(self, "field_receipts", _freeze(self.field_receipts))
        if isinstance(self.continuation, str):
            object.__setattr__(self, "continuation", ResolutionContinuation(self.continuation, reason=self.reason or ""))

    def to_dict(self) -> dict[str, Any]:
        return {"evidence_fingerprint": self.evidence_fingerprint, "resolved": self.resolved,
            "launch_authorized": self.launch_authorized, "target": self.target, "profile": self.profile,
            "reason": self.reason, "prompt_hash": self.prompt_hash, "policy": _thaw(self.policy),
            "provider": self.provider, "caller": self.caller, "allowlist": _thaw(self.allowlist),
            "authority": _thaw(self.authority), "validation_argv": _thaw(self.validation_argv),
            "context_cid": self.context_cid, "field_receipts": _thaw(self.field_receipts),
            "continuation": self.continuation.as_dict() if self.continuation else None}

    def identity(self) -> str:
        return hashlib.sha256(_canonical(self.to_dict()).encode("utf-8")).hexdigest()

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
    """Compatibility probe; the adapter performs the authoritative check."""
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        return False
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and isinstance(data.get("signature"), str) and bool(data["signature"])


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
    signed = profile_signed is True if profile_signed is not None else bool(found and _looks_signed(found))
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
                 required_fields: Sequence[str] = REQUIRED_LAUNCH_FIELDS) -> None:
        names = tuple(str(name) for name in required_fields)
        if len(set(names)) != len(names) or not names or names[0] != "repository":
            raise ValueError("resolution pipeline requires ordered fields beginning with repository")
        self._resolvers = dict(resolvers or {})
        self.required_fields = names

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

    def resolve_fields(self, context: InvocationContext) -> tuple[FrozenMapping, Optional[ResolutionContinuation]]:
        if not isinstance(context, InvocationContext):
            raise TypeError("canonical resolution requires an InvocationContext")
        resolved: dict[str, ResolutionField] = {}
        failures: list[tuple[str, str]] = []
        for name in self.required_fields:
            field = context.field(name)
            if field.value is None and name in self._resolvers:
                try:
                    field = self._as_field(self._call(self._resolvers[name], context, FrozenMapping(resolved)))
                except Exception as exc:  # Leaf failures are a denial, never a launch escape.
                    field = ResolutionField(value=None, source="leaf_resolver", freshness="unverified", confidence="none")
                    failures.append((name, f"resolver failed: {type(exc).__name__}"))
            resolved[name] = field
            if (self.required_fields == REQUIRED_LAUNCH_FIELDS and name == "repository" and context.transport == "local"
                    and field.source != "verified_git_worktree"):
                failures.append((name, "unverified Git worktree"))
            elif field.freshness in _STALE or field.confidence == "none":
                failures.append((name, "stale evidence"))
            elif field.value is None:
                failures.append((name, "multiple candidates" if field.alternatives else "zero candidates"))
        if not context.authenticated:
            failures.insert(0, ("context", "unauthenticated invocation context"))
        if not failures:
            return FrozenMapping(resolved), None
        names = tuple(name for name, _ in failures)
        details = "; ".join(f"{name}: {reason}" for name, reason in failures)
        if any(reason == "multiple candidates" for _, reason in failures): kind = "multiple_evidence"
        elif any(reason == "stale evidence" for _, reason in failures): kind = "stale_evidence"
        elif failures[0][0] == "context": kind = "unauthenticated_context"
        elif any("failed" in reason for _, reason in failures): kind = "resolver_failure"
        else: kind = "zero_evidence"
        return FrozenMapping(resolved), ResolutionContinuation(kind, names, details)


class SupervisorResolutionService:
    """Deterministically resolve a launch gate from the canonical pipeline."""

    # The original facade only represented repository/profile.  Callers that
    # are about to execute effects use the complete default pipeline; the
    # compatibility facade remains useful for previews and existing clients.
    def __init__(self, pipeline: Optional[CanonicalResolutionPipeline] = None) -> None:
        self.pipeline = pipeline or CanonicalResolutionPipeline(required_fields=("repository",))

    def resolve(self, prompt: str, context: InvocationContext, *, trusted_bindings: Optional[Mapping[str, Any]] = None,
                explicit_target: Optional[str] = None, explicit_profile: Optional[str] = None) -> ResolutionReceipt:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not isinstance(context, InvocationContext):
            raise TypeError("resolution requires a frozen InvocationContext")
        trusted = dict(trusted_bindings or {})
        fields, continuation = self.pipeline.resolve_fields(context)
        repository = fields.get("repository", ResolutionField())
        profile = fields.get("profile", context.field("profile"))
        extra_reasons: list[str] = []
        if explicit_target is not None and repository.value is not None and str(explicit_target) != str(repository.value):
            extra_reasons.append("repository: conflicting explicit target")
        if explicit_profile is not None and profile.value is not None and os.path.abspath(str(explicit_profile)) != os.path.abspath(str(profile.value)):
            extra_reasons.append("profile: conflicting explicit profile")
        if extra_reasons:
            continuation = ResolutionContinuation("conflicting_evidence", ("repository", "profile"), "; ".join(extra_reasons))
        field_receipts: dict[str, Any] = {name: item.as_dict() for name, item in fields.items()}
        for name, value in trusted.items():
            if name in PROMPT_FORBIDDEN_FIELDS:
                field_receipts[name] = ResolutionField(value=value, source="trusted_binding").as_dict()
        allowed = continuation is None
        reason = "prompt-only resolution from trusted context" if allowed else continuation.reason
        return ResolutionReceipt(context.cid, allowed, allowed,
            target=str(repository.value) if repository.value is not None else None,
            profile=str(profile.value) if profile.value is not None else None, reason=reason, prompt_hash=_hash_prompt(prompt),
            policy=trusted.get("policy"), provider=trusted.get("provider"), caller=trusted.get("caller"), allowlist=trusted.get("allowlist"),
            authority=trusted.get("authority"), validation_argv=trusted.get("validation_argv"), context_cid=context.cid,
            field_receipts=field_receipts, continuation=continuation)


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
           "SupervisorResolutionService", "collect_ambient_evidence", "launch_if_authorized", "orchestrate", "resolve_prompt_only",
           "sanitize_prompt_bindings"]
