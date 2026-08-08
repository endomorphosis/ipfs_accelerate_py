"""Resolve prompt-only invocations from frozen trusted context.

The prompt is intentionally only hashed here.  It cannot select a repository,
principal, policy, provider, validation command, or authority.  This is the
last gate before an entrypoint may ask a lifecycle component to cause effects.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .context_adapters import InvocationContext, ResolutionField

PROMPT_FORBIDDEN_FIELDS = frozenset({"allowlist", "caller", "policy", "provider", "validation_argv", "authority"})


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
class AmbientEvidence:
    """Compatibility view of local/server evidence, convertible to a context."""

    cwd: str
    profile_path: Optional[str] = None
    profile_signed: bool = False
    server_authenticated: bool = False
    server_context: Optional[Mapping[str, Any]] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical({"cwd": self.cwd, "profile_path": self.profile_path,
            "profile_signed": self.profile_signed, "server_authenticated": self.server_authenticated,
            "server_context": dict(self.server_context or {}), "extra": dict(self.extra)}).encode()).hexdigest()

    def is_sufficient_for_prompt_only(self) -> bool:
        return bool(self.cwd) and ((bool(self.profile_path) and self.profile_signed) or
                                   (self.server_authenticated and self.server_context is not None))

    def invocation_context(self) -> InvocationContext:
        if self.server_context is not None:
            # This compatibility route mirrors MCP only when the caller has
            # independently marked its server context authenticated.
            target = self.server_context.get("target")
            fields = {"repository": ResolutionField(value=target, source="authenticated_transport",
                freshness="fresh" if self.server_authenticated else "unverified")}
            for key in ("profile", "run", "objective", "task_source", "resources", "validation", "topology"):
                if key in self.server_context:
                    fields[key] = ResolutionField(value=self.server_context[key], source="authenticated_transport")
            return InvocationContext("mcp", self.server_authenticated, fields, {"repository": "authenticated_server_context"})
        values: dict[str, Any] = {"repository": ResolutionField(value=self.extra.get("default_target") or self.cwd, source="ambient_local")}
        if self.profile_path:
            values["profile"] = ResolutionField(value=self.profile_path,
                source="signed_profile" if self.profile_signed else "unverified_profile",
                freshness="fresh" if self.profile_signed else "unverified")
        return InvocationContext("local", bool(self.profile_path and self.profile_signed), values,
                                 {"repository": "local_cwd", "profile": "installed_profile"})


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
    field_receipts: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    continuation: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {"evidence_fingerprint": self.evidence_fingerprint, "resolved": self.resolved,
            "launch_authorized": self.launch_authorized, "target": self.target, "profile": self.profile,
            "reason": self.reason, "prompt_hash": self.prompt_hash, "policy": dict(self.policy) if self.policy else None,
            "provider": self.provider, "caller": self.caller, "allowlist": list(self.allowlist) if self.allowlist else None,
            "authority": dict(self.authority) if self.authority else None,
            "validation_argv": list(self.validation_argv) if self.validation_argv else None,
            "context_cid": self.context_cid, "field_receipts": {k: dict(v) for k, v in sorted(self.field_receipts.items())},
            "continuation": self.continuation}

    def identity(self) -> str:
        return hashlib.sha256(_canonical(self.to_dict()).encode()).hexdigest()


def _default_profile_search_paths(cwd: str) -> list[str]:
    home = Path.home()
    return [str(Path(cwd) / ".agent-supervisor" / "profile.signed.json"), str(Path(cwd) / "profile.signed.json"),
            str(home / ".agent-supervisor" / "profile.signed.json"), str(home / ".config" / "agent-supervisor" / "profile.signed.json")]


def _looks_signed(path: str) -> bool:
    candidate = Path(path)
    if not candidate.is_file(): return False
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
        return isinstance(data, dict) and bool(data.get("signature") or data.get("signed") or data.get("sig"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False


def collect_ambient_evidence(*, cwd: Optional[str] = None, profile_path: Optional[str] = None,
    profile_signed: Optional[bool] = None, server_context: Optional[Mapping[str, Any]] = None,
    server_authenticated: Optional[bool] = None, profile_search_paths: Optional[Sequence[str]] = None,
    extra: Optional[Mapping[str, Any]] = None) -> AmbientEvidence:
    resolved_cwd = os.path.abspath(cwd or os.getcwd())
    found = os.path.abspath(profile_path) if profile_path else None
    if found is None:
        for candidate in profile_search_paths or _default_profile_search_paths(resolved_cwd):
            if Path(candidate).is_file():
                found = str(Path(candidate).resolve()); break
    signed = bool(profile_signed) if profile_signed is not None else bool(found and _looks_signed(found))
    context = dict(server_context) if server_context is not None else None
    authenticated = bool(server_authenticated) if server_authenticated is not None else bool(context and context.get("authenticated") is True)
    return AmbientEvidence(resolved_cwd, found, signed, authenticated, context, dict(extra or {}))


def sanitize_prompt_bindings(prompt: str, bindings: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    if not isinstance(prompt, str): raise TypeError("prompt must be a string")
    for key in (bindings or {}):
        if key.lower() in PROMPT_FORBIDDEN_FIELDS:
            raise PromptContaminationError(f"prompt bindings must not populate forbidden field: {key}")
    # Only parse structural assignments: ordinary prose mentioning policy is safe.
    lowered = prompt.lower()
    for name in PROMPT_FORBIDDEN_FIELDS:
        for marker in (f'"{name}":', f"'{name}':", f"{name}=", f"{name}:"):
            if marker in lowered:
                raise PromptContaminationError(f"prompt text must not populate forbidden field: {name}")
    return dict(bindings or {})


class SupervisorResolutionService:
    """Deterministically resolve a complete effect gate from one frozen context."""

    _REQUIRED = ("repository", "profile")

    def resolve(self, prompt: str, context: InvocationContext, *, trusted_bindings: Optional[Mapping[str, Any]] = None,
                explicit_target: Optional[str] = None, explicit_profile: Optional[str] = None) -> ResolutionReceipt:
        if not isinstance(prompt, str): raise TypeError("prompt must be a string")
        trusted = dict(trusted_bindings or {})
        fields = {name: item.as_dict() for name, item in context.fields.items()}
        reasons: list[str] = []
        if not context.authenticated:
            reasons.append("unauthenticated invocation context")
        repository = context.field("repository")
        profile = context.field("profile")
        if repository.freshness.lower() in {"stale", "expired", "unverified"}:
            reasons.append("repository observation is not fresh")
        if repository.value is None:
            if repository.alternatives: reasons.append("multiple repository targets")
            else: reasons.append("zero repository targets")
        if explicit_target is not None and repository.value is not None and str(explicit_target) != str(repository.value):
            reasons.append("explicit target conflicts with trusted context")
        if explicit_profile is not None and profile.value is not None and os.path.abspath(str(explicit_profile)) != os.path.abspath(str(profile.value)):
            reasons.append("explicit profile conflicts with trusted context")
        # A signed profile is required locally; authenticated servers may supply policy instead.
        if context.transport == "local" and (profile.value is None or profile.freshness != "fresh"):
            reasons.append("no verified installed profile")
        for name, value in trusted.items():
            if name in PROMPT_FORBIDDEN_FIELDS:
                fields[name] = {"value": value, "source": "trusted_binding", "freshness": "fresh", "alternatives": [], "confidence": "high"}
        allowed = not reasons
        target_value = repository.value if repository.value is not None else None
        return ResolutionReceipt(context.cid, allowed, allowed, target=str(target_value) if target_value is not None else None,
            profile=str(profile.value) if profile.value is not None else None,
            reason="prompt-only resolution from trusted context" if allowed else "; ".join(reasons), prompt_hash=_hash_prompt(prompt),
            policy=trusted.get("policy"), provider=trusted.get("provider"), caller=trusted.get("caller"), allowlist=trusted.get("allowlist"),
            authority=trusted.get("authority"), validation_argv=trusted.get("validation_argv"), context_cid=context.cid,
            field_receipts=fields, continuation=None if allowed else "resolve_trusted_context")


def resolve_prompt_only(prompt: str, evidence: AmbientEvidence, *, trusted_bindings: Optional[Mapping[str, Any]] = None,
    prompt_bindings: Optional[Mapping[str, Any]] = None, target: Optional[str] = None, profile: Optional[str] = None,
    require_no_low_level_flags: bool = True) -> ResolutionReceipt:
    del require_no_low_level_flags
    sanitize_prompt_bindings(prompt, prompt_bindings)
    return SupervisorResolutionService().resolve(prompt, evidence.invocation_context(), trusted_bindings=trusted_bindings,
        explicit_target=target, explicit_profile=profile)


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


__all__ = ["AmbientEvidence", "AmbientInferenceError", "MaterialAmbiguityError", "PROMPT_FORBIDDEN_FIELDS", "PromptContaminationError",
           "ResolutionReceipt", "SupervisorResolutionService", "collect_ambient_evidence", "launch_if_authorized", "orchestrate",
           "resolve_prompt_only", "sanitize_prompt_bindings"]
