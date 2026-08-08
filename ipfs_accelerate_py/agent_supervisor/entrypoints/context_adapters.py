"""Trusted, transport-neutral invocation context adapters.

This module is deliberately small: adapters collect facts that a transport has
already authenticated and turn them into one immutable, canonical context.
They do not treat request payloads or prompt text as authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


class InvocationContextError(ValueError):
    """An invocation context is malformed or attempts to cross trust bounds."""


_ALIAS = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,127}$")
_STALE = frozenset({"stale", "expired", "unverified"})


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple((str(k), _freeze(v)) for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, tuple):
        if all(isinstance(item, tuple) and len(item) == 2 for item in value):
            return {key: _thaw(item) for key, item in value}
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ResolutionField:
    """A resolved value together with bounded provenance evidence."""

    value: Any = None
    source: str = "unavailable"
    freshness: str = "fresh"
    alternatives: tuple[str, ...] = ()
    confidence: str = "high"

    def __post_init__(self) -> None:
        object.__setattr__(self, "alternatives", tuple(sorted({str(v) for v in self.alternatives})))
        if self.freshness.lower() in _STALE:
            object.__setattr__(self, "confidence", "none")

    def as_dict(self) -> dict[str, Any]:
        return {"value": _thaw(self.value), "source": self.source, "freshness": self.freshness,
                "alternatives": list(self.alternatives), "confidence": self.confidence}


@dataclass(frozen=True)
class InvocationContext:
    """Frozen trusted evidence consumed by all runtime transports."""

    transport: str
    authenticated: bool
    fields: Mapping[str, ResolutionField] = field(default_factory=dict)
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", {str(k): v if isinstance(v, ResolutionField) else ResolutionField(value=_freeze(v))
                                              for k, v in sorted(self.fields.items())})
        object.__setattr__(self, "provenance", {str(k): str(v) for k, v in sorted(self.provenance.items())})

    def field(self, name: str) -> ResolutionField:
        return self.fields.get(name, ResolutionField())

    def to_dict(self) -> dict[str, Any]:
        return {"transport": self.transport, "authenticated": self.authenticated,
                "fields": {key: item.as_dict() for key, item in self.fields.items()},
                "provenance": dict(self.provenance)}

    @property
    def cid(self) -> str:
        return "sha256:" + hashlib.sha256(_canonical(self.to_dict()).encode()).hexdigest()


class TrustedEvidenceCollector:
    """Collect only transport-authenticated facts, preserving deterministic order."""

    def collect(self, *, transport: str, authenticated: bool,
                values: Optional[Mapping[str, Any]] = None,
                provenance: Optional[Mapping[str, str]] = None) -> InvocationContext:
        values = values or {}
        provenance = provenance or {}
        fields: dict[str, ResolutionField] = {}
        for name, raw in values.items():
            if isinstance(raw, ResolutionField):
                fields[str(name)] = raw
            elif isinstance(raw, Mapping) and {"value", "source"}.intersection(raw):
                fields[str(name)] = ResolutionField(value=_freeze(raw.get("value")), source=str(raw.get("source", "authenticated_transport")),
                    freshness=str(raw.get("freshness", "fresh")), alternatives=tuple(raw.get("alternatives", ())), confidence=str(raw.get("confidence", "high")))
            else:
                fields[str(name)] = ResolutionField(value=_freeze(raw), source="authenticated_transport")
        return InvocationContext(transport=str(transport), authenticated=bool(authenticated), fields=fields, provenance=provenance)


class LocalInvocationContextFactory:
    """Bind a local CLI invocation to its ambient working directory and profile."""

    def create(self, *, cwd: Optional[str] = None, profile_path: Optional[str] = None,
               profile_signed: bool = False, values: Optional[Mapping[str, Any]] = None) -> InvocationContext:
        raw_cwd = Path(cwd or os.getcwd())
        if raw_cwd.is_symlink():
            raise InvocationContextError("local invocation cwd must not be a symlink")
        local_cwd = os.path.abspath(str(raw_cwd))
        repository = self._nearest_git_root(Path(local_cwd))
        profile = os.path.abspath(profile_path) if profile_path else None
        payload = dict(values or {})
        payload.setdefault("repository", {"value": str(repository) if repository else None,
            "source": "ambient_local", "freshness": "fresh"})
        if profile:
            payload.setdefault("profile", {"value": profile, "source": "signed_profile" if profile_signed else "unverified_profile", "freshness": "fresh" if profile_signed else "unverified"})
        return TrustedEvidenceCollector().collect(transport="local", authenticated=bool(profile and profile_signed), values=payload,
            provenance={"repository": "local_cwd", "profile": "verified_installed_profile" if profile_signed else "none"})

    @staticmethod
    def _nearest_git_root(cwd: Path) -> Optional[Path]:
        """Find the nearest physical Git root without accepting a symlink escape."""
        try:
            probe = cwd.resolve(strict=True)
        except OSError:
            return None
        for candidate in (probe, *probe.parents):
            marker = candidate / ".git"
            if marker.exists() and not marker.is_symlink():
                return candidate
        return None


class PythonInvocationContextFactory:
    """Bind embedders to a preconfigured allowlist, never a client path."""

    def create(self, *, allowlisted_roots: Sequence[str], repository: Optional[str] = None,
               authenticated: bool = True, values: Optional[Mapping[str, Any]] = None) -> InvocationContext:
        roots = tuple(sorted({str(Path(root).resolve()) for root in allowlisted_roots}))
        if not roots:
            raise InvocationContextError("Python invocation requires a non-empty embedder allowlist")
        if repository is not None and str(Path(repository).resolve()) not in roots:
            raise InvocationContextError("client repository is outside the embedder allowlist")
        selected = str(Path(repository).resolve()) if repository else (roots[0] if len(roots) == 1 else None)
        payload = dict(values or {})
        payload["repository"] = {"value": selected, "source": "embedder_allowlist", "alternatives": roots if selected is None else ()}
        return TrustedEvidenceCollector().collect(transport="python", authenticated=authenticated, values=payload,
            provenance={"repository": "embedder_allowlist"})


class MCPInvocationContextFactory:
    """Bind MCP invocations to a server-owned alias, not client filesystem input."""

    def create(self, *, target_alias: Optional[str], authenticated: bool,
               values: Optional[Mapping[str, Any]] = None) -> InvocationContext:
        if target_alias is not None and not _ALIAS.fullmatch(target_alias):
            raise InvocationContextError("server target alias is invalid")
        payload = dict(values or {})
        payload["repository"] = {"value": target_alias, "source": "authenticated_server_alias",
                                 "freshness": "fresh" if authenticated else "unverified"}
        return TrustedEvidenceCollector().collect(transport="mcp", authenticated=authenticated, values=payload,
            provenance={"repository": "server_owned_alias"})


class MCPPlusPlusInvocationContextFactory(MCPInvocationContextFactory):
    """MCP++ requires a verified UCAN before it can carry authority."""

    def create(self, *, target_alias: Optional[str], ucan_verified: bool,
               values: Optional[Mapping[str, Any]] = None) -> InvocationContext:
        context = super().create(target_alias=target_alias, authenticated=ucan_verified, values=values)
        return InvocationContext(transport="mcp++", authenticated=context.authenticated, fields=context.fields,
                                 provenance={**context.provenance, "authority": "verified_ucan" if ucan_verified else "none"})


__all__ = ["InvocationContext", "InvocationContextError", "LocalInvocationContextFactory", "MCPInvocationContextFactory",
           "MCPPlusPlusInvocationContextFactory", "PythonInvocationContextFactory", "ResolutionField", "TrustedEvidenceCollector"]
