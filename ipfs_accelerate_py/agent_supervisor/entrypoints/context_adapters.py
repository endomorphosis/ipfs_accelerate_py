"""Trusted, transport-neutral invocation-context adapters.

Transport adapters are deliberately *collectors*, not resolvers.  They turn
facts already authenticated by their owning transport into an immutable core
and retain transport-specific authentication evidence in a separate envelope.
Request payloads, prompt text, and caller supplied path hints never establish
authority in this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


class InvocationContextError(ValueError):
    """An invocation context is malformed or crosses a trust boundary."""


_ALIAS = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,127}$")
_STALE = frozenset({"stale", "expired", "unverified"})


class FrozenMapping(Mapping[Any, Any]):
    """Small deterministic immutable mapping used in signed/hashed records."""

    __slots__ = ("_items", "_values")

    def __init__(self, items: Mapping[Any, Any] | Sequence[tuple[Any, Any]] = ()) -> None:
        source = items.items() if isinstance(items, Mapping) else items
        frozen = tuple(sorted(((_freeze(key), _freeze(value)) for key, value in source), key=lambda item: _canonical(item[0])))
        values: dict[Any, Any] = {}
        for key, value in frozen:
            try:
                if key in values:
                    raise InvocationContextError("canonical context contains duplicate mapping keys")
                values[key] = value
            except TypeError as exc:
                raise InvocationContextError("canonical context mapping keys must be hashable") from exc
        self._items = frozen
        self._values = values

    def __getitem__(self, key: Any) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def items(self) -> tuple[tuple[Any, Any], ...]:  # type: ignore[override]
        return self._items


def _canonical(value: Any) -> str:
    """Encode frozen data with type tags so mixed keys and sets are stable."""
    def encode(item: Any) -> Any:
        if isinstance(item, FrozenMapping):
            return {"@map": [[encode(key), encode(value)] for key, value in item.items()]}
        if isinstance(item, Mapping):
            return encode(_freeze(item))
        if isinstance(item, tuple):
            return {"@sequence": [encode(value) for value in item]}
        if isinstance(item, (set, frozenset, list)):
            return encode(_freeze(item))
        if item is None or isinstance(item, (bool, int, float, str)):
            return item
        return {"@repr": f"{type(item).__module__}.{type(item).__qualname__}", "value": str(item)}
    return json.dumps(encode(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _freeze(value: Any) -> Any:
    """Deep-freeze collection values before they can enter a receipt or CID."""
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=_canonical))
    return value


def _thaw(value: Any) -> Any:
    """Return a detached JSON-friendly projection; never expose frozen state."""
    if isinstance(value, FrozenMapping):
        return {_thaw(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _has_symlink_component(path: Path) -> bool:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        return absolute != absolute.resolve(strict=True)
    except OSError:
        return True


def _verified_signed_profile(path: str) -> Optional[str]:
    """Perform the minimum local installation checks before trusting a profile.

    Signature cryptography belongs to the profile verifier.  A boolean flag is
    never accepted as its substitute; this adapter at least requires a real,
    non-symlinked signed-profile document carrying a non-empty signature.
    """
    candidate = Path(path)
    if _has_symlink_component(candidate) or not candidate.is_file():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    signature = payload.get("signature") if isinstance(payload, Mapping) else None
    if not isinstance(signature, (str, bytes)) or not signature:
        return None
    return str(candidate.resolve(strict=True))


@dataclass(frozen=True)
class ResolutionField:
    """A resolved value and bounded provenance, frozen at construction time."""

    value: Any = None
    source: str = "unavailable"
    freshness: str = "fresh"
    alternatives: tuple[str, ...] = ()
    confidence: str = "high"

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _freeze(self.value))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "freshness", str(self.freshness).lower())
        object.__setattr__(self, "alternatives", tuple(sorted({str(value) for value in self.alternatives})))
        object.__setattr__(self, "confidence", "none" if self.freshness in _STALE else str(self.confidence).lower())

    def as_dict(self) -> dict[str, Any]:
        return {"value": _thaw(self.value), "source": self.source, "freshness": self.freshness,
                "alternatives": list(self.alternatives), "confidence": self.confidence}

    def core_dict(self) -> dict[str, Any]:
        """Transport-neutral fact used for cross-transport replay identity."""
        return {"value": _thaw(self.value), "freshness": self.freshness,
                "alternatives": list(self.alternatives), "confidence": self.confidence}


@dataclass(frozen=True)
class CanonicalResolutionCore:
    """The transport-independent, deeply frozen facts consumed by resolution."""

    fields: Mapping[str, ResolutionField] = field(default_factory=FrozenMapping)

    def __post_init__(self) -> None:
        values: dict[str, ResolutionField] = {}
        for name, item in self.fields.items():
            values[str(name)] = item if isinstance(item, ResolutionField) else ResolutionField(value=item)
        object.__setattr__(self, "fields", FrozenMapping(sorted(values.items())))

    def field(self, name: str) -> ResolutionField:
        return self.fields.get(name, ResolutionField())

    def to_dict(self) -> dict[str, Any]:
        return {name: item.core_dict() for name, item in self.fields.items()}

    @property
    def cid(self) -> str:
        return "sha256:" + hashlib.sha256(_canonical(self.to_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TransportEvidenceEnvelope:
    """Authenticated transport facts intentionally excluded from the core CID."""

    transport: str
    authenticated: bool
    provenance: Mapping[str, str] = field(default_factory=FrozenMapping)

    def __post_init__(self) -> None:
        object.__setattr__(self, "transport", str(self.transport))
        # Do not coerce truthy strings such as ``"true"`` into authority.
        object.__setattr__(self, "authenticated", self.authenticated is True)
        object.__setattr__(self, "provenance", FrozenMapping((str(key), str(value)) for key, value in self.provenance.items()))

    def to_dict(self) -> dict[str, Any]:
        return {"transport": self.transport, "authenticated": self.authenticated,
                "provenance": _thaw(self.provenance)}


@dataclass(frozen=True)
class InvocationContext:
    """Frozen core plus transport envelope, compatible with the original API."""

    transport: str
    authenticated: bool
    fields: Mapping[str, ResolutionField] = field(default_factory=FrozenMapping)
    provenance: Mapping[str, str] = field(default_factory=FrozenMapping)

    def __post_init__(self) -> None:
        core = CanonicalResolutionCore(self.fields)
        envelope = TransportEvidenceEnvelope(self.transport, self.authenticated, self.provenance)
        object.__setattr__(self, "transport", envelope.transport)
        object.__setattr__(self, "authenticated", envelope.authenticated)
        object.__setattr__(self, "fields", core.fields)
        object.__setattr__(self, "provenance", envelope.provenance)

    @property
    def core(self) -> CanonicalResolutionCore:
        return CanonicalResolutionCore(self.fields)

    @property
    def envelope(self) -> TransportEvidenceEnvelope:
        return TransportEvidenceEnvelope(self.transport, self.authenticated, self.provenance)

    def field(self, name: str) -> ResolutionField:
        return self.core.field(name)

    def to_dict(self) -> dict[str, Any]:
        return {"core": self.core.to_dict(), "envelope": self.envelope.to_dict()}

    @property
    def core_cid(self) -> str:
        return self.core.cid

    @property
    def cid(self) -> str:
        """CID of the core, deliberately stable across transport adapters."""
        return self.core_cid


# Clear public spelling for callers that want to avoid the legacy class name.
FrozenInvocationContext = InvocationContext


class TrustedEvidenceCollector:
    """Collect only already-authenticated facts into an immutable context."""

    def collect(self, *, transport: str, authenticated: bool,
                values: Optional[Mapping[str, Any]] = None,
                provenance: Optional[Mapping[str, str]] = None) -> InvocationContext:
        fields: dict[str, ResolutionField] = {}
        for name, raw in (values or {}).items():
            if isinstance(raw, ResolutionField):
                # Rebuild it so even a pre-built field cannot retain mutable state.
                fields[str(name)] = ResolutionField(**raw.as_dict())
            elif isinstance(raw, Mapping) and {"value", "source"}.intersection(raw):
                alternatives = raw.get("alternatives", ())
                if isinstance(alternatives, (str, bytes)):
                    raise InvocationContextError("resolution alternatives must be a sequence")
                fields[str(name)] = ResolutionField(value=raw.get("value"), source=raw.get("source", "authenticated_transport"),
                    freshness=raw.get("freshness", "fresh"), alternatives=tuple(alternatives), confidence=raw.get("confidence", "high"))
            else:
                fields[str(name)] = ResolutionField(value=raw, source="authenticated_transport")
        return InvocationContext(transport=transport, authenticated=authenticated, fields=fields, provenance=provenance or {})


class LocalInvocationContextFactory:
    """Bind local invocation to a real non-symlinked Git worktree and profile."""

    def create(self, *, cwd: Optional[str] = None, profile_path: Optional[str] = None,
               profile_signed: bool = False, values: Optional[Mapping[str, Any]] = None) -> InvocationContext:
        raw_cwd = Path(cwd or os.getcwd())
        if _has_symlink_component(raw_cwd):
            raise InvocationContextError("local invocation cwd must exist and contain no symlink component")
        repository = self._nearest_git_root(raw_cwd)
        if repository is None:
            raise InvocationContextError("local invocation cwd must be inside a real Git worktree")
        verified_profile = _verified_signed_profile(profile_path) if profile_path else None
        # ``profile_signed`` is only a cross-check from an upstream verifier;
        # it cannot make a missing, symlinked, or unsigned document trusted.
        profile_authenticated = profile_signed is True and verified_profile is not None
        payload = dict(values or {})
        real_worktree = self._is_real_git_worktree(raw_cwd)
        payload["repository"] = {"value": str(repository), "source": "verified_git_worktree" if real_worktree else "legacy_git_marker",
                                 "freshness": "fresh"}
        payload["profile"] = {"value": verified_profile, "source": "verified_installed_profile" if profile_authenticated else "unverified_profile",
                              "freshness": "fresh" if profile_authenticated else "unverified"}
        return TrustedEvidenceCollector().collect(transport="local", authenticated=profile_authenticated, values=payload,
            provenance={"repository": "git_rev_parse", "profile": "verified_installed_profile" if profile_authenticated else "none"})

    @staticmethod
    def _nearest_git_root(cwd: Path) -> Optional[Path]:
        try:
            completed = subprocess.run(("git", "-C", str(cwd), "rev-parse", "--show-toplevel", "--is-inside-work-tree"),
                capture_output=True, text=True, check=False, timeout=5)
            lines = completed.stdout.splitlines()
            if completed.returncode or len(lines) < 2 or lines[-1].strip().lower() != "true":
                marker = cwd / ".git"
                return cwd.resolve(strict=True) if marker.is_dir() and not marker.is_symlink() else None
            root = Path(lines[0].strip())
            if _has_symlink_component(root):
                return None
            return root.resolve(strict=True)
        except (OSError, subprocess.SubprocessError):
            # Kept solely for the long-standing in-process adapter contract:
            # a directory marker is an ambient *candidate*, never a portable
            # proof (the production complete pipeline rejects this source).
            marker = cwd / ".git"
            if marker.is_dir() and not marker.is_symlink():
                return cwd.resolve(strict=True)
            return None

    @staticmethod
    def _is_real_git_worktree(cwd: Path) -> bool:
        try:
            completed = subprocess.run(("git", "-C", str(cwd), "rev-parse", "--is-inside-work-tree"),
                capture_output=True, text=True, check=False, timeout=5)
            return completed.returncode == 0 and completed.stdout.strip().lower() == "true"
        except (OSError, subprocess.SubprocessError):
            return False


class PythonInvocationContextFactory:
    """Bind embedders to existing, non-symlinked configured allowlist roots."""

    def create(self, *, allowlisted_roots: Sequence[str], repository: Optional[str] = None,
               authenticated: bool = True, values: Optional[Mapping[str, Any]] = None) -> InvocationContext:
        roots: list[str] = []
        for root in allowlisted_roots:
            candidate = Path(root)
            if _has_symlink_component(candidate) or not candidate.is_dir():
                raise InvocationContextError("embedder allowlist roots must be real directories without symlinks")
            roots.append(str(candidate.resolve(strict=True)))
        roots = sorted(set(roots))
        if not roots:
            raise InvocationContextError("Python invocation requires a non-empty embedder allowlist")
        if repository is not None:
            candidate = Path(repository)
            if _has_symlink_component(candidate):
                raise InvocationContextError("client repository path contains a symlink")
            selected_path = str(candidate.resolve(strict=True))
            if selected_path not in roots:
                raise InvocationContextError("client repository is outside the embedder allowlist")
        else:
            selected_path = roots[0] if len(roots) == 1 else None
        payload = dict(values or {})
        payload["repository"] = {"value": selected_path, "source": "embedder_allowlist", "freshness": "fresh",
                                 "alternatives": () if selected_path is not None else roots}
        return TrustedEvidenceCollector().collect(transport="python", authenticated=authenticated is True, values=payload,
            provenance={"repository": "embedder_allowlist"})


class MCPInvocationContextFactory:
    """Bind MCP invocations to a server-owned alias, never client paths."""

    def create(self, *, target_alias: Optional[str], authenticated: bool,
               values: Optional[Mapping[str, Any]] = None) -> InvocationContext:
        if target_alias is not None and (not isinstance(target_alias, str) or not _ALIAS.fullmatch(target_alias)):
            raise InvocationContextError("server target alias is invalid")
        verified = authenticated is True
        payload = dict(values or {})
        payload["repository"] = {"value": target_alias, "source": "authenticated_server_alias" if verified else "unverified_server_alias",
                                 "freshness": "fresh" if verified else "unverified"}
        return TrustedEvidenceCollector().collect(transport="mcp", authenticated=verified, values=payload,
            provenance={"repository": "server_owned_alias"})


@dataclass(frozen=True)
class VerifiedUCAN:
    """Opaque result emitted by an MCP++ UCAN verifier, not a caller boolean."""

    audience: str
    capabilities: tuple[str, ...]
    signature: str

    def __post_init__(self) -> None:
        if (not _ALIAS.fullmatch(self.audience) or not self.signature or not self.capabilities
                or isinstance(self.capabilities, (str, bytes))):
            raise InvocationContextError("verified UCAN is malformed or lacks attenuation")
        object.__setattr__(self, "capabilities", tuple(sorted({str(item) for item in self.capabilities})))


class MCPPlusPlusInvocationContextFactory(MCPInvocationContextFactory):
    """MCP++ carries authority only when a verifier returns ``VerifiedUCAN``."""

    def create(self, *, target_alias: Optional[str], ucan_verified: Any,
               values: Optional[Mapping[str, Any]] = None) -> InvocationContext:
        verified = isinstance(ucan_verified, VerifiedUCAN)
        if verified and target_alias is not None and ucan_verified.audience != target_alias:
            raise InvocationContextError("verified UCAN audience does not match server alias")
        context = super().create(target_alias=target_alias, authenticated=verified, values=values)
        return InvocationContext(transport="mcp++", authenticated=context.authenticated, fields=context.fields,
            provenance={**_thaw(context.provenance), "authority": "verified_attenuated_ucan" if verified else "none"})


__all__ = ["CanonicalResolutionCore", "FrozenInvocationContext", "FrozenMapping", "InvocationContext", "InvocationContextError",
           "LocalInvocationContextFactory", "MCPInvocationContextFactory", "MCPPlusPlusInvocationContextFactory",
           "PythonInvocationContextFactory", "ResolutionField", "TransportEvidenceEnvelope", "TrustedEvidenceCollector", "VerifiedUCAN"]
