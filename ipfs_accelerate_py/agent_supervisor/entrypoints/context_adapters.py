"""Transport-neutral invocation-context adapters.

Transport adapters are deliberately *collectors*, not resolvers.  They turn
transport observations into an immutable candidate core and retain metadata in
a separate envelope.  Request payloads, prompt text, caller booleans, adapter
configuration, and caller-supplied path hints never establish authority in this
module; launch-capable resolvers independently verify every leaf.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


class InvocationContextError(ValueError):
    """An invocation context is malformed or crosses a trust boundary."""


_ALIAS = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,127}$")
_STALE = frozenset({"stale", "expired", "unverified"})
LOCAL_ADAPTER_BINDING_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/local-adapter-binding@1"
)
LOCAL_ADAPTER_BINDING_TTL_NS = 5 * 60 * 1_000_000_000


class FrozenMapping(Mapping[Any, Any]):
    """Small deterministic immutable mapping used in signed/hashed records."""

    __slots__ = ("_items", "_values")

    def __init__(self, items: Mapping[Any, Any] | Sequence[tuple[Any, Any]] = ()) -> None:
        source = items.items() if isinstance(items, Mapping) else items
        frozen = tuple(sorted(((_freeze(key), _freeze(value)) for key, value in source), key=lambda item: _canonical(item[0])))
        values: dict[str, Any] = {}
        for key, value in frozen:
            try:
                identity = _canonical(key)
                if identity in values:
                    raise InvocationContextError("canonical context contains duplicate mapping keys")
                values[identity] = value
            except (TypeError, InvocationContextError) as exc:
                raise InvocationContextError("canonical context mapping keys must be hashable") from exc
        self._items = frozen
        self._values = values

    def __getitem__(self, key: Any) -> Any:
        return self._values[_canonical(_freeze(key))]

    def __iter__(self) -> Iterator[Any]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._values)

    def items(self) -> tuple[tuple[Any, Any], ...]:  # type: ignore[override]
        return self._items


def _canonical(value: Any) -> str:
    """Encode frozen data with type tags so mixed keys and sets are stable."""
    def encode(item: Any) -> Any:
        if isinstance(item, ResolutionField):
            return {"@resolution_field": encode(item.core_dict())}
        if isinstance(item, FrozenMapping):
            return {"@map": [[encode(key), encode(value)] for key, value in item.items()]}
        if isinstance(item, Mapping):
            return encode(_freeze(item))
        if isinstance(item, tuple):
            return {"@sequence": [encode(value) for value in item]}
        if isinstance(item, (set, frozenset, list)):
            return encode(_freeze(item))
        # JSON's native encoding is not a canonical representation for Python
        # values (notably ``True == 1`` and ``1 == 1.0``).  Receipt identity
        # must not depend on whichever type happened to arrive first.
        if item is None:
            return {"@none": True}
        if isinstance(item, bool):
            return {"@bool": item}
        if isinstance(item, int):
            return {"@int": str(item)}
        if isinstance(item, float):
            if item != item or item in (float("inf"), float("-inf")):
                raise InvocationContextError("non-finite floats cannot enter a canonical context")
            return {"@float": item.hex()}
        if isinstance(item, str):
            return {"@str": item}
        # Never hash a repr containing an address.  A caller must reduce an
        # opaque object to a signed/typed receipt before it crosses this
        # boundary.
        raise InvocationContextError(
            f"unsupported value in canonical context: {type(item).__module__}.{type(item).__qualname__}"
        )
    return json.dumps(encode(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _freeze(value: Any) -> Any:
    """Deep-freeze collection values before they can enter a receipt or CID."""
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, ResolutionField):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        # Keep the historical human-facing string-first projection while the
        # canonical key still distinguishes scalar types.
        return tuple(sorted(
            (_freeze(item) for item in value),
            key=lambda item: (0 if isinstance(item, str) else 1, _canonical(item)),
        ))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise InvocationContextError(
        f"unsupported value in invocation context: {type(value).__module__}.{type(value).__qualname__}"
    )


def _thaw(value: Any) -> Any:
    """Return a detached JSON-friendly projection; never expose frozen state."""
    if isinstance(value, FrozenMapping):
        items = [(_thaw(key), _thaw(item)) for key, item in value.items()]
        # Keep the ordinary projection pleasant for the overwhelmingly common
        # string-key case.  Python dicts cannot faithfully represent mixed
        # keys such as ``True`` and ``1`` (or collection-valued keys), so those
        # use the same explicit map envelope as the canonical encoder.
        if all(isinstance(key, str) for key, _ in items):
            return {key: item for key, item in items}
        return {"@map": [[key, item] for key, item in items]}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _has_symlink_component(path: Path) -> bool:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        return absolute != absolute.resolve(strict=True)
    except OSError:
        return True


def _verified_installed_profile(repository_id: str) -> Any | None:
    """Ask the owning profile lifecycle to verify its installed profile.

    This module intentionally knows neither a signature wire format nor a key
    location.  Those are lifecycle policy and were already implemented by
    :mod:`local_profile`; duplicating them here previously made a second,
    attacker-selectable profile format look authoritative.
    """
    try:
        from .local_profile import (
            DEFAULT_PROFILE_DIR_ENV,
            KEY_FILENAME,
            PROFILE_FILENAME,
            SIGNATURE_FILENAME,
            SIGNING_KEY_ENV,
            LocalProfileInitializer,
        )

        directory = Path(os.environ.get(
            DEFAULT_PROFILE_DIR_ENV,
            Path.home() / ".ipfs_accelerate" / "agent_supervisor" / "local_profile",
        ))
        profile_file = directory / PROFILE_FILENAME
        signature_file = directory / SIGNATURE_FILENAME
        key_file = directory / KEY_FILENAME
        required = (directory, profile_file, signature_file)
        if key_file.exists() or not os.environ.get(SIGNING_KEY_ENV):
            required = (*required, key_file)
        if any(_has_symlink_component(item) for item in required):
            return None
        if not directory.is_dir() or any(not item.is_file() for item in required[1:]):
            return None

        return LocalProfileInitializer.verify(
            repository_cid=repository_id,
            source="local_transport_receipt_verifier",
        )
    except (ImportError, OSError, ValueError):
        return None


def _installed_profile_directory() -> Path | None:
    """Return the lifecycle-selected profile directory only when non-symlinked."""
    try:
        from .local_profile import DEFAULT_PROFILE_DIR_ENV

        directory = Path(os.environ.get(
            DEFAULT_PROFILE_DIR_ENV,
            Path.home() / ".ipfs_accelerate" / "agent_supervisor" / "local_profile",
        ))
        if _has_symlink_component(directory) or not directory.is_dir():
            return None
        return directory.resolve(strict=True)
    except (ImportError, OSError, RuntimeError):
        return None


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
    """Transport metadata and signed evidence excluded from the core CID."""

    transport: str
    authenticated: bool
    provenance: Mapping[str, str] = field(default_factory=FrozenMapping)
    adapter_receipt: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "transport", str(self.transport))
        # Do not coerce truthy strings such as ``"true"`` into authority.
        object.__setattr__(self, "authenticated", self.authenticated is True)
        object.__setattr__(self, "provenance", FrozenMapping((str(key), str(value)) for key, value in self.provenance.items()))
        object.__setattr__(self, "adapter_receipt",
                           _freeze(self.adapter_receipt) if self.adapter_receipt is not None else None)

    def to_dict(self) -> dict[str, Any]:
        return {"transport": self.transport, "authenticated": self.authenticated,
                "provenance": _thaw(self.provenance),
                "adapter_receipt": _thaw(self.adapter_receipt)}


@dataclass(frozen=True)
class InvocationContext:
    """Frozen candidates plus transport metadata.

    This public compatibility type deliberately contains no private token or
    process-local seal.  Its optional adapter receipt is public signed evidence
    whose exact core/profile/lifecycle binding is independently revalidated;
    all ordinary values remain candidates.  Python has no private module
    boundary, so underscore naming never creates authority.
    """

    transport: str
    authenticated: bool
    fields: Mapping[str, ResolutionField] = field(default_factory=FrozenMapping)
    provenance: Mapping[str, str] = field(default_factory=FrozenMapping)
    adapter_receipt: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        core = CanonicalResolutionCore(self.fields)
        envelope = TransportEvidenceEnvelope(
            self.transport, self.authenticated, self.provenance,
            self.adapter_receipt,
        )
        object.__setattr__(self, "transport", envelope.transport)
        object.__setattr__(self, "authenticated", envelope.authenticated)
        object.__setattr__(self, "fields", core.fields)
        object.__setattr__(self, "provenance", envelope.provenance)
        object.__setattr__(self, "adapter_receipt", envelope.adapter_receipt)

    @property
    def core(self) -> CanonicalResolutionCore:
        return CanonicalResolutionCore(self.fields)

    @property
    def envelope(self) -> TransportEvidenceEnvelope:
        return TransportEvidenceEnvelope(
            self.transport, self.authenticated, self.provenance,
            self.adapter_receipt,
        )

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


def _configured_capability_evidence(value: Optional[Mapping[str, Any]]) -> Any | None:
    """Freeze an adapter-level capability candidate separately from requests."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise InvocationContextError("configured capability evidence must be a mapping")
    return _freeze(value)


class TrustedEvidenceCollector:
    """Collect immutable candidates without assigning them launch authority."""

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
    """Bind a real Git worktree to the active profile's Ed25519 lifecycle."""

    def __init__(self, *, capability_evidence: Optional[Mapping[str, Any]] = None) -> None:
        # Retained as a candidate-only compatibility input.  The production
        # resolver owns the authoritative capability snapshot separately; an
        # adapter instance is ordinary caller-constructible Python state.
        self._capability_evidence = _configured_capability_evidence(capability_evidence)

    def create(self, *, cwd: Optional[str] = None, profile_path: Optional[str] = None,
               profile_signed: bool = False, values: Optional[Mapping[str, Any]] = None) -> InvocationContext:
        raw_cwd = Path(cwd or os.getcwd())
        if _has_symlink_component(raw_cwd):
            raise InvocationContextError("local invocation cwd must exist and contain no symlink component")
        repository = self._nearest_git_root(raw_cwd)
        if repository is None:
            raise InvocationContextError("local invocation cwd must be inside a real Git worktree")
        try:
            from .target_resolver import RepositoryTargetEvidence, resolve_repository_target

            target = resolve_repository_target(
                RepositoryTargetEvidence(
                    cwd=str(raw_cwd),
                    allowlisted_roots=(str(repository),),
                )
            )
        except (ImportError, OSError, ValueError):
            target = None

        binding = target.binding if target is not None and target.unique else None
        verified_profile = (
            _verified_installed_profile(binding.repository_id)
            if binding is not None
            else None
        )
        # Both legacy inputs are caller claims.  ``profile_path`` is retained
        # only as a candidate for diagnostics and ``profile_signed`` is
        # ignored; neither selects a key or a profile directory.  The owning
        # lifecycle's installed profile is the sole authentication source.
        del profile_signed
        payload = dict(values or {})
        real_worktree = binding is not None and self._is_real_git_worktree(raw_cwd)
        payload["repository"] = {"value": str(repository), "source": "verified_git_worktree" if real_worktree else "legacy_git_marker",
                                 "freshness": "fresh"}
        payload["profile"] = {"value": verified_profile.content_id if verified_profile is not None else profile_path,
                              "source": "local_profile_lifecycle" if verified_profile is not None else "unverified_profile_candidate",
                              "freshness": "fresh" if verified_profile is not None else "unverified"}
        payload["repository_allowlist"] = {
            "value": (str(repository),) if real_worktree else (),
            "source": "local_git_candidate" if real_worktree else "unavailable",
            "freshness": "fresh" if real_worktree else "unverified",
        }
        if self._capability_evidence is not None:
            payload["capability_evidence"] = self._capability_evidence
        candidate = TrustedEvidenceCollector().collect(
            transport="local",
            authenticated=False,
            values=payload,
            provenance={
                "repository": "target_resolver" if real_worktree else "none",
                "profile": "local_profile_lifecycle" if verified_profile is not None else "none",
                "capability_evidence": (
                    "adapter_candidate_snapshot"
                    if self._capability_evidence is not None else "none"
                ),
            },
        )
        adapter_receipt: Mapping[str, Any] | None = None
        if binding is not None and real_worktree and verified_profile is not None:
            try:
                from .local_profile import sign_profile_binding

                profile_directory = _installed_profile_directory()
                generation = verified_profile.lifecycle_generation
                identity_did = verified_profile.identity_did
                anchor_id = verified_profile.lifecycle_anchor_id
                if (
                    profile_directory is None
                    or not isinstance(generation, int)
                    or isinstance(generation, bool)
                    or generation < 1
                    or not isinstance(identity_did, str)
                    or not identity_did
                    or not isinstance(anchor_id, str)
                    or not anchor_id
                ):
                    raise InvocationContextError(
                        "installed profile lacks an anchored Ed25519 lifecycle"
                    )
                issued_at_ns = time.time_ns()
                body = {
                    "schema": LOCAL_ADAPTER_BINDING_SCHEMA,
                    "transport": "local",
                    "core_cid": candidate.core_cid,
                    "repository_id": binding.repository_id,
                    "repository_root": binding.repository_root,
                    "checkout_id": binding.checkout_id,
                    "profile_id": verified_profile.profile_id,
                    "profile_cid": verified_profile.content_id,
                    "identity_did": identity_did,
                    "lifecycle_generation": generation,
                    "lifecycle_anchor_id": anchor_id,
                    "issued_at_ns": issued_at_ns,
                    "expires_at_ns": issued_at_ns + LOCAL_ADAPTER_BINDING_TTL_NS,
                    "nonce": os.urandom(16).hex(),
                }
                signed = sign_profile_binding(
                    profile_dir=profile_directory,
                    payload=body,
                )
                if (
                    not isinstance(signed, Mapping)
                    or signed.get("identity") != identity_did
                    or signed.get("profile_id") != verified_profile.profile_id
                    or not isinstance(signed.get("signature"), str)
                    or not signed["signature"]
                ):
                    raise InvocationContextError(
                        "local lifecycle signed with a different active profile"
                    )
                adapter_receipt = {**body, "signature": signed["signature"]}
            except (AttributeError, ImportError, OSError, TypeError, ValueError):
                adapter_receipt = None
        return InvocationContext(
            transport="local",
            authenticated=adapter_receipt is not None,
            fields=candidate.fields,
            provenance=candidate.provenance,
            adapter_receipt=adapter_receipt,
        )

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

    def __init__(self, *, capability_evidence: Optional[Mapping[str, Any]] = None) -> None:
        self._capability_evidence = _configured_capability_evidence(capability_evidence)

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
        payload["repository_allowlist"] = {
            "value": roots,
            "source": "embedder_allowlist",
            "freshness": "fresh",
        }
        if self._capability_evidence is not None:
            payload["capability_evidence"] = self._capability_evidence
        return TrustedEvidenceCollector().collect(
            transport="python", authenticated=authenticated is True, values=payload,
            provenance={
                "repository": "embedder_allowlist",
                "capability_evidence": (
                    "embedder_candidate_snapshot"
                    if self._capability_evidence is not None else "none"
                ),
            },
        )


class MCPInvocationContextFactory:
    """Bind MCP invocations to a server-owned alias, never client paths."""

    def __init__(self, *, repository_aliases: Optional[Mapping[str, str]] = None,
                 capability_evidence: Optional[Mapping[str, Any]] = None) -> None:
        aliases: dict[str, str] = {}
        for alias, raw_path in (repository_aliases or {}).items():
            if not isinstance(alias, str) or not _ALIAS.fullmatch(alias):
                raise InvocationContextError("server repository alias is invalid")
            candidate = Path(raw_path)
            if _has_symlink_component(candidate) or not candidate.is_dir():
                raise InvocationContextError("server repository aliases must bind real non-symlinked directories")
            aliases[alias] = str(candidate.resolve(strict=True))
        self._repository_aliases = aliases
        self._capability_evidence = _configured_capability_evidence(capability_evidence)

    def create(self, *, target_alias: Optional[str], authenticated: bool,
               values: Optional[Mapping[str, Any]] = None) -> InvocationContext:
        if target_alias is not None and (not isinstance(target_alias, str) or not _ALIAS.fullmatch(target_alias)):
            raise InvocationContextError("server target alias is invalid")
        verified = authenticated is True
        payload = dict(values or {})
        repository = self._repository_aliases.get(target_alias or "")
        alias_verified = verified and repository is not None
        payload["repository_alias"] = {
            "value": target_alias,
            "source": "mcp_request_candidate",
            "freshness": "fresh" if target_alias else "unverified",
        }
        payload["repository"] = {
            "value": repository,
            "source": "authenticated_server_alias" if alias_verified else "unresolved_server_alias",
            "freshness": "fresh" if alias_verified else "unverified",
            "alternatives": () if repository is not None else ((target_alias,) if target_alias else ()),
        }
        payload["repository_allowlist"] = {
            "value": (repository,) if alias_verified else (),
            "source": "adapter_alias_candidate" if alias_verified else "unavailable",
            "freshness": "fresh" if alias_verified else "unverified",
        }
        if self._capability_evidence is not None:
            payload["capability_evidence"] = self._capability_evidence
        return TrustedEvidenceCollector().collect(
            transport="mcp", authenticated=verified, values=payload,
            provenance={
                "repository": "server_owned_alias" if alias_verified else "none",
                "capability_evidence": (
                    "server_adapter_candidate_snapshot"
                    if self._capability_evidence is not None else "none"
                ),
            },
        )


class MCPPlusPlusInvocationContextFactory(MCPInvocationContextFactory):
    """Preview an attenuated signed delegation chain at the MCP++ boundary.

    The caller supplies a raw chain, never a boolean or a constructible
    ``VerifiedUCAN`` wrapper.  Adapter keys produce diagnostic metadata only;
    the production resolver independently verifies the raw chain against its
    own trust anchors and revocation snapshot before granting authority.
    """

    def __init__(self, *, repository_aliases: Optional[Mapping[str, str]] = None,
                 issuer_public_keys: Optional[Mapping[str, str]] = None,
                 revoked_proof_cids: Sequence[str] = (),
                 ability: str = "agent-supervisor/invoke",
                 capability_evidence: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__(
            repository_aliases=repository_aliases,
            capability_evidence=capability_evidence,
        )
        self._issuer_public_keys = dict(issuer_public_keys or {})
        self._revoked_proof_cids = tuple(str(item) for item in revoked_proof_cids)
        self._ability = str(ability)

    def create(self, *, target_alias: Optional[str], ucan_verified: Any,
               values: Optional[Mapping[str, Any]] = None) -> InvocationContext:
        verified = False
        lineage: Sequence[str] = ()
        if (isinstance(ucan_verified, Sequence)
                and not isinstance(ucan_verified, (str, bytes, bytearray))
                and all(isinstance(item, Mapping) for item in ucan_verified)):
            try:
                from ipfs_accelerate_py.mcp_server.mcplusplus.delegation import (
                    parse_delegation_chain,
                    validate_raw_delegation_chain,
                )

                verdict = validate_raw_delegation_chain(
                    raw_chain=ucan_verified,
                    resource=str(target_alias or ""),
                    ability=self._ability,
                    actor=str(target_alias or ""),
                    require_signatures=True,
                    issuer_public_keys=self._issuer_public_keys,
                    revoked_proof_cids=self._revoked_proof_cids,
                )
                parsed = parse_delegation_chain(ucan_verified)
                leaf_caps = parsed[-1].capabilities if parsed else ()
                cryptographically_signed = bool(parsed) and all(
                    bool(self._issuer_public_keys.get(item.issuer))
                    and (
                        item.signature.startswith("ed25519:")
                        or item.signature.startswith("ed25519-hex:")
                        or item.signature.startswith("hex:")
                        or bool(re.fullmatch(r"[0-9A-Fa-f]{128}", item.signature))
                    )
                    for item in parsed
                )
                exactly_attenuated = bool(leaf_caps) and all(
                    cap.resource == str(target_alias or "")
                    and cap.ability == self._ability
                    and "*" not in (cap.resource, cap.ability)
                    for cap in leaf_caps
                )
                verified = (
                    verdict.allowed
                    and cryptographically_signed
                    and exactly_attenuated
                )
                lineage = tuple(verdict.proof_lineage)
            except (ImportError, TypeError, ValueError):
                verified = False
        payload = dict(values or {})
        if (isinstance(ucan_verified, Sequence)
                and not isinstance(ucan_verified, (str, bytes, bytearray))
                and all(isinstance(item, Mapping) for item in ucan_verified)):
            # Raw signed material remains a candidate.  Launch-capable
            # resolution re-verifies it against its independently owned key
            # map and revocation snapshot; this adapter's preview verdict can
            # never grant authority.
            payload["ucan_delegation_chain"] = tuple(ucan_verified)
        context = super().create(
            target_alias=target_alias,
            authenticated=verified,
            values=payload,
        )
        provenance = {
            **_thaw(context.provenance),
            "authority": "mcpplusplus_adapter_preview" if verified else "none",
            "ucan_proof_lineage": (
                "sha256:" + hashlib.sha256(_canonical(tuple(lineage)).encode()).hexdigest()
                if lineage else "none"
            ),
        }
        return InvocationContext(
            transport="mcp++",
            authenticated=context.authenticated,
            fields=context.fields,
            provenance=provenance,
        )


__all__ = ["CanonicalResolutionCore", "FrozenInvocationContext", "FrozenMapping", "InvocationContext", "InvocationContextError",
           "LOCAL_ADAPTER_BINDING_SCHEMA", "LOCAL_ADAPTER_BINDING_TTL_NS",
           "LocalInvocationContextFactory", "MCPInvocationContextFactory", "MCPPlusPlusInvocationContextFactory",
           "PythonInvocationContextFactory", "ResolutionField", "TransportEvidenceEnvelope", "TrustedEvidenceCollector"]
