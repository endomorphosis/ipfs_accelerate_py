"""Tiered, dependency-aware content-addressed storage for supervisor results.

The runtime store is intentionally a coordination layer, not a new authority
source.  Immutable result envelopes retain the exact generation-2
``ResultBinding`` that produced them, including the complete semantic
dependency population and producer, policy, and capability revisions.

Four storage tiers are modelled explicitly:

* a process-local immutable object cache;
* a host-durable, integrity checked object store;
* an optional shared immutable byte store; and
* mutable, current-tree-bound projections which may point only at fresh
  authoritative records.

Invalidation tombstones live beside the durable store.  Content blobs remain
immutable and may be retained, but a tombstoned result and every transitive
dependent are no longer reusable through either an exact key or projection.
Unrelated ancestors, siblings, and dependency components are untouched.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Final, Protocol, runtime_checkable

from .supervisor_v2_contracts import (
    EvidenceFreshness,
    ResultBinding,
    SemanticDependencyIdentity,
)


RUNTIME_CAS_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/runtime-cas@1"
RUNTIME_ARTIFACT_IDENTITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-artifact-identity@1"
)
RUNTIME_ARTIFACT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-artifact@1"
)
RUNTIME_ARTIFACT_KEY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-artifact-key@1"
)
RUNTIME_DEPENDENCY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-dependency@1"
)
RUNTIME_PROJECTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-projection@1"
)
RUNTIME_INVALIDATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-invalidation@1"
)
RUNTIME_INVALIDATION_TRANSACTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-invalidation-transaction@1"
)
RUNTIME_CAS_AUDIT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-cas-audit@1"
)
DEPENDENCY_CAS_REQUIREMENT_ID: Final = (
    "asi-100:tiered-dependency-aware-content-addressed-runtime-store"
)
DEFAULT_MAX_PAYLOAD_BYTES: Final = 64 * 1024 * 1024
DEFAULT_LOCK_TIMEOUT_SECONDS: Final = 30.0


class RuntimeCASError(RuntimeError):
    """Base error for malformed or unsafe runtime-CAS operations."""


class ArtifactIntegrityError(RuntimeCASError, ValueError):
    """An immutable artifact or projection failed canonical verification."""


class ForgedDependencyError(ArtifactIntegrityError):
    """A dependency claim did not match the referenced immutable record."""


class DependencyCycleError(RuntimeCASError, ValueError):
    """A dependency edge would introduce a cycle."""


class AuthorityIsolationError(RuntimeCASError, ValueError):
    """An operation attempted to merge incompatible authority classes."""


class ImmutableStoreError(RuntimeCASError):
    """A shared immutable store attempted to change existing content."""


class RuntimeTier(str, Enum):
    """Physical or logical tier from which a result was obtained."""

    PROCESS_LOCAL = "process_local"
    HOST_DURABLE = "host_durable"
    SHARED_IMMUTABLE = "shared_immutable"
    AUTHORITATIVE_PROJECTION = "authoritative_projection"

    # Concise compatibility spellings.
    PROCESS = "process_local"
    HOST = "host_durable"
    SHARED = "shared_immutable"
    PROJECTION = "authoritative_projection"


StorageTier = RuntimeTier
ArtifactTier = RuntimeTier


class RuntimeAuthority(str, Enum):
    """Closed namespace authority vocabulary.

    Values are classes, not a ranking.  In particular a draft cannot be
    upgraded by looking it up through an authoritative namespace.
    """

    AUTHORITATIVE = "authoritative"
    DIAGNOSTIC = "diagnostic"
    PROPOSAL = "proposal"
    DRAFT = "draft"

    RECEIPT = "authoritative"


ArtifactAuthority = RuntimeAuthority
NamespaceAuthority = RuntimeAuthority
ArtifactFreshness = EvidenceFreshness


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeCASError(f"{name} is required")
    result = value.strip()
    if "\x00" in result:
        raise RuntimeCASError(f"{name} must not contain NUL")
    return result


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ArtifactIntegrityError(
                "canonical JSON cannot contain NaN or infinity"
            )
        return value
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ArtifactIntegrityError(
                "canonical JSON object keys must be strings"
            )
        return {
            key: _canonical_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        return _canonical_value(converter())
    raise ArtifactIntegrityError(
        f"unsupported canonical JSON value: {type(value).__name__}"
    )


def canonical_runtime_json_bytes(value: Any) -> bytes:
    """Return deterministic JSON bytes for runtime identities and envelopes."""

    try:
        return json.dumps(
            _canonical_value(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ArtifactIntegrityError(
            "runtime artifact must contain canonical JSON values"
        ) from exc


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _coerce_authority(value: RuntimeAuthority | str) -> RuntimeAuthority:
    if isinstance(value, RuntimeAuthority):
        return value
    try:
        return RuntimeAuthority(str(value))
    except ValueError as exc:
        raise AuthorityIsolationError(
            "authority must be authoritative, diagnostic, proposal, or draft"
        ) from exc


def _coerce_freshness(value: EvidenceFreshness | str) -> EvidenceFreshness:
    if isinstance(value, EvidenceFreshness):
        return value
    try:
        return EvidenceFreshness(str(value))
    except ValueError as exc:
        raise ArtifactIntegrityError(
            "freshness must be fresh, stale, or unknown"
        ) from exc


def _coerce_binding(value: ResultBinding | Mapping[str, Any]) -> ResultBinding:
    if isinstance(value, ResultBinding):
        return value
    if isinstance(value, Mapping):
        return ResultBinding.from_dict(value)
    raise ArtifactIntegrityError("binding must be a ResultBinding")


@dataclass(frozen=True)
class ArtifactDependency:
    """Identity-bound edge from one immutable dependency to its dependent."""

    artifact_id: str
    namespace: str
    authority: RuntimeAuthority
    payload_digest: str
    binding_id: str
    schema: str = RUNTIME_DEPENDENCY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RUNTIME_DEPENDENCY_SCHEMA:
            raise ForgedDependencyError("unsupported runtime dependency schema")
        for name in (
            "artifact_id",
            "namespace",
            "payload_digest",
            "binding_id",
        ):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), name)
            )
        object.__setattr__(
            self, "authority", _coerce_authority(self.authority)
        )
        if not self.artifact_id.startswith("runtime-artifact:sha256:"):
            raise ForgedDependencyError("dependency artifact_id is not canonical")
        if not self.payload_digest.startswith("sha256:"):
            raise ForgedDependencyError(
                "dependency payload_digest is not canonical"
            )

    @classmethod
    def from_artifact(
        cls, artifact: "RuntimeArtifactRecord"
    ) -> "ArtifactDependency":
        return cls(
            artifact_id=artifact.artifact_id,
            namespace=artifact.identity.namespace,
            authority=artifact.identity.authority,
            payload_digest=artifact.identity.payload_digest,
            binding_id=artifact.binding.binding_id,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactDependency":
        if not isinstance(value, Mapping):
            raise ForgedDependencyError("dependency must be an object")
        allowed = {
            "schema",
            "artifact_id",
            "namespace",
            "authority",
            "payload_digest",
            "binding_id",
        }
        if set(value).difference(allowed):
            raise ForgedDependencyError("dependency contains unknown fields")
        return cls(
            schema=str(value.get("schema") or RUNTIME_DEPENDENCY_SCHEMA),
            artifact_id=value.get("artifact_id", ""),
            namespace=value.get("namespace", ""),
            authority=value.get("authority", ""),
            payload_digest=value.get("payload_digest", ""),
            binding_id=value.get("binding_id", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "artifact_id": self.artifact_id,
            "namespace": self.namespace,
            "authority": self.authority.value,
            "payload_digest": self.payload_digest,
            "binding_id": self.binding_id,
        }


DependencyReference = ArtifactDependency
DependencyEdge = ArtifactDependency


@dataclass(frozen=True)
class RuntimeArtifactKey:
    """Exact computation key used to find an already-produced artifact."""

    namespace: str
    artifact_kind: str
    authority: RuntimeAuthority
    binding: ResultBinding
    dependency_ids: tuple[str, ...] = ()
    payload_schema: str = ""
    schema: str = RUNTIME_ARTIFACT_KEY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RUNTIME_ARTIFACT_KEY_SCHEMA:
            raise ArtifactIntegrityError("unsupported runtime artifact key schema")
        object.__setattr__(
            self, "namespace", _required_text(self.namespace, "namespace")
        )
        object.__setattr__(
            self,
            "artifact_kind",
            _required_text(self.artifact_kind, "artifact_kind"),
        )
        object.__setattr__(
            self, "authority", _coerce_authority(self.authority)
        )
        object.__setattr__(self, "binding", _coerce_binding(self.binding))
        ids = tuple(
            sorted(
                {
                    _required_text(item, "dependency_id")
                    for item in self.dependency_ids
                }
            )
        )
        if len(ids) != len(tuple(self.dependency_ids)):
            raise ArtifactIntegrityError(
                "runtime artifact key contains duplicate dependencies"
            )
        object.__setattr__(self, "dependency_ids", ids)
        if self.payload_schema:
            object.__setattr__(
                self,
                "payload_schema",
                _required_text(self.payload_schema, "payload_schema"),
            )

    def _content(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "namespace": self.namespace,
            "artifact_kind": self.artifact_kind,
            "authority": self.authority.value,
            "binding_id": self.binding.binding_id,
            "semantic_dependency_ids": list(
                self.binding.semantic_dependency_ids
            ),
            "dependency_ids": list(self.dependency_ids),
            "payload_schema": self.payload_schema,
        }

    @property
    def key_id(self) -> str:
        return "runtime-key:sha256:" + hashlib.sha256(
            canonical_runtime_json_bytes(self._content())
        ).hexdigest()

    @property
    def semantic_key(self) -> str:
        return self.key_id

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._content(),
            "binding": self.binding.to_dict(),
            "key_id": self.key_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeArtifactKey":
        if not isinstance(value, Mapping):
            raise ArtifactIntegrityError("runtime artifact key must be an object")
        allowed = {
            "schema",
            "namespace",
            "artifact_kind",
            "authority",
            "binding",
            "binding_id",
            "semantic_dependency_ids",
            "dependency_ids",
            "payload_schema",
            "key_id",
        }
        if set(value).difference(allowed):
            raise ArtifactIntegrityError(
                "runtime artifact key contains unknown fields"
            )
        result = cls(
            schema=str(value.get("schema") or ""),
            namespace=value.get("namespace", ""),
            artifact_kind=value.get("artifact_kind", ""),
            authority=value.get("authority", ""),
            binding=value.get("binding"),
            dependency_ids=tuple(value.get("dependency_ids") or ()),
            payload_schema=str(value.get("payload_schema") or ""),
        )
        if value.get("binding_id") not in (None, result.binding.binding_id):
            raise ArtifactIntegrityError("runtime key binding identity mismatch")
        claimed_semantic = value.get("semantic_dependency_ids")
        if claimed_semantic is not None and tuple(claimed_semantic) != (
            result.binding.semantic_dependency_ids
        ):
            raise ForgedDependencyError(
                "runtime key semantic dependencies do not match binding"
            )
        if value.get("key_id") not in (None, result.key_id):
            raise ArtifactIntegrityError("runtime artifact key identity mismatch")
        return result


@dataclass(frozen=True)
class CanonicalArtifactIdentity:
    """Canonical result identity, including all trust and dependency inputs."""

    namespace: str
    artifact_kind: str
    authority: RuntimeAuthority
    binding_id: str
    payload_digest: str
    freshness: EvidenceFreshness
    created_at_ms: int
    expires_at_ms: int | None
    dependency_ids: tuple[str, ...] = ()
    payload_schema: str = ""
    schema: str = RUNTIME_ARTIFACT_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RUNTIME_ARTIFACT_IDENTITY_SCHEMA:
            raise ArtifactIntegrityError(
                "unsupported runtime artifact identity schema"
            )
        for name in (
            "namespace",
            "artifact_kind",
            "binding_id",
            "payload_digest",
        ):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), name)
            )
        object.__setattr__(
            self, "authority", _coerce_authority(self.authority)
        )
        object.__setattr__(
            self, "freshness", _coerce_freshness(self.freshness)
        )
        if not self.payload_digest.startswith("sha256:"):
            raise ArtifactIntegrityError("payload_digest is not canonical")
        if (
            isinstance(self.created_at_ms, bool)
            or not isinstance(self.created_at_ms, int)
            or self.created_at_ms < 0
        ):
            raise ArtifactIntegrityError(
                "created_at_ms must be a nonnegative integer"
            )
        if self.expires_at_ms is not None and (
            isinstance(self.expires_at_ms, bool)
            or not isinstance(self.expires_at_ms, int)
            or self.expires_at_ms <= self.created_at_ms
        ):
            raise ArtifactIntegrityError(
                "expires_at_ms must be later than created_at_ms"
            )
        ids = tuple(
            sorted(
                {
                    _required_text(item, "dependency_id")
                    for item in self.dependency_ids
                }
            )
        )
        if len(ids) != len(tuple(self.dependency_ids)):
            raise ArtifactIntegrityError(
                "artifact identity contains duplicate dependencies"
            )
        object.__setattr__(self, "dependency_ids", ids)
        if self.payload_schema:
            object.__setattr__(
                self,
                "payload_schema",
                _required_text(self.payload_schema, "payload_schema"),
            )

    def _content(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "namespace": self.namespace,
            "artifact_kind": self.artifact_kind,
            "authority": self.authority.value,
            "binding_id": self.binding_id,
            "payload_digest": self.payload_digest,
            "freshness": self.freshness.value,
            "created_at_ms": self.created_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "dependency_ids": list(self.dependency_ids),
            "payload_schema": self.payload_schema,
        }

    @property
    def artifact_id(self) -> str:
        return "runtime-artifact:sha256:" + hashlib.sha256(
            canonical_runtime_json_bytes(self._content())
        ).hexdigest()

    @property
    def content_id(self) -> str:
        return self.artifact_id

    @property
    def digest(self) -> str:
        return self.artifact_id.removeprefix("runtime-artifact:")

    def to_dict(self) -> dict[str, Any]:
        return {**self._content(), "artifact_id": self.artifact_id}

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> "CanonicalArtifactIdentity":
        if not isinstance(value, Mapping):
            raise ArtifactIntegrityError("artifact identity must be an object")
        allowed = {
            "schema",
            "namespace",
            "artifact_kind",
            "authority",
            "binding_id",
            "payload_digest",
            "freshness",
            "created_at_ms",
            "expires_at_ms",
            "dependency_ids",
            "payload_schema",
            "artifact_id",
            "content_id",
        }
        if set(value).difference(allowed):
            raise ArtifactIntegrityError(
                "artifact identity contains unknown fields"
            )
        result = cls(
            schema=str(value.get("schema") or ""),
            namespace=value.get("namespace", ""),
            artifact_kind=value.get("artifact_kind", ""),
            authority=value.get("authority", ""),
            binding_id=value.get("binding_id", ""),
            payload_digest=value.get("payload_digest", ""),
            freshness=value.get("freshness", ""),
            created_at_ms=value.get("created_at_ms"),
            expires_at_ms=value.get("expires_at_ms"),
            dependency_ids=tuple(value.get("dependency_ids") or ()),
            payload_schema=str(value.get("payload_schema") or ""),
        )
        for name in ("artifact_id", "content_id"):
            if value.get(name) not in (None, result.artifact_id):
                raise ArtifactIntegrityError(
                    "canonical artifact identity mismatch"
                )
        return result


ArtifactIdentity = CanonicalArtifactIdentity


@dataclass(frozen=True)
class RuntimeArtifactRecord:
    """One immutable, integrity-checked runtime artifact envelope."""

    identity: CanonicalArtifactIdentity
    binding: ResultBinding
    dependencies: tuple[ArtifactDependency, ...]
    payload: Any
    freshness: EvidenceFreshness
    created_at_ms: int
    expires_at_ms: int | None = None
    envelope_digest: str = ""
    schema: str = RUNTIME_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RUNTIME_ARTIFACT_SCHEMA:
            raise ArtifactIntegrityError("unsupported runtime artifact schema")
        if not isinstance(self.identity, CanonicalArtifactIdentity):
            object.__setattr__(
                self,
                "identity",
                CanonicalArtifactIdentity.from_dict(self.identity),
            )
        object.__setattr__(self, "binding", _coerce_binding(self.binding))
        dependencies = tuple(
            sorted(
                (
                    item
                    if isinstance(item, ArtifactDependency)
                    else ArtifactDependency.from_dict(item)
                    for item in self.dependencies
                ),
                key=lambda item: item.artifact_id,
            )
        )
        dependency_ids = tuple(item.artifact_id for item in dependencies)
        if len(set(dependency_ids)) != len(dependency_ids):
            raise ForgedDependencyError("artifact has duplicate dependencies")
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(
            self, "freshness", _coerce_freshness(self.freshness)
        )
        if (
            isinstance(self.created_at_ms, bool)
            or not isinstance(self.created_at_ms, int)
            or self.created_at_ms < 0
        ):
            raise ArtifactIntegrityError(
                "created_at_ms must be a nonnegative integer"
            )
        if self.expires_at_ms is not None and (
            isinstance(self.expires_at_ms, bool)
            or not isinstance(self.expires_at_ms, int)
            or self.expires_at_ms <= self.created_at_ms
        ):
            raise ArtifactIntegrityError(
                "expires_at_ms must be later than created_at_ms"
            )
        canonical_payload = _canonical_value(self.payload)
        object.__setattr__(self, "payload", canonical_payload)
        if self.identity.payload_digest != _sha256(
            canonical_runtime_json_bytes(canonical_payload)
        ):
            raise ArtifactIntegrityError("runtime artifact payload digest mismatch")
        if self.identity.binding_id != self.binding.binding_id:
            raise ArtifactIntegrityError("runtime artifact binding mismatch")
        if (
            self.identity.freshness is not self.freshness
            or self.identity.created_at_ms != self.created_at_ms
            or self.identity.expires_at_ms != self.expires_at_ms
        ):
            raise ArtifactIntegrityError(
                "runtime artifact freshness identity mismatch"
            )
        if self.identity.dependency_ids != dependency_ids:
            raise ForgedDependencyError(
                "artifact dependency identities do not match its envelope"
            )
        if (
            self.identity.authority is RuntimeAuthority.AUTHORITATIVE
            and any(
                item.authority is RuntimeAuthority.DRAFT
                for item in dependencies
            )
        ):
            raise AuthorityIsolationError(
                "authoritative records cannot depend on draft artifacts"
            )
        if self.envelope_digest and self.envelope_digest != self.computed_digest:
            raise ArtifactIntegrityError("runtime artifact envelope digest mismatch")

    @property
    def artifact_id(self) -> str:
        return self.identity.artifact_id

    @property
    def content_id(self) -> str:
        return self.artifact_id

    @property
    def key(self) -> RuntimeArtifactKey:
        return RuntimeArtifactKey(
            namespace=self.identity.namespace,
            artifact_kind=self.identity.artifact_kind,
            authority=self.identity.authority,
            binding=self.binding,
            dependency_ids=self.identity.dependency_ids,
            payload_schema=self.identity.payload_schema,
        )

    def _content(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "identity": self.identity.to_dict(),
            "artifact_id": self.artifact_id,
            "key": self.key.to_dict(),
            "binding": self.binding.to_dict(),
            "dependencies": [item.to_dict() for item in self.dependencies],
            "payload": self.payload,
            "freshness": self.freshness.value,
            "created_at_ms": self.created_at_ms,
            "expires_at_ms": self.expires_at_ms,
        }

    @property
    def computed_digest(self) -> str:
        return _sha256(canonical_runtime_json_bytes(self._content()))

    def is_fresh_at(self, now_ms: int) -> bool:
        return bool(
            self.freshness is EvidenceFreshness.FRESH
            and (
                self.expires_at_ms is None
                or now_ms < self.expires_at_ms
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._content(),
            "envelope_digest": self.envelope_digest or self.computed_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeArtifactRecord":
        if not isinstance(value, Mapping):
            raise ArtifactIntegrityError("runtime artifact must be an object")
        allowed = {
            "schema",
            "identity",
            "artifact_id",
            "key",
            "binding",
            "dependencies",
            "payload",
            "freshness",
            "created_at_ms",
            "expires_at_ms",
            "envelope_digest",
        }
        if set(value).difference(allowed):
            raise ArtifactIntegrityError(
                "runtime artifact contains unknown fields"
            )
        result = cls(
            schema=str(value.get("schema") or ""),
            identity=CanonicalArtifactIdentity.from_dict(
                value.get("identity")
            ),
            binding=value.get("binding"),
            dependencies=tuple(value.get("dependencies") or ()),
            payload=value.get("payload"),
            freshness=value.get("freshness", ""),
            created_at_ms=value.get("created_at_ms"),
            expires_at_ms=value.get("expires_at_ms"),
            envelope_digest=str(value.get("envelope_digest") or ""),
        )
        if value.get("artifact_id") not in (None, result.artifact_id):
            raise ArtifactIntegrityError("runtime artifact identity mismatch")
        claimed_key = value.get("key")
        if claimed_key is not None:
            decoded_key = RuntimeArtifactKey.from_dict(claimed_key)
            if decoded_key.key_id != result.key.key_id:
                raise ArtifactIntegrityError("runtime artifact key mismatch")
        return result


ArtifactRecord = RuntimeArtifactRecord
RuntimeArtifact = RuntimeArtifactRecord


@dataclass(frozen=True)
class RuntimeCASLookup:
    artifact: RuntimeArtifactRecord | None
    tier: RuntimeTier | None = None
    reason_codes: tuple[str, ...] = ()

    @property
    def hit(self) -> bool:
        return self.artifact is not None

    @property
    def value(self) -> Any:
        return self.artifact.payload if self.artifact is not None else None


ArtifactLookup = RuntimeCASLookup


@dataclass(frozen=True)
class AuthoritativeProjection:
    """Mutable current-tree pointer to one fresh authoritative artifact."""

    projection_key: str
    namespace: str
    tree_id: str
    artifact_id: str
    key_id: str
    updated_at_ms: int
    projection_digest: str = ""
    schema: str = RUNTIME_PROJECTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RUNTIME_PROJECTION_SCHEMA:
            raise ArtifactIntegrityError("unsupported runtime projection schema")
        for name in (
            "projection_key",
            "namespace",
            "tree_id",
            "artifact_id",
            "key_id",
        ):
            object.__setattr__(
                self, name, _required_text(getattr(self, name), name)
            )
        if (
            isinstance(self.updated_at_ms, bool)
            or not isinstance(self.updated_at_ms, int)
            or self.updated_at_ms < 0
        ):
            raise ArtifactIntegrityError(
                "updated_at_ms must be a nonnegative integer"
            )
        if self.projection_digest and (
            self.projection_digest != self.computed_digest
        ):
            raise ArtifactIntegrityError("runtime projection digest mismatch")

    def _content(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "projection_key": self.projection_key,
            "namespace": self.namespace,
            "tree_id": self.tree_id,
            "artifact_id": self.artifact_id,
            "key_id": self.key_id,
            "updated_at_ms": self.updated_at_ms,
        }

    @property
    def computed_digest(self) -> str:
        return _sha256(canonical_runtime_json_bytes(self._content()))

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._content(),
            "projection_digest": (
                self.projection_digest or self.computed_digest
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AuthoritativeProjection":
        if not isinstance(value, Mapping):
            raise ArtifactIntegrityError("projection must be an object")
        allowed = {
            "schema",
            "projection_key",
            "namespace",
            "tree_id",
            "artifact_id",
            "key_id",
            "updated_at_ms",
            "projection_digest",
        }
        if set(value).difference(allowed):
            raise ArtifactIntegrityError("projection contains unknown fields")
        return cls(
            schema=str(value.get("schema") or ""),
            projection_key=value.get("projection_key", ""),
            namespace=value.get("namespace", ""),
            tree_id=value.get("tree_id", ""),
            artifact_id=value.get("artifact_id", ""),
            key_id=value.get("key_id", ""),
            updated_at_ms=value.get("updated_at_ms"),
            projection_digest=str(value.get("projection_digest") or ""),
        )


@dataclass(frozen=True)
class InvalidationResult:
    root_artifact_ids: tuple[str, ...]
    invalidated_artifact_ids: tuple[str, ...]
    preserved_artifact_ids: tuple[str, ...] = ()
    reason: str = "semantic_dependency_changed"
    requirement_id: str = DEPENDENCY_CAS_REQUIREMENT_ID
    schema: str = RUNTIME_INVALIDATION_SCHEMA

    @property
    def invalidated_count(self) -> int:
        return len(self.invalidated_artifact_ids)

    @property
    def affected_artifact_ids(self) -> tuple[str, ...]:
        return self.invalidated_artifact_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "requirement_id": self.requirement_id,
            "root_artifact_ids": list(self.root_artifact_ids),
            "invalidated_artifact_ids": list(
                self.invalidated_artifact_ids
            ),
            "preserved_artifact_ids": list(self.preserved_artifact_ids),
            "invalidated_count": self.invalidated_count,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CASInvalidationReceipt:
    """Crash-recoverable exact closure for one batch invalidation."""

    root_artifact_ids: tuple[str, ...]
    semantic_dependency_ids: tuple[str, ...]
    invalidated_artifact_ids: tuple[str, ...]
    preserved_artifact_ids: tuple[str, ...]
    reason: str
    roots_id: str = ""
    event_cursor: str = ""
    committed: bool = True
    transaction_id: str = ""
    schema: str = RUNTIME_INVALIDATION_TRANSACTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RUNTIME_INVALIDATION_TRANSACTION_SCHEMA:
            raise ArtifactIntegrityError(
                "unsupported invalidation transaction schema"
            )
        for name in (
            "root_artifact_ids",
            "semantic_dependency_ids",
            "invalidated_artifact_ids",
            "preserved_artifact_ids",
        ):
            values = tuple(
                sorted({_required_text(item, f"{name} item") for item in getattr(self, name)})
            )
            if len(values) != len(tuple(getattr(self, name))):
                raise ArtifactIntegrityError(
                    f"{name} contains duplicate identities"
                )
            object.__setattr__(self, name, values)
        object.__setattr__(self, "reason", _required_text(self.reason, "reason"))
        object.__setattr__(self, "roots_id", str(self.roots_id or "").strip())
        object.__setattr__(
            self, "event_cursor", str(self.event_cursor or "").strip()
        )
        if set(self.invalidated_artifact_ids).intersection(
            self.preserved_artifact_ids
        ):
            raise ArtifactIntegrityError(
                "invalidation receipt cannot preserve an invalidated artifact"
            )
        identity_body = self.to_dict(include_identity=False)
        identity_body.pop("committed", None)
        expected = "runtime-invalidation:sha256:" + hashlib.sha256(
            canonical_runtime_json_bytes(identity_body)
        ).hexdigest()
        if self.transaction_id and self.transaction_id != expected:
            raise ArtifactIntegrityError(
                "invalidation transaction identity mismatch"
            )
        object.__setattr__(self, "transaction_id", expected)

    @property
    def invalidated_count(self) -> int:
        return len(self.invalidated_artifact_ids)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": self.schema,
            "root_artifact_ids": list(self.root_artifact_ids),
            "semantic_dependency_ids": list(self.semantic_dependency_ids),
            "invalidated_artifact_ids": list(self.invalidated_artifact_ids),
            "preserved_artifact_ids": list(self.preserved_artifact_ids),
            "invalidated_count": self.invalidated_count,
            "reason": self.reason,
            "roots_id": self.roots_id,
            "event_cursor": self.event_cursor,
            "committed": self.committed,
        }
        if include_identity:
            value["transaction_id"] = self.transaction_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CASInvalidationReceipt":
        if not isinstance(value, Mapping):
            raise ArtifactIntegrityError(
                "invalidation transaction must be an object"
            )
        allowed = {
            "schema",
            "root_artifact_ids",
            "semantic_dependency_ids",
            "invalidated_artifact_ids",
            "preserved_artifact_ids",
            "invalidated_count",
            "reason",
            "roots_id",
            "event_cursor",
            "committed",
            "transaction_id",
        }
        if set(value).difference(allowed):
            raise ArtifactIntegrityError(
                "invalidation transaction contains unknown fields"
            )
        if not isinstance(value.get("committed"), bool):
            raise ArtifactIntegrityError(
                "invalidation transaction committed flag must be boolean"
            )
        result = cls(
            schema=str(value.get("schema") or ""),
            root_artifact_ids=tuple(value.get("root_artifact_ids") or ()),
            semantic_dependency_ids=tuple(
                value.get("semantic_dependency_ids") or ()
            ),
            invalidated_artifact_ids=tuple(
                value.get("invalidated_artifact_ids") or ()
            ),
            preserved_artifact_ids=tuple(
                value.get("preserved_artifact_ids") or ()
            ),
            reason=str(value.get("reason") or ""),
            roots_id=str(value.get("roots_id") or ""),
            event_cursor=str(value.get("event_cursor") or ""),
            committed=value["committed"],
            transaction_id=str(value.get("transaction_id") or ""),
        )
        if value.get("invalidated_count") not in (
            None,
            result.invalidated_count,
        ):
            raise ArtifactIntegrityError(
                "invalidation transaction count mismatch"
            )
        return result


@dataclass(frozen=True)
class RuntimeCASAuditReceipt:
    """Content-addressed health statement for disposable CAS indexes."""

    artifact_ids: tuple[str, ...]
    tombstoned_artifact_ids: tuple[str, ...]
    issue_codes: tuple[str, ...] = ()
    rebuilt: bool = False
    schema: str = RUNTIME_CAS_AUDIT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != RUNTIME_CAS_AUDIT_SCHEMA:
            raise ArtifactIntegrityError(
                "unsupported runtime CAS audit schema"
            )
        for name in (
            "artifact_ids",
            "tombstoned_artifact_ids",
            "issue_codes",
        ):
            values = tuple(
                sorted(
                    {
                        _required_text(item, f"{name} item")
                        for item in getattr(self, name)
                    }
                )
            )
            if len(values) != len(tuple(getattr(self, name))):
                raise ArtifactIntegrityError(
                    f"{name} contains duplicate identities"
                )
            object.__setattr__(self, name, values)

    @property
    def healthy(self) -> bool:
        return not self.issue_codes

    @property
    def receipt_id(self) -> str:
        return "runtime-cas-audit:sha256:" + hashlib.sha256(
            canonical_runtime_json_bytes(self.to_dict(include_identity=False))
        ).hexdigest()

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        value = {
            "schema": self.schema,
            "artifact_ids": list(self.artifact_ids),
            "tombstoned_artifact_ids": list(
                self.tombstoned_artifact_ids
            ),
            "issue_codes": list(self.issue_codes),
            "rebuilt": self.rebuilt,
            "healthy": self.healthy,
        }
        if include_identity:
            value["receipt_id"] = self.receipt_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeCASAuditReceipt":
        if not isinstance(value, Mapping):
            raise ArtifactIntegrityError(
                "runtime CAS audit receipt must be an object"
            )
        result = cls(
            schema=str(value.get("schema") or ""),
            artifact_ids=tuple(value.get("artifact_ids") or ()),
            tombstoned_artifact_ids=tuple(
                value.get("tombstoned_artifact_ids") or ()
            ),
            issue_codes=tuple(value.get("issue_codes") or ()),
            rebuilt=bool(value.get("rebuilt", False)),
        )
        if value.get("healthy") not in (None, result.healthy):
            raise ArtifactIntegrityError(
                "runtime CAS audit health mismatch"
            )
        if value.get("receipt_id") not in (None, result.receipt_id):
            raise ArtifactIntegrityError(
                "runtime CAS audit receipt identity mismatch"
            )
        return result


@dataclass(frozen=True)
class RuntimeCASMetrics:
    lookups: int = 0
    process_hits: int = 0
    host_hits: int = 0
    shared_hits: int = 0
    projection_hits: int = 0
    misses: int = 0
    writes: int = 0
    exact_reuses: int = 0
    corruption_recoveries: int = 0
    invalidated: int = 0
    stale_rejections: int = 0
    forged_dependency_rejections: int = 0
    stale_authoritative_hits: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@runtime_checkable
class SharedImmutableStore(Protocol):
    """Minimal optional shared/P2P immutable byte-store interface."""

    def get(self, artifact_id: str) -> bytes | None:
        ...

    def put(self, artifact_id: str, payload: bytes) -> None:
        ...


class DirectorySharedImmutableStore:
    """Filesystem reference implementation of a shared immutable byte store."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True, mode=0o700)

    def _path(self, artifact_id: str) -> Path:
        digest = artifact_id.removeprefix("runtime-artifact:sha256:")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ArtifactIntegrityError("artifact_id is not canonical")
        return self.path / digest[:2] / f"{digest}.json"

    def get(self, artifact_id: str) -> bytes | None:
        try:
            return self._path(artifact_id).read_bytes()
        except FileNotFoundError:
            return None

    def put(self, artifact_id: str, payload: bytes) -> None:
        path = self._path(artifact_id)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            existing = path.read_bytes()
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if existing != payload:
                raise ImmutableStoreError(
                    "shared immutable artifact cannot be overwritten"
                )
            return
        _atomic_write(path, payload)


SharedCAS = DirectorySharedImmutableStore


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


_PROCESS_LOCKS: dict[str, threading.RLock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


def _process_lock(path: Path) -> threading.RLock:
    identity = str(path.absolute())
    with _PROCESS_LOCKS_GUARD:
        return _PROCESS_LOCKS.setdefault(identity, threading.RLock())


class RuntimeCAS:
    """Dependency-aware coordinator for the four runtime artifact tiers."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        root: str | os.PathLike[str] | None = None,
        shared_store: SharedImmutableStore | Any | None = None,
        current_tree_id: str | None = None,
        clock: Callable[[], float] = time.time,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        lock_timeout_seconds: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    ) -> None:
        if path is not None and root is not None:
            raise ValueError("pass path or root, not both")
        selected = root if root is not None else path
        if selected is None:
            selected = tempfile.mkdtemp(prefix="supervisor-runtime-cas-")
        if (
            isinstance(max_payload_bytes, bool)
            or not isinstance(max_payload_bytes, int)
            or max_payload_bytes < 1
        ):
            raise ValueError("max_payload_bytes must be a positive integer")
        if (
            isinstance(lock_timeout_seconds, bool)
            or not isinstance(lock_timeout_seconds, (int, float))
            or lock_timeout_seconds <= 0
        ):
            raise ValueError("lock_timeout_seconds must be positive")
        self.path = Path(selected)
        self.objects_path = self.path / "objects"
        self.keys_path = self.path / "keys"
        self.projections_path = self.path / "projections"
        self.tombstones_path = self.path / "tombstones"
        self.invalidations_path = self.path / "invalidation-transactions"
        self.quarantine_path = self.path / "quarantine"
        self.invalidation_head_path = self.path / "invalidation-head.json"
        self.locks_path = self.path / "locks"
        for directory in (
            self.objects_path,
            self.keys_path,
            self.projections_path,
            self.tombstones_path,
            self.invalidations_path,
            self.quarantine_path,
            self.locks_path,
        ):
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.shared_store = shared_store
        self.current_tree_id = (
            _required_text(current_tree_id, "current_tree_id")
            if current_tree_id is not None
            else None
        )
        self._clock = clock
        self.max_payload_bytes = max_payload_bytes
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self._memory: dict[str, RuntimeArtifactRecord] = {}
        self._key_memory: dict[str, str] = {}
        self._dependencies: dict[str, set[str]] = {}
        self._children: dict[str, set[str]] = {}
        self._semantic_children: dict[str, set[str]] = {}
        self._known_artifact_ids: set[str] = set()
        self._quarantine_reasons: set[str] = set()
        self._metrics_lock = threading.Lock()
        self._metrics_values = {
            name: 0 for name in RuntimeCASMetrics.__dataclass_fields__
        }
        self._graph_lock = _process_lock(self.path / ".graph")
        self._rebuild_graph()
        self._recover_invalidation_transactions()

    def _increment(self, name: str, amount: int = 1) -> None:
        with self._metrics_lock:
            self._metrics_values[name] += amount

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    @staticmethod
    def _digest_from_id(artifact_id: str) -> str:
        prefix = "runtime-artifact:sha256:"
        if not isinstance(artifact_id, str) or not artifact_id.startswith(prefix):
            raise ArtifactIntegrityError("artifact_id is not canonical")
        digest = artifact_id[len(prefix):]
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ArtifactIntegrityError("artifact_id is not canonical")
        return digest

    def _object_path(self, artifact_id: str) -> Path:
        digest = self._digest_from_id(artifact_id)
        return self.objects_path / digest[:2] / f"{digest}.json"

    @staticmethod
    def _key_digest(key_id: str) -> str:
        return hashlib.sha256(key_id.encode("utf-8")).hexdigest()

    def _key_path(self, key_id: str) -> Path:
        digest = self._key_digest(key_id)
        return self.keys_path / digest[:2] / f"{digest}.json"

    def _tombstone_path(self, artifact_id: str) -> Path:
        digest = self._digest_from_id(artifact_id)
        return self.tombstones_path / digest[:2] / f"{digest}.json"

    def _invalidation_path(self, transaction_id: str) -> Path:
        prefix = "runtime-invalidation:sha256:"
        if not transaction_id.startswith(prefix):
            raise ArtifactIntegrityError(
                "invalidation transaction ID is not canonical"
            )
        digest = transaction_id[len(prefix):]
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ArtifactIntegrityError(
                "invalidation transaction ID is not canonical"
            )
        return self.invalidations_path / f"{digest}.json"

    def _projection_path(self, namespace: str, projection_key: str) -> Path:
        namespace_digest = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
        key_digest = hashlib.sha256(projection_key.encode("utf-8")).hexdigest()
        return (
            self.projections_path
            / namespace_digest[:16]
            / f"{key_digest}.json"
        )

    @contextmanager
    def _key_lock(self, identity: str):
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        lock_path = self.locks_path / f"{digest}.lock"
        thread_lock = _process_lock(lock_path)
        with thread_lock:
            handle = lock_path.open("a+b")
            acquired = False
            deadline = time.monotonic() + self.lock_timeout_seconds
            try:
                while True:
                    try:
                        fcntl.flock(
                            handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                        )
                        acquired = True
                        break
                    except BlockingIOError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"timed out acquiring runtime CAS lock: {identity}"
                            )
                        time.sleep(0.01)
                yield
            finally:
                if acquired:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()

    def _decode(self, raw: bytes) -> RuntimeArtifactRecord:
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError(
                "runtime artifact JSON is corrupt"
            ) from exc
        return RuntimeArtifactRecord.from_dict(payload)

    def _register(self, artifact: RuntimeArtifactRecord) -> None:
        child = artifact.artifact_id
        parents = {item.artifact_id for item in artifact.dependencies}
        self._known_artifact_ids.add(child)
        self._dependencies[child] = parents
        for parent in parents:
            self._children.setdefault(parent, set()).add(child)
        for dependency_id in artifact.binding.semantic_dependency_ids:
            self._semantic_children.setdefault(dependency_id, set()).add(child)

    def _remove_graph_node(self, artifact_id: str) -> None:
        for parent in self._dependencies.pop(artifact_id, set()):
            children = self._children.get(parent)
            if children is not None:
                children.discard(artifact_id)
                if not children:
                    self._children.pop(parent, None)
        for child in self._children.pop(artifact_id, set()):
            parents = self._dependencies.get(child)
            if parents is not None:
                parents.discard(artifact_id)
        for dependency_id, children in tuple(self._semantic_children.items()):
            children.discard(artifact_id)
            if not children:
                self._semantic_children.pop(dependency_id, None)
        self._known_artifact_ids.discard(artifact_id)

    def _recover_corrupt(self, path: Path, artifact_id: str | None = None) -> None:
        try:
            path.unlink()
        except OSError:
            pass
        if artifact_id:
            self._memory.pop(artifact_id, None)
            self._remove_graph_node(artifact_id)
        self._increment("corruption_recoveries")

    def _quarantine_index_file(self, path: Path, reason: str) -> None:
        """Quarantine disposable coordination metadata and fail closed."""

        self._quarantine_reasons.add(_required_text(reason, "reason"))
        if not path.exists():
            return
        digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
        target = self.quarantine_path / f"{digest}-{path.name}"
        try:
            os.replace(path, target)
        except OSError:
            pass
        self._increment("corruption_recoveries")

    def _rebuild_graph(self) -> None:
        with self._graph_lock:
            self._dependencies.clear()
            self._children.clear()
            self._semantic_children.clear()
            self._known_artifact_ids.clear()
            for path in self.objects_path.glob("*/*.json"):
                try:
                    artifact = self._decode(path.read_bytes())
                    if path != self._object_path(artifact.artifact_id):
                        raise ArtifactIntegrityError(
                            "runtime artifact stored under forged path"
                        )
                    self._register(artifact)
                except (OSError, RuntimeCASError, TypeError, ValueError):
                    self._recover_corrupt(path)

    def _is_invalidated(self, artifact_id: str) -> bool:
        path = self._tombstone_path(artifact_id)
        try:
            value = json.loads(path.read_bytes())
        except FileNotFoundError:
            return False
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            self._quarantine_index_file(path, "corrupt_tombstone")
            return True
        try:
            if (
                not isinstance(value, Mapping)
                or value.get("schema") != RUNTIME_INVALIDATION_SCHEMA
                or value.get("artifact_id") != artifact_id
            ):
                raise ArtifactIntegrityError("tombstone binding mismatch")
            claimed = str(value.get("tombstone_digest") or "")
            body = {
                key: item
                for key, item in value.items()
                if key != "tombstone_digest"
            }
            if claimed != _sha256(canonical_runtime_json_bytes(body)):
                raise ArtifactIntegrityError("tombstone digest mismatch")
        except (RuntimeCASError, TypeError, ValueError):
            self._quarantine_index_file(path, "corrupt_tombstone")
            return True
        return True

    def _write_invalidation_tombstones(
        self, receipt: CASInvalidationReceipt
    ) -> None:
        now_ms = self._now_ms()
        for artifact_id in receipt.invalidated_artifact_ids:
            tombstone = {
                "schema": RUNTIME_INVALIDATION_SCHEMA,
                "artifact_id": artifact_id,
                "root_artifact_ids": list(receipt.root_artifact_ids),
                "semantic_dependency_ids": list(
                    receipt.semantic_dependency_ids
                ),
                "invalidation_transaction_id": receipt.transaction_id,
                "invalidated_at_ms": now_ms,
                "reason": receipt.reason,
                "roots_id": receipt.roots_id,
                "event_cursor": receipt.event_cursor,
            }
            tombstone["tombstone_digest"] = _sha256(
                canonical_runtime_json_bytes(tombstone)
            )
            _atomic_write(
                self._tombstone_path(artifact_id),
                canonical_runtime_json_bytes(tombstone) + b"\n",
            )
            self._memory.pop(artifact_id, None)

    def _commit_invalidation(
        self, receipt: CASInvalidationReceipt, *, count_metric: bool
    ) -> CASInvalidationReceipt:
        self._write_invalidation_tombstones(receipt)
        invalidated = set(receipt.invalidated_artifact_ids)
        for key_id, target in tuple(self._key_memory.items()):
            if target in invalidated:
                self._key_memory.pop(key_id, None)
        self._remove_live_pointers(invalidated)
        committed = replace(receipt, committed=True, transaction_id="")
        _atomic_write(
            self._invalidation_path(committed.transaction_id),
            canonical_runtime_json_bytes(committed.to_dict()) + b"\n",
        )
        head = {
            "schema": RUNTIME_INVALIDATION_TRANSACTION_SCHEMA,
            "transaction_id": committed.transaction_id,
        }
        head["head_digest"] = _sha256(canonical_runtime_json_bytes(head))
        _atomic_write(
            self.invalidation_head_path,
            canonical_runtime_json_bytes(head) + b"\n",
        )
        if count_metric:
            self._increment("invalidated", len(invalidated))
        return committed

    def _recover_invalidation_transactions(self) -> None:
        """Finish valid intents and quarantine malformed journals."""

        for path in sorted(self.invalidations_path.glob("*.json")):
            try:
                value = json.loads(path.read_bytes())
                receipt = CASInvalidationReceipt.from_dict(value)
                if path != self._invalidation_path(receipt.transaction_id):
                    raise ArtifactIntegrityError(
                        "invalidation journal path mismatch"
                    )
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                RuntimeCASError,
                TypeError,
                ValueError,
            ):
                self._quarantine_index_file(
                    path, "corrupt_invalidation_journal"
                )
                continue
            if receipt.committed:
                # Every committed receipt must have a complete tombstone set.
                if not all(
                    self._is_invalidated(item)
                    for item in receipt.invalidated_artifact_ids
                ):
                    self._quarantine_reasons.add(
                        "partial_invalidation_transaction"
                    )
                continue
            try:
                self._commit_invalidation(receipt, count_metric=False)
            except (OSError, RuntimeCASError, TypeError, ValueError):
                self._quarantine_reasons.add(
                    "partial_invalidation_transaction"
                )

    def _read_host(
        self, artifact_id: str
    ) -> RuntimeArtifactRecord | None:
        path = self._object_path(artifact_id)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError:
            return None
        try:
            artifact = self._decode(raw)
            if artifact.artifact_id != artifact_id:
                raise ArtifactIntegrityError(
                    "runtime artifact stored under forged identity"
                )
        except (RuntimeCASError, TypeError, ValueError):
            self._recover_corrupt(path, artifact_id)
            return None
        self._register(artifact)
        return artifact

    def _shared_get(self, artifact_id: str) -> bytes | None:
        if self.shared_store is None:
            return None
        getter = getattr(self.shared_store, "get", None)
        if getter is None:
            getter = getattr(self.shared_store, "get_bytes", None)
        if not callable(getter):
            raise TypeError("shared_store must provide get() or get_bytes()")
        value = getter(artifact_id)
        if value is None:
            return None
        if isinstance(value, str):
            value = value.encode("utf-8")
        if not isinstance(value, bytes):
            raise ArtifactIntegrityError(
                "shared immutable store returned non-bytes content"
            )
        return value

    def _shared_put(self, artifact_id: str, payload: bytes) -> None:
        if self.shared_store is None:
            return
        putter = getattr(self.shared_store, "put", None)
        if putter is None:
            putter = getattr(self.shared_store, "put_bytes", None)
        if not callable(putter):
            raise TypeError("shared_store must provide put() or put_bytes()")
        existing = self._shared_get(artifact_id)
        if existing is not None and existing != payload:
            raise ImmutableStoreError(
                "shared immutable artifact cannot be overwritten"
            )
        if existing is None:
            putter(artifact_id, payload)

    def _validate_artifact_dependencies(
        self,
        artifact: RuntimeArtifactRecord,
        *,
        stack: tuple[str, ...] = (),
        require_fresh: bool = False,
        reject_drafts: bool | None = None,
    ) -> bool:
        if reject_drafts is None:
            reject_drafts = (
                artifact.identity.authority
                is RuntimeAuthority.AUTHORITATIVE
            )
        if artifact.artifact_id in stack:
            raise DependencyCycleError(
                "runtime artifact dependency graph contains a cycle"
            )
        next_stack = (*stack, artifact.artifact_id)
        for claim in artifact.dependencies:
            if claim.artifact_id == artifact.artifact_id:
                raise DependencyCycleError(
                    "runtime artifact cannot depend on itself"
                )
            dependency = self._get_exact(
                claim.artifact_id,
                include_shared=True,
                validate_dependencies=False,
            )
            if dependency is None:
                raise ForgedDependencyError(
                    f"dependency does not exist: {claim.artifact_id}"
                )
            if (
                reject_drafts
                and dependency.identity.authority is RuntimeAuthority.DRAFT
            ):
                raise AuthorityIsolationError(
                    "authoritative dependency closure cannot contain drafts"
                )
            if require_fresh and not dependency.is_fresh_at(self._now_ms()):
                raise ForgedDependencyError(
                    f"dependency is stale: {claim.artifact_id}"
                )
            expected = ArtifactDependency.from_artifact(dependency)
            if claim != expected:
                raise ForgedDependencyError(
                    f"dependency claim is forged: {claim.artifact_id}"
                )
            if dependency.artifact_id in next_stack:
                raise DependencyCycleError(
                    "runtime artifact dependency graph contains a cycle"
                )
            self._validate_artifact_dependencies(
                dependency,
                stack=next_stack,
                require_fresh=require_fresh,
                reject_drafts=reject_drafts,
            )
        return True

    def _get_exact(
        self,
        artifact_id: str,
        *,
        include_shared: bool,
        validate_dependencies: bool = True,
    ) -> RuntimeArtifactRecord | None:
        self._digest_from_id(artifact_id)
        if self._is_invalidated(artifact_id):
            return None
        artifact = self._memory.get(artifact_id)
        if artifact is None:
            artifact = self._read_host(artifact_id)
        if artifact is None and include_shared:
            try:
                raw = self._shared_get(artifact_id)
                if raw is not None:
                    candidate = self._decode(raw)
                    if candidate.artifact_id != artifact_id:
                        raise ArtifactIntegrityError(
                            "shared artifact identity mismatch"
                        )
                    artifact = candidate
                    _atomic_write(self._object_path(artifact_id), raw)
                    self._register(artifact)
            except (
                OSError,
                RuntimeCASError,
                TypeError,
                ValueError,
            ):
                self._increment("corruption_recoveries")
                artifact = None
        if artifact is None:
            return None
        if validate_dependencies:
            try:
                self._validate_artifact_dependencies(artifact)
            except (
                AuthorityIsolationError,
                DependencyCycleError,
                ForgedDependencyError,
            ):
                self._recover_corrupt(
                    self._object_path(artifact.artifact_id),
                    artifact.artifact_id,
                )
                return None
        self._memory[artifact.artifact_id] = artifact
        return artifact

    def lookup(
        self,
        identity: str | RuntimeArtifactKey | Mapping[str, Any],
        *,
        expected_namespace: str | None = None,
        expected_authority: RuntimeAuthority | str | None = None,
        require_fresh: bool = False,
    ) -> RuntimeCASLookup:
        """Look up an exact artifact ID or complete computation key."""

        self._increment("lookups")
        if self._quarantine_reasons:
            self._increment("misses")
            return RuntimeCASLookup(
                None,
                reason_codes=(
                    "runtime_cas_quarantined",
                    *tuple(sorted(self._quarantine_reasons)),
                ),
            )
        key: RuntimeArtifactKey | None = None
        if isinstance(identity, RuntimeArtifactKey):
            key = identity
        elif isinstance(identity, Mapping):
            key = RuntimeArtifactKey.from_dict(identity)
        if key is not None:
            artifact_id = self._key_memory.get(key.key_id, "")
            if not artifact_id:
                pointer_path = self._key_path(key.key_id)
                try:
                    pointer = json.loads(pointer_path.read_bytes())
                    if (
                        not isinstance(pointer, Mapping)
                        or pointer.get("schema") != RUNTIME_CAS_SCHEMA
                        or pointer.get("key_id") != key.key_id
                    ):
                        raise ArtifactIntegrityError(
                            "runtime key pointer is corrupt"
                        )
                    artifact_id = _required_text(
                        pointer.get("artifact_id"), "artifact_id"
                    )
                    claimed_digest = pointer.get("pointer_digest")
                    pointer_content = {
                        "schema": RUNTIME_CAS_SCHEMA,
                        "key_id": key.key_id,
                        "artifact_id": artifact_id,
                    }
                    if claimed_digest != _sha256(
                        canonical_runtime_json_bytes(pointer_content)
                    ):
                        raise ArtifactIntegrityError(
                            "runtime key pointer digest mismatch"
                        )
                except FileNotFoundError:
                    self._increment("misses")
                    return RuntimeCASLookup(None, reason_codes=("key_miss",))
                except (
                    OSError,
                    TypeError,
                    ValueError,
                    RuntimeCASError,
                    json.JSONDecodeError,
                ):
                    self._recover_corrupt(pointer_path)
                    self._increment("misses")
                    return RuntimeCASLookup(
                        None, reason_codes=("corrupt_key_pointer",)
                    )
        else:
            artifact_id = _required_text(identity, "artifact_id")

        if self._is_invalidated(artifact_id):
            self._increment("misses")
            return RuntimeCASLookup(None, reason_codes=("invalidated",))

        tier: RuntimeTier | None = None
        artifact = self._memory.get(artifact_id)
        if artifact is not None:
            tier = RuntimeTier.PROCESS_LOCAL
        if artifact is None:
            artifact = self._read_host(artifact_id)
            if artifact is not None:
                tier = RuntimeTier.HOST_DURABLE
        if artifact is None:
            try:
                raw = self._shared_get(artifact_id)
                if raw is not None:
                    artifact = self._decode(raw)
                    if artifact.artifact_id != artifact_id:
                        raise ArtifactIntegrityError(
                            "shared artifact identity mismatch"
                        )
                    self._validate_artifact_dependencies(artifact)
                    _atomic_write(self._object_path(artifact_id), raw)
                    self._register(artifact)
                    tier = RuntimeTier.SHARED_IMMUTABLE
            except (
                OSError,
                RuntimeCASError,
                TypeError,
                ValueError,
            ):
                self._increment("corruption_recoveries")
                artifact = None
        if artifact is None:
            self._increment("misses")
            return RuntimeCASLookup(None, reason_codes=("artifact_miss",))
        authoritative_lookup = (
            artifact.identity.authority is RuntimeAuthority.AUTHORITATIVE
        )
        try:
            self._validate_artifact_dependencies(
                artifact,
                require_fresh=require_fresh or authoritative_lookup,
            )
        except (
            AuthorityIsolationError,
            DependencyCycleError,
            ForgedDependencyError,
        ):
            self._increment("forged_dependency_rejections")
            self._recover_corrupt(
                self._object_path(artifact.artifact_id),
                artifact.artifact_id,
            )
            self._increment("misses")
            return RuntimeCASLookup(
                None, reason_codes=("invalid_dependency_graph",)
            )
        if key is not None and artifact.key.key_id != key.key_id:
            self._recover_corrupt(self._key_path(key.key_id))
            self._increment("misses")
            return RuntimeCASLookup(
                None, reason_codes=("forged_key_binding",)
            )
        if (
            expected_namespace is not None
            and artifact.identity.namespace
            != _required_text(expected_namespace, "expected_namespace")
        ):
            self._increment("misses")
            return RuntimeCASLookup(
                None, reason_codes=("namespace_mismatch",)
            )
        if (
            expected_authority is not None
            and artifact.identity.authority
            is not _coerce_authority(expected_authority)
        ):
            self._increment("misses")
            return RuntimeCASLookup(
                None, reason_codes=("authority_mismatch",)
            )
        if (
            require_fresh or authoritative_lookup
        ) and not artifact.is_fresh_at(self._now_ms()):
            self._increment("stale_rejections")
            self._increment("misses")
            return RuntimeCASLookup(None, reason_codes=("stale_artifact",))
        self._memory[artifact.artifact_id] = artifact
        if tier is RuntimeTier.PROCESS_LOCAL:
            self._increment("process_hits")
        elif tier is RuntimeTier.HOST_DURABLE:
            self._increment("host_hits")
        elif tier is RuntimeTier.SHARED_IMMUTABLE:
            self._increment("shared_hits")
        if key is not None:
            self._increment("exact_reuses")
        return RuntimeCASLookup(
            artifact, tier=tier, reason_codes=("exact_hit",)
        )

    def get(
        self,
        identity: str | RuntimeArtifactKey | Mapping[str, Any],
        *,
        expected_namespace: str | None = None,
        expected_authority: RuntimeAuthority | str | None = None,
        require_fresh: bool = False,
    ) -> RuntimeArtifactRecord | None:
        return self.lookup(
            identity,
            expected_namespace=expected_namespace,
            expected_authority=expected_authority,
            require_fresh=require_fresh,
        ).artifact

    load = get

    def _coerce_dependency(
        self,
        value: (
            RuntimeArtifactRecord
            | ArtifactDependency
            | Mapping[str, Any]
            | str
        ),
    ) -> ArtifactDependency:
        if isinstance(value, RuntimeArtifactRecord):
            claim = ArtifactDependency.from_artifact(value)
        elif isinstance(value, ArtifactDependency):
            claim = value
        elif isinstance(value, str):
            dependency = self._get_exact(
                value, include_shared=True
            )
            if dependency is None:
                raise ForgedDependencyError(
                    f"dependency does not exist: {value}"
                )
            claim = ArtifactDependency.from_artifact(dependency)
        elif isinstance(value, Mapping):
            claim = ArtifactDependency.from_dict(value)
        else:
            raise ForgedDependencyError(
                "dependency must be an artifact, reference, mapping, or ID"
            )
        dependency = self._get_exact(
            claim.artifact_id, include_shared=True
        )
        if dependency is None or claim != ArtifactDependency.from_artifact(
            dependency
        ):
            self._increment("forged_dependency_rejections")
            raise ForgedDependencyError(
                f"dependency claim is missing or forged: {claim.artifact_id}"
            )
        return claim

    def put(
        self,
        payload: Any,
        *,
        binding: ResultBinding | Mapping[str, Any],
        namespace: str,
        artifact_kind: str = "artifact",
        authority: RuntimeAuthority | str = RuntimeAuthority.DIAGNOSTIC,
        dependencies: Sequence[
            RuntimeArtifactRecord
            | ArtifactDependency
            | Mapping[str, Any]
            | str
        ] = (),
        freshness: EvidenceFreshness | str = EvidenceFreshness.FRESH,
        ttl_seconds: int | None = None,
        tiers: Sequence[RuntimeTier | str] | None = None,
        payload_schema: str = "",
        projection_key: str | None = None,
        tree_id: str | None = None,
    ) -> RuntimeArtifactRecord:
        """Persist one immutable result and its exact computation-key pointer."""

        if self._quarantine_reasons:
            raise RuntimeCASError(
                "runtime CAS is quarantined and cannot accept writes"
            )
        native_binding = _coerce_binding(binding)
        native_namespace = _required_text(namespace, "namespace")
        native_authority = _coerce_authority(authority)
        native_freshness = _coerce_freshness(freshness)
        if (
            "draft" in native_namespace.casefold().replace("-", "_").split("_")
            and native_authority is not RuntimeAuthority.DRAFT
        ):
            raise AuthorityIsolationError(
                "draft namespaces require draft authority"
            )
        if ttl_seconds is not None and (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, int)
            or ttl_seconds < 1
        ):
            raise ValueError("ttl_seconds must be a positive integer or None")
        canonical_payload = _canonical_value(payload)
        encoded_payload = canonical_runtime_json_bytes(canonical_payload)
        if len(encoded_payload) > self.max_payload_bytes:
            raise ArtifactIntegrityError(
                f"runtime payload exceeds {self.max_payload_bytes} bytes"
            )
        claims = tuple(
            sorted(
                (self._coerce_dependency(item) for item in dependencies),
                key=lambda item: item.artifact_id,
            )
        )
        if len({item.artifact_id for item in claims}) != len(claims):
            raise ForgedDependencyError("dependencies contain duplicates")
        if (
            native_authority is RuntimeAuthority.AUTHORITATIVE
            and any(
                item.authority is RuntimeAuthority.DRAFT for item in claims
            )
        ):
            raise AuthorityIsolationError(
                "authoritative records cannot depend on drafts"
            )
        now_ms = self._now_ms()
        expires_at_ms = (
            now_ms + ttl_seconds * 1000
            if ttl_seconds is not None
            else None
        )
        identity = CanonicalArtifactIdentity(
            namespace=native_namespace,
            artifact_kind=_required_text(artifact_kind, "artifact_kind"),
            authority=native_authority,
            binding_id=native_binding.binding_id,
            payload_digest=_sha256(encoded_payload),
            freshness=native_freshness,
            created_at_ms=now_ms,
            expires_at_ms=expires_at_ms,
            dependency_ids=tuple(item.artifact_id for item in claims),
            payload_schema=payload_schema,
        )
        if identity.artifact_id in identity.dependency_ids:
            raise DependencyCycleError(
                "runtime artifact cannot depend on itself"
            )
        artifact = RuntimeArtifactRecord(
            identity=identity,
            binding=native_binding,
            dependencies=claims,
            payload=canonical_payload,
            freshness=native_freshness,
            created_at_ms=now_ms,
            expires_at_ms=expires_at_ms,
        )
        artifact = replace(
            artifact, envelope_digest=artifact.computed_digest
        )
        self._validate_artifact_dependencies(
            artifact,
            require_fresh=(
                native_authority is RuntimeAuthority.AUTHORITATIVE
            ),
        )
        if self._is_invalidated(artifact.artifact_id):
            raise RuntimeCASError(
                "cannot republish an invalidated artifact identity"
            )
        selected_tiers = (
            {
                RuntimeTier.PROCESS_LOCAL,
                RuntimeTier.HOST_DURABLE,
                *(
                    (RuntimeTier.SHARED_IMMUTABLE,)
                    if self.shared_store is not None
                    else ()
                ),
            }
            if tiers is None
            else {
                item
                if isinstance(item, RuntimeTier)
                else RuntimeTier(str(item))
                for item in tiers
            }
        )
        if (
            projection_key is not None
            or RuntimeTier.AUTHORITATIVE_PROJECTION in selected_tiers
        ):
            if projection_key is None:
                raise AuthorityIsolationError(
                    "authoritative projection tier requires projection_key"
                )
            if native_authority is not RuntimeAuthority.AUTHORITATIVE:
                raise AuthorityIsolationError(
                    "only authoritative artifacts can be projected"
                )
        if (
            projection_key is not None
            or RuntimeTier.AUTHORITATIVE_PROJECTION in selected_tiers
        ):
            # A mutable projection must never outlive the immutable record it
            # names merely because a process exited.
            selected_tiers.add(RuntimeTier.HOST_DURABLE)
        encoded = canonical_runtime_json_bytes(artifact.to_dict()) + b"\n"
        with self._key_lock(artifact.key.key_id):
            existing = self._read_host(artifact.artifact_id)
            if existing is not None:
                # Exact identities are immutable.  Reuse the existing envelope
                # rather than refreshing provenance timestamps silently.
                if (
                    existing.identity != artifact.identity
                    or existing.binding != artifact.binding
                    or existing.dependencies != artifact.dependencies
                    or existing.payload != artifact.payload
                ):
                    raise ImmutableStoreError(
                        "canonical artifact identity collision"
                    )
                artifact = existing
                encoded = (
                    canonical_runtime_json_bytes(artifact.to_dict()) + b"\n"
                )
            if RuntimeTier.HOST_DURABLE in selected_tiers and existing is None:
                _atomic_write(self._object_path(artifact.artifact_id), encoded)
            if RuntimeTier.SHARED_IMMUTABLE in selected_tiers:
                self._shared_put(artifact.artifact_id, encoded)
            pointer_content = {
                "schema": RUNTIME_CAS_SCHEMA,
                "key_id": artifact.key.key_id,
                "artifact_id": artifact.artifact_id,
            }
            pointer = {
                **pointer_content,
                "pointer_digest": _sha256(
                    canonical_runtime_json_bytes(pointer_content)
                ),
            }
            if RuntimeTier.HOST_DURABLE in selected_tiers:
                _atomic_write(
                    self._key_path(artifact.key.key_id),
                    canonical_runtime_json_bytes(pointer) + b"\n",
                )
            if RuntimeTier.PROCESS_LOCAL in selected_tiers:
                self._memory[artifact.artifact_id] = artifact
                self._key_memory[artifact.key.key_id] = artifact.artifact_id
            self._register(artifact)
            self._increment("writes")
            if projection_key is not None:
                self.project(
                    projection_key,
                    artifact,
                    namespace=native_namespace,
                    tree_id=tree_id,
                )
        return artifact

    store = put

    def get_or_compute(
        self,
        key: RuntimeArtifactKey | Mapping[str, Any],
        producer: Callable[[], Any],
        *,
        dependencies: Sequence[
            RuntimeArtifactRecord
            | ArtifactDependency
            | Mapping[str, Any]
            | str
        ] = (),
        freshness: EvidenceFreshness | str = EvidenceFreshness.FRESH,
        ttl_seconds: int | None = None,
        tiers: Sequence[RuntimeTier | str] | None = None,
        projection_key: str | None = None,
        tree_id: str | None = None,
    ) -> tuple[RuntimeArtifactRecord, bool]:
        """Return an exact warm result or produce and store it once.

        The boolean is ``True`` only when the producer was invoked.
        """

        native_key = (
            key
            if isinstance(key, RuntimeArtifactKey)
            else RuntimeArtifactKey.from_dict(key)
        )
        supplied_claims = tuple(
            self._coerce_dependency(item) for item in dependencies
        )
        if tuple(sorted(item.artifact_id for item in supplied_claims)) != (
            native_key.dependency_ids
        ):
            raise ForgedDependencyError(
                "get_or_compute dependencies do not match the exact key"
            )
        hit = self.get(native_key, require_fresh=True)
        if hit is not None:
            return hit, False
        value = producer()
        artifact = self.put(
            value,
            binding=native_key.binding,
            namespace=native_key.namespace,
            artifact_kind=native_key.artifact_kind,
            authority=native_key.authority,
            dependencies=supplied_claims,
            freshness=freshness,
            ttl_seconds=ttl_seconds,
            tiers=tiers,
            payload_schema=native_key.payload_schema,
            projection_key=projection_key,
            tree_id=tree_id,
        )
        return artifact, True

    coordinate = get_or_compute

    def project(
        self,
        projection_key: str,
        artifact: RuntimeArtifactRecord | str,
        *,
        namespace: str | None = None,
        tree_id: str | None = None,
    ) -> AuthoritativeProjection:
        """Atomically publish a fresh authoritative current-tree pointer."""

        if self._quarantine_reasons:
            raise RuntimeCASError(
                "runtime CAS is quarantined and cannot publish projections"
            )
        projection_key = _required_text(projection_key, "projection_key")
        record = (
            artifact
            if isinstance(artifact, RuntimeArtifactRecord)
            else self.get(artifact, require_fresh=True)
        )
        if record is None:
            raise ArtifactIntegrityError(
                "projection artifact is missing, invalidated, or stale"
            )
        if record.identity.authority is not RuntimeAuthority.AUTHORITATIVE:
            raise AuthorityIsolationError(
                "authoritative projections cannot reference non-authoritative artifacts"
            )
        if not record.is_fresh_at(self._now_ms()):
            raise AuthorityIsolationError(
                "authoritative projections require fresh artifacts"
            )
        expected_namespace = (
            record.identity.namespace
            if namespace is None
            else _required_text(namespace, "namespace")
        )
        if record.identity.namespace != expected_namespace:
            raise AuthorityIsolationError(
                "projection namespace does not match artifact namespace"
            )
        expected_tree = (
            _required_text(tree_id, "tree_id")
            if tree_id is not None
            else self.current_tree_id
        )
        if expected_tree is None:
            expected_tree = record.binding.tree_id
        if record.binding.tree_id != expected_tree:
            raise AuthorityIsolationError(
                "projection artifact is not bound to the current tree"
            )
        projection = AuthoritativeProjection(
            projection_key=projection_key,
            namespace=expected_namespace,
            tree_id=expected_tree,
            artifact_id=record.artifact_id,
            key_id=record.key.key_id,
            updated_at_ms=self._now_ms(),
        )
        projection = replace(
            projection, projection_digest=projection.computed_digest
        )
        _atomic_write(
            self._projection_path(expected_namespace, projection_key),
            canonical_runtime_json_bytes(projection.to_dict()) + b"\n",
        )
        return projection

    set_projection = project
    publish_projection = project

    def lookup_projection(
        self,
        projection_key: str,
        *,
        namespace: str,
        tree_id: str | None = None,
    ) -> RuntimeCASLookup:
        projection_key = _required_text(projection_key, "projection_key")
        namespace = _required_text(namespace, "namespace")
        expected_tree = tree_id or self.current_tree_id
        path = self._projection_path(namespace, projection_key)
        try:
            projection = AuthoritativeProjection.from_dict(
                json.loads(path.read_bytes())
            )
        except FileNotFoundError:
            self._increment("misses")
            return RuntimeCASLookup(None, reason_codes=("projection_miss",))
        except (
            OSError,
            TypeError,
            ValueError,
            RuntimeCASError,
            json.JSONDecodeError,
        ):
            self._recover_corrupt(path)
            self._increment("misses")
            return RuntimeCASLookup(
                None, reason_codes=("corrupt_projection",)
            )
        if (
            projection.projection_key != projection_key
            or projection.namespace != namespace
            or (expected_tree is not None and projection.tree_id != expected_tree)
        ):
            self._increment("misses")
            return RuntimeCASLookup(
                None, reason_codes=("projection_binding_mismatch",)
            )
        artifact = self.get(
            projection.artifact_id,
            expected_namespace=namespace,
            expected_authority=RuntimeAuthority.AUTHORITATIVE,
            require_fresh=True,
        )
        if (
            artifact is None
            or artifact.key.key_id != projection.key_id
            or artifact.binding.tree_id != projection.tree_id
        ):
            self._recover_corrupt(path)
            self._increment("misses")
            return RuntimeCASLookup(
                None, reason_codes=("stale_or_forged_projection",)
            )
        self._increment("projection_hits")
        return RuntimeCASLookup(
            artifact,
            tier=RuntimeTier.AUTHORITATIVE_PROJECTION,
            reason_codes=("current_tree_projection_hit",),
        )

    def get_projection(
        self,
        projection_key: str,
        *,
        namespace: str,
        tree_id: str | None = None,
    ) -> RuntimeArtifactRecord | None:
        return self.lookup_projection(
            projection_key, namespace=namespace, tree_id=tree_id
        ).artifact

    def dependencies_of(self, artifact_id: str) -> tuple[str, ...]:
        self._rebuild_graph()
        return tuple(sorted(self._dependencies.get(artifact_id, set())))

    def inspect_artifact(
        self, artifact_id: str
    ) -> RuntimeArtifactRecord | None:
        """Read an immutable envelope for audit, even when it is tombstoned."""

        self._digest_from_id(artifact_id)
        return self._read_host(artifact_id)

    def semantic_dependency_ids(
        self,
        *,
        namespace: str = "",
        key: str = "",
        revision: str = "",
        digest: str = "",
    ) -> tuple[str, ...]:
        """Resolve exact semantic identities without treating names as authority."""

        matches: set[str] = set()
        self._rebuild_graph()
        for artifact_id in sorted(self._known_artifact_ids):
            artifact = self._read_host(artifact_id)
            if artifact is None:
                continue
            for dependency in artifact.binding.semantic_dependencies:
                if (
                    (not namespace or dependency.namespace == namespace)
                    and (not key or dependency.key == key)
                    and (not revision or dependency.revision == revision)
                    and (not digest or dependency.digest == digest)
                ):
                    matches.add(dependency.dependency_id)
        return tuple(sorted(matches))

    def descendants_of(
        self, artifact_id: str, *, include_root: bool = False
    ) -> tuple[str, ...]:
        """Return a stable breadth-first invalidation traversal."""

        self._rebuild_graph()
        visited: set[str] = set()
        order: list[str] = []
        frontier = [artifact_id]
        while frontier:
            parent = frontier.pop(0)
            for child in sorted(self._children.get(parent, set())):
                if child not in visited:
                    visited.add(child)
                    order.append(child)
                    frontier.append(child)
        if include_root:
            return (artifact_id, *tuple(order))
        return tuple(order)

    def _remove_live_pointers(self, invalidated: set[str]) -> None:
        for path in self.keys_path.glob("*/*.json"):
            try:
                pointer = json.loads(path.read_bytes())
                if pointer.get("artifact_id") in invalidated:
                    path.unlink()
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                self._recover_corrupt(path)
        for path in self.projections_path.glob("*/*.json"):
            try:
                projection = json.loads(path.read_bytes())
                if projection.get("artifact_id") in invalidated:
                    path.unlink()
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                self._recover_corrupt(path)

    def invalidate_batch(
        self,
        *,
        artifact_ids: Sequence[str] = (),
        semantic_dependency_ids: Sequence[str] = (),
        include_artifact_roots: bool = True,
        reason: str = "semantic_dependency_changed",
        roots_id: str = "",
        event_cursor: str = "",
    ) -> CASInvalidationReceipt:
        """Atomically tombstone the union of exact reverse dependency closures.

        An intent is durable before the first tombstone and a committed receipt
        is durable after pointers are removed. Startup completes a valid intent
        idempotently; malformed or incomplete metadata places the store in a
        fail-closed quarantined state visible through :meth:`audit`.
        """

        roots = tuple(
            sorted({_required_text(item, "artifact_id") for item in artifact_ids})
        )
        semantic_roots = tuple(
            sorted(
                {
                    _required_text(item, "semantic_dependency_id")
                    for item in semantic_dependency_ids
                }
            )
        )
        if not roots and not semantic_roots:
            raise ValueError(
                "batch invalidation requires an artifact or semantic dependency"
            )
        for artifact_id in roots:
            self._digest_from_id(artifact_id)
        reason = _required_text(reason, "reason")
        with self._key_lock("dependency-graph-invalidation"):
            if self._quarantine_reasons:
                raise RuntimeCASError(
                    "runtime CAS is quarantined and cannot invalidate"
                )
            self._rebuild_graph()
            missing_roots = sorted(
                set(roots).difference(self._known_artifact_ids)
            )
            if missing_roots:
                raise ArtifactIntegrityError(
                    "invalidation roots do not exist: "
                    + ", ".join(missing_roots)
                )
            affected: set[str] = set()
            for artifact_id in roots:
                affected.update(
                    self.descendants_of(
                        artifact_id, include_root=include_artifact_roots
                    )
                )
            for dependency_id in semantic_roots:
                frontier = sorted(
                    self._semantic_children.get(dependency_id, set())
                )
                while frontier:
                    artifact_id = frontier.pop(0)
                    if artifact_id in affected:
                        continue
                    affected.add(artifact_id)
                    frontier.extend(
                        sorted(self._children.get(artifact_id, set()))
                    )
            live = {
                item
                for item in self._known_artifact_ids
                if not self._is_invalidated(item)
            }
            receipt = CASInvalidationReceipt(
                root_artifact_ids=roots,
                semantic_dependency_ids=semantic_roots,
                invalidated_artifact_ids=tuple(sorted(affected)),
                preserved_artifact_ids=tuple(sorted(live.difference(affected))),
                reason=reason,
                roots_id=str(roots_id or ""),
                event_cursor=str(event_cursor or ""),
                committed=False,
            )
            journal_path = self._invalidation_path(receipt.transaction_id)
            try:
                existing = CASInvalidationReceipt.from_dict(
                    json.loads(journal_path.read_bytes())
                )
            except FileNotFoundError:
                existing = None
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                RuntimeCASError,
                TypeError,
                ValueError,
            ):
                self._quarantine_index_file(
                    journal_path, "corrupt_invalidation_journal"
                )
                raise ArtifactIntegrityError(
                    "invalidation transaction journal is corrupt"
                )
            if existing is not None:
                if existing.transaction_id != receipt.transaction_id:
                    raise ArtifactIntegrityError(
                        "invalidation transaction binding conflict"
                    )
                if existing.committed:
                    return existing
                receipt = existing
            else:
                _atomic_write(
                    journal_path,
                    canonical_runtime_json_bytes(receipt.to_dict()) + b"\n",
                )
            return self._commit_invalidation(
                receipt, count_metric=True
            )

    def invalidate(
        self,
        artifact_id: str,
        *,
        include_root: bool = True,
        reason: str = "semantic_dependency_changed",
    ) -> InvalidationResult:
        """Tombstone an artifact and only its transitive dependents."""
        receipt = self.invalidate_batch(
            artifact_ids=(artifact_id,),
            include_artifact_roots=include_root,
            reason=reason,
        )
        return InvalidationResult(
            root_artifact_ids=(artifact_id,),
            invalidated_artifact_ids=receipt.invalidated_artifact_ids,
            preserved_artifact_ids=receipt.preserved_artifact_ids,
            reason=reason,
        )

    invalidate_descendants = invalidate

    def invalidate_dependency(
        self,
        artifact_id: str,
        *,
        reason: str = "semantic_dependency_changed",
    ) -> InvalidationResult:
        """Invalidate dependents while retaining the immutable dependency."""

        return self.invalidate(
            artifact_id, include_root=False, reason=reason
        )

    def invalidate_semantic_dependency(
        self,
        dependency: SemanticDependencyIdentity | Mapping[str, Any] | str,
        *,
        replacement: (
            SemanticDependencyIdentity | Mapping[str, Any] | None
        ) = None,
        reason: str = "semantic_dependency_changed",
    ) -> InvalidationResult:
        """Invalidate results bound to one semantic dependency and descendants."""

        if isinstance(dependency, SemanticDependencyIdentity):
            dependency_id = dependency.dependency_id
            native_dependency = dependency
        elif isinstance(dependency, Mapping):
            native_dependency = SemanticDependencyIdentity.from_dict(dependency)
            dependency_id = native_dependency.dependency_id
        else:
            native_dependency = None
            dependency_id = _required_text(dependency, "dependency_id")
        if replacement is not None:
            native_replacement = (
                replacement
                if isinstance(replacement, SemanticDependencyIdentity)
                else SemanticDependencyIdentity.from_dict(replacement)
            )
            if native_dependency is not None and (
                native_replacement.namespace != native_dependency.namespace
                or native_replacement.key != native_dependency.key
            ):
                raise ForgedDependencyError(
                    "semantic dependency replacement must retain namespace/key"
                )
            if native_replacement.dependency_id == dependency_id:
                raise ValueError(
                    "semantic dependency replacement must have a new identity"
                )
        receipt = self.invalidate_batch(
            semantic_dependency_ids=(dependency_id,),
            reason=reason,
        )
        return InvalidationResult(
            root_artifact_ids=(dependency_id,),
            invalidated_artifact_ids=receipt.invalidated_artifact_ids,
            preserved_artifact_ids=receipt.preserved_artifact_ids,
            reason=reason,
        )

    def audit(self, *, rebuild: bool = True) -> RuntimeCASAuditReceipt:
        """Verify the disposable reverse index, tombstones, and journals."""

        issues = set(self._quarantine_reasons)
        before = set(self._known_artifact_ids)
        if rebuild:
            self._rebuild_graph()
        if before and before != self._known_artifact_ids:
            issues.add("corrupt_dependency_index")
        for path in sorted(self.keys_path.glob("*/*.json")):
            try:
                pointer = json.loads(path.read_bytes())
                if not isinstance(pointer, Mapping):
                    raise ArtifactIntegrityError("key pointer must be an object")
                content = {
                    "schema": pointer.get("schema"),
                    "key_id": pointer.get("key_id"),
                    "artifact_id": pointer.get("artifact_id"),
                }
                if (
                    content["schema"] != RUNTIME_CAS_SCHEMA
                    or path != self._key_path(
                        _required_text(content["key_id"], "key_id")
                    )
                    or pointer.get("pointer_digest")
                    != _sha256(canonical_runtime_json_bytes(content))
                    or content["artifact_id"] not in self._known_artifact_ids
                    or self._is_invalidated(str(content["artifact_id"]))
                ):
                    raise ArtifactIntegrityError(
                        "key pointer binding is corrupt or stale"
                    )
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                RuntimeCASError,
                TypeError,
                ValueError,
            ):
                issues.add("corrupt_dependency_index")
                self._quarantine_index_file(
                    path, "corrupt_dependency_index"
                )
        for path in sorted(self.projections_path.glob("*/*.json")):
            try:
                projection = AuthoritativeProjection.from_dict(
                    json.loads(path.read_bytes())
                )
                if (
                    path
                    != self._projection_path(
                        projection.namespace, projection.projection_key
                    )
                    or projection.artifact_id
                    not in self._known_artifact_ids
                    or self._is_invalidated(projection.artifact_id)
                ):
                    raise ArtifactIntegrityError(
                        "authoritative projection is corrupt or stale"
                    )
                artifact = self._read_host(projection.artifact_id)
                if (
                    artifact is None
                    or artifact.key.key_id != projection.key_id
                    or artifact.binding.tree_id != projection.tree_id
                ):
                    raise ArtifactIntegrityError(
                        "authoritative projection target is stale"
                    )
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                RuntimeCASError,
                TypeError,
                ValueError,
            ):
                issues.add("corrupt_dependency_index")
                self._quarantine_index_file(
                    path, "corrupt_dependency_index"
                )
        tombstoned: list[str] = []
        for path in sorted(self.tombstones_path.glob("*/*.json")):
            digest = path.stem
            artifact_id = f"runtime-artifact:sha256:{digest}"
            if self._is_invalidated(artifact_id):
                tombstoned.append(artifact_id)
            else:
                issues.add("corrupt_tombstone")
        for path in sorted(self.invalidations_path.glob("*.json")):
            try:
                receipt = CASInvalidationReceipt.from_dict(
                    json.loads(path.read_bytes())
                )
                if (
                    not receipt.committed
                    or path != self._invalidation_path(receipt.transaction_id)
                ):
                    issues.add("partial_invalidation_transaction")
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                RuntimeCASError,
                TypeError,
                ValueError,
            ):
                issues.add("corrupt_invalidation_journal")
        journal_exists = any(self.invalidations_path.glob("*.json"))
        if self.invalidation_head_path.exists():
            try:
                self.latest_invalidation()
            except (OSError, RuntimeCASError, TypeError, ValueError):
                issues.add("corrupt_invalidation_head")
                self._quarantine_index_file(
                    self.invalidation_head_path,
                    "corrupt_invalidation_head",
                )
        elif journal_exists:
            issues.add("missing_invalidation_head")
            self._quarantine_reasons.add("missing_invalidation_head")
        issues.update(self._quarantine_reasons)
        return RuntimeCASAuditReceipt(
            artifact_ids=tuple(sorted(self._known_artifact_ids)),
            tombstoned_artifact_ids=tuple(sorted(set(tombstoned))),
            issue_codes=tuple(sorted(issues)),
            rebuilt=rebuild,
        )

    audit_dependency_index = audit

    @property
    def quarantined(self) -> bool:
        return bool(self._quarantine_reasons)

    def latest_invalidation(self) -> CASInvalidationReceipt | None:
        try:
            value = json.loads(self.invalidation_head_path.read_bytes())
            claimed = str(value.get("head_digest") or "")
            body = {
                key: item for key, item in value.items() if key != "head_digest"
            }
            if (
                value.get("schema")
                != RUNTIME_INVALIDATION_TRANSACTION_SCHEMA
                or claimed != _sha256(canonical_runtime_json_bytes(body))
            ):
                raise ArtifactIntegrityError("invalidation head is corrupt")
            path = self._invalidation_path(str(value["transaction_id"]))
            receipt = CASInvalidationReceipt.from_dict(
                json.loads(path.read_bytes())
            )
            if not receipt.committed:
                raise ArtifactIntegrityError(
                    "invalidation head references an incomplete transaction"
                )
            return receipt
        except FileNotFoundError:
            return None
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            RuntimeCASError,
            TypeError,
            ValueError,
        ) as exc:
            self._quarantine_reasons.add("corrupt_invalidation_head")
            raise ArtifactIntegrityError(
                "invalidation head is corrupt"
            ) from exc

    def semantic_dependency_changed(
        self,
        previous_artifact_id: str,
        replacement_artifact_id: str | None = None,
    ) -> InvalidationResult:
        if replacement_artifact_id is not None:
            replacement = self.get(replacement_artifact_id)
            if replacement is None:
                raise ForgedDependencyError(
                    "replacement semantic dependency does not exist"
                )
            if replacement_artifact_id == previous_artifact_id:
                raise ValueError(
                    "semantic dependency replacement must have a new identity"
                )
        return self.invalidate_dependency(previous_artifact_id)

    def clear_process_cache(self) -> None:
        self._memory.clear()
        self._key_memory.clear()

    clear_memory = clear_process_cache

    def metrics(self) -> RuntimeCASMetrics:
        with self._metrics_lock:
            return RuntimeCASMetrics(**self._metrics_values)

    stats = metrics


TieredRuntimeCAS = RuntimeCAS
DependencyAwareRuntimeStore = RuntimeCAS
RuntimeArtifactStore = RuntimeCAS


def artifact_key(
    *,
    namespace: str,
    artifact_kind: str,
    authority: RuntimeAuthority | str,
    binding: ResultBinding | Mapping[str, Any],
    dependencies: Sequence[
        RuntimeArtifactRecord | ArtifactDependency | Mapping[str, Any] | str
    ] = (),
    payload_schema: str = "",
) -> RuntimeArtifactKey:
    """Build an exact key from already verified dependency identities."""

    dependency_ids: list[str] = []
    for item in dependencies:
        if isinstance(item, RuntimeArtifactRecord):
            dependency_ids.append(item.artifact_id)
        elif isinstance(item, ArtifactDependency):
            dependency_ids.append(item.artifact_id)
        elif isinstance(item, Mapping):
            dependency_ids.append(
                ArtifactDependency.from_dict(item).artifact_id
            )
        else:
            dependency_ids.append(_required_text(item, "dependency_id"))
    return RuntimeArtifactKey(
        namespace=namespace,
        artifact_kind=artifact_kind,
        authority=_coerce_authority(authority),
        binding=_coerce_binding(binding),
        dependency_ids=tuple(dependency_ids),
        payload_schema=payload_schema,
    )


build_artifact_key = artifact_key
build_runtime_artifact_key = artifact_key


__all__ = [
    "ArtifactAuthority",
    "ArtifactDependency",
    "ArtifactFreshness",
    "ArtifactIdentity",
    "ArtifactIntegrityError",
    "ArtifactLookup",
    "ArtifactRecord",
    "ArtifactTier",
    "AuthorityIsolationError",
    "AuthoritativeProjection",
    "CASInvalidationReceipt",
    "CanonicalArtifactIdentity",
    "DEFAULT_LOCK_TIMEOUT_SECONDS",
    "DEFAULT_MAX_PAYLOAD_BYTES",
    "DEPENDENCY_CAS_REQUIREMENT_ID",
    "DependencyAwareRuntimeStore",
    "DependencyCycleError",
    "DependencyEdge",
    "DependencyReference",
    "DirectorySharedImmutableStore",
    "EvidenceFreshness",
    "ForgedDependencyError",
    "ImmutableStoreError",
    "InvalidationResult",
    "NamespaceAuthority",
    "RUNTIME_ARTIFACT_IDENTITY_SCHEMA",
    "RUNTIME_ARTIFACT_KEY_SCHEMA",
    "RUNTIME_ARTIFACT_SCHEMA",
    "RUNTIME_CAS_SCHEMA",
    "RUNTIME_DEPENDENCY_SCHEMA",
    "RUNTIME_INVALIDATION_SCHEMA",
    "RUNTIME_INVALIDATION_TRANSACTION_SCHEMA",
    "RUNTIME_CAS_AUDIT_SCHEMA",
    "RUNTIME_PROJECTION_SCHEMA",
    "RuntimeArtifact",
    "RuntimeArtifactKey",
    "RuntimeArtifactRecord",
    "RuntimeArtifactStore",
    "RuntimeAuthority",
    "RuntimeCAS",
    "RuntimeCASAuditReceipt",
    "RuntimeCASError",
    "RuntimeCASLookup",
    "RuntimeCASMetrics",
    "RuntimeTier",
    "SharedCAS",
    "SharedImmutableStore",
    "StorageTier",
    "TieredRuntimeCAS",
    "artifact_key",
    "build_artifact_key",
    "build_runtime_artifact_key",
    "canonical_runtime_json_bytes",
]
