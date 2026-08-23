"""Container execution and worker-lease contract family (EAAEF-050).

These records are the shared serialization boundary for OCI worker launches.
They are immutable, DAG-JSON compatible, content addressed, and strictly
versioned at major ``@1``.  Unknown schema names, unknown major versions,
floats, private material, and hidden chain-of-thought are rejected.

A profile binds one image digest, worktree, task, authority, resource
reservation and isolation policy.  Leases, artifact manifests, checkpoints
and receipts reference that bind.  Host acceptance authority is not a worker
field: workers cannot self-approve, and only an independent supervisor
verifier may accept a receipt.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final, TypeAlias, TypeVar

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)


CONTAINER_CONTRACT_VERSION: Final[int] = 1
CONTRACT_VERSION: Final[int] = CONTAINER_CONTRACT_VERSION
SCHEMA_VERSION: Final[int] = CONTAINER_CONTRACT_VERSION

CONTAINER_EXECUTION_PROFILE_INTERFACE: Final[str] = "ContainerExecutionProfile@1"
WORKER_LEASE_INTERFACE: Final[str] = "WorkerLease@1"
ARTIFACT_MANIFEST_INTERFACE: Final[str] = "ArtifactManifest@1"
CONTAINER_CHECKPOINT_INTERFACE: Final[str] = "ContainerCheckpoint@1"
CONTAINER_RECEIPT_INTERFACE: Final[str] = "ContainerReceipt@1"
RESOURCE_BOUNDS_INTERFACE: Final[str] = "ContainerResourceBounds@1"
ISOLATION_POLICY_INTERFACE: Final[str] = "ContainerIsolationPolicy@1"

CONTAINER_EXECUTION_PROFILE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/container-execution-profile@1"
)
WORKER_LEASE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/worker-lease@1"
)
ARTIFACT_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/artifact-manifest@1"
)
CONTAINER_CHECKPOINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/container-checkpoint@1"
)
CONTAINER_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/container-receipt@1"
)
RESOURCE_BOUNDS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/container-resource-bounds@1"
)
ISOLATION_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/container-isolation-policy@1"
)

CONTAINER_CONTRACT_FAMILY: Final[Mapping[str, str]] = MappingProxyType(
    {
        "profile": CONTAINER_EXECUTION_PROFILE_INTERFACE,
        "lease": WORKER_LEASE_INTERFACE,
        "artifact_manifest": ARTIFACT_MANIFEST_INTERFACE,
        "checkpoint": CONTAINER_CHECKPOINT_INTERFACE,
        "receipt": CONTAINER_RECEIPT_INTERFACE,
    }
)

ABSOLUTE_MAX_RECORD_BYTES: Final[int] = 1_048_576
ABSOLUTE_MAX_TEXT_BYTES: Final[int] = 4_096
ABSOLUTE_MAX_ID_BYTES: Final[int] = 256
ABSOLUTE_MAX_PATH_BYTES: Final[int] = 1_024
ABSOLUTE_MAX_REASON_BYTES: Final[int] = 256
ABSOLUTE_MAX_MOUNTS: Final[int] = 16
ABSOLUTE_MAX_ARTIFACTS: Final[int] = 512
ABSOLUTE_MAX_CPU_MILLICORES: Final[int] = 64_000
ABSOLUTE_MAX_RAM_MIB: Final[int] = 262_144
ABSOLUTE_MAX_DISK_MIB: Final[int] = 1_048_576
ABSOLUTE_MAX_TIMEOUT_SECONDS: Final[int] = 7 * 24 * 60 * 60
ABSOLUTE_MAX_GPU_COUNT: Final[int] = 8

DEFAULT_CPU_MILLICORES: Final[int] = 1_000
DEFAULT_RAM_MIB: Final[int] = 512
DEFAULT_DISK_MIB: Final[int] = 1_024
DEFAULT_TIMEOUT_SECONDS: Final[int] = 60

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
_CIDV1_RE: Final[re.Pattern[str]] = re.compile(r"^b[a-z2-7]{20,}$")
_HEX64_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")
_ID_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/=+-]{0,255}$"
)

_HIDDEN_CHAIN_OF_THOUGHT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "chain_of_thought",
        "cot",
        "hidden_chain_of_thought",
        "hidden_cot",
        "hidden_reasoning",
        "hidden_thoughts",
        "internal_monologue",
        "model_thoughts",
        "private_reasoning",
        "private_thinking",
        "scratchpad",
        "thinking",
        "thinking_blocks",
        "thinking_private",
        "thinking_text",
    }
)
_PRIVATE_FIELD_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "hidden_witness",
        "password",
        "private_key",
        "private_premise",
        "private_witness",
        "refresh_token",
        "secret",
        "session_token",
        "witness",
    }
)
_HOST_ACCEPTANCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "accepted",
        "accepted_by",
        "acceptor",
        "acceptor_id",
        "acceptor_principal_id",
        "acceptance",
        "acceptance_authority",
        "accepted_by_host",
        "completion_accepted",
        "completion_authority",
        "completion_eligible",
        "host_acceptance",
        "host_acceptance_authority",
        "host_acceptor",
        "self_approval",
        "self_approved",
        "worker_accepted",
        "worker_acceptance",
    }
)
_DOCKER_SOCKET_MARKERS: Final[tuple[str, ...]] = (
    "docker.sock",
    "/var/run/docker",
    "/run/docker",
    "/.docker/run",
)

TEnum = TypeVar("TEnum", bound=Enum)


class ContainerContractError(ContractValidationError):
    """Malformed or unsafe container execution contract."""


class ContainerBoundsError(ContainerContractError):
    """A container value exceeded a declared resource bound."""


class ContainerIdentityError(ContainerContractError):
    """A claimed content identity did not match its canonical payload."""


class ContainerVersionError(ContainerContractError):
    """Unsupported container schema name or contract version."""


class ContainerTrustError(ContainerContractError):
    """A worker record attempted to grant host acceptance or self-approve."""


class NetworkPolicy(str, Enum):
    """Closed network vocabulary.  @1 admits deny only."""

    DENY = "deny"


class MountKind(str, Enum):
    """Closed mount classes admitted on a worker profile."""

    WORKTREE = "worktree"
    PROVIDER_AUTH = "provider_auth"
    SECRET = "secret"
    OTHER = "other"


class ContainerOutcome(str, Enum):
    """Worker-reported outcome.  This is not host acceptance."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    QUARANTINED = "quarantined"


def _normalize_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def _enum(value: Any, enum_type: type[TEnum], name: str) -> TEnum:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ContainerContractError(f"{name} must be one of: {allowed}") from exc


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    max_bytes: int = ABSOLUTE_MAX_TEXT_BYTES,
) -> str:
    if value is None:
        result = ""
    elif not isinstance(value, str):
        raise ContainerContractError(f"{name} must be a string")
    else:
        result = value.strip()
    if required and not result:
        raise ContainerContractError(f"{name} is required")
    if "\x00" in result:
        raise ContainerContractError(f"{name} must not contain NUL")
    if len(result.encode("utf-8")) > max_bytes:
        raise ContainerBoundsError(f"{name} exceeds {max_bytes} UTF-8 bytes")
    return result


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ContainerContractError(f"{name} must be a boolean")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContainerContractError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result < 1:
        raise ContainerContractError(f"{name} must be at least 1")
    return result


def _major_version(name: str) -> int | None:
    if not isinstance(name, str) or "@" not in name:
        return None
    suffix = name.rsplit("@", 1)[-1]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _require_versioned_name(name: str, expected: str, field_name: str) -> None:
    if name != expected:
        supplied_major = _major_version(name)
        expected_major = _major_version(expected) or CONTAINER_CONTRACT_VERSION
        if supplied_major is not None and supplied_major != expected_major:
            raise ContainerVersionError(
                f"unsupported {field_name} {name!r}; rebuild with {expected}"
            )
        raise ContainerVersionError(
            f"unsupported {field_name} {name!r}; expected {expected}"
        )


def _schema_and_version(
    payload: Mapping[str, Any],
    expected_schema: str,
    expected_interface: str,
    *,
    artifact_name: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise ContainerContractError(f"{artifact_name} payload must be an object")
    schema = payload.get("schema")
    if schema not in (None, "", expected_schema):
        _require_versioned_name(str(schema), expected_schema, "schema")
    interface = payload.get("interface")
    if interface not in (None, "", expected_interface):
        _require_versioned_name(str(interface), expected_interface, "interface")
    for key in ("contract_version", "schema_version"):
        version = payload.get(key)
        if version not in (None, "", CONTAINER_CONTRACT_VERSION):
            raise ContainerVersionError(
                f"unsupported {artifact_name} contract version; rebuild with "
                f"{expected_interface}"
            )


def _reject_unknown(
    payload: Mapping[str, Any], allowed: Iterable[str], *, artifact_name: str
) -> None:
    extra = set(payload).difference(allowed)
    if extra:
        raise ContainerContractError(
            f"{artifact_name} contains unsupported fields; rebuild its canonical payload"
        )


def _claimed_identity(
    payload: Mapping[str, Any],
    actual: str,
    *,
    names: Sequence[str],
    artifact_name: str,
) -> None:
    for name in names:
        claimed = payload.get(name)
        if claimed not in (None, "") and claimed != actual:
            raise ContainerIdentityError(
                f"{artifact_name} content identity does not match payload"
            )


def _digest_sha256(value: Any, name: str, *, required: bool = True) -> str:
    text = _text(value, name, required=required, max_bytes=80)
    if not text:
        return ""
    if _HEX64_RE.fullmatch(text):
        return f"sha256:{text}"
    if _SHA256_RE.fullmatch(text):
        return text
    raise ContainerContractError(f"{name} must be a sha256 hex digest")


def _content_ref(
    value: Any,
    name: str,
    *,
    required: bool = True,
    max_bytes: int = ABSOLUTE_MAX_ID_BYTES,
) -> str:
    text = _text(value, name, required=required, max_bytes=max_bytes)
    if not text:
        return ""
    if _SHA256_RE.fullmatch(text) or _CIDV1_RE.fullmatch(text):
        return text
    raise ContainerContractError(f"{name} must be a sha256 or CIDv1 identity")


def _record_id(value: Any, name: str) -> str:
    text = _text(value, name, max_bytes=ABSOLUTE_MAX_ID_BYTES)
    if _SHA256_RE.fullmatch(text) or _CIDV1_RE.fullmatch(text) or _ID_RE.fullmatch(text):
        return text
    raise ContainerContractError(f"{name} must be a bounded identifier")


def _key_is_forbidden(key: str) -> str | None:
    normalized = _normalize_key(key)
    if normalized in _HIDDEN_CHAIN_OF_THOUGHT_KEYS:
        return "hidden_chain_of_thought"
    if normalized in _HOST_ACCEPTANCE_KEYS:
        return "host_acceptance"
    if any(
        normalized == marker or normalized.endswith("_" + marker) or marker in normalized
        for marker in _PRIVATE_FIELD_MARKERS
    ):
        return "private_material"
    return None


def _reject_forbidden_keys(
    value: Any, *, name: str, worker_id: str = ""
) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            reason = _key_is_forbidden(str(raw_key))
            if reason == "hidden_chain_of_thought":
                raise ContainerContractError(
                    f"{name} must not represent hidden chain-of-thought"
                )
            if reason == "host_acceptance":
                _raise_host_acceptance(value, worker_id=worker_id, artifact_name=name)
            if reason == "private_material":
                raise ContainerContractError(
                    f"{name} must not contain private material"
                )
            _reject_forbidden_keys(item, name=name, worker_id=worker_id)
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        for item in value:
            _reject_forbidden_keys(item, name=name, worker_id=worker_id)


def _raise_host_acceptance(
    payload: Mapping[str, Any], *, worker_id: str, artifact_name: str
) -> None:
    worker = str(worker_id or payload.get("worker_id") or "").strip()
    for key in payload:
        if _normalize_key(key) not in _HOST_ACCEPTANCE_KEYS:
            continue
        claimed = payload.get(key)
        if isinstance(claimed, str) and worker and claimed.strip() == worker:
            raise ContainerTrustError(
                f"{artifact_name} workers cannot self-approve; "
                "host acceptance authority is not a worker field"
            )
        if claimed is True:
            raise ContainerTrustError(
                f"{artifact_name} workers cannot self-approve; "
                "host acceptance authority is not a worker field"
            )
    raise ContainerTrustError(
        f"{artifact_name} host acceptance authority is not a worker field"
    )


def _contains_docker_socket(value: str) -> bool:
    lowered = value.strip().lower().replace("\\", "/")
    if not lowered:
        return False
    return any(marker in lowered for marker in _DOCKER_SOCKET_MARKERS)


def _reject_docker_socket(*values: str, artifact_name: str) -> None:
    for value in values:
        if _contains_docker_socket(value):
            raise ContainerTrustError(
                f"{artifact_name} docker.sock mounts are prohibited"
            )


def _relative_path(value: Any, name: str) -> str:
    text = _text(value, name, max_bytes=ABSOLUTE_MAX_PATH_BYTES).replace("\\", "/")
    _reject_docker_socket(text, artifact_name=name)
    candidate = PurePosixPath(text)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or (candidate.parts and candidate.parts[0].endswith(":"))
    ):
        raise ContainerContractError(f"{name} must be repository-relative")
    normalized = candidate.as_posix().removeprefix("./")
    if normalized in ("", "."):
        raise ContainerContractError(f"{name} must not be empty")
    return normalized


def _distinct_identities(pairs: Sequence[tuple[str, str]]) -> None:
    seen: dict[str, str] = {}
    for name, identity in pairs:
        if not identity:
            continue
        previous = seen.get(identity)
        if previous is not None and previous != name:
            raise ContainerIdentityError(
                f"{name} identity must be distinct from {previous}"
            )
        seen[identity] = name


def _envelope(interface: str, body: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "interface": interface,
        "contract_version": CONTAINER_CONTRACT_VERSION,
        **dict(body),
    }


def _require_record_bound(record: CanonicalContract, *, artifact_name: str) -> None:
    size = len(record.canonical_bytes())
    if size > ABSOLUTE_MAX_RECORD_BYTES:
        raise ContainerBoundsError(
            f"{artifact_name} exceeds the absolute record bound of "
            f"{ABSOLUTE_MAX_RECORD_BYTES} bytes"
        )


class _ContainerCanonicalContract(CanonicalContract):
    """Canonical mixin that preserves container error types on decode."""

    INTERFACE: ClassVar[str] = ""

    @property
    def schema_version(self) -> int:
        return CONTAINER_CONTRACT_VERSION

    @property
    def interface(self) -> str:
        return self.INTERFACE

    @classmethod
    def from_json(cls, payload: str) -> "_ContainerCanonicalContract":
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ContainerContractError(
                "container contract JSON is malformed"
            ) from exc
        if not isinstance(value, Mapping):
            raise ContainerContractError(
                "container contract JSON must contain an object"
            )
        decoder = getattr(cls, "from_dict", None)
        if decoder is None:
            raise ContainerContractError(f"{cls.__name__} does not support from_dict")
        return decoder(value)


_COMMON_WIRE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema",
        "interface",
        "contract_version",
        "schema_version",
        "content_id",
        "cid",
        "identity",
        "canonical_id",
    }
)


def _reject_worker_acceptance_fields(
    payload: Mapping[str, Any], *, artifact_name: str, worker_id: str = ""
) -> None:
    present = [key for key in payload if _normalize_key(key) in _HOST_ACCEPTANCE_KEYS]
    if present:
        _raise_host_acceptance(payload, worker_id=worker_id, artifact_name=artifact_name)


@dataclass(frozen=True)
class ResourceBounds(_ContainerCanonicalContract):
    """Integer resource reservation bound into every worker profile."""

    SCHEMA: ClassVar[str] = RESOURCE_BOUNDS_SCHEMA
    INTERFACE: ClassVar[str] = RESOURCE_BOUNDS_INTERFACE

    cpu_millicores: int
    ram_mib: int
    disk_mib: int
    timeout_seconds: int
    gpu_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cpu_millicores",
            _positive_int(self.cpu_millicores, "cpu_millicores"),
        )
        object.__setattr__(self, "ram_mib", _positive_int(self.ram_mib, "ram_mib"))
        object.__setattr__(self, "disk_mib", _positive_int(self.disk_mib, "disk_mib"))
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_int(self.timeout_seconds, "timeout_seconds"),
        )
        object.__setattr__(
            self, "gpu_count", _nonnegative_int(self.gpu_count, "gpu_count")
        )
        if self.cpu_millicores > ABSOLUTE_MAX_CPU_MILLICORES:
            raise ContainerBoundsError("cpu_millicores exceeds the absolute limit")
        if self.ram_mib > ABSOLUTE_MAX_RAM_MIB:
            raise ContainerBoundsError("ram_mib exceeds the absolute limit")
        if self.disk_mib > ABSOLUTE_MAX_DISK_MIB:
            raise ContainerBoundsError("disk_mib exceeds the absolute limit")
        if self.timeout_seconds > ABSOLUTE_MAX_TIMEOUT_SECONDS:
            raise ContainerBoundsError("timeout_seconds exceeds the absolute limit")
        if self.gpu_count > ABSOLUTE_MAX_GPU_COUNT:
            raise ContainerBoundsError("gpu_count exceeds the absolute limit")

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "cpu_millicores": self.cpu_millicores,
                "ram_mib": self.ram_mib,
                "disk_mib": self.disk_mib,
                "timeout_seconds": self.timeout_seconds,
                "gpu_count": self.gpu_count,
            },
        )

    def admits(self, used: "ResourceUse") -> bool:
        return (
            used.cpu_millicores <= self.cpu_millicores
            and used.ram_mib <= self.ram_mib
            and used.disk_mib <= self.disk_mib
            and used.elapsed_seconds <= self.timeout_seconds
            and used.gpu_count <= self.gpu_count
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResourceBounds":
        _schema_and_version(
            payload, cls.SCHEMA, cls.INTERFACE, artifact_name="resource bounds"
        )
        _reject_worker_acceptance_fields(payload, artifact_name="resource bounds")
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "cpu_millicores",
                    "ram_mib",
                    "disk_mib",
                    "timeout_seconds",
                    "gpu_count",
                }
            ),
            artifact_name="resource bounds",
        )
        result = cls(
            cpu_millicores=payload.get("cpu_millicores"),
            ram_mib=payload.get("ram_mib"),
            disk_mib=payload.get("disk_mib"),
            timeout_seconds=payload.get("timeout_seconds"),
            gpu_count=payload.get("gpu_count", 0),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id"),
            artifact_name="resource bounds",
        )
        return result


def _coerce_resource_bounds(value: Any) -> ResourceBounds:
    if isinstance(value, ResourceBounds):
        return value
    if isinstance(value, Mapping):
        return ResourceBounds.from_dict(value)
    raise ContainerContractError("resources must be a ResourceBounds object")


@dataclass(frozen=True)
class ResourceUse:
    """Observed integer consumption.  Must not exceed the reserved bounds."""

    cpu_millicores: int = 0
    ram_mib: int = 0
    disk_mib: int = 0
    elapsed_seconds: int = 0
    gpu_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cpu_millicores",
            _nonnegative_int(self.cpu_millicores, "cpu_millicores"),
        )
        object.__setattr__(self, "ram_mib", _nonnegative_int(self.ram_mib, "ram_mib"))
        object.__setattr__(
            self, "disk_mib", _nonnegative_int(self.disk_mib, "disk_mib")
        )
        object.__setattr__(
            self,
            "elapsed_seconds",
            _nonnegative_int(self.elapsed_seconds, "elapsed_seconds"),
        )
        object.__setattr__(
            self, "gpu_count", _nonnegative_int(self.gpu_count, "gpu_count")
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "cpu_millicores": self.cpu_millicores,
            "ram_mib": self.ram_mib,
            "disk_mib": self.disk_mib,
            "elapsed_seconds": self.elapsed_seconds,
            "gpu_count": self.gpu_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ResourceUse":
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ContainerContractError("resource_use must be an object")
        _reject_unknown(
            payload,
            {
                "cpu_millicores",
                "ram_mib",
                "disk_mib",
                "elapsed_seconds",
                "timeout_seconds",
                "gpu_count",
            },
            artifact_name="resource use",
        )
        elapsed = payload.get("elapsed_seconds", payload.get("timeout_seconds", 0))
        return cls(
            cpu_millicores=payload.get("cpu_millicores", 0),
            ram_mib=payload.get("ram_mib", 0),
            disk_mib=payload.get("disk_mib", 0),
            elapsed_seconds=elapsed,
            gpu_count=payload.get("gpu_count", 0),
        )


def _coerce_resource_use(value: Any) -> ResourceUse:
    if value is None:
        return ResourceUse()
    if isinstance(value, ResourceUse):
        return value
    if isinstance(value, Mapping):
        return ResourceUse.from_dict(value)
    raise ContainerContractError("resource_use must be a ResourceUse object")


@dataclass(frozen=True)
class ContainerMount:
    """One bind-mount.  Docker sockets and host engines are prohibited."""

    source: str
    target: str
    read_only: bool = True
    kind: MountKind = MountKind.OTHER

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source",
            _text(self.source, "mount source", max_bytes=ABSOLUTE_MAX_PATH_BYTES),
        )
        object.__setattr__(
            self,
            "target",
            _text(self.target, "mount target", max_bytes=ABSOLUTE_MAX_PATH_BYTES),
        )
        object.__setattr__(self, "read_only", _bool(self.read_only, "read_only"))
        object.__setattr__(self, "kind", _enum(self.kind, MountKind, "kind"))
        _reject_docker_socket(
            self.source, self.target, artifact_name="container mount"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "read_only": self.read_only,
            "kind": self.kind.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContainerMount":
        if not isinstance(payload, Mapping):
            raise ContainerContractError("mount must be an object")
        _reject_unknown(
            payload,
            {"source", "target", "read_only", "kind"},
            artifact_name="container mount",
        )
        return cls(
            source=payload.get("source", ""),
            target=payload.get("target", ""),
            read_only=payload.get("read_only", True),
            kind=payload.get("kind", MountKind.OTHER),
        )


def _coerce_mounts(value: Any) -> tuple[ContainerMount, ...]:
    if value is None:
        items: Sequence[Any] = ()
    elif isinstance(value, (str, bytes, bytearray, memoryview)) or not isinstance(
        value, Sequence
    ):
        raise ContainerContractError("mounts must be a sequence")
    else:
        items = value
    if len(items) > ABSOLUTE_MAX_MOUNTS:
        raise ContainerBoundsError("mounts exceeds its item-count limit")
    result: list[ContainerMount] = []
    seen: set[str] = set()
    for item in items:
        mount = item if isinstance(item, ContainerMount) else ContainerMount.from_dict(item)
        key = f"{mount.source}->{mount.target}"
        if key in seen:
            raise ContainerContractError("mounts must not contain duplicate targets")
        seen.add(key)
        result.append(mount)
    return tuple(result)


@dataclass(frozen=True)
class IsolationPolicy(_ContainerCanonicalContract):
    """Default-deny isolation.  Docker sockets and new privileges are forbidden."""

    SCHEMA: ClassVar[str] = ISOLATION_POLICY_SCHEMA
    INTERFACE: ClassVar[str] = ISOLATION_POLICY_INTERFACE

    network_policy: NetworkPolicy = NetworkPolicy.DENY
    docker_socket_mounted: bool = False
    no_new_privileges: bool = True
    read_only_base: bool = True
    privileged: bool = False
    mounts: tuple[ContainerMount, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "network_policy",
            _enum(self.network_policy, NetworkPolicy, "network_policy"),
        )
        object.__setattr__(
            self,
            "docker_socket_mounted",
            _bool(self.docker_socket_mounted, "docker_socket_mounted"),
        )
        object.__setattr__(
            self,
            "no_new_privileges",
            _bool(self.no_new_privileges, "no_new_privileges"),
        )
        object.__setattr__(
            self, "read_only_base", _bool(self.read_only_base, "read_only_base")
        )
        object.__setattr__(self, "privileged", _bool(self.privileged, "privileged"))
        object.__setattr__(self, "mounts", _coerce_mounts(self.mounts))
        if self.network_policy is not NetworkPolicy.DENY:
            raise ContainerTrustError("network policy must default-deny")
        if self.docker_socket_mounted:
            raise ContainerTrustError("docker.sock mounts are prohibited")
        if not self.no_new_privileges:
            raise ContainerTrustError("no-new-privileges is required")
        if self.privileged:
            raise ContainerTrustError("privileged workers are prohibited")
        if not self.read_only_base:
            raise ContainerTrustError("read-only base filesystem is required")

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "network_policy": self.network_policy.value,
                "docker_socket_mounted": False,
                "no_new_privileges": True,
                "read_only_base": True,
                "privileged": False,
                "mounts": [mount.to_dict() for mount in self.mounts],
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "IsolationPolicy":
        if payload is None:
            return cls()
        _schema_and_version(
            payload, cls.SCHEMA, cls.INTERFACE, artifact_name="isolation policy"
        )
        _reject_worker_acceptance_fields(payload, artifact_name="isolation policy")
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "network_policy",
                    "docker_socket_mounted",
                    "no_new_privileges",
                    "read_only_base",
                    "privileged",
                    "mounts",
                }
            ),
            artifact_name="isolation policy",
        )
        result = cls(
            network_policy=payload.get("network_policy", NetworkPolicy.DENY),
            docker_socket_mounted=payload.get("docker_socket_mounted", False),
            no_new_privileges=payload.get("no_new_privileges", True),
            read_only_base=payload.get("read_only_base", True),
            privileged=payload.get("privileged", False),
            mounts=payload.get("mounts", ()),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id"),
            artifact_name="isolation policy",
        )
        return result


def _coerce_policy(value: Any) -> IsolationPolicy:
    if value is None:
        return IsolationPolicy()
    if isinstance(value, IsolationPolicy):
        return value
    if isinstance(value, Mapping):
        return IsolationPolicy.from_dict(value)
    raise ContainerContractError("policy must be an IsolationPolicy object")


def _bind_ids(
    *,
    image_digest: str,
    worktree_id: str,
    task_id: str,
    authority_id: str,
    worker_id: str = "",
) -> tuple[str, str, str, str, str]:
    digest = _digest_sha256(image_digest, "image_digest")
    worktree = _record_id(worktree_id, "worktree_id")
    task = _record_id(task_id, "task_id")
    authority = _record_id(authority_id, "authority_id")
    worker = _record_id(worker_id, "worker_id") if worker_id else ""
    if worker and worker == authority:
        raise ContainerTrustError(
            "workers cannot self-approve; worker identity cannot equal authority identity"
        )
    _distinct_identities(
        (
            ("image_digest", digest),
            ("worktree", worktree),
            ("task", task),
            ("authority", authority),
            ("worker", worker),
        )
    )
    return digest, worktree, task, authority, worker


@dataclass(frozen=True)
class ArtifactEntry:
    """One content-addressed worker artifact.  Bodies stay out of the record."""

    path: str
    content_id: str
    byte_count: int = 0
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path, "path"))
        object.__setattr__(
            self, "content_id", _content_ref(self.content_id, "content_id")
        )
        object.__setattr__(
            self, "byte_count", _nonnegative_int(self.byte_count, "byte_count")
        )
        object.__setattr__(
            self,
            "media_type",
            _text(self.media_type, "media_type", max_bytes=ABSOLUTE_MAX_ID_BYTES),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "content_id": self.content_id,
            "byte_count": self.byte_count,
            "media_type": self.media_type,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactEntry":
        if not isinstance(payload, Mapping):
            raise ContainerContractError("artifact entry must be an object")
        _reject_unknown(
            payload,
            {"path", "content_id", "byte_count", "media_type"},
            artifact_name="artifact entry",
        )
        return cls(
            path=payload.get("path", ""),
            content_id=payload.get("content_id", ""),
            byte_count=payload.get("byte_count", 0),
            media_type=payload.get("media_type", "application/octet-stream"),
        )


def _coerce_artifacts(value: Any) -> tuple[ArtifactEntry, ...]:
    if value is None:
        items: Sequence[Any] = ()
    elif isinstance(value, (str, bytes, bytearray, memoryview)) or not isinstance(
        value, Sequence
    ):
        raise ContainerContractError("artifacts must be a sequence")
    else:
        items = value
    if len(items) > ABSOLUTE_MAX_ARTIFACTS:
        raise ContainerBoundsError("artifacts exceeds its item-count limit")
    result: list[ArtifactEntry] = []
    seen_paths: set[str] = set()
    seen_ids: set[str] = set()
    for item in items:
        entry = (
            item if isinstance(item, ArtifactEntry) else ArtifactEntry.from_dict(item)
        )
        if entry.path in seen_paths:
            raise ContainerContractError("artifacts must not contain duplicate paths")
        if entry.content_id in seen_ids:
            raise ContainerContractError(
                "artifacts must not contain duplicate identities"
            )
        seen_paths.add(entry.path)
        seen_ids.add(entry.content_id)
        result.append(entry)
    return tuple(result)


@dataclass(frozen=True)
class ContainerExecutionProfile(_ContainerCanonicalContract):
    """Bind image, worktree, task, authority, resources and isolation policy."""

    SCHEMA: ClassVar[str] = CONTAINER_EXECUTION_PROFILE_SCHEMA
    INTERFACE: ClassVar[str] = CONTAINER_EXECUTION_PROFILE_INTERFACE

    image_digest: str
    worktree_id: str
    task_id: str
    authority_id: str
    resources: ResourceBounds
    policy: IsolationPolicy = IsolationPolicy()
    created_at_ms: int = 0

    def __post_init__(self) -> None:
        digest, worktree, task, authority, _worker = _bind_ids(
            image_digest=self.image_digest,
            worktree_id=self.worktree_id,
            task_id=self.task_id,
            authority_id=self.authority_id,
        )
        object.__setattr__(self, "image_digest", digest)
        object.__setattr__(self, "worktree_id", worktree)
        object.__setattr__(self, "task_id", task)
        object.__setattr__(self, "authority_id", authority)
        object.__setattr__(self, "resources", _coerce_resource_bounds(self.resources))
        object.__setattr__(self, "policy", _coerce_policy(self.policy))
        object.__setattr__(
            self, "created_at_ms", _nonnegative_int(self.created_at_ms, "created_at_ms")
        )
        _reject_forbidden_keys(self.to_dict(), name="container execution profile")
        _require_record_bound(self, artifact_name="container execution profile")

    @property
    def profile_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "image_digest": self.image_digest,
                "worktree_id": self.worktree_id,
                "task_id": self.task_id,
                "authority_id": self.authority_id,
                "resources": self.resources.to_dict(),
                "policy": self.policy.to_dict(),
                "created_at_ms": self.created_at_ms,
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContainerExecutionProfile":
        _schema_and_version(
            payload,
            cls.SCHEMA,
            cls.INTERFACE,
            artifact_name="container execution profile",
        )
        _reject_worker_acceptance_fields(
            payload, artifact_name="container execution profile"
        )
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "image_digest",
                    "worktree_id",
                    "task_id",
                    "authority_id",
                    "resources",
                    "policy",
                    "created_at_ms",
                    "profile_id",
                }
            ),
            artifact_name="container execution profile",
        )
        result = cls(
            image_digest=payload.get("image_digest", ""),
            worktree_id=payload.get("worktree_id", ""),
            task_id=payload.get("task_id", ""),
            authority_id=payload.get("authority_id", ""),
            resources=payload.get("resources"),
            policy=payload.get("policy"),
            created_at_ms=payload.get("created_at_ms", 0),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "profile_id"),
            artifact_name="container execution profile",
        )
        return result


@dataclass(frozen=True)
class WorkerLease(_ContainerCanonicalContract):
    """Exclusive fenced lease for one worker on one bound profile."""

    SCHEMA: ClassVar[str] = WORKER_LEASE_SCHEMA
    INTERFACE: ClassVar[str] = WORKER_LEASE_INTERFACE

    worker_id: str
    profile_id: str
    image_digest: str
    worktree_id: str
    task_id: str
    authority_id: str
    fencing_token: int = 1
    issued_at_ms: int = 0
    expires_at_ms: int = 0
    active: bool = True

    def __post_init__(self) -> None:
        digest, worktree, task, authority, worker = _bind_ids(
            image_digest=self.image_digest,
            worktree_id=self.worktree_id,
            task_id=self.task_id,
            authority_id=self.authority_id,
            worker_id=self.worker_id,
        )
        object.__setattr__(self, "worker_id", worker)
        object.__setattr__(self, "profile_id", _content_ref(self.profile_id, "profile_id"))
        object.__setattr__(self, "image_digest", digest)
        object.__setattr__(self, "worktree_id", worktree)
        object.__setattr__(self, "task_id", task)
        object.__setattr__(self, "authority_id", authority)
        object.__setattr__(
            self, "fencing_token", _positive_int(self.fencing_token, "fencing_token")
        )
        object.__setattr__(
            self, "issued_at_ms", _nonnegative_int(self.issued_at_ms, "issued_at_ms")
        )
        object.__setattr__(
            self, "expires_at_ms", _nonnegative_int(self.expires_at_ms, "expires_at_ms")
        )
        object.__setattr__(self, "active", _bool(self.active, "active"))
        if self.expires_at_ms and self.expires_at_ms < self.issued_at_ms:
            raise ContainerContractError("expires_at_ms cannot precede issued_at_ms")
        _distinct_identities(
            (
                ("profile", self.profile_id),
                ("worker", self.worker_id),
                ("worktree", self.worktree_id),
                ("task", self.task_id),
                ("authority", self.authority_id),
                ("image_digest", self.image_digest),
            )
        )
        _reject_forbidden_keys(
            self.to_dict(), name="worker lease", worker_id=self.worker_id
        )
        _require_record_bound(self, artifact_name="worker lease")

    @property
    def lease_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "worker_id": self.worker_id,
                "profile_id": self.profile_id,
                "image_digest": self.image_digest,
                "worktree_id": self.worktree_id,
                "task_id": self.task_id,
                "authority_id": self.authority_id,
                "fencing_token": self.fencing_token,
                "issued_at_ms": self.issued_at_ms,
                "expires_at_ms": self.expires_at_ms,
                "active": self.active,
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkerLease":
        _schema_and_version(
            payload, cls.SCHEMA, cls.INTERFACE, artifact_name="worker lease"
        )
        _reject_worker_acceptance_fields(
            payload,
            artifact_name="worker lease",
            worker_id=str(payload.get("worker_id") or ""),
        )
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "worker_id",
                    "profile_id",
                    "image_digest",
                    "worktree_id",
                    "task_id",
                    "authority_id",
                    "fencing_token",
                    "issued_at_ms",
                    "expires_at_ms",
                    "active",
                    "lease_id",
                }
            ),
            artifact_name="worker lease",
        )
        result = cls(
            worker_id=payload.get("worker_id", ""),
            profile_id=payload.get("profile_id", ""),
            image_digest=payload.get("image_digest", ""),
            worktree_id=payload.get("worktree_id", ""),
            task_id=payload.get("task_id", ""),
            authority_id=payload.get("authority_id", ""),
            fencing_token=payload.get("fencing_token", 1),
            issued_at_ms=payload.get("issued_at_ms", 0),
            expires_at_ms=payload.get("expires_at_ms", 0),
            active=payload.get("active", True),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "lease_id"),
            artifact_name="worker lease",
        )
        return result


@dataclass(frozen=True)
class ArtifactManifest(_ContainerCanonicalContract):
    """Content-addressed artifact list bound to one leased worker execution."""

    SCHEMA: ClassVar[str] = ARTIFACT_MANIFEST_SCHEMA
    INTERFACE: ClassVar[str] = ARTIFACT_MANIFEST_INTERFACE

    profile_id: str
    lease_id: str
    worktree_id: str
    task_id: str
    artifacts: tuple[ArtifactEntry, ...] = ()
    created_at_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _content_ref(self.profile_id, "profile_id"))
        object.__setattr__(self, "lease_id", _content_ref(self.lease_id, "lease_id"))
        object.__setattr__(self, "worktree_id", _record_id(self.worktree_id, "worktree_id"))
        object.__setattr__(self, "task_id", _record_id(self.task_id, "task_id"))
        object.__setattr__(self, "artifacts", _coerce_artifacts(self.artifacts))
        object.__setattr__(
            self, "created_at_ms", _nonnegative_int(self.created_at_ms, "created_at_ms")
        )
        _distinct_identities(
            (
                ("profile", self.profile_id),
                ("lease", self.lease_id),
                ("worktree", self.worktree_id),
                ("task", self.task_id),
            )
        )
        _reject_forbidden_keys(self.to_dict(), name="artifact manifest")
        _require_record_bound(self, artifact_name="artifact manifest")

    @property
    def manifest_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "profile_id": self.profile_id,
                "lease_id": self.lease_id,
                "worktree_id": self.worktree_id,
                "task_id": self.task_id,
                "artifacts": [item.to_dict() for item in self.artifacts],
                "created_at_ms": self.created_at_ms,
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactManifest":
        _schema_and_version(
            payload, cls.SCHEMA, cls.INTERFACE, artifact_name="artifact manifest"
        )
        _reject_worker_acceptance_fields(payload, artifact_name="artifact manifest")
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "profile_id",
                    "lease_id",
                    "worktree_id",
                    "task_id",
                    "artifacts",
                    "created_at_ms",
                    "manifest_id",
                }
            ),
            artifact_name="artifact manifest",
        )
        result = cls(
            profile_id=payload.get("profile_id", ""),
            lease_id=payload.get("lease_id", ""),
            worktree_id=payload.get("worktree_id", ""),
            task_id=payload.get("task_id", ""),
            artifacts=payload.get("artifacts", ()),
            created_at_ms=payload.get("created_at_ms", 0),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "manifest_id"),
            artifact_name="artifact manifest",
        )
        return result


@dataclass(frozen=True)
class ContainerCheckpoint(_ContainerCanonicalContract):
    """Restart-safe checkpoint bound to one leased worker execution."""

    SCHEMA: ClassVar[str] = CONTAINER_CHECKPOINT_SCHEMA
    INTERFACE: ClassVar[str] = CONTAINER_CHECKPOINT_INTERFACE

    profile_id: str
    lease_id: str
    worktree_id: str
    task_id: str
    artifact_manifest_id: str
    sequence: int = 0
    tree_id: str = ""
    resource_use: ResourceUse = ResourceUse()
    restart_safe: bool = True
    created_at_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _content_ref(self.profile_id, "profile_id"))
        object.__setattr__(self, "lease_id", _content_ref(self.lease_id, "lease_id"))
        object.__setattr__(self, "worktree_id", _record_id(self.worktree_id, "worktree_id"))
        object.__setattr__(self, "task_id", _record_id(self.task_id, "task_id"))
        object.__setattr__(
            self,
            "artifact_manifest_id",
            _content_ref(self.artifact_manifest_id, "artifact_manifest_id"),
        )
        object.__setattr__(self, "sequence", _nonnegative_int(self.sequence, "sequence"))
        object.__setattr__(
            self,
            "tree_id",
            _text(self.tree_id, "tree_id", required=False, max_bytes=ABSOLUTE_MAX_ID_BYTES),
        )
        object.__setattr__(self, "resource_use", _coerce_resource_use(self.resource_use))
        object.__setattr__(self, "restart_safe", _bool(self.restart_safe, "restart_safe"))
        if not self.restart_safe:
            raise ContainerContractError("container checkpoints must be restart-safe")
        object.__setattr__(
            self, "created_at_ms", _nonnegative_int(self.created_at_ms, "created_at_ms")
        )
        _distinct_identities(
            (
                ("profile", self.profile_id),
                ("lease", self.lease_id),
                ("worktree", self.worktree_id),
                ("task", self.task_id),
                ("artifact_manifest", self.artifact_manifest_id),
                ("tree", self.tree_id),
            )
        )
        _reject_forbidden_keys(self.to_dict(), name="container checkpoint")
        _require_record_bound(self, artifact_name="container checkpoint")

    @property
    def checkpoint_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "profile_id": self.profile_id,
                "lease_id": self.lease_id,
                "worktree_id": self.worktree_id,
                "task_id": self.task_id,
                "artifact_manifest_id": self.artifact_manifest_id,
                "sequence": self.sequence,
                "tree_id": self.tree_id,
                "resource_use": self.resource_use.to_dict(),
                "restart_safe": True,
                "created_at_ms": self.created_at_ms,
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContainerCheckpoint":
        _schema_and_version(
            payload, cls.SCHEMA, cls.INTERFACE, artifact_name="container checkpoint"
        )
        _reject_worker_acceptance_fields(
            payload, artifact_name="container checkpoint"
        )
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "profile_id",
                    "lease_id",
                    "worktree_id",
                    "task_id",
                    "artifact_manifest_id",
                    "sequence",
                    "tree_id",
                    "resource_use",
                    "restart_safe",
                    "created_at_ms",
                    "checkpoint_id",
                }
            ),
            artifact_name="container checkpoint",
        )
        if payload.get("restart_safe") not in (None, True):
            raise ContainerContractError("container checkpoints must be restart-safe")
        result = cls(
            profile_id=payload.get("profile_id", ""),
            lease_id=payload.get("lease_id", ""),
            worktree_id=payload.get("worktree_id", ""),
            task_id=payload.get("task_id", ""),
            artifact_manifest_id=payload.get("artifact_manifest_id", ""),
            sequence=payload.get("sequence", 0),
            tree_id=payload.get("tree_id", ""),
            resource_use=payload.get("resource_use"),
            restart_safe=True,
            created_at_ms=payload.get("created_at_ms", 0),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "checkpoint_id"),
            artifact_name="container checkpoint",
        )
        return result


@dataclass(frozen=True)
class ContainerReceipt(_ContainerCanonicalContract):
    """Worker evidence receipt.  Host acceptance is never a field on this record."""

    SCHEMA: ClassVar[str] = CONTAINER_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = CONTAINER_RECEIPT_INTERFACE

    worker_id: str
    profile_id: str
    lease_id: str
    image_digest: str
    worktree_id: str
    task_id: str
    authority_id: str
    artifact_manifest_id: str
    checkpoint_id: str
    outcome: ContainerOutcome = ContainerOutcome.COMPLETED
    resource_use: ResourceUse = ResourceUse()
    reason_code: str = ""
    created_at_ms: int = 0

    def __post_init__(self) -> None:
        digest, worktree, task, authority, worker = _bind_ids(
            image_digest=self.image_digest,
            worktree_id=self.worktree_id,
            task_id=self.task_id,
            authority_id=self.authority_id,
            worker_id=self.worker_id,
        )
        object.__setattr__(self, "worker_id", worker)
        object.__setattr__(self, "profile_id", _content_ref(self.profile_id, "profile_id"))
        object.__setattr__(self, "lease_id", _content_ref(self.lease_id, "lease_id"))
        object.__setattr__(self, "image_digest", digest)
        object.__setattr__(self, "worktree_id", worktree)
        object.__setattr__(self, "task_id", task)
        object.__setattr__(self, "authority_id", authority)
        object.__setattr__(
            self,
            "artifact_manifest_id",
            _content_ref(self.artifact_manifest_id, "artifact_manifest_id"),
        )
        object.__setattr__(
            self, "checkpoint_id", _content_ref(self.checkpoint_id, "checkpoint_id")
        )
        object.__setattr__(
            self, "outcome", _enum(self.outcome, ContainerOutcome, "outcome")
        )
        object.__setattr__(self, "resource_use", _coerce_resource_use(self.resource_use))
        object.__setattr__(
            self,
            "reason_code",
            _text(
                self.reason_code,
                "reason_code",
                required=False,
                max_bytes=ABSOLUTE_MAX_REASON_BYTES,
            ),
        )
        object.__setattr__(
            self, "created_at_ms", _nonnegative_int(self.created_at_ms, "created_at_ms")
        )
        _distinct_identities(
            (
                ("profile", self.profile_id),
                ("lease", self.lease_id),
                ("artifact_manifest", self.artifact_manifest_id),
                ("checkpoint", self.checkpoint_id),
                ("worker", self.worker_id),
                ("worktree", self.worktree_id),
                ("task", self.task_id),
                ("authority", self.authority_id),
                ("image_digest", self.image_digest),
            )
        )
        _reject_forbidden_keys(
            self.to_dict(), name="container receipt", worker_id=self.worker_id
        )
        _require_record_bound(self, artifact_name="container receipt")

    @property
    def receipt_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return _envelope(
            self.INTERFACE,
            {
                "worker_id": self.worker_id,
                "profile_id": self.profile_id,
                "lease_id": self.lease_id,
                "image_digest": self.image_digest,
                "worktree_id": self.worktree_id,
                "task_id": self.task_id,
                "authority_id": self.authority_id,
                "artifact_manifest_id": self.artifact_manifest_id,
                "checkpoint_id": self.checkpoint_id,
                "outcome": self.outcome.value,
                "resource_use": self.resource_use.to_dict(),
                "reason_code": self.reason_code,
                "created_at_ms": self.created_at_ms,
            },
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContainerReceipt":
        _schema_and_version(
            payload, cls.SCHEMA, cls.INTERFACE, artifact_name="container receipt"
        )
        _reject_worker_acceptance_fields(
            payload,
            artifact_name="container receipt",
            worker_id=str(payload.get("worker_id") or ""),
        )
        _reject_unknown(
            payload,
            _COMMON_WIRE_FIELDS.union(
                {
                    "worker_id",
                    "profile_id",
                    "lease_id",
                    "image_digest",
                    "worktree_id",
                    "task_id",
                    "authority_id",
                    "artifact_manifest_id",
                    "checkpoint_id",
                    "outcome",
                    "resource_use",
                    "reason_code",
                    "created_at_ms",
                    "receipt_id",
                }
            ),
            artifact_name="container receipt",
        )
        result = cls(
            worker_id=payload.get("worker_id", ""),
            profile_id=payload.get("profile_id", ""),
            lease_id=payload.get("lease_id", ""),
            image_digest=payload.get("image_digest", ""),
            worktree_id=payload.get("worktree_id", ""),
            task_id=payload.get("task_id", ""),
            authority_id=payload.get("authority_id", ""),
            artifact_manifest_id=payload.get("artifact_manifest_id", ""),
            checkpoint_id=payload.get("checkpoint_id", ""),
            outcome=payload.get("outcome", ContainerOutcome.COMPLETED),
            resource_use=payload.get("resource_use"),
            reason_code=payload.get("reason_code", ""),
            created_at_ms=payload.get("created_at_ms", 0),
        )
        _claimed_identity(
            payload,
            result.content_id,
            names=("content_id", "cid", "identity", "canonical_id", "receipt_id"),
            artifact_name="container receipt",
        )
        return result


@dataclass(frozen=True)
class ContainerExecutionBinding:
    """One fail-closed bind of the five @1 container execution contracts."""

    profile: ContainerExecutionProfile
    lease: WorkerLease
    artifact_manifest: ArtifactManifest
    checkpoint: ContainerCheckpoint
    receipt: ContainerReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.profile, ContainerExecutionProfile):
            raise ContainerContractError("profile must be a ContainerExecutionProfile")
        if not isinstance(self.lease, WorkerLease):
            raise ContainerContractError("lease must be a WorkerLease")
        if not isinstance(self.artifact_manifest, ArtifactManifest):
            raise ContainerContractError(
                "artifact_manifest must be an ArtifactManifest"
            )
        if not isinstance(self.checkpoint, ContainerCheckpoint):
            raise ContainerContractError("checkpoint must be a ContainerCheckpoint")
        if not isinstance(self.receipt, ContainerReceipt):
            raise ContainerContractError("receipt must be a ContainerReceipt")
        if self.lease.profile_id != self.profile.profile_id:
            raise ContainerIdentityError("lease profile_id must match the bound profile")
        if self.artifact_manifest.lease_id != self.lease.lease_id:
            raise ContainerIdentityError("artifact manifest lease_id must match the lease")
        if self.checkpoint.artifact_manifest_id != self.artifact_manifest.manifest_id:
            raise ContainerIdentityError(
                "checkpoint artifact_manifest_id must match the manifest"
            )
        if self.receipt.checkpoint_id != self.checkpoint.checkpoint_id:
            raise ContainerIdentityError("receipt checkpoint_id must match the checkpoint")
        if not self.profile.resources.admits(self.receipt.resource_use):
            raise ContainerBoundsError("receipt resource_use exceeds reserved bounds")
        if not self.profile.resources.admits(self.checkpoint.resource_use):
            raise ContainerBoundsError("checkpoint resource_use exceeds reserved bounds")


def bind_container_execution(
    *,
    image_digest: str,
    worktree_id: str,
    task_id: str,
    authority_id: str,
    worker_id: str,
    resources: ResourceBounds | Mapping[str, Any],
    policy: IsolationPolicy | Mapping[str, Any] | None = None,
    artifacts: Sequence[ArtifactEntry | Mapping[str, Any]] = (),
    fencing_token: int = 1,
    issued_at_ms: int = 0,
    expires_at_ms: int = 0,
    tree_id: str = "",
    outcome: ContainerOutcome | str = ContainerOutcome.COMPLETED,
    resource_use: ResourceUse | Mapping[str, Any] | None = None,
    reason_code: str = "",
    created_at_ms: int = 0,
    **forbidden: Any,
) -> ContainerExecutionBinding:
    """Bind image, worktree, task, authority, resources, policy, artifacts and receipt.

    Host acceptance authority is rejected: workers cannot self-approve.
    """

    if forbidden:
        _raise_host_acceptance(
            forbidden,
            worker_id=str(worker_id or ""),
            artifact_name="container execution binding",
        )
    profile = ContainerExecutionProfile(
        image_digest=image_digest,
        worktree_id=worktree_id,
        task_id=task_id,
        authority_id=authority_id,
        resources=resources,
        policy=policy,
        created_at_ms=created_at_ms,
    )
    lease = WorkerLease(
        worker_id=worker_id,
        profile_id=profile.profile_id,
        image_digest=profile.image_digest,
        worktree_id=profile.worktree_id,
        task_id=profile.task_id,
        authority_id=profile.authority_id,
        fencing_token=fencing_token,
        issued_at_ms=issued_at_ms,
        expires_at_ms=expires_at_ms,
    )
    manifest = ArtifactManifest(
        profile_id=profile.profile_id,
        lease_id=lease.lease_id,
        worktree_id=profile.worktree_id,
        task_id=profile.task_id,
        artifacts=artifacts,
        created_at_ms=created_at_ms,
    )
    used = _coerce_resource_use(resource_use)
    if not profile.resources.admits(used):
        raise ContainerBoundsError("resource_use exceeds reserved bounds")
    checkpoint = ContainerCheckpoint(
        profile_id=profile.profile_id,
        lease_id=lease.lease_id,
        worktree_id=profile.worktree_id,
        task_id=profile.task_id,
        artifact_manifest_id=manifest.manifest_id,
        tree_id=tree_id,
        resource_use=used,
        created_at_ms=created_at_ms,
    )
    receipt = ContainerReceipt(
        worker_id=lease.worker_id,
        profile_id=profile.profile_id,
        lease_id=lease.lease_id,
        image_digest=profile.image_digest,
        worktree_id=profile.worktree_id,
        task_id=profile.task_id,
        authority_id=profile.authority_id,
        artifact_manifest_id=manifest.manifest_id,
        checkpoint_id=checkpoint.checkpoint_id,
        outcome=outcome,
        resource_use=used,
        reason_code=reason_code,
        created_at_ms=created_at_ms,
    )
    return ContainerExecutionBinding(
        profile=profile,
        lease=lease,
        artifact_manifest=manifest,
        checkpoint=checkpoint,
        receipt=receipt,
    )


_RECORD_DECODERS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        RESOURCE_BOUNDS_SCHEMA: ResourceBounds.from_dict,
        RESOURCE_BOUNDS_INTERFACE: ResourceBounds.from_dict,
        ISOLATION_POLICY_SCHEMA: IsolationPolicy.from_dict,
        ISOLATION_POLICY_INTERFACE: IsolationPolicy.from_dict,
        CONTAINER_EXECUTION_PROFILE_SCHEMA: ContainerExecutionProfile.from_dict,
        CONTAINER_EXECUTION_PROFILE_INTERFACE: ContainerExecutionProfile.from_dict,
        WORKER_LEASE_SCHEMA: WorkerLease.from_dict,
        WORKER_LEASE_INTERFACE: WorkerLease.from_dict,
        ARTIFACT_MANIFEST_SCHEMA: ArtifactManifest.from_dict,
        ARTIFACT_MANIFEST_INTERFACE: ArtifactManifest.from_dict,
        CONTAINER_CHECKPOINT_SCHEMA: ContainerCheckpoint.from_dict,
        CONTAINER_CHECKPOINT_INTERFACE: ContainerCheckpoint.from_dict,
        CONTAINER_RECEIPT_SCHEMA: ContainerReceipt.from_dict,
        CONTAINER_RECEIPT_INTERFACE: ContainerReceipt.from_dict,
    }
)

ContainerRecord: TypeAlias = (
    ResourceBounds
    | IsolationPolicy
    | ContainerExecutionProfile
    | WorkerLease
    | ArtifactManifest
    | ContainerCheckpoint
    | ContainerReceipt
)


def decode_container_contract(
    payload: Mapping[str, Any] | ContainerRecord,
) -> ContainerRecord:
    """Decode any strictly versioned container execution family record."""

    if isinstance(payload, CanonicalContract):
        return payload  # type: ignore[return-value]
    if not isinstance(payload, Mapping):
        raise ContainerContractError("container contract payload must be an object")
    for key in (payload.get("schema"), payload.get("interface")):
        decoder = _RECORD_DECODERS.get(str(key) if key is not None else "")
        if decoder is not None:
            return decoder(payload)
    raise ContainerVersionError("unsupported container contract schema")


def canonical_container_json_bytes(value: Any) -> bytes:
    """Encode one container value as canonical DAG-JSON UTF-8 bytes."""

    if isinstance(value, CanonicalContract):
        return value.canonical_bytes()
    return canonical_json_bytes(value)


__all__ = (
    "ABSOLUTE_MAX_CPU_MILLICORES",
    "ABSOLUTE_MAX_DISK_MIB",
    "ABSOLUTE_MAX_GPU_COUNT",
    "ABSOLUTE_MAX_RAM_MIB",
    "ABSOLUTE_MAX_RECORD_BYTES",
    "ABSOLUTE_MAX_TIMEOUT_SECONDS",
    "ARTIFACT_MANIFEST_INTERFACE",
    "ARTIFACT_MANIFEST_SCHEMA",
    "CONTAINER_CHECKPOINT_INTERFACE",
    "CONTAINER_CHECKPOINT_SCHEMA",
    "CONTAINER_CONTRACT_FAMILY",
    "CONTAINER_CONTRACT_VERSION",
    "CONTAINER_EXECUTION_PROFILE_INTERFACE",
    "CONTAINER_EXECUTION_PROFILE_SCHEMA",
    "CONTAINER_RECEIPT_INTERFACE",
    "CONTAINER_RECEIPT_SCHEMA",
    "CONTRACT_VERSION",
    "ISOLATION_POLICY_INTERFACE",
    "ISOLATION_POLICY_SCHEMA",
    "RESOURCE_BOUNDS_INTERFACE",
    "RESOURCE_BOUNDS_SCHEMA",
    "SCHEMA_VERSION",
    "WORKER_LEASE_INTERFACE",
    "WORKER_LEASE_SCHEMA",
    "ArtifactEntry",
    "ArtifactManifest",
    "ContainerBoundsError",
    "ContainerCheckpoint",
    "ContainerContractError",
    "ContainerExecutionBinding",
    "ContainerExecutionProfile",
    "ContainerIdentityError",
    "ContainerMount",
    "ContainerOutcome",
    "ContainerReceipt",
    "ContainerTrustError",
    "ContainerVersionError",
    "IsolationPolicy",
    "MountKind",
    "NetworkPolicy",
    "ResourceBounds",
    "ResourceUse",
    "WorkerLease",
    "bind_container_execution",
    "canonical_container_json_bytes",
    "decode_container_contract",
)
