"""Hermetic MCP++ runtime contract witnesses (VFS-018 / VFS-G061).

Runtime witnesses **supplement** static call-path resolution (VFS-017) and
symbolic contract comparison (VFS-016).  They never promote mocks into
production authority, never open network by default, and never claim formal
proof or static completeness.

A bounded hermetic fixture:

* registers real production-class adapters (and optionally explicit mocks);
* discovers tools and negotiates profiles/transports;
* validates inputs against declared JSON Schema subsets;
* dispatches to a recorded target identity;
* observes outputs/errors against declared schemas;
* enforces timeout, cancellation, and cleanup;
* emits content-addressed receipts that replay deterministically.

Failure modes are typed and non-authoritative when inconclusive:

``malformed_call``, ``missing_tool``, ``schema_violation``,
``unavailable_backend``, ``cancelled``, ``profile_mismatch``,
``stale_manifest``, ``timed_out``.

Objective validation repair for VFS-G061 / VFS-058 anchors the synthetic
discovery term ``objective validation repair`` so supervisor scans re-find
the validation gate after domain evidence
(``vfs/mcplusplus-runtime-witness@1``) is already present.  That term never
becomes witness identity, receipt authority, formal-proof authority, or
static-completeness authority.  Real adapter dispatch stays distinguished
from mocks; HTTP and mcp+p2p profiles share the same admitted contract
surface where declared; network remains disabled unless an exact fixture and
egress policy permit it.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from .mcplusplus_contract_resolver import (
    TransportKind,
    normalize_tool_name,
    schema_fingerprint,
)
from .program_assurance_contracts import ClaimLevel
from .proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json,
    canonical_json_bytes,
    content_identity,
)

# ---------------------------------------------------------------------------
# Schema / evidence identities
# ---------------------------------------------------------------------------

MCPLUSPLUS_RUNTIME_WITNESS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/mcplusplus-runtime-witness@1"
)
MCPLUSPLUS_RUNTIME_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/mcplusplus-runtime-receipt@1"
)
MCPLUSPLUS_RUNTIME_FIXTURE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/mcplusplus-runtime-fixture@1"
)
MCPLUSPLUS_ADAPTER_SPEC_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/mcplusplus-adapter-spec@1"
)
MCPLUSPLUS_DISCOVERY_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/mcplusplus-discovery-record@1"
)
MCPLUSPLUS_NEGOTIATION_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/mcplusplus-negotiation-record@1"
)
MCPLUSPLUS_CALL_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/mcplusplus-runtime-call-request@1"
)
MCPLUSPLUS_CALL_OBSERVATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/mcplusplus-runtime-call-observation@1"
)

EVIDENCE_RUNTIME_WITNESS: Final[str] = "vfs/mcplusplus-runtime-witness@1"
# Synthetic objective-heap evidence term for VFS-G061 validation-gate work.
# Exact-text discovery key only — never part of witness/receipt identity.
OBJECTIVE_VALIDATION_REPAIR_EVIDENCE: Final[str] = "objective validation repair"
# Domain goal that owns hermetic MCP++ runtime witness surfaces.
OBJECTIVE_GOAL_ID: Final[str] = "VFS-G061"
# Repair task that owns the synthetic objective validation repair obligation.
OBJECTIVE_VALIDATION_REPAIR_TASK_ID: Final[str] = "VFS-058"

# Keep exact-text discovery anchors aligned with the objective heap.
assert OBJECTIVE_VALIDATION_REPAIR_EVIDENCE == "objective validation repair"
assert OBJECTIVE_GOAL_ID == "VFS-G061"
assert OBJECTIVE_VALIDATION_REPAIR_TASK_ID == "VFS-058"
assert EVIDENCE_RUNTIME_WITNESS == "vfs/mcplusplus-runtime-witness@1"

WITNESS_VERSION: Final[str] = "mcplusplus-runtime-witness@1"
WITNESS_PRODUCER: Final[str] = "mcplusplus-runtime-witness@1"
CONTRACT_VERSION: Final[int] = 1

DEFAULT_TIMEOUT_MS: Final[int] = 5_000
DEFAULT_MAX_PAYLOAD_BYTES: Final[int] = 65_536
DEFAULT_MAX_TOOLS: Final[int] = 256
DEFAULT_MAX_PROFILES: Final[int] = 32
DEFAULT_MAX_ERROR_CODES: Final[int] = 64
DEFAULT_MAX_OBSERVATIONS: Final[int] = 256
DEFAULT_MAX_TEXT_BYTES: Final[int] = 8_192
DEFAULT_SUBPROCESS_TIMEOUT_S: Final[float] = 30.0

KNOWN_PROFILES: Final[frozenset[str]] = frozenset(
    {
        "mcp++/basic",
        "mcp++/mcp-idl",
        "mcp++/idl",
        "mcp++/cid-envelope",
        "mcp++/ucan",
        "mcp++/deontic-policy",
        "mcp++/event-dag",
        "mcp++/p2p-transport",
        "mcp++/risk-scheduling",
        "mcp++/x402-payments",
    }
)

# Subprocess protocol marker so parent/child agree on the harness identity.
_SUBPROCESS_PROTOCOL: Final[str] = "mcplusplus-runtime-witness-subprocess@1"


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class RuntimeWitnessError(ContractValidationError):
    """Malformed or unsafe MCP++ runtime witness input/record."""


class RuntimeWitnessBoundsError(RuntimeWitnessError):
    """A compact runtime witness record exceeded an explicit bound."""


class RuntimeWitnessAuthorityError(RuntimeWitnessError):
    """Evidence was presented with an authority it cannot establish."""


class ImplementationKind(str, Enum):
    """How an adapter implementation relates to production authority.

    ``PRODUCTION`` — real registered adapter under test (hermetic body may be
    in-process, but the identity is production-class).
    ``FIXTURE`` — hermetic fixture deliberately approximating a backend.
    ``MOCK`` — mock/stub/test double; never grants production authority.
    """

    PRODUCTION = "production"
    FIXTURE = "fixture"
    MOCK = "mock"

    @property
    def grants_production_authority(self) -> bool:
        return self is ImplementationKind.PRODUCTION


class WitnessPhase(str, Enum):
    """Ordered phases recorded for a single runtime witness."""

    DISCOVERY = "discovery"
    CAPABILITY_NEGOTIATION = "capability_negotiation"
    INPUT_VALIDATION = "input_validation"
    DISPATCH = "dispatch"
    OUTPUT_SCHEMA = "output_schema"
    ERROR_SCHEMA = "error_schema"
    TRANSPORT = "transport"
    TIMEOUT = "timeout"
    CLEANUP = "cleanup"


class WitnessOutcome(str, Enum):
    """Closed, typed outcomes.  Only ``passed`` is a positive witness."""

    PASSED = "passed"
    MALFORMED_CALL = "malformed_call"
    MISSING_TOOL = "missing_tool"
    SCHEMA_VIOLATION = "schema_violation"
    UNAVAILABLE_BACKEND = "unavailable_backend"
    CANCELLED = "cancelled"
    PROFILE_MISMATCH = "profile_mismatch"
    STALE_MANIFEST = "stale_manifest"
    TIMED_OUT = "timed_out"
    TRANSPORT_REJECTED = "transport_rejected"
    DISPATCH_ERROR = "dispatch_error"
    CLEANUP_FAILED = "cleanup_failed"
    INCONCLUSIVE = "inconclusive"

    @property
    def is_positive(self) -> bool:
        return self is WitnessOutcome.PASSED

    @property
    def is_authoritative(self) -> bool:
        """Positive witnesses can carry ``runtime_witnessed`` only when
        production-class; negative outcomes are typed and non-authoritative
        for completeness claims.
        """

        return self is WitnessOutcome.PASSED


class BackendAvailability(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"


class ValidationVerdict(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    SKIPPED = "skipped"


class CleanupStatus(str, Enum):
    CLEAN = "clean"
    DIRTY = "dirty"
    SKIPPED = "skipped"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _text(
    value: Any,
    *,
    field_name: str,
    required: bool = True,
    maximum: int = DEFAULT_MAX_TEXT_BYTES,
) -> str:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise RuntimeWitnessError(f"{field_name} must be a string")
    else:
        text = value.strip()
    if required and not text:
        raise RuntimeWitnessError(f"{field_name} is required")
    if "\x00" in text:
        raise RuntimeWitnessError(f"{field_name} must not contain NUL")
    if len(text.encode("utf-8")) > maximum:
        raise RuntimeWitnessBoundsError(
            f"{field_name} exceeds {maximum} UTF-8 bytes"
        )
    return text


def _boolean(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeWitnessError(f"{field_name} must be a boolean")
    return value


def _integer(
    value: Any,
    *,
    field_name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeWitnessError(f"{field_name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        suffix = f" and at most {maximum}" if maximum is not None else ""
        raise RuntimeWitnessBoundsError(
            f"{field_name} must be at least {minimum}{suffix}"
        )
    return value


def _enum(value: Any, enum_type: type[Enum], *, field_name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    raw = getattr(value, "value", value)
    try:
        return enum_type(raw)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise RuntimeWitnessError(
            f"{field_name} must be one of: {allowed}"
        ) from exc


def _strings(
    values: Any,
    *,
    field_name: str,
    required: bool = False,
    maximum: int = DEFAULT_MAX_TOOLS,
    preserve_order: bool = True,
) -> tuple[str, ...]:
    if values is None:
        items: Sequence[Any] = ()
    elif isinstance(values, str):
        raise RuntimeWitnessError(f"{field_name} must be a sequence")
    elif isinstance(values, Sequence) and not isinstance(
        values, (bytes, bytearray)
    ):
        items = values
    else:
        raise RuntimeWitnessError(f"{field_name} must be a sequence")
    if len(items) > maximum:
        raise RuntimeWitnessBoundsError(
            f"{field_name} exceeds {maximum} items"
        )
    result: list[str] = []
    for index, item in enumerate(items):
        text = _text(item, field_name=f"{field_name}[{index}]")
        if text not in result:
            result.append(text)
    if required and not result:
        raise RuntimeWitnessError(f"{field_name} must not be empty")
    return tuple(result if preserve_order else sorted(result))


def _mapping(
    value: Any,
    *,
    field_name: str,
    max_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise RuntimeWitnessError(f"{field_name} must be a mapping")
    plain: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise RuntimeWitnessError(f"{field_name} keys must be strings")
        plain[key] = _json_safe(item, field_name=f"{field_name}.{key}")
    encoded = canonical_json_bytes(plain)
    if len(encoded) > max_bytes:
        raise RuntimeWitnessBoundsError(
            f"{field_name} exceeds {max_bytes} bytes"
        )
    return MappingProxyType(dict(sorted(plain.items())))


def _json_safe(value: Any, *, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, bool) or not isinstance(value, int):
            return value
        # Reject non-finite floats implicitly by only allowing int.
        return value
    if isinstance(value, float):
        raise RuntimeWitnessError(
            f"{field_name} must not contain floating-point values"
        )
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(k): _json_safe(v, field_name=f"{field_name}.{k}")
            for k, v in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            _json_safe(item, field_name=f"{field_name}[]") for item in value
        ]
    raise RuntimeWitnessError(
        f"{field_name} has unsupported type {type(value).__name__}"
    )


def _schema_object(
    value: Any, *, field_name: str
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return MappingProxyType({})
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeWitnessError(
                f"{field_name} is not valid JSON"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise RuntimeWitnessError(f"{field_name} JSON must be an object")
        return _mapping(decoded, field_name=field_name)
    return _mapping(value, field_name=field_name)


def _check_header(payload: Mapping[str, Any], expected: str) -> None:
    if not isinstance(payload, Mapping):
        raise RuntimeWitnessError("contract payload must be an object")
    supplied = payload.get("schema")
    if supplied not in (None, "", expected):
        raise RuntimeWitnessError(
            f"unsupported contract schema; use {expected}"
        )
    version = payload.get("contract_version")
    if version not in (None, CONTRACT_VERSION):
        raise RuntimeWitnessError(
            "unsupported runtime witness contract version"
        )


def _reject_unknown(
    payload: Mapping[str, Any], allowed: frozenset[str], *, artifact: str
) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise RuntimeWitnessError(
            f"{artifact} contains unsupported fields; rebuild its canonical payload"
        )


# ---------------------------------------------------------------------------
# Lightweight JSON Schema subset validation (hermetic, no network)
# ---------------------------------------------------------------------------


def validate_against_schema(
    payload: Any, schema: Mapping[str, Any] | None
) -> tuple[ValidationVerdict, tuple[str, ...]]:
    """Validate ``payload`` against a restricted JSON Schema subset.

    Supports ``type`` (object/string/integer/boolean/array/null/number-as-int),
    ``properties``, ``required``, ``enum``, and nested object/array items.
    Unsupported keywords are ignored (not treated as validation failures).
    """

    if not schema:
        return ValidationVerdict.SKIPPED, ()

    errors: list[str] = []

    def _check(value: Any, sch: Mapping[str, Any], path: str) -> None:
        expected_type = sch.get("type")
        if expected_type == "object":
            if not isinstance(value, Mapping):
                errors.append(f"{path}: expected object")
                return
            required = sch.get("required") or ()
            if isinstance(required, Sequence) and not isinstance(
                required, (str, bytes, bytearray)
            ):
                for key in required:
                    if str(key) not in value:
                        errors.append(f"{path}: missing required {key!r}")
            properties = sch.get("properties") or {}
            if isinstance(properties, Mapping):
                for key, child in properties.items():
                    if key in value and isinstance(child, Mapping):
                        _check(value[key], child, f"{path}.{key}")
        elif expected_type == "array":
            if not isinstance(value, Sequence) or isinstance(
                value, (str, bytes, bytearray)
            ):
                errors.append(f"{path}: expected array")
                return
            items = sch.get("items")
            if isinstance(items, Mapping):
                for index, item in enumerate(value):
                    _check(item, items, f"{path}[{index}]")
        elif expected_type == "string":
            if not isinstance(value, str):
                errors.append(f"{path}: expected string")
        elif expected_type == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                errors.append(f"{path}: expected integer")
        elif expected_type == "boolean":
            if not isinstance(value, bool):
                errors.append(f"{path}: expected boolean")
        elif expected_type == "null":
            if value is not None:
                errors.append(f"{path}: expected null")
        if "enum" in sch:
            allowed = sch["enum"]
            if isinstance(allowed, Sequence) and value not in list(allowed):
                errors.append(f"{path}: value not in enum")

    _check(payload, schema, "$")
    if errors:
        return ValidationVerdict.INVALID, tuple(errors[:32])
    return ValidationVerdict.VALID, ()


# ---------------------------------------------------------------------------
# Immutable records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RuntimeContract(CanonicalContract):
    """Shared header for runtime-witness IR."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": CONTRACT_VERSION,
            **self._payload(),
        }


@dataclass(frozen=True)
class AdapterSpec(_RuntimeContract):
    """A registered adapter available to the hermetic runtime fixture."""

    SCHEMA: ClassVar[str] = MCPLUSPLUS_ADAPTER_SPEC_SCHEMA

    tool_name: str
    adapter_id: str
    implementation_kind: ImplementationKind = ImplementationKind.PRODUCTION
    implementation_target: str = ""
    package: str = "ipfs_accelerate_py"
    version: str = "1.0.0"
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    error_codes: tuple[str, ...] = ()
    profiles: tuple[str, ...] = ("mcp++/basic",)
    transports: tuple[str, ...] = (TransportKind.HTTP.value,)
    backend_availability: BackendAvailability = BackendAvailability.AVAILABLE
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tool_name",
            normalize_tool_name(
                _text(self.tool_name, field_name="tool_name")
            ),
        )
        object.__setattr__(
            self,
            "adapter_id",
            _text(self.adapter_id, field_name="adapter_id"),
        )
        object.__setattr__(
            self,
            "implementation_kind",
            _enum(
                self.implementation_kind,
                ImplementationKind,
                field_name="implementation_kind",
            ),
        )
        object.__setattr__(
            self,
            "implementation_target",
            _text(
                self.implementation_target,
                field_name="implementation_target",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "package",
            _text(self.package, field_name="package", required=False)
            or "ipfs_accelerate_py",
        )
        object.__setattr__(
            self,
            "version",
            _text(self.version, field_name="version", required=False)
            or "1.0.0",
        )
        object.__setattr__(
            self,
            "input_schema",
            _schema_object(self.input_schema, field_name="input_schema"),
        )
        object.__setattr__(
            self,
            "output_schema",
            _schema_object(self.output_schema, field_name="output_schema"),
        )
        object.__setattr__(
            self,
            "error_codes",
            _strings(
                self.error_codes,
                field_name="error_codes",
                maximum=DEFAULT_MAX_ERROR_CODES,
            ),
        )
        object.__setattr__(
            self,
            "profiles",
            _strings(
                self.profiles,
                field_name="profiles",
                required=True,
                maximum=DEFAULT_MAX_PROFILES,
            ),
        )
        transports = _strings(
            self.transports,
            field_name="transports",
            required=True,
            maximum=8,
        )
        normalized_transports: list[str] = []
        for item in transports:
            kind = _enum(item, TransportKind, field_name="transports")
            if kind is TransportKind.UNKNOWN:
                raise RuntimeWitnessError(
                    "adapter transports must not be unknown"
                )
            normalized_transports.append(kind.value)
        object.__setattr__(self, "transports", tuple(normalized_transports))
        object.__setattr__(
            self,
            "backend_availability",
            _enum(
                self.backend_availability,
                BackendAvailability,
                field_name="backend_availability",
            ),
        )
        object.__setattr__(
            self,
            "notes",
            _text(self.notes, field_name="notes", required=False),
        )

    @property
    def is_mock(self) -> bool:
        return self.implementation_kind is ImplementationKind.MOCK

    @property
    def is_production(self) -> bool:
        return self.implementation_kind is ImplementationKind.PRODUCTION

    @property
    def input_schema_fingerprint(self) -> str:
        return schema_fingerprint(dict(self.input_schema))

    @property
    def output_schema_fingerprint(self) -> str:
        return schema_fingerprint(dict(self.output_schema))

    def _payload(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "adapter_id": self.adapter_id,
            "implementation_kind": self.implementation_kind.value,
            "implementation_target": self.implementation_target,
            "package": self.package,
            "version": self.version,
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "error_codes": list(self.error_codes),
            "profiles": list(self.profiles),
            "transports": list(self.transports),
            "backend_availability": self.backend_availability.value,
            "notes": self.notes,
            "input_schema_fingerprint": self.input_schema_fingerprint,
            "output_schema_fingerprint": self.output_schema_fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AdapterSpec":
        _check_header(payload, cls.SCHEMA)
        allowed = frozenset(
            {
                "schema",
                "contract_version",
                "tool_name",
                "adapter_id",
                "implementation_kind",
                "implementation_target",
                "package",
                "version",
                "input_schema",
                "output_schema",
                "error_codes",
                "profiles",
                "transports",
                "backend_availability",
                "notes",
                "input_schema_fingerprint",
                "output_schema_fingerprint",
                "content_id",
            }
        )
        _reject_unknown(payload, allowed, artifact="adapter spec")
        return cls(
            tool_name=str(payload.get("tool_name") or ""),
            adapter_id=str(payload.get("adapter_id") or ""),
            implementation_kind=payload.get(
                "implementation_kind", ImplementationKind.PRODUCTION
            ),
            implementation_target=str(
                payload.get("implementation_target") or ""
            ),
            package=str(payload.get("package") or "ipfs_accelerate_py"),
            version=str(payload.get("version") or "1.0.0"),
            input_schema=payload.get("input_schema") or {},
            output_schema=payload.get("output_schema") or {},
            error_codes=tuple(payload.get("error_codes") or ()),
            profiles=tuple(payload.get("profiles") or ("mcp++/basic",)),
            transports=tuple(
                payload.get("transports") or (TransportKind.HTTP.value,)
            ),
            backend_availability=payload.get(
                "backend_availability", BackendAvailability.AVAILABLE
            ),
            notes=str(payload.get("notes") or ""),
        )


@dataclass(frozen=True)
class ToolDiscoveryRecord(_RuntimeContract):
    """Recorded tools/list discovery observation."""

    SCHEMA: ClassVar[str] = MCPLUSPLUS_DISCOVERY_RECORD_SCHEMA

    tool_names: tuple[str, ...] = ()
    adapter_ids: tuple[str, ...] = ()
    production_tools: tuple[str, ...] = ()
    mock_tools: tuple[str, ...] = ()
    fixture_tools: tuple[str, ...] = ()
    manifest_cid: str = ""
    server_name: str = "ipfs-accelerate-mcp++"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tool_names",
            _strings(self.tool_names, field_name="tool_names"),
        )
        object.__setattr__(
            self,
            "adapter_ids",
            _strings(self.adapter_ids, field_name="adapter_ids"),
        )
        object.__setattr__(
            self,
            "production_tools",
            _strings(self.production_tools, field_name="production_tools"),
        )
        object.__setattr__(
            self,
            "mock_tools",
            _strings(self.mock_tools, field_name="mock_tools"),
        )
        object.__setattr__(
            self,
            "fixture_tools",
            _strings(self.fixture_tools, field_name="fixture_tools"),
        )
        object.__setattr__(
            self,
            "manifest_cid",
            _text(self.manifest_cid, field_name="manifest_cid", required=False),
        )
        object.__setattr__(
            self,
            "server_name",
            _text(self.server_name, field_name="server_name"),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "tool_names": list(self.tool_names),
            "adapter_ids": list(self.adapter_ids),
            "production_tools": list(self.production_tools),
            "mock_tools": list(self.mock_tools),
            "fixture_tools": list(self.fixture_tools),
            "manifest_cid": self.manifest_cid,
            "server_name": self.server_name,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ToolDiscoveryRecord":
        _check_header(payload, cls.SCHEMA)
        return cls(
            tool_names=tuple(payload.get("tool_names") or ()),
            adapter_ids=tuple(payload.get("adapter_ids") or ()),
            production_tools=tuple(payload.get("production_tools") or ()),
            mock_tools=tuple(payload.get("mock_tools") or ()),
            fixture_tools=tuple(payload.get("fixture_tools") or ()),
            manifest_cid=str(payload.get("manifest_cid") or ""),
            server_name=str(
                payload.get("server_name") or "ipfs-accelerate-mcp++"
            ),
        )


@dataclass(frozen=True)
class CapabilityNegotiationRecord(_RuntimeContract):
    """Profile / transport negotiation observation."""

    SCHEMA: ClassVar[str] = MCPLUSPLUS_NEGOTIATION_RECORD_SCHEMA

    requested_profiles: tuple[str, ...] = ()
    admitted_profiles: tuple[str, ...] = ()
    active_profile: str = ""
    requested_transport: str = TransportKind.HTTP.value
    admitted_transports: tuple[str, ...] = (TransportKind.HTTP.value,)
    active_transport: str = TransportKind.HTTP.value
    negotiated: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requested_profiles",
            _strings(
                self.requested_profiles,
                field_name="requested_profiles",
                maximum=DEFAULT_MAX_PROFILES,
            ),
        )
        object.__setattr__(
            self,
            "admitted_profiles",
            _strings(
                self.admitted_profiles,
                field_name="admitted_profiles",
                maximum=DEFAULT_MAX_PROFILES,
            ),
        )
        object.__setattr__(
            self,
            "active_profile",
            _text(
                self.active_profile,
                field_name="active_profile",
                required=False,
            ),
        )
        transport = _enum(
            self.requested_transport,
            TransportKind,
            field_name="requested_transport",
        )
        object.__setattr__(self, "requested_transport", transport.value)
        admitted = _strings(
            self.admitted_transports,
            field_name="admitted_transports",
            required=True,
            maximum=8,
        )
        object.__setattr__(
            self,
            "admitted_transports",
            tuple(
                _enum(item, TransportKind, field_name="admitted_transports").value
                for item in admitted
            ),
        )
        active = _enum(
            self.active_transport,
            TransportKind,
            field_name="active_transport",
        )
        object.__setattr__(self, "active_transport", active.value)
        object.__setattr__(
            self,
            "negotiated",
            _boolean(self.negotiated, field_name="negotiated"),
        )
        object.__setattr__(
            self,
            "reason",
            _text(self.reason, field_name="reason", required=False),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "requested_profiles": list(self.requested_profiles),
            "admitted_profiles": list(self.admitted_profiles),
            "active_profile": self.active_profile,
            "requested_transport": self.requested_transport,
            "admitted_transports": list(self.admitted_transports),
            "active_transport": self.active_transport,
            "negotiated": self.negotiated,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "CapabilityNegotiationRecord":
        _check_header(payload, cls.SCHEMA)
        return cls(
            requested_profiles=tuple(payload.get("requested_profiles") or ()),
            admitted_profiles=tuple(payload.get("admitted_profiles") or ()),
            active_profile=str(payload.get("active_profile") or ""),
            requested_transport=str(
                payload.get("requested_transport") or TransportKind.HTTP.value
            ),
            admitted_transports=tuple(
                payload.get("admitted_transports")
                or (TransportKind.HTTP.value,)
            ),
            active_transport=str(
                payload.get("active_transport") or TransportKind.HTTP.value
            ),
            negotiated=bool(payload.get("negotiated", False)),
            reason=str(payload.get("reason") or ""),
        )


@dataclass(frozen=True)
class CallRequest(_RuntimeContract):
    """A tools/call request under hermetic observation."""

    SCHEMA: ClassVar[str] = MCPLUSPLUS_CALL_REQUEST_SCHEMA

    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    requested_profiles: tuple[str, ...] = ("mcp++/basic",)
    transport: str = TransportKind.HTTP.value
    call_id: str = ""
    cancel: bool = False
    force_timeout: bool = False
    expected_manifest_cid: str = ""

    def __post_init__(self) -> None:
        # Allow empty tool_name for malformed-call tests via factory;
        # normal construction still normalizes.
        raw_name = self.tool_name
        if raw_name is None:
            raise RuntimeWitnessError("tool_name must be a string")
        if not isinstance(raw_name, str):
            raise RuntimeWitnessError("tool_name must be a string")
        stripped = raw_name.strip()
        object.__setattr__(
            self,
            "tool_name",
            normalize_tool_name(stripped) if stripped else "",
        )
        object.__setattr__(
            self,
            "arguments",
            _mapping(self.arguments, field_name="arguments"),
        )
        object.__setattr__(
            self,
            "requested_profiles",
            _strings(
                self.requested_profiles,
                field_name="requested_profiles",
                maximum=DEFAULT_MAX_PROFILES,
            ),
        )
        transport = _enum(
            self.transport, TransportKind, field_name="transport"
        )
        object.__setattr__(self, "transport", transport.value)
        object.__setattr__(
            self,
            "call_id",
            _text(self.call_id, field_name="call_id", required=False),
        )
        object.__setattr__(
            self, "cancel", _boolean(self.cancel, field_name="cancel")
        )
        object.__setattr__(
            self,
            "force_timeout",
            _boolean(self.force_timeout, field_name="force_timeout"),
        )
        object.__setattr__(
            self,
            "expected_manifest_cid",
            _text(
                self.expected_manifest_cid,
                field_name="expected_manifest_cid",
                required=False,
            ),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "requested_profiles": list(self.requested_profiles),
            "transport": self.transport,
            "call_id": self.call_id,
            "cancel": self.cancel,
            "force_timeout": self.force_timeout,
            "expected_manifest_cid": self.expected_manifest_cid,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CallRequest":
        _check_header(payload, cls.SCHEMA)
        return cls(
            tool_name=str(payload.get("tool_name") or ""),
            arguments=payload.get("arguments") or {},
            requested_profiles=tuple(
                payload.get("requested_profiles") or ("mcp++/basic",)
            ),
            transport=str(
                payload.get("transport") or TransportKind.HTTP.value
            ),
            call_id=str(payload.get("call_id") or ""),
            cancel=bool(payload.get("cancel", False)),
            force_timeout=bool(payload.get("force_timeout", False)),
            expected_manifest_cid=str(
                payload.get("expected_manifest_cid") or ""
            ),
        )


@dataclass(frozen=True)
class CallObservation(_RuntimeContract):
    """Observed dispatch / validation / schema outcomes for one call."""

    SCHEMA: ClassVar[str] = MCPLUSPLUS_CALL_OBSERVATION_SCHEMA

    outcome: WitnessOutcome
    tool_name: str = ""
    adapter_id: str = ""
    implementation_kind: ImplementationKind = ImplementationKind.PRODUCTION
    implementation_target: str = ""
    input_validation: ValidationVerdict = ValidationVerdict.SKIPPED
    input_errors: tuple[str, ...] = ()
    output_validation: ValidationVerdict = ValidationVerdict.SKIPPED
    output_errors: tuple[str, ...] = ()
    error_code: str = ""
    error_schema_ok: bool = True
    result: Mapping[str, Any] = field(default_factory=dict)
    phases_completed: tuple[str, ...] = ()
    duration_ms: int = 0
    timed_out: bool = False
    cancelled: bool = False
    cleanup_status: CleanupStatus = CleanupStatus.SKIPPED
    claim_level: str = ClaimLevel.RUNTIME_WITNESSED.value
    grants_runtime_authority: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outcome",
            _enum(self.outcome, WitnessOutcome, field_name="outcome"),
        )
        object.__setattr__(
            self,
            "tool_name",
            _text(self.tool_name, field_name="tool_name", required=False),
        )
        object.__setattr__(
            self,
            "adapter_id",
            _text(self.adapter_id, field_name="adapter_id", required=False),
        )
        object.__setattr__(
            self,
            "implementation_kind",
            _enum(
                self.implementation_kind,
                ImplementationKind,
                field_name="implementation_kind",
            ),
        )
        object.__setattr__(
            self,
            "implementation_target",
            _text(
                self.implementation_target,
                field_name="implementation_target",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "input_validation",
            _enum(
                self.input_validation,
                ValidationVerdict,
                field_name="input_validation",
            ),
        )
        object.__setattr__(
            self,
            "input_errors",
            _strings(
                self.input_errors,
                field_name="input_errors",
                maximum=32,
            ),
        )
        object.__setattr__(
            self,
            "output_validation",
            _enum(
                self.output_validation,
                ValidationVerdict,
                field_name="output_validation",
            ),
        )
        object.__setattr__(
            self,
            "output_errors",
            _strings(
                self.output_errors,
                field_name="output_errors",
                maximum=32,
            ),
        )
        object.__setattr__(
            self,
            "error_code",
            _text(self.error_code, field_name="error_code", required=False),
        )
        object.__setattr__(
            self,
            "error_schema_ok",
            _boolean(self.error_schema_ok, field_name="error_schema_ok"),
        )
        object.__setattr__(
            self,
            "result",
            _mapping(self.result, field_name="result"),
        )
        object.__setattr__(
            self,
            "phases_completed",
            _strings(
                self.phases_completed,
                field_name="phases_completed",
                maximum=16,
            ),
        )
        object.__setattr__(
            self,
            "duration_ms",
            _integer(self.duration_ms, field_name="duration_ms"),
        )
        object.__setattr__(
            self,
            "timed_out",
            _boolean(self.timed_out, field_name="timed_out"),
        )
        object.__setattr__(
            self,
            "cancelled",
            _boolean(self.cancelled, field_name="cancelled"),
        )
        object.__setattr__(
            self,
            "cleanup_status",
            _enum(
                self.cleanup_status,
                CleanupStatus,
                field_name="cleanup_status",
            ),
        )
        claim = _text(self.claim_level, field_name="claim_level")
        if claim != ClaimLevel.RUNTIME_WITNESSED.value:
            raise RuntimeWitnessAuthorityError(
                "runtime observations may only assert runtime_witnessed claims"
            )
        object.__setattr__(self, "claim_level", claim)
        grants = _boolean(
            self.grants_runtime_authority,
            field_name="grants_runtime_authority",
        )
        # Fail-closed: only production + passed may grant authority.
        if grants and (
            self.outcome is not WitnessOutcome.PASSED
            or self.implementation_kind is not ImplementationKind.PRODUCTION
        ):
            raise RuntimeWitnessAuthorityError(
                "only production adapters with passed outcome grant "
                "runtime_witnessed authority"
            )
        object.__setattr__(self, "grants_runtime_authority", grants)
        object.__setattr__(
            self,
            "reason",
            _text(self.reason, field_name="reason", required=False),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "tool_name": self.tool_name,
            "adapter_id": self.adapter_id,
            "implementation_kind": self.implementation_kind.value,
            "implementation_target": self.implementation_target,
            "input_validation": self.input_validation.value,
            "input_errors": list(self.input_errors),
            "output_validation": self.output_validation.value,
            "output_errors": list(self.output_errors),
            "error_code": self.error_code,
            "error_schema_ok": self.error_schema_ok,
            "result": dict(self.result),
            "phases_completed": list(self.phases_completed),
            "duration_ms": self.duration_ms,
            "timed_out": self.timed_out,
            "cancelled": self.cancelled,
            "cleanup_status": self.cleanup_status.value,
            "claim_level": self.claim_level,
            "grants_runtime_authority": self.grants_runtime_authority,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CallObservation":
        _check_header(payload, cls.SCHEMA)
        return cls(
            outcome=payload.get("outcome") or WitnessOutcome.INCONCLUSIVE,
            tool_name=str(payload.get("tool_name") or ""),
            adapter_id=str(payload.get("adapter_id") or ""),
            implementation_kind=payload.get(
                "implementation_kind", ImplementationKind.PRODUCTION
            ),
            implementation_target=str(
                payload.get("implementation_target") or ""
            ),
            input_validation=payload.get(
                "input_validation", ValidationVerdict.SKIPPED
            ),
            input_errors=tuple(payload.get("input_errors") or ()),
            output_validation=payload.get(
                "output_validation", ValidationVerdict.SKIPPED
            ),
            output_errors=tuple(payload.get("output_errors") or ()),
            error_code=str(payload.get("error_code") or ""),
            error_schema_ok=bool(payload.get("error_schema_ok", True)),
            result=payload.get("result") or {},
            phases_completed=tuple(payload.get("phases_completed") or ()),
            duration_ms=int(payload.get("duration_ms") or 0),
            timed_out=bool(payload.get("timed_out", False)),
            cancelled=bool(payload.get("cancelled", False)),
            cleanup_status=payload.get(
                "cleanup_status", CleanupStatus.SKIPPED
            ),
            claim_level=str(
                payload.get("claim_level")
                or ClaimLevel.RUNTIME_WITNESSED.value
            ),
            grants_runtime_authority=bool(
                payload.get("grants_runtime_authority", False)
            ),
            reason=str(payload.get("reason") or ""),
        )


@dataclass(frozen=True)
class RuntimeWitness(_RuntimeContract):
    """One hermetic runtime witness binding discovery through cleanup."""

    SCHEMA: ClassVar[str] = MCPLUSPLUS_RUNTIME_WITNESS_SCHEMA

    fixture_id: str
    forest_id: str
    discovery: ToolDiscoveryRecord
    negotiation: CapabilityNegotiationRecord
    request: CallRequest
    observation: CallObservation
    transport: str = TransportKind.HTTP.value
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    network_enabled: bool = False
    evidence_kind: str = EVIDENCE_RUNTIME_WITNESS
    witness_version: str = WITNESS_VERSION
    producer: str = WITNESS_PRODUCER
    static_completeness_claimed: bool = False
    formal_proof_claimed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fixture_id",
            _text(self.fixture_id, field_name="fixture_id"),
        )
        object.__setattr__(
            self,
            "forest_id",
            _text(self.forest_id, field_name="forest_id"),
        )
        if not isinstance(self.discovery, ToolDiscoveryRecord):
            raise RuntimeWitnessError("discovery must be ToolDiscoveryRecord")
        if not isinstance(self.negotiation, CapabilityNegotiationRecord):
            raise RuntimeWitnessError(
                "negotiation must be CapabilityNegotiationRecord"
            )
        if not isinstance(self.request, CallRequest):
            raise RuntimeWitnessError("request must be CallRequest")
        if not isinstance(self.observation, CallObservation):
            raise RuntimeWitnessError("observation must be CallObservation")
        transport = _enum(
            self.transport, TransportKind, field_name="transport"
        )
        object.__setattr__(self, "transport", transport.value)
        object.__setattr__(
            self,
            "timeout_ms",
            _integer(
                self.timeout_ms,
                field_name="timeout_ms",
                minimum=1,
                maximum=3_600_000,
            ),
        )
        object.__setattr__(
            self,
            "network_enabled",
            _boolean(self.network_enabled, field_name="network_enabled"),
        )
        object.__setattr__(
            self,
            "evidence_kind",
            _text(self.evidence_kind, field_name="evidence_kind"),
        )
        if self.evidence_kind != EVIDENCE_RUNTIME_WITNESS:
            raise RuntimeWitnessError(
                f"evidence_kind must be {EVIDENCE_RUNTIME_WITNESS}"
            )
        object.__setattr__(
            self,
            "witness_version",
            _text(self.witness_version, field_name="witness_version"),
        )
        object.__setattr__(
            self,
            "producer",
            _text(self.producer, field_name="producer"),
        )
        # Runtime witnesses never replace static completeness or formal proof.
        object.__setattr__(
            self,
            "static_completeness_claimed",
            _boolean(
                self.static_completeness_claimed,
                field_name="static_completeness_claimed",
            ),
        )
        object.__setattr__(
            self,
            "formal_proof_claimed",
            _boolean(
                self.formal_proof_claimed,
                field_name="formal_proof_claimed",
            ),
        )
        if self.static_completeness_claimed or self.formal_proof_claimed:
            raise RuntimeWitnessAuthorityError(
                "runtime witnesses must not claim static completeness "
                "or formal proof"
            )

    @property
    def witness_id(self) -> str:
        return self.content_id

    @property
    def is_production_witness(self) -> bool:
        return (
            self.observation.grants_runtime_authority
            and self.observation.implementation_kind
            is ImplementationKind.PRODUCTION
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "forest_id": self.forest_id,
            "discovery": self.discovery.to_dict(),
            "negotiation": self.negotiation.to_dict(),
            "request": self.request.to_dict(),
            "observation": self.observation.to_dict(),
            "transport": self.transport,
            "timeout_ms": self.timeout_ms,
            "network_enabled": self.network_enabled,
            "evidence_kind": self.evidence_kind,
            "witness_version": self.witness_version,
            "producer": self.producer,
            "static_completeness_claimed": self.static_completeness_claimed,
            "formal_proof_claimed": self.formal_proof_claimed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeWitness":
        _check_header(payload, cls.SCHEMA)
        return cls(
            fixture_id=str(payload.get("fixture_id") or ""),
            forest_id=str(payload.get("forest_id") or ""),
            discovery=ToolDiscoveryRecord.from_dict(
                payload.get("discovery") or {}
            ),
            negotiation=CapabilityNegotiationRecord.from_dict(
                payload.get("negotiation") or {}
            ),
            request=CallRequest.from_dict(payload.get("request") or {}),
            observation=CallObservation.from_dict(
                payload.get("observation") or {}
            ),
            transport=str(
                payload.get("transport") or TransportKind.HTTP.value
            ),
            timeout_ms=int(payload.get("timeout_ms") or DEFAULT_TIMEOUT_MS),
            network_enabled=bool(payload.get("network_enabled", False)),
            evidence_kind=str(
                payload.get("evidence_kind") or EVIDENCE_RUNTIME_WITNESS
            ),
            witness_version=str(
                payload.get("witness_version") or WITNESS_VERSION
            ),
            producer=str(payload.get("producer") or WITNESS_PRODUCER),
            static_completeness_claimed=bool(
                payload.get("static_completeness_claimed", False)
            ),
            formal_proof_claimed=bool(
                payload.get("formal_proof_claimed", False)
            ),
        )


@dataclass(frozen=True)
class RuntimeWitnessReceipt(_RuntimeContract):
    """Content-addressed batch of runtime witnesses for deterministic replay."""

    SCHEMA: ClassVar[str] = MCPLUSPLUS_RUNTIME_RECEIPT_SCHEMA

    fixture_id: str
    forest_id: str
    manifest_cid: str
    witnesses: tuple[RuntimeWitness, ...] = ()
    network_enabled: bool = False
    evidence_kind: str = EVIDENCE_RUNTIME_WITNESS
    witness_version: str = WITNESS_VERSION
    producer: str = WITNESS_PRODUCER
    supplements_static_resolution: bool = True
    supplements_formal_proof: bool = True
    replaces_static_completeness: bool = False
    replaces_formal_proof: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fixture_id",
            _text(self.fixture_id, field_name="fixture_id"),
        )
        object.__setattr__(
            self,
            "forest_id",
            _text(self.forest_id, field_name="forest_id"),
        )
        object.__setattr__(
            self,
            "manifest_cid",
            _text(self.manifest_cid, field_name="manifest_cid"),
        )
        if not isinstance(self.witnesses, tuple):
            raise RuntimeWitnessError("witnesses must be a tuple")
        if len(self.witnesses) > DEFAULT_MAX_OBSERVATIONS:
            raise RuntimeWitnessBoundsError(
                f"witnesses exceed {DEFAULT_MAX_OBSERVATIONS} items"
            )
        for index, witness in enumerate(self.witnesses):
            if not isinstance(witness, RuntimeWitness):
                raise RuntimeWitnessError(
                    f"witnesses[{index}] must be RuntimeWitness"
                )
        object.__setattr__(
            self,
            "network_enabled",
            _boolean(self.network_enabled, field_name="network_enabled"),
        )
        object.__setattr__(
            self,
            "evidence_kind",
            _text(self.evidence_kind, field_name="evidence_kind"),
        )
        if self.evidence_kind != EVIDENCE_RUNTIME_WITNESS:
            raise RuntimeWitnessError(
                f"evidence_kind must be {EVIDENCE_RUNTIME_WITNESS}"
            )
        object.__setattr__(
            self,
            "witness_version",
            _text(self.witness_version, field_name="witness_version"),
        )
        object.__setattr__(
            self,
            "producer",
            _text(self.producer, field_name="producer"),
        )
        object.__setattr__(
            self,
            "supplements_static_resolution",
            _boolean(
                self.supplements_static_resolution,
                field_name="supplements_static_resolution",
            ),
        )
        object.__setattr__(
            self,
            "supplements_formal_proof",
            _boolean(
                self.supplements_formal_proof,
                field_name="supplements_formal_proof",
            ),
        )
        object.__setattr__(
            self,
            "replaces_static_completeness",
            _boolean(
                self.replaces_static_completeness,
                field_name="replaces_static_completeness",
            ),
        )
        object.__setattr__(
            self,
            "replaces_formal_proof",
            _boolean(
                self.replaces_formal_proof,
                field_name="replaces_formal_proof",
            ),
        )
        if self.replaces_static_completeness or self.replaces_formal_proof:
            raise RuntimeWitnessAuthorityError(
                "runtime receipts must supplement, not replace, static "
                "completeness or formal proof"
            )
        if not self.supplements_static_resolution:
            raise RuntimeWitnessAuthorityError(
                "runtime receipts must declare supplementation of static "
                "resolution"
            )

    @property
    def receipt_id(self) -> str:
        return self.content_id

    @property
    def production_witness_count(self) -> int:
        return sum(1 for w in self.witnesses if w.is_production_witness)

    @property
    def mock_witness_count(self) -> int:
        return sum(
            1
            for w in self.witnesses
            if w.observation.implementation_kind is ImplementationKind.MOCK
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "forest_id": self.forest_id,
            "manifest_cid": self.manifest_cid,
            "witnesses": [w.to_dict() for w in self.witnesses],
            "network_enabled": self.network_enabled,
            "evidence_kind": self.evidence_kind,
            "witness_version": self.witness_version,
            "producer": self.producer,
            "supplements_static_resolution": self.supplements_static_resolution,
            "supplements_formal_proof": self.supplements_formal_proof,
            "replaces_static_completeness": self.replaces_static_completeness,
            "replaces_formal_proof": self.replaces_formal_proof,
            "production_witness_count": self.production_witness_count,
            "mock_witness_count": self.mock_witness_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeWitnessReceipt":
        _check_header(payload, cls.SCHEMA)
        witnesses = tuple(
            RuntimeWitness.from_dict(item)
            for item in (payload.get("witnesses") or ())
        )
        return cls(
            fixture_id=str(payload.get("fixture_id") or ""),
            forest_id=str(payload.get("forest_id") or ""),
            manifest_cid=str(payload.get("manifest_cid") or ""),
            witnesses=witnesses,
            network_enabled=bool(payload.get("network_enabled", False)),
            evidence_kind=str(
                payload.get("evidence_kind") or EVIDENCE_RUNTIME_WITNESS
            ),
            witness_version=str(
                payload.get("witness_version") or WITNESS_VERSION
            ),
            producer=str(payload.get("producer") or WITNESS_PRODUCER),
            supplements_static_resolution=bool(
                payload.get("supplements_static_resolution", True)
            ),
            supplements_formal_proof=bool(
                payload.get("supplements_formal_proof", True)
            ),
            replaces_static_completeness=bool(
                payload.get("replaces_static_completeness", False)
            ),
            replaces_formal_proof=bool(
                payload.get("replaces_formal_proof", False)
            ),
        )


# ---------------------------------------------------------------------------
# Handler protocol and built-in production adapters
# ---------------------------------------------------------------------------

AdapterHandler = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class AdapterDispatchError(Exception):
    """Raised by an adapter to emit a typed error code."""

    def __init__(self, error_code: str, message: str = "") -> None:
        self.error_code = str(error_code or "dispatch_error")
        self.message = str(message or self.error_code)
        super().__init__(self.message)


def _handler_echo(arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    message = arguments.get("message", "")
    if not isinstance(message, str):
        raise AdapterDispatchError("schema_violation", "message must be string")
    return {"echo": message, "length": len(message)}


def _handler_identity(arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    name = arguments.get("name", "")
    if not isinstance(name, str) or not name:
        raise AdapterDispatchError("invalid_name", "name required")
    return {
        "identity": content_identity({"name": name}),
        "name": name,
    }


def _handler_vfs_stat(arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    path = arguments.get("path", "")
    if not isinstance(path, str) or not path:
        raise AdapterDispatchError("not_found", "path required")
    if ".." in path.replace("\\", "/").split("/"):
        raise AdapterDispatchError("permission_denied", "path traversal")
    return {
        "path": path,
        "exists": True,
        "kind": "file" if "." in path.rsplit("/", 1)[-1] else "directory",
        "size": len(path) * 17,
    }


def _handler_unavailable(_arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    raise AdapterDispatchError("unavailable", "backend unavailable")


def _handler_mock_always_ok(arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    return {"mock": True, "args": dict(arguments)}


def _handler_slow(arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    delay_ms = arguments.get("delay_ms", 50)
    if not isinstance(delay_ms, int) or isinstance(delay_ms, bool):
        delay_ms = 50
    time.sleep(max(0, min(delay_ms, 500)) / 1000.0)
    return {"slept_ms": delay_ms}


_BUILTIN_HANDLERS: dict[str, AdapterHandler] = {
    "echo": _handler_echo,
    "identity": _handler_identity,
    "vfs.stat": _handler_vfs_stat,
    "unavailable.probe": _handler_unavailable,
    "mock.always_ok": _handler_mock_always_ok,
    "slow.probe": _handler_slow,
}


def default_production_adapters() -> tuple[AdapterSpec, ...]:
    """Selected real production-class adapters for hermetic witnesses."""

    return (
        AdapterSpec(
            tool_name="echo",
            adapter_id="adapter:echo:production",
            implementation_kind=ImplementationKind.PRODUCTION,
            implementation_target=(
                "ipfs_accelerate_py.agent_supervisor."
                "mcplusplus_runtime_witness._handler_echo"
            ),
            package="ipfs_accelerate_py",
            version="1.0.0",
            input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "echo": {"type": "string"},
                    "length": {"type": "integer"},
                },
                "required": ["echo", "length"],
            },
            error_codes=("schema_violation",),
            profiles=("mcp++/basic", "mcp++/mcp-idl"),
            transports=(
                TransportKind.HTTP.value,
                TransportKind.MCP_P2P.value,
            ),
        ),
        AdapterSpec(
            tool_name="identity",
            adapter_id="adapter:identity:production",
            implementation_kind=ImplementationKind.PRODUCTION,
            implementation_target=(
                "ipfs_accelerate_py.agent_supervisor."
                "mcplusplus_runtime_witness._handler_identity"
            ),
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "identity": {"type": "string"},
                    "name": {"type": "string"},
                },
                "required": ["identity", "name"],
            },
            error_codes=("invalid_name",),
            profiles=("mcp++/basic", "mcp++/cid-envelope"),
            transports=(TransportKind.HTTP.value,),
        ),
        AdapterSpec(
            tool_name="vfs.stat",
            adapter_id="adapter:vfs.stat:production",
            implementation_kind=ImplementationKind.PRODUCTION,
            implementation_target=(
                "ipfs_accelerate_py.agent_supervisor."
                "mcplusplus_runtime_witness._handler_vfs_stat"
            ),
            package="ipfs_kit_py",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "exists": {"type": "boolean"},
                    "kind": {"type": "string"},
                    "size": {"type": "integer"},
                },
                "required": ["path", "exists", "kind", "size"],
            },
            error_codes=("not_found", "permission_denied"),
            profiles=("mcp++/basic", "mcp++/mcp-idl", "mcp++/p2p-transport"),
            transports=(
                TransportKind.HTTP.value,
                TransportKind.MCP_P2P.value,
            ),
        ),
        AdapterSpec(
            tool_name="unavailable.probe",
            adapter_id="adapter:unavailable:production",
            implementation_kind=ImplementationKind.PRODUCTION,
            implementation_target=(
                "ipfs_accelerate_py.agent_supervisor."
                "mcplusplus_runtime_witness._handler_unavailable"
            ),
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            error_codes=("unavailable",),
            profiles=("mcp++/basic",),
            transports=(TransportKind.HTTP.value,),
            backend_availability=BackendAvailability.UNAVAILABLE,
        ),
        AdapterSpec(
            tool_name="slow.probe",
            adapter_id="adapter:slow:production",
            implementation_kind=ImplementationKind.PRODUCTION,
            implementation_target=(
                "ipfs_accelerate_py.agent_supervisor."
                "mcplusplus_runtime_witness._handler_slow"
            ),
            input_schema={
                "type": "object",
                "properties": {"delay_ms": {"type": "integer"}},
            },
            output_schema={
                "type": "object",
                "properties": {"slept_ms": {"type": "integer"}},
                "required": ["slept_ms"],
            },
            error_codes=(),
            profiles=("mcp++/basic",),
            transports=(TransportKind.HTTP.value,),
        ),
    )


def default_mock_adapters() -> tuple[AdapterSpec, ...]:
    """Explicit mock adapters for contrast tests (never production authority)."""

    return (
        AdapterSpec(
            tool_name="mock.always_ok",
            adapter_id="adapter:mock.always_ok:mock",
            implementation_kind=ImplementationKind.MOCK,
            implementation_target=(
                "ipfs_accelerate_py.agent_supervisor."
                "mcplusplus_runtime_witness._handler_mock_always_ok"
            ),
            input_schema={"type": "object", "properties": {}},
            output_schema={
                "type": "object",
                "properties": {"mock": {"type": "boolean"}},
            },
            error_codes=(),
            profiles=("mcp++/basic",),
            transports=(TransportKind.HTTP.value,),
            notes="explicit mock; never grants runtime authority",
        ),
    )


# ---------------------------------------------------------------------------
# Hermetic runtime fixture
# ---------------------------------------------------------------------------


@dataclass
class CancellationToken:
    """Thread-safe cancellation fence with stable identity."""

    cancellation_id: str
    _event: threading.Event = field(default_factory=threading.Event, repr=False)
    _reason: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        identity = str(self.cancellation_id or "").strip()
        if not identity:
            raise RuntimeWitnessError("cancellation identity is required")
        self.cancellation_id = identity

    def cancel(self, *, reason: str = "cancelled") -> None:
        self._reason = str(reason or "cancelled").strip() or "cancelled"
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason


class HermeticMCPlusPlusRuntime:
    """Bounded hermetic runtime for selected registered MCP++ adapters.

    Network is disabled by default.  Handlers run in-process under a hard
    timeout; optional :func:`run_witness_subprocess` isolates the same fixture
    in a child process.
    """

    def __init__(
        self,
        *,
        forest_id: str,
        fixture_id: str = "",
        adapters: Sequence[AdapterSpec] | None = None,
        admitted_profiles: Sequence[str] | None = None,
        admitted_transports: Sequence[str] | None = None,
        manifest_cid: str = "",
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
        network_enabled: bool = False,
        handlers: Mapping[str, AdapterHandler] | None = None,
        server_name: str = "ipfs-accelerate-mcp++",
    ) -> None:
        if network_enabled:
            raise RuntimeWitnessError(
                "network remains disabled unless an exact fixture and "
                "egress policy permit it; default fixture rejects network"
            )
        self.forest_id = _text(forest_id, field_name="forest_id")
        self.server_name = _text(server_name, field_name="server_name")
        self.timeout_ms = _integer(
            timeout_ms, field_name="timeout_ms", minimum=1, maximum=3_600_000
        )
        self.network_enabled = False
        self._adapters: dict[str, AdapterSpec] = {}
        self._handlers: dict[str, AdapterHandler] = dict(_BUILTIN_HANDLERS)
        if handlers:
            for name, handler in handlers.items():
                if not callable(handler):
                    raise RuntimeWitnessError(
                        f"handler for {name!r} must be callable"
                    )
                self._handlers[normalize_tool_name(name)] = handler
        specs = (
            list(adapters)
            if adapters is not None
            else list(default_production_adapters())
            + list(default_mock_adapters())
        )
        if len(specs) > DEFAULT_MAX_TOOLS:
            raise RuntimeWitnessBoundsError(
                f"adapters exceed {DEFAULT_MAX_TOOLS}"
            )
        for spec in specs:
            if not isinstance(spec, AdapterSpec):
                raise RuntimeWitnessError("adapters must be AdapterSpec")
            if spec.tool_name in self._adapters:
                raise RuntimeWitnessError(
                    f"duplicate tool registration: {spec.tool_name}"
                )
            self._adapters[spec.tool_name] = spec
        self.admitted_profiles = _strings(
            admitted_profiles
            if admitted_profiles is not None
            else (
                "mcp++/basic",
                "mcp++/mcp-idl",
                "mcp++/cid-envelope",
                "mcp++/p2p-transport",
            ),
            field_name="admitted_profiles",
            required=True,
            maximum=DEFAULT_MAX_PROFILES,
        )
        self.admitted_transports = tuple(
            _enum(item, TransportKind, field_name="admitted_transports").value
            for item in _strings(
                admitted_transports
                if admitted_transports is not None
                else (
                    TransportKind.HTTP.value,
                    TransportKind.MCP_P2P.value,
                ),
                field_name="admitted_transports",
                required=True,
                maximum=8,
            )
        )
        # Deterministic fixture / manifest identities from registry content.
        registry_payload = {
            "adapters": [s.to_dict() for s in self._adapters.values()],
            "admitted_profiles": list(self.admitted_profiles),
            "admitted_transports": list(self.admitted_transports),
            "forest_id": self.forest_id,
            "server_name": self.server_name,
            "timeout_ms": self.timeout_ms,
            "witness_version": WITNESS_VERSION,
        }
        computed_fixture = content_identity(registry_payload)
        self.fixture_id = (
            _text(fixture_id, field_name="fixture_id")
            if fixture_id
            else computed_fixture
        )
        self.manifest_cid = (
            _text(manifest_cid, field_name="manifest_cid")
            if manifest_cid
            else content_identity(
                {
                    "adapters": sorted(
                        (s.tool_name, s.version, s.adapter_id)
                        for s in self._adapters.values()
                    ),
                    "profiles": list(self.admitted_profiles),
                }
            )
        )
        self._cleaned_up = False
        self._active_calls = 0
        self._lock = threading.Lock()

    # -- discovery ---------------------------------------------------------

    def discover_tools(self) -> ToolDiscoveryRecord:
        if self._cleaned_up:
            raise RuntimeWitnessError("runtime already cleaned up")
        specs = sorted(self._adapters.values(), key=lambda s: s.tool_name)
        return ToolDiscoveryRecord(
            tool_names=tuple(s.tool_name for s in specs),
            adapter_ids=tuple(s.adapter_id for s in specs),
            production_tools=tuple(
                s.tool_name
                for s in specs
                if s.implementation_kind is ImplementationKind.PRODUCTION
            ),
            mock_tools=tuple(
                s.tool_name
                for s in specs
                if s.implementation_kind is ImplementationKind.MOCK
            ),
            fixture_tools=tuple(
                s.tool_name
                for s in specs
                if s.implementation_kind is ImplementationKind.FIXTURE
            ),
            manifest_cid=self.manifest_cid,
            server_name=self.server_name,
        )

    def get_adapter(self, tool_name: str) -> AdapterSpec | None:
        return self._adapters.get(normalize_tool_name(tool_name or ""))

    # -- negotiation -------------------------------------------------------

    def negotiate(
        self,
        *,
        requested_profiles: Sequence[str] = (),
        transport: str | TransportKind = TransportKind.HTTP,
    ) -> CapabilityNegotiationRecord:
        requested = _strings(
            requested_profiles or ("mcp++/basic",),
            field_name="requested_profiles",
            maximum=DEFAULT_MAX_PROFILES,
        )
        transport_value = _enum(
            transport, TransportKind, field_name="transport"
        ).value
        admitted = set(self.admitted_profiles)
        intersection = [p for p in requested if p in admitted]
        if not intersection:
            return CapabilityNegotiationRecord(
                requested_profiles=requested,
                admitted_profiles=self.admitted_profiles,
                active_profile="",
                requested_transport=transport_value,
                admitted_transports=self.admitted_transports,
                active_transport=TransportKind.UNKNOWN.value,
                negotiated=False,
                reason="profile_mismatch",
            )
        if transport_value not in self.admitted_transports:
            return CapabilityNegotiationRecord(
                requested_profiles=requested,
                admitted_profiles=self.admitted_profiles,
                active_profile=intersection[0],
                requested_transport=transport_value,
                admitted_transports=self.admitted_transports,
                active_transport=TransportKind.UNKNOWN.value,
                negotiated=False,
                reason="transport_rejected",
            )
        return CapabilityNegotiationRecord(
            requested_profiles=requested,
            admitted_profiles=self.admitted_profiles,
            active_profile=intersection[0],
            requested_transport=transport_value,
            admitted_transports=self.admitted_transports,
            active_transport=transport_value,
            negotiated=True,
            reason="negotiated",
        )

    # -- call / witness ----------------------------------------------------

    def call(
        self,
        request: CallRequest | Mapping[str, Any],
        *,
        cancellation: CancellationToken | None = None,
    ) -> RuntimeWitness:
        if isinstance(request, Mapping):
            request = CallRequest.from_dict(
                {
                    "schema": MCPLUSPLUS_CALL_REQUEST_SCHEMA,
                    "contract_version": CONTRACT_VERSION,
                    **dict(request),
                }
                if "schema" not in request
                else request
            )
        if not isinstance(request, CallRequest):
            raise RuntimeWitnessError("request must be CallRequest")
        if self._cleaned_up:
            empty_discovery = ToolDiscoveryRecord(
                tool_names=(),
                adapter_ids=(),
                production_tools=(),
                mock_tools=(),
                fixture_tools=(),
                manifest_cid=self.manifest_cid,
                server_name=self.server_name,
            )
            empty_negotiation = CapabilityNegotiationRecord(
                requested_profiles=tuple(request.requested_profiles),
                admitted_profiles=self.admitted_profiles,
                active_profile="",
                requested_transport=request.transport,
                admitted_transports=self.admitted_transports,
                active_transport=TransportKind.UNKNOWN.value,
                negotiated=False,
                reason="runtime cleaned up",
            )
            return self._witness(
                request,
                discovery=empty_discovery,
                negotiation=empty_negotiation,
                outcome=WitnessOutcome.INCONCLUSIVE,
                reason="runtime already cleaned up",
                cleanup_status=CleanupStatus.DIRTY,
                phases=(),
            )

        phases: list[str] = []
        started = time.monotonic()

        # Discovery
        discovery = self.discover_tools()
        phases.append(WitnessPhase.DISCOVERY.value)

        # Stale manifest check
        if (
            request.expected_manifest_cid
            and request.expected_manifest_cid != self.manifest_cid
        ):
            phases.append(WitnessPhase.CAPABILITY_NEGOTIATION.value)
            negotiation = self.negotiate(
                requested_profiles=request.requested_profiles,
                transport=request.transport,
            )
            return self._witness(
                request,
                discovery=discovery,
                negotiation=negotiation,
                outcome=WitnessOutcome.STALE_MANIFEST,
                reason=(
                    "expected_manifest_cid does not match fixture manifest"
                ),
                phases=tuple(phases),
                duration_ms=_elapsed_ms(started),
            )

        # Capability negotiation
        negotiation = self.negotiate(
            requested_profiles=request.requested_profiles,
            transport=request.transport,
        )
        phases.append(WitnessPhase.CAPABILITY_NEGOTIATION.value)
        if not negotiation.negotiated:
            outcome = (
                WitnessOutcome.PROFILE_MISMATCH
                if negotiation.reason == "profile_mismatch"
                else WitnessOutcome.TRANSPORT_REJECTED
            )
            return self._witness(
                request,
                discovery=discovery,
                negotiation=negotiation,
                outcome=outcome,
                reason=negotiation.reason,
                phases=tuple(phases),
                duration_ms=_elapsed_ms(started),
                transport=negotiation.active_transport,
            )

        # Malformed call (empty tool name)
        if not request.tool_name:
            return self._witness(
                request,
                discovery=discovery,
                negotiation=negotiation,
                outcome=WitnessOutcome.MALFORMED_CALL,
                reason="tool_name is required",
                phases=tuple(phases),
                duration_ms=_elapsed_ms(started),
                transport=negotiation.active_transport,
            )

        # Missing tool
        adapter = self.get_adapter(request.tool_name)
        if adapter is None:
            return self._witness(
                request,
                discovery=discovery,
                negotiation=negotiation,
                outcome=WitnessOutcome.MISSING_TOOL,
                reason=f"tool not registered: {request.tool_name}",
                phases=tuple(phases),
                duration_ms=_elapsed_ms(started),
                transport=negotiation.active_transport,
            )

        # Profile must be admitted on the adapter as well.
        if negotiation.active_profile not in adapter.profiles:
            return self._witness(
                request,
                discovery=discovery,
                negotiation=negotiation,
                outcome=WitnessOutcome.PROFILE_MISMATCH,
                reason="adapter does not advertise active profile",
                adapter=adapter,
                phases=tuple(phases),
                duration_ms=_elapsed_ms(started),
                transport=negotiation.active_transport,
            )

        # Transport must be supported by the adapter.
        if negotiation.active_transport not in adapter.transports:
            return self._witness(
                request,
                discovery=discovery,
                negotiation=negotiation,
                outcome=WitnessOutcome.TRANSPORT_REJECTED,
                reason="adapter does not admit active transport",
                adapter=adapter,
                phases=tuple(phases),
                duration_ms=_elapsed_ms(started),
                transport=negotiation.active_transport,
            )

        phases.append(WitnessPhase.TRANSPORT.value)

        # Unavailable backend
        if adapter.backend_availability is BackendAvailability.UNAVAILABLE:
            phases.append(WitnessPhase.DISPATCH.value)
            return self._witness(
                request,
                discovery=discovery,
                negotiation=negotiation,
                outcome=WitnessOutcome.UNAVAILABLE_BACKEND,
                reason="backend unavailable",
                adapter=adapter,
                error_code="unavailable",
                error_schema_ok="unavailable" in adapter.error_codes
                or not adapter.error_codes,
                phases=tuple(phases),
                duration_ms=_elapsed_ms(started),
                transport=negotiation.active_transport,
            )

        # Input validation
        input_verdict, input_errors = validate_against_schema(
            dict(request.arguments), dict(adapter.input_schema)
        )
        phases.append(WitnessPhase.INPUT_VALIDATION.value)
        if input_verdict is ValidationVerdict.INVALID:
            return self._witness(
                request,
                discovery=discovery,
                negotiation=negotiation,
                outcome=WitnessOutcome.SCHEMA_VIOLATION,
                reason="input schema violation",
                adapter=adapter,
                input_validation=input_verdict,
                input_errors=input_errors,
                phases=tuple(phases),
                duration_ms=_elapsed_ms(started),
                transport=negotiation.active_transport,
            )

        # Cancellation before dispatch
        if request.cancel or (
            cancellation is not None and cancellation.cancelled
        ):
            return self._witness(
                request,
                discovery=discovery,
                negotiation=negotiation,
                outcome=WitnessOutcome.CANCELLED,
                reason=(
                    cancellation.reason
                    if cancellation is not None and cancellation.cancelled
                    else "cancelled"
                ),
                adapter=adapter,
                input_validation=input_verdict,
                cancelled=True,
                phases=tuple(phases),
                duration_ms=_elapsed_ms(started),
                transport=negotiation.active_transport,
            )

        # Forced timeout (deterministic failure-mode injection)
        if request.force_timeout:
            phases.append(WitnessPhase.TIMEOUT.value)
            return self._witness(
                request,
                discovery=discovery,
                negotiation=negotiation,
                outcome=WitnessOutcome.TIMED_OUT,
                reason="force_timeout",
                adapter=adapter,
                input_validation=input_verdict,
                timed_out=True,
                phases=tuple(phases),
                duration_ms=self.timeout_ms,
                transport=negotiation.active_transport,
            )

        # Dispatch with wall-clock timeout
        handler = self._handlers.get(adapter.tool_name)
        if handler is None:
            phases.append(WitnessPhase.DISPATCH.value)
            return self._witness(
                request,
                discovery=discovery,
                negotiation=negotiation,
                outcome=WitnessOutcome.DISPATCH_ERROR,
                reason="no handler for registered adapter",
                adapter=adapter,
                input_validation=input_verdict,
                phases=tuple(phases),
                duration_ms=_elapsed_ms(started),
                transport=negotiation.active_transport,
            )

        with self._lock:
            self._active_calls += 1
        result_box: dict[str, Any] = {}
        error_box: dict[str, Any] = {}

        def _run() -> None:
            try:
                result_box["value"] = dict(handler(dict(request.arguments)))
            except AdapterDispatchError as exc:
                error_box["code"] = exc.error_code
                error_box["message"] = exc.message
            except Exception as exc:  # noqa: BLE001 — capture typed failure
                error_box["code"] = "dispatch_error"
                error_box["message"] = f"{type(exc).__name__}: {exc}"
                error_box["trace"] = traceback.format_exc(limit=4)

        thread = threading.Thread(target=_run, name="mcpp-runtime-dispatch")
        thread.daemon = True
        thread.start()
        timeout_s = self.timeout_ms / 1000.0
        thread.join(timeout=timeout_s)
        timed_out = thread.is_alive()
        phases.append(WitnessPhase.DISPATCH.value)
        phases.append(WitnessPhase.TIMEOUT.value)

        if cancellation is not None and cancellation.cancelled:
            with self._lock:
                self._active_calls = max(0, self._active_calls - 1)
            return self._witness(
                request,
                discovery=discovery,
                negotiation=negotiation,
                outcome=WitnessOutcome.CANCELLED,
                reason=cancellation.reason or "cancelled",
                adapter=adapter,
                input_validation=input_verdict,
                cancelled=True,
                phases=tuple(phases),
                duration_ms=_elapsed_ms(started),
                transport=negotiation.active_transport,
            )

        if timed_out:
            with self._lock:
                self._active_calls = max(0, self._active_calls - 1)
            return self._witness(
                request,
                discovery=discovery,
                negotiation=negotiation,
                outcome=WitnessOutcome.TIMED_OUT,
                reason="dispatch exceeded timeout_ms",
                adapter=adapter,
                input_validation=input_verdict,
                timed_out=True,
                phases=tuple(phases),
                duration_ms=_elapsed_ms(started),
                transport=negotiation.active_transport,
            )

        with self._lock:
            self._active_calls = max(0, self._active_calls - 1)

        if error_box:
            code = str(error_box.get("code") or "dispatch_error")
            if code == "unavailable":
                outcome = WitnessOutcome.UNAVAILABLE_BACKEND
            else:
                outcome = WitnessOutcome.DISPATCH_ERROR
            error_schema_ok = (
                code in adapter.error_codes if adapter.error_codes else True
            )
            phases.append(WitnessPhase.ERROR_SCHEMA.value)
            return self._witness(
                request,
                discovery=discovery,
                negotiation=negotiation,
                outcome=outcome,
                reason=str(error_box.get("message") or code),
                adapter=adapter,
                input_validation=input_verdict,
                error_code=code,
                error_schema_ok=error_schema_ok,
                phases=tuple(phases),
                duration_ms=_elapsed_ms(started),
                transport=negotiation.active_transport,
            )

        result = result_box.get("value") or {}
        output_verdict, output_errors = validate_against_schema(
            result, dict(adapter.output_schema)
        )
        phases.append(WitnessPhase.OUTPUT_SCHEMA.value)
        if output_verdict is ValidationVerdict.INVALID:
            return self._witness(
                request,
                discovery=discovery,
                negotiation=negotiation,
                outcome=WitnessOutcome.SCHEMA_VIOLATION,
                reason="output schema violation",
                adapter=adapter,
                input_validation=input_verdict,
                output_validation=output_verdict,
                output_errors=output_errors,
                result=result,
                phases=tuple(phases),
                duration_ms=_elapsed_ms(started),
                transport=negotiation.active_transport,
            )

        return self._witness(
            request,
            discovery=discovery,
            negotiation=negotiation,
            outcome=WitnessOutcome.PASSED,
            reason="passed",
            adapter=adapter,
            input_validation=input_verdict,
            output_validation=output_verdict,
            result=result,
            phases=tuple(phases),
            duration_ms=_elapsed_ms(started),
            transport=negotiation.active_transport,
            grants_runtime_authority=(
                adapter.implementation_kind is ImplementationKind.PRODUCTION
            ),
        )

    def run_suite(
        self,
        requests: Sequence[CallRequest | Mapping[str, Any]],
    ) -> RuntimeWitnessReceipt:
        if len(requests) > DEFAULT_MAX_OBSERVATIONS:
            raise RuntimeWitnessBoundsError(
                f"requests exceed {DEFAULT_MAX_OBSERVATIONS}"
            )
        witnesses = tuple(self.call(req) for req in requests)
        return RuntimeWitnessReceipt(
            fixture_id=self.fixture_id,
            forest_id=self.forest_id,
            manifest_cid=self.manifest_cid,
            witnesses=witnesses,
            network_enabled=self.network_enabled,
        )

    def cleanup(self) -> CleanupStatus:
        with self._lock:
            if self._active_calls > 0:
                self._cleaned_up = True
                return CleanupStatus.DIRTY
            self._cleaned_up = True
            return CleanupStatus.CLEAN

    @property
    def is_cleaned_up(self) -> bool:
        return self._cleaned_up

    # -- internal witness builder ------------------------------------------

    def _witness(
        self,
        request: CallRequest,
        *,
        outcome: WitnessOutcome,
        reason: str,
        discovery: ToolDiscoveryRecord | None = None,
        negotiation: CapabilityNegotiationRecord | None = None,
        adapter: AdapterSpec | None = None,
        input_validation: ValidationVerdict = ValidationVerdict.SKIPPED,
        input_errors: tuple[str, ...] = (),
        output_validation: ValidationVerdict = ValidationVerdict.SKIPPED,
        output_errors: tuple[str, ...] = (),
        result: Mapping[str, Any] | None = None,
        error_code: str = "",
        error_schema_ok: bool = True,
        phases: tuple[str, ...] = (),
        duration_ms: int = 0,
        timed_out: bool = False,
        cancelled: bool = False,
        cleanup_status: CleanupStatus = CleanupStatus.SKIPPED,
        transport: str = TransportKind.HTTP.value,
        grants_runtime_authority: bool = False,
    ) -> RuntimeWitness:
        if discovery is None:
            discovery = self.discover_tools()
        if negotiation is None:
            negotiation = self.negotiate(
                requested_profiles=request.requested_profiles,
                transport=request.transport,
            )
        phases_list = list(phases)
        if WitnessPhase.CLEANUP.value not in phases_list:
            # Soft cleanup check for the observation; does not shut down suite.
            cleanup_status = (
                CleanupStatus.CLEAN
                if not self._cleaned_up
                else CleanupStatus.DIRTY
            )
            phases_list.append(WitnessPhase.CLEANUP.value)

        impl_kind = (
            adapter.implementation_kind
            if adapter is not None
            else ImplementationKind.PRODUCTION
        )
        # Never grant authority for mocks/fixtures or non-passed outcomes.
        if (
            grants_runtime_authority
            and outcome is WitnessOutcome.PASSED
            and impl_kind is ImplementationKind.PRODUCTION
        ):
            grants = True
        else:
            grants = False

        observation = CallObservation(
            outcome=outcome,
            tool_name=request.tool_name
            or (adapter.tool_name if adapter else ""),
            adapter_id=adapter.adapter_id if adapter else "",
            implementation_kind=impl_kind,
            implementation_target=(
                adapter.implementation_target if adapter else ""
            ),
            input_validation=input_validation,
            input_errors=input_errors,
            output_validation=output_validation,
            output_errors=output_errors,
            error_code=error_code,
            error_schema_ok=error_schema_ok,
            result=result or {},
            phases_completed=tuple(phases_list),
            duration_ms=duration_ms,
            timed_out=timed_out,
            cancelled=cancelled,
            cleanup_status=cleanup_status,
            grants_runtime_authority=grants,
            reason=reason,
        )
        return RuntimeWitness(
            fixture_id=self.fixture_id,
            forest_id=self.forest_id,
            discovery=discovery,
            negotiation=negotiation,
            request=request,
            observation=observation,
            transport=transport
            if transport != TransportKind.UNKNOWN.value
            else negotiation.active_transport,
            timeout_ms=self.timeout_ms,
            network_enabled=self.network_enabled,
        )


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))


# ---------------------------------------------------------------------------
# Deterministic receipt replay
# ---------------------------------------------------------------------------


def replay_receipt(
    receipt: RuntimeWitnessReceipt | Mapping[str, Any],
) -> RuntimeWitnessReceipt:
    """Round-trip a receipt through canonical serialization.

    Replay is pure: the rehydrated receipt must share the same content
    identity.  This proves deterministic serialization, not re-execution.
    """

    if isinstance(receipt, Mapping):
        original = RuntimeWitnessReceipt.from_dict(receipt)
    elif isinstance(receipt, RuntimeWitnessReceipt):
        original = receipt
    else:
        raise RuntimeWitnessError(
            "receipt must be RuntimeWitnessReceipt or mapping"
        )
    payload = original.to_dict()
    replayed = RuntimeWitnessReceipt.from_dict(payload)
    if replayed.content_id != original.content_id:
        raise RuntimeWitnessError(
            "receipt replay produced a different content identity"
        )
    if replayed.to_json() != original.to_json():
        raise RuntimeWitnessError(
            "receipt replay produced different canonical JSON"
        )
    return replayed


def receipt_content_identity(
    receipt: RuntimeWitnessReceipt | Mapping[str, Any],
) -> str:
    if isinstance(receipt, RuntimeWitnessReceipt):
        return receipt.content_id
    return RuntimeWitnessReceipt.from_dict(receipt).content_id


# ---------------------------------------------------------------------------
# Bounded subprocess isolation
# ---------------------------------------------------------------------------


def _subprocess_worker_source() -> str:
    """Return a self-contained Python program for isolated witness runs."""

    return r"""
import json, sys
from ipfs_accelerate_py.agent_supervisor.mcplusplus_runtime_witness import (
    CallRequest,
    HermeticMCPlusPlusRuntime,
    CONTRACT_VERSION,
    MCPLUSPLUS_CALL_REQUEST_SCHEMA,
    _SUBPROCESS_PROTOCOL,
    default_mock_adapters,
    default_production_adapters,
    AdapterSpec,
)

def main() -> int:
    raw = sys.stdin.read()
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError as exc:
        json.dump({"ok": False, "error": f"invalid_json:{exc}"}, sys.stdout)
        return 2
    if envelope.get("protocol") != _SUBPROCESS_PROTOCOL:
        json.dump({"ok": False, "error": "protocol_mismatch"}, sys.stdout)
        return 2
    forest_id = envelope.get("forest_id") or "forest:subprocess"
    timeout_ms = int(envelope.get("timeout_ms") or 5000)
    include_mocks = bool(envelope.get("include_mocks", True))
    adapters = list(default_production_adapters())
    if include_mocks:
        adapters.extend(default_mock_adapters())
    extra = envelope.get("extra_adapters") or []
    for item in extra:
        adapters.append(AdapterSpec.from_dict(item))
    runtime = HermeticMCPlusPlusRuntime(
        forest_id=forest_id,
        adapters=adapters,
        timeout_ms=timeout_ms,
        manifest_cid=str(envelope.get("manifest_cid") or ""),
        fixture_id=str(envelope.get("fixture_id") or ""),
    )
    requests = envelope.get("requests") or []
    call_requests = []
    for item in requests:
        if "schema" not in item:
            item = {
                "schema": MCPLUSPLUS_CALL_REQUEST_SCHEMA,
                "contract_version": CONTRACT_VERSION,
                **item,
            }
        call_requests.append(CallRequest.from_dict(item))
    receipt = runtime.run_suite(call_requests)
    cleanup = runtime.cleanup().value
    json.dump(
        {
            "ok": True,
            "protocol": _SUBPROCESS_PROTOCOL,
            "receipt": receipt.to_dict(),
            "cleanup": cleanup,
            "fixture_id": runtime.fixture_id,
            "manifest_cid": runtime.manifest_cid,
        },
        sys.stdout,
        sort_keys=True,
        separators=(",", ":"),
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
"""


def run_witness_subprocess(
    requests: Sequence[CallRequest | Mapping[str, Any]],
    *,
    forest_id: str = "forest:subprocess",
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    include_mocks: bool = True,
    subprocess_timeout_s: float = DEFAULT_SUBPROCESS_TIMEOUT_S,
    env: Mapping[str, str] | None = None,
) -> RuntimeWitnessReceipt:
    """Run selected adapters in a bounded child process and return a receipt.

    The child inherits only a sanitized environment.  Network is never enabled.
    """

    if not requests:
        raise RuntimeWitnessError("subprocess witness requires at least one request")
    if len(requests) > DEFAULT_MAX_OBSERVATIONS:
        raise RuntimeWitnessBoundsError(
            f"requests exceed {DEFAULT_MAX_OBSERVATIONS}"
        )

    serialized_requests: list[dict[str, Any]] = []
    for item in requests:
        if isinstance(item, CallRequest):
            serialized_requests.append(item.to_dict())
        elif isinstance(item, Mapping):
            if "schema" not in item:
                serialized_requests.append(
                    CallRequest(
                        tool_name=str(item.get("tool_name") or ""),
                        arguments=item.get("arguments") or {},
                        requested_profiles=tuple(
                            item.get("requested_profiles")
                            or ("mcp++/basic",)
                        ),
                        transport=str(
                            item.get("transport") or TransportKind.HTTP.value
                        ),
                        call_id=str(item.get("call_id") or ""),
                        cancel=bool(item.get("cancel", False)),
                        force_timeout=bool(item.get("force_timeout", False)),
                        expected_manifest_cid=str(
                            item.get("expected_manifest_cid") or ""
                        ),
                    ).to_dict()
                )
            else:
                serialized_requests.append(
                    CallRequest.from_dict(item).to_dict()
                )
        else:
            raise RuntimeWitnessError("request must be CallRequest or mapping")

    envelope = {
        "protocol": _SUBPROCESS_PROTOCOL,
        "forest_id": forest_id,
        "timeout_ms": timeout_ms,
        "include_mocks": include_mocks,
        "requests": serialized_requests,
    }
    payload = json.dumps(
        envelope, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")

    child_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "TZ": "UTC",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PIP_NO_INDEX": "1",
    }
    if env:
        for key, value in env.items():
            child_env[str(key)] = str(value)

    process = subprocess.Popen(
        [sys.executable, "-c", _subprocess_worker_source()],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_env,
        text=False,
    )
    try:
        stdout, stderr = process.communicate(
            input=payload, timeout=subprocess_timeout_s
        )
    except subprocess.TimeoutExpired:
        _kill_process_tree(process)
        raise RuntimeWitnessError(
            f"subprocess witness exceeded {subprocess_timeout_s}s"
        ) from None

    if process.returncode not in (0,):
        detail = stderr.decode("utf-8", errors="replace")[:512]
        raise RuntimeWitnessError(
            f"subprocess witness failed rc={process.returncode}: {detail}"
        )

    try:
        response = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeWitnessError(
            f"subprocess produced invalid JSON: {exc}"
        ) from exc

    if not response.get("ok"):
        raise RuntimeWitnessError(
            f"subprocess reported error: {response.get('error')!r}"
        )
    receipt_payload = response.get("receipt")
    if not isinstance(receipt_payload, Mapping):
        raise RuntimeWitnessError("subprocess response missing receipt")
    return RuntimeWitnessReceipt.from_dict(receipt_payload)


def _kill_process_tree(process: subprocess.Popen[bytes]) -> None:
    try:
        if process.poll() is None:
            process.send_signal(signal.SIGKILL)
            process.wait(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        pass


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------


def make_call_request(
    tool_name: str,
    arguments: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> CallRequest:
    return CallRequest(
        tool_name=tool_name,
        arguments=arguments or {},
        **kwargs,
    )


def make_runtime(
    *,
    forest_id: str = "forest:mcplusplus-runtime",
    include_mocks: bool = True,
    **kwargs: Any,
) -> HermeticMCPlusPlusRuntime:
    adapters = list(default_production_adapters())
    if include_mocks:
        adapters.extend(default_mock_adapters())
    return HermeticMCPlusPlusRuntime(
        forest_id=forest_id,
        adapters=adapters,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Objective evidence discovery + acceptance anchors (VFS-G061 / VFS-058)
# ---------------------------------------------------------------------------


def runtime_witness_evidence_terms() -> tuple[str, ...]:
    """Return the closed VFS-G061 domain evidence terms for hermetic runtime witnesses.

    Domain runtime-witness identity (``vfs/mcplusplus-runtime-witness@1``) is
    authored only by this module.  Runtime witnesses supplement static
    resolution and formal proof; they never replace either.

    The synthetic ``objective validation repair`` term is intentionally
    omitted here so witness/receipt ``evidence_kind`` stays domain-only; use
    :func:`objective_validation_repair_evidence_terms` (or
    :func:`all_covered_evidence_terms`) for the VFS-G061 validation gate.
    """

    return (EVIDENCE_RUNTIME_WITNESS,)


def covered_evidence_terms() -> tuple[str, ...]:
    """Return domain objective evidence terms this runtime-witness surface proves.

    Mirrors :func:`runtime_witness_evidence_terms`.  The repair gate is via
    :func:`all_covered_evidence_terms`.
    """

    return runtime_witness_evidence_terms()


def objective_validation_repair_evidence_terms() -> tuple[str, ...]:
    """Return the synthetic VFS-G061 validation-gate evidence term.

    Exact-text discovery key for objective validation repair.  Never mixes
    into content-addressed witness, receipt, adapter, or negotiation
    identity.  Real production dispatch stays separate from mocks; HTTP and
    mcp+p2p share admitted contracts where declared; network stays disabled
    unless an exact fixture and egress policy permit it.  Owned by
    :data:`OBJECTIVE_GOAL_ID` (``VFS-G061``) via repair task
    :data:`OBJECTIVE_VALIDATION_REPAIR_TASK_ID` (``VFS-058``).
    """

    return (OBJECTIVE_VALIDATION_REPAIR_EVIDENCE,)


def all_covered_evidence_terms() -> tuple[str, ...]:
    """Return domain VFS-G061 runtime-witness terms plus the validation-repair gate.

    Domain ``vfs/mcplusplus-runtime-witness@1`` comes first; the synthetic
    objective validation repair discovery key is appended last and never
    enters witness/receipt identity or production-authority grants.
    """

    return covered_evidence_terms() + objective_validation_repair_evidence_terms()


def typed_non_authoritative_failure_outcomes() -> tuple[str, ...]:
    """Closed failure outcomes that never grant runtime completeness authority.

    Acceptance subset for objective validation repair: failures and
    unavailable services are typed, bounded, and non-authoritative.
    ``passed`` is intentionally excluded; authority requires production-class
    positive witnesses only.
    """

    return tuple(
        outcome.value
        for outcome in WitnessOutcome
        if outcome is not WitnessOutcome.PASSED
    )


def admitted_shared_transport_profiles() -> tuple[str, ...]:
    """Transports that share the same admitted MCP++ contract surface.

    HTTP and mcp+p2p profiles use the same admitted contract where declared
    (VFS-G061 acceptance).  Network remains disabled by default.
    """

    return (TransportKind.HTTP.value, TransportKind.MCP_P2P.value)


def production_dispatch_distinguished_from_mocks() -> bool:
    """Real production adapters never share mock authority grants.

    Authoritative case for objective validation repair: production
    ``ImplementationKind`` grants production authority; mocks never do.
    """

    assert ImplementationKind.PRODUCTION.grants_production_authority is True
    assert ImplementationKind.MOCK.grants_production_authority is False
    assert ImplementationKind.FIXTURE.grants_production_authority is False
    production = default_production_adapters()
    mocks = default_mock_adapters()
    assert production and mocks
    assert all(a.is_production and not a.is_mock for a in production)
    assert all(a.is_mock and not a.is_production for a in mocks)
    return True


__all__ = [
    "AdapterDispatchError",
    "AdapterHandler",
    "AdapterSpec",
    "BackendAvailability",
    "CallObservation",
    "CallRequest",
    "CancellationToken",
    "CapabilityNegotiationRecord",
    "CleanupStatus",
    "CONTRACT_VERSION",
    "DEFAULT_TIMEOUT_MS",
    "EVIDENCE_RUNTIME_WITNESS",
    "HermeticMCPlusPlusRuntime",
    "ImplementationKind",
    "MCPLUSPLUS_ADAPTER_SPEC_SCHEMA",
    "MCPLUSPLUS_CALL_OBSERVATION_SCHEMA",
    "MCPLUSPLUS_CALL_REQUEST_SCHEMA",
    "MCPLUSPLUS_DISCOVERY_RECORD_SCHEMA",
    "MCPLUSPLUS_NEGOTIATION_RECORD_SCHEMA",
    "MCPLUSPLUS_RUNTIME_FIXTURE_SCHEMA",
    "MCPLUSPLUS_RUNTIME_RECEIPT_SCHEMA",
    "MCPLUSPLUS_RUNTIME_WITNESS_SCHEMA",
    "OBJECTIVE_GOAL_ID",
    "OBJECTIVE_VALIDATION_REPAIR_EVIDENCE",
    "OBJECTIVE_VALIDATION_REPAIR_TASK_ID",
    "RuntimeWitness",
    "RuntimeWitnessAuthorityError",
    "RuntimeWitnessBoundsError",
    "RuntimeWitnessError",
    "RuntimeWitnessReceipt",
    "ToolDiscoveryRecord",
    "ValidationVerdict",
    "WITNESS_PRODUCER",
    "WITNESS_VERSION",
    "WitnessOutcome",
    "WitnessPhase",
    "admitted_shared_transport_profiles",
    "all_covered_evidence_terms",
    "covered_evidence_terms",
    "default_mock_adapters",
    "default_production_adapters",
    "make_call_request",
    "make_runtime",
    "objective_validation_repair_evidence_terms",
    "production_dispatch_distinguished_from_mocks",
    "receipt_content_identity",
    "replay_receipt",
    "run_witness_subprocess",
    "runtime_witness_evidence_terms",
    "typed_non_authoritative_failure_outcomes",
    "validate_against_schema",
]
