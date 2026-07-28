"""Symbolic verification of supervisor coordination protocols.

This module is the protocol-level counterpart to the supervisor state and
authorization models.  It supplies reviewed, versioned Tamarin and ProVerif
models for:

* claimant authentication and lease-grant correspondence;
* fencing-token freshness and replay resistance;
* proof-receipt binding and merge authorization; and
* an optional remote-attestation exchange.

The external tools are intentionally treated as hostile integration
boundaries.  Finding an executable, obtaining a version, or successfully
running an installer never makes a lane available.  A lane becomes conformant
only after its pinned executable passes an end-to-end fixture which checks
four security-query classes *and* finds a known attack.  Verification receipts
bind the exact executable, fixture, model source, query set, command, and
bounded output.

Attack output is never retained verbatim.  It is reduced to deterministic,
role-normalized steps whose message values are content digests.  Every
counterexample also binds the reviewed abstraction for the violated query.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from .formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)
from .prover_matrix_registry import (
    CommandRequest,
    CommandResult,
    _default_command_runner as _bounded_command_runner,
)


PROTOCOL_VERIFICATION_VERSION = 1
PROTOCOL_QUERY_SCHEMA = "ipfs_accelerate_py/agent-supervisor/protocol-query@1"
PROTOCOL_MODEL_SCHEMA = "ipfs_accelerate_py/agent-supervisor/protocol-model@1"
PROTOCOL_FIXTURE_SCHEMA = "ipfs_accelerate_py/agent-supervisor/protocol-fixture@1"
PROTOCOL_CONFORMANCE_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/protocol-conformance-receipt@1"
)
PROTOCOL_TOOL_CAPABILITY_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/protocol-tool-capability@1"
)
PROTOCOL_ATTACK_COUNTEREXAMPLE_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/protocol-attack-counterexample@1"
)
PROTOCOL_TOOLCHAIN_RECEIPT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/protocol-toolchain-receipt@1"
)
PROTOCOL_QUERY_RESULT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/protocol-query-result@1"
)
PROTOCOL_LANE_RESULT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/protocol-lane-result@1"
)
PROTOCOL_SUITE_RESULT_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/protocol-suite-result@1"
)

DEFAULT_PROTOCOL_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_PROTOCOL_OUTPUT_BYTES = 256 * 1024
DEFAULT_MAX_EXECUTABLE_BYTES = 32 * 1024 * 1024

_IDENTITY_RE = re.compile(r"^(?:b[a-z2-7]+|sha256:[0-9a-f]{64})$")
_SYMBOL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_TRACE_LINE_RE = re.compile(
    r"^\s*TRACE\s+query=(?P<query>[A-Za-z0-9_.-]+)"
    r"\s+step=(?P<step>\d+)\s+event=(?P<event>[A-Za-z0-9_.:-]+)"
    r"(?:\s+actor=(?P<actor>[A-Za-z0-9_.:-]+))?"
    r"(?:\s+message=(?P<message>\S+))?\s*$",
    re.IGNORECASE,
)


class ProtocolValidationError(ContractValidationError):
    """Raised when protocol evidence is malformed or overclaims authority."""


class ProtocolTool(str, Enum):
    TAMARIN = "tamarin"
    PROVERIF = "proverif"


class ProtocolProperty(str, Enum):
    CLAIMANT_AUTHENTICATION = "claimant_authentication"
    LEASE_GRANTS = "lease_grants"
    FENCING_FRESHNESS = "fencing_freshness"
    REPLAY_RESISTANCE = "replay_resistance"
    RECEIPT_BINDING = "receipt_binding"
    MERGE_AUTHORIZATION = "merge_authorization"
    ATTESTATION_EXCHANGE = "attestation_exchange"


class ProtocolQueryKind(str, Enum):
    SECRECY = "secrecy"
    AUTHENTICITY = "authenticity"
    CORRESPONDENCE = "correspondence"
    REPLAY = "replay"


class ConformanceStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    ERROR = "error"
    NOT_RUN = "not_run"


class ToolCapabilityStatus(str, Enum):
    UNAVAILABLE = "unavailable"
    NONCONFORMANT = "nonconformant"
    CONFORMANT = "conformant"


class ProtocolVerdict(str, Enum):
    VERIFIED = "verified"
    VIOLATED = "violated"
    INCONCLUSIVE = "inconclusive"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class ToolRunStatus(str, Enum):
    PASSED = "passed"
    VIOLATED = "violated"
    INCONCLUSIVE = "inconclusive"
    TIMED_OUT = "timed_out"
    ERROR = "error"


def _text(value: Any, name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ProtocolValidationError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise ProtocolValidationError(f"{name} must not be empty")
    return result


def _source(value: Any, name: str) -> str:
    """Validate model text without stripping its identity-significant bytes."""

    if not isinstance(value, str) or not value.strip():
        raise ProtocolValidationError(f"{name} must be non-empty model text")
    if not value.endswith("\n"):
        raise ProtocolValidationError(f"{name} must end with a newline")
    return value


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    try:
        return kind(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise ProtocolValidationError(f"unsupported {name}") from exc


def _strings(
    values: Sequence[Any] | None,
    name: str,
    *,
    required: bool = False,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    if values is None:
        result: tuple[str, ...] = ()
    elif isinstance(values, (str, bytes, bytearray)):
        raise ProtocolValidationError(f"{name} must be a sequence")
    else:
        normalized = tuple(_text(item, name) for item in values)
        result = (
            tuple(dict.fromkeys(normalized))
            if preserve_order
            else tuple(sorted(set(normalized)))
        )
    if required and not result:
        raise ProtocolValidationError(f"{name} must not be empty")
    return result


def _identity(value: Any, name: str) -> str:
    result = _text(value, name)
    if not _IDENTITY_RE.fullmatch(result):
        raise ProtocolValidationError(f"{name} must be a canonical identity")
    return result


def _digest(value: str | bytes) -> str:
    raw = value.encode("utf-8", errors="replace") if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _schema(payload: Mapping[str, Any], expected: str) -> None:
    supplied = payload.get("schema")
    if supplied not in (None, "", expected):
        raise ProtocolValidationError(
            f"unsupported schema {supplied!r}; expected {expected}"
        )


def _claimed_identity(payload: Mapping[str, Any], actual: str, noun: str) -> None:
    claimed = payload.get("content_id") or payload.get("identity")
    if claimed and claimed != actual:
        raise ProtocolValidationError(f"{noun} identity does not match payload")


def _nonnegative(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProtocolValidationError(f"{name} must be a non-negative integer")
    return value


def _positive(value: Any, name: str) -> int:
    result = _nonnegative(value, name)
    if result < 1:
        raise ProtocolValidationError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class ProtocolQuery(CanonicalContract):
    """One reviewed security query and its explicit abstraction boundary."""

    SCHEMA = PROTOCOL_QUERY_SCHEMA

    query_id: str
    property: ProtocolProperty
    kind: ProtocolQueryKind
    abstraction: str
    assumptions: tuple[str, ...]
    excluded_behaviors: tuple[str, ...]
    tamarin_name: str
    proverif_label: str
    required: bool = True

    def __post_init__(self) -> None:
        query_id = _text(self.query_id, "query_id")
        if not re.fullmatch(r"[a-z][a-z0-9_.-]*", query_id):
            raise ProtocolValidationError("query_id must be a stable lowercase name")
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(
            self, "property", _enum(self.property, ProtocolProperty, "property")
        )
        object.__setattr__(
            self, "kind", _enum(self.kind, ProtocolQueryKind, "query kind")
        )
        object.__setattr__(
            self, "abstraction", _text(self.abstraction, "abstraction")
        )
        object.__setattr__(
            self,
            "assumptions",
            _strings(self.assumptions, "assumptions", required=True),
        )
        object.__setattr__(
            self,
            "excluded_behaviors",
            _strings(
                self.excluded_behaviors, "excluded_behaviors", required=True
            ),
        )
        tamarin_name = _text(self.tamarin_name, "tamarin_name")
        if not _SYMBOL_RE.fullmatch(tamarin_name):
            raise ProtocolValidationError("tamarin_name must be a Tamarin symbol")
        object.__setattr__(self, "tamarin_name", tamarin_name)
        object.__setattr__(
            self, "proverif_label", _text(self.proverif_label, "proverif_label")
        )
        if not isinstance(self.required, bool):
            raise ProtocolValidationError("required must be a boolean")

    def symbol_for(self, tool: ProtocolTool | str) -> str:
        selected = _enum(tool, ProtocolTool, "protocol tool")
        return (
            self.tamarin_name
            if selected is ProtocolTool.TAMARIN
            else self.proverif_label
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "protocol_verification_version": PROTOCOL_VERIFICATION_VERSION,
            "query_id": self.query_id,
            "property": self.property,
            "kind": self.kind,
            "abstraction": self.abstraction,
            "assumptions": self.assumptions,
            "excluded_behaviors": self.excluded_behaviors,
            "tamarin_name": self.tamarin_name,
            "proverif_label": self.proverif_label,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProtocolQuery":
        if not isinstance(payload, Mapping):
            raise ProtocolValidationError("protocol query must be an object")
        _schema(payload, cls.SCHEMA)
        result = cls(
            query_id=payload.get("query_id", ""),
            property=payload.get("property", ""),
            kind=payload.get("kind", ""),
            abstraction=payload.get("abstraction", ""),
            assumptions=tuple(payload.get("assumptions") or ()),
            excluded_behaviors=tuple(payload.get("excluded_behaviors") or ()),
            tamarin_name=payload.get("tamarin_name", ""),
            proverif_label=payload.get("proverif_label", ""),
            required=payload.get("required", True),
        )
        _claimed_identity(payload, result.content_id, "protocol query")
        return result


@dataclass(frozen=True)
class ProtocolModel(CanonicalContract):
    """Exact paired Tamarin/ProVerif model with reviewed queries."""

    SCHEMA = PROTOCOL_MODEL_SCHEMA

    model_id: str
    version: str
    description: str
    properties: tuple[ProtocolProperty, ...]
    queries: tuple[ProtocolQuery, ...]
    tamarin_source: str
    proverif_source: str
    protocol_steps: tuple[str, ...]
    optional_attestation: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, "model_id"))
        object.__setattr__(self, "version", _text(self.version, "version"))
        object.__setattr__(
            self, "description", _text(self.description, "description")
        )
        try:
            properties = tuple(
                sorted(
                    {
                        _enum(item, ProtocolProperty, "protocol property")
                        for item in self.properties
                    },
                    key=lambda item: item.value,
                )
            )
        except TypeError as exc:
            raise ProtocolValidationError("properties must be a sequence") from exc
        if not properties:
            raise ProtocolValidationError("properties must not be empty")
        object.__setattr__(self, "properties", properties)
        queries = tuple(self.queries)
        if not queries or any(not isinstance(item, ProtocolQuery) for item in queries):
            raise ProtocolValidationError(
                "queries must contain at least one ProtocolQuery"
            )
        query_ids = tuple(item.query_id for item in queries)
        if len(query_ids) != len(set(query_ids)):
            raise ProtocolValidationError("query ids must be unique")
        if any(item.property not in properties for item in queries):
            raise ProtocolValidationError("query refers to an undeclared property")
        uncovered = set(properties).difference(item.property for item in queries)
        if uncovered:
            raise ProtocolValidationError("every protocol property requires a query")
        object.__setattr__(self, "queries", queries)
        tamarin = _source(self.tamarin_source, "tamarin_source")
        proverif = _source(self.proverif_source, "proverif_source")
        for query in queries:
            if query.tamarin_name not in tamarin:
                raise ProtocolValidationError(
                    f"Tamarin source omits query {query.query_id}"
                )
            if query.proverif_label not in proverif:
                raise ProtocolValidationError(
                    f"ProVerif source omits query {query.query_id}"
                )
        object.__setattr__(self, "tamarin_source", tamarin)
        object.__setattr__(self, "proverif_source", proverif)
        object.__setattr__(
            self,
            "protocol_steps",
            _strings(
                self.protocol_steps,
                "protocol_steps",
                required=True,
                preserve_order=True,
            ),
        )
        if not isinstance(self.optional_attestation, bool):
            raise ProtocolValidationError("optional_attestation must be a boolean")
        contains_attestation = (
            ProtocolProperty.ATTESTATION_EXCHANGE in properties
        )
        if contains_attestation != self.optional_attestation:
            raise ProtocolValidationError(
                "attestation property and optional_attestation must agree"
            )

    @property
    def query_set_identity(self) -> str:
        return content_identity(
            {"query_ids": [item.content_id for item in self.queries]}
        )

    def source_for(self, tool: ProtocolTool | str) -> str:
        selected = _enum(tool, ProtocolTool, "protocol tool")
        return (
            self.tamarin_source
            if selected is ProtocolTool.TAMARIN
            else self.proverif_source
        )

    def source_identity_for(self, tool: ProtocolTool | str) -> str:
        return _digest(self.source_for(tool))

    def _payload(self) -> dict[str, Any]:
        return {
            "protocol_verification_version": PROTOCOL_VERIFICATION_VERSION,
            "model_id": self.model_id,
            "version": self.version,
            "description": self.description,
            "properties": self.properties,
            "queries": self.queries,
            "query_set_identity": self.query_set_identity,
            "tamarin_source": self.tamarin_source,
            "tamarin_source_identity": self.source_identity_for(
                ProtocolTool.TAMARIN
            ),
            "proverif_source": self.proverif_source,
            "proverif_source_identity": self.source_identity_for(
                ProtocolTool.PROVERIF
            ),
            "protocol_steps": self.protocol_steps,
            "optional_attestation": self.optional_attestation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProtocolModel":
        if not isinstance(payload, Mapping):
            raise ProtocolValidationError("protocol model must be an object")
        _schema(payload, cls.SCHEMA)
        result = cls(
            model_id=payload.get("model_id", ""),
            version=payload.get("version", ""),
            description=payload.get("description", ""),
            properties=tuple(payload.get("properties") or ()),
            queries=tuple(
                ProtocolQuery.from_dict(item)
                for item in (payload.get("queries") or ())
                if isinstance(item, Mapping)
            ),
            tamarin_source=payload.get("tamarin_source", ""),
            proverif_source=payload.get("proverif_source", ""),
            protocol_steps=tuple(payload.get("protocol_steps") or ()),
            optional_attestation=payload.get("optional_attestation", False),
        )
        bindings = (
            ("query_set_identity", result.query_set_identity),
            (
                "tamarin_source_identity",
                result.source_identity_for(ProtocolTool.TAMARIN),
            ),
            (
                "proverif_source_identity",
                result.source_identity_for(ProtocolTool.PROVERIF),
            ),
        )
        for name, actual in bindings:
            claimed = payload.get(name)
            if claimed and claimed != actual:
                raise ProtocolValidationError(f"model {name} does not match source")
        _claimed_identity(payload, result.content_id, "protocol model")
        return result


@dataclass(frozen=True)
class ProtocolConformanceFixture(CanonicalContract):
    """An end-to-end positive and negative semantic fixture."""

    SCHEMA = PROTOCOL_FIXTURE_SCHEMA

    tool: ProtocolTool
    fixture_id: str
    model_source: str
    file_name: str
    args: tuple[str, ...]
    safe_markers: tuple[str, ...]
    attack_marker: str
    query_kinds: tuple[ProtocolQueryKind, ...]
    translator_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tool", _enum(self.tool, ProtocolTool, "protocol tool")
        )
        object.__setattr__(self, "fixture_id", _text(self.fixture_id, "fixture_id"))
        source = _source(self.model_source, "fixture source")
        object.__setattr__(self, "model_source", source)
        file_name = _text(self.file_name, "file_name")
        if Path(file_name).name != file_name:
            raise ProtocolValidationError("fixture file_name must be a basename")
        object.__setattr__(self, "file_name", file_name)
        args = _strings(self.args, "args", required=True, preserve_order=True)
        if "{fixture}" not in args:
            raise ProtocolValidationError("fixture args must contain {fixture}")
        object.__setattr__(self, "args", args)
        object.__setattr__(
            self,
            "safe_markers",
            _strings(
                self.safe_markers,
                "safe_markers",
                required=True,
                preserve_order=True,
            ),
        )
        object.__setattr__(
            self, "attack_marker", _text(self.attack_marker, "attack_marker")
        )
        try:
            kinds = tuple(
                sorted(
                    {
                        _enum(item, ProtocolQueryKind, "query kind")
                        for item in self.query_kinds
                    },
                    key=lambda item: item.value,
                )
            )
        except TypeError as exc:
            raise ProtocolValidationError("query_kinds must be a sequence") from exc
        if set(kinds) != set(ProtocolQueryKind):
            raise ProtocolValidationError(
                "end-to-end fixture must exercise every protocol query kind"
            )
        object.__setattr__(self, "query_kinds", kinds)
        object.__setattr__(
            self, "translator_id", _text(self.translator_id, "translator_id")
        )

    @property
    def source_identity(self) -> str:
        return _digest(self.model_source)

    def _payload(self) -> dict[str, Any]:
        return {
            "protocol_verification_version": PROTOCOL_VERIFICATION_VERSION,
            "tool": self.tool,
            "fixture_id": self.fixture_id,
            "model_source": self.model_source,
            "source_identity": self.source_identity,
            "file_name": self.file_name,
            "args": self.args,
            "safe_markers": self.safe_markers,
            "attack_marker": self.attack_marker,
            "query_kinds": self.query_kinds,
            "translator_id": self.translator_id,
            "end_to_end": True,
            "contains_known_attack": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProtocolConformanceFixture":
        if not isinstance(payload, Mapping):
            raise ProtocolValidationError("protocol fixture must be an object")
        _schema(payload, cls.SCHEMA)
        if payload.get("end_to_end") not in (None, True) or payload.get(
            "contains_known_attack"
        ) not in (None, True):
            raise ProtocolValidationError(
                "protocol fixture must contain end-to-end positive and negative cases"
            )
        result = cls(
            tool=payload.get("tool", ""),
            fixture_id=payload.get("fixture_id", ""),
            model_source=payload.get("model_source", ""),
            file_name=payload.get("file_name", ""),
            args=tuple(payload.get("args") or ()),
            safe_markers=tuple(payload.get("safe_markers") or ()),
            attack_marker=payload.get("attack_marker", ""),
            query_kinds=tuple(payload.get("query_kinds") or ()),
            translator_id=payload.get("translator_id", ""),
        )
        claimed_source = payload.get("source_identity")
        if claimed_source and claimed_source != result.source_identity:
            raise ProtocolValidationError(
                "fixture source_identity does not match model source"
            )
        _claimed_identity(payload, result.content_id, "protocol fixture")
        return result


@dataclass(frozen=True)
class ProtocolConformanceReceipt(CanonicalContract):
    """Exact execution receipt for an end-to-end engine fixture."""

    SCHEMA = PROTOCOL_CONFORMANCE_RECEIPT_SCHEMA

    tool: ProtocolTool
    status: ConformanceStatus
    fixture_id: str
    fixture_identity: str
    fixture_source_identity: str
    executable_path: str
    executable_identity: str
    executable_version: str
    command: tuple[str, ...]
    command_identity: str
    returncode: int | None
    timed_out: bool
    output_sha256: str
    output_truncated: bool
    safe_markers_matched: tuple[str, ...]
    attack_marker_matched: bool
    duration_ms: int
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tool", _enum(self.tool, ProtocolTool, "protocol tool")
        )
        object.__setattr__(
            self,
            "status",
            _enum(self.status, ConformanceStatus, "conformance status"),
        )
        for name in ("fixture_id", "executable_path", "executable_version", "reason"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "fixture_identity",
            "fixture_source_identity",
            "executable_identity",
            "command_identity",
            "output_sha256",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        command = _strings(
            self.command, "command", required=True, preserve_order=True
        )
        object.__setattr__(self, "command", command)
        if self.returncode is not None and (
            isinstance(self.returncode, bool) or not isinstance(self.returncode, int)
        ):
            raise ProtocolValidationError("returncode must be an integer or null")
        if not isinstance(self.timed_out, bool) or not isinstance(
            self.output_truncated, bool
        ):
            raise ProtocolValidationError("receipt flags must be booleans")
        object.__setattr__(
            self,
            "safe_markers_matched",
            _strings(
                self.safe_markers_matched,
                "safe_markers_matched",
                preserve_order=True,
            ),
        )
        if not isinstance(self.attack_marker_matched, bool):
            raise ProtocolValidationError("attack_marker_matched must be a boolean")
        object.__setattr__(
            self, "duration_ms", _nonnegative(self.duration_ms, "duration_ms")
        )
        if self.command_identity != content_identity({"command": command}):
            raise ProtocolValidationError("command_identity does not match command")
        if self.status is ConformanceStatus.PASSED and (
            self.returncode != 0
            or self.timed_out
            or self.output_truncated
            or len(self.safe_markers_matched) != len(ProtocolQueryKind)
            or not self.attack_marker_matched
        ):
            raise ProtocolValidationError(
                "passing conformance requires complete end-to-end execution"
            )

    @property
    def passed(self) -> bool:
        return self.status is ConformanceStatus.PASSED

    def _payload(self) -> dict[str, Any]:
        return {
            "protocol_verification_version": PROTOCOL_VERIFICATION_VERSION,
            "tool": self.tool,
            "status": self.status,
            "fixture_id": self.fixture_id,
            "fixture_identity": self.fixture_identity,
            "fixture_source_identity": self.fixture_source_identity,
            "executable_path": self.executable_path,
            "executable_identity": self.executable_identity,
            "executable_version": self.executable_version,
            "command": self.command,
            "command_identity": self.command_identity,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "output_sha256": self.output_sha256,
            "output_truncated": self.output_truncated,
            "safe_markers_matched": self.safe_markers_matched,
            "attack_marker_matched": self.attack_marker_matched,
            "duration_ms": self.duration_ms,
            "reason": self.reason,
            "end_to_end_fixture_executed": True,
            "discovery_is_conformance": False,
            "installer_success_is_conformance": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProtocolConformanceReceipt":
        if not isinstance(payload, Mapping):
            raise ProtocolValidationError("conformance receipt must be an object")
        _schema(payload, cls.SCHEMA)
        if payload.get("end_to_end_fixture_executed") not in (None, True):
            raise ProtocolValidationError("receipt did not execute an end-to-end fixture")
        result = cls(
            tool=payload.get("tool", ""),
            status=payload.get("status", ""),
            fixture_id=payload.get("fixture_id", ""),
            fixture_identity=payload.get("fixture_identity", ""),
            fixture_source_identity=payload.get("fixture_source_identity", ""),
            executable_path=payload.get("executable_path", ""),
            executable_identity=payload.get("executable_identity", ""),
            executable_version=payload.get("executable_version", ""),
            command=tuple(payload.get("command") or ()),
            command_identity=payload.get("command_identity", ""),
            returncode=payload.get("returncode"),
            timed_out=payload.get("timed_out", False),
            output_sha256=payload.get("output_sha256", ""),
            output_truncated=payload.get("output_truncated", False),
            safe_markers_matched=tuple(
                payload.get("safe_markers_matched") or ()
            ),
            attack_marker_matched=payload.get("attack_marker_matched", False),
            duration_ms=payload.get("duration_ms", 0),
            reason=payload.get("reason", ""),
        )
        _claimed_identity(payload, result.content_id, "conformance receipt")
        return result


@dataclass(frozen=True)
class ProtocolToolCapability(CanonicalContract):
    """Capability which cannot be conformant without a passing fixture."""

    SCHEMA = PROTOCOL_TOOL_CAPABILITY_SCHEMA

    tool: ProtocolTool
    status: ToolCapabilityStatus
    reason: str
    executable_path: str | None = None
    executable_identity: str | None = None
    executable_version: str | None = None
    conformance_receipt: ProtocolConformanceReceipt | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tool", _enum(self.tool, ProtocolTool, "protocol tool")
        )
        object.__setattr__(
            self,
            "status",
            _enum(self.status, ToolCapabilityStatus, "capability status"),
        )
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        for name in ("executable_path", "executable_version"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name))
        if self.executable_identity is not None:
            object.__setattr__(
                self,
                "executable_identity",
                _identity(self.executable_identity, "executable_identity"),
            )
        receipt = self.conformance_receipt
        if isinstance(receipt, Mapping):
            receipt = ProtocolConformanceReceipt.from_dict(receipt)
            object.__setattr__(self, "conformance_receipt", receipt)
        if receipt is not None and receipt.tool is not self.tool:
            raise ProtocolValidationError("capability receipt belongs to another tool")
        if self.status is ToolCapabilityStatus.CONFORMANT:
            if (
                receipt is None
                or not receipt.passed
                or self.executable_path != receipt.executable_path
                or self.executable_identity != receipt.executable_identity
                or self.executable_version != receipt.executable_version
            ):
                raise ProtocolValidationError(
                    "conformant capability requires an exact passing fixture receipt"
                )

    @property
    def available(self) -> bool:
        return self.status is ToolCapabilityStatus.CONFORMANT

    @property
    def conformance_passed(self) -> bool:
        return bool(self.conformance_receipt and self.conformance_receipt.passed)

    def _payload(self) -> dict[str, Any]:
        return {
            "protocol_verification_version": PROTOCOL_VERIFICATION_VERSION,
            "tool": self.tool,
            "status": self.status,
            "reason": self.reason,
            "executable_path": self.executable_path,
            "executable_identity": self.executable_identity,
            "executable_version": self.executable_version,
            "conformance_receipt": self.conformance_receipt,
            "available": self.available,
            "conformance_passed": self.conformance_passed,
            "discovery_is_conformance": False,
            "installer_success_is_conformance": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProtocolToolCapability":
        if not isinstance(payload, Mapping):
            raise ProtocolValidationError("tool capability must be an object")
        _schema(payload, cls.SCHEMA)
        raw_receipt = payload.get("conformance_receipt")
        result = cls(
            tool=payload.get("tool", ""),
            status=payload.get("status", ""),
            reason=payload.get("reason", ""),
            executable_path=payload.get("executable_path"),
            executable_identity=payload.get("executable_identity"),
            executable_version=payload.get("executable_version"),
            conformance_receipt=(
                ProtocolConformanceReceipt.from_dict(raw_receipt)
                if isinstance(raw_receipt, Mapping)
                else None
            ),
        )
        if payload.get("available") not in (None, result.available):
            raise ProtocolValidationError("capability availability was forged")
        if payload.get("conformance_passed") not in (
            None,
            result.conformance_passed,
        ):
            raise ProtocolValidationError("capability conformance was forged")
        _claimed_identity(payload, result.content_id, "tool capability")
        return result


@dataclass(frozen=True)
class ProtocolAttackStep(CanonicalContract):
    """One sanitized, deterministic symbolic attack step."""

    SCHEMA = ""

    index: int
    event: str
    principal_role: str
    message_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _nonnegative(self.index, "index"))
        for name in ("event", "principal_role"):
            value = re.sub(
                r"[^a-z0-9_.:-]+",
                "_",
                _text(getattr(self, name), name).casefold(),
            ).strip("_")
            object.__setattr__(self, name, value or "unknown")
        object.__setattr__(
            self, "message_ref", _identity(self.message_ref, "message_ref")
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "event": self.event,
            "principal_role": self.principal_role,
            "message_ref": self.message_ref,
        }

    def to_dict(self) -> dict[str, Any]:
        return self._payload()


@dataclass(frozen=True)
class ProtocolAttackCounterexample(CanonicalContract):
    """Canonical counterexample with no raw solver transcript."""

    SCHEMA = PROTOCOL_ATTACK_COUNTEREXAMPLE_SCHEMA

    model_id: str
    model_identity: str
    tool: ProtocolTool
    query_id: str
    query_identity: str
    abstraction_identity: str
    source_output_sha256: str
    steps: tuple[ProtocolAttackStep, ...]
    minimized: bool = True
    redacted: bool = True

    def __post_init__(self) -> None:
        for name in ("model_id", "query_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "model_identity",
            "query_identity",
            "abstraction_identity",
            "source_output_sha256",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        object.__setattr__(
            self, "tool", _enum(self.tool, ProtocolTool, "protocol tool")
        )
        steps = tuple(self.steps)
        if not steps or any(not isinstance(item, ProtocolAttackStep) for item in steps):
            raise ProtocolValidationError("counterexample requires attack steps")
        ordered = tuple(sorted(steps, key=lambda item: (item.index, item.content_id)))
        if len(ordered) > 32:
            raise ProtocolValidationError("counterexample exceeds the canonical bound")
        object.__setattr__(self, "steps", ordered)
        if self.minimized is not True or self.redacted is not True:
            raise ProtocolValidationError(
                "protocol counterexamples must be minimized and redacted"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "protocol_verification_version": PROTOCOL_VERIFICATION_VERSION,
            "model_id": self.model_id,
            "model_identity": self.model_identity,
            "tool": self.tool,
            "query_id": self.query_id,
            "query_identity": self.query_identity,
            "abstraction_identity": self.abstraction_identity,
            "source_output_sha256": self.source_output_sha256,
            "steps": self.steps,
            "minimized": True,
            "redacted": True,
            "contains_raw_transcript": False,
            "contains_protocol_secrets": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProtocolAttackCounterexample":
        if not isinstance(payload, Mapping):
            raise ProtocolValidationError("counterexample must be an object")
        _schema(payload, cls.SCHEMA)
        if payload.get("contains_raw_transcript") not in (None, False):
            raise ProtocolValidationError("raw attack transcripts are forbidden")
        result = cls(
            model_id=payload.get("model_id", ""),
            model_identity=payload.get("model_identity", ""),
            tool=payload.get("tool", ""),
            query_id=payload.get("query_id", ""),
            query_identity=payload.get("query_identity", ""),
            abstraction_identity=payload.get("abstraction_identity", ""),
            source_output_sha256=payload.get("source_output_sha256", ""),
            steps=tuple(
                ProtocolAttackStep(
                    index=item.get("index", 0),
                    event=item.get("event", ""),
                    principal_role=item.get("principal_role", ""),
                    message_ref=item.get("message_ref", ""),
                )
                for item in (payload.get("steps") or ())
                if isinstance(item, Mapping)
            ),
            minimized=payload.get("minimized", False),
            redacted=payload.get("redacted", False),
        )
        _claimed_identity(payload, result.content_id, "protocol counterexample")
        return result


@dataclass(frozen=True)
class ProtocolQueryResult(CanonicalContract):
    SCHEMA = PROTOCOL_QUERY_RESULT_SCHEMA

    query_id: str
    query_identity: str
    property: ProtocolProperty
    kind: ProtocolQueryKind
    verdict: ProtocolVerdict
    abstraction_identity: str
    reason: str
    counterexample: ProtocolAttackCounterexample | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _text(self.query_id, "query_id"))
        object.__setattr__(
            self, "query_identity", _identity(self.query_identity, "query_identity")
        )
        object.__setattr__(
            self, "property", _enum(self.property, ProtocolProperty, "property")
        )
        object.__setattr__(
            self, "kind", _enum(self.kind, ProtocolQueryKind, "query kind")
        )
        object.__setattr__(
            self, "verdict", _enum(self.verdict, ProtocolVerdict, "verdict")
        )
        object.__setattr__(
            self,
            "abstraction_identity",
            _identity(self.abstraction_identity, "abstraction_identity"),
        )
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        if self.counterexample is not None:
            if self.verdict is not ProtocolVerdict.VIOLATED:
                raise ProtocolValidationError(
                    "only a violated query may contain a counterexample"
                )
            if (
                self.counterexample.query_id != self.query_id
                or self.counterexample.query_identity != self.query_identity
                or self.counterexample.abstraction_identity
                != self.abstraction_identity
            ):
                raise ProtocolValidationError(
                    "counterexample does not bind the exact query abstraction"
                )
        elif self.verdict is ProtocolVerdict.VIOLATED:
            raise ProtocolValidationError(
                "a violated protocol query requires a counterexample"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "protocol_verification_version": PROTOCOL_VERIFICATION_VERSION,
            "query_id": self.query_id,
            "query_identity": self.query_identity,
            "property": self.property,
            "kind": self.kind,
            "verdict": self.verdict,
            "abstraction_identity": self.abstraction_identity,
            "reason": self.reason,
            "counterexample": self.counterexample,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProtocolQueryResult":
        if not isinstance(payload, Mapping):
            raise ProtocolValidationError("query result must be an object")
        _schema(payload, cls.SCHEMA)
        raw = payload.get("counterexample")
        result = cls(
            query_id=payload.get("query_id", ""),
            query_identity=payload.get("query_identity", ""),
            property=payload.get("property", ""),
            kind=payload.get("kind", ""),
            verdict=payload.get("verdict", ""),
            abstraction_identity=payload.get("abstraction_identity", ""),
            reason=payload.get("reason", ""),
            counterexample=(
                ProtocolAttackCounterexample.from_dict(raw)
                if isinstance(raw, Mapping)
                else None
            ),
        )
        _claimed_identity(payload, result.content_id, "protocol query result")
        return result


@dataclass(frozen=True)
class ProtocolToolchainReceipt(CanonicalContract):
    """Exact model and toolchain binding for one full model run."""

    SCHEMA = PROTOCOL_TOOLCHAIN_RECEIPT_SCHEMA

    tool: ProtocolTool
    status: ToolRunStatus
    model_id: str
    model_identity: str
    model_source_identity: str
    query_set_identity: str
    capability_identity: str
    conformance_receipt_identity: str
    executable_path: str
    executable_identity: str
    executable_version: str
    command: tuple[str, ...]
    command_identity: str
    returncode: int | None
    timed_out: bool
    output_truncated: bool
    stdout_sha256: str
    stderr_sha256: str
    duration_ms: int
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tool", _enum(self.tool, ProtocolTool, "protocol tool")
        )
        object.__setattr__(
            self, "status", _enum(self.status, ToolRunStatus, "tool-run status")
        )
        for name in (
            "model_id",
            "executable_path",
            "executable_version",
            "reason",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "model_identity",
            "model_source_identity",
            "query_set_identity",
            "capability_identity",
            "conformance_receipt_identity",
            "executable_identity",
            "command_identity",
            "stdout_sha256",
            "stderr_sha256",
        ):
            object.__setattr__(self, name, _identity(getattr(self, name), name))
        command = _strings(
            self.command, "command", required=True, preserve_order=True
        )
        object.__setattr__(self, "command", command)
        if self.command_identity != content_identity({"command": command}):
            raise ProtocolValidationError("toolchain command binding is invalid")
        if self.returncode is not None and (
            isinstance(self.returncode, bool) or not isinstance(self.returncode, int)
        ):
            raise ProtocolValidationError("returncode must be an integer or null")
        if not isinstance(self.timed_out, bool) or not isinstance(
            self.output_truncated, bool
        ):
            raise ProtocolValidationError("toolchain flags must be booleans")
        object.__setattr__(
            self, "duration_ms", _nonnegative(self.duration_ms, "duration_ms")
        )
        if self.status in {ToolRunStatus.PASSED, ToolRunStatus.VIOLATED} and (
            self.returncode != 0 or self.timed_out or self.output_truncated
        ):
            raise ProtocolValidationError(
                "semantic tool result requires complete successful execution"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "protocol_verification_version": PROTOCOL_VERIFICATION_VERSION,
            "tool": self.tool,
            "status": self.status,
            "model_id": self.model_id,
            "model_identity": self.model_identity,
            "model_source_identity": self.model_source_identity,
            "query_set_identity": self.query_set_identity,
            "capability_identity": self.capability_identity,
            "conformance_receipt_identity": self.conformance_receipt_identity,
            "executable_path": self.executable_path,
            "executable_identity": self.executable_identity,
            "executable_version": self.executable_version,
            "command": self.command,
            "command_identity": self.command_identity,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "output_truncated": self.output_truncated,
            "stdout_sha256": self.stdout_sha256,
            "stderr_sha256": self.stderr_sha256,
            "duration_ms": self.duration_ms,
            "reason": self.reason,
            "exact_model_bound": True,
            "exact_toolchain_bound": True,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProtocolToolchainReceipt":
        if not isinstance(payload, Mapping):
            raise ProtocolValidationError("toolchain receipt must be an object")
        _schema(payload, cls.SCHEMA)
        result = cls(
            tool=payload.get("tool", ""),
            status=payload.get("status", ""),
            model_id=payload.get("model_id", ""),
            model_identity=payload.get("model_identity", ""),
            model_source_identity=payload.get("model_source_identity", ""),
            query_set_identity=payload.get("query_set_identity", ""),
            capability_identity=payload.get("capability_identity", ""),
            conformance_receipt_identity=payload.get(
                "conformance_receipt_identity", ""
            ),
            executable_path=payload.get("executable_path", ""),
            executable_identity=payload.get("executable_identity", ""),
            executable_version=payload.get("executable_version", ""),
            command=tuple(payload.get("command") or ()),
            command_identity=payload.get("command_identity", ""),
            returncode=payload.get("returncode"),
            timed_out=payload.get("timed_out", False),
            output_truncated=payload.get("output_truncated", False),
            stdout_sha256=payload.get("stdout_sha256", ""),
            stderr_sha256=payload.get("stderr_sha256", ""),
            duration_ms=payload.get("duration_ms", 0),
            reason=payload.get("reason", ""),
        )
        _claimed_identity(payload, result.content_id, "toolchain receipt")
        return result


@dataclass(frozen=True)
class ProtocolLaneResult(CanonicalContract):
    """One engine's authoritative or unavailable result for an exact model."""

    SCHEMA = PROTOCOL_LANE_RESULT_SCHEMA

    model_id: str
    model_identity: str
    tool: ProtocolTool
    verdict: ProtocolVerdict
    authoritative: bool
    reason: str
    query_results: tuple[ProtocolQueryResult, ...]
    capability_identity: str
    toolchain_receipt: ProtocolToolchainReceipt | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, "model_id"))
        object.__setattr__(
            self, "model_identity", _identity(self.model_identity, "model_identity")
        )
        object.__setattr__(
            self, "tool", _enum(self.tool, ProtocolTool, "protocol tool")
        )
        object.__setattr__(
            self, "verdict", _enum(self.verdict, ProtocolVerdict, "verdict")
        )
        if not isinstance(self.authoritative, bool):
            raise ProtocolValidationError("authoritative must be a boolean")
        object.__setattr__(self, "reason", _text(self.reason, "reason"))
        results = tuple(self.query_results)
        if any(not isinstance(item, ProtocolQueryResult) for item in results):
            raise ProtocolValidationError(
                "query_results must contain ProtocolQueryResult values"
            )
        if len({item.query_id for item in results}) != len(results):
            raise ProtocolValidationError("query results must be unique")
        object.__setattr__(self, "query_results", results)
        object.__setattr__(
            self,
            "capability_identity",
            _identity(self.capability_identity, "capability_identity"),
        )
        if self.authoritative:
            if self.toolchain_receipt is None:
                raise ProtocolValidationError(
                    "authoritative lane requires a toolchain receipt"
                )
            if (
                self.toolchain_receipt.model_identity != self.model_identity
                or self.toolchain_receipt.tool is not self.tool
                or self.toolchain_receipt.capability_identity
                != self.capability_identity
            ):
                raise ProtocolValidationError(
                    "lane does not bind its exact model, tool, and capability"
                )
        elif self.toolchain_receipt is not None:
            raise ProtocolValidationError(
                "non-authoritative lane cannot retain a toolchain receipt"
            )

    def _payload(self) -> dict[str, Any]:
        return {
            "protocol_verification_version": PROTOCOL_VERIFICATION_VERSION,
            "model_id": self.model_id,
            "model_identity": self.model_identity,
            "tool": self.tool,
            "verdict": self.verdict,
            "authoritative": self.authoritative,
            "reason": self.reason,
            "query_results": self.query_results,
            "capability_identity": self.capability_identity,
            "toolchain_receipt": self.toolchain_receipt,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProtocolLaneResult":
        if not isinstance(payload, Mapping):
            raise ProtocolValidationError("lane result must be an object")
        _schema(payload, cls.SCHEMA)
        raw = payload.get("toolchain_receipt")
        result = cls(
            model_id=payload.get("model_id", ""),
            model_identity=payload.get("model_identity", ""),
            tool=payload.get("tool", ""),
            verdict=payload.get("verdict", ""),
            authoritative=payload.get("authoritative", False),
            reason=payload.get("reason", ""),
            query_results=tuple(
                ProtocolQueryResult.from_dict(item)
                for item in (payload.get("query_results") or ())
                if isinstance(item, Mapping)
            ),
            capability_identity=payload.get("capability_identity", ""),
            toolchain_receipt=(
                ProtocolToolchainReceipt.from_dict(raw)
                if isinstance(raw, Mapping)
                else None
            ),
        )
        _claimed_identity(payload, result.content_id, "protocol lane result")
        return result


@dataclass(frozen=True)
class ProtocolSuiteResult(CanonicalContract):
    """Both symbolic lanes for one exact protocol model."""

    SCHEMA = PROTOCOL_SUITE_RESULT_SCHEMA

    model_id: str
    model_identity: str
    lane_results: tuple[ProtocolLaneResult, ...]
    verdict: ProtocolVerdict
    complete: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, "model_id"))
        object.__setattr__(
            self, "model_identity", _identity(self.model_identity, "model_identity")
        )
        lanes = tuple(self.lane_results)
        if any(not isinstance(item, ProtocolLaneResult) for item in lanes):
            raise ProtocolValidationError(
                "lane_results must contain ProtocolLaneResult values"
            )
        if len({item.tool for item in lanes}) != len(lanes):
            raise ProtocolValidationError("suite lanes must be unique")
        if any(
            item.model_id != self.model_id
            or item.model_identity != self.model_identity
            for item in lanes
        ):
            raise ProtocolValidationError("suite contains a different model")
        object.__setattr__(self, "lane_results", lanes)
        object.__setattr__(
            self, "verdict", _enum(self.verdict, ProtocolVerdict, "verdict")
        )
        if not isinstance(self.complete, bool):
            raise ProtocolValidationError("complete must be a boolean")
        actual_complete = (
            {item.tool for item in lanes} == set(ProtocolTool)
            and all(item.authoritative for item in lanes)
        )
        if self.complete != actual_complete:
            raise ProtocolValidationError("suite completeness is inconsistent")

    def _payload(self) -> dict[str, Any]:
        return {
            "protocol_verification_version": PROTOCOL_VERIFICATION_VERSION,
            "model_id": self.model_id,
            "model_identity": self.model_identity,
            "lane_results": self.lane_results,
            "verdict": self.verdict,
            "complete": self.complete,
            "required_tools": tuple(ProtocolTool),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProtocolSuiteResult":
        if not isinstance(payload, Mapping):
            raise ProtocolValidationError("suite result must be an object")
        _schema(payload, cls.SCHEMA)
        result = cls(
            model_id=payload.get("model_id", ""),
            model_identity=payload.get("model_identity", ""),
            lane_results=tuple(
                ProtocolLaneResult.from_dict(item)
                for item in (payload.get("lane_results") or ())
                if isinstance(item, Mapping)
            ),
            verdict=payload.get("verdict", ""),
            complete=payload.get("complete", False),
        )
        _claimed_identity(payload, result.content_id, "protocol suite result")
        return result


CommandRunner = Callable[[CommandRequest], CommandResult | Mapping[str, Any]]
ExecutableFinder = Callable[[str], str | None]


def _command_result(value: CommandResult | Mapping[str, Any]) -> CommandResult:
    if isinstance(value, CommandResult):
        return value
    if not isinstance(value, Mapping):
        return CommandResult(returncode=None, error="malformed runner result")
    return CommandResult(
        returncode=value.get("returncode"),
        stdout=str(value.get("stdout") or ""),
        stderr=str(value.get("stderr") or ""),
        timed_out=bool(value.get("timed_out", False)),
        error=(str(value["error"]) if value.get("error") else None),
        output_truncated=bool(value.get("output_truncated", False)),
    )


def _executable_identity(path: Path, maximum_bytes: int) -> str | None:
    try:
        if not path.is_file() or path.stat().st_size > maximum_bytes:
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(64 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except OSError:
        return None


def _abstraction_identity(query: ProtocolQuery) -> str:
    return content_identity(
        {
            "query_id": query.query_id,
            "abstraction": query.abstraction,
            "assumptions": query.assumptions,
            "excluded_behaviors": query.excluded_behaviors,
        }
    )


def canonicalize_attack_trace(
    model: ProtocolModel,
    tool: ProtocolTool | str,
    query: ProtocolQuery,
    output: str,
) -> ProtocolAttackCounterexample:
    """Reduce raw solver output to a bounded, redacted canonical attack."""

    if not isinstance(model, ProtocolModel) or not isinstance(query, ProtocolQuery):
        raise ProtocolValidationError("model and query must be reviewed contracts")
    selected = _enum(tool, ProtocolTool, "protocol tool")
    if query not in model.queries:
        raise ProtocolValidationError("query does not belong to model")
    raw = str(output)
    steps: list[ProtocolAttackStep] = []
    for line in raw.splitlines():
        match = _TRACE_LINE_RE.match(line)
        if not match or match.group("query") != query.query_id:
            continue
        message = match.group("message") or "<no-message>"
        steps.append(
            ProtocolAttackStep(
                index=int(match.group("step")),
                event=match.group("event"),
                principal_role=match.group("actor") or "network",
                message_ref=_digest(message),
            )
        )
        if len(steps) >= 32:
            break
    if not steps:
        # A violated result must be actionable even when a tool does not emit a
        # machine-readable trace.  Bind the exact output without retaining it.
        steps = [
            ProtocolAttackStep(
                index=0,
                event=f"{query.kind.value}_violation",
                principal_role="symbolic_adversary",
                message_ref=_digest(raw),
            )
        ]
    # Remove duplicate symbolic states and compact indexes deterministically.
    unique: dict[tuple[str, str, str], ProtocolAttackStep] = {}
    for item in sorted(steps, key=lambda step: (step.index, step.content_id)):
        unique.setdefault(
            (item.event, item.principal_role, item.message_ref), item
        )
    minimized = tuple(
        ProtocolAttackStep(
            index=index,
            event=item.event,
            principal_role=item.principal_role,
            message_ref=item.message_ref,
        )
        for index, item in enumerate(unique.values())
    )
    return ProtocolAttackCounterexample(
        model_id=model.model_id,
        model_identity=model.content_id,
        tool=selected,
        query_id=query.query_id,
        query_identity=query.content_id,
        abstraction_identity=_abstraction_identity(query),
        source_output_sha256=_digest(raw),
        steps=minimized,
    )


def _query_outcomes(
    model: ProtocolModel,
    tool: ProtocolTool,
    output: str,
) -> tuple[ProtocolQueryResult, ...]:
    folded = output.casefold()
    proverif_results = re.findall(
        r"^\s*RESULT\s+(.+?)\s+is\s+"
        r"(true|false|cannot be proved|unknown)\.?\s*$",
        output,
        re.IGNORECASE | re.MULTILINE,
    )
    results: list[ProtocolQueryResult] = []
    for index, query in enumerate(model.queries):
        symbol = query.symbol_for(tool)
        escaped = re.escape(symbol.casefold())
        verdict = ProtocolVerdict.INCONCLUSIVE
        reason = "tool output did not contain an exact result for the query"
        if tool is ProtocolTool.TAMARIN:
            negative = re.search(
                rf"\b{escaped}\b[^\n]*(?:falsified|attack|counterexample)",
                folded,
            )
            positive = re.search(
                rf"\b{escaped}\b[^\n]*(?:verified|proved)",
                folded,
            )
            if negative:
                verdict = ProtocolVerdict.VIOLATED
                reason = "Tamarin produced an attack for the reviewed lemma"
            elif positive:
                verdict = ProtocolVerdict.VERIFIED
                reason = "Tamarin verified the reviewed lemma"
        else:
            # Prefer a label-bearing RESULT line.  ProVerif does not support
            # query names, so generated sources include stable labels and the
            # ordered RESULT fallback is bound by query_set_identity.
            matching = [
                state
                for body, state in proverif_results
                if symbol.casefold() in body.casefold()
                or query.query_id.casefold() in body.casefold()
            ]
            state = matching[0] if matching else (
                proverif_results[index][1]
                if index < len(proverif_results)
                else ""
            )
            if state.casefold() == "true":
                verdict = ProtocolVerdict.VERIFIED
                reason = "ProVerif proved the reviewed ordered query"
            elif state.casefold() == "false":
                verdict = ProtocolVerdict.VIOLATED
                reason = "ProVerif produced an attack for the reviewed ordered query"
            elif state:
                reason = "ProVerif could not prove the reviewed ordered query"
        counterexample = (
            canonicalize_attack_trace(model, tool, query, output)
            if verdict is ProtocolVerdict.VIOLATED
            else None
        )
        results.append(
            ProtocolQueryResult(
                query_id=query.query_id,
                query_identity=query.content_id,
                property=query.property,
                kind=query.kind,
                verdict=verdict,
                abstraction_identity=_abstraction_identity(query),
                reason=reason,
                counterexample=counterexample,
            )
        )
    return tuple(results)


class ProtocolToolAdapter:
    """Bounded adapter shared by Tamarin and ProVerif."""

    tool: ProtocolTool
    executable_candidates: tuple[str, ...]
    fixture: ProtocolConformanceFixture
    model_extension: str
    model_args: tuple[str, ...]
    version_args: tuple[str, ...] = ("--version",)

    def __init__(
        self,
        *,
        which: ExecutableFinder | None = None,
        command_runner: CommandRunner | None = None,
        timeout_seconds: float = DEFAULT_PROTOCOL_TIMEOUT_SECONDS,
        max_output_bytes: int = DEFAULT_MAX_PROTOCOL_OUTPUT_BYTES,
        max_executable_bytes: int = DEFAULT_MAX_EXECUTABLE_BYTES,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ProtocolValidationError("timeout_seconds must be positive")
        self.which = which or shutil.which
        self.command_runner = command_runner or _bounded_command_runner
        self.timeout_seconds = float(timeout_seconds)
        self.max_output_bytes = _positive(max_output_bytes, "max_output_bytes")
        self.max_executable_bytes = _positive(
            max_executable_bytes, "max_executable_bytes"
        )
        self.monotonic = monotonic or time.monotonic

    def _run(self, request: CommandRequest) -> CommandResult:
        try:
            return _command_result(self.command_runner(request))
        except BaseException as exc:
            return CommandResult(returncode=None, error=type(exc).__name__)

    def _discover(self) -> str | None:
        for candidate in self.executable_candidates:
            try:
                path = self.which(candidate)
            except BaseException:
                continue
            if path:
                return str(Path(path).resolve())
        return None

    def probe(self, *, run_conformance: bool = True) -> ProtocolToolCapability:
        executable = self._discover()
        if executable is None:
            return ProtocolToolCapability(
                tool=self.tool,
                status=ToolCapabilityStatus.UNAVAILABLE,
                reason="executable not discovered; no protocol claims were made",
            )
        executable_id = _executable_identity(
            Path(executable), self.max_executable_bytes
        )
        if executable_id is None:
            return ProtocolToolCapability(
                tool=self.tool,
                status=ToolCapabilityStatus.UNAVAILABLE,
                executable_path=executable,
                reason="executable could not be bounded and content-pinned",
            )
        version_result = self._run(
            CommandRequest(
                command=(executable,) + self.version_args,
                stdin_text=None,
                cwd=None,
                timeout_seconds=min(3.0, self.timeout_seconds),
                max_output_bytes=min(8192, self.max_output_bytes),
            )
        )
        version_text = (
            version_result.stdout + "\n" + version_result.stderr
        ).strip()
        if (
            version_result.returncode != 0
            or version_result.timed_out
            or version_result.output_truncated
            or not version_text
        ):
            return ProtocolToolCapability(
                tool=self.tool,
                status=ToolCapabilityStatus.UNAVAILABLE,
                executable_path=executable,
                executable_identity=executable_id,
                reason="executable did not provide a bounded version",
            )
        version = version_text.splitlines()[0][:256]
        if not run_conformance:
            return ProtocolToolCapability(
                tool=self.tool,
                status=ToolCapabilityStatus.UNAVAILABLE,
                executable_path=executable,
                executable_identity=executable_id,
                executable_version=version,
                reason=(
                    "executable presence is insufficient; end-to-end model "
                    "fixture was not run"
                ),
            )
        receipt = self._run_conformance(executable, executable_id, version)
        return ProtocolToolCapability(
            tool=self.tool,
            status=(
                ToolCapabilityStatus.CONFORMANT
                if receipt.passed
                else ToolCapabilityStatus.NONCONFORMANT
            ),
            executable_path=executable,
            executable_identity=executable_id,
            executable_version=version,
            conformance_receipt=receipt,
            reason=receipt.reason,
        )

    capability = probe
    probe_capability = probe
    check_capability = probe

    def _run_conformance(
        self, executable: str, executable_id: str, version: str
    ) -> ProtocolConformanceReceipt:
        started = self.monotonic()
        with tempfile.TemporaryDirectory(
            prefix=f"{self.tool.value}-protocol-fixture-"
        ) as raw:
            fixture_path = Path(raw) / self.fixture.file_name
            fixture_path.write_text(self.fixture.model_source, encoding="utf-8")
            command = (executable,) + tuple(
                str(fixture_path) if item == "{fixture}" else item
                for item in self.fixture.args
            )
            result = self._run(
                CommandRequest(
                    command=command,
                    stdin_text=None,
                    cwd=raw,
                    timeout_seconds=self.timeout_seconds,
                    max_output_bytes=self.max_output_bytes,
                )
            )
        elapsed = max(0, round((self.monotonic() - started) * 1000))
        output = result.stdout + "\n" + result.stderr
        folded = output.casefold()
        if self.tool is ProtocolTool.TAMARIN:
            def tamarin_marker_seen(marker: str) -> bool:
                name, _, state = marker.partition(":")
                return bool(
                    re.search(
                        rf"(?m)^\s*{re.escape(name)}\b[^\n]*"
                        rf"\b{re.escape(state.strip())}\b",
                        output,
                        re.IGNORECASE,
                    )
                )

            matched = tuple(
                marker
                for marker in self.fixture.safe_markers
                if tamarin_marker_seen(marker)
            )
            attack = tamarin_marker_seen(self.fixture.attack_marker)
        else:
            # ProVerif prints normalized queries, not source comments.  Their
            # reviewed source order is therefore the semantic binding.
            states = tuple(
                item.casefold()
                for item in re.findall(
                    r"(?m)^\s*RESULT\s+.+?\s+is\s+"
                    r"(true|false|cannot be proved|unknown)\.?\s*$",
                    output,
                    re.IGNORECASE,
                )
            )
            direct = tuple(
                marker
                for marker in self.fixture.safe_markers
                if marker.casefold() in folded
            )
            matched = (
                self.fixture.safe_markers
                if len(states) >= 5
                and all(state == "true" for state in states[:4])
                else direct
            )
            attack = (
                len(states) >= 5 and states[4] == "false"
            ) or self.fixture.attack_marker.casefold() in folded
        passed = (
            result.returncode == 0
            and not result.timed_out
            and not result.output_truncated
            and len(matched) == len(self.fixture.safe_markers)
            and attack
        )
        if passed:
            status = ConformanceStatus.PASSED
            reason = (
                "end-to-end model fixture proved all query classes and "
                "detected the known attack"
            )
        elif result.timed_out:
            status = ConformanceStatus.TIMED_OUT
            reason = "end-to-end model fixture timed out"
        elif result.error:
            status = ConformanceStatus.ERROR
            reason = "end-to-end model fixture could not execute"
        else:
            status = ConformanceStatus.FAILED
            reason = (
                "end-to-end model fixture did not establish safe-query and "
                "known-attack semantics"
            )
        return ProtocolConformanceReceipt(
            tool=self.tool,
            status=status,
            fixture_id=self.fixture.fixture_id,
            fixture_identity=self.fixture.content_id,
            fixture_source_identity=self.fixture.source_identity,
            executable_path=executable,
            executable_identity=executable_id,
            executable_version=version,
            command=command,
            command_identity=content_identity({"command": command}),
            returncode=result.returncode,
            timed_out=result.timed_out,
            output_sha256=_digest(output),
            output_truncated=result.output_truncated,
            safe_markers_matched=matched,
            attack_marker_matched=attack,
            duration_ms=elapsed,
            reason=reason,
        )

    def render_model(self, model: ProtocolModel) -> str:
        if not isinstance(model, ProtocolModel):
            raise ProtocolValidationError("model must be a ProtocolModel")
        return model.source_for(self.tool)

    def verify(
        self,
        model: ProtocolModel,
        capability: ProtocolToolCapability | None = None,
    ) -> ProtocolLaneResult:
        if not isinstance(model, ProtocolModel):
            raise ProtocolValidationError("model must be a ProtocolModel")
        cap = capability or self.probe()
        if not isinstance(cap, ProtocolToolCapability) or cap.tool is not self.tool:
            raise ProtocolValidationError("capability belongs to another tool")
        if not cap.available or cap.conformance_receipt is None:
            return ProtocolLaneResult(
                model_id=model.model_id,
                model_identity=model.content_id,
                tool=self.tool,
                verdict=ProtocolVerdict.UNAVAILABLE,
                authoritative=False,
                reason=(
                    "protocol lane unavailable without a passing end-to-end "
                    "model fixture"
                ),
                query_results=(),
                capability_identity=cap.content_id,
            )
        executable = cap.executable_path
        assert executable is not None
        current_id = _executable_identity(
            Path(executable), self.max_executable_bytes
        )
        if current_id != cap.executable_identity:
            return ProtocolLaneResult(
                model_id=model.model_id,
                model_identity=model.content_id,
                tool=self.tool,
                verdict=ProtocolVerdict.UNAVAILABLE,
                authoritative=False,
                reason="toolchain drifted after its conformance fixture",
                query_results=(),
                capability_identity=cap.content_id,
            )

        started = self.monotonic()
        source = self.render_model(model)
        with tempfile.TemporaryDirectory(
            prefix=f"{self.tool.value}-protocol-model-"
        ) as raw:
            model_path = Path(raw) / f"model{self.model_extension}"
            model_path.write_text(source, encoding="utf-8")
            command = (executable,) + tuple(
                str(model_path) if item == "{model}" else item
                for item in self.model_args
            )
            run = self._run(
                CommandRequest(
                    command=command,
                    stdin_text=None,
                    cwd=raw,
                    timeout_seconds=self.timeout_seconds,
                    max_output_bytes=self.max_output_bytes,
                )
            )
        duration_ms = max(0, round((self.monotonic() - started) * 1000))
        output = run.stdout + "\n" + run.stderr
        if (
            run.returncode == 0
            and not run.timed_out
            and not run.output_truncated
            and not run.error
        ):
            query_results = _query_outcomes(model, self.tool, output)
            required = tuple(
                result
                for query, result in zip(model.queries, query_results)
                if query.required
            )
            if any(item.verdict is ProtocolVerdict.VIOLATED for item in required):
                verdict = ProtocolVerdict.VIOLATED
                run_status = ToolRunStatus.VIOLATED
                reason = "symbolic tool found a canonical protocol attack"
            elif required and all(
                item.verdict is ProtocolVerdict.VERIFIED for item in required
            ):
                verdict = ProtocolVerdict.VERIFIED
                run_status = ToolRunStatus.PASSED
                reason = "all required symbolic protocol queries were verified"
            else:
                verdict = ProtocolVerdict.INCONCLUSIVE
                run_status = ToolRunStatus.INCONCLUSIVE
                reason = "one or more required protocol queries were inconclusive"
        else:
            query_results = ()
            if run.timed_out:
                verdict = ProtocolVerdict.INCONCLUSIVE
                run_status = ToolRunStatus.TIMED_OUT
                reason = "symbolic protocol verification timed out"
            else:
                verdict = ProtocolVerdict.ERROR
                run_status = ToolRunStatus.ERROR
                reason = "symbolic protocol verification failed to execute completely"

        receipt = ProtocolToolchainReceipt(
            tool=self.tool,
            status=run_status,
            model_id=model.model_id,
            model_identity=model.content_id,
            model_source_identity=model.source_identity_for(self.tool),
            query_set_identity=model.query_set_identity,
            capability_identity=cap.content_id,
            conformance_receipt_identity=cap.conformance_receipt.content_id,
            executable_path=executable,
            executable_identity=current_id,
            executable_version=cap.executable_version or "",
            command=command,
            command_identity=content_identity({"command": command}),
            returncode=run.returncode,
            timed_out=run.timed_out,
            output_truncated=run.output_truncated,
            stdout_sha256=_digest(run.stdout),
            stderr_sha256=_digest(run.stderr),
            duration_ms=duration_ms,
            reason=reason,
        )
        return ProtocolLaneResult(
            model_id=model.model_id,
            model_identity=model.content_id,
            tool=self.tool,
            verdict=verdict,
            authoritative=True,
            reason=reason,
            query_results=query_results,
            capability_identity=cap.content_id,
            toolchain_receipt=receipt,
        )


_TAMARIN_FIXTURE = """\
theory SupervisorProtocolFixture
begin
builtins: hashing
rule MakeSecret: [ Fr(~s) ] --[ SecretMade(~s) ]-> [ Private(~s) ]
rule BeginFinish:
  [ Fr(~x) ] --[ Begin(~x), Finish(~x), FreshUse(~x) ]-> [ Done(~x) ]
rule DeliberateLeak:
  [ Fr(~leaked) ] --[ DeliberatelyLeaked(~leaked) ]-> [ Out(~leaked) ]
lemma fixture_secrecy:
  "All s #i. SecretMade(s) @ i ==> not (Ex #j. K(s) @ j)"
lemma fixture_authenticity:
  "All x #i. Finish(x) @ i
   ==> Ex #j. Begin(x) @ j & (#j < #i | #j = #i)"
lemma fixture_correspondence:
  "All x #i. Finish(x) @ i ==> Ex #j. Begin(x) @ j"
lemma fixture_replay:
  "All x #i #j. FreshUse(x) @ i & FreshUse(x) @ j ==> #i = #j"
lemma fixture_known_attack:
  "All x #i. DeliberatelyLeaked(x) @ i ==> not (Ex #j. K(x) @ j)"
end
"""

_PROVERIF_FIXTURE = """\
free net: channel.
free fixture_secret: bitstring [private].
free leaked: bitstring.
event fixture_begin(bitstring).
event fixture_finish(bitstring).
event fixture_use(bitstring).
(* fixture_secrecy *)
query attacker(fixture_secret).
(* fixture_authenticity *)
query x: bitstring; event(fixture_finish(x)) ==> event(fixture_begin(x)).
(* fixture_correspondence *)
query x: bitstring; inj-event(fixture_finish(x)) ==> inj-event(fixture_begin(x)).
(* fixture_replay *)
query x: bitstring; inj-event(fixture_use(x)) ==> inj-event(fixture_begin(x)).
(* fixture_known_attack *)
query attacker(leaked).
process
  event fixture_begin(fixture_secret);
  event fixture_finish(fixture_secret);
  event fixture_use(fixture_secret);
  out(net, leaked)
"""

TAMARIN_CONFORMANCE_FIXTURE = ProtocolConformanceFixture(
    tool=ProtocolTool.TAMARIN,
    fixture_id="supervisor-protocol-tamarin-e2e@1",
    model_source=_TAMARIN_FIXTURE,
    file_name="fixture.spthy",
    args=("--prove", "{fixture}"),
    safe_markers=(
        "fixture_secrecy: verified",
        "fixture_authenticity: verified",
        "fixture_correspondence: verified",
        "fixture_replay: verified",
    ),
    attack_marker="fixture_known_attack: falsified",
    query_kinds=tuple(ProtocolQueryKind),
    translator_id="supervisor-tamarin-protocol@1",
)

PROVERIF_CONFORMANCE_FIXTURE = ProtocolConformanceFixture(
    tool=ProtocolTool.PROVERIF,
    fixture_id="supervisor-protocol-proverif-e2e@1",
    model_source=_PROVERIF_FIXTURE,
    file_name="fixture.pv",
    args=("{fixture}",),
    safe_markers=(
        "fixture_secrecy is true",
        "fixture_authenticity is true",
        "fixture_correspondence is true",
        "fixture_replay is true",
    ),
    attack_marker="fixture_known_attack is false",
    query_kinds=tuple(ProtocolQueryKind),
    translator_id="supervisor-proverif-protocol@1",
)


class TamarinAdapter(ProtocolToolAdapter):
    tool = ProtocolTool.TAMARIN
    executable_candidates = ("tamarin-prover", "tamarin")
    fixture = TAMARIN_CONFORMANCE_FIXTURE
    model_extension = ".spthy"
    model_args = ("--prove", "{model}")


class ProVerifAdapter(ProtocolToolAdapter):
    tool = ProtocolTool.PROVERIF
    executable_candidates = ("proverif",)
    fixture = PROVERIF_CONFORMANCE_FIXTURE
    model_extension = ".pv"
    model_args = ("{model}",)
    version_args = ("-help",)


DEFAULT_PROTOCOL_ADAPTER_TYPES = (TamarinAdapter, ProVerifAdapter)


def probe_protocol_tools(
    adapters: Sequence[ProtocolToolAdapter] | None = None,
    *,
    run_conformance: bool = True,
) -> tuple[ProtocolToolCapability, ...]:
    selected = (
        tuple(adapters)
        if adapters is not None
        else tuple(kind() for kind in DEFAULT_PROTOCOL_ADAPTER_TYPES)
    )
    capabilities = tuple(
        adapter.probe(run_conformance=run_conformance) for adapter in selected
    )
    if len({item.tool for item in capabilities}) != len(capabilities):
        raise ProtocolValidationError("protocol tool adapters must be unique")
    return capabilities


class ProtocolVerifier:
    """Run exact models through both capability-gated symbolic lanes."""

    def __init__(
        self, adapters: Sequence[ProtocolToolAdapter] | None = None
    ) -> None:
        self.adapters = (
            tuple(adapters)
            if adapters is not None
            else tuple(kind() for kind in DEFAULT_PROTOCOL_ADAPTER_TYPES)
        )
        if len({item.tool for item in self.adapters}) != len(self.adapters):
            raise ProtocolValidationError("protocol tool adapters must be unique")

    def capabilities(
        self, *, run_conformance: bool = True
    ) -> tuple[ProtocolToolCapability, ...]:
        return probe_protocol_tools(
            self.adapters, run_conformance=run_conformance
        )

    def verify(
        self,
        model: ProtocolModel,
        *,
        capabilities: Sequence[ProtocolToolCapability] | None = None,
    ) -> ProtocolSuiteResult:
        if not isinstance(model, ProtocolModel):
            raise ProtocolValidationError("model must be a ProtocolModel")
        caps = (
            tuple(capabilities)
            if capabilities is not None
            else self.capabilities()
        )
        by_tool = {item.tool: item for item in caps}
        if len(by_tool) != len(caps):
            raise ProtocolValidationError("capabilities must be unique by tool")
        lanes: list[ProtocolLaneResult] = []
        for adapter in self.adapters:
            cap = by_tool.get(adapter.tool)
            if cap is None:
                cap = ProtocolToolCapability(
                    tool=adapter.tool,
                    status=ToolCapabilityStatus.UNAVAILABLE,
                    reason="no capability supplied for required protocol lane",
                )
            lanes.append(adapter.verify(model, cap))
        lane_tuple = tuple(lanes)
        complete = (
            {item.tool for item in lane_tuple} == set(ProtocolTool)
            and all(item.authoritative for item in lane_tuple)
        )
        if any(item.verdict is ProtocolVerdict.VIOLATED for item in lane_tuple):
            verdict = ProtocolVerdict.VIOLATED
        elif complete and all(
            item.verdict is ProtocolVerdict.VERIFIED for item in lane_tuple
        ):
            verdict = ProtocolVerdict.VERIFIED
        elif any(item.verdict is ProtocolVerdict.ERROR for item in lane_tuple):
            verdict = ProtocolVerdict.ERROR
        elif any(
            item.verdict is ProtocolVerdict.INCONCLUSIVE for item in lane_tuple
        ):
            verdict = ProtocolVerdict.INCONCLUSIVE
        else:
            verdict = ProtocolVerdict.UNAVAILABLE
        return ProtocolSuiteResult(
            model_id=model.model_id,
            model_identity=model.content_id,
            lane_results=lane_tuple,
            verdict=verdict,
            complete=complete,
        )

    def verify_all(
        self,
        models: Sequence[ProtocolModel] | None = None,
        *,
        capabilities: Sequence[ProtocolToolCapability] | None = None,
    ) -> tuple[ProtocolSuiteResult, ...]:
        selected = tuple(models) if models is not None else DEFAULT_PROTOCOL_MODELS
        caps = (
            tuple(capabilities)
            if capabilities is not None
            else self.capabilities()
        )
        return tuple(self.verify(model, capabilities=caps) for model in selected)


def _query(
    query_id: str,
    prop: ProtocolProperty,
    kind: ProtocolQueryKind,
    abstraction: str,
    *,
    tamarin_name: str | None = None,
    proverif_label: str | None = None,
    assumptions: tuple[str, ...] = (
        "long-term signing keys are uncompromised",
        "cryptographic constructors are perfect symbolic primitives",
    ),
    excluded: tuple[str, ...] = (
        "availability and denial-of-service behavior",
        "side channels and implementation-level key extraction",
    ),
) -> ProtocolQuery:
    symbol = tamarin_name or query_id.replace(".", "_").replace("-", "_")
    return ProtocolQuery(
        query_id=query_id,
        property=prop,
        kind=kind,
        abstraction=abstraction,
        assumptions=assumptions,
        excluded_behaviors=excluded,
        tamarin_name=symbol,
        proverif_label=proverif_label or query_id,
    )


CORE_PROTOCOL_QUERIES: tuple[ProtocolQuery, ...] = (
    _query(
        "claimant_key_secrecy",
        ProtocolProperty.CLAIMANT_AUTHENTICATION,
        ProtocolQueryKind.SECRECY,
        "The claimant signing key represents claimant identity and is never sent.",
    ),
    _query(
        "claimant_authenticity",
        ProtocolProperty.CLAIMANT_AUTHENTICATION,
        ProtocolQueryKind.AUTHENTICITY,
        "An accepted claim corresponds to a claimant-signed task, nonce, and epoch.",
    ),
    _query(
        "lease_grant_correspondence",
        ProtocolProperty.LEASE_GRANTS,
        ProtocolQueryKind.CORRESPONDENCE,
        "Every claimant-accepted lease was granted by the supervisor for the same tuple.",
    ),
    _query(
        "fencing_freshness",
        ProtocolProperty.FENCING_FRESHNESS,
        ProtocolQueryKind.REPLAY,
        "A mutation uses the unique current lease epoch; older epochs cannot authorize it.",
    ),
    _query(
        "claim_replay_resistance",
        ProtocolProperty.REPLAY_RESISTANCE,
        ProtocolQueryKind.REPLAY,
        "A signed claim nonce can be accepted at most once.",
    ),
    _query(
        "receipt_binding_authenticity",
        ProtocolProperty.RECEIPT_BINDING,
        ProtocolQueryKind.AUTHENTICITY,
        "A verified receipt binds task, tree, obligation, result, claimant, and fence.",
    ),
    _query(
        "merge_authorization_correspondence",
        ProtocolProperty.MERGE_AUTHORIZATION,
        ProtocolQueryKind.CORRESPONDENCE,
        "Every merge follows authorization of the exact verified receipt and current fence.",
    ),
)

_CORE_TAMARIN = """\
theory SupervisorClaimLeaseReceipt
begin
builtins: signing, hashing
functions: bind/6
rule RegisterClaimant:
  [ Fr(~skC) ] --[ ClaimantKey(~skC) ]-> [ !Claimant($C, ~skC), Out(pk(~skC)) ]
rule RegisterSupervisor:
  [ Fr(~skS) ] --> [ !Supervisor(~skS), Out(pk(~skS)) ]
rule SignedClaim:
  [ !Claimant(C, skC), Fr(~nonce), Fr(~epoch) ]
  --[ ClaimSent(C, $task, ~nonce, ~epoch) ]->
  [ Out(<C,$task,~nonce,~epoch,sign(<$task,~nonce,~epoch>,skC)>),
    UnusedClaim(C,$task,~nonce,~epoch),
    AwaitLease(C,$task,~nonce,~epoch) ]
rule GrantLease:
  [ !Supervisor(skS),
    In(<C,task,nonce,epoch,sign(<task,nonce,epoch>,skC)>),
    !Claimant(C, skC), UnusedClaim(C,task,nonce,epoch) ]
  --[ ClaimAccepted(C,task,nonce,epoch),
      LeaseGranted(C,task,nonce,epoch) ]->
  [ Out(<C,task,nonce,epoch,sign(<C,task,nonce,epoch>,skS)>),
    LeaseState(C,task,nonce,epoch) ]
rule AcceptLease:
  [ !Supervisor(skS),
    In(<C,task,nonce,epoch,sign(<C,task,nonce,epoch>,skS)>),
    AwaitLease(C,task,nonce,epoch) ]
  --[ LeaseAccepted(C,task,nonce,epoch) ]->
  [ LeaseUse(C,task,nonce,epoch) ]
rule ProduceReceipt:
  [ LeaseState(C,task,nonce,epoch), Fr(~result) ]
  --[ CurrentFence(task,epoch), Mutation(C,task,epoch),
      ReceiptProduced(C,task,$tree,$obligation,~result,epoch) ]->
  [ Out(bind(task,$tree,$obligation,~result,C,epoch)),
    ReceiptState(C,task,$tree,$obligation,~result,epoch) ]
rule AuthorizeMerge:
  [ ReceiptState(C,task,tree,obligation,result,epoch) ]
  --[ ReceiptVerified(C,task,tree,obligation,result,epoch),
      MergeAuthorized(C,task,tree,obligation,result,epoch),
      Merged(C,task,tree,obligation,result,epoch) ]->
  [ ]
lemma claimant_key_secrecy:
  "All sk #i. ClaimantKey(sk) @ i ==> not (Ex #j. K(sk) @ j)"
lemma claimant_authenticity:
  "All C t n e #i. ClaimAccepted(C,t,n,e) @ i
   ==> Ex #j. ClaimSent(C,t,n,e) @ j & #j < #i"
lemma lease_grant_correspondence:
  "All C t n e #i. LeaseAccepted(C,t,n,e) @ i
   ==> Ex #j. LeaseGranted(C,t,n,e) @ j & #j < #i"
lemma fencing_freshness:
  "All C t e #i #j. Mutation(C,t,e) @ i & Mutation(C,t,e) @ j ==> #i = #j"
lemma claim_replay_resistance:
  "All C t n e #i #j. ClaimAccepted(C,t,n,e) @ i
    & ClaimAccepted(C,t,n,e) @ j ==> #i = #j"
lemma receipt_binding_authenticity:
  "All C t tr o r e #i. ReceiptVerified(C,t,tr,o,r,e) @ i
   ==> Ex #j. ReceiptProduced(C,t,tr,o,r,e) @ j
    & (#j < #i | #j = #i)"
lemma merge_authorization_correspondence:
  "All C t tr o r e #i. Merged(C,t,tr,o,r,e) @ i
   ==> Ex #j #k. ReceiptVerified(C,t,tr,o,r,e) @ j
    & MergeAuthorized(C,t,tr,o,r,e) @ k
    & (#j < #i | #j = #i) & (#k < #i | #k = #i)"
end
"""

_CORE_PROVERIF = """\
type skey.
type pkey.
fun pk(skey): pkey.
fun sign(bitstring, skey): bitstring.
reduc forall m: bitstring, k: skey; checksign(sign(m,k),pk(k)) = m.
fun bind(bitstring,bitstring,bitstring,bitstring,bitstring,bitstring): bitstring.
free net: channel.
free claimant_sk: skey [private].
free supervisor_sk: skey [private].
free task, tree, obligation, result, claimant, fence, nonce: bitstring.
event ClaimSent(bitstring,bitstring,bitstring,bitstring).
event ClaimAccepted(bitstring,bitstring,bitstring,bitstring).
event LeaseGranted(bitstring,bitstring,bitstring,bitstring).
event LeaseAccepted(bitstring,bitstring,bitstring,bitstring).
event Mutation(bitstring,bitstring,bitstring).
event ReceiptProduced(bitstring,bitstring,bitstring,bitstring,bitstring,bitstring).
event ReceiptVerified(bitstring,bitstring,bitstring,bitstring,bitstring,bitstring).
event MergeAuthorized(bitstring,bitstring,bitstring,bitstring,bitstring,bitstring).
event Merged(bitstring,bitstring,bitstring,bitstring,bitstring,bitstring).
(* claimant_key_secrecy *)
query attacker(claimant_sk).
(* claimant_authenticity *)
query c,t,n,e: bitstring; event(ClaimAccepted(c,t,n,e)) ==> event(ClaimSent(c,t,n,e)).
(* lease_grant_correspondence *)
query c,t,n,e: bitstring; event(LeaseAccepted(c,t,n,e)) ==> event(LeaseGranted(c,t,n,e)).
(* fencing_freshness *)
query c,t,e: bitstring; inj-event(Mutation(c,t,e)) ==> inj-event(LeaseAccepted(c,t,nonce,e)).
(* claim_replay_resistance *)
query c,t,n,e: bitstring; inj-event(ClaimAccepted(c,t,n,e)) ==> inj-event(ClaimSent(c,t,n,e)).
(* receipt_binding_authenticity *)
query c,t,tr,o,r,e: bitstring; event(ReceiptVerified(c,t,tr,o,r,e)) ==> event(ReceiptProduced(c,t,tr,o,r,e)).
(* merge_authorization_correspondence *)
query c,t,tr,o,r,e: bitstring; event(Merged(c,t,tr,o,r,e)) ==> event(MergeAuthorized(c,t,tr,o,r,e)).
process
  event ClaimSent(claimant,task,nonce,fence);
  event ClaimAccepted(claimant,task,nonce,fence);
  event LeaseGranted(claimant,task,nonce,fence);
  event LeaseAccepted(claimant,task,nonce,fence);
  event Mutation(claimant,task,fence);
  event ReceiptProduced(claimant,task,tree,obligation,result,fence);
  event ReceiptVerified(claimant,task,tree,obligation,result,fence);
  event MergeAuthorized(claimant,task,tree,obligation,result,fence);
  event Merged(claimant,task,tree,obligation,result,fence)
"""

CORE_PROTOCOL_MODEL = ProtocolModel(
    model_id="supervisor.claim-lease-receipt-merge",
    version="1",
    description=(
        "Authenticated claims, supervisor leases, fresh fencing, bound receipts, "
        "and receipt-authorized merges over a Dolev-Yao network."
    ),
    properties=(
        ProtocolProperty.CLAIMANT_AUTHENTICATION,
        ProtocolProperty.LEASE_GRANTS,
        ProtocolProperty.FENCING_FRESHNESS,
        ProtocolProperty.REPLAY_RESISTANCE,
        ProtocolProperty.RECEIPT_BINDING,
        ProtocolProperty.MERGE_AUTHORIZATION,
    ),
    queries=CORE_PROTOCOL_QUERIES,
    tamarin_source=_CORE_TAMARIN,
    proverif_source=_CORE_PROVERIF,
    protocol_steps=(
        "claimant signs task, nonce, and requested fencing epoch",
        "supervisor authenticates the claim and signs the lease grant",
        "claimant accepts the exact grant and performs one current-fence mutation",
        "proof receipt binds task, tree, obligation, result, claimant, and fence",
        "merge consumes authorization for the exact verified receipt",
    ),
)


ATTESTATION_PROTOCOL_QUERIES: tuple[ProtocolQuery, ...] = (
    _query(
        "attestation_challenge_secrecy",
        ProtocolProperty.ATTESTATION_EXCHANGE,
        ProtocolQueryKind.SECRECY,
        "The verifier challenge remains confidential until bound into a quote.",
    ),
    _query(
        "attestation_quote_authenticity",
        ProtocolProperty.ATTESTATION_EXCHANGE,
        ProtocolQueryKind.AUTHENTICITY,
        "An accepted quote was signed by the attester for the same measurement and receipt.",
    ),
    _query(
        "attestation_accept_correspondence",
        ProtocolProperty.ATTESTATION_EXCHANGE,
        ProtocolQueryKind.CORRESPONDENCE,
        "Attestation acceptance corresponds to a verifier challenge and matching quote.",
    ),
    _query(
        "attestation_replay_resistance",
        ProtocolProperty.ATTESTATION_EXCHANGE,
        ProtocolQueryKind.REPLAY,
        "Each fresh challenge authorizes at most one accepted quote.",
    ),
)

_ATTESTATION_TAMARIN = """\
theory SupervisorOptionalAttestation
begin
builtins: asymmetric-encryption, signing, hashing
rule RegisterAttester:
  [ Fr(~skA) ] --[ AttesterKey(~skA) ]-> [ !Attester($A,~skA), Out(pk(~skA)) ]
rule Challenge:
  [ !Attester(A,skA), Fr(~challenge) ]
  --[ ChallengeMade($V,A,$receipt,~challenge) ]->
  [ Out(aenc(<$V,$receipt,~challenge>,pk(skA))),
    Pending($V,A,$receipt,~challenge) ]
rule Quote:
  [ !Attester(A,skA), In(aenc(<V,receipt,challenge>,pk(skA))) ]
  --[ QuoteMade(A,V,receipt,$measurement,challenge) ]->
  [ Out(sign(<A,V,receipt,$measurement,challenge>,skA)) ]
rule AcceptQuote:
  [ Pending(V,A,receipt,challenge),
    In(sign(<A,V,receipt,measurement,challenge>,skA)),
    !Attester(A,skA) ]
  --[ QuoteAccepted(A,V,receipt,measurement,challenge) ]->
  [ ]
lemma attestation_challenge_secrecy:
  "All V A r c #i. ChallengeMade(V,A,r,c) @ i ==> not (Ex #j. K(c) @ j)"
lemma attestation_quote_authenticity:
  "All A V r m c #i. QuoteAccepted(A,V,r,m,c) @ i
   ==> Ex #j. QuoteMade(A,V,r,m,c) @ j & #j < #i"
lemma attestation_accept_correspondence:
  "All A V r m c #i. QuoteAccepted(A,V,r,m,c) @ i
   ==> Ex #j. ChallengeMade(V,A,r,c) @ j & #j < #i"
lemma attestation_replay_resistance:
  "All A V r m c #i #j. QuoteAccepted(A,V,r,m,c) @ i
    & QuoteAccepted(A,V,r,m,c) @ j ==> #i = #j"
end
"""

_ATTESTATION_PROVERIF = """\
type skey.
type pkey.
fun pk(skey): pkey.
fun sign(bitstring,skey): bitstring.
reduc forall m: bitstring,k:skey; checksign(sign(m,k),pk(k)) = m.
free net: channel.
free attester_sk: skey [private].
free verifier, attester, receipt, measurement: bitstring.
free challenge: bitstring [private].
event ChallengeMade(bitstring,bitstring,bitstring,bitstring).
event QuoteMade(bitstring,bitstring,bitstring,bitstring,bitstring).
event QuoteAccepted(bitstring,bitstring,bitstring,bitstring,bitstring).
(* attestation_challenge_secrecy *)
query attacker(challenge).
(* attestation_quote_authenticity *)
query a,v,r,m,c: bitstring; event(QuoteAccepted(a,v,r,m,c)) ==> event(QuoteMade(a,v,r,m,c)).
(* attestation_accept_correspondence *)
query a,v,r,m,c: bitstring; event(QuoteAccepted(a,v,r,m,c)) ==> event(ChallengeMade(v,a,r,c)).
(* attestation_replay_resistance *)
query a,v,r,m,c: bitstring; inj-event(QuoteAccepted(a,v,r,m,c)) ==> inj-event(ChallengeMade(v,a,r,c)).
process
  event ChallengeMade(verifier,attester,receipt,challenge);
  event QuoteMade(attester,verifier,receipt,measurement,challenge);
  event QuoteAccepted(attester,verifier,receipt,measurement,challenge)
"""

ATTESTATION_PROTOCOL_MODEL = ProtocolModel(
    model_id="supervisor.optional-attestation-exchange",
    version="1",
    description=(
        "Optional challenge/quote exchange binding an attester measurement to "
        "the exact supervisor proof receipt."
    ),
    properties=(ProtocolProperty.ATTESTATION_EXCHANGE,),
    queries=ATTESTATION_PROTOCOL_QUERIES,
    tamarin_source=_ATTESTATION_TAMARIN,
    proverif_source=_ATTESTATION_PROVERIF,
    protocol_steps=(
        "verifier creates a fresh challenge bound to a receipt",
        "attester signs challenge, measurement, verifier, and receipt",
        "verifier accepts one matching quote for the pending challenge",
    ),
    optional_attestation=True,
)

DEFAULT_PROTOCOL_MODELS: tuple[ProtocolModel, ...] = (
    CORE_PROTOCOL_MODEL,
    ATTESTATION_PROTOCOL_MODEL,
)
DEFAULT_PROTOCOL_MODELS_BY_ID: Mapping[str, ProtocolModel] = {
    model.model_id: model for model in DEFAULT_PROTOCOL_MODELS
}


def default_protocol_models() -> tuple[ProtocolModel, ...]:
    return DEFAULT_PROTOCOL_MODELS


def protocol_model_for(model_id: str) -> ProtocolModel:
    try:
        return DEFAULT_PROTOCOL_MODELS_BY_ID[_text(model_id, "model_id")]
    except KeyError as exc:
        raise ProtocolValidationError("unknown protocol model") from exc


def verify_protocol_model(
    model: ProtocolModel,
    *,
    adapters: Sequence[ProtocolToolAdapter] | None = None,
    capabilities: Sequence[ProtocolToolCapability] | None = None,
) -> ProtocolSuiteResult:
    return ProtocolVerifier(adapters).verify(model, capabilities=capabilities)


# Compatibility names used by the formal-verification package vocabulary.
ProtocolEngine = ProtocolTool
ProtocolEngineCapability = ProtocolToolCapability
ProtocolEngineAdapter = ProtocolToolAdapter
TamarinEngineAdapter = TamarinAdapter
ProVerifEngineAdapter = ProVerifAdapter
AttackCounterexample = ProtocolAttackCounterexample
ToolchainReceipt = ProtocolToolchainReceipt
QueryKind = ProtocolQueryKind


__all__ = [
    "ATTESTATION_PROTOCOL_MODEL",
    "ATTESTATION_PROTOCOL_QUERIES",
    "AttackCounterexample",
    "CORE_PROTOCOL_MODEL",
    "CORE_PROTOCOL_QUERIES",
    "ConformanceStatus",
    "DEFAULT_MAX_EXECUTABLE_BYTES",
    "DEFAULT_MAX_PROTOCOL_OUTPUT_BYTES",
    "DEFAULT_PROTOCOL_ADAPTER_TYPES",
    "DEFAULT_PROTOCOL_MODELS",
    "DEFAULT_PROTOCOL_MODELS_BY_ID",
    "DEFAULT_PROTOCOL_TIMEOUT_SECONDS",
    "PROTOCOL_VERIFICATION_VERSION",
    "PROVERIF_CONFORMANCE_FIXTURE",
    "ProtocolAttackCounterexample",
    "ProtocolAttackStep",
    "ProtocolConformanceFixture",
    "ProtocolConformanceReceipt",
    "ProtocolEngine",
    "ProtocolEngineAdapter",
    "ProtocolEngineCapability",
    "ProtocolLaneResult",
    "ProtocolModel",
    "ProtocolProperty",
    "ProtocolQuery",
    "ProtocolQueryKind",
    "ProtocolQueryResult",
    "ProtocolSuiteResult",
    "ProtocolTool",
    "ProtocolToolAdapter",
    "ProtocolToolCapability",
    "ProtocolToolchainReceipt",
    "ProtocolValidationError",
    "ProtocolVerdict",
    "ProtocolVerifier",
    "ProVerifAdapter",
    "ProVerifEngineAdapter",
    "QueryKind",
    "TAMARIN_CONFORMANCE_FIXTURE",
    "TamarinAdapter",
    "TamarinEngineAdapter",
    "ToolCapabilityStatus",
    "ToolRunStatus",
    "ToolchainReceipt",
    "canonicalize_attack_trace",
    "default_protocol_models",
    "probe_protocol_tools",
    "protocol_model_for",
    "verify_protocol_model",
]
