"""Fail-closed JSON-RPC, schema, CID, and profile repair operators (DCR-042).

Interface: ``ProtocolRepairOperators@1``

Evidence: ``dcr/protocol-repair@1``

Implements structural preview/inverse bodies for the protocol operator kinds:

* :attr:`OperatorKind.REPAIR_JSONRPC_SCHEMA` — JSON-RPC 2.0 status/version/ID
  and result/error mutual-exclusivity checks
* :attr:`OperatorKind.REPAIR_REQUEST_ADAPTER` — closed schema bindings for
  request/response adapters
* :attr:`OperatorKind.REPAIR_ERROR_ENVELOPE` — local CID/receipt verification
  (never trusts server ``verified`` flags)
* :attr:`OperatorKind.REPAIR_PROFILE_BINDING` — capability subset negotiation
  and fail-closed policy / transport decisions

Normative rules (fail-closed)
-----------------------------
* HTTP errors, wrong JSON-RPC versions/IDs, dual result+error payloads, and
  missing result/error all **reject**.
* Bad schemas, CIDs, and receipts **reject**; server-supplied ``verified``
  claims never establish local verification.
* Unsupported required profiles and transport mismatches **reject**.
* Policy outage / missing decisions **deny** — never convert to allow/success.
* Operators remain proposal-only: they never grant write, proof, or semantic
  authority and never mutate production trees.

Predicted symbols: :class:`JsonRpcValidationOperator`,
:class:`SchemaBindingOperator`, :class:`CanonicalCidOperator`,
:class:`ProfileNegotiationOperator`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ...proof.formal_verification_contracts import (
    CanonicalContract,
    content_identity,
)
from .registry import (
    OperatorFamily,
    OperatorKind,
    build_default_operator_registry,
)


# ---------------------------------------------------------------------------
# Closed interface / evidence constants
# ---------------------------------------------------------------------------

PROTOCOL_REPAIR_OPERATORS_INTERFACE: Final[str] = "ProtocolRepairOperators@1"
PROTOCOL_REPAIR_EVIDENCE: Final[str] = "dcr/protocol-repair@1"
PROTOCOL_REPAIR_VERSION: Final[int] = 1

PROTOCOL_REPAIR_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/protocol-repair@1"
)
JSONRPC_ENVELOPE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/protocol-jsonrpc-envelope@1"
)
SCHEMA_BINDING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/protocol-schema-binding@1"
)
CID_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/protocol-cid-receipt@1"
)
PROFILE_NEGOTIATION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/protocol-profile-negotiation@1"
)
PROTOCOL_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/protocol-repair-request@1"
)
PROTOCOL_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/protocol-repair-receipt@1"
)
PROTOCOL_OPERATOR_VECTORS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/protocol-operator-vectors@1"
)

JSONRPC_VERSION: Final[str] = "2.0"

# MCP++ Profiles A–F (plus basic fallback token used by connectors).
MCP_PROFILE_A: Final[str] = "mcp++/mcp-idl"
MCP_PROFILE_B: Final[str] = "mcp++/cid-envelope"
MCP_PROFILE_C: Final[str] = "mcp++/ucan"
MCP_PROFILE_D: Final[str] = "mcp++/deontic-policy"
MCP_PROFILE_E: Final[str] = "mcp++/p2p-transport"
MCP_PROFILE_F: Final[str] = "mcp++/event-dag"
MCP_PROFILE_BASIC: Final[str] = "mcp++/basic"

MCP_PROFILES_A_F: Final[tuple[str, ...]] = (
    MCP_PROFILE_A,
    MCP_PROFILE_B,
    MCP_PROFILE_C,
    MCP_PROFILE_D,
    MCP_PROFILE_E,
    MCP_PROFILE_F,
)

SUPPORTED_TRANSPORTS: Final[frozenset[str]] = frozenset({"http", "libp2p"})

MAX_TEXT_BYTES: Final[int] = 4_096
MAX_ID_BYTES: Final[int] = 512
MAX_COLLECTION: Final[int] = 256
MAX_REASON_CODES: Final[int] = 32

_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9+][A-Za-z0-9+._:/-]{0,255}$"
)
_CID_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:bafy|bagu|bafk|bafkre|sha256:)[A-Za-z0-9:_-]{8,200}$"
)
_SCHEMA_REF_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:schema:|ipfs_accelerate_py/|mcp\+\+/|json-schema:)[A-Za-z0-9._:@+/-]{1,240}$"
)

_FORBIDDEN_PAYLOAD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "source_body",
        "source_text",
        "code",
        "code_body",
        "shell",
        "shell_fragment",
        "command",
        "script",
        "callable",
        "dynamic_import",
        "exec",
        "eval",
        "llm_prompt",
        "prose",
        "patch_body",
        "diff_body",
        "handler_body",
        "private_key",
        "secret",
        "password",
    }
)

# Closed method vocabulary for schema bindings (base MCP + MCP++ Profiles A–F).
REVIEWED_METHOD_SCHEMAS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "initialize": "schema:mcp/initialize@1",
        "ping": "schema:mcp/ping@1",
        "tools/list": "schema:mcp/tools-list@1",
        "tools/call": "schema:mcp/tools-call@1",
        "interfaces/list": "schema:mcp++/interfaces-list@1",
        "interfaces/get": "schema:mcp++/interfaces-get@1",
        "mcp++/execute": "schema:mcp++/execute@1",
        "mcp++/artifacts/get": "schema:mcp++/artifacts-get@1",
        "mcp++/policy/evaluate": "schema:mcp++/policy-evaluate@1",
        "mcp++/ucan/delegate": "schema:mcp++/ucan-delegate@1",
        "mcp++/dag/frontier": "schema:mcp++/dag-frontier@1",
    }
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ProtocolRepairError(ValueError):
    """Malformed protocol repair input or closed-boundary violation."""


class ProtocolRepairAbstention(ProtocolRepairError):
    """Operator cannot proceed without inventing protocol semantics."""


class RepairDisposition(str, Enum):  # noqa: UP042 - package supports Python 3.8
    """Closed outcomes for one protocol repair attempt."""

    PREVIEW_READY = "preview_ready"
    ALREADY_ALIGNED = "already_aligned"
    ABSTAIN = "abstain"
    REJECTED = "rejected"
    DENIED = "denied"
    ACCEPTED = "accepted"


class OperatorRole(str, Enum):  # noqa: UP042
    """Closed operator roles implementing DCR-042 protocol repairs."""

    JSONRPC_VALIDATION = "jsonrpc_validation"
    SCHEMA_BINDING = "schema_binding"
    CANONICAL_CID = "canonical_cid"
    PROFILE_NEGOTIATION = "profile_negotiation"


class ProtocolVerdict(str, Enum):  # noqa: UP042
    """Closed protocol evaluation verdicts."""

    PASS = "pass"
    REJECT = "reject"
    DENY = "deny"


class ReasonCode(str, Enum):  # noqa: UP042
    """Closed reason codes covering acceptance negative vectors."""

    OK = "ok"
    HTTP_ERROR = "http_error"
    WRONG_JSONRPC_VERSION = "wrong_jsonrpc_version"
    WRONG_ID = "wrong_id"
    MISSING_RESULT_AND_ERROR = "missing_result_and_error"
    BOTH_RESULT_AND_ERROR = "both_result_and_error"
    BAD_SCHEMA = "bad_schema"
    UNKNOWN_METHOD = "unknown_method"
    BAD_CID = "bad_cid"
    BAD_RECEIPT = "bad_receipt"
    SERVER_VERIFIED_UNTRUSTED = "server_verified_untrusted"
    LOCAL_CID_MISMATCH = "local_cid_mismatch"
    UNSUPPORTED_PROFILE = "unsupported_profile"
    PROFILE_DOWNGRADE = "profile_downgrade"
    POLICY_OUTAGE = "policy_outage"
    MISSING_POLICY_DECISION = "missing_policy_decision"
    POLICY_ALLOW_FROM_OUTAGE = "policy_allow_from_outage"
    TRANSPORT_MISMATCH = "transport_mismatch"
    MALFORMED_ENVELOPE = "malformed_envelope"
    INVENTED_SCHEMA = "invented_schema"
    ALREADY_ALIGNED = "already_aligned"
    PREVIEW_READY = "preview_ready"


class PolicyAvailability(str, Enum):  # noqa: UP042
    """Whether the policy evaluation surface is available."""

    AVAILABLE = "available"
    OUTAGE = "outage"
    MISSING = "missing"
    UNKNOWN = "unknown"


class TransportKind(str, Enum):  # noqa: UP042
    """Closed transport kinds for MCP++ connectors."""

    HTTP = "http"
    LIBP2P = "libp2p"


_ROLE_TO_KIND: Final[Mapping[OperatorRole, OperatorKind]] = MappingProxyType(
    {
        OperatorRole.JSONRPC_VALIDATION: OperatorKind.REPAIR_JSONRPC_SCHEMA,
        OperatorRole.SCHEMA_BINDING: OperatorKind.REPAIR_REQUEST_ADAPTER,
        OperatorRole.CANONICAL_CID: OperatorKind.REPAIR_ERROR_ENVELOPE,
        OperatorRole.PROFILE_NEGOTIATION: OperatorKind.REPAIR_PROFILE_BINDING,
    }
)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    maximum: int = MAX_TEXT_BYTES,
    identifier: bool = False,
) -> str:
    if not isinstance(value, str):
        raise ProtocolRepairError(f"{name} must be a string")
    result = value.strip()
    if required and not result:
        raise ProtocolRepairError(f"{name} must not be empty")
    if "\x00" in result:
        raise ProtocolRepairError(f"{name} must not contain NUL")
    if len(result.encode("utf-8")) > maximum:
        raise ProtocolRepairError(f"{name} exceeds its byte bound")
    if identifier and result and not _IDENTIFIER_RE.fullmatch(result):
        raise ProtocolRepairError(f"{name} must be a closed identifier")
    return result


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ProtocolRepairError(f"{name} must be a boolean")
    return value


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolRepairError(f"{name} must be an integer")
    return value


def _enum(value: Any, kind: type[Enum], name: str) -> Any:
    try:
        return kind(str(getattr(value, "value", value)))
    except (TypeError, ValueError) as exc:
        raise ProtocolRepairError(f"unsupported {name}: {value!r}") from exc


def _cid(value: Any, name: str, *, required: bool = True) -> str:
    text = _text(value, name, required=required, maximum=MAX_ID_BYTES)
    if text and not _CID_RE.fullmatch(text):
        if not text.startswith("sha256:") and not text.startswith("b"):
            raise ProtocolRepairError(f"{name} must be a content identity")
    return text


def _schema_ref(value: Any, name: str = "schema_ref") -> str:
    text = _text(value, name)
    if not _SCHEMA_REF_RE.fullmatch(text):
        raise ProtocolRepairError(f"{name} must be a reviewed schema reference")
    return text


def _string_tuple(
    values: Any,
    name: str,
    *,
    required: bool = False,
    maximum: int = MAX_COLLECTION,
    ordered: bool = True,
) -> tuple[str, ...]:
    if values is None:
        items: Sequence[Any] = ()
    elif isinstance(values, str):
        items = (values,)
    elif isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray)):
        items = values
    else:
        raise ProtocolRepairError(f"{name} must be a sequence of strings")
    if len(items) > maximum:
        raise ProtocolRepairError(f"{name} exceeds its item bound")
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        text = _text(item, f"{name}[{index}]")
        if text in seen:
            raise ProtocolRepairError(f"{name} must not contain duplicates")
        seen.add(text)
        result.append(text)
    if required and not result:
        raise ProtocolRepairError(f"{name} must not be empty")
    if ordered:
        return tuple(result)
    return tuple(sorted(result))


def _reject_forbidden_fields(payload: Mapping[str, Any], *, label: str) -> None:
    for key in payload:
        lowered = str(key).strip().lower()
        if lowered in _FORBIDDEN_PAYLOAD_KEYS:
            raise ProtocolRepairError(f"{label} contains forbidden field {lowered!r}")


def _reason_codes(value: Any, name: str = "reason_codes") -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ProtocolRepairError(f"{name} must be a sequence")
    if len(value) > MAX_REASON_CODES:
        raise ProtocolRepairError(f"{name} exceeds its item bound")
    return tuple(_text(item, f"{name}[{index}]") for index, item in enumerate(value))


def local_cid_for_bytes(payload: bytes) -> str:
    """Return a local ``sha256:`` content identity for raw payload bytes."""

    if not isinstance(payload, (bytes, bytearray)):
        raise ProtocolRepairError("payload must be bytes")
    return "sha256:" + hashlib.sha256(bytes(payload)).hexdigest()


def local_cid_for_json(value: Any) -> str:
    """Return a local content identity for a canonical JSON value."""

    return local_cid_for_bytes(_canonical_json_bytes(value))


def decode_bytes_base64(value: Any, name: str = "bytes_base64") -> bytes:
    text = _text(value, name, required=False)
    if not text:
        return b""
    try:
        return base64.b64decode(text, validate=True)
    except Exception as exc:  # noqa: BLE001 - closed decode boundary
        raise ProtocolRepairError(f"{name} must be valid base64") from exc


# ---------------------------------------------------------------------------
# Protocol contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JsonRpcEnvelope(CanonicalContract):
    """One JSON-RPC 2.0 request/response pair under HTTP or libp2p carriage."""

    SCHEMA: ClassVar[str] = JSONRPC_ENVELOPE_SCHEMA
    INTERFACE: ClassVar[str] = PROTOCOL_REPAIR_OPERATORS_INTERFACE

    request_id: str | int
    response_id: str | int | None
    jsonrpc: str
    method: str
    http_status: int | None = 200
    result: Any = None
    error: Mapping[str, Any] | None = None
    transport: TransportKind = TransportKind.HTTP
    has_result: bool = False
    has_error: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, (str, int)) or isinstance(
            self.request_id, bool
        ):
            raise ProtocolRepairError("request_id must be a string or integer")
        if self.response_id is not None and (
            not isinstance(self.response_id, (str, int))
            or isinstance(self.response_id, bool)
        ):
            raise ProtocolRepairError("response_id must be a string, integer, or null")
        object.__setattr__(self, "jsonrpc", _text(self.jsonrpc, "jsonrpc"))
        object.__setattr__(self, "method", _text(self.method, "method", identifier=True))
        status = _optional_int(self.http_status, "http_status")
        if status is not None and (status < 100 or status > 599):
            raise ProtocolRepairError("http_status must be an HTTP status code")
        object.__setattr__(self, "http_status", status)
        object.__setattr__(self, "transport", _enum(self.transport, TransportKind, "transport"))
        has_result = self.result is not None or self.has_result is True
        has_error = self.error is not None or self.has_error is True
        if self.error is not None and not isinstance(self.error, Mapping):
            raise ProtocolRepairError("error must be an object when present")
        object.__setattr__(self, "has_result", bool(has_result and self.result is not None) or (
            self.has_result is True and self.result is not None
        ))
        # Explicit flags allow tests to model null-result vs missing-result.
        if self.has_result is True and self.result is None and self.error is None:
            object.__setattr__(self, "has_result", True)
        else:
            object.__setattr__(
                self,
                "has_result",
                True if self.result is not None else bool(self.has_result),
            )
        object.__setattr__(
            self,
            "has_error",
            True if self.error is not None else bool(self.has_error),
        )
        if self.error is not None:
            object.__setattr__(self, "error", MappingProxyType(dict(self.error)))

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "request_id": self.request_id,
            "response_id": self.response_id,
            "jsonrpc": self.jsonrpc,
            "method": self.method,
            "http_status": self.http_status,
            "result": self.result,
            "error": None if self.error is None else dict(self.error),
            "transport": self.transport.value,
            "has_result": self.has_result,
            "has_error": self.has_error,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "JsonRpcEnvelope":
        if not isinstance(payload, Mapping):
            raise ProtocolRepairError("jsonrpc envelope must be an object")
        _reject_forbidden_fields(payload, label="jsonrpc envelope")
        error = payload.get("error")
        return cls(
            request_id=payload.get("request_id", 0),
            response_id=payload.get("response_id"),
            jsonrpc=str(payload.get("jsonrpc") or ""),
            method=str(payload.get("method") or ""),
            http_status=payload.get("http_status", 200),
            result=payload.get("result"),
            error=error if isinstance(error, Mapping) or error is None else None,
            transport=payload.get("transport", TransportKind.HTTP),
            has_result=bool(payload.get("has_result", "result" in payload)),
            has_error=bool(payload.get("has_error", error is not None)),
        )


@dataclass(frozen=True)
class SchemaBinding(CanonicalContract):
    """One reviewed method → schema binding for request adapters."""

    SCHEMA: ClassVar[str] = SCHEMA_BINDING_SCHEMA
    INTERFACE: ClassVar[str] = PROTOCOL_REPAIR_OPERATORS_INTERFACE

    method: str
    request_schema_ref: str
    response_schema_ref: str
    profile: str = MCP_PROFILE_A
    authority: str = "reviewed"

    def __post_init__(self) -> None:
        method = _text(self.method, "method", identifier=True)
        object.__setattr__(self, "method", method)
        object.__setattr__(
            self, "request_schema_ref", _schema_ref(self.request_schema_ref, "request_schema_ref")
        )
        object.__setattr__(
            self,
            "response_schema_ref",
            _schema_ref(self.response_schema_ref, "response_schema_ref"),
        )
        object.__setattr__(self, "profile", _text(self.profile, "profile", identifier=True))
        authority = _text(self.authority, "authority", identifier=True)
        if authority not in {"reviewed", "production", "fixture"}:
            raise ProtocolRepairError("schema binding authority must be reviewed")
        object.__setattr__(self, "authority", authority)
        expected = REVIEWED_METHOD_SCHEMAS.get(method)
        if expected is None:
            raise ProtocolRepairError(f"unknown method for schema binding: {method}")
        if self.request_schema_ref != expected and not self.request_schema_ref.startswith(
            "schema:"
        ):
            raise ProtocolRepairError("request_schema_ref is not a reviewed schema")

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "method": self.method,
            "request_schema_ref": self.request_schema_ref,
            "response_schema_ref": self.response_schema_ref,
            "profile": self.profile,
            "authority": self.authority,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SchemaBinding":
        if not isinstance(payload, Mapping):
            raise ProtocolRepairError("schema binding must be an object")
        _reject_forbidden_fields(payload, label="schema binding")
        return cls(
            method=str(payload.get("method") or ""),
            request_schema_ref=str(payload.get("request_schema_ref") or ""),
            response_schema_ref=str(payload.get("response_schema_ref") or ""),
            profile=str(payload.get("profile") or MCP_PROFILE_A),
            authority=str(payload.get("authority") or "reviewed"),
        )


@dataclass(frozen=True)
class CidReceipt(CanonicalContract):
    """One artifact/receipt claim under local CID verification rules.

    Stores the claimed artifact identity as ``claimed_cid`` so it does not
    collide with :class:`CanonicalContract`'s content-address ``cid`` property.
    """

    SCHEMA: ClassVar[str] = CID_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = PROTOCOL_REPAIR_OPERATORS_INTERFACE

    claimed_cid: str
    found: bool
    server_verified: bool
    bytes_base64: str = ""
    receipt_cid: str = ""
    backend: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "claimed_cid", _cid(self.claimed_cid, "claimed_cid"))
        object.__setattr__(self, "found", _bool(self.found, "found"))
        object.__setattr__(
            self, "server_verified", _bool(self.server_verified, "server_verified")
        )
        object.__setattr__(
            self,
            "bytes_base64",
            _text(self.bytes_base64, "bytes_base64", required=False, maximum=MAX_TEXT_BYTES),
        )
        object.__setattr__(
            self,
            "receipt_cid",
            _cid(self.receipt_cid, "receipt_cid", required=False)
            if self.receipt_cid
            else "",
        )
        object.__setattr__(
            self, "backend", _text(self.backend, "backend", required=False, identifier=True)
        )

    @property
    def payload_bytes(self) -> bytes:
        if not self.bytes_base64:
            return b""
        return decode_bytes_base64(self.bytes_base64)

    def local_verified(self) -> bool:
        """Independently verify content identity; never trust server flags."""

        if not self.found:
            return False
        payload = self.payload_bytes
        if not payload:
            return False
        expected = local_cid_for_bytes(payload)
        if self.claimed_cid.startswith("sha256:"):
            if self.claimed_cid != expected:
                return False
        else:
            # Multiformat CIDs without a local multihash decode path fail closed
            # unless the claimed CID equals the local sha256 digest form used by
            # fixtures that bind content identity directly.
            if self.claimed_cid != expected and not self.claimed_cid.startswith(("bafy", "bagu", "bafk")):
                return False
            if self.claimed_cid.startswith(("bafy", "bagu", "bafk")):
                # Without codec-aware multihash verification, multiformat CIDs
                # cannot be elevated by a server verified flag.
                return False
        if self.receipt_cid:
            if self.receipt_cid.startswith("sha256:") and self.receipt_cid != expected:
                return False
            if self.receipt_cid.startswith(("bafy", "bagu", "bafk")):
                return False
        return self.claimed_cid.startswith("sha256:") and self.claimed_cid == expected

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "cid": self.claimed_cid,
            "found": self.found,
            "server_verified": self.server_verified,
            "bytes_base64": self.bytes_base64,
            "receipt_cid": self.receipt_cid,
            "backend": self.backend,
            "local_verified": self.local_verified(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CidReceipt":
        if not isinstance(payload, Mapping):
            raise ProtocolRepairError("cid receipt must be an object")
        _reject_forbidden_fields(payload, label="cid receipt")
        return cls(
            claimed_cid=str(payload.get("cid") or payload.get("claimed_cid") or ""),
            found=bool(payload.get("found", False)),
            server_verified=bool(payload.get("server_verified", False)),
            bytes_base64=str(payload.get("bytes_base64") or ""),
            receipt_cid=str(payload.get("receipt_cid") or ""),
            backend=str(payload.get("backend") or ""),
        )


@dataclass(frozen=True)
class ProfileNegotiation(CanonicalContract):
    """Client/server profile negotiation with required-profile enforcement."""

    SCHEMA: ClassVar[str] = PROFILE_NEGOTIATION_SCHEMA
    INTERFACE: ClassVar[str] = PROTOCOL_REPAIR_OPERATORS_INTERFACE

    requested_profiles: tuple[str, ...]
    offered_profiles: tuple[str, ...]
    required_profiles: tuple[str, ...] = ()
    client_transport: TransportKind = TransportKind.HTTP
    server_transport: TransportKind = TransportKind.HTTP
    policy_availability: PolicyAvailability = PolicyAvailability.AVAILABLE
    policy_decision: str = "allow"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requested_profiles",
            _string_tuple(self.requested_profiles, "requested_profiles", ordered=True),
        )
        object.__setattr__(
            self,
            "offered_profiles",
            _string_tuple(self.offered_profiles, "offered_profiles", ordered=True),
        )
        object.__setattr__(
            self,
            "required_profiles",
            _string_tuple(self.required_profiles, "required_profiles", ordered=True),
        )
        object.__setattr__(
            self,
            "client_transport",
            _enum(self.client_transport, TransportKind, "client_transport"),
        )
        object.__setattr__(
            self,
            "server_transport",
            _enum(self.server_transport, TransportKind, "server_transport"),
        )
        object.__setattr__(
            self,
            "policy_availability",
            _enum(self.policy_availability, PolicyAvailability, "policy_availability"),
        )
        object.__setattr__(
            self,
            "policy_decision",
            _text(self.policy_decision, "policy_decision", required=False, identifier=True)
            or "",
        )

    def negotiated_subset(self) -> tuple[str, ...]:
        offered = set(self.offered_profiles)
        return tuple(profile for profile in self.requested_profiles if profile in offered)

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "requested_profiles": list(self.requested_profiles),
            "offered_profiles": list(self.offered_profiles),
            "required_profiles": list(self.required_profiles),
            "negotiated_profiles": list(self.negotiated_subset()),
            "client_transport": self.client_transport.value,
            "server_transport": self.server_transport.value,
            "policy_availability": self.policy_availability.value,
            "policy_decision": self.policy_decision,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProfileNegotiation":
        if not isinstance(payload, Mapping):
            raise ProtocolRepairError("profile negotiation must be an object")
        _reject_forbidden_fields(payload, label="profile negotiation")
        return cls(
            requested_profiles=payload.get("requested_profiles") or (),
            offered_profiles=payload.get("offered_profiles") or (),
            required_profiles=payload.get("required_profiles") or (),
            client_transport=payload.get("client_transport", TransportKind.HTTP),
            server_transport=payload.get("server_transport", TransportKind.HTTP),
            policy_availability=payload.get(
                "policy_availability", PolicyAvailability.AVAILABLE
            ),
            policy_decision=str(payload.get("policy_decision") or ""),
        )


# ---------------------------------------------------------------------------
# Validation engines (shared by operators and vector materialization)
# ---------------------------------------------------------------------------


def validate_jsonrpc_envelope(envelope: JsonRpcEnvelope) -> tuple[ProtocolVerdict, tuple[str, ...]]:
    """Fail-closed JSON-RPC 2.0 + HTTP status validation."""

    if not isinstance(envelope, JsonRpcEnvelope):
        raise ProtocolRepairError("envelope must be a JsonRpcEnvelope")
    reasons: list[str] = []
    if envelope.http_status is not None and not (200 <= envelope.http_status < 300):
        reasons.append(ReasonCode.HTTP_ERROR.value)
    if envelope.jsonrpc != JSONRPC_VERSION:
        reasons.append(ReasonCode.WRONG_JSONRPC_VERSION.value)
    if envelope.response_id is None or envelope.response_id != envelope.request_id:
        reasons.append(ReasonCode.WRONG_ID.value)
    if envelope.has_result and envelope.has_error:
        reasons.append(ReasonCode.BOTH_RESULT_AND_ERROR.value)
    if not envelope.has_result and not envelope.has_error:
        reasons.append(ReasonCode.MISSING_RESULT_AND_ERROR.value)
    if reasons:
        return ProtocolVerdict.REJECT, tuple(reasons)
    return ProtocolVerdict.PASS, (ReasonCode.OK.value,)


def validate_schema_binding(
    binding: SchemaBinding,
    *,
    method: str | None = None,
) -> tuple[ProtocolVerdict, tuple[str, ...]]:
    """Fail-closed schema binding validation against the reviewed method table."""

    if not isinstance(binding, SchemaBinding):
        raise ProtocolRepairError("binding must be a SchemaBinding")
    target_method = method or binding.method
    expected = REVIEWED_METHOD_SCHEMAS.get(target_method)
    if expected is None:
        return ProtocolVerdict.REJECT, (ReasonCode.UNKNOWN_METHOD.value,)
    if binding.method != target_method:
        return ProtocolVerdict.REJECT, (ReasonCode.BAD_SCHEMA.value, "method_mismatch")
    if binding.request_schema_ref != expected:
        return ProtocolVerdict.REJECT, (ReasonCode.BAD_SCHEMA.value, "request_schema_mismatch")
    if binding.authority not in {"reviewed", "production", "fixture"}:
        return ProtocolVerdict.REJECT, (ReasonCode.INVENTED_SCHEMA.value,)
    if binding.profile not in {*MCP_PROFILES_A_F, MCP_PROFILE_BASIC}:
        return ProtocolVerdict.REJECT, (ReasonCode.UNSUPPORTED_PROFILE.value,)
    return ProtocolVerdict.PASS, (ReasonCode.OK.value,)


def validate_cid_receipt(receipt: CidReceipt) -> tuple[ProtocolVerdict, tuple[str, ...]]:
    """Local CID/receipt verification; server verified flags are never trusted."""

    if not isinstance(receipt, CidReceipt):
        raise ProtocolRepairError("receipt must be a CidReceipt")
    reasons: list[str] = []
    if not receipt.found:
        reasons.append(ReasonCode.BAD_RECEIPT.value)
        return ProtocolVerdict.REJECT, tuple(reasons)
    try:
        _ = receipt.payload_bytes
    except ProtocolRepairError:
        return ProtocolVerdict.REJECT, (ReasonCode.BAD_RECEIPT.value, "bad_bytes")
    if not receipt.claimed_cid or not _CID_RE.fullmatch(receipt.claimed_cid):
        reasons.append(ReasonCode.BAD_CID.value)
    local_ok = receipt.local_verified()
    if receipt.server_verified and not local_ok:
        reasons.append(ReasonCode.SERVER_VERIFIED_UNTRUSTED.value)
    if not local_ok:
        reasons.append(ReasonCode.LOCAL_CID_MISMATCH.value)
        reasons.append(ReasonCode.BAD_CID.value)
    # Deduplicate while preserving order.
    deduped: list[str] = []
    seen: set[str] = set()
    for code in reasons:
        if code not in seen:
            seen.add(code)
            deduped.append(code)
    if deduped:
        return ProtocolVerdict.REJECT, tuple(deduped)
    return ProtocolVerdict.PASS, (ReasonCode.OK.value,)


def validate_profile_negotiation(
    negotiation: ProfileNegotiation,
) -> tuple[ProtocolVerdict, tuple[str, ...], tuple[str, ...]]:
    """Capability subset negotiation with fail-closed policy and transport rules."""

    if not isinstance(negotiation, ProfileNegotiation):
        raise ProtocolRepairError("negotiation must be a ProfileNegotiation")
    reasons: list[str] = []
    negotiated = negotiation.negotiated_subset()

    if negotiation.client_transport is not negotiation.server_transport:
        reasons.append(ReasonCode.TRANSPORT_MISMATCH.value)

    for profile in negotiation.required_profiles:
        if profile not in negotiated:
            reasons.append(ReasonCode.UNSUPPORTED_PROFILE.value)
            reasons.append(ReasonCode.PROFILE_DOWNGRADE.value)
            break

    for profile in negotiated:
        if profile not in {*MCP_PROFILES_A_F, MCP_PROFILE_BASIC}:
            reasons.append(ReasonCode.UNSUPPORTED_PROFILE.value)
            break

    # Policy outage / missing decision denies — never allow.
    if negotiation.policy_availability is PolicyAvailability.OUTAGE:
        reasons.append(ReasonCode.POLICY_OUTAGE.value)
        if negotiation.policy_decision in {"allow", "allow_with_obligations"}:
            reasons.append(ReasonCode.POLICY_ALLOW_FROM_OUTAGE.value)
        return ProtocolVerdict.DENY, tuple(dict.fromkeys(reasons)), negotiated

    if negotiation.policy_availability in {
        PolicyAvailability.MISSING,
        PolicyAvailability.UNKNOWN,
    }:
        reasons.append(ReasonCode.MISSING_POLICY_DECISION.value)
        return ProtocolVerdict.DENY, tuple(dict.fromkeys(reasons)), negotiated

    if negotiation.policy_decision not in {"allow", "deny", "allow_with_obligations"}:
        reasons.append(ReasonCode.MISSING_POLICY_DECISION.value)
        return ProtocolVerdict.DENY, tuple(dict.fromkeys(reasons)), negotiated

    if reasons:
        # Transport / profile failures reject (not silent success).
        return ProtocolVerdict.REJECT, tuple(dict.fromkeys(reasons)), negotiated

    if negotiation.policy_decision == "deny":
        return ProtocolVerdict.DENY, (ReasonCode.OK.value, "policy_deny"), negotiated

    return ProtocolVerdict.PASS, (ReasonCode.OK.value,), negotiated


# ---------------------------------------------------------------------------
# Repair request / receipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProtocolRepairRequest(CanonicalContract):
    """One proposal-only protocol repair request."""

    SCHEMA: ClassVar[str] = PROTOCOL_REQUEST_SCHEMA
    INTERFACE: ClassVar[str] = PROTOCOL_REPAIR_OPERATORS_INTERFACE

    role: OperatorRole
    envelope: JsonRpcEnvelope | None = None
    schema_binding: SchemaBinding | None = None
    reviewed_schema_binding: SchemaBinding | None = None
    cid_receipt: CidReceipt | None = None
    profile_negotiation: ProfileNegotiation | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _enum(self.role, OperatorRole, "role"))
        if self.envelope is not None and not isinstance(self.envelope, JsonRpcEnvelope):
            raise ProtocolRepairError("envelope must be a JsonRpcEnvelope")
        if self.schema_binding is not None and not isinstance(
            self.schema_binding, SchemaBinding
        ):
            raise ProtocolRepairError("schema_binding must be a SchemaBinding")
        if self.reviewed_schema_binding is not None and not isinstance(
            self.reviewed_schema_binding, SchemaBinding
        ):
            raise ProtocolRepairError("reviewed_schema_binding must be a SchemaBinding")
        if self.cid_receipt is not None and not isinstance(self.cid_receipt, CidReceipt):
            raise ProtocolRepairError("cid_receipt must be a CidReceipt")
        if self.profile_negotiation is not None and not isinstance(
            self.profile_negotiation, ProfileNegotiation
        ):
            raise ProtocolRepairError("profile_negotiation must be a ProfileNegotiation")

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "role": self.role.value,
            "envelope": None if self.envelope is None else self.envelope.to_dict(),
            "schema_binding": (
                None if self.schema_binding is None else self.schema_binding.to_dict()
            ),
            "reviewed_schema_binding": (
                None
                if self.reviewed_schema_binding is None
                else self.reviewed_schema_binding.to_dict()
            ),
            "cid_receipt": None if self.cid_receipt is None else self.cid_receipt.to_dict(),
            "profile_negotiation": (
                None
                if self.profile_negotiation is None
                else self.profile_negotiation.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProtocolRepairRequest":
        if not isinstance(payload, Mapping):
            raise ProtocolRepairError("protocol repair request must be an object")
        _reject_forbidden_fields(payload, label="protocol repair request")

        def _opt(key: str, factory):
            value = payload.get(key)
            return None if value is None else factory(value)

        return cls(
            role=payload.get("role", OperatorRole.JSONRPC_VALIDATION),
            envelope=_opt("envelope", JsonRpcEnvelope.from_dict),
            schema_binding=_opt("schema_binding", SchemaBinding.from_dict),
            reviewed_schema_binding=_opt(
                "reviewed_schema_binding", SchemaBinding.from_dict
            ),
            cid_receipt=_opt("cid_receipt", CidReceipt.from_dict),
            profile_negotiation=_opt(
                "profile_negotiation", ProfileNegotiation.from_dict
            ),
        )


@dataclass(frozen=True)
class ProtocolRepairReceipt(CanonicalContract):
    """Proposal-only receipt for one protocol operator application."""

    SCHEMA: ClassVar[str] = PROTOCOL_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = PROTOCOL_REPAIR_OPERATORS_INTERFACE

    disposition: RepairDisposition
    role: OperatorRole
    operator_kind: str
    reason_codes: tuple[str, ...]
    verdict: ProtocolVerdict = ProtocolVerdict.REJECT
    negotiated_profiles: tuple[str, ...] = ()
    preview_schema_binding: SchemaBinding | None = None
    inverse_schema_binding: SchemaBinding | None = None
    local_verified: bool = False
    proposal_only: bool = True
    grants_write_authority: bool = False
    grants_proof_authority: bool = False
    semantic_authority: bool = False
    evidence_id: str = PROTOCOL_REPAIR_EVIDENCE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "disposition", _enum(self.disposition, RepairDisposition, "disposition")
        )
        object.__setattr__(self, "role", _enum(self.role, OperatorRole, "role"))
        object.__setattr__(
            self, "operator_kind", _text(self.operator_kind, "operator_kind", identifier=True)
        )
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        object.__setattr__(self, "verdict", _enum(self.verdict, ProtocolVerdict, "verdict"))
        object.__setattr__(
            self,
            "negotiated_profiles",
            _string_tuple(self.negotiated_profiles, "negotiated_profiles"),
        )
        object.__setattr__(self, "local_verified", _bool(self.local_verified, "local_verified"))
        if self.proposal_only is not True:
            raise ProtocolRepairError("receipts must remain proposal-only")
        for flag in (
            "grants_write_authority",
            "grants_proof_authority",
            "semantic_authority",
        ):
            if getattr(self, flag) is not False:
                raise ProtocolRepairError(f"{flag} cannot be true on a repair receipt")
            object.__setattr__(self, flag, False)
        object.__setattr__(self, "proposal_only", True)
        object.__setattr__(
            self, "evidence_id", _text(self.evidence_id, "evidence_id")
        )
        if self.evidence_id != PROTOCOL_REPAIR_EVIDENCE:
            raise ProtocolRepairError(
                f"evidence_id must be exactly {PROTOCOL_REPAIR_EVIDENCE}"
            )

    @property
    def is_editable(self) -> bool:
        return self.disposition is RepairDisposition.PREVIEW_READY

    def _payload(self) -> dict[str, Any]:
        return {
            "interface": self.INTERFACE,
            "evidence_id": self.evidence_id,
            "disposition": self.disposition.value,
            "role": self.role.value,
            "operator_kind": self.operator_kind,
            "reason_codes": list(self.reason_codes),
            "verdict": self.verdict.value,
            "negotiated_profiles": list(self.negotiated_profiles),
            "preview_schema_binding": (
                None
                if self.preview_schema_binding is None
                else self.preview_schema_binding.to_dict()
            ),
            "inverse_schema_binding": (
                None
                if self.inverse_schema_binding is None
                else self.inverse_schema_binding.to_dict()
            ),
            "local_verified": self.local_verified,
            "proposal_only": True,
            "grants_write_authority": False,
            "grants_proof_authority": False,
            "semantic_authority": False,
            "version": PROTOCOL_REPAIR_VERSION,
            "profiles": list(MCP_PROFILES_A_F),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProtocolRepairReceipt":
        if not isinstance(payload, Mapping):
            raise ProtocolRepairError("protocol repair receipt must be an object")
        _reject_forbidden_fields(payload, label="protocol repair receipt")

        def _opt(key: str, factory):
            value = payload.get(key)
            return None if value is None else factory(value)

        return cls(
            disposition=payload.get("disposition", RepairDisposition.REJECTED),
            role=payload.get("role", OperatorRole.JSONRPC_VALIDATION),
            operator_kind=str(
                payload.get("operator_kind") or OperatorKind.REPAIR_JSONRPC_SCHEMA.value
            ),
            reason_codes=payload.get("reason_codes") or ("rejected",),
            verdict=payload.get("verdict", ProtocolVerdict.REJECT),
            negotiated_profiles=payload.get("negotiated_profiles") or (),
            preview_schema_binding=_opt("preview_schema_binding", SchemaBinding.from_dict),
            inverse_schema_binding=_opt("inverse_schema_binding", SchemaBinding.from_dict),
            local_verified=bool(payload.get("local_verified", False)),
            proposal_only=payload.get("proposal_only", True),
            grants_write_authority=payload.get("grants_write_authority", False),
            grants_proof_authority=payload.get("grants_proof_authority", False),
            semantic_authority=payload.get("semantic_authority", False),
            evidence_id=payload.get("evidence_id", PROTOCOL_REPAIR_EVIDENCE),
        )


# ---------------------------------------------------------------------------
# Operator implementations
# ---------------------------------------------------------------------------


def _registry_descriptor(kind: OperatorKind):
    registry = build_default_operator_registry()
    return registry.require_known(kind)


def _base_receipt(
    *,
    role: OperatorRole,
    disposition: RepairDisposition,
    reasons: Sequence[str],
    verdict: ProtocolVerdict,
    negotiated_profiles: Sequence[str] = (),
    preview_schema_binding: SchemaBinding | None = None,
    inverse_schema_binding: SchemaBinding | None = None,
    local_verified: bool = False,
) -> ProtocolRepairReceipt:
    kind = _ROLE_TO_KIND[role]
    return ProtocolRepairReceipt(
        disposition=disposition,
        role=role,
        operator_kind=kind.value,
        reason_codes=tuple(reasons) or (disposition.value,),
        verdict=verdict,
        negotiated_profiles=tuple(negotiated_profiles),
        preview_schema_binding=preview_schema_binding,
        inverse_schema_binding=inverse_schema_binding,
        local_verified=local_verified,
    )


def _guard_registry(role: OperatorRole) -> ProtocolRepairReceipt | None:
    kind = _ROLE_TO_KIND[role]
    descriptor = _registry_descriptor(kind)
    if descriptor.kind is not kind:
        return _base_receipt(
            role=role,
            disposition=RepairDisposition.REJECTED,
            reasons=("registry_kind_mismatch",),
            verdict=ProtocolVerdict.REJECT,
        )
    if descriptor.family is not OperatorFamily.PROTOCOL:
        return _base_receipt(
            role=role,
            disposition=RepairDisposition.REJECTED,
            reasons=("registry_family_mismatch",),
            verdict=ProtocolVerdict.REJECT,
        )
    if descriptor.proposal_only is not True or descriptor.grants_write_authority:
        return _base_receipt(
            role=role,
            disposition=RepairDisposition.REJECTED,
            reasons=("descriptor_authority_violation",),
            verdict=ProtocolVerdict.REJECT,
        )
    return None


class JsonRpcValidationOperator:
    """Validate JSON-RPC 2.0 status/version/ID/result-error envelopes."""

    ROLE: ClassVar[OperatorRole] = OperatorRole.JSONRPC_VALIDATION
    INTERFACE: ClassVar[str] = PROTOCOL_REPAIR_OPERATORS_INTERFACE

    def __init__(self) -> None:
        self.descriptor = _registry_descriptor(OperatorKind.REPAIR_JSONRPC_SCHEMA)

    @property
    def operator_id(self) -> str:
        return f"dcr-operator:{self.ROLE.value}@1"

    def apply(self, request: ProtocolRepairRequest) -> ProtocolRepairReceipt:
        if not isinstance(request, ProtocolRepairRequest):
            raise ProtocolRepairError("request must be a ProtocolRepairRequest")
        blocked = _guard_registry(self.ROLE)
        if blocked is not None:
            return blocked
        if request.envelope is None:
            return _base_receipt(
                role=self.ROLE,
                disposition=RepairDisposition.ABSTAIN,
                reasons=("missing_envelope", ReasonCode.MALFORMED_ENVELOPE.value),
                verdict=ProtocolVerdict.REJECT,
            )
        verdict, reasons = validate_jsonrpc_envelope(request.envelope)
        if verdict is ProtocolVerdict.PASS:
            return _base_receipt(
                role=self.ROLE,
                disposition=RepairDisposition.ACCEPTED,
                reasons=reasons,
                verdict=verdict,
            )
        return _base_receipt(
            role=self.ROLE,
            disposition=RepairDisposition.REJECTED,
            reasons=reasons,
            verdict=verdict,
        )

    def preview(self, request: ProtocolRepairRequest) -> ProtocolRepairReceipt:
        return self.apply(request)

    def inverse(self, receipt: ProtocolRepairReceipt) -> None:
        if not isinstance(receipt, ProtocolRepairReceipt):
            raise ProtocolRepairError("receipt must be a ProtocolRepairReceipt")
        return None


class SchemaBindingOperator:
    """Bind reviewed request/response schemas without inventing adapters."""

    ROLE: ClassVar[OperatorRole] = OperatorRole.SCHEMA_BINDING
    INTERFACE: ClassVar[str] = PROTOCOL_REPAIR_OPERATORS_INTERFACE

    def __init__(self) -> None:
        self.descriptor = _registry_descriptor(OperatorKind.REPAIR_REQUEST_ADAPTER)

    @property
    def operator_id(self) -> str:
        return f"dcr-operator:{self.ROLE.value}@1"

    def apply(self, request: ProtocolRepairRequest) -> ProtocolRepairReceipt:
        if not isinstance(request, ProtocolRepairRequest):
            raise ProtocolRepairError("request must be a ProtocolRepairRequest")
        blocked = _guard_registry(self.ROLE)
        if blocked is not None:
            return blocked
        reviewed = request.reviewed_schema_binding
        if reviewed is None:
            return _base_receipt(
                role=self.ROLE,
                disposition=RepairDisposition.ABSTAIN,
                reasons=(
                    "missing_reviewed_schema_binding",
                    ReasonCode.INVENTED_SCHEMA.value,
                ),
                verdict=ProtocolVerdict.REJECT,
            )
        verdict, reasons = validate_schema_binding(reviewed)
        if verdict is not ProtocolVerdict.PASS:
            return _base_receipt(
                role=self.ROLE,
                disposition=RepairDisposition.REJECTED,
                reasons=reasons,
                verdict=verdict,
            )
        current = request.schema_binding
        if current is not None and current.content_id == reviewed.content_id:
            return _base_receipt(
                role=self.ROLE,
                disposition=RepairDisposition.ALREADY_ALIGNED,
                reasons=(ReasonCode.ALREADY_ALIGNED.value, *reasons),
                verdict=ProtocolVerdict.PASS,
                preview_schema_binding=reviewed,
                inverse_schema_binding=current,
            )
        return _base_receipt(
            role=self.ROLE,
            disposition=RepairDisposition.PREVIEW_READY,
            reasons=(ReasonCode.PREVIEW_READY.value, "schema_binding_restored", *reasons),
            verdict=ProtocolVerdict.PASS,
            preview_schema_binding=reviewed,
            inverse_schema_binding=current,
        )

    def preview(self, request: ProtocolRepairRequest) -> ProtocolRepairReceipt:
        return self.apply(request)

    def inverse(self, receipt: ProtocolRepairReceipt) -> SchemaBinding | None:
        if not isinstance(receipt, ProtocolRepairReceipt):
            raise ProtocolRepairError("receipt must be a ProtocolRepairReceipt")
        return receipt.inverse_schema_binding


class CanonicalCidOperator:
    """Local CID/receipt verification; never trusts server verified flags."""

    ROLE: ClassVar[OperatorRole] = OperatorRole.CANONICAL_CID
    INTERFACE: ClassVar[str] = PROTOCOL_REPAIR_OPERATORS_INTERFACE

    def __init__(self) -> None:
        self.descriptor = _registry_descriptor(OperatorKind.REPAIR_ERROR_ENVELOPE)

    @property
    def operator_id(self) -> str:
        return f"dcr-operator:{self.ROLE.value}@1"

    def apply(self, request: ProtocolRepairRequest) -> ProtocolRepairReceipt:
        if not isinstance(request, ProtocolRepairRequest):
            raise ProtocolRepairError("request must be a ProtocolRepairRequest")
        blocked = _guard_registry(self.ROLE)
        if blocked is not None:
            return blocked
        if request.cid_receipt is None:
            return _base_receipt(
                role=self.ROLE,
                disposition=RepairDisposition.ABSTAIN,
                reasons=("missing_cid_receipt", ReasonCode.BAD_RECEIPT.value),
                verdict=ProtocolVerdict.REJECT,
            )
        verdict, reasons = validate_cid_receipt(request.cid_receipt)
        local_ok = request.cid_receipt.local_verified()
        if verdict is ProtocolVerdict.PASS and local_ok:
            return _base_receipt(
                role=self.ROLE,
                disposition=RepairDisposition.ACCEPTED,
                reasons=reasons,
                verdict=verdict,
                local_verified=True,
            )
        return _base_receipt(
            role=self.ROLE,
            disposition=RepairDisposition.REJECTED,
            reasons=reasons,
            verdict=verdict,
            local_verified=False,
        )

    def preview(self, request: ProtocolRepairRequest) -> ProtocolRepairReceipt:
        return self.apply(request)

    def inverse(self, receipt: ProtocolRepairReceipt) -> None:
        if not isinstance(receipt, ProtocolRepairReceipt):
            raise ProtocolRepairError("receipt must be a ProtocolRepairReceipt")
        return None


class ProfileNegotiationOperator:
    """Capability subset negotiation with fail-closed policy/transport rules."""

    ROLE: ClassVar[OperatorRole] = OperatorRole.PROFILE_NEGOTIATION
    INTERFACE: ClassVar[str] = PROTOCOL_REPAIR_OPERATORS_INTERFACE

    def __init__(self) -> None:
        self.descriptor = _registry_descriptor(OperatorKind.REPAIR_PROFILE_BINDING)

    @property
    def operator_id(self) -> str:
        return f"dcr-operator:{self.ROLE.value}@1"

    def apply(self, request: ProtocolRepairRequest) -> ProtocolRepairReceipt:
        if not isinstance(request, ProtocolRepairRequest):
            raise ProtocolRepairError("request must be a ProtocolRepairRequest")
        blocked = _guard_registry(self.ROLE)
        if blocked is not None:
            return blocked
        if request.profile_negotiation is None:
            return _base_receipt(
                role=self.ROLE,
                disposition=RepairDisposition.ABSTAIN,
                reasons=("missing_profile_negotiation", ReasonCode.UNSUPPORTED_PROFILE.value),
                verdict=ProtocolVerdict.REJECT,
            )
        verdict, reasons, negotiated = validate_profile_negotiation(
            request.profile_negotiation
        )
        if verdict is ProtocolVerdict.PASS:
            return _base_receipt(
                role=self.ROLE,
                disposition=RepairDisposition.ACCEPTED,
                reasons=reasons,
                verdict=verdict,
                negotiated_profiles=negotiated,
            )
        if verdict is ProtocolVerdict.DENY:
            return _base_receipt(
                role=self.ROLE,
                disposition=RepairDisposition.DENIED,
                reasons=reasons,
                verdict=verdict,
                negotiated_profiles=negotiated,
            )
        return _base_receipt(
            role=self.ROLE,
            disposition=RepairDisposition.REJECTED,
            reasons=reasons,
            verdict=verdict,
            negotiated_profiles=negotiated,
        )

    def preview(self, request: ProtocolRepairRequest) -> ProtocolRepairReceipt:
        return self.apply(request)

    def inverse(self, receipt: ProtocolRepairReceipt) -> None:
        if not isinstance(receipt, ProtocolRepairReceipt):
            raise ProtocolRepairError("receipt must be a ProtocolRepairReceipt")
        return None


class ProtocolRepairOperators:
    """Facade bundling the four DCR-042 protocol repair operators."""

    INTERFACE: ClassVar[str] = PROTOCOL_REPAIR_OPERATORS_INTERFACE
    EVIDENCE_ID: ClassVar[str] = PROTOCOL_REPAIR_EVIDENCE

    def __init__(
        self,
        *,
        jsonrpc_validation: JsonRpcValidationOperator | None = None,
        schema_binding: SchemaBindingOperator | None = None,
        canonical_cid: CanonicalCidOperator | None = None,
        profile_negotiation: ProfileNegotiationOperator | None = None,
    ) -> None:
        self.jsonrpc_validation = jsonrpc_validation or JsonRpcValidationOperator()
        self.schema_binding = schema_binding or SchemaBindingOperator()
        self.canonical_cid = canonical_cid or CanonicalCidOperator()
        self.profile_negotiation = profile_negotiation or ProfileNegotiationOperator()

    def to_artifact_dict(self) -> dict[str, Any]:
        payload = {
            "schema": PROTOCOL_OPERATOR_VECTORS_SCHEMA,
            "interface": self.INTERFACE,
            "evidence_id": self.EVIDENCE_ID,
            "roles": [role.value for role in OperatorRole],
            "operator_kinds": [kind.value for kind in _ROLE_TO_KIND.values()],
            "profiles": list(MCP_PROFILES_A_F),
            "proposal_only": True,
            "grants_write_authority": False,
            "grants_proof_authority": False,
            "semantic_authority": False,
            "version": PROTOCOL_REPAIR_VERSION,
        }
        return {**payload, "artifact_digest": _digest(payload)}


def build_protocol_repair_operators() -> ProtocolRepairOperators:
    """Construct the sealed DCR-042 protocol operator set."""

    return ProtocolRepairOperators()


def reviewed_schema_binding_for(method: str) -> SchemaBinding:
    """Return the reviewed schema binding for a known method."""

    method_text = _text(method, "method", identifier=True)
    schema_ref = REVIEWED_METHOD_SCHEMAS.get(method_text)
    if schema_ref is None:
        raise ProtocolRepairError(f"unknown method: {method_text}")
    return SchemaBinding(
        method=method_text,
        request_schema_ref=schema_ref,
        response_schema_ref=schema_ref.replace("@1", "/response@1")
        if schema_ref.endswith("@1")
        else f"{schema_ref}/response",
        profile=MCP_PROFILE_A if method_text.startswith("interfaces/") else (
            MCP_PROFILE_B if method_text.startswith("mcp++/") else MCP_PROFILE_BASIC
        ),
        authority="reviewed",
    )


def materialize_protocol_operator_vectors() -> dict[str, Any]:
    """Emit compact deterministic negative/positive vectors for acceptance evidence.

    Negative vectors cover HTTP errors, wrong IDs/version, bad schemas/CIDs/
    receipts, unsupported profiles, policy outage, and transport mismatch.
    """

    operators = build_protocol_repair_operators()
    payload_bytes = b'{"ok":true}'
    good_cid = local_cid_for_bytes(payload_bytes)
    good_b64 = base64.b64encode(payload_bytes).decode("ascii")

    cases: list[dict[str, Any]] = []

    def _run(name: str, role: OperatorRole, request: ProtocolRepairRequest) -> None:
        if role is OperatorRole.JSONRPC_VALIDATION:
            receipt = operators.jsonrpc_validation.apply(request)
        elif role is OperatorRole.SCHEMA_BINDING:
            receipt = operators.schema_binding.apply(request)
        elif role is OperatorRole.CANONICAL_CID:
            receipt = operators.canonical_cid.apply(request)
        else:
            receipt = operators.profile_negotiation.apply(request)
        cases.append(
            {
                "name": name,
                "role": role.value,
                "disposition": receipt.disposition.value,
                "verdict": receipt.verdict.value,
                "reason_codes": list(receipt.reason_codes),
                "local_verified": receipt.local_verified,
                "negotiated_profiles": list(receipt.negotiated_profiles),
                "receipt_identity": receipt.content_id,
            }
        )

    # Positive: valid JSON-RPC success envelope.
    _run(
        "jsonrpc_ok",
        OperatorRole.JSONRPC_VALIDATION,
        ProtocolRepairRequest(
            role=OperatorRole.JSONRPC_VALIDATION,
            envelope=JsonRpcEnvelope(
                request_id=1,
                response_id=1,
                jsonrpc=JSONRPC_VERSION,
                method="tools/list",
                http_status=200,
                result={"tools": []},
                has_result=True,
            ),
        ),
    )
    # Negative: HTTP error.
    _run(
        "http_error",
        OperatorRole.JSONRPC_VALIDATION,
        ProtocolRepairRequest(
            role=OperatorRole.JSONRPC_VALIDATION,
            envelope=JsonRpcEnvelope(
                request_id=1,
                response_id=1,
                jsonrpc=JSONRPC_VERSION,
                method="tools/list",
                http_status=503,
                result={"tools": []},
                has_result=True,
            ),
        ),
    )
    # Negative: wrong version.
    _run(
        "wrong_version",
        OperatorRole.JSONRPC_VALIDATION,
        ProtocolRepairRequest(
            role=OperatorRole.JSONRPC_VALIDATION,
            envelope=JsonRpcEnvelope(
                request_id=1,
                response_id=1,
                jsonrpc="1.0",
                method="tools/list",
                result={},
                has_result=True,
            ),
        ),
    )
    # Negative: wrong id.
    _run(
        "wrong_id",
        OperatorRole.JSONRPC_VALIDATION,
        ProtocolRepairRequest(
            role=OperatorRole.JSONRPC_VALIDATION,
            envelope=JsonRpcEnvelope(
                request_id=7,
                response_id=8,
                jsonrpc=JSONRPC_VERSION,
                method="tools/list",
                result={},
                has_result=True,
            ),
        ),
    )
    # Negative: both result and error.
    _run(
        "both_result_and_error",
        OperatorRole.JSONRPC_VALIDATION,
        ProtocolRepairRequest(
            role=OperatorRole.JSONRPC_VALIDATION,
            envelope=JsonRpcEnvelope(
                request_id=1,
                response_id=1,
                jsonrpc=JSONRPC_VERSION,
                method="tools/call",
                result={"ok": True},
                error={"code": -32000, "message": "nope"},
                has_result=True,
                has_error=True,
            ),
        ),
    )
    # Negative: bad schema / unknown method invents nothing.
    _run(
        "bad_schema",
        OperatorRole.SCHEMA_BINDING,
        ProtocolRepairRequest(
            role=OperatorRole.SCHEMA_BINDING,
            reviewed_schema_binding=None,
            schema_binding=None,
        ),
    )
    # Positive: reviewed schema binding.
    reviewed = reviewed_schema_binding_for("tools/call")
    _run(
        "schema_ok",
        OperatorRole.SCHEMA_BINDING,
        ProtocolRepairRequest(
            role=OperatorRole.SCHEMA_BINDING,
            reviewed_schema_binding=reviewed,
            schema_binding=None,
        ),
    )
    # Negative: server verified flag with mismatched CID.
    _run(
        "bad_cid_server_verified",
        OperatorRole.CANONICAL_CID,
        ProtocolRepairRequest(
            role=OperatorRole.CANONICAL_CID,
            cid_receipt=CidReceipt(claimed_cid="sha256:" + ("ab" * 32),
                found=True,
                server_verified=True,
                bytes_base64=good_b64,
            ),
        ),
    )
    # Positive: local CID verification.
    _run(
        "cid_ok",
        OperatorRole.CANONICAL_CID,
        ProtocolRepairRequest(
            role=OperatorRole.CANONICAL_CID,
            cid_receipt=CidReceipt(claimed_cid=good_cid,
                found=True,
                server_verified=False,
                bytes_base64=good_b64,
            ),
        ),
    )
    # Negative: unsupported required profile.
    _run(
        "unsupported_profile",
        OperatorRole.PROFILE_NEGOTIATION,
        ProtocolRepairRequest(
            role=OperatorRole.PROFILE_NEGOTIATION,
            profile_negotiation=ProfileNegotiation(
                requested_profiles=(MCP_PROFILE_B, MCP_PROFILE_D),
                offered_profiles=(MCP_PROFILE_B,),
                required_profiles=(MCP_PROFILE_D,),
                policy_decision="allow",
            ),
        ),
    )
    # Negative: policy outage never becomes allow.
    _run(
        "policy_outage",
        OperatorRole.PROFILE_NEGOTIATION,
        ProtocolRepairRequest(
            role=OperatorRole.PROFILE_NEGOTIATION,
            profile_negotiation=ProfileNegotiation(
                requested_profiles=(MCP_PROFILE_D,),
                offered_profiles=(MCP_PROFILE_D,),
                required_profiles=(MCP_PROFILE_D,),
                policy_availability=PolicyAvailability.OUTAGE,
                policy_decision="allow",
            ),
        ),
    )
    # Negative: transport mismatch.
    _run(
        "transport_mismatch",
        OperatorRole.PROFILE_NEGOTIATION,
        ProtocolRepairRequest(
            role=OperatorRole.PROFILE_NEGOTIATION,
            profile_negotiation=ProfileNegotiation(
                requested_profiles=(MCP_PROFILE_E,),
                offered_profiles=(MCP_PROFILE_E,),
                required_profiles=(MCP_PROFILE_E,),
                client_transport=TransportKind.HTTP,
                server_transport=TransportKind.LIBP2P,
                policy_decision="allow",
            ),
        ),
    )
    # Positive: subset negotiation.
    _run(
        "profile_ok",
        OperatorRole.PROFILE_NEGOTIATION,
        ProtocolRepairRequest(
            role=OperatorRole.PROFILE_NEGOTIATION,
            profile_negotiation=ProfileNegotiation(
                requested_profiles=MCP_PROFILES_A_F,
                offered_profiles=(MCP_PROFILE_A, MCP_PROFILE_B, MCP_PROFILE_F),
                required_profiles=(MCP_PROFILE_A, MCP_PROFILE_B),
                policy_decision="allow",
            ),
        ),
    )

    artifact = operators.to_artifact_dict()
    body = {
        "schema": PROTOCOL_OPERATOR_VECTORS_SCHEMA,
        "interface": PROTOCOL_REPAIR_OPERATORS_INTERFACE,
        "evidence_id": PROTOCOL_REPAIR_EVIDENCE,
        "profiles": list(MCP_PROFILES_A_F),
        "cases": cases,
        "negative_coverage": sorted(
            {
                "http_error",
                "wrong_id",
                "wrong_version",
                "bad_schema",
                "bad_cid",
                "bad_receipt",
                "unsupported_profile",
                "policy_outage",
                "transport_mismatch",
            }
        ),
        "server_verified_trusted": False,
        "policy_outage_denies": True,
        "artifact": artifact,
    }
    body["vector_digest"] = _digest(
        {key: value for key, value in body.items() if key != "vector_digest"}
    )
    body["content_id"] = content_identity(
        {key: value for key, value in body.items() if key not in {"content_id", "vector_digest"}}
    )
    return body


__all__ = (
    "PROTOCOL_REPAIR_OPERATORS_INTERFACE",
    "PROTOCOL_REPAIR_EVIDENCE",
    "JSONRPC_VERSION",
    "MCP_PROFILE_A",
    "MCP_PROFILE_B",
    "MCP_PROFILE_C",
    "MCP_PROFILE_D",
    "MCP_PROFILE_E",
    "MCP_PROFILE_F",
    "MCP_PROFILES_A_F",
    "ProtocolRepairError",
    "ProtocolRepairAbstention",
    "RepairDisposition",
    "OperatorRole",
    "ProtocolVerdict",
    "ReasonCode",
    "PolicyAvailability",
    "TransportKind",
    "JsonRpcEnvelope",
    "SchemaBinding",
    "CidReceipt",
    "ProfileNegotiation",
    "ProtocolRepairRequest",
    "ProtocolRepairReceipt",
    "JsonRpcValidationOperator",
    "SchemaBindingOperator",
    "CanonicalCidOperator",
    "ProfileNegotiationOperator",
    "ProtocolRepairOperators",
    "build_protocol_repair_operators",
    "validate_jsonrpc_envelope",
    "validate_schema_binding",
    "validate_cid_receipt",
    "validate_profile_negotiation",
    "local_cid_for_bytes",
    "local_cid_for_json",
    "reviewed_schema_binding_for",
    "materialize_protocol_operator_vectors",
    "REVIEWED_METHOD_SCHEMAS",
)
