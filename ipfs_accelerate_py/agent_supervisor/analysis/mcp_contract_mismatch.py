"""Classify and deduplicate deterministic MCP contract repair findings (DCR-024).

Interfaces
----------
* ``ContractMismatch@1`` — one earliest-broken-edge finding with typed class.
* ``RepairFindingKey@1`` — semantic dedupe key for repair findings.

Inputs (same epoch):
* DCR-021 ``McpContractGraph@1`` (declared graph + typed blockers)
* DCR-023 ``LiveContractTranscript@1`` (live observations)

Normative rules (fail-closed):

* Findings are keyed by package, operation, direction, schema/profile/transport
  roots, mismatch class, edge kind, and snapshot.  Duplicate dag.put/get-style
  rows collapse **only** when those semantic keys match.
* Independent protocol, schema, authority, liveness, identity, mediation, and
  implementation defects are preserved even when surface names coincide.
* ``expected_only``, ``missing``, ``ambiguous``, and ``unobserved`` findings
  remain nonpassing; they never promote to a green fixed point.
* Earliest broken edge is selected along the mandatory consumer path order
  (declaration → registration → dispatcher → handler → effect → response).
* Graph and live transcript must share one epoch identity; mixed epochs fail
  closed rather than inventing a merged backlog.

Evidence term: ``dcr/mismatch@1``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_verification_contracts import (
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)
from .mcp_contract_graph import (
    MANDATORY_EDGE_KINDS,
    BlockerKind,
    ContractBlocker,
    ContractEdge,
    ContractNodeKind,
    McpContractGraph,
    McpContractGraphError,
    load_mcp_contract_graph,
    materialize_mcp_contract_graph,
)
from .mcp_contract_identity import ContractDirection
from .mcp_live_observer import (
    LIVE_CONTRACT_TRANSCRIPT_SCHEMA,
    LIVE_OBSERVATION_EVIDENCE_TERM,
    load_mcp_live_transcript,
)

# ---------------------------------------------------------------------------
# Schemas / interfaces / constants
# ---------------------------------------------------------------------------

CONTRACT_MISMATCH_INTERFACE: Final = "ContractMismatch@1"
REPAIR_FINDING_KEY_INTERFACE: Final = "RepairFindingKey@1"
CONTRACT_MISMATCH_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/contract-mismatch@1"
)
REPAIR_FINDING_KEY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/repair-finding-key@1"
)
MISMATCH_FINDINGS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/mcp-contract-mismatch-findings@1"
)
MISMATCH_FINDINGS_INTERFACE: Final = "McpContractMismatchFindings@1"
MISMATCH_EVIDENCE_TERM: Final = "dcr/mismatch@1"
CONTRACT_VERSION: Final[int] = 1
FINDINGS_VERSION: Final = "1"

DCR_TASK_ID: Final = "DCR-024"
DCR_ARTIFACT_PATH: Final = (
    "data/agent_supervisor/deterministic_contract_repair/"
    "mcp_contract_mismatch_findings.json"
)
DEFAULT_MAX_BYTES: Final[int] = 1_048_576
_MAX_FIELD_BYTES: Final[int] = 4_096
_MAX_FINDINGS: Final[int] = 50_000

# Edge kinds that sit on the declaration / mediation side of the path.
_MEDIATION_EDGE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "ui_action_to_descriptor",
        "descriptor_to_orb_idl",
        "orb_idl_to_mcp_method_schema",
        "mcp_method_schema_to_mediator",
        "mediator_to_route",
    }
)
_IMPLEMENTATION_EDGE_KINDS: Final[frozenset[str]] = frozenset(
    {
        "route_to_dispatcher",
        "dispatcher_to_handler",
        "handler_to_effect",
        "effect_to_receipt",
        "receipt_to_runtime_identity",
    }
)

# Nonpassing classes that can never be treated as a green fixed point.
NONPASSING_MISMATCH_CLASSES: Final[frozenset[str]] = frozenset(
    {
        "expected_only",
        "missing",
        "ambiguous",
        "unobserved",
    }
)

# Package → live observation role for epoch correlation.
_PACKAGE_TO_ROLE: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ipfs_accelerate_py": "accelerate",
        "ipfs_datasets_py": "datasets",
        "ipfs_kit_py": "kit",
    }
)


class McpContractMismatchError(ValueError):
    """Mismatch classification input is malformed or violates a closed rule."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "mcp_contract_mismatch_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = dict(details or {})


class MismatchClass(str, Enum):
    """Closed vocabulary of independent contract defect classes.

    Independent classes are preserved even when surface names coincide so that
    a protocol defect never collapses into a schema defect, etc.
    """

    PROTOCOL = "protocol"
    SCHEMA = "schema"
    AUTHORITY = "authority"
    LIVENESS = "liveness"
    IDENTITY = "identity"
    MEDIATION = "mediation"
    IMPLEMENTATION = "implementation"
    EXPECTED_ONLY = "expected_only"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"
    UNOBSERVED = "unobserved"


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _norm_text(
    value: Any,
    *,
    field_name: str,
    required: bool = False,
    allow_empty: bool = False,
) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        raise McpContractMismatchError(
            f"{field_name} must be a string",
            reason_code="invalid_field_type",
            details={"field": field_name, "type": type(value).__name__},
        )
    if required and not text and not allow_empty:
        raise McpContractMismatchError(
            f"{field_name} is required",
            reason_code="missing_required_field",
            details={"field": field_name},
        )
    if len(text.encode("utf-8")) > _MAX_FIELD_BYTES:
        raise McpContractMismatchError(
            f"{field_name} exceeds the {_MAX_FIELD_BYTES}-byte limit",
            reason_code="field_too_large",
            details={"field": field_name},
        )
    return text


def _norm_enum(value: Any, enum_cls: type[Enum], *, field_name: str) -> Enum:
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return enum_cls(value.strip())
        except ValueError as exc:
            raise McpContractMismatchError(
                f"unknown {field_name}: {value!r}",
                reason_code="unknown_enum_value",
                details={"field": field_name, "value": value},
            ) from exc
    raise McpContractMismatchError(
        f"{field_name} must be a valid {enum_cls.__name__}",
        reason_code="invalid_enum",
        details={"field": field_name},
    )


def _plain(value: Any, *, depth: int = 0) -> Any:
    if depth > 24:
        raise McpContractMismatchError(
            "mismatch record exceeds nesting bound",
            reason_code="bounds_exceeded",
        )
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise McpContractMismatchError(
            "floating values are not canonical mismatch data",
            reason_code="non_canonical_json",
        )
    if isinstance(value, Mapping):
        if len(value) > 1_024 or not all(isinstance(key, str) for key in value):
            raise McpContractMismatchError(
                "mismatch mappings require at most 1024 string keys",
                reason_code="bounds_exceeded",
            )
        return {key: _plain(value[key], depth=depth + 1) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > 16_384:
            raise McpContractMismatchError(
                "mismatch sequence is oversized",
                reason_code="bounds_exceeded",
            )
        return [_plain(item, depth=depth + 1) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict(), depth=depth + 1)
    raise McpContractMismatchError(
        f"unsupported mismatch value: {type(value).__name__}",
        reason_code="non_canonical_json",
    )


def canonical_mismatch_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes for a mismatch artifact."""

    try:
        return canonical_json_bytes(_plain(value))
    except ContractValidationError as exc:
        raise McpContractMismatchError(
            "value is not canonical-JSON encodable",
            reason_code="non_canonical_json",
            details={"cause": str(exc)},
        ) from exc


def canonical_mismatch_cid(value: Any) -> str:
    """Return CIDv1 dag-json/sha2-256 for *value*."""

    try:
        return content_identity(_plain(value))
    except ContractValidationError as exc:
        raise McpContractMismatchError(
            "value is not canonical-JSON encodable",
            reason_code="non_canonical_json",
            details={"cause": str(exc)},
        ) from exc


def digest_for_canonical_bytes(data: bytes) -> str:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise McpContractMismatchError(
            "canonical bytes must be bytes-like",
            reason_code="invalid_field_type",
        )
    return "sha256:" + hashlib.sha256(bytes(data)).hexdigest()


def _default_workspace() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[5], here.parents[4], Path.cwd()):
        marker = candidate / "config" / "deterministic_contract_repair_services.json"
        if marker.is_file():
            return candidate
    return Path.cwd()


# ---------------------------------------------------------------------------
# RepairFindingKey@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RepairFindingKey:
    """Semantic dedupe key for one repair finding.

    Interface: ``RepairFindingKey@1``

    Rows collapse only when every field of this key matches.  Distinct
    directions (e.g. dag.put vs dag.get), classes, or edge kinds stay separate.
    """

    package: str
    operation: str
    direction: ContractDirection
    schema_root: str
    profile: str
    transport: str
    mismatch_class: MismatchClass
    edge_kind: str
    snapshot_id: str
    schema: str = REPAIR_FINDING_KEY_SCHEMA
    interface: str = REPAIR_FINDING_KEY_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "package",
            _norm_text(self.package, field_name="package", required=True),
        )
        object.__setattr__(
            self,
            "operation",
            _norm_text(self.operation, field_name="operation", required=True),
        )
        object.__setattr__(
            self,
            "direction",
            _norm_enum(self.direction, ContractDirection, field_name="direction"),
        )
        object.__setattr__(
            self,
            "schema_root",
            _norm_text(self.schema_root, field_name="schema_root", required=True),
        )
        object.__setattr__(
            self,
            "profile",
            _norm_text(self.profile, field_name="profile", required=True),
        )
        object.__setattr__(
            self,
            "transport",
            _norm_text(self.transport, field_name="transport", required=True),
        )
        object.__setattr__(
            self,
            "mismatch_class",
            _norm_enum(
                self.mismatch_class, MismatchClass, field_name="mismatch_class"
            ),
        )
        object.__setattr__(
            self,
            "edge_kind",
            _norm_text(self.edge_kind, field_name="edge_kind", required=True),
        )
        object.__setattr__(
            self,
            "snapshot_id",
            _norm_text(self.snapshot_id, field_name="snapshot_id", required=True),
        )
        object.__setattr__(
            self,
            "schema",
            _norm_text(self.schema, field_name="schema", required=True),
        )
        object.__setattr__(
            self,
            "interface",
            _norm_text(self.interface, field_name="interface", required=True),
        )
        if self.schema != REPAIR_FINDING_KEY_SCHEMA:
            raise McpContractMismatchError(
                "unsupported repair finding key schema",
                reason_code="unsupported_schema",
            )
        if self.interface != REPAIR_FINDING_KEY_INTERFACE:
            raise McpContractMismatchError(
                "unsupported repair finding key interface",
                reason_code="unsupported_interface",
            )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "package": self.package,
            "operation": self.operation,
            "direction": self.direction.value
            if isinstance(self.direction, ContractDirection)
            else str(self.direction),
            "schema_root": self.schema_root,
            "profile": self.profile,
            "transport": self.transport,
            "mismatch_class": self.mismatch_class.value
            if isinstance(self.mismatch_class, MismatchClass)
            else str(self.mismatch_class),
            "edge_kind": self.edge_kind,
            "snapshot_id": self.snapshot_id,
        }

    @property
    def key_id(self) -> str:
        return canonical_mismatch_cid(self._identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["key_id"] = self.key_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RepairFindingKey":
        if not isinstance(payload, Mapping):
            raise McpContractMismatchError(
                "repair finding key must be an object",
                reason_code="invalid_field_type",
            )
        return cls(
            package=str(payload.get("package") or ""),
            operation=str(payload.get("operation") or ""),
            direction=str(payload.get("direction") or ContractDirection.REQUEST.value),
            schema_root=str(payload.get("schema_root") or ""),
            profile=str(payload.get("profile") or ""),
            transport=str(payload.get("transport") or ""),
            mismatch_class=str(payload.get("mismatch_class") or ""),
            edge_kind=str(payload.get("edge_kind") or ""),
            snapshot_id=str(payload.get("snapshot_id") or ""),
            schema=str(payload.get("schema") or REPAIR_FINDING_KEY_SCHEMA),
            interface=str(payload.get("interface") or REPAIR_FINDING_KEY_INTERFACE),
        )


# ---------------------------------------------------------------------------
# ContractMismatch@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContractMismatch:
    """One deterministic earliest-broken-edge contract mismatch finding.

    Interface: ``ContractMismatch@1``
    """

    finding_key: RepairFindingKey
    mismatch_class: MismatchClass
    package: str
    operation: str
    direction: ContractDirection
    consumer_id: str
    edge_kind: str
    stage: str
    expected_edge: Mapping[str, Any] = field(default_factory=dict)
    observed_edge: Mapping[str, Any] = field(default_factory=dict)
    counterexample_seed: Mapping[str, Any] = field(default_factory=dict)
    reason_code: str = ""
    blocker_id: str = ""
    nonpassing: bool = True
    finding_id: str = ""
    schema: str = CONTRACT_MISMATCH_SCHEMA
    interface: str = CONTRACT_MISMATCH_INTERFACE

    def __post_init__(self) -> None:
        if not isinstance(self.finding_key, RepairFindingKey):
            if isinstance(self.finding_key, Mapping):
                object.__setattr__(
                    self, "finding_key", RepairFindingKey.from_dict(self.finding_key)
                )
            else:
                raise McpContractMismatchError(
                    "finding_key must be a RepairFindingKey",
                    reason_code="invalid_field_type",
                )
        object.__setattr__(
            self,
            "mismatch_class",
            _norm_enum(
                self.mismatch_class, MismatchClass, field_name="mismatch_class"
            ),
        )
        object.__setattr__(
            self,
            "package",
            _norm_text(self.package, field_name="package", required=True),
        )
        object.__setattr__(
            self,
            "operation",
            _norm_text(self.operation, field_name="operation", required=True),
        )
        object.__setattr__(
            self,
            "direction",
            _norm_enum(self.direction, ContractDirection, field_name="direction"),
        )
        object.__setattr__(
            self,
            "consumer_id",
            _norm_text(self.consumer_id, field_name="consumer_id", required=True),
        )
        object.__setattr__(
            self,
            "edge_kind",
            _norm_text(self.edge_kind, field_name="edge_kind", required=True),
        )
        object.__setattr__(
            self,
            "stage",
            _norm_text(self.stage, field_name="stage", required=True),
        )
        object.__setattr__(
            self, "expected_edge", MappingProxyType(dict(self.expected_edge or {}))
        )
        object.__setattr__(
            self, "observed_edge", MappingProxyType(dict(self.observed_edge or {}))
        )
        object.__setattr__(
            self,
            "counterexample_seed",
            MappingProxyType(dict(self.counterexample_seed or {})),
        )
        object.__setattr__(
            self,
            "reason_code",
            _norm_text(self.reason_code, field_name="reason_code", required=True),
        )
        object.__setattr__(
            self,
            "blocker_id",
            _norm_text(self.blocker_id, field_name="blocker_id"),
        )
        object.__setattr__(self, "nonpassing", bool(self.nonpassing))
        # Force nonpassing for the closed nonpassing class set.
        if self.mismatch_class.value in NONPASSING_MISMATCH_CLASSES:
            object.__setattr__(self, "nonpassing", True)
        object.__setattr__(
            self,
            "schema",
            _norm_text(self.schema, field_name="schema", required=True),
        )
        object.__setattr__(
            self,
            "interface",
            _norm_text(self.interface, field_name="interface", required=True),
        )
        if self.schema != CONTRACT_MISMATCH_SCHEMA:
            raise McpContractMismatchError(
                "unsupported contract mismatch schema",
                reason_code="unsupported_schema",
            )
        if self.interface != CONTRACT_MISMATCH_INTERFACE:
            raise McpContractMismatchError(
                "unsupported contract mismatch interface",
                reason_code="unsupported_interface",
            )
        # Keep finding_key class aligned with the finding class.
        if self.finding_key.mismatch_class != self.mismatch_class:
            raise McpContractMismatchError(
                "finding_key.mismatch_class must match mismatch_class",
                reason_code="finding_key_class_mismatch",
            )
        expected = canonical_mismatch_cid(self._identity_payload())
        claimed = _norm_text(self.finding_id, field_name="finding_id")
        if claimed and claimed != expected:
            raise McpContractMismatchError(
                "finding_id does not match recomputed identity",
                reason_code="forged_finding_id",
            )
        object.__setattr__(self, "finding_id", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "finding_key": self.finding_key._identity_payload(),
            "mismatch_class": self.mismatch_class.value
            if isinstance(self.mismatch_class, MismatchClass)
            else str(self.mismatch_class),
            "package": self.package,
            "operation": self.operation,
            "direction": self.direction.value
            if isinstance(self.direction, ContractDirection)
            else str(self.direction),
            "consumer_id": self.consumer_id,
            "edge_kind": self.edge_kind,
            "stage": self.stage,
            "expected_edge": dict(self.expected_edge),
            "observed_edge": dict(self.observed_edge),
            "counterexample_seed": dict(self.counterexample_seed),
            "reason_code": self.reason_code,
            "blocker_id": self.blocker_id,
            "nonpassing": bool(self.nonpassing),
        }

    @property
    def canonical_key(self) -> str:
        """Alias for the semantic finding key id (dedupe identity)."""

        return self.finding_key.key_id

    def to_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["finding_id"] = self.finding_id
        payload["canonical_key"] = self.canonical_key
        payload["finding_key"] = self.finding_key.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractMismatch":
        if not isinstance(payload, Mapping):
            raise McpContractMismatchError(
                "contract mismatch must be an object",
                reason_code="invalid_field_type",
            )
        key_payload = payload.get("finding_key") or {}
        return cls(
            finding_key=RepairFindingKey.from_dict(key_payload)
            if isinstance(key_payload, Mapping)
            else key_payload,  # type: ignore[arg-type]
            mismatch_class=str(payload.get("mismatch_class") or ""),
            package=str(payload.get("package") or ""),
            operation=str(payload.get("operation") or ""),
            direction=str(payload.get("direction") or ContractDirection.REQUEST.value),
            consumer_id=str(payload.get("consumer_id") or ""),
            edge_kind=str(payload.get("edge_kind") or ""),
            stage=str(payload.get("stage") or ""),
            expected_edge=dict(payload.get("expected_edge") or {}),
            observed_edge=dict(payload.get("observed_edge") or {}),
            counterexample_seed=dict(payload.get("counterexample_seed") or {}),
            reason_code=str(payload.get("reason_code") or ""),
            blocker_id=str(payload.get("blocker_id") or ""),
            nonpassing=bool(payload.get("nonpassing", True)),
            finding_id=str(payload.get("finding_id") or ""),
            schema=str(payload.get("schema") or CONTRACT_MISMATCH_SCHEMA),
            interface=str(payload.get("interface") or CONTRACT_MISMATCH_INTERFACE),
        )


# ---------------------------------------------------------------------------
# Findings catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class McpContractMismatchFindings:
    """Content-addressed catalog of classified, deduplicated findings."""

    snapshot_id: str
    graph_cid: str
    transcript_epoch: str
    findings: tuple[ContractMismatch, ...] = ()
    earliest_by_consumer: Mapping[str, str] = field(default_factory=dict)
    complete: bool = False
    model_calls: int = 0
    version: str = FINDINGS_VERSION
    schema: str = MISMATCH_FINDINGS_SCHEMA
    interface: str = MISMATCH_FINDINGS_INTERFACE
    evidence_term: str = MISMATCH_EVIDENCE_TERM
    findings_cid: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "snapshot_id",
            _norm_text(self.snapshot_id, field_name="snapshot_id", required=True),
        )
        object.__setattr__(
            self,
            "graph_cid",
            _norm_text(self.graph_cid, field_name="graph_cid", required=True),
        )
        object.__setattr__(
            self,
            "transcript_epoch",
            _norm_text(
                self.transcript_epoch, field_name="transcript_epoch", required=True
            ),
        )
        normalized: list[ContractMismatch] = []
        for item in self.findings:
            if isinstance(item, ContractMismatch):
                normalized.append(item)
            elif isinstance(item, Mapping):
                normalized.append(ContractMismatch.from_dict(item))
            else:
                raise McpContractMismatchError(
                    "findings must contain ContractMismatch values",
                    reason_code="invalid_field_type",
                )
        if len(normalized) > _MAX_FINDINGS:
            raise McpContractMismatchError(
                "too many mismatch findings",
                reason_code="bounds_exceeded",
                details={"count": len(normalized)},
            )
        # Deterministic order: package, operation, class, edge_kind, finding_id.
        normalized.sort(
            key=lambda f: (
                f.package,
                f.operation,
                f.mismatch_class.value,
                f.edge_kind,
                f.finding_id,
            )
        )
        object.__setattr__(self, "findings", tuple(normalized))
        object.__setattr__(
            self,
            "earliest_by_consumer",
            MappingProxyType(
                {
                    str(k): str(v)
                    for k, v in sorted((self.earliest_by_consumer or {}).items())
                }
            ),
        )
        object.__setattr__(self, "complete", bool(self.complete))
        object.__setattr__(self, "model_calls", int(self.model_calls))
        if self.model_calls != 0:
            raise McpContractMismatchError(
                "mismatch findings must report zero model calls",
                reason_code="model_calls_nonzero",
            )
        object.__setattr__(
            self,
            "version",
            _norm_text(self.version, field_name="version", required=True),
        )
        object.__setattr__(
            self,
            "schema",
            _norm_text(self.schema, field_name="schema", required=True),
        )
        object.__setattr__(
            self,
            "interface",
            _norm_text(self.interface, field_name="interface", required=True),
        )
        object.__setattr__(
            self,
            "evidence_term",
            _norm_text(self.evidence_term, field_name="evidence_term", required=True),
        )
        if self.schema != MISMATCH_FINDINGS_SCHEMA:
            raise McpContractMismatchError(
                "unsupported mismatch findings schema",
                reason_code="unsupported_schema",
            )
        if self.interface != MISMATCH_FINDINGS_INTERFACE:
            raise McpContractMismatchError(
                "unsupported mismatch findings interface",
                reason_code="unsupported_interface",
            )
        if self.evidence_term != MISMATCH_EVIDENCE_TERM:
            raise McpContractMismatchError(
                "unsupported mismatch evidence term",
                reason_code="unsupported_evidence_term",
            )
        expected = canonical_mismatch_cid(self._root_payload())
        claimed = _norm_text(self.findings_cid, field_name="findings_cid")
        if claimed and claimed != expected:
            raise McpContractMismatchError(
                "findings_cid does not match recomputed identity",
                reason_code="forged_findings_cid",
            )
        object.__setattr__(self, "findings_cid", expected)

    def _root_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": self.interface,
            "evidence_term": self.evidence_term,
            "version": self.version,
            "snapshot_id": self.snapshot_id,
            "graph_cid": self.graph_cid,
            "transcript_epoch": self.transcript_epoch,
            "complete": bool(self.complete),
            "model_calls": 0,
            "earliest_by_consumer": dict(self.earliest_by_consumer),
            "findings": [item._identity_payload() for item in self.findings],
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._root_payload()
        payload["findings_cid"] = self.findings_cid
        payload["canonical_digest"] = digest_for_canonical_bytes(
            canonical_mismatch_bytes(self._root_payload())
        )
        payload["finding_count"] = len(self.findings)
        payload["findings"] = [item.to_dict() for item in self.findings]
        payload["nonpassing_count"] = sum(1 for item in self.findings if item.nonpassing)
        return payload

    def to_artifact_bytes(self) -> bytes:
        text = json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=True)
        return (text + "\n").encode("utf-8")

    def verifies_cid(self) -> bool:
        return self.findings_cid == canonical_mismatch_cid(self._root_payload())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "McpContractMismatchFindings":
        if not isinstance(payload, Mapping):
            raise McpContractMismatchError(
                "mismatch findings must be an object",
                reason_code="invalid_field_type",
            )
        findings_raw = payload.get("findings") or ()
        findings = tuple(
            item
            if isinstance(item, ContractMismatch)
            else ContractMismatch.from_dict(item)
            for item in findings_raw
        )
        return cls(
            snapshot_id=str(payload.get("snapshot_id") or ""),
            graph_cid=str(payload.get("graph_cid") or ""),
            transcript_epoch=str(payload.get("transcript_epoch") or ""),
            findings=findings,
            earliest_by_consumer=dict(payload.get("earliest_by_consumer") or {}),
            complete=bool(payload.get("complete", False)),
            model_calls=int(payload.get("model_calls") or 0),
            version=str(payload.get("version") or FINDINGS_VERSION),
            schema=str(payload.get("schema") or MISMATCH_FINDINGS_SCHEMA),
            interface=str(payload.get("interface") or MISMATCH_FINDINGS_INTERFACE),
            evidence_term=str(payload.get("evidence_term") or MISMATCH_EVIDENCE_TERM),
            findings_cid=str(payload.get("findings_cid") or ""),
        )

    @classmethod
    def from_json(cls, data: bytes | str) -> "McpContractMismatchFindings":
        if isinstance(data, bytes):
            text = data.decode("utf-8")
        else:
            text = data
        payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise McpContractMismatchError(
                "mismatch findings JSON must be an object",
                reason_code="invalid_field_type",
            )
        return cls.from_dict(payload)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _blocker_to_mismatch_class(
    blocker: ContractBlocker,
    *,
    edge_kind: str,
) -> MismatchClass:
    """Map a graph blocker to an independent mismatch class."""

    kind = blocker.kind if isinstance(blocker.kind, BlockerKind) else BlockerKind(str(blocker.kind))
    if kind is BlockerKind.EXPECTED_ONLY:
        return MismatchClass.EXPECTED_ONLY
    if kind is BlockerKind.AMBIGUOUS:
        return MismatchClass.AMBIGUOUS
    if kind is BlockerKind.AUTHORITY_CONFLICT:
        return MismatchClass.AUTHORITY
    if kind is BlockerKind.PSEUDO_CID:
        return MismatchClass.IDENTITY
    if kind is BlockerKind.DUPLICATE_ALIAS:
        return MismatchClass.IDENTITY
    if kind is BlockerKind.MIXED_ROOT:
        return MismatchClass.AUTHORITY
    if kind is BlockerKind.OBSERVED_ONLY:
        # Observation without declaration is an implementation/identity defect.
        return MismatchClass.IMPLEMENTATION
    if kind is BlockerKind.UNRESOLVED:
        # Missing observation stages on an expected surface.
        if edge_kind in _IMPLEMENTATION_EDGE_KINDS:
            return MismatchClass.MISSING
        if edge_kind in _MEDIATION_EDGE_KINDS:
            return MismatchClass.MEDIATION
        return MismatchClass.MISSING
    return MismatchClass.MISSING


def _consumer_meta(
    graph: McpContractGraph, consumer_id: str
) -> dict[str, str]:
    """Recover package/operation/profile/transport for a consumer node."""

    package = ""
    operation = ""
    profile = "mcp++/default"
    transport = "stdio"
    schema_root = f"schemas/{consumer_id}.json"
    for node in graph.nodes:
        if node.kind is ContractNodeKind.CONSUMER and (
            node.stable_key == f"consumer:{consumer_id}"
            or node.payload.get("operation")
            and consumer_id.endswith(str(node.payload.get("operation") or ""))
        ):
            package = str(node.payload.get("package") or package)
            operation = str(node.payload.get("operation") or operation)
            profile = str(node.payload.get("profile") or profile)
            transport = str(node.payload.get("transport") or transport)
            break
    # Prefer stable_key match.
    for node in graph.nodes:
        if node.stable_key == f"consumer:{consumer_id}":
            package = str(node.payload.get("package") or package)
            operation = str(node.payload.get("operation") or operation)
            profile = str(node.payload.get("profile") or profile)
            transport = str(node.payload.get("transport") or transport)
            break
    if not package and "/" in consumer_id:
        # swissknife/ipfs_accelerate_py/tool → package = ipfs_accelerate_py
        parts = consumer_id.split("/")
        if len(parts) >= 2:
            package = parts[1]
    if not operation:
        operation = consumer_id
    # Schema root from method node when present.
    for node in graph.nodes:
        if (
            node.kind in {ContractNodeKind.MCP_METHOD, ContractNodeKind.MCP_SCHEMA}
            and consumer_id in node.stable_key
        ):
            for ref in node.source_refs:
                if ref.endswith(".json"):
                    schema_root = ref
                    break
            break
    tool = operation.rsplit(".", 1)[-1] if operation else "unknown"
    if schema_root.endswith(f"{consumer_id}.json"):
        schema_root = f"schemas/{tool}.json"
    return {
        "package": package or "unknown",
        "operation": operation or consumer_id,
        "profile": profile or "mcp++/default",
        "transport": transport or "stdio",
        "schema_root": schema_root,
    }


def _edge_summary(edge: ContractEdge | None) -> dict[str, Any]:
    if edge is None:
        return {}
    return {
        "edge_id": edge.edge_id,
        "kind": edge.kind.value if isinstance(edge.kind, Enum) else str(edge.kind),
        "resolution": edge.resolution.value
        if isinstance(edge.resolution, Enum)
        else str(edge.resolution),
        "source": edge.source,
        "target": edge.target,
        "authority": edge.authority.value
        if isinstance(edge.authority, Enum)
        else str(edge.authority),
        "mandatory": bool(edge.mandatory),
        "consumer_id": edge.consumer_id,
    }


def earliest_broken_edge(
    graph: McpContractGraph, consumer_id: str
) -> ContractBlocker | None:
    """Return the earliest typed blocker along the mandatory path order."""

    blockers = list(graph.blockers_for(consumer_id))
    if not blockers:
        return None
    by_kind = {item.edge_kind: item for item in blockers}
    for edge_kind in MANDATORY_EDGE_KINDS:
        if edge_kind in by_kind:
            return by_kind[edge_kind]
    # Fallback: stable order by edge_kind then blocker_id.
    blockers.sort(key=lambda b: (b.edge_kind, b.blocker_id))
    return blockers[0]


def _transcript_epoch(transcript: Mapping[str, Any]) -> str:
    """Derive a content-addressed epoch id from the live transcript."""

    if not isinstance(transcript, Mapping):
        raise McpContractMismatchError(
            "transcript must be an object",
            reason_code="invalid_field_type",
        )
    schema = str(transcript.get("schema") or "")
    evidence = str(transcript.get("evidence_term") or "")
    if schema and schema != LIVE_CONTRACT_TRANSCRIPT_SCHEMA:
        raise McpContractMismatchError(
            "unsupported live transcript schema",
            reason_code="unsupported_transcript_schema",
            details={"schema": schema},
        )
    if evidence and evidence != LIVE_OBSERVATION_EVIDENCE_TERM:
        raise McpContractMismatchError(
            "unsupported live transcript evidence term",
            reason_code="unsupported_transcript_evidence",
            details={"evidence_term": evidence},
        )
    # Prefer an explicit CID field when present; otherwise hash a stable subset.
    for key in ("transcript_cid", "receipt_cid", "local_cid"):
        value = transcript.get(key)
        if isinstance(value, str) and value.startswith("b"):
            return value
    seed = {
        "service_id": transcript.get("service_id"),
        "roles_observed": list(transcript.get("roles_observed") or ()),
        "passed": bool(transcript.get("passed")),
        "model_calls": int(transcript.get("model_calls") or 0),
        "exchange_count": len(transcript.get("exchanges") or ()),
        "process_witness_cid": (
            (transcript.get("process_witness") or {}).get("witness_cid")
            if isinstance(transcript.get("process_witness"), Mapping)
            else None
        ),
    }
    return canonical_mismatch_cid(seed)


def _live_tools_by_role(transcript: Mapping[str, Any]) -> dict[str, set[str]]:
    tools: dict[str, set[str]] = {}
    for exchange in transcript.get("exchanges") or ():
        if not isinstance(exchange, Mapping):
            continue
        role = str(exchange.get("role") or "")
        kind = str(exchange.get("kind") or "")
        if kind != "tools/list":
            continue
        details = exchange.get("details") or {}
        listed = details.get("tools") if isinstance(details, Mapping) else None
        bucket = tools.setdefault(role, set())
        if isinstance(listed, Sequence) and not isinstance(listed, (str, bytes)):
            for tool in listed:
                if isinstance(tool, str) and tool.strip():
                    bucket.add(tool.strip())
    return tools


def _live_role_terminal_states(
    transcript: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """role → {kind → terminal_state} for live exchanges."""

    out: dict[str, dict[str, str]] = {}
    for exchange in transcript.get("exchanges") or ():
        if not isinstance(exchange, Mapping):
            continue
        role = str(exchange.get("role") or "")
        kind = str(exchange.get("kind") or "")
        state = str(exchange.get("terminal_state") or "")
        if not role or not kind:
            continue
        out.setdefault(role, {})[kind] = state
    return out


def _tool_name_from_operation(operation: str) -> str:
    text = operation.strip()
    if text.startswith("tools.call."):
        return text[len("tools.call.") :]
    if text.startswith("tools/call."):
        return text[len("tools/call.") :]
    return text.rsplit(".", 1)[-1]


def classify_graph_blockers(
    graph: McpContractGraph,
    *,
    direction: ContractDirection = ContractDirection.REQUEST,
) -> list[ContractMismatch]:
    """Emit one finding per graph blocker (all blockers retained)."""

    findings: list[ContractMismatch] = []
    for blocker in graph.blockers:
        meta = _consumer_meta(graph, blocker.consumer_id)
        mismatch_class = _blocker_to_mismatch_class(
            blocker, edge_kind=blocker.edge_kind
        )
        # Expected edge: the mandatory edge that should have resolved.
        expected_edge = {
            "consumer_id": blocker.consumer_id,
            "edge_kind": blocker.edge_kind,
            "stage": blocker.stage,
            "resolution": "resolved",
            "mandatory": True,
        }
        observed_edge = {
            "consumer_id": blocker.consumer_id,
            "edge_kind": blocker.edge_kind,
            "stage": blocker.stage,
            "resolution": blocker.kind.value
            if isinstance(blocker.kind, BlockerKind)
            else str(blocker.kind),
            "blocker_id": blocker.blocker_id,
            "candidate_node_ids": list(blocker.candidate_node_ids),
            "details": dict(blocker.details),
        }
        # Prefer a concrete blocked_by edge when present.
        for edge in graph.edges:
            if (
                edge.consumer_id == blocker.consumer_id
                and edge.kind.value == "blocked_by"
                and str((edge.payload or {}).get("edge_kind") or "") == blocker.edge_kind
            ):
                observed_edge = _edge_summary(edge)
                observed_edge["blocker_kind"] = (
                    blocker.kind.value
                    if isinstance(blocker.kind, BlockerKind)
                    else str(blocker.kind)
                )
                break

        key = RepairFindingKey(
            package=meta["package"],
            operation=meta["operation"],
            direction=direction,
            schema_root=meta["schema_root"],
            profile=meta["profile"],
            transport=meta["transport"],
            mismatch_class=mismatch_class,
            edge_kind=blocker.edge_kind,
            snapshot_id=graph.snapshot_id,
        )
        seed = {
            "blocker_id": blocker.blocker_id,
            "reason_code": blocker.reason_code,
            "kind": blocker.kind.value
            if isinstance(blocker.kind, BlockerKind)
            else str(blocker.kind),
            "edge_kind": blocker.edge_kind,
            "stage": blocker.stage,
            "details": dict(blocker.details),
            "candidate_node_ids": list(blocker.candidate_node_ids),
        }
        findings.append(
            ContractMismatch(
                finding_key=key,
                mismatch_class=mismatch_class,
                package=meta["package"],
                operation=meta["operation"],
                direction=direction,
                consumer_id=blocker.consumer_id,
                edge_kind=blocker.edge_kind,
                stage=blocker.stage,
                expected_edge=expected_edge,
                observed_edge=observed_edge,
                counterexample_seed=seed,
                reason_code=blocker.reason_code or mismatch_class.value,
                blocker_id=blocker.blocker_id,
                nonpassing=True,
            )
        )
    return findings


def classify_live_observations(
    graph: McpContractGraph,
    transcript: Mapping[str, Any],
    *,
    direction: ContractDirection = ContractDirection.REQUEST,
) -> list[ContractMismatch]:
    """Emit liveness / unobserved findings from live transcript correlation."""

    findings: list[ContractMismatch] = []
    tools_by_role = _live_tools_by_role(transcript)
    terminals = _live_role_terminal_states(transcript)

    # Liveness / protocol failures per role.
    for role, states in terminals.items():
        for kind, state in states.items():
            if state in {"passed", "pass", "success"}:
                continue
            if kind in {"malformed_call", "unknown_tool"}:
                # Expected fail-closed outcomes are not defects.
                continue
            mismatch_class = (
                MismatchClass.PROTOCOL
                if kind in {"initialize", "tools/list", "tools/call"}
                else MismatchClass.LIVENESS
            )
            if kind in {"tools/call"} and state not in {"passed", "pass"}:
                mismatch_class = MismatchClass.LIVENESS
            package = {
                "accelerate": "ipfs_accelerate_py",
                "datasets": "ipfs_datasets_py",
                "kit": "ipfs_kit_py",
            }.get(role, role)
            operation = f"live.{kind}"
            edge_kind = f"live:{kind}"
            key = RepairFindingKey(
                package=package,
                operation=operation,
                direction=direction,
                schema_root=f"live/{role}/{kind}",
                profile="mcp++/live",
                transport="in_process",
                mismatch_class=mismatch_class,
                edge_kind=edge_kind,
                snapshot_id=graph.snapshot_id,
            )
            findings.append(
                ContractMismatch(
                    finding_key=key,
                    mismatch_class=mismatch_class,
                    package=package,
                    operation=operation,
                    direction=direction,
                    consumer_id=f"live/{role}",
                    edge_kind=edge_kind,
                    stage=kind,
                    expected_edge={
                        "role": role,
                        "kind": kind,
                        "terminal_state": "passed",
                    },
                    observed_edge={
                        "role": role,
                        "kind": kind,
                        "terminal_state": state,
                    },
                    counterexample_seed={
                        "role": role,
                        "kind": kind,
                        "terminal_state": state,
                    },
                    reason_code=f"live_{kind}_{state or 'failed'}",
                    nonpassing=True,
                )
            )

    # Expected consumers whose tool is absent from the live tools/list.
    for consumer_id in graph.consumer_ids:
        meta = _consumer_meta(graph, consumer_id)
        package = meta["package"]
        role = _PACKAGE_TO_ROLE.get(package)
        if not role:
            continue
        tool = _tool_name_from_operation(meta["operation"])
        listed = tools_by_role.get(role, set())
        # Reference "expected.only.tool" and other declaration-only tools are
        # already covered by graph blockers; still emit unobserved when the
        # tool is not in the live catalog so the class stays explicit.
        if tool in listed:
            continue
        # Skip if the consumer already has an expected_only finding for the
        # first implementation edge — still emit unobserved as an independent
        # class (names may coincide; classes must not collapse).
        key = RepairFindingKey(
            package=package,
            operation=meta["operation"],
            direction=direction,
            schema_root=meta["schema_root"],
            profile=meta["profile"],
            transport=meta["transport"],
            mismatch_class=MismatchClass.UNOBSERVED,
            edge_kind="live:tools/list",
            snapshot_id=graph.snapshot_id,
        )
        findings.append(
            ContractMismatch(
                finding_key=key,
                mismatch_class=MismatchClass.UNOBSERVED,
                package=package,
                operation=meta["operation"],
                direction=direction,
                consumer_id=consumer_id,
                edge_kind="live:tools/list",
                stage="tools/list",
                expected_edge={
                    "consumer_id": consumer_id,
                    "tool": tool,
                    "role": role,
                    "present_in_tools_list": True,
                },
                observed_edge={
                    "role": role,
                    "tools": sorted(listed),
                    "present_in_tools_list": False,
                },
                counterexample_seed={
                    "tool": tool,
                    "role": role,
                    "listed_tools": sorted(listed),
                },
                reason_code="tool_absent_from_live_tools_list",
                nonpassing=True,
            )
        )
    return findings


def classify_mismatches(
    graph: McpContractGraph,
    transcript: Mapping[str, Any] | None = None,
    *,
    direction: ContractDirection = ContractDirection.REQUEST,
) -> list[ContractMismatch]:
    """Classify graph blockers and optional live observations into findings."""

    findings = classify_graph_blockers(graph, direction=direction)
    if transcript is not None:
        findings.extend(
            classify_live_observations(graph, transcript, direction=direction)
        )
    return findings


def deduplicate_findings(
    findings: Sequence[ContractMismatch | Mapping[str, Any]],
) -> tuple[ContractMismatch, ...]:
    """Collapse findings that share an identical RepairFindingKey.

    Only exact semantic-key matches collapse.  Distinct classes, directions,
    edge kinds, or schema/profile/transport roots remain independent even when
    package/operation names coincide (e.g. dag.put vs dag.get).
    """

    by_key: dict[str, ContractMismatch] = {}
    order: list[str] = []
    for raw in findings:
        finding = (
            raw
            if isinstance(raw, ContractMismatch)
            else ContractMismatch.from_dict(raw)
        )
        key_id = finding.finding_key.key_id
        prior = by_key.get(key_id)
        if prior is None:
            by_key[key_id] = finding
            order.append(key_id)
            continue
        # Prefer the finding with a blocker_id / richer counterexample seed.
        if not prior.blocker_id and finding.blocker_id:
            by_key[key_id] = finding
            continue
        if len(finding.counterexample_seed) > len(prior.counterexample_seed):
            by_key[key_id] = finding
    return tuple(by_key[key_id] for key_id in order)


def classify_and_deduplicate(
    graph: McpContractGraph,
    transcript: Mapping[str, Any] | None = None,
    *,
    direction: ContractDirection = ContractDirection.REQUEST,
) -> tuple[ContractMismatch, ...]:
    """Classify then deduplicate findings for one graph/transcript epoch."""

    return deduplicate_findings(
        classify_mismatches(graph, transcript, direction=direction)
    )


def build_mismatch_findings(
    graph: McpContractGraph,
    transcript: Mapping[str, Any] | None = None,
    *,
    direction: ContractDirection = ContractDirection.REQUEST,
    require_shared_epoch: bool = True,
) -> McpContractMismatchFindings:
    """Build the content-addressed mismatch findings catalog."""

    if transcript is None:
        if require_shared_epoch:
            raise McpContractMismatchError(
                "live transcript is required for shared-epoch mismatch findings",
                reason_code="transcript_required",
            )
        epoch = "epoch:none"
    else:
        epoch = _transcript_epoch(transcript)
        if require_shared_epoch:
            # Graph snapshot + transcript epoch must both be non-empty and
            # bound into the catalog; mixed empty epochs fail closed.
            if not graph.snapshot_id or not epoch:
                raise McpContractMismatchError(
                    "graph and transcript must share a non-empty epoch",
                    reason_code="epoch_missing",
                )

    findings = classify_and_deduplicate(graph, transcript, direction=direction)

    earliest: dict[str, str] = {}
    for consumer_id in graph.consumer_ids:
        blocker = earliest_broken_edge(graph, consumer_id)
        if blocker is not None:
            earliest[consumer_id] = blocker.edge_kind
        else:
            # Live-only earliest: first unobserved finding for the consumer.
            for finding in findings:
                if finding.consumer_id == consumer_id:
                    earliest[consumer_id] = finding.edge_kind
                    break

    # complete when the graph has no blockers AND every finding is not
    # nonpassing — with current nonpassing classes this means zero findings.
    complete = (
        graph.complete
        and not any(item.nonpassing for item in findings)
        and not findings
    )

    return McpContractMismatchFindings(
        snapshot_id=graph.snapshot_id,
        graph_cid=graph.graph_cid,
        transcript_epoch=epoch,
        findings=findings,
        earliest_by_consumer=earliest,
        complete=complete,
        model_calls=0,
    )


def materialize_mcp_contract_mismatch_findings(
    *,
    repo_root: Path | None = None,
    graph: McpContractGraph | None = None,
    transcript: Mapping[str, Any] | None = None,
    require_shared_epoch: bool = True,
) -> McpContractMismatchFindings:
    """Materialize findings from the committed graph + live transcript."""

    root = Path(repo_root) if repo_root is not None else _default_workspace()
    if graph is None:
        graph_path = root.joinpath(*PurePosixPath(
            "data/agent_supervisor/deterministic_contract_repair/mcp_contract_graph.json"
        ).parts)
        if graph_path.is_file():
            try:
                graph = load_mcp_contract_graph(graph_path)
            except (McpContractGraphError, OSError, json.JSONDecodeError):
                graph = materialize_mcp_contract_graph()
        else:
            graph = materialize_mcp_contract_graph()
    if transcript is None and require_shared_epoch:
        try:
            transcript = load_mcp_live_transcript(repo_root=root)
        except Exception as exc:  # noqa: BLE001 - map to mismatch error
            raise McpContractMismatchError(
                f"unable to load live transcript for shared epoch: {exc}",
                reason_code="transcript_load_failed",
            ) from exc
    return build_mismatch_findings(
        graph,
        transcript,
        require_shared_epoch=require_shared_epoch,
    )


def write_mcp_contract_mismatch_findings(
    destination: str | Path | None = None,
    *,
    findings: McpContractMismatchFindings | None = None,
    repo_root: Path | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Path:
    """Atomically write the mismatch findings artifact."""

    root = Path(repo_root) if repo_root is not None else _default_workspace()
    artifact = findings or materialize_mcp_contract_mismatch_findings(repo_root=root)
    if not artifact.verifies_cid():
        raise McpContractMismatchError(
            "findings CID does not reconstruct from canonical bytes",
            reason_code="cid_reconstruction_failed",
        )
    data = artifact.to_artifact_bytes()
    if len(data) > max_bytes:
        raise McpContractMismatchError(
            f"artifact exceeds {max_bytes} bytes",
            reason_code="bounds_exceeded",
            details={"byte_length": len(data)},
        )
    if destination is None:
        path = root.joinpath(*PurePosixPath(DCR_ARTIFACT_PATH).parts)
    else:
        path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


def ensure_mcp_contract_mismatch_findings_artifact(
    *,
    repo_root: Path | None = None,
    force: bool = False,
) -> Path:
    """Ensure the declared DCR-024 artifact exists without unnecessary rewrites."""

    root = Path(repo_root) if repo_root is not None else _default_workspace()
    out = root.joinpath(*PurePosixPath(DCR_ARTIFACT_PATH).parts)
    if out.is_file() and not force:
        try:
            loaded = load_mcp_contract_mismatch_findings(out)
        except McpContractMismatchError:
            loaded = None
        if (
            loaded is not None
            and loaded.schema == MISMATCH_FINDINGS_SCHEMA
            and loaded.interface == MISMATCH_FINDINGS_INTERFACE
            and loaded.evidence_term == MISMATCH_EVIDENCE_TERM
            and loaded.verifies_cid()
        ):
            return out
    return write_mcp_contract_mismatch_findings(out, repo_root=root)


def load_mcp_contract_mismatch_findings(
    source: str | Path | None = None,
    *,
    repo_root: Path | None = None,
) -> McpContractMismatchFindings:
    """Load and revalidate a mismatch findings artifact."""

    root = Path(repo_root) if repo_root is not None else _default_workspace()
    if source is None:
        path = root.joinpath(*PurePosixPath(DCR_ARTIFACT_PATH).parts)
    else:
        path = Path(source)
    if not path.is_file():
        raise McpContractMismatchError(
            f"mismatch findings missing: {path}",
            reason_code="findings_missing",
            details={"path": str(path)},
        )
    raw = path.read_bytes()
    findings = McpContractMismatchFindings.from_json(raw)
    if not findings.verifies_cid():
        raise McpContractMismatchError(
            "loaded findings CID does not reconstruct from canonical bytes",
            reason_code="cid_reconstruction_failed",
        )
    return findings


__all__ = [
    "CONTRACT_MISMATCH_INTERFACE",
    "CONTRACT_MISMATCH_SCHEMA",
    "CONTRACT_VERSION",
    "DCR_ARTIFACT_PATH",
    "DCR_TASK_ID",
    "MISMATCH_EVIDENCE_TERM",
    "MISMATCH_FINDINGS_INTERFACE",
    "MISMATCH_FINDINGS_SCHEMA",
    "NONPASSING_MISMATCH_CLASSES",
    "REPAIR_FINDING_KEY_INTERFACE",
    "REPAIR_FINDING_KEY_SCHEMA",
    "ContractMismatch",
    "McpContractMismatchError",
    "McpContractMismatchFindings",
    "MismatchClass",
    "RepairFindingKey",
    "build_mismatch_findings",
    "canonical_mismatch_bytes",
    "canonical_mismatch_cid",
    "classify_and_deduplicate",
    "classify_graph_blockers",
    "classify_live_observations",
    "classify_mismatches",
    "deduplicate_findings",
    "digest_for_canonical_bytes",
    "earliest_broken_edge",
    "ensure_mcp_contract_mismatch_findings_artifact",
    "load_mcp_contract_mismatch_findings",
    "materialize_mcp_contract_mismatch_findings",
    "write_mcp_contract_mismatch_findings",
]
