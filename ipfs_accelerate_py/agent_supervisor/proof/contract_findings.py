"""Append-only, content-addressed contract finding ledger (VFS-029 / VFS-049).

This module persists immutable finding records and projects a mutable
*current* view over them.  History is never rewritten: supersession, stale
invalidation, and rejection append diagnostic events that only change the
current projection.

Objective VFS-G100 evidence identities closed here:

* ``vfs/finding-ledger@1`` — typed, deduplicated correctness/vulnerability
  ledger with separate append-only history and mutable current-tree
  projection;
* ``vfs/vulnerability-evidence-policy@1`` — a finding may carry the
  ``vulnerability`` label only when a threat-path CID and concrete impact
  statement are bound (fail-closed; no LLM classification).

Deduplication is deliberately narrow.  Two appends collide as duplicates only
when their semantic identity agrees:

* expected and observed contract CIDs;
* root-cause family;
* affected symbols / interfaces (sorted);
* merge fate.

Equal content under the same CID is an idempotent append.  Distinct content
that would hash to the same CID is an integrity collision.  Contradictory
claims (same scope, different claim/status/severity with unequal semantic
identity material) are both retained in history and surface as a projection
conflict.

Severity is evidence-bound: elevating severity without matching status,
claim level, and confidence is rejected as poisoned.  Partial findings may be
stored for audit but are not admitted into the actionable current projection.
Duplicates and stale admissions stay out of the actionable current set so
they cannot create repair work downstream.

Large source, AST, proof, and witness bodies stay outside these records as
artifact references.  The ledger is diagnostic storage; it never mutates
tasks, source trees, or merge authority.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Final, TypeVar

from ..core.multiformats_identity import cid_for_dag_json
from .program_assurance_contracts import (
    ClaimLevel,
    EvidenceFreshness,
    FindingSeverity,
    FindingStatus,
    ProgramAssuranceContractError,
)
from .formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)


CONTRACT_FINDINGS_VERSION: Final[int] = 1
SCHEMA_VERSION: Final[int] = CONTRACT_FINDINGS_VERSION
LEDGER_VERSION: Final[str] = "contract-finding-ledger@1"
# VFS-G100 objective evidence identities (ledger + vulnerability policy).
FINDING_LEDGER_EVIDENCE: Final[str] = "vfs/finding-ledger@1"
VULNERABILITY_EVIDENCE_POLICY: Final[str] = "vfs/vulnerability-evidence-policy@1"
FINDING_LEDGER_G100_EVIDENCE_TERMS: Final[tuple[str, ...]] = (
    FINDING_LEDGER_EVIDENCE,
    VULNERABILITY_EVIDENCE_POLICY,
)
VULNERABILITY_LABEL: Final[str] = "vulnerability"
GOAL_ID: Final[str] = "VFS-G100"

MAX_TEXT_BYTES: Final[int] = 8_192
MAX_CLAUSE_BYTES: Final[int] = 4_096
MAX_COLLECTION_ITEMS: Final[int] = 256
MAX_CALL_SLICE_STEPS: Final[int] = 64
MAX_RECORD_BYTES: Final[int] = 262_144
MAX_LEDGER_ENTRIES: Final[int] = 100_000
MAX_PROJECTION_ENTRIES: Final[int] = 50_000
MAX_REJECTION_REASONS: Final[int] = 64
MAX_ANALYZER_VERSIONS: Final[int] = 64
MAX_LABELS: Final[int] = 32
MILLION: Final[int] = 1_000_000

CALL_SLICE_STEP_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-finding/call-slice-step@1"
)
CALL_SLICE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-finding/call-slice@1"
)
EVIDENCE_REFS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-finding/evidence-refs@1"
)
ANALYZER_VERSIONS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-finding/analyzer-versions@1"
)
SEMANTIC_DEDUP_KEY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-finding/semantic-dedup-key@1"
)
CONTRACT_FINDING_RECORD_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-finding/record@1"
)
FINDING_PROJECTION_ENTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-finding/projection-entry@1"
)
LEDGER_EVENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-finding/ledger-event@1"
)
APPEND_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-finding/append-receipt@1"
)
PROJECTION_SNAPSHOT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-finding/projection-snapshot@1"
)

# Severity may never exceed what the claim level / status combination can
# justify.  Values are inclusive upper bounds.
_SEVERITY_RANK: Final[dict[FindingSeverity, int]] = {
    FindingSeverity.INFO: 0,
    FindingSeverity.LOW: 1,
    FindingSeverity.MEDIUM: 2,
    FindingSeverity.HIGH: 3,
    FindingSeverity.CRITICAL: 4,
}

_MAX_SEVERITY_FOR_STATUS: Final[dict[FindingStatus, FindingSeverity]] = {
    FindingStatus.CONTRACT_BROKEN: FindingSeverity.CRITICAL,
    FindingStatus.SUSPECTED: FindingSeverity.HIGH,
    FindingStatus.AMBIGUOUS: FindingSeverity.MEDIUM,
    FindingStatus.UNSUPPORTED: FindingSeverity.LOW,
    FindingStatus.INCONCLUSIVE: FindingSeverity.LOW,
    FindingStatus.STALE: FindingSeverity.INFO,
}

_MIN_CONFIDENCE_FOR_SEVERITY: Final[dict[FindingSeverity, int]] = {
    FindingSeverity.INFO: 0,
    FindingSeverity.LOW: 100_000,
    FindingSeverity.MEDIUM: 300_000,
    FindingSeverity.HIGH: 600_000,
    FindingSeverity.CRITICAL: 900_000,
}

_CONTRADICTION_FIELDS: Final[tuple[str, ...]] = (
    "claim_level",
    "status",
    "severity",
    "confidence_millionths",
    "verdict",
)


class ContractFindingError(ContractValidationError):
    """Base error for malformed or unsafe finding-ledger operations."""


class ContractFindingBoundsError(ContractFindingError):
    """A record exceeded an explicit item, text, or byte bound."""


class ForgedFindingIdentityError(ContractFindingError):
    """A caller-supplied identity or derived projection was forged."""


class FindingCollisionError(ContractFindingError):
    """Distinct payloads would share one content identity."""


class PoisonedSeverityError(ContractFindingError):
    """Severity was inflated beyond status, claim level, or confidence."""


class StaleFindingError(ContractFindingError):
    """Stale evidence was presented as current authority."""


class PartialFindingError(ContractFindingError):
    """A required field for admission was missing on a partial finding."""


class VulnerabilityEvidencePolicyError(ContractFindingError):
    """Vulnerability label without a bound threat path and impact."""


class LedgerConcurrencyError(ContractFindingError):
    """A concurrent ledger mutation could not be serialized safely."""


class LedgerCapacityError(ContractFindingBoundsError):
    """The ledger or projection exceeded its configured capacity."""


class FindingAdmissionState(str, Enum):
    """Lifecycle state of a finding inside the current projection."""

    ADMITTED = "admitted"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    STALE = "stale"
    PARTIAL = "partial"
    CONFLICT = "conflict"


class LedgerEventKind(str, Enum):
    """Append-only ledger event kinds.  History is never rewritten."""

    APPEND = "append"
    DUPLICATE = "duplicate"
    REJECT = "reject"
    SUPERSEDE = "supersede"
    INVALIDATE_STALE = "invalidate_stale"
    MARK_PARTIAL = "mark_partial"
    CONFLICT = "conflict"


class AppendOutcome(str, Enum):
    STORED = "stored"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"
    COLLISION = "collision"
    SUPERSEDED_PRIOR = "superseded_prior"


T = TypeVar("T")
E = TypeVar("E", bound=Enum)


def _text(
    value: Any,
    *,
    field_name: str,
    required: bool = True,
    maximum: int = MAX_TEXT_BYTES,
) -> str:
    if value is None:
        if required:
            raise ContractFindingError(f"{field_name} is required")
        return ""
    if not isinstance(value, str):
        raise ContractFindingError(f"{field_name} must be a string")
    if "\x00" in value:
        raise ContractFindingError(f"{field_name} must not contain NUL")
    if len(value.encode("utf-8")) > maximum:
        raise ContractFindingBoundsError(
            f"{field_name} exceeds {maximum} bytes"
        )
    if required and not value.strip():
        raise ContractFindingError(f"{field_name} must be non-empty")
    return value


def _boolean(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractFindingError(f"{field_name} must be a boolean")
    return value


def _integer(
    value: Any,
    *,
    field_name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractFindingError(f"{field_name} must be an integer")
    if value < minimum:
        raise ContractFindingError(f"{field_name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ContractFindingBoundsError(
            f"{field_name} exceeds maximum {maximum}"
        )
    return value


def _enum(value: Any, enum_type: type[E], *, field_name: str) -> E:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError as exc:
            raise ContractFindingError(
                f"{field_name} is not a valid {enum_type.__name__}"
            ) from exc
    raise ContractFindingError(
        f"{field_name} must be a {enum_type.__name__} or string"
    )


def _strings(
    values: Any,
    *,
    field_name: str,
    required: bool = False,
    maximum: int = MAX_COLLECTION_ITEMS,
    item_bytes: int = MAX_CLAUSE_BYTES,
    sort: bool = False,
    unique: bool = False,
) -> tuple[str, ...]:
    if values is None:
        if required:
            raise ContractFindingError(f"{field_name} is required")
        return ()
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ContractFindingError(
            f"{field_name} must be a sequence of strings"
        )
    if len(values) > maximum:
        raise ContractFindingBoundsError(
            f"{field_name} exceeds {maximum} items"
        )
    items: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(values):
        text = _text(
            raw,
            field_name=f"{field_name}[{index}]",
            required=True,
            maximum=item_bytes,
        )
        if unique and text in seen:
            raise ContractFindingError(
                f"{field_name} contains duplicate entry {text!r}"
            )
        seen.add(text)
        items.append(text)
    if sort:
        items = sorted(items)
    if required and not items:
        raise ContractFindingError(f"{field_name} must be non-empty")
    return tuple(items)


def _check_header(payload: Mapping[str, Any], expected_schema: str) -> None:
    if not isinstance(payload, Mapping):
        raise ContractFindingError("payload must be an object")
    schema = payload.get("schema")
    if schema != expected_schema:
        raise ContractFindingError(
            f"unsupported schema {schema!r}; expected {expected_schema!r}"
        )
    version = payload.get("schema_version", payload.get("contract_version"))
    if version is not None and int(version) != CONTRACT_FINDINGS_VERSION:
        raise ContractFindingError(
            f"unsupported schema_version {version!r}"
        )


def _reject_unknown(
    payload: Mapping[str, Any],
    allowed: set[str],
    *,
    artifact_name: str,
) -> None:
    unknown = set(payload) - allowed
    if unknown:
        raise ContractFindingError(
            f"{artifact_name} contains unknown fields: {sorted(unknown)}"
        )


def _check_identity(
    payload: Mapping[str, Any],
    actual: str,
    *,
    names: Sequence[str],
    artifact_name: str,
) -> None:
    for name in names:
        if name in payload and payload[name] not in (None, "", actual):
            raise ForgedFindingIdentityError(
                f"{artifact_name} {name} does not match derived identity"
            )


def _bounded(record: CanonicalContract, *, artifact_name: str) -> None:
    encoded = record.canonical_bytes()
    if len(encoded) > MAX_RECORD_BYTES:
        raise ContractFindingBoundsError(
            f"{artifact_name} exceeds {MAX_RECORD_BYTES} serialized bytes"
        )


def _record(
    value: Any,
    cls: type[T],
    *,
    field_name: str,
    optional: bool = False,
) -> T | None:
    if value is None:
        if optional:
            return None
        raise ContractFindingError(f"{field_name} is required")
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        from_dict = getattr(cls, "from_dict", None)
        if from_dict is None:
            raise ContractFindingError(
                f"{field_name} cannot be decoded from mapping"
            )
        return from_dict(value)
    raise ContractFindingError(
        f"{field_name} must be a {cls.__name__} or mapping"
    )


def _records(
    values: Any,
    cls: type[T],
    *,
    field_name: str,
    maximum: int = MAX_COLLECTION_ITEMS,
) -> tuple[T, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ContractFindingError(
            f"{field_name} must be a sequence of {cls.__name__}"
        )
    if len(values) > maximum:
        raise ContractFindingBoundsError(
            f"{field_name} exceeds {maximum} items"
        )
    result: list[T] = []
    for index, item in enumerate(values):
        field = f"{field_name}[{index}]"
        decoded = _record(item, cls, field_name=field)
        if decoded is None:
            raise ContractFindingError(f"{field} is required")
        result.append(decoded)
    return tuple(result)


def _header_fields() -> set[str]:
    return {
        "schema",
        "schema_version",
        "contract_version",
        "content_id",
        "cid",
    }


def finding_content_cid(payload: Mapping[str, Any] | Any) -> str:
    """Return a CIDv1 dag-json identity for a finding payload."""

    if hasattr(payload, "to_dict") and callable(payload.to_dict):
        material = payload.to_dict()
    elif isinstance(payload, Mapping):
        material = dict(payload)
    else:
        raise ContractFindingError("finding payload must be a mapping")
    # Prefer multiformats bridge; fall back to local content_identity.
    try:
        return cid_for_dag_json(material, for_identity=True)
    except Exception:
        return content_identity(material)


class _FindingContract(CanonicalContract):
    """Shared helpers for ledger records."""

    @property
    def schema_version(self) -> int:
        return CONTRACT_FINDINGS_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "schema_version": self.schema_version,
            "contract_version": CONTRACT_FINDINGS_VERSION,
            **self._payload(),
        }


@dataclass(frozen=True)
class CallSliceStep(_FindingContract):
    """One step on the shortest relevant call slice for a finding."""

    SCHEMA: ClassVar[str] = CALL_SLICE_STEP_SCHEMA

    symbol: str
    interface: str = ""
    repository_id: str = ""
    path: str = ""
    kind: str = "call"
    resolution: str = "resolved"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "symbol", _text(self.symbol, field_name="symbol")
        )
        for name in ("interface", "repository_id", "path", "kind", "resolution"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=False),
            )
        _bounded(self, artifact_name="call slice step")

    @property
    def step_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interface": self.interface,
            "repository_id": self.repository_id,
            "path": self.path,
            "kind": self.kind,
            "resolution": self.resolution,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "step_id": self.step_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CallSliceStep":
        _check_header(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            _header_fields()
            | {
                "symbol",
                "interface",
                "repository_id",
                "path",
                "kind",
                "resolution",
                "step_id",
            },
            artifact_name="call slice step",
        )
        result = cls(
            symbol=payload.get("symbol", ""),
            interface=payload.get("interface", ""),
            repository_id=payload.get("repository_id", ""),
            path=payload.get("path", ""),
            kind=payload.get("kind", "call"),
            resolution=payload.get("resolution", "resolved"),
        )
        _check_identity(
            payload,
            result.step_id,
            names=("step_id", "content_id", "cid"),
            artifact_name="call slice step",
        )
        return result


@dataclass(frozen=True)
class CallSlice(_FindingContract):
    """Shortest relevant call slice bound to a finding."""

    SCHEMA: ClassVar[str] = CALL_SLICE_SCHEMA

    steps: tuple[CallSliceStep, ...] = ()
    entry_symbol: str = ""
    exit_symbol: str = ""

    def __post_init__(self) -> None:
        steps = _records(
            self.steps,
            CallSliceStep,
            field_name="steps",
            maximum=MAX_CALL_SLICE_STEPS,
        )
        object.__setattr__(self, "steps", steps)
        object.__setattr__(
            self,
            "entry_symbol",
            _text(self.entry_symbol, field_name="entry_symbol", required=False),
        )
        object.__setattr__(
            self,
            "exit_symbol",
            _text(self.exit_symbol, field_name="exit_symbol", required=False),
        )
        if steps and not self.entry_symbol:
            object.__setattr__(self, "entry_symbol", steps[0].symbol)
        if steps and not self.exit_symbol:
            object.__setattr__(self, "exit_symbol", steps[-1].symbol)
        _bounded(self, artifact_name="call slice")

    @property
    def slice_id(self) -> str:
        return self.content_id

    @property
    def empty(self) -> bool:
        return not self.steps

    def _payload(self) -> dict[str, Any]:
        return {
            "steps": tuple(step.to_record() for step in self.steps),
            "entry_symbol": self.entry_symbol,
            "exit_symbol": self.exit_symbol,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "slice_id": self.slice_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CallSlice":
        _check_header(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            _header_fields()
            | {"steps", "entry_symbol", "exit_symbol", "slice_id"},
            artifact_name="call slice",
        )
        result = cls(
            steps=tuple(payload.get("steps") or ()),
            entry_symbol=payload.get("entry_symbol", ""),
            exit_symbol=payload.get("exit_symbol", ""),
        )
        _check_identity(
            payload,
            result.slice_id,
            names=("slice_id", "content_id", "cid"),
            artifact_name="call slice",
        )
        return result


@dataclass(frozen=True)
class EvidenceReferences(_FindingContract):
    """Compact references to counterexample, proof, runtime, and ZK evidence."""

    SCHEMA: ClassVar[str] = EVIDENCE_REFS_SCHEMA

    counterexample_cids: tuple[str, ...] = ()
    proof_cids: tuple[str, ...] = ()
    runtime_cids: tuple[str, ...] = ()
    zk_cids: tuple[str, ...] = ()
    artifact_cids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "counterexample_cids",
            "proof_cids",
            "runtime_cids",
            "zk_cids",
            "artifact_cids",
        ):
            object.__setattr__(
                self,
                name,
                _strings(
                    getattr(self, name),
                    field_name=name,
                    unique=True,
                    sort=True,
                ),
            )
        _bounded(self, artifact_name="evidence references")

    @property
    def refs_id(self) -> str:
        return self.content_id

    @property
    def empty(self) -> bool:
        return not (
            self.counterexample_cids
            or self.proof_cids
            or self.runtime_cids
            or self.zk_cids
            or self.artifact_cids
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "counterexample_cids": self.counterexample_cids,
            "proof_cids": self.proof_cids,
            "runtime_cids": self.runtime_cids,
            "zk_cids": self.zk_cids,
            "artifact_cids": self.artifact_cids,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "refs_id": self.refs_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceReferences":
        _check_header(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            _header_fields()
            | {
                "counterexample_cids",
                "proof_cids",
                "runtime_cids",
                "zk_cids",
                "artifact_cids",
                "refs_id",
            },
            artifact_name="evidence references",
        )
        result = cls(
            counterexample_cids=tuple(payload.get("counterexample_cids") or ()),
            proof_cids=tuple(payload.get("proof_cids") or ()),
            runtime_cids=tuple(payload.get("runtime_cids") or ()),
            zk_cids=tuple(payload.get("zk_cids") or ()),
            artifact_cids=tuple(payload.get("artifact_cids") or ()),
        )
        _check_identity(
            payload,
            result.refs_id,
            names=("refs_id", "content_id", "cid"),
            artifact_name="evidence references",
        )
        return result


@dataclass(frozen=True)
class AnalyzerVersions(_FindingContract):
    """Deterministic analyzer name → version bindings for a finding."""

    SCHEMA: ClassVar[str] = ANALYZER_VERSIONS_SCHEMA

    versions: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        raw = self.versions
        if raw is None:
            pairs: list[tuple[str, str]] = []
        elif isinstance(raw, Mapping):
            pairs = [
                (
                    _text(key, field_name="analyzer name"),
                    _text(value, field_name="analyzer version"),
                )
                for key, value in raw.items()
            ]
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            pairs = []
            for index, item in enumerate(raw):
                if (
                    isinstance(item, Sequence)
                    and not isinstance(item, (str, bytes))
                    and len(item) == 2
                ):
                    pairs.append(
                        (
                            _text(item[0], field_name=f"versions[{index}].name"),
                            _text(
                                item[1], field_name=f"versions[{index}].version"
                            ),
                        )
                    )
                else:
                    raise ContractFindingError(
                        "analyzer versions must be name/version pairs"
                    )
        else:
            raise ContractFindingError(
                "analyzer versions must be a mapping or sequence of pairs"
            )
        if len(pairs) > MAX_ANALYZER_VERSIONS:
            raise ContractFindingBoundsError(
                f"analyzer versions exceed {MAX_ANALYZER_VERSIONS} items"
            )
        names = [name for name, _ in pairs]
        if len(names) != len(set(names)):
            raise ContractFindingError("analyzer versions contain duplicate names")
        object.__setattr__(self, "versions", tuple(sorted(pairs, key=lambda p: p[0])))
        _bounded(self, artifact_name="analyzer versions")

    @property
    def versions_id(self) -> str:
        return self.content_id

    def as_mapping(self) -> dict[str, str]:
        return {name: version for name, version in self.versions}

    def _payload(self) -> dict[str, Any]:
        return {
            "versions": tuple(
                {"name": name, "version": version}
                for name, version in self.versions
            )
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "versions_id": self.versions_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnalyzerVersions":
        _check_header(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            _header_fields() | {"versions", "versions_id"},
            artifact_name="analyzer versions",
        )
        raw = payload.get("versions") or ()
        pairs: list[tuple[str, str]] = []
        if isinstance(raw, Mapping):
            pairs = list(raw.items())
        else:
            for item in raw:
                if isinstance(item, Mapping):
                    pairs.append(
                        (item.get("name", ""), item.get("version", ""))
                    )
                elif isinstance(item, Sequence) and len(item) == 2:
                    pairs.append((item[0], item[1]))
        result = cls(versions=tuple(pairs))
        _check_identity(
            payload,
            result.versions_id,
            names=("versions_id", "content_id", "cid"),
            artifact_name="analyzer versions",
        )
        return result


@dataclass(frozen=True)
class SemanticDedupKey(_FindingContract):
    """Narrow identity used for finding deduplication.

    Findings deduplicate only when contract, root cause, affected symbols,
    and merge fate agree.  Severity, confidence, claim level, and freshness
    do **not** participate.
    """

    SCHEMA: ClassVar[str] = SEMANTIC_DEDUP_KEY_SCHEMA

    expected_contract_cid: str
    observed_contract_cid: str
    root_cause_family: str
    merge_fate: str
    symbols: tuple[str, ...] = ()
    interfaces: tuple[str, ...] = ()
    repositories: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "expected_contract_cid",
            "observed_contract_cid",
            "root_cause_family",
            "merge_fate",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=True),
            )
        object.__setattr__(
            self,
            "symbols",
            _strings(self.symbols, field_name="symbols", unique=True, sort=True),
        )
        object.__setattr__(
            self,
            "interfaces",
            _strings(
                self.interfaces, field_name="interfaces", unique=True, sort=True
            ),
        )
        object.__setattr__(
            self,
            "repositories",
            _strings(
                self.repositories,
                field_name="repositories",
                unique=True,
                sort=True,
            ),
        )
        _bounded(self, artifact_name="semantic dedup key")

    @property
    def key_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "expected_contract_cid": self.expected_contract_cid,
            "observed_contract_cid": self.observed_contract_cid,
            "root_cause_family": self.root_cause_family,
            "merge_fate": self.merge_fate,
            "symbols": self.symbols,
            "interfaces": self.interfaces,
            "repositories": self.repositories,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "key_id": self.key_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SemanticDedupKey":
        _check_header(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            _header_fields()
            | {
                "expected_contract_cid",
                "observed_contract_cid",
                "root_cause_family",
                "merge_fate",
                "symbols",
                "interfaces",
                "repositories",
                "key_id",
            },
            artifact_name="semantic dedup key",
        )
        result = cls(
            expected_contract_cid=payload.get("expected_contract_cid", ""),
            observed_contract_cid=payload.get("observed_contract_cid", ""),
            root_cause_family=payload.get("root_cause_family", ""),
            merge_fate=payload.get("merge_fate", ""),
            symbols=tuple(payload.get("symbols") or ()),
            interfaces=tuple(payload.get("interfaces") or ()),
            repositories=tuple(payload.get("repositories") or ()),
        )
        _check_identity(
            payload,
            result.key_id,
            names=("key_id", "content_id", "cid"),
            artifact_name="semantic dedup key",
        )
        return result


def validate_severity_binding(
    *,
    status: FindingStatus | str,
    severity: FindingSeverity | str,
    claim_level: ClaimLevel | str,
    confidence_millionths: int,
    freshness: EvidenceFreshness | str,
    has_counterexample: bool,
) -> None:
    """Reject severity inflated beyond status, claim, confidence, or evidence."""

    status_e = _enum(status, FindingStatus, field_name="status")
    severity_e = _enum(severity, FindingSeverity, field_name="severity")
    claim_e = _enum(claim_level, ClaimLevel, field_name="claim_level")
    freshness_e = _enum(freshness, EvidenceFreshness, field_name="freshness")
    confidence = _integer(
        confidence_millionths,
        field_name="confidence_millionths",
        minimum=0,
        maximum=MILLION,
    )

    max_for_status = _MAX_SEVERITY_FOR_STATUS[status_e]
    if _SEVERITY_RANK[severity_e] > _SEVERITY_RANK[max_for_status]:
        raise PoisonedSeverityError(
            f"severity {severity_e.value} exceeds maximum "
            f"{max_for_status.value} for status {status_e.value}"
        )

    min_confidence = _MIN_CONFIDENCE_FOR_SEVERITY[severity_e]
    if confidence < min_confidence:
        raise PoisonedSeverityError(
            f"severity {severity_e.value} requires confidence "
            f">= {min_confidence}; got {confidence}"
        )

    if severity_e is FindingSeverity.CRITICAL:
        if status_e is not FindingStatus.CONTRACT_BROKEN:
            raise PoisonedSeverityError(
                "critical severity requires contract_broken status"
            )
        if claim_e not in {
            ClaimLevel.MODEL_DISPROVED,
            ClaimLevel.RUNTIME_WITNESSED,
        }:
            raise PoisonedSeverityError(
                "critical severity requires model_disproved or runtime_witnessed"
            )
        if not has_counterexample:
            raise PoisonedSeverityError(
                "critical severity requires a counterexample reference"
            )
        if freshness_e is EvidenceFreshness.STALE:
            raise PoisonedSeverityError(
                "critical severity cannot bind stale evidence"
            )

    if status_e is FindingStatus.CONTRACT_BROKEN:
        if claim_e not in {
            ClaimLevel.MODEL_DISPROVED,
            ClaimLevel.RUNTIME_WITNESSED,
        }:
            raise PoisonedSeverityError(
                "contract_broken requires model_disproved or runtime_witnessed"
            )
        if not has_counterexample:
            raise PoisonedSeverityError(
                "contract_broken requires a counterexample reference"
            )


def is_partial_finding(
    *,
    repositories: Sequence[str],
    symbols: Sequence[str],
    interfaces: Sequence[str],
    expected_contract_cid: str,
    observed_contract_cid: str,
    root_cause_family: str,
    merge_fate: str,
    claim_level: ClaimLevel | str | None,
    status: FindingStatus | str | None,
) -> tuple[bool, tuple[str, ...]]:
    """Return whether required admission fields are missing, with reasons."""

    missing: list[str] = []
    if not repositories:
        missing.append("repositories")
    if not symbols:
        missing.append("symbols")
    if not interfaces:
        missing.append("interfaces")
    if not expected_contract_cid:
        missing.append("expected_contract_cid")
    if not observed_contract_cid:
        missing.append("observed_contract_cid")
    if not root_cause_family:
        missing.append("root_cause_family")
    if not merge_fate:
        missing.append("merge_fate")
    if claim_level is None or claim_level == "":
        missing.append("claim_level")
    if status is None or status == "":
        missing.append("status")
    return (bool(missing), tuple(missing))


def is_vulnerability_labeled(labels: Sequence[str] | None) -> bool:
    """True when the finding carries the closed ``vulnerability`` label."""

    if not labels:
        return False
    return any(label == VULNERABILITY_LABEL for label in labels)


def vulnerability_evidence_requirements_met(
    *,
    labels: Sequence[str] | None,
    threat_path_cid: str | None,
    impact: str | None,
) -> tuple[bool, tuple[str, ...]]:
    """Return whether a vulnerability label is justified, with missing keys.

    Implements ``vfs/vulnerability-evidence-policy@1``: the ``vulnerability``
    label requires both a content-addressed threat-path reference and a
    concrete impact statement.  Correctness-only findings need neither.
    """

    if not is_vulnerability_labeled(labels):
        return (True, ())
    missing: list[str] = []
    if not (threat_path_cid or "").strip():
        missing.append("threat_path_cid")
    if not (impact or "").strip():
        missing.append("impact")
    return (not missing, tuple(missing))


def validate_vulnerability_evidence_policy(
    *,
    labels: Sequence[str] | None,
    threat_path_cid: str | None,
    impact: str | None,
) -> None:
    """Reject vulnerability labels that lack threat path and impact."""

    ok, missing = vulnerability_evidence_requirements_met(
        labels=labels,
        threat_path_cid=threat_path_cid,
        impact=impact,
    )
    if ok:
        return
    raise VulnerabilityEvidencePolicyError(
        "vulnerability label requires threat path and impact; missing "
        + ", ".join(missing)
    )


def finding_ledger_evidence_terms() -> tuple[str, ...]:
    """Return the closed VFS-G100 evidence terms covered by this ledger.

    Proves that the typed, deduplicated correctness and vulnerability ledger
    exists as a first-class content-addressed artifact
    (``vfs/finding-ledger@1``) with a separate vulnerability evidence policy
    (``vfs/vulnerability-evidence-policy@1``).
    """

    return FINDING_LEDGER_G100_EVIDENCE_TERMS


def covered_evidence_terms() -> tuple[str, ...]:
    """Alias of :func:`finding_ledger_evidence_terms` for discovery scanners."""

    return finding_ledger_evidence_terms()


@dataclass(frozen=True)
class ContractFindingRecord(_FindingContract):
    """Immutable, content-addressed contract finding ledger record."""

    SCHEMA: ClassVar[str] = CONTRACT_FINDING_RECORD_SCHEMA

    claim_level: ClaimLevel
    status: FindingStatus
    severity: FindingSeverity
    confidence_millionths: int
    freshness: EvidenceFreshness
    repositories: tuple[str, ...]
    symbols: tuple[str, ...]
    interfaces: tuple[str, ...]
    expected_contract_cid: str
    observed_contract_cid: str
    root_cause_family: str
    merge_fate: str
    summary: str
    call_slice: CallSlice = field(default_factory=CallSlice)
    evidence: EvidenceReferences = field(default_factory=EvidenceReferences)
    assumptions: tuple[str, ...] = ()
    analyzer_versions: AnalyzerVersions = field(
        default_factory=AnalyzerVersions
    )
    remediation_scope: tuple[str, ...] = ()
    supersedes_cids: tuple[str, ...] = ()
    superseded_by_cid: str = ""
    rejection_reasons: tuple[str, ...] = ()
    tree_id: str = ""
    policy_revision: str = ""
    repository_observation_id: str = ""
    verdict: str = ""
    # Vulnerability evidence policy (vfs/vulnerability-evidence-policy@1):
    # the closed "vulnerability" label requires threat_path_cid + impact.
    labels: tuple[str, ...] = ()
    threat_path_cid: str = ""
    impact: str = ""
    partial: bool = False
    partial_missing_fields: tuple[str, ...] = ()
    allow_poisoned_severity: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "claim_level",
            _enum(self.claim_level, ClaimLevel, field_name="claim_level"),
        )
        object.__setattr__(
            self,
            "status",
            _enum(self.status, FindingStatus, field_name="status"),
        )
        object.__setattr__(
            self,
            "severity",
            _enum(self.severity, FindingSeverity, field_name="severity"),
        )
        object.__setattr__(
            self,
            "freshness",
            _enum(self.freshness, EvidenceFreshness, field_name="freshness"),
        )
        object.__setattr__(
            self,
            "confidence_millionths",
            _integer(
                self.confidence_millionths,
                field_name="confidence_millionths",
                minimum=0,
                maximum=MILLION,
            ),
        )
        object.__setattr__(
            self,
            "repositories",
            _strings(
                self.repositories,
                field_name="repositories",
                unique=True,
                sort=True,
            ),
        )
        object.__setattr__(
            self,
            "symbols",
            _strings(
                self.symbols, field_name="symbols", unique=True, sort=True
            ),
        )
        object.__setattr__(
            self,
            "interfaces",
            _strings(
                self.interfaces,
                field_name="interfaces",
                unique=True,
                sort=True,
            ),
        )
        for name in (
            "expected_contract_cid",
            "observed_contract_cid",
            "root_cause_family",
            "merge_fate",
            "summary",
        ):
            # Partial records may omit some of these; completeness is tracked.
            required = name == "summary"
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name),
                    field_name=name,
                    required=required,
                )
                if required
                else _text(
                    getattr(self, name) or "",
                    field_name=name,
                    required=False,
                ),
            )
        object.__setattr__(
            self,
            "call_slice",
            _record(
                self.call_slice,
                CallSlice,
                field_name="call_slice",
                optional=True,
            )
            or CallSlice(),
        )
        object.__setattr__(
            self,
            "evidence",
            _record(
                self.evidence,
                EvidenceReferences,
                field_name="evidence",
                optional=True,
            )
            or EvidenceReferences(),
        )
        object.__setattr__(
            self,
            "assumptions",
            _strings(self.assumptions, field_name="assumptions"),
        )
        object.__setattr__(
            self,
            "analyzer_versions",
            _record(
                self.analyzer_versions,
                AnalyzerVersions,
                field_name="analyzer_versions",
                optional=True,
            )
            or AnalyzerVersions(),
        )
        object.__setattr__(
            self,
            "remediation_scope",
            _strings(
                self.remediation_scope,
                field_name="remediation_scope",
                maximum=64,
            ),
        )
        object.__setattr__(
            self,
            "supersedes_cids",
            _strings(
                self.supersedes_cids,
                field_name="supersedes_cids",
                unique=True,
                sort=True,
            ),
        )
        object.__setattr__(
            self,
            "superseded_by_cid",
            _text(
                self.superseded_by_cid or "",
                field_name="superseded_by_cid",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "rejection_reasons",
            _strings(
                self.rejection_reasons,
                field_name="rejection_reasons",
                maximum=MAX_REJECTION_REASONS,
            ),
        )
        for name in (
            "tree_id",
            "policy_revision",
            "repository_observation_id",
            "verdict",
            "threat_path_cid",
            "impact",
        ):
            object.__setattr__(
                self,
                name,
                _text(
                    getattr(self, name) or "",
                    field_name=name,
                    required=False,
                ),
            )
        object.__setattr__(
            self,
            "labels",
            _strings(
                self.labels,
                field_name="labels",
                unique=True,
                sort=True,
                maximum=MAX_LABELS,
            ),
        )
        object.__setattr__(
            self, "partial", _boolean(self.partial, field_name="partial")
        )
        object.__setattr__(
            self,
            "partial_missing_fields",
            _strings(
                self.partial_missing_fields,
                field_name="partial_missing_fields",
                unique=True,
                sort=True,
            ),
        )
        object.__setattr__(
            self,
            "allow_poisoned_severity",
            _boolean(
                self.allow_poisoned_severity,
                field_name="allow_poisoned_severity",
            ),
        )

        is_partial, missing = is_partial_finding(
            repositories=self.repositories,
            symbols=self.symbols,
            interfaces=self.interfaces,
            expected_contract_cid=self.expected_contract_cid,
            observed_contract_cid=self.observed_contract_cid,
            root_cause_family=self.root_cause_family,
            merge_fate=self.merge_fate,
            claim_level=self.claim_level,
            status=self.status,
        )
        if is_partial:
            object.__setattr__(self, "partial", True)
            if not self.partial_missing_fields:
                object.__setattr__(self, "partial_missing_fields", missing)
        elif self.partial and not self.partial_missing_fields:
            # Explicit partial flag without missing required fields is allowed
            # for staged admissions; keep the flag as given.
            pass

        if not self.partial and not self.allow_poisoned_severity:
            has_cex = bool(self.evidence.counterexample_cids)
            validate_severity_binding(
                status=self.status,
                severity=self.severity,
                claim_level=self.claim_level,
                confidence_millionths=self.confidence_millionths,
                freshness=self.freshness,
                has_counterexample=has_cex,
            )

        # Vulnerability labels are fail-closed even on partial records: the
        # label may not be claimed without threat path and impact evidence.
        validate_vulnerability_evidence_policy(
            labels=self.labels,
            threat_path_cid=self.threat_path_cid,
            impact=self.impact,
        )

        if (
            self.freshness is EvidenceFreshness.STALE
            and self.status is FindingStatus.CONTRACT_BROKEN
            and not self.partial
        ):
            raise StaleFindingError(
                "contract_broken findings cannot bind stale evidence"
            )

        _bounded(self, artifact_name="contract finding record")

    @property
    def finding_cid(self) -> str:
        return self.content_id

    @property
    def cid(self) -> str:
        return self.finding_cid

    @property
    def finding_id(self) -> str:
        return self.finding_cid

    @property
    def semantic_key(self) -> SemanticDedupKey:
        return SemanticDedupKey(
            expected_contract_cid=self.expected_contract_cid or "unset",
            observed_contract_cid=self.observed_contract_cid or "unset",
            root_cause_family=self.root_cause_family or "unset",
            merge_fate=self.merge_fate or "unset",
            symbols=self.symbols,
            interfaces=self.interfaces,
            repositories=self.repositories,
        )

    @property
    def semantic_key_id(self) -> str:
        return self.semantic_key.key_id

    @property
    def actionable(self) -> bool:
        return (
            not self.partial
            and not self.rejection_reasons
            and not self.superseded_by_cid
            and self.freshness is EvidenceFreshness.CURRENT
            and self.status is FindingStatus.CONTRACT_BROKEN
        )

    @property
    def is_vulnerability(self) -> bool:
        """True when the closed vulnerability label is bound with evidence."""

        return is_vulnerability_labeled(self.labels)

    def with_updates(self, **changes: Any) -> "ContractFindingRecord":
        """Return a new record with selected fields replaced (immutable copy)."""

        payload = {
            "claim_level": self.claim_level,
            "status": self.status,
            "severity": self.severity,
            "confidence_millionths": self.confidence_millionths,
            "freshness": self.freshness,
            "repositories": self.repositories,
            "symbols": self.symbols,
            "interfaces": self.interfaces,
            "expected_contract_cid": self.expected_contract_cid,
            "observed_contract_cid": self.observed_contract_cid,
            "root_cause_family": self.root_cause_family,
            "merge_fate": self.merge_fate,
            "summary": self.summary,
            "call_slice": self.call_slice,
            "evidence": self.evidence,
            "assumptions": self.assumptions,
            "analyzer_versions": self.analyzer_versions,
            "remediation_scope": self.remediation_scope,
            "supersedes_cids": self.supersedes_cids,
            "superseded_by_cid": self.superseded_by_cid,
            "rejection_reasons": self.rejection_reasons,
            "tree_id": self.tree_id,
            "policy_revision": self.policy_revision,
            "repository_observation_id": self.repository_observation_id,
            "verdict": self.verdict,
            "labels": self.labels,
            "threat_path_cid": self.threat_path_cid,
            "impact": self.impact,
            "partial": self.partial,
            "partial_missing_fields": self.partial_missing_fields,
            "allow_poisoned_severity": self.allow_poisoned_severity,
        }
        payload.update(changes)
        return ContractFindingRecord(**payload)

    def _payload(self) -> dict[str, Any]:
        return {
            "claim_level": self.claim_level.value,
            "status": self.status.value,
            "severity": self.severity.value,
            "confidence_millionths": self.confidence_millionths,
            "freshness": self.freshness.value,
            "repositories": self.repositories,
            "symbols": self.symbols,
            "interfaces": self.interfaces,
            "expected_contract_cid": self.expected_contract_cid,
            "observed_contract_cid": self.observed_contract_cid,
            "root_cause_family": self.root_cause_family,
            "merge_fate": self.merge_fate,
            "summary": self.summary,
            "call_slice": self.call_slice.to_record(),
            "evidence": self.evidence.to_record(),
            "assumptions": self.assumptions,
            "analyzer_versions": self.analyzer_versions.to_record(),
            "remediation_scope": self.remediation_scope,
            "supersedes_cids": self.supersedes_cids,
            "superseded_by_cid": self.superseded_by_cid,
            "rejection_reasons": self.rejection_reasons,
            "tree_id": self.tree_id,
            "policy_revision": self.policy_revision,
            "repository_observation_id": self.repository_observation_id,
            "verdict": self.verdict,
            "labels": self.labels,
            "threat_path_cid": self.threat_path_cid,
            "impact": self.impact,
            "partial": self.partial,
            "partial_missing_fields": self.partial_missing_fields,
            "semantic_key_id": self.semantic_key_id,
            "actionable": self.actionable,
            "is_vulnerability": self.is_vulnerability,
        }

    def to_record(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "finding_cid": self.finding_cid,
            "finding_id": self.finding_id,
            "cid": self.cid,
            "content_id": self.content_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContractFindingRecord":
        _check_header(payload, cls.SCHEMA)
        fields = {
            "claim_level",
            "status",
            "severity",
            "confidence_millionths",
            "freshness",
            "repositories",
            "symbols",
            "interfaces",
            "expected_contract_cid",
            "observed_contract_cid",
            "root_cause_family",
            "merge_fate",
            "summary",
            "call_slice",
            "evidence",
            "assumptions",
            "analyzer_versions",
            "remediation_scope",
            "supersedes_cids",
            "superseded_by_cid",
            "rejection_reasons",
            "tree_id",
            "policy_revision",
            "repository_observation_id",
            "verdict",
            "labels",
            "threat_path_cid",
            "impact",
            "partial",
            "partial_missing_fields",
            "allow_poisoned_severity",
        }
        _reject_unknown(
            payload,
            fields
            | _header_fields()
            | {
                "finding_cid",
                "finding_id",
                "semantic_key_id",
                "actionable",
                "is_vulnerability",
            },
            artifact_name="contract finding record",
        )
        result = cls(
            claim_level=payload.get("claim_level", ""),
            status=payload.get("status", ""),
            severity=payload.get("severity", ""),
            confidence_millionths=payload.get("confidence_millionths", 0),
            freshness=payload.get("freshness", ""),
            repositories=tuple(payload.get("repositories") or ()),
            symbols=tuple(payload.get("symbols") or ()),
            interfaces=tuple(payload.get("interfaces") or ()),
            expected_contract_cid=payload.get("expected_contract_cid", ""),
            observed_contract_cid=payload.get("observed_contract_cid", ""),
            root_cause_family=payload.get("root_cause_family", ""),
            merge_fate=payload.get("merge_fate", ""),
            summary=payload.get("summary", ""),
            call_slice=payload.get("call_slice") or CallSlice(),
            evidence=payload.get("evidence") or EvidenceReferences(),
            assumptions=tuple(payload.get("assumptions") or ()),
            analyzer_versions=payload.get("analyzer_versions")
            or AnalyzerVersions(),
            remediation_scope=tuple(payload.get("remediation_scope") or ()),
            supersedes_cids=tuple(payload.get("supersedes_cids") or ()),
            superseded_by_cid=payload.get("superseded_by_cid", ""),
            rejection_reasons=tuple(payload.get("rejection_reasons") or ()),
            tree_id=payload.get("tree_id", ""),
            policy_revision=payload.get("policy_revision", ""),
            repository_observation_id=payload.get(
                "repository_observation_id", ""
            ),
            verdict=payload.get("verdict", ""),
            labels=tuple(payload.get("labels") or ()),
            threat_path_cid=payload.get("threat_path_cid", ""),
            impact=payload.get("impact", ""),
            partial=payload.get("partial", False),
            partial_missing_fields=tuple(
                payload.get("partial_missing_fields") or ()
            ),
            allow_poisoned_severity=payload.get(
                "allow_poisoned_severity", False
            ),
        )
        if "semantic_key_id" in payload and payload[
            "semantic_key_id"
        ] != result.semantic_key_id:
            raise ForgedFindingIdentityError(
                "finding semantic_key_id does not match derived key"
            )
        if "actionable" in payload and bool(payload["actionable"]) != result.actionable:
            raise ForgedFindingIdentityError(
                "finding actionable projection does not match derived state"
            )
        if "is_vulnerability" in payload and bool(
            payload["is_vulnerability"]
        ) != result.is_vulnerability:
            raise ForgedFindingIdentityError(
                "finding is_vulnerability projection does not match labels"
            )
        _check_identity(
            payload,
            result.finding_cid,
            names=("finding_cid", "finding_id", "content_id", "cid"),
            artifact_name="contract finding record",
        )
        return result


@dataclass(frozen=True)
class FindingProjectionEntry(_FindingContract):
    """Current-tree projection entry pointing at an immutable finding CID."""

    SCHEMA: ClassVar[str] = FINDING_PROJECTION_ENTRY_SCHEMA

    finding_cid: str
    semantic_key_id: str
    admission: FindingAdmissionState
    tree_id: str = ""
    superseded_by_cid: str = ""
    rejection_reasons: tuple[str, ...] = ()
    conflict_cids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "finding_cid",
            _text(self.finding_cid, field_name="finding_cid"),
        )
        object.__setattr__(
            self,
            "semantic_key_id",
            _text(self.semantic_key_id, field_name="semantic_key_id"),
        )
        object.__setattr__(
            self,
            "admission",
            _enum(
                self.admission, FindingAdmissionState, field_name="admission"
            ),
        )
        object.__setattr__(
            self,
            "tree_id",
            _text(self.tree_id or "", field_name="tree_id", required=False),
        )
        object.__setattr__(
            self,
            "superseded_by_cid",
            _text(
                self.superseded_by_cid or "",
                field_name="superseded_by_cid",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "rejection_reasons",
            _strings(
                self.rejection_reasons,
                field_name="rejection_reasons",
                maximum=MAX_REJECTION_REASONS,
            ),
        )
        object.__setattr__(
            self,
            "conflict_cids",
            _strings(
                self.conflict_cids,
                field_name="conflict_cids",
                unique=True,
                sort=True,
            ),
        )
        _bounded(self, artifact_name="finding projection entry")

    @property
    def entry_id(self) -> str:
        return self.content_id

    @property
    def current(self) -> bool:
        return self.admission in {
            FindingAdmissionState.ADMITTED,
            FindingAdmissionState.CONFLICT,
            FindingAdmissionState.PARTIAL,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "finding_cid": self.finding_cid,
            "semantic_key_id": self.semantic_key_id,
            "admission": self.admission.value,
            "tree_id": self.tree_id,
            "superseded_by_cid": self.superseded_by_cid,
            "rejection_reasons": self.rejection_reasons,
            "conflict_cids": self.conflict_cids,
            "current": self.current,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "entry_id": self.entry_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FindingProjectionEntry":
        _check_header(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            _header_fields()
            | {
                "finding_cid",
                "semantic_key_id",
                "admission",
                "tree_id",
                "superseded_by_cid",
                "rejection_reasons",
                "conflict_cids",
                "current",
                "entry_id",
            },
            artifact_name="finding projection entry",
        )
        result = cls(
            finding_cid=payload.get("finding_cid", ""),
            semantic_key_id=payload.get("semantic_key_id", ""),
            admission=payload.get("admission", ""),
            tree_id=payload.get("tree_id", ""),
            superseded_by_cid=payload.get("superseded_by_cid", ""),
            rejection_reasons=tuple(payload.get("rejection_reasons") or ()),
            conflict_cids=tuple(payload.get("conflict_cids") or ()),
        )
        if "current" in payload and bool(payload["current"]) != result.current:
            raise ForgedFindingIdentityError(
                "projection current flag does not match admission state"
            )
        _check_identity(
            payload,
            result.entry_id,
            names=("entry_id", "content_id", "cid"),
            artifact_name="finding projection entry",
        )
        return result


@dataclass(frozen=True)
class LedgerEvent(_FindingContract):
    """One append-only ledger event.  Events are never mutated or deleted."""

    SCHEMA: ClassVar[str] = LEDGER_EVENT_SCHEMA

    kind: LedgerEventKind
    finding_cid: str
    sequence: int
    semantic_key_id: str = ""
    related_cids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    tree_id: str = ""
    payload_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "kind", _enum(self.kind, LedgerEventKind, field_name="kind")
        )
        object.__setattr__(
            self,
            "finding_cid",
            _text(self.finding_cid, field_name="finding_cid"),
        )
        object.__setattr__(
            self,
            "sequence",
            _integer(self.sequence, field_name="sequence", minimum=0),
        )
        object.__setattr__(
            self,
            "semantic_key_id",
            _text(
                self.semantic_key_id or "",
                field_name="semantic_key_id",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "related_cids",
            _strings(
                self.related_cids,
                field_name="related_cids",
                unique=True,
                sort=True,
            ),
        )
        object.__setattr__(
            self,
            "reasons",
            _strings(self.reasons, field_name="reasons", maximum=MAX_REJECTION_REASONS),
        )
        object.__setattr__(
            self,
            "tree_id",
            _text(self.tree_id or "", field_name="tree_id", required=False),
        )
        object.__setattr__(
            self,
            "payload_digest",
            _text(
                self.payload_digest or "",
                field_name="payload_digest",
                required=False,
            ),
        )
        _bounded(self, artifact_name="ledger event")

    @property
    def event_id(self) -> str:
        return self.content_id

    def _payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "finding_cid": self.finding_cid,
            "sequence": self.sequence,
            "semantic_key_id": self.semantic_key_id,
            "related_cids": self.related_cids,
            "reasons": self.reasons,
            "tree_id": self.tree_id,
            "payload_digest": self.payload_digest,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "event_id": self.event_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LedgerEvent":
        _check_header(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            _header_fields()
            | {
                "kind",
                "finding_cid",
                "sequence",
                "semantic_key_id",
                "related_cids",
                "reasons",
                "tree_id",
                "payload_digest",
                "event_id",
            },
            artifact_name="ledger event",
        )
        result = cls(
            kind=payload.get("kind", ""),
            finding_cid=payload.get("finding_cid", ""),
            sequence=payload.get("sequence", 0),
            semantic_key_id=payload.get("semantic_key_id", ""),
            related_cids=tuple(payload.get("related_cids") or ()),
            reasons=tuple(payload.get("reasons") or ()),
            tree_id=payload.get("tree_id", ""),
            payload_digest=payload.get("payload_digest", ""),
        )
        _check_identity(
            payload,
            result.event_id,
            names=("event_id", "content_id", "cid"),
            artifact_name="ledger event",
        )
        return result


@dataclass(frozen=True)
class AppendReceipt(_FindingContract):
    """Receipt returned by an append / reject / supersede operation."""

    SCHEMA: ClassVar[str] = APPEND_RECEIPT_SCHEMA

    outcome: AppendOutcome
    finding_cid: str
    sequence: int
    semantic_key_id: str = ""
    admission: FindingAdmissionState = FindingAdmissionState.ADMITTED
    prior_finding_cid: str = ""
    reasons: tuple[str, ...] = ()
    event_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outcome",
            _enum(self.outcome, AppendOutcome, field_name="outcome"),
        )
        object.__setattr__(
            self,
            "finding_cid",
            _text(self.finding_cid, field_name="finding_cid"),
        )
        object.__setattr__(
            self,
            "sequence",
            _integer(self.sequence, field_name="sequence", minimum=0),
        )
        object.__setattr__(
            self,
            "semantic_key_id",
            _text(
                self.semantic_key_id or "",
                field_name="semantic_key_id",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "admission",
            _enum(
                self.admission, FindingAdmissionState, field_name="admission"
            ),
        )
        object.__setattr__(
            self,
            "prior_finding_cid",
            _text(
                self.prior_finding_cid or "",
                field_name="prior_finding_cid",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "reasons",
            _strings(self.reasons, field_name="reasons"),
        )
        object.__setattr__(
            self,
            "event_id",
            _text(self.event_id or "", field_name="event_id", required=False),
        )
        _bounded(self, artifact_name="append receipt")

    @property
    def receipt_id(self) -> str:
        return self.content_id

    @property
    def stored(self) -> bool:
        return self.outcome in {
            AppendOutcome.STORED,
            AppendOutcome.SUPERSEDED_PRIOR,
        }

    def _payload(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "finding_cid": self.finding_cid,
            "sequence": self.sequence,
            "semantic_key_id": self.semantic_key_id,
            "admission": self.admission.value,
            "prior_finding_cid": self.prior_finding_cid,
            "reasons": self.reasons,
            "event_id": self.event_id,
            "stored": self.stored,
            "evidence": FINDING_LEDGER_EVIDENCE,
            "evidence_terms": FINDING_LEDGER_G100_EVIDENCE_TERMS,
            "goal_id": GOAL_ID,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "receipt_id": self.receipt_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AppendReceipt":
        _check_header(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            _header_fields()
            | {
                "outcome",
                "finding_cid",
                "sequence",
                "semantic_key_id",
                "admission",
                "prior_finding_cid",
                "reasons",
                "event_id",
                "stored",
                "receipt_id",
                "evidence",
                "evidence_terms",
                "goal_id",
            },
            artifact_name="append receipt",
        )
        result = cls(
            outcome=payload.get("outcome", ""),
            finding_cid=payload.get("finding_cid", ""),
            sequence=payload.get("sequence", 0),
            semantic_key_id=payload.get("semantic_key_id", ""),
            admission=payload.get(
                "admission", FindingAdmissionState.ADMITTED
            ),
            prior_finding_cid=payload.get("prior_finding_cid", ""),
            reasons=tuple(payload.get("reasons") or ()),
            event_id=payload.get("event_id", ""),
        )
        if "stored" in payload and bool(payload["stored"]) != result.stored:
            raise ForgedFindingIdentityError(
                "append receipt stored projection does not match outcome"
            )
        if "evidence" in payload and payload["evidence"] not in (
            None,
            "",
            FINDING_LEDGER_EVIDENCE,
        ):
            raise ForgedFindingIdentityError(
                f"append receipt evidence must be {FINDING_LEDGER_EVIDENCE!r}"
            )
        if "goal_id" in payload and payload["goal_id"] not in (
            None,
            "",
            GOAL_ID,
        ):
            raise ForgedFindingIdentityError(
                f"append receipt goal_id must be {GOAL_ID!r}"
            )
        claimed_terms = payload.get("evidence_terms")
        if claimed_terms is not None and tuple(claimed_terms) != (
            FINDING_LEDGER_G100_EVIDENCE_TERMS
        ):
            raise ForgedFindingIdentityError(
                "append receipt evidence_terms do not match VFS-G100 terms"
            )
        _check_identity(
            payload,
            result.receipt_id,
            names=("receipt_id", "content_id", "cid"),
            artifact_name="append receipt",
        )
        return result


@dataclass(frozen=True)
class ProjectionSnapshot(_FindingContract):
    """Deterministic snapshot of the current projection over history."""

    SCHEMA: ClassVar[str] = PROJECTION_SNAPSHOT_SCHEMA

    entries: tuple[FindingProjectionEntry, ...] = ()
    history_length: int = 0
    tree_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "entries",
            _records(
                self.entries,
                FindingProjectionEntry,
                field_name="entries",
                maximum=MAX_PROJECTION_ENTRIES,
            ),
        )
        object.__setattr__(
            self,
            "history_length",
            _integer(
                self.history_length,
                field_name="history_length",
                minimum=0,
                maximum=MAX_LEDGER_ENTRIES,
            ),
        )
        object.__setattr__(
            self,
            "tree_id",
            _text(self.tree_id or "", field_name="tree_id", required=False),
        )
        _bounded(self, artifact_name="projection snapshot")

    @property
    def snapshot_id(self) -> str:
        return self.content_id

    @property
    def admitted(self) -> tuple[FindingProjectionEntry, ...]:
        return tuple(
            e
            for e in self.entries
            if e.admission is FindingAdmissionState.ADMITTED
        )

    @property
    def stale(self) -> tuple[FindingProjectionEntry, ...]:
        return tuple(
            e for e in self.entries if e.admission is FindingAdmissionState.STALE
        )

    @property
    def conflicts(self) -> tuple[FindingProjectionEntry, ...]:
        return tuple(
            e
            for e in self.entries
            if e.admission is FindingAdmissionState.CONFLICT
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "entries": tuple(entry.to_record() for entry in self.entries),
            "history_length": self.history_length,
            "tree_id": self.tree_id,
            "admitted_count": len(self.admitted),
            "stale_count": len(self.stale),
            "conflict_count": len(self.conflicts),
            "evidence": FINDING_LEDGER_EVIDENCE,
            "evidence_terms": FINDING_LEDGER_G100_EVIDENCE_TERMS,
            "goal_id": GOAL_ID,
            # History is append-only; the projection is the mutable current tree.
            "history_is_append_only": True,
            "projection_is_mutable_current_tree": True,
        }

    def to_record(self) -> dict[str, Any]:
        return {**self.to_dict(), "snapshot_id": self.snapshot_id}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProjectionSnapshot":
        _check_header(payload, cls.SCHEMA)
        _reject_unknown(
            payload,
            _header_fields()
            | {
                "entries",
                "history_length",
                "tree_id",
                "admitted_count",
                "stale_count",
                "conflict_count",
                "snapshot_id",
                "evidence",
                "evidence_terms",
                "goal_id",
                "history_is_append_only",
                "projection_is_mutable_current_tree",
            },
            artifact_name="projection snapshot",
        )
        result = cls(
            entries=tuple(payload.get("entries") or ()),
            history_length=payload.get("history_length", 0),
            tree_id=payload.get("tree_id", ""),
        )
        for name, actual in (
            ("admitted_count", len(result.admitted)),
            ("stale_count", len(result.stale)),
            ("conflict_count", len(result.conflicts)),
        ):
            if name in payload and payload[name] != actual:
                raise ForgedFindingIdentityError(
                    f"projection snapshot {name} does not match entries"
                )
        if "evidence" in payload and payload["evidence"] not in (
            None,
            "",
            FINDING_LEDGER_EVIDENCE,
        ):
            raise ForgedFindingIdentityError(
                f"projection evidence must be {FINDING_LEDGER_EVIDENCE!r}"
            )
        if "goal_id" in payload and payload["goal_id"] not in (
            None,
            "",
            GOAL_ID,
        ):
            raise ForgedFindingIdentityError(
                f"projection goal_id must be {GOAL_ID!r}"
            )
        claimed_terms = payload.get("evidence_terms")
        if claimed_terms is not None and tuple(claimed_terms) != (
            FINDING_LEDGER_G100_EVIDENCE_TERMS
        ):
            raise ForgedFindingIdentityError(
                "projection evidence_terms do not match VFS-G100 terms"
            )
        if "history_is_append_only" in payload and payload[
            "history_is_append_only"
        ] is not True:
            raise ForgedFindingIdentityError(
                "projection must declare append-only history"
            )
        if "projection_is_mutable_current_tree" in payload and payload[
            "projection_is_mutable_current_tree"
        ] is not True:
            raise ForgedFindingIdentityError(
                "projection must declare mutable current-tree separation"
            )
        _check_identity(
            payload,
            result.snapshot_id,
            names=("snapshot_id", "content_id", "cid"),
            artifact_name="projection snapshot",
        )
        return result


def claims_contradict(
    left: ContractFindingRecord, right: ContractFindingRecord
) -> bool:
    """True when two findings share scope but disagree on claim material."""

    if left.semantic_key_id == right.semantic_key_id:
        # Same semantic/root-cause/merge-fate identity is a duplicate path,
        # not a contradiction.
        return False
    same_scope = (
        left.repositories == right.repositories
        and left.symbols == right.symbols
        and left.interfaces == right.interfaces
        and left.expected_contract_cid == right.expected_contract_cid
        and left.observed_contract_cid == right.observed_contract_cid
        and left.tree_id == right.tree_id
    )
    if not same_scope:
        return False
    for name in _CONTRADICTION_FIELDS:
        if getattr(left, name) != getattr(right, name):
            return True
    return False


def build_contract_finding(
    *,
    claim_level: ClaimLevel | str,
    status: FindingStatus | str,
    severity: FindingSeverity | str,
    confidence_millionths: int,
    freshness: EvidenceFreshness | str = EvidenceFreshness.CURRENT,
    repositories: Sequence[str] = (),
    symbols: Sequence[str] = (),
    interfaces: Sequence[str] = (),
    expected_contract_cid: str = "",
    observed_contract_cid: str = "",
    root_cause_family: str = "",
    merge_fate: str = "",
    summary: str = "",
    call_slice: CallSlice | Mapping[str, Any] | None = None,
    evidence: EvidenceReferences | Mapping[str, Any] | None = None,
    assumptions: Sequence[str] = (),
    analyzer_versions: AnalyzerVersions
    | Mapping[str, str]
    | Sequence[tuple[str, str]]
    | None = None,
    remediation_scope: Sequence[str] = (),
    supersedes_cids: Sequence[str] = (),
    superseded_by_cid: str = "",
    rejection_reasons: Sequence[str] = (),
    tree_id: str = "",
    policy_revision: str = "",
    repository_observation_id: str = "",
    verdict: str = "",
    labels: Sequence[str] = (),
    threat_path_cid: str = "",
    impact: str = "",
    partial: bool = False,
    allow_poisoned_severity: bool = False,
) -> ContractFindingRecord:
    """Construct a validated :class:`ContractFindingRecord`."""

    versions: AnalyzerVersions
    if analyzer_versions is None:
        versions = AnalyzerVersions()
    elif isinstance(analyzer_versions, AnalyzerVersions):
        versions = analyzer_versions
    else:
        versions = AnalyzerVersions(versions=analyzer_versions)  # type: ignore[arg-type]

    return ContractFindingRecord(
        claim_level=claim_level,
        status=status,
        severity=severity,
        confidence_millionths=confidence_millionths,
        freshness=freshness,
        repositories=tuple(repositories),
        symbols=tuple(symbols),
        interfaces=tuple(interfaces),
        expected_contract_cid=expected_contract_cid,
        observed_contract_cid=observed_contract_cid,
        root_cause_family=root_cause_family,
        merge_fate=merge_fate,
        summary=summary or "contract finding",
        call_slice=call_slice or CallSlice(),
        evidence=evidence or EvidenceReferences(),
        assumptions=tuple(assumptions),
        analyzer_versions=versions,
        remediation_scope=tuple(remediation_scope),
        supersedes_cids=tuple(supersedes_cids),
        superseded_by_cid=superseded_by_cid,
        rejection_reasons=tuple(rejection_reasons),
        tree_id=tree_id,
        policy_revision=policy_revision,
        repository_observation_id=repository_observation_id,
        verdict=verdict,
        labels=tuple(labels),
        threat_path_cid=threat_path_cid,
        impact=impact,
        partial=partial,
        allow_poisoned_severity=allow_poisoned_severity,
    )


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(payload)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _read_json(path: Path) -> Any:
    with path.open("rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


@dataclass
class ContractFindingLedger:
    """Durable append-only finding ledger with a mutable current projection.

    Layout under ``root``::

        records/<cid>.json     immutable finding bodies
        events.jsonl           append-only event log
        projection.json        current projection snapshot
        meta.json              sequence / capacity metadata
    """

    root: Path
    max_entries: int = MAX_LEDGER_ENTRIES
    max_projection_entries: int = MAX_PROJECTION_ENTRIES
    lock_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._records_dir = self.root / "records"
        self._records_dir.mkdir(parents=True, exist_ok=True)
        self._events_path = self.root / "events.jsonl"
        self._projection_path = self.root / "projection.json"
        self._meta_path = self.root / "meta.json"
        self._lock_path = self.root / ".ledger.lock"
        self._thread_lock = threading.RLock()
        if not self._meta_path.exists():
            _atomic_write_json(
                self._meta_path,
                {
                    "schema": "ipfs_accelerate_py/agent-supervisor/contract-finding/ledger-meta@1",
                    "ledger_version": LEDGER_VERSION,
                    "evidence": FINDING_LEDGER_EVIDENCE,
                    "evidence_terms": list(FINDING_LEDGER_G100_EVIDENCE_TERMS),
                    "goal_id": GOAL_ID,
                    "history_is_append_only": True,
                    "projection_is_mutable_current_tree": True,
                    "sequence": 0,
                    "record_count": 0,
                },
            )
        if not self._projection_path.exists():
            _atomic_write_json(
                self._projection_path,
                ProjectionSnapshot().to_record(),
            )
        if not self._events_path.exists():
            self._events_path.write_bytes(b"")

    # ------------------------------------------------------------------
    # locking
    # ------------------------------------------------------------------

    def _acquire_file_lock(self):
        handle = open(self._lock_path, "a+b")
        deadline = self.lock_timeout_seconds
        import time

        start = time.monotonic()
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return handle
            except BlockingIOError:
                if time.monotonic() - start >= deadline:
                    handle.close()
                    raise LedgerConcurrencyError(
                        "timed out acquiring finding ledger lock"
                    )
                time.sleep(0.01)

    def _release_file_lock(self, handle) -> None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    # ------------------------------------------------------------------
    # internal state
    # ------------------------------------------------------------------

    def _load_meta(self) -> dict[str, Any]:
        return _read_json(self._meta_path)

    def _save_meta(self, meta: Mapping[str, Any]) -> None:
        _atomic_write_json(self._meta_path, dict(meta))

    def _record_path(self, cid: str) -> Path:
        # CIDs are base32-ish; keep path-safe.
        safe = cid.replace("/", "_")
        return self._records_dir / f"{safe}.json"

    def _write_record(self, record: ContractFindingRecord) -> None:
        path = self._record_path(record.finding_cid)
        if path.exists():
            existing = ContractFindingRecord.from_dict(_read_json(path))
            if existing.finding_cid != record.finding_cid:
                raise FindingCollisionError(
                    "record path maps to a different finding CID"
                )
            if existing.to_dict() != record.to_dict():
                raise FindingCollisionError(
                    "distinct finding payloads share one content identity"
                )
            return
        _atomic_write_json(path, record.to_record())

    def _append_event(self, event: LedgerEvent) -> None:
        line = (
            canonical_json_bytes(event.to_record()).decode("utf-8") + "\n"
        )
        with self._events_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    def _load_projection(self) -> ProjectionSnapshot:
        return ProjectionSnapshot.from_dict(_read_json(self._projection_path))

    def _save_projection(self, snapshot: ProjectionSnapshot) -> None:
        if len(snapshot.entries) > self.max_projection_entries:
            raise LedgerCapacityError(
                f"projection exceeds {self.max_projection_entries} entries"
            )
        _atomic_write_json(self._projection_path, snapshot.to_record())

    def _next_sequence(self, meta: dict[str, Any]) -> int:
        sequence = int(meta.get("sequence", 0)) + 1
        meta["sequence"] = sequence
        return sequence

    def _admission_for(self, record: ContractFindingRecord) -> FindingAdmissionState:
        if record.rejection_reasons:
            return FindingAdmissionState.REJECTED
        if record.superseded_by_cid:
            return FindingAdmissionState.SUPERSEDED
        if record.freshness is EvidenceFreshness.STALE:
            return FindingAdmissionState.STALE
        if record.partial or record.status in {
            FindingStatus.AMBIGUOUS,
            FindingStatus.UNSUPPORTED,
            FindingStatus.INCONCLUSIVE,
            FindingStatus.STALE,
        }:
            if record.partial:
                return FindingAdmissionState.PARTIAL
            if record.status is FindingStatus.STALE:
                return FindingAdmissionState.STALE
            return FindingAdmissionState.PARTIAL
        return FindingAdmissionState.ADMITTED

    def _upsert_projection(
        self,
        snapshot: ProjectionSnapshot,
        record: ContractFindingRecord,
        admission: FindingAdmissionState,
        *,
        conflict_cids: Sequence[str] = (),
        rejection_reasons: Sequence[str] = (),
        superseded_by_cid: str = "",
    ) -> ProjectionSnapshot:
        entries = [
            entry
            for entry in snapshot.entries
            if entry.finding_cid != record.finding_cid
        ]
        entries.append(
            FindingProjectionEntry(
                finding_cid=record.finding_cid,
                semantic_key_id=record.semantic_key_id,
                admission=admission,
                tree_id=record.tree_id,
                superseded_by_cid=superseded_by_cid or record.superseded_by_cid,
                rejection_reasons=tuple(
                    rejection_reasons or record.rejection_reasons
                ),
                conflict_cids=tuple(conflict_cids),
            )
        )
        entries.sort(key=lambda e: (e.semantic_key_id, e.finding_cid))
        return ProjectionSnapshot(
            entries=tuple(entries),
            history_length=snapshot.history_length + 1
            if admission
            not in {
                FindingAdmissionState.DUPLICATE,
            }
            else snapshot.history_length,
            tree_id=record.tree_id or snapshot.tree_id,
        )

    def _mark_projection(
        self,
        snapshot: ProjectionSnapshot,
        finding_cid: str,
        admission: FindingAdmissionState,
        *,
        reasons: Sequence[str] = (),
        superseded_by_cid: str = "",
        conflict_cids: Sequence[str] = (),
    ) -> ProjectionSnapshot:
        entries: list[FindingProjectionEntry] = []
        found = False
        for entry in snapshot.entries:
            if entry.finding_cid == finding_cid:
                found = True
                entries.append(
                    FindingProjectionEntry(
                        finding_cid=entry.finding_cid,
                        semantic_key_id=entry.semantic_key_id,
                        admission=admission,
                        tree_id=entry.tree_id,
                        superseded_by_cid=superseded_by_cid
                        or entry.superseded_by_cid,
                        rejection_reasons=tuple(reasons)
                        if reasons
                        else entry.rejection_reasons,
                        conflict_cids=tuple(conflict_cids)
                        if conflict_cids
                        else entry.conflict_cids,
                    )
                )
            else:
                entries.append(entry)
        if not found:
            raise ContractFindingError(
                f"finding {finding_cid} is not in the current projection"
            )
        entries.sort(key=lambda e: (e.semantic_key_id, e.finding_cid))
        return ProjectionSnapshot(
            entries=tuple(entries),
            history_length=snapshot.history_length,
            tree_id=snapshot.tree_id,
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def get(self, finding_cid: str) -> ContractFindingRecord | None:
        path = self._record_path(finding_cid)
        if not path.exists():
            return None
        return ContractFindingRecord.from_dict(_read_json(path))

    def require(self, finding_cid: str) -> ContractFindingRecord:
        record = self.get(finding_cid)
        if record is None:
            raise ContractFindingError(f"unknown finding CID {finding_cid!r}")
        return record

    def history(self) -> tuple[LedgerEvent, ...]:
        if not self._events_path.exists():
            return ()
        events: list[LedgerEvent] = []
        with self._events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                events.append(LedgerEvent.from_dict(json.loads(text)))
        return tuple(events)

    def records(self) -> tuple[ContractFindingRecord, ...]:
        items: list[ContractFindingRecord] = []
        for path in sorted(self._records_dir.glob("*.json")):
            items.append(ContractFindingRecord.from_dict(_read_json(path)))
        items.sort(key=lambda r: r.finding_cid)
        return tuple(items)

    def projection(self) -> ProjectionSnapshot:
        with self._thread_lock:
            handle = self._acquire_file_lock()
            try:
                return self._load_projection()
            finally:
                self._release_file_lock(handle)

    def current_findings(
        self,
        *,
        admitted_only: bool = True,
    ) -> tuple[ContractFindingRecord, ...]:
        snapshot = self.projection()
        result: list[ContractFindingRecord] = []
        for entry in snapshot.entries:
            if admitted_only and entry.admission is not FindingAdmissionState.ADMITTED:
                continue
            if not admitted_only and not entry.current:
                continue
            record = self.get(entry.finding_cid)
            if record is not None:
                result.append(record)
        result.sort(key=lambda r: r.finding_cid)
        return tuple(result)

    def append(
        self,
        record: ContractFindingRecord | Mapping[str, Any],
        *,
        reject_reasons: Sequence[str] = (),
    ) -> AppendReceipt:
        """Append an immutable finding.  History is never rewritten.

        Deduplicates only equal semantic/root-cause/merge-fate findings.
        Exact content CID matches are idempotent.  Distinct content with the
        same CID raises :class:`FindingCollisionError`.
        """

        if not isinstance(record, ContractFindingRecord):
            record = ContractFindingRecord.from_dict(record)

        with self._thread_lock:
            handle = self._acquire_file_lock()
            try:
                return self._append_locked(record, reject_reasons=reject_reasons)
            finally:
                self._release_file_lock(handle)

    def _append_locked(
        self,
        record: ContractFindingRecord,
        *,
        reject_reasons: Sequence[str] = (),
    ) -> AppendReceipt:
        meta = self._load_meta()
        record_count = int(meta.get("record_count", 0))
        if record_count >= self.max_entries and not self._record_path(
            record.finding_cid
        ).exists():
            raise LedgerCapacityError(
                f"ledger exceeds max_entries={self.max_entries}"
            )

        snapshot = self._load_projection()
        reasons = tuple(reject_reasons) or record.rejection_reasons

        # Exact content identity already stored → idempotent duplicate.
        existing_same = self.get(record.finding_cid)
        if existing_same is not None:
            if existing_same.to_dict() != record.to_dict():
                raise FindingCollisionError(
                    "distinct finding payloads share one content identity"
                )
            sequence = self._next_sequence(meta)
            event = LedgerEvent(
                kind=LedgerEventKind.DUPLICATE,
                finding_cid=record.finding_cid,
                sequence=sequence,
                semantic_key_id=record.semantic_key_id,
                reasons=("exact_content_identity",),
                tree_id=record.tree_id,
                payload_digest=record.finding_cid,
            )
            self._append_event(event)
            self._save_meta(meta)
            return AppendReceipt(
                outcome=AppendOutcome.DUPLICATE,
                finding_cid=record.finding_cid,
                sequence=sequence,
                semantic_key_id=record.semantic_key_id,
                admission=FindingAdmissionState.DUPLICATE,
                prior_finding_cid=record.finding_cid,
                reasons=("exact_content_identity",),
                event_id=event.event_id,
            )

        # Semantic dedup: equal contract/root-cause/merge-fate only.
        semantic_match: ContractFindingRecord | None = None
        for entry in snapshot.entries:
            if entry.semantic_key_id != record.semantic_key_id:
                continue
            if entry.admission in {
                FindingAdmissionState.SUPERSEDED,
                FindingAdmissionState.REJECTED,
                FindingAdmissionState.STALE,
            }:
                continue
            candidate = self.get(entry.finding_cid)
            if candidate is None:
                continue
            if candidate.semantic_key_id == record.semantic_key_id:
                semantic_match = candidate
                break

        if semantic_match is not None:
            sequence = self._next_sequence(meta)
            # Store the body for auditability of the equal observation, but
            # do not expand the admitted projection.
            self._write_record(record)
            meta["record_count"] = int(meta.get("record_count", 0)) + 1
            event = LedgerEvent(
                kind=LedgerEventKind.DUPLICATE,
                finding_cid=record.finding_cid,
                sequence=sequence,
                semantic_key_id=record.semantic_key_id,
                related_cids=(semantic_match.finding_cid,),
                reasons=("equal_semantic_root_cause_merge_fate",),
                tree_id=record.tree_id,
                payload_digest=record.finding_cid,
            )
            self._append_event(event)
            # Projection keeps the prior admitted entry; note the duplicate.
            snapshot = self._upsert_projection(
                snapshot,
                record,
                FindingAdmissionState.DUPLICATE,
            )
            # Fix history_length accounting for stored-but-duplicate.
            snapshot = ProjectionSnapshot(
                entries=snapshot.entries,
                history_length=int(meta.get("sequence", sequence)),
                tree_id=snapshot.tree_id,
            )
            self._save_projection(snapshot)
            self._save_meta(meta)
            return AppendReceipt(
                outcome=AppendOutcome.DUPLICATE,
                finding_cid=record.finding_cid,
                sequence=sequence,
                semantic_key_id=record.semantic_key_id,
                admission=FindingAdmissionState.DUPLICATE,
                prior_finding_cid=semantic_match.finding_cid,
                reasons=("equal_semantic_root_cause_merge_fate",),
                event_id=event.event_id,
            )

        # Explicit rejection path.
        if reasons:
            record = record.with_updates(rejection_reasons=reasons)
            sequence = self._next_sequence(meta)
            self._write_record(record)
            meta["record_count"] = int(meta.get("record_count", 0)) + 1
            event = LedgerEvent(
                kind=LedgerEventKind.REJECT,
                finding_cid=record.finding_cid,
                sequence=sequence,
                semantic_key_id=record.semantic_key_id,
                reasons=reasons,
                tree_id=record.tree_id,
                payload_digest=record.finding_cid,
            )
            self._append_event(event)
            snapshot = self._upsert_projection(
                snapshot,
                record,
                FindingAdmissionState.REJECTED,
                rejection_reasons=reasons,
            )
            snapshot = ProjectionSnapshot(
                entries=snapshot.entries,
                history_length=int(meta.get("sequence", sequence)),
                tree_id=snapshot.tree_id,
            )
            self._save_projection(snapshot)
            self._save_meta(meta)
            return AppendReceipt(
                outcome=AppendOutcome.REJECTED,
                finding_cid=record.finding_cid,
                sequence=sequence,
                semantic_key_id=record.semantic_key_id,
                admission=FindingAdmissionState.REJECTED,
                reasons=reasons,
                event_id=event.event_id,
            )

        # Detect contradictions against currently admitted same-scope findings.
        conflict_cids: list[str] = []
        for entry in snapshot.entries:
            if entry.admission is not FindingAdmissionState.ADMITTED:
                continue
            other = self.get(entry.finding_cid)
            if other is None:
                continue
            if claims_contradict(record, other):
                conflict_cids.append(other.finding_cid)

        admission = self._admission_for(record)
        if conflict_cids:
            admission = FindingAdmissionState.CONFLICT

        sequence = self._next_sequence(meta)
        self._write_record(record)
        meta["record_count"] = int(meta.get("record_count", 0)) + 1
        kind = (
            LedgerEventKind.CONFLICT
            if conflict_cids
            else (
                LedgerEventKind.MARK_PARTIAL
                if admission is FindingAdmissionState.PARTIAL
                else LedgerEventKind.APPEND
            )
        )
        event = LedgerEvent(
            kind=kind,
            finding_cid=record.finding_cid,
            sequence=sequence,
            semantic_key_id=record.semantic_key_id,
            related_cids=tuple(conflict_cids),
            reasons=(
                ("contradictory_claims",) if conflict_cids else ()
            )
            + (
                tuple(f"missing:{field}" for field in record.partial_missing_fields)
                if record.partial
                else ()
            ),
            tree_id=record.tree_id,
            payload_digest=record.finding_cid,
        )
        self._append_event(event)

        # Also mark peers as conflict when contradiction is detected.
        if conflict_cids:
            for peer_cid in conflict_cids:
                snapshot = self._mark_projection(
                    snapshot,
                    peer_cid,
                    FindingAdmissionState.CONFLICT,
                    conflict_cids=(record.finding_cid,),
                    reasons=("contradictory_claims",),
                )

        snapshot = self._upsert_projection(
            snapshot,
            record,
            admission,
            conflict_cids=conflict_cids,
        )
        snapshot = ProjectionSnapshot(
            entries=snapshot.entries,
            history_length=int(meta.get("sequence", sequence)),
            tree_id=record.tree_id or snapshot.tree_id,
        )
        self._save_projection(snapshot)
        self._save_meta(meta)
        return AppendReceipt(
            outcome=AppendOutcome.STORED,
            finding_cid=record.finding_cid,
            sequence=sequence,
            semantic_key_id=record.semantic_key_id,
            admission=admission,
            reasons=event.reasons,
            event_id=event.event_id,
        )

    def reject(
        self,
        finding_cid: str,
        reasons: Sequence[str],
    ) -> AppendReceipt:
        """Append a rejection event and drop the finding from admitted view."""

        if not reasons:
            raise ContractFindingError("rejection requires at least one reason")
        with self._thread_lock:
            handle = self._acquire_file_lock()
            try:
                record = self.require(finding_cid)
                meta = self._load_meta()
                sequence = self._next_sequence(meta)
                rejected = record.with_updates(
                    rejection_reasons=tuple(reasons)
                )
                # New CID for the rejected view of the same observation so
                # history preserves the original admitted body.
                # Prefer updating projection only: keep original record.
                event = LedgerEvent(
                    kind=LedgerEventKind.REJECT,
                    finding_cid=finding_cid,
                    sequence=sequence,
                    semantic_key_id=record.semantic_key_id,
                    reasons=tuple(reasons),
                    tree_id=record.tree_id,
                    payload_digest=finding_cid,
                )
                self._append_event(event)
                snapshot = self._load_projection()
                snapshot = self._mark_projection(
                    snapshot,
                    finding_cid,
                    FindingAdmissionState.REJECTED,
                    reasons=reasons,
                )
                snapshot = ProjectionSnapshot(
                    entries=snapshot.entries,
                    history_length=int(meta.get("sequence", sequence)),
                    tree_id=snapshot.tree_id,
                )
                self._save_projection(snapshot)
                self._save_meta(meta)
                return AppendReceipt(
                    outcome=AppendOutcome.REJECTED,
                    finding_cid=finding_cid,
                    sequence=sequence,
                    semantic_key_id=record.semantic_key_id,
                    admission=FindingAdmissionState.REJECTED,
                    reasons=tuple(reasons),
                    event_id=event.event_id,
                )
            finally:
                self._release_file_lock(handle)

    def supersede(
        self,
        prior_cid: str,
        replacement: ContractFindingRecord | Mapping[str, Any],
    ) -> AppendReceipt:
        """Append a replacement finding and mark the prior as superseded."""

        if not isinstance(replacement, ContractFindingRecord):
            replacement = ContractFindingRecord.from_dict(replacement)

        with self._thread_lock:
            handle = self._acquire_file_lock()
            try:
                prior = self.require(prior_cid)
                # Bind supersession edges on the replacement body.
                supersedes = tuple(
                    sorted(set(replacement.supersedes_cids) | {prior_cid})
                )
                replacement = replacement.with_updates(
                    supersedes_cids=supersedes
                )
                # Append replacement first (without semantic-dedup against prior
                # because prior will be marked superseded).
                meta = self._load_meta()
                snapshot = self._load_projection()

                # If exact CID already present, treat as idempotent supersession.
                existing = self.get(replacement.finding_cid)
                if existing is not None and existing.to_dict() != replacement.to_dict():
                    raise FindingCollisionError(
                        "distinct finding payloads share one content identity"
                    )

                sequence = self._next_sequence(meta)
                if existing is None:
                    if int(meta.get("record_count", 0)) >= self.max_entries:
                        raise LedgerCapacityError(
                            f"ledger exceeds max_entries={self.max_entries}"
                        )
                    self._write_record(replacement)
                    meta["record_count"] = int(meta.get("record_count", 0)) + 1

                event = LedgerEvent(
                    kind=LedgerEventKind.SUPERSEDE,
                    finding_cid=replacement.finding_cid,
                    sequence=sequence,
                    semantic_key_id=replacement.semantic_key_id,
                    related_cids=(prior_cid,),
                    reasons=("supersession",),
                    tree_id=replacement.tree_id,
                    payload_digest=replacement.finding_cid,
                )
                self._append_event(event)

                snapshot = self._mark_projection(
                    snapshot,
                    prior_cid,
                    FindingAdmissionState.SUPERSEDED,
                    superseded_by_cid=replacement.finding_cid,
                    reasons=("superseded",),
                )
                admission = self._admission_for(replacement)
                snapshot = self._upsert_projection(
                    snapshot,
                    replacement,
                    admission,
                )
                snapshot = ProjectionSnapshot(
                    entries=snapshot.entries,
                    history_length=int(meta.get("sequence", sequence)),
                    tree_id=replacement.tree_id or snapshot.tree_id,
                )
                self._save_projection(snapshot)
                self._save_meta(meta)
                # Touch prior so static analyzers know we used it.
                _ = prior.finding_cid
                return AppendReceipt(
                    outcome=AppendOutcome.SUPERSEDED_PRIOR,
                    finding_cid=replacement.finding_cid,
                    sequence=sequence,
                    semantic_key_id=replacement.semantic_key_id,
                    admission=admission,
                    prior_finding_cid=prior_cid,
                    reasons=("supersession",),
                    event_id=event.event_id,
                )
            finally:
                self._release_file_lock(handle)

    def invalidate_stale(
        self,
        *,
        tree_id: str | None = None,
        repository_observation_id: str | None = None,
        finding_cids: Sequence[str] | None = None,
        reasons: Sequence[str] = ("stale_evidence",),
    ) -> tuple[str, ...]:
        """Invalidate current projection entries that bind stale evidence.

        History is preserved.  Only the current projection changes.
        """

        with self._thread_lock:
            handle = self._acquire_file_lock()
            try:
                meta = self._load_meta()
                snapshot = self._load_projection()
                invalidated: list[str] = []
                targets = set(finding_cids or ())
                for entry in list(snapshot.entries):
                    if entry.admission in {
                        FindingAdmissionState.SUPERSEDED,
                        FindingAdmissionState.REJECTED,
                        FindingAdmissionState.STALE,
                        FindingAdmissionState.DUPLICATE,
                    }:
                        continue
                    record = self.get(entry.finding_cid)
                    if record is None:
                        continue
                    match = False
                    if targets and record.finding_cid in targets:
                        match = True
                    if tree_id is not None and record.tree_id == tree_id:
                        match = True
                    if (
                        repository_observation_id is not None
                        and record.repository_observation_id
                        == repository_observation_id
                    ):
                        match = True
                    if finding_cids is None and tree_id is None and repository_observation_id is None:
                        # Explicit no-op filter: invalidate records already
                        # marked freshness=stale in the body.
                        if record.freshness is EvidenceFreshness.STALE:
                            match = True
                    if not match:
                        continue
                    sequence = self._next_sequence(meta)
                    event = LedgerEvent(
                        kind=LedgerEventKind.INVALIDATE_STALE,
                        finding_cid=record.finding_cid,
                        sequence=sequence,
                        semantic_key_id=record.semantic_key_id,
                        reasons=tuple(reasons),
                        tree_id=record.tree_id,
                        payload_digest=record.finding_cid,
                    )
                    self._append_event(event)
                    snapshot = self._mark_projection(
                        snapshot,
                        record.finding_cid,
                        FindingAdmissionState.STALE,
                        reasons=reasons,
                    )
                    invalidated.append(record.finding_cid)
                snapshot = ProjectionSnapshot(
                    entries=snapshot.entries,
                    history_length=int(meta.get("sequence", 0)),
                    tree_id=snapshot.tree_id,
                )
                self._save_projection(snapshot)
                self._save_meta(meta)
                return tuple(sorted(invalidated))
            finally:
                self._release_file_lock(handle)

    def replay(self) -> ProjectionSnapshot:
        """Rebuild the current projection purely from the append-only event log.

        Immutable records are the authority for finding bodies.  Events drive
        admission state transitions.  The rebuilt projection replaces the
        on-disk projection atomically.
        """

        with self._thread_lock:
            handle = self._acquire_file_lock()
            try:
                events = self.history()
                # state by finding cid
                admission: dict[str, FindingAdmissionState] = {}
                semantic: dict[str, str] = {}
                tree_ids: dict[str, str] = {}
                supersedes: dict[str, str] = {}
                rejections: dict[str, tuple[str, ...]] = {}
                conflicts: dict[str, set[str]] = {}
                order: list[str] = []

                for event in events:
                    cid = event.finding_cid
                    if cid not in admission:
                        order.append(cid)
                    semantic[cid] = event.semantic_key_id or semantic.get(cid, "")
                    tree_ids[cid] = event.tree_id or tree_ids.get(cid, "")
                    if event.kind is LedgerEventKind.APPEND:
                        admission[cid] = FindingAdmissionState.ADMITTED
                    elif event.kind is LedgerEventKind.MARK_PARTIAL:
                        admission[cid] = FindingAdmissionState.PARTIAL
                    elif event.kind is LedgerEventKind.DUPLICATE:
                        admission[cid] = FindingAdmissionState.DUPLICATE
                    elif event.kind is LedgerEventKind.REJECT:
                        admission[cid] = FindingAdmissionState.REJECTED
                        rejections[cid] = event.reasons
                    elif event.kind is LedgerEventKind.SUPERSEDE:
                        admission[cid] = FindingAdmissionState.ADMITTED
                        for prior in event.related_cids:
                            admission[prior] = FindingAdmissionState.SUPERSEDED
                            supersedes[prior] = cid
                    elif event.kind is LedgerEventKind.INVALIDATE_STALE:
                        admission[cid] = FindingAdmissionState.STALE
                        rejections[cid] = event.reasons
                    elif event.kind is LedgerEventKind.CONFLICT:
                        admission[cid] = FindingAdmissionState.CONFLICT
                        conflicts.setdefault(cid, set()).update(event.related_cids)
                        for peer in event.related_cids:
                            admission[peer] = FindingAdmissionState.CONFLICT
                            conflicts.setdefault(peer, set()).add(cid)

                entries: list[FindingProjectionEntry] = []
                for cid in order:
                    record = self.get(cid)
                    key = semantic.get(cid, "")
                    if record is not None:
                        key = key or record.semantic_key_id
                    entries.append(
                        FindingProjectionEntry(
                            finding_cid=cid,
                            semantic_key_id=key or "unknown",
                            admission=admission.get(
                                cid, FindingAdmissionState.ADMITTED
                            ),
                            tree_id=tree_ids.get(cid, "")
                            or (record.tree_id if record else ""),
                            superseded_by_cid=supersedes.get(cid, ""),
                            rejection_reasons=rejections.get(cid, ()),
                            conflict_cids=tuple(sorted(conflicts.get(cid, ()))),
                        )
                    )
                entries.sort(key=lambda e: (e.semantic_key_id, e.finding_cid))
                meta = self._load_meta()
                snapshot = ProjectionSnapshot(
                    entries=tuple(entries),
                    history_length=int(meta.get("sequence", len(events))),
                    tree_id=next(
                        (e.tree_id for e in entries if e.tree_id), ""
                    ),
                )
                self._save_projection(snapshot)
                return snapshot
            finally:
                self._release_file_lock(handle)

    def stats(self) -> dict[str, Any]:
        meta = self._load_meta()
        snapshot = self.projection()
        return {
            "ledger_version": LEDGER_VERSION,
            "evidence": FINDING_LEDGER_EVIDENCE,
            "evidence_terms": list(FINDING_LEDGER_G100_EVIDENCE_TERMS),
            "goal_id": GOAL_ID,
            "history_is_append_only": True,
            "projection_is_mutable_current_tree": True,
            "sequence": int(meta.get("sequence", 0)),
            "record_count": int(meta.get("record_count", 0)),
            "projection_entries": len(snapshot.entries),
            "admitted": len(snapshot.admitted),
            "stale": len(snapshot.stale),
            "conflicts": len(snapshot.conflicts),
            "history_events": len(self.history()),
        }


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "APPEND_RECEIPT_SCHEMA",
    "CONTRACT_FINDING_RECORD_SCHEMA",
    "CONTRACT_FINDINGS_VERSION",
    "FINDING_LEDGER_EVIDENCE",
    "FINDING_LEDGER_G100_EVIDENCE_TERMS",
    "GOAL_ID",
    "LEDGER_VERSION",
    "MAX_CALL_SLICE_STEPS",
    "MAX_COLLECTION_ITEMS",
    "MAX_LABELS",
    "MAX_LEDGER_ENTRIES",
    "MAX_RECORD_BYTES",
    "MAX_TEXT_BYTES",
    "VULNERABILITY_EVIDENCE_POLICY",
    "VULNERABILITY_LABEL",
    "AnalyzerVersions",
    "AppendOutcome",
    "AppendReceipt",
    "CallSlice",
    "CallSliceStep",
    "ContractFindingBoundsError",
    "ContractFindingError",
    "ContractFindingLedger",
    "ContractFindingRecord",
    "EvidenceReferences",
    "FindingAdmissionState",
    "FindingCollisionError",
    "FindingProjectionEntry",
    "ForgedFindingIdentityError",
    "LedgerCapacityError",
    "LedgerConcurrencyError",
    "LedgerEvent",
    "LedgerEventKind",
    "PartialFindingError",
    "PoisonedSeverityError",
    "ProjectionSnapshot",
    "SemanticDedupKey",
    "StaleFindingError",
    "VulnerabilityEvidencePolicyError",
    "build_contract_finding",
    "claims_contradict",
    "covered_evidence_terms",
    "finding_content_cid",
    "finding_ledger_evidence_terms",
    "is_partial_finding",
    "is_vulnerability_labeled",
    "validate_severity_binding",
    "validate_vulnerability_evidence_policy",
    "vulnerability_evidence_requirements_met",
    # Re-export assurance enums used at the ledger boundary.
    "ClaimLevel",
    "EvidenceFreshness",
    "FindingSeverity",
    "FindingStatus",
]
