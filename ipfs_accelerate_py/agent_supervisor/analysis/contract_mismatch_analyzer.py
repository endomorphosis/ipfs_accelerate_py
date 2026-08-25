"""Deterministic refinement of contract claims into repairable findings.

Interface: ``ContractMismatchAnalyzer@1`` / ``ContractFinding@1``.

This module is the narrow lifecycle boundary between parity/proof evidence and
the later edit-packet refinery.  It deliberately does not prove claims and it
does not infer repository ownership from names or prose.  A finding is emitted
only for an explicitly negative or incomplete claim state and carries:

* a dedupe identity over the exact snapshot, contract, claim family, affected
  symbol set, and compact counterexample;
* content-addressed evidence revisions which can change without minting a
  duplicate finding;
* exact claim/obligation/receipt/CAS handles and deterministic reproduction
  commands; and
* path-by-path ownership selected from reviewed repository-root prefixes.

Cache misses and unknown/open claims are intentionally ignored.  They mean
that evidence is absent, not that a contract was refuted.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from ..proof.formal_counterexamples import FormalCounterexample
from ..proof.code_proof_query import (
    ClaimQueryHit,
    CodeProofQuery,
    CodeProofQueryResult,
)
from ..proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)
from ..proof.mcp_contract_prover import McpContractProofResult
from .mcp_contract_analysis import (
    ContractCounterexample,
    McpContractAnalysis,
    ContractParityClaim,
)


CONTRACT_MISMATCH_ANALYZER_INTERFACE: Final = "ContractMismatchAnalyzer@1"
CONTRACT_FINDING_INTERFACE: Final = "ContractFinding@1"
CONTRACT_MISMATCH_ANALYZER_VERSION: Final = "1"
CONTRACT_FINDING_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/contract-finding@1"
)
CONTRACT_FINDING_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/contract-finding-evidence@1"
)
CONTRACT_REPRODUCTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/contract-finding-reproduction@1"
)
CONTRACT_MISMATCH_ANALYSIS_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/contract-mismatch-analysis@1"
)

DEFAULT_MAX_IMPACT_SYMBOLS: Final = 128
HARD_MAX_IMPACT_SYMBOLS: Final = 1_024
DEFAULT_MAX_IMPACT_DEPTH: Final = 6
HARD_MAX_IMPACT_DEPTH: Final = 32
MAX_FINDING_EVIDENCE_REVISIONS: Final = 64
MAX_REPRODUCTION_COMMANDS: Final = 32
MAX_COUNTEREXAMPLE_BYTES: Final = 64 * 1024


class ContractMismatchError(ValueError):
    """Mismatch evidence is malformed, ambiguous, or exceeds a hard bound."""


class MismatchState(str, Enum):
    """Claim states which are materialized as findings."""

    REFUTED = "refuted"
    STALE = "stale"
    CONTRADICTORY = "contradictory"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"
    NOT_MEASURED = "not_measured"


class FindingLifecycle(str, Enum):
    """Lifecycle of one snapshot-bound dedupe identity."""

    ACTIVE = "active"
    STALE = "stale"
    REOPENED = "reopened"
    RESOLVED = "resolved"


class SourceOwner(str, Enum):
    """Reviewed source repositories to which a repair may be routed."""

    ACCELERATOR = "accelerator"
    KIT = "kit"
    DATASETS = "datasets"
    SWISSKNIFE = "swissknife"
    MCP_PLUS_PLUS = "mcp_plus_plus"
    UNRESOLVED = "unresolved"

    # Descriptive aliases retained for callers which use package names.
    IPFS_ACCELERATE = "accelerator"
    IPFS_ACCELERATE_PY = "accelerator"
    IPFS_KIT = "kit"
    IPFS_KIT_PY = "kit"
    IPFS_DATASETS = "datasets"
    IPFS_DATASETS_PY = "datasets"
    MCPPP = "mcp_plus_plus"
    UNKNOWN = "unresolved"


# Prefixes are intentionally explicit.  A substring, package-like symbol, or
# documentation reference is never enough to assign write ownership.
_OWNER_PREFIXES: Final[tuple[tuple[str, SourceOwner], ...]] = (
    ("external/ipfs_accelerate", SourceOwner.ACCELERATOR),
    ("ipfs_accelerate_py", SourceOwner.ACCELERATOR),
    ("external/ipfs_kit", SourceOwner.KIT),
    ("ipfs_kit_py", SourceOwner.KIT),
    ("external/ipfs_datasets", SourceOwner.DATASETS),
    ("ipfs_datasets_py", SourceOwner.DATASETS),
    ("swissknife", SourceOwner.SWISSKNIFE),
    ("Mcp-Plus-Plus", SourceOwner.MCP_PLUS_PLUS),
)

_IGNORED_STATES: Final[frozenset[str]] = frozenset(
    {
        "",
        "current",
        "hit",
        "miss",
        "cache_miss",
        "not_cached",
        "unknown",
        "open",
        "pending",
        "satisfied",
        "proved",
        "pass",
        "passed",
        "success",
    }
)

_STATE_ALIASES: Final[Mapping[str, MismatchState]] = MappingProxyType(
    {
        "refuted": MismatchState.REFUTED,
        "disproved": MismatchState.REFUTED,
        "failed": MismatchState.REFUTED,
        "stale": MismatchState.STALE,
        "invalidated": MismatchState.STALE,
        "contradicted": MismatchState.CONTRADICTORY,
        "contradictory": MismatchState.CONTRADICTORY,
        "contradiction": MismatchState.CONTRADICTORY,
        "ambiguous": MismatchState.AMBIGUOUS,
        "inconclusive": MismatchState.AMBIGUOUS,
        "partial": MismatchState.AMBIGUOUS,
        "timed_out": MismatchState.AMBIGUOUS,
        "timeout": MismatchState.AMBIGUOUS,
        "unsupported": MismatchState.UNSUPPORTED,
        "not_measured": MismatchState.NOT_MEASURED,
        "unmeasured": MismatchState.NOT_MEASURED,
        "skipped": MismatchState.NOT_MEASURED,
    }
)


def _text(
    value: Any,
    name: str,
    *,
    required: bool = True,
    maximum: int = 16_384,
) -> str:
    if not isinstance(value, str):
        raise ContractMismatchError(f"{name} must be a string")
    if value != value.strip() or "\x00" in value:
        raise ContractMismatchError(
            f"{name} must not contain surrounding whitespace or NUL"
        )
    if required and not value:
        raise ContractMismatchError(f"{name} is required")
    if len(value.encode("utf-8")) > maximum:
        raise ContractMismatchError(f"{name} is oversized")
    return value


def _ids(
    values: Any,
    name: str,
    *,
    required: bool = False,
    maximum: int = 1_024,
) -> tuple[str, ...]:
    if values is None:
        source: Iterable[Any] = ()
    elif isinstance(values, str):
        source = (values,)
    elif isinstance(values, Sequence) and not isinstance(
        values, (bytes, bytearray, memoryview)
    ):
        source = values
    else:
        raise ContractMismatchError(f"{name} must be a sequence of strings")
    result: set[str] = set()
    for item in source:
        result.add(_text(item, name, maximum=4_096))
        if len(result) > maximum:
            raise ContractMismatchError(f"{name} exceeds its item bound")
    if required and not result:
        raise ContractMismatchError(f"{name} must not be empty")
    return tuple(sorted(result))


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    raw = str(getattr(value, "value", value))
    try:
        return enum_type(raw)
    except ValueError as exc:
        raise ContractMismatchError(f"unknown {name}: {value!r}") from exc


def _plain(value: Any, *, depth: int = 0) -> Any:
    """Detach evidence into bounded deterministic JSON values."""

    if depth > 16:
        raise ContractMismatchError("evidence exceeds its nesting bound")
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        raise ContractMismatchError("floating evidence is not canonical")
    if isinstance(value, Mapping):
        if len(value) > 2_048 or not all(isinstance(key, str) for key in value):
            raise ContractMismatchError(
                "evidence objects require at most 2048 string keys"
            )
        return {
            key: _plain(value[key], depth=depth + 1)
            for key in sorted(value)
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        if len(value) > 4_096:
            raise ContractMismatchError("evidence sequence is oversized")
        return [_plain(item, depth=depth + 1) for item in value]
    to_capsule = getattr(value, "to_capsule_dict", None)
    if callable(to_capsule):
        return _plain(to_capsule(), depth=depth + 1)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _plain(to_dict(), depth=depth + 1)
    raise ContractMismatchError(
        f"unsupported evidence value: {type(value).__name__}"
    )


def _repo_path(value: Any, name: str = "path") -> str:
    path = _text(value, name).replace("\\", "/")
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts or path.startswith("./"):
        raise ContractMismatchError(f"{name} must be repository-relative")
    normalized = candidate.as_posix()
    if normalized in ("", "."):
        raise ContractMismatchError(f"{name} is required")
    return normalized


def _paths(values: Any, name: str = "paths") -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)
    if not isinstance(values, Sequence) or isinstance(
        values, (bytes, bytearray, memoryview)
    ):
        raise ContractMismatchError(f"{name} must be a sequence")
    result = {_repo_path(value, name) for value in values}
    if len(result) > HARD_MAX_IMPACT_SYMBOLS:
        raise ContractMismatchError(f"{name} exceeds its item bound")
    return tuple(sorted(result))


def _state(value: Any) -> MismatchState | None:
    raw = str(getattr(value, "value", value) or "").strip().lower()
    if raw in _IGNORED_STATES:
        return None
    result = _STATE_ALIASES.get(raw)
    if result is None:
        raise ContractMismatchError(f"unknown claim state: {value!r}")
    return result


def route_source_owner(path: str) -> SourceOwner:
    """Return the owner selected by an exact reviewed repository prefix."""

    normalized = _repo_path(path)
    for prefix, owner in _OWNER_PREFIXES:
        if normalized == prefix or normalized.startswith(prefix + "/"):
            return owner
    return SourceOwner.UNRESOLVED


@dataclass(frozen=True, slots=True)
class SourceOwnership:
    """One path and the reviewed prefix which did (or did not) own it."""

    path: str
    owner: SourceOwner
    matched_prefix: str = ""
    reason_code: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _repo_path(self.path))
        object.__setattr__(
            self, "owner", _enum(self.owner, SourceOwner, "source owner")
        )
        object.__setattr__(
            self,
            "matched_prefix",
            _text(self.matched_prefix, "matched_prefix", required=False),
        )
        object.__setattr__(
            self,
            "reason_code",
            _text(self.reason_code, "reason_code", required=False),
        )
        expected_owner = route_source_owner(self.path)
        expected_prefix = next(
            (
                prefix
                for prefix, owner in _OWNER_PREFIXES
                if owner is expected_owner
                and (
                    self.path == prefix
                    or self.path.startswith(prefix + "/")
                )
            ),
            "",
        )
        if self.owner is not expected_owner:
            raise ContractMismatchError(
                f"source ownership claim disagrees for {self.path}"
            )
        if self.matched_prefix and self.matched_prefix != expected_prefix:
            raise ContractMismatchError(
                f"source ownership prefix disagrees for {self.path}"
            )
        if self.owner is SourceOwner.UNRESOLVED:
            object.__setattr__(self, "matched_prefix", "")
            object.__setattr__(
                self, "reason_code", self.reason_code or "owner_prefix_unrecognized"
            )
        else:
            object.__setattr__(self, "matched_prefix", expected_prefix)
            object.__setattr__(
                self, "reason_code", self.reason_code or "reviewed_prefix_match"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "owner": self.owner.value,
            "matched_prefix": self.matched_prefix,
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceOwnership":
        return cls(
            path=str(value.get("path") or ""),
            owner=value.get("owner", SourceOwner.UNRESOLVED),
            matched_prefix=str(value.get("matched_prefix") or ""),
            reason_code=str(value.get("reason_code") or ""),
        )


def resolve_source_ownership(paths: Sequence[str]) -> tuple[SourceOwnership, ...]:
    """Resolve every path independently; unknown or mixed sets stay explicit."""

    return tuple(
        SourceOwnership(path=path, owner=route_source_owner(path))
        for path in _paths(paths)
    )


@dataclass(frozen=True, slots=True)
class ReproductionHandles:
    """Exact handles sufficient to expand and rerun one mismatch."""

    snapshot_id: str
    contract_id: str
    claim_id: str
    counterexample_id: str
    obligation_ids: tuple[str, ...] = ()
    receipt_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    cas_handles: tuple[str, ...] = ()
    commands: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "snapshot_id",
            "contract_id",
            "claim_id",
            "counterexample_id",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "obligation_ids",
            "receipt_ids",
            "evidence_ids",
            "cas_handles",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        object.__setattr__(
            self,
            "commands",
            _ids(
                self.commands,
                "commands",
                maximum=MAX_REPRODUCTION_COMMANDS,
            ),
        )
        if not self.obligation_ids and not self.commands:
            raise ContractMismatchError(
                "reproduction requires an obligation id or exact command"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_REPRODUCTION_SCHEMA,
            "snapshot_id": self.snapshot_id,
            "contract_id": self.contract_id,
            "claim_id": self.claim_id,
            "counterexample_id": self.counterexample_id,
            "obligation_ids": list(self.obligation_ids),
            "receipt_ids": list(self.receipt_ids),
            "evidence_ids": list(self.evidence_ids),
            "cas_handles": list(self.cas_handles),
            "commands": list(self.commands),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReproductionHandles":
        if value.get("schema") not in (None, CONTRACT_REPRODUCTION_SCHEMA):
            raise ContractMismatchError("unsupported reproduction schema")
        return cls(
            snapshot_id=str(value.get("snapshot_id") or ""),
            contract_id=str(value.get("contract_id") or ""),
            claim_id=str(value.get("claim_id") or ""),
            counterexample_id=str(value.get("counterexample_id") or ""),
            obligation_ids=tuple(value.get("obligation_ids") or ()),
            receipt_ids=tuple(value.get("receipt_ids") or ()),
            evidence_ids=tuple(value.get("evidence_ids") or ()),
            cas_handles=tuple(value.get("cas_handles") or ()),
            commands=tuple(value.get("commands") or ()),
        )


@dataclass(frozen=True, slots=True)
class FindingEvidence:
    """One immutable observation of a stable finding identity."""

    claim_id: str
    state: MismatchState
    reason_codes: tuple[str, ...]
    premise_ids: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    receipt_ids: tuple[str, ...] = ()
    proof_result_ids: tuple[str, ...] = ()
    evidence_revision_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _text(self.claim_id, "claim_id"))
        object.__setattr__(
            self, "state", _enum(self.state, MismatchState, "mismatch state")
        )
        for name in (
            "reason_codes",
            "premise_ids",
            "obligation_ids",
            "evidence_ids",
            "receipt_ids",
            "proof_result_ids",
        ):
            object.__setattr__(self, name, _ids(getattr(self, name), name))
        if not self.reason_codes:
            raise ContractMismatchError("finding evidence requires a reason code")
        expected = content_identity(self._identity_payload())
        if self.evidence_revision_id and self.evidence_revision_id != expected:
            raise ContractMismatchError("finding evidence identity mismatch")
        object.__setattr__(self, "evidence_revision_id", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_FINDING_EVIDENCE_SCHEMA,
            "claim_id": self.claim_id,
            "state": self.state.value,
            "reason_codes": list(self.reason_codes),
            "premise_ids": list(self.premise_ids),
            "obligation_ids": list(self.obligation_ids),
            "evidence_ids": list(self.evidence_ids),
            "receipt_ids": list(self.receipt_ids),
            "proof_result_ids": list(self.proof_result_ids),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_revision_id": self.evidence_revision_id,
            **self._identity_payload(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FindingEvidence":
        if value.get("schema") not in (None, CONTRACT_FINDING_EVIDENCE_SCHEMA):
            raise ContractMismatchError("unsupported finding-evidence schema")
        return cls(
            claim_id=str(value.get("claim_id") or ""),
            state=value.get("state", ""),
            reason_codes=tuple(value.get("reason_codes") or ()),
            premise_ids=tuple(value.get("premise_ids") or ()),
            obligation_ids=tuple(value.get("obligation_ids") or ()),
            evidence_ids=tuple(value.get("evidence_ids") or ()),
            receipt_ids=tuple(value.get("receipt_ids") or ()),
            proof_result_ids=tuple(value.get("proof_result_ids") or ()),
            evidence_revision_id=str(value.get("evidence_revision_id") or ""),
        )


@dataclass(frozen=True, slots=True)
class ImpactClosure:
    """Bounded deterministic affected-symbol closure."""

    symbols: tuple[str, ...]
    truncated: bool
    max_depth: int
    max_symbols: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbols": list(self.symbols),
            "truncated": self.truncated,
            "max_depth": self.max_depth,
            "max_symbols": self.max_symbols,
        }


def bounded_impact_closure(
    seed_symbols: Sequence[str],
    impact_edges: Mapping[str, Sequence[str]] | None = None,
    *,
    max_depth: int = DEFAULT_MAX_IMPACT_DEPTH,
    max_symbols: int = DEFAULT_MAX_IMPACT_SYMBOLS,
) -> ImpactClosure:
    """Traverse a deterministic directed impact map under hard limits."""

    if isinstance(max_depth, bool) or not isinstance(max_depth, int):
        raise ContractMismatchError("max_depth must be an integer")
    if isinstance(max_symbols, bool) or not isinstance(max_symbols, int):
        raise ContractMismatchError("max_symbols must be an integer")
    if not 0 <= max_depth <= HARD_MAX_IMPACT_DEPTH:
        raise ContractMismatchError(
            f"max_depth must be between 0 and {HARD_MAX_IMPACT_DEPTH}"
        )
    if not 1 <= max_symbols <= HARD_MAX_IMPACT_SYMBOLS:
        raise ContractMismatchError(
            f"max_symbols must be between 1 and {HARD_MAX_IMPACT_SYMBOLS}"
        )
    seeds = _ids(seed_symbols, "seed_symbols", required=True)
    edges = impact_edges or {}
    if not isinstance(edges, Mapping):
        raise ContractMismatchError("impact_edges must be a mapping")
    normalized: dict[str, tuple[str, ...]] = {}
    for source, targets in edges.items():
        source_id = _text(source, "impact edge source")
        normalized[source_id] = _ids(targets, "impact edge targets")

    queue: deque[tuple[str, int]] = deque((item, 0) for item in seeds)
    seen: set[str] = set()
    truncated = False
    while queue:
        symbol, depth = queue.popleft()
        if symbol in seen:
            continue
        if len(seen) >= max_symbols:
            truncated = True
            break
        seen.add(symbol)
        neighbors = normalized.get(symbol, ())
        if neighbors and depth >= max_depth:
            truncated = True
            continue
        for neighbor in neighbors:
            if neighbor not in seen:
                queue.append((neighbor, depth + 1))
    return ImpactClosure(
        symbols=tuple(sorted(seen)),
        truncated=truncated,
        max_depth=max_depth,
        max_symbols=max_symbols,
    )


# Compatibility-friendly spelling for downstream task materializers.
compute_bounded_impact_closure = bounded_impact_closure


@dataclass(frozen=True, slots=True)
class ContractFinding:
    """One deterministic, snapshot-bound contract mismatch."""

    snapshot_id: str
    contract_id: str
    claim_family: str
    affected_symbols: tuple[str, ...]
    affected_paths: tuple[str, ...]
    counterexample_id: str
    counterexample: Mapping[str, Any]
    state: MismatchState
    lifecycle: FindingLifecycle
    ownership: tuple[SourceOwnership, ...]
    reproduction: ReproductionHandles
    evidence: tuple[FindingEvidence, ...]
    impact_truncated: bool = False
    finding_id: str = ""

    def __post_init__(self) -> None:
        for name in ("snapshot_id", "contract_id", "claim_family"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "affected_symbols",
            _ids(
                self.affected_symbols,
                "affected_symbols",
                required=True,
                maximum=HARD_MAX_IMPACT_SYMBOLS,
            ),
        )
        object.__setattr__(
            self, "affected_paths", _paths(self.affected_paths, "affected_paths")
        )
        object.__setattr__(
            self, "counterexample_id", _text(self.counterexample_id, "counterexample_id")
        )
        counterexample = _plain(self.counterexample)
        if not isinstance(counterexample, Mapping):
            raise ContractMismatchError("counterexample must be an object")
        if len(canonical_json_bytes(counterexample)) > MAX_COUNTEREXAMPLE_BYTES:
            raise ContractMismatchError("counterexample exceeds its byte bound")
        object.__setattr__(
            self, "counterexample", MappingProxyType(dict(counterexample))
        )
        object.__setattr__(
            self, "state", _enum(self.state, MismatchState, "mismatch state")
        )
        object.__setattr__(
            self,
            "lifecycle",
            _enum(self.lifecycle, FindingLifecycle, "finding lifecycle"),
        )
        owners = tuple(
            item
            if isinstance(item, SourceOwnership)
            else SourceOwnership.from_dict(item)
            for item in self.ownership
        )
        if tuple(item.path for item in owners) != self.affected_paths:
            raise ContractMismatchError(
                "ownership must account for every affected path exactly once"
            )
        object.__setattr__(self, "ownership", owners)
        if not isinstance(self.reproduction, ReproductionHandles):
            if not isinstance(self.reproduction, Mapping):
                raise ContractMismatchError("reproduction must be exact handles")
            object.__setattr__(
                self,
                "reproduction",
                ReproductionHandles.from_dict(self.reproduction),
            )
        if (
            self.reproduction.snapshot_id != self.snapshot_id
            or self.reproduction.contract_id != self.contract_id
            or self.reproduction.counterexample_id != self.counterexample_id
        ):
            raise ContractMismatchError(
                "reproduction handles are bound to another finding"
            )
        revisions = tuple(
            item
            if isinstance(item, FindingEvidence)
            else FindingEvidence.from_dict(item)
            for item in self.evidence
        )
        by_id = {item.evidence_revision_id: item for item in revisions}
        if not by_id:
            raise ContractMismatchError("finding requires evidence")
        if len(by_id) > MAX_FINDING_EVIDENCE_REVISIONS:
            raise ContractMismatchError("finding evidence history exceeds its bound")
        object.__setattr__(
            self, "evidence", tuple(by_id[key] for key in sorted(by_id))
        )
        if self.reproduction.claim_id not in {
            item.claim_id for item in self.evidence
        }:
            raise ContractMismatchError(
                "reproduction claim is absent from finding evidence"
            )
        if not isinstance(self.impact_truncated, bool):
            raise ContractMismatchError("impact_truncated must be boolean")
        expected = content_identity(self._dedupe_payload())
        if self.finding_id and self.finding_id != expected:
            raise ContractMismatchError("finding dedupe identity mismatch")
        object.__setattr__(self, "finding_id", expected)

    def _dedupe_payload(self) -> dict[str, Any]:
        """The exact five-dimensional identity required by SCA-090."""

        return {
            "schema": CONTRACT_FINDING_SCHEMA,
            "snapshot_id": self.snapshot_id,
            "contract_id": self.contract_id,
            "claim_family": self.claim_family,
            "affected_symbols": list(self.affected_symbols),
            "counterexample_id": self.counterexample_id,
        }

    @property
    def dedupe_id(self) -> str:
        return self.finding_id

    @property
    def owners(self) -> tuple[SourceOwner, ...]:
        return tuple(
            sorted({item.owner for item in self.ownership}, key=lambda item: item.value)
        )

    @property
    def source_owner(self) -> SourceOwner:
        """Return one owner only when every affected path agrees."""

        owners = self.owners
        return owners[0] if len(owners) == 1 else SourceOwner.UNRESOLVED

    @property
    def record_id(self) -> str:
        return content_identity(self.to_dict(include_record_id=False))

    def to_dict(self, *, include_record_id: bool = True) -> dict[str, Any]:
        payload = {
            "schema": CONTRACT_FINDING_SCHEMA,
            "interface": CONTRACT_FINDING_INTERFACE,
            "version": CONTRACT_MISMATCH_ANALYZER_VERSION,
            "finding_id": self.finding_id,
            "dedupe_id": self.finding_id,
            "snapshot_id": self.snapshot_id,
            "contract_id": self.contract_id,
            "claim_family": self.claim_family,
            "affected_symbols": list(self.affected_symbols),
            "affected_paths": list(self.affected_paths),
            "counterexample_id": self.counterexample_id,
            "counterexample": dict(self.counterexample),
            "state": self.state.value,
            "lifecycle": self.lifecycle.value,
            "ownership": [item.to_dict() for item in self.ownership],
            "source_owners": [item.value for item in self.owners],
            "reproduction": self.reproduction.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "impact_truncated": self.impact_truncated,
        }
        if include_record_id:
            payload["record_id"] = content_identity(payload)
        return payload

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContractFinding":
        if value.get("schema") not in (None, CONTRACT_FINDING_SCHEMA):
            raise ContractMismatchError("unsupported contract-finding schema")
        if value.get("interface") not in (None, CONTRACT_FINDING_INTERFACE):
            raise ContractMismatchError("unsupported contract-finding interface")
        result = cls(
            snapshot_id=str(value.get("snapshot_id") or ""),
            contract_id=str(value.get("contract_id") or ""),
            claim_family=str(value.get("claim_family") or ""),
            affected_symbols=tuple(value.get("affected_symbols") or ()),
            affected_paths=tuple(value.get("affected_paths") or ()),
            counterexample_id=str(value.get("counterexample_id") or ""),
            counterexample=value.get("counterexample") or {},
            state=value.get("state", ""),
            lifecycle=value.get("lifecycle", FindingLifecycle.ACTIVE),
            ownership=tuple(
                SourceOwnership.from_dict(item)
                for item in value.get("ownership") or ()
            ),
            reproduction=ReproductionHandles.from_dict(
                value.get("reproduction") or {}
            ),
            evidence=tuple(
                FindingEvidence.from_dict(item)
                for item in value.get("evidence") or ()
            ),
            impact_truncated=value.get("impact_truncated", False),
            finding_id=str(
                value.get("finding_id") or value.get("dedupe_id") or ""
            ),
        )
        if value.get("dedupe_id") not in (None, result.finding_id):
            raise ContractMismatchError("finding dedupe claim mismatch")
        if value.get("record_id") not in (None, result.record_id):
            raise ContractMismatchError("finding record identity mismatch")
        claimed_owners = value.get("source_owners")
        if claimed_owners is not None and tuple(claimed_owners) != tuple(
            item.value for item in result.owners
        ):
            raise ContractMismatchError("finding source-owner claim mismatch")
        return result

    @classmethod
    def from_json(cls, value: str | bytes) -> "ContractFinding":
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ContractMismatchError("finding JSON is malformed") from exc
        if not isinstance(payload, Mapping):
            raise ContractMismatchError("finding JSON must contain an object")
        return cls.from_dict(payload)


def _merge_ids(left: Sequence[str], right: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(set(left) | set(right)))


def merge_finding_evidence(
    previous: ContractFinding,
    current: ContractFinding,
) -> ContractFinding:
    """Upsert changed evidence without changing the dedupe identity."""

    if previous.finding_id != current.finding_id:
        raise ContractMismatchError("cannot merge different finding identities")
    revisions = {
        item.evidence_revision_id: item
        for item in (*previous.evidence, *current.evidence)
    }
    if len(revisions) > MAX_FINDING_EVIDENCE_REVISIONS:
        # Retention is deterministic: retain lexicographically greatest IDs.
        revisions = {
            key: revisions[key]
            for key in sorted(revisions)[-MAX_FINDING_EVIDENCE_REVISIONS:]
        }
    lifecycle = current.lifecycle
    if (
        current.state is not MismatchState.STALE
        and previous.lifecycle in {
            FindingLifecycle.STALE,
            FindingLifecycle.RESOLVED,
        }
    ):
        lifecycle = FindingLifecycle.REOPENED
    reproduction = replace(
        current.reproduction,
        obligation_ids=_merge_ids(
            previous.reproduction.obligation_ids,
            current.reproduction.obligation_ids,
        ),
        receipt_ids=_merge_ids(
            previous.reproduction.receipt_ids,
            current.reproduction.receipt_ids,
        ),
        evidence_ids=_merge_ids(
            previous.reproduction.evidence_ids,
            current.reproduction.evidence_ids,
        ),
        cas_handles=_merge_ids(
            previous.reproduction.cas_handles,
            current.reproduction.cas_handles,
        ),
        commands=_merge_ids(
            previous.reproduction.commands,
            current.reproduction.commands,
        ),
    )
    return replace(
        current,
        lifecycle=lifecycle,
        reproduction=reproduction,
        evidence=tuple(revisions[key] for key in sorted(revisions)),
    )


# Short name commonly used by stores/refineries.
upsert_finding = merge_finding_evidence


@dataclass(frozen=True, slots=True)
class MismatchAnalysis:
    """Deterministic analyzer output, including non-finding dispositions."""

    snapshot_id: str
    findings: tuple[ContractFinding, ...]
    ignored_claim_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    analysis_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "snapshot_id", _text(self.snapshot_id, "snapshot_id"))
        findings = tuple(sorted(self.findings, key=lambda item: item.finding_id))
        by_id = {item.finding_id: item for item in findings}
        if len(by_id) != len(findings):
            raise ContractMismatchError(
                "analysis contains duplicate finding identities"
            )
        if any(item.snapshot_id != self.snapshot_id for item in findings):
            raise ContractMismatchError("analysis contains a foreign snapshot")
        object.__setattr__(self, "findings", findings)
        object.__setattr__(
            self, "ignored_claim_ids", _ids(self.ignored_claim_ids, "ignored_claim_ids")
        )
        object.__setattr__(
            self, "reason_codes", _ids(self.reason_codes, "reason_codes")
        )
        expected = content_identity(self._identity_payload())
        if self.analysis_id and self.analysis_id != expected:
            raise ContractMismatchError("analysis identity mismatch")
        object.__setattr__(self, "analysis_id", expected)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema": CONTRACT_MISMATCH_ANALYSIS_SCHEMA,
            "interface": CONTRACT_MISMATCH_ANALYZER_INTERFACE,
            "version": CONTRACT_MISMATCH_ANALYZER_VERSION,
            "snapshot_id": self.snapshot_id,
            "findings": [item.to_dict() for item in self.findings],
            "ignored_claim_ids": list(self.ignored_claim_ids),
            "reason_codes": list(self.reason_codes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"analysis_id": self.analysis_id, **self._identity_payload()}

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MismatchAnalysis":
        if value.get("schema") not in (None, CONTRACT_MISMATCH_ANALYSIS_SCHEMA):
            raise ContractMismatchError("unsupported mismatch-analysis schema")
        if value.get("interface") not in (
            None,
            CONTRACT_MISMATCH_ANALYZER_INTERFACE,
        ):
            raise ContractMismatchError("unsupported mismatch-analysis interface")
        return cls(
            snapshot_id=str(value.get("snapshot_id") or ""),
            findings=tuple(
                ContractFinding.from_dict(item)
                for item in value.get("findings") or ()
            ),
            ignored_claim_ids=tuple(value.get("ignored_claim_ids") or ()),
            reason_codes=tuple(value.get("reason_codes") or ()),
            analysis_id=str(value.get("analysis_id") or ""),
        )

    @classmethod
    def from_json(cls, value: str | bytes) -> "MismatchAnalysis":
        try:
            payload = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ContractMismatchError(
                "mismatch-analysis JSON is malformed"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ContractMismatchError(
                "mismatch-analysis JSON must contain an object"
            )
        return cls.from_dict(payload)


# Compatibility aliases for callers which use the interface name as a type.
ContractMismatchAnalysis = MismatchAnalysis
ContractFindingState = MismatchState
ContractFindingLifecycle = FindingLifecycle
SourceOwnershipRoute = SourceOwnership


@dataclass(frozen=True, slots=True)
class _NormalizedClaim:
    claim_id: str
    family: str
    state: MismatchState | None
    reason_codes: tuple[str, ...]
    premise_ids: tuple[str, ...]
    obligation_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    receipt_ids: tuple[str, ...]
    proof_result_ids: tuple[str, ...]
    counterexamples: tuple[Any, ...]
    symbols: tuple[str, ...]
    paths: tuple[str, ...]


def _counterexample(value: Any) -> tuple[str, Mapping[str, Any]]:
    if isinstance(value, FormalCounterexample):
        return value.semantic_id, MappingProxyType(value.to_capsule_dict())
    if isinstance(value, ContractCounterexample):
        return value.counterexample_id, MappingProxyType(value.to_dict())
    if not isinstance(value, Mapping):
        raise ContractMismatchError("counterexample must be typed or an object")
    capsule = _plain(value)
    if not isinstance(capsule, Mapping):
        raise ContractMismatchError("counterexample must be an object")
    schema = capsule.get("schema")
    if schema == getattr(FormalCounterexample, "SCHEMA", None):
        restored = FormalCounterexample.from_dict(capsule)
        return restored.semantic_id, MappingProxyType(restored.to_capsule_dict())
    if schema == "ipfs_accelerate_py/agent-supervisor/mcp-contract-counterexample@1":
        restored = ContractCounterexample.from_dict(capsule)
        return restored.counterexample_id, MappingProxyType(restored.to_dict())
    claimed = str(
        capsule.get("counterexample_id")
        or capsule.get("semantic_id")
        or ""
    )
    # Generic query sketches do not share one canonical schema.  Bind their
    # exact retained projection.  A claimed upstream ID is evidence within the
    # projection, never an unchecked substitute for its content identity.
    identifier = content_identity(
        {
            "schema": "contract-mismatch-counterexample-projection@1",
            "claimed_counterexample_id": claimed,
            "value": capsule,
        }
    )
    return _text(identifier, "counterexample_id"), MappingProxyType(dict(capsule))


def _state_witness(
    *,
    claim_id: str,
    state: MismatchState,
    reason_codes: Sequence[str],
    premise_ids: Sequence[str],
) -> tuple[str, Mapping[str, Any]]:
    payload = {
        "schema": "ipfs_accelerate_py/agent-supervisor/claim-state-witness@1",
        "claim_id": claim_id,
        "state": state.value,
        "reason_codes": list(_ids(reason_codes, "reason_codes")),
        "premise_ids": list(_ids(premise_ids, "premise_ids")),
    }
    return content_identity(payload), MappingProxyType(payload)


def _mapping_values(
    value: Mapping[str, Any],
    *names: str,
    default: Any = None,
) -> Any:
    for name in names:
        if name in value:
            return value[name]
    return default


def _normalize_claim(
    claim: Any,
    *,
    proof_result: Any = None,
    claim_family: Any = "",
    affected_symbols: Sequence[str] = (),
    affected_paths: Sequence[str] = (),
) -> _NormalizedClaim:
    if isinstance(claim, ClaimQueryHit):
        claim = claim.to_dict()
    if isinstance(claim, ContractParityClaim):
        claim_id = claim.claim_id
        family = claim.family.value
        state_value: Any = claim.state
        reasons = claim.reason_codes
        premises = claim.premise_ids
        obligations: tuple[str, ...] = ()
        evidence_ids: tuple[str, ...] = ()
        receipts: tuple[str, ...] = ()
        proof_ids: tuple[str, ...] = ()
        counterexamples: tuple[Any, ...] = claim.counterexamples
        symbols: tuple[str, ...] = ()
        paths: tuple[str, ...] = ()
    elif isinstance(claim, Mapping):
        claim_id = str(
            _mapping_values(
                claim,
                "claim_id",
                "property_id",
                "analysis_id",
                default="",
            )
            or ""
        )
        family = str(
            _mapping_values(
                claim,
                "claim_family",
                "family",
                "property_class",
                default=claim_family,
            )
            or ""
        )
        state_value = _mapping_values(
            claim, "state", "status", "outcome", "verdict", default=""
        )
        reasons = _mapping_values(
            claim, "reason_codes", "reasons", default=()
        )
        premises = _mapping_values(
            claim, "premise_ids", "failed_premise_ids", default=()
        )
        obligations = _ids(
            _mapping_values(
                claim, "obligation_ids", "obligations", default=()
            ),
            "obligation_ids",
        )
        evidence_ids = _ids(
            _mapping_values(claim, "evidence_ids", default=()),
            "evidence_ids",
        )
        receipts = _ids(
            _mapping_values(claim, "receipt_ids", default=()),
            "receipt_ids",
        )
        proof_ids = _ids(
            _mapping_values(claim, "proof_result_ids", default=()),
            "proof_result_ids",
        )
        raw_counterexamples = _mapping_values(
            claim, "counterexamples", default=None
        )
        if raw_counterexamples is None:
            one = _mapping_values(claim, "counterexample", default=None)
            raw_counterexamples = () if one is None else (one,)
        if isinstance(raw_counterexamples, Mapping):
            raw_counterexamples = (raw_counterexamples,)
        counterexamples = tuple(raw_counterexamples or ())
        provenance = claim.get("provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        symbols = _ids(
            _mapping_values(
                claim,
                "affected_symbols",
                "symbols",
                default=provenance.get("symbols") or (),
            ),
            "affected_symbols",
        )
        paths = _paths(
            _mapping_values(
                claim,
                "affected_paths",
                "paths",
                default=provenance.get("paths") or (),
            ),
            "affected_paths",
        )
    else:
        raise ContractMismatchError(
            "claim must be a ContractParityClaim or canonical object"
        )

    claim_id = _text(claim_id, "claim_id")
    selected_family = claim_family or family
    family = _text(
        str(getattr(selected_family, "value", selected_family)),
        "claim_family",
    )
    state = _state(state_value)
    reasons = _ids(reasons, "reason_codes")
    premises = _ids(premises, "premise_ids")
    symbols = _ids((*symbols, *affected_symbols), "affected_symbols")
    paths = _paths((*paths, *affected_paths), "affected_paths")

    if proof_result is not None:
        if isinstance(proof_result, McpContractProofResult):
            state = _state(proof_result.outcome)
            reasons = _merge_ids(reasons, proof_result.reason_codes)
            obligations = _merge_ids(
                obligations, (proof_result.obligation_id,)
            )
            receipt_id = str(
                getattr(proof_result.receipt, "content_id", "") or ""
            )
            receipts = _merge_ids(
                receipts, (receipt_id,) if receipt_id else ()
            )
            proof_id = str(getattr(proof_result, "content_id", "") or "")
            proof_ids = _merge_ids(
                proof_ids, (proof_id,) if proof_id else ()
            )
            counterexamples = (
                ()
                if proof_result.counterexample is None
                else (proof_result.counterexample,)
            )
        elif isinstance(proof_result, Mapping):
            state = _state(
                _mapping_values(
                    proof_result,
                    "outcome",
                    "state",
                    "status",
                    "verdict",
                    default="",
                )
            )
            reasons = _merge_ids(
                reasons,
                _ids(
                    proof_result.get("reason_codes") or (),
                    "reason_codes",
                ),
            )
            obligation = str(proof_result.get("obligation_id") or "")
            obligations = _merge_ids(
                obligations, (obligation,) if obligation else ()
            )
            receipt = proof_result.get("receipt")
            receipt_id = ""
            if isinstance(receipt, Mapping):
                receipt_id = str(
                    receipt.get("content_id")
                    or receipt.get("receipt_id")
                    or ""
                )
            receipts = _merge_ids(
                receipts, (receipt_id,) if receipt_id else ()
            )
            proof_id = str(
                proof_result.get("content_id")
                or proof_result.get("proof_result_id")
                or ""
            )
            proof_ids = _merge_ids(
                proof_ids, (proof_id,) if proof_id else ()
            )
            one = proof_result.get("counterexample")
            counterexamples = () if one is None else (one,)
        else:
            raise ContractMismatchError(
                "proof_result must be an McpContractProofResult or object"
            )

    if state is not None and not reasons:
        reasons = (f"claim_{state.value}",)
    return _NormalizedClaim(
        claim_id=claim_id,
        family=family,
        state=state,
        reason_codes=reasons,
        premise_ids=premises,
        obligation_ids=obligations,
        evidence_ids=evidence_ids,
        receipt_ids=receipts,
        proof_result_ids=proof_ids,
        counterexamples=counterexamples,
        symbols=symbols,
        paths=paths,
    )


class ContractMismatchAnalyzer:
    """Deterministic claim-to-finding analyzer with bounded impact closure."""

    def __init__(
        self,
        *,
        max_impact_depth: int = DEFAULT_MAX_IMPACT_DEPTH,
        max_impact_symbols: int = DEFAULT_MAX_IMPACT_SYMBOLS,
        default_reproduction_commands: Sequence[str] = (),
    ) -> None:
        # Validate limits using a harmless one-node closure.
        bounded_impact_closure(
            ("limit-probe",),
            max_depth=max_impact_depth,
            max_symbols=max_impact_symbols,
        )
        self.max_impact_depth = max_impact_depth
        self.max_impact_symbols = max_impact_symbols
        self.default_reproduction_commands = _ids(
            default_reproduction_commands,
            "default_reproduction_commands",
            maximum=MAX_REPRODUCTION_COMMANDS,
        )

    def analyze_claim(
        self,
        claim: ContractParityClaim | Mapping[str, Any],
        *,
        snapshot_id: str,
        contract_id: str,
        affected_symbols: Sequence[str] = (),
        affected_paths: Sequence[str] = (),
        impact_edges: Mapping[str, Sequence[str]] | None = None,
        proof_result: McpContractProofResult | Mapping[str, Any] | None = None,
        claim_family: Any = "",
        obligation_ids: Sequence[str] = (),
        evidence_ids: Sequence[str] = (),
        cas_handles: Sequence[str] = (),
        reproduction_commands: Sequence[str] = (),
        previous: Sequence[ContractFinding] = (),
    ) -> tuple[ContractFinding, ...]:
        """Analyze one claim; a claim with several witnesses yields one per witness."""

        snapshot_id = _text(snapshot_id, "snapshot_id")
        contract_id = _text(contract_id, "contract_id")
        normalized = _normalize_claim(
            claim,
            proof_result=proof_result,
            claim_family=claim_family,
            affected_symbols=affected_symbols,
            affected_paths=affected_paths,
        )
        if normalized.state is None:
            return ()
        closure = bounded_impact_closure(
            normalized.symbols,
            impact_edges,
            max_depth=self.max_impact_depth,
            max_symbols=self.max_impact_symbols,
        )
        all_obligations = _merge_ids(
            normalized.obligation_ids, _ids(obligation_ids, "obligation_ids")
        )
        all_evidence = _merge_ids(
            normalized.evidence_ids, _ids(evidence_ids, "evidence_ids")
        )
        commands = _merge_ids(
            self.default_reproduction_commands,
            _ids(
                reproduction_commands,
                "reproduction_commands",
                maximum=MAX_REPRODUCTION_COMMANDS,
            ),
        )
        if not all_obligations and not commands:
            raise ContractMismatchError(
                "finding requires an exact obligation or reproduction command"
            )
        witnesses = normalized.counterexamples
        if not witnesses:
            if normalized.state is MismatchState.REFUTED:
                raise ContractMismatchError(
                    "refuted claim requires a compact counterexample"
                )
            witnesses = (
                _state_witness(
                    claim_id=normalized.claim_id,
                    state=normalized.state,
                    reason_codes=normalized.reason_codes,
                    premise_ids=normalized.premise_ids,
                ),
            )

        prior_by_id = {item.finding_id: item for item in previous}
        findings: list[ContractFinding] = []
        for witness in witnesses:
            if (
                isinstance(witness, tuple)
                and len(witness) == 2
                and isinstance(witness[0], str)
                and isinstance(witness[1], Mapping)
            ):
                counterexample_id, capsule = witness
            else:
                counterexample_id, capsule = _counterexample(witness)
            evidence = FindingEvidence(
                claim_id=normalized.claim_id,
                state=normalized.state,
                reason_codes=normalized.reason_codes,
                premise_ids=normalized.premise_ids,
                obligation_ids=all_obligations,
                evidence_ids=all_evidence,
                receipt_ids=normalized.receipt_ids,
                proof_result_ids=normalized.proof_result_ids,
            )
            lifecycle = (
                FindingLifecycle.STALE
                if normalized.state is MismatchState.STALE
                else FindingLifecycle.ACTIVE
            )
            finding = ContractFinding(
                snapshot_id=snapshot_id,
                contract_id=contract_id,
                claim_family=normalized.family,
                affected_symbols=closure.symbols,
                affected_paths=normalized.paths,
                counterexample_id=counterexample_id,
                counterexample=capsule,
                state=normalized.state,
                lifecycle=lifecycle,
                ownership=resolve_source_ownership(normalized.paths),
                reproduction=ReproductionHandles(
                    snapshot_id=snapshot_id,
                    contract_id=contract_id,
                    claim_id=normalized.claim_id,
                    counterexample_id=counterexample_id,
                    obligation_ids=all_obligations,
                    receipt_ids=normalized.receipt_ids,
                    evidence_ids=all_evidence,
                    cas_handles=_ids(cas_handles, "cas_handles"),
                    commands=commands,
                ),
                evidence=(evidence,),
                impact_truncated=closure.truncated,
            )
            prior = prior_by_id.get(finding.finding_id)
            findings.append(
                merge_finding_evidence(prior, finding)
                if prior is not None
                else finding
            )
        return tuple(sorted(findings, key=lambda item: item.finding_id))

    def analyze(
        self,
        claims: Sequence[
            ContractParityClaim | ClaimQueryHit | Mapping[str, Any]
        ],
        *,
        snapshot_id: str,
        contract_id: str,
        affected_symbols: Sequence[str] = (),
        affected_paths: Sequence[str] = (),
        impact_edges: Mapping[str, Sequence[str]] | None = None,
        proof_results: (
            Mapping[str, McpContractProofResult | Mapping[str, Any]]
            | Sequence[McpContractProofResult | Mapping[str, Any] | None]
            | None
        ) = None,
        claim_family: Any = "",
        obligation_ids: Sequence[str] = (),
        evidence_ids: Sequence[str] = (),
        cas_handles: Sequence[str] = (),
        reproduction_commands: Sequence[str] = (),
        previous: Sequence[ContractFinding] = (),
    ) -> MismatchAnalysis:
        """Analyze a stable sequence and deduplicate/upsert its findings."""

        if isinstance(claims, (str, bytes, bytearray)) or not isinstance(
            claims, Sequence
        ):
            raise ContractMismatchError("claims must be a sequence")
        previous_by_id = {item.finding_id: item for item in previous}
        by_id: dict[str, ContractFinding] = {}
        ignored: list[str] = []
        for index, claim in enumerate(claims):
            claim_id = (
                claim.claim_id
                if isinstance(claim, (ContractParityClaim, ClaimQueryHit))
                else str(
                    _mapping_values(
                        claim,
                        "claim_id",
                        "property_id",
                        "analysis_id",
                        default=f"claim:{index}",
                    )
                )
            )
            proof_result = None
            if isinstance(proof_results, Mapping):
                proof_result = proof_results.get(claim_id)
            elif proof_results is not None:
                if isinstance(proof_results, (str, bytes, bytearray)):
                    raise ContractMismatchError(
                        "proof_results must be a mapping or sequence"
                    )
                if index < len(proof_results):
                    proof_result = proof_results[index]
            findings = self.analyze_claim(
                claim,
                snapshot_id=snapshot_id,
                contract_id=contract_id,
                affected_symbols=affected_symbols,
                affected_paths=affected_paths,
                impact_edges=impact_edges,
                proof_result=proof_result,
                claim_family=claim_family,
                obligation_ids=obligation_ids,
                evidence_ids=evidence_ids,
                cas_handles=cas_handles,
                reproduction_commands=reproduction_commands,
                previous=previous,
            )
            if not findings:
                ignored.append(claim_id)
            for finding in findings:
                existing = by_id.get(finding.finding_id)
                by_id[finding.finding_id] = (
                    merge_finding_evidence(existing, finding)
                    if existing is not None
                    else finding
                )
        # Existing current-snapshot findings which disappeared are retained as
        # resolved history; cross-snapshot records have different identities.
        for finding_id, prior in previous_by_id.items():
            if prior.snapshot_id != snapshot_id or finding_id in by_id:
                continue
            by_id[finding_id] = replace(
                prior, lifecycle=FindingLifecycle.RESOLVED
            )
        return MismatchAnalysis(
            snapshot_id=snapshot_id,
            findings=tuple(by_id.values()),
            ignored_claim_ids=tuple(ignored),
            reason_codes=("cache_miss_and_unknown_are_not_refutations",),
        )

    def analyze_parity_report(
        self,
        report: McpContractAnalysis,
        *,
        snapshot_id: str,
        contract_id: str = "",
        **kwargs: Any,
    ) -> MismatchAnalysis:
        """Analyze every typed claim in one SCA-051 parity report."""

        if not isinstance(report, McpContractAnalysis):
            raise ContractMismatchError(
                "report must be an McpContractAnalysis"
            )
        selected_contract = contract_id or report.expected_contract_id
        evidence_ids = _merge_ids(
            kwargs.pop("evidence_ids", ()), (report.analysis_id,)
        )
        return self.analyze(
            report.claims,
            snapshot_id=snapshot_id,
            contract_id=selected_contract,
            evidence_ids=evidence_ids,
            **kwargs,
        )

    def analyze_code_proof_query(
        self,
        query: CodeProofQuery | CodeProofQueryResult,
        *,
        snapshot_id: str = "",
        contract_id: str = "",
        claim_family: Any = "",
        **kwargs: Any,
    ) -> MismatchAnalysis:
        """Consume the typed ``CodeProofQuery@1`` surface without coercion."""

        if isinstance(query, CodeProofQuery):
            hits = query.hits
            repository_tree_id = query.repository_tree_id
            query_evidence_id = content_identity(
                {
                    "interface": "CodeProofQuery@1",
                    "repository_tree_id": repository_tree_id,
                    "hits": [item.to_dict() for item in hits],
                }
            )
        elif isinstance(query, CodeProofQueryResult):
            hits = query.hits
            repository_tree_id = query.repository_tree_id
            query_evidence_id = query.result_id
        else:
            raise ContractMismatchError(
                "query must be a CodeProofQuery or CodeProofQueryResult"
            )
        selected_snapshot = snapshot_id or repository_tree_id
        if not selected_snapshot:
            raise ContractMismatchError(
                "snapshot_id is required when the query has no repository tree"
            )
        selected_contract = contract_id
        if not selected_contract:
            property_ids = {item.property_id for item in hits}
            if len(property_ids) != 1:
                raise ContractMismatchError(
                    "contract_id is required for a multi-property query"
                )
            selected_contract = next(iter(property_ids))
        evidence_ids = _merge_ids(
            kwargs.pop("evidence_ids", ()), (query_evidence_id,)
        )
        return self.analyze(
            hits,
            snapshot_id=selected_snapshot,
            contract_id=selected_contract,
            claim_family=claim_family,
            evidence_ids=evidence_ids,
            **kwargs,
        )


def analyze_contract_mismatches(
    claims: Sequence[
        ContractParityClaim | ClaimQueryHit | Mapping[str, Any]
    ],
    **kwargs: Any,
) -> MismatchAnalysis:
    """Functional entry point for :class:`ContractMismatchAnalyzer`."""

    analyzer_keys = {
        "max_impact_depth",
        "max_impact_symbols",
        "default_reproduction_commands",
    }
    analyzer_kwargs = {
        key: kwargs.pop(key) for key in tuple(kwargs) if key in analyzer_keys
    }
    return ContractMismatchAnalyzer(**analyzer_kwargs).analyze(claims, **kwargs)


__all__ = [
    "CONTRACT_FINDING_INTERFACE",
    "CONTRACT_FINDING_SCHEMA",
    "CONTRACT_MISMATCH_ANALYZER_INTERFACE",
    "CONTRACT_MISMATCH_ANALYZER_VERSION",
    "ContractFinding",
    "ContractFindingLifecycle",
    "ContractFindingState",
    "ContractMismatchAnalysis",
    "ContractMismatchAnalyzer",
    "ContractMismatchError",
    "FindingEvidence",
    "FindingLifecycle",
    "ImpactClosure",
    "MismatchAnalysis",
    "MismatchState",
    "ReproductionHandles",
    "SourceOwner",
    "SourceOwnership",
    "SourceOwnershipRoute",
    "analyze_contract_mismatches",
    "ClaimQueryHit",
    "bounded_impact_closure",
    "compute_bounded_impact_closure",
    "merge_finding_evidence",
    "resolve_source_ownership",
    "route_source_owner",
    "upsert_finding",
]
