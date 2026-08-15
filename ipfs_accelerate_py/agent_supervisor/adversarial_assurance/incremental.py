"""Incremental mutation verification composition (AAE-043).

Interface surface:

* ``IncrementalMutationVerifier@1`` — for each admitted mutant, integrate
  existing invalidation selection, exact-key verification-cache reuse,
  temporary proof forests, survivor broadening policy, and full-versus-
  incremental cost accounting.

Normative properties (acceptance):

* Only *affected* units invalidate; unrelated units remain reuse candidates.
* Cache reuse requires *complete* receipt keys; incomplete keys never reuse.
* Survivors broaden by explicit policy (never silently, never when disabled).
* Temporary proof forests never replace or publish as canonical seals.
* Full cost, incremental cost, and cache reuse are measured on every run.
* No production policy change; missing/invalid inputs fail closed.

Cold import is side-effect free: no Git, store, process, network, or
filesystem operations run at import time.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    content_identity,
)
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.metrics import (
    COST_COMPARISON_SCHEMA,
    COST_EVIDENCE,
    COST_RECORD_SCHEMA,
    CostKind,
    CostProvenance,
    CostValue,
    ProofCostComparison,
    ProofCostRecord,
    ProofMetricsCollector,
    RunDisposition,
    compare_costs,
)
from ipfs_accelerate_py.agent_supervisor.verification.selection import (
    FallbackMode,
    SelectionDisposition,
    SelectionPolicy,
    VerificationCatalog,
    select_affected_verification,
)

# ---------------------------------------------------------------------------
# Schema / interface / evidence
# ---------------------------------------------------------------------------

INCREMENTAL_MUTATION_VERIFIER_INTERFACE: Final[str] = (
    "IncrementalMutationVerifier@1"
)
INCREMENTAL_MUTATION_VERIFIER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "incremental-mutation-verifier@1"
)
INCREMENTAL_MUTATION_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "incremental-mutation-result@1"
)
INCREMENTAL_MUTATION_RESULT_INTERFACE: Final[str] = (
    "IncrementalMutationVerificationResult@1"
)
PROOF_UNIT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-proof-unit@1"
)
TEMPORARY_PROOF_FOREST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "temporary-proof-forest@1"
)
TEMPORARY_PROOF_FOREST_INTERFACE: Final[str] = "TemporaryProofForest@1"
SURVIVOR_BROADENING_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "survivor-broadening-policy@1"
)
MUTATION_COST_ACCOUNTING_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "mutation-cost-accounting@1"
)
CACHE_KEY_COMPLETENESS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/adversarial-assurance-"
    "cache-key-completeness@1"
)

AAE_INCREMENTAL_EVIDENCE: Final[str] = "aae/incremental-mutation-verifier@1"
ADAPTER_ID: Final[str] = "aae-incremental-mutation-verifier"
BOARD_NAMESPACE: Final[str] = "adversarial-assurance-engine-v1"
GENERATOR_ID: Final[str] = "incremental_mutation_verifier"
GENERATOR_VERSION: Final[str] = "1.0.0"

MAX_UNITS: Final[int] = 50_000
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_DIAGNOSTIC: Final[int] = 1_024
MAX_REASON_CODES: Final[int] = 128
MAX_EDGES: Final[int] = 50_000
DEFAULT_UNIT_CPU_MS: Final[int] = 5_000
DEFAULT_UNIT_WALL_MS: Final[int] = 5_000
DEFAULT_REUSE_CPU_MS: Final[int] = 1
DEFAULT_REUSE_WALL_MS: Final[int] = 1

# Closed reason codes.
REASON_AFFECTED_INVALIDATED: Final[str] = "affected_unit_invalidated"
REASON_UNAFFECTED_REUSE_CANDIDATE: Final[str] = "unaffected_reuse_candidate"
REASON_COMPLETE_KEY_REUSED: Final[str] = "complete_key_reused"
REASON_INCOMPLETE_KEY_REJECTED: Final[str] = "incomplete_key_rejected"
REASON_MISSING_KEY_REJECTED: Final[str] = "missing_key_rejected"
REASON_CACHE_MISS: Final[str] = "cache_miss_requires_reverify"
REASON_STALE_RECEIPT: Final[str] = "stale_receipt_rejected"
REASON_SURVIVOR_BROADENED: Final[str] = "survivor_broadened_by_policy"
REASON_SURVIVOR_FULL_SUITE: Final[str] = "survivor_full_suite_by_policy"
REASON_HIGH_RISK_FULL_SUITE: Final[str] = "high_risk_requires_full_suite"
REASON_UNCERTAINTY_BROADENED: Final[str] = "uncertainty_requires_broader"
REASON_BROADENING_DISABLED: Final[str] = "survivor_broadening_disabled"
REASON_TEMPORARY_FOREST_ONLY: Final[str] = "temporary_forest_only"
REASON_CANONICAL_SEAL_PRESERVED: Final[str] = "canonical_seal_not_replaced"
REASON_CANONICAL_REPLACE_REFUSED: Final[str] = "canonical_seal_replace_refused"
REASON_COSTS_MEASURED: Final[str] = "full_and_incremental_costs_measured"
REASON_CACHE_REUSE_MEASURED: Final[str] = "cache_reuse_measured"
REASON_UNRELATED_NOT_INVALIDATED: Final[str] = "unrelated_unit_not_invalidated"
REASON_PRODUCTION_POLICY_UNCHANGED: Final[str] = "production_policy_unchanged"

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.:/+-]{0,127}$")

# Required fields for a complete verification cache key binding.
REQUIRED_CACHE_KEY_FIELDS: Final[tuple[str, ...]] = (
    "key_cid",
    "repository_tree_cid",
    "semantic_state_root_cid",
    "environment_cid",
    "dependency_lock_cid",
    "kind",
    "check_id",
    "tool_name",
    "tool_version",
)


# ---------------------------------------------------------------------------
# Errors / closed vocabularies
# ---------------------------------------------------------------------------


class IncrementalVerificationError(ValueError):
    """Fail-closed error for incremental mutation verification."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "incremental_verification_error",
    ) -> None:
        super().__init__(message)
        self.reason_code = _token(reason_code, field_name="reason_code")


class IncrementalBoundsError(IncrementalVerificationError):
    """An input exceeded deterministic compactness bounds."""


class CanonicalSealProtectionError(IncrementalVerificationError):
    """Temporary forests must never replace or publish as canonical seals."""


class UnitDisposition(str, Enum):
    """Closed disposition for one proof/test unit under a mutant."""

    INVALIDATED = "invalidated"
    REUSED = "reused"
    REVERIFY = "reverify"
    BROADENED = "broadened"
    FULL_SUITE = "full_suite"
    UNAFFECTED = "unaffected"


class UnitKind(str, Enum):
    """Closed verification unit kinds (align with receipt kinds)."""

    STATIC_ANALYSIS = "static_analysis"
    TYPE_CHECK = "type_check"
    TEST = "test"
    PROOF = "proof"


class BroadeningMode(str, Enum):
    """How far survivor policy expands selection."""

    NONE = "none"
    BROADER = "broader"
    FULL_SUITE = "full_suite"


class MutantOutcomeClass(str, Enum):
    """Closed terminal class observed after incremental checks."""

    KILLED = "killed"
    SURVIVOR = "survivor"
    EQUIVALENT = "equivalent"
    BLOCKED = "blocked"
    INCONCLUSIVE = "inconclusive"


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _token(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise IncrementalVerificationError(f"{field_name} must be a string")
    text = value.strip().lower()
    if not text:
        raise IncrementalVerificationError(f"{field_name} must not be empty")
    if not _TOKEN_RE.match(text):
        raise IncrementalVerificationError(
            f"{field_name} must be a closed token (got {value!r})"
        )
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise IncrementalBoundsError(
            f"{field_name} exceeds {MAX_TEXT_BYTES} UTF-8 bytes"
        )
    return text


def _text(value: Any, *, field_name: str, required: bool = True) -> str:
    if value is None:
        if required:
            raise IncrementalVerificationError(f"{field_name} is required")
        return ""
    if not isinstance(value, str):
        raise IncrementalVerificationError(f"{field_name} must be a string")
    text = value.strip()
    if required and not text:
        raise IncrementalVerificationError(f"{field_name} must not be empty")
    if "\x00" in text:
        raise IncrementalVerificationError(f"{field_name} must not contain NUL")
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise IncrementalBoundsError(
            f"{field_name} exceeds {MAX_TEXT_BYTES} UTF-8 bytes"
        )
    return text


def _nonneg_int(
    value: Any,
    *,
    field_name: str,
    default: int = 0,
    maximum: int | None = None,
) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise IncrementalVerificationError(f"{field_name} must be an int")
    if value < 0:
        raise IncrementalVerificationError(f"{field_name} must be non-negative")
    if maximum is not None and value > maximum:
        raise IncrementalBoundsError(f"{field_name} exceeds {maximum}")
    return value


def _boolean(value: Any, *, field_name: str, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise IncrementalVerificationError(f"{field_name} must be a bool")
    return value


def _string_tuple(
    value: Any,
    *,
    field_name: str,
    maximum: int = MAX_UNITS,
    sort: bool = True,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = value
    else:
        raise IncrementalVerificationError(
            f"{field_name} must be a sequence of strings"
        )
    if len(items) > maximum:
        raise IncrementalBoundsError(f"{field_name} exceeds {maximum} items")
    result: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        text = _text(item, field_name=f"{field_name}[{index}]")
        if text not in seen:
            seen.add(text)
            result.append(text)
    if sort:
        result.sort()
    return tuple(result)


def _stable_unique(items: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


def _reason_codes(
    value: Any,
    *,
    field_name: str = "reason_codes",
    required: bool = False,
) -> tuple[str, ...]:
    if value is None:
        if required:
            raise IncrementalVerificationError(f"{field_name} is required")
        return ()
    if isinstance(value, str):
        items: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items = value
    else:
        raise IncrementalVerificationError(
            f"{field_name} must be a sequence of tokens"
        )
    if len(items) > MAX_REASON_CODES:
        raise IncrementalBoundsError(
            f"{field_name} exceeds {MAX_REASON_CODES} items"
        )
    return _stable_unique(
        tuple(_token(item, field_name=f"{field_name}[{i}]") for i, item in enumerate(items))
    )


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise IncrementalVerificationError(f"{field_name} must be a mapping")
    return value


def _structured_cid(schema: str, payload: Mapping[str, Any]) -> str:
    return content_identity({"schema": schema, "value": dict(payload)})


# ---------------------------------------------------------------------------
# Core records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CacheKeyBinding:
    """Exact-key material required for verification-cache reuse.

    Completeness is fail-closed: every field in
    :data:`REQUIRED_CACHE_KEY_FIELDS` must be a non-empty string.
    """

    key_cid: str
    repository_tree_cid: str
    semantic_state_root_cid: str
    environment_cid: str
    dependency_lock_cid: str
    kind: str
    check_id: str
    tool_name: str
    tool_version: str
    schema: str = CACHE_KEY_COMPLETENESS_SCHEMA

    def __post_init__(self) -> None:
        for name in REQUIRED_CACHE_KEY_FIELDS:
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name),
            )
        object.__setattr__(
            self,
            "kind",
            _token(self.kind, field_name="kind"),
        )
        object.__setattr__(
            self, "schema", _text(self.schema, field_name="schema")
        )

    @property
    def is_complete(self) -> bool:
        return all(
            bool(getattr(self, name).strip())
            for name in REQUIRED_CACHE_KEY_FIELDS
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "key_cid": self.key_cid,
            "repository_tree_cid": self.repository_tree_cid,
            "semantic_state_root_cid": self.semantic_state_root_cid,
            "environment_cid": self.environment_cid,
            "dependency_lock_cid": self.dependency_lock_cid,
            "kind": self.kind,
            "check_id": self.check_id,
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "complete": self.is_complete,
        }

    @classmethod
    def from_value(cls, value: Any) -> CacheKeyBinding | None:
        if value is None:
            return None
        if isinstance(value, CacheKeyBinding):
            return value
        if not isinstance(value, Mapping):
            raise IncrementalVerificationError(
                "cache_key must be a mapping or CacheKeyBinding"
            )
        missing = [
            name
            for name in REQUIRED_CACHE_KEY_FIELDS
            if not str(value.get(name) or "").strip()
        ]
        if missing:
            # Incomplete keys are representable only via the partial helper.
            return None
        return cls(
            key_cid=str(value["key_cid"]),
            repository_tree_cid=str(value["repository_tree_cid"]),
            semantic_state_root_cid=str(value["semantic_state_root_cid"]),
            environment_cid=str(value["environment_cid"]),
            dependency_lock_cid=str(value["dependency_lock_cid"]),
            kind=str(value["kind"]),
            check_id=str(value["check_id"]),
            tool_name=str(value["tool_name"]),
            tool_version=str(value["tool_version"]),
        )

    @classmethod
    def try_partial(cls, value: Any) -> tuple[CacheKeyBinding | None, tuple[str, ...]]:
        """Return ``(binding_or_none, missing_fields)`` without raising on gaps."""

        if value is None:
            return None, tuple(REQUIRED_CACHE_KEY_FIELDS)
        if isinstance(value, CacheKeyBinding):
            return value, ()
        if not isinstance(value, Mapping):
            raise IncrementalVerificationError(
                "cache_key must be a mapping or CacheKeyBinding"
            )
        missing = tuple(
            name
            for name in REQUIRED_CACHE_KEY_FIELDS
            if not str(value.get(name) or "").strip()
        )
        if missing:
            return None, missing
        return cls.from_value(value), ()


@dataclass(frozen=True, slots=True)
class ProofUnit:
    """One proof/test/static/type unit under incremental invalidation."""

    unit_id: str
    kind: UnitKind
    symbol_ids: tuple[str, ...] = ()
    path_ids: tuple[str, ...] = ()
    cache_key: CacheKeyBinding | None = None
    cached_receipt_cid: str = ""
    receipt_terminal: str = ""
    cpu_cost_ms: int = DEFAULT_UNIT_CPU_MS
    wall_cost_ms: int = DEFAULT_UNIT_WALL_MS
    schema: str = PROOF_UNIT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "unit_id", _text(self.unit_id, field_name="unit_id")
        )
        if isinstance(self.kind, UnitKind):
            kind = self.kind
        else:
            kind = UnitKind(_token(self.kind, field_name="kind"))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self,
            "symbol_ids",
            _string_tuple(self.symbol_ids, field_name="symbol_ids"),
        )
        object.__setattr__(
            self,
            "path_ids",
            _string_tuple(self.path_ids, field_name="path_ids"),
        )
        if self.cache_key is not None and not isinstance(
            self.cache_key, CacheKeyBinding
        ):
            binding, missing = CacheKeyBinding.try_partial(self.cache_key)
            if missing:
                object.__setattr__(self, "cache_key", None)
            else:
                object.__setattr__(self, "cache_key", binding)
        object.__setattr__(
            self,
            "cached_receipt_cid",
            _text(
                self.cached_receipt_cid,
                field_name="cached_receipt_cid",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "receipt_terminal",
            _text(
                self.receipt_terminal,
                field_name="receipt_terminal",
                required=False,
            ).lower(),
        )
        object.__setattr__(
            self,
            "cpu_cost_ms",
            _nonneg_int(self.cpu_cost_ms, field_name="cpu_cost_ms", default=DEFAULT_UNIT_CPU_MS),
        )
        object.__setattr__(
            self,
            "wall_cost_ms",
            _nonneg_int(
                self.wall_cost_ms, field_name="wall_cost_ms", default=DEFAULT_UNIT_WALL_MS
            ),
        )
        object.__setattr__(
            self, "schema", _text(self.schema, field_name="schema")
        )

    @property
    def has_complete_key(self) -> bool:
        return self.cache_key is not None and self.cache_key.is_complete

    def intersects(
        self,
        *,
        changed_symbols: Sequence[str],
        changed_paths: Sequence[str],
        affected_symbols: Sequence[str] = (),
        affected_paths: Sequence[str] = (),
    ) -> bool:
        symbols = set(changed_symbols) | set(affected_symbols)
        paths = set(changed_paths) | set(affected_paths)
        if symbols and set(self.symbol_ids) & symbols:
            return True
        if paths and set(self.path_ids) & paths:
            return True
        # Unit id itself may be a check/test node matching an affected set.
        if self.unit_id in symbols or self.unit_id in paths:
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "unit_id": self.unit_id,
            "kind": self.kind.value,
            "symbol_ids": self.symbol_ids,
            "path_ids": self.path_ids,
            "cache_key": None if self.cache_key is None else self.cache_key.to_dict(),
            "cached_receipt_cid": self.cached_receipt_cid,
            "receipt_terminal": self.receipt_terminal,
            "cpu_cost_ms": self.cpu_cost_ms,
            "wall_cost_ms": self.wall_cost_ms,
            "has_complete_key": self.has_complete_key,
        }

    @classmethod
    def from_value(cls, value: Any) -> ProofUnit:
        if isinstance(value, ProofUnit):
            return value
        if not isinstance(value, Mapping):
            raise IncrementalVerificationError(
                "proof unit must be a mapping or ProofUnit"
            )
        raw_key = value.get("cache_key")
        binding: CacheKeyBinding | None
        if raw_key is None:
            binding = None
        else:
            binding, _missing = CacheKeyBinding.try_partial(raw_key)
        return cls(
            unit_id=str(value.get("unit_id") or ""),
            kind=str(value.get("kind") or UnitKind.TEST.value),
            symbol_ids=tuple(value.get("symbol_ids") or ()),
            path_ids=tuple(value.get("path_ids") or ()),
            cache_key=binding,
            cached_receipt_cid=str(value.get("cached_receipt_cid") or ""),
            receipt_terminal=str(value.get("receipt_terminal") or ""),
            cpu_cost_ms=value.get("cpu_cost_ms", DEFAULT_UNIT_CPU_MS),
            wall_cost_ms=value.get("wall_cost_ms", DEFAULT_UNIT_WALL_MS),
        )


@dataclass(frozen=True, slots=True)
class UnitDecision:
    """Disposition for one unit after invalidation and cache evaluation."""

    unit_id: str
    disposition: UnitDisposition
    reason_codes: tuple[str, ...]
    affected: bool
    key_complete: bool
    reused_receipt_cid: str = ""
    cpu_cost_ms: int = 0
    wall_cost_ms: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "unit_id", _text(self.unit_id, field_name="unit_id")
        )
        if not isinstance(self.disposition, UnitDisposition):
            object.__setattr__(
                self,
                "disposition",
                UnitDisposition(_token(self.disposition, field_name="disposition")),
            )
        object.__setattr__(
            self,
            "reason_codes",
            _reason_codes(self.reason_codes, field_name="reason_codes", required=True),
        )
        object.__setattr__(
            self, "affected", _boolean(self.affected, field_name="affected")
        )
        object.__setattr__(
            self,
            "key_complete",
            _boolean(self.key_complete, field_name="key_complete"),
        )
        object.__setattr__(
            self,
            "reused_receipt_cid",
            _text(
                self.reused_receipt_cid,
                field_name="reused_receipt_cid",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "cpu_cost_ms",
            _nonneg_int(self.cpu_cost_ms, field_name="cpu_cost_ms"),
        )
        object.__setattr__(
            self,
            "wall_cost_ms",
            _nonneg_int(self.wall_cost_ms, field_name="wall_cost_ms"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "disposition": self.disposition.value,
            "reason_codes": self.reason_codes,
            "affected": self.affected,
            "key_complete": self.key_complete,
            "reused_receipt_cid": self.reused_receipt_cid,
            "cpu_cost_ms": self.cpu_cost_ms,
            "wall_cost_ms": self.wall_cost_ms,
        }


@dataclass(frozen=True, slots=True)
class SurvivorBroadeningPolicy:
    """Policy controlling how survivors expand beyond exact incremental checks."""

    schema: str = SURVIVOR_BROADENING_POLICY_SCHEMA
    broaden_survivors: bool = True
    full_suite_on_high_risk: bool = True
    full_suite_on_uncertainty: bool = True
    high_risk_classes: tuple[str, ...] = (
        "critical_security",
        "authorization",
        "proof_receipt_trust",
    )
    always_full_suite: bool = False
    max_broader_units: int = 10_000

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "schema", _text(self.schema, field_name="schema")
        )
        object.__setattr__(
            self,
            "broaden_survivors",
            _boolean(self.broaden_survivors, field_name="broaden_survivors"),
        )
        object.__setattr__(
            self,
            "full_suite_on_high_risk",
            _boolean(
                self.full_suite_on_high_risk, field_name="full_suite_on_high_risk"
            ),
        )
        object.__setattr__(
            self,
            "full_suite_on_uncertainty",
            _boolean(
                self.full_suite_on_uncertainty,
                field_name="full_suite_on_uncertainty",
            ),
        )
        object.__setattr__(
            self,
            "high_risk_classes",
            _string_tuple(self.high_risk_classes, field_name="high_risk_classes"),
        )
        object.__setattr__(
            self,
            "always_full_suite",
            _boolean(self.always_full_suite, field_name="always_full_suite"),
        )
        object.__setattr__(
            self,
            "max_broader_units",
            _nonneg_int(
                self.max_broader_units,
                field_name="max_broader_units",
                default=10_000,
                maximum=MAX_UNITS,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "broaden_survivors": self.broaden_survivors,
            "full_suite_on_high_risk": self.full_suite_on_high_risk,
            "full_suite_on_uncertainty": self.full_suite_on_uncertainty,
            "high_risk_classes": self.high_risk_classes,
            "always_full_suite": self.always_full_suite,
            "max_broader_units": self.max_broader_units,
        }

    @classmethod
    def from_value(cls, value: Any) -> SurvivorBroadeningPolicy:
        if value is None:
            return cls()
        if isinstance(value, SurvivorBroadeningPolicy):
            return value
        if not isinstance(value, Mapping):
            raise IncrementalVerificationError(
                "broadening_policy must be a mapping or SurvivorBroadeningPolicy"
            )
        return cls(
            broaden_survivors=value.get("broaden_survivors", True),
            full_suite_on_high_risk=value.get("full_suite_on_high_risk", True),
            full_suite_on_uncertainty=value.get("full_suite_on_uncertainty", True),
            high_risk_classes=tuple(value.get("high_risk_classes") or ()),
            always_full_suite=value.get("always_full_suite", False),
            max_broader_units=value.get("max_broader_units", 10_000),
        )


@dataclass(frozen=True, slots=True)
class TemporaryProofForest:
    """Ephemeral proof forest for one mutant; never a canonical seal.

    The forest may track unit proof CIDs and a derived forest root for the
    disposable mutant worktree.  Promotion or replacement of a repository
    canonical seal is always refused.
    """

    forest_id: str
    mutant_id: str
    repository_tree_cid: str
    unit_proof_cids: Mapping[str, str] = field(default_factory=dict)
    parent_canonical_seal_cid: str = ""
    is_canonical: bool = False
    is_temporary: bool = True
    schema: str = TEMPORARY_PROOF_FOREST_SCHEMA
    interface_id: str = TEMPORARY_PROOF_FOREST_INTERFACE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "forest_id", _text(self.forest_id, field_name="forest_id")
        )
        object.__setattr__(
            self, "mutant_id", _text(self.mutant_id, field_name="mutant_id")
        )
        object.__setattr__(
            self,
            "repository_tree_cid",
            _text(self.repository_tree_cid, field_name="repository_tree_cid"),
        )
        units: dict[str, str] = {}
        raw_units = self.unit_proof_cids or {}
        if not isinstance(raw_units, Mapping):
            raise IncrementalVerificationError(
                "unit_proof_cids must be a mapping"
            )
        if len(raw_units) > MAX_UNITS:
            raise IncrementalBoundsError(
                f"unit_proof_cids exceeds {MAX_UNITS} items"
            )
        for key, value in raw_units.items():
            unit_id = _text(key, field_name="unit_proof_cids.key")
            proof_cid = _text(value, field_name=f"unit_proof_cids[{unit_id}]")
            units[unit_id] = proof_cid
        object.__setattr__(
            self, "unit_proof_cids", MappingProxyType(dict(sorted(units.items())))
        )
        object.__setattr__(
            self,
            "parent_canonical_seal_cid",
            _text(
                self.parent_canonical_seal_cid,
                field_name="parent_canonical_seal_cid",
                required=False,
            ),
        )
        # Hard invariants: temporary forests are never canonical.
        object.__setattr__(self, "is_canonical", False)
        object.__setattr__(self, "is_temporary", True)
        object.__setattr__(
            self, "schema", _text(self.schema, field_name="schema")
        )
        object.__setattr__(
            self,
            "interface_id",
            _text(self.interface_id, field_name="interface_id"),
        )

    @property
    def forest_root_cid(self) -> str:
        return _structured_cid(
            TEMPORARY_PROOF_FOREST_SCHEMA,
            {
                "forest_id": self.forest_id,
                "mutant_id": self.mutant_id,
                "repository_tree_cid": self.repository_tree_cid,
                "unit_proof_cids": dict(self.unit_proof_cids),
                "is_temporary": True,
                "is_canonical": False,
            },
        )

    def with_units(self, unit_proof_cids: Mapping[str, str]) -> TemporaryProofForest:
        merged = dict(self.unit_proof_cids)
        for key, value in unit_proof_cids.items():
            merged[_text(key, field_name="unit_id")] = _text(
                value, field_name="proof_cid"
            )
        return TemporaryProofForest(
            forest_id=self.forest_id,
            mutant_id=self.mutant_id,
            repository_tree_cid=self.repository_tree_cid,
            unit_proof_cids=merged,
            parent_canonical_seal_cid=self.parent_canonical_seal_cid,
        )

    def replace_canonical_seal(self, canonical_seal_cid: str) -> None:
        """Always refuse: temporary forests never replace canonical seals."""

        _ = _text(canonical_seal_cid, field_name="canonical_seal_cid")
        raise CanonicalSealProtectionError(
            "temporary proof forest must never replace a canonical seal",
            reason_code=REASON_CANONICAL_REPLACE_REFUSED,
        )

    def publish_as_canonical(self) -> None:
        """Always refuse: temporary forests are never production seals."""

        raise CanonicalSealProtectionError(
            "temporary proof forest must never publish as a canonical seal",
            reason_code=REASON_CANONICAL_REPLACE_REFUSED,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface_id": self.interface_id,
            "forest_id": self.forest_id,
            "mutant_id": self.mutant_id,
            "repository_tree_cid": self.repository_tree_cid,
            "unit_proof_cids": dict(self.unit_proof_cids),
            "parent_canonical_seal_cid": self.parent_canonical_seal_cid,
            "forest_root_cid": self.forest_root_cid,
            "is_canonical": False,
            "is_temporary": True,
        }


@dataclass(frozen=True, slots=True)
class MutationCostAccounting:
    """Measured full versus incremental costs and cache reuse for one mutant."""

    schema: str = MUTATION_COST_ACCOUNTING_SCHEMA
    full: ProofCostRecord | None = None
    incremental: ProofCostRecord | None = None
    comparison: ProofCostComparison | None = None
    units_total: int = 0
    units_invalidated: int = 0
    units_reused: int = 0
    units_reverified: int = 0
    units_broadened: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    incomplete_key_rejections: int = 0
    full_cpu_ms: int = 0
    full_wall_ms: int = 0
    incremental_cpu_ms: int = 0
    incremental_wall_ms: int = 0
    compute_saved_cpu_ms: int | None = None
    compute_saved_wall_ms: int | None = None
    measured: bool = False
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "schema", _text(self.schema, field_name="schema")
        )
        for name in (
            "units_total",
            "units_invalidated",
            "units_reused",
            "units_reverified",
            "units_broadened",
            "cache_hits",
            "cache_misses",
            "incomplete_key_rejections",
            "full_cpu_ms",
            "full_wall_ms",
            "incremental_cpu_ms",
            "incremental_wall_ms",
        ):
            object.__setattr__(
                self,
                name,
                _nonneg_int(getattr(self, name), field_name=name),
            )
        for name in ("compute_saved_cpu_ms", "compute_saved_wall_ms"):
            raw = getattr(self, name)
            if raw is not None:
                object.__setattr__(
                    self,
                    name,
                    _nonneg_int(raw, field_name=name),
                )
        object.__setattr__(
            self, "measured", _boolean(self.measured, field_name="measured")
        )
        object.__setattr__(
            self,
            "reason_codes",
            _reason_codes(self.reason_codes, field_name="reason_codes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "full": None if self.full is None else self.full.to_canonical(),
            "incremental": (
                None
                if self.incremental is None
                else self.incremental.to_canonical()
            ),
            "comparison": (
                None
                if self.comparison is None
                else self.comparison.to_canonical()
            ),
            "units_total": self.units_total,
            "units_invalidated": self.units_invalidated,
            "units_reused": self.units_reused,
            "units_reverified": self.units_reverified,
            "units_broadened": self.units_broadened,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "incomplete_key_rejections": self.incomplete_key_rejections,
            "full_cpu_ms": self.full_cpu_ms,
            "full_wall_ms": self.full_wall_ms,
            "incremental_cpu_ms": self.incremental_cpu_ms,
            "incremental_wall_ms": self.incremental_wall_ms,
            "compute_saved_cpu_ms": self.compute_saved_cpu_ms,
            "compute_saved_wall_ms": self.compute_saved_wall_ms,
            "measured": self.measured,
            "reason_codes": self.reason_codes,
        }


@dataclass(frozen=True, slots=True)
class IncrementalMutationVerificationResult:
    """Sealed result of incremental verification for one mutant."""

    schema: str = INCREMENTAL_MUTATION_RESULT_SCHEMA
    interface_id: str = INCREMENTAL_MUTATION_RESULT_INTERFACE
    result_cid: str = ""
    mutant_id: str = ""
    repository_tree_cid: str = ""
    decisions: tuple[UnitDecision, ...] = ()
    temporary_forest: TemporaryProofForest | None = None
    cost_accounting: MutationCostAccounting | None = None
    broadening_mode: BroadeningMode = BroadeningMode.NONE
    mutant_outcome: MutantOutcomeClass = MutantOutcomeClass.INCONCLUSIVE
    selected_unit_ids: tuple[str, ...] = ()
    broadened_unit_ids: tuple[str, ...] = ()
    full_suite_unit_ids: tuple[str, ...] = ()
    invalidated_unit_ids: tuple[str, ...] = ()
    reused_unit_ids: tuple[str, ...] = ()
    canonical_seal_cid: str = ""
    canonical_seal_replaced: bool = False
    production_policy_changed: bool = False
    reason_codes: tuple[str, ...] = ()
    evidence_subset: str = AAE_INCREMENTAL_EVIDENCE

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "schema", _text(self.schema, field_name="schema")
        )
        object.__setattr__(
            self,
            "interface_id",
            _text(self.interface_id, field_name="interface_id"),
        )
        object.__setattr__(
            self, "mutant_id", _text(self.mutant_id, field_name="mutant_id")
        )
        object.__setattr__(
            self,
            "repository_tree_cid",
            _text(self.repository_tree_cid, field_name="repository_tree_cid"),
        )
        decisions = tuple(self.decisions or ())
        if len(decisions) > MAX_UNITS:
            raise IncrementalBoundsError(
                f"decisions exceeds {MAX_UNITS} items"
            )
        normalized: list[UnitDecision] = []
        for item in decisions:
            if not isinstance(item, UnitDecision):
                raise IncrementalVerificationError(
                    "decisions must contain UnitDecision records"
                )
            normalized.append(item)
        object.__setattr__(self, "decisions", tuple(normalized))
        if self.temporary_forest is not None and not isinstance(
            self.temporary_forest, TemporaryProofForest
        ):
            raise IncrementalVerificationError(
                "temporary_forest must be a TemporaryProofForest"
            )
        if self.cost_accounting is not None and not isinstance(
            self.cost_accounting, MutationCostAccounting
        ):
            raise IncrementalVerificationError(
                "cost_accounting must be a MutationCostAccounting"
            )
        if not isinstance(self.broadening_mode, BroadeningMode):
            object.__setattr__(
                self,
                "broadening_mode",
                BroadeningMode(
                    _token(self.broadening_mode, field_name="broadening_mode")
                ),
            )
        if not isinstance(self.mutant_outcome, MutantOutcomeClass):
            object.__setattr__(
                self,
                "mutant_outcome",
                MutantOutcomeClass(
                    _token(self.mutant_outcome, field_name="mutant_outcome")
                ),
            )
        for name in (
            "selected_unit_ids",
            "broadened_unit_ids",
            "full_suite_unit_ids",
            "invalidated_unit_ids",
            "reused_unit_ids",
        ):
            object.__setattr__(
                self,
                name,
                _string_tuple(getattr(self, name), field_name=name),
            )
        object.__setattr__(
            self,
            "canonical_seal_cid",
            _text(
                self.canonical_seal_cid,
                field_name="canonical_seal_cid",
                required=False,
            ),
        )
        # Hard invariant: temporary forests never replace canonical seals.
        object.__setattr__(self, "canonical_seal_replaced", False)
        object.__setattr__(self, "production_policy_changed", False)
        object.__setattr__(
            self,
            "reason_codes",
            _reason_codes(self.reason_codes, field_name="reason_codes"),
        )
        object.__setattr__(
            self,
            "evidence_subset",
            _text(self.evidence_subset, field_name="evidence_subset"),
        )
        if not self.result_cid:
            object.__setattr__(self, "result_cid", self.compute_result_cid())

    def compute_result_cid(self) -> str:
        payload = {
            "schema": self.schema,
            "interface_id": self.interface_id,
            "mutant_id": self.mutant_id,
            "repository_tree_cid": self.repository_tree_cid,
            "decisions": [item.to_dict() for item in self.decisions],
            "temporary_forest": (
                None
                if self.temporary_forest is None
                else self.temporary_forest.to_dict()
            ),
            "cost_accounting": (
                None
                if self.cost_accounting is None
                else self.cost_accounting.to_dict()
            ),
            "broadening_mode": self.broadening_mode.value,
            "mutant_outcome": self.mutant_outcome.value,
            "selected_unit_ids": self.selected_unit_ids,
            "broadened_unit_ids": self.broadened_unit_ids,
            "full_suite_unit_ids": self.full_suite_unit_ids,
            "invalidated_unit_ids": self.invalidated_unit_ids,
            "reused_unit_ids": self.reused_unit_ids,
            "canonical_seal_cid": self.canonical_seal_cid,
            "canonical_seal_replaced": False,
            "production_policy_changed": False,
            "reason_codes": self.reason_codes,
            "evidence_subset": self.evidence_subset,
        }
        return _structured_cid(INCREMENTAL_MUTATION_RESULT_SCHEMA, payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface_id": self.interface_id,
            "result_cid": self.result_cid,
            "mutant_id": self.mutant_id,
            "repository_tree_cid": self.repository_tree_cid,
            "decisions": [item.to_dict() for item in self.decisions],
            "temporary_forest": (
                None
                if self.temporary_forest is None
                else self.temporary_forest.to_dict()
            ),
            "cost_accounting": (
                None
                if self.cost_accounting is None
                else self.cost_accounting.to_dict()
            ),
            "broadening_mode": self.broadening_mode.value,
            "mutant_outcome": self.mutant_outcome.value,
            "selected_unit_ids": self.selected_unit_ids,
            "broadened_unit_ids": self.broadened_unit_ids,
            "full_suite_unit_ids": self.full_suite_unit_ids,
            "invalidated_unit_ids": self.invalidated_unit_ids,
            "reused_unit_ids": self.reused_unit_ids,
            "canonical_seal_cid": self.canonical_seal_cid,
            "canonical_seal_replaced": False,
            "production_policy_changed": False,
            "reason_codes": self.reason_codes,
            "evidence_subset": self.evidence_subset,
        }


# ---------------------------------------------------------------------------
# Invalidation, reuse, broadening, cost helpers
# ---------------------------------------------------------------------------


def _production_success_terminals() -> frozenset[str]:
    return frozenset({"passed", "proved"})


def classify_unit_invalidation(
    units: Sequence[ProofUnit | Mapping[str, Any]],
    *,
    changed_symbols: Sequence[str] = (),
    changed_paths: Sequence[str] = (),
    affected_symbols: Sequence[str] = (),
    affected_paths: Sequence[str] = (),
    affected_unit_ids: Sequence[str] = (),
) -> tuple[UnitDecision, ...]:
    """Mark only affected units as invalidated; leave others unaffected."""

    symbols = _string_tuple(changed_symbols, field_name="changed_symbols")
    paths = _string_tuple(changed_paths, field_name="changed_paths")
    aff_symbols = _string_tuple(affected_symbols, field_name="affected_symbols")
    aff_paths = _string_tuple(affected_paths, field_name="affected_paths")
    aff_units = set(
        _string_tuple(affected_unit_ids, field_name="affected_unit_ids")
    )

    if len(units) > MAX_UNITS:
        raise IncrementalBoundsError(f"units exceeds {MAX_UNITS} items")

    decisions: list[UnitDecision] = []
    for raw in units:
        unit = ProofUnit.from_value(raw)
        affected = unit.unit_id in aff_units or unit.intersects(
            changed_symbols=symbols,
            changed_paths=paths,
            affected_symbols=aff_symbols,
            affected_paths=aff_paths,
        )
        if affected:
            decisions.append(
                UnitDecision(
                    unit_id=unit.unit_id,
                    disposition=UnitDisposition.INVALIDATED,
                    reason_codes=(REASON_AFFECTED_INVALIDATED,),
                    affected=True,
                    key_complete=unit.has_complete_key,
                    cpu_cost_ms=unit.cpu_cost_ms,
                    wall_cost_ms=unit.wall_cost_ms,
                )
            )
        else:
            decisions.append(
                UnitDecision(
                    unit_id=unit.unit_id,
                    disposition=UnitDisposition.UNAFFECTED,
                    reason_codes=(
                        REASON_UNAFFECTED_REUSE_CANDIDATE,
                        REASON_UNRELATED_NOT_INVALIDATED,
                    ),
                    affected=False,
                    key_complete=unit.has_complete_key,
                    cpu_cost_ms=0,
                    wall_cost_ms=0,
                )
            )
    decisions.sort(key=lambda item: item.unit_id)
    return tuple(decisions)


def evaluate_cache_reuse(
    units: Sequence[ProofUnit | Mapping[str, Any]],
    invalidation: Sequence[UnitDecision],
    *,
    require_production_terminal: bool = True,
) -> tuple[UnitDecision, ...]:
    """Allow reuse only when the unit is unaffected and the key is complete."""

    unit_by_id: dict[str, ProofUnit] = {}
    for raw in units:
        unit = ProofUnit.from_value(raw)
        unit_by_id[unit.unit_id] = unit

    success = _production_success_terminals()
    out: list[UnitDecision] = []
    for decision in invalidation:
        unit = unit_by_id.get(decision.unit_id)
        if unit is None:
            raise IncrementalVerificationError(
                f"invalidation references unknown unit {decision.unit_id!r}"
            )
        if decision.disposition is UnitDisposition.INVALIDATED:
            out.append(
                UnitDecision(
                    unit_id=unit.unit_id,
                    disposition=UnitDisposition.REVERIFY,
                    reason_codes=_stable_unique(
                        (
                            *decision.reason_codes,
                            REASON_AFFECTED_INVALIDATED,
                            REASON_CACHE_MISS,
                        )
                    ),
                    affected=True,
                    key_complete=unit.has_complete_key,
                    cpu_cost_ms=unit.cpu_cost_ms,
                    wall_cost_ms=unit.wall_cost_ms,
                )
            )
            continue

        # Unaffected path: reuse requires complete keys + successful terminal.
        if not unit.has_complete_key:
            out.append(
                UnitDecision(
                    unit_id=unit.unit_id,
                    disposition=UnitDisposition.REVERIFY,
                    reason_codes=(
                        REASON_UNAFFECTED_REUSE_CANDIDATE,
                        REASON_INCOMPLETE_KEY_REJECTED
                        if unit.cache_key is None
                        or not unit.has_complete_key
                        else REASON_MISSING_KEY_REJECTED,
                    ),
                    affected=False,
                    key_complete=False,
                    cpu_cost_ms=unit.cpu_cost_ms,
                    wall_cost_ms=unit.wall_cost_ms,
                )
            )
            continue

        terminal = (unit.receipt_terminal or "").lower()
        if not unit.cached_receipt_cid:
            out.append(
                UnitDecision(
                    unit_id=unit.unit_id,
                    disposition=UnitDisposition.REVERIFY,
                    reason_codes=(
                        REASON_UNAFFECTED_REUSE_CANDIDATE,
                        REASON_CACHE_MISS,
                    ),
                    affected=False,
                    key_complete=True,
                    cpu_cost_ms=unit.cpu_cost_ms,
                    wall_cost_ms=unit.wall_cost_ms,
                )
            )
            continue

        if require_production_terminal and terminal not in success:
            reason = (
                REASON_STALE_RECEIPT
                if terminal in {"stale", "simulated", "invalid"}
                else REASON_CACHE_MISS
            )
            out.append(
                UnitDecision(
                    unit_id=unit.unit_id,
                    disposition=UnitDisposition.REVERIFY,
                    reason_codes=(
                        REASON_UNAFFECTED_REUSE_CANDIDATE,
                        reason,
                    ),
                    affected=False,
                    key_complete=True,
                    cpu_cost_ms=unit.cpu_cost_ms,
                    wall_cost_ms=unit.wall_cost_ms,
                )
            )
            continue

        out.append(
            UnitDecision(
                unit_id=unit.unit_id,
                disposition=UnitDisposition.REUSED,
                reason_codes=(
                    REASON_UNAFFECTED_REUSE_CANDIDATE,
                    REASON_COMPLETE_KEY_REUSED,
                ),
                affected=False,
                key_complete=True,
                reused_receipt_cid=unit.cached_receipt_cid,
                cpu_cost_ms=DEFAULT_REUSE_CPU_MS,
                wall_cost_ms=DEFAULT_REUSE_WALL_MS,
            )
        )
    out.sort(key=lambda item: item.unit_id)
    return tuple(out)


def resolve_broadening_mode(
    *,
    mutant_outcome: MutantOutcomeClass | str,
    risk_class: str = "",
    uncertainty: bool = False,
    policy: SurvivorBroadeningPolicy | Mapping[str, Any] | None = None,
) -> tuple[BroadeningMode, tuple[str, ...]]:
    """Decide broader/full-suite expansion for survivors under policy."""

    policy_obj = SurvivorBroadeningPolicy.from_value(policy)
    if isinstance(mutant_outcome, MutantOutcomeClass):
        outcome = mutant_outcome
    else:
        outcome = MutantOutcomeClass(
            _token(mutant_outcome, field_name="mutant_outcome")
        )
    risk = _text(risk_class, field_name="risk_class", required=False).lower()
    reasons: list[str] = []

    if policy_obj.always_full_suite:
        return BroadeningMode.FULL_SUITE, (REASON_HIGH_RISK_FULL_SUITE,)

    high_risk = bool(risk) and risk in set(policy_obj.high_risk_classes)
    if high_risk and policy_obj.full_suite_on_high_risk:
        return BroadeningMode.FULL_SUITE, (REASON_HIGH_RISK_FULL_SUITE,)

    if uncertainty and policy_obj.full_suite_on_uncertainty:
        return BroadeningMode.FULL_SUITE, (REASON_UNCERTAINTY_BROADENED,)

    if outcome is MutantOutcomeClass.SURVIVOR:
        if not policy_obj.broaden_survivors:
            return BroadeningMode.NONE, (REASON_BROADENING_DISABLED,)
        reasons.append(REASON_SURVIVOR_BROADENED)
        return BroadeningMode.BROADER, tuple(reasons)

    return BroadeningMode.NONE, ()


def broaden_unit_decisions(
    decisions: Sequence[UnitDecision],
    units: Sequence[ProofUnit | Mapping[str, Any]],
    *,
    mode: BroadeningMode,
    catalog_unit_ids: Sequence[str] = (),
    max_broader_units: int = 10_000,
) -> tuple[UnitDecision, ...]:
    """Expand survivor selection to broader or full-suite unit sets."""

    if mode is BroadeningMode.NONE:
        return tuple(sorted(decisions, key=lambda item: item.unit_id))

    unit_by_id: dict[str, ProofUnit] = {}
    for raw in units:
        unit = ProofUnit.from_value(raw)
        unit_by_id[unit.unit_id] = unit
    by_id = {item.unit_id: item for item in decisions}
    catalog = list(
        _string_tuple(catalog_unit_ids, field_name="catalog_unit_ids", sort=True)
    )
    if not catalog:
        catalog = sorted(unit_by_id.keys())

    if mode is BroadeningMode.FULL_SUITE:
        target_ids = catalog
        disposition = UnitDisposition.FULL_SUITE
        reason = REASON_SURVIVOR_FULL_SUITE
    else:
        # Broader: include all catalog units not already reused as free hits,
        # capped by policy.
        target_ids = catalog[: max(0, int(max_broader_units))]
        disposition = UnitDisposition.BROADENED
        reason = REASON_SURVIVOR_BROADENED

    out: dict[str, UnitDecision] = dict(by_id)
    for unit_id in target_ids:
        existing = out.get(unit_id)
        unit = unit_by_id.get(unit_id)
        cpu = unit.cpu_cost_ms if unit is not None else DEFAULT_UNIT_CPU_MS
        wall = unit.wall_cost_ms if unit is not None else DEFAULT_UNIT_WALL_MS
        if existing is not None and existing.disposition is UnitDisposition.REUSED:
            # Keep free reuse; do not force re-execution of proven-unaffected
            # complete-key units unless full suite explicitly re-includes them
            # as already-paid reuse (still reused).
            continue
        if existing is not None and existing.disposition in {
            UnitDisposition.REVERIFY,
            UnitDisposition.INVALIDATED,
            UnitDisposition.BROADENED,
            UnitDisposition.FULL_SUITE,
        }:
            # Already scheduled for work; upgrade disposition label only.
            out[unit_id] = UnitDecision(
                unit_id=unit_id,
                disposition=disposition,
                reason_codes=_stable_unique((*existing.reason_codes, reason)),
                affected=existing.affected,
                key_complete=existing.key_complete,
                reused_receipt_cid=existing.reused_receipt_cid,
                cpu_cost_ms=existing.cpu_cost_ms or cpu,
                wall_cost_ms=existing.wall_cost_ms or wall,
            )
            continue
        out[unit_id] = UnitDecision(
            unit_id=unit_id,
            disposition=disposition,
            reason_codes=(reason,),
            affected=existing.affected if existing is not None else False,
            key_complete=existing.key_complete if existing is not None else False,
            cpu_cost_ms=cpu,
            wall_cost_ms=wall,
        )
    return tuple(sorted(out.values(), key=lambda item: item.unit_id))


def measure_mutation_costs(
    units: Sequence[ProofUnit | Mapping[str, Any]],
    decisions: Sequence[UnitDecision],
) -> MutationCostAccounting:
    """Measure full-suite counterfactual cost and incremental executed cost."""

    unit_by_id: dict[str, ProofUnit] = {}
    for raw in units:
        unit = ProofUnit.from_value(raw)
        unit_by_id[unit.unit_id] = unit

    full_cpu = 0
    full_wall = 0
    for unit in unit_by_id.values():
        full_cpu += unit.cpu_cost_ms
        full_wall += unit.wall_cost_ms

    inc_cpu = 0
    inc_wall = 0
    invalidated = 0
    reused = 0
    reverified = 0
    broadened = 0
    cache_hits = 0
    cache_misses = 0
    incomplete = 0

    for decision in decisions:
        if decision.disposition is UnitDisposition.REUSED:
            reused += 1
            cache_hits += 1
            inc_cpu += decision.cpu_cost_ms
            inc_wall += decision.wall_cost_ms
        elif decision.disposition is UnitDisposition.INVALIDATED:
            invalidated += 1
            reverified += 1
            cache_misses += 1
            inc_cpu += decision.cpu_cost_ms
            inc_wall += decision.wall_cost_ms
        elif decision.disposition is UnitDisposition.REVERIFY:
            reverified += 1
            if not decision.key_complete and not decision.affected:
                incomplete += 1
            else:
                cache_misses += 1
            if decision.affected:
                invalidated += 1
            inc_cpu += decision.cpu_cost_ms
            inc_wall += decision.wall_cost_ms
        elif decision.disposition is UnitDisposition.BROADENED:
            broadened += 1
            reverified += 1
            inc_cpu += decision.cpu_cost_ms
            inc_wall += decision.wall_cost_ms
        elif decision.disposition is UnitDisposition.FULL_SUITE:
            broadened += 1
            reverified += 1
            inc_cpu += decision.cpu_cost_ms
            inc_wall += decision.wall_cost_ms
        elif decision.disposition is UnitDisposition.UNAFFECTED:
            # Not yet evaluated for reuse; treat as unpaid until reuse stage.
            pass

    full_collector = ProofMetricsCollector(estimated=False)
    full_collector.record_units(
        required=len(unit_by_id),
        reused=0,
        invalidated=len(unit_by_id),
        proved=0,
        cache_hits=0,
    )
    full_collector.observe_cpu_ms(full_cpu)
    full_collector.observe_wall_ms(full_wall)
    full_record = full_collector.snapshot()

    inc_collector = ProofMetricsCollector(estimated=False)
    inc_collector.record_units(
        required=len(unit_by_id),
        reused=reused,
        invalidated=invalidated,
        proved=0,
        cache_hits=cache_hits,
    )
    inc_collector.observe_cpu_ms(inc_cpu)
    inc_collector.observe_wall_ms(inc_wall)
    inc_record = inc_collector.snapshot()

    comparison = compare_costs(full_record, inc_record)
    # Direct measured savings from observed unit costs.  ProofCostComparison
    # may leave savings unknown when optional counters (e.g. storage) were not
    # observed; unit-level CPU/wall are always measured here.
    saved_cpu = max(0, full_cpu - inc_cpu)
    saved_wall = max(0, full_wall - inc_wall)

    return MutationCostAccounting(
        full=full_record,
        incremental=inc_record,
        comparison=comparison,
        units_total=len(unit_by_id),
        units_invalidated=invalidated,
        units_reused=reused,
        units_reverified=reverified,
        units_broadened=broadened,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        incomplete_key_rejections=incomplete,
        full_cpu_ms=full_cpu,
        full_wall_ms=full_wall,
        incremental_cpu_ms=inc_cpu,
        incremental_wall_ms=inc_wall,
        compute_saved_cpu_ms=saved_cpu,
        compute_saved_wall_ms=saved_wall,
        measured=True,
        reason_codes=(
            REASON_COSTS_MEASURED,
            REASON_CACHE_REUSE_MEASURED,
        ),
    )


def build_temporary_proof_forest(
    *,
    mutant_id: str,
    repository_tree_cid: str,
    decisions: Sequence[UnitDecision],
    units: Sequence[ProofUnit | Mapping[str, Any]] = (),
    parent_canonical_seal_cid: str = "",
    forest_id: str = "",
) -> TemporaryProofForest:
    """Build a temporary forest from unit decisions; never a canonical seal."""

    unit_receipts: dict[str, str] = {}
    for raw in units:
        unit = ProofUnit.from_value(raw)
        if unit.cached_receipt_cid:
            unit_receipts[unit.unit_id] = unit.cached_receipt_cid
    proofs: dict[str, str] = {}
    for decision in decisions:
        if decision.disposition is UnitDisposition.REUSED and decision.reused_receipt_cid:
            proofs[decision.unit_id] = decision.reused_receipt_cid
            continue
        if decision.disposition in {
            UnitDisposition.REVERIFY,
            UnitDisposition.INVALIDATED,
            UnitDisposition.BROADENED,
            UnitDisposition.FULL_SUITE,
        }:
            # Fresh provisional proof object for the temporary forest only.
            # Prefer binding any known unit receipt identity into the
            # provisional material for stable forest roots across retries.
            prior = unit_receipts.get(decision.unit_id, "")
            proofs[decision.unit_id] = _structured_cid(
                PROOF_UNIT_SCHEMA,
                {
                    "unit_id": decision.unit_id,
                    "mutant_id": mutant_id,
                    "disposition": decision.disposition.value,
                    "prior_receipt_cid": prior,
                    "provisional": True,
                },
            )
    resolved_forest_id = forest_id or _structured_cid(
        TEMPORARY_PROOF_FOREST_SCHEMA,
        {
            "mutant_id": mutant_id,
            "repository_tree_cid": repository_tree_cid,
            "kind": "temporary",
        },
    )
    return TemporaryProofForest(
        forest_id=resolved_forest_id,
        mutant_id=mutant_id,
        repository_tree_cid=repository_tree_cid,
        unit_proof_cids=proofs,
        parent_canonical_seal_cid=parent_canonical_seal_cid,
    )


# ---------------------------------------------------------------------------
# Public verifier
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IncrementalMutationVerifier:
    """``IncrementalMutationVerifier@1`` composition authority."""

    interface_id: str = INCREMENTAL_MUTATION_VERIFIER_INTERFACE
    schema: str = INCREMENTAL_MUTATION_VERIFIER_SCHEMA
    evidence_subset: str = AAE_INCREMENTAL_EVIDENCE
    adapter_id: str = ADAPTER_ID
    broadening_policy: SurvivorBroadeningPolicy = field(
        default_factory=SurvivorBroadeningPolicy
    )
    require_production_terminal: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "interface_id",
            _text(self.interface_id, field_name="interface_id"),
        )
        if self.interface_id != INCREMENTAL_MUTATION_VERIFIER_INTERFACE:
            raise IncrementalVerificationError(
                "interface_id must be IncrementalMutationVerifier@1"
            )
        object.__setattr__(
            self, "schema", _text(self.schema, field_name="schema")
        )
        object.__setattr__(
            self,
            "evidence_subset",
            _text(self.evidence_subset, field_name="evidence_subset"),
        )
        object.__setattr__(
            self, "adapter_id", _text(self.adapter_id, field_name="adapter_id")
        )
        object.__setattr__(
            self,
            "broadening_policy",
            SurvivorBroadeningPolicy.from_value(self.broadening_policy),
        )
        object.__setattr__(
            self,
            "require_production_terminal",
            _boolean(
                self.require_production_terminal,
                field_name="require_production_terminal",
            ),
        )

    def invalidate_units(
        self,
        units: Sequence[ProofUnit | Mapping[str, Any]],
        *,
        changed_symbols: Sequence[str] = (),
        changed_paths: Sequence[str] = (),
        affected_symbols: Sequence[str] = (),
        affected_paths: Sequence[str] = (),
        affected_unit_ids: Sequence[str] = (),
    ) -> tuple[UnitDecision, ...]:
        return classify_unit_invalidation(
            units,
            changed_symbols=changed_symbols,
            changed_paths=changed_paths,
            affected_symbols=affected_symbols,
            affected_paths=affected_paths,
            affected_unit_ids=affected_unit_ids,
        )

    def evaluate_reuse(
        self,
        units: Sequence[ProofUnit | Mapping[str, Any]],
        invalidation: Sequence[UnitDecision],
    ) -> tuple[UnitDecision, ...]:
        return evaluate_cache_reuse(
            units,
            invalidation,
            require_production_terminal=self.require_production_terminal,
        )

    def broaden_survivors(
        self,
        decisions: Sequence[UnitDecision],
        units: Sequence[ProofUnit | Mapping[str, Any]],
        *,
        mutant_outcome: MutantOutcomeClass | str,
        risk_class: str = "",
        uncertainty: bool = False,
        catalog_unit_ids: Sequence[str] = (),
    ) -> tuple[BroadeningMode, tuple[UnitDecision, ...], tuple[str, ...]]:
        mode, reasons = resolve_broadening_mode(
            mutant_outcome=mutant_outcome,
            risk_class=risk_class,
            uncertainty=uncertainty,
            policy=self.broadening_policy,
        )
        broadened = broaden_unit_decisions(
            decisions,
            units,
            mode=mode,
            catalog_unit_ids=catalog_unit_ids,
            max_broader_units=self.broadening_policy.max_broader_units,
        )
        return mode, broadened, reasons

    def measure_costs(
        self,
        units: Sequence[ProofUnit | Mapping[str, Any]],
        decisions: Sequence[UnitDecision],
    ) -> MutationCostAccounting:
        return measure_mutation_costs(units, decisions)

    def verify_mutant(
        self,
        *,
        mutant_id: str,
        repository_tree_cid: str,
        units: Sequence[ProofUnit | Mapping[str, Any]],
        changed_symbols: Sequence[str] = (),
        changed_paths: Sequence[str] = (),
        affected_symbols: Sequence[str] = (),
        affected_paths: Sequence[str] = (),
        affected_unit_ids: Sequence[str] = (),
        mutant_outcome: MutantOutcomeClass | str = MutantOutcomeClass.KILLED,
        risk_class: str = "",
        uncertainty: bool = False,
        catalog_unit_ids: Sequence[str] = (),
        canonical_seal_cid: str = "",
        parent_canonical_seal_cid: str = "",
        # Optional semantic selection integration inputs.
        edges: Sequence[Any] | None = None,
        selection_policy: SelectionPolicy | Mapping[str, Any] | None = None,
        catalog: VerificationCatalog | Mapping[str, Any] | None = None,
    ) -> IncrementalMutationVerificationResult:
        """Run the full incremental verification pipeline for one mutant."""

        mutant = _text(mutant_id, field_name="mutant_id")
        tree = _text(repository_tree_cid, field_name="repository_tree_cid")
        unit_list = [ProofUnit.from_value(item) for item in units]
        if not unit_list:
            raise IncrementalVerificationError("units must not be empty")

        # Optional: expand affected unit ids from semantic selection when edges
        # / catalog are supplied.  Selection failures fail closed.
        resolved_affected_units = list(
            _string_tuple(affected_unit_ids, field_name="affected_unit_ids")
        )
        resolved_affected_symbols = list(
            _string_tuple(affected_symbols, field_name="affected_symbols")
        )
        resolved_affected_paths = list(
            _string_tuple(affected_paths, field_name="affected_paths")
        )
        if edges is not None or catalog is not None:
            selection = select_affected_verification(
                changed_symbols=changed_symbols,
                changed_paths=changed_paths,
                edges=edges,
                catalog=catalog,
                policy=selection_policy,
                known_tests=tuple(
                    u.unit_id for u in unit_list if u.kind is UnitKind.TEST
                ),
                known_static_checks=tuple(
                    u.unit_id
                    for u in unit_list
                    if u.kind is UnitKind.STATIC_ANALYSIS
                ),
                known_type_checks=tuple(
                    u.unit_id for u in unit_list if u.kind is UnitKind.TYPE_CHECK
                ),
                known_proof_obligations=tuple(
                    u.unit_id for u in unit_list if u.kind is UnitKind.PROOF
                ),
            )
            resolved_affected_units = sorted(
                set(resolved_affected_units)
                | set(selection.affected_tests)
                | set(selection.required_static_checks)
                | set(selection.required_type_checks)
                | set(selection.affected_proof_obligation_cids)
                | set(selection.fallback_tests)
            )
            if selection.fallback_mode is FallbackMode.FULL_SUITE:
                uncertainty = True
            elif selection.fallback_mode is FallbackMode.BROADER:
                uncertainty = True or uncertainty

        invalidation = self.invalidate_units(
            unit_list,
            changed_symbols=changed_symbols,
            changed_paths=changed_paths,
            affected_symbols=resolved_affected_symbols,
            affected_paths=resolved_affected_paths,
            affected_unit_ids=resolved_affected_units,
        )
        reuse_decisions = self.evaluate_reuse(unit_list, invalidation)

        if isinstance(mutant_outcome, MutantOutcomeClass):
            outcome = mutant_outcome
        else:
            outcome = MutantOutcomeClass(
                _token(mutant_outcome, field_name="mutant_outcome")
            )

        catalog_ids = catalog_unit_ids or tuple(u.unit_id for u in unit_list)
        mode, final_decisions, broaden_reasons = self.broaden_survivors(
            reuse_decisions,
            unit_list,
            mutant_outcome=outcome,
            risk_class=risk_class,
            uncertainty=uncertainty,
            catalog_unit_ids=catalog_ids,
        )

        costs = self.measure_costs(unit_list, final_decisions)
        forest = build_temporary_proof_forest(
            mutant_id=mutant,
            repository_tree_cid=tree,
            decisions=final_decisions,
            units=unit_list,
            parent_canonical_seal_cid=parent_canonical_seal_cid
            or canonical_seal_cid,
        )

        # Explicit seal-protection check (defensive; forest methods refuse).
        if forest.is_canonical or not forest.is_temporary:
            raise CanonicalSealProtectionError(
                "constructed forest violated temporary-only invariant",
                reason_code=REASON_CANONICAL_REPLACE_REFUSED,
            )

        # Stable invalidated set: units that were affected by the change.
        invalidated_ids = tuple(
            sorted({d.unit_id for d in final_decisions if d.affected})
        )
        reused_ids = tuple(
            sorted(
                d.unit_id
                for d in final_decisions
                if d.disposition is UnitDisposition.REUSED
            )
        )
        selected_ids = tuple(
            sorted(
                d.unit_id
                for d in final_decisions
                if d.disposition
                in {
                    UnitDisposition.REVERIFY,
                    UnitDisposition.INVALIDATED,
                    UnitDisposition.BROADENED,
                    UnitDisposition.FULL_SUITE,
                    UnitDisposition.REUSED,
                }
            )
        )
        broadened_ids = tuple(
            sorted(
                d.unit_id
                for d in final_decisions
                if d.disposition is UnitDisposition.BROADENED
            )
        )
        full_suite_ids = tuple(
            sorted(
                d.unit_id
                for d in final_decisions
                if d.disposition is UnitDisposition.FULL_SUITE
            )
        )

        reasons = _stable_unique(
            (
                REASON_TEMPORARY_FOREST_ONLY,
                REASON_CANONICAL_SEAL_PRESERVED,
                REASON_PRODUCTION_POLICY_UNCHANGED,
                REASON_COSTS_MEASURED,
                REASON_CACHE_REUSE_MEASURED,
                *broaden_reasons,
                *(
                    code
                    for decision in final_decisions
                    for code in decision.reason_codes
                ),
            )
        )

        return IncrementalMutationVerificationResult(
            mutant_id=mutant,
            repository_tree_cid=tree,
            decisions=final_decisions,
            temporary_forest=forest,
            cost_accounting=costs,
            broadening_mode=mode,
            mutant_outcome=outcome,
            selected_unit_ids=selected_ids,
            broadened_unit_ids=broadened_ids,
            full_suite_unit_ids=full_suite_ids,
            invalidated_unit_ids=invalidated_ids,
            reused_unit_ids=reused_ids,
            canonical_seal_cid=canonical_seal_cid,
            reason_codes=reasons,
            evidence_subset=self.evidence_subset,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface_id": self.interface_id,
            "schema": self.schema,
            "evidence_subset": self.evidence_subset,
            "adapter_id": self.adapter_id,
            "broadening_policy": self.broadening_policy.to_dict(),
            "require_production_terminal": self.require_production_terminal,
            "board_namespace": BOARD_NAMESPACE,
            "production_policy_changed": False,
        }


def create_incremental_mutation_verifier(
    *,
    broadening_policy: SurvivorBroadeningPolicy | Mapping[str, Any] | None = None,
    require_production_terminal: bool = True,
) -> IncrementalMutationVerifier:
    """Factory for :class:`IncrementalMutationVerifier`."""

    return IncrementalMutationVerifier(
        broadening_policy=SurvivorBroadeningPolicy.from_value(broadening_policy),
        require_production_terminal=require_production_terminal,
    )


def verify_mutant_incremental(
    *,
    mutant_id: str,
    repository_tree_cid: str,
    units: Sequence[ProofUnit | Mapping[str, Any]],
    changed_symbols: Sequence[str] = (),
    changed_paths: Sequence[str] = (),
    affected_symbols: Sequence[str] = (),
    affected_paths: Sequence[str] = (),
    affected_unit_ids: Sequence[str] = (),
    mutant_outcome: MutantOutcomeClass | str = MutantOutcomeClass.KILLED,
    risk_class: str = "",
    uncertainty: bool = False,
    catalog_unit_ids: Sequence[str] = (),
    canonical_seal_cid: str = "",
    parent_canonical_seal_cid: str = "",
    edges: Sequence[Any] | None = None,
    selection_policy: SelectionPolicy | Mapping[str, Any] | None = None,
    catalog: VerificationCatalog | Mapping[str, Any] | None = None,
    broadening_policy: SurvivorBroadeningPolicy | Mapping[str, Any] | None = None,
    require_production_terminal: bool = True,
    verifier: IncrementalMutationVerifier | None = None,
) -> IncrementalMutationVerificationResult:
    """Module-level entry point for ``IncrementalMutationVerifier@1``."""

    active = verifier or create_incremental_mutation_verifier(
        broadening_policy=broadening_policy,
        require_production_terminal=require_production_terminal,
    )
    return active.verify_mutant(
        mutant_id=mutant_id,
        repository_tree_cid=repository_tree_cid,
        units=units,
        changed_symbols=changed_symbols,
        changed_paths=changed_paths,
        affected_symbols=affected_symbols,
        affected_paths=affected_paths,
        affected_unit_ids=affected_unit_ids,
        mutant_outcome=mutant_outcome,
        risk_class=risk_class,
        uncertainty=uncertainty,
        catalog_unit_ids=catalog_unit_ids,
        canonical_seal_cid=canonical_seal_cid,
        parent_canonical_seal_cid=parent_canonical_seal_cid,
        edges=edges,
        selection_policy=selection_policy,
        catalog=catalog,
    )


def incremental_mutation_verifier_descriptor() -> Mapping[str, Any]:
    """Stable discovery descriptor for inventory / adapter probing."""

    return MappingProxyType(
        {
            "interface_id": INCREMENTAL_MUTATION_VERIFIER_INTERFACE,
            "schema": INCREMENTAL_MUTATION_VERIFIER_SCHEMA,
            "evidence_subset": AAE_INCREMENTAL_EVIDENCE,
            "adapter_id": ADAPTER_ID,
            "board_namespace": BOARD_NAMESPACE,
            "generator_id": GENERATOR_ID,
            "generator_version": GENERATOR_VERSION,
            "public_exports": (
                "IncrementalMutationVerifier",
                "verify_mutant_incremental",
                "create_incremental_mutation_verifier",
                "TemporaryProofForest",
                "MutationCostAccounting",
                "classify_unit_invalidation",
                "evaluate_cache_reuse",
                "measure_mutation_costs",
            ),
            "acceptance": (
                "only_affected_units_invalidate",
                "reuse_requires_complete_keys",
                "survivors_broaden_by_policy",
                "temporary_forests_never_replace_canonical_seals",
                "full_and_incremental_costs_and_cache_reuse_measured",
            ),
            "cost_schemas": (
                COST_RECORD_SCHEMA,
                COST_COMPARISON_SCHEMA,
                COST_EVIDENCE,
            ),
            "production_policy_changed": False,
        }
    )


__all__ = [
    "AAE_INCREMENTAL_EVIDENCE",
    "ADAPTER_ID",
    "BOARD_NAMESPACE",
    "BroadeningMode",
    "CACHE_KEY_COMPLETENESS_SCHEMA",
    "CacheKeyBinding",
    "CanonicalSealProtectionError",
    "GENERATOR_ID",
    "GENERATOR_VERSION",
    "INCREMENTAL_MUTATION_RESULT_INTERFACE",
    "INCREMENTAL_MUTATION_RESULT_SCHEMA",
    "INCREMENTAL_MUTATION_VERIFIER_INTERFACE",
    "INCREMENTAL_MUTATION_VERIFIER_SCHEMA",
    "IncrementalBoundsError",
    "IncrementalMutationVerificationResult",
    "IncrementalMutationVerifier",
    "IncrementalVerificationError",
    "MUTATION_COST_ACCOUNTING_SCHEMA",
    "MutantOutcomeClass",
    "MutationCostAccounting",
    "PROOF_UNIT_SCHEMA",
    "ProofUnit",
    "REQUIRED_CACHE_KEY_FIELDS",
    "REASON_AFFECTED_INVALIDATED",
    "REASON_CACHE_MISS",
    "REASON_CACHE_REUSE_MEASURED",
    "REASON_CANONICAL_REPLACE_REFUSED",
    "REASON_CANONICAL_SEAL_PRESERVED",
    "REASON_COMPLETE_KEY_REUSED",
    "REASON_COSTS_MEASURED",
    "REASON_HIGH_RISK_FULL_SUITE",
    "REASON_INCOMPLETE_KEY_REJECTED",
    "REASON_MISSING_KEY_REJECTED",
    "REASON_PRODUCTION_POLICY_UNCHANGED",
    "REASON_STALE_RECEIPT",
    "REASON_SURVIVOR_BROADENED",
    "REASON_SURVIVOR_FULL_SUITE",
    "REASON_TEMPORARY_FOREST_ONLY",
    "REASON_UNAFFECTED_REUSE_CANDIDATE",
    "REASON_UNCERTAINTY_BROADENED",
    "REASON_UNRELATED_NOT_INVALIDATED",
    "REASON_BROADENING_DISABLED",
    "SURVIVOR_BROADENING_POLICY_SCHEMA",
    "SurvivorBroadeningPolicy",
    "TEMPORARY_PROOF_FOREST_INTERFACE",
    "TEMPORARY_PROOF_FOREST_SCHEMA",
    "TemporaryProofForest",
    "UnitDecision",
    "UnitDisposition",
    "UnitKind",
    "broaden_unit_decisions",
    "build_temporary_proof_forest",
    "classify_unit_invalidation",
    "create_incremental_mutation_verifier",
    "evaluate_cache_reuse",
    "incremental_mutation_verifier_descriptor",
    "measure_mutation_costs",
    "resolve_broadening_mode",
    "verify_mutant_incremental",
    # Re-exported cost types for callers/tests.
    "CostKind",
    "CostProvenance",
    "CostValue",
    "ProofCostComparison",
    "ProofCostRecord",
    "ProofMetricsCollector",
    "RunDisposition",
    "SelectionDisposition",
    "compare_costs",
]
