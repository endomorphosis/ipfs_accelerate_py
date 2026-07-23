"""Durable, conformance-bound evidence for multi-prover portfolios.

The generic formal-verification cache predates property-specific portfolios.
It intentionally cannot answer whether a cached model used the same finite
bounds, translation semantics, conformance fixtures, or prover portfolio as a
new request.  This module is the persistence boundary for those decisions.

Three properties are deliberately enforced here:

* :class:`ProverEvidenceKey` binds every semantic and trust dimension which can
  change a portfolio result.
* cache admission is re-evaluated on every lookup.  Expired, invalidated,
  model-only, non-conformant, inconclusive, or insufficient-assurance receipts
  are retained for audit but never promoted to a stronger cache hit.
* a DuckDB fenced lease and outcome channel deduplicate expensive equivalent
  requests between threads, processes, and separately constructed supervisors.

The raw durable receipt is not a dashboard interface.  ``project`` writes a
bounded public JSON document and normalized DuckDB sidecar containing
identifiers and summaries, without copying prover transcripts or witnesses.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, TypeVar

from .duckdb_state import (
    DuckDBConnection,
    DuckDBRow,
    initialize_duckdb_database,
    open_duckdb_connection,
    resolve_duckdb_path,
)
from .formal_verification_contracts import (
    AssuranceLevel,
    ContractValidationError,
    EvidenceFreshness,
    _canonical_value,
    canonical_json,
)
from .multi_prover_router import (
    AttemptOutcome,
    PortfolioResult,
    PortfolioVerdict,
    PropertyKind,
)
from .prover_conformance import ConformanceReport


PROVER_EVIDENCE_STORE_VERSION: Final = 1
PROVER_EVIDENCE_KEY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/prover-evidence-key@1"
)
PROVER_EVIDENCE_RECEIPT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/prover-evidence-receipt@1"
)
PROVER_EVIDENCE_PROJECTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/prover-evidence-projection@1"
)
PROVER_EVIDENCE_DUCKDB_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/prover-evidence-duckdb@1"
)
PROVER_EVIDENCE_FLIGHT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/prover-evidence-flight@1"
)
DEFAULT_EVIDENCE_TTL_SECONDS: Final = 24 * 60 * 60
DEFAULT_FLIGHT_LEASE_SECONDS: Final = 5 * 60
DEFAULT_FLIGHT_WAIT_SECONDS: Final = 10 * 60
DEFAULT_FLIGHT_OUTCOME_TTL_SECONDS: Final = 10 * 60
MAX_PROJECTION_ROWS: Final = 100_000


class EvidenceLookupStatus(str, Enum):
    HIT = "hit"
    MISS = "miss"
    REJECTED = "rejected"


class EvidenceRejectionReason(str, Enum):
    CACHE_MISS = "cache_miss"
    MALFORMED = "malformed_receipt"
    POISONED = "poisoned_receipt"
    STALE = "stale_receipt"
    INVALIDATED = "invalidated_receipt"
    MODEL_ONLY = "model_only_result"
    NON_CONFORMANT = "non_conformant_result"
    FIXTURE_SET_MISMATCH = "conformance_fixture_set_mismatch"
    INSUFFICIENT_CONFORMANCE_ASSURANCE = "conformance_assurance_not_satisfied"
    INSUFFICIENT_ASSURANCE = "required_assurance_not_satisfied"
    INCONCLUSIVE = "inconclusive_result"
    DISAGREEMENT = "prover_disagreement"
    FRESHNESS_NOT_SATISFIED = "freshness_requirement_not_satisfied"


# Compatibility names used by callers which treat this as a cache.
ProverCacheLookupStatus = EvidenceLookupStatus
ProverCacheRejectionReason = EvidenceRejectionReason


class ProverSingleFlightError(RuntimeError):
    """Base error for durable single-flight coordination."""


class ProverSingleFlightTimeout(ProverSingleFlightError, TimeoutError):
    """No leader outcome was observed before the caller's deadline."""


class ProverSingleFlightExecutionError(ProverSingleFlightError):
    """The single-flight owner failed while producing the shared result."""


def _json(value: Any, name: str) -> Any:
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        value = converter()
    try:
        return json.loads(canonical_json(_canonical_value(value)))
    except (TypeError, ValueError, ContractValidationError, json.JSONDecodeError) as exc:
        raise ContractValidationError(f"{name} must contain canonical JSON values") from exc


def _required(value: Any, name: str) -> Any:
    if value is None:
        raise ContractValidationError(f"{name} is required")
    if isinstance(value, str) and not value.strip():
        raise ContractValidationError(f"{name} must not be empty")
    return _json(value.strip() if isinstance(value, str) else value, name)


def _mapping(value: Any, name: str, *, nonempty: bool = False) -> dict[str, Any]:
    result = _json(value, name)
    if not isinstance(result, dict):
        raise ContractValidationError(f"{name} must be an object")
    if nonempty and not result:
        raise ContractValidationError(f"{name} must not be empty")
    return result


def _ordered_values(values: Iterable[Any], name: str) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ContractValidationError(f"{name} must be a sequence")
    normalized = [_json(item, name) for item in values]
    return tuple(sorted(normalized, key=canonical_json))


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def _timestamp_ms(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractValidationError(f"{name} must be a non-negative integer")
    return value


def _named_identity(value: Any, *names: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for name in names:
            candidate = value.get(name)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return ""


@dataclass(frozen=True)
class ProverEvidenceKey:
    """Content identity of a conformance-bound portfolio request.

    ``normalized_model`` is the reviewed semantic model, not provider output.
    Callers may use a string or a canonical structured AST.  Assumptions are
    order-insensitive; portfolio and kernel version mappings are not allowed to
    be empty, avoiding the dangerous identity ``"some installed version"``.
    """

    property_class: PropertyKind | str
    normalized_model: Any
    translator_profile: Any
    assumptions: tuple[Any, ...]
    finite_bounds: Mapping[str, Any]
    prover_versions: Mapping[str, Any]
    kernel_versions: Mapping[str, Any]
    policy: Any
    repository_tree_id: str
    conformance_fixture_set_id: str

    def __post_init__(self) -> None:
        try:
            property_class = PropertyKind(
                str(getattr(self.property_class, "value", self.property_class))
            )
        except ValueError as exc:
            raise ContractValidationError("property_class is unsupported") from exc
        object.__setattr__(self, "property_class", property_class)
        object.__setattr__(
            self, "normalized_model", _required(self.normalized_model, "normalized_model")
        )
        object.__setattr__(
            self, "translator_profile", _required(self.translator_profile, "translator_profile")
        )
        object.__setattr__(
            self, "assumptions", _ordered_values(self.assumptions, "assumptions")
        )
        object.__setattr__(
            self, "finite_bounds", _mapping(self.finite_bounds, "finite_bounds")
        )
        object.__setattr__(
            self,
            "prover_versions",
            _mapping(self.prover_versions, "prover_versions", nonempty=True),
        )
        object.__setattr__(
            self,
            "kernel_versions",
            _mapping(self.kernel_versions, "kernel_versions", nonempty=True),
        )
        object.__setattr__(self, "policy", _required(self.policy, "policy"))
        object.__setattr__(
            self, "repository_tree_id", _required(self.repository_tree_id, "repository_tree_id")
        )
        object.__setattr__(
            self,
            "conformance_fixture_set_id",
            _required(self.conformance_fixture_set_id, "conformance_fixture_set_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROVER_EVIDENCE_KEY_SCHEMA,
            "store_version": PROVER_EVIDENCE_STORE_VERSION,
            "property_class": self.property_class.value,
            "normalized_model": self.normalized_model,
            "translator_profile": self.translator_profile,
            "assumptions": list(self.assumptions),
            "finite_bounds": dict(self.finite_bounds),
            "prover_versions": dict(self.prover_versions),
            "kernel_versions": dict(self.kernel_versions),
            "policy": self.policy,
            "repository_tree_id": self.repository_tree_id,
            "conformance_fixture_set_id": self.conformance_fixture_set_id,
        }

    @property
    def key_id(self) -> str:
        return "prover-evidence-key:" + _digest(self.to_dict())

    @property
    def cache_key(self) -> str:
        return self.key_id

    @property
    def content_id(self) -> str:
        return self.key_id

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProverEvidenceKey":
        if not isinstance(value, Mapping):
            raise ContractValidationError("prover evidence key must be an object")
        if value.get("schema") not in (None, "", PROVER_EVIDENCE_KEY_SCHEMA):
            raise ContractValidationError("unsupported prover evidence key schema")
        return cls(
            property_class=value.get("property_class", ""),
            normalized_model=value.get("normalized_model"),
            translator_profile=value.get("translator_profile"),
            assumptions=tuple(value.get("assumptions") or ()),
            finite_bounds=value.get("finite_bounds") or {},
            prover_versions=value.get("prover_versions") or {},
            kernel_versions=value.get("kernel_versions") or {},
            policy=value.get("policy"),
            repository_tree_id=value.get("repository_tree_id", ""),
            conformance_fixture_set_id=value.get("conformance_fixture_set_id", ""),
        )


def build_prover_evidence_key(
    *,
    property_class: PropertyKind | str | None = None,
    property_kind: PropertyKind | str | None = None,
    normalized_model: Any = None,
    model: Any = None,
    translator_profile: Any = None,
    translator: Any = None,
    assumptions: Sequence[Any] = (),
    finite_bounds: Mapping[str, Any] | None = None,
    bounds: Mapping[str, Any] | None = None,
    prover_versions: Mapping[str, Any] | None = None,
    prover_version: Any | None = None,
    kernel_versions: Mapping[str, Any] | None = None,
    kernel_version: Any | None = None,
    policy: Any = None,
    policy_id: Any = None,
    repository_tree_id: str | None = None,
    candidate_tree: str | None = None,
    tree_id: str | None = None,
    conformance_fixture_set_id: str | None = None,
    fixture_set_id: str | None = None,
) -> ProverEvidenceKey:
    """Build a key while accepting common singular and tree aliases."""

    property_value = property_class if property_class is not None else property_kind
    if property_value is None:
        raise ContractValidationError("property_class is required")
    if normalized_model is not None and model is not None:
        if _json(normalized_model, "normalized_model") != _json(model, "model"):
            raise ContractValidationError("normalized model identities disagree")
    model_value = normalized_model if normalized_model is not None else model
    if translator_profile is not None and translator is not None:
        if _json(translator_profile, "translator_profile") != _json(
            translator, "translator"
        ):
            raise ContractValidationError("translator profile identities disagree")
    translator_value = (
        translator_profile if translator_profile is not None else translator
    )
    if finite_bounds is not None and bounds is not None:
        if _json(finite_bounds, "finite_bounds") != _json(bounds, "bounds"):
            raise ContractValidationError("finite bound identities disagree")
    bounds_value = finite_bounds if finite_bounds is not None else bounds
    if bounds_value is None:
        raise ContractValidationError(
            "finite_bounds is required (use an empty object for an unbounded property)"
        )
    if policy is not None and policy_id is not None:
        if _json(policy, "policy") != _json(policy_id, "policy_id"):
            raise ContractValidationError("policy identities disagree")
    policy_value = policy if policy is not None else policy_id
    trees = [item for item in (repository_tree_id, candidate_tree, tree_id) if item]
    if len(set(trees)) > 1:
        raise ContractValidationError("repository tree identities disagree")
    fixtures = [
        item
        for item in (conformance_fixture_set_id, fixture_set_id)
        if item
    ]
    if len(set(fixtures)) > 1:
        raise ContractValidationError("conformance fixture set identities disagree")
    if prover_versions is not None and prover_version is not None:
        raise ContractValidationError("use prover_versions or prover_version, not both")
    if kernel_versions is not None and kernel_version is not None:
        raise ContractValidationError("use kernel_versions or kernel_version, not both")
    prover_map = (
        prover_versions
        if prover_versions is not None
        else {"prover": _required(prover_version, "prover_version")}
    )
    kernel_map = (
        kernel_versions
        if kernel_versions is not None
        else {"kernel": _required(kernel_version, "kernel_version")}
    )
    return ProverEvidenceKey(
        property_class=property_value,
        normalized_model=model_value,
        translator_profile=translator_value,
        assumptions=tuple(assumptions),
        finite_bounds=bounds_value,
        prover_versions=prover_map,
        kernel_versions=kernel_map,
        policy=policy_value,
        repository_tree_id=trees[0] if trees else "",
        conformance_fixture_set_id=fixtures[0] if fixtures else "",
    )


make_prover_evidence_key = build_prover_evidence_key
ProverCacheKey = ProverEvidenceKey


@dataclass(frozen=True)
class ConformanceBinding:
    """Validated conformance reports bound to one fixture-set identity."""

    fixture_set_id: str
    report_ids: tuple[str, ...]
    passed: bool
    permitted_assurance: AssuranceLevel

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fixture_set_id", _required(self.fixture_set_id, "fixture_set_id")
        )
        object.__setattr__(
            self,
            "report_ids",
            tuple(sorted({_required(item, "report_id") for item in self.report_ids})),
        )
        if not self.report_ids:
            raise ContractValidationError("at least one conformance report is required")
        if not isinstance(self.passed, bool):
            raise ContractValidationError("conformance passed must be boolean")
        object.__setattr__(
            self, "permitted_assurance", AssuranceLevel(self.permitted_assurance)
        )
        if not self.passed and self.permitted_assurance is not AssuranceLevel.UNVERIFIED:
            raise ContractValidationError(
                "failed conformance cannot permit verified assurance"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_set_id": self.fixture_set_id,
            "report_ids": list(self.report_ids),
            "passed": self.passed,
            "permitted_assurance": self.permitted_assurance.value,
        }

    @classmethod
    def from_reports(
        cls,
        reports: Sequence[ConformanceReport],
        *,
        fixture_set_id: str | None = None,
    ) -> "ConformanceBinding":
        if not reports or any(not isinstance(item, ConformanceReport) for item in reports):
            raise ContractValidationError(
                "conformance_reports must contain typed ConformanceReport values"
            )
        identities = {item.fixture_set_id for item in reports}
        if fixture_set_id is not None:
            identities.add(str(fixture_set_id))
        if len(identities) != 1:
            raise ContractValidationError(
                "conformance reports do not bind one fixture set"
            )
        passed = all(item.passed for item in reports)
        permitted = min(
            (item.permitted_assurance for item in reports),
            key=lambda item: item.rank,
        )
        if not passed:
            permitted = AssuranceLevel.UNVERIFIED
        return cls(
            fixture_set_id=identities.pop(),
            report_ids=tuple(item.report_id for item in reports),
            passed=passed,
            permitted_assurance=permitted,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConformanceBinding":
        if not isinstance(value, Mapping):
            raise ContractValidationError("conformance binding must be an object")
        return cls(
            fixture_set_id=value.get("fixture_set_id", ""),
            report_ids=tuple(value.get("report_ids") or ()),
            passed=value.get("passed", False),
            permitted_assurance=value.get(
                "permitted_assurance", AssuranceLevel.UNVERIFIED
            ),
        )


@dataclass(frozen=True)
class ProverEvidenceReceipt:
    """One immutable portfolio result and its cache-admission evidence."""

    key: ProverEvidenceKey
    result: PortfolioResult
    conformance: ConformanceBinding
    created_at_ms: int
    expires_at_ms: int
    model_only: bool = False
    supersedes_receipt_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    receipt_digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.key, ProverEvidenceKey):
            raise ContractValidationError("key must be a ProverEvidenceKey")
        if not isinstance(self.result, PortfolioResult):
            raise ContractValidationError("result must be a PortfolioResult")
        if not isinstance(self.conformance, ConformanceBinding):
            raise ContractValidationError("conformance must be a ConformanceBinding")
        _timestamp_ms(self.created_at_ms, "created_at_ms")
        _timestamp_ms(self.expires_at_ms, "expires_at_ms")
        if self.expires_at_ms <= self.created_at_ms:
            raise ContractValidationError("receipt expiration must follow creation")
        if not isinstance(self.model_only, bool):
            raise ContractValidationError("model_only must be boolean")
        object.__setattr__(
            self, "supersedes_receipt_id",
            str(self.supersedes_receipt_id).strip(),
        )
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))
        if self.result.plan.obligation.property_kind is not self.key.property_class:
            raise ContractValidationError(
                "portfolio property class does not match evidence key"
            )
        policy_id = _named_identity(self.key.policy, "policy_id", "id")
        if policy_id and policy_id != self.result.plan.policy_id:
            raise ContractValidationError(
                "portfolio policy does not match evidence key"
            )
        obligation_tree = _named_identity(
            self.result.plan.obligation.metadata,
            "repository_tree_id",
            "candidate_tree",
            "tree_id",
        )
        if obligation_tree and obligation_tree != self.key.repository_tree_id:
            raise ContractValidationError(
                "portfolio repository tree does not match evidence key"
            )
        known_provers = set(self.key.prover_versions)
        known_kernels = set(self.key.kernel_versions)
        generic_prover_version = bool(
            known_provers & {"prover", "version", "default"}
        )
        generic_kernel_version = bool(
            known_kernels & {"kernel", "version", "default"}
        )
        for attempt in self.result.attempts:
            if attempt.role.value == "kernel":
                if (
                    attempt.prover_id not in known_kernels
                    and not generic_kernel_version
                ):
                    raise ContractValidationError(
                        f"kernel version is not bound for {attempt.prover_id}"
                    )
            elif (
                attempt.prover_id not in known_provers
                and not generic_prover_version
            ):
                raise ContractValidationError(
                    f"prover version is not bound for {attempt.prover_id}"
                )

    @property
    def authoritative_assurance(self) -> AssuranceLevel:
        if self.model_only or not self.conformance.passed:
            return AssuranceLevel.UNVERIFIED
        return min(
            (self.result.assurance, self.conformance.permitted_assurance),
            key=lambda item: item.rank,
        )

    def _unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema": PROVER_EVIDENCE_RECEIPT_SCHEMA,
            "store_version": PROVER_EVIDENCE_STORE_VERSION,
            "key_id": self.key.key_id,
            "key": self.key.to_dict(),
            "result_id": self.result.result_id,
            "result": self.result.to_dict(),
            "conformance": self.conformance.to_dict(),
            "created_at_ms": self.created_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "model_only": self.model_only,
            "authoritative_assurance": self.authoritative_assurance.value,
            "supersedes_receipt_id": self.supersedes_receipt_id,
            "metadata": dict(self.metadata),
        }

    @property
    def computed_digest(self) -> str:
        return _digest(self._unsigned_dict())

    @property
    def receipt_id(self) -> str:
        return "prover-evidence-receipt:" + self.computed_digest

    @property
    def content_id(self) -> str:
        return self.receipt_id

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._unsigned_dict(),
            "receipt_id": self.receipt_id,
            "receipt_digest": self.receipt_digest or self.computed_digest,
        }

    @classmethod
    def create(
        cls,
        *,
        key: ProverEvidenceKey,
        result: PortfolioResult,
        conformance: ConformanceBinding,
        created_at_ms: int,
        expires_at_ms: int,
        model_only: bool = False,
        supersedes_receipt_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> "ProverEvidenceReceipt":
        value = cls(
            key=key,
            result=result,
            conformance=conformance,
            created_at_ms=created_at_ms,
            expires_at_ms=expires_at_ms,
            model_only=model_only,
            supersedes_receipt_id=supersedes_receipt_id,
            metadata=metadata or {},
        )
        return cls(**{**value.__dict__, "receipt_digest": value.computed_digest})

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProverEvidenceReceipt":
        if not isinstance(value, Mapping):
            raise ContractValidationError("prover evidence receipt must be an object")
        if value.get("schema") not in (None, "", PROVER_EVIDENCE_RECEIPT_SCHEMA):
            raise ContractValidationError("unsupported prover evidence receipt schema")
        key = value.get("key")
        result = value.get("result")
        conformance = value.get("conformance")
        if not all(isinstance(item, Mapping) for item in (key, result, conformance)):
            raise ContractValidationError("receipt contains malformed contracts")
        receipt = cls(
            key=ProverEvidenceKey.from_dict(key),
            result=PortfolioResult.from_dict(result),
            conformance=ConformanceBinding.from_dict(conformance),
            created_at_ms=value.get("created_at_ms", -1),
            expires_at_ms=value.get("expires_at_ms", -1),
            model_only=value.get("model_only", False),
            supersedes_receipt_id=value.get("supersedes_receipt_id", ""),
            metadata=value.get("metadata") or {},
            receipt_digest=str(value.get("receipt_digest") or ""),
        )
        if value.get("key_id") != receipt.key.key_id:
            raise ContractValidationError("receipt key identity mismatch")
        if value.get("result_id") != receipt.result.result_id:
            raise ContractValidationError("receipt result identity mismatch")
        if value.get("receipt_id") != receipt.receipt_id:
            raise ContractValidationError("receipt identity mismatch")
        if receipt.receipt_digest != receipt.computed_digest:
            raise ContractValidationError("receipt digest mismatch")
        return receipt


@dataclass(frozen=True)
class EvidenceRequirements:
    required_assurance: AssuranceLevel = AssuranceLevel.SOLVER_CHECKED
    required_freshness: EvidenceFreshness = EvidenceFreshness.CURRENT
    max_age_seconds: int | None = None
    allow_disagreement: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "required_assurance", AssuranceLevel(self.required_assurance)
        )
        object.__setattr__(
            self, "required_freshness", EvidenceFreshness(self.required_freshness)
        )
        if self.max_age_seconds is not None and (
            isinstance(self.max_age_seconds, bool)
            or not isinstance(self.max_age_seconds, int)
            or self.max_age_seconds < 0
        ):
            raise ContractValidationError(
                "max_age_seconds must be a non-negative integer or None"
            )
        if not isinstance(self.allow_disagreement, bool):
            raise ContractValidationError("allow_disagreement must be boolean")


@dataclass(frozen=True)
class EvidenceLookupResult:
    status: EvidenceLookupStatus
    key: ProverEvidenceKey
    receipt: ProverEvidenceReceipt | None = None
    reason_codes: tuple[str, ...] = ()

    @property
    def hit(self) -> bool:
        return self.status is EvidenceLookupStatus.HIT

    @property
    def result(self) -> PortfolioResult | None:
        return self.receipt.result if self.hit and self.receipt else None

    @property
    def reason_code(self) -> str:
        return self.reason_codes[0] if self.reason_codes else ""


@dataclass(frozen=True)
class EvidenceStoreResult:
    stored: bool
    key: ProverEvidenceKey
    receipt: ProverEvidenceReceipt | None = None
    reason_codes: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.stored


@dataclass(frozen=True)
class SingleFlightResult:
    value: Any
    owner: bool
    fencing_token: int

    @property
    def shared(self) -> bool:
        return not self.owner


T = TypeVar("T")


class ProverEvidenceStore:
    """DuckDB receipt store and cross-supervisor single-flight coordinator."""

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        default_ttl_seconds: int = DEFAULT_EVIDENCE_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
        duckdb_timeout_seconds: int = 30,
        sqlite_timeout_seconds: int | None = None,
    ) -> None:
        if isinstance(default_ttl_seconds, bool) or default_ttl_seconds <= 0:
            raise ValueError("default_ttl_seconds must be positive")
        self.path, self._legacy_path = resolve_duckdb_path(
            path,
            default_filename="prover_evidence.duckdb",
            temporary_prefix="prover-evidence-store-",
        )
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.default_ttl_seconds = int(default_ttl_seconds)
        self._clock = clock
        self._duckdb_timeout_seconds = int(
            sqlite_timeout_seconds
            if sqlite_timeout_seconds is not None
            else duckdb_timeout_seconds
        )
        self._initialize()

    @property
    def db_path(self) -> Path:
        return self.path

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    def _connect(self) -> DuckDBConnection:
        return open_duckdb_connection(
            self.path,
            timeout_seconds=self._duckdb_timeout_seconds,
        )

    def _initialize(self) -> None:
        initialize_duckdb_database(
            self.path,
            legacy_sqlite_path=self._legacy_path,
            timeout_seconds=self._duckdb_timeout_seconds,
            table_names=(
                "prover_evidence_receipts",
                "prover_evidence_invalidations",
                "prover_evidence_flights",
                "prover_evidence_flight_outcomes",
            ),
            schema_sql="""
                CREATE TABLE IF NOT EXISTS prover_evidence_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    key_id TEXT NOT NULL,
                    key_json TEXT NOT NULL,
                    receipt_json TEXT NOT NULL,
                    created_at_ms BIGINT NOT NULL,
                    expires_at_ms BIGINT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS prover_evidence_key_idx
                    ON prover_evidence_receipts(key_id, created_at_ms DESC);
                CREATE TABLE IF NOT EXISTS prover_evidence_invalidations (
                    invalidation_id TEXT PRIMARY KEY,
                    receipt_id TEXT NOT NULL,
                    invalidated_at_ms BIGINT NOT NULL,
                    reason TEXT NOT NULL,
                    invalidated_by_receipt_id TEXT NOT NULL,
                    FOREIGN KEY(receipt_id)
                        REFERENCES prover_evidence_receipts(receipt_id)
                );
                CREATE INDEX IF NOT EXISTS prover_evidence_invalidation_idx
                    ON prover_evidence_invalidations(receipt_id, invalidated_at_ms);
                CREATE TABLE IF NOT EXISTS prover_evidence_flights (
                    key_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    token TEXT NOT NULL,
                    fencing_token BIGINT NOT NULL,
                    acquired_at_ms BIGINT NOT NULL,
                    expires_at_ms BIGINT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS prover_evidence_flight_outcomes (
                    key_id TEXT PRIMARY KEY,
                    fencing_token BIGINT NOT NULL,
                    status TEXT NOT NULL,
                    outcome_json TEXT NOT NULL,
                    outcome_digest TEXT NOT NULL,
                    created_at_ms BIGINT NOT NULL,
                    expires_at_ms BIGINT NOT NULL
                );
                """,
        )

    @staticmethod
    def _key(value: ProverEvidenceKey | Mapping[str, Any]) -> ProverEvidenceKey:
        return (
            value
            if isinstance(value, ProverEvidenceKey)
            else ProverEvidenceKey.from_dict(value)
        )

    def put(
        self,
        key: ProverEvidenceKey | Mapping[str, Any],
        result: PortfolioResult | Mapping[str, Any],
        *,
        conformance_reports: Sequence[ConformanceReport] = (),
        conformance: ConformanceBinding | Mapping[str, Any] | None = None,
        ttl_seconds: int | None = None,
        model_only: bool = False,
        supersedes_receipt_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> EvidenceStoreResult:
        """Persist every well-formed portfolio result, including negative ones."""

        cache_key = self._key(key)
        try:
            typed_result = (
                result
                if isinstance(result, PortfolioResult)
                else PortfolioResult.from_dict(result)
            )
            if conformance is None:
                binding = ConformanceBinding.from_reports(
                    conformance_reports,
                    fixture_set_id=cache_key.conformance_fixture_set_id,
                )
            else:
                binding = (
                    conformance
                    if isinstance(conformance, ConformanceBinding)
                    else ConformanceBinding.from_dict(conformance)
                )
        except (TypeError, ValueError, ContractValidationError):
            return EvidenceStoreResult(
                False,
                cache_key,
                reason_codes=(EvidenceRejectionReason.MALFORMED.value,),
            )
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        if isinstance(ttl, bool) or not isinstance(ttl, int) or ttl <= 0:
            raise ValueError("ttl_seconds must be a positive integer")
        now = self._now_ms()
        try:
            receipt = ProverEvidenceReceipt.create(
                key=cache_key,
                result=typed_result,
                conformance=binding,
                created_at_ms=now,
                expires_at_ms=now + ttl * 1000,
                model_only=model_only,
                supersedes_receipt_id=supersedes_receipt_id,
                metadata=metadata,
            )
        except ContractValidationError:
            return EvidenceStoreResult(
                False,
                cache_key,
                reason_codes=(EvidenceRejectionReason.MALFORMED.value,),
            )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT OR IGNORE INTO prover_evidence_receipts(
                    receipt_id, key_id, key_json, receipt_json,
                    created_at_ms, expires_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    receipt.receipt_id,
                    cache_key.key_id,
                    canonical_json(cache_key.to_dict()),
                    canonical_json(receipt.to_dict()),
                    receipt.created_at_ms,
                    receipt.expires_at_ms,
                ),
            )
            if supersedes_receipt_id:
                self._invalidate_in_transaction(
                    connection,
                    supersedes_receipt_id,
                    reason="superseded",
                    invalidated_by_receipt_id=receipt.receipt_id,
                    now_ms=now,
                )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
        return EvidenceStoreResult(True, cache_key, receipt)

    store = put

    def _decode(self, row: DuckDBRow) -> ProverEvidenceReceipt:
        value = json.loads(str(row["receipt_json"]))
        receipt = ProverEvidenceReceipt.from_dict(value)
        if (
            row["receipt_id"] != receipt.receipt_id
            or row["key_id"] != receipt.key.key_id
            or int(row["created_at_ms"]) != receipt.created_at_ms
            or int(row["expires_at_ms"]) != receipt.expires_at_ms
            or json.loads(str(row["key_json"])) != receipt.key.to_dict()
        ):
            raise ContractValidationError("poisoned durable receipt envelope")
        return receipt

    def lookup(
        self,
        key: ProverEvidenceKey | Mapping[str, Any],
        *,
        required_assurance: AssuranceLevel = AssuranceLevel.SOLVER_CHECKED,
        required_freshness: EvidenceFreshness = EvidenceFreshness.CURRENT,
        max_age_seconds: int | None = None,
        allow_disagreement: bool = False,
        requirements: EvidenceRequirements | None = None,
    ) -> EvidenceLookupResult:
        cache_key = self._key(key)
        requested = requirements or EvidenceRequirements(
            required_assurance=required_assurance,
            required_freshness=required_freshness,
            max_age_seconds=max_age_seconds,
            allow_disagreement=allow_disagreement,
        )
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT r.*, EXISTS(
                    SELECT 1 FROM prover_evidence_invalidations i
                    WHERE i.receipt_id=r.receipt_id
                ) AS invalidated
                FROM prover_evidence_receipts r
                WHERE r.key_id=?
                ORDER BY r.created_at_ms DESC, r.receipt_id DESC
                """,
                (cache_key.key_id,),
            ).fetchall()
        finally:
            connection.close()
        if not rows:
            return EvidenceLookupResult(
                EvidenceLookupStatus.MISS,
                cache_key,
                reason_codes=(EvidenceRejectionReason.CACHE_MISS.value,),
            )

        accumulated: set[str] = set()
        now = self._now_ms()
        for row in rows:
            try:
                receipt = self._decode(row)
            except (TypeError, ValueError, ContractValidationError, json.JSONDecodeError):
                accumulated.add(EvidenceRejectionReason.POISONED.value)
                continue
            reasons: set[str] = set()
            if bool(row["invalidated"]):
                reasons.add(EvidenceRejectionReason.INVALIDATED.value)
            if now >= receipt.expires_at_ms or (
                requested.max_age_seconds is not None
                and now - receipt.created_at_ms > requested.max_age_seconds * 1000
            ):
                reasons.update(
                    (
                        EvidenceRejectionReason.STALE.value,
                        EvidenceRejectionReason.FRESHNESS_NOT_SATISFIED.value,
                    )
                )
            if (
                requested.required_freshness is EvidenceFreshness.CURRENT
                and now >= receipt.expires_at_ms
            ):
                reasons.add(EvidenceRejectionReason.FRESHNESS_NOT_SATISFIED.value)
            if receipt.model_only:
                reasons.add(EvidenceRejectionReason.MODEL_ONLY.value)
            if not receipt.conformance.passed:
                reasons.add(EvidenceRejectionReason.NON_CONFORMANT.value)
            if (
                receipt.conformance.fixture_set_id
                != cache_key.conformance_fixture_set_id
            ):
                reasons.add(EvidenceRejectionReason.FIXTURE_SET_MISMATCH.value)
            if not receipt.conformance.permitted_assurance.satisfies(
                requested.required_assurance
            ):
                reasons.add(
                    EvidenceRejectionReason.INSUFFICIENT_CONFORMANCE_ASSURANCE.value
                )
            if not receipt.authoritative_assurance.satisfies(
                requested.required_assurance
            ):
                reasons.add(EvidenceRejectionReason.INSUFFICIENT_ASSURANCE.value)
            if receipt.result.verdict is not PortfolioVerdict.PROVED:
                reasons.add(EvidenceRejectionReason.INCONCLUSIVE.value)
            if receipt.result.disagreement and not requested.allow_disagreement:
                reasons.add(EvidenceRejectionReason.DISAGREEMENT.value)
            if not reasons:
                return EvidenceLookupResult(
                    EvidenceLookupStatus.HIT, cache_key, receipt
                )
            accumulated.update(reasons)
        return EvidenceLookupResult(
            EvidenceLookupStatus.REJECTED,
            cache_key,
            reason_codes=tuple(sorted(accumulated)),
        )

    get = lookup

    def _invalidate_in_transaction(
        self,
        connection: DuckDBConnection,
        receipt_id: str,
        *,
        reason: str,
        invalidated_by_receipt_id: str,
        now_ms: int,
    ) -> bool:
        exists = connection.execute(
            "SELECT 1 FROM prover_evidence_receipts WHERE receipt_id=?",
            (receipt_id,),
        ).fetchone()
        if exists is None:
            return False
        payload = {
            "receipt_id": receipt_id,
            "invalidated_at_ms": now_ms,
            "reason": reason,
            "invalidated_by_receipt_id": invalidated_by_receipt_id,
        }
        connection.execute(
            """
            INSERT OR IGNORE INTO prover_evidence_invalidations(
                invalidation_id, receipt_id, invalidated_at_ms,
                reason, invalidated_by_receipt_id
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "prover-invalidation:" + _digest(payload),
                receipt_id,
                now_ms,
                reason,
                invalidated_by_receipt_id,
            ),
        )
        return True

    def invalidate_receipt(
        self,
        receipt_id: str,
        *,
        reason: str,
        invalidated_by_receipt_id: str = "",
    ) -> bool:
        receipt_id = str(receipt_id).strip()
        reason = str(reason).strip()
        if not receipt_id or not reason:
            raise ValueError("receipt_id and reason must not be empty")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            key_row = connection.execute(
                "SELECT key_id FROM prover_evidence_receipts WHERE receipt_id=?",
                (receipt_id,),
            ).fetchone()
            changed = self._invalidate_in_transaction(
                connection,
                receipt_id,
                reason=reason,
                invalidated_by_receipt_id=str(invalidated_by_receipt_id).strip(),
                now_ms=self._now_ms(),
            )
            if changed and key_row is not None:
                connection.execute(
                    "DELETE FROM prover_evidence_flight_outcomes WHERE key_id=?",
                    (key_row["key_id"],),
                )
            connection.commit()
            return changed
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def invalidate(
        self,
        key: ProverEvidenceKey | Mapping[str, Any],
        *,
        reason: str,
        invalidated_by_receipt_id: str = "",
    ) -> int:
        cache_key = self._key(key)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT receipt_id FROM prover_evidence_receipts WHERE key_id=?",
                (cache_key.key_id,),
            ).fetchall()
            count = sum(
                self._invalidate_in_transaction(
                    connection,
                    str(row["receipt_id"]),
                    reason=str(reason).strip(),
                    invalidated_by_receipt_id=str(invalidated_by_receipt_id).strip(),
                    now_ms=self._now_ms(),
                )
                for row in rows
            )
            connection.execute(
                "DELETE FROM prover_evidence_flight_outcomes WHERE key_id=?",
                (cache_key.key_id,),
            )
            connection.commit()
            return count
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def receipts(self, *, limit: int = MAX_PROJECTION_ROWS) -> tuple[ProverEvidenceReceipt, ...]:
        if not 1 <= limit <= MAX_PROJECTION_ROWS:
            raise ValueError(f"limit must be between 1 and {MAX_PROJECTION_ROWS}")
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT * FROM prover_evidence_receipts
                ORDER BY created_at_ms, receipt_id LIMIT ?
                """,
                (limit,),
            ).fetchall()
        finally:
            connection.close()
        result = []
        for row in rows:
            try:
                result.append(self._decode(row))
            except (TypeError, ValueError, ContractValidationError, json.JSONDecodeError):
                continue
        return tuple(result)

    def _claim_flight(
        self,
        key: ProverEvidenceKey,
        *,
        owner_id: str,
        lease_seconds: int,
    ) -> tuple[bool, str, int]:
        now = self._now_ms()
        token = uuid.uuid4().hex
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            outcome = connection.execute(
                """
                SELECT fencing_token
                FROM prover_evidence_flight_outcomes
                WHERE key_id=? AND expires_at_ms>?
                """,
                (key.key_id, now),
            ).fetchone()
            if outcome is not None:
                # Publishing an outcome and deleting its active lease is one
                # transaction.  A follower can observe the deleted lease just
                # after its preceding outcome poll, then loop back here.  The
                # durable outcome must be checked under the same lock as lease
                # acquisition or that follower can become a second owner.
                connection.commit()
                return False, "", int(outcome["fencing_token"])
            row = connection.execute(
                """
                SELECT fencing_token, expires_at_ms
                FROM prover_evidence_flights WHERE key_id=?
                """,
                (key.key_id,),
            ).fetchone()
            if row is not None and int(row["expires_at_ms"]) > now:
                connection.commit()
                return False, "", int(row["fencing_token"])
            last = connection.execute(
                """
                SELECT MAX(fencing_token) AS value
                FROM prover_evidence_flight_outcomes WHERE key_id=?
                """,
                (key.key_id,),
            ).fetchone()
            fencing_token = max(
                int(last["value"] or 0),
                int(row["fencing_token"]) if row is not None else 0,
            ) + 1
            connection.execute(
                """
                INSERT INTO prover_evidence_flights(
                    key_id, owner_id, token, fencing_token,
                    acquired_at_ms, expires_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(key_id) DO UPDATE SET
                    owner_id=excluded.owner_id,
                    token=excluded.token,
                    fencing_token=excluded.fencing_token,
                    acquired_at_ms=excluded.acquired_at_ms,
                    expires_at_ms=excluded.expires_at_ms
                """,
                (
                    key.key_id,
                    owner_id,
                    token,
                    fencing_token,
                    now,
                    now + lease_seconds * 1000,
                ),
            )
            connection.commit()
            return True, token, fencing_token
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _publish_flight(
        self,
        key: ProverEvidenceKey,
        *,
        token: str,
        fencing_token: int,
        status: str,
        value: Any,
        outcome_ttl_seconds: int,
    ) -> None:
        now = self._now_ms()
        envelope = {
            "schema": PROVER_EVIDENCE_FLIGHT_SCHEMA,
            "key_id": key.key_id,
            "fencing_token": fencing_token,
            "status": status,
            "value": _json(value, "single-flight outcome"),
        }
        rendered = canonical_json(envelope)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            owned = connection.execute(
                """
                SELECT 1 FROM prover_evidence_flights
                WHERE key_id=? AND token=? AND fencing_token=? AND expires_at_ms>?
                """,
                (key.key_id, token, fencing_token, now),
            ).fetchone()
            if owned is None:
                raise ProverSingleFlightError("single-flight owner was fenced")
            connection.execute(
                """
                INSERT INTO prover_evidence_flight_outcomes(
                    key_id, fencing_token, status, outcome_json,
                    outcome_digest, created_at_ms, expires_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(key_id) DO UPDATE SET
                    fencing_token=excluded.fencing_token,
                    status=excluded.status,
                    outcome_json=excluded.outcome_json,
                    outcome_digest=excluded.outcome_digest,
                    created_at_ms=excluded.created_at_ms,
                    expires_at_ms=excluded.expires_at_ms
                """,
                (
                    key.key_id,
                    fencing_token,
                    status,
                    rendered,
                    _digest(envelope),
                    now,
                    now + outcome_ttl_seconds * 1000,
                ),
            )
            connection.execute(
                """
                DELETE FROM prover_evidence_flights
                WHERE key_id=? AND token=? AND fencing_token=?
                """,
                (key.key_id, token, fencing_token),
            )
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _renew_flight(
        self,
        key: ProverEvidenceKey,
        *,
        owner_id: str,
        token: str,
        fencing_token: int,
        lease_seconds: int,
    ) -> None:
        now = self._now_ms()
        connection = self._connect()
        try:
            cursor = connection.execute(
                """
                UPDATE prover_evidence_flights SET expires_at_ms=?
                WHERE key_id=? AND owner_id=? AND token=? AND fencing_token=?
                  AND expires_at_ms>?
                """,
                (
                    now + lease_seconds * 1000,
                    key.key_id,
                    owner_id,
                    token,
                    fencing_token,
                    now,
                ),
            )
            if cursor.rowcount != 1:
                raise ProverSingleFlightError(
                    "single-flight lease heartbeat was fenced"
                )
        finally:
            connection.close()

    def _flight_outcome(self, key: ProverEvidenceKey) -> tuple[Any, int] | None:
        now = self._now_ms()
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT * FROM prover_evidence_flight_outcomes
                WHERE key_id=? AND expires_at_ms>?
                """,
                (key.key_id, now),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        try:
            envelope = json.loads(str(row["outcome_json"]))
            if (
                _digest(envelope) != row["outcome_digest"]
                or envelope["schema"] != PROVER_EVIDENCE_FLIGHT_SCHEMA
                or envelope["key_id"] != key.key_id
                or int(envelope["fencing_token"]) != int(row["fencing_token"])
                or envelope["status"] != row["status"]
            ):
                raise ValueError("outcome envelope mismatch")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ProverSingleFlightExecutionError(
                "single-flight outcome is malformed"
            ) from exc
        if row["status"] == "error":
            raise ProverSingleFlightExecutionError(str(envelope["value"]))
        return envelope["value"], int(row["fencing_token"])

    def single_flight(
        self,
        key: ProverEvidenceKey | Mapping[str, Any],
        producer: Callable[[], T],
        *,
        owner_id: str | None = None,
        lease_seconds: int = DEFAULT_FLIGHT_LEASE_SECONDS,
        wait_timeout_seconds: float = DEFAULT_FLIGHT_WAIT_SECONDS,
        poll_interval_seconds: float = 0.02,
        outcome_ttl_seconds: int = DEFAULT_FLIGHT_OUTCOME_TTL_SECONDS,
    ) -> SingleFlightResult:
        """Run one JSON-valued producer for all equivalent active supervisors.

        Successful outcomes remain briefly reusable, so serial supervisors are
        deduplicated as well as concurrent ones.  A request identity change
        necessarily selects a different flight.
        """

        cache_key = self._key(key)
        if (
            lease_seconds <= 0
            or wait_timeout_seconds <= 0
            or poll_interval_seconds <= 0
            or outcome_ttl_seconds <= 0
        ):
            raise ValueError("single-flight time bounds must be positive")
        existing = self._flight_outcome(cache_key)
        if existing is not None:
            return SingleFlightResult(existing[0], False, existing[1])
        owner = owner_id or f"{os.getpid()}:{threading.get_ident()}:{uuid.uuid4().hex}"
        deadline = time.monotonic() + wait_timeout_seconds
        while time.monotonic() < deadline:
            acquired, token, fencing_token = self._claim_flight(
                cache_key, owner_id=owner, lease_seconds=lease_seconds
            )
            if acquired:
                heartbeat_stop = threading.Event()
                heartbeat_errors: list[BaseException] = []

                def heartbeat() -> None:
                    interval = max(0.05, lease_seconds / 3)
                    while not heartbeat_stop.wait(interval):
                        try:
                            self._renew_flight(
                                cache_key,
                                owner_id=owner,
                                token=token,
                                fencing_token=fencing_token,
                                lease_seconds=lease_seconds,
                            )
                        except BaseException as exc:
                            heartbeat_errors.append(exc)
                            return

                heartbeat_thread = threading.Thread(
                    target=heartbeat,
                    name=f"prover-evidence-flight-{fencing_token}",
                    daemon=True,
                )
                heartbeat_thread.start()
                try:
                    value = _json(producer(), "single-flight producer result")
                    heartbeat_stop.set()
                    heartbeat_thread.join()
                    if heartbeat_errors:
                        raise ProverSingleFlightError(
                            "single-flight lease heartbeat was fenced"
                        ) from heartbeat_errors[0]
                    self._publish_flight(
                        cache_key,
                        token=token,
                        fencing_token=fencing_token,
                        status="ok",
                        value=value,
                        outcome_ttl_seconds=outcome_ttl_seconds,
                    )
                    return SingleFlightResult(value, True, fencing_token)
                except BaseException as exc:
                    heartbeat_stop.set()
                    heartbeat_thread.join()
                    try:
                        self._publish_flight(
                            cache_key,
                            token=token,
                            fencing_token=fencing_token,
                            status="error",
                            value={"type": type(exc).__name__, "message": str(exc)},
                            outcome_ttl_seconds=outcome_ttl_seconds,
                        )
                    except BaseException:
                        pass
                    raise
                finally:
                    heartbeat_stop.set()
                    heartbeat_thread.join()

            observed_fence = fencing_token
            while time.monotonic() < deadline:
                outcome = self._flight_outcome(cache_key)
                if outcome is not None and outcome[1] >= observed_fence:
                    return SingleFlightResult(outcome[0], False, outcome[1])
                connection = self._connect()
                try:
                    active = connection.execute(
                        """
                        SELECT fencing_token, expires_at_ms
                        FROM prover_evidence_flights WHERE key_id=?
                        """,
                        (cache_key.key_id,),
                    ).fetchone()
                finally:
                    connection.close()
                if (
                    active is None
                    or int(active["expires_at_ms"]) <= self._now_ms()
                    or int(active["fencing_token"]) != observed_fence
                ):
                    break
                time.sleep(
                    min(
                        poll_interval_seconds,
                        max(0.0, deadline - time.monotonic()),
                    )
                )
        raise ProverSingleFlightTimeout(
            f"timed out waiting for prover evidence flight {cache_key.key_id}"
        )

    run_single_flight = single_flight
    execute_single_flight = single_flight

    def project(self, path: str | os.PathLike[str]) -> dict[str, Any]:
        return write_prover_evidence_projection(path, self)


@dataclass(frozen=True)
class ProverEvidenceProjectionPaths:
    json_path: Path
    duckdb_path: Path


def prover_evidence_projection_paths(
    path: str | os.PathLike[str],
) -> ProverEvidenceProjectionPaths:
    resolved = Path(path).resolve()
    if resolved.suffix.lower() == ".json":
        return ProverEvidenceProjectionPaths(resolved, resolved.with_suffix(".duckdb"))
    if resolved.suffix.lower() == ".duckdb":
        return ProverEvidenceProjectionPaths(resolved.with_suffix(".json"), resolved)
    raise ValueError("prover evidence projection path must end in .json or .duckdb")


def _public_attempt(
    receipt: ProverEvidenceReceipt, attempt: Any
) -> dict[str, Any]:
    lane = next(
        lane
        for lane in receipt.result.plan.lanes
        if lane.prover_id == attempt.prover_id
    )
    counterexample_id = ""
    if attempt.attempt_id == receipt.result.counterexample_attempt_id:
        counterexample_id = attempt.attempt_id
    return {
        "receipt_id": receipt.receipt_id,
        "result_id": receipt.result.result_id,
        "attempt_id": attempt.attempt_id,
        "prover_id": attempt.prover_id,
        "role": attempt.role.value,
        "stage": attempt.stage,
        "reported_outcome": attempt.reported_outcome.value,
        "effective_outcome": attempt.effective_outcome.value,
        "authoritative": attempt.authoritative,
        "conclusive": attempt.conclusive,
        "duration_ms": attempt.duration_ms,
        "capability_receipt_id": attempt.capability_receipt_id,
        "authority_capability": lane.authority_capability,
        "translation_path_id": lane.translation_path_id,
        "conformance_gate_id": attempt.conformance_gate_id,
        "cancellation_requested": attempt.cancellation_requested,
        "counterexample_id": counterexample_id,
    }


def _projection_payload(
    store: ProverEvidenceStore,
    receipts: Sequence[ProverEvidenceReceipt],
) -> dict[str, Any]:
    now = store._now_ms()
    connection = store._connect()
    try:
        invalidation_rows = connection.execute(
            """
            SELECT invalidation_id, receipt_id, invalidated_at_ms, reason,
                   invalidated_by_receipt_id
            FROM prover_evidence_invalidations
            ORDER BY invalidated_at_ms, invalidation_id
            """
        ).fetchall()
    finally:
        connection.close()
    invalidations = [dict(row) for row in invalidation_rows]
    invalidated_ids = {row["receipt_id"] for row in invalidations}
    receipt_rows: list[dict[str, Any]] = []
    cache_keys: dict[str, dict[str, Any]] = {}
    attempts: list[dict[str, Any]] = []
    capabilities: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    counterexamples: list[dict[str, Any]] = []
    for receipt in receipts:
        key = receipt.key
        cache_keys.setdefault(
            key.key_id,
            {
                "key_id": key.key_id,
                "property_class": key.property_class.value,
                "normalized_model_id": _digest(key.normalized_model),
                "translator_profile_id": _digest(key.translator_profile),
                "assumptions_id": _digest(list(key.assumptions)),
                "finite_bounds_id": _digest(key.finite_bounds),
                "prover_versions_id": _digest(key.prover_versions),
                "prover_versions": dict(key.prover_versions),
                "kernel_versions_id": _digest(key.kernel_versions),
                "kernel_versions": dict(key.kernel_versions),
                "policy_id": _digest(key.policy),
                "policy_name": _named_identity(key.policy, "policy_id", "id"),
                "repository_tree_id": key.repository_tree_id,
                "conformance_fixture_set_id": key.conformance_fixture_set_id,
            },
        )
        freshness = (
            EvidenceFreshness.STALE.value
            if receipt.expires_at_ms <= now or receipt.receipt_id in invalidated_ids
            else EvidenceFreshness.CURRENT.value
        )
        receipt_rows.append(
            {
                "receipt_id": receipt.receipt_id,
                "key_id": receipt.key.key_id,
                "result_id": receipt.result.result_id,
                "property_class": receipt.key.property_class.value,
                "repository_tree_id": receipt.key.repository_tree_id,
                "conformance_fixture_set_id": receipt.key.conformance_fixture_set_id,
                "conformance_passed": receipt.conformance.passed,
                "conformance_report_ids": list(receipt.conformance.report_ids),
                "verdict": receipt.result.verdict.value,
                "assurance": receipt.authoritative_assurance.value,
                "freshness": freshness,
                "created_at_ms": receipt.created_at_ms,
                "expires_at_ms": receipt.expires_at_ms,
                "model_only": receipt.model_only,
                "disagreement": receipt.result.disagreement,
                "invalidated": receipt.receipt_id in invalidated_ids,
                "supersedes_receipt_id": receipt.supersedes_receipt_id,
                "attempt_count": len(receipt.result.attempts),
                "counterexample_attempt_id": receipt.result.counterexample_attempt_id,
            }
        )
        for attempt in receipt.result.attempts:
            public = _public_attempt(receipt, attempt)
            attempts.append(public)
            capabilities.append(
                {
                    "receipt_id": receipt.receipt_id,
                    "prover_id": attempt.prover_id,
                    "role": attempt.role.value,
                    "capability_receipt_id": attempt.capability_receipt_id,
                    "authority_capability": public["authority_capability"],
                    "translation_path_id": public["translation_path_id"],
                    "conformance_gate_id": attempt.conformance_gate_id,
                    "authoritative": attempt.authoritative,
                }
            )
            if public["counterexample_id"]:
                counterexamples.append(
                    {
                        "receipt_id": receipt.receipt_id,
                        "result_id": receipt.result.result_id,
                        "attempt_id": attempt.attempt_id,
                        "prover_id": attempt.prover_id,
                        "property_class": receipt.key.property_class.value,
                        "conclusive": attempt.conclusive,
                    }
                )
        if receipt.result.disagreement:
            disagreements.append(
                {
                    "receipt_id": receipt.receipt_id,
                    "result_id": receipt.result.result_id,
                    "property_class": receipt.key.property_class.value,
                    "authority_attempt_ids": list(
                        receipt.result.authority_attempt_ids
                    ),
                    "counterexample_attempt_id": (
                        receipt.result.counterexample_attempt_id
                    ),
                    "fail_closed": receipt.result.fail_closed,
                }
            )
    assurance = {
        level.value: sum(row["assurance"] == level.value for row in receipt_rows)
        for level in (
            AssuranceLevel.UNVERIFIED,
            AssuranceLevel.CANDIDATE,
            AssuranceLevel.SOLVER_CHECKED,
            AssuranceLevel.KERNEL_VERIFIED,
            AssuranceLevel.ATTESTED,
        )
    }
    freshness = {
        value.value: sum(row["freshness"] == value.value for row in receipt_rows)
        for value in EvidenceFreshness
    }
    return {
        "schema": PROVER_EVIDENCE_PROJECTION_SCHEMA,
        "store_version": PROVER_EVIDENCE_STORE_VERSION,
        "generated_at_ms": now,
        "cache_keys": [cache_keys[key_id] for key_id in sorted(cache_keys)],
        "receipts": receipt_rows,
        "capabilities": capabilities,
        "attempts": attempts,
        "disagreements": disagreements,
        "counterexamples": counterexamples,
        "assurance": assurance,
        "freshness": freshness,
        "invalidations": invalidations,
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _duckdb_module() -> Any:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - declared project dependency
        raise RuntimeError(
            "DuckDB is required for prover evidence projections"
        ) from exc
    return duckdb


_PROJECTION_TABLE_FIELDS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "prover_evidence_cache_keys": (
        ("key_id", "VARCHAR"), ("property_class", "VARCHAR"),
        ("normalized_model_id", "VARCHAR"), ("translator_profile_id", "VARCHAR"),
        ("assumptions_id", "VARCHAR"), ("finite_bounds_id", "VARCHAR"),
        ("prover_versions_id", "VARCHAR"), ("prover_versions", "VARCHAR"),
        ("kernel_versions_id", "VARCHAR"), ("kernel_versions", "VARCHAR"),
        ("policy_id", "VARCHAR"), ("policy_name", "VARCHAR"),
        ("repository_tree_id", "VARCHAR"),
        ("conformance_fixture_set_id", "VARCHAR"),
    ),
    "prover_evidence_receipts": (
        ("receipt_id", "VARCHAR"), ("key_id", "VARCHAR"), ("result_id", "VARCHAR"),
        ("property_class", "VARCHAR"), ("repository_tree_id", "VARCHAR"),
        ("conformance_fixture_set_id", "VARCHAR"), ("conformance_passed", "BOOLEAN"),
        ("conformance_report_ids", "VARCHAR"), ("verdict", "VARCHAR"),
        ("assurance", "VARCHAR"), ("freshness", "VARCHAR"),
        ("created_at_ms", "BIGINT"), ("expires_at_ms", "BIGINT"),
        ("model_only", "BOOLEAN"), ("disagreement", "BOOLEAN"),
        ("invalidated", "BOOLEAN"), ("supersedes_receipt_id", "VARCHAR"),
        ("attempt_count", "BIGINT"), ("counterexample_attempt_id", "VARCHAR"),
    ),
    "prover_evidence_capabilities": (
        ("receipt_id", "VARCHAR"), ("prover_id", "VARCHAR"), ("role", "VARCHAR"),
        ("capability_receipt_id", "VARCHAR"), ("authority_capability", "VARCHAR"),
        ("translation_path_id", "VARCHAR"), ("conformance_gate_id", "VARCHAR"),
        ("authoritative", "BOOLEAN"),
    ),
    "prover_evidence_attempts": (
        ("receipt_id", "VARCHAR"), ("result_id", "VARCHAR"),
        ("attempt_id", "VARCHAR"), ("prover_id", "VARCHAR"), ("role", "VARCHAR"),
        ("stage", "BIGINT"), ("reported_outcome", "VARCHAR"),
        ("effective_outcome", "VARCHAR"), ("authoritative", "BOOLEAN"),
        ("conclusive", "BOOLEAN"), ("duration_ms", "BIGINT"),
        ("capability_receipt_id", "VARCHAR"), ("authority_capability", "VARCHAR"),
        ("translation_path_id", "VARCHAR"), ("conformance_gate_id", "VARCHAR"),
        ("cancellation_requested", "BOOLEAN"), ("counterexample_id", "VARCHAR"),
    ),
    "prover_evidence_disagreements": (
        ("receipt_id", "VARCHAR"), ("result_id", "VARCHAR"),
        ("property_class", "VARCHAR"), ("authority_attempt_ids", "VARCHAR"),
        ("counterexample_attempt_id", "VARCHAR"), ("fail_closed", "BOOLEAN"),
    ),
    "prover_evidence_counterexamples": (
        ("receipt_id", "VARCHAR"), ("result_id", "VARCHAR"),
        ("attempt_id", "VARCHAR"), ("prover_id", "VARCHAR"),
        ("property_class", "VARCHAR"), ("conclusive", "BOOLEAN"),
    ),
    "prover_evidence_invalidations": (
        ("invalidation_id", "VARCHAR"), ("receipt_id", "VARCHAR"),
        ("invalidated_at_ms", "BIGINT"), ("reason", "VARCHAR"),
        ("invalidated_by_receipt_id", "VARCHAR"),
    ),
}

_PROJECTION_VIEW_TARGETS: Mapping[str, str] = {
    "prover_cache_keys": "prover_evidence_cache_keys",
    "prover_receipts": "prover_evidence_receipts",
    "prover_capabilities": "prover_evidence_capabilities",
    "prover_attempts": "prover_evidence_attempts",
    "prover_disagreements": "prover_evidence_disagreements",
    "prover_counterexamples": "prover_evidence_counterexamples",
    "prover_invalidation_lineage": "prover_evidence_invalidations",
}


def _write_projection_duckdb(
    path: Path,
    payload: Mapping[str, Any],
    *,
    source_path: Path,
    source_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.unlink(missing_ok=True)
    connection = _duckdb_module().connect(str(temporary))
    try:
        connection.execute(
            """CREATE TABLE prover_evidence_catalog (
                schema VARCHAR NOT NULL,
                duckdb_schema VARCHAR NOT NULL,
                store_version BIGINT NOT NULL,
                generated_at_ms BIGINT NOT NULL,
                source_path VARCHAR NOT NULL,
                source_sha256 VARCHAR NOT NULL
            )"""
        )
        connection.execute(
            "INSERT INTO prover_evidence_catalog VALUES (?, ?, ?, ?, ?, ?)",
            (
                payload["schema"],
                PROVER_EVIDENCE_DUCKDB_SCHEMA,
                payload["store_version"],
                payload["generated_at_ms"],
                str(source_path),
                source_sha256,
            ),
        )
        sources = {
            "prover_evidence_cache_keys": "cache_keys",
            "prover_evidence_receipts": "receipts",
            "prover_evidence_capabilities": "capabilities",
            "prover_evidence_attempts": "attempts",
            "prover_evidence_disagreements": "disagreements",
            "prover_evidence_counterexamples": "counterexamples",
            "prover_evidence_invalidations": "invalidations",
        }
        for table, fields in _PROJECTION_TABLE_FIELDS.items():
            connection.execute(
                f"CREATE TABLE {table} ("
                + ", ".join(f"{name} {kind} NOT NULL" for name, kind in fields)
                + ")"
            )
            placeholders = ", ".join("?" for _ in fields)
            for row in payload[sources[table]]:
                values = []
                for name, _kind in fields:
                    value = row[name]
                    if isinstance(value, (list, dict)):
                        value = canonical_json(value)
                    values.append(value)
                connection.execute(
                    f"INSERT INTO {table} VALUES ({placeholders})", values
                )
        for view, target in _PROJECTION_VIEW_TARGETS.items():
            connection.execute(f"CREATE VIEW {view} AS SELECT * FROM {target}")
        connection.execute("CHECKPOINT")
    finally:
        connection.close()
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        Path(f"{temporary}.wal").unlink(missing_ok=True)


def write_prover_evidence_projection(
    path: str | os.PathLike[str],
    store: ProverEvidenceStore,
) -> dict[str, Any]:
    """Atomically write equivalent public JSON and normalized DuckDB views."""

    if not isinstance(store, ProverEvidenceStore):
        raise TypeError("store must be a ProverEvidenceStore")
    paths = prover_evidence_projection_paths(path)
    payload = _projection_payload(store, store.receipts())
    payload["query_store"] = {
        "schema": PROVER_EVIDENCE_DUCKDB_SCHEMA,
        "duckdb_path": paths.duckdb_path.name,
        "tables": list(_PROJECTION_TABLE_FIELDS),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _atomic_write(paths.json_path, rendered)
    _write_projection_duckdb(
        paths.duckdb_path,
        payload,
        source_path=paths.json_path,
        source_sha256=_digest(json.loads(rendered)),
    )
    return payload


_QUERY_TABLES = frozenset(
    {
        "prover_evidence_catalog",
        *_PROJECTION_TABLE_FIELDS.keys(),
        *_PROJECTION_VIEW_TARGETS.keys(),
    }
)
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def query_prover_evidence(
    path: str | os.PathLike[str],
    *,
    table: str = "prover_evidence_receipts",
    columns: Sequence[str] = ("*",),
    where: str | None = None,
    parameters: Sequence[Any] = (),
    limit: int = 100,
) -> dict[str, Any]:
    """Run a bounded read-only query over an allowlisted projection table."""

    if table not in _QUERY_TABLES:
        raise ValueError(f"unsupported prover evidence table: {table}")
    if not 1 <= limit <= 1000:
        raise ValueError("query limit must be between 1 and 1000")
    if not columns or any(
        item != "*" and not _IDENTIFIER.fullmatch(item) for item in columns
    ):
        raise ValueError("columns must be simple identifiers")
    if where and (
        ";" in where
        or re.search(
            r"\b(insert|update|delete|drop|attach|copy|call|pragma)\b",
            where,
            re.IGNORECASE,
        )
    ):
        raise ValueError("query predicate must be read-only")
    paths = prover_evidence_projection_paths(path)
    if not paths.duckdb_path.is_file():
        raise FileNotFoundError(paths.duckdb_path)
    sql = f"SELECT {', '.join(columns)} FROM {table}"
    if where:
        sql += f" WHERE {where}"
    sql += " LIMIT ?"
    connection = _duckdb_module().connect(str(paths.duckdb_path), read_only=True)
    try:
        cursor = connection.execute(sql, [*parameters, limit])
        names = [item[0] for item in cursor.description]
        rows = [dict(zip(names, row)) for row in cursor.fetchall()]
    finally:
        connection.close()
    return {"table": table, "columns": names, "rows": rows, "limit": limit}


# Concise compatibility spellings.
EvidenceCacheKey = ProverEvidenceKey
ProverEvidenceIdentity = ProverEvidenceKey
ConformanceBoundProverIdentity = ProverEvidenceKey
EvidenceReceipt = ProverEvidenceReceipt
ConformanceBoundProverReceipt = ProverEvidenceReceipt
ProverEvidenceRequirements = EvidenceRequirements
ProverEvidenceLookupResult = EvidenceLookupResult
ProverEvidenceCache = ProverEvidenceStore
write_evidence_projection = write_prover_evidence_projection
query_evidence_projection = query_prover_evidence


__all__ = [
    "PROVER_EVIDENCE_STORE_VERSION",
    "PROVER_EVIDENCE_KEY_SCHEMA",
    "PROVER_EVIDENCE_RECEIPT_SCHEMA",
    "PROVER_EVIDENCE_PROJECTION_SCHEMA",
    "PROVER_EVIDENCE_DUCKDB_SCHEMA",
    "EvidenceLookupStatus",
    "EvidenceRejectionReason",
    "EvidenceRequirements",
    "EvidenceLookupResult",
    "EvidenceStoreResult",
    "ProverEvidenceKey",
    "ProverEvidenceIdentity",
    "ConformanceBoundProverIdentity",
    "ProverCacheKey",
    "ConformanceBinding",
    "ProverEvidenceReceipt",
    "ConformanceBoundProverReceipt",
    "ProverEvidenceRequirements",
    "ProverEvidenceLookupResult",
    "ProverEvidenceStore",
    "ProverEvidenceCache",
    "SingleFlightResult",
    "ProverSingleFlightError",
    "ProverSingleFlightTimeout",
    "ProverSingleFlightExecutionError",
    "ProverEvidenceProjectionPaths",
    "build_prover_evidence_key",
    "make_prover_evidence_key",
    "prover_evidence_projection_paths",
    "write_prover_evidence_projection",
    "query_prover_evidence",
]
