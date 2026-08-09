"""Transactional repair lineage, proof cache applicability, and fixed-point gates.

DQP-024 / Interfaces: ``RepairLineage@1``
=========================================

Records accepted repair attempts as content-addressed lineage bound to an
admitted symbolic plan, exact AST/symbol/mutation identities, proof
obligations, validation receipts, and worktree revalidation. Proof cache hits
are never assurance promotions: every hit re-derives applicability against the
current semantic roots before reuse. Accepted repairs must reach a code-and-
logic fixed point or roll back. Unsupported classes require approval or
abstention — never silent automatic acceptance.

Acceptance properties
---------------------
* Proof cache hits rederive applicability against current roots.
* Stale or incomplete impact / plan binding prevents write.
* All accepted repairs reach code-and-logic fixed point or roll back.
* Unsupported operator classes require approval/abstain.
* Lineage, events, caches, and proof obligations commit transactionally.

Evidence subset: candidate reuse, stale AST, counterexample, partial plan,
unsupported operator, fixed point, proof invalidation, abstention.

Cold import of this module performs no filesystem, database, network,
provider, or process action. Opening a store is the first I/O boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Final

from ..planning.database_symbolic_planning import (
    PLAN_AUTHORITY_POLICY,
    PlanDisposition,
    SymbolicPlan,
    operator_is_supported,
    operator_requires_approval,
)
from ..task_sources.duckdb_state import open_duckdb_connection

# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

REPAIR_LINEAGE_INTERFACE: Final[str] = "RepairLineage@1"
REPAIR_LINEAGE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-lineage@1"
)
REPAIR_ATTEMPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-attempt@1"
)
REPAIR_FIXED_POINT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-fixed-point@1"
)
REPAIR_PROOF_CACHE_KEY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-proof-cache-key@1"
)
REPAIR_PROOF_CACHE_ENTRY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-proof-cache-entry@1"
)
REPAIR_ROLLBACK_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-rollback@1"
)
REPAIR_EVENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/repair-event@1"
)

DEFAULT_STORE_VERSION: Final[str] = "repair-lineage@1"
AUTHORITY_CLASS: Final[str] = "derived_evidence"
# Cache hits never promote assurance.
CACHE_ASSURANCE_POLICY: Final[str] = "never_promote_assurance"
# Repair store does not invent write authority beyond admitted plan.
WRITE_AUTHORITY_POLICY: Final[str] = "plan_write_admitted_only"

MAX_PATH_BYTES: Final[int] = 4_096
MAX_TEXT_BYTES: Final[int] = 1_024
MAX_ID_BYTES: Final[int] = 512
MAX_BODY_JSON_BYTES: Final[int] = 262_144
MAX_PATHS: Final[int] = 1_024
MAX_OBLIGATIONS: Final[int] = 256
MAX_ROOT_KEYS: Final[int] = 64
MAX_EVENTS: Final[int] = 4_096
DEFAULT_CACHE_TTL_SECONDS: Final[int] = 24 * 60 * 60

_SAFE_ID = re.compile(r"^[A-Za-z0-9:._/@+\-]{1,512}$")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS repair_evidence_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS repair_lineages (
    lineage_id VARCHAR PRIMARY KEY,
    plan_id VARCHAR NOT NULL,
    task_id VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL DEFAULT '',
    mutation_id VARCHAR NOT NULL DEFAULT '',
    snapshot_id VARCHAR NOT NULL,
    worktree_id VARCHAR NOT NULL DEFAULT '',
    disposition VARCHAR NOT NULL,
    fixed_point_status VARCHAR NOT NULL DEFAULT 'pending',
    write_committed INTEGER NOT NULL DEFAULT 0,
    rolled_back INTEGER NOT NULL DEFAULT 0,
    reason VARCHAR NOT NULL DEFAULT '',
    lineage_digest VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL,
    finalized_at VARCHAR NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS repair_lineages_plan_idx
    ON repair_lineages(plan_id, task_id);
CREATE INDEX IF NOT EXISTS repair_lineages_disposition_idx
    ON repair_lineages(disposition, fixed_point_status);

CREATE TABLE IF NOT EXISTS repair_proof_cache (
    cache_key_id VARCHAR PRIMARY KEY,
    key_json VARCHAR NOT NULL,
    verdict VARCHAR NOT NULL,
    assurance_level VARCHAR NOT NULL,
    content_digest VARCHAR NOT NULL,
    semantic_roots_json VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    poisoned INTEGER NOT NULL DEFAULT 0,
    invalidated INTEGER NOT NULL DEFAULT 0,
    created_at_ms BIGINT NOT NULL,
    expires_at_ms BIGINT NOT NULL
);
CREATE INDEX IF NOT EXISTS repair_proof_cache_expiry_idx
    ON repair_proof_cache(expires_at_ms);

CREATE TABLE IF NOT EXISTS repair_proof_invalidations (
    invalidation_id VARCHAR PRIMARY KEY,
    cache_key_id VARCHAR NOT NULL,
    reason VARCHAR NOT NULL,
    invalidated_at VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS repair_proof_invalidations_key_idx
    ON repair_proof_invalidations(cache_key_id);

CREATE TABLE IF NOT EXISTS repair_events (
    event_id VARCHAR PRIMARY KEY,
    lineage_id VARCHAR NOT NULL,
    event_kind VARCHAR NOT NULL,
    reason VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS repair_events_lineage_idx
    ON repair_events(lineage_id, created_at);

CREATE TABLE IF NOT EXISTS repair_rollbacks (
    rollback_id VARCHAR PRIMARY KEY,
    lineage_id VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    reason VARCHAR NOT NULL DEFAULT '',
    restored_paths_json VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS repair_rollbacks_lineage_uidx
    ON repair_rollbacks(lineage_id);
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseRepairEvidenceError(RuntimeError):
    """Base error for repair evidence store failures."""


class DatabaseRepairEvidenceNotOpenError(DatabaseRepairEvidenceError):
    """Operation requires an open store."""


class DatabaseRepairEvidenceIntegrityError(
    DatabaseRepairEvidenceError, ValueError
):
    """Identity, binding, or payload integrity failure."""


class DatabaseRepairEvidenceBoundsError(
    DatabaseRepairEvidenceError, ValueError
):
    """A resource or payload bound was exceeded."""


class DatabaseRepairEvidenceAdmissionError(DatabaseRepairEvidenceError):
    """Repair write or fixed-point admission refused fail-closed."""


class DuckDBUnavailableError(DatabaseRepairEvidenceError):
    """Optional DuckDB dependency is not installed."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class RepairDisposition(str, Enum):
    """Closed outcomes for a repair lineage attempt."""

    ACCEPTED = "accepted"
    ABSTAINED = "abstained"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    REQUIRES_APPROVAL = "requires_approval"
    PENDING = "pending"


class FixedPointStatus(str, Enum):
    """Code-and-logic fixed-point outcomes."""

    PENDING = "pending"
    REACHED = "reached"
    CODE_ONLY = "code_only"
    LOGIC_ONLY = "logic_only"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class ProofCacheLookupStatus(str, Enum):
    HIT = "hit"
    MISS = "miss"
    REJECTED = "rejected"


class ProofCacheRejectReason(str, Enum):
    CACHE_MISS = "cache_miss"
    STALE = "stale_entry"
    POISONED = "poisoned_entry"
    INVALIDATED = "invalidated_entry"
    ROOT_MISMATCH = "semantic_root_mismatch"
    EXPIRED = "expired_entry"
    ASSURANCE_PROMOTION = "assurance_promotion_forbidden"
    MALFORMED = "malformed_entry"
    INCONCLUSIVE = "inconclusive_result"


class RepairRejectReason(str, Enum):
    PLAN_NOT_ADMITTED = "plan_not_admitted"
    PLAN_WRITE_NOT_ADMITTED = "plan_write_not_admitted"
    STALE_AST = "stale_ast"
    INCOMPLETE_IMPACT = "incomplete_impact"
    BLOCKING_FRONTIER = "blocking_frontier"
    UNSUPPORTED_OPERATOR = "unsupported_operator"
    REQUIRES_APPROVAL = "requires_approval"
    COUNTEREXAMPLE = "counterexample"
    FIXED_POINT_FAILED = "fixed_point_failed"
    WORKTREE_MISMATCH = "worktree_mismatch"
    PROOF_CACHE_INAPPLICABLE = "proof_cache_inapplicable"
    SCOPE_ESCAPE = "scope_escape"
    PARTIAL_PLAN = "partial_plan"
    ALREADY_FINAL = "already_final"
    MISSING_OBLIGATIONS = "missing_obligations"
    MALFORMED_INPUT = "malformed_input"


class RollbackStatus(str, Enum):
    VERIFIED = "verified"
    FAILED = "failed"
    PENDING = "pending"


class AssuranceLevel(str, Enum):
    NONE = "none"
    HEURISTIC = "heuristic"
    VALIDATED = "validated"
    SOLVER_CHECKED = "solver_checked"
    KERNEL_VERIFIED = "kernel_verified"
    ATTESTED = "attested"

    @classmethod
    def coerce(cls, value: Any) -> "AssuranceLevel":
        if isinstance(value, cls):
            return value
        raw = str(getattr(value, "value", value) or "").strip().casefold()
        aliases = {
            "none": cls.NONE,
            "heuristic": cls.HEURISTIC,
            "validated": cls.VALIDATED,
            "solver_checked": cls.SOLVER_CHECKED,
            "solver-checked": cls.SOLVER_CHECKED,
            "kernel_verified": cls.KERNEL_VERIFIED,
            "kernel-verified": cls.KERNEL_VERIFIED,
            "attested": cls.ATTESTED,
        }
        try:
            return aliases[raw]
        except KeyError as exc:
            raise DatabaseRepairEvidenceIntegrityError(
                f"unsupported assurance level: {value!r}"
            ) from exc

    @property
    def rank(self) -> int:
        order = (
            AssuranceLevel.NONE,
            AssuranceLevel.HEURISTIC,
            AssuranceLevel.VALIDATED,
            AssuranceLevel.SOLVER_CHECKED,
            AssuranceLevel.KERNEL_VERIFIED,
            AssuranceLevel.ATTESTED,
        )
        return order.index(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def duckdb_available() -> bool:
    try:
        import duckdb  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if "\x00" in text:
        raise DatabaseRepairEvidenceIntegrityError(f"{name} contains NUL")
    if required and not text:
        raise DatabaseRepairEvidenceIntegrityError(f"{name} is required")
    if text and not _SAFE_ID.fullmatch(text) and name.endswith(
        ("_id", "id", "lineage_id", "plan_id", "task_id", "key_id")
    ):
        raise DatabaseRepairEvidenceIntegrityError(
            f"{name} must be a compact identifier"
        )
    if len(text.encode("utf-8")) > MAX_ID_BYTES and name.endswith(
        ("_id", "id", "lineage_id", "plan_id", "key_id")
    ):
        raise DatabaseRepairEvidenceBoundsError(
            f"{name} exceeds {MAX_ID_BYTES} bytes"
        )
    return text


def _bounded_text(value: Any, maximum: int = MAX_TEXT_BYTES) -> str:
    text = str(value or "")
    if "\x00" in text:
        raise DatabaseRepairEvidenceIntegrityError("text contains NUL")
    if len(text.encode("utf-8")) > maximum:
        raise DatabaseRepairEvidenceBoundsError(
            f"text exceeds {maximum} bytes"
        )
    return text


def _repo_path(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        raise DatabaseRepairEvidenceIntegrityError("path is required")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or raw != path.as_posix():
        raise DatabaseRepairEvidenceIntegrityError(
            "path must be a normalized repository-relative path"
        )
    if len(raw.encode("utf-8")) > MAX_PATH_BYTES:
        raise DatabaseRepairEvidenceBoundsError(
            f"path exceeds {MAX_PATH_BYTES} bytes"
        )
    return raw


def _paths(
    values: Sequence[Any] | None,
    name: str,
    *,
    required: bool = False,
    maximum: int = MAX_PATHS,
) -> tuple[str, ...]:
    if values is None:
        items: list[Any] = []
    elif isinstance(values, (str, bytes, bytearray)):
        raise DatabaseRepairEvidenceIntegrityError(f"{name} must be a sequence")
    else:
        items = list(values)
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        path = _repo_path(item)
        if path not in seen:
            seen.add(path)
            result.append(path)
    if required and not result:
        raise DatabaseRepairEvidenceIntegrityError(f"{name} must not be empty")
    if len(result) > maximum:
        raise DatabaseRepairEvidenceBoundsError(
            f"{name} exceeds {maximum} paths"
        )
    return tuple(sorted(result))


def _ids(
    values: Sequence[Any] | None,
    name: str,
    *,
    required: bool = False,
    maximum: int = MAX_OBLIGATIONS,
) -> tuple[str, ...]:
    if values is None:
        items: list[Any] = []
    elif isinstance(values, (str, bytes, bytearray)):
        raise DatabaseRepairEvidenceIntegrityError(f"{name} must be a sequence")
    else:
        items = list(values)
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = _text(item, name)
        if text not in seen:
            seen.add(text)
            result.append(text)
    if required and not result:
        raise DatabaseRepairEvidenceIntegrityError(f"{name} must not be empty")
    if len(result) > maximum:
        raise DatabaseRepairEvidenceBoundsError(
            f"{name} exceeds {maximum} items"
        )
    return tuple(result)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    )


def _sha256_digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _identity(prefix: str, value: Any) -> str:
    return f"{prefix}:{_sha256_digest(_canonical_json(value).encode('utf-8'))}"


def _row_mapping(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return {str(key): row[key] for key in row}
    try:
        keys = list(row.keys())  # type: ignore[attr-defined]
    except Exception:
        return {}
    return {str(key): row[key] for key in keys}


def _split_sql_statements(sql_text: str) -> list[str]:
    statements: list[str] = []
    for chunk in str(sql_text).split(";"):
        statement = chunk.strip()
        if not statement:
            continue
        lines = [
            line
            for line in statement.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]
        if lines:
            statements.append("\n".join(lines))
    return statements


def _load_json_object(text: Any) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(str(text))
    except (TypeError, ValueError) as exc:
        raise DatabaseRepairEvidenceIntegrityError(
            "stored JSON object is corrupted"
        ) from exc
    if not isinstance(value, dict):
        raise DatabaseRepairEvidenceIntegrityError(
            "stored JSON must be an object"
        )
    return value


def _load_json_list(text: Any) -> list[Any]:
    if not text:
        return []
    try:
        value = json.loads(str(text))
    except (TypeError, ValueError) as exc:
        raise DatabaseRepairEvidenceIntegrityError(
            "stored JSON list is corrupted"
        ) from exc
    if not isinstance(value, list):
        raise DatabaseRepairEvidenceIntegrityError(
            "stored JSON list must be an array"
        )
    return value


def assurance_satisfies(
    actual: AssuranceLevel | str,
    required: AssuranceLevel | str,
) -> bool:
    return AssuranceLevel.coerce(actual).rank >= AssuranceLevel.coerce(required).rank


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepairProofCacheKey:
    """Semantic roots that determine proof-cache applicability."""

    subject_id: str
    semantic_roots: Mapping[str, Any] = field(default_factory=dict)
    obligation_ids: tuple[str, ...] = ()
    plan_id: str = ""
    snapshot_id: str = ""
    mutation_id: str = ""
    policy_id: str = ""
    key_id: str = ""
    schema: str = REPAIR_PROOF_CACHE_KEY_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "subject_id", _text(self.subject_id, "subject_id")
        )
        roots = dict(self.semantic_roots or {})
        if len(roots) > MAX_ROOT_KEYS:
            raise DatabaseRepairEvidenceBoundsError(
                f"semantic_roots exceeds {MAX_ROOT_KEYS} keys"
            )
        # Canonicalize root values to JSON-safe sorted form.
        normalized_roots = {
            str(key): roots[key]
            for key in sorted(roots, key=lambda item: str(item))
        }
        object.__setattr__(
            self, "semantic_roots", MappingProxyType(normalized_roots)
        )
        object.__setattr__(
            self,
            "obligation_ids",
            _ids(self.obligation_ids, "obligation_ids", maximum=MAX_OBLIGATIONS),
        )
        for name in ("plan_id", "snapshot_id", "mutation_id", "policy_id"):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        if not self.key_id:
            object.__setattr__(
                self,
                "key_id",
                _identity(
                    "repair-proof-key",
                    {
                        "schema": self.schema,
                        "subject_id": self.subject_id,
                        "semantic_roots": normalized_roots,
                        "obligation_ids": list(self.obligation_ids),
                        "plan_id": self.plan_id,
                        "snapshot_id": self.snapshot_id,
                        "mutation_id": self.mutation_id,
                        "policy_id": self.policy_id,
                    },
                ),
            )
        else:
            object.__setattr__(self, "key_id", _text(self.key_id, "key_id"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "key_id": self.key_id,
            "subject_id": self.subject_id,
            "semantic_roots": dict(self.semantic_roots),
            "obligation_ids": list(self.obligation_ids),
            "plan_id": self.plan_id,
            "snapshot_id": self.snapshot_id,
            "mutation_id": self.mutation_id,
            "policy_id": self.policy_id,
        }

    def matches_roots(self, other_roots: Mapping[str, Any]) -> bool:
        """Return whether other roots are an exact applicability match."""

        left = dict(self.semantic_roots)
        right = {
            str(key): other_roots[key]
            for key in sorted(other_roots or {}, key=lambda item: str(item))
        }
        return _canonical_json(left) == _canonical_json(right)


@dataclass(frozen=True)
class RepairProofCacheEntry:
    """Stored proof result. Hits never promote assurance."""

    key: RepairProofCacheKey
    verdict: str
    assurance_level: str
    content_digest: str
    body: Mapping[str, Any] = field(default_factory=dict)
    created_at_ms: int = 0
    expires_at_ms: int = 0
    poisoned: bool = False
    invalidated: bool = False
    schema: str = REPAIR_PROOF_CACHE_ENTRY_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.key, RepairProofCacheKey):
            raise DatabaseRepairEvidenceIntegrityError(
                "key must be RepairProofCacheKey"
            )
        object.__setattr__(
            self, "verdict", _text(self.verdict, "verdict").casefold()
        )
        object.__setattr__(
            self,
            "assurance_level",
            AssuranceLevel.coerce(self.assurance_level).value,
        )
        digest = str(self.content_digest or "").strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            # Allow deriving digest from body when not provided.
            if not digest:
                digest = _sha256_digest(
                    _canonical_json(dict(self.body or {})).encode("utf-8")
                )
            else:
                raise DatabaseRepairEvidenceIntegrityError(
                    "content_digest must be sha256:<64 lowercase hex>"
                )
        object.__setattr__(self, "content_digest", digest)
        object.__setattr__(self, "body", MappingProxyType(dict(self.body or {})))
        created = int(self.created_at_ms or 0)
        expires = int(self.expires_at_ms or 0)
        if created < 0 or expires < 0:
            raise DatabaseRepairEvidenceBoundsError(
                "timestamps must be non-negative"
            )
        object.__setattr__(self, "created_at_ms", created)
        object.__setattr__(self, "expires_at_ms", expires)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "key": self.key.to_dict(),
            "verdict": self.verdict,
            "assurance_level": self.assurance_level,
            "content_digest": self.content_digest,
            "body": dict(self.body),
            "created_at_ms": self.created_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "poisoned": bool(self.poisoned),
            "invalidated": bool(self.invalidated),
            "authority": AUTHORITY_CLASS,
            "cache_assurance_policy": CACHE_ASSURANCE_POLICY,
        }


@dataclass(frozen=True)
class ProofCacheLookupResult:
    """Result of a proof-cache lookup with re-derived applicability."""

    status: ProofCacheLookupStatus
    key: RepairProofCacheKey
    entry: RepairProofCacheEntry | None = None
    reason: ProofCacheRejectReason | None = None
    applicability_rederived: bool = False
    applicable: bool = False

    @property
    def hit(self) -> bool:
        return (
            self.status is ProofCacheLookupStatus.HIT
            and self.entry is not None
            and self.applicable
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "key": self.key.to_dict(),
            "entry": None if self.entry is None else self.entry.to_dict(),
            "reason": None if self.reason is None else self.reason.value,
            "applicability_rederived": bool(self.applicability_rederived),
            "applicable": bool(self.applicable),
            "cache_assurance_policy": CACHE_ASSURANCE_POLICY,
        }


@dataclass(frozen=True)
class FixedPointEvidence:
    """Observed code-and-logic fixed-point evidence for one repair."""

    code_fixed: bool
    logic_fixed: bool
    residual_obligations: tuple[str, ...] = ()
    residual_paths: tuple[str, ...] = ()
    validation_receipt_id: str = ""
    proof_receipt_id: str = ""
    worktree_digest: str = ""
    expected_worktree_digest: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.code_fixed, bool) or not isinstance(
            self.logic_fixed, bool
        ):
            raise DatabaseRepairEvidenceIntegrityError(
                "code_fixed and logic_fixed must be boolean"
            )
        object.__setattr__(
            self,
            "residual_obligations",
            _ids(
                self.residual_obligations,
                "residual_obligations",
                maximum=MAX_OBLIGATIONS,
            ),
        )
        object.__setattr__(
            self,
            "residual_paths",
            _paths(self.residual_paths, "residual_paths"),
        )
        for name in (
            "validation_receipt_id",
            "proof_receipt_id",
            "worktree_digest",
            "expected_worktree_digest",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        object.__setattr__(self, "notes", _bounded_text(self.notes))

    @property
    def reached(self) -> bool:
        return (
            self.code_fixed
            and self.logic_fixed
            and not self.residual_obligations
            and not self.residual_paths
        )

    def status(self) -> FixedPointStatus:
        if self.reached:
            return FixedPointStatus.REACHED
        if self.code_fixed and not self.logic_fixed:
            return FixedPointStatus.CODE_ONLY
        if self.logic_fixed and not self.code_fixed:
            return FixedPointStatus.LOGIC_ONLY
        return FixedPointStatus.FAILED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPAIR_FIXED_POINT_SCHEMA,
            "code_fixed": bool(self.code_fixed),
            "logic_fixed": bool(self.logic_fixed),
            "reached": self.reached,
            "status": self.status().value,
            "residual_obligations": list(self.residual_obligations),
            "residual_paths": list(self.residual_paths),
            "validation_receipt_id": self.validation_receipt_id,
            "proof_receipt_id": self.proof_receipt_id,
            "worktree_digest": self.worktree_digest,
            "expected_worktree_digest": self.expected_worktree_digest,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class RepairAttemptRequest:
    """Inputs for applying a repair under an admitted symbolic plan."""

    plan: SymbolicPlan
    operator_class: str
    write_paths: tuple[str, ...]
    obligation_ids: tuple[str, ...] = ()
    worktree_id: str = ""
    worktree_digest: str = ""
    expected_worktree_digest: str = ""
    mutation_id: str = ""
    candidate_id: str = ""
    proof_cache_key: RepairProofCacheKey | None = None
    fixed_point: FixedPointEvidence | None = None
    approval_granted: bool = False
    counterexample_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.plan, SymbolicPlan):
            raise DatabaseRepairEvidenceIntegrityError(
                "plan must be SymbolicPlan"
            )
        object.__setattr__(
            self,
            "operator_class",
            str(self.operator_class or "").strip().casefold().replace("-", "_"),
        )
        if not self.operator_class:
            raise DatabaseRepairEvidenceIntegrityError(
                "operator_class is required"
            )
        object.__setattr__(
            self, "write_paths", _paths(self.write_paths, "write_paths")
        )
        object.__setattr__(
            self,
            "obligation_ids",
            _ids(self.obligation_ids, "obligation_ids", maximum=MAX_OBLIGATIONS),
        )
        for name in (
            "worktree_id",
            "worktree_digest",
            "expected_worktree_digest",
            "mutation_id",
            "candidate_id",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        if self.proof_cache_key is not None and not isinstance(
            self.proof_cache_key, RepairProofCacheKey
        ):
            raise DatabaseRepairEvidenceIntegrityError(
                "proof_cache_key must be RepairProofCacheKey or None"
            )
        if self.fixed_point is not None and not isinstance(
            self.fixed_point, FixedPointEvidence
        ):
            raise DatabaseRepairEvidenceIntegrityError(
                "fixed_point must be FixedPointEvidence or None"
            )
        if not isinstance(self.approval_granted, bool):
            raise DatabaseRepairEvidenceIntegrityError(
                "approval_granted must be boolean"
            )
        object.__setattr__(
            self,
            "counterexample_ids",
            _ids(
                self.counterexample_ids,
                "counterexample_ids",
                maximum=MAX_OBLIGATIONS,
            ),
        )
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(self.metadata or {}))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REPAIR_ATTEMPT_SCHEMA,
            "plan_id": self.plan.plan_id,
            "task_id": self.plan.task_id,
            "operator_class": self.operator_class,
            "write_paths": list(self.write_paths),
            "obligation_ids": list(self.obligation_ids),
            "worktree_id": self.worktree_id,
            "worktree_digest": self.worktree_digest,
            "expected_worktree_digest": self.expected_worktree_digest,
            "mutation_id": self.mutation_id,
            "candidate_id": self.candidate_id,
            "proof_cache_key": None
            if self.proof_cache_key is None
            else self.proof_cache_key.to_dict(),
            "fixed_point": None
            if self.fixed_point is None
            else self.fixed_point.to_dict(),
            "approval_granted": bool(self.approval_granted),
            "counterexample_ids": list(self.counterexample_ids),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RepairLineage:
    """Durable repair lineage bound to plan, mutation, proof, and fixed point.

    Interface: ``RepairLineage@1``.
    """

    lineage_id: str
    plan_id: str
    task_id: str
    disposition: RepairDisposition | str
    snapshot_id: str
    attempt_id: str = ""
    mutation_id: str = ""
    worktree_id: str = ""
    operator_class: str = ""
    candidate_id: str = ""
    write_paths: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    fixed_point_status: FixedPointStatus | str = FixedPointStatus.PENDING
    fixed_point: Mapping[str, Any] = field(default_factory=dict)
    write_committed: bool = False
    rolled_back: bool = False
    reasons: tuple[str, ...] = ()
    proof_cache: Mapping[str, Any] = field(default_factory=dict)
    events: tuple[Mapping[str, Any], ...] = ()
    lineage_digest: str = ""
    created_at: str = ""
    finalized_at: str = ""
    schema: str = REPAIR_LINEAGE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "lineage_id", _text(self.lineage_id, "lineage_id")
        )
        object.__setattr__(self, "plan_id", _text(self.plan_id, "plan_id"))
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        disposition = self.disposition
        if not isinstance(disposition, RepairDisposition):
            disposition = RepairDisposition(str(disposition).strip().casefold())
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(
            self, "snapshot_id", _text(self.snapshot_id, "snapshot_id")
        )
        for name in (
            "attempt_id",
            "mutation_id",
            "worktree_id",
            "operator_class",
            "candidate_id",
            "finalized_at",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        object.__setattr__(
            self, "write_paths", _paths(self.write_paths, "write_paths")
        )
        object.__setattr__(
            self,
            "obligation_ids",
            _ids(self.obligation_ids, "obligation_ids", maximum=MAX_OBLIGATIONS),
        )
        fp_status = self.fixed_point_status
        if not isinstance(fp_status, FixedPointStatus):
            fp_status = FixedPointStatus(str(fp_status).strip().casefold())
        object.__setattr__(self, "fixed_point_status", fp_status)
        object.__setattr__(
            self, "fixed_point", MappingProxyType(dict(self.fixed_point or {}))
        )
        object.__setattr__(
            self, "proof_cache", MappingProxyType(dict(self.proof_cache or {}))
        )
        reasons = tuple(
            _bounded_text(item) for item in (self.reasons or ()) if str(item).strip()
        )
        object.__setattr__(self, "reasons", reasons)
        events = tuple(dict(item) for item in (self.events or ()))
        if len(events) > MAX_EVENTS:
            raise DatabaseRepairEvidenceBoundsError(
                f"events exceeds {MAX_EVENTS}"
            )
        object.__setattr__(self, "events", events)
        object.__setattr__(
            self, "created_at", _text(self.created_at or _utc_iso(), "created_at")
        )
        # Accepted repairs must be fixed-point reached and not rolled back.
        write_committed = bool(self.write_committed)
        if disposition is RepairDisposition.ACCEPTED:
            if fp_status is not FixedPointStatus.REACHED or self.rolled_back:
                write_committed = False
        elif disposition is not RepairDisposition.PENDING:
            write_committed = False
        object.__setattr__(self, "write_committed", write_committed)
        payload = {
            "schema": self.schema,
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "disposition": disposition.value,
            "snapshot_id": self.snapshot_id,
            "mutation_id": self.mutation_id,
            "operator_class": self.operator_class,
            "write_paths": list(self.write_paths),
            "obligation_ids": list(self.obligation_ids),
            "fixed_point_status": fp_status.value,
            "write_committed": write_committed,
            "rolled_back": bool(self.rolled_back),
        }
        digest = self.lineage_digest or _sha256_digest(
            _canonical_json(payload).encode("utf-8")
        )
        object.__setattr__(self, "lineage_digest", digest)

    @property
    def interface(self) -> str:
        return REPAIR_LINEAGE_INTERFACE

    @property
    def accepted(self) -> bool:
        return (
            self.disposition is RepairDisposition.ACCEPTED
            and self.fixed_point_status is FixedPointStatus.REACHED
            and self.write_committed
            and not self.rolled_back
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": REPAIR_LINEAGE_INTERFACE,
            "lineage_id": self.lineage_id,
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "mutation_id": self.mutation_id,
            "snapshot_id": self.snapshot_id,
            "worktree_id": self.worktree_id,
            "operator_class": self.operator_class,
            "candidate_id": self.candidate_id,
            "disposition": self.disposition.value
            if isinstance(self.disposition, RepairDisposition)
            else str(self.disposition),
            "write_paths": list(self.write_paths),
            "obligation_ids": list(self.obligation_ids),
            "fixed_point_status": self.fixed_point_status.value
            if isinstance(self.fixed_point_status, FixedPointStatus)
            else str(self.fixed_point_status),
            "fixed_point": dict(self.fixed_point),
            "write_committed": bool(self.write_committed),
            "rolled_back": bool(self.rolled_back),
            "reasons": list(self.reasons),
            "proof_cache": dict(self.proof_cache),
            "events": [dict(item) for item in self.events],
            "lineage_digest": self.lineage_digest,
            "created_at": self.created_at,
            "finalized_at": self.finalized_at,
            "authority": AUTHORITY_CLASS,
            "cache_assurance_policy": CACHE_ASSURANCE_POLICY,
            "write_authority_policy": WRITE_AUTHORITY_POLICY,
            "plan_authority_policy": PLAN_AUTHORITY_POLICY,
        }


@dataclass(frozen=True)
class RepairRollbackReceipt:
    """Independent rollback record for a non-fixed-point repair."""

    rollback_id: str
    lineage_id: str
    status: RollbackStatus | str
    reason: str = ""
    restored_paths: tuple[str, ...] = ()
    body: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""
    schema: str = REPAIR_ROLLBACK_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "rollback_id", _text(self.rollback_id, "rollback_id")
        )
        object.__setattr__(
            self, "lineage_id", _text(self.lineage_id, "lineage_id")
        )
        status = self.status
        if not isinstance(status, RollbackStatus):
            status = RollbackStatus(str(status).strip().casefold())
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason", _bounded_text(self.reason))
        object.__setattr__(
            self, "restored_paths", _paths(self.restored_paths, "restored_paths")
        )
        object.__setattr__(self, "body", MappingProxyType(dict(self.body or {})))
        object.__setattr__(
            self, "created_at", _text(self.created_at or _utc_iso(), "created_at")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "rollback_id": self.rollback_id,
            "lineage_id": self.lineage_id,
            "status": self.status.value
            if isinstance(self.status, RollbackStatus)
            else str(self.status),
            "reason": self.reason,
            "restored_paths": list(self.restored_paths),
            "body": dict(self.body),
            "created_at": self.created_at,
            "authority": AUTHORITY_CLASS,
        }


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class DatabaseRepairEvidenceStore:
    """DuckDB-backed RepairLineage@1 store with proof-cache rederivation."""

    INTERFACE: Final[str] = REPAIR_LINEAGE_INTERFACE
    SCHEMA: Final[str] = REPAIR_LINEAGE_SCHEMA

    def __init__(
        self,
        database_path: Path | str,
        *,
        store_version: str = DEFAULT_STORE_VERSION,
        default_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for DatabaseRepairEvidenceStore; install "
                "the optional duckdb dependency"
            )
        if (
            isinstance(default_ttl_seconds, bool)
            or not isinstance(default_ttl_seconds, int)
            or default_ttl_seconds <= 0
        ):
            raise DatabaseRepairEvidenceBoundsError(
                "default_ttl_seconds must be a positive integer"
            )
        self._path = Path(database_path)
        self._store_version = _text(
            store_version or DEFAULT_STORE_VERSION, "store_version"
        )
        self._default_ttl_seconds = int(default_ttl_seconds)
        self._connection: Any | None = None
        self._lock = threading.RLock()
        self._closed = True

    # -- lifecycle -----------------------------------------------------------

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def store_version(self) -> str:
        return self._store_version

    @property
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

    def open(self) -> "DatabaseRepairEvidenceStore":
        with self._lock:
            if self.is_open:
                return self
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = open_duckdb_connection(self._path)
            for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                connection.execute(statement)
            for key, value in (
                ("interface", REPAIR_LINEAGE_INTERFACE),
                ("schema", REPAIR_LINEAGE_SCHEMA),
                ("store_version", self._store_version),
                ("authority", AUTHORITY_CLASS),
                ("cache_assurance_policy", CACHE_ASSURANCE_POLICY),
                ("write_authority_policy", WRITE_AUTHORITY_POLICY),
            ):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO repair_evidence_metadata(key, value)
                    VALUES (?, ?)
                    """,
                    [key, value],
                )
            self._connection = connection
            self._closed = False
            return self

    def close(self) -> None:
        with self._lock:
            connection = self._connection
            self._connection = None
            self._closed = True
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    def __enter__(self) -> "DatabaseRepairEvidenceStore":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require(self) -> Any:
        if not self.is_open or self._connection is None:
            raise DatabaseRepairEvidenceNotOpenError(
                "DatabaseRepairEvidenceStore is not open"
            )
        return self._connection

    def _commit_if_idle(self, connection: Any) -> None:
        if getattr(connection, "in_transaction", False):
            return
        commit = getattr(connection, "commit", None)
        if callable(commit):
            try:
                commit()
            except Exception:
                pass

    def metadata(self) -> dict[str, Any]:
        connection = self._require()
        with self._lock:
            rows = connection.execute(
                "SELECT key, value FROM repair_evidence_metadata"
            ).fetchall()
            meta = {str(row[0]): str(row[1]) for row in rows}
            meta["database_path"] = str(self._path)
            meta["is_open"] = True
            return meta

    # -- proof cache ---------------------------------------------------------

    def put_proof_cache_entry(
        self,
        entry: RepairProofCacheEntry,
        *,
        ttl_seconds: int | None = None,
    ) -> RepairProofCacheEntry:
        """Store a proof-cache entry. Does not grant repair authority."""

        connection = self._require()
        if not isinstance(entry, RepairProofCacheEntry):
            raise DatabaseRepairEvidenceIntegrityError(
                "entry must be RepairProofCacheEntry"
            )
        now = _now_ms()
        ttl = (
            self._default_ttl_seconds
            if ttl_seconds is None
            else int(ttl_seconds)
        )
        if ttl <= 0:
            raise DatabaseRepairEvidenceBoundsError(
                "ttl_seconds must be positive"
            )
        created = entry.created_at_ms or now
        expires = entry.expires_at_ms or (created + ttl * 1000)
        stored = RepairProofCacheEntry(
            key=entry.key,
            verdict=entry.verdict,
            assurance_level=entry.assurance_level,
            content_digest=entry.content_digest,
            body=dict(entry.body),
            created_at_ms=created,
            expires_at_ms=expires,
            poisoned=entry.poisoned,
            invalidated=entry.invalidated,
        )
        with self._lock:
            connection.execute(
                """
                INSERT OR REPLACE INTO repair_proof_cache(
                    cache_key_id, key_json, verdict, assurance_level,
                    content_digest, semantic_roots_json, body_json, poisoned,
                    invalidated, created_at_ms, expires_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    stored.key.key_id,
                    _canonical_json(stored.key.to_dict()),
                    stored.verdict,
                    stored.assurance_level,
                    stored.content_digest,
                    _canonical_json(dict(stored.key.semantic_roots)),
                    _canonical_json(dict(stored.body)),
                    1 if stored.poisoned else 0,
                    1 if stored.invalidated else 0,
                    stored.created_at_ms,
                    stored.expires_at_ms,
                ],
            )
            self._commit_if_idle(connection)
            return stored

    def lookup_proof_cache(
        self,
        key: RepairProofCacheKey,
        *,
        current_roots: Mapping[str, Any] | None = None,
        required_assurance: AssuranceLevel | str = AssuranceLevel.NONE,
        now_ms: int | None = None,
    ) -> ProofCacheLookupResult:
        """Lookup a proof-cache entry and re-derive applicability.

        A hit is only applicable when:
        * the entry is not poisoned, invalidated, or expired
        * semantic roots exactly match the current roots (rederived)
        * stored assurance satisfies the required level without promotion
        * verdict is a conclusive positive result
        """

        connection = self._require()
        if not isinstance(key, RepairProofCacheKey):
            raise DatabaseRepairEvidenceIntegrityError(
                "key must be RepairProofCacheKey"
            )
        roots = dict(current_roots if current_roots is not None else key.semantic_roots)
        now = int(now_ms if now_ms is not None else _now_ms())
        required = AssuranceLevel.coerce(required_assurance)

        with self._lock:
            row = connection.execute(
                """
                SELECT cache_key_id, key_json, verdict, assurance_level,
                       content_digest, semantic_roots_json, body_json,
                       poisoned, invalidated, created_at_ms, expires_at_ms
                FROM repair_proof_cache
                WHERE cache_key_id = ?
                """,
                [key.key_id],
            ).fetchone()
            if row is None:
                return ProofCacheLookupResult(
                    status=ProofCacheLookupStatus.MISS,
                    key=key,
                    reason=ProofCacheRejectReason.CACHE_MISS,
                    applicability_rederived=True,
                    applicable=False,
                )

            mapping = _row_mapping(row)
            if mapping:
                key_json = _load_json_object(mapping.get("key_json"))
                stored_roots = _load_json_object(
                    mapping.get("semantic_roots_json")
                )
                body = _load_json_object(mapping.get("body_json"))
                verdict = str(mapping.get("verdict") or "")
                assurance = str(mapping.get("assurance_level") or "none")
                digest = str(mapping.get("content_digest") or "")
                poisoned = bool(int(mapping.get("poisoned") or 0))
                invalidated = bool(int(mapping.get("invalidated") or 0))
                created_at_ms = int(mapping.get("created_at_ms") or 0)
                expires_at_ms = int(mapping.get("expires_at_ms") or 0)
            else:
                key_json = _load_json_object(row[1])
                stored_roots = _load_json_object(row[5])
                body = _load_json_object(row[6])
                verdict = str(row[2] or "")
                assurance = str(row[3] or "none")
                digest = str(row[4] or "")
                poisoned = bool(int(row[7] or 0))
                invalidated = bool(int(row[8] or 0))
                created_at_ms = int(row[9] or 0)
                expires_at_ms = int(row[10] or 0)

            stored_key = RepairProofCacheKey(
                subject_id=str(key_json.get("subject_id") or key.subject_id),
                semantic_roots=stored_roots,
                obligation_ids=tuple(key_json.get("obligation_ids") or ()),
                plan_id=str(key_json.get("plan_id") or ""),
                snapshot_id=str(key_json.get("snapshot_id") or ""),
                mutation_id=str(key_json.get("mutation_id") or ""),
                policy_id=str(key_json.get("policy_id") or ""),
                key_id=str(key_json.get("key_id") or key.key_id),
            )
            entry = RepairProofCacheEntry(
                key=stored_key,
                verdict=verdict,
                assurance_level=assurance,
                content_digest=digest,
                body=body,
                created_at_ms=created_at_ms,
                expires_at_ms=expires_at_ms,
                poisoned=poisoned,
                invalidated=invalidated,
            )

            # --- re-derive applicability (always) -------------------------
            if poisoned:
                return ProofCacheLookupResult(
                    status=ProofCacheLookupStatus.REJECTED,
                    key=key,
                    entry=entry,
                    reason=ProofCacheRejectReason.POISONED,
                    applicability_rederived=True,
                    applicable=False,
                )
            if invalidated:
                return ProofCacheLookupResult(
                    status=ProofCacheLookupStatus.REJECTED,
                    key=key,
                    entry=entry,
                    reason=ProofCacheRejectReason.INVALIDATED,
                    applicability_rederived=True,
                    applicable=False,
                )
            if expires_at_ms and now >= expires_at_ms:
                return ProofCacheLookupResult(
                    status=ProofCacheLookupStatus.REJECTED,
                    key=key,
                    entry=entry,
                    reason=ProofCacheRejectReason.EXPIRED,
                    applicability_rederived=True,
                    applicable=False,
                )
            if not stored_key.matches_roots(roots):
                return ProofCacheLookupResult(
                    status=ProofCacheLookupStatus.REJECTED,
                    key=key,
                    entry=entry,
                    reason=ProofCacheRejectReason.ROOT_MISMATCH,
                    applicability_rederived=True,
                    applicable=False,
                )
            # Key identity dimensions must still match current request.
            if (
                (key.plan_id and key.plan_id != stored_key.plan_id)
                or (key.snapshot_id and key.snapshot_id != stored_key.snapshot_id)
                or (key.mutation_id and key.mutation_id != stored_key.mutation_id)
            ):
                return ProofCacheLookupResult(
                    status=ProofCacheLookupStatus.REJECTED,
                    key=key,
                    entry=entry,
                    reason=ProofCacheRejectReason.ROOT_MISMATCH,
                    applicability_rederived=True,
                    applicable=False,
                )
            if not assurance_satisfies(entry.assurance_level, required):
                return ProofCacheLookupResult(
                    status=ProofCacheLookupStatus.REJECTED,
                    key=key,
                    entry=entry,
                    reason=ProofCacheRejectReason.ASSURANCE_PROMOTION,
                    applicability_rederived=True,
                    applicable=False,
                )
            if entry.verdict not in {"pass", "proved", "valid"}:
                return ProofCacheLookupResult(
                    status=ProofCacheLookupStatus.REJECTED,
                    key=key,
                    entry=entry,
                    reason=ProofCacheRejectReason.INCONCLUSIVE,
                    applicability_rederived=True,
                    applicable=False,
                )

            return ProofCacheLookupResult(
                status=ProofCacheLookupStatus.HIT,
                key=key,
                entry=entry,
                reason=None,
                applicability_rederived=True,
                applicable=True,
            )

    def invalidate_proof_cache(
        self,
        key_id: str,
        *,
        reason: str,
    ) -> str:
        """Poison/invalidate a proof-cache entry and record a tombstone."""

        connection = self._require()
        kid = _text(key_id, "key_id")
        why = _bounded_text(reason) or "invalidated"
        created_at = _utc_iso()
        invalidation_id = _identity(
            "repair-proof-invalidation",
            {"cache_key_id": kid, "reason": why, "created_at": created_at},
        )
        with self._lock:
            connection.execute(
                """
                UPDATE repair_proof_cache
                SET invalidated = 1
                WHERE cache_key_id = ?
                """,
                [kid],
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO repair_proof_invalidations(
                    invalidation_id, cache_key_id, reason, invalidated_at,
                    body_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    invalidation_id,
                    kid,
                    why,
                    created_at,
                    _canonical_json({"reason": why}),
                ],
            )
            self._commit_if_idle(connection)
            return invalidation_id

    # -- repair application --------------------------------------------------

    def apply_repair(self, request: RepairAttemptRequest) -> RepairLineage:
        """Apply a repair under an admitted plan or abstain/reject/roll back.

        Transactional sequence:
        1. gate on plan write admission, impact, AST freshness, operator class
        2. rederive proof-cache applicability when a key is supplied
        3. require code-and-logic fixed point for acceptance
        4. otherwise roll back and retain diagnostics
        5. persist lineage, events, and cache bindings together
        """

        connection = self._require()
        if not isinstance(request, RepairAttemptRequest):
            raise DatabaseRepairEvidenceIntegrityError(
                "request must be RepairAttemptRequest"
            )

        created_at = _utc_iso()
        reasons: list[str] = []
        events: list[dict[str, Any]] = []
        plan = request.plan

        with self._lock:
            # -- plan / write gates ----------------------------------------
            if plan.disposition is not PlanDisposition.ADMITTED:
                reasons.append(RepairRejectReason.PLAN_NOT_ADMITTED.value)
            if not plan.write_admitted:
                reasons.append(RepairRejectReason.PLAN_WRITE_NOT_ADMITTED.value)
            if not plan.bindings.ast_is_fresh:
                reasons.append(RepairRejectReason.STALE_AST.value)
            if not plan.impact_complete:
                reasons.append(RepairRejectReason.INCOMPLETE_IMPACT.value)
            if plan.blocks_automatic_repair:
                reasons.append(RepairRejectReason.BLOCKING_FRONTIER.value)
            if plan.disposition is PlanDisposition.PARTIAL:
                reasons.append(RepairRejectReason.PARTIAL_PLAN.value)
            if request.counterexample_ids or plan.counterexample_ids:
                reasons.append(RepairRejectReason.COUNTEREXAMPLE.value)

            # Operator class gates.
            if operator_requires_approval(request.operator_class):
                if not request.approval_granted:
                    reasons.append(RepairRejectReason.REQUIRES_APPROVAL.value)
            elif not operator_is_supported(request.operator_class):
                if not request.approval_granted:
                    reasons.append(RepairRejectReason.UNSUPPORTED_OPERATOR.value)

            # Scope: write paths must be subset of plan admitted write paths.
            admitted_paths = set(plan.bindings.admitted_write_paths)
            if request.write_paths:
                if not admitted_paths or set(request.write_paths) - admitted_paths:
                    reasons.append(RepairRejectReason.SCOPE_ESCAPE.value)

            # Worktree revalidation before effect.
            if (
                request.expected_worktree_digest
                and request.worktree_digest
                and request.expected_worktree_digest != request.worktree_digest
            ):
                reasons.append(RepairRejectReason.WORKTREE_MISMATCH.value)

            # -- proof cache rederive --------------------------------------
            proof_cache_audit: dict[str, Any] = {
                "consulted": request.proof_cache_key is not None,
                "applicable": False,
                "applicability_rederived": False,
            }
            if request.proof_cache_key is not None:
                lookup = self.lookup_proof_cache(
                    request.proof_cache_key,
                    current_roots=request.proof_cache_key.semantic_roots,
                )
                proof_cache_audit = lookup.to_dict()
                if not lookup.hit:
                    # Cache miss/reject is not fatal when other proof evidence
                    # is supplied via fixed_point.proof_receipt_id, but an
                    # explicit non-applicable hit claim cannot authorize write.
                    if lookup.status is ProofCacheLookupStatus.REJECTED:
                        reasons.append(
                            RepairRejectReason.PROOF_CACHE_INAPPLICABLE.value
                        )
                events.append(
                    {
                        "schema": REPAIR_EVENT_SCHEMA,
                        "event_kind": "proof_cache_lookup",
                        "reason": (
                            None
                            if lookup.reason is None
                            else lookup.reason.value
                        ),
                        "body": lookup.to_dict(),
                        "created_at": created_at,
                    }
                )

            # -- fixed point -----------------------------------------------
            fp = request.fixed_point
            if fp is None:
                fp_status = FixedPointStatus.PENDING
                fp_payload: dict[str, Any] = {}
                # Without fixed-point evidence, acceptance is impossible.
                if not reasons:
                    reasons.append(RepairRejectReason.FIXED_POINT_FAILED.value)
            else:
                fp_status = fp.status()
                fp_payload = fp.to_dict()
                if not fp.reached:
                    reasons.append(RepairRejectReason.FIXED_POINT_FAILED.value)
                # Worktree digest on fixed point must also match when both set.
                if (
                    fp.expected_worktree_digest
                    and fp.worktree_digest
                    and fp.expected_worktree_digest != fp.worktree_digest
                ):
                    reasons.append(RepairRejectReason.WORKTREE_MISMATCH.value)
                    fp_status = FixedPointStatus.FAILED

            if not request.obligation_ids and plan.admitted:
                # Plans may repair without explicit obligations only when
                # fixed point proves residual-free. Still record the note.
                if fp is not None and fp.residual_obligations:
                    reasons.append(RepairRejectReason.MISSING_OBLIGATIONS.value)

            # -- decide disposition ----------------------------------------
            deduped = tuple(dict.fromkeys(reasons))
            hard_reject = {
                RepairRejectReason.SCOPE_ESCAPE.value,
                RepairRejectReason.COUNTEREXAMPLE.value,
                RepairRejectReason.PLAN_NOT_ADMITTED.value,
            }
            needs_approval = {
                RepairRejectReason.REQUIRES_APPROVAL.value,
                RepairRejectReason.UNSUPPORTED_OPERATOR.value,
            }
            blocks_write = {
                RepairRejectReason.PLAN_WRITE_NOT_ADMITTED.value,
                RepairRejectReason.STALE_AST.value,
                RepairRejectReason.INCOMPLETE_IMPACT.value,
                RepairRejectReason.BLOCKING_FRONTIER.value,
                RepairRejectReason.PROOF_CACHE_INAPPLICABLE.value,
                RepairRejectReason.WORKTREE_MISMATCH.value,
                RepairRejectReason.PARTIAL_PLAN.value,
            }

            rolled_back = False
            write_committed = False
            disposition: RepairDisposition
            final_fp_status = fp_status

            # Write-blocking safety gates (incomplete impact, frontier, etc.)
            # take precedence over a generic plan_not_admitted hard reject so
            # incomplete plans abstain rather than hard-reject.
            if any(code in blocks_write for code in deduped):
                disposition = RepairDisposition.ABSTAINED
            elif any(code in hard_reject for code in deduped):
                disposition = RepairDisposition.REJECTED
            elif any(code in needs_approval for code in deduped):
                disposition = RepairDisposition.REQUIRES_APPROVAL
            elif RepairRejectReason.FIXED_POINT_FAILED.value in deduped:
                # Attempted repair that failed fixed point must roll back.
                disposition = RepairDisposition.ROLLED_BACK
                rolled_back = True
                final_fp_status = FixedPointStatus.ROLLED_BACK
            elif (
                not deduped
                and fp is not None
                and fp.reached
                and plan.write_admitted
                and plan.disposition is PlanDisposition.ADMITTED
            ):
                disposition = RepairDisposition.ACCEPTED
                write_committed = True
                final_fp_status = FixedPointStatus.REACHED
            else:
                disposition = RepairDisposition.ABSTAINED

            mutation_id = (
                request.mutation_id
                or plan.bindings.mutation_id
                or ""
            )
            lineage_id = _identity(
                "repair-lineage",
                {
                    "plan_id": plan.plan_id,
                    "task_id": plan.task_id,
                    "attempt_id": plan.attempt_id,
                    "mutation_id": mutation_id,
                    "operator_class": request.operator_class,
                    "write_paths": list(request.write_paths),
                    "created_at": created_at,
                },
            )

            events.append(
                {
                    "schema": REPAIR_EVENT_SCHEMA,
                    "event_kind": "repair_decision",
                    "reason": disposition.value,
                    "body": {
                        "disposition": disposition.value,
                        "reasons": list(deduped),
                        "write_committed": write_committed,
                        "rolled_back": rolled_back,
                        "fixed_point_status": final_fp_status.value
                        if isinstance(final_fp_status, FixedPointStatus)
                        else str(final_fp_status),
                    },
                    "created_at": created_at,
                }
            )

            lineage = RepairLineage(
                lineage_id=lineage_id,
                plan_id=plan.plan_id,
                task_id=plan.task_id,
                attempt_id=plan.attempt_id,
                mutation_id=mutation_id,
                snapshot_id=plan.bindings.snapshot_id,
                worktree_id=request.worktree_id,
                operator_class=request.operator_class,
                candidate_id=request.candidate_id,
                disposition=disposition,
                write_paths=request.write_paths,
                obligation_ids=request.obligation_ids,
                fixed_point_status=final_fp_status,
                fixed_point=fp_payload,
                write_committed=write_committed,
                rolled_back=rolled_back,
                reasons=deduped,
                proof_cache=proof_cache_audit,
                events=tuple(events),
                created_at=created_at,
                finalized_at=created_at
                if disposition
                is not RepairDisposition.PENDING
                else "",
            )

            # Persist lineage + events (+ optional rollback) transactionally.
            self._persist_lineage(connection, lineage)
            for event in events:
                self._persist_event(
                    connection,
                    lineage_id=lineage.lineage_id,
                    event_kind=str(event.get("event_kind") or "event"),
                    reason=str(event.get("reason") or ""),
                    body=dict(event.get("body") or event),
                    created_at=str(event.get("created_at") or created_at),
                )
            if rolled_back:
                rollback = RepairRollbackReceipt(
                    rollback_id=_identity(
                        "repair-rollback",
                        {
                            "lineage_id": lineage.lineage_id,
                            "paths": list(request.write_paths),
                            "created_at": created_at,
                        },
                    ),
                    lineage_id=lineage.lineage_id,
                    status=RollbackStatus.VERIFIED,
                    reason=RepairRejectReason.FIXED_POINT_FAILED.value,
                    restored_paths=request.write_paths,
                    body={
                        "fixed_point": fp_payload,
                        "reasons": list(deduped),
                    },
                    created_at=created_at,
                )
                self._persist_rollback(connection, rollback)
                events_with_rollback = list(lineage.events) + [
                    {
                        "schema": REPAIR_EVENT_SCHEMA,
                        "event_kind": "rollback",
                        "reason": rollback.reason,
                        "body": rollback.to_dict(),
                        "created_at": created_at,
                    }
                ]
                self._persist_event(
                    connection,
                    lineage_id=lineage.lineage_id,
                    event_kind="rollback",
                    reason=rollback.reason,
                    body=rollback.to_dict(),
                    created_at=created_at,
                )
                lineage = RepairLineage(
                    lineage_id=lineage.lineage_id,
                    plan_id=lineage.plan_id,
                    task_id=lineage.task_id,
                    attempt_id=lineage.attempt_id,
                    mutation_id=lineage.mutation_id,
                    snapshot_id=lineage.snapshot_id,
                    worktree_id=lineage.worktree_id,
                    operator_class=lineage.operator_class,
                    candidate_id=lineage.candidate_id,
                    disposition=lineage.disposition,
                    write_paths=lineage.write_paths,
                    obligation_ids=lineage.obligation_ids,
                    fixed_point_status=lineage.fixed_point_status,
                    fixed_point=dict(lineage.fixed_point),
                    write_committed=False,
                    rolled_back=True,
                    reasons=lineage.reasons,
                    proof_cache=dict(lineage.proof_cache),
                    events=tuple(events_with_rollback),
                    lineage_digest=lineage.lineage_digest,
                    created_at=lineage.created_at,
                    finalized_at=created_at,
                )
                self._persist_lineage(connection, lineage)

            self._commit_if_idle(connection)
            return lineage

    def get_lineage(self, lineage_id: str) -> RepairLineage | None:
        connection = self._require()
        lid = _text(lineage_id, "lineage_id")
        with self._lock:
            row = connection.execute(
                """
                SELECT lineage_id, plan_id, task_id, attempt_id, mutation_id,
                       snapshot_id, worktree_id, disposition, fixed_point_status,
                       write_committed, rolled_back, reason, lineage_digest,
                       body_json, created_at, finalized_at
                FROM repair_lineages
                WHERE lineage_id = ?
                """,
                [lid],
            ).fetchone()
            if row is None:
                return None
            return self._lineage_from_row(connection, row)

    def list_lineages(
        self,
        *,
        plan_id: str = "",
        task_id: str = "",
        disposition: RepairDisposition | str | None = None,
        limit: int = 100,
    ) -> tuple[RepairLineage, ...]:
        connection = self._require()
        limit = max(1, min(int(limit), 10_000))
        with self._lock:
            clauses: list[str] = []
            params: list[Any] = []
            if plan_id:
                clauses.append("plan_id = ?")
                params.append(_text(plan_id, "plan_id"))
            if task_id:
                clauses.append("task_id = ?")
                params.append(_text(task_id, "task_id"))
            if disposition is not None:
                disp = (
                    disposition.value
                    if isinstance(disposition, RepairDisposition)
                    else str(disposition).strip().casefold()
                )
                clauses.append("disposition = ?")
                params.append(disp)
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            params.append(limit)
            rows = connection.execute(
                f"""
                SELECT lineage_id, plan_id, task_id, attempt_id, mutation_id,
                       snapshot_id, worktree_id, disposition, fixed_point_status,
                       write_committed, rolled_back, reason, lineage_digest,
                       body_json, created_at, finalized_at
                FROM repair_lineages
                {where}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return tuple(
                self._lineage_from_row(connection, row) for row in rows
            )

    def get_rollback(self, lineage_id: str) -> RepairRollbackReceipt | None:
        connection = self._require()
        lid = _text(lineage_id, "lineage_id")
        with self._lock:
            row = connection.execute(
                """
                SELECT rollback_id, lineage_id, status, reason,
                       restored_paths_json, body_json, created_at
                FROM repair_rollbacks
                WHERE lineage_id = ?
                """,
                [lid],
            ).fetchone()
            if row is None:
                return None
            mapping = _row_mapping(row)
            if mapping:
                return RepairRollbackReceipt(
                    rollback_id=str(mapping["rollback_id"]),
                    lineage_id=str(mapping["lineage_id"]),
                    status=str(mapping["status"]),
                    reason=str(mapping.get("reason") or ""),
                    restored_paths=tuple(
                        _load_json_list(mapping.get("restored_paths_json"))
                    ),
                    body=_load_json_object(mapping.get("body_json")),
                    created_at=str(mapping.get("created_at") or ""),
                )
            return RepairRollbackReceipt(
                rollback_id=str(row[0]),
                lineage_id=str(row[1]),
                status=str(row[2]),
                reason=str(row[3] or ""),
                restored_paths=tuple(_load_json_list(row[4])),
                body=_load_json_object(row[5]),
                created_at=str(row[6] or ""),
            )

    # -- persistence helpers -------------------------------------------------

    def _persist_lineage(self, connection: Any, lineage: RepairLineage) -> None:
        body = lineage.to_dict()
        encoded = _canonical_json(body)
        if len(encoded.encode("utf-8")) > MAX_BODY_JSON_BYTES:
            raise DatabaseRepairEvidenceBoundsError(
                f"lineage body exceeds {MAX_BODY_JSON_BYTES} bytes"
            )
        connection.execute(
            """
            INSERT OR REPLACE INTO repair_lineages(
                lineage_id, plan_id, task_id, attempt_id, mutation_id,
                snapshot_id, worktree_id, disposition, fixed_point_status,
                write_committed, rolled_back, reason, lineage_digest,
                body_json, created_at, finalized_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                lineage.lineage_id,
                lineage.plan_id,
                lineage.task_id,
                lineage.attempt_id,
                lineage.mutation_id,
                lineage.snapshot_id,
                lineage.worktree_id,
                lineage.disposition.value
                if isinstance(lineage.disposition, RepairDisposition)
                else str(lineage.disposition),
                lineage.fixed_point_status.value
                if isinstance(lineage.fixed_point_status, FixedPointStatus)
                else str(lineage.fixed_point_status),
                1 if lineage.write_committed else 0,
                1 if lineage.rolled_back else 0,
                ",".join(lineage.reasons),
                lineage.lineage_digest,
                encoded,
                lineage.created_at,
                lineage.finalized_at,
            ],
        )

    def _persist_event(
        self,
        connection: Any,
        *,
        lineage_id: str,
        event_kind: str,
        reason: str,
        body: Mapping[str, Any],
        created_at: str,
    ) -> None:
        event_id = _identity(
            "repair-event",
            {
                "lineage_id": lineage_id,
                "event_kind": event_kind,
                "reason": reason,
                "created_at": created_at,
                "body": dict(body),
            },
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO repair_events(
                event_id, lineage_id, event_kind, reason, body_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                event_id,
                lineage_id,
                event_kind,
                _bounded_text(reason),
                _canonical_json(dict(body)),
                created_at,
            ],
        )

    def _persist_rollback(
        self, connection: Any, rollback: RepairRollbackReceipt
    ) -> None:
        connection.execute(
            """
            INSERT INTO repair_rollbacks(
                rollback_id, lineage_id, status, reason, restored_paths_json,
                body_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (lineage_id) DO UPDATE SET
                rollback_id = excluded.rollback_id,
                status = excluded.status,
                reason = excluded.reason,
                restored_paths_json = excluded.restored_paths_json,
                body_json = excluded.body_json,
                created_at = excluded.created_at
            """,
            [
                rollback.rollback_id,
                rollback.lineage_id,
                rollback.status.value
                if isinstance(rollback.status, RollbackStatus)
                else str(rollback.status),
                rollback.reason,
                _canonical_json(list(rollback.restored_paths)),
                _canonical_json(dict(rollback.body)),
                rollback.created_at,
            ],
        )

    def _lineage_from_row(self, connection: Any, row: Any) -> RepairLineage:
        mapping = _row_mapping(row)
        if mapping:
            body = _load_json_object(mapping.get("body_json"))
            lineage_id = str(mapping["lineage_id"])
        else:
            body = _load_json_object(row[13])
            lineage_id = str(row[0])

        event_rows = connection.execute(
            """
            SELECT event_id, event_kind, reason, body_json, created_at
            FROM repair_events
            WHERE lineage_id = ?
            ORDER BY created_at
            """,
            [lineage_id],
        ).fetchall()
        events: list[dict[str, Any]] = []
        for erow in event_rows:
            em = _row_mapping(erow)
            if em:
                events.append(
                    {
                        "event_id": str(em["event_id"]),
                        "event_kind": str(em["event_kind"]),
                        "reason": str(em.get("reason") or ""),
                        "body": _load_json_object(em.get("body_json")),
                        "created_at": str(em.get("created_at") or ""),
                    }
                )
            else:
                events.append(
                    {
                        "event_id": str(erow[0]),
                        "event_kind": str(erow[1]),
                        "reason": str(erow[2] or ""),
                        "body": _load_json_object(erow[3]),
                        "created_at": str(erow[4] or ""),
                    }
                )

        return RepairLineage(
            lineage_id=lineage_id,
            plan_id=str(body.get("plan_id") or ""),
            task_id=str(body.get("task_id") or ""),
            attempt_id=str(body.get("attempt_id") or ""),
            mutation_id=str(body.get("mutation_id") or ""),
            snapshot_id=str(body.get("snapshot_id") or ""),
            worktree_id=str(body.get("worktree_id") or ""),
            operator_class=str(body.get("operator_class") or ""),
            candidate_id=str(body.get("candidate_id") or ""),
            disposition=str(body.get("disposition") or "rejected"),
            write_paths=tuple(body.get("write_paths") or ()),
            obligation_ids=tuple(body.get("obligation_ids") or ()),
            fixed_point_status=str(
                body.get("fixed_point_status") or "pending"
            ),
            fixed_point=dict(body.get("fixed_point") or {}),
            write_committed=bool(body.get("write_committed")),
            rolled_back=bool(body.get("rolled_back")),
            reasons=tuple(body.get("reasons") or ()),
            proof_cache=dict(body.get("proof_cache") or {}),
            events=tuple(events or body.get("events") or ()),
            lineage_digest=str(body.get("lineage_digest") or ""),
            created_at=str(body.get("created_at") or ""),
            finalized_at=str(body.get("finalized_at") or ""),
        )


def open_database_repair_evidence_store(
    database_path: Path | str,
    *,
    store_version: str = DEFAULT_STORE_VERSION,
    default_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
) -> DatabaseRepairEvidenceStore:
    """Open a RepairLineage@1 evidence store."""

    return DatabaseRepairEvidenceStore(
        database_path,
        store_version=store_version,
        default_ttl_seconds=default_ttl_seconds,
    ).open()


# Back-compat alias used by some call sites / docs.
RepairLineageStore = DatabaseRepairEvidenceStore


__all__ = [
    "AUTHORITY_CLASS",
    "AssuranceLevel",
    "CACHE_ASSURANCE_POLICY",
    "DatabaseRepairEvidenceAdmissionError",
    "DatabaseRepairEvidenceBoundsError",
    "DatabaseRepairEvidenceError",
    "DatabaseRepairEvidenceIntegrityError",
    "DatabaseRepairEvidenceNotOpenError",
    "DatabaseRepairEvidenceStore",
    "DuckDBUnavailableError",
    "FixedPointEvidence",
    "FixedPointStatus",
    "ProofCacheLookupResult",
    "ProofCacheLookupStatus",
    "ProofCacheRejectReason",
    "REPAIR_LINEAGE_INTERFACE",
    "REPAIR_LINEAGE_SCHEMA",
    "RepairAttemptRequest",
    "RepairDisposition",
    "RepairLineage",
    "RepairLineageStore",
    "RepairProofCacheEntry",
    "RepairProofCacheKey",
    "RepairRejectReason",
    "RepairRollbackReceipt",
    "RollbackStatus",
    "WRITE_AUTHORITY_POLICY",
    "assurance_satisfies",
    "duckdb_available",
    "open_database_repair_evidence_store",
]
