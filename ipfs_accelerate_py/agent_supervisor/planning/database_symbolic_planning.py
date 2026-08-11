"""Database-bound symbolic planning over AST, mutation, and impact state.

DQP-024 / Interfaces: ``DatabaseSymbolicPlanner@1``
===================================================

Composes existing mutation ledger, impact-graph, and repair-operator surfaces
into a fail-closed planner. Deterministic discovery and impact queries always
precede candidate synthesis. Admitted plans reference exact AST, symbol, and
mutation identities. This module never grants write or completion authority;
it only admits (or abstains from) symbolic plans.

Acceptance properties
---------------------
* LLM proposals cannot invent scope or semantics outside the admitted plan.
* Stale AST identity or incomplete / blocking impact prevents write admission.
* Unsupported operator classes require approval or abstain (never silent auto).
* Candidate reuse is exact-identity reuse, not semantic nomination.

Evidence subset: candidate reuse, stale AST, counterexample, partial plan,
unsupported operator, fixed point (via repair evidence), proof invalidation,
abstention.

Cold import of this module performs no filesystem, database, network,
provider, or process action. Opening a planner is the first I/O boundary.
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

from ..analysis.database_impact_graph import (
    ImpactClosure,
    ImpactCompleteness,
    duckdb_available as impact_duckdb_available,
)
from ..analysis.mutation_ledger import (
    MutationSet,
    MutationStatus,
    duckdb_available as mutation_duckdb_available,
)
from ..task_sources.duckdb_state import open_duckdb_connection

# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DATABASE_SYMBOLIC_PLANNER_INTERFACE: Final[str] = "DatabaseSymbolicPlanner@1"
DATABASE_SYMBOLIC_PLANNER_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-symbolic-planner@1"
)
SYMBOLIC_PLAN_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-symbolic-plan@1"
)
SYMBOLIC_PLAN_REQUEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-symbolic-plan-request@1"
)
SYMBOLIC_CANDIDATE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-symbolic-candidate@1"
)
SYMBOLIC_PLAN_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-symbolic-plan-receipt@1"
)
LLM_PROPOSAL_AUDIT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/database-symbolic-llm-proposal-audit@1"
)

DEFAULT_PLANNER_VERSION: Final[str] = "database-symbolic-planner@1"
AUTHORITY_CLASS: Final[str] = "derived_evidence"
# Plans never grant mutation / completion authority.
PLAN_AUTHORITY_POLICY: Final[str] = "no_write_authority"
LLM_OUTPUT_POLICY: Final[str] = "nomination_only"

MAX_PATH_BYTES: Final[int] = 4_096
MAX_TEXT_BYTES: Final[int] = 1_024
MAX_ID_BYTES: Final[int] = 512
MAX_BODY_JSON_BYTES: Final[int] = 262_144
MAX_SEEDS: Final[int] = 4_096
MAX_PATHS: Final[int] = 1_024
MAX_CANDIDATES: Final[int] = 64
MAX_OPERATORS: Final[int] = 32
MAX_OBLIGATIONS: Final[int] = 256
MAX_REASON_CODES: Final[int] = 64
MAX_AST_IDS: Final[int] = 4_096
MAX_SYMBOL_IDS: Final[int] = 4_096

# Supported operator classes that may auto-admit without approval.
SUPPORTED_OPERATOR_CLASSES: Final[frozenset[str]] = frozenset(
    {
        "constraint_rewrite",
        "equality_rewrite",
        "enumerative",
        "cegis",
        "doctor_deterministic",
        "formatting_only",
        "symbol_rename",
        "import_fix",
        "signature_align",
        "test_update",
    }
)

# Operator classes that always require human approval / abstain.
APPROVAL_REQUIRED_OPERATOR_CLASSES: Final[frozenset[str]] = frozenset(
    {
        "dynamic_dispatch",
        "cross_language",
        "generated_code",
        "unsupported_language",
        "llm_freeform",
        "semantic_invention",
        "architecture_rewrite",
        "dependency_add",
        "authority_override",
    }
)

_SAFE_ID = re.compile(r"^[^\x00\r\n\t]{1,512}$")
_SCOPE_WIDEN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "extra_paths",
        "new_dependencies",
        "dependency_paths",
        "write_paths",
        "requested_write_paths",
        "authority_override",
        "policy_override",
        "completion_claim",
        "semantic_change",
        "meaning_change",
        "import_additions",
        "extra_imports",
        "extra_files",
        "extra_symbols",
        "extra_ast_ids",
        "scope_paths",
        "admitted",
        "accepted",
        "allowed",
        "approved",
        "valid",
        "passed",
        "completed",
    }
)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_BOOKKEEPING_SQL: Final[str] = """
CREATE TABLE IF NOT EXISTS symbolic_planner_metadata (
    key VARCHAR PRIMARY KEY,
    value VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS symbolic_plans (
    plan_id VARCHAR PRIMARY KEY,
    task_id VARCHAR NOT NULL,
    attempt_id VARCHAR NOT NULL DEFAULT '',
    snapshot_id VARCHAR NOT NULL,
    tree_id VARCHAR NOT NULL DEFAULT '',
    repository_id VARCHAR NOT NULL DEFAULT '',
    mutation_id VARCHAR NOT NULL DEFAULT '',
    impact_query_id VARCHAR NOT NULL DEFAULT '',
    impact_revision_id VARCHAR NOT NULL DEFAULT '',
    disposition VARCHAR NOT NULL,
    write_admitted INTEGER NOT NULL DEFAULT 0,
    reason VARCHAR NOT NULL DEFAULT '',
    plan_digest VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS symbolic_plans_task_idx
    ON symbolic_plans(task_id, attempt_id);
CREATE INDEX IF NOT EXISTS symbolic_plans_snapshot_idx
    ON symbolic_plans(snapshot_id, disposition);

CREATE TABLE IF NOT EXISTS symbolic_candidates (
    candidate_id VARCHAR PRIMARY KEY,
    plan_id VARCHAR NOT NULL,
    operator_class VARCHAR NOT NULL,
    disposition VARCHAR NOT NULL,
    reused INTEGER NOT NULL DEFAULT 0,
    source_candidate_id VARCHAR NOT NULL DEFAULT '',
    symbol_ids_json VARCHAR NOT NULL,
    ast_mutation_ids_json VARCHAR NOT NULL,
    write_paths_json VARCHAR NOT NULL,
    body_json VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS symbolic_candidates_plan_idx
    ON symbolic_candidates(plan_id);

CREATE TABLE IF NOT EXISTS symbolic_plan_events (
    event_id VARCHAR PRIMARY KEY,
    plan_id VARCHAR NOT NULL,
    event_kind VARCHAR NOT NULL,
    reason VARCHAR NOT NULL DEFAULT '',
    body_json VARCHAR NOT NULL,
    created_at VARCHAR NOT NULL
);
CREATE INDEX IF NOT EXISTS symbolic_plan_events_plan_idx
    ON symbolic_plan_events(plan_id, created_at);
"""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DatabaseSymbolicPlanningError(RuntimeError):
    """Base error for database symbolic planning failures."""


class DatabaseSymbolicPlanningNotOpenError(DatabaseSymbolicPlanningError):
    """Operation requires an open planner store."""


class DatabaseSymbolicPlanningIntegrityError(
    DatabaseSymbolicPlanningError, ValueError
):
    """Identity, binding, or payload integrity failure."""


class DatabaseSymbolicPlanningBoundsError(
    DatabaseSymbolicPlanningError, ValueError
):
    """A resource or payload bound was exceeded."""


class DatabaseSymbolicPlanningAdmissionError(DatabaseSymbolicPlanningError):
    """Plan or write admission refused fail-closed."""


class DuckDBUnavailableError(DatabaseSymbolicPlanningError):
    """Optional DuckDB dependency is not installed."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class PlanDisposition(str, Enum):
    """Closed outcomes for symbolic plan admission."""

    ADMITTED = "admitted"
    ABSTAINED = "abstained"
    REJECTED = "rejected"
    PARTIAL = "partial"
    REQUIRES_APPROVAL = "requires_approval"


class CandidateDisposition(str, Enum):
    """Closed outcomes for one symbolic candidate."""

    ADMITTED = "admitted"
    REUSED = "reused"
    REJECTED = "rejected"
    ABSTAINED = "abstained"
    REQUIRES_APPROVAL = "requires_approval"
    OUT_OF_SCOPE = "out_of_scope"
    STALE = "stale"
    UNSUPPORTED = "unsupported"


class PlanRejectReason(str, Enum):
    """Stable machine-readable rejection / abstention codes."""

    STALE_AST = "stale_ast"
    STALE_SNAPSHOT = "stale_snapshot"
    INCOMPLETE_IMPACT = "incomplete_impact"
    BLOCKING_FRONTIER = "blocking_frontier"
    MISSING_MUTATION = "missing_mutation"
    MISSING_IMPACT = "missing_impact"
    SCOPE_INVENTION = "scope_invention"
    SEMANTIC_INVENTION = "semantic_invention"
    UNSUPPORTED_OPERATOR = "unsupported_operator"
    REQUIRES_APPROVAL = "requires_approval"
    COUNTEREXAMPLE = "counterexample"
    PARTIAL_PLAN = "partial_plan"
    EMPTY_SEEDS = "empty_seeds"
    PATH_ESCAPE = "path_escape"
    PROVIDER_CLAIM = "provider_claim_rejected"
    MUTATION_NOT_ACCEPTED = "mutation_not_accepted"
    AST_BINDING_MISMATCH = "ast_binding_mismatch"
    SYMBOL_BINDING_MISMATCH = "symbol_binding_mismatch"
    CANDIDATE_REUSE_MISMATCH = "candidate_reuse_mismatch"
    NO_CANDIDATES = "no_candidates"
    MALFORMED_INPUT = "malformed_input"


class DiscoveryStage(str, Enum):
    """Ordered discovery stages that precede candidate synthesis."""

    BINDINGS = "bindings"
    MUTATION_DISCOVERY = "mutation_discovery"
    IMPACT_QUERY = "impact_query"
    AST_FRESHNESS = "ast_freshness"
    CANDIDATE_SYNTHESIS = "candidate_synthesis"
    LLM_SCOPE_AUDIT = "llm_scope_audit"
    ADMISSION = "admission"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def duckdb_available() -> bool:
    """Return whether DuckDB and required sibling stores can be used."""

    try:
        import duckdb  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return mutation_duckdb_available() and impact_duckdb_available()


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _text(value: Any, name: str, *, required: bool = True) -> str:
    text = str(value or "").strip()
    if "\x00" in text:
        raise DatabaseSymbolicPlanningIntegrityError(f"{name} contains NUL")
    if required and not text:
        raise DatabaseSymbolicPlanningIntegrityError(f"{name} is required")
    if len(text.encode("utf-8")) > MAX_ID_BYTES and name.endswith(
        ("_id", "id", "query_id", "revision_id")
    ):
        raise DatabaseSymbolicPlanningBoundsError(
            f"{name} exceeds {MAX_ID_BYTES} bytes"
        )
    if text and not _SAFE_ID.fullmatch(text) and name.endswith(
        ("_id", "id", "query_id", "revision_id", "task_id", "attempt_id")
    ):
        # Allow colon-prefixed identities used throughout the supervisor.
        if not re.fullmatch(r"^[A-Za-z0-9:._/@+\-]{1,512}$", text):
            raise DatabaseSymbolicPlanningIntegrityError(
                f"{name} must be a compact identifier"
            )
    return text


def _bounded_text(value: Any, maximum: int = MAX_TEXT_BYTES) -> str:
    text = str(value or "")
    if "\x00" in text:
        raise DatabaseSymbolicPlanningIntegrityError("text contains NUL")
    encoded = text.encode("utf-8")
    if len(encoded) > maximum:
        raise DatabaseSymbolicPlanningBoundsError(
            f"text exceeds {maximum} bytes"
        )
    return text


def _repo_path(value: Any) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        raise DatabaseSymbolicPlanningIntegrityError("path is required")
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or raw != path.as_posix():
        raise DatabaseSymbolicPlanningIntegrityError(
            "path must be a normalized repository-relative path"
        )
    if len(raw.encode("utf-8")) > MAX_PATH_BYTES:
        raise DatabaseSymbolicPlanningBoundsError(
            f"path exceeds {MAX_PATH_BYTES} bytes"
        )
    return raw


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


def _ids(
    values: Sequence[Any] | None,
    name: str,
    *,
    required: bool = False,
    maximum: int = MAX_SEEDS,
) -> tuple[str, ...]:
    if values is None:
        items: list[Any] = []
    elif isinstance(values, (str, bytes, bytearray)):
        raise DatabaseSymbolicPlanningIntegrityError(f"{name} must be a sequence")
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
        raise DatabaseSymbolicPlanningIntegrityError(f"{name} must not be empty")
    if len(result) > maximum:
        raise DatabaseSymbolicPlanningBoundsError(
            f"{name} exceeds {maximum} items"
        )
    return tuple(result)


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
        raise DatabaseSymbolicPlanningIntegrityError(f"{name} must be a sequence")
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
        raise DatabaseSymbolicPlanningIntegrityError(f"{name} must not be empty")
    if len(result) > maximum:
        raise DatabaseSymbolicPlanningBoundsError(
            f"{name} exceeds {maximum} paths"
        )
    return tuple(sorted(result))


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


def _load_json_list(text: Any) -> list[Any]:
    if not text:
        return []
    try:
        value = json.loads(str(text))
    except (TypeError, ValueError) as exc:
        raise DatabaseSymbolicPlanningIntegrityError(
            "stored JSON list is corrupted"
        ) from exc
    if not isinstance(value, list):
        raise DatabaseSymbolicPlanningIntegrityError(
            "stored JSON list must be an array"
        )
    return value


def _load_json_object(text: Any) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(str(text))
    except (TypeError, ValueError) as exc:
        raise DatabaseSymbolicPlanningIntegrityError(
            "stored JSON object is corrupted"
        ) from exc
    if not isinstance(value, dict):
        raise DatabaseSymbolicPlanningIntegrityError(
            "stored JSON must be an object"
        )
    return value


def normalize_operator_class(value: Any) -> str:
    """Normalize an operator class token to a compact snake_case id."""

    text = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    if not text:
        raise DatabaseSymbolicPlanningIntegrityError("operator_class is required")
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", text):
        raise DatabaseSymbolicPlanningIntegrityError(
            f"unsupported operator_class shape: {value!r}"
        )
    return text


def operator_requires_approval(operator_class: str) -> bool:
    return normalize_operator_class(operator_class) in APPROVAL_REQUIRED_OPERATOR_CLASSES


def operator_is_supported(operator_class: str) -> bool:
    token = normalize_operator_class(operator_class)
    if token in APPROVAL_REQUIRED_OPERATOR_CLASSES:
        return False
    return token in SUPPORTED_OPERATOR_CLASSES


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolicPlanBindings:
    """Exact AST / mutation / impact identities a plan must bind."""

    snapshot_id: str
    tree_id: str = ""
    repository_id: str = ""
    mutation_id: str = ""
    impact_query_id: str = ""
    impact_revision_id: str = ""
    parser_id: str = ""
    policy_id: str = ""
    schema_id: str = ""
    seed_symbols: tuple[str, ...] = ()
    ast_mutation_ids: tuple[str, ...] = ()
    symbol_ids: tuple[str, ...] = ()
    admitted_write_paths: tuple[str, ...] = ()
    admitted_read_paths: tuple[str, ...] = ()
    current_ast_digest: str = ""
    expected_ast_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "snapshot_id", _text(self.snapshot_id, "snapshot_id")
        )
        for name in (
            "tree_id",
            "repository_id",
            "mutation_id",
            "impact_query_id",
            "impact_revision_id",
            "parser_id",
            "policy_id",
            "schema_id",
            "current_ast_digest",
            "expected_ast_digest",
        ):
            object.__setattr__(
                self, name, _text(getattr(self, name), name, required=False)
            )
        object.__setattr__(
            self,
            "seed_symbols",
            _ids(self.seed_symbols, "seed_symbols", maximum=MAX_SEEDS),
        )
        object.__setattr__(
            self,
            "ast_mutation_ids",
            _ids(self.ast_mutation_ids, "ast_mutation_ids", maximum=MAX_AST_IDS),
        )
        object.__setattr__(
            self,
            "symbol_ids",
            _ids(self.symbol_ids, "symbol_ids", maximum=MAX_SYMBOL_IDS),
        )
        object.__setattr__(
            self,
            "admitted_write_paths",
            _paths(self.admitted_write_paths, "admitted_write_paths"),
        )
        object.__setattr__(
            self,
            "admitted_read_paths",
            _paths(self.admitted_read_paths, "admitted_read_paths"),
        )

    @property
    def ast_is_fresh(self) -> bool:
        if not self.expected_ast_digest:
            return True
        return bool(self.current_ast_digest) and (
            self.current_ast_digest == self.expected_ast_digest
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "tree_id": self.tree_id,
            "repository_id": self.repository_id,
            "mutation_id": self.mutation_id,
            "impact_query_id": self.impact_query_id,
            "impact_revision_id": self.impact_revision_id,
            "parser_id": self.parser_id,
            "policy_id": self.policy_id,
            "schema_id": self.schema_id,
            "seed_symbols": list(self.seed_symbols),
            "ast_mutation_ids": list(self.ast_mutation_ids),
            "symbol_ids": list(self.symbol_ids),
            "admitted_write_paths": list(self.admitted_write_paths),
            "admitted_read_paths": list(self.admitted_read_paths),
            "current_ast_digest": self.current_ast_digest,
            "expected_ast_digest": self.expected_ast_digest,
            "ast_is_fresh": self.ast_is_fresh,
        }


@dataclass(frozen=True)
class SymbolicCandidateSpec:
    """One deterministic or residual candidate proposed for a plan."""

    operator_class: str
    write_paths: tuple[str, ...] = ()
    symbol_ids: tuple[str, ...] = ()
    ast_mutation_ids: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    source: str = "deterministic"
    source_candidate_id: str = ""
    body: Mapping[str, Any] = field(default_factory=dict)
    candidate_id: str = ""

    def __post_init__(self) -> None:
        op = normalize_operator_class(self.operator_class)
        object.__setattr__(self, "operator_class", op)
        object.__setattr__(
            self, "write_paths", _paths(self.write_paths, "write_paths")
        )
        object.__setattr__(
            self, "symbol_ids", _ids(self.symbol_ids, "symbol_ids", maximum=MAX_SYMBOL_IDS)
        )
        object.__setattr__(
            self,
            "ast_mutation_ids",
            _ids(self.ast_mutation_ids, "ast_mutation_ids", maximum=MAX_AST_IDS),
        )
        object.__setattr__(
            self,
            "obligation_ids",
            _ids(self.obligation_ids, "obligation_ids", maximum=MAX_OBLIGATIONS),
        )
        object.__setattr__(
            self, "source", _text(self.source or "deterministic", "source")
        )
        object.__setattr__(
            self,
            "source_candidate_id",
            _text(self.source_candidate_id, "source_candidate_id", required=False),
        )
        body = dict(self.body or {})
        encoded = _canonical_json(body).encode("utf-8")
        if len(encoded) > MAX_BODY_JSON_BYTES:
            raise DatabaseSymbolicPlanningBoundsError(
                f"candidate body exceeds {MAX_BODY_JSON_BYTES} bytes"
            )
        object.__setattr__(self, "body", MappingProxyType(body))
        if not self.candidate_id:
            object.__setattr__(
                self,
                "candidate_id",
                _identity(
                    "symbolic-candidate",
                    {
                        "schema": SYMBOLIC_CANDIDATE_SCHEMA,
                        "operator_class": op,
                        "write_paths": list(self.write_paths),
                        "symbol_ids": list(self.symbol_ids),
                        "ast_mutation_ids": list(self.ast_mutation_ids),
                        "obligation_ids": list(self.obligation_ids),
                        "source": self.source,
                        "source_candidate_id": self.source_candidate_id,
                        "body": body,
                    },
                ),
            )
        else:
            object.__setattr__(
                self, "candidate_id", _text(self.candidate_id, "candidate_id")
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SYMBOLIC_CANDIDATE_SCHEMA,
            "candidate_id": self.candidate_id,
            "operator_class": self.operator_class,
            "write_paths": list(self.write_paths),
            "symbol_ids": list(self.symbol_ids),
            "ast_mutation_ids": list(self.ast_mutation_ids),
            "obligation_ids": list(self.obligation_ids),
            "source": self.source,
            "source_candidate_id": self.source_candidate_id,
            "body": dict(self.body),
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class LLMProposal:
    """Nomination-only residual model output audited against admitted scope."""

    proposed_write_paths: tuple[str, ...] = ()
    proposed_symbols: tuple[str, ...] = ()
    proposed_ast_ids: tuple[str, ...] = ()
    proposed_operator_class: str = ""
    claims: Mapping[str, Any] = field(default_factory=dict)
    source: str = "llm_residual"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "proposed_write_paths",
            _paths(self.proposed_write_paths, "proposed_write_paths"),
        )
        object.__setattr__(
            self,
            "proposed_symbols",
            _ids(self.proposed_symbols, "proposed_symbols", maximum=MAX_SYMBOL_IDS),
        )
        object.__setattr__(
            self,
            "proposed_ast_ids",
            _ids(self.proposed_ast_ids, "proposed_ast_ids", maximum=MAX_AST_IDS),
        )
        op = str(self.proposed_operator_class or "").strip()
        if op:
            object.__setattr__(
                self, "proposed_operator_class", normalize_operator_class(op)
            )
        else:
            object.__setattr__(self, "proposed_operator_class", "")
        claims = dict(self.claims or {})
        object.__setattr__(self, "claims", MappingProxyType(claims))
        object.__setattr__(
            self, "source", _text(self.source or "llm_residual", "source")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposed_write_paths": list(self.proposed_write_paths),
            "proposed_symbols": list(self.proposed_symbols),
            "proposed_ast_ids": list(self.proposed_ast_ids),
            "proposed_operator_class": self.proposed_operator_class,
            "claims": dict(self.claims),
            "source": self.source,
            "policy": LLM_OUTPUT_POLICY,
        }


@dataclass(frozen=True)
class SymbolicPlanRequest:
    """Inputs for one symbolic plan admission attempt."""

    task_id: str
    bindings: SymbolicPlanBindings
    attempt_id: str = ""
    seed_symbols: tuple[str, ...] = ()
    operator_classes: tuple[str, ...] = ()
    candidates: tuple[SymbolicCandidateSpec, ...] = ()
    llm_proposal: LLMProposal | None = None
    counterexample_ids: tuple[str, ...] = ()
    approval_granted: bool = False
    allow_partial: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        if not isinstance(self.bindings, SymbolicPlanBindings):
            raise DatabaseSymbolicPlanningIntegrityError(
                "bindings must be SymbolicPlanBindings"
            )
        object.__setattr__(
            self, "attempt_id", _text(self.attempt_id, "attempt_id", required=False)
        )
        seeds = self.seed_symbols or self.bindings.seed_symbols
        object.__setattr__(
            self, "seed_symbols", _ids(seeds, "seed_symbols", maximum=MAX_SEEDS)
        )
        ops = tuple(
            normalize_operator_class(item) for item in (self.operator_classes or ())
        )
        if len(ops) > MAX_OPERATORS:
            raise DatabaseSymbolicPlanningBoundsError(
                f"operator_classes exceeds {MAX_OPERATORS}"
            )
        object.__setattr__(self, "operator_classes", ops)
        candidates = tuple(self.candidates or ())
        if len(candidates) > MAX_CANDIDATES:
            raise DatabaseSymbolicPlanningBoundsError(
                f"candidates exceeds {MAX_CANDIDATES}"
            )
        for item in candidates:
            if not isinstance(item, SymbolicCandidateSpec):
                raise DatabaseSymbolicPlanningIntegrityError(
                    "candidates must be SymbolicCandidateSpec"
                )
        object.__setattr__(self, "candidates", candidates)
        if self.llm_proposal is not None and not isinstance(
            self.llm_proposal, LLMProposal
        ):
            raise DatabaseSymbolicPlanningIntegrityError(
                "llm_proposal must be LLMProposal or None"
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
        if not isinstance(self.approval_granted, bool):
            raise DatabaseSymbolicPlanningIntegrityError(
                "approval_granted must be boolean"
            )
        if not isinstance(self.allow_partial, bool):
            raise DatabaseSymbolicPlanningIntegrityError(
                "allow_partial must be boolean"
            )
        meta = dict(self.metadata or {})
        object.__setattr__(self, "metadata", MappingProxyType(meta))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SYMBOLIC_PLAN_REQUEST_SCHEMA,
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "bindings": self.bindings.to_dict(),
            "seed_symbols": list(self.seed_symbols),
            "operator_classes": list(self.operator_classes),
            "candidates": [item.to_dict() for item in self.candidates],
            "llm_proposal": None
            if self.llm_proposal is None
            else self.llm_proposal.to_dict(),
            "counterexample_ids": list(self.counterexample_ids),
            "approval_granted": bool(self.approval_granted),
            "allow_partial": bool(self.allow_partial),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SymbolicCandidateRecord:
    """Persisted admission outcome for one candidate under a plan."""

    candidate_id: str
    plan_id: str
    operator_class: str
    disposition: CandidateDisposition | str
    write_paths: tuple[str, ...] = ()
    symbol_ids: tuple[str, ...] = ()
    ast_mutation_ids: tuple[str, ...] = ()
    obligation_ids: tuple[str, ...] = ()
    reused: bool = False
    source_candidate_id: str = ""
    reason: str = ""
    source: str = "deterministic"
    body: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = ""
    schema: str = SYMBOLIC_CANDIDATE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "candidate_id", _text(self.candidate_id, "candidate_id")
        )
        object.__setattr__(self, "plan_id", _text(self.plan_id, "plan_id"))
        object.__setattr__(
            self,
            "operator_class",
            normalize_operator_class(self.operator_class),
        )
        disposition = self.disposition
        if not isinstance(disposition, CandidateDisposition):
            disposition = CandidateDisposition(str(disposition).strip().casefold())
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(
            self, "write_paths", _paths(self.write_paths, "write_paths")
        )
        object.__setattr__(
            self, "symbol_ids", _ids(self.symbol_ids, "symbol_ids", maximum=MAX_SYMBOL_IDS)
        )
        object.__setattr__(
            self,
            "ast_mutation_ids",
            _ids(self.ast_mutation_ids, "ast_mutation_ids", maximum=MAX_AST_IDS),
        )
        object.__setattr__(
            self,
            "obligation_ids",
            _ids(self.obligation_ids, "obligation_ids", maximum=MAX_OBLIGATIONS),
        )
        object.__setattr__(
            self,
            "source_candidate_id",
            _text(self.source_candidate_id, "source_candidate_id", required=False),
        )
        object.__setattr__(self, "reason", _bounded_text(self.reason))
        object.__setattr__(
            self, "source", _text(self.source or "deterministic", "source")
        )
        object.__setattr__(self, "body", MappingProxyType(dict(self.body or {})))
        object.__setattr__(
            self, "created_at", _text(self.created_at or _utc_iso(), "created_at")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "candidate_id": self.candidate_id,
            "plan_id": self.plan_id,
            "operator_class": self.operator_class,
            "disposition": self.disposition.value
            if isinstance(self.disposition, CandidateDisposition)
            else str(self.disposition),
            "write_paths": list(self.write_paths),
            "symbol_ids": list(self.symbol_ids),
            "ast_mutation_ids": list(self.ast_mutation_ids),
            "obligation_ids": list(self.obligation_ids),
            "reused": bool(self.reused),
            "source_candidate_id": self.source_candidate_id,
            "reason": self.reason,
            "source": self.source,
            "body": dict(self.body),
            "created_at": self.created_at,
            "authority": AUTHORITY_CLASS,
        }


@dataclass(frozen=True)
class SymbolicPlan:
    """Admitted, abstained, or rejected symbolic plan.

    Interface-adjacent contract under ``DatabaseSymbolicPlanner@1``.
    """

    plan_id: str
    task_id: str
    disposition: PlanDisposition | str
    bindings: SymbolicPlanBindings
    attempt_id: str = ""
    candidates: tuple[SymbolicCandidateRecord, ...] = ()
    seed_symbols: tuple[str, ...] = ()
    write_admitted: bool = False
    reasons: tuple[str, ...] = ()
    discovery_stages: tuple[str, ...] = ()
    impact_complete: bool = False
    blocks_automatic_repair: bool = True
    llm_audit: Mapping[str, Any] = field(default_factory=dict)
    counterexample_ids: tuple[str, ...] = ()
    plan_digest: str = ""
    created_at: str = ""
    schema: str = SYMBOLIC_PLAN_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _text(self.plan_id, "plan_id"))
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        disposition = self.disposition
        if not isinstance(disposition, PlanDisposition):
            disposition = PlanDisposition(str(disposition).strip().casefold())
        object.__setattr__(self, "disposition", disposition)
        if not isinstance(self.bindings, SymbolicPlanBindings):
            raise DatabaseSymbolicPlanningIntegrityError(
                "bindings must be SymbolicPlanBindings"
            )
        object.__setattr__(
            self, "attempt_id", _text(self.attempt_id, "attempt_id", required=False)
        )
        candidates = tuple(self.candidates or ())
        if len(candidates) > MAX_CANDIDATES:
            raise DatabaseSymbolicPlanningBoundsError(
                f"candidates exceeds {MAX_CANDIDATES}"
            )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self,
            "seed_symbols",
            _ids(self.seed_symbols, "seed_symbols", maximum=MAX_SEEDS),
        )
        reasons = tuple(
            _bounded_text(item) for item in (self.reasons or ()) if str(item).strip()
        )
        if len(reasons) > MAX_REASON_CODES:
            raise DatabaseSymbolicPlanningBoundsError(
                f"reasons exceeds {MAX_REASON_CODES}"
            )
        object.__setattr__(self, "reasons", reasons)
        stages = tuple(
            str(item).strip()
            for item in (self.discovery_stages or ())
            if str(item).strip()
        )
        object.__setattr__(self, "discovery_stages", stages)
        object.__setattr__(
            self, "llm_audit", MappingProxyType(dict(self.llm_audit or {}))
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
        # Write admission is only possible for fully admitted plans.
        write_admitted = bool(self.write_admitted)
        if disposition is not PlanDisposition.ADMITTED:
            write_admitted = False
        if not self.impact_complete or self.blocks_automatic_repair:
            write_admitted = False
        if not self.bindings.ast_is_fresh:
            write_admitted = False
        object.__setattr__(self, "write_admitted", write_admitted)
        object.__setattr__(
            self, "created_at", _text(self.created_at or _utc_iso(), "created_at")
        )
        payload = {
            "schema": self.schema,
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "disposition": disposition.value,
            "bindings": self.bindings.to_dict(),
            "seed_symbols": list(self.seed_symbols),
            "candidates": [item.to_dict() for item in candidates],
            "reasons": list(reasons),
            "write_admitted": write_admitted,
            "impact_complete": bool(self.impact_complete),
            "blocks_automatic_repair": bool(self.blocks_automatic_repair),
            "counterexample_ids": list(self.counterexample_ids),
        }
        digest = self.plan_digest or _sha256_digest(
            _canonical_json(payload).encode("utf-8")
        )
        object.__setattr__(self, "plan_digest", digest)

    @property
    def interface(self) -> str:
        return DATABASE_SYMBOLIC_PLANNER_INTERFACE

    @property
    def admitted(self) -> bool:
        return self.disposition is PlanDisposition.ADMITTED

    @property
    def abstained(self) -> bool:
        return self.disposition is PlanDisposition.ABSTAINED

    @property
    def rejected(self) -> bool:
        return self.disposition is PlanDisposition.REJECTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": DATABASE_SYMBOLIC_PLANNER_INTERFACE,
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "attempt_id": self.attempt_id,
            "disposition": self.disposition.value
            if isinstance(self.disposition, PlanDisposition)
            else str(self.disposition),
            "bindings": self.bindings.to_dict(),
            "candidates": [item.to_dict() for item in self.candidates],
            "seed_symbols": list(self.seed_symbols),
            "write_admitted": bool(self.write_admitted),
            "reasons": list(self.reasons),
            "discovery_stages": list(self.discovery_stages),
            "impact_complete": bool(self.impact_complete),
            "blocks_automatic_repair": bool(self.blocks_automatic_repair),
            "llm_audit": dict(self.llm_audit),
            "counterexample_ids": list(self.counterexample_ids),
            "plan_digest": self.plan_digest,
            "created_at": self.created_at,
            "authority": AUTHORITY_CLASS,
            "plan_authority_policy": PLAN_AUTHORITY_POLICY,
            "llm_output_policy": LLM_OUTPUT_POLICY,
        }


@dataclass(frozen=True)
class SymbolicPlanReceipt:
    """Durable receipt for one plan admission attempt."""

    receipt_id: str
    plan: SymbolicPlan
    stages: tuple[str, ...] = ()
    created_at: str = ""
    schema: str = SYMBOLIC_PLAN_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.plan, SymbolicPlan):
            raise DatabaseSymbolicPlanningIntegrityError(
                "plan must be SymbolicPlan"
            )
        object.__setattr__(
            self,
            "stages",
            tuple(str(item) for item in (self.stages or self.plan.discovery_stages)),
        )
        object.__setattr__(
            self, "created_at", _text(self.created_at or _utc_iso(), "created_at")
        )
        if not self.receipt_id:
            object.__setattr__(
                self,
                "receipt_id",
                _identity(
                    "symbolic-plan-receipt",
                    {
                        "plan_id": self.plan.plan_id,
                        "plan_digest": self.plan.plan_digest,
                        "disposition": self.plan.disposition.value
                        if isinstance(self.plan.disposition, PlanDisposition)
                        else str(self.plan.disposition),
                    },
                ),
            )
        else:
            object.__setattr__(
                self, "receipt_id", _text(self.receipt_id, "receipt_id")
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "receipt_id": self.receipt_id,
            "plan": self.plan.to_dict(),
            "stages": list(self.stages),
            "created_at": self.created_at,
            "authority": AUTHORITY_CLASS,
        }


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class DatabaseSymbolicPlanner:
    """Fail-closed symbolic planner bound to AST / mutation / impact state.

    Interface: ``DatabaseSymbolicPlanner@1``.
    """

    INTERFACE: Final[str] = DATABASE_SYMBOLIC_PLANNER_INTERFACE
    SCHEMA: Final[str] = DATABASE_SYMBOLIC_PLANNER_SCHEMA

    def __init__(
        self,
        database_path: Path | str,
        *,
        planner_version: str = DEFAULT_PLANNER_VERSION,
    ) -> None:
        if not duckdb_available():
            raise DuckDBUnavailableError(
                "DuckDB is required for DatabaseSymbolicPlanner; install the "
                "optional duckdb dependency"
            )
        self._path = Path(database_path)
        self._planner_version = _text(
            planner_version or DEFAULT_PLANNER_VERSION, "planner_version"
        )
        self._connection: Any | None = None
        self._lock = threading.RLock()
        self._closed = True
        # Exact-identity candidate reuse index: candidate_id -> body digest.
        self._reuse_index: dict[str, str] = {}

    # -- lifecycle -----------------------------------------------------------

    @property
    def database_path(self) -> Path:
        return self._path

    @property
    def planner_version(self) -> str:
        return self._planner_version

    @property
    def is_open(self) -> bool:
        return not self._closed and self._connection is not None

    def open(self) -> "DatabaseSymbolicPlanner":
        with self._lock:
            if self.is_open:
                return self
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = open_duckdb_connection(self._path)
            for statement in _split_sql_statements(_BOOKKEEPING_SQL):
                connection.execute(statement)
            for key, value in (
                ("interface", DATABASE_SYMBOLIC_PLANNER_INTERFACE),
                ("schema", DATABASE_SYMBOLIC_PLANNER_SCHEMA),
                ("planner_version", self._planner_version),
                ("authority", AUTHORITY_CLASS),
                ("plan_authority_policy", PLAN_AUTHORITY_POLICY),
                ("llm_output_policy", LLM_OUTPUT_POLICY),
            ):
                connection.execute(
                    """
                    INSERT OR REPLACE INTO symbolic_planner_metadata(key, value)
                    VALUES (?, ?)
                    """,
                    [key, value],
                )
            self._connection = connection
            self._closed = False
            self._load_reuse_index(connection)
            return self

    def close(self) -> None:
        with self._lock:
            connection = self._connection
            self._connection = None
            self._closed = True
            self._reuse_index = {}
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    def __enter__(self) -> "DatabaseSymbolicPlanner":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require(self) -> Any:
        if not self.is_open or self._connection is None:
            raise DatabaseSymbolicPlanningNotOpenError(
                "DatabaseSymbolicPlanner is not open"
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
                "SELECT key, value FROM symbolic_planner_metadata"
            ).fetchall()
            meta = {str(row[0]): str(row[1]) for row in rows}
            meta["database_path"] = str(self._path)
            meta["is_open"] = True
            return meta

    def _load_reuse_index(self, connection: Any) -> None:
        rows = connection.execute(
            """
            SELECT candidate_id, body_json
            FROM symbolic_candidates
            WHERE disposition IN (?, ?)
            """,
            [
                CandidateDisposition.ADMITTED.value,
                CandidateDisposition.REUSED.value,
            ],
        ).fetchall()
        index: dict[str, str] = {}
        for row in rows:
            mapping = _row_mapping(row)
            if mapping:
                cid = str(mapping["candidate_id"])
                body = str(mapping.get("body_json") or "{}")
            else:
                cid = str(row[0])
                body = str(row[1] or "{}")
            index[cid] = _sha256_digest(body.encode("utf-8"))
        self._reuse_index = index

    # -- public planning API -------------------------------------------------

    def plan(
        self,
        request: SymbolicPlanRequest,
        *,
        impact_closure: ImpactClosure | None = None,
        mutation: MutationSet | None = None,
    ) -> SymbolicPlanReceipt:
        """Run discovery then admit, abstain, or reject a symbolic plan.

        Order is fixed and fail-closed:

        1. bind identities
        2. discover mutation / AST lineage
        3. require impact query completeness for write admission
        4. re-check AST freshness
        5. synthesize / audit candidates (including exact reuse)
        6. audit residual LLM proposal against admitted scope
        7. admit or abstain
        """

        connection = self._require()
        if not isinstance(request, SymbolicPlanRequest):
            raise DatabaseSymbolicPlanningIntegrityError(
                "request must be SymbolicPlanRequest"
            )

        stages: list[str] = []
        reasons: list[str] = []
        created_at = _utc_iso()

        with self._lock:
            stages.append(DiscoveryStage.BINDINGS.value)
            bindings = request.bindings
            seeds = request.seed_symbols or bindings.seed_symbols

            # -- mutation discovery -----------------------------------------
            stages.append(DiscoveryStage.MUTATION_DISCOVERY.value)
            mutation_status_ok = True
            if mutation is not None:
                if not isinstance(mutation, MutationSet):
                    raise DatabaseSymbolicPlanningIntegrityError(
                        "mutation must be MutationSet when provided"
                    )
                if mutation.mutation_id and bindings.mutation_id:
                    if mutation.mutation_id != bindings.mutation_id:
                        reasons.append(PlanRejectReason.AST_BINDING_MISMATCH.value)
                        mutation_status_ok = False
                if mutation.status is not MutationStatus.ACCEPTED:
                    # Formatting-only / no-op can still plan but cannot write
                    # semantic repairs without an accepted mutation.
                    if mutation.status not in {
                        MutationStatus.NO_OP,
                    }:
                        reasons.append(PlanRejectReason.MUTATION_NOT_ACCEPTED.value)
                        mutation_status_ok = False
                # Enrich bindings from mutation when missing.
                if not bindings.mutation_id and mutation.mutation_id:
                    bindings = SymbolicPlanBindings(
                        snapshot_id=bindings.snapshot_id or mutation.before_snapshot_id,
                        tree_id=bindings.tree_id or mutation.before_tree_id,
                        repository_id=bindings.repository_id or mutation.repository_id,
                        mutation_id=mutation.mutation_id,
                        impact_query_id=bindings.impact_query_id,
                        impact_revision_id=bindings.impact_revision_id,
                        parser_id=bindings.parser_id,
                        policy_id=bindings.policy_id,
                        schema_id=bindings.schema_id,
                        seed_symbols=seeds or bindings.seed_symbols,
                        ast_mutation_ids=bindings.ast_mutation_ids,
                        symbol_ids=bindings.symbol_ids,
                        admitted_write_paths=bindings.admitted_write_paths,
                        admitted_read_paths=bindings.admitted_read_paths,
                        current_ast_digest=bindings.current_ast_digest,
                        expected_ast_digest=bindings.expected_ast_digest,
                    )
            elif not bindings.mutation_id and not request.candidates:
                # Plans without mutation binding and without candidates cannot
                # invent repair scope.
                reasons.append(PlanRejectReason.MISSING_MUTATION.value)

            # -- impact query -----------------------------------------------
            stages.append(DiscoveryStage.IMPACT_QUERY.value)
            impact_complete = False
            blocks_automatic_repair = True
            impact_query_id = bindings.impact_query_id
            impact_revision_id = bindings.impact_revision_id
            if impact_closure is not None:
                if not isinstance(impact_closure, ImpactClosure):
                    raise DatabaseSymbolicPlanningIntegrityError(
                        "impact_closure must be ImpactClosure when provided"
                    )
                impact_query_id = impact_closure.query_id
                impact_revision_id = impact_closure.revision_id
                # Snapshot binding must match.
                if (
                    impact_closure.snapshot_id
                    and bindings.snapshot_id
                    and impact_closure.snapshot_id != bindings.snapshot_id
                ):
                    reasons.append(PlanRejectReason.STALE_SNAPSHOT.value)
                completeness = impact_closure.completeness
                if isinstance(completeness, ImpactCompleteness):
                    impact_complete = completeness is ImpactCompleteness.COMPLETE
                else:
                    impact_complete = str(completeness).casefold() == "complete"
                blocks_automatic_repair = bool(
                    impact_closure.blocks_automatic_repair
                )
                if not impact_complete:
                    reasons.append(PlanRejectReason.INCOMPLETE_IMPACT.value)
                if blocks_automatic_repair:
                    reasons.append(PlanRejectReason.BLOCKING_FRONTIER.value)
                # Merge consumer symbols into seed / symbol identity set.
                consumer_symbols = tuple(
                    item.symbol for item in impact_closure.consumers
                )
                symbol_ids = tuple(
                    dict.fromkeys(
                        list(bindings.symbol_ids)
                        + list(seeds)
                        + list(consumer_symbols)
                    )
                )
                bindings = SymbolicPlanBindings(
                    snapshot_id=bindings.snapshot_id,
                    tree_id=bindings.tree_id,
                    repository_id=bindings.repository_id,
                    mutation_id=bindings.mutation_id,
                    impact_query_id=impact_query_id,
                    impact_revision_id=impact_revision_id,
                    parser_id=bindings.parser_id or impact_closure.parser_id,
                    policy_id=bindings.policy_id or impact_closure.policy_id,
                    schema_id=bindings.schema_id or impact_closure.schema_id,
                    seed_symbols=seeds or impact_closure.seed_symbols,
                    ast_mutation_ids=bindings.ast_mutation_ids,
                    symbol_ids=symbol_ids,
                    admitted_write_paths=bindings.admitted_write_paths,
                    admitted_read_paths=bindings.admitted_read_paths,
                    current_ast_digest=bindings.current_ast_digest,
                    expected_ast_digest=bindings.expected_ast_digest,
                )
            else:
                reasons.append(PlanRejectReason.MISSING_IMPACT.value)
                impact_complete = False
                blocks_automatic_repair = True

            # -- AST freshness ----------------------------------------------
            stages.append(DiscoveryStage.AST_FRESHNESS.value)
            if not bindings.ast_is_fresh:
                reasons.append(PlanRejectReason.STALE_AST.value)

            if not seeds and not bindings.seed_symbols:
                reasons.append(PlanRejectReason.EMPTY_SEEDS.value)

            # -- candidate synthesis / audit --------------------------------
            stages.append(DiscoveryStage.CANDIDATE_SYNTHESIS.value)
            plan_id = _identity(
                "symbolic-plan",
                {
                    "task_id": request.task_id,
                    "attempt_id": request.attempt_id,
                    "bindings": bindings.to_dict(),
                    "seeds": list(seeds),
                    "created_at": created_at,
                },
            )
            admitted_paths = set(bindings.admitted_write_paths)
            admitted_symbols = set(bindings.symbol_ids) | set(seeds)
            admitted_ast = set(bindings.ast_mutation_ids)

            candidate_records: list[SymbolicCandidateRecord] = []
            any_requires_approval = False
            any_out_of_scope = False
            any_admitted_candidate = False

            # Deterministic candidates from request operator classes when none
            # supplied — still bound to admitted paths/symbols only.
            specs: list[SymbolicCandidateSpec] = list(request.candidates)
            if not specs and request.operator_classes and admitted_paths:
                for op in request.operator_classes:
                    specs.append(
                        SymbolicCandidateSpec(
                            operator_class=op,
                            write_paths=tuple(sorted(admitted_paths)),
                            symbol_ids=tuple(sorted(admitted_symbols))[:MAX_SYMBOL_IDS],
                            ast_mutation_ids=tuple(sorted(admitted_ast)),
                            source="deterministic_discovery",
                        )
                    )

            for spec in specs:
                record = self._admit_candidate(
                    plan_id=plan_id,
                    spec=spec,
                    admitted_paths=admitted_paths,
                    admitted_symbols=admitted_symbols,
                    admitted_ast=admitted_ast,
                    approval_granted=request.approval_granted,
                    created_at=created_at,
                )
                candidate_records.append(record)
                if record.disposition is CandidateDisposition.REQUIRES_APPROVAL:
                    any_requires_approval = True
                if record.disposition is CandidateDisposition.OUT_OF_SCOPE:
                    any_out_of_scope = True
                    reasons.append(PlanRejectReason.SCOPE_INVENTION.value)
                if record.disposition is CandidateDisposition.UNSUPPORTED:
                    any_requires_approval = True
                    reasons.append(PlanRejectReason.UNSUPPORTED_OPERATOR.value)
                if record.disposition in {
                    CandidateDisposition.ADMITTED,
                    CandidateDisposition.REUSED,
                }:
                    any_admitted_candidate = True

            if not candidate_records and not request.llm_proposal:
                reasons.append(PlanRejectReason.NO_CANDIDATES.value)

            # -- LLM scope audit --------------------------------------------
            stages.append(DiscoveryStage.LLM_SCOPE_AUDIT.value)
            llm_audit: dict[str, Any] = {
                "schema": LLM_PROPOSAL_AUDIT_SCHEMA,
                "policy": LLM_OUTPUT_POLICY,
                "present": request.llm_proposal is not None,
                "accepted_as_nomination": False,
                "rejected_reasons": [],
            }
            if request.llm_proposal is not None:
                llm_result = self._audit_llm_proposal(
                    proposal=request.llm_proposal,
                    admitted_paths=admitted_paths,
                    admitted_symbols=admitted_symbols,
                    admitted_ast=admitted_ast,
                    approval_granted=request.approval_granted,
                )
                llm_audit.update(llm_result)
                for code in llm_result.get("rejected_reasons") or []:
                    if code not in reasons:
                        reasons.append(str(code))
                if llm_result.get("requires_approval"):
                    any_requires_approval = True
                if llm_result.get("scope_invention"):
                    any_out_of_scope = True
                # Accepted residual nominations become candidates only when
                # fully inside admitted scope and operator-supported.
                if llm_result.get("accepted_as_nomination"):
                    residual = SymbolicCandidateSpec(
                        operator_class=(
                            request.llm_proposal.proposed_operator_class
                            or "enumerative"
                        ),
                        write_paths=request.llm_proposal.proposed_write_paths,
                        symbol_ids=request.llm_proposal.proposed_symbols,
                        ast_mutation_ids=request.llm_proposal.proposed_ast_ids,
                        source="llm_residual",
                        body={"nomination_only": True},
                    )
                    residual_record = self._admit_candidate(
                        plan_id=plan_id,
                        spec=residual,
                        admitted_paths=admitted_paths,
                        admitted_symbols=admitted_symbols,
                        admitted_ast=admitted_ast,
                        approval_granted=request.approval_granted,
                        created_at=created_at,
                    )
                    candidate_records.append(residual_record)
                    if residual_record.disposition in {
                        CandidateDisposition.ADMITTED,
                        CandidateDisposition.REUSED,
                    }:
                        any_admitted_candidate = True

            # -- counterexample gate ----------------------------------------
            if request.counterexample_ids:
                reasons.append(PlanRejectReason.COUNTEREXAMPLE.value)

            # -- final admission --------------------------------------------
            stages.append(DiscoveryStage.ADMISSION.value)
            disposition = self._decide_disposition(
                reasons=reasons,
                mutation_status_ok=mutation_status_ok,
                impact_complete=impact_complete,
                blocks_automatic_repair=blocks_automatic_repair,
                ast_fresh=bindings.ast_is_fresh,
                any_admitted_candidate=any_admitted_candidate,
                any_requires_approval=any_requires_approval,
                any_out_of_scope=any_out_of_scope,
                has_counterexample=bool(request.counterexample_ids),
                allow_partial=request.allow_partial,
                approval_granted=request.approval_granted,
            )

            # Deduplicate reasons while preserving order.
            deduped_reasons = tuple(dict.fromkeys(reasons))

            write_admitted = (
                disposition is PlanDisposition.ADMITTED
                and impact_complete
                and not blocks_automatic_repair
                and bindings.ast_is_fresh
                and any_admitted_candidate
                and not request.counterexample_ids
            )

            plan = SymbolicPlan(
                plan_id=plan_id,
                task_id=request.task_id,
                attempt_id=request.attempt_id,
                disposition=disposition,
                bindings=bindings,
                candidates=tuple(candidate_records),
                seed_symbols=seeds or bindings.seed_symbols,
                write_admitted=write_admitted,
                reasons=deduped_reasons,
                discovery_stages=tuple(stages),
                impact_complete=impact_complete,
                blocks_automatic_repair=blocks_automatic_repair,
                llm_audit=llm_audit,
                counterexample_ids=request.counterexample_ids,
                created_at=created_at,
            )
            self._persist_plan(connection, plan)
            for record in candidate_records:
                self._persist_candidate(connection, record)
                if record.disposition in {
                    CandidateDisposition.ADMITTED,
                    CandidateDisposition.REUSED,
                }:
                    self._reuse_index[record.candidate_id] = _sha256_digest(
                        _canonical_json(dict(record.body)).encode("utf-8")
                    )
            self._persist_event(
                connection,
                plan_id=plan.plan_id,
                event_kind="plan_admission",
                reason=plan.disposition.value
                if isinstance(plan.disposition, PlanDisposition)
                else str(plan.disposition),
                body=plan.to_dict(),
                created_at=created_at,
            )
            self._commit_if_idle(connection)
            return SymbolicPlanReceipt(
                receipt_id="",
                plan=plan,
                stages=tuple(stages),
                created_at=created_at,
            )

    def get_plan(self, plan_id: str) -> SymbolicPlan | None:
        connection = self._require()
        pid = _text(plan_id, "plan_id")
        with self._lock:
            row = connection.execute(
                """
                SELECT plan_id, task_id, attempt_id, disposition, write_admitted,
                       reason, plan_digest, body_json, created_at
                FROM symbolic_plans
                WHERE plan_id = ?
                """,
                [pid],
            ).fetchone()
            if row is None:
                return None
            return self._plan_from_row(connection, row)

    def list_plans(
        self,
        *,
        task_id: str = "",
        disposition: PlanDisposition | str | None = None,
        limit: int = 100,
    ) -> tuple[SymbolicPlan, ...]:
        connection = self._require()
        limit = max(1, min(int(limit), 10_000))
        with self._lock:
            clauses: list[str] = []
            params: list[Any] = []
            if task_id:
                clauses.append("task_id = ?")
                params.append(_text(task_id, "task_id"))
            if disposition is not None:
                disp = (
                    disposition.value
                    if isinstance(disposition, PlanDisposition)
                    else str(disposition).strip().casefold()
                )
                clauses.append("disposition = ?")
                params.append(disp)
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            params.append(limit)
            rows = connection.execute(
                f"""
                SELECT plan_id, task_id, attempt_id, disposition, write_admitted,
                       reason, plan_digest, body_json, created_at
                FROM symbolic_plans
                {where}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
            return tuple(self._plan_from_row(connection, row) for row in rows)

    def register_reusable_candidate(
        self,
        candidate: SymbolicCandidateSpec,
    ) -> str:
        """Register an exact-identity reusable candidate for later plan hits."""

        connection = self._require()
        if not isinstance(candidate, SymbolicCandidateSpec):
            raise DatabaseSymbolicPlanningIntegrityError(
                "candidate must be SymbolicCandidateSpec"
            )
        with self._lock:
            body_digest = _sha256_digest(
                _canonical_json(dict(candidate.body)).encode("utf-8")
            )
            self._reuse_index[candidate.candidate_id] = body_digest
            # Persist a sentinel row under plan_id = reuse-index for restarts.
            connection.execute(
                """
                INSERT OR REPLACE INTO symbolic_candidates(
                    candidate_id, plan_id, operator_class, disposition, reused,
                    source_candidate_id, symbol_ids_json, ast_mutation_ids_json,
                    write_paths_json, body_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    candidate.candidate_id,
                    "plan:reuse-index",
                    candidate.operator_class,
                    CandidateDisposition.ADMITTED.value,
                    0,
                    candidate.source_candidate_id,
                    _canonical_json(list(candidate.symbol_ids)),
                    _canonical_json(list(candidate.ast_mutation_ids)),
                    _canonical_json(list(candidate.write_paths)),
                    _canonical_json(dict(candidate.body)),
                    _utc_iso(),
                ],
            )
            self._commit_if_idle(connection)
            return candidate.candidate_id

    # -- admission helpers ---------------------------------------------------

    def _admit_candidate(
        self,
        *,
        plan_id: str,
        spec: SymbolicCandidateSpec,
        admitted_paths: set[str],
        admitted_symbols: set[str],
        admitted_ast: set[str],
        approval_granted: bool,
        created_at: str,
    ) -> SymbolicCandidateRecord:
        reasons: list[str] = []
        disposition = CandidateDisposition.ADMITTED
        reused = False

        # Exact reuse: same candidate_id with same body digest.
        prior = self._reuse_index.get(spec.candidate_id)
        if prior is not None:
            body_digest = _sha256_digest(
                _canonical_json(dict(spec.body)).encode("utf-8")
            )
            if prior == body_digest:
                disposition = CandidateDisposition.REUSED
                reused = True
            else:
                disposition = CandidateDisposition.REJECTED
                reasons.append(PlanRejectReason.CANDIDATE_REUSE_MISMATCH.value)

        # Operator class gates.
        if disposition is not CandidateDisposition.REJECTED:
            if operator_requires_approval(spec.operator_class):
                if approval_granted:
                    disposition = CandidateDisposition.ADMITTED
                    reasons.append(PlanRejectReason.REQUIRES_APPROVAL.value)
                else:
                    disposition = CandidateDisposition.REQUIRES_APPROVAL
                    reasons.append(PlanRejectReason.REQUIRES_APPROVAL.value)
            elif not operator_is_supported(spec.operator_class):
                disposition = CandidateDisposition.UNSUPPORTED
                reasons.append(PlanRejectReason.UNSUPPORTED_OPERATOR.value)

        # Scope: write paths must be subset of admitted write paths when any
        # admitted paths are declared. Empty admitted set means no write scope
        # has been granted yet — any write path is out of scope.
        if disposition not in {
            CandidateDisposition.REJECTED,
            CandidateDisposition.UNSUPPORTED,
        }:
            if spec.write_paths:
                if not admitted_paths:
                    disposition = CandidateDisposition.OUT_OF_SCOPE
                    reasons.append(PlanRejectReason.SCOPE_INVENTION.value)
                else:
                    extras = set(spec.write_paths) - admitted_paths
                    if extras:
                        disposition = CandidateDisposition.OUT_OF_SCOPE
                        reasons.append(PlanRejectReason.SCOPE_INVENTION.value)

            if spec.symbol_ids and admitted_symbols:
                extras = set(spec.symbol_ids) - admitted_symbols
                if extras:
                    disposition = CandidateDisposition.OUT_OF_SCOPE
                    reasons.append(PlanRejectReason.SYMBOL_BINDING_MISMATCH.value)

            if spec.ast_mutation_ids and admitted_ast:
                extras = set(spec.ast_mutation_ids) - admitted_ast
                if extras:
                    disposition = CandidateDisposition.OUT_OF_SCOPE
                    reasons.append(PlanRejectReason.AST_BINDING_MISMATCH.value)

            # Body must not smuggle provider authority / scope-widen claims.
            for key in spec.body:
                if str(key).casefold() in _SCOPE_WIDEN_KEYS:
                    disposition = CandidateDisposition.REJECTED
                    reasons.append(PlanRejectReason.PROVIDER_CLAIM.value)
                    break

        return SymbolicCandidateRecord(
            candidate_id=spec.candidate_id,
            plan_id=plan_id,
            operator_class=spec.operator_class,
            disposition=disposition,
            write_paths=spec.write_paths,
            symbol_ids=spec.symbol_ids,
            ast_mutation_ids=spec.ast_mutation_ids,
            obligation_ids=spec.obligation_ids,
            reused=reused,
            source_candidate_id=spec.source_candidate_id or (
                spec.candidate_id if reused else ""
            ),
            reason=",".join(dict.fromkeys(reasons)),
            source=spec.source,
            body=dict(spec.body),
            created_at=created_at,
        )

    def _audit_llm_proposal(
        self,
        *,
        proposal: LLMProposal,
        admitted_paths: set[str],
        admitted_symbols: set[str],
        admitted_ast: set[str],
        approval_granted: bool,
    ) -> dict[str, Any]:
        rejected: list[str] = []
        scope_invention = False
        requires_approval = False
        semantic_invention = False

        # Provider claims never become authority.
        for key, value in proposal.claims.items():
            key_cf = str(key).casefold()
            if key_cf in _SCOPE_WIDEN_KEYS:
                rejected.append(PlanRejectReason.PROVIDER_CLAIM.value)
                if key_cf in {
                    "semantic_change",
                    "meaning_change",
                    "completion_claim",
                    "authority_override",
                }:
                    semantic_invention = True
            if isinstance(value, bool) and value and key_cf in {
                "admitted",
                "accepted",
                "allowed",
                "approved",
                "valid",
                "passed",
                "completed",
            }:
                rejected.append(PlanRejectReason.PROVIDER_CLAIM.value)

        if proposal.proposed_write_paths:
            if not admitted_paths:
                scope_invention = True
                rejected.append(PlanRejectReason.SCOPE_INVENTION.value)
            else:
                extras = set(proposal.proposed_write_paths) - admitted_paths
                if extras:
                    scope_invention = True
                    rejected.append(PlanRejectReason.SCOPE_INVENTION.value)

        if proposal.proposed_symbols and admitted_symbols:
            extras = set(proposal.proposed_symbols) - admitted_symbols
            if extras:
                scope_invention = True
                rejected.append(PlanRejectReason.SYMBOL_BINDING_MISMATCH.value)

        if proposal.proposed_ast_ids and admitted_ast:
            extras = set(proposal.proposed_ast_ids) - admitted_ast
            if extras:
                scope_invention = True
                rejected.append(PlanRejectReason.AST_BINDING_MISMATCH.value)

        if proposal.proposed_operator_class:
            if operator_requires_approval(proposal.proposed_operator_class):
                requires_approval = not approval_granted
                rejected.append(PlanRejectReason.REQUIRES_APPROVAL.value)
            elif not operator_is_supported(proposal.proposed_operator_class):
                requires_approval = True
                rejected.append(PlanRejectReason.UNSUPPORTED_OPERATOR.value)

        if semantic_invention:
            rejected.append(PlanRejectReason.SEMANTIC_INVENTION.value)

        accepted = (
            not scope_invention
            and not semantic_invention
            and not requires_approval
            and PlanRejectReason.PROVIDER_CLAIM.value not in rejected
            and bool(
                proposal.proposed_write_paths
                or proposal.proposed_symbols
                or proposal.proposed_operator_class
            )
        )
        return {
            "schema": LLM_PROPOSAL_AUDIT_SCHEMA,
            "policy": LLM_OUTPUT_POLICY,
            "present": True,
            "accepted_as_nomination": accepted,
            "rejected_reasons": list(dict.fromkeys(rejected)),
            "scope_invention": scope_invention,
            "semantic_invention": semantic_invention,
            "requires_approval": requires_approval,
            "proposal": proposal.to_dict(),
        }

    def _decide_disposition(
        self,
        *,
        reasons: Sequence[str],
        mutation_status_ok: bool,
        impact_complete: bool,
        blocks_automatic_repair: bool,
        ast_fresh: bool,
        any_admitted_candidate: bool,
        any_requires_approval: bool,
        any_out_of_scope: bool,
        has_counterexample: bool,
        allow_partial: bool,
        approval_granted: bool,
    ) -> PlanDisposition:
        reason_set = set(reasons)

        # Hard rejects — cannot invent scope / semantics.
        if any_out_of_scope or PlanRejectReason.SCOPE_INVENTION.value in reason_set:
            return PlanDisposition.REJECTED
        if PlanRejectReason.SEMANTIC_INVENTION.value in reason_set:
            return PlanDisposition.REJECTED
        if PlanRejectReason.PROVIDER_CLAIM.value in reason_set:
            return PlanDisposition.REJECTED
        if PlanRejectReason.PATH_ESCAPE.value in reason_set:
            return PlanDisposition.REJECTED
        if has_counterexample:
            return PlanDisposition.REJECTED

        # Stale AST / snapshot cannot write and cannot admit auto repair.
        if not ast_fresh or PlanRejectReason.STALE_AST.value in reason_set:
            return PlanDisposition.ABSTAINED
        if PlanRejectReason.STALE_SNAPSHOT.value in reason_set:
            return PlanDisposition.ABSTAINED

        # Incomplete / blocking impact prevents write; abstain for auto path.
        if not impact_complete or blocks_automatic_repair:
            if any_requires_approval and not approval_granted:
                return PlanDisposition.REQUIRES_APPROVAL
            return PlanDisposition.ABSTAINED

        if not mutation_status_ok:
            return PlanDisposition.ABSTAINED

        if any_requires_approval and not approval_granted:
            return PlanDisposition.REQUIRES_APPROVAL

        if not any_admitted_candidate:
            if allow_partial:
                return PlanDisposition.PARTIAL
            return PlanDisposition.ABSTAINED

        if allow_partial and PlanRejectReason.PARTIAL_PLAN.value in reason_set:
            return PlanDisposition.PARTIAL

        return PlanDisposition.ADMITTED

    # -- persistence ---------------------------------------------------------

    def _persist_plan(self, connection: Any, plan: SymbolicPlan) -> None:
        body = plan.to_dict()
        encoded = _canonical_json(body)
        if len(encoded.encode("utf-8")) > MAX_BODY_JSON_BYTES:
            raise DatabaseSymbolicPlanningBoundsError(
                f"plan body exceeds {MAX_BODY_JSON_BYTES} bytes"
            )
        connection.execute(
            """
            INSERT OR REPLACE INTO symbolic_plans(
                plan_id, task_id, attempt_id, snapshot_id, tree_id,
                repository_id, mutation_id, impact_query_id, impact_revision_id,
                disposition, write_admitted, reason, plan_digest, body_json,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                plan.plan_id,
                plan.task_id,
                plan.attempt_id,
                plan.bindings.snapshot_id,
                plan.bindings.tree_id,
                plan.bindings.repository_id,
                plan.bindings.mutation_id,
                plan.bindings.impact_query_id,
                plan.bindings.impact_revision_id,
                plan.disposition.value
                if isinstance(plan.disposition, PlanDisposition)
                else str(plan.disposition),
                1 if plan.write_admitted else 0,
                ",".join(plan.reasons),
                plan.plan_digest,
                encoded,
                plan.created_at,
            ],
        )

    def _persist_candidate(
        self, connection: Any, record: SymbolicCandidateRecord
    ) -> None:
        connection.execute(
            """
            INSERT OR REPLACE INTO symbolic_candidates(
                candidate_id, plan_id, operator_class, disposition, reused,
                source_candidate_id, symbol_ids_json, ast_mutation_ids_json,
                write_paths_json, body_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                record.candidate_id,
                record.plan_id,
                record.operator_class,
                record.disposition.value
                if isinstance(record.disposition, CandidateDisposition)
                else str(record.disposition),
                1 if record.reused else 0,
                record.source_candidate_id,
                _canonical_json(list(record.symbol_ids)),
                _canonical_json(list(record.ast_mutation_ids)),
                _canonical_json(list(record.write_paths)),
                _canonical_json(dict(record.body)),
                record.created_at,
            ],
        )

    def _persist_event(
        self,
        connection: Any,
        *,
        plan_id: str,
        event_kind: str,
        reason: str,
        body: Mapping[str, Any],
        created_at: str,
    ) -> None:
        event_id = _identity(
            "symbolic-plan-event",
            {
                "plan_id": plan_id,
                "event_kind": event_kind,
                "reason": reason,
                "created_at": created_at,
            },
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO symbolic_plan_events(
                event_id, plan_id, event_kind, reason, body_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                event_id,
                plan_id,
                event_kind,
                _bounded_text(reason),
                _canonical_json(dict(body)),
                created_at,
            ],
        )

    def _plan_from_row(self, connection: Any, row: Any) -> SymbolicPlan:
        mapping = _row_mapping(row)
        if mapping:
            body = _load_json_object(mapping.get("body_json"))
            plan_id = str(mapping["plan_id"])
            created_at = str(mapping.get("created_at") or "")
        else:
            body = _load_json_object(row[7])
            plan_id = str(row[0])
            created_at = str(row[8] or "")

        bindings_raw = dict(body.get("bindings") or {})
        bindings = SymbolicPlanBindings(
            snapshot_id=str(bindings_raw.get("snapshot_id") or ""),
            tree_id=str(bindings_raw.get("tree_id") or ""),
            repository_id=str(bindings_raw.get("repository_id") or ""),
            mutation_id=str(bindings_raw.get("mutation_id") or ""),
            impact_query_id=str(bindings_raw.get("impact_query_id") or ""),
            impact_revision_id=str(bindings_raw.get("impact_revision_id") or ""),
            parser_id=str(bindings_raw.get("parser_id") or ""),
            policy_id=str(bindings_raw.get("policy_id") or ""),
            schema_id=str(bindings_raw.get("schema_id") or ""),
            seed_symbols=tuple(bindings_raw.get("seed_symbols") or ()),
            ast_mutation_ids=tuple(bindings_raw.get("ast_mutation_ids") or ()),
            symbol_ids=tuple(bindings_raw.get("symbol_ids") or ()),
            admitted_write_paths=tuple(
                bindings_raw.get("admitted_write_paths") or ()
            ),
            admitted_read_paths=tuple(
                bindings_raw.get("admitted_read_paths") or ()
            ),
            current_ast_digest=str(bindings_raw.get("current_ast_digest") or ""),
            expected_ast_digest=str(
                bindings_raw.get("expected_ast_digest") or ""
            ),
        )
        candidate_rows = connection.execute(
            """
            SELECT candidate_id, plan_id, operator_class, disposition, reused,
                   source_candidate_id, symbol_ids_json, ast_mutation_ids_json,
                   write_paths_json, body_json, created_at
            FROM symbolic_candidates
            WHERE plan_id = ?
            ORDER BY created_at
            """,
            [plan_id],
        ).fetchall()
        candidates: list[SymbolicCandidateRecord] = []
        for crow in candidate_rows:
            cm = _row_mapping(crow)
            if cm:
                candidates.append(
                    SymbolicCandidateRecord(
                        candidate_id=str(cm["candidate_id"]),
                        plan_id=str(cm["plan_id"]),
                        operator_class=str(cm["operator_class"]),
                        disposition=str(cm["disposition"]),
                        write_paths=tuple(
                            _load_json_list(cm.get("write_paths_json"))
                        ),
                        symbol_ids=tuple(
                            _load_json_list(cm.get("symbol_ids_json"))
                        ),
                        ast_mutation_ids=tuple(
                            _load_json_list(cm.get("ast_mutation_ids_json"))
                        ),
                        reused=bool(int(cm.get("reused") or 0)),
                        source_candidate_id=str(
                            cm.get("source_candidate_id") or ""
                        ),
                        body=_load_json_object(cm.get("body_json")),
                        created_at=str(cm.get("created_at") or ""),
                    )
                )
            else:
                candidates.append(
                    SymbolicCandidateRecord(
                        candidate_id=str(crow[0]),
                        plan_id=str(crow[1]),
                        operator_class=str(crow[2]),
                        disposition=str(crow[3]),
                        write_paths=tuple(_load_json_list(crow[8])),
                        symbol_ids=tuple(_load_json_list(crow[6])),
                        ast_mutation_ids=tuple(_load_json_list(crow[7])),
                        reused=bool(int(crow[4] or 0)),
                        source_candidate_id=str(crow[5] or ""),
                        body=_load_json_object(crow[9]),
                        created_at=str(crow[10] or ""),
                    )
                )
        return SymbolicPlan(
            plan_id=plan_id,
            task_id=str(body.get("task_id") or (mapping or {}).get("task_id") or ""),
            attempt_id=str(body.get("attempt_id") or ""),
            disposition=str(body.get("disposition") or "rejected"),
            bindings=bindings,
            candidates=tuple(candidates),
            seed_symbols=tuple(body.get("seed_symbols") or ()),
            write_admitted=bool(body.get("write_admitted")),
            reasons=tuple(body.get("reasons") or ()),
            discovery_stages=tuple(body.get("discovery_stages") or ()),
            impact_complete=bool(body.get("impact_complete")),
            blocks_automatic_repair=bool(
                body.get("blocks_automatic_repair", True)
            ),
            llm_audit=dict(body.get("llm_audit") or {}),
            counterexample_ids=tuple(body.get("counterexample_ids") or ()),
            plan_digest=str(body.get("plan_digest") or ""),
            created_at=created_at or str(body.get("created_at") or ""),
        )


def open_database_symbolic_planner(
    database_path: Path | str,
    *,
    planner_version: str = DEFAULT_PLANNER_VERSION,
) -> DatabaseSymbolicPlanner:
    """Open a DatabaseSymbolicPlanner@1 store."""

    return DatabaseSymbolicPlanner(
        database_path, planner_version=planner_version
    ).open()


__all__ = [
    "APPROVAL_REQUIRED_OPERATOR_CLASSES",
    "AUTHORITY_CLASS",
    "CandidateDisposition",
    "DATABASE_SYMBOLIC_PLANNER_INTERFACE",
    "DATABASE_SYMBOLIC_PLANNER_SCHEMA",
    "DatabaseSymbolicPlanner",
    "DatabaseSymbolicPlanningAdmissionError",
    "DatabaseSymbolicPlanningBoundsError",
    "DatabaseSymbolicPlanningError",
    "DatabaseSymbolicPlanningIntegrityError",
    "DatabaseSymbolicPlanningNotOpenError",
    "DiscoveryStage",
    "DuckDBUnavailableError",
    "LLM_OUTPUT_POLICY",
    "LLMProposal",
    "PLAN_AUTHORITY_POLICY",
    "PlanDisposition",
    "PlanRejectReason",
    "SUPPORTED_OPERATOR_CLASSES",
    "SYMBOLIC_CANDIDATE_SCHEMA",
    "SYMBOLIC_PLAN_RECEIPT_SCHEMA",
    "SYMBOLIC_PLAN_REQUEST_SCHEMA",
    "SYMBOLIC_PLAN_SCHEMA",
    "SymbolicCandidateRecord",
    "SymbolicCandidateSpec",
    "SymbolicPlan",
    "SymbolicPlanBindings",
    "SymbolicPlanReceipt",
    "SymbolicPlanRequest",
    "duckdb_available",
    "normalize_operator_class",
    "open_database_symbolic_planner",
    "operator_is_supported",
    "operator_requires_approval",
]
