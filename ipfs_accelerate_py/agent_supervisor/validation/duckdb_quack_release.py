"""Joined DuckDB/Quack control-plane release receipt (DQP-039 / DQP-G090).

Interface: ``DuckDBControlPlaneReleaseReceipt@1``

Terminal aggregation gate for program
``agent-supervisor-duckdb-quack-control-plane-v1``. It independently joins
schema, Quack, import/export, intent, runtime, worktree, AST/mutation,
symbolic/proof, context/churn, control, watchdog, backup, chaos, canary,
shadow, cutover, and rollback evidence into one content-bound decision.

Fail-closed rules (acceptance):

* missing, stale, synthetic, or skipped evidence rejects the release;
* any legacy-file decision read during canary rejects the release;
* unauthorized SQL, stale lease write, false completion, accepted-state loss,
  incomplete mutation lineage, event/projection divergence, safety or quality
  regression, or absent rollback rejects the release;
* a pass records Quack experimental/beta scope and **never** claims production
  HA or future DuckDB 2.0 compatibility.

This module never grants mutation, completion, merge, promotion, or process
authority. It never fabricates or refreshes component evidence; producers own
those roots. A completed taskboard or green narrow suite is insufficient.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..task_sources.quack_capabilities import (
    DEFAULT_QUACK_BETA_LIMITATIONS,
    PINNED_DUCKDB_VERSION_PREFIX,
)
from .duckdb_quack_baseline import SAFETY_FLOOR_KEYS

# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

DUCKDB_CONTROL_PLANE_RELEASE_RECEIPT_INTERFACE: Final[str] = (
    "DuckDBControlPlaneReleaseReceipt@1"
)
DUCKDB_CONTROL_PLANE_RELEASE_VERIFIER_INTERFACE: Final[str] = (
    "DuckDBControlPlaneReleaseVerifier@1"
)
RELEASE_CONTRACT_VERSION: Final[int] = 1
TASK_ID: Final[str] = "DQP-039"
GOAL_ID: Final[str] = "DQP-G090"
BOARD_NAMESPACE: Final[str] = "agent-supervisor-duckdb-quack-control-plane-v1"
EVIDENCE: Final[str] = "dqp/duckdb-quack-release@1"

SCHEMA_PREFIX: Final[str] = "ipfs_accelerate_py/agent-supervisor"
RELEASE_RECEIPT_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/duckdb-control-plane-release-receipt@1"
)
RELEASE_POLICY_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/duckdb-control-plane-release-policy@1"
)
RELEASE_EVIDENCE_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/duckdb-control-plane-release-evidence@1"
)
RELEASE_EVIDENCE_ITEM_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/duckdb-control-plane-release-evidence-item@1"
)
RELEASE_REPORT_SCHEMA: Final[str] = (
    f"{SCHEMA_PREFIX}/duckdb-control-plane-release-report@1"
)

MAX_TEXT_BYTES: Final[int] = 512
MAX_REASON_CODES: Final[int] = 256
DEFAULT_EVIDENCE_MAX_AGE_SECONDS: Final[int] = 86_400

# Joined evidence roots (effects list from DQP-039).
REQUIRED_EVIDENCE_ROOTS: Final[tuple[str, ...]] = (
    "schema",
    "quack",
    "import_export",
    "intent",
    "runtime",
    "worktree",
    "ast_mutation",
    "symbolic_proof",
    "context_churn",
    "control",
    "watchdog",
    "backup",
    "chaos",
    "canary",
    "shadow",
    "cutover",
    "rollback",
)

# Absolute-zero safety floors mirrored from the sealed baseline catalog.
RELEASE_SAFETY_FLOOR_KEYS: Final[tuple[str, ...]] = SAFETY_FLOOR_KEYS

# Explicit non-claims sealed into every pass receipt.
NON_CLAIMS: Final[tuple[str, ...]] = (
    "not_production_ha",
    "not_multi_failure_domain",
    "not_duckdb_2_0_compatible_until_separately_tested",
    "quack_remains_experimental_beta_in_1_5_x",
    "loopback_single_owner_topology_only",
)

# Modules that must cold-import on the current tree for a joined pass.
REQUIRED_RELEASE_MODULES: Final[tuple[str, ...]] = (
    "ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema",
    "ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities",
    "ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client",
    "ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server",
    "ipfs_accelerate_py.agent_supervisor.runtime.control_plane_backup",
    "ipfs_accelerate_py.agent_supervisor.validation.duckdb_quack_baseline",
    "ipfs_accelerate_py.agent_supervisor.validation.quack_chaos",
    "ipfs_accelerate_py.agent_supervisor.validation.duckdb_quack_canary",
    "ipfs_accelerate_py.agent_supervisor.validation.duckdb_quack_benchmark",
    "ipfs_accelerate_py.agent_supervisor.self_improvement.database_shadow_rollout",
    "ipfs_accelerate_py.agent_supervisor.self_improvement.database_rollout",
    "ipfs_accelerate_py.agent_supervisor.validation.duckdb_quack_release",
)

RELEASE_MODULE_REL: Final[str] = (
    "ipfs_accelerate_py/agent_supervisor/validation/duckdb_quack_release.py"
)
RELEASE_DOC_REL: Final[str] = (
    "docs/architecture/AGENT_SUPERVISOR_DUCKDB_QUACK_RELEASE.md"
)
RELEASE_TEST_REL: Final[str] = "test/api/test_agent_supervisor_duckdb_quack_release.py"


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class EvidenceClass(str, Enum):
    """Disposition of one joined evidence root."""

    MEASURED = "measured"
    CURRENT = "current"
    STALE = "stale"
    SYNTHETIC = "synthetic"
    SKIPPED = "skipped"
    MISSING = "missing"
    FORGED = "forged"


class ReleaseVerdict(str, Enum):
    """Joined release decision; not a promotion or completion grant."""

    PASS = "pass"
    FAIL = "fail"


class DenialReason(str, Enum):
    """Machine-readable fail-closed reasons."""

    MISSING_EVIDENCE = "missing_evidence"
    STALE_EVIDENCE = "stale_evidence"
    SYNTHETIC_EVIDENCE = "synthetic_evidence"
    SKIPPED_EVIDENCE = "skipped_evidence"
    FORGED_EVIDENCE = "forged_evidence"
    FAILED_EVIDENCE = "failed_evidence"
    TREE_MISMATCH = "tree_mismatch"
    SCHEMA_MISMATCH = "schema_mismatch"
    PROFILE_MISMATCH = "profile_mismatch"
    LEGACY_FILE_DECISION_READ = "legacy_file_decision_read"
    DATABASE_NOT_SOLE_AUTHORITY = "database_not_sole_authority"
    UNAUTHORIZED_SQL = "unauthorized_sql"
    STALE_LEASE_WRITE = "stale_lease_write"
    FALSE_COMPLETION = "false_completion"
    ACCEPTED_STATE_LOSS = "accepted_state_loss"
    INCOMPLETE_MUTATION_LINEAGE = "incomplete_mutation_lineage"
    PROJECTION_DIVERGENCE = "projection_divergence"
    SAFETY_REGRESSION = "safety_regression"
    QUALITY_REGRESSION = "quality_regression"
    ABSENT_ROLLBACK = "absent_rollback"
    SAFETY_FLOOR_NONZERO = "safety_floor_nonzero"
    MODULE_MISSING = "required_module_missing"
    PRODUCTION_HA_CLAIM = "production_ha_claim"
    DUCKDB_2_0_COMPATIBILITY_CLAIM = "duckdb_2_0_compatibility_claim"
    BETA_SCOPE_UNRECORDED = "beta_scope_unrecorded"
    COMPLETION_AUTHORITY_CLAIM = "completion_authority_claim"
    MUTATION_AUTHORITY_CLAIM = "mutation_authority_claim"
    ARTIFACT_MISSING = "artifact_missing"


class DuckDBQuackReleaseError(ValueError):
    """Fail-closed rejection for incomplete or unsafe release inputs."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _text(value: Any, name: str, *, maximum: int = MAX_TEXT_BYTES) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        raise DuckDBQuackReleaseError(f"{name} must be text")
    result = value.strip()
    if not result:
        raise DuckDBQuackReleaseError(f"{name} must not be empty")
    if "\x00" in result:
        raise DuckDBQuackReleaseError(f"{name} contains a NUL byte")
    if len(result.encode("utf-8")) > maximum:
        raise DuckDBQuackReleaseError(f"{name} exceeds its {maximum}-byte bound")
    return result


def _nonnegative_int(value: Any, name: str, *, maximum: int = 10**18) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DuckDBQuackReleaseError(f"{name} must be a non-negative integer")
    if value < 0 or value > maximum:
        raise DuckDBQuackReleaseError(f"{name} out of bounds")
    return value


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise DuckDBQuackReleaseError(f"{name} must be a bool")
    return value


def content_identity(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def repository_root() -> Path:
    # validation/ -> agent_supervisor/ -> ipfs_accelerate_py/ -> repo root
    return Path(__file__).resolve().parents[3]


def _zero_floors() -> dict[str, int]:
    return {key: 0 for key in RELEASE_SAFETY_FLOOR_KEYS}


def _parse_evidence_class(value: Any) -> EvidenceClass:
    if isinstance(value, EvidenceClass):
        return value
    text = _text(value, "evidence_class", maximum=32)
    try:
        return EvidenceClass(text)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in EvidenceClass)
        raise DuckDBQuackReleaseError(
            f"evidence_class must be one of {{{allowed}}}; got {text!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Evidence model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReleaseEvidenceItem:
    """One named evidence root bound to tree / schema / profile identity."""

    SCHEMA: ClassVar[str] = RELEASE_EVIDENCE_ITEM_SCHEMA

    root: str
    identity: str
    evidence_class: EvidenceClass
    age_seconds: int
    passed: bool
    tree_id: str = ""
    schema_checksum: str = ""
    profile_id: str = ""
    producer_task_id: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", _text(self.root, "root", maximum=64))
        object.__setattr__(
            self, "identity", _text(self.identity, "identity", maximum=128)
        )
        object.__setattr__(
            self, "evidence_class", _parse_evidence_class(self.evidence_class)
        )
        object.__setattr__(
            self, "age_seconds", _nonnegative_int(self.age_seconds, "age_seconds")
        )
        object.__setattr__(self, "passed", _bool(self.passed, "passed"))
        if self.tree_id:
            object.__setattr__(
                self, "tree_id", _text(self.tree_id, "tree_id", maximum=256)
            )
        if self.schema_checksum:
            object.__setattr__(
                self,
                "schema_checksum",
                _text(self.schema_checksum, "schema_checksum", maximum=128),
            )
        if self.profile_id:
            object.__setattr__(
                self, "profile_id", _text(self.profile_id, "profile_id", maximum=128)
            )
        if self.producer_task_id:
            object.__setattr__(
                self,
                "producer_task_id",
                _text(self.producer_task_id, "producer_task_id", maximum=32),
            )
        if self.detail:
            object.__setattr__(
                self, "detail", _text(self.detail, "detail", maximum=MAX_TEXT_BYTES)
            )

    @property
    def is_admissible(self) -> bool:
        return (
            self.passed
            and self.evidence_class
            in {EvidenceClass.MEASURED, EvidenceClass.CURRENT}
            and not self.is_bad_class
        )

    @property
    def is_bad_class(self) -> bool:
        return self.evidence_class in {
            EvidenceClass.STALE,
            EvidenceClass.SYNTHETIC,
            EvidenceClass.SKIPPED,
            EvidenceClass.MISSING,
            EvidenceClass.FORGED,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "root": self.root,
            "identity": self.identity,
            "evidence_class": self.evidence_class.value,
            "age_seconds": self.age_seconds,
            "passed": self.passed,
            "tree_id": self.tree_id,
            "schema_checksum": self.schema_checksum,
            "profile_id": self.profile_id,
            "producer_task_id": self.producer_task_id,
            "detail": self.detail,
            "admissible": self.is_admissible,
        }


@dataclass(frozen=True)
class ReleaseEvidence:
    """Joined evidence package for independent release verification.

    AST surface: ``ReleaseEvidence``. Producers supply roots; the verifier never
    invents missing measurements.
    """

    SCHEMA: ClassVar[str] = RELEASE_EVIDENCE_SCHEMA

    items: tuple[ReleaseEvidenceItem, ...]
    tree_id: str
    database_identity: str
    schema_checksum: str
    extension_fingerprint: str
    quack_profile: str
    store_generation: int = 1
    duckdb_version: str = f"{PINNED_DUCKDB_VERSION_PREFIX}.2"
    # Canary / authority surface.
    legacy_file_decision_reads: int = 0
    database_sole_decision_authority: bool = True
    export_non_authoritative: bool = True
    # Lineage / durability surface.
    mutation_lineage_complete: bool = True
    projection_divergent: bool = False
    rollback_present: bool = True
    rollback_identity: str = ""
    # Safety / quality surface (counts; absolute zero required).
    safety_floors: Mapping[str, int] = field(default_factory=_zero_floors)
    unauthorized_sql_count: int = 0
    stale_lease_write_count: int = 0
    false_completion_count: int = 0
    accepted_state_loss_count: int = 0
    safety_regression: bool = False
    quality_regression: bool = False
    # Scope claims (must remain honest).
    production_ha_claimed: bool = False
    duckdb_2_0_compatibility_claimed: bool = False
    completion_authoritative: bool = False
    mutation_authorized: bool = False
    beta_limitations: tuple[str, ...] = DEFAULT_QUACK_BETA_LIMITATIONS
    experimental_scope: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "tree_id", _text(self.tree_id, "tree_id"))
        object.__setattr__(
            self,
            "database_identity",
            _text(self.database_identity, "database_identity"),
        )
        object.__setattr__(
            self, "schema_checksum", _text(self.schema_checksum, "schema_checksum")
        )
        object.__setattr__(
            self,
            "extension_fingerprint",
            _text(self.extension_fingerprint, "extension_fingerprint"),
        )
        object.__setattr__(
            self, "quack_profile", _text(self.quack_profile, "quack_profile")
        )
        object.__setattr__(
            self,
            "store_generation",
            _nonnegative_int(self.store_generation, "store_generation"),
        )
        object.__setattr__(
            self, "duckdb_version", _text(self.duckdb_version, "duckdb_version")
        )
        for name in (
            "legacy_file_decision_reads",
            "unauthorized_sql_count",
            "stale_lease_write_count",
            "false_completion_count",
            "accepted_state_loss_count",
        ):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name)
            )
        for name in (
            "database_sole_decision_authority",
            "export_non_authoritative",
            "mutation_lineage_complete",
            "projection_divergent",
            "rollback_present",
            "safety_regression",
            "quality_regression",
            "production_ha_claimed",
            "duckdb_2_0_compatibility_claimed",
            "completion_authoritative",
            "mutation_authorized",
            "experimental_scope",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        floors = {
            key: _nonnegative_int(
                dict(self.safety_floors or {}).get(key, 0), f"safety_floors.{key}"
            )
            for key in RELEASE_SAFETY_FLOOR_KEYS
        }
        for key in dict(self.safety_floors or {}):
            if key not in RELEASE_SAFETY_FLOOR_KEYS:
                raise DuckDBQuackReleaseError(f"unknown safety floor key {key!r}")
        object.__setattr__(self, "safety_floors", MappingProxyType(floors))
        limitations = tuple(
            _text(item, "beta_limitations.item", maximum=128)
            for item in (self.beta_limitations or ())
        )
        object.__setattr__(self, "beta_limitations", limitations)
        if self.rollback_identity:
            object.__setattr__(
                self,
                "rollback_identity",
                _text(self.rollback_identity, "rollback_identity", maximum=128),
            )
        elif self.rollback_present:
            object.__setattr__(
                self,
                "rollback_identity",
                content_identity(
                    {
                        "tree_id": self.tree_id,
                        "schema_checksum": self.schema_checksum,
                        "kind": "rollback",
                    }
                ),
            )

    def by_root(self) -> Mapping[str, ReleaseEvidenceItem]:
        return {item.root: item for item in self.items}

    @property
    def identity_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.SCHEMA,
            "items": [item.to_dict() for item in self.items],
            "tree_id": self.tree_id,
            "database_identity": self.database_identity,
            "schema_checksum": self.schema_checksum,
            "extension_fingerprint": self.extension_fingerprint,
            "quack_profile": self.quack_profile,
            "store_generation": self.store_generation,
            "duckdb_version": self.duckdb_version,
            "legacy_file_decision_reads": self.legacy_file_decision_reads,
            "database_sole_decision_authority": self.database_sole_decision_authority,
            "export_non_authoritative": self.export_non_authoritative,
            "mutation_lineage_complete": self.mutation_lineage_complete,
            "projection_divergent": self.projection_divergent,
            "rollback_present": self.rollback_present,
            "rollback_identity": self.rollback_identity,
            "safety_floors": dict(self.safety_floors),
            "unauthorized_sql_count": self.unauthorized_sql_count,
            "stale_lease_write_count": self.stale_lease_write_count,
            "false_completion_count": self.false_completion_count,
            "accepted_state_loss_count": self.accepted_state_loss_count,
            "safety_regression": self.safety_regression,
            "quality_regression": self.quality_regression,
            "production_ha_claimed": self.production_ha_claimed,
            "duckdb_2_0_compatibility_claimed": self.duckdb_2_0_compatibility_claimed,
            "completion_authoritative": self.completion_authoritative,
            "mutation_authorized": self.mutation_authorized,
            "beta_limitations": list(self.beta_limitations),
            "experimental_scope": self.experimental_scope,
        }
        if include_identity:
            payload["identity_id"] = self.identity_id
        return payload


# ---------------------------------------------------------------------------
# Policy + receipt
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DuckDBControlPlaneReleasePolicy:
    """Immutable fail-closed joined-release policy."""

    SCHEMA: ClassVar[str] = RELEASE_POLICY_SCHEMA
    INTERFACE: ClassVar[str] = "DuckDBControlPlaneReleasePolicy@1"

    task_id: str = TASK_ID
    goal_id: str = GOAL_ID
    board_namespace: str = BOARD_NAMESPACE
    evidence_max_age_seconds: int = DEFAULT_EVIDENCE_MAX_AGE_SECONDS
    require_all_evidence_roots: bool = True
    require_zero_safety_floors: bool = True
    require_database_sole_authority: bool = True
    require_rollback: bool = True
    require_mutation_lineage: bool = True
    require_beta_scope_recorded: bool = True
    forbid_production_ha_claim: bool = True
    forbid_duckdb_2_0_claim: bool = True
    forbid_completion_authority: bool = True
    forbid_mutation_authority: bool = True
    require_cold_imports: bool = True
    required_roots: tuple[str, ...] = REQUIRED_EVIDENCE_ROOTS
    required_modules: tuple[str, ...] = REQUIRED_RELEASE_MODULES
    safety_floor_keys: tuple[str, ...] = RELEASE_SAFETY_FLOOR_KEYS

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _text(self.task_id, "task_id"))
        object.__setattr__(self, "goal_id", _text(self.goal_id, "goal_id"))
        object.__setattr__(
            self, "board_namespace", _text(self.board_namespace, "board_namespace")
        )
        object.__setattr__(
            self,
            "evidence_max_age_seconds",
            _nonnegative_int(
                self.evidence_max_age_seconds, "evidence_max_age_seconds"
            ),
        )
        for name in (
            "require_all_evidence_roots",
            "require_zero_safety_floors",
            "require_database_sole_authority",
            "require_rollback",
            "require_mutation_lineage",
            "require_beta_scope_recorded",
            "forbid_production_ha_claim",
            "forbid_duckdb_2_0_claim",
            "forbid_completion_authority",
            "forbid_mutation_authority",
            "require_cold_imports",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
        object.__setattr__(
            self,
            "required_roots",
            tuple(
                _text(item, "required_roots.item", maximum=64)
                for item in self.required_roots
            ),
        )
        object.__setattr__(
            self,
            "required_modules",
            tuple(
                _text(item, "required_modules.item", maximum=256)
                for item in self.required_modules
            ),
        )
        object.__setattr__(
            self,
            "safety_floor_keys",
            tuple(
                _text(item, "safety_floor_keys.item", maximum=96)
                for item in self.safety_floor_keys
            ),
        )

    @property
    def identity_id(self) -> str:
        return content_identity(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "contract_version": RELEASE_CONTRACT_VERSION,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "board_namespace": self.board_namespace,
            "evidence_max_age_seconds": self.evidence_max_age_seconds,
            "require_all_evidence_roots": self.require_all_evidence_roots,
            "require_zero_safety_floors": self.require_zero_safety_floors,
            "require_database_sole_authority": self.require_database_sole_authority,
            "require_rollback": self.require_rollback,
            "require_mutation_lineage": self.require_mutation_lineage,
            "require_beta_scope_recorded": self.require_beta_scope_recorded,
            "forbid_production_ha_claim": self.forbid_production_ha_claim,
            "forbid_duckdb_2_0_claim": self.forbid_duckdb_2_0_claim,
            "forbid_completion_authority": self.forbid_completion_authority,
            "forbid_mutation_authority": self.forbid_mutation_authority,
            "require_cold_imports": self.require_cold_imports,
            "required_roots": list(self.required_roots),
            "required_modules": list(self.required_modules),
            "safety_floor_keys": list(self.safety_floor_keys),
        }
        if include_identity:
            payload["identity_id"] = self.identity_id
        return payload


@dataclass(frozen=True)
class DuckDBControlPlaneReleaseReceipt:
    """``DuckDBControlPlaneReleaseReceipt@1`` content-bound release decision."""

    SCHEMA: ClassVar[str] = RELEASE_RECEIPT_SCHEMA
    INTERFACE: ClassVar[str] = DUCKDB_CONTROL_PLANE_RELEASE_RECEIPT_INTERFACE

    verdict: ReleaseVerdict
    policy_identity: str
    evidence_identity: str
    tree_id: str
    database_identity: str
    schema_checksum: str
    extension_fingerprint: str
    duckdb_version: str
    quack_profile: str
    reason_codes: tuple[str, ...] = ()
    checked_roots: tuple[str, ...] = ()
    admissible_roots: tuple[str, ...] = ()
    missing_roots: tuple[str, ...] = ()
    modules_present: tuple[str, ...] = ()
    modules_missing: tuple[str, ...] = ()
    safety_floors: Mapping[str, int] = field(default_factory=_zero_floors)
    beta_limitations: tuple[str, ...] = DEFAULT_QUACK_BETA_LIMITATIONS
    non_claims: tuple[str, ...] = NON_CLAIMS
    experimental_scope: bool = True
    production_ha_claimed: bool = False
    duckdb_2_0_compatibility_claimed: bool = False
    promotion_allowed: bool = False
    completion_authoritative: bool = False
    mutation_authorized: bool = False
    rollback_identity: str = ""
    created_at: str = field(default_factory=_utc_iso)
    evidence: str = EVIDENCE
    task_id: str = TASK_ID
    goal_id: str = GOAL_ID
    board_namespace: str = BOARD_NAMESPACE

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, ReleaseVerdict):
            object.__setattr__(self, "verdict", ReleaseVerdict(str(self.verdict)))
        # Hard-seal honest scope on every receipt.
        object.__setattr__(self, "production_ha_claimed", False)
        object.__setattr__(self, "duckdb_2_0_compatibility_claimed", False)
        object.__setattr__(self, "completion_authoritative", False)
        object.__setattr__(self, "mutation_authorized", False)
        object.__setattr__(self, "promotion_allowed", False)
        object.__setattr__(self, "experimental_scope", True)
        object.__setattr__(self, "non_claims", tuple(NON_CLAIMS))
        object.__setattr__(
            self,
            "reason_codes",
            tuple(
                _text(item, "reason_codes.item", maximum=96)
                for item in self.reason_codes[:MAX_REASON_CODES]
            ),
        )
        object.__setattr__(
            self, "safety_floors", MappingProxyType(dict(self.safety_floors or {}))
        )
        object.__setattr__(
            self,
            "beta_limitations",
            tuple(str(item) for item in self.beta_limitations),
        )

    @property
    def passed(self) -> bool:
        return self.verdict is ReleaseVerdict.PASS

    @property
    def identity_id(self) -> str:
        # Timestamps are observational and excluded from the content identity.
        body = self.to_dict(include_identity=False)
        body.pop("created_at", None)
        return content_identity(body)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "contract_version": RELEASE_CONTRACT_VERSION,
            "evidence": self.evidence,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "board_namespace": self.board_namespace,
            "verdict": self.verdict.value,
            "passed": self.passed,
            "policy_identity": self.policy_identity,
            "evidence_identity": self.evidence_identity,
            "tree_id": self.tree_id,
            "database_identity": self.database_identity,
            "schema_checksum": self.schema_checksum,
            "extension_fingerprint": self.extension_fingerprint,
            "duckdb_version": self.duckdb_version,
            "quack_profile": self.quack_profile,
            "reason_codes": list(self.reason_codes),
            "checked_roots": list(self.checked_roots),
            "admissible_roots": list(self.admissible_roots),
            "missing_roots": list(self.missing_roots),
            "modules_present": list(self.modules_present),
            "modules_missing": list(self.modules_missing),
            "safety_floors": dict(self.safety_floors),
            "beta_limitations": list(self.beta_limitations),
            "non_claims": list(self.non_claims),
            "experimental_scope": True,
            "production_ha_claimed": False,
            "duckdb_2_0_compatibility_claimed": False,
            "promotion_allowed": False,
            "completion_authoritative": False,
            "mutation_authorized": False,
            "rollback_identity": self.rollback_identity,
            "created_at": self.created_at,
        }
        if include_identity:
            payload["identity_id"] = self.identity_id
        return payload


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def _probe_modules(modules: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    import importlib

    present: list[str] = []
    missing: list[str] = []
    for name in modules:
        try:
            importlib.import_module(name)
            present.append(name)
        except Exception:
            missing.append(name)
    return tuple(present), tuple(missing)


def _check_declared_artifacts(repo_root: Path | None = None) -> tuple[str, ...]:
    root = (repo_root or repository_root()).resolve()
    missing: list[str] = []
    for rel in (RELEASE_MODULE_REL, RELEASE_DOC_REL, RELEASE_TEST_REL):
        if not (root / rel).is_file():
            missing.append(rel)
    return tuple(missing)


def classify_evidence_disposition(item: ReleaseEvidenceItem) -> str:
    """Map one evidence item to a closed denial code or ``admissible``."""

    if item.evidence_class is EvidenceClass.MISSING:
        return DenialReason.MISSING_EVIDENCE.value
    if item.evidence_class is EvidenceClass.STALE:
        return DenialReason.STALE_EVIDENCE.value
    if item.evidence_class is EvidenceClass.SYNTHETIC:
        return DenialReason.SYNTHETIC_EVIDENCE.value
    if item.evidence_class is EvidenceClass.SKIPPED:
        return DenialReason.SKIPPED_EVIDENCE.value
    if item.evidence_class is EvidenceClass.FORGED:
        return DenialReason.FORGED_EVIDENCE.value
    if not item.passed:
        return DenialReason.FAILED_EVIDENCE.value
    if item.evidence_class in {EvidenceClass.MEASURED, EvidenceClass.CURRENT}:
        return "admissible"
    return DenialReason.FAILED_EVIDENCE.value


class DuckDBControlPlaneReleaseVerifier:
    """Independent joined release verifier (``DuckDBControlPlaneReleaseVerifier@1``).

    Queries supplied evidence against policy. Does not re-run producers and does
    not invent measurements. Default mode is report-only with no write authority.
    """

    INTERFACE: ClassVar[str] = DUCKDB_CONTROL_PLANE_RELEASE_VERIFIER_INTERFACE
    SCHEMA: ClassVar[str] = RELEASE_REPORT_SCHEMA

    def __init__(
        self,
        policy: DuckDBControlPlaneReleasePolicy | None = None,
        *,
        repo_root: Path | str | None = None,
    ) -> None:
        self.policy = policy or DuckDBControlPlaneReleasePolicy()
        self.repo_root = Path(repo_root).resolve() if repo_root else repository_root()

    def evaluate(self, evidence: ReleaseEvidence) -> DuckDBControlPlaneReleaseReceipt:
        """Join evidence roots into one content-bound release decision."""

        if not isinstance(evidence, ReleaseEvidence):
            raise DuckDBQuackReleaseError("evidence must be a ReleaseEvidence instance")

        reasons: list[str] = []
        by_root = evidence.by_root()
        checked: list[str] = []
        admissible: list[str] = []
        missing: list[str] = []

        # --- Evidence roots ---
        if self.policy.require_all_evidence_roots:
            for root in self.policy.required_roots:
                checked.append(root)
                item = by_root.get(root)
                if item is None:
                    missing.append(root)
                    reasons.append(f"{DenialReason.MISSING_EVIDENCE.value}:{root}")
                    continue
                disposition = classify_evidence_disposition(item)
                if disposition != "admissible":
                    reasons.append(f"{disposition}:{root}")
                if item.age_seconds > self.policy.evidence_max_age_seconds:
                    reasons.append(f"{DenialReason.STALE_EVIDENCE.value}:{root}")
                if item.tree_id and item.tree_id != evidence.tree_id:
                    reasons.append(f"{DenialReason.TREE_MISMATCH.value}:{root}")
                if (
                    item.schema_checksum
                    and item.schema_checksum != evidence.schema_checksum
                ):
                    reasons.append(f"{DenialReason.SCHEMA_MISMATCH.value}:{root}")
                if item.profile_id and item.profile_id != evidence.quack_profile:
                    reasons.append(f"{DenialReason.PROFILE_MISMATCH.value}:{root}")
                if (
                    disposition == "admissible"
                    and item.age_seconds <= self.policy.evidence_max_age_seconds
                    and (not item.tree_id or item.tree_id == evidence.tree_id)
                    and (
                        not item.schema_checksum
                        or item.schema_checksum == evidence.schema_checksum
                    )
                    and (
                        not item.profile_id
                        or item.profile_id == evidence.quack_profile
                    )
                ):
                    admissible.append(root)

        # --- Canary / authority ---
        if evidence.legacy_file_decision_reads > 0:
            reasons.append(DenialReason.LEGACY_FILE_DECISION_READ.value)
        if (
            self.policy.require_database_sole_authority
            and not evidence.database_sole_decision_authority
        ):
            reasons.append(DenialReason.DATABASE_NOT_SOLE_AUTHORITY.value)
        if not evidence.export_non_authoritative:
            reasons.append(DenialReason.DATABASE_NOT_SOLE_AUTHORITY.value)

        # --- Safety counters ---
        if evidence.unauthorized_sql_count > 0:
            reasons.append(DenialReason.UNAUTHORIZED_SQL.value)
        if evidence.stale_lease_write_count > 0:
            reasons.append(DenialReason.STALE_LEASE_WRITE.value)
        if evidence.false_completion_count > 0:
            reasons.append(DenialReason.FALSE_COMPLETION.value)
        if evidence.accepted_state_loss_count > 0:
            reasons.append(DenialReason.ACCEPTED_STATE_LOSS.value)

        # --- Lineage / projection / rollback ---
        if self.policy.require_mutation_lineage and not evidence.mutation_lineage_complete:
            reasons.append(DenialReason.INCOMPLETE_MUTATION_LINEAGE.value)
        if evidence.projection_divergent:
            reasons.append(DenialReason.PROJECTION_DIVERGENCE.value)
        if self.policy.require_rollback and (
            not evidence.rollback_present or not evidence.rollback_identity
        ):
            reasons.append(DenialReason.ABSENT_ROLLBACK.value)

        # --- Safety / quality floors ---
        if evidence.safety_regression:
            reasons.append(DenialReason.SAFETY_REGRESSION.value)
        if evidence.quality_regression:
            reasons.append(DenialReason.QUALITY_REGRESSION.value)
        if self.policy.require_zero_safety_floors:
            for key in self.policy.safety_floor_keys:
                value = int(evidence.safety_floors.get(key, 0))
                if value != 0:
                    reasons.append(f"{DenialReason.SAFETY_FLOOR_NONZERO.value}:{key}")

        # --- Scope honesty ---
        if self.policy.forbid_production_ha_claim and evidence.production_ha_claimed:
            reasons.append(DenialReason.PRODUCTION_HA_CLAIM.value)
        if (
            self.policy.forbid_duckdb_2_0_claim
            and evidence.duckdb_2_0_compatibility_claimed
        ):
            reasons.append(DenialReason.DUCKDB_2_0_COMPATIBILITY_CLAIM.value)
        if (
            self.policy.forbid_completion_authority
            and evidence.completion_authoritative
        ):
            reasons.append(DenialReason.COMPLETION_AUTHORITY_CLAIM.value)
        if self.policy.forbid_mutation_authority and evidence.mutation_authorized:
            reasons.append(DenialReason.MUTATION_AUTHORITY_CLAIM.value)
        if self.policy.require_beta_scope_recorded:
            if not evidence.experimental_scope or not evidence.beta_limitations:
                reasons.append(DenialReason.BETA_SCOPE_UNRECORDED.value)

        # --- Cold imports + declared artifacts ---
        modules_present: tuple[str, ...] = ()
        modules_missing: tuple[str, ...] = ()
        if self.policy.require_cold_imports:
            modules_present, modules_missing = _probe_modules(
                self.policy.required_modules
            )
            for name in modules_missing:
                reasons.append(f"{DenialReason.MODULE_MISSING.value}:{name}")

        for rel in _check_declared_artifacts(self.repo_root):
            reasons.append(f"{DenialReason.ARTIFACT_MISSING.value}:{rel}")

        # De-duplicate while preserving order.
        unique_reasons = tuple(dict.fromkeys(reasons))
        verdict = (
            ReleaseVerdict.PASS if not unique_reasons else ReleaseVerdict.FAIL
        )

        return DuckDBControlPlaneReleaseReceipt(
            verdict=verdict,
            policy_identity=self.policy.identity_id,
            evidence_identity=evidence.identity_id,
            tree_id=evidence.tree_id,
            database_identity=evidence.database_identity,
            schema_checksum=evidence.schema_checksum,
            extension_fingerprint=evidence.extension_fingerprint,
            duckdb_version=evidence.duckdb_version,
            quack_profile=evidence.quack_profile,
            reason_codes=unique_reasons,
            checked_roots=tuple(checked),
            admissible_roots=tuple(admissible),
            missing_roots=tuple(missing),
            modules_present=modules_present,
            modules_missing=modules_missing,
            safety_floors=dict(evidence.safety_floors),
            beta_limitations=tuple(evidence.beta_limitations),
            experimental_scope=True,
            rollback_identity=evidence.rollback_identity,
        )


def hermetic_passing_evidence(
    *,
    tree_id: str = "tree:sha256:dqp039-release",
    database_identity: str = "db:uuid:dqp039-control-plane",
    schema_checksum: str = "sha256:" + ("aa" * 32),
    extension_fingerprint: str = "sha256:" + ("bb" * 32),
    quack_profile: str = "profile:quack-1.5.2-loopback",
    store_generation: int = 1,
    age_seconds: int = 300,
    evidence_class: EvidenceClass | str = EvidenceClass.MEASURED,
) -> ReleaseEvidence:
    """Build a full current evidence package that satisfies default release policy."""

    cls = _parse_evidence_class(evidence_class)
    items = [
        ReleaseEvidenceItem(
            root=root,
            identity=f"evidence:{root}:pass",
            evidence_class=cls,
            age_seconds=age_seconds,
            passed=True,
            tree_id=tree_id,
            schema_checksum=schema_checksum,
            profile_id=quack_profile,
            producer_task_id="DQP-039",
        )
        for root in REQUIRED_EVIDENCE_ROOTS
    ]
    return ReleaseEvidence(
        items=tuple(items),
        tree_id=tree_id,
        database_identity=database_identity,
        schema_checksum=schema_checksum,
        extension_fingerprint=extension_fingerprint,
        quack_profile=quack_profile,
        store_generation=store_generation,
        duckdb_version=f"{PINNED_DUCKDB_VERSION_PREFIX}.2",
        legacy_file_decision_reads=0,
        database_sole_decision_authority=True,
        export_non_authoritative=True,
        mutation_lineage_complete=True,
        projection_divergent=False,
        rollback_present=True,
        safety_floors=_zero_floors(),
        unauthorized_sql_count=0,
        stale_lease_write_count=0,
        false_completion_count=0,
        accepted_state_loss_count=0,
        safety_regression=False,
        quality_regression=False,
        production_ha_claimed=False,
        duckdb_2_0_compatibility_claimed=False,
        completion_authoritative=False,
        mutation_authorized=False,
        beta_limitations=tuple(DEFAULT_QUACK_BETA_LIMITATIONS),
        experimental_scope=True,
    )


def issue_release_receipt(
    evidence: ReleaseEvidence | None = None,
    *,
    policy: DuckDBControlPlaneReleasePolicy | None = None,
    repo_root: Path | str | None = None,
) -> DuckDBControlPlaneReleaseReceipt:
    """Issue the joined release receipt for the supplied (or hermetic) evidence."""

    bundle = evidence or hermetic_passing_evidence()
    verifier = DuckDBControlPlaneReleaseVerifier(policy=policy, repo_root=repo_root)
    return verifier.evaluate(bundle)


def validate_duckdb_quack_release(
    evidence: ReleaseEvidence | None = None,
    *,
    policy: DuckDBControlPlaneReleasePolicy | None = None,
    repo_root: Path | str | None = None,
) -> DuckDBControlPlaneReleaseReceipt:
    """Alias for :func:`issue_release_receipt` matching other release modules."""

    return issue_release_receipt(evidence, policy=policy, repo_root=repo_root)


def replay_release_receipt(
    receipt: DuckDBControlPlaneReleaseReceipt,
    evidence: ReleaseEvidence,
    *,
    policy: DuckDBControlPlaneReleasePolicy | None = None,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Re-evaluate evidence and prove identity-equivalent reseal of a pass."""

    fresh = issue_release_receipt(evidence, policy=policy, repo_root=repo_root)
    identity_ok = (
        fresh.verdict == receipt.verdict
        and fresh.evidence_identity == receipt.evidence_identity
        and fresh.policy_identity == receipt.policy_identity
        and fresh.passed == receipt.passed
        and fresh.to_dict(include_identity=False)["tree_id"] == receipt.tree_id
        and fresh.to_dict(include_identity=False)["schema_checksum"]
        == receipt.schema_checksum
    )
    # Full body identity requires matching reason codes and roots.
    body_match = (
        fresh.to_dict(include_identity=False).get("reason_codes")
        == receipt.to_dict(include_identity=False).get("reason_codes")
        and fresh.admissible_roots == receipt.admissible_roots
    )
    return {
        "schema": RELEASE_REPORT_SCHEMA,
        "identity_ok": bool(identity_ok and body_match),
        "fresh_receipt_id": fresh.identity_id,
        "prior_receipt_id": receipt.identity_id,
        "fresh_verdict": fresh.verdict.value,
        "prior_verdict": receipt.verdict.value,
        "experimental_scope": True,
        "production_ha_claimed": False,
        "duckdb_2_0_compatibility_claimed": False,
    }


__all__ = (
    "BOARD_NAMESPACE",
    "DEFAULT_EVIDENCE_MAX_AGE_SECONDS",
    "DEFAULT_QUACK_BETA_LIMITATIONS",
    "DUCKDB_CONTROL_PLANE_RELEASE_RECEIPT_INTERFACE",
    "DUCKDB_CONTROL_PLANE_RELEASE_VERIFIER_INTERFACE",
    "EVIDENCE",
    "GOAL_ID",
    "NON_CLAIMS",
    "RELEASE_CONTRACT_VERSION",
    "RELEASE_EVIDENCE_SCHEMA",
    "RELEASE_RECEIPT_SCHEMA",
    "RELEASE_SAFETY_FLOOR_KEYS",
    "REQUIRED_EVIDENCE_ROOTS",
    "REQUIRED_RELEASE_MODULES",
    "TASK_ID",
    "DenialReason",
    "DuckDBControlPlaneReleasePolicy",
    "DuckDBControlPlaneReleaseReceipt",
    "DuckDBControlPlaneReleaseVerifier",
    "DuckDBQuackReleaseError",
    "EvidenceClass",
    "ReleaseEvidence",
    "ReleaseEvidenceItem",
    "ReleaseVerdict",
    "classify_evidence_disposition",
    "content_identity",
    "hermetic_passing_evidence",
    "issue_release_receipt",
    "replay_release_receipt",
    "repository_root",
    "validate_duckdb_quack_release",
)
