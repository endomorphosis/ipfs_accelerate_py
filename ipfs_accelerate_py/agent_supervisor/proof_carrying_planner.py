"""Restartable proof-carrying planning and implementation orchestration.

This module joins the formal-planning components without erasing their trust
boundaries.  A bounded plan check can admit a plan, a model can propose a
proof, Hammer can reconstruct one, a kernel can accept one, tests can exercise
one, and a ZKP can attest one.  Those facts are deliberately represented as
different evidence and never promoted merely because a provider used an
optimistic status string.

The coordinator is integration-neutral: callers provide Codex, prover, merge,
and optional monitoring callables.  The coordinator owns scheduling,
changed-scope admission, durable decisions, restart recovery, and focused
counterexample repair.  Every durable transition is projected to both a JSON
artifact and DuckDB; a restart refuses to continue if the projections differ.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Final

from .formal_plan_compiler import (
    CompilationStatus,
    FormalPlanCompiler,
    PlanCompilationResult,
)
from .formal_plan_validator import (
    FormalPlanValidator,
    PlanValidationResult,
    PlanValidationStatus,
    ValidationBounds,
)
from .formal_verification_contracts import AssuranceLevel
from .runtime_temporal_monitor import (
    MonitorVerdict,
    TemporalMonitorConfig,
    TemporalMonitorPolicy,
    monitor_event_trace,
)


PROOF_CARRYING_PLANNER_VERSION: Final = 1
PROOF_CARRYING_WORKFLOW_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-carrying-workflow@1"
)
PROOF_CARRYING_DECISION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-carrying-decision@1"
)
PROOF_CARRYING_EVIDENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-carrying-evidence@1"
)
PROOF_CARRYING_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/proof-carrying-result@1"
)
DEFAULT_STATE_FILENAME: Final = "proof_carrying_workflow.json"
DEFAULT_DATABASE_FILENAME: Final = "proof_carrying_workflow.duckdb"
_UNKNOWN_SCOPE: Final = "scope:unknown"
_MERGE_RESOURCE: Final = "resource:repository-merge"
_PRIVATE_FIELD_MARKERS: Final = frozenset(
    {
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "hidden_witness",
        "password",
        "private_key",
        "private_premise",
        "private_witness",
        "refresh_token",
        "secret",
        "session_token",
        "witness",
    }
)


class ProofCarryingPlannerError(RuntimeError):
    """Base error for workflow construction, execution, and replay."""


class WorkflowPersistenceError(ProofCarryingPlannerError):
    """Raised when paired artifacts are missing, corrupt, or disagree."""


class WorkflowConfigurationError(ProofCarryingPlannerError, ValueError):
    """Raised when an unsafe or ambiguous workflow configuration is supplied."""


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    REPAIRING = "repairing"
    COMPLETED = "completed"
    REJECTED = "rejected"
    BLOCKED = "blocked"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {
            WorkflowStatus.COMPLETED,
            WorkflowStatus.REJECTED,
            WorkflowStatus.BLOCKED,
            WorkflowStatus.FAILED,
        }


class WorkflowNodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self not in {WorkflowNodeStatus.PENDING, WorkflowNodeStatus.RUNNING}

    @property
    def dependency_satisfied(self) -> bool:
        return self is WorkflowNodeStatus.ACCEPTED


class WorkflowNodeKind(str, Enum):
    COMPILE_PLAN = "compile_plan"
    VERIFY_PLAN = "verify_plan"
    HAMMER_RECONSTRUCTION = "hammer_reconstruction"
    LEAN_KERNEL = "lean_kernel"
    COQ_KERNEL = "coq_kernel"
    LEANSTRAL_SHADOW = "leanstral_shadow"
    ZKP_ATTESTATION = "zkp_attestation"
    TEST_FALLBACK = "test_fallback"
    CODEX_IMPLEMENTATION = "codex_implementation"
    CHANGED_SCOPE_VERIFICATION = "changed_scope_verification"
    MERGE = "merge"
    RUNTIME_MONITOR = "runtime_monitor"
    COUNTEREXAMPLE_REPAIR = "counterexample_repair"


class ProverLane(str, Enum):
    """Stable matrix lanes with explicit trust roles."""

    HAMMER = "hammer"
    LEAN = "lean"
    COQ = "coq"
    LEANSTRAL_SHADOW = "leanstral_shadow"
    ZKP = "zkp"
    TEST_FALLBACK = "test_fallback"


class EvidenceRole(str, Enum):
    PLAN_CHECK = "plan_check"
    RECONSTRUCTION = "reconstruction"
    KERNEL = "kernel"
    SHADOW = "shadow"
    ATTESTATION = "attestation"
    TEST = "test"
    SCOPE = "scope"
    MERGE = "merge"
    RUNTIME_OBSERVATION = "runtime_observation"


class EvidenceVerdict(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"
    INCONCLUSIVE = "inconclusive"


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _canonical_value(value.to_dict())
    if hasattr(value, "to_record") and callable(value.to_record):
        return _canonical_value(value.to_record())
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        items = [_canonical_value(item) for item in value]
        if isinstance(value, (set, frozenset)):
            items.sort(key=_canonical_json)
        return items
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"value is not canonical JSON: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        _canonical_json(value).encode("utf-8")
    ).hexdigest()


def _strict_mapping(value: Any, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise WorkflowConfigurationError(f"{field_name} must be an object")
    try:
        decoded = json.loads(_canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise WorkflowConfigurationError(
            f"{field_name} must contain strict JSON values"
        ) from exc
    if not isinstance(decoded, dict):
        raise WorkflowConfigurationError(f"{field_name} must be an object")
    return decoded


def _private_field_path(value: Any, path: str = "$") -> str:
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key).strip().lower().replace("-", "_")
            if any(marker in name for marker in _PRIVATE_FIELD_MARKERS):
                return f"{path}.{key}"
            nested = _private_field_path(item, f"{path}.{key}")
            if nested:
                return nested
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            nested = _private_field_path(item, f"{path}[{index}]")
            if nested:
                return nested
    return ""


def _strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values: Iterable[Any] = (value,)
    elif isinstance(value, Iterable) and not isinstance(value, Mapping):
        values = value
    else:
        values = (value,)
    return tuple(
        sorted(
            {
                str(item).strip()
                for item in values
                if str(item).strip()
            }
        )
    )


def _assurance(value: Any) -> AssuranceLevel:
    try:
        return AssuranceLevel(str(getattr(value, "value", value)))
    except ValueError:
        return AssuranceLevel.UNVERIFIED


@dataclass(frozen=True)
class ProofCarryingPlannerConfig:
    """Bounded scheduling and admission policy for one workflow."""

    max_workers: int = 4
    maximum_repairs: int = 2
    max_model_context_bytes: int = 256 * 1024
    required_plan_assurance: AssuranceLevel = AssuranceLevel.CANDIDATE
    require_changed_scope: bool = True
    require_validation: bool = True
    allow_test_fallback: bool = True
    enable_zkp: bool = True
    enabled_lanes: tuple[ProverLane, ...] = tuple(ProverLane)
    validation_bounds: ValidationBounds = field(default_factory=ValidationBounds)

    def __post_init__(self) -> None:
        if isinstance(self.max_workers, bool) or self.max_workers <= 0:
            raise WorkflowConfigurationError("max_workers must be positive")
        if isinstance(self.maximum_repairs, bool) or self.maximum_repairs < 0:
            raise WorkflowConfigurationError("maximum_repairs must be non-negative")
        if (
            isinstance(self.max_model_context_bytes, bool)
            or self.max_model_context_bytes < 4 * 1024
        ):
            raise WorkflowConfigurationError(
                "max_model_context_bytes must be at least 4096"
            )
        object.__setattr__(
            self,
            "required_plan_assurance",
            _assurance(self.required_plan_assurance),
        )
        lanes = tuple(
            sorted(
                {ProverLane(getattr(item, "value", item)) for item in self.enabled_lanes},
                key=lambda item: item.value,
            )
        )
        object.__setattr__(self, "enabled_lanes", lanes)
        if not isinstance(self.validation_bounds, ValidationBounds):
            object.__setattr__(
                self,
                "validation_bounds",
                ValidationBounds.from_dict(self.validation_bounds),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_workers": self.max_workers,
            "maximum_repairs": self.maximum_repairs,
            "max_model_context_bytes": self.max_model_context_bytes,
            "required_plan_assurance": self.required_plan_assurance.value,
            "require_changed_scope": self.require_changed_scope,
            "require_validation": self.require_validation,
            "allow_test_fallback": self.allow_test_fallback,
            "enable_zkp": self.enable_zkp,
            "enabled_lanes": [item.value for item in self.enabled_lanes],
            "validation_bounds": self.validation_bounds.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProofCarryingPlannerConfig":
        return cls(
            max_workers=payload.get("max_workers", 4),
            maximum_repairs=payload.get("maximum_repairs", 2),
            max_model_context_bytes=payload.get(
                "max_model_context_bytes", 256 * 1024
            ),
            required_plan_assurance=payload.get(
                "required_plan_assurance", AssuranceLevel.CANDIDATE
            ),
            require_changed_scope=payload.get("require_changed_scope", True),
            require_validation=payload.get("require_validation", True),
            allow_test_fallback=payload.get("allow_test_fallback", True),
            enable_zkp=payload.get("enable_zkp", True),
            enabled_lanes=tuple(payload.get("enabled_lanes") or tuple(ProverLane)),
            validation_bounds=ValidationBounds.from_dict(
                payload.get("validation_bounds") or {}
            ),
        )


@dataclass(frozen=True)
class ChangedScope:
    """Declared or observed code scope used by dispatch and merge admission."""

    paths: tuple[str, ...] = ()
    ast_scope_ids: tuple[str, ...] = ()
    shared_resources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "paths", _strings(self.paths))
        object.__setattr__(self, "ast_scope_ids", _strings(self.ast_scope_ids))
        object.__setattr__(self, "shared_resources", _strings(self.shared_resources))

    @property
    def empty(self) -> bool:
        return not self.paths and not self.ast_scope_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "paths": list(self.paths),
            "ast_scope_ids": list(self.ast_scope_ids),
            "shared_resources": list(self.shared_resources),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "ChangedScope":
        payload = payload or {}
        nested = payload.get("changed_scope")
        if isinstance(nested, Mapping):
            payload = {**payload, **nested}
        return cls(
            paths=_strings(
                payload.get("changed_paths")
                or payload.get("paths")
                or payload.get("allowed_paths")
            ),
            ast_scope_ids=_strings(
                payload.get("changed_ast_scope_ids")
                or payload.get("ast_scope_ids")
                or payload.get("changed_symbols")
                or payload.get("symbols")
            ),
            shared_resources=_strings(
                payload.get("shared_resources")
                or payload.get("exclusive_resources")
            ),
        )


@dataclass(frozen=True)
class WorkflowEvidence:
    evidence_id: str
    node_id: str
    role: EvidenceRole
    verdict: EvidenceVerdict
    assurance: AssuranceLevel
    authoritative: bool
    authoritative_for: tuple[str, ...]
    lane: str = ""
    reason: str = ""
    artifact: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", EvidenceRole(self.role))
        object.__setattr__(self, "verdict", EvidenceVerdict(self.verdict))
        object.__setattr__(self, "assurance", _assurance(self.assurance))
        object.__setattr__(
            self, "authoritative_for", _strings(self.authoritative_for)
        )
        object.__setattr__(
            self, "artifact", _strict_mapping(self.artifact, field_name="artifact")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROOF_CARRYING_EVIDENCE_SCHEMA,
            "evidence_id": self.evidence_id,
            "node_id": self.node_id,
            "role": self.role.value,
            "verdict": self.verdict.value,
            "assurance": self.assurance.value,
            "authoritative": self.authoritative,
            "authoritative_for": list(self.authoritative_for),
            "lane": self.lane,
            "reason": self.reason,
            "artifact": dict(self.artifact),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkflowEvidence":
        return cls(
            evidence_id=str(payload.get("evidence_id") or ""),
            node_id=str(payload.get("node_id") or ""),
            role=payload.get("role", EvidenceRole.PLAN_CHECK),
            verdict=payload.get("verdict", EvidenceVerdict.INCONCLUSIVE),
            assurance=payload.get("assurance", AssuranceLevel.UNVERIFIED),
            authoritative=bool(payload.get("authoritative", False)),
            authoritative_for=tuple(payload.get("authoritative_for") or ()),
            lane=str(payload.get("lane") or ""),
            reason=str(payload.get("reason") or ""),
            artifact=payload.get("artifact") or {},
        )


@dataclass(frozen=True)
class WorkflowDecision:
    sequence: int
    decision_id: str
    kind: str
    node_id: str
    outcome: str
    rationale: str
    inputs_digest: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROOF_CARRYING_DECISION_SCHEMA,
            "sequence": self.sequence,
            "decision_id": self.decision_id,
            "kind": self.kind,
            "node_id": self.node_id,
            "outcome": self.outcome,
            "rationale": self.rationale,
            "inputs_digest": self.inputs_digest,
            "details": _canonical_value(self.details),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkflowDecision":
        return cls(
            sequence=int(payload.get("sequence", 0)),
            decision_id=str(payload.get("decision_id") or ""),
            kind=str(payload.get("kind") or ""),
            node_id=str(payload.get("node_id") or ""),
            outcome=str(payload.get("outcome") or ""),
            rationale=str(payload.get("rationale") or ""),
            inputs_digest=str(payload.get("inputs_digest") or ""),
            details=_strict_mapping(payload.get("details") or {}, field_name="details"),
        )


@dataclass(frozen=True)
class WorkflowNode:
    node_id: str
    kind: WorkflowNodeKind
    status: WorkflowNodeStatus
    depends_on: tuple[str, ...] = ()
    task_id: str = ""
    lane: str = ""
    scopes: tuple[str, ...] = ()
    shared_resources: tuple[str, ...] = ()
    attempt: int = 0
    output: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", WorkflowNodeKind(self.kind))
        object.__setattr__(self, "status", WorkflowNodeStatus(self.status))
        object.__setattr__(self, "depends_on", _strings(self.depends_on))
        object.__setattr__(self, "scopes", _strings(self.scopes))
        object.__setattr__(
            self, "shared_resources", _strings(self.shared_resources)
        )
        object.__setattr__(
            self, "output", _strict_mapping(self.output, field_name="node output")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "status": self.status.value,
            "depends_on": list(self.depends_on),
            "task_id": self.task_id,
            "lane": self.lane,
            "scopes": list(self.scopes),
            "shared_resources": list(self.shared_resources),
            "attempt": self.attempt,
            "output": _canonical_value(self.output),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkflowNode":
        return cls(
            node_id=str(payload.get("node_id") or ""),
            kind=payload.get("kind", WorkflowNodeKind.COMPILE_PLAN),
            status=payload.get("status", WorkflowNodeStatus.PENDING),
            depends_on=tuple(payload.get("depends_on") or ()),
            task_id=str(payload.get("task_id") or ""),
            lane=str(payload.get("lane") or ""),
            scopes=tuple(payload.get("scopes") or ()),
            shared_resources=tuple(payload.get("shared_resources") or ()),
            attempt=int(payload.get("attempt", 0)),
            output=payload.get("output") or {},
            reason=str(payload.get("reason") or ""),
        )


@dataclass(frozen=True)
class WorkflowReplay:
    workflow_id: str
    status: WorkflowStatus
    source: Mapping[str, Any]
    config: ProofCarryingPlannerConfig
    nodes: tuple[WorkflowNode, ...]
    decisions: tuple[WorkflowDecision, ...]
    evidence: tuple[WorkflowEvidence, ...]
    artifact_digest: str
    json_path: Path
    duckdb_path: Path

    @property
    def reproducible(self) -> bool:
        return bool(self.artifact_digest)

    def decision(self, decision_id: str) -> WorkflowDecision | None:
        return next(
            (item for item in self.decisions if item.decision_id == decision_id),
            None,
        )


@dataclass(frozen=True)
class ProofCarryingWorkflowResult:
    workflow_id: str
    status: WorkflowStatus
    plan_id: str
    compilation_status: str
    validation_status: str
    nodes: tuple[WorkflowNode, ...]
    decisions: tuple[WorkflowDecision, ...]
    evidence: tuple[WorkflowEvidence, ...]
    merged_task_ids: tuple[str, ...]
    counterexamples: tuple[Mapping[str, Any], ...]
    repaired_counterexample_ids: tuple[str, ...]
    authoritative_assurance: AssuranceLevel
    json_path: Path
    duckdb_path: Path
    artifact_digest: str

    @property
    def complete(self) -> bool:
        return self.status is WorkflowStatus.COMPLETED

    @property
    def accepted(self) -> bool:
        return self.complete

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PROOF_CARRYING_RESULT_SCHEMA,
            "workflow_id": self.workflow_id,
            "status": self.status.value,
            "complete": self.complete,
            "plan_id": self.plan_id,
            "compilation_status": self.compilation_status,
            "validation_status": self.validation_status,
            "nodes": [item.to_dict() for item in self.nodes],
            "decisions": [item.to_dict() for item in self.decisions],
            "evidence": [item.to_dict() for item in self.evidence],
            "merged_task_ids": list(self.merged_task_ids),
            "counterexamples": [_canonical_value(item) for item in self.counterexamples],
            "repaired_counterexample_ids": list(self.repaired_counterexample_ids),
            "authoritative_assurance": self.authoritative_assurance.value,
            "artifacts": {
                "json": str(self.json_path),
                "duckdb": str(self.duckdb_path),
                "digest": self.artifact_digest,
            },
        }


@dataclass(frozen=True)
class WorkflowAdapters:
    """External effects used by the workflow.

    Every callable receives one canonical mapping and returns a mapping, a
    boolean, an object with ``to_dict()``, or ``None`` for unavailable.
    """

    codex_dispatch: Callable[[Mapping[str, Any]], Any] | None = None
    prover_lane: Callable[[Mapping[str, Any]], Any] | None = None
    merge: Callable[[Mapping[str, Any]], Any] | None = None
    monitor: Callable[[Mapping[str, Any]], Any] | None = None
    repair_dispatch: Callable[[Mapping[str, Any]], Any] | None = None
    scope_verify: Callable[[Mapping[str, Any]], Any] | None = None


@dataclass(frozen=True)
class _NodeSpec:
    node_id: str
    kind: WorkflowNodeKind
    depends_on: tuple[str, ...] = ()
    task_id: str = ""
    lane: str = ""
    scope: ChangedScope = field(default_factory=ChangedScope)


class _PairedWorkflowStore:
    """Atomic paired JSON/DuckDB projection with fail-closed replay."""

    def __init__(self, root: Path | str) -> None:
        supplied = Path(root)
        if supplied.suffix.lower() == ".json":
            self.json_path = supplied
            self.root = supplied.parent
        else:
            self.root = supplied
            self.json_path = supplied / DEFAULT_STATE_FILENAME
        self.duckdb_path = self.json_path.with_name(DEFAULT_DATABASE_FILENAME)
        self.lock_path = self.json_path.with_name(
            f".{self.json_path.name}.workflow.lock"
        )
        self.root.mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.RLock()

    def exists(self) -> bool:
        return self.json_path.is_file() or self.duckdb_path.is_file()

    def _locked(self):
        class _Lock:
            def __init__(inner, outer: "_PairedWorkflowStore") -> None:
                inner.outer = outer
                inner.handle = None

            def __enter__(inner):
                inner.outer._thread_lock.acquire()
                inner.handle = inner.outer.lock_path.open("a+b")
                fcntl.flock(inner.handle.fileno(), fcntl.LOCK_EX)
                return inner

            def __exit__(inner, exc_type, exc, traceback):
                assert inner.handle is not None
                fcntl.flock(inner.handle.fileno(), fcntl.LOCK_UN)
                inner.handle.close()
                inner.outer._thread_lock.release()
                return False

        return _Lock(self)

    @staticmethod
    def _with_digest(state: Mapping[str, Any]) -> dict[str, Any]:
        result = _strict_mapping(state, field_name="workflow state")
        result.pop("artifact_digest", None)
        result["artifact_digest"] = _digest(result)
        return result

    def write(self, state: Mapping[str, Any]) -> dict[str, Any]:
        projected = self._with_digest(state)
        encoded = json.dumps(
            projected,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
        try:
            import duckdb  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency is tested elsewhere
            raise WorkflowPersistenceError(
                "DuckDB is required for paired proof-carrying artifacts"
            ) from exc
        with self._locked():
            json_fd, json_name = tempfile.mkstemp(
                prefix=f".{self.json_path.name}.",
                suffix=".tmp",
                dir=self.root,
            )
            os.close(json_fd)
            database_fd, database_name = tempfile.mkstemp(
                prefix=f".{self.duckdb_path.name}.",
                suffix=".tmp",
                dir=self.root,
            )
            os.close(database_fd)
            json_temp = Path(json_name)
            database_temp = Path(database_name)
            try:
                json_temp.write_text(encoded, encoding="utf-8")
                # DuckDB refuses an existing zero-byte file.  ``mkstemp`` is
                # still used to reserve a unique same-directory name, then
                # removed before DuckDB creates the database itself.
                database_temp.unlink()
                connection = duckdb.connect(str(database_temp))
                try:
                    connection.execute(
                        "CREATE TABLE workflow_snapshot "
                        "(workflow_id VARCHAR, artifact_digest VARCHAR, snapshot_json VARCHAR)"
                    )
                    connection.execute(
                        "CREATE TABLE workflow_decisions "
                        "(sequence BIGINT, decision_id VARCHAR, decision_json VARCHAR)"
                    )
                    connection.execute(
                        "CREATE TABLE workflow_nodes "
                        "(node_id VARCHAR, status VARCHAR, node_json VARCHAR)"
                    )
                    connection.execute(
                        "CREATE TABLE workflow_evidence "
                        "(evidence_id VARCHAR, node_id VARCHAR, evidence_json VARCHAR)"
                    )
                    connection.execute(
                        "INSERT INTO workflow_snapshot VALUES (?, ?, ?)",
                        [
                            projected["workflow_id"],
                            projected["artifact_digest"],
                            _canonical_json(projected),
                        ],
                    )
                    for item in projected.get("decisions", []):
                        connection.execute(
                            "INSERT INTO workflow_decisions VALUES (?, ?, ?)",
                            [
                                item["sequence"],
                                item["decision_id"],
                                _canonical_json(item),
                            ],
                        )
                    for item in projected.get("nodes", []):
                        connection.execute(
                            "INSERT INTO workflow_nodes VALUES (?, ?, ?)",
                            [
                                item["node_id"],
                                item["status"],
                                _canonical_json(item),
                            ],
                        )
                    for item in projected.get("evidence", []):
                        connection.execute(
                            "INSERT INTO workflow_evidence VALUES (?, ?, ?)",
                            [
                                item["evidence_id"],
                                item["node_id"],
                                _canonical_json(item),
                            ],
                        )
                    connection.execute("CHECKPOINT")
                finally:
                    connection.close()
                os.replace(database_temp, self.duckdb_path)
                os.replace(json_temp, self.json_path)
            finally:
                if json_temp.exists():
                    json_temp.unlink()
                if database_temp.exists():
                    database_temp.unlink()
        return projected

    def read(self) -> dict[str, Any]:
        if not self.json_path.is_file() or not self.duckdb_path.is_file():
            raise WorkflowPersistenceError(
                "both JSON and DuckDB workflow artifacts are required"
            )
        try:
            import duckdb  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise WorkflowPersistenceError(
                "DuckDB is required to replay proof-carrying artifacts"
            ) from exc
        with self._locked():
            try:
                json_state = json.loads(self.json_path.read_text(encoding="utf-8"))
                connection = duckdb.connect(str(self.duckdb_path), read_only=True)
                try:
                    row = connection.execute(
                        "SELECT artifact_digest, snapshot_json "
                        "FROM workflow_snapshot"
                    ).fetchone()
                    decision_rows = connection.execute(
                        "SELECT decision_json FROM workflow_decisions ORDER BY sequence"
                    ).fetchall()
                finally:
                    connection.close()
            except (OSError, ValueError, json.JSONDecodeError, Exception) as exc:
                if isinstance(exc, WorkflowPersistenceError):
                    raise
                raise WorkflowPersistenceError(
                    f"cannot read paired workflow artifacts: {type(exc).__name__}: {exc}"
                ) from exc
        if row is None:
            raise WorkflowPersistenceError("DuckDB workflow snapshot is missing")
        database_state = json.loads(row[1])
        claimed = str(json_state.get("artifact_digest") or "")
        core = dict(json_state)
        core.pop("artifact_digest", None)
        computed = _digest(core)
        if (
            not claimed
            or claimed != computed
            or claimed != str(row[0])
            or _canonical_json(json_state) != _canonical_json(database_state)
        ):
            raise WorkflowPersistenceError(
                "JSON and DuckDB workflow projections disagree"
            )
        decisions = [json.loads(item[0]) for item in decision_rows]
        if _canonical_json(decisions) != _canonical_json(
            json_state.get("decisions") or []
        ):
            raise WorkflowPersistenceError(
                "JSON and DuckDB decision projections disagree"
            )
        return json_state


class ProofCarryingPlanner:
    """Compile, prove, implement, merge, monitor, repair, and resume one plan."""

    def __init__(
        self,
        source: Mapping[str, Any] | None = None,
        *,
        artifact_path: Path | str | None = None,
        state_path: Path | str | None = None,
        adapters: WorkflowAdapters | None = None,
        config: ProofCarryingPlannerConfig | Mapping[str, Any] | None = None,
        compiler: FormalPlanCompiler | None = None,
        validator: FormalPlanValidator | None = None,
        codex_dispatcher: Callable[[Mapping[str, Any]], Any] | None = None,
        prover_executor: Callable[[Mapping[str, Any]], Any] | None = None,
        merge_executor: Callable[[Mapping[str, Any]], Any] | None = None,
        monitor_executor: Callable[[Mapping[str, Any]], Any] | None = None,
        repair_dispatcher: Callable[[Mapping[str, Any]], Any] | None = None,
        scope_verifier: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> None:
        if artifact_path is not None and state_path is not None:
            raise WorkflowConfigurationError(
                "provide artifact_path or state_path, not both"
            )
        durable_path = artifact_path if artifact_path is not None else state_path
        if durable_path is None:
            raise WorkflowConfigurationError(
                "artifact_path or state_path is required"
            )
        self.store = _PairedWorkflowStore(durable_path)
        supplied_config = config or ProofCarryingPlannerConfig()
        self.config = (
            supplied_config
            if isinstance(supplied_config, ProofCarryingPlannerConfig)
            else ProofCarryingPlannerConfig.from_dict(supplied_config)
        )
        base = adapters or WorkflowAdapters()
        self.adapters = WorkflowAdapters(
            codex_dispatch=codex_dispatcher or base.codex_dispatch,
            prover_lane=prover_executor or base.prover_lane,
            merge=merge_executor or base.merge,
            monitor=monitor_executor or base.monitor,
            repair_dispatch=repair_dispatcher or base.repair_dispatch,
            scope_verify=scope_verifier or base.scope_verify,
        )
        self.compiler = compiler or FormalPlanCompiler()
        self.validator = validator or FormalPlanValidator(
            self.config.validation_bounds
        )
        if source is None:
            if not self.store.exists():
                raise WorkflowConfigurationError(
                    "source is required when no restart artifacts exist"
                )
            persisted = self.store.read()
            self.source = _strict_mapping(
                persisted.get("source") or {}, field_name="persisted source"
            )
            persisted_config = ProofCarryingPlannerConfig.from_dict(
                persisted.get("config") or {}
            )
            if config is None:
                self.config = persisted_config
                self.validator = validator or FormalPlanValidator(
                    self.config.validation_bounds
                )
            elif self.config.to_dict() != persisted_config.to_dict():
                raise WorkflowConfigurationError(
                    "restart config does not match persisted workflow config"
                )
        else:
            self.source = _strict_mapping(source, field_name="source")
        private_path = _private_field_path(self.source)
        if private_path:
            raise WorkflowConfigurationError(
                "private proving material must remain backend-owned; "
                f"public workflow source contains {private_path}"
            )
        identity_input = {
            "schema": PROOF_CARRYING_WORKFLOW_SCHEMA,
            "source": self.source,
            "config": self.config.to_dict(),
        }
        self.workflow_id = _digest(identity_input)
        self._state: dict[str, Any] = {}
        self._compilation: PlanCompilationResult | None = None
        self._validation: PlanValidationResult | None = None
        self._task_records = self._index_task_records()
        self._specs: dict[str, _NodeSpec] = {}

    @classmethod
    def restart(
        cls,
        artifact_path: Path | str,
        **kwargs: Any,
    ) -> "ProofCarryingPlanner":
        return cls(None, artifact_path=artifact_path, **kwargs)

    @classmethod
    def replay(cls, artifact_path: Path | str) -> WorkflowReplay:
        store = _PairedWorkflowStore(artifact_path)
        state = store.read()
        return WorkflowReplay(
            workflow_id=str(state["workflow_id"]),
            status=WorkflowStatus(state["status"]),
            source=state["source"],
            config=ProofCarryingPlannerConfig.from_dict(state["config"]),
            nodes=tuple(
                WorkflowNode.from_dict(item) for item in state.get("nodes", [])
            ),
            decisions=tuple(
                WorkflowDecision.from_dict(item)
                for item in state.get("decisions", [])
            ),
            evidence=tuple(
                WorkflowEvidence.from_dict(item)
                for item in state.get("evidence", [])
            ),
            artifact_digest=str(state["artifact_digest"]),
            json_path=store.json_path,
            duckdb_path=store.duckdb_path,
        )

    def _index_task_records(self) -> dict[str, dict[str, Any]]:
        records: list[Any] = []
        for key in ("tasks", "taskboard", "task_records"):
            value = self.source.get(key) or ()
            if isinstance(value, Mapping):
                records.extend(value.values())
            elif isinstance(value, Sequence) and not isinstance(value, str):
                records.extend(value)
        indexed: dict[str, dict[str, Any]] = {}
        for value in records:
            if not isinstance(value, Mapping):
                continue
            record = _strict_mapping(value, field_name="task record")
            for key in ("task_id", "task_cid", "id", "cid"):
                identifier = str(record.get(key) or "").strip()
                if identifier:
                    indexed[identifier] = record
        return indexed

    def _initial_state(self) -> dict[str, Any]:
        return {
            "schema": PROOF_CARRYING_WORKFLOW_SCHEMA,
            "version": PROOF_CARRYING_PLANNER_VERSION,
            "workflow_id": self.workflow_id,
            "status": WorkflowStatus.PENDING.value,
            "source": self.source,
            "config": self.config.to_dict(),
            "plan_id": "",
            "compilation": {},
            "validation": {},
            "nodes": [],
            "decisions": [],
            "evidence": [],
            "counterexamples": [],
            "repaired_counterexample_ids": [],
        }

    def _load_or_create(self) -> None:
        if self.store.exists():
            self._state = self.store.read()
            if self._state.get("workflow_id") != self.workflow_id:
                raise WorkflowPersistenceError(
                    "persisted workflow identity does not match source and config"
                )
            recovered = False
            for node in self._state.get("nodes", []):
                if node.get("status") == WorkflowNodeStatus.RUNNING.value:
                    node["status"] = WorkflowNodeStatus.PENDING.value
                    node["reason"] = "recovered_after_restart"
                    recovered = True
            if recovered:
                self._decision(
                    kind="restart_recovery",
                    node_id="workflow",
                    outcome="requeued",
                    rationale="running nodes have no durable completion and were requeued",
                    details={},
                    persist=False,
                )
                self._persist()
        else:
            self._state = self._initial_state()
            self._persist()

    def _persist(self) -> None:
        self._state["nodes"] = sorted(
            self._state.get("nodes", []), key=lambda item: item["node_id"]
        )
        self._state["evidence"] = sorted(
            self._state.get("evidence", []), key=lambda item: item["evidence_id"]
        )
        self._state = self.store.write(self._state)

    def _decision(
        self,
        *,
        kind: str,
        node_id: str,
        outcome: str,
        rationale: str,
        details: Mapping[str, Any],
        persist: bool = True,
    ) -> WorkflowDecision:
        sequence = len(self._state.setdefault("decisions", [])) + 1
        canonical_details = _strict_mapping(details, field_name="decision details")
        identity_payload = {
            "workflow_id": self.workflow_id,
            "sequence": sequence,
            "kind": kind,
            "node_id": node_id,
            "outcome": outcome,
            "rationale": rationale,
            "details": canonical_details,
        }
        decision = WorkflowDecision(
            sequence=sequence,
            decision_id=_digest(identity_payload),
            kind=kind,
            node_id=node_id,
            outcome=outcome,
            rationale=rationale,
            inputs_digest=_digest(canonical_details),
            details=canonical_details,
        )
        self._state["decisions"].append(decision.to_dict())
        if persist:
            self._persist()
        return decision

    def _upsert_node(
        self,
        spec: _NodeSpec,
        status: WorkflowNodeStatus = WorkflowNodeStatus.PENDING,
        *,
        output: Mapping[str, Any] | None = None,
        reason: str = "",
        increment_attempt: bool = False,
    ) -> WorkflowNode:
        nodes = self._state.setdefault("nodes", [])
        existing = next(
            (item for item in nodes if item.get("node_id") == spec.node_id), None
        )
        if (
            existing is not None
            and status is WorkflowNodeStatus.PENDING
            and output is None
            and not reason
            and not increment_attempt
            and WorkflowNodeStatus(existing["status"]).terminal
        ):
            # Graph reconstruction on restart declares the same node specs but
            # must not erase a durable terminal result.
            self._specs[spec.node_id] = spec
            return WorkflowNode.from_dict(existing)
        attempt = int(existing.get("attempt", 0)) if existing else 0
        if increment_attempt:
            attempt += 1
        record = WorkflowNode(
            node_id=spec.node_id,
            kind=spec.kind,
            status=status,
            depends_on=spec.depends_on,
            task_id=spec.task_id,
            lane=spec.lane,
            scopes=tuple(
                sorted(
                    set(spec.scope.paths)
                    | set(spec.scope.ast_scope_ids)
                    | ({_UNKNOWN_SCOPE} if spec.scope.empty else set())
                )
            ),
            shared_resources=spec.scope.shared_resources,
            attempt=attempt,
            output=output or (existing.get("output") if existing else {}) or {},
            reason=reason,
        )
        if existing is None:
            nodes.append(record.to_dict())
        else:
            existing.clear()
            existing.update(record.to_dict())
        self._specs[spec.node_id] = spec
        return record

    def _node(self, node_id: str) -> WorkflowNode:
        item = next(
            value
            for value in self._state.get("nodes", [])
            if value["node_id"] == node_id
        )
        return WorkflowNode.from_dict(item)

    def _add_evidence(
        self,
        *,
        node_id: str,
        role: EvidenceRole,
        verdict: EvidenceVerdict,
        assurance: AssuranceLevel,
        authoritative: bool,
        authoritative_for: Iterable[str],
        lane: str = "",
        reason: str = "",
        artifact: Mapping[str, Any] | None = None,
    ) -> WorkflowEvidence:
        payload = {
            "workflow_id": self.workflow_id,
            "node_id": node_id,
            "role": role.value,
            "verdict": verdict.value,
            "assurance": assurance.value,
            "authoritative": authoritative,
            "authoritative_for": sorted(set(authoritative_for)),
            "lane": lane,
            "reason": reason,
            "artifact": artifact or {},
        }
        evidence = WorkflowEvidence(
            evidence_id=_digest(payload),
            node_id=node_id,
            role=role,
            verdict=verdict,
            assurance=assurance,
            authoritative=authoritative,
            authoritative_for=tuple(authoritative_for),
            lane=lane,
            reason=reason,
            artifact=artifact or {},
        )
        records = self._state.setdefault("evidence", [])
        if not any(item["evidence_id"] == evidence.evidence_id for item in records):
            records.append(evidence.to_dict())
        return evidence

    @staticmethod
    def _adapter_output(value: Any) -> dict[str, Any]:
        if value is None:
            return {"status": "unavailable", "accepted": False}
        if isinstance(value, bool):
            return {"status": "accepted" if value else "rejected", "accepted": value}
        if isinstance(value, Mapping):
            result = _strict_mapping(value, field_name="adapter output")
        elif hasattr(value, "to_dict") and callable(value.to_dict):
            result = _strict_mapping(value.to_dict(), field_name="adapter output")
        else:
            raise TypeError(
                "workflow adapters must return a mapping, boolean, "
                "to_dict object, or None"
            )
        private_path = _private_field_path(result)
        if private_path:
            raise WorkflowConfigurationError(
                "adapter output contains backend-private material at "
                f"{private_path}"
            )
        return result

    @staticmethod
    def _accepted(output: Mapping[str, Any]) -> bool:
        explicit = output.get("accepted")
        if isinstance(explicit, bool):
            return explicit
        status = str(
            output.get("status")
            or output.get("verdict")
            or output.get("outcome")
            or ""
        ).lower()
        return status in {
            "accepted",
            "passed",
            "proved",
            "verified",
            "succeeded",
            "success",
            "consistent",
            "merged",
            "no_violation_observed",
        }

    @staticmethod
    def _unavailable(output: Mapping[str, Any]) -> bool:
        return str(output.get("status") or "").lower() in {
            "unavailable",
            "unsupported",
            "not_run",
            "skipped",
        }

    def _compile(self) -> bool:
        spec = _NodeSpec("plan:compile", WorkflowNodeKind.COMPILE_PLAN)
        self._upsert_node(spec)
        existing = self._node(spec.node_id)
        self._compilation = self.compiler.compile(self.source)
        if existing.status is WorkflowNodeStatus.ACCEPTED:
            if self._state.get("plan_id") != self._compilation.plan_id:
                raise WorkflowPersistenceError(
                    "recompiled plan identity differs from persisted plan"
                )
            return True
        self._upsert_node(
            spec, WorkflowNodeStatus.RUNNING, increment_attempt=True
        )
        self._persist()
        accepted = (
            self._compilation.status is CompilationStatus.COMPILED
            and self._compilation.plan is not None
        )
        status = (
            WorkflowNodeStatus.ACCEPTED
            if accepted
            else WorkflowNodeStatus.REJECTED
        )
        output = self._compilation.to_dict()
        self._state["compilation"] = output
        self._state["plan_id"] = self._compilation.plan_id
        self._upsert_node(
            spec,
            status,
            output={
                "status": self._compilation.status.value,
                "plan_id": self._compilation.plan_id,
                "issue_count": len(self._compilation.issues),
            },
            reason="" if accepted else "formal plan compilation failed",
        )
        self._decision(
            kind="plan_compilation",
            node_id=spec.node_id,
            outcome=status.value,
            rationale=(
                "reviewed planning semantics compiled"
                if accepted
                else "invalid or unsupported planning semantics cannot be dispatched"
            ),
            details={
                "compilation_status": self._compilation.status.value,
                "plan_id": self._compilation.plan_id,
                "issue_codes": [item.code.value for item in self._compilation.issues],
            },
        )
        return accepted

    def _task_scope(self, formal_task: Any) -> ChangedScope:
        record: Mapping[str, Any] = {}
        candidates = [formal_task.task_id]
        candidates.extend(formal_task.metadata.get("source_cids") or ())
        for candidate in candidates:
            if str(candidate) in self._task_records:
                record = self._task_records[str(candidate)]
                break
        metadata = formal_task.metadata
        combined = {
            **dict(record),
            "changed_ast_scope_ids": (
                record.get("changed_ast_scope_ids")
                or record.get("changed_ast_scopes")
                or metadata.get("changed_ast_scope_ids")
            ),
            "shared_resources": (
                record.get("shared_resources")
                or record.get("exclusive_resources")
                or ()
            ),
        }
        return ChangedScope.from_mapping(combined)

    def _build_specs(self) -> None:
        assert self._compilation is not None and self._compilation.plan is not None
        proof_specs = [
            _NodeSpec(
                "plan:verify",
                WorkflowNodeKind.VERIFY_PLAN,
                depends_on=("plan:compile",),
            )
        ]
        lane_specs = {
            ProverLane.HAMMER: _NodeSpec(
                "proof:hammer",
                WorkflowNodeKind.HAMMER_RECONSTRUCTION,
                depends_on=("plan:compile",),
                lane=ProverLane.HAMMER.value,
                scope=ChangedScope(shared_resources=("resource:cpu-proof",)),
            ),
            ProverLane.LEAN: _NodeSpec(
                "proof:lean",
                WorkflowNodeKind.LEAN_KERNEL,
                depends_on=("plan:compile",),
                lane=ProverLane.LEAN.value,
                scope=ChangedScope(shared_resources=("resource:lean-kernel",)),
            ),
            ProverLane.COQ: _NodeSpec(
                "proof:coq",
                WorkflowNodeKind.COQ_KERNEL,
                depends_on=("plan:compile",),
                lane=ProverLane.COQ.value,
                scope=ChangedScope(shared_resources=("resource:coq-kernel",)),
            ),
            ProverLane.LEANSTRAL_SHADOW: _NodeSpec(
                "proof:leanstral-shadow",
                WorkflowNodeKind.LEANSTRAL_SHADOW,
                depends_on=("plan:compile",),
                lane=ProverLane.LEANSTRAL_SHADOW.value,
                scope=ChangedScope(shared_resources=("resource:model",)),
            ),
            ProverLane.TEST_FALLBACK: _NodeSpec(
                "proof:test-fallback",
                WorkflowNodeKind.TEST_FALLBACK,
                depends_on=("plan:compile",),
                lane=ProverLane.TEST_FALLBACK.value,
                scope=ChangedScope(shared_resources=("resource:test",)),
            ),
        }
        for lane in self.config.enabled_lanes:
            if lane in lane_specs:
                proof_specs.append(lane_specs[lane])
        kernel_dependencies = tuple(
            spec.node_id
            for spec in proof_specs
            if spec.kind in {
                WorkflowNodeKind.LEAN_KERNEL,
                WorkflowNodeKind.COQ_KERNEL,
            }
        )
        if self.config.enable_zkp and ProverLane.ZKP in self.config.enabled_lanes:
            proof_specs.append(
                _NodeSpec(
                    "proof:zkp",
                    WorkflowNodeKind.ZKP_ATTESTATION,
                    depends_on=kernel_dependencies or ("plan:compile",),
                    lane=ProverLane.ZKP.value,
                    scope=ChangedScope(
                        shared_resources=("resource:cryptographic-prover",)
                    ),
                )
            )
        for spec in proof_specs:
            self._specs[spec.node_id] = spec
            self._upsert_node(spec)

        merge_by_task = {
            task.task_id: f"task:{task.task_id}:merge"
            for task in self._compilation.plan.tasks
        }
        for task in self._compilation.plan.tasks:
            scope = self._task_scope(task)
            implementation_id = f"task:{task.task_id}:implement"
            scope_id = f"task:{task.task_id}:scope"
            merge_id = merge_by_task[task.task_id]
            dependencies = ("plan:verify",) + tuple(
                merge_by_task[item] for item in task.depends_on
            )
            specs = (
                _NodeSpec(
                    implementation_id,
                    WorkflowNodeKind.CODEX_IMPLEMENTATION,
                    depends_on=dependencies,
                    task_id=task.task_id,
                    scope=scope,
                ),
                _NodeSpec(
                    scope_id,
                    WorkflowNodeKind.CHANGED_SCOPE_VERIFICATION,
                    depends_on=(implementation_id,),
                    task_id=task.task_id,
                    scope=scope,
                ),
                _NodeSpec(
                    merge_id,
                    WorkflowNodeKind.MERGE,
                    depends_on=(scope_id,),
                    task_id=task.task_id,
                    scope=ChangedScope(
                        paths=scope.paths,
                        ast_scope_ids=scope.ast_scope_ids,
                        shared_resources=(_MERGE_RESOURCE,),
                    ),
                ),
            )
            for spec in specs:
                self._specs[spec.node_id] = spec
                self._upsert_node(spec)

        monitor_dependencies = tuple(
            sorted(
                [
                    spec.node_id
                    for spec in self._specs.values()
                    if spec.kind is WorkflowNodeKind.MERGE
                ]
                + [
                    spec.node_id
                    for spec in proof_specs
                    if spec.kind is not WorkflowNodeKind.ZKP_ATTESTATION
                ]
            )
        )
        monitor = _NodeSpec(
            "runtime:monitor",
            WorkflowNodeKind.RUNTIME_MONITOR,
            depends_on=monitor_dependencies,
        )
        self._specs[monitor.node_id] = monitor
        self._upsert_node(monitor)
        self._persist()

    def _formal_task(self, task_id: str) -> Any:
        assert self._compilation is not None and self._compilation.plan is not None
        return next(
            item for item in self._compilation.plan.tasks if item.task_id == task_id
        )

    @staticmethod
    def _compact_context_value(
        value: Any,
        *,
        depth: int = 0,
        max_string_bytes: int = 4096,
    ) -> Any:
        """Bound open-ended adapter metadata before it reaches a model."""

        if depth >= 7:
            return "<depth-limit>"
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, str):
            encoded = value.encode("utf-8")
            if len(encoded) <= max_string_bytes:
                return value
            suffix = "...<truncated>"
            kept = encoded[
                : max(0, max_string_bytes - len(suffix.encode("utf-8")))
            ].decode("utf-8", errors="ignore")
            return kept + suffix
        if isinstance(value, Mapping):
            return {
                str(key): ProofCarryingPlanner._compact_context_value(
                    item,
                    depth=depth + 1,
                    max_string_bytes=max_string_bytes,
                )
                for key, item in sorted(
                    value.items(), key=lambda pair: str(pair[0])
                )[:128]
            }
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return [
                ProofCarryingPlanner._compact_context_value(
                    item,
                    depth=depth + 1,
                    max_string_bytes=max_string_bytes,
                )
                for item in value[:128]
            ]
        return str(value)[:max_string_bytes]

    def _bounded_context(self, context: Mapping[str, Any]) -> dict[str, Any]:
        context_without_binding = dict(context)
        context_without_binding.pop("context_binding", None)
        full_digest = _digest(context_without_binding)
        bounded = self._compact_context_value(context_without_binding)
        assert isinstance(bounded, dict)
        truncated = _canonical_json(bounded) != _canonical_json(context)
        binding = {
            "bounded": True,
            "limit_bytes": self.config.max_model_context_bytes,
            "full_context_digest": full_digest,
            "truncated": truncated,
        }
        bounded["context_binding"] = binding
        encoded = _canonical_json(bounded).encode("utf-8")
        if len(encoded) > self.config.max_model_context_bytes:
            # Retain the task's actionable contract and replace potentially
            # large upstream outputs with identities.
            task = bounded.get("task")
            task_source = bounded.get("task_source")
            dependencies = bounded.get("dependencies")
            counterexample = bounded.get("counterexample")
            if isinstance(task, Mapping):
                bounded["task"] = {
                    key: task[key]
                    for key in (
                        "task_id",
                        "goal_id",
                        "actor_ids",
                        "depends_on",
                        "evidence_requirement_ids",
                    )
                    if key in task
                }
            if isinstance(task_source, Mapping):
                bounded["task_source"] = {
                    key: task_source[key]
                    for key in (
                        "task_id",
                        "task_cid",
                        "acceptance_criteria",
                        "validation_commands",
                    )
                    if key in task_source
                }
            if isinstance(dependencies, Mapping):
                bounded["dependencies"] = {
                    str(key): {
                        "output_digest": _digest(value),
                        "status": (
                            value.get("status", "")
                            if isinstance(value, Mapping)
                            else ""
                        ),
                    }
                    for key, value in sorted(dependencies.items())
                }
            if isinstance(counterexample, Mapping):
                focused_counterexample = {
                    key: counterexample[key]
                    for key in (
                        "counterexample_id",
                        "id",
                        "task_id",
                        "subject_id",
                        "property",
                        "property_kind",
                        "reason",
                        "trace",
                    )
                    if key in counterexample
                }
                bounded["counterexample"] = self._compact_context_value(
                    focused_counterexample,
                    max_string_bytes=512,
                )
            binding["truncated"] = True
            encoded = _canonical_json(bounded).encode("utf-8")
        if len(encoded) > self.config.max_model_context_bytes:
            raise WorkflowConfigurationError(
                "essential model context exceeds max_model_context_bytes"
            )
        binding["used_bytes"] = len(_canonical_json(bounded).encode("utf-8"))
        # Adding the byte count can add a few bytes.  Recompute to make the
        # reported usage exact and enforce the limit on the final object.
        binding["used_bytes"] = len(_canonical_json(bounded).encode("utf-8"))
        if binding["used_bytes"] > self.config.max_model_context_bytes:
            raise WorkflowConfigurationError(
                "bounded model context exceeds max_model_context_bytes"
            )
        return bounded

    def _context(self, spec: _NodeSpec) -> dict[str, Any]:
        dependencies = {
            item: self._node(item).output for item in spec.depends_on
        }
        context: dict[str, Any] = {
            "workflow_id": self.workflow_id,
            "plan_id": self._state.get("plan_id", ""),
            "node_id": spec.node_id,
            "kind": spec.kind.value,
            "task_id": spec.task_id,
            "lane": spec.lane,
            "declared_scope": spec.scope.to_dict(),
            "dependencies": dependencies,
            "source_identity": _digest(self.source),
        }
        if spec.task_id:
            task = self._formal_task(spec.task_id)
            context["task"] = task.to_dict()
            record = next(
                (
                    self._task_records[candidate]
                    for candidate in [
                        task.task_id,
                        *(task.metadata.get("source_cids") or ()),
                    ]
                    if candidate in self._task_records
                ),
                {},
            )
            context["task_source"] = record
        return self._bounded_context(context)

    def _verify_plan(self) -> dict[str, Any]:
        assert self._compilation is not None and self._compilation.plan is not None
        self._validation = self.validator.validate(
            self._compilation.plan, self._compilation.formulas
        )
        return {
            "status": self._validation.status.value,
            "accepted": self._validation.status is PlanValidationStatus.CONSISTENT,
            "validation_id": self._validation.validation_id,
            "consistency_level": self._validation.consistency_level.value,
            "bounds_id": self._validation.bounds.bounds_id,
            "plan_check_only": self._validation.plan_check_only,
            "finding_codes": [item.code.value for item in self._validation.findings],
            "counterexample": (
                self._validation.countermodel.to_dict()
                if self._validation.countermodel is not None
                else None
            ),
        }

    def _execute_lane(self, spec: _NodeSpec) -> dict[str, Any]:
        context = self._context(spec)
        if spec.kind is WorkflowNodeKind.ZKP_ATTESTATION:
            kernel_evidence = [
                item
                for item in self._state.get("evidence", [])
                if item.get("role") == EvidenceRole.KERNEL.value
                and item.get("authoritative")
                and item.get("verdict") == EvidenceVerdict.ACCEPTED.value
            ]
            context["kernel_evidence"] = kernel_evidence
            if not kernel_evidence:
                return {
                    "status": "skipped",
                    "accepted": False,
                    "reason": "ZKP cannot attest without accepted kernel evidence",
                }
            context = self._bounded_context(context)
        if self.adapters.prover_lane is None:
            return {
                "status": "unavailable",
                "accepted": False,
                "reason": f"no adapter configured for {spec.lane}",
            }
        return self._adapter_output(self.adapters.prover_lane(context))

    def _execute_implementation(self, spec: _NodeSpec) -> dict[str, Any]:
        if self.adapters.codex_dispatch is None:
            return {
                "status": "unavailable",
                "accepted": False,
                "reason": "no Codex dispatcher configured",
            }
        return self._adapter_output(self.adapters.codex_dispatch(self._context(spec)))

    @staticmethod
    def _path_within(actual: str, declared: str) -> bool:
        actual_path = actual.replace("\\", "/").strip("/")
        declared_path = declared.replace("\\", "/").strip("/")
        return (
            actual_path == declared_path
            or actual_path.startswith(f"{declared_path}/")
        )

    def _built_in_scope_check(
        self, declared: ChangedScope, actual: ChangedScope
    ) -> tuple[bool, tuple[str, ...]]:
        violations: list[str] = []
        if self.config.require_changed_scope and actual.empty:
            violations.append("implementation did not report any changed scope")
        for path in actual.paths:
            if not any(self._path_within(path, item) for item in declared.paths):
                violations.append(f"path outside declared scope: {path}")
        for symbol in actual.ast_scope_ids:
            if symbol not in declared.ast_scope_ids:
                violations.append(f"AST scope outside declared scope: {symbol}")
        if declared.empty and not actual.empty:
            violations.append("task has no authoritative declared changed scope")
        return not violations, tuple(sorted(set(violations)))

    def _execute_scope_verification(self, spec: _NodeSpec) -> dict[str, Any]:
        implementation = self._node(spec.depends_on[0]).output
        actual = ChangedScope.from_mapping(implementation)
        accepted, violations = self._built_in_scope_check(spec.scope, actual)
        validation_passed = implementation.get("validation_passed")
        if validation_passed is None:
            validation_passed = implementation.get("tests_passed")
        if self.config.require_validation and validation_passed is not True:
            accepted = False
            violations = (*violations, "required implementation validation did not pass")
        external: dict[str, Any] = {}
        if self.adapters.scope_verify is not None:
            external = self._adapter_output(
                self.adapters.scope_verify(
                    {
                        **self._context(spec),
                        "implementation": implementation,
                        "actual_scope": actual.to_dict(),
                        "built_in_accepted": accepted,
                        "built_in_violations": list(violations),
                    }
                )
            )
            if not self._accepted(external):
                accepted = False
                violations = (
                    *violations,
                    str(external.get("reason") or "external scope verifier rejected"),
                )
        return {
            "status": "accepted" if accepted else "rejected",
            "accepted": accepted,
            "declared_scope": spec.scope.to_dict(),
            "actual_scope": actual.to_dict(),
            "validation_passed": validation_passed is True,
            "violations": list(sorted(set(violations))),
            "external_verification": external,
        }

    def _execute_merge(self, spec: _NodeSpec) -> dict[str, Any]:
        scope_output = self._node(spec.depends_on[0]).output
        implementation_id = spec.depends_on[0].replace(":scope", ":implement")
        implementation = self._node(implementation_id).output
        if not self._accepted(scope_output):
            return {
                "status": "rejected",
                "accepted": False,
                "reason": "changed scope was not accepted",
            }
        if self.adapters.merge is None:
            return {
                "status": "unavailable",
                "accepted": False,
                "reason": "no merge adapter configured",
            }
        return self._adapter_output(
            self.adapters.merge(
                {
                    **self._context(spec),
                    "implementation": implementation,
                    "scope_verification": scope_output,
                }
            )
        )

    def _monitor_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        sequence = 0
        origin = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for task_id in sorted(
            {
                spec.task_id
                for spec in self._specs.values()
                if spec.task_id
            }
        ):
            merge_id = f"task:{task_id}:merge"
            if merge_id not in self._specs:
                continue
            for event_type in (
                "lease_acquired",
                "implementation_started",
                "resource_acquired",
                "implementation_finished",
                "proof_verified",
                "merge_started",
                "task_completed",
                "resource_released",
                "lease_released",
            ):
                sequence += 1
                timestamp = origin + timedelta(seconds=sequence)
                events.append(
                    {
                        "event_id": f"{task_id}:{event_type}",
                        "type": event_type,
                        "task_id": task_id,
                        "lane_id": "proof-carrying",
                        "repository_tree_id": self.source.get(
                            "repository_tree_id", "tree:unknown"
                        ),
                        "policy_id": TemporalMonitorPolicy().policy_id,
                        "restart_epoch": self.workflow_id,
                        "sequence": sequence,
                        "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                        "status": (
                            "completed"
                            if event_type == "task_completed"
                            else "succeeded"
                        ),
                        "proof_verified": event_type == "proof_verified",
                        "expires_at": (
                            (timestamp + timedelta(minutes=5))
                            .isoformat()
                            .replace("+00:00", "Z")
                            if event_type == "lease_acquired"
                            else None
                        ),
                        "resource_id": (
                            f"workflow:{task_id}"
                            if event_type
                            in {"resource_acquired", "resource_released"}
                            else None
                        ),
                    }
                )
        return events

    def _execute_monitor(self, spec: _NodeSpec) -> dict[str, Any]:
        events = self._monitor_events()
        report = monitor_event_trace(
            events,
            config=TemporalMonitorConfig(out_of_order_window_seconds=0),
            now=(
                datetime(2024, 1, 1, tzinfo=timezone.utc)
                + timedelta(seconds=len(events) + 1)
            ).isoformat(),
        )
        output = {
            "status": report.verdict.value,
            "accepted": report.verdict is MonitorVerdict.NO_VIOLATION_OBSERVED,
            "runtime_observation_only": True,
            "proved": False,
            "report": report.to_dict(),
            "counterexamples": [
                item.to_dict() for item in report.counterexamples
            ],
        }
        if self.adapters.monitor is not None:
            external = self._adapter_output(
                self.adapters.monitor(
                    {
                        **self._context(spec),
                        "events": events,
                        "built_in_report": report.to_dict(),
                        "repair_round": len(
                            self._state.get("repaired_counterexample_ids", [])
                        ),
                    }
                )
            )
            output["external"] = external
            external_counterexamples = external.get("counterexamples") or ()
            if external.get("counterexample"):
                external_counterexamples = (
                    *external_counterexamples,
                    external["counterexample"],
                )
            output["counterexamples"] = [
                *output["counterexamples"],
                *[
                    _strict_mapping(item, field_name="runtime counterexample")
                    for item in external_counterexamples
                    if isinstance(item, Mapping)
                ],
            ]
            if (
                external.get("violated") is True
                or str(external.get("status") or "").lower()
                in {"violated", "rejected", "counterexample"}
            ):
                output["accepted"] = False
                output["status"] = "violated"
            elif self._accepted(external):
                output["accepted"] = output["accepted"] and True
        return output

    def _execute_spec(self, spec: _NodeSpec) -> dict[str, Any]:
        if spec.kind is WorkflowNodeKind.VERIFY_PLAN:
            return self._verify_plan()
        if spec.kind in {
            WorkflowNodeKind.HAMMER_RECONSTRUCTION,
            WorkflowNodeKind.LEAN_KERNEL,
            WorkflowNodeKind.COQ_KERNEL,
            WorkflowNodeKind.LEANSTRAL_SHADOW,
            WorkflowNodeKind.ZKP_ATTESTATION,
            WorkflowNodeKind.TEST_FALLBACK,
        }:
            return self._execute_lane(spec)
        if spec.kind is WorkflowNodeKind.CODEX_IMPLEMENTATION:
            return self._execute_implementation(spec)
        if spec.kind is WorkflowNodeKind.CHANGED_SCOPE_VERIFICATION:
            return self._execute_scope_verification(spec)
        if spec.kind is WorkflowNodeKind.MERGE:
            return self._execute_merge(spec)
        if spec.kind is WorkflowNodeKind.RUNTIME_MONITOR:
            return self._execute_monitor(spec)
        raise ProofCarryingPlannerError(f"unsupported node kind: {spec.kind.value}")

    @staticmethod
    def _scope_conflict(left: WorkflowNode, right: WorkflowNode) -> bool:
        if set(left.shared_resources) & set(right.shared_resources):
            return True
        if _UNKNOWN_SCOPE in left.scopes or _UNKNOWN_SCOPE in right.scopes:
            return (
                left.kind is WorkflowNodeKind.CODEX_IMPLEMENTATION
                and right.kind is WorkflowNodeKind.CODEX_IMPLEMENTATION
            )
        for first in left.scopes:
            for second in right.scopes:
                if first == second:
                    return True
                if "/" in first or "/" in second:
                    if ProofCarryingPlanner._path_within(
                        first, second
                    ) or ProofCarryingPlanner._path_within(second, first):
                        return True
        return False

    def _dependency_state(
        self, spec: _NodeSpec
    ) -> tuple[bool, tuple[str, ...]]:
        missing: list[str] = []
        for dependency in spec.depends_on:
            node = self._node(dependency)
            if node.status is WorkflowNodeStatus.ACCEPTED:
                continue
            # Non-authoritative proof lanes may be unavailable without blocking
            # monitoring or optional attestation.
            if (
                spec.kind
                in {
                    WorkflowNodeKind.RUNTIME_MONITOR,
                    WorkflowNodeKind.ZKP_ATTESTATION,
                }
                and node.kind
                in {
                    WorkflowNodeKind.HAMMER_RECONSTRUCTION,
                    WorkflowNodeKind.LEAN_KERNEL,
                    WorkflowNodeKind.COQ_KERNEL,
                    WorkflowNodeKind.LEANSTRAL_SHADOW,
                    WorkflowNodeKind.TEST_FALLBACK,
                }
                and node.status.terminal
            ):
                continue
            missing.append(dependency)
        if (
            spec.kind is WorkflowNodeKind.CODEX_IMPLEMENTATION
            and not self._required_assurance_satisfied()
        ):
            missing.append(
                f"assurance:{self.config.required_plan_assurance.value}"
            )
        return not missing, tuple(missing)

    def _record_node_evidence(
        self, spec: _NodeSpec, output: Mapping[str, Any]
    ) -> None:
        accepted = self._accepted(output)
        unavailable = self._unavailable(output)
        verdict = (
            EvidenceVerdict.ACCEPTED
            if accepted
            else EvidenceVerdict.UNAVAILABLE
            if unavailable
            else EvidenceVerdict.REJECTED
        )
        role = EvidenceRole.PLAN_CHECK
        assurance = AssuranceLevel.UNVERIFIED
        authoritative = False
        authoritative_for: tuple[str, ...] = ()
        if spec.kind is WorkflowNodeKind.VERIFY_PLAN:
            role = EvidenceRole.PLAN_CHECK
            assurance = AssuranceLevel.CANDIDATE if accepted else AssuranceLevel.UNVERIFIED
            authoritative = accepted
            authoritative_for = ("plan_admission",)
        elif spec.kind is WorkflowNodeKind.HAMMER_RECONSTRUCTION:
            role = EvidenceRole.RECONSTRUCTION
            assurance = AssuranceLevel.CANDIDATE if accepted else AssuranceLevel.UNVERIFIED
            authoritative_for = ("kernel_input",) if accepted else ()
        elif spec.kind in {
            WorkflowNodeKind.LEAN_KERNEL,
            WorkflowNodeKind.COQ_KERNEL,
        }:
            role = EvidenceRole.KERNEL
            # A callback must explicitly attest that an independent kernel
            # checked the exact plan binding; provider-claimed "accepted" alone
            # remains solver-level evidence.
            kernel_checked = (
                accepted
                and output.get("kernel_checked") is True
                and output.get("binding_verified") is True
            )
            assurance = (
                AssuranceLevel.KERNEL_VERIFIED
                if kernel_checked
                else AssuranceLevel.SOLVER_CHECKED
                if accepted
                else AssuranceLevel.UNVERIFIED
            )
            authoritative = kernel_checked
            authoritative_for = ("formal_proof",) if kernel_checked else ()
        elif spec.kind is WorkflowNodeKind.LEANSTRAL_SHADOW:
            role = EvidenceRole.SHADOW
            assurance = AssuranceLevel.CANDIDATE if accepted else AssuranceLevel.UNVERIFIED
            # Shadow output is retained but never authoritative, even if a
            # provider attempts to label it so.
            authoritative = False
            authoritative_for = ()
        elif spec.kind is WorkflowNodeKind.ZKP_ATTESTATION:
            role = EvidenceRole.ATTESTATION
            cryptographic = (
                accepted
                and output.get("cryptographic") is True
                and output.get("simulated") is not True
                and output.get("binding_verified") is True
            )
            assurance = (
                AssuranceLevel.ATTESTED
                if cryptographic
                else AssuranceLevel.UNVERIFIED
            )
            authoritative = cryptographic
            authoritative_for = ("proof_receipt_attestation",) if cryptographic else ()
        elif spec.kind is WorkflowNodeKind.TEST_FALLBACK:
            role = EvidenceRole.TEST
            assurance = AssuranceLevel.CANDIDATE if accepted else AssuranceLevel.UNVERIFIED
            authoritative = accepted and self.config.allow_test_fallback
            authoritative_for = ("regression",) if authoritative else ()
        elif spec.kind is WorkflowNodeKind.CHANGED_SCOPE_VERIFICATION:
            role = EvidenceRole.SCOPE
            assurance = AssuranceLevel.CANDIDATE if accepted else AssuranceLevel.UNVERIFIED
            authoritative = accepted
            authoritative_for = ("changed_scope",) if accepted else ()
        elif spec.kind is WorkflowNodeKind.MERGE:
            role = EvidenceRole.MERGE
            assurance = AssuranceLevel.CANDIDATE if accepted else AssuranceLevel.UNVERIFIED
            authoritative = accepted
            authoritative_for = ("merge",) if accepted else ()
        elif spec.kind is WorkflowNodeKind.RUNTIME_MONITOR:
            role = EvidenceRole.RUNTIME_OBSERVATION
            assurance = AssuranceLevel.UNVERIFIED
            authoritative = False
            authoritative_for = ()
        else:
            return
        self._add_evidence(
            node_id=spec.node_id,
            role=role,
            verdict=verdict,
            assurance=assurance,
            authoritative=authoritative,
            authoritative_for=authoritative_for,
            lane=spec.lane,
            reason=str(output.get("reason") or ""),
            artifact=output,
        )

    def _finish_node(
        self,
        spec: _NodeSpec,
        output: Mapping[str, Any] | None,
        error: BaseException | None,
    ) -> None:
        if error is not None:
            status = WorkflowNodeStatus.FAILED
            result = {
                "status": "failed",
                "accepted": False,
                "error_type": type(error).__name__,
                "error": str(error)[:512],
            }
            rationale = "bounded node execution raised an error"
        else:
            result = _strict_mapping(output or {}, field_name="node output")
            if self._accepted(result):
                status = WorkflowNodeStatus.ACCEPTED
                rationale = "node output satisfied its typed admission rule"
            elif self._unavailable(result):
                status = (
                    WorkflowNodeStatus.SKIPPED
                    if str(result.get("status")).lower() == "skipped"
                    else WorkflowNodeStatus.UNAVAILABLE
                )
                rationale = "lane was explicitly unavailable or unsupported"
            else:
                status = WorkflowNodeStatus.REJECTED
                rationale = "node output did not satisfy its typed admission rule"
        if spec.kind is WorkflowNodeKind.VERIFY_PLAN and self._validation is not None:
            self._state["validation"] = self._validation.to_dict()
        self._upsert_node(
            spec,
            status,
            output=result,
            reason=str(result.get("reason") or result.get("error") or ""),
        )
        self._record_node_evidence(spec, result)
        self._decision(
            kind=spec.kind.value,
            node_id=spec.node_id,
            outcome=status.value,
            rationale=rationale,
            details={
                "task_id": spec.task_id,
                "lane": spec.lane,
                "output": result,
            },
        )

    def _run_graph(self) -> None:
        running: dict[Future[dict[str, Any]], _NodeSpec] = {}
        with ThreadPoolExecutor(
            max_workers=self.config.max_workers,
            thread_name_prefix="proof-carrying",
        ) as executor:
            while True:
                pending = [
                    self._specs[node.node_id]
                    for node in (
                        WorkflowNode.from_dict(item)
                        for item in self._state.get("nodes", [])
                    )
                    if node.status is WorkflowNodeStatus.PENDING
                    and node.node_id in self._specs
                    and node.kind is not WorkflowNodeKind.COMPILE_PLAN
                ]
                active_nodes = [self._node(spec.node_id) for spec in running.values()]
                launch_specs: list[_NodeSpec] = []
                for spec in sorted(pending, key=lambda item: item.node_id):
                    if (
                        len(running) + len(launch_specs)
                        >= self.config.max_workers
                    ):
                        break
                    ready, _ = self._dependency_state(spec)
                    if not ready:
                        continue
                    candidate = self._node(spec.node_id)
                    if any(
                        self._scope_conflict(candidate, active)
                        for active in active_nodes
                    ):
                        continue
                    self._upsert_node(
                        spec,
                        WorkflowNodeStatus.RUNNING,
                        increment_attempt=True,
                    )
                    self._decision(
                        kind="dispatch",
                        node_id=spec.node_id,
                        outcome="running",
                        rationale=(
                            "dependencies are accepted and no active scope or "
                            "shared-resource conflict exists"
                        ),
                        details={
                            "depends_on": list(spec.depends_on),
                            "scopes": list(candidate.scopes),
                            "shared_resources": list(candidate.shared_resources),
                        },
                        persist=False,
                    )
                    launch_specs.append(spec)
                    active_nodes.append(self._node(spec.node_id))
                if launch_specs:
                    # Persist the whole conflict-free admission slice before
                    # launching it.  This both makes crash recovery exact and
                    # avoids serializing independent callbacks behind one
                    # artifact write per node.
                    self._persist()
                    for spec in launch_specs:
                        future = executor.submit(self._execute_spec, spec)
                        running[future] = spec
                if running:
                    completed, _ = wait(
                        tuple(running), return_when=FIRST_COMPLETED
                    )
                    for future in sorted(
                        completed, key=lambda item: running[item].node_id
                    ):
                        spec = running.pop(future)
                        try:
                            output = future.result()
                            error = None
                        except BaseException as exc:
                            output = None
                            error = exc
                        self._finish_node(spec, output, error)
                    continue
                if not pending:
                    break
                # No runnable node remains.  Persist why every pending node is
                # blocked rather than silently ending the workflow.
                for spec in sorted(pending, key=lambda item: item.node_id):
                    _, missing = self._dependency_state(spec)
                    self._upsert_node(
                        spec,
                        WorkflowNodeStatus.BLOCKED,
                        reason="unsatisfied dependencies: " + ", ".join(missing),
                    )
                    self._decision(
                        kind="dependency_block",
                        node_id=spec.node_id,
                        outcome=WorkflowNodeStatus.BLOCKED.value,
                        rationale="authoritative dependency constraints remain unsatisfied",
                        details={"unsatisfied_dependencies": list(missing)},
                    )
                break

    @staticmethod
    def _counterexample_id(counterexample: Mapping[str, Any]) -> str:
        claimed = str(
            counterexample.get("counterexample_id")
            or counterexample.get("id")
            or ""
        ).strip()
        return claimed or _digest(counterexample)

    def _focused_repair_context(
        self, counterexample: Mapping[str, Any], repair_round: int
    ) -> dict[str, Any]:
        task_id = str(
            counterexample.get("task_id")
            or counterexample.get("subject_id")
            or ""
        )
        if not task_id:
            merged = [
                node.task_id
                for node in (
                    WorkflowNode.from_dict(item)
                    for item in self._state.get("nodes", [])
                )
                if node.kind is WorkflowNodeKind.MERGE
                and node.status is WorkflowNodeStatus.ACCEPTED
            ]
            task_id = sorted(merged)[0] if merged else ""
        scope = (
            self._task_scope(self._formal_task(task_id))
            if task_id
            and self._compilation is not None
            and self._compilation.plan is not None
            and any(item.task_id == task_id for item in self._compilation.plan.tasks)
            else ChangedScope()
        )
        return self._bounded_context({
            "workflow_id": self.workflow_id,
            "plan_id": self._state.get("plan_id", ""),
            "repair_round": repair_round,
            "task_id": task_id,
            "counterexample_id": self._counterexample_id(counterexample),
            "counterexample": _canonical_value(counterexample),
            "focus_scope": scope.to_dict(),
            "bounded_context": True,
            "repository_wide_analysis": False,
        })

    def _repair_counterexamples(self) -> None:
        monitor = self._node("runtime:monitor")
        output = monitor.output
        counterexamples = [
            _strict_mapping(item, field_name="counterexample")
            for item in output.get("counterexamples") or ()
            if isinstance(item, Mapping)
        ]
        known = {
            self._counterexample_id(item)
            for item in self._state.get("counterexamples", [])
        }
        for item in counterexamples:
            if self._counterexample_id(item) not in known:
                self._state.setdefault("counterexamples", []).append(item)
        self._persist()
        if monitor.status is WorkflowNodeStatus.ACCEPTED:
            return
        repaired = set(self._state.get("repaired_counterexample_ids", []))
        for repair_round in range(1, self.config.maximum_repairs + 1):
            outstanding = [
                item
                for item in self._state.get("counterexamples", [])
                if self._counterexample_id(item) not in repaired
            ]
            if not outstanding:
                break
            counterexample = outstanding[0]
            counterexample_id = self._counterexample_id(counterexample)
            node_id = f"repair:{counterexample_id}:{repair_round}"
            context = self._focused_repair_context(counterexample, repair_round)
            task_id = str(context.get("task_id") or "")
            scope = ChangedScope.from_mapping(context.get("focus_scope"))
            spec = _NodeSpec(
                node_id,
                WorkflowNodeKind.COUNTEREXAMPLE_REPAIR,
                task_id=task_id,
                scope=scope,
            )
            self._specs[node_id] = spec
            existing = next(
                (
                    WorkflowNode.from_dict(item)
                    for item in self._state.get("nodes", [])
                    if item["node_id"] == node_id
                ),
                None,
            )
            if existing is not None and existing.status is WorkflowNodeStatus.ACCEPTED:
                repaired.add(counterexample_id)
                continue
            self._state["status"] = WorkflowStatus.REPAIRING.value
            self._upsert_node(
                spec, WorkflowNodeStatus.RUNNING, increment_attempt=True
            )
            self._decision(
                kind="counterexample_repair_dispatch",
                node_id=node_id,
                outcome="running",
                rationale="runtime counterexample produced focused bounded repair work",
                details=context,
            )
            adapter = self.adapters.repair_dispatch or self.adapters.codex_dispatch
            try:
                repair = (
                    self._adapter_output(adapter(context))
                    if adapter is not None
                    else {
                        "status": "unavailable",
                        "accepted": False,
                        "reason": "no repair dispatcher configured",
                    }
                )
            except BaseException as exc:
                repair = {
                    "status": "failed",
                    "accepted": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:512],
                }
            actual = ChangedScope.from_mapping(repair)
            scope_accepted, violations = self._built_in_scope_check(scope, actual)
            validation_passed = (
                repair.get("validation_passed") is True
                or repair.get("tests_passed") is True
            )
            accepted = (
                self._accepted(repair)
                and scope_accepted
                and (validation_passed or not self.config.require_validation)
            )
            merge_output: dict[str, Any] = {}
            if accepted and self.adapters.merge is not None:
                merge_output = self._adapter_output(
                    self.adapters.merge(
                        {
                            **context,
                            "kind": "counterexample_repair_merge",
                            "repair": repair,
                            "scope_verification": {
                                "accepted": scope_accepted,
                                "violations": list(violations),
                            },
                        }
                    )
                )
                accepted = self._accepted(merge_output)
            elif accepted:
                accepted = False
                merge_output = {
                    "status": "unavailable",
                    "reason": "no merge adapter configured",
                }
            repair_output = {
                **repair,
                "scope_accepted": scope_accepted,
                "scope_violations": list(violations),
                "repair_merge": merge_output,
                "counterexample_id": counterexample_id,
                "accepted": accepted,
                "status": "accepted" if accepted else "rejected",
            }
            self._upsert_node(
                spec,
                (
                    WorkflowNodeStatus.ACCEPTED
                    if accepted
                    else WorkflowNodeStatus.REJECTED
                ),
                output=repair_output,
                reason="" if accepted else "focused repair was not admitted",
            )
            self._decision(
                kind="counterexample_repair",
                node_id=node_id,
                outcome="accepted" if accepted else "rejected",
                rationale=(
                    "focused repair passed scope, validation, and merge admission"
                    if accepted
                    else "focused repair failed scope, validation, or merge admission"
                ),
                details=repair_output,
            )
            if not accepted:
                break
            repaired.add(counterexample_id)
            self._state["repaired_counterexample_ids"] = sorted(repaired)
            self._persist()
            # Re-run only the runtime observation.  Formal proof results retain
            # their original assurance; a clean trace is not a proof upgrade.
            monitor_spec = self._specs["runtime:monitor"]
            self._upsert_node(
                monitor_spec,
                WorkflowNodeStatus.RUNNING,
                increment_attempt=True,
            )
            try:
                next_output = self._execute_monitor(monitor_spec)
                error = None
            except BaseException as exc:
                next_output = None
                error = exc
            self._finish_node(monitor_spec, next_output, error)
            new_items = [
                _strict_mapping(item, field_name="counterexample")
                for item in self._node("runtime:monitor").output.get(
                    "counterexamples", ()
                )
                if isinstance(item, Mapping)
            ]
            for item in new_items:
                identifier = self._counterexample_id(item)
                if not any(
                    self._counterexample_id(existing) == identifier
                    for existing in self._state["counterexamples"]
                ):
                    self._state["counterexamples"].append(item)
            if self._node("runtime:monitor").status is WorkflowNodeStatus.ACCEPTED:
                break
        self._state["repaired_counterexample_ids"] = sorted(repaired)
        self._persist()

    def _required_assurance_satisfied(self) -> bool:
        levels = [
            _assurance(item.get("assurance"))
            for item in self._state.get("evidence", [])
            if item.get("authoritative")
            and item.get("verdict") == EvidenceVerdict.ACCEPTED.value
            and (
                "plan_admission" in item.get("authoritative_for", [])
                or "formal_proof" in item.get("authoritative_for", [])
                or "proof_receipt_attestation"
                in item.get("authoritative_for", [])
            )
        ]
        highest = max(levels, key=lambda item: item.rank, default=AssuranceLevel.UNVERIFIED)
        return highest.satisfies(self.config.required_plan_assurance)

    def _finalize(self) -> ProofCarryingWorkflowResult:
        nodes = tuple(
            sorted(
                (
                    WorkflowNode.from_dict(item)
                    for item in self._state.get("nodes", [])
                ),
                key=lambda item: item.node_id,
            )
        )
        merges = [
            item
            for item in nodes
            if item.kind is WorkflowNodeKind.MERGE
        ]
        monitor = next(
            (item for item in nodes if item.kind is WorkflowNodeKind.RUNTIME_MONITOR),
            None,
        )
        compilation_accepted = self._node(
            "plan:compile"
        ).status is WorkflowNodeStatus.ACCEPTED
        validation_accepted = (
            "plan:verify" in self._specs
            and self._node("plan:verify").status is WorkflowNodeStatus.ACCEPTED
        )
        completed = (
            compilation_accepted
            and validation_accepted
            and bool(merges)
            and all(item.status is WorkflowNodeStatus.ACCEPTED for item in merges)
            and monitor is not None
            and monitor.status is WorkflowNodeStatus.ACCEPTED
            and self._required_assurance_satisfied()
        )
        if completed:
            status = WorkflowStatus.COMPLETED
            rationale = "plan, implementations, scopes, merges, and runtime trace were accepted"
        elif not compilation_accepted or not validation_accepted:
            status = WorkflowStatus.REJECTED
            rationale = "formal plan compilation or bounded validation was rejected"
        elif any(item.status is WorkflowNodeStatus.FAILED for item in nodes):
            status = WorkflowStatus.FAILED
            rationale = "one or more workflow nodes failed"
        else:
            status = WorkflowStatus.BLOCKED
            rationale = "required merge, monitoring, repair, or assurance evidence is absent"
        self._state["status"] = status.value
        if not self._state.get("decisions") or self._state["decisions"][-1].get(
            "kind"
        ) != "workflow_finalization":
            self._decision(
                kind="workflow_finalization",
                node_id="workflow",
                outcome=status.value,
                rationale=rationale,
                details={
                    "merged_task_ids": sorted(
                        item.task_id
                        for item in merges
                        if item.status is WorkflowNodeStatus.ACCEPTED
                    ),
                    "monitor_status": monitor.status.value if monitor else "missing",
                    "required_assurance": self.config.required_plan_assurance.value,
                },
            )
        else:
            self._persist()
        evidence = tuple(
            WorkflowEvidence.from_dict(item)
            for item in self._state.get("evidence", [])
        )
        authoritative_assurance = max(
            (
                item.assurance
                for item in evidence
                if item.authoritative
                and item.verdict is EvidenceVerdict.ACCEPTED
            ),
            key=lambda item: item.rank,
            default=AssuranceLevel.UNVERIFIED,
        )
        return ProofCarryingWorkflowResult(
            workflow_id=self.workflow_id,
            status=status,
            plan_id=str(self._state.get("plan_id") or ""),
            compilation_status=str(
                self._state.get("compilation", {}).get("status") or ""
            ),
            validation_status=str(
                self._state.get("validation", {}).get("status") or ""
            ),
            nodes=nodes,
            decisions=tuple(
                WorkflowDecision.from_dict(item)
                for item in self._state.get("decisions", [])
            ),
            evidence=evidence,
            merged_task_ids=tuple(
                sorted(
                    item.task_id
                    for item in merges
                    if item.status is WorkflowNodeStatus.ACCEPTED
                )
            ),
            counterexamples=tuple(self._state.get("counterexamples", [])),
            repaired_counterexample_ids=tuple(
                self._state.get("repaired_counterexample_ids", [])
            ),
            authoritative_assurance=authoritative_assurance,
            json_path=self.store.json_path,
            duckdb_path=self.store.duckdb_path,
            artifact_digest=str(self._state.get("artifact_digest") or ""),
        )

    def run(self) -> ProofCarryingWorkflowResult:
        """Run or resume the workflow until it reaches a durable terminal state."""

        self._load_or_create()
        if not self._compile():
            self._state["status"] = WorkflowStatus.REJECTED.value
            return self._finalize()
        self._build_specs()
        self._state["status"] = WorkflowStatus.RUNNING.value
        self._persist()
        self._run_graph()
        if (
            "runtime:monitor" in self._specs
            and self._node("runtime:monitor").status
            in {WorkflowNodeStatus.REJECTED, WorkflowNodeStatus.FAILED}
        ):
            self._repair_counterexamples()
        return self._finalize()


ProofCarryingPlanningWorkflow = ProofCarryingPlanner
ProofCarryingPlannerResult = ProofCarryingWorkflowResult


def execute_proof_carrying_workflow(
    source: Mapping[str, Any],
    *,
    artifact_path: Path | str | None = None,
    state_path: Path | str | None = None,
    adapters: WorkflowAdapters | None = None,
    config: ProofCarryingPlannerConfig | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ProofCarryingWorkflowResult:
    """Convenience wrapper for one complete or resumed workflow."""

    return ProofCarryingPlanner(
        source,
        artifact_path=artifact_path,
        state_path=state_path,
        adapters=adapters,
        config=config,
        **kwargs,
    ).run()


def replay_proof_carrying_workflow(
    artifact_path: Path | str,
) -> WorkflowReplay:
    """Verify paired artifacts and return their deterministic decision replay."""

    return ProofCarryingPlanner.replay(artifact_path)


__all__ = [
    "ChangedScope",
    "DEFAULT_DATABASE_FILENAME",
    "DEFAULT_STATE_FILENAME",
    "EvidenceRole",
    "EvidenceVerdict",
    "PROOF_CARRYING_DECISION_SCHEMA",
    "PROOF_CARRYING_EVIDENCE_SCHEMA",
    "PROOF_CARRYING_PLANNER_VERSION",
    "PROOF_CARRYING_RESULT_SCHEMA",
    "PROOF_CARRYING_WORKFLOW_SCHEMA",
    "ProofCarryingPlanner",
    "ProofCarryingPlannerConfig",
    "ProofCarryingPlannerError",
    "ProofCarryingPlannerResult",
    "ProofCarryingPlanningWorkflow",
    "ProofCarryingWorkflowResult",
    "ProverLane",
    "WorkflowAdapters",
    "WorkflowConfigurationError",
    "WorkflowDecision",
    "WorkflowEvidence",
    "WorkflowNode",
    "WorkflowNodeKind",
    "WorkflowNodeStatus",
    "WorkflowPersistenceError",
    "WorkflowReplay",
    "WorkflowStatus",
    "execute_proof_carrying_workflow",
    "replay_proof_carrying_workflow",
]
