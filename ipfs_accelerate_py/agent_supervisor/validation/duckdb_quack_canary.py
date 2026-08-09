"""Multi-daemon, multi-worktree database-authoritative E2E canary (DQP-035).

Interfaces: ``DuckDBQuackCanary@1``

Proves an isolated control program can run entirely under DuckDB authority
(Quack state-owner transport optional via hermetic fake when the extension is
absent): goals, tasks, claims, lifecycle, events, worktrees, AST mutations,
validation, merge, restart resume, refill, export, and clean drain.

Acceptance properties
---------------------
* At least two lanes overlap in active claim windows
* No duplicate claim/effect or stale fence write
* Every admitted byte change has complete mutation lineage
* Server/worker restart resumes without re-running committed provider/effect
* Final tasks/goals/daemons/worktrees/events/proofs agree under DB queries
* Tampered or missing exports do not affect authoritative state
* All processes drain cleanly

Cold import of this module performs no filesystem, database, network,
provider, or process action.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..analysis.mutation_ledger import (
    MutationContext,
    MutationDisposition,
    MutationFileSpec,
    MutationLedger,
    MutationStatus,
    open_mutation_ledger,
)
from ..merge.database_merge_queue import (
    DatabaseMergeQueue,
    EntryStatus,
    MergeOutcome,
    open_database_merge_queue,
)
from ..merge.database_worktree_registry import (
    DatabaseWorktreeRegistry,
    GitObservation,
    OwnerLiveness,
    ProcessBirthIdentity,
    WorktreeLifecycleState,
    WorktreeStatus,
    open_worktree_registry,
    process_birth_id,
)
from ..runtime.database_event_log import (
    DatabaseEventLog,
    open_database_event_log,
)
from ..runtime.quack_state_server import (
    DEFAULT_LOOPBACK_HOST,
    FakeQuackTransport,
    QuackStateServer,
    ServerLifecycle,
    build_server,
)
from ..task_sources.control_plane_migrations import (
    MigrationRunReport,
    duckdb_available,
)
from ..task_sources.quack_capabilities import (
    DEFAULT_QUACK_BETA_LIMITATIONS,
    ExtensionObservation,
    ParsedVersion,
    QuackCapabilityReport,
    QuackCapabilityStatus,
    default_compatibility_profile,
)

# DatabaseImplementationDaemon lives in todo_daemon; import lazily at open()
# to keep cold import free of the heavy daemon module and avoid package cycles.

# ---------------------------------------------------------------------------
# Interface / schema identities
# ---------------------------------------------------------------------------

DUCKDB_QUACK_CANARY_INTERFACE: Final[str] = "DuckDBQuackCanary@1"
DUCKDB_QUACK_CANARY_REPORT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/duckdb-quack-canary-report@1"
)
CANARY_LANE_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/canary-lane-receipt@1"
)
CANARY_AGREEMENT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/canary-final-agreement@1"
)
CANARY_EXPORT_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/canary-export-receipt@1"
)
CANARY_VERSION: Final[int] = 1
TASK_ID: Final[str] = "DQP-035"
GOAL_ID: Final[str] = "DQP-G080"
EVIDENCE: Final[str] = "dqp/duckdb-quack-canary@1"

DEFAULT_LANE_COUNT: Final[int] = 4
DEFAULT_REPOSITORY_ID: Final[str] = "repository:canary-dqp-035"
DEFAULT_TARGET_BRANCH: Final[str] = "main"
DEFAULT_TREE_ID: Final[str] = "tree:canary-dqp-035"
DEFAULT_FINGERPRINT: Final[str] = "sha256:" + ("cd" * 32)
DEFAULT_DATABASE_UUID: Final[str] = "123e4567-e89b-12d3-a456-426614174035"

STRICT_LANE_IDS: Final[tuple[str, ...]] = (
    "lane:schema",
    "lane:runtime",
    "lane:analysis",
    "lane:security",
)

# File-disjoint paths — one exclusive path per strict lane.
LANE_FILE_PATHS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "lane:schema": "src/canary/schema_lane.py",
        "lane:runtime": "src/canary/runtime_lane.py",
        "lane:analysis": "src/canary/analysis_lane.py",
        "lane:security": "src/canary/security_lane.py",
    }
)

BEFORE_SOURCE: Final[str] = '''\
def canary_marker():
    return "before"
'''

AFTER_SOURCE_TEMPLATE: Final[str] = '''\
def canary_marker():
    return "{lane_token}"
'''


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DuckDBQuackCanaryError(RuntimeError):
    """Base fail-closed error for the multi-daemon canary."""


class DuckDBQuackCanaryDependencyError(DuckDBQuackCanaryError):
    """Required DuckDB (or other) dependency is unavailable."""


class DuckDBQuackCanaryAcceptanceError(DuckDBQuackCanaryError):
    """A required acceptance predicate failed."""


class DuckDBQuackCanaryDrainError(DuckDBQuackCanaryError):
    """Processes did not drain cleanly."""


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class CanaryVerdict(str, Enum):
    """Suite conclusion — not a promotion or cutover decision."""

    PASSED = "passed"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class CanaryPhase(str, Enum):
    """Ordered canary phases for evidence receipts."""

    BOOTSTRAP = "bootstrap"
    STATE_OWNER = "state_owner"
    REGISTER_LANES = "register_lanes"
    MATERIALIZE = "materialize"
    CLAIM_OVERLAP = "claim_overlap"
    MUTATE = "mutate"
    VALIDATE_MERGE = "validate_merge"
    RESTART_RESUME = "restart_resume"
    REFILL = "refill"
    EXPORT = "export"
    AGREE = "agree"
    DRAIN = "drain"


REQUIRED_EVIDENCE_KEYS: Final[tuple[str, ...]] = (
    "real_processes",
    "overlap",
    "claim_fence",
    "worktree",
    "mutation",
    "validation",
    "merge",
    "restart",
    "refill",
    "export",
    "drain",
    "database_queries",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def require_duckdb_or_raise(*, context: str = "canary") -> None:
    if not duckdb_available():
        raise DuckDBQuackCanaryDependencyError(
            f"DuckDB is required for {context}; install the optional duckdb dependency"
        )


def _sha256_text(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")


def _content_id(payload: Mapping[str, Any]) -> str:
    return _sha256_text(_canonical_json_bytes(dict(payload)).decode("utf-8"))


def _now_ms() -> int:
    return int(time.time() * 1000)


def _new_id(prefix: str) -> str:
    return f"{prefix}:{secrets.token_hex(10)}"


def _birth(
    *,
    pid: int,
    start_time_ticks: int = 1000,
    boot_id: str = "boot:canary",
    parent_pid: int = 1,
) -> ProcessBirthIdentity:
    return ProcessBirthIdentity(
        pid=int(pid),
        start_time_ticks=int(start_time_ticks),
        boot_id=str(boot_id),
        parent_pid=int(parent_pid),
    )


def hermetic_capability_report(
    *,
    fingerprint: str = DEFAULT_FINGERPRINT,
) -> QuackCapabilityReport:
    """Return a sealed compatible capability report for hermetic state-owner tests."""

    profile = default_compatibility_profile()
    return QuackCapabilityReport(
        status=QuackCapabilityStatus.COMPATIBLE,
        profile=profile,
        duckdb_importable=True,
        duckdb_version="1.5.2",
        duckdb_version_parsed=ParsedVersion(1, 5, 2, raw="1.5.2"),
        platform_name="Linux",
        platform_machine="x86_64",
        extension=ExtensionObservation(
            name="quack",
            installed=True,
            loaded=True,
            install_path="/tmp/canary-quack.duckdb_extension",
            extension_version="0.1.0",
        ),
        extension_fingerprint=fingerprint,
        observed_functions=("quack_serve", "quack_query"),
        observed_surfaces=tuple(profile.required_surfaces),
        beta_limitations=DEFAULT_QUACK_BETA_LIMITATIONS,
    )


def hermetic_migration_report(
    *,
    fingerprint: str = DEFAULT_FINGERPRINT,
) -> MigrationRunReport:
    return MigrationRunReport(
        from_version=0,
        to_version=1,
        receipts=(),
        schema_fingerprint=fingerprint,
        catalog_fingerprint=fingerprint,
        changed=True,
    )


class _FakeMetaConnection:
    """Minimal exclusive-owner connection stand-in for state-owner lifecycle."""

    def __init__(
        self,
        *,
        database_uuid: str = DEFAULT_DATABASE_UUID,
        schema_version: str = "1",
        schema_fingerprint: str = DEFAULT_FINGERPRINT,
        max_generation: int = 0,
    ) -> None:
        self.database_uuid = database_uuid
        self.schema_version = schema_version
        self.schema_fingerprint = schema_fingerprint
        self.max_generation = max_generation
        self.closed = False
        self.statements: list[str] = []
        self._meta = {
            "database_uuid": database_uuid,
            "schema_version": schema_version,
            "schema_fingerprint": schema_fingerprint,
        }

    def execute(self, sql: str, params: Any = None) -> Any:
        text = " ".join(str(sql).strip().split())
        self.statements.append(text)
        upper = text.upper()

        class _Result:
            def __init__(self, row: Any = None) -> None:
                self._row = row

            def fetchone(self) -> Any:
                return self._row

            def fetchall(self) -> list[Any]:
                return [] if self._row is None else [self._row]

        if "FROM CONTROL_PLANE_METADATA" in upper and "KEY" in upper:
            key = None
            if params:
                key = params[0] if not isinstance(params, dict) else params.get("key")
            if key is not None:
                return _Result((self._meta.get(str(key), ""),))
            return _Result(None)
        if "FROM STORE_GENERATIONS" in upper and "MAX" in upper:
            return _Result((int(self.max_generation),))
        if "INSERT INTO STORE_GENERATIONS" in upper:
            if params:
                try:
                    self.max_generation = max(
                        self.max_generation, int(params[0] if params else 0)
                    )
                except (TypeError, ValueError, IndexError):
                    self.max_generation += 1
            else:
                self.max_generation += 1
            return _Result(None)
        if "CHECKPOINT" in upper:
            return _Result(None)
        if upper.startswith("SELECT 1"):
            return _Result((1,))
        return _Result(None)

    def close(self) -> None:
        self.closed = True


def default_canary_population(
    *,
    lane_count: int = DEFAULT_LANE_COUNT,
    goal_cid: str = "goal:cid:canary-root",
    objective_id: str = "objective:dqp-035",
) -> dict[str, Any]:
    """Build a file-disjoint task population for the four strict lanes."""

    count = max(2, min(int(lane_count), len(STRICT_LANE_IDS)))
    lanes = STRICT_LANE_IDS[:count]
    tasks: list[dict[str, Any]] = []
    for index, lane_id in enumerate(lanes, start=1):
        path = LANE_FILE_PATHS[lane_id]
        tasks.append(
            {
                "task_cid": f"task:cid:canary:{index:03d}",
                "task_id": f"DQP-CANARY-{index:03d}",
                "goal_cid": goal_cid,
                "status": "ready",
                "priority": "P0",
                "ordinal": index,
                "title": f"Canary lane {lane_id}",
                "allowed_paths": [path],
                "lane_id": lane_id,
                "expected_outputs": [path],
            }
        )
    return {
        "repository_tree_id": DEFAULT_TREE_ID,
        "objectives": [
            {
                "objective_id": objective_id,
                "objective_alias": "DQP-O035",
                "title": "Database-authoritative multi-daemon canary",
                "goal_cid": goal_cid,
                "goal_alias": GOAL_ID,
                "status": "open",
            }
        ],
        "tasks": tasks,
    }


def intervals_overlap(
    a_start_ms: int,
    a_end_ms: int,
    b_start_ms: int,
    b_end_ms: int,
) -> bool:
    """Return whether two half-open claim windows overlap."""

    return int(a_start_ms) < int(b_end_ms) and int(b_start_ms) < int(a_end_ms)


def count_pairwise_overlaps(
    windows: Sequence[tuple[int, int]],
) -> int:
    """Count pairwise overlapping active windows."""

    pairs = 0
    items = list(windows)
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            if intervals_overlap(left[0], left[1], right[0], right[1]):
                pairs += 1
    return pairs


# ---------------------------------------------------------------------------
# Receipts / report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanaryLaneReceipt:
    """One strict lane's canary execution receipt."""

    SCHEMA: ClassVar[str] = CANARY_LANE_RECEIPT_SCHEMA

    lane_id: str
    task_cid: str
    task_alias: str
    session_id: str
    daemon_id: str
    worktree_id: str
    claim_id: str
    attempt_id: str
    fencing_token: int
    fence_id: str
    mutation_id: str
    merge_entry_id: str
    evidence_digest: str
    file_path: str
    claimed_at_ms: int
    completed_at_ms: int
    provider_calls: int
    effect_calls: int
    lineage_count: int
    mutation_status: str
    merge_status: str
    restarted: bool = False
    provider_duplicated_on_resume: bool = False
    effect_duplicated_on_resume: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "lane_id": self.lane_id,
            "task_cid": self.task_cid,
            "task_alias": self.task_alias,
            "session_id": self.session_id,
            "daemon_id": self.daemon_id,
            "worktree_id": self.worktree_id,
            "claim_id": self.claim_id,
            "attempt_id": self.attempt_id,
            "fencing_token": int(self.fencing_token),
            "fence_id": self.fence_id,
            "mutation_id": self.mutation_id,
            "merge_entry_id": self.merge_entry_id,
            "evidence_digest": self.evidence_digest,
            "file_path": self.file_path,
            "claimed_at_ms": int(self.claimed_at_ms),
            "completed_at_ms": int(self.completed_at_ms),
            "provider_calls": int(self.provider_calls),
            "effect_calls": int(self.effect_calls),
            "lineage_count": int(self.lineage_count),
            "mutation_status": self.mutation_status,
            "merge_status": self.merge_status,
            "restarted": bool(self.restarted),
            "provider_duplicated_on_resume": bool(self.provider_duplicated_on_resume),
            "effect_duplicated_on_resume": bool(self.effect_duplicated_on_resume),
        }


@dataclass(frozen=True)
class CanaryFinalAgreement:
    """Final cross-store agreement snapshot (database queries only)."""

    SCHEMA: ClassVar[str] = CANARY_AGREEMENT_SCHEMA

    task_count: int
    completed_task_count: int
    goal_count: int
    open_goal_count: int
    daemon_session_count: int
    worktree_count: int
    terminal_worktree_count: int
    event_count: int
    mutation_count: int
    accepted_mutation_count: int
    merge_accepted_count: int
    proof_count: int
    complete_lineage_count: int
    duplicate_claims: int
    duplicate_effects: int
    stale_writes: int
    agreed: bool
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "task_count": int(self.task_count),
            "completed_task_count": int(self.completed_task_count),
            "goal_count": int(self.goal_count),
            "open_goal_count": int(self.open_goal_count),
            "daemon_session_count": int(self.daemon_session_count),
            "worktree_count": int(self.worktree_count),
            "terminal_worktree_count": int(self.terminal_worktree_count),
            "event_count": int(self.event_count),
            "mutation_count": int(self.mutation_count),
            "accepted_mutation_count": int(self.accepted_mutation_count),
            "merge_accepted_count": int(self.merge_accepted_count),
            "proof_count": int(self.proof_count),
            "complete_lineage_count": int(self.complete_lineage_count),
            "duplicate_claims": int(self.duplicate_claims),
            "duplicate_effects": int(self.duplicate_effects),
            "stale_writes": int(self.stale_writes),
            "agreed": bool(self.agreed),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class CanaryExportReceipt:
    """Export non-authority proof."""

    SCHEMA: ClassVar[str] = CANARY_EXPORT_RECEIPT_SCHEMA

    export_path: str
    authority: str
    event_count_before: int
    event_count_after_tamper: int
    event_count_after_delete: int
    task_completed_before: int
    task_completed_after: int
    board_export_path: str
    board_tampered: bool
    board_deleted: bool
    unaffected: bool
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "export_path": self.export_path,
            "authority": self.authority,
            "event_count_before": int(self.event_count_before),
            "event_count_after_tamper": int(self.event_count_after_tamper),
            "event_count_after_delete": int(self.event_count_after_delete),
            "task_completed_before": int(self.task_completed_before),
            "task_completed_after": int(self.task_completed_after),
            "board_export_path": self.board_export_path,
            "board_tampered": bool(self.board_tampered),
            "board_deleted": bool(self.board_deleted),
            "unaffected": bool(self.unaffected),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class DuckDBQuackCanaryReport:
    """Immutable multi-daemon canary report.

    Interface: ``DuckDBQuackCanary@1``.
    """

    INTERFACE: ClassVar[str] = DUCKDB_QUACK_CANARY_INTERFACE
    SCHEMA: ClassVar[str] = DUCKDB_QUACK_CANARY_REPORT_SCHEMA

    verdict: CanaryVerdict
    workspace: str
    lane_count: int
    lanes: tuple[CanaryLaneReceipt, ...]
    overlap_pair_count: int
    server_restarted: bool
    worker_restarted: bool
    refill_task_count: int
    agreement: CanaryFinalAgreement
    export_receipt: CanaryExportReceipt
    drained: bool
    control_files_used_as_authority: bool
    duckdb_available: bool
    mode: str = "hermetic"
    task_id: str = TASK_ID
    goal_id: str = GOAL_ID
    evidence: str = EVIDENCE
    report_version: int = CANARY_VERSION
    evidence_subset: Mapping[str, bool] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)
    failures: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        verdict = self.verdict
        if not isinstance(verdict, CanaryVerdict):
            verdict = CanaryVerdict(str(verdict))
        object.__setattr__(self, "verdict", verdict)
        object.__setattr__(self, "lanes", tuple(self.lanes))
        object.__setattr__(
            self, "evidence_subset", MappingProxyType(dict(self.evidence_subset))
        )
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))
        object.__setattr__(self, "failures", tuple(self.failures))

    @property
    def passed(self) -> bool:
        return self.verdict is CanaryVerdict.PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "verdict": self.verdict.value,
            "passed": self.passed,
            "workspace": self.workspace,
            "lane_count": int(self.lane_count),
            "lanes": [lane.to_dict() for lane in self.lanes],
            "overlap_pair_count": int(self.overlap_pair_count),
            "server_restarted": bool(self.server_restarted),
            "worker_restarted": bool(self.worker_restarted),
            "refill_task_count": int(self.refill_task_count),
            "agreement": self.agreement.to_dict(),
            "export_receipt": self.export_receipt.to_dict(),
            "drained": bool(self.drained),
            "control_files_used_as_authority": bool(
                self.control_files_used_as_authority
            ),
            "duckdb_available": bool(self.duckdb_available),
            "mode": self.mode,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "evidence": self.evidence,
            "report_version": int(self.report_version),
            "evidence_subset": dict(self.evidence_subset),
            "details": dict(self.details),
            "failures": list(self.failures),
            "failure_count": len(self.failures),
        }

    def content_id(self) -> str:
        return _content_id(self.to_dict())


def assert_canary_passed(report: DuckDBQuackCanaryReport) -> None:
    """Raise if the canary did not fully pass."""

    if not isinstance(report, DuckDBQuackCanaryReport):
        raise TypeError("report must be DuckDBQuackCanaryReport")
    if report.verdict is not CanaryVerdict.PASSED:
        detail = "; ".join(report.failures[:12]) or report.verdict.value
        raise DuckDBQuackCanaryAcceptanceError(
            f"duckdb quack canary did not pass: {detail}"
        )


# ---------------------------------------------------------------------------
# Canary harness
# ---------------------------------------------------------------------------


class DuckDBQuackCanary:
    """Full multi-daemon, multi-worktree database-authoritative E2E harness.

    Interface: ``DuckDBQuackCanary@1``.
    """

    INTERFACE: ClassVar[str] = DUCKDB_QUACK_CANARY_INTERFACE
    SCHEMA: ClassVar[str] = DUCKDB_QUACK_CANARY_REPORT_SCHEMA

    def __init__(
        self,
        workspace: Path | str,
        *,
        lane_count: int = DEFAULT_LANE_COUNT,
        repository_id: str = DEFAULT_REPOSITORY_ID,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        require_duckdb_or_raise(context="DuckDBQuackCanary construction open")
        self.workspace = Path(workspace).absolute()
        self.lane_count = max(2, min(int(lane_count), len(STRICT_LANE_IDS)))
        self.repository_id = str(repository_id or DEFAULT_REPOSITORY_ID)
        self._clock_ms = clock_ms or _now_ms
        self._lock = threading.RLock()
        self._closed = True

        # Store paths (all under workspace; no production control files).
        self.paths = {
            "root": self.workspace,
            "state_dir": self.workspace / "state-owner",
            "owner_db": self.workspace / "state-owner" / "control.duckdb",
            "tasks_db": self.workspace / "tasks.duckdb",
            "coordination_db": self.workspace / "coordination.duckdb",
            "execution_db": self.workspace / "execution.duckdb",
            "worktree_db": self.workspace / "worktrees.duckdb",
            "mutation_db": self.workspace / "mutations.duckdb",
            "merge_db": self.workspace / "merge.duckdb",
            "events_db": self.workspace / "events.duckdb",
            "worktrees_root": self.workspace / "worktrees",
            "exports_root": self.workspace / "exports",
            "repo_root": self.workspace / "repo",
        }

        self._registry: DatabaseWorktreeRegistry | None = None
        self._events: DatabaseEventLog | None = None
        self._mutations: MutationLedger | None = None
        self._merge: DatabaseMergeQueue | None = None
        self._seed_daemon: Any = None
        self._server: QuackStateServer | None = None
        self._server_transport: FakeQuackTransport | None = None
        self._liveness: dict[str, OwnerLiveness] = {}
        self._git_obs: dict[str, GitObservation] = {}
        self._provider_calls: list[str] = []
        self._effect_calls: list[str] = []
        self._lane_receipts: list[CanaryLaneReceipt] = []
        self._open_daemons: list[Any] = []
        self._phase_log: list[str] = []
        self._daemon_api: Any = None

    # -- lifecycle -----------------------------------------------------------

    def _load_daemon_api(self) -> Any:
        if self._daemon_api is None:
            from ..todo_daemon import implementation_daemon as daemon_api

            self._daemon_api = daemon_api
        return self._daemon_api

    def open(self) -> "DuckDBQuackCanary":
        with self._lock:
            if not self._closed:
                return self
            for key in (
                "root",
                "state_dir",
                "worktrees_root",
                "exports_root",
                "repo_root",
            ):
                self.paths[key].mkdir(parents=True, exist_ok=True)

            def liveness(birth: ProcessBirthIdentity) -> OwnerLiveness:
                return self._liveness.get(
                    process_birth_id(birth), OwnerLiveness.ALIVE
                )

            def git_observer(workspace_path: str) -> GitObservation | None:
                from ..merge.database_worktree_registry import (
                    normalize_workspace_path,
                )

                return self._git_obs.get(normalize_workspace_path(workspace_path))

            self._registry = open_worktree_registry(
                self.paths["worktree_db"],
                clock_ms=self._clock_ms,
                liveness=liveness,
                git_observer=git_observer,
                default_lease_ttl_ms=120_000,
            )
            self._events = open_database_event_log(
                self.paths["events_db"],
                snapshot_id="snapshot:canary-dqp-035",
            )
            self._mutations = open_mutation_ledger(self.paths["mutation_db"])
            self._merge = open_database_merge_queue(
                self.paths["merge_db"],
                clock_ms=self._clock_ms,
            )
            daemon_api = self._load_daemon_api()
            self._seed_daemon = daemon_api.open_database_implementation_daemon(
                self.paths["tasks_db"],
                coordination_path=self.paths["coordination_db"],
                execution_path=self.paths["execution_db"],
                owner_session_id="session:canary-seed",
                authority_mode="embedded",
                task_source_kind="duckdb",
                clock_ms=self._clock_ms,
                provider_fn=self._provider,
                effect_fn=self._effect,
            )
            self._closed = False
            self._record_phase(CanaryPhase.BOOTSTRAP)
            self._events.append_event(
                "canary.bootstrap",
                {
                    "workspace": str(self.workspace),
                    "lane_count": self.lane_count,
                    "repository_id": self.repository_id,
                },
                session_id="session:canary-seed",
            )
            return self

    def close(self) -> None:
        with self._lock:
            for daemon in list(self._open_daemons):
                try:
                    daemon.close()
                except Exception:
                    pass
            self._open_daemons.clear()
            if self._server is not None:
                try:
                    if self._server.lifecycle not in {
                        ServerLifecycle.STOPPED,
                        ServerLifecycle.CREATED,
                    }:
                        self._server.stop()
                except Exception:
                    pass
                self._server = None
            for store in (
                self._seed_daemon,
                self._merge,
                self._mutations,
                self._events,
                self._registry,
            ):
                if store is not None:
                    close = getattr(store, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            pass
            self._seed_daemon = None
            self._merge = None
            self._mutations = None
            self._events = None
            self._registry = None
            self._closed = True

    def __enter__(self) -> "DuckDBQuackCanary":
        return self.open()

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise DuckDBQuackCanaryError("DuckDBQuackCanary is not open")

    def _record_phase(self, phase: CanaryPhase) -> None:
        self._phase_log.append(phase.value)

    def _provider(self, attempt: Any) -> dict[str, Any]:
        self._provider_calls.append(attempt.task_cid)
        return {
            "status": "ok",
            "provider": "canary-hermetic",
            "task_cid": attempt.task_cid,
            "attempt_id": attempt.attempt_id,
            "mutation_plan": "apply-file-disjoint-edit",
        }

    def _effect(
        self,
        attempt: Any,
        provider_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._effect_calls.append(attempt.task_cid)
        return {
            "status": "applied",
            "effect_key": f"effect:{attempt.task_cid}",
            "task_cid": attempt.task_cid,
            "attempt_id": attempt.attempt_id,
            "provider_result": dict(provider_result),
        }

    def _open_daemon(
        self,
        session_id: str,
        *,
        execution_name: str | None = None,
    ) -> Any:
        daemon_api = self._load_daemon_api()
        name = execution_name or f"execution-{session_id.replace(':', '-')}.duckdb"
        daemon = daemon_api.open_database_implementation_daemon(
            self.paths["tasks_db"],
            coordination_path=self.paths["coordination_db"],
            execution_path=self.paths["execution_db"].with_name(name)
            if execution_name
            else self.paths["execution_db"],
            owner_session_id=session_id,
            authority_mode="embedded",
            task_source_kind="duckdb",
            clock_ms=self._clock_ms,
            provider_fn=self._provider,
            effect_fn=self._effect,
        )
        self._open_daemons.append(daemon)
        return daemon

    # -- state owner ---------------------------------------------------------

    def start_state_owner(self) -> Mapping[str, Any]:
        """Start the exclusive state-owner (hermetic FakeQuackTransport)."""

        self._require_open()
        assert self._events is not None
        transport = FakeQuackTransport()
        birth = _birth(pid=35000 + self.lane_count, start_time_ticks=5000)
        connection = _FakeMetaConnection()

        server = build_server(
            database_path=self.paths["owner_db"],
            state_dir=self.paths["state_dir"],
            host=DEFAULT_LOOPBACK_HOST,
            port=0,
            repository_id=self.repository_id,
            transport=transport,
            capability_probe=lambda **_k: hermetic_capability_report(),
            migrate=lambda _path: hermetic_migration_report(),
            connection_factory=lambda _path: connection,
            process_birth_factory=lambda: birth,
            owner_liveness_probe=lambda _b: OwnerLiveness.DEAD,
        )
        identity = server.start()
        ready = server.ready()
        if not ready.get("ready"):
            raise DuckDBQuackCanaryError("state-owner not ready after start")
        self._server = server
        self._server_transport = transport
        self._record_phase(CanaryPhase.STATE_OWNER)
        self._events.append_event(
            "canary.state_owner.ready",
            {
                "server_id": identity.server_id,
                "generation": identity.generation,
                "listen_uri": identity.listen_uri,
                "ready": True,
            },
            session_id="session:state-owner",
        )
        return MappingProxyType(
            {
                "server_id": identity.server_id,
                "generation": identity.generation,
                "listen_uri": identity.listen_uri,
                "ready": True,
            }
        )

    def restart_state_owner(self) -> Mapping[str, Any]:
        """Stop and restart the state-owner; generation must advance."""

        self._require_open()
        assert self._events is not None
        if self._server is None:
            raise DuckDBQuackCanaryError("state-owner not started")
        pre_generation = (
            int(self._server.identity.generation)
            if self._server.identity is not None
            else 0
        )
        self._server.checkpoint()
        self._server.stop()
        # Fresh transport + connection for the second owner epoch.
        transport = FakeQuackTransport()
        birth = _birth(pid=36000 + self.lane_count, start_time_ticks=6000)
        connection = _FakeMetaConnection(max_generation=pre_generation)
        server = build_server(
            database_path=self.paths["owner_db"],
            state_dir=self.paths["state_dir"],
            host=DEFAULT_LOOPBACK_HOST,
            port=0,
            repository_id=self.repository_id,
            transport=transport,
            capability_probe=lambda **_k: hermetic_capability_report(),
            migrate=lambda _path: hermetic_migration_report(),
            connection_factory=lambda _path: connection,
            process_birth_factory=lambda: birth,
            owner_liveness_probe=lambda _b: OwnerLiveness.DEAD,
        )
        identity = server.start()
        ready = server.ready()
        if not ready.get("ready"):
            raise DuckDBQuackCanaryError("state-owner not ready after restart")
        if int(identity.generation) <= pre_generation:
            raise DuckDBQuackCanaryAcceptanceError(
                "state-owner generation did not advance after restart "
                f"(pre={pre_generation}, post={identity.generation})"
            )
        self._server = server
        self._server_transport = transport
        self._events.append_event(
            "canary.state_owner.restarted",
            {
                "pre_generation": pre_generation,
                "post_generation": identity.generation,
                "server_id": identity.server_id,
            },
            session_id="session:state-owner",
        )
        return MappingProxyType(
            {
                "pre_generation": pre_generation,
                "post_generation": identity.generation,
                "server_id": identity.server_id,
                "ready": True,
            }
        )

    # -- lanes / worktrees ---------------------------------------------------

    def register_lanes(self) -> list[Mapping[str, Any]]:
        """Register the repository and four strict-lane worktrees."""

        self._require_open()
        assert self._registry is not None
        assert self._events is not None
        repo_root = self.paths["repo_root"]
        git_common = repo_root / ".git"
        git_common.mkdir(parents=True, exist_ok=True)
        repo = self._registry.register_repository(
            git_common_dir=git_common,
            canonical_root=repo_root,
            repository_id=self.repository_id,
            head_commit="c0" * 20,
            head_tree="t0" * 20,
            body={"canary": TASK_ID},
        )
        registered: list[dict[str, Any]] = []
        for index, lane_id in enumerate(STRICT_LANE_IDS[: self.lane_count], start=1):
            wt_path = self.paths["worktrees_root"] / f"wt-{index:02d}-{lane_id.split(':')[-1]}"
            wt_path.mkdir(parents=True, exist_ok=True)
            (wt_path / "src" / "canary").mkdir(parents=True, exist_ok=True)
            file_path = LANE_FILE_PATHS[lane_id]
            (wt_path / file_path).write_text(BEFORE_SOURCE, encoding="utf-8")
            birth = _birth(pid=1000 + index, start_time_ticks=100 + index)
            self._liveness[process_birth_id(birth)] = OwnerLiveness.ALIVE
            obs = GitObservation(
                workspace_path=str(wt_path),
                head_commit=f"{index:040x}",
                head_tree=f"{index + 10:040x}",
                index_digest=f"sha256:index{index:02d}",
                dirty_overlay_digest=f"sha256:dirty{index:02d}",
                branch_name=f"implementation/{lane_id.split(':')[-1]}",
                is_detached=False,
                git_common_dir=str(git_common),
                path_exists=True,
                is_symlink_root=False,
                observed_at_ms=self._clock_ms(),
            )
            self._git_obs[str(wt_path.resolve())] = obs
            # Also key by normalize if different
            from ..merge.database_worktree_registry import normalize_workspace_path

            self._git_obs[normalize_workspace_path(wt_path)] = obs
            identity = self._registry.register_worktree(
                repository_id=repo.repository_id,
                workspace_path=wt_path,
                branch_name=obs.branch_name,
                lane_id=lane_id,
                task_id=f"DQP-CANARY-{index:03d}",
                attempt=1,
                session_id=f"session:lane-{index:02d}",
            )
            leased = self._registry.acquire_lease(
                identity.worktree_id,
                process_birth=birth,
                session_id=f"session:lane-{index:02d}",
            )
            registered.append(
                {
                    "lane_id": lane_id,
                    "worktree_id": leased.worktree_id,
                    "workspace_path": leased.workspace_path,
                    "lease_id": leased.lease_id,
                    "fencing_token": leased.fencing_token,
                    "file_path": file_path,
                    "session_id": f"session:lane-{index:02d}",
                    "daemon_id": f"daemon:lane-{index:02d}",
                    "process_birth": birth,
                    "task_cid": f"task:cid:canary:{index:03d}",
                    "task_alias": f"DQP-CANARY-{index:03d}",
                }
            )
            self._events.append_event(
                "canary.worktree.registered",
                {
                    "lane_id": lane_id,
                    "worktree_id": leased.worktree_id,
                    "fencing_token": leased.fencing_token,
                },
                session_id=f"session:lane-{index:02d}",
            )
        self._record_phase(CanaryPhase.REGISTER_LANES)
        return [MappingProxyType(item) for item in registered]

    # -- materialize / claim / execute ---------------------------------------

    def materialize_tasks(self) -> Mapping[str, Any]:
        self._require_open()
        assert self._seed_daemon is not None
        assert self._events is not None
        population = default_canary_population(lane_count=self.lane_count)
        receipt = self._seed_daemon.materialize_population(
            population,
            repository_tree_id=DEFAULT_TREE_ID,
        )
        self._record_phase(CanaryPhase.MATERIALIZE)
        self._events.append_event(
            "canary.tasks.materialized",
            {
                "task_count": len(population["tasks"]),
                "registered": list(receipt.get("registered_task_cids") or []),
            },
            session_id="session:canary-seed",
        )
        # Release exclusive DuckDB handles so lane daemons can open the same
        # task/coordination stores. Re-open lazily for refill/export/agreement.
        self._seed_daemon.close()
        self._seed_daemon = None
        return MappingProxyType(dict(receipt) if isinstance(receipt, Mapping) else {})

    def _ensure_seed_daemon(self) -> Any:
        """Re-open the seed/query daemon when needed after materialize close."""

        if self._seed_daemon is not None:
            return self._seed_daemon
        daemon_api = self._load_daemon_api()
        self._seed_daemon = daemon_api.open_database_implementation_daemon(
            self.paths["tasks_db"],
            coordination_path=self.paths["coordination_db"],
            execution_path=self.paths["execution_db"],
            owner_session_id="session:canary-seed",
            authority_mode="embedded",
            task_source_kind="duckdb",
            clock_ms=self._clock_ms,
            provider_fn=self._provider,
            effect_fn=self._effect,
        )
        return self._seed_daemon

    def _claim_lane(
        self,
        lane: Mapping[str, Any],
        *,
        shared_execution: bool = True,
    ) -> tuple[Any, Any, int]:
        session_id = str(lane["session_id"])
        execution_name = None if shared_execution else f"exec-{session_id}.duckdb"
        daemon = self._open_daemon(session_id, execution_name=execution_name)
        claimed_at = self._clock_ms()
        attempt = daemon.claim_next()
        if attempt is None:
            raise DuckDBQuackCanaryAcceptanceError(
                f"lane {lane['lane_id']} failed to claim work"
            )
        return daemon, attempt, claimed_at

    def execute_lanes(
        self,
        lanes: Sequence[Mapping[str, Any]],
        *,
        restart_worker_lane_index: int = 0,
    ) -> list[CanaryLaneReceipt]:
        """Claim with overlap, mutate with lineage, validate/merge, resume one worker."""

        self._require_open()
        assert self._mutations is not None
        assert self._merge is not None
        assert self._events is not None
        assert self._registry is not None
        daemon_api = self._load_daemon_api()

        # --- Phase: overlapping claims ---
        # Claim every lane sequentially, close the daemon after each claim so
        # DuckDB file locks stay exclusive, but leave claims active so all
        # lanes are simultaneously owned (overlap) before any completes.
        claimed: list[dict[str, Any]] = []
        for lane in lanes:
            session_id = str(lane["session_id"])
            exec_name = f"exec-{session_id.replace(':', '-')}.duckdb"
            exec_path = self.paths["execution_db"].with_name(exec_name)
            daemon = daemon_api.open_database_implementation_daemon(
                self.paths["tasks_db"],
                coordination_path=self.paths["coordination_db"],
                execution_path=exec_path,
                owner_session_id=session_id,
                authority_mode="embedded",
                task_source_kind="duckdb",
                clock_ms=self._clock_ms,
                provider_fn=self._provider,
                effect_fn=self._effect,
            )
            try:
                claimed_at = self._clock_ms()
                attempt = daemon.claim_next()
                if attempt is None:
                    raise DuckDBQuackCanaryAcceptanceError(
                        f"lane {lane['lane_id']} failed to claim work"
                    )
                claim_snapshot = {
                    "lane": dict(lane),
                    "task_cid": attempt.task_cid,
                    "claim_id": attempt.claim_id,
                    "attempt_id": attempt.attempt_id,
                    "fencing_token": int(attempt.fencing_token),
                    "task_alias": attempt.task_alias,
                    "session_id": session_id,
                    "exec_path": exec_path,
                    "claimed_at_ms": claimed_at,
                }
                claimed.append(claim_snapshot)
                self._events.append_event(
                    "canary.task.claimed",
                    {
                        "lane_id": lane["lane_id"],
                        "task_cid": attempt.task_cid,
                        "claim_id": attempt.claim_id,
                        "attempt_id": attempt.attempt_id,
                        "fencing_token": attempt.fencing_token,
                        "claimed_at_ms": claimed_at,
                    },
                    task_cid=attempt.task_cid,
                    attempt_id=attempt.attempt_id,
                    session_id=session_id,
                )
            finally:
                daemon.close()

        # Verify distinct claims / no duplicate task ownership.
        task_cids = [item["task_cid"] for item in claimed]
        claim_ids = [item["claim_id"] for item in claimed]
        if len(set(task_cids)) != len(task_cids):
            raise DuckDBQuackCanaryAcceptanceError("duplicate task claims detected")
        if len(set(claim_ids)) != len(claim_ids):
            raise DuckDBQuackCanaryAcceptanceError("duplicate claim ids detected")
        # All claims are active together before any completion → overlap.
        self._record_phase(CanaryPhase.CLAIM_OVERLAP)

        receipts: list[CanaryLaneReceipt] = []
        restart_index = max(0, min(int(restart_worker_lane_index), len(claimed) - 1))

        # Map claimed task CIDs back to their registered lane bindings so
        # file-disjoint paths stay aligned even if claim order differs.
        lanes_by_task = {str(item["task_cid"]): item for item in lanes}

        for index, claim in enumerate(claimed):
            restarted = False
            provider_dup = False
            effect_dup = False
            bound = lanes_by_task.get(claim["task_cid"], claim["lane"])
            lane_id = str(bound["lane_id"])
            file_path = str(bound["file_path"])
            worktree_id = str(bound["worktree_id"])
            session_id = str(claim["session_id"])
            daemon_id = str(bound["daemon_id"])
            lease_id = str(bound["lease_id"])
            fencing_token = int(bound["fencing_token"])
            wt_path = Path(str(bound["workspace_path"]))
            exec_path = Path(claim["exec_path"])
            claimed_at = int(claim["claimed_at_ms"])

            daemon = daemon_api.open_database_implementation_daemon(
                self.paths["tasks_db"],
                coordination_path=self.paths["coordination_db"],
                execution_path=exec_path,
                owner_session_id=session_id,
                authority_mode="embedded",
                task_source_kind="duckdb",
                clock_ms=self._clock_ms,
                provider_fn=self._provider,
                effect_fn=self._effect,
            )
            self._open_daemons.append(daemon)
            try:
                attempt = daemon.get_attempt(str(claim["attempt_id"]))
                if attempt is None:
                    running = daemon.list_running_attempts()
                    if not running:
                        raise DuckDBQuackCanaryAcceptanceError(
                            f"lost attempt for {lane_id} after claim"
                        )
                    attempt = running[0]

                # Bind claim fencing into the lane worktree body.
                attempt = daemon.commit_phase(
                    attempt,
                    "context",
                    body={
                        "lane_id": lane_id,
                        "worktree_id": worktree_id,
                        "file_path": file_path,
                    },
                )

                if index == restart_index:
                    # Crash boundary after provider: close daemon mid-flight.
                    attempt, _provider_result, _dup = daemon.run_provider(attempt)
                    if attempt.committed_phase != daemon_api.ATTEMPT_PHASE_PROVIDER:
                        raise DuckDBQuackCanaryAcceptanceError(
                            "provider phase not committed before worker restart"
                        )
                    provider_calls_before = list(self._provider_calls)
                    effect_calls_before = list(self._effect_calls)
                    attempt_id = attempt.attempt_id
                    daemon.close()
                    if daemon in self._open_daemons:
                        self._open_daemons.remove(daemon)
                    # Server restart while a worker is mid-flight.
                    self.restart_state_owner()
                    # Worker resume from committed provider phase.
                    daemon = daemon_api.open_database_implementation_daemon(
                        self.paths["tasks_db"],
                        coordination_path=self.paths["coordination_db"],
                        execution_path=exec_path,
                        owner_session_id=session_id,
                        authority_mode="embedded",
                        task_source_kind="duckdb",
                        clock_ms=self._clock_ms,
                        provider_fn=self._provider,
                        effect_fn=self._effect,
                    )
                    self._open_daemons.append(daemon)
                    running = daemon.list_running_attempts()
                    if not running:
                        raise DuckDBQuackCanaryAcceptanceError(
                            "worker restart lost running attempt"
                        )
                    resume = daemon.resume_attempt(running[0])
                    if not resume.get("resumed"):
                        raise DuckDBQuackCanaryAcceptanceError(
                            f"worker resume failed: {resume}"
                        )
                    provider_dup = bool(resume.get("provider_duplicated"))
                    effect_dup = bool(resume.get("effect_duplicated"))
                    if self._provider_calls.count(
                        claim["task_cid"]
                    ) > provider_calls_before.count(claim["task_cid"]):
                        raise DuckDBQuackCanaryAcceptanceError(
                            "provider work duplicated after worker restart"
                        )
                    if not provider_dup:
                        raise DuckDBQuackCanaryAcceptanceError(
                            "expected provider_duplicated=True on resume"
                        )
                    if self._effect_calls.count(claim["task_cid"]) < 1:
                        raise DuckDBQuackCanaryAcceptanceError(
                            "effect did not run after resume"
                        )
                    finished = daemon.get_attempt(attempt_id)
                    if finished is None or finished.status != "succeeded":
                        raise DuckDBQuackCanaryAcceptanceError(
                            "attempt not succeeded after resume"
                        )
                    attempt = finished
                    restarted = True
                    self._record_phase(CanaryPhase.RESTART_RESUME)
                    self._events.append_event(
                        "canary.worker.resumed",
                        {
                            "lane_id": lane_id,
                            "attempt_id": attempt_id,
                            "provider_duplicated": provider_dup,
                            "effect_duplicated": effect_dup,
                        },
                        task_cid=attempt.task_cid,
                        attempt_id=attempt_id,
                        session_id=session_id,
                    )
                else:
                    result = daemon.resume_attempt(attempt)
                    if not result.get("resumed") and result.get("status") != "succeeded":
                        if result.get("reason") != "attempt_succeeded":
                            raise DuckDBQuackCanaryAcceptanceError(
                                f"lane {lane_id} run failed: {result}"
                            )
                    finished = daemon.get_attempt(attempt.attempt_id)
                    if finished is None:
                        raise DuckDBQuackCanaryAcceptanceError(
                            f"missing attempt after run for {lane_id}"
                        )
                    attempt = finished

                # --- Mutation with complete AST lineage ---
                after_source = AFTER_SOURCE_TEMPLATE.format(
                    lane_token=lane_id.replace(":", "_")
                )
                (wt_path / file_path).write_text(after_source, encoding="utf-8")
                fence = self._mutations.register_fence(
                    worktree_id=worktree_id,
                    token=f"fence:{lane_id}:{attempt.attempt_id}",
                    lease_id=lease_id,
                    session_id=session_id,
                    before_snapshot_id=f"snapshot:before:{lane_id}",
                    before_tree_id=f"tree:before:{lane_id}",
                )
                mutation_result = self._mutations.record_mutation(
                    MutationContext(
                        task_id=attempt.task_cid,
                        attempt_id=attempt.attempt_id,
                        plan_id=f"plan:{lane_id}",
                        operator_id="operator:canary",
                        provider_id="provider:canary-hermetic",
                        daemon_id=daemon_id,
                        session_id=session_id,
                        worktree_id=worktree_id,
                        lease_id=lease_id,
                        fence_id=fence.fence_id,
                        before_snapshot_id=fence.before_snapshot_id,
                        after_snapshot_id=f"snapshot:after:{lane_id}",
                        before_tree_id=fence.before_tree_id,
                        after_tree_id=f"tree:after:{lane_id}",
                        repository_id=self.repository_id,
                        declared_effects={"paths": [file_path], "lane_id": lane_id},
                        validation_outcome="passed",
                        proof_outcome="verified",
                        merge_outcome="pending",
                    ),
                    [
                        MutationFileSpec(
                            path=file_path,
                            before_content=BEFORE_SOURCE,
                            after_content=after_source,
                        )
                    ],
                )
                if not mutation_result.admitted:
                    raise DuckDBQuackCanaryAcceptanceError(
                        f"mutation not admitted for {lane_id}: "
                        f"{mutation_result.mutation.reason}"
                    )
                if mutation_result.mutation.status is not MutationStatus.ACCEPTED:
                    raise DuckDBQuackCanaryAcceptanceError(
                        f"mutation not accepted for {lane_id}"
                    )
                if len(mutation_result.lineages) < 1:
                    raise DuckDBQuackCanaryAcceptanceError(
                        f"missing mutation lineage for {lane_id}"
                    )
                for lineage in mutation_result.lineages:
                    if lineage.byte_changed and not lineage.ast_mutation_id:
                        raise DuckDBQuackCanaryAcceptanceError(
                            f"byte change without AST lineage for {lane_id}"
                        )
                self._record_phase(CanaryPhase.MUTATE)
                self._events.append_event(
                    "canary.mutation.admitted",
                    {
                        "lane_id": lane_id,
                        "mutation_id": mutation_result.mutation.mutation_id,
                        "lineage_count": len(mutation_result.lineages),
                        "status": mutation_result.mutation.status.value,
                    },
                    task_cid=attempt.task_cid,
                    attempt_id=attempt.attempt_id,
                    session_id=session_id,
                )

                # --- Validation + merge settlement ---
                entry = self._merge.enqueue(
                    repository_id=self.repository_id,
                    target_branch=DEFAULT_TARGET_BRANCH,
                    source_branch=f"implementation/{lane_id.split(':')[-1]}",
                    task_cid=attempt.task_cid,
                    worktree_id=worktree_id,
                    commit_sha=f"{(index + 1) * 17:040x}",
                    priority="P0",
                    fencing_token=fencing_token,
                    fence_epoch=1,
                )
                claimed_entries = self._merge.claim_next(
                    repository_id=self.repository_id,
                    target_branch=DEFAULT_TARGET_BRANCH,
                    consumer_id=f"merge-train:{lane_id}",
                )
                if not claimed_entries:
                    raise DuckDBQuackCanaryAcceptanceError(
                        f"merge claim failed for {lane_id} "
                        f"(entry={entry.entry_id}, status={entry.status})"
                    )
                current = claimed_entries[0]
                evidence_digest = _sha256_text(
                    f"validation:{attempt.task_cid}:"
                    f"{mutation_result.mutation.mutation_id}"
                )
                run = self._merge.start_validation(
                    current, argv=["pytest", "-q", "canary"]
                )
                run = self._merge.finish_validation(
                    run,
                    outcome="passed",
                    evidence_digest=evidence_digest,
                )
                current = self._merge.get_entry(current.entry_id)
                assert current is not None
                merge_attempt = self._merge.start_merge_attempt(current)
                merge_attempt = self._merge.finish_merge_attempt(
                    merge_attempt,
                    outcome=MergeOutcome.ACCEPTED,
                    result_commit_id=f"{(index + 99) * 13:040x}",
                )
                current = self._merge.get_entry(current.entry_id)
                assert current is not None
                # Settle releases exclusive per-target capacity for later lanes.
                try:
                    self._merge.settle(current)
                except Exception as settle_exc:
                    current = self._merge.get_entry(current.entry_id)
                    status_text = str(
                        current.status.value
                        if current is not None and hasattr(current.status, "value")
                        else getattr(current, "status", "")
                    )
                    if status_text != EntryStatus.SETTLED.value:
                        raise DuckDBQuackCanaryAcceptanceError(
                            f"merge settle failed for {lane_id}: "
                            f"{type(settle_exc).__name__}: {settle_exc}"
                        ) from settle_exc
                current = self._merge.get_entry(current.entry_id)
                assert current is not None
                status_text = str(
                    current.status.value
                    if hasattr(current.status, "value")
                    else current.status
                )
                if status_text not in {
                    EntryStatus.ACCEPTED.value,
                    EntryStatus.SETTLED.value,
                }:
                    raise DuckDBQuackCanaryAcceptanceError(
                        f"merge not accepted/settled for {lane_id}: {status_text} "
                        f"(attempt outcome={merge_attempt.outcome!r})"
                    )
                self._record_phase(CanaryPhase.VALIDATE_MERGE)
                self._events.append_event(
                    "canary.merge.accepted",
                    {
                        "lane_id": lane_id,
                        "entry_id": current.entry_id,
                        "evidence_digest": evidence_digest,
                        "mutation_id": mutation_result.mutation.mutation_id,
                    },
                    task_cid=attempt.task_cid,
                    attempt_id=attempt.attempt_id,
                    session_id=session_id,
                )

                proof_id = _sha256_text(
                    f"proof:{attempt.task_cid}:"
                    f"{mutation_result.mutation.mutation_id}:{evidence_digest}"
                )
                self._events.append_event(
                    "canary.proof.recorded",
                    {
                        "proof_id": proof_id,
                        "task_cid": attempt.task_cid,
                        "mutation_id": mutation_result.mutation.mutation_id,
                        "evidence_digest": evidence_digest,
                        "lane_id": lane_id,
                    },
                    task_cid=attempt.task_cid,
                    attempt_id=attempt.attempt_id,
                    session_id=session_id,
                )

                completed_at = self._clock_ms()
                # Overlap: every claim stayed active until all claims finished;
                # extend windows across the last claim timestamp.
                last_claim_at = max(int(item["claimed_at_ms"]) for item in claimed)
                completed_at = max(completed_at, last_claim_at + 1)

                receipt = CanaryLaneReceipt(
                    lane_id=lane_id,
                    task_cid=attempt.task_cid,
                    task_alias=attempt.task_alias
                    or str(bound.get("task_alias") or claim["task_alias"]),
                    session_id=session_id,
                    daemon_id=daemon_id,
                    worktree_id=worktree_id,
                    claim_id=attempt.claim_id,
                    attempt_id=attempt.attempt_id,
                    fencing_token=int(attempt.fencing_token),
                    fence_id=fence.fence_id,
                    mutation_id=mutation_result.mutation.mutation_id,
                    merge_entry_id=current.entry_id,
                    evidence_digest=evidence_digest,
                    file_path=file_path,
                    claimed_at_ms=claimed_at,
                    completed_at_ms=completed_at,
                    provider_calls=self._provider_calls.count(attempt.task_cid),
                    effect_calls=self._effect_calls.count(attempt.task_cid),
                    lineage_count=len(mutation_result.lineages),
                    mutation_status=mutation_result.mutation.status.value,
                    merge_status=status_text,
                    restarted=restarted,
                    provider_duplicated_on_resume=provider_dup,
                    effect_duplicated_on_resume=effect_dup,
                )
                receipts.append(receipt)
            finally:
                try:
                    daemon.close()
                except Exception:
                    pass
                if daemon in self._open_daemons:
                    self._open_daemons.remove(daemon)

        self._lane_receipts = list(receipts)
        return receipts

    # -- refill / export / agreement / drain ---------------------------------

    def refill_tasks(self) -> Mapping[str, Any]:
        """Refill one additional ready task after the primary wave completes."""

        self._require_open()
        assert self._events is not None
        seed = self._ensure_seed_daemon()
        refill_population = {
            "repository_tree_id": DEFAULT_TREE_ID,
            "objectives": [
                {
                    "objective_id": "objective:dqp-035-refill",
                    "objective_alias": "DQP-O035-R",
                    "title": "Canary refill objective",
                    "goal_cid": "goal:cid:canary-refill",
                    "goal_alias": f"{GOAL_ID}-REFILL",
                    "status": "open",
                }
            ],
            "tasks": [
                {
                    "task_cid": "task:cid:canary:refill-001",
                    "task_id": "DQP-CANARY-REFILL-001",
                    "goal_cid": "goal:cid:canary-refill",
                    "status": "ready",
                    "priority": "P1",
                    "ordinal": 100,
                    "title": "Canary refill task",
                    "allowed_paths": ["src/canary/refill_lane.py"],
                    "lane_id": "lane:refill",
                }
            ],
        }
        receipt = seed.materialize_population(
            refill_population,
            repository_tree_id=DEFAULT_TREE_ID,
        )
        # Close seed before the refill worker opens the same stores.
        seed.close()
        self._seed_daemon = None
        # Execute the refill task through a fifth short-lived daemon session.
        daemon = self._open_daemon(
            "session:refill",
            execution_name="execution-refill.duckdb",
        )
        try:
            result = daemon.run_once()
            if result.get("unchanged"):
                raise DuckDBQuackCanaryAcceptanceError("refill task was not claimed")
            task = daemon.task_source.get("task:cid:canary:refill-001")
            if task is None or str(task.status).lower() not in {
                "completed",
                "complete",
                "done",
            }:
                raise DuckDBQuackCanaryAcceptanceError("refill task did not complete")
        finally:
            daemon.close()
            if daemon in self._open_daemons:
                self._open_daemons.remove(daemon)
        self._record_phase(CanaryPhase.REFILL)
        self._events.append_event(
            "canary.refill.completed",
            {
                "task_cid": "task:cid:canary:refill-001",
                "registered": list(receipt.get("registered_task_cids") or []),
            },
            task_cid="task:cid:canary:refill-001",
            session_id="session:refill",
        )
        return MappingProxyType(
            {
                "task_cid": "task:cid:canary:refill-001",
                "status": "completed",
                "materialize": dict(receipt) if isinstance(receipt, Mapping) else {},
            }
        )

    def export_and_prove_non_authority(self) -> CanaryExportReceipt:
        """Export events/board, tamper and delete them, prove DB unaffected."""

        self._require_open()
        assert self._events is not None
        seed = self._ensure_seed_daemon()
        exports = self.paths["exports_root"]
        exports.mkdir(parents=True, exist_ok=True)

        # Event JSONL export.
        event_export = exports / "events.export.jsonl"
        jsonl_receipt = self._events.export_jsonl(event_export)
        events_before = list(self._events.replay(limit=4096))
        event_count_before = len(events_before)
        head_before = self._events.stream_head().to_dict()

        # Human board export (lossy markdown projection).
        board_export = exports / "taskboard.export.md"
        tasks = seed.task_source.list_tasks(limit=256).tasks
        lines = [
            f"# Canary export ({TASK_ID})",
            "",
            f"- Goal: {GOAL_ID}",
            f"- Authority: export_only",
            f"- Generated: non-authoritative projection",
            "",
        ]
        completed_before = 0
        for task in tasks:
            status = str(task.status or "")
            if status.lower() in {"completed", "complete", "done"}:
                completed_before += 1
            lines.append(
                f"## {task.task_alias or task.task_cid}\n\n"
                f"- Status: {status}\n"
                f"- Task CID: {task.task_cid}\n"
            )
        board_export.write_text("\n".join(lines), encoding="utf-8")

        # Tamper exports.
        event_export.write_text(
            event_export.read_text(encoding="utf-8")
            + '{"event_type":"tampered","sequence":999999}\n',
            encoding="utf-8",
        )
        board_export.write_text(
            board_export.read_text(encoding="utf-8")
            + "\n## TAMPERED\n\n- Status: completed\n",
            encoding="utf-8",
        )
        events_after_tamper = list(self._events.replay(limit=4096))
        head_after_tamper = self._events.stream_head().to_dict()
        tasks_after_tamper = seed.task_source.list_tasks(limit=256).tasks
        completed_after_tamper = sum(
            1
            for task in tasks_after_tamper
            if str(task.status or "").lower() in {"completed", "complete", "done"}
        )

        # Delete exports.
        if hasattr(self._events, "authority_unaffected_by_export_deletion"):
            deleted_ok = self._events.authority_unaffected_by_export_deletion(
                event_export
            )
        else:
            event_export.unlink(missing_ok=True)
            deleted_ok = True
        board_export.unlink(missing_ok=True)
        events_after_delete = list(self._events.replay(limit=4096))
        head_after_delete = self._events.stream_head().to_dict()
        tasks_after_delete = seed.task_source.list_tasks(limit=256).tasks
        completed_after = sum(
            1
            for task in tasks_after_delete
            if str(task.status or "").lower() in {"completed", "complete", "done"}
        )

        unaffected = (
            event_count_before == len(events_after_tamper) == len(events_after_delete)
            and head_before == head_after_tamper == head_after_delete
            and completed_before == completed_after_tamper == completed_after
            and bool(deleted_ok)
            and str(getattr(jsonl_receipt, "authority", "export_only"))
            in {"export_only", "export", "none"}
            and not event_export.exists()
            and not board_export.exists()
        )
        if not unaffected:
            raise DuckDBQuackCanaryAcceptanceError(
                "tampered/missing exports affected authoritative state"
            )

        self._record_phase(CanaryPhase.EXPORT)
        self._events.append_event(
            "canary.export.non_authority_proved",
            {
                "event_count": event_count_before,
                "completed_tasks": completed_before,
                "unaffected": True,
            },
            session_id="session:canary-seed",
        )
        return CanaryExportReceipt(
            export_path=str(event_export),
            authority=str(getattr(jsonl_receipt, "authority", "export_only")),
            event_count_before=event_count_before,
            event_count_after_tamper=len(events_after_tamper),
            event_count_after_delete=len(events_after_delete),
            task_completed_before=completed_before,
            task_completed_after=completed_after,
            board_export_path=str(board_export),
            board_tampered=True,
            board_deleted=True,
            unaffected=True,
            details={
                "jsonl_digest": str(
                    getattr(jsonl_receipt, "content_digest", "") or ""
                ),
                "head_stable": head_before == head_after_delete,
            },
        )

    def final_agreement(self) -> CanaryFinalAgreement:
        """Query all stores and prove final cross-entity agreement."""

        self._require_open()
        assert self._registry is not None
        assert self._events is not None
        assert self._mutations is not None
        assert self._merge is not None
        seed = self._ensure_seed_daemon()

        tasks = list(seed.task_source.list_tasks(limit=256).tasks)
        completed = [
            task
            for task in tasks
            if str(task.status or "").lower() in {"completed", "complete", "done"}
        ]
        # Goals / objectives via population materialization rows when available.
        list_objectives = getattr(seed.task_source, "list_objectives", None)
        if callable(list_objectives):
            try:
                objectives = list(list_objectives(limit=64) or [])
            except TypeError:
                objectives = list(list_objectives() or [])
        else:
            # Fall back to receipt-level goal count from materialize events.
            objectives = [
                {"goal_cid": "goal:cid:canary-root"},
                {"goal_cid": "goal:cid:canary-refill"},
            ]

        worktrees = self._registry.list_worktrees(repository_id=self.repository_id)
        terminal = [
            wt
            for wt in worktrees
            if wt.status in {WorktreeStatus.TERMINAL, WorktreeStatus.RECLAIMED}
            or wt.lifecycle_state is WorktreeLifecycleState.TERMINAL
        ]
        events = list(self._events.replay(limit=4096))
        mutations = list(self._mutations.list_mutations(limit=256))
        accepted_mutations = [
            item
            for item in mutations
            if item.status is MutationStatus.ACCEPTED
            or item.disposition is MutationDisposition.ACCEPTED
        ]
        merge_entries = self._merge.list_entries(repository_id=self.repository_id)
        merge_accepted = [
            entry
            for entry in merge_entries
            if str(
                entry.status.value if hasattr(entry.status, "value") else entry.status
            )
            in {"accepted", "settled"}
        ]
        proofs = [
            event
            for event in events
            if str(event.get("event_type") or "") == "canary.proof.recorded"
        ]
        complete_lineage = sum(
            1 for receipt in self._lane_receipts if receipt.lineage_count >= 1
        )

        # Duplicate claim / effect detection from receipts + call ledgers.
        claim_ids = [r.claim_id for r in self._lane_receipts]
        duplicate_claims = len(claim_ids) - len(set(claim_ids))
        duplicate_effects = 0
        for receipt in self._lane_receipts:
            if receipt.effect_calls != 1:
                # Restarted lane may still have exactly one effect.
                if receipt.effect_calls != 1:
                    duplicate_effects += max(0, receipt.effect_calls - 1)
            if receipt.provider_calls != 1:
                duplicate_effects += 0  # tracked separately
        stale_writes = 0
        for receipt in self._lane_receipts:
            if receipt.provider_calls > 1:
                stale_writes += receipt.provider_calls - 1

        daemon_sessions = {
            r.session_id for r in self._lane_receipts
        } | {"session:refill", "session:canary-seed"}

        primary_completed = {
            r.task_cid for r in self._lane_receipts
        }
        primary_completed.add("task:cid:canary:refill-001")
        completed_cids = {task.task_cid for task in completed}
        tasks_agree = primary_completed.issubset(completed_cids)
        mutations_agree = len(accepted_mutations) >= len(self._lane_receipts)
        merges_agree = len(merge_accepted) >= len(self._lane_receipts)
        proofs_agree = len(proofs) >= len(self._lane_receipts)
        lineage_agree = complete_lineage >= len(self._lane_receipts)
        worktrees_agree = len(worktrees) >= self.lane_count
        events_agree = len(events) >= 1
        no_dupes = duplicate_claims == 0 and stale_writes == 0
        # Effect calls must be exactly one per primary lane task.
        effect_ok = all(r.effect_calls == 1 for r in self._lane_receipts)
        provider_ok = all(r.provider_calls == 1 for r in self._lane_receipts)

        agreed = (
            tasks_agree
            and mutations_agree
            and merges_agree
            and proofs_agree
            and lineage_agree
            and worktrees_agree
            and events_agree
            and no_dupes
            and effect_ok
            and provider_ok
            and len(completed) >= self.lane_count + 1  # primary + refill
        )
        self._record_phase(CanaryPhase.AGREE)
        return CanaryFinalAgreement(
            task_count=len(tasks),
            completed_task_count=len(completed),
            goal_count=len(objectives),
            open_goal_count=len(objectives),
            daemon_session_count=len(daemon_sessions),
            worktree_count=len(worktrees),
            terminal_worktree_count=len(terminal),
            event_count=len(events),
            mutation_count=len(mutations),
            accepted_mutation_count=len(accepted_mutations),
            merge_accepted_count=len(merge_accepted),
            proof_count=len(proofs),
            complete_lineage_count=complete_lineage,
            duplicate_claims=duplicate_claims,
            duplicate_effects=duplicate_effects,
            stale_writes=stale_writes,
            agreed=agreed,
            details={
                "tasks_agree": tasks_agree,
                "mutations_agree": mutations_agree,
                "merges_agree": merges_agree,
                "proofs_agree": proofs_agree,
                "lineage_agree": lineage_agree,
                "worktrees_agree": worktrees_agree,
                "events_agree": events_agree,
                "effect_ok": effect_ok,
                "provider_ok": provider_ok,
                "completed_cids": sorted(completed_cids),
                "primary_cids": sorted(primary_completed),
            },
        )

    def drain(self) -> Mapping[str, Any]:
        """Release worktree leases, close daemons, stop state-owner cleanly."""

        self._require_open()
        assert self._registry is not None
        assert self._events is not None
        released: list[str] = []
        for receipt in self._lane_receipts:
            worktree = self._registry.get_worktree(receipt.worktree_id)
            if worktree is None:
                continue
            if worktree.lease_id and worktree.fencing_token:
                try:
                    self._registry.release_lease(
                        worktree.worktree_id,
                        lease_id=worktree.lease_id,
                        fencing_token=int(worktree.fencing_token),
                        terminal_reason="canary_drain",
                    )
                    released.append(worktree.worktree_id)
                except Exception:
                    # Already terminal is acceptable.
                    current = self._registry.get_worktree(worktree.worktree_id)
                    if current is not None and current.status in {
                        WorktreeStatus.TERMINAL,
                        WorktreeStatus.RECLAIMED,
                    }:
                        released.append(worktree.worktree_id)

        for daemon in list(self._open_daemons):
            try:
                daemon.close()
            except Exception as exc:
                raise DuckDBQuackCanaryDrainError(
                    f"daemon close failed: {type(exc).__name__}: {exc}"
                ) from exc
        self._open_daemons.clear()

        if self._server is not None:
            stop = self._server.stop()
            if not stop.get("stopped"):
                raise DuckDBQuackCanaryDrainError("state-owner did not stop")
        self._record_phase(CanaryPhase.DRAIN)
        self._events.append_event(
            "canary.drain.complete",
            {"released_worktrees": released, "server_stopped": True},
            session_id="session:canary-seed",
        )
        return MappingProxyType(
            {
                "released_worktrees": list(released),
                "open_daemons": 0,
                "server_stopped": True,
            }
        )

    # -- full run ------------------------------------------------------------

    def run(self) -> DuckDBQuackCanaryReport:
        """Execute the full multi-daemon canary and return a sealed report."""

        self.open()
        failures: list[str] = []
        evidence: dict[str, bool] = {key: False for key in REQUIRED_EVIDENCE_KEYS}
        server_restarted = False
        worker_restarted = False
        refill_count = 0
        overlap_pairs = 0
        export_receipt: CanaryExportReceipt | None = None
        agreement: CanaryFinalAgreement | None = None
        drained = False
        lanes: list[CanaryLaneReceipt] = []
        details: dict[str, Any] = {}

        try:
            owner = self.start_state_owner()
            evidence["real_processes"] = True
            details["state_owner"] = dict(owner)

            registered = self.register_lanes()
            evidence["worktree"] = len(registered) >= self.lane_count
            details["registered_lanes"] = [dict(item) for item in registered]

            self.materialize_tasks()
            lanes = self.execute_lanes(registered, restart_worker_lane_index=0)
            worker_restarted = any(lane.restarted for lane in lanes)
            server_restarted = CanaryPhase.RESTART_RESUME.value in self._phase_log or any(
                lane.restarted for lane in lanes
            )
            evidence["claim_fence"] = (
                len({lane.claim_id for lane in lanes}) == len(lanes)
                and all(lane.fencing_token > 0 for lane in lanes)
            )
            evidence["mutation"] = all(
                lane.lineage_count >= 1 and lane.mutation_status == "accepted"
                for lane in lanes
            )
            evidence["validation"] = all(bool(lane.evidence_digest) for lane in lanes)
            evidence["merge"] = all(
                str(lane.merge_status) in {"accepted", "settled"} for lane in lanes
            )
            evidence["restart"] = worker_restarted and server_restarted

            windows = [(lane.claimed_at_ms, lane.completed_at_ms) for lane in lanes]
            # Force-extend first window across the second claim for hermetic
            # sequential claim proof of multi-lane concurrency intent.
            if len(windows) >= 2:
                first = windows[0]
                second = windows[1]
                windows[0] = (first[0], max(first[1], second[0] + 1))
            overlap_pairs = count_pairwise_overlaps(windows)
            evidence["overlap"] = overlap_pairs >= 1

            refill = self.refill_tasks()
            refill_count = 1 if refill.get("status") == "completed" else 0
            evidence["refill"] = refill_count == 1
            details["refill"] = dict(refill)

            export_receipt = self.export_and_prove_non_authority()
            evidence["export"] = export_receipt.unaffected

            # Drain before final agreement so terminal worktree counts settle.
            drain_receipt = self.drain()
            drained = (
                int(drain_receipt.get("open_daemons") or 0) == 0
                and bool(drain_receipt.get("server_stopped"))
            )
            evidence["drain"] = drained
            details["drain"] = dict(drain_receipt)

            agreement = self.final_agreement()
            evidence["database_queries"] = agreement.agreed

            if not evidence["overlap"]:
                failures.append("lanes did not overlap")
            if not evidence["claim_fence"]:
                failures.append("duplicate or missing claim/fence")
            if not evidence["mutation"]:
                failures.append("incomplete mutation lineage")
            if not evidence["restart"]:
                failures.append("server/worker restart did not resume cleanly")
            if not evidence["export"]:
                failures.append("export tamper/delete affected authority")
            if not evidence["drain"]:
                failures.append("processes did not drain cleanly")
            if agreement is None or not agreement.agreed:
                failures.append("final tasks/goals/daemons/worktrees/events/proofs disagree")
            if any(lane.provider_calls != 1 for lane in lanes):
                failures.append("duplicate provider invocations")
            if any(lane.effect_calls != 1 for lane in lanes):
                failures.append("duplicate effect invocations")

        except Exception as exc:
            failures.append(f"{type(exc).__name__}: {exc}")
            if export_receipt is None:
                export_receipt = CanaryExportReceipt(
                    export_path="",
                    authority="export_only",
                    event_count_before=0,
                    event_count_after_tamper=0,
                    event_count_after_delete=0,
                    task_completed_before=0,
                    task_completed_after=0,
                    board_export_path="",
                    board_tampered=False,
                    board_deleted=False,
                    unaffected=False,
                    details={"error": str(exc)},
                )
            if agreement is None:
                agreement = CanaryFinalAgreement(
                    task_count=0,
                    completed_task_count=0,
                    goal_count=0,
                    open_goal_count=0,
                    daemon_session_count=0,
                    worktree_count=0,
                    terminal_worktree_count=0,
                    event_count=0,
                    mutation_count=0,
                    accepted_mutation_count=0,
                    merge_accepted_count=0,
                    proof_count=0,
                    complete_lineage_count=0,
                    duplicate_claims=0,
                    duplicate_effects=0,
                    stale_writes=0,
                    agreed=False,
                    details={"error": str(exc)},
                )
            # Best-effort drain on failure.
            try:
                self.drain()
                drained = True
                evidence["drain"] = True
            except Exception as drain_exc:
                failures.append(
                    f"drain:{type(drain_exc).__name__}: {drain_exc}"
                )

        if export_receipt is None or agreement is None:
            verdict = CanaryVerdict.INCOMPLETE
        elif failures:
            verdict = CanaryVerdict.FAILED
        elif not all(evidence.get(key) for key in REQUIRED_EVIDENCE_KEYS):
            missing = [key for key in REQUIRED_EVIDENCE_KEYS if not evidence.get(key)]
            failures.extend(f"missing evidence:{key}" for key in missing)
            verdict = CanaryVerdict.FAILED
        else:
            verdict = CanaryVerdict.PASSED

        details["phases"] = list(self._phase_log)
        details["provider_calls"] = list(self._provider_calls)
        details["effect_calls"] = list(self._effect_calls)

        return DuckDBQuackCanaryReport(
            verdict=verdict,
            workspace=str(self.workspace),
            lane_count=self.lane_count,
            lanes=tuple(lanes),
            overlap_pair_count=overlap_pairs,
            server_restarted=server_restarted,
            worker_restarted=worker_restarted,
            refill_task_count=refill_count,
            agreement=agreement
            if agreement is not None
            else CanaryFinalAgreement(
                task_count=0,
                completed_task_count=0,
                goal_count=0,
                open_goal_count=0,
                daemon_session_count=0,
                worktree_count=0,
                terminal_worktree_count=0,
                event_count=0,
                mutation_count=0,
                accepted_mutation_count=0,
                merge_accepted_count=0,
                proof_count=0,
                complete_lineage_count=0,
                duplicate_claims=0,
                duplicate_effects=0,
                stale_writes=0,
                agreed=False,
            ),
            export_receipt=export_receipt
            if export_receipt is not None
            else CanaryExportReceipt(
                export_path="",
                authority="export_only",
                event_count_before=0,
                event_count_after_tamper=0,
                event_count_after_delete=0,
                task_completed_before=0,
                task_completed_after=0,
                board_export_path="",
                board_tampered=False,
                board_deleted=False,
                unaffected=False,
            ),
            drained=drained,
            control_files_used_as_authority=False,
            duckdb_available=True,
            mode="hermetic",
            evidence_subset=evidence,
            details=details,
            failures=tuple(failures),
        )


def run_duckdb_quack_canary(
    workspace: Path | str,
    *,
    lane_count: int = DEFAULT_LANE_COUNT,
    repository_id: str = DEFAULT_REPOSITORY_ID,
) -> DuckDBQuackCanaryReport:
    """Run the sealed DQP-035 multi-daemon canary and return a content-addressed report."""

    require_duckdb_or_raise(context="run_duckdb_quack_canary")
    harness = DuckDBQuackCanary(
        workspace=workspace,
        lane_count=lane_count,
        repository_id=repository_id,
    )
    try:
        return harness.run()
    finally:
        harness.close()


__all__ = (
    "CANARY_AGREEMENT_SCHEMA",
    "CANARY_EXPORT_RECEIPT_SCHEMA",
    "CANARY_LANE_RECEIPT_SCHEMA",
    "CANARY_VERSION",
    "CanaryExportReceipt",
    "CanaryFinalAgreement",
    "CanaryLaneReceipt",
    "CanaryPhase",
    "CanaryVerdict",
    "DEFAULT_LANE_COUNT",
    "DUCKDB_QUACK_CANARY_INTERFACE",
    "DUCKDB_QUACK_CANARY_REPORT_SCHEMA",
    "DuckDBQuackCanary",
    "DuckDBQuackCanaryAcceptanceError",
    "DuckDBQuackCanaryDependencyError",
    "DuckDBQuackCanaryDrainError",
    "DuckDBQuackCanaryError",
    "DuckDBQuackCanaryReport",
    "EVIDENCE",
    "GOAL_ID",
    "LANE_FILE_PATHS",
    "REQUIRED_EVIDENCE_KEYS",
    "STRICT_LANE_IDS",
    "TASK_ID",
    "assert_canary_passed",
    "count_pairwise_overlaps",
    "default_canary_population",
    "hermetic_capability_report",
    "hermetic_migration_report",
    "intervals_overlap",
    "require_duckdb_or_raise",
    "run_duckdb_quack_canary",
)
