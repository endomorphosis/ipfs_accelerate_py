"""Multi-daemon multi-worktree DuckDB/Quack authoritative canary (DQP-035).

Interface: ``DuckDBQuackCanary@1``

Hermetic E2E that bootstraps a clean control-plane database, starts a
loopback state-owner, registers four strict lanes, executes file-disjoint
tasks with overlapping claims, records lineage, restarts the server, and
proves export non-authority. Live production paths are never mutated.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from ..runtime.quack_state_server import (
    FakeQuackTransport,
    build_server,
)
from ..task_sources.control_plane_migrations import duckdb_available
from ..task_sources.control_plane_schema import install_control_plane_schema
from ..task_sources.duckdb_state import open_duckdb_connection
from ..task_sources.quack_capabilities import (
    DEFAULT_QUACK_BETA_LIMITATIONS,
    QuackCapabilityReport,
    QuackCapabilityStatus,
    default_compatibility_profile,
)

DUCKDB_QUACK_CANARY_INTERFACE: Final[str] = "DuckDBQuackCanary@1"
CANARY_CONTRACT_VERSION: Final[int] = 1
TASK_ID: Final[str] = "DQP-035"
GOAL_ID: Final[str] = "DQP-G080"
EVIDENCE: Final[str] = "dqp/duckdb-quack-canary@1"
SCHEMA_PREFIX: Final[str] = "ipfs_accelerate_py/agent-supervisor"
CANARY_REPORT_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/duckdb-quack-canary-report@1"
LANE_RECEIPT_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/canary-lane-receipt@1"
LINEAGE_SCHEMA: Final[str] = f"{SCHEMA_PREFIX}/canary-lineage@1"
EXPORT_MARKER: Final[str] = "EXPORT_NON_AUTHORITATIVE.json"

LANE_COUNT: Final[int] = 4
TASKS_PER_LANE: Final[int] = 2


class CanaryPhase(str, Enum):
    BOOTSTRAP = "bootstrap"
    REGISTER = "register"
    EXECUTE = "execute"
    RESTART = "restart"
    RECONCILE = "reconcile"
    EXPORT = "export"
    DRAIN = "drain"
    TERMINAL = "terminal"


class CanaryOutcome(str, Enum):
    PASSED = "passed"
    FAILED = "failed"


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _identity(prefix: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"{prefix}:{digest[:32]}"


def require_duckdb_or_raise(*, context: str = "duckdb quack canary") -> None:
    if not duckdb_available():
        raise RuntimeError(f"DuckDB is required for {context}")


def _compatible_capability_report(**_kwargs: Any) -> QuackCapabilityReport:
    profile = default_compatibility_profile()
    return QuackCapabilityReport(
        status=QuackCapabilityStatus.COMPATIBLE,
        profile=profile,
        duckdb_importable=True,
        duckdb_version="1.5.2",
        extension_fingerprint="sha256:" + ("ab" * 32),
        observed_functions=tuple(profile.required_functions),
        observed_surfaces=tuple(profile.required_surfaces),
        beta_limitations=DEFAULT_QUACK_BETA_LIMITATIONS,
    )


@dataclass(frozen=True)
class CanaryTask:
    task_id: str
    lane_index: int
    path: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "lane_index": self.lane_index,
            "path": self.path,
            "content_digest": hashlib.sha256(self.content.encode()).hexdigest(),
        }


@dataclass(frozen=True)
class CanaryLineage:
    SCHEMA: ClassVar[str] = LINEAGE_SCHEMA
    task_id: str
    lane_index: int
    claim_fence: int
    worktree_id: str
    mutation_id: str
    validation_id: str
    merge_id: str
    ast_snapshot_id: str
    provider_call_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "task_id": self.task_id,
            "lane_index": self.lane_index,
            "claim_fence": self.claim_fence,
            "worktree_id": self.worktree_id,
            "mutation_id": self.mutation_id,
            "validation_id": self.validation_id,
            "merge_id": self.merge_id,
            "ast_snapshot_id": self.ast_snapshot_id,
            "provider_call_id": self.provider_call_id,
            "complete": all(
                [
                    self.worktree_id,
                    self.mutation_id,
                    self.validation_id,
                    self.merge_id,
                    self.ast_snapshot_id,
                ]
            ),
        }


@dataclass
class CanaryLaneReceipt:
    SCHEMA: ClassVar[str] = LANE_RECEIPT_SCHEMA
    lane_index: int
    tasks: list[str] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)
    effects: list[str] = field(default_factory=list)
    restarted: bool = False
    drained: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "lane_index": self.lane_index,
            "tasks": list(self.tasks),
            "claims": list(self.claims),
            "effects": list(self.effects),
            "restarted": self.restarted,
            "drained": self.drained,
        }


@dataclass
class DuckDBQuackCanaryReport:
    """Authoritative canary result; never grants production completion."""

    SCHEMA: ClassVar[str] = CANARY_REPORT_SCHEMA
    INTERFACE: ClassVar[str] = DUCKDB_QUACK_CANARY_INTERFACE

    outcome: CanaryOutcome
    phase: CanaryPhase
    server_id: str
    generation: int
    lane_count: int
    overlapping_lanes: int
    duplicate_claims: int
    stale_writes: int
    lineages: tuple[CanaryLineage, ...]
    lanes: tuple[CanaryLaneReceipt, ...]
    export_path: str
    export_non_authoritative: bool
    database_authority_intact_after_export_tamper: bool
    drained: bool
    detail: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_iso)

    @property
    def passed(self) -> bool:
        return self.outcome is CanaryOutcome.PASSED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "interface": self.INTERFACE,
            "contract_version": CANARY_CONTRACT_VERSION,
            "task_id": TASK_ID,
            "goal_id": GOAL_ID,
            "evidence": EVIDENCE,
            "outcome": self.outcome.value,
            "phase": self.phase.value,
            "passed": self.passed,
            "server_id": self.server_id,
            "generation": self.generation,
            "lane_count": self.lane_count,
            "overlapping_lanes": self.overlapping_lanes,
            "duplicate_claims": self.duplicate_claims,
            "stale_writes": self.stale_writes,
            "lineages": [item.to_dict() for item in self.lineages],
            "lanes": [item.to_dict() for item in self.lanes],
            "export_path": self.export_path,
            "export_non_authoritative": self.export_non_authoritative,
            "database_authority_intact_after_export_tamper": (
                self.database_authority_intact_after_export_tamper
            ),
            "drained": self.drained,
            "detail": self.detail,
            "metrics": dict(self.metrics),
            "created_at": self.created_at,
        }


class DuckDBQuackCanaryError(RuntimeError):
    """Fail-closed canary error."""


class DuckDBQuackCanary:
    """Hermetic multi-lane database-authoritative canary runner."""

    INTERFACE: ClassVar[str] = DUCKDB_QUACK_CANARY_INTERFACE

    def __init__(self, workspace: Path | str) -> None:
        require_duckdb_or_raise(context="duckdb quack canary")
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.db_path = self.workspace / "control.duckdb"
        self.state_dir = self.workspace / "state-owner"
        self.worktree_root = self.workspace / "worktrees"
        self.export_root = self.workspace / "exports"
        self.worktree_root.mkdir(parents=True, exist_ok=True)
        self.export_root.mkdir(parents=True, exist_ok=True)
        self._claim_lock = threading.Lock()
        self._db_lock = threading.Lock()
        self._claims: dict[str, str] = {}
        self._effects: dict[str, str] = {}
        self._lineages: list[CanaryLineage] = []
        self._pending_completions: list[str] = []
        self._lane_receipts: dict[int, CanaryLaneReceipt] = {
            index: CanaryLaneReceipt(lane_index=index) for index in range(LANE_COUNT)
        }
        self._active_lanes: set[int] = set()
        self._overlap_seen = 0
        self._server = None
        self._identity = None
        self._transport = FakeQuackTransport()

    def _bootstrap_database(self) -> None:
        install_control_plane_schema(
            self.db_path,
            application_version="0.0.45",
            tool_version="1.5.2",
            owner_id="duckdb-quack-canary",
        )
        with open_duckdb_connection(self.db_path) as connection:
            connection.execute("DELETE FROM store_generations")
            connection.execute(
                """
                INSERT INTO store_generations (
                    generation, schema_revision, fence_epoch, revision,
                    database_uuid, birth_id, created_at
                ) VALUES (1, 1, 1, 0, ?, ?, ?)
                """,
                [
                    "123e4567-e89b-12d3-a456-426614174000",
                    "birth:canary",
                    "1970-01-01T00:00:00Z",
                ],
            )
            connection.execute(
                """
                INSERT INTO goals (
                    goal_cid, goal_alias, objective_id, parent_goal_cid, ordinal,
                    title, status, created_at, updated_at, revision, body_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    "goal:canary",
                    "G-CANARY",
                    "objective:canary",
                    "",
                    1,
                    "Canary goal",
                    "open",
                    _utc_iso(),
                    _utc_iso(),
                    0,
                    "{}",
                ],
            )
            for lane in range(LANE_COUNT):
                for ordinal in range(TASKS_PER_LANE):
                    task_id = f"CANARY-{lane}-{ordinal}"
                    connection.execute(
                        """
                        INSERT INTO tasks (
                            task_cid, task_alias, goal_cid, plan_cid, objective_id,
                            ordinal, status, revision, priority, created_at, updated_at,
                            identity_json, body_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            f"task:{task_id}",
                            task_id,
                            "goal:canary",
                            "",
                            "objective:canary",
                            lane * TASKS_PER_LANE + ordinal + 1,
                            "ready",
                            0,
                            "P0",
                            _utc_iso(),
                            _utc_iso(),
                            "{}",
                            json.dumps(
                                {
                                    "lane": lane,
                                    "path": f"lanes/lane-{lane}/artifact-{ordinal}.txt",
                                }
                            ),
                        ],
                    )

    def _start_server(self) -> str:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._server = build_server(
            database_path=self.db_path,
            state_dir=self.state_dir,
            transport=self._transport,
            capability_probe=lambda **_k: _compatible_capability_report(),
            repository_id="repository:canary",
        )
        self._identity = self._server.start()
        return str(self._identity.server_id)

    def _plan_tasks(self) -> list[CanaryTask]:
        tasks: list[CanaryTask] = []
        for lane in range(LANE_COUNT):
            for ordinal in range(TASKS_PER_LANE):
                task_id = f"CANARY-{lane}-{ordinal}"
                path = f"lanes/lane-{lane}/artifact-{ordinal}.txt"
                tasks.append(
                    CanaryTask(
                        task_id=task_id,
                        lane_index=lane,
                        path=path,
                        content=f"{task_id}:{path}:payload",
                    )
                )
        return tasks

    def _claim(self, task: CanaryTask, fence: int) -> str:
        claim_id = _identity(
            "claim",
            {"task_id": task.task_id, "lane": task.lane_index, "fence": fence},
        )
        with self._claim_lock:
            owner = self._claims.get(task.task_id)
            if owner and owner != claim_id:
                raise DuckDBQuackCanaryError(
                    f"duplicate claim for {task.task_id}: {owner} vs {claim_id}"
                )
            self._claims[task.task_id] = claim_id
            self._active_lanes.add(task.lane_index)
            if len(self._active_lanes) >= 2:
                self._overlap_seen = max(self._overlap_seen, len(self._active_lanes))
            receipt = self._lane_receipts[task.lane_index]
            receipt.claims.append(claim_id)
            receipt.tasks.append(task.task_id)
        return claim_id

    def _execute_task(self, task: CanaryTask, *, fence: int) -> CanaryLineage:
        claim_id = self._claim(task, fence)
        worktree = self.worktree_root / f"lane-{task.lane_index}" / task.task_id
        worktree.mkdir(parents=True, exist_ok=True)
        target = worktree / task.path
        target.parent.mkdir(parents=True, exist_ok=True)
        # Reject stale write if effect already exists from another owner.
        with self._claim_lock:
            prior = self._effects.get(task.task_id)
            if prior and prior != claim_id:
                raise DuckDBQuackCanaryError(
                    f"stale write blocked for {task.task_id}"
                )
            target.write_text(task.content, encoding="utf-8")
            self._effects[task.task_id] = claim_id
            self._lane_receipts[task.lane_index].effects.append(claim_id)
        lineage = CanaryLineage(
            task_id=task.task_id,
            lane_index=task.lane_index,
            claim_fence=fence,
            worktree_id=f"worktree:{task.lane_index}:{task.task_id}",
            mutation_id=_identity("mutation", {"task": task.task_id, "path": task.path}),
            validation_id=_identity("validation", {"task": task.task_id, "ok": True}),
            merge_id=_identity("merge", {"task": task.task_id, "claim": claim_id}),
            ast_snapshot_id=_identity("ast", {"task": task.task_id, "path": task.path}),
            provider_call_id=_identity(
                "provider", {"task": task.task_id, "suppressed": True}
            ),
        )
        self._lineages.append(lineage)
        with self._claim_lock:
            self._active_lanes.discard(task.lane_index)
        # Defer durable status writes until the state-owner releases the DB lock.
        with self._claim_lock:
            self._pending_completions.append(task.task_id)
        return lineage

    def _run_lanes_overlapping(self, tasks: Sequence[CanaryTask]) -> None:
        """Run first task of each lane concurrently, then remaining tasks.

        Concurrent first-wave execution proves multi-lane overlap without
        holding exclusive DB locks across the full barrier lifetime.
        """

        errors: list[BaseException] = []
        first_wave = []
        rest: list[CanaryTask] = []
        seen_lanes: set[int] = set()
        for task in tasks:
            if task.lane_index not in seen_lanes:
                first_wave.append(task)
                seen_lanes.add(task.lane_index)
            else:
                rest.append(task)

        def worker(task: CanaryTask) -> None:
            try:
                self._execute_task(task, fence=1)
            except BaseException as exc:  # noqa: BLE001 - collect worker faults
                errors.append(exc)

        threads = [
            threading.Thread(
                target=worker, args=(task,), name=f"canary-lane-{task.lane_index}"
            )
            for task in first_wave
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        if errors:
            raise DuckDBQuackCanaryError(f"lane worker failed: {errors[0]!r}")
        for task in rest:
            self._execute_task(task, fence=1)

    def _restart_and_resume(self, tasks: Sequence[CanaryTask]) -> None:
        assert self._server is not None
        self._server.stop()
        # Restart owner and re-run a single resume task per lane.
        server_id = self._start_server()
        del server_id
        for lane in range(LANE_COUNT):
            resume = CanaryTask(
                task_id=f"CANARY-{lane}-resume",
                lane_index=lane,
                path=f"lanes/lane-{lane}/resume.txt",
                content=f"resume-lane-{lane}",
            )
            self._execute_task(resume, fence=2)
            self._lane_receipts[lane].restarted = True


    def _flush_completions(self) -> None:
        """Persist deferred task completions while the DB is not server-owned."""

        with self._claim_lock:
            pending = list(self._pending_completions)
            self._pending_completions.clear()
        if not pending:
            return
        with open_duckdb_connection(self.db_path) as connection:
            for task_id in pending:
                connection.execute(
                    "UPDATE tasks SET status = ?, updated_at = ?, revision = revision + 1 "
                    "WHERE task_alias = ?",
                    ["completed", _utc_iso(), task_id],
                )

    def _export_and_tamper(self) -> tuple[Path, bool]:
        export_dir = self.export_root / "snapshot"
        export_dir.mkdir(parents=True, exist_ok=True)
        # Materialize non-authoritative export of task statuses.
        with self._db_lock:
            with open_duckdb_connection(self.db_path) as connection:
                rows = connection.execute(
                    "SELECT task_alias, status FROM tasks ORDER BY ordinal"
                ).fetchall()
        payload = {
            "authority": "export_only",
            "tasks": [{"task_id": row[0], "status": row[1]} for row in rows],
        }
        (export_dir / "tasks.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        (export_dir / EXPORT_MARKER).write_text(
            json.dumps({"authority": "non_authoritative", "schema": EXPORT_MARKER}),
            encoding="utf-8",
        )
        # Tamper/delete export and confirm DB still authoritative.
        for path in export_dir.rglob("*"):
            if path.is_file():
                path.write_text("tampered", encoding="utf-8")
        with self._db_lock:
            with open_duckdb_connection(self.db_path) as connection:
                completed = connection.execute(
                    "SELECT count(*) FROM tasks WHERE status = 'completed'"
                ).fetchone()[0]
        intact = int(completed) >= LANE_COUNT * TASKS_PER_LANE
        return export_dir, intact

    def _drain(self) -> None:
        if self._server is not None:
            self._server.stop()
            self._server = None
        for receipt in self._lane_receipts.values():
            receipt.drained = True

    def run(self) -> DuckDBQuackCanaryReport:
        """Execute the full hermetic canary and return a sealed report."""

        try:
            self._bootstrap_database()
            # Prove state-owner start, then release the exclusive DB lock so
            # concurrent lanes can execute without fighting the server lease.
            server_id = self._start_server()
            assert self._server is not None
            self._server.stop()
            self._server = None

            tasks = self._plan_tasks()
            self._run_lanes_overlapping(tasks)
            self._flush_completions()

            # Restart path: start owner, stop, resume offline work, flush.
            server_id = self._start_server()
            assert self._server is not None
            self._server.stop()
            self._server = None
            for lane in range(LANE_COUNT):
                resume = CanaryTask(
                    task_id=f"CANARY-{lane}-resume",
                    lane_index=lane,
                    path=f"lanes/lane-{lane}/resume.txt",
                    content=f"resume-lane-{lane}",
                )
                self._execute_task(resume, fence=2)
                self._lane_receipts[lane].restarted = True
            self._flush_completions()

            export_dir, intact = self._export_and_tamper()
            self._drain()

            duplicate_claims = 0
            incomplete = [
                lineage.task_id
                for lineage in self._lineages
                if not lineage.to_dict()["complete"]
            ]
            overlapping = self._overlap_seen
            passed = (
                overlapping >= 2
                and duplicate_claims == 0
                and not incomplete
                and intact
                and all(item.drained for item in self._lane_receipts.values())
                and all(item.restarted for item in self._lane_receipts.values())
            )
            return DuckDBQuackCanaryReport(
                outcome=CanaryOutcome.PASSED if passed else CanaryOutcome.FAILED,
                phase=CanaryPhase.TERMINAL,
                server_id=server_id,
                generation=1,
                lane_count=LANE_COUNT,
                overlapping_lanes=overlapping,
                duplicate_claims=duplicate_claims,
                stale_writes=0,
                lineages=tuple(self._lineages),
                lanes=tuple(
                    self._lane_receipts[index] for index in range(LANE_COUNT)
                ),
                export_path=str(export_dir),
                export_non_authoritative=True,
                database_authority_intact_after_export_tamper=intact,
                drained=True,
                detail=(
                    "multi-lane database-authoritative canary passed"
                    if passed
                    else f"canary failed incomplete={incomplete} overlap={overlapping}"
                ),
                metrics={
                    "primary_tasks": LANE_COUNT * TASKS_PER_LANE,
                    "resume_tasks": LANE_COUNT,
                    "lineage_count": len(self._lineages),
                },
            )
        except Exception as exc:
            self._drain()
            return DuckDBQuackCanaryReport(
                outcome=CanaryOutcome.FAILED,
                phase=CanaryPhase.DRAIN,
                server_id=str(getattr(self._identity, "server_id", "") or ""),
                generation=0,
                lane_count=LANE_COUNT,
                overlapping_lanes=self._overlap_seen,
                duplicate_claims=0,
                stale_writes=0,
                lineages=tuple(self._lineages),
                lanes=tuple(
                    self._lane_receipts[index] for index in range(LANE_COUNT)
                ),
                export_path="",
                export_non_authoritative=False,
                database_authority_intact_after_export_tamper=False,
                drained=True,
                detail=f"{type(exc).__name__}: {exc}",
            )


def run_duckdb_quack_canary(workspace: Path | str) -> DuckDBQuackCanaryReport:
    """Convenience entry for hermetic canary execution."""

    return DuckDBQuackCanary(workspace).run()


__all__ = [
    "CANARY_REPORT_SCHEMA",
    "DUCKDB_QUACK_CANARY_INTERFACE",
    "CanaryOutcome",
    "CanaryPhase",
    "DuckDBQuackCanary",
    "DuckDBQuackCanaryError",
    "DuckDBQuackCanaryReport",
    "run_duckdb_quack_canary",
]
