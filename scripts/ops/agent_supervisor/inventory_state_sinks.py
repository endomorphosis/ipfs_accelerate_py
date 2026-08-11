#!/usr/bin/env python3
"""Deterministic inventory of mutable agent-supervisor state sinks.

DQP-001 / SupervisorStateSinkInventory@1
========================================

Every path that the supervisor currently treats as writable orchestration
state is classified here. The scanner:

1. walks ``ipfs_accelerate_py/agent_supervisor/**/*.py`` and extracts string
   literals (tokenize stream, including f-string constant fragments) looking
   for concrete path markers (``.duckdb``, ``.sqlite*``, ``.jsonl``,
   ``.pid``, ``.lock``, ``status.json``, ``progress.json``, objectives and
   taskboard markdown);
2. matches each discovery against a curated catalog;
3. fails closed when a mutable marker has no classification;
4. records direct DuckDB writers, cross-file atomicity gaps, and reuse
   candidates; and
5. distinguishes Git/source bytes (not supervisor state) from orchestration
   sinks that must migrate into the DuckDB + Quack control plane.

The catalog is the authority for classification. Discovery is the CI gate
that prevents silent new sinks.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import sys
import tokenize
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final

# ---------------------------------------------------------------------------
# Contract identity
# ---------------------------------------------------------------------------

INTERFACE_ID: Final[str] = "SupervisorStateSinkInventory@1"
INVENTORY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/state-sink-inventory@1"
)
INVENTORY_VERSION: Final[str] = "1"
INVENTORY_DOC_RELATIVE: Final[str] = (
    "docs/architecture/AGENT_SUPERVISOR_STATE_SINK_INVENTORY.md"
)
AGENT_SUPERVISOR_PACKAGE: Final[str] = "ipfs_accelerate_py/agent_supervisor"

# Inventory observations never authorize mutation, completion, or proof.
INVENTORY_IS_COMPLETION_EVIDENCE: Final[bool] = False
INVENTORY_IS_PROOF_EVIDENCE: Final[bool] = False
INVENTORY_AUTHORIZES_MUTATION: Final[bool] = False


class SinkClassification(str, Enum):
    """Closed taxonomy for every inventoried mutable sink."""

    AUTHORITY = "authority"
    STATIC_INPUT = "static_input"
    IMMUTABLE_EVIDENCE = "immutable_evidence"
    CACHE = "cache"
    EXPORT = "export"
    OS_BOOTSTRAP = "os_bootstrap"
    EMERGENCY_DIAGNOSTIC = "emergency_diagnostic"


class DestinationDomain(str, Enum):
    """Canonical control-plane domains from the migration plan."""

    SCHEMA_DEPLOYMENT = "schema_deployment"
    REPOSITORY_WORKTREE = "repository_worktree"
    OBJECTIVES_PLANS_TASKS = "objectives_plans_tasks"
    EXECUTION_LIFECYCLE = "execution_lifecycle"
    EVENTS_LOGS_METRICS = "events_logs_metrics"
    CODE_INTELLIGENCE = "code_intelligence"
    ARTIFACTS_PROOFS = "artifacts_proofs"
    CONTEXT_BUDGETS = "context_budgets"
    NON_STATE = "non_state"


class RetirementStage(str, Enum):
    """When the sink loses write authority under the staged cutover."""

    RETAIN_PERMANENT = "retain_permanent"
    BOOTSTRAP = "bootstrap"
    FOUNDATION = "foundation"
    SHADOW_IMPORT = "shadow_import"
    DUAL_OBSERVATION = "dual_observation"
    DATABASE_AUTHORITY_CANARY = "database_authority_canary"
    DEFAULT_CUTOVER = "default_cutover"
    LEGACY_RETIREMENT = "legacy_retirement"


class MediaType(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"
    JSONL = "jsonl"
    PID = "pid"
    LOCK = "lock"
    SQLITE = "sqlite"
    DUCKDB = "duckdb"
    CACHE_DIR = "cache_dir"
    ARTIFACT = "artifact"
    DIRECTORY = "directory"
    GIT_SOURCE = "git_source"


class AtomicityModel(str, Enum):
    """How the sink coordinates multi-writer / multi-file consistency today."""

    SINGLE_FILE_ATOMIC = "single_file_atomic"
    SINGLE_FILE_TRANSACTION = "single_file_transaction"
    FLOCK_PLUS_TRANSACTION = "flock_plus_transaction"
    APPEND_ONLY = "append_only"
    BEST_EFFORT_MIRROR = "best_effort_mirror"
    CROSS_FILE_NON_ATOMIC = "cross_file_non_atomic"
    OS_PROCESS_HANDLE = "os_process_handle"
    NOT_APPLICABLE = "not_applicable"


# Path markers that indicate a mutable supervisor state sink.
# Intentionally narrower than "any .json" so config/schema fixtures are not
# treated as orchestration writers; authority JSON basenames are enumerated.
_SINK_BASENAME_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:"
    r"[\w.-]+\.duckdb|"
    r"[\w.-]+\.sqlite3?|"
    r"[\w.-]+\.jsonl|"
    r"[\w.-]+\.pid|"
    r"[\w.-]+\.lock|"
    r"[\w.-]*status\.json|"
    r"[\w.-]*progress\.json|"
    r"[\w.-]+\.objectives\.md|"
    r"[\w.-]+\.todo\.md|"
    r"events\.jsonl|"
    r"active\.json|"
    r"index\.json|"
    r"prior_active\.json|"
    r"supersessions\.jsonl|"
    r"bundle_lanes\.json|"
    r"scheduler_metrics\.json|"
    r"scheduler_decision_metrics\.json|"
    r"task_queue\.json|"
    r"coordination\.duckdb|"
    r"tasks\.duckdb|"
    r"merge_queue\.duckdb|"
    r"merge_resolver\.duckdb|"
    r"consumer\.lock|"
    r"implementation-main-merge\.lock|"
    r"\.run-registry\.lock|"
    r"\.analysis-cache\.lock|"
    r"\.repository-index\.lock|"
    r"cas\.lock|"
    r"fence\.lock|"
    r"agent-llm-resolver\.lock"
    r")",
    re.IGNORECASE,
)

# Suffixes / basenames that are never treated as mutable supervisor state.
# Matching is case-insensitive (see ``_is_non_sink_basename``).
_NON_SINK_BASENAME_ALLOWLIST: Final[frozenset[str]] = frozenset(
    {
        # Package / packaging metadata, not runtime state.
        "pyproject.toml",
        "package.json",
        "setup.py",
        # Documentation examples and fixture names that are not writers.
        "canary.sqlite",
        "example.duckdb",
        "fixture.duckdb",
        "sample.db",
        # Language/package-manager lockfiles (source tree inputs, not
        # supervisor orchestration state). Git index / ref locks are also
        # non-state (byte-authority side of Git, not supervisor authority).
        "cargo.lock",
        "composer.lock",
        "config.lock",
        "deno.lock",
        "gemfile.lock",
        "head.lock",
        "index.lock",
        "package-lock.json",
        "packed-refs.lock",
        "pipfile.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "requirements.lock",
        "shallow.lock",
        "uv.lock",
        "yarn.lock",
    }
)

# Glob basenames (``*.suffix``) that may appear in production source and still
# indicate a mutable supervisor sink family. Arbitrary globs such as ``*.py``,
# ``*.json``, or ``*.pem`` are source/scan patterns, not state sinks.
_SINK_GLOB_BASENAMES: Final[frozenset[str]] = frozenset(
    {
        "*.duckdb",
        "*.sqlite",
        "*.sqlite3",
        "*.jsonl",
        "*.pid",
        "*.lock",
        "*.objectives.md",
        "*.todo.md",
        "*status.json",
        "*progress.json",
    }
)

# Media families that indicate a mutable supervisor state sink.
_ALL_MUTABLE_MEDIA: Final[frozenset[MediaType]] = frozenset(
    {
        MediaType.DUCKDB,
        MediaType.SQLITE,
        MediaType.JSON,
        MediaType.JSONL,
        MediaType.PID,
        MediaType.LOCK,
        MediaType.MARKDOWN,
        MediaType.ARTIFACT,
        MediaType.DIRECTORY,
        MediaType.CACHE_DIR,
    }
)

# Modules that mention path shapes only as documentation / type examples.
_SCAN_SKIP_PARTS: Final[frozenset[str]] = frozenset(
    {
        "__pycache__",
        ".git",
    }
)


@dataclass(frozen=True)
class StateSink:
    """One classified mutable supervisor state sink."""

    sink_id: str
    writer_module: str
    path_template: str
    media_type: MediaType
    classification: SinkClassification
    destination_domain: DestinationDomain
    retirement_stage: RetirementStage
    reuse_candidate: str
    atomicity_model: AtomicityModel
    discovery_basenames: tuple[str, ...] = ()
    discovery_modules: tuple[str, ...] = ()
    is_direct_duckdb_writer: bool = False
    is_git_source_bytes: bool = False
    cross_file_atomicity_gap: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.sink_id or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{1,127}", self.sink_id):
            raise ValueError(f"invalid sink_id: {self.sink_id!r}")
        if self.is_git_source_bytes and self.classification != SinkClassification.STATIC_INPUT:
            raise ValueError(
                f"git/source bytes sink {self.sink_id} must be static_input"
            )
        if self.is_direct_duckdb_writer and self.media_type not in {
            MediaType.DUCKDB,
            MediaType.SQLITE,
            MediaType.ARTIFACT,
        }:
            raise ValueError(
                f"direct DuckDB writer {self.sink_id} requires duckdb/sqlite/artifact media"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["media_type"] = self.media_type.value
        payload["classification"] = self.classification.value
        payload["destination_domain"] = self.destination_domain.value
        payload["retirement_stage"] = self.retirement_stage.value
        payload["atomicity_model"] = self.atomicity_model.value
        payload["discovery_basenames"] = list(self.discovery_basenames)
        payload["discovery_modules"] = list(self.discovery_modules)
        return payload


@dataclass(frozen=True)
class DiscoveredMarker:
    """A path marker found in production agent-supervisor source."""

    module: str
    basename: str
    line: int
    literal: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InventoryReport:
    """Deterministic inventory result."""

    schema: str
    version: str
    interface_id: str
    sinks: tuple[StateSink, ...]
    discoveries: tuple[DiscoveredMarker, ...]
    unclassified: tuple[DiscoveredMarker, ...]
    direct_duckdb_writers: tuple[str, ...]
    cross_file_atomicity_gaps: tuple[str, ...]
    reuse_candidates: tuple[str, ...]
    git_source_distinctions: tuple[str, ...]
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.unclassified and not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "version": self.version,
            "interface_id": self.interface_id,
            "ok": self.ok,
            "sink_count": len(self.sinks),
            "discovery_count": len(self.discoveries),
            "unclassified_count": len(self.unclassified),
            "sinks": [sink.to_dict() for sink in self.sinks],
            "discoveries": [item.to_dict() for item in self.discoveries],
            "unclassified": [item.to_dict() for item in self.unclassified],
            "direct_duckdb_writers": list(self.direct_duckdb_writers),
            "cross_file_atomicity_gaps": list(self.cross_file_atomicity_gaps),
            "reuse_candidates": list(self.reuse_candidates),
            "git_source_distinctions": list(self.git_source_distinctions),
            "errors": list(self.errors),
            "inventory_is_completion_evidence": INVENTORY_IS_COMPLETION_EVIDENCE,
            "inventory_is_proof_evidence": INVENTORY_IS_PROOF_EVIDENCE,
            "inventory_authorizes_mutation": INVENTORY_AUTHORIZES_MUTATION,
        }


class UnclassifiedMutableSinkError(RuntimeError):
    """Raised when discovery finds a mutable sink outside the catalog."""

    def __init__(self, unclassified: Sequence[DiscoveredMarker]) -> None:
        self.unclassified = tuple(unclassified)
        details = ", ".join(
            f"{item.module}:{item.line}:{item.basename}" for item in self.unclassified[:12]
        )
        if len(self.unclassified) > 12:
            details += f", ... (+{len(self.unclassified) - 12} more)"
        super().__init__(
            f"unclassified mutable supervisor state sink(s): {details}"
        )


def _sink(
    sink_id: str,
    *,
    writer_module: str,
    path_template: str,
    media_type: MediaType,
    classification: SinkClassification,
    destination_domain: DestinationDomain,
    retirement_stage: RetirementStage,
    reuse_candidate: str,
    atomicity_model: AtomicityModel,
    discovery_basenames: Sequence[str] = (),
    discovery_modules: Sequence[str] = (),
    is_direct_duckdb_writer: bool = False,
    is_git_source_bytes: bool = False,
    cross_file_atomicity_gap: str = "",
    notes: str = "",
) -> StateSink:
    return StateSink(
        sink_id=sink_id,
        writer_module=writer_module,
        path_template=path_template,
        media_type=media_type,
        classification=classification,
        destination_domain=destination_domain,
        retirement_stage=retirement_stage,
        reuse_candidate=reuse_candidate,
        atomicity_model=atomicity_model,
        discovery_basenames=tuple(discovery_basenames),
        discovery_modules=tuple(discovery_modules or (writer_module,)),
        is_direct_duckdb_writer=is_direct_duckdb_writer,
        is_git_source_bytes=is_git_source_bytes,
        cross_file_atomicity_gap=cross_file_atomicity_gap,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Curated catalog — ordered by sink_id for deterministic emission
# ---------------------------------------------------------------------------

KNOWN_STATE_SINKS: Final[tuple[StateSink, ...]] = tuple(
    sorted(
        (
            # --- Git / source bytes (explicit non-orchestration) ---
            _sink(
                "git-source-bytes",
                writer_module="ipfs_accelerate_py/agent_supervisor/analysis/repository_forest.py",
                path_template="{repo_root}/.git/** and worktree source files",
                media_type=MediaType.GIT_SOURCE,
                classification=SinkClassification.STATIC_INPUT,
                destination_domain=DestinationDomain.NON_STATE,
                retirement_stage=RetirementStage.RETAIN_PERMANENT,
                reuse_candidate=(
                    "Repository forest + worktree snapshot identities; never "
                    "store source blobs as control-plane authority"
                ),
                atomicity_model=AtomicityModel.NOT_APPLICABLE,
                discovery_basenames=(),
                is_git_source_bytes=True,
                notes=(
                    "Git remains the byte authority for source. The control plane "
                    "stores identities, AST, and mutation history only."
                ),
            ),
            # --- Objectives / taskboards (markdown authority today) ---
            _sink(
                "objective-heap-markdown",
                writer_module="ipfs_accelerate_py/agent_supervisor/objectives/objective_tracker.py",
                path_template="docs/architecture/*.objectives.md",
                media_type=MediaType.MARKDOWN,
                classification=SinkClassification.AUTHORITY,
                destination_domain=DestinationDomain.OBJECTIVES_PLANS_TASKS,
                retirement_stage=RetirementStage.DATABASE_AUTHORITY_CANARY,
                reuse_candidate=(
                    "objective_graph + goal_completion identity/revision contracts"
                ),
                atomicity_model=AtomicityModel.CROSS_FILE_NON_ATOMIC,
                discovery_basenames=("*.objectives.md",),
                discovery_modules=(
                    "ipfs_accelerate_py/agent_supervisor/objectives/objective_tracker.py",
                    "ipfs_accelerate_py/agent_supervisor/objectives/objective_graph.py",
                    "ipfs_accelerate_py/agent_supervisor/objectives/backlog_refinery.py",
                    "ipfs_accelerate_py/agent_supervisor/objectives/objective_daemon.py",
                ),
                cross_file_atomicity_gap=(
                    "Objective heap rewrites and paired taskboard/status updates "
                    "are not one transaction; crash can diverge intent vs schedule"
                ),
                notes="Durable intent authority today; becomes import + export only.",
            ),
            _sink(
                "taskboard-markdown",
                writer_module="ipfs_accelerate_py/agent_supervisor/task_sources/markdown_task_source.py",
                path_template="docs/architecture/*.todo.md",
                media_type=MediaType.MARKDOWN,
                classification=SinkClassification.AUTHORITY,
                destination_domain=DestinationDomain.OBJECTIVES_PLANS_TASKS,
                retirement_stage=RetirementStage.DATABASE_AUTHORITY_CANARY,
                reuse_candidate=(
                    "TaskboardStore materialization journal + DuckDBTaskSource "
                    "fenced projection"
                ),
                atomicity_model=AtomicityModel.CROSS_FILE_NON_ATOMIC,
                discovery_basenames=("*.todo.md",),
                discovery_modules=(
                    "ipfs_accelerate_py/agent_supervisor/task_sources/markdown_task_source.py",
                    "ipfs_accelerate_py/agent_supervisor/task_sources/taskboard_store.py",
                    "ipfs_accelerate_py/agent_supervisor/todo_daemon/runner.py",
                    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
                    "ipfs_accelerate_py/agent_supervisor/objectives/backlog_refinery.py",
                ),
                cross_file_atomicity_gap=(
                    "Board status writes, events.jsonl, and daemon status JSON "
                    "are separate files without a shared transaction"
                ),
                notes="Schedulable projection; still treated as claim authority pre-cutover.",
            ),
            _sink(
                "taskboard-store-events",
                writer_module="ipfs_accelerate_py/agent_supervisor/task_sources/taskboard_store.py",
                path_template="{board}.events.jsonl + .{board}.store.lock",
                media_type=MediaType.JSONL,
                classification=SinkClassification.AUTHORITY,
                destination_domain=DestinationDomain.EVENTS_LOGS_METRICS,
                retirement_stage=RetirementStage.DEFAULT_CUTOVER,
                reuse_candidate="RotatingEventLog / domain_events stream",
                atomicity_model=AtomicityModel.APPEND_ONLY,
                discovery_basenames=("events.jsonl",),
                discovery_modules=(
                    "ipfs_accelerate_py/agent_supervisor/task_sources/taskboard_store.py",
                ),
                cross_file_atomicity_gap=(
                    "Event append and markdown board rewrite are not atomic together"
                ),
            ),
            _sink(
                "taskboard-materialization-lock",
                writer_module="ipfs_accelerate_py/agent_supervisor/task_sources/taskboard_store.py",
                path_template=".{board}.materialization.lock / .{board}.store.lock",
                media_type=MediaType.LOCK,
                classification=SinkClassification.OS_BOOTSTRAP,
                destination_domain=DestinationDomain.EXECUTION_LIFECYCLE,
                retirement_stage=RetirementStage.DEFAULT_CUTOVER,
                reuse_candidate="Process-birth lease + fencing epoch in control plane",
                atomicity_model=AtomicityModel.OS_PROCESS_HANDLE,
                discovery_basenames=(),
                notes="Lock files serialize writers; they are not claim authority.",
            ),
            # --- DuckDB direct writers ---
            _sink(
                "duckdb-task-source",
                writer_module="ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py",
                path_template="{state_root}/tasks.duckdb (caller-supplied path)",
                media_type=MediaType.DUCKDB,
                classification=SinkClassification.AUTHORITY,
                destination_domain=DestinationDomain.OBJECTIVES_PLANS_TASKS,
                retirement_stage=RetirementStage.FOUNDATION,
                reuse_candidate=(
                    "Primary reuse target for control-plane tasks/claims schema "
                    "via duckdb_state flock + short transactions"
                ),
                atomicity_model=AtomicityModel.FLOCK_PLUS_TRANSACTION,
                discovery_basenames=("tasks.duckdb",),
                discovery_modules=(
                    "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_task_source.py",
                    "ipfs_accelerate_py/agent_supervisor/entrypoints/objective_resolver.py",
                    "ipfs_accelerate_py/agent_supervisor/entrypoints/profile_resolver.py",
                ),
                is_direct_duckdb_writer=True,
                notes="Versioned fenced task projection; foundational for Quack cutover.",
            ),
            _sink(
                "duckdb-state-primitives",
                writer_module="ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
                path_template="{path}.duckdb + .{name}.lock (+ optional legacy .sqlite3)",
                media_type=MediaType.DUCKDB,
                classification=SinkClassification.AUTHORITY,
                destination_domain=DestinationDomain.SCHEMA_DEPLOYMENT,
                retirement_stage=RetirementStage.FOUNDATION,
                reuse_candidate=(
                    "Shared exclusive_file_lock + open_duckdb_connection for all "
                    "single-writer DuckDB stores"
                ),
                atomicity_model=AtomicityModel.FLOCK_PLUS_TRANSACTION,
                discovery_basenames=(),
                is_direct_duckdb_writer=True,
                notes="Not a store itself; primitives every DuckDB writer reuses.",
            ),
            _sink(
                "lease-coordination-duckdb",
                writer_module="ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py",
                path_template="{state_root}/coordination.duckdb",
                media_type=MediaType.DUCKDB,
                classification=SinkClassification.AUTHORITY,
                destination_domain=DestinationDomain.EXECUTION_LIFECYCLE,
                retirement_stage=RetirementStage.FOUNDATION,
                reuse_candidate=(
                    "LeaseCoordinator fencing/CAS model becomes control-plane leases"
                ),
                atomicity_model=AtomicityModel.FLOCK_PLUS_TRANSACTION,
                discovery_basenames=("coordination.duckdb",),
                discovery_modules=(
                    "ipfs_accelerate_py/agent_supervisor/merge/lease_coordination.py",
                    "ipfs_accelerate_py/agent_supervisor/objectives/bundle_supervisor.py",
                    "ipfs_accelerate_py/agent_supervisor/merge/leased_lane.py",
                ),
                is_direct_duckdb_writer=True,
            ),
            _sink(
                "merge-queue-duckdb",
                writer_module="ipfs_accelerate_py/agent_supervisor/merge/merge_queue.py",
                path_template="{queue_dir}/merge_queue.duckdb (+ legacy merge_queue.sqlite3)",
                media_type=MediaType.DUCKDB,
                classification=SinkClassification.AUTHORITY,
                destination_domain=DestinationDomain.REPOSITORY_WORKTREE,
                retirement_stage=RetirementStage.DEFAULT_CUTOVER,
                reuse_candidate="merge_queue_entries + resource_claims tables",
                atomicity_model=AtomicityModel.FLOCK_PLUS_TRANSACTION,
                discovery_basenames=(
                    "merge_queue.duckdb",
                    "merge_queue.sqlite3",
                ),
                is_direct_duckdb_writer=True,
                cross_file_atomicity_gap=(
                    "Queue DB, train receipts directory, and checkout lock files "
                    "are independent writers"
                ),
            ),
            _sink(
                "merge-resolver-duckdb",
                writer_module="ipfs_accelerate_py/agent_supervisor/merge/merge_resolver.py",
                path_template="{state_dir}/merge_resolver.duckdb (+ legacy .sqlite3)",
                media_type=MediaType.DUCKDB,
                classification=SinkClassification.AUTHORITY,
                destination_domain=DestinationDomain.REPOSITORY_WORKTREE,
                retirement_stage=RetirementStage.DEFAULT_CUTOVER,
                reuse_candidate="merge_attempts domain events + MergeQueue fence",
                atomicity_model=AtomicityModel.FLOCK_PLUS_TRANSACTION,
                discovery_basenames=(
                    "merge_resolver.duckdb",
                    "merge_resolver.sqlite3",
                ),
                is_direct_duckdb_writer=True,
            ),
            _sink(
                "artifact-store-json-duckdb",
                writer_module="ipfs_accelerate_py/agent_supervisor/runtime/artifact_store.py",
                path_template="{artifact}.json + sidecar {artifact}.duckdb",
                media_type=MediaType.ARTIFACT,
                classification=SinkClassification.IMMUTABLE_EVIDENCE,
                destination_domain=DestinationDomain.ARTIFACTS_PROOFS,
                retirement_stage=RetirementStage.DUAL_OBSERVATION,
                reuse_candidate=(
                    "Bounded ArtifactStore dual JSON/DuckDB projection pattern"
                ),
                atomicity_model=AtomicityModel.CROSS_FILE_NON_ATOMIC,
                discovery_basenames=(),
                discovery_modules=(
                    "ipfs_accelerate_py/agent_supervisor/runtime/artifact_store.py",
                ),
                is_direct_duckdb_writer=True,
                cross_file_atomicity_gap=(
                    "JSON body and DuckDB sidecar are dual-written; crash can leave "
                    "one side stale until rebuild"
                ),
                notes="JSON is portable interchange; DuckDB is a query projection.",
            ),
            _sink(
                "proof-scheduler-duckdb",
                writer_module="ipfs_accelerate_py/agent_supervisor/proof/proof_scheduler.py",
                path_template="{root}/proof_scheduler.duckdb",
                media_type=MediaType.DUCKDB,
                classification=SinkClassification.AUTHORITY,
                destination_domain=DestinationDomain.ARTIFACTS_PROOFS,
                retirement_stage=RetirementStage.DEFAULT_CUTOVER,
                reuse_candidate="duckdb_state resolve_duckdb_path + flock pattern",
                atomicity_model=AtomicityModel.FLOCK_PLUS_TRANSACTION,
                discovery_basenames=("proof_scheduler.duckdb",),
                is_direct_duckdb_writer=True,
            ),
            _sink(
                "prover-evidence-duckdb",
                writer_module="ipfs_accelerate_py/agent_supervisor/proof/prover_evidence_store.py",
                path_template="{root}/prover_evidence.duckdb (+ .json projection)",
                media_type=MediaType.DUCKDB,
                classification=SinkClassification.IMMUTABLE_EVIDENCE,
                destination_domain=DestinationDomain.ARTIFACTS_PROOFS,
                retirement_stage=RetirementStage.DUAL_OBSERVATION,
                reuse_candidate="Artifact metadata tables + content-addressed receipts",
                atomicity_model=AtomicityModel.FLOCK_PLUS_TRANSACTION,
                discovery_basenames=("prover_evidence.duckdb",),
                is_direct_duckdb_writer=True,
            ),
            _sink(
                "formal-verification-cache-duckdb",
                writer_module="ipfs_accelerate_py/agent_supervisor/proof/formal_verification_cache.py",
                path_template="{root}/formal_verification_cache.duckdb",
                media_type=MediaType.DUCKDB,
                classification=SinkClassification.CACHE,
                destination_domain=DestinationDomain.ARTIFACTS_PROOFS,
                retirement_stage=RetirementStage.LEGACY_RETIREMENT,
                reuse_candidate="Content-addressed proof cache keyed by obligation digest",
                atomicity_model=AtomicityModel.FLOCK_PLUS_TRANSACTION,
                discovery_basenames=("formal_verification_cache.duckdb",),
                is_direct_duckdb_writer=True,
                notes="Cache only; never completion or proof authority by itself.",
            ),
            _sink(
                "legacy-landed-review-duckdb",
                writer_module="ipfs_accelerate_py/agent_supervisor/todo_daemon/legacy_landed_result_cache.py",
                path_template="{root}/legacy_landed_review_results.duckdb",
                media_type=MediaType.DUCKDB,
                classification=SinkClassification.CACHE,
                destination_domain=DestinationDomain.EXECUTION_LIFECYCLE,
                retirement_stage=RetirementStage.LEGACY_RETIREMENT,
                reuse_candidate="Attempt result projection tables",
                atomicity_model=AtomicityModel.FLOCK_PLUS_TRANSACTION,
                discovery_basenames=("legacy_landed_review_results.duckdb",),
                is_direct_duckdb_writer=True,
            ),
            _sink(
                "bundle-index-duckdb-sidecar",
                writer_module="ipfs_accelerate_py/agent_supervisor/task_sources/todo_vector_index.py",
                path_template="{bundle_index}.json + {bundle_index}.duckdb",
                media_type=MediaType.DUCKDB,
                classification=SinkClassification.CACHE,
                destination_domain=DestinationDomain.CONTEXT_BUDGETS,
                retirement_stage=RetirementStage.LEGACY_RETIREMENT,
                reuse_candidate="ArtifactStore dual-write + bounded query tables",
                atomicity_model=AtomicityModel.CROSS_FILE_NON_ATOMIC,
                discovery_basenames=(),
                discovery_modules=(
                    "ipfs_accelerate_py/agent_supervisor/task_sources/todo_vector_index.py",
                    "ipfs_accelerate_py/agent_supervisor/objectives/objective_graph.py",
                    "ipfs_accelerate_py/agent_supervisor/objectives/bundle_supervisor.py",
                ),
                is_direct_duckdb_writer=True,
                cross_file_atomicity_gap=(
                    "Bundle planning JSON and DuckDB sidecar may diverge after crash"
                ),
            ),
            _sink(
                "doctor-proof-cache-sqlite",
                writer_module="ipfs_accelerate_py/agent_supervisor/proof/doctor_proof_cache.py",
                path_template="{root}/doctor_proof_cache.sqlite3 (caller path)",
                media_type=MediaType.SQLITE,
                classification=SinkClassification.CACHE,
                destination_domain=DestinationDomain.ARTIFACTS_PROOFS,
                retirement_stage=RetirementStage.LEGACY_RETIREMENT,
                reuse_candidate="Migrate onto formal_verification_cache.duckdb primitives",
                atomicity_model=AtomicityModel.SINGLE_FILE_TRANSACTION,
                discovery_basenames=(),
                is_direct_duckdb_writer=False,
                notes="Legacy SQLite proof memo; not orchestration authority.",
            ),
            _sink(
                "analysis-singleflight-sqlite",
                writer_module="ipfs_accelerate_py/agent_supervisor/analysis/cache_coordinator.py",
                path_template="{cache}/single-flight.sqlite3",
                media_type=MediaType.SQLITE,
                classification=SinkClassification.CACHE,
                destination_domain=DestinationDomain.CODE_INTELLIGENCE,
                retirement_stage=RetirementStage.LEGACY_RETIREMENT,
                reuse_candidate="Distributed single-flight via LeaseCoordinator",
                atomicity_model=AtomicityModel.SINGLE_FILE_TRANSACTION,
                discovery_basenames=("single-flight.sqlite3",),
            ),
            # --- Plan revision store (multi-file) ---
            _sink(
                "plan-revision-store",
                writer_module="ipfs_accelerate_py/agent_supervisor/task_sources/plan_revision_store.py",
                path_template=(
                    "{root}/index.json + active.json + prior_active.json + "
                    "events.jsonl + supersessions.jsonl + .plan-revision-store.lock"
                ),
                media_type=MediaType.JSON,
                classification=SinkClassification.AUTHORITY,
                destination_domain=DestinationDomain.OBJECTIVES_PLANS_TASKS,
                retirement_stage=RetirementStage.DEFAULT_CUTOVER,
                reuse_candidate="plan_revisions + planning_decisions tables with CAS",
                atomicity_model=AtomicityModel.CROSS_FILE_NON_ATOMIC,
                discovery_basenames=(
                    "events.jsonl",
                    "supersessions.jsonl",
                    "index.json",
                    "active.json",
                    "prior_active.json",
                ),
                cross_file_atomicity_gap=(
                    "Active projection, index, and event log are separate files; "
                    "recovery rebuilds active from events but crash windows exist"
                ),
            ),
            # --- Event logs ---
            _sink(
                "runtime-event-log-jsonl",
                writer_module="ipfs_accelerate_py/agent_supervisor/runtime/event_log.py",
                path_template="{state}/events.jsonl (+ rotated archives)",
                media_type=MediaType.JSONL,
                classification=SinkClassification.AUTHORITY,
                destination_domain=DestinationDomain.EVENTS_LOGS_METRICS,
                retirement_stage=RetirementStage.DEFAULT_CUTOVER,
                reuse_candidate="domain_events append stream with monotonic sequence",
                atomicity_model=AtomicityModel.APPEND_ONLY,
                discovery_basenames=("events.jsonl",),
                discovery_modules=(
                    "ipfs_accelerate_py/agent_supervisor/runtime/event_log.py",
                    "ipfs_accelerate_py/agent_supervisor/merge/leased_lane.py",
                    "ipfs_accelerate_py/agent_supervisor/merge/merge_resolver.py",
                ),
            ),
            # --- Persistent queue / daemon status ---
            _sink(
                "persistent-task-queue-json",
                writer_module="ipfs_accelerate_py/agent_supervisor/task_sources/persistent_task_queue.py",
                path_template="{daemon_state}/task_queue.json",
                media_type=MediaType.JSON,
                classification=SinkClassification.AUTHORITY,
                destination_domain=DestinationDomain.EXECUTION_LIFECYCLE,
                retirement_stage=RetirementStage.DEFAULT_CUTOVER,
                reuse_candidate="task_claims + selection penalty columns",
                atomicity_model=AtomicityModel.SINGLE_FILE_ATOMIC,
                discovery_basenames=("task_queue.json",),
                discovery_modules=(
                    "ipfs_accelerate_py/agent_supervisor/task_sources/persistent_task_queue.py",
                    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
                    "ipfs_accelerate_py/agent_supervisor/objectives/bundle_supervisor.py",
                ),
            ),
            _sink(
                "daemon-status-json",
                writer_module="ipfs_accelerate_py/agent_supervisor/todo_daemon/status.py",
                path_template="{state}/*status.json / *.progress.json",
                media_type=MediaType.JSON,
                classification=SinkClassification.OS_BOOTSTRAP,
                destination_domain=DestinationDomain.EXECUTION_LIFECYCLE,
                retirement_stage=RetirementStage.DEFAULT_CUTOVER,
                reuse_candidate="heartbeats + attempt_phases projections",
                atomicity_model=AtomicityModel.SINGLE_FILE_ATOMIC,
                discovery_basenames=(
                    "status.json",
                    "progress.json",
                    "todo.status.json",
                    "todo.progress.json",
                    "current_status.json",
                ),
                discovery_modules=(
                    "ipfs_accelerate_py/agent_supervisor/todo_daemon/status.py",
                    "ipfs_accelerate_py/agent_supervisor/todo_daemon/runner.py",
                    "ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_loop.py",
                    "ipfs_accelerate_py/agent_supervisor/todo_daemon/legal_parser.py",
                    "ipfs_accelerate_py/agent_supervisor/todo_daemon/logic_port.py",
                    "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
                    "ipfs_accelerate_py/agent_supervisor/objectives/bundle_supervisor.py",
                ),
                notes="Mirrors lease/session state for operators; not claim authority.",
            ),
            _sink(
                "daemon-pid-files",
                writer_module="ipfs_accelerate_py/agent_supervisor/todo_daemon/wrapper.py",
                path_template="{state}/*.pid",
                media_type=MediaType.PID,
                classification=SinkClassification.OS_BOOTSTRAP,
                destination_domain=DestinationDomain.EXECUTION_LIFECYCLE,
                retirement_stage=RetirementStage.DEFAULT_CUTOVER,
                reuse_candidate="process-birth identity on daemon_sessions",
                atomicity_model=AtomicityModel.OS_PROCESS_HANDLE,
                discovery_basenames=(
                    "todo.supervisor.pid",
                    "todo.child.pid",
                    "configured-board-master.pid",
                    "bundle_scheduler.pid",
                    "bundle_scheduler_worker.pid",
                ),
                discovery_modules=(
                    "ipfs_accelerate_py/agent_supervisor/todo_daemon/wrapper.py",
                    "ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_loop.py",
                    "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
                    "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
                    "ipfs_accelerate_py/agent_supervisor/objectives/bundle_supervisor.py",
                ),
                notes="PID files never grant lease or completion authority.",
            ),
            # --- Worktrees / merge train / locks ---
            _sink(
                "worktree-lifecycle-records",
                writer_module="ipfs_accelerate_py/agent_supervisor/merge/worktree_lifecycle.py",
                path_template="{state}/worktrees/*.json",
                media_type=MediaType.JSON,
                classification=SinkClassification.AUTHORITY,
                destination_domain=DestinationDomain.REPOSITORY_WORKTREE,
                retirement_stage=RetirementStage.DEFAULT_CUTOVER,
                reuse_candidate="worktrees + leases tables with process-birth fencing",
                atomicity_model=AtomicityModel.SINGLE_FILE_ATOMIC,
                discovery_basenames=(),
            ),
            _sink(
                "merge-train-state-dir",
                writer_module="ipfs_accelerate_py/agent_supervisor/merge/merge_train.py",
                path_template="{queue}/train/{receipts,worktrees,gate-cache,consumer.lock}",
                media_type=MediaType.DIRECTORY,
                classification=SinkClassification.AUTHORITY,
                destination_domain=DestinationDomain.REPOSITORY_WORKTREE,
                retirement_stage=RetirementStage.DEFAULT_CUTOVER,
                reuse_candidate="merge_attempts + validation_runs + worktree rows",
                atomicity_model=AtomicityModel.CROSS_FILE_NON_ATOMIC,
                discovery_basenames=("consumer.lock",),
                cross_file_atomicity_gap=(
                    "Train receipts, consumer lock, gate cache, and merge queue DB "
                    "are not one atomic unit"
                ),
            ),
            _sink(
                "checkout-mutation-lock",
                writer_module="ipfs_accelerate_py/agent_supervisor/merge/checkout_lock.py",
                path_template="{repo}/implementation-main-merge.lock (+ related locks)",
                media_type=MediaType.LOCK,
                classification=SinkClassification.OS_BOOTSTRAP,
                destination_domain=DestinationDomain.REPOSITORY_WORKTREE,
                retirement_stage=RetirementStage.DEFAULT_CUTOVER,
                reuse_candidate="path_claims + maintenance_leases",
                atomicity_model=AtomicityModel.OS_PROCESS_HANDLE,
                discovery_basenames=(
                    "implementation-main-merge.lock",
                    "implementation-protected-path-maintenance.lock",
                    "implementation-protected-path-crash-fence-recon.lock",
                ),
            ),
            _sink(
                "merge-checkpoint-json",
                writer_module="ipfs_accelerate_py/agent_supervisor/merge/merge_checkpoint.py",
                path_template="{state}/merge_checkpoint.json",
                media_type=MediaType.JSON,
                classification=SinkClassification.AUTHORITY,
                destination_domain=DestinationDomain.REPOSITORY_WORKTREE,
                retirement_stage=RetirementStage.DEFAULT_CUTOVER,
                reuse_candidate="merge_attempts checkpoint columns",
                atomicity_model=AtomicityModel.SINGLE_FILE_ATOMIC,
                discovery_basenames=(),
            ),
            # --- Recovery / rescue ---
            _sink(
                "recovery-receipts-and-locks",
                writer_module="ipfs_accelerate_py/agent_supervisor/rescue/supervisor_recovery.py",
                path_template="{state}/{receipts,incidents,quarantine,*.lock}",
                media_type=MediaType.DIRECTORY,
                classification=SinkClassification.EMERGENCY_DIAGNOSTIC,
                destination_domain=DestinationDomain.EXECUTION_LIFECYCLE,
                retirement_stage=RetirementStage.DEFAULT_CUTOVER,
                reuse_candidate="recovery_actions + quarantine tables",
                atomicity_model=AtomicityModel.CROSS_FILE_NON_ATOMIC,
                discovery_basenames=(),
                cross_file_atomicity_gap=(
                    "Incident locks, receipts, and quarantine directories update "
                    "independently of lease/task authority files"
                ),
                notes="Emergency path; must not become a second authority plane.",
            ),
            # --- Bundle supervisor / scheduler metrics ---
            _sink(
                "bundle-lane-manifest",
                writer_module="ipfs_accelerate_py/agent_supervisor/objectives/bundle_supervisor.py",
                path_template="{state_root}/bundle_lanes.json + scheduler_metrics.json",
                media_type=MediaType.JSON,
                classification=SinkClassification.AUTHORITY,
                destination_domain=DestinationDomain.EXECUTION_LIFECYCLE,
                retirement_stage=RetirementStage.DEFAULT_CUTOVER,
                reuse_candidate="daemon_instances + scheduler metrics tables",
                atomicity_model=AtomicityModel.CROSS_FILE_NON_ATOMIC,
                discovery_basenames=(
                    "bundle_lanes.json",
                    "scheduler_metrics.json",
                    "scheduler_decision_metrics.json",
                ),
                cross_file_atomicity_gap=(
                    "Lane manifest, metrics JSON, coordination.duckdb, and PID files "
                    "are updated without a shared transaction"
                ),
            ),
            # --- Dataset / audit jsonl ---
            _sink(
                "dataset-store-jsonl",
                writer_module="ipfs_accelerate_py/agent_supervisor/task_sources/dataset_store.py",
                path_template="{root}/{dataset_id}.jsonl (+ optional parquet)",
                media_type=MediaType.JSONL,
                classification=SinkClassification.IMMUTABLE_EVIDENCE,
                destination_domain=DestinationDomain.ARTIFACTS_PROOFS,
                retirement_stage=RetirementStage.DUAL_OBSERVATION,
                reuse_candidate="artifact metadata + content digests",
                atomicity_model=AtomicityModel.SINGLE_FILE_ATOMIC,
                discovery_basenames=(),
            ),
            # --- Analysis caches ---
            _sink(
                "analysis-program-cache",
                writer_module="ipfs_accelerate_py/agent_supervisor/analysis/program_analysis_cache.py",
                path_template="{cache_root}/program-analysis/**",
                media_type=MediaType.CACHE_DIR,
                classification=SinkClassification.CACHE,
                destination_domain=DestinationDomain.CODE_INTELLIGENCE,
                retirement_stage=RetirementStage.LEGACY_RETIREMENT,
                reuse_candidate="source_snapshots + parse_runs keyed by tree id",
                atomicity_model=AtomicityModel.BEST_EFFORT_MIRROR,
                discovery_basenames=(),
                notes="Accelerator only; never lease or completion authority.",
            ),
            _sink(
                "scheduler-config-json",
                writer_module="ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
                path_template="config/*_scheduler.json",
                media_type=MediaType.JSON,
                classification=SinkClassification.STATIC_INPUT,
                destination_domain=DestinationDomain.SCHEMA_DEPLOYMENT,
                retirement_stage=RetirementStage.BOOTSTRAP,
                reuse_candidate="launch policy rows imported once with provenance",
                atomicity_model=AtomicityModel.NOT_APPLICABLE,
                discovery_basenames=(),
                notes="Operator-sealed static input; not rewritten by daemons.",
            ),
            # --- Control plane audit / lifecycle ---
            _sink(
                "control-audit-jsonl",
                writer_module="ipfs_accelerate_py/agent_supervisor/control/control_plane.py",
                path_template=(
                    "{state}/control-audit.jsonl + control-idempotency.jsonl + "
                    "control-transactions.jsonl + supervisor-lifecycle-events.jsonl"
                ),
                media_type=MediaType.JSONL,
                classification=SinkClassification.AUTHORITY,
                destination_domain=DestinationDomain.EVENTS_LOGS_METRICS,
                retirement_stage=RetirementStage.DEFAULT_CUTOVER,
                reuse_candidate="domain_events + idempotency_records streams",
                atomicity_model=AtomicityModel.APPEND_ONLY,
                discovery_basenames=(
                    "control-audit.jsonl",
                    "control-idempotency.jsonl",
                    "control-transactions.jsonl",
                    "supervisor-lifecycle-events.jsonl",
                ),
                cross_file_atomicity_gap=(
                    "Control audit, idempotency, transaction, and lifecycle JSONL "
                    "files are independent append streams"
                ),
            ),
            # --- Planning stores ---
            _sink(
                "proof-carrying-workflow-duckdb",
                writer_module="ipfs_accelerate_py/agent_supervisor/planning/proof_carrying_planner.py",
                path_template="{store}/proof_carrying_workflow.duckdb (+ JSON twin)",
                media_type=MediaType.DUCKDB,
                classification=SinkClassification.AUTHORITY,
                destination_domain=DestinationDomain.OBJECTIVES_PLANS_TASKS,
                retirement_stage=RetirementStage.DEFAULT_CUTOVER,
                reuse_candidate="plan_revisions + ArtifactStore dual JSON/DuckDB pattern",
                atomicity_model=AtomicityModel.CROSS_FILE_NON_ATOMIC,
                discovery_basenames=("proof_carrying_workflow.duckdb",),
                discovery_modules=(
                    "ipfs_accelerate_py/agent_supervisor/planning/proof_carrying_planner.py",
                    "ipfs_accelerate_py/agent_supervisor/planning/formal_plan_compiler.py",
                    "ipfs_accelerate_py/agent_supervisor/planning/formal_plan_conformance.py",
                ),
                is_direct_duckdb_writer=True,
                cross_file_atomicity_gap=(
                    "Proof-carrying planner JSON and DuckDB twin are dual-written"
                ),
            ),
            # --- Prompt workflow materialization ---
            _sink(
                "prompt-workflow-duckdb-materialization",
                writer_module="ipfs_accelerate_py/agent_supervisor/prompt/prompt_workflow.py",
                path_template="{output_policy.duckdb_path} via DuckDBTaskSource",
                media_type=MediaType.DUCKDB,
                classification=SinkClassification.AUTHORITY,
                destination_domain=DestinationDomain.OBJECTIVES_PLANS_TASKS,
                retirement_stage=RetirementStage.FOUNDATION,
                reuse_candidate="DuckDBTaskSource materializer already used by prompt workflow",
                atomicity_model=AtomicityModel.FLOCK_PLUS_TRANSACTION,
                discovery_basenames=(),
                discovery_modules=(
                    "ipfs_accelerate_py/agent_supervisor/prompt/prompt_workflow.py",
                    "ipfs_accelerate_py/agent_supervisor/prompt/prompt_directory_scanner.py",
                ),
                is_direct_duckdb_writer=True,
                notes="Prompt workflow reuses DuckDBTaskSource; scanner must classify path markers.",
            ),
            # --- Validation / rollout OS bootstrap ---
            _sink(
                "validation-rollout-pid-status",
                writer_module="ipfs_accelerate_py/agent_supervisor/validation/logic_repair_rollout.py",
                path_template="{runtime}/master.pid + *_supervisor_status.json",
                media_type=MediaType.PID,
                classification=SinkClassification.OS_BOOTSTRAP,
                destination_domain=DestinationDomain.EXECUTION_LIFECYCLE,
                retirement_stage=RetirementStage.DEFAULT_CUTOVER,
                reuse_candidate="daemon_sessions process-birth identity + heartbeats",
                atomicity_model=AtomicityModel.OS_PROCESS_HANDLE,
                discovery_basenames=("master.pid",),
                discovery_modules=(
                    "ipfs_accelerate_py/agent_supervisor/validation/logic_repair_rollout.py",
                    "ipfs_accelerate_py/agent_supervisor/validation/change_propagation_rollout.py",
                    "ipfs_accelerate_py/agent_supervisor/validation/proof_test_reuse_objective_contracts.py",
                ),
            ),
            # --- Entrypoint / analysis / proof OS bootstrap locks ---
            _sink(
                "run-registry-lock",
                writer_module="ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py",
                path_template="{registry_root}/.run-registry.lock (+ registry JSON rows)",
                media_type=MediaType.LOCK,
                classification=SinkClassification.OS_BOOTSTRAP,
                destination_domain=DestinationDomain.EXECUTION_LIFECYCLE,
                retirement_stage=RetirementStage.DEFAULT_CUTOVER,
                reuse_candidate="client_sessions + resource_claims under Quack",
                atomicity_model=AtomicityModel.OS_PROCESS_HANDLE,
                discovery_basenames=(".run-registry.lock",),
                discovery_modules=(
                    "ipfs_accelerate_py/agent_supervisor/entrypoints/run_registry.py",
                ),
                notes="Serializes run-registry mutations; not claim authority.",
            ),
            _sink(
                "analysis-cache-lock",
                writer_module="ipfs_accelerate_py/agent_supervisor/analysis/analysis_cache.py",
                path_template="{cache}/.analysis-cache.lock",
                media_type=MediaType.LOCK,
                classification=SinkClassification.CACHE,
                destination_domain=DestinationDomain.CODE_INTELLIGENCE,
                retirement_stage=RetirementStage.LEGACY_RETIREMENT,
                reuse_candidate="parse_runs + source_snapshots with content digests",
                atomicity_model=AtomicityModel.OS_PROCESS_HANDLE,
                discovery_basenames=(".analysis-cache.lock",),
            ),
            _sink(
                "repository-index-lock",
                writer_module="ipfs_accelerate_py/agent_supervisor/analysis/repository_indexer.py",
                path_template="{index_root}/.repository-index.lock",
                media_type=MediaType.LOCK,
                classification=SinkClassification.CACHE,
                destination_domain=DestinationDomain.CODE_INTELLIGENCE,
                retirement_stage=RetirementStage.LEGACY_RETIREMENT,
                reuse_candidate="repository_revisions + source_files index tables",
                atomicity_model=AtomicityModel.OS_PROCESS_HANDLE,
                discovery_basenames=(".repository-index.lock",),
            ),
            _sink(
                "proof-certificate-locks",
                writer_module="ipfs_accelerate_py/agent_supervisor/proof/test_certificate_store.py",
                path_template="{root}/locks/{cas,fence}.lock (+ token locks)",
                media_type=MediaType.LOCK,
                classification=SinkClassification.OS_BOOTSTRAP,
                destination_domain=DestinationDomain.ARTIFACTS_PROOFS,
                retirement_stage=RetirementStage.DEFAULT_CUTOVER,
                reuse_candidate="artifact retention leases + content-addressed receipts",
                atomicity_model=AtomicityModel.OS_PROCESS_HANDLE,
                discovery_basenames=("cas.lock", "fence.lock"),
                discovery_modules=(
                    "ipfs_accelerate_py/agent_supervisor/proof/test_certificate_store.py",
                    "ipfs_accelerate_py/agent_supervisor/proof/test_candidate_context_store.py",
                    "ipfs_accelerate_py/agent_supervisor/proof/contract_findings.py",
                ),
            ),
            _sink(
                "integration-tool-install-locks",
                writer_module="ipfs_accelerate_py/agent_supervisor/integrations/contract_repair_dependencies.py",
                path_template="{managed_root}/*dependencies*.lock / agent-llm-resolver.lock",
                media_type=MediaType.LOCK,
                classification=SinkClassification.OS_BOOTSTRAP,
                destination_domain=DestinationDomain.SCHEMA_DEPLOYMENT,
                retirement_stage=RetirementStage.LEGACY_RETIREMENT,
                reuse_candidate="capability_snapshots + digest-bound toolchain installs",
                atomicity_model=AtomicityModel.OS_PROCESS_HANDLE,
                discovery_basenames=(
                    "agent-llm-resolver.lock",
                    "ipfs_accelerate_py-contract-repair-python-dependencies.lock",
                ),
                discovery_modules=(
                    "ipfs_accelerate_py/agent_supervisor/integrations/contract_repair_dependencies.py",
                    "ipfs_accelerate_py/agent_supervisor/integrations/llm_merge_resolver_fallback.py",
                ),
                notes="Tool-install single-flight only; never orchestration authority.",
            ),
            _sink(
                "control-plane-migration-lock",
                writer_module="ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_migrations.py",
                path_template=".{database}.migration.lock (+ control.duckdb bookkeeping)",
                media_type=MediaType.LOCK,
                classification=SinkClassification.OS_BOOTSTRAP,
                destination_domain=DestinationDomain.SCHEMA_DEPLOYMENT,
                retirement_stage=RetirementStage.FOUNDATION,
                reuse_candidate=(
                    "Migration ownership rows + process-birth fencing; lock is "
                    "bootstrap only around the state-owner"
                ),
                atomicity_model=AtomicityModel.OS_PROCESS_HANDLE,
                discovery_basenames=(),
                notes="Serializes checksum-bound migrations; not claim authority.",
            ),
            _sink(
                "accepted-work-ledger-jsonl",
                writer_module="ipfs_accelerate_py/agent_supervisor/todo_daemon/artifacts.py",
                path_template="{daemon_state}/accepted-work.jsonl (+ accepted_changes.jsonl)",
                media_type=MediaType.JSONL,
                classification=SinkClassification.IMMUTABLE_EVIDENCE,
                destination_domain=DestinationDomain.EVENTS_LOGS_METRICS,
                retirement_stage=RetirementStage.DEFAULT_CUTOVER,
                reuse_candidate="completion_receipts + domain_events append stream",
                atomicity_model=AtomicityModel.APPEND_ONLY,
                discovery_basenames=(
                    "accepted-work.jsonl",
                    "accepted_changes.jsonl",
                ),
                discovery_modules=(
                    "ipfs_accelerate_py/agent_supervisor/todo_daemon/artifacts.py",
                    "ipfs_accelerate_py/agent_supervisor/todo_daemon/legal_parser_daemon.py",
                    "ipfs_accelerate_py/agent_supervisor/todo_daemon/app.py",
                ),
            ),
            _sink(
                "goal-tactician-lifecycle-jsonl",
                writer_module="ipfs_accelerate_py/agent_supervisor/proof/goal_tactician_lifecycle.py",
                path_template=(
                    "{root}/goal_tactician_lifecycle.journal.jsonl + "
                    "leanstral-goal-lifecycle.audit.jsonl"
                ),
                media_type=MediaType.JSONL,
                classification=SinkClassification.IMMUTABLE_EVIDENCE,
                destination_domain=DestinationDomain.ARTIFACTS_PROOFS,
                retirement_stage=RetirementStage.DUAL_OBSERVATION,
                reuse_candidate="domain_events + proof_attempts audit streams",
                atomicity_model=AtomicityModel.APPEND_ONLY,
                discovery_basenames=(
                    "goal_tactician_lifecycle.journal.jsonl",
                    "leanstral-goal-lifecycle.audit.jsonl",
                ),
                discovery_modules=(
                    "ipfs_accelerate_py/agent_supervisor/proof/goal_tactician_lifecycle.py",
                    "ipfs_accelerate_py/agent_supervisor/proof/leanstral_goal_lifecycle.py",
                ),
            ),
        ),
        key=lambda sink: sink.sink_id,
    )
)


def repo_root_from(start: Path | None = None) -> Path:
    """Locate the repository root that contains the agent_supervisor package."""

    here = (start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        package = candidate / AGENT_SUPERVISOR_PACKAGE
        if package.is_dir() and (candidate / "docs" / "architecture").is_dir():
            return candidate
    raise FileNotFoundError(
        f"could not locate repository root from {here}"
    )


def known_sinks() -> tuple[StateSink, ...]:
    """Return the curated catalog in deterministic sink_id order."""

    return KNOWN_STATE_SINKS


def _normalize_module(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _is_non_sink_basename(name: str) -> bool:
    """Return True when *name* is a known non-orchestration basename."""

    lowered = name.casefold()
    if lowered in _NON_SINK_BASENAME_ALLOWLIST:
        return True
    # Also accept exact (case-sensitive) allowlist entries for non-ASCII safety.
    return name in _NON_SINK_BASENAME_ALLOWLIST


def _basename_of_literal(value: str) -> str | None:
    text = value.strip().replace("\\", "/")
    if not text or "\x00" in text:
        return None
    if len(text) > 240:
        return None
    # Ignore URIs, schemas, and prose.
    if "://" in text or "@" in text or "\n" in text:
        return None
    if " " in text and not text.endswith((".md", ".json", ".jsonl", ".duckdb")):
        return None
    name = Path(text).name
    if not name or name in {".", ".."}:
        return None
    if _is_non_sink_basename(name):
        return None
    # Avoid treating generic English or identifiers as path sinks.
    if name in {"lock", "locked", "blocking", "blocked", "db", "pid"}:
        return None
    # Glob-style basenames are only sinks when the suffix is a known mutable
    # media family. Patterns like ``*.json`` / ``*.py`` / ``*.pem`` are source
    # scan filters and must not fail the inventory closed.
    if "*" in name:
        lowered = name.casefold()
        if lowered in _SINK_GLOB_BASENAMES:
            return name
        if lowered.endswith(
            (
                ".duckdb",
                ".sqlite",
                ".sqlite3",
                ".jsonl",
                ".pid",
                ".lock",
                ".objectives.md",
                ".todo.md",
                "status.json",
                "progress.json",
            )
        ):
            return name
        return None
    if _SINK_BASENAME_RE.fullmatch(name):
        return name
    if name.endswith((".objectives.md", ".todo.md")):
        return name
    return None

_BARE_SINK_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        ".duckdb",
        ".sqlite",
        ".sqlite3",
        ".jsonl",
        ".pid",
        ".lock",
    }
)

# Process-local cache so module-scoped fixtures and repeated inventory builds
# do not re-scan the package tree.
_DISCOVERY_CACHE: dict[str, tuple[DiscoveredMarker, ...]] = {}


def _decode_string_token(raw: str) -> str | None:
    """Decode a ``tokenize.STRING`` token into a Python ``str`` value."""

    if not raw:
        return None
    prefix_end = 0
    for index, char in enumerate(raw):
        if char in "'\"":
            prefix_end = index
            break
        if char not in "rRbBuUfF":
            return None
    else:
        return None
    prefix = raw[:prefix_end].lower()
    body = raw[prefix_end:]
    # Expression-free f-strings may still be emitted as STRING tokens; strip f.
    if "f" in prefix:
        prefix = prefix.replace("f", "")
        raw = prefix + body
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, str) else None


def _iter_string_literals_via_tokenize(source: str) -> Iterable[tuple[int, str]]:
    """Yield ``(lineno, literal)`` via the tokenize stream (Python 3.12+).

    Streaming tokenization is much cheaper than ``ast.parse`` on multi-10k-line
    supervisor modules while still capturing ordinary strings and f-string
    constant fragments (``FSTRING_MIDDLE``).
    """

    readline = io.StringIO(source).readline
    fstring_middle = getattr(tokenize, "FSTRING_MIDDLE", None)
    for tok in tokenize.generate_tokens(readline):
        if tok.type == tokenize.STRING:
            value = _decode_string_token(tok.string)
            if value is not None:
                yield int(tok.start[0] or 0), value
            continue
        # f-string constant pieces: f"{name}.duckdb" → middle ".duckdb"
        if fstring_middle is not None and tok.type == fstring_middle:
            piece = tok.string
            if piece:
                yield int(tok.start[0] or 0), piece


def _iter_string_literals_via_ast(source: str) -> Iterable[tuple[int, str]]:
    """AST fallback when tokenization fails on a module."""

    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield int(getattr(node, "lineno", 0) or 0), node.value
        elif isinstance(node, ast.JoinedStr):
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    yield int(getattr(node, "lineno", 0) or 0), part.value


def _iter_string_literals_from_source(source: str) -> Iterable[tuple[int, str]]:
    """Yield string literals from source, preferring tokenize over AST."""

    try:
        yield from _iter_string_literals_via_tokenize(source)
        return
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    yield from _iter_string_literals_via_ast(source)


def _iter_string_literals(tree: ast.AST) -> Iterable[tuple[int, str]]:
    """AST helper used by callers that already parsed a module."""

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield int(getattr(node, "lineno", 0) or 0), node.value
        elif isinstance(node, ast.JoinedStr):
            # f-strings: collect constant parts that look like path suffixes.
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    yield int(getattr(node, "lineno", 0) or 0), part.value


def _marker_basename(literal: str) -> str | None:
    basename = _basename_of_literal(literal)
    if basename is not None:
        return basename
    stripped = literal.strip()
    if stripped in _BARE_SINK_SUFFIXES:
        return stripped
    return None


def discover_markers(root: Path) -> tuple[DiscoveredMarker, ...]:
    """Discover mutable path markers under the agent_supervisor package."""

    package = root / AGENT_SUPERVISOR_PACKAGE
    if not package.is_dir():
        raise FileNotFoundError(f"missing package tree: {package}")

    cache_key = str(package.resolve())
    cached = _DISCOVERY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    discoveries: list[DiscoveredMarker] = []
    for path in sorted(package.rglob("*.py")):
        if any(part in _SCAN_SKIP_PARTS for part in path.parts):
            continue
        try:
            size = path.stat().st_size
            if size > 32 * 1024 * 1024:
                continue
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        module = _normalize_module(path, root)
        seen_local: set[tuple[str, int, str]] = set()
        for line, literal in _iter_string_literals_from_source(source):
            basename = _marker_basename(literal)
            if basename is None:
                continue
            key = (module, line, basename)
            if key in seen_local:
                continue
            seen_local.add(key)
            discoveries.append(
                DiscoveredMarker(
                    module=module,
                    basename=basename,
                    line=line,
                    literal=literal if len(literal) <= 240 else literal[:237] + "...",
                )
            )
    discoveries.sort(key=lambda item: (item.module, item.line, item.basename, item.literal))
    result = tuple(discoveries)
    _DISCOVERY_CACHE[cache_key] = result
    return result


def _basename_matches(pattern: str, basename: str) -> bool:
    if pattern == basename:
        return True
    if pattern.startswith("*.") and basename.endswith(pattern[1:]):
        return True
    if pattern.startswith("*") and basename.endswith(pattern[1:]):
        return True
    # Generic status/progress family.
    if pattern == "status.json" and basename.endswith("status.json"):
        return True
    if pattern == "progress.json" and basename.endswith("progress.json"):
        return True
    if pattern == "events.jsonl" and basename.endswith("events.jsonl"):
        return True
    if pattern.endswith(".pid") and basename.endswith(".pid") and pattern == basename:
        return True
    if pattern.endswith(".lock") and basename.endswith(".lock") and pattern == basename:
        return True
    return False


def _media_for_basename(basename: str) -> MediaType | None:
    name = basename.lower()
    # Strip a single leading glob star so ``*.duckdb`` maps like ``.duckdb``.
    if name.startswith("*") and len(name) > 1:
        name = name[1:]
    if name in {".duckdb"} or name.endswith(".duckdb"):
        return MediaType.DUCKDB
    if name in {".sqlite", ".sqlite3"} or name.endswith((".sqlite", ".sqlite3")):
        return MediaType.SQLITE
    if name in {".jsonl"} or name.endswith(".jsonl"):
        return MediaType.JSONL
    if name in {".pid"} or name.endswith(".pid"):
        return MediaType.PID
    if name in {".lock"} or name.endswith(".lock"):
        return MediaType.LOCK
    if name.endswith("status.json") or name.endswith("progress.json"):
        return MediaType.JSON
    if name in {
        "index.json",
        "active.json",
        "prior_active.json",
        "bundle_lanes.json",
        "scheduler_metrics.json",
        "scheduler_decision_metrics.json",
        "task_queue.json",
    }:
        return MediaType.JSON
    if name.endswith(".objectives.md") or name.endswith(".todo.md"):
        return MediaType.MARKDOWN
    return None


def _module_under(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix.rstrip("/") + "/")


# Package prefixes known to host mutable supervisor state sinks. Every known
# subpackage is covered for the full mutable media set so family classification
# stays complete; modules outside these prefixes fail closed (CI gate).
_PACKAGE_MEDIA_COVERAGE: Final[tuple[tuple[str, frozenset[MediaType]], ...]] = (
    ("ipfs_accelerate_py/agent_supervisor/task_sources", _ALL_MUTABLE_MEDIA),
    ("ipfs_accelerate_py/agent_supervisor/merge", _ALL_MUTABLE_MEDIA),
    ("ipfs_accelerate_py/agent_supervisor/todo_daemon", _ALL_MUTABLE_MEDIA),
    ("ipfs_accelerate_py/agent_supervisor/runtime", _ALL_MUTABLE_MEDIA),
    ("ipfs_accelerate_py/agent_supervisor/objectives", _ALL_MUTABLE_MEDIA),
    ("ipfs_accelerate_py/agent_supervisor/proof", _ALL_MUTABLE_MEDIA),
    ("ipfs_accelerate_py/agent_supervisor/rescue", _ALL_MUTABLE_MEDIA),
    ("ipfs_accelerate_py/agent_supervisor/analysis", _ALL_MUTABLE_MEDIA),
    ("ipfs_accelerate_py/agent_supervisor/entrypoints", _ALL_MUTABLE_MEDIA),
    ("ipfs_accelerate_py/agent_supervisor/core", _ALL_MUTABLE_MEDIA),
    ("ipfs_accelerate_py/agent_supervisor/contract_analysis", _ALL_MUTABLE_MEDIA),
    ("ipfs_accelerate_py/agent_supervisor/context", _ALL_MUTABLE_MEDIA),
    ("ipfs_accelerate_py/agent_supervisor/self_improvement", _ALL_MUTABLE_MEDIA),
    ("ipfs_accelerate_py/agent_supervisor/control", _ALL_MUTABLE_MEDIA),
    ("ipfs_accelerate_py/agent_supervisor/planning", _ALL_MUTABLE_MEDIA),
    ("ipfs_accelerate_py/agent_supervisor/prompt", _ALL_MUTABLE_MEDIA),
    ("ipfs_accelerate_py/agent_supervisor/validation", _ALL_MUTABLE_MEDIA),
    ("ipfs_accelerate_py/agent_supervisor/integrations", _ALL_MUTABLE_MEDIA),
)


def _package_covers(module: str, media: MediaType) -> bool:
    for prefix, media_set in _PACKAGE_MEDIA_COVERAGE:
        if _module_under(module, prefix) and media in media_set:
            return True
    return False


def _family_representative(
    media: MediaType,
    sinks: Sequence[StateSink],
) -> StateSink | None:
    """Pick a stable catalog sink that owns a media family."""

    preferred = {
        MediaType.DUCKDB: (
            "duckdb-task-source",
            "lease-coordination-duckdb",
            "duckdb-state-primitives",
            "proof-carrying-workflow-duckdb",
            "prompt-workflow-duckdb-materialization",
        ),
        MediaType.SQLITE: (
            "doctor-proof-cache-sqlite",
            "analysis-singleflight-sqlite",
        ),
        MediaType.JSONL: (
            "runtime-event-log-jsonl",
            "control-audit-jsonl",
            "taskboard-store-events",
            "dataset-store-jsonl",
        ),
        MediaType.PID: (
            "daemon-pid-files",
            "validation-rollout-pid-status",
        ),
        MediaType.LOCK: (
            "checkout-mutation-lock",
            "taskboard-materialization-lock",
            "run-registry-lock",
            "analysis-cache-lock",
            "proof-certificate-locks",
            "integration-tool-install-locks",
        ),
        MediaType.JSON: (
            "daemon-status-json",
            "plan-revision-store",
            "persistent-task-queue-json",
        ),
        MediaType.MARKDOWN: (
            "taskboard-markdown",
            "objective-heap-markdown",
        ),
        MediaType.ARTIFACT: ("artifact-store-json-duckdb",),
        MediaType.DIRECTORY: (
            "merge-train-state-dir",
            "recovery-receipts-and-locks",
        ),
        MediaType.CACHE_DIR: ("analysis-program-cache",),
    }
    by_id = {sink.sink_id: sink for sink in sinks}
    for sink_id in preferred.get(media, ()):
        if sink_id in by_id:
            return by_id[sink_id]
    for sink in sinks:
        if sink.media_type == media and not sink.is_git_source_bytes:
            return sink
    return None


def classify_discovery(
    marker: DiscoveredMarker,
    sinks: Sequence[StateSink],
) -> StateSink | None:
    """Return the catalog entry that covers a discovery, if any.

    Matching order:
    1. Explicit ``discovery_basenames`` patterns on a catalog sink whose
       module list includes the discovery module (or its package directory).
    2. Package media-family coverage for known agent_supervisor subpackages,
       linked to a representative catalog sink.
    """

    # 1) Explicit basename patterns.
    for sink in sinks:
        if sink.is_git_source_bytes:
            continue
        modules = sink.discovery_modules or (sink.writer_module,)
        module_ok = False
        for module in modules:
            parent = str(Path(module).parent.as_posix())
            if marker.module == module or _module_under(marker.module, parent):
                module_ok = True
                break
        if not module_ok:
            continue
        for pattern in sink.discovery_basenames:
            if _basename_matches(pattern, marker.basename):
                return sink
        if marker.basename in {
            ".duckdb",
            ".sqlite",
            ".sqlite3",
            ".jsonl",
            ".pid",
            ".lock",
        }:
            media = _media_for_basename(marker.basename)
            if media is not None and sink.media_type in {
                media,
                MediaType.ARTIFACT,
            }:
                return sink

    # 2) Family coverage for known packages (still fail closed outside them).
    media = _media_for_basename(marker.basename)
    if media is None:
        return None
    if not _package_covers(marker.module, media):
        return None
    return _family_representative(media, sinks)


def build_inventory(
    root: Path | None = None,
    *,
    sinks: Sequence[StateSink] | None = None,
    discoveries: Sequence[DiscoveredMarker] | None = None,
    fail_on_unclassified: bool = True,
) -> InventoryReport:
    """Build a deterministic inventory report for the repository tree."""

    repo = repo_root_from(root) if root is not None else repo_root_from()
    catalog = tuple(sinks) if sinks is not None else known_sinks()
    # Enforce catalog ordering / unique ids.
    ids = [sink.sink_id for sink in catalog]
    errors: list[str] = []
    if ids != sorted(ids):
        errors.append("catalog sink_id order is not sorted")
    if len(ids) != len(set(ids)):
        errors.append("catalog contains duplicate sink_id values")

    found = (
        tuple(discoveries)
        if discoveries is not None
        else discover_markers(repo)
    )
    unclassified: list[DiscoveredMarker] = []
    for marker in found:
        if classify_discovery(marker, catalog) is None:
            unclassified.append(marker)

    duckdb_writers = tuple(
        sink.sink_id for sink in catalog if sink.is_direct_duckdb_writer
    )
    gaps = tuple(
        f"{sink.sink_id}: {sink.cross_file_atomicity_gap}"
        for sink in catalog
        if sink.cross_file_atomicity_gap
    )
    reuse = tuple(
        f"{sink.sink_id}: {sink.reuse_candidate}"
        for sink in catalog
        if sink.reuse_candidate
    )
    git_bits = tuple(
        f"{sink.sink_id}: {sink.notes or sink.path_template}"
        for sink in catalog
        if sink.is_git_source_bytes
    )

    report = InventoryReport(
        schema=INVENTORY_SCHEMA,
        version=INVENTORY_VERSION,
        interface_id=INTERFACE_ID,
        sinks=catalog,
        discoveries=found,
        unclassified=tuple(unclassified),
        direct_duckdb_writers=duckdb_writers,
        cross_file_atomicity_gaps=gaps,
        reuse_candidates=reuse,
        git_source_distinctions=git_bits,
        errors=tuple(errors),
    )
    if fail_on_unclassified and unclassified:
        raise UnclassifiedMutableSinkError(unclassified)
    if fail_on_unclassified and errors:
        raise RuntimeError("; ".join(errors))
    return report


def render_markdown(report: InventoryReport) -> str:
    """Render the human inventory document from a report (deterministic)."""

    lines: list[str] = [
        "# Agent Supervisor State Sink Inventory",
        "",
        f"Interface: `{report.interface_id}`  ",
        f"Schema: `{report.schema}`  ",
        f"Version: `{report.version}`  ",
        "Program: `agent-supervisor-duckdb-quack-control-plane-v1` / DQP-001",
        "",
        "This inventory classifies every mutable supervisor state sink that the",
        "DuckDB + Quack control-plane migration must absorb, demote, or retain.",
        "It is generated from the curated catalog in",
        "`scripts/ops/agent_supervisor/inventory_state_sinks.py` and gated by a",
        "deterministic scanner that **fails CI on any unclassified mutable sink**.",
        "",
        "## Authority boundary: Git/source bytes vs supervisor state",
        "",
        "| Kind | Authority? | Migrates into control.duckdb? | Notes |",
        "| --- | --- | --- | --- |",
        "| Git objects / worktree source files | **Yes for source bytes** | **No** | Git remains byte authority; DB stores identities, AST, mutations |",
        "| Markdown/JSON/JSONL/PID/lock/SQLite/DuckDB orchestration sinks | Yes for orchestration today | **Yes** (authority rows) or export/cache only | Classified below |",
        "| Operator-sealed scheduler/config inputs | Static input | Import once with provenance | Not daemon-rewritten authority |",
        "| Caches and dual-write query sidecars | No | Optional projections | Never grant leases or completion |",
        "",
        "The scanner treats Git/source bytes as `static_input` / `non_state` and",
        "marks `is_git_source_bytes=true`. They are **not** mutable supervisor",
        "orchestration sinks.",
        "",
        "## Classification taxonomy",
        "",
        "| Classification | Meaning |",
        "| --- | --- |",
    ]
    meanings = {
        SinkClassification.AUTHORITY: "Current write authority for orchestration decisions",
        SinkClassification.STATIC_INPUT: "Operator/source input; not rewritten as live authority",
        SinkClassification.IMMUTABLE_EVIDENCE: "Content-addressed or append-only evidence/receipts",
        SinkClassification.CACHE: "Rebuildable accelerator; never lease/completion authority",
        SinkClassification.EXPORT: "Read-only rendering of authoritative state",
        SinkClassification.OS_BOOTSTRAP: "PID/lock/status handles required by the OS or operators",
        SinkClassification.EMERGENCY_DIAGNOSTIC: "Recovery/quarantine diagnostics; fail-closed side path",
    }
    for item in SinkClassification:
        lines.append(f"| `{item.value}` | {meanings[item]} |")

    lines.extend(
        [
            "",
            "## Catalog summary",
            "",
            f"- Sink count: **{len(report.sinks)}**",
            f"- Direct DuckDB writers: **{len(report.direct_duckdb_writers)}**",
            f"- Cross-file atomicity gaps: **{len(report.cross_file_atomicity_gaps)}**",
            f"- Reuse candidates recorded: **{len(report.reuse_candidates)}**",
            f"- Discovery markers scanned: **{len(report.discoveries)}**",
            f"- Unclassified markers: **{len(report.unclassified)}**",
            "",
            "## Direct DuckDB writers",
            "",
            "These modules open DuckDB files for read-write orchestration or",
            "projection and must funnel through the future Quack state-owner:",
            "",
        ]
    )
    for sink in report.sinks:
        if not sink.is_direct_duckdb_writer:
            continue
        lines.append(
            f"- `{sink.sink_id}` — `{sink.writer_module}` → `{sink.path_template}` "
            f"(reuse: {sink.reuse_candidate})"
        )

    lines.extend(
        [
            "",
            "## Cross-file atomicity gaps",
            "",
            "Multi-file sinks that cannot commit intent, events, and status as one",
            "transaction today (migration must close these):",
            "",
        ]
    )
    if report.cross_file_atomicity_gaps:
        for gap in report.cross_file_atomicity_gaps:
            lines.append(f"- {gap}")
    else:
        lines.append("- _(none recorded)_")

    lines.extend(
        [
            "",
            "## Reuse candidates",
            "",
            "Existing primitives the control plane should generalize rather than",
            "re-implement:",
            "",
        ]
    )
    for item in report.reuse_candidates:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Full sink catalog",
            "",
            "| sink_id | class | media | domain | retirement | module | path |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for sink in report.sinks:
        lines.append(
            "| `{id}` | `{cls}` | `{media}` | `{domain}` | `{ret}` | `{mod}` | `{path}` |".format(
                id=sink.sink_id,
                cls=sink.classification.value,
                media=sink.media_type.value,
                domain=sink.destination_domain.value,
                ret=sink.retirement_stage.value,
                mod=sink.writer_module,
                path=sink.path_template.replace("|", "\\|"),
            )
        )

    lines.extend(
        [
            "",
            "### Sink details",
            "",
        ]
    )
    for sink in report.sinks:
        lines.extend(
            [
                f"#### `{sink.sink_id}`",
                "",
                f"- **Classification:** `{sink.classification.value}`",
                f"- **Media:** `{sink.media_type.value}`",
                f"- **Destination domain:** `{sink.destination_domain.value}`",
                f"- **Retirement stage:** `{sink.retirement_stage.value}`",
                f"- **Writer module:** `{sink.writer_module}`",
                f"- **Path template:** `{sink.path_template}`",
                f"- **Atomicity model:** `{sink.atomicity_model.value}`",
                f"- **Direct DuckDB writer:** `{'yes' if sink.is_direct_duckdb_writer else 'no'}`",
                f"- **Git/source bytes:** `{'yes' if sink.is_git_source_bytes else 'no'}`",
                f"- **Reuse candidate:** {sink.reuse_candidate}",
            ]
        )
        if sink.cross_file_atomicity_gap:
            lines.append(
                f"- **Cross-file atomicity gap:** {sink.cross_file_atomicity_gap}"
            )
        if sink.notes:
            lines.append(f"- **Notes:** {sink.notes}")
        lines.append("")

    lines.extend(
        [
            "## Scanner contract",
            "",
            "```text",
            "python scripts/ops/agent_supervisor/inventory_state_sinks.py --check",
            "python -m pytest -q test/api/test_agent_supervisor_state_sink_inventory.py",
            "```",
            "",
            "`--check` exits non-zero when any discovered mutable path marker is",
            "not covered by this catalog. Adding a new DuckDB/JSONL/PID/lock/status",
            "writer without classifying it fails CI.",
            "",
            "Source-tree scan globs such as `*.json`, `*.py`, or `*.pem` are **not**",
            "mutable supervisor sinks; only sink-family globs (`*.duckdb`, `*.jsonl`,",
            "`*.pid`, `*.lock`, `*.todo.md`, `*.objectives.md`, and status/progress",
            "patterns) are admitted into discovery. Package-manager lockfiles and Git",
            "index/ref locks (`index.lock`, `HEAD.lock`, …) are also excluded as",
            "Git/source inputs rather than orchestration state.",
            "",
            "## Non-goals",
            "",
            "- This inventory does not migrate state.",
            "- This inventory is not completion, proof, or lease authority.",
            "- Source code and Git objects stay on the filesystem; only their",
            "  identities and derived structure enter the control plane.",
            "",
        ]
    )
    return "\n".join(lines)


def inventory_doc_path(root: Path | None = None) -> Path:
    repo = repo_root_from(root) if root is not None else repo_root_from()
    return repo / INVENTORY_DOC_RELATIVE


def write_inventory_doc(
    root: Path | None = None,
    *,
    report: InventoryReport | None = None,
) -> Path:
    """Atomically write the inventory markdown document."""

    repo = repo_root_from(root) if root is not None else repo_root_from()
    built = report or build_inventory(repo, fail_on_unclassified=True)
    path = inventory_doc_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = render_markdown(built)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inventory and classify mutable agent-supervisor state sinks.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root (default: auto-detect from this script).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the inventory report as JSON on stdout.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if any discovered mutable sink is unclassified.",
    )
    parser.add_argument(
        "--write-doc",
        action="store_true",
        help=f"Write {INVENTORY_DOC_RELATIVE} from the catalog.",
    )
    parser.add_argument(
        "--allow-unclassified",
        action="store_true",
        help="Report unclassified markers without exiting non-zero (debug only).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = repo_root_from(args.root) if args.root is not None else repo_root_from()
    fail = bool(args.check) and not bool(args.allow_unclassified)
    try:
        report = build_inventory(root, fail_on_unclassified=fail)
    except UnclassifiedMutableSinkError as exc:
        payload = {
            "ok": False,
            "error": str(exc),
            "unclassified": [item.to_dict() for item in exc.unclassified],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    if args.write_doc:
        path = write_inventory_doc(root, report=report)
        if not args.json:
            print(f"wrote {path}")

    if args.json or args.check:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(
            f"{INTERFACE_ID}: sinks={len(report.sinks)} "
            f"discoveries={len(report.discoveries)} "
            f"unclassified={len(report.unclassified)} ok={report.ok}"
        )

    if not report.ok and not args.allow_unclassified:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
