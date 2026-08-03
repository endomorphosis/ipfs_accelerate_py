"""Backlog-refinery helpers for autonomous agent supervisors.

This module ports the repo-local supervisor feed logic into ``ipfs_accelerate_py``
without depending on the ``ipfs_datasets_py`` implementation package.  It keeps
the reusable pieces close to the accelerator daemon runtime:

* refill low todo queues from an objective heap,
* scan tracked code for small bug/improvement findings,
* turn repeated implementation, validation, or merge failures into
  evidence-backed follow-up tasks instead of allowing indefinite retry loops.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from hashlib import sha1, sha256
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..analysis.analyzer_health import (
    AnalyzerCanaryReport,
    AnalyzerHealthReport,
    AnalyzerHealthStatus,
    AnalyzerHealthThresholds,
    classify_analyzer_health,
    run_analyzer_canaries,
)
from ..merge.checkout_lock import (
    BACKLOG_REFINERY_AUTHOR_EMAIL,
    checkout_mutation_lock_path,
    generated_protected_board_commit_subject,
)
from ..runtime.event_log import read_jsonl_events
from .goal_completion import (
    DEFAULT_CLOCK_SKEW_SECONDS,
    DEFAULT_EVIDENCE_FRESHNESS_SECONDS,
)
from .objective_graph import (
    DEFAULT_DISCOVERY_OUTPUT_PATH,
    DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX,
    DEFAULT_SURPLUS_FINDINGS_PER_GOAL,
    DEFAULT_SURPLUS_MIN_TERMS_PER_TODO,
    LAUNCH_PLAYWRIGHT_VALIDATION_COMMAND,
    LAUNCH_PLAYWRIGHT_VALIDATION_GATE_EVIDENCE,
    LAUNCH_PLAYWRIGHT_VALIDATION_MARKERS,
    OBJECTIVE_SCAN_ANALYZER_VERSION,
    SUCCESSFUL_MERGE_RECEIPT_STATUSES,
    ObjectiveWorkProposal,
    ObjectiveGoal,
    bundle_path,
    generate_objective_todos_result,
    parse_goal_heap,
    repo_relative_path,
    safe_bundle_key,
    taskboard_namespace_from_todo,
)
from .scan_receipts import (
    DEFAULT_EXHAUSTION_QUORUM_SIZE,
    ExhaustionBinding,
    ExhaustionQuorumResult,
    RefillScanResult,
    RepositoryTreeIdentity,
    ScanAccounting,
    ScanTerminalReason,
    build_scan_result,
    evaluate_exhaustion_quorum,
    objective_revision as canonical_objective_revision,
    scan_configuration_revision,
    scan_identity,
)
from ..task_sources.task_identity import (
    TaskIdentity,
    canonical_task_identity,
    normalize_board_namespace,
)
from ..todo_daemon.implementation_daemon import (
    RECONCILIATION_GUARDRAIL_SCHEMA,
    RETRY_BUDGET_REPAIR_SCHEMA,
    is_retry_budget_repair_task,
    parse_task_file,
    retry_budget_repair_source,
)
from ..task_sources.taskboard_store import (
    locked_taskboard,
    replace_locked_taskboard,
    task_ids_from_artifact_names,
)
from ..validation.validation_commands import (
    infer_validation_impact_paths,
    normalize_validation_command_text,
    split_validation_commands,
)
from ..core.wrapper_utils import AgentSupervisorNamespacePaths


logger = logging.getLogger("ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery")

DEFAULT_CODEBASE_SCAN_MIN_OPEN_TASKS = int(os.environ.get("IPFS_ACCELERATE_AGENT_CODEBASE_SCAN_MIN_OPEN_TASKS", "5"))
DEFAULT_CODEBASE_SCAN_MAX_FINDINGS = int(os.environ.get("IPFS_ACCELERATE_AGENT_CODEBASE_SCAN_MAX_FINDINGS", "5"))
DEFAULT_CODEBASE_SCAN_COOLDOWN_SECONDS = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_CODEBASE_SCAN_COOLDOWN_SECONDS", "21600")
)
DEFAULT_OBJECTIVE_SCAN_MIN_OPEN_TASKS = int(os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_SCAN_MIN_OPEN_TASKS", "5"))
DEFAULT_OBJECTIVE_SCAN_MAX_FINDINGS = int(os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_SCAN_MAX_FINDINGS", "5"))
DEFAULT_OBJECTIVE_SCAN_COOLDOWN_SECONDS = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_OBJECTIVE_SCAN_COOLDOWN_SECONDS", "21600")
)
DEFAULT_VALIDATION_RETRY_BUDGET = int(os.environ.get("IPFS_ACCELERATE_AGENT_VALIDATION_RETRY_BUDGET", "3"))
DEFAULT_MERGE_RETRY_BUDGET = int(os.environ.get("IPFS_ACCELERATE_AGENT_MERGE_RETRY_BUDGET", "3"))
DEFAULT_IMPLEMENTATION_RETRY_BUDGET = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_IMPLEMENTATION_RETRY_BUDGET", "3")
)
DEFAULT_STALE_GIT_LOCK_SECONDS = float(
    os.environ.get("IPFS_ACCELERATE_AGENT_STALE_GIT_LOCK_SECONDS", "300")
)
DEFAULT_GENERATED_DIRTY_HARD_PATH_CAP = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_GENERATED_DIRTY_HARD_PATH_CAP", "200")
)
DEFAULT_GENERATED_DIRTY_MAX_DELETE_PATHS = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_GENERATED_DIRTY_MAX_DELETE_PATHS", "0")
)
DEFAULT_GENERATED_DIRTY_ALLOW_DELETIONS = (
    os.environ.get("IPFS_ACCELERATE_AGENT_GENERATED_DIRTY_ALLOW_DELETIONS", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)
DEFAULT_DEPENDENCY_GUARDRAIL_MAX_FINDINGS = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_DEPENDENCY_GUARDRAIL_MAX_FINDINGS", "5")
)
DEFAULT_RECONCILIATION_GUARDRAIL_MAX_FINDINGS = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_RECONCILIATION_GUARDRAIL_MAX_FINDINGS", "3")
)
DEFAULT_TASK_ID_PREFIX = "AUTO-"
DEFAULT_TASK_HEADER_PREFIX = "## AUTO-"
DEFAULT_REFILL_OPEN_TASK_HEADROOM = int(
    os.environ.get("IPFS_ACCELERATE_AGENT_REFILL_OPEN_TASK_HEADROOM", "1")
)
DEFAULT_SELF_IMPROVEMENT_SUCCESSOR_COOLDOWN_SECONDS = int(
    os.environ.get(
        "IPFS_ACCELERATE_AGENT_SELF_IMPROVEMENT_SUCCESSOR_COOLDOWN_SECONDS",
        "21600",
    )
)


def align_completion_gate_force_goal_ids(
    force_goal_ids: Sequence[str] = (),
    *,
    completion_gate_decisions: Mapping[str, Any] | None = None,
    repository_id: str = "",
    repository_tree: str = "",
    now: datetime | str | None = None,
    freshness_seconds: float = DEFAULT_EVIDENCE_FRESHNESS_SECONDS,
    clock_skew_seconds: float = DEFAULT_CLOCK_SKEW_SECONDS,
) -> tuple[str, ...]:
    """Keep proof-incomplete objective parents in the supervisor refill heap.

    The objective tracker owns lifecycle interpretation.  This backlog helper
    only merges its fail-closed actionable projection with explicitly forced
    goals, preserving deterministic order and avoiding duplicate tasks.
    """

    from .objective_tracker import completion_gate_actionable_goal_ids

    aligned = {
        str(goal_id).strip()
        for goal_id in force_goal_ids
        if str(goal_id).strip()
    }
    for goal_id, decision in sorted(
        (completion_gate_decisions or {}).items(),
        key=lambda item: str(item[0]),
    ):
        aligned.update(
            completion_gate_actionable_goal_ids(
                str(goal_id),
                decision,
                repository_id=repository_id,
                repository_tree=repository_tree,
                now=now,
                freshness_seconds=freshness_seconds,
                clock_skew_seconds=clock_skew_seconds,
            )
        )
    return tuple(sorted(aligned))


DEFAULT_SELF_IMPROVEMENT_SUCCESSOR_RECORD_LIMIT = int(
    os.environ.get(
        "IPFS_ACCELERATE_AGENT_SELF_IMPROVEMENT_SUCCESSOR_RECORD_LIMIT",
        "4096",
    )
)
SELF_IMPROVEMENT_SUCCESSOR_REJECTION_DETAIL_LIMIT = 512
SELF_IMPROVEMENT_SUCCESSOR_RECORD_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.self_improvement_successor_admission.v1"
)
SELF_IMPROVEMENT_SUCCESSOR_RECORDS_KEY = (
    "self_improvement_successor_admission_records"
)
CODEBASE_SCAN_ANALYZER_VERSION = "codebase-annotation-analyzer/v2"
CODEBASE_AUDIT_SCANNER_VERSION = "codebase-audit/v1"
CODEBASE_SCAN_REASON_SAMPLE_LIMIT = 10


def _bounded_unique_representative_paths(paths: Iterable[Any]) -> list[str]:
    """Return stable, non-empty path samples accepted by scan receipts."""

    representatives: list[str] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = str(raw_path or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        representatives.append(path)
        if len(representatives) >= CODEBASE_SCAN_REASON_SAMPLE_LIMIT:
            break
    return representatives


CODEBASE_SCAN_MAX_FILE_BYTES = int(os.environ.get("IPFS_ACCELERATE_AGENT_CODEBASE_SCAN_MAX_FILE_BYTES", "262144"))
CODEBASE_SCAN_SUFFIXES = {
    ".cjs",
    ".css",
    ".html",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".mjs",
    ".py",
    ".rs",
    ".sh",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
CODEBASE_SCAN_SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "playwright-report",
    "test-results",
}
CODEBASE_SCAN_SKIP_PREFIXES = (
    "archive/",
    "backup/",
    "cleanup-archive/",
    "data/agent_supervisor/discovery/",
    "data/agent_supervisor/objective_bundles/",
    "data/agent_supervisor/objective_datasets/",
    "data/agent_supervisor/state/",
    "data/agent_supervisor/worktrees/",
    "external/ipfs_accelerate/test/duckdb_api/",
    "external/ipfs_accelerate/test/generators/",
    "external/ipfs_accelerate/test/huggingface_transformers/",
    "external/ipfs_accelerate/test/skills/",
    "external/ipfs_kit/archive/",
    "external/ipfs_kit/backup/",
)
ANNOTATION_FOLLOWUP_RE = re.compile(
    r"""
    (?:
        ^\s*(?:[-*]\s*)?(?P<line_marker>todo|fixme|hack|xxx)\b\s*(?::|\(|-)
        |
        (?P<comment_prefix>\#|//|/\*|<!--|--)\s*(?P<comment_marker>todo|fixme|hack|xxx)\b\s*(?::|\(|-)
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class CodebaseFinding:
    """One static codebase finding that can be converted to a todo task."""

    fingerprint: str
    kind: str
    priority: str
    track: str
    root_relative_path: str
    line_number: int
    snippet: str
    summary: str
    validation: str
    objective_goal_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelfImprovementSuccessorRejection:
    """One successor excluded before objective materialization.

    Rejections deliberately retain both proposal identities.  Canonical IDs
    distinguish exact proposal content while semantic keys prevent a cosmetic
    rewrite from bypassing lifecycle or cooldown deduplication.
    """

    canonical_id: str
    semantic_key: str
    reason: "SelfImprovementSuccessorRejectionReason | str"
    detail: str = ""

    def __post_init__(self) -> None:
        try:
            reason = (
                self.reason
                if isinstance(
                    self.reason, SelfImprovementSuccessorRejectionReason
                )
                else SelfImprovementSuccessorRejectionReason(str(self.reason))
            )
        except ValueError as exc:
            raise ValueError(
                f"unsupported successor rejection reason {self.reason!r}"
            ) from exc
        # Preserve the longstanding public ``str`` field while validating it
        # against the closed vocabulary above.
        object.__setattr__(self, "reason", reason.value)
        object.__setattr__(
            self,
            "detail",
            bounded_successor_rejection_detail(self.detail),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "canonical_id": self.canonical_id,
            "semantic_key": self.semantic_key,
            "reason": self.reason,
            "detail": self.detail,
        }


class SelfImprovementSuccessorRejectionReason(str, Enum):
    """Closed reason vocabulary for bounded successor pre-admission."""

    INVALID_PROPOSAL = "invalid_proposal"
    CANDIDATE_LIMIT = "candidate_limit"
    LIFECYCLE_DUPLICATE = "lifecycle_duplicate"
    PRIOR_ADMISSION_DUPLICATE = "prior_admission_duplicate"
    SUCCESSOR_COOLDOWN = "successor_cooldown"
    BATCH_DUPLICATE = "batch_duplicate"
    SEMANTIC_DUPLICATE = "semantic_duplicate"
    UNSUPPORTED_DEPENDENCY = "unsupported_dependency"


def bounded_successor_rejection_detail(
    detail: Any,
    *,
    max_bytes: int = SELF_IMPROVEMENT_SUCCESSOR_REJECTION_DETAIL_LIMIT,
) -> str:
    """Return UTF-8-safe rejection detail within one hard byte budget."""

    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes < 0
    ):
        raise ValueError("max_bytes must be a non-negative integer")
    encoded = str(detail or "").encode("utf-8")
    if len(encoded) <= max_bytes:
        return encoded.decode("utf-8")
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


@dataclass(frozen=True)
class SelfImprovementSuccessorFilterResult:
    """Deterministic pre-admission accounting for one successor candidate set."""

    eligible: tuple[ObjectiveWorkProposal, ...]
    rejected: tuple[SelfImprovementSuccessorRejection, ...]
    lifecycle_canonical_ids: tuple[str, ...] = ()
    lifecycle_semantic_keys: tuple[str, ...] = ()
    cooldown_canonical_ids: tuple[str, ...] = ()
    cooldown_semantic_keys: tuple[str, ...] = ()

    @property
    def candidate_count(self) -> int:
        return len(self.eligible) + len(self.rejected)

    @property
    def eligible_count(self) -> int:
        return len(self.eligible)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": (
                "ipfs_accelerate_py.agent_supervisor."
                "self_improvement_successor_filter.v1"
            ),
            "candidate_count": self.candidate_count,
            "eligible_count": self.eligible_count,
            "rejected_count": self.rejected_count,
            "eligible": [item.to_dict() for item in self.eligible],
            "rejected": [item.to_dict() for item in self.rejected],
            "lifecycle_canonical_ids": list(self.lifecycle_canonical_ids),
            "lifecycle_semantic_keys": list(self.lifecycle_semantic_keys),
            "cooldown_canonical_ids": list(self.cooldown_canonical_ids),
            "cooldown_semantic_keys": list(self.cooldown_semantic_keys),
        }


@dataclass
class CodebaseScanInventory:
    """Auditable coverage and candidate funnel for one codebase scan.

    ``excluded_files`` and ``parser_failures`` retain every path for the
    durable details artifact.  Receipts project those collections to bounded
    reason summaries so their size does not grow with the repository.
    """

    findings: list[CodebaseFinding] = field(default_factory=list)
    git_roots: list[str] = field(default_factory=list)
    expected_git_roots: list[str] = field(default_factory=list)
    tracked_file_count: int = 0
    tracked_paths: list[str] = field(default_factory=list)
    eligible_paths: list[str] = field(default_factory=list)
    eligible_file_count: int = 0
    parsed_file_count: int = 0
    cache_hit_count: int = 0
    excluded_files: list[dict[str, str]] = field(default_factory=list)
    parser_failures: list[dict[str, str]] = field(default_factory=list)
    raw_candidate_count: int = 0
    seen_candidate_count: int = 0
    deduplicated_candidate_count: int = 0
    rejected_candidate_count: int = 0
    complete: bool = True

    @property
    def excluded_file_count(self) -> int:
        return len(self.excluded_files)

    @property
    def parser_failure_count(self) -> int:
        return len(self.parser_failures)

    def coverage_dict(self) -> dict[str, int]:
        return {
            "git_roots": len(self.git_roots),
            "tracked_files": self.tracked_file_count,
            "eligible_files": self.eligible_file_count,
            "parsed_files": self.parsed_file_count,
            "cache_hits": self.cache_hit_count,
            "excluded_files": self.excluded_file_count,
            "parser_failures": self.parser_failure_count,
        }

    def health_inventory_dict(
        self,
        *,
        appended_tasks: int,
        late_deduplicated_candidates: int = 0,
        additional_rejected_candidates: int = 0,
    ) -> dict[str, Any]:
        """Return all counters required by fail-closed health evaluation."""

        return {
            **self.coverage_dict(),
            "expected_git_root_count": len(self.expected_git_roots),
            "raw_candidates": self.raw_candidate_count,
            "seen_candidates": self.seen_candidate_count,
            "deduplicated_candidates": (
                self.deduplicated_candidate_count + late_deduplicated_candidates
            ),
            "rejected_candidates": (
                self.rejected_candidate_count
                + max(0, int(additional_rejected_candidates))
            ),
            "appended_tasks": appended_tasks,
            "coverage_complete": self.complete,
        }

    def reason_summaries(self) -> dict[str, list[dict[str, Any]]]:
        def summarize(records: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
            grouped: dict[str, list[str]] = {}
            for record in records:
                grouped.setdefault(str(record["reason_code"]), []).append(str(record["path"]))
            return [
                {
                    "reason_code": reason_code,
                    "count": len(paths),
                    "representative_paths": _bounded_unique_representative_paths(paths),
                }
                for reason_code, paths in sorted(grouped.items())
            ]

        return {
            "exclusions": summarize(self.excluded_files),
            "parser_failures": summarize(self.parser_failures),
        }

    def details_dict(self) -> dict[str, Any]:
        return {
            "analyzer_version": CODEBASE_SCAN_ANALYZER_VERSION,
            "coverage": self.coverage_dict(),
            "git_roots": list(self.git_roots),
            "expected_git_roots": list(self.expected_git_roots),
            "tracked_paths": list(self.tracked_paths),
            "eligible_paths": list(self.eligible_paths),
            "excluded_files": list(self.excluded_files),
            "parser_failures": list(self.parser_failures),
            "candidate_accounting": {
                "raw_candidates": self.raw_candidate_count,
                "seen_candidates": self.seen_candidate_count,
                "deduplicated_candidates": self.deduplicated_candidate_count,
                "selected_candidates": len(self.findings),
                "rejected_candidates": self.rejected_candidate_count,
            },
            "coverage_complete": self.complete,
        }


@dataclass(frozen=True)
class CodebaseRefillAdmission:
    """Goal-scoped disposition of an objective-agnostic scan inventory."""

    findings: tuple[CodebaseFinding, ...] = ()
    rejections: tuple[Mapping[str, Any], ...] = ()
    policy_errors: tuple[Mapping[str, str], ...] = ()
    allow_unscoped: bool = False
    max_findings: int | None = None

    @property
    def policy_valid(self) -> bool:
        return not self.policy_errors

    @property
    def rejected_candidate_count(self) -> int:
        return len(self.rejections)

    @property
    def admission_rejections(self) -> tuple[Mapping[str, Any], ...]:
        """Compatibility name used by early callers of the admission stage."""

        return self.rejections

    def reason_summaries(self) -> list[dict[str, Any]]:
        grouped: dict[str, list[str]] = {}
        for record in self.rejections:
            reason_code = str(record.get("reason_code") or "no_goal_lineage")
            grouped.setdefault(reason_code, []).append(str(record.get("path") or ""))
        return [
            {
                "reason_code": reason_code,
                "count": len(paths),
                "representative_paths": _bounded_unique_representative_paths(paths),
            }
            for reason_code, paths in sorted(grouped.items())
        ]

    def details_dict(self) -> dict[str, Any]:
        return {
            "allow_unscoped": self.allow_unscoped,
            "max_findings": self.max_findings,
            "policy_valid": self.policy_valid,
            "policy_errors": [dict(item) for item in self.policy_errors],
            "admitted_candidate_count": len(self.findings),
            "rejected_candidate_count": self.rejected_candidate_count,
            "rejections": [dict(record) for record in self.rejections],
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_MARKDOWN_HEADING_PREFIX_RE = re.compile(r"^\s*#{1,6}\s*")


def task_id_prefix(value: str) -> str:
    """Return a canonical display-ID prefix without Markdown rendering.

    Older wrapper configurations used values such as ``"## AUTO-"``.  Treat
    that form as a boundary adapter, including malformed values that acquired
    more than one heading marker, and keep every internal ID operation on the
    canonical ``"AUTO-"`` form.
    """

    normalized = str(value or DEFAULT_TASK_ID_PREFIX).strip()
    while _MARKDOWN_HEADING_PREFIX_RE.match(normalized):
        normalized = _MARKDOWN_HEADING_PREFIX_RE.sub("", normalized, count=1).strip()
    return normalized or DEFAULT_TASK_ID_PREFIX


def task_header_prefix(value: str) -> str:
    """Render a canonical task ID prefix as one level-two Markdown heading."""

    return f"## {task_id_prefix(value or DEFAULT_TASK_HEADER_PREFIX)}"


def normalize_task_id(value: Any) -> str:
    """Normalize a task-ID alias supplied in legacy heading form."""

    normalized = str(value or "").strip()
    while _MARKDOWN_HEADING_PREFIX_RE.match(normalized):
        normalized = _MARKDOWN_HEADING_PREFIX_RE.sub("", normalized, count=1).strip()
    return normalized.split(None, 1)[0] if normalized else ""


def task_id_pattern(task_prefix: str = DEFAULT_TASK_ID_PREFIX) -> re.Pattern[str]:
    """Return the strict Markdown heading parser for one numeric task family."""

    prefix = task_id_prefix(task_prefix)
    return re.compile(
        rf"^##\s+({re.escape(prefix)}(?P<number>\d+))(?=\s|$)",
        flags=re.MULTILINE,
    )


def normalize_task_block_heading(block: str, task_id: str) -> str:
    """Render the first task-block heading exactly once.

    Callers may still provide a legacy block beginning ``## ## AUTO-001``.
    Normalize that input at this append boundary without rewriting the body.
    """

    text = str(block or "").strip()
    canonical_id = normalize_task_id(task_id)
    if not text or not canonical_id:
        return text
    lines = text.splitlines()
    heading = lines[0].strip()
    while _MARKDOWN_HEADING_PREFIX_RE.match(heading):
        heading = _MARKDOWN_HEADING_PREFIX_RE.sub("", heading, count=1).strip()
    parts = heading.split(None, 1)
    title = parts[1] if len(parts) > 1 and normalize_task_id(parts[0]) == canonical_id else ""
    lines[0] = f"## {canonical_id}{f' {title}' if title else ''}"
    return "\n".join(lines)


def split_csv(values: Iterable[str] | str) -> list[str]:
    raw_values = [values] if isinstance(values, str) else list(values)
    items: list[str] = []
    for value in raw_values:
        for raw in str(value).split(","):
            item = " ".join(raw.strip().split())
            if item and item.lower() not in {"none", "n/a"}:
                items.append(item)
    return items


def task_ids_from_todo_text(todo_text: str, *, task_prefix: str = DEFAULT_TASK_ID_PREFIX) -> list[str]:
    return [match.group(1) for match in task_id_pattern(task_prefix).finditer(todo_text)]


def task_block_is_present(todo_text: str, task_id: str) -> bool:
    """Return whether a markdown task block for ``task_id`` is already present."""

    escaped_task_id = re.escape(normalize_task_id(task_id))
    if not escaped_task_id:
        return False
    return re.search(rf"^##\s+{escaped_task_id}(?:\s|$)", todo_text, flags=re.MULTILINE) is not None


def task_block_semantic_identities(text: str) -> set[str]:
    """Collect explicit canonical identities suitable for exact deduplication."""

    identities: set[str] = set()
    fields = (
        "Canonical task key",
        "Canonical task CID",
        "Semantic identity",
        "Evidence obligation key",
        "Dedupe key",
        "Todo vector key",
    )
    for field_name in fields:
        for match in re.finditer(
            rf"^-\s*{re.escape(field_name)}:\s*(\S+)\s*$",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        ):
            value = match.group(1).strip().casefold()
            if value and value not in {"none", "n/a"}:
                identities.add(f"identity:{value}")
    return identities


def ensure_task_blocks_present(
    todo_path: Path,
    task_blocks: Mapping[str, str] | Sequence[tuple[str, str]],
) -> bool:
    """Append missing markdown task blocks to a todo board in caller-provided order."""

    if not todo_path.exists():
        return False
    todo_text = todo_path.read_text(encoding="utf-8")
    existing_semantic_identities = task_block_semantic_identities(todo_text)
    entries = task_blocks.items() if isinstance(task_blocks, Mapping) else task_blocks
    additions: list[str] = []
    for raw_task_id, raw_block in entries:
        task_id = normalize_task_id(raw_task_id)
        block = normalize_task_block_heading(raw_block, task_id)
        semantic_identities = task_block_semantic_identities(block)
        is_semantic_duplicate = bool(
            semantic_identities and semantic_identities & existing_semantic_identities
        )
        if (
            block
            and task_id
            and not task_block_is_present(todo_text, task_id)
            and not is_semantic_duplicate
        ):
            additions.append(block)
            existing_semantic_identities.update(semantic_identities)
    if not additions:
        return False
    todo_path.write_text(todo_text.rstrip() + "\n\n" + "\n\n".join(additions) + "\n", encoding="utf-8")
    return True


def build_task_blocks_ensurer(
    task_blocks: Mapping[str, str] | Sequence[tuple[str, str]],
    *,
    default_todo_path: Path | None = None,
) -> Callable[[Path | None], bool]:
    """Build a callback that appends configured task blocks to a todo board."""

    configured_blocks = dict(task_blocks.items() if isinstance(task_blocks, Mapping) else task_blocks)

    def ensurer(todo_path: Path | None = None) -> bool:
        path = todo_path or default_todo_path
        if path is None:
            raise ValueError("todo_path is required when no default todo path is configured")
        return ensure_task_blocks_present(path, configured_blocks)

    return ensurer


def next_task_id(
    todo_text: str,
    *,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
    reserved_task_ids: Iterable[str] = (),
) -> str:
    prefix = task_id_prefix(task_prefix)
    highest = 0
    width = 3
    task_ids = [
        *task_ids_from_todo_text(todo_text, task_prefix=prefix),
        *(str(item) for item in reserved_task_ids),
    ]
    id_re = re.compile(rf"^{re.escape(prefix)}(?P<number>\d+)$")
    for current in task_ids:
        match = id_re.fullmatch(normalize_task_id(current))
        if match is None:
            continue
        number = match.group("number")
        highest = max(highest, int(number))
        width = max(width, len(number))
    return f"{prefix}{highest + 1:0{width}d}"


def task_statuses_from_todo_text(todo_text: str, *, task_prefix: str = DEFAULT_TASK_ID_PREFIX) -> dict[str, str]:
    statuses: dict[str, str] = {}
    current_task_id = ""
    for line in todo_text.splitlines():
        heading = task_id_pattern(task_prefix).match(line)
        if heading is not None:
            current_task_id = heading.group(1)
            continue
        if line.startswith("## "):
            current_task_id = ""
            continue
        if current_task_id and line.startswith("- Status:"):
            statuses[current_task_id] = line.split(":", 1)[1].strip().lower()
            current_task_id = ""
    return statuses


def state_statuses_match_todo_statuses(
    todo_statuses: Mapping[str, str],
    state_statuses: Mapping[str, str],
) -> bool:
    """Return whether daemon state still matches the markdown board statuses."""

    if set(todo_statuses) != set(state_statuses):
        return False
    compatible_state_statuses = {
        "blocked": {"blocked"},
        "completed": {"completed"},
        "in_progress": {"in_progress"},
        "ready": {"ready", "todo"},
        "todo": {"ready", "todo"},
        "waiting": {"waiting"},
    }
    for task_id, todo_status in todo_statuses.items():
        normalized_todo = str(todo_status or "").lower()
        normalized_state = str(state_statuses.get(task_id) or "").lower()
        if normalized_todo not in {"blocked", "completed"} and normalized_state in {"blocked", "completed"}:
            continue
        allowed = compatible_state_statuses.get(normalized_todo, {normalized_todo})
        if normalized_state not in allowed:
            return False
    return True


def mark_task_statuses_in_todo_text(
    todo_text: str,
    task_ids: Sequence[str],
    *,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
    status: str = "completed",
) -> tuple[str, list[str]]:
    """Return todo text with selected task status lines rewritten."""

    target_task_ids = {
        normalize_task_id(task_id)
        for task_id in task_ids
        if normalize_task_id(task_id)
    }
    if not target_task_ids:
        return todo_text, []

    lines = todo_text.splitlines(keepends=True)
    current_task_id = ""
    updated_task_ids: list[str] = []
    for index, line in enumerate(lines):
        heading = task_id_pattern(task_prefix).match(line)
        if heading is not None:
            current_task_id = heading.group(1)
            continue
        if line.startswith("## "):
            current_task_id = ""
            continue
        if current_task_id not in target_task_ids or not line.startswith("- Status:"):
            continue
        current_status = line.split(":", 1)[1].strip().lower()
        if current_status == status.lower():
            current_task_id = ""
            continue
        newline = "\n" if line.endswith("\n") else ""
        lines[index] = f"- Status: {status}{newline}"
        updated_task_ids.append(current_task_id)
        current_task_id = ""
    if not updated_task_ids:
        return todo_text, []
    return "".join(lines), updated_task_ids


def open_task_count(todo_text: str, *, task_prefix: str = DEFAULT_TASK_ID_PREFIX) -> int:
    statuses = task_statuses_from_todo_text(todo_text, task_prefix=task_prefix)
    return sum(1 for status in statuses.values() if status not in {"completed", "blocked"})


def load_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_strategy(path: Path) -> dict[str, Any]:
    repair_reason = ""
    if not path.exists():
        strategy: dict[str, Any] = {}
        repair_reason = "missing_strategy_file"
    else:
        try:
            raw_text = path.read_text(encoding="utf-8").strip()
        except OSError:
            strategy = {}
            repair_reason = "unreadable_strategy_file"
        else:
            if not raw_text:
                strategy = {}
                repair_reason = "empty_strategy_file"
            else:
                try:
                    payload = json.loads(raw_text)
                except json.JSONDecodeError:
                    strategy = {}
                    repair_reason = "invalid_strategy_json"
                else:
                    if isinstance(payload, dict):
                        strategy = dict(payload)
                    else:
                        strategy = {}
                        repair_reason = "non_object_strategy_json"
    if not strategy:
        strategy = {"blocked_tasks": []}
    blocked = strategy.get("blocked_tasks")
    strategy["blocked_tasks"] = [str(item) for item in blocked] if isinstance(blocked, list) else []
    if repair_reason:
        strategy["last_strategy_repair_at"] = utc_now()
        strategy["last_strategy_repair_reason"] = repair_reason
        write_json(path, strategy)
    return strategy


def effective_open_task_count(
    todo_text: str,
    *,
    state_path: Path | None = None,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
) -> int:
    if state_path is None or not state_path.exists():
        return open_task_count(todo_text, task_prefix=task_prefix)
    payload = load_json_dict(state_path)
    statuses = payload.get("task_statuses")
    if not isinstance(statuses, dict):
        return open_task_count(todo_text, task_prefix=task_prefix)
    todo_statuses = task_statuses_from_todo_text(todo_text, task_prefix=task_prefix)
    task_ids = set(todo_statuses)
    normalized = {str(task_id): str(status).lower() for task_id, status in statuses.items()}
    if set(normalized) != task_ids or not state_statuses_match_todo_statuses(todo_statuses, normalized):
        return open_task_count(todo_text, task_prefix=task_prefix)
    try:
        state_task_count = int(payload.get("task_count") or 0)
    except (TypeError, ValueError):
        return open_task_count(todo_text, task_prefix=task_prefix)
    if state_task_count != len(task_ids):
        return open_task_count(todo_text, task_prefix=task_prefix)
    return sum(1 for status in normalized.values() if status not in {"completed", "blocked"})


def refill_state_counts(
    todo_text: str,
    *,
    state_path: Path | None = None,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
) -> dict[str, int]:
    if state_path is None or not state_path.exists():
        return {}
    payload = load_json_dict(state_path)
    statuses = payload.get("task_statuses")
    if not isinstance(statuses, dict):
        return {}
    todo_statuses = task_statuses_from_todo_text(todo_text, task_prefix=task_prefix)
    task_ids = set(todo_statuses)
    normalized = {str(task_id): str(status).lower() for task_id, status in statuses.items()}
    if set(normalized) != task_ids:
        return {}
    try:
        state_task_count = int(payload.get("task_count") or 0)
    except (TypeError, ValueError):
        return {}
    if state_task_count != len(task_ids):
        return {}

    def count(name: str, fallback: int) -> int:
        try:
            return int(payload.get(name))
        except (TypeError, ValueError):
            return fallback

    completed = sum(1 for status in normalized.values() if status == "completed")
    blocked = sum(1 for status in normalized.values() if status == "blocked")
    ready = sum(1 for status in normalized.values() if status == "todo")
    waiting = sum(1 for status in normalized.values() if status == "waiting")
    return {
        "task_count": state_task_count,
        "completed_count": count("completed_count", completed),
        "blocked_count": count("blocked_count", blocked),
        "ready_count": count("ready_count", ready),
        "selectable_ready_count": count("selectable_ready_count", ready),
        "eligible_ready_count": count("eligible_ready_count", ready),
        "strict_deprioritized_ready_count": count("strict_deprioritized_ready_count", 0),
        "waiting_count": count("waiting_count", waiting),
    }


def parse_iso_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _self_improvement_successor_timestamp(
    value: datetime | str | None,
    *,
    field_name: str,
) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = parse_iso_timestamp(str(value).strip())
        if parsed is None:
            raise ValueError(f"{field_name} must be an ISO-8601 timestamp")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _successor_semantic_tokens(value: Any) -> frozenset[str]:
    """Project successor content to stable lexical terms.

    Identity digests, timestamps, and mapping keys are deliberately excluded:
    novelty is about the proposed work, not serialization details.  Proposal
    objects use the same evidence/scope fields as objective refinement.
    """

    if isinstance(value, ObjectiveWorkProposal):
        value = (
            value.kind.value,
            value.title,
            value.parent_goal_id,
            value.parent_objective_terms,
            value.expected_evidence_delta,
            value.predicted_files,
            value.predicted_symbols,
            value.acceptance_subset,
            value.effects,
            value.evidence_subset,
        )
    elif isinstance(value, ObjectiveGoal):
        value = (
            value.title,
            value.fields.get("goal", ""),
            value.required_evidence,
            value.predicted_files,
            value.predicted_symbols,
        )
    elif isinstance(value, Mapping):
        semantic_fields = (
            "kind",
            "title",
            "goal",
            "parent_goal_id",
            "parent_objective_terms",
            "expected_evidence_delta",
            "predicted_files",
            "predicted_symbols",
            "acceptance_subset",
            "acceptance",
            "effects",
            "evidence_subset",
        )
        value = tuple(value.get(name) for name in semantic_fields if name in value)

    pieces: list[str] = []

    def append(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, str):
            pieces.append(item)
            return
        if isinstance(item, Mapping):
            for child in item.values():
                append(child)
            return
        if isinstance(item, Iterable) and not isinstance(
            item, (bytes, bytearray)
        ):
            for child in item:
                append(child)
            return
        pieces.append(str(item))

    append(value)
    return frozenset(
        re.findall(r"[a-z0-9]+", " ".join(pieces).casefold())
    )


def semantic_novelty_distance(
    candidate_text: Any,
    existing_texts: Iterable[Any] | Any = (),
) -> float:
    """Return the candidate's deterministic distance from its nearest peer.

    Distance is one minus lexical Jaccard similarity and is therefore finite
    and bounded in ``[0, 1]``.  No history means fully novel.  Empty candidate
    content is never treated as novel when a comparison population exists.
    """

    if existing_texts is None:
        references = ()
    elif isinstance(
        existing_texts,
        (str, bytes, bytearray, Mapping, ObjectiveWorkProposal, ObjectiveGoal),
    ):
        references = (existing_texts,)
    else:
        references = tuple(existing_texts)
    if not references:
        return 1.0
    candidate_tokens = _successor_semantic_tokens(candidate_text)
    if not candidate_tokens:
        return 0.0
    nearest_similarity = 0.0
    for reference in references:
        reference_tokens = _successor_semantic_tokens(reference)
        union = candidate_tokens | reference_tokens
        similarity = (
            len(candidate_tokens & reference_tokens) / len(union)
            if union
            else 1.0
        )
        nearest_similarity = max(nearest_similarity, similarity)
    return max(0.0, min(1.0, 1.0 - nearest_similarity))


def unsupported_successor_dependencies(
    dependencies: Iterable[Any],
    supported_dependencies: Iterable[Any],
) -> tuple[str, ...]:
    """Return exact declared dependencies absent from a capability snapshot."""

    dependency_values = (
        (dependencies,)
        if isinstance(dependencies, (str, bytes, bytearray))
        else tuple(dependencies)
    )
    supported_values = (
        (supported_dependencies,)
        if isinstance(supported_dependencies, (str, bytes, bytearray))
        else tuple(supported_dependencies)
    )
    supported = {
        str(item).strip().casefold()
        for item in supported_values
        if str(item).strip()
    }
    unsupported: dict[str, str] = {}
    for raw in dependency_values:
        dependency = str(raw).strip()
        if dependency and dependency.casefold() not in supported:
            unsupported.setdefault(dependency.casefold(), dependency)
    return tuple(
        unsupported[key]
        for key in sorted(unsupported, key=lambda item: (item, unsupported[item]))
    )


def self_improvement_successor_lifecycle_identities(
    objective_text: str,
) -> tuple[set[str], set[str]]:
    """Return proposal identities owned by every objective lifecycle state.

    Completed, rejected, blocked, and reopened goals remain deduplication
    authority.  Looking only at schedulable goals would permit the same work
    to be regenerated as soon as it reached a terminal state.
    """

    canonical_ids: set[str] = set()
    semantic_keys: set[str] = set()
    for goal in parse_goal_heap(objective_text):
        canonical_id = goal.canonical_proposal_id
        semantic_key = goal.semantic_key
        if canonical_id:
            canonical_ids.add(canonical_id)
        if semantic_key:
            semantic_keys.add(semantic_key)
    return canonical_ids, semantic_keys


def self_improvement_successor_admission_records(
    strategy: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate and return the durable successor admission/cooldown ledger."""

    raw = strategy.get(SELF_IMPROVEMENT_SUCCESSOR_RECORDS_KEY) or {}
    if not isinstance(raw, Mapping):
        raise ValueError(
            f"{SELF_IMPROVEMENT_SUCCESSOR_RECORDS_KEY} must be an object"
        )
    result: dict[str, dict[str, Any]] = {}
    allowed_statuses = {
        "admitted",
        "committed",
        "cooldown",
        "failed",
        "materialized",
        "prepared",
        "rejected",
        "review_required",
    }
    allowed_fields = {
        "schema",
        "version",
        "canonical_id",
        "semantic_key",
        "status",
        "epoch_id",
        "transaction_id",
        "recorded_at",
        "cooldown_until",
        "reason_codes",
        "attempts",
    }
    for raw_key, raw_record in raw.items():
        if not isinstance(raw_record, Mapping):
            raise ValueError("successor admission records must be objects")
        record = dict(raw_record)
        unknown_fields = sorted(
            str(key) for key in record if str(key) not in allowed_fields
        )
        if unknown_fields:
            raise ValueError(
                "successor admission record contains unknown fields: "
                + ", ".join(unknown_fields)
            )
        canonical_id = str(record.get("canonical_id") or "").strip()
        semantic_key = str(record.get("semantic_key") or "").strip()
        status = str(record.get("status") or "").strip().lower()
        if not canonical_id or str(raw_key) != canonical_id:
            raise ValueError(
                "successor admission record key must match canonical_id"
            )
        if not semantic_key:
            raise ValueError("successor admission records require semantic_key")
        version = record.get("version")
        if (
            record.get("schema") != SELF_IMPROVEMENT_SUCCESSOR_RECORD_SCHEMA
            or isinstance(version, bool)
            or version != 1
        ):
            raise ValueError("unsupported successor admission record schema")
        if status not in allowed_statuses:
            raise ValueError(
                f"unsupported successor admission status {status!r}"
            )
        recorded_at = parse_iso_timestamp(str(record.get("recorded_at") or ""))
        if recorded_at is None:
            raise ValueError("successor admission records require recorded_at")
        cooldown_until = str(record.get("cooldown_until") or "").strip()
        if cooldown_until and parse_iso_timestamp(cooldown_until) is None:
            raise ValueError(
                "successor admission cooldown_until must be an ISO-8601 timestamp"
            )
        transaction_id = str(record.get("transaction_id") or "").strip()
        if status in {"admitted", "committed", "materialized"} and not transaction_id:
            raise ValueError(
                "successful successor admission records require transaction_id"
            )
        if status not in {"admitted", "committed", "materialized"} and not cooldown_until:
            raise ValueError(
                "non-admitted successor records require cooldown_until"
            )
        raw_reasons = record.get("reason_codes") or ()
        if not isinstance(raw_reasons, Sequence) or isinstance(
            raw_reasons, (str, bytes)
        ):
            raise ValueError("successor admission reason_codes must be a list")
        normalized = {
            **record,
            "canonical_id": canonical_id,
            "semantic_key": semantic_key,
            "status": status,
            "recorded_at": recorded_at.astimezone(timezone.utc).isoformat(),
            "cooldown_until": cooldown_until,
            "epoch_id": str(record.get("epoch_id") or "").strip(),
            "transaction_id": transaction_id,
            "reason_codes": sorted(
                {
                    str(item).strip()
                    for item in raw_reasons
                    if str(item).strip()
                }
            ),
        }
        attempts = normalized.get("attempts") or ()
        if not isinstance(attempts, Sequence) or isinstance(
            attempts, (str, bytes)
        ):
            raise ValueError("successor admission attempts must be a list")
        if any(not isinstance(item, Mapping) for item in attempts):
            raise ValueError("successor admission attempts must contain objects")
        normalized["attempts"] = [
            dict(item) for item in attempts
        ][-16:]
        result[canonical_id] = normalized
    return result


def filter_self_improvement_successor_candidates(
    proposals: Iterable[ObjectiveWorkProposal | Mapping[str, Any]],
    *,
    objective_text: str,
    strategy: Mapping[str, Any],
    observed_at: datetime | str | None = None,
) -> SelfImprovementSuccessorFilterResult:
    """Exclude lifecycle, admission-history, cooldown, and batch duplicates.

    This is a pre-admission filter, not materialization authority.  Callers
    must still run the objective quality/refinement preview and commit its
    immutable result through the objective materialization transaction.
    """

    now = _self_improvement_successor_timestamp(
        observed_at, field_name="observed_at"
    )
    lifecycle_canonical, lifecycle_semantic = (
        self_improvement_successor_lifecycle_identities(objective_text)
    )
    records = self_improvement_successor_admission_records(strategy)
    permanent_statuses = {"admitted", "committed", "materialized"}
    ledger_canonical: set[str] = set()
    ledger_semantic: set[str] = set()
    active_cooldown_canonical: set[str] = set()
    active_cooldown_semantic: set[str] = set()
    for record in records.values():
        canonical_id = str(record["canonical_id"])
        semantic_key = str(record["semantic_key"])
        status = str(record["status"])
        cooldown_until = parse_iso_timestamp(
            str(record.get("cooldown_until") or "")
        )
        if status in permanent_statuses:
            ledger_canonical.add(canonical_id)
            ledger_semantic.add(semantic_key)
        elif cooldown_until is not None and now < cooldown_until:
            active_cooldown_canonical.add(canonical_id)
            active_cooldown_semantic.add(semantic_key)

    eligible: list[ObjectiveWorkProposal] = []
    rejected: list[SelfImprovementSuccessorRejection] = []
    batch_canonical: set[str] = set()
    batch_semantic: set[str] = set()
    normalized: list[ObjectiveWorkProposal] = []
    for raw in proposals:
        try:
            proposal = (
                raw
                if isinstance(raw, ObjectiveWorkProposal)
                else ObjectiveWorkProposal.from_dict(raw)
            )
        except (TypeError, ValueError) as exc:
            rejected.append(
                SelfImprovementSuccessorRejection(
                    canonical_id="",
                    semantic_key="",
                    reason="invalid_proposal",
                    detail=str(exc),
                )
            )
            continue
        normalized.append(proposal)
    normalized.sort(
        key=lambda item: (
            item.depth,
            item.parent_goal_id.casefold(),
            item.semantic_key,
            item.canonical_id,
        )
    )
    for proposal in normalized:
        reason = ""
        detail = ""
        if (
            proposal.canonical_id in lifecycle_canonical
            or proposal.semantic_key in lifecycle_semantic
        ):
            reason = "lifecycle_duplicate"
            detail = "equivalent work exists in the objective heap"
        elif (
            proposal.canonical_id in ledger_canonical
            or proposal.semantic_key in ledger_semantic
        ):
            reason = "prior_admission_duplicate"
            detail = "equivalent work has a durable successful admission record"
        elif (
            proposal.canonical_id in active_cooldown_canonical
            or proposal.semantic_key in active_cooldown_semantic
        ):
            reason = "successor_cooldown"
            detail = "equivalent work is inside its durable cooldown window"
        elif (
            proposal.canonical_id in batch_canonical
            or proposal.semantic_key in batch_semantic
        ):
            reason = "batch_duplicate"
            detail = "equivalent work already appeared in this candidate batch"
        if reason:
            rejected.append(
                SelfImprovementSuccessorRejection(
                    canonical_id=proposal.canonical_id,
                    semantic_key=proposal.semantic_key,
                    reason=reason,
                    detail=detail,
                )
            )
            continue
        batch_canonical.add(proposal.canonical_id)
        batch_semantic.add(proposal.semantic_key)
        eligible.append(proposal)
    return SelfImprovementSuccessorFilterResult(
        eligible=tuple(eligible),
        rejected=tuple(rejected),
        lifecycle_canonical_ids=tuple(sorted(lifecycle_canonical)),
        lifecycle_semantic_keys=tuple(sorted(lifecycle_semantic)),
        cooldown_canonical_ids=tuple(
            sorted(active_cooldown_canonical)
        ),
        cooldown_semantic_keys=tuple(
            sorted(active_cooldown_semantic)
        ),
    )


def record_self_improvement_successor_admission(
    strategy_path: Path,
    *,
    epoch_id: str,
    proposals: Iterable[ObjectiveWorkProposal | Mapping[str, Any]],
    admitted_proposal_ids: Sequence[str] = (),
    transaction_id: str = "",
    rejection_reasons: Mapping[str, Sequence[str] | str] | None = None,
    recorded_at: datetime | str | None = None,
    cooldown_seconds: int = DEFAULT_SELF_IMPROVEMENT_SUCCESSOR_COOLDOWN_SECONDS,
    record_limit: int = DEFAULT_SELF_IMPROVEMENT_SUCCESSOR_RECORD_LIMIT,
) -> dict[str, Any]:
    """Durably record committed admissions and finite rejected-work cooldowns.

    The update is locked and durably flushed with the rest of the strategy.
    An admitted record requires the objective transaction identity; recording
    it before commit is therefore impossible through this API.  Expired
    non-admission records are pruned before the hard ledger bound is applied.
    """

    epoch = str(epoch_id or "").strip()
    transaction = str(transaction_id or "").strip()
    if not epoch:
        raise ValueError("epoch_id is required")
    if (
        isinstance(cooldown_seconds, bool)
        or int(cooldown_seconds) < 0
    ):
        raise ValueError("cooldown_seconds must be a non-negative integer")
    if isinstance(record_limit, bool) or int(record_limit) <= 0:
        raise ValueError("record_limit must be a positive integer")
    now = _self_improvement_successor_timestamp(
        recorded_at, field_name="recorded_at"
    )
    normalized: dict[str, ObjectiveWorkProposal] = {}
    for raw in proposals:
        proposal = (
            raw
            if isinstance(raw, ObjectiveWorkProposal)
            else ObjectiveWorkProposal.from_dict(raw)
        )
        prior = normalized.get(proposal.canonical_id)
        if prior is not None and prior.semantic_key != proposal.semantic_key:
            raise ValueError("canonical proposal identity collision")
        normalized[proposal.canonical_id] = proposal
    admitted = {
        str(item).strip() for item in admitted_proposal_ids if str(item).strip()
    }
    unknown_admissions = admitted - set(normalized)
    if unknown_admissions:
        raise ValueError(
            "admitted proposal IDs were not present in the candidate set: "
            + ", ".join(sorted(unknown_admissions))
        )
    if admitted and not transaction:
        raise ValueError(
            "transaction_id is required for admitted successor proposals"
        )
    reasons_by_id: dict[str, list[str]] = {}
    for canonical_id, raw_reasons in (rejection_reasons or {}).items():
        values = (
            (raw_reasons,)
            if isinstance(raw_reasons, str)
            else tuple(raw_reasons)
        )
        reasons_by_id[str(canonical_id)] = sorted(
            {
                str(item).strip()
                for item in values
                if str(item).strip()
            }
        )

    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    with locked_taskboard(strategy_path) as stream:
        raw_text = stream.read().strip()
        if raw_text:
            try:
                loaded = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "cannot update corrupt self-improvement strategy JSON"
                ) from exc
            if not isinstance(loaded, Mapping):
                raise ValueError(
                    "self-improvement strategy must contain a JSON object"
                )
            strategy = dict(loaded)
        else:
            strategy = {"blocked_tasks": []}
        records = self_improvement_successor_admission_records(strategy)
        permanent_statuses = {"admitted", "committed", "materialized"}
        retained: dict[str, dict[str, Any]] = {}
        for canonical_id, record in records.items():
            cooldown_until = parse_iso_timestamp(
                str(record.get("cooldown_until") or "")
            )
            if (
                str(record.get("status") or "") in permanent_statuses
                or (cooldown_until is not None and now < cooldown_until)
            ):
                retained[canonical_id] = record
        for canonical_id, proposal in sorted(normalized.items()):
            is_admitted = canonical_id in admitted
            status = "admitted" if is_admitted else "rejected"
            reason_codes = (
                []
                if is_admitted
                else reasons_by_id.get(canonical_id, ["not_admitted"])
            )
            attempt = {
                "epoch_id": epoch,
                "transaction_id": transaction if is_admitted else "",
                "status": status,
                "recorded_at": now.isoformat(),
                "reason_codes": reason_codes,
            }
            prior = retained.get(canonical_id)
            prior_attempts = prior.get("attempts", ()) if prior else ()
            if prior is not None and not is_admitted:
                # Never downgrade committed authority or perpetually extend an
                # existing cooldown merely because another epoch proposed the
                # same work.  The attempt is still auditable.
                retained[canonical_id] = {
                    **prior,
                    "attempts": [*prior_attempts, attempt][-16:],
                }
                continue
            retained[canonical_id] = {
                "schema": SELF_IMPROVEMENT_SUCCESSOR_RECORD_SCHEMA,
                "version": 1,
                "canonical_id": canonical_id,
                "semantic_key": proposal.semantic_key,
                "status": status,
                "epoch_id": epoch,
                "transaction_id": transaction if is_admitted else "",
                "recorded_at": now.isoformat(),
                "cooldown_until": (
                    ""
                    if is_admitted
                    else (
                        now + timedelta(seconds=int(cooldown_seconds))
                    ).isoformat()
                ),
                "reason_codes": reason_codes,
                "attempts": [*prior_attempts, attempt][-16:],
            }
        if len(retained) > int(record_limit):
            raise RuntimeError(
                "self-improvement successor admission ledger limit reached; "
                "refusing to discard live deduplication authority"
            )
        strategy[SELF_IMPROVEMENT_SUCCESSOR_RECORDS_KEY] = {
            key: retained[key] for key in sorted(retained)
        }
        strategy["last_self_improvement_successor_admission_at"] = (
            now.isoformat()
        )
        strategy["last_self_improvement_successor_epoch_id"] = epoch
        if admitted:
            strategy["last_self_improvement_successor_transaction_id"] = (
                transaction
            )
        replace_locked_taskboard(
            stream,
            json.dumps(strategy, indent=2, sort_keys=True) + "\n",
        )
    return strategy


def should_refill_backlog(
    *,
    todo_text: str,
    state_path: Path | None,
    strategy: Mapping[str, Any],
    last_scan_key: str,
    last_drained_scan_task_count_key: str,
    task_prefix: str,
    min_open_tasks: int,
    cooldown_seconds: int,
    force: bool = False,
) -> tuple[bool, str, int, int]:
    current_open = effective_open_task_count(todo_text, state_path=state_path, task_prefix=task_prefix)
    task_count = len(task_ids_from_todo_text(todo_text, task_prefix=task_prefix))
    state_counts = refill_state_counts(todo_text, state_path=state_path, task_prefix=task_prefix)
    eligible_ready_for_refill = int(state_counts.get("eligible_ready_count", state_counts.get("ready_count") or 0) or 0)
    ready_for_refill = int(state_counts.get("selectable_ready_count", eligible_ready_for_refill) or 0)
    no_ready_existing_work = (
        bool(state_counts)
        and ready_for_refill == 0
        and int(state_counts.get("completed_count") or 0) > 0
        and (int(state_counts.get("waiting_count") or 0) > 0 or int(state_counts.get("blocked_count") or 0) > 0)
    )
    if force:
        return True, "force", current_open, task_count
    if current_open > min_open_tasks and not no_ready_existing_work:
        return False, "open_task_threshold", current_open, task_count
    drained = current_open == 0
    try:
        last_drained_count = int(strategy.get(last_drained_scan_task_count_key) or -1)
    except (TypeError, ValueError):
        last_drained_count = -1
    if drained and last_drained_count != task_count:
        return True, "drained_exhaustive", current_open, task_count
    if no_ready_existing_work and last_drained_count != task_count:
        return True, "runnable_drained_exhaustive", current_open, task_count
    last_scan_at = parse_iso_timestamp(str(strategy.get(last_scan_key) or ""))
    if last_scan_at is None:
        return True, "runnable_drained_low_backlog" if no_ready_existing_work else "low_backlog", current_open, task_count
    elapsed = (datetime.now(timezone.utc) - last_scan_at).total_seconds()
    if elapsed >= cooldown_seconds:
        return True, "runnable_drained_low_backlog" if no_ready_existing_work else "low_backlog", current_open, task_count
    return False, "cooldown", current_open, task_count


def refill_open_task_capacity(
    *,
    current_open: int,
    min_open_tasks: int,
    max_findings: int,
    headroom: int = DEFAULT_REFILL_OPEN_TASK_HEADROOM,
) -> int:
    """Bound one refill so generated work cannot create unbounded pressure.

    The existing low-watermark behavior scans when the board is at or below
    ``min_open_tasks``.  One item of headroom prevents refill thrashing at the
    exact watermark while still placing a hard ceiling on newly opened work.
    """

    target = max(0, int(min_open_tasks)) + max(0, int(headroom))
    available = max(0, target - max(0, int(current_open)))
    return min(max(0, int(max_findings)), available)


def self_improvement_epoch_wait_active(
    strategy: Mapping[str, Any],
    *,
    epoch_id: str,
    evidence_id: str = "",
    requirement_id: str = "",
    next_triggers: Sequence[str] = (),
) -> bool:
    """Return whether a proved healthy epoch suppresses an identical refill.

    A timestamp or empty finding list is deliberately insufficient.  The
    strategy must name the exact content-addressed epoch, its healthy
    exhaustion evidence, and the explicit wait state written after all proof
    gates passed.
    """

    expected = str(epoch_id or "").strip()
    if not expected:
        return False
    recorded_evidence = str(
        strategy.get("last_self_improvement_exhaustion_evidence_id") or ""
    ).strip()
    recorded_requirement = str(
        strategy.get("last_self_improvement_requirement_id") or ""
    ).strip()
    raw_quorum = strategy.get("last_self_improvement_exhaustion_quorum")
    try:
        quorum = (
            ExhaustionQuorumResult.from_dict(raw_quorum)
            if isinstance(raw_quorum, Mapping)
            else None
        )
    except (TypeError, ValueError):
        quorum = None
    recorded_triggers = tuple(
        sorted(
            str(item).strip()
            for item in (
                strategy.get("self_improvement_next_triggers") or ()
            )
            if str(item).strip()
        )
    )
    expected_triggers = tuple(
        sorted(str(item).strip() for item in next_triggers if str(item).strip())
    )
    return bool(
        str(strategy.get("last_self_improvement_epoch_id") or "") == expected
        and str(strategy.get("last_self_improvement_epoch_status") or "")
        == "healthy_exhausted"
        and str(strategy.get("self_improvement_refill_state") or "")
        == "waiting_for_meaningful_trigger"
        and recorded_evidence
        and recorded_requirement
        and quorum is not None
        and quorum.satisfied
        and recorded_triggers
        and (
            not str(evidence_id or "").strip()
            or recorded_evidence == str(evidence_id).strip()
        )
        and (
            not str(requirement_id or "").strip()
            or recorded_requirement == str(requirement_id).strip()
        )
        and (not expected_triggers or recorded_triggers == expected_triggers)
    )


def record_self_improvement_exhaustion(
    strategy_path: Path,
    *,
    epoch_id: str,
    evidence_id: str,
    requirement_id: str,
    quorum: Mapping[str, Any],
    next_triggers: Sequence[str],
    recorded_at: str,
) -> dict[str, Any]:
    """Persist the supervisor wait state after a qualified healthy epoch.

    This helper does not decide that exhaustion is healthy; the typed
    self-improvement witness owns that decision.  It accepts only a satisfied
    quorum and non-empty content identities so a generic empty objective scan
    cannot advance the drained marker.
    """

    epoch = str(epoch_id or "").strip()
    evidence = str(evidence_id or "").strip()
    requirement = str(requirement_id or "").strip()
    triggers = tuple(
        dict.fromkeys(str(item).strip() for item in next_triggers if str(item).strip())
    )
    if not epoch or not evidence or not requirement:
        raise ValueError(
            "epoch_id, evidence_id, and requirement_id are required"
        )
    try:
        parsed_quorum = (
            ExhaustionQuorumResult.from_dict(quorum)
            if isinstance(quorum, Mapping)
            else None
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("a valid exhaustion quorum is required") from exc
    if parsed_quorum is None or not parsed_quorum.satisfied:
        raise ValueError("a satisfied exhaustion quorum is required")
    if not triggers:
        raise ValueError("at least one meaningful next trigger is required")
    strategy = load_strategy(strategy_path)
    strategy.update(
        {
            "last_self_improvement_epoch_id": epoch,
            "last_self_improvement_epoch_status": "healthy_exhausted",
            "last_self_improvement_exhaustion_evidence_id": evidence,
            "last_self_improvement_requirement_id": requirement,
            "last_self_improvement_exhaustion_quorum": parsed_quorum.to_dict(),
            "last_self_improvement_exhausted_at": str(recorded_at or utc_now()),
            "self_improvement_refill_state": "waiting_for_meaningful_trigger",
            "self_improvement_next_triggers": list(triggers),
        }
    )
    write_json(strategy_path, strategy)
    return strategy


def git_toplevel_for_path(cwd: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return Path(result.stdout.strip()).resolve()


def path_status(repo: Path, relative: str) -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", relative],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def unmerged_worktree_paths(repo: Path) -> set[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def commit_specific_path(repo: Path, relative: str, *, subject: str) -> dict[str, Any]:
    if not repo_relative_path_safe(relative):
        return {"committed": False, "reason": "unsafe_path", "repo": str(repo), "path": relative}
    unmerged = unmerged_worktree_paths(repo)
    if unmerged and relative not in unmerged:
        return {
            "committed": False,
            "reason": "repo_has_unrelated_unmerged_paths",
            "repo": str(repo),
            "path": relative,
            "unmerged_paths": sorted(unmerged),
        }
    status = path_status(repo, relative)
    if not status:
        return {"committed": False, "reason": "no_changes", "repo": str(repo), "path": relative}
    add = subprocess.run(["git", "add", "--", relative], cwd=repo, text=True, capture_output=True, check=False)
    if add.returncode != 0:
        return {
            "committed": False,
            "reason": "git_add_failed",
            "repo": str(repo),
            "path": relative,
            "returncode": add.returncode,
            "stdout": add.stdout[-4000:],
            "stderr": add.stderr[-4000:],
        }
    staged = subprocess.run(["git", "diff", "--cached", "--quiet", "--", relative], cwd=repo, check=False)
    if staged.returncode == 0:
        return {"committed": False, "reason": "no_staged_changes", "repo": str(repo), "path": relative}
    commit = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Accelerator Backlog Refinery",
            "-c",
            f"user.email={BACKLOG_REFINERY_AUTHOR_EMAIL}",
            "commit",
            "-m",
            subject,
            "--",
            relative,
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if commit.returncode != 0:
        return {
            "committed": False,
            "reason": "git_commit_failed",
            "repo": str(repo),
            "path": relative,
            "returncode": commit.returncode,
            "stdout": commit.stdout[-4000:],
            "stderr": commit.stderr[-4000:],
        }
    ref = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=False)
    return {"committed": True, "repo": str(repo), "path": relative, "commit": ref.stdout.strip(), "status": status}


def parent_git_toplevel_for_repo(repo: Path) -> Path | None:
    parent = git_toplevel_for_path(repo.resolve().parent)
    if parent is None or parent.resolve() == repo.resolve():
        return None
    try:
        repo.resolve().relative_to(parent.resolve())
    except ValueError:
        return None
    return parent


def commit_parent_gitlink_updates(child_repo: Path, *, repo_root: Path, subject: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    current = child_repo.resolve()
    root = repo_root.resolve()
    while current != root:
        parent = parent_git_toplevel_for_repo(current)
        if parent is None:
            break
        relative = repo_relative_path(parent, current)
        if not relative:
            break
        results.append(commit_specific_path(parent, relative, subject=subject))
        current = parent.resolve()
    return results


def commit_generated_outputs(paths: Sequence[Path], *, repo_root: Path, subject: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in paths:
        repo = git_toplevel_for_path(path.parent)
        if repo is None:
            results.append({"committed": False, "reason": "not_in_git_repo", "path": str(path)})
            continue
        relative = repo_relative_path(repo, path)
        if not relative:
            results.append({"committed": False, "reason": "path_outside_repo", "path": str(path), "repo": str(repo)})
            continue
        result = commit_specific_path(repo, relative, subject=subject)
        if result.get("committed"):
            parent_results = commit_parent_gitlink_updates(repo, repo_root=repo_root, subject=subject)
            if parent_results:
                result["parent_gitlink_commits"] = parent_results
        results.append(result)
    return results


def git_status_porcelain(repo: Path) -> list[str]:
    """Return short porcelain status lines, including untracked files."""

    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line.rstrip() for line in result.stdout.splitlines() if line.strip()]


def git_dir_for_repo(repo: Path) -> Path | None:
    """Return the resolved git metadata directory for a worktree."""

    result = subprocess.run(
        ["git", "rev-parse", "--git-dir"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    git_dir = Path(result.stdout.strip())
    if not git_dir.is_absolute():
        git_dir = repo / git_dir
    return git_dir.resolve()


def git_index_lock_path(repo: Path) -> Path | None:
    git_dir = git_dir_for_repo(repo)
    if git_dir is None:
        return None
    return git_dir / "index.lock"


def git_merge_head_path(repo: Path) -> Path | None:
    git_dir = git_dir_for_repo(repo)
    if git_dir is None:
        return None
    return git_dir / "MERGE_HEAD"


def generated_dirty_commit_blocker(repo: Path) -> dict[str, Any] | None:
    """Return a checkout state that should defer generated-output commits."""

    merge_head = git_merge_head_path(repo)
    if merge_head is not None and merge_head.exists():
        return {
            "repo": str(repo),
            "reason": "repo_merge_in_progress",
            "merge_head_path": str(merge_head),
        }

    lock_path = checkout_mutation_lock_path(repo)
    if lock_path.exists():
        try:
            lock_metadata = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            lock_metadata = {}
        try:
            lock_pid = int(
                lock_metadata.get("pid") or 0
                if isinstance(lock_metadata, Mapping)
                else 0
            )
        except (TypeError, ValueError):
            lock_pid = 0
        lock_worktree_root = (
            str(
                lock_metadata.get("worktree_root")
                or lock_metadata.get("repo_root")
                or ""
            ).strip()
            if isinstance(lock_metadata, Mapping)
            else ""
        )
        owned_generated_repair = (
            isinstance(lock_metadata, Mapping)
            and str(lock_metadata.get("kind") or "") == "merge"
            and str(lock_metadata.get("operation") or "")
            == "generated_dirty_repair"
            and lock_pid == os.getpid()
            and bool(lock_worktree_root)
            and Path(lock_worktree_root).resolve() == repo.resolve()
        )
        if owned_generated_repair:
            return None
        return {
            "repo": str(repo),
            "reason": "checkout_mutation_lock_exists",
            "lock_path": str(lock_path),
        }
    return None


def _path_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except (OSError, ValueError):
        return False
    return True


def _cmdline_has_git_process(cmdline: str) -> bool:
    if not cmdline:
        return False
    for token in cmdline.replace("\x00", " ").split():
        name = Path(token).name
        if name == "git" or name.startswith("git-"):
            return True
    return False


def active_git_processes_for_repo(repo: Path, git_dir: Path) -> list[dict[str, Any]]:
    """Best-effort check for active git processes that could own a lock."""

    proc_root = Path("/proc")
    if not proc_root.exists():
        return []
    repo = repo.resolve()
    git_dir = git_dir.resolve()
    active: list[dict[str, Any]] = []
    current_pid = os.getpid()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        try:
            raw_cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        cmdline = raw_cmdline.decode("utf-8", errors="replace").replace("\x00", " ").strip()
        if not _cmdline_has_git_process(cmdline):
            continue
        cwd_text = ""
        try:
            cwd = Path(os.readlink(entry / "cwd")).resolve()
            cwd_text = str(cwd)
        except OSError:
            cwd = None
        mentions_repo = str(repo) in cmdline or str(git_dir) in cmdline
        cwd_matches = bool(cwd and (_path_inside(cwd, repo) or _path_inside(cwd, git_dir)))
        if mentions_repo or cwd_matches:
            active.append({"pid": pid, "cwd": cwd_text, "cmdline": cmdline[:500]})
    return active


def repair_stale_git_index_lock(
    repo: Path,
    *,
    stale_seconds: float = DEFAULT_STALE_GIT_LOCK_SECONDS,
) -> dict[str, Any]:
    """Remove an inactive stale ``index.lock`` for one git worktree.

    Git leaves ``index.lock`` behind when an add/commit process crashes. The
    supervisor can safely remove it only when the lock is old enough and there
    is no active git process associated with the same worktree/git directory.
    """

    repo = repo.resolve()
    lock_path = git_index_lock_path(repo)
    if lock_path is None:
        return {"attempted": False, "repo": str(repo), "reason": "not_git_repo"}
    if not lock_path.exists():
        return {"attempted": False, "repo": str(repo), "lock_path": str(lock_path), "reason": "no_lock"}
    try:
        stat = lock_path.stat()
    except OSError as exc:
        return {
            "attempted": True,
            "repo": str(repo),
            "lock_path": str(lock_path),
            "removed": False,
            "reason": "lock_stat_failed",
            "error": str(exc),
        }
    age_seconds = max(0.0, time.time() - stat.st_mtime)
    git_dir = lock_path.parent
    active_processes = active_git_processes_for_repo(repo, git_dir)
    if active_processes:
        return {
            "attempted": True,
            "repo": str(repo),
            "lock_path": str(lock_path),
            "removed": False,
            "reason": "active_git_process",
            "age_seconds": age_seconds,
            "active_processes": active_processes[:10],
        }
    if age_seconds < float(stale_seconds):
        return {
            "attempted": True,
            "repo": str(repo),
            "lock_path": str(lock_path),
            "removed": False,
            "reason": "lock_not_stale",
            "age_seconds": age_seconds,
            "stale_seconds": stale_seconds,
        }
    try:
        lock_path.unlink()
    except OSError as exc:
        return {
            "attempted": True,
            "repo": str(repo),
            "lock_path": str(lock_path),
            "removed": False,
            "reason": "lock_unlink_failed",
            "age_seconds": age_seconds,
            "error": str(exc),
        }
    return {
        "attempted": True,
        "repo": str(repo),
        "lock_path": str(lock_path),
        "removed": True,
        "reason": "stale_lock_removed",
        "age_seconds": age_seconds,
    }


def _resolve_existing_path_for_git_root(path: Path) -> Path:
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def _relative_filter_for_git_root(
    relative: str,
    *,
    repo_root: Path,
    git_root: Path,
) -> str:
    path_text = normalize_status_path(relative)
    if not path_text:
        return ""
    try:
        full_path = (repo_root / path_text).resolve()
        root = git_root.resolve()
    except OSError:
        return ""
    if full_path == root:
        return ""
    try:
        return full_path.relative_to(root).as_posix()
    except ValueError:
        return ""


def generated_status_filters_for_git_root(
    *,
    repo_root: Path,
    git_root: Path,
    generated_paths: Sequence[str] = (),
    generated_prefixes: Sequence[str] = (),
) -> tuple[list[str], list[str]]:
    """Convert repo-root-relative generated filters to one git root."""

    if git_root.resolve() == repo_root.resolve():
        return (
            [normalize_status_path(path) for path in generated_paths if normalize_status_path(path)],
            [normalize_status_path(path) for path in generated_prefixes if normalize_status_path(path)],
        )
    return (
        list(
            dict.fromkeys(
                rel
                for rel in (
                    _relative_filter_for_git_root(path, repo_root=repo_root, git_root=git_root)
                    for path in generated_paths
                )
                if rel
            )
        ),
        list(
            dict.fromkeys(
                rel
                for rel in (
                    _relative_filter_for_git_root(path, repo_root=repo_root, git_root=git_root)
                    for path in generated_prefixes
                )
                if rel
            )
        ),
    )


def _git_root_candidates_for_dirty_generated_outputs(
    *,
    repo_root: Path,
    generated_paths: Sequence[str],
    generated_prefixes: Sequence[str],
    candidate_git_roots: Sequence[Path | str],
) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()

    def add(candidate: Path) -> None:
        top = git_toplevel_for_path(_resolve_existing_path_for_git_root(candidate))
        if top is None:
            return
        try:
            top.resolve().relative_to(repo_root.resolve())
        except ValueError:
            return
        key = str(top.resolve())
        if key not in seen:
            seen.add(key)
            roots.append(top.resolve())

    add(repo_root)
    for candidate in candidate_git_roots:
        add(repo_root / candidate if not Path(candidate).is_absolute() else Path(candidate))
    for relative in [*generated_paths, *generated_prefixes]:
        path_text = normalize_status_path(relative)
        if path_text:
            add(repo_root / path_text)
    for submodule_root in _initialized_submodule_git_roots(repo_root):
        add(submodule_root)
    return sorted(roots, key=lambda path: len(path.resolve().parts), reverse=True)


def _initialized_submodule_git_roots(repo_root: Path) -> list[Path]:
    """Return initialized submodule worktree roots under ``repo_root``."""

    result = subprocess.run(
        ["git", "submodule", "foreach", "--quiet", "--recursive", "pwd"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    roots: list[Path] = []
    for line in result.stdout.splitlines():
        path_text = line.strip()
        if not path_text:
            continue
        path = Path(path_text)
        if not path.is_absolute():
            path = repo_root / path
        roots.append(path)
    return roots


def _path_is_gitlink(repo: Path, relative: str) -> bool:
    if not repo_relative_path_safe(relative):
        return False
    result = subprocess.run(
        ["git", "ls-files", "--stage", "--", relative],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    return any(line.startswith("160000 ") for line in result.stdout.splitlines())


def _clean_child_git_root(repo: Path, relative: str) -> str:
    child = repo / relative
    child_root = git_toplevel_for_path(child)
    if child_root is None:
        return ""
    if git_status_porcelain(child_root):
        return ""
    return str(child_root)


def _status_line_is_clean_gitlink_update(repo: Path, line: str) -> tuple[bool, str]:
    code = line[:2]
    relative = status_line_path(line)
    if not relative or "U" in code or "R" in code or "C" in code:
        return False, ""
    if code == "??" or not _path_is_gitlink(repo, relative):
        return False, ""
    child_root = _clean_child_git_root(repo, relative)
    return bool(child_root), child_root


def _commit_selected_dirty_paths(
    repo: Path,
    paths: Sequence[str],
    *,
    subject: str,
    protected_board_paths: Sequence[str] = (),
) -> dict[str, Any]:
    selected_paths = [path for path in dict.fromkeys(paths) if repo_relative_path_safe(path)]
    if not selected_paths:
        return {
            "committed": False,
            "reason": "no_safe_paths",
            "repo": str(repo),
            "selected_paths": [],
        }
    add = subprocess.run(
        ["git", "add", "--", *selected_paths],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if add.returncode != 0:
        return {
            "committed": False,
            "reason": "git_add_failed",
            "repo": str(repo),
            "selected_paths": selected_paths,
            "returncode": add.returncode,
            "stdout": add.stdout[-4000:],
            "stderr": add.stderr[-4000:],
        }
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", *selected_paths],
        cwd=repo,
        check=False,
    )
    if staged.returncode == 0:
        return {
            "committed": False,
            "reason": "no_staged_changes",
            "repo": str(repo),
            "selected_paths": selected_paths,
        }
    protected_selected_paths = sorted(
        set(selected_paths).intersection(
            {
                normalize_status_path(path)
                for path in protected_board_paths
                if normalize_status_path(path)
            }
        )
    )
    protected_board_commit = bool(protected_selected_paths)
    commit_subject = (
        generated_protected_board_commit_subject(subject)
        if protected_board_commit
        else subject
    )
    author_name = (
        "Accelerator Backlog Refinery"
        if protected_board_commit
        else "Agent Supervisor"
    )
    author_email = (
        BACKLOG_REFINERY_AUTHOR_EMAIL
        if protected_board_commit
        else "agent-supervisor@example.invalid"
    )
    commit = subprocess.run(
        [
            "git",
            "-c",
            f"user.name={author_name}",
            "-c",
            f"user.email={author_email}",
            "commit",
            "-m",
            commit_subject,
            "--",
            *selected_paths,
        ],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if commit.returncode != 0:
        return {
            "committed": False,
            "reason": "git_commit_failed",
            "repo": str(repo),
            "selected_paths": selected_paths,
            "returncode": commit.returncode,
            "stdout": commit.stdout[-4000:],
            "stderr": commit.stderr[-4000:],
            "protected_board_paths": protected_selected_paths,
            "protected_board_commit": protected_board_commit,
        }
    ref = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=False)
    return {
        "committed": True,
        "repo": str(repo),
        "selected_paths": selected_paths,
        "commit": ref.stdout.strip(),
        "stdout": commit.stdout[-4000:],
        "author_email": author_email,
        "subject": commit_subject,
        "protected_board_paths": protected_selected_paths,
        "protected_board_commit": protected_board_commit,
    }


def commit_generated_dirty_outputs(
    *,
    repo_root: Path,
    generated_paths: Sequence[str] = (),
    generated_prefixes: Sequence[str] = (),
    protected_paths: Sequence[str] = (),
    candidate_git_roots: Sequence[Path | str] = (),
    subject: str = "Agent: commit generated supervisor outputs",
    include_clean_submodule_gitlinks: bool = True,
    max_paths: int = 200,
    stale_git_lock_seconds: float = DEFAULT_STALE_GIT_LOCK_SECONDS,
) -> dict[str, Any]:
    """Commit safe supervisor-generated dirt across nested git roots.

    The repair is deliberately conservative: it stages only paths matching the
    generated-output filters, plus clean submodule gitlink pointer updates when
    requested. Unknown dirty files are reported but left untouched.
    """

    repo_root = repo_root.resolve()
    roots = _git_root_candidates_for_dirty_generated_outputs(
        repo_root=repo_root,
        generated_paths=generated_paths,
        generated_prefixes=generated_prefixes,
        candidate_git_roots=candidate_git_roots,
    )
    hard_path_cap = max(1, int(DEFAULT_GENERATED_DIRTY_HARD_PATH_CAP))
    configured_budget = max(0, int(max_paths))
    remaining_budget = min(configured_budget, hard_path_cap)
    max_delete_paths = max(0, int(DEFAULT_GENERATED_DIRTY_MAX_DELETE_PATHS))
    allow_generated_deletions = bool(DEFAULT_GENERATED_DIRTY_ALLOW_DELETIONS)
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    lock_repairs: list[dict[str, Any]] = []
    selected_path_count = 0
    for git_root in roots:
        lock_repair = repair_stale_git_index_lock(
            git_root,
            stale_seconds=stale_git_lock_seconds,
        )
        if lock_repair.get("attempted"):
            lock_repairs.append(lock_repair)
        if lock_repair.get("attempted") and not lock_repair.get("removed"):
            skipped.append(
                {
                    "repo": str(git_root),
                    "reason": str(lock_repair.get("reason") or "git_index_lock_blocked"),
                    "lock_repair": lock_repair,
                }
            )
            continue
        commit_blocker = generated_dirty_commit_blocker(git_root)
        if commit_blocker is not None:
            skipped.append(commit_blocker)
            continue
        status = git_status_porcelain(git_root)
        if not status:
            continue
        status_codes: dict[str, str] = {}
        for line in status:
            relative = status_line_path(line)
            if relative and relative not in status_codes:
                status_codes[relative] = line[:2]
        unmerged = sorted(unmerged_worktree_paths(git_root))
        if unmerged:
            skipped.append(
                {
                    "repo": str(git_root),
                    "reason": "repo_has_unmerged_paths",
                    "unmerged_paths": unmerged[:50],
                }
            )
            continue
        repo_generated_paths, repo_generated_prefixes = generated_status_filters_for_git_root(
            repo_root=repo_root,
            git_root=git_root,
            generated_paths=generated_paths,
            generated_prefixes=generated_prefixes,
        )
        repo_protected_paths, _ = generated_status_filters_for_git_root(
            repo_root=repo_root,
            git_root=git_root,
            generated_paths=protected_paths,
        )
        selected: list[str] = []
        selected_reasons: dict[str, str] = {}
        for line in status:
            if remaining_budget <= 0:
                break
            code = line[:2]
            relative = status_line_path(line)
            if not relative or not repo_relative_path_safe(relative):
                continue
            if "U" in code or "R" in code or "C" in code:
                continue
            if path_is_generated_status_output(
                relative,
                generated_paths=repo_generated_paths,
                generated_prefixes=repo_generated_prefixes,
            ):
                if _path_is_gitlink(git_root, relative):
                    continue
                selected.append(relative)
                selected_reasons[relative] = "generated_output"
                remaining_budget -= 1
                continue
            if include_clean_submodule_gitlinks:
                gitlink, child_root = _status_line_is_clean_gitlink_update(git_root, line)
                if gitlink:
                    selected.append(relative)
                    selected_reasons[relative] = f"clean_submodule_gitlink:{child_root}"
                    remaining_budget -= 1
        selected_deletions = [
            relative
            for relative in selected
            if "D" in str(status_codes.get(relative) or "")
        ]
        if selected_deletions and not allow_generated_deletions:
            skipped.append(
                {
                    "repo": str(git_root),
                    "reason": "generated_deletions_blocked",
                    "selected_deletion_paths": selected_deletions[:50],
                    "status_short": status[:50],
                }
            )
            continue
        if len(selected_deletions) > max_delete_paths:
            skipped.append(
                {
                    "repo": str(git_root),
                    "reason": "generated_deletion_path_limit_exceeded",
                    "selected_deletion_count": len(selected_deletions),
                    "max_delete_paths": max_delete_paths,
                    "selected_deletion_paths": selected_deletions[:50],
                    "status_short": status[:50],
                }
            )
            continue
        if not selected:
            skipped.append(
                {
                    "repo": str(git_root),
                    "reason": "no_safe_dirty_paths",
                    "status_short": status[:50],
                }
            )
            continue
        result = _commit_selected_dirty_paths(
            git_root,
            selected,
            subject=subject,
            protected_board_paths=repo_protected_paths,
        )
        result["selected_reasons"] = selected_reasons
        result["status_short_before"] = status[:50]
        selected_path_count += len(selected)
        results.append(result)

    final_status = git_status_porcelain(repo_root)
    return {
        "attempted": True,
        "repo_root": str(repo_root),
        "git_root_count": len(roots),
        "selected_path_count": selected_path_count,
        "committed_count": sum(1 for item in results if item.get("committed")),
        "results": results,
        "lock_repairs": lock_repairs,
        "skipped": skipped[:50],
        "remaining_status_short": final_status[:50],
        "remaining_status_count": len(final_status),
        "max_paths": max_paths,
        "hard_path_cap": hard_path_cap,
        "effective_max_paths": min(max(0, int(max_paths)), hard_path_cap),
        "max_delete_paths": max_delete_paths,
        "allow_generated_deletions": allow_generated_deletions,
    }


def repo_relative_path_safe(relative: str) -> bool:
    if not relative or relative.startswith("/") or "\0" in relative:
        return False
    return ".." not in Path(relative).parts


def path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def discovery_output_path_for(
    repo_root: Path,
    discovery_dir: Path,
    *,
    default: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
) -> str:
    """Return a repo-relative discovery output path, or a stable fallback."""

    try:
        return discovery_dir.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return default


def task_dependencies_if_present(
    todo_path: Path,
    *,
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX,
    dependency_ids: Sequence[str] = (),
) -> list[str]:
    """Return dependency ids that are already declared in a todo board."""

    if not dependency_ids or not todo_path.exists():
        return []
    todo_text = todo_path.read_text(encoding="utf-8")
    task_prefix = task_id_prefix(task_header_prefix_value)
    declared_task_ids = set(task_ids_from_todo_text(todo_text, task_prefix=task_prefix))
    return [dependency_id for dependency_id in dependency_ids if dependency_id in declared_task_ids]


def codebase_scan_path_skipped(
    path: Path,
    *,
    repo_root: Path,
    skip_prefixes: Sequence[str] = CODEBASE_SCAN_SKIP_PREFIXES,
) -> bool:
    try:
        relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        relative = path.as_posix()
    if any(relative == prefix.rstrip("/") or relative.startswith(prefix) for prefix in skip_prefixes):
        return True
    return any(part in CODEBASE_SCAN_SKIP_PARTS for part in path.parts)


def discover_git_worktrees(
    repo_root: Path,
    *,
    skip_prefixes: Sequence[str] = CODEBASE_SCAN_SKIP_PREFIXES,
) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()

    def add_if_worktree(candidate: Path) -> None:
        top = git_toplevel_for_path(candidate)
        if top is None:
            return
        resolved = top.resolve()
        if not path_is_under(resolved, repo_root):
            return
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            roots.append(resolved)

    add_if_worktree(repo_root)
    for current, dirnames, _filenames in os.walk(repo_root):
        current_path = Path(current)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in CODEBASE_SCAN_SKIP_PARTS
            and not codebase_scan_path_skipped(current_path / dirname, repo_root=repo_root, skip_prefixes=skip_prefixes)
        ]
        if current_path != repo_root and (current_path / ".git").exists():
            add_if_worktree(current_path)
            dirnames[:] = []
    return roots


def expected_git_worktrees(
    repo_root: Path,
    *,
    skip_prefixes: Sequence[str] = CODEBASE_SCAN_SKIP_PREFIXES,
) -> list[Path]:
    """Inventory visible Git markers independently from root discovery.

    The independent marker walk is a canary for ``discover_git_worktrees``:
    equating expected roots with that function's result would make a discovery
    regression indistinguishable from a repository with no Git roots.
    """

    expected: list[Path] = []
    seen: set[str] = set()
    for current, dirnames, _filenames in os.walk(repo_root):
        current_path = Path(current)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in CODEBASE_SCAN_SKIP_PARTS
            and not codebase_scan_path_skipped(
                current_path / dirname,
                repo_root=repo_root,
                skip_prefixes=skip_prefixes,
            )
        ]
        marker = current_path / ".git"
        if marker.exists():
            resolved = current_path.resolve()
            key = str(resolved)
            if key not in seen:
                seen.add(key)
                expected.append(resolved)
            if current_path != repo_root:
                dirnames[:] = []
    return expected


def tracked_files(repo: Path) -> list[Path]:
    if not repo.is_dir():
        return []
    try:
        result = subprocess.run(["git", "ls-files", "-z"], cwd=repo, capture_output=True, check=False)
    except (FileNotFoundError, OSError):
        logger.debug("Skipping vanished git root during codebase scan: %s", repo)
        return []
    if result.returncode != 0:
        return []
    files: list[Path] = []
    for raw_path in result.stdout.split(b"\0"):
        if not raw_path:
            continue
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        if not repo_relative_path_safe(relative):
            continue
        path = repo / relative
        if path.is_file():
            files.append(path)
    return files


def root_relative_path(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def file_is_scan_candidate(
    path: Path,
    *,
    repo_root: Path,
    skip_prefixes: Sequence[str] = CODEBASE_SCAN_SKIP_PREFIXES,
    include_prefixes: Sequence[str] = (),
    allowed_tracks: Sequence[str] = (),
) -> bool:
    return not codebase_scan_file_exclusion_reason(
        path,
        repo_root=repo_root,
        skip_prefixes=skip_prefixes,
        include_prefixes=include_prefixes,
        allowed_tracks=allowed_tracks,
    )


def codebase_scan_file_exclusion_reason(
    path: Path,
    *,
    repo_root: Path,
    skip_prefixes: Sequence[str] = CODEBASE_SCAN_SKIP_PREFIXES,
    include_prefixes: Sequence[str] = (),
    allowed_tracks: Sequence[str] = (),
) -> str:
    """Return a stable, bounded reason code when ``path`` is ineligible."""

    try:
        relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
    except (OSError, ValueError):
        relative = path.as_posix()
    if any(relative == prefix.rstrip("/") or relative.startswith(prefix) for prefix in skip_prefixes):
        return "excluded_prefix"
    normalized_prefixes = tuple(
        str(prefix).strip().strip("/") for prefix in include_prefixes if str(prefix).strip().strip("/")
    )
    if normalized_prefixes and not any(
        relative == prefix or relative.startswith(f"{prefix}/")
        for prefix in normalized_prefixes
    ):
        return "outside_scope_prefix"
    normalized_tracks = {str(track).strip().lower() for track in allowed_tracks if str(track).strip()}
    if normalized_tracks and scan_track_for_path(relative) not in normalized_tracks:
        return "outside_scope_track"
    if any(part in CODEBASE_SCAN_SKIP_PARTS for part in path.parts):
        return "excluded_directory"
    if "-codebase-scan-" in path.name or "retry-budget" in path.name:
        return "generated_artifact"
    if path.name == "todo.md" or path.name.endswith(".todo.md"):
        return "todo_board"
    if path.suffix.lower() not in CODEBASE_SCAN_SUFFIXES:
        return "unsupported_suffix"
    try:
        stat = path.stat()
    except FileNotFoundError:
        return "missing_file"
    except OSError:
        return "stat_failed"
    if path.is_symlink() or not path.is_file():
        return "not_regular_file"
    if stat.st_size > CODEBASE_SCAN_MAX_FILE_BYTES:
        return "file_too_large"
    return ""


def scan_fingerprint(*, kind: str, root_relative_path: str, line_number: int, snippet: str) -> str:
    normalized = " ".join(snippet.strip().split())
    payload = f"{kind}\0{root_relative_path}\0{line_number}\0{normalized}"
    return sha1(payload.encode("utf-8")).hexdigest()


def scan_track_for_path(path: str) -> str:
    lowered = path.lower()
    if "/test/" in lowered or lowered.startswith("tests/") or "test_" in Path(lowered).name:
        return "quality"
    if "ui" in Path(lowered).parts or "frontend" in Path(lowered).parts:
        return "ui"
    if lowered.endswith((".md", ".rst")):
        return "docs"
    if lowered.endswith((".py", ".rs", ".sh")):
        return "runtime"
    return "ops"


def scan_validation_for_path(root_relative: str) -> str:
    quoted = shlex.quote(root_relative)
    suffix = Path(root_relative).suffix.lower()
    if suffix == ".py":
        return f"python3 -m py_compile {quoted}"
    if suffix == ".json":
        return f"python3 -m json.tool {quoted} >/dev/null"
    if suffix in {".yaml", ".yml"}:
        source = 'import pathlib, sys; p=pathlib.Path(sys.argv[1]); assert p.read_text(encoding="utf-8").strip()'
        return f"python3 -c {shlex.quote(source)} {quoted}"
    return f"test -f {quoted}"


GOAL_ALIGNMENT_STOPWORDS = frozenset(
    {
        "and",
        "code",
        "current",
        "data",
        "file",
        "for",
        "from",
        "goal",
        "implementation",
        "project",
        "repository",
        "state",
        "task",
        "test",
        "the",
        "with",
    }
)


def objective_goals_for_codebase_refill(objective_path: Path | None) -> list[ObjectiveGoal]:
    """Load objective nodes used for schedulable targets and their ancestry."""

    if objective_path is None or not objective_path.is_file():
        return []
    return parse_goal_heap(objective_path.read_text(encoding="utf-8"))


def codebase_refill_goal_graph_errors(
    goals: Sequence[ObjectiveGoal],
) -> tuple[Mapping[str, str], ...]:
    """Return structural defects that make objective lineage unsafe to emit."""

    errors: list[dict[str, str]] = []

    def add(reason_code: str, message: str) -> None:
        record = {"reason_code": reason_code, "message": message}
        if record not in errors:
            errors.append(record)

    goal_ids = [str(goal.goal_id).strip() for goal in goals]
    duplicates = sorted(
        goal_id
        for goal_id in set(goal_ids)
        if goal_id and goal_ids.count(goal_id) > 1
    )
    if duplicates:
        add(
            "invalid_goal_record",
            "duplicate objective goal ids: " + ", ".join(duplicates),
        )

    nodes: dict[str, ObjectiveGoal] = {}
    for goal in goals:
        goal_id = str(goal.goal_id).strip()
        if not goal_id or not str(goal.title).strip():
            add("invalid_goal_record", "objective goal records require an id and title")
            continue
        if not str(goal.fields.get("status") or "").strip():
            add(
                "invalid_goal_record",
                f"objective record {goal_id} has no explicit status",
            )
        else:
            try:
                goal.lifecycle_state
            except (TypeError, ValueError) as exc:
                add(
                    "invalid_goal_record",
                    f"objective record {goal_id} has invalid status: {exc}",
                )
        nodes.setdefault(goal_id, goal)

    missing_edges = sorted(
        {
            f"{goal.goal_id}->{parent_id}"
            for goal in goals
            for parent_id in goal.parent_goal_ids
            if parent_id and parent_id not in nodes
        }
    )
    if missing_edges:
        add(
            "dangling_goal_parent",
            "objective goal parents do not exist: " + ", ".join(missing_edges),
        )

    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(goal_id: str) -> None:
        current = state.get(goal_id, 0)
        if current == 2:
            return
        if current == 1:
            try:
                cycle_start = stack.index(goal_id)
            except ValueError:
                cycle_start = 0
            cycle = stack[cycle_start:] + [goal_id]
            add(
                "cyclic_goal_lineage",
                "objective goal parent cycle: " + " -> ".join(cycle),
            )
            return
        state[goal_id] = 1
        stack.append(goal_id)
        for parent_id in nodes[goal_id].parent_goal_ids:
            if parent_id in nodes:
                visit(parent_id)
        stack.pop()
        state[goal_id] = 2

    for goal_id in sorted(nodes):
        visit(goal_id)
    return tuple(errors)


def _goal_scope_path_matches(candidate_path: str, scope_path: str) -> bool:
    candidate = str(candidate_path).strip().strip("/")
    scope = str(scope_path).strip().strip("/").rstrip("*").rstrip("/")
    if not candidate or not scope:
        return False
    if candidate == scope:
        return True
    return candidate.startswith(f"{scope}/")


def _alignment_tokens(value: str) -> set[str]:
    aliases = {
        "doc": "documentation",
        "docs": "documentation",
        "documents": "documentation",
    }
    return {
        aliases.get(token, token)
        for token in re.findall(r"[a-z0-9][a-z0-9_+-]*", str(value).lower())
        if len(token) > 2 and token not in GOAL_ALIGNMENT_STOPWORDS
    }


def align_codebase_finding_to_goals(
    finding: CodebaseFinding,
    goals: Sequence[ObjectiveGoal],
    *,
    mission_terms: Sequence[str] = (),
) -> CodebaseFinding | None:
    """Bind a candidate to existing goals or reject it as scope creep.

    Declared goal outputs are authoritative.  Semantic matching is a fallback
    for goals without output paths and requires multiple distinctive terms, so
    a generic TODO/FIXME marker cannot create its own scope.  ``mission_terms``
    is retained for API compatibility but never expands a goal's lineage.
    """

    del mission_terms
    if codebase_refill_goal_graph_errors(goals):
        return None
    goals_by_id = {goal.goal_id: goal for goal in goals}

    def lineage(goal: ObjectiveGoal) -> tuple[str, ...]:
        ordered = [goal.goal_id]
        seen = {goal.goal_id}
        pending = list(goal.parent_goal_ids)
        while pending:
            parent_id = pending.pop(0)
            if parent_id in seen:
                continue
            seen.add(parent_id)
            ordered.append(parent_id)
            parent = goals_by_id.get(parent_id)
            if parent is not None:
                pending.extend(parent.parent_goal_ids)
        return tuple(ordered)

    def graph_depth(goal: ObjectiveGoal) -> int:
        return max(0, len(lineage(goal)) - 1)

    direct_matches: list[tuple[tuple[int, int, int, int, int], ObjectiveGoal]] = []
    semantic_matches: list[tuple[tuple[int, int], ObjectiveGoal]] = []
    candidate_tokens = _alignment_tokens(
        " ".join((finding.summary, finding.snippet))
    )
    for goal in goals:
        if not goal.is_schedulable:
            continue
        scope_paths = [*goal.predicted_files, *goal.required_evidence]
        matching_paths = [
            str(scope_path).strip().strip("/").rstrip("*").rstrip("/")
            for scope_path in scope_paths
            if _goal_scope_path_matches(finding.root_relative_path, scope_path)
        ]
        goal_text = " ".join(
            (
                goal.title,
                goal.fields.get("goal", ""),
                goal.fields.get("gap_task", ""),
                goal.fields.get("embedding_query", ""),
                goal.fields.get("ast_query", ""),
                goal.fields.get("acceptance", ""),
                goal.fields.get("acceptance_criteria", ""),
            )
        )
        token_overlap = len(candidate_tokens & _alignment_tokens(goal_text))
        if matching_paths:
            best_scope = max(
                matching_paths,
                key=lambda path: (len(Path(path).parts), len(path)),
            )
            broad_directory_scope = (
                len(Path(best_scope).parts) == 1
                and not Path(best_scope).suffix
            )
            if broad_directory_scope and token_overlap < 2:
                # A top-level directory such as ``scripts`` is an inventory
                # boundary, not proof that every file advances this goal.
                # Path tokens do not count: the finding itself must share at
                # least two distinctive terms with the declared goal.
                continue
            direct_matches.append(
                (
                    (
                        int(finding.root_relative_path.strip("/") == best_scope),
                        len(Path(best_scope).parts),
                        len(best_scope),
                        graph_depth(goal),
                        token_overlap,
                    ),
                    goal,
                )
            )
            continue
        if scope_paths:
            continue
        if token_overlap >= 3:
            semantic_matches.append(((token_overlap, graph_depth(goal)), goal))

    ranked: Sequence[tuple[tuple[int, ...], ObjectiveGoal]]
    ranked = direct_matches if direct_matches else semantic_matches
    if not ranked:
        return None
    best_score = max(score for score, _goal in ranked)
    best_goals = [goal for score, goal in ranked if score == best_score]
    if len(best_goals) != 1:
        # Ambiguous sibling scopes fail closed instead of inventing lineage.
        return None
    return replace(finding, objective_goal_ids=lineage(best_goals[0]))


def annotation_scan_text(line: str) -> str:
    """Remove path-like tokens that should not count as TODO annotations."""

    text = re.sub(r"(?i)[A-Za-z0-9_./-]*\.todo\.md\b", "", line)
    # Long CLI option names such as ``--todo-path`` are configuration
    # identifiers, not SQL-style ``-- TODO:`` comments.
    return re.sub(r"(?i)(?<![A-Za-z0-9_-])--[a-z][a-z0-9]*(?:-[a-z0-9]+)+\b", "", text)


def _position_in_simple_quoted_string(text: str, index: int) -> bool:
    quote = ""
    escaped = False
    for char in text[:index]:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
    return bool(quote)


def annotation_followup_marker(line: str) -> str:
    """Return the TODO-like marker when the line looks like a real annotation."""

    text = annotation_scan_text(line)
    for match in ANNOTATION_FOLLOWUP_RE.finditer(text):
        marker = str(match.group("line_marker") or match.group("comment_marker") or "").lower()
        start = match.start("line_marker") if match.group("line_marker") else match.start("comment_prefix")
        if _position_in_simple_quoted_string(text, start):
            continue
        return marker
    return ""


def codebase_parser_path(relative_path: str) -> str:
    """Return the current versioned parser path for a relative path."""

    return "markdown_fenced" if Path(relative_path).suffix.lower() in {".md", ".rst"} else "line_source"


def _python_swallowed_exception_lines(source: str) -> frozenset[int]:
    """Return real broad handlers whose direct body discards the failure.

    The v1 line matcher treated string literals, comments, and detector
    fixtures containing ``except Exception`` as executable handlers.  Besides
    producing false positives, that made autonomous refill scan its own
    evidence catalogs and generate an unbounded repair loop.  Python sources
    are now classified from their AST; a syntax error fails closed to no
    swallowed-exception claims rather than falling back to lexical guessing.
    """

    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return frozenset()

    result: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        broad = node.type is None or (
            isinstance(node.type, ast.Name) and node.type.id == "Exception"
        )
        if not broad:
            continue
        discards = any(
            isinstance(statement, ast.Pass)
            or (
                isinstance(statement, ast.Return)
                and (
                    statement.value is None
                    or (
                        isinstance(statement.value, ast.Constant)
                        and statement.value.value is None
                    )
                )
            )
            for statement in node.body
        )
        if discards:
            result.add(int(node.lineno))
    return frozenset(result)


def scan_findings_in_source(source: str, *, root_relative: str) -> list[CodebaseFinding]:
    """Run the current semantic matcher over an in-memory source fixture."""

    lines = str(source).splitlines()
    findings: list[CodebaseFinding] = []
    in_fenced_block = False
    scan_fences = codebase_parser_path(root_relative) == "markdown_fenced"
    python_source = Path(root_relative).suffix.lower() == ".py"
    python_swallowed_lines = (
        _python_swallowed_exception_lines(source)
        if python_source
        else frozenset()
    )
    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        if scan_fences and (stripped.startswith("```") or stripped.startswith("~~~")):
            in_fenced_block = not in_fenced_block
            continue
        if in_fenced_block or not stripped:
            continue
        lowered = stripped.lower()
        kind = ""
        priority = "P2"
        summary = ""
        annotation_marker = annotation_followup_marker(stripped)
        if annotation_marker:
            kind = "annotated_followup"
            priority = "P2" if annotation_marker in {"fixme", "hack", "xxx"} else "P3"
            summary = f"Resolve code annotation in {root_relative}:{index}"
        elif (
            index in python_swallowed_lines
            if python_source
            else (
                re.search(r"\bexcept\s*:\s*$", stripped) is not None
                or re.search(r"\bexcept\s+Exception\b", stripped) is not None
            )
        ):
            window = "\n".join(lines[index : min(len(lines), index + 3)]).lower()
            if python_source or "pass" in window or "return none" in window:
                kind = "swallowed_exception"
                priority = "P1"
                summary = f"Review swallowed exception path in {root_relative}:{index}"
        elif "assert false" in lowered or "raise notimplementederror" in lowered:
            kind = "placeholder_runtime_path"
            priority = "P1"
            summary = f"Replace placeholder runtime path in {root_relative}:{index}"
        if not kind:
            continue
        fingerprint = scan_fingerprint(
            kind=kind,
            root_relative_path=root_relative,
            line_number=index,
            snippet=stripped,
        )
        findings.append(
            CodebaseFinding(
                fingerprint=fingerprint,
                kind=kind,
                priority=priority,
                track=scan_track_for_path(root_relative),
                root_relative_path=root_relative,
                line_number=index,
                snippet=stripped[:240],
                summary=summary,
                validation=scan_validation_for_path(root_relative),
            )
        )
    return findings


def _scan_findings_in_file(
    path: Path,
    *,
    repo_root: Path,
) -> tuple[list[CodebaseFinding], dict[str, str] | None]:
    root_relative = root_relative_path(repo_root, path)
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [], {
            "path": root_relative,
            "reason_code": "read_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return scan_findings_in_source(source, root_relative=root_relative), None


def run_codebase_analyzer_canaries() -> AnalyzerCanaryReport:
    """Exercise every registered v1 finding kind and parser path."""

    def analyze(source: str, relative_path: str) -> tuple[Sequence[CodebaseFinding], str, str]:
        return (
            scan_findings_in_source(source, root_relative=relative_path),
            codebase_parser_path(relative_path),
            "",
        )

    return run_analyzer_canaries(CODEBASE_SCAN_ANALYZER_VERSION, analyze)


def scan_findings_in_file(path: Path, *, repo_root: Path) -> list[CodebaseFinding]:
    """Parse one eligible file, retaining the historical list-only API."""

    findings, _failure = _scan_findings_in_file(path, repo_root=repo_root)
    return findings


def scan_codebase_findings(
    repo_root: Path,
    *,
    max_findings: int | None,
    seen_fingerprints: Iterable[str] = (),
    exhaustive: bool = False,
    skip_prefixes: Sequence[str] = CODEBASE_SCAN_SKIP_PREFIXES,
    include_prefixes: Sequence[str] = (),
    allowed_tracks: Sequence[str] = (),
    return_inventory: bool = False,
) -> list[CodebaseFinding] | CodebaseScanInventory:
    """Scan tracked files for candidates, optionally returning full accounting.

    ``max_findings=None`` is the explicit uncapped audit mode.  The default
    remains the original list API.  Receipt-producing callers use
    ``return_inventory=True`` so skipped paths and failures are never collapsed
    into the same empty result as a genuinely clean scan.
    """

    inventory = CodebaseScanInventory()
    seen = {str(item).strip().lower() for item in seen_fingerprints if str(item).strip()}
    selected_fingerprints: set[str] = set()

    def already_seen(fingerprint: str) -> bool:
        return any(fingerprint == item or fingerprint.startswith(item) for item in seen)

    expected_roots = expected_git_worktrees(repo_root, skip_prefixes=skip_prefixes)
    inventory.expected_git_roots = [
        root_relative_path(repo_root, path) or "." for path in expected_roots
    ]
    git_roots = [
        path
        for path in discover_git_worktrees(repo_root, skip_prefixes=skip_prefixes)
        if path.is_dir()
    ]
    inventory.git_roots = [root_relative_path(repo_root, path) or "." for path in git_roots]
    eligible_paths: list[Path] = []
    for git_root in git_roots:
        try:
            result = subprocess.run(
                ["git", "ls-files", "-z"],
                cwd=git_root,
                capture_output=True,
                check=False,
            )
        except (FileNotFoundError, OSError) as exc:
            if not git_root.exists():
                continue
            raise RuntimeError(f"git inventory failed for {git_root}: {exc}") from exc
        if result.returncode != 0:
            raise RuntimeError(
                "git inventory failed for "
                f"{git_root}: {result.stderr.decode('utf-8', errors='replace')[-1000:]}"
            )
        for raw_path in result.stdout.split(b"\0"):
            if not raw_path:
                continue
            inventory.tracked_file_count += 1
            relative = raw_path.decode("utf-8", errors="surrogateescape")
            if not repo_relative_path_safe(relative):
                inventory.excluded_files.append(
                    {"path": relative, "reason_code": "unsafe_git_path"}
                )
                continue
            path = git_root / relative
            inventory.tracked_paths.append(root_relative_path(repo_root, path))
            reason = codebase_scan_file_exclusion_reason(
                path,
                repo_root=repo_root,
                skip_prefixes=skip_prefixes,
                include_prefixes=include_prefixes,
                allowed_tracks=allowed_tracks,
            )
            if reason:
                inventory.excluded_files.append(
                    {"path": root_relative_path(repo_root, path), "reason_code": reason}
                )
                continue
            inventory.eligible_file_count += 1
            inventory.eligible_paths.append(root_relative_path(repo_root, path))
            eligible_paths.append(path)

    for path in eligible_paths:
        file_findings, failure = _scan_findings_in_file(path, repo_root=repo_root)
        if failure is not None:
            inventory.parser_failures.append(failure)
            continue
        inventory.parsed_file_count += 1
        for finding in file_findings:
            inventory.raw_candidate_count += 1
            if already_seen(finding.fingerprint):
                inventory.seen_candidate_count += 1
                continue
            if finding.fingerprint in selected_fingerprints:
                inventory.deduplicated_candidate_count += 1
                continue
            if max_findings is None or len(inventory.findings) < max_findings:
                inventory.findings.append(finding)
                selected_fingerprints.add(finding.fingerprint)
                continue
            inventory.rejected_candidate_count += 1
            if not exhaustive:
                inventory.complete = False
                return inventory if return_inventory else inventory.findings
    return inventory if return_inventory else inventory.findings


def admit_codebase_refill_candidates(
    inventory: CodebaseScanInventory,
    *,
    objective_goals: Sequence[ObjectiveGoal],
    mission_terms: Sequence[str] = (),
    max_findings: int | None,
    allow_unscoped: bool = False,
    objective_scope_configured: bool = False,
) -> CodebaseRefillAdmission:
    """Apply goal policy after scanning and before any taskboard mutation.

    The scanner remains objective-agnostic.  This admission stage either binds
    each candidate to existing goal lineage or records why it was not allowed
    to become a task.  ``allow_unscoped`` exists only for explicit legacy
    maintenance boards without an objective heap.
    """

    admitted_findings: list[CodebaseFinding] = []
    rejections: list[Mapping[str, Any]] = []
    policy_errors = list(codebase_refill_goal_graph_errors(objective_goals))
    if allow_unscoped and (objective_scope_configured or objective_goals):
        policy_errors.insert(
            0,
            {
                "reason_code": "incompatible_unscoped_refill",
                "message": (
                    "allow_unscoped is only valid when no objective heap is "
                    "configured"
                ),
            },
        )
    if (
        not policy_errors
        and not allow_unscoped
        and not any(goal.is_schedulable for goal in objective_goals)
    ):
        policy_errors.append(
            {
                "reason_code": "no_schedulable_goal",
                "message": "objective heap has no schedulable goal or subgoal",
            }
        )

    def rejection(finding: CodebaseFinding, reason_code: str) -> dict[str, Any]:
        finding_payload = finding.to_dict()
        finding_payload["objective_goal_ids"] = list(finding.objective_goal_ids)
        return {
            "path": finding.root_relative_path,
            "reason_code": reason_code,
            "fingerprint": finding.fingerprint,
            "summary": finding.summary,
            "finding": finding_payload,
        }

    if policy_errors:
        reason_code = str(policy_errors[0]["reason_code"])
        rejections.extend(
            rejection(finding, reason_code)
            for finding in inventory.findings
        )
        return CodebaseRefillAdmission(
            findings=(),
            rejections=tuple(rejections),
            policy_errors=tuple(policy_errors),
            allow_unscoped=allow_unscoped,
            max_findings=max_findings,
        )

    for finding in inventory.findings:
        admitted = align_codebase_finding_to_goals(
            finding,
            objective_goals,
            mission_terms=mission_terms,
        )
        if admitted is None:
            if allow_unscoped:
                admitted = finding
            else:
                rejections.append(rejection(finding, "no_goal_lineage"))
                continue
        if max_findings is not None and len(admitted_findings) >= max_findings:
            rejections.append(rejection(admitted, "admission_limit"))
            continue
        admitted_findings.append(admitted)
    return CodebaseRefillAdmission(
        findings=tuple(admitted_findings),
        rejections=tuple(rejections),
        policy_errors=(),
        allow_unscoped=allow_unscoped,
        max_findings=max_findings,
    )


def codebase_source_tree_identity(
    repo_root: Path,
    inventory: CodebaseScanInventory,
) -> str:
    """Hash source/configuration material inspected by the codebase analyzer.

    Supervisor state, taskboards, and generated discoveries are intentionally
    absent because the inventory excludes them.  Common tracked configuration
    formats that the annotation parser does not parse are included so a build
    or analyzer configuration edit invalidates exhaustion evidence.
    """

    relevant = set(inventory.eligible_paths)
    configuration_names = {
        ".gitmodules",
        "cargo.lock",
        "deno.lock",
        "package-lock.json",
        "pnpm-lock.yaml",
        "poetry.lock",
        "pyproject.toml",
        "requirements.txt",
        "uv.lock",
        "yarn.lock",
    }
    for relative in inventory.tracked_paths:
        path = Path(relative)
        if path.name.lower() in configuration_names or path.suffix.lower() in {".toml", ".lock"}:
            relevant.add(relative)
    digest = sha256()
    digest.update(b"codebase-source-tree/v1\0")
    for relative in sorted(relevant):
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        path = repo_root / relative
        try:
            digest.update(f"mode:{path.stat().st_mode & 0o111:o}\0".encode("ascii"))
            if path.is_symlink():
                digest.update(b"symlink\0")
                digest.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
            elif path.is_file():
                digest.update(b"file\0")
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
            else:
                digest.update(b"missing\0")
        except OSError as exc:
            digest.update(f"error:{type(exc).__name__}".encode("ascii", errors="replace"))
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def codebase_exhaustion_configuration(
    *,
    skip_prefixes: Sequence[str],
    include_prefixes: Sequence[str] = (),
    allowed_tracks: Sequence[str] = (),
    health_thresholds: AnalyzerHealthThresholds,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonical behavior-affecting configuration shared by normal/audit scans."""

    return {
        "audit_scanner_version": CODEBASE_AUDIT_SCANNER_VERSION,
        "analyzer_version": CODEBASE_SCAN_ANALYZER_VERSION,
        "max_file_bytes": CODEBASE_SCAN_MAX_FILE_BYTES,
        "suffixes": sorted(CODEBASE_SCAN_SUFFIXES),
        "skip_prefixes": sorted(str(item) for item in skip_prefixes),
        "include_prefixes": sorted(str(item) for item in include_prefixes),
        "allowed_tracks": sorted(str(item).strip().lower() for item in allowed_tracks),
        "health_thresholds": health_thresholds.to_dict(),
        "exhaustive": True,
        "extra": dict(extra or {}),
    }


def write_codebase_scan_discovery(
    *,
    discovery_dir: Path,
    task_id: str,
    finding: CodebaseFinding,
) -> Path:
    date = datetime.now(timezone.utc).date().isoformat()
    path = discovery_dir / f"{date}-{task_id.lower()}-codebase-scan-{finding.fingerprint[:12]}.md"
    discovery_dir.mkdir(parents=True, exist_ok=True)
    content = f"""# {task_id} Codebase Scan Finding

Date: {date}
Fingerprint: {finding.fingerprint}
Kind: {finding.kind}
Source: {finding.root_relative_path}:{finding.line_number}
Priority: {finding.priority}
Track: {finding.track}
Objective goals: {", ".join(finding.objective_goal_ids)}

## Evidence

```text
{finding.snippet}
```

## Suggested Handling

Resolve only the work needed to advance the existing objective lineage shown
above. Do not broaden the task to adjacent cleanup. If the finding is a false
positive or does not actually support that lineage, record that disposition in
the discovery evidence so the supervisor does not keep re-adding it.
"""
    path.write_text(content, encoding="utf-8")
    return path


def codebase_finding_task_identity(finding: CodebaseFinding) -> TaskIdentity:
    """Return the canonical work identity for one codebase-scan finding."""

    return canonical_task_identity(
        {
            "dedupe_key": f"codebase-scan:{finding.fingerprint}",
            "title": finding.summary,
            "outputs": [finding.root_relative_path],
            "acceptance": [
                f"Resolve {finding.kind} at "
                f"{finding.root_relative_path}:{finding.line_number}"
            ],
        },
        board_namespace="codebase-scan",
        source_path=finding.root_relative_path,
    )


def codebase_scan_task_block(
    *,
    task_id: str,
    finding: CodebaseFinding,
    discovery_path: Path,
    depends_on: Sequence[str] = (),
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    bundle_key: str = "",
    bundle_shard: str = "",
    ast_symbols: Sequence[str] = (),
    board_namespace: str = "codebase-scan",
) -> str:
    outputs = [discovery_output_path, finding.root_relative_path]
    identity = codebase_finding_task_identity(finding)
    lineage = list(finding.objective_goal_ids)
    goal_id = lineage[0] if lineage else ""
    parent_goal_ids = lineage[1:]
    planning_lines: list[str] = [
        *(
            [
                f"- Graph parents: {', '.join(parent_goal_ids) or 'none'}",
                f"- Graph depth: {len(parent_goal_ids)}",
                f"- Goal id: {goal_id}",
                f"- Goal lineage: {', '.join(lineage)}",
                "- Goal registration: existing",
            ]
            if goal_id
            else ["- Goal registration: unscoped_legacy"]
        ),
        f"- Canonical task key: {identity.canonical_task_key}",
        f"- Canonical task CID: {identity.canonical_task_cid}",
        f"- Semantic identity: {identity.semantic_fingerprint}",
        f"- Acceptance subset: Resolve {finding.kind} at {finding.root_relative_path}:{finding.line_number}",
        f"- Preconditions: {finding.root_relative_path} exists and the scan evidence remains applicable",
        f"- Effects: resolve {finding.kind} in {finding.root_relative_path} and pass focused validation",
        f"- Evidence subset: {finding.root_relative_path}:{finding.line_number}, {discovery_path}",
        "- Resource class: cpu-small",
        "- Token class: small",
        "- Context budget tokens: 2048",
        "- Provider role: grok-implement, codex-review",
        "- Resources: python, focused validation runner",
        f"- Merge fate: {finding.root_relative_path}",
        "- Rejection reasons: none",
        f"- Missing evidence: {finding.summary}",
        "- Candidate kind: codebase_scan",
        f"- Todo vector key: {finding.fingerprint[:16]}",
    ]
    if bundle_key:
        planning_lines.extend(
        [
            f"- Bundle: {bundle_key}",
            f"- Bundle shard: {bundle_shard}",
            "- Bundle strategy: codebase_file_ast",
            f"- Parallel lane: {bundle_key}",
            "- Conflict policy: serialize findings for the same file; allow independent file bundles to run concurrently",
            f"- Predicted files: {finding.root_relative_path}",
            f"- AST symbols: {', '.join(ast_symbols)}",
            "- AST symbol scope: file",
            f"- Merge key: {bundle_key}",
            f"- Merge family: {finding.root_relative_path}",
            "- Merge role: codebase_scan",
            "- Work item count: 1",
            "- Work scope: codebase_file_ast",
        ]
        )
    planning = "\n" + "\n".join(planning_lines)
    return f"""## {task_id} {finding.summary}

- Status: todo
- Completion: manual
- Priority: {finding.priority}
- Track: {finding.track}
- Depends on: {", ".join(depends_on)}
- Outputs: {", ".join(outputs)}
- Validation: {finding.validation}
- Board namespace: {normalize_board_namespace(board_namespace)}{planning}
- Acceptance: Goal-scoped refill admitted this finding from {finding.root_relative_path}:{finding.line_number} for {goal_id or "an explicitly unscoped legacy board"}. Use evidence in {discovery_path}, make only the smallest change required by that goal lineage, add or update focused validation when appropriate, and do not expand into adjacent cleanup.
"""


def codebase_scan_bundle_key(finding: CodebaseFinding) -> str:
    """Return a file-local bundle key for a generated codebase finding."""

    source_key = safe_bundle_key(Path(finding.root_relative_path).with_suffix("").as_posix())
    return f"codebase/{safe_bundle_key(finding.track)}/{source_key}"


def write_codebase_scan_bundle_shards(
    *,
    bundle_dir: Path,
    repo_root: Path,
    todo_path: Path,
    records: Sequence[Mapping[str, Any]],
) -> list[Path]:
    """Merge codebase-scan records into file-local shards and their bundle index."""

    if not records:
        return []
    bundle_dir.mkdir(parents=True, exist_ok=True)
    index_path = bundle_dir / "index.json"
    try:
        index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        index_payload = {}
    if not isinstance(index_payload, dict):
        index_payload = {}
    bundles = index_payload.get("bundles")
    if not isinstance(bundles, dict):
        bundles = {}

    generated_paths: list[Path] = []
    source_todo = repo_relative_path(repo_root, todo_path)
    for record in records:
        task_id = str(record.get("task_id") or "")
        bundle_key = str(record.get("bundle_key") or "")
        task_block = str(record.get("task_block") or "")
        task_payload = record.get("task_payload")
        if not task_id or not bundle_key or not task_block or not isinstance(task_payload, Mapping):
            continue

        shard_path = bundle_path(bundle_dir, bundle_key)
        try:
            shard_text = shard_path.read_text(encoding="utf-8")
        except OSError:
            shard_text = (
                f"# Codebase Bundle: {bundle_key}\n\n"
                f"Source todo: {source_todo}\n"
                "Purpose: group generated codebase findings by source file and AST locality.\n"
                "Conflict policy: serialize edits to one file; allow independent file bundles to run concurrently.\n"
            )
        if f"## {task_id} " not in shard_text:
            shard_path.write_text(shard_text.rstrip() + "\n\n" + task_block.strip() + "\n", encoding="utf-8")
            generated_paths.append(shard_path)

        existing = bundles.get(bundle_key)
        info = dict(existing) if isinstance(existing, Mapping) else {}
        task_map = {
            str(item.get("task_id")): dict(item)
            for item in info.get("tasks", [])
            if isinstance(item, Mapping) and str(item.get("task_id") or "")
        }
        task_map[task_id] = dict(task_payload)
        info.update(
            {
                "bundle_key": bundle_key,
                "shard_path": repo_relative_path(repo_root, shard_path),
                "parallel_lane": bundle_key,
                "bundle_strategy": "codebase_file_ast",
                "conflict_policy": (
                    "serialize findings for the same file; allow independent file bundles to run concurrently"
                ),
                "tasks": [task_map[key] for key in sorted(task_map)],
            }
        )
        bundles[bundle_key] = info

    index_payload.update(
        {
            "generated_at": utc_now(),
            "source_todo": source_todo,
            "bundles": bundles,
        }
    )
    from ..runtime.artifact_store import write_bundle_index_artifact

    write_bundle_index_artifact(index_path, index_payload)
    generated_paths.append(index_path)
    return list(dict.fromkeys(generated_paths))


def duplicate_task_id_records(tasks: Sequence[Any]) -> list[dict[str, Any]]:
    """Return todo-board records for task ids that appear more than once."""

    task_groups: dict[str, list[Any]] = {}
    for task in tasks:
        task_id = str(getattr(task, "task_id", "") or "").strip()
        if not task_id:
            continue
        task_groups.setdefault(task_id, []).append(task)

    records: list[dict[str, Any]] = []
    for task_id, duplicates in sorted(task_groups.items()):
        if len(duplicates) < 2:
            continue
        titles = [str(getattr(task, "title", "") or "") for task in duplicates]
        source_lines: list[int] = []
        for task in duplicates:
            try:
                source_line = int(getattr(task, "source_line", 0) or 0)
            except (TypeError, ValueError):
                continue
            if source_line > 0:
                source_lines.append(source_line)
        fingerprint = sha1(
            json.dumps(
                {
                    "kind": "duplicate_task_id",
                    "task_id": task_id,
                    "titles": sorted(title for title in titles if title),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        records.append(
            {
                "source_task_id": task_id,
                "source_title": "Duplicate task id",
                "missing_dependencies": [],
                "self_references": [],
                "dependency_cycle": [],
                "duplicate_task_id": task_id,
                "duplicate_task_lines": source_lines,
                "duplicate_task_titles": titles,
                "fingerprint": fingerprint,
            }
        )
    return records


def dependency_guardrail_records(tasks: Sequence[Any]) -> list[dict[str, Any]]:
    """Return todo-board records that can keep tasks from becoming ready."""

    task_ids = {str(task.task_id) for task in tasks}
    task_ids_by_goal: dict[str, set[str]] = {}
    for task in tasks:
        metadata = getattr(task, "metadata", {})
        if not isinstance(metadata, Mapping):
            continue
        for goal_id in split_csv(str(metadata.get("goal id") or "")):
            task_ids_by_goal.setdefault(goal_id, set()).add(str(task.task_id))
    open_task_ids = {
        str(task.task_id)
        for task in tasks
        if str(task.status).lower() not in {"completed", "blocked"}
    }
    dependency_graph = {
        str(task.task_id): sorted(
            {
                dependency_task_id
                for dep in task.depends_on
                if str(dep).strip()
                for dependency_task_id in (
                    [str(dep)]
                    if str(dep) in open_task_ids
                    else []
                    if str(dep) in task_ids
                    else task_ids_by_goal.get(str(dep), set()) & open_task_ids
                )
            }
        )
        for task in tasks
        if str(task.task_id) in open_task_ids
    }
    records: list[dict[str, Any]] = duplicate_task_id_records(tasks)

    def cycle_containing(start: str) -> list[str]:
        """Return a dependency cycle only when ``start`` is a member.

        A task which merely waits on a cyclic prerequisite is blocked by that
        prerequisite, but its own metadata is not cyclic.  Filing a separate
        repair for every downstream waiter obscures the root defect and can
        exhaust the bounded repair-task budget before the cycle member is
        reached.
        """

        path = [start]

        def visit(node: str) -> list[str]:
            for dependency in dependency_graph.get(node, []):
                if dependency == start:
                    return [*path, start]
                if dependency in path:
                    # This is a reachable cycle which does not contain the
                    # source task.  Its own members receive their own records.
                    continue
                path.append(dependency)
                cycle = visit(dependency)
                path.pop()
                if cycle:
                    return cycle
            return []

        return visit(start)

    for task in tasks:
        if str(task.status).lower() in {"completed", "blocked"}:
            continue
        dependencies = [str(dep) for dep in task.depends_on if str(dep).strip()]
        missing = sorted(
            dep
            for dep in dependencies
            if dep not in task_ids and dep not in task_ids_by_goal
        )
        self_references = sorted(dep for dep in dependencies if dep == task.task_id)
        dependency_cycle = cycle_containing(task.task_id)
        if not missing and not self_references and not dependency_cycle:
            continue
        fingerprint = sha1(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "missing_dependencies": missing,
                    "self_references": self_references,
                    "dependency_cycle": dependency_cycle,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        records.append(
            {
                "source_task_id": task.task_id,
                "source_title": task.title,
                "missing_dependencies": missing,
                "self_references": self_references,
                "dependency_cycle": dependency_cycle,
                "fingerprint": fingerprint,
            }
        )
    return records


def write_dependency_guardrail_discovery(
    *,
    discovery_dir: Path,
    task_id: str,
    record: Mapping[str, Any],
) -> Path:
    date = datetime.now(timezone.utc).date().isoformat()
    path = discovery_dir / f"{date}-{task_id.lower()}-dependency-guardrail.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = ", ".join(str(item) for item in record.get("missing_dependencies", []) or []) or "none"
    self_references = ", ".join(str(item) for item in record.get("self_references", []) or []) or "none"
    dependency_cycle = " -> ".join(str(item) for item in record.get("dependency_cycle", []) or []) or "none"
    duplicate_task_id = str(record.get("duplicate_task_id") or "") or "none"
    duplicate_lines = ", ".join(str(item) for item in record.get("duplicate_task_lines", []) or []) or "none"
    duplicate_titles = "\n".join(
        f"- {title}" for title in record.get("duplicate_task_titles", []) or [] if str(title).strip()
    )
    duplicate_titles = duplicate_titles or "- none"
    content = f"""# Dependency Guardrail: {record.get("source_task_id")}

Created: {utc_now()}
Fingerprint: {record.get("fingerprint")}
Source task: {record.get("source_task_id")} {record.get("source_title") or ""}
Missing dependencies: {missing}
Self-referential dependencies: {self_references}
Dependency cycle: {dependency_cycle}
Duplicate task id: {duplicate_task_id}
Duplicate source lines: {duplicate_lines}

## Duplicate Task Titles

{duplicate_titles}

## Why This Blocks Progress

The implementation daemon only selects tasks whose dependencies are completed.
When an open task depends on a task id that is not present on the board, or on
itself, or participates in a dependency cycle, the task can remain waiting
indefinitely while the supervisor reports no ready work. Duplicate task ids are
also ambiguous because status maps, dependency resolution, and guardrail
releases all key by task id.

## Suggested Repair

Inspect the source task metadata and either add the missing prerequisite task,
remove the stale dependency, break the dependency cycle, rename duplicate task
ids so each task is unique, or replace stale references with the correct existing
task id. Keep the todo board parseable after the repair.
"""
    path.write_text(content, encoding="utf-8")
    return path


def dependency_guardrail_task_block(
    *,
    task_id: str,
    source_task_id: str,
    discovery_path: Path,
    todo_output_path: str,
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
) -> str:
    return f"""## {task_id} Resolve dependency guardrail for {source_task_id}

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: {discovery_output_path}, {todo_output_path}
- Validation: test -f {shlex.quote(str(discovery_path))}
- Acceptance: Dependency guardrail filed this because {source_task_id} has missing, self-referential, cyclic, or duplicate task-id metadata. Use the evidence in {discovery_path} to repair the todo board metadata or add the missing prerequisite task, then verify the original task can become ready once its real dependencies complete.
"""


def reconciliation_guardrail_records(
    *,
    reconciliation_result: Mapping[str, Any] | None = None,
    cleanup_result: Mapping[str, Any] | None = None,
    generated_status_paths: Sequence[str] = (),
    generated_status_prefixes: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Return grouped cleanup/reconciliation blockers that need deliberate repair."""

    records: list[dict[str, Any]] = []
    reconciliation = dict(reconciliation_result or {})
    cleanup = dict(cleanup_result or {})

    if reconciliation.get("attempted") and reconciliation.get("main_checkout_dirty"):
        candidate_count = int(reconciliation.get("candidate_count") or 0)
        if candidate_count > 0:
            status_short = [str(item) for item in reconciliation.get("main_status_short", []) if str(item).strip()]
            main_dirty_evidence = (
                dict(reconciliation.get("main_dirty_evidence") or {})
                if isinstance(reconciliation.get("main_dirty_evidence"), Mapping)
                else {}
            )
            status_short, main_dirty_evidence = filter_generated_main_checkout_evidence(
                status_short=status_short,
                evidence=main_dirty_evidence,
                generated_paths=generated_status_paths,
                generated_prefixes=generated_status_prefixes,
            )
            if status_short:
                candidates = [
                    {
                        "branch": str(item.get("branch") or ""),
                        "path": str(item.get("path") or ""),
                        "target_ref": str(item.get("target_ref") or reconciliation.get("target_ref") or ""),
                    }
                    for item in reconciliation.get("candidates", [])
                    if isinstance(item, Mapping)
                ]
                fingerprint = sha1(
                    json.dumps(
                        {
                            "kind": "main_checkout_dirty",
                            "status_short": status_short,
                            "candidate_branches": [item["branch"] for item in candidates],
                        },
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()
                records.append(
                    {
                        "kind": "main_checkout_dirty",
                        "priority": "P1",
                        "track": "ops",
                        "summary": f"Resolve dirty main checkout blocking {candidate_count} worktree merges",
                        "fingerprint": fingerprint,
                        "candidate_count": candidate_count,
                        "status_short": status_short,
                        "main_dirty_evidence": main_dirty_evidence,
                        "samples": candidates[:20],
                        "reason": "main_checkout_dirty",
                        "dedupe_key": "reconciliation_guardrail:main_checkout_dirty",
                    }
                )

    preflight_samples: list[dict[str, Any]] = []
    conflict_path_counts: dict[str, int] = {}
    for item in reconciliation.get("processed", []) or []:
        if not isinstance(item, Mapping):
            continue
        preflight_result = item.get("preflight_result") or {}
        if not isinstance(preflight_result, Mapping):
            continue
        if preflight_result.get("mergeable") is not False:
            continue
        conflict_paths = [
            str(path).strip()
            for path in preflight_result.get("conflict_paths", []) or []
            if str(path).strip()
        ]
        for path in conflict_paths:
            conflict_path_counts[path] = conflict_path_counts.get(path, 0) + 1
        preflight_samples.append(
            {
                "branch": str(item.get("branch") or preflight_result.get("branch") or ""),
                "path": str(item.get("path") or ""),
                "target_ref": str(item.get("target_ref") or preflight_result.get("target_ref") or ""),
                "conflict_paths": conflict_paths[:20],
                "reason": str(preflight_result.get("reason") or "preflight_merge_conflict"),
            }
        )
    if preflight_samples:
        fingerprint = sha1(
            json.dumps(
                {
                    "kind": "preflight_merge_conflict",
                    "branches": [item["branch"] for item in preflight_samples],
                    "conflict_path_counts": conflict_path_counts,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        records.append(
            {
                "kind": "preflight_merge_conflict",
                "priority": "P1",
                "track": "ops",
                "summary": (
                    f"Resolve {len(preflight_samples)} preflight-conflicting "
                    "backlogged worktree merges"
                ),
                "fingerprint": fingerprint,
                "candidate_count": len(preflight_samples),
                "status_short": [],
                "samples": preflight_samples[:20],
                "reason": "preflight_merge_conflict",
                "conflict_path_counts": conflict_path_counts,
                "dedupe_key": "reconciliation_guardrail:preflight_merge_conflict",
            }
        )

    dirty_groups: dict[str, dict[str, Any]] = {}
    grouped_payload = cleanup.get("dirty_worktree_groups")
    if isinstance(grouped_payload, Mapping) and grouped_payload:
        for dirty_reason, payload in grouped_payload.items():
            if not isinstance(payload, Mapping):
                continue
            dirty_groups[str(dirty_reason)] = {
                "count": int(payload.get("count") or 0),
                "samples": [dict(item) for item in payload.get("samples", []) if isinstance(item, Mapping)],
            }
    else:
        for item in cleanup.get("skipped", []):
            if not isinstance(item, Mapping) or str(item.get("reason") or "") != "dirty_worktree":
                continue
            dirty_redundancy = item.get("dirty_redundancy") or {}
            dirty_reason = (
                str(dirty_redundancy.get("reason") or "dirty_worktree")
                if isinstance(dirty_redundancy, Mapping)
                else "dirty_worktree"
            )
            group = dirty_groups.setdefault(dirty_reason, {"count": 0, "samples": []})
            group["count"] += 1
            if len(group["samples"]) < 20:
                group["samples"].append(
                    {
                        "branch": str(item.get("branch") or ""),
                        "path": str(item.get("path") or ""),
                        "status_short": [str(line) for line in item.get("status_short", []) if str(line).strip()],
                        "dirty_reason": dirty_reason,
                        "dirty_evidence": dict(item.get("dirty_evidence") or {}),
                    }
                )

    for dirty_reason, group in sorted(dirty_groups.items()):
        samples = list(group.get("samples") or [])
        count = int(group.get("count") or len(samples))
        fingerprint = sha1(
            json.dumps(
                {
                    "kind": "dirty_backlogged_worktree",
                    "dirty_reason": dirty_reason,
                    "branches": [item["branch"] for item in samples],
                    "paths": [item["path"] for item in samples],
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        records.append(
            {
                "kind": "dirty_backlogged_worktree",
                "priority": "P1" if dirty_reason == "unsupported_status" else "P2",
                "track": "ops",
                "summary": f"Resolve {count} dirty backlogged worktrees blocked by {dirty_reason}",
                "fingerprint": fingerprint,
                "candidate_count": count,
                "status_short": [],
                "samples": samples[:20],
                "reason": dirty_reason,
                "dedupe_key": f"reconciliation_guardrail:dirty_backlogged_worktree:{dirty_reason}",
            }
        )

    return records


def status_line_path(line: str) -> str:
    path_text = line[3:].strip() if len(line) > 3 else line.strip()
    if " -> " in path_text:
        path_text = path_text.split(" -> ", 1)[-1].strip()
    return path_text.rstrip("/")


def status_line_category(line: str) -> str:
    code = line[:2]
    if code == "??":
        return "untracked"
    if "U" in code:
        return "unmerged"
    if "D" in code:
        return "deleted"
    if "R" in code:
        return "renamed"
    if "A" in code:
        return "added"
    if "M" in code:
        return "modified"
    if code.strip():
        return "other_dirty"
    return "clean"


def normalize_status_path(path: str) -> str:
    path_text = str(path).strip()
    if " -> " in path_text:
        path_text = path_text.split(" -> ", 1)[-1].strip()
    return path_text.rstrip("/")


def name_status_path(line: str) -> str:
    parts = str(line).split("\t")
    if len(parts) > 1:
        return normalize_status_path(parts[-1])
    return normalize_status_path(str(line).split(maxsplit=1)[-1] if str(line).split() else "")


def path_is_generated_status_output(
    path: str,
    *,
    generated_paths: Sequence[str] = (),
    generated_prefixes: Sequence[str] = (),
) -> bool:
    path_text = normalize_status_path(path)
    if not path_text:
        return False
    exact = {normalize_status_path(item) for item in generated_paths if normalize_status_path(item)}
    if path_text in exact:
        return True
    for prefix in generated_prefixes:
        prefix_text = normalize_status_path(str(prefix))
        if not prefix_text:
            continue
        if path_text == prefix_text or path_text.startswith(prefix_text + "/"):
            return True
    return False


def filter_generated_main_checkout_evidence(
    *,
    status_short: Sequence[str],
    evidence: Mapping[str, Any],
    generated_paths: Sequence[str] = (),
    generated_prefixes: Sequence[str] = (),
) -> tuple[list[str], dict[str, Any]]:
    """Remove supervisor-generated todo/discovery output paths from dirty-main evidence."""

    filtered_status: list[str] = []
    filtered_paths: list[str] = []
    removed_paths: list[str] = []
    for line in status_short:
        line_text = str(line)
        path = status_line_path(line_text)
        if path_is_generated_status_output(
            path,
            generated_paths=generated_paths,
            generated_prefixes=generated_prefixes,
        ):
            if path and path not in removed_paths:
                removed_paths.append(path)
            continue
        filtered_status.append(line_text)
        if path and path not in filtered_paths:
            filtered_paths.append(path)

    filtered_evidence: dict[str, Any] = dict(evidence or {})
    filtered_evidence["status_short"] = filtered_status[:50]
    filtered_evidence["status_paths"] = filtered_paths[:50]
    path_categories: dict[str, int] = {}
    for line in filtered_status:
        category = status_line_category(line)
        path_categories[category] = path_categories.get(category, 0) + 1
    filtered_evidence["path_categories"] = path_categories
    for key in ("untracked_paths",):
        values = []
        for item in filtered_evidence.get(key, []) or []:
            path = normalize_status_path(str(item))
            if path and not path_is_generated_status_output(
                path,
                generated_paths=generated_paths,
                generated_prefixes=generated_prefixes,
            ):
                values.append(path)
            elif path and path not in removed_paths:
                removed_paths.append(path)
        if values:
            filtered_evidence[key] = values[:50]
        else:
            filtered_evidence.pop(key, None)
    for key in ("name_status", "staged_name_status"):
        lines = []
        for line in str(filtered_evidence.get(key) or "").splitlines():
            path = name_status_path(line)
            if path and path_is_generated_status_output(
                path,
                generated_paths=generated_paths,
                generated_prefixes=generated_prefixes,
            ):
                if path not in removed_paths:
                    removed_paths.append(path)
                continue
            if line.strip():
                lines.append(line)
        if lines:
            filtered_evidence[key] = "\n".join(lines)
        else:
            filtered_evidence.pop(key, None)
    if removed_paths:
        filtered_evidence["filtered_generated_status_paths"] = removed_paths[:50]
        filtered_evidence.pop("diff_stat", None)
        filtered_evidence.pop("submodule_summary", None)
    return filtered_status, filtered_evidence


def relative_status_path(path: Path, *, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def generated_status_filter_path(path: Path | str, *, repo_root: Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return relative_status_path(candidate, repo_root=repo_root)
    return normalize_status_path(str(path))


def generated_guardrail_status_filters(
    *,
    todo_path: Path,
    discovery_dir: Path,
    repo_root: Path,
    additional_generated_paths: Sequence[Path | str] = (),
    additional_generated_prefixes: Sequence[Path | str] = (),
) -> tuple[list[str], list[str]]:
    generated_paths: list[str] = []
    generated_prefixes: list[str] = []
    todo_relative = relative_status_path(todo_path, repo_root=repo_root)
    if todo_relative:
        generated_paths.append(todo_relative)
    discovery_relative = relative_status_path(discovery_dir, repo_root=repo_root)
    if discovery_relative:
        generated_prefixes.append(discovery_relative)
    for path in additional_generated_paths:
        relative = generated_status_filter_path(path, repo_root=repo_root)
        if relative:
            generated_paths.append(relative)
    for prefix in additional_generated_prefixes:
        relative = generated_status_filter_path(prefix, repo_root=repo_root)
        if relative:
            generated_prefixes.append(relative)

    parts = Path(todo_relative).parts
    for end in range(1, len(parts)):
        ancestor = Path(*parts[:end])
        if ancestor.as_posix() in {".", ""}:
            continue
        if (repo_root / ancestor / ".git").exists():
            generated_paths.append(ancestor.as_posix())
    return list(dict.fromkeys(generated_paths)), list(dict.fromkeys(generated_prefixes))


def reconciliation_guardrail_plan(record: Mapping[str, Any]) -> dict[str, Any]:
    """Build a bounded reconciliation plan for a cleanup blocker record."""

    kind = str(record.get("kind") or "")
    reason = str(record.get("reason") or "")
    samples = [dict(item) for item in record.get("samples", []) or [] if isinstance(item, Mapping)]
    main_dirty_evidence = (
        dict(record.get("main_dirty_evidence") or {})
        if isinstance(record.get("main_dirty_evidence"), Mapping)
        else {}
    )
    sample_status_paths: list[str] = []
    for line in main_dirty_evidence.get("status_short", []) or []:
        path = status_line_path(str(line))
        if path and path not in sample_status_paths:
            sample_status_paths.append(path)
    for path in main_dirty_evidence.get("status_paths", []) or []:
        path_text = str(path).strip()
        if path_text and path_text not in sample_status_paths:
            sample_status_paths.append(path_text)
    for sample in samples:
        for path in sample.get("conflict_paths", []) or []:
            path_text = str(path).strip()
            if path_text and path_text not in sample_status_paths:
                sample_status_paths.append(path_text)
        for line in sample.get("status_short", []) or []:
            path = status_line_path(str(line))
            if path and path not in sample_status_paths:
                sample_status_paths.append(path)
        evidence = sample.get("dirty_evidence") or {}
        if isinstance(evidence, Mapping):
            for line in str(evidence.get("name_status") or "").splitlines():
                parts = line.split("\t")
                path = parts[-1].strip() if parts else ""
                if path and path not in sample_status_paths:
                    sample_status_paths.append(path)
            for path in evidence.get("untracked_paths", []) or []:
                path_text = str(path).strip()
                if path_text and path_text not in sample_status_paths:
                    sample_status_paths.append(path_text)

    conflict_path_counts: dict[str, int] = {}
    top_conflict_paths: list[str] = []
    safety_constraints = [
        "Do not discard dirty or untracked content unless it is proven redundant with the target ref.",
        "Prefer commits, merges, or explicit follow-up tasks over destructive cleanup.",
        "Keep todo, objective, discovery, and strategy files parseable after reconciliation.",
    ]
    success_signals = [
        "candidate_count_decreases",
        "dirty_worktree_group_count_decreases",
        "main_checkout_dirty_becomes_false",
        "cleanup_or_reconciliation_pass_processes_candidates",
    ]

    if kind == "preflight_merge_conflict":
        conflict_path_counts = {
            str(path): int(count)
            for path, count in (
                record.get("conflict_path_counts") or {}
                if isinstance(record.get("conflict_path_counts"), Mapping)
                else {}
            ).items()
        }
        top_conflict_paths = [
            path
            for path, _count in sorted(
                conflict_path_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        actions = [
            {
                "action": "bundle_preflight_conflicts_by_path",
                "scope": "backlogged_worktrees",
                "automation": "group blocked branches by shared conflict paths before resolving individual branches",
            },
            {
                "action": "resolve_markdown_and_discovery_conflicts_deterministically",
                "scope": "append_only_docs",
                "automation": "use deterministic append-only markdown/objective/todo merge repair where conflict paths are documentation or discovery files",
            },
            {
                "action": "resolve_code_or_submodule_conflicts_in_isolated_worktree",
                "scope": "code_and_gitlinks",
                "automation": "stage conflicts in a temporary reconciliation worktree or invoke the configured LLM resolver before mutating main",
            },
            {
                "action": "rerun_worktree_reconciliation",
                "scope": "backlogged_worktrees",
                "automation": "rerun reconcile_backlogged_worktrees and confirm preflight_blocked_count decreases",
            },
        ]
        safety_constraints = [
            "Do not run conflict-producing merges directly in main without a preflight or isolated resolver plan.",
            "Preserve submodule gitlink intent explicitly; never pick a gitlink side without recording why.",
            "Keep todo, objective, discovery, and strategy files parseable after reconciliation.",
        ]
        success_signals = [
            "preflight_blocked_count_decreases",
            "conflict_path_count_decreases",
            "reconciled_count_increases",
            "main_checkout_dirty_becomes_false",
        ]
    elif kind == "main_checkout_dirty":
        actions = [
            {
                "action": "classify_main_checkout_changes",
                "scope": "repo_root",
                "automation": "inspect git status, diff stats, submodule status, and generated artifacts before merges",
            },
            {
                "action": "preserve_or_split_main_checkout_work",
                "scope": "repo_root",
                "automation": "commit intentional changes or convert unresolved changes into follow-up tasks; never discard unknown work",
            },
            {
                "action": "rerun_worktree_reconciliation",
                "scope": "backlogged_worktrees",
                "automation": "rerun reconcile_backlogged_worktrees once the main checkout is clean enough to mutate",
            },
        ]
    else:
        actions = [
            {
                "action": "classify_dirty_worktree_group",
                "scope": "sampled_worktrees",
                "automation": "inspect sampled dirty statuses and compare against the target ref",
            },
            {
                "action": "preserve_or_merge_backlogged_work",
                "scope": "dirty_worktrees",
                "automation": "merge valuable branch work, commit preserved changes, or file follow-up tasks for unresolved work",
            },
            {
                "action": "rerun_cleanup_pass",
                "scope": "worktree_root",
                "automation": "rerun cleanup_backlogged_worktrees after preserving or merging dirty worktree content",
            },
        ]
        if reason == "content_not_in_target":
            actions.insert(
                1,
                {
                    "action": "compare_dirty_content_to_target",
                    "scope": "dirty_worktrees",
                    "automation": "separate real unmerged content from generated duplicates before deleting worktrees",
                },
            )
        elif reason == "unsupported_status":
            actions.insert(
                1,
                {
                    "action": "resolve_unsupported_statuses",
                    "scope": "dirty_worktrees",
                    "automation": "handle deletes, renames, unmerged paths, or unusual index states with an explicit resolver pass",
                },
            )

    return {
        "kind": kind,
        "reason": reason,
        "dedupe_key": str(record.get("dedupe_key") or ""),
        "fingerprint": str(record.get("fingerprint") or ""),
        "candidate_count": int(record.get("candidate_count") or 0),
        "sample_count": len(samples),
        "sample_branches": [str(item.get("branch") or "") for item in samples[:20] if str(item.get("branch") or "")],
        "sample_worktrees": [str(item.get("path") or "") for item in samples[:20] if str(item.get("path") or "")],
        "sample_status_paths": sample_status_paths[:40],
        "conflict_path_counts": conflict_path_counts,
        "top_conflict_paths": top_conflict_paths[:20],
        "main_dirty_evidence": main_dirty_evidence,
        "actions": actions,
        "safety_constraints": safety_constraints,
        "success_signals": success_signals,
    }


def reconciliation_guardrail_plan_markdown(record: Mapping[str, Any]) -> str:
    plan = reconciliation_guardrail_plan(record)
    action_lines = [
        f"- `{item['action']}`: {item['automation']}"
        for item in plan.get("actions", [])
        if isinstance(item, Mapping)
    ]
    constraint_lines = [f"- {item}" for item in plan.get("safety_constraints", [])]
    signal_lines = [f"- `{item}`" for item in plan.get("success_signals", [])]
    manifest = json.dumps(plan, indent=2, sort_keys=True)
    return f"""## Reconciliation Plan

Work surface: `{plan["candidate_count"]}` candidates, `{plan["sample_count"]}` sampled records.

### Suggested Actions

{chr(10).join(action_lines) or "- none"}

### Safety Constraints

{chr(10).join(constraint_lines) or "- none"}

### Success Signals

{chr(10).join(signal_lines) or "- none"}

## Machine Readable Manifest

```json
{manifest}
```
"""


def reconciliation_evidence_markdown(evidence: Mapping[str, Any] | None) -> str:
    if not isinstance(evidence, Mapping) or not evidence:
        return "- none"
    lines: list[str] = []
    path_categories = evidence.get("path_categories") or {}
    if isinstance(path_categories, Mapping) and path_categories:
        category_text = ", ".join(
            f"{key}={value}" for key, value in sorted(path_categories.items())
        )
        lines.append(f"- Path categories: `{category_text}`")
    status_paths = [str(item) for item in evidence.get("status_paths", []) if str(item).strip()]
    if status_paths:
        lines.append("- Status paths:")
        lines.extend(f"  - `{item}`" for item in status_paths[:20])
    for key, label in (
        ("name_status", "Name status"),
        ("staged_name_status", "Staged name status"),
        ("diff_stat", "Diff stat"),
        ("submodule_summary", "Submodule summary"),
    ):
        value = str(evidence.get(key) or "").strip()
        if not value:
            continue
        lines.append(f"- {label}:")
        lines.extend(f"  - `{line}`" for line in value.splitlines()[:20])
    untracked_paths = [str(item) for item in evidence.get("untracked_paths", []) if str(item).strip()]
    if untracked_paths:
        lines.append("- Untracked paths:")
        lines.extend(f"  - `{item}`" for item in untracked_paths[:20])
    return "\n".join(lines) or "- none"


def write_reconciliation_guardrail_discovery(
    *,
    discovery_dir: Path,
    task_id: str,
    record: Mapping[str, Any],
) -> Path:
    date = datetime.now(timezone.utc).date().isoformat()
    fingerprint = str(record.get("fingerprint") or "")
    path = discovery_dir / f"{date}-{task_id.lower()}-reconciliation-{fingerprint[:12]}.md"
    write_reconciliation_guardrail_discovery_path(
        path=path,
        task_id=task_id,
        record=record,
        date=date,
        discovery_dir=discovery_dir,
    )
    return path


def preserved_reconciliation_discovery_sections(
    existing_text: str,
    *,
    reopened: bool = False,
) -> list[str]:
    """Return manual resolution sections to carry across guardrail refreshes."""

    preserved: list[str] = []
    for match in re.finditer(r"^##\s+([^\n]+)\n.*?(?=^##\s+|\Z)", existing_text, flags=re.MULTILINE | re.DOTALL):
        title = " ".join(match.group(1).strip().lower().split())
        is_resolution = title == "resolution" or title.startswith("resolution ")
        is_history = title.startswith("historical resolution ")
        if not is_resolution and not is_history:
            continue
        section = match.group(0).strip()
        if reopened and is_resolution:
            section = re.sub(
                r"^##\s+[^\n]+",
                "## Historical Resolution Evidence (prior occurrence)",
                section,
                count=1,
            )
        if section and section not in preserved:
            preserved.append(section)
    return preserved


def reconciliation_discovery_path_is_owned(
    path: Path,
    *,
    discovery_dir: Path,
) -> bool:
    """Return whether an evidence path is a non-symlinked direct child."""

    try:
        if discovery_dir.is_symlink() or path.is_symlink():
            return False
        return (
            path.parent.resolve(strict=False)
            == discovery_dir.resolve(strict=False)
            and (not path.exists() or path.is_file())
        )
    except (OSError, RuntimeError):
        return False


def write_reconciliation_guardrail_discovery_path(
    *,
    path: Path,
    task_id: str,
    record: Mapping[str, Any],
    discovery_dir: Path,
    date: str | None = None,
    history_source_path: Path | None = None,
    reopened: bool = False,
) -> Path:
    discovery_dir.mkdir(parents=True, exist_ok=True)
    if discovery_dir.is_symlink():
        raise ValueError("reconciliation discovery directory must not be a symlink")
    if not reconciliation_discovery_path_is_owned(
        path,
        discovery_dir=discovery_dir,
    ):
        if path.is_symlink():
            raise ValueError(
                "reconciliation discovery destination must not be a symlink"
            )
        raise ValueError(
            "reconciliation discovery path must be a regular direct child "
            "of the configured directory"
        )

    date = date or datetime.now(timezone.utc).date().isoformat()
    fingerprint = str(record.get("fingerprint") or "")
    history_texts: list[str] = []
    for source_path in (path, history_source_path):
        if source_path is None:
            continue
        if not reconciliation_discovery_path_is_owned(
            source_path,
            discovery_dir=discovery_dir,
        ):
            continue
        try:
            source_text = source_path.read_text(encoding="utf-8")
        except OSError:
            continue
        if source_text not in history_texts:
            history_texts.append(source_text)
    preserved_sections: list[str] = []
    for history_text in history_texts:
        for section in preserved_reconciliation_discovery_sections(
            history_text,
            reopened=reopened,
        ):
            if section not in preserved_sections:
                preserved_sections.append(section)
    status_lines = "\n".join(f"- `{line}`" for line in record.get("status_short", []) or []) or "- none"
    main_checkout_evidence = reconciliation_evidence_markdown(
        record.get("main_dirty_evidence")
        if isinstance(record.get("main_dirty_evidence"), Mapping)
        else None
    )
    sample_lines = []
    for sample in record.get("samples", []) or []:
        if not isinstance(sample, Mapping):
            continue
        branch = str(sample.get("branch") or "unknown-branch")
        path_text = str(sample.get("path") or "unknown-path")
        status = "; ".join(str(line) for line in sample.get("status_short", []) or [])
        suffix = f" status: `{status}`" if status else ""
        sample_lines.append(f"- `{branch}` at `{path_text}`{suffix}")
        conflict_paths = [str(path).strip() for path in sample.get("conflict_paths", []) or [] if str(path).strip()]
        if conflict_paths:
            sample_lines.append("  - Conflict paths:")
            sample_lines.extend(f"    - `{path}`" for path in conflict_paths[:12])
        evidence = sample.get("dirty_evidence") or {}
        if isinstance(evidence, Mapping):
            diff_stat = str(evidence.get("diff_stat") or "").strip()
            name_status = str(evidence.get("name_status") or "").strip()
            untracked_paths = [str(item) for item in evidence.get("untracked_paths", []) if str(item).strip()]
            if name_status:
                sample_lines.append("  - Name status:")
                sample_lines.extend(f"    - `{line}`" for line in name_status.splitlines()[:12])
            if diff_stat:
                sample_lines.append("  - Diff stat:")
                sample_lines.extend(f"    - `{line}`" for line in diff_stat.splitlines()[:12])
            if untracked_paths:
                sample_lines.append("  - Untracked paths:")
                sample_lines.extend(f"    - `{path}`" for path in untracked_paths[:12])
    samples = "\n".join(sample_lines) or "- none"
    plan_markdown = reconciliation_guardrail_plan_markdown(record)
    content = f"""# {task_id} Reconciliation Guardrail

Date: {date}
Fingerprint: {fingerprint}
Kind: {record.get("kind")}
Reason: {record.get("reason")}
Candidate count: {record.get("candidate_count")}
Priority: {record.get("priority")}
Track: {record.get("track")}

## Main Checkout Status

{status_lines}

## Main Checkout Evidence

{main_checkout_evidence}

## Sample Branches Or Worktrees

{samples}

## Why This Blocks Progress

The implementation supervisor can only merge clean inactive implementation
worktrees when the main checkout is safe to mutate. Dirty main checkouts and
dirty backlogged worktrees are preserved until a deliberate reconciliation task
decides whether to commit, merge, discard generated duplicates, or split
unresolved work into follow-up tasks.

## Suggested Repair

Inspect the dirty paths and sampled worktrees, resolve any real work into
reviewable commits or follow-up tasks, rerun the supervisor reconciliation pass,
and verify that either the candidate merge count decreases or the dirty
worktree cleanup skip count decreases.

{plan_markdown.rstrip()}
"""
    if preserved_sections:
        content = content.rstrip() + "\n\n" + "\n\n".join(preserved_sections).rstrip() + "\n"
    path.write_text(content, encoding="utf-8")
    return path


@dataclass(frozen=True)
class ReconciliationGuardrailBoardProfile:
    """Strict task-board metadata inherited by reconciliation guardrails."""

    board_namespace: str = ""
    goal_id: str = ""
    graph_parents: str = ""
    bundle: str = ""
    parallel_lane: str = ""
    resource_class: str = ""


def reconciliation_guardrail_board_profile(
    todo_path: Path,
    *,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
) -> ReconciliationGuardrailBoardProfile:
    """Return a root-bound complete profile that is already board-valid.

    Reconciliation findings are operational tasks, but they still belong to
    the board that discovered them.  Copying one internally consistent profile
    avoids inventing a namespace, goal, bundle, lane, or resource vocabulary
    that a strict board validator may reject.  A profile is only admitted when
    every strict field is present on the same existing task.
    """

    try:
        tasks = parse_task_file(todo_path, task_header_prefix(task_prefix))
    except (OSError, UnicodeDecodeError):
        return ReconciliationGuardrailBoardProfile()
    profiles: list[ReconciliationGuardrailBoardProfile] = []
    parent_counts: dict[str, int] = {}
    for task in tasks:
        metadata = getattr(task, "metadata", {}) or {}
        profile = ReconciliationGuardrailBoardProfile(
            board_namespace=str(metadata.get("board namespace") or "").strip(),
            goal_id=str(metadata.get("goal id") or "").strip(),
            graph_parents=str(metadata.get("graph parents") or "").strip(),
            bundle=str(metadata.get("bundle") or "").strip(),
            parallel_lane=str(metadata.get("parallel lane") or "").strip(),
            resource_class=str(metadata.get("resource class") or "").strip(),
        )
        if all(
            (
                profile.board_namespace,
                profile.goal_id,
                profile.bundle,
                profile.parallel_lane,
                profile.resource_class,
            )
        ):
            profiles.append(profile)
            for parent in split_csv(str(metadata.get("graph parents") or "")):
                parent_counts[parent] = parent_counts.get(parent, 0) + 1
    # Operational review-only findings belong to the board root, not to the
    # most recently active implementation goal.  Binding them to a leaf goal
    # could otherwise make an operator cleanup artifact look like objective
    # completion evidence.
    for profile in profiles:
        if re.search(r"(?:^|[-_])G0+$", profile.goal_id, flags=re.IGNORECASE):
            return profile
    for root_goal_id in sorted(
        parent_counts,
        key=lambda item: (-parent_counts[item], item),
    ):
        for profile in profiles:
            if profile.goal_id == root_goal_id:
                return profile
    if profiles:
        return profiles[-1]
    return ReconciliationGuardrailBoardProfile()


def reconciliation_guardrail_safe_outputs(
    values: Sequence[Path | str],
    *,
    repo_root: Path,
) -> tuple[str, ...]:
    """Normalize output declarations and reject paths outside the repository."""

    try:
        root = repo_root.resolve()
    except OSError:
        root = repo_root
    outputs: list[str] = []
    for raw_value in values:
        value = str(raw_value or "").strip().replace("\\", "/")
        if not value or "\x00" in value:
            continue
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                value = candidate.resolve().relative_to(root).as_posix()
            except (OSError, ValueError):
                continue
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or not path.parts
            or ".." in path.parts
            or path.as_posix() in {".", ".."}
            or (path.parts and path.parts[0].endswith(":"))
        ):
            continue
        normalized = path.as_posix()
        if normalized not in outputs:
            outputs.append(normalized)
    return tuple(outputs)


def _reconciliation_guardrail_profile_lines(
    profile: ReconciliationGuardrailBoardProfile | None,
) -> list[str]:
    if profile is None:
        return []
    fields = (
        ("Board namespace", profile.board_namespace),
        ("Goal id", profile.goal_id),
        ("Graph parents", profile.graph_parents),
        ("Bundle", profile.bundle),
        ("Parallel lane", profile.parallel_lane),
        ("Resource class", profile.resource_class),
    )
    return [f"- {label}: {value}" for label, value in fields if value]


def reconciliation_guardrail_task_block(
    *,
    task_id: str,
    record: Mapping[str, Any],
    discovery_path: Path,
    todo_output_path: str,
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    board_profile: ReconciliationGuardrailBoardProfile | None = None,
    repo_root: Path | None = None,
) -> str:
    outputs = reconciliation_guardrail_safe_outputs(
        (discovery_output_path, todo_output_path),
        repo_root=repo_root or Path.cwd(),
    )
    provenance = reconciliation_guardrail_provenance_metadata(
        record=record,
        discovery_path=discovery_path,
    )
    profile_lines = _reconciliation_guardrail_profile_lines(board_profile)
    profile_markdown = "\n".join(profile_lines)
    if profile_markdown:
        profile_markdown = f"\n{profile_markdown}"
    return f"""## {task_id} {record.get("summary")}

- Status: blocked
- Completion: manual
- Is schedulable: false
- Review only: true
- Blocked reason: operator_reconciliation_required
- Priority: {record.get("priority") or "P1"}
- Track: {record.get("track") or "ops"}
{provenance}
- Fingerprint: {record.get("fingerprint") or ""}
- Dedupe key: {record.get("dedupe_key") or ""}
- Depends on:
- Outputs: {", ".join(outputs)}{profile_markdown}
- Validation: test -f {shlex.quote(str(discovery_path))}
- Acceptance: Reconciliation guardrail filed this because {record.get("candidate_count")} branch or worktree cleanup candidates are blocked by {record.get("reason")}. This task is intentionally operator-gated because unknown dirty checkout content must not be committed, stashed, or discarded automatically. Use evidence and the machine-readable reconciliation plan in {discovery_path}, reconcile the dirty checkout or dirty worktree group deliberately, then rerun the supervisor cleanup/reconciliation pass and confirm that the blocked candidate count decreases.
"""


def reconciliation_guardrail_provenance_metadata(
    *,
    record: Mapping[str, Any],
    discovery_path: Path,
) -> str:
    """Render the explicit provenance required for an operational appendix."""

    return "\n".join(
        (
            f"- Generated by: {RECONCILIATION_GUARDRAIL_SCHEMA}",
            f"- Reconciliation kind: {str(record.get('kind') or '')}",
            f"- Reconciliation reason: {str(record.get('reason') or '')}",
            f"- Reconciliation fingerprint: {str(record.get('fingerprint') or '')}",
            f"- Reconciliation discovery: {discovery_path}",
            "- Canonical board task: false",
        )
    )


def reconciliation_guardrail_block_has_provenance(
    block: str,
    record: Mapping[str, Any],
) -> bool:
    """Return whether an existing block carries exact reconciliation provenance."""

    validation_path = reconciliation_task_validation_path(block)
    if validation_path is None:
        return False
    expected = reconciliation_guardrail_provenance_metadata(
        record=record,
        discovery_path=validation_path,
    )
    return all(
        re.search(rf"^{re.escape(line)}$", block, flags=re.MULTILINE)
        for line in expected.splitlines()
    )


def reconciliation_record_matches_block(block: str, record: Mapping[str, Any]) -> bool:
    status_match = re.search(
        r"^-\s*Status:\s*(\S+)\s*$",
        block,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    resolved = (
        status_match is not None
        and status_match.group(1).casefold().replace("-", "_")
        in {
            "complete",
            "completed",
            "done",
            "succeeded",
        }
    )
    fingerprint = str(record.get("fingerprint") or "")
    dedupe_key = str(record.get("dedupe_key") or "")
    kind = str(record.get("kind") or "")
    reason = str(record.get("reason") or "")
    if resolved:
        # Preflight-conflict cards are persistent operator work: reopen the
        # same card so its discovery and strict board metadata are repaired
        # atomically. Other resolved findings remain append-only evidence and
        # a later regression receives a new task.
        if fingerprint and fingerprint in block:
            return True
        return bool(
            dedupe_key
            and dedupe_key in block
            and kind == "preflight_merge_conflict"
        )
    if fingerprint and fingerprint in block:
        return True
    if dedupe_key and dedupe_key in block:
        return True
    if kind == "main_checkout_dirty" and re.search(
        r"^##\s+\S+\s+Resolve dirty main checkout blocking \d+ worktree merges",
        block,
        flags=re.MULTILINE,
    ):
        return True
    if kind == "dirty_backlogged_worktree" and re.search(
        rf"^##\s+\S+\s+Resolve \d+ dirty backlogged worktrees blocked by {re.escape(reason)}",
        block,
        flags=re.MULTILINE,
    ):
        return True
    if kind == "preflight_merge_conflict" and re.search(
        r"^##\s+\S+\s+Resolve \d+ preflight-conflicting backlogged worktree merges",
        block,
        flags=re.MULTILINE,
    ):
        return True
    return False


def reconciliation_guardrail_blocks(
    todo_text: str,
    *,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
    include_completed: bool = False,
) -> list[dict[str, str]]:
    """Return active, supervisor-owned reconciliation guardrail cards.

    Retirement is intentionally limited to cards carrying the exact generated
    blocked-reason and reconciliation dedupe namespace.  Similar hand-authored
    tasks remain outside automatic completion authority.
    """

    records: list[dict[str, str]] = []
    heading_pattern = task_id_pattern(task_prefix)
    for _start, _end, block in task_blocks_with_spans(todo_text):
        heading = heading_pattern.match(block)
        if heading is None:
            continue
        blocked_reason = re.search(
            r"^-\s*Blocked reason:\s*(\S+)\s*$",
            block,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        dedupe_key = re.search(
            r"^-\s*Dedupe key:\s*(\S+)\s*$",
            block,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        status = re.search(
            r"^-\s*Status:\s*(\S+)\s*$",
            block,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if (
            blocked_reason is None
            or blocked_reason.group(1).casefold()
            != "operator_reconciliation_required"
            or dedupe_key is None
            or not dedupe_key.group(1).startswith(
                "reconciliation_guardrail:"
            )
        ):
            continue
        normalized_status = (
            status.group(1).casefold().replace("-", "_")
            if status is not None
            else ""
        )
        if not include_completed and normalized_status in {
            "complete",
            "completed",
            "done",
            "succeeded",
        }:
            continue
        fingerprint = re.search(
            r"^-\s*Fingerprint:\s*(\S+)\s*$",
            block,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        records.append(
            {
                "task_id": heading.group(1),
                "status": normalized_status,
                "dedupe_key": dedupe_key.group(1),
                "fingerprint": (
                    fingerprint.group(1)
                    if fingerprint is not None
                    else ""
                ),
            }
        )
    return records


def resolved_reconciliation_guardrail_keys(
    *,
    reconciliation_result: Mapping[str, Any] | None = None,
    cleanup_result: Mapping[str, Any] | None = None,
    replay_result: Mapping[str, Any] | None = None,
) -> set[str]:
    """Return guardrail identities backed by a conclusive clean rescan.

    A missing/disabled scan, unavailable checkout status, any effective dirty
    status (including staged or unknown status), and all cleanup/preflight
    guardrails currently fail closed.  A main-checkout guardrail can also
    retire when an exact rescan proves that its blocked candidate population
    reached zero and the replay and cleanup passes completed without residual
    work.  That second proof deliberately permits unrelated parent-checkout
    dirt because there is no longer a candidate for it to block.
    """

    reconciliation = (
        dict(reconciliation_result)
        if isinstance(reconciliation_result, Mapping)
        else {}
    )
    if (
        reconciliation.get("attempted") is True
        and reconciliation.get("main_checkout_status_available") is True
        and reconciliation.get("main_checkout_dirty") is False
        and isinstance(reconciliation.get("main_status_short"), list)
        and not reconciliation.get("main_status_short")
    ):
        return {"reconciliation_guardrail:main_checkout_dirty"}
    if (
        _zero_candidate_reconciliation_is_conclusive(reconciliation)
        and _reconciliation_replay_is_conclusive(replay_result)
        and _worktree_cleanup_is_conclusive(cleanup_result)
    ):
        return {"reconciliation_guardrail:main_checkout_dirty"}
    return set()


def _explicit_nonnegative_int(
    record: Mapping[str, Any],
    key: str,
) -> int | None:
    """Return an explicitly encoded non-negative integer, excluding booleans."""

    if key not in record:
        return None
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _zero_candidate_reconciliation_is_conclusive(
    reconciliation: Mapping[str, Any],
) -> bool:
    """Prove that reconciliation found no candidate or residual blocker."""

    zero_count_fields = (
        "candidate_count",
        "processed_count",
        "reconciled_count",
        "preflight_blocked_count",
        "preflight_resolver_escalation_count",
        "cleanup_count",
        "skipped_count",
    )
    main_status_short = reconciliation.get("main_status_short")
    main_checkout_dirty = reconciliation.get("main_checkout_dirty")
    if (
        reconciliation.get("attempted") is not True
        or reconciliation.get("main_checkout_status_available") is not True
        or reconciliation.get("main_checkout_status_error") != ""
        or not isinstance(main_checkout_dirty, bool)
        or not isinstance(main_status_short, list)
        or main_checkout_dirty != bool(main_status_short)
        or any(
            _explicit_nonnegative_int(reconciliation, key) != 0
            for key in zero_count_fields
        )
        or reconciliation.get("candidates") != []
        or reconciliation.get("processed") != []
        or reconciliation.get("skipped") != []
    ):
        return False
    return not any(
        reconciliation.get(key)
        for key in ("error", "errors", "exception_type")
    )


def _reconciliation_replay_is_conclusive(
    replay_result: Mapping[str, Any] | None,
) -> bool:
    """Prove that every replay candidate is settled without deferral."""

    if not isinstance(replay_result, Mapping):
        return False
    replay = dict(replay_result)
    counts = {
        key: _explicit_nonnegative_int(replay, key)
        for key in (
            "pending_count",
            "processed_count",
            "completed_count",
            "failed_count",
            "deferred_count",
        )
    }
    if any(value is None for value in counts.values()):
        return False
    pending_count = counts["pending_count"]
    processed_count = counts["processed_count"]
    completed_count = counts["completed_count"]
    reason = str(replay.get("reason") or "")
    if (
        pending_count != processed_count
        or counts["failed_count"] != 0
        or counts["deferred_count"] != 0
        or any(replay.get(key) for key in ("error", "errors", "exception_type"))
    ):
        return False
    results = replay.get("results")
    if (
        reason
        not in {
            "no_pending_reconciliation_replays",
            "reconciliation_replays_processed",
        }
        or not isinstance(results, list)
        or len(results) != pending_count
        or len(results) != processed_count
    ):
        return False
    if reason == "no_pending_reconciliation_replays":
        return (
            replay.get("attempted") is False
            and not results
            and not any(counts.values())
        )
    if replay.get("attempted") is not True or not results:
        return False

    completed_results = 0
    for result in results:
        if (
            not isinstance(result, Mapping)
            or result.get("attempted") is not True
            or result.get("settled") is not True
            or (
                result.get("completed") is not True
                and result.get("queued") is not True
            )
            or any(
                result.get(key)
                for key in ("error", "errors", "exception_type")
            )
        ):
            return False
        if result.get("completed") is True:
            completed_results += 1
    return completed_count == completed_results


def _worktree_cleanup_is_conclusive(
    cleanup_result: Mapping[str, Any] | None,
) -> bool:
    """Prove that cleanup observed no dirty, skipped, or failed worktree."""

    if not isinstance(cleanup_result, Mapping):
        return False
    cleanup = dict(cleanup_result)
    skipped_count = _explicit_nonnegative_int(cleanup, "skipped_count")
    removed_count = _explicit_nonnegative_int(cleanup, "removed_count")
    dirty_groups = cleanup.get("dirty_worktree_groups", {})
    removed = cleanup.get("removed")
    if (
        cleanup.get("attempted") is not True
        or _explicit_nonnegative_int(cleanup, "prune_returncode") != 0
        or skipped_count != 0
        or removed_count is None
        or cleanup.get("skipped") != []
        or not isinstance(dirty_groups, Mapping)
        or bool(dirty_groups)
        or not isinstance(removed, list)
        or len(removed) != removed_count
        or cleanup.get("reason")
        or any(cleanup.get(key) for key in ("error", "errors", "exception_type"))
    ):
        return False
    return all(_worktree_cleanup_removal_is_conclusive(item) for item in removed)


def _worktree_cleanup_removal_is_conclusive(item: Any) -> bool:
    """Prove one cleanup removal, including any branch deletion, succeeded."""

    if (
        not isinstance(item, Mapping)
        or item.get("removed") is not True
        or _explicit_nonnegative_int(item, "returncode") != 0
        or any(
            item.get(key)
            for key in ("error", "errors", "exception_type")
        )
    ):
        return False
    branch_delete = item.get("branch_delete")
    if branch_delete in (None, {}):
        return True
    return (
        isinstance(branch_delete, Mapping)
        and branch_delete.get("attempted") is True
        and branch_delete.get("deleted") is True
        and _explicit_nonnegative_int(branch_delete, "returncode") == 0
        and not any(
            branch_delete.get(key)
            for key in ("error", "errors", "exception_type")
        )
    )


def reconciliation_guardrail_refresh_is_noise(block: str, record: Mapping[str, Any]) -> bool:
    """Return whether refreshing an existing guardrail would only churn metadata."""

    if re.search(r"^- Status:\s+completed\s*$", block, flags=re.MULTILINE):
        return False
    kind = str(record.get("kind") or "")
    dedupe_key = str(record.get("dedupe_key") or "")
    stable_dedupe_kinds = {
        "dirty_backlogged_worktree",
        "main_checkout_dirty",
        "preflight_merge_conflict",
    }
    if kind not in stable_dedupe_kinds:
        return False
    if not dedupe_key or dedupe_key not in block:
        return False
    if not reconciliation_guardrail_block_has_provenance(block, record):
        return False

    heading = re.search(r"^##\s+\S+\s+(.*?)\s*$", block, flags=re.MULTILINE)
    if heading is None or heading.group(1) != str(record.get("summary") or ""):
        return False
    validation_path = reconciliation_task_validation_path(block)
    acceptance = re.search(
        r"^- Acceptance:\s+(.*?)\s*$",
        block,
        flags=re.MULTILINE,
    )
    if validation_path is None or acceptance is None:
        return False
    candidate_count = int(record.get("candidate_count") or 0)
    reason = str(record.get("reason") or "")
    acceptance_text = acceptance.group(1)
    return all(
        marker in acceptance_text
        for marker in (
            (
                "Reconciliation guardrail filed this because "
                f"{candidate_count} branch or worktree cleanup candidates"
            ),
            f"blocked by {reason}",
            "machine-readable reconciliation plan",
            str(validation_path),
        )
    )


def reconciliation_guardrail_discovery_needs_repair(
    path: Path | None,
    record: Mapping[str, Any],
    *,
    discovery_dir: Path,
) -> bool:
    """Return whether a deduplicated guardrail's required evidence is incomplete."""

    if path is None:
        return True
    if not reconciliation_discovery_path_is_owned(
        path,
        discovery_dir=discovery_dir,
    ):
        # Never read a board-supplied path outside the configured state root.
        return True
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return True
    manifest_match = re.search(
        r"^## Machine Readable Manifest\s*\n\s*```json\s*\n(.*?)\n```",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if manifest_match is None:
        return True
    try:
        manifest = json.loads(manifest_match.group(1))
    except json.JSONDecodeError:
        return True
    return (
        not isinstance(manifest, Mapping)
        or dict(manifest) != reconciliation_guardrail_plan(record)
    )

def task_blocks_with_spans(todo_text: str) -> list[tuple[int, int, str]]:
    starts = [match.start() for match in re.finditer(r"^##\s+\S+", todo_text, flags=re.MULTILINE)]
    blocks: list[tuple[int, int, str]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(todo_text)
        blocks.append((start, end, todo_text[start:end]))
    return blocks


def reconciliation_guardrail_block_is_completed(block: str) -> bool:
    """Return whether a reconciliation task block is immutable history."""

    match = re.search(r"^- Status:\s*(.*?)\s*$", block, flags=re.MULTILINE)
    if match is None:
        return False
    status = match.group(1).strip().lower().replace("-", "_").replace(" ", "_")
    return status in {"complete", "completed", "done"}


def reconciliation_task_validation_path(block: str) -> Path | None:
    match = re.search(r"^- Validation:\s+test -f\s+(.+?)\s*$", block, flags=re.MULTILINE)
    if not match:
        return None
    raw_path = match.group(1).strip()
    try:
        parts = shlex.split(raw_path)
    except ValueError:
        parts = [raw_path]
    if not parts:
        return None
    return Path(parts[0])


def reconciliation_refresh_discovery_path(
    path: Path,
    *,
    task_id: str,
    fingerprint: str,
    discovery_dir: Path,
) -> Path:
    """Return the fingerprint-bound evidence path for a refreshed incident."""

    date_match = re.match(r"^(?P<date>\d{4}-\d{2}-\d{2})-", path.name)
    date = (
        date_match.group("date")
        if date_match is not None
        else datetime.now(timezone.utc).date().isoformat()
    )
    return discovery_dir / (
        f"{date}-{task_id.lower()}-reconciliation-{fingerprint[:12]}.md"
    )


def _task_block_metadata_value(block: str, label: str) -> str:
    match = re.search(
        rf"^-\s+{re.escape(label)}:\s*(.*?)\s*$",
        block,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def _upsert_task_block_metadata(
    block: str,
    label: str,
    value: str,
    *,
    after: str,
) -> tuple[str, bool]:
    replacement = f"- {label}: {value}"
    pattern = rf"^-\s+{re.escape(label)}:\s*.*$"
    if re.search(pattern, block, flags=re.MULTILINE | re.IGNORECASE):
        updated = re.sub(
            pattern,
            replacement,
            block,
            count=1,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        return updated, updated != block
    anchor = re.search(
        rf"^-\s+{re.escape(after)}:\s*.*$",
        block,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if anchor is None:
        anchor = re.search(r"^##\s+\S+.*$", block, flags=re.MULTILINE)
    if anchor is None:
        return block, False
    updated = block[: anchor.end()] + f"\n{replacement}" + block[anchor.end() :]
    return updated, True


def reconciliation_guardrail_block_needs_refresh(
    block: str,
    *,
    board_profile: ReconciliationGuardrailBoardProfile | None = None,
    output_paths: Sequence[str] = (),
) -> bool:
    """Return whether a matching persistent blocker needs semantic repair."""

    required = (
        ("Status", "blocked"),
        ("Completion", "manual"),
        ("Is schedulable", "false"),
        ("Review only", "true"),
        ("Blocked reason", "operator_reconciliation_required"),
    )
    if any(_task_block_metadata_value(block, label).lower() != value for label, value in required):
        return True
    if output_paths:
        current_outputs = tuple(
            item.strip()
            for item in _task_block_metadata_value(block, "Outputs").split(",")
            if item.strip()
        )
        if current_outputs != tuple(output_paths):
            return True
    if board_profile is not None:
        expected_profile = (
            ("Board namespace", board_profile.board_namespace),
            ("Goal id", board_profile.goal_id),
            ("Graph parents", board_profile.graph_parents),
            ("Bundle", board_profile.bundle),
            ("Parallel lane", board_profile.parallel_lane),
            ("Resource class", board_profile.resource_class),
        )
        if any(
            value and _task_block_metadata_value(block, label) != value
            for label, value in expected_profile
        ):
            return True
    return False


def refresh_reconciliation_guardrail_block(
    block: str,
    record: Mapping[str, Any],
    *,
    discovery_dir: Path,
    discovery_output_path: str,
    todo_output_path: str,
    board_profile: ReconciliationGuardrailBoardProfile | None = None,
    output_paths: Sequence[str] = (),
) -> tuple[str, str, Path | None, bool]:
    heading_match = re.match(r"^##\s+(\S+)\s+[^\n]*", block)
    if not heading_match:
        return block, "", None, False
    task_id = heading_match.group(1)
    changed = False
    validation_path = reconciliation_task_validation_path(block)
    updated = re.sub(
        r"^##\s+\S+\s+.*$",
        f"## {task_id} {record.get('summary')}",
        block,
        count=1,
        flags=re.MULTILINE,
    )
    if updated != block:
        changed = True
    block = updated
    canonical_fields = (
        ("Status", "blocked", "heading"),
        ("Completion", "manual", "Status"),
        ("Is schedulable", "false", "Completion"),
        ("Review only", "true", "Is schedulable"),
        ("Blocked reason", "operator_reconciliation_required", "Review only"),
    )
    for label, value, after in canonical_fields:
        updated, field_changed = _upsert_task_block_metadata(
            block,
            label,
            value,
            after=after,
        )
        changed = changed or field_changed
        block = updated
    fingerprint = str(record.get("fingerprint") or "")
    dedupe_key = str(record.get("dedupe_key") or "")
    if fingerprint and re.search(r"^- Fingerprint:", block, flags=re.MULTILINE):
        updated = re.sub(r"^- Fingerprint:.*$", f"- Fingerprint: {fingerprint}", block, count=1, flags=re.MULTILINE)
        changed = changed or updated != block
        block = updated
    elif fingerprint:
        updated = re.sub(r"^- Track:.*$", lambda match: f"{match.group(0)}\n- Fingerprint: {fingerprint}", block, count=1, flags=re.MULTILINE)
        changed = changed or updated != block
        block = updated
    if dedupe_key and re.search(r"^- Dedupe key:", block, flags=re.MULTILINE):
        updated = re.sub(r"^- Dedupe key:.*$", f"- Dedupe key: {dedupe_key}", block, count=1, flags=re.MULTILINE)
        changed = changed or updated != block
        block = updated
    elif dedupe_key:
        updated = re.sub(
            r"^- Fingerprint:.*$",
            lambda match: f"{match.group(0)}\n- Dedupe key: {dedupe_key}",
            block,
            count=1,
            flags=re.MULTILINE,
        )
        changed = changed or updated != block
        block = updated
    if validation_path is not None and re.fullmatch(r"[0-9a-f]{40}", fingerprint):
        refreshed_validation_path = reconciliation_refresh_discovery_path(
            validation_path,
            task_id=task_id,
            fingerprint=fingerprint,
            discovery_dir=discovery_dir,
        )
        if refreshed_validation_path != validation_path:
            updated = re.sub(
                r"^- Validation:.*$",
                f"- Validation: test -f {shlex.quote(str(refreshed_validation_path))}",
                block,
                count=1,
                flags=re.MULTILINE,
            )
            changed = changed or updated != block
            block = updated
            validation_path = refreshed_validation_path
    validation_path = reconciliation_task_validation_path(block)
    if validation_path is not None:
        provenance_labels = (
            "Generated by",
            "Reconciliation kind",
            "Reconciliation reason",
            "Reconciliation fingerprint",
            "Reconciliation discovery",
            "Resolution receipt digest",
            "Canonical board task",
        )
        updated = block
        for label in provenance_labels:
            updated = re.sub(
                rf"^- {re.escape(label)}:.*\n?",
                "",
                updated,
                flags=re.MULTILINE,
            )
        provenance = reconciliation_guardrail_provenance_metadata(
            record=record,
            discovery_path=validation_path,
        )
        updated = re.sub(
            r"^- Track:.*$",
            lambda match: f"{match.group(0)}\n{provenance}",
            updated,
            count=1,
            flags=re.MULTILINE,
        )
        changed = changed or updated != block
        block = updated
    expected_outputs = (
        f"- Outputs: {discovery_output_path}, {todo_output_path}"
    )
    if re.search(r"^- Outputs:.*$", block, flags=re.MULTILINE):
        updated = re.sub(
            r"^- Outputs:.*$",
            expected_outputs,
            block,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        updated = re.sub(
            r"^- Validation:.*$",
            lambda match: f"{expected_outputs}\n{match.group(0)}",
            block,
            count=1,
            flags=re.MULTILINE,
        )
    changed = changed or updated != block
    block = updated
    if output_paths:
        updated, field_changed = _upsert_task_block_metadata(
            block,
            "Outputs",
            ", ".join(output_paths),
            after="Depends on",
        )
        changed = changed or field_changed
        block = updated
    profile_anchor = "Outputs"
    if board_profile is not None:
        profile_fields = (
            ("Board namespace", board_profile.board_namespace),
            ("Goal id", board_profile.goal_id),
            ("Graph parents", board_profile.graph_parents),
            ("Bundle", board_profile.bundle),
            ("Parallel lane", board_profile.parallel_lane),
            ("Resource class", board_profile.resource_class),
        )
        for label, value in profile_fields:
            if not value:
                continue
            updated, field_changed = _upsert_task_block_metadata(
                block,
                label,
                value,
                after=profile_anchor,
            )
            changed = changed or field_changed
            block = updated
            profile_anchor = label
    validation_path = reconciliation_task_validation_path(block)
    if validation_path is not None:
        replacement = (
            f"- Acceptance: Reconciliation guardrail filed this because {record.get('candidate_count')} "
            f"branch or worktree cleanup candidates are blocked by {record.get('reason')}. "
            f"Use evidence and the machine-readable reconciliation plan in {validation_path}, "
            "reconcile the dirty checkout or dirty worktree group deliberately, "
            "then rerun the supervisor cleanup/reconciliation pass and confirm that the blocked candidate count decreases."
        )
        updated = re.sub(r"^- Acceptance:.*$", replacement, block, count=1, flags=re.MULTILINE)
        changed = changed or updated != block
        block = updated
    return block, task_id, validation_path, changed


def refresh_existing_reconciliation_guardrails(
    *,
    todo_text: str,
    records: Sequence[Mapping[str, Any]],
    discovery_dir: Path,
    discovery_output_path: str,
    todo_output_path: str,
    board_profile: ReconciliationGuardrailBoardProfile | None = None,
    output_paths: Sequence[str] = (),
) -> tuple[str, list[dict[str, Any]]]:
    blocks = task_blocks_with_spans(todo_text)
    if not blocks:
        return todo_text, []
    replacements: dict[tuple[int, int], str] = {}
    refreshes: list[dict[str, Any]] = []
    for record in records:
        for start, end, block in blocks:
            if (start, end) in replacements:
                block = replacements[(start, end)]
            if not reconciliation_record_matches_block(block, record):
                continue
            if (
                reconciliation_guardrail_block_is_completed(block)
                and str(record.get("kind") or "")
                != "preflight_merge_conflict"
            ):
                # A completed operator reconciliation is immutable evidence.
                # A recurrence must receive a fresh task/discovery identity.
                continue
            needs_refresh = reconciliation_guardrail_block_needs_refresh(
                block,
                board_profile=board_profile,
                output_paths=output_paths,
            )
            if reconciliation_guardrail_refresh_is_noise(block, record) and not needs_refresh:
                validation_path = reconciliation_task_validation_path(block)
                if not reconciliation_guardrail_discovery_needs_repair(
                    validation_path,
                    record,
                    discovery_dir=discovery_dir,
                ):
                    break
            was_completed = bool(
                re.search(
                    r"^- Status:\s+completed\s*$",
                    block,
                    flags=re.MULTILINE,
                )
            )
            previous_validation_path = reconciliation_task_validation_path(block)
            refreshed_block, task_id, validation_path, changed = refresh_reconciliation_guardrail_block(
                block,
                record,
                discovery_dir=discovery_dir,
                discovery_output_path=discovery_output_path,
                todo_output_path=todo_output_path,
                board_profile=board_profile,
                output_paths=output_paths,
            )
            discovery_changed = False
            if validation_path is not None and task_id:
                before_discovery = ""
                if reconciliation_discovery_path_is_owned(
                    validation_path,
                    discovery_dir=discovery_dir,
                ):
                    try:
                        before_discovery = validation_path.read_text(
                            encoding="utf-8"
                        )
                    except OSError:
                        pass
                write_reconciliation_guardrail_discovery_path(
                    path=validation_path,
                    task_id=task_id,
                    record=record,
                    discovery_dir=discovery_dir,
                    history_source_path=(
                        previous_validation_path
                        if previous_validation_path != validation_path
                        else None
                    ),
                    reopened=was_completed,
                )
                try:
                    after_discovery = validation_path.read_text(encoding="utf-8")
                except OSError:
                    after_discovery = ""
                discovery_changed = before_discovery != after_discovery
            if changed:
                replacements[(start, end)] = refreshed_block
            if changed or discovery_changed:
                refreshes.append(
                    {
                        "follow_up_task_id": task_id,
                        "fingerprint": str(record.get("fingerprint") or ""),
                        "kind": str(record.get("kind") or ""),
                        "reason": str(record.get("reason") or ""),
                        "candidate_count": int(record.get("candidate_count") or 0),
                        "discovery_path": str(validation_path or ""),
                        "refreshed": True,
                        "reopened": was_completed,
                    }
                )
            break
    if not replacements:
        return todo_text, refreshes
    pieces: list[str] = []
    cursor = 0
    for start, end, block in blocks:
        pieces.append(todo_text[cursor:start])
        pieces.append(replacements.get((start, end), block))
        cursor = end
    pieces.append(todo_text[cursor:])
    return "".join(pieces), refreshes


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl_events(path, repair=True)


def event_merge_result(event: Mapping[str, Any]) -> dict[str, Any]:
    merge_result = event.get("merge_result") or {}
    if isinstance(merge_result, Mapping) and merge_result:
        return dict(merge_result)
    event_type = str(event.get("type") or "")
    if event_type in {"merge_reconcile_skipped", "merge_reconcile_exception"}:
        return {
            "attempted": True,
            "merged": False,
            "reason": str(event.get("reason") or event_type),
            "branch": str(event.get("branch") or ""),
        }
    return {}


def validation_result_is_failure(value: Any) -> bool:
    """Return whether a validation result represents a real failed gate.

    Pre-dispatch failures such as a missing impact declaration deliberately set
    ``attempted`` to false because no command was allowed to run.  They are
    still validation failures and must consume the validation retry budget
    instead of being misclassified as implementation failures.
    """

    if not isinstance(value, Mapping) or value.get("passed", False):
        return False
    if value.get("attempted", False):
        return True
    if value.get("error") or value.get("coverage_errors"):
        return True
    reason = str(value.get("reason") or "").strip()
    if reason and reason not in {"no_commands", "not_run"}:
        return True
    try:
        return int(value.get("returncode")) != 0
    except (TypeError, ValueError):
        return False


def validation_failure_label(
    validation: Mapping[str, Any],
    *,
    source_task: Any | None = None,
) -> str:
    """Return a stable command or typed pre-dispatch failure label."""

    failed_command = str(validation.get("failed_command") or "").strip()
    if failed_command:
        return failed_command
    if not validation.get("attempted", False):
        error = str(validation.get("error") or "validation_gate_failed").strip()
        reason = str(validation.get("reason") or "pre_dispatch").strip()
        return f"validation_pre_dispatch:{error}:{reason}"
    for node in validation.get("nodes", ()) or ():
        if not isinstance(node, Mapping):
            continue
        if str(node.get("disposition") or "") != "failed":
            continue
        command = str(node.get("command") or "").strip()
        if command:
            return command
    selection = validation.get("selection") or {}
    if isinstance(selection, Mapping):
        for decision in selection.get("decisions", ()) or ():
            if not isinstance(decision, Mapping) or not decision.get(
                "selected", False
            ):
                continue
            command = str(decision.get("command") or "").strip()
            if command:
                return command
    if source_task is not None:
        for command in getattr(source_task, "validation", ()) or ():
            if str(command).strip():
                return str(command).strip()
    return "validation_gate_failed"


def consecutive_validation_failures(events: Sequence[Mapping[str, Any]], task_id: str) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for event in reversed(events):
        if str(event.get("type") or "") != "implementation_finished":
            continue
        if str(event.get("task_id") or "") != task_id:
            continue
        validation = event.get("validation_result") or {}
        if not validation_result_is_failure(validation):
            break
        failures.append(dict(event))
    failures.reverse()
    return failures


def consecutive_merge_failures(events: Sequence[Mapping[str, Any]], task_id: str) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for event in reversed(events):
        event_type = str(event.get("type") or "")
        if event_type not in {
            "implementation_finished",
            "merge_reconciled",
            "merge_reconcile_skipped",
            "merge_reconcile_exception",
        }:
            continue
        if str(event.get("task_id") or "") != task_id:
            continue

        merge_result = event_merge_result(event)
        if event_type == "merge_reconciled" and event.get("resolved", False):
            break
        if event_type == "implementation_finished":
            validation = event.get("validation_result") or {}
            if validation_result_is_failure(validation):
                break
            if merge_result.get("merged", False):
                break

        if not merge_result.get("attempted", False):
            continue
        if merge_result.get("merged", False):
            break
        if str(merge_result.get("reason") or "") == "not_attempted":
            continue
        failures.append(dict(event))

    failures.reverse()
    return failures


def implementation_failure_label(event: Mapping[str, Any]) -> str:
    exception = event.get("exception_result") or {}
    if isinstance(exception, Mapping) and exception:
        exception_type = str(exception.get("exception_type") or "unknown")
        return f"implementation_exception:{exception_type}"
    try:
        returncode = int(event.get("returncode"))
    except (TypeError, ValueError):
        returncode = 1
    if returncode == 124:
        return "implementation_timeout"
    return f"implementation_command_returncode:{returncode}"


def consecutive_implementation_failures(events: Sequence[Mapping[str, Any]], task_id: str) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for event in reversed(events):
        if str(event.get("type") or "") != "implementation_finished":
            continue
        if str(event.get("task_id") or "") != task_id:
            continue

        validation = event.get("validation_result") or {}
        if validation_result_is_failure(validation):
            break

        merge_result = event_merge_result(event)
        if merge_result.get("attempted", False):
            if merge_result.get("merged", False):
                break
            if str(merge_result.get("reason") or "") != "not_attempted":
                break

        try:
            returncode = int(event.get("returncode"))
        except (TypeError, ValueError):
            returncode = 1 if event.get("exception_result") else 0
        if returncode == 0:
            break
        failures.append(dict(event))

    failures.reverse()
    return failures


def write_retry_budget_discovery(
    *,
    discovery_dir: Path,
    task_id: str,
    source_task_id: str,
    failed_command: str,
    failures: Sequence[Mapping[str, Any]],
    retry_budget: int,
    failure_kind: str = "validation",
) -> Path:
    date = datetime.now(timezone.utc).date().isoformat()
    suffix = (
        "merge-retry-budget"
        if failure_kind == "merge"
        else "implementation-retry-budget"
        if failure_kind == "implementation"
        else "retry-budget"
    )
    path = discovery_dir / f"{date}-{task_id.lower()}-{source_task_id.lower()}-{suffix}.md"
    discovery_dir.mkdir(parents=True, exist_ok=True)
    log_paths = [str(event.get("log_path") or "") for event in failures if event.get("log_path")]
    attempt_numbers = [str(event.get("attempt") or "") for event in failures if event.get("attempt")]
    merge_result = event_merge_result(failures[-1]) if failures and failure_kind == "merge" else {}
    merge_evidence = ""
    if merge_result:
        dirty_paths = merge_result.get("dirty_paths") or []
        dirty_paths_text = ", ".join(str(path) for path in dirty_paths) if isinstance(dirty_paths, list) else str(dirty_paths)
        merge_evidence = "\n".join(
            [
                f"- Merge reason: `{str(merge_result.get('reason') or 'not recorded')}`",
                f"- Dirty paths: {dirty_paths_text or 'not recorded'}",
                f"- Branch: `{str(merge_result.get('branch') or 'not recorded')}`",
                f"- Main worktree: `{str(merge_result.get('main_worktree_path') or 'not recorded')}`",
            ]
        )
    implementation_evidence = ""
    if failures and failure_kind == "implementation":
        latest = failures[-1]
        exception = latest.get("exception_result") or {}
        exception_text = ""
        if isinstance(exception, Mapping) and exception:
            exception_text = "\n".join(
                [
                    f"- Exception type: `{str(exception.get('exception_type') or 'not recorded')}`",
                    f"- Exception phase: `{str(exception.get('phase') or 'not recorded')}`",
                    f"- Exception message: {str(exception.get('message') or 'not recorded')}",
                ]
            )
        implementation_evidence = "\n".join(
            [
                f"- Return code: `{str(latest.get('returncode') or 'not recorded')}`",
                f"- Branch: `{str(latest.get('branch') or 'not recorded')}`",
                f"- Worktree: `{str(latest.get('worktree_path') or 'not recorded')}`",
                exception_text,
            ]
        ).strip()
    validation_evidence = ""
    if failures and failure_kind == "validation":
        latest_validation = failures[-1].get("validation_result") or {}
        if isinstance(latest_validation, Mapping):
            coverage_errors = latest_validation.get("coverage_errors") or []
            if isinstance(coverage_errors, str):
                coverage_errors = [coverage_errors]
            def _bounded_items(value: Any, *, limit: int = 12) -> list[str]:
                if isinstance(value, str):
                    value = [value]
                if not isinstance(value, Sequence):
                    return []
                items: list[str] = []
                for item in value:
                    compact = " ".join(str(item or "").split())[:300]
                    if compact and compact not in items:
                        items.append(compact)
                    if len(items) >= limit:
                        break
                return items

            failed_tests = _bounded_items(
                latest_validation.get("failed_tests")
            )
            failed_test_paths = _bounded_items(
                latest_validation.get("failed_test_paths")
            )
            validation_impact_paths = _bounded_items(
                latest_validation.get("validation_impact_paths"),
                limit=16,
            )
            failure_head = " ".join(
                str(latest_validation.get("failure_head") or "").split()
            )[:2000]
            validation_evidence = "\n".join(
                [
                    f"- Validation attempted: `{bool(latest_validation.get('attempted', False))}`",
                    f"- Validation return code: `{str(latest_validation.get('returncode') or 'not recorded')}`",
                    f"- Validation error: `{str(latest_validation.get('error') or 'not recorded')}`",
                    f"- Validation reason: `{str(latest_validation.get('reason') or 'not recorded')}`",
                    "- Failed tests: "
                    + (", ".join(failed_tests) or "not recorded"),
                    "- Failed test paths: "
                    + (", ".join(failed_test_paths) or "not recorded"),
                    "- Validation target paths: "
                    + (
                        ", ".join(validation_impact_paths)
                        or "not recorded"
                    ),
                    f"- Failure summary: {failure_head or 'not recorded'}",
                    "- Coverage errors: "
                    + (
                        ", ".join(str(item) for item in coverage_errors)
                        or "not recorded"
                    ),
                    f"- Configuration detail: {str(latest_validation.get('configuration_detail') or 'not recorded')[:1000]}",
                ]
            )
    content = f"""# {task_id} {failure_kind.title()} Retry-Budget Finding: {source_task_id}

Date: {date}
Source task: {source_task_id}
Follow-up task: {task_id}
Retry budget: {retry_budget}
Observed consecutive {failure_kind} failures: {len(failures)}

## Evidence

- Failed command: `{failed_command}`
- Attempts: {", ".join(attempt_numbers) or "not recorded"}
- Logs: {", ".join(log_paths) or "not recorded"}
{merge_evidence}
{implementation_evidence}
{validation_evidence}

## Guardrail Result

The accelerator backlog refinery classified this as backlog work instead of
allowing another implementation attempt to loop on the same failure. The source
task is added to the strategy `blocked_tasks` list and the follow-up task below
is appended for normal daemon parsing.
"""
    path.write_text(content, encoding="utf-8")
    return path


def _bounded_validation_failure_paths(
    failed_test_paths: Sequence[str],
    *,
    limit: int = 16,
) -> list[str]:
    """Return safe exact failed-test paths for diagnostics and validation."""

    raw_values: Sequence[Any]
    if isinstance(failed_test_paths, (str, bytes, bytearray)):
        raw_values = (failed_test_paths,)
    else:
        raw_values = failed_test_paths
    paths: list[str] = []
    for raw_path in raw_values:
        normalized = str(raw_path or "").strip().replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        parts = tuple(part for part in normalized.split("/") if part)
        canonical_path = "/".join(parts)
        suffix = Path(canonical_path).suffix.lower()
        test_named = bool(parts) and (
            any(part.lower() in {"test", "tests", "e2e"} for part in parts[:-1])
            or parts[-1].lower().startswith("test_")
            or "_test." in parts[-1].lower()
            or ".spec." in parts[-1].lower()
            or ".test." in parts[-1].lower()
        )
        if (
            not normalized
            or normalized.startswith("/")
            or "\0" in normalized
            or "\n" in normalized
            or "\r" in normalized
            or any(character in normalized for character in "*?[")
            or ".." in parts
            or (parts and parts[0].endswith(":"))
            or suffix
            not in {
                ".cjs",
                ".cts",
                ".js",
                ".jsx",
                ".mjs",
                ".mts",
                ".py",
                ".pyi",
                ".ts",
                ".tsx",
            }
            or not test_named
            or canonical_path in paths
        ):
            continue
        paths.append(canonical_path)
        if len(paths) >= max(1, int(limit)):
            break
    return paths


def _focused_npm_playwright_retry_command(
    command: str,
    *,
    failed_test_paths: Sequence[str],
) -> str:
    """Narrow an npm-prefix Playwright clause to exact reported test files.

    The original command is returned unchanged unless every transformation is
    source-bound and shell-safe. This preserves the fail-closed fallback for
    unknown runners, unqualified paths, and complex shell expressions.
    """

    exact_paths = _bounded_validation_failure_paths(failed_test_paths)
    if not exact_paths:
        return command
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError:
        return command
    if not tokens or any(
        token in {"|", "||", ";", "&", ">", ">>", "<", "<<"}
        or "$(" in token
        or "`" in token
        or "\n" in token
        for token in tokens
    ):
        return command

    clauses: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token == "&&":
            if not current:
                return command
            clauses.append(current)
            current = []
            continue
        current.append(token)
    if not current:
        return command
    clauses.append(current)

    focused = False
    rendered_clauses: list[str] = []
    for clause in clauses:
        executable = Path(clause[0]).name.lower().removesuffix(".cmd")
        if executable != "npm" or "test" not in clause:
            rendered_clauses.append(shlex.join(clause))
            continue
        prefix = ""
        for index, token in enumerate(clause):
            if token == "--prefix" and index + 1 < len(clause):
                prefix = clause[index + 1]
                break
            if token.startswith("--prefix="):
                prefix = token.split("=", 1)[1]
                break
        prefix = prefix.strip().replace("\\", "/").strip("/")
        if (
            not prefix
            or prefix in {".", ".."}
            or ".." in prefix.split("/")
            or prefix.split("/", 1)[0].endswith(":")
        ):
            rendered_clauses.append(shlex.join(clause))
            continue

        runner_paths: list[str] = []
        for path in exact_paths:
            expected_prefix = f"{prefix}/"
            if not path.startswith(expected_prefix):
                continue
            relative = path[len(expected_prefix) :]
            relative_parts = tuple(
                part for part in relative.split("/") if part
            )
            suffix = Path(relative).suffix.lower()
            test_named = (
                any(part in {"test", "tests", "e2e"} for part in relative_parts)
                or ".spec." in relative
                or ".test." in relative
            )
            if (
                not relative
                or ".." in relative_parts
                or suffix
                not in {
                    ".cjs",
                    ".cts",
                    ".js",
                    ".jsx",
                    ".mjs",
                    ".mts",
                    ".ts",
                    ".tsx",
                }
                or not test_named
            ):
                continue
            if relative not in runner_paths:
                runner_paths.append(relative)
        if not runner_paths:
            rendered_clauses.append(shlex.join(clause))
            continue
        rendered_clauses.append(shlex.join([*clause, *runner_paths]))
        focused = True

    if not focused:
        return command
    return " && ".join(rendered_clauses)


def validation_retry_task_block(
    *,
    task_id: str,
    source_task: Any,
    failed_command: str,
    discovery_path: Path,
    failed_test_paths: Sequence[str] = (),
    depends_on: Sequence[str] = (),
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    launch_playwright_validation_gate: bool = False,
) -> str:
    outputs = list(getattr(source_task, "outputs", []) or [])
    # Retry discovery is supervisor-written evidence that already exists
    # before this task is dispatched.  It is an input to the repair, not a
    # candidate artifact.  Declaring an ignored runtime discovery directory as
    # an output makes the proposal gate require an impossible staged change
    # and exhausts every retry before validation can run.
    _ = discovery_output_path
    exact_failure_paths = _bounded_validation_failure_paths(
        failed_test_paths
    )
    # A test runner can report files outside the source task's ownership.
    # Preserve those paths as bounded evidence and focused-validation inputs,
    # but never turn external diagnostics into write authority.
    validation_command = safe_retry_validation_command(
        failed_command,
        discovery_path=discovery_path,
    )
    validation_command = _focused_npm_playwright_retry_command(
        validation_command,
        failed_test_paths=exact_failure_paths,
    )
    validation_target_paths = list(exact_failure_paths)
    if not validation_target_paths:
        validation_target_paths.extend(
            infer_validation_impact_paths(validation_command)
        )
    validation_scope_label = (
        "validation failure paths"
        if exact_failure_paths
        else "validation target paths"
    )
    validation_scope_acceptance = (
        f" The declared {validation_scope_label} "
        f"({', '.join(validation_target_paths)}) are bounded "
        "diagnostic/read-only metadata: they may be inspected and used to "
        "focus validation, but do not grant write authority. Repair edits "
        "remain limited to the source task Outputs; do not weaken correct "
        "assertions or policy."
        if validation_target_paths
        else ""
    )
    launch_gate_acceptance = (
        f" For launch tasks, this repair validation preserves the {LAUNCH_PLAYWRIGHT_VALIDATION_GATE_EVIDENCE}."
        if launch_playwright_validation_gate
        else ""
    )
    execution_metadata = retry_task_execution_metadata(source_task)
    provenance_metadata = retry_budget_repair_provenance_metadata(
        source_task_id=source_task.task_id,
        failure_kind="validation",
        discovery_path=discovery_path,
    )
    validation_failure_metadata = (
        "- Validation failure paths: "
        + ", ".join(exact_failure_paths)
        + "\n- Validation failure path authority: diagnostic-read-only\n"
        if exact_failure_paths
        else ""
    )
    return f"""## {task_id} Resolve validation retry-budget failure for {source_task.task_id}

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: {", ".join(depends_on)}
- Outputs: {", ".join(outputs)}
- Validation: {validation_command}
{execution_metadata}
{provenance_metadata}
{validation_failure_metadata.rstrip()}
- Acceptance: Retry-budget guardrail filed this from repeated validation failures in {source_task.task_id}. Use evidence in {discovery_path} to fix the validation blocker, then mark this repair task completed so the supervisor can release {source_task.task_id} from strategy blocked_tasks.{validation_scope_acceptance}{launch_gate_acceptance}
"""


def retry_task_execution_metadata(
    source_task: Any,
    *,
    predicted_files: str | None = None,
) -> str:
    """Preserve reviewed execution and write-scope bounds on repair work."""

    raw_metadata = getattr(source_task, "metadata", {}) or {}
    if not isinstance(raw_metadata, Mapping):
        return ""
    metadata = {
        str(key).strip().lower().replace("_", " "): str(value).strip()
        for key, value in raw_metadata.items()
        if str(value).strip()
    }
    if predicted_files is not None:
        metadata["predicted files"] = str(predicted_files).strip()
    lines: list[str] = []
    inherited_fields = (
        ("provider role", "Provider role"),
        ("context budget tokens", "Context budget tokens"),
        ("parallel lane", "Parallel lane"),
        ("predicted files", "Predicted files"),
        ("allow concurrent with", "Allow concurrent with"),
        ("conflict policy", "Conflict policy"),
    )
    for field, label in inherited_fields:
        value = metadata.get(field, "")
        if value:
            lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def retry_budget_repair_provenance_metadata(
    *,
    source_task_id: str,
    failure_kind: str,
    discovery_path: Path,
) -> str:
    """Render tamper-evident provenance for an operational repair appendix."""

    return "\n".join(
        (
            "- Generated by: "
            f"{RETRY_BUDGET_REPAIR_SCHEMA}",
            f"- Retry repair source: {source_task_id}",
            f"- Retry failure kind: {failure_kind}",
            f"- Retry repair discovery: {discovery_path}",
            "- Canonical board task: false",
        )
    )


def safe_retry_validation_command(command: str, *, discovery_path: Path) -> str:
    """Return a parseable validation command for a retry-budget follow-up task."""

    stripped = normalize_validation_command_text(command)
    typed_failure_label = stripped.startswith(
        (
            "validation_pre_dispatch:",
            "validation_gate_failed",
        )
    )
    if stripped and not typed_failure_label:
        commands = split_validation_commands(stripped)
        try:
            for parsed_command in commands:
                shlex.split(parsed_command)
        except ValueError:
            commands = []
        if commands:
            return "; ".join(commands)
    return f"test -f {shlex.quote(str(discovery_path))}"


def task_requires_launch_playwright_validation(task: Any) -> bool:
    """Return whether a retry repair must keep the launch Playwright gate attached."""

    values = [
        getattr(task, "task_id", ""),
        getattr(task, "title", ""),
        getattr(task, "track", ""),
        getattr(task, "priority", ""),
        getattr(task, "completion", ""),
        getattr(task, "acceptance", ""),
        *(getattr(task, "outputs", []) or []),
        *(getattr(task, "validation", []) or []),
        *(getattr(task, "depends_on", []) or []),
    ]
    metadata = getattr(task, "metadata", {}) or {}
    if isinstance(metadata, Mapping):
        values.extend(str(value) for value in metadata.values())
    haystack = " ".join(str(value) for value in values).lower()
    if str(getattr(task, "track", "") or "").strip().lower() == "launch":
        return True
    return LAUNCH_PLAYWRIGHT_VALIDATION_GATE_EVIDENCE.lower() in haystack or (
        "playwright" in haystack and "launch" in haystack
    )


def launch_playwright_validation_repair_command(command: str, *, source_task: Any) -> str:
    """Append the launch Playwright validation gate to launch retry repairs."""

    stripped = str(command or "").strip()
    if not task_requires_launch_playwright_validation(source_task):
        return stripped
    lowered = stripped.lower()
    if any(marker.lower() in lowered for marker in LAUNCH_PLAYWRIGHT_VALIDATION_MARKERS):
        return stripped
    if not stripped:
        return LAUNCH_PLAYWRIGHT_VALIDATION_COMMAND
    return f"{stripped} && {LAUNCH_PLAYWRIGHT_VALIDATION_COMMAND}"


def implementation_retry_task_block(
    *,
    task_id: str,
    source_task: Any,
    discovery_path: Path,
    strategy_path: Path,
    depends_on: Sequence[str] = (),
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
) -> str:
    outputs = list(getattr(source_task, "outputs", []) or [])
    # As with validation repairs, discovery is pre-dispatch evidence rather
    # than an implementation output.  Keep the argument for configured-runner
    # API compatibility, but never grant or require write authority to it.
    _ = discovery_output_path
    validation_command = f"test -f {shlex.quote(str(discovery_path))}"
    execution_metadata = retry_task_execution_metadata(source_task)
    provenance_metadata = retry_budget_repair_provenance_metadata(
        source_task_id=source_task.task_id,
        failure_kind="implementation",
        discovery_path=discovery_path,
    )
    return f"""## {task_id} Resolve implementation retry-budget failure for {source_task.task_id}

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: {", ".join(depends_on)}
- Outputs: {", ".join(outputs)}
- Validation: {validation_command}
{execution_metadata}
{provenance_metadata}
- Acceptance: Implementation retry-budget guardrail filed this from repeated implementation failures in {source_task.task_id}. Use evidence in {discovery_path} to fix the setup, runtime, or timeout blocker, then mark this repair task completed so the supervisor can release {source_task.task_id} from strategy blocked_tasks.
"""


def merge_command_label(merge_result: Mapping[str, Any]) -> str:
    command = merge_result.get("command")
    if isinstance(command, list) and command:
        return shlex.join(str(part) for part in command)
    if command:
        return str(command)
    return f"git merge ({str(merge_result.get('reason') or 'merge_failed')})"


def merge_retry_task_block(
    *,
    task_id: str,
    source_task: Any,
    discovery_path: Path,
    strategy_path: Path,
    depends_on: Sequence[str] = (),
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
) -> str:
    outputs = list(getattr(source_task, "outputs", []) or [])
    # Retry discovery is already-written supervisor evidence.  A merge
    # repair reconciles the source implementation and therefore keeps the
    # source task's exact write authority; treating discovery as a candidate
    # output both broadens that authority and makes the proposal gate demand
    # a staged change to runtime state.
    _ = discovery_output_path
    source_validation_commands = [
        normalize_validation_command_text(str(command))
        for command in getattr(source_task, "validation", ()) or ()
        if normalize_validation_command_text(str(command))
    ]
    # A pre-existing discovery finding proves only that the merge failed; it
    # cannot prove that the source contract has since landed.  Re-run the
    # source task's declared gate so an unchanged merge target cannot retire
    # the repair merely because its diagnostic receipt exists.  Boards are
    # expected to give executable tasks a validation contract; fail closed if
    # a legacy task omitted one.
    validation_command = (
        " && ".join(source_validation_commands)
        if source_validation_commands
        else "false # merge repair source has no validation contract"
    )
    execution_metadata = retry_task_execution_metadata(source_task)
    provenance_metadata = retry_budget_repair_provenance_metadata(
        source_task_id=source_task.task_id,
        failure_kind="merge",
        discovery_path=discovery_path,
    )
    return f"""## {task_id} Resolve merge retry-budget failure for {source_task.task_id}

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: {", ".join(depends_on)}
- Outputs: {", ".join(outputs)}
- Validation: {validation_command}
{execution_metadata}
{provenance_metadata}
- Acceptance: Merge retry-budget guardrail filed this from repeated merge failures in {source_task.task_id}. Use evidence in {discovery_path} to fix the merge blocker, verify the intended implementation changes are committed in their owning repository or submodule, run `ipfs-accelerate-agent-merge-resolver --events-path ... --apply` when the conflict is semantic, then mark this repair task completed so the supervisor can release {source_task.task_id} from strategy blocked_tasks.
"""


def record_retry_budget_findings(
    *,
    todo_path: Path,
    events_path: Path,
    strategy_path: Path,
    discovery_dir: Path,
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
    validation_retry_budget: int = DEFAULT_VALIDATION_RETRY_BUDGET,
    merge_retry_budget: int = DEFAULT_MERGE_RETRY_BUDGET,
    implementation_retry_budget: int = DEFAULT_IMPLEMENTATION_RETRY_BUDGET,
    validation_depends_on: Sequence[str] = (),
    validation_task_command_transform: Callable[[str], str] | None = None,
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    commit_outputs: bool = False,
    repo_root: Path | None = None,
    commit_subject: str = "Agent: record retry-budget guardrail outputs",
) -> list[dict[str, Any]]:
    """Append follow-up tasks for repeated implementation, validation, or merge failures."""

    if not todo_path.exists():
        return []
    tasks = parse_task_file(todo_path, task_header_prefix(task_header_prefix_value))
    if not tasks:
        return []

    todo_text = todo_path.read_text(encoding="utf-8")
    task_ids = set(task_ids_from_todo_text(todo_text, task_prefix=task_prefix))
    completed_task_ids = {task.task_id for task in tasks if task.status == "completed"}
    retry_budget_repair_task_ids = {
        task.task_id
        for task in tasks
        if is_retry_budget_repair_task(task)
    }
    events = iter_jsonl(events_path)
    strategy = load_strategy(strategy_path)
    blocked_tasks = [str(item) for item in strategy.get("blocked_tasks", []) if str(item).strip()]
    findings: list[dict[str, Any]] = []
    generated_paths: list[Path] = []

    if implementation_retry_budget > 0:
        for task in tasks:
            if task.task_id in completed_task_ids:
                continue
            if task.task_id in retry_budget_repair_task_ids:
                continue
            marker = f"implementation retry-budget failure for {task.task_id}"
            if marker in todo_text:
                continue
            failures = consecutive_implementation_failures(events, task.task_id)
            if len(failures) < implementation_retry_budget:
                continue
            follow_up_task_id = next_task_id(todo_text, task_prefix=task_prefix)
            failed_command = implementation_failure_label(failures[-1])
            discovery_path = write_retry_budget_discovery(
                discovery_dir=discovery_dir,
                task_id=follow_up_task_id,
                source_task_id=task.task_id,
                failed_command=failed_command,
                failures=failures,
                retry_budget=implementation_retry_budget,
                failure_kind="implementation",
            )
            generated_paths.append(discovery_path)
            task_block = implementation_retry_task_block(
                task_id=follow_up_task_id,
                source_task=task,
                discovery_path=discovery_path,
                strategy_path=strategy_path,
                depends_on=task.depends_on,
                discovery_output_path=discovery_output_path,
            )
            todo_text = todo_text.rstrip() + "\n\n" + task_block.strip() + "\n"
            task_ids.add(follow_up_task_id)
            if task.task_id not in blocked_tasks:
                blocked_tasks.append(task.task_id)
            findings.append(
                {
                    "source_task_id": task.task_id,
                    "follow_up_task_id": follow_up_task_id,
                    "failure_count": len(failures),
                    "failed_command": failed_command,
                    "discovery_path": str(discovery_path),
                    "failure_kind": "implementation",
                }
            )

    if validation_retry_budget > 0:
        for task in tasks:
            if task.task_id in completed_task_ids:
                continue
            if task.task_id in retry_budget_repair_task_ids:
                continue
            marker = f"retry-budget failure for {task.task_id}"
            if marker in todo_text:
                continue
            failures = consecutive_validation_failures(events, task.task_id)
            if len(failures) < validation_retry_budget:
                continue
            latest_validation = failures[-1].get("validation_result") or {}
            failed_command = validation_failure_label(
                latest_validation,
                source_task=task,
            )
            follow_up_task_id = next_task_id(todo_text, task_prefix=task_prefix)
            discovery_path = write_retry_budget_discovery(
                discovery_dir=discovery_dir,
                task_id=follow_up_task_id,
                source_task_id=task.task_id,
                failed_command=failed_command,
                failures=failures,
                retry_budget=validation_retry_budget,
            )
            generated_paths.append(discovery_path)
            depends_on = list(validation_depends_on) if validation_depends_on else list(task.depends_on)
            validation_command = (
                validation_task_command_transform(failed_command)
                if validation_task_command_transform is not None
                else failed_command
            )
            validation_command = launch_playwright_validation_repair_command(
                validation_command,
                source_task=task,
            )
            launch_playwright_validation_gate = task_requires_launch_playwright_validation(task)
            task_block = validation_retry_task_block(
                task_id=follow_up_task_id,
                source_task=task,
                failed_command=validation_command,
                discovery_path=discovery_path,
                failed_test_paths=(
                    latest_validation.get("failed_test_paths") or ()
                ),
                depends_on=depends_on,
                discovery_output_path=discovery_output_path,
                launch_playwright_validation_gate=launch_playwright_validation_gate,
            )
            todo_text = todo_text.rstrip() + "\n\n" + task_block.strip() + "\n"
            task_ids.add(follow_up_task_id)
            if task.task_id not in blocked_tasks:
                blocked_tasks.append(task.task_id)
            findings.append(
                {
                    "source_task_id": task.task_id,
                    "follow_up_task_id": follow_up_task_id,
                    "failure_count": len(failures),
                    "failed_command": failed_command,
                    "discovery_path": str(discovery_path),
                    "failure_kind": "validation",
                    "launch_playwright_validation_gate": launch_playwright_validation_gate,
                }
            )

    if merge_retry_budget > 0:
        for task in tasks:
            if task.task_id in completed_task_ids:
                continue
            if task.task_id in retry_budget_repair_task_ids:
                continue
            marker = f"merge retry-budget failure for {task.task_id}"
            if marker in todo_text:
                continue
            failures = consecutive_merge_failures(events, task.task_id)
            if len(failures) < merge_retry_budget:
                continue
            latest_merge_result = event_merge_result(failures[-1])
            if not latest_merge_result:
                continue
            follow_up_task_id = next_task_id(todo_text, task_prefix=task_prefix)
            failed_command = merge_command_label(latest_merge_result)
            discovery_path = write_retry_budget_discovery(
                discovery_dir=discovery_dir,
                task_id=follow_up_task_id,
                source_task_id=task.task_id,
                failed_command=failed_command,
                failures=failures,
                retry_budget=merge_retry_budget,
                failure_kind="merge",
            )
            generated_paths.append(discovery_path)
            task_block = merge_retry_task_block(
                task_id=follow_up_task_id,
                source_task=task,
                discovery_path=discovery_path,
                strategy_path=strategy_path,
                depends_on=task.depends_on,
                discovery_output_path=discovery_output_path,
            )
            todo_text = todo_text.rstrip() + "\n\n" + task_block.strip() + "\n"
            task_ids.add(follow_up_task_id)
            if task.task_id not in blocked_tasks:
                blocked_tasks.append(task.task_id)
            findings.append(
                {
                    "source_task_id": task.task_id,
                    "follow_up_task_id": follow_up_task_id,
                    "failure_count": len(failures),
                    "failed_command": failed_command,
                    "discovery_path": str(discovery_path),
                    "failure_kind": "merge",
                }
            )

    if not findings:
        return []

    todo_path.write_text(todo_text, encoding="utf-8")
    strategy["blocked_tasks"] = blocked_tasks
    strategy["last_retry_budget_guardrail_at"] = utc_now()
    strategy["retry_budget_findings"] = findings
    write_json(strategy_path, strategy)
    if commit_outputs:
        generated_paths.insert(0, todo_path)
        commit_results = commit_generated_outputs(
            generated_paths,
            repo_root=repo_root or todo_path.parent,
            subject=commit_subject,
        )
        if commit_results:
            strategy["last_retry_budget_commit_results"] = commit_results
            write_json(strategy_path, strategy)
    return findings


def record_dependency_guardrail_findings(
    *,
    todo_path: Path,
    strategy_path: Path,
    discovery_dir: Path,
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
    max_findings: int = DEFAULT_DEPENDENCY_GUARDRAIL_MAX_FINDINGS,
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    commit_outputs: bool = False,
    repo_root: Path | None = None,
    commit_subject: str = "Agent: record dependency guardrail outputs",
) -> list[dict[str, Any]]:
    """Append ready repair tasks for missing or self-referential dependencies."""

    if max_findings <= 0 or not todo_path.exists():
        return []
    tasks = parse_task_file(todo_path, task_header_prefix(task_header_prefix_value))
    if not tasks:
        return []

    todo_text = todo_path.read_text(encoding="utf-8")
    strategy = load_strategy(strategy_path)
    blocked_tasks = [str(item) for item in strategy.get("blocked_tasks", []) if str(item).strip()]
    seen = {str(item) for item in strategy.get("dependency_guardrail_seen_fingerprints", []) if str(item).strip()}
    open_dependency_repair_sources: set[str] = set()
    for task in tasks:
        match = re.match(
            r"^Resolve dependency guardrail for (\S+)\s*$",
            str(getattr(task, "title", "") or "").strip(),
        )
        if match is None:
            continue
        if (
            str(getattr(task, "status", "") or "").lower()
            in {"complete", "completed", "done", "succeeded"}
        ):
            continue
        open_dependency_repair_sources.add(match.group(1))
    records = [
        record
        for record in dependency_guardrail_records(tasks)
        if str(record.get("source_task_id") or "")
        not in open_dependency_repair_sources
    ][:max_findings]
    if not records:
        return []

    findings: list[dict[str, Any]] = []
    generated_paths: list[Path] = []
    try:
        todo_output_path = todo_path.resolve().relative_to((repo_root or todo_path.parent).resolve()).as_posix()
    except ValueError:
        todo_output_path = todo_path.as_posix()
    for record in records:
        follow_up_task_id = next_task_id(todo_text, task_prefix=task_prefix)
        discovery_path = write_dependency_guardrail_discovery(
            discovery_dir=discovery_dir,
            task_id=follow_up_task_id,
            record=record,
        )
        generated_paths.append(discovery_path)
        source_task_id = str(record.get("source_task_id") or "")
        task_block = dependency_guardrail_task_block(
            task_id=follow_up_task_id,
            source_task_id=source_task_id,
            discovery_path=discovery_path,
            todo_output_path=todo_output_path,
            discovery_output_path=discovery_output_path,
        )
        todo_text = todo_text.rstrip() + "\n\n" + task_block.strip() + "\n"
        if source_task_id and source_task_id not in blocked_tasks:
            blocked_tasks.append(source_task_id)
        findings.append(
            {
                "source_task_id": source_task_id,
                "follow_up_task_id": follow_up_task_id,
                "missing_dependencies": list(record.get("missing_dependencies", []) or []),
                "self_references": list(record.get("self_references", []) or []),
                "dependency_cycle": list(record.get("dependency_cycle", []) or []),
                "duplicate_task_id": str(record.get("duplicate_task_id") or ""),
                "duplicate_task_lines": list(record.get("duplicate_task_lines", []) or []),
                "discovery_path": str(discovery_path),
                "fingerprint": str(record.get("fingerprint") or ""),
            }
        )

    todo_path.write_text(todo_text, encoding="utf-8")
    strategy["blocked_tasks"] = blocked_tasks
    strategy["dependency_guardrail_seen_fingerprints"] = sorted(
        seen | {str(record.get("fingerprint") or "") for record in records if record.get("fingerprint")}
    )
    strategy["last_dependency_guardrail_at"] = utc_now()
    prior_findings = [
        dict(item)
        for item in strategy.get("dependency_guardrail_findings", [])
        if isinstance(item, Mapping)
    ]
    findings_by_identity: dict[str, dict[str, Any]] = {}
    for item in [*prior_findings, *findings]:
        identity = str(item.get("fingerprint") or "").strip()
        if not identity:
            identity = "|".join(
                [
                    str(item.get("source_task_id") or "").strip(),
                    str(item.get("follow_up_task_id") or "").strip(),
                ]
            )
        if identity:
            findings_by_identity[identity] = item
    strategy["dependency_guardrail_findings"] = list(
        findings_by_identity.values()
    )
    write_json(strategy_path, strategy)
    if commit_outputs:
        generated_paths.insert(0, todo_path)
        commit_results = commit_generated_outputs(
            generated_paths,
            repo_root=repo_root or todo_path.parent,
            subject=commit_subject,
        )
        if commit_results:
            strategy["last_dependency_guardrail_commit_results"] = commit_results
            write_json(strategy_path, strategy)
    return findings


def record_reconciliation_guardrail_findings(
    *,
    todo_path: Path,
    strategy_path: Path,
    discovery_dir: Path,
    reconciliation_result: Mapping[str, Any] | None = None,
    cleanup_result: Mapping[str, Any] | None = None,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
    max_findings: int = DEFAULT_RECONCILIATION_GUARDRAIL_MAX_FINDINGS,
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    commit_outputs: bool = False,
    repo_root: Path | None = None,
    commit_subject: str = "Agent: record reconciliation guardrail outputs",
    additional_generated_status_paths: Sequence[Path | str] = (),
    additional_generated_status_prefixes: Sequence[Path | str] = (),
) -> list[dict[str, Any]]:
    """Append deliberate cleanup tasks for blocked worktree reconciliation."""

    if max_findings <= 0 or not todo_path.exists():
        return []
    todo_text = todo_path.read_text(encoding="utf-8")
    strategy = load_strategy(strategy_path)
    seen = {
        str(item)
        for item in strategy.get("reconciliation_guardrail_seen_fingerprints", [])
        if str(item).strip()
    }
    active_guardrail_blocks = [
        block
        for _start, _end, block in task_blocks_with_spans(todo_text)
        if not re.search(
            r"^-\s*Status:\s*(?:complete|completed|done|succeeded)\s*$",
            block,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    ]

    def matching_blocks(record: Mapping[str, Any]) -> list[str]:
        return [
            block
            for _start, _end, block in task_blocks_with_spans(todo_text)
            if reconciliation_record_matches_block(block, record)
        ]

    def already_present(record: Mapping[str, Any]) -> bool:
        return any(
            not reconciliation_guardrail_block_is_completed(block)
            for block in matching_blocks(record)
        )

    def suppressed_by_seen_history(record: Mapping[str, Any]) -> bool:
        fingerprint = str(record.get("fingerprint") or "")
        if not fingerprint or fingerprint not in seen:
            return False
        # A completed matching task proves the fingerprint was handled only
        # for that historical incident.  A currently observed recurrence must
        # not be hidden by the strategy's append-only seen set.
        return not any(
            reconciliation_guardrail_block_is_completed(block)
            for block in matching_blocks(record)
        )

    filter_repo_root = (repo_root or todo_path.parent).resolve()
    generated_paths, generated_prefixes = generated_guardrail_status_filters(
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        repo_root=filter_repo_root,
        additional_generated_paths=additional_generated_status_paths,
        additional_generated_prefixes=additional_generated_status_prefixes,
    )
    board_profile = reconciliation_guardrail_board_profile(
        todo_path,
        task_prefix=task_prefix,
    )
    try:
        todo_output_path = todo_path.resolve().relative_to(filter_repo_root).as_posix()
    except ValueError:
        todo_output_path = todo_path.as_posix()
    guardrail_output_paths = reconciliation_guardrail_safe_outputs(
        (discovery_output_path, todo_output_path),
        repo_root=filter_repo_root,
    )
    all_records = reconciliation_guardrail_records(
        reconciliation_result=reconciliation_result,
        cleanup_result=cleanup_result,
        generated_status_paths=generated_paths,
        generated_status_prefixes=generated_prefixes,
    )
    try:
        todo_output_path = todo_path.resolve().relative_to(
            (repo_root or todo_path.parent).resolve()
        ).as_posix()
    except ValueError:
        todo_output_path = todo_path.as_posix()
    refreshed_todo_text, refreshes = refresh_existing_reconciliation_guardrails(
        todo_text=todo_text,
        records=all_records,
        discovery_dir=discovery_dir,
        discovery_output_path=discovery_output_path,
        todo_output_path=todo_output_path,
        board_profile=board_profile,
        output_paths=guardrail_output_paths,
    )
    if refreshes:
        todo_text = refreshed_todo_text
        active_guardrail_blocks = [
            block
            for _start, _end, block in task_blocks_with_spans(todo_text)
            if not re.search(
                r"^-\s*Status:\s*(?:complete|completed|done|succeeded)\s*$",
                block,
                flags=re.IGNORECASE | re.MULTILINE,
            )
        ]

    records = [
        record
        for record in all_records
        if not suppressed_by_seen_history(record)
        and not already_present(record)
    ][:max_findings]
    if not records and not refreshes:
        return []

    findings: list[dict[str, Any]] = []
    generated_paths: list[Path] = []
    for record in records:
        follow_up_task_id = next_task_id(todo_text, task_prefix=task_prefix)
        discovery_path = write_reconciliation_guardrail_discovery(
            discovery_dir=discovery_dir,
            task_id=follow_up_task_id,
            record=record,
        )
        generated_paths.append(discovery_path)
        task_block = reconciliation_guardrail_task_block(
            task_id=follow_up_task_id,
            record=record,
            discovery_path=discovery_path,
            todo_output_path=todo_output_path,
            discovery_output_path=discovery_output_path,
            board_profile=board_profile,
            repo_root=filter_repo_root,
        )
        todo_text = todo_text.rstrip() + "\n\n" + task_block.strip() + "\n"
        findings.append(
            {
                "follow_up_task_id": follow_up_task_id,
                "fingerprint": str(record.get("fingerprint") or ""),
                "kind": str(record.get("kind") or ""),
                "reason": str(record.get("reason") or ""),
                "candidate_count": int(record.get("candidate_count") or 0),
                "discovery_path": str(discovery_path),
                "sample_count": len(record.get("samples", []) or []),
            }
        )

    todo_path.write_text(todo_text, encoding="utf-8")
    strategy["reconciliation_guardrail_seen_fingerprints"] = sorted(
        seen | {str(record.get("fingerprint") or "") for record in records if record.get("fingerprint")}
    )
    strategy["last_reconciliation_guardrail_at"] = utc_now()
    strategy["reconciliation_guardrail_findings"] = [*refreshes, *findings]
    write_json(strategy_path, strategy)
    if commit_outputs and (generated_paths or refreshes):
        generated_paths.insert(0, todo_path)
        generated_paths.extend(
            Path(item["discovery_path"])
            for item in refreshes
            if str(item.get("discovery_path") or "").strip()
        )
        commit_results = commit_generated_outputs(
            generated_paths,
            repo_root=repo_root or todo_path.parent,
            subject=commit_subject,
        )
        if commit_results:
            strategy["last_reconciliation_guardrail_commit_results"] = commit_results
            write_json(strategy_path, strategy)
    return [*refreshes, *findings]


def completed_retry_budget_repairs_by_source(tasks: Sequence[Any]) -> dict[str, dict[str, str]]:
    """Map source task ids to completed retry-budget repair task metadata."""

    repairs: dict[str, dict[str, str]] = {}
    for task in tasks:
        if str(getattr(task, "status", "") or "").lower() != "completed":
            continue
        source_task_id, failure_kind = retry_budget_repair_source(task)
        if not source_task_id:
            continue
        repairs[source_task_id] = {
            "follow_up_task_id": str(getattr(task, "task_id", "") or ""),
            "failure_kind": failure_kind,
        }
    return repairs


def repair_generated_packet_internal_dependencies(
    todo_text: str,
    *,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
) -> tuple[str, list[dict[str, Any]]]:
    """Remove proven packet-internal goal prerequisites from aggregates.

    Objective packet aggregates intentionally collapse several goals into one
    execution unit.  Older projections retained a packet goal in ``Depends
    on`` even when the same aggregate carried explicit completion evidence for
    that goal.  The goal-to-task projection then produced a self-cycle.

    This repair is deliberately narrow: it accepts only active, schedulable,
    generated packet aggregates with canonical packet identities and non-empty
    evidence bindings.  Ambiguous or hand-authored metadata is left unchanged
    for the normal fail-closed dependency guardrail.
    """

    replacements: list[tuple[int, int, str]] = []
    repairs: list[dict[str, Any]] = []

    def field(block: str, label: str) -> str:
        match = re.search(
            rf"^-\s*{re.escape(label)}:\s*(.*?)\s*$",
            block,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        return match.group(1).strip() if match is not None else ""

    for start, end, block in task_blocks_with_spans(todo_text):
        heading = task_id_pattern(task_prefix).match(block)
        if heading is None:
            continue
        task_id = heading.group(1)
        status = field(block, "Status").casefold().replace("-", "_")
        if status in {"complete", "completed", "done", "succeeded"}:
            continue
        if (
            field(block, "Candidate kind").casefold()
            != "goal_packet_aggregate"
            or field(block, "Goal packet role").casefold()
            != "packet_aggregate"
            or field(block, "Is schedulable").casefold() != "true"
            or field(block, "Review only").casefold() != "false"
        ):
            continue
        semantic_identity = field(block, "Semantic identity")
        evidence_obligation_key = field(block, "Evidence obligation key")
        if (
            not semantic_identity.startswith(
                "objective-evidence-packet/v1/"
            )
            or not evidence_obligation_key.startswith(
                "objective-evidence-packet/v1/"
            )
        ):
            continue
        packet_key = field(block, "Goal packet")
        packet_goal_ids = {
            goal_id
            for goal_id in split_csv(field(block, "Goal packet goals"))
            if goal_id
        }
        if not packet_key or not packet_goal_ids:
            continue
        try:
            raw_bindings = json.loads(
                field(block, "Completion goal bindings")
            )
        except json.JSONDecodeError:
            continue
        if not isinstance(raw_bindings, Mapping):
            continue
        evidenced_packet_goals = {
            str(goal_id).strip()
            for goal_id, requirements in raw_bindings.items()
            if str(goal_id).strip() in packet_goal_ids
            and isinstance(requirements, list)
            and any(str(requirement).strip() for requirement in requirements)
        }
        if not evidenced_packet_goals:
            continue
        dependencies = split_csv(field(block, "Depends on"))
        removed_dependencies = [
            dependency
            for dependency in dependencies
            if dependency in evidenced_packet_goals
        ]
        if not removed_dependencies:
            continue
        retained_dependencies = [
            dependency
            for dependency in dependencies
            if dependency not in evidenced_packet_goals
        ]
        updated_block, replacement_count = re.subn(
            r"^-\s*Depends on:.*$",
            f"- Depends on: {', '.join(retained_dependencies)}",
            block,
            count=1,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if replacement_count != 1:
            continue
        repair_fingerprint = sha256(
            json.dumps(
                {
                    "task_id": task_id,
                    "packet_key": packet_key,
                    "removed_dependencies": removed_dependencies,
                    "retained_dependencies": retained_dependencies,
                    "semantic_identity": semantic_identity,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        replacements.append((start, end, updated_block))
        repairs.append(
            {
                "source_task_id": task_id,
                "follow_up_task_id": "",
                "guardrail_kind": "dependency_guardrail",
                "reason": "objective_packet_internal_dependency_removed",
                "packet_key": packet_key,
                "removed_dependencies": removed_dependencies,
                "retained_dependencies": retained_dependencies,
                "repair_fingerprint": repair_fingerprint,
            }
        )

    if not replacements:
        return todo_text, []
    updated_parts: list[str] = []
    cursor = 0
    for start, end, updated_block in replacements:
        updated_parts.append(todo_text[cursor:start])
        updated_parts.append(updated_block)
        cursor = end
    updated_parts.append(todo_text[cursor:])
    return "".join(updated_parts), repairs


def release_completed_guardrail_blocks(
    *,
    todo_path: Path,
    strategy_path: Path,
    reconciliation_result: Mapping[str, Any] | None = None,
    cleanup_result: Mapping[str, Any] | None = None,
    replay_result: Mapping[str, Any] | None = None,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
    commit_outputs: bool = False,
    repo_root: Path | None = None,
    commit_subject: str = "Agent: retire resolved guardrail tasks",
) -> list[dict[str, Any]]:
    """Unblock source tasks after guardrail repair or stale strategy state clears."""

    if not todo_path.exists() or not strategy_path.exists():
        return []
    with locked_taskboard(todo_path) as taskboard:
        todo_text = taskboard.read()
        todo_text, packet_dependency_repairs = (
            repair_generated_packet_internal_dependencies(
                todo_text,
                task_prefix=task_prefix,
            )
        )
        if packet_dependency_repairs:
            replace_locked_taskboard(taskboard, todo_text)
    statuses = task_statuses_from_todo_text(todo_text, task_prefix=task_prefix)
    if not statuses:
        return []
    tasks = parse_task_file(todo_path, task_header_prefix(task_prefix))
    completed_retry_repairs = completed_retry_budget_repairs_by_source(tasks)
    retry_budget_repair_sources_by_task_id = {
        str(getattr(task, "task_id", "") or ""): retry_budget_repair_source(task)
        for task in tasks
    }
    retry_budget_repair_task_ids = {
        task_id
        for task_id, (source_task_id, _failure_kind) in retry_budget_repair_sources_by_task_id.items()
        if source_task_id
    }
    pending_retry_repair_sources = {
        source_task_id
        for task in tasks
        if str(getattr(task, "status", "") or "").lower() != "completed"
        for source_task_id, _failure_kind in (retry_budget_repair_source(task),)
        if source_task_id
    }
    strategy = load_strategy(strategy_path)
    blocked_tasks = [str(item) for item in strategy.get("blocked_tasks", []) if str(item).strip()]
    todo_changed = bool(packet_dependency_repairs)

    releases: list[dict[str, Any]] = list(packet_dependency_repairs)
    if packet_dependency_repairs:
        strategy["last_objective_packet_dependency_repair_at"] = utc_now()
        strategy["last_objective_packet_dependency_repairs"] = list(
            packet_dependency_repairs
        )

    newly_retired_retry_task_ids: set[str] = set()
    source_completed_retry_repairs = {
        task_id: (source_task_id, failure_kind)
        for task_id, (
            source_task_id,
            failure_kind,
        ) in retry_budget_repair_sources_by_task_id.items()
        if task_id
        and statuses.get(task_id) != "completed"
        and statuses.get(source_task_id) == "completed"
    }
    if source_completed_retry_repairs:
        todo_text, retired_task_ids = mark_task_statuses_in_todo_text(
            todo_text,
            list(source_completed_retry_repairs),
            task_prefix=task_prefix,
            status="completed",
        )
        if retired_task_ids:
            todo_path.write_text(todo_text, encoding="utf-8")
            todo_changed = True
            statuses.update(
                {task_id: "completed" for task_id in retired_task_ids}
            )
            retired_task_id_set = set(retired_task_ids)
            newly_retired_retry_task_ids.update(retired_task_id_set)
            retired_source_task_ids = {
                source_completed_retry_repairs[task_id][0]
                for task_id in retired_task_ids
            }
            pending_retry_repair_sources.difference_update(
                retired_source_task_ids
            )
            blocked_tasks = [
                task_id
                for task_id in blocked_tasks
                if task_id not in retired_task_id_set
                and task_id not in retired_source_task_ids
            ]
            raw_retry_budget_findings = strategy.get(
                "retry_budget_findings"
            )
            if isinstance(raw_retry_budget_findings, list):
                strategy["retry_budget_findings"] = [
                    finding
                    for finding in raw_retry_budget_findings
                    if not (
                        isinstance(finding, Mapping)
                        and (
                            str(finding.get("source_task_id") or "")
                            in retired_source_task_ids
                            or str(finding.get("follow_up_task_id") or "")
                            in retired_task_id_set
                        )
                    )
                ]
            strategy[
                "last_source_completed_retry_repair_retired_task_ids"
            ] = retired_task_ids
            releases.extend(
                {
                    "source_task_id": source_completed_retry_repairs[
                        task_id
                    ][0],
                    "follow_up_task_id": task_id,
                    "guardrail_kind": "retry_budget",
                    "failure_kind": source_completed_retry_repairs[
                        task_id
                    ][1],
                    "reason": "source_completed_repair_retired",
                }
                for task_id in retired_task_ids
            )

    # A peer lane may have retired the board task after this lane loaded its
    # strategy.  Reconcile that stale local projection even when no Markdown
    # status mutation remains for this pass.
    completed_source_retry_repairs = {
        task_id: (source_task_id, failure_kind)
        for task_id, (
            source_task_id,
            failure_kind,
        ) in retry_budget_repair_sources_by_task_id.items()
        if task_id
        and statuses.get(task_id) == "completed"
        and statuses.get(source_task_id) == "completed"
    }
    if completed_source_retry_repairs:
        completed_retry_task_ids = set(completed_source_retry_repairs)
        completed_retry_source_ids = {
            source_task_id
            for source_task_id, _failure_kind
            in completed_source_retry_repairs.values()
        }
        pending_retry_repair_sources.difference_update(
            completed_retry_source_ids
        )
        stale_projection_task_ids: set[str] = set()
        retained_blocked_tasks: list[str] = []
        for blocked_task_id in blocked_tasks:
            if blocked_task_id in completed_retry_task_ids:
                stale_projection_task_ids.add(blocked_task_id)
                continue
            if blocked_task_id in completed_retry_source_ids:
                stale_projection_task_ids.update(
                    task_id
                    for task_id, (
                        source_task_id,
                        _failure_kind,
                    ) in completed_source_retry_repairs.items()
                    if source_task_id == blocked_task_id
                )
                continue
            retained_blocked_tasks.append(blocked_task_id)
        blocked_tasks = retained_blocked_tasks

        raw_retry_budget_findings = strategy.get(
            "retry_budget_findings"
        )
        if isinstance(raw_retry_budget_findings, list):
            retained_retry_budget_findings: list[Any] = []
            for finding in raw_retry_budget_findings:
                if not isinstance(finding, Mapping):
                    retained_retry_budget_findings.append(finding)
                    continue
                finding_source_task_id = str(
                    finding.get("source_task_id") or ""
                )
                finding_follow_up_task_id = str(
                    finding.get("follow_up_task_id") or ""
                )
                matching_task_ids = {
                    task_id
                    for task_id, (
                        source_task_id,
                        _failure_kind,
                    ) in completed_source_retry_repairs.items()
                    if task_id == finding_follow_up_task_id
                    or source_task_id == finding_source_task_id
                }
                if matching_task_ids:
                    stale_projection_task_ids.update(matching_task_ids)
                    continue
                retained_retry_budget_findings.append(finding)
            strategy["retry_budget_findings"] = (
                retained_retry_budget_findings
            )

        projection_repair_task_ids = sorted(
            stale_projection_task_ids - newly_retired_retry_task_ids
        )
        if projection_repair_task_ids:
            strategy[
                "last_repaired_source_completed_retry_projection_task_ids"
            ] = projection_repair_task_ids
            releases.extend(
                {
                    "source_task_id": (
                        completed_source_retry_repairs[task_id][0]
                    ),
                    "follow_up_task_id": task_id,
                    "guardrail_kind": "retry_budget",
                    "failure_kind": (
                        completed_source_retry_repairs[task_id][1]
                    ),
                    "reason": (
                        "source_completed_retry_repair_projection_repaired"
                    ),
                }
                for task_id in projection_repair_task_ids
            )

    current_reconciliation_records = reconciliation_guardrail_records(
        reconciliation_result=reconciliation_result,
        cleanup_result=cleanup_result,
    )
    active_reconciliation_keys = {
        str(record.get("dedupe_key") or "")
        for record in current_reconciliation_records
        if str(record.get("dedupe_key") or "").strip()
    }
    resolved_reconciliation_keys = (
        resolved_reconciliation_guardrail_keys(
            reconciliation_result=reconciliation_result,
            cleanup_result=cleanup_result,
            replay_result=replay_result,
        )
        - active_reconciliation_keys
    )
    resolved_reconciliation_cards = [
        card
        for card in reconciliation_guardrail_blocks(
            todo_text,
            task_prefix=task_prefix,
            include_completed=True,
        )
        if card["dedupe_key"] in resolved_reconciliation_keys
    ]
    if resolved_reconciliation_cards:
        terminal_statuses = {
            "complete",
            "completed",
            "done",
            "succeeded",
        }
        active_reconciliation_cards = [
            card
            for card in resolved_reconciliation_cards
            if card["status"] not in terminal_statuses
        ]
        retired_ids = [
            card["task_id"] for card in active_reconciliation_cards
        ]
        todo_text, retired_task_ids = mark_task_statuses_in_todo_text(
            todo_text,
            retired_ids,
            task_prefix=task_prefix,
            status="completed",
        )
        if retired_task_ids:
            todo_path.write_text(todo_text, encoding="utf-8")
            todo_changed = True
            statuses.update(
                {task_id: "completed" for task_id in retired_task_ids}
            )
        retired_set = set(retired_task_ids)
        already_completed_set = {
            card["task_id"]
            for card in resolved_reconciliation_cards
            if card["status"] in terminal_statuses
        }
        authoritative_resolved_ids = (
            retired_set | already_completed_set
        )
        resolved_cards_by_id = {
            card["task_id"]: card
            for card in resolved_reconciliation_cards
            if card["task_id"] in authoritative_resolved_ids
        }
        stale_projection_ids: set[str] = set()
        resolved_fingerprints = {
            card["fingerprint"]
            for card in resolved_cards_by_id.values()
            if card["fingerprint"]
        }
        raw_reconciliation_findings = strategy.get(
            "reconciliation_guardrail_findings"
        )
        if isinstance(raw_reconciliation_findings, list):
            retained_reconciliation_findings: list[Any] = []
            for finding in raw_reconciliation_findings:
                finding_task_id = (
                    str(finding.get("follow_up_task_id") or "")
                    if isinstance(finding, Mapping)
                    else ""
                )
                if finding_task_id in authoritative_resolved_ids:
                    stale_projection_ids.add(finding_task_id)
                    fingerprint = str(
                        finding.get("fingerprint") or ""
                    )
                    if fingerprint:
                        resolved_fingerprints.add(fingerprint)
                    continue
                retained_reconciliation_findings.append(finding)
            strategy["reconciliation_guardrail_findings"] = (
                retained_reconciliation_findings
            )
        seen_fingerprints = strategy.get(
            "reconciliation_guardrail_seen_fingerprints"
        )
        if isinstance(seen_fingerprints, list):
            present_seen = {
                str(fingerprint) for fingerprint in seen_fingerprints
            }
            for task_id, card in resolved_cards_by_id.items():
                if card["fingerprint"] in present_seen:
                    stale_projection_ids.add(task_id)
            strategy["reconciliation_guardrail_seen_fingerprints"] = [
                str(fingerprint)
                for fingerprint in seen_fingerprints
                if str(fingerprint) not in resolved_fingerprints
            ]
        stale_projection_ids.update(
            task_id
            for task_id in blocked_tasks
            if task_id in authoritative_resolved_ids
        )
        blocked_tasks = [
            task_id
            for task_id in blocked_tasks
            if task_id not in authoritative_resolved_ids
        ]
        if retired_task_ids:
            strategy[
                "last_resolved_reconciliation_guardrail_task_ids"
            ] = retired_task_ids
            releases.extend(
                {
                    "source_task_id": card["task_id"],
                    "follow_up_task_id": "",
                    "guardrail_kind": "reconciliation_guardrail",
                    "reason": "reconciliation_finding_resolved",
                    "dedupe_key": card["dedupe_key"],
                }
                for card in resolved_reconciliation_cards
                if card["task_id"] in retired_set
            )
        projection_repair_ids = sorted(
            stale_projection_ids - retired_set
        )
        if projection_repair_ids:
            strategy[
                "last_repaired_reconciliation_guardrail_projection_ids"
            ] = projection_repair_ids
            releases.extend(
                {
                    "source_task_id": task_id,
                    "follow_up_task_id": "",
                    "guardrail_kind": "reconciliation_guardrail",
                    "reason": "resolved_reconciliation_projection_repaired",
                    "dedupe_key": resolved_cards_by_id[task_id][
                        "dedupe_key"
                    ],
                }
                for task_id in projection_repair_ids
            )

    deduplicated_blocked_tasks = list(dict.fromkeys(blocked_tasks))
    if len(deduplicated_blocked_tasks) != len(blocked_tasks):
        duplicate_ids = sorted(
            {
                task_id
                for task_id in blocked_tasks
                if blocked_tasks.count(task_id) > 1
            }
        )
        releases.extend(
            {
                "source_task_id": task_id,
                "follow_up_task_id": "",
                "guardrail_kind": "stale_strategy_block",
                "reason": "duplicate_strategy_block",
            }
            for task_id in duplicate_ids
        )
        blocked_tasks = deduplicated_blocked_tasks

    active_dependency_records = dependency_guardrail_records(
        parse_task_file(todo_path, task_header_prefix(task_prefix))
    )
    active_dependency_fingerprints = {
        str(record.get("fingerprint") or "")
        for record in active_dependency_records
        if str(record.get("fingerprint") or "").strip()
    }
    active_dependency_sources = {
        str(record.get("source_task_id") or "")
        for record in active_dependency_records
        if str(record.get("source_task_id") or "").strip()
    }
    resolved_dependency_repair_tasks: dict[str, str] = {}
    for task in tasks:
        match = re.match(
            r"^Resolve dependency guardrail for (\S+)\s*$",
            str(getattr(task, "title", "") or "").strip(),
        )
        if match is None:
            continue
        source_task_id = match.group(1)
        if (
            str(getattr(task, "status", "") or "").lower()
            not in {"complete", "completed", "done", "succeeded"}
            and source_task_id not in active_dependency_sources
        ):
            resolved_dependency_repair_tasks[task.task_id] = source_task_id
    raw_dependency_findings = strategy.get("dependency_guardrail_findings")
    if isinstance(raw_dependency_findings, list):
        retained_dependency_findings: list[Any] = []
        pruned_dependency_findings = False
        for raw_record in raw_dependency_findings:
            if not isinstance(raw_record, Mapping):
                pruned_dependency_findings = True
                continue
            source_task_id = str(raw_record.get("source_task_id") or "")
            follow_up_task_id = str(raw_record.get("follow_up_task_id") or "")
            fingerprint = str(raw_record.get("fingerprint") or "")
            source_resolved = (
                bool(source_task_id)
                and source_task_id not in active_dependency_sources
            )
            fingerprint_resolved = (
                bool(fingerprint)
                and fingerprint not in active_dependency_fingerprints
            )
            if source_resolved or fingerprint_resolved:
                if source_task_id in active_dependency_sources:
                    retained_dependency_findings.append(raw_record)
                    continue
                pruned_dependency_findings = True
                if source_task_id:
                    if source_task_id in blocked_tasks:
                        blocked_tasks = [task_id for task_id in blocked_tasks if task_id != source_task_id]
                    releases.append(
                        {
                            "source_task_id": source_task_id,
                            "follow_up_task_id": follow_up_task_id,
                            "guardrail_kind": "dependency_guardrail",
                            "reason": "dependency_metadata_resolved",
                        }
                    )
                continue
            retained_dependency_findings.append(raw_record)
        if pruned_dependency_findings:
            strategy["dependency_guardrail_findings"] = retained_dependency_findings

    guardrail_groups = (
        ("retry_budget", strategy.get("retry_budget_findings")),
        ("dependency_guardrail", strategy.get("dependency_guardrail_findings")),
    )
    active_guardrail_sources: set[str] = set()
    active_retry_budget_sources: set[str] = set()
    for guardrail_kind, raw_records in guardrail_groups:
        if not isinstance(raw_records, list):
            continue
        for raw_record in raw_records:
            if not isinstance(raw_record, Mapping):
                continue
            source_task_id = str(raw_record.get("source_task_id") or "")
            follow_up_task_id = str(raw_record.get("follow_up_task_id") or "")
            if not source_task_id or not follow_up_task_id:
                continue
            active_guardrail_sources.add(source_task_id)
            if guardrail_kind == "retry_budget":
                active_retry_budget_sources.add(source_task_id)
            if source_task_id not in blocked_tasks:
                continue
            if statuses.get(follow_up_task_id) != "completed":
                continue
            blocked_tasks = [task_id for task_id in blocked_tasks if task_id != source_task_id]
            releases.append(
                {
                    "source_task_id": source_task_id,
                    "follow_up_task_id": follow_up_task_id,
                    "guardrail_kind": guardrail_kind,
                }
            )

    for source_task_id in list(blocked_tasks):
        repair = completed_retry_repairs.get(source_task_id)
        if not repair:
            continue
        follow_up_task_id = str(repair.get("follow_up_task_id") or "")
        if not follow_up_task_id:
            continue
        blocked_tasks = [task_id for task_id in blocked_tasks if task_id != source_task_id]
        releases.append(
            {
                "source_task_id": source_task_id,
                "follow_up_task_id": follow_up_task_id,
                "guardrail_kind": "retry_budget",
                "failure_kind": str(repair.get("failure_kind") or ""),
                "reason": "historical_retry_repair_completed",
            }
        )

    for source_task_id in list(blocked_tasks):
        original_source_task_id, failure_kind = retry_budget_repair_sources_by_task_id.get(
            source_task_id,
            ("", ""),
        )
        if not original_source_task_id:
            continue
        if source_task_id not in pending_retry_repair_sources:
            continue
        blocked_tasks = [task_id for task_id in blocked_tasks if task_id != source_task_id]
        releases.append(
            {
                "source_task_id": source_task_id,
                "follow_up_task_id": "",
                "guardrail_kind": "stale_strategy_block",
                "failure_kind": failure_kind,
                "reason": "recursive_retry_repair_block",
                "original_source_task_id": original_source_task_id,
            }
        )

    for source_task_id in list(blocked_tasks):
        status = statuses.get(source_task_id)
        if status is None or status == "completed":
            continue
        if source_task_id not in retry_budget_repair_task_ids:
            continue
        if source_task_id in active_guardrail_sources or source_task_id in active_dependency_sources:
            continue
        if source_task_id in pending_retry_repair_sources:
            continue
        blocked_tasks = [task_id for task_id in blocked_tasks if task_id != source_task_id]
        releases.append(
            {
                "source_task_id": source_task_id,
                "follow_up_task_id": "",
                "guardrail_kind": "stale_strategy_block",
                "reason": "no_guardrail_repair_path",
            }
        )

    for source_task_id in list(blocked_tasks):
        status = statuses.get(source_task_id)
        if status is None:
            blocked_tasks = [task_id for task_id in blocked_tasks if task_id != source_task_id]
            releases.append(
                {
                    "source_task_id": source_task_id,
                    "follow_up_task_id": "",
                    "guardrail_kind": "stale_strategy_block",
                    "reason": "missing_task",
                }
            )
            continue
        if status == "completed":
            blocked_tasks = [task_id for task_id in blocked_tasks if task_id != source_task_id]
            releases.append(
                {
                    "source_task_id": source_task_id,
                    "follow_up_task_id": "",
                    "guardrail_kind": "stale_strategy_block",
                    "reason": "source_completed",
                }
            )

    recursive_retry_repair_task_ids: list[str] = []
    for task_id, (source_task_id, failure_kind) in retry_budget_repair_sources_by_task_id.items():
        if not task_id or not source_task_id:
            continue
        if task_id not in statuses:
            continue
        if statuses.get(task_id) == "completed":
            continue
        if source_task_id not in retry_budget_repair_task_ids:
            continue
        original_source_task_id, _original_failure_kind = retry_budget_repair_sources_by_task_id.get(
            source_task_id,
            ("", ""),
        )
        recursive_retry_repair_task_ids.append(task_id)
        releases.append(
            {
                "source_task_id": task_id,
                "follow_up_task_id": "",
                "guardrail_kind": "retry_budget",
                "failure_kind": failure_kind,
                "reason": "recursive_retry_repair_task_retired",
                "parent_repair_task_id": source_task_id,
                "original_source_task_id": original_source_task_id,
            }
        )

    if recursive_retry_repair_task_ids:
        todo_text, retired_task_ids = mark_task_statuses_in_todo_text(
            todo_text,
            recursive_retry_repair_task_ids,
            task_prefix=task_prefix,
            status="completed",
        )
        if retired_task_ids:
            todo_path.write_text(todo_text, encoding="utf-8")
            todo_changed = True
            statuses.update({task_id: "completed" for task_id in retired_task_ids})
            strategy["last_recursive_retry_repair_retired_task_ids"] = retired_task_ids

    if resolved_dependency_repair_tasks:
        todo_text, retired_task_ids = mark_task_statuses_in_todo_text(
            todo_text,
            list(resolved_dependency_repair_tasks),
            task_prefix=task_prefix,
            status="completed",
        )
        if retired_task_ids:
            todo_path.write_text(todo_text, encoding="utf-8")
            todo_changed = True
            statuses.update({task_id: "completed" for task_id in retired_task_ids})
            strategy["last_resolved_dependency_guardrail_task_ids"] = (
                retired_task_ids
            )
            strategy["dependency_guardrail_seen_fingerprints"] = sorted(
                active_dependency_fingerprints
            )
            janitor_owned_sources = {
                str(receipt.get("task_id") or "")
                for receipt in strategy.get("objective_task_janitor_receipts", [])
                if isinstance(receipt, Mapping)
                and str(receipt.get("action") or "") == "block"
                and str(receipt.get("task_id") or "")
            }
            quarantined_sources = {
                str(record.get("task_id") or "")
                for record in strategy.get("autonomous_unstall_quarantines", [])
                if isinstance(record, Mapping)
                and str(record.get("task_id") or "")
            }
            retired_dependency_sources = {
                resolved_dependency_repair_tasks[task_id]
                for task_id in retired_task_ids
                if task_id in resolved_dependency_repair_tasks
            }
            releasable_dependency_sources = (
                retired_dependency_sources
                - active_retry_budget_sources
                - pending_retry_repair_sources
                - janitor_owned_sources
                - quarantined_sources
            )
            if releasable_dependency_sources:
                blocked_tasks = [
                    task_id
                    for task_id in blocked_tasks
                    if task_id not in releasable_dependency_sources
                ]
            releases.extend(
                {
                    "source_task_id": source_task_id,
                    "follow_up_task_id": task_id,
                    "guardrail_kind": "dependency_guardrail",
                    "reason": "resolved_repair_task_retired",
                }
                for task_id, source_task_id in resolved_dependency_repair_tasks.items()
                if task_id in retired_task_ids
            )

    if not releases:
        return []
    strategy["blocked_tasks"] = blocked_tasks
    strategy["last_guardrail_unblock_at"] = utc_now()
    strategy["guardrail_unblock_releases"] = releases
    write_json(strategy_path, strategy)
    if commit_outputs and todo_changed:
        commit_results = commit_generated_outputs(
            [todo_path],
            repo_root=repo_root or todo_path.parent,
            subject=commit_subject,
        )
        if commit_results:
            strategy["last_guardrail_unblock_commit_results"] = commit_results
            write_json(strategy_path, strategy)
    return releases


def codebase_scan_fingerprint_hints(
    todo_text: str,
    *,
    discovery_dir: Path,
) -> set[str]:
    """Collect durable full or prefix fingerprints shared by all supervisors."""

    hints: set[str] = set()
    for _start, _end, block in task_blocks_with_spans(todo_text):
        if not re.search(r"^- Candidate kind:\s*codebase_scan\s*$", block, flags=re.MULTILINE):
            continue
        match = re.search(
            r"^- Todo vector key:\s*([0-9a-f]{8,40})\s*$",
            block,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if match:
            hints.add(match.group(1).lower())
    try:
        artifact_names = (path.name for path in discovery_dir.glob("*-codebase-scan-*.md"))
        for name in artifact_names:
            match = re.search(r"-codebase-scan-([0-9a-f]{8,40})\.md$", name, flags=re.IGNORECASE)
            if match:
                hints.add(match.group(1).lower())
    except OSError:
        pass
    return hints


def retire_duplicate_codebase_scan_tasks(
    todo_path: Path,
    *,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
) -> dict[str, str]:
    """Complete later codebase-scan aliases that share one vector identity."""

    retired: dict[str, str] = {}
    with locked_taskboard(todo_path) as taskboard:
        todo_text = taskboard.read()
        by_vector_key: dict[str, list[tuple[str, str]]] = {}
        prefix = task_id_prefix(task_prefix)
        for _start, _end, block in task_blocks_with_spans(todo_text):
            heading = re.match(rf"^##\s+({re.escape(prefix)}\S*)", block)
            if heading is None or not re.search(
                r"^- Candidate kind:\s*codebase_scan\s*$",
                block,
                flags=re.MULTILINE,
            ):
                continue
            vector_key = re.search(
                r"^- Todo vector key:\s*(\S+)\s*$",
                block,
                flags=re.MULTILINE,
            )
            status = re.search(r"^- Status:\s*(\S+)\s*$", block, flags=re.MULTILINE)
            if vector_key is None:
                continue
            by_vector_key.setdefault(vector_key.group(1).lower(), []).append(
                (heading.group(1), status.group(1).lower() if status else "todo")
            )

        for records in by_vector_key.values():
            if len(records) < 2:
                continue
            keeper = records[0][0]
            for task_id, status in records[1:]:
                if status not in SUCCESSFUL_MERGE_RECEIPT_STATUSES:
                    retired[task_id] = keeper
        if not retired:
            return {}

        updated, updated_task_ids = mark_task_statuses_in_todo_text(
            todo_text,
            list(retired),
            task_prefix=task_prefix,
            status="completed",
        )
        if not updated_task_ids:
            return {}
        for task_id in updated_task_ids:
            block_pattern = re.compile(
                rf"(^##\s+{re.escape(task_id)}(?:\s|$).*?)(?=^##\s+\S+|\Z)",
                flags=re.MULTILINE | re.DOTALL,
            )

            def mark_completion(match: re.Match[str], *, keeper: str = retired[task_id]) -> str:
                return re.sub(
                    r"^- Completion:.*$",
                    f"- Completion: deduplicated:{keeper}",
                    match.group(1),
                    count=1,
                    flags=re.MULTILINE,
                )

            updated = block_pattern.sub(mark_completion, updated, count=1)
        replace_locked_taskboard(taskboard, updated)
    return {task_id: retired[task_id] for task_id in updated_task_ids}


def persist_codebase_scan_inventory(
    inventory: CodebaseScanInventory,
    *,
    admission: CodebaseRefillAdmission | None = None,
    repo_root: Path,
    discovery_dir: Path,
    dataset_dir: Path | None,
    started_at: datetime,
    appended_tasks: int,
    late_deduplicated_candidates: int,
) -> dict[str, Any]:
    """Persist unbounded per-path scan details and return its artifact record."""

    from ..task_sources.dataset_store import ObjectiveDatasetStore

    scan_key = sha1(
        f"{repo_root.resolve()}\0{started_at.isoformat()}\0{time.time_ns()}".encode("utf-8")
    ).hexdigest()[:16]
    scan_id = f"codebase-scan-{started_at.strftime('%Y%m%dT%H%M%S')}-{scan_key}"
    detail_rows = [
        {"detail_kind": "excluded_file", **record}
        for record in inventory.excluded_files
    ] + [
        {"detail_kind": "parser_failure", **record}
        for record in inventory.parser_failures
    ] + [
        {"detail_kind": "admission_rejection", **record}
        for record in (admission.rejections if admission is not None else ())
    ]
    final_deduplicated = inventory.deduplicated_candidate_count + late_deduplicated_candidates
    admission_rejected = (
        admission.rejected_candidate_count if admission is not None else 0
    )
    candidate_accounting = {
        "raw_candidates": inventory.raw_candidate_count,
        "seen_candidates": inventory.seen_candidate_count,
        "deduplicated_candidates": final_deduplicated,
        "rejected_candidates": (
            inventory.rejected_candidate_count + admission_rejected
        ),
        "appended_tasks": appended_tasks,
    }
    reason_summaries = {
        **inventory.reason_summaries(),
        "admission_rejections": (
            admission.reason_summaries() if admission is not None else []
        ),
    }
    artifact = ObjectiveDatasetStore(dataset_dir or discovery_dir).persist_scan_details(
        scan_id=scan_id,
        details=detail_rows,
        metadata={
            "analyzer_version": CODEBASE_SCAN_ANALYZER_VERSION,
            "repo_root": str(repo_root.resolve()),
            "started_at": started_at.isoformat(),
            "coverage": inventory.coverage_dict(),
            "git_roots": list(inventory.git_roots),
            "expected_git_roots": list(inventory.expected_git_roots),
            "expected_git_root_count": len(inventory.expected_git_roots),
            "coverage_complete": inventory.complete,
            "reason_summaries": reason_summaries,
            "candidate_accounting": candidate_accounting,
            "admission": (
                admission.details_dict() if admission is not None else {}
            ),
        },
    )
    return artifact.to_dict()


def codebase_scan_accounting_metadata(
    inventory: CodebaseScanInventory,
    *,
    admission: CodebaseRefillAdmission | None = None,
    appended_tasks: int,
    late_deduplicated_candidates: int = 0,
    details_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the stable JSON projection used by receipts and artifacts."""

    deduplicated = inventory.deduplicated_candidate_count + late_deduplicated_candidates
    admission_rejected = (
        admission.rejected_candidate_count if admission is not None else 0
    )
    candidates = {
        "raw_candidates": inventory.raw_candidate_count,
        "seen_candidates": inventory.seen_candidate_count,
        "deduplicated_candidates": deduplicated,
        "rejected_candidates": (
            inventory.rejected_candidate_count + admission_rejected
        ),
        "appended_tasks": appended_tasks,
    }
    accounted = sum(
        candidates[key]
        for key in (
            "seen_candidates",
            "deduplicated_candidates",
            "rejected_candidates",
            "appended_tasks",
        )
    )
    if accounted != inventory.raw_candidate_count:
        raise ValueError(
            "unbalanced codebase candidate accounting: "
            f"raw={inventory.raw_candidate_count}, accounted={accounted}"
        )
    coverage = inventory.coverage_dict()
    reason_summaries = {
        **inventory.reason_summaries(),
        "admission_rejections": (
            admission.reason_summaries() if admission is not None else []
        ),
    }
    return {
        "coverage": coverage,
        "candidate_accounting": candidates,
        "reason_summaries": reason_summaries,
        "details_artifact": dict(details_artifact or {}),
        "admission": admission.details_dict() if admission is not None else {},
        "coverage_complete": inventory.complete,
        "expected_git_root_count": len(inventory.expected_git_roots),
        "expected_git_roots": list(inventory.expected_git_roots),
        # Flat aliases make operational JSON queries and mixed-version stores
        # straightforward while typed consumers use receipt.accounting.
        **coverage,
        **candidates,
    }


def safe_codebase_scan_accounting_metadata(
    inventory: CodebaseScanInventory,
    *,
    admission: CodebaseRefillAdmission | None = None,
    appended_tasks: int,
    late_deduplicated_candidates: int = 0,
    details_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize valid accounting or preserve invalid evidence without raising.

    ``ScanAccounting`` correctly rejects impossible funnels.  A damaged
    analyzer must still return a typed failed/partial receipt, so invalid raw
    counters are namespaced instead of being passed off as valid accounting.
    """

    try:
        return codebase_scan_accounting_metadata(
            inventory,
            admission=admission,
            appended_tasks=appended_tasks,
            late_deduplicated_candidates=late_deduplicated_candidates,
            details_artifact=details_artifact,
        )
    except (TypeError, ValueError) as exc:
        return {
            "invalid_scan_accounting": inventory.health_inventory_dict(
                appended_tasks=appended_tasks,
                late_deduplicated_candidates=late_deduplicated_candidates,
                additional_rejected_candidates=(
                    admission.rejected_candidate_count
                    if admission is not None
                    else 0
                ),
            ),
            "invalid_scan_accounting_error": f"{type(exc).__name__}: {exc}",
            "scan_details_artifact": dict(details_artifact or {}),
        }


def empty_codebase_scan_accounting_metadata() -> dict[str, Any]:
    """Accounting dimensions for attempts which stop before inventory."""

    return codebase_scan_accounting_metadata(
        CodebaseScanInventory(),
        appended_tasks=0,
    )


def typed_codebase_scan_accounting(metadata: Mapping[str, Any]) -> ScanAccounting:
    """Validate the JSON projection against the public receipt contract."""

    return ScanAccounting.from_dict(metadata)


def classify_codebase_scan_health(
    inventory: CodebaseScanInventory,
    *,
    admission: CodebaseRefillAdmission | None = None,
    appended_tasks: int,
    late_deduplicated_candidates: int = 0,
    canaries: AnalyzerCanaryReport | Mapping[str, Any] | None = None,
    thresholds: AnalyzerHealthThresholds | Mapping[str, Any] | None = None,
) -> AnalyzerHealthReport:
    """Classify a finalized codebase inventory and candidate disposition."""

    return classify_analyzer_health(
        inventory.health_inventory_dict(
            appended_tasks=appended_tasks,
            late_deduplicated_candidates=late_deduplicated_candidates,
            additional_rejected_candidates=(
                admission.rejected_candidate_count
                if admission is not None
                else 0
            ),
        ),
        canaries=canaries,
        thresholds=thresholds,
    )


def analyzer_health_metadata(
    *,
    thresholds: AnalyzerHealthThresholds,
    canaries: AnalyzerCanaryReport | None = None,
    health: AnalyzerHealthReport | None = None,
) -> dict[str, Any]:
    """Return stable policy/evidence fields embedded in every scan receipt."""

    metadata: dict[str, Any] = {"health_thresholds": thresholds.to_dict()}
    if canaries is not None:
        metadata["analyzer_canaries"] = canaries.to_dict()
    if health is not None:
        metadata["analyzer_health"] = health.to_dict()
    return metadata


def fail_closed_scan_outcome(
    nominal_reason: ScanTerminalReason,
    health: AnalyzerHealthReport,
    *,
    generated_count: int,
    exhaustive: bool,
) -> tuple[ScanTerminalReason, bool, str | None]:
    """Apply analyzer health without ever promoting failed scan evidence."""

    if nominal_reason in {ScanTerminalReason.FAILED, ScanTerminalReason.TIMED_OUT}:
        return nominal_reason, False, None
    if health.status is AnalyzerHealthStatus.UNHEALTHY:
        if generated_count:
            return ScanTerminalReason.PARTIAL, False, None
        reasons = ", ".join(health.reasons) or "unknown analyzer health failure"
        return ScanTerminalReason.FAILED, False, f"unhealthy analyzer scan: {reasons}"
    if health.status is AnalyzerHealthStatus.PARTIAL:
        return ScanTerminalReason.PARTIAL, False, None
    safe = nominal_reason is ScanTerminalReason.EXHAUSTED and exhaustive
    return nominal_reason, safe, None


def record_codebase_scan_findings(
    *,
    todo_path: Path,
    state_path: Path | None,
    strategy_path: Path,
    discovery_dir: Path,
    repo_root: Path,
    bundle_dir: Path | None = None,
    dataset_dir: Path | None = None,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
    depends_on: Sequence[str] = (),
    min_open_tasks: int = DEFAULT_CODEBASE_SCAN_MIN_OPEN_TASKS,
    max_findings: int = DEFAULT_CODEBASE_SCAN_MAX_FINDINGS,
    cooldown_seconds: int = DEFAULT_CODEBASE_SCAN_COOLDOWN_SECONDS,
    force: bool = False,
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    skip_prefixes: Sequence[str] = CODEBASE_SCAN_SKIP_PREFIXES,
    include_prefixes: Sequence[str] = (),
    allowed_tracks: Sequence[str] = (),
    objective_path: Path | None = None,
    mission_terms: Sequence[str] = (),
    allow_unscoped_codebase_refill: bool = False,
    health_thresholds: AnalyzerHealthThresholds | Mapping[str, Any] | None = None,
    exhaustion_quorum_size: int = DEFAULT_EXHAUSTION_QUORUM_SIZE,
    objective_revision: str = "",
    exhaustion_receipts: Iterable[RefillScanResult[Any] | Mapping[str, Any]] = (),
    commit_outputs: bool = False,
    commit_subject: str = "Agent: record codebase scan backlog findings",
) -> RefillScanResult[dict[str, Any]]:
    """Feed a low backlog and return a typed account of the scan attempt."""

    # Normalize the legacy Markdown-style option once at this public boundary.
    # Downstream ID allocation and rendering only receive the canonical prefix.
    task_prefix = task_id_prefix(task_prefix)
    started_at = datetime.now(timezone.utc)
    initial_identity = scan_identity(repo_root)
    health_policy = AnalyzerHealthThresholds.from_value(health_thresholds)
    policy_metadata = analyzer_health_metadata(thresholds=health_policy)
    if not todo_path.exists():
        return build_scan_result(
            ScanTerminalReason.FAILED,
            "preflight",
            CODEBASE_SCAN_ANALYZER_VERSION,
            repo_root,
            started_at,
            error=f"todo board does not exist: {todo_path}",
            metadata={
                **policy_metadata,
                **empty_codebase_scan_accounting_metadata(),
                "missing_input": "todo_path",
            },
        )
    retired_duplicates = retire_duplicate_codebase_scan_tasks(
        todo_path,
        task_prefix=task_prefix,
    )
    todo_text = todo_path.read_text(encoding="utf-8")
    strategy = load_strategy(strategy_path)
    objective_source = (
        objective_path.read_text(encoding="utf-8")
        if objective_path is not None and objective_path.is_file()
        else ""
    )
    objective_id = (
        objective_revision
        or canonical_objective_revision(objective_source)
    )
    gate_strategy: Mapping[str, Any] = strategy
    if (
        objective_path is not None
        and str(strategy.get("last_codebase_scan_objective_revision") or "")
        != objective_id
    ):
        gate_strategy = {
            **strategy,
            "last_codebase_scan_at": "",
            "last_drained_codebase_scan_task_count": -1,
        }
    strategy_seen = {
        str(item)
        for item in strategy.get("codebase_scan_seen_fingerprints", [])
        if str(item).strip()
    }
    shared_seen = codebase_scan_fingerprint_hints(
        todo_text,
        discovery_dir=discovery_dir,
    )
    seen = strategy_seen | shared_seen
    strategy["codebase_scan_seen_fingerprints"] = sorted(seen)
    if retired_duplicates:
        strategy["codebase_scan_duplicate_tasks_retired"] = retired_duplicates
        strategy["last_codebase_scan_duplicate_retirement_at"] = utc_now()
    if max_findings <= 0:
        if retired_duplicates or seen != strategy_seen:
            write_json(strategy_path, strategy)
        return build_scan_result(
            ScanTerminalReason.DISABLED,
            "disabled",
            CODEBASE_SCAN_ANALYZER_VERSION,
            repo_root,
            started_at,
            metadata={
                **policy_metadata,
                **empty_codebase_scan_accounting_metadata(),
                "cause": "non_positive_max_findings",
            },
        )
    should_scan, mode, current_open, task_count = should_refill_backlog(
        todo_text=todo_text,
        state_path=state_path,
        strategy=gate_strategy,
        last_scan_key="last_codebase_scan_at",
        last_drained_scan_task_count_key="last_drained_codebase_scan_task_count",
        task_prefix=task_prefix,
        min_open_tasks=min_open_tasks,
        cooldown_seconds=cooldown_seconds,
        force=force,
    )
    if not should_scan:
        if retired_duplicates or seen != strategy_seen:
            write_json(strategy_path, strategy)
        reason = (
            ScanTerminalReason.COOLDOWN
            if mode == "cooldown"
            else ScanTerminalReason.THRESHOLD_SATISFIED
        )
        return build_scan_result(
            reason,
            mode,
            CODEBASE_SCAN_ANALYZER_VERSION,
            repo_root,
            started_at,
            metadata={
                **policy_metadata,
                **empty_codebase_scan_accounting_metadata(),
                "open_task_count": current_open,
                "task_count": task_count,
            },
        )
    capacity_open_count = (
        0 if mode.startswith("runnable_drained") else current_open
    )
    refill_capacity = refill_open_task_capacity(
        current_open=capacity_open_count,
        min_open_tasks=min_open_tasks,
        max_findings=max_findings,
    )
    if refill_capacity <= 0:
        return build_scan_result(
            ScanTerminalReason.THRESHOLD_SATISFIED,
            "open_task_pressure_bound",
            CODEBASE_SCAN_ANALYZER_VERSION,
            repo_root,
            started_at,
            metadata={
                **policy_metadata,
                **empty_codebase_scan_accounting_metadata(),
                "open_task_count": current_open,
                "task_count": task_count,
                "refill_capacity": 0,
                "open_task_target": max(0, int(min_open_tasks))
                + max(0, DEFAULT_REFILL_OPEN_TASK_HEADROOM),
            },
        )

    canaries = run_codebase_analyzer_canaries()
    scan_metadata = analyzer_health_metadata(
        thresholds=health_policy,
        canaries=canaries,
    )
    try:
        objective_goals = objective_goals_for_codebase_refill(objective_path)
        inventory = scan_codebase_findings(
            repo_root,
            max_findings=None,
            seen_fingerprints=seen,
            exhaustive=True,
            skip_prefixes=skip_prefixes,
            include_prefixes=include_prefixes,
            allowed_tracks=allowed_tracks,
            return_inventory=True,
        )
        if not isinstance(inventory, CodebaseScanInventory):  # pragma: no cover - defensive
            raise TypeError("instrumented codebase scan did not return inventory")
        admission = admit_codebase_refill_candidates(
            inventory,
            objective_goals=objective_goals,
            mission_terms=mission_terms,
            max_findings=max_findings,
            allow_unscoped=allow_unscoped_codebase_refill,
            objective_scope_configured=objective_path is not None,
        )
        findings = list(admission.findings)
    except TimeoutError as exc:
        return build_scan_result(
            ScanTerminalReason.TIMED_OUT,
            mode,
            CODEBASE_SCAN_ANALYZER_VERSION,
            repo_root,
            started_at,
            error=str(exc) or type(exc).__name__,
            metadata={**scan_metadata, **empty_codebase_scan_accounting_metadata()},
        )
    except Exception as exc:
        logger.exception("Codebase refill scan failed")
        return build_scan_result(
            ScanTerminalReason.FAILED,
            mode,
            CODEBASE_SCAN_ANALYZER_VERSION,
            repo_root,
            started_at,
            error=f"{type(exc).__name__}: {exc}",
            metadata={**scan_metadata, **empty_codebase_scan_accounting_metadata()},
        )
    if not admission.policy_valid:
        details_artifact = persist_codebase_scan_inventory(
            inventory,
            admission=admission,
            repo_root=repo_root,
            discovery_dir=discovery_dir,
            dataset_dir=dataset_dir,
            started_at=started_at,
            appended_tasks=0,
            late_deduplicated_candidates=0,
        )
        source_identity = RepositoryTreeIdentity(
            initial_identity.repository_id,
            codebase_source_tree_identity(repo_root, inventory),
        )
        policy_error = "; ".join(
            str(item.get("message") or item.get("reason_code") or "")
            for item in admission.policy_errors
        )
        return build_scan_result(
            ScanTerminalReason.FAILED,
            mode,
            CODEBASE_SCAN_ANALYZER_VERSION,
            repo_root,
            started_at,
            safe_for_completion_reasoning=False,
            error=f"codebase refill admission policy failed: {policy_error}",
            metadata={
                **scan_metadata,
                **safe_codebase_scan_accounting_metadata(
                    inventory,
                    admission=admission,
                    appended_tasks=0,
                    details_artifact=details_artifact,
                ),
                "objective_revision": objective_id,
                "admission_policy_errors": [
                    dict(item) for item in admission.policy_errors
                ],
            },
            identity=source_identity,
        )
    strategy["last_codebase_scan_at"] = utc_now()
    strategy["last_codebase_scan_mode"] = mode
    strategy["last_codebase_scan_objective_revision"] = objective_id
    strategy["last_codebase_scan_scope"] = {
        "include_prefixes": sorted(
            str(prefix).strip().strip("/")
            for prefix in include_prefixes
            if str(prefix).strip().strip("/")
        ),
        "allowed_tracks": sorted(
            str(track).strip().lower()
            for track in allowed_tracks
            if str(track).strip()
        ),
        "allow_unscoped_codebase_refill": bool(allow_unscoped_codebase_refill),
        "objective_path": str(objective_path or ""),
        "objective_goal_ids": [
            goal.goal_id for goal in objective_goals if goal.is_schedulable
        ],
    }
    strategy["codebase_scan_seen_fingerprints"] = sorted(seen | {finding.fingerprint for finding in findings})
    if not findings:
        details_artifact = persist_codebase_scan_inventory(
            inventory,
            admission=admission,
            repo_root=repo_root,
            discovery_dir=discovery_dir,
            dataset_dir=dataset_dir,
            started_at=started_at,
            appended_tasks=0,
            late_deduplicated_candidates=0,
        )
        nominal_reason = (
            ScanTerminalReason.DUPLICATE_ONLY
            if inventory.seen_candidate_count or inventory.deduplicated_candidate_count
            else ScanTerminalReason.EXHAUSTED
        )
        health = classify_codebase_scan_health(
            inventory,
            admission=admission,
            appended_tasks=0,
            canaries=canaries,
            thresholds=health_policy,
        )
        terminal_reason, health_completion_safe, health_error = fail_closed_scan_outcome(
            nominal_reason,
            health,
            generated_count=0,
            exhaustive=mode.endswith("exhaustive"),
        )
        source_identity = RepositoryTreeIdentity(
            initial_identity.repository_id,
            codebase_source_tree_identity(repo_root, inventory),
        )
        configuration_id = scan_configuration_revision(
            codebase_exhaustion_configuration(
                skip_prefixes=skip_prefixes,
                include_prefixes=include_prefixes,
                allowed_tracks=allowed_tracks,
                health_thresholds=health_policy,
            )
        )
        binding = ExhaustionBinding(
            repository_id=source_identity.repository_id,
            tree_id=source_identity.tree_id,
            analyzer_version=CODEBASE_SCAN_ANALYZER_VERSION,
            configuration_revision=configuration_id,
            objective_revision=objective_id,
        )
        receipt_metadata = {
            **analyzer_health_metadata(
                thresholds=health_policy,
                canaries=canaries,
                health=health,
            ),
            **safe_codebase_scan_accounting_metadata(
                inventory,
                admission=admission,
                appended_tasks=0,
                details_artifact=details_artifact,
            ),
            "duplicate_candidate_count": (
                inventory.seen_candidate_count
                + inventory.deduplicated_candidate_count
            ),
            "open_task_count": current_open,
            "task_count": task_count,
            "exhaustive": mode.endswith("exhaustive"),
            "evidence_channel": "codebase:normal",
            "configuration_revision": configuration_id,
            "objective_revision": objective_id,
            "exhaustion_binding": binding.to_dict(),
        }
        candidate_receipt = build_scan_result(
            terminal_reason,
            mode,
            CODEBASE_SCAN_ANALYZER_VERSION,
            repo_root,
            started_at,
            safe_for_completion_reasoning=False,
            error=health_error,
            metadata=receipt_metadata,
            identity=source_identity,
        )
        from ..task_sources.dataset_store import ObjectiveDatasetStore

        quorum_store = ObjectiveDatasetStore(dataset_dir or discovery_dir)
        stored_quorum = quorum_store.load_exhaustion_quorum(binding.repository_id)
        stored_members = (
            stored_quorum.get("members", ())
            if isinstance(stored_quorum, Mapping)
            else ()
        )
        quorum = evaluate_exhaustion_quorum(
            [*stored_members, *exhaustion_receipts, candidate_receipt],
            binding=binding,
            required_members=exhaustion_quorum_size,
        )
        completion_safe = health_completion_safe and quorum.satisfied
        receipt_metadata["exhaustion_quorum"] = quorum.to_dict()
        receipt = build_scan_result(
            terminal_reason,
            mode,
            CODEBASE_SCAN_ANALYZER_VERSION,
            repo_root,
            started_at,
            finished_at=candidate_receipt.finished_at,
            safe_for_completion_reasoning=completion_safe,
            error=health_error,
            metadata=receipt_metadata,
            identity=source_identity,
        )
        quorum = evaluate_exhaustion_quorum(
            [*stored_members, *exhaustion_receipts, receipt],
            binding=binding,
            required_members=exhaustion_quorum_size,
        )
        receipt_metadata["exhaustion_quorum"] = quorum.to_dict()
        # Rebuild once so the canonical receipt carries the final projection.
        receipt = build_scan_result(
            terminal_reason,
            mode,
            CODEBASE_SCAN_ANALYZER_VERSION,
            repo_root,
            started_at,
            finished_at=candidate_receipt.finished_at,
            safe_for_completion_reasoning=completion_safe,
            error=health_error,
            metadata=receipt_metadata,
            identity=source_identity,
        )
        quorum_store.persist_exhaustion_quorum(quorum)
        strategy["last_codebase_scan_findings"] = []
        strategy["last_codebase_scan_health"] = health.to_dict()
        strategy["last_codebase_scan_exhaustion_quorum"] = quorum.to_dict()
        # This task-count marker is scheduler state, not proof authority.  A
        # healthy exhaustive pass must observe the configured cooldown even
        # while it awaits a second independent exhaustion-quorum channel.
        if health_completion_safe:
            strategy["last_drained_codebase_scan_task_count"] = task_count
        write_json(strategy_path, strategy)
        return receipt

    appended: list[dict[str, Any]] = []
    bundle_records: list[dict[str, Any]] = []
    generated_paths: list[Path] = []
    if bundle_dir is not None:
        from ..task_sources.todo_vector_index import collect_output_symbols

    detected_count = len(findings)
    with locked_taskboard(todo_path) as taskboard:
        todo_text = taskboard.read()
        board_namespace = taskboard_namespace_from_todo(todo_text, todo_path)
        latest_seen = codebase_scan_fingerprint_hints(
            todo_text,
            discovery_dir=discovery_dir,
        )
        findings = [
            finding
            for finding in findings
            if not any(
                finding.fingerprint == item or finding.fingerprint.startswith(item)
                for item in latest_seen
            )
        ]
        reserved_task_ids = task_ids_from_artifact_names(
            discovery_dir,
            task_prefix=task_prefix,
        )
        for finding in findings:
            identity = codebase_finding_task_identity(finding)
            follow_up_task_id = next_task_id(
                todo_text,
                task_prefix=task_prefix,
                reserved_task_ids=reserved_task_ids,
            )
            reserved_task_ids.add(follow_up_task_id)
            discovery_path = write_codebase_scan_discovery(
                discovery_dir=discovery_dir,
                task_id=follow_up_task_id,
                finding=finding,
            )
            generated_paths.append(discovery_path)
            bundle_key = codebase_scan_bundle_key(finding) if bundle_dir is not None else ""
            shard_path = bundle_path(bundle_dir, bundle_key) if bundle_dir is not None else None
            bundle_shard = repo_relative_path(repo_root, shard_path) if shard_path is not None else ""
            ast_symbols = (
                collect_output_symbols(repo_root, [finding.root_relative_path])[:80]
                if bundle_dir is not None
                else []
            )
            task_block = codebase_scan_task_block(
                task_id=follow_up_task_id,
                finding=finding,
                discovery_path=discovery_path,
                depends_on=depends_on,
                discovery_output_path=discovery_output_path,
                bundle_key=bundle_key,
                bundle_shard=bundle_shard,
                ast_symbols=ast_symbols,
                board_namespace=board_namespace,
            )
            todo_text = todo_text.rstrip() + "\n\n" + task_block.strip() + "\n"
            finding_record = {
                "follow_up_task_id": follow_up_task_id,
                "fingerprint": finding.fingerprint,
                "kind": finding.kind,
                "source": f"{finding.root_relative_path}:{finding.line_number}",
                "discovery_path": str(discovery_path),
                "canonical_task_key": identity.canonical_task_key,
                "canonical_task_cid": identity.canonical_task_cid,
                "semantic_identity": identity.semantic_fingerprint,
                "board_namespace": board_namespace,
                "objective_goal_ids": list(finding.objective_goal_ids),
            }
            if bundle_key:
                finding_record.update({"bundle_key": bundle_key, "bundle_shard": bundle_shard})
                lineage = list(finding.objective_goal_ids)
                goal_id = lineage[0] if lineage else ""
                parent_goal_ids = lineage[1:]
                bundle_records.append(
                    {
                        "task_id": follow_up_task_id,
                        "bundle_key": bundle_key,
                        "task_block": task_block,
                        "task_payload": {
                            "task_id": follow_up_task_id,
                            "board_namespace": board_namespace,
                            "canonical_task_key": identity.canonical_task_key,
                            "canonical_task_cid": identity.canonical_task_cid,
                            "semantic_identity": identity.semantic_fingerprint,
                            "status": "todo",
                            "title": finding.summary,
                            "priority": finding.priority,
                            "track": finding.track,
                            "goal_id": goal_id,
                            "parent_goal_id": (
                                parent_goal_ids[0] if parent_goal_ids else ""
                            ),
                            "subgoal_id": goal_id if parent_goal_ids else "",
                            "parent_goal_ids": parent_goal_ids,
                            "graph_depth": len(parent_goal_ids),
                            "rationale": finding.summary,
                            "preconditions": [
                                f"{finding.root_relative_path} exists",
                                "scan evidence remains applicable",
                            ],
                            "effects": [
                                f"resolve {finding.kind} in {finding.root_relative_path}",
                                "pass focused validation",
                            ],
                            "evidence_subset": [
                                f"{finding.root_relative_path}:{finding.line_number}",
                                repo_relative_path(repo_root, discovery_path),
                            ],
                            "resource_class": "cpu-small",
                            "token_class": "small",
                            "resources": ["python", "focused-validation-runner"],
                            "merge_fate": finding.root_relative_path,
                            "rejection_reasons": [],
                            "acceptance": [
                                f"Resolve the {finding.kind} finding at "
                                f"{finding.root_relative_path}:{finding.line_number}."
                            ],
                            "validation": [finding.validation],
                            "paths": [finding.root_relative_path],
                            "outputs": [finding.root_relative_path],
                            "predicted_files": [finding.root_relative_path],
                            "ast_symbols": ast_symbols,
                            "ast_symbol_scope": "file",
                            "generated_artifacts": [repo_relative_path(repo_root, discovery_path)],
                            "depends_on": list(depends_on),
                            "bundle_strategy": "codebase_file_ast",
                            "surplus_group": goal_id,
                            "merge_key": bundle_key,
                            "merge_family": finding.root_relative_path,
                            "merge_role": "codebase_scan",
                            "work_item_count": 1,
                            "work_scope": "codebase_file_ast",
                            "candidate_kind": "codebase_scan",
                            "goal_registration": (
                                "existing" if goal_id else "unscoped_legacy"
                            ),
                            "todo_vector_key": finding.fingerprint[:16],
                            "discovery_path": repo_relative_path(repo_root, discovery_path),
                        },
                    }
                )
            appended.append(finding_record)

        replace_locked_taskboard(taskboard, todo_text)
    if bundle_dir is not None:
        generated_paths.extend(
            write_codebase_scan_bundle_shards(
                bundle_dir=bundle_dir,
                repo_root=repo_root,
                todo_path=todo_path,
                records=bundle_records,
            )
        )
    late_deduplicated_candidates = detected_count - len(appended)
    details_artifact = persist_codebase_scan_inventory(
        inventory,
        admission=admission,
        repo_root=repo_root,
        discovery_dir=discovery_dir,
        dataset_dir=dataset_dir,
        started_at=started_at,
        appended_tasks=len(appended),
        late_deduplicated_candidates=late_deduplicated_candidates,
    )
    strategy["last_codebase_scan_findings"] = appended
    if mode.endswith("drained_exhaustive"):
        strategy["last_drained_codebase_scan_task_count"] = task_count
    write_json(strategy_path, strategy)
    if commit_outputs:
        generated_paths.insert(0, todo_path)
        commit_results = commit_generated_outputs(
            generated_paths,
            repo_root=repo_root,
            subject=commit_subject,
        )
        if commit_results:
            strategy["last_codebase_scan_commit_results"] = commit_results
            write_json(strategy_path, strategy)
    nominal_reason = (
        ScanTerminalReason.GENERATED if appended else ScanTerminalReason.DUPLICATE_ONLY
    )
    health = classify_codebase_scan_health(
        inventory,
        admission=admission,
        appended_tasks=len(appended),
        late_deduplicated_candidates=late_deduplicated_candidates,
        canaries=canaries,
        thresholds=health_policy,
    )
    reason, completion_safe, health_error = fail_closed_scan_outcome(
        nominal_reason,
        health,
        generated_count=len(appended),
        exhaustive=mode.endswith("exhaustive"),
    )
    strategy["last_codebase_scan_health"] = health.to_dict()
    write_json(strategy_path, strategy)
    return build_scan_result(
        reason,
        mode,
        CODEBASE_SCAN_ANALYZER_VERSION,
        repo_root,
        started_at,
        appended,
        safe_for_completion_reasoning=completion_safe,
        error=health_error,
        metadata={
            **analyzer_health_metadata(
                thresholds=health_policy,
                canaries=canaries,
                health=health,
            ),
            **safe_codebase_scan_accounting_metadata(
                inventory,
                admission=admission,
                appended_tasks=len(appended),
                late_deduplicated_candidates=late_deduplicated_candidates,
                details_artifact=details_artifact,
            ),
            "detected_count": detected_count,
            "duplicate_count": detected_count - len(appended),
            "open_task_count": current_open,
            "task_count": task_count,
            "refill_capacity": refill_capacity,
            "open_task_target": max(0, int(min_open_tasks))
            + max(0, DEFAULT_REFILL_OPEN_TASK_HEADROOM),
        },
    )


def record_codebase_scan_findings_legacy(**kwargs: Any) -> list[dict[str, Any]]:
    """Explicit list compatibility wrapper for pre-receipt callers."""

    return list(record_codebase_scan_findings(**kwargs).items)


def record_codebase_audit_findings(
    *,
    repo_root: Path,
    dataset_dir: Path | None = None,
    **kwargs: Any,
) -> Any:
    """Run the independent audit path without touching taskboard/seen state.

    This deliberately accepts no todo or strategy path.  Audit persistence is
    confined to the dataset store, making it impossible for this wrapper to
    synchronize, retire, or append normal refill fingerprints.
    """

    from ..analysis.audit_scanner import run_audit_scan

    return run_audit_scan(repo_root, dataset_dir=dataset_dir, **kwargs)


def record_objective_backlog_findings(
    *,
    repo_root: Path,
    objective_path: Path,
    todo_path: Path,
    discovery_dir: Path,
    bundle_dir: Path,
    strategy_path: Path,
    state_path: Path | None = None,
    dataset_dir: Path | None = None,
    task_prefix: str = DEFAULT_TASK_ID_PREFIX,
    depends_on: Sequence[str] = (),
    min_open_tasks: int = DEFAULT_OBJECTIVE_SCAN_MIN_OPEN_TASKS,
    max_findings: int = DEFAULT_OBJECTIVE_SCAN_MAX_FINDINGS,
    cooldown_seconds: int = DEFAULT_OBJECTIVE_SCAN_COOLDOWN_SECONDS,
    force: bool = False,
    persist_ast_dataset: bool = True,
    write_todo_vector_index: bool = True,
    todo_vector_index_path: Path | None = None,
    surplus_findings_per_goal: int = DEFAULT_SURPLUS_FINDINGS_PER_GOAL,
    surplus_min_terms_per_todo: int = DEFAULT_SURPLUS_MIN_TERMS_PER_TODO,
    summary_prefix: str = DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX,
    discovery_output_path: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    force_goal_ids: Sequence[str] = (),
    commit_outputs: bool = False,
    commit_subject: str = "Agent: record objective backlog findings",
) -> RefillScanResult[dict[str, Any]]:
    """Feed a todo board from objective gaps and return a typed scan result."""

    # Objective generation is an external write boundary, so never forward a
    # legacy ``"## PREFIX-"`` value to its heading renderer.
    task_prefix = task_id_prefix(task_prefix)
    started_at = datetime.now(timezone.utc)
    if max_findings <= 0:
        return build_scan_result(
            ScanTerminalReason.DISABLED,
            "disabled",
            OBJECTIVE_SCAN_ANALYZER_VERSION,
            repo_root,
            started_at,
            metadata={"cause": "non_positive_max_findings"},
        )
    missing_inputs = [
        name
        for name, path in (("todo_path", todo_path), ("objective_path", objective_path))
        if not path.exists()
    ]
    if missing_inputs:
        return build_scan_result(
            ScanTerminalReason.FAILED,
            "preflight",
            OBJECTIVE_SCAN_ANALYZER_VERSION,
            repo_root,
            started_at,
            error=f"missing required scan input: {', '.join(missing_inputs)}",
            metadata={"missing_inputs": missing_inputs},
        )
    todo_text = todo_path.read_text(encoding="utf-8")
    strategy = load_strategy(strategy_path)
    should_scan, mode, current_open, task_count = should_refill_backlog(
        todo_text=todo_text,
        state_path=state_path,
        strategy=strategy,
        last_scan_key="last_objective_goal_scan_at",
        last_drained_scan_task_count_key="last_drained_objective_goal_scan_task_count",
        task_prefix=task_prefix,
        min_open_tasks=min_open_tasks,
        cooldown_seconds=cooldown_seconds,
        force=force,
    )
    if not should_scan:
        reason = (
            ScanTerminalReason.COOLDOWN
            if mode == "cooldown"
            else ScanTerminalReason.THRESHOLD_SATISFIED
        )
        return build_scan_result(
            reason,
            mode,
            OBJECTIVE_SCAN_ANALYZER_VERSION,
            repo_root,
            started_at,
            metadata={"open_task_count": current_open, "task_count": task_count},
        )
    capacity_open_count = (
        0 if mode.startswith("runnable_drained") else current_open
    )
    refill_capacity = refill_open_task_capacity(
        current_open=capacity_open_count,
        min_open_tasks=min_open_tasks,
        max_findings=max_findings,
    )
    if refill_capacity <= 0:
        return build_scan_result(
            ScanTerminalReason.THRESHOLD_SATISFIED,
            "open_task_pressure_bound",
            OBJECTIVE_SCAN_ANALYZER_VERSION,
            repo_root,
            started_at,
            metadata={
                "open_task_count": current_open,
                "task_count": task_count,
                "refill_capacity": 0,
                "open_task_target": max(0, int(min_open_tasks))
                + max(0, DEFAULT_REFILL_OPEN_TASK_HEADROOM),
            },
        )

    seen = {str(item) for item in strategy.get("objective_goal_seen_fingerprints", []) if str(item).strip()}
    generation_result = generate_objective_todos_result(
        scan_mode=mode,
        repo_root=repo_root,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        dataset_dir=dataset_dir,
        task_prefix=task_prefix,
        depends_on=depends_on,
        max_findings=refill_capacity,
        seen_fingerprints=seen,
        persist_ast_dataset=persist_ast_dataset,
        write_todo_vector_index=write_todo_vector_index,
        todo_vector_index_path=todo_vector_index_path,
        surplus_findings_per_goal=surplus_findings_per_goal,
        surplus_min_terms_per_todo=surplus_min_terms_per_todo,
        summary_prefix=summary_prefix,
        discovery_output_path=discovery_output_path,
        force_goal_ids=force_goal_ids,
    )
    records = list(generation_result.items)
    strategy["last_objective_goal_scan_at"] = utc_now()
    strategy["last_objective_goal_scan_mode"] = mode
    strategy["objective_goal_seen_fingerprints"] = sorted(
        seen | {record.finding.fingerprint for record in records}
    )

    appended = [
        {
            "follow_up_task_id": record.task_id,
            "fingerprint": record.finding.fingerprint,
            "kind": "objective_goal_gap",
            "goal_id": record.finding.goal_id,
            "missing_evidence": record.finding.missing_evidence,
            "bundle_key": record.finding.bundle_key,
            "bundle_shard": repo_relative_path(repo_root, bundle_dir / f"{safe_bundle_key(record.finding.bundle_key)}.todo.md"),
            "bundle_strategy": record.finding.bundle_strategy,
            "graph_depth": record.finding.graph_depth,
            "parent_goal_ids": record.finding.parent_goal_ids,
            "candidate_kind": record.finding.candidate_kind,
            "merge_key": record.finding.merge_key,
            "merge_family": record.finding.merge_family or record.finding.surplus_group,
            "merge_role": record.finding.merge_role or record.finding.candidate_kind,
            "work_item_count": record.finding.work_item_count or len(record.finding.missing_evidence),
            "work_scope": record.finding.work_scope,
            "goal_packet_key": record.finding.goal_packet_key,
            "goal_packet_role": record.finding.goal_packet_role,
            "goal_packet_goal_ids": record.finding.goal_packet_goal_ids,
            "goal_packet_task_count": record.finding.goal_packet_task_count,
            "goal_packet_work_item_count": record.finding.goal_packet_work_item_count,
            "todo_vector_key": record.finding.todo_vector_key,
            "discovery_path": str(record.discovery_path),
        }
        for record in records
    ]
    strategy["last_objective_todo_vector_index_path"] = str(
        todo_vector_index_path or bundle_dir / "todo_vector_index.json"
    )
    strategy["last_objective_surplus_findings_per_goal"] = surplus_findings_per_goal
    strategy["last_objective_surplus_min_terms_per_todo"] = surplus_min_terms_per_todo
    strategy["last_objective_goal_scan_findings"] = appended
    if mode.startswith("drained") and generation_result.terminal_reason in {
        ScanTerminalReason.GENERATED,
        ScanTerminalReason.EXHAUSTED,
    }:
        strategy["last_drained_objective_goal_scan_task_count"] = task_count
    write_json(strategy_path, strategy)
    if commit_outputs and records:
        generated_paths = [todo_path]
        generated_paths.extend(record.discovery_path for record in records)
        generated_paths.append(bundle_dir / "index.json")
        generated_paths.extend(bundle_path(bundle_dir, record.finding.bundle_key) for record in records)
        commit_results = commit_generated_outputs(
            generated_paths,
            repo_root=repo_root,
            subject=commit_subject,
        )
        if commit_results:
            strategy["last_objective_goal_commit_results"] = commit_results
            write_json(strategy_path, strategy)
    if generation_result.terminal_reason not in {
        ScanTerminalReason.GENERATED,
        ScanTerminalReason.EXHAUSTED,
    }:
        return build_scan_result(
            generation_result.terminal_reason,
            mode,
            generation_result.analyzer_version,
            repo_root,
            started_at,
            error=generation_result.error,
            metadata=generation_result.metadata,
        )
    return build_scan_result(
        ScanTerminalReason.GENERATED if appended else ScanTerminalReason.EXHAUSTED,
        mode,
        generation_result.analyzer_version,
        repo_root,
        started_at,
        appended,
        # An empty objective-gap result is proposal evidence, not healthy
        # exhaustion authority.  The benchmark-driven self-improvement epoch
        # separately requires explicit analyzer health, an exact context
        # binding, and an independent quorum before it advances the durable
        # drained marker.
        safe_for_completion_reasoning=False,
        metadata={
            **generation_result.metadata,
            "open_task_count": current_open,
            "task_count": task_count,
            "refill_capacity": refill_capacity,
            "open_task_target": max(0, int(min_open_tasks))
            + max(0, DEFAULT_REFILL_OPEN_TASK_HEADROOM),
            "healthy_epoch_required_for_completion": True,
        },
    )


def record_objective_backlog_findings_legacy(**kwargs: Any) -> list[dict[str, Any]]:
    """Explicit list compatibility wrapper for pre-receipt callers."""

    return list(record_objective_backlog_findings(**kwargs).items)


def record_configured_objective_backlog_findings(
    *,
    repo_root: Path,
    objective_path: Path,
    todo_path: Path,
    discovery_dir: Path,
    strategy_path: Path,
    state_path: Path | None = None,
    bundle_dir: Path | None = None,
    dataset_dir: Path | None = None,
    default_bundle_dir: Path | None = None,
    default_dataset_dir: Path | None = None,
    todo_vector_index_path: Path | None = None,
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX,
    depends_on_if_present: Sequence[str] = (),
    min_open_tasks: int = DEFAULT_OBJECTIVE_SCAN_MIN_OPEN_TASKS,
    max_findings: int = DEFAULT_OBJECTIVE_SCAN_MAX_FINDINGS,
    cooldown_seconds: int = DEFAULT_OBJECTIVE_SCAN_COOLDOWN_SECONDS,
    force: bool = False,
    persist_ast_dataset: bool = True,
    write_todo_vector_index: bool = True,
    surplus_findings_per_goal: int = DEFAULT_SURPLUS_FINDINGS_PER_GOAL,
    surplus_min_terms_per_todo: int = DEFAULT_SURPLUS_MIN_TERMS_PER_TODO,
    summary_prefix: str = DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX,
    discovery_output_path: str | None = None,
    discovery_output_path_default: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    force_goal_ids: Sequence[str] = (),
    completion_gate_decisions: Mapping[str, Any] | None = None,
    completion_gate_now: datetime | str | None = None,
    completion_gate_freshness_seconds: float = (
        DEFAULT_EVIDENCE_FRESHNESS_SECONDS
    ),
    completion_gate_clock_skew_seconds: float = DEFAULT_CLOCK_SKEW_SECONDS,
    commit_outputs: bool = False,
    commit_subject: str = "Agent: record objective backlog findings",
) -> RefillScanResult[dict[str, Any]]:
    """Run objective backlog refill with common wrapper-level defaults."""

    resolved_bundle_dir = (
        bundle_dir
        or default_bundle_dir
        or repo_root / "data" / "agent_supervisor" / "objective_bundles"
    )
    resolved_dataset_dir = dataset_dir if dataset_dir is not None else default_dataset_dir
    completion_identity = None
    if completion_gate_decisions:
        from .objective_tracker import completion_tree_identity

        completion_identity = completion_tree_identity(
            repo_root,
            objective_path=objective_path,
        )
    return record_objective_backlog_findings(
        repo_root=repo_root,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=resolved_bundle_dir,
        dataset_dir=resolved_dataset_dir,
        strategy_path=strategy_path,
        state_path=state_path,
        task_prefix=task_id_prefix(task_header_prefix_value),
        depends_on=task_dependencies_if_present(
            todo_path,
            task_header_prefix_value=task_header_prefix_value,
            dependency_ids=depends_on_if_present,
        ),
        min_open_tasks=min_open_tasks,
        max_findings=max_findings,
        cooldown_seconds=cooldown_seconds,
        force=force,
        persist_ast_dataset=persist_ast_dataset,
        write_todo_vector_index=write_todo_vector_index,
        todo_vector_index_path=todo_vector_index_path,
        surplus_findings_per_goal=surplus_findings_per_goal,
        surplus_min_terms_per_todo=surplus_min_terms_per_todo,
        summary_prefix=summary_prefix,
        discovery_output_path=discovery_output_path
        or discovery_output_path_for(repo_root, discovery_dir, default=discovery_output_path_default),
        force_goal_ids=align_completion_gate_force_goal_ids(
            force_goal_ids,
            completion_gate_decisions=completion_gate_decisions,
            repository_id=(
                completion_identity.repository_id
                if completion_identity is not None
                else ""
            ),
            repository_tree=(
                completion_identity.tree_id
                if completion_identity is not None
                else ""
            ),
            now=completion_gate_now,
            freshness_seconds=completion_gate_freshness_seconds,
            clock_skew_seconds=completion_gate_clock_skew_seconds,
        ),
        commit_outputs=commit_outputs,
        commit_subject=commit_subject,
    )


def record_configured_codebase_scan_findings(
    *,
    todo_path: Path,
    state_path: Path | None,
    strategy_path: Path,
    discovery_dir: Path,
    repo_root: Path,
    bundle_dir: Path | None = None,
    default_bundle_dir: Path | None = None,
    dataset_dir: Path | None = None,
    default_dataset_dir: Path | None = None,
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX,
    depends_on_if_present: Sequence[str] = (),
    min_open_tasks: int = DEFAULT_CODEBASE_SCAN_MIN_OPEN_TASKS,
    max_findings: int = DEFAULT_CODEBASE_SCAN_MAX_FINDINGS,
    cooldown_seconds: int = DEFAULT_CODEBASE_SCAN_COOLDOWN_SECONDS,
    force: bool = False,
    discovery_output_path: str | None = None,
    discovery_output_path_default: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    skip_prefixes: Sequence[str] = CODEBASE_SCAN_SKIP_PREFIXES,
    include_prefixes: Sequence[str] = (),
    allowed_tracks: Sequence[str] = (),
    objective_path: Path | None = None,
    mission_terms: Sequence[str] = (),
    allow_unscoped_codebase_refill: bool = False,
    health_thresholds: AnalyzerHealthThresholds | Mapping[str, Any] | None = None,
    commit_outputs: bool = False,
    commit_subject: str = "Agent: record codebase scan backlog findings",
) -> RefillScanResult[dict[str, Any]]:
    """Run codebase backlog refill with common wrapper-level defaults."""

    return record_codebase_scan_findings(
        todo_path=todo_path,
        state_path=state_path,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        repo_root=repo_root,
        bundle_dir=bundle_dir
        or default_bundle_dir
        or repo_root / "data" / "agent_supervisor" / "objective_bundles",
        dataset_dir=dataset_dir
        or default_dataset_dir
        or repo_root / "data" / "agent_supervisor" / "objective_datasets",
        task_prefix=task_id_prefix(task_header_prefix_value),
        depends_on=task_dependencies_if_present(
            todo_path,
            task_header_prefix_value=task_header_prefix_value,
            dependency_ids=depends_on_if_present,
        ),
        min_open_tasks=min_open_tasks,
        max_findings=max_findings,
        cooldown_seconds=cooldown_seconds,
        force=force,
        discovery_output_path=discovery_output_path
        or discovery_output_path_for(repo_root, discovery_dir, default=discovery_output_path_default),
        skip_prefixes=skip_prefixes,
        include_prefixes=include_prefixes,
        allowed_tracks=allowed_tracks,
        objective_path=objective_path,
        mission_terms=mission_terms,
        allow_unscoped_codebase_refill=allow_unscoped_codebase_refill,
        health_thresholds=health_thresholds,
        commit_outputs=commit_outputs,
        commit_subject=commit_subject,
    )


def record_configured_retry_budget_findings(
    *,
    todo_path: Path,
    events_path: Path,
    strategy_path: Path,
    discovery_dir: Path,
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX,
    validation_retry_budget: int = DEFAULT_VALIDATION_RETRY_BUDGET,
    merge_retry_budget: int = DEFAULT_MERGE_RETRY_BUDGET,
    implementation_retry_budget: int = DEFAULT_IMPLEMENTATION_RETRY_BUDGET,
    validation_depends_on_if_present: Sequence[str] = (),
    validation_task_command_transform: Callable[[str], str] | None = None,
    discovery_output_path: str | None = None,
    discovery_output_path_default: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    strip_validation_failure_kind: bool = False,
    commit_outputs: bool = False,
    repo_root: Path | None = None,
    commit_subject: str = "Agent: record retry-budget guardrail outputs",
) -> list[dict[str, Any]]:
    """Run retry-budget guardrails with common wrapper-level defaults."""

    if not todo_path.exists():
        return []
    resolved_repo_root = repo_root or git_toplevel_for_path(todo_path.parent) or todo_path.parent
    task_prefix_value = task_id_prefix(task_header_prefix_value)
    findings = record_retry_budget_findings(
        todo_path=todo_path,
        events_path=events_path,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        task_header_prefix_value=task_header_prefix_value,
        task_prefix=task_prefix_value,
        validation_retry_budget=validation_retry_budget,
        merge_retry_budget=merge_retry_budget,
        implementation_retry_budget=implementation_retry_budget,
        validation_depends_on=task_dependencies_if_present(
            todo_path,
            task_header_prefix_value=task_header_prefix_value,
            dependency_ids=validation_depends_on_if_present,
        ),
        validation_task_command_transform=validation_task_command_transform,
        discovery_output_path=discovery_output_path
        or discovery_output_path_for(
            resolved_repo_root,
            discovery_dir,
            default=discovery_output_path_default,
        ),
        commit_outputs=commit_outputs,
        repo_root=resolved_repo_root,
        commit_subject=commit_subject,
    )
    if strip_validation_failure_kind:
        for finding in findings:
            if finding.get("failure_kind") == "validation":
                finding.pop("failure_kind", None)
    return findings


def _configured_recorder_kwargs(
    recorder: object,
    overrides: Mapping[str, Any],
    *,
    aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    params = {
        field.name: getattr(recorder, field.name)
        for field in fields(recorder)
        if field.name != "prepare_environment"
    }
    translated = dict(overrides)
    translated.pop("prepare_environment", None)
    for source, target in (aliases or {}).items():
        if source not in translated:
            continue
        if target in translated:
            raise TypeError(f"received both {source!r} and {target!r}")
        translated[target] = translated.pop(source)
    params.update(translated)
    return params


def _prepare_configured_recorder(callback: Callable[[], None] | None) -> None:
    if callback is not None:
        callback()


@dataclass(frozen=True)
class ConfiguredObjectiveBacklogRecorder:
    """Callable objective-refill recorder with wrapper-specific defaults."""

    repo_root: Path
    objective_path: Path
    todo_path: Path
    discovery_dir: Path
    strategy_path: Path
    state_path: Path | None = None
    bundle_dir: Path | None = None
    dataset_dir: Path | None = None
    default_bundle_dir: Path | None = None
    default_dataset_dir: Path | None = None
    todo_vector_index_path: Path | None = None
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX
    depends_on_if_present: Sequence[str] = ()
    min_open_tasks: int = DEFAULT_OBJECTIVE_SCAN_MIN_OPEN_TASKS
    max_findings: int = DEFAULT_OBJECTIVE_SCAN_MAX_FINDINGS
    cooldown_seconds: int = DEFAULT_OBJECTIVE_SCAN_COOLDOWN_SECONDS
    force: bool = False
    persist_ast_dataset: bool = True
    write_todo_vector_index: bool = True
    surplus_findings_per_goal: int = DEFAULT_SURPLUS_FINDINGS_PER_GOAL
    surplus_min_terms_per_todo: int = DEFAULT_SURPLUS_MIN_TERMS_PER_TODO
    summary_prefix: str = DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX
    discovery_output_path: str | None = None
    discovery_output_path_default: str = DEFAULT_DISCOVERY_OUTPUT_PATH
    force_goal_ids: Sequence[str] = ()
    completion_gate_decisions: Mapping[str, Any] | None = None
    completion_gate_now: datetime | str | None = None
    completion_gate_freshness_seconds: float = (
        DEFAULT_EVIDENCE_FRESHNESS_SECONDS
    )
    completion_gate_clock_skew_seconds: float = DEFAULT_CLOCK_SKEW_SECONDS
    commit_outputs: bool = False
    commit_subject: str = "Agent: record objective backlog findings"
    prepare_environment: Callable[[], None] | None = None

    def __call__(self, **overrides: Any) -> RefillScanResult[dict[str, Any]]:
        _prepare_configured_recorder(self.prepare_environment)
        return record_configured_objective_backlog_findings(
            **_configured_recorder_kwargs(
                self,
                overrides,
                aliases={"task_header_prefix": "task_header_prefix_value"},
            )
        )


@dataclass(frozen=True)
class ConfiguredCodebaseScanRecorder:
    """Callable codebase-scan recorder with wrapper-specific defaults."""

    todo_path: Path
    state_path: Path | None
    strategy_path: Path
    discovery_dir: Path
    repo_root: Path
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX
    depends_on_if_present: Sequence[str] = ()
    min_open_tasks: int = DEFAULT_CODEBASE_SCAN_MIN_OPEN_TASKS
    max_findings: int = DEFAULT_CODEBASE_SCAN_MAX_FINDINGS
    cooldown_seconds: int = DEFAULT_CODEBASE_SCAN_COOLDOWN_SECONDS
    force: bool = False
    discovery_output_path: str | None = None
    discovery_output_path_default: str = DEFAULT_DISCOVERY_OUTPUT_PATH
    skip_prefixes: Sequence[str] = CODEBASE_SCAN_SKIP_PREFIXES
    include_prefixes: Sequence[str] = ()
    allowed_tracks: Sequence[str] = ()
    objective_path: Path | None = None
    mission_terms: Sequence[str] = ()
    allow_unscoped_codebase_refill: bool = False
    health_thresholds: AnalyzerHealthThresholds | Mapping[str, Any] | None = None
    commit_outputs: bool = False
    commit_subject: str = "Agent: record codebase scan backlog findings"
    bundle_dir: Path | None = None
    default_bundle_dir: Path | None = None
    dataset_dir: Path | None = None
    default_dataset_dir: Path | None = None
    prepare_environment: Callable[[], None] | None = None

    def __call__(self, **overrides: Any) -> RefillScanResult[dict[str, Any]]:
        _prepare_configured_recorder(self.prepare_environment)
        return record_configured_codebase_scan_findings(
            **_configured_recorder_kwargs(
                self,
                overrides,
                aliases={"task_header_prefix": "task_header_prefix_value"},
            )
        )


@dataclass(frozen=True)
class ConfiguredRetryBudgetRecorder:
    """Callable retry-budget recorder with wrapper-specific defaults."""

    todo_path: Path
    events_path: Path
    strategy_path: Path
    discovery_dir: Path
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX
    validation_retry_budget: int = DEFAULT_VALIDATION_RETRY_BUDGET
    merge_retry_budget: int = DEFAULT_MERGE_RETRY_BUDGET
    implementation_retry_budget: int = DEFAULT_IMPLEMENTATION_RETRY_BUDGET
    validation_depends_on_if_present: Sequence[str] = ()
    validation_task_command_transform: Callable[[str], str] | None = None
    discovery_output_path: str | None = None
    discovery_output_path_default: str = DEFAULT_DISCOVERY_OUTPUT_PATH
    strip_validation_failure_kind: bool = False
    commit_outputs: bool = False
    repo_root: Path | None = None
    commit_subject: str = "Agent: record retry-budget guardrail outputs"
    prepare_environment: Callable[[], None] | None = None

    def __call__(self, **overrides: Any) -> list[dict[str, Any]]:
        _prepare_configured_recorder(self.prepare_environment)
        return record_configured_retry_budget_findings(
            **_configured_recorder_kwargs(
                self,
                overrides,
                aliases={
                    "retry_budget": "validation_retry_budget",
                    "task_header_prefix": "task_header_prefix_value",
                },
            )
        )


ConfiguredBacklogRecordCallback = Callable[
    ..., RefillScanResult[dict[str, Any]] | list[dict[str, Any]]
]
ConfiguredBootstrapExtraKwargsFactory = Callable[[Mapping[str, Path | str]], Mapping[str, Any] | None]


@dataclass(frozen=True)
class ConfiguredBacklogRecorderBundle:
    """Reusable bridge from configured backlog recorders to runtime hook factories."""

    objective_recorder: ConfiguredBacklogRecordCallback | None = None
    codebase_scan_recorder: ConfiguredBacklogRecordCallback | None = None
    retry_budget_recorder: ConfiguredBacklogRecordCallback | None = None

    def daemon_refill_hooks_factory(
        self,
        *,
        discovery_dir_key: str | None = None,
        discovery_dir: Path | str | None = None,
        objective_path_key: str | None = None,
        objective_path: Path | str | None = None,
        repo_root: Path | None = None,
        objective_extra_kwargs: Mapping[str, Any] | None = None,
        objective_extra_kwargs_factory: ConfiguredBootstrapExtraKwargsFactory | None = None,
        codebase_scan_extra_kwargs: Mapping[str, Any] | None = None,
        codebase_scan_extra_kwargs_factory: ConfiguredBootstrapExtraKwargsFactory | None = None,
        retry_budget_extra_kwargs: Mapping[str, Any] | None = None,
        retry_budget_extra_kwargs_factory: ConfiguredBootstrapExtraKwargsFactory | None = None,
        scope_label: str = "",
        before: bool = True,
        after: bool = True,
        after_order: Sequence[str] | None = None,
        log_level: int = logging.WARNING,
    ) -> Callable[[Mapping[str, Path | str]], tuple[Any, ...]]:
        """Build daemon refill hooks from this bundle without repo-local wiring."""

        from ..todo_daemon.implementation_daemon_runner import build_daemon_refill_hooks_factory_from_recorders

        return build_daemon_refill_hooks_factory_from_recorders(
            discovery_dir_key=discovery_dir_key,
            discovery_dir=discovery_dir,
            objective_recorder=self.objective_recorder,
            codebase_scan_recorder=self.codebase_scan_recorder,
            retry_budget_recorder=self.retry_budget_recorder,
            objective_path_key=objective_path_key,
            objective_path=objective_path,
            repo_root=repo_root,
            objective_extra_kwargs=objective_extra_kwargs,
            objective_extra_kwargs_factory=objective_extra_kwargs_factory,
            codebase_scan_extra_kwargs=codebase_scan_extra_kwargs,
            codebase_scan_extra_kwargs_factory=codebase_scan_extra_kwargs_factory,
            retry_budget_extra_kwargs=retry_budget_extra_kwargs,
            retry_budget_extra_kwargs_factory=retry_budget_extra_kwargs_factory,
            scope_label=scope_label,
            before=before,
            after=after,
            after_order=after_order,
            log_level=log_level,
        )

    def supervisor_refill_hooks_factory(
        self,
        *,
        discovery_dir_key: str | None = None,
        discovery_dir: Path | str | None = None,
        objective_path_key: str | None = None,
        objective_path: Path | str | None = None,
        repo_root: Path | None = None,
        objective_extra_kwargs: Mapping[str, Any] | None = None,
        objective_extra_kwargs_factory: ConfiguredBootstrapExtraKwargsFactory | None = None,
        codebase_scan_extra_kwargs: Mapping[str, Any] | None = None,
        codebase_scan_extra_kwargs_factory: ConfiguredBootstrapExtraKwargsFactory | None = None,
        retry_budget_extra_kwargs: Mapping[str, Any] | None = None,
        retry_budget_extra_kwargs_factory: ConfiguredBootstrapExtraKwargsFactory | None = None,
        scope_label: str = "",
        before: bool = True,
        after_once: bool = True,
        after_once_order: Sequence[str] | None = None,
        log_level: int = logging.WARNING,
    ) -> Callable[[Mapping[str, Path | str]], tuple[Any, ...]]:
        """Build supervisor refill hooks from this bundle without repo-local wiring."""

        from ..todo_daemon.implementation_supervisor_runner import build_supervisor_refill_hooks_factory_from_recorders

        return build_supervisor_refill_hooks_factory_from_recorders(
            discovery_dir_key=discovery_dir_key,
            discovery_dir=discovery_dir,
            objective_recorder=self.objective_recorder,
            codebase_scan_recorder=self.codebase_scan_recorder,
            retry_budget_recorder=self.retry_budget_recorder,
            objective_path_key=objective_path_key,
            objective_path=objective_path,
            repo_root=repo_root,
            objective_extra_kwargs=objective_extra_kwargs,
            objective_extra_kwargs_factory=objective_extra_kwargs_factory,
            codebase_scan_extra_kwargs=codebase_scan_extra_kwargs,
            codebase_scan_extra_kwargs_factory=codebase_scan_extra_kwargs_factory,
            retry_budget_extra_kwargs=retry_budget_extra_kwargs,
            retry_budget_extra_kwargs_factory=retry_budget_extra_kwargs_factory,
            scope_label=scope_label,
            before=before,
            after_once=after_once,
            after_once_order=after_once_order,
            log_level=log_level,
        )


def build_configured_backlog_recorder_bundle(
    *,
    objective_recorder: ConfiguredBacklogRecordCallback | None = None,
    codebase_scan_recorder: ConfiguredBacklogRecordCallback | None = None,
    retry_budget_recorder: ConfiguredBacklogRecordCallback | None = None,
) -> ConfiguredBacklogRecorderBundle:
    """Collect configured backlog recorders for daemon/supervisor reuse."""

    return ConfiguredBacklogRecorderBundle(
        objective_recorder=objective_recorder,
        codebase_scan_recorder=codebase_scan_recorder,
        retry_budget_recorder=retry_budget_recorder,
    )


def build_namespace_objective_backlog_recorder(
    *,
    repo_root: Path | str,
    namespace_paths: AgentSupervisorNamespacePaths,
    objective_path: Path | str,
    todo_path: Path | str,
    strategy_path: Path | str,
    state_path: Path | str | None = None,
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX,
    depends_on_if_present: Sequence[str] = (),
    min_open_tasks: int = DEFAULT_OBJECTIVE_SCAN_MIN_OPEN_TASKS,
    max_findings: int = DEFAULT_OBJECTIVE_SCAN_MAX_FINDINGS,
    cooldown_seconds: int = DEFAULT_OBJECTIVE_SCAN_COOLDOWN_SECONDS,
    force: bool = False,
    persist_ast_dataset: bool = True,
    write_todo_vector_index: bool = True,
    surplus_findings_per_goal: int = DEFAULT_SURPLUS_FINDINGS_PER_GOAL,
    surplus_min_terms_per_todo: int = DEFAULT_SURPLUS_MIN_TERMS_PER_TODO,
    summary_prefix: str = DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX,
    discovery_output_path: str | None = None,
    discovery_output_path_default: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    force_goal_ids: Sequence[str] = (),
    completion_gate_decisions: Mapping[str, Any] | None = None,
    completion_gate_now: datetime | str | None = None,
    completion_gate_freshness_seconds: float = (
        DEFAULT_EVIDENCE_FRESHNESS_SECONDS
    ),
    completion_gate_clock_skew_seconds: float = DEFAULT_CLOCK_SKEW_SECONDS,
    commit_outputs: bool = False,
    commit_subject: str = "Agent: record objective backlog findings",
    prepare_environment: Callable[[], None] | None = None,
) -> ConfiguredObjectiveBacklogRecorder:
    """Build an objective recorder using standard namespace artifact paths."""

    return ConfiguredObjectiveBacklogRecorder(
        repo_root=Path(repo_root),
        objective_path=Path(objective_path),
        todo_path=Path(todo_path),
        discovery_dir=namespace_paths.discovery_dir,
        default_bundle_dir=namespace_paths.objective_bundle_dir,
        default_dataset_dir=namespace_paths.objective_dataset_dir,
        todo_vector_index_path=namespace_paths.objective_todo_vector_index_path,
        strategy_path=Path(strategy_path),
        state_path=Path(state_path) if state_path is not None else None,
        task_header_prefix_value=task_header_prefix_value,
        depends_on_if_present=tuple(depends_on_if_present),
        min_open_tasks=min_open_tasks,
        max_findings=max_findings,
        cooldown_seconds=cooldown_seconds,
        force=force,
        persist_ast_dataset=persist_ast_dataset,
        write_todo_vector_index=write_todo_vector_index,
        surplus_findings_per_goal=surplus_findings_per_goal,
        surplus_min_terms_per_todo=surplus_min_terms_per_todo,
        summary_prefix=summary_prefix,
        discovery_output_path=discovery_output_path,
        discovery_output_path_default=discovery_output_path_default,
        force_goal_ids=tuple(force_goal_ids),
        completion_gate_decisions=completion_gate_decisions,
        completion_gate_now=completion_gate_now,
        completion_gate_freshness_seconds=(
            completion_gate_freshness_seconds
        ),
        completion_gate_clock_skew_seconds=(
            completion_gate_clock_skew_seconds
        ),
        commit_outputs=commit_outputs,
        commit_subject=commit_subject,
        prepare_environment=prepare_environment,
    )


def build_namespace_codebase_scan_recorder(
    *,
    repo_root: Path | str,
    namespace_paths: AgentSupervisorNamespacePaths,
    todo_path: Path | str,
    strategy_path: Path | str,
    state_path: Path | str | None = None,
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX,
    depends_on_if_present: Sequence[str] = (),
    min_open_tasks: int = DEFAULT_CODEBASE_SCAN_MIN_OPEN_TASKS,
    max_findings: int = DEFAULT_CODEBASE_SCAN_MAX_FINDINGS,
    cooldown_seconds: int = DEFAULT_CODEBASE_SCAN_COOLDOWN_SECONDS,
    force: bool = False,
    discovery_output_path: str | None = None,
    discovery_output_path_default: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    skip_prefixes: Sequence[str] = CODEBASE_SCAN_SKIP_PREFIXES,
    include_prefixes: Sequence[str] = (),
    allowed_tracks: Sequence[str] = (),
    objective_path: Path | str | None = None,
    mission_terms: Sequence[str] = (),
    allow_unscoped_codebase_refill: bool = False,
    health_thresholds: AnalyzerHealthThresholds | Mapping[str, Any] | None = None,
    commit_outputs: bool = False,
    commit_subject: str = "Agent: record codebase scan backlog findings",
    prepare_environment: Callable[[], None] | None = None,
) -> ConfiguredCodebaseScanRecorder:
    """Build a codebase-scan recorder using a standard namespace discovery path."""

    return ConfiguredCodebaseScanRecorder(
        repo_root=Path(repo_root),
        todo_path=Path(todo_path),
        state_path=Path(state_path) if state_path is not None else None,
        strategy_path=Path(strategy_path),
        discovery_dir=namespace_paths.discovery_dir,
        default_bundle_dir=namespace_paths.objective_bundle_dir,
        default_dataset_dir=namespace_paths.objective_dataset_dir,
        task_header_prefix_value=task_header_prefix_value,
        depends_on_if_present=tuple(depends_on_if_present),
        min_open_tasks=min_open_tasks,
        max_findings=max_findings,
        cooldown_seconds=cooldown_seconds,
        force=force,
        discovery_output_path=discovery_output_path,
        discovery_output_path_default=discovery_output_path_default,
        skip_prefixes=tuple(skip_prefixes),
        include_prefixes=tuple(include_prefixes),
        allowed_tracks=tuple(allowed_tracks),
        objective_path=Path(objective_path) if objective_path is not None else None,
        mission_terms=tuple(mission_terms),
        allow_unscoped_codebase_refill=allow_unscoped_codebase_refill,
        health_thresholds=health_thresholds,
        commit_outputs=commit_outputs,
        commit_subject=commit_subject,
        prepare_environment=prepare_environment,
    )


def build_namespace_retry_budget_recorder(
    *,
    namespace_paths: AgentSupervisorNamespacePaths,
    todo_path: Path | str,
    events_path: Path | str,
    strategy_path: Path | str,
    task_header_prefix_value: str = DEFAULT_TASK_HEADER_PREFIX,
    validation_retry_budget: int = DEFAULT_VALIDATION_RETRY_BUDGET,
    merge_retry_budget: int = DEFAULT_MERGE_RETRY_BUDGET,
    implementation_retry_budget: int = DEFAULT_IMPLEMENTATION_RETRY_BUDGET,
    validation_depends_on_if_present: Sequence[str] = (),
    validation_task_command_transform: Callable[[str], str] | None = None,
    discovery_output_path: str | None = None,
    discovery_output_path_default: str = DEFAULT_DISCOVERY_OUTPUT_PATH,
    strip_validation_failure_kind: bool = False,
    commit_outputs: bool = False,
    repo_root: Path | str | None = None,
    commit_subject: str = "Agent: record retry-budget guardrail outputs",
    prepare_environment: Callable[[], None] | None = None,
) -> ConfiguredRetryBudgetRecorder:
    """Build a retry-budget recorder using a standard namespace discovery path."""

    return ConfiguredRetryBudgetRecorder(
        todo_path=Path(todo_path),
        events_path=Path(events_path),
        strategy_path=Path(strategy_path),
        discovery_dir=namespace_paths.discovery_dir,
        task_header_prefix_value=task_header_prefix_value,
        validation_retry_budget=validation_retry_budget,
        merge_retry_budget=merge_retry_budget,
        implementation_retry_budget=implementation_retry_budget,
        validation_depends_on_if_present=tuple(validation_depends_on_if_present),
        validation_task_command_transform=validation_task_command_transform,
        discovery_output_path=discovery_output_path,
        discovery_output_path_default=discovery_output_path_default,
        strip_validation_failure_kind=strip_validation_failure_kind,
        commit_outputs=commit_outputs,
        repo_root=Path(repo_root) if repo_root is not None else None,
        commit_subject=commit_subject,
        prepare_environment=prepare_environment,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refill and guard an accelerator todo backlog")
    health_defaults = AnalyzerHealthThresholds()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--todo-path", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, default=None)
    parser.add_argument("--strategy-path", type=Path, default=None)
    parser.add_argument("--events-path", type=Path, default=None)
    parser.add_argument("--discovery-dir", type=Path, default=None)
    parser.add_argument("--discovery-output-path", default=DEFAULT_DISCOVERY_OUTPUT_PATH)
    parser.add_argument("--objective-path", type=Path, default=None)
    parser.add_argument("--bundle-dir", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--task-prefix", default=DEFAULT_TASK_ID_PREFIX)
    parser.add_argument("--task-header-prefix", default=DEFAULT_TASK_HEADER_PREFIX)
    parser.add_argument("--depends-on", action="append", default=[])
    parser.add_argument("--validation-depends-on", action="append", default=[])
    parser.add_argument("--skip-prefix", action="append", default=[])
    parser.add_argument("--objective-scan", action="store_true")
    parser.add_argument("--codebase-scan", action="store_true")
    parser.add_argument(
        "--allow-unscoped-codebase-refill",
        action="store_true",
        help=(
            "Allow codebase findings without objective lineage to become tasks. "
            "Unsafe for goal-backed boards."
        ),
    )
    parser.add_argument("--retry-budget", action="store_true")
    parser.add_argument("--dependency-guardrail", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-open-tasks", type=int, default=DEFAULT_CODEBASE_SCAN_MIN_OPEN_TASKS)
    parser.add_argument("--max-findings", type=int, default=DEFAULT_CODEBASE_SCAN_MAX_FINDINGS)
    parser.add_argument("--cooldown-seconds", type=int, default=DEFAULT_CODEBASE_SCAN_COOLDOWN_SECONDS)
    parser.add_argument(
        "--analyzer-max-parser-failures",
        type=int,
        default=health_defaults.max_parser_failures,
    )
    parser.add_argument(
        "--analyzer-max-parser-failure-ratio",
        type=float,
        default=health_defaults.max_parser_failure_ratio,
    )
    parser.add_argument(
        "--analyzer-max-excluded-file-ratio",
        type=float,
        default=health_defaults.max_excluded_file_ratio,
    )
    parser.add_argument(
        "--analyzer-min-git-root-ratio",
        type=float,
        default=health_defaults.min_git_root_discovery_ratio,
    )
    parser.add_argument(
        "--analyzer-min-git-roots",
        type=int,
        default=health_defaults.min_git_roots,
    )
    parser.add_argument(
        "--no-analyzer-canaries",
        action="store_true",
        help="Allow scans without canaries, while still classifying them partial.",
    )
    parser.add_argument("--validation-retry-budget", type=int, default=DEFAULT_VALIDATION_RETRY_BUDGET)
    parser.add_argument("--merge-retry-budget", type=int, default=DEFAULT_MERGE_RETRY_BUDGET)
    parser.add_argument("--implementation-retry-budget", type=int, default=DEFAULT_IMPLEMENTATION_RETRY_BUDGET)
    parser.add_argument("--no-persist-ast-dataset", action="store_true")
    parser.add_argument("--no-objective-todo-vector-index", action="store_true")
    parser.add_argument("--objective-todo-vector-index-path", type=Path, default=None)
    parser.add_argument(
        "--objective-surplus-findings-per-goal",
        type=int,
        default=DEFAULT_SURPLUS_FINDINGS_PER_GOAL,
    )
    parser.add_argument(
        "--objective-surplus-min-terms-per-todo",
        type=int,
        default=DEFAULT_SURPLUS_MIN_TERMS_PER_TODO,
    )
    parser.add_argument("--objective-summary-prefix", default=DEFAULT_OBJECTIVE_TASK_SUMMARY_PREFIX)
    parser.add_argument("--commit-generated-outputs", action="store_true")
    return parser


def run_backlog_refinery(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    state_root = repo_root / "data" / "agent_supervisor"
    strategy_path = (args.strategy_path or state_root / "strategy.json").resolve()
    discovery_dir = (args.discovery_dir or state_root / "discovery").resolve()
    bundle_dir = (args.bundle_dir or state_root / "objective_bundles").resolve()
    state_path = args.state_path.resolve() if args.state_path else None
    events_path = args.events_path.resolve() if args.events_path else state_root / "events.jsonl"
    depends_on = split_csv(args.depends_on)
    validation_depends_on = split_csv(args.validation_depends_on)
    skip_prefixes = tuple(args.skip_prefix) if args.skip_prefix else CODEBASE_SCAN_SKIP_PREFIXES
    health_defaults = AnalyzerHealthThresholds()
    health_thresholds = AnalyzerHealthThresholds(
        require_canaries=not bool(getattr(args, "no_analyzer_canaries", False)),
        max_parser_failures=getattr(
            args, "analyzer_max_parser_failures", health_defaults.max_parser_failures
        ),
        max_parser_failure_ratio=getattr(
            args,
            "analyzer_max_parser_failure_ratio",
            health_defaults.max_parser_failure_ratio,
        ),
        max_excluded_file_ratio=getattr(
            args,
            "analyzer_max_excluded_file_ratio",
            health_defaults.max_excluded_file_ratio,
        ),
        min_git_root_discovery_ratio=getattr(
            args,
            "analyzer_min_git_root_ratio",
            health_defaults.min_git_root_discovery_ratio,
        ),
        require_git_root=health_defaults.require_git_root,
        min_git_roots=getattr(args, "analyzer_min_git_roots", health_defaults.min_git_roots),
        require_complete_funnel=health_defaults.require_complete_funnel,
    )

    run_all = not (args.objective_scan or args.codebase_scan or args.retry_budget or args.dependency_guardrail)
    objective_findings: list[dict[str, Any]] = []
    codebase_findings: list[dict[str, Any]] = []
    retry_findings: list[dict[str, Any]] = []
    dependency_findings: list[dict[str, Any]] = []

    if (args.objective_scan or run_all) and args.objective_path:
        objective_result = record_objective_backlog_findings(
            repo_root=repo_root,
            objective_path=args.objective_path.resolve(),
            todo_path=args.todo_path.resolve(),
            discovery_dir=discovery_dir,
            bundle_dir=bundle_dir,
            dataset_dir=args.dataset_dir.resolve() if args.dataset_dir else None,
            strategy_path=strategy_path,
            state_path=state_path,
            task_prefix=args.task_prefix,
            depends_on=depends_on,
            min_open_tasks=args.min_open_tasks,
            max_findings=args.max_findings,
            cooldown_seconds=args.cooldown_seconds,
            force=args.force,
            persist_ast_dataset=not args.no_persist_ast_dataset,
            write_todo_vector_index=not args.no_objective_todo_vector_index,
            todo_vector_index_path=args.objective_todo_vector_index_path,
            surplus_findings_per_goal=args.objective_surplus_findings_per_goal,
            surplus_min_terms_per_todo=args.objective_surplus_min_terms_per_todo,
            summary_prefix=args.objective_summary_prefix,
            discovery_output_path=args.discovery_output_path,
            commit_outputs=args.commit_generated_outputs,
        )
        objective_findings = list(objective_result.items)
    if args.codebase_scan or run_all:
        codebase_result = record_codebase_scan_findings(
            todo_path=args.todo_path.resolve(),
            state_path=state_path,
            strategy_path=strategy_path,
            discovery_dir=discovery_dir,
            repo_root=repo_root,
            bundle_dir=bundle_dir,
            task_prefix=args.task_prefix,
            depends_on=depends_on,
            min_open_tasks=args.min_open_tasks,
            max_findings=args.max_findings,
            cooldown_seconds=args.cooldown_seconds,
            force=args.force,
            discovery_output_path=args.discovery_output_path,
            skip_prefixes=skip_prefixes,
            objective_path=args.objective_path.resolve() if args.objective_path else None,
            allow_unscoped_codebase_refill=args.allow_unscoped_codebase_refill,
            health_thresholds=health_thresholds,
            commit_outputs=args.commit_generated_outputs,
        )
        codebase_findings = list(codebase_result.items)
    if args.retry_budget or run_all:
        retry_findings = record_retry_budget_findings(
            todo_path=args.todo_path.resolve(),
            events_path=events_path,
            strategy_path=strategy_path,
            discovery_dir=discovery_dir,
            task_header_prefix_value=args.task_header_prefix,
            task_prefix=args.task_prefix,
            validation_retry_budget=args.validation_retry_budget,
            merge_retry_budget=args.merge_retry_budget,
            implementation_retry_budget=args.implementation_retry_budget,
            validation_depends_on=validation_depends_on,
            discovery_output_path=args.discovery_output_path,
            commit_outputs=args.commit_generated_outputs,
            repo_root=repo_root,
        )
    if args.dependency_guardrail or run_all:
        dependency_findings = record_dependency_guardrail_findings(
            todo_path=args.todo_path.resolve(),
            strategy_path=strategy_path,
            discovery_dir=discovery_dir,
            task_header_prefix_value=args.task_header_prefix,
            task_prefix=args.task_prefix,
            max_findings=args.max_findings,
            discovery_output_path=args.discovery_output_path,
            commit_outputs=args.commit_generated_outputs,
            repo_root=repo_root,
        )

    return {
        "schema": "ipfs_accelerate_py.agent_supervisor.objectives.backlog_refinery",
        "repo_root": str(repo_root),
        "todo_path": str(args.todo_path.resolve()),
        "strategy_path": str(strategy_path),
        "objective_generated_count": len(objective_findings),
        "codebase_generated_count": len(codebase_findings),
        "retry_budget_generated_count": len(retry_findings),
        "dependency_guardrail_generated_count": len(dependency_findings),
        "objective_findings": objective_findings,
        "codebase_findings": codebase_findings,
        "retry_budget_findings": retry_findings,
        "dependency_guardrail_findings": dependency_findings,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    payload = run_backlog_refinery(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
