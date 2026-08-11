#!/usr/bin/env python3
"""Cheap, fail-closed validator for the IncrementalProofSealer board.

The validator deliberately uses only the Python standard library.  It checks
the reviewed control-plane contract; it does not import project packages, run
proof backends, install dependencies, or mutate any repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT
    / "config/agent_supervisor_incremental_proof_sealer_scheduler.json"
)
PLAN_PATH = REPO_ROOT / "docs/architecture/INCREMENTAL_PROOF_SEALER_PLAN.md"
OBJECTIVES_PATH = (
    REPO_ROOT
    / "docs/architecture/incremental_proof_sealer.objectives.md"
)
TASKBOARD_PATH = (
    REPO_ROOT / "docs/architecture/incremental_proof_sealer.todo.md"
)

BOARD_NAMESPACE = "incremental-proof-sealer-v1"
BRANCH = "agent/incremental-proof-sealer-v1"
ACCELERATE_REVISION = "8881344bb2162f3f8d82f22d8348bc0ac7536f95"
DATASETS_REVISION = "bd2ff6245ebe476fc744d45c7c66235c92b0e19c"
KIT_REVISION = "5a7a2df8181cfdc33bc19be09989df7ff83f2d4e"

TASK_IDS = tuple(f"IPS-{index:03d}" for index in range(57))
GOAL_IDS = ("IPS-G000",) + tuple(
    f"IPS-G{index:03d}" for index in range(10, 131, 10)
)
INITIAL_COMPLETED = frozenset({"IPS-000"})
INITIAL_READY = frozenset({"IPS-001", "IPS-002", "IPS-003"})
TERMINAL_TASK = "IPS-056"
ARTIFACT_CHECK_TASKS = (
    "IPS-001",
    "IPS-002",
    "IPS-003",
    "IPS-053",
    "IPS-054",
    "IPS-055",
)

EXPECTED_TASK_GROUPS: Mapping[str, tuple[str, ...]] = {
    "IPS-G010": tuple(f"IPS-{index:03d}" for index in range(0, 5)),
    "IPS-G020": tuple(f"IPS-{index:03d}" for index in range(5, 13)),
    "IPS-G030": tuple(f"IPS-{index:03d}" for index in range(13, 18)),
    "IPS-G040": tuple(f"IPS-{index:03d}" for index in range(18, 23)),
    "IPS-G050": tuple(f"IPS-{index:03d}" for index in range(23, 28)),
    "IPS-G060": tuple(f"IPS-{index:03d}" for index in range(28, 32)),
    "IPS-G070": tuple(f"IPS-{index:03d}" for index in range(32, 38)),
    "IPS-G080": tuple(f"IPS-{index:03d}" for index in range(38, 43)),
    "IPS-G090": tuple(f"IPS-{index:03d}" for index in range(43, 45)),
    "IPS-G100": tuple(f"IPS-{index:03d}" for index in range(45, 48)),
    "IPS-G110": tuple(f"IPS-{index:03d}" for index in range(48, 52)),
    "IPS-G120": tuple(f"IPS-{index:03d}" for index in range(52, 55)),
    "IPS-G130": tuple(f"IPS-{index:03d}" for index in range(55, 57)),
}
EXPECTED_TASK_TO_GOAL = {
    task_id: goal_id
    for goal_id, task_ids in EXPECTED_TASK_GROUPS.items()
    for task_id in task_ids
}

REQUIRED_TASK_FIELDS = (
    "status",
    "completion",
    "is schedulable",
    "review only",
    "priority",
    "track",
    "depends on",
    "goal id",
    "outputs",
    "validation",
    "board namespace",
    "bundle",
    "parallel lane",
    "resource class",
    "resource stage",
    "estimated tokens",
    "implementation timeout seconds",
    "predicted files",
    "submodules",
    "interfaces",
    "allow concurrent with",
    "conflict policy",
    "preconditions",
    "effects",
    "evidence subset",
    "symbolic first",
    "acceptance",
    "embedding query",
)
REQUIRED_GOAL_FIELDS = (
    "status",
    "parent",
    "depends on",
    "fib priority",
    "priority",
    "track",
    "bundle",
    "parallel lane",
    "resource class",
    "goal",
    "evidence",
    "acceptance criteria",
    "outputs",
    "validation",
    "acceptance",
    "gap task",
    "refinement",
    "conflict policy",
)

CONTROL_PATHS = (
    ".gitignore",
    "config/agent_supervisor_incremental_proof_sealer_scheduler.json",
    "docs/architecture/INCREMENTAL_PROOF_SEALER_PLAN.md",
    "docs/architecture/incremental_proof_sealer.objectives.md",
    "docs/architecture/incremental_proof_sealer.todo.md",
    "scripts/validate_incremental_proof_sealer_board.py",
)

REQUIRED_PLAN_CONCEPTS = (
    "IntegrityCommitment",
    "SignedExecutionReceipt",
    "ReceiptAggregationZkProof",
    "DirectExecutionProof",
    "IncrementalCommitSeal",
    "ProofUnit@1",
    "VerificationRequirementManifest@1",
    "ProofCacheKey@1",
    "source_root_cid",
    "repository_state_cid",
    "source_depends_on",
    "schema_depends_on",
    "fixture_depends_on",
    "config_depends_on",
    "proof_depends_on",
    "aggregate_contains",
    "supersedes",
    "invalidates",
    "RepositoryProofForest",
    "source_integrity_root",
    "static_analysis_root",
    "type_check_root",
    "unit_test_root",
    "integration_test_root",
    "property_test_root",
    "formal_obligation_root",
    "direct_zk_root",
    "release_invariant_root",
    "FullCheckpointSeal",
    "DeltaSeal@1",
    "compare-and-swap",
    "WAL phases",
    "manifest aggregation",
    "bounded fan-in",
    "chain compaction",
)
REQUIRED_CLI_TERMS = (
    "`full`",
    "`incremental`",
    "`verify`",
    "`plan`",
    "`explain-reuse`",
    "`explain-invalidation`",
    "`benchmark`",
    "`cache-status`",
    "`force-full`",
    "`compact`",
)
REQUIRED_INVALIDATION_TERMS = (
    "Source implementation change",
    "public interface",
    "Test source change",
    "Deleted tests",
    "Added selected tests",
    "Dependency-lock change",
    "Fixture or configuration change",
    "Circuit or proving/verification-key change",
    "Canonicalization or dependency-graph schema change",
    "Environment-policy change",
    "policy changes",
    "Documentation-only changes",
)
REQUIRED_NEGATIVE_TERMS = (
    "source root",
    "environment",
    "selector",
    "verification key",
    "circuit",
    "dependency closure",
    "public input",
    "invalid cryptography",
    "unsigned required receipt",
    "unauthorized test removal",
    "changed manifest with old aggregate",
    "wrong parent",
    "missing invalidated unit",
    "simulated production unit",
    "unknown/timeout",
    "lost unaffected leaf",
    "stale CAS writer",
)
REQUIRED_CRASH_TERMS = (
    "before proof execution",
    "after proof execution, before receipt persistence",
    "after receipt persistence, before forest update",
    "after forest update, before aggregate generation",
    "after aggregate generation, before seal persistence",
    "after seal persistence, before current-root CAS",
    "after CAS, before transaction cleanup",
)


class DuplicateJsonKey(ValueError):
    """Raised when a JSON object repeats a key."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DuplicateJsonKey(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _load_json(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        result = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"cannot load duplicate-free config {path}: {exc}")
        return {}
    if not isinstance(result, dict):
        errors.append("scheduler config must be one JSON object")
        return {}
    return result


def _read(path: Path, errors: list[str]) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        errors.append(f"cannot read {path}: {exc}")
        return ""


def _parse_markdown_records(
    text: str,
    heading_pattern: re.Pattern[str],
    label: str,
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    matches = list(heading_pattern.finditer(text))
    records: dict[str, dict[str, Any]] = {}
    seen_titles: set[str] = set()
    for index, match in enumerate(matches):
        record_id = match.group(1)
        title = match.group(2).strip()
        if record_id in records:
            errors.append(f"duplicate {label} heading: {record_id}")
            continue
        full_title = f"{record_id} {title}".casefold()
        if full_title in seen_titles:
            errors.append(f"duplicate {label} title: {record_id} {title}")
        seen_titles.add(full_title)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end]
        fields: dict[str, str] = {}
        for field_match in re.finditer(
            r"^- ([^:\n]+):[ \t]*(.*)$", body, flags=re.MULTILINE
        ):
            key = field_match.group(1).strip().casefold()
            if key in fields:
                errors.append(f"{record_id} repeats metadata field {key!r}")
            else:
                fields[key] = field_match.group(2).strip()
        records[record_id] = {"title": title, "fields": fields}
    return records


def _ids(value: str, pattern: str) -> tuple[str, ...]:
    return tuple(re.findall(pattern, value))


def _as_bool(value: str) -> bool | None:
    folded = value.strip().casefold()
    if folded == "true":
        return True
    if folded == "false":
        return False
    return None


def _as_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _cycle_nodes(graph: Mapping[str, Iterable[str]]) -> set[str]:
    """Return nodes participating in, or downstream from, a dependency cycle."""

    indegree = {node: 0 for node in graph}
    dependents: dict[str, set[str]] = defaultdict(set)
    for node, dependencies in graph.items():
        for dependency in dependencies:
            if dependency in indegree:
                indegree[node] += 1
                dependents[dependency].add(node)
    queue = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    visited: set[str] = set()
    while queue:
        node = queue.popleft()
        visited.add(node)
        for dependent in sorted(dependents[node]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                queue.append(dependent)
    return set(graph) - visited


def _reachable_from(
    start: Iterable[str], dependencies: Mapping[str, set[str]]
) -> set[str]:
    dependents: dict[str, set[str]] = defaultdict(set)
    for node, prerequisites in dependencies.items():
        for prerequisite in prerequisites:
            dependents[prerequisite].add(node)
    reached = set(start)
    queue = deque(sorted(reached))
    while queue:
        for dependent in sorted(dependents[queue.popleft()]):
            if dependent not in reached:
                reached.add(dependent)
                queue.append(dependent)
    return reached


def _ancestors(node: str, dependencies: Mapping[str, set[str]]) -> set[str]:
    reached: set[str] = set()
    queue = deque(sorted(dependencies.get(node, set())))
    while queue:
        dependency = queue.popleft()
        if dependency in reached:
            continue
        reached.add(dependency)
        queue.extend(sorted(dependencies.get(dependency, set())))
    return reached


def _git(
    *args: str, cwd: Path = REPO_ROOT, timeout: float = 3.0
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ("git", *args),
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(
            args=("git", *args), returncode=124, stdout="", stderr=str(exc)
        )


def _check_equal(
    actual: Any, expected: Any, name: str, errors: list[str]
) -> None:
    if actual != expected:
        errors.append(f"{name} must be {expected!r}; got {actual!r}")


def _validate_config(config: dict[str, Any], errors: list[str]) -> None:
    _check_equal(config.get("board_namespace"), BOARD_NAMESPACE, "board_namespace", errors)
    _check_equal(config.get("merge_target_branch"), BRANCH, "merge_target_branch", errors)
    _check_equal(config.get("task_prefix"), "## IPS-", "task_prefix", errors)
    _check_equal(config.get("goal_prefix"), "IPS-G", "goal_prefix", errors)
    _check_equal(config.get("max_lanes"), 3, "max_lanes", errors)
    _check_equal(config.get("strict_task_sharding"), True, "strict_task_sharding", errors)
    _check_equal(
        config.get("exit_when_all_tracks_terminal"),
        True,
        "exit_when_all_tracks_terminal",
        errors,
    )
    _check_equal(config.get("objective_refill_enabled"), False, "objective_refill_enabled", errors)
    _check_equal(config.get("codebase_refill_enabled"), False, "codebase_refill_enabled", errors)
    _check_equal(
        set(config.get("worktree_submodule_paths", ()))
        if isinstance(config.get("worktree_submodule_paths"), list)
        else config.get("worktree_submodule_paths"),
        {"ipfs_datasets_py", "ipfs_kit_py"},
        "worktree_submodule_paths",
        errors,
    )
    _check_equal(
        set(config.get("protected_paths", ()))
        if isinstance(config.get("protected_paths"), list)
        else config.get("protected_paths"),
        set(CONTROL_PATHS),
        "protected_paths",
        errors,
    )
    validation_workers = config.get("validation_max_workers")
    if not isinstance(validation_workers, int) or validation_workers <= 0:
        errors.append("validation_max_workers must be a positive integer")
    provider = config.get("provider")
    if not isinstance(provider, dict):
        errors.append("provider must be an object")
    else:
        expected_route = {
            "primary_provider_id": "grok_cli",
            "primary_model_id": "grok-4.5",
            "fallback_provider_id": "codex",
            "fallback_model_id": "gpt-5.6-terra",
            "fallback_trigger": "primary_quota_exhausted",
            "fallback_reasoning_effort": "medium",
            "max_concurrency": 3,
            "secrets_from_environment_only": True,
            "secrets_in_argv_prompts_logs_or_receipts": False,
        }
        for key, expected in expected_route.items():
            _check_equal(provider.get(key), expected, f"provider.{key}", errors)
    _check_equal(config.get("taskboard_path"), str(TASKBOARD_PATH.relative_to(REPO_ROOT)), "taskboard_path", errors)
    _check_equal(config.get("objectives_path"), str(OBJECTIVES_PATH.relative_to(REPO_ROOT)), "objectives_path", errors)
    _check_equal(config.get("plan_path"), str(PLAN_PATH.relative_to(REPO_ROOT)), "plan_path", errors)
    _check_equal(
        config.get("validator_path"),
        str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "validator_path",
        errors,
    )

    projection = config.get("initial_projection")
    if not isinstance(projection, dict):
        errors.append("initial_projection must be an object")
    else:
        _check_equal(projection.get("task_count"), len(TASK_IDS), "initial task_count", errors)
        _check_equal(
            projection.get("completed_task_ids"),
            sorted(INITIAL_COMPLETED),
            "initial completed_task_ids",
            errors,
        )
        _check_equal(
            projection.get("ready_task_ids"),
            sorted(INITIAL_READY),
            "initial ready_task_ids",
            errors,
        )
        _check_equal(projection.get("blocked_task_ids"), [], "initial blocked_task_ids", errors)
        _check_equal(projection.get("terminal_task_id"), TERMINAL_TASK, "initial terminal_task_id", errors)
        _check_equal(projection.get("goal_count"), len(GOAL_IDS), "initial goal_count", errors)
        _check_equal(projection.get("root_goal_id"), "IPS-G000", "root_goal_id", errors)

    source = config.get("source_binding")
    if not isinstance(source, dict):
        errors.append("source_binding must be an object")
    else:
        expected_source = {
            "accelerator_required_ancestor": ACCELERATE_REVISION,
            "accelerator_required_branch": BRANCH,
            "accelerator_planning_revision": ACCELERATE_REVISION,
            "ipfs_datasets_submodule_path": "ipfs_datasets_py",
            "ipfs_datasets_planning_revision": DATASETS_REVISION,
            "ipfs_kit_submodule_path": "ipfs_kit_py",
            "ipfs_kit_planning_revision": KIT_REVISION,
            "require_initialized_gitlinks": True,
            "require_superproject_gitlink_equals_nested_head": True,
            "require_clean_nested_worktree_at_task_start": True,
            "changed_revision_requires_fresh_inventory_and_baseline": True,
            "planning_revision_is_runtime_completion_evidence": False,
        }
        for key, expected in expected_source.items():
            _check_equal(source.get(key), expected, f"source_binding.{key}", errors)

    actual_groups = config.get("task_groups")
    if not isinstance(actual_groups, dict):
        errors.append("task_groups must be an object")
    else:
        normalized_groups = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in actual_groups.items()
        }
        _check_equal(normalized_groups, dict(EXPECTED_TASK_GROUPS), "task_groups", errors)

    lanes = config.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != 3:
        errors.append("lanes must contain exactly three entries")
    else:
        seen_indices: set[int] = set()
        lane_tasks: set[str] = set()
        for lane in lanes:
            if not isinstance(lane, dict):
                errors.append("each lane must be an object")
                continue
            index = lane.get("index")
            remainder = lane.get("strict_shard_remainder")
            if not isinstance(index, int) or index not in range(3):
                errors.append(f"invalid lane index: {index!r}")
                continue
            if index in seen_indices:
                errors.append(f"duplicate lane index: {index}")
            seen_indices.add(index)
            _check_equal(remainder, index, f"lane {index} shard remainder", errors)
            initial = lane.get("initial_task_ids")
            if not isinstance(initial, list) or len(initial) != 1:
                errors.append(f"lane {index} must have exactly one initial task")
                continue
            task_id = initial[0]
            if task_id in lane_tasks:
                errors.append(f"initial lane task repeated: {task_id}")
            lane_tasks.add(task_id)
            shard = int(hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8], 16) % 3
            if shard != index:
                errors.append(
                    f"{task_id} hashes to strict shard {shard}, not configured lane {index}"
                )
        _check_equal(lane_tasks, set(INITIAL_READY), "lane initial task set", errors)


def _validate_tasks(text: str, config: dict[str, Any], errors: list[str]) -> dict[str, set[str]]:
    raw_headings = re.findall(r"^## (IPS-[^\s]+)(?:\s+.*)?$", text, re.MULTILINE)
    records = _parse_markdown_records(
        text,
        re.compile(r"^## (IPS-\d{3})\s+([^\n]+)$", re.MULTILINE),
        "task",
        errors,
    )
    actual_ids = set(records)
    expected_ids = set(TASK_IDS)
    if len(raw_headings) != len(records):
        errors.append(
            "one or more IPS task headings is duplicated, malformed, or lacks a title"
        )
    for task_id in sorted(expected_ids - actual_ids):
        errors.append(f"missing task heading: {task_id}")
    for task_id in sorted(actual_ids - expected_ids):
        errors.append(f"unexpected task heading: {task_id}")

    dependencies: dict[str, set[str]] = {}
    for task_id in TASK_IDS:
        record = records.get(task_id)
        if record is None:
            dependencies[task_id] = set()
            continue
        fields = record["fields"]
        for field in REQUIRED_TASK_FIELDS:
            if field not in fields:
                errors.append(f"{task_id} is missing metadata field {field!r}")
        for field in ("outputs", "validation", "conflict policy", "acceptance"):
            if not fields.get(field, "").strip():
                errors.append(f"{task_id} metadata field {field!r} may not be empty")

        validation = fields.get("validation", "")
        try:
            validation_argv = shlex.split(validation)
        except ValueError as exc:
            errors.append(f"{task_id} validation command does not parse: {exc}")
            validation_argv = []
        if validation_argv:
            executable = validation_argv[0].replace("\\", "/").rsplit("/", 1)[-1]
            if executable in {
                "bash",
                "cmd",
                "dash",
                "fish",
                "ksh",
                "powershell",
                "pwsh",
                "sh",
                "zsh",
            }:
                errors.append(f"{task_id} validation uses a forbidden shell")
            if (
                len(validation_argv) >= 2
                and executable in {"node", "perl", "python", "python3", "ruby"}
                and validation_argv[1] in {"-c", "-e", "--eval"}
            ):
                errors.append(f"{task_id} validation uses forbidden dynamic eval")

        predicted_files = fields.get("predicted files", "")
        predicted_submodules: list[str] = []
        if re.search(
            r"(?:^|[,\s])ipfs_datasets_py(?:/|[,\s]|$)", predicted_files
        ):
            predicted_submodules.append("ipfs_datasets_py")
        if re.search(r"(?:^|[,\s])ipfs_kit_py(?:/|[,\s]|$)", predicted_files):
            predicted_submodules.append("ipfs_kit_py")
        expected_submodules = ", ".join(predicted_submodules) or "none"
        _check_equal(
            fields.get("submodules"),
            expected_submodules,
            f"{task_id} submodules derived from Predicted files",
            errors,
        )

        status = fields.get("status", "").casefold()
        if status not in {"todo", "in_progress", "blocked", "completed"}:
            errors.append(f"{task_id} has invalid status {status!r}")
        schedulable = _as_bool(fields.get("is schedulable", ""))
        timeout = _as_int(fields.get("implementation timeout seconds", ""))
        if task_id == "IPS-000":
            if status != "completed":
                errors.append("IPS-000 must remain completed")
            if fields.get("completion", "").casefold() != "manual":
                errors.append("IPS-000 completion must be manual")
            # The daemon skips this already-completed bootstrap card, but its
            # parser still requires every card to carry a schedulable shape
            # and a positive timeout.
            if schedulable is not True:
                errors.append("IPS-000 must retain a schedulable card shape")
            if timeout is None or timeout <= 0:
                errors.append("IPS-000 must have a positive parser-safe timeout")
        else:
            if fields.get("completion", "").casefold() != "auto":
                errors.append(f"{task_id} completion must be auto")
            if schedulable is not True:
                errors.append(f"{task_id} must be schedulable")
            if timeout is None or timeout <= 0:
                errors.append(f"{task_id} must have a positive implementation timeout")

        _check_equal(
            fields.get("board namespace"),
            BOARD_NAMESPACE,
            f"{task_id} board namespace",
            errors,
        )
        _check_equal(
            fields.get("goal id"),
            EXPECTED_TASK_TO_GOAL.get(task_id),
            f"{task_id} goal id",
            errors,
        )

        dependency_ids = set(_ids(fields.get("depends on", ""), r"IPS-\d{3}"))
        dependency_text = fields.get("depends on", "").strip()
        if dependency_text and not re.fullmatch(
            r"IPS-\d{3}(?:\s*,\s*IPS-\d{3})*", dependency_text
        ):
            errors.append(f"{task_id} has malformed dependency metadata")
        if task_id in dependency_ids:
            errors.append(f"{task_id} depends on itself")
        unknown_dependencies = dependency_ids - expected_ids
        if unknown_dependencies:
            errors.append(
                f"{task_id} has unknown dependencies: {sorted(unknown_dependencies)}"
            )
        dependencies[task_id] = dependency_ids & expected_ids

        concurrent = set(
            _ids(fields.get("allow concurrent with", ""), r"IPS-\d{3}")
        )
        unknown_concurrent = concurrent - expected_ids
        if unknown_concurrent:
            errors.append(
                f"{task_id} has unknown concurrency peers: {sorted(unknown_concurrent)}"
            )
        if task_id in concurrent:
            errors.append(f"{task_id} lists itself as a concurrency peer")

    cycles = _cycle_nodes(dependencies)
    if cycles:
        errors.append(f"task dependency graph is cyclic: {sorted(cycles)}")

    initially_ready = {
        task_id
        for task_id, prerequisites in dependencies.items()
        if task_id not in INITIAL_COMPLETED
        and prerequisites.issubset(INITIAL_COMPLETED)
    }
    _check_equal(initially_ready, set(INITIAL_READY), "DAG initial ready set", errors)

    reachable = _reachable_from(INITIAL_COMPLETED, dependencies)
    if reachable != expected_ids:
        errors.append(
            "tasks unreachable from IPS-000: "
            f"{sorted(expected_ids - reachable)}"
        )
    terminal_ancestors = _ancestors(TERMINAL_TASK, dependencies)
    expected_ancestors = expected_ids - {TERMINAL_TASK}
    if terminal_ancestors != expected_ancestors:
        errors.append(
            "IPS-056 is not a true terminal fan-in; missing ancestors: "
            f"{sorted(expected_ancestors - terminal_ancestors)}"
        )

    projection = config.get("initial_projection", {})
    if isinstance(projection, dict):
        _check_equal(
            projection.get("terminal_task_id"), TERMINAL_TASK, "terminal task", errors
        )
    return dependencies


def _validate_goals(text: str, errors: list[str]) -> None:
    raw_headings = re.findall(r"^## (IPS-G[^\s]+)(?:\s+.*)?$", text, re.MULTILINE)
    records = _parse_markdown_records(
        text,
        re.compile(r"^## (IPS-G\d{3})\s+([^\n]+)$", re.MULTILINE),
        "goal",
        errors,
    )
    actual_ids = set(records)
    expected_ids = set(GOAL_IDS)
    if len(raw_headings) != len(records):
        errors.append(
            "one or more IPS goal headings is duplicated, malformed, or lacks a title"
        )
    for goal_id in sorted(expected_ids - actual_ids):
        errors.append(f"missing goal heading: {goal_id}")
    for goal_id in sorted(actual_ids - expected_ids):
        errors.append(f"unexpected goal heading: {goal_id}")

    dependencies: dict[str, set[str]] = {}
    for goal_id in GOAL_IDS:
        record = records.get(goal_id)
        if record is None:
            dependencies[goal_id] = set()
            continue
        fields = record["fields"]
        for field in REQUIRED_GOAL_FIELDS:
            if field not in fields:
                errors.append(f"{goal_id} is missing metadata field {field!r}")
        for field in ("goal", "evidence", "outputs", "validation", "acceptance", "conflict policy"):
            if not fields.get(field, "").strip():
                errors.append(f"{goal_id} metadata field {field!r} may not be empty")
        if fields.get("status", "").casefold() not in {
            "active",
            "provisionally_complete",
            "verified_complete",
            "analysis_inconclusive",
            "blocked",
            "reopened",
        }:
            errors.append(f"{goal_id} has an invalid status")
        parent = fields.get("parent", "").strip()
        if goal_id == "IPS-G000":
            if parent:
                errors.append("IPS-G000 must not have a parent")
        elif parent != "IPS-G000":
            errors.append(f"{goal_id} parent must be IPS-G000")
        dependency_ids = set(_ids(fields.get("depends on", ""), r"IPS-G\d{3}"))
        dependency_text = fields.get("depends on", "").strip()
        if dependency_text and not re.fullmatch(
            r"IPS-G\d{3}(?:\s*,\s*IPS-G\d{3})*", dependency_text
        ):
            errors.append(f"{goal_id} has malformed dependency metadata")
        if goal_id in dependency_ids:
            errors.append(f"{goal_id} depends on itself")
        unknown = dependency_ids - expected_ids
        if unknown:
            errors.append(f"{goal_id} has unknown dependencies: {sorted(unknown)}")
        dependencies[goal_id] = dependency_ids & expected_ids

        gap_tasks = set(_ids(fields.get("gap task", ""), r"IPS-\d{3}"))
        if goal_id == "IPS-G000":
            expected_gap_tasks = {TERMINAL_TASK}
        else:
            expected_gap_tasks = set(EXPECTED_TASK_GROUPS.get(goal_id, ()))
            expected_gap_tasks.discard("IPS-000")
        if gap_tasks != expected_gap_tasks:
            errors.append(
                f"{goal_id} gap task set must be {sorted(expected_gap_tasks)}; "
                f"got {sorted(gap_tasks)}"
            )

    cycles = _cycle_nodes(dependencies)
    if cycles:
        errors.append(f"goal dependency graph is cyclic: {sorted(cycles)}")


def _require_terms(
    text: str, terms: Iterable[str], category: str, errors: list[str]
) -> None:
    folded = text.casefold()
    whitespace_normalized = re.sub(r"\s+", " ", folded)
    missing = [
        term
        for term in terms
        if term.casefold() not in folded
        and re.sub(r"\s+", " ", term.casefold()) not in whitespace_normalized
    ]
    if missing:
        errors.append(f"plan is missing {category} terms: {missing}")


def _validate_plan(text: str, config: dict[str, Any], errors: list[str]) -> None:
    _require_terms(text, REQUIRED_PLAN_CONCEPTS, "architecture", errors)
    _require_terms(text, REQUIRED_CLI_TERMS, "CLI", errors)
    _require_terms(text, REQUIRED_INVALIDATION_TERMS, "invalidation", errors)
    _require_terms(text, REQUIRED_NEGATIVE_TERMS, "negative-test", errors)
    _require_terms(text, REQUIRED_CRASH_TERMS, "crash-recovery", errors)

    transition_rows = {
        int(match.group(1))
        for match in re.finditer(r"^\|\s*(\d{2})\s*\|", text, flags=re.MULTILINE)
    }
    expected_rows = set(range(40))
    if transition_rows != expected_rows:
        errors.append(
            "benchmark must define exactly transition rows 00..39; "
            f"missing={sorted(expected_rows - transition_rows)}, "
            f"extra={sorted(transition_rows - expected_rows)}"
        )
    benchmark = config.get("benchmark_policy")
    if not isinstance(benchmark, dict):
        errors.append("benchmark_policy must be an object")
    else:
        _check_equal(
            benchmark.get("sequential_commit_count"),
            40,
            "benchmark sequential_commit_count",
            errors,
        )
        _check_equal(
            benchmark.get("full_and_incremental_compared_per_commit"),
            True,
            "benchmark full/incremental comparison",
            errors,
        )
        _check_equal(
            benchmark.get("targets_are_not_reported_as_results"),
            True,
            "benchmark target honesty",
            errors,
        )


def _check_git_result(
    result: subprocess.CompletedProcess[str], description: str, errors: list[str]
) -> str:
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        errors.append(f"{description} failed: {detail}")
        return ""
    return result.stdout.strip()


def _validate_git_state(config: dict[str, Any], errors: list[str]) -> None:
    branch = _check_git_result(
        _git("branch", "--show-current"), "resolve control branch", errors
    )
    if branch and branch != BRANCH:
        errors.append(f"control branch must be {BRANCH!r}; got {branch!r}")
    ancestor = _git("merge-base", "--is-ancestor", ACCELERATE_REVISION, "HEAD")
    if ancestor.returncode != 0:
        errors.append(
            f"accelerate planning revision {ACCELERATE_REVISION} is not an ancestor of HEAD"
        )

    source = config.get("source_binding", {})
    if not isinstance(source, dict):
        return
    nested_specs = (
        ("ipfs_datasets_py", DATASETS_REVISION),
        ("ipfs_kit_py", KIT_REVISION),
    )
    for relative, planning_revision in nested_specs:
        nested = REPO_ROOT / relative
        if not nested.is_dir():
            errors.append(f"required initialized submodule is missing: {relative}")
            continue
        nested_head = _check_git_result(
            _git("rev-parse", "HEAD", cwd=nested),
            f"resolve {relative} HEAD",
            errors,
        )
        gitlink = _check_git_result(
            _git("rev-parse", f"HEAD:{relative}"),
            f"resolve superproject gitlink {relative}",
            errors,
        )
        if nested_head and gitlink and nested_head != gitlink:
            errors.append(
                f"{relative} nested HEAD {nested_head} does not equal gitlink {gitlink}"
            )
        nested_ancestor = _git(
            "merge-base", "--is-ancestor", planning_revision, "HEAD", cwd=nested
        )
        if nested_ancestor.returncode != 0:
            errors.append(
                f"{relative} planning revision {planning_revision} is not an ancestor of HEAD"
            )
        dirty = _check_git_result(
            _git("status", "--porcelain=v1", "--untracked-files=normal", cwd=nested),
            f"inspect {relative} worktree",
            errors,
        )
        if dirty:
            errors.append(f"{relative} nested worktree is dirty: {dirty.splitlines()[:8]}")

    for relative in CONTROL_PATHS:
        tracked = _git("ls-files", "--error-unmatch", "--", relative)
        if tracked.returncode != 0:
            errors.append(f"control file is not tracked: {relative}")
    status = _check_git_result(
        _git("status", "--porcelain=v1", "--", *CONTROL_PATHS),
        "inspect control-file cleanliness",
        errors,
    )
    if status:
        errors.append(f"control files are dirty: {status.splitlines()}")


def validate(*, check_all: bool) -> dict[str, Any]:
    errors: list[str] = []
    config = _load_json(CONFIG_PATH, errors)
    task_text = _read(TASKBOARD_PATH, errors)
    goal_text = _read(OBJECTIVES_PATH, errors)
    plan_text = _read(PLAN_PATH, errors)

    _validate_config(config, errors)
    dependencies = _validate_tasks(task_text, config, errors)
    _validate_goals(goal_text, errors)
    _validate_plan(plan_text, config, errors)
    if check_all:
        _validate_git_state(config, errors)

    edge_count = sum(len(value) for value in dependencies.values())
    return {
        "valid": not errors,
        "check_all": check_all,
        "errors": errors,
        "counts": {
            "tasks_expected": len(TASK_IDS),
            "task_dependency_edges": edge_count,
            "goals_expected": len(GOAL_IDS),
            "strict_lanes": 3,
            "benchmark_transitions": 40,
            "errors": len(errors),
        },
        "source_binding": {
            "accelerate": ACCELERATE_REVISION,
            "ipfs_datasets_py": DATASETS_REVISION,
            "ipfs_kit_py": KIT_REVISION,
        },
    }


def _artifact_json(relative: str, errors: list[str]) -> Mapping[str, Any]:
    """Load one task-owned JSON artifact without importing project code."""

    payload = _load_json(REPO_ROOT / relative, errors)
    if not isinstance(payload, Mapping):
        errors.append(f"artifact must contain a JSON object: {relative}")
        return {}
    return payload


def _require_nonempty_file(relative: str, errors: list[str]) -> str:
    path = REPO_ROOT / relative
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        errors.append(f"cannot read artifact {relative}: {type(exc).__name__}")
        return ""
    if not text.strip():
        errors.append(f"artifact is empty: {relative}")
    return text


def validate_artifact(task_id: str) -> dict[str, Any]:
    """Validate the six data/document tasks that cannot use inline eval.

    The implementation supervisor deliberately rejects ``python -c`` and
    other dynamic-eval validation commands.  These bounded, standard-library
    checks retain the reviewed assertions as a normal executable entry point.
    """

    task_id = str(task_id or "").strip().upper()
    errors: list[str] = []
    inventory_specs = {
        "IPS-001": (
            "docs/architecture/incremental_proof_sealer_inventory/accelerate.json",
            ACCELERATE_REVISION,
        ),
        "IPS-002": (
            "ipfs_datasets_py/docs/architecture/incremental_proof_sealer_inventory.json",
            DATASETS_REVISION,
        ),
        "IPS-003": (
            "ipfs_kit_py/docs/architecture/incremental_proof_sealer_inventory.json",
            KIT_REVISION,
        ),
    }
    if task_id in inventory_specs:
        relative, revision = inventory_specs[task_id]
        payload = _artifact_json(relative, errors)
        serialized = json.dumps(payload, sort_keys=True)
        if payload.get("repository_commit") != revision:
            errors.append(f"{task_id} repository_commit does not match {revision}")
        baseline = payload.get("baseline")
        if not isinstance(baseline, Mapping):
            errors.append(f"{task_id} baseline must be an object")
        else:
            exit_code = baseline.get("exit_code")
            if isinstance(exit_code, bool) or not isinstance(exit_code, int):
                errors.append(f"{task_id} baseline.exit_code must be an integer")
            if not baseline.get("command"):
                errors.append(f"{task_id} baseline.command must be non-empty")
            if baseline.get("results_populated") is False:
                errors.append(f"{task_id} baseline results are still a placeholder")
            if "pending-local-run" in serialized.casefold():
                errors.append(f"{task_id} baseline contains pending-local-run")
            outcome_fields = ("passed", "failed", "errors", "skipped", "deselected")
            outcomes = [baseline.get(field) for field in outcome_fields]
            if all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in outcomes
            ) and sum(outcomes) == 0:
                errors.append(f"{task_id} baseline contains a zero-count success")
        classifications = payload.get("classifications")
        if not isinstance(classifications, (list, dict)) or not classifications:
            errors.append(f"{task_id} classifications must be non-empty")
        required_terms = {
            "IPS-001": (
                "proof_attestation",
                "proof_reuse_real_groth16_fixture",
                "kernel_verification.py",
                "prover_conformance.py",
                "proof_fallbacks.py",
                "proof_metrics.py",
                "prover_evidence_store.py",
                "manual_completion_seal.py",
                "release_evidence.py",
                "repository_forest",
            ),
            "IPS-002": (
                "cec_zkp_integration.py",
                "cec_proof_cache.py",
                "tdfol_zkp_integration.py",
                "tdfol_proof_cache.py",
                "flogic_zkp_integration.py",
                "flogic_proof_cache.py",
                "event_dag_zkp.py",
                "provekit_ffi.py",
                "wallet/proofs.py",
                "test_execution_certificate.py",
                "test_pass.py",
                "proof_receipt_attestation.py",
                "ensure_setup",
            ),
            "IPS-003": (
                "profile_d_policy.py",
                "mcplusplus/artifacts.py",
                "iroh/release.py",
                "test_joined_release_receipt.py",
                "install_lotus.py",
                "proof_certificate_store.py",
                "event_dag.py",
            ),
        }
        for term in required_terms[task_id]:
            if term not in serialized:
                errors.append(f"{task_id} inventory is missing required surface {term}")
    elif task_id == "IPS-053":
        payload = _artifact_json(
            "artifacts/agent_supervisor/incremental_proof_sealer/benchmark.json",
            errors,
        )
        if not payload.get("schema_version"):
            errors.append("IPS-053 schema_version must be non-empty")
        transitions = payload.get("transitions")
        if not isinstance(transitions, list) or len(transitions) != 40:
            errors.append("IPS-053 transitions must contain exactly 40 rows")
        elif any(
            not isinstance(row, Mapping)
            or row.get("measurement_provenance")
            not in {"measured", "estimated", "mixed"}
            for row in transitions
        ):
            errors.append("IPS-053 transition provenance is incomplete or invalid")
        if not payload.get("source_revisions"):
            errors.append("IPS-053 source_revisions must be non-empty")
        _require_nonempty_file(
            "artifacts/agent_supervisor/incremental_proof_sealer/benchmark.csv",
            errors,
        )
    elif task_id == "IPS-054":
        payload = _artifact_json(
            "artifacts/agent_supervisor/incremental_proof_sealer/summary.json",
            errors,
        )
        if payload.get("transition_count") != 40:
            errors.append("IPS-054 transition_count must equal 40")
        for field in ("average_reuse_rate", "average_compute_reduction"):
            if field not in payload:
                errors.append(f"IPS-054 is missing {field}")
        for field in ("best_case", "worst_case", "target_assessment"):
            if not payload.get(field):
                errors.append(f"IPS-054 {field} must be non-empty")
        _require_nonempty_file(
            "docs/architecture/INCREMENTAL_PROOF_SEALER_BENCHMARK.md",
            errors,
        )
    elif task_id == "IPS-055":
        trust = _require_nonempty_file(
            "docs/architecture/INCREMENTAL_PROOF_SEALER_TRUST_MODEL.md",
            errors,
        )
        migration = _require_nonempty_file(
            "docs/architecture/INCREMENTAL_PROOF_SEALER_MIGRATION.md",
            errors,
        )
        for term in (
            "Integrity commitment",
            "Signed execution receipt",
            "Receipt-aggregation ZK proof",
            "Direct execution proof",
            "Incremental or recursive commit seal",
        ):
            if term not in trust:
                errors.append(f"IPS-055 trust model is missing {term!r}")
        for term in ("accept", "reverify", "reject", "simulated"):
            if term not in migration:
                errors.append(f"IPS-055 migration guide is missing {term!r}")
    else:
        errors.append(f"unsupported artifact check task: {task_id}")

    return {
        "valid": not errors,
        "check_artifact": task_id,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the IncrementalProofSealer fixed supervisor board"
    )
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="also verify bound Git ancestry, gitlinks, and clean tracked controls",
    )
    parser.add_argument(
        "--check-artifact",
        choices=ARTIFACT_CHECK_TASKS,
        help="run a bounded non-eval validation for one data/document task",
    )
    args = parser.parse_args(argv)
    if args.check_all and args.check_artifact:
        parser.error("--check-all and --check-artifact are mutually exclusive")
    result = (
        validate_artifact(args.check_artifact)
        if args.check_artifact
        else validate(check_all=args.check_all)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
