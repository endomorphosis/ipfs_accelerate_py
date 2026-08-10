#!/usr/bin/env python3
"""Fail-closed semantic validator for the IPFS Kit kernel-VFS board."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = REPO_ROOT / "docs/architecture/IPFS_KIT_FUSE_VFS_PLAN.md"
OBJECTIVE_PATH = REPO_ROOT / "docs/architecture/ipfs_kit_fuse_vfs.objectives.md"
TODO_PATH = REPO_ROOT / "docs/architecture/ipfs_kit_fuse_vfs.todo.md"
CONFIG_PATH = REPO_ROOT / "config/agent_supervisor_ipfs_kit_fuse_vfs_scheduler.json"

NAMESPACE = "ipfs-kit-kernel-vfs-fuse-v1"
BRANCH = "agent/ipfs-kit-fuse-vfs"
ACCELERATOR_ANCESTOR = "ea11293bb996f052d620eae989f5377a956764b1"
IPFS_KIT_REVISION = "69091bf8f11a3ef1fb0e04e11a6d8a4c87f3fa78"

TASK_IDS = (
    "KVFS-000",
    "KVFS-100", "KVFS-101", "KVFS-103", "KVFS-108",
    "KVFS-200", "KVFS-202", "KVFS-201", "KVFS-204", "KVFS-203",
    "KVFS-205", "KVFS-208", "KVFS-210", "KVFS-206",
    "KVFS-303", "KVFS-309", "KVFS-300", "KVFS-301", "KVFS-304",
    "KVFS-400", "KVFS-401", "KVFS-404", "KVFS-403",
    "KVFS-503", "KVFS-500", "KVFS-506", "KVFS-501",
    "KVFS-608", "KVFS-600", "KVFS-601", "KVFS-603",
    "KVFS-703", "KVFS-701", "KVFS-700", "KVFS-702",
    "KVFS-808", "KVFS-800", "KVFS-802", "KVFS-801", "KVFS-811",
)
GOAL_IDS = (
    "KVFS-G000", "KVFS-G100", "KVFS-G200", "KVFS-G300",
    "KVFS-G400", "KVFS-G500", "KVFS-G600", "KVFS-G700", "KVFS-G800",
)
INITIAL_COMPLETED = ("KVFS-000",)
INITIAL_READY = ("KVFS-100", "KVFS-101", "KVFS-103", "KVFS-108")
INITIAL_SHARDS = {
    0: "KVFS-103",
    1: "KVFS-101",
    2: "KVFS-108",
    3: "KVFS-100",
}
TERMINAL_TASK = "KVFS-811"
RETRY_BUDGET_REPAIR_SCHEMA = (
    "ipfs_accelerate_py.agent_supervisor.retry-budget-repair@1"
)
MAX_OPERATIONAL_REPAIR_TASKS = 16
FIRST_OPERATIONAL_REPAIR_NUMBER = max(
    int(task_id.rsplit("-", 1)[1]) for task_id in TASK_IDS
) + 1
PERSISTED_OPERATIONAL_STATES = frozenset({"todo", "completed"})

RETRY_BUDGET_REPAIR_TITLE_RE = re.compile(
    r"^Resolve\s+(?P<kind>validation|implementation|merge)\s+"
    r"retry-budget\s+failure\s+for\s+(?P<source>KVFS-\d{3})$",
    re.IGNORECASE,
)
RETRY_BUDGET_REPAIR_ACCEPTANCE_RE = re.compile(
    r"\b(?:release|remove)\s+(?P<source>KVFS-\d{3})\s+from\s+"
    r"(?:the\s+)?strategy\s+blocked_tasks\b",
    re.IGNORECASE,
)

TASK_DEPENDENCIES = {
    "KVFS-000": (),
    "KVFS-100": ("KVFS-000",),
    "KVFS-101": ("KVFS-000",),
    "KVFS-103": ("KVFS-000",),
    "KVFS-108": ("KVFS-000",),
    "KVFS-200": ("KVFS-100", "KVFS-101"),
    "KVFS-202": ("KVFS-100", "KVFS-101"),
    "KVFS-201": ("KVFS-101", "KVFS-103"),
    "KVFS-204": ("KVFS-101", "KVFS-103"),
    "KVFS-203": ("KVFS-200", "KVFS-201", "KVFS-202", "KVFS-204"),
    "KVFS-205": ("KVFS-200", "KVFS-204"),
    "KVFS-208": ("KVFS-202", "KVFS-204"),
    "KVFS-210": ("KVFS-203",),
    "KVFS-206": ("KVFS-203", "KVFS-205", "KVFS-208", "KVFS-210"),
    "KVFS-303": ("KVFS-100", "KVFS-101"),
    "KVFS-309": ("KVFS-203", "KVFS-205", "KVFS-303"),
    "KVFS-300": ("KVFS-309",),
    "KVFS-301": ("KVFS-208", "KVFS-300", "KVFS-309"),
    "KVFS-304": ("KVFS-301",),
    "KVFS-400": ("KVFS-101", "KVFS-103", "KVFS-200"),
    "KVFS-401": ("KVFS-203", "KVFS-400"),
    "KVFS-404": ("KVFS-301", "KVFS-309", "KVFS-401"),
    "KVFS-403": ("KVFS-304", "KVFS-400", "KVFS-404"),
    "KVFS-503": ("KVFS-100", "KVFS-108"),
    "KVFS-500": ("KVFS-206", "KVFS-300", "KVFS-404", "KVFS-503"),
    "KVFS-506": ("KVFS-301", "KVFS-304", "KVFS-403", "KVFS-500"),
    "KVFS-501": ("KVFS-506",),
    "KVFS-608": ("KVFS-100", "KVFS-108", "KVFS-503"),
    "KVFS-600": ("KVFS-201", "KVFS-202", "KVFS-608"),
    "KVFS-601": ("KVFS-206", "KVFS-300", "KVFS-301", "KVFS-404", "KVFS-600"),
    "KVFS-603": ("KVFS-403", "KVFS-601"),
    "KVFS-703": ("KVFS-503", "KVFS-608"),
    "KVFS-701": ("KVFS-500", "KVFS-703"),
    "KVFS-700": ("KVFS-506", "KVFS-701"),
    "KVFS-702": ("KVFS-500", "KVFS-601", "KVFS-703"),
    "KVFS-808": ("KVFS-500", "KVFS-601"),
    "KVFS-800": ("KVFS-206", "KVFS-301", "KVFS-403", "KVFS-600"),
    "KVFS-802": ("KVFS-506", "KVFS-603", "KVFS-700", "KVFS-800"),
    "KVFS-801": ("KVFS-403", "KVFS-506", "KVFS-603", "KVFS-700"),
    "KVFS-811": (
        "KVFS-501", "KVFS-603", "KVFS-700", "KVFS-702", "KVFS-808",
        "KVFS-802", "KVFS-800", "KVFS-801",
    ),
}

TASK_GROUPS = {
    "KVFS-G100": ("KVFS-100", "KVFS-101", "KVFS-103", "KVFS-108"),
    "KVFS-G200": (
        "KVFS-200", "KVFS-202", "KVFS-201", "KVFS-204", "KVFS-203",
        "KVFS-205", "KVFS-208", "KVFS-210", "KVFS-206",
    ),
    "KVFS-G300": ("KVFS-303", "KVFS-309", "KVFS-300", "KVFS-301", "KVFS-304"),
    "KVFS-G400": ("KVFS-400", "KVFS-401", "KVFS-404", "KVFS-403"),
    "KVFS-G500": ("KVFS-503", "KVFS-500", "KVFS-506", "KVFS-501"),
    "KVFS-G600": ("KVFS-608", "KVFS-600", "KVFS-601", "KVFS-603"),
    "KVFS-G700": ("KVFS-703", "KVFS-701", "KVFS-700", "KVFS-702"),
    "KVFS-G800": ("KVFS-808", "KVFS-800", "KVFS-802", "KVFS-801", "KVFS-811"),
}

GOAL_DEPENDENCIES = {
    "KVFS-G000": (),
    "KVFS-G100": (),
    "KVFS-G200": ("KVFS-G100",),
    "KVFS-G300": ("KVFS-G100",),
    "KVFS-G400": ("KVFS-G100",),
    "KVFS-G500": ("KVFS-G200", "KVFS-G300", "KVFS-G400"),
    "KVFS-G600": ("KVFS-G200", "KVFS-G300", "KVFS-G400"),
    "KVFS-G700": ("KVFS-G500", "KVFS-G600"),
    "KVFS-G800": ("KVFS-G500", "KVFS-G600", "KVFS-G700"),
}

REQUIRED_TASK_FIELDS = (
    "status", "completion", "is schedulable", "review only", "priority",
    "track", "depends on", "goal id", "board namespace", "outputs",
    "validation", "scope paths", "conflict policy", "acceptance",
)
REQUIRED_GOAL_FIELDS = (
    "status", "parent", "depends on", "fib priority", "track", "priority",
    "bundle", "goal", "evidence", "outputs", "validation", "acceptance",
    "gap task", "refinement", "conflict policy",
)
PROTECTED_PATHS = (
    ".gitignore",
    "docs/architecture/IPFS_KIT_FUSE_VFS_PLAN.md",
    "docs/architecture/ipfs_kit_fuse_vfs.objectives.md",
    "docs/architecture/ipfs_kit_fuse_vfs.todo.md",
    "config/agent_supervisor_ipfs_kit_fuse_vfs_scheduler.json",
    "scripts/validate_ipfs_kit_fuse_vfs_board.py",
    "test/api/test_ipfs_kit_fuse_vfs_board.py",
)


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in re.split(r"[,;]", value) if item.strip())


def _blocks(path: Path, pattern: re.Pattern[str]) -> list[tuple[str, str, dict[str, str]]]:
    text = path.read_text(encoding="utf-8")
    matches = list(pattern.finditer(text))
    records: list[tuple[str, str, dict[str, str]]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields: dict[str, str] = {}
        prior = ""
        for line in text[match.end():end].splitlines():
            field = re.match(r"^- ([^:]+):\s*(.*)$", line)
            if field:
                prior = field.group(1).strip().lower()
                if prior in fields:
                    raise ValueError(f"{match.group(1)} duplicates field {prior!r}")
                fields[prior] = field.group(2).strip()
            elif prior and line.startswith(("  ", "\t")) and line.strip():
                fields[prior] = f"{fields[prior]} {line.strip()}".strip()
        records.append((match.group(1), match.group(2).strip(), fields))
    return records


def parse_tasks() -> list[tuple[str, str, dict[str, str]]]:
    return _blocks(TODO_PATH, re.compile(r"^## (KVFS-\d{3}) (.+)$", re.MULTILINE))


def parse_goals() -> list[tuple[str, str, dict[str, str]]]:
    return _blocks(OBJECTIVE_PATH, re.compile(r"^## (KVFS-G\d{3}) (.+)$", re.MULTILINE))


def _safe_relative(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts and "\x00" not in value


def _path_is_within_scope(value: str, scope: str) -> bool:
    """Return whether a safe relative path is owned by a declared scope path."""

    value_path = PurePosixPath(value)
    scope_path = PurePosixPath(scope)
    return value_path == scope_path or value_path.parts[: len(scope_path.parts)] == scope_path.parts


def _partition_canonical_and_operational_tasks(
    tasks: Iterable[tuple[str, str, dict[str, str]]],
    errors: list[str],
) -> tuple[
    list[tuple[str, str, dict[str, str]]],
    list[tuple[str, str, dict[str, str]]],
]:
    """Keep the sealed program separate from generated recovery appendices."""

    canonical: list[tuple[str, str, dict[str, str]]] = []
    operational: list[tuple[str, str, dict[str, str]]] = []
    appendix_started = False
    all_ids: list[str] = []
    for task in tasks:
        task_id = task[0]
        all_ids.append(task_id)
        if task_id in TASK_IDS:
            if appendix_started:
                errors.append(
                    f"canonical task {task_id} appears after the operational appendix"
                )
            canonical.append(task)
            continue
        appendix_started = True
        operational.append(task)
    if len(all_ids) != len(set(all_ids)):
        errors.append("task IDs are not unique")
    return canonical, operational


def _validate_operational_repair_tasks(
    repairs: Iterable[tuple[str, str, dict[str, str]]],
    *,
    canonical_by_id: Mapping[str, dict[str, str]],
    errors: list[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate bounded retry repairs without adding aliases to the sealed DAG."""

    repair_list = list(repairs)
    if len(repair_list) > MAX_OPERATIONAL_REPAIR_TASKS:
        errors.append("operational retry-repair appendix exceeds its finite bound")

    operational_ids: list[str] = []
    pending_ids: list[str] = []
    active_source_kinds: set[tuple[str, str]] = set()
    discovery_root = (
        REPO_ROOT
        / "data/agent_supervisor/ipfs_kit_fuse_vfs/state/discovery"
    ).resolve()
    for offset, (task_id, title, fields) in enumerate(repair_list):
        operational_ids.append(task_id)
        expected_id = f"KVFS-{FIRST_OPERATIONAL_REPAIR_NUMBER + offset:03d}"
        if task_id != expected_id:
            errors.append(
                "operational retry-repair IDs must be contiguous: "
                f"expected {expected_id}, got {task_id}"
            )

        title_match = RETRY_BUDGET_REPAIR_TITLE_RE.fullmatch(title)
        acceptance = fields.get("acceptance", "")
        acceptance_match = RETRY_BUDGET_REPAIR_ACCEPTANCE_RE.search(acceptance)
        source_task_id = fields.get("retry repair source", "")
        failure_kind = fields.get("retry failure kind", "").lower()
        source = canonical_by_id.get(source_task_id)
        if (
            title_match is None
            or acceptance_match is None
            or title_match.group("source").upper() != source_task_id
            or acceptance_match.group("source").upper() != source_task_id
            or title_match.group("kind").lower() != failure_kind
            or source is None
            or failure_kind not in {"validation", "implementation", "merge"}
        ):
            errors.append(f"{task_id} is not a recognized retry repair")
            continue

        if (
            fields.get("generated by") != RETRY_BUDGET_REPAIR_SCHEMA
            or fields.get("canonical board task") != "false"
        ):
            errors.append(f"{task_id} lacks exact operational provenance")
        if fields.get("completion") != "manual":
            errors.append(f"{task_id} must use manual completion")
        if fields.get("priority") != "P1":
            errors.append(f"{task_id} must use P1 priority")
        if fields.get("track") != "ops":
            errors.append(f"{task_id} must use the ops track")

        status = fields.get("status", "")
        if status not in PERSISTED_OPERATIONAL_STATES:
            errors.append(f"{task_id} has invalid operational status {status!r}")
        if status != "completed":
            pending_ids.append(task_id)
            if source.get("status") == "completed":
                errors.append(
                    f"{task_id} is pending after source {source_task_id} completed"
                )

        dependencies = _csv(fields.get("depends on", ""))
        source_dependencies = _csv(source.get("depends on", ""))
        if dependencies != source_dependencies:
            errors.append(
                f"{task_id} dependency scope differs from {source_task_id}"
            )

        repair_outputs = _csv(fields.get("outputs", ""))
        source_outputs = _csv(source.get("outputs", ""))
        if repair_outputs != source_outputs:
            errors.append(
                f"{task_id} output scope differs from source {source_task_id}"
            )
        source_scope = _csv(source.get("scope paths", ""))
        declared_scope = _csv(fields.get("scope paths", ""))
        effective_scope = declared_scope or source_scope
        if declared_scope and declared_scope != source_scope:
            errors.append(
                f"{task_id} declared scope differs from source {source_task_id}"
            )
        if not source_scope or any(
            not any(_path_is_within_scope(output, scope) for scope in effective_scope)
            for output in repair_outputs
        ):
            errors.append(
                f"{task_id} outputs escape source {source_task_id} scope paths"
            )
        for field in ("parallel lane", "conflict policy"):
            if fields.get(field, "") != source.get(field, ""):
                errors.append(
                    f"{task_id} {field} differs from source {source_task_id}"
                )

        discovery_text = fields.get("retry repair discovery", "")
        discovery_path = Path(discovery_text)
        expected_suffix = {
            "validation": "retry-budget",
            "implementation": "implementation-retry-budget",
            "merge": "merge-retry-budget",
        }[failure_kind]
        expected_discovery_name = re.compile(
            rf"\d{{4}}-\d{{2}}-\d{{2}}-{task_id.lower()}-"
            rf"{source_task_id.lower()}-{expected_suffix}\.md"
        )
        if (
            not discovery_text
            or not discovery_path.is_absolute()
            or discovery_path.parent.resolve() != discovery_root
            or expected_discovery_name.fullmatch(discovery_path.name) is None
            or discovery_text not in acceptance
        ):
            errors.append(f"{task_id} has invalid retry discovery provenance")
        if discovery_text in repair_outputs:
            errors.append(
                f"{task_id} grants write authority to discovery evidence"
            )

        failure_paths = _csv(fields.get("validation failure paths", ""))
        failure_path_authority = fields.get(
            "validation failure path authority", ""
        )
        if failure_paths and failure_path_authority != "diagnostic-read-only":
            errors.append(
                f"{task_id} validation failure paths are not diagnostic-read-only"
            )
        if failure_path_authority and not failure_paths:
            errors.append(
                f"{task_id} declares validation failure authority without paths"
            )
        for path in failure_paths:
            if not _safe_relative(path):
                errors.append(
                    f"{task_id} has unsafe validation failure path {path!r}"
                )
        if not fields.get("validation", ""):
            errors.append(f"{task_id} has no validation command")
        if not acceptance:
            errors.append(f"{task_id} has empty acceptance")

        source_kind = (source_task_id, failure_kind)
        if status != "completed" and source_kind in active_source_kinds:
            errors.append(
                f"{task_id} duplicates an active repair for {source_kind}"
            )
        if status != "completed":
            active_source_kinds.add(source_kind)

    return tuple(operational_ids), tuple(pending_ids)


def _acyclic(nodes: Iterable[str], dependencies: Mapping[str, Iterable[str]], errors: list[str], label: str) -> None:
    state: dict[str, int] = {}
    trail: list[str] = []

    def visit(node: str) -> None:
        if state.get(node) == 2:
            return
        if state.get(node) == 1:
            start = trail.index(node) if node in trail else 0
            errors.append(f"{label} dependency cycle: {' -> '.join([*trail[start:], node])}")
            return
        state[node] = 1
        trail.append(node)
        for dependency in dependencies.get(node, ()):
            if dependency in state or dependency in dependencies:
                visit(dependency)
        trail.pop()
        state[node] = 2

    for node in nodes:
        visit(node)


def _shard(task_id: str, lanes: int = 4) -> int:
    return int(hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8], 16) % lanes


def _load_config(errors: list[str]) -> dict[str, object]:
    try:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"scheduler config unreadable: {type(exc).__name__}: {exc}")
        return {}
    if not isinstance(payload, dict):
        errors.append("scheduler config root is not an object")
        return {}
    return payload


def validate() -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    for path in (PLAN_PATH, OBJECTIVE_PATH, TODO_PATH, CONFIG_PATH):
        if not path.is_file():
            errors.append(f"missing control file: {path.relative_to(REPO_ROOT)}")

    try:
        parsed_tasks = parse_tasks() if TODO_PATH.is_file() else []
        goals = parse_goals() if OBJECTIVE_PATH.is_file() else []
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        errors.append(f"control markdown parse failed: {type(exc).__name__}: {exc}")
        parsed_tasks, goals = [], []

    tasks, operational_repairs = _partition_canonical_and_operational_tasks(
        parsed_tasks,
        errors,
    )
    task_ids = tuple(item[0] for item in tasks)
    goal_ids = tuple(item[0] for item in goals)
    if task_ids != TASK_IDS:
        errors.append(f"task IDs/order differ: expected {list(TASK_IDS)}, got {list(task_ids)}")
    if goal_ids != GOAL_IDS:
        errors.append(f"goal IDs/order differ: expected {list(GOAL_IDS)}, got {list(goal_ids)}")
    if len(set(goal_ids)) != len(goal_ids):
        errors.append("goal IDs are not unique")

    task_by_id = {task_id: fields for task_id, _title, fields in tasks}
    task_dependencies: dict[str, tuple[str, ...]] = {}
    output_owners: dict[str, list[str]] = {}
    for task_id, _title, fields in tasks:
        missing = [field for field in REQUIRED_TASK_FIELDS if field not in fields]
        if missing:
            errors.append(f"{task_id} missing fields: {missing}")
        status = fields.get("status", "")
        if status not in {"todo", "in_progress", "completed", "blocked"}:
            errors.append(f"{task_id} has invalid status {status!r}")
        if fields.get("completion") not in {"auto", "manual"}:
            errors.append(f"{task_id} has invalid completion")
        if fields.get("is schedulable") not in {"true", "false"}:
            errors.append(f"{task_id} has invalid is schedulable")
        if fields.get("review only") not in {"true", "false"}:
            errors.append(f"{task_id} has invalid review only")
        if fields.get("board namespace") != NAMESPACE:
            errors.append(f"{task_id} board namespace mismatch")
        dependencies = _csv(fields.get("depends on", ""))
        task_dependencies[task_id] = dependencies
        expected_dependencies = TASK_DEPENDENCIES.get(task_id)
        if expected_dependencies is not None and dependencies != expected_dependencies:
            errors.append(
                f"{task_id} dependency mismatch: expected {list(expected_dependencies)}, got {list(dependencies)}"
            )
        for dependency in dependencies:
            if dependency not in TASK_IDS:
                errors.append(f"{task_id} references unknown dependency {dependency}")
        goal_id = fields.get("goal id", "")
        expected_goal = "KVFS-G000" if task_id == "KVFS-000" else next(
            (candidate for candidate, members in TASK_GROUPS.items() if task_id in members), ""
        )
        if goal_id != expected_goal:
            errors.append(f"{task_id} goal mismatch: expected {expected_goal}, got {goal_id}")
        outputs = _csv(fields.get("outputs", ""))
        if not outputs:
            errors.append(f"{task_id} has no outputs")
        scope_paths = _csv(fields.get("scope paths", ""))
        if not scope_paths:
            errors.append(f"{task_id} has no scope paths")
        for scope_path in scope_paths:
            if not _safe_relative(scope_path):
                errors.append(f"{task_id} has unsafe scope path {scope_path!r}")
        for output in outputs:
            if not _safe_relative(output):
                errors.append(f"{task_id} has unsafe output {output!r}")
            if task_id != "KVFS-000" and output in PROTECTED_PATHS:
                errors.append(f"{task_id} writes protected control path {output}")
            if scope_paths and not any(
                _path_is_within_scope(output, scope_path)
                for scope_path in scope_paths
            ):
                errors.append(f"{task_id} output escapes its scope paths: {output}")
            output_owners.setdefault(output, []).append(task_id)
        if task_id in INITIAL_READY:
            native_text = " ".join((fields.get("outputs", ""), fields.get("validation", ""))).lower()
            if any(term in native_text for term in ("live_mount", "live_winfsp", "live_container")):
                errors.append(f"initial task {task_id} requires a native live harness")

    for output, owners in sorted(output_owners.items()):
        if len(owners) > 1:
            errors.append(f"output has multiple owners: {output}: {owners}")
    if set(task_dependencies) == set(TASK_IDS):
        _acyclic(TASK_IDS, task_dependencies, errors, "task")

    operational_task_ids, pending_operational_task_ids = (
        _validate_operational_repair_tasks(
            operational_repairs,
            canonical_by_id=task_by_id,
            errors=errors,
        )
    )

    completed = {task_id for task_id, fields in task_by_id.items() if fields.get("status") == "completed"}
    blocked = {task_id for task_id, fields in task_by_id.items() if fields.get("status") == "blocked"}
    missing_initial_completion = sorted(
        set(INITIAL_COMPLETED) - completed
    )
    if missing_initial_completion:
        errors.append(
            "sealed initial completions regressed: "
            f"{missing_initial_completion}"
        )
    incomplete_dependencies = {
        task_id: sorted(
            dependency
            for dependency in task_dependencies.get(task_id, ())
            if dependency not in completed
        )
        for task_id in sorted(completed)
        if any(
            dependency not in completed
            for dependency in task_dependencies.get(task_id, ())
        )
    }
    if incomplete_dependencies:
        errors.append(
            "completed canonical tasks are not dependency-closed: "
            + json.dumps(incomplete_dependencies, sort_keys=True)
        )
    ready = tuple(
        task_id for task_id in TASK_IDS
        if task_by_id.get(task_id, {}).get("status") == "todo"
        and all(dependency in completed for dependency in task_dependencies.get(task_id, ()))
    )
    initial_shards = {_shard(task_id): task_id for task_id in INITIAL_READY}
    if initial_shards != INITIAL_SHARDS or len(initial_shards) != len(INITIAL_READY):
        errors.append(f"sealed initial strict shard coverage differs: {initial_shards}")
    current_ready_shards: dict[str, list[str]] = {
        str(index): [] for index in range(4)
    }
    for task_id in ready:
        current_ready_shards[str(_shard(task_id))].append(task_id)

    goal_dependencies: dict[str, tuple[str, ...]] = {}
    for goal_id, _title, fields in goals:
        missing = [field for field in REQUIRED_GOAL_FIELDS if field not in fields]
        if missing:
            errors.append(f"{goal_id} missing fields: {missing}")
        if fields.get("status") not in {
            "active", "provisionally_complete", "verified_complete",
            "analysis_inconclusive", "blocked", "reopened",
        }:
            errors.append(f"{goal_id} has invalid status {fields.get('status')!r}")
        parent = fields.get("parent", "")
        expected_parent = "" if goal_id == "KVFS-G000" else "KVFS-G000"
        if parent != expected_parent:
            errors.append(f"{goal_id} parent mismatch: expected {expected_parent!r}, got {parent!r}")
        dependencies = _csv(fields.get("depends on", ""))
        goal_dependencies[goal_id] = dependencies
        if dependencies != GOAL_DEPENDENCIES.get(goal_id, ()):
            errors.append(f"{goal_id} dependency mismatch: {list(dependencies)}")
        for reference in (*dependencies, *((parent,) if parent else ())):
            if reference not in GOAL_IDS:
                errors.append(f"{goal_id} references unknown goal {reference}")
        evidence = _csv(fields.get("evidence", ""))
        for reference in evidence:
            if reference not in TASK_IDS and reference not in GOAL_IDS:
                errors.append(f"{goal_id} references unknown evidence {reference}")
    if set(goal_dependencies) == set(GOAL_IDS):
        _acyclic(GOAL_IDS, goal_dependencies, errors, "goal")

    config = _load_config(errors)
    exact_config = {
        "schema": "ipfs_accelerate_py.agent_supervisor.ipfs_kit_fuse_vfs.scheduler_config@1",
        "taskboard_path": "docs/architecture/ipfs_kit_fuse_vfs.todo.md",
        "objectives_path": "docs/architecture/ipfs_kit_fuse_vfs.objectives.md",
        "plan_path": "docs/architecture/IPFS_KIT_FUSE_VFS_PLAN.md",
        "validator_path": "scripts/validate_ipfs_kit_fuse_vfs_board.py",
        "task_prefix": "KVFS-",
        "goal_prefix": "KVFS-G",
        "board_namespace": NAMESPACE,
        "merge_target_branch": BRANCH,
        "max_lanes": 4,
        "strict_task_sharding": True,
        "exit_when_all_tracks_terminal": True,
        "objective_refill_enabled": False,
        "codebase_refill_enabled": False,
    }
    for field, expected in exact_config.items():
        if config.get(field) != expected:
            errors.append(f"scheduler {field} mismatch: expected {expected!r}, got {config.get(field)!r}")
    projection = config.get("initial_projection", {})
    expected_projection = {
        "task_count": 40,
        "completed_task_ids": list(INITIAL_COMPLETED),
        "ready_task_ids": list(INITIAL_READY),
        "blocked_task_ids": [],
        "terminal_task_id": TERMINAL_TASK,
        "goal_count": 9,
        "root_goal_id": "KVFS-G000",
    }
    if projection != expected_projection:
        errors.append("scheduler initial projection mismatch")
    source = config.get("source_binding", {})
    for field, expected in {
        "accelerator_required_ancestor": ACCELERATOR_ANCESTOR,
        "accelerator_required_branch": BRANCH,
        "ipfs_kit_submodule_path": "ipfs_kit_py",
        "ipfs_kit_planning_revision": IPFS_KIT_REVISION,
    }.items():
        if not isinstance(source, dict) or source.get(field) != expected:
            errors.append(f"scheduler source_binding.{field} mismatch")
    if config.get("worktree_submodule_paths") != ["ipfs_kit_py"]:
        errors.append("scheduler worktree_submodule_paths mismatch")
    if tuple(config.get("protected_paths", ())) != PROTECTED_PATHS:
        errors.append("scheduler protected_paths mismatch")
    configured_groups = config.get("task_groups", {})
    if not isinstance(configured_groups, dict) or {
        key: tuple(value) if isinstance(value, list) else ()
        for key, value in configured_groups.items()
    } != TASK_GROUPS:
        errors.append("scheduler task_groups mismatch")
    lanes = config.get("lanes", [])
    if not isinstance(lanes, list) or len(lanes) != 4:
        errors.append("scheduler must define exactly four lanes")
    else:
        for index, lane in enumerate(lanes):
            expected_task = INITIAL_SHARDS[index]
            if not isinstance(lane, dict) or lane.get("index") != index or lane.get("strict_shard_remainder") != index or lane.get("initial_task_ids") != [expected_task]:
                errors.append(f"scheduler lane {index} mismatch")
    provider = config.get("provider", {})
    provider_seal = {
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.5",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_trigger": "primary_quota_exhausted",
        "fallback_reasoning_effort": "medium",
        "max_concurrency": 4,
        "secrets_from_environment_only": True,
        "secrets_in_argv_prompts_logs_or_receipts": False,
    }
    if provider != provider_seal:
        errors.append("scheduler ordered provider seal mismatch")
    runtime = config.get("runtime_paths", {})
    expected_root = "data/agent_supervisor/ipfs_kit_fuse_vfs"
    if not isinstance(runtime, dict) or runtime.get("root") != expected_root:
        errors.append("scheduler runtime root mismatch")
    elif any(
        not isinstance(value, str) or (key not in {"root", "generated_runtime_artifacts_are_completion_authority"} and not value.startswith(expected_root + "/"))
        for key, value in runtime.items()
        if key != "generated_runtime_artifacts_are_completion_authority"
    ):
        errors.append("scheduler runtime paths escape runtime root")
    capability = config.get("native_capability_policy", {})
    required_capability = {
        "doctor_timeout_seconds": 5,
        "mount_readiness_timeout_seconds": 15,
        "integration_case_timeout_seconds": 60,
        "mount_runs_as_bounded_child_process": True,
        "exclusive_mountpoint_and_drive_leases": True,
        "cleanup_finally_and_watchdog_required": True,
        "capability_absence_receipt": "capability_unavailable",
        "capability_absence_may_leave_task_running": False,
        "linux_windows_and_container_certification_independent": True,
    }
    if capability != required_capability:
        errors.append("scheduler native capability anti-stall policy mismatch")

    if PLAN_PATH.is_file():
        plan = PLAN_PATH.read_text(encoding="utf-8").lower()
        required_terms = (
            "canonicalvfsservice", "wal", "generationboundarc", "fusepy",
            "winfsp", "/dev/fuse", "sys_admin", "fsync", "recovery",
            "sha256(task_id)", "capability_unavailable", "rollback",
        )
        for term in required_terms:
            if term not in plan:
                errors.append(f"plan omits required term {term!r}")

    ignore = subprocess.run(
        ("git", "check-ignore", "-q", "data/agent_supervisor/ipfs_kit_fuse_vfs/probe"),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if ignore.returncode != 0:
        errors.append("configured runtime path is not ignored")

    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/ipfs-kit-fuse-vfs-board-validation@1",
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "task_count": len(tasks),
        "parsed_task_count": len(parsed_tasks),
        "operational_task_count": len(operational_task_ids),
        "operational_task_ids": list(operational_task_ids),
        "pending_operational_task_ids": list(pending_operational_task_ids),
        "goal_count": len(goals),
        "completed_task_ids": sorted(completed),
        "blocked_task_ids": sorted(blocked),
        "ready_task_ids": list(ready),
        "initial_ready_task_ids": list(INITIAL_READY),
        "initial_shards": {
            str(index): task_id for index, task_id in sorted(initial_shards.items())
        },
        "current_ready_shards": current_ready_shards,
        "terminal_task_id": TERMINAL_TASK,
        "board_namespace": NAMESPACE,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-all", action="store_true", help="validate all sealed board invariants")
    parser.parse_args(argv)
    report = validate()
    json.dump(report, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
