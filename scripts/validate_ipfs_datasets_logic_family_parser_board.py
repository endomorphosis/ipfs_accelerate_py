#!/usr/bin/env python3
"""Fail-closed validator for the IPFS Datasets logic-family parser board."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (  # noqa: E402
    materialize_task_dependency_dag,
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (  # noqa: E402
    ConfiguredBoardError,
    configured_board_launch_plan,
    load_configured_board,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (  # noqa: E402
    parse_task_text,
)

PLAN_PATH = REPO_ROOT / "docs/architecture/IPFS_DATASETS_LOGIC_FAMILY_PARSER_PLAN.md"
OBJECTIVE_PATH = (
    REPO_ROOT / "docs/architecture/ipfs_datasets_logic_family_parser.objectives.md"
)
TODO_PATH = REPO_ROOT / "docs/architecture/ipfs_datasets_logic_family_parser.todo.md"
SCHEDULER_PATH = (
    REPO_ROOT
    / "config/agent_supervisor_ipfs_datasets_logic_family_parser_scheduler.json"
)

BOARD_NAMESPACE = "ipfs-datasets-logic-family-parser-v1"
MERGE_TARGET_BRANCH = "agent/logic-family-parser-supervisor"
ACCELERATOR_REQUIRED_ANCESTOR = "34420f615d3eebfefa3cc1a3e4ebf8f51b16afac"
DATASETS_REVISION = "a2f5400b7cb89c8481819379a1b7b9959fe81d45"
RUNTIME_ROOT = "data/agent_supervisor/ipfs_datasets_logic_family_parser"
TASK_IDS = tuple(f"LFP-{index:03d}" for index in range(48))
GOAL_IDS = (
    "LFP-G000",
    "LFP-G010",
    "LFP-G020",
    "LFP-G030",
    "LFP-G040",
    "LFP-G050",
    "LFP-G060",
    "LFP-G070",
    "LFP-G080",
    "LFP-G090",
    "LFP-G100",
)
INITIAL_COMPLETED = ("LFP-000",)
INITIAL_READY = ("LFP-001", "LFP-002", "LFP-003", "LFP-004")
TERMINAL_TASK = "LFP-047"

# This is filled with the normalized seed-board digest once LFP-000..047 are
# materialized. Status is the only mutable field excluded from the seal.
SEALED_TASKBOARD_DEFINITION_SHA256 = (
    "sha256:f5d01bcc13c0b62d35b713cccb2e04abe49da454e9fa6f35cd28a5ad4b72eb44"
)

EXPECTED_TASK_GROUPS: Mapping[str, tuple[str, ...]] = {
    "LFP-G010": tuple(f"LFP-{index:03d}" for index in range(1, 6)),
    "LFP-G020": tuple(f"LFP-{index:03d}" for index in range(6, 11)),
    "LFP-G030": tuple(f"LFP-{index:03d}" for index in range(11, 17)),
    "LFP-G040": tuple(f"LFP-{index:03d}" for index in range(17, 23)),
    "LFP-G050": tuple(f"LFP-{index:03d}" for index in range(23, 29)),
    "LFP-G060": tuple(f"LFP-{index:03d}" for index in range(29, 34)),
    "LFP-G070": tuple(f"LFP-{index:03d}" for index in range(34, 40)),
    "LFP-G080": tuple(f"LFP-{index:03d}" for index in range(40, 44)),
    "LFP-G090": tuple(f"LFP-{index:03d}" for index in range(44, 47)),
    "LFP-G100": ("LFP-047",),
}
EXPECTED_TASK_TO_GOAL = {
    "LFP-000": "LFP-G000",
    **{
        task_id: goal_id
        for goal_id, task_ids in EXPECTED_TASK_GROUPS.items()
        for task_id in task_ids
    },
}
GOAL_STATES = frozenset(
    {
        "active",
        "provisionally_complete",
        "verified_complete",
        "analysis_inconclusive",
        "blocked",
        "reopened",
    }
)
PERSISTED_TASK_STATES = frozenset({"todo", "completed"})
REQUIRED_GOAL_FIELDS = (
    "status",
    "parent",
    "depends_on",
    "fib_priority",
    "track",
    "priority",
    "bundle",
    "parallel_lane",
    "resource_class",
    "goal",
    "evidence",
    "evidence_criteria",
    "evidence_source_policy",
    "outputs",
    "predicted_files",
    "interfaces",
    "validation",
    "acceptance",
    "gap_task",
    "refinement",
    "embedding_query",
    "ast_query",
    "conflict_policy",
)
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
    "interfaces",
    "allow concurrent with",
    "conflict policy",
    "preconditions",
    "effects",
    "evidence subset",
    "symbolic first",
    "llm context budget bytes",
    "acceptance",
    "embedding query",
)
REQUIRED_PLAN_TERMS = (
    "syntax_core",
    "security_ir",
    "crypto_ir",
    "intent_ir",
    "legal_ir",
    "ui_ux_ir",
    "z3",
    "cvc5",
    "tlc",
    "apalache",
    "secpal",
    "proverif",
    "tamarin",
    "hyperltl",
    "vampire",
    "lean",
    "rocq",
    "isabelle",
    "runtime mtl",
    "ergoai",
    "symbolicai",
    "refill",
)

CONTROL_PATHS = frozenset(
    {
        "docs/architecture/IPFS_DATASETS_LOGIC_FAMILY_PARSER_PLAN.md",
        "docs/architecture/ipfs_datasets_logic_family_parser.objectives.md",
        "docs/architecture/ipfs_datasets_logic_family_parser.todo.md",
        "config/agent_supervisor_ipfs_datasets_logic_family_parser_scheduler.json",
        "scripts/validate_ipfs_datasets_logic_family_parser_board.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/grok_cli_runner.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
    }
)
EXPECTED_PROVIDER = {
    "primary_provider_id": "grok_cli",
    "primary_model_id": "grok-4.5",
    "fallback_provider_id": "codex",
    "fallback_model_id": "gpt-5.6-terra",
    "fallback_trigger": "primary_quota_exhausted",
    "fallback_reasoning_effort": "high",
    "max_concurrency": 4,
    "secrets_from_environment_only": True,
    "secrets_in_argv_prompts_logs_or_receipts": False,
}
EXPECTED_SOURCE_BINDING = {
    "accelerator_required_ancestor": ACCELERATOR_REQUIRED_ANCESTOR,
    "accelerator_required_branch": MERGE_TARGET_BRANCH,
    "ipfs_datasets_submodule_path": "ipfs_datasets_py",
    "ipfs_datasets_planning_revision": DATASETS_REVISION,
    "require_initialized_gitlinks": True,
    "require_superproject_gitlink_equals_nested_head": True,
    "require_clean_nested_worktree_at_task_start": True,
    "record_recursive_repository_forest_at_launch": True,
    "changed_revision_requires_fresh_inventory_and_baseline": True,
    "planning_revision_is_runtime_completion_evidence": False,
}
EXPECTED_DERIVED_REFILL = {
    "max_goals_per_epoch": 8,
    "max_tasks_per_epoch": 24,
    "min_open_tasks": 8,
    "max_open_tasks": 48,
    "max_refinement_depth": 3,
    "max_unchanged_failure_retries": 2,
    "cooldown_seconds": 3600,
    "mutate_seed_board": False,
}


def _csv(value: object) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in re.split(r"[,;]", str(value or ""))
        if item.strip()
    )


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _seed_taskboard_definition_sha256(text: str) -> str:
    """Hash LFP-000..047 while normalizing their mutable status fields."""

    canonical_text = text
    for match in re.finditer(r"^## (LFP-\d+)\b", text, flags=re.MULTILINE):
        if match.group(1) in TASK_IDS:
            continue
        canonical_text = text[: match.start()].rstrip("\r\n") + "\n"
        break

    normalized: list[str] = []
    current_task_id = ""
    for line in canonical_text.splitlines(keepends=True):
        heading = re.match(r"^## (LFP-\d+)\b", line)
        if heading:
            current_task_id = heading.group(1)
        if current_task_id in TASK_IDS and line.startswith("- Status:"):
            newline = (
                "\r\n"
                if line.endswith("\r\n")
                else "\n"
                if line.endswith("\n")
                else ""
            )
            status = "completed" if current_task_id == "LFP-000" else "todo"
            line = f"- Status: {status}{newline}"
        normalized.append(line)
    return "sha256:" + hashlib.sha256(
        "".join(normalized).encode("utf-8")
    ).hexdigest()


def _safe_relative_paths(values: Iterable[str], *, field: str) -> list[str]:
    errors: list[str] = []
    for raw in values:
        value = str(raw).strip().replace("\\", "/")
        path = PurePosixPath(value)
        if (
            not value
            or "\x00" in value
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() in {".", ".."}
            or (path.parts and path.parts[0].endswith(":"))
        ):
            errors.append(f"{field} contains unsafe path {raw!r}")
    return errors


def _cycle_nodes(edges: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    visiting: set[str] = set()
    visited: set[str] = set()
    cycle: set[str] = set()

    def visit(node: str, lineage: tuple[str, ...]) -> None:
        if node in visited:
            return
        if node in visiting:
            cycle.update(lineage[lineage.index(node) :] if node in lineage else (node,))
            return
        visiting.add(node)
        for dependency in edges.get(node, ()):
            if dependency in edges:
                visit(dependency, (*lineage, node))
        visiting.remove(node)
        visited.add(node)

    for node in sorted(edges):
        visit(node, ())
    return tuple(sorted(cycle))


def _transitive_dependencies(
    task_id: str,
    edges: Mapping[str, Sequence[str]],
) -> set[str]:
    result: set[str] = set()
    pending = list(edges.get(task_id, ()))
    while pending:
        dependency = pending.pop()
        if dependency in result or dependency not in edges:
            continue
        result.add(dependency)
        pending.extend(edges.get(dependency, ()))
    return result


def _positive_int(value: object, *, noun: str, errors: list[str]) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        errors.append(f"{noun} is not an integer")
        return -1
    if parsed <= 0:
        errors.append(f"{noun} must be positive")
    return parsed


def _load_json(path: Path, errors: list[str]) -> dict[str, object]:
    if not path.is_file():
        errors.append(f"JSON file is missing: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid JSON in {path}: {type(exc).__name__}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"JSON root must be an object: {path}")
        return {}
    return value


def _git(*args: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ("git", *args),
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(
            ("git", *args), 124, "", f"{type(exc).__name__}: {exc}"
        )


def _validate_actual_source_binding(errors: list[str]) -> dict[str, object]:
    branch = _git("branch", "--show-current")
    if branch.returncode != 0 or branch.stdout.strip() != MERGE_TARGET_BRANCH:
        errors.append("checkout branch does not match the sealed merge target")
    ancestor = _git(
        "merge-base", "--is-ancestor", ACCELERATOR_REQUIRED_ANCESTOR, "HEAD"
    )
    if ancestor.returncode != 0:
        errors.append("accelerator required ancestor is not an ancestor of HEAD")

    gitlink = _git("ls-tree", "HEAD", "--", "ipfs_datasets_py")
    gitlink_match = re.fullmatch(
        r"160000 commit ([0-9a-f]{40})\tipfs_datasets_py\n?", gitlink.stdout
    )
    gitlink_revision = gitlink_match.group(1) if gitlink_match else ""
    if not gitlink_revision:
        errors.append("ipfs_datasets_py gitlink is missing or malformed")

    nested = REPO_ROOT / "ipfs_datasets_py"
    nested_top = _git("rev-parse", "--show-toplevel", cwd=nested)
    exact_nested_root = bool(
        nested_top.returncode == 0
        and Path(nested_top.stdout.strip()).resolve() == nested.resolve()
    )
    if not exact_nested_root:
        errors.append("ipfs_datasets_py is not an initialized exact nested worktree")
    nested_head = _git("rev-parse", "HEAD", cwd=nested) if exact_nested_root else None
    actual_nested_head = nested_head.stdout.strip() if nested_head else ""
    if gitlink_revision and actual_nested_head != gitlink_revision:
        errors.append("ipfs_datasets_py gitlink does not equal nested HEAD")
    planning_revision_ancestor = False
    if actual_nested_head:
        planning_ancestor = _git(
            "merge-base",
            "--is-ancestor",
            DATASETS_REVISION,
            actual_nested_head,
            cwd=nested,
        )
        planning_revision_ancestor = planning_ancestor.returncode == 0
        if not planning_revision_ancestor:
            errors.append(
                "ipfs_datasets_py nested HEAD does not descend from the sealed "
                "planning revision"
            )
    nested_status = (
        _git("status", "--porcelain=v1", "--untracked-files=all", cwd=nested)
        if exact_nested_root
        else None
    )
    nested_dirty = nested_status.stdout.splitlines() if nested_status else []
    if nested_status is None or nested_status.returncode != 0 or nested_dirty:
        errors.append("ipfs_datasets_py nested worktree is not clean")
    return {
        "branch": branch.stdout.strip(),
        "required_ancestor": ACCELERATOR_REQUIRED_ANCESTOR,
        "planning_revision": DATASETS_REVISION,
        "planning_revision_ancestor": planning_revision_ancestor,
        "gitlink": gitlink_revision,
        "nested_head": actual_nested_head,
        "nested_exact_worktree": exact_nested_root,
        "nested_dirty": nested_dirty,
    }


def validate(
    *,
    plan_path: Path,
    objective_path: Path,
    todo_path: Path,
    scheduler_path: Path,
) -> dict[str, object]:
    errors: list[str] = []
    for noun, path in (
        ("plan", plan_path),
        ("objective heap", objective_path),
        ("taskboard", todo_path),
        ("scheduler", scheduler_path),
    ):
        if not path.is_file():
            errors.append(f"{noun} is missing: {path}")
    if errors:
        return {"valid": False, "errors": errors}

    plan_text = plan_path.read_text(encoding="utf-8")
    plan_lower = plan_text.lower()
    for term in REQUIRED_PLAN_TERMS:
        if term not in plan_lower:
            errors.append(f"plan is missing required coverage term {term!r}")

    goals = parse_goal_heap(objective_path.read_text(encoding="utf-8"))
    goal_ids = tuple(goal.goal_id for goal in goals)
    goal_id_set = set(goal_ids)
    if goal_ids != GOAL_IDS:
        errors.append(f"goal IDs/order differ: expected {GOAL_IDS}, got {goal_ids}")
    if len(goal_ids) != len(goal_id_set):
        errors.append("objective heap contains duplicate goal IDs")

    goal_parent_edges: dict[str, tuple[str, ...]] = {}
    goal_dependency_edges: dict[str, tuple[str, ...]] = {}
    for goal in goals:
        missing = [field for field in REQUIRED_GOAL_FIELDS if field not in goal.fields]
        if missing:
            errors.append(f"{goal.goal_id} missing goal fields: {missing}")
        if not re.fullmatch(r"LFP-G\d{3}", goal.goal_id):
            errors.append(f"invalid goal ID {goal.goal_id!r}")
        if goal.status not in GOAL_STATES:
            errors.append(f"{goal.goal_id} has invalid status {goal.status!r}")
        parent = str(goal.fields.get("parent") or "").strip()
        parents = (parent,) if parent else ()
        goal_parent_edges[goal.goal_id] = parents
        if parent and parent not in goal_id_set:
            errors.append(f"{goal.goal_id} has unknown parent {parent!r}")
        dependencies = _csv(goal.fields.get("depends_on"))
        goal_dependency_edges[goal.goal_id] = dependencies
        for dependency in dependencies:
            if dependency not in goal_id_set:
                errors.append(
                    f"{goal.goal_id} has unknown goal dependency {dependency!r}"
                )
            elif dependency == goal.goal_id:
                errors.append(f"{goal.goal_id} depends on itself")
        _positive_int(
            goal.fields.get("fib_priority"),
            noun=f"{goal.goal_id} Fib priority",
            errors=errors,
        )
        if str(goal.fields.get("priority") or "") not in {"P0", "P1", "P2", "P3"}:
            errors.append(f"{goal.goal_id} has invalid priority")
        for field in REQUIRED_GOAL_FIELDS:
            if field not in {"parent", "depends_on"} and not str(
                goal.fields.get(field) or ""
            ).strip():
                errors.append(f"{goal.goal_id} has empty {field}")
        for field in ("outputs", "predicted_files"):
            errors.extend(
                f"{goal.goal_id}: {error}"
                for error in _safe_relative_paths(
                    _csv(goal.fields.get(field)), field=field
                )
            )

        evidence = _csv(goal.fields.get("evidence"))
        if goal.goal_id == "LFP-G000":
            expected_evidence = GOAL_IDS[1:]
            subgoals = _csv(goal.fields.get("subgoals"))
            if subgoals != expected_evidence:
                errors.append("LFP-G000 Subgoals do not match the ten child goals")
        else:
            expected_evidence = EXPECTED_TASK_GROUPS.get(goal.goal_id, ())
        if evidence != expected_evidence:
            errors.append(
                f"{goal.goal_id} evidence differs: expected {expected_evidence}, "
                f"got {evidence}"
            )

    roots = tuple(
        sorted(goal_id for goal_id, parents in goal_parent_edges.items() if not parents)
    )
    if roots != ("LFP-G000",):
        errors.append(f"expected only LFP-G000 as root, got {roots}")
    for goal_id in GOAL_IDS[1:]:
        if goal_parent_edges.get(goal_id) != ("LFP-G000",):
            errors.append(f"{goal_id} must be a direct subgoal of LFP-G000")
    for noun, edges in (
        ("goal parent", goal_parent_edges),
        ("goal dependency", goal_dependency_edges),
    ):
        cycle = _cycle_nodes(edges)
        if cycle:
            errors.append(f"{noun} graph contains a cycle: {list(cycle)}")

    todo_text = todo_path.read_text(encoding="utf-8")
    parsed_tasks = parse_task_text(
        todo_text,
        path=todo_path,
        task_header_prefix="## LFP-",
    )
    parsed_task_ids = tuple(task.task_id for task in parsed_tasks)
    if len(parsed_task_ids) != len(set(parsed_task_ids)):
        errors.append("taskboard contains duplicate task IDs")
    seed_tasks = parsed_tasks[: len(TASK_IDS)]
    refill_tasks = parsed_tasks[len(TASK_IDS) :]
    seed_task_ids = tuple(task.task_id for task in seed_tasks)
    if seed_task_ids != TASK_IDS:
        errors.append(
            f"seed task IDs/order differ: expected {TASK_IDS}, got {seed_task_ids}"
        )
    expected_refill_ids = tuple(
        f"LFP-{index:03d}"
        for index in range(len(TASK_IDS), len(TASK_IDS) + len(refill_tasks))
    )
    actual_refill_ids = tuple(task.task_id for task in refill_tasks)
    if actual_refill_ids != expected_refill_ids:
        errors.append(
            "refill task IDs must be contiguous and append-only: "
            f"expected {expected_refill_ids}, got {actual_refill_ids}"
        )

    taskboard_definition_sha256 = _seed_taskboard_definition_sha256(todo_text)
    if taskboard_definition_sha256 != SEALED_TASKBOARD_DEFINITION_SHA256:
        errors.append(
            "seed taskboard topology or metadata differs from the sealed projection"
        )

    all_task_ids = set(parsed_task_ids)
    task_edges: dict[str, tuple[str, ...]] = {}
    task_records: list[dict[str, object]] = []
    completed: set[str] = set()
    blocked: set[str] = set()
    task_goal_map: dict[str, str] = {}
    for task in parsed_tasks:
        metadata = task.metadata or {}
        is_seed = task.task_id in TASK_IDS
        missing = [field for field in REQUIRED_TASK_FIELDS if field not in metadata]
        if missing:
            errors.append(f"{task.task_id} missing task fields: {missing}")
        if task.status not in PERSISTED_TASK_STATES:
            errors.append(
                f"{task.task_id} has non-persistent status {task.status!r}; "
                "only todo/completed are sealed"
            )
        if task.status == "completed":
            completed.add(task.task_id)
        if task.status == "blocked":
            blocked.add(task.task_id)
        if task.priority not in {"P0", "P1", "P2", "P3"}:
            errors.append(f"{task.task_id} has invalid priority {task.priority!r}")
        expected_schedulable = "false" if task.task_id == "LFP-000" else "true"
        if metadata.get("is schedulable") != expected_schedulable:
            errors.append(
                f"{task.task_id} must declare Is schedulable: "
                f"{expected_schedulable}"
            )
        if metadata.get("review only") not in {"true", "false"}:
            errors.append(f"{task.task_id} has invalid review-only flag")
        if metadata.get("symbolic first") != "true":
            errors.append(f"{task.task_id} must declare Symbolic first: true")
        for field in (
            "estimated tokens",
            "implementation timeout seconds",
            "llm context budget bytes",
        ):
            _positive_int(
                metadata.get(field), noun=f"{task.task_id} {field}", errors=errors
            )
        if task.task_id == "LFP-000":
            if task.status != "completed":
                errors.append("LFP-000 must remain completed")
            if metadata.get("review only") != "true":
                errors.append("LFP-000 must remain review-only")

        goal_id = str(metadata.get("goal id") or "").strip()
        task_goal_map[task.task_id] = goal_id
        if goal_id not in goal_id_set:
            errors.append(f"{task.task_id} has unknown goal ID {goal_id!r}")
        expected_goal = EXPECTED_TASK_TO_GOAL.get(task.task_id)
        if is_seed and goal_id != expected_goal:
            errors.append(
                f"{task.task_id} goal differs: expected {expected_goal}, got {goal_id}"
            )

        dependencies = tuple(task.depends_on)
        task_edges[task.task_id] = tuple(
            dependency for dependency in dependencies if dependency in all_task_ids
        )
        for dependency in dependencies:
            if dependency not in all_task_ids and dependency not in goal_id_set:
                errors.append(f"{task.task_id} has unknown dependency {dependency!r}")
            if dependency == task.task_id:
                errors.append(f"{task.task_id} depends on itself")

        if not task.outputs:
            errors.append(f"{task.task_id} has no outputs")
        if not task.validation:
            errors.append(f"{task.task_id} has no validation command")
        if not task.acceptance:
            errors.append(f"{task.task_id} has empty acceptance")
        if task.board_namespace != BOARD_NAMESPACE:
            errors.append(
                f"{task.task_id} has unexpected board namespace "
                f"{task.board_namespace!r}"
            )
        predicted_files = _csv(metadata.get("predicted files"))
        if not predicted_files:
            errors.append(f"{task.task_id} has no predicted files")
        for field, values in (
            ("outputs", task.outputs),
            ("predicted files", predicted_files),
        ):
            errors.extend(
                f"{task.task_id}: {error}"
                for error in _safe_relative_paths(values, field=field)
            )
        if task.task_id != "LFP-000":
            allowed_output_prefixes = ["ipfs_datasets_py/"]
            if task.task_id == "LFP-046":
                allowed_output_prefixes.append(f"{RUNTIME_ROOT}/")
            outside_nested = sorted(
                path
                for path in (*task.outputs, *predicted_files)
                if not any(
                    path.startswith(prefix) for prefix in allowed_output_prefixes
                )
            )
            if outside_nested:
                errors.append(
                    f"{task.task_id} owns paths outside ipfs_datasets_py: "
                    f"{outside_nested}"
                )
            protected_overlap = sorted(
                path
                for path in (*task.outputs, *predicted_files)
                if path in CONTROL_PATHS
            )
            if protected_overlap:
                errors.append(
                    f"{task.task_id} owns protected paths: {protected_overlap}"
                )
        task_records.append(
            {
                "task_id": task.task_id,
                "title": task.title,
                "status": task.status,
                "goal_id": goal_id,
                "depends_on": list(dependencies),
                "outputs": list(task.outputs),
                "acceptance": task.acceptance,
                "board_namespace": task.board_namespace,
                "canonical_task_cid": task.canonical_task_cid,
            }
        )

    task_cycle = _cycle_nodes(task_edges)
    if task_cycle:
        errors.append(f"task dependency graph contains a cycle: {list(task_cycle)}")
    incomplete_completed = {
        task_id: sorted(
            dependency
            for dependency in task_edges.get(task_id, ())
            if dependency not in completed
        )
        for task_id in sorted(completed)
        if any(
            dependency not in completed for dependency in task_edges.get(task_id, ())
        )
    }
    if incomplete_completed:
        errors.append(
            "completed tasks are not dependency-closed: "
            + json.dumps(incomplete_completed, sort_keys=True)
        )

    terminal_ancestors = _transitive_dependencies(TERMINAL_TASK, task_edges)
    missing_terminal_ancestors = sorted(set(TASK_IDS[1:-1]) - terminal_ancestors)
    if missing_terminal_ancestors:
        errors.append(
            "seed tasks are not transitively upstream of LFP-047: "
            f"{missing_terminal_ancestors}"
        )

    ready = tuple(
        task_id
        for task_id in TASK_IDS
        if task_id in all_task_ids
        and task_id not in completed
        and task_id not in blocked
        and task_goal_map.get(task_id) in goal_id_set
        and all(
            dependency in completed
            for dependency in task_edges.get(task_id, ())
        )
    )
    if completed == set(INITIAL_COMPLETED) and ready != INITIAL_READY:
        errors.append(
            f"initial ready set differs: expected {INITIAL_READY}, got {ready}"
        )

    seed_records = [
        record for record in task_records if str(record["task_id"]) in TASK_IDS
    ]
    dependency_graph = materialize_task_dependency_dag(seed_records)
    if dependency_graph.invalid_task_cids:
        errors.append(
            "typed seed dependency graph has invalid task CIDs: "
            f"{list(dependency_graph.invalid_task_cids)}"
        )
    if dependency_graph.repair_evidence:
        errors.append(
            "typed seed dependency graph requires repair: "
            + json.dumps(
                [item.to_dict() for item in dependency_graph.repair_evidence],
                sort_keys=True,
            )
        )

    scheduler = _load_json(scheduler_path, errors)
    configured_board = None
    try:
        configured_board = load_configured_board(
            scheduler_path,
            repo_root=REPO_ROOT,
        )
    except ConfiguredBoardError as exc:
        errors.append(f"configured scheduler schema rejected: {exc}")

    expected_scheduler_scalars = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "ipfs_datasets_logic_family_parser.scheduler_config@1"
        ),
        "taskboard_path": todo_path.relative_to(REPO_ROOT).as_posix(),
        "objectives_path": objective_path.relative_to(REPO_ROOT).as_posix(),
        "plan_path": plan_path.relative_to(REPO_ROOT).as_posix(),
        "validator_path": (
            "scripts/validate_ipfs_datasets_logic_family_parser_board.py"
        ),
        "task_prefix": "LFP-",
        "goal_prefix": "LFP-G",
        "board_namespace": BOARD_NAMESPACE,
        "merge_target_branch": MERGE_TARGET_BRANCH,
        "max_lanes": 4,
        "strict_task_sharding": False,
        "exit_when_all_tracks_terminal": False,
        "objective_refill_enabled": True,
        "codebase_refill_enabled": False,
    }
    for key, expected in expected_scheduler_scalars.items():
        if scheduler.get(key) != expected:
            errors.append(
                f"scheduler {key} mismatch: expected {expected!r}, "
                f"got {scheduler.get(key)!r}"
            )

    expected_projection = {
        "task_count": 48,
        "completed_task_ids": list(INITIAL_COMPLETED),
        "ready_task_ids": list(INITIAL_READY),
        "blocked_task_ids": [],
        "terminal_task_id": TERMINAL_TASK,
        "goal_count": 11,
        "root_goal_id": "LFP-G000",
    }
    if scheduler.get("initial_projection") != expected_projection:
        errors.append("scheduler initial_projection differs from the launch seal")

    if scheduler.get("provider") != EXPECTED_PROVIDER:
        errors.append("scheduler provider route is not the exact quota-only Grok/Terra contract")
    if scheduler.get("source_binding") != EXPECTED_SOURCE_BINDING:
        errors.append("scheduler source_binding differs from the sealed source contract")
    if scheduler.get("worktree_submodule_paths") != ["ipfs_datasets_py"]:
        errors.append("scheduler worktree_submodule_paths must be exactly ipfs_datasets_py")
    protected_paths = scheduler.get("protected_paths")
    if not isinstance(protected_paths, list) or set(protected_paths) != CONTROL_PATHS:
        errors.append("scheduler protected_paths do not exactly match control/routing files")
    elif len(protected_paths) != len(CONTROL_PATHS):
        errors.append("scheduler protected_paths contain duplicates")

    runtime_paths = scheduler.get("runtime_paths")
    expected_runtime_paths = {
        "root": RUNTIME_ROOT,
        "state": f"{RUNTIME_ROOT}/state",
        "worktrees": f"{RUNTIME_ROOT}/worktrees",
        "merge_queue": f"{RUNTIME_ROOT}/merge-queue",
        "logs": f"{RUNTIME_ROOT}/logs",
        "evidence": f"{RUNTIME_ROOT}/evidence",
        "generated_runtime_artifacts_are_completion_authority": False,
    }
    if runtime_paths != expected_runtime_paths:
        errors.append("scheduler runtime_paths differ from the isolated runtime contract")

    lanes = scheduler.get("lanes")
    if not isinstance(lanes, list) or len(lanes) != 4:
        errors.append("scheduler must define exactly four lanes")
    else:
        for index, lane in enumerate(lanes):
            if not isinstance(lane, dict):
                errors.append(f"scheduler lane {index} is not an object")
                continue
            if lane.get("index") != index:
                errors.append(f"scheduler lane {index} has wrong index")
            if lane.get("strict_shard_remainder") != index:
                errors.append(f"scheduler lane {index} has wrong shard remainder")
            if "initial_task_ids" in lane:
                errors.append(
                    f"scheduler lane {index} declares unused initial_task_ids"
                )

    task_groups = scheduler.get("task_groups")
    expected_groups_json = {
        goal_id: list(task_ids)
        for goal_id, task_ids in EXPECTED_TASK_GROUPS.items()
    }
    if task_groups != expected_groups_json:
        errors.append("scheduler task_groups differ from objective evidence groups")

    refill_policy = scheduler.get("refill_policy")
    expected_refill_policy = {
        "source": "objective_heap",
        "append_only": True,
        "content_addressed": True,
        "seed_tasks_are_immutable": True,
        "unscoped_codebase_refill_allowed": False,
        "empty_scan_may_claim_completion": False,
        "derived_refill": EXPECTED_DERIVED_REFILL,
    }
    if refill_policy != expected_refill_policy:
        errors.append("scheduler refill_policy differs from the bounded append-only contract")

    if configured_board is not None:
        launch_plan = configured_board_launch_plan(
            configured_board,
            implement=True,
            detach=True,
            stamp="validator",
        )
        launch_argv = launch_plan["argv"]
        common_prefix = "--common-arg="
        common_args = [
            item[len(common_prefix) :]
            for item in launch_argv
            if isinstance(item, str) and item.startswith(common_prefix)
        ]
        if "--implementation-supervisor-strict-task-sharding" in launch_argv:
            errors.append("scheduler launch unexpectedly enables strict master sharding")
        if "--strict-task-sharding" in common_args:
            errors.append("scheduler launch unexpectedly enables strict lane sharding")
        expected_refill_args = {
            "--objective-scan-min-open-tasks": "8",
            "--objective-scan-max-findings": "24",
            "--objective-scan-cooldown-seconds": "3600",
        }
        for flag, expected_value in expected_refill_args.items():
            positions = [
                index
                for index, value in enumerate(common_args)
                if value == flag
            ]
            if (
                len(positions) != 1
                or positions[0] + 1 >= len(common_args)
                or common_args[positions[0] + 1] != expected_value
            ):
                errors.append(
                    f"scheduler launch does not seal {flag}={expected_value}"
                )

    actual_source_binding = _validate_actual_source_binding(errors)
    return {
        "schema": (
            "ipfs_accelerate_py/"
            "ipfs-datasets-logic-family-parser-preflight@1"
        ),
        "valid": not errors,
        "errors": errors,
        "plan_path": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "objective_path": str(objective_path),
        "objective_sha256": _sha256(objective_path),
        "goal_count": len(goals),
        "root_goal_ids": list(roots),
        "todo_path": str(todo_path),
        "todo_sha256": _sha256(todo_path),
        "taskboard_definition_sha256": taskboard_definition_sha256,
        "seed_task_count": len(seed_tasks),
        "refill_task_count": len(refill_tasks),
        "completed_task_ids": sorted(completed.intersection(TASK_IDS)),
        "ready_task_ids": list(ready),
        "blocked_task_ids": sorted(blocked),
        "terminal_task_id": TERMINAL_TASK,
        "scheduler_path": str(scheduler_path),
        "scheduler_sha256": _sha256(scheduler_path),
        "dependency_graph_id": _canonical_sha256(dependency_graph.to_dict()),
        "source_binding": actual_source_binding,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-all", action="store_true")
    parser.add_argument("--plan-path", type=Path, default=PLAN_PATH)
    parser.add_argument("--objective-path", type=Path, default=OBJECTIVE_PATH)
    parser.add_argument("--todo-path", type=Path, default=TODO_PATH)
    parser.add_argument("--scheduler-path", type=Path, default=SCHEDULER_PATH)
    args = parser.parse_args()
    del args.check_all
    result = validate(
        plan_path=args.plan_path.resolve(),
        objective_path=args.objective_path.resolve(),
        todo_path=args.todo_path.resolve(),
        scheduler_path=args.scheduler_path.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
