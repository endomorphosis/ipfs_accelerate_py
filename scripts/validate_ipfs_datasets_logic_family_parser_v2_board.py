#!/usr/bin/env python3
"""Fail-closed validator for the IPFS Datasets logic-parser Wave-2 board."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (  # noqa: E402
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

PLAN_PATH = REPO_ROOT / "docs/architecture/IPFS_DATASETS_LOGIC_FAMILY_PARSER_V2_PLAN.md"
OBJECTIVE_PATH = REPO_ROOT / "docs/architecture/ipfs_datasets_logic_family_parser_v2.objectives.md"
TODO_PATH = REPO_ROOT / "docs/architecture/ipfs_datasets_logic_family_parser_v2.todo.md"
SCHEDULER_PATH = REPO_ROOT / "config/agent_supervisor_ipfs_datasets_logic_family_parser_v2_scheduler.json"

BOARD_NAMESPACE = "ipfs-datasets-logic-family-parser-v2"
MERGE_TARGET_BRANCH = "agent/logic-family-parser-v2-supervisor"
RUNTIME_ROOT = "data/agent_supervisor/ipfs_datasets_logic_family_parser_v2"
TASK_IDS = tuple(f"LFP2-{index:03d}" for index in range(51))
GOAL_IDS = ("LFP2-G000",) + tuple(
    f"LFP2-G{index:03d}" for index in range(10, 101, 10)
)
INITIAL_COMPLETED = ("LFP2-000",)
INITIAL_READY = ("LFP2-001", "LFP2-002", "LFP2-003", "LFP2-004")
TERMINAL_TASK = "LFP2-050"

PREDECESSOR_ACCELERATOR_COMMIT = "e162c19d087d4e6511f8eb97fd34ecb449777897"
PREDECESSOR_DATASETS_COMMIT = "fc49cbb3e0e96bf07b367859da32123187d706c1"
PREDECESSOR_SEED_DEFINITION = (
    "sha256:f5d01bcc13c0b62d35b713cccb2e04abe49da454e9fa6f35cd28a5ad4b72eb44"
)
PREDECESSOR_RELEASE_SHA256 = (
    "sha256:86412a60bfde9b8a13156ab097b44443a4a8f70a7b286f1c7a707366c93757ce"
)
PREDECESSOR_FILE_DIGESTS = {
    "docs/architecture/IPFS_DATASETS_LOGIC_FAMILY_PARSER_PLAN.md": (
        "sha256:9d07ef064e80081a67d13f754fff10b84b6176facf687b88ed1164d71a90e9c0"
    ),
    "docs/architecture/ipfs_datasets_logic_family_parser.objectives.md": (
        "sha256:1bc111b24e44508d56f4932da4ce0a76357eaaf01bf5ea22842cf06621b24217"
    ),
    "docs/architecture/ipfs_datasets_logic_family_parser.todo.md": (
        "sha256:8e851a11e3fbd1a0b174e2077abaa398c15fecdf9b9bb8baf9592b3311f5aaa8"
    ),
    "ipfs_datasets_py/data/logic/conformance/logic_family_parser_release.json": (
        PREDECESSOR_RELEASE_SHA256
    ),
}

# Filled after the 51 seed cards are materialized. Only Status values are
# normalized, so implementation progress cannot mutate semantic task identity.
SEALED_SEED_DEFINITION_SHA256 = (
    "sha256:770912c9d7123f12bf06cc1428900bf5f8e2e7b662cd3c95e9d0be11181d3e22"
)

EXPECTED_TASK_GROUPS: Mapping[str, tuple[str, ...]] = {
    "LFP2-G010": tuple(f"LFP2-{index:03d}" for index in range(1, 5)),
    "LFP2-G020": tuple(f"LFP2-{index:03d}" for index in range(5, 10)),
    "LFP2-G030": tuple(f"LFP2-{index:03d}" for index in range(10, 16)),
    "LFP2-G040": tuple(f"LFP2-{index:03d}" for index in range(16, 22)),
    "LFP2-G050": tuple(f"LFP2-{index:03d}" for index in range(22, 28)),
    "LFP2-G060": tuple(f"LFP2-{index:03d}" for index in range(28, 37)),
    "LFP2-G070": tuple(f"LFP2-{index:03d}" for index in range(37, 44)),
    "LFP2-G080": tuple(f"LFP2-{index:03d}" for index in range(44, 48)),
    "LFP2-G090": ("LFP2-048", "LFP2-049"),
    "LFP2-G100": ("LFP2-050",),
}
EXPECTED_TASK_TO_GOAL = {
    "LFP2-000": "LFP2-G000",
    **{
        task_id: goal_id
        for goal_id, task_ids in EXPECTED_TASK_GROUPS.items()
        for task_id in task_ids
    },
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
REQUIRED_GOAL_FIELDS = (
    "status",
    "review_only",
    "parent",
    "depends_on",
    "fib_priority",
    "track",
    "priority",
    "bundle",
    "parallel_lane",
    "resource_class",
    "goal",
    "seed_tasks",
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
REQUIRED_PLAN_TERMS = (
    "syntax_core",
    "security_ir",
    "crypto_ir",
    "intent_ir",
    "legal_ir",
    "ui_ux_ir",
    "z3",
    "cvc5",
    "tla_tlc",
    "apalache",
    "datalog_secpal",
    "proverif",
    "tamarin",
    "hyperltl_autohyper_mchyper",
    "vampire",
    "eprover",
    "hammer",
    "lean",
    "rocq",
    "isabelle",
    "ergoai",
    "symbolicai",
    "runtime_mtl",
    "description-logic",
    "argumentation",
    "mu-calculus",
    "finite-field",
    "session",
    "refill",
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
EXPECTED_ENVIRONMENT = {
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_PROVIDER": "grok_cli",
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_PROVIDER": "codex",
    "IPFS_ACCELERATE_AGENT_IMPLEMENTATION_FALLBACK_TRIGGER": (
        "primary_quota_exhausted"
    ),
    "IPFS_ACCELERATE_AGENT_GROK_MODEL": "grok-4.5",
    "IPFS_ACCELERATE_AGENT_CODEX_MODEL": "gpt-5.6-terra",
    "IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT": "high",
}
CONTROL_PATHS = frozenset(
    {
        "docs/architecture/IPFS_DATASETS_LOGIC_FAMILY_PARSER_V2_PLAN.md",
        "docs/architecture/ipfs_datasets_logic_family_parser_v2.objectives.md",
        "docs/architecture/ipfs_datasets_logic_family_parser_v2.todo.md",
        "config/agent_supervisor_ipfs_datasets_logic_family_parser_v2_scheduler.json",
        "scripts/validate_ipfs_datasets_logic_family_parser_v2_board.py",
        *PREDECESSOR_FILE_DIGESTS,
    }
)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _safe_relative(value: str) -> bool:
    if not value or "\x00" in value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and "." not in path.parts


def _git(*args: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def _seed_text(text: str) -> str:
    start = text.find("## LFP2-000 ")
    if start < 0:
        return ""
    appended = re.search(r"(?m)^## LFP2-(?:05[1-9]|0[6-9][0-9]|[1-9][0-9]{2,}) ", text[start:])
    end = start + appended.start() if appended else len(text)
    seed = text[start:end].rstrip() + "\n"
    return re.sub(r"(?m)^- Status: .+$", "- Status: <normalized>", seed)


def _seed_digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(_seed_text(text).encode("utf-8")).hexdigest()


def _task_blocks(text: str) -> Mapping[str, str]:
    matches = list(re.finditer(r"(?m)^## (LFP2-[0-9]{3,}) .+$", text))
    result: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        result[match.group(1)] = text[match.start():end]
    return result


def _validate_goals(text: str, scheduler: Mapping[str, object], errors: list[str]) -> None:
    goals = parse_goal_heap(text)
    by_id = {goal.goal_id: goal for goal in goals}
    if tuple(by_id) != GOAL_IDS:
        errors.append(f"goal IDs/order differ: {tuple(by_id)!r}")
    if len(by_id) != len(goals):
        errors.append("duplicate goal ID")
    for goal_id, goal in by_id.items():
        missing = [field for field in REQUIRED_GOAL_FIELDS if field not in goal.fields]
        if missing:
            errors.append(f"{goal_id} missing goal fields: {missing}")
            continue
        if re.search(r"\bLFP2-[0-9]{3}\b", goal.fields["evidence"]):
            errors.append(f"{goal_id} Evidence conflates task IDs with evidence")
        if goal_id == "LFP2-G000":
            expected_seed = ("LFP2-000",)
            expected_parent = ""
        else:
            expected_seed = EXPECTED_TASK_GROUPS.get(goal_id, ())
            expected_parent = "LFP2-G000"
        if _split_csv(goal.fields["seed_tasks"]) != expected_seed:
            errors.append(f"{goal_id} Seed tasks differ from sealed task group")
        if goal.fields["parent"].strip() != expected_parent:
            errors.append(f"{goal_id} parent differs from sealed hierarchy")
    if scheduler.get("task_groups") != {
        goal: list(tasks) for goal, tasks in EXPECTED_TASK_GROUPS.items()
    }:
        errors.append("scheduler task_groups differ from objective Seed tasks")


def _validate_tasks(text: str, errors: list[str]) -> dict[str, object]:
    tasks = parse_task_text(text, path=TODO_PATH, task_header_prefix="## LFP2-")
    by_id = {task.task_id: task for task in tasks}
    if len(by_id) != len(tasks):
        errors.append("duplicate task ID")
    actual_ids = tuple(by_id)
    if actual_ids[: len(TASK_IDS)] != TASK_IDS:
        errors.append("seed task IDs/order are not exactly LFP2-000..LFP2-050")
    for offset, task_id in enumerate(actual_ids[len(TASK_IDS):], start=len(TASK_IDS)):
        if task_id != f"LFP2-{offset:03d}":
            errors.append(f"appended task ID discontinuity at {task_id}")
    blocks = _task_blocks(text)
    completed: set[str] = set()
    open_ids: set[str] = set()
    dependencies: dict[str, tuple[str, ...]] = {}
    output_sets: dict[str, set[str]] = {}
    for position, task in enumerate(tasks):
        metadata = task.metadata
        missing = [field for field in REQUIRED_TASK_FIELDS if field not in metadata]
        if missing:
            errors.append(f"{task.task_id} missing task fields: {missing}")
            continue
        block = blocks.get(task.task_id, "")
        normalized_keys = re.findall(r"(?m)^- ([^:\n]+):", block)
        duplicates = sorted({key.lower() for key in normalized_keys if sum(1 for item in normalized_keys if item.lower() == key.lower()) > 1})
        if duplicates:
            errors.append(f"{task.task_id} duplicate metadata keys: {duplicates}")
        seed = position < len(TASK_IDS)
        allowed_states = {"todo", "completed"} if seed else {"todo", "completed", "blocked"}
        if task.status not in allowed_states:
            errors.append(f"{task.task_id} invalid status {task.status!r}")
        if task.status == "completed":
            completed.add(task.task_id)
        else:
            open_ids.add(task.task_id)
        if metadata["board namespace"] != BOARD_NAMESPACE:
            errors.append(f"{task.task_id} board namespace mismatch")
        if seed and EXPECTED_TASK_TO_GOAL.get(task.task_id) != metadata["goal id"]:
            errors.append(f"{task.task_id} goal mapping mismatch")
        if not seed:
            generated_by = metadata.get("generated by", "")
            if not generated_by.startswith("ipfs_accelerate_py.agent_supervisor."):
                errors.append(f"{task.task_id} appended card lacks trusted provenance")
        if metadata["completion"] != "manual":
            errors.append(f"{task.task_id} completion must be manual")
        schedulable = metadata["is schedulable"].lower()
        if task.task_id == "LFP2-000":
            if schedulable != "false" or metadata["review only"].lower() != "true":
                errors.append("LFP2-000 must be non-schedulable review-only")
        elif seed and schedulable != "true":
            errors.append(f"{task.task_id} seed implementation task must be schedulable")
        if metadata["symbolic first"].lower() != "true":
            errors.append(f"{task.task_id} Symbolic first must be true")
        for field in ("estimated tokens", "implementation timeout seconds", "llm context budget bytes"):
            try:
                if int(metadata[field]) <= 0:
                    raise ValueError
            except ValueError:
                errors.append(f"{task.task_id} {field} must be a positive integer")
        deps = tuple(task.depends_on)
        dependencies[task.task_id] = deps
        for dependency in deps:
            if dependency not in by_id:
                errors.append(f"{task.task_id} has unknown dependency {dependency}")
            elif actual_ids.index(dependency) >= position:
                errors.append(f"{task.task_id} dependency {dependency} is not earlier")
        outputs = set(task.outputs)
        predicted = set(_split_csv(metadata["predicted files"]))
        output_sets[task.task_id] = outputs
        if outputs != predicted:
            errors.append(f"{task.task_id} Outputs/Predicted files differ")
        for output in outputs:
            if not _safe_relative(output):
                errors.append(f"{task.task_id} has unsafe output {output!r}")
            if task.task_id != "LFP2-000" and output in CONTROL_PATHS:
                errors.append(f"{task.task_id} owns protected control output {output}")
            allowed_output = output.startswith("ipfs_datasets_py/") or (
                task.task_id == "LFP2-049" and output.startswith(f"{RUNTIME_ROOT}/refill/")
            ) or task.task_id == "LFP2-000"
            if not allowed_output:
                errors.append(f"{task.task_id} output is outside admitted owner roots: {output}")
    for task_id in completed:
        missing_completed = set(dependencies.get(task_id, ())) - completed
        if missing_completed:
            errors.append(f"{task_id} completed before dependencies {sorted(missing_completed)}")
    ancestors: set[str] = set()
    stack = list(dependencies.get(TERMINAL_TASK, ()))
    while stack:
        item = stack.pop()
        if item in ancestors:
            continue
        ancestors.add(item)
        stack.extend(dependencies.get(item, ()))
    if set(TASK_IDS[:-1]) - ancestors:
        errors.append(f"terminal task does not cover: {sorted(set(TASK_IDS[:-1]) - ancestors)}")
    ready = tuple(
        task_id
        for task_id in actual_ids
        if task_id in open_ids and set(dependencies.get(task_id, ())).issubset(completed)
    )
    if completed == set(INITIAL_COMPLETED) and ready != INITIAL_READY:
        errors.append(f"initial ready set differs: expected {INITIAL_READY}, got {ready}")
    for left_index, left in enumerate(INITIAL_READY):
        for right in INITIAL_READY[left_index + 1:]:
            overlap = output_sets.get(left, set()) & output_sets.get(right, set())
            if overlap:
                errors.append(f"initial tasks {left}/{right} overlap outputs: {sorted(overlap)}")
    if TERMINAL_TASK in completed:
        if open_ids:
            errors.append("terminal release completed while tasks remain open")
        fixed_point = REPO_ROOT / RUNTIME_ROOT / "refill/fixed_point_receipt.json"
        try:
            receipt = json.loads(fixed_point.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            receipt = {}
        if receipt.get("is_fixed_point") is not True or int(receipt.get("consecutive_empty_scans", 0)) < 2:
            errors.append("terminal release lacks a current two-scan fixed-point receipt")
    return {
        "task_count": len(tasks),
        "completed_task_ids": sorted(completed),
        "ready_task_ids": list(ready),
        "open_task_ids": sorted(open_ids),
        "refill_task_count": max(0, len(tasks) - len(TASK_IDS)),
    }


def _validate_plan(text: str, errors: list[str]) -> None:
    lowered = text.lower().replace("-", "_")
    for term in REQUIRED_PLAN_TERMS:
        normalized = term.lower().replace("-", "_")
        if normalized not in lowered:
            errors.append(f"plan missing required term: {term}")


def _validate_predecessor(scheduler: Mapping[str, object], errors: list[str]) -> None:
    for relative, expected in PREDECESSOR_FILE_DIGESTS.items():
        path = REPO_ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            errors.append(f"Wave-1 predecessor artifact changed: {relative}")
    expected_binding = {
        "predecessor_board_namespace": "ipfs-datasets-logic-family-parser-v1",
        "predecessor_terminal_task_id": "LFP-047",
        "predecessor_accelerator_commit": PREDECESSOR_ACCELERATOR_COMMIT,
        "predecessor_datasets_commit": PREDECESSOR_DATASETS_COMMIT,
        "predecessor_seed_definition_sha256": PREDECESSOR_SEED_DEFINITION,
        "predecessor_release_receipt_path": "ipfs_datasets_py/data/logic/conformance/logic_family_parser_release.json",
        "predecessor_release_receipt_sha256": PREDECESSOR_RELEASE_SHA256,
    }
    if scheduler.get("predecessor_binding") != expected_binding:
        errors.append("scheduler predecessor_binding differs from release seal")
    if _git("merge-base", "--is-ancestor", PREDECESSOR_ACCELERATOR_COMMIT, "HEAD").returncode != 0:
        errors.append("Wave-1 accelerator release is not an ancestor of HEAD")
    nested = REPO_ROOT / "ipfs_datasets_py"
    if _git("merge-base", "--is-ancestor", PREDECESSOR_DATASETS_COMMIT, "HEAD", cwd=nested).returncode != 0:
        errors.append("Wave-1 datasets release is not an ancestor of nested HEAD")


def _common_args(plan: Mapping[str, object]) -> list[str]:
    prefix = "--common-arg="
    return [
        item[len(prefix):]
        for item in plan.get("argv", [])
        if isinstance(item, str) and item.startswith(prefix)
    ]


def _validate_scheduler(scheduler: Mapping[str, object], errors: list[str]) -> None:
    expected_projection = {
        "task_count": 51,
        "completed_task_ids": list(INITIAL_COMPLETED),
        "ready_task_ids": list(INITIAL_READY),
        "blocked_task_ids": [],
        "terminal_task_id": TERMINAL_TASK,
        "goal_count": 11,
        "root_goal_id": "LFP2-G000",
    }
    if scheduler.get("initial_projection") != expected_projection:
        errors.append("scheduler initial_projection differs from launch seal")
    if scheduler.get("provider") != EXPECTED_PROVIDER:
        errors.append("scheduler provider route differs from Grok/quota-only Terra-high seal")
    if scheduler.get("merge_target_branch") != MERGE_TARGET_BRANCH:
        errors.append("scheduler merge target differs")
    if scheduler.get("board_namespace") != BOARD_NAMESPACE:
        errors.append("scheduler board namespace differs")
    if scheduler.get("task_prefix") != "LFP2-" or scheduler.get("goal_prefix") != "LFP2-G":
        errors.append("scheduler task/goal prefix differs")
    if scheduler.get("max_lanes") != 4 or scheduler.get("strict_task_sharding") is not False:
        errors.append("scheduler must use four dynamic work-stealing lanes")
    if scheduler.get("objective_refill_enabled") is not True or scheduler.get("codebase_refill_enabled") is not False:
        errors.append("scheduler refill mode differs")
    if scheduler.get("objective_goal_refinement_enabled") is not False:
        errors.append("static objective heap must disable goal refinement")
    runtime_paths = scheduler.get("runtime_paths")
    if not isinstance(runtime_paths, Mapping) or runtime_paths.get("root") != RUNTIME_ROOT:
        errors.append("scheduler v2 runtime root differs")
    if isinstance(runtime_paths, Mapping) and any(
        str(value).startswith("data/agent_supervisor/ipfs_datasets_logic_family_parser/")
        for value in runtime_paths.values()
    ):
        errors.append("v2 runtime overlaps Wave-1 runtime")
    source = scheduler.get("source_binding")
    if not isinstance(source, Mapping) or source.get("accelerator_required_ancestor") != PREDECESSOR_ACCELERATOR_COMMIT or source.get("accelerator_required_branch") != MERGE_TARGET_BRANCH or source.get("ipfs_datasets_planning_revision") != PREDECESSOR_DATASETS_COMMIT:
        errors.append("scheduler source binding differs from v2 predecessor seal")
    refill = scheduler.get("refill_policy")
    derived = refill.get("derived_refill") if isinstance(refill, Mapping) else None
    expected_refill = {
        "max_goals_per_epoch": 8,
        "max_tasks_per_epoch": 24,
        "min_open_tasks": 8,
        "max_open_tasks": 48,
        "max_refinement_depth": 3,
        "max_unchanged_failure_retries": 2,
        "cooldown_seconds": 3600,
        "mutate_seed_board": False,
        "mutate_seed_objectives": False,
    }
    if derived != expected_refill:
        errors.append("scheduler derived refill policy differs")
    try:
        board = load_configured_board(SCHEDULER_PATH, repo_root=REPO_ROOT)
        plan = configured_board_launch_plan(
            board,
            implement=True,
            detach=True,
            duration_seconds=300,
            stamp="20260809T000000Z",
        )
    except (ConfiguredBoardError, OSError, ValueError) as exc:
        errors.append(f"scheduler loader/renderer rejected config: {type(exc).__name__}: {exc}")
        return
    if plan.get("environment") != EXPECTED_ENVIRONMENT:
        errors.append("rendered provider environment differs from quota-only route")
    common = _common_args(plan)
    for flag in (
        "--objective-refill-scan",
        "--no-objective-goal-refinement",
        "--no-objective-goal-completion-reconcile",
        "--no-objective-goal-migration",
        "--no-objective-task-janitor",
    ):
        if common.count(flag) != 1:
            errors.append(f"rendered launch must contain exactly one {flag}")
    for forbidden in ("--strict-task-sharding", "--codebase-refill-scan"):
        if forbidden in common:
            errors.append(f"rendered launch unexpectedly contains {forbidden}")


def validate_all() -> dict[str, object]:
    errors: list[str] = []
    for path in (PLAN_PATH, OBJECTIVE_PATH, TODO_PATH, SCHEDULER_PATH):
        if not path.is_file():
            errors.append(f"missing control file: {path.relative_to(REPO_ROOT)}")
    if errors:
        return {"valid": False, "errors": errors}
    try:
        scheduler = json.loads(SCHEDULER_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [f"scheduler unreadable: {exc}"]}
    plan_text = PLAN_PATH.read_text(encoding="utf-8")
    objective_text = OBJECTIVE_PATH.read_text(encoding="utf-8")
    todo_text = TODO_PATH.read_text(encoding="utf-8")
    seed_digest = _seed_digest(todo_text)
    if SEALED_SEED_DEFINITION_SHA256 != "TO_BE_FILLED" and seed_digest != SEALED_SEED_DEFINITION_SHA256:
        errors.append("Wave-2 seed task definition differs from sealed digest")
    _validate_plan(plan_text, errors)
    _validate_predecessor(scheduler, errors)
    _validate_goals(objective_text, scheduler, errors)
    task_report = _validate_tasks(todo_text, errors)
    _validate_scheduler(scheduler, errors)
    return {
        "schema": "ipfs_accelerate_py/ipfs-datasets-logic-family-parser-v2-preflight@1",
        "valid": not errors,
        "errors": errors,
        "board_namespace": BOARD_NAMESPACE,
        "plan_path": str(PLAN_PATH),
        "objective_path": str(OBJECTIVE_PATH),
        "todo_path": str(TODO_PATH),
        "scheduler_path": str(SCHEDULER_PATH),
        "plan_sha256": _sha256(PLAN_PATH),
        "objective_sha256": _sha256(OBJECTIVE_PATH),
        "todo_sha256": _sha256(TODO_PATH),
        "seed_definition_sha256": seed_digest,
        "seed_task_count": len(TASK_IDS),
        "goal_count": len(GOAL_IDS),
        "terminal_task_id": TERMINAL_TASK,
        "root_goal_ids": ["LFP2-G000"],
        "predecessor_accelerator_commit": PREDECESSOR_ACCELERATOR_COMMIT,
        "predecessor_datasets_commit": PREDECESSOR_DATASETS_COMMIT,
        **task_report,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-all", action="store_true")
    parser.parse_args(argv)
    report = validate_all()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
