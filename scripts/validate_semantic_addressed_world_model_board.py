#!/usr/bin/env python3
"""Deterministic, fail-closed validator for the SAWM R2 control program.

The Markdown documents are immutable operator inputs, never task-completion
authority.  This validator checks their closed structure and the scheduler
binding only; accepted task state remains in the datasets-authoritative
DuckDB store reached through the current Quack owner.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BOARD_NAMESPACE = "semantic-addressed-world-model-v1"
PLAN_REVISION = "SAWM-PLAN-R2"
ROOT_GOAL = "SAWM-G000"
TASK_IDS = tuple(f"SAWM-{index:03d}" for index in range(45))
GOAL_IDS = (
    "SAWM-G000",
    "SAWM-G010", "SAWM-G011", "SAWM-G012", "SAWM-G013",
    "SAWM-G020", "SAWM-G021", "SAWM-G022", "SAWM-G023",
    "SAWM-G030", "SAWM-G031", "SAWM-G032", "SAWM-G033",
    "SAWM-G040", "SAWM-G041", "SAWM-G042", "SAWM-G043",
    "SAWM-G050", "SAWM-G051", "SAWM-G052", "SAWM-G053", "SAWM-G054",
    "SAWM-G060", "SAWM-G061", "SAWM-G062", "SAWM-G063",
    "SAWM-G070", "SAWM-G071", "SAWM-G072",
)

PLAN_PATH = REPO_ROOT / "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md"
OBJECTIVES_PATH = REPO_ROOT / "docs/architecture/semantic_addressed_world_model.objectives.md"
TODO_PATH = REPO_ROOT / "docs/architecture/semantic_addressed_world_model.todo.md"
INVENTORY_ROOT = REPO_ROOT / "docs/architecture/semantic_addressed_world_model_inventory"
SEAL_PATH = REPO_ROOT / "config/semantic_addressed_world_model_dependencies.seal.json"
SCHEDULER_PATH = REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"
BENCHMARK_PATH = REPO_ROOT / "benchmarks/agent_supervisor/semantic_addressed_world_model/benchmark_freeze.json"

INVENTORY_PATHS = tuple(
    INVENTORY_ROOT / name
    for name in (
        "repository_baseline.json",
        "authority_matrix.json",
        "overlap_gap_matrix.json",
        "identity_inventory.json",
        "interface_inventory.json",
        "dependency_graph.json",
        "capability_matrix.json",
        "rollout_baseline.json",
    )
)

CONTROL_RELATIVE_PATHS = (
    ".gitignore",
    "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md",
    "docs/architecture/semantic_addressed_world_model.objectives.md",
    "docs/architecture/semantic_addressed_world_model.todo.md",
    *(path.relative_to(REPO_ROOT).as_posix() for path in INVENTORY_PATHS),
    "config/semantic_addressed_world_model_dependencies.seal.json",
    "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
    "scripts/validate_semantic_addressed_world_model_dependencies.py",
    "scripts/validate_semantic_addressed_world_model_board.py",
    "scripts/materialize_semantic_addressed_world_model_program.py",
    "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
    "test/api/semantic_world/test_semantic_addressed_world_model_board.py",
    "benchmarks/agent_supervisor/semantic_addressed_world_model/benchmark_freeze.json",
)

GOAL_PARENT = {
    "SAWM-G000": "",
    "SAWM-G010": "SAWM-G000",
    "SAWM-G011": "SAWM-G010", "SAWM-G012": "SAWM-G010", "SAWM-G013": "SAWM-G010",
    "SAWM-G020": "SAWM-G000",
    "SAWM-G021": "SAWM-G020", "SAWM-G022": "SAWM-G020", "SAWM-G023": "SAWM-G020",
    "SAWM-G030": "SAWM-G000",
    "SAWM-G031": "SAWM-G030", "SAWM-G032": "SAWM-G030", "SAWM-G033": "SAWM-G030",
    "SAWM-G040": "SAWM-G000",
    "SAWM-G041": "SAWM-G040", "SAWM-G042": "SAWM-G040", "SAWM-G043": "SAWM-G040",
    "SAWM-G050": "SAWM-G000",
    "SAWM-G051": "SAWM-G050", "SAWM-G052": "SAWM-G050", "SAWM-G053": "SAWM-G050", "SAWM-G054": "SAWM-G050",
    "SAWM-G060": "SAWM-G000",
    "SAWM-G061": "SAWM-G060", "SAWM-G062": "SAWM-G060", "SAWM-G063": "SAWM-G060",
    "SAWM-G070": "SAWM-G000",
    "SAWM-G071": "SAWM-G070", "SAWM-G072": "SAWM-G070",
}

TASK_FIELDS = (
    "stable task id", "status", "completion", "completion mode",
    "is schedulable", "review only", "priority", "track", "depends on",
    "dependencies json", "goal id", "parent goal id", "subgoal id",
    "owning repository", "exact inputs", "outputs", "outputs json",
    "predicted files", "predicted files json", "predicted symbols",
    "public interfaces", "interfaces", "preconditions", "declared effects", "validation",
    "validation commands json", "evidence requirements", "acceptance",
    "conflict policy", "context budget tokens", "no-model route",
    "model fallback", "rollout mode", "protected paths", "limitations",
    "bundle", "parallel lane", "resource class", "implementation stage",
    "implementation timeout seconds", "provider role", "network policy",
    "risk class", "write scope", "external effect scope", "prohibited effects",
    "rollback or compensation procedure", "board namespace", "plan revision",
)

GOAL_FIELDS = (
    "stable goal id", "status", "parent", "parent goal ids json", "depends on",
    "dependencies json", "child goal ids json", "fib priority", "priority",
    "track", "bundle", "parallel lane", "resource class", "goal",
    "refinement", "producing tasks", "evidence", "evidence requirements json",
    "evidence criteria", "outputs", "predicted files", "predicted files json",
    "interfaces", "validation", "acceptance", "gap tasks", "rollout constraint",
    "authority constraint", "conflict policy",
)

ALLOWED_TASK_STATUSES = frozenset({"todo", "completed"})
ALLOWED_GOAL_STATUSES = frozenset(
    {"open", "active", "reopened", "provisionally_complete", "analysis_inconclusive"}
)
ROLLOUT_ORDER = ("bootstrap", "shadow_write", "shadow_read", "guarded", "required")
VALIDATION_PREFIX = (
    "PYTHONPATH=ipfs_datasets_py:ipfs_kit_py:. "
    "/home/barberb/.local/bin/python "
)
LEARNED_TASKS = frozenset(f"SAWM-{index:03d}" for index in range(25, 32))
REQUIRED_MODE_TASKS = frozenset(f"SAWM-{index:03d}" for index in range(38, 45))


@dataclass(frozen=True)
class Card:
    identifier: str
    title: str
    metadata: Mapping[str, str]


def _reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"{path.relative_to(REPO_ROOT)} must contain an object")
    return value


def _normalize_field(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower().replace("_", " "))


def _parse_cards(path: Path, *, goal: bool) -> tuple[Card, ...]:
    text = path.read_text(encoding="utf-8")
    identity = r"SAWM-G\d{3}" if goal else r"SAWM-\d{3}"
    matches = list(re.finditer(rf"^## ({identity})\b\s*(?:[-—:]\s*)?(.*)$", text, re.MULTILINE))
    cards: list[Card] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.end():end]
        metadata: dict[str, str] = {}
        current = ""
        for raw in block.splitlines():
            item = re.match(r"^\s*-\s+([^:]+):\s*(.*)$", raw)
            if item:
                current = _normalize_field(item.group(1))
                if current in metadata:
                    raise ValueError(f"{match.group(1)} has duplicate field {current!r}")
                metadata[current] = item.group(2).strip()
            elif current and raw.startswith(("  ", "\t")) and raw.strip():
                metadata[current] = (metadata[current] + " " + raw.strip()).strip()
            elif raw.strip():
                current = ""
        cards.append(Card(match.group(1), match.group(2).strip(), metadata))
    return tuple(cards)


def _json_list(card: Card, field: str, errors: list[str]) -> list[Any]:
    raw = card.metadata.get(field, "")
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicates)
    except (ValueError, json.JSONDecodeError) as exc:
        errors.append(f"{card.identifier}: {field} is invalid JSON: {exc}")
        return []
    if not isinstance(value, list):
        errors.append(f"{card.identifier}: {field} must be a JSON array")
        return []
    return value


def _bool(value: str) -> bool | None:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def _csv(value: str) -> tuple[str, ...]:
    if value.strip().lower() in {"", "none", "[]", "n/a"}:
        return ()
    return tuple(item.strip() for item in re.split(r"[,;]", value) if item.strip())


def _safe_relative(value: str) -> bool:
    text = value.strip().replace("\\", "/")
    path = PurePosixPath(text)
    return bool(
        text
        and not path.is_absolute()
        and ".." not in path.parts
        and "\x00" not in text
        and not any(character in text for character in "*?[]{}")
    )


def _paths(items: Iterable[Any], *, card: Card, field: str, errors: list[str]) -> tuple[str, ...]:
    selected: list[str] = []
    for item in items:
        if isinstance(item, str):
            path = item.strip()
        elif isinstance(item, Mapping):
            path = str(item.get("path") or "").strip()
        else:
            errors.append(f"{card.identifier}: {field} entry must be string or object")
            continue
        if not _safe_relative(path):
            errors.append(f"{card.identifier}: {field} path is not exact and contained: {path!r}")
            continue
        selected.append(path)
    if len(selected) != len(set(selected)):
        errors.append(f"{card.identifier}: {field} contains duplicate exact paths")
    return tuple(selected)


def _acyclic(adjacency: Mapping[str, Iterable[str]]) -> tuple[bool, tuple[str, ...]]:
    visiting: set[str] = set()
    visited: set[str] = set()
    cycle: list[str] = []

    def visit(node: str, trail: tuple[str, ...]) -> bool:
        if node in visiting:
            cycle.extend((*trail, node))
            return False
        if node in visited:
            return True
        visiting.add(node)
        for dependency in adjacency.get(node, ()):
            if dependency in adjacency and not visit(dependency, (*trail, node)):
                return False
        visiting.remove(node)
        visited.add(node)
        return True

    valid = all(visit(node, ()) for node in adjacency if node not in visited)
    return valid, tuple(cycle)


def _depends_transitively(adjacency: Mapping[str, tuple[str, ...]], task: str, dependency: str) -> bool:
    pending = list(adjacency.get(task, ()))
    seen: set[str] = set()
    while pending:
        item = pending.pop()
        if item == dependency:
            return True
        if item in seen:
            continue
        seen.add(item)
        pending.extend(adjacency.get(item, ()))
    return False


def _append(checks: list[dict[str, Any]], errors: list[str], name: str, passed: bool, detail: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})
    if not passed:
        errors.append(f"{name}: {detail}")


def _repository_for_path(path: str) -> str:
    if path.startswith("ipfs_datasets_py/"):
        return "ipfs_datasets_py"
    if path.startswith("ipfs_kit_py/"):
        return "ipfs_kit_py"
    return "ipfs_accelerate_py"


def validate_program(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []

    required = (
        root / PLAN_PATH.relative_to(REPO_ROOT),
        root / OBJECTIVES_PATH.relative_to(REPO_ROOT),
        root / TODO_PATH.relative_to(REPO_ROOT),
        root / SEAL_PATH.relative_to(REPO_ROOT),
        root / SCHEDULER_PATH.relative_to(REPO_ROOT),
        root / BENCHMARK_PATH.relative_to(REPO_ROOT),
        *(root / path.relative_to(REPO_ROOT) for path in INVENTORY_PATHS),
        *(root / path for path in CONTROL_RELATIVE_PATHS if path.startswith(("scripts/", "test/", "benchmarks/"))),
    )
    missing = sorted(path.relative_to(root).as_posix() for path in required if not path.is_file())
    _append(checks, errors, "control_files_present", not missing, missing)
    if missing:
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/semantic-addressed-world-model-board-validation@1",
            "valid": False,
            "board_namespace": BOARD_NAMESPACE,
            "plan_revision": PLAN_REVISION,
            "errors": errors,
            "warnings": warnings,
            "checks": checks,
        }

    try:
        tasks = _parse_cards(root / TODO_PATH.relative_to(REPO_ROOT), goal=False)
        goals = _parse_cards(root / OBJECTIVES_PATH.relative_to(REPO_ROOT), goal=True)
        config = _load_json(root / SCHEDULER_PATH.relative_to(REPO_ROOT))
        seal = _load_json(root / SEAL_PATH.relative_to(REPO_ROOT))
        benchmark = _load_json(root / BENCHMARK_PATH.relative_to(REPO_ROOT))
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        _append(checks, errors, "control_documents_parse", False, f"{type(exc).__name__}: {exc}")
        return {
            "schema": "ipfs_accelerate_py/agent-supervisor/semantic-addressed-world-model-board-validation@1",
            "valid": False,
            "board_namespace": BOARD_NAMESPACE,
            "plan_revision": PLAN_REVISION,
            "errors": errors,
            "warnings": warnings,
            "checks": checks,
        }

    task_ids = tuple(card.identifier for card in tasks)
    goal_ids = tuple(card.identifier for card in goals)
    _append(checks, errors, "task_population", task_ids == TASK_IDS, {"expected": TASK_IDS, "observed": task_ids})
    _append(checks, errors, "task_ids_unique", len(set(task_ids)) == len(task_ids), sorted(k for k, v in Counter(task_ids).items() if v != 1))
    _append(checks, errors, "goal_population", goal_ids == GOAL_IDS, {"expected": GOAL_IDS, "observed": goal_ids})
    _append(checks, errors, "goal_ids_unique", len(set(goal_ids)) == len(goal_ids), sorted(k for k, v in Counter(goal_ids).items() if v != 1))

    missing_task_fields = {
        card.identifier: [field for field in TASK_FIELDS if field not in card.metadata]
        for card in tasks
        if any(field not in card.metadata for field in TASK_FIELDS)
    }
    missing_goal_fields = {
        card.identifier: [field for field in GOAL_FIELDS if field not in card.metadata]
        for card in goals
        if any(field not in card.metadata for field in GOAL_FIELDS)
    }
    _append(checks, errors, "task_fields_closed", not missing_task_fields, missing_task_fields)
    _append(checks, errors, "goal_fields_closed", not missing_goal_fields, missing_goal_fields)

    goal_dependencies: dict[str, tuple[str, ...]] = {}
    goal_structure_errors: list[str] = []
    for card in goals:
        meta = card.metadata
        if meta.get("stable goal id") != card.identifier:
            goal_structure_errors.append(f"{card.identifier}: stable goal id mismatch")
        if meta.get("status", "").lower() not in ALLOWED_GOAL_STATUSES:
            goal_structure_errors.append(f"{card.identifier}: closed goal status violation")
        parent = meta.get("parent", "").strip()
        if parent.lower() in {"none", "null", "root", ""}:
            parent = ""
        if parent != GOAL_PARENT.get(card.identifier, "unknown"):
            goal_structure_errors.append(
                f"{card.identifier}: parent {parent!r} != {GOAL_PARENT.get(card.identifier)!r}"
            )
        parent_json = [str(item) for item in _json_list(card, "parent goal ids json", goal_structure_errors)]
        expected_parent_json = [] if not parent else [parent]
        if parent_json != expected_parent_json:
            goal_structure_errors.append(f"{card.identifier}: parent goal IDs JSON mismatch")
        deps = tuple(str(item) for item in _json_list(card, "dependencies json", goal_structure_errors))
        if set(deps) != set(_csv(meta.get("depends on", ""))):
            goal_structure_errors.append(f"{card.identifier}: dependency text/JSON mismatch")
        if any(dep not in GOAL_IDS or dep == card.identifier for dep in deps):
            goal_structure_errors.append(f"{card.identifier}: invalid goal dependency")
        goal_dependencies[card.identifier] = deps
    goal_acyclic, goal_cycle = _acyclic(goal_dependencies)
    if not goal_acyclic:
        goal_structure_errors.append("goal dependency cycle: " + " -> ".join(goal_cycle))
    _append(checks, errors, "goal_hierarchy_and_dependencies", not goal_structure_errors, goal_structure_errors)

    task_dependencies: dict[str, tuple[str, ...]] = {}
    task_structure_errors: list[str] = []
    task_outputs: dict[str, tuple[str, ...]] = {}
    task_predicted: dict[str, tuple[str, ...]] = {}
    for card in tasks:
        meta = card.metadata
        if meta.get("stable task id") != card.identifier:
            task_structure_errors.append(f"{card.identifier}: stable task id mismatch")
        status = meta.get("status", "").lower()
        if status not in ALLOWED_TASK_STATUSES:
            task_structure_errors.append(f"{card.identifier}: status {status!r} is outside the closed initial set")
        expected_status = "completed" if card.identifier == "SAWM-000" else "todo"
        if status != expected_status:
            task_structure_errors.append(f"{card.identifier}: initial status must be {expected_status}")
        deps = tuple(str(item) for item in _json_list(card, "dependencies json", task_structure_errors))
        if set(deps) != set(_csv(meta.get("depends on", ""))):
            task_structure_errors.append(f"{card.identifier}: dependency text/JSON mismatch")
        if any(dep not in TASK_IDS or dep == card.identifier for dep in deps):
            task_structure_errors.append(f"{card.identifier}: invalid task dependency")
        task_dependencies[card.identifier] = deps
        for field in ("goal id", "parent goal id", "subgoal id"):
            if meta.get(field) not in GOAL_IDS:
                task_structure_errors.append(f"{card.identifier}: {field} is not a current goal")
        if meta.get("board namespace") != BOARD_NAMESPACE or meta.get("plan revision") != PLAN_REVISION:
            task_structure_errors.append(f"{card.identifier}: namespace or plan revision mismatch")
        task_outputs[card.identifier] = _paths(
            _json_list(card, "outputs json", task_structure_errors),
            card=card, field="outputs json", errors=task_structure_errors,
        )
        task_predicted[card.identifier] = _paths(
            _json_list(card, "predicted files json", task_structure_errors),
            card=card, field="predicted files json", errors=task_structure_errors,
        )
    task_acyclic, task_cycle = _acyclic(task_dependencies)
    if not task_acyclic:
        task_structure_errors.append("task dependency cycle: " + " -> ".join(task_cycle))
    if task_dependencies.get("SAWM-000"):
        task_structure_errors.append("SAWM-000 must not depend on an implementation task")
    if task_dependencies.get("SAWM-001") != ("SAWM-000",):
        task_structure_errors.append("SAWM-001 must be the sole initial implementation frontier")
    _append(checks, errors, "task_dependencies_and_goal_bindings", not task_structure_errors, task_structure_errors)

    rollout_errors: list[str] = []
    rollout_chain = (("SAWM-035", "SAWM-034"), ("SAWM-036", "SAWM-035"), ("SAWM-037", "SAWM-036"))
    for task, dependency in rollout_chain:
        if not _depends_transitively(task_dependencies, task, dependency):
            rollout_errors.append(f"{task} must depend transitively on {dependency}")
    for task in REQUIRED_MODE_TASKS:
        card = tasks[TASK_IDS.index(task)] if task in task_ids else None
        if card is not None and card.metadata.get("rollout mode", "").lower() != "required":
            rollout_errors.append(f"{task}: rollout mode must be required")
        if task != "SAWM-038" and not _depends_transitively(task_dependencies, task, "SAWM-037"):
            rollout_errors.append(f"{task} must transitively follow required-mode activation")
    if not _depends_transitively(task_dependencies, "SAWM-044", "SAWM-043"):
        rollout_errors.append("SAWM-044 must transitively follow the self-hosted capstone")
    _append(checks, errors, "rollout_ordering", not rollout_errors, rollout_errors)

    completion_errors: list[str] = []
    for card in tasks:
        meta = card.metadata
        if card.identifier == "SAWM-000":
            if meta.get("completion mode", "").lower() != "trusted_manual":
                completion_errors.append("SAWM-000 completion mode must be trusted_manual")
            if _bool(meta.get("is schedulable", "")) is not False or _bool(meta.get("review only", "")) is not True:
                completion_errors.append("SAWM-000 must be non-schedulable operator review")
            operator_text = " ".join(
                meta.get(field, "") for field in ("provider role", "evidence requirements", "acceptance")
            ).lower()
            if "operator" not in operator_text or "current" not in operator_text or "receipt" not in operator_text:
                completion_errors.append("SAWM-000 must require operator current-tree receipt evidence")
        else:
            if meta.get("completion mode", "").lower() != "automatic":
                completion_errors.append(f"{card.identifier}: completion mode must be automatic")
            if _bool(meta.get("is schedulable", "")) is not True or _bool(meta.get("review only", "")) is not False:
                completion_errors.append(f"{card.identifier}: implementation task schedulability mismatch")
        if not meta.get("evidence requirements", "").strip() or not meta.get("acceptance", "").strip():
            completion_errors.append(f"{card.identifier}: evidence/acceptance is empty")
    if config.get("authority_policy", {}).get("markdown_is_task_completion_authority") is not False:
        completion_errors.append("scheduler must make Markdown non-authoritative for completion")
    _append(checks, errors, "completion_authority", not completion_errors, completion_errors)

    validation_errors: list[str] = []
    forbidden_validation = re.compile(r"(?:^|\s)(?:pip|pip3)\s+install\b|\bcurl\b|\bwget\b|git\s+clean|reset\s+--hard", re.I)
    for card in tasks:
        commands = _json_list(card, "validation commands json", validation_errors)
        if not commands:
            validation_errors.append(f"{card.identifier}: validation command list is empty")
        for command in commands:
            if not isinstance(command, str) or not command.startswith(VALIDATION_PREFIX):
                validation_errors.append(f"{card.identifier}: validation is not bound to the current hermetic interpreter")
            elif forbidden_validation.search(command):
                validation_errors.append(f"{card.identifier}: validation contains forbidden installer/network/destructive command")
        if not card.metadata.get("validation", "").strip():
            validation_errors.append(f"{card.identifier}: human-readable validation is empty")
        network = card.metadata.get("network policy", "").lower()
        if not any(term in network for term in ("no network", "network disabled", "offline", "deny")):
            validation_errors.append(f"{card.identifier}: ordinary validation must be network-disabled")
    _append(checks, errors, "current_validation_commands", not validation_errors, validation_errors)

    ownership_errors: list[str] = []
    owners_by_path: dict[str, list[str]] = defaultdict(list)
    protected = set(CONTROL_RELATIVE_PATHS)
    for card in tasks:
        owner = card.metadata.get("owning repository", "").strip().lower()
        normalized_owner = "operator" if "operator" in owner else next(
            (candidate for candidate in ("ipfs_datasets_py", "ipfs_kit_py", "ipfs_accelerate_py") if candidate in owner),
            owner,
        )
        if normalized_owner not in {"operator", "ipfs_datasets_py", "ipfs_kit_py", "ipfs_accelerate_py"}:
            ownership_errors.append(f"{card.identifier}: unknown repository owner {owner!r}")
        for path in task_outputs.get(card.identifier, ()):
            owners_by_path[path].append(card.identifier)
            if path in protected and card.identifier != "SAWM-000":
                ownership_errors.append(f"{card.identifier}: implementation task owns protected path {path}")
            actual_owner = _repository_for_path(path)
            if normalized_owner not in {"operator", actual_owner}:
                ownership_errors.append(
                    f"{card.identifier}: {path} belongs to {actual_owner}, not {normalized_owner}"
                )
        if card.identifier != "SAWM-000":
            for path in task_predicted.get(card.identifier, ()):
                if path in protected:
                    ownership_errors.append(f"{card.identifier}: predicts a protected-path mutation {path}")
    for path, task_list in sorted(owners_by_path.items()):
        if len(task_list) < 2:
            continue
        ordered = all(
            _depends_transitively(task_dependencies, later, earlier)
            or _depends_transitively(task_dependencies, earlier, later)
            for index, later in enumerate(task_list)
            for earlier in task_list[:index]
        )
        policies = " ".join(tasks[TASK_IDS.index(task)].metadata.get("conflict policy", "") for task in task_list).lower()
        if not ordered or not any(term in policies for term in ("serial", "successor", "integration owner")):
            ownership_errors.append(f"{path}: duplicate output owners are not explicitly serialized: {task_list}")
    _append(checks, errors, "exact_output_ownership_and_protection", not ownership_errors, ownership_errors)

    authority_errors: list[str] = []
    forbidden_positive = re.compile(
        r"\b(?:model|worker|prediction|retrieval result|ann result)\s+(?:may|can|shall|will)\s+"
        r"(?:self[- ]?)?(?:approve|accept|authorize|complete|prove)\b",
        re.I,
    )
    for card in tasks:
        rendered = " ".join(card.metadata.values())
        if forbidden_positive.search(rendered):
            authority_errors.append(f"{card.identifier}: declares model/worker self-authority")
        if not card.metadata.get("prohibited effects", "").strip():
            authority_errors.append(f"{card.identifier}: prohibited effects are empty")
        fallback = card.metadata.get("model fallback", "").lower()
        if fallback and not any(
            term in fallback
            for term in (
                "proposal", "untrusted", "residual", "none", "unavailable",
                "abstain", "forbidden", "bounded",
            )
        ):
            authority_errors.append(f"{card.identifier}: model fallback is not proposal/residual bounded")
    authority_policy = config.get("authority_policy")
    if not isinstance(authority_policy, Mapping):
        authority_errors.append("scheduler authority_policy is missing")
    else:
        required_false = (
            "markdown_is_task_completion_authority", "ann_or_neural_result_is_authoritative",
            "model_can_self_approve", "worker_can_self_approve",
            "model_can_create_proof_authority", "model_can_create_completion_authority",
            "automatic_file_fallback_from_quack", "raw_llm_sql_allowed",
            "implementation_provider_receives_state_credentials",
        )
        for field in required_false:
            if authority_policy.get(field) is not False:
                authority_errors.append(f"authority_policy.{field} must be false")
    _append(checks, errors, "no_self_approval_or_authority_weakening", not authority_errors, authority_errors)

    learned_errors: list[str] = []
    # Candidate/benchmark tasks are allowed to terminate typed-unavailable;
    # only the foundry/promotion task may admit a checkpoint.  Bind candidate
    # tasks to the corpus and frozen benchmark, then require the full promotion
    # vocabulary exactly where authority can actually be granted.
    for task_id in tuple(f"SAWM-{index:03d}" for index in range(25, 30)):
        if not _depends_transitively(task_dependencies, task_id, "SAWM-023"):
            learned_errors.append(f"{task_id}: does not transitively bind the admitted corpus")
        if not _depends_transitively(task_dependencies, task_id, "SAWM-024"):
            learned_errors.append(f"{task_id}: does not transitively bind the frozen held-out benchmark")
        card = tasks[TASK_IDS.index(task_id)]
        evidence = " ".join(card.metadata.get(field, "") for field in
                            ("preconditions", "evidence requirements", "acceptance", "limitations")).lower()
        if "calibration" not in evidence and "training_unavailable" not in evidence:
            learned_errors.append(f"{task_id}: lacks calibration or typed training unavailability")
        if not any(term in evidence for term in ("unavailable", "abstain", "ood")):
            learned_errors.append(f"{task_id}: lacks unavailable/OOD abstention")
    promotion = tasks[TASK_IDS.index("SAWM-030")]
    promotion_text = " ".join(promotion.metadata.get(field, "") for field in
                              ("preconditions", "evidence requirements", "acceptance", "limitations")).lower().replace("-", "_")
    for concept, alternatives in {
        "corpus": ("corpus", "training_unavailable"),
        "checkpoint": ("checkpoint", "training_unavailable"),
        "calibration": ("calibration", "training_unavailable"),
        "held_out": ("held_out", "held out", "training_unavailable"),
        "lineage": ("lineage", "training_unavailable"),
    }.items():
        if not any(term in promotion_text for term in alternatives):
            learned_errors.append(f"SAWM-030: missing promotion gate {concept}")
    if not _depends_transitively(task_dependencies, "SAWM-031", "SAWM-030"):
        learned_errors.append("SAWM-031 must consume the admitted foundry/promotion result")
    _append(checks, errors, "learned_capability_promotion_gates", not learned_errors, learned_errors)

    config_errors: list[str] = []
    if config.get("board_namespace") != BOARD_NAMESPACE or config.get("plan_revision") != PLAN_REVISION:
        config_errors.append("scheduler namespace/revision mismatch")
    projection = config.get("initial_projection") if isinstance(config.get("initial_projection"), Mapping) else {}
    if projection.get("task_count") != 45 or projection.get("goal_count") != 29:
        config_errors.append("initial projection population mismatch")
    if projection.get("completed_task_ids") != ["SAWM-000"] or projection.get("ready_task_ids") != ["SAWM-001"]:
        config_errors.append("initial projection frontier mismatch")
    if config.get("max_lanes") != 1:
        config_errors.append("one lane is required until sidecars are lane-scoped")
    provider = config.get("provider") if isinstance(config.get("provider"), Mapping) else {}
    expected_provider = {
        "primary_provider_id": "grok_cli", "primary_model_id": "grok-4.6",
        "fallback_provider_id": "codex", "fallback_model_id": "gpt-5.6-terra",
    }
    if any(provider.get(key) != value for key, value in expected_provider.items()):
        config_errors.append("ordered provider route mismatch")
    program = config.get("database_program") if isinstance(config.get("database_program"), Mapping) else {}
    if (
        program.get("authority_mode") != "quack"
        or program.get("task_source_kind") != "duckdb"
        or program.get("quack_endpoint") != "quack:127.0.0.1:45247"
        or program.get("endpoint_secret_handle") != "env://SAWM_QUACK_TOKEN"
        or program.get("failover_policy") != "fail_closed"
    ):
        config_errors.append("DuckDB + Quack authority binding mismatch")
    ducklake = config.get("ducklake_history_projection") if isinstance(config.get("ducklake_history_projection"), Mapping) else {}
    if not (
        ducklake.get("authority") is False
        and ducklake.get("scheduling_prerequisite") is False
        and ducklake.get("completion_prerequisite") is False
        and ducklake.get("load_local_only") is True
        and ducklake.get("install_or_network_forbidden") is True
    ):
        config_errors.append("DuckLake must be local-only and non-authoritative")
    if tuple(config.get("protected_paths") or ()) != CONTROL_RELATIVE_PATHS:
        config_errors.append("scheduler protected paths differ from the exact operator controls")
    expected_direction = {
        "ipfs_datasets_py": [], "ipfs_kit_py": [],
        "ipfs_accelerate_py": ["ipfs_datasets_py", "ipfs_kit_py"],
    }
    if config.get("package_dependency_direction") != expected_direction:
        config_errors.append("package dependency direction mismatch")
    if config.get("rollout_order") != list(ROLLOUT_ORDER):
        config_errors.append("closed rollout order mismatch")
    _append(checks, errors, "scheduler_authority_binding", not config_errors, config_errors)

    try:
        from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
            load_configured_board,
        )

        loaded = load_configured_board(
            root / SCHEDULER_PATH.relative_to(REPO_ROOT), repo_root=root
        )
        scheduler_load_detail: Any = {
            "namespace": loaded.board_namespace,
            "lanes": loaded.max_lanes,
            "database_authority": loaded.resolved_database_program().authority_mode,
        }
        scheduler_load_valid = True
    except Exception as exc:  # typed into deterministic report
        scheduler_load_valid = False
        scheduler_load_detail = f"{type(exc).__name__}: {exc}"
    _append(checks, errors, "current_scheduler_schema_load", scheduler_load_valid, scheduler_load_detail)

    benchmark_errors: list[str] = []
    if benchmark.get("board_namespace") != BOARD_NAMESPACE or benchmark.get("plan_revision") != PLAN_REVISION:
        benchmark_errors.append("benchmark namespace/revision mismatch")
    if benchmark.get("status") != "design_frozen" or benchmark.get("results_available") is not False:
        benchmark_errors.append("benchmark must be a result-free frozen design")
    ablations = benchmark.get("ablation_ladder")
    if not isinstance(ablations, list) or [item.get("id") for item in ablations if isinstance(item, Mapping)] != list("ABCDEFGHIJKL"):
        benchmark_errors.append("benchmark ablation ladder must contain A through L")
    floors = benchmark.get("safety_floors") if isinstance(benchmark.get("safety_floors"), Mapping) else {}
    if not floors or any(value != 0 for value in floors.values()):
        benchmark_errors.append("all frozen safety floors must be zero")
    if any(config.get("release_safety_floors", {}).get(key) != value for key, value in floors.items()):
        benchmark_errors.append("benchmark safety floors differ from scheduler release floors")
    _append(checks, errors, "benchmark_freeze", not benchmark_errors, benchmark_errors)

    if seal.get("board_namespace") != BOARD_NAMESPACE or seal.get("plan_revision") != PLAN_REVISION:
        _append(checks, errors, "dependency_seal_binding", False, "seal namespace/revision mismatch")
    else:
        _append(checks, errors, "dependency_seal_binding", True, seal.get("schema"))

    plan_text = (root / PLAN_PATH.relative_to(REPO_ROOT)).read_text(encoding="utf-8").lower()
    required_plan_terms = (
        "exact identity", "semantic projection", "typed semantic relation", "authority",
        "physical ipld/merkle", "logical program graph", "duckdb", "quack", "ducklake",
        "bootstrap", "shadow_write", "shadow_read", "guarded", "required",
        "tactician", "hammer", "cegis", "contextcompiler", "world root",
    )
    absent_terms = [term for term in required_plan_terms if term not in plan_text]
    _append(checks, errors, "architecture_control_terms", not absent_terms, absent_terms)

    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/semantic-addressed-world-model-board-validation@1",
        "valid": not errors,
        "board_namespace": BOARD_NAMESPACE,
        "plan_revision": PLAN_REVISION,
        "task_count": len(tasks),
        "goal_count": len(goals),
        "markdown_completion_is_authority": False,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-all", action="store_true", help="run the complete static board gate")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        report = validate_program(args.repo_root)
    except Exception as exc:  # fail closed while retaining machine-readable output
        report = {
            "schema": "ipfs_accelerate_py/agent-supervisor/semantic-addressed-world-model-board-validation@1",
            "valid": False,
            "board_namespace": BOARD_NAMESPACE,
            "plan_revision": PLAN_REVISION,
            "errors": [f"unhandled_validator_error: {type(exc).__name__}: {exc}"],
            "warnings": [],
            "checks": [],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("valid") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
