#!/usr/bin/env python3
"""Deterministic, side-effect-free validator for the LPC board.

``--check-all`` validates control files and the goal/task DAG.  Artifact
checks used by later tasks are separate flags so preflight can pass before
inventory exists.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PLAN_PATH = Path("docs/architecture/LOGIC_PLATFORM_CANONICALIZATION_PLAN.md")
OBJECTIVE_PATH = Path(
    "docs/architecture/logic_platform_canonicalization.objectives.md"
)
TODO_PATH = Path("docs/architecture/logic_platform_canonicalization.todo.md")
SCHEDULER_PATH = Path(
    "config/agent_supervisor_logic_platform_canonicalization_scheduler.json"
)
VALIDATOR_PATH = Path(
    "scripts/validate_logic_platform_canonicalization_board.py"
)
EMITTER_PATH = Path(
    "scripts/emit_logic_platform_canonicalization_formal_plan.py"
)

TASK_PREFIX = "LPC-"
GOAL_RE = re.compile(r"^## (LPC-G\d{3})\b", re.MULTILINE)
TASK_RE = re.compile(r"^## (LPC-\d{3})\b", re.MULTILINE)
FIELD_RE = re.compile(r"^- ([A-Za-z][A-Za-z0-9 ]*):[ \t]*(.*)$")
REQUIRED_GOAL_FIELDS = ("Status", "Goal", "Acceptance", "Bundle", "Conflict policy")
REQUIRED_TASK_FIELDS = (
    "Status",
    "Goal id",
    "Outputs",
    "Validation",
    "Conflict policy",
    "Acceptance",
)
INVENTORY_CATEGORIES = (
    "public_logic_api",
    "registry_generation",
    "family_profile_property_provider",
    "ast_or_typed_expression",
    "formalization_artifact",
    "domain_logic_slice",
    "backend_request",
    "provider_protocol",
    "translation_contract",
    "proof_plan",
    "receipt_and_evidence",
    "cache_key",
    "provider_matrix",
    "status_enum",
    "authority_enum",
    "boundedness_enum",
    "alias_table",
    "installer_mutation_boundary",
    "supervisor_import_into_datasets",
    "duplicate_supervisor_semantic_type",
    "mcp_cli_python_exposure",
    "compatibility_shim",
    "deprecated_module",
    "test_and_conformance_corpus",
)
FINAL_REPORT_SECTIONS = (
    "Exact source revisions inspected",
    "Current-state inventory",
    "Canonical ownership map",
    "APIs and schemas added or changed",
    "Registry and catalog migration",
    "Status and authority consolidation",
    "Provider-protocol migration",
    "Formalization and domain-slice improvements",
    "Proof tactician changes",
    "Cache and receipt changes",
    "Supervisor integration changes",
    "Compatibility adapters retained",
    "Deprecated surfaces",
    "Tests and commands run",
    "Test results",
    "Real-provider results",
    "Packaging results",
    "CI results",
    "Documentation changes",
    "Known unresolved gaps",
    "Explicit recommendation for the next work board",
)


def _read(relative: Path) -> str:
    return (REPO_ROOT / relative).read_text(encoding="utf-8")


def _blocks(text: str, pattern: re.Pattern[str]) -> dict[str, str]:
    matches = list(pattern.finditer(text))
    blocks: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks[match.group(1)] = text[match.start() : end]
    return blocks


def _fields(block: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in block.splitlines():
        match = FIELD_RE.match(raw_line)
        if match is None:
            continue
        values[match.group(1).strip()] = match.group(2).strip()
    return values


def _csv(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _check_control_files() -> list[str]:
    errors: list[str] = []
    for relative in (
        PLAN_PATH,
        OBJECTIVE_PATH,
        TODO_PATH,
        SCHEDULER_PATH,
        VALIDATOR_PATH,
        EMITTER_PATH,
    ):
        if not (REPO_ROOT / relative).is_file():
            errors.append(f"missing control file: {relative.as_posix()}")
    return errors


def _check_board() -> tuple[list[str], dict[str, object]]:
    errors: list[str] = []
    objectives = _read(OBJECTIVE_PATH)
    todo = _read(TODO_PATH)
    goals = _blocks(objectives, GOAL_RE)
    tasks = _blocks(todo, TASK_RE)
    if "LPC-G000" not in goals:
        errors.append("root goal LPC-G000 is missing")
    if "LPC-G010" not in goals:
        errors.append("inventory goal LPC-G010 is missing")
    if "LPC-000" not in tasks:
        errors.append("control task LPC-000 is missing")
    seal_fields = _fields(tasks.get("LPC-000", ""))
    if seal_fields.get("Status") != "completed":
        errors.append("LPC-000 must remain completed after operator seal")
    for output in _csv(seal_fields.get("Outputs", "")):
        if output.startswith("docs/architecture/") or output.startswith("config/") or output.startswith("scripts/"):
            errors.append("LPC-000 must not list protected control files as Outputs")

    goal_ids = set(goals)
    for goal_id, block in goals.items():
        fields = _fields(block)
        for name in REQUIRED_GOAL_FIELDS:
            if not fields.get(name):
                errors.append(f"{goal_id} missing field {name}")
        for dependency in _csv(fields.get("Depends on", "")):
            if dependency.startswith("LPC-G") and dependency not in goal_ids:
                errors.append(f"{goal_id} depends on unknown goal {dependency}")

    task_ids = set(tasks)
    inventory_outputs: dict[str, str] = {}
    for task_id, block in tasks.items():
        fields = _fields(block)
        for name in REQUIRED_TASK_FIELDS:
            if not fields.get(name):
                errors.append(f"{task_id} missing field {name}")
        goal_id = fields.get("Goal id", "")
        if goal_id and goal_id not in goal_ids:
            errors.append(f"{task_id} Goal id {goal_id} is not a declared goal")
        for dependency in _csv(fields.get("Depends on", "")):
            if dependency.startswith("LPC-") and dependency not in task_ids:
                errors.append(f"{task_id} depends on unknown task {dependency}")
        if fields.get("Goal id") == "LPC-G010" and task_id != "LPC-008":
            for output in _csv(fields.get("Outputs", "")):
                owner = inventory_outputs.get(output)
                if owner and owner != task_id:
                    errors.append(
                        f"inventory output {output} owned by both {owner} and {task_id}"
                    )
                inventory_outputs[output] = task_id

    scheduler = json.loads(_read(SCHEDULER_PATH))
    if scheduler.get("task_prefix") not in {TASK_PREFIX, f"## {TASK_PREFIX}"}:
        errors.append("scheduler task_prefix must be LPC-")
    if scheduler.get("merge_target_branch") != "agent/logic-platform-canonicalization":
        errors.append("scheduler merge_target_branch mismatch")
    config_rel = SCHEDULER_PATH.as_posix()
    if config_rel not in scheduler.get("protected_paths", []):
        errors.append("scheduler must protect its own path")

    detail = {
        "goal_count": len(goals),
        "task_count": len(tasks),
        "goals": sorted(goals),
        "tasks": sorted(tasks),
    }
    return errors, detail


def _check_inventory() -> list[str]:
    path = (
        REPO_ROOT
        / "data"
        / "agent_supervisor"
        / "logic_platform_canonicalization"
        / "inventory"
        / "inventory.json"
    )
    index = path.with_name("INDEX.md")
    errors: list[str] = []
    if not path.is_file():
        errors.append("inventory.json is missing")
        return errors
    if not index.is_file():
        errors.append("inventory INDEX.md is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"inventory.json is not JSON: {exc}"]
    if not isinstance(payload, dict):
        return ["inventory.json root must be an object"]
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        errors.append("inventory.json items must be a nonempty list")
        return errors
    seen_categories = {
        item.get("category")
        for item in items
        if isinstance(item, dict)
    }
    for category in INVENTORY_CATEGORIES:
        if category not in seen_categories:
            errors.append(f"inventory missing category {category}")
    return errors


def _check_note(relative: str, label: str) -> list[str]:
    path = REPO_ROOT / relative
    if not path.is_file():
        return [f"{label} is missing: {relative}"]
    if path.stat().st_size < 32:
        return [f"{label} is too small: {relative}"]
    return []


def _check_final_report() -> list[str]:
    path = (
        REPO_ROOT
        / "data"
        / "agent_supervisor"
        / "logic_platform_canonicalization"
        / "final_report.md"
    )
    if not path.is_file():
        return ["final report is missing"]
    text = path.read_text(encoding="utf-8")
    errors = []
    for section in FINAL_REPORT_SECTIONS:
        if section not in text:
            errors.append(f"final report missing section: {section}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-all", action="store_true")
    parser.add_argument("--check-inventory", action="store_true")
    parser.add_argument("--check-test-matrix", action="store_true")
    parser.add_argument("--check-ci", action="store_true")
    parser.add_argument("--check-docs", action="store_true")
    parser.add_argument("--check-final-report", action="store_true")
    args = parser.parse_args(argv)

    selected = any(
        (
            args.check_all,
            args.check_inventory,
            args.check_test_matrix,
            args.check_ci,
            args.check_docs,
            args.check_final_report,
        )
    )
    if not selected:
        args.check_all = True

    errors: list[str] = []
    detail: dict[str, object] = {}
    if args.check_all:
        errors.extend(_check_control_files())
        board_errors, detail = _check_board()
        errors.extend(board_errors)
    if args.check_inventory:
        errors.extend(_check_inventory())
    if args.check_test_matrix:
        errors.extend(
            _check_note(
                "data/agent_supervisor/logic_platform_canonicalization/notes/test_matrix.md",
                "test matrix",
            )
        )
    if args.check_ci:
        errors.extend(
            _check_note(
                "data/agent_supervisor/logic_platform_canonicalization/notes/packaging_ci.md",
                "packaging/CI note",
            )
        )
        errors.extend(
            _check_note(
                "data/agent_supervisor/logic_platform_canonicalization/notes/ci_lanes.md",
                "CI lanes note",
            )
        )
    if args.check_docs:
        errors.extend(
            _check_note(
                "data/agent_supervisor/logic_platform_canonicalization/notes/documentation.md",
                "documentation note",
            )
        )
    if args.check_final_report:
        errors.extend(_check_final_report())

    payload = {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/"
            "logic-platform-canonicalization-board-check@1"
        ),
        "valid": not errors,
        "errors": errors,
        "detail": detail,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
