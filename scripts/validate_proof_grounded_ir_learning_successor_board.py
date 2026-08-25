#!/usr/bin/env python3
"""Validate the PGIR successor board and its current-supervisor source binding."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "docs/architecture/proof_grounded_ir_learning/successor.todo.md"
SOURCE_BOARD = ROOT / "docs/architecture/proof_grounded_ir_learning/next.todo.md"
HISTORICAL_BOARD = ROOT / "docs/architecture/proof_grounded_ir_learning.todo.md"
OBJECTIVES = ROOT / "docs/architecture/proof_grounded_ir_learning.objectives.md"
FINAL_REPORT = ROOT / "docs/architecture/proof_grounded_ir_learning/final_report.md"
PLAN = ROOT / "docs/architecture/PROOF_GROUNDED_IR_LEARNING_SUCCESSOR_PLAN.md"
CONFIG = (
    ROOT / "config/agent_supervisor_proof_grounded_ir_learning_successor_scheduler.json"
)
PREDECESSOR_REPLAY = ROOT / "scripts/verify_proof_grounded_ir_learning_predecessor.py"
DATASETS = ROOT / "ipfs_datasets_py"

ACCELERATOR_BASE = "22173f9cf4f357ab20040024f87af53c1cd89c9a"
DATASETS_BASE = "c30ccbec997868b061c4cadac38d30468c46ea2d"
ACCELERATOR_PGIR_MERGE = "19eddcc83"
DATASETS_PGIR_MERGE = "09ed0f2f0"
DECISION_CID = "baguqeeraejs56hwzs3bqtgzoayrc2fxwgfnhcsxjthi4dh7gh64wptlkfhwa"

ANCHORS = ("PGIR-072", "PGIR-090", "PGIR-100", "PGIR-111")
SOURCE_OPEN_TASKS = tuple(f"PGIR-{number}" for number in range(200, 208))
CORRECTIVE_TASKS = ("PGIR-208", "PGIR-209", "PGIR-210")
OPEN_TASKS = (*SOURCE_OPEN_TASKS, *CORRECTIVE_TASKS)
EXPECTED_TASKS = (*ANCHORS, *OPEN_TASKS)
EXPECTED_DEPENDENCIES = {
    "PGIR-072": (),
    "PGIR-090": (),
    "PGIR-100": (),
    "PGIR-111": (),
    "PGIR-200": ("PGIR-111",),
    "PGIR-201": ("PGIR-200",),
    "PGIR-202": ("PGIR-201",),
    "PGIR-203": ("PGIR-202",),
    "PGIR-204": ("PGIR-202",),
    "PGIR-205": (
        "PGIR-200",
        "PGIR-201",
        "PGIR-202",
        "PGIR-203",
        "PGIR-204",
        "PGIR-208",
        "PGIR-209",
        "PGIR-210",
    ),
    "PGIR-206": ("PGIR-205",),
    "PGIR-207": ("PGIR-072", "PGIR-090", "PGIR-100", "PGIR-206"),
    "PGIR-208": ("PGIR-200", "PGIR-201", "PGIR-202"),
    "PGIR-209": ("PGIR-202", "PGIR-204"),
    "PGIR-210": ("PGIR-204", "PGIR-208"),
}
EXPECTED_GOALS = {
    "PGIR-072": "PGIR-G090",
    "PGIR-090": "PGIR-G100",
    "PGIR-100": "PGIR-G100",
    "PGIR-111": "PGIR-G110",
    "PGIR-200": "PGIR-G020",
    "PGIR-201": "PGIR-G020",
    "PGIR-202": "PGIR-G020",
    "PGIR-203": "PGIR-G050",
    "PGIR-204": "PGIR-G040",
    "PGIR-205": "PGIR-G030",
    "PGIR-206": "PGIR-G110",
    "PGIR-207": "PGIR-G110",
    "PGIR-208": "PGIR-G030",
    "PGIR-209": "PGIR-G040",
    "PGIR-210": "PGIR-G030",
}
BOARD_NAMESPACE = "proof-grounded-ir-learning-successor-v1"
ALLOWED_OPEN_STATUSES = {
    "todo",
    "queued",
    "proposed",
    "admitted",
    "ready",
    "in_progress",
    "running",
    "blocked",
    "completed",
    "failed",
    "rejected",
    "quarantined",
}
TASK_RE = re.compile(
    r"^## (?P<task_id>PGIR-\d{3}) (?P<title>[^\n]+)\n(?P<body>.*?)(?=^## PGIR-|\Z)",
    re.MULTILINE | re.DOTALL,
)
FIELD_RE = re.compile(r"^- (?P<name>[^:\n]+):\s*(?P<value>.*)$", re.MULTILINE)


def run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args),
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


def parse_board(path: Path) -> dict[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    tasks: dict[str, dict[str, Any]] = {}
    for match in TASK_RE.finditer(text):
        task_id = match.group("task_id")
        if task_id in tasks:
            raise ValueError(f"duplicate task ID in {path}: {task_id}")
        fields = {
            field.group("name")
            .strip()
            .casefold()
            .replace(" ", "_"): field.group("value")
            .strip()
            for field in FIELD_RE.finditer(match.group("body"))
        }
        tasks[task_id] = {
            "title": match.group("title").strip(),
            "fields": fields,
        }
    return tasks


def split_csv(value: str) -> tuple[str, ...]:
    selected = value.strip()
    if not selected or selected.casefold() == "none":
        return ()
    return tuple(item.strip() for item in selected.split(",") if item.strip())


class Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.errors: list[str] = []

    def check(self, name: str, passed: bool, detail: Any) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            self.errors.append(f"{name}: {detail}")

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "proof-grounded-ir-learning-successor-validation@1",
            "board_namespace": "proof-grounded-ir-learning-successor-v1",
            "valid": not self.errors,
            "errors": self.errors,
            "checks": self.checks,
        }


def validate_anchors(report: Report) -> None:
    try:
        historical = parse_board(HISTORICAL_BOARD)
    except (OSError, UnicodeError, ValueError) as exc:
        report.check("historical_board_readable", False, str(exc))
        return
    missing = [task_id for task_id in ANCHORS if task_id not in historical]
    report.check("historical_anchors_present", not missing, missing)
    bad_status = {
        task_id: historical[task_id]["fields"].get("status")
        for task_id in ANCHORS
        if task_id in historical
        and historical[task_id]["fields"].get("status", "").casefold() != "completed"
    }
    report.check("historical_anchors_completed", not bad_status, bad_status)
    try:
        final_text = FINAL_REPORT.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        report.check("historical_final_report_readable", False, str(exc))
        return
    report.check(
        "historical_decision_bound",
        "- Decision: `no_go`" in final_text and DECISION_CID in final_text,
        DECISION_CID,
    )
    report.check(
        "historical_next_tasks_bound",
        all(task_id in final_text for task_id in SOURCE_OPEN_TASKS),
        list(SOURCE_OPEN_TASKS),
    )


def validate_projection(report: Report) -> None:
    try:
        tasks = parse_board(BOARD)
        source_tasks = parse_board(SOURCE_BOARD)
    except (OSError, UnicodeError, ValueError) as exc:
        report.check("taskboards_readable", False, str(exc))
        return
    report.check(
        "task_population",
        tuple(tasks) == EXPECTED_TASKS,
        {"expected": list(EXPECTED_TASKS), "actual": list(tasks)},
    )
    report.check(
        "protected_successor_population",
        tuple(source_tasks) == SOURCE_OPEN_TASKS,
        {"expected": list(SOURCE_OPEN_TASKS), "actual": list(source_tasks)},
    )
    source_statuses = {
        task_id: source_tasks.get(task_id, {}).get("fields", {}).get("status")
        for task_id in SOURCE_OPEN_TASKS
    }
    report.check(
        "protected_successor_initial_statuses",
        all(value == "todo" for value in source_statuses.values()),
        source_statuses,
    )
    statuses = {
        task_id: tasks.get(task_id, {}).get("fields", {}).get("status", "").casefold()
        for task_id in EXPECTED_TASKS
    }
    report.check(
        "anchor_projection_statuses",
        all(statuses.get(task_id) == "completed" for task_id in ANCHORS),
        {task_id: statuses.get(task_id) for task_id in ANCHORS},
    )
    report.check(
        "open_projection_statuses",
        all(statuses.get(task_id) in ALLOWED_OPEN_STATUSES for task_id in OPEN_TASKS),
        {task_id: statuses.get(task_id) for task_id in OPEN_TASKS},
    )
    dependencies = {
        task_id: split_csv(
            tasks.get(task_id, {}).get("fields", {}).get("depends_on", "")
        )
        for task_id in EXPECTED_TASKS
    }
    report.check(
        "dependency_dag",
        dependencies == EXPECTED_DEPENDENCIES,
        {task_id: list(value) for task_id, value in dependencies.items()},
    )
    unknown_dependencies = sorted(
        {
            dependency
            for values in dependencies.values()
            for dependency in values
            if dependency not in tasks
        }
    )
    report.check("dependency_closure", not unknown_dependencies, unknown_dependencies)
    missing_fields: dict[str, list[str]] = {}
    required = (
        "status",
        "objective",
        "depends_on",
        "acceptance",
        "outputs",
        "validation",
        "predicted_files",
        "board_namespace",
        "goal_id",
    )
    for task_id, task in tasks.items():
        absent = [name for name in required if not task["fields"].get(name)]
        if absent:
            missing_fields[task_id] = absent
    report.check("required_task_fields", not missing_fields, missing_fields)
    identity_mismatches = {
        task_id: {
            "board_namespace": tasks[task_id]["fields"].get("board_namespace"),
            "goal_id": tasks[task_id]["fields"].get("goal_id"),
            "parent_goal": tasks[task_id]["fields"].get("parent_goal"),
        }
        for task_id in EXPECTED_TASKS
        if task_id in tasks
        and (
            tasks[task_id]["fields"].get("board_namespace") != BOARD_NAMESPACE
            or tasks[task_id]["fields"].get("goal_id") != EXPECTED_GOALS[task_id]
            or tasks[task_id]["fields"].get("parent_goal") != EXPECTED_GOALS[task_id]
        )
    }
    report.check(
        "canonical_task_identity_fields", not identity_mismatches, identity_mismatches
    )
    non_generation_outputs = {
        task_id: {
            "outputs": tasks[task_id]["fields"].get("outputs", ""),
            "predicted_files": tasks[task_id]["fields"].get("predicted_files", ""),
        }
        for task_id in OPEN_TASKS
        if task_id in tasks
        and (
            "successor-v1" not in tasks[task_id]["fields"].get("outputs", "")
            or "successor-v1" not in tasks[task_id]["fields"].get("predicted_files", "")
        )
    }
    report.check(
        "append_only_generation_paths",
        not non_generation_outputs,
        non_generation_outputs,
    )


def validate_sources(report: Report) -> None:
    head = run_git(ROOT, "rev-parse", "HEAD")
    report.check("accelerator_head_readable", head.returncode == 0, head.stderr.strip())
    ancestor = run_git(ROOT, "merge-base", "--is-ancestor", ACCELERATOR_BASE, "HEAD")
    report.check(
        "accelerator_base_ancestor", ancestor.returncode == 0, ACCELERATOR_BASE
    )
    pgir = run_git(ROOT, "merge-base", "--is-ancestor", ACCELERATOR_PGIR_MERGE, "HEAD")
    report.check(
        "accelerator_pgir_merged", pgir.returncode == 0, ACCELERATOR_PGIR_MERGE
    )

    nested_head = run_git(DATASETS, "rev-parse", "HEAD")
    actual_nested_head = (
        nested_head.stdout.strip() if nested_head.returncode == 0 else ""
    )
    datasets_base = run_git(
        DATASETS, "merge-base", "--is-ancestor", DATASETS_BASE, "HEAD"
    )
    report.check(
        "datasets_planning_base_ancestor",
        datasets_base.returncode == 0,
        {"required_ancestor": DATASETS_BASE, "actual": actual_nested_head},
    )
    gitlink = run_git(ROOT, "ls-tree", "HEAD", "ipfs_datasets_py")
    actual_gitlink = (
        gitlink.stdout.split()[2] if len(gitlink.stdout.split()) >= 3 else ""
    )
    report.check(
        "datasets_gitlink_bound",
        bool(actual_gitlink)
        and actual_nested_head == actual_gitlink
        and datasets_base.returncode == 0,
        {
            "planning_ancestor": DATASETS_BASE,
            "gitlink": actual_gitlink,
            "nested_head": actual_nested_head,
        },
    )
    datasets_pgir = run_git(
        DATASETS, "merge-base", "--is-ancestor", DATASETS_PGIR_MERGE, "HEAD"
    )
    report.check(
        "datasets_pgir_merged", datasets_pgir.returncode == 0, DATASETS_PGIR_MERGE
    )
    nested_status = run_git(
        DATASETS, "status", "--porcelain=v1", "--untracked-files=all"
    )
    report.check(
        "datasets_checkout_clean",
        nested_status.returncode == 0 and not nested_status.stdout.strip(),
        nested_status.stdout.splitlines()[:20],
    )


def validate_controls(report: Report) -> None:
    required_paths = (
        BOARD,
        SOURCE_BOARD,
        HISTORICAL_BOARD,
        OBJECTIVES,
        FINAL_REPORT,
        PLAN,
        CONFIG,
        PREDECESSOR_REPLAY,
    )
    missing = [
        str(path.relative_to(ROOT)) for path in required_paths if not path.is_file()
    ]
    report.check("control_files_present", not missing, missing)
    if not CONFIG.is_file():
        return
    try:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        report.check("scheduler_config_readable", False, str(exc))
        return
    source = config.get("source_binding") if isinstance(config, dict) else None
    source = source if isinstance(source, dict) else {}
    report.check(
        "scheduler_source_binding",
        source.get("accelerator_required_ancestor") == ACCELERATOR_BASE
        and source.get("ipfs_datasets_planning_revision") == DATASETS_BASE,
        source,
    )
    report.check(
        "scheduler_taskboard_binding",
        config.get("taskboard_path")
        == "docs/architecture/proof_grounded_ir_learning/successor.todo.md",
        config.get("taskboard_path"),
    )
    database_program = config.get("database_program")
    report.check(
        "scheduler_explicit_legacy_authority",
        database_program
        == {
            "authority_mode": "legacy_markdown",
            "task_source_kind": "legacy-markdown",
            "failover_policy": "fail_closed",
            "explicit_legacy": True,
        },
        database_program,
    )
    provider = config.get("provider")
    expected_provider = {
        "primary_provider_id": "grok_cli",
        "primary_model_id": "grok-4.6",
        "fallback_provider_id": "codex",
        "fallback_model_id": "gpt-5.6-terra",
        "fallback_trigger": "primary_quota_exhausted",
        "fallback_reasoning_effort": "high",
        "max_concurrency": 2,
        "secrets_from_environment_only": True,
        "secrets_in_argv_prompts_logs_or_receipts": False,
    }
    report.check(
        "scheduler_reviewed_provider_route",
        provider == expected_provider,
        provider,
    )
    try:
        corpus = json.loads(
            (DATASETS / "data/ir_learning/corpora/corpus_root.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        report.check("historical_corpus_root_readable", False, str(exc))
    else:
        report.check(
            "historical_corpus_root_typed",
            corpus.get("kind") == "ir-sealed-corpus-root/v1"
            and isinstance(corpus.get("training_admitted_rows"), int),
            {
                "kind": corpus.get("kind"),
                "training_admitted_rows": corpus.get("training_admitted_rows"),
            },
        )


def validate_predecessor_replay(report: Report) -> None:
    try:
        process = subprocess.run(
            (sys.executable, str(PREDECESSOR_REPLAY)),
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        report.check("historical_predecessor_replay", False, str(exc))
        return
    try:
        outcome = json.loads(process.stdout)
    except json.JSONDecodeError:
        report.check(
            "historical_predecessor_replay",
            False,
            process.stderr.strip() or process.stdout.strip(),
        )
        return
    passed = (
        process.returncode == 0
        and isinstance(outcome, dict)
        and outcome.get("verified") is True
        and outcome.get("descendant_execution_authorized") is False
        and outcome.get("source_commit") == "04fbb09b4a8b34e77d11bd8da6642e0978baa02c"
        and outcome.get("datasets_commit") == "b20bd9e3cfae79e8888929daf64f52b2f8a5689a"
    )
    report.check("historical_predecessor_replay", passed, outcome)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-all", action="store_true")
    mode.add_argument("--check-anchors", action="store_true")
    args = parser.parse_args()

    report = Report()
    validate_anchors(report)
    if not args.check_anchors:
        validate_projection(report)
        validate_sources(report)
        validate_controls(report)
        validate_predecessor_replay(report)
    print(json.dumps(report.payload(), sort_keys=True))
    return 0 if not report.errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
