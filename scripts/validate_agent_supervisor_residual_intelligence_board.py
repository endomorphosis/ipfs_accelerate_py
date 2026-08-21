#!/usr/bin/env python3
"""Validate the Verified Residual Intelligence Foundry bootstrap board.

The Markdown files are a sealed bootstrap projection.  This validator checks
their complete, bounded intent before the records are materialized into the
DuckDB control plane.  It deliberately does not treat a Markdown status as
completion evidence.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    REPO_ROOT / "docs/architecture/AGENT_SUPERVISOR_VERIFIED_RESIDUAL_INTELLIGENCE_FOUNDRY_PLAN.md"
)
OBJECTIVE_PATH = (
    REPO_ROOT / "docs/architecture/agent_supervisor_residual_intelligence.objectives.md"
)
TODO_PATH = REPO_ROOT / "docs/architecture/agent_supervisor_residual_intelligence.todo.md"
VALIDATOR_PATH = Path(__file__).resolve()
TEST_PATH = REPO_ROOT / "test/api/residual_intelligence/test_board.py"

PROGRAM_ID = "agent-supervisor-verified-residual-intelligence-foundry-v1"
TARGET_BRANCH = "codex/verified-residual-intelligence-foundry-v1"
ROOT_GOAL = "VRIF-G000"
TASK_IDS = tuple(f"VRIF-{index:03d}" for index in range(33))
GOAL_IDS = (
    "VRIF-G000",
    "VRIF-G010",
    "VRIF-G011",
    "VRIF-G020",
    "VRIF-G021",
    "VRIF-G030",
    "VRIF-G031",
    "VRIF-G040",
    "VRIF-G041",
)

TASK_TITLES = (
    "Seal current authority and prerequisite baseline",
    "Inventory residual model calls and task families",
    "Define residual intelligence contracts",
    "Define training corpus admission and rights contracts",
    "Build first-party trajectory corpus",
    "Build synthetic and adversarial corpus",
    "Implement lineage-safe semantic splits",
    "Implement compact ResidualIntelligenceIR",
    "Implement structured-output grammars",
    "Implement deterministic and linear baselines",
    "Implement task-family expert specifications",
    "Implement calibration and abstention contracts",
    "Implement OOD and boundary detection",
    "Implement expert cascade router",
    "Implement local classification and ranking experts",
    "Implement constrained structured-decoder expert",
    "Integrate procedure-hole resolution",
    "Integrate proof and tactic experts",
    "Integrate patch-sketch experts",
    "Implement teacher-disagreement handling",
    "Implement proof-grounded label production",
    "Implement active-learning planner",
    "Implement continual-learning epoch contracts",
    "Integrate learning-checkpoint lineage",
    "Implement quantization and packaging qualification",
    "Integrate shared model serving and batching",
    "Implement expert drift, demotion, and revocation",
    "Implement privacy and information-flow gates",
    "Run adversarial assurance campaign",
    "Add control service, CLI, and MCP surfaces",
    "Build frozen paired benchmark",
    "Implement promotion and rollback gates",
    "Produce current-tree release and residual-gap report",
)

EXPECTED_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "VRIF-000": (),
    "VRIF-001": ("VRIF-000",),
    "VRIF-002": ("VRIF-000",),
    "VRIF-003": ("VRIF-000", "VRIF-002"),
    "VRIF-004": ("VRIF-001", "VRIF-003"),
    "VRIF-005": ("VRIF-003",),
    "VRIF-006": ("VRIF-004", "VRIF-005"),
    "VRIF-007": ("VRIF-002", "VRIF-003"),
    "VRIF-008": ("VRIF-002", "VRIF-007"),
    "VRIF-009": ("VRIF-006", "VRIF-007", "VRIF-008"),
    "VRIF-010": ("VRIF-001", "VRIF-002", "VRIF-007"),
    "VRIF-011": ("VRIF-002", "VRIF-006", "VRIF-007"),
    "VRIF-012": ("VRIF-002", "VRIF-006", "VRIF-007"),
    "VRIF-013": ("VRIF-009", "VRIF-010", "VRIF-011", "VRIF-012"),
    "VRIF-014": ("VRIF-009", "VRIF-010", "VRIF-011", "VRIF-012"),
    "VRIF-015": ("VRIF-008", "VRIF-010", "VRIF-011", "VRIF-012"),
    "VRIF-016": ("VRIF-013", "VRIF-015"),
    "VRIF-017": ("VRIF-013", "VRIF-014"),
    "VRIF-018": ("VRIF-013", "VRIF-015"),
    "VRIF-019": ("VRIF-005", "VRIF-014", "VRIF-015"),
    "VRIF-020": ("VRIF-004", "VRIF-005", "VRIF-017", "VRIF-019"),
    "VRIF-021": ("VRIF-006", "VRIF-011", "VRIF-019", "VRIF-020"),
    "VRIF-022": ("VRIF-003", "VRIF-006", "VRIF-011", "VRIF-021"),
    "VRIF-023": ("VRIF-022",),
    "VRIF-024": ("VRIF-011", "VRIF-012", "VRIF-014", "VRIF-015", "VRIF-023"),
    "VRIF-025": ("VRIF-013", "VRIF-014", "VRIF-015", "VRIF-024"),
    "VRIF-026": ("VRIF-011", "VRIF-012", "VRIF-023", "VRIF-024", "VRIF-025"),
    "VRIF-027": ("VRIF-003", "VRIF-013", "VRIF-025", "VRIF-026"),
    "VRIF-028": tuple(f"VRIF-{index:03d}" for index in range(16, 28)),
    "VRIF-029": (
        "VRIF-013",
        "VRIF-021",
        "VRIF-022",
        "VRIF-023",
        "VRIF-026",
        "VRIF-027",
    ),
    "VRIF-030": (
        "VRIF-005",
        "VRIF-006",
        "VRIF-008",
        "VRIF-014",
        "VRIF-015",
        "VRIF-019",
        "VRIF-020",
    ),
    "VRIF-031": ("VRIF-024", "VRIF-026", "VRIF-027", "VRIF-028", "VRIF-030"),
    "VRIF-032": ("VRIF-028", "VRIF-029", "VRIF-030", "VRIF-031"),
}

EXPECTED_TASK_GOALS = {
    **{f"VRIF-{index:03d}": "VRIF-G011" for index in range(0, 9)},
    **{f"VRIF-{index:03d}": "VRIF-G021" for index in range(9, 16)},
    **{f"VRIF-{index:03d}": "VRIF-G031" for index in range(16, 28)},
    **{f"VRIF-{index:03d}": "VRIF-G041" for index in range(28, 33)},
}

EXPECTED_GOAL_PARENTS = {
    "VRIF-G000": "",
    "VRIF-G010": "VRIF-G000",
    "VRIF-G011": "VRIF-G010",
    "VRIF-G020": "VRIF-G000",
    "VRIF-G021": "VRIF-G020",
    "VRIF-G030": "VRIF-G000",
    "VRIF-G031": "VRIF-G030",
    "VRIF-G040": "VRIF-G000",
    "VRIF-G041": "VRIF-G040",
}

EXPECTED_GOAL_DEPENDENCIES = {
    "VRIF-G000": (),
    "VRIF-G010": (),
    "VRIF-G011": (),
    "VRIF-G020": ("VRIF-G010",),
    "VRIF-G021": ("VRIF-G011",),
    "VRIF-G030": ("VRIF-G020",),
    "VRIF-G031": ("VRIF-G021",),
    "VRIF-G040": ("VRIF-G030",),
    "VRIF-G041": ("VRIF-G031",),
}

REQUIRED_TASK_FIELDS = (
    "status",
    "goal",
    "depends on",
    "objective",
    "acceptance subset",
    "predicted files",
    "predicted symbols",
    "data rights",
    "privacy class",
    "effect class",
    "risk class",
    "resource class",
    "token budget",
    "training budget",
    "validation",
    "proof requirements",
    "rollback",
    "conflict policy",
)
REQUIRED_GOAL_FIELDS = (
    "status",
    "parent",
    "depends on",
    "objective",
    "completion contract",
    "evidence",
    "acceptance criteria",
    "outputs",
    "validation",
    "conflict policy",
)
PRIVACY_CLASSES = frozenset(
    {
        "public",
        "internal",
        "repository_private",
        "tenant_private",
        "matter_confidential",
        "credential",
        "personal_data",
        "health_data",
        "legal_privileged",
        "proof_witness",
    }
)
RISK_CLASSES = frozenset({"R0", "R1", "R2", "R3", "R4", "R5"})
REQUIRED_PLAN_TERMS = (
    PROGRAM_ID,
    TARGET_BRANCH,
    "VerifiedResidualIntelligenceFoundry",
    "DuckDB",
    "Quack",
    "DuckLake",
    "non-authoritative",
    "training_unavailable",
    "TrainingCorpusAdmission",
    "zero model-created authority",
    "zero model-created completion",
    "zero training on unadmitted data",
    "candidate_only",
    "exact cache",
    "verified procedure",
    "local linear expert",
    "remote strong model",
    "human review",
    "45% fewer remote-model calls",
    "99% precision",
    "false completion",
)
REQUIRED_BOARD_TERMS = (
    "task-board status is not completion evidence",
    "DuckDB is authoritative",
    "DuckLake is non-authoritative",
    "training_unavailable",
    "no training on unadmitted data",
)

_TASK_HEADER = re.compile(r"^## (VRIF-\d{3}) (.+)$")
_GOAL_HEADER = re.compile(r"^## (VRIF-G\d{3}) (.+)$")
_BOLD_FIELD = re.compile(r"^- \*\*([^*]+):\*\*\s*(.*)$")


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in re.split(r"[,;]", value) if item.strip())


def _parse_records(
    text: str, header_pattern: re.Pattern[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    current: dict[str, Any] | None = None
    seen_fields: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        header = header_pattern.fullmatch(line)
        if header:
            if current is not None:
                records.append(current)
            current = {
                "id": header.group(1),
                "title": header.group(2).strip(),
                "fields": {},
                "line": line_number,
            }
            seen_fields = set()
            continue
        if current is None:
            continue
        field_match = _BOLD_FIELD.fullmatch(line)
        if not field_match:
            continue
        key = " ".join(field_match.group(1).strip().lower().split())
        if key in seen_fields:
            errors.append(f"{current['id']}: duplicate field {key!r}")
            continue
        seen_fields.add(key)
        current["fields"][key] = field_match.group(2).strip()
    if current is not None:
        records.append(current)
    return records, errors


def _acyclic(adjacency: Mapping[str, Iterable[str]]) -> tuple[bool, list[str]]:
    visiting: set[str] = set()
    visited: set[str] = set()
    cycle: list[str] = []

    def visit(node: str, trail: list[str]) -> bool:
        if node in visiting:
            cycle.extend([*trail, node])
            return False
        if node in visited:
            return True
        visiting.add(node)
        for dependency in adjacency.get(node, ()):
            if not visit(dependency, [*trail, node]):
                return False
        visiting.remove(node)
        visited.add(node)
        return True

    return all(visit(node, []) for node in adjacency if node not in visited), cycle


def _safe_path(value: str) -> bool:
    path = PurePosixPath(value.strip().replace("\\", "/"))
    return bool(
        value.strip() and not path.is_absolute() and ".." not in path.parts and "\x00" not in value
    )


def _append(
    checks: list[dict[str, Any]],
    errors: list[str],
    *,
    name: str,
    passed: bool,
    detail: Any,
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})
    if not passed:
        errors.append(f"{name}: {detail}")


def validate_program() -> dict[str, Any]:
    """Return a deterministic machine-readable validation report."""

    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    required_files = (PLAN_PATH, OBJECTIVE_PATH, TODO_PATH, VALIDATOR_PATH, TEST_PATH)
    missing = [
        path.relative_to(REPO_ROOT).as_posix() for path in required_files if not path.is_file()
    ]
    _append(
        checks,
        errors,
        name="required_files_present",
        passed=not missing,
        detail=missing,
    )
    if missing:
        return _report(checks, errors, task_count=0, goal_count=0)

    plan_text = PLAN_PATH.read_text(encoding="utf-8")
    objective_text = OBJECTIVE_PATH.read_text(encoding="utf-8")
    todo_text = TODO_PATH.read_text(encoding="utf-8")
    tasks, task_parse_errors = _parse_records(todo_text, _TASK_HEADER)
    goals, goal_parse_errors = _parse_records(objective_text, _GOAL_HEADER)
    errors.extend(task_parse_errors)
    errors.extend(goal_parse_errors)

    missing_terms = [
        term for term in REQUIRED_PLAN_TERMS if term.casefold() not in plan_text.casefold()
    ]
    _append(
        checks,
        errors,
        name="plan_governance_terms",
        passed=not missing_terms,
        detail=missing_terms,
    )
    missing_board_terms = [
        term for term in REQUIRED_BOARD_TERMS if term.casefold() not in todo_text.casefold()
    ]
    _append(
        checks,
        errors,
        name="board_authority_terms",
        passed=not missing_board_terms,
        detail=missing_board_terms,
    )

    task_ids = [str(record["id"]) for record in tasks]
    _append(
        checks,
        errors,
        name="exact_task_population",
        passed=tuple(task_ids) == TASK_IDS,
        detail={"expected": list(TASK_IDS), "observed": task_ids},
    )
    title_errors = [
        f"{record['id']}: {record['title']!r} != {TASK_TITLES[index]!r}"
        for index, record in enumerate(tasks[: len(TASK_TITLES)])
        if record["title"] != TASK_TITLES[index]
    ]
    _append(
        checks,
        errors,
        name="exact_task_titles",
        passed=not title_errors and len(tasks) == len(TASK_TITLES),
        detail=title_errors,
    )

    task_by_id = {str(record["id"]): record for record in tasks}
    task_errors: list[str] = []
    observed_dependencies: dict[str, tuple[str, ...]] = {}
    for task_id in TASK_IDS:
        record = task_by_id.get(task_id)
        if record is None:
            continue
        fields: dict[str, str] = record["fields"]
        missing_fields = [field for field in REQUIRED_TASK_FIELDS if field not in fields]
        if missing_fields:
            task_errors.append(f"{task_id}: missing fields {missing_fields}")
            continue
        expected_status = "completed" if task_id <= "VRIF-008" else "todo"
        if fields["status"].casefold() != expected_status:
            task_errors.append(
                f"{task_id}: status {fields['status']!r}, expected {expected_status!r}"
            )
        if task_id <= "VRIF-008":
            evidence = fields.get("completion evidence", "").casefold()
            if not all(term in evidence for term in ("exact", "current-tree", "validation")):
                task_errors.append(
                    f"{task_id}: completed bootstrap row lacks exact current-tree validation evidence"
                )
        dependencies = _csv(fields["depends on"])
        observed_dependencies[task_id] = dependencies
        if dependencies != EXPECTED_DEPENDENCIES[task_id]:
            task_errors.append(
                f"{task_id}: dependencies {dependencies!r}, expected "
                f"{EXPECTED_DEPENDENCIES[task_id]!r}"
            )
        if fields["goal"] != EXPECTED_TASK_GOALS[task_id]:
            task_errors.append(
                f"{task_id}: goal {fields['goal']!r}, expected {EXPECTED_TASK_GOALS[task_id]!r}"
            )
        paths = _csv(fields["predicted files"])
        if not paths or not all(_safe_path(path) for path in paths):
            task_errors.append(f"{task_id}: predicted files are empty or unsafe")
        privacy_values = _csv(fields["privacy class"])
        if not privacy_values or not set(privacy_values) <= PRIVACY_CLASSES:
            task_errors.append(f"{task_id}: invalid privacy class {privacy_values!r}")
        if fields["risk class"] not in RISK_CLASSES:
            task_errors.append(f"{task_id}: invalid risk class {fields['risk class']!r}")
        token_budget = fields["token budget"]
        if not re.fullmatch(r"input=\d+; output=\d+", token_budget):
            task_errors.append(f"{task_id}: invalid token budget {token_budget!r}")
        training_budget = fields["training budget"].casefold()
        if "trainingcorpusadmission" not in training_budget:
            task_errors.append(f"{task_id}: training budget lacks admission gate")
        if task_id <= "VRIF-008" and not training_budget.startswith("0"):
            task_errors.append(f"{task_id}: Tranche 1 training budget must be zero")
        if not fields["validation"].startswith("python"):
            task_errors.append(f"{task_id}: validation is not an explicit Python command")
        for field in REQUIRED_TASK_FIELDS:
            if not fields[field] and field != "depends on":
                task_errors.append(f"{task_id}: empty {field}")
    _append(
        checks,
        errors,
        name="task_contracts",
        passed=not task_errors,
        detail=task_errors,
    )

    dependency_unknown = {
        task_id: sorted(set(dependencies) - set(TASK_IDS))
        for task_id, dependencies in observed_dependencies.items()
        if set(dependencies) - set(TASK_IDS)
    }
    _append(
        checks,
        errors,
        name="dependency_references",
        passed=not dependency_unknown,
        detail=dependency_unknown,
    )
    acyclic, cycle = _acyclic(observed_dependencies)
    _append(checks, errors, name="task_dag_acyclic", passed=acyclic, detail=cycle)

    completed = {f"VRIF-{index:03d}" for index in range(9)}
    ready = tuple(
        task_id
        for task_id in TASK_IDS
        if task_id not in completed and set(observed_dependencies.get(task_id, ())) <= completed
    )
    _append(
        checks,
        errors,
        name="parallel_ready_frontier",
        passed=ready == ("VRIF-009", "VRIF-010", "VRIF-011", "VRIF-012"),
        detail=ready,
    )

    goal_ids = [str(record["id"]) for record in goals]
    _append(
        checks,
        errors,
        name="exact_goal_population",
        passed=tuple(goal_ids) == GOAL_IDS,
        detail={"expected": list(GOAL_IDS), "observed": goal_ids},
    )
    goal_errors: list[str] = []
    adjacency: dict[str, tuple[str, ...]] = {}
    for record in goals:
        goal_id = str(record["id"])
        fields: dict[str, str] = record["fields"]
        missing_fields = [field for field in REQUIRED_GOAL_FIELDS if field not in fields]
        if missing_fields:
            goal_errors.append(f"{goal_id}: missing fields {missing_fields}")
            continue
        if fields["parent"] != EXPECTED_GOAL_PARENTS.get(goal_id):
            goal_errors.append(
                f"{goal_id}: parent {fields['parent']!r}, expected "
                f"{EXPECTED_GOAL_PARENTS.get(goal_id)!r}"
            )
        dependencies = _csv(fields["depends on"])
        adjacency[goal_id] = dependencies
        if dependencies != EXPECTED_GOAL_DEPENDENCIES.get(goal_id):
            goal_errors.append(
                f"{goal_id}: dependencies {dependencies!r}, expected "
                f"{EXPECTED_GOAL_DEPENDENCIES.get(goal_id)!r}"
            )
        for field in REQUIRED_GOAL_FIELDS:
            if not fields[field] and field not in {"parent", "depends on"}:
                goal_errors.append(f"{goal_id}: empty {field}")
    _append(
        checks,
        errors,
        name="goal_contracts",
        passed=not goal_errors,
        detail=goal_errors,
    )
    goals_acyclic, goal_cycle = _acyclic(adjacency)
    _append(
        checks,
        errors,
        name="goal_dag_acyclic",
        passed=goals_acyclic,
        detail=goal_cycle,
    )

    task_path = TODO_PATH.relative_to(REPO_ROOT).as_posix()
    objective_path = OBJECTIVE_PATH.relative_to(REPO_ROOT).as_posix()
    declared_files = "\n".join(field for record in tasks for field in record["fields"].values())
    _append(
        checks,
        errors,
        name="required_artifacts_declared",
        passed=all(
            name in plan_text + objective_text + todo_text + declared_files
            for name in (
                task_path,
                objective_path,
                "docs/architecture/residual_intelligence_inventory/",
                "benchmarks/agent_supervisor/residual_intelligence/",
                "scripts/validate_agent_supervisor_residual_intelligence_board.py",
                "test/api/residual_intelligence/",
            )
        ),
        detail="required plans, inventories, benchmark, validator, and tests",
    )
    return _report(checks, errors, task_count=len(tasks), goal_count=len(goals))


def _report(
    checks: Sequence[Mapping[str, Any]],
    errors: Sequence[str],
    *,
    task_count: int,
    goal_count: int,
) -> dict[str, Any]:
    return {
        "schema": (
            "ipfs_accelerate_py/agent-supervisor/verified-residual-intelligence-board-validation@1"
        ),
        "program_id": PROGRAM_ID,
        "root_goal": ROOT_GOAL,
        "valid": not errors,
        "task_count": task_count,
        "goal_count": goal_count,
        "ready_frontier": ["VRIF-009", "VRIF-010", "VRIF-011", "VRIF-012"],
        "errors": list(errors),
        "checks": list(checks),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="compatibility flag; all checks are always run",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = validate_program()
    if args.json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        state = "PASS" if report["valid"] else "FAIL"
        print(
            f"{state}: {report['task_count']} tasks, {report['goal_count']} goals, "
            f"{len(report['checks'])} checks"
        )
        for error in report["errors"]:
            print(f"ERROR: {error}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
