#!/usr/bin/env python3
"""Fail-closed static validator for the bounded ASEH goal heap and task DAG.

The validator deliberately imports no project package and opens no control
store. It is safe to run before materialization and while Quack owns DuckDB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/architecture/AGENT_SUPERVISOR_EFFICIENCY_AND_STATE_HARDENING_PLAN.md"
OBJECTIVES = ROOT / "docs/architecture/agent_supervisor_efficiency_state_hardening.objectives.md"
BOARD = ROOT / "docs/architecture/agent_supervisor_efficiency_state_hardening.todo.md"
CONFIG = ROOT / "config/agent_supervisor_efficiency_state_hardening_scheduler.json"
BASELINE = ROOT / "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/bootstrap_baseline.json"
REQUIREMENTS = ROOT / "docs/architecture/agent_supervisor_efficiency_state_hardening.requirements.json"
PROGRAMS = ROOT / "docs/architecture/agent_supervisor/PROGRAMS.md"
OPERATOR = ROOT / "scripts/run_agent_supervisor_efficiency_state_hardening.py"
BOOTSTRAP_TEST = ROOT / "test/api/test_agent_supervisor_configured_typed_grant_handoff.py"

PROGRAM = "agent-supervisor-efficiency-and-state-hardening-v1"
PLAN_REVISION = "ASEH-PLAN-R1"
BRANCH = "agent/agent-supervisor-efficiency-and-state-hardening-v1"
SCHEMA = "ipfs_accelerate_py/agent-supervisor/aseh-board-validation@1"

TASK_TITLES = (
    ("ASEH-000", "Inventory current authorities and bypasses"),
    ("ASEH-001", "Seal current repository and supervisor baseline"),
    ("ASEH-010", "Define telemetry and efficiency receipt schemas"),
    ("ASEH-011", "Instrument provider and model usage"),
    ("ASEH-012", "Instrument tests proofs retries merges and human intervention"),
    ("ASEH-013", "Build paired benchmark harness"),
    ("ASEH-014", "Build historical replay corpus"),
    ("ASEH-015", "Add live shadow and canary cohort"),
    ("ASEH-020", "Consolidate deterministic-first routing"),
    ("ASEH-021", "Add unresolved-question contract"),
    ("ASEH-022", "Add deterministic cache AST static test and prover route stages"),
    ("ASEH-023", "Add small medium and frontier escalation policy"),
    ("ASEH-024", "Add route receipts and escalation metrics"),
    ("ASEH-030", "Define canonical ContextPack contract"),
    ("ASEH-031", "Implement Datasets semantic pack builder"),
    ("ASEH-032", "Implement Kit pack storage and current-root CAS"),
    ("ASEH-033", "Implement Accelerate freshness and selection"),
    ("ASEH-034", "Implement incremental pack expansion"),
    ("ASEH-035", "Benchmark pack reuse omissions and net savings"),
    ("ASEH-040", "Define canonical state machine"),
    ("ASEH-041", "Enforce transition authority"),
    ("ASEH-042", "Add lease fence and idempotency invariants"),
    ("ASEH-043", "Add unknown-outcome reconciliation"),
    ("ASEH-044", "Add owner-loss and restart recovery"),
    ("ASEH-045", "Add model property and crash tests"),
    ("ASEH-050", "Consolidate dependency and impact analysis"),
    ("ASEH-051", "Add assumption guarantee task contracts"),
    ("ASEH-052", "Add affected-suffix replanning"),
    ("ASEH-053", "Add counterexample and unsat-core context minimization"),
    ("ASEH-054", "Add bounded deterministic synthesis"),
    ("ASEH-055", "Add PatchPlan validation and merge admission"),
    ("ASEH-060", "Inventory and retire duplicate writable authorities"),
    ("ASEH-061", "Add compatibility migration adapters"),
    ("ASEH-062", "Add current-head cross-repository qualification"),
    ("ASEH-070", "Run hermetic development benchmark"),
    ("ASEH-071", "Run historical paired replay"),
    ("ASEH-072", "Run live shadow cohort"),
    ("ASEH-073", "Run low-risk canary"),
    ("ASEH-074", "Produce promotion or honest non-promotion receipt"),
    ("ASEH-075", "Publish final residual-gap report"),
)
EXPECTED_TASKS = tuple(item[0] for item in TASK_TITLES)
EXPECTED_TITLES = dict(TASK_TITLES)
EXPECTED_GOALS = (
    "ASEH-G000", "ASEH-G010", "ASEH-G020", "ASEH-G030", "ASEH-G040",
    "ASEH-G050", "ASEH-G060", "ASEH-G070", "ASEH-G080",
)
EXPECTED_PARENTS = {
    "ASEH-G000": "",
    **{goal: "ASEH-G000" for goal in EXPECTED_GOALS[1:]},
}
EXPECTED_GOAL_DEPENDENCIES = {
    "ASEH-G000": [],
    "ASEH-G010": [],
    "ASEH-G020": ["ASEH-G010"],
    "ASEH-G030": ["ASEH-G010"],
    "ASEH-G040": ["ASEH-G010"],
    "ASEH-G050": ["ASEH-G010"],
    "ASEH-G060": ["ASEH-G010"],
    "ASEH-G070": [
        "ASEH-G020", "ASEH-G030", "ASEH-G040", "ASEH-G050", "ASEH-G060",
    ],
    "ASEH-G080": ["ASEH-G070"],
}
EXPECTED_DEPENDENCIES = {
    "ASEH-000": [], "ASEH-001": [],
    "ASEH-010": ["ASEH-000", "ASEH-001"],
    "ASEH-011": ["ASEH-010"], "ASEH-012": ["ASEH-010", "ASEH-011"],
    "ASEH-013": ["ASEH-010", "ASEH-011", "ASEH-012"],
    "ASEH-014": ["ASEH-013"], "ASEH-015": ["ASEH-013", "ASEH-014"],
    "ASEH-020": ["ASEH-000", "ASEH-001"],
    "ASEH-021": ["ASEH-010", "ASEH-020"], "ASEH-022": ["ASEH-020", "ASEH-021"],
    "ASEH-023": ["ASEH-022"], "ASEH-024": ["ASEH-011", "ASEH-012", "ASEH-023"],
    "ASEH-030": ["ASEH-000", "ASEH-001"], "ASEH-031": ["ASEH-030"],
    "ASEH-032": ["ASEH-030"], "ASEH-033": ["ASEH-030", "ASEH-031", "ASEH-032"],
    "ASEH-034": ["ASEH-033"], "ASEH-035": ["ASEH-013", "ASEH-024", "ASEH-034"],
    "ASEH-040": ["ASEH-000", "ASEH-001"], "ASEH-041": ["ASEH-040"],
    "ASEH-042": ["ASEH-040", "ASEH-041"],
    "ASEH-043": ["ASEH-041", "ASEH-042"],
    "ASEH-044": ["ASEH-041", "ASEH-042", "ASEH-043"],
    "ASEH-045": ["ASEH-040", "ASEH-041", "ASEH-042", "ASEH-043", "ASEH-044"],
    "ASEH-050": ["ASEH-000", "ASEH-001"], "ASEH-051": ["ASEH-040", "ASEH-050"],
    "ASEH-052": ["ASEH-050", "ASEH-051"], "ASEH-053": ["ASEH-050", "ASEH-051"],
    "ASEH-054": ["ASEH-020", "ASEH-050", "ASEH-051"],
    "ASEH-055": ["ASEH-041", "ASEH-051", "ASEH-052", "ASEH-053", "ASEH-054"],
    "ASEH-060": ["ASEH-024", "ASEH-035", "ASEH-045", "ASEH-055"],
    "ASEH-061": ["ASEH-045", "ASEH-060"],
    "ASEH-062": ["ASEH-031", "ASEH-032", "ASEH-033", "ASEH-034", "ASEH-061"],
    "ASEH-070": ["ASEH-013", "ASEH-024", "ASEH-035", "ASEH-045", "ASEH-055", "ASEH-062"],
    "ASEH-071": ["ASEH-014", "ASEH-070"], "ASEH-072": ["ASEH-015", "ASEH-071"],
    "ASEH-073": ["ASEH-072"],
    "ASEH-074": ["ASEH-070", "ASEH-071", "ASEH-072", "ASEH-073"],
    "ASEH-075": ["ASEH-074"],
}
EXPECTED_GROUPS = {
    "ASEH-G010": ["ASEH-000", "ASEH-001"],
    "ASEH-G020": ["ASEH-010", "ASEH-011", "ASEH-012", "ASEH-013", "ASEH-014", "ASEH-015"],
    "ASEH-G030": ["ASEH-020", "ASEH-021", "ASEH-022", "ASEH-023", "ASEH-024"],
    "ASEH-G040": ["ASEH-030", "ASEH-031", "ASEH-032", "ASEH-033", "ASEH-034", "ASEH-035"],
    "ASEH-G050": ["ASEH-040", "ASEH-041", "ASEH-042", "ASEH-043", "ASEH-044", "ASEH-045"],
    "ASEH-G060": ["ASEH-050", "ASEH-051", "ASEH-052", "ASEH-053", "ASEH-054", "ASEH-055"],
    "ASEH-G070": ["ASEH-060", "ASEH-061", "ASEH-062"],
    "ASEH-G080": ["ASEH-070", "ASEH-071", "ASEH-072", "ASEH-073", "ASEH-074", "ASEH-075"],
}
EXPECTED_TASK_GOALS = {
    task_id: goal_id
    for goal_id, task_ids in EXPECTED_GROUPS.items()
    for task_id in task_ids
}
BASES = {
    "ipfs_accelerate_py": ("755f45475cc2d13dacd8b330036c1d597afeddde", "729da9f8293ecfa046a0136381a3d3808f9ed140"),
    "ipfs_datasets_py": ("209dbe2765593fbc6efe8e9281c34f2e8f6e37a6", "95f54df34585d0b736706fd90c83f55954489ad9"),
    "ipfs_kit_py": ("ba5508d940fb5b23a6d0d9b2084f5195cd26a671", "7c71efa93c4e4124d12fa05515868df3a5344b2e"),
}
SELECTED_REFS = {
    "ipfs_accelerate_py": "origin/main",
    "ipfs_datasets_py": "origin/integration/datasets-ui-capability-recovery-20260823",
    "ipfs_kit_py": "origin/feat/pcce-009-proof-context-v01",
}
BASELINE_GITLINKS = {
    "ipfs_datasets_py": "66a02063496fd200f2372b3083e376f1978c6be1",
    "ipfs_kit_py": "2564aea1ae35061f2165872aff91e8a40801ab7e",
}
REQUIRED_TASK_FIELDS = (
    "Stable task ID", "Status", "Completion", "Is schedulable", "Review only",
    "Priority", "Track", "Goal id", "Parent goal ID", "Subgoal ID",
    "Owning repository", "Board namespace", "Base revision",
    "Base repository tree", "Base plan revision", "Objective", "Depends on",
    "Exact declared outputs", "Outputs", "Owned paths", "Predicted files",
    "Worktree isolation", "Write scope", "External effect scope", "Risk class",
    "Authority requirement", "Acceptance conditions", "Validation",
    "Proof requirements", "Required evidence", "Model-route ceiling",
    "Lease and fencing", "Terminal success criteria",
    "Terminal non-success criteria", "Rollback", "Safety gates",
)
REQUIRED_GOAL_FIELDS = (
    "Status", "Parent", "Depends on", "Priority", "Track", "Goal",
    "Completion contract", "Evidence", "Acceptance criteria", "Outputs",
    "Validation", "Acceptance", "Gap task",
)
RISK_CLASSES = {
    "R0_PURE", "R1_READ_ONLY", "R2_REVERSIBLE_LOCAL",
    "R3_BOUNDED_REPOSITORY_MUTATION",
    "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
    "R5_IRREVERSIBLE_EXTERNAL_OR_LEGAL",
}
BOOTSTRAP_TASK_FIELDS = {
    "task_id", "stable_identity", "owning_repository", "dependencies",
    "exact_declared_outputs", "risk_class", "authority_requirement",
    "terminal_success_criteria", "terminal_non_success_criteria", "validation",
    "creates_new_supervisor", "creates_new_store", "required_before_materialization",
}
BOOTSTRAP_OUTPUTS = {
    ".gitignore",
    "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/process_security.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/intent_repository.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/typed_state_owner.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_loop.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_runtime.py",
    "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
}
REQUIRED_PROTECTED_PATHS = {
    ".gitignore",
    "docs/architecture/AGENT_SUPERVISOR_EFFICIENCY_AND_STATE_HARDENING_PLAN.md",
    "docs/architecture/agent_supervisor_efficiency_state_hardening.requirements.json",
    "docs/architecture/agent_supervisor_efficiency_state_hardening.objectives.md",
    "docs/architecture/agent_supervisor_efficiency_state_hardening.todo.md",
    "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/bootstrap_baseline.json",
    "docs/architecture/agent_supervisor/PROGRAMS.md",
    "config/agent_supervisor_efficiency_state_hardening_scheduler.json",
    "scripts/validate_agent_supervisor_efficiency_state_hardening_board.py",
    "scripts/run_agent_supervisor_efficiency_state_hardening.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/process_security.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/typed_state_owner.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/quack_state_client.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/database_task_source.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_loop.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_runtime.py",
    "test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
}
TASK_RE = re.compile(r"^## (ASEH-\d{3}) (.+)$", re.MULTILINE)
GOAL_RE = re.compile(r"^## (ASEH-G\d{3}) (.+)$", re.MULTILINE)
META_RE = re.compile(r"^- ([^:\n]+):[ \t]*(.*)$", re.MULTILINE)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(_read(path))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _blocks(text: str, pattern: re.Pattern[str]) -> list[tuple[str, str, dict[str, str]]]:
    matches = list(pattern.finditer(text))
    result: list[tuple[str, str, dict[str, str]]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields: dict[str, str] = {}
        for key, value in META_RE.findall(text[match.end():end]):
            if key in fields:
                fields[key] = "\0duplicate\0"
            else:
                fields[key] = value.strip()
        result.append((match.group(1), match.group(2).strip(), fields))
    return result


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _git(*args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *args), cwd=cwd, text=True, capture_output=True,
        check=False, timeout=60,
    )


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def validate(*, check_git: bool) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    required_files = (
        PLAN, OBJECTIVES, BOARD, CONFIG, BASELINE, REQUIREMENTS, PROGRAMS,
        OPERATOR, BOOTSTRAP_TEST,
    )
    for path in required_files:
        if not path.is_file():
            errors.append(f"missing required control file: {path.relative_to(ROOT)}")
    if errors:
        return {"schema": SCHEMA, "valid": False, "errors": errors, "warnings": warnings}

    try:
        config = _json(CONFIG)
        baseline = _json(BASELINE)
        requirements = _json(REQUIREMENTS)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"control JSON invalid: {type(exc).__name__}: {exc}")
        return {"schema": SCHEMA, "valid": False, "errors": errors, "warnings": warnings}

    task_blocks = _blocks(_read(BOARD), TASK_RE)
    goal_blocks = _blocks(_read(OBJECTIVES), GOAL_RE)
    actual_task_ids = tuple(item[0] for item in task_blocks)
    actual_goal_ids = tuple(item[0] for item in goal_blocks)
    if actual_task_ids != EXPECTED_TASKS:
        errors.append("task IDs/order differ from the sealed 40-task set")
    if actual_goal_ids != EXPECTED_GOALS:
        errors.append("goal IDs/order differ from the sealed 9-goal heap")

    task_fields: dict[str, dict[str, str]] = {}
    observed: set[str] = set()
    output_owners: dict[tuple[str, str], str] = {}
    projected_output_owners: dict[str, str] = {}
    dependency_ancestors: dict[str, set[str]] = {}
    for task_id in EXPECTED_TASKS:
        ancestors: set[str] = set()
        for dependency in EXPECTED_DEPENDENCIES.get(task_id, ()):
            ancestors.add(dependency)
            ancestors.update(dependency_ancestors.get(dependency, set()))
        dependency_ancestors[task_id] = ancestors
    for task_id, title, fields in task_blocks:
        task_fields[task_id] = fields
        if title != EXPECTED_TITLES.get(task_id):
            errors.append(f"{task_id}: title differs")
        missing = [field for field in REQUIRED_TASK_FIELDS if field not in fields]
        if missing:
            errors.append(f"{task_id}: missing fields {missing}")
            continue
        duplicates = [field for field, value in fields.items() if value == "\0duplicate\0"]
        if duplicates:
            errors.append(f"{task_id}: duplicate fields {duplicates}")
        if fields["Stable task ID"] != task_id:
            errors.append(f"{task_id}: stable identity differs")
        if fields["Status"] != "todo" or fields["Completion"] != "auto":
            errors.append(f"{task_id}: initial status/completion differs")
        if fields["Is schedulable"] != "true" or fields["Review only"] != "false":
            errors.append(f"{task_id}: scheduling flags differ")
        if fields["Board namespace"] != PROGRAM or fields["Base plan revision"] != PLAN_REVISION:
            errors.append(f"{task_id}: board or plan binding differs")
        expected_goal = EXPECTED_TASK_GOALS.get(task_id)
        if (
            fields["Goal id"] != expected_goal
            or fields["Subgoal ID"] != expected_goal
            or fields["Parent goal ID"] != "ASEH-G000"
        ):
            errors.append(f"{task_id}: goal/subgoal/parent binding differs")
        repo = fields["Owning repository"]
        if repo not in BASES:
            errors.append(f"{task_id}: unknown owning repository {repo!r}")
        elif (fields["Base revision"], fields["Base repository tree"]) != BASES[repo]:
            errors.append(f"{task_id}: exact repository baseline differs")
        expected_deps = EXPECTED_DEPENDENCIES.get(task_id, [])
        dependencies = _csv(fields["Depends on"])
        if dependencies != expected_deps:
            errors.append(f"{task_id}: dependencies differ: {dependencies!r}")
        future = [item for item in dependencies if item not in observed]
        if future:
            errors.append(f"{task_id}: dependency must precede task: {future}")
        observed.add(task_id)
        if fields["Risk class"] not in RISK_CLASSES:
            errors.append(f"{task_id}: risk class is outside the closed vocabulary")
        if not fields["Authority requirement"] or not fields["Terminal success criteria"] or not fields["Terminal non-success criteria"]:
            errors.append(f"{task_id}: authority or terminal contract is empty")
        if not fields["Validation"].startswith("python3 "):
            errors.append(f"{task_id}: validation command is not exact python3 invocation")
        outputs = _csv(fields["Exact declared outputs"])
        if outputs != _csv(fields["Outputs"]) or not outputs:
            errors.append(f"{task_id}: output declarations differ or are empty")
        if outputs != _csv(fields["Owned paths"]):
            errors.append(f"{task_id}: owned paths differ from exact outputs")
        if outputs != _csv(fields["Predicted files"]):
            errors.append(f"{task_id}: predicted files differ from exact outputs")
        for output in outputs:
            pure = PurePosixPath(output)
            if (
                pure.is_absolute()
                or ".." in pure.parts
                or "." in pure.parts
                or not pure.parts
                or output.endswith("/")
                or pure.as_posix() != output
            ):
                errors.append(f"{task_id}: unsafe output path {output!r}")
                continue
            # Outputs are relative to the owning repository.  The canonical
            # DatabasePortalBridge prepends a sibling repository namespace
            # exactly once, so a datasets-local package path intentionally
            # projects as ipfs_datasets_py/ipfs_datasets_py/... in the outer
            # worktree.
            projected = output if repo == "ipfs_accelerate_py" else f"{repo}/{output}"
            projected_output_owners[projected] = task_id
            key = (repo, output)
            prior = output_owners.get(key)
            if prior and prior not in dependency_ancestors.get(task_id, set()):
                errors.append(f"{task_id}: output collision with unordered {prior}: {output}")
            output_owners[key] = task_id

    required_outputs = {
        "ASEH-000": {
            "docs/architecture/decisions/0007-agent-supervisor-efficiency-state-authorities.md",
        },
        "ASEH-001": {
            "benchmarks/agent_supervisor/efficiency_state_hardening/provider_model_config.json",
            "benchmarks/agent_supervisor/efficiency_state_hardening/price_snapshot.json",
            "benchmarks/agent_supervisor/efficiency_state_hardening/environment_identity.json",
            "benchmarks/agent_supervisor/efficiency_state_hardening/sealed_input_manifest.json",
        },
        "ASEH-010": {
            "ipfs_accelerate_py/agent_supervisor/runtime/schemas/task_efficiency_receipt.schema.json",
            "ipfs_accelerate_py/agent_supervisor/runtime/schemas/paired_benchmark_manifest.schema.json",
        },
        "ASEH-011": {
            "ipfs_accelerate_py/agent_supervisor/runtime/benchmark_telemetry.py",
        },
        "ASEH-012": {
            "ipfs_accelerate_py/agent_supervisor/runtime/benchmark_telemetry.py",
        },
        "ASEH-013": {
            "benchmarks/agent_supervisor/efficiency_state_hardening/hermetic_manifest.json",
            "benchmarks/agent_supervisor/efficiency_state_hardening/hermetic_vectors.jsonl",
        },
        "ASEH-021": {
            "ipfs_accelerate_py/agent_supervisor/semantic_state/schemas/unresolved_question.schema.json",
        },
        "ASEH-023": {
            "ipfs_accelerate_py/agent_supervisor/verification/model_route.py",
            "ipfs_accelerate_py/agent_supervisor/autonomy/route_policy.py",
        },
        "ASEH-024": {
            "ipfs_accelerate_py/agent_supervisor/runtime/decision_receipts.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/schemas/routing_decision.schema.json",
        },
        "ASEH-030": {
            "ipfs_datasets_py/proof_context/schemas/context_pack.schema.json",
        },
        "ASEH-032": {
            "ipfs_kit_py/proof_context/state_store.py",
            "ipfs_kit_py/proof_context/incremental_seal_store.py",
            "ipfs_kit_py/proof_context/verification_store.py",
        },
        "ASEH-040": {
            "ipfs_accelerate_py/agent_supervisor/control/schemas/state_transition.schema.json",
            "ipfs_accelerate_py/agent_supervisor/control/schemas/state_transition_table.json",
        },
        "ASEH-043": {
            "ipfs_accelerate_py/agent_supervisor/control/provider_attempt_store.py",
            "ipfs_accelerate_py/agent_supervisor/rescue/database_recovery.py",
            "ipfs_accelerate_py/agent_supervisor/control/schemas/recovery_decision.schema.json",
        },
        "ASEH-044": {
            "ipfs_accelerate_py/agent_supervisor/rescue/supervisor_recovery.py",
        },
        "ASEH-050": {
            "ipfs_accelerate_py/agent_supervisor/analysis/semantic_dependency_graph.py",
            "ipfs_accelerate_py/agent_supervisor/analysis/contract_change_impact.py",
            "ipfs_accelerate_py/agent_supervisor/analysis/schema_protocol_change_impact.py",
        },
        "ASEH-054": {
            "ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_synthesis.py",
            "ipfs_accelerate_py/agent_supervisor/planning/deterministic_doctor_transforms.py",
        },
        "ASEH-055": {
            "ipfs_accelerate_py/agent_supervisor/merge/schemas/patch_plan.schema.json",
        },
        "ASEH-061": {
            "ipfs_accelerate_py/agent_supervisor/task_sources/intent_repository.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py",
        },
        "ASEH-074": {
            "ipfs_accelerate_py/agent_supervisor/control/schemas/promotion_decision.schema.json",
        },
        "ASEH-075": {
            "test/api/agent_supervisor/efficiency_state_hardening/test_release_report.py",
        },
    }
    for task_id, required in required_outputs.items():
        observed_outputs = set(_csv(task_fields.get(task_id, {}).get("Exact declared outputs", "")))
        missing_outputs = sorted(required - observed_outputs)
        if missing_outputs:
            errors.append(f"{task_id}: required deliverable outputs absent: {missing_outputs}")
    forbidden_outputs = {
        "ipfs_accelerate_py/agent_supervisor/control/authority_registry.py",
        "ipfs_accelerate_py/agent_supervisor/control/compatibility_adapters.py",
        "ipfs_accelerate_py/agent_supervisor/control/model_escalation_policy.py",
        "ipfs_accelerate_py/agent_supervisor/control/provider_reconciliation.py",
        "ipfs_accelerate_py/agent_supervisor/control/state_recovery.py",
        "ipfs_accelerate_py/agent_supervisor/analysis/change_impact.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/provider_usage_telemetry.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/route_receipts.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/work_telemetry.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/deterministic_repair.py",
        "ipfs_kit_py/mcp_server/mcplusplus/context_pack_store.py",
    }
    for task_id, fields in task_fields.items():
        collisions = sorted(
            forbidden_outputs.intersection(_csv(fields.get("Exact declared outputs", "")))
        )
        if collisions:
            errors.append(f"{task_id}: competing-path outputs are forbidden: {collisions}")
    if not task_fields.get("ASEH-061", {}).get("Maintenance prerequisite"):
        errors.append("ASEH-061: owner-paused maintenance prerequisite is absent")

    for goal_id, _title, fields in goal_blocks:
        missing = [field for field in REQUIRED_GOAL_FIELDS if field not in fields]
        if missing:
            errors.append(f"{goal_id}: missing fields {missing}")
            continue
        if fields["Parent"] != EXPECTED_PARENTS[goal_id]:
            errors.append(f"{goal_id}: parent differs")
        dependencies = _csv(fields["Depends on"])
        if dependencies != EXPECTED_GOAL_DEPENDENCIES[goal_id]:
            errors.append(f"{goal_id}: goal dependencies differ: {dependencies!r}")
        if fields["Status"] != "active":
            errors.append(f"{goal_id}: initial goal status differs")

    if config.get("program_identifier") != PROGRAM or config.get("board_namespace") != PROGRAM:
        errors.append("scheduler program/namespace differs")
    if config.get("task_prefix") != "ASEH-" or config.get("goal_prefix") != "ASEH-G":
        errors.append("scheduler prefixes differ")
    if config.get("merge_target_branch") != BRANCH:
        errors.append("scheduler branch differs")
    projection = config.get("initial_projection")
    if not isinstance(projection, Mapping):
        errors.append("initial_projection is absent")
        projection = {}
    expected_dep_count = sum(len(value) for value in EXPECTED_DEPENDENCIES.values())
    expected_projection = {
        "task_count": 40, "task_dependency_count": expected_dep_count,
        "completed_task_ids": [], "ready_task_ids": ["ASEH-000", "ASEH-001"],
        "blocked_task_ids": [], "terminal_task_id": "ASEH-075",
        "goal_count": 9, "root_goal_id": "ASEH-G000",
    }
    if dict(projection) != expected_projection:
        errors.append("initial projection differs from computed sealed frontier")
    if config.get("task_groups") != EXPECTED_GROUPS:
        errors.append("scheduler task_groups differ")
    hierarchy = config.get("goal_hierarchy")
    if hierarchy != {"ASEH-G000": list(EXPECTED_GOALS[1:])}:
        errors.append("scheduler goal_hierarchy differs")
    waves = config.get("waves")
    wave_tasks = [
        task for wave in waves if isinstance(waves, list) and isinstance(wave, Mapping)
        for task in wave.get("task_ids", [])
    ] if isinstance(waves, list) else []
    if sorted(wave_tasks) != sorted(EXPECTED_TASKS) or len(wave_tasks) != len(set(wave_tasks)):
        errors.append("scheduler waves do not partition the exact task set")
    if isinstance(waves, list):
        wave_ids = [wave.get("id") for wave in waves if isinstance(wave, Mapping)]
        if wave_ids != [f"W{index}" for index in range(len(wave_ids))]:
            errors.append("scheduler wave identities are not contiguous and ordered")
        wave_index = {
            task_id: index
            for index, wave in enumerate(waves)
            if isinstance(wave, Mapping)
            for task_id in wave.get("task_ids", [])
        }
        for task_id, dependencies in EXPECTED_DEPENDENCIES.items():
            for dependency in dependencies:
                if wave_index.get(dependency, -1) >= wave_index.get(task_id, -1):
                    errors.append(
                        f"scheduler wave order violates {dependency} -> {task_id}"
                    )
    program = config.get("database_program")
    if not isinstance(program, Mapping) or any((
        program.get("authority_mode") != "quack",
        program.get("task_source_kind") != "duckdb",
        program.get("failover_policy") != "fail_closed",
        program.get("store_id") != "data/aseh/control.duckdb",
        program.get("quack_endpoint") != "quack:127.0.0.1:41487",
        program.get("runtime_registry_path") != "data/aseh/q",
    )):
        errors.append("database program does not seal DuckDB + typed Quack fail-closed authority")
    provider = config.get("provider")
    if not isinstance(provider, Mapping) or provider.get("fallback_trigger") != "primary_quota_exhausted":
        errors.append("provider fallback is not restricted to primary quota exhaustion")
    ducklake = config.get("ducklake_projection_program")
    if not isinstance(ducklake, Mapping) or any(
        ducklake.get(field) is not False
        for field in ("authority", "scheduling_prerequisite", "acceptance_prerequisite", "completion_prerequisite")
    ):
        errors.append("DuckLake is not sealed as non-authoritative")
    repair = config.get("bootstrap_handoff_repair")
    if not isinstance(repair, Mapping):
        errors.append("narrow bootstrap handoff repair is not sealed")
    else:
        missing_repair_fields = sorted(BOOTSTRAP_TASK_FIELDS - set(repair))
        if missing_repair_fields:
            errors.append(
                f"bootstrap repair task contract is incomplete: {missing_repair_fields}"
            )
        if any((
            repair.get("task_id") != "ASEH-BOOTSTRAP-001",
            repair.get("stable_identity")
            != f"{PROGRAM}/ASEH-BOOTSTRAP-001@{PLAN_REVISION}",
            repair.get("owning_repository") != "ipfs_accelerate_py",
            repair.get("dependencies") != [],
            set(repair.get("exact_declared_outputs", [])) != BOOTSTRAP_OUTPUTS,
            repair.get("risk_class") != "R4_SECURITY_OR_PROTOCOL_SENSITIVE",
            not repair.get("authority_requirement"),
            not repair.get("terminal_success_criteria"),
            not repair.get("terminal_non_success_criteria"),
            repair.get("validation")
            != "python3 -m pytest -q test/api/test_agent_supervisor_configured_typed_grant_handoff.py",
            repair.get("creates_new_supervisor") is not False,
            repair.get("creates_new_store") is not False,
            repair.get("required_before_materialization") is not True,
        )):
            errors.append("bootstrap repair task contract differs from the sealed narrow repair")
    if config.get("worktree_submodule_paths") != ["ipfs_datasets_py", "ipfs_kit_py"]:
        errors.append("cross-repository worktree paths differ")
    source_binding = config.get("source_binding")
    expected_source_binding = {
        "accelerator_planning_revision": BASES["ipfs_accelerate_py"][0],
        "accelerator_planning_tree": BASES["ipfs_accelerate_py"][1],
        "accelerator_selected_remote_ref": SELECTED_REFS["ipfs_accelerate_py"],
        "datasets_planning_revision": BASES["ipfs_datasets_py"][0],
        "datasets_planning_tree": BASES["ipfs_datasets_py"][1],
        "datasets_selected_remote_ref": SELECTED_REFS["ipfs_datasets_py"],
        "datasets_gitlink_at_accelerator_baseline": BASELINE_GITLINKS["ipfs_datasets_py"],
        "kit_planning_revision": BASES["ipfs_kit_py"][0],
        "kit_planning_tree": BASES["ipfs_kit_py"][1],
        "kit_selected_remote_ref": SELECTED_REFS["ipfs_kit_py"],
        "kit_gitlink_at_accelerator_baseline": BASELINE_GITLINKS["ipfs_kit_py"],
    }
    if not isinstance(source_binding, Mapping):
        errors.append("source binding is absent")
    else:
        for field, value in expected_source_binding.items():
            if source_binding.get(field) != value:
                errors.append(f"source binding differs for {field}")

    requirements_relative = REQUIREMENTS.relative_to(ROOT).as_posix()
    if config.get("requirements_path") != requirements_relative:
        errors.append("scheduler requirements path differs")
    if config.get("requirements_digest") != _digest(REQUIREMENTS):
        errors.append("scheduler requirements digest differs from protected matrix")
    if any((
        requirements.get("schema")
        != "ipfs_accelerate_py/agent-supervisor/aseh-protected-requirements@1",
        requirements.get("program_id") != PROGRAM,
        requirements.get("plan_revision") != PLAN_REVISION,
        len(requirements.get("truth_and_safety_invariants", [])) != 15,
        len(requirements.get("promotion_hard_gates_required_zero", [])) != 14,
        len(requirements.get("scope_freeze_forbidden_new_categories", [])) != 13,
        len(requirements.get("required_deliverables", [])) != 12,
        len(requirements.get("final_report_required_fields", [])) != 18,
        len(requirements.get("completion_criteria", [])) != 12,
    )):
        errors.append("protected requirements identity or required cardinalities differ")
    state_machine = requirements.get("canonical_state_machine")
    if not isinstance(state_machine, Mapping) or "compensated" not in state_machine.get("states", []):
        errors.append("protected requirements omit the canonical compensated state")
    planning_requirements = requirements.get("analysis_planning_synthesis")
    required_plan_bindings = {
        "objective_and_revision", "assumptions", "guarantees",
        "affected_dependency_cone", "required_interfaces", "permitted_paths",
        "side_effect_class", "acceptance_conditions", "test_and_proof_obligations",
        "unresolved_questions", "execution_budget", "fallback_and_recovery_rules",
    }
    if not isinstance(planning_requirements, Mapping) or set(
        planning_requirements.get("task_plan_bindings", [])
    ) != required_plan_bindings:
        errors.append("protected requirements omit canonical task-plan bindings")
    routing_requirements = requirements.get("deterministic_first_routing")
    if not isinstance(routing_requirements, Mapping) or len(
        routing_requirements.get("normally_no_large_model_examples", [])
    ) != 12:
        errors.append("protected requirements omit deterministic no-large-model cases")
    if len(requirements.get("sealed_baseline_required_fields", [])) != 8:
        errors.append("protected requirements omit sealed-baseline identities")
    protected_paths = config.get("protected_paths")
    if not isinstance(protected_paths, list) or any(
        not isinstance(item, str) or not item for item in protected_paths
    ):
        errors.append("scheduler protected_paths are absent or malformed")
        protected_paths = []
    protected_output_conflicts = sorted(
        path for path in protected_paths if path in projected_output_owners
    )
    if protected_output_conflicts:
        errors.append(
            "task outputs are globally implementation-protected: "
            + ", ".join(protected_output_conflicts)
        )
    missing_protection = sorted(REQUIRED_PROTECTED_PATHS - set(protected_paths))
    if missing_protection:
        errors.append(f"bootstrap/control inputs are not protected: {missing_protection}")
    staged_integration_paths = {
        "ipfs_accelerate_py/agent_supervisor/task_sources/intent_repository.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py",
    }
    incorrectly_protected = sorted(staged_integration_paths.intersection(protected_paths))
    if incorrectly_protected:
        errors.append(
            "ASEH-061 exact integration outputs cannot be globally protected: "
            + ", ".join(incorrectly_protected)
        )

    baseline_repos = baseline.get("repositories")
    if not isinstance(baseline_repos, Mapping):
        errors.append("baseline repositories are absent")
    else:
        for repo, (commit, tree) in BASES.items():
            row = baseline_repos.get(repo)
            if (
                not isinstance(row, Mapping)
                or row.get("commit") != commit
                or row.get("tree") != tree
                or row.get("selected_ref") != SELECTED_REFS[repo]
            ):
                errors.append(f"baseline identity differs for {repo}")
        accelerator_row = baseline_repos.get("ipfs_accelerate_py", {})
        if not isinstance(accelerator_row, Mapping) or accelerator_row.get(
            "baseline_gitlinks"
        ) != BASELINE_GITLINKS:
            errors.append("accelerator baseline gitlink provenance differs")
    if baseline.get("baseline_kind") != "exact_accelerator_main_with_selected_sibling_authority_snapshots":
        errors.append("baseline kind incorrectly represents sibling snapshots as main")
    pending_forest = baseline.get("bootstrap_forest_binding")
    if not isinstance(pending_forest, Mapping) or any((
        pending_forest.get("status") != "pending_committed_bootstrap_identity",
        pending_forest.get("commit") != "unavailable_until_bootstrap_commit",
        pending_forest.get("tree") != "unavailable_until_bootstrap_commit",
        pending_forest.get("claim_is_runtime_or_completion_evidence") is not False,
    )):
        errors.append("pending bootstrap forest identity is not explicitly non-evidence")

    plan_text = _read(PLAN)
    required_plan_terms = (
        "non_promoted_unmeasured", "non_promoted_safety_or_quality",
        "non_promoted_efficiency", "promotion_eligible_operator_authorization_required",
        "60+", "20+", "10+", "DuckLake", "QuackStateServer@1",
        "DatabaseTaskSource@1", "DatasetsContextPackAuthority@0.1",
        "agent_supervisor_efficiency_state_hardening.requirements.json",
    )
    for term in required_plan_terms:
        if term not in plan_text:
            errors.append(f"plan omits required term {term!r}")
    if "materialize_agent_supervisor_efficiency_state_hardening_board.py" in plan_text:
        errors.append("plan names a nonexistent competing ASEH materializer")
    if "scripts/ops/agent_supervisor/quack_state_server.py" in plan_text:
        errors.append("plan starts a standalone owner without the ASEH grant handoff")
    board_text = _read(BOARD)
    if "rejected plan delta" in board_text:
        errors.append("board requires rejected rather than admitted plan deltas")
    if not task_fields.get("ASEH-015", {}).get("Enrollment deadline"):
        errors.append("ASEH-015: bounded immutable UTC enrollment deadline is absent")
    for term in ("--allow-honest-nonpromotion", "--allow-not-admitted"):
        if term not in board_text:
            errors.append(f"qualification board omits terminal negative-evidence route {term!r}")
    if "| `ASEH-` | **Agent-supervisor efficiency and state hardening**" not in _read(PROGRAMS):
        errors.append("ASEH program prefix is absent from the canonical programs glossary")

    if check_git:
        branch = _git("branch", "--show-current")
        if branch.returncode != 0 or branch.stdout.strip() != BRANCH:
            errors.append("current branch differs from sealed scheduler branch")
        root_status = _git("status", "--porcelain=v1", "--untracked-files=all")
        if root_status.returncode != 0 or root_status.stdout.strip():
            errors.append("accelerator launch worktree is not clean")
        for repo, (commit, tree) in BASES.items():
            cwd = ROOT if repo == "ipfs_accelerate_py" else ROOT / repo
            result = _git("merge-base", "--is-ancestor", commit, "HEAD", cwd=cwd)
            if result.returncode != 0:
                errors.append(f"{repo}: baseline commit is not an ancestor")
            commit_tree = _git("rev-parse", f"{commit}^{{tree}}", cwd=cwd)
            if commit_tree.returncode != 0 or commit_tree.stdout.strip() != tree:
                errors.append(f"{repo}: sealed commit/tree identity is invalid")
            if repo != "ipfs_accelerate_py":
                status = _git("status", "--porcelain=v1", "--untracked-files=all", cwd=cwd)
                if status.returncode != 0 or status.stdout.strip():
                    errors.append(f"{repo}: nested worktree is dirty")
                head = _git("rev-parse", "HEAD", cwd=cwd)
                head_tree = _git("rev-parse", "HEAD^{tree}", cwd=cwd)
                if (
                    head.returncode != 0
                    or head.stdout.strip() != commit
                    or head_tree.returncode != 0
                    or head_tree.stdout.strip() != tree
                ):
                    errors.append(f"{repo}: nested HEAD differs from selected exact snapshot")
                gitlink = _git("ls-tree", "HEAD", repo)
                if gitlink.returncode != 0 or head.returncode != 0:
                    errors.append(f"{repo}: gitlink/head unavailable")
                else:
                    fields = gitlink.stdout.split()
                    if len(fields) < 3 or fields[2] != head.stdout.strip():
                        errors.append(f"{repo}: gitlink does not equal nested HEAD")

    report = {
        "schema": SCHEMA,
        "valid": not errors,
        "program_id": PROGRAM,
        "plan_revision": PLAN_REVISION,
        "task_count": len(task_blocks),
        "goal_count": len(goal_blocks),
        "dependency_count": sum(len(value) for value in EXPECTED_DEPENDENCIES.values()),
        "initial_ready_task_ids": ["ASEH-000", "ASEH-001"],
        "owners": sorted(BASES),
        "bootstrap_repair_task_id": "ASEH-BOOTSTRAP-001",
        "control_digests": {
            path.relative_to(ROOT).as_posix(): _digest(path)
            for path in required_files
        },
        "errors": errors,
        "warnings": warnings,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-all", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = validate(check_git=bool(args.check_all))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
