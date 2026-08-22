#!/usr/bin/env python3
"""Fail-closed validator for the PCAR goal heap, task DAG, and scheduler seal.

This module deliberately has no project-package dependency.  It can validate
the bootstrap documents before importing the supervisor, and it emits one JSON
object suitable for ``configured_board_scheduler.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/"
    "proof-carrying-architecture-refactorer-board-validation@1"
)
PROGRAM_IDENTIFIER = "agent-supervisor-proof-carrying-architecture-refactorer-v1"
BOARD_NAMESPACE = PROGRAM_IDENTIFIER
PLAN_REVISION = "PCAR-PLAN-R1"
BASE_REVISION = "bbf7f68799072c2b81f7d96eac91f2df3c4b3952"
BASE_TREE = "a698da9e4b54e2929adacb613bc61ba3e72eed58"
BRANCH = "codex/proof-carrying-architecture-refactorer-v1"

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/architecture/AGENT_SUPERVISOR_PROOF_CARRYING_ARCHITECTURE_REFACTORER_PLAN.md"
OBJECTIVES = ROOT / "docs/architecture/agent_supervisor_architecture_refactorer.objectives.md"
BOARD = ROOT / "docs/architecture/agent_supervisor_architecture_refactorer.todo.md"
CONFIG = ROOT / "config/agent_supervisor_architecture_refactorer_scheduler.json"
INVENTORY_MANIFEST = (
    ROOT / "docs/architecture/architecture_refactorer_inventory/manifest.json"
)
BENCHMARK_MANIFEST = (
    ROOT / "benchmarks/agent_supervisor/architecture_refactorer/manifest.json"
)

TASK_RE = re.compile(r"^## (PCAR-(\d{3})) (.+)$", re.MULTILINE)
GOAL_RE = re.compile(r"^## (PCAR-G\d{3}) (.+)$", re.MULTILINE)
META_RE = re.compile(r"^- ([^:\n]+):(?: (.*))?$", re.MULTILINE)
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")

EXPECTED_TASK_TITLES = {
    "PCAR-000": "Seal current source and prerequisite baseline",
    "PCAR-001": "Inventory packages, entrypoints, authorities, and state stores",
    "PCAR-002": "Define ArchitectureIR and graph contracts",
    "PCAR-003": "Implement architecture graph extraction",
    "PCAR-004": "Define semantic-entropy metrics",
    "PCAR-005": "Implement dependency-cone and context-burden analysis",
    "PCAR-006": "Implement canonical authority ownership graph",
    "PCAR-007": "Implement duplicate-authority detection",
    "PCAR-008": "Implement semantic duplicate detector",
    "PCAR-009": "Implement bounded e-graph normalization",
    "PCAR-010": "Implement contract-candidate extraction",
    "PCAR-011": "Implement interface-boundary synthesis",
    "PCAR-012": "Define closed refactor-operator grammar",
    "PCAR-013": "Implement public-surface manifest",
    "PCAR-014": "Implement state-ownership model",
    "PCAR-015": "Inventory legacy, compatibility, fixture, and simulation paths",
    "PCAR-016": "Implement compatibility and simulation quarantine checks",
    "PCAR-017": "Implement isolated refactor-candidate execution",
    "PCAR-018": "Implement differential behavioral validation",
    "PCAR-019": "Implement effect and authority comparison",
    "PCAR-020": "Implement translation validation for generated adapters",
    "PCAR-021": "Integrate existing procedure compiler or expose future adapter",
    "PCAR-022": "Implement autonomous architecture-refactor planner",
    "PCAR-023": "Implement bounded autonomous refactor execution",
    "PCAR-024": "Implement architecture drift monitor",
    "PCAR-025": "Implement cross-repository read-only contract audit",
    "PCAR-026": "Generate current architecture projections",
    "PCAR-027": "Add control service, CLI, and MCP surfaces",
    "PCAR-028": "Build frozen context and architecture benchmark",
    "PCAR-029": "Run adversarial architecture-assurance campaign",
    "PCAR-030": "Implement promotion, rollback, and release gates",
    "PCAR-031": "Produce current-tree qualification and residual-gap report",
}
EXPECTED_TASK_IDS = tuple(EXPECTED_TASK_TITLES)
EXPECTED_GOAL_IDS = (
    "PCAR-G000",
    "PCAR-G010",
    "PCAR-G011",
    "PCAR-G012",
    "PCAR-G013",
    "PCAR-G020",
    "PCAR-G021",
    "PCAR-G022",
    "PCAR-G030",
    "PCAR-G031",
    "PCAR-G032",
    "PCAR-G040",
    "PCAR-G041",
    "PCAR-G042",
)
EXPECTED_PARENTS = {
    "PCAR-G000": "",
    "PCAR-G010": "PCAR-G000",
    "PCAR-G011": "PCAR-G010",
    "PCAR-G012": "PCAR-G010",
    "PCAR-G013": "PCAR-G010",
    "PCAR-G020": "PCAR-G000",
    "PCAR-G021": "PCAR-G020",
    "PCAR-G022": "PCAR-G020",
    "PCAR-G030": "PCAR-G000",
    "PCAR-G031": "PCAR-G030",
    "PCAR-G032": "PCAR-G030",
    "PCAR-G040": "PCAR-G000",
    "PCAR-G041": "PCAR-G040",
    "PCAR-G042": "PCAR-G040",
}
EXPECTED_GOAL_HIERARCHY = {
    "PCAR-G000": ["PCAR-G010", "PCAR-G020", "PCAR-G030", "PCAR-G040"],
    "PCAR-G010": ["PCAR-G011", "PCAR-G012", "PCAR-G013"],
    "PCAR-G020": ["PCAR-G021", "PCAR-G022"],
    "PCAR-G030": ["PCAR-G031", "PCAR-G032"],
    "PCAR-G040": ["PCAR-G041", "PCAR-G042"],
}
EXPECTED_TASK_GROUPS = {
    "PCAR-G011": ["PCAR-000", "PCAR-001"],
    "PCAR-G012": ["PCAR-002", "PCAR-003", "PCAR-004", "PCAR-005"],
    "PCAR-G013": ["PCAR-006", "PCAR-007"],
    "PCAR-G021": ["PCAR-008", "PCAR-009", "PCAR-010", "PCAR-011", "PCAR-012"],
    "PCAR-G022": ["PCAR-013", "PCAR-014", "PCAR-015", "PCAR-016"],
    "PCAR-G031": ["PCAR-017", "PCAR-018", "PCAR-019", "PCAR-020", "PCAR-021"],
    "PCAR-G032": ["PCAR-022", "PCAR-023", "PCAR-024", "PCAR-025"],
    "PCAR-G041": ["PCAR-026", "PCAR-027"],
    "PCAR-G042": ["PCAR-028", "PCAR-029", "PCAR-030", "PCAR-031"],
}
EXPECTED_DEPENDENCIES = {
    "PCAR-000": [],
    "PCAR-001": ["PCAR-000"],
    "PCAR-002": ["PCAR-000"],
    "PCAR-003": ["PCAR-001", "PCAR-002"],
    "PCAR-004": ["PCAR-002"],
    "PCAR-005": ["PCAR-003", "PCAR-004"],
    "PCAR-006": ["PCAR-001", "PCAR-003"],
    "PCAR-007": ["PCAR-006"],
    "PCAR-008": ["PCAR-003", "PCAR-007"],
    "PCAR-009": ["PCAR-002", "PCAR-008"],
    "PCAR-010": ["PCAR-002", "PCAR-003", "PCAR-007"],
    "PCAR-011": ["PCAR-004", "PCAR-006", "PCAR-010"],
    "PCAR-012": ["PCAR-002", "PCAR-006", "PCAR-010", "PCAR-011"],
    "PCAR-013": ["PCAR-001", "PCAR-003", "PCAR-006"],
    "PCAR-014": ["PCAR-001", "PCAR-003", "PCAR-006"],
    "PCAR-015": ["PCAR-001", "PCAR-003", "PCAR-006"],
    "PCAR-016": ["PCAR-012", "PCAR-013", "PCAR-014", "PCAR-015"],
    "PCAR-017": ["PCAR-012", "PCAR-014", "PCAR-016"],
    "PCAR-018": ["PCAR-017"],
    "PCAR-019": ["PCAR-006", "PCAR-014", "PCAR-018"],
    "PCAR-020": ["PCAR-009", "PCAR-018", "PCAR-019"],
    "PCAR-021": ["PCAR-012", "PCAR-020"],
    "PCAR-022": ["PCAR-005", "PCAR-011", "PCAR-012", "PCAR-014", "PCAR-019", "PCAR-020"],
    "PCAR-023": ["PCAR-017", "PCAR-018", "PCAR-019", "PCAR-020", "PCAR-021", "PCAR-022"],
    "PCAR-024": ["PCAR-003", "PCAR-005", "PCAR-006", "PCAR-007", "PCAR-013", "PCAR-014", "PCAR-015", "PCAR-016"],
    "PCAR-025": ["PCAR-000", "PCAR-001", "PCAR-002"],
    "PCAR-026": ["PCAR-003", "PCAR-004", "PCAR-005", "PCAR-006", "PCAR-007", "PCAR-013", "PCAR-014", "PCAR-015", "PCAR-024", "PCAR-025"],
    "PCAR-027": ["PCAR-002", "PCAR-006", "PCAR-013", "PCAR-014", "PCAR-016", "PCAR-022", "PCAR-023", "PCAR-024"],
    "PCAR-028": ["PCAR-005", "PCAR-018", "PCAR-019", "PCAR-023", "PCAR-027"],
    "PCAR-029": ["PCAR-016", "PCAR-018", "PCAR-019", "PCAR-020", "PCAR-021", "PCAR-022", "PCAR-023", "PCAR-024", "PCAR-027", "PCAR-028"],
    "PCAR-030": ["PCAR-017", "PCAR-018", "PCAR-019", "PCAR-020", "PCAR-021", "PCAR-022", "PCAR-023", "PCAR-028", "PCAR-029"],
    "PCAR-031": ["PCAR-026", "PCAR-027", "PCAR-028", "PCAR-029", "PCAR-030"],
}

REQUIRED_TASK_FIELDS = (
    "Stable task ID",
    "Status",
    "Completion",
    "Is schedulable",
    "Review only",
    "Priority",
    "Track",
    "Goal id",
    "Parent goal ID",
    "Subgoal ID",
    "Owning repository",
    "Board namespace",
    "Base revision",
    "Base repository tree",
    "Base plan revision",
    "Objective",
    "Depends on",
    "Owned paths",
    "Predicted files",
    "Predicted symbols",
    "Read scope",
    "Write scope",
    "External effect scope",
    "Authority impact",
    "Effect class",
    "Public API impact",
    "State impact",
    "Preconditions",
    "Permitted effects",
    "Prohibited effects",
    "Resource class",
    "Token budget",
    "Resource demand",
    "Model-route class",
    "Parallel lane",
    "Concurrency group",
    "Conflict policy",
    "Lease and fencing",
    "Acceptance subset",
    "Completion contract",
    "Validation",
    "Proof requirements",
    "Rollback",
    "Required evidence",
    "Final result identity",
    "Outputs",
    "Raw-source requirements",
    "Capability blockers",
)
REQUIRED_GOAL_FIELDS = (
    "Status",
    "Parent",
    "Depends on",
    "Priority",
    "Track",
    "Goal",
    "Completion contract",
    "Evidence",
    "Acceptance criteria",
    "Outputs",
    "Validation",
    "Acceptance",
    "Gap task",
)
RESOURCE_FIELDS = {
    "cpu_ms",
    "cpu_concurrency",
    "ram_mib",
    "gpu_memory_mib",
    "gpu_compute_class",
    "disk_mib",
    "disk_bandwidth_mib_s",
    "network",
    "network_bandwidth_kib_s",
    "subprocesses",
    "worktree_slots",
    "provider_quota_units",
    "provider_concurrency",
    "prover_class",
    "prover_concurrency",
    "exclusive_keys",
    "merge_slots",
    "persistence_kib_s",
}
RESOURCE_TEXT_FIELDS = {"gpu_compute_class", "network", "prover_class", "exclusive_keys"}
PROTECTED_PATHS = {
    "docs/architecture/AGENT_SUPERVISOR_PROOF_CARRYING_ARCHITECTURE_REFACTORER_PLAN.md",
    "docs/architecture/agent_supervisor_architecture_refactorer.objectives.md",
    "docs/architecture/agent_supervisor_architecture_refactorer.todo.md",
    "config/agent_supervisor_architecture_refactorer_scheduler.json",
    "scripts/validate_agent_supervisor_architecture_refactorer_board.py",
    "scripts/run_agent_supervisor_architecture_refactorer.py",
    "test/api/architecture_refactorer/test_board.py",
}
READ_ONLY_PREFIXES = ("ipfs_datasets_py", "ipfs_kit_py", "ipfs_accelerate_py/mcplusplus")


def _identity(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _read_text(path: Path, errors: list[str]) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(f"missing/unreadable {path.relative_to(ROOT)}: {exc}")
        return ""


def _read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid JSON {path.relative_to(ROOT)}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"JSON root is not an object: {path.relative_to(ROOT)}")
        return {}
    return value


def _safe_source_path(value: Any, *, label: str, errors: list[str]) -> Path | None:
    text = str(value or "").strip()
    candidate = PurePosixPath(text)
    if not text or candidate.is_absolute() or ".." in candidate.parts:
        errors.append(f"{label} must be a safe repository-relative path")
        return None
    path = (ROOT / Path(*candidate.parts)).resolve(strict=False)
    try:
        path.relative_to(ROOT)
    except ValueError:
        errors.append(f"{label} escapes repository")
        return None
    if path.is_symlink() or not path.is_file():
        errors.append(f"{label} is not a regular repository file: {text}")
        return None
    return path


def _validate_required_artifacts(errors: list[str]) -> dict[str, str]:
    """Validate compact required inventories and the frozen benchmark corpus."""

    identities: dict[str, str] = {}
    inventory = _read_json(INVENTORY_MANIFEST, errors)
    if inventory.get("schema") != (
        "ipfs_accelerate_py.agent_supervisor."
        "architecture-refactorer.inventory-manifest@1"
    ):
        errors.append("inventory manifest schema mismatch")
    inventory_root = INVENTORY_MANIFEST.parent
    entries = inventory.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append("inventory manifest entries must be a nonempty list")
        entries = []
    for raw_entry in entries:
        name = str(raw_entry or "").strip()
        candidate = PurePosixPath(name)
        if (
            not name
            or candidate.is_absolute()
            or ".." in candidate.parts
            or len(candidate.parts) != 1
        ):
            errors.append(f"unsafe inventory manifest entry: {name!r}")
            continue
        path = inventory_root / name
        payload = _read_json(path, errors)
        if payload and payload.get("schema") in (None, ""):
            errors.append(f"inventory entry lacks schema: {name}")
        if path.is_file() and not path.is_symlink():
            identities[path.relative_to(ROOT).as_posix()] = _identity(path.read_bytes())

    benchmark = _read_json(BENCHMARK_MANIFEST, errors)
    if benchmark.get("schema") != (
        "ipfs_accelerate_py.agent_supervisor."
        "architecture-refactorer.frozen-benchmark@1"
    ):
        errors.append("benchmark manifest schema mismatch")
    if benchmark.get("frozen") is not True:
        errors.append("benchmark manifest must be frozen")
    cases_path = _safe_source_path(
        benchmark.get("cases_path"), label="benchmark cases_path", errors=errors
    )
    cases: list[Any] = []
    if cases_path is not None:
        cases_payload = _read_json(cases_path, errors)
        raw_cases = cases_payload.get("cases")
        if isinstance(raw_cases, list):
            cases = raw_cases
        else:
            errors.append("benchmark cases must be a list")
        identities[cases_path.relative_to(ROOT).as_posix()] = _identity(
            cases_path.read_bytes()
        )
    required_domains = set(benchmark.get("required_domains") or [])
    observed_domains = {
        str(case.get("domain") or "") for case in cases if isinstance(case, Mapping)
    }
    if observed_domains != required_domains:
        errors.append("benchmark cases do not cover the exact required domains")
    required_task_types = set(benchmark.get("required_task_types") or [])
    observed_task_types = {
        str(case.get("task_type") or "") for case in cases if isinstance(case, Mapping)
    }
    if not required_task_types.issubset(observed_task_types):
        errors.append("benchmark cases omit required task types")
    required_case_fields = {
        "case_id",
        "domain",
        "task_type",
        "objective",
        "acceptance_criteria",
        "required_evidence",
        "repository_tree",
        "provider_fixture",
        "tokenizer",
        "policy",
        "fault_schedule",
    }
    case_ids: list[str] = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            errors.append(f"benchmark case {index} is not an object")
            continue
        missing = sorted(required_case_fields - set(case))
        if missing:
            errors.append(f"benchmark case {index} missing fields {missing}")
        case_ids.append(str(case.get("case_id") or ""))
    if not case_ids or len(case_ids) != len(set(case_ids)) or "" in case_ids:
        errors.append("benchmark case IDs must be nonempty and unique")
    fixtures = benchmark.get("fixtures")
    if not isinstance(fixtures, Mapping):
        errors.append("benchmark fixtures must be an object")
        fixtures = {}
    if set(fixtures) != {"provider", "tokenizer", "policy", "fault_schedules"}:
        errors.append("benchmark fixture set mismatch")
    for name, raw_path in fixtures.items():
        path = _safe_source_path(
            raw_path, label=f"benchmark fixture {name}", errors=errors
        )
        if path is None:
            continue
        payload = _read_json(path, errors)
        if payload and payload.get("schema") in (None, ""):
            errors.append(f"benchmark fixture {name} lacks schema")
        identities[path.relative_to(ROOT).as_posix()] = _identity(path.read_bytes())
    if INVENTORY_MANIFEST.is_file():
        identities[INVENTORY_MANIFEST.relative_to(ROOT).as_posix()] = _identity(
            INVENTORY_MANIFEST.read_bytes()
        )
    if BENCHMARK_MANIFEST.is_file():
        identities[BENCHMARK_MANIFEST.relative_to(ROOT).as_posix()] = _identity(
            BENCHMARK_MANIFEST.read_bytes()
        )
    return dict(sorted(identities.items()))


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _metadata(body: str, *, record_id: str, errors: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for match in META_RE.finditer(body):
        key = match.group(1).strip()
        value = (match.group(2) or "").strip()
        if key in result:
            errors.append(f"{record_id}: duplicate field {key!r}")
        result[key] = value
    return result


def _parse_records(
    text: str,
    pattern: re.Pattern[str],
    *,
    errors: list[str],
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    matches = list(pattern.finditer(text))
    records: dict[str, dict[str, str]] = {}
    titles: dict[str, str] = {}
    for index, match in enumerate(matches):
        record_id = match.group(1)
        title = match.group(3 if pattern is TASK_RE else 2).strip()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        if record_id in records:
            errors.append(f"duplicate heading: {record_id}")
            continue
        records[record_id] = _metadata(
            text[match.end() : end], record_id=record_id, errors=errors
        )
        titles[record_id] = title
    return records, titles


def _safe_paths(value: str, *, task_id: str, field: str, errors: list[str]) -> list[str]:
    paths = _csv(value)
    if not paths:
        errors.append(f"{task_id}: {field} is empty")
    for item in paths:
        path = PurePosixPath(item)
        if (
            path.is_absolute()
            or item in {".", ".."}
            or ".." in path.parts
            or item.startswith("-")
            or any(character in item for character in "*?[]\x00")
        ):
            errors.append(f"{task_id}: unsafe {field} path {item!r}")
        if any(path.parts[: len(PurePosixPath(prefix).parts)] == PurePosixPath(prefix).parts for prefix in READ_ONLY_PREFIXES):
            errors.append(f"{task_id}: {field} writes read-only sibling/gitlink path {item!r}")
    if len(paths) != len(set(paths)):
        errors.append(f"{task_id}: duplicate {field} path")
    return paths


def _resource_vector(value: str, *, task_id: str, errors: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for segment in value.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        if "=" not in segment:
            errors.append(f"{task_id}: malformed resource segment {segment!r}")
            continue
        key, raw = (part.strip() for part in segment.split("=", 1))
        if key in result:
            errors.append(f"{task_id}: duplicate resource field {key!r}")
        result[key] = raw
    missing = sorted(RESOURCE_FIELDS - set(result))
    unknown = sorted(set(result) - RESOURCE_FIELDS)
    if missing:
        errors.append(f"{task_id}: resource vector missing {missing}")
    if unknown:
        errors.append(f"{task_id}: resource vector has unknown {unknown}")
    for field in sorted((RESOURCE_FIELDS - RESOURCE_TEXT_FIELDS) & set(result)):
        if re.fullmatch(r"\d+", result[field]) is None:
            errors.append(f"{task_id}: resource {field} must be a nonnegative integer")
    if result.get("network") not in {"deny", "read-only", "fetch-origin-main"}:
        errors.append(f"{task_id}: resource network must use a closed policy")
    return result


def _topological_layers(
    nodes: Sequence[str], dependencies: Mapping[str, Sequence[str]]
) -> tuple[list[str], list[list[str]]]:
    indegree = {node: 0 for node in nodes}
    children: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        for dependency in dependencies.get(node, ()):
            if dependency in indegree:
                indegree[node] += 1
                children[dependency].append(node)
    ready = sorted(node for node, degree in indegree.items() if degree == 0)
    topological: list[str] = []
    layers: list[list[str]] = []
    while ready:
        layer = ready
        layers.append(layer)
        next_ready: list[str] = []
        for node in layer:
            topological.append(node)
            for child in children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    next_ready.append(child)
        ready = sorted(next_ready)
    return topological, layers


def _dependency_closure(task_id: str, dependencies: Mapping[str, Sequence[str]]) -> set[str]:
    result: set[str] = set()
    pending = list(dependencies.get(task_id, ()))
    while pending:
        dependency = pending.pop()
        if dependency in result:
            continue
        result.add(dependency)
        pending.extend(dependencies.get(dependency, ()))
    return result


def _paths_overlap(left: str, right: str) -> bool:
    left_parts = PurePosixPath(left).parts
    right_parts = PurePosixPath(right).parts
    return (
        left_parts[: len(right_parts)] == right_parts
        or right_parts[: len(left_parts)] == left_parts
    )


def _git(arguments: Sequence[str], *, cwd: Path = ROOT) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 124, "", f"{type(exc).__name__}: {exc}"
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def validate_program(
    *,
    check_source: bool = False,
    require_database: bool = False,
) -> dict[str, Any]:
    """Validate the immutable bootstrap projection and optionally its checkout."""

    errors: list[str] = []
    warnings: list[str] = []
    plan_text = _read_text(PLAN, errors)
    objectives_text = _read_text(OBJECTIVES, errors)
    board_text = _read_text(BOARD, errors)
    config = _read_json(CONFIG, errors)
    required_artifact_identities = _validate_required_artifacts(errors)

    tasks, task_titles = _parse_records(board_text, TASK_RE, errors=errors)
    goals, _goal_titles = _parse_records(objectives_text, GOAL_RE, errors=errors)
    if tuple(tasks) != EXPECTED_TASK_IDS:
        errors.append(f"task population/order mismatch: {list(tasks)}")
    if tuple(goals) != EXPECTED_GOAL_IDS:
        errors.append(f"goal population/order mismatch: {list(goals)}")
    for task_id, expected_title in EXPECTED_TASK_TITLES.items():
        if task_titles.get(task_id) != expected_title:
            errors.append(f"{task_id}: title mismatch: {task_titles.get(task_id)!r}")

    dependencies: dict[str, list[str]] = {}
    owned: dict[str, list[str]] = {}
    for task_id, metadata in tasks.items():
        missing = [field for field in REQUIRED_TASK_FIELDS if field not in metadata]
        if missing:
            errors.append(f"{task_id}: missing fields {missing}")
        for field in REQUIRED_TASK_FIELDS:
            if field == "Depends on":
                continue
            if field in metadata and not metadata[field]:
                errors.append(f"{task_id}: empty required field {field!r}")
        if metadata.get("Stable task ID") != task_id:
            errors.append(f"{task_id}: Stable task ID mismatch")
        if metadata.get("Status") != "todo":
            errors.append(f"{task_id}: initial status must be todo, never blocked/completed")
        if (
            metadata.get("Completion") != "auto"
            or metadata.get("Is schedulable") != "true"
            or metadata.get("Review only") != "false"
        ):
            errors.append(f"{task_id}: every initial task must be automatically schedulable")
        if metadata.get("Board namespace") != BOARD_NAMESPACE:
            errors.append(f"{task_id}: board namespace mismatch")
        if metadata.get("Owning repository") != "ipfs_accelerate_py":
            errors.append(f"{task_id}: only the local accelerator repository may be owned")
        if metadata.get("Parent goal ID") != "PCAR-G000":
            errors.append(f"{task_id}: Parent goal ID must be PCAR-G000")
        goal_id = metadata.get("Goal id", "")
        if metadata.get("Subgoal ID") != goal_id or goal_id not in EXPECTED_TASK_GROUPS:
            errors.append(f"{task_id}: task must bind one configured leaf subgoal")
        if metadata.get("Base revision") != BASE_REVISION:
            errors.append(f"{task_id}: Base revision must name the exact starting commit")
        if metadata.get("Base repository tree") != BASE_TREE:
            errors.append(f"{task_id}: Base repository tree must name the exact starting tree")
        if metadata.get("Base plan revision") != PLAN_REVISION:
            errors.append(f"{task_id}: Base plan revision mismatch")
        parsed_dependencies = _csv(metadata.get("Depends on", ""))
        dependencies[task_id] = parsed_dependencies
        if parsed_dependencies != EXPECTED_DEPENDENCIES.get(task_id):
            errors.append(
                f"{task_id}: dependency list mismatch: {parsed_dependencies}; "
                f"expected {EXPECTED_DEPENDENCIES.get(task_id)}"
            )
        for dependency in parsed_dependencies:
            if dependency not in tasks:
                errors.append(f"{task_id}: unknown dependency {dependency}")
            elif dependency >= task_id:
                errors.append(f"{task_id}: dependency locality requires a lower PCAR task ID")
        owned_paths = _safe_paths(
            metadata.get("Owned paths", ""), task_id=task_id, field="Owned paths", errors=errors
        )
        predicted_paths = _safe_paths(
            metadata.get("Predicted files", ""), task_id=task_id, field="Predicted files", errors=errors
        )
        output_paths = _safe_paths(
            metadata.get("Outputs", ""), task_id=task_id, field="Outputs", errors=errors
        )
        owned[task_id] = owned_paths
        if owned_paths != predicted_paths or owned_paths != output_paths:
            errors.append(f"{task_id}: Owned paths, Predicted files and Outputs must match exactly")
        for path in owned_paths:
            if path in PROTECTED_PATHS:
                errors.append(f"{task_id}: workers may not own protected bootstrap control {path}")
        _resource_vector(metadata.get("Resource demand", ""), task_id=task_id, errors=errors)
        if re.search(r"\d", metadata.get("Token budget", "")) is None:
            errors.append(f"{task_id}: Token budget must include a numeric bound")
        if "lease" not in metadata.get("Lease and fencing", "").lower():
            errors.append(f"{task_id}: Lease and fencing must name a lease")
        if "fenc" not in metadata.get("Lease and fencing", "").lower():
            errors.append(f"{task_id}: Lease and fencing must name fencing")
        if "python" not in metadata.get("Validation", "").lower():
            errors.append(f"{task_id}: Validation must declare an executable Python command")
        if not metadata.get("Final result identity", "").lower().startswith("pending"):
            errors.append(f"{task_id}: initial Final result identity must remain pending")

    topological, layers = _topological_layers(list(tasks), dependencies)
    if len(topological) != len(tasks):
        errors.append("task dependency graph contains a cycle")
    edges = sorted(
        [dependency, task_id]
        for task_id, task_dependencies in dependencies.items()
        for dependency in task_dependencies
    )
    dependency_root = _identity(
        json.dumps(edges, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    completed = sorted(
        task_id for task_id, metadata in tasks.items() if metadata.get("Status") == "completed"
    )
    blocked = sorted(
        task_id for task_id, metadata in tasks.items() if metadata.get("Status") == "blocked"
    )
    ready = sorted(
        task_id
        for task_id, metadata in tasks.items()
        if metadata.get("Status") == "todo"
        and metadata.get("Is schedulable") == "true"
        and all(dependency in completed for dependency in dependencies.get(task_id, ()))
    )
    if completed:
        errors.append(f"initial completed task population must be empty: {completed}")
    if blocked:
        errors.append(f"initial blocked task population must be empty: {blocked}")
    if ready != ["PCAR-000"]:
        errors.append(f"initial ready task population must be exactly PCAR-000: {ready}")

    task_ids = list(tasks)
    for left_index, left_id in enumerate(task_ids):
        left_closure = _dependency_closure(left_id, dependencies)
        for right_id in task_ids[left_index + 1 :]:
            right_closure = _dependency_closure(right_id, dependencies)
            if left_id in right_closure or right_id in left_closure:
                continue
            for left_path in owned.get(left_id, ()):
                for right_path in owned.get(right_id, ()):
                    if _paths_overlap(left_path, right_path):
                        errors.append(
                            "unordered owned-path collision: "
                            f"{left_id}:{left_path} <> {right_id}:{right_path}"
                        )

    goal_dependencies: dict[str, list[str]] = {}
    for goal_id, metadata in goals.items():
        missing = [field for field in REQUIRED_GOAL_FIELDS if field not in metadata]
        if missing:
            errors.append(f"{goal_id}: missing fields {missing}")
        for field in REQUIRED_GOAL_FIELDS:
            if field in {"Parent", "Depends on"}:
                continue
            if field in metadata and not metadata[field]:
                errors.append(f"{goal_id}: empty required field {field!r}")
        if metadata.get("Status") != "active":
            errors.append(f"{goal_id}: initial goal status must be active")
        if metadata.get("Parent", "") != EXPECTED_PARENTS.get(goal_id):
            errors.append(
                f"{goal_id}: parent mismatch: {metadata.get('Parent', '')!r}; "
                f"expected {EXPECTED_PARENTS.get(goal_id)!r}"
            )
        goal_dependencies[goal_id] = _csv(metadata.get("Depends on", ""))
        for dependency in goal_dependencies[goal_id]:
            if dependency not in goals:
                errors.append(f"{goal_id}: unknown goal dependency {dependency}")
            elif dependency == goal_id:
                errors.append(f"{goal_id}: self dependency")
    goal_topological, _goal_layers = _topological_layers(list(goals), goal_dependencies)
    if len(goal_topological) != len(goals):
        errors.append("goal dependency graph contains a cycle")

    for leaf_goal, expected_tasks in EXPECTED_TASK_GROUPS.items():
        actual = [
            task_id
            for task_id, metadata in tasks.items()
            if metadata.get("Subgoal ID") == leaf_goal
        ]
        if actual != expected_tasks:
            errors.append(f"{leaf_goal}: task membership mismatch: {actual}")

    expected_paths = {
        "taskboard_path": "docs/architecture/agent_supervisor_architecture_refactorer.todo.md",
        "objectives_path": "docs/architecture/agent_supervisor_architecture_refactorer.objectives.md",
        "plan_path": "docs/architecture/AGENT_SUPERVISOR_PROOF_CARRYING_ARCHITECTURE_REFACTORER_PLAN.md",
        "validator_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
    }
    if config.get("schema") != (
        "ipfs_accelerate_py.agent_supervisor."
        "proof-carrying-architecture-refactorer.scheduler_config@1"
    ):
        errors.append("config scheduler schema mismatch")
    if config.get("program_identifier") != PROGRAM_IDENTIFIER:
        errors.append("config program identifier mismatch")
    if config.get("board_namespace") != BOARD_NAMESPACE:
        errors.append("config board namespace mismatch")
    for field, expected in expected_paths.items():
        if config.get(field) != expected:
            errors.append(f"config {field} mismatch")
    if config.get("task_prefix") != "PCAR-" or config.get("goal_prefix") != "PCAR-G":
        errors.append("config task/goal prefix mismatch")
    if config.get("accepted_plan_revision_alias") != PLAN_REVISION:
        errors.append("config plan revision mismatch")
    if config.get("merge_target_branch") != BRANCH:
        errors.append("config merge target branch mismatch")

    source_binding = config.get("source_binding")
    if not isinstance(source_binding, dict):
        errors.append("config source_binding is required")
        source_binding = {}
    expected_source = {
        "accelerator_required_ancestor": BASE_REVISION,
        "accelerator_planning_revision": BASE_REVISION,
        "accelerator_planning_tree": BASE_TREE,
        "accelerator_required_branch": BRANCH,
        "bootstrap_task_source": "duckdb",
        "datasets_submodule_path": "ipfs_datasets_py",
        "datasets_planning_revision": "480a1666f144ad606fcb3cacb66e59775f28d0d1",
        "kit_submodule_path": "ipfs_kit_py",
        "kit_planning_revision": "2564aea1ae35061f2165872aff91e8a40801ab7e",
        "mcp_plus_plus_submodule_path": "ipfs_accelerate_py/mcplusplus",
        "mcp_plus_plus_planning_revision": "5ac0ab162f420264fd224073a5df3f2d7c054ae3",
    }
    for field, expected in expected_source.items():
        if source_binding.get(field) != expected:
            errors.append(f"config source_binding.{field} mismatch")
    for field in (
        "require_clean_checkout_at_launch",
        "record_repository_revision_at_launch",
        "require_initialized_gitlinks",
        "require_superproject_gitlink_equals_nested_head",
        "require_clean_nested_worktree_at_task_start",
        "record_recursive_repository_forest_at_launch",
        "changed_revision_requires_fresh_inventory_and_baseline",
    ):
        if source_binding.get(field) is not True:
            errors.append(f"config source_binding.{field} must be true")
    if source_binding.get("planning_revision_is_runtime_completion_evidence") is not False:
        errors.append("planning revision must not be runtime completion evidence")

    projection = config.get("initial_projection")
    if not isinstance(projection, dict):
        errors.append("config initial_projection is required")
        projection = {}
    expected_projection = {
        "task_count": len(tasks),
        "task_dependency_count": len(edges),
        "task_dependency_root_cid": dependency_root,
        "completed_task_ids": completed,
        "ready_task_ids": ready,
        "blocked_task_ids": blocked,
        "terminal_task_id": "PCAR-031",
        "goal_count": len(goals),
        "root_goal_id": "PCAR-G000",
    }
    for field, expected in expected_projection.items():
        if projection.get(field) != expected:
            errors.append(
                f"config initial_projection.{field} mismatch: "
                f"{projection.get(field)!r}; expected {expected!r}"
            )

    database = config.get("database_program")
    if not isinstance(database, dict):
        errors.append("config database_program is required")
        database = {}
    expected_database = {
        "authority_mode": "quack",
        "task_source_kind": "duckdb",
        "endpoint_secret_handle": "handle:pcar-v1",
        "quack_endpoint": "quack:127.0.0.1:41317",
        "store_id": "data/agent_supervisor/proof_carrying_architecture_refactorer/control.duckdb",
        "store_generation": "pcar-v1",
        "schema_revision": "1",
        "failover_policy": "fail_closed",
        "explicit_legacy": False,
    }
    for field, expected in expected_database.items():
        if database.get(field) != expected:
            errors.append(f"config database_program.{field} mismatch")
    if not str(database.get("store_id") or "").endswith("/control.duckdb"):
        errors.append("database store_id must be a repository-relative control.duckdb")

    control = config.get("operational_control_plane")
    if not isinstance(control, dict):
        errors.append("config operational_control_plane is required")
        control = {}
    if control.get("direct_multi_process_duckdb_file_open_permitted") is not False:
        errors.append("direct multi-process DuckDB file opens must be prohibited")
    if control.get("automatic_file_fallback_permitted") is not False:
        errors.append("Quack outage must not silently fall back to file authority")
    if control.get("outage_policy") != "fail_closed":
        errors.append("operational control plane must fail closed")
    if control.get("markdown_is_authority") is not False:
        errors.append("Markdown must remain a bootstrap projection")

    ducklake = config.get("ducklake_projection_program")
    if not isinstance(ducklake, dict):
        errors.append("config ducklake_projection_program is required")
        ducklake = {}
    if ducklake.get("mode") != "enabled_non_authoritative":
        errors.append("DuckLake history/benchmark projection must be explicitly enabled")
    for field in (
        "authority",
        "scheduling_prerequisite",
        "acceptance_prerequisite",
        "completion_prerequisite",
        "may_grant_authority",
    ):
        if ducklake.get(field) is not False:
            errors.append(f"DuckLake projection {field} must be false")
    if "DuckDB" not in str(ducklake.get("source_authority") or ""):
        errors.append("DuckLake projection must replay from the DuckDB authority cursor")

    authority = config.get("authority_policy")
    if not isinstance(authority, dict):
        errors.append("config authority_policy is required")
        authority = {}
    required_true = (
        "duckdb_transactional_authority",
        "quack_exclusive_state_owner_transport",
        "one_authoritative_store_per_mutable_semantic_fact",
        "deterministic_current_tree_admission_required",
        "exact_receipt_identity_required",
        "sibling_repositories_are_read_only",
    )
    required_false = (
        "ducklake_projection_authority",
        "ducklake_projection_is_scheduling_prerequisite",
        "ducklake_projection_is_acceptance_prerequisite",
        "markdown_is_completion_authority",
        "task_board_status_is_completion_evidence",
        "direct_multi_process_duckdb_file_open_permitted",
        "automatic_quack_to_file_fallback",
        "provider_or_model_claim_is_completion_authority",
        "architecture_candidate_self_promotion",
        "procedure_self_authorization",
    )
    for field in required_true:
        if authority.get(field) is not True:
            errors.append(f"authority_policy.{field} must be true")
    for field in required_false:
        if authority.get(field) is not False:
            errors.append(f"authority_policy.{field} must be false")
    if authority.get("unknown_ownership_disposition") != "typed_blocker":
        errors.append("unknown architecture ownership must produce a typed blocker")

    if config.get("max_lanes") != 3 or config.get("strict_task_sharding") is not True:
        errors.append("config must use three strict conflict-aware lanes")
    if config.get("idle_lane_work_stealing") != "":
        errors.append(
            "idle work stealing must remain disabled because the current "
            "implementation entrypoint does not accept that optional flag"
        )
    for field in ("objective_refill_enabled", "codebase_refill_enabled", "objective_goal_refinement_enabled"):
        if config.get(field) is not False:
            errors.append(f"config {field} must be false for the sealed program")
    lanes = config.get("lanes")
    if not isinstance(lanes, list) or [lane.get("index") for lane in lanes if isinstance(lane, dict)] != [0, 1, 2]:
        errors.append("config lanes must be ordered 0, 1, 2")
    elif [lane.get("strict_shard_remainder") for lane in lanes] != [0, 1, 2]:
        errors.append("config strict shard remainders mismatch")
    elif lanes[0].get("initial_task_ids") != ["PCAR-000"] or any(
        lane.get("initial_task_ids") for lane in lanes[1:]
    ):
        errors.append("only PCAR-000 may appear in the initial lane projection")

    protected = config.get("protected_paths")
    if not isinstance(protected, list) or set(protected) != PROTECTED_PATHS:
        errors.append("config protected paths differ from the sealed bootstrap controls")
    if config.get("worktree_submodule_paths") != list(READ_ONLY_PREFIXES):
        errors.append("config worktree submodule paths mismatch")
    if config.get("goal_hierarchy") != EXPECTED_GOAL_HIERARCHY:
        errors.append("config goal hierarchy mismatch")
    if config.get("task_groups") != EXPECTED_TASK_GROUPS:
        errors.append("config task groups mismatch")
    configured_waves = config.get("waves")
    expected_waves = [
        {"id": "W0", "task_ids": ["PCAR-000"]},
        {"id": "W1", "task_ids": ["PCAR-001", "PCAR-002"]},
        {"id": "W2", "task_ids": ["PCAR-003", "PCAR-004", "PCAR-025"]},
        {"id": "W3", "task_ids": ["PCAR-005", "PCAR-006"]},
        {"id": "W4", "task_ids": ["PCAR-007", "PCAR-013", "PCAR-014", "PCAR-015"]},
        {"id": "W5", "task_ids": ["PCAR-008", "PCAR-010"]},
        {"id": "W6", "task_ids": ["PCAR-009", "PCAR-011"]},
        {"id": "W7", "task_ids": ["PCAR-012"]},
        {"id": "W8", "task_ids": ["PCAR-016"]},
        {"id": "W9", "task_ids": ["PCAR-017", "PCAR-024"]},
        {"id": "W10", "task_ids": ["PCAR-018"]},
        {"id": "W11", "task_ids": ["PCAR-019"]},
        {"id": "W12", "task_ids": ["PCAR-020"]},
        {"id": "W13", "task_ids": ["PCAR-021", "PCAR-022"]},
        {"id": "W14", "task_ids": ["PCAR-023"]},
        {"id": "W15", "task_ids": ["PCAR-026", "PCAR-027"]},
        {"id": "W16", "task_ids": ["PCAR-028"]},
        {"id": "W17", "task_ids": ["PCAR-029"]},
        {"id": "W18", "task_ids": ["PCAR-030"]},
        {"id": "W19", "task_ids": ["PCAR-031"]},
    ]
    if configured_waves != expected_waves:
        errors.append("config waves differ from the sealed conflict-aware schedule")
    wave_index: dict[str, int] = {}
    for index, wave in enumerate(configured_waves if isinstance(configured_waves, list) else []):
        if not isinstance(wave, dict) or not isinstance(wave.get("task_ids"), list):
            errors.append(f"config wave {index} is malformed")
            continue
        for task_id in wave["task_ids"]:
            if str(task_id) in wave_index:
                errors.append(f"task appears in multiple configured waves: {task_id}")
            wave_index[str(task_id)] = index
    if sorted(wave_index) != sorted(tasks):
        errors.append("configured waves must contain each task exactly once")
    for task_id, task_dependencies in dependencies.items():
        for dependency in task_dependencies:
            if wave_index.get(dependency, -1) >= wave_index.get(task_id, -1):
                errors.append(
                    f"configured wave violates dependency order: {dependency} -> {task_id}"
                )
    provider = config.get("provider")
    if not isinstance(provider, dict) or (
        provider.get("primary_provider_id") != "grok_cli"
        or provider.get("primary_model_id") != "grok-4.6"
        or provider.get("fallback_provider_id") != "codex"
        or provider.get("fallback_model_id") != "gpt-5.6-terra"
        or provider.get("fallback_trigger") != "primary_quota_exhausted"
        or provider.get("fallback_reasoning_effort") != "high"
        or provider.get("max_concurrency") != 3
        or provider.get("secrets_from_environment_only") is not True
        or provider.get("secrets_in_argv_prompts_logs_or_receipts") is not False
    ):
        errors.append(
            "provider must be the reviewed bounded Grok 4.6 quota route with "
            "Codex gpt-5.6-terra/high fallback and concurrency 3"
        )
    autonomy = config.get("autonomy_ceiling")
    if not isinstance(autonomy, dict) or (
        autonomy.get("candidate_may_raise_ceiling") is not False
        or autonomy.get("high_risk_requires_human_approval") is not True
        or autonomy.get("public_api_state_provider_receipt_or_cross_package_migration_mode") != "proposal_only"
    ):
        errors.append("config autonomy ceiling is incomplete or self-promotable")

    required_plan_phrases = (
        PROGRAM_IDENTIFIER,
        "ProofCarryingArchitectureRefactorer",
        "ArchitectureIR",
        "SemanticEntropyReport",
        "AuthorityOwnershipGraph",
        "NoAuthorityWeakening",
        "DuckDB",
        "Quack",
        "DuckLake",
        "PCAR-000..007",
    )
    for phrase in required_plan_phrases:
        if phrase not in plan_text:
            errors.append(f"plan missing required phrase {phrase!r}")
    if PROGRAM_IDENTIFIER not in objectives_text:
        errors.append("objectives do not bind the program identifier")
    if BOARD_NAMESPACE not in board_text:
        errors.append("task board does not bind the board namespace")

    branch = ""
    source_tree = ""
    if check_source:
        code, branch, detail = _git(["branch", "--show-current"])
        if code != 0 or branch != BRANCH:
            errors.append(f"wrong launch branch: {branch or detail}")
        code, _output, detail = _git(["merge-base", "--is-ancestor", BASE_REVISION, "HEAD"])
        if code != 0:
            errors.append(f"starting commit is not an ancestor of HEAD: {detail}")
        code, source_tree, detail = _git(["rev-parse", f"{BASE_REVISION}^{{tree}}"])
        if code != 0 or source_tree != BASE_TREE:
            errors.append(f"starting tree identity mismatch: {source_tree or detail}")
        for relative, expected in (
            ("ipfs_datasets_py", "480a1666f144ad606fcb3cacb66e59775f28d0d1"),
            ("ipfs_kit_py", "2564aea1ae35061f2165872aff91e8a40801ab7e"),
            ("ipfs_accelerate_py/mcplusplus", "5ac0ab162f420264fd224073a5df3f2d7c054ae3"),
        ):
            path = ROOT / relative
            code, head, detail = _git(["rev-parse", "HEAD"], cwd=path)
            if code != 0 or head != expected:
                errors.append(f"submodule {relative} identity mismatch: {head or detail}")
            code, dirty, detail = _git(["status", "--porcelain=v1", "--untracked-files=all"], cwd=path)
            if code != 0 or dirty:
                errors.append(f"submodule {relative} is not clean: {dirty or detail}")
    if require_database:
        store_path = ROOT / str(database.get("store_id") or "")
        try:
            stat = store_path.stat()
        except OSError as exc:
            errors.append(f"materialized DuckDB authority is unavailable: {exc}")
        else:
            if not store_path.is_file() or stat.st_size <= 0:
                errors.append("materialized DuckDB authority is not a nonempty regular file")

    artifact_identities: dict[str, str] = {}
    for path in (PLAN, OBJECTIVES, BOARD, CONFIG, Path(__file__).resolve()):
        try:
            try:
                label = path.relative_to(ROOT).as_posix()
            except ValueError:
                label = path.name
            artifact_identities[label] = _identity(path.read_bytes())
        except OSError:
            pass
    return {
        "schema": SCHEMA,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "program_identifier": PROGRAM_IDENTIFIER,
        "board_namespace": BOARD_NAMESPACE,
        "task_count": len(tasks),
        "goal_count": len(goals),
        "task_dependency_count": len(edges),
        "task_dependency_root_cid": dependency_root,
        "completed_task_ids": completed,
        "blocked_task_ids": blocked,
        "initial_ready_task_ids": ready,
        "terminal_task_id": "PCAR-031",
        "topological_task_ids": topological,
        "task_waves": expected_waves,
        "goal_topological_ids": goal_topological,
        "source_checks_performed": check_source,
        "database_presence_checked": require_database,
        "source": {
            "branch": branch,
            "starting_revision": BASE_REVISION,
            "starting_tree": source_tree or BASE_TREE,
        },
        "authority_split": {
            "duckdb": "transactional authority",
            "quack": "exclusive state-owner transport",
            "ducklake": "non-authoritative replayable projection",
        },
        "control_artifact_identities": artifact_identities,
        "required_artifact_identities": required_artifact_identities,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="check the exact checkout, initialized gitlinks, and materialized DuckDB file",
    )
    parser.add_argument(
        "--check-source",
        action="store_true",
        help="check the exact launch branch, starting tree, and initialized gitlinks",
    )
    parser.add_argument(
        "--require-database",
        action="store_true",
        help="require the configured DuckDB authority file to exist and be nonempty",
    )
    arguments = parser.parse_args()
    report = validate_program(
        check_source=arguments.check_all or arguments.check_source,
        require_database=arguments.check_all or arguments.require_database,
    )
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
