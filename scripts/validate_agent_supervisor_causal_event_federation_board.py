#!/usr/bin/env python3
"""Fail-closed validator for the sealed CASF goal heap, task DAG, and scheduler."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

SCHEMA = "ipfs_accelerate_py.agent_supervisor.causal-event-federation-board-validation@1"
PROGRAM_IDENTIFIER = "agent-supervisor-causal-event-federation-v1"
BOARD_NAMESPACE = PROGRAM_IDENTIFIER
PLAN_REVISION = "CASF-PLAN-R1"
BASE_REVISION = "84a056e41e48a81d4484be43840196578d6c87da"
BASE_TREE = "40f0771e77d394ac91d92cc1edb02f7860f6131b"
BRANCH = "codex/causal-event-supervisor-federation-v1"
ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/architecture/AGENT_SUPERVISOR_CAUSAL_EVENT_FEDERATION_PLAN.md"
OBJECTIVES = ROOT / "docs/architecture/agent_supervisor_causal_event_federation.objectives.md"
BOARD = ROOT / "docs/architecture/agent_supervisor_causal_event_federation.todo.md"
CONFIG = ROOT / "config/agent_supervisor_causal_event_federation_scheduler.json"
INVENTORY = ROOT / "docs/architecture/causal_event_federation_inventory"
TASK_RE = re.compile(r"^## (CASF-(\d{3})) (.+)$", re.MULTILINE)
GOAL_RE = re.compile(r"^## (CASF-G\d{3}) (.+)$", re.MULTILINE)
META_RE = re.compile(r"^- ([^:\n]+):(?: (.*))?$", re.MULTILINE)

EXPECTED_TITLES = (
    "Seal current authority and prerequisite baseline",
    "Inventory DuckDB, Quack, DuckLake, runner, event, and causal surfaces",
    "Define federation, supervisor, agent, shard, and budget contracts",
    "Define event, outbox, subscription, cursor, and dead-letter contracts",
    "Define causal node, edge, abstraction, intervention, and frontier contracts",
    "Extend normalized control-plane schema and migrations",
    "Implement federation and supervisor registries",
    "Implement logical subagent registry and bounded execution pools",
    "Implement authenticated external-agent trigger gateway",
    "Implement transactionally atomic domain event and outbox writes",
    "Implement state-owner event wait and notification path",
    "Implement bounded subscriptions and consumer cursors",
    "Implement event coalescing, retry, backpressure, and dead letters",
    "Implement multilevel causal graph store",
    "Integrate exact causal evidence and nomination-only retrieval evidence",
    "Implement causal abstraction maps and intervention consistency checks",
    "Implement causal frontier compilation",
    "Implement federation world snapshots",
    "Integrate AST, symbols, semantic roots, and capsule projections",
    "Integrate proof, test, cache, and seal projections",
    "Integrate knowledge graph, vector, and BM25 index projections",
    "Implement event-driven supervisor wake and cursor advancement",
    "Implement duplicate-work and task-subsumption detection",
    "Implement conflict-free parallel frontier",
    "Implement hierarchical resource and token budgets",
    "Implement supervisor sharding and specialization",
    "Implement work stealing",
    "Implement shard rebalancing and fencing",
    "Integrate worktrees, merge queue, and merge train",
    "Implement supervisor and subagent crash recovery",
    "Implement federation-level fixed-point detection",
    "Implement real DuckLake history projection",
    "Implement DuckLake recovery, security, and projection receipts",
    "Implement architecture and event drift monitoring",
    "Add federation control service",
    "Add CLI and MCP adapters",
    "Add TLA+/state-machine specifications and model checks",
    "Build adversarial and chaos suites",
    "Build event-driven idle benchmark",
    "Build twelve-supervisor parallel benchmark",
    "Build 256-agent bounded-load benchmark",
    "Build cross-supervisor token-efficiency benchmark",
    "Implement promotion, rollback, and quarantine gates",
    "Produce current-tree qualification and residual-gap report",
)
EXPECTED_TASK_IDS = tuple(f"CASF-{index:03d}" for index in range(44))
EXPECTED_INVENTORY_NO_CHANGE_TASK_IDS = frozenset(("CASF-000", "CASF-001"))
if len(EXPECTED_TASK_IDS) != len(EXPECTED_TITLES):
    raise RuntimeError("sealed CASF task identities and titles differ in length")
EXPECTED_TASK_TITLES = {
    task_id: EXPECTED_TITLES[index] for index, task_id in enumerate(EXPECTED_TASK_IDS)
}
_DEPENDENCY_NUMBERS = (
    (),
    (0,),
    (0, 1),
    (2,),
    (2,),
    (2, 3, 4),
    (5,),
    (5, 6),
    (5, 6),
    (3, 5, 6),
    (9,),
    (3, 9, 10),
    (3, 9, 11),
    (4, 5, 9),
    (13,),
    (4, 13, 14),
    (13, 14, 15),
    (5, 6, 13, 16),
    (5, 13, 17),
    (5, 13, 14, 17),
    (5, 13, 14, 17),
    (10, 11, 12, 16, 17, 18, 19, 20),
    (5, 13, 16, 21),
    (5, 16, 22),
    (5, 6, 7, 23),
    (6, 7, 16, 23, 24),
    (7, 23, 24, 25),
    (9, 10, 23, 24, 25, 26),
    (9, 19, 23, 24, 25, 27),
    (9, 10, 11, 12, 21, 24, 27, 28),
    (9, 21, 23, 24, 28, 29),
    (9, 12, 30),
    (31,),
    (12, 13, 16, 21, 29, 31, 32),
    (6, 8, 11, 16, 21, 24, 27, 30, 32),
    (8, 34),
    (3, 4, 9, 10, 15, 27, 29, 30),
    (8, 12, 15, 21, 27, 29, 32, 34, 36),
    (10, 12, 21, 29, 33, 37),
    (23, 24, 25, 27, 28, 29, 37),
    (7, 24, 25, 26, 27, 29, 37),
    (17, 18, 19, 20, 21, 22, 37),
    (30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41),
    (31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42),
)
if len(EXPECTED_TASK_IDS) != len(_DEPENDENCY_NUMBERS):
    raise RuntimeError("sealed CASF task identities and dependency rows differ in length")
EXPECTED_DEPENDENCIES = {
    task_id: [f"CASF-{dependency:03d}" for dependency in _DEPENDENCY_NUMBERS[index]]
    for index, task_id in enumerate(EXPECTED_TASK_IDS)
}
EXPECTED_GOAL_IDS = (
    "CASF-G000",
    "CASF-G010",
    "CASF-G011",
    "CASF-G012",
    "CASF-G013",
    "CASF-G020",
    "CASF-G021",
    "CASF-G022",
    "CASF-G023",
    "CASF-G030",
    "CASF-G031",
    "CASF-G032",
    "CASF-G033",
    "CASF-G040",
    "CASF-G041",
    "CASF-G042",
    "CASF-G043",
)
EXPECTED_PARENTS = {
    "CASF-G000": "",
    "CASF-G010": "CASF-G000",
    "CASF-G011": "CASF-G010",
    "CASF-G012": "CASF-G010",
    "CASF-G013": "CASF-G010",
    "CASF-G020": "CASF-G000",
    "CASF-G021": "CASF-G020",
    "CASF-G022": "CASF-G020",
    "CASF-G023": "CASF-G020",
    "CASF-G030": "CASF-G000",
    "CASF-G031": "CASF-G030",
    "CASF-G032": "CASF-G030",
    "CASF-G033": "CASF-G030",
    "CASF-G040": "CASF-G000",
    "CASF-G041": "CASF-G040",
    "CASF-G042": "CASF-G040",
    "CASF-G043": "CASF-G040",
}
EXPECTED_GOAL_HIERARCHY = {
    "CASF-G000": ["CASF-G010", "CASF-G020", "CASF-G030", "CASF-G040"],
    "CASF-G010": ["CASF-G011", "CASF-G012", "CASF-G013"],
    "CASF-G020": ["CASF-G021", "CASF-G022", "CASF-G023"],
    "CASF-G030": ["CASF-G031", "CASF-G032", "CASF-G033"],
    "CASF-G040": ["CASF-G041", "CASF-G042", "CASF-G043"],
}


def _ids(numbers: Sequence[int]) -> list[str]:
    return [f"CASF-{number:03d}" for number in numbers]


EXPECTED_TASK_GROUPS = {
    "CASF-G011": _ids((0, 1)),
    "CASF-G012": _ids((2, 4, 5, 6, 7, 8)),
    "CASF-G013": _ids((3, 9, 10, 11, 12)),
    "CASF-G021": _ids((13, 14, 15, 16, 17)),
    "CASF-G022": _ids((18, 19, 20)),
    "CASF-G023": _ids((21,)),
    "CASF-G031": _ids((22, 23, 24)),
    "CASF-G032": _ids((25, 26, 27)),
    "CASF-G033": _ids((28, 29, 30)),
    "CASF-G041": _ids((31, 32, 33)),
    "CASF-G042": _ids((34, 35, 36, 37)),
    "CASF-G043": _ids((38, 39, 40, 41, 42, 43)),
}
_WAVE_NUMBERS = (
    (0,),
    (1,),
    (2,),
    (3, 4),
    (5,),
    (6,),
    (7, 8, 9),
    (10, 13),
    (11, 14),
    (12, 15),
    (16,),
    (17,),
    (18, 19, 20),
    (21,),
    (22,),
    (23,),
    (24,),
    (25,),
    (26,),
    (27,),
    (28,),
    (29,),
    (30,),
    (31, 36),
    (32,),
    (33, 34),
    (35, 37),
    (38, 39, 40, 41),
    (42,),
    (43,),
)
EXPECTED_WAVES = [
    {"id": f"W{index}", "task_ids": _ids(numbers)} for index, numbers in enumerate(_WAVE_NUMBERS)
]
EXPECTED_DEPENDENCY_COUNT = 191
EXPECTED_DEPENDENCY_ROOT = "sha256:29be8bb6fbc6f37352ee7a312b2ecf87c15d575624ebcfd7bbc971968d688ecb"
EXPECTED_COMPLETED: list[str] = []
EXPECTED_READY = ["CASF-000"]
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
    "Database migrations",
    "Event effects",
    "Causal effects",
    "Authority class",
    "Risk class",
    "Read scope",
    "Write scope",
    "External effect scope",
    "Effect class",
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
    "Proof/model-checking requirements",
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
    "docs/architecture/AGENT_SUPERVISOR_CAUSAL_EVENT_FEDERATION_PLAN.md",
    "docs/architecture/agent_supervisor_causal_event_federation.objectives.md",
    "docs/architecture/agent_supervisor_causal_event_federation.todo.md",
    "config/agent_supervisor_causal_event_federation_scheduler.json",
    "scripts/validate_agent_supervisor_causal_event_federation_board.py",
    "test/api/causal_federation/test_board.py",
}
READ_ONLY_PREFIXES = ("ipfs_datasets_py", "ipfs_kit_py", "ipfs_accelerate_py/mcplusplus")
CLOSED_RISKS = {"low", "high", "critical"}
CLOSED_AUTHORITIES = {
    "evidence-only; no operational authority",
    "operational coordination only through canonical typed Quack state-owner boundary",
    "non-authoritative projection/observation/benchmark/qualification evidence",
}


def _identity(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _read_text(path: Path, errors: list[str]) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(f"missing/unreadable {path}: {exc}")
        return ""


def _read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid JSON {path}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"JSON root must be an object: {path}")
        return {}
    return value


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _metadata(body: str, record_id: str, errors: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for match in META_RE.finditer(body):
        key, value = match.group(1).strip(), (match.group(2) or "").strip()
        if key in result:
            errors.append(f"{record_id}: duplicate field {key!r}")
        result[key] = value
    return result


def _records(
    text: str, pattern: re.Pattern[str], errors: list[str]
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
        records[record_id] = _metadata(text[match.end() : end], record_id, errors)
        titles[record_id] = title
    return records, titles


def _safe_paths(value: str, task_id: str, field: str, errors: list[str]) -> list[str]:
    result = _csv(value)
    if not result:
        errors.append(f"{task_id}: empty {field}")
    for raw in result:
        path = PurePosixPath(raw)
        if (
            path.is_absolute()
            or ".." in path.parts
            or raw in {".", ".."}
            or "\\" in raw
            or bool(PureWindowsPath(raw).drive)
            or path.as_posix() != raw
            or any(ch in raw for ch in "*?[]\x00")
            or any(ord(ch) < 32 for ch in raw)
        ):
            errors.append(f"{task_id}: unsafe {field} path {raw!r}")
        for prefix in READ_ONLY_PREFIXES:
            prefix_parts = PurePosixPath(prefix).parts
            if path.parts[: len(prefix_parts)] == prefix_parts:
                errors.append(f"{task_id}: {field} writes read-only sibling path {raw!r}")
    if len(result) != len(set(result)):
        errors.append(f"{task_id}: duplicate {field} path")
    return result


def _resource(value: str, task_id: str, errors: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for segment in value.split(";"):
        if not segment.strip():
            continue
        if "=" not in segment:
            errors.append(f"{task_id}: malformed resource segment {segment!r}")
            continue
        key, raw = (part.strip() for part in segment.split("=", 1))
        if key in result:
            errors.append(f"{task_id}: duplicate resource field {key!r}")
        result[key] = raw
    missing, unknown = sorted(RESOURCE_FIELDS - set(result)), sorted(set(result) - RESOURCE_FIELDS)
    if missing:
        errors.append(f"{task_id}: resource vector missing {missing}")
    if unknown:
        errors.append(f"{task_id}: resource vector has unknown {unknown}")
    for name in (RESOURCE_FIELDS - RESOURCE_TEXT_FIELDS) & set(result):
        if not re.fullmatch(r"\d+", result[name]):
            errors.append(f"{task_id}: resource {name} must be a nonnegative integer")
    if result.get("network") != "deny":
        errors.append(f"{task_id}: bootstrap resource network must be deny")
    return result


def _layers(
    nodes: Sequence[str], dependencies: Mapping[str, Sequence[str]]
) -> tuple[list[str], list[list[str]]]:
    indegree = {node: 0 for node in nodes}
    children: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        for dependency in dependencies.get(node, ()):
            if dependency in indegree:
                indegree[node] += 1
                children[dependency].append(node)
    ready = sorted(node for node, value in indegree.items() if value == 0)
    ordered: list[str] = []
    waves: list[list[str]] = []
    while ready:
        layer, ready = ready, []
        waves.append(layer)
        for node in layer:
            ordered.append(node)
            for child in children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        ready.sort()
    return ordered, waves


def _closure(task_id: str, dependencies: Mapping[str, Sequence[str]]) -> set[str]:
    result: set[str] = set()
    pending = list(dependencies.get(task_id, ()))
    while pending:
        item = pending.pop()
        if item not in result:
            result.add(item)
            pending.extend(dependencies.get(item, ()))
    return result


def _overlap(left: str, right: str) -> bool:
    a, b = PurePosixPath(left).parts, PurePosixPath(right).parts
    return a[: len(b)] == b or b[: len(a)] == a


def _inventory(errors: list[str]) -> dict[str, str]:
    identities: dict[str, str] = {}
    expected = {
        "starting_tree.json": "casf_starting_tree_inventory",
        "authorities.json": "casf_authority_inventory",
        "capability_snapshot.json": "casf_capability_snapshot",
    }
    for name, artifact_type in expected.items():
        path = INVENTORY / name
        payload = _read_json(path, errors)
        if not path.is_file() or path.is_symlink():
            errors.append(f"{name}: inventory artifact must be a regular file")
        if payload.get("artifact_type") != artifact_type:
            errors.append(f"{name}: artifact_type mismatch")
        if payload.get("artifact_version") != 1:
            errors.append(f"{name}: artifact_version must be 1")
        if payload.get("program_id") != PROGRAM_IDENTIFIER:
            errors.append(f"{name}: program identity mismatch")
        if payload.get("root_objective_id") != "CASF-G000":
            errors.append(f"{name}: root objective mismatch")
        if name == "starting_tree.json":
            repository = payload.get("repository")
            if not isinstance(repository, Mapping):
                errors.append("starting_tree.json: repository must be an object")
                repository = {}
            if payload.get("inventory_tasks") != ["CASF-000", "CASF-001"]:
                errors.append("starting_tree.json: inventory task population mismatch")
            if (
                repository.get("starting_commit") != BASE_REVISION
                or repository.get("starting_tree") != BASE_TREE
                or repository.get("branch") != BRANCH
                or repository.get("baseline_kind") != "committed_git_tree"
                or repository.get("starting_worktree_was_clean") is not True
            ):
                errors.append("starting_tree.json: sealed repository baseline mismatch")
            if payload.get("inventory_refs") != {
                "human_readable": (
                    "docs/architecture/causal_event_federation_inventory/README.md"
                ),
                "authorities": (
                    "docs/architecture/causal_event_federation_inventory/authorities.json"
                ),
                "capabilities": (
                    "docs/architecture/causal_event_federation_inventory/"
                    "capability_snapshot.json"
                ),
            }:
                errors.append("starting_tree.json: inventory reference set mismatch")
        elif (
            payload.get("starting_commit") != BASE_REVISION
            or payload.get("starting_tree") != BASE_TREE
        ):
            errors.append(f"{name}: sealed repository baseline mismatch")
        if path.is_file() and not path.is_symlink():
            identities[path.relative_to(ROOT).as_posix()] = _identity(path.read_bytes())
    readme = INVENTORY / "README.md"
    text = _read_text(readme, errors)
    if not readme.is_file() or readme.is_symlink():
        errors.append("README.md: inventory artifact must be a regular file")
    for phrase in (
        "available_with_caveats",
        "typed blockers",
        "explicit nonclaims",
        BASE_REVISION,
        BASE_TREE,
    ):
        if phrase.lower() not in text.lower():
            errors.append(f"inventory README lacks {phrase!r}")
    if readme.is_file() and not readme.is_symlink():
        identities[readme.relative_to(ROOT).as_posix()] = _identity(readme.read_bytes())
    authorities = _read_json(INVENTORY / "authorities.json", errors)
    statuses = {"available", "available_with_caveats", "stale", "incompatible", "missing"}
    if set(authorities.get("status_vocabulary") or ()) != statuses:
        errors.append("authority inventory status vocabulary mismatch")
    rows = authorities.get("authorities")
    required = {
        "ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts",
        "ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations",
        "ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema",
        "ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_repository",
        "ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_transactions",
        "ipfs_accelerate_py.agent_supervisor.task_sources.quack_capabilities",
        "ipfs_accelerate_py.agent_supervisor.task_sources.quack_state_client",
        "ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server",
        "ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner",
        "ipfs_accelerate_py.agent_supervisor.semantic_state.world_snapshot_builder",
        "ipfs_accelerate_py.agent_supervisor.analysis.doctor_causal_localization",
        "ipfs_accelerate_py.agent_supervisor.integrations.ducklake_history_projection",
    }
    if not isinstance(rows, list):
        errors.append("authority inventory authorities must be a list")
        rows = []
    observed = {str(row.get("id") or "") for row in rows if isinstance(row, Mapping)}
    if not required <= observed:
        errors.append(f"authority inventory omits required IDs: {sorted(required - observed)}")
    for row in rows:
        if isinstance(row, Mapping) and row.get("status") not in statuses:
            errors.append(f"authority {row.get('id')}: invalid status")
    return dict(sorted(identities.items()))


def _production_parse(board_text: str, objectives_text: str, errors: list[str]) -> None:
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
            materialize_task_dependency_dag,
            parse_goal_heap,
        )
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
            parse_task_text,
        )

        tasks = parse_task_text(board_text, path=BOARD, task_header_prefix="## CASF-")
        goals = parse_goal_heap(objectives_text)
        graph = materialize_task_dependency_dag(tasks)
    except Exception as exc:
        errors.append(f"production parser failure: {type(exc).__name__}: {exc}")
        return
    if [task.task_id for task in tasks] != list(EXPECTED_TASK_IDS):
        errors.append("production task parser population/order mismatch")
    if [goal.goal_id for goal in goals] != list(EXPECTED_GOAL_IDS):
        errors.append("production objective parser population/order mismatch")
    if graph.invalid_task_cids:
        errors.append(f"production DAG rejected tasks: {graph.invalid_task_cids}")
    if len(graph.nodes) != 44 or len(graph.edges) != EXPECTED_DEPENDENCY_COUNT:
        errors.append("production DAG node/edge count mismatch")


def _git(arguments: Sequence[str]) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=ROOT, text=True, capture_output=True, check=False, timeout=60
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 124, "", f"{type(exc).__name__}: {exc}"
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def validate_program(
    *, check_source: bool = False, require_database: bool = False, inventory_only: bool = False
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    inventory_identities = _inventory(errors)
    if inventory_only:
        return {
            "schema": SCHEMA,
            "valid": not errors,
            "program_identifier": PROGRAM_IDENTIFIER,
            "inventory_only": True,
            "inventory_artifact_identities": inventory_identities,
            "errors": errors,
            "warnings": warnings,
        }
    plan_text = _read_text(PLAN, errors)
    objectives_text = _read_text(OBJECTIVES, errors)
    board_text = _read_text(BOARD, errors)
    config = _read_json(CONFIG, errors)
    tasks, titles = _records(board_text, TASK_RE, errors)
    goals, _ = _records(objectives_text, GOAL_RE, errors)
    if tuple(tasks) != EXPECTED_TASK_IDS:
        errors.append(f"task population/order mismatch: {list(tasks)}")
    if tuple(goals) != EXPECTED_GOAL_IDS:
        errors.append(f"goal population/order mismatch: {list(goals)}")
    for task_id, title in EXPECTED_TASK_TITLES.items():
        if titles.get(task_id) != title:
            errors.append(f"{task_id}: title mismatch")
    dependencies: dict[str, list[str]] = {}
    owned: dict[str, list[str]] = {}
    for task_id, metadata in tasks.items():
        missing = [field for field in REQUIRED_TASK_FIELDS if field not in metadata]
        if missing:
            errors.append(f"{task_id}: missing fields {missing}")
        for field in REQUIRED_TASK_FIELDS:
            if field not in {"Depends on"} and field in metadata and not metadata[field]:
                errors.append(f"{task_id}: empty required field {field!r}")
        if metadata.get("Stable task ID") != task_id:
            errors.append(f"{task_id}: Stable task ID mismatch")
        expected_status = "todo"
        if metadata.get("Status") != expected_status:
            errors.append(f"{task_id}: status must be {expected_status}")
        if (
            metadata.get("Completion") != "auto"
            or metadata.get("Is schedulable") != "true"
            or metadata.get("Review only") != "false"
        ):
            errors.append(f"{task_id}: task must be automatic and schedulable")
        no_change_completion = metadata.get("No-change completion")
        if task_id in EXPECTED_INVENTORY_NO_CHANGE_TASK_IDS:
            if no_change_completion != "allowed":
                errors.append(
                    f"{task_id}: sealed inventory must allow exact validated no-change completion"
                )
        elif task_id >= "CASF-033" and no_change_completion is not None:
            errors.append(
                f"{task_id}: unlanded task must remain outside no-change completion"
            )
        if metadata.get("Board namespace") != BOARD_NAMESPACE:
            errors.append(f"{task_id}: board namespace mismatch")
        if metadata.get("Owning repository") != "ipfs_accelerate_py":
            errors.append(f"{task_id}: owning repository mismatch")
        if metadata.get("Parent goal ID") != "CASF-G000":
            errors.append(f"{task_id}: parent goal mismatch")
        goal = metadata.get("Goal id", "")
        if metadata.get("Subgoal ID") != goal or goal not in EXPECTED_TASK_GROUPS:
            errors.append(f"{task_id}: leaf subgoal binding mismatch")
        if (
            metadata.get("Base revision") != BASE_REVISION
            or metadata.get("Base repository tree") != BASE_TREE
        ):
            errors.append(f"{task_id}: starting source binding mismatch")
        if metadata.get("Base plan revision") != PLAN_REVISION:
            errors.append(f"{task_id}: plan revision mismatch")
        parsed = _csv(metadata.get("Depends on", ""))
        dependencies[task_id] = parsed
        if parsed != EXPECTED_DEPENDENCIES.get(task_id):
            errors.append(f"{task_id}: dependency list mismatch")
        for dependency in parsed:
            if dependency not in tasks or dependency >= task_id:
                errors.append(
                    f"{task_id}: dependency must be an existing lower CASF task: {dependency}"
                )
        owned_paths = _safe_paths(metadata.get("Owned paths", ""), task_id, "Owned paths", errors)
        predicted = _safe_paths(
            metadata.get("Predicted files", ""), task_id, "Predicted files", errors
        )
        outputs = _safe_paths(metadata.get("Outputs", ""), task_id, "Outputs", errors)
        owned[task_id] = owned_paths
        if owned_paths != predicted or owned_paths != outputs:
            errors.append(f"{task_id}: Owned paths, Predicted files and Outputs differ")
        if set(owned_paths) & PROTECTED_PATHS:
            errors.append(f"{task_id}: worker owns protected planning/control path")
        _resource(metadata.get("Resource demand", ""), task_id, errors)
        if not re.fullmatch(
            r"input_tokens=\d+; output_tokens=\d+", metadata.get("Token budget", "")
        ):
            errors.append(f"{task_id}: token budget is not a closed numeric pair")
        if metadata.get("Risk class") not in CLOSED_RISKS:
            errors.append(f"{task_id}: risk class is not closed")
        if metadata.get("Authority class") not in CLOSED_AUTHORITIES:
            errors.append(f"{task_id}: authority class is not closed")
        migration = metadata.get("Database migrations", "")
        if not (
            migration.startswith("none")
            or migration.startswith("0002_causal_event_federation_core")
        ):
            errors.append(f"{task_id}: database migration declaration is unrecognized")
        for field in (
            "Event effects",
            "Causal effects",
            "Acceptance subset",
            "Conflict policy",
            "Rollback",
            "Proof/model-checking requirements",
        ):
            if not metadata.get(field):
                errors.append(f"{task_id}: missing required declaration {field}")
        lease = metadata.get("Lease and fencing", "").lower()
        if "lease" not in lease or "fenc" not in lease:
            errors.append(f"{task_id}: lease/fencing declaration incomplete")
        if "python3" not in metadata.get("Validation", ""):
            errors.append(f"{task_id}: validation must be a Python command")
        final_result_identity = metadata.get("Final result identity", "")
        if not final_result_identity.startswith("pending;"):
            errors.append(f"{task_id}: final result must remain pending")
        if metadata.get("Status") == "completed" and final_result_identity.startswith(
            "pending;"
        ):
            errors.append(
                f"{task_id}: pending final result identity cannot materialize as completed"
            )
    ordered, layers = _layers(list(tasks), dependencies)
    if len(ordered) != len(tasks):
        errors.append("task DAG contains a cycle")
    if [{"id": f"W{i}", "task_ids": wave} for i, wave in enumerate(layers)] != EXPECTED_WAVES:
        errors.append("computed task waves mismatch sealed waves")
    edges = sorted([dependency, task] for task, ds in dependencies.items() for dependency in ds)
    dependency_root = _identity(json.dumps(edges, sort_keys=True, separators=(",", ":")).encode())
    if len(edges) != EXPECTED_DEPENDENCY_COUNT or dependency_root != EXPECTED_DEPENDENCY_ROOT:
        errors.append("dependency count/root identity mismatch")
    completed = sorted(task for task, meta in tasks.items() if meta.get("Status") == "completed")
    blocked = sorted(task for task, meta in tasks.items() if meta.get("Status") == "blocked")
    ready = sorted(
        task
        for task, meta in tasks.items()
        if meta.get("Status") == "todo" and all(dep in completed for dep in dependencies[task])
    )
    if completed != EXPECTED_COMPLETED:
        errors.append(f"completed population mismatch: {completed}")
    if blocked:
        errors.append(f"blocked population must be empty: {blocked}")
    if ready != EXPECTED_READY:
        errors.append(f"ready population mismatch: {ready}")
    for left_index, left in enumerate(tasks):
        left_closure = _closure(left, dependencies)
        for right in list(tasks)[left_index + 1 :]:
            right_closure = _closure(right, dependencies)
            if left in right_closure or right in left_closure:
                continue
            for a in owned.get(left, ()):
                for b in owned.get(right, ()):
                    if _overlap(a, b):
                        errors.append(f"unordered owned-path collision: {left}:{a} <> {right}:{b}")
    goal_dependencies: dict[str, list[str]] = {}
    for goal_id, metadata in goals.items():
        missing = [field for field in REQUIRED_GOAL_FIELDS if field not in metadata]
        if missing:
            errors.append(f"{goal_id}: missing fields {missing}")
        for field in REQUIRED_GOAL_FIELDS:
            if field not in {"Parent", "Depends on"} and field in metadata and not metadata[field]:
                errors.append(f"{goal_id}: empty required field {field}")
        if metadata.get("Status") != "active":
            errors.append(f"{goal_id}: status must be active")
        if metadata.get("Parent", "") != EXPECTED_PARENTS.get(goal_id):
            errors.append(f"{goal_id}: parent mismatch")
        goal_dependencies[goal_id] = _csv(metadata.get("Depends on", ""))
        if any(dep not in goals or dep == goal_id for dep in goal_dependencies[goal_id]):
            errors.append(f"{goal_id}: invalid goal dependency")
    if len(_layers(list(goals), goal_dependencies)[0]) != len(goals):
        errors.append("goal DAG contains a cycle")
    for leaf, expected in EXPECTED_TASK_GROUPS.items():
        actual = [task for task, meta in tasks.items() if meta.get("Subgoal ID") == leaf]
        if actual != expected:
            errors.append(f"{leaf}: task membership mismatch")
    _production_parse(board_text, objectives_text, errors)
    required_plan_phrases = (
        PROGRAM_IDENTIFIER,
        "CausalAbstractionSupervisorFederation",
        "zero unauthorized",
        "exclusive Quack state owner",
        "wait_for_events",
        "12 supervisors",
        "256",
        "64 concurrent",
        "exactly-once network delivery",
        "lost-wakeup",
        "CASF-000..012",
        "CASF-031..043",
        "Final report",
        "DuckLake",
    )
    for phrase in required_plan_phrases:
        if phrase.lower() not in plan_text.lower():
            errors.append(f"plan lacks required phrase {phrase!r}")
    expected_paths = {
        "taskboard_path": "docs/architecture/agent_supervisor_causal_event_federation.todo.md",
        "objectives_path": "docs/architecture/agent_supervisor_causal_event_federation.objectives.md",
        "plan_path": "docs/architecture/AGENT_SUPERVISOR_CAUSAL_EVENT_FEDERATION_PLAN.md",
        "validator_path": "scripts/validate_agent_supervisor_causal_event_federation_board.py",
    }
    if (
        config.get("schema")
        != "ipfs_accelerate_py.agent_supervisor.causal-event-supervisor-federation.scheduler_config@1"
    ):
        errors.append("config schema mismatch")
    if (
        config.get("program_identifier") != PROGRAM_IDENTIFIER
        or config.get("board_namespace") != BOARD_NAMESPACE
    ):
        errors.append("config program/namespace mismatch")
    for field, expected in expected_paths.items():
        if config.get(field) != expected:
            errors.append(f"config {field} mismatch")
    if (
        config.get("task_prefix") != "CASF-"
        or config.get("goal_prefix") != "CASF-G"
        or config.get("accepted_plan_revision_alias") != PLAN_REVISION
    ):
        errors.append("config prefix/plan revision mismatch")
    if config.get("merge_target_branch") != BRANCH:
        errors.append("config branch mismatch")
    source = (
        config.get("source_binding") if isinstance(config.get("source_binding"), Mapping) else {}
    )
    if (
        source.get("accelerator_required_ancestor") != BASE_REVISION
        or source.get("accelerator_planning_tree") != BASE_TREE
    ):
        errors.append("config source binding mismatch")
    if (
        source.get("bootstrap_task_source") != "duckdb"
        or source.get("planning_revision_is_runtime_completion_evidence") is not False
    ):
        errors.append("config source authority mismatch")
    projection = config.get("initial_projection")
    expected_projection = {
        "task_count": 44,
        "task_dependency_count": EXPECTED_DEPENDENCY_COUNT,
        "task_dependency_root_cid": EXPECTED_DEPENDENCY_ROOT,
        "completed_task_ids": EXPECTED_COMPLETED,
        "ready_task_ids": EXPECTED_READY,
        "blocked_task_ids": [],
        "terminal_task_id": "CASF-043",
        "goal_count": 17,
        "root_goal_id": "CASF-G000",
    }
    if projection != expected_projection:
        errors.append("config initial projection mismatch")
    if (
        config.get("goal_hierarchy") != EXPECTED_GOAL_HIERARCHY
        or config.get("task_groups") != EXPECTED_TASK_GROUPS
        or config.get("waves") != EXPECTED_WAVES
    ):
        errors.append("config goal/task/wave projection mismatch")
    database = (
        config.get("database_program")
        if isinstance(config.get("database_program"), Mapping)
        else {}
    )
    if (
        database.get("authority_mode") != "quack"
        or database.get("task_source_kind") != "duckdb"
        or database.get("failover_policy") != "fail_closed"
        or database.get("explicit_legacy") is not False
    ):
        errors.append("database program must use fail-closed Quack-authoritative DuckDB")
    if (
        database.get("quack_endpoint") != "quack:127.0.0.1:41417"
        or database.get("endpoint_secret_handle") != "handle:casf-v1"
    ):
        errors.append("database endpoint/opaque handle mismatch")
    endpoint = (
        config.get("endpoint_allocation_policy")
        if isinstance(config.get("endpoint_allocation_policy"), Mapping)
        else {}
    )
    if (
        endpoint.get("host") != "127.0.0.1"
        or endpoint.get("port") != 41417
        or endpoint.get("mandatory_prelaunch_recheck") is not True
        or endpoint.get("collision_disposition")
        != "fail_closed_without_starting_or_killing_existing_listener"
        or endpoint.get("agent_supplied_endpoint_permitted") is not False
    ):
        errors.append("endpoint collision policy is not fail closed")
    control = (
        config.get("operational_control_plane")
        if isinstance(config.get("operational_control_plane"), Mapping)
        else {}
    )
    if (
        control.get("direct_multi_process_duckdb_file_open_permitted") is not False
        or control.get("automatic_file_fallback_permitted") is not False
        or control.get("arbitrary_sql_from_agents_permitted") is not False
    ):
        errors.append("operational state-owner boundary is unsafe")
    ducklake = (
        config.get("ducklake_projection_program")
        if isinstance(config.get("ducklake_projection_program"), Mapping)
        else {}
    )
    for field in (
        "authority",
        "scheduling_prerequisite",
        "lease_prerequisite",
        "policy_prerequisite",
        "acceptance_prerequisite",
        "completion_prerequisite",
        "may_grant_authority",
    ):
        if ducklake.get(field) is not False:
            errors.append(f"DuckLake {field} must be false")
    if (
        config.get("max_lanes") != 1
        or config.get("strict_task_sharding") is not True
        or config.get("idle_lane_work_stealing") != ""
    ):
        errors.append("bootstrap must use one strict lane without work stealing")
    lanes = config.get("lanes")
    if lanes != [
        {
            "index": 0,
            "name": "casf-bootstrap-lane-0",
            "strict_shard_remainder": 0,
            "initial_task_ids": ["CASF-000"],
            "initial_focus": "seal current authority and prerequisite baseline",
        }
    ]:
        errors.append("bootstrap lane declaration mismatch")
    provider = config.get("provider") if isinstance(config.get("provider"), Mapping) else {}
    if (
        provider.get("max_concurrency") != 1
        or provider.get("secrets_from_environment_only") is not True
        or provider.get("secrets_in_argv_prompts_logs_or_receipts") is not False
    ):
        errors.append("bootstrap provider capacity/security mismatch")
    wait = (
        config.get("event_wait_policy")
        if isinstance(config.get("event_wait_policy"), Mapping)
        else {}
    )
    if (
        wait.get("bootstrap_qualification")
        != "CASF-010_server_owned_typed_wait_qualified"
        or wait.get("server_owned_typed_wait_qualified") is not True
        or wait.get("federation_event_driven_execution_qualified") is not False
        or wait.get("federation_event_driven_execution_gate")
        != "unavailable_until_CASF-021"
    ):
        errors.append("event wait qualification projection mismatch")
    for field in (
        "compatibility_polling_for_qualified_mode",
        "busy_loop_permitted",
        "idle_full_board_scan_permitted",
        "idle_model_call_permitted",
        "idle_context_rebuild_permitted",
        "idle_unchanged_write_permitted",
        "event_driven_claim_permitted_at_bootstrap",
    ):
        if wait.get(field) is not False:
            errors.append(f"event wait policy {field} must be false")
    gate = (
        config.get("high_concurrency_gate")
        if isinstance(config.get("high_concurrency_gate"), Mapping)
        else {}
    )
    expected_gate_tasks = ["CASF-005", "CASF-009", "CASF-010", "CASF-016", "CASF-024", "CASF-029"]
    if (
        gate.get("enabled_at_bootstrap") is not False
        or gate.get("required_accepted_task_ids") != expected_gate_tasks
        or gate.get("target_profile")
        != {"supervisors": 12, "registered_logical_subagents": 256, "maximum_active_subagents": 64}
        or gate.get("missing_or_stale_telemetry_adds_capacity") is not False
    ):
        errors.append("high-concurrency gate mismatch")
    promotion_claims = (
        config.get("promotion_claims")
        if isinstance(config.get("promotion_claims"), Mapping)
        else {}
    )
    expected_promotion_claims = {
        "high_concurrency_qualified": False,
        "multi_supervisor_qualified": False,
        "parallel_execution_qualified": False,
        "causal_coordination_qualified": False,
        "production_ready": False,
        "ducklake_projection_promoted": False,
    }
    if promotion_claims != expected_promotion_claims:
        errors.append("premature federation promotion claim")
    authority = (
        config.get("authority_policy")
        if isinstance(config.get("authority_policy"), Mapping)
        else {}
    )
    true_fields = (
        "duckdb_transactional_authority",
        "quack_exclusive_state_owner_transport",
        "one_authoritative_store_per_mutable_semantic_fact",
    )
    false_fields = (
        "ducklake_projection_authority",
        "ducklake_projection_is_scheduling_prerequisite",
        "ducklake_projection_is_acceptance_prerequisite",
        "ducklake_projection_is_completion_authority",
        "markdown_is_completion_authority",
        "task_board_status_is_completion_evidence",
        "direct_multi_process_duckdb_file_open_permitted",
        "automatic_quack_to_file_fallback",
        "arbitrary_sql_from_agent_permitted",
        "model_created_authority_permitted",
        "model_created_policy_permission_permitted",
        "model_created_completion_permitted",
        "retrieval_similarity_is_causal_authority",
        "simulated_evidence_is_live_evidence",
        "cross_tenant_state_leakage_permitted",
    )
    if any(authority.get(field) is not True for field in true_fields) or any(
        authority.get(field) is not False for field in false_fields
    ):
        errors.append("authority policy mismatch")
    protected = set(config.get("protected_paths") or ())
    if not PROTECTED_PATHS <= protected:
        errors.append("config omits protected planning/control paths")
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
            load_configured_board,
        )

        loaded = load_configured_board(CONFIG, repo_root=ROOT)
        program = loaded.resolved_database_program()
        if (
            loaded.max_lanes != 1
            or program.authority_mode != "quack"
            or program.task_source_kind != "duckdb"
        ):
            errors.append("production configured-board loader changed sealed authority/capacity")
    except Exception as exc:
        errors.append(
            f"production configured-board loader rejected config: {type(exc).__name__}: {exc}"
        )
    if check_source:
        code, head, _ = _git(("rev-parse", "HEAD"))
        code2, tree, _ = _git(("rev-parse", f"{BASE_REVISION}^{{tree}}"))
        code3, branch, _ = _git(("branch", "--show-current"))
        code4, _, _ = _git(("merge-base", "--is-ancestor", BASE_REVISION, "HEAD"))
        if code or code2 or code3 or code4 or tree != BASE_TREE or branch != BRANCH:
            errors.append(
                "current Git source does not descend from the sealed branch/tree baseline"
            )
    database_checked = False
    if require_database:
        database_checked = True
        raw = str(database.get("store_id") or "")
        path = ROOT / raw
        if not raw or path.is_symlink() or not path.is_file():
            errors.append("required authoritative DuckDB store is absent or unsafe")
    return {
        "schema": SCHEMA,
        "valid": not errors,
        "program_identifier": PROGRAM_IDENTIFIER,
        "root_objective_id": "CASF-G000",
        "task_prefix": "CASF-",
        "plan_revision": PLAN_REVISION,
        "task_count": len(tasks),
        "goal_count": len(goals),
        "task_dependency_count": len(edges),
        "task_dependency_root_cid": dependency_root,
        "completed_task_ids": completed,
        "blocked_task_ids": blocked,
        "initial_ready_task_ids": ready,
        "terminal_task_id": "CASF-043",
        "task_waves": [{"id": f"W{i}", "task_ids": wave} for i, wave in enumerate(layers)],
        "inventory_artifact_identities": inventory_identities,
        "source_checks_performed": check_source,
        "database_presence_checked": database_checked,
        "quack_authority_required": True,
        "ducklake_authoritative": False,
        "high_concurrency_gate_open": False,
        "errors": errors,
        "warnings": warnings,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-source", action="store_true")
    parser.add_argument("--require-database", action="store_true")
    parser.add_argument("--check-all", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args(argv)
    report = validate_program(
        check_source=args.check_source or args.check_all,
        require_database=args.require_database or args.check_all,
        inventory_only=args.inventory_only,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
