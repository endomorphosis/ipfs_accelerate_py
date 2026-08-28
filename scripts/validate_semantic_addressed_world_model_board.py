#!/usr/bin/env python3
"""Deterministic, fail-closed validator for the SAWM R2 control program.

The Markdown documents are immutable operator inputs, never task-completion
authority.  This validator checks their closed structure and the scheduler
binding only; accepted task state remains in the datasets-authoritative
DuckDB store reached through the current Quack owner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
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
NATIVE_AUTHORIZATION_PATH = (
    "config/semantic_addressed_world_model_native_dependency.authorization.json"
)

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
        "prior_materialization_migration.json",
    )
)

CONTROL_RELATIVE_PATHS = (
    ".gitignore",
    "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md",
    "docs/architecture/semantic_addressed_world_model.objectives.md",
    "docs/architecture/semantic_addressed_world_model.todo.md",
    *(path.relative_to(REPO_ROOT).as_posix() for path in INVENTORY_PATHS),
    "config/semantic_addressed_world_model_dependencies.seal.json",
    "config/semantic_addressed_world_model_native_dependency.authorization.json",
    "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
    "scripts/validate_semantic_addressed_world_model_dependencies.py",
    "scripts/validate_semantic_addressed_world_model_board.py",
    "scripts/materialize_semantic_addressed_world_model_program.py",
    "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
    "ipfs_accelerate_py/agent_implementation_route.py",
    "ipfs_accelerate_py/agent_supervisor/merge/database_coordination.py",
    "ipfs_accelerate_py/agent_supervisor/merge/merge_resolver.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_extension_projection.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_live_capsule.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/multi_supervisor_runner.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/provider_command_binding.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/board_control_plane.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/quack_owner_mutation.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/core.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/legacy_landed_attestation.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_loop.py",
    "ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_runtime.py",
    "ipfs_accelerate_py/agent_supervisor/validation/project_dependency_preflight.py",
    "test/api/semantic_world/test_semantic_addressed_world_model_board.py",
    "test/api/semantic_world/test_semantic_addressed_world_model_quack_protocol.py",
    "test/api/test_agent_supervisor_configured_board_extension_projection.py",
    "test/api/test_agent_supervisor_configured_board_live_capsule.py",
    "test/api/test_agent_supervisor_configured_board_scheduler.py",
    "test/api/test_agent_supervisor_database_coordination.py",
    "test/api/test_agent_supervisor_database_implementation_daemon.py",
    "test/api/test_agent_supervisor_database_portal_bridge.py",
    "test/api/test_agent_supervisor_native_dependency_pin.py",
    "test/api/test_agent_supervisor_project_dependency_preflight.py",
    "test/api/test_agent_supervisor_provider_command_binding.py",
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


def _canonical_identity(value: Mapping[str, Any], *, identity_field: str) -> str:
    body = dict(value)
    body.pop(identity_field, None)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _m5_migration_errors(
    config: Mapping[str, Any],
    seal: Mapping[str, Any],
    migration: Mapping[str, Any],
) -> list[str]:
    """Verify the frozen M4 authority and the one bounded M5 recovery CAS."""

    errors: list[str] = []
    prior_store = (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m4/control.duckdb"
    )
    target_store = (
        "data/agent_supervisor/semantic_addressed_world_model/"
        "run-r2-m5/control.duckdb"
    )
    m4_control_sha256 = (
        "d0c531d2ea30c512beb3587152527c4f605b5a5bf89e6311bf7aaa1974bff670"
    )
    m4_event_prefix = (
        "77b1a6b834658038c0f0cc44870a28f9fb480750e34dda408ab5718e83ee8911"
    )
    m4_projection = (
        "baguqeera65d24eqqbusuk6vznpmhm4nr65fas3i75dytebq5bev72bs6jaha"
    )
    m4_source_binding = (
        "sha256:72ae538afc063f98a4e7b0a799a619d6a8c499ceb51c9ceb4bffe95de5f9d323"
    )
    sawm_001_cid = (
        "sha256:76bcefe7428550da2bcf3e582b87b2106e0e393a0f1a84515518ebe3f6f16e76"
    )
    target_projection = (
        "baguqeerafx22x24mx7qrjjkfmyfdamtjkjd3ikrmhfqrp2gesqhe33l5467q"
    )
    expected_requeue = {
        "schema": "sawm/nonterminal-task-requeue-authorization@1",
        "authorized": True,
        "authority": "operator_source_migration",
        "task_alias": "SAWM-001",
        "task_cid": sawm_001_cid,
        "from_status": "in_progress",
        "from_revision": 2,
        "to_status": "todo",
        "to_revision": 3,
        "reason": (
            "bounded_preprovider_capsule_loader_and_attempt_settlement_recovery"
        ),
        "provider_invocation_count": 0,
        "effect_claim_count": 0,
        "accepted_definition_changes": 0,
        "accepted_completion_changes": 0,
        "expected_event_watermark": 119,
        "worker_self_approval": False,
    }

    expected_inventory = {
        "schema": "sawm/prior-materialization-migration-inventory@3",
        "migration_revision": "SAWM-R2-M5",
        "migration_kind": (
            "bounded_preprovider_capsule_loader_and_attempt_settlement_recovery"
        ),
        "supersession_reason": (
            "source_authority_revision_and_preprovider_task_requeue"
        ),
        "prior_store_id": prior_store,
        "target_store_id": target_store,
        "prior_plan_revision": 5,
        "target_plan_revision": 6,
        "target_generation": 7,
        "prior_event_watermark": 116,
        "prior_event_prefix_sha256": m4_event_prefix,
        "prior_projection_cid": m4_projection,
        "prior_control_store_sha256": m4_control_sha256,
        "prior_source_binding_cid": m4_source_binding,
        "prior_source_head": "1322089b8317c4ccb4678e0f4a2469b827a2dbec",
        "prior_source_tree": "38ed294b53fa9dde76b1ba92c57a9b931e49978c",
        "prior_materialization_receipt_cid": (
            "sha256:b24b3a2a0d325540761aaee03ea79bc88836b58577dab8583fec43f0de4c2a21"
        ),
        "prior_authority_preserved": True,
    }
    if any(migration.get(key) != value for key, value in expected_inventory.items()):
        errors.append("M5 inventory does not bind the exact frozen M4 authority")

    program = config.get("database_program")
    prior = config.get("prior_materialization")
    if not isinstance(program, Mapping) or (
        program.get("store_id") != target_store
        or program.get("store_generation") != "7"
    ):
        errors.append("M5 scheduler does not bind run-r2-m5 generation 7")
    if not isinstance(prior, Mapping):
        errors.append("M5 scheduler prior-materialization binding is absent")
        prior = {}
    expected_prior = {
        "migration_revision": "SAWM-R2-M5",
        "reason": "source_authority_revision_and_preprovider_task_requeue",
        "store_id": prior_store,
        "source_binding_cid": m4_source_binding,
        "source_head": "1322089b8317c4ccb4678e0f4a2469b827a2dbec",
        "source_tree": "38ed294b53fa9dde76b1ba92c57a9b931e49978c",
        "projection_cid": m4_projection,
        "control_store_sha256": m4_control_sha256,
        "event_watermark": 116,
        "event_prefix_sha256": m4_event_prefix,
        "migration_history_count": 4,
        "prior_plan_revision": 5,
        "target_plan_revision": 6,
        "migration_event_watermark": 118,
        "target_event_watermark": 119,
        "target_projection_cid": target_projection,
        "preserve_append_only": True,
    }
    if any(prior.get(key) != value for key, value in expected_prior.items()):
        errors.append("M5 scheduler prior-materialization fields are not exact")
    if dict(prior.get("nonterminal_task_requeue") or {}) != expected_requeue:
        errors.append("M5 scheduler does not authorize exactly one SAWM-001 recovery CAS")

    sealed = seal.get("source_migration")
    if not isinstance(sealed, Mapping):
        errors.append("M5 dependency-seal source migration is absent")
        sealed = {}
    expected_sealed = {
        "migration_revision": "SAWM-R2-M5",
        "migration_kind": (
            "bounded_preprovider_capsule_loader_and_attempt_settlement_recovery"
        ),
        "supersession_reason": (
            "source_authority_revision_and_preprovider_task_requeue"
        ),
        "mode": "append_only_source_authority_revision",
        "prior_store_id": prior_store,
        "target_store_id": target_store,
        "prior_plan_revision": 5,
        "target_plan_revision": 6,
        "prior_event_watermark": 116,
        "prior_event_prefix_sha256": m4_event_prefix,
        "prior_projection_cid": m4_projection,
        "prior_control_store_sha256": m4_control_sha256,
        "prior_source_binding_cid": m4_source_binding,
        "prior_source_head": "1322089b8317c4ccb4678e0f4a2469b827a2dbec",
        "prior_source_tree": "38ed294b53fa9dde76b1ba92c57a9b931e49978c",
        "prior_migration_count": 4,
        "prior_migration_receipt_cid": (
            "sha256:b24b3a2a0d325540761aaee03ea79bc88836b58577dab8583fec43f0de4c2a21"
        ),
        "migration_event_watermark": 118,
        "target_event_watermark": 119,
        "target_projection_cid": target_projection,
        "accepted_definition_rewrite_allowed": False,
        "accepted_completion_replay_allowed": False,
        "prior_authority_preserved": True,
    }
    if any(sealed.get(key) != value for key, value in expected_sealed.items()):
        errors.append("M5 dependency seal does not preserve the exact M4 authority")
    if dict(sealed.get("nonterminal_task_requeue") or {}) != expected_requeue:
        errors.append("M5 dependency seal recovery authorization is not exact")

    history = migration.get("migration_history")
    if (
        not isinstance(history, list)
        or len(history) != 4
        or any(not isinstance(entry, Mapping) for entry in history)
    ):
        errors.append("M1-through-M4 migration history is not a four-entry sequence")
        history = []
    if history:
        if [entry.get("migration_revision") for entry in history] != [
            "SAWM-R2-M1",
            "SAWM-R2-M2",
            "SAWM-R2-M3",
            "SAWM-R2-M4",
        ]:
            errors.append("M1-through-M4 migration revisions are not contiguous")
        for index, entry in enumerate(history[:3]):
            if (
                entry.get("schema") != "sawm/source-migration-history-entry@1"
                or not isinstance(entry.get("prior_event_watermark"), int)
                or entry.get("target_event_watermark")
                != entry.get("prior_event_watermark") + 2
            ):
                errors.append(
                    f"M{index + 1} history must use @1 and add exactly two events"
                )
        for previous, current in zip(history, history[1:], strict=False):
            if any(
                current.get(prior_key) != previous.get(target_key)
                for prior_key, target_key in (
                    ("prior_store_id", "target_store_id"),
                    ("prior_control_store_sha256", "target_control_store_sha256"),
                    ("prior_event_watermark", "target_event_watermark"),
                    ("prior_event_prefix_sha256", "target_event_prefix_sha256"),
                    ("prior_source_binding_cid", "current_source_binding_cid"),
                    ("prior_source_head", "current_source_head"),
                    ("prior_source_tree", "current_source_tree"),
                )
            ):
                errors.append("migration history predecessor/target continuity is broken")
                break
        m4 = history[3]
        expected_m4 = {
            "schema": "sawm/source-migration-history-entry@2",
            "migration_revision": "SAWM-R2-M4",
            "prior_event_watermark": 113,
            "migration_event_watermark": 115,
            "target_event_watermark": 116,
            "migration_event_prefix_sha256": (
                "724c47eaf1c70d6ddcf61059f133c752f2787a619aba040e4f2334ecbd287731"
            ),
            "target_event_prefix_sha256": m4_event_prefix,
            "migration_projection_cid": (
                "baguqeeramvvb5ij2hs4eniw735kyvfv3t253vp3tn7atk6vmw2qrcfuqf3sq"
            ),
            "projection_cid": m4_projection,
            "target_control_store_sha256": m4_control_sha256,
            "target_store_id": prior_store,
            "current_source_binding_cid": m4_source_binding,
            "post_migration_event_id": (
                "baguqeera3qvtzkbfquvcwhk3weocoesvqlajklm736surlkl25jn2ljkllha"
            ),
            "post_migration_event_type": "intent.task_status_changed",
            "post_migration_task_cid": sawm_001_cid,
            "post_migration_task_revision": 2,
            "post_migration_task_status": "in_progress",
        }
        if any(m4.get(key) != value for key, value in expected_m4.items()):
            errors.append(
                "M4 @2 history does not distinguish watermark 115 migration "
                "from watermark 116 frozen target"
            )
        if (
            m4.get("migration_receipt_cid")
            != migration.get("prior_materialization_receipt_cid")
            or m4.get("migration_receipt_path")
            != migration.get("prior_materialization_receipt_path")
        ):
            errors.append("M4 history receipt does not bind the frozen predecessor")

    failure = migration.get("preprovider_task_failure")
    if not isinstance(failure, Mapping):
        errors.append("frozen M4 pre-provider task failure is absent")
        failure = {}
    expected_failure = {
        "schema": "sawm/pre-provider-task-failure@1",
        "authority_class": "operator_frozen_predecessor_observation",
        "authoritative_completion_evidence": False,
        "store_id": prior_store,
        "control_store_sha256": m4_control_sha256,
        "database_uuid": "c6b5c6a1-eaaa-4c09-b401-6ee7998602b4",
        "plan_revision": 5,
        "owner_generation": 6,
        "source_binding_cid": m4_source_binding,
        "source_head": "1322089b8317c4ccb4678e0f4a2469b827a2dbec",
        "source_tree": "38ed294b53fa9dde76b1ba92c57a9b931e49978c",
        "canonical_event_watermark": 116,
        "canonical_event_prefix_sha256": m4_event_prefix,
        "canonical_projection_cid": m4_projection,
        "canonical_claim_event_id": (
            "baguqeera3qvtzkbfquvcwhk3weocoesvqlajklm736surlkl25jn2ljkllha"
        ),
        "canonical_claim_event_type": "intent.task_status_changed",
        "task_alias": "SAWM-001",
        "task_cid": sawm_001_cid,
        "task_previous_status": "todo",
        "task_status": "in_progress",
        "task_revision": 2,
        "execution_store_id": (
            "data/agent_supervisor/semantic_addressed_world_model/"
            "run-r2-m4/control.execution.duckdb"
        ),
        "execution_store_sha256": (
            "84e010b517a791039a6db16c2860a3292f6fb123d9884521ea52dc31f4223989"
        ),
        "coordination_store_id": (
            "data/agent_supervisor/semantic_addressed_world_model/"
            "run-r2-m4/control.coordination.duckdb"
        ),
        "coordination_store_sha256": (
            "6e4211ed41f02e84e95877a20f94d56660f94d00e2bb80769d4fa4550032a3a2"
        ),
        "portal_attempt_path": (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m4/"
            "state/sawm_database_portal_attempts/72474c065ac95c1e878159f3"
        ),
        "portal_attempt_binding_path": (
            "data/agent_supervisor/semantic_addressed_world_model/run-r2-m4/"
            "state/sawm_database_portal_attempts/72474c065ac95c1e878159f3/"
            "database-attempt-binding.json"
        ),
        "portal_attempt_binding_id": (
            "sha256:7c9ef3b030749b39ee01b772a3b54b43a77035dbb16f7ea1ec440f4646009f83"
        ),
        "portal_attempt_binding_sha256": (
            "db00bde3459f4eb4973532110647baabe2b2ca335a6704b3dbd22f706ef0e92f"
        ),
        "portal_task_projection_sha256": (
            "5710f2c084a967201a9c0ae1fd19d4aaaf7cb5257799462fba2eb768ac31dd01"
        ),
        "portal_event_log_sha256": (
            "aca7c522a6868965405696bfcadcc88ca47236eebfca6ca50171e20c9c9a035e"
        ),
        "portal_event_manifest_sha256": (
            "48fd39f4e4fe4a4c0d744111589d21eb84cc524fbc8a8c3ae67e458bec0bc67f"
        ),
        "portal_event_snapshot_id": (
            "event-log-snapshot:sha256:"
            "806ba18bf8230033aadf344c166a193b85c8cf77fd4a21792e4e61cd8ad02171"
        ),
        "portal_event_tail_id": (
            "sha256:46e15c941d784c8e7e986b2668bd2a457a13777c3c08297ec8f936c31732590d"
        ),
        "portal_event_count": 13,
        "portal_event_first_sequence": 1,
        "portal_event_last_sequence": 13,
        "database_attempt_terminal_reason": (
            "external_protected_checkout_recovery_required"
        ),
        "attempt_consumed": False,
        "retry_deferred": True,
        "successor_migration_revision": "SAWM-R2-M5",
        "successor_plan_revision": 6,
        "successor_generation": 7,
        "successor_store_id": target_store,
    }
    if any(failure.get(key) != value for key, value in expected_failure.items()):
        errors.append("frozen M4 control/companion-store/portal evidence is not exact")
    if failure.get("canonical_authority_counts") != {
        "completion_receipts": 1,
        "effect_claims": 0,
        "merge_attempts": 0,
        "provider_calls": 0,
        "provider_invocations": 0,
        "provider_responses": 0,
        "task_assignments": 0,
        "task_attempts": 0,
        "task_claims": 0,
    }:
        errors.append("frozen M4 canonical authority counts are not exact")
    if failure.get("execution_authority_counts") != {
        "attempt_phases": 3,
        "database_task_attempts": 1,
        "effect_claims": 0,
        "provider_invocations": 0,
    } or failure.get("coordination_authority_counts") != {
        "fenced_leases": 1,
        "resource_claims": 0,
        "task_attempts": 1,
        "task_claims": 1,
        "task_completions": 0,
    }:
        errors.append("frozen M4 companion authority counts are not exact")
    for field in (
        "provider_call_allowed",
        "provider_dispatch_attempted",
        "provider_dispatched",
        "provider_invocation_recorded",
        "implementation_provider_invoked",
        "effect_claim_recorded",
        "implementation_commit_created",
        "merge_attempted",
        "task_completed",
    ):
        if failure.get(field) is not False:
            errors.append(f"frozen M4 evidence must retain {field}=false")
    if failure.get("merge_queue_request_count") != 0:
        errors.append("frozen M4 evidence must retain zero merge requests")

    historical_failure = migration.get("preworker_launch_failure")
    if not isinstance(historical_failure, Mapping) or (
        len(history) == 4
        and (
            historical_failure.get("store_id") != history[2].get("target_store_id")
            or historical_failure.get("source_head")
            != history[2].get("current_source_head")
            or historical_failure.get("source_tree")
            != history[2].get("current_source_tree")
        )
    ):
        errors.append("historical M3 pre-worker failure binding is not preserved")

    repair_paths = tuple(migration.get("bounded_control_plane_repair_paths") or ())
    required_repair_paths = {
        "ipfs_accelerate_py/agent_supervisor/merge/database_coordination.py",
        "test/api/test_agent_supervisor_database_coordination.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/database_portal_bridge.py",
        "test/api/test_agent_supervisor_database_portal_bridge.py",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_daemon.py",
        "test/api/test_agent_supervisor_database_implementation_daemon.py",
        "ipfs_accelerate_py/agent_supervisor/validation/project_dependency_preflight.py",
        "test/api/test_agent_supervisor_project_dependency_preflight.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/provider_command_binding.py",
        "test/api/test_agent_supervisor_provider_command_binding.py",
    }
    if (
        not repair_paths
        or len(repair_paths) != len(set(repair_paths))
        or any(path not in CONTROL_RELATIVE_PATHS for path in repair_paths)
        or not required_repair_paths.issubset(repair_paths)
    ):
        errors.append("M5 bounded repair sources/tests are not exact protected controls")
    return errors


def _stable_regular_bytes(path: Path, *, maximum: int) -> bytes:
    if not path.is_absolute():
        raise ValueError("path is not absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("source is unavailable") from exc
    if resolved != path:
        raise ValueError("source path is noncanonical or contains a symlink")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("source is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or not 0 < before.st_size <= maximum
        ):
            raise ValueError("source is not stable owner-held regular-file evidence")
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            block = os.pread(
                descriptor,
                min(1024 * 1024, before.st_size - offset),
                offset,
            )
            if not block:
                break
            chunks.append(block)
            offset += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ValueError("source changed after read") from exc

    def identity(item: os.stat_result) -> tuple[int, ...]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_mode,
            item.st_uid,
            item.st_nlink,
            item.st_size,
            item.st_mtime_ns,
            item.st_ctime_ns,
        )

    raw = b"".join(chunks)
    if (
        len(raw) != before.st_size
        or identity(before) != identity(after)
        or identity(after) != identity(current)
    ):
        raise ValueError("source changed while read")
    return raw


def _extension_pin_errors(
    name: str,
    configured: object,
    sealed: object,
) -> list[str]:
    errors: list[str] = []
    extra = {"service_external_access_limitation"} if name == "quack" else set()
    fields = {
        "path",
        "info_path",
        "version",
        "sha256",
        "size",
        "info_sha256",
        "info_size",
        "network_install_allowed",
        "unsigned_extension_allowed",
        *extra,
    }
    if type(configured) is not dict or type(sealed) is not dict:
        return [f"{name} extension pin must be an exact object"]
    if configured != sealed:
        errors.append(f"scheduler {name} pin differs from the dependency seal")
    pin = sealed
    if set(pin) != fields:
        errors.append(f"sealed {name} extension pin fields are noncanonical")
        return errors
    path_value = pin.get("path")
    info_path_value = pin.get("info_path")
    version = pin.get("version")
    payload_digest = pin.get("sha256")
    info_digest = pin.get("info_sha256")
    payload_size = pin.get("size")
    info_size = pin.get("info_size")
    if (
        type(path_value) is not str
        or type(info_path_value) is not str
        or type(version) is not str
        or re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z.+_-]{0,63}", version) is None
        or type(payload_digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", payload_digest) is None
        or type(info_digest) is not str
        or re.fullmatch(r"[0-9a-f]{64}", info_digest) is None
        or type(payload_size) is not int
        or not 0 < payload_size <= 64 * 1024 * 1024
        or type(info_size) is not int
        or not 0 < info_size <= 64 * 1024
        or pin.get("network_install_allowed") is not False
        or pin.get("unsigned_extension_allowed") is not False
        or (
            name == "quack"
            and pin.get("service_external_access_limitation")
            != "canonical_writer_sealed; "
            "pinned_extension_preloaded_only_in_locked_read_only_loopback_replica"
        )
    ):
        errors.append(f"sealed {name} extension pin values are noncanonical")
        return errors
    payload_path = Path(path_value)
    info_path = Path(info_path_value)
    if (
        payload_path.name != f"{name}.duckdb_extension"
        or info_path != payload_path.with_name(f"{payload_path.name}.info")
    ):
        errors.append(f"sealed {name} extension paths are not exact")
        return errors
    try:
        payload = _stable_regular_bytes(payload_path, maximum=64 * 1024 * 1024)
        info = _stable_regular_bytes(info_path, maximum=64 * 1024)
    except ValueError as exc:
        errors.append(f"sealed {name} extension source is invalid: {exc}")
        return errors
    if len(payload) != payload_size or hashlib.sha256(payload).hexdigest() != payload_digest:
        errors.append(f"sealed {name} extension payload differs from its pin")
    if len(info) != info_size or hashlib.sha256(info).hexdigest() != info_digest:
        errors.append(f"sealed {name} extension metadata differs from its pin")
    return errors


def _native_extension_identity(
    seal: Mapping[str, Any],
) -> tuple[str, str, list[str]]:
    """Derive the only extension engine/platform admitted by native DuckDB."""

    native = seal.get("configured_board_native_dependency")
    native_pin = native.get("pin") if type(native) is dict else None
    toolchain = seal.get("toolchain")
    if type(native_pin) is not dict or type(toolchain) is not dict:
        return "", "", ["native DuckDB/toolchain extension identity is absent"]

    native_platforms = {
        ("linux", "aarch64"): "linux_arm64",
        ("linux", "x86_64"): "linux_amd64",
        ("darwin", "arm64"): "osx_arm64",
        ("darwin", "x86_64"): "osx_amd64",
        ("win32", "AMD64"): "windows_amd64",
    }
    toolchain_platforms = {
        ("Linux", "aarch64"): "linux_arm64",
        ("Linux", "x86_64"): "linux_amd64",
        ("Darwin", "arm64"): "osx_arm64",
        ("Darwin", "x86_64"): "osx_amd64",
        ("Windows", "AMD64"): "windows_amd64",
    }
    native_platform = native_platforms.get(
        (
            str(native_pin.get("platform_name") or ""),
            str(native_pin.get("platform_machine") or ""),
        )
    )
    toolchain_platform = toolchain_platforms.get(
        (
            str(toolchain.get("operating_system") or ""),
            str(toolchain.get("machine") or ""),
        )
    )
    engine = str(native_pin.get("engine_version") or "")
    distribution = str(native_pin.get("distribution_version") or "")
    toolchain_version = str(toolchain.get("duckdb_distribution_version") or "")
    errors: list[str] = []
    if (
        not engine
        or engine != f"v{distribution}"
        or distribution != toolchain_version
    ):
        errors.append("native DuckDB engine identity differs from the sealed toolchain")
    if (
        native_platform is None
        or toolchain_platform is None
        or native_platform != toolchain_platform
    ):
        errors.append("native DuckDB platform identity differs from the sealed toolchain")
    return engine, native_platform or "", errors


def _configured_board_dependency_errors(
    root: Path,
    config: Mapping[str, Any],
    seal: Mapping[str, Any],
) -> list[str]:
    """Validate exact configured-board dependency and local extension custody."""

    errors: list[str] = []
    expected_paths = {
        "validator_path": "scripts/validate_semantic_addressed_world_model_board.py",
        "dependency_validator_path": (
            "scripts/validate_semantic_addressed_world_model_dependencies.py"
        ),
        "dependency_seal_path": (
            "config/semantic_addressed_world_model_dependencies.seal.json"
        ),
    }
    for field, expected in expected_paths.items():
        if config.get(field) != expected:
            errors.append(f"scheduler {field} does not bind {expected}")
    environment_policy = seal.get("environment_policy")
    fixed_environment = (
        environment_policy.get("fixed")
        if isinstance(environment_policy, Mapping)
        and isinstance(environment_policy.get("fixed"), Mapping)
        else {}
    )
    if (
        not isinstance(environment_policy, Mapping)
        or environment_policy.get("duckdb_extension_home")
        != "exact_private_read_only_projection"
        or fixed_environment.get(
            "IPFS_ACCELERATE_AGENT_BOARD_EXTENSION_INSTALL_POLICY"
        )
        != "disabled"
    ):
        errors.append(
            "configured-board extension projection/install environment is not sealed"
        )
    validation = seal.get("validation")
    if (
        type(validation) is not dict
        or set(validation)
        != {
            "validator",
            "expected_result",
            "current_tree_revalidation_required_before_launch",
            "network_required",
            "database_open_required_for_static_seal_validation",
            "seal_mismatch_disposition",
        }
        or validation.get("validator") != expected_paths["dependency_validator_path"]
        or validation.get("current_tree_revalidation_required_before_launch") is not True
        or validation.get("network_required") is not False
        or validation.get("database_open_required_for_static_seal_validation") is not False
        or validation.get("seal_mismatch_disposition") != "fail_closed"
    ):
        errors.append("dependency seal validation policy is not exact and fail-closed")
    if any(
        expected not in tuple(config.get("protected_paths") or ())
        for expected in expected_paths.values()
    ):
        errors.append("scheduler validator/seal bindings are not protected paths")

    quack_owner = config.get("quack_owner")
    if not isinstance(quack_owner, Mapping):
        errors.append("scheduler Quack owner is absent")
        quack_owner = {}
    expected_engine, expected_platform, identity_errors = (
        _native_extension_identity(seal)
    )
    errors.extend(identity_errors)
    quack_pin = seal.get("quack_extension_pin")
    httpfs_pin = seal.get("httpfs_extension_pin")
    errors.extend(
        _extension_pin_errors("quack", quack_owner.get("pinned_extension"), quack_pin)
    )
    errors.extend(
        _extension_pin_errors(
            "httpfs",
            quack_owner.get("pinned_httpfs_extension"),
            httpfs_pin,
        )
    )
    if type(quack_pin) is dict and type(httpfs_pin) is dict:
        quack_path = Path(str(quack_pin.get("path") or ""))
        httpfs_path = Path(str(httpfs_pin.get("path") or ""))
        if quack_path.parent != httpfs_path.parent:
            errors.append("Quack and httpfs pins do not share one exact engine/platform root")
        for name, extension_path in (
            ("quack", quack_path),
            ("httpfs", httpfs_path),
        ):
            if (
                extension_path.parent.name != expected_platform
                or extension_path.parent.parent.name != expected_engine
            ):
                errors.append(
                    f"{name} extension path engine/platform differs from "
                    "native DuckDB/toolchain"
                )

    projection = seal.get("configured_board_quack_projection")
    expected_projection_fields = {
        "schema",
        "source_path",
        "info_path",
        "pin",
        "load_policy",
        "network_install_allowed",
        "unsigned_extension_allowed",
    }
    if type(projection) is not dict or set(projection) != expected_projection_fields:
        errors.append("configured-board Quack projection fields are noncanonical")
    else:
        try:
            from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_extension_projection import (
                inspect_configured_board_extension_sources,
                parse_configured_board_extension_pin,
            )

            projection_pin = parse_configured_board_extension_pin(projection.get("pin"))
            observed_projection = inspect_configured_board_extension_sources(
                projection.get("source_path"),
                projection.get("info_path"),
                name=projection_pin.name,
                engine_version=projection_pin.engine_version,
                platform=projection_pin.platform,
            )
        except (ImportError, OSError, TypeError, ValueError) as exc:
            errors.append(
                "configured-board Quack projection is invalid: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            if (
                observed_projection != projection_pin
                or projection.get("schema")
                != "semantic-addressed-world-model/configured-board-quack-projection@1"
                or projection.get("load_policy") != "local_load_only"
                or projection.get("network_install_allowed") is not False
                or projection.get("unsigned_extension_allowed") is not False
                or type(quack_pin) is not dict
                or projection.get("source_path") != quack_pin.get("path")
                or projection.get("info_path") != quack_pin.get("info_path")
                or projection_pin.payload_sha256
                != "sha256:" + str(quack_pin.get("sha256") or "")
                or projection_pin.payload_size != quack_pin.get("size")
                or projection_pin.info_sha256
                != "sha256:" + str(quack_pin.get("info_sha256") or "")
                or projection_pin.info_size != quack_pin.get("info_size")
            ):
                errors.append("configured-board Quack projection differs from its sealed source")
            if (
                projection_pin.engine_version != expected_engine
                or projection_pin.platform != expected_platform
            ):
                errors.append(
                    "configured-board extension engine/platform differs from "
                    "native DuckDB/toolchain"
                )
            projection_source = Path(str(projection.get("source_path") or ""))
            if (
                projection_source.parent.name != projection_pin.platform
                or projection_source.parent.parent.name
                != projection_pin.engine_version
            ):
                errors.append("configured-board Quack projection engine/platform path mismatches")

    native = seal.get("configured_board_native_dependency")
    native_fields = {
        "schema",
        "source_path",
        "acceptance",
        "pin",
        "sealed_memfd_required",
        "ambient_site_import_allowed",
        "ambient_loader_environment_allowed",
    }
    if type(native) is not dict or set(native) != native_fields:
        errors.append("configured-board native dependency fields are noncanonical")
        return errors
    acceptance = native.get("acceptance")
    if type(acceptance) is not dict or set(acceptance) != {
        "schema",
        "path",
        "sha256",
        "size",
        "authorization_id",
    }:
        errors.append("native dependency authorization reference is noncanonical")
        return errors
    if (
        native.get("schema")
        != "semantic-addressed-world-model/configured-board-native-dependency@1"
        or type(native.get("source_path")) is not str
        or not Path(native["source_path"]).is_absolute()
        or native.get("sealed_memfd_required") is not True
        or native.get("ambient_site_import_allowed") is not False
        or native.get("ambient_loader_environment_allowed") is not False
        or acceptance.get("schema")
        != "semantic-addressed-world-model/native-dependency-authorization-reference@1"
        or acceptance.get("path") != NATIVE_AUTHORIZATION_PATH
    ):
        errors.append("configured-board native dependency policy is not exact")
        return errors
    try:
        authorization_path = (root / NATIVE_AUTHORIZATION_PATH).resolve(strict=True)
        if authorization_path != root / NATIVE_AUTHORIZATION_PATH:
            raise ValueError("authorization path is noncanonical")
        authorization_raw = _stable_regular_bytes(
            authorization_path,
            maximum=32 * 1024,
        )
        authorization = json.loads(
            authorization_raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
        )
        from ipfs_accelerate_py.agent_implementation_route import (
            inspect_agent_supervisor_native_dependency_source,
            parse_agent_supervisor_native_dependency_pin,
        )

        native_pin = parse_agent_supervisor_native_dependency_pin(native.get("pin"))
        observed_native_pin = inspect_agent_supervisor_native_dependency_source(
            native.get("source_path"),
            distribution_version=native_pin.distribution_version,
            engine_version=native_pin.engine_version,
        )
    except (
        ImportError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        errors.append(
            "configured-board native dependency is invalid: "
            f"{type(exc).__name__}: {exc}"
        )
        return errors
    authorization_fields = {
        "schema",
        "board_namespace",
        "plan_revision",
        "status",
        "scope",
        "dependency_id",
        "payload_sha256",
        "python_executable_sha256",
        "authority_basis",
        "inspection_is_authority",
        "authorization_may_claim_task_completion",
        "authorization_id",
    }
    if type(authorization) is not dict or set(authorization) != authorization_fields:
        errors.append("native dependency authorization fields are noncanonical")
        return errors
    authorization_id = _canonical_identity(
        authorization,
        identity_field="authorization_id",
    )
    if (
        observed_native_pin != native_pin
        or native.get("source_path") != str(Path(native.get("source_path")))
        or Path(native.get("source_path")).name != native_pin.extension_filename
        or acceptance.get("size") != len(authorization_raw)
        or acceptance.get("sha256")
        != "sha256:" + hashlib.sha256(authorization_raw).hexdigest()
        or acceptance.get("authorization_id") != authorization_id
        or authorization.get("authorization_id") != authorization_id
        or authorization.get("schema")
        != "semantic-addressed-world-model/native-dependency-launch-authorization@1"
        or authorization.get("board_namespace") != BOARD_NAMESPACE
        or authorization.get("plan_revision") != PLAN_REVISION
        or authorization.get("status") != "accepted"
        or authorization.get("scope") != "configured-board-live-control-plane"
        or authorization.get("authority_basis")
        != "operator-owned protected control inside the accepted immutable source capsule"
        or authorization.get("dependency_id") != native_pin.dependency_id
        or authorization.get("payload_sha256") != native_pin.payload_sha256
        or authorization.get("python_executable_sha256")
        != native_pin.python_executable_sha256
        or authorization.get("inspection_is_authority") is not False
        or authorization.get("authorization_may_claim_task_completion") is not False
    ):
        errors.append("native dependency authorization does not bind the exact sealed pin")
    toolchain = seal.get("toolchain") if isinstance(seal.get("toolchain"), Mapping) else {}
    if (
        native_pin.distribution_version != toolchain.get("duckdb_distribution_version")
        or native_pin.engine_version
        != f"v{toolchain.get('duckdb_distribution_version', '')}"
    ):
        errors.append("native dependency version differs from the sealed DuckDB toolchain")
    return errors


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
        migration = _load_json(
            root
            / "docs/architecture/semantic_addressed_world_model_inventory/prior_materialization_migration.json"
        )
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
        or program.get("store_generation") != "7"
        or program.get("store_id") != migration.get("target_store_id")
    ):
        config_errors.append("DuckDB + Quack authority binding mismatch")
    prior = config.get("prior_materialization") if isinstance(config.get("prior_materialization"), Mapping) else {}
    if (
        prior.get("program_definition_cid") != migration.get("prior_program_definition_cid")
        or prior.get("plan_root_cid") != migration.get("prior_plan_root_cid")
        or set(migration.get("prior_task_cids") or {}) != set(TASK_IDS)
        or set(migration.get("prior_goal_cids") or {}) != set(GOAL_IDS)
        or migration.get("prior_task_count") != len(TASK_IDS)
        or migration.get("prior_goal_count") != len(GOAL_IDS)
        or migration.get("definition_source_binding_cid")
        != "sha256:cf4d9fa1ba595286866f5406e61b2ac71e4ed3730af70a2b07f88d0c16905e5e"
    ):
        config_errors.append("append-only prior-SAWM migration binding mismatch")
    config_errors.extend(_m5_migration_errors(config, seal, migration))
    config_errors.extend(_configured_board_dependency_errors(root, config, seal))
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
    live_policy = config.get("configured_board_live_capsule")
    if not isinstance(live_policy, Mapping):
        config_errors.append("configured-board live capsule policy is absent")
    else:
        try:
            from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_live_capsule import (
                parse_configured_board_live_capsule_policy,
            )

            live_paths = parse_configured_board_live_capsule_policy(live_policy)
        except (ImportError, ValueError) as exc:
            config_errors.append(
                "configured-board live capsule policy is invalid: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            if frozenset(live_paths) != frozenset(CONTROL_RELATIVE_PATHS):
                config_errors.append(
                    "configured-board live capsule paths differ from operator controls"
                )
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
