"""Admit bounded native source-child tasks for one pinned inventory.

Each concrete unit is one source, config, split, jurisdiction and shard.
The unit admits five ordered tasks: normalize, index, geography, immutable
publication and verification. Identities are a pure function of that unit
and the bound inventory and active plan revisions.

WWL-017 keeps a required receipt for every admitted child. Existing blocker
records stay in place. The persisted watchdog task set gains the new ids and
its action, repair and incident budgets stay as they were.

This module does not download a corpus and does not grant completion.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .worldwide_legal_search_blockers import BLOCKER_KINDS, RECHECKABLE, BlockerBoard
from .worldwide_legal_search_completion import MAX_EXPANSION
from .worldwide_legal_search_runtime import atomic_json, identity
from .worldwide_legal_search_watchdog import read_owned_json

TASK_ID = "WWL-018"
JOIN_TASK_ID = "WWL-017"
PARENT_TASK_ID = "WWL-018"
RUN_ID = "worldwide-legal-search-v1"
GOAL_ID = "WWL-G020"
SUBGOAL_ID = "WWL-G020.S2"
ROOT_GOAL_ID = "WWL-G000"
DEPENDS_ON = ("WWL-016", "WWL-047", "WWL-081")

INVENTORY_SCHEMA = "worldwide-legal-search.source-inventory@1"
CHILD_SCHEMA = "worldwide-legal-search.source-child@1"
JOIN_SCHEMA = "worldwide-legal-search.source-expansion-join@1"
ADMISSION_SCHEMA = "worldwide-legal-search.source-expansion-admission@1"
CONTRACT_SCHEMA = "worldwide-legal-search.source-expansion@1"
WATCHDOG_STATE_SCHEMA = "wwl.watchdog.state@1"
READINESS_SCHEMA = "wwl.watchdog.ready@1"
JOURNAL_SCHEMA = "wwl.watchdog.restart-journal@1"

PHASES = (
    {"id": "normalize", "title": "normalize", "grain": "source"},
    {"id": "index", "title": "index", "grain": "config"},
    {"id": "geo", "title": "geography", "grain": "jurisdiction"},
    {"id": "publish", "title": "immutable publication", "grain": "shard"},
    {"id": "verify", "title": "verification", "grain": "split"},
)
PHASE_IDS = tuple(item["id"] for item in PHASES)
PHASE_BY_ID = {item["id"]: item for item in PHASES}
MAX_CHILDREN = MAX_EXPANSION
MAX_UNITS = MAX_CHILDREN // len(PHASES)
LEVELS = ("national", "regional", "local", "supranational", "unknown")
RUNNABLE_LEVELS = ("national", "regional", "local", "supranational")
ACCESS_KIND = {401: "credentials", 403: "permission", 408: "source-outage", 429: "source-outage", 500: "source-outage", 502: "source-outage", 503: "source-outage", 504: "source-outage"}
PRESERVED_STATE_KEYS = ("actions", "repairs", "incidents", "budgets", "progress_token", "policy_digest")
PRESERVED_JOURNAL_KEYS = ("actions", "guardian_restarts", "sealed_hold_markers", "loop_policy_digest")

_SHA = re.compile(r"^sha256:[0-9a-f]{64}$")
_ID = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,63}$")
_CHILD = re.compile(r"^WWL-SRC-[0-9a-f]{16}-(?:normalize|index|geo|publish|verify)-[0-9]{4}$")
_REASON = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._:/-]{0,160}$")
_RESET_KEYS = frozenset({"reset", "reset_budget", "reset_budgets", "reset_progress", "refund", "budgets_reset", "progress_reset"})
_SECRET_KEYS = frozenset({"secret", "token", "password", "api_key", "apikey", "authorization", "credential", "access_token", "private_key", "refresh_token"})
_INVENTORY_KEYS = frozenset({"schema", "namespace", "complete", "sources", "inventory_revision", "active_plan_revision"})
_SOURCE_KEYS = frozenset({"source_id", "resolved_revision", "eligible", "configs", "blocker", "access_error"})
_CONFIG_KEYS = frozenset({"config_id", "splits", "jurisdictions", "shards"})
_JURISDICTION_KEYS = frozenset({"jurisdiction_id", "level"})
_SHARD_KEYS = frozenset({"shard_id", "bytes"})
_BLOCKER_KEYS = frozenset({"kind", "reason"})
_ACCESS_KEYS = frozenset({"status", "reason"})
_SYNTHETIC = {"config_id": "unavailable", "split_id": "unavailable", "jurisdiction_id": "unknown", "level": "unknown", "shard_id": "0000", "bytes": 0}


class SourceExpansionError(ValueError):
    """Inventory cannot be admitted without dropping work or resetting budgets."""


def published_contract() -> dict[str, Any]:
    """Closed contract published at docs/worldwide/source-expansion.json."""
    return {
        "schema": CONTRACT_SCHEMA,
        "task_id": TASK_ID,
        "title": "Expand the inventory into bounded native source child tasks",
        "namespace": RUN_ID,
        "root_goal_id": ROOT_GOAL_ID,
        "goal_id": GOAL_ID,
        "subgoal_id": SUBGOAL_ID,
        "depends_on": list(DEPENDS_ON),
        "join_task_id": JOIN_TASK_ID,
        "parent_task_id": PARENT_TASK_ID,
        "inventory_schema": INVENTORY_SCHEMA,
        "child_schema": CHILD_SCHEMA,
        "join_schema": JOIN_SCHEMA,
        "admission_schema": ADMISSION_SCHEMA,
        "phases": [dict(item) for item in PHASES],
        "max_children": MAX_CHILDREN,
        "max_units": MAX_UNITS,
        "child_id_pattern": _CHILD.pattern,
        "levels": list(LEVELS),
        "blocker_kinds": list(BLOCKER_KINDS),
        "access_status_kinds": {str(status): kind for status, kind in sorted(ACCESS_KIND.items())},
        "watchdog_state_schema": WATCHDOG_STATE_SCHEMA,
        "readiness_schema": READINESS_SCHEMA,
        "journal_schema": JOURNAL_SCHEMA,
        "preserved_state_keys": list(PRESERVED_STATE_KEYS),
        "preserved_journal_keys": list(PRESERVED_JOURNAL_KEYS),
        "native_api": "ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source.DatabaseTaskSource",
        "completion_authority": False,
        "execution_completed": False,
        "self_certified_completion": False,
    }


def expand_inventory(inventory: Mapping[str, Any], *, active_plan_revision: str, run_id: str = RUN_ID) -> dict[str, Any]:
    """Return the deterministic child plan. This does not open a database."""
    _require_sha(active_plan_revision, "active plan revision")
    normalized, revision = _normalize_inventory(inventory, run_id=run_id, active_plan_revision=active_plan_revision)
    units = _units(normalized)
    if len(units) > MAX_UNITS or len(units) * len(PHASES) > MAX_CHILDREN:
        raise SourceExpansionError("expansion exceeds the bounded child task set")
    children = [_child(unit, index, phase, revision, active_plan_revision, normalized["namespace"]) for index, unit in enumerate(units) for phase in PHASES]
    if len(children) != len({item["task_id"] for item in children}) or len(children) != len({item["task_cid"] for item in children}):
        raise SourceExpansionError("source child identities collided")
    if any(_CHILD.fullmatch(item["task_id"]) is None for item in children):
        raise SourceExpansionError("source child id is outside the completion contract")
    counts = {level: sum(1 for unit in units if unit["level"] == level) for level in LEVELS}
    return {
        "schema": ADMISSION_SCHEMA,
        "inventory_revision": revision,
        "active_plan_revision": active_plan_revision,
        "namespace": normalized["namespace"],
        "units": units,
        "children": children,
        "national_count": counts["national"],
        "regional_count": counts["regional"],
        "local_count": counts["local"],
        "supranational_count": counts["supranational"],
        "unknown_count": counts["unknown"],
        "jurisdiction_counts": _jurisdiction_counts(units),
        "completion_authority": False,
        "execution_completed": False,
    }


def admit_source_expansion(
    database_path: Path | str,
    inventory: Mapping[str, Any],
    *,
    active_plan_revision: str,
    state_root: Path | str,
    blocker_path: Path | str | None = None,
    now: int,
    run_id: str = RUN_ID,
) -> dict[str, Any]:
    """Insert missing children and join every required receipt onto WWL-017.

    Preconditions are checked before any task, watchdog or blocker write.
    A second call with the same inventory is a replay: statuses, revisions
    and budgets stay where they are.
    """
    if type(now) is not int or now < 0:
        raise SourceExpansionError("expansion clock must be a whole timestamp")
    plan = expand_inventory(inventory, active_plan_revision=active_plan_revision, run_id=run_id)
    root = Path(state_root)
    if not root.is_dir():
        raise SourceExpansionError("watchdog state root is missing")
    state_path = root / "watchdog" / "state.json"
    readiness_path = root / "campaign-readiness.json"
    journal_path = root / "watchdog" / "restart-journal.json"
    state = _load_state(state_path, run_id)
    readiness = _load_optional(readiness_path, READINESS_SCHEMA, run_id, required_tasks=True)
    journal = _load_optional(journal_path, JOURNAL_SCHEMA, run_id, required_tasks=True)
    board = BlockerBoard.load(blocker_path) if blocker_path is not None else None
    if board is not None and board.run_id != run_id:
        raise SourceExpansionError("blocker run does not match the expansion")
    _reject_blocker_conflicts(board, plan["children"])
    from ..task_sources.database_task_source import DatabaseTaskSource

    path = Path(database_path)
    if not path.is_file():
        raise SourceExpansionError("native task database is missing")
    with DatabaseTaskSource(path, owner_id=run_id + ":source-expansion") as source:
        parent = source.get_task(PARENT_TASK_ID)
        join = source.get_task(JOIN_TASK_ID)
        _require_parent(parent, join, active_plan_revision, run_id)
        existing = {task.task_alias: task for task in _all_tasks(source)}
        _require_existing_children(existing, plan["children"])
        inserted = 0
        for child in plan["children"]:
            stored = existing.get(child["task_id"])
            if stored is None:
                _insert_child(source, child, parent)
                inserted += 1
            elif stored.task_cid != child["task_cid"] or stored.plan_cid != active_plan_revision:
                raise SourceExpansionError("source child contract changed")
        join = source.get_task(JOIN_TASK_ID)
        rewritten = _join_parent(source, join, plan, active_plan_revision)
        joined = source.get_task(JOIN_TASK_ID)
    child_ids = [item["task_id"] for item in plan["children"]]
    watchdog = _refresh_task_set(state_path, state, child_ids)
    if readiness is not None:
        _refresh_task_set(readiness_path, readiness, child_ids, budget_keys=())
    if journal is not None:
        _refresh_task_set(journal_path, journal, child_ids, budget_keys=PRESERVED_JOURNAL_KEYS)
    preserved = _preserve_blockers(Path(blocker_path), board, plan["children"], now) if board is not None and blocker_path is not None else []
    receipts = _join_receipts(plan)
    stored_join = dict(joined.body.get("source_expansion_join") or {})
    if [item["receipt_id"] for item in stored_join.get("required_receipts", [])] != [item["receipt_id"] for item in receipts]:
        raise SourceExpansionError("WWL-017 did not keep every required receipt")
    if not {item["task_cid"] for item in plan["children"]} <= set(joined.dependencies):
        raise SourceExpansionError("WWL-017 dependencies omit a required source child")
    return {
        "schema": ADMISSION_SCHEMA,
        "task_id": TASK_ID,
        "join_task_id": JOIN_TASK_ID,
        "inventory_revision": plan["inventory_revision"],
        "active_plan_revision": active_plan_revision,
        "task_ids": child_ids,
        "task_cids_by_alias": {item["task_id"]: item["task_cid"] for item in plan["children"]},
        "required_receipts": receipts,
        "inserted_count": inserted,
        "join_rewritten": rewritten,
        "idempotent_replay": inserted == 0 and not rewritten and not watchdog["written"],
        "national_count": plan["national_count"],
        "regional_count": plan["regional_count"],
        "local_count": plan["local_count"],
        "supranational_count": plan["supranational_count"],
        "unknown_count": plan["unknown_count"],
        "jurisdiction_counts": plan["jurisdiction_counts"],
        "blocked_task_ids": [item["task_id"] for item in plan["children"] if item["blocker_kind"]],
        "runnable_task_ids": [item["task_id"] for item in plan["children"] if not item["blocker_kind"]],
        "watchdog": {"actions": watchdog["actions"], "repairs": watchdog["repairs"], "budgets_reset": False, "required_task_ids": watchdog["required_task_ids"], "written": watchdog["written"]},
        "blockers_preserved": preserved,
        "completion_authority": False,
        "execution_completed": False,
    }


def verify_child_id(task_id: str) -> dict[str, Any]:
    """Identity check for one admitted child. This does not ingest a corpus."""
    if not isinstance(task_id, str) or _CHILD.fullmatch(task_id) is None:
        raise SourceExpansionError("source child id is outside the completion contract")
    phase = task_id.rsplit("-", 2)[-2]
    return {"schema": CHILD_SCHEMA, "task_id": task_id, "phase": phase, "completion_authority": False, "execution_completed": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check one admitted source-child identity.")
    parser.add_argument("command", choices=("verify-child", "contract"))
    parser.add_argument("task_id", nargs="?")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "contract":
            payload = published_contract()
        else:
            if not args.task_id:
                raise SourceExpansionError("source child id is outside the completion contract")
            payload = verify_child_id(args.task_id)
    except SourceExpansionError as error:
        print(json.dumps({"schema": ADMISSION_SCHEMA, "error": str(error), "completion_authority": False}))
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise SourceExpansionError(name + " is not a sha256 digest")
    return value


def _reject_forbidden(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in _RESET_KEYS:
                raise SourceExpansionError("inventory asks to reset budgets")
            if lowered in _SECRET_KEYS:
                raise SourceExpansionError("inventory contains a secret field")
            _reject_forbidden(item)
    elif isinstance(value, list):
        for item in value:
            _reject_forbidden(item)


def _exact_keys(value: Mapping[str, Any], allowed: frozenset[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise SourceExpansionError(name + " has unknown fields")


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise SourceExpansionError(name + " is not a bounded identifier")
    return value


def _reason(value: Any, name: str) -> str:
    if not isinstance(value, str) or _REASON.fullmatch(value) is None:
        raise SourceExpansionError(name + " must be bounded text")
    return value


def _normalize_inventory(inventory: Mapping[str, Any], *, run_id: str, active_plan_revision: str) -> tuple[dict[str, Any], str]:
    if not isinstance(inventory, Mapping):
        raise SourceExpansionError("inventory must be an object")
    _reject_forbidden(inventory)
    _exact_keys(inventory, _INVENTORY_KEYS, "inventory")
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise SourceExpansionError("inventory schema is not supported")
    if inventory.get("namespace") != run_id:
        raise SourceExpansionError("inventory namespace does not match this run")
    if inventory.get("complete") is not True:
        raise SourceExpansionError("partial inventory cannot admit source children")
    declared_plan = inventory.get("active_plan_revision")
    if declared_plan is not None and declared_plan != active_plan_revision:
        raise SourceExpansionError("active plan revision does not match the inventory binding")
    sources = inventory.get("sources")
    if not isinstance(sources, list) or not sources:
        raise SourceExpansionError("inventory has no sources")
    if len(sources) > MAX_UNITS:
        raise SourceExpansionError("expansion exceeds the bounded child task set")
    normalized_sources = [_normalize_source(item) for item in sources]
    if len({item["source_id"] for item in normalized_sources}) != len(normalized_sources):
        raise SourceExpansionError("inventory repeats a source")
    normalized = {"schema": INVENTORY_SCHEMA, "namespace": run_id, "complete": True, "sources": sorted(normalized_sources, key=lambda item: item["source_id"])}
    revision = identity(normalized)
    declared = inventory.get("inventory_revision")
    if declared is not None and declared != revision:
        raise SourceExpansionError("inventory revision does not match the inventory body")
    return normalized, revision


def _normalize_source(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise SourceExpansionError("source must be an object")
    _exact_keys(raw, _SOURCE_KEYS, "source")
    source_id = _identifier(raw.get("source_id"), "source id")
    eligible = raw.get("eligible")
    if type(eligible) is not bool:
        raise SourceExpansionError("source eligibility must be a boolean")
    blocker = _blocker(raw.get("blocker"), raw.get("access_error"))
    revision = raw.get("resolved_revision")
    if blocker is None:
        if eligible is not True:
            raise SourceExpansionError("ineligible source needs an explicit blocker")
        _require_sha(revision, "source revision")
    elif revision not in ("", None) and (not isinstance(revision, str) or _SHA.fullmatch(revision) is None):
        raise SourceExpansionError("source revision is not a sha256 digest")
    configs = raw.get("configs")
    if configs is None:
        configs = []
    if not isinstance(configs, list):
        raise SourceExpansionError("source configs must be a list")
    if not configs:
        if blocker is None:
            raise SourceExpansionError("source has no bounded units")
        configs = [_synthetic_config()]
    normalized = [_normalize_config(item) for item in configs]
    if len({item["config_id"] for item in normalized}) != len(normalized):
        raise SourceExpansionError("source repeats a config")
    return {
        "source_id": source_id,
        "resolved_revision": "" if revision in ("", None) else revision,
        "eligible": eligible,
        "configs": sorted(normalized, key=lambda item: item["config_id"]),
        "blocker_kind": "" if blocker is None else blocker["kind"],
        "blocker_reason": "" if blocker is None else blocker["reason"],
        "access_status": None if blocker is None else blocker["access_status"],
    }


def _blocker(blocker: Any, access: Any) -> dict[str, Any] | None:
    parsed = None
    if blocker is not None:
        if not isinstance(blocker, Mapping):
            raise SourceExpansionError("source blocker must be an object")
        _exact_keys(blocker, _BLOCKER_KEYS, "source blocker")
        kind = blocker.get("kind")
        if kind not in BLOCKER_KINDS:
            raise SourceExpansionError("source blocker kind is not recognized")
        parsed = {"kind": kind, "reason": _reason(blocker.get("reason"), "blocker reason"), "access_status": None}
    if access is not None:
        if not isinstance(access, Mapping):
            raise SourceExpansionError("access error must be an object")
        _exact_keys(access, _ACCESS_KEYS, "access error")
        status = access.get("status")
        if type(status) is not int or status not in ACCESS_KIND:
            raise SourceExpansionError("access error has no blocker kind")
        reason = _reason(access.get("reason"), "access error reason")
        if parsed is None:
            parsed = {"kind": ACCESS_KIND[status], "reason": reason, "access_status": status}
        else:
            parsed["access_status"] = status
    return parsed


def _synthetic_config() -> dict[str, Any]:
    return {
        "config_id": _SYNTHETIC["config_id"],
        "splits": [_SYNTHETIC["split_id"]],
        "jurisdictions": [{"jurisdiction_id": _SYNTHETIC["jurisdiction_id"], "level": _SYNTHETIC["level"]}],
        "shards": [{"shard_id": _SYNTHETIC["shard_id"], "bytes": _SYNTHETIC["bytes"]}],
    }


def _normalize_config(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise SourceExpansionError("config must be an object")
    _exact_keys(raw, _CONFIG_KEYS, "config")
    config_id = _identifier(raw.get("config_id"), "config id")
    splits = raw.get("splits")
    jurisdictions = raw.get("jurisdictions")
    shards = raw.get("shards")
    if not isinstance(splits, list) or not splits or not isinstance(jurisdictions, list) or not jurisdictions or not isinstance(shards, list) or not shards:
        raise SourceExpansionError("config is missing a split, jurisdiction or shard")
    if len(splits) * len(jurisdictions) * len(shards) > MAX_UNITS:
        raise SourceExpansionError("expansion exceeds the bounded child task set")
    norm_splits = [_identifier(item, "split id") for item in splits]
    if len(set(norm_splits)) != len(norm_splits):
        raise SourceExpansionError("config repeats a split")
    norm_jurisdictions = [_normalize_jurisdiction(item) for item in jurisdictions]
    if len({item["jurisdiction_id"] for item in norm_jurisdictions}) != len(norm_jurisdictions):
        raise SourceExpansionError("config repeats a jurisdiction")
    norm_shards = [_normalize_shard(item) for item in shards]
    if len({item["shard_id"] for item in norm_shards}) != len(norm_shards):
        raise SourceExpansionError("config repeats a shard")
    return {
        "config_id": config_id,
        "splits": sorted(norm_splits),
        "jurisdictions": sorted(norm_jurisdictions, key=lambda item: item["jurisdiction_id"]),
        "shards": sorted(norm_shards, key=lambda item: item["shard_id"]),
    }


def _normalize_jurisdiction(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise SourceExpansionError("jurisdiction must be an object")
    _exact_keys(raw, _JURISDICTION_KEYS, "jurisdiction")
    level = raw.get("level")
    if level not in LEVELS:
        raise SourceExpansionError("jurisdiction level is not recognized")
    return {"jurisdiction_id": _identifier(raw.get("jurisdiction_id"), "jurisdiction id"), "level": level}


def _normalize_shard(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise SourceExpansionError("shard must be an object")
    _exact_keys(raw, _SHARD_KEYS, "shard")
    size = raw.get("bytes")
    if type(size) is not int or size < 0:
        raise SourceExpansionError("shard size must be a whole byte count")
    return {"shard_id": _identifier(raw.get("shard_id"), "shard id"), "bytes": size}


def _units(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    units = []
    for source in inventory["sources"]:
        blocked = bool(source["blocker_kind"])
        for config in source["configs"]:
            for split_id in config["splits"]:
                for jurisdiction in config["jurisdictions"]:
                    if not blocked and jurisdiction["level"] not in RUNNABLE_LEVELS:
                        raise SourceExpansionError("runnable jurisdiction needs an explicit level")
                    for shard in config["shards"]:
                        units.append({
                            "source_id": source["source_id"],
                            "config_id": config["config_id"],
                            "split_id": split_id,
                            "jurisdiction_id": jurisdiction["jurisdiction_id"],
                            "level": jurisdiction["level"],
                            "shard_id": shard["shard_id"],
                            "shard_bytes": shard["bytes"],
                            "resolved_revision": source["resolved_revision"],
                            "blocker_kind": source["blocker_kind"],
                            "blocker_reason": source["blocker_reason"],
                            "access_status": source["access_status"],
                        })
    units.sort(key=lambda item: (item["source_id"], item["config_id"], item["split_id"], item["jurisdiction_id"], item["shard_id"]))
    if len(units) > MAX_UNITS:
        raise SourceExpansionError("expansion exceeds the bounded child task set")
    return units


def _unit_digest(unit: Mapping[str, Any], phase: str) -> str:
    payload = {key: unit[key] for key in ("source_id", "config_id", "split_id", "jurisdiction_id", "shard_id")}
    payload["phase"] = phase
    return identity(payload).removeprefix("sha256:")[:16]


def _child(unit: Mapping[str, Any], ordinal: int, phase: Mapping[str, str], revision: str, plan_revision: str, namespace: str) -> dict[str, Any]:
    phase_id = phase["id"]
    task_id = f"WWL-SRC-{_unit_digest(unit, phase_id)}-{phase_id}-{ordinal:04d}"
    previous = PHASE_IDS[PHASE_IDS.index(phase_id) - 1] if phase_id != PHASE_IDS[0] else ""
    depends_on = [PARENT_TASK_ID] if not previous else []
    task_cid = identity({
        "schema": "worldwide-legal-search.source-child-cid@1",
        "active_plan_revision": plan_revision,
        "inventory_revision": revision,
        "task_id": task_id,
        "phase": phase_id,
        "unit": {key: unit[key] for key in ("source_id", "config_id", "split_id", "jurisdiction_id", "shard_id")},
    })
    receipt_id = identity({
        "schema": "worldwide-legal-search.source-child-receipt@1",
        "task_id": task_id,
        "task_cid": task_cid,
        "inventory_revision": revision,
        "active_plan_revision": plan_revision,
        "phase": phase_id,
        "blocker_kind": unit["blocker_kind"],
        "jurisdiction_id": unit["jurisdiction_id"],
        "jurisdiction_level": unit["level"],
    })
    body = {
        "schema": CHILD_SCHEMA,
        "task_id": task_id,
        "task_cid": task_cid,
        "phase": phase_id,
        "phase_title": phase["title"],
        "grain": phase["grain"],
        "source_id": unit["source_id"],
        "config_id": unit["config_id"],
        "split_id": unit["split_id"],
        "jurisdiction_id": unit["jurisdiction_id"],
        "jurisdiction_level": unit["level"],
        "shard_id": unit["shard_id"],
        "shard_bytes": unit["shard_bytes"],
        "resolved_revision": unit["resolved_revision"],
        "inventory_revision": revision,
        "active_plan_revision": plan_revision,
        "parent_task_id": PARENT_TASK_ID,
        "join_task_id": JOIN_TASK_ID,
        "blocker_kind": unit["blocker_kind"],
        "blocker_reason": unit["blocker_reason"],
        "access_status": unit["access_status"],
        "ordinal": ordinal,
        "receipt_id": receipt_id,
        "depends_on": depends_on,
        "board_namespace": namespace,
        "owning_repository": "accelerate",
        "completion": "auto",
        "completion_authority": False,
        "execution_completed": False,
    }
    return {**body, "status": "blocked" if unit["blocker_kind"] else "todo", "previous_phase": previous}


def _jurisdiction_counts(units: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], int] = {}
    for unit in units:
        key = (unit["jurisdiction_id"], unit["level"])
        grouped[key] = grouped.get(key, 0) + 1
    return [{"jurisdiction_id": key[0], "level": key[1], "units": grouped[key]} for key in sorted(grouped)]


def _load_state(path: Path, run_id: str) -> dict[str, Any]:
    try:
        value = read_owned_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SourceExpansionError("watchdog task set is absent or unreadable") from error
    if value.get("schema") != WATCHDOG_STATE_SCHEMA or value.get("run_id") != run_id:
        raise SourceExpansionError("watchdog task set run differs")
    for key in ("actions", "repairs"):
        if type(value.get(key)) is not int or value[key] < 0:
            raise SourceExpansionError("watchdog budget counter is invalid")
    if not isinstance(value.get("incidents"), dict) or not isinstance(value.get("required_task_ids"), list):
        raise SourceExpansionError("watchdog task set is missing incidents or tasks")
    return value


def _load_optional(path: Path, schema: str, run_id: str, *, required_tasks: bool) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = read_owned_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SourceExpansionError("persisted task set is unreadable") from error
    if value.get("schema") != schema or value.get("run_id") != run_id:
        raise SourceExpansionError("persisted task set run differs")
    if required_tasks and not isinstance(value.get("required_task_ids"), list):
        raise SourceExpansionError("persisted task set is missing tasks")
    if value.get("completion_authority") is True:
        raise SourceExpansionError("persisted task set grants completion authority")
    return value


def _reject_blocker_conflicts(board: BlockerBoard | None, children: Sequence[Mapping[str, Any]]) -> None:
    if board is None:
        return
    for child in children:
        existing = board.records.get(child["task_id"])
        if existing is None:
            continue
        if existing["kind"] != child["blocker_kind"]:
            raise SourceExpansionError("blocker kind would change")
        if existing["exhausted_budgets"] and child["blocker_kind"] == "":
            raise SourceExpansionError("blocker kind would change")


def _require_parent(parent: Any, join: Any, active_plan_revision: str, run_id: str) -> None:
    if parent is None or join is None:
        raise SourceExpansionError("native store is missing WWL-017 or WWL-018")
    if parent.plan_cid != active_plan_revision or join.plan_cid != active_plan_revision:
        raise SourceExpansionError("active plan revision does not match the native store")
    if parent.body.get("board_namespace") not in (None, run_id) or join.body.get("board_namespace") not in (None, run_id):
        raise SourceExpansionError("native store belongs to a different run")
    if parent.goal_cid != join.goal_cid or not parent.goal_cid:
        raise SourceExpansionError("WWL-017 and WWL-018 are not bound to one goal")


def _all_tasks(source: Any) -> list[Any]:
    tasks = []
    cursor = ""
    while True:
        page = source.list_tasks(limit=500, cursor=cursor)
        tasks.extend(page.tasks)
        if not page.next_cursor:
            return tasks
        cursor = page.next_cursor
        if len(tasks) > MAX_CHILDREN + 256:
            raise SourceExpansionError("native task population exceeds the expansion bound")


def _require_existing_children(existing: Mapping[str, Any], children: Sequence[Mapping[str, Any]]) -> None:
    planned = {item["task_id"]: item for item in children}
    for alias, task in existing.items():
        body = task.body
        if not isinstance(body, Mapping) or body.get("schema") != CHILD_SCHEMA:
            continue
        if alias not in planned:
            raise SourceExpansionError("required source child disappeared")
        current = planned[alias]
        for key in ("task_cid", "receipt_id", "inventory_revision", "active_plan_revision", "phase", "blocker_kind", "source_id", "config_id", "split_id", "jurisdiction_id", "shard_id", "shard_bytes", "jurisdiction_level"):
            if body.get(key) != current[key]:
                raise SourceExpansionError("source child contract changed")
        if task.status == "blocked" and not current["blocker_kind"]:
            raise SourceExpansionError("blocker kind would change")
        if task.status != "blocked" and current["blocker_kind"] and task.status not in {"todo", "blocked"}:
            raise SourceExpansionError("blocker kind would change")


def _insert_child(source: Any, child: Mapping[str, Any], parent: Any) -> None:
    dependencies = _dependency_cids(source, child)
    output = "artifacts/worldwide-legal-search/receipts/" + child["task_id"] + ".json"
    argv = ["python3", "-m", "ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_source_expansion", "verify-child", child["task_id"]]
    source.intent.upsert_task(
        task_cid=child["task_cid"],
        task_alias=child["task_id"],
        goal_cid=parent.goal_cid,
        plan_cid=parent.plan_cid,
        objective_id=parent.objective_id,
        ordinal=1000 + int(child["ordinal"]) * len(PHASES) + PHASE_IDS.index(child["phase"]),
        status=child["status"],
        priority="P1",
        body={key: child[key] for key in child if key not in {"status", "previous_phase"}},
        identity={"task_cid": child["task_cid"], "task_alias": child["task_id"], "inventory_revision": child["inventory_revision"], "active_plan_revision": child["active_plan_revision"]},
        dependencies=dependencies,
        outputs=[{"path": output, "effect_id": child["receipt_id"]}],
        acceptance=["Admit one bounded source child without granting completion."],
        validations=[{"argv": argv}],
    )


def _dependency_cids(source: Any, child: Mapping[str, Any]) -> list[str]:
    if child["previous_phase"]:
        previous_id = f"WWL-SRC-{_unit_digest(_child_unit(child), child['previous_phase'])}-{child['previous_phase']}-{int(child['ordinal']):04d}"
        previous = source.get_task(previous_id)
        if previous is None:
            raise SourceExpansionError("previous source phase is missing")
        return [previous.task_cid]
    parent = source.get_task(PARENT_TASK_ID)
    if parent is None:
        raise SourceExpansionError("native store is missing WWL-017 or WWL-018")
    return [parent.task_cid]


def _child_unit(child: Mapping[str, Any]) -> dict[str, Any]:
    return {key: child[key] for key in ("source_id", "config_id", "split_id", "jurisdiction_id", "shard_id")}


def _join_receipts(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    receipts = []
    for child in plan["children"]:
        receipts.append({
            "task_id": child["task_id"],
            "task_cid": child["task_cid"],
            "receipt_id": child["receipt_id"],
            "phase": child["phase"],
            "source_id": child["source_id"],
            "jurisdiction_id": child["jurisdiction_id"],
            "jurisdiction_level": child["jurisdiction_level"],
            "blocker_kind": child["blocker_kind"],
            "ordinal": child["ordinal"],
        })
    return receipts


def _join_document(plan: Mapping[str, Any], active_plan_revision: str) -> dict[str, Any]:
    receipts = _join_receipts(plan)
    return {
        "schema": JOIN_SCHEMA,
        "join_task_id": JOIN_TASK_ID,
        "parent_task_id": PARENT_TASK_ID,
        "inventory_revision": plan["inventory_revision"],
        "active_plan_revision": active_plan_revision,
        "required_receipts": receipts,
        "required_receipt_ids": [item["receipt_id"] for item in receipts],
        "national_count": plan["national_count"],
        "regional_count": plan["regional_count"],
        "local_count": plan["local_count"],
        "supranational_count": plan["supranational_count"],
        "unknown_count": plan["unknown_count"],
        "jurisdiction_counts": plan["jurisdiction_counts"],
        "completion_authority": False,
    }


def _join_parent(source: Any, join: Any, plan: Mapping[str, Any], active_plan_revision: str) -> bool:
    document = _join_document(plan, active_plan_revision)
    desired = list(dict.fromkeys([*join.dependencies, *(item["task_cid"] for item in plan["children"])]))
    current = dict(join.body.get("source_expansion_join") or {})
    if current == document and set(join.dependencies) == set(desired):
        return False
    body = json.loads(json.dumps(dict(join.body)))
    body["source_expansion_join"] = document
    identity_map = json.loads(json.dumps(dict(getattr(join, "body", {}))))
    stored_identity = source.intent.get_task(join.task_cid)
    if stored_identity is not None and isinstance(stored_identity.get("identity"), Mapping):
        identity_map = json.loads(json.dumps(dict(stored_identity["identity"])))
    source.intent.upsert_task(
        task_cid=join.task_cid,
        task_alias=join.task_alias,
        goal_cid=join.goal_cid,
        plan_cid=join.plan_cid,
        objective_id=join.objective_id,
        ordinal=int(join.ordinal),
        status=join.status,
        priority=join.priority or "P1",
        body=body,
        identity=identity_map,
        dependencies=desired,
        expected_revision=int(join.revision),
    )
    return True


def _extend_task_ids(existing: Any, child_ids: Sequence[str]) -> list[str]:
    if not isinstance(existing, list) or not all(isinstance(item, str) and item for item in existing):
        raise SourceExpansionError("persisted task set is invalid")
    if len(set(existing)) != len(existing):
        raise SourceExpansionError("persisted task set repeats a task")
    extended = list(existing)
    for task_id in child_ids:
        if task_id not in extended:
            extended.append(task_id)
    if not set(existing) <= set(extended):
        raise SourceExpansionError("persisted task set would shrink")
    return extended


def _budget_view(value: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: copy.deepcopy(value.get(key)) for key in keys}


def _refresh_task_set(path: Path, value: Mapping[str, Any], child_ids: Sequence[str], budget_keys: Sequence[str] = PRESERVED_STATE_KEYS) -> dict[str, Any]:
    before = _budget_view(value, budget_keys)
    updated = copy.deepcopy(dict(value))
    updated["required_task_ids"] = _extend_task_ids(value.get("required_task_ids"), child_ids)
    if _budget_view(updated, budget_keys) != before:
        raise SourceExpansionError("watchdog budget snapshot changed")
    written = updated != dict(value)
    if written:
        atomic_json(path, updated)
        stored = read_owned_json(path)
        if _budget_view(stored, budget_keys) != before:
            raise SourceExpansionError("watchdog budget snapshot changed")
        if stored.get("required_task_ids") != updated["required_task_ids"]:
            raise SourceExpansionError("persisted task set was not refreshed")
    return {
        "required_task_ids": list(updated["required_task_ids"]),
        "actions": updated.get("actions"),
        "repairs": updated.get("repairs"),
        "written": written,
    }


def _preserve_blockers(path: Path, board: BlockerBoard, children: Sequence[Mapping[str, Any]], now: int) -> list[str]:
    original = {task_id: dict(record) for task_id, record in board.records.items()}
    original_budgets = {task_id: list(record["exhausted_budgets"]) for task_id, record in original.items()}
    original_probes = board.to_dict()["probes"]
    appended = []
    for child in children:
        kind = child["blocker_kind"]
        if not kind:
            continue
        existing = board.records.get(child["task_id"])
        if existing is None:
            if kind not in BLOCKER_KINDS or (kind in RECHECKABLE) is not True:
                raise SourceExpansionError("source blocker kind is not recognized")
            board.records[child["task_id"]] = {
                "task_id": child["task_id"],
                "classification": kind,
                "kind": kind,
                "prerequisite_digest": child["receipt_id"],
                "exhausted_budgets": [],
                "exhaustion_source": "",
                "incident_key": "",
                "classified_at": now,
                "recheckable": True,
            }
            appended.append(child["task_id"])
        elif existing["kind"] != kind or list(existing["exhausted_budgets"]) != list(original_budgets.get(child["task_id"], existing["exhausted_budgets"])):
            raise SourceExpansionError("blocker kind would change")
    for task_id, record in original.items():
        current = board.records.get(task_id)
        if current is None or current["kind"] != record["kind"] or list(current["exhausted_budgets"]) != list(record["exhausted_budgets"]) or current["incident_key"] != record["incident_key"] or current["classified_at"] != record["classified_at"]:
            raise SourceExpansionError("existing blocker was rewritten")
    if board.to_dict()["probes"] != original_probes:
        raise SourceExpansionError("existing blocker probe was rewritten")
    board.save(path)
    reloaded = BlockerBoard.load(path)
    for task_id, record in original.items():
        current = reloaded.records[task_id]
        if current["kind"] != record["kind"] or list(current["exhausted_budgets"]) != list(record["exhausted_budgets"]):
            raise SourceExpansionError("existing blocker was rewritten")
    return sorted(original)


if __name__ == "__main__":
    raise SystemExit(main())
