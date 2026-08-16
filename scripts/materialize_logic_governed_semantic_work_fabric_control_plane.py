#!/usr/bin/env python3
"""Materialize and verify the LGSWF board in the existing DuckDB control plane.

The Markdown documents are a sealed bootstrap input only.  Once this command
creates the store, ``DatabaseTaskSource@1`` is task authority and the files are
never used for task-status mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/logic_governed_semantic_work_fabric_scheduler.json"
TASK_RE = re.compile(r"^## (LGSWF-(\d{3})) (.+)$", re.MULTILINE)
GOAL_RE = re.compile(r"^## (LGSWF-G\d{3}) (.+)$", re.MULTILINE)
META_RE = re.compile(r"^- ([^:\n]+):(?: (.*))?$", re.MULTILINE)
SCHEMA = "ipfs_accelerate_py/agent-supervisor/lgswf-duckdb-materialization@1"


class MaterializationError(RuntimeError):
    """Fail-closed bootstrap materialization error."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _identity(value: Any) -> str:
    payload = value if isinstance(value, bytes) else _canonical_bytes(value)
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _load_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MaterializationError("scheduler config root must be an object")
    program = value.get("database_program")
    if not isinstance(program, dict):
        raise MaterializationError("database_program is required")
    if program.get("task_source_kind") != "duckdb":
        raise MaterializationError("task_source_kind must be duckdb")
    if program.get("authority_mode") != "embedded":
        raise MaterializationError(
            "bootstrap materializer only admits the bounded embedded single writer"
        )
    writer = value.get("bootstrap_writer_policy")
    if not isinstance(writer, dict) or writer.get("maximum_processes") != 1:
        raise MaterializationError("bootstrap writer policy must cap processes at one")
    return value


def _relative_path(value: Any, *, field: str) -> Path:
    text = str(value or "").strip()
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise MaterializationError(f"{field} must be a safe repository-relative path")
    resolved = (ROOT / path).resolve(strict=False)
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise MaterializationError(f"{field} escapes repository") from exc
    return resolved


def _metadata(body: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for match in META_RE.finditer(body):
        key = match.group(1).strip()
        if key in result:
            raise MaterializationError(f"duplicate metadata field: {key}")
        result[key] = (match.group(2) or "").strip()
    return result


def _records(text: str, pattern: re.Pattern[str]) -> list[tuple[str, str, str]]:
    matches = list(pattern.finditer(text))
    records: list[tuple[str, str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end]
        records.append((match.group(1), match.group(match.lastindex or 1), body))
    return records


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalized_body(
    *, task_id: str, title: str, metadata: Mapping[str, str], block: str
) -> dict[str, Any]:
    body = {
        re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_"): value
        for key, value in metadata.items()
    }
    body.update(
        {
            "task_id": task_id,
            "title": title,
            "objective": metadata.get("Objective", title),
            "completion": metadata.get("Completion", "auto"),
            "completion_contract": metadata.get("Completion contract", ""),
            "validation": metadata.get("Validation", ""),
            "validation_requirements": metadata.get("Validation requirements", ""),
            "proof_requirements": metadata.get("Proof requirements", ""),
            "acceptance": metadata.get("Acceptance", ""),
            "acceptance_criteria": metadata.get("Acceptance", ""),
            "outputs": _csv(metadata.get("Outputs", "")),
            "predicted_files": _csv(metadata.get("Predicted files", "")),
            "depends_on": _csv(metadata.get("Depends on", "")),
            "source_block_sha256": _identity(
                f"## {task_id} {title}{block}".encode("utf-8")
            ),
        }
    )
    return body


def build_population(config: Mapping[str, Any]) -> dict[str, Any]:
    taskboard_path = _relative_path(config.get("taskboard_path"), field="taskboard_path")
    objectives_path = _relative_path(config.get("objectives_path"), field="objectives_path")
    plan_path = _relative_path(config.get("plan_path"), field="plan_path")
    board_bytes = taskboard_path.read_bytes()
    objective_bytes = objectives_path.read_bytes()
    plan_bytes = plan_path.read_bytes()
    config_bytes = CONFIG_PATH.read_bytes()

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    base = str(
        (config.get("source_binding") or {}).get("accelerator_required_ancestor")
        or ""
    )
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", base, "HEAD"], cwd=ROOT
    ).returncode != 0:
        raise MaterializationError("configured accelerator base is not an ancestor")

    plan_root = _identity(
        {
            "schema": "lgswf-plan-root@1",
            "board": _identity(board_bytes),
            "objectives": _identity(objective_bytes),
            "plan": _identity(plan_bytes),
            "config": _identity(config_bytes),
            "source_head": head,
            "source_tree": tree,
        }
    )
    objective_text = objective_bytes.decode("utf-8")
    goals: list[dict[str, Any]] = []
    goal_cids: dict[str, str] = {}
    for ordinal, (goal_id, title, body_text) in enumerate(
        _records(objective_text, GOAL_RE), start=1
    ):
        metadata = _metadata(body_text)
        goal_cid = _identity(
            {
                "goal_id": goal_id,
                "title": title,
                "body_sha256": _identity(body_text.encode("utf-8")),
                "plan_root_cid": plan_root,
            }
        )
        goal_cids[goal_id] = goal_cid
        goals.append(
            {
                "goal_cid": goal_cid,
                "goal_id": goal_id,
                "goal_alias": goal_id,
                "title": title,
                "ordinal": ordinal,
                "status": "open",
                "objective_id": "objective:lgswf-root" if goal_id == "LGSWF-G000" else "",
                "objective_alias": "LGSWF-G000",
                "priority": metadata.get("Priority", "P0"),
                "body": metadata,
            }
        )
    if len(goals) != int((config.get("initial_projection") or {}).get("goal_count", -1)):
        raise MaterializationError("goal count differs from sealed projection")

    task_text = board_bytes.decode("utf-8")
    tasks: list[dict[str, Any]] = []
    task_cids: dict[str, str] = {}
    parsed = _records(task_text, TASK_RE)
    for task_id, title, body_text in parsed:
        metadata = _metadata(body_text)
        task_cids[task_id] = _identity(
            {
                "task_id": task_id,
                "block_sha256": _identity(
                    f"## {task_id} {title}{body_text}".encode("utf-8")
                ),
                "plan_root_cid": plan_root,
                "repository_tree_id": tree,
            }
        )
    for ordinal, (task_id, title, body_text) in enumerate(parsed, start=1):
        metadata = _metadata(body_text)
        dependency_aliases = _csv(metadata.get("Depends on", ""))
        try:
            dependencies = [task_cids[item] for item in dependency_aliases]
        except KeyError as exc:
            raise MaterializationError(f"{task_id} has unknown dependency {exc}") from exc
        goal_alias = metadata.get("Subgoal ID") or metadata.get("Goal id") or "LGSWF-G000"
        if goal_alias not in goal_cids:
            raise MaterializationError(f"{task_id} has unknown goal {goal_alias}")
        output_paths = _csv(metadata.get("Outputs", ""))
        normalized = _normalized_body(
            task_id=task_id,
            title=title,
            metadata=metadata,
            block=body_text,
        )
        tasks.append(
            {
                **normalized,
                "task_cid": task_cids[task_id],
                "task_id": task_id,
                "task_alias": task_id,
                "goal_cid": goal_cids[goal_alias],
                "plan_cid": plan_root,
                "objective_id": "objective:lgswf-root",
                "ordinal": ordinal,
                "status": metadata.get("Status", "todo"),
                "priority": metadata.get("Priority", "P0"),
                "title": title,
                "dependencies": dependencies,
                "outputs": [
                    {"path": path, "effect_id": _identity({"task": task_id, "path": path})}
                    for path in output_paths
                ],
                "acceptance": [metadata.get("Acceptance", "")],
                "validations": [metadata.get("Validation", "")],
            }
        )
    if len(tasks) != int((config.get("initial_projection") or {}).get("task_count", -1)):
        raise MaterializationError("task count differs from sealed projection")
    return {
        "schema": "ipfs_accelerate_py/agent-supervisor/lgswf-population@1",
        "repository_tree_id": tree,
        "source_head": head,
        "plan_root_cid": plan_root,
        "objectives": goals,
        "plans": [
            {
                "plan_cid": plan_root,
                "plan_alias": "LGSWF-PLAN-ACTUAL-R1",
                "goal_cid": goal_cids["LGSWF-G000"],
                "status": "active",
                "source_head": head,
                "repository_tree_id": tree,
            }
        ],
        "tasks": tasks,
        "task_cids_by_alias": task_cids,
        "goal_cids_by_alias": goal_cids,
    }


def _paths(config: Mapping[str, Any]) -> dict[str, Path]:
    program = config["database_program"]
    control = _relative_path(program.get("store_id"), field="database_program.store_id")
    return {
        "control": control,
        "coordination": control.with_name(f"{control.stem}.coordination.duckdb"),
        "execution": control.with_name(f"{control.stem}.execution.duckdb"),
    }


def _verify_store(config: Mapping[str, Any], population: Mapping[str, Any]) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import (
        DatabaseTaskSource,
    )
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import (
        DatabaseCoordinationNotReadyError,
        open_database_coordinator,
    )

    paths = _paths(config)
    missing = [key for key, path in paths.items() if not path.is_file()]
    if missing:
        raise MaterializationError(f"control-plane files missing: {missing}")
    task_source = DatabaseTaskSource(
        paths["control"],
        owner_id="lgswf-materializer:verify",
        repository_tree_id=str(population["repository_tree_id"]),
        plan_root_cid=str(population["plan_root_cid"]),
        install_schema=False,
    )
    coordinator = open_database_coordinator(paths["coordination"])
    try:
        snapshot = task_source.snapshot()
        page = task_source.list_tasks(limit=100)
        ready = task_source.ready_tasks(limit=100)
        aliases = [item.task_alias for item in page.tasks]
        ready_aliases = [item.task_alias for item in ready.tasks]
        expected = list((config.get("initial_projection") or {}).get("ready_task_ids") or [])
        if len(aliases) != len(population["tasks"]):
            raise MaterializationError("database task population is incomplete")
        if ready_aliases != expected:
            raise MaterializationError(
                f"database ready frontier mismatch: {ready_aliases!r} != {expected!r}"
            )
        completed_cid = population["task_cids_by_alias"]["LGSWF-000"]
        try:
            coordinator.claim_task(
                task_cid=completed_cid,
                owner_session_id="lgswf-materializer:completed-probe",
            )
        except DatabaseCoordinationNotReadyError as exc:
            if exc.evidence.get("reason") != "already_completed":
                raise MaterializationError(
                    "bootstrap coordination completion has the wrong evidence"
                ) from exc
        else:
            raise MaterializationError(
                "bootstrap completion is absent from coordination"
            )
        if coordinator.claimability(population["task_cids_by_alias"]["LGSWF-001"])[
            "claimable"
        ] is not True:
            raise MaterializationError("LGSWF-001 is not claimable after bootstrap seal")
        return {
            "task_source_snapshot": snapshot.to_dict(),
            "task_aliases": aliases,
            "ready_task_aliases": ready_aliases,
            "completed_task_aliases": ["LGSWF-000"],
            "database_identities": {
                key: _sha256_file(path) for key, path in sorted(paths.items())
            },
        }
    finally:
        coordinator.close()
        task_source.close()


def materialize(config: Mapping[str, Any], population: Mapping[str, Any]) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        DatabaseImplementationDaemon,
    )

    paths = _paths(config)
    existing = [path.relative_to(ROOT).as_posix() for path in paths.values() if path.exists()]
    if existing:
        raise MaterializationError(
            "refusing to overwrite an existing control plane: " + ", ".join(existing)
        )
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    daemon = DatabaseImplementationDaemon(
        database_path=paths["control"],
        coordination_path=paths["coordination"],
        execution_path=paths["execution"],
        owner_session_id="lgswf-materializer:single-writer",
        authority_mode="embedded",
        task_source_kind="duckdb",
        install_schema=True,
    )
    try:
        database_receipt = daemon.materialize_population(
            population,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"]),
        )
        completed_cid = population["task_cids_by_alias"]["LGSWF-000"]
        daemon.coordinator.mark_task_complete(
            completed_cid,
            status="succeeded",
            body={
                "kind": "imported_control_seal",
                "plan_root_cid": population["plan_root_cid"],
                "source_head": population["source_head"],
            },
        )
    finally:
        daemon.close()
    verified = _verify_store(config, population)
    receipt = {
        "schema": SCHEMA,
        "authority_mode": "embedded",
        "maximum_writer_processes": 1,
        "task_source_kind": "duckdb",
        "source_head": population["source_head"],
        "repository_tree_id": population["repository_tree_id"],
        "plan_root_cid": population["plan_root_cid"],
        "database_paths": {
            key: path.relative_to(ROOT).as_posix() for key, path in sorted(paths.items())
        },
        "materialization": dict(database_receipt),
        "verification": verified,
        "population_cid": _identity(population),
    }
    receipt["receipt_cid"] = _identity(receipt)
    evidence_root = _relative_path(
        config["runtime_paths"]["evidence"], field="runtime_paths.evidence"
    )
    receipt_path = evidence_root / "bootstrap" / "duckdb-materialization.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=receipt_path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, receipt_path)
    return {**receipt, "receipt_path": receipt_path.relative_to(ROOT).as_posix()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("population", "materialize", "verify"), nargs="?", default="verify"
    )
    args = parser.parse_args(argv)
    try:
        config = _load_config()
        population = build_population(config)
        if args.command == "population":
            result: Mapping[str, Any] = {
                "schema": population["schema"],
                "population_cid": _identity(population),
                "plan_root_cid": population["plan_root_cid"],
                "repository_tree_id": population["repository_tree_id"],
                "task_count": len(population["tasks"]),
                "goal_count": len(population["objectives"]),
            }
        elif args.command == "materialize":
            result = materialize(config, population)
        else:
            result = {
                "schema": SCHEMA,
                "valid": True,
                "plan_root_cid": population["plan_root_cid"],
                "verification": _verify_store(config, population),
            }
        json.dump(result, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0
    except Exception as exc:
        json.dump(
            {
                "schema": SCHEMA,
                "valid": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            sys.stdout,
            indent=2,
            sort_keys=True,
        )
        sys.stdout.write("\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
