#!/usr/bin/env python3
"""Render, install and verify the append-only SAWM R2 supervisor program.

The default action creates a new datasets-authoritative operational DuckDB
store, proves the exact population, then atomically publishes it.  Existing
exact populations are verified without mutation; any different population is
reported as ``migration_required`` and is never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NAMESPACE = "semantic-addressed-world-model-v1"
REVISION = "SAWM-PLAN-R2"
ROOT_GOAL = "SAWM-G000"
SCHEMA = "ipfs_accelerate_py/agent-supervisor/semantic-addressed-world-model-materialization@1"
CONFIG_PATH = REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"


class MaterializationError(RuntimeError):
    """Fail-closed SAWM bootstrap error."""


class MigrationRequired(MaterializationError):
    """An existing append-only authority contains a different population."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _identity(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    def closed(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=closed)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _board_module(root: Path):
    import importlib.util
    name = "_sawm_board_validator_for_materializer"
    path = root / "scripts/validate_semantic_addressed_world_model_board.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise MaterializationError("unable to load the sealed board parser")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=False, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20,
    )
    if result.returncode:
        raise MaterializationError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _source_binding(root: Path) -> dict[str, Any]:
    controls = (
        ".gitignore",
        "docs/architecture/SEMANTIC_ADDRESSED_WORLD_MODEL_PLAN.md",
        "docs/architecture/semantic_addressed_world_model.objectives.md",
        "docs/architecture/semantic_addressed_world_model.todo.md",
        "docs/architecture/semantic_addressed_world_model_inventory/repository_baseline.json",
        "docs/architecture/semantic_addressed_world_model_inventory/authority_matrix.json",
        "docs/architecture/semantic_addressed_world_model_inventory/overlap_gap_matrix.json",
        "docs/architecture/semantic_addressed_world_model_inventory/identity_inventory.json",
        "docs/architecture/semantic_addressed_world_model_inventory/interface_inventory.json",
        "docs/architecture/semantic_addressed_world_model_inventory/dependency_graph.json",
        "docs/architecture/semantic_addressed_world_model_inventory/capability_matrix.json",
        "docs/architecture/semantic_addressed_world_model_inventory/rollout_baseline.json",
        "config/semantic_addressed_world_model_dependencies.seal.json",
        "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
        "scripts/validate_semantic_addressed_world_model_dependencies.py",
        "scripts/validate_semantic_addressed_world_model_board.py",
        "scripts/materialize_semantic_addressed_world_model_program.py",
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "test/api/semantic_world/test_semantic_addressed_world_model_board.py",
        "benchmarks/agent_supervisor/semantic_addressed_world_model/benchmark_freeze.json",
    )
    file_digests = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in controls
        if (root / path).is_file()
    }
    payload = {
        "schema": "sawm/current-source-binding@1",
        "head": _git(root, "rev-parse", "HEAD"),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "branch": _git(root, "branch", "--show-current"),
        "datasets_gitlink": _git(root, "rev-parse", "HEAD:ipfs_datasets_py"),
        "kit_gitlink": _git(root, "rev-parse", "HEAD:ipfs_kit_py"),
        "control_sha256": file_digests,
    }
    return {**payload, "source_binding_cid": _identity(payload)}


def _metadata_payload(card: Any) -> dict[str, Any]:
    """Retain every closed board field without treating Markdown as state."""
    return {str(key).replace(" ", "_"): str(value) for key, value in sorted(card.metadata.items())}


def build_population(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    board = _board_module(root)
    tasks = board._parse_cards(root / "docs/architecture/semantic_addressed_world_model.todo.md", goal=False)
    goals = board._parse_cards(root / "docs/architecture/semantic_addressed_world_model.objectives.md", goal=True)
    if tuple(item.identifier for item in tasks) != board.TASK_IDS or tuple(item.identifier for item in goals) != board.GOAL_IDS:
        raise MaterializationError("board population differs from the sealed 45-task/29-goal identity")
    source = _source_binding(root)

    goal_cids: dict[str, str] = {}
    goal_definitions: dict[str, dict[str, Any]] = {}
    for ordinal, card in enumerate(goals, 1):
        definition = {
            "schema": "sawm/goal-definition@1", "board_namespace": NAMESPACE,
            "plan_revision": REVISION, "goal_id": card.identifier,
            "title": card.title, "ordinal": ordinal,
            "metadata": _metadata_payload(card), "source_binding_cid": source["source_binding_cid"],
        }
        goal_definitions[card.identifier] = definition
        goal_cids[card.identifier] = _identity(definition)

    objective_id = _identity({"schema": "sawm/objective@1", "namespace": NAMESPACE, "revision": REVISION})
    objective_rows: list[dict[str, Any]] = []
    goal_edges: list[dict[str, Any]] = []
    for ordinal, card in enumerate(goals, 1):
        parent_alias = board.GOAL_PARENT[card.identifier]
        definition = goal_definitions[card.identifier]
        row = {
            "goal_cid": goal_cids[card.identifier], "goal_id": card.identifier,
            "goal_alias": card.identifier, "title": card.title, "ordinal": ordinal,
            "status": str(card.metadata.get("status") or "open"),
            "parent_goal_cid": goal_cids[parent_alias] if parent_alias else "",
            "definition_cid": _identity(definition), "definition": definition,
            "board_namespace": NAMESPACE, "plan_revision": REVISION,
        }
        if card.identifier == ROOT_GOAL:
            row.update({"objective_id": objective_id, "objective_alias": ROOT_GOAL,
                        "priority": str(card.metadata.get("priority") or "P0")})
        objective_rows.append(row)
        if parent_alias:
            goal_edges.append({"parent_goal_cid": goal_cids[parent_alias],
                               "child_goal_cid": goal_cids[card.identifier],
                               "edge_kind": "goal_refinement"})

    plan_definition = {
        "schema": "sawm/plan-definition@1", "board_namespace": NAMESPACE,
        "plan_revision": REVISION, "root_goal_cid": goal_cids[ROOT_GOAL],
        "goal_definition_cids": [goal_cids[item.identifier] for item in goals],
        "source_binding_cid": source["source_binding_cid"],
    }
    plan_cid = _identity(plan_definition)

    task_definitions: dict[str, dict[str, Any]] = {}
    task_cids: dict[str, str] = {}
    for ordinal, card in enumerate(tasks, 1):
        definition = {
            "schema": "sawm/task-definition@1", "board_namespace": NAMESPACE,
            "plan_revision": REVISION, "task_id": card.identifier,
            "title": card.title, "ordinal": ordinal,
            "metadata": _metadata_payload(card), "source_binding_cid": source["source_binding_cid"],
        }
        task_definitions[card.identifier] = definition
        task_cids[card.identifier] = _identity(definition)

    taskboard: list[dict[str, Any]] = []
    for ordinal, card in enumerate(tasks, 1):
        meta = card.metadata
        deps = json.loads(meta["dependencies json"])
        outputs = json.loads(meta["outputs json"])
        validations = json.loads(meta["validation commands json"])
        goal_alias = str(meta["goal id"])
        definition = task_definitions[card.identifier]
        taskboard.append(
            {
                "task_cid": task_cids[card.identifier], "task_id": card.identifier,
                "task_alias": card.identifier, "title": card.title, "ordinal": ordinal,
                # The operator card is deliberately born ready.  Its Markdown
                # completed marker cannot bypass current evidence and CAS.
                "status": "ready" if card.identifier == "SAWM-000" else "todo",
                "priority": str(meta.get("priority") or "P2"),
                "goal_cid": goal_cids[goal_alias], "goal_id": goal_alias,
                "plan_cid": plan_cid,
                "depends_on": [task_cids[str(dep)] for dep in deps],
                # IntentRepository output keys use the closed safe-ID grammar;
                # retain the exact file path inside the canonical effect body.
                "outputs": [{"effect_id": _identity({"task": card.identifier, "path": str(path)}),
                             "declared_path": str(path), "effect": "declared_output"}
                            for path in outputs],
                "acceptance_criteria": [{
                    "ordinal": 1, "criterion": str(meta.get("acceptance") or ""),
                    # IntentRepository stores this complete mapping as the
                    # evidence policy, so the required kind belongs here.
                    **({"evidence_kind": "operator_control_validation"}
                       if card.identifier == "SAWM-000" else {}),
                }],
                "validation_commands": validations,
                "definition_cid": _identity(definition), "definition": definition,
                "board_namespace": NAMESPACE, "plan_revision": REVISION,
                "completion_mode": str(meta.get("completion mode") or "automatic"),
                "protected_paths": str(meta.get("protected paths") or ""),
                "repository_owner": str(meta.get("owning repository") or ""),
                "provider_role": str(meta.get("provider role") or ""),
                "rollout_mode": str(meta.get("rollout mode") or ""),
            }
        )

    program_definition = {
        "schema": "sawm/program-definition@1", "board_namespace": NAMESPACE,
        "plan_revision": REVISION, "source_binding_cid": source["source_binding_cid"],
        "plan_cid": plan_cid, "goal_cids": [goal_cids[item.identifier] for item in goals],
        "task_cids": [task_cids[item.identifier] for item in tasks],
    }
    program_definition_cid = _identity(program_definition)
    for row in (*objective_rows, *taskboard):
        row["program_definition_cid"] = program_definition_cid
    return {
        "schema": "sawm/program-population@1", "board_namespace": NAMESPACE,
        "plan_revision": REVISION, "program_definition_cid": program_definition_cid,
        "program_definition": program_definition, "repository_tree_id": source["source_binding_cid"],
        "source_binding": source, "plan_root_cid": plan_cid,
        # DatabaseTaskSource consumes `objectives` before `goals`; all 29 goal
        # records are intentionally supplied in this authoritative key.
        "objectives": objective_rows, "goal_edges": goal_edges,
        "plans": [{"plan_cid": plan_cid, "plan_alias": REVISION,
                   "goal_cid": goal_cids[ROOT_GOAL], "status": "active", **plan_definition}],
        "taskboard": taskboard,
    }


def _verify_store(path: Path, population: Mapping[str, Any], *, require_operator_complete: bool) -> dict[str, Any]:
    # Direct DuckDB verification is an offline operation.  Once the Quack
    # state owner is live, every read and write must pass through that owner;
    # opening the file here would violate the single-owner authority boundary.
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import discover_live_quack_endpoint
    discovery = discover_live_quack_endpoint(path)
    owner_marker = path.with_name(f".{path.name}.state-owner.json")
    if discovery.uri or owner_marker.exists():
        detail = discovery.reason or "active_owner_marker"
        raise MaterializationError(
            f"offline store verification refused while Quack ownership may be active: {detail}"
        )
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import verify_datasets_authoritative_operational_schema
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource
    schema = verify_datasets_authoritative_operational_schema(path)
    if schema.get("valid") is not True:
        raise MigrationRequired("existing database does not verify as datasets-authoritative operational schema")
    source = DatabaseTaskSource(path, install_schema=False,
                                repository_tree_id=str(population["repository_tree_id"]),
                                plan_root_cid=str(population["plan_root_cid"]))
    try:
        snap = source.snapshot()
        expected_tasks = list(population["taskboard"])
        expected_goals = list(population["objectives"])
        if snap.task_count != 45 or snap.goal_count != 29 or snap.plan_root_cid != population["plan_root_cid"]:
            raise MigrationRequired(f"population counts/root conflict: tasks={snap.task_count} goals={snap.goal_count} root={snap.plan_root_cid}")
        status_by_alias: dict[str, str] = {}
        for expected in expected_tasks:
            observed = source.get_task(str(expected["task_cid"]))
            expected_outputs = [str(item["declared_path"]) for item in expected["outputs"]]
            expected_acceptance = [str(item["criterion"]) for item in expected["acceptance_criteria"]]
            expected_validations = [[str(command)] for command in expected["validation_commands"]]
            if (
                observed is None
                or observed.task_alias != expected["task_id"]
                or observed.body.get("definition_cid") != expected["definition_cid"]
                or sorted(observed.dependencies) != sorted(expected["depends_on"])
                or [str((item.get("effect") or {}).get("declared_path")) for item in observed.outputs] != expected_outputs
                or [str(item.get("criterion")) for item in observed.acceptance] != expected_acceptance
                or [list(item.get("argv") or ()) for item in observed.validations] != expected_validations
            ):
                raise MigrationRequired(f"task definition conflict: {expected['task_id']}")
            status_by_alias[observed.task_alias] = observed.status
        for expected in expected_goals:
            observed = source.get_goal(str(expected["goal_cid"]))
            if observed is None or str(observed.get("goal_alias")) != expected["goal_id"] or (observed.get("body") or {}).get("definition_cid") != expected["definition_cid"]:
                raise MigrationRequired(f"goal definition conflict: {expected['goal_id']}")
        if require_operator_complete and status_by_alias.get("SAWM-000") not in {"completed", "complete", "done"}:
            raise MigrationRequired("SAWM-000 lacks an admitted completion CAS")
        # Replay on a private copy so an exact existing authority remains a
        # true no-op.  Rebuild itself is the landed deterministic verifier.
        replay_copy = path.with_name(path.name + f".replay-check.{os.getpid()}")
        if replay_copy.exists():
            raise MaterializationError(f"preserved replay-check file requires inspection: {replay_copy}")
        shutil.copyfile(path, replay_copy)
        replay = DatabaseTaskSource(replay_copy, install_schema=False,
                                    repository_tree_id=str(population["repository_tree_id"]),
                                    plan_root_cid=str(population["plan_root_cid"]))
        try:
            projection_matches = replay.projection_matches_events()
        finally:
            replay.close()
            replay_copy.unlink(missing_ok=True)
        if not projection_matches:
            raise MigrationRequired("event replay projection differs from the accepted projection")
        return {"valid": True, "task_count": snap.task_count, "goal_count": snap.goal_count,
                "projection_cid": snap.projection_cid, "event_watermark": snap.event_cursor,
                "projection_matches_events": True, "statuses": status_by_alias}
    finally:
        source.close()


def _validator_report(root: Path, script: str) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(root / script), "--check-all", "--repo-root", str(root)],
        cwd=root, check=False, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, timeout=180,
        env={**os.environ, "PYTHONPATH": os.pathsep.join((str(root / "ipfs_datasets_py"), str(root / "ipfs_kit_py"), str(root)))},
    )
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MaterializationError(f"{script} did not emit deterministic JSON: {exc}; stderr={result.stderr[-1000:]}") from exc
    if result.returncode or report.get("valid") is not True:
        raise MaterializationError(f"{script} failed current-tree validation: {report.get('errors')}")
    return report


def _ducklake_projection(root: Path, config: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
    policy = config.get("ducklake_history_projection") or {}
    receipt_path = root / str(policy.get("receipt_path"))
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "schema": "sawm/ducklake-history-projection-receipt@1", "authority": False,
        "scheduling_prerequisite": False, "completion_prerequisite": False,
        "install_attempted": False, "network_used": False,
    }
    connection = None
    try:
        import duckdb
        from ipfs_accelerate_py.agent_supervisor.integrations.ducklake_history_projection import project_history
        connection = duckdb.connect(":memory:")
        connection.execute("SET autoinstall_known_extensions = false")
        connection.execute("SET autoload_known_extensions = false")
        row = connection.execute("SELECT installed, install_path FROM duckdb_extensions() WHERE extension_name = 'ducklake'").fetchone()
        if not row or not bool(row[0]) or not str(row[1] or ""):
            raise RuntimeError("ducklake_extension_not_locally_installed")
        connection.execute("LOAD ducklake")  # local LOAD only; INSTALL is forbidden
        catalog = (root / str(policy["catalog_path"])).resolve()
        data = (root / str(policy["data_path"])).resolve()
        catalog.parent.mkdir(parents=True, exist_ok=True); data.mkdir(parents=True, exist_ok=True)
        def literal(value: Path) -> str:
            return "'" + str(value).replace("'", "''") + "'"
        connection.execute(
            "ATTACH " + literal(Path("ducklake:" + str(catalog)))
            + " AS sawm_history (DATA_PATH " + literal(data) + ")"
        )
        connection.execute("CREATE TABLE IF NOT EXISTS sawm_history.control_history (program_definition_cid VARCHAR, projection_cid VARCHAR, authoritative BOOLEAN)")
        connection.execute("INSERT INTO sawm_history.control_history VALUES (?, ?, false)", [record["program_definition_cid"], record["projection_cid"]])
        projection = dict(project_history({"receipt": record}))
        receipt.update({"status": "available", "typed_unavailability": None,
                        "projection": projection, "catalog_path": str(catalog), "data_path": str(data)})
    except Exception as exc:
        receipt.update({"status": "typed_unavailability", "typed_unavailability": type(exc).__name__ + ": " + str(exc), "projection": None})
    finally:
        if connection is not None:
            connection.close()
    receipt["receipt_cid"] = _identity(receipt)
    temporary = receipt_path.with_name(receipt_path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(_canonical(receipt) + b"\n")
    os.replace(temporary, receipt_path)
    return receipt


def materialize(repo_root: Path | str = REPO_ROOT, config_path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = _load_json(config_file)
    population = build_population(root)
    target = root / str(config["database_program"]["store_id"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        verified = _verify_store(target, population, require_operator_complete=True)
        return {"schema": SCHEMA, "valid": True, "action": "verified_existing_noop",
                "migration_required": False, "database_path": str(target),
                "program_definition_cid": population["program_definition_cid"], **verified}

    dependency = _validator_report(root, "scripts/validate_semantic_addressed_world_model_dependencies.py")
    board = _validator_report(root, "scripts/validate_semantic_addressed_world_model_board.py")
    validation_digest = _identity({"dependency": dependency, "board": board,
                                   "program_definition_cid": population["program_definition_cid"]})
    preserved_stages = sorted(target.parent.glob(target.name + ".installing.*"))
    if preserved_stages:
        raise MaterializationError(
            "preserved prior staging attempt requires inspection: "
            + ", ".join(str(item) for item in preserved_stages)
        )
    stage = target.with_name(target.name + f".installing.{os.getpid()}.{validation_digest[-12:]}")
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import install_datasets_authoritative_operational_schema
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource
    install_datasets_authoritative_operational_schema(stage, application_version="0.0.45",
        tool_version="SAWM-PLAN-R2", owner_id="sawm-r2-materializer")
    source = DatabaseTaskSource(stage, install_schema=False,
                                repository_tree_id=str(population["repository_tree_id"]),
                                plan_root_cid=str(population["plan_root_cid"]),
                                owner_id="sawm-r2-materializer")
    try:
        materialization = dict(source.materialize(population,
            repository_tree_id=str(population["repository_tree_id"]),
            plan_root_cid=str(population["plan_root_cid"])))
        operator = source.get_task("SAWM-000")
        if operator is None or operator.status != "ready":
            raise MaterializationError("SAWM-000 was not born ready before evidence/CAS")
        evidence = source.record_evidence(task_cid=operator.task_cid,
            evidence_kind="operator_control_validation", digest=validation_digest,
            body={"program_definition_cid": population["program_definition_cid"],
                  "dependency_valid": True, "board_valid": True,
                  "markdown_completion_authority": False})
        source.record_validation_result(task_cid=operator.task_cid, outcome="passed",
            evidence_digest=validation_digest,
            argv=["current-tree SAWM dependency and board validators"],
            attempt_id="SAWM-000-operator-bootstrap",
            body={"evidence_event_id": evidence.event_id})
        cas = source.compare_and_set_status(operator.task_cid, operator.revision, "completed",
            receipt={"schema": "sawm/operator-bootstrap-completion@1",
                     "program_definition_cid": population["program_definition_cid"],
                     "validation_digest": validation_digest,
                     "worker_self_approval": False, "markdown_completion_authority": False},
            evidence_digests=[validation_digest])
        if not cas.changed or cas.task.status != "completed":
            raise MaterializationError("SAWM-000 completion CAS did not advance")
        if not source.projection_matches_events():
            raise MaterializationError("event replay projection differs before publication")
    finally:
        source.close()
    verified = _verify_store(stage, population, require_operator_complete=True)
    history = _ducklake_projection(root, config, {"program_definition_cid": population["program_definition_cid"],
                                                   "projection_cid": verified["projection_cid"]})
    lock = target.with_name(target.name + ".publish.lock")
    fd = None
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        if target.exists():
            raise MigrationRequired("another writer published a control store; inspect it append-only")
        os.link(stage, target)
        os.unlink(stage)
    finally:
        if fd is not None:
            os.close(fd)
            lock.unlink(missing_ok=True)
    report = {"schema": SCHEMA, "valid": True, "action": "materialized",
            "migration_required": False, "database_path": str(target),
            "program_definition_cid": population["program_definition_cid"],
            "validation_digest": validation_digest, "materialization": materialization,
            "ducklake_history": history, **verified}
    receipt = {
        "schema": "sawm/non-authoritative-materialization-receipt@1",
        "authoritative": False, "database_is_authority": True,
        "program_definition_cid": population["program_definition_cid"],
        "projection_cid": verified["projection_cid"],
        "validation_digest": validation_digest,
        "database_path": str(target.relative_to(root)),
    }
    receipt["receipt_cid"] = _identity(receipt)
    receipt_path = target.parent / "materialization-receipt.json"
    receipt_stage = receipt_path.with_name(receipt_path.name + f".tmp.{os.getpid()}")
    receipt_stage.write_bytes(_canonical(receipt) + b"\n")
    if receipt_path.exists():
        receipt_stage.unlink(missing_ok=True)
        raise MigrationRequired("a materialization receipt already exists and was not rewritten")
    os.link(receipt_stage, receipt_path)
    receipt_stage.unlink()
    return {**report, "receipt": receipt}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("materialize", "render", "check"), default="materialize")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        population = build_population(args.repo_root)
        if args.action == "render":
            report: dict[str, Any] = population
        elif args.action == "check":
            config = _load_json(args.config if args.config.is_absolute() else args.repo_root / args.config)
            path = Path(args.repo_root).resolve() / str(config["database_program"]["store_id"])
            report = {"schema": SCHEMA, "valid": True, "action": "checked",
                      "database_path": str(path), "program_definition_cid": population["program_definition_cid"],
                      **_verify_store(path, population, require_operator_complete=True)}
        else:
            report = materialize(args.repo_root, args.config)
    except MigrationRequired as exc:
        report = {"schema": SCHEMA, "valid": False, "action": "migration_required",
                  "migration_required": True, "error": str(exc)}
    except Exception as exc:
        report = {"schema": SCHEMA, "valid": False, "action": "failed",
                  "migration_required": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("valid") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
