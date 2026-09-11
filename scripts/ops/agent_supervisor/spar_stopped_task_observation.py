"""Observe a fenced closed-store copy through the accepted SPAR native lineage.

The worker never opens the canonical task store, installs schemas, grants an
owner, mutates a task, or admits a source amendment. It runs in a fresh process
so the accepted operator and its runtime cannot reuse candidate imports.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import runpy
import sys
import types

SCHEMA = "spar/stopped-task-store-observation@1"
MAX_BYTES = 16 * 1024 * 1024


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def observe(root, config_path, copy_path, *, runtime_manifest, runtime_sha256, runtime_helper_sha256, probe=False):
    root, config_path = (Path(p).absolute() for p in (root, config_path))
    copy_path = None if copy_path is None else Path(copy_path).absolute()
    require(probe or copy_path is not None and not copy_path.is_relative_to(root),
            "observation copy is inside native checkout")
    require(not any(name == "ipfs_accelerate_py" or name.startswith("ipfs_accelerate_py.")
                    for name in sys.modules), "candidate runtime was already imported")
    sys.path.insert(0, str(root))
    helper_path = Path(__file__).with_name("spar_capture_runtime.py")
    helper_bytes = helper_path.read_bytes()
    require(hashlib.sha256(helper_bytes).hexdigest() == runtime_helper_sha256,
            "closed observation runtime helper differs")
    helper = types.ModuleType("_spar_stopped_runtime_helper")
    helper.__file__ = str(helper_path)
    sys.modules[helper.__name__] = helper
    exec(compile(helper_bytes, str(helper_path), "exec"), helper.__dict__)
    manifest_bytes = Path(runtime_manifest).read_bytes()
    require(hashlib.sha256(manifest_bytes).hexdigest() == runtime_sha256,
            "closed observation runtime manifest differs")
    runtime = helper.BoundDuckDBRuntime(json.loads(manifest_bytes))
    runtime.load()
    native = runpy.run_path(str(root / "scripts/materialize_semantic_preserving_remodularization_program.py"),
                           run_name="_spar_stopped_native_observation")
    board, config = native["_load_config"](config_path)
    paths = native["_runtime_paths"](board)
    require(probe or copy_path != Path(paths["database"]), "canonical task observation refused")
    head, tree = native["_assert_clean_current_tree"](config)
    forest = native["_source_forest"](config, head=head)
    source_binding = {"head": head, "tree": tree, "forest": forest,
                      "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}
    bootstrap = native["_json_object"](paths["bootstrap_receipt"])
    require(bootstrap.get("bootstrap_receipt_id") == native["_identity"](
        {k: v for k, v in bootstrap.items() if k != "bootstrap_receipt_id"}), "bootstrap seal differs")
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import open_duckdb_connection
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource
    from ipfs_accelerate_py.agent_supervisor.task_sources.closeout_snapshot import capture_closeout_facts
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import connect_duckdb_with_policy
    import duckdb
    connection = connect_duckdb_with_policy(duckdb, ":memory:")
    try:
        require(connection.execute("SELECT 1").fetchone()[0] == 1, "closed runtime SQL unavailable")
    finally:
        connection.close()
    runtime.require_current()
    dependency = {"manifest_sha256": runtime_sha256, "helper_sha256": runtime_helper_sha256,
                  "isolated": bool(sys.flags.isolated), "duckdb_version": duckdb.__version__}
    require(dependency["isolated"], "closed observation requires isolated interpreter")
    if probe:
        require(native["_assert_clean_current_tree"](config) == (head, tree), "native source changed")
        return {"schema": "spar/stopped-task-runtime-probe@1", "source": source_binding,
                "runtime": dependency, "canonical_database_opened": False}
    # WAL recovery, if needed, is confined to this separately preserved copy.
    with open_duckdb_connection(copy_path, prefer_quack=False) as connection:
        connection.execute("BEGIN TRANSACTION")
        intent = IntentRepository(database_path=copy_path, bound_connection=connection,
                                  owner_id="spar-stopped-copy:observation", install_schema=False)
        source = DatabaseTaskSource(intent=intent, repository_tree_id=bootstrap["repository_tree_id"],
                                    plan_root_cid=bootstrap["plan_root_cid"], install_schema=False)
        try:
            tasks = tuple(native["_database_tasks"](source))
            snapshot = source.snapshot()
            from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import (
                EXECUTION_ROUTE_RECEIPT_FIELDS, POST_MERGE_RETRY_RECOVERY_OPERATIONS,
                VIRGIN_TASK_TRANSFER_RECEIPT_FIELDS,
            )
            histories = {}
            # Match the accepted operator's read-only history selection. Passing
            # SPAR-000's operator completion as retry history changes semantics.
            for task in tasks:
                receipt = task.body.get("completion_receipt") if isinstance(task.body, Mapping) else None
                if (task.task_alias != "SPAR-000" and int(task.revision) > 1 and task.status == "retrying"
                    and isinstance(receipt, Mapping)
                    and receipt.get("operation") in POST_MERGE_RETRY_RECOVERY_OPERATIONS
                    and not set(receipt).intersection(EXECUTION_ROUTE_RECEIPT_FIELDS | VIRGIN_TASK_TRANSFER_RECEIPT_FIELDS)):
                    histories[task.task_cid] = source.task_revision_history_projection(task.task_cid)["revisions"]
            route = native["_resume_execution_route_policy"](
                bootstrap=bootstrap, snapshot=snapshot, tasks=tasks, histories_by_task=histories)
            forest_receipt = native["_launch_source_forest_receipt"](
                source_head=head, repository_tree=tree, source_forest=forest)
            _candidate, _replay, plan, exact_tasks = native["_launch_source_amendment_context"](
                source=source, board=board, config=config, paths=paths, source_head=head,
                repository_tree=tree, source_forest_receipt=forest_receipt, execution_route_policy=route)
            facts = capture_closeout_facts(connection)
            def rows(query):
                cursor = connection.execute(query)
                from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import _result_columns
                names = _result_columns(cursor)
                values = cursor.fetchall()
                require(len(values) <= 512, "closed identity population exceeds bound")
                return [dict(row) if isinstance(row, Mapping) else dict(zip(names, row)) for row in values]
            generations = rows("SELECT generation,schema_revision,fence_epoch,revision,database_uuid,birth_id FROM store_generations ORDER BY generation DESC LIMIT 2")
            servers = rows("SELECT server_id,store_id,database_uuid,process_birth_id,schema_revision,generation,status,stopped_at FROM state_servers ORDER BY generation DESC LIMIT 2")
            epochs = rows("SELECT server_id,epoch,fence_epoch,ended_at FROM server_epochs ORDER BY epoch DESC LIMIT 2")
            value = {"schema": SCHEMA, "source": source_binding, "bootstrap": bootstrap,
                     "runtime": dependency,
                     "plan": native["_plain_json"](plan),
                     "plan_revisions": [native["_plain_json"](item) for item in source.plans.list_revisions(bootstrap["plan_root_cid"])],
                     "task_cids": [task.task_cid for task in exact_tasks], "facts": facts,
                     "store_generations": generations, "state_servers": servers, "server_epochs": epochs,
                     "completion_authority": False, "callback_settled": False,
                     "launch_amendment_admitted": False, "canonical_database_opened": False}
            connection.execute("ROLLBACK")
        finally:
            source.close()
    require(native["_assert_clean_current_tree"](config) == (head, tree), "native source changed")
    require(native["_json_object"](paths["bootstrap_receipt"]) == bootstrap, "bootstrap changed")
    runtime.require_current()
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode()
    require(len(raw) <= MAX_BYTES, "closed observation exceeds bound")
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--copy")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--runtime-helper-sha256", required=True)
    args = parser.parse_args()
    try:
        value = observe(args.root, args.config, args.copy, runtime_manifest=args.runtime_manifest,
            runtime_sha256=args.runtime_sha256, runtime_helper_sha256=args.runtime_helper_sha256, probe=args.probe)
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "detail": str(exc)[:256]}))
        return 1
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
