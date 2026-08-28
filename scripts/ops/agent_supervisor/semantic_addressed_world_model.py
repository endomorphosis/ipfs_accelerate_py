#!/usr/bin/env python3
"""Thin operator facade for the existing SAWM supervisor authorities.

Import and ``--help`` are side-effect free.  All operational work is delegated
to the landed validators, DatabaseTaskSource, Quack owner, provider router and
configured-board scheduler; this module is not a second agent framework.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "config/agent_supervisor_semantic_addressed_world_model_scheduler.json"


class OperatorError(RuntimeError):
    pass


def _load_script(relative: str, name: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise OperatorError(f"cannot load sealed operator script: {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise OperatorError("scheduler config must be an object")
    return value


def _emit(value: Mapping[str, Any]) -> int:
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value.get("valid", True) is True else 2


def _validator(relative: str, function: str) -> dict[str, Any]:
    module = _load_script(relative, "_sawm_operator_" + function)
    return dict(getattr(module, function)(REPO_ROOT))


def _materializer():
    return _load_script("scripts/materialize_semantic_addressed_world_model_program.py", "_sawm_operator_materializer")


def _quack_args(config: Mapping[str, Any], command: str) -> list[str]:
    owner = config["quack_owner"]
    return [
        "--database", str(REPO_ROOT / owner["database_path"]),
        "--state-dir", str(REPO_ROOT / owner["state_dir"]),
        "--host", str(owner["host"]), "--port", str(owner["port"]),
        "--store-id", str(owner["store_id"]),
        "--repository-id", str(owner["repository_id"]),
        "--secret-handle", str(owner["secret_handle"]), "--json", command,
    ]


def _start_quack(config: Mapping[str, Any]) -> int:
    owner = config["quack_owner"]
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import build_server
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import install_datasets_authoritative_operational_schema
    ops = _load_script("scripts/ops/agent_supervisor/quack_state_server.py", "_sawm_landed_quack_ops")
    server = build_server(
        database_path=REPO_ROOT / owner["database_path"],
        state_dir=REPO_ROOT / owner["state_dir"], host=str(owner["host"]),
        port=int(owner["port"]), repository_id=str(owner["repository_id"]),
        store_id=str(owner["store_id"]), allow_experimental=True,
        secret_handle=str(owner["secret_handle"]),
        # Critical authority boundary: never install the generic full schema.
        migrate=install_datasets_authoritative_operational_schema,
    )
    identity = server.start()
    print(json.dumps(identity.to_dict(), indent=2, sort_keys=True))
    sys.stdout.flush()
    result = ops._serve_until_stop(server)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _live_preflight(config: Mapping[str, Any], *, probe_provider: bool = True) -> dict[str, Any]:
    dependency = _validator("scripts/validate_semantic_addressed_world_model_dependencies.py", "validate_dependencies")
    board = _validator("scripts/validate_semantic_addressed_world_model_board.py", "validate_program")
    if dependency.get("valid") is not True or board.get("valid") is not True:
        raise OperatorError("sealed dependency or board validation failed")
    materializer = _materializer()
    population = materializer.build_population(REPO_ROOT)
    store = REPO_ROOT / config["database_program"]["store_id"]

    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        QUACK_ENDPOINT_ENV, QUACK_STORE_ID_ENV, QUACK_TOKEN_ENV,
        discover_live_quack_endpoint,
    )
    discovery = discover_live_quack_endpoint(store)
    expected_uri = str(config["database_program"]["quack_endpoint"])
    if not discovery.uri or discovery.uri != expected_uri or not discovery.token:
        raise OperatorError(f"exact live Quack owner unavailable: {discovery.reason}")
    # Resolve the opaque secret handle only into this coordinator environment.
    # No raw token is printed, put in argv, receipts or provider inputs.
    os.environ[QUACK_ENDPOINT_ENV] = discovery.uri
    os.environ[QUACK_STORE_ID_ENV] = str(store)
    os.environ[QUACK_TOKEN_ENV] = discovery.token
    os.environ["SAWM_QUACK_TOKEN"] = discovery.token
    from ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source import DatabaseTaskSource
    live = DatabaseTaskSource(discovery.uri, install_schema=False,
                              repository_tree_id=population["repository_tree_id"],
                              plan_root_cid=population["plan_root_cid"],
                              owner_id="sawm-r2-live-preflight")
    try:
        live_snapshot = live.snapshot().to_dict()
        if live_snapshot["task_count"] != 45 or live_snapshot["goal_count"] != 29 or live_snapshot["plan_root_cid"] != population["plan_root_cid"]:
            raise OperatorError("live Quack snapshot differs from the exact program root/counts")
        statuses: dict[str, str] = {}
        for expected in population["taskboard"]:
            observed = live.get_task(expected["task_cid"])
            if (
                observed is None
                or observed.task_alias != expected["task_id"]
                or observed.body.get("definition_cid") != expected["definition_cid"]
                or sorted(observed.dependencies) != sorted(expected["depends_on"])
                or [(item.get("effect") or {}).get("declared_path") for item in observed.outputs]
                   != [item["declared_path"] for item in expected["outputs"]]
                or [item.get("criterion") for item in observed.acceptance]
                   != [item["criterion"] for item in expected["acceptance_criteria"]]
                or [list(item.get("argv") or ()) for item in observed.validations]
                   != [[command] for command in expected["validation_commands"]]
            ):
                raise OperatorError(f"live Quack task definition conflict: {expected['task_id']}")
            statuses[observed.task_alias] = observed.status
        for expected in population["objectives"]:
            observed = live.get_goal(expected["goal_cid"])
            if observed is None or observed.get("goal_alias") != expected["goal_id"] or (observed.get("body") or {}).get("definition_cid") != expected["definition_cid"]:
                raise OperatorError(f"live Quack goal definition conflict: {expected['goal_id']}")
        if statuses.get("SAWM-000") not in {"completed", "complete", "done"}:
            raise OperatorError("live Quack authority lacks the SAWM-000 completion CAS")
    finally:
        live.close()
    store_report = {
        "valid": True, "task_count": live_snapshot["task_count"],
        "goal_count": live_snapshot["goal_count"],
        "projection_cid": live_snapshot["projection_cid"],
        "event_cursor": live_snapshot["event_cursor"],
        "queried_through_live_quack_only": True,
        "direct_authoritative_file_opened": False,
        "statuses": statuses,
    }

    provider_report: dict[str, Any] = {"probed": False}
    if probe_provider:
        provider = config["provider"]
        from ipfs_accelerate_py.llm_router import probe_grok_codex_agent_route_readiness
        readiness = probe_grok_codex_agent_route_readiness(
            grok_model=str(provider["primary_model_id"]),
            codex_model=str(provider["fallback_model_id"]),
            codex_reasoning_effort=str(provider["fallback_reasoning_effort"]),
        )
        failure = readiness.failure_kind.value if readiness.failure_kind is not None else ""
        provider_report = {**dataclasses.asdict(readiness), "failure_kind": failure, "probed": True}
        if not readiness.effective_provider:
            raise OperatorError(f"ordered provider route unavailable: {readiness.reason_code}")
        if readiness.effective_provider == "codex" and (
            provider["fallback_trigger"] != "primary_quota_exhausted"
            or failure != "grok_quota_exhausted"
        ):
            raise OperatorError("Codex fallback is not admitted by the reviewed quota-only trigger")
    return {
        "schema": "sawm/live-control-preflight@1", "valid": True,
        "dependency_valid": True, "board_valid": True,
        "store": store_report,
        "quack": {"uri": discovery.uri, "source": discovery.source,
                  "reason": discovery.reason, "token_present": True,
                  "live_query": True, "task_count": live_snapshot["task_count"]},
        "provider": provider_report,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in (
        "validate-dependencies", "validate-board", "materialize", "render", "check",
        "quack-start", "quack-status", "quack-ready", "quack-stop", "preflight", "dry-run",
    ):
        sub.add_parser(command)
    launch = sub.add_parser("launch")
    launch.add_argument("--foreground", action="store_true")
    launch.add_argument("--duration-seconds", type=float, default=float("inf"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
        config = _config(config_path)
        if args.command == "validate-dependencies":
            return _emit(_validator("scripts/validate_semantic_addressed_world_model_dependencies.py", "validate_dependencies"))
        if args.command == "validate-board":
            return _emit(_validator("scripts/validate_semantic_addressed_world_model_board.py", "validate_program"))
        if args.command in {"materialize", "render", "check"}:
            materializer = _materializer()
            if args.command == "render":
                return _emit({"valid": True, **materializer.build_population(REPO_ROOT)})
            if args.command == "check":
                population = materializer.build_population(REPO_ROOT)
                store = REPO_ROOT / config["database_program"]["store_id"]
                from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import discover_live_quack_endpoint
                discovery = discover_live_quack_endpoint(store)
                if discovery.uri:
                    return _emit({"action": "checked_live", **_live_preflight(config, probe_provider=False)})
                return _emit({"valid": True, "action": "checked", **materializer._verify_store(store, population, require_operator_complete=True)})
            return _emit(materializer.materialize(REPO_ROOT, config_path))
        if args.command == "quack-start":
            return _start_quack(config)
        if args.command in {"quack-status", "quack-ready", "quack-stop"}:
            ops = _load_script("scripts/ops/agent_supervisor/quack_state_server.py", "_sawm_landed_quack_ops")
            return int(ops.main(_quack_args(config, args.command.removeprefix("quack-"))))

        live = _live_preflight(config)
        from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import main as scheduler_main
        scheduler_args = ["--repo-root", str(REPO_ROOT), "--config", str(config_path)]
        if args.command == "preflight":
            result = int(scheduler_main([*scheduler_args, "preflight"]))
        else:
            launch_args = [*scheduler_args, "launch", "--implement"]
            if args.command == "dry-run":
                launch_args.append("--dry-run")
            else:
                if args.foreground:
                    launch_args.append("--foreground")
                if math.isfinite(args.duration_seconds):
                    launch_args.extend(["--duration-seconds", str(args.duration_seconds)])
            result = int(scheduler_main(launch_args))
        if result:
            return result
        # Only secret-free preflight facts are emitted by this facade.
        print(json.dumps({"schema": "sawm/operator-delegation@1", "valid": True,
                          "command": args.command, "live_preflight": live}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        return _emit({"schema": "sawm/operator-error@1", "valid": False,
                      "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    raise SystemExit(main())
