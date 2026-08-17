"""Narrow local zk-seal CLI (IPS-043).

Commands: full, incremental, verify, plan, explain-reuse, explain-invalidation,
benchmark, cache-status, force-full.

The CLI is local-only: no service, GUI, agent framework, auto-install, or
network.  Machine-readable JSON is the default.  Simulated evidence cannot
become a production seal.  Missing optional capabilities are typed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

CLI_EVIDENCE = "ips/cli@1"
CLI_SCHEMA = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/zk-seal-cli@1"
)

COMMANDS: tuple[str, ...] = (
    "full",
    "incremental",
    "verify",
    "plan",
    "explain-reuse",
    "explain-invalidation",
    "benchmark",
    "cache-status",
    "force-full",
)

CLOSED_STATUSES: frozenset[str] = frozenset(
    {
        "sealed_full",
        "sealed_incremental",
        "verification_failed",
        "proof_failed",
        "unknown",
        "timeout",
        "unavailable",
        "stale_parent",
        "invalid_cache",
        "incomplete_manifest",
        "full_reproof_required",
        "cancelled",
        "simulated_only",
        "ok",
        "planned",
        "explained",
        "compared",
    }
)


class CliError(ValueError):
    """Typed CLI failure."""

    def __init__(self, status: str, message: str, *, details: Mapping[str, Any] | None = None):
        if status not in CLOSED_STATUSES:
            status = "unknown"
        self.status = status
        self.message = message
        self.details = dict(details or {})
        super().__init__(message)


def _emit(payload: Mapping[str, Any], *, stream: Any = None) -> int:
    target = sys.stdout if stream is None else stream
    target.write(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True))
    target.write("\n")
    return 0 if payload.get("ok") is True else 2


def _envelope(
    *,
    command: str,
    status: str,
    result: Any = None,
    error: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ok = status in {"ok", "sealed_full", "sealed_incremental", "planned", "explained", "compared"}
    return {
        "schema": CLI_SCHEMA,
        "evidence": CLI_EVIDENCE,
        "command": command,
        "status": status,
        "ok": ok,
        "result": result,
        "error": error,
        "details": dict(details or {}),
        "side_effects": {
            "network": False,
            "process_spawn": False,
            "key_generation": False,
            "state_mutation": False,
        },
    }


def _load_json(path: str | None) -> Any:
    if not path:
        raise CliError("unknown", "required JSON path is missing")
    file_path = Path(path)
    if not file_path.is_file():
        raise CliError("unavailable", f"input not found: {path}")
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CliError("unknown", f"invalid JSON: {path}: {exc}") from exc


def _canonical(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_canonical") and callable(value.to_canonical):
        return value.to_canonical()
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _reject_simulated(payload: Any, *, command: str) -> None:
    text = json.dumps(payload, default=str).lower() if payload is not None else ""
    if "simulated" in text and command in {"full", "incremental", "force-full"}:
        raise CliError(
            "simulated_only",
            "production seals reject simulated evidence",
            details={"command": command},
        )


def _cmd_full(args: argparse.Namespace) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.full_checkpoint import (
        create_full_checkpoint,
    )

    state = _load_json(args.state)
    policy = _load_json(args.policy) if args.policy else {}
    units = _load_json(args.units) if args.units else []
    _reject_simulated({"state": state, "policy": policy, "units": units}, command="full")
    seal = create_full_checkpoint(state, policy, units=units)
    payload = _canonical(seal)
    status = "sealed_full"
    if isinstance(payload, Mapping) and payload.get("accepted") is False:
        status = str(payload.get("reason") or "proof_failed")
        if status not in CLOSED_STATUSES:
            status = "proof_failed"
    return _envelope(command="full", status=status, result=payload)


def _cmd_incremental(args: argparse.Namespace) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.executor import (
        execute_incremental_plan,
    )
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.planner import (
        create_incremental_plan,
    )

    if not args.parent:
        raise CliError("stale_parent", "incremental requires --parent")
    parent = _load_json(args.parent)
    old = args.old or args.state
    new = args.new or args.state
    if not old or not new:
        raise CliError("unknown", "incremental requires --old/--new or --state")
    _reject_simulated({"parent": parent}, command="incremental")
    plan = create_incremental_plan(
        parent,
        _load_json(old),
        _load_json(new),
        _load_json(args.policy) if args.policy else {},
        units=_load_json(args.units) if args.units else (),
    )
    result = execute_incremental_plan(plan)
    payload = {"plan": _canonical(plan), "execution": _canonical(result)}
    status = "sealed_incremental" if getattr(result, "succeeded", False) else "proof_failed"
    if getattr(result, "outcome", None) is not None:
        outcome = getattr(result.outcome, "value", str(result.outcome))
        if outcome in {"cancelled", "timeout", "unavailable"}:
            status = outcome
    return _envelope(command="incremental", status=status, result=payload)


def _cmd_verify(args: argparse.Namespace) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.verification import (
        verify_seal,
    )

    seal = _load_json(args.seal)
    keys = _load_json(args.keys) if args.keys else None
    policy = _load_json(args.policy) if args.policy else None
    result = verify_seal(seal, keys, policy)
    payload = _canonical(result)
    accepted = bool(getattr(result, "accepted", False))
    status = "ok" if accepted else "verification_failed"
    return _envelope(command="verify", status=status, result=payload)


def _cmd_plan(args: argparse.Namespace) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.planner import (
        create_incremental_plan,
    )

    if not args.parent:
        raise CliError("stale_parent", "plan requires --parent")
    old = args.old or args.state
    new = args.new or args.state
    if not old or not new:
        raise CliError("unknown", "plan requires --old/--new or --state")
    plan = create_incremental_plan(
        _load_json(args.parent),
        _load_json(old),
        _load_json(new),
        _load_json(args.policy) if args.policy else {},
        units=_load_json(args.units) if args.units else (),
    )
    return _envelope(command="plan", status="planned", result=_canonical(plan))


def _cmd_explain_reuse(args: argparse.Namespace) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.explanations import (
        explain_reuse,
    )

    if not args.unit:
        raise CliError("unknown", "explain-reuse requires --unit")
    result = explain_reuse(_load_json(args.seal), args.unit)
    return _envelope(command="explain-reuse", status="explained", result=_canonical(result))


def _cmd_explain_invalidation(args: argparse.Namespace) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.explanations import (
        explain_invalidation,
    )

    if not args.unit:
        raise CliError("unknown", "explain-invalidation requires --unit")
    source = args.plan or args.seal
    if not source:
        raise CliError("unknown", "explain-invalidation requires --plan or --seal")
    result = explain_invalidation(_load_json(source), args.unit)
    return _envelope(
        command="explain-invalidation",
        status="explained",
        result=_canonical(result),
    )


def _cmd_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.metrics import (
        ProofCostRecord,
        compare_costs,
    )

    if not args.full or not args.incremental:
        raise CliError(
            "unavailable",
            "benchmark requires measured --full and --incremental cost records",
        )
    full_raw = _load_json(args.full)
    inc_raw = _load_json(args.incremental)
    if not isinstance(full_raw, Mapping) or not isinstance(inc_raw, Mapping):
        raise CliError("unknown", "benchmark inputs must be JSON objects")
    if "schema" not in full_raw or "schema" not in inc_raw:
        raise CliError("unavailable", "benchmark records are not ProofCostRecord objects")
    # Reconstruct is not required; compare_costs needs ProofCostRecord instances.
    # If the payload already came from snapshot().to_canonical(), typed compare
    # is unavailable without a loader — report that honestly.
    raise CliError(
        "unavailable",
        "live benchmark reconstruction from JSON is not implemented; "
        "use compare_full_and_incremental or ProofMetricsCollector snapshots",
        details={"capability": "json_cost_record_loader"},
    )


def _cmd_cache_status(args: argparse.Namespace) -> dict[str, Any]:
    del args
    return _envelope(
        command="cache-status",
        status="unavailable",
        error="kit cache adapter is not bound in this CLI process",
        details={"capability": "proof_seal_store_cache_index", "typed": True},
    )


def _cmd_force_full(args: argparse.Namespace) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.planner import (
        create_incremental_plan,
    )

    old = args.old or args.state
    new = args.new or args.state
    if not old or not new:
        raise CliError("unknown", "force-full requires --old/--new or --state")
    payload_units = _load_json(args.units) if args.units else ()
    _reject_simulated({"units": payload_units}, command="force-full")
    plan = create_incremental_plan(
        None,
        _load_json(old),
        _load_json(new),
        {"full_fallback_required": True},
        units=payload_units,
        full_fallback_required=True,
    )
    return _envelope(
        command="force-full",
        status="full_reproof_required",
        result=_canonical(plan),
    )


_HANDLERS: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
    "full": _cmd_full,
    "incremental": _cmd_incremental,
    "verify": _cmd_verify,
    "plan": _cmd_plan,
    "explain-reuse": _cmd_explain_reuse,
    "explain-invalidation": _cmd_explain_invalidation,
    "benchmark": _cmd_benchmark,
    "cache-status": _cmd_cache_status,
    "force-full": _cmd_force_full,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zk-seal",
        description="Local IncrementalProofSealer CLI. No service, GUI, or auto-install.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        cmd = sub.add_parser(name, help=f"{name} operation")
        cmd.add_argument("--state", help="repository state JSON")
        cmd.add_argument("--policy", help="verification policy JSON")
        cmd.add_argument("--units", help="units JSON array")
        cmd.add_argument("--parent", help="parent seal JSON")
        cmd.add_argument("--old", help="old repository state JSON")
        cmd.add_argument("--new", help="new repository state JSON")
        cmd.add_argument("--seal", help="seal JSON")
        cmd.add_argument("--keys", help="trusted keys / policy JSON")
        cmd.add_argument("--plan", help="plan JSON")
        cmd.add_argument("--unit", help="proof unit id")
        cmd.add_argument("--full", help="full cost record JSON")
        cmd.add_argument("--incremental", dest="incremental", help="incremental cost record JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    command = str(args.command)
    try:
        payload = _HANDLERS[command](args)
    except CliError as exc:
        payload = _envelope(
            command=command,
            status=exc.status,
            error=exc.message,
            details=exc.details,
        )
    except Exception as exc:  # noqa: BLE001 - CLI must stay typed
        payload = _envelope(
            command=command,
            status="unknown",
            error=str(exc),
        )
    return _emit(payload)


if __name__ == "__main__":
    raise SystemExit(main())
