"""Narrowly scoped ``zk-seal`` CLI for IncrementalProofSealer (IPS-043).

Operations (closed freeze)
--------------------------
``full``, ``incremental``, ``verify``, ``plan``, ``explain-reuse``,
``explain-invalidation``, ``benchmark``, ``cache-status``, ``force-full``.

Cold ``--help`` / import performs no process spawn, network I/O, key
generation, or user-state mutation.  Heavy sealing modules load only when a
command actually executes.  Outputs are machine-readable JSON envelopes with
stable status/error fields.  Proving keys and witness material are never
emitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

# Keep module-level imports cold: stdlib only.  Public sealing APIs resolve
# lazily inside command handlers.

CLI_SUBSET = "ips/cli@1"
CLI_SCHEMA = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
    "zk-seal-cli-result@1"
)
CLI_PROG = "zk-seal"

CLI_OPERATIONS: tuple[str, ...] = (
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

_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "proving_key",
        "proving_key_bytes",
        "witness",
        "witness_bytes",
        "private_key",
        "secret",
        "trapdoor",
        "token",
        "password",
    }
)

EXIT_OK = 0
EXIT_OPERATION = 1
EXIT_USAGE = 2


class CliError(ValueError):
    """Fail-closed CLI contract or input error."""

    def __init__(self, message: str, *, status: str = "error") -> None:
        super().__init__(message)
        self.status = status


def _scrub(value: Any) -> Any:
    """Drop sensitive fields from nested JSON-like structures."""

    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name in _SENSITIVE_KEYS or any(
                token in name.lower()
                for token in ("proving_key", "witness", "private_key", "trapdoor")
            ):
                continue
            out[name] = _scrub(item)
        return out
    if isinstance(value, list):
        return [_scrub(item) for item in value]
    if isinstance(value, tuple):
        return [_scrub(item) for item in value]
    return value


def _envelope(
    *,
    command: str,
    status: str,
    result: Mapping[str, Any] | None = None,
    error: str | None = None,
    exit_code: int = EXIT_OK,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": CLI_SCHEMA,
        "evidence_subset": CLI_SUBSET,
        "prog": CLI_PROG,
        "command": command,
        "status": status,
        "exit_code": exit_code,
        "proving_key_exported": False,
        "witness_exported": False,
        "network_accessed": False,
        "keys_generated": False,
        "auto_install": False,
    }
    if result is not None:
        payload["result"] = _scrub(dict(result))
    if error is not None:
        payload["error"] = error
    return payload


def _load_json(path: str | Path | None, *, label: str) -> Any:
    if path is None:
        raise CliError(f"{label} path is required")
    target = Path(path)
    if not target.is_file():
        raise CliError(f"{label} file not found: {target}")
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise CliError(f"failed to read {label}: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise CliError(f"{label} is not valid JSON: {exc}") from exc


def _require_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CliError(f"{label} must be a JSON object")
    return dict(value)


def _require_list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise CliError(f"{label} must be a JSON array")
    return list(value)


def _canonical(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "to_canonical") and callable(obj.to_canonical):
        payload = obj.to_canonical()
        if isinstance(payload, Mapping):
            return dict(payload)
    if isinstance(obj, Mapping):
        return dict(obj)
    raise CliError(f"cannot canonicalize {type(obj).__name__}")


def _seal_status_value(obj: Any) -> str:
    if hasattr(obj, "seal_status"):
        status = obj.seal_status
        return str(getattr(status, "value", status))
    if isinstance(obj, Mapping):
        return str(obj.get("seal_status") or "")
    return ""


def _is_accepted_seal(status: str) -> bool:
    return status in {"sealed_full", "sealed_incremental"}


def build_parser() -> argparse.ArgumentParser:
    """Build the cold argparse graph (stdlib only, no sealing imports)."""

    parser = argparse.ArgumentParser(
        prog=CLI_PROG,
        description=(
            "Narrow IncrementalProofSealer CLI.  Emits JSON envelopes.  "
            "Cold help performs no process, network, key, or state side effects."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="print CLI evidence subset and operation freeze as JSON",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    full = sub.add_parser("full", help="create a full checkpoint seal")
    full.add_argument("--state", required=True, help="repository state JSON path")
    full.add_argument("--policy", required=True, help="verification policy JSON path")
    full.add_argument("--units", required=True, help="required unit evidence JSON array")
    full.add_argument(
        "--parent-seal-cid",
        default=None,
        help="optional parent seal CID (omit for genesis/first-state)",
    )
    full.add_argument(
        "--fallback-reason",
        action="append",
        default=[],
        dest="fallback_reasons",
        help="fallback reason (repeatable); defaults to first_state when omitted",
    )

    incremental = sub.add_parser(
        "incremental",
        help="plan and execute an incremental seal transition",
    )
    incremental.add_argument(
        "--parent",
        required=True,
        help="parent seal context JSON path",
    )
    incremental.add_argument("--old", required=True, help="old repository state JSON/CID")
    incremental.add_argument("--new", required=True, help="new repository state JSON/CID")
    incremental.add_argument(
        "--units",
        required=True,
        help="unit planning input JSON array",
    )
    incremental.add_argument(
        "--policy",
        default=None,
        help="optional verification policy / flag JSON path",
    )
    incremental.add_argument(
        "--resource-policy",
        default=None,
        help="optional resource policy JSON path",
    )
    incremental.add_argument(
        "--backend-available",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether a real backend is available (default: true)",
    )

    verify = sub.add_parser("verify", help="re-verify a seal under trusted keys")
    verify.add_argument("--seal", required=True, help="seal JSON path")
    verify.add_argument(
        "--trusted-keys",
        required=True,
        help="trusted key ids JSON array or trusted policy JSON object",
    )
    verify.add_argument("--policy", default=None, help="verification policy JSON path")
    verify.add_argument("--parent", default=None, help="optional parent seal JSON path")

    plan = sub.add_parser("plan", help="create an incremental-or-full plan")
    plan.add_argument("--parent", default=None, help="parent seal context JSON path")
    plan.add_argument("--old", required=True, help="old repository state JSON/CID")
    plan.add_argument("--new", required=True, help="new repository state JSON/CID")
    plan.add_argument("--units", required=True, help="unit planning input JSON array")
    plan.add_argument(
        "--policy",
        default=None,
        help="optional verification policy / flag JSON path",
    )

    explain_reuse = sub.add_parser(
        "explain-reuse",
        help="explain reuse disposition for one unit on a seal",
    )
    explain_reuse.add_argument("--seal", required=True, help="seal JSON path")
    explain_reuse.add_argument("--unit", required=True, help="proof unit id")

    explain_inv = sub.add_parser(
        "explain-invalidation",
        help="explain invalidation disposition for one planned unit",
    )
    explain_inv.add_argument(
        "--parent",
        default=None,
        help="parent seal context JSON path",
    )
    explain_inv.add_argument("--old", required=True, help="old repository state JSON/CID")
    explain_inv.add_argument("--new", required=True, help="new repository state JSON/CID")
    explain_inv.add_argument(
        "--units",
        required=True,
        help="unit planning input JSON array",
    )
    explain_inv.add_argument("--unit", required=True, help="proof unit id")
    explain_inv.add_argument(
        "--policy",
        default=None,
        help="optional verification policy / flag JSON path",
    )

    benchmark = sub.add_parser(
        "benchmark",
        help="compare full versus incremental estimated work",
    )
    benchmark.add_argument("--state", required=True, help="repository state JSON/CID")
    benchmark.add_argument("--parent", default=None, help="parent seal JSON path")
    benchmark.add_argument("--policy", default=None, help="verification policy JSON path")
    benchmark.add_argument(
        "--units",
        required=True,
        help="unit planning input JSON array",
    )
    benchmark.add_argument("--old", default=None, help="optional old repository state")

    cache_status = sub.add_parser(
        "cache-status",
        help="report planned reuse/reprove disposition without executing proofs",
    )
    cache_status.add_argument("--parent", default=None, help="parent seal context JSON")
    cache_status.add_argument("--old", required=True, help="old repository state JSON/CID")
    cache_status.add_argument("--new", required=True, help="new repository state JSON/CID")
    cache_status.add_argument(
        "--units",
        required=True,
        help="unit planning input JSON array",
    )
    cache_status.add_argument(
        "--policy",
        default=None,
        help="optional verification policy / flag JSON path",
    )

    force_full = sub.add_parser(
        "force-full",
        help="force a full-checkpoint plan (and optional seal) for a transition",
    )
    force_full.add_argument("--parent", default=None, help="parent seal context JSON")
    force_full.add_argument("--old", required=True, help="old repository state JSON/CID")
    force_full.add_argument("--new", required=True, help="new repository state JSON/CID")
    force_full.add_argument(
        "--units",
        required=True,
        help="unit planning input JSON array (planning) or unit evidence when --seal",
    )
    force_full.add_argument(
        "--policy",
        default=None,
        help="optional verification policy / flag JSON path",
    )
    force_full.add_argument(
        "--state",
        default=None,
        help="repository state for optional full seal construction",
    )
    force_full.add_argument(
        "--seal",
        action="store_true",
        help="also construct a full checkpoint from --state and unit evidence",
    )

    return parser


def _state_or_cid(raw: Any) -> Any:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, Mapping):
        return dict(raw)
    raise CliError("repository state must be a CID string or JSON object")


def _load_state_arg(path: str) -> Any:
    raw = _load_json(path, label="state")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, Mapping):
        # Allow {"cid": "..."} or full state objects.
        if "repository_state_cid" in raw or "identity_cid" in raw or "revision" in raw:
            return dict(raw)
        cid = raw.get("cid") or raw.get("repository_state_cid")
        if isinstance(cid, str) and cid:
            return cid
        return dict(raw)
    raise CliError("state JSON must be a string CID or object")


def _load_units(path: str) -> list[dict[str, Any]]:
    raw = _load_json(path, label="units")
    items = _require_list(raw, label="units")
    out: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise CliError(f"units[{index}] must be a JSON object")
        out.append(dict(item))
    return out


def _load_parent(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return _require_mapping(_load_json(path, label="parent"), label="parent")


def _load_policy(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return _require_mapping(_load_json(path, label="policy"), label="policy")


def _plan_flags(policy: Mapping[str, Any] | None) -> dict[str, Any]:
    if not policy:
        return {}
    allowed = {
        "trust_policy_changed",
        "schema_changed",
        "canonicalization_changed",
        "environment_changed",
        "circuit_or_key_changed",
        "full_fallback_required",
        "changed_root_cids",
        "new_source_root_cid",
    }
    return {key: policy[key] for key in allowed if key in policy}


def _verification_policy_view(policy: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not policy:
        return None
    # Strip planner flags when a mixed policy object is supplied.
    keys = {
        "policy_cid",
        "proof_schema_version",
        "canonicalization_version",
        "dependency_graph_schema_version",
        "circuit_id",
        "verification_key_id",
    }
    view = {key: policy[key] for key in keys if key in policy}
    return view or dict(policy)


def cmd_full(args: argparse.Namespace) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.full_checkpoint import (
        create_full_checkpoint,
    )

    state = _require_mapping(_load_json(args.state, label="state"), label="state")
    policy = _require_mapping(_load_json(args.policy, label="policy"), label="policy")
    units = _load_units(args.units)
    fallback = tuple(args.fallback_reasons) or ("first_state", "missing_parent")
    seal = create_full_checkpoint(
        state,
        policy,
        units=units,
        parent_seal_cid=args.parent_seal_cid,
        fallback_reasons=fallback,
    )
    status = _seal_status_value(seal)
    accepted = bool(getattr(seal, "sealed", False)) and _is_accepted_seal(status)
    return _envelope(
        command="full",
        status="ok" if accepted else status or "rejected",
        result={
            "seal": _canonical(seal),
            "seal_cid": seal.seal_cid(),
            "seal_status": status,
            "sealed": bool(getattr(seal, "sealed", False)),
            "reason": str(getattr(getattr(seal, "reason", ""), "value", getattr(seal, "reason", ""))),
        },
        exit_code=EXIT_OK if accepted else EXIT_OPERATION,
    )


def cmd_plan(args: argparse.Namespace) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.planner import (
        create_incremental_plan,
    )

    parent = _load_parent(args.parent)
    old = _load_state_arg(args.old)
    new = _load_state_arg(args.new)
    units = _load_units(args.units)
    policy = _load_policy(args.policy)
    plan = create_incremental_plan(
        parent,
        old,
        new,
        _verification_policy_view(policy),
        units=units,
        **_plan_flags(policy),
    )
    return _envelope(
        command="plan",
        status="ok",
        result={
            "plan": _canonical(plan),
            "plan_cid": plan.plan_cid(),
            "mode": plan.mode.value,
            "fallback_reasons": list(plan.fallback_reasons),
            "reusable_unit_ids": list(plan.reusable_unit_ids),
            "invalidated_unit_ids": list(plan.invalidated_unit_ids),
            "added_unit_ids": list(plan.added_unit_ids),
            "removed_unit_ids": list(plan.removed_unit_ids),
        },
    )


def cmd_incremental(args: argparse.Namespace) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.executor import (
        execute_incremental_plan,
    )
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.planner import (
        create_incremental_plan,
    )

    parent = _load_parent(args.parent)
    if parent is None:
        raise CliError("incremental requires --parent")
    old = _load_state_arg(args.old)
    new = _load_state_arg(args.new)
    units = _load_units(args.units)
    policy = _load_policy(args.policy)
    resource = None
    if args.resource_policy:
        resource = _require_mapping(
            _load_json(args.resource_policy, label="resource-policy"),
            label="resource-policy",
        )
    plan = create_incremental_plan(
        parent,
        old,
        new,
        _verification_policy_view(policy),
        units=units,
        **_plan_flags(policy),
    )
    result = execute_incremental_plan(
        plan,
        resource,
        backend_available=bool(args.backend_available),
    )
    ok = bool(result.succeeded)
    return _envelope(
        command="incremental",
        status="ok" if ok else result.outcome.value,
        result={
            "plan": _canonical(plan),
            "plan_cid": plan.plan_cid(),
            "execution": _canonical(result),
            "outcome": result.outcome.value,
            "succeeded": ok,
            "may_aggregate": result.may_aggregate,
            "reason_codes": list(result.reason_codes),
        },
        exit_code=EXIT_OK if ok else EXIT_OPERATION,
    )


def cmd_verify(args: argparse.Namespace) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.verification import (
        verify_seal,
    )

    seal = _require_mapping(_load_json(args.seal, label="seal"), label="seal")
    trusted_raw = _load_json(args.trusted_keys, label="trusted-keys")
    if isinstance(trusted_raw, list):
        trusted_keys: Any = [str(item) for item in trusted_raw]
    elif isinstance(trusted_raw, Mapping):
        trusted_keys = dict(trusted_raw)
    else:
        raise CliError("trusted-keys must be a JSON array or object")
    policy = _load_policy(args.policy)
    parent = _load_parent(args.parent)
    verification = verify_seal(
        seal,
        trusted_keys,
        _verification_policy_view(policy),
        parent_seal=parent,
    )
    ok = bool(verification.accepted)
    return _envelope(
        command="verify",
        status="ok" if ok else verification.reason.value,
        result={
            "verification": _canonical(verification),
            "accepted": ok,
            "reason": verification.reason.value,
            "seal_cid": verification.seal_cid,
            "seal_status": verification.seal_status,
        },
        exit_code=EXIT_OK if ok else EXIT_OPERATION,
    )


def cmd_explain_reuse(args: argparse.Namespace) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.explanations import (
        explain_reuse,
    )

    seal = _require_mapping(_load_json(args.seal, label="seal"), label="seal")
    explanation = explain_reuse(seal, args.unit)
    return _envelope(
        command="explain-reuse",
        status="ok",
        result={
            "explanation": _canonical(explanation),
            "unit_id": explanation.unit_id,
            "reused": explanation.reused,
            "disposition": explanation.disposition.value,
            "reason": explanation.reason,
            "substitutes_for_verification": False,
        },
    )


def cmd_explain_invalidation(args: argparse.Namespace) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.explanations import (
        explain_invalidation,
    )
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.planner import (
        create_incremental_plan,
    )

    parent = _load_parent(args.parent)
    old = _load_state_arg(args.old)
    new = _load_state_arg(args.new)
    units = _load_units(args.units)
    policy = _load_policy(args.policy)
    plan = create_incremental_plan(
        parent,
        old,
        new,
        _verification_policy_view(policy),
        units=units,
        **_plan_flags(policy),
    )
    explanation = explain_invalidation(plan, args.unit)
    return _envelope(
        command="explain-invalidation",
        status="ok",
        result={
            "explanation": _canonical(explanation),
            "plan_cid": plan.plan_cid(),
            "unit_id": explanation.unit_id,
            "invalidated": explanation.invalidated,
            "disposition": explanation.disposition.value,
            "reason": explanation.reason,
            "substitutes_for_verification": False,
        },
    )


def cmd_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.explanations import (
        compare_full_and_incremental,
    )

    state = _load_state_arg(args.state)
    parent = _load_parent(args.parent)
    policy = _load_policy(args.policy)
    units = _load_units(args.units)
    old = _load_state_arg(args.old) if args.old else None
    comparison = compare_full_and_incremental(
        state,
        parent,
        _verification_policy_view(policy),
        units=units,
        estimated=True,
        old_repository_state=old,
        **_plan_flags(policy),
    )
    return _envelope(
        command="benchmark",
        status="ok",
        result={
            "comparison": _canonical(comparison),
            "mode_selected": comparison.mode_selected,
            "estimated": comparison.estimated,
            "estimated_as_measured": False,
            "full_required_units": comparison.full_required_units,
            "incremental_prove_units": comparison.incremental_prove_units,
            "incremental_reuse_units": comparison.incremental_reuse_units,
            "fallback_reasons": list(comparison.fallback_reasons),
        },
    )


def cmd_cache_status(args: argparse.Namespace) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.planner import (
        create_incremental_plan,
    )

    parent = _load_parent(args.parent)
    old = _load_state_arg(args.old)
    new = _load_state_arg(args.new)
    units = _load_units(args.units)
    policy = _load_policy(args.policy)
    plan = create_incremental_plan(
        parent,
        old,
        new,
        _verification_policy_view(policy),
        units=units,
        **_plan_flags(policy),
    )
    unit_rows = []
    for item in plan.units:
        unit_rows.append(
            {
                "unit_id": item.unit_id,
                "kind": item.kind.value,
                "cache_key_complete": item.cache_key_complete,
                "admitted": item.admitted,
                "candidate_present": item.candidate_present,
                "reason": item.reason,
                "cache_fast_path": False,
            }
        )
    return _envelope(
        command="cache-status",
        status="ok",
        result={
            "plan_cid": plan.plan_cid(),
            "mode": plan.mode.value,
            "reusable_unit_ids": list(plan.reusable_unit_ids),
            "invalidated_unit_ids": list(plan.invalidated_unit_ids),
            "added_unit_ids": list(plan.added_unit_ids),
            "removed_unit_ids": list(plan.removed_unit_ids),
            "units": unit_rows,
            "cache_is_hint_only": True,
            "cache_authorizes_reuse": False,
            "executed": False,
        },
    )


def cmd_force_full(args: argparse.Namespace) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.planner import (
        create_incremental_plan,
    )

    parent = _load_parent(args.parent)
    old = _load_state_arg(args.old)
    new = _load_state_arg(args.new)
    units = _load_units(args.units)
    policy = _load_policy(args.policy) or {}
    flags = _plan_flags(policy)
    flags["full_fallback_required"] = True
    plan = create_incremental_plan(
        parent,
        old,
        new,
        _verification_policy_view(policy),
        units=units,
        **flags,
    )
    result: dict[str, Any] = {
        "plan": _canonical(plan),
        "plan_cid": plan.plan_cid(),
        "mode": plan.mode.value,
        "fallback_reasons": list(plan.fallback_reasons),
        "forced_full": plan.mode.value == "full",
    }
    if args.seal:
        if not args.state:
            raise CliError("force-full --seal requires --state")
        from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing.full_checkpoint import (
            create_full_checkpoint,
        )

        state = _require_mapping(_load_json(args.state, label="state"), label="state")
        vpolicy = _verification_policy_view(policy)
        if not vpolicy or "policy_cid" not in vpolicy:
            raise CliError("force-full --seal requires policy_cid in --policy")
        parent_cid = None
        if parent is not None:
            parent_cid = str(parent.get("seal_cid") or "") or None
        seal = create_full_checkpoint(
            state,
            vpolicy,
            units=units,
            parent_seal_cid=parent_cid,
            fallback_reasons=plan.fallback_reasons or ("full_fallback_required",),
        )
        status = _seal_status_value(seal)
        result["seal"] = _canonical(seal)
        result["seal_cid"] = seal.seal_cid()
        result["seal_status"] = status
        result["sealed"] = bool(getattr(seal, "sealed", False))
        accepted = bool(getattr(seal, "sealed", False))
        return _envelope(
            command="force-full",
            status="ok" if accepted else status or "rejected",
            result=result,
            exit_code=EXIT_OK if accepted else EXIT_OPERATION,
        )
    return _envelope(command="force-full", status="ok", result=result)


_COMMANDS = {
    "full": cmd_full,
    "incremental": cmd_incremental,
    "verify": cmd_verify,
    "plan": cmd_plan,
    "explain-reuse": cmd_explain_reuse,
    "explain-invalidation": cmd_explain_invalidation,
    "benchmark": cmd_benchmark,
    "cache-status": cmd_cache_status,
    "force-full": cmd_force_full,
}


def discovery_manifest() -> dict[str, Any]:
    """Cold discovery payload: no sealing side effects."""

    return {
        "schema": (
            "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/"
            "zk-seal-cli-discovery@1"
        ),
        "evidence_subset": CLI_SUBSET,
        "prog": CLI_PROG,
        "operations": list(CLI_OPERATIONS),
        "processes_started": False,
        "network_accessed": False,
        "keys_generated": False,
        "user_state_mutated": False,
        "auto_install": False,
        "proving_key_exported": False,
        "witness_exported": False,
    }


def _system_exit_code(exc: SystemExit) -> int:
    """Map argparse SystemExit to a process exit code (help is success)."""

    code = exc.code
    if code in (0, None):
        return EXIT_OK
    if isinstance(code, int):
        return code
    try:
        return int(code)
    except (TypeError, ValueError):
        return EXIT_USAGE


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """CLI entrypoint.  Returns a process exit code."""

    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        # --help / argparse usage exit here; no sealing side effects.
        return _system_exit_code(exc)

    if getattr(args, "version", False) and not args.command:
        payload = discovery_manifest()
        payload["package_schema_version"] = "1"
        out.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        return EXIT_OK

    if not args.command:
        parser.print_help(err)
        return EXIT_USAGE

    handler = _COMMANDS.get(args.command)
    if handler is None:
        envelope = _envelope(
            command=str(args.command),
            status="unknown_command",
            error=f"unknown command {args.command!r}",
            exit_code=EXIT_USAGE,
        )
        out.write(json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n")
        return EXIT_USAGE

    try:
        envelope = handler(args)
    except CliError as exc:
        envelope = _envelope(
            command=str(args.command),
            status=exc.status,
            error=str(exc),
            exit_code=EXIT_USAGE if exc.status == "error" else EXIT_OPERATION,
        )
    except Exception as exc:  # noqa: BLE001 — CLI boundary converts to JSON
        envelope = _envelope(
            command=str(args.command),
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            exit_code=EXIT_OPERATION,
        )

    out.write(json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n")
    return int(envelope.get("exit_code", EXIT_OPERATION))


if __name__ == "__main__":
    raise SystemExit(main())
