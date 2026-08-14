"""Narrowly scoped ``zk-seal`` CLI for IncrementalProofSealer (IPS-043).

Commands: ``full``, ``incremental``, ``verify``, ``plan``, ``explain-reuse``,
``explain-invalidation``, ``benchmark``, ``cache-status``, ``force-full``.

Cold ``--help`` / parser construction imports no backends, keys, kit store, or
datasets modules and starts no process, network, or state side effects.
Machine-readable JSON statuses/errors are emitted on stdout; exit codes are
stable and fail-closed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, TextIO

CLI_SUBSET = "ips/cli@1"
CLI_SCHEMA = (
    "ipfs_accelerate_py/agent_supervisor/proof/incremental_sealing/zk-seal-cli@1"
)
CLI_PROG = "zk-seal"

# Nine CLI operations required by IPS-043 effects/acceptance.
CLI_COMMANDS: tuple[str, ...] = (
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

# Closed CLI outcome vocabulary (aligned with seal/plan statuses).
CLI_STATUS_OK = "ok"
CLI_STATUS_ERROR = "error"
CLI_STATUS_REJECTED = "rejected"
CLI_STATUS_UNAVAILABLE = "unavailable"
CLI_STATUS_SIMULATED_ONLY = "simulated_only"
CLI_STATUS_SEALED_FULL = "sealed_full"
CLI_STATUS_SEALED_INCREMENTAL = "sealed_incremental"
CLI_STATUS_FULL_REPROOF_REQUIRED = "full_reproof_required"

_SENSITIVE_KEYS = frozenset(
    {
        "proving_key",
        "proving_key_bytes",
        "witness",
        "witness_bytes",
        "private_key",
        "secret",
        "trapdoor",
    }
)


class CliError(ValueError):
    """Fail-closed CLI contract violation with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def build_parser() -> argparse.ArgumentParser:
    """Build the ``zk-seal`` argument parser without importing heavy modules."""

    parser = argparse.ArgumentParser(
        prog=CLI_PROG,
        description=(
            "Narrow IncrementalProofSealer CLI.  Emits machine-readable JSON. "
            "Production seals reject simulated evidence."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{CLI_PROG} {CLI_SUBSET}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    full = sub.add_parser("full", help="Create a full checkpoint seal.")
    _add_json_input(full, "--state", required=True, help="Repository state JSON/file.")
    _add_json_input(full, "--policy", required=True, help="Verification policy JSON/file.")
    _add_json_input(full, "--units", required=False, help="Required unit evidence JSON list.")
    full.add_argument("--parent", default=None, help="Optional parent seal CID.")
    full.add_argument(
        "--fallback-reason",
        action="append",
        default=[],
        dest="fallback_reasons",
        help="Fallback reason (repeatable).",
    )

    incremental = sub.add_parser(
        "incremental",
        help="Plan and execute an incremental seal transition.",
    )
    _add_parent_arg(incremental, required=True)
    _add_json_input(
        incremental,
        "--old-state",
        required=True,
        help="Old repository state CID or JSON.",
    )
    _add_json_input(
        incremental,
        "--new-state",
        required=True,
        help="New repository state CID or JSON.",
    )
    _add_json_input(
        incremental,
        "--units",
        required=False,
        help="Unit planning inputs JSON list.",
    )
    _add_json_input(
        incremental,
        "--policy",
        required=False,
        help="Optional verification policy / flags JSON.",
    )
    _add_json_input(
        incremental,
        "--resource-policy",
        required=False,
        help="Optional resource policy JSON.",
    )
    incremental.add_argument(
        "--execute",
        action="store_true",
        help="Execute the plan (requires injected prove/fetch via env fixtures).",
    )

    verify = sub.add_parser("verify", help="Re-verify a seal under trusted keys.")
    _add_json_input(verify, "--seal", required=True, help="Seal JSON/file.")
    _add_json_input(
        verify,
        "--trusted-keys",
        required=False,
        help="Trusted key ids JSON list or policy mapping.",
    )
    _add_json_input(
        verify,
        "--policy",
        required=False,
        help="Verification policy JSON/file.",
    )
    _add_json_input(
        verify,
        "--parent",
        required=False,
        help="Optional parent seal JSON/file.",
    )

    plan = sub.add_parser("plan", help="Create an incremental proof plan.")
    _add_parent_arg(plan, required=False)
    _add_json_input(plan, "--old-state", required=True, help="Old repository state.")
    _add_json_input(plan, "--new-state", required=True, help="New repository state.")
    _add_json_input(plan, "--units", required=False, help="Unit planning inputs.")
    _add_json_input(plan, "--policy", required=False, help="Policy / change flags.")
    plan.add_argument(
        "--force-full",
        action="store_true",
        help="Force full fallback for the plan.",
    )

    explain_reuse = sub.add_parser(
        "explain-reuse",
        help="Explain reuse disposition for one unit on a seal.",
    )
    _add_json_input(explain_reuse, "--seal", required=True, help="Seal JSON/file.")
    explain_reuse.add_argument("--unit-id", required=True, help="Proof unit id.")

    explain_inv = sub.add_parser(
        "explain-invalidation",
        help="Explain invalidation disposition for one unit on a plan.",
    )
    _add_json_input(
        explain_inv,
        "--plan",
        required=False,
        help="Precomputed plan JSON (optional if planning inputs given).",
    )
    _add_parent_arg(explain_inv, required=False)
    _add_json_input(explain_inv, "--old-state", required=False, help="Old state.")
    _add_json_input(explain_inv, "--new-state", required=False, help="New state.")
    _add_json_input(explain_inv, "--units", required=False, help="Unit planning inputs.")
    _add_json_input(explain_inv, "--policy", required=False, help="Policy flags.")
    explain_inv.add_argument("--unit-id", required=True, help="Proof unit id.")

    benchmark = sub.add_parser(
        "benchmark",
        help="Compare full versus incremental cost for one step.",
    )
    _add_json_input(benchmark, "--state", required=True, help="Repository state.")
    _add_parent_arg(benchmark, required=False)
    _add_json_input(benchmark, "--policy", required=False, help="Policy JSON.")
    _add_json_input(benchmark, "--units", required=False, help="Unit planning inputs.")
    _add_json_input(
        benchmark,
        "--old-state",
        required=False,
        help="Optional old repository state.",
    )

    cache = sub.add_parser(
        "cache-status",
        help="Report typed cache/optional capability status (no network).",
    )
    cache.add_argument(
        "--backend",
        action="append",
        default=[],
        dest="backends",
        help="Optional backend id to probe (repeatable).",
    )

    force_full = sub.add_parser(
        "force-full",
        help="Force a full checkpoint (explicit full reproof path).",
    )
    _add_json_input(force_full, "--state", required=True, help="Repository state.")
    _add_json_input(force_full, "--policy", required=True, help="Verification policy.")
    _add_json_input(force_full, "--units", required=False, help="Required units.")
    force_full.add_argument("--parent", default=None, help="Optional parent seal CID.")

    return parser


def _add_parent_arg(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument(
        "--parent",
        required=required,
        default=None,
        help="Parent seal CID or parent seal context JSON/file.",
    )


def _add_json_input(
    parser: argparse.ArgumentParser,
    option: str,
    *,
    required: bool,
    help: str,
) -> None:
    dest = option.lstrip("-").replace("-", "_")
    parser.add_argument(option, dest=dest, required=required, default=None, help=help)


def _load_json_value(raw: str | None, *, field: str) -> Any:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        raise CliError("empty_input", f"{field} must not be empty")
    # File path takes precedence when the path exists.
    path = Path(text)
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CliError("invalid_json_file", f"{field}: {exc}") from exc
        return payload
    # Inline JSON object/array/string.
    if text[0] in "[{'\"" or text in {"true", "false", "null"} or text[:1].isdigit():
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            # Bare CID / string token.
            if text.startswith("sha256:") or "/" in text or text.isalnum():
                return text
            raise CliError("invalid_json", f"{field}: {exc}") from exc
    return text


def _state_value(raw: Any) -> Any:
    if raw is None:
        raise CliError("missing_state", "repository state is required")
    return raw


def _policy_value(raw: Any) -> Any:
    return raw


def _units_value(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, Mapping) and "units" in raw:
        units = raw["units"]
        if isinstance(units, list):
            return units
    raise CliError("invalid_units", "units must be a JSON list")


def _parent_context(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return raw
    if isinstance(raw, str) and raw:
        # Bare seal CID becomes a minimal parent context.
        return {
            "seal_cid": raw,
            "repository_state_cid": "",
            "source_root_cid": "",
        }
    raise CliError("invalid_parent", "parent must be a seal CID or JSON mapping")


def _canonical_or_mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_canonical") and callable(value.to_canonical):
        payload = value.to_canonical()
        if isinstance(payload, Mapping):
            return _strip_sensitive(dict(payload))
    if isinstance(value, Mapping):
        return _strip_sensitive(dict(value))
    return {"value": str(value)}


def _strip_sensitive(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _SENSITIVE_KEYS:
            continue
        if isinstance(value, Mapping):
            cleaned[key] = _strip_sensitive(dict(value))
        elif isinstance(value, list):
            cleaned[key] = [
                _strip_sensitive(dict(item)) if isinstance(item, Mapping) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    cleaned.setdefault("proving_key_exported", False)
    cleaned.setdefault("witness_exported", False)
    return cleaned


def _envelope(
    command: str,
    *,
    status: str,
    result: Mapping[str, Any] | None = None,
    error: Mapping[str, Any] | None = None,
    exit_code: int = 0,
) -> dict[str, Any]:
    return {
        "schema": CLI_SCHEMA,
        "evidence_subset": CLI_SUBSET,
        "command": command,
        "status": status,
        "result": dict(result or {}),
        "error": None if error is None else dict(error),
        "exit_code": exit_code,
        "proving_key_exported": False,
        "witness_exported": False,
    }


def _emit(payload: Mapping[str, Any], stream: TextIO) -> None:
    stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    stream.write("\n")


def _status_for_seal(seal: Any) -> str:
    sealed = bool(getattr(seal, "sealed", False))
    status_obj = getattr(seal, "seal_status", None)
    status = getattr(status_obj, "value", status_obj)
    if sealed and status == "sealed_full":
        return CLI_STATUS_SEALED_FULL
    if sealed and status == "sealed_incremental":
        return CLI_STATUS_SEALED_INCREMENTAL
    if status == "simulated_only":
        return CLI_STATUS_SIMULATED_ONLY
    if status in {"verification_failed", "proof_failed", "invalid_cache", "incomplete_manifest"}:
        return CLI_STATUS_REJECTED
    if status == "unavailable":
        return CLI_STATUS_UNAVAILABLE
    return str(status or CLI_STATUS_REJECTED)


def _exit_for_status(status: str) -> int:
    if status in {
        CLI_STATUS_OK,
        CLI_STATUS_SEALED_FULL,
        CLI_STATUS_SEALED_INCREMENTAL,
    }:
        return 0
    if status == CLI_STATUS_UNAVAILABLE:
        return 3
    if status == CLI_STATUS_SIMULATED_ONLY:
        return 4
    if status == CLI_STATUS_REJECTED:
        return 2
    return 1


def _cmd_full(args: argparse.Namespace) -> dict[str, Any]:
    from .full_checkpoint import create_full_checkpoint

    state = _load_json_value(args.state, field="state")
    policy = _load_json_value(args.policy, field="policy")
    units = _units_value(_load_json_value(args.units, field="units"))
    seal = create_full_checkpoint(
        _state_value(state),
        _policy_value(policy),
        units=units,
        parent_seal_cid=args.parent,
        fallback_reasons=tuple(args.fallback_reasons or ()),
    )
    status = _status_for_seal(seal)
    return _envelope(
        "full",
        status=status,
        result=_canonical_or_mapping(seal),
        error=(
            None
            if status in {CLI_STATUS_SEALED_FULL, CLI_STATUS_OK}
            else {
                "code": str(getattr(getattr(seal, "reason", None), "value", "rejected")),
                "message": f"full checkpoint status={status}",
            }
        ),
        exit_code=_exit_for_status(status),
    )


def _cmd_force_full(args: argparse.Namespace) -> dict[str, Any]:
    from .full_checkpoint import create_full_checkpoint

    state = _load_json_value(args.state, field="state")
    policy = _load_json_value(args.policy, field="policy")
    units = _units_value(_load_json_value(args.units, field="units"))
    seal = create_full_checkpoint(
        _state_value(state),
        _policy_value(policy),
        units=units,
        parent_seal_cid=args.parent,
        fallback_reasons=("explicit_force", "full_fallback_required"),
    )
    status = _status_for_seal(seal)
    result = _canonical_or_mapping(seal)
    result["forced_full"] = True
    result["force_reason"] = "explicit_force"
    return _envelope(
        "force-full",
        status=status,
        result=result,
        error=(
            None
            if status == CLI_STATUS_SEALED_FULL
            else {
                "code": str(getattr(getattr(seal, "reason", None), "value", "rejected")),
                "message": f"force-full status={status}",
            }
        ),
        exit_code=_exit_for_status(status),
    )


def _cmd_plan(args: argparse.Namespace) -> dict[str, Any]:
    from .planner import PlanMode, create_incremental_plan

    parent_raw = _load_json_value(args.parent, field="parent")
    parent = _parent_context(parent_raw)
    old_state = _load_json_value(args.old_state, field="old_state")
    new_state = _load_json_value(args.new_state, field="new_state")
    units = _units_value(_load_json_value(args.units, field="units"))
    policy = _load_json_value(args.policy, field="policy")
    flags: dict[str, Any] = {}
    if isinstance(policy, Mapping):
        flags.update(dict(policy))
    if getattr(args, "force_full", False):
        flags["full_fallback_required"] = True
    plan = create_incremental_plan(
        parent,
        old_state if old_state is not None else "",
        new_state if new_state is not None else "",
        flags or None,
        units=units,
    )
    status = (
        CLI_STATUS_FULL_REPROOF_REQUIRED
        if plan.mode is PlanMode.FULL
        else CLI_STATUS_OK
    )
    return _envelope(
        "plan",
        status=status,
        result={
            **_canonical_or_mapping(plan),
            "plan_cid": plan.plan_cid(),
        },
        exit_code=0,
    )


def _cmd_incremental(args: argparse.Namespace) -> dict[str, Any]:
    from .executor import execute_incremental_plan
    from .planner import PlanMode, create_incremental_plan

    parent = _parent_context(_load_json_value(args.parent, field="parent"))
    old_state = _load_json_value(args.old_state, field="old_state")
    new_state = _load_json_value(args.new_state, field="new_state")
    units = _units_value(_load_json_value(args.units, field="units"))
    policy = _load_json_value(args.policy, field="policy")
    resource_policy = _load_json_value(args.resource_policy, field="resource_policy")
    plan = create_incremental_plan(
        parent,
        old_state if old_state is not None else "",
        new_state if new_state is not None else "",
        policy if isinstance(policy, Mapping) else None,
        units=units,
    )
    payload: dict[str, Any] = {
        "plan": {**_canonical_or_mapping(plan), "plan_cid": plan.plan_cid()},
    }
    if not args.execute:
        status = (
            CLI_STATUS_FULL_REPROOF_REQUIRED
            if plan.mode is PlanMode.FULL
            else CLI_STATUS_OK
        )
        return _envelope(
            "incremental",
            status=status,
            result=payload,
            exit_code=0,
        )

    # Execution without injected prove/fetch remains fail-closed hermetic:
    # missing candidates reject reuse; prove defaults are not auto-installed.
    result = execute_incremental_plan(
        plan,
        resource_policy if isinstance(resource_policy, Mapping) else resource_policy,
        backend_available=True,
    )
    payload["execution"] = _canonical_or_mapping(result)
    if result.succeeded:
        status = CLI_STATUS_OK
        exit_code = 0
    elif "simulated_forbidden" in result.reason_codes:
        status = CLI_STATUS_SIMULATED_ONLY
        exit_code = 4
    elif result.outcome.value == "unavailable":
        status = CLI_STATUS_UNAVAILABLE
        exit_code = 3
    else:
        status = CLI_STATUS_REJECTED
        exit_code = 2
    return _envelope(
        "incremental",
        status=status,
        result=payload,
        error=(
            None
            if exit_code == 0
            else {
                "code": result.outcome.value,
                "message": ",".join(result.reason_codes) or result.outcome.value,
            }
        ),
        exit_code=exit_code,
    )


def _cmd_verify(args: argparse.Namespace) -> dict[str, Any]:
    from .verification import verify_seal

    seal = _load_json_value(args.seal, field="seal")
    if seal is None:
        raise CliError("missing_seal", "seal is required")
    trusted = _load_json_value(args.trusted_keys, field="trusted_keys")
    policy = _load_json_value(args.policy, field="policy")
    parent = _load_json_value(args.parent, field="parent")
    result = verify_seal(
        seal,
        trusted_keys=trusted,
        verification_policy=policy,
        parent_seal=parent,
    )
    status = CLI_STATUS_OK if result.accepted else CLI_STATUS_REJECTED
    return _envelope(
        "verify",
        status=status,
        result=_canonical_or_mapping(result),
        error=(
            None
            if result.accepted
            else {
                "code": result.reason.value,
                "message": result.message,
            }
        ),
        exit_code=0 if result.accepted else 2,
    )


def _cmd_explain_reuse(args: argparse.Namespace) -> dict[str, Any]:
    from .explanations import explain_reuse

    seal = _load_json_value(args.seal, field="seal")
    if seal is None:
        raise CliError("missing_seal", "seal is required")
    explanation = explain_reuse(seal, args.unit_id)
    return _envelope(
        "explain-reuse",
        status=CLI_STATUS_OK,
        result=_canonical_or_mapping(explanation),
        exit_code=0,
    )


def _cmd_explain_invalidation(args: argparse.Namespace) -> dict[str, Any]:
    from .explanations import explain_invalidation
    from .planner import create_incremental_plan

    plan_raw = _load_json_value(args.plan, field="plan")
    if plan_raw is not None:
        # Plans must be IncrementalProofPlan objects; rebuild from planning inputs
        # when a mapping is provided without a live plan object.
        if hasattr(plan_raw, "units"):
            plan = plan_raw
        else:
            raise CliError(
                "plan_rebuild_required",
                "pass planning inputs (--parent/--old-state/--new-state/--units) "
                "instead of raw plan JSON; plan objects are not reconstructed from "
                "arbitrary mappings",
            )
    else:
        parent = _parent_context(_load_json_value(args.parent, field="parent"))
        old_state = _load_json_value(args.old_state, field="old_state")
        new_state = _load_json_value(args.new_state, field="new_state")
        if old_state is None or new_state is None:
            raise CliError(
                "missing_plan_inputs",
                "explain-invalidation requires --plan or planning inputs",
            )
        units = _units_value(_load_json_value(args.units, field="units"))
        policy = _load_json_value(args.policy, field="policy")
        plan = create_incremental_plan(
            parent,
            old_state,
            new_state,
            policy if isinstance(policy, Mapping) else None,
            units=units,
        )
    explanation = explain_invalidation(plan, args.unit_id)
    return _envelope(
        "explain-invalidation",
        status=CLI_STATUS_OK,
        result=_canonical_or_mapping(explanation),
        exit_code=0,
    )


def _cmd_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from .explanations import compare_full_and_incremental

    state = _load_json_value(args.state, field="state")
    parent = _parent_context(_load_json_value(args.parent, field="parent"))
    policy = _load_json_value(args.policy, field="policy")
    units = _units_value(_load_json_value(args.units, field="units"))
    old_state = _load_json_value(args.old_state, field="old_state")
    comparison = compare_full_and_incremental(
        _state_value(state),
        parent,
        policy if isinstance(policy, Mapping) else policy,
        units=units,
        old_repository_state=old_state,
        estimated=True,
    )
    return _envelope(
        "benchmark",
        status=CLI_STATUS_OK,
        result=_canonical_or_mapping(comparison),
        exit_code=0,
    )


def _cmd_cache_status(args: argparse.Namespace) -> dict[str, Any]:
    # Lazy import of package facade for typed optional capabilities only.
    from . import report_optional_capabilities
    from .backends import probe_backend_capability

    capabilities = report_optional_capabilities()
    backend_details: dict[str, Any] = dict(capabilities.get("backends") or {})
    for backend_id in args.backends or ():
        try:
            cap = probe_backend_capability(
                str(backend_id),
                allow_recursion_probe=False,
            )
            backend_details[str(backend_id)] = {
                "status": cap.status.value,
                "production_seal_allowed": cap.production_seal_allowed,
                "recursive_verification": cap.recursive_verification,
                "reason_code": cap.reason_code,
                "message": cap.message,
            }
        except Exception as exc:
            backend_details[str(backend_id)] = {
                "status": "unknown",
                "production_seal_allowed": False,
                "recursive_verification": False,
                "reason_code": "probe_error",
                "message": str(exc),
            }

    kit = capabilities.get("kit_store") or {}
    kit_status = str(kit.get("status") or "unavailable")
    # Cache status itself is always reportable; absence is typed, not fatal.
    status = CLI_STATUS_OK if kit_status != "error" else CLI_STATUS_ERROR
    return _envelope(
        "cache-status",
        status=status,
        result={
            "cache": {
                "status": kit_status,
                "reason_code": kit.get("reason_code"),
                "message": kit.get("message"),
                "writable": False,
                "networked": False,
                "injected": False,
            },
            "optional_capabilities": capabilities,
            "backends": backend_details,
        },
        exit_code=0,
    )


_COMMAND_HANDLERS: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
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


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run ``zk-seal`` and return a process exit code."""

    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    parser = build_parser()
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        # argparse --help / usage errors.
        code = int(exc.code) if isinstance(exc.code, int) else 2
        return code

    command = str(args.command)
    handler = _COMMAND_HANDLERS.get(command)
    if handler is None:
        payload = _envelope(
            command,
            status=CLI_STATUS_ERROR,
            error={"code": "unknown_command", "message": f"unknown command {command!r}"},
            exit_code=2,
        )
        _emit(payload, out)
        return 2

    try:
        payload = handler(args)
    except CliError as exc:
        payload = _envelope(
            command,
            status=CLI_STATUS_ERROR,
            error={"code": exc.code, "message": exc.message},
            exit_code=1,
        )
    except Exception as exc:
        payload = _envelope(
            command,
            status=CLI_STATUS_ERROR,
            error={
                "code": type(exc).__name__,
                "message": str(exc),
            },
            exit_code=1,
        )

    _emit(payload, out)
    if payload.get("error") and payload.get("status") == CLI_STATUS_ERROR:
        # Mirror short error to stderr for operators without dumping secrets.
        err.write(f"{CLI_PROG}: {payload['error'].get('message', 'error')}\n")
    return int(payload.get("exit_code", 1))


def run(argv: Sequence[str] | None = None) -> int:
    """Alias for :func:`main` used by packaging entry points."""

    return main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = (
    "CLI_COMMANDS",
    "CLI_PROG",
    "CLI_SCHEMA",
    "CLI_SUBSET",
    "CliError",
    "build_parser",
    "main",
    "run",
)
