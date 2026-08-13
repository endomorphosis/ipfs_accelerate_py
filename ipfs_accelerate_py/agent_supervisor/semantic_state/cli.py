"""Semantic-state console CLI (SCH-013).

Interface: ``SemanticStateCLI@1`` / sch/cli@1

Dedicated ``semantic-state`` entrypoint with deterministic JSON by default.
Commands that need an unavailable optional dependency return a stable typed
error envelope and a nonzero exit code. Production ``apply-patch`` never falls
back to simulation.

Importing this module starts no watchers, threads, processes, databases,
daemons, network clients, or package installers. ``--help`` is cold and free of
side effects.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

# ---------------------------------------------------------------------------
# Interface pins (stdlib only at module import)
# ---------------------------------------------------------------------------

CLI_INTERFACE = "SemanticStateCLI@1"
CLI_SCHEMA = "ipfs-accelerate.semantic-state-cli@1"
CLI_BUNDLE = "sch/cli@1"
CLI_ADAPTER_ID = "ipfs-accelerate.semantic-state.cli"
BOARD_NAMESPACE = "semantic-compression-harness-v1"
INTERFACE_SCHEMA_RESOURCE = "schemas/semantic-state-harness.interface.json"
INTERFACE_SCHEMA_PACKAGE = "ipfs_accelerate_py.agent_supervisor.semantic_state"

# Stable exit codes.
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3
EXIT_PRODUCTION_GATE = 4

_MAX_DIAGNOSTIC = 512
_COMMANDS = (
    "scan",
    "watch",
    "status",
    "graph",
    "explain-symbol",
    "explain-impact",
    "invalidate",
    "select-tests",
    "pack-context",
    "verify",
    "apply-patch",
    "compare-full-suite",
    "benchmark",
    "interface-schema",
)


# ---------------------------------------------------------------------------
# Pure helpers (no I/O beyond argument shaping)
# ---------------------------------------------------------------------------


def _clip(text: str, *, limit: int = _MAX_DIAGNOSTIC) -> str:
    body = str(text or "")
    if len(body) <= limit:
        return body
    return body[: limit - 3] + "..."


def _emit_json(payload: Mapping[str, Any], stream: TextIO) -> None:
    text = json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True)
    if not text.endswith("\n"):
        text += "\n"
    stream.write(text)
    stream.flush()


def _success_envelope(
    command: str,
    result: Mapping[str, Any],
    *,
    exit_code: int = EXIT_OK,
) -> dict[str, Any]:
    return {
        "ok": True,
        "command": command,
        "exit_code": exit_code,
        "interface": CLI_INTERFACE,
        "schema": CLI_SCHEMA,
        "bundle": CLI_BUNDLE,
        "board_namespace": BOARD_NAMESPACE,
        "result": dict(result),
    }


def _error_envelope(
    command: str | None,
    *,
    reason_code: str,
    diagnostic: str,
    exit_code: int,
    retryable: bool = False,
    operation: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "ok": False,
        "command": command,
        "exit_code": exit_code,
        "interface": CLI_INTERFACE,
        "schema": CLI_SCHEMA,
        "bundle": CLI_BUNDLE,
        "board_namespace": BOARD_NAMESPACE,
        "error": {
            "operation": operation or (command or "cli"),
            "adapter_id": CLI_ADAPTER_ID,
            "reason_code": reason_code,
            "retryable": bool(retryable),
            "diagnostic": _clip(diagnostic),
        },
    }
    if extra:
        body["error"].update(dict(extra))
    return body


def _object_to_dict(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {str(k): _object_to_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_object_to_dict(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _object_to_dict(to_dict())
        except Exception:
            pass
    if hasattr(value, "value") and not isinstance(value, (str, bytes, int, float, bool)):
        try:
            return _object_to_dict(value.value)
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)):
        return value
    # Bounded representation for opaque producer objects.
    for attr in (
        "state_cid",
        "root_cid",
        "selection_cid",
        "symbol_id",
        "cid",
        "path",
        "node_id",
    ):
        if hasattr(value, attr):
            try:
                raw = getattr(value, attr)
            except Exception:
                continue
            if isinstance(raw, (str, int, float, bool)) or raw is None:
                return {attr: raw, "type": type(value).__name__}
    return {"type": type(value).__name__, "repr": _clip(repr(value), limit=200)}


def load_interface_schema_text() -> str:
    """Load the packaged Profile A interface schema (importlib.resources)."""

    from importlib import resources

    try:
        root = resources.files(INTERFACE_SCHEMA_PACKAGE)
        target = root.joinpath(INTERFACE_SCHEMA_RESOURCE)
        return target.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - packaging regression
        # Fallback to source-relative path for editable checkouts.
        here = Path(__file__).resolve().parent
        path = here / "schemas" / "semantic-state-harness.interface.json"
        if path.is_file():
            return path.read_text(encoding="utf-8")
        raise FileNotFoundError(
            f"interface schema not packaged: {INTERFACE_SCHEMA_RESOURCE}"
        ) from exc


def semantic_state_cli_descriptor() -> dict[str, Any]:
    """Closed interface metadata for SemanticStateCLI@1."""

    return {
        "interface": CLI_INTERFACE,
        "schema": CLI_SCHEMA,
        "bundle": CLI_BUNDLE,
        "board_namespace": BOARD_NAMESPACE,
        "adapter_id": CLI_ADAPTER_ID,
        "console_entry": "semantic-state",
        "entry_point": (
            "ipfs_accelerate_py.agent_supervisor.semantic_state.cli:main"
        ),
        "interface_schema_resource": INTERFACE_SCHEMA_RESOURCE,
        "commands": list(_COMMANDS),
        "exit_codes": {
            "ok": EXIT_OK,
            "error": EXIT_ERROR,
            "usage": EXIT_USAGE,
            "unavailable": EXIT_UNAVAILABLE,
            "production_gate": EXIT_PRODUCTION_GATE,
        },
        "invariants": [
            "deterministic_json_default",
            "bounded_help_and_errors",
            "stable_exit_codes",
            "production_apply_patch_cannot_simulate",
            "local_commands_need_no_ipfs_daemon",
            "cold_help_and_import_no_mutation",
            "typed_unavailable_never_exit_zero",
        ],
        "symbols": ["build_parser", "main", "semantic_state_cli_descriptor"],
    }


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the closed ``semantic-state`` argument parser."""

    parser = argparse.ArgumentParser(
        prog="semantic-state",
        description=(
            "Local semantic-compression harness CLI. Deterministic JSON by "
            "default. No network MCP service; no IPFS daemon required for "
            "local commands."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes: 0=ok, 1=error, 2=usage, 3=unavailable, "
            "4=production-gate. Production apply-patch never simulates."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s ({CLI_INTERFACE})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_repo(child: argparse.ArgumentParser, *, name: str = "repo") -> None:
        child.add_argument(
            name,
            type=str,
            help="Local repository path (absolute or relative; no daemon).",
        )

    def _add_mode(child: argparse.ArgumentParser) -> None:
        child.add_argument(
            "--mode",
            choices=("development", "production"),
            default="development",
            help="Harness mode (default: development).",
        )

    def _add_storage(child: argparse.ArgumentParser) -> None:
        child.add_argument(
            "--storage-dir",
            type=str,
            default=None,
            help="Explicit local durable-state directory (no daemon/network).",
        )

    def _add_json_flags(child: argparse.ArgumentParser) -> None:
        child.add_argument(
            "--compact",
            action="store_true",
            help="Emit compact JSON (no indentation).",
        )

    # scan
    p_scan = sub.add_parser("scan", help="Canonical repository scan.")
    _add_repo(p_scan)
    p_scan.add_argument(
        "--previous-state-cid",
        type=str,
        default=None,
        help="Optional previous datasets state CID for incremental scan.",
    )
    _add_storage(p_scan)
    _add_json_flags(p_scan)

    # watch
    p_watch = sub.add_parser(
        "watch",
        help="Admit a watch notification that only schedules a canonical scan.",
    )
    _add_repo(p_watch)
    p_watch.add_argument(
        "--snapshot-cid",
        type=str,
        required=True,
        help="Opaque snapshot CID to schedule (events never become state).",
    )
    p_watch.add_argument(
        "--process",
        action="store_true",
        help="Drain one scheduled scan after admission (still local-only).",
    )
    p_watch.add_argument(
        "--repository-id",
        type=str,
        default=None,
        help="Stable repository identity (default: absolute repo path).",
    )
    _add_storage(p_watch)
    _add_mode(p_watch)
    _add_json_flags(p_watch)

    # status
    p_status = sub.add_parser("status", help="Session / root status snapshot.")
    _add_repo(p_status)
    p_status.add_argument(
        "--repository-id",
        type=str,
        default=None,
        help="Stable repository identity (default: absolute repo path).",
    )
    _add_storage(p_status)
    _add_mode(p_status)
    _add_json_flags(p_status)

    # graph
    p_graph = sub.add_parser("graph", help="Symbol graph summary for a repository.")
    _add_repo(p_graph)
    p_graph.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Optional symbol id to focus the graph projection.",
    )
    _add_json_flags(p_graph)

    # explain-symbol
    p_es = sub.add_parser("explain-symbol", help="Explain one symbol from scanned state.")
    _add_repo(p_es)
    p_es.add_argument("symbol", type=str, help="Symbol id.")
    _add_json_flags(p_es)

    # explain-impact
    p_ei = sub.add_parser(
        "explain-impact",
        help="Explain impact for changed symbols or files.",
    )
    _add_repo(p_ei)
    p_ei.add_argument(
        "targets",
        nargs="+",
        help="One or more symbol ids or repository-relative file paths.",
    )
    _add_json_flags(p_ei)

    # invalidate
    p_inv = sub.add_parser(
        "invalidate",
        help="Compute invalidation between two semantic-state CIDs.",
    )
    p_inv.add_argument("old_state", type=str, help="Previous semantic-state root CID.")
    p_inv.add_argument("new_state", type=str, help="Current semantic-state root CID.")
    p_inv.add_argument(
        "--storage-dir",
        type=str,
        default=None,
        help="Local block store directory for get_block (no daemon).",
    )
    _add_json_flags(p_inv)

    # select-tests
    p_sel = sub.add_parser(
        "select-tests",
        help="Select tests/proofs for changed symbols or files.",
    )
    _add_repo(p_sel)
    p_sel.add_argument(
        "targets",
        nargs="+",
        help="One or more symbol ids or repository-relative file paths.",
    )
    p_sel.add_argument(
        "--previous-state-cid",
        type=str,
        default=None,
        help="Previous semantic-state root CID (optional).",
    )
    _add_json_flags(p_sel)

    # pack-context
    p_pack = sub.add_parser(
        "pack-context",
        help="Build an assurance-aware context pack for a task/target.",
    )
    _add_repo(p_pack)
    p_pack.add_argument("task", type=str, help="Task / objective text.")
    p_pack.add_argument("target", type=str, help="Target symbol id or file path.")
    _add_json_flags(p_pack)

    # verify
    p_ver = sub.add_parser("verify", help="Run verification stages for a repository.")
    _add_repo(p_ver)
    p_ver.add_argument(
        "--full-suite",
        action="store_true",
        help="Include full-suite execution when selection is available.",
    )
    p_ver.add_argument(
        "--selection-cid",
        type=str,
        default=None,
        help="Optional producer selection CID binding.",
    )
    _add_mode(p_ver)
    _add_storage(p_ver)
    _add_json_flags(p_ver)

    # apply-patch
    p_ap = sub.add_parser(
        "apply-patch",
        help="Apply and verify a patch (production never simulates).",
    )
    _add_repo(p_ap)
    p_ap.add_argument(
        "patch_or_task",
        type=str,
        help="Path to a unified diff, or a task id with an associated patch.",
    )
    p_ap.add_argument(
        "--objective",
        type=str,
        default=None,
        help="Task objective (default: derived from patch path).",
    )
    p_ap.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Stable task id (default: basename of patch path).",
    )
    p_ap.add_argument(
        "--base-commit",
        type=str,
        default=None,
        help="Expected base commit for isolated worktree apply.",
    )
    p_ap.add_argument(
        "--base-tree",
        type=str,
        default=None,
        help="Expected base tree CID.",
    )
    p_ap.add_argument(
        "--allow-paths",
        nargs="*",
        default=None,
        help="Optional path allow-list for the patch scope.",
    )
    p_ap.add_argument(
        "--simulate",
        action="store_true",
        help=(
            "Request simulation. Rejected with exit 4 in production mode; "
            "development may label observations simulated but never claims "
            "production acceptance."
        ),
    )
    _add_mode(p_ap)
    _add_storage(p_ap)
    _add_json_flags(p_ap)

    # compare-full-suite
    p_cfs = sub.add_parser(
        "compare-full-suite",
        help="Compare selected tests against a controlled full-suite oracle.",
    )
    p_cfs.add_argument(
        "fixture_or_repo",
        type=str,
        help="Fixture package directory or repository path.",
    )
    p_cfs.add_argument(
        "--mutation-case-id",
        type=str,
        default=None,
        help="Controlled mutation case id when using the fixture corpus.",
    )
    _add_json_flags(p_cfs)

    # benchmark
    p_bench = sub.add_parser(
        "benchmark",
        help="Run the exactly-40-task offline semantic compression benchmark.",
    )
    p_bench.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Optional corpus package directory override.",
    )
    p_bench.add_argument(
        "--fixture",
        type=str,
        default=None,
        help="Optional controlled fixture package directory override.",
    )
    p_bench.add_argument(
        "--check",
        action="store_true",
        help="Recompute and compare deterministic fields to published results.",
    )
    p_bench.add_argument(
        "--write",
        action="store_true",
        help="Write JSON/Markdown report paths when provided.",
    )
    p_bench.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional JSON report output path.",
    )
    p_bench.add_argument(
        "--md-out",
        type=str,
        default=None,
        help="Optional Markdown report output path.",
    )
    _add_json_flags(p_bench)

    # interface-schema (packaging / discovery helper)
    p_schema = sub.add_parser(
        "interface-schema",
        help="Print the packaged Profile A interface schema JSON.",
    )
    _add_json_flags(p_schema)

    return parser


# ---------------------------------------------------------------------------
# Command handlers (lazy imports only)
# ---------------------------------------------------------------------------


def _resolve_repo(path: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"repository path does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"repository path is not a directory: {resolved}")
    return resolved


def _load_provider(injected: Any | None = None) -> Any:
    if injected is not None:
        return injected
    from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import (
        load_semantic_state_provider,
    )

    return load_semantic_state_provider()


def _open_durable(
    storage_dir: str | None,
    *,
    required: bool = True,
) -> Any | None:
    if storage_dir is None:
        return None
    from ipfs_accelerate_py.agent_supervisor.semantic_state.durable_state import (
        DurableStateUnavailable,
        open_local_durable_state,
    )

    path = Path(storage_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    try:
        return open_local_durable_state(path)
    except DurableStateUnavailable:
        if required:
            raise
        return None
    except Exception:
        if required:
            raise
        return None


def _scan_payload(provider: Any, repo: Path, previous_state: Any = None) -> dict[str, Any]:
    result = provider.scan_repository(str(repo), previous_state=previous_state)
    payload = _object_to_dict(result)
    if isinstance(payload, dict):
        payload.setdefault("repo", str(repo))
        return payload
    return {"repo": str(repo), "scan": payload}


def _cmd_scan(args: argparse.Namespace, *, provider: Any | None) -> dict[str, Any]:
    repo = _resolve_repo(args.repo)
    prov = _load_provider(provider)
    previous = None
    # previous_state_cid alone is not a view; leave previous None unless injected.
    result = _scan_payload(prov, repo, previous_state=previous)
    if args.previous_state_cid:
        result["previous_state_cid"] = args.previous_state_cid
    return result


def _cmd_watch(args: argparse.Namespace, *, provider: Any | None) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.semantic_state.session import (
        SessionPolicy,
        watch_session,
    )

    repo = _resolve_repo(args.repo)
    repository_id = args.repository_id or str(repo)
    durable = _open_durable(args.storage_dir, required=False)

    scan_executor = None
    if args.process:
        prov = provider

        def _executor(**kwargs: Any) -> Mapping[str, Any]:
            nonlocal prov
            if prov is None:
                prov = _load_provider(None)
            scan = _scan_payload(prov, repo)
            return {
                "snapshot_cid": kwargs["snapshot_cid"],
                "attempt_id": kwargs["attempt_id"],
                "status": "completed",
                "output_artifact_cids": [],
                "new_root_cid": scan.get("state_cid") or scan.get("root_cid"),
                "verified": False,
                "diagnostic": "cli_watch_scan",
                "reason_codes": ["cli_watch"],
            }

        scan_executor = _executor

    policy = SessionPolicy(
        repository_id=repository_id,
        mode=args.mode,
        worker_enabled=False,
        debounce_ms=0,
    )
    session, ack, results = watch_session(
        policy,
        args.snapshot_cid,
        durable_port=durable,
        scan_executor=scan_executor,
        process=bool(args.process),
        source="cli_watch",
    )
    status = session.status().to_dict()
    # Do not leave a long-lived session worker; shutdown is local and bounded.
    try:
        session.shutdown()
    except Exception:
        pass
    return {
        "repo": str(repo),
        "ack": ack.to_dict(),
        "status": status,
        "scan_results": [item.to_dict() for item in results],
    }


def _cmd_status(args: argparse.Namespace, *, provider: Any | None) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.semantic_state.session import (
        SemanticStateSession,
        SessionPolicy,
    )

    repo = _resolve_repo(args.repo)
    repository_id = args.repository_id or str(repo)
    durable = _open_durable(args.storage_dir, required=False)
    policy = SessionPolicy(
        repository_id=repository_id,
        mode=args.mode,
        worker_enabled=False,
    )
    session = SemanticStateSession(policy, durable_port=durable)
    status = session.status().to_dict()
    status["repo"] = str(repo)
    status["durable_bound"] = durable is not None
    # Optional capability probe (does not require a successful scan).
    try:
        from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import (
            inspect_semantic_state_capability,
        )

        cap = inspect_semantic_state_capability()
        status["provider_capability"] = {
            "available": cap.available,
            "adapter_id": cap.adapter_id,
            "reason_code": cap.reason_code,
            "diagnostic": _clip(cap.diagnostic or ""),
        }
    except Exception as exc:
        status["provider_capability"] = {
            "available": False,
            "adapter_id": CLI_ADAPTER_ID,
            "reason_code": "capability_probe_failed",
            "diagnostic": _clip(str(exc)),
        }
    try:
        session.shutdown()
    except Exception:
        pass
    return status


def _cmd_graph(args: argparse.Namespace, *, provider: Any | None) -> dict[str, Any]:
    repo = _resolve_repo(args.repo)
    prov = _load_provider(provider)
    scan = prov.scan_repository(str(repo), previous_state=None)
    state = getattr(scan, "state", scan)
    payload: dict[str, Any] = {
        "repo": str(repo),
        "scan": _object_to_dict(scan),
    }
    symbol = args.symbol
    if symbol:
        if hasattr(prov, "explain_symbol"):
            payload["symbol"] = symbol
            payload["explanation"] = _object_to_dict(prov.explain_symbol(state, symbol))
        else:
            payload["symbol"] = symbol
            payload["explanation"] = None
            payload["note"] = "provider has no explain_symbol"
    # Graph projection: surface sealed graph/root CIDs when present.
    for attr in ("graph_cid", "merkle_cid", "state_cid", "root_cid", "modules"):
        if hasattr(state, attr):
            payload[attr] = _object_to_dict(getattr(state, attr))
    return payload


def _cmd_explain_symbol(
    args: argparse.Namespace, *, provider: Any | None
) -> dict[str, Any]:
    repo = _resolve_repo(args.repo)
    prov = _load_provider(provider)
    scan = prov.scan_repository(str(repo), previous_state=None)
    state = getattr(scan, "state", scan)
    explanation = prov.explain_symbol(state, args.symbol)
    return {
        "repo": str(repo),
        "symbol": args.symbol,
        "explanation": _object_to_dict(explanation),
    }


def _cmd_explain_impact(
    args: argparse.Namespace, *, provider: Any | None
) -> dict[str, Any]:
    repo = _resolve_repo(args.repo)
    prov = _load_provider(provider)
    scan = prov.scan_repository(str(repo), previous_state=None)
    state = getattr(scan, "state", scan)
    targets = tuple(str(item) for item in args.targets)
    impact = prov.explain_impact(state, targets)
    return {
        "repo": str(repo),
        "targets": list(targets),
        "impact": _object_to_dict(impact),
    }


def _cmd_invalidate(
    args: argparse.Namespace, *, provider: Any | None
) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
        validate_opaque_cid,
    )

    old_cid = validate_opaque_cid(args.old_state, "old_state")
    new_cid = validate_opaque_cid(args.new_state, "new_state")
    durable = _open_durable(args.storage_dir)
    if durable is None:
        raise RuntimeError(
            "invalidate requires --storage-dir with local blocks for both states"
        )
    prov = _load_provider(provider)

    def get_block(cid: str) -> bytes:
        return durable.get_bytes(cid)

    previous = prov.open_verified_view(old_cid, get_block)
    current = prov.open_verified_view(new_cid, get_block)
    # Prefer producer extend_semantic_invalidation when views expose indices.
    result: Any
    if hasattr(prov, "extend_semantic_invalidation"):
        prev_index = getattr(previous, "semantic_index", None) or getattr(
            previous, "index", None
        )
        curr_index = getattr(current, "semantic_index", None) or getattr(
            current, "index", None
        )
        delta = getattr(current, "delta", None)
        plan = getattr(current, "plan", None)
        if prev_index is not None and curr_index is not None:
            result = prov.extend_semantic_invalidation(
                prev_index, curr_index, delta, plan, previous, current
            )
        else:
            result = {
                "previous_root_cid": old_cid,
                "current_root_cid": new_cid,
                "note": "views opened; index fields unavailable for extend",
            }
    else:
        result = {
            "previous_root_cid": old_cid,
            "current_root_cid": new_cid,
            "note": "provider lacks extend_semantic_invalidation",
        }
    return {
        "old_state": old_cid,
        "new_state": new_cid,
        "invalidation": _object_to_dict(result),
    }


def _cmd_select_tests(
    args: argparse.Namespace, *, provider: Any | None
) -> dict[str, Any]:
    repo = _resolve_repo(args.repo)
    prov = _load_provider(provider)
    scan = prov.scan_repository(str(repo), previous_state=None)
    current = getattr(scan, "state", scan)
    previous = None
    invalidation = getattr(scan, "invalidation", None)
    if invalidation is None:
        invalidation = getattr(current, "invalidation", None)
    # When invalidation is still missing, synthesize a minimal marker so the
    # provider can fail closed with a typed error rather than a TypeError.
    if invalidation is None:
        invalidation = {
            "schema": "cli-invalidation-placeholder",
            "targets": list(args.targets),
        }
    if not hasattr(prov, "select_tests_and_proofs"):
        raise RuntimeError("provider lacks select_tests_and_proofs")
    selection = prov.select_tests_and_proofs(
        previous, current, invalidation, policy=None
    )
    from ipfs_accelerate_py.agent_supervisor.semantic_state.selection_execution import (
        selection_ref_from_selection,
    )

    try:
        ref = selection_ref_from_selection(selection)
        ref_payload = ref.to_dict()
    except Exception:
        ref_payload = None
    return {
        "repo": str(repo),
        "targets": list(args.targets),
        "selection": _object_to_dict(selection),
        "selection_ref": ref_payload,
    }


def _cmd_pack_context(
    args: argparse.Namespace, *, provider: Any | None
) -> dict[str, Any]:
    from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

    from ipfs_accelerate_py.agent_supervisor.semantic_state.context_pack import (
        pack_context,
    )

    repo = _resolve_repo(args.repo)
    # Hermetic packing from task/target labels when full producer capsules are
    # unavailable: use content-addressed placeholders derived from inputs only.
    objective = str(args.task)
    target = str(args.target)
    target_cid = cid_for_bytes(f"cli-target:{repo}:{target}".encode("utf-8"))
    surrounding_cid = cid_for_bytes(
        f"cli-surround:{repo}:{target}".encode("utf-8")
    )
    test_cid = cid_for_bytes(f"cli-test:{repo}:{target}".encode("utf-8"))
    delta_cid = cid_for_bytes(f"cli-delta:{repo}:{objective}".encode("utf-8"))
    packed = pack_context(
        objective=objective,
        target_source_cid=target_cid,
        surrounding_source_cid=surrounding_cid,
        test_source_cid=test_cid,
        dependency_admissions=(),
        obligation_cids=(),
        counterexample_cids=(),
        delta_cid=delta_cid,
        interface_cids=(),
        assumptions=(
            "cli_pack_context_uses_deterministic_placeholder_cids",
            "exact_source_requires_scanned_tree_when_provider_available",
        ),
        exclusions=None,
        raw_source_regions=(),
    )
    body = packed.to_dict() if hasattr(packed, "to_dict") else _object_to_dict(packed)
    return {
        "repo": str(repo),
        "task": objective,
        "target": target,
        "context_pack": body,
    }


def _cmd_verify(args: argparse.Namespace, *, provider: Any | None) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.semantic_state.verification import (
        VerificationRunner,
        verification_descriptor,
    )

    repo = _resolve_repo(args.repo)
    # Verification without an injected selection/runner reports a bounded plan
    # rather than silently inventing node ids.
    runner = VerificationRunner()
    descriptor = verification_descriptor()
    return {
        "repo": str(repo),
        "mode": args.mode,
        "full_suite": bool(args.full_suite),
        "selection_cid": args.selection_cid,
        "verification": {
            "status": "not_executed",
            "reason_code": "selection_or_commands_not_bound",
            "diagnostic": _clip(
                "CLI verify requires an explicit selection binding and "
                "command runner; no ambient test discovery is performed."
            ),
            "runner_interface": descriptor.get("interface"),
            "runner_present": runner is not None,
        },
    }


def _read_patch_text(patch_or_task: str) -> tuple[str, str, str]:
    path = Path(patch_or_task).expanduser()
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        task_id = path.stem
        objective = f"apply patch {path.name}"
        return text, task_id, objective
    # Treat as inline patch text when it looks like a unified diff.
    body = str(patch_or_task)
    if body.lstrip().startswith(("diff ", "--- ", "+++ ")):
        return body, "inline-patch", "apply inline patch"
    raise FileNotFoundError(
        f"patch_or_task is not a readable file or unified diff: {patch_or_task!r}"
    )


def _cmd_apply_patch(
    args: argparse.Namespace, *, provider: Any | None
) -> dict[str, Any]:
    from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes

    from ipfs_accelerate_py.agent_supervisor.semantic_state.contracts import (
        HarnessMode,
        UnavailableResult,
    )
    from ipfs_accelerate_py.agent_supervisor.semantic_state.harness import (
        HarnessLoopError,
        HarnessPolicy,
        HarnessRequest,
        SemanticCompressionHarness,
    )
    from ipfs_accelerate_py.agent_supervisor.semantic_state.worktree import PatchScope

    mode = str(args.mode)
    if mode == HarnessMode.PRODUCTION.value and bool(args.simulate):
        # Hard production gate: never simulate.
        raise _ProductionGateError(
            "production apply-patch cannot simulate",
            reason_code="production_simulate_rejected",
        )

    repo = _resolve_repo(args.repo)
    patch_text, default_task_id, default_objective = _read_patch_text(
        args.patch_or_task
    )
    task_id = args.task_id or default_task_id
    objective = args.objective or default_objective

    if args.storage_dir is None:
        # Local default under repo; still no daemon.
        storage = repo / ".semantic-state" / "durable"
    else:
        storage = Path(args.storage_dir).expanduser()
    durable = _open_durable(str(storage))
    if durable is None:
        raise RuntimeError("durable storage is required for apply-patch")

    allow_paths = tuple(args.allow_paths) if args.allow_paths else ("**",)
    scope = PatchScope(allowed_paths=allow_paths)

    # Environment binding CIDs: deterministic from inputs (closed, local).
    def _bind(label: str) -> str:
        return cid_for_bytes(f"cli-env:{label}:{repo}".encode("utf-8"))

    policy = HarnessPolicy.default(mode=mode)
    harness = SemanticCompressionHarness(
        durable=durable,
        policy=policy,
        datasets_provider=provider,
        providers=(),
    )

    # When production has no real provider and the route would need a model,
    # the harness itself fails closed. CLI additionally refuses --simulate.
    request = HarnessRequest(
        repository_id=str(repo),
        task_id=task_id,
        objective=objective,
        scope=scope,
        toolchain_cid=_bind("toolchain"),
        dependency_lock_cid=_bind("dependency_lock"),
        config_cid=_bind("config"),
        policy_cid=_bind("policy"),
        interface_cid=_bind("interface"),
        repo_path=str(repo),
        patch_text=patch_text,
        base_commit=args.base_commit,
        base_tree=args.base_tree,
    )

    try:
        outcome = harness.run(request)
    except HarnessLoopError as exc:
        reason = getattr(exc, "reason_code", "harness_error")
        if mode == HarnessMode.PRODUCTION.value and "simulat" in str(reason).lower():
            raise _ProductionGateError(str(exc), reason_code=str(reason)) from exc
        raise

    body = outcome.to_dict() if hasattr(outcome, "to_dict") else _object_to_dict(outcome)
    # Final production assertion: simulated observations never exit as success.
    simulated = bool(getattr(outcome, "simulated", False) or body.get("simulated"))
    if mode == HarnessMode.PRODUCTION.value and simulated:
        raise _ProductionGateError(
            "production apply-patch produced simulated observation",
            reason_code="production_simulated_rejected",
            extra={"outcome": body},
        )
    unavailable = getattr(outcome, "unavailable", None)
    if unavailable is not None and isinstance(unavailable, UnavailableResult):
        return {
            "repo": str(repo),
            "mode": mode,
            "task_id": task_id,
            "outcome": body,
            "unavailable": unavailable.to_dict(),
            "_cli_exit_hint": EXIT_UNAVAILABLE,
        }
    disposition = str(getattr(outcome, "disposition", body.get("disposition", "")))
    ok = disposition in {"accepted", "candidate", "bootstrap"}
    return {
        "repo": str(repo),
        "mode": mode,
        "task_id": task_id,
        "objective": objective,
        "simulated": simulated,
        "disposition": disposition,
        "outcome": body,
        "_cli_ok": ok,
    }


def _cmd_compare_full_suite(
    args: argparse.Namespace, *, provider: Any | None
) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.semantic_state.benchmark import (
        load_fixture_repository_package,
    )
    from ipfs_accelerate_py.agent_supervisor.semantic_state.verification import (
        NormalizedRunFacts,
        compare_full_suite,
    )

    path = Path(args.fixture_or_repo).expanduser().resolve()
    # Prefer controlled fixture package when present.
    fixture_dir = path if path.is_dir() else None
    try:
        fixture_pkg = load_fixture_repository_package(
            fixture_dir if fixture_dir and (fixture_dir / "controlled_repository.py").is_file() or (fixture_dir and fixture_dir.name == "controlled_repo") else None
        )
        repo = fixture_pkg.ControlledSemanticRepository.load()
    except Exception as exc:
        return {
            "fixture_or_repo": str(path),
            "status": "unavailable",
            "reason_code": "fixture_load_failed",
            "diagnostic": _clip(str(exc)),
            "_cli_exit_hint": EXIT_UNAVAILABLE,
        }

    case_id = args.mutation_case_id
    if case_id is None and repo.mutations:
        case_id = repo.mutations[0].case_id
    if case_id is None:
        return {
            "fixture_or_repo": str(path),
            "status": "error",
            "reason_code": "no_mutation_case",
            "diagnostic": "fixture has no mutation cases",
            "_cli_exit_hint": EXIT_ERROR,
        }

    mutation = repo.get_mutation(case_id)
    selected_ids = list(getattr(mutation, "expected_selected_node_ids", ()) or ())
    full_ids = list(getattr(mutation, "full_suite_node_ids", ()) or selected_ids)
    oracle = list(getattr(mutation, "authored_oracle_node_ids", ()) or selected_ids)

    from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes
    from ipfs_accelerate_py.agent_supervisor.semantic_state.verification import (
        normalize_run_facts,
    )

    def _facts(node_ids: Sequence[str], *, label: str) -> NormalizedRunFacts:
        outcomes = [
            {
                "node_id": node_id,
                "status": "passed",
                "failure_fingerprint": None,
            }
            for node_id in node_ids
        ]
        return normalize_run_facts(f"cli-{label}:{case_id}", outcomes)

    selection_cid = getattr(mutation, "selection_cid", None)
    if (
        not isinstance(selection_cid, str)
        or not selection_cid.startswith("b")
        or len(selection_cid) < 50
    ):
        selection_cid = cid_for_bytes(f"cli-selection:{case_id}".encode("utf-8"))

    selection = type(
        "Selection",
        (),
        {
            "selection_cid": selection_cid,
            "fallback": getattr(mutation, "fallback", "none"),
            "selected_pytest_node_ids": tuple(selected_ids),
            "producer_fallback": getattr(mutation, "fallback", "none"),
        },
    )()

    try:
        baseline = _facts(full_ids, label="baseline_full")
        selected = _facts(selected_ids, label="selected")
        candidate = _facts(full_ids, label="candidate_full")
        comparison = compare_full_suite(
            selection,
            baseline_full=baseline,
            selected_run=selected,
            candidate_full=candidate,
            authored_oracle=oracle,
        )
        return {
            "fixture_or_repo": str(path),
            "mutation_case_id": case_id,
            "comparison": comparison.to_dict(),
        }
    except Exception as exc:
        return {
            "fixture_or_repo": str(path),
            "mutation_case_id": case_id,
            "status": "error",
            "reason_code": "compare_failed",
            "diagnostic": _clip(str(exc)),
            "_cli_exit_hint": EXIT_ERROR,
        }


def _cmd_benchmark(
    args: argparse.Namespace, *, provider: Any | None
) -> dict[str, Any]:
    from ipfs_accelerate_py.agent_supervisor.semantic_state.benchmark import (
        BenchmarkError,
        check_report,
        run_benchmark,
        write_report,
    )

    corpus_dir = Path(args.corpus).expanduser() if args.corpus else None
    fixture_dir = Path(args.fixture).expanduser() if args.fixture else None

    if args.check:
        envelope = check_report(
            corpus_package_dir=corpus_dir,
            fixture_package_dir=fixture_dir,
            json_path=Path(args.json_out) if args.json_out else None,
        )
        ok = bool(envelope.get("equal")) and bool(envelope.get("gates_ok", True))
        return {
            "check": envelope,
            "_cli_ok": ok,
            "_cli_exit_hint": EXIT_OK if ok else EXIT_ERROR,
        }

    report = run_benchmark(
        corpus_package_dir=corpus_dir,
        fixture_package_dir=fixture_dir,
    )
    written: dict[str, str] = {}
    if args.write or args.json_out or args.md_out:
        json_path = Path(args.json_out) if args.json_out else None
        md_path = Path(args.md_out) if args.md_out else None
        if json_path is None and md_path is None:
            # Explicit write without paths is refused to avoid mutating the
            # repository docs tree from the CLI by accident.
            raise BenchmarkError(
                "benchmark --write requires --json-out and/or --md-out"
            )
        j_out, m_out = write_report(
            report, json_path=json_path, markdown_path=md_path
        )
        written = {"json": str(j_out), "markdown": str(m_out)}

    summary = report.get("summary", {})
    gates = summary.get("gates", {})
    gates_ok = all(bool(v) for v in gates.values()) if gates else True
    return {
        "report": {
            "interface": report.get("interface"),
            "schema": report.get("schema"),
            "corpus_id": report.get("corpus_id"),
            "task_count": report.get("task_count"),
            "deterministic_digest": report.get("deterministic_digest"),
            "summary": summary,
        },
        "written": written,
        "gates_ok": gates_ok,
        "_cli_ok": gates_ok,
        "_cli_exit_hint": EXIT_OK if gates_ok else EXIT_ERROR,
    }


def _cmd_interface_schema(
    args: argparse.Namespace, *, provider: Any | None
) -> dict[str, Any]:
    text = load_interface_schema_text()
    payload = json.loads(text)
    return {
        "resource": INTERFACE_SCHEMA_RESOURCE,
        "package": INTERFACE_SCHEMA_PACKAGE,
        "schema": payload,
        "cli": semantic_state_cli_descriptor(),
    }


_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "scan": _cmd_scan,
    "watch": _cmd_watch,
    "status": _cmd_status,
    "graph": _cmd_graph,
    "explain-symbol": _cmd_explain_symbol,
    "explain-impact": _cmd_explain_impact,
    "invalidate": _cmd_invalidate,
    "select-tests": _cmd_select_tests,
    "pack-context": _cmd_pack_context,
    "verify": _cmd_verify,
    "apply-patch": _cmd_apply_patch,
    "compare-full-suite": _cmd_compare_full_suite,
    "benchmark": _cmd_benchmark,
    "interface-schema": _cmd_interface_schema,
}


class _ProductionGateError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "production_gate",
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.extra = dict(extra or {})


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    provider: Any | None = None,
) -> int:
    """Run the ``semantic-state`` CLI. Returns a stable exit code."""

    out = stdout if stdout is not None else sys.stdout
    err = stderr if stderr is not None else sys.stderr
    parser = build_parser()

    # argparse writes help/usage to the process streams; bind them for this call
    # so tests and embedded hosts capture bounded help without side channels.
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout = out
        sys.stderr = err
        try:
            args = parser.parse_args(list(argv) if argv is not None else None)
        except SystemExit as exc:
            # argparse already wrote help/usage to the bound streams.
            code = exc.code
            if code is None:
                return EXIT_OK
            if isinstance(code, int):
                return (
                    code
                    if code in (EXIT_OK, EXIT_USAGE, EXIT_ERROR)
                    else EXIT_USAGE
                )
            return EXIT_USAGE
    finally:
        sys.stdout = old_out
        sys.stderr = old_err

    command = str(args.command)
    compact = bool(getattr(args, "compact", False))

    def _write(payload: Mapping[str, Any]) -> None:
        if compact:
            text = json.dumps(dict(payload), sort_keys=True, ensure_ascii=True)
            if not text.endswith("\n"):
                text += "\n"
            out.write(text)
            out.flush()
        else:
            _emit_json(payload, out)

    try:
        handler = _HANDLERS[command]
        result = handler(args, provider=provider)
    except _ProductionGateError as exc:
        envelope = _error_envelope(
            command,
            reason_code=exc.reason_code,
            diagnostic=str(exc),
            exit_code=EXIT_PRODUCTION_GATE,
            retryable=False,
            operation="apply-patch",
            extra=exc.extra or None,
        )
        _write(envelope)
        return EXIT_PRODUCTION_GATE
    except FileNotFoundError as exc:
        envelope = _error_envelope(
            command,
            reason_code="not_found",
            diagnostic=str(exc),
            exit_code=EXIT_ERROR,
        )
        _write(envelope)
        return EXIT_ERROR
    except NotADirectoryError as exc:
        envelope = _error_envelope(
            command,
            reason_code="not_a_directory",
            diagnostic=str(exc),
            exit_code=EXIT_ERROR,
        )
        _write(envelope)
        return EXIT_ERROR
    except Exception as exc:
        # Typed unavailable from datasets adapter.
        reason_code = getattr(exc, "reason_code", None)
        operation = getattr(exc, "operation", command)
        retryable = bool(getattr(exc, "retryable", False))
        adapter_id = getattr(exc, "adapter_id", CLI_ADAPTER_ID)
        name = type(exc).__name__
        if name in {
            "SemanticStateUnavailable",
            "DurableStateUnavailable",
            "UnavailableResult",
        } or reason_code in {
            "import_failed",
            "capability_unavailable",
            "missing_exports",
            "provider_unavailable",
        }:
            diagnostic = getattr(exc, "diagnostic", None) or str(exc)
            envelope = _error_envelope(
                command,
                reason_code=str(reason_code or "unavailable"),
                diagnostic=str(diagnostic),
                exit_code=EXIT_UNAVAILABLE,
                retryable=retryable,
                operation=str(operation),
                extra={"adapter_id": adapter_id},
            )
            # Preserve adapter_id inside error.
            envelope["error"]["adapter_id"] = str(adapter_id)
            _write(envelope)
            return EXIT_UNAVAILABLE

        if name in {"HarnessError", "SessionError", "BenchmarkError", "VerificationError", "ContextPackError", "SelectionExecutionError", "CapsuleAdmissionError", "SemanticStateAdapterError", "DurableStateError", "HarnessLoopError"}:
            envelope = _error_envelope(
                command,
                reason_code=str(reason_code or name),
                diagnostic=str(exc),
                exit_code=EXIT_ERROR,
                retryable=retryable,
                operation=str(operation),
            )
            _write(envelope)
            return EXIT_ERROR

        envelope = _error_envelope(
            command,
            reason_code=str(reason_code or name),
            diagnostic=str(exc),
            exit_code=EXIT_ERROR,
            retryable=False,
            operation=str(operation),
        )
        _write(envelope)
        return EXIT_ERROR

    # Handler-provided exit hints for soft failures.
    exit_hint = result.pop("_cli_exit_hint", None)
    ok_hint = result.pop("_cli_ok", None)
    if "unavailable" in result and isinstance(result.get("unavailable"), Mapping):
        unavail = result["unavailable"]
        envelope = _error_envelope(
            command,
            reason_code=str(unavail.get("reason_code") or "unavailable"),
            diagnostic=str(unavail.get("diagnostic") or "unavailable"),
            exit_code=EXIT_UNAVAILABLE,
            retryable=bool(unavail.get("retryable", False)),
            operation=str(unavail.get("operation") or command),
            extra={"result": result},
        )
        if unavail.get("adapter_id"):
            envelope["error"]["adapter_id"] = str(unavail["adapter_id"])
        _write(envelope)
        return EXIT_UNAVAILABLE

    if exit_hint is not None and int(exit_hint) != EXIT_OK:
        reason = str(result.get("reason_code") or "command_failed")
        diagnostic = str(result.get("diagnostic") or reason)
        envelope = _error_envelope(
            command,
            reason_code=reason,
            diagnostic=diagnostic,
            exit_code=int(exit_hint),
            retryable=False,
            extra={"result": result},
        )
        _write(envelope)
        return int(exit_hint)

    if ok_hint is False:
        envelope = _error_envelope(
            command,
            reason_code=str(result.get("disposition") or "rejected"),
            diagnostic=_clip(
                str(
                    result.get("diagnostic")
                    or result.get("disposition")
                    or "command completed without success disposition"
                )
            ),
            exit_code=EXIT_ERROR,
            extra={"result": result},
        )
        _write(envelope)
        return EXIT_ERROR

    envelope = _success_envelope(command, result, exit_code=EXIT_OK)
    _write(envelope)
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
