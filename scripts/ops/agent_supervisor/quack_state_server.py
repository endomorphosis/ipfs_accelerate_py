#!/usr/bin/env python3
"""Thin ops facade for the loopback Quack state-owner service (DQP-006).

Parses closed subcommands and delegates to
:class:`~ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server.QuackStateServer`.

Cold import and ``--help`` start no process, open no database, and load no
optional providers. Auth tokens are never accepted on argv; only opaque secret
handles may be supplied.

``start`` keeps the process alive until a fenced stop request is observed or
SIGINT/SIGTERM arrives.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2

# Forbidden argv tokens that smuggle credential material.
_FORBIDDEN_ARGV_MARKERS = (
    "--token",
    "--auth-token",
    "--quack-token",
    "--password",
    "--secret",
    "--api-key",
    "--apikey",
    "--authorization",
    "--bearer",
    "--credential",
    "--private-key",
    "--cookie",
)

_SUBCOMMANDS = (
    "start",
    "stop",
    "status",
    "ready",
    "checkpoint",
    "export-identity",
    "reclaim-stale",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_repo_path() -> None:
    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _reject_forbidden_argv(argv: Sequence[str]) -> None:
    lowered = [str(item).strip().lower() for item in argv]
    for item in lowered:
        name = item.split("=", 1)[0]
        if name in _FORBIDDEN_ARGV_MARKERS:
            raise SystemExit(
                f"refusing argv credential flag {name!r}; use --secret-handle only"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quack_state_server",
        description=(
            "Thin facade for the loopback Quack state-owner. "
            "Never accepts raw auth tokens on argv."
        ),
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Path to control.duckdb (required for start/stop/status/ready)",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help="Directory for status, stop control, and secret-handle token files",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host (loopback by default; non-loopback needs reviewed policy)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Bind port (0 allocates an ephemeral loopback port)",
    )
    parser.add_argument(
        "--store-id",
        default="control.duckdb",
        help="Logical store identity",
    )
    parser.add_argument(
        "--repository-id",
        default="",
        help="Repository identity bound into published store identity",
    )
    parser.add_argument(
        "--secret-handle",
        default="",
        help="Opaque secret handle (never a raw token)",
    )
    parser.add_argument(
        "--allow-experimental",
        action="store_true",
        help="Admit experimental Quack capability reports",
    )
    parser.add_argument(
        "--remote-policy-json",
        default=None,
        help="Path to a separately reviewed remote bind policy JSON object",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON on stdout",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    for name in _SUBCOMMANDS:
        sub.add_parser(name, help=f"{name} the Quack state-owner")
    return parser


def _load_remote_policy(path: str | None) -> Any:
    if not path:
        return None
    _ensure_repo_path()
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        RemoteBindPolicy,
    )

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("remote policy must be a JSON object")
    return RemoteBindPolicy(
        policy_id=str(payload.get("policy_id") or ""),
        reviewed_by=str(payload.get("reviewed_by") or ""),
        review_receipt=str(payload.get("review_receipt") or ""),
        allowed_hosts=tuple(payload.get("allowed_hosts") or ()),
        require_tls=bool(payload.get("require_tls", True)),
        notes=str(payload.get("notes") or ""),
    )


def _require_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if not args.database:
        raise SystemExit("--database is required")
    if not args.state_dir:
        raise SystemExit("--state-dir is required")
    return Path(args.database), Path(args.state_dir)


def _build_server(args: argparse.Namespace) -> Any:
    _ensure_repo_path()
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
        build_server,
    )

    database, state_dir = _require_paths(args)
    policy = _load_remote_policy(args.remote_policy_json)
    return build_server(
        database_path=database,
        state_dir=state_dir,
        host=str(args.host),
        port=int(args.port),
        repository_id=str(args.repository_id or ""),
        store_id=str(args.store_id or "control.duckdb"),
        allow_experimental=bool(args.allow_experimental),
        remote_bind_policy=policy,
        secret_handle=str(args.secret_handle or ""),
    )


def _emit(payload: Mapping[str, Any] | Sequence[Any] | str, *, as_json: bool) -> None:
    if as_json or isinstance(payload, (Mapping, list, tuple)):
        text = json.dumps(payload, sort_keys=True, indent=2)
    else:
        text = str(payload)
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")


def _serve_until_stop(server: Any) -> dict[str, Any]:
    """Block while the state-owner is ready; stop on control file or signal."""

    stop_requested = {"value": False}

    def _handle_signal(signum: int, _frame: Any) -> None:
        del signum
        stop_requested["value"] = True

    previous_int = signal.signal(signal.SIGINT, _handle_signal)
    previous_term = signal.signal(signal.SIGTERM, _handle_signal)
    try:
        control_path = server.stop_control_path()
        while server.lifecycle.value == "ready" and not stop_requested["value"]:
            if control_path.is_file():
                break
            time.sleep(0.25)
        return server.stop()
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)


def _write_external_stop_request(args: argparse.Namespace) -> dict[str, Any]:
    """Write a fenced stop request for a live out-of-process owner."""

    database, state_dir = _require_paths(args)
    status_path = Path(state_dir) / "quack-state-server.status.json"
    control_path = Path(state_dir) / "quack-state-server.stop"
    marker_path = Path(database).with_name(f".{Path(database).name}.state-owner.json")

    server_id = ""
    fence_token = ""
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            status = {}
        identity = status.get("identity") if isinstance(status, dict) else None
        if isinstance(identity, dict):
            server_id = str(identity.get("server_id") or "")
    if marker_path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            marker = {}
        if isinstance(marker, dict):
            server_id = server_id or str(marker.get("server_id") or "")
            fence_token = str(marker.get("fence_token") or "")
    if not server_id or not fence_token:
        raise RuntimeError(
            "cannot stop: no live owner marker/status with fence token"
        )
    payload = {
        "schema": "ipfs_accelerate_py/agent-supervisor/quack-stop-request@1",
        "server_id": server_id,
        "fence_token": fence_token,
        "requested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    control_path.parent.mkdir(parents=True, exist_ok=True)
    control_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    try:
        control_path.chmod(0o600)
    except OSError:
        pass
    return {
        "requested": True,
        "server_id": server_id,
        "control_path": str(control_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    try:
        _reject_forbidden_argv(raw)
    except SystemExit as exc:
        message = str(exc)
        if message:
            sys.stderr.write(message + "\n")
        return EXIT_USAGE

    parser = build_parser()
    try:
        args = parser.parse_args(raw)
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE
        return code if code != 0 else EXIT_SUCCESS

    try:
        if args.command == "reclaim-stale":
            database, _state_dir = _require_paths(args)
            _ensure_repo_path()
            from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import (
                reclaim_stale_owner_marker,
            )

            db = Path(database)
            result = reclaim_stale_owner_marker(
                marker_path=db.with_name(f".{db.name}.state-owner.json"),
                lock_path=db.with_name(f".{db.name}.state-owner.lock"),
            )
            _emit(result, as_json=True)
            return (
                EXIT_SUCCESS
                if result.get("reclaimed") or result.get("reason") == "no_marker"
                else EXIT_FAILURE
            )

        if args.command == "stop":
            # Out-of-process fenced stop: write control file for the live owner.
            result = _write_external_stop_request(args)
            _emit(result, as_json=True)
            return EXIT_SUCCESS

        if args.command == "status":
            _ensure_repo_path()
            database, state_dir = _require_paths(args)
            status_path = Path(state_dir) / "quack-state-server.status.json"
            if status_path.is_file():
                payload = json.loads(status_path.read_text(encoding="utf-8"))
                _emit(payload, as_json=True)
                return EXIT_SUCCESS
            _emit(
                {
                    "lifecycle": "stopped",
                    "database_path": str(database),
                    "state_dir": str(state_dir),
                    "ready": False,
                },
                as_json=True,
            )
            return EXIT_SUCCESS

        server = _build_server(args)
        if args.command == "start":
            identity = server.start()
            # Emit identity once, then stay alive as the exclusive owner.
            _emit(identity.to_dict(), as_json=True)
            sys.stdout.flush()
            result = _serve_until_stop(server)
            _emit(result, as_json=True)
            return EXIT_SUCCESS
        if args.command == "ready":
            # Readiness against a live status file when not owning in-process.
            status_path = Path(args.state_dir) / "quack-state-server.status.json"
            if status_path.is_file():
                payload = json.loads(status_path.read_text(encoding="utf-8"))
                if payload.get("lifecycle") == "ready" and payload.get("identity"):
                    identity = payload["identity"]
                    _emit(
                        {
                            "ready": True,
                            "server_id": identity.get("server_id"),
                            "store_id": identity.get("store_id"),
                            "generation": identity.get("generation"),
                            "schema_revision": identity.get("schema_revision"),
                            "schema_fingerprint": identity.get("schema_fingerprint"),
                            "database_uuid": identity.get("database_uuid"),
                            "process_birth_id": identity.get("process_birth_id"),
                            "listen_uri": identity.get("listen_uri"),
                            "secret_handle": identity.get("secret_handle"),
                            "live": True,
                        },
                        as_json=True,
                    )
                    return EXIT_SUCCESS
            _emit({"ready": False}, as_json=True)
            return EXIT_FAILURE
        if args.command == "checkpoint":
            # Checkpoint requires an in-process owner; fail closed otherwise.
            sys.stderr.write(
                "checkpoint requires the live state-owner process control path\n"
            )
            return EXIT_FAILURE
        if args.command == "export-identity":
            status_path = Path(args.state_dir) / "quack-state-server.status.json"
            if not status_path.is_file():
                raise RuntimeError("no status file to export identity from")
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            identity = payload.get("identity")
            if not identity:
                raise RuntimeError("status file has no identity")
            _emit(
                {
                    "export": True,
                    "authority_class": "export",
                    "identity": identity,
                },
                as_json=True,
            )
            return EXIT_SUCCESS
        sys.stderr.write(f"unknown command: {args.command}\n")
        return EXIT_USAGE
    except Exception as exc:
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        return EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
