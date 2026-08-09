#!/usr/bin/env python3
"""Thin ops facade for deterministic control-plane state export (DQP-011).

Delegates to
:class:`~ipfs_accelerate_py.agent_supervisor.task_sources.state_export.StateExporter`.

Cold import and ``--help`` start no process, open no database, and load no
optional providers. Auth tokens are never accepted on argv; only opaque secret
handles may be supplied for repository attach (never written into exports).

Export destinations are non-authoritative projections. Tampering with or
deleting an export cannot change runtime decisions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2

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
    "export",
    "export-all",
    "profiles",
    "round-trip",
    "render",
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
                f"refusing argv credential flag {name!r}; "
                "use --secret-handle only when attaching a repository"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="export_control_plane_state",
        description=(
            "Deterministic Markdown/JSON/JSONL/CSV/Parquet control-plane "
            "exports. Destinations are never authoritative."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    profiles = sub.add_parser(
        "profiles", help="List closed export profiles and formats"
    )
    profiles.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--source-json",
        type=Path,
        help="Path to ExportSourceData JSON (snapshot-bound payload)",
    )
    common.add_argument(
        "--database",
        type=Path,
        help="Optional control.duckdb path for embedded repository export",
    )
    common.add_argument(
        "--store-id",
        default="control.duckdb",
        help="Logical store id when attaching --database (default: control.duckdb)",
    )
    common.add_argument(
        "--secret-handle",
        default="",
        help="Opaque secret handle for Quack attach (never inlined into exports)",
    )
    common.add_argument(
        "--quack-endpoint",
        default="",
        help="Optional Quack endpoint URI for remote repository export",
    )
    common.add_argument(
        "--dry-run",
        action="store_true",
        help="Render only; do not write destinations",
    )

    export_cmd = sub.add_parser(
        "export",
        parents=[common],
        help="Render one profile to a destination",
    )
    export_cmd.add_argument(
        "--profile",
        required=True,
        help="Closed export profile (see 'profiles')",
    )
    export_cmd.add_argument(
        "--destination",
        required=True,
        help="Output file or portable-bundle directory",
    )
    export_cmd.add_argument(
        "--export-id",
        default="",
        help="Optional export id (default derived from profile + destination)",
    )
    export_cmd.add_argument(
        "--cursor",
        type=int,
        default=0,
        help="Pagination cursor (default 0)",
    )
    export_cmd.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Page limit (default exporter DEFAULT_PAGE_LIMIT)",
    )
    export_cmd.add_argument(
        "--domain",
        default="tasks",
        help="Analysis domain for CSV/Parquet (tasks|events)",
    )
    export_cmd.add_argument(
        "--parameters-json",
        type=Path,
        help="Optional extra parameters JSON object",
    )

    render_cmd = sub.add_parser(
        "render",
        parents=[common],
        help="Render one profile to stdout (bytes as UTF-8 text when possible)",
    )
    render_cmd.add_argument("--profile", required=True)
    render_cmd.add_argument("--cursor", type=int, default=0)
    render_cmd.add_argument("--limit", type=int, default=0)
    render_cmd.add_argument("--domain", default="tasks")
    render_cmd.add_argument("--parameters-json", type=Path)

    export_all = sub.add_parser(
        "export-all",
        parents=[common],
        help="Render every closed profile under a destination directory",
    )
    export_all.add_argument(
        "--destination-dir",
        required=True,
        help="Directory that will receive one artifact per profile",
    )

    round_trip = sub.add_parser(
        "round-trip",
        parents=[common],
        help="Verify lossless portable-bundle round-trip byte identity",
    )
    round_trip.add_argument(
        "--destination",
        default="",
        help="Optional durable bundle path for the round-trip",
    )

    return parser


def _load_json_object(path: Path | None, label: str) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SystemExit(f"{label} JSON must be an object: {path}")
    return dict(payload)


def _load_source(args: argparse.Namespace) -> Any:
    from ipfs_accelerate_py.agent_supervisor.task_sources.state_export import (  # noqa: WPS433
        ExportSourceData,
    )

    if args.source_json is not None:
        payload = _load_json_object(args.source_json, "source")
        if payload is None:
            raise SystemExit("--source-json is empty")
        return ExportSourceData.from_dict(payload)

    if args.database is not None or args.quack_endpoint:
        return _load_source_from_repository(args)

    raise SystemExit(
        "provide --source-json or --database / --quack-endpoint"
    )


def _load_source_from_repository(args: argparse.Namespace) -> Any:
    from ipfs_accelerate_py.agent_supervisor.task_sources.state_export import (  # noqa: WPS433
        ExportSourceData,
    )

    # Lazy import repository stack only when an operator requests attach.
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_repository import (  # noqa: WPS433
        open_embedded_repository,
        open_quack_repository,
    )

    secret_handle = str(getattr(args, "secret_handle", "") or "")
    if secret_handle and not (
        secret_handle.startswith("env://")
        or secret_handle.startswith("vault://")
        or secret_handle.startswith("handle:")
        or secret_handle.startswith("secret-handle:")
    ):
        raise SystemExit(
            "--secret-handle must be an opaque handle (env://, vault://, "
            "handle:, secret-handle:), never raw credential material"
        )

    if args.quack_endpoint:
        repo = open_quack_repository(
            str(args.quack_endpoint),
            store_id=str(args.store_id),
            secret_handle=secret_handle,
        )
    else:
        repo = open_embedded_repository(
            Path(args.database),
            store_id=str(args.store_id),
        )
    try:
        return ExportSourceData.from_repository(repo)
    finally:
        close = getattr(repo, "close", None) or getattr(repo, "detach", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass


def _build_request(args: argparse.Namespace, *, destination: str) -> Any:
    from ipfs_accelerate_py.agent_supervisor.task_sources.state_export import (  # noqa: WPS433
        DEFAULT_PAGE_LIMIT,
        ExportRequest,
    )

    parameters = _load_json_object(
        getattr(args, "parameters_json", None), "parameters"
    ) or {}
    limit = int(getattr(args, "limit", 0) or 0)
    if limit <= 0:
        limit = DEFAULT_PAGE_LIMIT
    kwargs: dict[str, Any] = {
        "profile": str(args.profile),
        "destination": destination,
        "cursor": int(getattr(args, "cursor", 0) or 0),
        "limit": limit,
        "domain": str(getattr(args, "domain", "tasks") or "tasks"),
        "parameters": parameters,
    }
    export_id = str(getattr(args, "export_id", "") or "")
    if export_id:
        kwargs["export_id"] = export_id
    return ExportRequest(**kwargs)


def _emit_result(result: Any) -> None:
    sys.stdout.write(
        json.dumps(result.to_dict(), sort_keys=True, indent=2, ensure_ascii=False)
        + "\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    try:
        _reject_forbidden_argv(raw_argv)
    except SystemExit as exc:
        message = str(exc)
        if message:
            print(message, file=sys.stderr)
        return EXIT_USAGE

    parser = build_parser()
    try:
        args = parser.parse_args(raw_argv)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return EXIT_SUCCESS
        return int(code) if isinstance(code, int) else EXIT_USAGE

    if args.command not in _SUBCOMMANDS:
        print(f"unknown command: {args.command}", file=sys.stderr)
        return EXIT_USAGE

    _ensure_repo_path()

    try:
        from ipfs_accelerate_py.agent_supervisor.task_sources.state_export import (  # noqa: WPS433
            ExportProfile,
            StateExporter,
            closed_export_formats,
            closed_export_profiles,
            portable_round_trip_identical,
        )

        if args.command == "profiles":
            payload = {
                "interface": StateExporter.INTERFACE,
                "profiles": list(closed_export_profiles()),
                "formats": list(closed_export_formats()),
                "is_authority": False,
            }
            if args.json:
                sys.stdout.write(
                    json.dumps(payload, sort_keys=True, indent=2) + "\n"
                )
            else:
                sys.stdout.write("profiles:\n")
                for name in payload["profiles"]:
                    sys.stdout.write(f"  - {name}\n")
                sys.stdout.write("formats:\n")
                for name in payload["formats"]:
                    sys.stdout.write(f"  - {name}\n")
            return EXIT_SUCCESS

        exporter = StateExporter()
        source = _load_source(args)

        if args.command == "round-trip":
            dest = str(args.destination or "").strip() or None
            ok = portable_round_trip_identical(
                source, destination=dest
            )
            payload = {
                "round_trip_identical": bool(ok),
                "profile": ExportProfile.PORTABLE_BUNDLE.value,
                "is_authority": False,
            }
            sys.stdout.write(
                json.dumps(payload, sort_keys=True, indent=2) + "\n"
            )
            return EXIT_SUCCESS if ok else EXIT_FAILURE

        if args.command == "export-all":
            results = exporter.export_all_profiles(
                source,
                args.destination_dir,
                write=not args.dry_run,
            )
            payload = {
                "results": [item.to_dict() for item in results],
                "count": len(results),
                "is_authority": False,
            }
            sys.stdout.write(
                json.dumps(payload, sort_keys=True, indent=2) + "\n"
            )
            return EXIT_SUCCESS

        if args.command == "render":
            request = _build_request(args, destination="stdout")
            artifact = exporter.render(source, request)
            # Prefer text for markdown/json/jsonl/csv; binary otherwise.
            if artifact.format.value in {"markdown", "json", "jsonl", "csv"}:
                sys.stdout.write(artifact.body.decode("utf-8"))
            else:
                sys.stdout.buffer.write(artifact.body)
            return EXIT_SUCCESS

        # export
        request = _build_request(args, destination=str(args.destination))
        result = exporter.export(
            source, request, write=not bool(args.dry_run)
        )
        _emit_result(result)
        return EXIT_SUCCESS
    except SystemExit as exc:
        message = str(exc)
        if message:
            print(message, file=sys.stderr)
        code = exc.code
        if code is None:
            return EXIT_FAILURE
        return int(code) if isinstance(code, int) else EXIT_FAILURE
    except Exception as exc:  # noqa: BLE001 - facade maps all failures
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        name = type(exc).__name__
        if "Parameter" in name or "Authority" in name or "Identity" in name:
            return EXIT_USAGE
        return EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
