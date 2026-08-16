#!/usr/bin/env python3
"""CLI for deterministic control-plane state export (DQP-011 / StateExporter@1).

Renders Markdown, JSON, JSONL, CSV, or Parquet projections from a snapshot-bound
payload. Exports are non-authoritative: runtime decisions never read them.

By default the CLI loads a portable JSON payload (from a prior export or a
fixture) and re-renders it. When ``--database`` is supplied, it opens an
embedded control-plane repository, freezes one consistent snapshot, and
exports that. The destination is written atomically.

Cold import and ``--help`` open no database and start no process beyond
argument parsing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (  # noqa: E402
    StateSnapshot,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.state_export import (  # noqa: E402
    DEFAULT_PAGE_LIMIT,
    EXPORTER_VERSION,
    STATE_EXPORTER_INTERFACE,
    ExportMediaType,
    ExportView,
    StateExportPayload,
    StateExportRequest,
    StateExporter,
    media_type_from_path,
)


def _load_payload_from_json(path: Path) -> StateExportPayload:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise SystemExit(f"payload root must be an object: {path}")
    # Accept either a portable export envelope or a bare payload with snapshot.
    if "snapshot" in document and (
        document.get("schema", "").endswith("portable-state-export@1")
        or "tasks" in document
    ):
        return StateExportPayload.from_portable_dict(document)
    if "snapshot" in document:
        snapshot = StateSnapshot.from_dict(document["snapshot"])
        return StateExportPayload(
            snapshot=snapshot,
            store_identity=document.get("store_identity") or {},
            generation=document.get("generation") or {},
            tasks=tuple(document.get("tasks") or ()),
            leases=tuple(document.get("leases") or ()),
            events=tuple(document.get("events") or ()),
            commands=tuple(document.get("commands") or ()),
            schema_fingerprint=str(document.get("schema_fingerprint") or ""),
        )
    raise SystemExit(
        f"payload {path} is missing a snapshot binding; "
        "provide a portable-state-export@1 document"
    )


def _load_payload_from_database(database: Path, store_id: str) -> StateExportPayload:
    try:
        from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_repository import (
            open_embedded_repository,
        )
    except ImportError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"control-plane repository unavailable: {exc}") from exc

    # open_embedded_repository attaches; export freezes one snapshot then closes.
    repository = open_embedded_repository(database, store_id=store_id)
    try:
        return StateExportPayload.from_repository(repository)
    finally:
        try:
            repository.close()
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a snapshot-bound control-plane projection "
            f"({STATE_EXPORTER_INTERFACE}, {EXPORTER_VERSION}). "
            "Exports are non-authoritative and never mutate state."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--payload",
        type=Path,
        help="Portable JSON payload path (snapshot-bound fixture or prior export)",
    )
    source.add_argument(
        "--database",
        type=Path,
        help="Embedded control.duckdb path (opens once, freezes snapshot)",
    )
    parser.add_argument(
        "--store-id",
        default="control.duckdb",
        help="Logical store id when using --database (default: control.duckdb)",
    )
    parser.add_argument(
        "--destination",
        "-o",
        required=True,
        type=Path,
        help="Export destination path (written atomically)",
    )
    parser.add_argument(
        "--media-type",
        choices=[item.value for item in ExportMediaType],
        default=None,
        help="Export media type (default: inferred from destination suffix)",
    )
    parser.add_argument(
        "--view",
        choices=[item.value for item in ExportView],
        default=ExportView.PORTABLE.value,
        help="Versioned export view (default: portable)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Pagination offset (default: 0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_PAGE_LIMIT,
        help=f"Pagination limit (default: {DEFAULT_PAGE_LIMIT})",
    )
    parser.add_argument(
        "--domains",
        default="tasks,leases,events,commands",
        help="Comma-separated domains to include (default: tasks,leases,events,commands)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render and print receipt metadata without writing destination",
    )
    parser.add_argument(
        "--print-receipt",
        action="store_true",
        help="Print the export receipt JSON to stdout after success",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.payload is not None:
        payload = _load_payload_from_json(Path(args.payload))
    else:
        payload = _load_payload_from_database(
            Path(args.database),
            store_id=str(args.store_id),
        )

    media_type = (
        ExportMediaType(args.media_type)
        if args.media_type
        else media_type_from_path(args.destination)
    )
    domains = tuple(
        part.strip() for part in str(args.domains).split(",") if part.strip()
    )
    request = StateExportRequest(
        destination=str(args.destination),
        media_type=media_type,
        view=ExportView(args.view),
        offset=int(args.offset),
        limit=int(args.limit),
        domains=domains,
        parameters={"cli": "export_control_plane_state"},
    )

    exporter = StateExporter()
    if args.dry_run:
        artifact = exporter.render(payload, request)
        receipt = exporter.build_receipt(
            payload=payload,
            request=request,
            artifact_digest="",
            artifact_bytes=artifact,
        )
        # Dry-run must not write destination; still emit receipt for operators.
        print(json.dumps(receipt.to_record(), sort_keys=True, indent=2))
        print(
            f"# dry-run bytes={len(artifact)} intentional_loss={receipt.intentional_loss}",
            file=sys.stderr,
        )
        return 0

    receipt = exporter.export(payload, request)
    if args.print_receipt:
        print(json.dumps(receipt.to_record(), sort_keys=True, indent=2))
    else:
        print(
            f"exported {receipt.destination} digest={receipt.artifact_digest} "
            f"export_id={receipt.export_id} intentional_loss={receipt.intentional_loss}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
