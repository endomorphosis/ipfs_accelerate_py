"""Local command interface for reversible repository views and revisions.

Commands operate on explicit, hash-bound packages. They do not configure the
provider's terminal tools, run target code, or establish token/proof benefits.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys


MAX_OPERATIONS_BYTES = 1024 * 1024


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _operations(path: Path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_OPERATIONS_BYTES:
            raise ValueError("operations must be a bounded regular file")
        raw = stream.read(MAX_OPERATIONS_BYTES + 1)
    if len(raw) > MAX_OPERATIONS_BYTES:
        raise ValueError("operations exceed the byte limit")
    def nonfinite(_value):
        raise ValueError("nonfinite JSON value")
    value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=nonfinite)
    if type(value) is not list:
        raise ValueError("operations must be a JSON array")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="retain originals and render compact views")
    create.add_argument("repository", type=Path)
    create.add_argument("package", type=Path)
    create.add_argument("--max-files", type=int, default=4096)
    create.add_argument("--max-total-bytes", type=int, default=64 * 1024 * 1024)
    create.add_argument("--max-file-bytes", type=int, default=1024 * 1024)
    for name in ("verify", "list", "read", "restore", "revise"):
        command = commands.add_parser(name)
        command.add_argument("package", type=Path)
        command.add_argument("--expected-manifest-sha256", required=True)
        if name == "list":
            command.add_argument("--prefix", default="")
            command.add_argument("--limit", type=int, default=64)
        if name == "read":
            command.add_argument("path")
        if name == "restore":
            command.add_argument("destination", type=Path)
        if name in {"read", "restore"}:
            command.add_argument("--representation", choices=("original", "compact"),
                                 default="original" if name == "restore" else "compact")
        if name == "revise":
            command.add_argument("output", type=Path)
            command.add_argument("--operations", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        from . import repository_source_view as views
        if args.command == "create":
            result = views.create_repository_view(args.repository, args.package,
                max_files=args.max_files, max_total_bytes=args.max_total_bytes,
                max_file_bytes=args.max_file_bytes)
        else:
            expected = args.expected_manifest_sha256
            if args.command in {"verify", "list"}:
                manifest = views.load_repository_view(args.package,
                    expected_manifest_sha256=expected)
                result = {"schema": "repository-source-command@1", "operation": args.command,
                    "manifest_sha256": expected,
                    "source_tree_sha256": manifest["source_tree_sha256"],
                    "entries_verified": len(manifest["entries"]),
                    "proof_authority": False, "token_savings_measured": False}
                if args.command == "list":
                    if not 1 <= args.limit <= 128 or len(args.prefix.encode()) > 1024:
                        raise ValueError("listing bounds exceeded")
                    rows = [(name, row) for name, row in sorted(manifest["entries"].items())
                            if name.startswith(args.prefix)]
                    result.update(entries={name: row for name, row in rows[:args.limit]},
                                  matched_entries=len(rows), truncated=len(rows) > args.limit)
            elif args.command == "read":
                result = views.read_repository_file(args.package,
                    expected_manifest_sha256=expected, path=args.path,
                    representation=args.representation)
            elif args.command == "restore":
                result = views.restore_repository_view(args.package, args.destination,
                    expected_manifest_sha256=expected, representation=args.representation)
            else:
                from .repository_source_revision import revise_repository_view
                result = revise_repository_view(args.package, args.output,
                    expected_manifest_sha256=expected, operations=_operations(args.operations))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False))
        return 0
    except (ValueError, OSError, RecursionError) as error:
        print(json.dumps({"schema": "repository-source-command-error@1",
            "error": type(error).__name__, "reason": getattr(error, "code", "invalid_request")},
            sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
