"""Transactional edits of retained repository originals into a new package.

The parent package and its source checkout are never edited. Compact symbol
replacements use the existing source-backed editor in a private restored tree.
An unchanged compact proposal must match the delivered view exactly: comment or
formatting edits use the explicit full-original-source ``write_file`` operation.
Syntax checks do not execute source or establish behavioral equivalence.
"""
from __future__ import annotations

import ast
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import tokenize

from .code_source_view import CodeSourceView, MAX_AST_NODES, SourceViewError
from .repository_source_view import (
    RepositoryViewError,
    create_repository_view,
    load_repository_view,
    restore_repository_view,
)

SCHEMA = "repository-source-revision@1"
MAX_OPERATIONS = 128
MAX_BATCH_TEXT_BYTES = 8 * 1024 * 1024
_FIELDS = {
    "replace_symbol": ({"operation", "path", "expected_sha256", "symbol", "replacement"}, set()),
    "write_file": ({"operation", "path", "expected_sha256", "content"}, {"mode"}),
    "delete_file": ({"operation", "path", "expected_sha256"}, set()),
    "rename_file": ({"operation", "path", "destination", "expected_sha256"}, set()),
}


class RepositoryRevisionError(RepositoryViewError):
    """A bounded, content-free error code for an uncommitted revision."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _relative(value: object) -> str:
    if (type(value) is not str or not value or len(value) > 1024
            or "\\" in value or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise RepositoryRevisionError("invalid_operation_path")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError:
        raise RepositoryRevisionError("invalid_operation_path") from None
    parts = value.split("/")
    if (len(encoded) > 1024 or len(parts) > 64 or PurePosixPath(value).is_absolute()
            or any(part in {"", ".", ".."} or part.casefold() == ".git" for part in parts)):
        raise RepositoryRevisionError("invalid_operation_path")
    return value


def _overlaps(first: str, second: str) -> bool:
    return first == second or first.startswith(second + "/") or second.startswith(first + "/")


def _disjoint(first: Path, second: Path) -> bool:
    return first != second and first not in second.parents and second not in first.parents


def _output_path(value: Path, package: Path, source_root: Path) -> Path:
    result = Path(os.path.abspath(value))
    if (result.resolve() != result or not result.parent.is_dir()
            or os.path.lexists(result)
            or not _disjoint(result, package) or not _disjoint(result, source_root)):
        raise RepositoryRevisionError("invalid_successor_destination")
    # The private edit tree will live under this parent. It must not be in the
    # source checkout or the parent package either.
    if any(root == result.parent or root in result.parent.parents for root in (package, source_root)):
        raise RepositoryRevisionError("invalid_successor_destination")
    return result


def _text(value: object, limit: int) -> bytes:
    if type(value) is not str:
        raise RepositoryRevisionError("invalid_operation_content")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError:
        raise RepositoryRevisionError("invalid_operation_content") from None
    if len(encoded) > limit:
        raise RepositoryRevisionError("revision_content_limit")
    return encoded


def _python(data: bytes) -> None:
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        if encoding not in {"utf-8", "utf-8-sig"}:
            raise RepositoryRevisionError("unsupported_python_encoding")
        tree = ast.parse(data.decode(encoding), type_comments=True)
        if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
            raise RepositoryRevisionError("revision_ast_limit")
        compile(tree, "<repository-revision>", "exec", dont_inherit=True)
    except (UnicodeError, SyntaxError, RecursionError, MemoryError, ValueError) as exc:
        if isinstance(exc, RepositoryRevisionError):
            raise
        raise RepositoryRevisionError("invalid_python") from None


def _preflight(operations: object, manifest: dict) -> tuple[list[dict], set[str]]:
    if type(operations) is not list or not operations or len(operations) > MAX_OPERATIONS:
        raise RepositoryRevisionError("revision_operation_limit")
    entries = manifest["entries"]
    selected: list[dict] = []
    touched: set[str] = set()
    text_bytes = 0
    for original in operations:
        if type(original) is not dict or type(original.get("operation")) is not str:
            raise RepositoryRevisionError("invalid_operation")
        operation = original["operation"]
        if operation not in _FIELDS:
            raise RepositoryRevisionError("unsupported_operation")
        required, optional = _FIELDS[operation]
        if not required <= original.keys() or original.keys() - required - optional:
            raise RepositoryRevisionError("invalid_operation_fields")
        row = dict(original)
        path = _relative(row["path"])
        paths = [path]
        if operation == "rename_file":
            paths.append(_relative(row["destination"]))
        for candidate in paths:
            if any(_overlaps(candidate, prior) for prior in touched):
                raise RepositoryRevisionError("conflicting_operations")
            touched.add(candidate)
            parts = candidate.split("/")
            for size in range(1, len(parts)):
                parent = entries.get("/".join(parts[:size]))
                if parent is not None and parent["kind"] != "directory":
                    raise RepositoryRevisionError("unsupported_operation_topology")
        existing = entries.get(path)
        expected = row["expected_sha256"]
        if existing is None:
            if operation != "write_file" or expected is not None:
                raise RepositoryRevisionError("missing_operation_source")
        else:
            if existing["kind"] != "file":
                raise RepositoryRevisionError("unsupported_operation_topology")
            if type(expected) is not str or not re.fullmatch(r"[a-f0-9]{64}", expected):
                raise RepositoryRevisionError("invalid_expected_hash")
            if existing["original_sha256"] != expected:
                raise RepositoryRevisionError("stale_source")
        if operation == "rename_file" and row["destination"] in entries:
            raise RepositoryRevisionError("rename_destination_exists")
        if operation == "write_file":
            mode = row.get("mode", existing["mode"] if existing else 0o644)
            if type(mode) is not int or mode < 0 or mode > 0o777:
                raise RepositoryRevisionError("invalid_operation_mode")
            row["mode"] = mode
            row["_content_bytes"] = _text(row["content"], manifest["limits"]["max_file_bytes"])
            text_bytes += len(row["_content_bytes"])
            if path.endswith(".py"):
                _python(row["_content_bytes"])
        if operation == "replace_symbol":
            if type(row["symbol"]) is not str or not row["symbol"] or len(row["symbol"]) > 512:
                raise RepositoryRevisionError("invalid_symbol")
            text_bytes += len(_text(row["replacement"], manifest["limits"]["max_file_bytes"]))
        if text_bytes > MAX_BATCH_TEXT_BYTES:
            raise RepositoryRevisionError("revision_batch_content_limit")
        selected.append(row)
    return selected, touched


def _file(root: Path, path: str) -> tuple[bytes, int]:
    target = root / path
    info = target.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RepositoryRevisionError("unsupported_operation_topology")
    return target.read_bytes(), stat.S_IMODE(info.st_mode)


def _apply(root: Path, operations: list[dict], manifest: dict) -> list[dict]:
    entries = manifest["entries"]
    helper = CodeSourceView(root)
    receipts = []
    for op in operations:
        name, path = op["operation"], op["path"]
        prior = entries.get(path)
        before, before_mode = _file(root, path) if prior else (None, None)
        if prior and _sha(before) != op["expected_sha256"]:
            raise RepositoryRevisionError("stale_source")
        target = root / path
        row = {
            "operation": name,
            "path": path,
            "input_kind": "compact_symbol" if name == "replace_symbol" else
                          "full_original_source" if name == "write_file" else "file_topology",
            "source_sha256": _sha(before) if before is not None else None,
            "source_mode": before_mode,
        }
        if name == "replace_symbol":
            try:
                delivered = helper.read(path, op["symbol"], "compact")
                result = helper.replace(path, op["symbol"], op["expected_sha256"], op["replacement"])
            except SourceViewError as exc:
                raise RepositoryRevisionError(exc.code) from None
            if result["semantic_no_op"] and op["replacement"] != delivered["content"]:
                raise RepositoryRevisionError("ambiguous_symbol_noop_use_write_file")
            row.update(symbol=op["symbol"], semantic_no_op=result["semantic_no_op"],
                       untouched_bytes_preserved=result["untouched_bytes_preserved"])
        elif name == "write_file":
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target.chmod(0o600)
            target.write_bytes(op["_content_bytes"])
            target.chmod(op["mode"])
        elif name == "delete_file":
            target.unlink()
        else:
            destination = root / op["destination"]
            if op["destination"].endswith(".py"):
                _python(before)
            destination.parent.mkdir(parents=True, exist_ok=True)
            target.rename(destination)
            target = destination
            row["destination"] = op["destination"]
        after, after_mode = _file(root, target.relative_to(root).as_posix()) if name != "delete_file" else (None, None)
        row.update(result_sha256=_sha(after) if after is not None else None,
                   result_mode=after_mode,
                   changed=(name == "rename_file" or before != after or before_mode != after_mode))
        receipts.append(row)
    return receipts


def _check_unchanged(root: Path, manifest: dict, touched: set[str]) -> dict:
    files = links = directories = 0
    for path, entry in manifest["entries"].items():
        if path in touched:
            continue
        target = root / path
        info = target.lstat()
        if entry["kind"] == "file":
            data, mode = _file(root, path)
            if _sha(data) != entry["original_sha256"] or mode != entry["mode"]:
                raise RepositoryRevisionError("unchanged_source_mismatch")
            files += 1
        elif entry["kind"] == "directory":
            if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != entry["mode"]:
                raise RepositoryRevisionError("unchanged_directory_mismatch")
            directories += 1
        elif entry["kind"] == "symlink":
            if not stat.S_ISLNK(info.st_mode) or os.readlink(target) != entry["target"]:
                raise RepositoryRevisionError("unchanged_symlink_mismatch")
            links += 1
    if stat.S_IMODE(root.stat().st_mode) != manifest["root_mode"]:
        raise RepositoryRevisionError("unchanged_root_mode_mismatch")
    return {"unchanged_files": files, "unchanged_directories": directories,
            "unchanged_symlinks": links, "unchanged_files_preserved": True,
            "unchanged_topology_preserved": True}


def revise_repository_view(package: Path, output: Path, *, expected_manifest_sha256: str,
                           operations: list[dict]) -> dict:
    """Build an immutable successor, validating an entire batch before publication.

    Every expected hash refers to the parent snapshot. Paths may be touched once
    per batch; renames cannot overwrite or form dependent rename chains. Safe
    untouched symlinks survive, but operations cannot target or traverse links.
    ``write_file`` content is explicitly complete original source, not a compact
    module implicitly expanded into the old comments or formatting.
    """
    manifest = load_repository_view(package, expected_manifest_sha256=expected_manifest_sha256)
    package_path = Path(os.path.abspath(package))
    source_root = Path(manifest["origin_source_root"])
    if not _disjoint(Path(os.path.abspath(output)), Path(manifest["source_root"])):
        raise RepositoryRevisionError("invalid_successor_destination")
    output_path = _output_path(output, package_path, source_root)
    planned, touched = _preflight(operations, manifest)
    with tempfile.TemporaryDirectory(prefix="repository-source-revision-", dir=output_path.parent) as temporary:
        root = Path(temporary) / "repository"
        restore_repository_view(package_path, root, expected_manifest_sha256=expected_manifest_sha256,
                                representation="original")
        # Grant writes only within the private tree, then restore all retained
        # directory modes before sealing the successor package.
        root.chmod(manifest["root_mode"] | 0o700)
        for path, entry in manifest["entries"].items():
            if entry["kind"] == "directory":
                (root / path).chmod(entry["mode"] | 0o700)
        operation_receipts = _apply(root, planned, manifest)
        for path, entry in sorted(manifest["entries"].items(), key=lambda item: item[0].count("/"), reverse=True):
            if entry["kind"] == "directory":
                (root / path).chmod(entry["mode"])
        root.chmod(manifest["root_mode"])
        unchanged = _check_unchanged(root, manifest, touched)
        # Recheck the complete parent before publishing, including blobs not
        # touched by this revision.
        load_repository_view(package_path, expected_manifest_sha256=expected_manifest_sha256)
        successor = create_repository_view(root, output_path, origin_source_root=source_root, **manifest["limits"])
    return {
        "schema": SCHEMA,
        "parent_manifest_sha256": expected_manifest_sha256,
        "successor_manifest_sha256": successor["manifest_sha256"],
        "parent_source_tree_sha256": manifest["source_tree_sha256"],
        "successor_source_tree_sha256": successor["source_tree_sha256"],
        "successor_directory_fsync_complete": successor["directory_fsync_complete"],
        "original_source_root": str(source_root),
        "operation_count": len(operation_receipts),
        "operations": operation_receipts,
        "touched_paths": sorted(touched),
        **unchanged,
        "parent_package_mutated": False,
        "source_checkout_mutated": False,
        "syntax_validation_executes_source": False,
        "behavioral_equivalence_verified": False,
        "proof_authority": False,
        "task_completion_authority": False,
        "token_savings_verified": False,
    }
