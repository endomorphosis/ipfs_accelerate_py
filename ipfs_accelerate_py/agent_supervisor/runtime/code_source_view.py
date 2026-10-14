"""Bounded, source-backed Python views and guarded symbol edits.

Canonical files remain authoritative. Compact views preserve the parsed AST, not
comments or formatting; an unchanged AST round-trips through the retained original
bytes. Edits support unambiguous module definitions and directly nested class
members. Syntax validation is not proof of behavioral equivalence. Locks serialize
cooperating users of this helper; unrelated editors must honor expected hashes.
This module also runs directly as a standard-library-only command line helper.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
import textwrap
import time
import tokenize
import uuid

SCHEMA = "code-source-view@1"
MAX_SOURCE_BYTES = 1024 * 1024
MAX_CONTENT_BYTES = 256 * 1024
MAX_AST_NODES = 50000
MAX_SYMBOLS = 2048
MAX_PATHS = 128
MAX_AUDIT_BYTES = 1024 * 1024


class SourceViewError(ValueError):
    """A bounded, content-free error code suitable for tool transport."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _parse(source: str) -> ast.Module:
    try:
        tree = ast.parse(source, type_comments=True)
        if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
            raise SourceViewError("ast_limit")
        # Parsing alone accepts invalid constructs such as duplicate arguments.
        # Compile for syntax/scope validation only; never execute the code object.
        compile(tree, "<source-view>", "exec", dont_inherit=True)
        return tree
    except (SyntaxError, RecursionError, MemoryError, ValueError) as exc:
        if isinstance(exc, SourceViewError):
            raise
        raise SourceViewError("invalid_python") from None


def _ast_key(tree: ast.AST) -> str:
    try:
        return ast.dump(tree, include_attributes=False)
    except RecursionError:
        raise SourceViewError("ast_depth_limit") from None


def _path(value: str, *, python: bool = True) -> str:
    if (not isinstance(value, str) or not value or len(value) > 1024
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
            or "\\" in value):
        raise SourceViewError("invalid_path")
    parts = value.split("/")
    if (PurePosixPath(value).is_absolute()
            or any(not part or part in {".", ".."} or part.startswith(".") for part in parts)
            or (python and not value.endswith(".py"))):
        raise SourceViewError("unsupported_path")
    return value


@contextmanager
def _parent(root: Path, relative: str):
    """Walk with directory descriptors, refusing every symlink component."""
    descriptors = []
    try:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(fd)
        pieces = relative.split("/")
        for part in pieces[:-1]:
            fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            descriptors.append(fd)
        yield fd, pieces[-1]
    except OSError:
        raise SourceViewError("path_unavailable") from None
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def _read_fd(parent: int, name: str) -> tuple[bytes, os.stat_result]:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise SourceViewError("unsupported_file")
        if before.st_size > MAX_SOURCE_BYTES:
            raise SourceViewError("source_limit")
        chunks = []
        size = 0
        while size <= MAX_SOURCE_BYTES:
            chunk = os.read(fd, min(65536, MAX_SOURCE_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        after = os.fstat(fd)
        if size > MAX_SOURCE_BYTES:
            raise SourceViewError("source_limit")
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise SourceViewError("source_changed_during_read")
        return b"".join(chunks), after
    finally:
        os.close(fd)


def _decode(data: bytes) -> tuple[str, str]:
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
        if encoding not in {"utf-8", "utf-8-sig"}:
            raise SourceViewError("unsupported_encoding")
        return data.decode(encoding), encoding
    except (UnicodeError, SyntaxError):
        raise SourceViewError("unsupported_encoding") from None


def _symbols(tree: ast.Module) -> dict[str, list[ast.AST]]:
    result: dict[str, list[ast.AST]] = {}

    def visit(body, prefix=""):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + node.name
                result.setdefault(name, []).append(node)
                if sum(map(len, result.values())) > MAX_SYMBOLS:
                    raise SourceViewError("symbol_limit")
                if isinstance(node, ast.ClassDef):
                    visit(node.body, name + ".")
    visit(tree.body)
    return result


def _selection(source: str, symbol: str, *, tree=None, symbols=None):
    tree = _parse(source) if tree is None else tree
    if symbol == "module":
        return tree, source, 0, len(source.encode("utf-8")), "", "module"
    if not isinstance(symbol, str) or len(symbol) > 512:
        raise SourceViewError("invalid_symbol")
    symbols = _symbols(tree) if symbols is None else symbols
    candidates = symbols.get(symbol, [])
    if not candidates:
        raise SourceViewError("unsupported_or_missing_symbol")
    if len(candidates) != 1:
        raise SourceViewError("ambiguous_symbol")
    # A duplicated ancestor makes an otherwise unique member ambiguous as well.
    names = symbol.split(".")
    for length in range(1, len(names)):
        if len(symbols.get(".".join(names[:length]), [])) != 1:
            raise SourceViewError("ambiguous_symbol")
    node = candidates[0]
    lines = source.encode("utf-8").splitlines(keepends=True)
    first = min([node.lineno] + [item.lineno for item in node.decorator_list]) - 1
    start = sum(map(len, lines[:first]))
    end = sum(map(len, lines[:node.end_lineno - 1])) + node.end_col_offset
    indent_bytes = lines[first][:node.col_offset]
    if indent_bytes.strip(b" \t"):
        raise SourceViewError("unsupported_symbol_layout")
    indent = indent_bytes.decode("utf-8")
    selected = source.encode("utf-8")[start:end].decode("utf-8")
    selected = textwrap.dedent(selected)
    # Tabs and unusual multiline indentation can prevent a lossless standalone
    # selection. Refuse edits/views instead of guessing at lexical structure.
    standalone = _parse(selected)
    if len(standalone.body) != 1 or _ast_key(standalone.body[0]) != _ast_key(node):
        raise SourceViewError("unsupported_symbol_layout")
    return tree, selected, start, end, indent, type(node).__name__


def render_source_bytes(path: str, data: bytes, symbol: str = "module", format: str = "compact", *, read_only_assignment: bool = False) -> dict:
    """Render already captured bytes without filesystem access or execution.

    Source hashes bind the exact retained bytes; callers own scope, provenance
    and freshness checks. This rendering does not grant proof or edit authority.
    """
    if format not in {"raw", "compact"}:
        raise SourceViewError("invalid_format")
    path = _path(path)
    if type(data) is not bytes or len(data) > MAX_SOURCE_BYTES:
        raise SourceViewError("source_limit")
    source, _encoding = _decode(data)
    if type(read_only_assignment) is not bool:
        raise SourceViewError("invalid_read_scope")
    if read_only_assignment:
        tree = _parse(source)
        candidates = []
        for node in tree.body:
            target = (node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1
                      else node.target if isinstance(node, ast.AnnAssign) else None)
            if isinstance(target, ast.Name) and target.id == symbol:
                candidates.append(node)
        # Definitions/imports or another assignment with this name are not
        # silently disambiguated: the verified producer also checks its facet.
        if len(candidates) != 1 or symbol in _symbols(tree):
            raise SourceViewError("unsupported_or_missing_symbol")
        node = candidates[0]
        selected = ast.get_source_segment(source, node)
        if selected is None or _ast_key(_parse(selected).body[0]) != _ast_key(node):
            raise SourceViewError("unsupported_symbol_layout")
        kind = type(node).__name__
    else:
        _tree, selected, _start, _end, _indent, kind = _selection(source, symbol)
    actual, content, fallback = "raw", selected, None
    if format == "compact":
        try:
            tree = _parse(selected)
            candidate = ast.unparse(tree) + "\n"
            if _ast_key(_parse(candidate)) != _ast_key(tree):
                fallback = "ast_mismatch"
            else:
                actual, content = "compact", candidate
        except (RecursionError, SourceViewError, ValueError):
            fallback = "unsupported_compact_rendering"
    if len(content.encode("utf-8")) > MAX_CONTENT_BYTES:
        raise SourceViewError("view_limit")
    return dict(schema=SCHEMA, operation="read", path=path,
                source_sha256=_sha(data), symbol_id=f"{path}::{symbol}",
                symbol=symbol, kind=kind, requested_format=format, format=actual,
                fallback_reason=fallback, content=content, ast_equivalent=True,
                source_bytes=len(data), selected_source_bytes=len(selected.encode("utf-8")),
                content_bytes=len(content.encode("utf-8")),
                edit_supported=symbol != "module" and not read_only_assignment,
                edit_scope="module_definitions_and_direct_class_members",
                unsupported_edits=["module", "nested_functions", "conditional_definitions", "renames"],
                behavioral_equivalence_verified=False)



class CodeSourceView:
    def __init__(self, root: str | Path, *, audit_store: str | Path | None = None):
        absolute = Path(os.path.abspath(root))
        if absolute.resolve() != absolute or not absolute.is_dir():
            raise SourceViewError("invalid_root")
        self.root = absolute
        self.audit_store = Path(audit_store) if audit_store is not None else None
        self.counts: Counter = Counter()

    def _snapshot(self, path: str):
        path = _path(path)
        with _parent(self.root, path) as (parent, name):
            data, info = _read_fd(parent, name)
        source, encoding = _decode(data)
        return data, info, source, encoding

    def _view(self, path: str, symbol: str, format: str) -> dict:
        data, _info, _source, _encoding = self._snapshot(path)
        return render_source_bytes(path, data, symbol, format)

    def _finish(self, result: dict) -> dict:
        operation = result["operation"]
        self.counts[operation] += 1
        if operation == "read":
            self.counts["read_" + result["format"]] += 1
        if result.get("semantic_no_op"):
            self.counts["no_op"] += 1
        result["operation_counts"] = dict(self.counts)
        result["audit_status"] = "disabled" if self.audit_store is None else "recorded"
        result["response_bytes"] = 0
        for _ in range(6):
            size = len(_json(result)) + 1  # The CLI emits one trailing newline.
            if result["response_bytes"] == size:
                break
            result["response_bytes"] = size
        if self.audit_store is not None:
            result["audit_status"] = self._audit(result)
            for _ in range(6):
                size = len(_json(result)) + 1
                if result["response_bytes"] == size:
                    break
                result["response_bytes"] = size
        return result

    def _audit(self, result: dict) -> str:
        """Content-free best effort evidence; failure never hides a committed edit."""
        fd = None
        try:
            path = Path(os.path.abspath(self.audit_store))
            if path.parent.resolve() != path.parent or not path.parent.is_dir():
                return "unavailable"
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW
                         | os.O_NONBLOCK, 0o600)
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                    or info.st_uid != os.getuid() or info.st_mode & 0o077):
                return "unavailable"
            self._flock(fd)
            receipt = {"schema": "code-source-view-operation@1", "operation_id": uuid.uuid4().hex,
                       "root_sha256": _sha(os.fsencode(self.root)), "operation": result["operation"],
                       "source_sha256": result.get("source_sha256"),
                       "result_sha256": result.get("result_sha256"),
                       "format": result.get("format"), "requested_format": result.get("requested_format"),
                       "content_bytes": result.get("content_bytes"),
                       "response_bytes": result["response_bytes"],
                       "changed": result.get("changed"), "semantic_no_op": result.get("semantic_no_op"),
                       "path_sha256": _sha(result.get("path", "").encode()),
                       "symbol_sha256": _sha(result.get("symbol_id", "").encode()),
                       "task_completion_authority": False}
            payload = _json(receipt) + b"\n"
            if os.fstat(fd).st_size + len(payload) > MAX_AUDIT_BYTES:
                return "capacity_reached"
            if os.write(fd, payload) != len(payload):
                return "unavailable"
            os.fsync(fd)
            return "recorded"
        except (OSError, SourceViewError):
            return "unavailable"
        finally:
            if fd is not None:
                os.close(fd)

    def index(self, paths: list[str]) -> dict:
        if not isinstance(paths, list) or not paths or len(paths) > MAX_PATHS:
            raise SourceViewError("path_count_limit")
        if len(set(paths)) != len(paths):
            raise SourceViewError("duplicate_path")
        files = []
        total_symbols = 0
        total_bytes = 0
        for path in paths:
            data, _info, source, _encoding = self._snapshot(path)
            total_bytes += len(data)
            if total_bytes > 4 * MAX_SOURCE_BYTES:
                raise SourceViewError("index_source_limit")
            tree = _parse(source)
            symbols = _symbols(tree)
            total_symbols += sum(map(len, symbols.values()))
            if total_symbols > MAX_SYMBOLS:
                raise SourceViewError("symbol_limit")
            rows = []
            for name, nodes in symbols.items():
                reason = None
                try:
                    _selection(source, name, tree=tree, symbols=symbols)
                except SourceViewError as exc:
                    reason = exc.code
                rows.append({"symbol": name, "symbol_id": f"{path}::{name}",
                             "kind": type(nodes[0]).__name__, "ambiguous": reason == "ambiguous_symbol",
                             "edit_supported": reason is None, "unsupported_reason": reason})
            files.append({"path": path, "source_sha256": _sha(data), "source_bytes": len(data),
                          "symbols": rows})
        result = dict(schema=SCHEMA, operation="index", files=files,
                      scan_scope="explicit_visible_python_paths", behavioral_equivalence_verified=False)
        if len(_json(result)) > MAX_CONTENT_BYTES:
            raise SourceViewError("view_limit")
        return self._finish(result)

    def read(self, path: str, symbol: str = "module", format: str = "compact") -> dict:
        return self._finish(self._view(path, symbol, format))

    @staticmethod
    def _flock(fd: int):
        deadline = time.monotonic() + 1.0
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise SourceViewError("edit_lock_busy") from None
                time.sleep(0.01)

    @contextmanager
    def _locked(self, path: str):
        directory = Path(tempfile.gettempdir()) / f"code-source-view-{os.getuid()}"
        try:
            directory.mkdir(mode=0o700, exist_ok=True)
            info = directory.lstat()
            if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_mode & 0o077 or directory.resolve() != directory):
                raise SourceViewError("edit_lock_unavailable")
            name = _sha(os.fsencode(self.root) + b"\0" + path.encode()) + ".lock"
            fd = os.open(directory / name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
                         | os.O_NONBLOCK, 0o600)
            try:
                info = os.fstat(fd)
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                        or info.st_nlink != 1 or info.st_mode & 0o077):
                    raise SourceViewError("edit_lock_unavailable")
                self._flock(fd)
                yield
            finally:
                os.close(fd)
        except OSError:
            raise SourceViewError("edit_lock_unavailable") from None

    def replacement_text(self, path: str) -> str:
        """Read an explicitly named, visible snippet; it need not end in .py."""
        with _parent(self.root, _path(path, python=False)) as (parent, name):
            data, _info = _read_fd(parent, name)
        return _decode(data)[0]

    def replace(self, path: str, symbol: str, expected_sha256: str, replacement: str, format: str = "compact") -> dict:
        path = _path(path)
        if format not in {"raw", "compact"}:
            raise SourceViewError("invalid_format")
        if symbol == "module":
            raise SourceViewError("unsupported_module_edit")
        if not isinstance(expected_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_sha256):
            raise SourceViewError("invalid_expected_hash")
        if not isinstance(replacement, str) or len(replacement.encode("utf-8")) > MAX_CONTENT_BYTES:
            raise SourceViewError("replacement_limit")
        with self._locked(path):
            with _parent(self.root, path) as (parent, name):
                original, info = _read_fd(parent, name)
                if _sha(original) != expected_sha256:
                    raise SourceViewError("stale_source")
                source, encoding = _decode(original)
                _tree, selected, start, end, indent, kind = _selection(source, symbol)
                snippet = textwrap.dedent(replacement).strip("\r\n")
                proposal = _parse(snippet)
                prior = _parse(selected)
                if (len(proposal.body) != 1 or type(proposal.body[0]).__name__ != kind
                        or getattr(proposal.body[0], "name", None) != symbol.split(".")[-1]):
                    raise SourceViewError("replacement_symbol_mismatch")
                no_op = _ast_key(proposal) == _ast_key(prior)
                updated = original
                if not no_op:
                    # Normalize snippet newlines only; preserve every byte outside
                    # the selected definition, including trailing inline comments.
                    newline = "\r\n" if b"\r\n" in original else "\n"
                    snippet = snippet.replace("\r\n", "\n").replace("\r", "\n")
                    indented = textwrap.indent(snippet, indent).replace("\n", newline)
                    utf8 = source.encode("utf-8")
                    updated_text = (utf8[:start] + indented.encode("utf-8") + utf8[end:]).decode("utf-8")
                    updated_tree = _parse(updated_text)
                    selected_nodes = _symbols(updated_tree).get(symbol, [])
                    if len(selected_nodes) != 1 or _ast_key(selected_nodes[0]) != _ast_key(proposal.body[0]):
                        raise SourceViewError("replacement_context_mismatch")
                    updated = updated_text.encode(encoding)
                    if len(updated) > MAX_SOURCE_BYTES:
                        raise SourceViewError("source_limit")
                    temporary = ".source-view-" + uuid.uuid4().hex + ".tmp"
                    temp_fd = None
                    try:
                        temp_fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                                          | os.O_NOFOLLOW, 0o600, dir_fd=parent)
                        os.fchmod(temp_fd, stat.S_IMODE(info.st_mode))
                        with os.fdopen(temp_fd, "wb", closefd=False) as stream:
                            stream.write(updated)
                            stream.flush()
                            os.fsync(temp_fd)
                        current, current_info = _read_fd(parent, name)
                        if (_sha(current) != expected_sha256
                                or (current_info.st_dev, current_info.st_ino) != (info.st_dev, info.st_ino)):
                            raise SourceViewError("stale_source")
                        # Ensure a renamed/replaced parent is still the selected
                        # visible directory before publishing through its fd.
                        with _parent(self.root, path) as (current_parent, _):
                            if (os.fstat(current_parent).st_dev, os.fstat(current_parent).st_ino) != (os.fstat(parent).st_dev, os.fstat(parent).st_ino):
                                raise SourceViewError("stale_source")
                        os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
                        os.fsync(parent)
                    finally:
                        if temp_fd is not None:
                            os.close(temp_fd)
                        try:
                            os.unlink(temporary, dir_fd=parent)
                        except FileNotFoundError:
                            pass
                refreshed = self._view(path, symbol, format)
                if refreshed["source_sha256"] != _sha(updated):
                    raise SourceViewError("source_changed_after_edit")
                result = dict(schema=SCHEMA, operation="replace", path=path, symbol=symbol,
                              symbol_id=f"{path}::{symbol}", source_sha256=expected_sha256,
                              result_sha256=_sha(updated), changed=not no_op, semantic_no_op=no_op,
                              untouched_bytes_preserved=True, syntax_validated=True,
                              behavioral_equivalence_verified=False, canonical_source_retained=True,
                              refreshed_view=refreshed)
        return self._finish(result)



def summarize_audit_store(path: str | Path, *, expected_root: str | Path) -> dict:
    """Aggregate bounded operation evidence without exporting log contents.

    These local receipts demonstrate helper calls, not provider tokens, proof
    validity, or agent task completion. A process with the same OS identity can
    alter them; they are not a security attestation.
    """
    result = dict(schema="code-source-view-usage@1", status="observed", operations=0,
                  operation_counts={}, format_counts={}, response_bytes=0,
                  content_bytes=0, changed_edits=0, no_op_edits=0,
                  raw_content_exported=False, provider_tokens_observed=False,
                  proof_authority=False, task_completion_authority=False)
    fd = None
    try:
        path = Path(os.path.abspath(path))
        root = Path(os.path.abspath(expected_root))
        if path.parent.resolve() != path.parent or root.resolve() != root:
            raise SourceViewError("invalid_audit_path")
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            result["status"] = "not_observed"
            return result
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                or info.st_uid != os.getuid() or info.st_mode & 0o077
                or info.st_size > MAX_AUDIT_BYTES):
            raise SourceViewError("invalid_audit_file")
        CodeSourceView._flock(fd)
        with os.fdopen(fd, "rb", closefd=False) as stream:
            payload = stream.read(MAX_AUDIT_BYTES + 1)
        if len(payload) > MAX_AUDIT_BYTES or (payload and not payload.endswith(b"\n")):
            raise SourceViewError("invalid_audit_file")
        rows = payload.splitlines()
        if len(rows) > 4096:
            raise SourceViewError("audit_row_limit")
        operations, formats, identifiers = Counter(), Counter(), set()
        fields = {"schema", "operation_id", "root_sha256", "operation", "source_sha256",
                  "result_sha256", "format", "requested_format", "content_bytes",
                  "response_bytes", "changed", "semantic_no_op", "path_sha256",
                  "symbol_sha256", "task_completion_authority"}
        def unique(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise SourceViewError("invalid_audit_row")
                value[key] = item
            return value
        for line in rows:
            if len(line) > 4096:
                raise SourceViewError("invalid_audit_row")
            row = json.loads(line, object_pairs_hook=unique)
            if (not isinstance(row, dict) or set(row) != fields
                    or row["schema"] != "code-source-view-operation@1"
                    or row["root_sha256"] != _sha(os.fsencode(root))
                    or row["task_completion_authority"] is not False
                    or row["operation"] not in {"index", "read", "replace"}
                    or not isinstance(row["operation_id"], str)
                    or not re.fullmatch(r"[a-f0-9]{32}", row["operation_id"])
                    or row["operation_id"] in identifiers):
                raise SourceViewError("invalid_audit_row")
            for key in ("root_sha256", "source_sha256", "result_sha256", "path_sha256", "symbol_sha256"):
                if row[key] is not None and (not isinstance(row[key], str)
                        or not re.fullmatch(r"[a-f0-9]{64}", row[key])):
                    raise SourceViewError("invalid_audit_row")
            if type(row["response_bytes"]) is not int or not 0 <= row["response_bytes"] <= 4 * MAX_SOURCE_BYTES:
                raise SourceViewError("invalid_audit_row")
            content = row["content_bytes"]
            if content is not None and (type(content) is not int or not 0 <= content <= MAX_CONTENT_BYTES):
                raise SourceViewError("invalid_audit_row")
            if row["operation"] == "read":
                if (row["format"] not in {"raw", "compact"}
                        or row["requested_format"] not in {"raw", "compact"}
                        or content is None or row["source_sha256"] is None
                        or row["changed"] is not None or row["semantic_no_op"] is not None):
                    raise SourceViewError("invalid_audit_row")
                formats[row["format"]] += 1
            elif (row["format"] is not None or row["requested_format"] is not None or content is not None):
                raise SourceViewError("invalid_audit_row")
            if row["operation"] == "replace":
                if (type(row["changed"]) is not bool or type(row["semantic_no_op"]) is not bool
                        or row["changed"] == row["semantic_no_op"]
                        or row["source_sha256"] is None or row["result_sha256"] is None):
                    raise SourceViewError("invalid_audit_row")
                result["changed_edits"] += int(row["changed"])
                result["no_op_edits"] += int(row["semantic_no_op"])
            elif row["changed"] is not None or row["semantic_no_op"] is not None:
                raise SourceViewError("invalid_audit_row")
            identifiers.add(row["operation_id"])
            operations[row["operation"]] += 1
            result["response_bytes"] += row["response_bytes"]
            result["content_bytes"] += content or 0
        result.update(operations=len(rows), operation_counts=dict(operations), format_counts=dict(formats))
        if not rows:
            result["status"] = "not_observed"
        return result
    except (OSError, SourceViewError, UnicodeError, ValueError, TypeError):
        return dict(schema="code-source-view-usage@1", status="invalid", operations=None,
                    operation_counts=None, format_counts=None, response_bytes=None,
                    content_bytes=None, changed_edits=None, no_op_edits=None,
                    raw_content_exported=False, provider_tokens_observed=False,
                    proof_authority=False, task_completion_authority=False)
    finally:
        if fd is not None:
            os.close(fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--audit-store")
    commands = parser.add_subparsers(dest="operation", required=True)
    index = commands.add_parser("index")
    index.add_argument("paths", nargs="+")
    read = commands.add_parser("read")
    read.add_argument("path")
    read.add_argument("--symbol", default="module")
    read.add_argument("--format", choices=("raw", "compact"), default="compact")
    edit = commands.add_parser("replace")
    edit.add_argument("path")
    edit.add_argument("--symbol", required=True)
    edit.add_argument("--expected-sha256", required=True)
    edit.add_argument("--replacement-file", required=True)
    edit.add_argument("--format", choices=("raw", "compact"), default="compact")
    args = parser.parse_args(argv)
    try:
        helper = CodeSourceView(args.root, audit_store=args.audit_store)
        if args.operation == "index":
            result = helper.index(args.paths)
        elif args.operation == "read":
            result = helper.read(args.path, args.symbol, args.format)
        else:
            result = helper.replace(args.path, args.symbol, args.expected_sha256,
                                    helper.replacement_text(args.replacement_file), args.format)
        sys.stdout.buffer.write(_json(result) + b"\n")
        return 0
    except (SourceViewError, OSError) as exc:
        error = exc.code if isinstance(exc, SourceViewError) else "filesystem_unavailable"
        sys.stderr.buffer.write(_json({"schema": SCHEMA, "error": error,
                                     "source_content_disclosed": False}) + b"\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
