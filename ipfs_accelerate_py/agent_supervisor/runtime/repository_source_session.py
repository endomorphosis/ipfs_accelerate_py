"""Generation-bound repository tools over original source and its native index.

Compact text is a read representation. Writes target the canonical checkout and
use its whole-file digest; native datasets indexing always consumes original
bytes. The broker executes no repository code and grants no proof authority.
"""
from __future__ import annotations

import copy
import base64
import hashlib
import json
import os
from pathlib import Path
import stat
import threading
import uuid

from . import repository_source_view as views
from .code_source_view import CodeSourceView, SourceViewError

SCHEMA = "supervisor-repository-source-session@1"
CACHE_INVENTORY_SCHEMA = "repository-source-retained-opaque-cache@1"
CACHE_SUMMARY_SCHEMA = "repository-source-retained-cache-summary@1"
_CACHE_COUNTS = ("entry_count", "file_count", "directory_count", "symlink_count", "original_bytes")
_CACHE_AUTHORITY = ("native_indexed", "proof_authority", "execution_authority", "completion_authority")
DEFAULT_LIMITS = dict(max_files=512, max_total_bytes=16 * 1024 * 1024,
                      max_file_bytes=256 * 1024, max_generations=8)
MAX_READ_LINES = 4096
MAX_READ_LINE_BYTES = 65536


class RepositorySourceSessionError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code):
    raise RepositorySourceSessionError(code)


def _integer(value, lower, upper):
    return type(value) is int and lower <= value <= upper


def _outside_git(path):
    # Even a .git file identifies an ancestor worktree. The native snapshotter
    # would otherwise broaden a restored directory to that enclosing checkout.
    return all(not os.path.lexists(parent / ".git") for parent in (path, *path.parents))


class RepositorySourceSession:
    """A bounded, serialized session; private state is owned by the caller.

    At most eight generations are retained by default, including at most 128 MiB
    of original source bytes before representation and native-index overhead.
    ``close`` releases handles; the caller removes its private state directory.
    """

    def __init__(self, workspace: Path, state_dir: Path, *, representation="compact",
                 index_factory=None, max_files=512, max_total_bytes=16 * 1024 * 1024,
                 max_file_bytes=256 * 1024, max_generations=8):
        if not isinstance(representation, str) or representation not in {"original", "compact"}:
            _fail("invalid_representation")
        if not _integer(max_generations, 1, DEFAULT_LIMITS["max_generations"]):
            _fail("invalid_limits")
        for value, key in ((max_files, "max_files"), (max_total_bytes, "max_total_bytes"),
                           (max_file_bytes, "max_file_bytes")):
            if not _integer(value, 1, DEFAULT_LIMITS[key]):
                _fail("invalid_limits")
        self.workspace = views._absolute(workspace)
        self.state_dir = views._absolute(state_dir)
        self.representation = representation
        self.limits = dict(max_files=max_files, max_total_bytes=max_total_bytes,
                           max_file_bytes=max_file_bytes, max_generations=max_generations)
        self.session_id = uuid.uuid4().hex
        self.generation = 0
        self._lock = threading.RLock()
        self._closed = False
        self._refresh_required = True
        self._index = None
        self._manifest = self._capture_receipt = self._index_receipt = None
        self._materialization_receipt = self._package = None
        self._previous_manifest_sha256 = None
        self._validate_locations()
        try:
            if index_factory is None:
                from .repository_source_index import NativeRepositoryIndex
                index_factory = NativeRepositoryIndex
            self._index = index_factory(self.state_dir / "index")
            self._capture()
        except BaseException:
            self.close()
            raise

    def _validate_locations(self):
        views._disjoint(self.workspace, self.state_dir)
        fd = views._open_directory(self.workspace)
        os.close(fd)
        if not _outside_git(self.state_dir):
            _fail("state_inside_git")
        # Opening the parent first forbids symlink components, including a
        # pre-existing state_dir symlink. No caller-owned directory is removed.
        parent = views._open_directory(self.state_dir.parent)
        try:
            try:
                os.mkdir(self.state_dir.name, 0o700, dir_fd=parent)
            except FileExistsError:
                pass
            fd = os.open(self.state_dir.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                         dir_fd=parent)
            try:
                info = os.fstat(fd)
                if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
                    _fail("state_not_private")
                self._state_identity = (info.st_dev, info.st_ino)
                with os.scandir(fd) as iterator:
                    if next(iterator, None) is not None:
                        _fail("state_not_empty")
            finally:
                os.close(fd)
        except OSError:
            _fail("unsafe_state_directory")
        finally:
            os.close(parent)

    def _open(self):
        if self._closed:
            _fail("session_closed")
        fd = self._state_fd()
        os.close(fd)

    def _state_fd(self):
        fd = views._open_directory(self.state_dir)
        info = os.fstat(fd)
        if ((info.st_dev, info.st_ino) != self._state_identity
                or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700):
            os.close(fd)
            _fail("state_replaced")
        return fd

    def _generation(self, generation):
        self._open()
        if not _integer(generation, 1, DEFAULT_LIMITS["max_generations"]):
            _fail("invalid_generation")
        if generation != self.generation:
            _fail("stale_generation")

    def _current(self):
        if self._refresh_required:
            _fail("refresh_required")
        try:
            views.assert_repository_current(self.workspace, self._manifest)
        except (views.RepositorySourceViewError, OSError):
            self._refresh_required = True
            _fail("stale_source")

    def _identity(self):
        receipt = self._capture_receipt
        native = self._index_receipt
        return copy.deepcopy(dict(schema=SCHEMA, session_id=self.session_id, generation=self.generation,
                    package_manifest_sha256=receipt["manifest_sha256"],
                    previous_package_manifest_sha256=self._previous_manifest_sha256,
                    source_tree_sha256=receipt["source_tree_sha256"],
                    datasets_head=native["head"], semantic_state_cid=native["semantic_state_cid"],
                    index_coverage=native["coverage"], representation=self.representation,
                    proof_authority=False, execution_authority=False, completion_authority=False))

    def _capacity(self):
        if self.generation >= self.limits["max_generations"]:
            _fail("generation_limit")

    def _capture(self):
        self._capacity()
        self._refresh_required = True
        # Failed candidates are removed. Only fully indexed, source-current
        # captures advance the public generation or expose new query handles.
        name = "generation-" + uuid.uuid4().hex
        candidate = self.state_dir / name
        state_fd = self._state_fd()
        os.mkdir(name, 0o700, dir_fd=state_fd)
        package, materialized = candidate / "package", candidate / "original"
        try:
            receipt = views.create_repository_view(
                self.workspace, package,
                **{key: self.limits[key] for key in ("max_files", "max_total_bytes", "max_file_bytes")})
            manifest = views.load_repository_view(package, expected_manifest_sha256=receipt["manifest_sha256"])
            materialization = views.restore_repository_view(
                package, materialized, expected_manifest_sha256=receipt["manifest_sha256"],
                representation="original")
            native = self._index.refresh(materialized, manifest)
            if (not isinstance(native, dict) or "head" not in native
                    or "semantic_state_cid" not in native or "coverage" not in native
                    or native.get("source_tree_sha256") != receipt["source_tree_sha256"]
                    or native.get("package_manifest_sha256") != receipt["manifest_sha256"]
                    or native.get("source_scope_verified") is not True):
                _fail("invalid_index_receipt")
            views.assert_repository_current(self.workspace, manifest)
            check_fd = self._state_fd()
            os.close(check_fd)
        except BaseException:
            views._cleanup(state_fd, name)
            raise
        finally:
            os.close(state_fd)
        self._previous_manifest_sha256 = (self._capture_receipt or {}).get("manifest_sha256")
        self._capture_receipt, self._manifest = receipt, manifest
        self._index_receipt, self._materialization_receipt = native, materialization
        self._package = package
        self.generation += 1
        self._refresh_required = False

    def _retained_cache(self):
        cache = self._index_receipt.get("retained_opaque_cache")
        if (type(cache) is not dict or cache.get("schema") != CACHE_INVENTORY_SCHEMA
                or type(cache.get("entries")) is not dict or type(cache.get("roots")) is not list
                or not views._is_hash(cache.get("inventory_sha256"))
                or any(type(cache.get(key)) is not int or cache[key] < 0 for key in _CACHE_COUNTS)
                or cache["entry_count"] != len(cache["entries"])
                or cache["entry_count"] > self.limits["max_files"]
                or cache["file_count"] + cache["directory_count"] + cache["symlink_count"] != cache["entry_count"]
                or cache["original_bytes"] > self.limits["max_total_bytes"]
                or len(cache["roots"]) > cache["directory_count"]
                or cache.get("classification") != "directory_name_only"
                or any(cache.get(key) is not False for key in _CACHE_AUTHORITY)):
            _fail("cache_inventory_unavailable")
        return cache

    def _status(self, *, summary):
        self._open()
        if not self._refresh_required:
            try:
                self._current()
            except RepositorySourceSessionError:
                pass
        if summary:
            cache = self._retained_cache()
            # Do not copy the full inventory just to discard it. The retained
            # native receipt and package still own every original entry.
            native = copy.deepcopy({key: value for key, value in self._index_receipt.items()
                                    if key != "retained_opaque_cache"})
            native["retained_opaque_cache"] = dict(
                schema=CACHE_SUMMARY_SCHEMA, inventory_schema=cache["schema"],
                inventory_sha256=cache["inventory_sha256"], roots_count=len(cache["roots"]),
                **{key: cache[key] for key in (*_CACHE_COUNTS, "classification", *_CACHE_AUTHORITY)},
                details_tool="repository_cache_inventory")
        else:
            native = copy.deepcopy(self._index_receipt)
        return dict(self._identity(), operation="status", source_current=not self._refresh_required,
                    refresh_required=self._refresh_required, limits=dict(self.limits),
                    capture_counts={key: self._capture_receipt[key] for key in
                                    ("entry_count", "file_count", "original_bytes", "compact_bytes", "compacted_files")},
                    native_index=native)

    def status(self) -> dict:
        """Return complete internal evidence; model delivery uses status_summary."""
        with self._lock:
            return self._status(summary=False)

    def status_summary(self) -> dict:
        """Report health and cache counts without enumerating retained paths.

        Stale sessions remain inspectable and explicitly require refresh. The
        inventory digest identifies the last retained generation, never a claim
        that an externally modified source tree is still current.
        """
        with self._lock:
            return self._status(summary=True)

    def cache_inventory(self, *, generation: int, expected_inventory_sha256: str,
                        offset: int = 0, limit: int = 100) -> dict:
        """Read exact retained cache metadata in generation/digest-bound pages.

        ``cache_root`` marks roots on their directory rows, so collecting every
        page can reconstruct the original inventory identity without sending
        an unbounded roots list beside each page. No source bytes are returned.
        """
        with self._lock:
            self._generation(generation)
            if not views._is_hash(expected_inventory_sha256):
                _fail("invalid_inventory_hash")
            if not _integer(offset, 0, self.limits["max_files"]) or not _integer(limit, 1, 100):
                _fail("invalid_pagination")
            self._current()
            cache = self._retained_cache()
            if expected_inventory_sha256 != cache["inventory_sha256"]:
                _fail("stale_inventory_hash")
            paths = sorted(cache["entries"])
            if offset > len(paths):
                _fail("invalid_pagination")
            roots = set(cache["roots"])
            rows = [dict(path=path, **copy.deepcopy(cache["entries"][path]), cache_root=path in roots)
                    for path in paths[offset:offset + limit]]
            self._current()
            end = offset + len(rows)
            return dict(self._identity(), operation="cache_inventory", inventory_schema=cache["schema"],
                        inventory_sha256=cache["inventory_sha256"], roots_count=len(roots),
                        total=len(paths), offset=offset, entries=rows,
                        next_offset=end if end < len(paths) else None)

    def list_files(self, *, generation: int, offset: int = 0, limit: int = 100) -> dict:
        with self._lock:
            self._generation(generation)
            if not _integer(offset, 0, self.limits["max_files"]) or not _integer(limit, 1, 100):
                _fail("invalid_pagination")
            self._current()
            paths = sorted(self._manifest["entries"])
            rows = [dict(path=path, **self._manifest["entries"][path]) for path in paths[offset:offset + limit]]
            self._current()
            end = offset + len(rows)
            return dict(self._identity(), operation="list_files", entries=rows, total=len(paths),
                        next_offset=end if end < len(paths) else None)

    def symbols(self, *, generation: int, path: str | None = None, query: str = "", limit: int = 100) -> dict:
        with self._lock:
            self._generation(generation)
            if (not isinstance(query, str) or len(query.encode("utf-8")) > 1024
                    or not _integer(limit, 1, 100)):
                _fail("invalid_query")
            if path is not None:
                path = views._relative(path)
            self._current()
            try:
                rows = self._index.symbols(path=path, query=query, limit=limit)
            except Exception:
                self._refresh_required = True
                raise
            self._current()
            return dict(self._identity(), operation="symbols", symbols=rows)

    def read(self, *, generation: int, path: str, representation: str | None = None,
             symbol_id: str | None = None, start_line: int | None = None,
             end_line: int | None = None) -> dict:
        """Read a full file, indexed definition, or exact original LF line range.

        Line coordinates are inclusive and require both endpoints. A range is
        deliberately original-only: compact formatting can change coordinates.
        The canonical whole-file digest and repository freshness fences remain
        the same as a full read; the range does not grant an edit selector.
        """
        with self._lock:
            self._generation(generation)
            self._current()
            representation = self.representation if representation is None else representation
            line_range = start_line is not None or end_line is not None
            if line_range:
                if symbol_id is not None:
                    _fail("invalid_read_selector")
                if (not _integer(start_line, 1, 2**31 - 1)
                        or not _integer(end_line, 1, 2**31 - 1)
                        or start_line > end_line or end_line - start_line + 1 > MAX_READ_LINES):
                    _fail("invalid_line_range")
                if representation != "original":
                    _fail("line_range_requires_original")
            if symbol_id is not None:
                return self._read_symbol(path, symbol_id, representation)
            result = views.read_repository_file(
                self._package, expected_manifest_sha256=self._capture_receipt["manifest_sha256"],
                path=path, representation=representation)
            if line_range:
                if result["encoding"] != "utf-8":
                    _fail("line_range_requires_text")
                # Only LF defines a physical source line. str.splitlines()
                # would also split lone CR and Unicode separators, changing
                # coordinates and losing the exact source-read contract.
                pieces = result["content"].split("\n")
                lines = [piece + "\n" for piece in pieces[:-1]]
                if pieces[-1]:
                    lines.append(pieces[-1])
                if end_line > len(lines):
                    _fail("line_range_out_of_bounds")
                content = "".join(lines[start_line - 1:end_line])
                raw = content.encode("utf-8")
                if len(raw) > min(MAX_READ_LINE_BYTES, self.limits["max_file_bytes"]):
                    _fail("line_range_byte_limit")
                result.update(content=content, view_bytes=len(raw),
                    view_sha256=hashlib.sha256(raw).hexdigest(), scope="line_range",
                    start_line=start_line, end_line=end_line, total_lines=len(lines),
                    omitted_lines=len(lines) - (end_line - start_line + 1))
            self._current()
            identity = self._identity()
            identity["representation"] = result["representation"]
            return dict(result, **identity)

    def _read_symbol(self, path, symbol_id, representation):
        """Resolve native identity against retained original bytes, under lock."""
        from .repository_source_symbol import render_indexed_symbol

        path = views._relative(path)
        if type(representation) is not str or representation not in {"original", "compact"}:
            _fail("invalid_representation")
        try:
            row = self._index.symbol(symbol_id=symbol_id)
        except Exception as error:
            if getattr(error, "preserves_index", False) is not True:
                self._refresh_required = True
            raise
        if type(row) is not dict or row.get("symbol_id") != symbol_id:
            self._refresh_required = True
            _fail("indexed_symbol_source_differs")
        # A caller's mismatched path is a bounded refusal, not index corruption.
        if row["path"] != path:
            _fail("symbol_path_differs")
        retained = views.read_repository_file(
            self._package, expected_manifest_sha256=self._capture_receipt["manifest_sha256"],
            path=path, representation="original")
        data = (retained["content"].encode("utf-8") if retained["encoding"] == "utf-8"
                else base64.b64decode(retained["content"], validate=True))
        try:
            result = render_indexed_symbol(path, data, row, representation)
        except SourceViewError as error:
            if error.code == "indexed_symbol_source_differs":
                self._refresh_required = True
            raise
        self._current()
        identity = self._identity()
        identity["representation"] = result["representation"]
        return dict(result, **identity)

    def _context_call(self, operation, *, generation, max_bytes, **arguments):
        """Join bounded native context to the same generation used for edits."""
        with self._lock:
            self._generation(generation)
            if not _integer(max_bytes, 1024, 65536):
                _fail("invalid_context_limit")
            self._current()
            try:
                # Reserve the generation field's serialized envelope before
                # native selection; never trim a returned capsule or admission.
                result = getattr(self._index, operation)(max_bytes=max_bytes - 64, **arguments)
                if (type(result) is not dict
                        or result.get("source_tree_sha256") != self._capture_receipt["source_tree_sha256"]
                        or result.get("semantic_state_cid") != self._index_receipt["semantic_state_cid"]
                        or result.get("authority") != "structural_only"
                        or any(result.get(key) is not False for key in (
                            "proof_authority", "execution_authority", "completion_authority",
                            "semantic_equivalence_claimed"))):
                    _fail("invalid_context_binding")
                result = dict(result, generation=self.generation)
                if len(json.dumps(result, ensure_ascii=False, separators=(",", ":"),
                                  allow_nan=False).encode("utf-8")) > max_bytes:
                    _fail("context_response_limit")
            except Exception as error:
                # Only the native owner can distinguish a bounded optional
                # refusal from loss of its current source/head integrity.
                if getattr(error, "preserves_index", False) is not True:
                    self._refresh_required = True
                raise
            self._current()
            return result

    def context_query(self, *, generation: int, query: str, limit: int = 8,
                      max_bytes: int = 32768) -> dict:
        """Select structural capsule handles; matches do not discharge intent."""
        return self._context_call("context_query", generation=generation,
                                  max_bytes=max_bytes, query=query, limit=limit)

    def capsule_expand(self, *, generation: int, semantic_root_cid: str,
                       stable_symbol_ids: list[str], max_bytes: int = 32768) -> dict:
        """Expand exact native capsules or refuse the complete requested set."""
        return self._context_call("capsule_expand", generation=generation,
            max_bytes=max_bytes, semantic_root_cid=semantic_root_cid,
            stable_symbol_ids=stable_symbol_ids)

    def context_adapter(self, compiler):
        """Use this source owner with the existing bounded planning compiler."""
        from .repository_context_compiler import RepositoryContextAdapter
        with self._lock:
            self._open()
            return RepositoryContextAdapter(self, compiler)

    def refresh(self, *, generation: int) -> dict:
        with self._lock:
            self._generation(generation)
            self._capture()
            return dict(self.status(), operation="refresh")

    def _preflight_edit(self, path, symbol, expected_sha256, replacement, edit_format):
        """Apply the same editor to a disposable original, never the index root."""
        retained = views.read_repository_file(
            self._package, expected_manifest_sha256=self._capture_receipt["manifest_sha256"],
            path=path, representation="original")
        data = (retained["content"].encode("utf-8") if retained["encoding"] == "utf-8"
                else base64.b64decode(retained["content"], validate=True))
        state_fd = self._state_fd()
        name = "proposal-" + uuid.uuid4().hex
        try:
            os.mkdir(name, 0o700, dir_fd=state_fd)
            proposal = self.state_dir / name
            target = proposal / path
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            parent_fd = views._open_directory(target.parent)
            try:
                views._write_new(parent_fd, target.name, data, mode=0o600)
                editor = CodeSourceView(proposal)
                delivered = editor.read(path, symbol, edit_format)
                edit = editor.replace(path, symbol, expected_sha256, replacement, format=edit_format)
                if edit["semantic_no_op"] and replacement != delivered["content"]:
                    _fail("ambiguous_symbol_noop_use_original_edit")
                updated, _ = views._read_regular(parent_fd, target.name, self.limits["max_file_bytes"])
                if (self._capture_receipt["original_bytes"] - len(data) + len(updated)
                        > self.limits["max_total_bytes"]):
                    _fail("total_bytes_limit")
                if views._sha(updated) != edit["result_sha256"]:
                    _fail("proposal_changed")
                return edit["result_sha256"]
            finally:
                os.close(parent_fd)
        finally:
            try:
                views._cleanup(state_fd, name)
            finally:
                os.close(state_fd)

    def replace_symbol(self, *, generation: int, path: str, symbol: str,
                       expected_sha256: str, replacement: str,
                       representation: str | None = None) -> dict:
        with self._lock:
            self._generation(generation)
            self._capacity()  # Never mutate if the next generation cannot fit.
            self._current()
            path = views._relative(path)
            entry = self._manifest["entries"].get(path)
            if not entry or entry["kind"] != "file":
                _fail("not_regular_file")
            if not views._is_hash(expected_sha256):
                _fail("invalid_expected_hash")
            if expected_sha256 != entry["original_sha256"]:
                _fail("stale_source_hash")
            if not isinstance(symbol, str) or not symbol or len(symbol) > 512:
                _fail("invalid_symbol")
            if not isinstance(replacement, str):
                _fail("invalid_replacement")
            try:
                if len(replacement.encode("utf-8")) > self.limits["max_file_bytes"]:
                    _fail("replacement_limit")
            except UnicodeError:
                _fail("invalid_replacement")
            representation = self.representation if representation is None else representation
            if type(representation) is not str or representation not in {"original", "compact"}:
                _fail("invalid_representation")
            edit_format = "compact" if representation == "compact" else "raw"
            proposed_sha256 = self._preflight_edit(path, symbol, expected_sha256, replacement, edit_format)
            self._current()
            try:
                editor = CodeSourceView(self.workspace)
                edit = editor.replace(path, symbol, expected_sha256, replacement, format=edit_format)
            except (SourceViewError, OSError, UnicodeError, ValueError):
                # The editor can raise after atomic replacement. Only a fresh
                # full-tree observation can establish that bytes are unchanged.
                try:
                    views.assert_repository_current(self.workspace, self._manifest)
                except (views.RepositorySourceViewError, OSError):
                    self._refresh_required = True
                    return dict(self._identity(), operation="replace_symbol", path=path,
                                changed=None, canonical_changed=None, canonical_change="uncertain", refresh_required=True,
                                error="edit_outcome_uncertain")
                raise
            edit.pop("refreshed_view", None)
            if edit["result_sha256"] != proposed_sha256:
                self._refresh_required = True
                return dict(self._identity(), operation="replace_symbol", path=path,
                            changed=None, canonical_changed=None, canonical_change="uncertain",
                            refresh_required=True, error="edit_result_differs_from_proposal")
            try:
                self._capture()
            except Exception:
                self._refresh_required = True
                return dict(self._identity(), operation="replace_symbol", edit=edit,
                            changed=edit["changed"], canonical_changed=edit["changed"],
                            canonical_change="observed" if edit["changed"] else "none",
                            refresh_required=True, error="index_refresh_failed")
            return dict(self._identity(), operation="replace_symbol", edit=edit,
                        changed=edit["changed"], canonical_changed=edit["changed"],
                        canonical_change="observed" if edit["changed"] else "none",
                        refresh_required=False)

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                if self._index is not None:
                    self._index.close()

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self.close()
