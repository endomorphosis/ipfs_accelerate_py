"""Source-current datasets indexing for immutable repository source sessions.

The native datasets owner keeps AST/CID authority. This adapter joins a retained
package to an original materialization and exposes bounded advisory symbols. It
does not execute source, run proofs, train models, or configure provider tools.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import threading

from . import repository_source_view as view

SCHEMA = "repository-source-native-index@1"
MAX_SYMBOLS = 8192
MAX_RESULTS = 1000
MAX_QUERY_BYTES = 4096
MAX_MEMORY_MB = 2048
MAX_GENERATIONS = 128


class RepositorySourceIndexError(ValueError):
    """A content-free failure; old query results are never a refresh fallback."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(value, code):
    if not value:
        raise RepositorySourceIndexError(code)


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def _directory(path):
    path = Path(os.path.abspath(path))
    _require(path.resolve(strict=True) == path and path.is_dir(), "invalid_directory")
    return path


def _without_git(root):
    # Check markers ourselves as well as Git discovery, including broken links.
    _require(not any(os.path.lexists(parent / ".git") for parent in (root, *root.parents)),
             "git_ancestor_not_supported")
    from ipfs_datasets_py.logic.software_contracts.semantic_index.snapshot import _git_root
    _require(_git_root(root) is None, "git_ancestor_not_supported")


def _check_materialization(root, manifest):
    _without_git(root)
    descriptor = view._open_directory(root)
    try:
        entries, root_mode, excluded, _identities = view._scan(descriptor, manifest["limits"])
    finally:
        os.close(descriptor)
    _require(not excluded and root_mode == manifest["root_mode"]
             and entries == view._original_entries(manifest["entries"]),
             "materialization_differs_from_package")


class NativeRepositoryIndex:
    """One exclusive datasets SQL/CAS owner; roots stay outside source checkouts.

    ``scheduler`` is an optional native datasets scheduler. Without one, native
    preparation uses the shared default admission/backoff policy. Its memory
    reservation is cooperative; the separately configured SQL limits constrain
    DuckDB, not the Python parser. No target code is imported or executed.
    """

    def __init__(self, state_dir: Path, *, scheduler=None):
        self._lock = threading.RLock()
        self._closed = False
        self._current = None
        self._cx = None
        self._owner_fd = None
        self._scheduler = scheduler
        self._pid = os.getpid()
        try:
            path = Path(os.path.abspath(state_dir))
            _directory(path.parent)
            path.mkdir(mode=0o700, exist_ok=True)
            self.state_dir = _directory(path)
            info = self.state_dir.stat()
            _require(info.st_uid == os.geteuid() and stat.S_IMODE(info.st_mode) == 0o700,
                     "private_index_state_required")
            self._state_identity = (info.st_dev, info.st_ino)
            self._owner_fd = os.open(path / "owner.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            info = os.fstat(self._owner_fd)
            _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                     and info.st_uid == os.geteuid() and stat.S_IMODE(info.st_mode) == 0o600,
                     "private_index_lock_required")
            try:
                fcntl.flock(self._owner_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RepositorySourceIndexError("index_owner_busy") from error
            for name in ("native.duckdb", "native.duckdb.wal", "cas", "spill"):
                _require(not (path / name).is_symlink(), "invalid_index_state")
            import duckdb
            from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog
            from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
            from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex
            from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
            from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor

            self._cx = duckdb.connect(str(path / "native.duckdb"), config={
                "threads": "1", "memory_limit": "128MB", "max_temp_directory_size": "128MB",
                "temp_directory": str(path / "spill"),
            })
            self._cx.execute("SET enable_external_access = false")
            store = DuckDBASTStore(connection=self._cx)
            artifacts = ImmutableCAS(path / "cas")
            self._ingestor = DuckDBASTIngestor(store=store)
            self._index = RepositoryCodebaseIndex(ingestor=self._ingestor, artifacts=artifacts,
                                                  catalog=CodebaseCatalog(store, artifacts))
            self.repository_id = "repository-source-session:" + _hash(os.fsencode(path))
        except BaseException:
            self.close()
            raise

    def _check_owner(self):
        _require(not self._closed and self._pid == os.getpid(), "index_closed_or_foreign_owner")
        info = self.state_dir.stat()
        _require((info.st_dev, info.st_ino) == self._state_identity
                 and self.state_dir.resolve() == self.state_dir, "index_state_replaced")

    def _controls(self, memory_mb):
        return dict(scheduler=self._scheduler, timeout_seconds=30.0,
                    admission_timeout_seconds=10.0, memory_mb=memory_mb)

    def _join(self, observation, package_manifest, root):
        native = observation.manifest
        _require(native.snapshot.mode == "filesystem", "non_filesystem_snapshot")
        expected = {name: entry for name, entry in package_manifest["entries"].items()
                    if entry["kind"] != "directory"}
        actual = {entry.path: entry for entry in native.snapshot.entries}
        _require(set(actual) == set(expected), "native_snapshot_scope_differs")
        for name, entry in expected.items():
            captured = actual[name]
            if entry["kind"] == "symlink":
                _require(captured.is_opaque and captured.opaque_reason == "symlink_or_nonregular",
                         "native_symlink_scope_differs")
                continue
            _require(captured.source_cid is not None and captured.size_bytes == entry["original_bytes"],
                     "native_source_unavailable")
            if captured.is_opaque:
                # Native snapshots retain an exact CID for undecodable binary
                # bytes, but intentionally do not ingest them into the AST CAS.
                # Bind that opaque inventory row to the retained package without
                # converting it into a parsed or otherwise understood source.
                _require(captured.opaque_reason == "undecodable", "native_source_unavailable")
                parent = view._open_directory((root / name).parent)
                try:
                    raw, _info = view._read_regular(parent, Path(name).name,
                                                   package_manifest["limits"]["max_file_bytes"])
                finally:
                    os.close(parent)
                try:
                    raw.decode("utf-8", "strict")
                except UnicodeDecodeError:
                    pass
                else:
                    raise RepositorySourceIndexError("native_opaque_reason_differs")
                from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
                _require(cid_for_bytes(raw) == captured.source_cid, "native_source_cid_differs")
            else:
                raw = self._index.artifacts.get_bytes(captured.source_cid)
            _require(_hash(raw) == entry["original_sha256"], "native_source_digest_differs")
        _require(len(native.semantic_state.symbols) <= MAX_SYMBOLS, "symbol_population_limit")
        return native

    def refresh(self, materialized_root: Path, manifest: dict) -> dict:
        """Publish a complete native successor only after exact source joining.

        Failed refreshes expose no current symbols. A native transaction may
        already have published when a later freshness fence fails; the owner
        retains that complete historical generation and reobserves before reuse.
        """
        with self._lock:
            self._check_owner()
            self._current = None
            try:
                view._validate_manifest(manifest)
                manifest = copy.deepcopy(manifest)
                root = _directory(materialized_root)
                view._disjoint(root, self.state_dir)
                _check_materialization(root, manifest)
                from ipfs_datasets_py.logic.software_contracts.codebase_ir import CodebaseScanLimits
                from ipfs_datasets_py.logic.software_contracts.semantic_index.snapshot import (
                    _exclusion_raw, _ignored_raw,
                )
                _require(not any(_ignored_raw(os.fsencode(name), _exclusion_raw(None))
                                 for name in manifest["entries"]), "native_excluded_path_not_supported")
                units = [entry for entry in manifest["entries"].values() if entry["kind"] != "directory"]
                maximum = max((entry.get("original_bytes", 0) for entry in units), default=1)
                limits = CodebaseScanLimits(max(1, len(units)), max(1, maximum))
                reservation = max(256, math.ceil(limits.max_entries * limits.max_file_bytes * 16 / (1024 * 1024)))
                _require(reservation <= MAX_MEMORY_MB, "native_index_reservation_limit")
                previous = self._index.current(self.repository_id)
                _require(previous is None or previous.generation < MAX_GENERATIONS, "native_generation_limit")
                digest = _hash(_canonical(manifest))
                operation = "repository-view:" + _hash(_canonical({
                    "package_manifest_sha256": digest,
                    "previous_head": previous.to_dict() if previous is not None else None,
                }))
                before = self._ingestor.stats()
                publication = self._index.prepare_current(root, repository_id=self.repository_id,
                    operation_id=operation, expected_head=previous, limits=limits, **self._controls(reservation))
                observation = self._index.observe_current(root, expected_head=publication.head,
                                                          **self._controls(reservation))
                native = self._join(observation, manifest, root)
                _check_materialization(root, manifest)
                after = self._ingestor.stats()
                receipt = dict(schema=SCHEMA, package_manifest_sha256=digest,
                    source_tree_sha256=manifest["source_tree_sha256"], head=publication.head.to_dict(),
                    semantic_state_cid=native.semantic_state.state_cid, coverage=native.coverage,
                    ast_stats=after, ast_stats_delta={key: after[key] - before.get(key, 0) for key in after},
                    semantic_reconstruction="fresh", source_scope_verified=True,
                    resource_limits=dict(duckdb_threads=1, duckdb_memory="128MB", duckdb_spill="128MB",
                                         admission_memory_mb=reservation, parser_hard_rss_limit=False),
                    authority="structural_only", proof_authority=False, execution_authority=False,
                    completion_authority=False)
                self._current = (root, manifest, publication.head, reservation, receipt)
                return copy.deepcopy(receipt)
            except RepositorySourceIndexError:
                raise
            except Exception as error:
                raise RepositorySourceIndexError("native_index_refresh_failed") from error

    def symbols(self, *, path: str | None = None, query: str = "", limit: int = 100) -> list[dict]:
        """Return bounded current native symbol metadata after freshness checks."""
        with self._lock:
            self._check_owner()
            _require(self._current is not None, "index_not_current")
            _require(type(limit) is int and 1 <= limit <= MAX_RESULTS, "invalid_symbol_limit")
            _require(type(query) is str and len(query) <= MAX_QUERY_BYTES, "invalid_symbol_query")
            try:
                query_bytes = query.encode("utf-8")
            except UnicodeError as error:
                raise RepositorySourceIndexError("invalid_symbol_query") from error
            _require(len(query_bytes) <= MAX_QUERY_BYTES, "invalid_symbol_query")
            if path is not None:
                view._relative(path)
            root, manifest, head, reservation, _receipt = self._current
            try:
                _check_materialization(root, manifest)
                observed = self._index.observe_current(root, expected_head=head, **self._controls(reservation))
                native = self._join(observed, manifest, root)
                rows = []
                for symbol in native.semantic_state.symbols:
                    if path is not None and symbol.module_path != path:
                        continue
                    if query.casefold() not in (symbol.qualified_name + " " + symbol.module_path).casefold():
                        continue
                    entry = manifest["entries"].get(symbol.module_path)
                    _require(entry is not None and entry["kind"] == "file", "symbol_source_outside_package")
                    rows.append(dict(symbol_id=symbol.stable_id, path=symbol.module_path,
                        name=symbol.qualified_name, kind=str(symbol.kind), source_sha256=entry["original_sha256"],
                        source_cid=symbol.source_cid, version_cid=symbol.version_cid,
                        edit_handle_available=False,
                        span=symbol.span.to_dict() if symbol.span is not None else None))
                rows.sort(key=lambda row: (row["path"], row["name"], row["symbol_id"]))
                # Observe again before releasing results; no stale cached row is
                # a successful response if the source/head moved during assembly.
                self._index.observe_current(root, expected_head=head, **self._controls(reservation))
                _check_materialization(root, manifest)
                return rows[:limit]
            except Exception as error:
                self._current = None
                if isinstance(error, RepositorySourceIndexError):
                    raise
                raise RepositorySourceIndexError("native_index_query_failed") from error

    def close(self):
        """Release this owner only; historical source/CAS artifacts are retained."""
        with self._lock:
            self._current = None
            self._closed = True
            try:
                if self._cx is not None:
                    self._cx.close()
            finally:
                self._cx = None
                if self._owner_fd is not None:
                    os.close(self._owner_fd)
                    self._owner_fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
