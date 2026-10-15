"""Live complete-source correspondence for existing scoped header evidence.

This joins an original repository session to an independently signed task
analysis. The filesystem-view digest, local signed-source tree and native task
scope keep their different namespaces. No Git-tree equivalence is inferred.
Checks are sequential observations, not a writer lease or proof authority.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import threading
from weakref import WeakKeyDictionary

from ..proof.formal_verification_contracts import content_identity
from . import repository_source_view as views
from .doctor_scoped_analysis import ScopedDoctorAnalysis, _selection
from .repository_source_index import NativeRepositoryIndex
from .repository_source_session import RepositorySourceSession
from .supervisor_meta_index import public_replay_without_metadata

SCHEMA = "supervisor-repository-proof-source-binding@1"
SOURCE_CORRESPONDENCE = "complete-signed-regular-file-population@1"
MAX_BINDING_BYTES = 16 * 1024 * 1024
_CREATION_LOCK = threading.RLock()
_CREATING = WeakKeyDictionary()


class RepositoryProofSourceBindingError(ValueError):
    """An original source owner, generation or full population refused reuse."""


def _require(value, reason):
    if not value:
        raise RepositoryProofSourceBindingError(reason)


def _json(value):
    try:
        body = json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False)
        _require(len(body.encode("utf-8")) <= MAX_BINDING_BYTES,
                 "source binding exceeds retained byte bound")
        return body
    except (TypeError, ValueError, UnicodeError) as exc:
        if isinstance(exc, RepositoryProofSourceBindingError):
            raise
        raise RepositoryProofSourceBindingError("noncanonical source binding") from exc


def _text(value):
    _require(type(value) is str and value and value == value.strip()
             and len(value.encode("utf-8")) <= 4096
             and not any(ord(character) < 32 for character in value),
             "bounded independently expected identity required")
    return value


def _directory(path):
    _require(type(path) is type(Path()) and path.is_absolute()
             and path.resolve(strict=True) == path, "canonical source directory required")
    info = path.stat(follow_symlinks=False)
    _require(stat.S_ISDIR(info.st_mode), "source directory unavailable")
    return str(path), info.st_dev, info.st_ino


def _fd_identity(descriptor):
    info = os.fstat(descriptor)
    return info.st_dev, info.st_ino, info.st_uid, stat.S_IMODE(info.st_mode)


def _file_identity(path):
    info = path.stat(follow_symlinks=False)
    _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
             and info.st_uid == os.getuid(), "original native index file required")
    return info.st_dev, info.st_ino, info.st_uid, stat.S_IMODE(info.st_mode)


class _SourceGuard:
    def __init__(self, session, scoped, generation, manifest_cid, task_cid, scope_id):
        _require(type(session) is RepositorySourceSession
                 and type(scoped) is ScopedDoctorAnalysis
                 and type(session._index) is NativeRepositoryIndex,
                 "original native repository session and scoped analysis required")
        _require(type(generation) is int and generation >= 1,
                 "exact repository generation required")
        self.session, self.scoped, self.index = session, scoped, session._index
        self.generation = generation
        self.manifest_cid, self.task_cid, self.scope_id = (
            _text(manifest_cid), _text(task_cid), _text(scope_id))
        self._expected = (generation, self.manifest_cid, self.task_cid, self.scope_id)
        self._lock = threading.RLock()
        self._active = False
        self.owner = None
        self.root = _directory(session.workspace)
        _require(scoped.repository == session.workspace
                 and _directory(scoped.repository) == self.root,
                 "proof and session require the same physical canonical repository")
        _require(session.generation == generation and not session._refresh_required
                 and not session._closed and not self.index._closed,
                 "repository generation is not current")
        native = self.index._current
        _require(type(native) is tuple and len(native) == 5,
                 "original complete native index state required")
        self._pins = self._original_pins = (
            session, scoped, self.index, session._manifest, session._capture_receipt,
            session._index_receipt, session._materialization_receipt, native,
            scoped.sources, scoped.snapshot, scoped.diagnostic_snapshot,
            self.index._index, self.index._ingestor, self.index._cx,
            self.index._index.artifacts, self.index._index.catalog,
            scoped.findings,
        )
        self._store = self.index._ingestor.store
        self._cas_limit = self.index._index.artifacts.max_object_bytes
        self._paths = (session.workspace, session.state_dir, session._package,
                       self.index.state_dir, native[0], self.index._index.artifacts.root,
                       self.index._index.artifacts.structured_root,
                       self.index._index.artifacts.source_root)
        self._directories = tuple(_directory(path) for path in self._paths)
        self._scalars = (session.session_id, session.generation, session.representation,
                         session._previous_manifest_sha256, self.index.repository_id,
                         self.index._pid, self.index._owner_fd, self.index._state_identity,
                         session._state_identity)
        self._owner_fd = _fd_identity(self.index._owner_fd)
        self._db_path = self.index.state_dir / "native.duckdb"
        self._db_identity = _file_identity(self._db_path)
        self._sources = dict(scoped.sources)
        _require(all(type(value) is bytes for value in self._sources.values()),
                 "original inert scoped bytes required")
        self._retained_json = self._state_json()
        self._observation_json = None

    def _state_json(self):
        session, scoped, index = self.session, self.scoped, self.index
        native = index._current
        _require(type(native) is tuple and len(native) == 5,
                 "native index state unavailable")
        return _json(dict(
            admission=scoped._admission_json, report=scoped._report_json,
            runtime_snapshot=scoped.snapshot.to_dict(),
            diagnostic_snapshot=scoped.diagnostic_snapshot.to_dict(),
            manifest=session._manifest, capture=session._capture_receipt,
            index_receipt=session._index_receipt,
            materialization=session._materialization_receipt, limits=session.limits,
            native_manifest=native[1], native_head=native[2].to_dict(),
            native_reservation=native[3], native_receipt=native[4],
            scoped_task_cid=scoped.task_cid,
            findings=[finding.to_dict() for finding in scoped.findings],
        ))

    def _retained(self, *, physical=True):
        session, scoped, index = self.session, self.scoped, self.index
        _require(type(self._pins) is tuple and self._pins is self._original_pins
                 and len(self._pins) == 17
                 and type(session) is RepositorySourceSession
                 and type(scoped) is ScopedDoctorAnalysis
                 and type(index) is NativeRepositoryIndex,
                 "original source binding objects changed")
        actual = (session, scoped, session._index, session._manifest, session._capture_receipt,
                  session._index_receipt, session._materialization_receipt, index._current,
                  scoped.sources, scoped.snapshot, scoped.diagnostic_snapshot,
                  index._index, index._ingestor, index._cx,
                  index._index.artifacts, index._index.catalog, scoped.findings)
        _require(all(value is original for value, original in zip(actual, self._pins)),
                 "retained source owner or generation object replaced")
        _require((self.generation, self.manifest_cid, self.task_cid, self.scope_id) == self._expected,
                 "independently expected source identities changed")
        _require((session.session_id, session.generation, session.representation,
                  session._previous_manifest_sha256, index.repository_id, index._pid,
                  index._owner_fd, index._state_identity, session._state_identity) == self._scalars
                 and session.generation == self.generation and not session._closed
                 and not session._refresh_required and not index._closed
                 and index._pid == os.getpid(), "source generation or owner changed")
        paths = (session.workspace, session.state_dir, session._package,
                 index.state_dir, index._current[0], index._index.artifacts.root,
                 index._index.artifacts.structured_root, index._index.artifacts.source_root)
        _require(paths == self._paths
                 and scoped.repository == session.workspace
                 and index._index.ingestor is index._ingestor
                 and index._ingestor.store is self._store
                 and index._index.catalog.store is self._store
                 and self._store._connection is index._cx
                 and index._index.catalog.artifacts is index._index.artifacts
                 and index._index.artifacts.max_object_bytes == self._cas_limit
                 and index.state_dir / "native.duckdb" == self._db_path,
                 "original source directories or native lock changed")
        if physical:
            _require(tuple(_directory(path) for path in paths) == self._directories
                     and _directory(scoped.repository) == self.root
                     and _fd_identity(index._owner_fd) == self._owner_fd
                     and _file_identity(self._db_path) == self._db_identity,
                     "original source directory or lock identity changed")
        _require(self._state_json() == self._retained_json
                 and dict(scoped.sources) == self._sources,
                 "retained source/admission/report/snapshot bytes changed")

    @contextmanager
    def operation(self):
        with self._lock:
            _require(not self._active, "repository proof source binding reentry refused")
            self._active = True
        try:
            yield
        except RepositoryProofSourceBindingError:
            raise
        except Exception as exc:
            raise RepositoryProofSourceBindingError("current repository proof source refused") from exc
        finally:
            with self._lock:
                self._active = False

    def _signed(self):
        self._retained()
        self.scoped.assert_current()
        verified, spec, paths = _selection(
            self.scoped.repository, json.loads(self.scoped._admission_json), self.task_cid)
        self._retained()
        report, manifest = self.scoped.report, verified["manifest"]
        scope = dict(schema="supervisor-doctor-signed-task-scope@1",
                     manifest_cid=verified["receipt"]["manifest_cid"], task_cid=self.task_cid,
                     task_spec_cid=content_identity(spec),
                     source_tree_id=verified["current_source_tree_id"],
                     sources={path: manifest["sources"][path] for path in paths})
        omitted = {path: row for path, row in manifest["sources"].items() if path not in paths}
        roots = self.scoped.snapshot.roots
        _require(scope["manifest_cid"] == self.manifest_cid == report["manifest_cid"]
                 and self.scoped.task_cid == self.task_cid == report["task_cid"]
                 and content_identity(scope) == self.scope_id == report["scope_id"]
                 and report["task_spec_cid"] == scope["task_spec_cid"]
                 and report["source_tree_id"] == scope["source_tree_id"]
                 and roots.repository_id == manifest["repository_cid"]
                 and all(getattr(roots, name) == self.scope_id
                         for name in ("forest_id", "tree_id", "overlay_id"))
                 and tuple(self._sources) == paths
                 and report["source_hashes"] == {
                     path: manifest["sources"][path]["sha256"] for path in paths}
                 and report["source_count"] == len(paths)
                 and report["source_bytes"] == sum(map(len, self._sources.values()))
                 and report["omitted_source_count"] == len(omitted)
                 and report["omitted_source_inventory_cid"] == content_identity({"sources": omitted}),
                 "signed source task scope differs from independently expected identities")
        body = dict(report)
        analysis_cid = body.pop("analysis_cid")
        _require(content_identity(body) == analysis_cid,
                 "scoped analysis observation identity differs")
        return manifest, report

    def _join(self, signed):
        session, index = self.session, self.index
        with session._lock:
            self._retained()
            session._generation(self.generation)
            session._current()
            digest = session._capture_receipt["manifest_sha256"]
            retained = views.load_repository_view(session._package, expected_manifest_sha256=digest)
            receipt = session._index_receipt
            head = index._current[2].to_dict()
            _require(_json(retained) == _json(session._manifest)
                     and _json(retained) == _json(index._current[1])
                     and receipt == index._current[4]
                     and receipt["head"] == head
                     and head["repository_id"] == index.repository_id
                     and receipt["package_manifest_sha256"] == digest
                     and receipt["source_tree_sha256"] == retained["source_tree_sha256"]
                     and receipt["source_scope_verified"] is True
                     and session._capture_receipt["source_tree_sha256"] == retained["source_tree_sha256"]
                     and session._materialization_receipt["manifest_sha256"] == digest
                     and session._materialization_receipt["source_tree_sha256"] == retained["source_tree_sha256"]
                     and session._materialization_receipt["representation"] == "original"
                     and session._materialization_receipt["destination"] == str(index._current[0])
                     and retained["source_root"] == str(session.workspace)
                     and retained["origin_source_root"] == str(session.workspace)
                     and retained["excluded_paths"] == [".git"],
                     "original package or excluded source population differs")
            entries = retained["entries"]
            files = {path: row for path, row in entries.items() if row["kind"] == "file"}
            _require(all(row["kind"] in {"file", "directory"} for row in entries.values())
                     and set(files) == set(signed["sources"]),
                     "complete signed regular-file population differs from session")
            directories = set()
            for path in files:
                parent = PurePosixPath(path).parent
                while parent.as_posix() != ".":
                    directories.add(parent.as_posix())
                    parent = parent.parent
            _require({path for path, row in entries.items() if row["kind"] == "directory"} == directories,
                     "extra directory or unsupported source frontier")
            package_fd, blobs_fd = views._package_fds(session._package)
            try:
                for path, row in files.items():
                    raw = views._read_blob(blobs_fd, row["original_sha256"], row["original_bytes"])
                    parent_fd = views._open_directory((session.workspace / path).parent)
                    try:
                        current, info = views._read_regular(
                            parent_fd, Path(path).name, retained["limits"]["max_file_bytes"])
                    finally:
                        os.close(parent_fd)
                    source = signed["sources"][path]
                    _require(current == raw and hashlib.sha256(raw).hexdigest() == source["sha256"]
                             and len(raw) == row["original_bytes"]
                             and views._mode(info) == row["mode"]
                             and bool(row["mode"] & 0o111) is source["executable"]
                             and (path not in self._sources or raw == self._sources[path]),
                             "complete original bytes or executable bits differ from signed sources")
            finally:
                os.close(blobs_fd)
                os.close(package_fd)
            session._current()
            self._retained()
            return retained, len(files), len(directories)

    def check(self):
        self._retained()
        signed, report = self._signed()
        retained, file_count, directory_count = self._join(signed)
        # Native SQL/CAS/resource callbacks run without a source-server lock.
        # Recheck the signed population and working bytes after they finish.
        self.index.symbols(limit=1)
        self._retained()
        closing_signed, closing_report = self._signed()
        _require(_json(closing_signed) == _json(signed) and closing_report == report,
                 "signed source changed during native observation")
        closing, closing_files, closing_directories = self._join(closing_signed)
        _require(closing == retained and (closing_files, closing_directories) ==
                 (file_count, directory_count), "source correspondence changed during observation")
        observation = dict(
            schema=SCHEMA, source_correspondence=SOURCE_CORRESPONDENCE,
            session_id=self.session.session_id, generation=self.generation,
            session_repository_id=self.index.repository_id,
            proof_repository_id=signed["repository_cid"],
            signed_manifest_cid=self.manifest_cid, task_cid=self.task_cid, scope_id=self.scope_id,
            local_source_tree_id=report["source_tree_id"],
            source_tree_sha256=retained["source_tree_sha256"],
            package_manifest_sha256=self.session._capture_receipt["manifest_sha256"],
            semantic_state_cid=self.session._index_receipt["semantic_state_cid"],
            native_head=self.index._current[2].to_dict(), file_count=file_count,
            directory_count=directory_count,
            source_bytes=sum(row["original_bytes"] for row in retained["entries"].values()
                             if row["kind"] == "file"),
            scoped_file_count=report["source_count"], omitted_source_count=report["omitted_source_count"],
            full_signed_population_joined=True, view_metadata_separately_bound=True,
            git_tree_equivalence_verified=False, proof_authority=False, execution_authority=False,
            mutation_authority=False, completion_authority=False, omission_authority=False,
            whole_program_proved=False,
        )
        body = _json(observation)
        _require(self._observation_json is None or body == self._observation_json,
                 "original source correspondence metadata changed")
        self._retained()
        return body


@dataclass(frozen=True, slots=True, init=False)
class RepositoryProofSourceBinding:
    """Original live source join; detached observations grant no capability."""

    _guard: _SourceGuard = field(repr=False)

    def __init__(self):
        raise RepositoryProofSourceBindingError("use bind_repository_proof_source with existing owners")

    def _original(self):
        _require(type(self) is RepositoryProofSourceBinding
                 and type(self._guard) is _SourceGuard and self._guard.owner is self,
                 "original repository proof source binding required")
        return self._guard

    def assert_bound(self, *, session, scoped_analysis, generation, expected_manifest_cid,
                     expected_task_cid, expected_scope_id):
        """Check original objects/expected joins; this is not currentness."""
        guard = self._original()
        with guard.operation():
            _require(session is guard.session and scoped_analysis is guard.scoped
                     and type(generation) is int
                     and all(type(value) is str for value in
                             (expected_manifest_cid, expected_task_cid, expected_scope_id))
                     and (generation, expected_manifest_cid, expected_task_cid, expected_scope_id) ==
                     guard._expected, "source binding differs from independently admitted inputs")
            guard._retained(physical=False)

    def assert_current(self):
        guard = self._original()
        with guard.operation(), public_replay_without_metadata():
            guard.check()

    def assert_source_current(self):
        """Close over original filesystem bytes without native callbacks.

        This checks retained owners/state and the complete original view after
        a separate final proof-admission fence. It calls no native SQL, CAS,
        scheduler or resource-clock operation and creates no writer lease.
        """
        guard = self._original()
        with guard.operation(), public_replay_without_metadata():
            guard._retained()
            views.assert_repository_current(guard._paths[0], guard._pins[3])
            guard._retained()

    def observation(self):
        """Return detached body-free metadata after a fresh live check."""
        guard = self._original()
        with guard.operation(), public_replay_without_metadata():
            return json.loads(guard.check())


def bind_repository_proof_source(*, session, scoped_analysis, generation,
                                 expected_manifest_cid, expected_task_cid, expected_scope_id):
    """Bind one complete signed current candidate without creating owners.

    Generic Git proof keys, symlink/gitlink populations, additional untracked
    files, nested Git metadata and extra empty directories are unsupported.
    This accepts fully signed current candidate bytes even when they differ
    from the baseline commit; no committed-tree correspondence is claimed.
    """
    _require(type(session) is RepositorySourceSession, "existing original repository session required")
    marker = object()
    with _CREATION_LOCK:
        _require(session not in _CREATING, "repository proof source creation reentry refused")
        _CREATING[session] = marker
    try:
        with public_replay_without_metadata():
            guard = _SourceGuard(session, scoped_analysis, generation,
                                 expected_manifest_cid, expected_task_cid, expected_scope_id)
            result = object.__new__(RepositoryProofSourceBinding)
            object.__setattr__(result, "_guard", guard)
            guard.owner = result
            with guard.operation():
                guard._observation_json = guard.check()
            return result
    except RepositoryProofSourceBindingError:
        raise
    except Exception as exc:
        raise RepositoryProofSourceBindingError("repository proof source admission refused") from exc
    finally:
        with _CREATION_LOCK:
            _require(_CREATING.get(session) is marker, "source creation marker changed")
            del _CREATING[session]
