"""Owner-side bounded views of an already admitted public task source scope.

The caller must authenticate the task and admit the path/hash population before
constructing this service. It adds no store, task authority or worker transport.
Baseline reads require canonical and allocated sources to match admission.
Working reads capture a separate, unverified candidate generation of the same
admitted paths. Original bytes and codec maps stay in owner memory; only requested
slices are returned. Checks are sequential and do not lock out filesystem writers.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import threading

from .reversible_source_view import build_source_view, current_codec_binding, restore_source_view
from .source_formalization_context import _relative, _workspace

PAGE_SCHEMA = "supervisor-worker-source-view-page@1"
DISCOVERY_SCHEMA = "supervisor-worker-source-view-discovery@1"
CURSOR_SCHEMA = "supervisor-worker-source-view-cursor@1"
ADMISSION_SCHEMA = "supervisor-worker-source-view-admission@1"
MAX_SOURCE_BYTES = 1024 * 1024
MAX_FILES = 64
MAX_RESPONSE_BYTES = 65536
MAX_CONTINUATIONS = 256
_FLAGS = dict(read_only=True, proof_authority=False, completion_authority=False,
    execution_authority=False, mutation_authority=False, omission_authority=False,
    semantic_equivalence_claimed=False, source_obligations_discharged=0)


def canonical_json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError("invalid source-view value") from exc


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _need(value, reason):
    if not value:
        raise ValueError(reason)


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _path(value):
    _relative(value)
    _need(".git" not in Path(value).parts, "Git metadata is not public source")
    return value


def _identity(value):
    return value.st_dev, value.st_ino, value.st_mode


def _safe_read(root, expected_root, name):
    """No-follow descriptor walk and full pathname rewalk, never target execution."""
    fds, rewalk = [], []
    fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
    identity = lambda value: tuple(getattr(value, key) for key in fields)
    try:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        fds.append(fd)
        _need(_identity(os.fstat(fd)) == expected_root, "source root identity changed")
        parts = name.split("/")
        for part in parts[:-1]:
            fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=fd)
            fds.append(fd)
        leaf = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=fd)
        fds.append(leaf)
        before = os.fstat(leaf)
        _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
            and before.st_size <= MAX_SOURCE_BYTES, "bounded single-link regular public source required")
        chunks, size = [], 0
        while size <= MAX_SOURCE_BYTES:
            chunk = os.read(leaf, min(65536, MAX_SOURCE_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        after = os.fstat(leaf)
        _need(size <= MAX_SOURCE_BYTES and size == before.st_size and identity(before) == identity(after),
            "source changed during read")
        fresh = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        rewalk.append(fresh)
        for depth, part in enumerate((None, *parts[:-1])):
            if part is not None:
                fresh = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=fresh)
                rewalk.append(fresh)
            _need(_identity(os.fstat(fresh)) == _identity(os.fstat(fds[depth])),
                "source parent pathname changed")
        _need(identity(os.stat(parts[-1], dir_fd=fresh, follow_symlinks=False)) == identity(after),
            "source pathname changed")
        _need(root.resolve(strict=True) == root and _identity(root.lstat()) == expected_root,
            "source root pathname changed")
        return b"".join(chunks)
    except OSError as exc:
        raise ValueError("public source path is unavailable or unsafe") from exc
    finally:
        for descriptor in reversed(rewalk + fds):
            os.close(descriptor)


class WorkerSourceViews:
    """Read-only session service; construction requires a trusted owner admission."""

    def __init__(self, *, repository, workspace, source_hashes, task_id, scope_cid,
                 semantic_root_cid, semantic_sha256, native_prompt_sha256,
                 source_manifest_sha256, profile="python-whitespace@1"):
        _need(type(source_hashes) is dict and 1 <= len(source_hashes) <= MAX_FILES,
            "bounded owner-admitted source hashes required")
        for name, digest in source_hashes.items():
            _path(name)
            _need(_digest(digest), "exact admitted source SHA256 required")
        _need(type(profile) is str and profile in {"identity@1", "python-whitespace@1", "python-comments@1"},
            "unsupported source-view profile")
        for value in (task_id, scope_cid, semantic_root_cid):
            _need(type(value) is str and 0 < len(value.encode()) <= 4096
                and not any(c in value for c in "\0\r\n"), "bounded task/source identity required")
        for value in (semantic_sha256, native_prompt_sha256, source_manifest_sha256):
            _need(_digest(value), "exact admitted semantic context SHA256 required")
        self._root, self._workspace = _workspace(repository, workspace)
        self._roots = {root: _identity(root.lstat()) for root in (self._root, self._workspace)}
        self._hashes = dict(sorted(source_hashes.items()))
        self._profile = profile
        self._codec = canonical_json(current_codec_binding())
        self._scope = dict(schema=ADMISSION_SCHEMA, task_id=task_id, scope_cid=scope_cid,
            semantic_root_cid=semantic_root_cid, semantic_sha256=semantic_sha256,
            native_prompt_sha256=native_prompt_sha256, source_manifest_sha256=source_manifest_sha256,
            source_hashes_sha256=_sha(canonical_json(self._hashes)), source_files=len(self._hashes),
            requested_profile=profile, codec_binding_sha256=_sha(self._codec),
            workspace_binding_sha256=_sha(canonical_json({str(root): list(identity)
                for root, identity in self._roots.items()})))
        self._scope_raw = canonical_json(self._scope)
        self._scope_sha = _sha(self._scope_raw)
        self._continuations = {}
        self._lock = threading.Lock()
        self._working = None
        self._working_counter = 0
        self._working_session = secrets.token_hex(16)
        self._originals = self._check_sources()
        self._build_views()
        self._check_sources()

    def _build_views(self):
        self._views = {}
        for name, original in self._originals.items():
            codec_profile = self._profile if name.endswith(".py") else "identity@1"
            projected = build_source_view(original, profile=codec_profile)
            _need(restore_source_view(original=original, view=projected.view, manifest=projected.manifest,
                expected_manifest_sha256=projected.manifest_sha256) == original,
                "source-view restoration differs")
            self._views[name] = (projected.view, canonical_json(projected.manifest), projected.manifest_sha256)

    def _check_binding(self):
        _need(canonical_json(self._scope) == self._scope_raw and _sha(self._scope_raw) == self._scope_sha
            and _sha(canonical_json(self._hashes)) == self._scope["source_hashes_sha256"]
            and self._profile == self._scope["requested_profile"]
            and _sha(canonical_json({str(root): list(identity) for root, identity in self._roots.items()}))
                == self._scope["workspace_binding_sha256"], "source admission changed")
        _need(canonical_json(current_codec_binding()) == self._codec, "source-view codec changed")
        _need(_workspace(self._root, self._workspace) == (self._root, self._workspace),
            "allocated source repository changed")

    def _capture(self, root, hashes=None):
        captured, total = {}, 0
        for name in self._hashes:
            raw = _safe_read(root, self._roots[root], name)
            total += len(raw)
            _need(total <= MAX_SOURCE_BYTES, "admitted source scope exceeds byte bound")
            if hashes is not None:
                _need(_sha(raw) == hashes[name], "admitted source is stale")
            captured[name] = raw
        return captured

    def _check_baseline(self):
        self._check_binding()
        return self._capture(self._root, self._hashes)

    def _check_sources(self):
        captured = self._check_baseline()
        self._capture(self._workspace, self._hashes)
        return captured

    def baseline_admission(self):
        """Verify immutable canonical custody without adopting workspace edits."""
        with self._lock:
            self._check_baseline()
            return {**json.loads(self._scope_raw), "scope_sha256": self._scope_sha, **_FLAGS}

    def working_admission(self, *, expected_workspace_generation=None):
        """Check a candidate snapshot without refreshing or granting proof reuse."""
        with self._lock:
            candidate = self._working
            _need(candidate is not None, "working source generation unavailable")
            _need(expected_workspace_generation is None
                or expected_workspace_generation == candidate.generation,
                "working source generation differs")
            return candidate.admission()

    def working_page(self, *, path, view="compact", cursor=None, max_bytes=8192,
                     expected_source_sha256=None, expected_view_sha256=None,
                     expected_workspace_generation=None):
        """Read candidate bytes; only an unpinned first page can refresh the cache.

        A generation binds every admitted working path, not just the requested
        file. This is a session-local source observation, never index publication
        or proof refresh. Unadmitted files are neither scanned nor served.
        """
        with self._lock:
            _path(path)
            _need(path in self._hashes, "source path outside admitted public scope")
            pinned = cursor is not None or expected_source_sha256 is not None or expected_view_sha256 is not None
            _need(not pinned or _digest(expected_workspace_generation),
                "working continuation or expansion requires generation")
            _need(expected_workspace_generation is None or _digest(expected_workspace_generation),
                "invalid working source generation")
            self._check_baseline()
            candidate = self._working
            if expected_workspace_generation is None:
                try:
                    captured = self._capture(self._workspace)
                except (OSError, ValueError):
                    self._discard_working()
                    raise
                hashes = {name: _sha(raw) for name, raw in captured.items()}
                if candidate is None or hashes != candidate._hashes:
                    self._discard_working()
                    self._working_counter += 1
                    candidate = _WorkingSourceViews(self, captured, self._working_counter)
                    # Construction checks both sides again after projection.
                    self._working = candidate
            _need(candidate is not None and (expected_workspace_generation is None
                or candidate.generation == expected_workspace_generation), "working source generation differs")
            return candidate.page(path=path, view=view, cursor=cursor, max_bytes=max_bytes,
                expected_source_sha256=expected_source_sha256, expected_view_sha256=expected_view_sha256)

    def _discard_working(self):
        if self._working is not None:
            self._working._continuations.clear()
        self._working = None

    def _page_metadata(self, path):
        return {}

    def admission(self):
        """Small owner descriptor binding; this is not a worker authentication API."""
        with self._lock:
            self._check_sources()
            return {**json.loads(self._scope_raw), "scope_sha256": self._scope_sha, **_FLAGS}

    def _query(self, operation, max_bytes, **fields):
        _need(type(max_bytes) is int and 1024 <= max_bytes <= MAX_RESPONSE_BYTES,
            "invalid complete source-view page byte bound")
        return dict(operation=operation, scope_sha256=self._scope_sha, max_bytes=max_bytes, **fields)

    def _offset(self, cursor, query):
        if cursor is None:
            return 0
        _need(type(cursor) is dict and set(cursor) == {"schema", "request_sha256", "offset", "token"}
            and cursor["schema"] == CURSOR_SCHEMA and _digest(cursor["request_sha256"])
            and type(cursor["offset"]) is int and cursor["offset"] > 0
            and type(cursor["token"]) is str and re.fullmatch(r"[0-9a-f]{32}", cursor["token"]),
            "invalid source-view continuation")
        _need(cursor["request_sha256"] == _sha(canonical_json(query))
            and self._continuations.get(cursor["token"]) == canonical_json(cursor),
            "source-view continuation is foreign, forged or consumed")
        return cursor["offset"]

    def _next(self, query, offset, token):
        return dict(schema=CURSOR_SCHEMA, request_sha256=_sha(canonical_json(query)), offset=offset, token=token)

    def _finish(self, result, cursor, max_bytes):
        _need(len(canonical_json(result)) <= max_bytes, "source-view page exceeds byte bound")
        self._check_sources()
        successor = result["next_cursor"]
        _need(successor is None or len(self._continuations) - (cursor is not None) < MAX_CONTINUATIONS,
            "source-view continuation population exceeds bound")
        if cursor is not None:
            del self._continuations[cursor["token"]]
        if successor is not None:
            self._continuations[successor["token"]] = canonical_json(successor)
        return result

    def page(self, *, path, view="compact", cursor=None, max_bytes=8192,
             expected_source_sha256=None, expected_view_sha256=None):
        """Return one bounded slice; original expansion never applies compact edits.

        A caller expanding a previously observed view should provide both pins.
        Supplying either pin requires the pair. Continuations additionally pin
        the exact representation and whole-page byte budget, and are single-use.
        The transport must reserve its own envelope from ``max_bytes``.
        """
        with self._lock:
            _path(path)
            _need(path in self._hashes, "source path outside admitted public scope")
            _need(type(view) is str and view in {"compact", "original"}, "invalid source representation")
            compact, manifest_raw, manifest_sha = self._views[path]
            manifest = json.loads(manifest_raw)
            original = self._originals[path]
            _need(_sha(original) == self._hashes[path] and _sha(compact) == manifest["view_sha256"]
                and _sha(manifest_raw) == manifest_sha, "retained source-view bytes changed")
            if expected_source_sha256 is not None or expected_view_sha256 is not None:
                _need(_digest(expected_source_sha256) and _digest(expected_view_sha256)
                    and expected_source_sha256 == self._hashes[path]
                    and expected_view_sha256 == manifest["view_sha256"], "requested source/view pins differ")
            query = self._query("read", max_bytes, path=path, view=view,
                source_sha256=self._hashes[path], view_sha256=manifest["view_sha256"])
            offset = self._offset(cursor, query)
            self._check_sources()
            data = compact if view == "compact" else original
            _need(offset == 0 if not data else 0 <= offset < len(data), "source offset outside representation")
            token = secrets.token_hex(16)
            def result(count):
                end = offset + count
                part = data[offset:end]
                try:
                    content, encoding = part.decode("utf-8"), "utf-8"
                except UnicodeDecodeError:
                    content, encoding = base64.b64encode(part).decode("ascii"), "base64"
                binding = {key: self._scope[key] for key in ("task_id", "scope_cid", "semantic_root_cid",
                    "semantic_sha256", "native_prompt_sha256", "source_manifest_sha256")}
                binding["profile"] = self._profile
                return dict(schema=PAGE_SCHEMA, scope_sha256=self._scope_sha, binding=binding, path=path, view=view,
                    source_sha256=self._hashes[path], view_sha256=manifest["view_sha256"],
                    manifest_sha256=manifest_sha, source_bytes=len(original), view_bytes=len(compact),
                    requested_profile=self._profile, effective_profile=manifest["effective_profile"],
                    fallback_reason=manifest["fallback_reason"] if path.endswith(".py") else "non_python_identity",
                    representation_sha256=_sha(data), slice_sha256=_sha(part), encoding=encoding, content=content,
                    offset=offset, end_offset=end, representation_bytes=len(data), exhausted=end == len(data),
                    next_cursor=None if end == len(data) else self._next(query, end, token),
                    canonical_source_current=True, workspace_source_current=True,
                    source_check_profile="owner-admitted-exact-source@1", **_FLAGS) | self._page_metadata(path)
            # Whole encoded JSON, including escaping/base64 and continuation, is
            # counted. No partial code point is dropped; base64 preserves it.
            lower, upper = 0, min(len(data) - offset, max_bytes)
            # Terminal pages lose their continuation envelope; first try the
            # complete remainder rather than assuming size is monotone there.
            if upper == len(data) - offset and len(canonical_json(result(upper))) <= max_bytes:
                return self._finish(result(upper), cursor, max_bytes)
            while lower < upper:
                middle = (lower + upper + 1) // 2
                if len(canonical_json(result(middle))) <= max_bytes:
                    lower = middle
                else:
                    upper = middle - 1
            selected = result(lower)
            _need(lower > 0 or not data, "source-view page cannot fit one source byte")
            return self._finish(selected, cursor, max_bytes)

    def discovery(self, *, cursor=None, max_bytes=8192):
        """Bounded path/pin discovery without source bodies or restoration maps."""
        with self._lock:
            query = self._query("discover", max_bytes)
            offset = self._offset(cursor, query)
            self._check_sources()
            names = tuple(self._hashes)
            _need(offset < len(names), "discovery offset outside admitted population")
            token = secrets.token_hex(16)
            rows = []
            selected = None
            for name in names[offset:]:
                compact, raw, manifest_sha = self._views[name]
                manifest = json.loads(raw)
                rows.append(dict(path=name, source_sha256=self._hashes[name],
                    view_sha256=manifest["view_sha256"], manifest_sha256=manifest_sha,
                    source_bytes=len(self._originals[name]), view_bytes=len(compact),
                    effective_profile=manifest["effective_profile"]))
                end = offset + len(rows)
                candidate = dict(schema=DISCOVERY_SCHEMA, scope_sha256=self._scope_sha,
                    total_count=len(names), returned_count=len(rows), offset=offset, end_offset=end,
                    rows=list(rows), exhausted=end == len(names),
                    next_cursor=None if end == len(names) else self._next(query, end, token), **_FLAGS)
                # The final entry can fit after the continuation disappears,
                # even when its predecessor did not. Population is <=64.
                if len(canonical_json(candidate)) <= max_bytes:
                    selected = candidate
            _need(selected is not None, "source-view discovery cannot fit one complete entry")
            return self._finish(selected, cursor, max_bytes)


class _WorkingSourceViews(WorkerSourceViews):
    """One private candidate generation, using the baseline codec and pager."""

    def __init__(self, parent, captured, counter):
        self._parent = parent
        self._root, self._workspace = parent._root, parent._workspace
        self._roots, self._profile, self._codec = dict(parent._roots), parent._profile, parent._codec
        self._hashes = {name: _sha(raw) for name, raw in captured.items()}
        self.generation = _sha(canonical_json(dict(baseline_scope_sha256=parent._scope_sha,
            session=parent._working_session, counter=counter, source_hashes=self._hashes)))
        self._scope = {**parent._scope, "schema": "supervisor-worker-working-source-admission@1",
            "source_hashes_sha256": _sha(canonical_json(self._hashes)),
            "baseline_scope_sha256": parent._scope_sha, "workspace_generation": self.generation,
            "source_state": "working-candidate", "baseline_evidence_applicable": False}
        self._scope_raw = canonical_json(self._scope)
        self._scope_sha = _sha(self._scope_raw)
        self._continuations, self._lock = {}, threading.Lock()
        self._originals = dict(captured)
        self._build_views()
        self._check_sources()

    def _check_sources(self):
        try:
            self._parent._check_baseline()
            self._check_binding()
            _need(self.generation == self._scope["workspace_generation"], "working generation changed")
            return self._capture(self._workspace, self._hashes)
        except (OSError, ValueError):
            self._continuations.clear()
            if self._parent._working is self:
                self._parent._discard_working()
            raise

    def _page_metadata(self, path):
        return dict(schema="supervisor-worker-source-view-page@2", source_state="working-candidate",
            workspace_generation=self.generation, baseline_source_sha256=self._parent._hashes[path],
            baseline_evidence_applicable=False, source_check_profile="owner-working-source-candidate@1")


__all__ = ["WorkerSourceViews", "PAGE_SCHEMA", "DISCOVERY_SCHEMA", "ADMISSION_SCHEMA", "canonical_json"]
