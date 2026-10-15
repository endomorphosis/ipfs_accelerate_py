"""Bounded exact working-source reads attached to an existing typed owner.

This first profile admits clean committed regular files only. Historical CAS
integrity, native task custody and selected working-file equality are checked
separately. It grants no proof, edit, omission or completion authority. Checks
are sequential; they cannot lock out arbitrary filesystem writers.
"""
from __future__ import annotations

import base64
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import stat
import time

from ipfs_datasets_py.logic.software_contracts.content import (
    cid_for_bytes, cid_for_structured,
)
from ipfs_datasets_py.logic.software_contracts.semantic_index.committed_snapshot import (
    preflight_committed_repository,
)
from ..task_sources.control_plane_contracts import canonical_json_bytes, content_identity
from .repository_source_inventory import (
    SourceInventoryLimits, SourceInventoryRequest, _BoundCAS, _cid, _input_guard,
    load_source_inventory,
)

REQUEST_SCHEMA = "supervisor-current-source-request@1"
RESPONSE_SCHEMA = "supervisor-current-source-response@1"
BINDING_SCHEMA = "supervisor-current-source-binding@1"
TYPED_SCHEMA = "ipfs_accelerate_py/agent-supervisor/typed-state-owner-command@1"
MAX_RESPONSE_BYTES = 65536
MAX_SLICE_BYTES = 16384
_REQUEST_FIELDS = {"schema", "binding_cid", "raw_path_hex", "source_cid",
                   "offset", "max_bytes", "continuation"}
_OWNER_FIELDS = ("server_id", "process_birth_id", "store_id", "database_uuid",
                 "generation", "fence_epoch")
_BINDING_FIELDS = {"schema", "task_cid", "task_revision", "expected_task_body_cid", "context_cid",
    "repository_id", "expected_tree", "source_inventory_cid", "source_request",
    "max_response_bytes", "max_slice_bytes", "workspace_binding_cid", "source_store_binding_cid",
    "owner_identity", "allowed_path_count", "allowed_paths_cid", "source_limits_cid", "binding_cid"}
_RESULT_FIELDS = {"schema", "binding", "binding_cid", "request_cid", "request_id", "caller_grant_id",
    "raw_path_hex", "source_cid", "slice_cid", "encoding", "content", "offset",
    "end_offset", "source_bytes", "source_complete", "next_continuation",
    "unresolved_frontier", "working_source_verified", "source_check_profile",
    "repository_complete", "read_only", "completion_authority", "proof_authority",
    "edit_authority", "omission_authority", "response_bytes", "response_wire_bytes"}


class CurrentSourceLookupError(ValueError):
    """The owner cannot return an admitted current-source response."""


def _require(condition, reason):
    if not condition:
        raise CurrentSourceLookupError(reason)


def _path(value):
    _require(type(value) is str and 0 < len(value) <= 8192 and len(value) % 2 == 0
             and re.fullmatch(r"[0-9a-f]+", value), "invalid raw source path")
    raw = bytes.fromhex(value)
    _require(all(part and part not in (b".", b"..", b".git")
                 for part in raw.split(b"/")) and b"\0" not in raw,
             "unsafe raw source path")
    return raw


def _owner(identity):
    _require(isinstance(identity, dict) or hasattr(identity, "items"), "owner identity required")
    result = {key: identity.get(key) for key in _OWNER_FIELDS}
    for key in _OWNER_FIELDS[:4]:
        value = result[key]
        _require(type(value) is str and 0 < len(value.encode()) <= 4096,
                 "invalid current owner identity")
    for key in _OWNER_FIELDS[4:]:
        _require(type(result[key]) is int and result[key] >= 1, "invalid owner generation or fence")
    return result


def _payload(value):
    _require(type(value) is dict and set(value) == _REQUEST_FIELDS,
             "source request must contain exact fields")
    _require(len(canonical_json_bytes(value)) <= 8192, "source request exceeds byte bound")
    _require(value["schema"] == REQUEST_SCHEMA, "source request schema differs")
    _cid(value["binding_cid"])
    _cid(value["source_cid"], "raw")
    _path(value["raw_path_hex"])
    _require(type(value["offset"]) is int and 0 <= value["offset"] <= 128 * 1024**2,
             "invalid source offset")
    _require(type(value["max_bytes"]) is int and 1 <= value["max_bytes"] <= MAX_SLICE_BYTES,
             "invalid slice byte bound")
    if value["continuation"] is not None:
        _cid(value["continuation"])
    return json.loads(canonical_json_bytes(value))


def _cursor(binding_cid, grant_id, path, source_cid, offset, max_bytes):
    return cid_for_structured(dict(schema="supervisor-source-continuation@1",
        binding_cid=binding_cid, caller_grant_id=grant_id, raw_path_hex=path, source_cid=source_cid,
        offset=offset, max_bytes=max_bytes))


def _content(data):
    try:
        return "utf-8", data.decode("utf-8")
    except UnicodeDecodeError:
        return "base64", base64.b64encode(data).decode("ascii")


def _validate_binding(binding):
    _require(type(binding) is dict and set(binding) == _BINDING_FIELDS
             and binding["schema"] == BINDING_SCHEMA, "invalid source admission binding")
    _require(type(binding["task_cid"]) is str and 0 < len(binding["task_cid"].encode()) <= 4096
             and type(binding["task_revision"]) is int and binding["task_revision"] >= 1,
             "invalid source task binding")
    source = binding["source_request"]
    _require(type(source) is dict and set(source) == set(SourceInventoryRequest.__dataclass_fields__),
             "invalid retained source request binding")
    request = SourceInventoryRequest(**source)
    _require(binding["repository_id"] == request.repository_id
             and binding["expected_tree"] == request.expected_tree, "source repository binding differs")
    for key in ("expected_task_body_cid", "context_cid", "source_inventory_cid", "workspace_binding_cid",
                "source_store_binding_cid", "allowed_paths_cid", "source_limits_cid", "binding_cid"):
        _cid(binding[key])
    _require(type(binding["owner_identity"]) is dict and set(binding["owner_identity"]) == set(_OWNER_FIELDS)
             and _owner(binding["owner_identity"]) == binding["owner_identity"], "invalid source owner binding")
    _require(type(binding["allowed_path_count"]) is int and 1 <= binding["allowed_path_count"] <= 256
             and type(binding["max_response_bytes"]) is int
             and 2048 <= binding["max_response_bytes"] <= MAX_RESPONSE_BYTES
             and type(binding["max_slice_bytes"]) is int
             and 1 <= binding["max_slice_bytes"] <= MAX_SLICE_BYTES, "invalid source policy bounds")
    material = {key: value for key, value in binding.items() if key != "binding_cid"}
    _require(cid_for_structured(material) == binding["binding_cid"], "source binding identity differs")
    return binding


def _envelope(result, request_id):
    return dict(schema=TYPED_SCHEMA, request_id=request_id, ok=True, result=result)


def _count_response(result, request_id):
    # Both decimal counters contribute bytes; solve their tiny fixed point.
    result["response_bytes"] = result["response_wire_bytes"] = 0
    for _ in range(12):
        size = len(canonical_json_bytes(_envelope(result, request_id))) + 1
        wire = size + 3  # 4-byte typed framing replaces the CLI's one LF.
        if (size, wire) == (result["response_bytes"], result["response_wire_bytes"]):
            return size
        result["response_bytes"], result["response_wire_bytes"] = size, wire
    raise CurrentSourceLookupError("response byte accounting failed")


class _SourceBindingAttempt:
    """Private one-attempt custody; serialized observations cannot replace it."""

    __slots__ = ("_checks", "_started", "_published")

    def __init__(self):
        self._checks = ()
        self._started = False
        self._published = None

    def _add_checks(self, unlocked, locked):
        _require(not self._started and self._published is None,
                 "source binding attempt is already consumed")
        self._checks += ((unlocked, locked),)


class CurrentSourceLookup:
    """Host-admitted public path scope on the existing source store and owner."""

    def __init__(self, repository, *, cas, expected_inventory_cid, request,
                 limits=SourceInventoryLimits(), task_cid, task_revision,
                 expected_task_body_cid, context_cid, allowed_paths,
                 max_response_bytes=16384, max_slice_bytes=4096):
        self._configure(repository, cas=cas, expected_inventory_cid=expected_inventory_cid,
            request=request, limits=limits, task_cid=task_cid, task_revision=task_revision,
            expected_task_body_cid=expected_task_body_cid, context_cid=context_cid,
            allowed_paths=allowed_paths, max_response_bytes=max_response_bytes,
            max_slice_bytes=max_slice_bytes)
        self._replay_binding()

    @classmethod
    def _binding_candidate(cls, **configuration):
        """Configure the literal candidate without entering retained CAS."""
        _require(cls is CurrentSourceLookup, "exact source reader type required")
        service = cls.__new__(cls)
        service._configure(**configuration)
        return service

    def _configure(self, repository, *, cas, expected_inventory_cid, request,
                   limits=SourceInventoryLimits(), task_cid, task_revision,
                   expected_task_body_cid, context_cid, allowed_paths,
                   max_response_bytes=16384, max_slice_bytes=4096):
        self._guard = _input_guard(request, limits)
        self._request = SourceInventoryRequest(**asdict(request))
        self._limits = SourceInventoryLimits(**asdict(limits))
        self._store = _BoundCAS(cas, self._guard)
        self._original_store, self._original_cas = self._store, cas
        self._inventory_cid = _cid(expected_inventory_cid)
        self._root = Path(repository).absolute()
        _require(self._root == self._root.resolve(strict=True), "workspace path must be canonical")
        info = self._root.stat(follow_symlinks=False)
        _require(stat.S_ISDIR(info.st_mode), "workspace must be a directory")
        self._root_identity = info.st_dev, info.st_ino
        _require(type(task_cid) is str and 0 < len(task_cid.encode()) <= 4096,
                 "native task identity required")
        _require(type(task_revision) is int and task_revision >= 1, "invalid task revision")
        _cid(expected_task_body_cid)
        _cid(context_cid)
        _require(type(max_response_bytes) is int and 2048 <= max_response_bytes <= MAX_RESPONSE_BYTES,
                 "invalid complete response byte bound")
        _require(type(max_slice_bytes) is int and 1 <= max_slice_bytes <= MAX_SLICE_BYTES,
                 "invalid host slice byte bound")
        _require(type(allowed_paths) is tuple and 1 <= len(allowed_paths) <= 256,
                 "bounded host public path scope required")
        for path in allowed_paths:
            _path(path)
        _require(tuple(sorted(set(allowed_paths))) == allowed_paths,
                 "public path scope must be sorted and unique")
        self._paths = allowed_paths
        self._scope = dict(schema=BINDING_SCHEMA, task_cid=task_cid,
            task_revision=task_revision, expected_task_body_cid=expected_task_body_cid,
            context_cid=context_cid, repository_id=self._request.repository_id,
            expected_tree=self._request.expected_tree, source_inventory_cid=self._inventory_cid,
            source_request=asdict(self._request), source_limits=asdict(self._limits),
            allowed_paths=list(allowed_paths), max_response_bytes=max_response_bytes,
            max_slice_bytes=max_slice_bytes,
            workspace_binding_cid=cid_for_structured(dict(
                path=str(self._root), device=info.st_dev, inode=info.st_ino)),
            source_store_binding_cid=cid_for_structured(json.loads(json.dumps(self._store.binding))))
        self._scope_bytes = canonical_json_bytes(self._scope)
        self._binding = None
        self._continuations = {}
        self._active_read = None

    def _replay_binding(self):
        """Replay original retained source before publication, without SERVER."""
        self._check_source()
        restored = self._load()
        _require(set(self._paths).issubset(e.entry.raw_path_hex for e in restored.entries),
                 "public scope contains unretained source")
        self._check_source()

    def _check(self):
        _require(self._store is self._original_store and self._store.cas is self._original_cas,
                 "original source store object changed")
        self._store.check()
        _require(canonical_json_bytes(self._scope) == self._scope_bytes,
                 "source admission scope changed")
        _require(asdict(self._request) == self._scope["source_request"]
                 and asdict(self._limits) == self._scope["source_limits"]
                 and self._paths == tuple(self._scope["allowed_paths"]), "private source inputs changed")
        _require(self._inventory_cid == self._scope["source_inventory_cid"]
                 and cid_for_structured(json.loads(json.dumps(self._store._binding())))
                 == self._scope["source_store_binding_cid"], "source custody binding changed")
        info = self._root.stat(follow_symlinks=False)
        _require(stat.S_ISDIR(info.st_mode) and (info.st_dev, info.st_ino) == self._root_identity
                 and self._root.resolve(strict=True) == self._root, "workspace binding changed")
        _require(cid_for_structured(dict(path=str(self._root), device=info.st_dev, inode=info.st_ino))
                 == self._scope["workspace_binding_cid"], "workspace admission changed")

    def _check_source(self):
        self._check()
        request, limits = self._request, self._limits
        plan = preflight_committed_repository(self._root, expected_commit=request.expected_commit,
            expected_tree=request.expected_tree, repository_id=request.repository_id,
            max_entries=limits.max_entries, max_file_bytes=limits.max_file_bytes,
            max_total_bytes=limits.max_total_bytes, max_metadata_bytes=limits.max_metadata_bytes)
        _require(not plan.budget_violations and plan.population_cid == request.expected_population_cid,
                 "current source population differs")
        self._check()

    def _load(self):
        self._check()
        result = load_source_inventory(cas=self._store.cas, expected_inventory_cid=self._inventory_cid,
            request=self._request, limits=self._limits)
        self._check()
        return result

    def binding(self, owner_identity):
        self._check()
        material = {key: value for key, value in self._scope.items()
                    if key not in ("allowed_paths", "source_limits")}
        material.update(owner_identity=_owner(owner_identity),
            allowed_path_count=len(self._paths), allowed_paths_cid=cid_for_structured(list(self._paths)),
            source_limits_cid=cid_for_structured(self._scope["source_limits"]))
        result = dict(material, binding_cid=cid_for_structured(material))
        _validate_binding(result)
        if self._binding is None:
            self._binding = canonical_json_bytes(result)
        _require(canonical_json_bytes(result) == self._binding, "lookup owner binding changed")
        return json.loads(self._binding)

    def check_native(self, connection, owner_identity):
        binding = self.binding(owner_identity)
        owner = binding["owner_identity"]
        row = connection.execute(
            "SELECT generation, fence_epoch, database_uuid FROM store_generations "
            "ORDER BY generation DESC LIMIT 1").fetchone()
        _require(row is not None and tuple(row[index] for index in range(3)) == (
            owner["generation"], owner["fence_epoch"], owner["database_uuid"]),
            "current owner generation or fence differs")
        task = connection.execute("SELECT revision, body_json FROM tasks WHERE task_cid = ?",
                                  [binding["task_cid"]]).fetchone()
        _require(task is not None and task[0] == binding["task_revision"]
                 and type(task[1]) is str and len(task[1].encode()) <= 256 * 1024,
                 "native task revision or body is unavailable")
        body = json.loads(task[1])
        _require(type(body) is dict and content_identity(body) == binding["expected_task_body_cid"],
                 "native task body differs")
        self._check()

    def _working_file(self, entry, original):
        """Read exact raw path with no-follow descriptors, never execute targets."""
        raw = _path(entry.raw_path_hex)
        fds = []
        rewalk = []
        try:
            fd = os.open(self._root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            fds.append(fd)
            root_info = os.fstat(fd)
            _require((root_info.st_dev, root_info.st_ino) == self._root_identity,
                     "opened workspace differs")
            for part in raw.split(b"/")[:-1]:
                fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                             dir_fd=fd)
                fds.append(fd)
            leaf = raw.split(b"/")[-1]
            fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                         dir_fd=fd)
            fds.append(fd)
            before = os.fstat(fd)
            _require(stat.S_ISREG(before.st_mode) and before.st_size == len(original)
                     and ("100755" if before.st_mode & stat.S_IXUSR else "100644") == entry.git_mode,
                     "working source kind, mode or size differs")
            data = bytearray()
            while len(data) <= len(original):
                chunk = os.read(fd, min(65536, len(original) + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
            after = os.fstat(fd)
            fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
            _require(all(getattr(before, key) == getattr(after, key) for key in fields)
                     and bytes(data) == original, "working source bytes changed or differ")
            named = os.stat(leaf, dir_fd=fds[-2], follow_symlinks=False)
            _require(all(getattr(after, key) == getattr(named, key) for key in fields),
                     "working source pathname changed")
            # A stable detached parent descriptor is insufficient. Reopen the
            # complete path from the admitted root and match every directory.
            fresh = os.open(self._root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            rewalk.append(fresh)
            for depth, part in enumerate((None, *raw.split(b"/")[:-1])):
                if part is not None:
                    fresh = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                    dir_fd=fresh)
                    rewalk.append(fresh)
                current, previous = os.fstat(fresh), os.fstat(fds[depth])
                _require((current.st_dev, current.st_ino, current.st_mode)
                         == (previous.st_dev, previous.st_ino, previous.st_mode),
                         "working source parent pathname changed")
            named = os.stat(leaf, dir_fd=fresh, follow_symlinks=False)
            _require(all(getattr(after, key) == getattr(named, key) for key in fields),
                     "rewalked working source pathname changed")
        finally:
            for fd in reversed(rewalk):
                os.close(fd)
            for fd in reversed(fds):
                os.close(fd)
        self._check()

    def _prepare_read(self, payload, *, owner_identity, grant, connection,
                      request_id, checked_at_ms):
        """Capture one private read under the owner's original SERVER lock."""
        _require(self._active_read is None, "current source read is already active")
        _require(type(checked_at_ms) is int and checked_at_ms >= 0, "invalid source clock sample")
        payload = _payload(payload)
        binding = self.binding(owner_identity)
        _require(payload["binding_cid"] == binding["binding_cid"], "lookup binding differs")
        scopes = dict(grant.entity_scopes)
        _require(all(scopes.get(key) == expected for key, expected in (
            ("task_cid", binding["task_cid"]), ("repository_id", binding["repository_id"]),
            ("tree_id", binding["expected_tree"]))), "source read outside grant scope")
        _require(payload["raw_path_hex"] in self._paths, "source path outside public scope")
        _require(payload["max_bytes"] <= binding["max_slice_bytes"], "slice exceeds host bound")
        offset = payload["offset"]
        expected_cursor = None if offset == 0 else _cursor(binding["binding_cid"], grant.grant_id,
            payload["raw_path_hex"], payload["source_cid"], offset, payload["max_bytes"])
        _require(payload["continuation"] == expected_cursor, "continuation binding differs")
        live = {key: expiry for key, expiry in self._continuations.items() if expiry > checked_at_ms}
        _require(offset == 0 or (grant.grant_id, expected_cursor) in live,
                 "continuation was not issued to this caller")
        _require(len(live) < 4096, "continuation population exceeds bound")
        state = dict(service=self, payload=payload, payload_bytes=canonical_json_bytes(payload),
            binding=binding, binding_bytes=canonical_json_bytes(binding),
            owner=json.loads(canonical_json_bytes(dict(owner_identity))), connection=connection,
            grant=grant, grant_bytes=canonical_json_bytes(grant.public_dict()), request_id=request_id,
            registry=self._continuations,
            registry_before=dict(self._continuations), phase="prepared")
        state["original"] = (connection, grant, request_id,
            canonical_json_bytes(state["owner"]), state["payload_bytes"], state["binding_bytes"],
            self._continuations, tuple(self._continuations.items()))
        self._active_read = state
        try:
            self._read_current(state)
            self.check_native(connection, state["owner"])
            self._check_source()
            self._read_current(state)
            return state
        except BaseException:
            self._discard_read(state)
            raise

    def _read_current(self, state):
        _require(self._active_read is state and state["service"] is self,
                 "original current source read changed")
        self._check()
        original = state["original"]
        _require(state["connection"] is original[0] and state["grant"] is original[1]
                 and state["request_id"] == original[2]
                 and canonical_json_bytes(state["owner"]) == original[3]
                 and state["payload_bytes"] == original[4] and state["binding_bytes"] == original[5]
                 and self._continuations is original[6]
                 and tuple(self._continuations.items()) == original[7],
                 "original current source read witnesses changed")
        _require(canonical_json_bytes(state["payload"]) == state["payload_bytes"]
                 and canonical_json_bytes(state["binding"]) == state["binding_bytes"]
                 and canonical_json_bytes(self.binding(state["owner"])) == state["binding_bytes"]
                 and canonical_json_bytes(state["grant"].public_dict()) == state["grant_bytes"]
                 and self._continuations is state["registry"]
                 and self._continuations == state["registry_before"],
                 "current source read inputs or continuation registry changed")

    def _replay_read(self, state):
        """Only retained CAS replay; the gateway releases SERVER first."""
        self._read_current(state)
        _require(state["phase"] == "prepared", "source read replay is out of order")
        restored = self._load()
        self._read_current(state)
        state["restored"], state["phase"] = restored, "replayed"

    def _stage_read(self, state):
        """Build response bytes privately; no cursor is consumed or issued."""
        self._read_current(state)
        _require(state["phase"] == "replayed", "source read staging is out of order")
        payload, binding, grant = state["payload"], state["binding"], state["grant"]
        connection, owner_identity = state["connection"], state["owner"]
        request_id, offset, restored = state["request_id"], payload["offset"], state["restored"]
        selected = next(e for e in restored.entries if e.entry.raw_path_hex == payload["raw_path_hex"])
        _require(selected.entry.git_mode in ("100644", "100755") and selected.data is not None,
                 "only regular source files are admitted; other kinds require separate scope")
        _require(selected.source_cid == payload["source_cid"], "source content identity differs")
        data = selected.data
        _require((offset == 0 if not data else offset < len(data)), "source offset outside file")
        self._working_file(selected.entry, data)

        def response(count):
            end = offset + count
            slice_data = data[offset:end]
            encoding, content = _content(slice_data)
            complete = end == len(data)
            next_cursor = None if complete else _cursor(binding["binding_cid"], grant.grant_id,
                payload["raw_path_hex"], selected.source_cid, end, payload["max_bytes"])
            result = dict(schema=RESPONSE_SCHEMA, binding=binding,
                binding_cid=binding["binding_cid"], request_cid=cid_for_structured(payload),
                request_id=request_id, caller_grant_id=grant.grant_id, raw_path_hex=payload["raw_path_hex"],
                source_cid=selected.source_cid, slice_cid=cid_for_bytes(slice_data),
                encoding=encoding, content=content, offset=offset, end_offset=end,
                source_bytes=len(data), source_complete=complete, next_continuation=next_cursor,
                unresolved_frontier=[] if complete else [dict(kind="remaining-source-bytes",
                    offset=end, end_offset=len(data), source_cid=selected.source_cid)],
                working_source_verified=True, source_check_profile="clean-committed-regular@1",
                repository_complete=False, read_only=True, completion_authority=False,
                proof_authority=False, edit_authority=False, omission_authority=False)
            size = _count_response(result, request_id)
            return result, size

        count = min(payload["max_bytes"], len(data) - offset)
        result, size = response(count)
        # Conservative halving handles UTF-8/base64 boundary changes without
        # assuming monotone encoded sizes. Every shortened read has a frontier.
        while size > binding["max_response_bytes"] and count > 1:
            count //= 2
            result, size = response(count)
        _require(size <= binding["max_response_bytes"] and (count > 0 or not data),
                 "complete response envelope cannot fit host bound")
        state["selected"], state["response"] = selected, dict(ok=True, result=result)
        state["response_bytes"], state["phase"] = canonical_json_bytes(state["response"]), "staged"
        self._close_read(state)

    def _close_read(self, state):
        self._read_current(state)
        _require(state["phase"] == "staged", "source read closing is out of order")
        self.check_native(state["connection"], state["owner"])
        self._check_source()
        selected = state["selected"]
        self._working_file(selected.entry, selected.data)
        self._read_current(state)
        _require(canonical_json_bytes(state["response"]) == state["response_bytes"],
                 "staged current source response changed")

    def _publish_read(self, state, *, checked_at_ms):
        """Commit bounded cursors only after the gateway's closing authority checks."""
        self._read_current(state)
        _require(state["phase"] == "staged"
                 and canonical_json_bytes(state["response"]) == state["response_bytes"],
                 "source read publication is out of order or changed")
        _require(type(checked_at_ms) is int and checked_at_ms >= 0, "invalid source clock sample")
        grant, cursor = state["grant"], state["payload"]["continuation"]
        live = {key: expiry for key, expiry in self._continuations.items() if expiry > checked_at_ms}
        _require(cursor is None or (grant.grant_id, cursor) in live,
                 "continuation was not issued to this caller")
        _require(checked_at_ms < grant.expires_at and len(live) < 4096,
                 "source grant expired or continuation population exceeds bound")
        if cursor is not None:
            del live[grant.grant_id, cursor]
        successor = state["response"]["result"]["next_continuation"]
        if successor is not None:
            live[grant.grant_id, successor] = grant.expires_at
        result = json.loads(state["response_bytes"])
        self._continuations.clear()
        self._continuations.update(live)
        state["phase"] = "published"
        self._discard_read(state)
        return result

    def _discard_read(self, state):
        if self._active_read is state:
            self._active_read = None

    def execute(self, payload, *, owner_identity, grant, connection, request_id):
        """Compatibility entry point; the wire gateway uses the detached phases."""
        state = self._prepare_read(payload, owner_identity=owner_identity, grant=grant,
            connection=connection, request_id=request_id, checked_at_ms=int(time.time() * 1000))
        try:
            self._replay_read(state)
            self._stage_read(state)
            closing_time = int(time.time() * 1000)
            self._close_read(state)
            return self._publish_read(state, checked_at_ms=closing_time)
        finally:
            self._discard_read(state)


def validate_source_response(response, *, payload, owner_identity, grant):
    """Validate identities, bytes, frontier and full encoded response on client."""
    payload = _payload(payload)
    _require(type(response) is dict and set(response) == {"schema", "request_id", "ok", "result"}
             and response["schema"] == TYPED_SCHEMA and response["ok"] is True,
             "invalid source response envelope")
    result = response["result"]
    _require(type(result) is dict and set(result) == _RESULT_FIELDS
             and result["schema"] == RESPONSE_SCHEMA, "invalid source response fields")
    binding = _validate_binding(result["binding"])
    material = {key: value for key, value in binding.items() if key != "binding_cid"}
    _require(cid_for_structured(material) == binding.get("binding_cid") == payload["binding_cid"]
             == result["binding_cid"] and binding.get("owner_identity") == _owner(owner_identity),
             "source response owner or admission differs")
    scopes = dict(grant.get("entity_scopes", ()))
    _require(all(scopes.get(key) == expected for key, expected in (
        ("task_cid", binding.get("task_cid")), ("repository_id", binding.get("repository_id")),
        ("tree_id", binding.get("expected_tree")))), "source response grant differs")
    _require(result["request_id"] == response["request_id"]
             and result["caller_grant_id"] == grant.get("grant_id")
             and result["request_cid"] == cid_for_structured(payload)
             and result["raw_path_hex"] == payload["raw_path_hex"]
             and result["source_cid"] == payload["source_cid"], "source response request differs")
    _require(payload["max_bytes"] <= binding["max_slice_bytes"], "source response slice exceeds policy")
    _require(type(result["content"]) is str, "source content is not encoded text")
    try:
        if result["encoding"] == "utf-8":
            data = result["content"].encode("utf-8")
        elif result["encoding"] == "base64":
            data = base64.b64decode(result["content"], validate=True)
            _require(base64.b64encode(data).decode() == result["content"], "noncanonical source encoding")
            _require(_content(data)[0] == "base64", "unnecessary source encoding")
        else:
            raise CurrentSourceLookupError("invalid source encoding")
    except (ValueError, UnicodeError) as exc:
        raise CurrentSourceLookupError("source response encoding differs") from exc
    for key in ("offset", "end_offset", "source_bytes", "response_bytes", "response_wire_bytes"):
        _require(type(result[key]) is int and result[key] >= 0, "invalid response count")
    _require(result["offset"] == payload["offset"]
             and result["end_offset"] == result["offset"] + len(data)
             and result["end_offset"] <= result["source_bytes"] <= 128 * 1024**2
             and len(data) <= payload["max_bytes"]
             and (len(data) > 0 or result["source_bytes"] == 0)
             and result["slice_cid"] == cid_for_bytes(data), "source response byte range differs")
    complete = result["end_offset"] == result["source_bytes"]
    if complete and result["offset"] == 0:
        _require(cid_for_bytes(data) == payload["source_cid"], "complete response source CID differs")
    expected_cursor = None if complete else _cursor(payload["binding_cid"], grant.get("grant_id"),
        payload["raw_path_hex"], payload["source_cid"], result["end_offset"], payload["max_bytes"])
    frontier = [] if complete else [dict(kind="remaining-source-bytes", offset=result["end_offset"],
        end_offset=result["source_bytes"], source_cid=payload["source_cid"])]
    _require(result["source_complete"] is complete and result["next_continuation"] == expected_cursor
             and canonical_json_bytes(result["unresolved_frontier"]) == canonical_json_bytes(frontier),
             "source response frontier differs")
    _require(all(result[key] is False for key in ("repository_complete", "completion_authority",
        "proof_authority", "edit_authority", "omission_authority"))
        and result["working_source_verified"] is True and result["read_only"] is True
        and result["source_check_profile"] == "clean-committed-regular@1", "invalid source authority flags")
    size = len(canonical_json_bytes(response)) + 1
    _require(result["response_bytes"] == size and result["response_wire_bytes"] == size + 3
             and type(binding.get("max_response_bytes")) is int
             and 2048 <= binding["max_response_bytes"] <= MAX_RESPONSE_BYTES
             and size <= binding["max_response_bytes"], "complete source response byte bound differs")
    return json.loads(canonical_json_bytes(response))
