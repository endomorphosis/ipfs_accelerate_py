"""Bounded task-local legend sessions over existing native worktree gates.

Returned read/native-source bodies are delivery candidates. Explicit caller
acknowledgements describe transport history; neither proves model consumption
or grants task acceptance. Private source is never used to fill visibility.
"""
from __future__ import annotations

import hashlib
import math
import uuid
from collections.abc import Mapping
from time import monotonic as _monotonic
from time import time as _wall_time

from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json

from ..merge.worktree_lifecycle import WorkspaceLifecycleState
from ..semantic_refactoring.python_code_view import _canonical, _raw
from ..semantic_refactoring.python_code_view_session import (
    restore_python_code_view_session,
    split_python_code_view_session,
)
from ..semantic_refactoring.python_code_view_wire import pack_python_code_view_packet
from ..semantic_state.worktree import IsolatedWorktree, PatchScope
from .python_code_view_wire_worktree import (
    expand_python_code_view_wire_edit,
    parse_python_code_view_wire_edit,
    prepare_worktree_code_view_wire_expression,
    restore_worktree_code_view_wire,
)
from .python_code_view_worktree import (
    NativeViewPatch,
    _owner,
    _source,
    apply_code_view_patch,
    capture_worktree_code_view,
    read_worktree_code_view,
)

MAX_VIEWS = 4
MAX_LEGENDS = 4
MAX_DELIVERIES = 16
MAX_RETAINED_BYTES = 8 * 1024 * 1024
MAX_SESSION_SECONDS = 300


class CodeViewSessionError(ValueError):
    """A session, delivery, source, owner, or retention binding failed."""


class PythonCodeViewSessionBroker:
    """One explicit controller session; no shared database or provider registry."""

    def __init__(self, *, worktree: IsolatedWorktree, scope: PatchScope,
                 session_id: str, lease_id: str, fence: int, ttl_seconds=120):
        _raw(session_id, 128, "session identity")
        if (not session_id or any(char.isspace() for char in session_id)
                or type(ttl_seconds) not in {int, float} or not math.isfinite(ttl_seconds)
                or not 0 < ttl_seconds <= MAX_SESSION_SECONDS):
            raise CodeViewSessionError("bounded explicit session identity and lifetime required")
        _owner(worktree, scope, lease_id, fence)
        self._worktree, self._scope = worktree, scope
        self._session_id, self._lease_id, self._fence = session_id, lease_id, fence
        self._broker_id = "broker:" + uuid.uuid4().hex
        self._scope_bytes = _canonical(scope.to_dict())
        self._binding = self._current_binding()
        self._deadline = _monotonic() + ttl_seconds
        self._closed = False
        self._views, self._legends, self._reads, self._native = {}, {}, {}, {}
        self._known_legends, self._known_reads, self._known_native = set(), set(), set()
        self._check(session_id)

    def _current_binding(self):
        worktree = self._worktree
        return (str(worktree.repo_root), str(worktree.worktree_path), worktree.task_id,
                worktree.attempt, worktree.base_commit, worktree.base_tree,
                worktree.lease_id, worktree.fence)

    def _clear(self):
        self._views.clear()
        self._legends.clear()
        self._reads.clear()
        self._native.clear()
        self._known_legends.clear()
        self._known_reads.clear()
        self._known_native.clear()

    def _check(self, session_id):
        if type(session_id) is not str or session_id != self._session_id:
            raise CodeViewSessionError("session identity differs")
        now, wall = _monotonic(), _wall_time()
        if self._closed or not math.isfinite(now) or now >= self._deadline:
            self._clear()
            self._closed = True
            raise CodeViewSessionError("session is closed or expired")
        try:
            _owner(self._worktree, self._scope, self._lease_id, self._fence)
        except Exception:
            self._clear()
            self._closed = True
            raise
        record = self._worktree.lifecycle_store.load_workspace(self._worktree.worktree_path)
        if (record is None or record.state is not WorkspaceLifecycleState.ACTIVE
                or not math.isfinite(wall) or not math.isfinite(record.expires_at)
                or wall >= record.expires_at):
            self._clear()
            self._closed = True
            raise CodeViewSessionError("native lifecycle is inactive or expired")
        if (self._current_binding() != self._binding
                or _canonical(self._scope.to_dict()) != self._scope_bytes):
            self._clear()
            self._closed = True
            raise CodeViewSessionError("session owner or scope binding changed")

    def _view(self, view_id):
        if type(view_id) is not str or view_id not in self._views:
            raise CodeViewSessionError("unknown or invalidated session view")
        view = self._views[view_id]
        read_worktree_code_view(worktree=self._worktree, scope=self._scope, view=view,
            expected_view_id=view_id, lease_id=self._lease_id, fence=self._fence)
        return view

    @staticmethod
    def _retained(views, legends, reads, native):
        # Encoded retention only; this is not a Python heap/RSS assertion.
        return (sum(len(view.artifact_json.encode()) for view in views.values())
                + sum(len(_canonical(legend)) for legend in legends.values())
                + len(_canonical(reads)) + len(_canonical(native)))

    def _budget(self, views, legends, reads, native):
        if (len(views) > MAX_VIEWS or len(legends) > MAX_LEGENDS
                or len(reads) > MAX_DELIVERIES or len(native) > MAX_DELIVERIES
                or self._retained(views, legends, reads, native) > MAX_RETAINED_BYTES):
            raise CodeViewSessionError("session retention bound exceeded")

    def read_view(self, *, path: str, session_id: str, known_read_id: str | None = None) -> dict:
        with self._worktree._lock:
            self._check(session_id)
            view = capture_worktree_code_view(worktree=self._worktree, scope=self._scope,
                path=path, lease_id=self._lease_id, fence=self._fence)
            wire, _ = pack_python_code_view_packet(view.model_packet())
            legend, state, _ = split_python_code_view_session(wire)
            legend_id, view_id = legend["legend_id"], view.view_id
            if known_read_id is not None:
                if (type(known_read_id) is not str or known_read_id not in self._known_reads
                        or self._reads.get(known_read_id) != [view_id, legend_id]):
                    raise CodeViewSessionError("snapshot reference is not acknowledged for this exact current view")
                self._view_recheck(view)
                return {"schema": "supervisor-python-code-view-session-reference@1",
                        "session_id": session_id, "broker_id": self._broker_id,
                        "view_id": view_id, "legend_id": legend_id, "read_id": known_read_id}
            include_legend = legend_id not in self._known_legends
            body = {"schema": "supervisor-python-code-view-session-read@1",
                    "session_id": session_id, "broker_id": self._broker_id, "state": state,
                    "legend": legend if include_legend else None}
            read_id = cid_for_dag_json(body)
            response = {**body, "read_id": read_id}
            _canonical(response)
            views, legends, reads = dict(self._views), dict(self._legends), dict(self._reads)
            views[view_id], legends[legend_id] = view, legend
            reads[read_id] = [view_id, legend_id]
            self._budget(views, legends, reads, self._native)
            self._view_recheck(view)
            self._views, self._legends, self._reads = views, legends, reads
            # Copies prevent a caller changing the retained immutable legend.
            from ..semantic_refactoring.python_code_view import _read_json
            return _read_json(_canonical(response).decode())

    def _view_recheck(self, view):
        self._check(self._session_id)
        read_worktree_code_view(worktree=self._worktree, scope=self._scope, view=view,
            expected_view_id=view.view_id, lease_id=self._lease_id, fence=self._fence)

    def acknowledge_legend(self, *, legend_id: str, read_id: str, session_id: str) -> dict:
        with self._worktree._lock:
            self._check(session_id)
            row = self._reads.get(read_id) if type(read_id) is str else None
            if row is None or type(legend_id) is not str or row[1] != legend_id:
                raise CodeViewSessionError("legend was not emitted by this session/read")
            self._view(row[0])
            self._known_legends.add(legend_id)
            return {"schema": "supervisor-python-code-view-legend-ack@1",
                    "session_id": session_id, "broker_id": self._broker_id, "legend_id": legend_id,
                    "acknowledgement_basis": "explicit_caller_transport_feedback"}

    def acknowledge_read(self, *, read_id: str, view_id: str, session_id: str) -> dict:
        with self._worktree._lock:
            self._check(session_id)
            row = self._reads.get(read_id) if type(read_id) is str else None
            if (row is None or type(view_id) is not str or row[0] != view_id
                    or row[1] not in self._known_legends):
                raise CodeViewSessionError("snapshot and legend have no matching session acknowledgements")
            self._view(view_id)
            self._known_reads.add(read_id)
            return {"schema": "supervisor-python-code-view-read-ack@1", "session_id": session_id,
                    "broker_id": self._broker_id, "read_id": read_id,
                    "acknowledgement_basis": "explicit_caller_transport_feedback"}

    def restore_read(self, *, response: dict, session_id: str) -> dict:
        with self._worktree._lock:
            self._check(session_id)
            _canonical(response)
            if type(response) is dict and response.get("schema") == "supervisor-python-code-view-session-reference@1":
                if (len(response) != 6 or set(response) != {"schema", "session_id", "broker_id", "view_id", "legend_id", "read_id"}
                        or response["session_id"] != session_id or response["broker_id"] != self._broker_id
                        or type(response["read_id"]) is not str or response["read_id"] not in self._known_reads
                        or self._reads.get(response["read_id"]) != [response["view_id"], response["legend_id"]]):
                    raise CodeViewSessionError("closed acknowledged snapshot reference required")
                view = self._view(response["view_id"])
                return read_worktree_code_view(worktree=self._worktree, scope=self._scope, view=view,
                    expected_view_id=view.view_id, lease_id=self._lease_id, fence=self._fence)
            if (type(response) is not dict or len(response) != 6
                    or set(response) != {"schema", "session_id", "broker_id", "state", "legend", "read_id"}
                    or response["schema"] != "supervisor-python-code-view-session-read@1"
                    or response["session_id"] != session_id or response["broker_id"] != self._broker_id):
                raise CodeViewSessionError("closed session read response required")
            body = {key: value for key, value in response.items() if key != "read_id"}
            if cid_for_dag_json(body) != response["read_id"] or type(response["state"]) is not dict:
                raise CodeViewSessionError("session read identity differs")
            legend_id = response["state"].get("legend_id")
            if (type(response["read_id"]) is not str
                    or self._reads.get(response["read_id"]) != [response["state"].get("view_id"), legend_id]):
                raise CodeViewSessionError("read response was not emitted by this session/view")
            if type(legend_id) is not str or legend_id not in self._legends:
                raise CodeViewSessionError("unknown session legend")
            legend = response["legend"]
            if legend is None:
                if legend_id not in self._known_legends:
                    raise CodeViewSessionError("legend has no caller acknowledgement")
                legend = self._legends[legend_id]
            elif _canonical(legend) != _canonical(self._legends[legend_id]):
                raise CodeViewSessionError("legend differs from this session's emitted legend")
            wire = restore_python_code_view_session(legend, response["state"])
            view = self._view(wire["view_id"])
            return restore_worktree_code_view_wire(worktree=self._worktree, scope=self._scope,
                view=view, expected_view_id=view.view_id, wire_packet=wire,
                lease_id=self._lease_id, fence=self._fence)

    def deliver_native_source(self, *, view_id: str, session_id: str,
                              known_delivery_id: str | None = None) -> dict:
        with self._worktree._lock:
            self._check(session_id)
            view = self._view(view_id)
            # Actual confined filesystem read; retained private originals are
            # not used as delivery or model visibility evidence.
            source = _source(self._worktree, self._scope, view.path)
            self._view_recheck(view)
            source_sha = hashlib.sha256(source.encode()).hexdigest()
            if known_delivery_id is not None:
                if (type(known_delivery_id) is not str or known_delivery_id not in self._known_native
                        or self._native.get(known_delivery_id) != [view_id, view.path, source_sha]):
                    raise CodeViewSessionError("native reference is not acknowledged for this exact current preimage")
                return {"schema": "supervisor-python-code-view-native-reference@1",
                        "session_id": session_id, "broker_id": self._broker_id, "view_id": view_id,
                        "path": view.path, "source_sha256": source_sha, "delivery_id": known_delivery_id}
            body = {"schema": "supervisor-python-code-view-native-delivery@1",
                    "session_id": session_id, "broker_id": self._broker_id, "view_id": view_id, "path": view.path,
                    "source_sha256": source_sha, "source": source}
            delivery_id = cid_for_dag_json(body)
            native = dict(self._native)
            native[delivery_id] = [view_id, view.path, source_sha]
            self._budget(self._views, self._legends, self._reads, native)
            self._native = native
            return {**body, "delivery_id": delivery_id}

    def acknowledge_native_source(self, *, delivery_id: str, view_id: str, session_id: str) -> dict:
        with self._worktree._lock:
            self._check(session_id)
            row = self._native.get(delivery_id) if type(delivery_id) is str else None
            if row is None or type(view_id) is not str or row[0] != view_id:
                raise CodeViewSessionError("native source was not delivered by this session/view")
            self._view(view_id)
            self._known_native.add(delivery_id)
            return {"schema": "supervisor-python-code-view-native-ack@1", "session_id": session_id,
                    "broker_id": self._broker_id,
                    "delivery_id": delivery_id,
                    "acknowledgement_basis": "explicit_caller_transport_feedback"}

    def _visible(self, view, visible_sources):
        if not isinstance(visible_sources, Mapping) or view.path not in visible_sources:
            raise CodeViewSessionError("caller native visibility and acknowledged delivery required")
        source = visible_sources[view.path]
        if type(source) is str:
            raw = _raw(source, 64 * 1024, "caller visible native source")
        elif type(source) is bytes and len(source) <= 64 * 1024:
            raw = source
        else:
            raise CodeViewSessionError("exact visible native bytes required")
        digest = hashlib.sha256(raw).hexdigest()
        if not any(self._native[key] == [view.view_id, view.path, digest]
                   for key in self._known_native):
            raise CodeViewSessionError("current native preimage has no acknowledged delivery")

    def prepare_edit(self, *, wire_request: dict | str, visible_sources, session_id: str) -> NativeViewPatch:
        with self._worktree._lock:
            self._check(session_id)
            request = parse_python_code_view_wire_edit(wire_request) if type(wire_request) is str else wire_request
            edit = expand_python_code_view_wire_edit(request)
            view = self._view(edit["view_id"])
            self._visible(view, visible_sources)
            return prepare_worktree_code_view_wire_expression(worktree=self._worktree,
                scope=self._scope, view=view, expected_view_id=view.view_id, wire_request=request,
                lease_id=self._lease_id, fence=self._fence, visible_sources=visible_sources)

    def apply_edit(self, *, prepared: NativeViewPatch, visible_sources, session_id: str):
        with self._worktree._lock:
            self._check(session_id)
            if type(prepared) is not NativeViewPatch:
                raise CodeViewSessionError("exact prepared native candidate required")
            view = self._view(prepared.view.view_id)
            if view.artifact_json != prepared.view.artifact_json:
                raise CodeViewSessionError("prepared view differs from session capture")
            self._visible(view, visible_sources)
            result = apply_code_view_patch(worktree=self._worktree, scope=self._scope,
                prepared=prepared, expected_view_id=view.view_id, lease_id=self._lease_id,
                fence=self._fence, visible_sources=visible_sources)
            if result.applied:
                invalid = {key for key, item in self._views.items() if item.path == view.path}
                self._views = {key: item for key, item in self._views.items() if key not in invalid}
                self._reads = {key: row for key, row in self._reads.items() if row[0] not in invalid}
                self._known_reads.intersection_update(self._reads)
                removed = {key for key, row in self._native.items() if row[0] in invalid}
                self._native = {key: row for key, row in self._native.items() if key not in removed}
                self._known_native.difference_update(removed)
            return result

    def close_session(self, *, session_id: str) -> dict:
        if type(session_id) is not str or session_id != self._session_id:
            raise CodeViewSessionError("session identity differs")
        with self._worktree._lock:
            self._clear()
            self._closed = True
            return {"schema": "supervisor-python-code-view-session-closed@1", "session_id": session_id,
                    "broker_id": self._broker_id,
                    "retained_encoded_bytes": 0, "completion_authority": False, "lean_admits": 0}
