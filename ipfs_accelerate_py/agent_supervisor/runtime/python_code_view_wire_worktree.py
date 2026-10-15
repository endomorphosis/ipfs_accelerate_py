"""Lossless code-view transport over the existing native worktree boundary.

Only the wire representation is changed. Current original-source, task,
scope, fence, visibility and native patch validation remain authoritative.
This module registers no provider tool and grants no acceptance or proof.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

from ..semantic_refactoring.python_code_view import (
    MAX_EXPRESSION_BYTES,
    PythonCodeView,
    _canonical,
    _raw,
    _read_json,
)
from ..semantic_refactoring.python_code_view_wire import (
    pack_python_code_view_packet,
    unpack_python_code_view_packet,
)
from ..semantic_state.worktree import IsolatedWorktree, PatchScope
from .python_code_view_worktree import (
    NativeViewPatch,
    prepare_code_view_expression,
    read_worktree_code_view,
)

EDIT_WIRE_SCHEMA = "supervisor-python-code-view-wire-edit@1"
MAX_EDIT_WIRE_BYTES = 64 * 1024


class CodeViewWireWorktreeError(ValueError):
    """A transport request differs from the bounded current native view."""


def _edit_row(view_id, slot_id, before_sha256, expression):
    for value, label in ((view_id, "view identity"), (slot_id, "slot identity")):
        _raw(value, 1024, label)
        if not value or any(char.isspace() for char in value):
            raise CodeViewWireWorktreeError("nonempty exact edit identities required")
    if type(before_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", before_sha256) is None:
        raise CodeViewWireWorktreeError("exact original-expression SHA-256 required")
    _raw(expression, MAX_EXPRESSION_BYTES, "replacement expression")
    if not expression or any(char in expression for char in ("\r", "\n", "\\")):
        raise CodeViewWireWorktreeError("nonempty single-line replacement required")
    return [view_id, slot_id, before_sha256, expression]


def build_python_code_view_wire_edit(*, view_id: str, slot_id: str,
                                    before_sha256: str, expression: str) -> dict:
    """Use fixed columns while retaining both full identities and literal text."""
    return {"schema": EDIT_WIRE_SCHEMA,
            "edit": _edit_row(view_id, slot_id, before_sha256, expression)}


def expand_python_code_view_wire_edit(request: dict) -> dict:
    """Expand one closed bounded row; this supplies no native source visibility."""
    if (type(request) is not dict or len(request) != 2 or set(request) != {"schema", "edit"}
            or type(request["schema"]) is not str or request["schema"] != EDIT_WIRE_SCHEMA
            or type(request["edit"]) is not list or len(request["edit"]) != 4):
        raise CodeViewWireWorktreeError("closed versioned expression-edit row required")
    view_id, slot_id, before_sha256, expression = _edit_row(*request["edit"])
    return {"view_id": view_id, "slot_id": slot_id,
            "before_sha256": before_sha256, "expression": expression}


def parse_python_code_view_wire_edit(text: str) -> dict:
    """Retain raw JSON duplicate-key evidence until the strict parser checks it."""
    _raw(text, MAX_EDIT_WIRE_BYTES, "edit wire JSON")
    request = _read_json(text)
    expand_python_code_view_wire_edit(request)
    return request


def read_worktree_code_view_wire(*, worktree: IsolatedWorktree, scope: PatchScope,
                                view: PythonCodeView, expected_view_id: str,
                                lease_id: str, fence: int) -> tuple[dict, dict]:
    """Pack an actually current public view; statistics stay controller-side."""
    arguments = dict(worktree=worktree, scope=scope, view=view, expected_view_id=expected_view_id,
                     lease_id=lease_id, fence=fence)
    with worktree._lock:
        packet = read_worktree_code_view(**arguments)
        wire, stats = pack_python_code_view_packet(packet)
        if _canonical(unpack_python_code_view_packet(wire)) != _canonical(packet):
            raise CodeViewWireWorktreeError("wire differs from its current producer packet")
        if _canonical(read_worktree_code_view(**arguments)) != _canonical(packet):
            raise CodeViewWireWorktreeError("native view changed during transport read")
        return wire, stats


def restore_worktree_code_view_wire(*, worktree: IsolatedWorktree, scope: PatchScope,
                                   view: PythonCodeView, expected_view_id: str,
                                   wire_packet: dict, lease_id: str, fence: int) -> dict:
    """Require exact restored equality to the fresh bound native producer."""
    arguments = dict(worktree=worktree, scope=scope, view=view, expected_view_id=expected_view_id,
                     lease_id=lease_id, fence=fence)
    with worktree._lock:
        current = read_worktree_code_view(**arguments)
        restored = unpack_python_code_view_packet(wire_packet)
        if _canonical(restored) != _canonical(current):
            raise CodeViewWireWorktreeError("wire differs from the captured current native view")
        if _canonical(read_worktree_code_view(**arguments)) != _canonical(current):
            raise CodeViewWireWorktreeError("native view changed during transport restoration")
        return restored


def prepare_worktree_code_view_wire_expression(*, worktree: IsolatedWorktree,
        scope: PatchScope, view: PythonCodeView, expected_view_id: str,
        wire_request: dict | str, lease_id: str, fence: int,
        visible_sources: Mapping[str, str | bytes | None]) -> NativeViewPatch:
    """Expand a request and enter the unchanged source/scope/native validator."""
    request = parse_python_code_view_wire_edit(wire_request) if type(wire_request) is str else wire_request
    edit = expand_python_code_view_wire_edit(request)
    if type(expected_view_id) is not str or edit["view_id"] != expected_view_id:
        raise CodeViewWireWorktreeError("wire edit differs from expected captured view identity")
    return prepare_code_view_expression(worktree=worktree, scope=scope, view=view,
        expected_view_id=expected_view_id, slot_id=edit["slot_id"],
        before_sha256=edit["before_sha256"], expression=edit["expression"],
        lease_id=lease_id, fence=fence, visible_sources=visible_sources)
