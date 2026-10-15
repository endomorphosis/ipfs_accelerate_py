"""Mediate bounded code-view reads and expression edits through native worktrees.

The model-facing representation is never executed. Original source preimages,
the existing patch scope and the live worktree fence remain authoritative for
native mutation. Applying a candidate supplies no task or Lean qualification.
"""
from __future__ import annotations

import difflib
import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ..semantic_refactoring.python_code_view import (
    NativeCodeViewEdit,
    PythonCodeView,
    build_python_code_view,
    replace_python_expression,
    restore_python_code_view,
)
from ..semantic_state.worktree import (
    IsolatedWorktree,
    PatchApplyResult,
    PatchScope,
    PatchValidationError,
)
from .semantic_router_translation import _read

SOURCE_LIMIT = 64 * 1024
SAFE_PATH = re.compile(r"[A-Za-z0-9_./-]+\.py")


class CodeViewWorktreeError(ValueError):
    """A code-view source, task, scope, or patch binding failed."""


@dataclass(frozen=True)
class NativeViewPatch:
    """An expression request and its native patch; every apply re-prepares it."""
    view: PythonCodeView
    slot_id: str
    before_sha256: str
    expression: str
    native_edit: NativeCodeViewEdit
    patch_text: str

    def receipt(self) -> dict:
        return {"schema": "supervisor-native-code-view-patch@1", "view_id": self.view.view_id,
                "slot_id": self.slot_id, "path": self.native_edit.path,
                "native_before_sha256": self.native_edit.before_sha256,
                "native_after_sha256": self.native_edit.after_sha256,
                "patch_sha256": hashlib.sha256(self.patch_text.encode()).hexdigest(),
                "patch_bytes": len(self.patch_text.encode()), "candidate_only": True,
                "execution_authority": False, "completion_authority": False,
                "semantic_acceptance_authority": False, "lean_admits": 0}


def _owner(worktree, scope, lease_id, fence):
    if (not isinstance(worktree, IsolatedWorktree) or not isinstance(scope, PatchScope)
            or type(lease_id) is not str or not lease_id or type(fence) is not int or fence < 1):
        raise CodeViewWorktreeError("explicit native worktree, patch scope and live owner binding required")
    if worktree._closed or worktree.worktree_path.resolve() == worktree.repo_root.resolve():
        raise CodeViewWorktreeError("an open isolated worktree is required")
    worktree._require_owner(lease_id=lease_id, fence=fence)
    worktree.assert_base_current()


def _source(worktree, scope, path):
    if type(path) is not str or SAFE_PATH.fullmatch(path) is None:
        raise CodeViewWorktreeError("supported literal Python path required")
    admitted, reason = scope.admits(path)
    if not admitted:
        raise CodeViewWorktreeError("native code-view path is outside scope: " + reason)
    root = Path(worktree.worktree_path).absolute()
    if root.resolve(strict=True) != root:
        raise CodeViewWorktreeError("canonical native worktree path required")
    try:
        raw = _read(root, path)  # Existing no-follow, canonical, race-checked reader.
    except OSError as error:
        raise CodeViewWorktreeError("native source could not be read through the confined reader") from error
    if len(raw) > SOURCE_LIMIT:
        raise CodeViewWorktreeError("native code-view source exceeds byte bound")
    try:
        return raw.decode("utf-8")
    except UnicodeError as error:
        raise CodeViewWorktreeError("native code-view source must be UTF-8") from error


def _bound_source(worktree, scope, view, expected_view_id):
    if not isinstance(view, PythonCodeView) or type(expected_view_id) is not str or view.view_id != expected_view_id:
        raise CodeViewWorktreeError("captured code-view identity differs")
    binding = view.to_dict()["bindings"]
    if (binding["repository_id"] != str(worktree.repo_root)
            or binding["tree_id"] != worktree.base_tree or binding["task_id"] != worktree.task_id):
        raise CodeViewWorktreeError("code-view repository/tree/task binding differs")
    source = _source(worktree, scope, view.path)
    restore_python_code_view(view=view, current_source=source)
    return source


def capture_worktree_code_view(*, worktree: IsolatedWorktree, scope: PatchScope,
                              path: str, lease_id: str, fence: int) -> PythonCodeView:
    """Capture current original bytes; leave both repositories unchanged."""
    with worktree._lock:
        _owner(worktree, scope, lease_id, fence)
        source = _source(worktree, scope, path)
        view = build_python_code_view(source=source, path=path, repository_id=str(worktree.repo_root),
                                      tree_id=worktree.base_tree, task_id=worktree.task_id)
        if _source(worktree, scope, path) != source:
            raise CodeViewWorktreeError("native source changed during code-view capture")
        return view


def read_worktree_code_view(*, worktree: IsolatedWorktree, scope: PatchScope,
                           view: PythonCodeView, expected_view_id: str,
                           lease_id: str, fence: int) -> dict:
    """Return only the model packet after current source/task/fence checks."""
    with worktree._lock:
        _owner(worktree, scope, lease_id, fence)
        source = _bound_source(worktree, scope, view, expected_view_id)
        packet = view.model_packet()
        if _source(worktree, scope, view.path) != source:
            raise CodeViewWorktreeError("native source changed during code-view read")
        return packet


def _native_diff(edit):
    # Full-file context makes git's preimage check include every original line,
    # including unchanged lines outside the expression slot. This private native
    # patch is not model context; compact edit requests do not echo raw source.
    before = edit.before_source.splitlines(keepends=True)
    after = edit.after_source.splitlines(keepends=True)
    pieces = [f"diff --git a/{edit.path} b/{edit.path}\n"]
    for line in difflib.unified_diff(before, after, fromfile="a/" + edit.path,
                                     tofile="b/" + edit.path, n=max(len(before), len(after)), lineterm="\n"):
        if line.endswith("\n"):
            pieces.append(line)
        else:
            pieces.extend((line + "\n", "\\ No newline at end of file\n"))
    return "".join(pieces)


def prepare_code_view_expression(*, worktree: IsolatedWorktree, scope: PatchScope,
                                view: PythonCodeView, expected_view_id: str,
                                slot_id: str, before_sha256: str, expression: str,
                                lease_id: str, fence: int,
                                visible_sources: Mapping[str, str | bytes | None]) -> NativeViewPatch:
    """Invert an explicit expression request and validate its ordinary native diff."""
    with worktree._lock:
        _owner(worktree, scope, lease_id, fence)
        if not isinstance(visible_sources, Mapping):
            raise CodeViewWorktreeError("caller-supplied native source visibility evidence required")
        source = _bound_source(worktree, scope, view, expected_view_id)
        edit = replace_python_expression(view=view, slot_id=slot_id, before_sha256=before_sha256,
                                         expression=expression, current_source=source)
        patch = _native_diff(edit)
        # A private preimage or reversible view does not establish that the
        # model read required raw source. Preserve the caller's existing native
        # visibility evidence instead of filling it from retained originals.
        result = worktree.validate_patch(patch, scope, lease_id=lease_id, fence=fence,
                                         visible_sources=visible_sources)
        if not result.accepted:
            raise PatchValidationError("reconstructed code-view patch refused: " + ",".join(result.reason_codes),
                                       reason_code="code_view_native_patch_refused")
        if _source(worktree, scope, view.path) != source:
            raise CodeViewWorktreeError("native source changed during code-view patch preparation")
        return NativeViewPatch(view, slot_id, before_sha256, expression, edit, patch)


def apply_code_view_patch(*, worktree: IsolatedWorktree, scope: PatchScope,
                         prepared: NativeViewPatch, expected_view_id: str,
                         lease_id: str, fence: int,
                         visible_sources: Mapping[str, str | bytes | None]) -> PatchApplyResult:
    """Re-prepare under the live fence and use the existing native patch executor.

    The controller still owns ordinary acceptance validation. No model-facing
    source is materialized, imported or executed by this adapter.
    """
    if not isinstance(prepared, NativeViewPatch):
        raise CodeViewWorktreeError("prepared native code-view patch required")
    with worktree._lock:
        fresh = prepare_code_view_expression(worktree=worktree, scope=scope, view=prepared.view,
            expected_view_id=expected_view_id, slot_id=prepared.slot_id,
            before_sha256=prepared.before_sha256, expression=prepared.expression,
            lease_id=lease_id, fence=fence, visible_sources=visible_sources)
        if fresh != prepared:
            raise CodeViewWorktreeError("prepared native patch differs from current inversion")
        result = worktree.apply_patch(fresh.patch_text, scope, lease_id=lease_id, fence=fence,
                                      visible_sources=visible_sources)
        if result.applied and _source(worktree, scope, fresh.native_edit.path) != fresh.native_edit.after_source:
            raise CodeViewWorktreeError("native code-view patch postimage differs")
        return result
