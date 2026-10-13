"""Map one compact edit and apply it through an existing fenced worktree owner.

This host-only adapter supports a fresh clean candidate, not incremental writes
to an arbitrary dirty overlay. Scope and lifecycle ownership come from the
caller, never the compact edit. Checks are sequential; external writers are not
locked out. A refused post-write check can leave a failed isolated candidate.
Application is not atomic across files and grants no proof or publication right.
"""
from __future__ import annotations

import difflib
import hashlib
from pathlib import Path
import re

from ..semantic_state.worktree import IsolatedWorktree, PatchScope, WorktreePhase
from .doctor_candidate_runner import _git
from .source_view_edits import map_source_view_edit
from .worker_source_views import MAX_SOURCE_BYTES, WorkerSourceViews, canonical_json, _digest, _path, _workspace

SCHEMA = "supervisor-source-view-edit-application@1"
MAX_PATCH_BYTES = 2_000_000


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _patch(path, original, candidate):
    """Render exact UTF-8 LF/CRLF text, including missing final-newline markers."""
    _path(path)
    _need(len(path.encode()) <= 512 and re.fullmatch(r"[A-Za-z0-9_./-]+", path),
        "compact edit patch requires a supported unquoted path")
    texts = []
    for raw in (original, candidate):
        _need(type(raw) is bytes and b"\0" not in raw, "binary compact edit patches are unsupported")
        text = raw.decode("utf-8", "strict")
        _need(not any(char in text for char in "\v\f\x1c\x1d\x1e\x85\u2028\u2029")
            and "\r" not in text.replace("\r\n", ""), "unsupported compact edit text boundaries")
        texts.append(text)
    lines = difflib.unified_diff(texts[0].splitlines(keepends=True), texts[1].splitlines(keepends=True),
        fromfile="a/" + path, tofile="b/" + path)
    parts = ["diff --git a/" + path + " b/" + path + "\n"]
    for line in lines:
        parts.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
    _need(len(parts) > 1, "compact edit must change original bytes")
    result = "".join(parts)
    _need(len(result.encode()) <= MAX_PATCH_BYTES, "compact edit patch exceeds byte bound")
    return result


def _owner_binding(owner, views, *, lease_id, fence, require_fresh):
    _need(type(owner) is IsolatedWorktree and type(views) is WorkerSourceViews,
        "exact native worktree and source-view owners required")
    _need(type(lease_id) is str and bool(lease_id) and type(fence) is int and fence > 0,
        "explicit native lease and fence required")
    _need(not owner._closed, "compact edit owner is closed")
    _need(Path(owner.repo_root) == views._root and Path(owner.worktree_path) == views._workspace,
        "compact edit canonical/workspace owner pairing differs")
    _need(_workspace(owner.repo_root, owner.worktree_path) == (views._root, views._workspace),
        "compact edit Git allocation differs")
    _need((owner.worktree_path / ".git").is_file() and not (owner.worktree_path / ".git").is_symlink(),
        "compact edit requires a linked isolated worktree")
    owner._require_owner(lease_id=lease_id, fence=fence)
    owner.assert_base_current()
    owner.assert_root_unmutated()
    baseline = views.baseline_admission()
    _need(owner.task_id == baseline["task_id"], "compact edit task identity differs")
    if require_fresh:
        _need(owner.phase in {WorktreePhase.READY, WorktreePhase.REJECTED},
            "compact edit requires a fresh candidate owner")
        _need(not _git(owner.worktree_path, "status", "--porcelain=v1", "-z", "--untracked-files=all"),
            "compact edit requires a clean worktree")
    return baseline


def apply_source_view_edit_candidate(*, owner, views, scope, lease_id, fence, path,
        expected_workspace_generation, expected_source_sha256, expected_view_sha256,
        expected_manifest_sha256, view_start, view_end, expected_view_preimage, replacement):
    """Apply one exact mapped edit; return only a pending-validation receipt.

    The owner supplies an existing PatchScope and allocation. All source/view
    pins are mandatory; this function never refreshes a stale input generation.
    After successful application it explicitly captures the resulting generation.
    """
    _need(type(owner) is IsolatedWorktree and type(views) is WorkerSourceViews
        and type(scope) is PatchScope, "exact native owner, source views and owner-supplied patch scope required")
    _path(path)
    for digest in (expected_workspace_generation, expected_source_sha256,
            expected_view_sha256, expected_manifest_sha256):
        _need(_digest(digest), "exact compact edit generation/source/view/manifest pins required")
    with owner._lock:
        baseline = _owner_binding(owner, views, lease_id=lease_id, fence=fence, require_fresh=True)
        _need(scope.admits(path)[0], "compact edit path outside owner patch scope")
        source = views.working_sources(expected_workspace_generation=expected_workspace_generation)
        _need(path in source, "compact edit path outside admitted source population")
        original = source[path]
        projection = views.working_projection(path=path,
            expected_workspace_generation=expected_workspace_generation)
        _need(_sha(original) == expected_source_sha256 and _sha(projection.view) == expected_view_sha256
            and projection.manifest_sha256 == expected_manifest_sha256,
            "compact edit source/view/manifest pins differ")
        mapped = map_source_view_edit(original=original, view=projection.view, manifest=projection.manifest,
            expected_manifest_sha256=expected_manifest_sha256, view_start=view_start, view_end=view_end,
            expected_view_preimage=expected_view_preimage, replacement=replacement)
        patch = _patch(path, original, mapped.candidate)
        expected = dict(source)
        expected[path] = mapped.candidate
        _need(sum(map(len, expected.values())) <= MAX_SOURCE_BYTES,
            "compact edit candidate exceeds admitted source byte bound")
        _need(_owner_binding(owner, views, lease_id=lease_id, fence=fence, require_fresh=True) == baseline,
            "compact edit baseline admission changed")
        _need(views.working_sources(expected_workspace_generation=expected_workspace_generation) == source,
            "compact edit source population changed before application")
        try:
            application = owner.apply_patch(patch, scope, lease_id=lease_id, fence=fence, visible_sources=source)
            _need(application.applied, "native compact edit application refused: " + ",".join(application.reason_codes))
            _need(_owner_binding(owner, views, lease_id=lease_id, fence=fence, require_fresh=False) == baseline,
                "compact edit owner changed after application")
            _need(_git(owner.worktree_path, "diff", "--cached", "--name-only", "--no-renames", "-z", "HEAD", "--")
                    == path.encode() + b"\0"
                and not _git(owner.worktree_path, "diff", "--name-only", "-z", "--")
                and not _git(owner.worktree_path, "ls-files", "--others", "--exclude-standard", "-z"),
                "compact edit staged or working path set differs")
            _need(_git(owner.worktree_path, "show", ":" + path) == mapped.candidate,
                "compact edit staged candidate bytes differ")
            with views._lock:
                views._check_baseline()
                _need(views._capture(views._workspace) == expected,
                    "compact edit complete source postimage differs")
                views._discard_working()
            refreshed = views.working_page(path=path, max_bytes=65536)
            generation = refreshed["workspace_generation"]
            _need(generation != expected_workspace_generation
                and refreshed["source_sha256"] == _sha(mapped.candidate)
                and refreshed["view_sha256"] == _sha(mapped.view)
                and refreshed["manifest_sha256"] == mapped.manifest_sha256,
                "compact edit refreshed source view differs")
            _need(views.working_sources(expected_workspace_generation=generation) == expected,
                "compact edit source changed after refresh")
            _need(_owner_binding(owner, views, lease_id=lease_id, fence=fence, require_fresh=False) == baseline,
                "compact edit final owner binding differs")
            _need(views.working_sources(expected_workspace_generation=generation) == expected,
                "compact edit source changed during final owner check")
        except Exception:
            with views._lock:
                views._discard_working()
            raise
        result = dict(schema=SCHEMA, status="applied_validation_pending", task_id=owner.task_id,
            path=path, baseline_scope_sha256=baseline["scope_sha256"],
            source_generation=expected_workspace_generation, workspace_generation=generation,
            before_source_sha256=expected_source_sha256, after_source_sha256=_sha(mapped.candidate),
            before_view_sha256=expected_view_sha256, after_view_sha256=_sha(mapped.view),
            manifest_sha256=mapped.manifest_sha256, mapping=mapped.receipt,
            source_population_sha256=_sha(canonical_json({name: _sha(raw) for name, raw in expected.items()})),
            patch_sha256=_sha(patch.encode()), patch_bytes=len(patch.encode()), application=application.to_dict(),
            baseline_evidence_applicable=False, validation_pending=True, proof_authority=False,
            execution_authority=False, publication_authority=False, completion_authority=False,
            source_obligations_discharged=0, provider_calls=0, prover_calls=0,
            atomic_application_claimed=False)
        if "view_policy_binding" in baseline:
            result.update(schema="supervisor-source-view-edit-application@2",
                view_policy_binding_sha256=_sha(canonical_json(baseline["view_policy_binding"])))
        result["receipt_sha256"] = _sha(canonical_json(result))
        return result


__all__ = ["apply_source_view_edit_candidate", "SCHEMA"]
