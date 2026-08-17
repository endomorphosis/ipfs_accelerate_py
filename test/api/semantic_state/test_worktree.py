"""SCH-010 safe fenced worktree and patch validation tests."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    CleanupDisposition,
    WorktreeLifecycleStore,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.worktree import (
    ISOLATED_PATCH_WORKTREE_INTERFACE,
    IsolatedWorktree,
    PatchScope,
    PatchValidationError,
    WorktreeFenceError,
    WorktreePhase,
    apply_patch,
    create_isolated_worktree,
    isolated_patch_worktree_descriptor,
    recover_isolated_worktree,
    validate_patch,
)


def _git(cwd: Path, *args: str, check: bool = True, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and completed.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed: {completed.stderr or completed.stdout}"
        )
    return completed


def _init_repo(root: Path) -> tuple[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init")
    _git(root, "config", "user.email", "sch-010@example.com")
    _git(root, "config", "user.name", "SCH-010")
    pkg = root / "pkg"
    pkg.mkdir()
    (pkg / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
    (pkg / "other.py").write_text("OTHER = 0\n", encoding="utf-8")
    (root / "README.md").write_text("# fixture\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "initial")
    head = _git(root, "rev-parse", "HEAD").stdout.strip()
    tree = _git(root, "rev-parse", "HEAD^{tree}").stdout.strip()
    return head, tree


def _scope(**overrides: object) -> PatchScope:
    payload: dict[str, object] = {
        "allowed_paths": ("pkg/",),
        "effect_paths": ("pkg/target.py",),
        "task_owned_paths": ("pkg/",),
    }
    payload.update(overrides)
    return PatchScope.from_dict(payload)


def _simple_patch(
    *,
    path: str = "pkg/target.py",
    old: str = "VALUE = 1",
    new: str = "VALUE = 2",
) -> str:
    return textwrap.dedent(
        f"""\
        diff --git a/{path} b/{path}
        --- a/{path}
        +++ b/{path}
        @@ -1 +1 @@
        -{old}
        +{new}
        """
    )


def _lifecycle_store(repo: Path, tmp_path: Path) -> WorktreeLifecycleStore:
    return WorktreeLifecycleStore(
        repo_root=repo,
        store_dir=tmp_path / "lifecycle",
        lease_seconds=300.0,
    )


def test_cold_import_is_side_effect_free() -> None:
    module = importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.worktree"
    )
    assert module.ISOLATED_PATCH_WORKTREE_INTERFACE == "IsolatedPatchWorktree@1"
    descriptor = isolated_patch_worktree_descriptor()
    assert descriptor["interface"] == ISOLATED_PATCH_WORKTREE_INTERFACE
    assert "create_isolated_worktree" in descriptor["symbols"]


def test_malformed_patch_is_rejected_without_worktree() -> None:
    result = validate_patch("not a patch", _scope())
    assert result.accepted is False
    assert "malformed_patch" in result.reason_codes


def test_out_of_scope_path_is_rejected() -> None:
    patch = _simple_patch(path="README.md", old="# fixture", new="# changed")
    result = validate_patch(patch, _scope())
    assert result.accepted is False
    assert any(
        code in result.reason_codes
        for code in ("outside_allowlist", "outside_effect_scope", "outside_task_owned")
    )


def test_forbidden_control_path_is_rejected() -> None:
    patch = _simple_patch(path=".git/config", old="x", new="y")
    result = validate_patch(
        patch,
        _scope(
            allowed_paths=(".git/", "pkg/"),
            effect_paths=(),
            task_owned_paths=(".git/", "pkg/"),
            forbidden_paths=(".git/",),
        ),
        run_apply_check=False,
    )
    # Malformed/unsafe path or forbidden — either is fail-closed.
    assert result.accepted is False


def test_binary_patch_is_rejected() -> None:
    patch = textwrap.dedent(
        """\
        diff --git a/pkg/target.py b/pkg/target.py
        GIT binary patch
        literal 0
        HcmV?d00001

        """
    )
    result = validate_patch(patch, _scope(), run_apply_check=False)
    assert result.accepted is False
    assert "malformed_patch" in result.reason_codes or "binary_forbidden" in result.reason_codes


def test_invisible_preimage_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    wt_path = tmp_path / "wts" / "invisible"
    with create_isolated_worktree(
        repo_root=repo,
        worktree_path=wt_path,
        base_commit=head,
        base_tree=tree,
        task_id="SCH-010-invis",
        attempt=1,
        lifecycle_store=store,
    ) as isolated:
        root_before = (repo / "pkg" / "target.py").read_text(encoding="utf-8")
        wt_before = (isolated.worktree_path / "pkg" / "target.py").read_text(
            encoding="utf-8"
        )
        patch = _simple_patch()
        result = isolated.validate_patch(
            patch,
            _scope(),
            lease_id=isolated.lease_id,
            fence=isolated.fence,
            visible_sources={},  # provider saw nothing
        )
        assert result.accepted is False
        assert "invisible_preimage" in result.reason_codes
        assert (isolated.worktree_path / "pkg" / "target.py").read_text(
            encoding="utf-8"
        ) == wt_before
        assert (repo / "pkg" / "target.py").read_text(encoding="utf-8") == root_before


def test_stale_base_causes_no_mutation(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    wt_path = tmp_path / "wts" / "stale"
    with create_isolated_worktree(
        repo_root=repo,
        worktree_path=wt_path,
        base_commit=head,
        base_tree=tree,
        task_id="SCH-010-stale",
        attempt=1,
        lifecycle_store=store,
    ) as isolated:
        # Advance the bound base identity artificially so validation sees staleness.
        isolated.base_commit = "0" * 40
        before = (isolated.worktree_path / "pkg" / "target.py").read_text(encoding="utf-8")
        root_before = (repo / "pkg" / "target.py").read_text(encoding="utf-8")
        result = isolated.validate_patch(
            _simple_patch(),
            _scope(),
            lease_id=isolated.lease_id,
            fence=isolated.fence,
            visible_sources={"pkg/target.py": "VALUE = 1\n"},
        )
        assert result.accepted is False
        assert "stale_base" in result.reason_codes
        assert (isolated.worktree_path / "pkg" / "target.py").read_text(
            encoding="utf-8"
        ) == before
        assert (repo / "pkg" / "target.py").read_text(encoding="utf-8") == root_before


def test_failed_apply_check_causes_no_mutation(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    wt_path = tmp_path / "wts" / "check-fail"
    with create_isolated_worktree(
        repo_root=repo,
        worktree_path=wt_path,
        base_commit=head,
        base_tree=tree,
        task_id="SCH-010-check",
        attempt=1,
        lifecycle_store=store,
    ) as isolated:
        # Provider saw a preimage that does not match the worktree base, so
        # visibility can pass while git apply --check still fails closed.
        bad = _simple_patch(old="VALUE = 99", new="VALUE = 100")
        before = (isolated.worktree_path / "pkg" / "target.py").read_text(encoding="utf-8")
        root_before = (repo / "pkg" / "target.py").read_text(encoding="utf-8")
        status_before = _git(
            isolated.worktree_path, "status", "--porcelain"
        ).stdout
        result = apply_patch(
            bad,
            worktree_root=isolated.worktree_path,
            scope=_scope(),
            expected_base_commit=isolated.base_commit,
            expected_base_tree=isolated.base_tree,
            visible_sources={"pkg/target.py": "VALUE = 99\n"},
        )
        assert result.applied is False
        assert "apply_check_failed" in result.reason_codes
        assert (isolated.worktree_path / "pkg" / "target.py").read_text(
            encoding="utf-8"
        ) == before
        assert (repo / "pkg" / "target.py").read_text(encoding="utf-8") == root_before
        status_after = _git(
            isolated.worktree_path, "status", "--porcelain"
        ).stdout
        assert status_after == status_before
        # Root HEAD unchanged.
        assert _git(repo, "rev-parse", "HEAD").stdout.strip() == head


def test_allowed_text_patch_applies_deterministically(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    wt_path = tmp_path / "wts" / "apply-ok"
    patch = _simple_patch()
    with create_isolated_worktree(
        repo_root=repo,
        worktree_path=wt_path,
        base_commit=head,
        base_tree=tree,
        task_id="SCH-010-apply",
        attempt=1,
        lifecycle_store=store,
    ) as isolated:
        assert isolated.phase is WorktreePhase.READY
        first = isolated.apply_patch(
            patch,
            _scope(),
            lease_id=isolated.lease_id,
            fence=isolated.fence,
            visible_sources={"pkg/target.py": "VALUE = 1\n"},
        )
        assert first.applied is True
        assert first.pre_tree == tree
        assert first.post_tree
        assert first.post_tree != first.pre_tree
        assert (isolated.worktree_path / "pkg" / "target.py").read_text(
            encoding="utf-8"
        ) == "VALUE = 2\n"
        # User checkout never mutated.
        assert (repo / "pkg" / "target.py").read_text(encoding="utf-8") == "VALUE = 1\n"
        assert _git(repo, "rev-parse", "HEAD").stdout.strip() == head

    # Recreate and re-apply for determinism of post_tree.
    wt_path2 = tmp_path / "wts" / "apply-ok-2"
    with create_isolated_worktree(
        repo_root=repo,
        worktree_path=wt_path2,
        base_commit=head,
        base_tree=tree,
        task_id="SCH-010-apply",
        attempt=2,
        lifecycle_store=store,
    ) as isolated2:
        second = isolated2.apply_patch(
            patch,
            _scope(),
            lease_id=isolated2.lease_id,
            fence=isolated2.fence,
            visible_sources={"pkg/target.py": "VALUE = 1\n"},
        )
        assert second.applied is True
        assert second.post_tree == first.post_tree


def test_stale_owner_cannot_publish_or_clean_live_peer(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    wt_path = tmp_path / "wts" / "peer"
    isolated = create_isolated_worktree(
        repo_root=repo,
        worktree_path=wt_path,
        base_commit=head,
        base_tree=tree,
        task_id="SCH-010-peer",
        attempt=1,
        lifecycle_store=store,
    )
    try:
        live_lease = isolated.lease_id
        live_fence = isolated.fence
        stale_fence = live_fence - 1 if live_fence > 1 else 0

        with pytest.raises(WorktreeFenceError):
            isolated.publish(
                lease_id=live_lease,
                fence=stale_fence,
                result={"ok": True},
            )

        with pytest.raises(WorktreeFenceError):
            isolated.cleanup(
                lease_id="not-the-owner",
                fence=live_fence,
            )

        decision = store.evaluate_cleanup(workspace_path=wt_path)
        assert decision.allowed is False
        assert decision.disposition is CleanupDisposition.DENY

        # Live owner can still publish and clean.
        published = isolated.publish(
            lease_id=live_lease,
            fence=isolated.fence,
            result={"candidate": True},
        )
        assert published["published"] is True
        cleaned = isolated.cleanup(
            lease_id=isolated.lease_id,
            fence=isolated.fence,
            reason="test_done",
        )
        assert cleaned["cleaned"] is True
        assert not wt_path.exists()
    finally:
        if not isolated._closed:
            try:
                isolated.cleanup(
                    lease_id=isolated.lease_id,
                    fence=isolated.fence,
                    reason="finally",
                )
            except Exception:
                recover_isolated_worktree(
                    lifecycle_store=store,
                    worktree_path=wt_path,
                    repo_root=repo,
                )


def test_peer_cannot_clean_while_preparing_without_checkout(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    workspace = tmp_path / "wts" / "preparing-only"
    record = store.begin_preparing(
        task_id="SCH-010-prep",
        attempt=1,
        lane_id="lane",
        workspace_path=workspace,
        branch="semantic/prep-a1",
        merge_target="HEAD",
    )
    decision = store.authorize_cleanup(
        workspace_path=workspace,
        caller_lease_id="peer-cleaner",
    )
    assert decision.allowed is False
    assert record.state.value == "preparing"


def test_interrupted_prepare_recovers(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    workspace = tmp_path / "wts" / "interrupted-prep"
    record = store.begin_preparing(
        task_id="SCH-010-irecover",
        attempt=1,
        lane_id="lane",
        workspace_path=workspace,
        branch="semantic/irecover-a1",
        merge_target="HEAD",
    )
    # Simulate a journal left mid-prepare without a materialised worktree.
    import hashlib

    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        normalize_workspace_path,
    )

    digest = hashlib.sha256(
        normalize_workspace_path(workspace).encode("utf-8")
    ).hexdigest()[:16]
    journal_path = Path(store.store_dir) / f"attempt-{digest}.json"  # type: ignore[arg-type]
    journal_path.write_text(
        json.dumps(
            {
                "schema": "semantic-state-worktree-attempt@1",
                "phase": "preparing",
                "lease_id": record.lease_id,
                "fence": record.fence,
                "repo_root": str(repo),
                "worktree_path": str(workspace),
                "base_commit": head,
                "base_tree": tree,
                "task_id": "SCH-010-irecover",
                "attempt": 1,
            }
        ),
        encoding="utf-8",
    )
    recovery = recover_isolated_worktree(
        lifecycle_store=store,
        worktree_path=workspace,
        repo_root=repo,
        caller_lease_id=record.lease_id,
    )
    assert recovery["recovered"] is True
    assert "marked_terminal" in recovery["actions"]
    loaded = store.load_workspace(workspace)
    assert loaded is not None and loaded.is_terminal
    decision = store.evaluate_cleanup(workspace_path=workspace)
    assert decision.allowed is True


def test_interrupted_apply_recovers_dirty_worktree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    wt_path = tmp_path / "wts" / "interrupted-apply"
    isolated = create_isolated_worktree(
        repo_root=repo,
        worktree_path=wt_path,
        base_commit=head,
        base_tree=tree,
        task_id="SCH-010-iapply",
        attempt=1,
        lifecycle_store=store,
    )
    try:
        # Simulate crash mid-apply: dirty worktree + APPLYING journal.
        (isolated.worktree_path / "pkg" / "target.py").write_text(
            "VALUE = dirty\n", encoding="utf-8"
        )
        isolated.phase = WorktreePhase.APPLYING
        isolated._write_journal()
        recovery = recover_isolated_worktree(
            lifecycle_store=store,
            worktree_path=wt_path,
            repo_root=repo,
            caller_lease_id=isolated.lease_id,
        )
        assert recovery["recovered"] is True
        assert "reset_base" in recovery["actions"] or "removed" in recovery["actions"]
        if wt_path.exists():
            # Either cleaned away or reset to base content.
            content = (wt_path / "pkg" / "target.py").read_text(encoding="utf-8")
            assert content == "VALUE = 1\n"
        # Root never touched.
        assert (repo / "pkg" / "target.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    finally:
        recover_isolated_worktree(
            lifecycle_store=store,
            worktree_path=wt_path,
            repo_root=repo,
            caller_lease_id=isolated.lease_id,
        )


def test_create_rejects_mismatched_base_tree(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, _tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    with pytest.raises(PatchValidationError) as excinfo:
        create_isolated_worktree(
            repo_root=repo,
            worktree_path=tmp_path / "wts" / "bad-tree",
            base_commit=head,
            base_tree="0" * 40,
            task_id="SCH-010-badtree",
            attempt=1,
            lifecycle_store=store,
        )
    assert excinfo.value.reason_code == "stale_base"


def test_context_manager_cleans_up(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    wt_path = tmp_path / "wts" / "cm"
    with create_isolated_worktree(
        repo_root=repo,
        worktree_path=wt_path,
        base_commit=head,
        base_tree=tree,
        task_id="SCH-010-cm",
        attempt=1,
        lifecycle_store=store,
    ) as isolated:
        assert isolated.worktree_path.is_dir()
        assert isolated.phase is WorktreePhase.READY
    assert not wt_path.exists()
    record = store.load_workspace(wt_path)
    assert record is not None
    assert record.is_terminal


def test_module_level_apply_patch_is_deterministic(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    patch = _simple_patch()
    trees: list[str] = []
    for attempt in (1, 2):
        wt = tmp_path / "wts" / f"det-{attempt}"
        with create_isolated_worktree(
            repo_root=repo,
            worktree_path=wt,
            base_commit=head,
            base_tree=tree,
            task_id="SCH-010-det",
            attempt=attempt,
            lifecycle_store=store,
        ) as isolated:
            result = apply_patch(
                patch,
                worktree_root=isolated.worktree_path,
                scope=_scope(),
                expected_base_commit=head,
                expected_base_tree=tree,
                visible_sources={"pkg/target.py": "VALUE = 1\n"},
            )
            assert result.applied
            trees.append(result.post_tree)
    assert trees[0] == trees[1]


@pytest.mark.skipif(os.name != "posix", reason="lifecycle fencing assumes POSIX")
def test_duplicate_attempt_claim_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    wt1 = tmp_path / "wts" / "dup-1"
    first = create_isolated_worktree(
        repo_root=repo,
        worktree_path=wt1,
        base_commit=head,
        base_tree=tree,
        task_id="SCH-010-dup",
        attempt=1,
        lifecycle_store=store,
    )
    try:
        with pytest.raises(WorktreeFenceError) as excinfo:
            create_isolated_worktree(
                repo_root=repo,
                worktree_path=tmp_path / "wts" / "dup-2",
                base_commit=head,
                base_tree=tree,
                task_id="SCH-010-dup",
                attempt=1,
                lifecycle_store=store,
            )
        assert excinfo.value.reason_code == "duplicate_attempt"
    finally:
        first.cleanup(lease_id=first.lease_id, fence=first.fence)


def test_patch_scope_round_trip() -> None:
    scope = _scope(max_files=32)
    restored = PatchScope.from_dict(scope.to_dict())
    assert restored.allowed_paths == scope.allowed_paths
    assert restored.effect_paths == scope.effect_paths
    assert restored.max_files == 32


def test_isolated_worktree_to_dict_is_closed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head, tree = _init_repo(repo)
    store = _lifecycle_store(repo, tmp_path)
    with create_isolated_worktree(
        repo_root=repo,
        worktree_path=tmp_path / "wts" / "dict",
        base_commit=head,
        base_tree=tree,
        task_id="SCH-010-dict",
        attempt=1,
        lifecycle_store=store,
    ) as isolated:
        payload = isolated.to_dict()
        assert payload["interface"] == ISOLATED_PATCH_WORKTREE_INTERFACE
        assert payload["base_commit"] == head
        assert payload["base_tree"] == tree
        assert "lease_id" in payload
        json.dumps(payload)  # serializable
