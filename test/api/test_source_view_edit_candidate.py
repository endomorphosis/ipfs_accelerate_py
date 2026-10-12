"""Real Git mapped edits through the existing isolated worktree lifecycle."""
import hashlib
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import WorktreeLifecycleStore
from ipfs_accelerate_py.agent_supervisor.semantic_state.worktree import (
    IsolatedWorktree, PatchScope, WorktreeFenceError, create_isolated_worktree)
from ipfs_accelerate_py.agent_supervisor.runtime import source_view_edit_candidate as subject
from ipfs_accelerate_py.agent_supervisor.runtime.reversible_source_view import build_source_view
from ipfs_accelerate_py.agent_supervisor.runtime.worker_source_views import WorkerSourceViews, canonical_json


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True).stdout


@pytest.fixture
def case(tmp_path, request):
    root = tmp_path / "canonical"
    root.mkdir()
    (root / "pkg").mkdir()
    function = b"def answer():\n    value = 1  # ordinary explanation\n    return value\n"
    notes = b"# ordinary long fixture explanation, not part of returned value\n" * 80
    raw = function + notes
    variant = getattr(request, "param", "lf")
    if variant == "crlf":
        raw = raw.replace(b"\n", b"\r\n")
    elif variant == "no_final_lf":
        # The changed hunk reaches the EOF line, exercising Git's explicit
        # missing-final-newline marker rather than just leaving distant EOF alone.
        raw = notes + function.rstrip(b"\n")
    sources = {"pkg/app.py": raw, "pkg/other.py": b"OTHER = 3\n", "support.txt": b"public input\n"}
    for path, content in sources.items():
        (root / path).write_bytes(content)
    git(root, "init", "-q")
    git(root, "-c", "core.autocrlf=false", "add", "--all")
    git(root, "-c", "user.name=Authored Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "baseline")
    lifecycle = WorktreeLifecycleStore(repo_root=root, store_dir=tmp_path / "lifecycle", lease_seconds=300)
    with create_isolated_worktree(repo_root=root, worktree_path=tmp_path / "allocated",
            task_id="compact-edit-task", lifecycle_store=lifecycle) as owner:
        views = WorkerSourceViews(repository=root, workspace=owner.worktree_path,
            source_hashes={path: sha(content) for path, content in sources.items()},
            task_id=owner.task_id, scope_cid="scope:authored", semantic_root_cid="semantic:authored",
            semantic_sha256="a" * 64, native_prompt_sha256="b" * 64,
            source_manifest_sha256="c" * 64, profile="python-comments@1")
        first = views.working_page(path="pkg/app.py")
        compact = build_source_view(raw, profile="python-comments@1").view
        start = compact.index(b"1")
        scope = PatchScope(allowed_paths=("pkg/",), effect_paths=("pkg/app.py",), task_owned_paths=("pkg/",))
        args = dict(owner=owner, views=views, scope=scope, lease_id=owner.lease_id, fence=owner.fence,
            path="pkg/app.py", expected_workspace_generation=first["workspace_generation"],
            expected_source_sha256=first["source_sha256"], expected_view_sha256=first["view_sha256"],
            expected_manifest_sha256=first["manifest_sha256"], view_start=start, view_end=start + 1,
            expected_view_preimage=b"1", replacement=b"2")
        yield dict(root=root, owner=owner, views=views, sources=sources, args=args,
            expected=raw.replace(b"value = 1", b"value = 2", 1), baseline=views.baseline_admission())


def invoke(case, **kwargs):
    return subject.apply_source_view_edit_candidate(**(case["args"] | kwargs))


def assert_unwritten(case):
    assert all((case["root"] / path).read_bytes() == raw for path, raw in case["sources"].items())
    assert all((case["owner"].worktree_path / path).read_bytes() == raw for path, raw in case["sources"].items())
    assert not git(case["owner"].worktree_path, "status", "--porcelain=v1")


@pytest.mark.parametrize("case", ["lf", "crlf", "no_final_lf"], indirect=True)
def test_real_native_patch_preserves_bytes_stages_exact_result_and_refreshes_generation(case):
    old = case["views"].working_page(path="pkg/app.py", view="original", max_bytes=2400,
        expected_workspace_generation=case["args"]["expected_workspace_generation"])
    assert old["next_cursor"] is not None
    result = invoke(case)
    assert result["status"] == "applied_validation_pending"
    assert result["application"]["applied"] is True
    assert result["source_generation"] != result["workspace_generation"]
    assert result["after_source_sha256"] == sha(case["expected"])
    assert result["validation_pending"] is True
    assert result["baseline_evidence_applicable"] is result["proof_authority"] is False
    assert result["execution_authority"] is result["publication_authority"] is result["completion_authority"] is False
    assert result["source_obligations_discharged"] == result["prover_calls"] == result["provider_calls"] == 0
    assert result["receipt_sha256"] == sha(canonical_json({key: value for key, value in result.items() if key != "receipt_sha256"}))
    expected = case["sources"] | {"pkg/app.py": case["expected"]}
    assert case["views"].working_sources(expected_workspace_generation=result["workspace_generation"]) == expected
    assert case["views"].baseline_admission() == case["baseline"]
    assert all((case["root"] / path).read_bytes() == raw for path, raw in case["sources"].items())
    assert git(case["owner"].worktree_path, "show", ":pkg/app.py") == case["expected"]
    assert git(case["owner"].worktree_path, "diff", "--cached", "--name-only", "-z") == b"pkg/app.py\0"
    assert not git(case["owner"].worktree_path, "diff", "--name-only")
    with pytest.raises(ValueError, match="generation"):
        case["views"].working_page(path="pkg/app.py", view="original", max_bytes=2400,
            cursor=old["next_cursor"], expected_workspace_generation=old["workspace_generation"])


@pytest.mark.parametrize("field", ["expected_workspace_generation", "expected_source_sha256", "expected_view_sha256", "expected_manifest_sha256"])
def test_changed_required_pin_refuses_before_write(case, field):
    with pytest.raises(ValueError):
        invoke(case, **{field: "0" * 64})
    assert_unwritten(case)


@pytest.mark.parametrize("field", ["lease_id", "fence"])
def test_stale_native_lease_or_fence_refuses(case, field):
    with pytest.raises((ValueError, WorktreeFenceError)):
        invoke(case, **{field: "stale-owner" if field == "lease_id" else case["owner"].fence + 1})
    assert_unwritten(case)


@pytest.mark.parametrize("location", ["root", "workspace"])
@pytest.mark.parametrize("path", ["pkg/app.py", "pkg/other.py", "support.txt"])
def test_any_admitted_source_drift_refuses_without_mapping_or_application(case, location, path):
    base = case["root"] if location == "root" else case["owner"].worktree_path
    (base / path).write_bytes(b"concurrent edit\n")
    with pytest.raises(ValueError):
        invoke(case)
    assert (base / path).read_bytes() == b"concurrent edit\n"
    if path != "pkg/app.py" or location == "root":
        assert (case["owner"].worktree_path / "pkg/app.py").read_bytes() == case["sources"]["pkg/app.py"]
    assert not git(case["owner"].worktree_path, "diff", "--cached", "--name-only")


def test_owner_supplied_scope_cannot_be_widened_by_edit(case):
    scope = PatchScope(allowed_paths=("pkg/other.py",), task_owned_paths=("pkg/other.py",))
    with pytest.raises(ValueError, match="scope"):
        invoke(case, scope=scope)
    assert_unwritten(case)


@pytest.mark.parametrize("kind", ["canonical", "foreign_workspace", "foreign_task"])
def test_wrong_owner_pairing_or_task_refuses(case, kind, tmp_path):
    owner = case["owner"]
    before_path, before_task = owner.worktree_path, owner.task_id
    try:
        if kind == "canonical":
            owner.worktree_path = case["root"]
        elif kind == "foreign_workspace":
            other = tmp_path / "other"
            git(case["root"], "worktree", "add", "--detach", "-q", str(other))
            owner.worktree_path = other
        else:
            owner.task_id = "different-task"
        with pytest.raises(ValueError):
            invoke(case)
    finally:
        owner.worktree_path, owner.task_id = before_path, before_task
    assert_unwritten(case)


def test_unadmitted_file_is_not_made_editable_by_broad_scope(case):
    with pytest.raises(ValueError, match="population"):
        invoke(case, path="missing.py", scope=PatchScope(allowed_paths=("*",)))
    assert_unwritten(case)


def test_unadmitted_dirty_file_refuses_without_changing_it(case):
    target = case["owner"].worktree_path / "unadmitted.txt"
    target.write_bytes(b"unrelated work\n")
    with pytest.raises(ValueError, match="clean"):
        invoke(case)
    assert target.read_bytes() == b"unrelated work\n"
    assert (case["owner"].worktree_path / "pkg/app.py").read_bytes() == case["sources"]["pkg/app.py"]


def test_second_mapped_edit_requires_new_clean_candidate(case):
    invoke(case)
    with pytest.raises(ValueError, match="fresh"):
        invoke(case)
    assert (case["owner"].worktree_path / "pkg/app.py").read_bytes() == case["expected"]


def test_source_drift_during_mapper_replay_is_caught(case, monkeypatch):
    original = subject.map_source_view_edit
    def racing(**kwargs):
        result = original(**kwargs)
        (case["owner"].worktree_path / "support.txt").write_bytes(b"changed during map\n")
        return result
    monkeypatch.setattr(subject, "map_source_view_edit", racing)
    with pytest.raises(ValueError):
        invoke(case)
    assert (case["owner"].worktree_path / "pkg/app.py").read_bytes() == case["sources"]["pkg/app.py"]


def test_candidate_whole_scope_byte_budget_is_checked_before_mutation(case, monkeypatch):
    monkeypatch.setattr(subject, "MAX_SOURCE_BYTES", sum(map(len, case["sources"].values())) + 1)
    with pytest.raises(ValueError, match="source byte bound"):
        invoke(case, replacement=b"200")
    assert_unwritten(case)


@pytest.mark.parametrize("drift", ["fence", "base"])
def test_postapplication_native_fence_or_base_drift_refuses_receipt(case, monkeypatch, drift):
    original = IsolatedWorktree.apply_patch
    renewed = []
    def racing(owner, *args, **kwargs):
        result = original(owner, *args, **kwargs)
        assert result.applied
        if drift == "fence":
            renewed.append(owner.lifecycle_store.renew_lease(owner.worktree_path,
                lease_id=owner.lease_id, expected_fence=owner.fence))
        else:
            git(owner.worktree_path, "-c", "user.name=Authored Test", "-c", "user.email=test@example.invalid",
                "commit", "-qm", "authored concurrent candidate commit")
        return result
    monkeypatch.setattr(IsolatedWorktree, "apply_patch", racing)
    try:
        with pytest.raises((ValueError, WorktreeFenceError)):
            invoke(case)
        assert case["views"]._working is None
        assert (case["owner"].worktree_path / "pkg/app.py").read_bytes() == case["expected"]
    finally:
        if renewed:
            case["owner"]._sync_fence_from_record(renewed[0])


@pytest.mark.parametrize("drift", ["sibling", "extra_staged", "target"])
def test_postapplication_drift_never_emits_success_or_undoes_concurrent_bytes(case, monkeypatch, drift):
    original = IsolatedWorktree.apply_patch
    def racing(owner, *args, **kwargs):
        result = original(owner, *args, **kwargs)
        assert result.applied
        path = "pkg/other.py" if drift == "sibling" else "pkg/app.py" if drift == "target" else "extra.txt"
        (owner.worktree_path / path).write_bytes(b"post-apply concurrent bytes\n")
        if drift == "extra_staged":
            git(owner.worktree_path, "add", "--", path)
        return result
    monkeypatch.setattr(IsolatedWorktree, "apply_patch", racing)
    with pytest.raises(ValueError, match="path set|postimage"):
        invoke(case)
    assert case["views"]._working is None
    path = "pkg/other.py" if drift == "sibling" else "pkg/app.py" if drift == "target" else "extra.txt"
    assert (case["owner"].worktree_path / path).read_bytes() == b"post-apply concurrent bytes\n"


@pytest.mark.parametrize("kind", ["owner", "views", "scope"])
def test_only_exact_native_owners_and_scope_are_accepted(case, kind):
    with pytest.raises(ValueError, match="exact native"):
        invoke(case, **{kind: {}})
    assert_unwritten(case)


@pytest.mark.parametrize("path", ["white space.py", "unicodé.py", "line\nbreak.py", "../code.py"])
def test_unsupported_patch_path_is_explicitly_refused(path):
    with pytest.raises(ValueError):
        subject._patch(path, b"x = 1\n", b"x = 2\n")


@pytest.mark.parametrize("raw", [b"x\0=1\n", b"x\xff=1\n", b"x\r=1\n", "x\u2028=1\n".encode()])
def test_unsupported_text_is_explicitly_refused(raw):
    with pytest.raises(ValueError):
        subject._patch("code.py", raw, b"x=2\n")


def test_no_final_newline_marker_is_emitted_without_normalizing_crlf():
    patch = subject._patch("code.py", b"x = 1\r\ny = 1", b"x = 2\r\ny = 2")
    assert "-x = 1\r\n" in patch and "+x = 2\r\n" in patch
    assert patch.count("\\ No newline at end of file\n") == 2
