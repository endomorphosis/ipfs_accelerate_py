"""Shared host view recipes across historical custody, working reads and edits."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import WorktreeLifecycleStore
from ipfs_accelerate_py.agent_supervisor.semantic_state.worktree import PatchScope, create_isolated_worktree
from ipfs_accelerate_py.agent_supervisor.runtime import repository_code_view_inventory as inventory
from ipfs_accelerate_py.agent_supervisor.runtime import source_view_edit_candidate as edits
from ipfs_accelerate_py.agent_supervisor.runtime import source_view_policy as policies
from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as client
from ipfs_accelerate_py.agent_supervisor.runtime.reversible_source_view import build_source_view, current_codec_binding
from ipfs_accelerate_py.agent_supervisor.runtime.worker_source_views import WorkerSourceViews, canonical_json
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
from test.api.test_repository_source_inventory import _commit_request, _repository, _retain
from test.api.test_source_view_edit_candidate import git
from test.api.test_worker_source_client import descriptor, request, response
from test.api.test_worker_source_views import decode, pages, sha


RAW = (b"VALUE: int = 1      \r\n"
       b"# blue annotation padding       \r\n"
       + 'OTHER: str = "caf\u00e9"       \r\n'.encode()
       + b"# ordinary explanatory padding repeated for exact pagination\r\n" * 80)


@pytest.fixture
def case(tmp_path):
    files = {b"pkg/module.py": RAW, b"pkg/types.pyi": RAW,
             b"support.txt": b"public support    \r\n"}
    root = _repository(tmp_path, files)
    (root / "pkg/types.pyi").chmod(0o755)
    source = _retain(root, tmp_path)
    lifecycle = WorktreeLifecycleStore(repo_root=root, store_dir=tmp_path / "lifecycle", lease_seconds=300)
    with create_isolated_worktree(repo_root=root, worktree_path=tmp_path / "allocated",
            task_id=source.request.task_id, lifecycle_store=lifecycle) as owner:
        kwargs = dict(repository=root, workspace=owner.worktree_path,
            source_hashes={path.decode(): sha(raw) for path, raw in files.items()},
            task_id=owner.task_id, scope_cid="scope:shared-policy", semantic_root_cid="semantic:shared-policy",
            semantic_sha256="a" * 64, native_prompt_sha256="b" * 64, source_manifest_sha256="c" * 64)
        yield SimpleNamespace(root=root, source=source, owner=owner, files=files, kwargs=kwargs)


def policy(**changes):
    return policies.SourceViewPolicy.from_mapping(**{
        "profile": "python-comments@1",
        "protected_lines": {b"pkg/module.py".hex(): [2], b"pkg/types.pyi".hex(): [2]},
    } | changes)


def retain(case, **changes):
    return inventory.retain_code_view_inventory(cas=case.source.cas,
        expected_source_inventory_cid=case.source.ref.inventory_cid,
        request=case.source.request, source_limits=case.source.limits, **changes)


def load(case, ref, **changes):
    return inventory.load_code_view_inventory(cas=case.source.cas,
        expected_inventory_cid=ref.inventory_cid,
        expected_source_inventory_cid=case.source.ref.inventory_cid,
        request=case.source.request, source_limits=case.source.limits, **changes)


def no_call(*args, **kwargs):
    pytest.fail("unqualified producer or writer was called")


def test_policy_detaches_input_and_exported_bindings_and_is_frozen():
    lines = {b"pkg/types.pyi".hex(): [2]}
    selected = policies.SourceViewPolicy.from_mapping(profile="python-comments@1", protected_lines=lines)
    before = selected.binding()
    lines[b"pkg/types.pyi".hex()].append(3)
    exported = selected.to_dict()
    exported["protected_lines"][b"pkg/types.pyi".hex()].append(4)
    binding = selected.binding()
    binding["policy"]["protected_lines"].clear()
    assert selected.binding() == before
    assert selected.resolve(raw_path=b"pkg/types.pyi", regular_file=True).protected_lines == (2,)
    with pytest.raises(FrozenInstanceError):
        selected.profile = "identity@1"


@pytest.mark.parametrize("path", [b"pkg/module.py", b"pkg/types.pyi"])
def test_identity_profile_retains_python_eligibility_and_explicit_line_policy(path):
    selected = policy(profile="identity@1")
    resolved = selected.resolve(raw_path=path, regular_file=True)
    assert resolved.python_eligible is True
    assert resolved.profile == "identity@1" and resolved.protected_lines == (2,)
    projected = selected.project(RAW, raw_path=path, regular_file=True)
    assert projected.view == RAW and projected.manifest["protected_lines"] == [2]
    assert projected.manifest["effective_profile"] == "identity@1"


@pytest.mark.parametrize("changes", [
    {"profile": "unknown@1"},
    {"protected_lines": []},
    {"protected_lines": {"zz": [2]}},
    {"protected_lines": {b"pkg/types.pyi".hex().upper(): [2]}},
    {"protected_lines": {b"pkg/types.pyi".hex(): []}},
    {"protected_lines": {b"pkg/types.pyi".hex(): [True]}},
    {"protected_lines": {b"pkg/types.pyi".hex(): [0]}},
    {"protected_lines": {b"pkg/types.pyi".hex(): [3, 2]}},
    {"protected_lines": {b"pkg/types.pyi".hex(): [2, 2]}},
])
def test_malformed_policy_has_no_usable_recipe(changes):
    with pytest.raises(ValueError):
        policy(**changes)


@pytest.mark.parametrize("path,regular", [(b"notes.txt", True), (b"alias.py", False)])
def test_nonpython_and_nonregular_protected_references_refuse(path, regular):
    selected = policy(protected_lines={path.hex(): [1]})
    with pytest.raises(ValueError):
        selected.resolve(raw_path=path, regular_file=regular)


@pytest.mark.parametrize("path,regular", [(b"notes.txt", True), (b"alias.py", False)])
def test_unprotected_nonpython_and_nonregular_sources_remain_identity(path, regular):
    selected = policy(protected_lines={})
    resolved = selected.resolve(raw_path=path, regular_file=regular)
    assert resolved.python_eligible is False and resolved.profile == "identity@1"
    assert selected.project(b"not Python    \r\n", raw_path=path, regular_file=regular).view == b"not Python    \r\n"


@pytest.mark.parametrize("path,regular", [("module.py", True), (b"", True), (b"module.py", 1), (b"module.py", None)])
def test_resolution_requires_exact_path_bytes_and_boolean_file_kind(path, regular):
    with pytest.raises(ValueError):
        policy(protected_lines={}).resolve(raw_path=path, regular_file=regular)


def test_source_relative_protected_line_bounds_are_checked_at_projection():
    selected = policy(protected_lines={b"module.py".hex(): [3]})
    with pytest.raises(ValueError):
        selected.project(b"VALUE = 1\n", raw_path=b"module.py", regular_file=True)


def test_shared_inventory_worker_and_direct_codec_match_for_py_and_executable_pyi(case):
    selected = policy()
    before_codec = current_codec_binding()
    ref = retain(case, view_policy=selected)
    restored = load(case, ref, view_policy=selected)
    root = case.source.cas.get(ref.inventory_cid)
    assert root["schema"] == "supervisor-historical-code-view-inventory@2"
    assert root["view_policy_binding"] == selected.binding()
    entries = {bytes.fromhex(item.entry.raw_path_hex): item for item in restored.entries}
    worker = WorkerSourceViews(**case.kwargs, view_policy=selected)
    admission = worker.admission()
    assert admission["schema"] == "supervisor-worker-source-view-admission@2"
    assert admission["view_policy_binding"] == selected.binding()
    for path in (b"pkg/module.py", b"pkg/types.pyi"):
        expected = build_source_view(RAW, profile="python-comments@1", protected_lines=(2,))
        actual, rows = pages(worker, path=path.decode(), max_bytes=2400)
        assert actual == entries[path].view == expected.view
        assert entries[path].manifest == expected.manifest
        assert rows[0]["manifest_sha256"] == expected.manifest_sha256
        assert rows[0]["schema"] == "supervisor-worker-source-view-page@3"
        assert rows[0]["view_policy_binding_sha256"] == sha(canonical_json(selected.binding()))
        assert b"# blue annotation padding       \r\n" in actual
        assert b"ordinary explanatory padding" not in actual
    assert entries[b"pkg/types.pyi"].entry.git_mode == "100755"
    assert decode(worker.page(path="support.txt")) == case.files[b"support.txt"]
    assert worker.discovery()["schema"] == "supervisor-worker-source-view-discovery@2"
    assert current_codec_binding() == before_codec


def test_identity_policy_with_protected_lines_is_admitted_consistently(case):
    selected = policy(profile="identity@1")
    ref = retain(case, view_policy=selected)
    restored = load(case, ref, view_policy=selected)
    worker = WorkerSourceViews(**case.kwargs, view_policy=selected)
    for item in restored.entries:
        path = bytes.fromhex(item.entry.raw_path_hex)
        assert item.view == case.files[path]
        assert decode(worker.page(path=path.decode())) == case.files[path]
        if path.endswith((b".py", b".pyi")):
            assert item.manifest["protected_lines"] == [2]


def test_legacy_inventory_identity_and_worker_stub_default_are_unchanged(case):
    before = retain(case)
    original_root = case.source.cas.get(before.inventory_cid)
    legacy_worker = WorkerSourceViews(**case.kwargs)
    legacy_page = legacy_worker.page(path="pkg/types.pyi")
    assert legacy_page["schema"] == "supervisor-worker-source-view-page@1"
    assert legacy_page["effective_profile"] == "identity@1"
    assert legacy_page["fallback_reason"] == "non_python_identity"
    assert decode(legacy_page) == RAW
    assert "view_policy_binding" not in legacy_worker.admission()
    assert "view_policy_binding_sha256" not in legacy_page
    retain(case, view_policy=policy())
    after = retain(case)
    assert after.inventory_cid == before.inventory_cid
    assert case.source.cas.get(after.inventory_cid) == original_root
    assert original_root["schema"] == "supervisor-historical-code-view-inventory@1"
    assert "view_policy_binding" not in original_root
    expected = build_source_view(RAW)
    legacy_entries = {bytes.fromhex(item.entry.raw_path_hex): item for item in load(case, after).entries}
    assert legacy_entries[b"pkg/types.pyi"].manifest == expected.manifest
    assert legacy_entries[b"pkg/types.pyi"].view == expected.view


@pytest.mark.parametrize("changes", [{"profile": "python-comments@1"}, {"protected_lines": {}}])
def test_shared_inventory_refuses_duplicate_legacy_configuration(case, changes):
    with pytest.raises(ValueError):
        retain(case, view_policy=policy(), **changes)


def test_shared_worker_refuses_duplicate_legacy_profile(case):
    with pytest.raises(ValueError):
        WorkerSourceViews(**case.kwargs, profile="python-comments@1", view_policy=policy())


@pytest.mark.parametrize("consumer", ["inventory", "worker"])
def test_shared_configuration_rejects_default_equal_profile_subclass(case, consumer):
    class Profile(str):
        pass

    selected = policy()
    with pytest.raises(ValueError):
        if consumer == "inventory":
            retain(case, profile=Profile("python-whitespace@1"), view_policy=selected)
        else:
            WorkerSourceViews(**case.kwargs, profile=Profile("python-whitespace@1"), view_policy=selected)


@pytest.mark.parametrize("raw_path", [b"foreign.py", b"support.txt"])
def test_policy_cannot_add_foreign_or_nonpython_source_scope(case, raw_path):
    selected = policy(protected_lines={raw_path.hex(): [1]})
    with pytest.raises(ValueError):
        retain(case, view_policy=selected)
    with pytest.raises(ValueError):
        WorkerSourceViews(**case.kwargs, view_policy=selected)


def test_inventory_refuses_protected_symlink_even_when_name_is_python(tmp_path):
    root = _repository(tmp_path, {b"module.py": b"VALUE = 1\n"})
    (root / "alias.py").symlink_to("module.py")
    source = _retain(root, tmp_path, request=_commit_request(root))
    selected = policy(protected_lines={b"alias.py".hex(): [1]})
    with pytest.raises(ValueError):
        inventory.retain_code_view_inventory(cas=source.cas,
            expected_source_inventory_cid=source.ref.inventory_cid, request=source.request,
            source_limits=source.limits, view_policy=selected)


def test_shared_inventory_replay_requires_the_exact_policy(case):
    selected = policy()
    ref = retain(case, view_policy=selected)
    with pytest.raises(ValueError):
        load(case, ref)
    with pytest.raises(ValueError):
        load(case, ref, view_policy=policy(protected_lines={b"pkg/module.py".hex(): [3]}))


def test_mutated_policy_during_inventory_io_refuses_before_artifact_bytes(case, monkeypatch):
    selected = policy()
    ref = retain(case, view_policy=selected)
    original = ImmutableCAS.get
    changed = []

    def read_then_mutate(store, cid, **kwargs):
        value = original(store, cid, **kwargs)
        if cid == ref.inventory_cid and not changed:
            changed.append(cid)
            object.__setattr__(selected, "profile", "identity@1")
        return value

    monkeypatch.setattr(ImmutableCAS, "get", read_then_mutate)
    monkeypatch.setattr(ImmutableCAS, "get_bytes", no_call)
    with pytest.raises(ValueError):
        load(case, ref, view_policy=selected)
    assert changed == [ref.inventory_cid]


@pytest.mark.parametrize("malformed", [False, True])
@pytest.mark.parametrize("operation", ["baseline", "working", "projection", "continuation"])
def test_policy_object_mutation_invalidates_all_existing_worker_custody(case, malformed, operation):
    selected = policy()
    worker = WorkerSourceViews(**case.kwargs, view_policy=selected)
    first = worker.working_page(path="pkg/types.pyi", view="original", max_bytes=2400)
    assert first["next_cursor"] is not None
    if malformed:
        object.__setattr__(selected, "protected_lines", (("zz", (2,)),))
    else:
        object.__setattr__(selected, "profile", "identity@1")
    with pytest.raises(ValueError):
        if operation == "baseline":
            worker.baseline_admission()
        elif operation == "working":
            worker.working_admission(expected_workspace_generation=first["workspace_generation"])
        elif operation == "projection":
            worker.working_projection(path="pkg/types.pyi", expected_workspace_generation=first["workspace_generation"])
        else:
            worker.working_page(path="pkg/types.pyi", view="original", max_bytes=2400,
                cursor=first["next_cursor"], expected_workspace_generation=first["workspace_generation"])


def test_fresh_working_stub_edit_keeps_policy_and_refuses_old_generation(case):
    selected = policy()
    worker = WorkerSourceViews(**case.kwargs, view_policy=selected)
    first = worker.working_page(path="pkg/types.pyi")
    before = worker.baseline_admission()
    changed = RAW.replace(b"VALUE: int = 1", b"VALUE: int = 2", 1)
    (case.owner.worktree_path / "pkg/types.pyi").write_bytes(changed)
    latest = worker.working_page(path="pkg/types.pyi")
    assert latest["schema"] == "supervisor-worker-source-view-page@4"
    assert latest["workspace_generation"] != first["workspace_generation"]
    assert latest["view_policy_binding_sha256"] == first["view_policy_binding_sha256"]
    projected = worker.working_projection(path="pkg/types.pyi", expected_workspace_generation=latest["workspace_generation"])
    assert projected.manifest == selected.project(changed, raw_path=b"pkg/types.pyi", regular_file=True).manifest
    assert b"# blue annotation padding       \r\n" in projected.view
    assert worker.baseline_admission() == before
    assert worker.working_admission(expected_workspace_generation=latest["workspace_generation"])["view_policy_binding"] == selected.binding()
    with pytest.raises(ValueError):
        worker.working_projection(path="pkg/types.pyi", expected_workspace_generation=first["workspace_generation"])


@pytest.mark.parametrize("tampered_path", ["pkg/types.pyi", "pkg/module.py"])
def test_coherently_tampered_cached_original_and_view_do_not_authenticate_a_generation(case, tampered_path):
    selected = policy()
    worker = WorkerSourceViews(**case.kwargs, view_policy=selected)
    first = worker.working_page(path="pkg/types.pyi", view="original", max_bytes=2400)
    candidate = worker._working
    changed = RAW.replace(b"VALUE: int = 1", b"VALUE: int = 99", 1)
    projected = selected.project(changed, raw_path=tampered_path.encode(), regular_file=True)
    candidate._originals[tampered_path] = changed
    candidate._views[tampered_path] = (projected.view, canonical_json(projected.manifest), projected.manifest_sha256)
    with pytest.raises(ValueError):
        worker.working_projection(path="pkg/types.pyi", expected_workspace_generation=first["workspace_generation"])
    assert worker._working is None and candidate._continuations == {}
    assert (case.owner.worktree_path / tampered_path).read_bytes() == RAW


@pytest.mark.parametrize("path", ["pkg/module.py", "pkg/types.pyi"])
@pytest.mark.parametrize("preimage,replacement", [(b"blue", b"green"), (b"VALUE: int = 1", b"VALUE: int = 2")])
def test_real_mapped_edit_uses_shared_recipe_preserves_protected_comment_and_refreshes(case, path, preimage, replacement):
    selected = policy()
    worker = WorkerSourceViews(**case.kwargs, view_policy=selected)
    first = worker.working_page(path=path)
    old = worker.working_page(path=path, view="original", max_bytes=2400,
        expected_workspace_generation=first["workspace_generation"])
    assert old["next_cursor"] is not None
    projected = worker.working_projection(path=path, expected_workspace_generation=first["workspace_generation"])
    start = projected.view.index(preimage)
    scope = PatchScope(allowed_paths=("pkg/",), effect_paths=(path,), task_owned_paths=("pkg/",))
    result = edits.apply_source_view_edit_candidate(owner=case.owner, views=worker, scope=scope,
        lease_id=case.owner.lease_id, fence=case.owner.fence, path=path,
        expected_workspace_generation=first["workspace_generation"],
        expected_source_sha256=first["source_sha256"], expected_view_sha256=first["view_sha256"],
        expected_manifest_sha256=first["manifest_sha256"], view_start=start, view_end=start + len(preimage),
        expected_view_preimage=preimage, replacement=replacement)
    expected = RAW.replace(preimage, replacement, 1)
    assert result["schema"] == "supervisor-source-view-edit-application@2"
    assert result["view_policy_binding_sha256"] == sha(canonical_json(selected.binding()))
    assert result["status"] == "applied_validation_pending" and result["validation_pending"] is True
    assert result["source_generation"] == first["workspace_generation"]
    assert result["workspace_generation"] != first["workspace_generation"]
    assert git(case.owner.worktree_path, "show", ":" + path) == expected
    assert (case.root / path).read_bytes() == RAW
    assert b"ordinary explanatory padding" in expected
    latest = worker.working_projection(path=path, expected_workspace_generation=result["workspace_generation"])
    assert latest.manifest == selected.project(expected, raw_path=path.encode(), regular_file=True).manifest
    assert b"# " + (b"green" if preimage == b"blue" else b"blue") + b" annotation padding       \r\n" in latest.view
    assert all(result[name] is False for name in ("proof_authority", "execution_authority", "publication_authority", "completion_authority"))
    assert result["provider_calls"] == result["prover_calls"] == result["source_obligations_discharged"] == 0
    with pytest.raises(ValueError):
        worker.working_page(path=path, view="original", max_bytes=2400,
            cursor=old["next_cursor"], expected_workspace_generation=old["workspace_generation"])


def test_policy_mutation_during_mapping_refuses_before_native_application(case, monkeypatch):
    selected = policy()
    worker = WorkerSourceViews(**case.kwargs, view_policy=selected)
    first = worker.working_page(path="pkg/types.pyi")
    projected = worker.working_projection(path="pkg/types.pyi", expected_workspace_generation=first["workspace_generation"])
    start = projected.view.index(b"blue")
    original = edits.map_source_view_edit

    def map_then_mutate(**kwargs):
        result = original(**kwargs)
        object.__setattr__(selected, "profile", "identity@1")
        return result

    monkeypatch.setattr(edits, "map_source_view_edit", map_then_mutate)
    monkeypatch.setattr(case.owner, "apply_patch", no_call)
    scope = PatchScope(allowed_paths=("pkg/",), effect_paths=("pkg/types.pyi",), task_owned_paths=("pkg/",))
    with pytest.raises(ValueError):
        edits.apply_source_view_edit_candidate(owner=case.owner, views=worker, scope=scope,
            lease_id=case.owner.lease_id, fence=case.owner.fence, path="pkg/types.pyi",
            expected_workspace_generation=first["workspace_generation"],
            expected_source_sha256=first["source_sha256"], expected_view_sha256=first["view_sha256"],
            expected_manifest_sha256=first["manifest_sha256"], view_start=start, view_end=start + 4,
            expected_view_preimage=b"blue", replacement=b"green")
    assert (case.owner.worktree_path / "pkg/types.pyi").read_bytes() == RAW
    assert not git(case.owner.worktree_path, "status", "--porcelain=v1")


@pytest.mark.parametrize("working", [False, True])
def test_existing_closed_wire_rejects_shared_host_pages_instead_of_legacy_relabeling(case, working):
    worker = WorkerSourceViews(**case.kwargs, view_policy=policy())
    query = request(path="pkg/types.pyi", view="compact")
    if working:
        query["schema"] = client.REQUEST_SCHEMA_V2
        query["query"] = dict(source_state="working-candidate", expected_workspace_generation=None) | query["query"]
        page = worker.working_page(path="pkg/types.pyi")
    else:
        page = worker.page(path="pkg/types.pyi")
    with pytest.raises(ValueError):
        client.validate_response(response(page, query), request=query, descriptor=descriptor(page["binding"]))
