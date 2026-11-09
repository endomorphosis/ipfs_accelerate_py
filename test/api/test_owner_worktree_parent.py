"""Protected managed parents are born private and never permission-normalized.

Descriptors, directory creation, modes and named-inode races are real. Narrow
metadata seams author ownership, ACL or mount refusals; no privileged chown,
mount operation, provider or native scheduler reconfiguration is performed.
"""
import os
import stat
import time
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import owner_worktree_parent as parents
from ipfs_accelerate_py.agent_supervisor.todo_daemon import implementation_daemon as implementation
from test.api.test_owner_writable_worktree import snapshot, tracked_descriptors


@pytest.fixture
def parent_case(tmp_path):
    base = tmp_path / "owner"
    base.mkdir(mode=0o700)
    prefix = base / "protected"
    prefix.mkdir(mode=0o750)
    outside = base / "outside"
    outside.mkdir(mode=0o700)
    (outside / "sentinel").write_bytes(b"outside owner state unchanged")
    (outside / "sentinel").chmod(0o600)
    return SimpleNamespace(base=base, prefix=prefix, directory=prefix / "new" / "managed", outside=outside)


def prepare(case, **changes):
    return parents.prepare_owner_worktree_parent(**dict(directory=case.directory,
        absolute_deadline_ns=time.monotonic_ns() + 30_000_000_000) | changes)


def forbid_normalization(monkeypatch):
    for name in ("chmod", "fchmod", "chown", "fchown", "umask"):
        monkeypatch.setattr(parents.os, name,
            lambda *_args, **_kwargs: pytest.fail("managed parent preparation normalized permissions or owner state"))


def no_creation(case, monkeypatch, **changes):
    before, outside = snapshot(case.prefix), snapshot(case.outside)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("managed parent refusal created a directory")
    forbid_normalization(monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(parents.os, "mkdir", forbidden)
        with tracked_descriptors(patch):
            with pytest.raises(parents.OwnerWorktreeParentError) as error:
                prepare(case, **changes)
    assert calls == []
    assert error.value.mutation_attempted is False
    assert error.value.created_prefix == ()
    assert snapshot(case.prefix) == before
    assert snapshot(case.outside) == outside
    return error.value


def test_missing_components_are_born_0700_and_existing_prefix_is_unchanged(parent_case, monkeypatch):
    case = parent_case
    before = case.prefix.stat()
    actual = os.mkdir
    births = []
    def mkdir(path, mode=0o777, *, dir_fd=None):
        result = actual(path, mode, dir_fd=dir_fd)
        info = os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
        births.append((mode, stat.S_IMODE(info.st_mode)))
        return result
    forbid_normalization(monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(parents.os, "mkdir", mkdir)
        with tracked_descriptors(patch):
            receipt = prepare(case)
    assert births == [(0o700, 0o700), (0o700, 0o700)]
    assert stat.S_IMODE(case.directory.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(case.directory.stat().st_mode) == 0o700
    after = case.prefix.stat()
    assert (after.st_dev, after.st_ino, after.st_uid, after.st_gid, stat.S_IMODE(after.st_mode)) == (
        before.st_dev, before.st_ino, before.st_uid, before.st_gid, 0o750)
    assert type(receipt) is dict
    assert receipt["completion_authority"] is receipt["execution_authority"] is False


@pytest.mark.parametrize("mode", [0o700, 0o750])
def test_existing_protected_parent_is_reused_without_chmod_or_mkdir(parent_case, monkeypatch, mode):
    case = parent_case
    case.prefix.chmod(mode)
    case.directory = case.prefix
    before = snapshot(case.prefix)
    forbid_normalization(monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(parents.os, "mkdir", lambda *_args, **_kwargs:
            pytest.fail("existing protected parent was recreated"))
        with tracked_descriptors(patch):
            receipt = prepare(case)
    assert snapshot(case.prefix) == before
    assert type(receipt) is dict


@pytest.mark.parametrize("kind", ["writable", "setgid_prefix", "symlink_target", "symlink_ancestor", "file_target"])
def test_unsafe_existing_paths_refuse_without_creation_or_outside_effects(parent_case, monkeypatch, kind):
    case = parent_case
    if kind == "writable":
        case.prefix.chmod(0o775)
    elif kind == "setgid_prefix":
        case.prefix.chmod(0o2750)
    elif kind == "symlink_target":
        case.directory = case.prefix / "target-link"
        case.directory.symlink_to(case.outside, target_is_directory=True)
    elif kind == "symlink_ancestor":
        alias = case.base / "prefix-link"
        alias.symlink_to(case.prefix, target_is_directory=True)
        case.directory = alias / "new"
    else:
        case.directory = case.prefix / "not-a-directory"
        case.directory.write_bytes(b"ordinary owner file")
    no_creation(case, monkeypatch)


@pytest.mark.parametrize("field", ["st_uid", "st_gid"])
def test_consistently_foreign_opened_prefix_identity_refuses(parent_case, monkeypatch, field):
    case = parent_case
    expected = case.prefix.stat()
    actual_stat, actual_fstat = os.stat, os.fstat
    def foreign(info):
        if (info.st_dev, info.st_ino) == (expected.st_dev, expected.st_ino):
            values = {name: getattr(info, name) for name in dir(info) if name.startswith("st_")}
            values[field] += 1
            return SimpleNamespace(**values)
        return info
    monkeypatch.setattr(parents.os, "stat", lambda *args, **kwargs: foreign(actual_stat(*args, **kwargs)))
    monkeypatch.setattr(parents.os, "fstat", lambda fd: foreign(actual_fstat(fd)))
    no_creation(case, monkeypatch)


@pytest.mark.parametrize("name", ["system.posix_acl_access", "system.posix_acl_default"])
def test_acl_on_existing_prefix_refuses_before_creation(parent_case, monkeypatch, name):
    case = parent_case
    expected = case.prefix.stat()
    actual_list = os.listxattr
    def with_acl(descriptor):
        result = actual_list(descriptor)
        info = os.fstat(descriptor)
        return list(result) + [name] if (info.st_dev, info.st_ino) == (
            expected.st_dev, expected.st_ino) else result
    monkeypatch.setattr(parents.os, "listxattr", with_acl)
    no_creation(case, monkeypatch)


def test_existing_managed_target_mount_cannot_differ_from_its_parent(parent_case, monkeypatch):
    case = parent_case
    case.directory = case.prefix
    expected = case.prefix.stat()
    actual = parents._mount_identity
    def changed(descriptor):
        result = actual(descriptor)
        info = os.fstat(descriptor)
        return (result[0] + 1, result[1]) if (info.st_dev, info.st_ino) == (
            expected.st_dev, expected.st_ino) else result
    monkeypatch.setattr(parents, "_mount_identity", changed)
    no_creation(case, monkeypatch)


@pytest.mark.parametrize("constant,value", [("MAX_DEPTH", 1), ("MAX_PATH_BYTES", 1)])
def test_parent_path_bounds_refuse_without_mkdir(parent_case, monkeypatch, constant, value):
    monkeypatch.setattr(parents, constant, value)
    no_creation(parent_case, monkeypatch)


def test_expired_deadline_refuses_without_mkdir(parent_case, monkeypatch):
    no_creation(parent_case, monkeypatch, absolute_deadline_ns=time.monotonic_ns() - 1)


def test_suppressed_recursive_preparation_invalidates_outer_before_creation(parent_case, monkeypatch):
    case = parent_case
    actual = parents._existing_prefix
    nested = []
    def recursive(directory, **kwargs):
        result = actual(directory, **kwargs)
        if not nested:
            with pytest.raises(parents.OwnerWorktreeParentError):
                prepare(case)
            nested.append(True)
        return result
    monkeypatch.setattr(parents, "_existing_prefix", recursive)
    no_creation(case, monkeypatch)
    assert nested == [True]


def test_existing_prefix_replacement_after_preflight_refuses_before_creation(parent_case, monkeypatch):
    case = parent_case
    actual = parents._existing_prefix
    retained = case.outside / "retained-prefix"
    original = case.prefix.stat()
    actual_mkdir = os.mkdir
    replaced = []
    def changed(directory, **kwargs):
        result = actual(directory, **kwargs)
        if not replaced:
            case.prefix.rename(retained)
            actual_mkdir(case.prefix, 0o750)
            replaced.append(True)
        return result
    monkeypatch.setattr(parents, "_existing_prefix", changed)
    forbid_normalization(monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(parents.os, "mkdir", lambda *_args, **_kwargs:
            pytest.fail("stale prefix reached managed directory creation"))
        with tracked_descriptors(patch):
            with pytest.raises(parents.OwnerWorktreeParentError) as error:
                prepare(case)
    assert error.value.mutation_attempted is False
    assert error.value.created_prefix == ()
    assert (retained.stat().st_dev, retained.stat().st_ino) == (original.st_dev, original.st_ino)
    assert not case.directory.exists()


def test_eexist_race_is_never_adopted_as_owned_creation(parent_case, monkeypatch):
    case = parent_case
    actual = os.mkdir
    attempts = []
    def raced(path, mode=0o777, *, dir_fd=None):
        actual(path, 0o700, dir_fd=dir_fd)
        attempts.append(str(path))
        raise FileExistsError("authored concurrent directory creator")
    forbid_normalization(monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(parents.os, "mkdir", raced)
        with tracked_descriptors(patch):
            with pytest.raises(parents.OwnerWorktreeParentError) as error:
                prepare(case)
    assert len(attempts) == 1
    assert error.value.mutation_attempted is True
    assert error.value.created_prefix == ()
    assert error.value.creation_effects_complete is False
    assert case.directory.parent.is_dir() and not case.directory.exists()
    assert stat.S_IMODE(case.directory.parent.stat().st_mode) == 0o700


@pytest.fixture
def factory_owner(parent_case, monkeypatch):
    case = parent_case
    repository = case.base / "repository"
    repository.mkdir(mode=0o700)
    orchestration = case.base / "orchestration"
    orchestration.mkdir(mode=0o700)
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(orchestration))
    def construct(root):
        return implementation.PortalImplementationDaemon(todo_path=case.base / "tasks.todo.md",
            state_path=case.base / "state.json", strategy_path=case.base / "strategy.json",
            events_path=case.base / "events.jsonl", repo_root=repository,
            worktree_root=root, worktree_pool_enabled=False, worktree_submodule_paths=())
    root = case.base / "managed-factory-root"
    return SimpleNamespace(owner=construct(root), root=root, construct=construct)


@pytest.mark.parametrize("invalid", ["flag_int", "flag_string", "flag_none", "seed_true",
    "seed_integer", "pool", "configured_submodule", "task_submodule"])
def test_protected_factory_rejects_incompatible_profile_before_parent_or_git_mutation(
        factory_owner, monkeypatch, invalid):
    fixture = factory_owner
    owner = fixture.owner
    options = dict(allow_pool=False, seed_context=False, offline_local_only=True,
        protected_initial_parent=True)
    if invalid.startswith("flag_"):
        options["protected_initial_parent"] = {"flag_int": 1,
            "flag_string": "true", "flag_none": None}[invalid]
    elif invalid.startswith("seed_"):
        options["seed_context"] = True if invalid == "seed_true" else 0
    elif invalid == "pool":
        owner.worktree_pool = SimpleNamespace(acquire=lambda **_kwargs:
            pytest.fail("incompatible protected profile reached pooled acquisition"))
        options["allow_pool"] = True
    elif invalid == "configured_submodule":
        owner.worktree_submodule_paths = ("deps/child",)
    else:
        options["task"] = implementation.PortalTask(task_id="PARENT-001", title="Protected parent",
            status="todo", completion="manual", priority="P1", track="test", outputs=[],
            metadata={"submodules": "deps/child"})
    monkeypatch.setattr(owner, "_run_git", lambda *_args, **_kwargs:
        pytest.fail("incompatible protected profile reached Git creation"))
    monkeypatch.setattr(parents, "prepare_owner_worktree_parent", lambda **_kwargs:
        pytest.fail("incompatible protected profile reached parent preparation"))
    with pytest.raises(ValueError):
        owner._create_seeded_worktree(fixture.root / "candidate", "implementation/protected-parent", **options)
    assert not fixture.root.exists()


@pytest.mark.parametrize("kind", ["empty_directory", "broken_symlink"])
def test_protected_factory_requires_an_absent_candidate_before_parent_preparation(
        factory_owner, monkeypatch, kind):
    fixture = factory_owner
    fixture.root.mkdir(mode=0o700)
    candidate = fixture.root / "candidate"
    if kind == "empty_directory":
        candidate.mkdir(mode=0o700)
    else:
        candidate.symlink_to(fixture.root / "missing-target")
    before = snapshot(fixture.root)
    monkeypatch.setattr(fixture.owner, "_run_git", lambda *_args, **_kwargs:
        pytest.fail("existing candidate reached Git creation"))
    monkeypatch.setattr(parents, "prepare_owner_worktree_parent", lambda **_kwargs:
        pytest.fail("existing candidate reached parent preparation"))
    with pytest.raises(ValueError):
        fixture.owner._create_seeded_worktree(candidate, "implementation/protected-parent",
            allow_pool=False, seed_context=False, protected_initial_parent=True)
    assert snapshot(fixture.root) == before


@pytest.mark.parametrize("location", ["root_itself", "outside_root"])
def test_protected_factory_requires_a_proper_candidate_descendant(
        factory_owner, monkeypatch, location):
    fixture = factory_owner
    candidate = fixture.root if location == "root_itself" else fixture.root.parent / "foreign-candidate"
    monkeypatch.setattr(fixture.owner, "_run_git", lambda *_args, **_kwargs:
        pytest.fail("out-of-scope candidate reached Git creation"))
    monkeypatch.setattr(parents, "prepare_owner_worktree_parent", lambda **_kwargs:
        pytest.fail("out-of-scope candidate reached parent preparation"))
    with pytest.raises(ValueError):
        fixture.owner._create_seeded_worktree(candidate, "implementation/protected-parent",
            allow_pool=False, seed_context=False, protected_initial_parent=True)
    assert not fixture.root.exists() and not candidate.exists()


def test_protected_factory_retains_and_refuses_a_configured_symlink_alias(
        factory_owner, parent_case, monkeypatch):
    fixture = factory_owner
    fixture.root.mkdir(mode=0o700)
    alias = parent_case.base / "aliased-factory-root"
    alias.symlink_to(fixture.root, target_is_directory=True)
    owner = fixture.construct(alias)
    assert owner.worktree_root == fixture.root
    monkeypatch.setattr(owner, "_run_git", lambda *_args, **_kwargs:
        pytest.fail("configured root alias reached Git creation"))
    monkeypatch.setattr(parents, "prepare_owner_worktree_parent", lambda **_kwargs:
        pytest.fail("configured root alias reached parent preparation"))
    with pytest.raises(ValueError):
        owner._create_seeded_worktree(fixture.root / "candidate", "implementation/protected-parent",
            allow_pool=False, seed_context=False, protected_initial_parent=True)
    assert list(fixture.root.iterdir()) == []


def test_partial_creation_keeps_verified_private_prefix_without_rollback(parent_case, monkeypatch):
    case = parent_case
    actual = os.mkdir
    attempts = []
    def failed(path, mode=0o777, *, dir_fd=None):
        attempts.append(str(path))
        if len(attempts) == 2:
            raise OSError("authored second mkdir failure")
        return actual(path, mode, dir_fd=dir_fd)
    forbid_normalization(monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(parents.os, "mkdir", failed)
        with tracked_descriptors(patch):
            with pytest.raises(parents.OwnerWorktreeParentError) as error:
                prepare(case)
    assert error.value.mutation_attempted is True
    assert error.value.created_prefix == (str(case.directory.parent),)
    assert error.value.attempted_path == str(case.directory)
    assert error.value.creation_effects_complete is False
    assert case.directory.parent.is_dir() and not case.directory.exists()
    assert stat.S_IMODE(case.directory.parent.stat().st_mode) == 0o700


def test_mkdir_that_applies_then_raises_reports_unverified_creation(parent_case, monkeypatch):
    case = parent_case
    actual = os.mkdir
    def applied(path, mode=0o777, *, dir_fd=None):
        actual(path, mode, dir_fd=dir_fd)
        raise OSError("authored failure after mkdir effect")
    forbid_normalization(monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(parents.os, "mkdir", applied)
        with tracked_descriptors(patch):
            with pytest.raises(parents.OwnerWorktreeParentError) as error:
                prepare(case)
    assert error.value.mutation_attempted is True
    assert error.value.created_prefix == ()
    assert error.value.creation_effects_complete is False
    assert error.value.attempted_path == str(case.directory.parent)
    assert case.directory.parent.is_dir() and not case.directory.exists()
    assert stat.S_IMODE(case.directory.parent.stat().st_mode) == 0o700


def test_deadline_expiry_after_first_creation_retains_only_verified_prefix(parent_case, monkeypatch):
    case = parent_case
    deadline = time.monotonic_ns() + 30_000_000_000
    actual = os.mkdir
    calls = []
    def expired(path, mode=0o777, *, dir_fd=None):
        result = actual(path, mode, dir_fd=dir_fd)
        calls.append(True)
        monkeypatch.setattr(parents.time, "monotonic_ns", lambda: deadline + 1)
        return result
    forbid_normalization(monkeypatch)
    with monkeypatch.context() as patch:
        patch.setattr(parents.os, "mkdir", expired)
        with tracked_descriptors(patch):
            with pytest.raises(parents.OwnerWorktreeParentError) as error:
                prepare(case, absolute_deadline_ns=deadline)
    assert calls == [True]
    assert error.value.mutation_attempted is True
    assert error.value.created_prefix == (str(case.directory.parent),)
    assert error.value.creation_effects_complete is False
    assert case.directory.parent.is_dir() and not case.directory.exists()
    assert stat.S_IMODE(case.directory.parent.stat().st_mode) == 0o700
