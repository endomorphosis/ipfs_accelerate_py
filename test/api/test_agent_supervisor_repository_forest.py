"""Tests for independently bound repository descriptors and authority forests."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.repository_forest import (
    AuthorityMode,
    CaseUnicodePolicy,
    DEFAULT_ACCELERATOR_ALIAS,
    DEFAULT_SWISSKNIFE_ALIAS,
    DEFAULT_SWISSKNIFE_ROOT,
    ForestPolicy,
    ForestRootSpec,
    GITLINK_ENTRY_SCHEMA,
    IgnorePolicy,
    RepositoryAuthority,
    RepositoryForest,
    RepositoryForestError,
    build_repository_descriptor,
    build_repository_forest,
    empty_dirty_overlay_digest,
    forests_share_portable_identity,
    initial_vfs_assurance_forest_policy,
    make_repository_id,
    path_within_repository,
    resolve_repository_root,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_identity import (
    canonical_content_cid,
)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return (completed.stdout or "").strip()


def _init_repo(path: Path, *, name: str = "seed") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init")
    _git(path, "checkout", "-b", "main")
    _git(path, "config", "user.name", "Test User")
    _git(path, "config", "user.email", "test@example.invalid")
    (path / "README.md").write_text(f"# {name}\n", encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "-m", f"seed {name}")
    return path


def _add_submodule(parent: Path, child: Path, name: str) -> None:
    _git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child),
        name,
    )
    _git(parent, "commit", "-am", f"add {name} gitlink")


def _two_root_policy(
    swissknife: Path,
    accelerator: Path,
    **kwargs: object,
) -> ForestPolicy:
    return ForestPolicy(
        roots=(
            ForestRootSpec(
                alias=DEFAULT_SWISSKNIFE_ALIAS,
                root_path=swissknife,
                authority=RepositoryAuthority(mode=AuthorityMode.READ_ONLY.value),
            ),
            ForestRootSpec(
                alias=DEFAULT_ACCELERATOR_ALIAS,
                root_path=accelerator,
                authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
            ),
        ),
        sole_write_alias=DEFAULT_ACCELERATOR_ALIAS,
        **kwargs,
    )


def test_initial_policy_models_swissknife_readonly_and_accelerator_write(
    tmp_path: Path,
) -> None:
    swiss = _init_repo(tmp_path / "swissknife", name="swissknife")
    accel = _init_repo(tmp_path / "accelerator", name="accelerator")
    policy = initial_vfs_assurance_forest_policy(
        accelerator_root=accel,
        swissknife_root=swiss,
    )
    assert policy.sole_write_alias == DEFAULT_ACCELERATOR_ALIAS
    by_alias = {root.alias: root for root in policy.roots}
    assert DEFAULT_SWISSKNIFE_ALIAS in by_alias
    assert DEFAULT_ACCELERATOR_ALIAS in by_alias
    swiss_auth = by_alias[DEFAULT_SWISSKNIFE_ALIAS].authority
    accel_auth = by_alias[DEFAULT_ACCELERATOR_ALIAS].authority
    assert isinstance(swiss_auth, RepositoryAuthority)
    assert isinstance(accel_auth, RepositoryAuthority)
    assert swiss_auth.mode == AuthorityMode.READ_ONLY.value
    assert accel_auth.mode == AuthorityMode.READ_WRITE.value
    # Default constant matches the frozen plan location.
    assert DEFAULT_SWISSKNIFE_ROOT == "/home/barberb/swissknife"


def test_clean_descriptor_is_deterministic(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    first = build_repository_descriptor(
        repo,
        alias="ipfs_accelerate_py",
        authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
    )
    second = build_repository_descriptor(
        repo,
        alias="ipfs_accelerate_py",
        authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
    )
    assert first.dirty is False
    assert first.dirty_overlay_digest == empty_dirty_overlay_digest()
    assert first.repository_id == second.repository_id
    assert first.descriptor_cid == second.descriptor_cid
    assert first.commit == second.commit
    assert first.tree == second.tree
    assert first.to_portable_dict() == second.to_portable_dict()


def test_dirty_overlay_changes_identity(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    clean = build_repository_descriptor(
        repo,
        alias="ipfs_accelerate_py",
        authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
    )
    (repo / "dirty.txt").write_text("one\n", encoding="utf-8")
    dirty_one = build_repository_descriptor(
        repo,
        alias="ipfs_accelerate_py",
        authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
    )
    (repo / "dirty.txt").write_text("two\n", encoding="utf-8")
    dirty_two = build_repository_descriptor(
        repo,
        alias="ipfs_accelerate_py",
        authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
    )
    assert dirty_one.dirty is True
    assert dirty_one.dirty_overlay_digest != empty_dirty_overlay_digest()
    assert dirty_one.descriptor_cid != clean.descriptor_cid
    assert dirty_two.dirty_overlay_digest != dirty_one.dirty_overlay_digest
    assert dirty_two.descriptor_cid != dirty_one.descriptor_cid


def test_submodule_gitlink_closure_affects_identity(tmp_path: Path) -> None:
    child = _init_repo(tmp_path / "child", name="child")
    parent = _init_repo(tmp_path / "parent", name="parent")
    without = build_repository_descriptor(
        parent,
        alias="ipfs_accelerate_py",
        authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
    )
    _add_submodule(parent, child, "child-component")
    with_link = build_repository_descriptor(
        parent,
        alias="ipfs_accelerate_py",
        authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
    )
    assert without.portable_closure.gitlinks == ()
    assert len(with_link.portable_closure.gitlinks) == 1
    assert with_link.descriptor_cid != without.descriptor_cid
    # Opaque gitlink ids must not leak host paths into portable JSON.
    portable = json.dumps(with_link.to_portable_dict(), sort_keys=True)
    assert "child-component" not in portable
    assert str(child) not in portable


def test_nested_gitlinks_are_scoped_to_their_distinct_parent_mounts(
    tmp_path: Path,
) -> None:
    leaf = _init_repo(tmp_path / "leaf", name="leaf")
    child = _init_repo(tmp_path / "child", name="child")
    _add_submodule(child, leaf, "nested/leaf")
    child_commit = _git(child, "rev-parse", "HEAD")

    parent = _init_repo(tmp_path / "parent", name="parent")
    _add_submodule(parent, child, "vendor/child-one")
    _add_submodule(parent, child, "vendor/child-two")

    descriptor = build_repository_descriptor(
        parent,
        alias="ipfs_accelerate_py",
        authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
    )
    entries = descriptor.portable_closure.gitlinks
    mounted_children = [
        item for item in entries if item.depth == 0 and item.commit == child_commit
    ]
    nested_leaves = [item for item in entries if item.depth == 1]

    assert len(mounted_children) == 2
    assert len({item.gitlink_id for item in mounted_children}) == 2
    assert {item.gitlink_id for item in mounted_children} == {
        canonical_content_cid(
            {
                "schema": GITLINK_ENTRY_SCHEMA + "/location",
                "parent_commit": descriptor.commit,
                "location": location,
            }
        )
        for location in ("vendor/child-one", "vendor/child-two")
    }
    assert len(nested_leaves) == 2
    assert len({item.gitlink_id for item in nested_leaves}) == 2
    assert {item.parent_gitlink_id for item in nested_leaves} == {
        item.gitlink_id for item in mounted_children
    }


def test_submodule_content_change_changes_closure(tmp_path: Path) -> None:
    child = _init_repo(tmp_path / "child", name="child")
    parent = _init_repo(tmp_path / "parent", name="parent")
    _add_submodule(parent, child, "child-component")
    before = build_repository_descriptor(
        parent,
        alias="ipfs_accelerate_py",
        authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
    )
    checkout = parent / "child-component"
    (checkout / "extra.py").write_text("X = 1\n", encoding="utf-8")
    _git(checkout, "add", "extra.py")
    _git(
        checkout,
        "-c",
        "user.name=Test User",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "advance child",
    )
    after = build_repository_descriptor(
        parent,
        alias="ipfs_accelerate_py",
        authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
    )
    assert after.portable_closure.gitlinks[0].tree != before.portable_closure.gitlinks[0].tree
    assert after.descriptor_cid != before.descriptor_cid
    assert "gitlink_head_mismatch" in after.reason_codes


def test_path_escape_is_rejected(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    descriptor = build_repository_descriptor(
        repo,
        alias="ipfs_accelerate_py",
        authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
    )
    with pytest.raises(RepositoryForestError) as escaped:
        path_within_repository(descriptor, "../outside.txt")
    assert escaped.value.reason_code == "path_escape"
    with pytest.raises(RepositoryForestError) as absolute:
        path_within_repository(descriptor, outside)
    assert absolute.value.reason_code == "path_escape"
    ok = path_within_repository(descriptor, "README.md", require_existing=True)
    assert ok == (descriptor.root_path / "README.md").resolve()


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    outside = tmp_path / "outside-dir"
    outside.mkdir()
    (outside / "secret.txt").write_text("nope\n", encoding="utf-8")
    link = repo / "escape-link"
    link.symlink_to(outside, target_is_directory=True)
    descriptor = build_repository_descriptor(
        repo,
        alias="ipfs_accelerate_py",
        authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
    )
    with pytest.raises(RepositoryForestError) as excinfo:
        path_within_repository(descriptor, "escape-link/secret.txt")
    assert excinfo.value.reason_code == "path_escape"


def test_missing_root_fails_closed(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(RepositoryForestError) as excinfo:
        resolve_repository_root(missing)
    assert excinfo.value.reason_code == "missing_root"
    swiss = _init_repo(tmp_path / "swiss")
    with pytest.raises(RepositoryForestError) as forest_exc:
        build_repository_forest(
            _two_root_policy(swiss, missing),
            fail_on_missing_required=True,
        )
    assert forest_exc.value.reason_code == "missing_root"


def test_duplicate_aliases_rejected(tmp_path: Path) -> None:
    repo_a = _init_repo(tmp_path / "a")
    repo_b = _init_repo(tmp_path / "b")
    with pytest.raises(RepositoryForestError) as excinfo:
        ForestPolicy(
            roots=(
                ForestRootSpec(
                    alias="same",
                    root_path=repo_a,
                    authority=RepositoryAuthority(
                        mode=AuthorityMode.READ_WRITE.value
                    ),
                ),
                ForestRootSpec(
                    alias="same",
                    root_path=repo_b,
                    authority=RepositoryAuthority(
                        mode=AuthorityMode.READ_ONLY.value
                    ),
                ),
            ),
            sole_write_alias="same",
        )
    assert excinfo.value.reason_code == "duplicate_alias"


def test_sibling_roots_never_share_git_authority(tmp_path: Path) -> None:
    """Two independent checkouts remain separate Git authority domains."""

    swiss = _init_repo(tmp_path / "swissknife")
    accel = _init_repo(tmp_path / "accelerator")
    forest = build_repository_forest(_two_root_policy(swiss, accel))
    swiss_desc = forest.descriptor_for_alias(DEFAULT_SWISSKNIFE_ALIAS)
    accel_desc = forest.descriptor_for_alias(DEFAULT_ACCELERATOR_ALIAS)
    assert (
        swiss_desc.local_locator.local_repository_binding_id
        != accel_desc.local_locator.local_repository_binding_id
    )
    assert swiss_desc.repository_id != accel_desc.repository_id
    assert swiss_desc.authority.mode == AuthorityMode.READ_ONLY.value
    assert accel_desc.authority.mode == AuthorityMode.READ_WRITE.value
    assert forest.write_descriptor().alias == DEFAULT_ACCELERATOR_ALIAS

    # Binding the same Git checkout under two aliases is rejected.
    with pytest.raises(RepositoryForestError) as excinfo:
        build_repository_forest(
            ForestPolicy(
                roots=(
                    ForestRootSpec(
                        alias=DEFAULT_SWISSKNIFE_ALIAS,
                        root_path=accel,
                        authority=RepositoryAuthority(
                            mode=AuthorityMode.READ_ONLY.value
                        ),
                    ),
                    ForestRootSpec(
                        alias=DEFAULT_ACCELERATOR_ALIAS,
                        root_path=accel,
                        authority=RepositoryAuthority(
                            mode=AuthorityMode.READ_WRITE.value
                        ),
                    ),
                ),
                sole_write_alias=DEFAULT_ACCELERATOR_ALIAS,
            )
        )
    assert excinfo.value.reason_code == "shared_git_authority_rejected"


def test_nested_path_does_not_inherit_parent_git_authority(
    tmp_path: Path,
) -> None:
    parent = _init_repo(tmp_path / "parent")
    nested = parent / "src"
    nested.mkdir()
    (nested / "module.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(RepositoryForestError) as excinfo:
        build_repository_descriptor(
            nested,
            alias="ipfs_accelerate_py",
            authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
        )
    assert excinfo.value.reason_code == "nested_path_not_repository_root"


def test_portable_replay_preserves_forest_identity(tmp_path: Path) -> None:
    swiss = _init_repo(tmp_path / "swissknife")
    accel = _init_repo(tmp_path / "accelerator")
    forest = build_repository_forest(_two_root_policy(swiss, accel))
    portable = forest.to_portable_dict()
    # Host paths must not appear in portable projection.
    encoded = json.dumps(portable, sort_keys=True)
    assert str(swiss) not in encoded
    assert str(accel) not in encoded
    assert "root_path" not in encoded
    assert "resolved_root_path" not in encoded
    assert "local_locator" not in encoded

    replayed = RepositoryForest.from_portable_dict(portable)
    assert replayed.forest_id == forest.forest_id
    assert forests_share_portable_identity(forest, replayed)
    assert replayed.to_portable_dict()["forest_id"] == portable["forest_id"]

    # Relocated checkouts with identical trees/policies share portable identity.
    swiss2 = tmp_path / "relocated" / "swissknife"
    accel2 = tmp_path / "relocated" / "accelerator"
    shutil.copytree(swiss, swiss2)
    shutil.copytree(accel, accel2)
    relocated = build_repository_forest(_two_root_policy(swiss2, accel2))
    assert relocated.forest_id == forest.forest_id
    assert (
        relocated.descriptor_for_alias(DEFAULT_ACCELERATOR_ALIAS)
        .local_locator.resolved_root_path
        != forest.descriptor_for_alias(DEFAULT_ACCELERATOR_ALIAS)
        .local_locator.resolved_root_path
    )


def test_authority_and_policy_affect_identity(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    read_only = build_repository_descriptor(
        repo,
        alias="swissknife",
        authority=RepositoryAuthority(mode=AuthorityMode.READ_ONLY.value),
    )
    read_write = build_repository_descriptor(
        repo,
        alias="swissknife",
        authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
    )
    ignored = build_repository_descriptor(
        repo,
        alias="swissknife",
        authority=RepositoryAuthority(mode=AuthorityMode.READ_ONLY.value),
        ignore_policy=IgnorePolicy(
            allow_dirty_overlay=False,
            exclude_patterns=("dist/**",),
        ),
    )
    casefold = build_repository_descriptor(
        repo,
        alias="swissknife",
        authority=RepositoryAuthority(mode=AuthorityMode.READ_ONLY.value),
        case_unicode_policy=CaseUnicodePolicy(case_sensitive=False),
    )
    assert read_only.descriptor_cid != read_write.descriptor_cid
    assert read_only.descriptor_cid != ignored.descriptor_cid
    assert read_only.descriptor_cid != casefold.descriptor_cid


def test_forest_identity_changes_with_tree_or_dirty_state(
    tmp_path: Path,
) -> None:
    swiss = _init_repo(tmp_path / "swissknife")
    accel = _init_repo(tmp_path / "accelerator")
    baseline = build_repository_forest(_two_root_policy(swiss, accel))
    (accel / "README.md").write_text("# accelerator changed\n", encoding="utf-8")
    _git(accel, "add", "README.md")
    _git(accel, "commit", "-m", "advance accelerator")
    advanced = build_repository_forest(_two_root_policy(swiss, accel))
    assert advanced.forest_id != baseline.forest_id
    (accel / "scratch.txt").write_text("dirty\n", encoding="utf-8")
    dirty = build_repository_forest(_two_root_policy(swiss, accel))
    assert dirty.forest_id != advanced.forest_id


def test_repository_id_independent_of_local_path(tmp_path: Path) -> None:
    left = make_repository_id(
        logical_name="swissknife",
        remote_url="https://user:secret@example.com/swissknife.git",
    )
    right = make_repository_id(
        logical_name="swissknife",
        remote_url="https://example.com/swissknife.git",
    )
    assert left == right
    assert left.startswith("repository:")
    other = make_repository_id(logical_name="ipfs_accelerate_py")
    assert other != left


def test_only_one_write_root_allowed(tmp_path: Path) -> None:
    a = _init_repo(tmp_path / "a")
    b = _init_repo(tmp_path / "b")
    with pytest.raises(RepositoryForestError) as excinfo:
        ForestPolicy(
            roots=(
                ForestRootSpec(
                    alias="a",
                    root_path=a,
                    authority=RepositoryAuthority(
                        mode=AuthorityMode.READ_WRITE.value
                    ),
                ),
                ForestRootSpec(
                    alias="b",
                    root_path=b,
                    authority=RepositoryAuthority(
                        mode=AuthorityMode.READ_WRITE.value
                    ),
                ),
            ),
            sole_write_alias="a",
        )
    assert excinfo.value.reason_code == "unexpected_write_root"


def test_broken_symlink_root_fails_closed(tmp_path: Path) -> None:
    missing_target = tmp_path / "missing-target"
    link = tmp_path / "broken-link"
    link.symlink_to(missing_target)
    with pytest.raises(RepositoryForestError) as excinfo:
        resolve_repository_root(link)
    assert excinfo.value.reason_code in {
        "missing_root",
        "root_unresolvable",
    }


def test_optional_missing_root_can_be_skipped(tmp_path: Path) -> None:
    swiss = _init_repo(tmp_path / "swiss")
    accel = _init_repo(tmp_path / "accel")
    policy = ForestPolicy(
        roots=(
            ForestRootSpec(
                alias=DEFAULT_SWISSKNIFE_ALIAS,
                root_path=swiss,
                authority=RepositoryAuthority(mode=AuthorityMode.READ_ONLY.value),
                required=True,
            ),
            ForestRootSpec(
                alias=DEFAULT_ACCELERATOR_ALIAS,
                root_path=accel,
                authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
                required=True,
            ),
            ForestRootSpec(
                alias="ipfs_kit_py",
                root_path=tmp_path / "absent-kit",
                authority=RepositoryAuthority(mode=AuthorityMode.READ_ONLY.value),
                required=False,
            ),
        ),
        sole_write_alias=DEFAULT_ACCELERATOR_ALIAS,
    )
    forest = build_repository_forest(policy, fail_on_missing_required=True)
    assert {item.alias for item in forest.descriptors} == {
        DEFAULT_SWISSKNIFE_ALIAS,
        DEFAULT_ACCELERATOR_ALIAS,
    }
    assert any(
        code.startswith("ipfs_kit_py:") for code in forest.reason_codes
    )


def test_canonical_records_round_trip(tmp_path: Path) -> None:
    swiss = _init_repo(tmp_path / "swiss")
    accel = _init_repo(tmp_path / "accel")
    forest = build_repository_forest(_two_root_policy(swiss, accel))
    full = forest.to_dict()
    # Ensure every required canonical surface is present.
    for descriptor in full["descriptors"]:
        assert descriptor["repository_id"]
        assert descriptor["commit"]
        assert descriptor["tree"]
        assert "gitlinks" in descriptor
        assert descriptor["local_locator"]["resolved_root_path"]
        assert descriptor["dirty_overlay_digest"]
        assert descriptor["ignore_policy"]
        assert descriptor["case_unicode_policy"]
        assert descriptor["authority"]
    assert full["forest_id"]
    portable = forest.to_portable_dict()
    # Round-trip portable projection through JSON.
    restored = RepositoryForest.from_portable_dict(
        json.loads(json.dumps(portable))
    )
    assert restored.forest_id == forest.forest_id


def test_dirty_overlay_forbidden_still_marks_dirty(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo")
    (repo / "dirty.txt").write_text("x\n", encoding="utf-8")
    descriptor = build_repository_descriptor(
        repo,
        alias="ipfs_accelerate_py",
        authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
        ignore_policy=IgnorePolicy(allow_dirty_overlay=False),
    )
    assert descriptor.dirty is True
    assert descriptor.dirty_overlay_digest == empty_dirty_overlay_digest()
    assert "dirty_overlay_forbidden" in descriptor.reason_codes
