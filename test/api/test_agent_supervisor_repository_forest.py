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
    DEFAULT_DATASETS_ALIAS,
    DEFAULT_KIT_ALIAS,
    DEFAULT_SWISSKNIFE_ALIAS,
    DEFAULT_SWISSKNIFE_ROOT,
    ForestPolicy,
    ForestRootSpec,
    INITIAL_FOUR_REPOSITORY_ALIASES,
    IgnorePolicy,
    OBJECTIVE_DOMAIN_EVIDENCE_TERMS,
    OBJECTIVE_GOAL_ID,
    OBJECTIVE_PARENT_GOAL_ID,
    OBJECTIVE_TASK_ID,
    REPOSITORY_DESCRIPTOR_EVIDENCE,
    REPOSITORY_DESCRIPTOR_GOAL_ID,
    REPOSITORY_DESCRIPTOR_TASK_ID,
    REPOSITORY_FOREST_MANIFEST_EVIDENCE,
    REPOSITORY_FOREST_MANIFEST_GOAL_ID,
    REPOSITORY_FOREST_MANIFEST_TASK_ID,
    REPOSITORY_FOREST_REPLAY_CLAIM_SCHEMA,
    REPOSITORY_FOREST_REPLAY_EVIDENCE,
    REPOSITORY_FOREST_REPLAY_INVARIANTS,
    REPOSITORY_IDENTITY_GOAL_PACKET_ID,
    REPOSITORY_IDENTITY_PACKET_EVIDENCE_TERMS,
    REPOSITORY_IDENTITY_PACKET_GOAL_IDS,
    REPOSITORY_IDENTITY_PACKET_TASK_ID,
    RepositoryAuthority,
    RepositoryForest,
    RepositoryForestError,
    all_covered_evidence_terms,
    build_initial_four_repository_forest,
    build_repository_descriptor,
    build_repository_forest,
    covered_evidence_terms,
    descriptor_satisfies_repository_descriptor,
    empty_dirty_overlay_digest,
    forest_satisfies_repository_forest_manifest,
    forest_satisfies_repository_forest_replay,
    forests_share_portable_identity,
    freeze_repository_forest,
    initial_four_repository_forest_policy,
    initial_vfs_assurance_forest_policy,
    make_repository_id,
    path_within_repository,
    portable_projection_excludes_host_state,
    prove_repository_descriptor,
    prove_repository_forest_manifest,
    prove_repository_forest_replay,
    prove_repository_identity_packet,
    replay_repository_forest,
    repository_descriptor_evidence_terms,
    repository_forest_manifest_evidence_terms,
    repository_forest_replay_evidence_terms,
    repository_identity_completion_goal_bindings,
    repository_identity_packet_evidence_terms,
    resolve_repository_root,
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


# ---------------------------------------------------------------------------
# VFS-G140 / VFS-G011 evidence: vfs/repository-forest-replay@1
# ---------------------------------------------------------------------------


def _four_root_paths(tmp_path: Path) -> dict[str, Path]:
    """Hermetic four independent Git roots for freeze/replay fixtures."""

    return {
        "swissknife": _init_repo(tmp_path / "swissknife", name="swissknife"),
        "accelerator": _init_repo(tmp_path / "accelerator", name="accelerator"),
        "kit": _init_repo(tmp_path / "kit", name="kit"),
        "datasets": _init_repo(tmp_path / "datasets", name="datasets"),
    }


def test_repository_forest_replay_evidence_terms_are_bound() -> None:
    """Prove vfs/repository-forest-replay@1 discovery anchors and goal bindings."""

    assert REPOSITORY_FOREST_REPLAY_EVIDENCE == "vfs/repository-forest-replay@1"
    assert OBJECTIVE_DOMAIN_EVIDENCE_TERMS == ("vfs/repository-forest-replay@1",)
    assert repository_forest_replay_evidence_terms() == OBJECTIVE_DOMAIN_EVIDENCE_TERMS
    assert covered_evidence_terms() == repository_forest_replay_evidence_terms()
    assert all_covered_evidence_terms() == covered_evidence_terms()
    assert OBJECTIVE_GOAL_ID == "VFS-G140"
    assert OBJECTIVE_PARENT_GOAL_ID == "VFS-G011"
    assert OBJECTIVE_TASK_ID == "VFS-070"
    assert set(INITIAL_FOUR_REPOSITORY_ALIASES) == {
        DEFAULT_ACCELERATOR_ALIAS,
        DEFAULT_DATASETS_ALIAS,
        DEFAULT_KIT_ALIAS,
        DEFAULT_SWISSKNIFE_ALIAS,
    }
    assert "identical trees and policy reproduce the same portable forest CID" in (
        REPOSITORY_FOREST_REPLAY_INVARIANTS
    )
    assert "unavailable required roots fail closed with a typed reason" in (
        REPOSITORY_FOREST_REPLAY_INVARIANTS
    )
    assert REPOSITORY_FOREST_REPLAY_CLAIM_SCHEMA.endswith(
        "repository-forest-replay-claim@1"
    )


def test_four_repository_freeze_and_replay_preserves_forest_cid(
    tmp_path: Path,
) -> None:
    """Identical trees and policy reproduce the same portable forest CID."""

    roots = _four_root_paths(tmp_path)
    forest = build_initial_four_repository_forest(
        accelerator_root=roots["accelerator"],
        swissknife_root=roots["swissknife"],
        kit_root=roots["kit"],
        datasets_root=roots["datasets"],
        require_all_four=True,
    )
    assert {item.alias for item in forest.descriptors} == set(
        INITIAL_FOUR_REPOSITORY_ALIASES
    )
    assert forest.sole_write_alias == DEFAULT_ACCELERATOR_ALIAS
    assert forest.write_descriptor().authority.mode == AuthorityMode.READ_WRITE.value

    portable = freeze_repository_forest(forest)
    encoded = json.dumps(portable, sort_keys=True)
    for path in roots.values():
        assert str(path) not in encoded
    assert portable_projection_excludes_host_state(portable)
    assert "root_path" not in encoded
    assert "local_locator" not in encoded

    replayed = replay_repository_forest(portable)
    assert replayed.forest_id == forest.forest_id
    assert forests_share_portable_identity(forest, replayed)
    assert forest_satisfies_repository_forest_replay(
        forest,
        require_four_aliases=True,
    )

    # Relocated checkouts with identical trees/policies share portable identity.
    relocated_base = tmp_path / "relocated"
    relocated: dict[str, Path] = {}
    for alias, src in roots.items():
        dest = relocated_base / alias
        shutil.copytree(src, dest)
        relocated[alias] = dest
    twin = build_initial_four_repository_forest(
        accelerator_root=relocated["accelerator"],
        swissknife_root=relocated["swissknife"],
        kit_root=relocated["kit"],
        datasets_root=relocated["datasets"],
        require_all_four=True,
    )
    assert twin.forest_id == forest.forest_id
    assert forest_satisfies_repository_forest_replay(forest, twin=twin, require_four_aliases=True)

    claim = prove_repository_forest_replay(
        forest,
        twin=twin,
        require_four_aliases=True,
    )
    assert claim["schema"] == REPOSITORY_FOREST_REPLAY_CLAIM_SCHEMA
    assert claim["evidence"] == "vfs/repository-forest-replay@1"
    assert claim["evidence_terms"] == ["vfs/repository-forest-replay@1"]
    assert claim["goal_id"] == "VFS-G140"
    assert claim["parent_goal_id"] == "VFS-G011"
    assert claim["task_id"] == "VFS-070"
    assert claim["satisfied"] is True
    assert claim["forest_id"] == forest.forest_id
    assert claim["replayed_forest_id"] == forest.forest_id
    assert claim["portable_host_state_excluded"] is True
    assert claim["identical_trees_and_policy_share_cid"] is True
    assert set(claim["aliases"]) == set(INITIAL_FOUR_REPOSITORY_ALIASES)
    for invariant in REPOSITORY_FOREST_REPLAY_INVARIANTS:
        assert invariant in claim["invariants"]
    # Goal labels must not enter portable forest identity.
    assert "VFS-G140" not in forest.forest_id
    assert "vfs/repository-forest-replay@1" not in forest.forest_id


def test_changed_commit_tree_overlay_or_policy_changes_forest_cid(
    tmp_path: Path,
) -> None:
    """A changed commit, tree, gitlink, overlay, or policy changes forest CID."""

    roots = _four_root_paths(tmp_path)
    baseline = build_initial_four_repository_forest(
        accelerator_root=roots["accelerator"],
        swissknife_root=roots["swissknife"],
        kit_root=roots["kit"],
        datasets_root=roots["datasets"],
    )
    baseline_id = baseline.forest_id

    # Commit / tree change on the sole-write root.
    (roots["accelerator"] / "README.md").write_text(
        "# accelerator advanced\n",
        encoding="utf-8",
    )
    _git(roots["accelerator"], "add", "README.md")
    _git(roots["accelerator"], "commit", "-m", "advance accelerator")
    after_commit = build_initial_four_repository_forest(
        accelerator_root=roots["accelerator"],
        swissknife_root=roots["swissknife"],
        kit_root=roots["kit"],
        datasets_root=roots["datasets"],
    )
    assert after_commit.forest_id != baseline_id
    after_commit_id = after_commit.forest_id

    # Dirty overlay change.
    (roots["accelerator"] / "scratch.txt").write_text("dirty\n", encoding="utf-8")
    after_overlay = build_initial_four_repository_forest(
        accelerator_root=roots["accelerator"],
        swissknife_root=roots["swissknife"],
        kit_root=roots["kit"],
        datasets_root=roots["datasets"],
    )
    assert after_overlay.forest_id != after_commit_id
    (roots["accelerator"] / "scratch.txt").unlink()

    # Policy change (analyzer profile) alters forest identity via policy_cid.
    profile_a = initial_four_repository_forest_policy(
        accelerator_root=roots["accelerator"],
        swissknife_root=roots["swissknife"],
        kit_root=roots["kit"],
        datasets_root=roots["datasets"],
        analyzer_profile={"profile_name": "policy-a"},
    )
    profile_b = initial_four_repository_forest_policy(
        accelerator_root=roots["accelerator"],
        swissknife_root=roots["swissknife"],
        kit_root=roots["kit"],
        datasets_root=roots["datasets"],
        analyzer_profile={"profile_name": "policy-b"},
    )
    forest_a = build_repository_forest(profile_a)
    forest_b = build_repository_forest(profile_b)
    assert forest_a.policy_cid != forest_b.policy_cid
    assert forest_a.forest_id != forest_b.forest_id

    # Gitlink change on one root changes that descriptor and forest identity.
    child = _init_repo(tmp_path / "child-sub", name="child-sub")
    before_gitlink = build_initial_four_repository_forest(
        accelerator_root=roots["accelerator"],
        swissknife_root=roots["swissknife"],
        kit_root=roots["kit"],
        datasets_root=roots["datasets"],
    )
    _add_submodule(roots["kit"], child, "nested-child")
    after_gitlink = build_initial_four_repository_forest(
        accelerator_root=roots["accelerator"],
        swissknife_root=roots["swissknife"],
        kit_root=roots["kit"],
        datasets_root=roots["datasets"],
    )
    assert after_gitlink.forest_id != before_gitlink.forest_id
    kit_desc = after_gitlink.descriptor_for_alias(DEFAULT_KIT_ALIAS)
    assert len(kit_desc.portable_closure.gitlinks) >= 1


def test_unavailable_required_four_repository_root_fails_closed(
    tmp_path: Path,
) -> None:
    """Unavailable required roots fail closed with a typed reason."""

    roots = _four_root_paths(tmp_path)
    missing_kit = tmp_path / "absent-kit"
    with pytest.raises(RepositoryForestError) as excinfo:
        build_initial_four_repository_forest(
            accelerator_root=roots["accelerator"],
            swissknife_root=roots["swissknife"],
            kit_root=missing_kit,
            datasets_root=roots["datasets"],
            require_all_four=True,
            fail_on_missing_required=True,
        )
    assert excinfo.value.reason_code == "missing_root"
    assert DEFAULT_KIT_ALIAS in str(excinfo.value)

    # Optional kit may be skipped when not required.
    optional = build_initial_four_repository_forest(
        accelerator_root=roots["accelerator"],
        swissknife_root=roots["swissknife"],
        kit_root=missing_kit,
        datasets_root=roots["datasets"],
        require_all_four=False,
        fail_on_missing_required=True,
    )
    assert DEFAULT_KIT_ALIAS not in {item.alias for item in optional.descriptors}
    assert any(code.startswith(f"{DEFAULT_KIT_ALIAS}:") for code in optional.reason_codes)


def test_replay_forest_id_mismatch_fails_closed(tmp_path: Path) -> None:
    """Tampered portable forest_id is rejected on replay."""

    roots = _four_root_paths(tmp_path)
    forest = build_initial_four_repository_forest(
        accelerator_root=roots["accelerator"],
        swissknife_root=roots["swissknife"],
        kit_root=roots["kit"],
        datasets_root=roots["datasets"],
    )
    portable = freeze_repository_forest(forest)
    portable["forest_id"] = "baguqeera" + ("0" * 50)
    with pytest.raises(RepositoryForestError) as excinfo:
        replay_repository_forest(portable)
    assert excinfo.value.reason_code == "forest_id_mismatch"


def test_freeze_replay_round_trip_via_json_file(tmp_path: Path) -> None:
    """Portable freeze can be serialized to disk and replayed from path."""

    roots = _four_root_paths(tmp_path)
    forest = build_initial_four_repository_forest(
        accelerator_root=roots["accelerator"],
        swissknife_root=roots["swissknife"],
        kit_root=roots["kit"],
        datasets_root=roots["datasets"],
    )
    portable = freeze_repository_forest(forest)
    path = tmp_path / "forest-portable.json"
    path.write_text(json.dumps(portable, sort_keys=True), encoding="utf-8")
    replayed = replay_repository_forest(path)
    assert replayed.forest_id == forest.forest_id


# ---------------------------------------------------------------------------
# VFS-G136 / VFS-G137 repository-identity packet evidence
# ---------------------------------------------------------------------------


def test_repository_identity_packet_evidence_terms_match_objective_heap() -> None:
    """Bind both packet evidence terms to their supervisor goal/task lineage."""

    assert REPOSITORY_DESCRIPTOR_EVIDENCE == "vfs/repository-descriptor@1"
    assert (
        REPOSITORY_FOREST_MANIFEST_EVIDENCE
        == "vfs/repository-forest-manifest@1"
    )
    assert REPOSITORY_IDENTITY_PACKET_EVIDENCE_TERMS == (
        "vfs/repository-descriptor@1",
        "vfs/repository-forest-manifest@1",
    )
    assert repository_descriptor_evidence_terms() == (
        "vfs/repository-descriptor@1",
    )
    assert repository_forest_manifest_evidence_terms() == (
        "vfs/repository-forest-manifest@1",
    )
    assert (
        repository_identity_packet_evidence_terms()
        == REPOSITORY_IDENTITY_PACKET_EVIDENCE_TERMS
    )
    assert REPOSITORY_DESCRIPTOR_GOAL_ID == "VFS-G136"
    assert REPOSITORY_FOREST_MANIFEST_GOAL_ID == "VFS-G137"
    assert REPOSITORY_IDENTITY_PACKET_GOAL_IDS == ("VFS-G136", "VFS-G137")
    assert REPOSITORY_IDENTITY_PACKET_TASK_ID == "VFS-066"
    assert REPOSITORY_DESCRIPTOR_TASK_ID == "VFS-067"
    assert REPOSITORY_FOREST_MANIFEST_TASK_ID == "VFS-068"
    assert REPOSITORY_IDENTITY_GOAL_PACKET_ID == (
        "goal_packet/repository_identity/ipfs_accelerate_py/786b6c4ff552"
    )
    assert repository_identity_completion_goal_bindings() == {
        "VFS-G136": ["vfs/repository-descriptor@1"],
        "VFS-G137": ["vfs/repository-forest-manifest@1"],
    }


def test_descriptor_evidence_claim_binds_every_identity_component(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo")
    descriptor = build_repository_descriptor(
        repo,
        alias=DEFAULT_ACCELERATOR_ALIAS,
        authority=RepositoryAuthority(mode=AuthorityMode.READ_WRITE.value),
    )

    assert descriptor_satisfies_repository_descriptor(descriptor) is True
    claim = prove_repository_descriptor(descriptor)
    assert claim["evidence"] == "vfs/repository-descriptor@1"
    assert claim["evidence_terms"] == ["vfs/repository-descriptor@1"]
    assert claim["packet_evidence_terms"] == [
        "vfs/repository-descriptor@1",
        "vfs/repository-forest-manifest@1",
    ]
    assert claim["goal_id"] == "VFS-G136"
    assert claim["parent_goal_id"] == "VFS-G010"
    assert claim["task_id"] == "VFS-067"
    assert claim["packet_task_id"] == "VFS-066"
    assert claim["descriptor_cid"] == descriptor.descriptor_cid
    assert claim["repository_id"] == descriptor.repository_id
    assert claim["identity_components"] == {
        "commit": descriptor.commit,
        "tree": descriptor.tree,
        "gitlink_closure_cid": (
            descriptor.portable_closure.gitlink_closure_cid
        ),
        "gitlink_closure_complete": True,
        "dirty": False,
        "dirty_overlay_digest": descriptor.dirty_overlay_digest,
        "ignore_policy_cid": descriptor.ignore_policy.policy_cid,
        "case_unicode_policy_cid": descriptor.case_unicode_policy.policy_cid,
        "authority_cid": descriptor.authority.authority_cid,
    }
    assert claim["satisfied"] is True
    assert claim["authoritative"] is False
    assert claim["completion_authoritative"] is False

    # Evidence labels and local checkout paths do not alter portable identity.
    encoded = json.dumps(claim, sort_keys=True)
    assert str(repo) not in encoded
    assert "local_locator" not in encoded
    assert "evidence" not in descriptor.to_portable_dict()


def test_four_repository_forest_emits_complete_packet_claim(
    tmp_path: Path,
) -> None:
    swiss = _init_repo(tmp_path / DEFAULT_SWISSKNIFE_ALIAS)
    accelerator = _init_repo(tmp_path / DEFAULT_ACCELERATOR_ALIAS)
    kit = _init_repo(tmp_path / DEFAULT_KIT_ALIAS)
    datasets = _init_repo(tmp_path / DEFAULT_DATASETS_ALIAS)
    forest = build_repository_forest(
        initial_vfs_assurance_forest_policy(
            accelerator_root=accelerator,
            swissknife_root=swiss,
            kit_root=kit,
            datasets_root=datasets,
        )
    )

    assert forest_satisfies_repository_forest_manifest(forest) is True
    manifest_claim = prove_repository_forest_manifest(forest)
    assert manifest_claim["evidence"] == "vfs/repository-forest-manifest@1"
    assert manifest_claim["evidence_terms"] == [
        "vfs/repository-forest-manifest@1"
    ]
    assert manifest_claim["goal_id"] == "VFS-G137"
    assert manifest_claim["task_id"] == "VFS-068"
    assert manifest_claim["forest_id"] == forest.forest_id
    assert manifest_claim["portable_manifest"] == forest.to_portable_dict()
    assert manifest_claim["reason_codes"] == []
    assert manifest_claim["satisfied"] is True

    packet = prove_repository_identity_packet(forest)
    assert packet["evidence_terms"] == [
        "vfs/repository-descriptor@1",
        "vfs/repository-forest-manifest@1",
    ]
    assert packet["goal_packet"] == REPOSITORY_IDENTITY_GOAL_PACKET_ID
    assert packet["packet_goal_ids"] == ["VFS-G136", "VFS-G137"]
    assert packet["packet_task_id"] == "VFS-066"
    assert packet["task_ids"] == ["VFS-067", "VFS-068"]
    assert packet["completion_goal_bindings"] == {
        "VFS-G136": ["vfs/repository-descriptor@1"],
        "VFS-G137": ["vfs/repository-forest-manifest@1"],
    }
    descriptor_claims = packet["claims"]["vfs/repository-descriptor@1"]
    assert len(descriptor_claims) == 4
    assert all(item["satisfied"] for item in descriptor_claims)
    assert (
        packet["claims"]["vfs/repository-forest-manifest@1"]
        == manifest_claim
    )
    assert packet["satisfied"] is True
    assert packet["completion_authoritative"] is False

    encoded = json.dumps(packet, sort_keys=True)
    for root in (swiss, accelerator, kit, datasets):
        assert str(root) not in encoded
    assert "local_locator" not in encoded

    # Portable replay preserves both forest identity and evidence satisfaction.
    replayed_packet = prove_repository_identity_packet(
        forest.to_portable_dict()
    )
    assert replayed_packet["forest_id"] == packet["forest_id"]
    assert replayed_packet["satisfied"] is True


def test_incomplete_initial_forest_cannot_satisfy_manifest_evidence(
    tmp_path: Path,
) -> None:
    swiss = _init_repo(tmp_path / "swissknife")
    accelerator = _init_repo(tmp_path / "accelerator")
    forest = build_repository_forest(_two_root_policy(swiss, accelerator))

    assert forest_satisfies_repository_forest_manifest(forest) is False
    claim = prove_repository_forest_manifest(forest)
    assert claim["satisfied"] is False
    assert claim["reason_codes"] == [
        "missing_repository:ipfs_datasets_py",
        "missing_repository:ipfs_kit_py",
    ]
    assert prove_repository_identity_packet(forest)["satisfied"] is False
