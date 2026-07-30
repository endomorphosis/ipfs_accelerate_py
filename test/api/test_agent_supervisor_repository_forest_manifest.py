"""Tests for the frozen four-repository forest manifest loader and replay validator."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.repository_forest import (
    AnalyzerProfile,
    AuthorityMode,
    DEFAULT_ACCELERATOR_ALIAS,
    DEFAULT_DATASETS_ALIAS,
    DEFAULT_KIT_ALIAS,
    DEFAULT_SWISSKNIFE_ALIAS,
    IgnorePolicy,
    RepositoryForest,
    forests_share_portable_identity,
)
from ipfs_accelerate_py.agent_supervisor.repository_forest_manifest import (
    FROZEN_FOUR_ROOT_ALIASES,
    LOCAL_PROJECTION_FILENAME,
    PORTABLE_PROJECTION_FILENAME,
    ReviewedForestManifest,
    ReviewedManifestRoot,
    RepositoryForestManifestError,
    default_reviewed_four_repository_manifest,
    load_local_projection,
    load_portable_projection,
    load_reviewed_manifest,
    materialize_forest_from_manifest,
    materialize_initial_four_repository_forest,
    persist_manifest_projections,
    validate_manifest_replay,
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


def _four_repos(tmp_path: Path) -> dict[str, Path]:
    swiss = _init_repo(tmp_path / "swissknife", name="swissknife")
    accel = _init_repo(tmp_path / "accelerator", name="accelerator")
    kit = _init_repo(tmp_path / "kit", name="kit")
    datasets = _init_repo(tmp_path / "datasets", name="datasets")
    return {
        DEFAULT_SWISSKNIFE_ALIAS: swiss,
        DEFAULT_ACCELERATOR_ALIAS: accel,
        DEFAULT_KIT_ALIAS: kit,
        DEFAULT_DATASETS_ALIAS: datasets,
    }


def test_default_reviewed_manifest_covers_four_roots(tmp_path: Path) -> None:
    roots = _four_repos(tmp_path)
    manifest = default_reviewed_four_repository_manifest(
        swissknife_root=roots[DEFAULT_SWISSKNIFE_ALIAS],
        accelerator_root=roots[DEFAULT_ACCELERATOR_ALIAS],
        kit_root=roots[DEFAULT_KIT_ALIAS],
        datasets_root=roots[DEFAULT_DATASETS_ALIAS],
    )
    assert set(manifest.expected_aliases()) == set(FROZEN_FOUR_ROOT_ALIASES)
    assert manifest.sole_write_alias == DEFAULT_ACCELERATOR_ALIAS
    by_alias = {item.alias: item for item in manifest.roots}
    assert by_alias[DEFAULT_SWISSKNIFE_ALIAS].authority_mode == (
        AuthorityMode.READ_ONLY.value
    )
    assert by_alias[DEFAULT_ACCELERATOR_ALIAS].authority_mode == (
        AuthorityMode.READ_WRITE.value
    )
    assert by_alias[DEFAULT_KIT_ALIAS].authority_mode == AuthorityMode.READ_ONLY.value
    assert by_alias[DEFAULT_DATASETS_ALIAS].authority_mode == (
        AuthorityMode.READ_ONLY.value
    )
    # Reviewed commits are present as observations only.
    assert by_alias[DEFAULT_SWISSKNIFE_ALIAS].reviewed_commit
    portable = json.dumps(manifest.to_portable_dict(), sort_keys=True)
    assert str(roots[DEFAULT_SWISSKNIFE_ALIAS]) not in portable
    assert "local_root_paths" not in portable


def test_materialize_derives_fresh_descriptors_not_reviewed_commits(
    tmp_path: Path,
) -> None:
    roots = _four_repos(tmp_path)
    manifest = default_reviewed_four_repository_manifest(
        swissknife_root=roots[DEFAULT_SWISSKNIFE_ALIAS],
        accelerator_root=roots[DEFAULT_ACCELERATOR_ALIAS],
        kit_root=roots[DEFAULT_KIT_ALIAS],
        datasets_root=roots[DEFAULT_DATASETS_ALIAS],
    )
    # Force reviewed commits to a known-wrong value.
    forced = ReviewedForestManifest(
        roots=tuple(
            ReviewedManifestRoot(
                alias=item.alias,
                authority_mode=item.authority_mode,
                required=item.required,
                reviewed_commit="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )
            for item in manifest.roots
        ),
        sole_write_alias=manifest.sole_write_alias,
        analyzer_profile=manifest.analyzer_profile,
        local_root_paths=manifest.local_root_paths,
    )
    materialization = materialize_forest_from_manifest(forced)
    forest = materialization.forest
    assert len(forest.descriptors) == 4
    for descriptor in forest.descriptors:
        # Live HEAD must win over the reviewed observation.
        assert descriptor.commit != "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        assert len(descriptor.commit) in {40, 64}
    assert set(materialization.observed_commit_mismatches) == set(
        FROZEN_FOUR_ROOT_ALIASES
    )
    assert any(
        code.endswith(":reviewed_commit_drift")
        for code in materialization.reason_codes
    )


def test_persist_portable_and_local_projections_separately(tmp_path: Path) -> None:
    roots = _four_repos(tmp_path)
    materialization = materialize_initial_four_repository_forest(
        swissknife_root=roots[DEFAULT_SWISSKNIFE_ALIAS],
        accelerator_root=roots[DEFAULT_ACCELERATOR_ALIAS],
        kit_root=roots[DEFAULT_KIT_ALIAS],
        datasets_root=roots[DEFAULT_DATASETS_ALIAS],
    )
    out = tmp_path / "projections"
    portable_path, local_path = persist_manifest_projections(materialization, out)
    assert portable_path.name == PORTABLE_PROJECTION_FILENAME
    assert local_path.name == LOCAL_PROJECTION_FILENAME
    assert portable_path.is_file()
    assert local_path.is_file()

    portable = load_portable_projection(portable_path)
    local = load_local_projection(local_path)
    portable_text = json.dumps(portable, sort_keys=True)
    for path in roots.values():
        assert str(path) not in portable_text
    assert "local_locator" not in portable_text
    assert "resolved_root_path" not in portable_text
    assert "local_roots" in local
    assert DEFAULT_ACCELERATOR_ALIAS in local["local_roots"]
    assert local["forest_id"] == portable["forest_id"]
    assert local["portable_forest_id"] == portable["forest_id"]


def test_replay_validates_expected_roots_authority_and_policy(
    tmp_path: Path,
) -> None:
    roots = _four_repos(tmp_path)
    materialization = materialize_initial_four_repository_forest(
        swissknife_root=roots[DEFAULT_SWISSKNIFE_ALIAS],
        accelerator_root=roots[DEFAULT_ACCELERATOR_ALIAS],
        kit_root=roots[DEFAULT_KIT_ALIAS],
        datasets_root=roots[DEFAULT_DATASETS_ALIAS],
    )
    portable_path, _ = persist_manifest_projections(
        materialization,
        tmp_path / "out",
    )
    result = validate_manifest_replay(
        portable_path,
        expected_manifest=materialization.manifest,
        require_aliases=FROZEN_FOUR_ROOT_ALIASES,
    )
    assert result.valid is True
    assert result.reason_codes == ()
    assert set(result.observed_aliases) == set(FROZEN_FOUR_ROOT_ALIASES)
    assert result.forest_id == materialization.forest_id


def test_replay_rejects_authority_drift(tmp_path: Path) -> None:
    roots = _four_repos(tmp_path)
    materialization = materialize_initial_four_repository_forest(
        swissknife_root=roots[DEFAULT_SWISSKNIFE_ALIAS],
        accelerator_root=roots[DEFAULT_ACCELERATOR_ALIAS],
        kit_root=roots[DEFAULT_KIT_ALIAS],
        datasets_root=roots[DEFAULT_DATASETS_ALIAS],
    )
    portable = materialization.to_portable_projection()
    # Tamper a descriptor's ignore policy away from the reviewed expectation.
    for descriptor in portable["forest"]["descriptors"]:
        if descriptor["logical_name"] == DEFAULT_KIT_ALIAS:
            descriptor["ignore_policy"] = {
                "schema": descriptor["ignore_policy"]["schema"],
                "include_gitignored": True,
                "allow_dirty_overlay": False,
                "exclude_patterns": ["build/**"],
                "include_patterns": [],
            }
            descriptor.pop("ignore_policy_cid", None)
    portable["forest"].pop("forest_id", None)
    portable.pop("forest_id", None)
    result = validate_manifest_replay(
        portable,
        expected_manifest=materialization.manifest,
    )
    assert result.valid is False
    assert any(
        code.endswith("ignore_policy_mismatch")
        or code in {"policy_cid_mismatch", "forest_id_mismatch"}
        for code in result.reason_codes
    )


def test_replay_rejects_expected_policy_mismatch(tmp_path: Path) -> None:
    roots = _four_repos(tmp_path)
    materialization = materialize_initial_four_repository_forest(
        swissknife_root=roots[DEFAULT_SWISSKNIFE_ALIAS],
        accelerator_root=roots[DEFAULT_ACCELERATOR_ALIAS],
        kit_root=roots[DEFAULT_KIT_ALIAS],
        datasets_root=roots[DEFAULT_DATASETS_ALIAS],
    )
    # Divergent expected manifest (datasets ignore policy differs from live).
    altered_roots = []
    for item in materialization.manifest.roots:
        if item.alias == DEFAULT_DATASETS_ALIAS:
            altered_roots.append(
                ReviewedManifestRoot(
                    alias=item.alias,
                    authority_mode=item.authority_mode,
                    required=item.required,
                    reviewed_commit=item.reviewed_commit,
                    ignore_policy=IgnorePolicy(include_gitignored=True),
                )
            )
        else:
            altered_roots.append(item)
    expected = ReviewedForestManifest(
        roots=tuple(altered_roots),
        sole_write_alias=materialization.manifest.sole_write_alias,
        analyzer_profile=materialization.manifest.analyzer_profile,
        local_root_paths=materialization.manifest.local_root_paths,
    )
    result = validate_manifest_replay(
        materialization.to_portable_projection(),
        expected_manifest=expected,
    )
    assert result.valid is False
    assert any(
        "ignore_policy_mismatch" in code
        or code
        in {
            "expected_manifest_mismatch",
            "policy_cid_mismatch",
        }
        for code in result.reason_codes
    )


def test_tree_change_changes_identity(tmp_path: Path) -> None:
    roots = _four_repos(tmp_path)
    before = materialize_initial_four_repository_forest(
        swissknife_root=roots[DEFAULT_SWISSKNIFE_ALIAS],
        accelerator_root=roots[DEFAULT_ACCELERATOR_ALIAS],
        kit_root=roots[DEFAULT_KIT_ALIAS],
        datasets_root=roots[DEFAULT_DATASETS_ALIAS],
    )
    accel = roots[DEFAULT_ACCELERATOR_ALIAS]
    (accel / "README.md").write_text("# advanced\n", encoding="utf-8")
    _git(accel, "add", "README.md")
    _git(accel, "commit", "-m", "advance tree")
    after = materialize_initial_four_repository_forest(
        swissknife_root=roots[DEFAULT_SWISSKNIFE_ALIAS],
        accelerator_root=roots[DEFAULT_ACCELERATOR_ALIAS],
        kit_root=roots[DEFAULT_KIT_ALIAS],
        datasets_root=roots[DEFAULT_DATASETS_ALIAS],
    )
    assert after.forest_id != before.forest_id
    assert (
        after.forest.descriptor_for_alias(DEFAULT_ACCELERATOR_ALIAS).tree
        != before.forest.descriptor_for_alias(DEFAULT_ACCELERATOR_ALIAS).tree
    )


def test_gitlink_change_changes_identity(tmp_path: Path) -> None:
    roots = _four_repos(tmp_path)
    child = _init_repo(tmp_path / "child", name="child")
    before = materialize_initial_four_repository_forest(
        swissknife_root=roots[DEFAULT_SWISSKNIFE_ALIAS],
        accelerator_root=roots[DEFAULT_ACCELERATOR_ALIAS],
        kit_root=roots[DEFAULT_KIT_ALIAS],
        datasets_root=roots[DEFAULT_DATASETS_ALIAS],
    )
    _add_submodule(roots[DEFAULT_ACCELERATOR_ALIAS], child, "extra-component")
    after = materialize_initial_four_repository_forest(
        swissknife_root=roots[DEFAULT_SWISSKNIFE_ALIAS],
        accelerator_root=roots[DEFAULT_ACCELERATOR_ALIAS],
        kit_root=roots[DEFAULT_KIT_ALIAS],
        datasets_root=roots[DEFAULT_DATASETS_ALIAS],
    )
    assert after.forest_id != before.forest_id
    accel_desc = after.forest.descriptor_for_alias(DEFAULT_ACCELERATOR_ALIAS)
    assert len(accel_desc.portable_closure.gitlinks) >= 1


def test_dirty_overlay_change_changes_identity(tmp_path: Path) -> None:
    roots = _four_repos(tmp_path)
    before = materialize_initial_four_repository_forest(
        swissknife_root=roots[DEFAULT_SWISSKNIFE_ALIAS],
        accelerator_root=roots[DEFAULT_ACCELERATOR_ALIAS],
        kit_root=roots[DEFAULT_KIT_ALIAS],
        datasets_root=roots[DEFAULT_DATASETS_ALIAS],
    )
    (roots[DEFAULT_ACCELERATOR_ALIAS] / "scratch.txt").write_text(
        "dirty\n",
        encoding="utf-8",
    )
    after = materialize_initial_four_repository_forest(
        swissknife_root=roots[DEFAULT_SWISSKNIFE_ALIAS],
        accelerator_root=roots[DEFAULT_ACCELERATOR_ALIAS],
        kit_root=roots[DEFAULT_KIT_ALIAS],
        datasets_root=roots[DEFAULT_DATASETS_ALIAS],
    )
    assert after.forest_id != before.forest_id
    dirty = after.forest.descriptor_for_alias(DEFAULT_ACCELERATOR_ALIAS)
    assert dirty.dirty is True


def test_policy_change_changes_identity(tmp_path: Path) -> None:
    roots = _four_repos(tmp_path)
    base_manifest = default_reviewed_four_repository_manifest(
        swissknife_root=roots[DEFAULT_SWISSKNIFE_ALIAS],
        accelerator_root=roots[DEFAULT_ACCELERATOR_ALIAS],
        kit_root=roots[DEFAULT_KIT_ALIAS],
        datasets_root=roots[DEFAULT_DATASETS_ALIAS],
    )
    base = materialize_forest_from_manifest(base_manifest)
    altered_roots = []
    for item in base_manifest.roots:
        if item.alias == DEFAULT_KIT_ALIAS:
            altered_roots.append(
                ReviewedManifestRoot(
                    alias=item.alias,
                    authority_mode=item.authority_mode,
                    required=item.required,
                    reviewed_commit=item.reviewed_commit,
                    ignore_policy=IgnorePolicy(
                        allow_dirty_overlay=False,
                        exclude_patterns=("dist/**",),
                    ),
                )
            )
        else:
            altered_roots.append(item)
    altered_manifest = ReviewedForestManifest(
        roots=tuple(altered_roots),
        sole_write_alias=base_manifest.sole_write_alias,
        analyzer_profile=base_manifest.analyzer_profile,
        local_root_paths=base_manifest.local_root_paths,
    )
    altered = materialize_forest_from_manifest(altered_manifest)
    assert altered.forest_id != base.forest_id
    assert altered.forest.policy_cid != base.forest.policy_cid
    assert altered.manifest.manifest_cid != base.manifest.manifest_cid


def test_analyzer_profile_change_changes_identity(tmp_path: Path) -> None:
    roots = _four_repos(tmp_path)
    profile_a = AnalyzerProfile(
        profile_name="profile-a",
        analyzer_versions=(("ast", "1.0.0"),),
    )
    profile_b = AnalyzerProfile(
        profile_name="profile-b",
        analyzer_versions=(("ast", "2.0.0"),),
    )
    first = materialize_initial_four_repository_forest(
        swissknife_root=roots[DEFAULT_SWISSKNIFE_ALIAS],
        accelerator_root=roots[DEFAULT_ACCELERATOR_ALIAS],
        kit_root=roots[DEFAULT_KIT_ALIAS],
        datasets_root=roots[DEFAULT_DATASETS_ALIAS],
        analyzer_profile=profile_a,
    )
    second = materialize_initial_four_repository_forest(
        swissknife_root=roots[DEFAULT_SWISSKNIFE_ALIAS],
        accelerator_root=roots[DEFAULT_ACCELERATOR_ALIAS],
        kit_root=roots[DEFAULT_KIT_ALIAS],
        datasets_root=roots[DEFAULT_DATASETS_ALIAS],
        analyzer_profile=profile_b,
    )
    assert first.forest_id != second.forest_id
    assert first.manifest.manifest_cid != second.manifest.manifest_cid
    assert (
        first.forest.analyzer_profile.profile_cid
        != second.forest.analyzer_profile.profile_cid
    )


def test_equivalent_relocation_retains_portable_identity(tmp_path: Path) -> None:
    roots = _four_repos(tmp_path)
    original = materialize_initial_four_repository_forest(
        swissknife_root=roots[DEFAULT_SWISSKNIFE_ALIAS],
        accelerator_root=roots[DEFAULT_ACCELERATOR_ALIAS],
        kit_root=roots[DEFAULT_KIT_ALIAS],
        datasets_root=roots[DEFAULT_DATASETS_ALIAS],
    )
    relocated_base = tmp_path / "relocated"
    relocated: dict[str, Path] = {}
    for alias, path in roots.items():
        dest = relocated_base / alias
        shutil.copytree(path, dest)
        relocated[alias] = dest
    moved = materialize_initial_four_repository_forest(
        swissknife_root=relocated[DEFAULT_SWISSKNIFE_ALIAS],
        accelerator_root=relocated[DEFAULT_ACCELERATOR_ALIAS],
        kit_root=relocated[DEFAULT_KIT_ALIAS],
        datasets_root=relocated[DEFAULT_DATASETS_ALIAS],
    )
    assert moved.forest_id == original.forest_id
    assert forests_share_portable_identity(original.forest, moved.forest)
    assert (
        moved.forest.descriptor_for_alias(DEFAULT_ACCELERATOR_ALIAS)
        .local_locator.resolved_root_path
        != original.forest.descriptor_for_alias(DEFAULT_ACCELERATOR_ALIAS)
        .local_locator.resolved_root_path
    )
    # Portable projections remain equal on forest identity.
    assert (
        original.to_portable_projection()["forest_id"]
        == moved.to_portable_projection()["forest_id"]
    )


def test_load_reviewed_manifest_from_json_round_trip(tmp_path: Path) -> None:
    roots = _four_repos(tmp_path)
    manifest = default_reviewed_four_repository_manifest(
        swissknife_root=roots[DEFAULT_SWISSKNIFE_ALIAS],
        accelerator_root=roots[DEFAULT_ACCELERATOR_ALIAS],
        kit_root=roots[DEFAULT_KIT_ALIAS],
        datasets_root=roots[DEFAULT_DATASETS_ALIAS],
    )
    path = tmp_path / "reviewed.json"
    path.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True, indent=2),
        encoding="utf-8",
    )
    loaded = load_reviewed_manifest(path)
    assert loaded.manifest_cid == manifest.manifest_cid
    assert loaded.expected_aliases() == manifest.expected_aliases()
    materialization = materialize_forest_from_manifest(loaded)
    assert len(materialization.forest.descriptors) == 4


def test_missing_required_root_fails_closed(tmp_path: Path) -> None:
    roots = _four_repos(tmp_path)
    manifest = default_reviewed_four_repository_manifest(
        swissknife_root=roots[DEFAULT_SWISSKNIFE_ALIAS],
        accelerator_root=roots[DEFAULT_ACCELERATOR_ALIAS],
        kit_root=tmp_path / "absent-kit",
        datasets_root=roots[DEFAULT_DATASETS_ALIAS],
        require_all_four=True,
    )
    with pytest.raises(RepositoryForestManifestError) as excinfo:
        materialize_forest_from_manifest(manifest, fail_on_missing_required=True)
    assert excinfo.value.reason_code in {
        "missing_root",
        "not_a_git_repository",
        "root_unresolvable",
    }


def test_secret_material_rejected_from_manifest(tmp_path: Path) -> None:
    roots = _four_repos(tmp_path)
    with pytest.raises(RepositoryForestManifestError) as excinfo:
        ReviewedForestManifest.from_dict(
            {
                "schema": (
                    "ipfs_accelerate_py.agent_supervisor."
                    "repository-forest-manifest@1"
                ),
                "sole_write_alias": DEFAULT_ACCELERATOR_ALIAS,
                "api_key": "should-never-appear",
                "roots": [
                    {
                        "alias": DEFAULT_SWISSKNIFE_ALIAS,
                        "authority_mode": AuthorityMode.READ_ONLY.value,
                    },
                    {
                        "alias": DEFAULT_ACCELERATOR_ALIAS,
                        "authority_mode": AuthorityMode.READ_WRITE.value,
                    },
                ],
                "local_root_paths": {
                    DEFAULT_SWISSKNIFE_ALIAS: str(
                        roots[DEFAULT_SWISSKNIFE_ALIAS]
                    ),
                    DEFAULT_ACCELERATOR_ALIAS: str(
                        roots[DEFAULT_ACCELERATOR_ALIAS]
                    ),
                },
            }
        )
    assert excinfo.value.reason_code == "secret_material_rejected"


def test_analyzer_profile_rejects_credential_versions() -> None:
    with pytest.raises(Exception) as excinfo:
        AnalyzerProfile(
            profile_name="bad",
            analyzer_versions=(("tool", "token=super-secret"),),
        )
    assert getattr(excinfo.value, "reason_code", "") == "secret_material_rejected"


def test_replay_of_persisted_projections_is_stable(tmp_path: Path) -> None:
    roots = _four_repos(tmp_path)
    materialization = materialize_initial_four_repository_forest(
        swissknife_root=roots[DEFAULT_SWISSKNIFE_ALIAS],
        accelerator_root=roots[DEFAULT_ACCELERATOR_ALIAS],
        kit_root=roots[DEFAULT_KIT_ALIAS],
        datasets_root=roots[DEFAULT_DATASETS_ALIAS],
    )
    portable_path, local_path = persist_manifest_projections(
        materialization,
        tmp_path / "store",
    )
    portable = load_portable_projection(portable_path)
    local = load_local_projection(local_path)
    replayed = RepositoryForest.from_portable_dict(portable["forest"])
    assert replayed.forest_id == materialization.forest_id
    assert forests_share_portable_identity(replayed, materialization.forest)
    # Local projection carries host paths but shares portable forest identity.
    assert local["forest_id"] == materialization.forest_id
    validation = validate_manifest_replay(portable)
    assert validation.valid is True


def test_logging_never_emits_environment_secrets(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "super-secret-value-xyz")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-should-not-log")
    roots = _four_repos(tmp_path)
    with caplog.at_level(logging.DEBUG):
        materialization = materialize_initial_four_repository_forest(
            swissknife_root=roots[DEFAULT_SWISSKNIFE_ALIAS],
            accelerator_root=roots[DEFAULT_ACCELERATOR_ALIAS],
            kit_root=roots[DEFAULT_KIT_ALIAS],
            datasets_root=roots[DEFAULT_DATASETS_ALIAS],
        )
        persist_manifest_projections(materialization, tmp_path / "logs")
    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert "super-secret-value-xyz" not in joined
    assert "sk-test-should-not-log" not in joined
    assert "AWS_SECRET_ACCESS_KEY" not in joined
    assert "OPENAI_API_KEY" not in joined


def test_forest_policy_from_manifest_matches_initial_authority(
    tmp_path: Path,
) -> None:
    roots = _four_repos(tmp_path)
    materialization = materialize_initial_four_repository_forest(
        swissknife_root=roots[DEFAULT_SWISSKNIFE_ALIAS],
        accelerator_root=roots[DEFAULT_ACCELERATOR_ALIAS],
        kit_root=roots[DEFAULT_KIT_ALIAS],
        datasets_root=roots[DEFAULT_DATASETS_ALIAS],
    )
    swiss = materialization.forest.descriptor_for_alias(DEFAULT_SWISSKNIFE_ALIAS)
    accel = materialization.forest.descriptor_for_alias(DEFAULT_ACCELERATOR_ALIAS)
    assert swiss.authority.mode == AuthorityMode.READ_ONLY.value
    assert accel.authority.mode == AuthorityMode.READ_WRITE.value
    assert materialization.forest.write_descriptor().alias == DEFAULT_ACCELERATOR_ALIAS
