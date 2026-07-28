"""Exact SCA repository snapshot and coverage disposition contracts."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.repository_snapshot import (
    COVERAGE_DISPOSITION_SCHEMA,
    CoverageDisposition,
    CoverageIncompleteError,
    CoverageKind,
    DependencyIdentityKind,
    EntryKind,
    GitStatus,
    REPOSITORY_SNAPSHOT_SCHEMA,
    SCOPE_POLICY_SCHEMA,
    RepositoryPathEscapeError,
    RepositorySnapshot,
    RepositoryStateError,
    ScopePolicyError,
    SymlinkEscapeError,
    build_repository_snapshot,
    classify_coverage_kind,
    load_scope_policy,
    repo_path,
    scope_policy_from_mapping,
    snapshot_analyzer_health_inventory,
)


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _minimal_policy(**overrides: object) -> dict[str, object]:
    policy: dict[str, object] = {
        "schema": SCOPE_POLICY_SCHEMA,
        "schemaVersion": 1,
        "scopeId": "test-sca-scope-v1",
        "primaryRepository": "repository",
        "primaryRoot": ".",
        "providerScopes": ["external/ipfs_accelerate"],
        "skipPrefixes": ["node_modules", "tmp", "playwright-report"],
        "skipDirectoryNames": [
            ".git",
            "node_modules",
            "__pycache__",
            "playwright-report",
        ],
        "dependencyDirectoryNames": ["node_modules"],
        "dependencyLockFiles": [
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
        ],
        "dependencyManifestFiles": ["package.json"],
        "workingTreeOverlay": {
            "mode": "tracked_plus_allowlisted_untracked_source",
            "allowDirtyAnalysis": True,
            "allowlistedUntrackedSuffixes": [
                ".ts",
                ".js",
                ".py",
                ".json",
                ".md",
            ],
            "allowlistedUntrackedExactNames": ["package.json"],
        },
        "dispositionRules": {
            "semanticExtensions": [".ts", ".tsx", ".js", ".jsx", ".py", ".mjs"],
            "structuredExtensions": [".json", ".yaml", ".yml"],
            "textExtensions": [".md", ".txt", ".sh", ".css"],
            "binaryExtensions": [".png", ".wasm", ".zip"],
            "generatedSuffixes": [".map", ".d.ts"],
            "generatedPathParts": ["dist", "build"],
        },
        "silentExclusionsAllowed": False,
        "trackedCoverageRequired": 1.0,
    }
    policy.update(overrides)
    return policy


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "SCA Snapshot Test")
    _git(repository, "config", "user.email", "sca-snapshot@example.invalid")
    _write(
        repository / "src" / "service.ts",
        "export function dispatch(x: number): number { return x + 1; }\n",
    )
    _write(repository / "src" / "util.py", "def add(a, b):\n    return a + b\n")
    _write(repository / "README.md", "fixture\n")
    _write(
        repository / "package.json",
        json.dumps({"name": "fixture", "version": "0.0.1"}, indent=2) + "\n",
    )
    _write(repository / "yarn.lock", "# yarn lockfile v1\n")
    _write(repository / "assets" / "logo.png", "not-a-real-png")
    _write(repository / "dist" / "bundle.js", "console.log(1)\n")
    _write(repository / "schema" / "tool.json", '{"type":"object"}\n')
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "fixture")
    return repository


def _snapshot(repository: Path, **kwargs: object) -> RepositorySnapshot:
    policy = kwargs.pop("scope_policy", _minimal_policy())
    return build_repository_snapshot(
        repository,
        scope_policy=policy,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )


def test_load_reviewed_scope_config_from_superproject() -> None:
    root = Path(__file__).resolve()
    # Walk up until the monorepo config is visible.
    config = None
    for parent in root.parents:
        candidate = parent / "config" / "swissknife_symbolic_contract_scope.json"
        if candidate.is_file():
            config = candidate
            break
    assert config is not None, "reviewed scope config must exist in the monorepo"
    policy = load_scope_policy(config)
    assert policy.scope_id == "swissknife-symbolic-contract-scope-v1"
    assert policy.primary_repository == "swissknife"
    assert policy.primary_root == "swissknife"
    assert policy.silent_exclusions_allowed is False
    assert policy.tracked_coverage_required == 1.0
    assert ".ts" in policy.semantic_extensions
    assert "node_modules" in policy.dependency_directory_names
    assert "yarn.lock" in policy.dependency_lock_files
    assert policy.policy_id.startswith("sca-scope-policy:sha256:")
    rendered = policy.to_dict()
    assert rendered["schema"] == SCOPE_POLICY_SCHEMA


def test_clean_snapshot_assigns_exactly_one_disposition_per_tracked_path(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)

    snapshot = _snapshot(repository)

    tracked_paths = set(
        _git(repository, "ls-files").splitlines()
    )
    assert snapshot.is_clean is True
    assert snapshot.stats.tracked_path_count == len(tracked_paths)
    assert len(snapshot.tracked_dispositions()) == len(tracked_paths)
    assert {item.path for item in snapshot.tracked_dispositions()} == tracked_paths
    assert all(item.git_status is GitStatus.CLEAN for item in snapshot.dispositions)
    # Exactly one disposition per path.
    assert len(snapshot.dispositions) == len({item.path for item in snapshot.dispositions})
    snapshot.assert_exhaustive_tracked_coverage()

    by_path = {item.path: item for item in snapshot.dispositions}
    assert by_path["src/service.ts"].kind is CoverageKind.SEMANTIC_AST
    assert by_path["src/util.py"].kind is CoverageKind.SEMANTIC_AST
    assert by_path["README.md"].kind is CoverageKind.TEXT_REFERENCE
    assert by_path["schema/tool.json"].kind is CoverageKind.STRUCTURED_DATA
    assert by_path["assets/logo.png"].kind is CoverageKind.BINARY_OR_GENERATED
    assert by_path["dist/bundle.js"].kind is CoverageKind.BINARY_OR_GENERATED
    assert by_path["yarn.lock"].kind is CoverageKind.DEPENDENCY_TOOL_IDENTITY
    assert by_path["package.json"].kind is CoverageKind.DEPENDENCY_TOOL_IDENTITY
    assert all(item.reason_code and item.policy_rule for item in snapshot.dispositions)
    assert all(
        item.disposition_id.startswith("sca-coverage-disposition:sha256:")
        for item in snapshot.dispositions
    )
    assert snapshot.snapshot_id.startswith("sca-repository-snapshot:sha256:")
    assert snapshot.head_tree_id == _git(repository, "rev-parse", "HEAD^{tree}")
    payload = snapshot.to_dict()
    assert payload["schema"] == REPOSITORY_SNAPSHOT_SCHEMA
    assert "export function" not in snapshot.to_json()


def test_dirty_modified_and_staged_change_snapshot_identity(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    clean = _snapshot(repository)

    service = repository / "src" / "service.ts"
    service.write_text("export const dirty = 1\n", encoding="utf-8")
    dirty = _snapshot(repository)
    assert dirty.dirty is True
    assert dirty.snapshot_id != clean.snapshot_id
    changed = dirty.disposition_for_path("src/service.ts")
    assert changed is not None
    assert changed.git_status is GitStatus.MODIFIED
    assert changed.overlay is True
    assert changed.content_digest != clean.disposition_for_path(
        "src/service.ts"
    ).content_digest  # type: ignore[union-attr]

    _git(repository, "add", "src/service.ts")
    service.write_text("export const dirty_again = 2\n", encoding="utf-8")
    staged_and_modified = _snapshot(repository)
    again = staged_and_modified.disposition_for_path("src/service.ts")
    assert again is not None
    assert again.git_status is GitStatus.STAGED_AND_MODIFIED
    assert staged_and_modified.snapshot_id != dirty.snapshot_id


def test_deleted_path_remains_in_ledger_with_deleted_status(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    (repository / "README.md").unlink()

    snapshot = _snapshot(repository)
    deleted = snapshot.disposition_for_path("README.md")
    assert deleted is not None
    assert deleted.tracked is True
    assert deleted.git_status is GitStatus.DELETED
    assert deleted.kind is CoverageKind.TEXT_REFERENCE
    assert snapshot.stats.deleted_path_count >= 1
    # Tracked ledger still accounts for the path.
    assert "README.md" in {item.path for item in snapshot.tracked_dispositions()}


def test_allowlisted_untracked_overlay_binds_and_changes_identity(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    clean = _snapshot(repository)
    _write(repository / "src" / "extra.ts", "export const extra = true\n")
    _write(repository / "noise.bin", "not-allowlisted")

    snapshot = _snapshot(repository)
    assert snapshot.snapshot_id != clean.snapshot_id
    extra = snapshot.disposition_for_path("src/extra.ts")
    assert extra is not None
    assert extra.tracked is False
    assert extra.git_status is GitStatus.UNTRACKED
    assert extra.kind is CoverageKind.SEMANTIC_AST
    assert extra.overlay is True
    # Non-allowlisted untracked must not enter authority.
    assert snapshot.disposition_for_path("noise.bin") is None
    assert snapshot.stats.untracked_path_count >= 1


def test_rename_is_explicit_with_rename_from(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    _write(repository / "old.ts", "export const value = 1\n")
    _git(repository, "add", "old.ts")
    _git(repository, "commit", "-qm", "add old")
    _git(repository, "mv", "old.ts", "new.ts")

    snapshot = _snapshot(repository)
    renamed = snapshot.disposition_for_path("new.ts")
    assert renamed is not None
    assert renamed.git_status is GitStatus.RENAMED
    assert renamed.rename_from == "old.ts"
    # Source may appear as deleted or be absent from index; ledger stays exhaustive.
    tracked_paths = {item.path for item in snapshot.tracked_dispositions()}
    assert "new.ts" in tracked_paths


def test_submodule_gitlink_is_dependency_identity_not_source(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    nested = tmp_path / "nested"
    nested.mkdir()
    _git(nested, "init", "-q")
    _git(nested, "config", "user.name", "Nested")
    _git(nested, "config", "user.email", "nested@example.invalid")
    _write(nested / "nested.txt", "nested\n")
    _git(nested, "add", ".")
    _git(nested, "commit", "-qm", "nested")
    nested_commit = _git(nested, "rev-parse", "HEAD")

    # Register as a gitlink without requiring network submodule add.
    _git(
        repository,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{nested_commit},vendor/nested",
    )
    # Materialize the gitlink worktree entry so the index is consistent.
    vendor = repository / "vendor" / "nested"
    vendor.mkdir(parents=True)
    subprocess.run(
        ("git", "-C", str(repository), "commit", "-qm", "add gitlink"),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    snapshot = _snapshot(repository)
    gitlink = snapshot.disposition_for_path("vendor/nested")
    assert gitlink is not None
    assert gitlink.entry_kind is EntryKind.GITLINK
    assert gitlink.kind is CoverageKind.DEPENDENCY_TOOL_IDENTITY
    assert gitlink.reason_code == "gitlink_submodule"
    assert snapshot.stats.gitlink_count == 1
    assert len(snapshot.gitlinks) == 1
    assert snapshot.gitlinks[0].path == "vendor/nested"
    assert snapshot.gitlinks[0].commit_id == nested_commit.lower()
    assert any(
        item.kind is DependencyIdentityKind.GITLINK for item in snapshot.dependency_identities
    )


def test_symlink_is_hashed_and_escape_is_rejected(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    os.symlink("src/service.ts", repository / "alias.ts")
    _git(repository, "add", "alias.ts")
    _git(repository, "commit", "-qm", "symlink")

    snapshot = _snapshot(repository)
    alias = snapshot.disposition_for_path("alias.ts")
    assert alias is not None
    assert alias.entry_kind is EntryKind.SYMLINK
    assert alias.git_mode == "120000"
    assert alias.content_digest.startswith("sha256:")

    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    os.symlink(outside, repository / "escape-link")
    # Tracked escapes are rejected when the worktree is hashed (dirty overlay
    # path).  Stage the untracked allowlisted path with a suffix that would
    # otherwise be admitted.
    os.symlink(outside, repository / "escape.ts")
    with pytest.raises(SymlinkEscapeError):
        _snapshot(repository)


def test_path_escape_helpers_and_malformed_paths() -> None:
    assert repo_path("src/service.ts") == "src/service.ts"
    assert repo_path("./src/service.ts") == "src/service.ts"
    with pytest.raises(RepositoryPathEscapeError):
        repo_path("../secret")
    with pytest.raises(RepositoryPathEscapeError):
        repo_path("/abs/path")
    with pytest.raises(RepositoryPathEscapeError):
        repo_path("a/../../b")
    with pytest.raises(RepositoryPathEscapeError):
        repo_path("a\x00b")


def test_canonical_ordering_is_path_sorted_and_deterministic(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    first = _snapshot(repository)
    second = _snapshot(repository)
    assert first.snapshot_id == second.snapshot_id
    assert [item.path for item in first.dispositions] == sorted(
        item.path for item in first.dispositions
    )
    assert [item.to_dict() for item in first.dispositions] == [
        item.to_dict() for item in second.dispositions
    ]
    # Identity is stable under reverse construction of the same ledger.
    rebuilt = RepositorySnapshot(
        primary_root=first.primary_root,
        head_commit_id=first.head_commit_id,
        head_tree_id=first.head_tree_id,
        index_tree_id=first.index_tree_id,
        scope_policy_id=first.scope_policy_id,
        scope_id=first.scope_id,
        dispositions=tuple(reversed(first.dispositions)),
        dependency_identities=tuple(reversed(first.dependency_identities)),
        gitlinks=tuple(reversed(first.gitlinks)),
        stats=first.stats,
        allow_dirty_analysis=first.allow_dirty_analysis,
    )
    assert rebuilt.snapshot_id == first.snapshot_id
    assert [item.path for item in rebuilt.dispositions] == [
        item.path for item in first.dispositions
    ]


def test_dependency_directories_are_lock_and_tool_identities_not_source(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    # Track a file under node_modules: must be dependency identity / excluded
    # by directory policy, never semantic AST authority.
    _write(
        repository / "node_modules" / "left-pad" / "index.js",
        "module.exports = function(){}\n",
    )
    _git(repository, "add", "-f", "node_modules/left-pad/index.js")
    _git(repository, "commit", "-qm", "vendor dep path")

    snapshot = _snapshot(repository)
    dep = snapshot.disposition_for_path("node_modules/left-pad/index.js")
    assert dep is not None
    # skip_directory_names / skip_prefixes win as explicit exclusion, or
    # dependency_directory_names bind as dependency tool identity.
    assert dep.kind in {
        CoverageKind.EXCLUDED,
        CoverageKind.DEPENDENCY_TOOL_IDENTITY,
    }
    assert dep.kind is not CoverageKind.SEMANTIC_AST
    locks = [
        item
        for item in snapshot.dependency_identities
        if item.path in {"yarn.lock", "package.json"}
    ]
    assert locks
    assert all(item.digest for item in locks)


def test_classify_coverage_kind_rules_are_explicit() -> None:
    from ipfs_accelerate_py.agent_supervisor.analysis.repository_snapshot import (
        primary_relative_prefixes,
    )

    policy = scope_policy_from_mapping(_minimal_policy())
    kind, reason, rule = classify_coverage_kind(
        "src/a.ts", policy=policy, entry_kind=EntryKind.REGULAR
    )
    assert kind is CoverageKind.SEMANTIC_AST
    assert reason == "semantic_extension"
    kind, reason, rule = classify_coverage_kind(
        "node_modules/x/index.js", policy=policy, entry_kind=EntryKind.REGULAR
    )
    assert kind is CoverageKind.EXCLUDED
    assert "skip" in rule or reason.startswith("excluded")
    kind, reason, rule = classify_coverage_kind(
        "pkg.d.ts", policy=policy, entry_kind=EntryKind.REGULAR
    )
    assert kind is CoverageKind.BINARY_OR_GENERATED
    kind, reason, rule = classify_coverage_kind(
        "unknown.xyz", policy=policy, entry_kind=EntryKind.REGULAR
    )
    assert kind is CoverageKind.UNSUPPORTED
    assert reason == "unsupported_extension"

    # Superproject prefixes expand only for the primary root segment.
    expanded = primary_relative_prefixes(
        (
            "swissknife/node_modules",
            "hallucinate_app/node_modules",
            "node_modules",
        ),
        primary_root="swissknife",
        primary_repository="swissknife",
    )
    assert "node_modules" in expanded
    assert "swissknife/node_modules" in expanded
    assert "hallucinate_app/node_modules" in expanded
    # Primary-relative path must not match an unrelated superproject prefix.
    kind, reason, rule = classify_coverage_kind(
        "node_modules/x/index.js",
        policy=scope_policy_from_mapping(
            _minimal_policy(
                skipPrefixes=["hallucinate_app/node_modules"],
                skipDirectoryNames=[".git"],
            )
        ),
        entry_kind=EntryKind.REGULAR,
        skip_prefixes=primary_relative_prefixes(
            ("hallucinate_app/node_modules",),
            primary_root="swissknife",
            primary_repository="swissknife",
        ),
    )
    assert kind is CoverageKind.DEPENDENCY_TOOL_IDENTITY
    assert reason == "dependency_directory"


def test_allow_dirty_analysis_false_rejects_dirty_tree(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    (repository / "src" / "service.ts").write_text("export const x = 1\n", encoding="utf-8")
    with pytest.raises(RepositoryStateError):
        _snapshot(repository, allow_dirty_analysis=False)


def test_duplicate_disposition_construction_fails() -> None:
    from ipfs_accelerate_py.agent_supervisor.analysis.repository_snapshot import (
        RepositorySnapshotStats,
    )

    first = CoverageDisposition(
        path="a.ts",
        kind=CoverageKind.SEMANTIC_AST,
        git_status=GitStatus.CLEAN,
        entry_kind=EntryKind.REGULAR,
        reason_code="semantic_extension",
        policy_rule="semantic_extensions:.ts",
        content_digest="sha256:" + "0" * 64,
    )
    stats = RepositorySnapshotStats(
        tracked_path_count=1,
        disposition_count=2,
        overlay_path_count=0,
        excluded_path_count=0,
        dependency_identity_count=0,
        gitlink_count=0,
        dirty_path_count=0,
        deleted_path_count=0,
        untracked_path_count=0,
        semantic_path_count=1,
        unsupported_path_count=0,
        hashed_bytes=0,
    )
    with pytest.raises(CoverageIncompleteError):
        RepositorySnapshot(
            primary_root=".",
            head_commit_id="a" * 40,
            head_tree_id="b" * 40,
            index_tree_id="b" * 40,
            scope_policy_id="policy",
            scope_id="scope",
            dispositions=(first, first),
            dependency_identities=(),
            gitlinks=(),
            stats=stats,
        )


def test_snapshot_analyzer_health_inventory_shape(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    snapshot = _snapshot(repository)
    inventory = snapshot_analyzer_health_inventory(snapshot)
    assert inventory["coverage_complete"] is True
    assert inventory["tracked_files"] == snapshot.stats.tracked_path_count
    assert inventory["git_roots"] == 1
    assert inventory["snapshot_id"] == snapshot.snapshot_id


def test_invalid_scope_policy_fails_closed() -> None:
    with pytest.raises(ScopePolicyError):
        scope_policy_from_mapping({"schemaVersion": 1})
    with pytest.raises(ScopePolicyError):
        scope_policy_from_mapping(
            {
                "schema": "other@1",
                "scopeId": "x",
                "primaryRoot": ".",
            }
        )


def test_coverage_disposition_schema_fields_are_stable(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    snapshot = _snapshot(repository)
    item = snapshot.dispositions[0]
    payload = item.to_dict()
    assert payload["schema"] == COVERAGE_DISPOSITION_SCHEMA
    assert payload["kind"] in {kind.value for kind in CoverageKind}
    assert payload["git_status"] in {status.value for status in GitStatus}
    assert payload["entry_kind"] in {kind.value for kind in EntryKind}
