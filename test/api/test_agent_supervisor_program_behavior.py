from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.core import program_behavior
from ipfs_accelerate_py.agent_supervisor.artifact_store import (
    ArtifactQuotaPolicy,
    BoundedArtifactStore,
)
from ipfs_accelerate_py.agent_supervisor.core.program_behavior import (
    ProgramObservationKind,
    ProposedEffect,
    ProposedEffectKind,
    ProposedEffectManifest,
    RepositoryEntryStatus,
    RepositoryPathEscapeError,
    RepositoryRaceError,
    RequiredInputTooLargeError,
    SnapshotBounds,
    SymlinkEscapeError,
    UnsupportedEffectError,
    build_program_behavior,
    build_repository_snapshot,
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


def _repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Behavior Test")
    _git(repository, "config", "user.email", "behavior@example.invalid")
    (repository / "src").mkdir()
    (repository / "src" / "service.py").write_text(
        """class Service:
    def dispatch(self, request):
        self.status = "running"
        return transform(request)
""",
        encoding="utf-8",
    )
    (repository / "README.md").write_text("fixture\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "-qm", "fixture")
    return repository


def _effect(
    effect_id: str = "effect:file",
    *,
    kind: ProposedEffectKind = ProposedEffectKind.FILE,
    operation: str = "write",
    target: str = "src/service.py",
    paths: tuple[str, ...] = ("src/service.py",),
    credentials: tuple[str, ...] = (),
) -> ProposedEffect:
    return ProposedEffect(
        effect_id=effect_id,
        kind=kind,
        operation=operation,
        target=target,
        repository_paths=paths,
        credential_ids=credentials,
    )


def test_clean_snapshot_uses_exact_git_tree_and_referenced_blobs(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)

    snapshot = build_repository_snapshot(repository)

    assert snapshot.is_clean is True
    assert snapshot.execution_tree_root == _git(repository, "rev-parse", "HEAD^{tree}")
    assert snapshot.head_tree_id == snapshot.index_tree_id
    assert {item.status for item in snapshot.entries} == {
        RepositoryEntryStatus.CLEAN
    }
    rendered = snapshot.to_json()
    assert "class Service" not in rendered
    assert "fixture\\n" not in rendered
    assert all(item.worktree_blob is not None for item in snapshot.entries)
    assert all(
        item.worktree_blob.artifact_id.startswith("blob:sha256:")
        for item in snapshot.entries
    )
    scoped = build_repository_snapshot(repository, scopes=("src",))
    assert scoped.is_clean is False
    assert scoped.execution_tree_root == scoped.snapshot_id


def test_dirty_snapshot_binds_head_index_worktree_untracked_hidden_and_modes(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    original = build_repository_snapshot(repository)
    service = repository / "src" / "service.py"
    service.write_text("def changed():\n    return 2\n", encoding="utf-8")
    _git(repository, "add", "src/service.py")
    service.write_text("def changed_again():\n    return 3\n", encoding="utf-8")
    (repository / ".gitignore").write_text(".hidden-runtime\n", encoding="utf-8")
    (repository / ".hidden-runtime").write_bytes(b"decision-changing")
    (repository / "script.sh").write_text("#!/bin/sh\ntrue\n", encoding="utf-8")
    os.chmod(repository / "script.sh", 0o755)

    dirty = build_repository_snapshot(repository, previous=original)
    changed = dirty.entry_for_path("src/service.py")
    hidden = dirty.entry_for_path(".hidden-runtime")
    script = dirty.entry_for_path("script.sh")

    assert dirty.dirty is True
    assert dirty.execution_tree_root == dirty.snapshot_id
    assert dirty.snapshot_id != original.snapshot_id
    assert changed is not None
    assert changed.status is RepositoryEntryStatus.STAGED_AND_MODIFIED
    assert len(
        {
            changed.head_blob.digest,
            changed.index_blob.digest,
            changed.worktree_blob.digest,
        }
    ) == 3
    assert hidden is not None
    assert hidden.status is RepositoryEntryStatus.UNTRACKED
    assert hidden.worktree_blob.size_bytes == len(b"decision-changing")
    assert script is not None and script.worktree_mode == "100755"

    (repository / ".hidden-runtime").write_bytes(b"changed")
    changed_hidden = build_repository_snapshot(repository)
    assert changed_hidden.snapshot_id != dirty.snapshot_id


def test_deletion_rename_and_internal_symlink_are_explicit(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    (repository / "old.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repository, "add", "old.py")
    _git(repository, "commit", "-qm", "add old")
    _git(repository, "mv", "old.py", "new.py")
    (repository / "README.md").unlink()
    os.symlink("new.py", repository / "alias.py")

    snapshot = build_repository_snapshot(repository)

    renamed = snapshot.entry_for_path("new.py")
    deleted = snapshot.entry_for_path("README.md")
    alias = snapshot.entry_for_path("alias.py")
    assert renamed is not None
    assert renamed.status is RepositoryEntryStatus.RENAMED
    assert renamed.rename_from == "old.py"
    assert deleted is not None
    assert deleted.status is RepositoryEntryStatus.DELETED
    assert deleted.worktree_blob is None
    assert alias is not None
    assert alias.worktree_mode == "120000"
    assert alias.worktree_blob.size_bytes == len(b"new.py")


def test_incremental_ast_and_observations_reuse_exact_unchanged_records(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    effect = _effect()

    cold = build_program_behavior(repository, effects=(effect,))
    warm = build_program_behavior(repository, effects=(effect,), previous=cold)

    assert warm.behavior_root == cold.behavior_root
    assert warm.repository.execution_tree_root == cold.repository.execution_tree_root
    assert warm.analysis.ast_index.stats.reused_blob_count == 1
    assert (
        warm.analysis.ast_index.path_records[0].ast_record
        is cold.analysis.ast_index.path_records[0].ast_record
    )
    assert {
        item.kind for item in warm.analysis.observations
    }.issuperset(
        {
            ProgramObservationKind.AST,
            ProgramObservationKind.SYMBOL,
            ProgramObservationKind.INTERFACE,
            ProgramObservationKind.CALL,
            ProgramObservationKind.DATA_FLOW,
        }
    )
    rendered = warm.to_json()
    assert "self.status" not in rendered
    assert '"source_text"' not in rendered
    assert warm.analysis.ast_index_blob.artifact_id.startswith("blob:sha256:")
    assert warm.analysis.observations_blob.artifact_id.startswith("blob:sha256:")


def test_behavior_root_binds_tools_environment_and_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    monkeypatch.setenv("BEHAVIOR_PROFILE", "one")
    effect = _effect()
    first = build_program_behavior(
        repository,
        effects=(effect,),
        environment_variable_names=("BEHAVIOR_PROFILE",),
    )
    monkeypatch.setenv("BEHAVIOR_PROFILE", "two")
    environment_changed = build_program_behavior(
        repository,
        effects=(effect,),
        environment_variable_names=("BEHAVIOR_PROFILE",),
    )
    effect_changed = build_program_behavior(
        repository,
        effects=(
            ProposedEffect(
                "effect:file",
                ProposedEffectKind.FILE,
                "delete",
                "src/service.py",
                ("src/service.py",),
            ),
        ),
        environment_variable_names=("BEHAVIOR_PROFILE",),
    )

    assert first.tools.tools
    assert all(item.version and item.version_digest for item in first.tools.tools)
    assert first.behavior_root != environment_changed.behavior_root
    assert environment_changed.behavior_root != effect_changed.behavior_root
    assert (
        first.effects.manifest_root
        != effect_changed.effects.manifest_root
    )


def test_all_typed_effect_domains_are_supported_and_unknowns_fail_closed() -> None:
    effects = (
        _effect(),
        _effect(
            "effect:process",
            kind=ProposedEffectKind.PROCESS,
            operation="execute",
            target="pytest",
            paths=(),
        ),
        _effect(
            "effect:network",
            kind=ProposedEffectKind.NETWORK,
            operation="request",
            target="https://example.invalid/api",
            paths=(),
        ),
        _effect(
            "effect:credential",
            kind=ProposedEffectKind.CREDENTIAL,
            operation="use",
            target="github",
            paths=(),
            credentials=("credential:github-token",),
        ),
        _effect(
            "effect:dataset",
            kind=ProposedEffectKind.DATASET,
            operation="publish",
            target="dataset:analysis",
            paths=(),
        ),
        _effect(
            "effect:board",
            kind=ProposedEffectKind.TASK_BOARD,
            operation="update",
            target="task:ASI-129",
            paths=(),
        ),
        _effect(
            "effect:commit",
            kind=ProposedEffectKind.COMMIT,
            operation="create",
            target="repository:test",
            paths=(),
        ),
        _effect(
            "effect:merge",
            kind=ProposedEffectKind.MERGE,
            operation="merge",
            target="branch:main",
            paths=(),
        ),
    )
    manifest = ProposedEffectManifest(effects)

    assert {item.kind for item in manifest.effects} == set(ProposedEffectKind)
    assert manifest.manifest_root.startswith("proposed-effect-manifest:sha256:")
    with pytest.raises(UnsupportedEffectError, match="unsupported"):
        ProposedEffect("bad", "email", "send", "user")  # type: ignore[arg-type]
    with pytest.raises(UnsupportedEffectError, match="unsupported file"):
        ProposedEffect("bad", ProposedEffectKind.FILE, "execute", "x", ("x",))
    with pytest.raises(UnsupportedEffectError, match="credential_ids"):
        ProposedEffect(
            "bad",
            ProposedEffectKind.CREDENTIAL,
            "use",
            "github",
        )
    with pytest.raises(program_behavior.ProgramBehaviorError, match="credential"):
        ProposedEffect(
            "bad",
            ProposedEffectKind.NETWORK,
            "request",
            "https://example.invalid",
            parameters={"token": "do-not-embed"},
        )


def test_artifact_store_contains_the_exact_referenced_bytes(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    store = BoundedArtifactStore(
        tmp_path / "artifacts",
        quotas=ArtifactQuotaPolicy(
            max_bytes=8 * 1024 * 1024,
            max_blob_bytes=2 * 1024 * 1024,
            max_blobs=100,
            max_projections=10,
        ),
    )

    behavior = build_program_behavior(
        repository,
        effects=(_effect(),),
        artifact_store=store,
    )
    source_ref = behavior.repository.entry_for_path(
        "src/service.py"
    ).worktree_blob

    assert store.verify_blob(source_ref)
    assert store.read_blob(source_ref) == (
        repository / "src" / "service.py"
    ).read_bytes()
    assert store.verify_blob(behavior.analysis.ast_index_blob)
    assert store.verify_blob(behavior.component_manifest_blob)
    assert "source" not in json.loads(
        store.read_blob(behavior.analysis.ast_index_blob).decode("utf-8")
    )["path_records"][0]["ast_record"]
    store.close()


def test_root_and_symlink_escapes_and_oversized_inputs_are_rejected(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path)
    outside = tmp_path / "outside"
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(RepositoryPathEscapeError, match="escapes"):
        build_repository_snapshot(repository, scopes=("../outside",))

    os.symlink(outside, repository / "outside-link")
    with pytest.raises(SymlinkEscapeError, match="escapes"):
        build_repository_snapshot(repository)
    (repository / "outside-link").unlink()

    os.symlink("../README.md", repository / "src" / "scoped-link")
    with pytest.raises(SymlinkEscapeError, match="byte scope"):
        build_repository_snapshot(repository, scopes=("src",))
    (repository / "src" / "scoped-link").unlink()

    (repository / "large.bin").write_bytes(b"x" * 33)
    with pytest.raises(RequiredInputTooLargeError, match="32 bytes"):
        build_repository_snapshot(
            repository,
            bounds=SnapshotBounds(
                max_file_bytes=32,
                max_total_bytes=1024,
                max_files=100,
                max_observations=100,
            ),
        )


def test_post_hash_change_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = _repository(tmp_path)
    original = program_behavior.build_program_analysis

    def mutate_after_analysis(*args: object, **kwargs: object):
        result = original(*args, **kwargs)
        (repository / "src" / "service.py").write_text(
            "def raced():\n    return False\n", encoding="utf-8"
        )
        return result

    monkeypatch.setattr(
        program_behavior, "build_program_analysis", mutate_after_analysis
    )
    with pytest.raises(RepositoryRaceError, match="changed after"):
        build_program_behavior(repository, effects=(_effect(),))
