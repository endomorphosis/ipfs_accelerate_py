"""Tests for RepositoryReasoningSnapshot@1 (PDR-010)."""

from __future__ import annotations

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.repository_reasoning_snapshot import (
    REPOSITORY_REASONING_SNAPSHOT_INTERFACE,
    REPOSITORY_REASONING_SNAPSHOT_SCHEMA,
    ReasoningCoverageKind,
    ReasoningEntryKind,
    ReasoningGitlinkEntry,
    ReasoningPathEntry,
    ReasoningPathStatus,
    ReasoningStability,
    ReasoningToolRoots,
    ReasoningTruncation,
    RepositoryReasoningAuthorityError,
    RepositoryReasoningInstabilityError,
    RepositoryReasoningSnapshot,
    RepositoryReasoningSnapshotError,
    RepositoryReasoningTamperError,
    TaskSourceBinding,
    build_repository_reasoning_snapshot,
    gitlink_from_sca_record,
    map_git_status,
    path_entry_from_sca_disposition,
    reasoning_snapshot_from_sca_snapshot,
)
from ipfs_accelerate_py.agent_supervisor.analysis.repository_snapshot import (
    CoverageDisposition,
    CoverageKind,
    EntryKind,
    GitStatus,
    GitlinkRecord,
    RepositorySnapshot,
    RepositorySnapshotStats,
)


def _roots(**overrides: str) -> ReasoningToolRoots:
    base = {
        "repository_id": "repository:fixture",
        "forest_id": "forest:fixture",
        "tree_id": "tree:fixture",
        "overlay_id": "overlay:fixture",
        "head_commit_id": "a" * 40,
        "head_tree_id": "b" * 40,
        "index_tree_id": "c" * 40,
        "parser_root": "parser:program-ast-adapters@1",
        "index_root": "index:fixture",
        "toolchain_root": "toolchain:fixture",
        "capability_root": "capability:fixture",
        "policy_root": "policy:fixture",
        "ir_root": "ir:fixture",
        "intent_ir_root": "intent-ir:fixture",
        "legal_ir_root": "legal-ir:fixture",
        "security_ir_root": "security-ir:fixture",
        "program_behavior_root": "program-behavior:fixture",
        "ast_root": "ast:fixture",
        "scope_policy_id": "scope-policy:fixture",
        "scanner_root": "scanner:fixture",
    }
    base.update(overrides)
    return ReasoningToolRoots(**base)


def _path(
    path: str,
    status: ReasoningPathStatus,
    *,
    tracked: bool = True,
    coverage: ReasoningCoverageKind = ReasoningCoverageKind.ADMITTED,
    rename_from: str = "",
    reason_code: str = "admitted",
    policy_rule: str = "default",
    overlay: bool = False,
) -> ReasoningPathEntry:
    return ReasoningPathEntry(
        path=path,
        status=status,
        coverage=coverage,
        entry_kind=ReasoningEntryKind.REGULAR,
        tracked=tracked,
        overlay=overlay,
        rename_from=rename_from,
        reason_code=reason_code,
        policy_rule=policy_rule,
        worktree_digest=f"sha256:{path.replace('/', '_')}",
    )


def _full_snapshot(**kwargs: object) -> RepositoryReasoningSnapshot:
    paths = kwargs.pop(
        "paths",
        [
            _path("src/a.py", ReasoningPathStatus.TRACKED),
            _path("src/b.py", ReasoningPathStatus.STAGED, overlay=True),
            _path("src/c.py", ReasoningPathStatus.MODIFIED, overlay=True),
            _path("src/d.py", ReasoningPathStatus.DELETED, overlay=True),
            _path(
                "src/e.py",
                ReasoningPathStatus.RENAMED,
                rename_from="src/e_old.py",
                overlay=True,
            ),
            _path(
                "scratch/note.py",
                ReasoningPathStatus.ADMITTED_UNTRACKED,
                tracked=False,
                overlay=True,
                reason_code="policy_admitted_untracked",
            ),
            _path(
                "vendor/lib.py",
                ReasoningPathStatus.EXCLUDED,
                coverage=ReasoningCoverageKind.EXCLUDED,
                reason_code="excluded_prefix",
                policy_rule="skip_prefixes",
            ),
        ],
    )
    gitlinks = kwargs.pop(
        "gitlinks",
        [
            ReasoningGitlinkEntry(
                path="vendor/sub",
                commit_id="d" * 40,
                depth=0,
                nested=(
                    ReasoningGitlinkEntry(
                        path="vendor/sub/nested",
                        commit_id="e" * 40,
                        depth=1,
                        parent_path="vendor/sub",
                    ),
                ),
            )
        ],
    )
    task_source = kwargs.pop(
        "task_source",
        TaskSourceBinding(
            revision=3,
            status="ready",
            evidence_id="evidence:task:1",
            event_cursor="cursor:event:42",
            plan_root="plan:fixture",
            board_namespace="agent-supervisor-proof-directed-planner-doctor-v1",
            source_kind="duckdb",
            task_population_id="population:fixture",
            evidence_refs=("evidence:task:1", "evidence:goal:1"),
        ),
    )
    return build_repository_reasoning_snapshot(
        roots=kwargs.pop("roots", _roots()),
        paths=paths,  # type: ignore[arg-type]
        gitlinks=gitlinks,  # type: ignore[arg-type]
        exclusions=kwargs.pop("exclusions", ("vendor/lib.py", "vendor/")),
        task_source=task_source,  # type: ignore[arg-type]
        stability=kwargs.pop("stability", ReasoningStability(stable=True)),
        truncation=kwargs.pop("truncation", ReasoningTruncation(truncated=False)),
        primary_root=str(kwargs.pop("primary_root", ".")),
        scope_id=str(kwargs.pop("scope_id", "scope:fixture")),
        dirty_overlay_id=str(kwargs.pop("dirty_overlay_id", "overlay:fixture")),
        completeness=str(kwargs.pop("completeness", "complete")),
        notes=kwargs.pop("notes", ("fixture",)),  # type: ignore[arg-type]
    )


def test_interface_constant() -> None:
    assert REPOSITORY_REASONING_SNAPSHOT_INTERFACE == "RepositoryReasoningSnapshot@1"
    assert REPOSITORY_REASONING_SNAPSHOT_SCHEMA.endswith(
        "repository-reasoning-snapshot@1"
    )


def test_snapshot_covers_all_required_path_statuses() -> None:
    snap = _full_snapshot()
    statuses = {item.status for item in snap.paths}
    assert ReasoningPathStatus.TRACKED in statuses
    assert ReasoningPathStatus.STAGED in statuses
    assert ReasoningPathStatus.MODIFIED in statuses
    assert ReasoningPathStatus.DELETED in statuses
    assert ReasoningPathStatus.RENAMED in statuses
    assert ReasoningPathStatus.ADMITTED_UNTRACKED in statuses
    assert ReasoningPathStatus.EXCLUDED in statuses

    assert len(snap.tracked_paths()) >= 1
    assert len(snap.staged_paths()) >= 1
    assert len(snap.modified_paths()) >= 1
    assert len(snap.deleted_paths()) >= 1
    assert len(snap.renamed_paths()) == 1
    assert snap.renamed_paths()[0].rename_from == "src/e_old.py"
    assert len(snap.admitted_untracked_paths()) == 1
    assert snap.admitted_untracked_paths()[0].tracked is False


def test_recursive_gitlinks_and_exclusions() -> None:
    snap = _full_snapshot()
    flat = snap.recursive_gitlinks()
    assert len(flat) == 2
    assert flat[0].depth == 0
    assert flat[1].depth == 1
    assert flat[1].parent_path == "vendor/sub"
    assert "vendor/lib.py" in snap.exclusions
    # Trailing-slash prefixes normalize to the directory path.
    assert "vendor" in snap.exclusions


def test_all_tool_and_policy_roots_bound() -> None:
    roots = _roots()
    for name in (
        "parser_root",
        "index_root",
        "toolchain_root",
        "capability_root",
        "policy_root",
        "ir_root",
        "intent_ir_root",
        "legal_ir_root",
        "security_ir_root",
        "program_behavior_root",
        "ast_root",
        "scope_policy_id",
        "scanner_root",
    ):
        assert getattr(roots, name), name
    snap = _full_snapshot(roots=roots)
    assert snap.roots.repository_id == "repository:fixture"
    assert snap.roots.forest_id == "forest:fixture"
    assert snap.roots.tree_id == "tree:fixture"


def test_task_source_revision_status_evidence_event_cursor() -> None:
    snap = _full_snapshot()
    assert snap.task_source is not None
    assert snap.task_source.revision == 3
    assert snap.task_source.status == "ready"
    assert snap.task_source.evidence_id == "evidence:task:1"
    assert snap.task_source.event_cursor == "cursor:event:42"
    assert "evidence:task:1" in snap.task_source.evidence_refs


def test_instability_and_truncation_witnesses() -> None:
    unstable = _full_snapshot(
        stability=ReasoningStability(
            stable=False,
            instability_codes=("bytes_changed_during_scan",),
            preflight_digest="sha256:pre",
            postflight_digest="sha256:post",
            witnesses=("preflight", "postflight"),
        ),
        truncation=ReasoningTruncation(
            truncated=True,
            reasons=("max_paths", "max_symbols"),
            max_paths=16,
            omitted_path_count=2,
            omitted_symbol_count=10,
        ),
        completeness="partial_with_frontier",
    )
    assert unstable.stability.stable is False
    assert "bytes_changed_during_scan" in unstable.stability.instability_codes
    assert unstable.truncation.truncated is True
    assert "max_paths" in unstable.truncation.reasons
    with pytest.raises(RepositoryReasoningInstabilityError):
        unstable.assert_stable()

    with pytest.raises(RepositoryReasoningSnapshotError):
        ReasoningStability(stable=True, instability_codes=("x",))
    with pytest.raises(RepositoryReasoningSnapshotError):
        ReasoningTruncation(truncated=True, reasons=())
    with pytest.raises(RepositoryReasoningSnapshotError):
        _full_snapshot(
            truncation=ReasoningTruncation(
                truncated=True, reasons=("max_paths",), omitted_path_count=1
            ),
            completeness="complete",
        )


def test_round_trip_to_dict_from_dict() -> None:
    snap = _full_snapshot()
    payload = snap.to_dict()
    restored = RepositoryReasoningSnapshot.from_dict(payload)
    assert restored.content_id == snap.content_id
    assert restored.snapshot_id == snap.snapshot_id
    assert restored.inventory()["path_count"] == snap.inventory()["path_count"]
    assert restored.task_source is not None
    assert restored.task_source.event_cursor == "cursor:event:42"

    record = snap.to_record()
    assert record["content_id"] == snap.content_id
    assert record["inventory"]["gitlink_count"] == 2


def test_tampering_and_unknown_fields_fail_closed() -> None:
    snap = _full_snapshot()
    payload = snap.to_dict()
    payload["content_id"] = "b" + "a" * 58
    with pytest.raises(RepositoryReasoningTamperError):
        RepositoryReasoningSnapshot.from_dict(payload)

    payload = snap.to_dict()
    payload["unexpected_field"] = "nope"
    with pytest.raises(RepositoryReasoningSnapshotError):
        RepositoryReasoningSnapshot.from_dict(payload)


def test_body_and_secret_material_rejected() -> None:
    roots = _roots()
    payload = {
        "schema": REPOSITORY_REASONING_SNAPSHOT_SCHEMA,
        "contract_version": 1,
        "roots": roots.to_dict(),
        "paths": [],
        "source": "def evil():\n    pass\n",
    }
    with pytest.raises(RepositoryReasoningSnapshotError):
        RepositoryReasoningSnapshot.from_dict(payload)

    payload = {
        "schema": REPOSITORY_REASONING_SNAPSHOT_SCHEMA,
        "contract_version": 1,
        "roots": roots.to_dict(),
        "paths": [],
        "api_key": "sk-secret",
    }
    with pytest.raises(RepositoryReasoningSnapshotError):
        RepositoryReasoningSnapshot.from_dict(payload)

    # Nested body marker inside an otherwise valid path payload.
    bad_path = {
        "schema": "ipfs_accelerate_py/agent-supervisor/repository-reasoning-path-entry@1",
        "contract_version": 1,
        "path": "src/a.py",
        "status": "tracked",
        "source_text": "print('nope')",
    }
    with pytest.raises(RepositoryReasoningSnapshotError):
        ReasoningPathEntry.from_dict(bad_path)


def test_cross_repository_require_repository() -> None:
    snap = _full_snapshot()
    snap.require_repository("repository:fixture")
    with pytest.raises(RepositoryReasoningAuthorityError):
        snap.require_repository("repository:other")
    other = _roots(repository_id="repository:other")
    with pytest.raises(RepositoryReasoningAuthorityError):
        snap.roots.require_same_repository(other)


def test_duplicate_paths_fail_closed() -> None:
    with pytest.raises(RepositoryReasoningSnapshotError):
        build_repository_reasoning_snapshot(
            roots=_roots(),
            paths=[
                _path("src/a.py", ReasoningPathStatus.TRACKED),
                _path("src/a.py", ReasoningPathStatus.MODIFIED),
            ],
        )


def test_renamed_requires_rename_from() -> None:
    with pytest.raises(RepositoryReasoningSnapshotError):
        _path("src/e.py", ReasoningPathStatus.RENAMED)


def test_admitted_untracked_cannot_be_tracked() -> None:
    with pytest.raises(RepositoryReasoningSnapshotError):
        ReasoningPathEntry(
            path="scratch/x.py",
            status=ReasoningPathStatus.ADMITTED_UNTRACKED,
            tracked=True,
            reason_code="policy_admitted_untracked",
            policy_rule="default",
        )


def test_path_escape_rejected() -> None:
    with pytest.raises(RepositoryReasoningAuthorityError):
        _path("../outside.py", ReasoningPathStatus.TRACKED)
    with pytest.raises(RepositoryReasoningAuthorityError):
        _path("/abs/path.py", ReasoningPathStatus.TRACKED)


def test_map_git_status_and_sca_bridges() -> None:
    assert map_git_status("staged") is ReasoningPathStatus.STAGED
    assert map_git_status(GitStatus.UNTRACKED) is ReasoningPathStatus.ADMITTED_UNTRACKED
    assert map_git_status("renamed") is ReasoningPathStatus.RENAMED

    disposition = CoverageDisposition(
        path="pkg/mod.py",
        kind=CoverageKind.SEMANTIC_AST,
        git_status=GitStatus.MODIFIED,
        entry_kind=EntryKind.REGULAR,
        reason_code="tracked_modified",
        policy_rule="default",
        content_digest="sha256:abc",
        overlay=True,
        tracked=True,
    )
    entry = path_entry_from_sca_disposition(disposition)
    assert entry.path == "pkg/mod.py"
    assert entry.status is ReasoningPathStatus.MODIFIED
    assert entry.overlay is True

    gitlink = gitlink_from_sca_record(
        GitlinkRecord(path="vendor/dep", commit_id="f" * 40)
    )
    assert gitlink.path == "vendor/dep"
    assert gitlink.commit_id == "f" * 40


def test_reasoning_snapshot_from_sca_snapshot() -> None:
    dispositions = (
        CoverageDisposition(
            path="src/main.py",
            kind=CoverageKind.SEMANTIC_AST,
            git_status=GitStatus.CLEAN,
            entry_kind=EntryKind.REGULAR,
            reason_code="tracked_clean",
            policy_rule="default",
            content_digest="sha256:1",
            tracked=True,
        ),
        CoverageDisposition(
            path="src/new.py",
            kind=CoverageKind.SEMANTIC_AST,
            git_status=GitStatus.UNTRACKED,
            entry_kind=EntryKind.REGULAR,
            reason_code="policy_admitted_untracked",
            policy_rule="untracked_allow",
            content_digest="sha256:2",
            tracked=False,
            overlay=True,
        ),
        CoverageDisposition(
            path="build/out.bin",
            kind=CoverageKind.EXCLUDED,
            git_status=GitStatus.CLEAN,
            entry_kind=EntryKind.REGULAR,
            reason_code="excluded_prefix",
            policy_rule="skip_prefixes",
            tracked=True,
        ),
    )
    stats = RepositorySnapshotStats(
        tracked_path_count=3,
        disposition_count=3,
        overlay_path_count=1,
        excluded_path_count=1,
        dependency_identity_count=0,
        gitlink_count=1,
        dirty_path_count=1,
        deleted_path_count=0,
        untracked_path_count=1,
        semantic_path_count=2,
        unsupported_path_count=0,
        hashed_bytes=12,
    )
    sca = RepositorySnapshot(
        primary_root=".",
        head_commit_id="1" * 40,
        head_tree_id="2" * 40,
        index_tree_id="3" * 40,
        scope_policy_id="scope-policy:test",
        scope_id="scope:test",
        dispositions=dispositions,
        dependency_identities=(),
        gitlinks=(GitlinkRecord(path="third_party/x", commit_id="4" * 40),),
        stats=stats,
    )
    snap = reasoning_snapshot_from_sca_snapshot(
        sca,
        roots=_roots(
            head_commit_id="",
            head_tree_id="",
            index_tree_id="",
        ),
        task_source=TaskSourceBinding(
            revision=1,
            status="pending",
            evidence_id="evidence:sca",
            event_cursor="cursor:0",
        ),
    )
    assert snap.roots.head_commit_id == "1" * 40
    assert any(
        item.status is ReasoningPathStatus.ADMITTED_UNTRACKED for item in snap.paths
    )
    assert any(
        item.coverage is ReasoningCoverageKind.EXCLUDED for item in snap.paths
    )
    assert len(snap.gitlinks) == 1
    assert snap.task_source is not None
    assert snap.task_source.revision == 1


def test_task_source_revision_must_be_positive() -> None:
    with pytest.raises(RepositoryReasoningSnapshotError):
        TaskSourceBinding(revision=0, status="ready")
