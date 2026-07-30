"""Tests for exhaustive Git-aware multi-repository corpus inventories.

Covers ``vfs/exhaustive-file-inventory@1`` (VFS-G138) and packet co-binding
with ``vfs/incremental-ast-index@1`` under goal packet
``goal_packet/corpus_index/ipfs_accelerate_py/26d54d2206f9``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.repository_corpus_index import (
    CORPUS_INDEX_G020_EVIDENCE_TERMS,
    EXHAUSTIVE_FILE_INVENTORY_EVIDENCE,
    EXHAUSTIVE_FILE_INVENTORY_INVARIANTS,
    GOAL_PACKET_ID,
    INCREMENTAL_AST_INDEX_EVIDENCE,
    OBJECTIVE_DOMAIN_EVIDENCE_TERMS,
    OBJECTIVE_GOAL_ID,
    OBJECTIVE_PARENT_GOAL_ID,
    OBJECTIVE_TASK_ID,
    OBJECTIVE_VALIDATION_REPAIR_EVIDENCE,
    OBJECTIVE_VALIDATION_REPAIR_INVARIANTS,
    OBJECTIVE_VALIDATION_REPAIR_TASK_ID,
    PACKET_GOAL_IDS,
    CorpusClassification,
    EntryOrigin,
    InventoryLimits,
    RepositoryCorpusIndex,
    RepositoryCorpusIndexError,
    all_covered_evidence_terms,
    build_repository_corpus_index,
    covered_evidence_terms,
    exhaustive_file_inventory_evidence_terms,
    inventory_repository_descriptor,
    inventory_satisfies_exhaustive_file_inventory,
    objective_validation_repair_evidence_terms,
    packet_evidence_terms,
    parent_objective_evidence_terms,
    prove_exhaustive_file_inventory,
    prove_objective_validation_repair,
)
from ipfs_accelerate_py.agent_supervisor.repository_forest import (
    AuthorityMode,
    CaseUnicodePolicy,
    ForestPolicy,
    ForestRootSpec,
    IgnorePolicy,
    RepositoryAuthority,
    build_repository_descriptor,
    build_repository_forest,
)


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        text=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return completed.stdout.strip()


def _init_repo(path: Path, files: dict[str, bytes | str]) -> Path:
    path.mkdir(parents=True)
    _git(path, "init")
    _git(path, "checkout", "-b", "main")
    _git(path, "config", "user.name", "Corpus Test")
    _git(path, "config", "user.email", "corpus@example.invalid")
    for relative, content in files.items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "seed corpus")
    return path


def _descriptor(
    repo: Path,
    *,
    alias: str = "swissknife",
    ignore_policy: IgnorePolicy | None = None,
    case_policy: CaseUnicodePolicy | None = None,
):
    return build_repository_descriptor(
        repo,
        alias=alias,
        authority=RepositoryAuthority(mode=AuthorityMode.READ_ONLY.value),
        ignore_policy=ignore_policy,
        case_unicode_policy=case_policy,
    )


def _entry(index: RepositoryCorpusIndex, path: str, origin: str = "committed"):
    return next(
        item
        for item in index.entries
        if item.relative_path == path and item.origin == origin
    )


def test_swissknife_typescript_tree_is_exhaustive_and_classified(
    tmp_path: Path,
) -> None:
    repo = _init_repo(
        tmp_path / "swissknife",
        {
            "src/services/ipfs.ts": "export async function stat() {}\n",
            "src/components/App.tsx": "export const App = () => <main />;\n",
            "src/generated/client.generated.ts": "export const SDK = {};\n",
            "schemas/mcp.schema.json": '{"type":"object"}\n',
            "docs/vfs.md": "# VFS\n",
            "tests/ipfs.spec.ts": "test('ipfs', () => {});\n",
            "tests/fixtures/sample.ts": "export const sample = 1;\n",
            "vendor/dependency.js": "module.exports = {};\n",
            "dist/bundle.js": "(()=>{})();\n",
            "archives/sdk.tar.gz": b"\x1f\x8barchive",
            "assets/logo.png": b"\x89PNG\r\n\x1a\n\x00",
        },
    )
    result = inventory_repository_descriptor(_descriptor(repo))

    assert result.exhaustive is True
    assert result.reason_codes == ()
    assert len(result.entries) == 11
    assert _entry(result, "src/services/ipfs.ts").included
    assert _entry(result, "src/components/App.tsx").included
    generated = _entry(result, "src/generated/client.generated.ts")
    assert generated.included
    assert CorpusClassification.GENERATED_SOURCE.value in generated.classifications
    assert CorpusClassification.SCHEMA.value in _entry(
        result, "schemas/mcp.schema.json"
    ).classifications
    assert CorpusClassification.DOCS.value in _entry(
        result, "docs/vfs.md"
    ).classifications
    test_entry = _entry(result, "tests/ipfs.spec.ts")
    assert {"source", "tests"}.issubset(test_entry.classifications)
    fixture = _entry(result, "tests/fixtures/sample.ts")
    assert {"source", "tests", "fixtures"}.issubset(fixture.classifications)
    assert _entry(result, "vendor/dependency.js").reason_codes == (
        "vendored_dependency",
    )
    assert "build_output" in _entry(result, "dist/bundle.js").reason_codes
    assert {"archive", "binary"}.issubset(
        _entry(result, "archives/sdk.tar.gz").classifications
    )
    assert "binary" in _entry(result, "assets/logo.png").classifications
    assert all(item.blob_oid and item.canonical_path for item in result.entries)
    assert result.repositories[0].observed_entry_count == len(result.entries)


def test_committed_content_comes_from_git_and_dirty_overlay_supersedes_it(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo", {"src/tool.ts": "export const x = 1;\n"})
    clean_descriptor = _descriptor(repo)
    committed_oid = _git(repo, "rev-parse", "HEAD:src/tool.ts")
    (repo / "src/tool.ts").write_text("export const x = 2;\n", encoding="utf-8")
    (repo / "src/new.tsx").write_text("export const New = <p />;\n", encoding="utf-8")
    dirty_descriptor = _descriptor(repo)

    stale = inventory_repository_descriptor(clean_descriptor)
    assert stale.exhaustive is False
    assert stale.reason_codes == ("stale_repository_descriptor",)

    result = inventory_repository_descriptor(dirty_descriptor)
    committed = _entry(result, "src/tool.ts", EntryOrigin.COMMITTED.value)
    overlay = _entry(result, "src/tool.ts", EntryOrigin.DIRTY_OVERLAY.value)
    untracked = _entry(result, "src/new.tsx", EntryOrigin.DIRTY_OVERLAY.value)
    assert result.exhaustive
    assert committed.blob_oid == committed_oid
    assert committed.content_sha256 != overlay.content_sha256
    assert committed.included is False
    assert "superseded_by_dirty_overlay" in committed.reason_codes
    assert overlay.included and untracked.included
    assert overlay.base_blob_oid == committed_oid


def test_dirty_overlay_must_be_explicitly_allowed(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo", {"src/tool.ts": "export {};\n"})
    (repo / "dirty.ts").write_text("export const dirty = true;\n", encoding="utf-8")
    descriptor = _descriptor(
        repo,
        ignore_policy=IgnorePolicy(allow_dirty_overlay=False),
    )
    result = inventory_repository_descriptor(descriptor)
    assert result.exhaustive is False
    assert "dirty_overlay_forbidden" in result.reason_codes
    dirty = _entry(result, "dirty.ts", EntryOrigin.DIRTY_OVERLAY.value)
    assert dirty.included is False
    assert "dirty_overlay_forbidden" in dirty.reason_codes


def test_submodule_and_symlink_are_accounted_without_following(
    tmp_path: Path,
) -> None:
    child = _init_repo(tmp_path / "child", {"child.ts": "export {};\n"})
    parent = _init_repo(tmp_path / "parent", {"src/main.ts": "export {};\n"})
    _git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child),
        "packages/child",
    )
    (parent / "inside-link").symlink_to("src/main.ts")
    (parent / "escape-link").symlink_to("../outside.ts")
    _git(parent, "add", ".")
    _git(parent, "commit", "-m", "gitlink and links")

    result = inventory_repository_descriptor(_descriptor(parent))
    submodule = _entry(result, "packages/child")
    assert submodule.mode == "160000"
    assert submodule.object_type == "submodule"
    assert submodule.classifications == ("submodule",)
    assert submodule.reason_codes == ("submodule_gitlink",)
    assert _entry(result, "inside-link").classifications == ("symlink",)
    escaped = _entry(result, "escape-link")
    assert "symlink_target_escape" in escaped.reason_codes
    assert result.exhaustive is False
    assert "symlink_target_escape" in result.reason_codes


def test_unicode_and_case_collisions_fail_closed(tmp_path: Path) -> None:
    repo = _init_repo(
        tmp_path / "repo",
        {
            "src/Alpha.ts": "export const upper = 1;\n",
            "src/alpha.ts": "export const lower = 1;\n",
            "src/café.ts": "export const composed = 1;\n",
            "src/cafe\u0301.ts": "export const decomposed = 1;\n",
        },
    )
    descriptor = _descriptor(
        repo,
        case_policy=CaseUnicodePolicy(
            case_sensitive=False,
            unicode_normalization="NFC",
            reject_encoding_collisions=True,
        ),
    )
    result = inventory_repository_descriptor(descriptor)
    assert result.exhaustive is False
    assert "canonical_path_collision" in result.reason_codes
    collided = [
        item
        for item in result.entries
        if "canonical_path_collision" in item.reason_codes
    ]
    assert len(collided) == 4
    assert not any(item.included for item in collided)


def test_ignored_output_is_enumerated_and_policy_controls_admission(
    tmp_path: Path,
) -> None:
    repo = _init_repo(
        tmp_path / "repo",
        {
            ".gitignore": "ignored/\n",
            "src/main.ts": "export {};\n",
        },
    )
    ignored = repo / "ignored"
    ignored.mkdir()
    (ignored / "cache.ts").write_text("export const cache = 1;\n", encoding="utf-8")

    excluded = inventory_repository_descriptor(_descriptor(repo))
    ignored_entry = _entry(excluded, "ignored/cache.ts", EntryOrigin.IGNORED.value)
    assert {"ignored", "source"}.issubset(ignored_entry.classifications)
    assert ignored_entry.included is False
    assert "gitignored_by_policy" in ignored_entry.reason_codes
    assert excluded.exhaustive

    admitted_descriptor = _descriptor(
        repo,
        ignore_policy=IgnorePolicy(include_gitignored=True),
    )
    admitted = inventory_repository_descriptor(admitted_descriptor)
    admitted_entry = _entry(admitted, "ignored/cache.ts", EntryOrigin.IGNORED.value)
    assert admitted_entry.included
    assert admitted.exhaustive
    assert admitted.inventory_cid != excluded.inventory_cid


def test_binary_oversized_and_archive_decisions(tmp_path: Path) -> None:
    repo = _init_repo(
        tmp_path / "repo",
        {
            "src/large.ts": "x" * 65,
            "src/binary.ts": b"export\x00const x = 1;",
            "release.zip": b"PK\x03\x04payload",
        },
    )
    result = inventory_repository_descriptor(
        _descriptor(repo),
        limits=InventoryLimits(max_parser_bytes=64),
    )
    large = _entry(result, "src/large.ts")
    assert {"source", "oversized"}.issubset(large.classifications)
    assert large.reason_codes == ("parser_size_limit",)
    binary = _entry(result, "src/binary.ts")
    assert {"source", "binary"}.issubset(binary.classifications)
    assert binary.reason_codes == ("binary_not_parser_input",)
    archive = _entry(result, "release.zip")
    assert {"archive", "binary"}.issubset(archive.classifications)


def test_deterministic_order_round_trip_and_incremental_reuse(
    tmp_path: Path,
) -> None:
    repo = _init_repo(
        tmp_path / "repo",
        {
            "z.ts": "export const z = 1;\n",
            "a.ts": "export const a = 1;\n",
            "nested/m.tsx": "export const M = <m />;\n",
        },
    )
    descriptor = _descriptor(repo)
    first = inventory_repository_descriptor(descriptor)
    second = inventory_repository_descriptor(descriptor, previous_index=first)
    assert [item.relative_path for item in first.entries] == [
        "a.ts",
        "nested/m.tsx",
        "z.ts",
    ]
    assert first.inventory_cid == second.inventory_cid
    assert first.to_portable_dict() == second.to_portable_dict()
    assert second.reused_entry_count == len(second.entries)
    replay = RepositoryCorpusIndex.from_dict(second.to_dict())
    assert replay.inventory_cid == first.inventory_cid
    assert replay.to_dict() == second.to_dict()


def test_incremental_reuse_invalidates_changed_blob_only(tmp_path: Path) -> None:
    repo = _init_repo(
        tmp_path / "repo",
        {"a.ts": "export const a = 1;\n", "b.ts": "export const b = 1;\n"},
    )
    first = inventory_repository_descriptor(_descriptor(repo))
    (repo / "a.ts").write_text("export const a = 2;\n", encoding="utf-8")
    _git(repo, "add", "a.ts")
    _git(repo, "commit", "-m", "change a")
    descriptor = _descriptor(repo)
    second = inventory_repository_descriptor(descriptor, previous_index=first)
    # Descriptor binding changed, so cross-tree cache authority is rejected.
    assert second.reused_entry_count == 0
    assert _entry(second, "a.ts").blob_oid != _entry(first, "a.ts").blob_oid
    assert _entry(second, "b.ts").blob_oid == _entry(first, "b.ts").blob_oid


def test_bounded_manifest_reports_every_omission(tmp_path: Path) -> None:
    repo = _init_repo(
        tmp_path / "repo",
        {f"src/file_{index:02d}.ts": f"export const x{index} = {index};\n" for index in range(8)},
    )
    result = inventory_repository_descriptor(
        _descriptor(repo),
        limits=InventoryLimits(max_entries=3),
    )
    summary = result.repositories[0]
    assert result.exhaustive is False
    assert "manifest_entry_bound_exceeded" in result.reason_codes
    assert "manifest_entries_truncated" in result.reason_codes
    assert len(result.entries) == 3
    assert summary.observed_entry_count == 8
    assert summary.emitted_entry_count == 3
    assert summary.omitted_entry_count == 5
    assert [item.relative_path for item in result.entries] == [
        "src/file_00.ts",
        "src/file_01.ts",
        "src/file_02.ts",
    ]


def test_multiple_descriptors_remain_independently_bound(tmp_path: Path) -> None:
    swiss = _init_repo(tmp_path / "swiss", {"src/app.tsx": "export const A=<a/>;\n"})
    accelerator = _init_repo(
        tmp_path / "accelerator", {"module.py": "VALUE = 1\n"}
    )
    forest = build_repository_forest(
        ForestPolicy(
            roots=(
                ForestRootSpec(
                    alias="swissknife",
                    root_path=swiss,
                    authority=RepositoryAuthority(
                        mode=AuthorityMode.READ_ONLY.value
                    ),
                ),
                ForestRootSpec(
                    alias="ipfs_accelerate_py",
                    root_path=accelerator,
                    authority=RepositoryAuthority(
                        mode=AuthorityMode.READ_WRITE.value
                    ),
                ),
            ),
            sole_write_alias="ipfs_accelerate_py",
        )
    )
    result = build_repository_corpus_index(forest)
    assert result.exhaustive
    assert result.forest_id == forest.forest_id
    assert [item.repository_alias for item in result.repositories] == [
        "ipfs_accelerate_py",
        "swissknife",
    ]
    assert {
        (item.repository_alias, item.canonical_path)
        for item in result.entries
    } == {
        ("ipfs_accelerate_py", "ipfs_accelerate_py/module.py"),
        ("swissknife", "swissknife/src/app.tsx"),
    }
    assert len({item.repository_id for item in result.entries}) == 2


def test_deleted_and_renamed_overlay_paths_are_explicit(tmp_path: Path) -> None:
    repo = _init_repo(
        tmp_path / "repo",
        {
            "src/deleted.ts": "export const gone = 1;\n",
            "src/old.ts": "export const renamed = 1;\n",
        },
    )
    os.unlink(repo / "src/deleted.ts")
    os.rename(repo / "src/old.ts", repo / "src/new.ts")
    descriptor = _descriptor(repo)
    result = inventory_repository_descriptor(descriptor)
    deleted = _entry(result, "src/deleted.ts", EntryOrigin.DIRTY_OVERLAY.value)
    renamed_from = _entry(result, "src/old.ts", EntryOrigin.DIRTY_OVERLAY.value)
    renamed_to = _entry(result, "src/new.ts", EntryOrigin.DIRTY_OVERLAY.value)
    assert deleted.object_type == "deleted"
    assert renamed_from.object_type == "deleted"
    assert renamed_to.included
    assert result.exhaustive


def test_policy_exclusions_are_reasoned(tmp_path: Path) -> None:
    repo = _init_repo(
        tmp_path / "repo",
        {
            "src/public.ts": "export {};\n",
            "src/private.ts": "export {};\n",
            "docs/readme.md": "# read\n",
        },
    )
    descriptor = _descriptor(
        repo,
        ignore_policy=IgnorePolicy(
            include_patterns=("src/**",),
            exclude_patterns=("src/private.ts",),
        ),
    )
    result = inventory_repository_descriptor(descriptor)
    assert _entry(result, "src/public.ts").included
    assert "excluded_by_policy" in _entry(
        result, "src/private.ts"
    ).reason_codes
    assert "not_included_by_policy" in _entry(
        result, "docs/readme.md"
    ).reason_codes
    assert result.exhaustive


def test_forged_round_trip_identity_is_rejected(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path / "repo", {"src/app.ts": "export {};\n"})
    result = inventory_repository_descriptor(_descriptor(repo))
    payload = result.to_dict()
    payload["entries"][0]["size"] += 1
    with pytest.raises(RepositoryCorpusIndexError) as excinfo:
        RepositoryCorpusIndex.from_dict(payload)
    assert excinfo.value.reason_code == "inventory_cid_mismatch"


# ---------------------------------------------------------------------------
# VFS-G138 / VFS-G020 packet evidence: vfs/exhaustive-file-inventory@1
# ---------------------------------------------------------------------------


def test_exhaustive_file_inventory_evidence_terms_are_bound() -> None:
    """Prove vfs/exhaustive-file-inventory@1 and packet co-binding."""

    assert EXHAUSTIVE_FILE_INVENTORY_EVIDENCE == "vfs/exhaustive-file-inventory@1"
    assert INCREMENTAL_AST_INDEX_EVIDENCE == "vfs/incremental-ast-index@1"
    assert OBJECTIVE_DOMAIN_EVIDENCE_TERMS == ("vfs/exhaustive-file-inventory@1",)
    assert CORPUS_INDEX_G020_EVIDENCE_TERMS == (
        "vfs/exhaustive-file-inventory@1",
        "vfs/incremental-ast-index@1",
    )
    assert exhaustive_file_inventory_evidence_terms() == OBJECTIVE_DOMAIN_EVIDENCE_TERMS
    assert covered_evidence_terms() == exhaustive_file_inventory_evidence_terms()
    assert packet_evidence_terms() == CORPUS_INDEX_G020_EVIDENCE_TERMS
    # Packet domain scanners stay inventory+AST only; repair is a separate gate.
    assert all_covered_evidence_terms() == packet_evidence_terms()
    assert OBJECTIVE_GOAL_ID == "VFS-G138"
    assert OBJECTIVE_PARENT_GOAL_ID == "VFS-G020"
    assert OBJECTIVE_TASK_ID == "VFS-063"
    assert PACKET_GOAL_IDS == ("VFS-G138", "VFS-G139")
    assert GOAL_PACKET_ID == (
        "goal_packet/corpus_index/ipfs_accelerate_py/26d54d2206f9"
    )
    assert "included and excluded populations publish with reasons" in (
        EXHAUSTIVE_FILE_INVENTORY_INVARIANTS
    )


def test_objective_validation_repair_evidence_term_discoverable() -> None:
    """VFS-G020 objective validation repair: exact-text discovery key present.

    Anchors the synthetic phrase ``objective validation repair`` so objective
    scans re-find the validation gate.  Domain evidence stays separate
    (``vfs/exhaustive-file-inventory@1`` / ``vfs/incremental-ast-index@1``).
    The repair term never enters inventory CIDs or portable entry identity.
    Owned by VFS-G020 via repair task VFS-064.  Conflict domains stay split
    across inventory, language adapters, and incremental persistence.
    """

    assert OBJECTIVE_VALIDATION_REPAIR_EVIDENCE == "objective validation repair"
    assert OBJECTIVE_VALIDATION_REPAIR_TASK_ID == "VFS-064"
    assert OBJECTIVE_PARENT_GOAL_ID == "VFS-G020"
    assert OBJECTIVE_TASK_ID == "VFS-063"
    assert objective_validation_repair_evidence_terms() == (
        "objective validation repair",
    )
    assert parent_objective_evidence_terms() == (
        "vfs/exhaustive-file-inventory@1",
        "vfs/incremental-ast-index@1",
        "objective validation repair",
    )

    # Domain envelope evidence remains inventory-only on this surface.
    assert exhaustive_file_inventory_evidence_terms() == (
        "vfs/exhaustive-file-inventory@1",
    )
    assert covered_evidence_terms() == ("vfs/exhaustive-file-inventory@1",)
    assert "objective validation repair" not in covered_evidence_terms()
    assert "objective validation repair" not in packet_evidence_terms()
    assert "objective validation repair" not in all_covered_evidence_terms()
    assert OBJECTIVE_VALIDATION_REPAIR_EVIDENCE in parent_objective_evidence_terms()
    assert OBJECTIVE_VALIDATION_REPAIR_EVIDENCE in (
        objective_validation_repair_evidence_terms()
    )

    # Adapter surface co-owns the same synthetic gate and packet domain keys.
    from ipfs_accelerate_py.agent_supervisor import program_ast_adapters as adapters

    assert adapters.OBJECTIVE_VALIDATION_REPAIR_EVIDENCE == (
        "objective validation repair"
    )
    assert adapters.OBJECTIVE_VALIDATION_REPAIR_TASK_ID == "VFS-064"
    assert adapters.objective_validation_repair_evidence_terms() == (
        objective_validation_repair_evidence_terms()
    )
    assert adapters.parent_objective_evidence_terms() == (
        parent_objective_evidence_terms()
    )
    assert adapters.packet_evidence_terms() == packet_evidence_terms()
    assert adapters.all_covered_evidence_terms() == all_covered_evidence_terms()
    assert "objective validation repair" not in adapters.covered_evidence_terms()
    assert "inventory, language adapters, and incremental persistence stay conflict-domain split" in (
        OBJECTIVE_VALIDATION_REPAIR_INVARIANTS
    )
    assert adapters.OBJECTIVE_VALIDATION_REPAIR_INVARIANTS == (
        OBJECTIVE_VALIDATION_REPAIR_INVARIANTS
    )


def test_objective_validation_repair_claim_and_identity_separation(
    tmp_path: Path,
) -> None:
    """Repair claims bind VFS-G020 without polluting inventory identity."""

    repo = _init_repo(
        tmp_path / "swissknife",
        {
            "src/services/ipfs.ts": "export async function stat() {}\n",
            "src/components/App.tsx": "export const App = () => <main />;\n",
            "src/legacy.js": "module.exports = { ready: true };\n",
            "src/service.py": "def run(x: int) -> int:\n    return x + 1\n",
            "schemas/mcp.schema.json": '{"type":"object"}\n',
            "docs/vfs.md": "# VFS\n\nOperators MUST inventory every path.\n",
        },
    )
    result = inventory_repository_descriptor(_descriptor(repo))
    assert result.exhaustive is True
    assert inventory_satisfies_exhaustive_file_inventory(result) is True

    bare = prove_objective_validation_repair()
    assert bare["evidence"] == "objective validation repair"
    assert bare["requirement_id"] == "objective validation repair"
    assert bare["goal_id"] == "VFS-G020"
    assert bare["task_id"] == "VFS-064"
    assert bare["domain_task_id"] == "VFS-063"
    assert bare["satisfied"] is True
    assert bare["inventory_cid"] is None
    assert bare["packet_evidence_terms"] == list(CORPUS_INDEX_G020_EVIDENCE_TERMS)
    assert bare["parent_objective_evidence_terms"] == list(
        parent_objective_evidence_terms()
    )
    assert bare["authoritative"] is False
    assert bare["completion_authoritative"] is False
    assert bare["conflict_domains"] == (
        "repository_corpus_index",
        "program_ast_adapters",
        "incremental_persistence",
    )

    claim = prove_objective_validation_repair(result)
    assert claim["evidence"] == "objective validation repair"
    assert claim["satisfied"] is True
    assert claim["exhaustive"] is True
    assert claim["inventory_satisfied"] is True
    assert claim["inventory_cid"] == result.inventory_cid
    # Domain inventory claim stays free of the synthetic repair phrase.
    domain = prove_exhaustive_file_inventory(result)
    assert domain["evidence"] == "vfs/exhaustive-file-inventory@1"
    assert "objective validation repair" not in domain["evidence"]
    assert "objective validation repair" not in domain["evidence_terms"]
    assert "objective validation repair" not in domain["packet_evidence_terms"]
    portable = result.to_portable_dict()
    assert "evidence" not in portable
    assert "objective validation repair" not in str(portable)
    # Repair metadata must not alter content-addressed inventory identity.
    cold = inventory_repository_descriptor(_descriptor(repo))
    assert cold.inventory_cid == result.inventory_cid

    from ipfs_accelerate_py.agent_supervisor.program_ast_adapters import (
        SourceDocument,
        build_program_evidence_index,
        prove_objective_validation_repair as prove_adapter_repair,
    )

    documents = (
        SourceDocument(
            path="src/service.py",
            source="def run(x: int) -> int:\n    return x + 1\n",
        ),
        SourceDocument(
            path="src/services/ipfs.ts",
            source="export async function stat(): Promise<void> {}\n",
        ),
        SourceDocument(
            path="src/components/App.tsx",
            source="export const App = () => <main />;\n",
        ),
        SourceDocument(
            path="src/legacy.js",
            source="module.exports = { ready: true };\n",
        ),
        SourceDocument(
            path="schemas/mcp.schema.json",
            source='{"type":"object"}\n',
        ),
        SourceDocument(
            path="docs/vfs.md",
            source="# VFS\n\nOperators MUST inventory every path.\n",
        ),
    )
    index = build_program_evidence_index(documents)
    adapter_claim = prove_adapter_repair(index)
    assert adapter_claim["evidence"] == "objective validation repair"
    assert adapter_claim["task_id"] == "VFS-064"
    assert adapter_claim["satisfied"] is True
    assert adapter_claim["index_satisfied"] is True
    assert adapter_claim["analysis_index_id"] == index.analysis_index.index_id
    # AST blob identity stays free of the synthetic repair phrase.
    for item in index.results:
        if item.ast_record is not None:
            assert "objective validation repair" not in item.ast_record.blob_identity
            assert "objective validation repair" not in item.source_sha256


def test_inventory_receipt_binds_exhaustive_file_inventory_evidence(
    tmp_path: Path,
) -> None:
    repo = _init_repo(
        tmp_path / "swissknife",
        {
            "src/services/ipfs.ts": "export async function stat() {}\n",
            "src/components/App.tsx": "export const App = () => <main />;\n",
            "schemas/mcp.schema.json": '{"type":"object"}\n',
            "docs/vfs.md": "# VFS\n",
            "vendor/dep.js": "module.exports = {};\n",
        },
    )
    result = inventory_repository_descriptor(_descriptor(repo))

    assert result.exhaustive is True
    assert inventory_satisfies_exhaustive_file_inventory(result) is True
    assert result.satisfies_exhaustive_file_inventory() is True

    payload = result.to_dict()
    assert payload["evidence"] == "vfs/exhaustive-file-inventory@1"
    assert payload["evidence_terms"] == ["vfs/exhaustive-file-inventory@1"]
    assert payload["goal_id"] == "VFS-G138"
    assert payload["parent_goal_id"] == "VFS-G020"
    assert payload["task_id"] == "VFS-063"
    assert payload["goal_packet"] == GOAL_PACKET_ID
    assert payload["packet_goal_ids"] == ["VFS-G138", "VFS-G139"]
    # Evidence metadata must not alter content-addressed inventory identity.
    cold = inventory_repository_descriptor(_descriptor(repo))
    assert cold.inventory_cid == result.inventory_cid
    assert "evidence" not in result.to_portable_dict()

    claim = prove_exhaustive_file_inventory(result)
    assert claim == result.to_evidence_claim()
    assert claim["evidence"] == "vfs/exhaustive-file-inventory@1"
    assert claim["requirement_id"] == "vfs/exhaustive-file-inventory@1"
    assert claim["satisfied"] is True
    assert claim["exhaustive"] is True
    assert claim["inventory_cid"] == result.inventory_cid
    assert claim["included_entry_count"] == len(result.included_entries)
    assert claim["excluded_entry_count"] == len(result.excluded_entries)
    assert claim["included_entry_count"] >= 1
    assert claim["excluded_entry_count"] >= 1
    assert all(
        entry["path"] and entry.get("classifications") is not None
        for entry in claim["populations"]["included"]
    )
    assert all(
        entry["path"] and entry["reason_codes"]
        for entry in claim["populations"]["excluded"]
    )
    assert set(claim["packet_evidence_terms"]) == {
        "vfs/exhaustive-file-inventory@1",
        "vfs/incremental-ast-index@1",
    }
    assert claim["authoritative"] is False
    assert claim["completion_authoritative"] is False


def test_unexplained_skips_and_truncation_block_exhaustive_inventory(
    tmp_path: Path,
) -> None:
    """Truncation and stale descriptors prevent an exhaustive verdict."""

    repo = _init_repo(
        tmp_path / "repo",
        {
            "src/a.ts": "export const a = 1;\n",
            "src/b.ts": "export const b = 2;\n",
            "src/c.ts": "export const c = 3;\n",
        },
    )
    full = inventory_repository_descriptor(_descriptor(repo))
    assert full.exhaustive is True
    assert inventory_satisfies_exhaustive_file_inventory(full) is True

    bounded = inventory_repository_descriptor(
        _descriptor(repo),
        limits=InventoryLimits(max_entries=1, max_manifest_bytes=4096),
    )
    assert bounded.exhaustive is False
    assert bounded.reason_codes
    assert any(
        code in bounded.reason_codes
        for code in (
            "manifest_entry_bound_exceeded",
            "manifest_entries_truncated",
            "manifest_byte_bound_exceeded",
        )
    )
    # Truncated receipts remain structurally valid (populations reasoned) but
    # must not claim a satisfied exhaustive-file-inventory verdict.
    assert inventory_satisfies_exhaustive_file_inventory(bounded) is True
    claim = prove_exhaustive_file_inventory(bounded)
    assert claim["satisfied"] is False
    assert claim["exhaustive"] is False
    assert claim["reason_codes"]

    # Stale descriptors fail closed rather than forging an exhaustive scan.
    clean_descriptor = _descriptor(repo)
    (repo / "src/a.ts").write_text("export const a = 99;\n", encoding="utf-8")
    stale = inventory_repository_descriptor(clean_descriptor)
    assert stale.exhaustive is False
    assert "stale_repository_descriptor" in stale.reason_codes
    assert prove_exhaustive_file_inventory(stale)["satisfied"] is False


def test_incremental_committed_entry_reuse_preserves_inventory_identity(
    tmp_path: Path,
) -> None:
    """Unchanged committed blobs are reused without altering inventory_cid."""

    repo = _init_repo(
        tmp_path / "repo",
        {
            "src/stable.ts": "export const stable = 1;\n",
            "src/other.ts": "export const other = 1;\n",
        },
    )
    first = inventory_repository_descriptor(_descriptor(repo))
    second = inventory_repository_descriptor(
        _descriptor(repo), previous_index=first
    )
    assert second.inventory_cid == first.inventory_cid
    assert second.reused_entry_count == len(first.entries)
    assert second.exhaustive is True
    claim = prove_exhaustive_file_inventory(second)
    assert claim["reused_entry_count"] == second.reused_entry_count
    assert claim["satisfied"] is True
