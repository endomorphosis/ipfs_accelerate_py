"""Program AST adapter evidence and incremental index conformance tests.

Covers ``vfs/incremental-ast-index@1`` (VFS-G139) and packet co-binding with
``vfs/exhaustive-file-inventory@1`` under goal packet
``goal_packet/corpus_index/ipfs_accelerate_py/26d54d2206f9``.

Also proves ``vfs/language-edge-resolution@1`` (VFS-G021 / VFS-G143): every
projected language edge cites a source span and resolver rule; ambiguous and
unsupported constructs stay explicit; name collisions and re-exports never
become forged direct calls.

Language-specific extraction depth lives in the mixed/typescript suites; this
module proves the objective evidence surface, multi-language provenance,
incremental blob reuse, and fail-closed exhaustive verdicts.
"""

from __future__ import annotations

import copy
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.analysis.analysis_ast_index import (
    build_analysis_ast_index,
)
from ipfs_accelerate_py.agent_supervisor.core.conflict_graph import ASTBlobRecord
from ipfs_accelerate_py.agent_supervisor.program_ast_adapters import (
    CORPUS_INDEX_G020_EVIDENCE_TERMS,
    EXHAUSTIVE_FILE_INVENTORY_EVIDENCE,
    GOAL_PACKET_ID,
    INCREMENTAL_AST_INDEX_EVIDENCE,
    INCREMENTAL_AST_INDEX_INVARIANTS,
    LANGUAGE_EDGE_RESOLUTION_CHILD_GOAL_ID,
    LANGUAGE_EDGE_RESOLUTION_EVIDENCE,
    LANGUAGE_EDGE_RESOLUTION_GOAL_ID,
    LANGUAGE_EDGE_RESOLUTION_INVARIANTS,
    LANGUAGE_EDGE_RESOLUTION_TASK_ID,
    OBJECTIVE_DOMAIN_EVIDENCE_TERMS,
    OBJECTIVE_GOAL_ID,
    OBJECTIVE_PARENT_GOAL_ID,
    OBJECTIVE_TASK_ID,
    PACKET_GOAL_IDS,
    PROVENANCE_LANGUAGES,
    InventoryProgramEvidenceReceipt,
    ProgramEvidenceIndex,
    SourceDocument,
    adapt_program_source,
    all_covered_evidence_terms,
    build_incremental_ast_index,
    build_inventory_program_evidence_index,
    build_language_edge_program_graph,
    build_program_evidence_index,
    covered_evidence_terms,
    detect_program_language,
    incremental_ast_index_evidence_terms,
    index_satisfies_incremental_ast_index,
    language_edge_candidate_cites_span_and_rule,
    language_edge_resolution_evidence_terms,
    language_edge_resolution_satisfies,
    packet_evidence_terms,
    project_language_edge_candidates,
    project_language_edge_candidates_from_index,
    prove_incremental_ast_index,
    prove_language_edge_resolution,
)
from ipfs_accelerate_py.agent_supervisor.program_graph import (
    LANGUAGE_EDGE_RESOLUTION_EVIDENCE as GRAPH_LANGUAGE_EDGE_EVIDENCE,
    ProgramEdgeKind,
    ResolverStatus,
    edge_cites_source_span_and_resolver_rule,
    graph_satisfies_language_edge_resolution,
    language_edge_forged_direct_call_reason,
    language_edge_resolution_evidence_terms as graph_language_edge_evidence_terms,
    make_edge,
    make_node,
    build_program_graph,
    prove_language_edge_resolution as prove_graph_language_edge_resolution,
    ProgramNodeKind,
)
from ipfs_accelerate_py.agent_supervisor.repository_corpus_index import (
    InventoryLimits,
    inventory_repository_descriptor,
)
from ipfs_accelerate_py.agent_supervisor.repository_forest import (
    AuthorityMode,
    RepositoryAuthority,
    build_repository_descriptor,
)


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        text=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def _init_repo(path: Path, files: dict[str, str]) -> Path:
    path.mkdir(parents=True)
    _git(path, "init")
    _git(path, "checkout", "-b", "main")
    _git(path, "config", "user.name", "AST Inventory Test")
    _git(path, "config", "user.email", "ast-inventory@example.invalid")
    for relative_path, source in files.items():
        target = path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "seed mixed corpus")
    return path


def _descriptor(repo: Path):
    return build_repository_descriptor(
        repo,
        alias="swissknife",
        authority=RepositoryAuthority(mode=AuthorityMode.READ_ONLY.value),
    )


def _documents(repo: Path, paths: tuple[str, ...]) -> tuple[SourceDocument, ...]:
    return tuple(
        SourceDocument(path, (repo / path).read_text(encoding="utf-8"))
        for path in paths
    )


def test_incremental_ast_index_evidence_terms_are_bound() -> None:
    """Prove vfs/incremental-ast-index@1 and packet co-binding."""

    assert INCREMENTAL_AST_INDEX_EVIDENCE == "vfs/incremental-ast-index@1"
    assert EXHAUSTIVE_FILE_INVENTORY_EVIDENCE == "vfs/exhaustive-file-inventory@1"
    assert LANGUAGE_EDGE_RESOLUTION_EVIDENCE == "vfs/language-edge-resolution@1"
    assert OBJECTIVE_DOMAIN_EVIDENCE_TERMS == ("vfs/incremental-ast-index@1",)
    assert CORPUS_INDEX_G020_EVIDENCE_TERMS == (
        "vfs/exhaustive-file-inventory@1",
        "vfs/incremental-ast-index@1",
    )
    assert incremental_ast_index_evidence_terms() == OBJECTIVE_DOMAIN_EVIDENCE_TERMS
    assert covered_evidence_terms() == (
        "vfs/incremental-ast-index@1",
        "vfs/language-edge-resolution@1",
    )
    assert packet_evidence_terms() == CORPUS_INDEX_G020_EVIDENCE_TERMS
    assert all_covered_evidence_terms() == (
        "vfs/exhaustive-file-inventory@1",
        "vfs/incremental-ast-index@1",
        "vfs/language-edge-resolution@1",
    )
    assert language_edge_resolution_evidence_terms() == (
        "vfs/language-edge-resolution@1",
    )
    assert OBJECTIVE_GOAL_ID == "VFS-G139"
    assert OBJECTIVE_PARENT_GOAL_ID == "VFS-G020"
    assert OBJECTIVE_TASK_ID == "VFS-063"
    assert LANGUAGE_EDGE_RESOLUTION_GOAL_ID == "VFS-G021"
    assert LANGUAGE_EDGE_RESOLUTION_CHILD_GOAL_ID == "VFS-G143"
    assert LANGUAGE_EDGE_RESOLUTION_TASK_ID == "VFS-069"
    assert PACKET_GOAL_IDS == ("VFS-G138", "VFS-G139")
    assert GOAL_PACKET_ID == (
        "goal_packet/corpus_index/ipfs_accelerate_py/26d54d2206f9"
    )
    assert PROVENANCE_LANGUAGES == frozenset(
        {
            "python",
            "javascript",
            "jsx",
            "typescript",
            "tsx",
            "json",
            "json-schema",
            "mcp-manifest",
            "markdown",
        }
    )
    assert "unchanged blobs are reused from the previous snapshot" in (
        INCREMENTAL_AST_INDEX_INVARIANTS
    )
    assert "every projected edge cites a source span and resolver rule" in (
        LANGUAGE_EDGE_RESOLUTION_INVARIANTS
    )


def test_mixed_language_snapshot_has_provenance_and_binds_evidence() -> None:
    """TS/TSX/JS/Python/JSON/Markdown inputs carry content-bound provenance."""

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
            source='{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object"}\n',
        ),
        SourceDocument(
            path="docs/vfs.md",
            source="# VFS\n\nOperators MUST inventory every path.\n",
        ),
    )
    index = build_program_evidence_index(documents)
    assert build_incremental_ast_index is build_program_evidence_index
    assert index_satisfies_incremental_ast_index(index) is True
    assert index.satisfies_incremental_ast_index() is True
    assert index.exhaustive is True
    assert index.reason_codes == ()
    assert index.truncated is False
    assert len(index.results) == 6
    assert {item.path for item in index.results} == {item.path for item in documents}

    for result in index.results:
        assert result.language in PROVENANCE_LANGUAGES
        assert result.source_sha256.startswith("sha256:")
        assert result.blob_identity
        assert result.status in {"success", "partial"}
        assert result.ast_record is not None
        assert result.ast_record.source_sha256 == result.source_sha256
        assert result.ast_record.blob_identity == result.blob_identity
        detected = detect_program_language(path=result.path)
        # JSON Schema / MCP manifests stay on the JSON adapter path.
        if detected == "json":
            assert result.language in {"json", "json-schema", "mcp-manifest"}
        else:
            assert result.language == detected

    payload = index.to_dict()
    assert payload["evidence"] == "vfs/incremental-ast-index@1"
    assert payload["evidence_terms"] == ["vfs/incremental-ast-index@1"]
    assert payload["goal_id"] == "VFS-G139"
    assert payload["parent_goal_id"] == "VFS-G020"
    assert payload["task_id"] == "VFS-063"
    assert payload["goal_packet"] == GOAL_PACKET_ID
    assert payload["packet_goal_ids"] == ["VFS-G138", "VFS-G139"]
    assert payload["exhaustive"] is True
    assert payload["reused_result_count"] == 0

    claim = prove_incremental_ast_index(index)
    assert claim == index.to_evidence_claim()
    assert claim["evidence"] == "vfs/incremental-ast-index@1"
    assert claim["requirement_id"] == "vfs/incremental-ast-index@1"
    assert claim["satisfied"] is True
    assert claim["exhaustive"] is True
    assert claim["result_count"] == 6
    assert {"python", "typescript", "tsx", "javascript", "markdown"}.issubset(
        set(claim["languages"])
    )
    assert any(
        language in {"json", "json-schema", "mcp-manifest"}
        for language in claim["languages"]
    )
    assert set(claim["packet_evidence_terms"]) == {
        "vfs/exhaustive-file-inventory@1",
        "vfs/incremental-ast-index@1",
    }
    assert claim["authoritative"] is False
    assert claim["completion_authoritative"] is False
    assert claim["analysis_index_id"] == index.analysis_index.index_id


def test_unchanged_blobs_are_reused_and_changed_blobs_invalidate() -> None:
    """Incremental index reuses exact blobs and rebuilds only changed paths."""

    docs = {
        "src/stable.py": "def stable() -> int:\n    return 1\n",
        "src/mutable.ts": "export const value = 1;\n",
    }
    first = build_program_evidence_index(
        tuple(SourceDocument(path=path, source=source) for path, source in docs.items())
    )
    assert first.exhaustive is True
    assert first.reused_result_count == 0

    second = build_program_evidence_index(
        tuple(SourceDocument(path=path, source=source) for path, source in docs.items()),
        previous=first,
    )
    assert second.reused_result_count == 2
    assert all(item.reused for item in second.results)
    assert second.analysis_index.stats.reused_blob_count >= 1
    assert second.exhaustive is True
    warm_claim = prove_incremental_ast_index(second)
    assert warm_claim["reused_result_count"] == 2
    assert warm_claim["satisfied"] is True

    docs["src/mutable.ts"] = "export const value = 2;\n"
    third = build_program_evidence_index(
        tuple(SourceDocument(path=path, source=source) for path, source in docs.items()),
        previous=second,
    )
    by_path = {item.path: item for item in third.results}
    assert by_path["src/stable.py"].reused is True
    assert by_path["src/mutable.ts"].reused is False
    assert by_path["src/mutable.ts"].source_sha256 != by_path["src/stable.py"].source_sha256
    assert third.reused_result_count == 1
    assert third.exhaustive is True


def test_parser_failures_and_truncation_prevent_exhaustive_verdict() -> None:
    """Malformed inputs and hard bounds block an exhaustive AST verdict."""

    malformed = build_program_evidence_index(
        (
            SourceDocument(path="src/ok.py", source="x = 1\n"),
            SourceDocument(path="src/bad.py", source="def broken(\n"),
        )
    )
    assert malformed.malformed_results
    assert "parser_failures" in malformed.reason_codes
    assert malformed.exhaustive is False
    assert index_satisfies_incremental_ast_index(malformed) is True
    claim = prove_incremental_ast_index(malformed)
    assert claim["satisfied"] is True
    assert claim["exhaustive"] is False
    assert claim["malformed_count"] >= 1
    assert "parser_failures" in claim["reason_codes"]

    oversized = adapt_program_source(
        "x = 1\n",
        path="src/huge.py",
        max_source_bytes=1,
    )
    assert oversized.status == "unsupported"
    assert any(
        item.code == "source_size_bound_exceeded" for item in oversized.diagnostics
    )

    # JSON emits one fact per member; bound to force fact_bound_exceeded.
    rich_json = (
        "{"
        + ",".join(f'"field_{index}":{index}' for index in range(20))
        + "}\n"
    )
    truncated = build_program_evidence_index(
        (SourceDocument(path="src/facts.json", source=rich_json),),
        max_facts=2,
    )
    assert truncated.truncated is True
    assert "truncation" in truncated.reason_codes
    assert truncated.exhaustive is False
    assert any(item.status == "partial" for item in truncated.results)
    assert any(
        any(diag.code == "fact_bound_exceeded" for diag in item.diagnostics)
        for item in truncated.results
    )
    trunc_claim = prove_incremental_ast_index(truncated)
    assert trunc_claim["exhaustive"] is False
    assert trunc_claim["truncated"] is True
    assert "truncation" in trunc_claim["reason_codes"]


def test_unsupported_language_is_accounted_without_forging_records() -> None:
    """Unsupported paths remain explicit and do not invent AST records."""

    index = build_program_evidence_index(
        (
            SourceDocument(path="src/ok.py", source="x = 1\n"),
            SourceDocument(path="src/native.rs", source="fn main() {}\n"),
        )
    )
    unsupported = index.unsupported_results
    assert len(unsupported) == 1
    assert unsupported[0].path == "src/native.rs"
    assert unsupported[0].ast_record is None
    assert unsupported[0].status == "unsupported"
    # Unsupported is accounted; without parser failure/truncation the scan
    # remains exhaustive for the admitted snapshot.
    assert index.exhaustive is True
    assert index_satisfies_incremental_ast_index(index) is True
    claim = prove_incremental_ast_index(index)
    assert claim["unsupported_count"] == 1
    assert claim["satisfied"] is True
    assert claim["exhaustive"] is True


def test_packet_evidence_terms_align_with_corpus_inventory_surface() -> None:
    """Packet discovery keys stay identical across inventory and adapter modules."""

    from ipfs_accelerate_py.agent_supervisor import repository_corpus_index as corpus

    assert packet_evidence_terms() == corpus.packet_evidence_terms()
    # Packet pair stays inventory + AST; language-edge is adapters-owned sibling.
    assert set(packet_evidence_terms()).issubset(set(all_covered_evidence_terms()))
    assert set(corpus.all_covered_evidence_terms()).issubset(
        set(all_covered_evidence_terms())
    )
    assert EXHAUSTIVE_FILE_INVENTORY_EVIDENCE == corpus.EXHAUSTIVE_FILE_INVENTORY_EVIDENCE
    assert INCREMENTAL_AST_INDEX_EVIDENCE == corpus.INCREMENTAL_AST_INDEX_EVIDENCE
    assert GOAL_PACKET_ID == corpus.GOAL_PACKET_ID
    assert set(PACKET_GOAL_IDS) == set(corpus.PACKET_GOAL_IDS)
    assert OBJECTIVE_PARENT_GOAL_ID == corpus.OBJECTIVE_PARENT_GOAL_ID
    assert OBJECTIVE_TASK_ID == corpus.OBJECTIVE_TASK_ID
    # Domain ownership: inventory owns G138; adapters own G139 + G021 language edges.
    assert covered_evidence_terms() == (
        "vfs/incremental-ast-index@1",
        "vfs/language-edge-resolution@1",
    )
    assert corpus.covered_evidence_terms() == ("vfs/exhaustive-file-inventory@1",)
    assert set(packet_evidence_terms()) == {
        "vfs/exhaustive-file-inventory@1",
        "vfs/incremental-ast-index@1",
    }
    assert LANGUAGE_EDGE_RESOLUTION_EVIDENCE not in packet_evidence_terms()
    assert LANGUAGE_EDGE_RESOLUTION_EVIDENCE in all_covered_evidence_terms()
    assert LANGUAGE_EDGE_RESOLUTION_EVIDENCE not in corpus.covered_evidence_terms()


def test_cross_path_cache_reuses_only_canonical_path_independent_records() -> None:
    malformed_source = "def broken(:\n"
    malformed_old = build_program_evidence_index(
        (SourceDocument("src/old.py", malformed_source),)
    )
    malformed_warm = build_program_evidence_index(
        (SourceDocument("src/new.py", malformed_source),),
        previous=malformed_old,
    )
    malformed_cold = build_program_evidence_index(
        (SourceDocument("src/new.py", malformed_source),)
    )

    assert malformed_warm.results[0].reused is True
    assert "new.py" in malformed_warm.results[0].diagnostics[0].message
    assert replace(malformed_warm.results[0], reused=False) == (
        malformed_cold.results[0]
    )
    assert malformed_warm.analysis_index.index_id == (
        malformed_cold.analysis_index.index_id
    )

    typescript_source = "export const value: number = 1;\n"
    typescript_old = build_program_evidence_index(
        (
            SourceDocument(
                "src/old.ts",
                typescript_source,
                generated=False,
            ),
        )
    )
    typescript_warm = build_program_evidence_index(
        (
            SourceDocument(
                "generated/new.ts",
                typescript_source,
                generated=True,
            ),
        ),
        previous=typescript_old,
    )
    typescript_cold = build_program_evidence_index(
        (
            SourceDocument(
                "generated/new.ts",
                typescript_source,
                generated=True,
            ),
        )
    )

    assert typescript_warm.results[0].reused is True
    assert typescript_warm.results[0].facts
    assert all(item.generated for item in typescript_warm.results[0].facts)
    assert replace(typescript_warm.results[0], reused=False) == (
        typescript_cold.results[0]
    )
    assert typescript_warm.analysis_index.index_id == (
        typescript_cold.analysis_index.index_id
    )


def test_inventory_bound_index_covers_supported_inputs_and_reuses_same_inventory(
    tmp_path: Path,
) -> None:
    sources = {
        "src/service.ts": "export function stat(path: string) { return path; }\n",
        "src/view.tsx": "export const View = () => <main />;\n",
        "src/client.js": "export const read = (path) => path;\n",
        "src/worker.py": "def pin(cid: str):\n    return cid\n",
        "schemas/vfs.json": '{"type":"object","properties":{"cid":{"type":"string"}}}\n',
        "docs/vfs.md": "# VFS\n\nClients MUST call `pin`.\n",
    }
    repo = _init_repo(tmp_path / "swissknife", sources)
    inventory = inventory_repository_descriptor(_descriptor(repo))
    first = build_inventory_program_evidence_index(
        inventory,
        _documents(repo, tuple(sources)),
    )

    assert isinstance(first, InventoryProgramEvidenceReceipt)
    assert inventory.exhaustive
    assert first.exhaustive
    assert first.reason_codes == ()
    assert first.program_index.exhaustive
    assert len(first.results) == len(sources)
    assert len(first.analysis_index.path_records) == len(sources)
    assert "evidence" not in inventory.to_portable_dict()

    payload = first.to_dict()
    assert payload["receipt_cid"] == first.receipt_cid
    assert first.to_portable_dict()["receipt_cid"] == first.receipt_cid
    assert payload["evidence"] == "vfs/incremental-ast-index@1"
    assert payload["evidence_terms"] == ["vfs/incremental-ast-index@1"]
    assert payload["packet_evidence_terms"] == [
        "vfs/exhaustive-file-inventory@1",
        "vfs/incremental-ast-index@1",
    ]
    assert payload["inventory_cid"] == inventory.inventory_cid
    assert payload["coverage"]["language_counts"] == {
        "javascript": 1,
        "json": 1,
        "markdown": 1,
        "python": 1,
        "tsx": 1,
        "typescript": 1,
    }
    entries = {
        item.canonical_path: item
        for item in inventory.included_entries
    }
    assert all(
        result.blob_identity == entries[result.path].blob_oid
        and result.source_sha256 == "sha256:" + entries[result.path].content_sha256
        for result in first.results
    )

    warm = build_inventory_program_evidence_index(
        inventory,
        _documents(repo, tuple(reversed(tuple(sources)))),
        previous=first,
    )
    assert warm.exhaustive
    assert warm.reused_result_count == len(sources)
    assert warm.analysis_index.stats.reused_blob_count == len(sources)
    assert warm.analysis_index.index_id == first.analysis_index.index_id
    assert all(result.reused for result in warm.results)
    assert warm.receipt_cid == first.receipt_cid

    (repo / "src/service.ts").write_text(
        "export function stat(path: string) { return path.toUpperCase(); }\n",
        encoding="utf-8",
    )
    _git(repo, "mv", "src/client.js", "src/client-renamed.js")
    _git(repo, "add", "src/service.ts")
    _git(repo, "commit", "-m", "change one blob and rename another")
    changed_paths = tuple(
        "src/client-renamed.js" if path == "src/client.js" else path
        for path in sources
    )
    changed_inventory = inventory_repository_descriptor(_descriptor(repo))
    with pytest.raises(ValueError, match="does not match inventory CID"):
        build_inventory_program_evidence_index(
            changed_inventory,
            _documents(repo, changed_paths),
            previous=warm,
        )
    changed = build_inventory_program_evidence_index(
        changed_inventory,
        _documents(repo, changed_paths),
    )

    assert changed.exhaustive
    assert changed.inventory_cid != first.inventory_cid
    assert changed.reused_result_count == 0


def test_inventory_bound_index_rejects_language_spoofing(
    tmp_path: Path,
) -> None:
    source = "def broken(:\n"
    repo = _init_repo(tmp_path / "repo", {"src/broken.py": source})
    inventory = inventory_repository_descriptor(_descriptor(repo))

    with pytest.raises(ValueError, match="language conflicts with inventory path"):
        build_inventory_program_evidence_index(
            inventory,
            (SourceDocument("src/broken.py", source, language="markdown"),),
        )

    result = build_inventory_program_evidence_index(
        inventory,
        (SourceDocument("src/broken.py", source, language="py"),),
    )
    assert result.results[0].language == "python"
    assert result.results[0].status == "malformed"
    assert result.exhaustive is False
    assert result.reason_codes == ("parser_failures",)


def test_inventory_bound_index_uses_inventory_generated_classification(
    tmp_path: Path,
) -> None:
    sources = {
        "src/main.ts": "export const sourceValue = 1;\n",
        "generated/client.ts": "export const generatedValue = 2;\n",
    }
    repo = _init_repo(tmp_path / "repo", sources)
    inventory = inventory_repository_descriptor(_descriptor(repo))

    with pytest.raises(
        ValueError, match="generated classification conflicts with inventory"
    ):
        build_inventory_program_evidence_index(
            inventory,
            (
                SourceDocument(
                    "src/main.ts",
                    sources["src/main.ts"],
                    generated=True,
                ),
                SourceDocument(
                    "generated/client.ts",
                    sources["generated/client.ts"],
                ),
            ),
        )

    result = build_inventory_program_evidence_index(
        inventory,
        _documents(repo, tuple(sources)),
    )
    by_path = {item.path: item for item in result.results}
    ordinary = by_path["swissknife/src/main.ts"]
    generated = by_path["swissknife/generated/client.ts"]
    assert ordinary.generated is False
    assert all(not item.generated for item in ordinary.facts)
    assert generated.generated is True
    assert generated.facts
    assert all(item.generated for item in generated.facts)


def test_inventory_bound_reuse_rejects_bare_program_indexes(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo", {"src/main.py": "VALUE = 1\n"})
    inventory = inventory_repository_descriptor(_descriptor(repo))
    first = build_inventory_program_evidence_index(
        inventory,
        _documents(repo, ("src/main.py",)),
    )

    with pytest.raises(
        TypeError, match="verified InventoryProgramEvidenceReceipt"
    ):
        build_inventory_program_evidence_index(
            inventory,
            _documents(repo, ("src/main.py",)),
            previous=first.program_index,
        )


def test_same_inventory_receipt_cannot_poison_fresh_canonical_records(
    tmp_path: Path,
) -> None:
    sources = {
        "src/broken.py": "def broken(:\n",
        "src/service.ts": "export function read(path: string) { return path; }\n",
    }
    repo = _init_repo(tmp_path / "repo", sources)
    inventory = inventory_repository_descriptor(_descriptor(repo))
    cold = build_inventory_program_evidence_index(
        inventory,
        _documents(repo, tuple(sources)),
    )

    forged_results = []
    forged_path_records = []
    for result in cold.results:
        forged_record = ASTBlobRecord(
            blob_identity=result.blob_identity,
            source_sha256=result.source_sha256,
            language=result.language,
        )
        forged_result = replace(
            result,
            status="success",
            ast_record=forged_record,
            facts=(),
            diagnostics=(),
            reused=False,
        )
        forged_results.append(forged_result)
        forged_path_records.append((result.path, forged_record))
    forged_index = ProgramEvidenceIndex(
        analysis_index=build_analysis_ast_index(forged_path_records),
        results=tuple(forged_results),
    )
    forged_receipt = InventoryProgramEvidenceReceipt(
        program_index=forged_index,
        inventory_cid=inventory.inventory_cid,
        inventory_exhaustive=inventory.exhaustive,
        expected_paths=cold.expected_paths,
    )
    assert forged_receipt.exhaustive is True

    rebuilt = build_inventory_program_evidence_index(
        inventory,
        _documents(repo, tuple(sources)),
        previous=forged_receipt,
    )
    cold_by_path = {item.path: item for item in cold.results}
    assert rebuilt.exhaustive is False
    assert rebuilt.reason_codes == ("parser_failures",)
    assert all(item.reused is False for item in rebuilt.results)
    assert all(
        replace(item, reused=False) == cold_by_path[item.path]
        for item in rebuilt.results
    )
    assert rebuilt.analysis_index.index_id == cold.analysis_index.index_id


def test_inventory_receipt_cid_round_trip_and_structural_validation(
    tmp_path: Path,
) -> None:
    repo = _init_repo(
        tmp_path / "repo",
        {"src/main.py": "def read(path: str) -> str:\n    return path\n"},
    )
    inventory = inventory_repository_descriptor(_descriptor(repo))
    receipt = build_inventory_program_evidence_index(
        inventory,
        _documents(repo, ("src/main.py",)),
    )

    restored = InventoryProgramEvidenceReceipt.from_dict(receipt.to_dict())
    portable_restored = InventoryProgramEvidenceReceipt.from_dict(
        receipt.to_portable_dict()
    )
    assert restored == receipt
    assert restored.receipt_cid == receipt.receipt_cid
    assert portable_restored.receipt_cid == receipt.receipt_cid
    assert portable_restored.to_portable_dict() == receipt.to_portable_dict()

    tampered = copy.deepcopy(receipt.to_dict())
    tampered["program_index"]["results"][0]["generated"] = True
    with pytest.raises(ValueError, match="receipt CID does not match payload"):
        InventoryProgramEvidenceReceipt.from_dict(tampered)

    forged_fact = copy.deepcopy(receipt.to_dict())
    forged_fact["program_index"]["results"][0]["facts"][0]["name"] = "forged"
    with pytest.raises(ValueError, match="fact identity does not match"):
        InventoryProgramEvidenceReceipt.from_dict(forged_fact)

    invalid_cid = copy.deepcopy(receipt.to_dict())
    invalid_cid["receipt_cid"] = "fake-not-a-cid"
    with pytest.raises(ValueError, match="CID"):
        InventoryProgramEvidenceReceipt.from_dict(invalid_cid)

    with pytest.raises(ValueError, match="CID"):
        InventoryProgramEvidenceReceipt(
            program_index=receipt.program_index,
            inventory_cid="fake-not-a-cid",
            inventory_exhaustive=True,
            expected_paths=receipt.expected_paths,
        )

    empty_index = ProgramEvidenceIndex(
        analysis_index=build_analysis_ast_index(()),
        results=receipt.results,
    )
    with pytest.raises(ValueError, match="AST paths do not match"):
        InventoryProgramEvidenceReceipt(
            program_index=empty_index,
            inventory_cid=inventory.inventory_cid,
            inventory_exhaustive=True,
            expected_paths=receipt.expected_paths,
        )

    missing_ast_result = replace(receipt.results[0], ast_record=None)
    missing_ast_index = ProgramEvidenceIndex(
        analysis_index=build_analysis_ast_index(()),
        results=(missing_ast_result,),
    )
    with pytest.raises(ValueError, match="require canonical AST records"):
        InventoryProgramEvidenceReceipt(
            program_index=missing_ast_index,
            inventory_cid=inventory.inventory_cid,
            inventory_exhaustive=True,
            expected_paths=receipt.expected_paths,
        )


def test_inventory_bound_index_rejects_forged_source_or_blob_provenance(
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path / "repo", {"src/main.py": "VALUE = 1\n"})
    inventory = inventory_repository_descriptor(_descriptor(repo))

    with pytest.raises(ValueError, match="content does not match inventory provenance"):
        build_inventory_program_evidence_index(
            inventory,
            (SourceDocument("src/main.py", "VALUE = 2\n"),),
        )
    with pytest.raises(ValueError, match="blob identity does not match"):
        build_inventory_program_evidence_index(
            inventory,
            (
                SourceDocument(
                    "src/main.py",
                    "VALUE = 1\n",
                    blob_identity="forged-blob",
                ),
            ),
        )


def test_inventory_bound_index_fails_closed_for_coverage_and_parse_gaps(
    tmp_path: Path,
) -> None:
    sources = {
        "src/broken.py": "def broken(:\n",
        "src/unsupported.go": "package main\nfunc main() {}\n",
        "schemas/required.json": '{"type":"object"}\n',
        "docs/rules.md": (
            "# Rules\n\nClients MUST call `one` and MUST call `two`.\n"
        ),
    }
    repo = _init_repo(tmp_path / "repo", sources)
    inventory = inventory_repository_descriptor(_descriptor(repo))
    result = build_inventory_program_evidence_index(
        inventory,
        _documents(
            repo,
            ("src/broken.py", "src/unsupported.go", "docs/rules.md"),
        ),
        max_facts=1,
    )

    assert inventory.exhaustive
    assert result.exhaustive is False
    assert set(result.reason_codes) == {
        "inventory_inputs_missing",
        "parser_failures",
        "truncation",
        "unsupported_parser_input",
    }
    payload = result.to_dict()
    assert payload["coverage"]["expected_path_count"] == 4
    assert payload["coverage"]["adapted_path_count"] == 3
    assert payload["coverage"]["missing_paths"] == [
        "swissknife/schemas/required.json"
    ]
    assert payload["coverage"]["status_counts"] == {
        "malformed": 1,
        "partial": 1,
        "unsupported": 1,
    }


def test_truncated_inventory_blocks_inventory_bound_exhaustive_verdict(
    tmp_path: Path,
) -> None:
    sources = {
        f"src/file_{index}.py": f"VALUE_{index} = {index}\n"
        for index in range(4)
    }
    repo = _init_repo(tmp_path / "repo", sources)
    inventory = inventory_repository_descriptor(
        _descriptor(repo),
        limits=InventoryLimits(max_entries=2),
    )
    emitted_paths = tuple(
        item.relative_path for item in inventory.included_entries
    )
    result = build_inventory_program_evidence_index(
        inventory,
        _documents(repo, emitted_paths),
    )

    assert inventory.exhaustive is False
    assert "manifest_entries_truncated" in inventory.reason_codes
    assert result.program_index.exhaustive is True
    assert result.exhaustive is False
    assert result.reason_codes == ("inventory_not_exhaustive",)


def test_language_edge_resolution_evidence_terms_are_bound() -> None:
    """Prove vfs/language-edge-resolution@1 is discoverable on adapters + graph."""

    assert LANGUAGE_EDGE_RESOLUTION_EVIDENCE == "vfs/language-edge-resolution@1"
    assert GRAPH_LANGUAGE_EDGE_EVIDENCE == "vfs/language-edge-resolution@1"
    assert language_edge_resolution_evidence_terms() == (
        "vfs/language-edge-resolution@1",
    )
    assert graph_language_edge_evidence_terms() == (
        "vfs/language-edge-resolution@1",
    )
    assert LANGUAGE_EDGE_RESOLUTION_EVIDENCE in covered_evidence_terms()
    assert LANGUAGE_EDGE_RESOLUTION_EVIDENCE in all_covered_evidence_terms()
    assert LANGUAGE_EDGE_RESOLUTION_GOAL_ID == "VFS-G021"
    assert LANGUAGE_EDGE_RESOLUTION_CHILD_GOAL_ID == "VFS-G143"
    assert LANGUAGE_EDGE_RESOLUTION_TASK_ID == "VFS-069"
    claim = prove_language_edge_resolution()
    assert claim["evidence"] == "vfs/language-edge-resolution@1"
    assert claim["satisfied"] is True
    assert claim["forges_direct_calls"] is False
    graph_claim = prove_graph_language_edge_resolution()
    assert graph_claim["evidence"] == "vfs/language-edge-resolution@1"
    assert graph_claim["satisfied"] is True


def test_language_edge_candidates_cite_span_and_resolver_rule() -> None:
    """Every projected language edge cites a source span and resolver rule."""

    python_source = (
        "import os\n"
        "from pkg import helper as h\n"
        "from star import *\n"
        "\n"
        "def entry(x):\n"
        "    return h(x) + os.getcwd()\n"
        "\n"
        "entry(1)\n"
        "setattr(entry, 'patched', True)\n"
    )
    ts_source = (
        "import { read } from './fs';\n"
        "export { read as load } from './fs';\n"
        "export function run(cb: () => void) {\n"
        "  const dyn = import('./lazy');\n"
        "  read();\n"
        "  cb();\n"
        "}\n"
    )
    index = build_program_evidence_index(
        (
            SourceDocument(path="src/service.py", source=python_source),
            SourceDocument(path="src/service.ts", source=ts_source),
        )
    )
    candidates = project_language_edge_candidates_from_index(index)
    assert candidates
    assert language_edge_resolution_satisfies(candidates) is True
    for candidate in candidates:
        assert language_edge_candidate_cites_span_and_rule(candidate)
        assert candidate.resolver_rule.startswith("rule:")
        assert candidate.span.line_start > 0 or candidate.span.line_end > 0
        assert candidate.status in {
            "resolved_static",
            "candidate",
            "ambiguous",
            "external",
            "unknown",
            "unsupported",
            "unresolved",
        }

    claim = prove_language_edge_resolution(index)
    assert claim["evidence"] == "vfs/language-edge-resolution@1"
    assert claim["satisfied"] is True
    assert claim["candidate_count"] == len(candidates)
    assert claim["missing_rule_count"] == 0
    assert claim["missing_span_count"] == 0
    assert claim["direct_call_count"] == 0
    assert claim["forges_direct_calls"] is False


def test_language_edge_ambiguous_and_unsupported_remain_explicit() -> None:
    """Ambiguous and unsupported constructs stay explicit frontier statuses."""

    source = (
        "def run(fn):\n"
        "    return fn()\n"
        "\n"
        "run(lambda: setattr(run, 'x', 1))\n"
        "value = getattr(run, 'x', None)\n"
    )
    result = adapt_program_source(source, path="src/dyn.py", language="python")
    candidates = project_language_edge_candidates(result)
    assert candidates
    kinds = {item.kind for item in candidates}
    assert "call" in kinds or "monkey_patch" in kinds or "callback" in kinds
    for item in candidates:
        if item.kind in {
            "monkey_patch",
            "callback",
            "dynamic_import",
            "decorator",
            "registration",
            "unsupported_node",
        }:
            assert item.status in {"ambiguous", "unsupported", "unknown"}
            assert item.allows_direct_call is False
        if item.reason.startswith("dynamic") or item.status in {
            "ambiguous",
            "unsupported",
            "unknown",
        }:
            assert item.allows_direct_call is False

    # Unsupported language remains accounted without inventing call edges.
    unsupported = adapt_program_source(
        "fn main() {}", path="src/main.rs", language="rust"
    )
    assert unsupported.status == "unsupported"
    assert project_language_edge_candidates(unsupported) == ()
    claim = prove_language_edge_resolution(results=(result, unsupported))
    assert claim["satisfied"] is True
    assert claim["direct_call_count"] == 0


def test_language_edge_name_collisions_and_reexports_cannot_forge_direct_calls() -> None:
    """Adversarial name collisions and re-exports cannot become forged direct calls."""

    # ECMAScript: same-name multi-definition collision + re-export surface.
    collision_source = (
        "function shared() { return 1; }\n"
        "function shared() { return 2; }\n"
        "export { shared as alias } from './other';\n"
        "shared();\n"
    )
    result = adapt_program_source(
        collision_source, path="src/collide.ts", language="typescript"
    )
    assert any(
        diagnostic.code == "ecmascript_name_collision"
        for diagnostic in result.diagnostics
    )
    candidates = project_language_edge_candidates(result)
    assert candidates
    for item in candidates:
        assert item.allows_direct_call is False
        if item.reason in {"same_name_collision", "re_export_not_direct_call"}:
            assert item.status == "ambiguous"
            assert "collision" in item.resolver_rule or "re_export" in item.resolver_rule
    assert any(item.kind == "re_export" for item in candidates) or any(
        item.reason == "re_export_not_direct_call" for item in candidates
    )
    claim = prove_language_edge_resolution(results=(result,))
    assert claim["satisfied"] is True
    assert claim["direct_call_count"] == 0
    assert claim["forged_direct_call_blocked_count"] >= 1

    graph = build_language_edge_program_graph((result,))
    assert graph_satisfies_language_edge_resolution(graph) is True
    for edge in graph.edges:
        if edge.kind in {ProgramEdgeKind.CALLS, ProgramEdgeKind.RESOLVES_TO}:
            assert edge.binding.resolver_status is not ResolverStatus.RESOLVED_STATIC
        assert edge_cites_source_span_and_resolver_rule(edge)
        assert language_edge_forged_direct_call_reason(edge) == ""

    graph_claim = prove_graph_language_edge_resolution(graph)
    assert graph_claim["evidence"] == "vfs/language-edge-resolution@1"
    assert graph_claim["satisfied"] is True
    assert graph_claim["forged_direct_call_count"] == 0
    assert graph_claim["missing_rule_count"] == 0
    assert graph_claim["missing_span_count"] == 0


def test_language_edge_graph_rejects_forged_resolved_static_collision_edges() -> None:
    """Graph-side checks refuse forged resolved_static collision call edges."""

    forest_id = "forest:forged-check"
    blob = "blob:forged"
    caller = make_node(
        kind=ProgramNodeKind.SYMBOL,
        record_key="site:caller",
        producer="test",
        blob_cid=blob,
        forest_id=forest_id,
        span={"line_start": 1, "column_start": 0, "line_end": 1, "column_end": 4},
    )
    target_a = make_node(
        kind=ProgramNodeKind.SYMBOL,
        record_key="site:target_a",
        producer="test",
        blob_cid=blob,
        forest_id=forest_id,
        span={"line_start": 2, "column_start": 0, "line_end": 2, "column_end": 4},
    )
    forged = make_edge(
        source=caller.node_id,
        target=target_a.node_id,
        kind=ProgramEdgeKind.CALLS,
        producer="test",
        blob_cid=blob,
        forest_id=forest_id,
        span={"line_start": 3, "column_start": 0, "line_end": 3, "column_end": 8},
        resolver_status=ResolverStatus.RESOLVED_STATIC,
        resolver_rule="rule:test:same_name_collision",
        record={
            "reason": "same_name_collision",
            "reason_code": "same_name_collision",
        },
    )
    graph = build_program_graph(
        forest_id=forest_id,
        nodes=(caller, target_a),
        edges=(forged,),
    )
    assert language_edge_forged_direct_call_reason(forged)
    assert graph_satisfies_language_edge_resolution(graph) is False
    claim = prove_graph_language_edge_resolution(graph)
    assert claim["satisfied"] is False
    assert claim["forged_direct_call_count"] == 1

    # Honest ambiguous collision edge with span+rule is accepted.
    honest = make_edge(
        source=caller.node_id,
        target=target_a.node_id,
        kind=ProgramEdgeKind.CALLS,
        producer="test",
        blob_cid=blob,
        forest_id=forest_id,
        span={"line_start": 3, "column_start": 0, "line_end": 3, "column_end": 8},
        resolver_status=ResolverStatus.AMBIGUOUS,
        resolver_rule="rule:test:same_name_collision",
        record={
            "reason": "same_name_collision",
            "reason_code": "same_name_collision",
        },
    )
    honest_graph = build_program_graph(
        forest_id=forest_id,
        nodes=(caller, target_a),
        edges=(honest,),
    )
    assert edge_cites_source_span_and_resolver_rule(honest)
    assert graph_satisfies_language_edge_resolution(honest_graph) is True
    assert prove_graph_language_edge_resolution(honest_graph)["satisfied"] is True
