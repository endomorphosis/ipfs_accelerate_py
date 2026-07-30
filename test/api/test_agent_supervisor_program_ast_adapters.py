"""Program AST adapter evidence and incremental index conformance tests.

Covers ``vfs/incremental-ast-index@1`` (VFS-G139) and packet co-binding with
``vfs/exhaustive-file-inventory@1`` under goal packet
``goal_packet/corpus_index/ipfs_accelerate_py/26d54d2206f9``.

Language-specific extraction depth lives in the mixed/typescript suites; this
module proves the objective evidence surface, multi-language provenance,
incremental blob reuse, and fail-closed exhaustive verdicts.
"""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.program_ast_adapters import (
    CORPUS_INDEX_G020_EVIDENCE_TERMS,
    EXHAUSTIVE_FILE_INVENTORY_EVIDENCE,
    GOAL_PACKET_ID,
    INCREMENTAL_AST_INDEX_EVIDENCE,
    INCREMENTAL_AST_INDEX_INVARIANTS,
    OBJECTIVE_DOMAIN_EVIDENCE_TERMS,
    OBJECTIVE_GOAL_ID,
    OBJECTIVE_PARENT_GOAL_ID,
    OBJECTIVE_TASK_ID,
    PACKET_GOAL_IDS,
    PROVENANCE_LANGUAGES,
    SourceDocument,
    adapt_program_source,
    all_covered_evidence_terms,
    build_incremental_ast_index,
    build_program_evidence_index,
    covered_evidence_terms,
    detect_program_language,
    incremental_ast_index_evidence_terms,
    index_satisfies_incremental_ast_index,
    packet_evidence_terms,
    prove_incremental_ast_index,
)


def test_incremental_ast_index_evidence_terms_are_bound() -> None:
    """Prove vfs/incremental-ast-index@1 and packet co-binding."""

    assert INCREMENTAL_AST_INDEX_EVIDENCE == "vfs/incremental-ast-index@1"
    assert EXHAUSTIVE_FILE_INVENTORY_EVIDENCE == "vfs/exhaustive-file-inventory@1"
    assert OBJECTIVE_DOMAIN_EVIDENCE_TERMS == ("vfs/incremental-ast-index@1",)
    assert CORPUS_INDEX_G020_EVIDENCE_TERMS == (
        "vfs/exhaustive-file-inventory@1",
        "vfs/incremental-ast-index@1",
    )
    assert incremental_ast_index_evidence_terms() == OBJECTIVE_DOMAIN_EVIDENCE_TERMS
    assert covered_evidence_terms() == incremental_ast_index_evidence_terms()
    assert packet_evidence_terms() == CORPUS_INDEX_G020_EVIDENCE_TERMS
    assert all_covered_evidence_terms() == packet_evidence_terms()
    assert OBJECTIVE_GOAL_ID == "VFS-G139"
    assert OBJECTIVE_PARENT_GOAL_ID == "VFS-G020"
    assert OBJECTIVE_TASK_ID == "VFS-063"
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
    assert all_covered_evidence_terms() == corpus.all_covered_evidence_terms()
    assert EXHAUSTIVE_FILE_INVENTORY_EVIDENCE == corpus.EXHAUSTIVE_FILE_INVENTORY_EVIDENCE
    assert INCREMENTAL_AST_INDEX_EVIDENCE == corpus.INCREMENTAL_AST_INDEX_EVIDENCE
    assert GOAL_PACKET_ID == corpus.GOAL_PACKET_ID
    assert set(PACKET_GOAL_IDS) == set(corpus.PACKET_GOAL_IDS)
    assert OBJECTIVE_PARENT_GOAL_ID == corpus.OBJECTIVE_PARENT_GOAL_ID
    assert OBJECTIVE_TASK_ID == corpus.OBJECTIVE_TASK_ID
    # Domain ownership stays split: inventory owns G138, adapters own G139.
    assert covered_evidence_terms() == ("vfs/incremental-ast-index@1",)
    assert corpus.covered_evidence_terms() == ("vfs/exhaustive-file-inventory@1",)
    assert set(packet_evidence_terms()) == {
        "vfs/exhaustive-file-inventory@1",
        "vfs/incremental-ast-index@1",
    }
