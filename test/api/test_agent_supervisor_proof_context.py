from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.artifact_store import (
    query_code_evidence_neighborhood,
    write_code_evidence_graph_artifact,
)
from ipfs_accelerate_py.agent_supervisor.code_evidence_graph import (
    build_code_evidence_graph,
)
from ipfs_accelerate_py.agent_supervisor.proof_context import (
    ContextTrust,
    ProofContextBudgetError,
    ProofContextBuilder,
    ProofContextCapsule,
    ProofContextLimits,
    ProofContextQuery,
    ProofContextTarget,
    estimate_context_tokens,
)


SYMBOL = "agent_supervisor.proof_context.ProofContextCapsule"
UNRELATED_SYMBOL = "unrelated.package.RepositoryWideRecord"


def _graph_path(tmp_path: Path) -> Path:
    graph = build_code_evidence_graph(
        task_records=[
            {
                "task_id": "REF-252",
                "canonical_task_cid": "task-cid-252",
                "title": "Generate bounded proof context capsules",
                "depends_on": ["REF-250"],
                "acceptance": ["Capsules are bounded."],
            },
            {
                "task_id": "REF-250",
                "canonical_task_cid": "task-cid-250",
                "title": "Build evidence graph",
            },
            {
                "task_id": "REF-999",
                "canonical_task_cid": "task-cid-999",
                "title": "Unrelated repository task",
            },
        ],
        ast_records=[
            {
                "scope_id": "scope-capsule",
                "kind": "qualified_symbol",
                "qualified_symbol": SYMBOL,
                "path": "agent_supervisor/proof_context.py",
                "line_start": 10,
                "line_end": 14,
                "full_ast": "repository-wide AST must not be copied",
            },
            {
                "scope_id": "scope-unrelated",
                "kind": "qualified_symbol",
                "qualified_symbol": UNRELATED_SYMBOL,
                "path": "unrelated/repository.py",
            },
        ],
        obligations=[
            {
                "obligation_id": "obligation-graph",
                "task_id": "REF-250",
                "statement": "The graph projection is deterministic.",
            },
            {
                "obligation_id": "obligation-unrelated",
                "task_id": "REF-999",
                "ast_scope_ids": ["scope-unrelated"],
                "statement": "Unrelated statement.",
            },
            {
                "obligation_id": "obligation-context",
                "task_id": "REF-252",
                "ast_scope_ids": ["scope-capsule"],
                "premise_ids": ["obligation-graph"],
                "statement": "The context is bounded before prompt assembly.",
                "template_id": "unsupported-proof-fail-closed",
                "support_status": "unsupported",
                "unsupported_semantics": ["arbitrary Python semantics"],
                "fallback_checks": ["pytest:test_context_fallback"],
            },
        ],
        proof_records=[
            {
                "receipt_id": "receipt-context",
                "task_id": "REF-252",
                "obligation_id": "obligation-context",
                "verdict": "proved",
                "freshness": "current",
                "authoritative_assurance": "kernel_verified",
                "kernel_transcript": "relevant transcript " * 200,
                "private_witness": "hidden-witness-value",
            },
            {
                "receipt_id": "receipt-unrelated",
                "task_id": "REF-999",
                "obligation_id": "obligation-unrelated",
                "verdict": "proved",
                "freshness": "current",
                "authoritative_assurance": "kernel_verified",
                "kernel_transcript": "UNRELATED-TRANSCRIPT",
            },
        ],
        validation_records=[
            {
                "validation_receipt_id": "contradiction-context",
                "task_id": "REF-252",
                "status": "failed",
                "contradiction_id": "contradiction-context",
                "contradiction": "A counterexample invalidated the prior claim.",
            },
            {
                "validation_receipt_id": "validation-unrelated",
                "task_id": "REF-999",
                "status": "passed",
            },
        ],
        enrichments=[
            {
                "id": "model-suggestion",
                "source": "GraphRAG",
                "edge_kind": "suggests",
                "targets": ["REF-252"],
                "suggestion": "Try a smaller proof premise set.",
            },
            {
                "id": "unrelated-suggestion",
                "source": "GraphRAG",
                "edge_kind": "mentions",
                "targets": ["REF-999"],
                "summary": "Unrelated model output.",
            },
        ],
    )
    path = tmp_path / "evidence.json"
    write_code_evidence_graph_artifact(path, graph)
    return path


def test_exact_query_selectors_and_safe_graph_hops(tmp_path: Path) -> None:
    path = _graph_path(tmp_path)

    result = query_code_evidence_neighborhood(
        path,
        task_id="REF-252",
        symbols=[SYMBOL],
        dependency_task_ids=["REF-250"],
        obligation_ids=["obligation-context"],
        receipt_ids=["receipt-context"],
        contradiction_ids=["contradiction-context"],
        max_hops=3,
        limit=40,
    )

    record_keys = {node["record_key"] for node in result["nodes"]}
    assert {
        "task-cid-252",
        "task-cid-250",
        "scope-capsule",
        "obligation-context",
        "receipt-context",
        "contradiction-context",
    }.issubset(record_keys)
    assert "task-cid-999" not in record_keys
    assert "scope-unrelated" not in record_keys
    assert "receipt-unrelated" not in record_keys
    assert result["row_count"] <= 40
    assert result["max_hops"] == 3

    with pytest.raises(ValueError, match="exact"):
        query_code_evidence_neighborhood(path, task_id="*", symbols=["*"])


def test_capsule_partitions_trust_and_excludes_prohibited_context(
    tmp_path: Path,
) -> None:
    path = _graph_path(tmp_path)
    limits = ProofContextLimits(
        max_rows=40,
        max_bytes=24_000,
        max_tokens=8_000,
        max_graph_hops=3,
        max_source_excerpts=1,
        max_source_excerpt_bytes=96,
        max_source_bytes=96,
        max_proof_transcripts=1,
        max_proof_transcript_bytes=80,
        max_proof_transcript_bytes_total=80,
    )
    capsule = ProofContextBuilder(path).build(
        ProofContextQuery(
            task_id="REF-252",
            symbols=(SYMBOL,),
            dependency_task_ids=("REF-250",),
            obligation_ids=("obligation-context",),
            receipt_ids=("receipt-context",),
            contradiction_ids=("contradiction-context",),
        ),
        target=ProofContextTarget.LEANSTRAL,
        limits=limits,
        source_excerpts={
            SYMBOL: "def capsule():\n    return 'bounded'\n" * 20,
            UNRELATED_SYMBOL: "UNRELATED-SOURCE",
        },
    )

    assert capsule.target is ProofContextTarget.LEANSTRAL
    assert all(
        item.trust is ContextTrust.TRUSTED_FACT for item in capsule.trusted_facts
    )
    assert {item.record_id for item in capsule.untrusted_suggestions} == {
        "model-suggestion"
    }
    assert capsule.unsupported_semantics
    assert capsule.required_fallback_checks == ("pytest:test_context_fallback",)
    assert len(capsule.source_excerpts) == 1
    assert capsule.source_excerpts[0].symbol == SYMBOL
    assert capsule.source_excerpts[0].byte_count <= 96
    assert len(capsule.proof_transcripts) == 1
    assert capsule.proof_transcripts[0].receipt_id == "receipt-context"
    assert capsule.proof_transcripts[0].byte_count <= 80

    prompt = capsule.to_prompt()
    assert "hidden-witness-value" not in prompt
    assert "private_witness" not in prompt
    assert "repository-wide AST" not in prompt
    assert "UNRELATED-TRANSCRIPT" not in prompt
    assert "UNRELATED-SOURCE" not in prompt
    assert UNRELATED_SYMBOL not in prompt
    assert "unrelated-suggestion" not in prompt
    assert len(prompt.encode("utf-8")) == capsule.usage.bytes <= limits.max_bytes
    assert estimate_context_tokens(prompt) == capsule.usage.tokens <= limits.max_tokens
    assert ProofContextCapsule.from_json(prompt) == capsule


def test_all_dimensions_are_enforced_before_rendering(tmp_path: Path) -> None:
    path = _graph_path(tmp_path)
    limits = ProofContextLimits(
        max_rows=8,
        max_bytes=5_000,
        max_tokens=1_250,
        max_graph_hops=1,
        max_source_excerpts=1,
        max_source_excerpt_bytes=32,
        max_source_bytes=32,
        max_proof_transcripts=1,
        max_proof_transcript_bytes=24,
        max_proof_transcript_bytes_total=24,
    )

    capsule = ProofContextBuilder(path).build(
        task_id="REF-252",
        symbols=(SYMBOL,),
        obligation_ids=("obligation-context",),
        receipt_ids=("receipt-context",),
        limits=limits,
        source_excerpts={SYMBOL: "x" * 1_000},
    )

    assert capsule.usage.rows <= 8
    assert capsule.usage.graph_hops == 1
    assert capsule.usage.source_excerpts <= 1
    assert capsule.usage.source_bytes <= 32
    assert capsule.usage.proof_transcripts <= 1
    assert capsule.usage.proof_transcript_bytes <= 24
    assert capsule.usage.bytes <= 5_000
    assert capsule.usage.tokens <= 1_250
    # Prompt generation is a pure serialization of the already bounded value.
    assert capsule.render_for_codex() == capsule.to_json()


def test_impossibly_small_prompt_budget_fails_closed(tmp_path: Path) -> None:
    path = _graph_path(tmp_path)

    with pytest.raises(ProofContextBudgetError, match="mandatory"):
        ProofContextBuilder(path).build(
            task_id="REF-252",
            limits=ProofContextLimits(max_bytes=64, max_tokens=16),
        )
