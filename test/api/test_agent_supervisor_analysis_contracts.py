from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis_contracts import (
    ANALYSIS_EVIDENCE_PACKET_SCHEMA,
    MILLION,
    AnalysisCacheDisposition,
    AnalysisContractValidationError,
    AnalysisCost,
    AnalysisEvidencePacket,
    AnalysisFreshness,
    AnalysisLimits,
    AnalysisOutcome,
    AnalysisStageReceipt,
    AnalysisStageStatus,
    ArtifactReference,
    CandidateProposal,
    ProvenanceKind,
    ProvenanceReference,
    canonical_analysis_json_bytes,
)


NOW = datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc)


def _artifact(identifier: str = "ast-index") -> ArtifactReference:
    return ArtifactReference(
        artifact_id=identifier,
        kind="ast_index",
        uri=f"data/analysis/{identifier}.json",
        artifact_content_id=f"bafy-{identifier}",
        sha256="a" * 64,
        byte_count=8_000_000,
        record_count=50_000,
    )


def _provenance(identifier: str = "symbol-record") -> ProvenanceReference:
    return ProvenanceReference(
        reference_id=identifier,
        kind=ProvenanceKind.AST_RECORD,
        artifact=_artifact(),
        repository_id="repo:example",
        tree_id="tree:abc",
        path="src/example.py",
        symbol="Example.run",
        record_id="ast:123",
        line_start=10,
        line_end=20,
    )


def _proposal(identifier: str = "candidate") -> CandidateProposal:
    return CandidateProposal(
        summary=f"Implement bounded {identifier}",
        objective_terms=("bounded analysis", "evidence"),
        predicted_files=("src/example.py",),
        predicted_symbols=("Example.run",),
        validation_commands=("python -m pytest test_example.py -q",),
        confidence=0.875,
        novelty=0.625,
        cost=2.5,
        provenance=(_provenance(),),
        artifacts=(_artifact(),),
        source_stage="exhaustive_ast",
    )


def _receipt(
    *,
    status: AnalysisStageStatus | str = AnalysisStageStatus.COMPLETED,
    outcome: AnalysisOutcome | str = AnalysisOutcome.CONCLUSIVE,
    freshness: AnalysisFreshness | str = AnalysisFreshness.FRESH,
    cache_disposition: AnalysisCacheDisposition | str = (
        AnalysisCacheDisposition.NOT_CACHED
    ),
    coverage_complete: bool = True,
    truncated: bool = False,
    error_code: str = "",
    reason_code: str = "bounded_search_complete",
) -> AnalysisStageReceipt:
    proposals = (
        ()
        if status
        in {
            AnalysisStageStatus.FAILED,
            AnalysisStageStatus.TIMED_OUT,
            AnalysisStageStatus.SKIPPED,
            "failed",
            "timed_out",
            "skipped",
        }
        else (_proposal(),)
    )
    return AnalysisStageReceipt(
        stage="exhaustive_ast",
        status=status,
        outcome=outcome,
        analyzer_id="analysis-ast",
        analyzer_version="analysis-ast-v1",
        repository_id="repo:example",
        tree_id="tree:abc",
        objective_revision="sha256:objective",
        configuration_digest="sha256:configuration",
        query_digest="sha256:query",
        policy_digest="sha256:policy",
        freshness=freshness,
        cache_disposition=cache_disposition,
        coverage_complete=coverage_complete,
        truncated=truncated,
        reason_code=reason_code,
        error_code=error_code,
        started_at=NOW,
        finished_at=NOW,
        confidence=0.9,
        novelty=0.7,
        cost=AnalysisCost(
            wall_time_ms=250,
            cpu_time_ms=175,
            input_bytes=80_000,
            output_bytes=2_000,
            records_examined=500,
        ),
        proposals=proposals,
        provenance=(_provenance(),),
        artifacts=(_artifact(),),
    )


def _packet(**overrides: object) -> AnalysisEvidencePacket:
    values: dict[str, object] = {
        "repository_id": "repo:example",
        "tree_id": "tree:abc",
        "objective_revision": "sha256:objective",
        "outcome": AnalysisOutcome.CONCLUSIVE,
        "conclusion_code": "bounded_analysis_complete",
        "stage_receipts": (_receipt(),),
    }
    values.update(overrides)
    return AnalysisEvidencePacket(**values)


def test_full_packet_round_trip_preserves_typed_state_and_identity() -> None:
    packet = _packet()

    encoded = packet.to_json()
    restored = AnalysisEvidencePacket.from_json(encoded)

    assert restored == packet
    assert restored.packet_id == packet.packet_id
    assert restored.to_json() == encoded
    assert restored.outcome is AnalysisOutcome.CONCLUSIVE
    assert restored.stage_receipts[0].status is AnalysisStageStatus.COMPLETED
    assert restored.stage_receipts[0].cost.records_examined == 500
    assert restored.candidate_proposals == ()
    assert restored.safe_for_completion_reasoning is True
    assert restored.completion_evidence_receipt_ids == (
        restored.stage_receipts[0].receipt_id,
    )
    assert json.loads(encoded)["schema"] == ANALYSIS_EVIDENCE_PACKET_SCHEMA


def test_nested_contracts_round_trip_independently() -> None:
    values = (
        AnalysisLimits(),
        _artifact(),
        _provenance(),
        AnalysisCost(model_calls=1, input_tokens=100, output_tokens=20),
        _proposal(),
        _receipt(),
    )

    for value in values:
        restored = type(value).from_json(value.to_json())
        assert restored == value
        assert restored.content_id == value.content_id


def test_serialization_and_identity_are_deterministic_across_input_order() -> None:
    first = CandidateProposal(
        "Stable proposal",
        objective_terms=("zeta", "alpha", "zeta"),
        predicted_files=("z.py", "a.py"),
        predicted_symbols=("z", "a"),
        confidence=0.1 + 0.2,
        novelty=0.3,
        cost=1,
        provenance=(
            ProvenanceReference("z", kind="tool"),
            ProvenanceReference("a", kind="tool"),
        ),
        artifacts=(_artifact("z"), _artifact("a")),
    )
    second = CandidateProposal(
        "Stable proposal",
        objective_terms=("alpha", "zeta"),
        predicted_files=("a.py", "z.py"),
        predicted_symbols=("a", "z"),
        confidence="0.30000000000000004",
        novelty="0.3",
        cost="1.0",
        provenance=(
            ProvenanceReference("a", kind="tool"),
            ProvenanceReference("z", kind="tool"),
        ),
        artifacts=(_artifact("a"), _artifact("z")),
    )

    assert first.confidence_millionths == 300_000
    assert first.content_id == second.content_id
    assert first.to_json() == second.to_json()
    assert canonical_analysis_json_bytes(first) == canonical_analysis_json_bytes(second)


def test_fixed_point_metrics_are_finite_bounded_and_canonical() -> None:
    proposal = _proposal()
    assert proposal.confidence_millionths == 875_000
    assert proposal.novelty_millionths == 625_000
    assert proposal.cost_millionths == 2_500_000
    assert proposal.confidence == 0.875

    with pytest.raises(AnalysisContractValidationError, match="confidence"):
        CandidateProposal("bad", ("term",), confidence=float("nan"))
    with pytest.raises(AnalysisContractValidationError, match="novelty"):
        CandidateProposal("bad", ("term",), novelty=1.01)
    with pytest.raises(AnalysisContractValidationError, match="cost"):
        CandidateProposal("bad", ("term",), cost=-1)
    with pytest.raises(AnalysisContractValidationError, match="confidence"):
        CandidateProposal(
            "bad",
            ("term",),
            confidence_millionths=MILLION + 1,
        )


def test_contracts_are_frozen_and_normalize_nested_collections() -> None:
    packet = _packet()

    with pytest.raises(FrozenInstanceError):
        packet.tree_id = "different"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        packet.stage_receipts[0].status = AnalysisStageStatus.FAILED  # type: ignore[misc]
    assert isinstance(packet.stage_receipts, tuple)
    assert isinstance(packet.stage_receipts[0].proposals, tuple)


def test_artifact_references_keep_large_bodies_out_of_packet() -> None:
    artifact = _artifact()
    packet = _packet(artifacts=(artifact,))
    serialized = packet.to_json()

    assert artifact.byte_count == 8_000_000
    assert artifact.artifact_content_id == "bafy-ast-index"
    assert len(serialized.encode("utf-8")) < 20_000
    assert "source_body" not in serialized
    assert "ast_body" not in serialized
    assert "model_response" not in serialized

    invalid = artifact.to_dict()
    invalid["source_body"] = "def unbounded(): ..."
    with pytest.raises(AnalysisContractValidationError, match="unsupported fields"):
        ArtifactReference.from_dict(invalid)


@pytest.mark.parametrize(
    ("limits", "override", "match"),
    (
        (
            AnalysisLimits(max_stage_receipts=1),
            {"stage_receipts": (_receipt(), _receipt(reason_code="again"))},
            "stage_receipts count",
        ),
        (
            AnalysisLimits(max_candidate_proposals=1),
            {
                "outcome": AnalysisOutcome.INCONCLUSIVE,
                "stage_receipts": (),
                "candidate_proposals": (_proposal("one"), _proposal("two")),
            },
            "candidate_proposals count",
        ),
        (
            AnalysisLimits(max_provenance_references=1),
            {
                "outcome": AnalysisOutcome.INCONCLUSIVE,
                "stage_receipts": (),
                "provenance": (
                    ProvenanceReference("one"),
                    ProvenanceReference("two"),
                ),
            },
            "provenance_references count",
        ),
        (
            AnalysisLimits(max_artifact_references=1),
            {
                "outcome": AnalysisOutcome.INCONCLUSIVE,
                "stage_receipts": (),
                "artifacts": (_artifact("one"), _artifact("two")),
            },
            "artifact_references count",
        ),
    ),
)
def test_configurable_total_count_bounds_cover_nested_evidence(
    limits: AnalysisLimits, override: dict[str, object], match: str
) -> None:
    with pytest.raises(AnalysisContractValidationError, match=match):
        _packet(limits=limits, **override)


def test_configurable_text_record_and_packet_byte_bounds() -> None:
    proposal = CandidateProposal("x" * 100, ("term",))
    with pytest.raises(AnalysisContractValidationError, match="summary"):
        _packet(
            outcome=AnalysisOutcome.INCONCLUSIVE,
            stage_receipts=(),
            candidate_proposals=(proposal,),
            limits=AnalysisLimits(max_text_bytes=50),
        )

    with pytest.raises(AnalysisContractValidationError, match="max_record_bytes"):
        _packet(
            outcome=AnalysisOutcome.INCONCLUSIVE,
            stage_receipts=(),
            candidate_proposals=(proposal,),
            limits=AnalysisLimits(
                max_text_bytes=128,
                max_record_bytes=256,
                max_serialized_bytes=10_000,
            ),
        )

    with pytest.raises(
        AnalysisContractValidationError, match="max_serialized_bytes"
    ):
        _packet(
            outcome=AnalysisOutcome.INCONCLUSIVE,
            stage_receipts=(),
            candidate_proposals=(proposal,),
            limits=AnalysisLimits(
                max_text_bytes=128,
                max_record_bytes=1_000,
                max_serialized_bytes=1_000,
            ),
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"status": AnalysisStageStatus.FAILED, "error_code": "tool_failed"},
        {"status": AnalysisStageStatus.PARTIAL},
        {"status": AnalysisStageStatus.TIMED_OUT, "error_code": "deadline"},
        {"freshness": AnalysisFreshness.STALE},
        {"cache_disposition": AnalysisCacheDisposition.NEGATIVE_HIT},
        {"coverage_complete": False},
        {"truncated": True},
        {"error_code": "unexpected_error"},
    ),
)
def test_invalid_stage_states_cannot_claim_conclusive_outcome(
    changes: dict[str, object],
) -> None:
    with pytest.raises(
        AnalysisContractValidationError, match="conclusive stage receipts"
    ):
        _receipt(**changes)


@pytest.mark.parametrize(
    "changes",
    (
        {"status": AnalysisStageStatus.FAILED, "error_code": "tool_failed"},
        {"status": AnalysisStageStatus.PARTIAL},
        {"status": AnalysisStageStatus.TIMED_OUT, "error_code": "deadline"},
        {"freshness": AnalysisFreshness.STALE},
        {"cache_disposition": AnalysisCacheDisposition.NEGATIVE_HIT},
        {"coverage_complete": False},
        {"truncated": True},
    ),
)
def test_failed_partial_stale_and_negative_receipts_remain_explicitly_inconclusive(
    changes: dict[str, object],
) -> None:
    receipt = _receipt(outcome=AnalysisOutcome.INCONCLUSIVE, **changes)
    packet = _packet(
        outcome=AnalysisOutcome.INCONCLUSIVE,
        stage_receipts=(receipt,),
    )

    assert receipt.safe_for_completion_reasoning is False
    assert packet.safe_for_completion_reasoning is False
    assert packet.completion_evidence_receipts == ()
    with pytest.raises(TypeError):
        bool(receipt)
    with pytest.raises(TypeError):
        bool(packet)
    with pytest.raises(
        AnalysisContractValidationError, match="not safe for completion"
    ):
        packet.require_completion_evidence()


def test_conclusive_packet_requires_real_completion_receipt() -> None:
    partial = _receipt(
        status=AnalysisStageStatus.PARTIAL,
        outcome=AnalysisOutcome.INCONCLUSIVE,
    )

    with pytest.raises(
        AnalysisContractValidationError, match="completion-eligible stage receipt"
    ):
        _packet(stage_receipts=(partial,))
    with pytest.raises(
        AnalysisContractValidationError, match="completion-eligible stage receipt"
    ):
        _packet(stage_receipts=())
    with pytest.raises(
        AnalysisContractValidationError, match="completion-eligible stage receipt"
    ):
        _packet(coverage_complete=False)
    with pytest.raises(
        AnalysisContractValidationError, match="completion-eligible stage receipt"
    ):
        _packet(truncated=True)


def test_completion_claims_and_content_id_claims_are_verified_on_decode() -> None:
    packet = _packet()
    forged = packet.to_record()
    forged["safe_for_completion_reasoning"] = False
    with pytest.raises(AnalysisContractValidationError, match="completion-evidence"):
        AnalysisEvidencePacket.from_dict(forged)

    forged = packet.to_record()
    forged["completion_evidence_receipt_ids"] = []
    with pytest.raises(AnalysisContractValidationError, match="completion receipt"):
        AnalysisEvidencePacket.from_dict(forged)

    forged = packet.to_record()
    forged["packet_id"] = "bafy-forged"
    with pytest.raises(AnalysisContractValidationError, match="content identity"):
        AnalysisEvidencePacket.from_dict(forged)

    receipt = _receipt()
    forged_receipt = receipt.to_record()
    forged_receipt["safe_for_completion_reasoning"] = False
    with pytest.raises(AnalysisContractValidationError, match="completion-evidence"):
        AnalysisStageReceipt.from_dict(forged_receipt)


def test_packet_rejects_receipts_for_a_different_bound_context() -> None:
    receipt = AnalysisStageReceipt(
        stage="static",
        status="completed",
        outcome="conclusive",
        analyzer_id="static",
        analyzer_version="1",
        repository_id="repo:example",
        tree_id="tree:stale",
        objective_revision="sha256:objective",
    )

    with pytest.raises(AnalysisContractValidationError, match="tree_id"):
        _packet(stage_receipts=(receipt,))
