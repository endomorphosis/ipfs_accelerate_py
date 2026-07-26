from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis_cache import AnalysisCache
from ipfs_accelerate_py.agent_supervisor.analysis_consensus import (
    AnalysisClaimProvenance,
    AnalysisClaimStatus,
    AnalysisConsensusClaim,
    AnalysisConsensusError,
    AnalysisConsensusOutcome,
    AnalysisConsensusPolicy,
    AnalysisConsensusReceipt,
    AnalysisConsensusResolution,
    AnalysisProducerKind,
    DeterministicDisagreementPolicy,
    build_analysis_consensus_receipt,
)
from ipfs_accelerate_py.agent_supervisor.analysis_pipeline import (
    AnalysisPipeline,
    AnalysisPipelinePolicy,
    AnalysisPipelineRequest,
    PipelineCacheStatus,
    make_analysis_stage_receipt,
)


TREE = "tree:sha256:consensus-fixture"


def _provenance(
    kind: AnalysisProducerKind,
    *,
    producer_id: str | None = None,
) -> AnalysisClaimProvenance:
    return AnalysisClaimProvenance(
        source_id=f"source:{kind.value}",
        dataset_id=f"dataset:{kind.value}",
        graph_id=f"graph:{kind.value}",
        chunk_id=f"chunk:{kind.value}",
        producer_id=producer_id or f"producer:{kind.value}",
        model_id=f"model:{kind.value}",
        policy_id="policy:fixture@1",
        capability_id=f"capability:{kind.value}@1",
        tree_id=TREE,
    )


def _claim(
    kind: AnalysisProducerKind,
    verdict: str,
    *,
    status: AnalysisClaimStatus = AnalysisClaimStatus.CONCLUSIVE,
    confidence: int = 0,
    proposal_only: bool | None = None,
    validates_claim_id: str = "",
    producer_id: str | None = None,
    reference_id: str = "evidence:shared",
) -> AnalysisConsensusClaim:
    return AnalysisConsensusClaim(
        producer_kind=kind,
        result_id=f"result:{kind.value}:{verdict}:{confidence}",
        verdict=verdict,
        status=status,
        provenance=_provenance(kind, producer_id=producer_id),
        evidence_references=(
            {
                "reference_id": reference_id,
                "source_id": f"source-ref:{kind.value}",
                "dataset_id": f"dataset:{kind.value}",
                "graph_id": f"graph:{kind.value}",
                "chunk_id": f"chunk:{kind.value}",
                "producer_id": f"producer:{kind.value}",
                "model_id": f"model:{kind.value}",
                "tree_id": TREE,
            },
        ),
        proposal_only=(
            kind is not AnalysisProducerKind.LOCAL
            if proposal_only is None
            else proposal_only
        ),
        confidence_millionths=confidence,
        validates_claim_id=validates_claim_id,
    )


def _receipt(
    local: AnalysisConsensusClaim,
    datasets: AnalysisConsensusClaim | None = None,
    **values: object,
) -> AnalysisConsensusReceipt:
    return build_analysis_consensus_receipt(
        repository_id="repository:fixture",
        tree_id=TREE,
        objective_revision="objective@1",
        operation="graphrag_retrieval",
        local_claim=local,
        datasets_claim=datasets,
        **values,
    )


def test_agreement_is_canonical_and_preserves_full_compact_provenance() -> None:
    local = _claim(AnalysisProducerKind.LOCAL, "same", confidence=1)
    datasets = _claim(
        AnalysisProducerKind.DATASETS, "same", confidence=999_999
    )

    receipt = _receipt(local, datasets)
    restored = AnalysisConsensusReceipt.from_dict(receipt.to_dict())

    assert receipt.outcome is AnalysisConsensusOutcome.AGREEMENT
    assert receipt.resolution is AnalysisConsensusResolution.AGREEMENT
    assert restored == receipt
    assert restored.receipt_id == receipt.receipt_id
    assert all(
        (
            claim.provenance.source_id,
            claim.provenance.dataset_id,
            claim.provenance.graph_id,
            claim.provenance.chunk_id,
            claim.provenance.producer_id,
            claim.provenance.model_id,
            claim.provenance.policy_id,
            claim.provenance.capability_id,
            claim.provenance.tree_id,
        )
        for claim in receipt.claims
    )
    assert not receipt.completion_authority
    assert not receipt.is_completion_evidence
    assert not receipt.safe_for_completion_reasoning
    assert receipt.completion_eligible_claim_ids == ()


def test_unresolved_disagreement_retains_uncertainty_and_selects_nothing() -> None:
    receipt = _receipt(
        _claim(AnalysisProducerKind.LOCAL, "left"),
        _claim(AnalysisProducerKind.DATASETS, "right", confidence=1_000_000),
    )

    assert receipt.outcome is AnalysisConsensusOutcome.DISAGREEMENT
    assert receipt.resolution is (
        AnalysisConsensusResolution.EXPLICIT_UNCERTAINTY
    )
    assert receipt.selected_claim is None
    assert receipt.residual_uncertainty


@pytest.mark.parametrize(
    ("rule", "selected_kind"),
    [
        (
            DeterministicDisagreementPolicy.PREFER_LOCAL,
            AnalysisProducerKind.LOCAL,
        ),
        (
            DeterministicDisagreementPolicy.PREFER_DATASETS,
            AnalysisProducerKind.DATASETS,
        ),
    ],
)
def test_disagreement_selection_requires_an_explicit_deterministic_rule(
    rule: DeterministicDisagreementPolicy,
    selected_kind: AnalysisProducerKind,
) -> None:
    receipt = _receipt(
        _claim(AnalysisProducerKind.LOCAL, "left", confidence=0),
        _claim(AnalysisProducerKind.DATASETS, "right", confidence=1_000_000),
        policy=AnalysisConsensusPolicy(disagreement_policy=rule),
    )

    assert receipt.outcome is AnalysisConsensusOutcome.DISAGREEMENT
    assert receipt.resolution is (
        AnalysisConsensusResolution.DETERMINISTIC_POLICY
    )
    assert receipt.selected_claim is not None
    assert receipt.selected_claim.producer_kind is selected_kind


def test_confidence_changes_neither_agreement_nor_deterministic_selection() -> None:
    local = _claim(AnalysisProducerKind.LOCAL, "left", confidence=0)
    remote_low = _claim(
        AnalysisProducerKind.DATASETS, "right", confidence=1
    )
    remote_high = replace(
        remote_low, confidence_millionths=1_000_000, claim_id=""
    )
    policy = AnalysisConsensusPolicy(
        disagreement_policy=(
            DeterministicDisagreementPolicy.LEXICOGRAPHIC_CLAIM_ID
        )
    )

    low = _receipt(local, remote_low, policy=policy)
    high = _receipt(local, remote_high, policy=policy)

    # Claim IDs retain confidence for audit, but semantic agreement and the
    # allowlisted policy do not compare it.  Authority is always fixed false.
    assert local.semantic_id != remote_low.semantic_id
    assert remote_low.semantic_id == remote_high.semantic_id
    assert low.outcome is high.outcome is AnalysisConsensusOutcome.DISAGREEMENT
    assert low.selected_claim is not None
    assert high.selected_claim is not None
    assert (
        low.selected_claim.producer_kind
        is high.selected_claim.producer_kind
    )
    assert not low.completion_authority
    assert not high.completion_authority


def test_forged_deterministic_resolution_must_match_declared_policy() -> None:
    local = _claim(AnalysisProducerKind.LOCAL, "left")
    datasets = _claim(AnalysisProducerKind.DATASETS, "right")

    with pytest.raises(
        AnalysisConsensusError, match="must match the declared policy"
    ):
        AnalysisConsensusReceipt(
            repository_id="repository:fixture",
            tree_id=TREE,
            objective_revision="objective@1",
            operation="graph_retrieval",
            policy=AnalysisConsensusPolicy(),
            outcome=AnalysisConsensusOutcome.DISAGREEMENT,
            resolution=AnalysisConsensusResolution.DETERMINISTIC_POLICY,
            claims=(local, datasets),
            selected_claim_id=local.claim_id,
            residual_uncertainty=("claims disagree",),
        )


def test_independent_validation_requires_a_third_producer() -> None:
    local = _claim(AnalysisProducerKind.LOCAL, "left")
    datasets = _claim(AnalysisProducerKind.DATASETS, "right")
    validator = _claim(
        AnalysisProducerKind.VALIDATOR,
        "validated",
        validates_claim_id=datasets.claim_id,
        producer_id="producer:independent-validator",
    )

    receipt = _receipt(local, datasets, validator_claim=validator)

    assert receipt.outcome is AnalysisConsensusOutcome.INDEPENDENT_VALIDATION
    assert receipt.resolution is (
        AnalysisConsensusResolution.INDEPENDENT_VALIDATOR
    )
    assert receipt.selected_claim_id == datasets.claim_id
    assert not receipt.is_completion_evidence

    same_producer = replace(
        validator,
        provenance=replace(
            validator.provenance, producer_id=local.provenance.producer_id
        ),
        claim_id="",
    )
    unresolved = _receipt(local, datasets, validator_claim=same_producer)
    assert unresolved.outcome is AnalysisConsensusOutcome.DISAGREEMENT
    assert unresolved.selected_claim_id == ""


def test_degraded_fallback_is_explicit_and_selects_only_local() -> None:
    local = _claim(AnalysisProducerKind.LOCAL, "local")
    failed = _claim(
        AnalysisProducerKind.DATASETS,
        "failed",
        status=AnalysisClaimStatus.FAILED,
    )

    receipt = _receipt(
        local,
        failed,
        fallback_reason_code="datasets_provider_unavailable",
    )

    assert receipt.outcome is AnalysisConsensusOutcome.DEGRADED_FALLBACK
    assert receipt.resolution is AnalysisConsensusResolution.LOCAL_FALLBACK
    assert receipt.fallback_explicit
    assert receipt.fallback_reason_code == "datasets_provider_unavailable"
    assert receipt.selected_claim_id == local.claim_id
    assert failed.claim_id in receipt.excluded_claim_ids


@pytest.mark.parametrize(
    "status",
    [
        AnalysisClaimStatus.PARTIAL,
        AnalysisClaimStatus.STALE,
        AnalysisClaimStatus.INCONCLUSIVE,
    ],
)
def test_partial_stale_and_inconclusive_results_are_never_selected(
    status: AnalysisClaimStatus,
) -> None:
    local = _claim(
        AnalysisProducerKind.LOCAL,
        "local",
        status=status,
        proposal_only=status is not AnalysisClaimStatus.CONCLUSIVE,
    )
    receipt = _receipt(local)

    assert receipt.outcome is AnalysisConsensusOutcome.PARTIAL_RESULT
    assert receipt.resolution is AnalysisConsensusResolution.PARTIAL_ONLY
    assert receipt.selected_claim_id == ""
    assert local.claim_id in receipt.excluded_claim_ids
    assert receipt.residual_uncertainty
    assert receipt.completion_eligible_claim_ids == ()


def test_proposal_only_claims_cannot_gain_completion_authority_by_agreement() -> None:
    local = _claim(
        AnalysisProducerKind.LOCAL, "same", proposal_only=True
    )
    datasets = _claim(AnalysisProducerKind.DATASETS, "same")
    receipt = _receipt(local, datasets)

    assert receipt.outcome is AnalysisConsensusOutcome.AGREEMENT
    assert set(receipt.excluded_claim_ids) == {
        local.claim_id,
        datasets.claim_id,
    }
    assert receipt.completion_eligible_claim_ids == ()
    assert not receipt.safe_for_completion_reasoning


def test_receipt_bounds_and_heavy_payload_rejection() -> None:
    with pytest.raises(AnalysisConsensusError, match="forbidden payload"):
        _claim(
            AnalysisProducerKind.LOCAL,
            "local",
            reference_id="evidence:local",
        ).__class__(
            producer_kind=AnalysisProducerKind.LOCAL,
            result_id="result:heavy",
            verdict="local",
            status=AnalysisClaimStatus.CONCLUSIVE,
            provenance=_provenance(AnalysisProducerKind.LOCAL),
            evidence_references=(
                {"reference_id": "evidence:heavy", "source_text": "x" * 10},
            ),
        )

    small_policy = AnalysisConsensusPolicy(
        max_receipt_bytes=1024, max_reference_bytes=512
    )
    with pytest.raises(AnalysisConsensusError, match="max_receipt_bytes"):
        _receipt(
            _claim(
                AnalysisProducerKind.LOCAL,
                "v" * 700,
                reference_id="evidence:large",
            ),
            policy=small_policy,
        )


class _Analyzer:
    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, context):
        self.calls += 1
        return make_analysis_stage_receipt(
            context.request,
            successful=True,
            reason_code="consensus_pipeline_fixture",
        )


def test_pipeline_cold_and_warm_results_restore_the_same_consensus_receipt(
    tmp_path: Path,
) -> None:
    analyzer = _Analyzer()
    pipeline = AnalysisPipeline(AnalysisCache(tmp_path), analyzer)
    request = AnalysisPipelineRequest(
        repository_id="repository:fixture",
        tree_id=TREE,
        objective_revision="objective@1",
        query={"text": "consensus"},
        analyzer_id="fixture.consensus",
        analyzer_version="fixture.consensus@1",
        provider_operation="graph_retrieval",
    )

    cold = pipeline.analyze(request)
    warm = pipeline.analyze(request)

    assert cold.cache_status is PipelineCacheStatus.PRODUCED
    assert warm.cache_status is PipelineCacheStatus.EXACT_HIT
    assert analyzer.calls == 1
    assert cold.consensus_receipt is not None
    assert warm.consensus_receipt is not None
    assert cold.consensus_receipt.equivalent_to(warm.consensus_receipt)
    assert (
        cold.to_dict()["consensus_receipt"]
        == warm.to_dict()["consensus_receipt"]
    )
    assert cold.consensus_receipt.fallback_explicit
    assert cold.consensus_receipt.serialized_byte_count <= (
        cold.consensus_receipt.policy.max_receipt_bytes
    )
    assert not warm.consensus_receipt.completion_authority


def test_joined_uncached_partial_result_retains_the_same_receipt(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    class PartialAnalyzer:
        def analyze(self, context):
            entered.set()
            assert release.wait(5)
            return make_analysis_stage_receipt(
                context.request,
                successful=False,
                reason_code="partial_fixture",
            )

    pipeline = AnalysisPipeline(
        AnalysisCache(tmp_path),
        PartialAnalyzer(),
        policy=AnalysisPipelinePolicy(cache_negative_results=False),
    )
    request = AnalysisPipelineRequest(
        repository_id="repository:fixture",
        tree_id=TREE,
        objective_revision="objective@1",
        query={"text": "partial consensus"},
        analyzer_id="fixture.partial-consensus",
        analyzer_version="fixture.partial-consensus@1",
        provider_operation="graph_retrieval",
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        leader = executor.submit(pipeline.analyze, request)
        assert entered.wait(5)
        follower = executor.submit(pipeline.analyze, request)
        deadline = time.monotonic() + 5
        while (
            pipeline.coordinator.metrics().followers < 1
            and time.monotonic() < deadline
        ):
            time.sleep(0.001)
        assert pipeline.coordinator.metrics().followers == 1
        release.set()
        results = (leader.result(timeout=10), follower.result(timeout=10))

    assert {item.cache_status for item in results} == {
        PipelineCacheStatus.INCONCLUSIVE,
        PipelineCacheStatus.JOINED,
    }
    assert all(item.consensus_receipt is not None for item in results)
    assert results[0].consensus_receipt is not None
    assert results[1].consensus_receipt is not None
    assert results[0].consensus_receipt.equivalent_to(
        results[1].consensus_receipt
    )
