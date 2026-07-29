from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.context.decision_runtime_benchmark import (
    DecisionRuntimeBenchmark,
    DecisionRuntimeBenchmarkError,
    DecisionRuntimePath,
    IrrelevantCorpus,
    REQUIRED_IRRELEVANT_CORPORA,
    build_frozen_decision_runtime_benchmark,
    recompute_proof_dependency_scaling,
    verify_proof_dependency_scaling_report,
)


def test_closed_receipt_population_proves_independent_context_scaling():
    benchmark = build_frozen_decision_runtime_benchmark()
    report = recompute_proof_dependency_scaling(benchmark)

    assert report.passed
    assert set(report.scale_dimensions_passed) == {
        item.value for item in REQUIRED_IRRELEVANT_CORPORA
    }
    assert report.context_scaling_passed
    assert report.invalidation_precision_passed
    assert report.effect_parity_passed
    assert report.terminal_parity_passed
    assert report.deterministic_degraded_passed
    assert report.lazy_discovery_passed
    assert not report.authoritative
    assert not report.completion_authoritative
    assert verify_proof_dependency_scaling_report(report, benchmark)

    proof = [
        receipt
        for receipt in benchmark.receipts
        if receipt.adversarial_fixture is None
        and receipt.path is DecisionRuntimePath.PROOF_DIRECTED
    ]
    baseline = next(item for item in proof if item.scale.intervention is None)
    for dimension in IrrelevantCorpus:
        grown = next(
            item
            for item in proof
            if item.identity.identity_id == baseline.identity.identity_id
            and item.scale.intervention is dimension
        )
        assert (
            grown.metrics.total_corpus_bytes
            >= baseline.metrics.total_corpus_bytes * 10
        )
        assert (
            grown.metrics.provider_input_tokens
            == baseline.metrics.provider_input_tokens
        )
        assert grown.mandatory_closure_id == baseline.mandatory_closure_id
    assert report.closure_token_correlation_millionths >= 900_000
    assert report.corpus_token_correlation_millionths <= 500_000


def test_report_is_recomputed_and_producer_receipt_ids_cannot_be_forged():
    benchmark = build_frozen_decision_runtime_benchmark()
    report = recompute_proof_dependency_scaling(benchmark)
    assert not verify_proof_dependency_scaling_report(
        replace(report, retries=report.retries + 1), benchmark
    )

    payload = benchmark.to_dict()
    payload["receipts"][0]["receipt_id"] = "sha256:" + "0" * 64
    with pytest.raises(DecisionRuntimeBenchmarkError, match="ID mismatch"):
        DecisionRuntimeBenchmark.from_dict(payload)
