from dataclasses import replace

from ipfs_accelerate_py.agent_supervisor.context.decision_runtime_benchmark import (
    REQUIRED_ADVERSARIAL_FIXTURES,
    DecisionRuntimeBenchmark,
    recompute_proof_dependency_scaling,
    build_frozen_decision_runtime_benchmark,
)


def test_every_required_adversarial_fixture_is_non_compensable():
    benchmark = build_frozen_decision_runtime_benchmark()
    report = recompute_proof_dependency_scaling(benchmark)

    assert set(report.adversarial_fixtures_passed) == {
        item.value for item in REQUIRED_ADVERSARIAL_FIXTURES
    }
    assert report.passed

    target = next(
        receipt
        for receipt in benchmark.receipts
        if receipt.adversarial_fixture is not None
    )
    changed = tuple(
        replace(receipt, escape_count=1)
        if receipt.receipt_id == target.receipt_id
        else receipt
        for receipt in benchmark.receipts
    )
    failed = recompute_proof_dependency_scaling(DecisionRuntimeBenchmark(changed))
    assert not failed.passed
    assert (
        f"adversarial-escape:{target.adversarial_fixture.value}"
        in failed.failure_codes
    )


def test_missing_fixture_and_imprecise_invalidation_cannot_be_averaged_away():
    benchmark = build_frozen_decision_runtime_benchmark()
    omitted = DecisionRuntimeBenchmark(
        tuple(
            receipt
            for receipt in benchmark.receipts
            if receipt.adversarial_fixture
            is not REQUIRED_ADVERSARIAL_FIXTURES[0]
        )
    )
    report = recompute_proof_dependency_scaling(omitted)
    assert not report.passed
    assert any(code.startswith("missing-adversarial-fixture:") for code in report.failure_codes)

    target = next(
        receipt
        for receipt in benchmark.receipts
        if receipt.adversarial_fixture is None
    )
    metrics = replace(target.metrics, invalidation_actual=2)
    imprecise = DecisionRuntimeBenchmark(
        tuple(
            replace(receipt, metrics=metrics)
            if receipt.receipt_id == target.receipt_id
            else receipt
            for receipt in benchmark.receipts
        )
    )
    report = recompute_proof_dependency_scaling(imprecise)
    assert not report.invalidation_precision_passed
    assert "imprecise-invalidation" in report.failure_codes

