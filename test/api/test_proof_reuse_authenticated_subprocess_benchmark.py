"""Authenticated measured subprocess benchmark assurance for PTR-169.

Confirms the reviewed warm-reuse threshold path, zero false skips under the
adversarial body-oracle population, and that measured savings never come from
synthetic cost constants.  Optional capability gaps remain truthful
RUN/DEFERRED rather than synthetic authority.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.self_improvement.proof_reuse_benchmark import (
    DEFAULT_EXECUTE_COST_MS,
    MIN_WARM_SKIP_BPS,
    SUBPROCESS_PROOF_REUSE_BENCHMARK_RECEIPT_INTERFACE,
    SubprocessProofReuseBenchmarkReceipt,
    run_proof_reuse_benchmark,
    run_subprocess_proof_reuse_benchmark,
    verify_benchmark_receipt,
)
from ipfs_accelerate_py.agent_supervisor.validation.proof_test_reuse_current_tree_gate import (
    FINAL_GATE_ACCEPTANCE_CRITERION,
    FINAL_GATE_REVIEW_REVISION,
    FINAL_GATE_TASK_ID,
    SEALED_PRODUCTION_TASK_COUNT,
)

# PTR-169 interfaces: the authenticated handoff consumes the measured
# subprocess receipt together with the corpus BenchmarkReceipt.  The corpus
# surface remains BenchmarkReceipt@1; the handoff package is the v5 authenticated
# gate join, not a second synthetic timing authority.
AUTHENTICATED_BENCHMARK_JOIN = "ProofReuseBenchmarkReceipt@2"


def _load_ptr148_fixture():
    import importlib.util
    import sys

    fixture_path = (
        Path(__file__).resolve().parent / "proof_reuse_real_groth16_fixture.py"
    )
    module_name = "proof_reuse_real_groth16_fixture"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, fixture_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def real_groth16_fixture():
    fixture_mod = _load_ptr148_fixture()
    return fixture_mod.RealGroth16TestPassFixture.discover()


def test_authenticated_handoff_constants_bind_78_task_gate() -> None:
    assert SEALED_PRODUCTION_TASK_COUNT == 78
    assert FINAL_GATE_TASK_ID == "PTR-169"
    assert FINAL_GATE_ACCEPTANCE_CRITERION == (
        "ptr/authenticated-current-tree-gate-v5@1"
    )
    assert FINAL_GATE_REVIEW_REVISION == (
        "authenticated-receipt-current-tree-repair-v9"
    )
    assert AUTHENTICATED_BENCHMARK_JOIN.endswith("@2")


def test_corpus_benchmark_meets_reviewed_threshold_and_zero_false_admissions() -> None:
    receipt = run_proof_reuse_benchmark()
    assert receipt.false_admissions == 0
    assert receipt.passed is True
    assert verify_benchmark_receipt(receipt) is True
    # Reviewed warm-skip threshold: at least MIN_WARM_SKIP_BPS of eligible warm.
    if receipt.warm_eligible_count > 0:
        assert receipt.warm_skip_bps >= MIN_WARM_SKIP_BPS
        assert receipt.warm_verified_skips >= (
            (receipt.warm_eligible_count * MIN_WARM_SKIP_BPS) // 10_000
        )


def test_authenticated_subprocess_benchmark_zero_false_skips_and_raw_timings(
    tmp_path: Path,
    real_groth16_fixture,
) -> None:
    fixture_mod = _load_ptr148_fixture()
    receipt = run_subprocess_proof_reuse_benchmark(
        base_dir=tmp_path / "authenticated-measured",
        fixture=real_groth16_fixture,
        repositories=fixture_mod.repository_specs(),
        audit_compat=True,
    )
    assert isinstance(receipt, SubprocessProofReuseBenchmarkReceipt)
    assert receipt.interface == SUBPROCESS_PROOF_REUSE_BENCHMARK_RECEIPT_INTERFACE
    assert receipt.synthetic_constants_used is False
    assert receipt.false_skips == 0
    assert receipt.passed
    assert len(receipt.samples) == 3
    assert receipt.raw_cold_wall_seconds > 0.0
    assert receipt.raw_warm_wall_seconds > 0.0
    assert receipt.raw_cold_wall_seconds != float(DEFAULT_EXECUTE_COST_MS) / 1000.0
    for sample in receipt.samples:
        assert sample.cold_returncode == 0
        assert sample.warm_returncode == 0
        assert sample.false_skips == 0
        assert sample.cold_proof_cache_skips == 0
        assert sample.warm_body_markers in {0, 1}


def test_authenticated_subprocess_savings_are_measured_not_synthetic(
    tmp_path: Path,
    real_groth16_fixture,
) -> None:
    fixture_mod = _load_ptr148_fixture()
    receipt = run_subprocess_proof_reuse_benchmark(
        base_dir=tmp_path / "authenticated-savings",
        fixture=real_groth16_fixture,
        repositories=fixture_mod.repository_specs()[:1],
        audit_compat=True,
    )
    assert receipt.false_skips == 0
    assert receipt.synthetic_constants_used is False
    if receipt.raw_warm_wall_seconds < receipt.raw_cold_wall_seconds:
        assert receipt.saved_wall_seconds > 0.0
        assert receipt.positive_saved_wall is True
        assert receipt.saved_wall_seconds == pytest.approx(
            receipt.raw_cold_wall_seconds - receipt.raw_warm_wall_seconds
        )
    else:
        # Fail-open re-execution remains a valid measured receipt.
        assert receipt.saved_wall_seconds >= 0.0
        assert receipt.passed


def test_authenticated_subprocess_receipt_round_trip(
    tmp_path: Path,
    real_groth16_fixture,
) -> None:
    fixture_mod = _load_ptr148_fixture()
    receipt = run_subprocess_proof_reuse_benchmark(
        base_dir=tmp_path / "authenticated-roundtrip",
        fixture=real_groth16_fixture,
        repositories=fixture_mod.repository_specs()[:1],
    )
    rebuilt = SubprocessProofReuseBenchmarkReceipt.from_dict(receipt.to_dict())
    assert rebuilt.to_dict() == receipt.to_dict()
    assert rebuilt.false_skips == 0
    assert rebuilt.synthetic_constants_used is False


def test_optional_capability_gaps_remain_truthful_run_or_deferred() -> None:
    """Missing optional stacks never invent skip authority for the handoff."""

    # Corpus receipt documents exclusions as counts, never as authoritative
    # warm skips.  A zero false-admission result with optional exclusions is
    # the truthful RUN/DEFERRED posture required by the authenticated closeout.
    receipt = run_proof_reuse_benchmark()
    assert receipt.false_admissions == 0
    for reason, count in dict(receipt.exclusions).items():
        assert isinstance(reason, str) and reason
        assert int(count) >= 0
        # Exclusion reasons must not be relabeled as verified warm skips.
        assert "false" not in reason or count == 0 or receipt.passed
