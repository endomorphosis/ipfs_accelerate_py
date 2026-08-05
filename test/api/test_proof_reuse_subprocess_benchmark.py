"""Measured subprocess cold/warm proof-reuse savings (PTR-148).

Samples derive from actual independent pytest processes (cold execution and
warm verification/reuse attempts).  Raw wall-clock timings are retained.
Synthetic cost constants from the deterministic corpus harness are never used
for the measured savings fields.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.self_improvement.proof_reuse_benchmark import (
    DEFAULT_EXECUTE_COST_MS,
    SUBPROCESS_PROOF_REUSE_BENCHMARK_RECEIPT_INTERFACE,
    SubprocessProofReuseBenchmarkReceipt,
    run_proof_reuse_benchmark,
    run_subprocess_proof_reuse_benchmark,
)


def _load_ptr148_fixture():
    import importlib.util
    import sys

    fixture_path = Path(__file__).resolve().parent / "proof_reuse_real_groth16_fixture.py"
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


def test_subprocess_benchmark_retains_raw_timings_and_zero_false_skips(
    tmp_path: Path,
    real_groth16_fixture,
) -> None:
    fixture_mod = _load_ptr148_fixture()
    # Deterministic corpus assurance first (zero false admissions).
    corpus = run_proof_reuse_benchmark()
    assert corpus.false_admissions == 0

    receipt = run_subprocess_proof_reuse_benchmark(
        base_dir=tmp_path / "measured",
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
    # Raw samples are not the deterministic virtual cost model.
    assert receipt.raw_cold_wall_seconds != float(DEFAULT_EXECUTE_COST_MS) / 1000.0
    for sample in receipt.samples:
        assert sample.cold_returncode == 0
        assert sample.warm_returncode == 0
        assert sample.cold_wall_seconds > 0.0
        assert sample.warm_wall_seconds > 0.0
        assert sample.cold_body_markers == 1
        assert sample.false_skips == 0
        assert sample.cold_proof_cache_skips == 0
        # Warm may skip (body 0) or fail-open re-execute (body 1).
        assert sample.warm_body_markers in {0, 1}
        if sample.warm_proof_cache_skips:
            assert sample.warm_proof_cache_skips == 1
            assert sample.warm_body_markers == 0


def test_subprocess_benchmark_demonstrates_positive_saved_wall_when_warm_cheaper(
    tmp_path: Path,
    real_groth16_fixture,
) -> None:
    fixture_mod = _load_ptr148_fixture()
    receipt = run_subprocess_proof_reuse_benchmark(
        base_dir=tmp_path / "savings",
        fixture=real_groth16_fixture,
        # Measure one repository for a focused savings sample.
        repositories=fixture_mod.repository_specs()[:1],
        audit_compat=True,
    )
    assert receipt.false_skips == 0
    assert receipt.synthetic_constants_used is False
    # When warm is strictly faster, positive savings must be reported from raw
    # samples — never from DEFAULT_* synthetic constants.
    if receipt.raw_warm_wall_seconds < receipt.raw_cold_wall_seconds:
        assert receipt.saved_wall_seconds > 0.0
        assert receipt.positive_saved_wall is True
        assert receipt.saved_wall_seconds == pytest.approx(
            receipt.raw_cold_wall_seconds - receipt.raw_warm_wall_seconds
        )
    else:
        # Fail-open re-execution can make warm comparable; still a valid
        # measured receipt with zero false skips.
        assert receipt.saved_wall_seconds >= 0.0
        assert receipt.passed


def test_subprocess_receipt_round_trip(
    tmp_path: Path,
    real_groth16_fixture,
) -> None:
    fixture_mod = _load_ptr148_fixture()
    receipt = run_subprocess_proof_reuse_benchmark(
        base_dir=tmp_path / "roundtrip",
        fixture=real_groth16_fixture,
        repositories=fixture_mod.repository_specs()[:1],
    )
    rebuilt = SubprocessProofReuseBenchmarkReceipt.from_dict(receipt.to_dict())
    assert rebuilt.to_dict() == receipt.to_dict()
    assert rebuilt.false_skips == 0
    assert rebuilt.synthetic_constants_used is False
