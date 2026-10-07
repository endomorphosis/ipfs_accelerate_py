"""Scoped algebra checks and actual isolated target checks, without providers."""
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.spectral_kernel_proof import prove_spectral_adapter_lemmas
from ipfs_accelerate_py.agent_supervisor.runtime import spectral_target_qualification as target
from ipfs_accelerate_py.agent_supervisor.runtime.spectral_eigen_kernel import candidate_source


def test_conditional_algebra_verified_with_explicit_limits():
    pytest.importorskip("z3")
    digest = hashlib.sha256(candidate_source().encode()).hexdigest()
    report = prove_spectral_adapter_lemmas(kernel_source_sha256=digest)
    assert report["all_lemmas_verified"] and len(report["checks"]) == 13
    assert report["kernel_source_sha256"] == digest
    assert report["assumptions"]
    for field in ("python_program_mechanically_verified", "lapack_implementation_proved",
                  "ieee_floating_point_accuracy_proved", "timing_proved", "kernel_proved",
                  "whole_program_proved", "proof_authority", "completion_authority"):
        assert report[field] is False


def test_order_by_real_value_has_symbolic_counterexample():
    z3 = pytest.importorskip("z3")
    selected, rejected = z3.Reals("selected rejected")
    solver = z3.Solver()
    solver.add(selected == 2, rejected == -3, selected > rejected,
               selected * selected < rejected * rejected)
    assert solver.check() == z3.sat


def test_zero_vector_has_residual_without_an_eigenpair():
    z3 = pytest.importorskip("z3")
    value, vector, applied = z3.Reals("value vector applied")
    solver = z3.Solver()
    solver.add(vector == 0, applied == 2 * vector, value == 999, applied == value * vector)
    assert solver.check() == z3.sat


def test_real_target_uses_isolation_and_reports_actual_dependencies():
    report = target.qualify_spectral_target_runtime(interpreter=Path(sys.executable))
    if report["reason_code"] == "dependency_unavailable":
        pytest.skip("isolated interpreter lacks numerical dependencies")
    assert report["accepted"] and report["case_count"] == 22
    assert report["source_sha256"] == hashlib.sha256(candidate_source().encode()).hexdigest()
    assert report["numpy_version"] and report["scipy_version"]
    assert report["provider_calls"] == 0 and report["timing_qualified"] is False
    assert report["kernel_proved"] is False and report["completion_authority"] is False


def test_missing_target_is_residual(tmp_path):
    report = target.qualify_spectral_target_runtime(interpreter=tmp_path / "missing-python")
    assert not report["accepted"]


def test_timeout_is_residual(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(target.subprocess, "run", timeout)
    report = target.qualify_spectral_target_runtime(interpreter=Path(sys.executable))
    assert not report["accepted"] and report["reason_code"] == "target_probe_timeout"


def test_affirmative_wrong_source_report_is_rejected(monkeypatch):
    monkeypatch.setattr(target.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0,
        stdout='{"accepted":true,"source_sha256":"wrong","python_version":"3.12",'
               '"numpy_version":"1","scipy_version":"1","case_count":22,'
               '"reason_code":"accepted_finite_fixture_checks"}'))
    report = target.qualify_spectral_target_runtime(interpreter=Path(sys.executable))
    assert not report["accepted"] and report["reason_code"] == "target_probe_report_invalid"


@pytest.mark.parametrize("error", ["wrong_sign", "tiny_wrong_vector"])
def test_actual_target_probe_rejects_dominant_modulus_with_invalid_pair(monkeypatch, error):
    from ipfs_accelerate_py.agent_supervisor.runtime import spectral_eigen_kernel as kernel
    source = "import numpy as np\ndef find_dominant_eigenvalue_and_eigenvector(A):\n" \
             "    values, vectors = np.linalg.eig(A)\n    j = np.abs(values).argmax()\n"
    if error == "wrong_sign":
        source += "    return -values[j], vectors[:, j] * 1e-100\n"
    else:
        source += "    return values[j], np.ones(len(A)) * 1e-100\n"
    monkeypatch.setattr(kernel, "candidate_source", lambda backend: source)
    report = target.qualify_spectral_target_runtime(interpreter=Path(sys.executable))
    if report["reason_code"] == "dependency_unavailable":
        pytest.skip("isolated target lacks dependencies")
    assert not report["accepted"] and report["reason_code"] == "numerical_qualification_failed"
