"""Reviewed numerical candidate for the public small real eigenpair contract.

This module does not invoke a provider, interpret task-selected code, prove a
floating-point eigensolver correct, or certify a speed improvement. The emitted
candidate is standalone. SciPy is an explicit dependency of that candidate;
the optional NumPy path is recorded as an unoptimized baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
import textwrap

import numpy as np


SCIPY_BACKEND = "scipy_dgeev"
NUMPY_BACKEND = "numpy_eig"

_INPUT_SOURCE = '''
def _check_input(A):
    if not isinstance(A, np.ndarray):
        raise TypeError("a NumPy array is required")
    if A.ndim != 2 or A.shape[0] != A.shape[1] or not 1 <= A.shape[0] <= 10:
        raise ValueError("a square matrix of size 1 through 10 is required")
    if A.dtype.kind != "f" or A.dtype.itemsize != 8:
        raise TypeError("a real float64 matrix is required")
    magnitude = float(np.abs(A).max())
    if not np.isfinite(magnitude):
        raise ValueError("finite matrix entries are required")
    return magnitude
'''

_SCIPY_SOURCE = '''
from scipy.linalg.lapack import dgeev as _dgeev, dgeev_lwork as _dgeev_lwork

# Workspace queries are setup costs, performed once for the bounded domain.
_WORKSPACE = {}
for _n in range(1, 11):
    _work, _info = _dgeev_lwork(_n, compute_vl=0, compute_vr=1)
    if _info != 0 or not np.isfinite(_work) or _work < 4 * _n:
        raise np.linalg.LinAlgError("DGEEV workspace query failed")
    _WORKSPACE[_n] = int(_work)


def _largest_modulus_index(wr, wi):
    # Hypot avoids squaring tiny/large components. Scaling also handles a
    # modulus outside float64 when both complex components are still finite.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        moduli = np.hypot(wr, wi)
    j = int(moduli.argmax())
    if not np.isfinite(moduli[j]):
        if not (np.isfinite(wr).all() and np.isfinite(wi).all()):
            raise np.linalg.LinAlgError("DGEEV returned nonfinite eigenvalues")
        scale = max(float(np.abs(wr).max()), float(np.abs(wi).max()))
        j = int(np.hypot(wr / scale, wi / scale).argmax())
    return j


def _eigenpair_from_real_geev(wr, wi, vr, j):
    if wi[j] == 0:
        return wr[j], vr[:, j]
    if wi[j] > 0:
        if j + 1 >= len(wr):
            raise np.linalg.LinAlgError("DGEEV complex pair layout is invalid")
        vector = vr[:, j].astype(np.complex128)
        vector.imag = vr[:, j + 1]
    else:
        if j == 0:
            raise np.linalg.LinAlgError("DGEEV complex pair layout is invalid")
        vector = vr[:, j - 1].astype(np.complex128)
        vector.imag = -vr[:, j]
    return np.complex128(complex(wr[j], wi[j])), vector


def find_dominant_eigenvalue_and_eigenvector(A: np.ndarray):
    """Return a computed maximum-modulus right eigenpair, preserving A.

    Domain: finite real float64 NumPy square matrices of size 1 through 10.
    Ties may select either dominant eigenvalue. Convergence and representable
    results remain numerical conditions; this is not an exact root certificate.
    """
    magnitude = _check_input(A)
    n = A.shape[0]
    if n == 1:
        return np.float64(A[0, 0]), np.ones(1, dtype=np.float64)
    # overwrite_a applies only to this owned copy, including readonly inputs.
    owned = np.array(A, dtype=np.float64, order="F", copy=True)
    scale = magnitude if magnitude != 0 and (magnitude < 1e-100 or magnitude > 1e100) else 1.0
    if scale != 1.0:
        # Explicit scaling also avoids relying on vendor LAPACK's extreme
        # input rescaling path. Positive common scaling preserves ordering.
        owned /= scale
    wr, wi, _vl, vr, info = _dgeev(
        owned, compute_vl=0, compute_vr=1, lwork=_WORKSPACE[n], overwrite_a=1
    )
    if info != 0:
        raise np.linalg.LinAlgError("DGEEV failed with info=" + str(info))
    j = _largest_modulus_index(wr, wi)
    eigenvalue, eigenvector = _eigenpair_from_real_geev(wr, wi, vr, j)
    if scale != 1.0:
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            eigenvalue = eigenvalue * scale
    if not np.isfinite(eigenvalue):
        raise np.linalg.LinAlgError("DGEEV eigenvalue is not representable")
    if not np.isfinite(eigenvector).all() or not eigenvector.any():
        raise np.linalg.LinAlgError("DGEEV returned an invalid eigenvector")
    return eigenvalue, eigenvector
'''

_NUMPY_SOURCE = '''
def find_dominant_eigenvalue_and_eigenvector(A: np.ndarray):
    """Unoptimized public NumPy reference with explicit domain validation."""
    _check_input(A)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    idx = np.argmax(np.abs(eigenvalues))
    return eigenvalues[idx], eigenvectors[:, idx]
'''


def candidate_source(backend: str = SCIPY_BACKEND) -> str:
    """Return fixed reviewed standalone eigen.py text, never caller code."""
    if backend not in (SCIPY_BACKEND, NUMPY_BACKEND):
        raise ValueError("unsupported spectral backend")
    body = _SCIPY_SOURCE if backend == SCIPY_BACKEND else _NUMPY_SOURCE
    return "import numpy as np\n\n" + textwrap.dedent(_INPUT_SOURCE).lstrip() + "\n" + textwrap.dedent(body).lstrip()


@lru_cache(maxsize=2)
def _candidate_namespace(backend: str) -> dict:
    namespace: dict = {"__name__": "reviewed_spectral_candidate"}
    exec(compile(candidate_source(backend), "<reviewed-spectral-candidate>", "exec"), namespace)
    return namespace


@dataclass(frozen=True)
class EigenpairResult:
    eigenvalue: np.generic
    eigenvector: np.ndarray
    backend: str
    fallback_reason: str | None
    optimization_candidate: bool


def solve_dominant_eigenpair(
    A: np.ndarray, *, backend: str = SCIPY_BACKEND, allow_numpy_fallback: bool = False
) -> EigenpairResult:
    """Compute using trusted source; fallback is explicit and unoptimized."""
    fallback_reason = None
    try:
        namespace = _candidate_namespace(backend)
    except ImportError:
        if backend != SCIPY_BACKEND or not allow_numpy_fallback:
            raise
        backend, fallback_reason = NUMPY_BACKEND, "scipy_unavailable"
        namespace = _candidate_namespace(backend)
    eigenvalue, eigenvector = namespace["find_dominant_eigenvalue_and_eigenvector"](A)
    return EigenpairResult(eigenvalue, eigenvector, backend, fallback_reason, backend == SCIPY_BACKEND)


def find_dominant_eigenvalue_and_eigenvector(A: np.ndarray):
    """Runtime pair entry point using the same emitted reviewed implementation."""
    result = solve_dominant_eigenpair(A)
    return result.eigenvalue, result.eigenvector


@dataclass(frozen=True)
class NumericQualification:
    accepted: bool
    residual_passed: bool
    dominant_passed: bool
    nonzero_vector: bool
    reference_backend: str
    reason_code: str
    reference_max_scaled_modulus: float | None = None
    candidate_scaled_modulus: float | None = None
    modulus_scale: float | None = None
    normalized_residual_passed: bool = False
    spectrum_membership_passed: bool = False
    raw_allclose_passed: bool = False


def validate_eigenpair(A: np.ndarray, eigenvalue, eigenvector) -> NumericQualification:
    """Independent spectrum, normalized residual and public allclose check.

    Numerical dominance uses relative tolerance 1e-7 with zero absolute
    tolerance in a scaled representation. This is an empirical cross-check,
    not a proof about exact eigenvalues of a nonnormal floating-point matrix.
    """
    def report(reason, residual=False, dominant=False, nonzero=False,
               normalized_residual=False, spectrum_member=False, raw_allclose=False, **metrics):
        return NumericQualification(
            residual and normalized_residual and spectrum_member and dominant and nonzero,
            residual, dominant, nonzero, "numpy.linalg.eig", reason,
            normalized_residual_passed=normalized_residual,
            spectrum_membership_passed=spectrum_member,
            raw_allclose_passed=raw_allclose, **metrics
        )
    # Independently written checks; the candidate's input/selection helpers
    # are deliberately not used by this checker.
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        return report("input_invalid")
    n = A.shape[0]
    if A.shape != (n, n) or not 1 <= n <= 10 or A.dtype.kind != "f" or A.dtype.itemsize != 8:
        return report("input_invalid")
    if not np.isfinite(A).all():
        return report("input_nonfinite")
    try:
        value_array = np.asarray(eigenvalue)
        vector = np.asarray(eigenvector)
        if value_array.ndim != 0 or not np.isfinite(value_array).all():
            return report("eigenvalue_invalid")
        if vector.shape != (n,) or not np.isfinite(vector).all():
            return report("eigenvector_invalid")
        value = complex(value_array.item())
        nonzero = bool(np.any(vector != 0))
        if not nonzero:
            return report("zero_eigenvector")
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            left, right = A @ vector, value * vector
        # Infinite residual products must not pass allclose(inf, inf).
        raw_allclose = bool(np.allclose(left, right))
        residual = bool(np.isfinite(left).all() and np.isfinite(right).all() and raw_allclose)
        # A tiny nonzero vector can make an incorrect pair pass the public
        # absolute residual tolerance. Normalize the vector by real/imaginary
        # components first, avoiding abs(complex) overflow or reciprocal
        # overflow when the scale is subnormal.
        vector_scale = max(float(np.abs(vector.real).max()), float(np.abs(vector.imag).max()))
        scaled_vector = (vector.real / vector_scale).astype(np.complex128)
        scaled_vector.imag = vector.imag / vector_scale
        unit_vector = scaled_vector / math.sqrt(float(np.vdot(scaled_vector, scaled_vector).real))
        residual_scale = max(float(np.abs(A).max()), abs(value.real), abs(value.imag))
        if residual_scale == 0:
            normalized_residual = True
        else:
            normalized_matrix = np.array(A, dtype=np.float64, copy=True) / residual_scale
            normalized_value = complex(value.real / residual_scale, value.imag / residual_scale)
            normalized_left = normalized_matrix @ unit_vector
            normalized_right = normalized_value * unit_vector
            normalized_residual = bool(np.isfinite(normalized_left).all() and np.isfinite(normalized_right).all()
                                       and np.allclose(normalized_left, normalized_right))
        reference, _vectors = np.linalg.eig(np.array(A, dtype=np.float64, copy=True))
        if not np.isfinite(reference).all():
            return report("reference_nonfinite", residual=residual, nonzero=nonzero,
                          normalized_residual=normalized_residual, raw_allclose=raw_allclose)
        components = [abs(value.real), abs(value.imag)]
        components.extend(abs(complex(item).real) for item in reference)
        components.extend(abs(complex(item).imag) for item in reference)
        scale = max(components)
        if scale == 0:
            candidate_modulus = reference_modulus = 0.0
        else:
            # math.hypot, independent from the candidate's vectorized selector.
            candidate_modulus = math.hypot(value.real / scale, value.imag / scale)
            reference_modulus = max(math.hypot(complex(item).real / scale, complex(item).imag / scale) for item in reference)
        dominant = math.isclose(candidate_modulus, reference_modulus, rel_tol=1e-7, abs_tol=0.0)
        # Membership retains sign and phase, unlike modulus dominance. Use a
        # separate pair scale for each reference root so small spectral roots
        # are not erased by an unrelated very large root. Zero absolute
        # tolerance prevents a tiny wrong-sign eigenvalue from being admitted.
        spectrum_member = False
        for item in reference:
            item = complex(item)
            pair_scale = max(abs(value.real), abs(value.imag), abs(item.real), abs(item.imag))
            if pair_scale == 0:
                spectrum_member = True
                break
            candidate_root = complex(value.real / pair_scale, value.imag / pair_scale)
            reference_root = complex(item.real / pair_scale, item.imag / pair_scale)
            difference = math.hypot(candidate_root.real - reference_root.real, candidate_root.imag - reference_root.imag)
            reference_size = math.hypot(reference_root.real, reference_root.imag)
            if difference <= 1e-7 * reference_size:
                spectrum_member = True
                break
        reason = ("residual_failed" if not residual else "normalized_residual_failed" if not normalized_residual
                  else "spectrum_membership_failed" if not spectrum_member else "nondominant_eigenpair" if not dominant
                  else "accepted")
        return report(reason, residual=residual, dominant=dominant, nonzero=nonzero,
                      normalized_residual=normalized_residual, spectrum_member=spectrum_member,
                      raw_allclose=raw_allclose,
                      reference_max_scaled_modulus=reference_modulus,
                      candidate_scaled_modulus=candidate_modulus, modulus_scale=scale)
    except (TypeError, ValueError, np.linalg.LinAlgError, OverflowError):
        return report("numerical_check_failed")


def _qualification_cases():
    rng = np.random.default_rng(270710)
    cases = [("general_" + str(n), rng.normal(size=(n, n)).astype(np.float64)) for n in range(1, 11)]
    cases.extend([
        ("rotation", np.array([[0., -2.], [2., 0.]])),
        ("negative_dominant", np.diag([-3., 2.])),
        ("equal_modulus_tie", np.diag([-3., 3.])),
        ("zero", np.zeros((4, 4))),
        ("defective_jordan", np.array([[2., 1., 0.], [0., 2., 1.], [0., 0., 2.]])),
        ("nonnormal", np.array([[1., 1000., -2000.], [0., -2., 500.], [0., 0., 3.]])),
        ("large_rotation", np.array([[0., -2.], [2., 0.]]) * 1e150),
        ("small_rotation", np.array([[0., -2.], [2., 0.]]) * 1e-150),
    ])
    base = rng.normal(size=(5, 5)).astype(np.float64)
    readonly = base.copy()
    readonly.flags.writeable = False
    cases.extend([("fortran_layout", np.asfortranarray(base)), ("transpose_layout", base.T),
                  ("negative_stride_layout", base[::-1, ::-1]), ("readonly_layout", readonly)])
    return cases


def qualify_spectral_kernel(backend: str = SCIPY_BACKEND) -> dict:
    """Run fixed authored local fixtures against only the reviewed candidate.

    Availability belongs to this interpreter. No task-selected source, target
    interpreter, hidden tests, benchmark harness or model is executed here.
    """
    source = candidate_source(backend)
    receipt = {
        "schema": "spectral-numeric-qualification@1", "backend": backend,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "available": False, "accepted": False, "case_count": 0, "cases": [],
        "qualification_scope": "authored_local_numerical_only",
        "optimization_candidate": backend == SCIPY_BACKEND,
        "speed_improvement_qualified": False, "exact_spectral_certificate": False,
    }
    try:
        namespace = _candidate_namespace(backend)
    except ImportError:
        receipt["reason_code"] = "dependency_unavailable"
        return receipt
    except (ValueError, np.linalg.LinAlgError):
        receipt["reason_code"] = "kernel_setup_failed"
        return receipt
    receipt["available"] = True
    function = namespace["find_dominant_eigenvalue_and_eigenvector"]
    for case_id, matrix in _qualification_cases():
        original = matrix.tobytes()
        try:
            value, vector = function(matrix)
            checked = validate_eigenpair(matrix, value, vector)
            unchanged = matrix.tobytes() == original
            accepted = checked.accepted and unchanged
            reason = checked.reason_code if unchanged else "input_mutated"
        except (TypeError, ValueError, np.linalg.LinAlgError):
            accepted, reason, unchanged = False, "kernel_execution_failed", matrix.tobytes() == original
        receipt["cases"].append({"case_id": case_id, "accepted": accepted, "reason_code": reason, "input_unchanged": unchanged})
    receipt["case_count"] = len(receipt["cases"])
    receipt["accepted"] = all(case["accepted"] for case in receipt["cases"])
    receipt["reason_code"] = "accepted" if receipt["accepted"] else "numeric_qualification_failed"
    return receipt
