"""Authored public-domain checks; no Terminal Bench evaluation inputs."""
import builtins
import hashlib

import numpy as np
import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import spectral_eigen_kernel as kernel


def compiled_candidate(backend=kernel.SCIPY_BACKEND):
    namespace = {"__name__": "authored_candidate_test"}
    exec(compile(kernel.candidate_source(backend), "authored_eigen.py", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("n", range(1, 11))
def test_actual_emitted_candidate_general_sizes_and_immutability(n):
    rng = np.random.default_rng(700 + n)
    matrix = rng.normal(size=(n, n)).astype(np.float64)
    original = matrix.tobytes()
    value, vector = compiled_candidate()["find_dominant_eigenvalue_and_eigenvector"](matrix)
    checked = kernel.validate_eigenpair(matrix, value, vector)
    assert checked.accepted, checked
    assert matrix.tobytes() == original
    assert vector.shape == (n,)
    assert isinstance(value, np.generic)


@pytest.mark.parametrize("matrix,expected_modulus", [
    (np.array([[0., -2.], [2., 0.]]), 2.),
    (np.diag([-3., 2.]), 3.),
    (np.diag([-3., 3.]), 3.),
    (np.zeros((3, 3)), 0.),
    (np.eye(3), 1.),
    (np.array([[2., 1., 0.], [0., 2., 1.], [0., 0., 2.]]), 2.),
    (np.array([[1., 1000., -2000.], [0., -2., 500.], [0., 0., 3.]]), 3.),
])
def test_analytic_fixtures_are_dominant_nonzero(matrix, expected_modulus):
    result = kernel.solve_dominant_eigenpair(matrix)
    assert abs(result.eigenvalue) == pytest.approx(expected_modulus)
    assert np.any(result.eigenvector)
    assert kernel.validate_eigenpair(matrix, result.eigenvalue, result.eigenvector).accepted
    assert result.backend == kernel.SCIPY_BACKEND
    assert result.fallback_reason is None
    assert result.optimization_candidate


@pytest.mark.parametrize("scale", [1e150, 1e-150, 1e-300])
def test_scaled_general_and_complex_matrices(scale):
    for matrix in (np.diag([-3., 2.]), np.array([[0., -2.], [2., 0.]])):
        matrix = matrix * scale
        value, vector = kernel.find_dominant_eigenvalue_and_eigenvector(matrix)
        assert abs(value / scale) == pytest.approx(3. if matrix[0, 0] else 2.)
        assert kernel.validate_eigenpair(matrix, value, vector).accepted


@pytest.mark.parametrize("layout", ["C", "F", "transpose", "negative_stride", "slice", "readonly", "big_endian"])
def test_input_layouts_are_preserved(layout):
    rng = np.random.default_rng(14)
    base = rng.normal(size=(4, 4)).astype(np.float64)
    if layout == "F":
        matrix = np.asfortranarray(base)
    elif layout == "transpose":
        matrix = base.T
    elif layout == "negative_stride":
        matrix = base[::-1, ::-1]
    elif layout == "slice":
        backing = np.zeros((8, 8))
        backing[::2, ::2] = base
        matrix = backing[::2, ::2]
    elif layout == "big_endian":
        matrix = base.astype(">f8")
    else:
        matrix = base
        if layout == "readonly":
            matrix.flags.writeable = False
    original = matrix.tobytes()
    value, vector = kernel.find_dominant_eigenvalue_and_eigenvector(matrix)
    assert kernel.validate_eigenpair(matrix, value, vector).accepted
    assert matrix.tobytes() == original
    assert not np.shares_memory(vector, matrix)


@pytest.mark.parametrize("matrix,error", [
    ([[1.]], TypeError),
    (np.ones((2, 2), dtype=np.float32), TypeError),
    (np.eye(2, dtype=np.complex128), TypeError),
    (np.ones(2), ValueError),
    (np.ones((2, 3)), ValueError),
    (np.empty((0, 0)), ValueError),
    (np.eye(11), ValueError),
    (np.array([[np.nan]]), ValueError),
    (np.array([[np.inf]]), ValueError),
])
def test_explicit_finite_real_nonempty_bounded_precondition(matrix, error):
    with pytest.raises(error):
        kernel.find_dominant_eigenvalue_and_eigenvector(matrix)


def test_positive_and_negative_conjugate_reconstruction():
    namespace = compiled_candidate()
    # Real packing for A=[[0,-2],[2,0]]: p=(1,0), q=(0,-1).
    wr, wi = np.zeros(2), np.array([2., -2.])
    vr = np.array([[1., 0.], [0., -1.]])
    matrix = np.array([[0., -2.], [2., 0.]])
    positive = namespace["_eigenpair_from_real_geev"](wr, wi, vr, 0)
    negative = namespace["_eigenpair_from_real_geev"](wr, wi, vr, 1)
    assert positive[0] == 2j
    assert negative[0] == -2j
    assert np.array_equal(negative[1], np.conjugate(positive[1]))
    assert kernel.validate_eigenpair(matrix, *positive).accepted
    assert kernel.validate_eigenpair(matrix, *negative).accepted


def test_hypot_selection_avoids_naive_square_overflow_and_underflow():
    select = compiled_candidate()["_largest_modulus_index"]
    assert select(np.array([1e200, 2e200]), np.zeros(2)) == 1
    assert select(np.array([1e-300, 2e-300]), np.zeros(2)) == 1
    assert select(np.array([1.3e308, 1.7e308]), np.array([1.3e308, 0.])) == 0
    with pytest.raises(np.linalg.LinAlgError, match="nonfinite"):
        select(np.array([1., np.inf]), np.zeros(2))


def test_finite_complex_components_with_unrepresentable_modulus():
    magnitude = 1.3e308
    matrix = np.array([[magnitude, -magnitude], [magnitude, magnitude]])
    value, vector = kernel.find_dominant_eigenvalue_and_eigenvector(matrix)
    assert np.isfinite(value)
    assert value.real / magnitude == pytest.approx(1.)
    assert abs(value.imag / magnitude) == pytest.approx(1.)
    checked = kernel.validate_eigenpair(matrix, value, vector)
    assert checked.accepted
    assert checked.candidate_scaled_modulus == pytest.approx(2. ** .5)


@pytest.mark.parametrize("info", [-1, 1, 2])
def test_dgeev_info_failure_is_not_accepted(info):
    namespace = compiled_candidate()
    namespace["_dgeev"] = lambda *args, **kwargs: (np.array([1., 2.]), np.zeros(2), None, np.eye(2), info)
    with pytest.raises(np.linalg.LinAlgError, match="info=" + str(info)):
        namespace["find_dominant_eigenvalue_and_eigenvector"](np.diag([1., 2.]))


def test_dgeev_zero_or_nonfinite_vector_is_not_accepted():
    for vectors in (np.zeros((2, 2)), np.full((2, 2), np.nan)):
        namespace = compiled_candidate()
        namespace["_dgeev"] = lambda *args, **kwargs: (np.array([1., 2.]), np.zeros(2), None, vectors, 0)
        with pytest.raises(np.linalg.LinAlgError, match="invalid eigenvector"):
            namespace["find_dominant_eigenvalue_and_eigenvector"](np.diag([1., 2.]))


def test_independent_checker_rejects_nondominant_and_zero_vector():
    matrix = np.diag([1., 2.])
    nondominant = kernel.validate_eigenpair(matrix, 1., np.array([1., 0.]))
    assert nondominant.residual_passed
    assert not nondominant.dominant_passed
    assert nondominant.reason_code == "nondominant_eigenpair"
    zero = kernel.validate_eigenpair(matrix, 2., np.zeros(2))
    assert not zero.accepted
    assert not zero.nonzero_vector
    assert zero.reason_code == "zero_eigenvector"
    tiny = kernel.validate_eigenpair(matrix * 1e-300, 1e-300, np.array([1., 0.]))
    assert tiny.residual_passed
    assert not tiny.dominant_passed


def test_independent_checker_residual_and_shape_failures():
    matrix = np.diag([1., 2.])
    bad = kernel.validate_eigenpair(matrix, 2., np.array([1., 0.]))
    assert not bad.residual_passed
    assert bad.dominant_passed
    assert not bad.accepted
    assert not kernel.validate_eigenpair(matrix, [2.], np.ones(2)).accepted
    assert not kernel.validate_eigenpair(matrix, 2., np.ones((2, 1))).accepted
    assert not kernel.validate_eigenpair(matrix, np.inf, np.ones(2)).accepted


@pytest.mark.parametrize("matrix_scale,vector_scale", [(1., 1e-100), (1e-300, 1e-100), (1e-300, 1e-320)])
def test_checker_rejects_wrong_sign_with_tiny_nonzero_vector_or_matrix(matrix_scale, vector_scale):
    matrix = np.diag([2., 1.]) * matrix_scale
    wrong_value = -2. * matrix_scale
    vector = np.array([vector_scale, 0.])
    checked = kernel.validate_eigenpair(matrix, wrong_value, vector)
    assert checked.residual_passed  # Literal public tolerance alone is too weak.
    assert checked.raw_allclose_passed
    assert checked.dominant_passed
    assert checked.nonzero_vector
    assert not checked.normalized_residual_passed
    assert not checked.spectrum_membership_passed
    assert not checked.accepted


def test_checker_accepts_correct_pair_with_tiny_complex_vector():
    matrix = np.array([[0., -2.], [2., 0.]])
    vector = np.array([1e-320 + 0j, -1e-320j])
    checked = kernel.validate_eigenpair(matrix, 2j, vector)
    assert checked.accepted
    assert checked.normalized_residual_passed
    assert checked.spectrum_membership_passed


def test_checker_rejects_infinite_residual_products():
    matrix = np.diag([1e308, 1e308])
    checked = kernel.validate_eigenpair(matrix, 1e308, np.array([2., 2.]))
    with np.errstate(over="ignore", invalid="ignore"):
        literal = np.allclose(matrix @ np.array([2., 2.]), complex(1e308) * np.array([2., 2.]))
    assert checked.raw_allclose_passed == literal
    assert not checked.residual_passed
    assert not checked.accepted


def test_numpy_backend_is_explicitly_unoptimized():
    matrix = np.diag([1., 2.])
    result = kernel.solve_dominant_eigenpair(matrix, backend=kernel.NUMPY_BACKEND)
    assert result.backend == kernel.NUMPY_BACKEND
    assert not result.optimization_candidate
    assert result.fallback_reason is None
    assert kernel.validate_eigenpair(matrix, result.eigenvalue, result.eigenvector).accepted


def test_missing_scipy_requires_explicit_fallback(monkeypatch):
    real_import = builtins.__import__
    def without_scipy(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ModuleNotFoundError("authored missing scipy")
        return real_import(name, *args, **kwargs)
    kernel._candidate_namespace.cache_clear()
    monkeypatch.setattr(builtins, "__import__", without_scipy)
    try:
        with pytest.raises(ModuleNotFoundError):
            kernel.solve_dominant_eigenpair(np.eye(2))
        fallback = kernel.solve_dominant_eigenpair(np.eye(2), allow_numpy_fallback=True)
        assert fallback.backend == kernel.NUMPY_BACKEND
        assert fallback.fallback_reason == "scipy_unavailable"
        assert not fallback.optimization_candidate
        qualification = kernel.qualify_spectral_kernel()
        assert not qualification["available"]
        assert not qualification["accepted"]
        assert qualification["reason_code"] == "dependency_unavailable"
        with pytest.raises(ModuleNotFoundError):
            compiled_candidate()
    finally:
        kernel._candidate_namespace.cache_clear()


def test_standalone_source_is_fixed_and_qualification_is_not_speed_proof():
    source = kernel.candidate_source()
    assert "ipfs_accelerate_py" not in source
    assert "scipy.linalg.lapack" in source
    receipt = kernel.qualify_spectral_kernel()
    assert receipt["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert receipt["available"] and receipt["accepted"]
    assert receipt["case_count"] == 22
    assert all(case["input_unchanged"] for case in receipt["cases"])
    assert not receipt["speed_improvement_qualified"]
    assert not receipt["exact_spectral_certificate"]
    with pytest.raises(ValueError, match="unsupported"):
        kernel.candidate_source("task-selected-module")
    with pytest.raises(ValueError, match="unsupported"):
        kernel.solve_dominant_eigenpair(np.eye(2), backend="task-selected-module")
