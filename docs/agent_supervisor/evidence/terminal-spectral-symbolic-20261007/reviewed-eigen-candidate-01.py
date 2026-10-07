import numpy as np

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
