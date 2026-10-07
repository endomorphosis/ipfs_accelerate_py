"""Bounded target-interpreter checks of a fixed, owner-reviewed numerical recipe.

The child runs authored fixtures, never evaluator files or task-selected code.
Acceptance is finite numerical evidence, without timing or proof authority.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess


_PROBE = r'''
import hashlib, json, sys
payload = json.loads(sys.stdin.read(65537))
if set(payload) != {"source", "source_sha256"}:
    raise ValueError("closed trusted source payload required")
source = payload["source"]
if hashlib.sha256(source.encode()).hexdigest() != payload["source_sha256"]:
    raise ValueError("source digest differs")
result = {"accepted": False, "source_sha256": payload["source_sha256"],
          "python_version": sys.version.split()[0], "numpy_version": "",
          "scipy_version": "", "case_count": 0, "reason_code": "dependency_unavailable"}
try:
    import numpy as np
    import scipy
except ImportError:
    print(json.dumps(result, allow_nan=False)); sys.exit(0)
result.update(numpy_version=np.__version__, scipy_version=scipy.__version__)
try:
    namespace = {"__name__": "reviewed_target_spectral_candidate"}
    exec(compile(source, "<reviewed-target-spectral-candidate>", "exec"), namespace)
    solve = namespace["find_dominant_eigenvalue_and_eigenvector"]
    rng = np.random.default_rng(270710)
    cases = [rng.normal(size=(n, n)).astype(np.float64) for n in range(1, 11)]
    cases += [np.array([[0., -2.], [2., 0.]]), np.diag([-3., 2.]),
              np.diag([-3., 3.]), np.zeros((4, 4)),
              np.array([[2., 1., 0.], [0., 2., 1.], [0., 0., 2.]]),
              np.array([[1., 1000., -2000.], [0., -2., 500.], [0., 0., 3.]])]
    cases += [np.array([[0., -2.], [2., 0.]]) * scale for scale in (1e150, 1e-150)]
    base = rng.normal(size=(5, 5)).astype(np.float64)
    readonly = base.copy(); readonly.flags.writeable = False
    cases += [np.asfortranarray(base), base.T, base[::-1, ::-1], readonly]
    for matrix in cases:
        before = matrix.copy()
        value, vector = solve(matrix)
        value = np.asarray(value); vector = np.asarray(vector)
        if value.ndim != 0 or vector.shape != (len(matrix),):
            raise ValueError("output_shape")
        if not np.isfinite(value).all() or not np.isfinite(vector).all() or not np.any(vector != 0):
            raise ValueError("output_finite_nonzero")
        if not np.array_equal(matrix, before):
            raise ValueError("input_modified")
        vector_scale = float(np.abs(vector).max())
        normalized = vector / vector_scale
        normalized /= np.linalg.norm(normalized)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            left, right = matrix @ vector, value * vector
        if not np.isfinite(left).all() or not np.isfinite(right).all() or not np.allclose(left, right):
            raise ValueError("residual_failed")
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            left, right = matrix @ normalized, value * normalized
        if not np.isfinite(left).all() or not np.isfinite(right).all() or not np.allclose(left, right):
            raise ValueError("normalized_residual_failed")
        scalar = complex(value.item())
        equation_scale = max(float(np.abs(matrix).max()), abs(scalar.real), abs(scalar.imag))
        if equation_scale:
            left = (matrix / equation_scale) @ normalized
            right = (scalar / equation_scale) * normalized
            if not np.allclose(left, right):
                raise ValueError("scaled_normalized_residual_failed")
        reference, _ = np.linalg.eig(before)
        if not np.isfinite(reference).all():
            raise ValueError("reference_nonfinite")
        import math
        components = [abs(scalar.real), abs(scalar.imag)]
        for entry in reference:
            components += [abs(complex(entry).real), abs(complex(entry).imag)]
        scale = max(components)
        observed = math.hypot(scalar.real / scale, scalar.imag / scale) if scale else 0.
        dominant = max(math.hypot(complex(entry).real / scale, complex(entry).imag / scale)
                       for entry in reference) if scale else 0.
        if not math.isclose(observed, dominant, rel_tol=1e-7, abs_tol=0.):
            raise ValueError("dominance_failed")
        if scale:
            distances = [abs(scalar / scale - complex(entry) / scale) for entry in reference]
            if min(distances) > 1e-7 * dominant:
                raise ValueError("spectrum_membership_failed")
        result["case_count"] += 1
    result.update(accepted=True, reason_code="accepted_finite_fixture_checks")
except Exception:
    result.update(accepted=False, reason_code="numerical_qualification_failed")
print(json.dumps(result, allow_nan=False))
'''


def expected_probe_sha256() -> str:
    """Bind workflow evidence to this installed, fixed authored probe."""
    return hashlib.sha256(_PROBE.encode()).hexdigest()


def qualify_spectral_target_runtime(*, interpreter: Path, timeout_seconds: int = 30) -> dict:
    """Check the fixed source in the explicitly chosen isolated Python runtime.

    Preserve a virtual environment's invocation path; separately bind the
    resolved executable. This does not identify an official benchmark image.
    """
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 60:
        raise ValueError("bounded integer target timeout required")
    interpreter = Path(interpreter)
    if not interpreter.is_absolute() or ".." in interpreter.parts:
        raise ValueError("explicit absolute interpreter required")
    from .spectral_eigen_kernel import SCIPY_BACKEND, candidate_source
    source = candidate_source(SCIPY_BACKEND)
    source_sha = hashlib.sha256(source.encode()).hexdigest()
    if len(source.encode()) > 60_000:
        raise ValueError("bounded reviewed source required")
    result = {"schema": "spectral-target-numeric-qualification@1", "accepted": False,
        "source_sha256": source_sha, "interpreter": str(interpreter), "backend": SCIPY_BACKEND,
        "target_probe_sha256": expected_probe_sha256(),
        "case_count": 0, "numpy_version": "", "scipy_version": "", "python_version": "",
        "provider_calls": 0, "proof_authority": False, "publication_authority": False,
        "completion_authority": False, "kernel_proved": False, "timing_qualified": False,
        "reason_code": "target_interpreter_unavailable"}
    try:
        executable = interpreter.resolve(strict=True)
        metadata = executable.stat()
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid not in {0, os.getuid()}
                or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
                or not os.access(interpreter, os.X_OK)):
            result["reason_code"] = "target_interpreter_untrusted"
            return result
        binary_sha = hashlib.sha256(executable.read_bytes()).hexdigest()
        result.update(executable=str(executable), executable_sha256=binary_sha)
        payload = json.dumps({"source": source, "source_sha256": source_sha}, allow_nan=False)
        child = subprocess.run([str(interpreter), "-I", "-c", _PROBE], input=payload,
            text=True, capture_output=True, timeout=timeout_seconds, check=False)
        if child.returncode != 0 or len(child.stdout.encode()) > 65_536:
            result["reason_code"] = "target_probe_failed"
            return result
        observed = json.loads(child.stdout)
        keys = {"accepted", "source_sha256", "python_version", "numpy_version",
                "scipy_version", "case_count", "reason_code"}
        if (type(observed) is not dict or set(observed) != keys
                or type(observed["accepted"]) is not bool or observed["source_sha256"] != source_sha
                or type(observed["case_count"]) is not int or not 0 <= observed["case_count"] <= 22
                or any(type(observed[key]) is not str or len(observed[key]) > 128
                       for key in ("python_version", "numpy_version", "scipy_version", "reason_code"))
                or (observed["accepted"] and (observed["case_count"] != 22
                    or not observed["numpy_version"] or not observed["scipy_version"]
                    or observed["reason_code"] != "accepted_finite_fixture_checks"))):
            result["reason_code"] = "target_probe_report_invalid"
            return result
        if hashlib.sha256(executable.read_bytes()).hexdigest() != binary_sha:
            result["reason_code"] = "target_executable_changed"
            return result
        result.update(observed)
    except subprocess.TimeoutExpired:
        result["reason_code"] = "target_probe_timeout"
    except (OSError, ValueError, TypeError):
        result["reason_code"] = "target_probe_failed"
    return result
