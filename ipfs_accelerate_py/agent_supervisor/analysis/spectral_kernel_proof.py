"""SMT checks for ideal spectral-adapter lemmas, not a LAPACK/IEEE proof.

The backend's complete spectrum and packed-vector equations are assumptions.
The Python recipe needs separate review, numerical checks and native validation.
"""
from __future__ import annotations

import hashlib
import re


EXPECTED_LEMMA_NAMES = (
    "packed_complex_vector_sign_1", "packed_complex_vector_sign_-1",
    "normalized_packed_vector_is_nonzero",
    *("bounded_magnitude_argmax_" + str(size) for size in range(1, 11)),
)


def prove_spectral_adapter_lemmas(*, kernel_source_sha256: str, timeout_ms: int = 2000) -> dict:
    if not isinstance(kernel_source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", kernel_source_sha256):
        raise ValueError("a bound reviewed kernel source digest is required")
    if type(timeout_ms) is not int or not 1 <= timeout_ms <= 10_000:
        raise ValueError("bounded integer SMT timeout required")
    report = {"schema": "spectral-adapter-smt-lemmas@1", "kernel_source_sha256": kernel_source_sha256,
        "status": "unknown", "all_lemmas_verified": False, "checks": [],
        "scope": "conditional ideal complex-vector algebra and bounded computed-magnitude selection",
        "assumptions": ["The trusted solver supplies a complete numerical spectrum.",
            "Packed real columns satisfy the declared ideal backend eigenvector equations.",
            "Selector inputs are finite nonnegative computed magnitudes."],
        "python_program_mechanically_verified": False, "lapack_implementation_proved": False,
        "ieee_floating_point_accuracy_proved": False, "timing_proved": False,
        "kernel_proved": False, "whole_program_proved": False, "provider_calls": 0,
        "proof_authority": False, "publication_authority": False, "completion_authority": False}
    try:
        import z3
    except ImportError:
        report.update(status="unsupported", reason="z3_unavailable")
        return report
    report["solver"] = {"name": "z3", "version": z3.get_version_string()}

    def check(name, assumptions, counterexample):
        solver = z3.Solver()
        solver.set(timeout=timeout_ms)
        solver.add(*assumptions, counterexample)
        wire = solver.to_smt2().encode()
        verdict = solver.check()
        report["checks"].append({"name": name, "verdict": str(verdict), "verified": verdict == z3.unsat,
            "smtlib_sha256": hashlib.sha256(wire).hexdigest(), "smtlib_bytes": len(wire),
            "unknown_reason": solver.reason_unknown() if verdict == z3.unknown else None})

    a, b, u, v, au, av = z3.Reals("a b u v Au Av")
    for sign in (1, -1):
        # A(u+i*s*v), compared with (a+i*s*b)(u+i*s*v).
        check("packed_complex_vector_sign_" + str(sign),
            [au == a*u-b*v, av == b*u+a*v],
            z3.Or(au != a*u-(sign*b)*(sign*v), sign*av != a*(sign*v)+(sign*b)*u))
    norm_real, norm_imag = z3.Reals("norm_real_squared norm_imag_squared")
    check("normalized_packed_vector_is_nonzero",
        [norm_real >= 0, norm_imag >= 0, norm_real+norm_imag == 1],
        norm_real+norm_imag <= 0)
    for size in range(1, 11):
        magnitudes = [z3.Real("m_" + str(size) + "_" + str(i)) for i in range(size)]
        largest = magnitudes[0]
        for magnitude in magnitudes[1:]:
            largest = z3.If(magnitude > largest, magnitude, largest)
        check("bounded_magnitude_argmax_" + str(size), [m >= 0 for m in magnitudes],
            z3.Or(*[m > largest for m in magnitudes]))
    report["all_lemmas_verified"] = all(row["verified"] for row in report["checks"])
    report["status"] = "verified_scoped_lemmas" if report["all_lemmas_verified"] else "unknown"
    report["analysis_source_sha256"] = hashlib.sha256(__import__("pathlib").Path(__file__).read_bytes()).hexdigest()
    return report
