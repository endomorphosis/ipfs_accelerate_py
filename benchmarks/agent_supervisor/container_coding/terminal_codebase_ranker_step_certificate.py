"""Exact rational step prerequisites for the existing binary64 scratch head.

No fit or gradient update is performed. These inequalities certify constants;
the logistic Hessian theorem, actual floating point update error and optimizer
convergence remain separate obligations.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import re

from . import terminal_codebase_intent_ranker_training as ranker

SCHEMA = "terminal-codebase-ranker-step-certificate@1"


class RankerStepCertificateError(ValueError):
    """Native parameter replay or an exact step prerequisite was refused."""


def _wire(value):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError) as error:
        raise RankerStepCertificateError("plain finite JSON required") from error
    if len(raw) > 32 * 1024 * 1024:
        raise RankerStepCertificateError("certificate exceeds bound")
    return raw


def _rational(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _lean(value):
    return f"(({value.numerator} : Rat) / {value.denominator})"


def certify_ranker_step_parameters(*, differences, regularization, step_size):
    """Certify exact real embeddings of explicitly supplied finite binary64s."""
    if (type(differences) is not list or not 1 <= len(differences) <= ranker.MAX_PAIRS or
            type(regularization) is not float or type(step_size) is not float or
            not math.isfinite(regularization) or not math.isfinite(step_size)):
        raise RankerStepCertificateError("bounded native finite parameter vectors required")
    for row in differences:
        if (type(row) is not list or len(row) != 80 or
                any(type(x) is not float or not math.isfinite(x) for x in row)):
            raise RankerStepCertificateError("exact eighty-dimensional binary64 rows required")
    mu = Fraction.from_float(regularization)
    step = Fraction.from_float(step_size)
    norms = [sum((Fraction.from_float(x) ** 2 for x in row), Fraction(0)) for row in differences]
    # The analytic logistic Hessian coefficient <= 1/4 is a separate obligation.
    # All arithmetic here is exact over the real embeddings of actual float bits.
    conditional_smoothness = mu + max(norms) / 4
    contraction = 1 - mu * step
    if not (mu > 0 and step > 0 and step * conditional_smoothness <= 1 and 0 <= contraction < 1):
        raise RankerStepCertificateError("positive safe step / contraction prerequisites failed")
    return {"regularization_exact": _rational(mu), "step_exact": _rational(step),
        "pair_squared_norms_exact": [_rational(x) for x in norms],
        "conditional_smoothness_exact": _rational(conditional_smoothness),
        "conditional_contraction_exact": _rational(contraction)}


def build_terminal_ranker_step_certificate(*, corpus_receipt, original_inputs, expected_corpus_sha256):
    """Replay native corpus/features/step selection, without retraining a head."""
    if (type(expected_corpus_sha256) is not str or
            re.fullmatch(r"sha256:[0-9a-f]{64}", expected_corpus_sha256) is None):
        raise RankerStepCertificateError("independent native corpus pin required")
    try:
        prepared = ranker._prepare(corpus_receipt, original_inputs)
    except (TypeError, ValueError, RecursionError) as error:
        raise RankerStepCertificateError("native parameter replay refused") from error
    corpus, _, _, pairs, differences, profile, numeric_smoothness, step = prepared
    if corpus["corpus_sha256"] != expected_corpus_sha256:
        raise RankerStepCertificateError("native corpus differs from independent pin")
    exact = certify_ranker_step_parameters(differences=differences,
        regularization=ranker.L2, step_size=step)
    mu = Fraction.from_float(ranker.L2); eta = Fraction.from_float(step)
    smooth = Fraction(int(exact["conditional_smoothness_exact"]["numerator"]),
        int(exact["conditional_smoothness_exact"]["denominator"]))
    q = 1 - mu * eta
    source = "\n".join(["import Std", "set_option autoImplicit false", "namespace RankerStepPrerequisites",
        "-- Exact rational checks only; no logistic calculus or binary64 convergence theorem.",
        "noncomputable def mu : Rat := " + _lean(mu),
        "noncomputable def eta : Rat := " + _lean(eta),
        "noncomputable def conditionalSmoothness : Rat := " + _lean(smooth),
        "noncomputable def q : Rat := " + _lean(q),
        "theorem exact_positive_safe_step :", "    0 < mu ∧ 0 < eta ∧ eta * conditionalSmoothness ≤ 1 ∧",
        "    q = 1 - mu * eta ∧ 0 ≤ q ∧ q < 1 := by decide +kernel",
        "end RankerStepPrerequisites", ""])
    result = {"schema": SCHEMA, "status": "exact_parameters_unchecked_lean_prerequisites",
        "corpus_sha256": expected_corpus_sha256, "feature_profile_sha256": ranker._digest(profile),
        "native_difference_vectors_sha256": ranker._digest(differences),
        "native_train_pairs": pairs, "dimension": 80, "native_numeric_smoothness": numeric_smoothness,
        "native_step_size": step, **exact,
        "lean_source": source, "lean_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "analytic_obligations": ["Prove the mathematical logistic Hessian eigenvalue bound from these exact feature vectors.",
            "Connect exact smooth strongly convex gradient descent to the contraction rate.",
            "Bound errors in binary64 dot products, libm exp/log1p, gradient and updates.",
            "Authenticate every intermediate weight state before using a finite trace certificate.",
            "Qualify held-out semantic alignment and task outcomes separately from training objective convergence."],
        "exact_parameter_inequalities_checked_in_python": True, "lean_checker_status": "not_run",
        "training_calls": 0, "optimizer_updates": 0, "checker_calls": 0,
        "logistic_hessian_bound_proved": False, "binary64_error_bound_proved": False,
        "asymptotic_optimizer_convergence_proved": False, "autoencoder_convergence_proved": False,
        "full_task_satisfaction": "unknown", "planning_handoff": "abstained", **ranker._AUTHORITY}
    result["certificate_sha256"] = hashlib.sha256(_wire(result)).hexdigest()
    return json.loads(_wire(result))


def validate_terminal_ranker_step_certificate(receipt, **arguments):
    expected = build_terminal_ranker_step_certificate(**arguments)
    if _wire(receipt) != _wire(expected):
        raise RankerStepCertificateError("certificate differs from independent native replay")
    return expected
