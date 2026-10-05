"""Authenticate a bounded finite scratch-ranker trace by native numeric replay.

This additive consumer keeps the existing passive validator's contract intact.
It checks every recorded observation and weight-state digest by replaying the
declared all-zero, full-batch update profile. Agreement authenticates finite
artifact consistency under this runtime and independently supplied digest pins;
it does not authenticate historical execution origin or prove convergence.
The compact receipt records every state digest; the independently pinned
original ranker result supplies the complete observed metric trace.
No fit entry point, checkpoint activation, filesystem or worker is used.
"""
from __future__ import annotations

import math
import re
import sys

from . import terminal_codebase_intent_ranker_training as ranker

SCHEMA = "terminal-codebase-ranker-training-trace-authentication@1"
NATIVE_UPDATE_PROFILE_SCHEMA = "terminal-codebase-ranker-native-update-profile@1"
DIMENSION = 80
MAX_COORDINATE_OPERATIONS = 1_000_000
DEFAULT_MAX_COORDINATE_OPERATIONS = MAX_COORDINATE_OPERATIONS
_HASH = re.compile(r"[0-9a-f]{64}")
_TRACE_FIELDS = {"epoch", "objective", "pair_logistic_loss", "L2_penalty",
                 "gradient_norm", "weights_sha256"}
_OBSERVATION_FIELDS = ("objective", "pair_logistic_loss", "L2_penalty", "gradient_norm")


class RankerTraceAuthenticationError(ValueError):
    """Independent pins, replay budget or a recorded finite state disagrees."""


def _need(condition, message):
    if condition is not True:
        raise RankerTraceAuthenticationError(message)


def _pin(value, name):
    _need(type(value) is str and _HASH.fullmatch(value) is not None,
          "independent lowercase 64-hex " + name + " pin required")


def _clone(value):
    try:
        return ranker._json(value)
    except (TypeError, ValueError, KeyError, IndexError, RecursionError, OverflowError) as error:
        raise RankerTraceAuthenticationError("bounded plain finite JSON required") from error


def _preflight(receipt, max_coordinate_operations):
    """Bound optimizer work using exact, cheap shape checks before native replay."""
    _need(type(receipt) is dict, "plain ranker receipt object required")
    training, checkpoint = receipt.get("training_receipt"), receipt.get("checkpoint")
    _need(type(training) is dict and type(checkpoint) is dict,
          "plain training receipt and checkpoint objects required")
    epochs, pairs = training.get("epochs"), training.get("train_pairs")
    _need(type(epochs) is int and 1 <= epochs <= ranker.MAX_EPOCHS,
          "1..128 exact integer epochs required")
    _need(type(pairs) is list and 1 <= len(pairs) <= ranker.MAX_PAIRS,
          "1..4096 recorded train pairs required")
    weights = checkpoint.get("weights")
    _need(type(weights) is list and len(weights) == DIMENSION and all(
        type(value) is float and math.isfinite(value) for value in weights),
        "eighty finite binary64 checkpoint weights required")
    trace = training.get("trace")
    _need(type(trace) is list and len(trace) == epochs + 1,
          "complete bounded epoch trace required")
    for epoch, row in enumerate(trace):
        _need(type(row) is dict and len(row) == len(_TRACE_FIELDS) and set(row) == _TRACE_FIELDS
              and type(row["epoch"]) is int and row["epoch"] == epoch,
              "exact complete ordered epoch observations required")
        _pin(row["weights_sha256"], "recorded weight-state")
        _need(all(type(row[name]) is float and math.isfinite(row[name])
                  and row[name] >= 0 for name in _OBSERVATION_FIELDS),
              "finite nonnegative binary64 epoch observations required")
    # Per objective: pair margins and pair-gradient coordinates (2*p*d),
    # weight norm, L2 gradient coordinates and gradient norm (3*d).
    # Each of the epochs updates visits d coordinates once. This is a native
    # scalar-coordinate visit count, not a count of every floating operation.
    operations = DIMENSION * ((2 * len(pairs) + 3) * (epochs + 1) + epochs)
    _need(operations <= max_coordinate_operations,
          "native trace replay exceeds coordinate-operation budget")
    return epochs, len(pairs), operations


def _native_profile():
    _need(sys.implementation.name == "cpython"
          and sys.float_info.radix == 2 and sys.float_info.mant_dig == 53
          and sys.float_info.max_exp == 1024 and sys.float_info.min_exp == -1021,
          "CPython binary64 native replay profile required")
    _need(len(ranker._FEATURE_NAMES) == DIMENSION,
          "native eighty-dimensional feature profile required")
    return {
        "schema": NATIVE_UPDATE_PROFILE_SCHEMA,
        "python_implementation": "cpython",
        "python_version": list(sys.version_info[:3]),
        "float_radix": sys.float_info.radix,
        "float_mantissa_bits": sys.float_info.mant_dig,
        "float_max_exponent": sys.float_info.max_exp,
        "float_min_exponent": sys.float_info.min_exp,
        "float_rounding_mode_reported_by_python": sys.float_info.rounds,
        "objective_evaluator": "terminal_codebase_intent_ranker_training._objective",
        "summation": "CPython math.fsum",
        "nonlinear_functions": ["math.exp", "math.log1p", "math.sqrt"],
        "nonlinear_function_implementation": "current CPython platform libm",
        "platform_libm_identity_independently_pinned": False,
        "initialization": "eighty_all_positive_zero_binary64_weights",
        "optimizer": "deterministic_binary64_fullbatch_gradient_descent",
        "update_rule": "weights[i] - native_step_size * native_gradient[i]",
        "feature_iteration_order": "native prepared train-pair order and fixed feature order",
        "binary64_error_bound_proved": False,
        "math_fsum_error_bound_proved": False,
        "libm_error_bound_proved": False,
        "cross_platform_bitwise_reproducibility_proved": False,
    }


def _replay_observation(weights, differences):
    """Use exactly the existing native evaluator, without calling its fit API."""
    return ranker._objective(weights, differences)


def authenticate_terminal_ranker_training_trace(receipt, *, corpus_receipt,
        original_inputs, expected_checkpoint_sha256,
        expected_training_receipt_sha256, expected_ranker_result_sha256,
        max_coordinate_operations=DEFAULT_MAX_COORDINATE_OPERATIONS):
    """Replay every finite epoch under mandatory independent artifact pins.

    Checkpoint pin: native canonical digest of the entire checkpoint object.
    Training pin: native ``receipt_sha256`` (digest before its self field).
    Result pin: native canonical digest of the entire ranker result INCLUDING
    its ``result_sha256`` self field; it is not a raw-file SHA256. Callers must
    establish the external origin of these pins separately.

    ``max_coordinate_operations`` is an exact integer in 1..1,000,000. The
    budget counts full optimizer replay coordinate visits, including the
    final observation, and refuses excess work before calling the passive
    validator. Existing corpus/JSON/preparation bounds still apply; their work
    and the passive validator's two endpoint evaluations are outside this
    explicitly reported replay budget. No populations or epochs are truncated.
    """
    # Pins and budget are checked before JSON cloning, corpus preparation or
    # any gradient evaluation, including the passive endpoint evaluations.
    _pin(expected_checkpoint_sha256, "checkpoint")
    _pin(expected_training_receipt_sha256, "training receipt")
    _pin(expected_ranker_result_sha256, "complete ranker result")
    _need(type(max_coordinate_operations) is int
          and 1 <= max_coordinate_operations <= MAX_COORDINATE_OPERATIONS,
          "exact integer coordinate-operation budget in 1..1000000 required")
    _need(type(corpus_receipt) is dict and type(original_inputs) is dict,
          "independent plain corpus receipt and original arguments required")
    epochs, pair_count, operations = _preflight(receipt, max_coordinate_operations)
    supplied = _clone(receipt)
    checkpoint, training = supplied["checkpoint"], supplied["training_receipt"]
    _need(ranker._digest(checkpoint) == expected_checkpoint_sha256,
          "externally frozen checkpoint differs")
    _need(training.get("receipt_sha256") == expected_training_receipt_sha256
          and ranker._digest({key: value for key, value in training.items()
                             if key != "receipt_sha256"}) == expected_training_receipt_sha256,
          "externally frozen training receipt differs")
    _need(ranker._digest(supplied) == expected_ranker_result_sha256,
          "externally frozen complete ranker result differs")
    native_profile = _native_profile()
    try:
        validated = ranker.validate_terminal_codebase_intent_ranker(supplied,
            corpus_receipt=corpus_receipt, original_inputs=original_inputs,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_training_receipt_sha256=expected_training_receipt_sha256)
        prepared = ranker._prepare(corpus_receipt, original_inputs)
    except (TypeError, ValueError, KeyError, IndexError, RecursionError, OverflowError) as error:
        raise RankerTraceAuthenticationError("native passive validation/preparation refused") from error
    _need(ranker._digest(validated) == expected_ranker_result_sha256,
          "validated complete ranker result differs from independent pin")
    corpus, _, _, pairs, differences, feature_profile, smoothness, step = prepared
    _need(len(pairs) == pair_count and len(differences) == pair_count
          and all(type(row) is list and len(row) == DIMENSION and all(
              type(value) is float and math.isfinite(value) for value in row)
                  for row in differences),
          "native prepared train population differs from bounded replay plan")
    _need(type(step) is float and math.isfinite(step) and step > 0,
          "finite positive native update step required")
    weights = [0.0] * DIMENSION
    initial_weights_sha256 = ranker._digest(weights)
    states, authenticated_trace = [], []
    for epoch in range(epochs + 1):
        try:
            observation, gradient = _replay_observation(weights, differences)
        except (TypeError, ValueError, OverflowError) as error:
            raise RankerTraceAuthenticationError("native replay observation refused") from error
        _need(type(gradient) is list and len(gradient) == DIMENSION
              and all(type(value) is float and math.isfinite(value) for value in gradient),
              "finite native replay gradient required")
        row = {"epoch": epoch, **observation, "weights_sha256": ranker._digest(weights)}
        # Canonical digests preserve exact float representations, including
        # signed zero, instead of Python equality's numeric coercions.
        _need(ranker._digest(row) == ranker._digest(training["trace"][epoch]),
              "recorded epoch " + str(epoch) + " differs from native update replay")
        authenticated_trace.append(row)
        states.append({
            "epoch": epoch,
            "weights_sha256": row["weights_sha256"],
            "gradient_sha256": ranker._digest(gradient),
            "finite_state_sha256": ranker._digest({"epoch": epoch, "weights": weights,
                "gradient": gradient, "observation": observation}),
        })
        if epoch < epochs:
            weights = [weight - step * component for weight, component in zip(weights, gradient)]
            _need(all(type(value) is float and math.isfinite(value) for value in weights),
                  "nonfinite native replay update")
    replayed_checkpoint = {"schema": ranker.CHECKPOINT_SCHEMA,
        "corpus_sha256": corpus["corpus_sha256"],
        "feature_profile_sha256": ranker._digest(feature_profile),
        "weights": weights, "weights_sha256": ranker._digest(weights),
        "initialization": "new_all_zero_scratch_head"}
    _need(ranker._digest(replayed_checkpoint) == expected_checkpoint_sha256
          and ranker._digest(weights) == checkpoint["weights_sha256"],
          "final checkpoint differs from full native update replay")
    result = {
        "schema": SCHEMA,
        "status": "authenticated_finite_native_training_trace",
        "authentication_scope": "same_declared_native_update_profile_and_independently_supplied_artifact_pins",
        "independent_pin_origin_authenticated": False,
        "artifact_digest_conventions": {
            "checkpoint_sha256": "native canonical complete checkpoint object",
            "training_receipt_sha256": "native receipt_sha256 excluding its self field",
            "ranker_result_sha256": "native canonical complete ranker result including its result_sha256 self field",
            "raw_file_digests_accepted": False,
        },
        "corpus_sha256": corpus["corpus_sha256"],
        "feature_schema": ranker.FEATURE_SCHEMA,
        "feature_profile_sha256": ranker._digest(feature_profile),
        "train_pairs_sha256": ranker._digest(pairs),
        "train_pair_feature_differences_sha256": ranker._digest(differences),
        "checkpoint_sha256": expected_checkpoint_sha256,
        "training_receipt_sha256": expected_training_receipt_sha256,
        "ranker_result_sha256": expected_ranker_result_sha256,
        "native_ranker_result_self_sha256": validated["result_sha256"],
        "native_update_profile": native_profile,
        "native_update_profile_sha256": ranker._digest(native_profile),
        "epochs": epochs, "train_pair_count": pair_count, "dimension": DIMENSION,
        "L2": ranker.L2, "native_step_size": step,
        "native_numeric_smoothness": smoothness,
        "initial_weights_sha256": initial_weights_sha256,
        "final_weights_sha256": ranker._digest(weights),
        "recorded_trace_sha256": ranker._digest(training["trace"]),
        "authenticated_trace_sha256": ranker._digest(authenticated_trace),
        "authenticated_states": states,
        "authenticated_states_sha256": ranker._digest(states),
        "finite_state_digest_representation": {
            "stored_fields": ["epoch", "weights_sha256", "gradient_sha256", "finite_state_sha256"],
            "finite_state_digest_commits": ["epoch", "weights", "gradient", "observation"],
            "complete_observed_metrics_location": "independently pinned original ranker result training_receipt.trace",
            "all_epoch_states_retained": True,
        },
        "replay_budget": {
            "max_coordinate_operations": max_coordinate_operations,
            "hard_max_coordinate_operations": MAX_COORDINATE_OPERATIONS,
            "coordinate_operations": operations,
            "scope": "full optimizer replay only; excludes existing passive validation and preparation",
            "charge_formula": "80 * ((2 * train_pair_count + 3) * (epochs + 1) + epochs)",
            "scalar_coordinate_visits_not_total_floating_operations": True,
            "population_or_epoch_truncation": False,
        },
        "replay_coordinate_operations": operations,
        "endpoint_validation_gradient_evaluations": 2,
        "replay_gradient_evaluations": epochs + 1,
        "gradient_evaluations": epochs + 3,
        "native_preparation_calls": 2,
        "passive_validator_calls": 1,
        "optimizer_replay_updates": epochs,
        "new_training_fit_calls": 0,
        "checkpoint_activation_calls": 0,
        "checked_trace_rows": epochs + 1,
        "checked_weight_state_digests": epochs + 1,
        "finite_native_update_sequence_authenticated": True,
        "intermediate_training_history_authenticated_here": True,
        "historical_training_provenance_authenticated_here": False,
        "historical_execution_origin_authenticated": False,
        "all_recorded_epoch_observations_match_native_replay": True,
        "all_recorded_weight_digests_match_native_replay": True,
        "final_checkpoint_matches_native_replay": True,
        "new_training_fit": False, "checkpoint_activated": False,
        "finite_recorded_trace": True,
        "native_update_error_bound_proved": False,
        "binary64_error_bound_proved": False,
        "real_logistic_descent_proved": False,
        "asymptotic_convergence_proved": False,
        "global_optimizer_convergence_proved": False,
        "autoencoder_convergence_proved": False,
        "generalized_ranking_gain_qualified": False,
        "complete_prompt_interpretation_qualified": False,
        "full_task_satisfaction": "unknown",
        "planning_handoff": "abstained",
        **ranker._AUTHORITY,
    }
    result["authentication_sha256"] = ranker._digest(result)
    return _clone(result)


def validate_terminal_ranker_training_trace_authentication(authentication_receipt,
        *, receipt, corpus_receipt, original_inputs, expected_checkpoint_sha256,
        expected_training_receipt_sha256, expected_ranker_result_sha256,
        max_coordinate_operations=DEFAULT_MAX_COORDINATE_OPERATIONS):
    """Validate the complete companion receipt by independent bounded replay."""
    # Preserve fail-before-work pin and budget rules even when validating an
    # authentication receipt; authentication performs their complete checks.
    _pin(expected_checkpoint_sha256, "checkpoint")
    _pin(expected_training_receipt_sha256, "training receipt")
    _pin(expected_ranker_result_sha256, "complete ranker result")
    _need(type(max_coordinate_operations) is int
          and 1 <= max_coordinate_operations <= MAX_COORDINATE_OPERATIONS,
          "exact integer coordinate-operation budget in 1..1000000 required")
    supplied = _clone(authentication_receipt)
    _need(type(supplied) is dict and type(supplied.get("authentication_sha256")) is str
          and supplied["authentication_sha256"] == ranker._digest({key: value
              for key, value in supplied.items() if key != "authentication_sha256"}),
          "authentication receipt self-digest differs")
    expected = authenticate_terminal_ranker_training_trace(receipt,
        corpus_receipt=corpus_receipt, original_inputs=original_inputs,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_training_receipt_sha256=expected_training_receipt_sha256,
        expected_ranker_result_sha256=expected_ranker_result_sha256,
        max_coordinate_operations=max_coordinate_operations)
    _need(ranker._digest(supplied) == ranker._digest(expected),
          "authentication receipt differs from independent full native replay")
    return expected


__all__ = ["SCHEMA", "NATIVE_UPDATE_PROFILE_SCHEMA", "DIMENSION",
    "MAX_COORDINATE_OPERATIONS", "DEFAULT_MAX_COORDINATE_OPERATIONS",
    "RankerTraceAuthenticationError", "authenticate_terminal_ranker_training_trace",
    "validate_terminal_ranker_training_trace_authentication"]
