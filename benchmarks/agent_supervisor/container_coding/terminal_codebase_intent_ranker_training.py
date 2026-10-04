"""Bounded scratch query-conditioned navigation head, with passive replay.

Only reviewed train positive/negative_navigation pairs fit the new head. Other
candidates remain unjudged. Native AST features carry their declared historical
source exposure; neither their old reconstruction weights nor latents are used.
No filesystem, tensor library, search, checker or worker operation occurs here.
"""
from __future__ import annotations

import hashlib
import math
import re
import sys

from .terminal_codebase_intent_training_join import _AUTHORITY, _digest, _json

SCHEMA = "terminal-codebase-intent-ranker-training@1"
CHECKPOINT_SCHEMA = "terminal-codebase-intent-ranker-checkpoint@1"
RECEIPT_SCHEMA = "terminal-codebase-intent-ranker-fit-receipt@1"
FEATURE_SCHEMA = "fixed-query-code-lexical-native-AST-interactions@1"
L2 = 0.01
MAX_EPOCHS = 128
MAX_PAIRS = 4096
MAX_QUERY_CANDIDATES = 65536
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_FEATURE_NAMES = (["shared_token_sha256_bucket_" + str(i) for i in range(32)]
    + ["query_coverage", "code_coverage", "token_jaccard", "shared_token_fraction_of_both_sets"]
    + ["native_AST_" + str(i) + "_times_query_sha256_bucket_" + str(i % 16)
       for i in range(44)])


class IntentRankerTrainingError(ValueError):
    """Corpus, scratch-head or inference identities disagree."""


def _need(condition, message):
    if condition is not True:
        raise IntentRankerTrainingError(message)


def _tokens(text):
    _need(type(text) is str and len(text.encode()) <= 1024 * 1024,
          "bounded replay-bound query/code text required")
    return set(_TOKEN.findall(text.casefold()))


def _bucket(token, count):
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big") % count


def _feature(query_tokens, candidate):
    code = _tokens(candidate["normalized_body"])
    shared = query_tokens & code
    buckets = {_bucket(token, 32) for token in shared}
    query_buckets = {_bucket(token, 16) for token in query_tokens}
    qn, cn, sn = len(query_tokens), len(code), len(shared)
    qcoverage = sn / max(1, qn)
    lexical = [qcoverage, sn / max(1, cn), sn / max(1, len(query_tokens | code)),
               sn / max(1, qn + cn)]
    ast = candidate["features"]
    _need(type(ast) is list and len(ast) == 44 and all(
        type(x) in (int, float) and math.isfinite(x) and abs(x) <= 1 for x in ast),
        "44 bounded native AST features required")
    values = [float(i in buckets) for i in range(32)] + lexical + [
        float(x) * float(i % 16 in query_buckets) for i, x in enumerate(ast)]
    return values, qcoverage


def _prepare(corpus_receipt, original_inputs):
    from .terminal_codebase_intent_corpus import validate_terminal_intent_relevance_corpus
    _need(type(original_inputs) is dict, "independent original corpus arguments required")
    corpus = validate_terminal_intent_relevance_corpus(corpus_receipt, **original_inputs)
    _need(sys.float_info.radix == 2 and sys.float_info.mant_dig == 53,
          "binary64 Python float profile required")
    queries, banks = corpus["queries"], corpus["candidate_banks"]
    _need(len(queries) <= 64 and sum(len(banks[q["codebase_id"]]) for q in queries)
          <= MAX_QUERY_CANDIDATES, "bounded complete query candidate populations required")
    features, lexical, by_query = {}, {}, {}
    for query in queries:
        qid = query["query_id"]
        _need(qid not in by_query, "unique replayed query identity required")
        by_query[qid] = query
        features[qid], lexical[qid] = {}, {}
        tokens = _tokens(query["navigation_text"])
        for candidate in banks[query["codebase_id"]]:
            cid = candidate["candidate_id"]
            _need(cid not in features[qid], "unique complete bank identity required")
            features[qid][cid], lexical[qid][cid] = _feature(tokens, candidate)
        _need(set(features[qid]) == set(query["candidate_ids"]),
              "query refers to another complete candidate bank")
    pairs, differences = [], []
    for pair in corpus["pairs"]:
        if pair["task_role"] != "train":
            continue
        qid = pair["query_id"]
        _need(by_query[qid]["task_role"] == "train", "held-out pair cannot enter fit")
        positive = features[qid][pair["positive_candidate_id"]]
        negative = features[qid][pair["negative_candidate_id"]]
        pairs.append(pair)
        differences.append([a - b for a, b in zip(positive, negative)])
    _need(1 <= len(pairs) <= MAX_PAIRS, "bounded explicit reviewed train pairs required")
    norms = [math.fsum(x * x for x in row) for row in differences]
    _need(max(norms) > 0, "train pairs have no distinguishable public features")
    # Logistic Hessian <= mean(||difference||^2)/4 + L2. Double the
    # maximum-pair bound with a rounding margin and take a further half-step.
    smoothness = L2 + max(norms) / 2 + 1e-12
    step = 0.5 / smoothness
    profile = {"schema": FEATURE_SCHEMA, "names": _FEATURE_NAMES,
        "dimension": len(_FEATURE_NAMES), "tokenization": "Unicode alphanumeric sets, casefold, underscore split",
        "hash": "SHA256 UTF8 first-eight-bytes big-endian modulo fixed32/16",
        "query_text": "authored navigation_text, not complete prompt interpretation",
        "code_text": "replay-bound normalized_body, including normalization frontiers",
        "candidate_identity_label_split_or_old_AE_weights_as_features": False,
        "vocabulary_or_heldout_normalization_fit": False,
        "lexical_baseline": "complete query-token-coverage order; not prior native lexical prefix"}
    return corpus, features, lexical, pairs, differences, profile, smoothness, step


def _dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def _objective(weights, differences):
    margins = [_dot(weights, row) for row in differences]
    data = math.fsum(max(0.0, -z) + math.log1p(math.exp(-abs(z))) for z in margins) / len(margins)
    regularizer = L2 * _dot(weights, weights) / 2
    factors = [math.exp(-z) / (1 + math.exp(-z)) if z >= 0 else 1 / (1 + math.exp(z))
               for z in margins]
    gradient = [L2 * weights[i] - math.fsum(f * row[i] for f, row in zip(factors, differences))
                / len(differences) for i in range(len(weights))]
    _need(all(math.isfinite(x) for x in [data, regularizer, *gradient]),
          "nonfinite scratch-head objective or gradient")
    return {"objective": data + regularizer, "pair_logistic_loss": data,
            "L2_penalty": regularizer, "gradient_norm": math.sqrt(_dot(gradient, gradient))}, gradient


def _rank(scores):
    return [{"candidate_id": cid, "position": i, "score": score}
            for i, (cid, score) in enumerate(sorted(scores.items(), key=lambda x: (-x[1], x[0])), 1)]


def _metrics(rows, positives, *, available):
    if not available or not positives:
        return {"status": "no_ranking" if not available else "unjudged",
                "first_positive_rank": None, "reciprocal_first_rank": None,
                "recall_at": {str(k): None for k in (1, 5, 10)}}
    positions = [row["position"] for row in rows if row["candidate_id"] in positives]
    _need(len(positions) == len(positives), "complete ranking lost reviewed positive units")
    return {"status": "reviewed_positive_navigation_singleton_degenerate" if len(rows) == 1
            else "reviewed_positive_navigation_only", "first_positive_rank": min(positions),
        "reciprocal_first_rank": 1 / min(positions),
        "recall_at": {str(k): sum(i <= k for i in positions) / len(positives) for k in (1, 5, 10)}}


def _output(prepared, checkpoint, receipt):
    corpus, features, lexical, pairs, differences, profile, smoothness, step = prepared
    weights = checkpoint["weights"]
    results = []
    for query in corpus["queries"]:
        qid = query["query_id"]
        trained = _rank({cid: _dot(weights, vector) for cid, vector in features[qid].items()})
        reverse = [{**row, "position": i} for i, row in enumerate(reversed(trained), 1)]
        rankings = {"trained": trained, "model_off": [],
            "zero": _rank({cid: 0.0 for cid in features[qid]}),
            "reverse": reverse, "lexical": _rank(lexical[qid])}
        positives = {j["candidate_id"] for j in corpus["reviewed_judgments"]
                     if j["query_id"] == qid and j["label"] == "positive"}
        negatives = [j["candidate_id"] for j in corpus["reviewed_judgments"]
                     if j["query_id"] == qid and j["label"] == "negative_navigation"]
        results.append({"query_id": qid, "codebase_id": query["codebase_id"],
            "task_role": query["task_role"], "candidate_count": len(features[qid]),
            "degenerate_singleton_bank": len(features[qid]) == 1,
            "reviewed_positive_ids": sorted(positives), "reviewed_negative_navigation_ids": negatives,
            "other_candidate_label": "unjudged_not_implicit_negative",
            "residual_requirements": query["residual_requirements"],
            "rankings": rankings, "metrics": {name: _metrics(rows, positives, available=name != "model_off")
                for name, rows in rankings.items()}})
    result = {"schema": SCHEMA, "corpus_sha256": corpus["corpus_sha256"],
        "checkpoint": checkpoint, "training_receipt": receipt, "feature_profile": profile,
        "query_results": results, "historical_source_exposure": corpus["historical_exposure"],
        "new_autoencoder_fit": False, "head_is_new_scratch_parameters": True,
        "reverse_is_existing_order_permutation_not_fresh_inference": True,
        "zero_ties_use_candidate_ID_only_after_identical_scores": True,
        "generalized_ranking_gain_qualified": False,
        "complete_prompt_interpretation_qualified": False,
        "intermediate_training_history_authenticated_here": False, **_AUTHORITY}
    result["result_sha256"] = _digest(result)
    return _json(result)


def train_terminal_codebase_intent_ranker(*, corpus_receipt, original_inputs, epochs=64):
    """Fit one scratch binary64 head using only explicit reviewed train pairs."""
    _need(type(epochs) is int and 1 <= epochs <= MAX_EPOCHS, "1..128 fixed epochs required")
    prepared = _prepare(corpus_receipt, original_inputs)
    corpus, features, lexical, pairs, differences, profile, smoothness, step = prepared
    weights, trace = [0.0] * len(_FEATURE_NAMES), []
    for epoch in range(epochs + 1):
        observation, gradient = _objective(weights, differences)
        trace.append({"epoch": epoch, **observation, "weights_sha256": _digest(weights)})
        if epoch < epochs:
            weights = [w - step * g for w, g in zip(weights, gradient)]
            _need(all(math.isfinite(x) for x in weights), "nonfinite fitted head weights")
    checkpoint = {"schema": CHECKPOINT_SCHEMA, "corpus_sha256": corpus["corpus_sha256"],
        "feature_profile_sha256": _digest(profile), "weights": weights,
        "weights_sha256": _digest(weights), "initialization": "new_all_zero_scratch_head"}
    receipt = {"schema": RECEIPT_SCHEMA, "corpus_sha256": corpus["corpus_sha256"],
        "checkpoint_sha256": _digest(checkpoint), "feature_profile_sha256": _digest(profile),
        "epochs": epochs, "optimizer": "deterministic_binary64_fullbatch_gradient_descent",
        "objective": "mean_train_pair_softplus_negative_margin_plus_L2_half_squared_norm",
        "L2": L2, "smoothness_upper_bound_numeric": smoothness, "step_size": step,
        "numeric_bound_is_formal_certificate": False,
        "train_query_ids": sorted({pair["query_id"] for pair in pairs}), "train_pairs": pairs,
        "train_pair_feature_differences_sha256": _digest(differences),
        "fit_validation_or_test_labels_used": False, "fit_selection_or_normalization_on_heldout": False,
        "trace": trace, "weights_changed_from_zero": any(w != 0 for w in weights),
        "finite_recorded_trace": True, "asymptotic_convergence_proved": False}
    receipt["receipt_sha256"] = _digest(receipt)
    return _output(prepared, checkpoint, receipt)


def validate_terminal_codebase_intent_ranker(receipt, *, corpus_receipt, original_inputs,
        expected_checkpoint_sha256=None, expected_training_receipt_sha256=None):
    """Replay corpus, public features and inference without fitting again.

Externally frozen digests bind actual fit provenance. Consistent self-resealed
weights/trace alone do not authenticate a training run or intermediate history.
"""
    supplied = _json(receipt)
    prepared = _prepare(corpus_receipt, original_inputs)
    corpus, features, lexical, pairs, differences, profile, smoothness, step = prepared
    checkpoint, training = supplied["checkpoint"], supplied["training_receipt"]
    _need(set(checkpoint) == {"schema", "corpus_sha256", "feature_profile_sha256", "weights",
        "weights_sha256", "initialization"} and checkpoint["schema"] == CHECKPOINT_SCHEMA
        and checkpoint["corpus_sha256"] == corpus["corpus_sha256"]
        and checkpoint["feature_profile_sha256"] == _digest(profile)
        and checkpoint["initialization"] == "new_all_zero_scratch_head", "head checkpoint differs from corpus/profile")
    weights = checkpoint["weights"]
    _need(type(weights) is list and len(weights) == len(_FEATURE_NAMES)
        and all(type(w) is float and math.isfinite(w) for w in weights)
        and checkpoint["weights_sha256"] == _digest(weights), "bounded binary64 head weights required")
    epochs = training["epochs"]
    _need(type(epochs) is int and 1 <= epochs <= MAX_EPOCHS, "bounded recorded epochs required")
    trace = training["trace"]
    _need(type(trace) is list and len(trace) == epochs + 1, "complete finite epoch trace required")
    for i, row in enumerate(trace):
        _need(type(row) is dict and set(row) == {"epoch", "objective", "pair_logistic_loss", "L2_penalty",
            "gradient_norm", "weights_sha256"} and type(row["epoch"]) is int and row["epoch"] == i
            and type(row["weights_sha256"]) is str
            and re.fullmatch(r"[0-9a-f]{64}", row["weights_sha256"]) is not None
            and all(type(row[k]) is float and math.isfinite(row[k]) and row[k] >= 0
                    for k in ("objective", "pair_logistic_loss", "L2_penalty", "gradient_norm")),
              "recorded trace has invalid finite fields")
    for row, vector in [(trace[0], [0.0] * len(weights)), (trace[-1], weights)]:
        observation, _ = _objective(vector, differences)
        _need(row == {"epoch": row["epoch"], **observation, "weights_sha256": _digest(vector)},
              "recorded endpoint differs from supplied head inference")
    expected = {"schema": RECEIPT_SCHEMA, "corpus_sha256": corpus["corpus_sha256"],
        "checkpoint_sha256": _digest(checkpoint), "feature_profile_sha256": _digest(profile),
        "epochs": epochs, "optimizer": "deterministic_binary64_fullbatch_gradient_descent",
        "objective": "mean_train_pair_softplus_negative_margin_plus_L2_half_squared_norm",
        "L2": L2, "smoothness_upper_bound_numeric": smoothness, "step_size": step,
        "numeric_bound_is_formal_certificate": False,
        "train_query_ids": sorted({pair["query_id"] for pair in pairs}), "train_pairs": pairs,
        "train_pair_feature_differences_sha256": _digest(differences),
        "fit_validation_or_test_labels_used": False, "fit_selection_or_normalization_on_heldout": False,
        "trace": trace, "weights_changed_from_zero": any(w != 0 for w in weights),
        "finite_recorded_trace": True, "asymptotic_convergence_proved": False}
    expected["receipt_sha256"] = _digest(expected)
    _need(_digest(training) == _digest(expected), "training membership/profile/receipt differs from original inputs")
    if expected_checkpoint_sha256 is not None:
        _need(_digest(checkpoint) == expected_checkpoint_sha256, "externally frozen checkpoint differs")
    if expected_training_receipt_sha256 is not None:
        _need(training["receipt_sha256"] == expected_training_receipt_sha256,
              "externally frozen fit receipt differs")
    rebuilt = _output(prepared, checkpoint, expected)
    _need(_digest(supplied) == _digest(rebuilt), "ranker inference/metrics/authority differs from original-input replay")
    return rebuilt


__all__ = ["SCHEMA", "FEATURE_SCHEMA", "IntentRankerTrainingError",
    "train_terminal_codebase_intent_ranker", "validate_terminal_codebase_intent_ranker"]
