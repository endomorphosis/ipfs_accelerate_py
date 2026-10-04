"""Pure transfer of a pinned navigation head to expanded development banks.

The original checkpoint and fit receipt keep their original corpus identity.
New labels affect evaluation metrics only. Every old source, query, judgment,
split, function bank and train feature must survive unchanged; declared history
may accumulate references. No optimizer, filesystem, SQL or checker runs here.
"""
from __future__ import annotations

import re

from .terminal_codebase_intent_corpus import (
    _AUTHORITY, _digest, _json, _wire, validate_terminal_intent_relevance_corpus,
)
from . import terminal_codebase_intent_ranker_training as head

SCHEMA = "terminal-codebase-intent-ranker-transfer@1"
LINEAGE_SCHEMA = "terminal-codebase-intent-ranker-transfer-lineage@1"
_HASH = re.compile(r"[0-9a-f]{64}")


class IntentRankerTransferError(ValueError):
    """Original fit lineage or expanded evaluation membership differs."""


def _need(condition, message):
    if condition is not True:
        raise IntentRankerTransferError(message)


def _same(left, right, message):
    _need(_wire(left) == _wire(right), message)


def _hash(value, name, *, prefixed=False):
    _need(type(value) is str and (not prefixed or value.startswith("sha256:"))
          and _HASH.fullmatch(value[7:] if prefixed else value) is not None,
          "exact externally pinned " + name + " SHA256 required")


def _subset(old_rows, new_rows, keys, label):
    old = {tuple(row[key] for key in keys): row for row in old_rows}
    new = {tuple(row[key] for key in keys): row for row in new_rows}
    _need(len(old) == len(old_rows) and len(new) == len(new_rows), "unique " + label + " identities required")
    _need(set(old) <= set(new), "expanded corpus removed original " + label)
    for key, row in old.items():
        _same(row, new[key], "expanded corpus changed original " + label)
    return old, new


def _history(old, new):
    _same(old["schema"], new["schema"], "history declaration schema differs")
    for family, keys, identity in (
        ("source_history", ("codebase_id", "path"), ("codebase_id", "path", "source_sha256")),
        ("query_history", ("query_id",), ("query_id", "instruction_sha256")),
    ):
        original = {tuple(row[key] for key in keys): row for row in old[family]}
        expanded = {tuple(row[key] for key in keys): row for row in new[family]}
        _need(set(original) <= set(expanded), "expanded history removed original identities")
        for key, row in original.items():
            current = expanded[key]
            _same({name: row[name] for name in identity}, {name: current[name] for name in identity},
                  "historical original source/query hash changed")
            for field in ("prior_fit_refs", "prior_review_refs"):
                _need(set(row[field]) <= set(current[field]), "expanded history removed prior exposure references")


def _expansion(old, new, original, expanded):
    source_keys, all_sources = _subset(original["source_records"], expanded["source_records"],
                                     ("codebase_id", "path"), "source declarations")
    old_codebases = set(old["candidate_banks"])
    _need(all(row["codebase_id"] not in old_codebases for key, row in all_sources.items() if key not in source_keys),
          "new sources may not enlarge an original complete codebase bank")
    queries, all_queries = _subset(original["query_records"], expanded["query_records"],
                                  ("query_id",), "query declarations")
    _need(bool(set(all_queries) - set(queries)), "expanded evaluation needs additional declared queries")
    for key in set(all_queries) - set(queries):
        query = all_queries[key]
        _need(expanded["split_assignments"][query["query_id"]] != "train"
              and query["codebase_id"] not in old_codebases,
              "additional evaluation queries require new nontrain codebase families")
    for qid, role in original["split_assignments"].items():
        _same(role, expanded["split_assignments"][qid], "original development task role changed")
    # A query has several explicit judgments; use its complete candidate
    # reference as the key rather than collapsing rows by query identity.
    old_judgments = {(_digest(row["candidate_ref"]), row["query_id"]): row for row in original["reviewed_judgments"]}
    new_judgments = {(_digest(row["candidate_ref"]), row["query_id"]): row for row in expanded["reviewed_judgments"]}
    _need(set(old_judgments) <= set(new_judgments), "expanded corpus removed original reviewed judgments")
    for key, row in old_judgments.items():
        _same(row, new_judgments[key], "original reviewed navigation judgment changed")
    old_qids = {key[0] for key in queries}
    _need(all(row["query_id"] not in old_qids for key, row in new_judgments.items() if key not in old_judgments),
          "additional judgments may not alter original query coverage")
    old_queries = {row["query_id"]: row for row in old["queries"]}
    new_queries = {row["query_id"]: row for row in new["queries"]}
    for qid, query in old_queries.items():
        _same(query, new_queries[qid], "original native intent, residuals or complete query bank changed")
    for codebase in old_codebases:
        _same(old["candidate_banks"][codebase], new["candidate_banks"][codebase],
              "original complete source/AST/span/feature bank changed")
    _history(old["historical_exposure"], new["historical_exposure"])


def infer_terminal_codebase_intent_transfer(*, training_corpus_receipt, training_original_inputs,
        training_ranker_receipt, evaluation_corpus_receipt, evaluation_original_inputs,
        expected_training_corpus_sha256, expected_evaluation_corpus_sha256,
        expected_checkpoint_sha256, expected_training_receipt_sha256):
    """Replay frozen lineage and apply the same head to every evaluation unit.

    External pins and original input origins are caller-owned. This function
    does not attest publication time, global exposure or a blind evaluation.
    """
    for value, name in ((expected_training_corpus_sha256, "training corpus"),
                        (expected_evaluation_corpus_sha256, "evaluation corpus")):
        _hash(value, name, prefixed=True)
    _hash(expected_checkpoint_sha256, "checkpoint")
    _hash(expected_training_receipt_sha256, "fit receipt")
    original, expanded = _json(training_original_inputs), _json(evaluation_original_inputs)
    try:
        old = validate_terminal_intent_relevance_corpus(training_corpus_receipt, **original)
        new = validate_terminal_intent_relevance_corpus(evaluation_corpus_receipt, **expanded)
        trained = head.validate_terminal_codebase_intent_ranker(training_ranker_receipt,
            corpus_receipt=old, original_inputs=original,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_training_receipt_sha256=expected_training_receipt_sha256)
        old_prepared, new_prepared = head._prepare(old, original), head._prepare(new, expanded)
    except (ValueError, TypeError, KeyError) as error:
        raise IntentRankerTransferError("native corpus/head replay refused: " + str(error)) from error
    _need(old["corpus_sha256"] == expected_training_corpus_sha256
          and new["corpus_sha256"] == expected_evaluation_corpus_sha256
          and expected_training_corpus_sha256 != expected_evaluation_corpus_sha256,
          "independently pinned distinct training/evaluation corpora required")
    _expansion(old, new, original, expanded)
    _same(old_prepared[3:], new_prepared[3:], "original train pairs, differences or fixed feature/update profile changed")
    train_qids = trained["training_receipt"]["train_query_ids"]
    old_features = {qid: old_prepared[1][qid] for qid in train_qids}
    new_features = {qid: new_prepared[1][qid] for qid in train_qids}
    _same(old_features, new_features, "complete original train-query feature population changed")
    checkpoint, fit = trained["checkpoint"], trained["training_receipt"]
    results = []
    for query in new["queries"]:
        qid = query["query_id"]
        scores = {cid: head._dot(checkpoint["weights"], vector) for cid, vector in new_prepared[1][qid].items()}
        ranking = head._rank(scores)
        controls = {"trained": ranking, "model_off": [],
            "zero": head._rank({cid: 0.0 for cid in scores}),
            "reverse": [{**row, "position": i} for i, row in enumerate(reversed(ranking), 1)],
            "lexical": head._rank(new_prepared[2][qid])}
        positives = {row["candidate_id"] for row in new["reviewed_judgments"]
                     if row["query_id"] == qid and row["label"] == "positive"}
        negatives = [row["candidate_id"] for row in new["reviewed_judgments"]
                     if row["query_id"] == qid and row["label"] == "negative_navigation"]
        results.append({"query_id": qid, "codebase_id": query["codebase_id"], "task_role": query["task_role"],
            "candidate_count": len(scores), "degenerate_singleton_bank": len(scores) == 1,
            "reviewed_positive_ids": sorted(positives), "reviewed_negative_navigation_ids": negatives,
            "other_candidate_label": "unjudged_not_implicit_negative",
            "residual_requirements": query["residual_requirements"], "rankings": controls,
            "metrics": {name: head._metrics(rows, positives, available=name != "model_off")
                        for name, rows in controls.items()}})
    lineage = {"schema": LINEAGE_SCHEMA, "training_corpus_sha256": expected_training_corpus_sha256,
        "evaluation_corpus_sha256": expected_evaluation_corpus_sha256,
        "checkpoint_sha256": expected_checkpoint_sha256, "training_receipt_sha256": expected_training_receipt_sha256,
        "original_ranker_result_sha256": trained["result_sha256"],
        "feature_profile_sha256": checkpoint["feature_profile_sha256"],
        "complete_train_query_features_sha256": _digest(old_features),
        "train_pair_feature_differences_sha256": fit["train_pair_feature_differences_sha256"],
        "train_query_ids": train_qids, "train_pairs_sha256": _digest(old_prepared[3]),
        "original_source_query_judgment_split_and_bank_preserved": True,
        "historical_references_policy": "monotonic_supersets_for_exact_original_identities",
        "train_features_and_update_profile_unchanged": True,
        "original_checkpoint_and_fit_receipt_preserved": True}
    result = {"schema": SCHEMA, "training_corpus_sha256": expected_training_corpus_sha256,
        "evaluation_corpus_sha256": expected_evaluation_corpus_sha256,
        "checkpoint": checkpoint, "training_receipt": fit, "transfer_lineage": lineage,
        "feature_profile": trained["feature_profile"], "query_results": results,
        "historical_source_exposure": new["historical_exposure"],
        "new_fit": False, "new_fit_receipt_created": False, "new_optimizer_steps": 0,
        "new_autoencoder_fit": False, "reverse_is_existing_order_permutation_not_fresh_inference": True,
        "zero_ties_use_candidate_ID_only_after_identical_scores": True,
        "evaluation_scope": "expanded_predeclared_development_navigation_not_blind_holdout",
        "input_origins_and_publication_time_authenticated_here": False,
        "generalized_ranking_gain_qualified": False, "complete_prompt_interpretation_qualified": False,
        "intermediate_training_history_authenticated_here": False, "planning_handoff": "abstained", **_AUTHORITY}
    result["transfer_sha256"] = _digest(result)
    return _json(result)


def validate_terminal_codebase_intent_transfer(receipt, **original_arguments):
    """Rebuild all transfer controls from independent inputs without fitting."""
    supplied = _json(receipt)
    rebuilt = infer_terminal_codebase_intent_transfer(**original_arguments)
    _same(supplied, rebuilt, "transfer differs from complete independent lineage/inference replay")
    return rebuilt


def project_terminal_intent_transfer_metadata(receipt, **original_arguments):
    """Preserve old fit artifacts and new evaluation rows in thirteen families."""
    from .terminal_codebase_analysis_order import _project

    transfer = validate_terminal_codebase_intent_transfer(receipt, **original_arguments)
    expanded = _json(original_arguments["evaluation_original_inputs"])
    corpus = validate_terminal_intent_relevance_corpus(original_arguments["evaluation_corpus_receipt"], **expanded)
    records = _project(corpus, {key: transfer[key] for key in (
        "checkpoint", "training_receipt", "query_results")}, expanded)
    binding = {key: transfer["transfer_lineage"][key] for key in (
        "training_corpus_sha256", "evaluation_corpus_sha256", "checkpoint_sha256", "training_receipt_sha256")}
    for row in records["ranking"]:
        row.update(binding)
    records["ranker_transfer"] = [{"schema": SCHEMA, "transfer_sha256": transfer["transfer_sha256"],
        "transfer_lineage": transfer["transfer_lineage"], "query_count": len(transfer["query_results"]),
        "complete_candidate_count": corpus["complete_candidate_count"],
        "new_fit": False, "new_optimizer_steps": 0, "planning_handoff": "abstained", **_AUTHORITY}]
    records["coverage_status"][0].update({
        "scope": "complete admitted public initial Python function banks; original head cross-corpus evaluation only",
        "logical_projections": "not_generated_in_this_transfer_experiment",
        "cross_corpus_transfer_inference_only": True, "new_optimizer_steps": 0,
        "new_fit_receipt_created": False, **binding})
    return _json(records)


__all__ = ["SCHEMA", "LINEAGE_SCHEMA", "IntentRankerTransferError",
    "infer_terminal_codebase_intent_transfer", "validate_terminal_codebase_intent_transfer",
    "project_terminal_intent_transfer_metadata"]
