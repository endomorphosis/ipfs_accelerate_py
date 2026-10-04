"""Pure full-bank consumption of a pinned development navigation ranking.

Independent original inputs, current source observations and a pinned historical
metadata readback are replayed before ordering. No files or native stores are
opened here. Navigation order supplies no requirements, facts or planning
authority; the full original IntentIR and its unknown residuals remain intact.
"""
from __future__ import annotations

import ast
import hashlib
import re

from .terminal_codebase_intent_corpus import (
    _AUTHORITY, _digest, _json, _wire, validate_terminal_intent_relevance_corpus,
)
from .terminal_codebase_intent_ranker_training import validate_terminal_codebase_intent_ranker

SCHEMA = "terminal-codebase-analysis-order@1"
QUERY_SCHEMA = "terminal-codebase-analysis-query@1"
METHODS = ("trained", "lexical", "zero", "reverse", "model_off")
_HASH = re.compile(r"[0-9a-f]{64}")


class TerminalAnalysisOrderError(ValueError):
    """Independent query, source, model or metadata bindings differ."""


def _need(condition, message):
    if condition is not True:
        raise TerminalAnalysisOrderError(message)


def _hash(value, name, *, prefixed=False):
    _need(type(value) is str and (not prefixed or value.startswith("sha256:"))
          and _HASH.fullmatch(value[7:] if prefixed else value) is not None,
          "exact " + name + " SHA256 required")


def _same(left, right, message):
    _need(_wire(left) == _wire(right), message)


def _replay(corpus_receipt, ranker_receipt, original_inputs, **pins):
    try:
        corpus = validate_terminal_intent_relevance_corpus(corpus_receipt, **original_inputs)
        ranker = validate_terminal_codebase_intent_ranker(ranker_receipt,
            corpus_receipt=corpus, original_inputs=original_inputs, **pins)
    except (ValueError, KeyError, TypeError) as error:
        raise TerminalAnalysisOrderError("native corpus/ranker replay refused: " + str(error)) from error
    return corpus, ranker


def bind_terminal_analysis_query(corpus_receipt, query_id):
    """Select a strict query identity; the consumer independently replays it."""
    corpus = _json(corpus_receipt)
    _need(type(query_id) is str, "exact query identity required")
    selected = [row for row in corpus["queries"] if row["query_id"] == query_id]
    _need(len(selected) == 1, "one exact corpus query required")
    query = selected[0]
    sources = query["intent_document"]["sources"]
    _need(type(sources) is list and len(sources) == 2,
          "both original-instruction and authored-navigation SourceRefs required")
    return _json({"schema": QUERY_SCHEMA, **{key: query[key] for key in (
        "query_id", "codebase_id", "instruction_sha256", "navigation_sha256",
        "native_intent_sha256", "statement_ids")}, "source_refs": sources})


def _project(corpus, ranker, original_inputs):
    from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formula_corpus import _qualified_functions

    nodes = {}
    for source in original_inputs["source_records"]:
        for node, symbol, _ in _qualified_functions(ast.parse(source["source_text"])):
            key = (source["codebase_id"], source["path"], symbol, node.lineno)
            _need(key not in nodes, "duplicate exact original function identity")
            nodes[key] = node
    # This profile is the complete v1 experiment_inputs.metadata_records
    # projection, including its fixed coverage declaration and empty contracts.
    records = {"sources": corpus["sources"], "ast": [], "kg": [], "vectors": [],
        "contracts": [], "queries": corpus["queries"], "judgments": corpus["reviewed_judgments"],
        "split_history": [corpus["native_split_guard"], corpus["historical_exposure"]],
        "ranker_checkpoint": [ranker["checkpoint"]], "ranker_training": [ranker["training_receipt"]],
        "ranking": [], "coverage_status": [{
            "scope": "complete admitted public initial Python function banks only",
            "function_count": corpus["complete_candidate_count"],
            "contracts": "unknown_not_inferred_from_navigation",
            "logical_projections": "not_generated_in_this_head_experiment",
            "Lean_compilations": 0, "proof_cache_hits_verified_here": 0,
            "abstract_terminal_method_implemented": False,
            "whole_repository_metadata_coverage": False,
            "proof_authority": False, "asymptotic_convergence_proved": False}]}
    for codebase, candidates in corpus["candidate_banks"].items():
        for candidate in candidates:
            key = (codebase, candidate["path"], candidate["symbol"], candidate["line"])
            _need(key in nodes, "candidate is absent from independent original AST")
            dumped = ast.dump(nodes[key], include_attributes=False)
            _need(hashlib.sha256(dumped.encode()).hexdigest() == candidate["ast_sha256"],
                  "candidate exact AST differs from independent originals")
            binding = {key: candidate[key] for key in (
                "candidate_id", "codebase_id", "path", "symbol", "line", "end_line",
                "source_sha256", "ast_sha256", "source_binding")}
            records["ast"].append({**binding, "AST_dump": dumped, "proof_authority": False})
            records["vectors"].append({**binding, "profile": "native_python_AST44_L2_features",
                "vector": candidate["features"], "learned_autoencoder_latent": False,
                "proof_authority": False})
            records["kg"].append({"kind": "source_contains_function", **binding,
                "edge_semantics": "syntactic_membership_only", "proof_authority": False})
    records["kg"].extend({"kind": "authored_navigation_judgment", **judgment,
        "edge_semantics": "reviewed_navigation_not_behavioral_entailment"}
        for judgment in corpus["reviewed_judgments"])
    for query in ranker["query_results"]:
        for method, rows in query["rankings"].items():
            records["ranking"].extend({"query_id": query["query_id"], "codebase_id": query["codebase_id"],
                "task_role": query["task_role"], "method": method, **row,
                "corpus_sha256": corpus["corpus_sha256"], "proof_authority": False} for row in rows)
    return _json(records)


def project_terminal_intent_metadata(corpus_receipt, ranker_receipt, original_inputs):
    """Reproduce all twelve historical payload families without hydration."""
    corpus, ranker = _replay(corpus_receipt, ranker_receipt, original_inputs)
    return _project(corpus, ranker, original_inputs)


def _metadata(payloads, metadata_rows, metadata_report, expected_sha256):
    from . import codebase_ir_metadata as native

    _hash(expected_sha256, "external metadata-report")
    report = _json(metadata_report)
    _need(_digest(report) == expected_sha256, "externally pinned historical metadata report differs")
    fields = {"schema", "output", "manifest_sha256", "source_snapshot_sha256", "row_count",
        "row_root_sha256", "family_counts", "family_digests", "family_views", "exports",
        "native_runtime", "limits", "lake_layout", "lake_packet_count", "lake_snapshot_ids",
        "lake_snapshot_digest", "lake_receipts", "qualification", "fresh_process_readback"}
    _need(set(report) == fields and report["schema"] == native.REPORT_SCHEMA,
          "closed native historical metadata-readback profile required")
    _same(report["qualification"], native.QUALIFICATION, "metadata authority or qualification differs")
    _same(report["limits"], native.LIMITS, "native metadata limits differ")
    _same(report["lake_layout"], native.LAKE_LAYOUT, "native row-packet layout differs")
    _need(type(report["output"]) is str and bool(report["output"]), "historical metadata namespace required")
    for field in ("manifest_sha256", "source_snapshot_sha256", "row_root_sha256", "lake_snapshot_digest"):
        _hash(report[field], field, prefixed=True)
    rows = native._families(payloads, report["source_snapshot_sha256"])
    _same(_json(metadata_rows), rows, "complete native metadata rows differ from original source/model projection")
    _need(type(report["row_count"]) is int and report["row_count"] == sum(map(len, rows.values()))
          and report["row_root_sha256"] == native._row_root(rows), "complete metadata population/root differs")
    _same(report["family_counts"], {name: len(values) for name, values in rows.items()},
          "complete metadata family counts differ")
    _same(report["family_digests"], {name: native._digest(native._row_array(values))
          for name, values in rows.items()}, "complete metadata family digests differ")
    _same(report["family_views"], {name: "family_" + name for name in rows}, "native family views differ")
    exports = {}
    for name, values in rows.items():
        raw = b"".join(native._wire(row) + b"\n" for row in values)
        exports[name] = {"relative_path": "exports/" + name + ".jsonl",
                         "bytes": len(raw), "sha256": native._digest(raw)}
    _same(report["exports"], exports, "complete export bytes and identities differ")
    packets = native._packets(rows)
    _need(type(report["lake_packet_count"]) is int and report["lake_packet_count"] == len(packets),
          "complete historical packet count differs")
    receipts, event_ids, snapshot_ids = report["lake_receipts"], [], []
    _need(type(receipts) is list and 0 < len(receipts) <= len(packets), "bounded historical lake receipts required")
    receipt_fields = {"schema", "admitted", "production_activated", "batch_id", "event_count", "event_ids",
        "event_payloads_verified", "history_id", "scope", "snapshot_id", "source_id"}
    for receipt in receipts:
        _need(type(receipt) is dict and set(receipt) == receipt_fields
              and receipt["schema"] == "autoencoder-ducklake-commit-v1"
              and receipt["admitted"] is False and receipt["production_activated"] is False
              and receipt["event_payloads_verified"] is True and receipt["scope"] == "isolated_history"
              and receipt["source_id"] == "codebase-ir-experiment:" + report["source_snapshot_sha256"][7:]
              and type(receipt["snapshot_id"]) is int and receipt["snapshot_id"] >= 1
              and type(receipt["event_ids"]) is list and type(receipt["event_count"]) is int
              and receipt["event_count"] == len(receipt["event_ids"]) > 0,
              "historical isolated lake receipt profile differs")
        for field in ("batch_id", "history_id"):
            _hash(receipt[field], "lake " + field, prefixed=True)
        for event_id in receipt["event_ids"]:
            _hash(event_id, "lake event", prefixed=True)
        event_ids.extend(receipt["event_ids"])
        snapshot_ids.append(receipt["snapshot_id"])
    _need(len(event_ids) == len(set(event_ids)) == len(packets)
          and set(event_ids) == {packet["packet_id"] for packet in packets},
          "historical receipts lost or changed a complete native row packet")
    _same(report["lake_snapshot_ids"], snapshot_ids, "historical snapshot receipt identities differ")
    _need(report["lake_snapshot_digest"] == native._digest(native._json(receipts)),
          "historical snapshot receipt digest differs")
    fresh = report["fresh_process_readback"]
    _same(fresh, {"method": "new_python_process_native_duckdb_and_ducklake_readback",
        "verified": True, **{field: report[field] for field in (
            "manifest_sha256", "row_count", "row_root_sha256", "lake_snapshot_digest")}},
        "pinned historical native fresh-process readback binding differs")
    runtime = report["native_runtime"]
    _need(type(runtime) is dict and set(runtime) == {"duckdb", "platform", "extensions"}
          and all(type(runtime[field]) is str and bool(runtime[field]) for field in ("duckdb", "platform"))
          and type(runtime["extensions"]) is dict
          and set(runtime["extensions"]) == {"ducklake", "httpfs", "quack"},
          "historical native runtime profile differs")
    for extension, pin in runtime["extensions"].items():
        _hash(pin, "historical " + extension)
    return report


def build_terminal_codebase_analysis_order(*, corpus_receipt, original_inputs, ranker_receipt,
        expected_corpus_sha256, expected_checkpoint_sha256, expected_training_receipt_sha256,
        query_binding, current_source_records, metadata_rows, metadata_report,
        expected_metadata_report_sha256, enabled=False, method="trained"):
    """Consume only a complete current-source bank; abstain from planning.

    The caller supplies genuine source observations and externally retained
    artifact pins. Their origins are not authenticated by this pure function.
    Passive feature/head replay neither fits a model nor reopens native SQL.
    """
    _need(type(enabled) is bool and type(method) is str and method in METHODS,
          "exact opt-in and closed advisory ordering method required")
    _hash(expected_corpus_sha256, "external corpus", prefixed=True)
    _hash(expected_checkpoint_sha256, "external checkpoint")
    _hash(expected_training_receipt_sha256, "external fit-receipt")
    originals = _json(original_inputs)
    corpus, ranker = _replay(corpus_receipt, ranker_receipt, originals,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_training_receipt_sha256=expected_training_receipt_sha256)
    _need(corpus["corpus_sha256"] == expected_corpus_sha256, "externally pinned corpus differs")
    current = _json(current_source_records)
    _need(type(current) is list and all(type(row) is dict for row in current), "complete current source records required")
    current.sort(key=lambda row: (row.get("codebase_id", ""), row.get("path", "")))
    expected_sources = sorted(originals["source_records"], key=lambda row: (row["codebase_id"], row["path"]))
    _same(current, expected_sources, "independently observed complete current sources differ from admitted originals")
    binding = _json(query_binding)
    _need(type(binding) is dict and type(binding.get("query_id")) is str, "strict query binding required")
    _same(binding, bind_terminal_analysis_query(corpus, binding["query_id"]), "exact query binding differs")
    query = next(row for row in corpus["queries"] if row["query_id"] == binding["query_id"])
    report = _metadata(_project(corpus, ranker, originals), metadata_rows,
                       metadata_report, expected_metadata_report_sha256)
    bank = corpus["candidate_banks"][query["codebase_id"]]
    by_id = {candidate["candidate_id"]: candidate for candidate in bank}
    selected = next(row for row in ranker["query_results"] if row["query_id"] == query["query_id"])
    active = enabled and method != "model_off"
    order = selected["rankings"][method] if active else [
        {"candidate_id": candidate["candidate_id"], "position": i, "score": None}
        for i, candidate in enumerate(bank, 1)]
    ids = [row["candidate_id"] for row in order]
    _need(len(ids) == len(set(ids)) == len(bank) and set(ids) == set(by_id),
          "advisory ordering must preserve the complete candidate bank exactly once")
    result = {"schema": SCHEMA, "enabled": enabled, "requested_method": method,
        "effective_method": method if active else "original_full_bank_order",
        "corpus_sha256": expected_corpus_sha256, "checkpoint_sha256": expected_checkpoint_sha256,
        "training_receipt_sha256": expected_training_receipt_sha256,
        "query_binding": binding, "current_source_records_sha256": _digest(current),
        "metadata_report_sha256": expected_metadata_report_sha256,
        "metadata_row_root_sha256": report["row_root_sha256"], "metadata_row_count": report["row_count"],
        "metadata_scope": "exact_twelve_family_historical_native_readback_replayed_no_native_reopen",
        "candidate_count": len(bank), "ordered_candidate_ids": ids, "order": order,
        "ordered_candidates": [by_id[cid] for cid in ids],
        "intent_document": query["intent_document"], "residual_requirements": query["residual_requirements"],
        "original_prompt_residuals": query["original_prompt_residuals"],
        "complete_candidate_population_preserved": True, "navigation_analysis_order_only": True,
        "planning_handoff": "abstained", "current_source_origin_authenticated_here": False,
        "native_stores_reopened_here": False, "fit_executed_here": False,
        "full_prompt_interpreted": False, "generalized_ranking_gain_qualified": False, **_AUTHORITY}
    result["analysis_order_sha256"] = _digest(result)
    return _json(result)


def validate_terminal_codebase_analysis_order(receipt, **original_arguments):
    """Rebuild against independent original inputs and external current pins."""
    supplied = _json(receipt)
    rebuilt = build_terminal_codebase_analysis_order(**original_arguments)
    _same(supplied, rebuilt, "analysis order differs from complete independent replay")
    return rebuilt


__all__ = ["SCHEMA", "QUERY_SCHEMA", "METHODS", "TerminalAnalysisOrderError",
    "project_terminal_intent_metadata", "bind_terminal_analysis_query",
    "build_terminal_codebase_analysis_order", "validate_terminal_codebase_analysis_order"]
