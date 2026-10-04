"""Post-fit development relevance of retained CodebaseIR nomination orders.

No fit, inference, query, checker, filesystem or worker runs in this consumer.
Labels are explicit positive units, not a complete semantic interpretation or
blind holdout. The full native feature universe is scored before authored focus
selection; a retained lexical prefix never supplies an invented ranking tail.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import PurePosixPath

from .terminal_codebase_intent_training_join import (
    _AUTHORITY, _digest, _json, validate_terminal_intent_training_join,
)

SCHEMA = "terminal-codebase-intent-relevance@1"
LABEL_SCHEMA = "terminal-codebase-intent-development-labels@1"
_CUTOFFS = (1, 5, 10)
_LABEL_POLICY = "postfit_authored_development"


class RelevanceEvaluationError(ValueError):
    """A development label or ranking differs from the original source join."""


def _need(condition, message):
    if condition is not True:
        raise RelevanceEvaluationError(message)


def _fields(value, names, label):
    _need(type(value) is dict and set(value) == set(names),
          "exact " + label + " fields required")


def _text(value, label):
    _need(type(value) is str and 0 < len(value) <= 512, "bounded " + label + " required")


def _inventory(arguments):
    """Reconstruct exact byte/AST identities for every native feature row."""
    sources = {row["path"]: row["source_text"].encode()
               for row in arguments["source_records"]}
    units = {}
    for path in arguments["features"]["paths"]:
        if not path.endswith(".py"):
            continue
        raw = sources[path]
        offsets = [0]
        for line in raw.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))
        try:
            tree = ast.parse(raw.decode("utf-8"))
        except (SyntaxError, UnicodeError, ValueError):
            continue  # The original join separately replays every frontier.

        def visit(node, prefix=""):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbol = prefix + node.name
                start = offsets[node.lineno - 1] + node.col_offset
                end = offsets[node.end_lineno - 1] + node.end_col_offset
                key = (path, symbol, node.lineno)
                _need(key not in units, "ambiguous complete source unit identity")
                units[key] = {
                    "path": path, "symbol": symbol, "line": node.lineno,
                    "end_line": node.end_lineno,
                    "source_sha256": hashlib.sha256(raw).hexdigest(),
                    "source_ast_sha256": hashlib.sha256(
                        ast.dump(node, include_attributes=False).encode()).hexdigest(),
                    "source_span": {"start_byte": start, "end_byte": end,
                                    "sha256": hashlib.sha256(raw[start:end]).hexdigest()},
                }
                prefix += node.name + "."
            elif isinstance(node, ast.ClassDef):
                prefix += node.name + "."
            for child in ast.iter_child_nodes(node):
                visit(child, prefix)

        visit(tree)
    result = []
    for row in arguments["features"]["rows"]:
        key = (row["path"], row["symbol"], row["line"])
        _need(key in units and units[key]["source_ast_sha256"] == row["ast_sha256"],
              "feature row differs from complete source AST unit")
        result.append({"feature_row_id": row["row_id"], **units[key]})
    _need(len({row["feature_row_id"] for row in result}) == len(result),
          "unique complete candidate universe required")
    return result


def _labels(manifest, join, inventory):
    _fields(manifest, {"schema", "label_policy", "review_ref", "join_sha256",
        "source_context_sha256", "candidate_universe_sha256", "cases"}, "label manifest")
    _text(manifest["review_ref"], "independent development review reference")
    match = join["original_native_match"]
    _need(manifest["schema"] == LABEL_SCHEMA and manifest["label_policy"] == _LABEL_POLICY,
          "post-fit authored development labels required; no held-out claim")
    _need(manifest["join_sha256"] == join["join_sha256"]
          and manifest["source_context_sha256"] == match["current_source_snapshot"]["source_context_sha256"]
          and manifest["candidate_universe_sha256"] == join["full_trained_row_inventory_sha256"],
          "independent labels refer to another source/model/candidate universe")
    cases = manifest["cases"]
    _need(type(cases) is list and 1 <= len(cases) <= 64, "bounded explicit development cases required")
    statements = {row["statement_id"] for row in match["intent_document"]["statements"]}
    by_id = {row["feature_row_id"]: row for row in inventory}
    seen = set()
    for case in cases:
        _fields(case, {"case_id", "statement_ids", "label_status", "relevant_units"}, "case")
        _text(case["case_id"], "case identity")
        _need(case["case_id"] not in seen, "duplicate development case identity")
        seen.add(case["case_id"])
        ids = case["statement_ids"]
        _need(type(ids) is list and 1 <= len(ids) <= len(statements)
              and all(type(item) is str and item in statements for item in ids)
              and len(set(ids)) == len(ids), "unique original native statement bindings required")
        relevant = case["relevant_units"]
        _need(type(relevant) is list and len(relevant) <= len(inventory), "bounded positive source units required")
        _need(case["label_status"] in ("positive_units", "unjudged")
              and bool(relevant) == (case["label_status"] == "positive_units"),
              "unjudged cases have no semantic negative labels")
        identities = set()
        for unit in relevant:
            _need(type(unit) is dict and type(unit.get("feature_row_id")) is str,
                  "exact positive feature unit required")
            identity = unit["feature_row_id"]
            _need(identity in by_id and _digest(unit) == _digest(by_id[identity]),
                  "positive label source/AST/feature binding differs")
            _need(identity not in identities, "duplicate positive source label")
            identities.add(identity)
    return cases


def _lexical(join, inventory):
    """Keep original prefix positions, including non-function symbol hits."""
    lexical = join["fixed_candidates"]["lexical"]
    _need(type(lexical) is dict and lexical.get("schema") == "supervisor-code-retrieval-context@1"
          and lexical.get("status") == "current" and lexical.get("stale_paths") == []
          and lexical.get("query_text") == join["original_native_match"]["intent_source"]["text"],
          "original current native lexical prefix required")
    sources = {row["path"]: row for row in join["source_records"]}
    ledger = lexical.get("source_sha256")
    _need(type(ledger) is dict and bool(ledger) and all(
        path in sources and sources[path]["source_sha256"] == digest
        for path, digest in ledger.items()), "lexical source ledger differs from original sources")
    hits = lexical.get("hits")
    _need(type(hits) is list and len(hits) <= 1024, "bounded retained lexical prefix required")
    by_unit = {(row["path"], row["symbol"], row["line"]): row for row in inventory}
    result, seen, mapped = [], set(), set()
    previous_score = float("inf")
    for position, hit in enumerate(hits, 1):
        _need(type(hit) is dict and type(hit.get("row_id")) is str,
              "native lexical row identity required")
        _need(hit["row_id"] not in seen, "duplicate lexical row identity")
        seen.add(hit["row_id"])
        path = hit.get("path")
        _need(type(path) is str and path in ledger and type(hit.get("symbol")) is str
              and type(hit.get("line_start")) is int and type(hit.get("line_end")) is int
              and hit["line_start"] >= 1 and hit["line_end"] >= hit["line_start"]
              and type(hit.get("rank")) is int and hit["rank"] == position,
              "ordered original lexical source positions required")
        score = hit.get("score")
        _need(type(score) in (float, int) and score <= previous_score,
              "finite descending lexical scores required")  # _json rejects nonfinite numbers.
        previous_score = score
        module_parts = list(PurePosixPath(path.removesuffix(".py")).parts)
        if module_parts and module_parts[-1] == "__init__":
            module_parts.pop()
        module = ".".join(module_parts)
        prefix = module + "." if module else ""
        _need(hit["symbol"].startswith(prefix), "lexical symbol has another module prefix")
        symbol = hit["symbol"][len(prefix):]
        unit = by_unit.get((path, symbol, hit["line_start"]))
        if unit is not None:
            _need(unit["end_line"] == hit["line_end"], "lexical hit source span differs from feature unit")
            _need(unit["feature_row_id"] not in mapped, "duplicate lexical feature association")
            mapped.add(unit["feature_row_id"])
        result.append({"position": position, "lexical_row_id": hit["row_id"],
            "feature_row_id": None if unit is None else unit["feature_row_id"],
            "path": path, "symbol": hit["symbol"], "line": hit["line_start"],
            "end_line": hit["line_end"], "score": score,
            "disposition": "outside_function_feature_universe" if unit is None else "exact_function_unit"})
    return {"status": "retained_prefix_only", "rows": result,
        "ranking_complete": False, "ranking_count": len(result),
        "score_policy": "original_full_public_query_lexical_prefix_no_new_search",
        "producer_scores_authenticated_here": False}


def _metrics(ranking, relevant, *, judged):
    rows = ranking["rows"]
    available = ranking["status"] != "no_ranking" and judged
    positives = set(relevant)
    positions = [row["position"] for row in rows if row["feature_row_id"] in positives]
    first = min(positions) if positions else None
    complete = ranking["ranking_complete"]
    reciprocal = (1.0 / first if first is not None else 0.0 if complete else None) if available else None
    recall = {}
    for cutoff in _CUTOFFS:
        count = sum(position <= cutoff for position in positions) if available else None
        measured = available and (complete or len(rows) >= cutoff)
        recall[str(cutoff)] = {"value": count / len(positives) if measured else None,
            "relevant_found": count, "positive_label_count": len(positives) if judged else None,
            "cutoff_covered": bool(measured)}
    return {"status": "unjudged" if not judged else ranking["status"],
        "ranking_count": len(rows), "ranking_complete": complete,
        "first_relevant_rank": first if available else None,
        "reciprocal_first_rank": reciprocal,
        "reciprocal_first_rank_bounds": None if not available else
            [reciprocal, reciprocal] if reciprocal is not None else [0.0, 1.0 / (len(rows) + 1)],
        "recall_at": recall}


def evaluate_terminal_codebase_intent_relevance(*, expected_join, original_arguments, label_manifest):
    """Score complete retained orders against independently supplied positive labels."""
    values = _json({"join": expected_join, "arguments": original_arguments, "labels": label_manifest})
    arguments, labels = values["arguments"], values["labels"]
    join = validate_terminal_intent_training_join(values["join"], **arguments)
    inventory = _inventory(arguments)
    cases = _labels(labels, join, inventory)
    methods = {}
    for outcome in join["model_control_outcomes"]:
        control = outcome["control"]
        ranks = control["ranking"]
        if control["name"] == "zero_heads":
            _need(ranks == sorted(ranks, key=lambda row: (-row["reconstruction_error"], row["row_id"])),
                  "zero diagnostic ordering differs from native full ranking")
        rows = [{"position": position, "feature_row_id": row["row_id"]}
                for position, row in enumerate(ranks, 1)]
        methods[control["name"]] = {"status": "no_ranking" if control["name"] == "model_off" else "complete_retained_order",
            "rows": rows, "ranking_complete": control["name"] != "model_off",
            "ranking_count": len(rows), "score_policy": control["inference_policy"],
            "producer_scores_authenticated_here": False}
    methods["lexical"] = _lexical(join, inventory)
    scored = []
    for case in cases:
        relevant = [row["feature_row_id"] for row in case["relevant_units"]]
        scored.append({**case, "metrics": {name: _metrics(ranking, relevant,
            judged=case["label_status"] == "positive_units") for name, ranking in methods.items()}})
    result = {"schema": SCHEMA, "label_policy": _LABEL_POLICY,
        "label_manifest_sha256": _digest(labels), "join_sha256": join["join_sha256"],
        "source_context": join["source_context"], "artifact_roots": join["artifact_roots"],
        "candidate_universe_sha256": join["full_trained_row_inventory_sha256"],
        "candidate_universe_count": len(inventory), "candidate_universe": inventory,
        "authored_focus_pool_count": len(join["source_unit_nominations"]),
        "ranking_methods": methods, "cases": scored,
        "fixed_candidates_sha256": _digest(join["fixed_candidates"]),
        "unranked_kg_count": len(join["fixed_candidates"]["kg"]),
        "original_native_match": join["original_native_match"],
        "residual_requirements": join["residual_requirements"],
        "recall_scope": "explicit_authored_positive_units_not_exhaustive_semantic_relevance",
        "case_independence_established": False, "blind_holdout_evaluated": False,
        "labels_used_in_fit_verified_here": False,
        "rankings_conditioned_on_case_labels": False,
        "live_checkout_or_native_storage_verified_here": False,
        "trained_inference_replayed_here": False, "ranking_gain_generalizes": False,
        "training_inference_search_SQL_checker_or_worker_operations_here": 0,
        **_AUTHORITY}
    result["evaluation_sha256"] = "sha256:" + _digest(result)
    return _json(result)


def validate_terminal_codebase_intent_relevance(receipt, **original_inputs):
    rebuilt = evaluate_terminal_codebase_intent_relevance(**original_inputs)
    _need(_json(receipt) == rebuilt, "entire relevance receipt differs from original-input replay")
    return rebuilt


__all__ = ["SCHEMA", "LABEL_SCHEMA", "RelevanceEvaluationError",
    "evaluate_terminal_codebase_intent_relevance", "validate_terminal_codebase_intent_relevance"]
