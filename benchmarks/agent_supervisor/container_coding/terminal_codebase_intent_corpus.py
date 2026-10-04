"""Pure, source-bound development corpus for native intent/code relevance.

Candidate banks contain every native Python function feature row. Authored
navigation judgments are separate from the original prompt and its unresolved
requirements. No fit, inference, file access, SQL, or proof is performed here.
Development task roles do not establish blind or historically unseen holdouts.
"""
from __future__ import annotations

import ast
from collections import defaultdict
import hashlib
import json
import math
from pathlib import PurePosixPath
import re

SCHEMA = "terminal-codebase-native-intent-relevance-corpus@1"
HISTORY_SCHEMA = "terminal-intent-corpus-history@1"
MAX_BYTES = 32 * 1024 * 1024
MAX_FUNCTIONS = 4096
ROLES = ("train", "validation", "test")
_HASH = re.compile(r"[0-9a-f]{64}")
_AUTHORITY = {name: False for name in (
    "semantic_alignment_verified", "source_semantics_verified", "whole_program_proved",
    "asymptotic_optimizer_convergence_proved", "proof_authority", "formalization_authority",
    "execution_authority", "mutation_authority", "omission_authority", "completion_authority",
    "behavioral_satisfaction")}


class IntentCorpusError(ValueError):
    """A corpus differs from its independent sources, labels, or split scope."""


def _need(condition, reason):
    if condition is not True:
        raise IntentCorpusError(reason)


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return _sha(_wire(value))


def _json(value):
    pending, count = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        count += 1
        _need(depth <= 48 and count <= 2_000_000, "bounded corpus JSON structure required")
        if type(item) is dict:
            _need(all(type(key) is str for key in item), "string corpus JSON keys required")
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is float:
            _need(math.isfinite(item), "finite corpus JSON values required")
        else:
            _need(type(item) in (str, int, bool, type(None)), "plain corpus JSON values required")
    raw = _wire(value)
    _need(len(raw) <= MAX_BYTES, "bounded complete corpus required; no truncation")
    return json.loads(raw)


def _fields(value, names, label):
    _need(type(value) is dict and set(value) == set(names), "exact " + label + " fields required")


def _text(value, label, maximum=1024):
    _need(type(value) is str and 0 < len(value) <= maximum and "\x00" not in value,
          "bounded nonempty " + label + " required")


def _hash(value, label):
    _need(type(value) is str and _HASH.fullmatch(value) is not None, "exact SHA256 " + label + " required")


def _refs(value, label):
    _need(type(value) is list and len(value) <= 128, "bounded " + label + " list required")
    for item in value:
        _text(item, label)
    _need(value == sorted(set(value)), "sorted unique " + label + " references required")


def _sources(records):
    _need(type(records) is list and 1 <= len(records) <= 32, "bounded exact source record list required")
    seen = set()
    for row in records:
        _fields(row, {"codebase_id", "path", "source_text", "source_sha256", "source_uri"}, "source record")
        _text(row["codebase_id"], "codebase identity", 256)
        _text(row["path"], "relative source path")
        path = PurePosixPath(row["path"])
        _need(not path.is_absolute() and str(path) == row["path"] and
              all(part not in (".", "..") for part in path.parts) and "\\" not in row["path"],
              "canonical repository-relative source path required")
        _text(row["source_uri"], "source provenance URI")
        _need(type(row["source_text"]) is str and 0 < len(row["source_text"].encode()) <= 4_000_000,
              "bounded original source text required")
        if row["path"].endswith(".py"):
            _need(not any(byte in row["source_text"].encode() for byte in (b"\r", b"\v", b"\f", b"\x00")),
                  "source line mapping refuses CR/VT/FF/NUL separators")
        _hash(row["source_sha256"], "source identity")
        _need(_sha(row["source_text"].encode()) == row["source_sha256"], "original source bytes differ")
        key = (row["codebase_id"], row["path"])
        _need(key not in seen, "duplicate codebase source path")
        seen.add(key)
    return sorted(records, key=lambda row: (row["codebase_id"], row["path"]))


def _banks(sources):
    from ipfs_datasets_py.logic.formalization.autoencoder.security.codebase_autoencoder import _features
    from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formula_corpus import (
        _qualified_functions, _verify_span)
    from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formalization_evaluation import _function_span
    from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formula_grammar import source_shape

    by_codebase = defaultdict(list)
    for row in sources:
        by_codebase[row["codebase_id"]].append(row)
    banks, frontiers, count = {}, [], 0
    for codebase, records in sorted(by_codebase.items()):
        raw_sources = {row["path"]: row["source_text"].encode() for row in records}
        for path, raw in raw_sources.items():
            if path.endswith(".py"):
                try:
                    tree = ast.parse(raw.decode("utf-8"))
                except (SyntaxError, UnicodeError, ValueError):
                    continue  # Native feature projection retains this frontier.
                except RecursionError as error:
                    raise IntentCorpusError("native source AST depth bound exceeded") from error
                _need(sum(1 for _ in ast.walk(tree)) <= 200_000,
                      "native source AST node bound exceeded; no partial inventory")
        try:
            features, unsupported = _features(raw_sources, sorted(raw_sources), MAX_FUNCTIONS)
        except ValueError as error:
            raise IntentCorpusError("native complete feature projection refused: " + str(error)) from error
        frontiers.extend({"codebase_id": codebase, **row} for row in unsupported)
        feature_units = {(row["path"], row["symbol"], row["line"]): row for row in features}
        _need(len(feature_units) == len(features), "ambiguous native feature association")
        candidates = []
        for record in records:
            path, raw = record["path"], raw_sources[record["path"]]
            if not path.endswith(".py"):
                frontiers.append({"codebase_id": codebase, "path": path, "reason": "non_python_source"})
                continue
            try:
                tree = ast.parse(raw.decode("utf-8"))
            except (SyntaxError, UnicodeError, ValueError):
                continue  # The native unsupported frontier is retained above.
            for node, symbol, _ in _qualified_functions(tree):
                feature = feature_units.pop((path, symbol, node.lineno), None)
                ast_sha = _sha(ast.dump(node, include_attributes=False).encode())
                _need(feature is not None and feature["ast_sha256"] == ast_sha,
                      "qualified function differs from exact native feature row")
                body, binding = _function_span(raw, node)
                _verify_span(raw, body, binding)
                try:
                    alpha_shape = source_shape(body)
                except (SyntaxError, UnicodeError, ValueError):
                    alpha_shape = None
                identity = {"codebase_id": codebase, "path": path, "symbol": symbol,
                    "line": node.lineno, "end_line": node.end_lineno,
                    "source_sha256": record["source_sha256"], "ast_sha256": ast_sha,
                    "source_binding": binding, "native_feature_row_id": feature["row_id"]}
                candidates.append({**identity, "candidate_id": "sha256:" + _digest(identity),
                    "source_uri": record["source_uri"], "features": feature["features"],
                    "native_feature_row": feature, "normalized_body": body.decode("utf-8"),
                    "normalized_body_sha256": _sha(body), "alpha_shape_sha256": alpha_shape,
                    "alpha_shape_status": "exact_native_normalized_AST" if alpha_shape is not None else "unsupported",
                    **_AUTHORITY})
        _need(not feature_units and len(candidates) == len(features), "complete feature bank lost a source function")
        count += len(candidates)
        _need(count <= MAX_FUNCTIONS, "complete corpus function bound exceeded; no truncation")
        banks[codebase] = sorted(candidates, key=lambda row: (row["path"], row["line"], row["symbol"]))
    return banks, sorted(frontiers, key=lambda row: (row["codebase_id"], row["path"], row["reason"]))


def _queries(records, banks, assignments):
    from ipfs_datasets_py.logic.intent_ir.decoder import decode_intent_ir
    from ipfs_datasets_py.logic.intent_ir.schema import validate_intent_ir
    from ipfs_datasets_py.logic.intent_ir.canonicalize import canonical_intent_ir_bytes

    _need(type(records) is list and 1 <= len(records) <= 64, "bounded native query list required")
    _need(type(assignments) is dict, "explicit development task-role mapping required")
    found, queries = set(), []
    for row in records:
        _fields(row, {"query_id", "codebase_id", "instruction_text", "instruction_sha256", "instruction_uri",
                      "intent_document", "navigation_text", "statement_ids"}, "query record")
        _text(row["query_id"], "query identity", 256)
        _need(row["query_id"] not in found, "duplicate query identity")
        found.add(row["query_id"])
        _text(row["codebase_id"], "query codebase identity", 256)
        _need(row["codebase_id"] in banks, "query codebase is absent")
        _text(row["instruction_text"], "original instruction", 128 * 1024)
        _text(row["instruction_uri"], "original instruction URI", 512)
        _text(row["navigation_text"], "authored navigation view", 8192)
        _hash(row["instruction_sha256"], "original instruction")
        _need(_sha(row["instruction_text"].encode()) == row["instruction_sha256"], "original instruction bytes differ")
        _need(row["query_id"] in assignments and assignments[row["query_id"]] in ROLES,
              "exact development train/validation/test role required")
        try:
            native = validate_intent_ir(decode_intent_ir(row["intent_document"]))
        except (ValueError, TypeError) as error:
            raise IntentCorpusError("native IntentIR validation refused: " + str(error)) from error
        _need(len(native.sources) == 2 and len(native.statements) <= 256 and
              len(native.actions) <= 256 and len(native.control_edges) <= 512,
              "bounded native two-source query document required")
        expected_sources = {
            row["instruction_uri"]: (row["instruction_text"], row["instruction_sha256"]),
            row["instruction_uri"] + "#navigation:" + row["query_id"]:
                (row["navigation_text"], _sha(row["navigation_text"].encode()))}
        native_sources = {source.source_uri: source for source in native.sources}
        _need(set(native_sources) == set(expected_sources), "native original/navigation source identities differ")
        for uri, (text, text_sha) in expected_sources.items():
            source = native_sources[uri]
            _need(source.content_sha256 == source.source_id == source.source_revision == text_sha and
                  source.span is not None and
                  source.span.start_char == 0 and source.span.end_char == len(text),
                  "native exact complete source span/hash differs")
        original_ref = native_sources[row["instruction_uri"]].ref_id
        navigation_ref = native_sources[row["instruction_uri"] + "#navigation:" + row["query_id"]].ref_id
        statements = {statement.statement_id: statement for statement in native.statements}
        selected = row["statement_ids"]
        _need(type(selected) is list and bool(selected) and
              all(type(identifier) is str and identifier in statements for identifier in selected) and
              selected == sorted(set(selected)),
              "unique exact native navigation statement IDs required")
        _need(any(statement.kind.value == "guard" and statement.grounding.value == "grounded" and
                  statement.normalized_text == row["instruction_text"] and
                  set(statement.source_ref_ids) == {original_ref} for statement in native.statements),
              "whole original opaque guard must remain grounded and retained")
        for identifier in selected:
            statement = statements[identifier]
            _need(statement.kind.value == "goal" and statement.normalized_text == row["navigation_text"] and
                  statement.grounding.value == "inferred" and statement.review_status.value == "machine_extracted" and
                  set(statement.source_ref_ids) == {navigation_ref}, "selected statement is not the separate authored navigation goal")
        document = native.to_dict()
        residuals = [{"requirement_id": statement["statement_id"], "statement": statement,
            "original_source_refs": [source for source in document["sources"] if source["ref_id"] in statement["source_ref_ids"]],
            "selected_for_navigation": statement["statement_id"] in selected, "status": "unknown",
            "reasons": ["authored_navigation_judgment_is_not_semantic_satisfaction"],
            "behavioral_satisfaction": False} for statement in document["statements"]]
        queries.append({**row, "intent_document": document, "task_role": assignments[row["query_id"]],
            "native_intent_sha256": _sha(canonical_intent_ir_bytes(native)),
            "navigation_sha256": _sha(row["navigation_text"].encode()),
            "residual_requirements": residuals, "original_prompt_residuals": residuals,
            "candidate_ids": [candidate["candidate_id"] for candidate in banks[row["codebase_id"]]],
            "navigation_is_authored_view": True, "original_instruction_interpreted_completely": False,
            **_AUTHORITY})
    _need(set(assignments) == found, "development role assignments must exactly cover original queries")
    return sorted(queries, key=lambda row: row["query_id"])


def _splits(queries, banks):
    from ipfs_datasets_py.logic.intent_ir.evaluation.splits import (
        IntentSplitExample, IntentSplitConfig, IntentSplitManifest, _hashed_shingles,
        require_leakage_safe_splits)

    config = IntentSplitConfig()
    examples, assignments = [], {}
    roles = defaultdict(set)
    for query in queries:
        roles[query["codebase_id"]].add(query["task_role"])
        for kind, text, text_sha in (("instruction", query["instruction_text"], query["instruction_sha256"]),
                                     ("navigation", query["navigation_text"], query["navigation_sha256"])):
            identifier = query["query_id"] + ":" + kind
            examples.append(IntentSplitExample(sample_id=identifier,
                repository_ids=(query["codebase_id"],), content_digests=(text_sha,),
                near_duplicate_signature=_hashed_shingles(text)))
            assignments[identifier] = query["task_role"]
    _need(all(len(values) == 1 for values in roles.values()), "one codebase/instruction family crosses development task roles")
    _need(set(roles) == set(banks), "every complete codebase bank requires an explicit query family")
    manifest = IntentSplitManifest(examples=tuple(examples), assignments=assignments,
        config_digest=config.digest, metadata={"near_duplicate_jaccard_threshold": config.near_duplicate_jaccard_threshold,
                                               "seed": config.seed})
    try:
        guard = require_leakage_safe_splits(manifest)
    except ValueError as error:
        raise IntentCorpusError("native query split guard refused: " + str(error)) from error
    same_role_duplicates, unsupported_shapes = [], []
    for field in ("source_sha256", "normalized_body_sha256", "alpha_shape_sha256"):
        groups = defaultdict(list)
        for codebase, candidates in banks.items():
            role, = roles[codebase]
            for candidate in candidates:
                if candidate[field] is not None:
                    groups[candidate[field]].append((role, candidate["candidate_id"]))
                elif field == "alpha_shape_sha256":
                    unsupported_shapes.append(candidate["candidate_id"])
        for identity, members in sorted(groups.items()):
            task_roles = sorted({role for role, _ in members})
            _need(len(task_roles) == 1,
                  "cross-role " + field + " collision " + identity + " in " + ",".join(task_roles))
            if len(members) > 1:
                same_role_duplicates.append({"kind": field, "sha256": identity,
                    "task_role": task_roles[0], "candidate_ids": sorted({identity for _, identity in members})})
    return {"manifest": manifest.to_dict(), "guard": guard.to_dict(),
        "same_role_duplicates": same_role_duplicates, "unsupported_alpha_shape_candidate_ids": sorted(unsupported_shapes),
        "near_duplicate_threshold": config.near_duplicate_jaccard_threshold,
        "scope": "declared_development_task_roles_not_blind_holdout",
        "historical_exposure_or_semantic_duplicate_detection_verified": False}


def _judgments(records, queries, banks):
    _need(type(records) is list and len(records) <= 4096, "bounded explicit reviewed judgments required")
    by_query = {query["query_id"]: query for query in queries}
    by_ref = {}
    for candidates in banks.values():
        for candidate in candidates:
            reference = {name: candidate[name] for name in ("codebase_id", "path", "symbol", "line", "source_sha256", "ast_sha256")}
            _need(_digest(reference) not in by_ref, "ambiguous complete candidate identity")
            by_ref[_digest(reference)] = candidate
    found, rows = set(), []
    for row in records:
        _fields(row, {"query_id", "candidate_ref", "label", "reason", "review_ref"}, "reviewed judgment")
        _fields(row["candidate_ref"], {"codebase_id", "path", "symbol", "line", "source_sha256", "ast_sha256"}, "reviewed candidate reference")
        _need(type(row["candidate_ref"]["line"]) is int and row["candidate_ref"]["line"] >= 1,
              "exact integer reviewed source line required")
        _need(row["query_id"] in by_query and _digest(row["candidate_ref"]) in by_ref,
              "reviewed source/AST binding differs from complete originals")
        candidate, query = by_ref[_digest(row["candidate_ref"])], by_query[row["query_id"]]
        _need(candidate["codebase_id"] == query["codebase_id"], "reviewed candidate is outside query codebase")
        _need(row["label"] in ("positive", "negative_navigation"), "unjudged is not an implicit negative")
        _text(row["reason"], "explicit reviewed navigation reason", 4096)
        _text(row["review_ref"], "review authorship reference")
        key = (row["query_id"], candidate["candidate_id"])
        _need(key not in found, "duplicate or conflicting reviewed judgment")
        found.add(key)
        rows.append({**row, "candidate_id": candidate["candidate_id"], "task_role": query["task_role"],
                     "judgment_scope": "authored_navigation_relevance_not_semantic_negative_truth", **_AUTHORITY})
    rows.sort(key=lambda row: (row["query_id"], row["candidate_id"]))
    pairs, summaries = [], []
    for query in queries:
        subset = [row for row in rows if row["query_id"] == query["query_id"]]
        positive = [row["candidate_id"] for row in subset if row["label"] == "positive"]
        negative = [row["candidate_id"] for row in subset if row["label"] == "negative_navigation"]
        for left in positive:
            for right in negative:
                pairs.append({"query_id": query["query_id"], "task_role": query["task_role"],
                              "positive_candidate_id": left, "negative_candidate_id": right})
        summaries.append({"query_id": query["query_id"], "task_role": query["task_role"],
                          "positive_count": len(positive), "negative_navigation_count": len(negative),
                          "unjudged_count": len(query["candidate_ids"]) - len(subset)})
    _need(len(pairs) <= 16384, "complete reviewed pair budget exceeded; no pair sampling")
    return rows, pairs, summaries


def _history(value, sources, queries):
    _fields(value, {"schema", "source_history", "query_history"}, "historical exposure")
    _need(value["schema"] == HISTORY_SCHEMA, "versioned historical exposure declarations required")
    expected_sources = {(row["codebase_id"], row["path"]): row["source_sha256"] for row in sources}
    expected_queries = {row["query_id"]: row["instruction_sha256"] for row in queries}
    seen_sources, seen_queries = set(), set()
    _need(type(value["source_history"]) is list and type(value["query_history"]) is list,
          "complete original source/query history lists required")
    for row in value["source_history"]:
        _fields(row, {"codebase_id", "path", "source_sha256", "prior_fit_refs", "prior_review_refs"}, "source history")
        key = (row["codebase_id"], row["path"])
        _need(key in expected_sources and key not in seen_sources and row["source_sha256"] == expected_sources[key],
              "historical source identity/population differs")
        seen_sources.add(key)
        for field in ("prior_fit_refs", "prior_review_refs"):
            _refs(row[field], "historical source " + field)
    for row in value["query_history"]:
        _fields(row, {"query_id", "instruction_sha256", "prior_fit_refs", "prior_review_refs"}, "query history")
        key = row["query_id"]
        _need(key in expected_queries and key not in seen_queries and row["instruction_sha256"] == expected_queries[key],
              "historical query identity/population differs")
        seen_queries.add(key)
        for field in ("prior_fit_refs", "prior_review_refs"):
            _refs(row[field], "historical query " + field)
    _need(seen_sources == set(expected_sources) and seen_queries == set(expected_queries),
          "historical declarations must cover every exact original source/query")
    return {"schema": HISTORY_SCHEMA,
            "source_history": sorted(value["source_history"], key=lambda row: (row["codebase_id"], row["path"])),
            "query_history": sorted(value["query_history"], key=lambda row: row["query_id"])}


def build_terminal_intent_relevance_corpus(*, source_records, query_records, reviewed_judgments,
                                         split_assignments, historical_exposure):
    """Derive complete banks and validate an independently authored corpus.

    A digest freezes content, not time or global model history. A separate
    owner must retain this corpus before initiating any subsequent training.
    """
    values = _json({"source_records": source_records, "query_records": query_records,
                    "reviewed_judgments": reviewed_judgments, "split_assignments": split_assignments,
                    "historical_exposure": historical_exposure})
    sources = _sources(values["source_records"])
    banks, frontiers = _banks(sources)
    queries = _queries(values["query_records"], banks, values["split_assignments"])
    split_guard = _splits(queries, banks)
    judgments, pairs, summaries = _judgments(values["reviewed_judgments"], queries, banks)
    history = _history(values["historical_exposure"], sources, queries)
    result = {"schema": SCHEMA, "sources": sources, "candidate_banks": banks, "queries": queries,
        "reviewed_judgments": judgments, "pairs": pairs, "judgment_summary": summaries,
        "split_assignments": dict(sorted(values["split_assignments"].items())),
        "historical_exposure": history, "native_split_guard": split_guard,
        "source_frontiers": frontiers, "complete_candidate_count": sum(map(len, banks.values())),
        "task_role_query_counts": {role: sum(query["task_role"] == role for query in queries) for role in ROLES},
        "original_inputs_sha256": _digest(values), "candidate_banks_sha256": _digest(banks),
        "candidate_scope": "all_native_Python_function_feature_rows_without_authored_focus_trimming",
        "label_policy": "explicit_reviewed_navigation_judgments_rest_unjudged",
        "partition_claim": "predeclared_development_task_families",
        "historical_exposure_scope": "caller_declared_known_lineage_not_global_never_exposed_attestation",
        "historical_exposure_independently_verified": False, "frozen_before_new_fit_verified_here": False,
        "blind_holdout_evaluated": False, "source_generalization_established": False,
        "ranking_gain_generalizes": False, "native_formula_or_plan_generated_here": False,
        "training_inference_SQL_filesystem_checker_or_worker_operations_here": 0, **_AUTHORITY}
    result["corpus_sha256"] = "sha256:" + _digest(result)
    return _json(result)


def validate_terminal_intent_relevance_corpus(receipt, **original_inputs):
    """Rebuild every row from independent originals, rather than a self-hash."""
    rebuilt = build_terminal_intent_relevance_corpus(**original_inputs)
    _need(_wire(_json(receipt)) == _wire(rebuilt), "whole corpus differs from independent original-input replay")
    return rebuilt


__all__ = ["SCHEMA", "HISTORY_SCHEMA", "IntentCorpusError", "build_terminal_intent_relevance_corpus",
           "validate_terminal_intent_relevance_corpus"]
