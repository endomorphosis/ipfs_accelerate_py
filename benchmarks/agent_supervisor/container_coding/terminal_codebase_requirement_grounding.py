"""Reviewed batching requirement candidates and a conditional finite witness.

The original prompt, navigation document and unresolved requirements survive.
Native Intent projections declare reviewed atoms; they do not prove meaning or
program behavior. The generated Lean sources are inert until independently
checked. No source execution, ranking, training, SQL or checker is performed.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re

SCHEMA = "terminal-codebase-batching-requirement-grounding@1"
PUBLIC_SOURCE_SHA256 = "547d230d5e6d93197803480cb81cab0ec6ac63fcfa0a24b246f549523f216c4e"
PUBLIC_INSTRUCTION_SHA256 = "0817374767533fbf51bb93345ee1bba266263c216c6cc55c096e71b4a1d37dfc"
PUBLIC_SOURCE_PATH = "environment/task_file/scripts/baseline_packer.py"
PUBLIC_INSTRUCTION_URI = "terminal-bench-instruction://llm-inference-batching-scheduler/instruction.md"
ALLOWED_QUERIES = ("batching-build-plan", "batching-representative")
MAX_RECEIPT_BYTES = 32 * 1024 * 1024
AUTHORITY = {key: False for key in (
    "semantic_alignment_verified", "source_semantics_verified", "source_runtime_equivalence_verified",
    "whole_program_proved", "complete_prompt_interpretation_qualified", "behavioral_satisfaction",
    "proof_authority", "formalization_authority", "execution_authority", "mutation_authority",
    "omission_authority", "completion_authority", "generalized_ranking_gain_qualified",
    "asymptotic_optimizer_convergence_proved")}
_BULLETS = (
    "  * All input requests are included exactly once (no missing/duplicate request_ids)",
    "  * Each batch uses shape (seq_align, heads_align=32, hidden_align=4096) where seq_align >= ceil(prompt_len/64)*64. I.e., seq_align is a multiple of 64.",
    "  * Max 8 unique shapes (seq_align, heads_align, hidden_align) across both buckets (MAX_SHAPES=8)",
    "  * One record per request_id, identical shapes within each batch_id",
)
# These are authored review choices, not model output or automatic NL parsing.
_ATOMS = (
    (0, "included_exactly_once", "Include every input request exactly once.", ("all_input_requests", "all_output_records")),
    (1, "heads_align_equals", "Every batch shape has heads_align equal to 32.", ("every_batch_shape", "32")),
    (1, "hidden_align_equals", "Every batch shape has hidden_align equal to 4096.", ("every_batch_shape", "4096")),
    (1, "seq_align_covers_prompt", "Every request has seq_align at least ceil(prompt_len/64)*64.", ("every_request", "seq_align", "prompt_len", "64")),
    (1, "seq_align_multiple_of", "Every batch shape has seq_align divisible by 64.", ("every_batch_shape", "64")),
    (2, "global_unique_shapes_at_most", "Across both buckets at most eight distinct shape triples are used.", ("both_buckets", "seq_align_heads_align_hidden_align", "8")),
    (3, "one_record_per_request", "Emit one output record per request_id.", ("all_request_ids", "all_output_records")),
    (3, "identical_shapes_within_batch", "Within each batch_id all output shapes are identical.", ("every_batch_id", "seq_align_heads_align_hidden_align")),
)


class RequirementGroundingError(ValueError):
    """A reviewed requirement receipt differs from its independent originals."""


def _need(condition, reason):
    if condition is not True:
        raise RequirementGroundingError(reason)


def _wire(value, *, native_report=False):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=native_report, allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError) as error:
        raise RequirementGroundingError("bounded plain JSON grounding required") from error
    _need(len(raw) <= MAX_RECEIPT_BYTES, "complete grounding exceeds retained-byte bound")
    return raw


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value, *, native_report=False):
    return _sha(_wire(value, native_report=native_report))


def _reviewed_intent(query, review_ref):
    from ipfs_datasets_py.logic.intent_ir.schema import (
        IntentIRDocument, IntentKind, IntentModality, IntentStatement, NodeGrounding,
        ReviewStatus, SourceRef, SourceSpan, StatementKind, validate_intent_ir)

    text = query["instruction_text"]
    sources, units = [], []
    for index, bullet in enumerate(_BULLETS):
        _need(text.count(bullet) == 1, "exact unique original batching bullet required")
        left, right = text.index(bullet), text.index(bullet) + len(bullet)
        if text[right:right + 1] == "\n":
            right += 1
        unit_text = text[left:right]
        unit_id = "batching-bullet-" + str(index + 1)
        sources.append(SourceRef(ref_id=unit_id, source_uri=query["instruction_uri"],
            source_id=PUBLIC_INSTRUCTION_SHA256, source_revision=PUBLIC_INSTRUCTION_SHA256,
            content_sha256=PUBLIC_INSTRUCTION_SHA256, review_status=ReviewStatus.MACHINE_EXTRACTED,
            span=SourceSpan(left, right)))
        units.append({"unit_id": unit_id, "start_char": left, "end_char": right,
            "start_byte": len(text[:left].encode()), "end_byte": len(text[:right].encode()),
            "text": unit_text, "sha256": _sha(unit_text.encode()),
            "disposition": "interpreted_candidate", "reason": "explicit_reviewed_atomic_decomposition"})
    statements = []
    for index, (bullet, predicate, normalized, arguments) in enumerate(_ATOMS):
        statements.append(IntentStatement(statement_id=f"batching-requirement-{index + 1:02d}",
            kind=StatementKind.GOAL, modality=IntentModality.REQUIRED, normalized_text=normalized,
            predicate=predicate, arguments=arguments, source_ref_ids=(sources[bullet].ref_id,),
            confidence=0.0, grounding=NodeGrounding.INFERRED,
            review_status=ReviewStatus.MACHINE_EXTRACTED))
    document = IntentIRDocument(document_id="batching-reviewed-requirements:" + query["query_id"],
        title="Explicit reviewed batching constraint candidates", intent_kind=IntentKind.DECLARATIVE,
        sources=tuple(sources), statements=tuple(statements), tags=("reviewed_candidate",))
    validate_intent_ir(document)
    candidates = []
    for index, source in enumerate(sources):
        local = IntentIRDocument(document_id=document.document_id + ":bullet-" + str(index + 1),
            title="Reviewed original batching constraint bullet", intent_kind=IntentKind.DECLARATIVE,
            sources=(source,), statements=tuple(s for s in statements if source.ref_id in s.source_ref_ids))
        validate_intent_ir(local)
        candidates.append({"unit_id": source.ref_id, "candidate_intent_ir": local.to_dict()})
    report = {"schema": "intent-reviewed-source-report@1", "source_sha256": PUBLIC_INSTRUCTION_SHA256,
        "source_bytes": len(text.encode()), "source_characters": len(text),
        "producer": {"name": SCHEMA, "revision": review_ref}, "interpretation_status": "reviewed_candidate",
        "units": units, "candidates": candidates, "proof_authority": False,
        "execution_authority": False, "completion_authority": False, "source_semantics_verified": False}
    # The native reviewed-report codec uses ensure_ascii=True, independently of
    # this receipt's UTF-8 canonical wire format.
    report["report_sha256"] = _digest(report, native_report=True)
    return document, report


def _source_context(corpus, query):
    from ipfs_datasets_py.logic.security_ir.code_program_derivation import describe_code_program_derivation_profile

    records = [row for row in corpus["sources"] if row["codebase_id"] == "batching"]
    _need(len(records) == 1, "exact single admitted batching source required")
    source = records[0]
    raw = source["source_text"].encode()
    _need(source["path"] == PUBLIC_SOURCE_PATH and source["source_sha256"] == PUBLIC_SOURCE_SHA256
          and len(raw) == 4380 and _sha(raw) == PUBLIC_SOURCE_SHA256,
          "exact public baseline source profile required")
    bank = corpus["candidate_banks"]["batching"]
    _need(len(bank) == 6, "complete six-function native batching bank required")
    selected_names = ("_plan_for_requests", "_plan_for_requests.assign_rep", "build_plan")
    selected = []
    for symbol in selected_names:
        matches = [row for row in bank if row["symbol"] == symbol]
        _need(len(matches) == 1, "unique original batching source unit required")
        selected.append(matches[0])
    tree = ast.parse(source["source_text"], type_comments=True)
    planner = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_plan_for_requests")
    build = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_plan")
    calls = [node for node in ast.walk(build) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "_plan_for_requests"]
    arguments = [[ast.unparse(arg) for arg in call.args] for call in calls]
    _need(arguments == [["reqs1", "GRAN", "MAX_SHAPES"], ["reqs2", "GRAN", "MAX_SHAPES"]],
          "original independent bucket helper call syntax differs")
    _need(any(isinstance(node, ast.FunctionDef) and node.name == "assign_rep" for node in planner.body),
          "original nested representative helper absent")
    return {"schema": "terminal-batching-reviewed-source-context@1", "source_record": source,
        "complete_candidate_bank": bank, "selected_source_units": selected,
        "syntactic_observations": {"independent_bucket_planner_call_arguments": arguments,
            "representative_helper_is_nested": True, "representatives_recomputed_per_call": True},
        "native_generic_program_profile": describe_code_program_derivation_profile(),
        "generic_program_derivation_status": "unsupported_fragment_not_invoked",
        "generic_program_frontiers": ["annotated_signatures", "loops", "closure_list_and_indexing",
            "calls_and_unverified_imported_bindings"], "instruction_sha256": query["instruction_sha256"],
        "facts_are_syntactic_only": True, **AUTHORITY}


def _finite_lean(buckets, *, false_control=False):
    def shape_list(values):
        return "[" + ", ".join(f"({value}, 32, 4096)" for value in values) + "]"
    lines = ["import Std", "set_option autoImplicit false", "namespace BatchingFiniteShapeWitness",
        "-- Authored conditional finite shape model; no Python or imported binding semantics proved.",
        "abbrev Shape := Nat × Nat × Nat",
        "noncomputable def bucketOne : List Shape := " + shape_list(buckets[0]),
        "noncomputable def bucketTwo : List Shape := " + shape_list(buckets[1]),
        "noncomputable def allShapes : List Shape := bucketOne ++ bucketTwo",
        "noncomputable def distinctShapes : List Shape := allShapes.eraseDups"]
    if false_control:
        lines += ["theorem incorrect_global_shape_cap : distinctShapes.length ≤ 8 := by decide"]
    else:
        lines += ["theorem conditional_finite_global_cap_counterexample :",
            "    bucketOne.length = 8 ∧ bucketTwo.length = 8 ∧",
            "    bucketOne.length ≤ 8 ∧ bucketTwo.length ≤ 8 ∧",
            "    allShapes.Nodup ∧ distinctShapes.length = 16 ∧",
            "    allShapes.all (fun shape => shape.1 % 64 == 0 &&",
            "      shape.2.1 == 32 && shape.2.2 == 4096) = true ∧",
            "    ¬ distinctShapes.length ≤ 8 := by decide +kernel"]
    return "\n".join([*lines, "end BatchingFiniteShapeWitness", ""])


def _finite_witness():
    buckets = [list(range(64, 513, 64)), list(range(576, 1025, 64))]
    shapes = [[value, 32, 4096] for bucket in buckets for value in bucket]
    _need(len({tuple(shape) for shape in shapes}) == 16, "complete distinct shape-triple witness required")
    rows = [{"bucket_id": f"model-bucket-{index + 1}", "aligned_sequences": values,
        "unique_sequence_values": values, "representatives": values,
        "representative_count": 8, "shape_triples": [[value, 32, 4096] for value in values],
        "request_ids": [f"model-b{index + 1}-r{item + 1:02d}" for item in range(8)],
        "no_reduction_branch": True} for index, values in enumerate(buckets)]
    assumptions = [
        "This is an authored finite conditional model, not observed runtime input or execution.",
        "Both bucket calls independently receive max_shapes=8 and granularity=64.",
        "Each bucket has eight unique aligned sequence values, hence the no-reduction branch reps=unique sequences.",
        "Request IDs are unique across both modeled buckets; each modeled prompt_len equals its aligned sequence value.",
        "The representative helper dispatch selects the same value from the sorted representatives for each modeled request.",
        "Imported cost_model.align, HEADS=32, HIDDEN=4096 and Python builtins are supplied modeling premises, unread and unverified here.",
        "Shape identity is exactly (seq_align, heads_align, hidden_align); no fourth component is introduced."]
    return {"schema": "terminal-batching-finite-conditional-shape-witness@1", "buckets": rows,
        "shape_components": ["seq_align", "heads_align", "hidden_align"], "all_shape_triples": shapes,
        "global_unique_shape_count": 16, "required_global_cap": 8,
        "model_global_cap_satisfied": False, "model_counterexample_present": True,
        "assumptions": assumptions, "actual_program_violation_established": False,
        "actual_benchmark_inputs_used": False, "dependency_semantics_verified": False,
        "checker_executed": False, "lean_validation_status": "not_run", **AUTHORITY}, {
            "finite_countermodel": _finite_lean(buckets),
            "false_global_cap_control": _finite_lean(buckets, false_control=True)}


def build_terminal_batching_requirement_grounding(*, corpus_receipt, original_inputs,
        expected_corpus_sha256, query_id, review_ref):
    """Reproduce eight explicitly reviewed candidates; never grant authority."""
    from .terminal_codebase_intent_corpus import validate_terminal_intent_relevance_corpus
    from ipfs_datasets_py.logic.intent_ir.formalize.requirements import build_intent_requirement_ledger
    from ipfs_datasets_py.logic.intent_ir.formalize.extended_projections import (
        project_intent_families, DEFAULT_FAMILIES, ADDITIONAL_REQUIREMENTS)
    from ipfs_datasets_py.logic.intent_ir.formalize.lean_projection import project_lean_family

    _need(type(expected_corpus_sha256) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", expected_corpus_sha256) is not None,
          "mandatory independent corpus pin required")
    _need(type(query_id) is str and query_id in ALLOWED_QUERIES, "explicit admitted batching query required")
    _need(type(review_ref) is str and 0 < len(review_ref) <= 1024 and bool(review_ref.strip())
          and not any(ord(char) < 32 for char in review_ref), "explicit bounded review reference required")
    _need(type(original_inputs) is dict, "independent original corpus inputs required")
    try:
        corpus = validate_terminal_intent_relevance_corpus(corpus_receipt, **original_inputs)
    except (ValueError, TypeError, RecursionError) as error:
        raise RequirementGroundingError("native corpus replay refused: " + str(error)) from error
    _need(corpus["corpus_sha256"] == expected_corpus_sha256, "independent corpus pin differs")
    matches = [row for row in corpus["queries"] if row["query_id"] == query_id]
    _need(len(matches) == 1, "unique admitted original query required")
    query = matches[0]
    _need(query["codebase_id"] == "batching" and query["instruction_uri"] == PUBLIC_INSTRUCTION_URI
          and query["instruction_sha256"] == PUBLIC_INSTRUCTION_SHA256
          and len(query["instruction_text"].encode()) == 4365
          and _sha(query["instruction_text"].encode()) == PUBLIC_INSTRUCTION_SHA256,
          "exact entire public batching instruction profile required")
    _need(len(query["residual_requirements"]) == 2 and
          all(row["status"] == "unknown" for row in query["residual_requirements"]),
          "original full query and unknown residuals must survive")
    context = _source_context(corpus, query)
    native, report = _reviewed_intent(query, review_ref)
    ledger = build_intent_requirement_ledger(query["instruction_text"], source_report=report,
        source_identity={"source_uri": query["instruction_uri"], "source_sha256": PUBLIC_INSTRUCTION_SHA256})
    _need(len(ledger["requirements"]) == 8, "eight native candidate requirements required")
    requested_families = [*DEFAULT_FAMILIES, *ADDITIONAL_REQUIREMENTS]
    projections = project_intent_families(native, requested_families=requested_families)
    reports = projections["projections"]
    families = {row["family_id"] for row in reports}
    _need(len(requested_families) == len(families) == 16 and families == set(requested_families),
          "complete native sixteen-family inventory required")
    _need(len(reports) == 17 and len({row["projection_id"] for row in reports}) == 17
          and {row["profile_id"] for row in reports if row["family_id"] == "transition_system"} == {None, "tla_plus"},
          "all seventeen native projection reports and both transition profiles required")
    lean = project_lean_family(native)
    _need(lean in projections["projections"], "native Intent Lean projection differs from aggregate replay")
    statements = {row["statement_id"]: row for row in native.to_dict()["statements"]}
    sources = {row["ref_id"]: row for row in native.to_dict()["sources"]}
    units = {row["unit_id"]: row for row in report["units"]}
    links = []
    for requirement in ledger["requirements"]:
        statement_id, = requirement["statement_ids"]
        statement = statements[statement_id]
        links.append({"requirement_id": requirement["requirement_id"], "statement_id": statement_id,
            "source_unit": units[requirement["source_unit_id"]],
            "original_source_ref": sources[statement["source_ref_ids"][0]],
            "reviewed_statement": statement, "native_candidate_document_sha256": requirement["native_document_sha256"],
            "review_ref": review_ref, "source_candidate_ids": [row["candidate_id"] for row in context["selected_source_units"]],
            "link_status": "reviewed_syntactic_candidate_not_satisfaction", "status": "unknown", **AUTHORITY})
    witness, lean_sources = _finite_witness()
    lean_sources["intent"] = lean["representation"]["source"]
    unknown = [{"source_unit_id": row["unit_id"], "start_char": row["start_char"], "end_char": row["end_char"],
        "sha256": row["sha256"], "native_disposition": row["disposition"], "status": "unknown",
        "reason": row["unsupported_reason"]} for row in ledger["source_units"]
        if row["disposition"] == "unsupported"]
    coverage = {"schema": "terminal-batching-grounding-coverage@1", "reviewed_bullets": 4,
        "reviewed_atomic_candidates": 8, "source_accounting_complete": ledger["source_accounting_complete"],
        "unreported_source_units": unknown, "performance_thresholds_and_file_immutability": "unknown_unreported",
        "full_original_query_preserved": True, "original_unknown_residual_count": 2,
        "native_family_count": len(families), "native_projection_count": len(reports),
        "native_family_statuses": {family: sorted({row["status"] for row in reports if row["family_id"] == family})
            for family in sorted(families)},
        "native_projection_statuses": [{key: row[key] for key in ("projection_id", "family_id", "profile_id", "status")}
            for row in reports],
        "all_family_views_are_candidate_declarations": True, "generic_list_program_fragment_supported": False,
        "contracts": "unchanged_unknown_empty", "complete_original_meaning_formalized": False,
        "lean_checker_calls": 0, "training_calls": 0, "inference_calls": 0, "SQL_calls": 0,
        "planning_handoff": "abstained", "canonical_tasks": [], "facts": [], "effects": [], **AUTHORITY}
    result = {"schema": SCHEMA, "status": "reviewed_candidates_and_unchecked_conditional_countermodel",
        "corpus_sha256": expected_corpus_sha256, "query_id": query_id, "review_ref": review_ref,
        "original_query": query, "reviewed_native_intent": native.to_dict(), "reviewed_source_report": report,
        "requirement_ledger": ledger, "requirement_links": sorted(links, key=lambda row: row["statement_id"]),
        "family_projection": projections, "source_context": context, "finite_witness": witness,
        "lean_sources": lean_sources, "lean_source_sha256": {key: _sha(value.encode()) for key, value in lean_sources.items()},
        "coverage": coverage, "planning_handoff": "abstained", "checker_calls": 0,
        "canonical_tasks": [], "facts": [], "effects": [], **AUTHORITY}
    result["grounding_sha256"] = _digest(result)
    return json.loads(_wire(result))


def validate_terminal_batching_requirement_grounding(receipt, **original_arguments):
    """Rebuild the full native receipt from independent originals, not its seal."""
    expected = build_terminal_batching_requirement_grounding(**original_arguments)
    _need(_wire(receipt) == _wire(expected), "grounding differs from exact source/native replay")
    return expected


def project_terminal_batching_grounding_metadata(receipt, **original_arguments):
    """Return additive families only; preserve all complete native projection rows."""
    value = validate_terminal_batching_requirement_grounding(receipt, **original_arguments)
    family = value["family_projection"]
    targets = family["native_targets"]
    records = {
        "batching_requirement_ledger": [value["requirement_ledger"]],
        "batching_native_intent": [{"reviewed_document": value["reviewed_native_intent"], "original_query": value["original_query"]}],
        "batching_requirement_links": value["requirement_links"],
        "batching_projection_reports": family["projections"],
        "batching_native_targets": [{"projection_index": index, "target": row} for index, row in enumerate(targets["projections"])],
        "batching_source_context": [value["source_context"]],
        "batching_finite_witness": [{"witness": value["finite_witness"], "lean_sources": value["lean_sources"],
            "lean_source_sha256": value["lean_source_sha256"]}],
        "batching_grounding_coverage": [{"coverage": value["coverage"], "grounding_sha256": value["grounding_sha256"],
            "family_projection_envelope": {key: item for key, item in family.items() if key not in ("projections", "native_targets")},
            "native_targets_envelope": {key: item for key, item in targets.items() if key != "projections"}}],
    }
    _need(all(len(_wire(row)) <= 262144 for rows in records.values() for row in rows),
          "complete delta row exceeds native metadata row bound; no truncation")
    return json.loads(_wire(records))


__all__ = ["RequirementGroundingError", "SCHEMA", "PUBLIC_SOURCE_SHA256", "PUBLIC_INSTRUCTION_SHA256",
    "build_terminal_batching_requirement_grounding", "validate_terminal_batching_requirement_grounding",
    "project_terminal_batching_grounding_metadata"]
