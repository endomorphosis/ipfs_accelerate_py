"""Closed, reviewed lowering of the pinned batching representative helper.

This profile accounts for exactly one AST fragment. It does not execute the
baseline, widen generic ProgramIR, or qualify the enclosing planner. Generated
proof source is inert until a separately recorded native checker accepts it.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re

from . import terminal_codebase_requirement_grounding as grounding

SCHEMA = "terminal-codebase-selector-semantics@1"
PROFILE = "python-exact-int-immutable-list-first-ge-last@1"
_HELPER = """def assign_rep(s_val: int) -> int:
    for rep in reps:
        if rep >= s_val:
            return rep
    return reps[-1]
"""
_LOGIC = """import Init
set_option autoImplicit false
namespace BatchingSelector
-- None models the IndexError of indexing an empty representative list.
noncomputable def sourceLoop (s : Int) (fallback : Option Int) : List Int → Option Int
  | [] => fallback
  | r :: rs => if s ≤ r then some r else sourceLoop s fallback rs
noncomputable def sourceSemantics (rs : List Int) (s : Int) : Option Int :=
  sourceLoop s rs.getLast? rs
noncomputable def irSemantics (rs : List Int) (s : Int) : Option Int :=
  match rs.find? (fun r => decide (s ≤ r)) with
  | some r => some r
  | none => rs.getLast?
theorem loop_correspondence (rs : List Int) (s : Int) (fallback : Option Int) :
    sourceLoop s fallback rs =
      (match rs.find? (fun r => decide (s ≤ r)) with
       | some r => some r | none => fallback) := by
  induction rs with
  | nil => simp [sourceLoop]
  | cons r rs ih =>
    by_cases h : s ≤ r <;> simp [sourceLoop, List.find?, h, ih]
theorem source_to_ir (rs : List Int) (s : Int) :
    sourceSemantics rs s = irSemantics rs s := by
  exact loop_correspondence rs s rs.getLast?
theorem output_membership (rs : List Int) (s v : Int)
    (h : irSemantics rs s = some v) : v ∈ rs := by
  unfold irSemantics at h
  cases hf : rs.find? (fun r => decide (s ≤ r)) with
  | none => simp only [hf] at h; exact List.mem_of_getLast? h
  | some r =>
    simp only [hf, Option.some.injEq] at h
    subst v
    exact List.mem_of_find?_eq_some hf
theorem first_eligible (rs : List Int) (s v : Int)
    (h : rs.find? (fun r => decide (s ≤ r)) = some v) :
    s ≤ v ∧ ∃ before after, rs = before ++ v :: after ∧
      ∀ r ∈ before, r < s := by
  have result := List.find?_eq_some_iff_append.mp h
  simpa using result
theorem no_eligible_fallback (rs : List Int) (s : Int)
    (h : ∀ r ∈ rs, r < s) : irSemantics rs s = rs.getLast? := by
  have hf : rs.find? (fun r => decide (s ≤ r)) = none := by
    simp only [List.find?_eq_none, Bool.not_eq_true, decide_eq_false_iff_not]
    intro r hr
    exact Int.not_le.mpr (h r hr)
  simp [irSemantics, hf]
theorem empty_index_error (s : Int) : sourceSemantics [] s = none := by
  simp [sourceSemantics, sourceLoop]
-- Deliberately does not assert unconditional coverage: the fallback can be small.
theorem unconditional_coverage_counterexample : sourceSemantics [1] 2 = some 1 := by
  decide +kernel
end BatchingSelector
"""


class SelectorSemanticsError(ValueError):
    """The independent source or its closed reviewed profile does not match."""


def _wire(value):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError) as error:
        raise SelectorSemanticsError("bounded plain JSON required") from error
    if len(raw) > grounding.MAX_RECEIPT_BYTES:
        raise SelectorSemanticsError("selector receipt exceeds bound")
    return raw


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def build_terminal_batching_selector_semantics(*, grounding_receipt,
        grounding_arguments, expected_grounding_sha256, current_source_records):
    """Replay the requirement envelope and independently bind present source."""
    if (type(expected_grounding_sha256) is not str or
            re.fullmatch(r"[0-9a-f]{64}", expected_grounding_sha256) is None or
            type(grounding_arguments) is not dict):
        raise SelectorSemanticsError("independent grounding digest and arguments required")
    try:
        checked = grounding.validate_terminal_batching_requirement_grounding(
            grounding_receipt, **grounding_arguments)
    except (TypeError, ValueError, RecursionError) as error:
        raise SelectorSemanticsError("native grounding replay refused") from error
    if checked["grounding_sha256"] != expected_grounding_sha256:
        raise SelectorSemanticsError("independent grounding digest differs")
    if type(current_source_records) is not list or len(current_source_records) > 1024:
        raise SelectorSemanticsError("bounded independent current source records required")
    _wire(current_source_records)
    rows = [r for r in current_source_records if type(r) is dict and
            r.get("codebase_id") == "batching" and r.get("path") == grounding.PUBLIC_SOURCE_PATH]
    original = checked["source_context"]["source_record"]
    if len(rows) != 1 or _wire(rows[0]) != _wire(original):
        raise SelectorSemanticsError("present source differs from complete admitted original")
    source = rows[0]
    if _sha(source["source_text"].encode()) != grounding.PUBLIC_SOURCE_SHA256:
        raise SelectorSemanticsError("current source content pin differs")
    tree = ast.parse(source["source_text"], type_comments=True)
    parent = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_plan_for_requests")
    helpers = [n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == "assign_rep"]
    expected = ast.parse(_HELPER).body[0]
    if len(helpers) != 1 or ast.dump(helpers[0], include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise SelectorSemanticsError("closed helper AST lowering refused")
    helper = helpers[0]
    fragment = "\n".join(source["source_text"].splitlines()[helper.lineno-1:helper.end_lineno]) + "\n"
    result = {"schema": SCHEMA, "profile": PROFILE, "status": "exact_ast_lowering_unchecked_conditional_model",
        "grounding_sha256": expected_grounding_sha256,
        "source_binding": {"path": source["path"], "source_sha256": source["source_sha256"],
            "source_bytes": len(source["source_text"].encode()), "symbol": "_plan_for_requests.assign_rep",
            "start_line": helper.lineno, "end_line": helper.end_lineno,
            "fragment_sha256": _sha(fragment.encode()),
            "ast_sha256": _sha(ast.dump(helper, include_attributes=False).encode())},
        "ir": {"kind": "first_eligible_or_last", "comparison": "greater_than_or_equal",
            "iteration_order": "source_list_order", "fallback_index": -1,
            "empty_result": "IndexError", "input_domains": {"s_val": "exact_python_int",
                "reps": "immutable_snapshot_of_exact_python_int_list"}},
        "model_assumptions": ["Exact integers; bools, subclasses and custom comparison are outside the profile.",
            "Closure reps is a fixed list during this call; concurrent mutation is outside the profile.",
            "Lean Int and List Int model mathematical values and list order; CPython runtime semantics are not proved."],
        "closed_profile_ast_lowering_verified": True,
        "lean_source": _LOGIC, "lean_source_sha256": _sha(_LOGIC.encode()),
        "conditional_source_model_proof_status": "not_run",
        "caller_and_imported_bindings_status": "unproved", "generic_program_profile_widened": False,
        "full_task_satisfaction": "unknown", "model_convergence": "unproved",
        "training_calls": 0, "checker_calls": 0, "planning_handoff": "abstained",
        "canonical_tasks": [], "facts": [], "effects": [], **grounding.AUTHORITY}
    result["selector_sha256"] = _sha(_wire(result))
    return json.loads(_wire(result))


def validate_terminal_batching_selector_semantics(receipt, **arguments):
    expected = build_terminal_batching_selector_semantics(**arguments)
    if _wire(receipt) != _wire(expected):
        raise SelectorSemanticsError("selector receipt differs from independent replay")
    return expected


def evaluate_selector_model(representatives, sequence):
    """Executable IR reference under the explicitly closed integer/list domain."""
    if (type(sequence) is not int or type(representatives) is not list or
            len(representatives) > 10000 or any(type(r) is not int for r in representatives)):
        raise SelectorSemanticsError("bounded exact integer/list model domain required")
    snapshot = tuple(representatives)
    eligible = next((r for r in snapshot if r >= sequence), None)
    if eligible is not None:
        return eligible
    if not snapshot:
        raise IndexError("representative list is empty")
    return snapshot[-1]
