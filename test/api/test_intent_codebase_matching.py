"""Independent identity and authority checks for conditional-model nomination.

The key/receipt fixtures are authored controls, not executed proofs. The native
IntentIR decoder, compiler and both native cache key implementations are real;
even a completely consistent control must leave software behavior unresolved.
"""
from copy import deepcopy
import ast
import hashlib
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.planning import intent_codebase_matching as matching


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _index_sha(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode())


def _match_sha(value):
    return "sha256:" + _sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=True, allow_nan=False).encode())


AUTHORITY = {"proof_authority": False, "execution_authority": False,
    "mutation_authority": False, "completion_authority": False,
    "source_semantics_verified": False, "whole_program_proved": False,
    "asymptotic_optimizer_convergence_proved": False, "behavioral_satisfaction": False}


def _document(text="agent must repair bottle.", *, modality="required", extra_statement=False):
    from ipfs_datasets_py.logic.intent_ir.schema import (
        IntentIRDocument, IntentKind, IntentModality, IntentStatement,
        NodeGrounding, ReviewStatus, SourceRef, SourceSpan, StatementKind,
    )
    digest = _sha(text.encode())
    source = SourceRef(ref_id="authored-source", source_uri="authored:control.md",
        source_id=digest, source_revision=digest, content_sha256=digest,
        review_status=ReviewStatus.TRUSTED_FIXTURE, span=SourceSpan(0, len(text)))
    statement = IntentStatement(statement_id="repair-goal", kind=StatementKind.GOAL,
        modality=IntentModality(modality), normalized_text=text, source_ref_ids=(source.ref_id,),
        predicate="repair", arguments=("agent", "bottle"), confidence=0.0,
        grounding=NodeGrounding.GROUNDED, review_status=ReviewStatus.TRUSTED_FIXTURE)
    statements = (statement,)
    if extra_statement:
        statements += (IntentStatement(statement_id="extra-guard", kind=StatementKind.GUARD,
            modality=IntentModality.ASSERTED, normalized_text="The existing guards remain declared.",
            source_ref_ids=(source.ref_id,), predicate="preserve", arguments=("guard",),
            grounding=NodeGrounding.INFERRED),)
    document = IntentIRDocument(document_id="authored-control", title="Authored test control",
        intent_kind=IntentKind.DECLARATIVE, sources=(source,), statements=statements)
    document.validate()
    return document.to_dict(), {field: getattr(source, field) for field in
        ("ref_id", "source_uri", "source_id", "source_revision", "content_sha256")}


def _units(raw):
    rows = []
    lines = raw.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    for node in ast.parse(raw).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        rows.append({"symbol": node.name, "role": {"_hkey": "field_name", "_hval": "field_value"}[node.name],
            "line": node.lineno, "end_line": node.end_lineno,
            "source_span": {"start_byte": start, "end_byte": end, "sha256": _sha(raw[start:end])},
            "source_ast_sha256": _sha(ast.dump(node, include_attributes=False).encode()),
            "conversion_symbol": "touni", "guarded": False,
            "normalization_ops": ["replace_underscore_hyphen", "title"] if node.name == "_hkey" else ["identity"]})
    return rows


def _relationship(dimensions):
    # Cross-package public producer: the matcher independently recomputes both
    # native bodies, so these controls also check the agreed owner interface.
    from benchmarks.agent_supervisor.container_coding.terminal_codebase_proof_index import (
        build_terminal_codebase_proof_key_relationship,
    )
    return build_terminal_codebase_proof_key_relationship(dimensions=dimensions)


def _seal_row(row):
    row["key_relationship"] = _relationship(row["key_relationship"]["dimensions"])
    entry = {"schema": "terminal-codebase-proof-index-entry@1",
        "key_relationship": row["key_relationship"], "evidence": row["evidence"]}
    row["entry_id"] = "sha256:" + _index_sha(entry)
    return row


@pytest.fixture
def control():
    from ipfs_datasets_py.logic.security_ir import code_header_derivation as header

    raw = ("# μ: authored source binding control\n"
           "def _hkey(key):\n    return touni(key).replace('_', '-').title()\n\n"
           "def _hval(value):\n    return touni(value)\n").encode()
    units = _units(raw)
    source = {"source_path": "bottle.py", "source_sha256": _sha(raw), "source_bytes": len(raw),
        "checkpoint": {"weights_sha256": _sha(b"authored-checkpoint-control")},
        "annotation": "μ-control, not executed source"}
    environment = {"schema": "terminal-codebase-proof-environment-ref@1",
        "environment_sha256": _sha(b"independently-pinned-environment-control"),
        "lean": {"executable": "/fixture/lean", "version": "Lean fixture", "sha256": _sha(b"lean")},
        "z3": {"executable": "/fixture/z3", "version": "Z3 fixture", "sha256": _sha(b"z3")},
        "python": "CPython fixture", "packages": [{"name": name, "version": "fixture",
            "module": {"path": "/fixture/" + name + ".py", "sha256": _sha(name.encode()), "bytes": 10}}
            for name in ("duckdb", "torch", "numpy")]}
    translation = {"schema": "authored-translator-control@1", "implementation": _sha(b"translator")}
    snapshot = {"schema": "terminal-codebase-proof-source-snapshot@1", "source_path": "bottle.py",
        "source_sha256": _sha(raw), "source_bytes": len(raw), "source_unit_bindings": units,
        "source_context_sha256": _index_sha(source),
        "environment_sha256": environment["environment_sha256"],
        "environment_ref_sha256": _index_sha(environment), "translation_sha256": _index_sha(translation)}
    premises = [*header.describe_header_semantics_profile()["assumptions"],
        {"reviewed_protocol": {"callback_parameter": "start_response", "review_ref": "authored-reviewed-protocol"}}]
    rows = []
    for unit in units:
        for target in header._obligations(unit, _sha(raw)):
            compilation = target["compilation"]
            answer = target["expected_model_answer"]
            receipt = {"solver_answer": answer, "symbol": unit["symbol"], "kind": target["kind"],
                "matches_model_expectation": True, "compilation_id": compilation["compilation_id"],
                "script_sha256": _sha(compilation["script"]["source"].encode()),
                "script_digest": compilation["script"]["digest"],
                "query_mode": target["obligation"]["query_mode"], "solver_version": environment["z3"]["version"],
                "status": "satisfiable" if answer == "sat" else "unsatisfiable"}
            dimensions = {"source": source, "expression": compilation["script"]["source"],
                "formalization": compilation, "slice": [unit], "obligation": target["obligation"],
                "assumptions": premises, "bounds": {"timeout_ms": 5000}, "translation": translation,
                "provider": "native-z3-header-model-v1", "environment": environment,
                "policy": {"profile": "authored_model_only_control"}, "schema": {"control": "fixture@1"},
                "checker": "native-z3-software-verification-v1", "network_policy": {"network": "disabled"},
                "evidence_kind": "solver_result", "authority_ceiling": "bounded",
                "kernel": {"scope": "no_kernel_certificate"}, "theorem_registry": {"kind": target["kind"]}}
            evidence = {"schema": "terminal-codebase-conditional-model-evidence@1", "symbol": unit["symbol"],
                "property": target["kind"], "model_domain": "conditional_header_string_model",
                "classification": "conditional_model_sat_witness" if answer == "sat" else "conditional_model_unsat",
                "source_path": "bottle.py", "source_sha256": _sha(raw), "source_unit_bindings": [unit],
                "premises": premises, "open_frontiers": header.describe_header_semantics_profile()["open_frontiers"],
                "checker_receipt": receipt, **AUTHORITY}
            row = {"schema": "terminal-codebase-model-evidence-lookup@1", "status": "hit",
                "entry_id": "pending", "key_relationship": {"dimensions": dimensions}, "evidence": evidence,
                "expected_environment_sha256": environment["environment_sha256"], **AUTHORITY}
            rows.append(_seal_row(row))
    document, identity = _document()
    query = {"schema": matching.QUERY_SCHEMA, "review_ref": "authored-header-focus-control@1",
        "statement": {"statement_id": "repair-goal", "predicate": "repair", "arguments": ["agent", "bottle"]},
        "source_path": "bottle.py", "symbols": ["_hkey", "_hval"], "property": "header_delimiter_rejection",
        "polarity": "positive", "domain": matching.reviewed_header_matching_domain(),
        "semantic_alignment_verified": False}
    # Separate serialized owner records have no shared dictionaries. Preserve
    # that boundary so a changed row cannot also mutate the expected snapshot.
    return json.loads(json.dumps({"intent_document": document, "source_text": "agent must repair bottle.",
        "source_identity": identity, "query": query, "evidence_rows": rows,
        "current_source_snapshot": snapshot}))


def test_exact_native_atoms_and_conditional_hits_remain_residual(control):
    from ipfs_datasets_py.logic.intent_ir.canonicalize import intent_ir_sha256
    from ipfs_datasets_py.logic.intent_ir.decoder import decode_intent_ir

    result = matching.match_intent_codebase(**control)
    assert result["schema"] == "intent-codebase-match@1"
    assert result["status"] == "nominated_conditional_model_only"
    assert len(result["model_nominations"]) == 6
    assert sum(row["local_model_counterexample"] for row in result["model_nominations"]) == 4
    assert all(row["runtime_refutation"] is False for row in result["model_nominations"])
    assert result["software_behavior_status"] == "unresolved_software_behavior"
    assert result["current_source_snapshot"] == control["current_source_snapshot"]
    assert result["native_document_sha256"] == intent_ir_sha256(decode_intent_ir(control["intent_document"]))
    residual = result["residual_requirements"][0]
    assert residual["statement"]["predicate"] == "repair"
    assert residual["statement"]["arguments"] == ["agent", "bottle"]
    assert residual["original_source_refs"][0]["original_text"] == control["source_text"]
    assert residual["status"] == "unresolved_software_behavior"
    assert {"query_semantic_alignment_unproved", "request_domain_coverage_unproved",
        "conditional_model_source_semantics_unqualified"} <= set(residual["reasons"])
    for field in ("semantic_alignment_verified", "source_semantics_verified", "proof_authority",
                  "execution_authority", "completion_authority", "mutation_authority", "behavioral_satisfaction",
                  "domain_coverage_verified", "native_persistence_verified_here"):
        assert result[field] is False
    for field in ("current_behavioral_facts", "behavioral_satisfied_requirements", "runtime_refutations", "removed_task_ids"):
        assert result[field] == []


def test_missing_evidence_and_prohibition_do_not_imply_satisfaction(control):
    control["intent_document"], control["source_identity"] = _document(modality="prohibited")
    control["query"]["polarity"] = "prohibition"
    control["evidence_rows"] = []
    result = matching.match_intent_codebase(**control)
    assert result["status"] == "unknown"
    assert result["residual_requirements"][0]["statement"]["modality"] == "prohibited"
    assert "prohibition_requires_qualified_evidence_not_absence" in result["residual_requirements"][0]["reasons"]
    assert result["behavioral_satisfaction"] is False


def test_native_guard_and_unselected_statements_are_not_lost(control):
    control["intent_document"], control["source_identity"] = _document(extra_statement=True)
    result = matching.match_intent_codebase(**control)
    assert len(result["residual_requirements"]) == 2
    assert {row["statement"]["kind"] for row in result["residual_requirements"]} == {"goal", "guard"}
    guard = next(row for row in result["residual_requirements"] if row["statement"]["kind"] == "guard")
    assert guard["selected_for_authored_focus"] is False
    assert guard["statement"]["arguments"] == ["guard"]


@pytest.mark.parametrize("field,value,reason", [
    ("property", "arbitrary_python_behavior", "unsupported_property"),
    ("source_path", "other.py", "unsupported_source_path"),
    ("symbols", ["_hkey", "_hkey"], "ambiguous_symbol_selection"),
    ("symbols", ["_hkey", "dynamic_name"], "unsupported_symbol_selection"),
    ("symbols", [], "missing_symbol_selection"),
])
def test_unsupported_or_ambiguous_focus_is_unknown(control, field, value, reason):
    control["query"][field] = value
    result = matching.match_intent_codebase(**control)
    assert result["status"] == "unknown"
    assert result["model_nominations"] == []
    assert reason in result["residual_requirements"][0]["reasons"]


@pytest.mark.parametrize("field,value", [
    ("guards", []), ("input_types", [{"argument": "value", "type": "any"}]),
    ("quantifier", {"kind": "exists", "variables": ["value"], "domain": "integers"}),
    ("effects", [{"kind": "raises", "target": "invalid_header", "value": "TypeError"}]),
])
def test_uncovered_guard_type_quantifier_or_effect_domain_stays_unknown(control, field, value):
    control["query"]["domain"][field] = value
    result = matching.match_intent_codebase(**control)
    assert result["query"]["domain"][field] == value
    assert result["status"] == "unknown"
    assert "unsupported_or_ambiguous_request_domain" in result["residual_requirements"][0]["reasons"]
    assert result["domain_coverage_verified"] is False


@pytest.mark.parametrize("field", ["statement_id", "predicate", "arguments"])
def test_query_must_bind_exact_native_atom(control, field):
    control["query"]["statement"][field] = ["agent", "elsewhere"] if field == "arguments" else "different"
    with pytest.raises(matching.IntentCodebaseMatchingError, match="exact native statement"):
        matching.match_intent_codebase(**control)


@pytest.mark.parametrize("field", ["ref_id", "source_uri", "source_id", "source_revision", "content_sha256"])
def test_exact_original_source_reference_required(control, field):
    control["source_identity"][field] = _sha(b"different") if field == "content_sha256" else "different"
    with pytest.raises(matching.IntentCodebaseMatchingError, match="source"):
        matching.match_intent_codebase(**control)


def test_original_intent_source_mutation_fails(control):
    control["source_text"] += " changed"
    with pytest.raises(matching.IntentCodebaseMatchingError, match="source bytes differ"):
        matching.match_intent_codebase(**control)


@pytest.mark.parametrize("span", [None, {"start_char": 0, "end_char": 0}, {"start_char": 0, "end_char": 999}])
def test_missing_empty_or_out_of_bounds_native_source_span_fails(control, span):
    control["intent_document"]["sources"][0]["span"] = span
    with pytest.raises(matching.IntentCodebaseMatchingError, match="source span"):
        matching.match_intent_codebase(**control)


@pytest.mark.parametrize("field", ["source_sha256", "source_context_sha256", "environment_sha256", "environment_ref_sha256", "translation_sha256"])
def test_current_source_checkpoint_environment_or_translation_drift_rejected(control, field):
    control["current_source_snapshot"][field] = _sha(b"changed-current-root")
    with pytest.raises(matching.IntentCodebaseMatchingError, match="differs"):
        matching.match_intent_codebase(**control)


def test_resealed_old_checkpoint_context_is_still_stale(control):
    row = control["evidence_rows"][0]
    row["key_relationship"]["dimensions"]["source"]["checkpoint"]["weights_sha256"] = _sha(b"new-weights")
    _seal_row(row)
    with pytest.raises(matching.IntentCodebaseMatchingError, match="source/checkpoint context"):
        matching.match_intent_codebase(**control)


def test_resealed_compact_environment_pin_cannot_alias_original_inventory(control):
    row = control["evidence_rows"][0]
    row["key_relationship"]["dimensions"]["environment"]["lean"]["sha256"] = _sha(b"changed-lean")
    _seal_row(row)
    with pytest.raises(matching.IntentCodebaseMatchingError, match="environment or translation"):
        matching.match_intent_codebase(**control)


def test_changed_premises_remain_rejected_even_with_consistent_key_and_entry_hashes(control):
    row = control["evidence_rows"][0]
    row["evidence"]["premises"][0] = "unreviewed_wsgi_binding"
    row["key_relationship"]["dimensions"]["assumptions"] = deepcopy(row["evidence"]["premises"])
    _seal_row(row)
    with pytest.raises(matching.IntentCodebaseMatchingError, match="premises differ"):
        matching.match_intent_codebase(**control)


def test_resealed_changed_unit_binding_cannot_alias_current_source(control):
    row = control["evidence_rows"][0]
    row["evidence"]["source_unit_bindings"][0]["source_span"]["sha256"] = _sha(b"different-unit")
    row["key_relationship"]["dimensions"]["slice"] = deepcopy(row["evidence"]["source_unit_bindings"])
    _seal_row(row)
    with pytest.raises(matching.IntentCodebaseMatchingError, match="source unit binding"):
        matching.match_intent_codebase(**control)


@pytest.mark.parametrize("scope", ["query", "lookup", "evidence", "snapshot"])
def test_forged_eligible_or_current_fact_fields_are_rejected(control, scope):
    targets = {"query": control["query"], "lookup": control["evidence_rows"][0],
               "evidence": control["evidence_rows"][0]["evidence"], "snapshot": control["current_source_snapshot"]}
    targets[scope]["eligible"] = True
    with pytest.raises(matching.IntentCodebaseMatchingError, match="exact .* fields"):
        matching.match_intent_codebase(**control)


@pytest.mark.parametrize("scope,field", [("query", "semantic_alignment_verified"),
    ("lookup", "source_semantics_verified"), ("evidence", "behavioral_satisfaction"),
    ("evidence", "proof_authority")])
def test_caller_authority_escalation_fails(control, scope, field):
    targets = {"query": control["query"], "lookup": control["evidence_rows"][0],
               "evidence": control["evidence_rows"][0]["evidence"]}
    targets[scope][field] = True
    with pytest.raises(matching.IntentCodebaseMatchingError):
        matching.match_intent_codebase(**control)


@pytest.mark.parametrize("field", ["datasets_key_id", "accelerate_key_id", "relationship_id"])
def test_native_key_and_relationship_identity_tampering_fails(control, field):
    control["evidence_rows"][0]["key_relationship"][field] = "forged"
    with pytest.raises(matching.IntentCodebaseMatchingError, match="relationship identity"):
        matching.match_intent_codebase(**control)


def test_resealed_false_solver_receipt_not_bound_to_compilation_fails(control):
    row = control["evidence_rows"][0]
    row["evidence"]["checker_receipt"]["script_sha256"] = _sha(b"another-script")
    _seal_row(row)
    with pytest.raises(matching.IntentCodebaseMatchingError, match="solver receipt"):
        matching.match_intent_codebase(**control)


def _boolean_model_control(control):
    row = deepcopy(control["evidence_rows"][0])
    dims = row["key_relationship"]["dimensions"]
    expression = "theorem authored_control : True := True.intro\n"
    source_hash = _sha(expression.encode())
    compiled = [{"path": "/fixture/Control.olean", "sha256": _sha(b"authored-compiled-control"), "bytes": 32}]
    dims.update(expression=expression, slice=deepcopy(control["current_source_snapshot"]["source_unit_bindings"]),
        formalization={"file": "Control.lean", "source_sha256": source_hash,
            "compiled_artifacts": compiled, "theorem_ids": ["authored_control"]},
        obligation={"theorem_ids": ["authored_control"], "scope": "kernel_checked_generated_model_statement_only"},
        provider="native-lean-header-model-v1", checker="native-lean-kernel-v1",
        evidence_kind="kernel_checked_proof", kernel=deepcopy(dims["environment"]["lean"]))
    receipt = {"status": "passed", "returncode": 0, "backend_executed": True,
        "expected_success": True, "matches_expectation": True, "timed_out": False,
        "output_truncated": False, "workspace_limit_exceeded": False, "resource_exhausted": False,
        "executable_sha256": dims["environment"]["lean"]["sha256"], "file": "Control.lean",
        "artifact": {"path": "/fixture/Control.lean", "sha256": source_hash, "bytes": len(expression.encode())},
        "compiled_artifacts": compiled, "proof_scope": "kernel_checked_generated_model_statement_only"}
    row["evidence"].update(symbol="_hkey+_hval", property="header_boolean_model",
        model_domain="conditional_header_boolean_model", classification="conditional_model_kernel_checked",
        source_unit_bindings=deepcopy(dims["slice"]), checker_receipt=receipt)
    return _seal_row(row)


def test_consistent_kernel_receipt_control_cannot_establish_runtime_behavior(control):
    control["evidence_rows"] = [_boolean_model_control(control)]
    result = matching.match_intent_codebase(**control)
    assert result["status"] == "nominated_conditional_model_only"
    assert result["model_nominations"][0]["classification"] == "conditional_model_kernel_checked"
    assert result["model_nominations"][0]["local_model_counterexample"] is False
    assert result["behavioral_satisfaction"] is False
    assert result["current_behavioral_facts"] == []
    assert result["residual_requirements"][0]["status"] == "unresolved_software_behavior"


@pytest.mark.parametrize("field,value", [("returncode", True), ("backend_executed", False),
    ("timed_out", True), ("artifact", []), ("proof_scope", "whole_program"),
    ("compiled_artifacts", [])])
def test_resealed_malformed_or_unexecuted_kernel_receipt_is_rejected(control, field, value):
    row = _boolean_model_control(control)
    row["evidence"]["checker_receipt"][field] = value
    _seal_row(row)
    control["evidence_rows"] = [row]
    with pytest.raises(matching.IntentCodebaseMatchingError):
        matching.match_intent_codebase(**control)


@pytest.mark.parametrize("field,value", [("formalization", []), ("obligation", 4),
    ("environment", None), ("expression", 0)])
def test_malformed_raw_proof_dimensions_raise_contract_error(control, field, value):
    control["evidence_rows"][0]["key_relationship"]["dimensions"][field] = value
    with pytest.raises(matching.IntentCodebaseMatchingError):
        matching.match_intent_codebase(**control)


def test_arbitrary_to_dict_object_is_not_executed_as_native_intent(control):
    class ForgedIntent:
        called = False

        def to_dict(self):
            self.called = True
            return control["intent_document"]

    forged = ForgedIntent()
    control["intent_document"] = forged
    with pytest.raises(matching.IntentCodebaseMatchingError):
        matching.match_intent_codebase(**control)
    assert forged.called is False


def test_unknown_native_version_or_decoder_extra_field_fails(control):
    control["intent_document"]["schema_version"] = "intent-ir@unknown"
    with pytest.raises(matching.IntentCodebaseMatchingError, match="invalid native IntentIR"):
        matching.match_intent_codebase(**control)


def test_closed_exact_key_miss_is_unknown_not_a_proof_of_absence(control):
    row = control["evidence_rows"][0]
    miss = {key: deepcopy(value) for key, value in row.items() if key not in {"entry_id", "evidence"}}
    miss.update(status="miss", evidence=None, reason="exact_key_absent")
    control["evidence_rows"] = [miss]
    result = matching.match_intent_codebase(**control)
    assert result["status"] == "unknown"
    assert result["model_nominations"] == []
    assert result["evidence_rows"][0]["status"] == "miss"


def test_duplicate_keys_are_ambiguous_and_rejected(control):
    control["evidence_rows"] = [control["evidence_rows"][0]] * 2
    with pytest.raises(matching.IntentCodebaseMatchingError, match="duplicate or ambiguous"):
        matching.match_intent_codebase(**control)


def test_output_and_policy_are_detached_and_roots_cover_exact_frozen_inputs(control):
    pristine = deepcopy(control)
    result = matching.match_intent_codebase(**control)
    raw_id = result.pop("match_sha256")
    assert raw_id == _match_sha(result)
    for root_field, value in (("query_sha256", result["query"]),
        ("current_source_snapshot_sha256", result["current_source_snapshot"]),
        ("evidence_rows_sha256", result["evidence_rows"]), ("policy_sha256", result["policy"])):
        assert result["roots"][root_field] == _match_sha(value)
    result["query"]["domain"]["guards"].clear()
    result["policy"]["symbols"].clear()
    result["evidence_rows"][0]["evidence"]["premises"].clear()
    assert control == pristine
    assert matching.match_intent_codebase(**control)["match_sha256"] == raw_id
    domain = matching.reviewed_header_matching_domain()
    domain["guards"].clear()
    assert matching.reviewed_header_matching_domain()["guards"]


def test_evidence_order_is_canonical_not_caller_order(control):
    first = matching.match_intent_codebase(**control)
    control["evidence_rows"].reverse()
    assert matching.match_intent_codebase(**control) == first


def test_finite_json_and_explicit_evidence_bounds(control):
    control["query"]["domain"]["quantifier"]["domain"] = float("nan")
    with pytest.raises(matching.IntentCodebaseMatchingError, match="finite"):
        matching.match_intent_codebase(**control)
    control["query"]["domain"]["quantifier"]["domain"] = "ordinary_strings"
    control["evidence_rows"] = control["evidence_rows"] * 2
    with pytest.raises(matching.IntentCodebaseMatchingError, match="bounded native model evidence"):
        matching.match_intent_codebase(**control)
