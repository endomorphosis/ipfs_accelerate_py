"""Experimental, source-bound nomination of conditional code models.

This is deliberately separate from the administrative planning admission
profile. An authored request focus is not the meaning of a native ``repair``
atom, and a checked conditional model is not a proof of Python behavior.
All requirements remain residual; this module cannot remove tasks or grant
proof, execution, completion, mutation, or semantic authority.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import PurePosixPath


SCHEMA = "intent-codebase-match@1"
QUERY_SCHEMA = "intent-codebase-query@1"
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_JSON_NODES = 1_000_000
MAX_EVIDENCE_ROWS = 8
_AUTHORITY = {
    "semantic_alignment_verified": False,
    "source_semantics_verified": False,
    "proof_authority": False,
    "execution_authority": False,
    "completion_authority": False,
    "mutation_authority": False,
}
_POLICY = {
    "schema": "intent-codebase-matching-policy@1",
    "profile": "reviewed_bottle_conditional_header_nomination_only",
    "source_path": "bottle.py",
    "symbols": ["_hkey", "_hval"],
    "property": "header_delimiter_rejection",
    "domain": {
        "guards": [{"predicate": "ordinary_string_after_successful_conversion",
                    "arguments": ["value"]}],
        "input_types": [{"argument": "value", "type": "str"}],
        "quantifier": {"kind": "forall", "variables": ["value"],
                       "domain": "ordinary_strings"},
        "effects": [{"kind": "raises", "target": "invalid_header", "value": "ValueError"}],
    },
    "behavioral_satisfaction": False,
    "current_facts_allowed": False,
    "task_omission_allowed": False,
    **_AUTHORITY,
}
_MODEL_AUTHORITY = {
    "proof_authority": False, "execution_authority": False,
    "mutation_authority": False, "completion_authority": False,
    "source_semantics_verified": False, "whole_program_proved": False,
    "asymptotic_optimizer_convergence_proved": False,
    "behavioral_satisfaction": False,
}
_PREMISES = [
    "reviewed_wsgi_callback_identity", "reviewed_unique_property_receiver_binding",
    "ordinary_unmodified_python_builtins_and_helper_bindings",
    "recognized_conversion_returns_an_ordinary_string",
    "model_starts_after_successful_conversion_and_ignores_conversion_exceptions",
    "normalization_terminates_and_preserves_NUL_LF_CR_membership",
]
_STRING_PROPERTIES = {
    "unsafe_converted_input_accepted", "safe_normalization_preserved",
    "forbidden_output_accepted",
}


class IntentCodebaseMatchingError(ValueError):
    """Malformed or inconsistent identity supplied to the bounded matcher."""


def _wire(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise IntentCodebaseMatchingError("finite bounded JSON required") from exc


def _sha(value):
    return "sha256:" + hashlib.sha256(_wire(value)).hexdigest()


def _index_digest(value):
    # The native experimental index intentionally uses UTF-8 JSON; native
    # IntentIR and this match artifact have their own declared canonical bytes.
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _json(value):
    pending, count = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if depth > 48 or count > MAX_JSON_NODES:
            raise IntentCodebaseMatchingError("matching JSON structure bound exceeded")
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise IntentCodebaseMatchingError("JSON object keys must be strings")
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is float:
            if not math.isfinite(item):
                raise IntentCodebaseMatchingError("finite matching JSON numbers required")
        elif type(item) not in (str, int, bool, type(None)):
            raise IntentCodebaseMatchingError("exact matching JSON types required")
    raw = _wire(value)
    if len(raw) > MAX_JSON_BYTES:
        raise IntentCodebaseMatchingError("matching JSON byte bound exceeded")
    return json.loads(raw)


def _object(value, fields, name):
    if type(value) is not dict or set(value) != set(fields):
        raise IntentCodebaseMatchingError(f"exact {name} fields required")
    return value


def _text(value, name, maximum=512):
    if (type(value) is not str or not value or value.strip() != value
            or len(value.encode("utf-8")) > maximum
            or any(char in value for char in "\r\n\0")):
        raise IntentCodebaseMatchingError(f"bounded {name} required")
    return value


def _array(value, name, maximum=64):
    if type(value) is not list or len(value) > maximum:
        raise IntentCodebaseMatchingError(f"bounded {name} array required")
    return value


def _strings(value, name):
    return [_text(item, name) for item in _array(value, name)]


def _digest(value, name):
    if (type(value) is not str or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise IntentCodebaseMatchingError(f"exact lowercase {name} SHA256 required")
    return value


def _query(value):
    value = _json(value)
    _object(value, {"schema", "review_ref", "statement", "source_path", "symbols",
                    "property", "polarity", "domain", "semantic_alignment_verified"}, "matching query")
    if value["schema"] != QUERY_SCHEMA or value["semantic_alignment_verified"] is not False:
        raise IntentCodebaseMatchingError("unqualified versioned reviewed query required")
    _text(value["review_ref"], "explicit focus review reference")
    atom = _object(value["statement"], {"statement_id", "predicate", "arguments"}, "native query atom")
    for field in ("statement_id", "predicate"):
        _text(atom[field], "query " + field)
    _strings(atom["arguments"], "query atom argument")
    path = _text(value["source_path"], "query source path", 4096)
    if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts or str(PurePosixPath(path)) != path:
        raise IntentCodebaseMatchingError("canonical source-relative query path required")
    _strings(value["symbols"], "query symbol")
    _text(value["property"], "query property")
    if value["polarity"] not in {"positive", "prohibition"}:
        raise IntentCodebaseMatchingError("explicit positive or prohibition query polarity required")
    domain = _object(value["domain"], {"guards", "input_types", "quantifier", "effects"}, "request domain")
    for guard in _array(domain["guards"], "request guards"):
        _object(guard, {"predicate", "arguments"}, "request guard")
        _text(guard["predicate"], "request guard predicate")
        _strings(guard["arguments"], "request guard argument")
    for item in _array(domain["input_types"], "request input types"):
        _object(item, {"argument", "type"}, "request input type")
        _text(item["argument"], "request typed argument")
        _text(item["type"], "request argument type")
    quantifier = _object(domain["quantifier"], {"kind", "variables", "domain"}, "request quantifier")
    _text(quantifier["kind"], "request quantifier kind")
    _strings(quantifier["variables"], "request quantifier variable")
    _text(quantifier["domain"], "request quantified domain")
    for item in _array(domain["effects"], "request effects"):
        _object(item, {"kind", "target", "value"}, "request effect")
        for field in ("kind", "target", "value"):
            _text(item[field], "request effect " + field)
    return value


def _native(document, source_text, source_identity):
    from ipfs_datasets_py.logic.intent_ir.canonicalize import canonical_intent_ir_bytes
    from ipfs_datasets_py.logic.intent_ir.decoder import decode_intent_ir
    from ipfs_datasets_py.logic.intent_ir.schema import IntentIRDocument, validate_intent_ir

    if type(source_text) is not str or not source_text or len(source_text.encode("utf-8")) > 128 * 1024:
        raise IntentCodebaseMatchingError("bounded exact original intent source text required")
    identity = _json(source_identity)
    _object(identity, {"ref_id", "source_uri", "source_id", "source_revision", "content_sha256"}, "intent source identity")
    for field in ("ref_id", "source_uri", "source_id", "source_revision"):
        _text(identity[field], "intent source " + field, 4096)
    _digest(identity["content_sha256"], "intent source")
    if hashlib.sha256(source_text.encode("utf-8")).hexdigest() != identity["content_sha256"]:
        raise IntentCodebaseMatchingError("original intent source bytes differ")
    try:
        # Re-decode dataclass inputs as well: no caller-owned typed object bypass.
        if type(document) is IntentIRDocument:
            document = document.to_dict()
        native = validate_intent_ir(decode_intent_ir(_json(document)))
        canonical = canonical_intent_ir_bytes(native)
    except (TypeError, ValueError) as exc:
        raise IntentCodebaseMatchingError("invalid native IntentIR: " + str(exc)) from exc
    if len(native.statements) > 256 or len(native.actions) > 256 or len(native.control_edges) > 512:
        raise IntentCodebaseMatchingError("bounded native IntentIR statement/action inventory required")
    if len(native.sources) != 1:
        raise IntentCodebaseMatchingError("this source-bound profile requires one exact native source reference")
    source = native.sources[0]
    if {field: getattr(source, field) for field in identity} != identity:
        raise IntentCodebaseMatchingError("native original source reference identity differs")
    if (source.span is None or source.span.start_char == source.span.end_char
            or source.span.end_char > len(source_text)):
        raise IntentCodebaseMatchingError("exact nonempty native original source span required")
    return native, "sha256:" + hashlib.sha256(canonical).hexdigest(), identity


def _profile_reasons(query):
    reasons = []
    if query["source_path"] != _POLICY["source_path"]:
        reasons.append("unsupported_source_path")
    symbols = query["symbols"]
    if not symbols:
        reasons.append("missing_symbol_selection")
    elif len(set(symbols)) != len(symbols):
        reasons.append("ambiguous_symbol_selection")
    elif not set(symbols) <= set(_POLICY["symbols"]):
        reasons.append("unsupported_symbol_selection")
    if query["property"] != _POLICY["property"]:
        reasons.append("unsupported_property")
    if query["domain"] != _POLICY["domain"]:
        reasons.append("unsupported_or_ambiguous_request_domain")
    return reasons


def _units(value, source_bytes):
    rows = _array(value, "native source unit bindings", 2)
    seen = set()
    for row in rows:
        _object(row, {"symbol", "role", "line", "end_line", "source_span",
                      "source_ast_sha256", "conversion_symbol", "normalization_ops", "guarded"},
                "native source unit binding")
        if row["symbol"] not in {"_hkey", "_hval"} or row["symbol"] in seen:
            raise IntentCodebaseMatchingError("unique native header source units required")
        seen.add(row["symbol"])
        if row["role"] != {"_hkey": "field_name", "_hval": "field_value"}[row["symbol"]]:
            raise IntentCodebaseMatchingError("native header unit role differs")
        if (type(row["line"]) is not int or type(row["end_line"]) is not int
                or not 1 <= row["line"] <= row["end_line"] <= source_bytes):
            raise IntentCodebaseMatchingError("bounded native header line range required")
        span = _object(row["source_span"], {"start_byte", "end_byte", "sha256"}, "native code source span")
        if (type(span["start_byte"]) is not int or type(span["end_byte"]) is not int
                or not 0 <= span["start_byte"] < span["end_byte"] <= source_bytes):
            raise IntentCodebaseMatchingError("bounded exact native code byte span required")
        _digest(span["sha256"], "code source unit")
        _digest(row["source_ast_sha256"], "code source AST")
        _text(row["conversion_symbol"], "native conversion symbol")
        _strings(row["normalization_ops"], "native normalization operation")
        if type(row["guarded"]) is not bool:
            raise IntentCodebaseMatchingError("exact native syntactic guard Boolean required")
    return rows


def _snapshot(value):
    value = _json(value)
    _object(value, {"schema", "source_path", "source_sha256", "source_bytes", "source_unit_bindings",
                    "source_context_sha256", "environment_sha256", "environment_ref_sha256", "translation_sha256"},
            "current proof-index source snapshot")
    if value["schema"] != "terminal-codebase-proof-source-snapshot@1" or value["source_path"] != "bottle.py":
        raise IntentCodebaseMatchingError("exact native proof-index source snapshot profile required")
    for field in ("source_sha256", "source_context_sha256", "environment_sha256", "environment_ref_sha256", "translation_sha256"):
        _digest(value[field], field)
    if type(value["source_bytes"]) is not int or not 0 < value["source_bytes"] <= 2_000_000:
        raise IntentCodebaseMatchingError("bounded complete current code source bytes required")
    _units(value["source_unit_bindings"], value["source_bytes"])
    return value


def _relationship(value):
    from ipfs_datasets_py.logic.common.canonical_cache_key import CanonicalProofCacheKey
    from ..proof.formal_verification_cache import ProofCacheKey

    _object(value, {"schema", "dimensions", "datasets_key", "datasets_key_id",
                    "accelerate_key", "accelerate_key_id", "relationship_id"}, "complete proof key relationship")
    if value["schema"] != "terminal-codebase-proof-key-relationship@1":
        raise IntentCodebaseMatchingError("exact proof key relationship version required")
    names = {"source", "expression", "formalization", "slice", "obligation", "assumptions",
             "bounds", "translation", "provider", "environment", "policy", "schema", "checker",
             "network_policy", "evidence_kind", "authority_ceiling", "kernel", "theorem_registry"}
    dims = _object(value["dimensions"], names, "complete raw proof key dimensions")
    for field in ("source", "formalization", "obligation", "bounds", "translation", "environment",
                  "policy", "schema", "network_policy", "kernel", "theorem_registry"):
        if type(dims[field]) is not dict or not dims[field]:
            raise IntentCodebaseMatchingError("nonempty native proof key object required: " + field)
    if (type(dims["expression"]) is not str or not dims["expression"]
            or len(dims["expression"].encode()) > 262144 or "\0" in dims["expression"]):
        raise IntentCodebaseMatchingError("bounded native model source expression required")
    _array(dims["slice"], "native modeled source slice", 2)
    _text(dims["provider"], "native proof provider")
    _text(dims["checker"], "native proof checker")
    _array(dims["assumptions"], "native model assumptions", 16)
    if dims["authority_ceiling"] != "bounded":
        raise IntentCodebaseMatchingError("conditional model key has bounded authority ceiling only")
    try:
        canonical = CanonicalProofCacheKey.build(**{key: dims[key] for key in names
            if key not in {"kernel", "theorem_registry"}})
        supervisor = ProofCacheKey(obligation=dims["obligation"],
            premises=tuple(dims["assumptions"]), translator=dims["translation"],
            solver={"provider": dims["provider"], "checker": dims["checker"],
                    "environment": dims["environment"]}, kernel=dims["kernel"],
            toolchain=dims["environment"], theorem_registry=dims["theorem_registry"],
            policy=dims["policy"], resource_budget=dims["bounds"],
            candidate_tree={"source": dims["source"], "slice": dims["slice"],
                "expression": dims["expression"], "formalization": dims["formalization"],
                "schema": dims["schema"], "network_policy": dims["network_policy"],
                "evidence_kind": dims["evidence_kind"], "authority_ceiling": dims["authority_ceiling"]})
    except (TypeError, ValueError) as exc:
        raise IntentCodebaseMatchingError("invalid native proof key dimensions: " + str(exc)) from exc
    rebuilt = {"schema": value["schema"], "dimensions": dims,
        "datasets_key": canonical.to_dict(), "datasets_key_id": canonical.key_id,
        "accelerate_key": supervisor.to_dict(), "accelerate_key_id": supervisor.key_id}
    rebuilt["relationship_id"] = "sha256:" + _index_digest(rebuilt)
    if rebuilt != value:
        raise IntentCodebaseMatchingError("complete native proof key relationship identity differs")
    return dims


def _evidence_row(value, snapshot):
    value = _json(value)
    if type(value) is not dict:
        raise IntentCodebaseMatchingError("exact native model lookup object required")
    status = value.get("status")
    _object(value, {"schema", "status", "key_relationship", "evidence", "expected_environment_sha256",
                    "entry_id" if status == "hit" else "reason", *_MODEL_AUTHORITY}, "native model lookup")
    if value["schema"] != "terminal-codebase-model-evidence-lookup@1" or status not in {"hit", "miss"}:
        raise IntentCodebaseMatchingError("exact native model lookup version and status required")
    if any(value[field] is not False for field in _MODEL_AUTHORITY):
        raise IntentCodebaseMatchingError("model lookup grants no behavioral or runtime authority")
    dims = _relationship(value["key_relationship"])
    source = dims["source"]
    if (type(source) is not dict or source.get("source_path") != snapshot["source_path"]
            or source.get("source_sha256") != snapshot["source_sha256"]
            or source.get("source_bytes") != snapshot["source_bytes"]
            or _index_digest(source) != snapshot["source_context_sha256"]):
        raise IntentCodebaseMatchingError("current complete source/checkpoint context differs from native evidence")
    environment = _object(dims["environment"], {"schema", "environment_sha256", "lean", "z3", "python", "packages"},
                          "native proof environment reference")
    for checker in ("lean", "z3"):
        descriptor = _object(environment[checker], {"executable", "version", "sha256"}, "native checker descriptor")
        _text(descriptor["executable"], "native checker executable", 4096)
        _text(descriptor["version"], "native checker version", 4096)
        _digest(descriptor["sha256"], "native checker")
    _text(environment["python"], "native Python version", 4096)
    packages = _array(environment["packages"], "native package pins", 3)
    if len(packages) != 3 or {item.get("name") for item in packages if type(item) is dict} != {"duckdb", "torch", "numpy"}:
        raise IntentCodebaseMatchingError("complete unique native package pins required")
    for package in packages:
        _object(package, {"name", "version", "module"}, "native package pin")
        _text(package["version"], "native package version")
        pin = _object(package["module"], {"path", "sha256", "bytes"}, "native package module pin")
        _text(pin["path"], "native package module path", 4096)
        _digest(pin["sha256"], "native package module")
        if type(pin["bytes"]) is not int or not 0 <= pin["bytes"] <= 16 * 1024 * 1024:
            raise IntentCodebaseMatchingError("bounded native package module byte count required")
    if (environment["schema"] != "terminal-codebase-proof-environment-ref@1"
            or environment["environment_sha256"] != snapshot["environment_sha256"]
            or _index_digest(environment) != snapshot["environment_ref_sha256"]
            or value["expected_environment_sha256"] != snapshot["environment_sha256"]
            or _index_digest(dims["translation"]) != snapshot["translation_sha256"]):
        raise IntentCodebaseMatchingError("current native environment or translation differs from evidence")
    if status == "miss":
        if value["evidence"] is not None or value["reason"] != "exact_key_absent":
            raise IntentCodebaseMatchingError("exact absent native key lookup required")
        return value
    evidence = _object(value["evidence"], {"schema", "symbol", "property", "model_domain", "classification",
        "source_path", "source_sha256", "source_unit_bindings", "premises", "open_frontiers",
        "checker_receipt", *_MODEL_AUTHORITY}, "conditional model evidence")
    if evidence["schema"] != "terminal-codebase-conditional-model-evidence@1":
        raise IntentCodebaseMatchingError("exact conditional model evidence version required")
    if any(evidence[field] is not False for field in _MODEL_AUTHORITY):
        raise IntentCodebaseMatchingError("conditional evidence grants no behavioral or runtime authority")
    if (evidence["source_path"] != snapshot["source_path"]
            or evidence["source_sha256"] != snapshot["source_sha256"]):
        raise IntentCodebaseMatchingError("conditional model full current source identity differs")
    units = _units(evidence["source_unit_bindings"], snapshot["source_bytes"])
    current = {row["symbol"]: row for row in snapshot["source_unit_bindings"]}
    if not units or any(current.get(row["symbol"]) != row for row in units) or dims["slice"] != units:
        raise IntentCodebaseMatchingError("exact current native source unit binding differs")
    premises = _array(evidence["premises"], "conditional model premises", 7)
    if len(premises) != 7 or premises[:6] != _PREMISES or dims["assumptions"] != premises:
        raise IntentCodebaseMatchingError("complete conditional model premises differ")
    protocol = _object(premises[6], {"reviewed_protocol"}, "conditional protocol premise")["reviewed_protocol"]
    _object(protocol, {"callback_parameter", "review_ref"}, "reviewed WSGI protocol")
    if protocol["callback_parameter"] != "start_response":
        raise IntentCodebaseMatchingError("exact reviewed WSGI protocol role required")
    _text(protocol["review_ref"], "reviewed protocol premise", 4096)
    if not _strings(evidence["open_frontiers"], "open model frontier"):
        raise IntentCodebaseMatchingError("conditional model must retain explicit open frontiers")
    receipt = evidence["checker_receipt"]
    if type(receipt) is not dict:
        raise IntentCodebaseMatchingError("full native checker receipt required")
    if evidence["model_domain"] == "conditional_header_string_model":
        if (evidence["symbol"] not in {"_hkey", "_hval"}
                or [row["symbol"] for row in units] != [evidence["symbol"]]
                or evidence["property"] not in _STRING_PROPERTIES):
            raise IntentCodebaseMatchingError("exact string-model property and symbol relationship required")
        answer = receipt.get("solver_answer")
        classification = {"sat": "conditional_model_sat_witness", "unsat": "conditional_model_unsat"}.get(answer)
        if (classification is None or evidence["classification"] != classification
                or receipt.get("symbol") != evidence["symbol"] or receipt.get("kind") != evidence["property"]
                or dims["evidence_kind"] != "solver_result"
                or receipt.get("matches_model_expectation") is not True):
            raise IntentCodebaseMatchingError("native string model answer classification differs")
        from ipfs_datasets_py.logic.backends.smt.compiler import SoftwareVerificationSMTCompiler, SmtObligation
        try:
            compiled = SoftwareVerificationSMTCompiler().compile(SmtObligation.from_dict(dims["obligation"]))
        except (TypeError, ValueError) as exc:
            raise IntentCodebaseMatchingError("invalid native model compilation: " + str(exc)) from exc
        if (compiled.to_dict() != dims["formalization"] or compiled.smtlib != dims["expression"]
                or receipt.get("compilation_id") != compiled.compilation_id
                or receipt.get("script_sha256") != hashlib.sha256(compiled.smtlib.encode()).hexdigest()
                or receipt.get("script_digest") != compiled.script.digest
                or receipt.get("query_mode") != dims["obligation"].get("query_mode")
                or receipt.get("solver_version") != environment["z3"].get("version")):
            raise IntentCodebaseMatchingError("native solver receipt and complete compiled key differ")
    elif evidence["model_domain"] == "conditional_header_boolean_model":
        if (evidence["symbol"] != "_hkey+_hval" or {row["symbol"] for row in units} != {"_hkey", "_hval"}
                or evidence["property"] != "header_boolean_model"
                or evidence["classification"] != "conditional_model_kernel_checked"
                or dims["evidence_kind"] != "kernel_checked_proof"
                or receipt.get("status") != "passed" or type(receipt.get("returncode")) is not int
                or receipt["returncode"] != 0 or receipt.get("backend_executed") is not True):
            raise IntentCodebaseMatchingError("native conditional Boolean model check differs")
        artifact = _object(receipt.get("artifact"), {"path", "sha256", "bytes"}, "native Lean source artifact")
        artifacts = _array(receipt.get("compiled_artifacts"), "native Lean compiled artifacts", 1)
        if len(artifacts) != 1:
            raise IntentCodebaseMatchingError("one native compiled Lean artifact required")
        for pin in [artifact, *artifacts]:
            _object(pin, {"path", "sha256", "bytes"}, "native Lean artifact pin")
            _text(pin["path"], "native Lean artifact path", 4096)
            _digest(pin["sha256"], "native Lean artifact")
            if type(pin["bytes"]) is not int or not 0 < pin["bytes"] <= 16 * 1024 * 1024:
                raise IntentCodebaseMatchingError("bounded nonempty native Lean artifact required")
        if (receipt.get("expected_success") is not True or receipt.get("matches_expectation") is not True
                or any(receipt.get(field) is not False for field in (
                    "timed_out", "output_truncated", "workspace_limit_exceeded", "resource_exhausted"))
                or receipt.get("executable_sha256") != environment["lean"].get("sha256")
                or artifact["sha256"] != hashlib.sha256(dims["expression"].encode()).hexdigest()
                or artifact["bytes"] != len(dims["expression"].encode())
                or dims["formalization"].get("source_sha256") != artifact["sha256"]
                or dims["formalization"].get("file") != receipt.get("file")
                or receipt.get("compiled_artifacts") != dims["formalization"].get("compiled_artifacts")
                or receipt.get("proof_scope") != "kernel_checked_generated_model_statement_only"
                or receipt.get("proof_scope") != dims["obligation"].get("scope")):
            raise IntentCodebaseMatchingError("native kernel receipt and complete conditional model key differ")
    else:
        raise IntentCodebaseMatchingError("unsupported native conditional model domain")
    entry = {"schema": "terminal-codebase-proof-index-entry@1",
             "key_relationship": value["key_relationship"], "evidence": evidence}
    if value["entry_id"] != "sha256:" + _index_digest(entry):
        raise IntentCodebaseMatchingError("exact complete native evidence entry identity differs")
    return value


def match_intent_codebase(*, intent_document, source_text, source_identity, query,
                         evidence_rows, current_source_snapshot):
    """Freeze native intent, request focus, current source, and residual model hits.

    The owner must first validate live source/checkpoint/index availability.
    This pure consumer rechecks closed envelopes and complete key identities;
    it does not rerun source inference, solvers, kernels, or semantic lowering.
    Even an exact SAT witness is only a counterexample in its declared local
    model, never a refutation of the requested runtime software behavior.
    """
    native, native_id, identity = _native(intent_document, source_text, source_identity)
    query = _query(query)
    snapshot = _snapshot(current_source_snapshot)
    statement = next((row for row in native.statements
                      if row.statement_id == query["statement"]["statement_id"]), None)
    if statement is None or {"statement_id": statement.statement_id,
            "predicate": statement.predicate, "arguments": list(statement.arguments)} != query["statement"]:
        raise IntentCodebaseMatchingError("query must bind one exact native statement ID, predicate and arguments")
    rows = [_evidence_row(row, snapshot) for row in _array(evidence_rows, "native model evidence", MAX_EVIDENCE_ROWS)]
    relationships = [row["key_relationship"]["relationship_id"] for row in rows]
    if len(relationships) != len(set(relationships)):
        raise IntentCodebaseMatchingError("duplicate or ambiguous native evidence key relationship")
    rows.sort(key=lambda row: row["key_relationship"]["relationship_id"])
    reasons = _profile_reasons(query)
    if (statement.predicate != "repair" or list(statement.arguments) != ["agent", "bottle"]
            or statement.kind.value != "goal" or statement.modality.value not in {"required", "intended", "prohibited"}):
        reasons.append("unsupported_native_atom_for_reviewed_focus")
    nominations = []
    if not reasons:
        for row in rows:
            if row["status"] == "hit" and set(query["symbols"]) & {
                    unit["symbol"] for unit in row["evidence"]["source_unit_bindings"]}:
                evidence = row["evidence"]
                nominations.append({"entry_id": row["entry_id"],
                    "relationship_id": row["key_relationship"]["relationship_id"],
                    "symbol": evidence["symbol"], "property": evidence["property"],
                    "model_domain": evidence["model_domain"], "classification": evidence["classification"],
                    "matching_rule": "reviewed_header_focus_related_conditional_model_property@1",
                    "behavioral_satisfaction": False, "runtime_refutation": False,
                    "local_model_counterexample": evidence["classification"] == "conditional_model_sat_witness"})
    selected_reasons = [*reasons, "query_semantic_alignment_unproved",
        "request_domain_coverage_unproved", "conditional_model_source_semantics_unqualified"]
    if not nominations:
        selected_reasons.append("no_matching_evidence_means_unknown")
    if query["polarity"] == "prohibition":
        selected_reasons.append("prohibition_requires_qualified_evidence_not_absence")
    residuals = []
    for item in sorted(native.statements, key=lambda row: row.statement_id):
        selected = item.statement_id == statement.statement_id
        source_refs = []
        for ref in native.sources:
            if ref.ref_id in item.source_ref_ids:
                span = ref.span
                source_refs.append({"reference": ref.to_dict(),
                    "original_text": source_text[span.start_char:span.end_char]})
        residuals.append({"requirement_id": "native-statement:" + item.statement_id,
            "statement": item.to_dict(), "original_source_refs": source_refs,
            "status": "unresolved_software_behavior",
            "selected_for_authored_focus": selected,
            "behavioral_satisfaction": False,
            "reasons": selected_reasons if selected else ["native_statement_not_covered_by_authored_focus"],
            "query_polarity": query["polarity"] if selected else None})
    result = {"schema": SCHEMA,
        "status": "nominated_conditional_model_only" if nominations else "unknown",
        "software_behavior_status": "unresolved_software_behavior",
        "intent_document": native.to_dict(), "native_document_sha256": native_id,
        "intent_source": {"identity": identity, "text": source_text},
        "query": query, "current_source_snapshot": snapshot,
        "evidence_rows": rows, "model_nominations": nominations,
        "residual_requirements": residuals, "policy": _json(_POLICY),
        "roots": {"native_document_sha256": native_id,
            "intent_source_sha256": "sha256:" + identity["content_sha256"],
            "intent_source_identity_sha256": _sha(identity), "query_sha256": _sha(query),
            "current_source_snapshot_sha256": _sha(snapshot), "evidence_rows_sha256": _sha(rows),
            "policy_sha256": _sha(_POLICY)},
        "current_behavioral_facts": [], "behavioral_satisfied_requirements": [],
        "runtime_refutations": [], "removed_task_ids": [],
        "behavioral_satisfaction": False, "domain_coverage_verified": False,
        "native_checker_invocations_here": 0, "source_inference_replayed_here": False,
        "native_persistence_verified_here": False,
        **_AUTHORITY}
    result["match_sha256"] = _sha(result)
    return _json(result)


def reviewed_header_matching_domain():
    """Return a detached explicit request focus, never inferred native meaning."""
    return _json(_POLICY["domain"])


__all__ = ["IntentCodebaseMatchingError", "SCHEMA", "QUERY_SCHEMA",
           "match_intent_codebase", "reviewed_header_matching_domain"]
