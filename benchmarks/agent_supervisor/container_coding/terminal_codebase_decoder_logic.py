"""Replay learned source candidates before bounded native model checking.

The decoder supplies function candidates. Exact source correspondence is checked
independently; native logic compilation remains a deterministic downstream step.
Authored oracle controls have a separate entry point and cannot qualify as a
learned decoder run. No source is executed or repaired by this module.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath

from . import terminal_codebase_logic_qualification as logic

SCHEMA = "terminal-codebase-decoder-logic@1"
ORACLE_SCHEMA = "terminal-codebase-decoder-oracle@1"
AUTHORITY = {**logic.AUTHORITY, "learned_decoder_semantics_verified": False}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _input(source_bytes, source_path, output):
    if (type(source_bytes) is not bytes or not 0 < len(source_bytes) <= 1_048_576
            or type(source_path) is not str or not source_path
            or PurePosixPath(source_path).is_absolute() or "\\" in source_path
            or any(part in ("", ".", "..") for part in source_path.split("/"))):
        raise ValueError("bounded exact source bytes and canonical source path required")
    output = Path(output).absolute()
    if output.exists() or output.is_symlink() or output.resolve() != output:
        raise ValueError("fresh canonical decoder logic output required")
    return output


def _source_units(source_bytes):
    from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formula_corpus import _qualified_functions
    from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formalization_evaluation import _function_span

    tree = ast.parse(source_bytes.decode("utf-8"), type_comments=True)
    result = {}
    for node, qualified_name, scope in _qualified_functions(tree):
        if qualified_name in result:
            # Unrelated conditional/nested duplicate definitions must not erase
            # the unique selected header units. Selecting an ambiguous unit
            # itself fails the correspondence gate below.
            result[qualified_name] = None
            continue
        body, binding = _function_span(source_bytes, node)
        result[qualified_name] = {"node": node, "scope": scope, "body": body,
            "binding": binding, "source_ast_sha256": _sha(ast.dump(node, include_attributes=False).encode())}
    return result


def _correspondence(row, *, units, source_bytes, source_path, protocol):
    from ipfs_datasets_py.logic.security_ir import code_header_derivation as header
    from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formula_logic_v2 import lower_pure_function_v2

    unit = units.get(row["qualified_name"])
    if unit is None or (row["symbol"] != unit["node"].name
            or row["source_path"] != source_path or row["source_sha256"] != _sha(source_bytes)
            or row["normalized_body_sha256"] != _sha(unit["body"])
            or row["source_ast_sha256"] != unit["source_ast_sha256"]
            or any(row["source_binding"].get(key) != value for key, value in unit["binding"].items())):
        raise ValueError("decoder candidate source span/hash/AST identity differs")
    candidate = row["candidate_source"]
    receipt = {"schema": "terminal-decoder-source-correspondence@1",
        "unit_id": row["unit_id"], "source_path": source_path,
        "qualified_name": row["qualified_name"], "symbol": row["symbol"],
        "source_sha256": _sha(source_bytes), "source_binding": unit["binding"],
        "source_ast_sha256": unit["source_ast_sha256"],
        "candidate_source_sha256": None, "candidate_ast_sha256": None,
        "candidate_ast_matches_source": False, "native_header_candidate": None,
        "pure_v2_status": "not_run", "pure_v2_frontier": None,
        "status": "rejected", "frontiers": [], **AUTHORITY}
    if type(candidate) is not str or not 0 < len(candidate.encode()) <= 65536:
        receipt["frontiers"] = ["no_bounded_learned_candidate_function"]
        return receipt
    receipt["candidate_source_sha256"] = _sha(candidate.encode())
    try:
        tree = ast.parse(candidate, type_comments=True)
        if (len(tree.body) != 1 or type(tree.body[0]) is not ast.FunctionDef
                or tree.type_ignores or tree.body[0].name != row["symbol"]):
            raise ValueError("candidate_must_be_exact_selected_function")
        candidate_ast = ast.dump(tree.body[0], include_attributes=False)
        receipt["candidate_ast_sha256"] = _sha(candidate_ast.encode())
        if receipt["candidate_ast_sha256"] != unit["source_ast_sha256"]:
            raise ValueError("candidate_AST_differs_from_source")
    except (SyntaxError, ValueError, RecursionError) as exc:
        receipt["frontiers"] = [str(exc)]
        return receipt
    receipt["candidate_ast_matches_source"] = True
    try:
        candidate_model = lower_pure_function_v2(candidate.encode())
        original_model = lower_pure_function_v2(unit["body"])
        if candidate_model != original_model:
            raise ValueError("independently_lowered_scalar_models_differ")
        receipt.update(pure_v2_status="typed_models_match", candidate_typed_ir=candidate_model.to_dict(),
                       source_typed_ir=original_model.to_dict())
    except (SyntaxError, ValueError, RecursionError) as exc:
        receipt.update(pure_v2_status="unsupported", pure_v2_frontier=str(exc))
    try:
        if unit["scope"] or row["qualified_name"] != row["symbol"]:
            raise ValueError("nested_source_unit_outside_declared_top_level_header_binding")
        native = header.validate_header_candidate_function(source_bytes=source_bytes,
            source_path=source_path, protocol=protocol, symbol=row["symbol"],
            candidate_function_source=candidate)
        receipt.update(status="source_matched_header_candidate", native_header_candidate=native)
    except ValueError as exc:
        receipt.update(status="source_matched_other_candidate", frontiers=[str(exc)])
    return receipt


def _qualify(*, source_bytes, source_path, decoder, output, provenance,
             learned, lean_executable, z3_executable):
    from ipfs_datasets_py.logic.security_ir import code_header_derivation as header

    units = _source_units(source_bytes)
    rows = decoder["candidates"]
    if type(rows) is not list or not 1 <= len(rows) <= 1024:
        raise ValueError("complete bounded decoder candidate array required")
    if len({row["qualified_name"] for row in rows}) != len(rows):
        raise ValueError("duplicate decoder candidate source binding")
    protocol = header.WsgiHeaderProtocolContract(logic.REVIEW_PREMISE)
    derivation = header.derive_header_semantics(source_bytes=source_bytes,
        source_path=source_path, protocol=protocol)
    correspondence = [_correspondence(row, units=units, source_bytes=source_bytes,
        source_path=source_path, protocol=protocol) for row in rows]
    required = {row["symbol"] for row in derivation["modeled_symbols"]}
    admitted = {row["symbol"] for row in correspondence if row["status"] == "source_matched_header_candidate"}
    closed = bool(required) and required <= admitted
    output.mkdir(parents=True, mode=0o700)
    checks, receipts, theorem_ids = None, [], []
    executable, unavailable = None, None
    if closed:
        checks = header.check_header_semantics(derivation, source_bytes=source_bytes,
            source_path=source_path, protocol=protocol, z3_executable=z3_executable)
        executable, unavailable = logic._native_lean(lean_executable)
        if executable is not None:
            positive, theorem_ids = logic._header_lean(derivation)
            negative, _ = logic._header_lean(derivation, negative=True)
            receipts = [logic._compile_lean(executable=executable, filename="LearnedSourceHeaderModel.lean",
                            source=positive, output=output, expected_success=True),
                        logic._compile_lean(executable=executable, filename="FalseLearnedGuardControl.lean",
                            source=negative, output=output, expected_success=False)]
    checked = bool(checks and checks["status"] == "checked_local_model")
    lean_passed = len(receipts) == 2 and all(row["matches_expectation"] for row in receipts)
    families = logic._family_inventory(derivation,
        checks or {"status": "not_run", "solver_calls": 0}, lean_passed)
    # Native declarations remain unavailable when learned header coverage fails.
    if not closed:
        for row in families:
            row.update(status="unsupported", reason="required_learned_header_candidates_not_source_matched")
    targets = derivation["native_targets"] if closed else None
    result = {"schema": SCHEMA, "status": ("qualified_learned_header_model" if learned else "qualified_oracle_control")
        if closed and checked and lean_passed else "not_qualified",
        "source_path": source_path, "source_sha256": _sha(source_bytes), "output": str(output),
        "decoder_provenance": provenance, "candidate_generation": "learned_decoder" if learned else "authored_oracle_control",
        "decoder_inference_independently_replayed": learned,
        "ambiguous_source_function_names": sorted(name for name, unit in units.items() if unit is None),
        "candidate_source_correspondence": correspondence,
        "required_header_symbols": sorted(required), "source_matched_header_symbols": sorted(admitted),
        "missing_header_symbols": sorted(required - admitted), "header_candidate_coverage_complete": closed,
        "native_header_derivation": derivation if closed else None,
        "native_smt_check": checks, "lean_checks": receipts, "lean_unavailable_reason": unavailable,
        "header_model_theorem_ids": theorem_ids, "family_inventory": families,
        "decoder_candidate_inventory_count": len(rows),
        "model_generated_function_candidate_count": sum(type(row["candidate_source"]) is str
            and bool(row["candidate_source"]) for row in rows) if learned else 0,
        "learned_source_matched_header_candidate_count": len(admitted) if learned else 0,
        "learned_formula_count": 0,
        "formula_attribution": "deterministic_native_compilation_after_exact_candidate_source_AST_match",
        "lean_scope": "conditional_Boolean_header_model; normalization_and_Python_execution_not_proved",
        "source_executed": False, "canonical_tasks_created": False, "provider_calls": 0,
        "solver_calls": checks["solver_calls"] if checks else 0, "official_reward": None,
        "official_verifier_executed": False,
        "frontiers": list(derivation["open_frontiers"]) + ["no_learned_logic_family_decoder",
            "no_generic_typed_semantics_for_calls_mutation_heap_exceptions_or_dynamic_bindings"], **AUTHORITY}
    result["metadata_records"] = {
        "decoder_provenance": [provenance], "decoder_candidates": rows,
        "decoder_source_correspondence": correspondence,
        "decoder_native_header": [{"derivation": derivation, "targets": targets}] if targets else [],
        "decoder_smt_checks": [checks] if checks else [], "decoder_lean_checks": receipts,
        "decoder_family_status": families,
        "decoder_frontiers": [{"reason": reason, **AUTHORITY} for reason in result["frontiers"]],
    }
    result["qualification_sha256"] = _sha(_wire(result))
    logic._write(output / "decoder-logic-result.json", result)
    return result


def qualify_terminal_codebase_decoder_logic(*, source_bytes, source_path, decoder,
        output, lean_executable=None, z3_executable="z3"):
    """Independently replay a native trained decoder before source/model gates."""
    output = _input(source_bytes, source_path, output)
    from .terminal_codebase_decoder_training import validate_terminal_codebase_decoder, PUBLIC_BOTTLE_SHA256
    if _sha(source_bytes) != PUBLIC_BOTTLE_SHA256:
        raise ValueError("actual decoder lane requires the exact declared public Bottle source")
    validated = validate_terminal_codebase_decoder(expected=decoder,
        source_bytes=source_bytes, source_path=source_path)
    if _wire(validated) != _wire(decoder):
        raise ValueError("decoder descriptor differs from actual checkpoint/inference replay")
    provenance = {"schema": "terminal-learned-decoder-provenance@1",
        "descriptor_sha256": _sha(_wire(validated)), "checkpoint": validated["checkpoint"],
        "training_artifacts_validated": True, "decoder_inference_replayed": True, "optimizer_steps_here": 0,
        "attribution": "actual_native_decoder_inference; deterministic_logic_compiler", **AUTHORITY}
    return _qualify(source_bytes=source_bytes, source_path=source_path, decoder=validated,
        output=output, provenance=provenance, learned=True,
        lean_executable=lean_executable, z3_executable=z3_executable)


def qualify_terminal_codebase_decoder_oracle(*, source_bytes, source_path, candidates,
        output, lean_executable=None, z3_executable="z3"):
    """Named authored test control; never credited as learned inference."""
    output = _input(source_bytes, source_path, output)
    provenance = {"schema": ORACLE_SCHEMA, "checkpoint": None,
        "training_artifacts_validated": False, "decoder_inference_replayed": False, "optimizer_steps_here": 0,
        "attribution": "authored_oracle_control_not_model_output", **AUTHORITY}
    return _qualify(source_bytes=source_bytes, source_path=source_path,
        decoder={"candidates": candidates}, output=output, provenance=provenance,
        learned=False, lean_executable=lean_executable, z3_executable=z3_executable)


__all__ = ["qualify_terminal_codebase_decoder_logic", "qualify_terminal_codebase_decoder_oracle"]
