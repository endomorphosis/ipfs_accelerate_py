"""Bounded source-bound header models and finite numerical Lean evidence.

The feature autoencoder does not decode these formulas. Datasets independently
recognizes the exact source, creates native SecurityIR declarations and checks
its conditional ordinary-string model with Z3. Lean checks a deliberately
smaller Boolean abstraction and the exact decimal ordering of recorded losses.
No source is imported, no worker is dispatched, and no proof grants authority.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil


SCHEMA = "terminal-codebase-logic-qualification@1"
AUTHORITY = {"proof_authority": False, "execution_authority": False,
    "mutation_authority": False, "completion_authority": False,
    "source_semantics_verified": False, "whole_program_proved": False,
    "asymptotic_optimizer_convergence_proved": False}
REVIEW_PREMISE = (
    "public-source local WSGI model: start_response denotes the WSGI callback; "
    "its second argument denotes the unique recognized response-header property; "
    "these caller-reviewed roles are explicit premises, not proved source facts"
)


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write(path, value):
    raw = value.encode() if type(value) is str else _wire(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
    return {"path": str(path), "sha256": _sha(raw), "bytes": len(raw)}


def _loss_evidence(metrics):
    if type(metrics) is not dict:
        raise ValueError("bounded actual training metrics are required")
    rationals = []
    for key in ("before_reconstruction_loss", "after_reconstruction_loss"):
        value = metrics.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("finite nonnegative recorded losses are required")
        decimal = str(value)
        exact = Fraction(decimal)
        if max(len(str(exact.numerator)), len(str(exact.denominator))) > 128:
            raise ValueError("bounded exact decimal losses are required")
        rationals.append({"decimal": decimal, "numerator": exact.numerator,
                          "denominator": exact.denominator})
    try:
        metrics_raw = _wire(metrics)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError("bounded finite JSON training metrics are required") from exc
    if len(metrics_raw) > 262144:
        raise ValueError("bounded actual training metrics are required")
    before, after = rationals
    if not (after["numerator"] * before["denominator"] <
            before["numerator"] * after["denominator"]):
        raise ValueError("recorded endpoints do not establish a strict loss decrease")
    return {"schema": "finite-recorded-loss-ordering@1", "before": before, "after": after,
        "metrics_sha256": _sha(metrics_raw), "exact_decimal_cross_product_decrease": True,
        "scope": "ordering_of_recorded_decimal_endpoints_only",
        "float_measurement_rounding_removed": False, "training_run_independently_replayed": False,
        "training_metrics_binding": "caller_supplied_metrics_pinned_by_sha256",
        "held_out_accuracy_proved": False, **AUTHORITY}


def _loss_lean(evidence, *, negative=False):
    before, after = evidence["before"], evidence["after"]
    left, right = (before, after) if negative else (after, before)
    theorem = "incorrect_reverse_loss_order" if negative else "recorded_loss_decreased"
    return f'''import Std
set_option autoImplicit false
namespace RecordedLoss
-- Exact nonnegative rational ordering, defined by positive-denominator cross products.
-- These are decimal representations of recorded measurements, not an optimizer theorem.
def exactLess (leftNum leftDen rightNum rightDen : Nat) : Prop :=
  0 < leftDen ∧ 0 < rightDen ∧ leftNum * rightDen < rightNum * leftDen
theorem {theorem} :
    exactLess {left["numerator"]} {left["denominator"]} {right["numerator"]} {right["denominator"]} := by
  unfold exactLess
  decide
end RecordedLoss
'''


def _header_lean(derivation, *, negative=False):
    lines = ['import Std', 'set_option autoImplicit false',
        'namespace SourceHeaderModel',
        '-- Ordinary converted-string abstraction. Normalization is a supplied String value.',
        '-- Forbidden-control membership and its preservation are explicit model premises.',
        'def accepted (guarded bad : Bool) : Bool := if guarded then !bad else true',
        'def normalizedResult (guarded bad : Bool) (normalized : String) : String :=',
        '  if accepted guarded bad then normalized else ""']
    theorem_ids = []
    for index, row in enumerate(derivation["modeled_symbols"]):
        guard = "true" if row["guarded"] else "false"
        expected = "false" if row["guarded"] else "true"
        if negative:
            expected = "true" if row["guarded"] else "false"
        lines += [f'-- Original symbol {row["symbol"]}; AST sha256 {row["source_ast_sha256"]}.',
                  f'def guarded_{index} : Bool := {guard}']
        if negative:
            name = f"incorrect_guard_claim_{index}"
            lines += [f'theorem {name} : accepted guarded_{index} true = {expected} := by decide']
            theorem_ids.append(name)
            continue
        name = f"source_{index}_unsafe_acceptance_rule"
        lines += [f'theorem {name} : accepted guarded_{index} true = {expected} := by decide']
        theorem_ids.append(name)
        name = f"source_{index}_safe_normalization_preserved"
        lines += [f'theorem {name} (normalized : String) :',
            f'    accepted guarded_{index} false = true ∧ normalizedResult guarded_{index} false normalized = normalized := by',
            f'  simp [guarded_{index}, accepted, normalizedResult]']
        theorem_ids.append(name)
        name = f"source_{index}_forbidden_output_characterization"
        lines += [f'theorem {name} (normalized : String) (hasForbidden : String → Bool)',
            '    (bad : Bool) (normalizationPreservesControls : hasForbidden normalized = bad) :',
            f'    (accepted guarded_{index} bad = true ∧ hasForbidden (normalizedResult guarded_{index} bad normalized) = true) ↔',
            f'      (accepted guarded_{index} bad = true ∧ bad = true) := by',
            f'  cases bad <;> simp [guarded_{index}, accepted, normalizedResult, normalizationPreservesControls]']
        theorem_ids.append(name)
    lines.append('end SourceHeaderModel')
    return "\n".join(lines) + "\n", theorem_ids


def _native_lean(executable):
    if executable is None:
        # Select an installed native binary. Elan shims may trigger downloads.
        candidates = []
        root = Path.home() / ".elan" / "toolchains"
        if root.is_dir():
            for path in root.glob("leanprover--lean4---v*/bin/lean"):
                match = re.fullmatch(r"leanprover--lean4---v(\d+)\.(\d+)\.(\d+)", path.parent.parent.name)
                if match and path.is_file():
                    candidates.append((tuple(map(int, match.groups())), path))
        if candidates:
            executable = str(max(candidates)[1])
        else:
            executable = shutil.which("lean")
    found = None if executable is None else shutil.which(str(executable))
    if found is None:
        return None, "installed_native_lean_missing"
    path = Path(found).absolute()
    if path.parent.name == "bin" and path.parent.parent.name == ".elan":
        return None, "select_installed_native_lean_not_elan_shim"
    if not path.is_file():
        return None, "installed_native_lean_missing"
    return path, None


def _compile_lean(*, executable, filename, source, output, expected_success):
    from ipfs_datasets_py.logic.backends.process import BoundedToolRunner, ToolRunRequest, ToolRunLimits

    pin = _write(output / filename, source)
    object_file = str(PurePosixPath(filename).with_suffix(".olean"))
    command = ((str(executable), "-o", object_file, filename) if expected_success
               else (str(executable), filename))
    run = BoundedToolRunner().run(ToolRunRequest(argv=command,
        input_files={filename: source}, output_paths=(object_file,) if expected_success else (),
        limits=ToolRunLimits(timeout_seconds=20,
            cpu_seconds=20, max_input_bytes=262144, max_output_bytes=65536,
            max_workspace_bytes=16 * 1024 * 1024)))
    passed = run.ok and not run.output_truncated and not run.workspace_limit_exceeded
    expected_failure = (run.returncode is not None and run.returncode != 0
        and not any((run.timed_out, run.cancelled, run.unavailable, run.resource_exhausted,
                     run.output_truncated, run.workspace_limit_exceeded))
        and "Tactic `decide` proved that the proposition" in run.stdout and "is false" in run.stdout)
    compiled_artifacts = []
    for name, raw in run.output_files.items():
        with (output / name).open("xb") as stream:
            stream.write(raw)
        compiled_artifacts.append({"path": str(output / name), "sha256": _sha(raw), "bytes": len(raw)})
    if expected_success and not compiled_artifacts:
        passed = False
    return {"schema": "terminal-codebase-lean-check@1", "file": filename, "artifact": pin,
        "status": "passed" if passed else "rejected" if expected_failure else "inconclusive",
        "expected_success": expected_success,
        "matches_expectation": passed if expected_success else expected_failure,
        "compiled_artifacts": compiled_artifacts,
        "backend_executed": True, "executable": str(executable),
        "executable_sha256": _sha(executable.read_bytes()), "command": list(run.command),
        "returncode": run.returncode, "stdout": run.stdout, "stderr": run.stderr,
        "timed_out": run.timed_out, "output_truncated": run.output_truncated,
        "workspace_limit_exceeded": run.workspace_limit_exceeded,
        "resource_exhausted": run.resource_exhausted, "workspace_cleaned": run.workspace_cleaned,
        "elapsed_seconds": run.elapsed_seconds,
        "proof_scope": "kernel_checked_generated_model_statement_only", **AUTHORITY}


def _family_inventory(derivation, check, lean_passed):
    from ipfs_datasets_py.logic.formalization.autoencoder.family_training import family_training_catalog

    catalog = family_training_catalog("security_ir")
    rows = []
    for item in catalog["family_inventory"]:
        row = {**item, "status": "unsupported", "source_sha256": derivation["source_sha256"],
            "native_training_target_count": 0, "learned_family_decoder_trained": False,
            "checked_model_obligation_count": 0, "lean_compiled": False,
            "reason": "required_explicit_typed_source_model_not_provided", **AUTHORITY}
        if item["family_id"] == "first_order" and check["status"] == "checked_local_model":
            row.update(status="checked_local_header_string_model", checked_model_obligation_count=check["solver_calls"],
                reason="QF_SLIA_SMT_header_profile_under_explicit_premises_not_generic_code_to_FOL_decoder")
        elif item["family_id"] == "propositional" and lean_passed:
            row.update(status="checked_conditional_header_boolean_model", lean_compiled=True,
                reason="independent_Boolean_abstraction_not_native_family_training_target")
        elif item["family_id"] == "transition_system" and derivation["status"] == "modeled":
            row.update(status="native_security_ir_declarations_only",
                reason="SecurityIR_state_machine_records_not_executable_typed_transition_relation")
        rows.append(row)
    return rows


def qualify_terminal_codebase_logic(*, source_bytes, source_path, output, training_metrics,
                                    lean_executable=None, z3_executable="z3"):
    """Create fresh native artifacts, actual checker receipts and full frontiers.

    Positive qualification requires actual Z3 and Lean results and rejected
    false-guard/reverse-loss controls. Losses must already have been authenticated
    by the caller's training receipt; this helper pins their exact supplied values.
    """
    from ipfs_datasets_py.logic.security_ir import code_header_derivation as header
    from ipfs_datasets_py.logic.security_ir.model import SecurityIR
    from ipfs_datasets_py.logic.backends.process import BoundedToolRunner, ToolRunRequest, ToolRunLimits

    if (type(source_bytes) is not bytes or not 0 < len(source_bytes) <= header.MAX_SOURCE_BYTES
            or type(source_path) is not str or PurePosixPath(source_path).is_absolute()
            or "\\" in source_path or any(part in ("", ".", "..") for part in source_path.split("/"))):
        raise ValueError("bounded exact bytes and a canonical repository source path are required")
    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("fresh canonical logic output is required")
    loss = _loss_evidence(training_metrics)
    protocol = header.WsgiHeaderProtocolContract(REVIEW_PREMISE)
    derivation = header.derive_header_semantics(source_bytes=source_bytes, source_path=source_path, protocol=protocol)
    header.validate_header_semantics(derivation, source_bytes=source_bytes, source_path=source_path, protocol=protocol)
    if derivation["status"] == "modeled":
        native = derivation["native_targets"]
        if SecurityIR.from_dict(native["declaration"]).cid != native["declaration_cid"]:
            raise ValueError("native SecurityIR declaration differs from source-bound replay")
        for row in derivation["modeled_symbols"]:
            span = row["source_span"]
            if _sha(source_bytes[span["start_byte"]:span["end_byte"]]) != span["sha256"]:
                raise ValueError("source normalizer byte span differs")
    check = header.check_header_semantics(derivation, source_bytes=source_bytes,
        source_path=source_path, protocol=protocol, z3_executable=z3_executable)
    output.mkdir(parents=True, mode=0o700)
    artifacts = {"derivation": _write(output / "header-derivation.json", derivation),
                 "smt_check": _write(output / "header-smt-check.json", check),
                 "loss": _write(output / "recorded-loss.json", loss)}
    executable, unavailable = _native_lean(lean_executable)
    receipts, names = [], []
    probe = None
    if executable is not None:
        probe_run = BoundedToolRunner().run(ToolRunRequest(argv=(str(executable), "--version"),
            limits=ToolRunLimits(timeout_seconds=5, max_output_bytes=16384)))
        probe = {"returncode": probe_run.returncode, "stdout": probe_run.stdout, "stderr": probe_run.stderr}
        if not probe_run.ok or not re.search(r"Lean \(version \d+\.\d+\.\d+", probe_run.stdout):
            unavailable = "native_lean_version_probe_failed"
            executable = None
    if executable is not None:
        if derivation["status"] == "modeled":
            positive, names = _header_lean(derivation)
            negative, _ = _header_lean(derivation, negative=True)
            receipts.append(_compile_lean(executable=executable, filename="SourceHeaderModel.lean",
                source=positive, output=output, expected_success=True))
            receipts.append(_compile_lean(executable=executable, filename="FalseGuardControl.lean",
                source=negative, output=output, expected_success=False))
        receipts.append(_compile_lean(executable=executable, filename="RecordedLoss.lean",
            source=_loss_lean(loss), output=output, expected_success=True))
        receipts.append(_compile_lean(executable=executable, filename="ReverseLossControl.lean",
            source=_loss_lean(loss, negative=True), output=output, expected_success=False))
    lean_passed = (len(receipts) == 4 and all(row["matches_expectation"] for row in receipts))
    artifact = derivation.get("native_targets", {}).get("formalization") if derivation["native_targets"] else None
    views = []
    if artifact:
        counts = Counter(row["view_id"] for row in artifact["formulas"])
        for view in artifact["view_registry"]["views"]:
            views.append({"view_id": view["view_id"], "logic_family": view["logic_family"],
                "formula_count": counts[view["view_id"]], "status": "native_declarations_only",
                "source_sha256": derivation["source_sha256"], **AUTHORITY})
    families = _family_inventory(derivation, check, lean_passed)
    status = "qualified_local_model_and_recorded_losses" if (lean_passed and check["status"] == "checked_local_model") else "unavailable_or_failed"
    for receipt in receipts:
        receipt.update(source_sha256=derivation["source_sha256"], derivation_cid=derivation["derivation_cid"],
            metrics_sha256=loss["metrics_sha256"], lean_version_probe=probe)
    artifacts["lean_checks"] = _write(output / "lean-checks.json", receipts)
    result = {"schema": SCHEMA, "status": status, "source_path": source_path,
        "source_sha256": derivation["source_sha256"], "output": str(output), "protocol": derivation["protocol"],
        "native_header_derivation": derivation, "native_smt_check": check,
        "native_formalization_views": views, "family_inventory": families,
        "lean_checks": receipts, "lean_unavailable_reason": unavailable,
        "header_model_theorem_ids": names, "finite_loss_evidence": loss, "artifacts": artifacts,
        "learned_formula_count": 0, "feature_autoencoder_generated_these_formulas": False,
        "formula_attribution": "deterministic_native_header_model_independently_replayed_from_source",
        "lean_header_scope": "conditional_Boolean_acceptance_and_abstract_normalized_String_model",
        "lean_normalization_implementation_proved": False, "lean_Python_execution_proved": False,
        "source_guard_present": [row["guarded"] for row in derivation["modeled_symbols"]],
        "provider_calls": 0, "download_calls": 0, "source_executed": False,
        "official_reward": None, "official_verifier_executed": False,
        "implementation_sha256": _sha(Path(__file__).read_bytes()), **AUTHORITY}
    result["metadata_records"] = {
        "source_header_contracts": [derivation],
        "native_security_declarations": [derivation["native_targets"]["declaration"]] if artifact else [],
        "native_formalization_claims": artifact["formulas"] if artifact else [],
        "native_formalization_obligations": artifact["proof_obligations"] if artifact else [],
        "source_bound_smt_obligations": derivation["smt_targets"],
        "source_bound_smt_check_receipts": [check],
        "source_bound_lean_check_receipts": receipts,
        "logic_family_projection_status": families,
        "finite_loss_ordering_evidence": [loss],
    }
    result["qualification_sha256"] = _sha(_wire(result))
    _write(output / "qualification.json", result)
    return result


__all__ = ["qualify_terminal_codebase_logic"]
