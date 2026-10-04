"""Join learned public-source reconstruction to checked models and native storage.

This bounded successor experiment preserves the structural run's evidence and
limits. A finite numeric certificate is not an optimizer convergence theorem.
"""
from __future__ import annotations

import argparse
import base64
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
import time

SCHEMA = "terminal-codebase-decoder-experiment@1"
TAIL_WINDOW = 4
TAIL_TOLERANCE_DECIMAL = "0.0002"


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_wire(value) + b"\n")


def _progress(stage, **details):
    print(json.dumps({"stage": stage, **details}, sort_keys=True), flush=True)


def assess_decoder_training(training, final_evaluation):
    """Assess one unchanged native CE recipe, using exact recorded decimals."""
    losses, gradients = training["training_losses"], training["gradient_norms"]
    final = final_evaluation["native_production_cross_entropy"]
    if (type(losses) is not list or type(gradients) is not list or not losses
            or len(losses) != training["epochs"] or len(gradients) != len(losses)
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                   for value in [*losses, *gradients, final])
            or type(training["native_kernel_calls"]) is not int or training["native_kernel_calls"] <= 0):
        raise ValueError("complete finite native decoder loss and gradient evidence required")
    if losses[0] <= 0:
        raise ValueError("positive initial native production loss required")
    tolerance = Fraction(TAIL_TOLERANCE_DECIMAL)
    tail = [Fraction(str(value)) for value in losses[-(TAIL_WINDOW + 1):]]
    deltas = [abs(right-left) for left, right in zip(tail, tail[1:])]
    return {"schema": "finite-decoder-training-assessment@1",
        "objective": "native_family_logits_production_cross_entropy",
        "initial_pre_update_loss": losses[0], "final_checkpoint_loss": final,
        "last_native_trace_loss": losses[-1],
        "native_trace_observation": "before_each_update; final_checkpoint_evaluated_separately",
        "relative_loss_reduction": 1-final/losses[0], "loss_decreased": final < losses[0],
        "weights_changed": training["initial_head_sha256"] != training["final_head_sha256"],
        "nonzero_gradient_observed": any(value > 0 for value in gradients),
        "epochs": training["epochs"], "tail_window": TAIL_WINDOW,
        "tail_tolerance_decimal": TAIL_TOLERANCE_DECIMAL,
        "tail_decimal_rational_changes": [{"numerator": value.numerator, "denominator": value.denominator}
                                           for value in deltas],
        "finite_window_stable": len(deltas) == TAIL_WINDOW and all(value <= tolerance for value in deltas),
        "every_recorded_step_monotone": all(right <= left for left, right in zip(losses, losses[1:])),
        "training_receipt_sha256": _sha(_wire(training)),
        "final_evaluation_sha256": _sha(_wire(final_evaluation)),
        "asymptotic_optimizer_convergence_proved": False,
        "held_out_generalization_proved": False, "proof_authority": False}


def evaluate_final_decoder_objective(*, checkpoint, train_sources):
    """Re-evaluate actual frozen weights with the same native loss kernel."""
    import torch
    from types import SimpleNamespace
    from ipfs_datasets_py.logic.formalization.autoencoder.security import security_formula_decoder as decoder
    from ipfs_datasets_py.logic.formalization.autoencoder.security import security_formula_grammar as grammar
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import modal_autoencoder_cuda as kernel
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import modal_autoencoder_batching as batching

    loaded = decoder.load_security_formula_decoder(checkpoint)
    observations = [grammar.parse_formula_source(source) for source in train_sources]
    data_rows = [row for observed in observations for row in decoder._numeric_rows(observed, loaded["weights"]["lexical"])]
    targets = [grammar.PRODUCTIONS.index(node["teacher_production"])
               for observed in observations for node in observed["nodes"]]
    if not data_rows or len(data_rows) > 8192:
        raise ValueError("bounded train-only objective population required")
    parameters = [torch.tensor(value, dtype=torch.float64) for value in loaded["weights"]["parameters"]]
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            data = torch.tensor(data_rows, dtype=torch.float64)
            one_hot = torch.nn.functional.one_hot(torch.tensor(targets), num_classes=len(grammar.PRODUCTIONS)).double()
            state = SimpleNamespace(torch=torch, device=torch.device("cpu"), family_targets=one_hot,
                family_mask=torch.ones(len(data_rows), dtype=torch.bool))
            session = SimpleNamespace(blocks={}, parameters=parameters,
                parameter_count=sum(parameter.numel() for parameter in parameters))
            logits = decoder._forward(torch, data, parameters)
            empty = torch.zeros((len(data_rows), 0), dtype=torch.float64)
            plan = batching.plan_gradient_accumulation(len(data_rows), microbatch_size=128)
            objective, calls = 0.0, 0
            for start, stop in plan.ranges:
                loss, _, count = kernel._loss_chunk(state, session, (empty, logits, empty),
                    {"family_logits"}, start, stop, len(data_rows), 0., 0., False)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("final decoder checkpoint loss is not finite")
                objective += float(loss); calls += count
    finally:
        torch.set_num_threads(previous_threads)
    decoder.load_security_formula_decoder(checkpoint)
    return {"schema": "frozen-native-decoder-objective@1",
        "native_production_cross_entropy": objective, "native_kernel_calls": calls,
        "source_hashes": [_sha(source) for source in train_sources],
        "production_rows": len(data_rows), "checkpoint": checkpoint,
        "parameters_sha256": loaded["training"]["final_head_sha256"],
        "implementation_sha256": _sha(Path(kernel.__file__).read_bytes()),
        "optimizer_steps": 0, "gradient_updates": 0, "train_only": True,
        "validation_or_test_used_for_this_objective": False, "proof_authority": False}


def qualify_finite_decoder_numbers(*, training, final_evaluation, output):
    """Execute Lean on exact recorded CE ordering and finite tail bounds."""
    from . import terminal_codebase_logic_qualification as lean
    assessment = assess_decoder_training(training, final_evaluation)
    before = Fraction(str(assessment["initial_pre_update_loss"]))
    after = Fraction(str(assessment["final_checkpoint_loss"]))
    if after >= before:
        raise ValueError("recorded decoder objective decrease required for numeric qualification")
    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("fresh numeric qualification output required")
    executable, unavailable = lean._native_lean(None)
    if executable is None:
        raise ValueError("actual Lean required: " + str(unavailable))
    output.mkdir(parents=True, mode=0o700)
    before_cross = before.numerator * after.denominator
    after_cross = after.numerator * before.denominator
    tolerance = Fraction(TAIL_TOLERANCE_DECIMAL)
    predicates = []
    for change in assessment["tail_decimal_rational_changes"]:
        numerator = change["numerator"] * tolerance.denominator
        denominator = tolerance.numerator * change["denominator"]
        predicates.append(f"({numerator} : Nat) <= {denominator}")
    if len(predicates) != TAIL_WINDOW:
        raise ValueError("complete fixed finite stability window required")
    # A failing stability criterion gets a checked failure witness, never a
    # falsely named convergence theorem or a relaxed threshold.
    observed = " ∧ ".join(predicates)
    stability_statement = f"({observed})" if assessment["finite_window_stable"] else f"¬ ({observed})"
    positive = ("-- Recorded decimal observations only; no optimizer asymptotics.\n"
        f"-- Training receipt: {assessment['training_receipt_sha256']}\n"
        f"-- Frozen evaluation: {assessment['final_evaluation_sha256']}\n"
        f"theorem recordedDecoderLossDecreased : ({after_cross} : Nat) < {before_cross} := by decide\n"
        f"theorem recordedFiniteStabilityOutcome : {stability_statement} := by decide\n")
    negative = f"theorem falseDecoderLossControl : ({before_cross} : Nat) <= {after_cross} := by decide\n"
    receipts = [lean._compile_lean(executable=executable, filename="RecordedDecoderTraining.lean",
                    source=positive, output=output, expected_success=True),
                lean._compile_lean(executable=executable, filename="FalseDecoderLossControl.lean",
                    source=negative, output=output, expected_success=False)]
    for receipt in receipts:
        receipt.update(training_receipt_sha256=assessment["training_receipt_sha256"],
            final_evaluation_sha256=assessment["final_evaluation_sha256"],
            numeric_proof_scope="exact_order_and_fixed_tail_predicate_over_recorded_decimals",
            asymptotic_optimizer_convergence_proved=False)
    if not all(receipt["backend_executed"] and receipt["matches_expectation"] for receipt in receipts):
        raise ValueError("actual Lean numeric qualification failed")
    result = {"schema": "finite-decoder-numeric-qualification@1", "status": "checked_recorded_numbers",
        "assessment": assessment, "lean_checks": receipts,
        "float_measurement_rounding_removed": False,
        "future_epochs_or_optimizer_convergence_proved": False, "proof_authority": False}
    _write(output / "qualification.json", result)
    return result


def _artifact(path, *, include_body=False):
    raw = Path(path).read_bytes()
    result = {"path": str(path), "sha256": _sha(raw), "bytes": len(raw)}
    if include_body:
        result["encoding"] = "base64"
        result["body_base64"] = base64.b64encode(raw).decode("ascii")
    return result


def _captured_source(prepared_experiment):
    """Bind this lane to the earlier real supervisor capture, without training."""
    from .terminal_codebase_ir_experiment import PUBLIC_BOTTLE_SHA256
    root = Path(prepared_experiment).resolve(strict=True)
    manifest_path = root / "codebase-ir-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    identifier = manifest.pop("manifest_id")
    if "sha256:" + _sha(_wire(manifest)) != identifier:
        raise ValueError("prepared supervisor IR manifest differs")
    result = json.loads((root / "result.json").read_bytes())
    if result["status"] != "completed" or result["codebase_ir_manifest_id"] != identifier:
        raise ValueError("a completed native supervisor capture is required")
    snapshot = manifest["source_snapshot"]
    for relative, descriptor in snapshot["sources"].items():
        if _sha((Path(snapshot["repository"]) / relative).read_bytes()) != descriptor["sha256"]:
            raise ValueError("prepared supervisor source changed: " + relative)
    source = Path(snapshot["repository"]) / "bottle.py"
    raw = source.read_bytes()
    if _sha(raw) != PUBLIC_BOTTLE_SHA256:
        raise ValueError("exact captured public Bottle source required")
    metadata_manifest = _artifact(root / "metadata/manifest.json")
    if "sha256:" + metadata_manifest["sha256"] != manifest["metadata"]["manifest_sha256"]:
        raise ValueError("referenced parent metadata manifest changed")
    binding = {"schema": "terminal-codebase-decoder-parent-capture@1",
        "prepared_experiment": str(root), "parent_ir_manifest_id": identifier,
        "parent_manifest": _artifact(manifest_path),
        "parent_metadata_manifest": metadata_manifest, "source_snapshot": snapshot,
        "parent_metadata_role": "historical_exact_reference; no current proof facts consumed",
        "source": _artifact(source), "scope": "reuse_captured_supervisor_source_no_worker_dispatch"}
    return raw, binding


def _train_only_sources(source_bytes, training):
    import ast
    from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formula_corpus import _qualified_functions
    from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formalization_evaluation import _function_span
    bodies = {}
    for node, _, _ in _qualified_functions(ast.parse(source_bytes.decode("utf-8"), type_comments=True)):
        body, _ = _function_span(source_bytes, node)
        bodies.setdefault(_sha(body), []).append(body)
    selected = []
    for row in training["splits"]["train"]:
        matching = bodies.get(row["source_sha256"], [])
        if len(matching) != 1:
            raise ValueError("training body is not uniquely bound to captured public source")
        selected.append(matching[0])
    return selected


def run_decoder_experiment(*, prepared_experiment, parent_checkpoint, published_binding, output):
    from .terminal_codebase_decoder_training import train_terminal_codebase_decoder, validate_terminal_codebase_decoder
    from .terminal_codebase_decoder_logic import qualify_terminal_codebase_decoder_logic
    from .terminal_codebase_supervisor_fixture import bound_terminal_codebase_metadata_records, reconstruct_terminal_codebase_metadata_records
    from .codebase_ir_metadata import hydrate_codebase_ir_metadata, validate_codebase_ir_metadata
    from ipfs_datasets_py.logic.formalization.autoencoder.security import security_formula_decoder as native_decoder
    from ipfs_datasets_py.ducklake.autoencoder_history import _native_connection

    output = Path(output).absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("fresh canonical decoder experiment output required")
    raw, parent_capture = _captured_source(prepared_experiment)
    parent_path = Path(parent_checkpoint).resolve(strict=True)
    binding_path = Path(published_binding).resolve(strict=True)
    parent_raw, binding_raw = parent_path.read_bytes(), binding_path.read_bytes()
    parent = json.loads(parent_raw)
    binding = json.loads(binding_raw)
    native_decoder.load_security_formula_decoder(parent)
    output.mkdir(parents=True)
    started = time.monotonic()
    sources = sorted({Path(__file__).resolve(),
        Path(sys.modules[train_terminal_codebase_decoder.__module__].__file__).resolve(),
        Path(sys.modules[qualify_terminal_codebase_decoder_logic.__module__].__file__).resolve()})
    execution_sources = [_artifact(path) for path in sources]
    _write(output / "experiment-policy.json", {"schema": SCHEMA,
        "public_source_sha256": _sha(raw), "parent_capture": parent_capture,
        "audited_parent_descriptor": _artifact(parent_path), "published_binding_descriptor": _artifact(binding_path),
        "source_roles": {"_hkey": "train", "_hval": "validation", "html_escape": "test"},
        "optimizer_epochs": 160, "seed": 1729, "production_head_initialization": "fresh_seeded_head",
        "lexical_initialization": "exact_frozen_published_rows", "continuation_training": False,
        "tail_window": TAIL_WINDOW, "tail_tolerance_decimal": TAIL_TOLERANCE_DECIMAL,
        "model_success_selected_retry": False, "provider_calls_permitted": 0,
        "source_execution": False, "planning_or_admission": False,
        "execution_sources": execution_sources})
    connection, runtime = _native_connection()
    try:
        engine_version = connection.execute("SELECT version()").fetchone()[0]
    finally:
        connection.close()
    import duckdb
    import numpy
    import torch
    _write(output / "runtime-preflight.json", {"python_executable": sys.executable,
        "python_version": sys.version, "native_runtime": runtime, "duckdb_engine_version": engine_version,
        "duckdb_module": duckdb.__file__, "torch_version": torch.__version__, "torch_module": torch.__file__,
        "numpy_version": numpy.__version__, "numpy_module": numpy.__file__, "training_device": "cpu"})
    _progress("decoder_actual_training_start")
    phase_started = time.monotonic()
    decoder = train_terminal_codebase_decoder(source_bytes=raw, source_path="bottle.py",
        output=output / "decoder", parent_checkpoint=parent, published_binding=binding)
    _write(output / "decoder-result.json", decoder)
    training_seconds = time.monotonic()-phase_started
    validated = validate_terminal_codebase_decoder(expected=decoder, source_bytes=raw, source_path="bottle.py")
    if _wire(validated) != _wire(decoder):
        raise ValueError("trained decoder inference replay differs")
    loaded = native_decoder.load_security_formula_decoder(decoder["checkpoint"])
    training = loaded["training"]
    if training["epochs"] != 160 or training["seed"] != 1729:
        raise ValueError("declared immutable decoder training budget differs")
    final_evaluation = evaluate_final_decoder_objective(checkpoint=decoder["checkpoint"],
        train_sources=_train_only_sources(raw, training))
    _write(output / "final-checkpoint-objective.json", final_evaluation)
    numeric = qualify_finite_decoder_numbers(training=training, final_evaluation=final_evaluation,
        output=output / "numeric-logic")
    _progress("decoder_training_and_numeric_checks_complete", **numeric["assessment"])
    phase_started = time.monotonic()
    logic = qualify_terminal_codebase_decoder_logic(source_bytes=raw, source_path="bottle.py",
        decoder=decoder, output=output / "logic")
    _write(output / "logic-result.json", logic)
    logic_seconds = time.monotonic()-phase_started
    _progress("learned_source_model_checks_complete", status=logic["status"],
        matched_headers=logic["source_matched_header_symbols"], missing_headers=logic["missing_header_symbols"])

    records = dict(logic["metadata_records"])
    records["decoder_training_receipts"] = [training]
    records["decoder_training_assessment"] = [numeric["assessment"], final_evaluation]
    records["decoder_numeric_lean_receipts"] = numeric["lean_checks"]
    records["decoder_full_report"] = [decoder]
    records["parent_preplanning_capture"] = [parent_capture]
    records["decoder_execution_sources"] = execution_sources
    records["decoder_model_artifacts"] = [_artifact(Path(decoder["checkpoint"]["output"])/filename,
        include_body=True) for filename in ("manifest.json", "weights.json", "config.json", "training.json")]
    records["decoder_public_sources"] = [{"source_path": "bottle.py", "source_sha256": _sha(raw),
        "source_bytes": len(raw), "source_text": raw.decode("utf-8"),
        "origin": "exact_previously_captured_public_terminal_bench_environment",
        "hidden_verifier_or_solution_input": False}]
    # These required family views are explicitly empty in the child namespace;
    # complete AST/KG/vector/contract metadata remains in the exact parent.
    for family in ("ast", "kg", "vectors", "contracts"):
        records.setdefault(family, [])
    records = json.loads(_wire(records))
    complete_root = _sha(_wire(records))
    bounded = bound_terminal_codebase_metadata_records(records)
    if _wire(reconstruct_terminal_codebase_metadata_records(bounded)) != _wire(records):
        raise ValueError("bounded decoder metadata changed original fields")
    source_snapshot = {"schema": "terminal-codebase-decoder-source-snapshot@1",
        "public_source_sha256": _sha(raw), "parent_ir_manifest_id": parent_capture["parent_ir_manifest_id"],
        "parent_preplanning_manifest_sha256": parent_capture["parent_manifest"]["sha256"],
        "decoder_checkpoint": decoder["checkpoint"], "decoder_report_sha256": _sha(_wire(decoder)),
        "complete_metadata_sha256": complete_root,
        "source_roles": {"_hkey": "train", "_hval": "validation", "html_escape": "test"},
        "official_verifier_in_context": False}
    phase_started = time.monotonic()
    _progress("decoder_metadata_hydration_start", families=len(bounded), rows=sum(map(len,bounded.values())))
    hydration = hydrate_codebase_ir_metadata(records=bounded, output=output / "metadata", source_snapshot=source_snapshot)
    replay = validate_codebase_ir_metadata(output=output / "metadata", expected=hydration, fresh_process=True)
    restored = {family: [json.loads(line)["payload"] for line in
        (output/"metadata"/descriptor["relative_path"]).read_text().splitlines()]
        for family, descriptor in hydration["exports"].items()}
    recovered = reconstruct_terminal_codebase_metadata_records(restored)
    if _sha(_wire(recovered)) != complete_root:
        raise ValueError("complete decoder metadata native roundtrip differs")
    reconstruction = {"status": "exact", "original_records_sha256": complete_root,
        "recovered_records_sha256": _sha(_wire(recovered)), "zero_truncation": True,
        "original_family_counts": {family:len(rows) for family,rows in records.items()},
        "bounded_family_counts": hydration["family_counts"],
        "parent_28_family_catalog": "retained by exact reference; not duplicated or narrowed"}
    _write(output / "metadata-reconstruction.json", reconstruction)
    _write(output / "metadata-result.json", hydration)
    _write(output / "metadata-replay.json", replay)
    if (_captured_source(prepared_experiment)[0] != raw or parent_path.read_bytes() != parent_raw
            or binding_path.read_bytes() != binding_raw
            or any(_sha(Path(row["path"]).read_bytes()) != row["sha256"] for row in execution_sources)):
        raise ValueError("source, parent binding or experiment implementation changed during execution")
    if _wire(validate_terminal_codebase_decoder(expected=decoder, source_bytes=raw, source_path="bottle.py")) != _wire(decoder):
        raise ValueError("frozen decoder changed before completion")
    manifest = {"schema": "codebase-ir-learned-candidate-experiment@1", "source_snapshot": source_snapshot,
        "parent_capture": parent_capture, "decoder": decoder, "numeric_evidence": numeric,
        "logic": logic, "metadata": hydration, "metadata_reconstruction": reconstruction,
        "execution_sources": execution_sources, "execution_authority": False,
        "completion_authority": False, "whole_repository_proved": False,
        "asymptotic_optimizer_convergence_proved": False}
    manifest_id = "sha256:" + _sha(_wire(manifest))
    _write(output / "codebase-ir-manifest.json", {**manifest, "manifest_id": manifest_id})
    result = {"schema": SCHEMA, "status": "completed", "status_meaning": "bounded experiment completed; qualification outcomes separate",
        "output": str(output), "manifest_id": manifest_id, "source_snapshot": source_snapshot,
        "decoder": decoder, "numeric_evidence": numeric, "logic": logic,
        "metadata": hydration, "metadata_fresh_process_replay": replay,
        "metadata_reconstruction": reconstruction, "parent_capture": parent_capture,
        "qualification_outcomes": {
            "actual_train_only_optimizer_execution_and_frozen_inference_replay": "passed",
            "frozen_checkpoint_loss_reduction": "passed" if numeric["assessment"]["loss_decreased"] else "not_met",
            "finite_window_stability": "passed" if numeric["assessment"]["finite_window_stable"] else "not_met",
            "learned_header_candidate_source_matching_and_checking": "passed" if logic["status"] == "qualified_learned_header_model" else "not_qualified",
            "native_duckdb_ducklake_exact_fresh_process_metadata": "passed",
            "whole_repository_typed_semantics": "unproved", "all_logic_family_compilation": "not_qualified",
            "asymptotic_optimizer_convergence": "unproved", "held_out_generalization": "unproved",
            "intent_ir_planning_and_admission": "not_exercised"},
        "phase_seconds": {"decoder_training": training_seconds, "learned_source_model_checking": logic_seconds,
            "metadata_hydration_and_replay": time.monotonic()-phase_started},
        "elapsed_seconds": time.monotonic()-started, "provider_calls": 0,
        "official_reward": None, "official_verifier_executed": False,
        "execution_sources": execution_sources, "asymptotic_optimizer_convergence_proved": False}
    _write(output / "result.json", result)
    _progress("decoder_experiment_complete", result=str(output/"result.json"), seconds=result["elapsed_seconds"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-experiment", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--published-binding", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    existed = args.output.exists()
    try:
        run_decoder_experiment(prepared_experiment=args.prepared_experiment,
            parent_checkpoint=args.parent_checkpoint, published_binding=args.published_binding, output=args.output)
    except Exception as exc:
        _progress("decoder_experiment_failed", error_type=type(exc).__name__, message=str(exc))
        if not existed and (args.output/"experiment-policy.json").is_file():
            _write(args.output/"failure.json", {"schema": SCHEMA, "status": "failed",
                "error_type": type(exc).__name__, "message": str(exc), "successful_completion_claimed": False})
        raise


if __name__ == "__main__":
    main()
