"""Run a bounded, public-source repository IR qualification experiment.

This exercises the native supervisor preplanning path, real code-only training,
isolated metadata persistence and local mathematical checkers. It neither
dispatches a coding worker nor claims an official Terminal-Bench result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time


SCHEMA = "terminal-codebase-ir-experiment@1"
PUBLIC_BOTTLE_SHA256 = "761756ce31753e526c48d28ccbca13a5d2493b16fe37aff3e1e4d2efaf3a2bba"
# A finite-window numerical test, fixed before this experiment. It is not a
# theorem that Adam on a nonconvex autoencoder converges for arbitrary inputs.
TAIL_WINDOW = 4
TAIL_ABSOLUTE_TOLERANCE = 0.0002


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_wire(value))
        stream.write(b"\n")


def _progress(stage, **details):
    print(json.dumps({"stage": stage, **details}, sort_keys=True), flush=True)


def verify_instruction_binding(fixture, original_instruction):
    expected = fixture["instruction_provenance"]["sha256"]
    captured = Path(fixture["repository"]) / ".supervisor-instruction.md"
    if (_sha(Path(original_instruction).read_bytes()) != expected
            or _sha(captured.read_bytes()) != expected):
        raise ValueError("public instruction changed from captured fixture provenance")
    return expected


def _execution_sources(functions):
    paths = {Path(__file__).resolve()}
    paths.update(Path(sys.modules[function.__module__].__file__).resolve() for function in functions)
    result = []
    for path in sorted(paths):
        raw = path.read_bytes()
        result.append({"path": str(path), "sha256": _sha(raw), "bytes": len(raw)})
    return result


def assess_training(metrics):
    """Report the declared finite training criteria without hiding failures."""
    before = metrics["before_reconstruction_loss"]
    after = metrics["after_reconstruction_loss"]
    losses = metrics["native_objective_losses"]
    gradients = metrics["native_gradient_norms"]
    if (not losses or len(losses) != metrics["epochs"] or len(gradients) != len(losses)
            or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
                   for x in [before, after, *losses, *gradients])):
        raise ValueError("complete finite loss and gradient evidence required")
    if before <= 0:
        raise ValueError("positive initial loss required for improvement measurement")
    # The native objective also contains weighted cosine/auxiliary losses.
    # Its per-epoch trajectory must never append the endpoint MSE as if the
    # quantities were interchangeable.
    curve = losses
    tail = curve[-(TAIL_WINDOW + 1):]
    changes = [abs(b - a) for a, b in zip(tail, tail[1:])]
    changed = metrics["initial_weights_sha256"] != metrics["final_weights_sha256"]
    return {
        "before_loss": before, "after_loss": after,
        "loss_ratio": after / before,
        "relative_loss_reduction": 1 - after / before,
        "weights_changed": changed,
        "nonzero_gradient_observed": any(x > 0 for x in gradients),
        "finite_loss_and_gradients": True,
        "loss_decreased": after < before,
        "tail_window": TAIL_WINDOW,
        "tail_absolute_tolerance": TAIL_ABSOLUTE_TOLERANCE,
        "tail_absolute_changes": changes,
        "tail_metric": "native_combined_reconstruction_cosine_objective",
        "finite_window_stable": len(changes) == TAIL_WINDOW
            and max(changes) <= TAIL_ABSOLUTE_TOLERANCE,
        "every_native_objective_step_monotone": all(b <= a for a, b in zip(curve, curve[1:])),
        "asymptotic_optimizer_convergence_proved": False,
        "held_out_accuracy_measured": metrics["holdout_evaluated"],
        "training_scope": metrics["training_scope"],
    }


def _training_controls(fixture, output):
    import torch
    from ipfs_datasets_py.logic.formalization.autoencoder.security import codebase_autoencoder as ae

    learner = fixture["learner"]
    repository = Path(learner["repository"])
    variants = [("same_seed_replay", learner["metrics"]["seed"], learner["epochs_completed"]),
                ("independent_seed_2718", 2718, 32),
                ("independent_seed_31415", 31415, 32)]
    controls = []
    for name, seed, epochs in variants:
        started = time.monotonic()
        trained = ae.train_codebase_autoencoder(repository=repository,
            paths=list(learner["source_hashes"]), source_hashes=learner["source_hashes"],
            output=output / name / "code-autoencoder", seed=seed, epochs=epochs)
        checked = ae.validate_codebase_autoencoder(repository=repository, expected_receipt=trained)
        assessment = assess_training(trained["metrics"])
        if not (assessment["weights_changed"] and assessment["nonzero_gradient_observed"]
                and assessment["loss_decreased"] and checked["status"] == "verified"):
            raise ValueError("actual training improvement qualification failed: " + name)
        if name == "same_seed_replay" and (
                trained["checkpoint_sha256"] != learner["checkpoint_sha256"]
                or trained["metrics"]["native_objective_losses"] != learner["metrics"]["native_objective_losses"]):
            raise ValueError("same-source same-seed training is not reproducible")
        _write(output / name / "descriptor.json", trained)
        controls.append({"name": name, "seed": seed, "epochs": epochs,
            "checkpoint_sha256": trained["checkpoint_sha256"],
            "receipt_sha256": trained["receipt_sha256"],
            "assessment": assessment, "seconds": time.monotonic() - started})
        _progress("training_control", name=name, **assessment)

    checkpoint = json.loads((Path(learner["output"]) / "checkpoint.json").read_text())
    features = json.loads((Path(learner["output"]) / "features.json").read_text())
    parameters = [torch.zeros_like(torch.tensor(value, dtype=torch.float32))
                  for value in checkpoint["weights"]]
    model_off = ae._inference(torch, features["rows"], parameters)
    controls.append({"name": "zero_weights_inference", "checkpoint_mutated": False,
        "mean_reconstruction_error": model_off["mean_reconstruction_error"],
        "trained_mean_reconstruction_error": learner["metrics"]["after_reconstruction_loss"],
        "trained_better_than_zero_weights": learner["metrics"]["after_reconstruction_loss"]
            < model_off["mean_reconstruction_error"],
        "source_models_or_proofs_derived_from_rank": False})
    return controls


def run_experiment(*, source: Path, instruction: Path, output: Path):
    from .terminal_codebase_supervisor_fixture import (
        prepare_terminal_codebase_fixture, extract_terminal_codebase_metadata_records,
        bound_terminal_codebase_metadata_records, reconstruct_terminal_codebase_metadata_records,
    )
    from .terminal_codebase_logic_qualification import qualify_terminal_codebase_logic
    from .codebase_ir_metadata import hydrate_codebase_ir_metadata, validate_codebase_ir_metadata

    source = source.resolve(strict=True)
    instruction = instruction.resolve(strict=True)
    output = output.absolute()
    if output.exists() or output.resolve() != output:
        raise ValueError("a fresh canonical experiment output is required")
    raw = source.read_bytes()
    if _sha(raw) != PUBLIC_BOTTLE_SHA256:
        raise ValueError("this first experiment requires the exact public image Bottle source")
    output.mkdir(parents=True)
    started = time.monotonic()
    phases = {}
    execution_sources = _execution_sources([prepare_terminal_codebase_fixture,
        qualify_terminal_codebase_logic, hydrate_codebase_ir_metadata])
    _write(output / "experiment-policy.json", {
        "schema": SCHEMA, "source_sha256": PUBLIC_BOTTLE_SHA256,
        "training_scope": "transductive_public_source",
        "tail_window": TAIL_WINDOW, "tail_absolute_tolerance": TAIL_ABSOLUTE_TOLERANCE,
        "control_seeds": [2718, 31415], "control_epochs": 32,
        "provider_calls_permitted": 0, "official_verifier_inputs_permitted": False,
        "worker_dispatch": False, "production_catalog_changes": False,
        "asymptotic_optimizer_convergence_claim": False,
        "execution_sources": execution_sources,
    })
    # Fail before supervisor indexing/training if this interpreter cannot open
    # the actual digest-pinned native history backend. Never relax its pins.
    stage = time.monotonic()
    from ipfs_datasets_py.ducklake.autoencoder_history import _native_connection
    import duckdb
    import numpy
    import torch
    connection, native_runtime = _native_connection()
    try:
        engine_version = connection.execute("SELECT version()").fetchone()[0]
    finally:
        connection.close()
    runtime_preflight = {"python_executable": sys.executable, "python_version": sys.version,
        "duckdb_module": duckdb.__file__, "duckdb_version": duckdb.__version__,
        "duckdb_engine_version": engine_version, "native_runtime": native_runtime,
        "torch_module": torch.__file__, "torch_version": torch.__version__,
        "numpy_module": numpy.__file__, "numpy_version": numpy.__version__}
    _write(output / "runtime-preflight.json", runtime_preflight)
    phases["native_backend_preflight"] = time.monotonic() - stage
    _progress("native_backend_preflight_complete", duckdb=duckdb.__version__, torch=torch.__version__)
    stage = time.monotonic()
    _progress("supervisor_preplanning_start")
    fixture = prepare_terminal_codebase_fixture(source=source, instruction=instruction,
                                                output=output / "supervisor")
    instruction_sha256 = verify_instruction_binding(fixture, instruction)
    _write(output / "fixture.json", fixture)
    phases["supervisor_preplanning_and_training"] = time.monotonic() - stage
    learner = fixture["learner"]
    assessment = assess_training(learner["metrics"])
    _progress("supervisor_preplanning_complete", samples=learner["sample_count"], **assessment)
    if not (assessment["weights_changed"] and assessment["loss_decreased"]):
        raise ValueError("initial real training failed its improvement gate")

    stage = time.monotonic()
    controls = _training_controls(fixture, output / "training-controls")
    phases["training_controls"] = time.monotonic() - stage
    stage = time.monotonic()
    _progress("logic_and_lean_start")
    logic = qualify_terminal_codebase_logic(source_bytes=raw, source_path="bottle.py",
        output=output / "logic", training_metrics=learner["metrics"])
    if logic.get("status") != "qualified_local_model_and_recorded_losses":
        raise ValueError("actual Lean/solver qualification failed or is unavailable")
    _write(output / "logic-result.json", logic)
    phases["logic_projection_and_actual_checkers"] = time.monotonic() - stage
    _progress("logic_and_lean_complete", status=logic.get("status"))

    stage = time.monotonic()
    records = reconstruct_terminal_codebase_metadata_records(
        extract_terminal_codebase_metadata_records(fixture))
    for family, rows in logic.get("metadata_records", {}).items():
        if family in records:
            records[family].extend(rows)
        else:
            records[family] = rows
    records["experiment_training_controls"] = controls
    records["logic_qualification"] = [logic]
    records["convergence_observations"] = [assessment]
    records["execution_sources"] = execution_sources
    verify_instruction_binding(fixture, instruction)
    source_snapshot = {"schema": "terminal-codebase-ir-source-snapshot@1",
        "public_bottle_sha256": PUBLIC_BOTTLE_SHA256,
        "public_instruction_sha256": instruction_sha256,
        "repository": fixture["repository"],
        "sources": fixture["prepared"]["manifest"]["payload"]["sources"],
        "training_sources": learner["source_hashes"],
        "baseline_commit": fixture["baseline_commit"],
        "signed_manifest_sha256": _sha(_wire(fixture["prepared"]["manifest"])),
        "checkpoint_sha256": learner["checkpoint_sha256"],
        "model_domain": learner["domain"],
        "official_verifier_in_context": False}
    original_records_sha256 = _sha(_wire(records))
    stored_records = bound_terminal_codebase_metadata_records(records)
    if _wire(reconstruct_terminal_codebase_metadata_records(stored_records)) != _wire(records):
        raise ValueError("bounded metadata packaging changed complete producer records")
    _progress("metadata_hydration_start", family_counts={k:len(v) for k,v in stored_records.items()})
    hydration = hydrate_codebase_ir_metadata(records=stored_records, output=output / "metadata",
                                           source_snapshot=source_snapshot)
    replay = validate_codebase_ir_metadata(output=output / "metadata", expected=hydration,
                                          fresh_process=True)
    restored = {}
    for family, export in hydration["exports"].items():
        export_path = output / "metadata" / export["relative_path"]
        restored[family] = [json.loads(line)["payload"] for line in export_path.read_text().splitlines()]
    recovered = reconstruct_terminal_codebase_metadata_records(restored)
    if _sha(_wire(recovered)) != original_records_sha256:
        raise ValueError("fresh-process hydrated artifacts differ from complete producer records")
    reconstruction = {"status": "exact", "original_records_sha256": original_records_sha256,
        "recovered_records_sha256": _sha(_wire(recovered)),
        "family_counts": {k:len(v) for k,v in records.items()},
        "bounded_family_counts": hydration["family_counts"],
        "all_fields_preserved": True, "truncated_records": 0}
    _write(output / "metadata-reconstruction.json", reconstruction)
    _write(output / "metadata-result.json", hydration)
    _write(output / "metadata-replay.json", replay)
    phases["metadata_hydration_and_fresh_process_replay"] = time.monotonic() - stage
    _progress("metadata_hydration_complete", family_counts=hydration["family_counts"])

    if source.read_bytes() != raw or (Path(fixture["repository"]) / "bottle.py").read_bytes() != raw:
        raise ValueError("public input source changed during the experiment")
    verify_instruction_binding(fixture, instruction)
    if any(_sha(Path(row["path"]).read_bytes()) != row["sha256"] for row in execution_sources):
        raise ValueError("experiment implementation changed while the process was running")
    manifest = {"schema": "codebase-ir-experiment-manifest@1", "source_snapshot": source_snapshot,
        "training": {"learner_receipt_sha256": learner["receipt_sha256"],
                     "checkpoint_sha256": learner["checkpoint_sha256"], "assessment": assessment},
        "metadata": hydration, "metadata_reconstruction": reconstruction, "logic": logic,
        "scope": "one_public_terminal_bench_repository_preplanning",
        "learned_feature_representation": True,
        "source_semantics_verified_for_whole_repository": False,
        "asymptotic_optimizer_convergence_proved": False,
        "execution_authority": False, "completion_authority": False}
    root = "sha256:" + _sha(_wire(manifest))
    _write(output / "codebase-ir-manifest.json", {**manifest, "manifest_id": root})
    outcomes = {
        "supervisor_preplanning_and_actual_training": "passed",
        "same_seed_checkpoint_and_objective_replay": "passed",
        "independent_seed_loss_reduction": "passed",
        "native_metadata_and_lossless_fresh_process_replay": "passed",
        "source_bound_local_smt_and_lean_checks": "passed",
        "finite_window_training_stability": "passed" if assessment["finite_window_stable"] else "not_met",
        "asymptotic_optimizer_convergence": "unproved",
        "whole_repository_semantics": "unproved",
        "all_logic_family_projections": "not_qualified",
        "learned_code_to_formula_decoder": "not_exercised",
    }
    result = {"schema": SCHEMA, "status": "completed", "output": str(output),
        "status_meaning": "experiment completed; individual qualification outcomes remain separate",
        "qualification_outcomes": outcomes,
        "codebase_ir_manifest_id": root, "source_snapshot": source_snapshot,
        "sample_count": learner["sample_count"], "training_assessment": assessment,
        "training_controls": controls, "logic": logic, "metadata": hydration,
        "metadata_reconstruction": reconstruction,
        "execution_sources": execution_sources,
        "runtime_preflight": runtime_preflight,
        "metadata_fresh_process_replay": replay, "phase_seconds": phases,
        "elapsed_seconds": time.monotonic() - started,
        "provider_calls": 0, "production_catalog_mutated": False,
        "official_reward": None, "official_verifier_executed": False,
        "whole_repository_proved": False, "asymptotic_optimizer_convergence_proved": False}
    _write(output / "result.json", result)
    _progress("experiment_complete", result=str(output / "result.json"), elapsed=result["elapsed_seconds"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--instruction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    existed_before = args.output.exists()
    try:
        run_experiment(source=args.source, instruction=args.instruction, output=args.output)
    except Exception as exc:
        _progress("experiment_failed", error_type=type(exc).__name__, message=str(exc))
        if not existed_before and (args.output / "experiment-policy.json").is_file():
            _write(args.output / "failure.json", {"schema": SCHEMA, "status": "failed",
                "error_type": type(exc).__name__, "message": str(exc),
                "successful_completion_claimed": False,
                "retained_artifacts": "prior completed stages remain inspectable"})
        raise


if __name__ == "__main__":
    main()
