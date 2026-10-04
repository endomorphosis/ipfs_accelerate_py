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
INTENT_MATCHING_CONTEXT_SCHEMA = "terminal-codebase-intent-matching-context@1"
MAX_INTENT_MATCHING_CONTEXT_BYTES = 4 * 1024 * 1024
INTENT_NOMINATION_METADATA_SCHEMA = "terminal-codebase-intent-training-nomination-metadata@1"
INTENT_NOMINATION_FAMILY = "intent_codebase_training_nominations"


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


def _training_controls(fixture, output, *, repeat=True, retain_inference=False):
    import torch
    from ipfs_datasets_py.logic.formalization.autoencoder.security import codebase_autoencoder as ae

    learner = fixture["learner"]
    repository = Path(learner["repository"])
    variants = [("same_seed_replay", learner["metrics"]["seed"], learner["epochs_completed"]),
                ("independent_seed_2718", 2718, 32),
                ("independent_seed_31415", 31415, 32)]
    if not repeat:
        variants = []
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
    if retain_inference:
        controls[-1].update(inference=model_off,
            control_weights_sha256=_sha(_wire([value.tolist() for value in parameters])))
    return controls


def _load_intent_matching_context(value, *, instruction_bytes):
    """Load an explicit reviewed context; never derive intent from code or labels."""
    from .terminal_codebase_supervisor_fixture import _read
    from .codebase_ir_metadata import _plain

    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate intent matching context key")
            result[key] = item
        return result

    if isinstance(value, Path):
        raw = _read(value, MAX_INTENT_MATCHING_CONTEXT_BYTES)
        context = json.loads(raw, object_pairs_hook=unique)
    else:
        context = value
    _plain(context)
    raw = _wire(context)
    if len(raw) > MAX_INTENT_MATCHING_CONTEXT_BYTES:
        raise ValueError("intent matching context exceeds the byte bound")
    required = {"schema", "intent_document", "source_text", "source_identity", "query"}
    if (type(context) is not dict or set(context) != required
            or context["schema"] != INTENT_MATCHING_CONTEXT_SCHEMA
            or type(context["source_text"]) is not str
            or context["source_text"].encode("utf-8") != instruction_bytes):
        raise ValueError("explicit native matching context must bind the complete public instruction")
    if any(type(context[field]) is not dict for field in ("intent_document", "source_identity", "query")):
        raise ValueError("native matching document, identity and query objects required")
    return json.loads(raw)


def _nomination_model_controls(*, learner, training_receipt, learned_index, zero_control):
    """Retain actual zero inference and explicit non-inference order controls."""
    base = {"input_features_sha256": training_receipt["features_sha256"],
        "source_hashes": learner["source_hashes"],
        "checkpoint_sha256": learner["checkpoint_sha256"], "training_executed": False}
    ranks = learned_index["ranks"]
    rows = [
        {**base, "name": "trained", "ranking": ranks,
         "inference_policy": "trained_checkpoint_reconstruction"},
        {**base, "name": "model_off", "ranking": [],
         "inference_policy": "model_off_no_inference"},
        {**base, "name": "zero_heads", "ranking": zero_control["inference"]["ranks"],
         "inference_policy": "all_parameters_zero_diagnostic",
         "control_weights_sha256": zero_control["control_weights_sha256"]},
        {**base, "name": "shuffled_order", "ranking": list(reversed(ranks)),
         "inference_policy": "same_trained_rank_reverse_permutation"},
    ]
    return json.loads(_wire(rows))


def _prepare_intent_training_nomination(*, fixture, records, context, logic,
                                       runtime_preflight, controls):
    """Join owner-verified current producers without manufacturing proof hits."""
    from .terminal_codebase_supervisor_fixture import _read
    from .terminal_initial_context import load_initial_context
    from .terminal_codebase_intent_training_join import (
        build_terminal_intent_training_join, validate_terminal_intent_training_join,
    )
    from ipfs_accelerate_py.agent_supervisor.planning.intent_codebase_matching import match_intent_codebase

    loaded = load_initial_context(state=Path(fixture["state"]),
        prepared=fixture["prepared"], require_empty_owner=True)
    learner = loaded["descriptor"]["codebase_autoencoder"]
    learning = Path(learner["output"])
    assets = {name: json.loads(_read(learning / (name + ".json"), 4_000_000))
              for name in ("receipt", "checkpoint", "features", "index")}
    source = next(row for row in records["sources"] if row["path"] == "bottle.py")
    source_context = {"schema": "terminal-codebase-training-source-context@1",
        "source_path": "bottle.py", "source_sha256": source["source_sha256"],
        "source_bytes": source["bytes"], "model_domain": learner["domain"],
        "training_source_hashes": learner["source_hashes"],
        "checkpoint": {"checkpoint_sha256": learner["checkpoint_sha256"],
            "receipt_sha256": learner["receipt_sha256"],
            "features_sha256": assets["receipt"]["features_sha256"],
            "index_sha256": assets["receipt"]["index_sha256"]}}
    derivation = logic["native_header_derivation"]
    if (derivation["source_sha256"] != source["source_sha256"]
            or source["source_text"].encode() != _read(Path(fixture["repository"]) / "bottle.py", 2_000_000)):
        raise ValueError("nomination source differs from current owner and actual logic producer")
    environment = {"schema": "terminal-training-advisory-runtime@1", "runtime": runtime_preflight}
    translation = {"schema": "terminal-training-advisory-translation@1",
        "logic_implementation_sha256": logic["implementation_sha256"],
        "derivation_schema": derivation["schema"]}
    snapshot = {"schema": "terminal-codebase-proof-source-snapshot@1",
        "source_path": "bottle.py", "source_sha256": source["source_sha256"],
        "source_bytes": source["bytes"], "source_unit_bindings": derivation["modeled_symbols"],
        "source_context_sha256": _sha(_wire(source_context)),
        "environment_sha256": _sha(_wire(environment)),
        "environment_ref_sha256": _sha(_wire(environment)),
        "translation_sha256": _sha(_wire(translation))}
    # The standalone experiment's checkers do not create old decoder-qualified
    # proof-index lookup rows. Preserve those checks separately and pass no hits.
    match = match_intent_codebase(intent_document=context["intent_document"],
        source_text=context["source_text"], source_identity=context["source_identity"],
        query=context["query"], evidence_rows=[], current_source_snapshot=snapshot)
    zero_control = next(row for row in controls if row["name"] == "zero_weights_inference")
    arguments = {"match_result": match, "source_records": records["sources"],
        "source_context": source_context, "learner": learner,
        "training_receipt": assets["receipt"], "checkpoint": assets["checkpoint"],
        "features": assets["features"], "learned_index": assets["index"],
        "fixed_candidates": {"lexical": loaded["retrieval"], "kg": records["kg"]},
        "model_controls": _nomination_model_controls(learner=learner,
            training_receipt=assets["receipt"], learned_index=assets["index"], zero_control=zero_control)}
    receipt = build_terminal_intent_training_join(**arguments)
    validate_terminal_intent_training_join(receipt, **arguments)
    return receipt, arguments


def _retain_intent_join_failure(output, error, *, phase):
    try:
        _write(output / "intent-join-failure.json", {"schema": "terminal-codebase-intent-join-failure@1",
            "status": "failed", "phase": phase, "error_type": type(error).__name__,
            "message": str(error), "successful_completion_claimed": False,
            "automatic_instruction_interpretation_claimed": False})
    except Exception as retention_error:
        _progress("intent_join_failure_retention_failed", error_type=type(retention_error).__name__)


def _validate_recovered_intent_nomination(recovered, expected_source_context=None):
    """Replay a detached envelope against separately reconstructed producer rows."""
    from .terminal_codebase_intent_training_join import validate_terminal_intent_training_join

    rows = recovered.get(INTENT_NOMINATION_FAMILY)
    if type(rows) is not list or len(rows) != 1:
        raise ValueError("one complete recovered intent nomination envelope required")
    envelope = rows[0]
    if (type(envelope) is not dict or set(envelope) != {"schema", "receipt", "replay_inputs"}
            or envelope["schema"] != INTENT_NOMINATION_METADATA_SCHEMA):
        raise ValueError("exact recovered nomination metadata envelope required")
    arguments = envelope["replay_inputs"]
    training = recovered.get("training")
    if type(training) is not list or len(training) != 1 or arguments["source_records"] != recovered["sources"]:
        raise ValueError("nomination differs from independently reconstructed source/training inventories")
    for name in ("learner", "receipt", "checkpoint"):
        key = "training_receipt" if name == "receipt" else name
        if arguments[key] != training[0][name]:
            raise ValueError("nomination differs from reconstructed native " + name)
    if (arguments["features"]["rows"] != recovered["features"]
            or arguments["features"]["unsupported"] != recovered["feature_frontiers"]
            or arguments["fixed_candidates"]["kg"] != recovered["kg"]):
        raise ValueError("nomination differs from reconstructed feature/frontier/KG records")
    rank_fields = ("row_id", "path", "symbol", "line", "reconstruction_error", "latent")
    raw_ranks = [{key: row[key] for key in rank_fields} for row in recovered["vectors"]]
    if (raw_ranks != arguments["learned_index"]["ranks"]
            or any(row["checkpoint_sha256"] != arguments["learner"]["checkpoint_sha256"]
                   for row in recovered["vectors"])):
        raise ValueError("nomination differs from reconstructed trained vector inventory")
    controls = recovered["experiment_training_controls"]
    zero = [row for row in controls if row["name"] == "zero_weights_inference"]
    if len(zero) != 1 or arguments["model_controls"] != _nomination_model_controls(
            learner=arguments["learner"], training_receipt=arguments["training_receipt"],
            learned_index=arguments["learned_index"], zero_control=zero[0]):
        raise ValueError("nomination controls differ from reconstructed actual inference records")
    lexical = arguments["fixed_candidates"]["lexical"]
    plans = recovered.get("planning")
    if type(plans) is not list or len(plans) != 1:
        raise ValueError("one reconstructed planning producer required")
    retrieval = plans[0]["descriptor"]["retrieval"]
    source_hashes = {row["path"]: row["source_sha256"] for row in recovered["sources"]}
    if (lexical["status"] != "current" or lexical["stale_paths"] != []
            or lexical["query_text"] != arguments["match_result"]["intent_source"]["text"]
            or lexical["source_sha256"] != {"bottle.py": source_hashes["bottle.py"]}
            or any(lexical[name] != retrieval[name] for name in ("index_id", "query_id", "result_id"))):
        raise ValueError("nomination lexical context differs from reconstructed source/query/index bindings")
    actual_vectors = {row["row_id"]: row for row in recovered["retrieval_vectors"]}
    if any(row["row_id"] not in actual_vectors for row in lexical["hits"]):
        raise ValueError("nomination lexical hit is absent from reconstructed vector inventory")
    if expected_source_context is not None and arguments["source_context"] != expected_source_context:
        raise ValueError("nomination differs from independently expected source context")
    return validate_terminal_intent_training_join(envelope["receipt"], **arguments)


def run_experiment(*, source: Path, instruction: Path, output: Path,
                   intent_matching_context=None, repeat_training_controls=True):
    from .terminal_codebase_supervisor_fixture import (
        prepare_terminal_codebase_fixture, extract_terminal_codebase_metadata_records,
        bound_terminal_codebase_metadata_records, reconstruct_terminal_codebase_metadata_records,
    )
    from .terminal_codebase_logic_qualification import qualify_terminal_codebase_logic
    from .codebase_ir_metadata import hydrate_codebase_ir_metadata, validate_codebase_ir_metadata

    if type(repeat_training_controls) is not bool:
        raise ValueError("repeat_training_controls must be an explicit boolean")
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
    producer_functions = [prepare_terminal_codebase_fixture,
        qualify_terminal_codebase_logic, hydrate_codebase_ir_metadata]
    if intent_matching_context is not None:
        from .terminal_codebase_intent_training_join import (
            build_terminal_intent_training_join, validate_terminal_intent_training_join,
        )
        from ipfs_accelerate_py.agent_supervisor.planning.intent_codebase_matching import match_intent_codebase
        producer_functions.extend([build_terminal_intent_training_join,
            validate_terminal_intent_training_join, match_intent_codebase])
    execution_sources = _execution_sources(producer_functions)
    _write(output / "experiment-policy.json", {
        "schema": SCHEMA, "source_sha256": PUBLIC_BOTTLE_SHA256,
        "training_scope": "transductive_public_source",
        "tail_window": TAIL_WINDOW, "tail_absolute_tolerance": TAIL_ABSOLUTE_TOLERANCE,
        "control_seeds": [2718, 31415] if repeat_training_controls else [],
        "control_epochs": 32 if repeat_training_controls else None,
        "repeat_training_controls": repeat_training_controls,
        "explicit_intent_matching_context_selected": intent_matching_context is not None,
        "provider_calls_permitted": 0, "official_verifier_inputs_permitted": False,
        "worker_dispatch": False, "production_catalog_changes": False,
        "asymptotic_optimizer_convergence_claim": False,
        "execution_sources": execution_sources,
    })
    matching_context = None
    if intent_matching_context is not None:
        try:
            matching_context = _load_intent_matching_context(intent_matching_context,
                instruction_bytes=instruction.read_bytes())
            _write(output / "intent-matching-context.json", matching_context)
        except Exception as error:
            _retain_intent_join_failure(output, error, phase="explicit_context_acquisition")
            raise
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
    controls = _training_controls(fixture, output / "training-controls",
        repeat=repeat_training_controls, retain_inference=matching_context is not None)
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
    nomination = nomination_arguments = None
    if matching_context is not None:
        try:
            nomination, nomination_arguments = _prepare_intent_training_nomination(
                fixture=fixture, records=records, context=matching_context, logic=logic,
                runtime_preflight=runtime_preflight, controls=controls)
            records[INTENT_NOMINATION_FAMILY] = [{"schema": INTENT_NOMINATION_METADATA_SCHEMA,
                "receipt": nomination, "replay_inputs": nomination_arguments}]
            _write(output / "intent-training-nomination.json", nomination)
        except Exception as error:
            _retain_intent_join_failure(output, error, phase="owner_verified_training_match_join")
            raise
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
    if nomination is not None:
        _validate_recovered_intent_nomination(recovered)
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
        "same_seed_checkpoint_and_objective_replay": "passed" if repeat_training_controls else "not_exercised_explicit_opt_out",
        "independent_seed_loss_reduction": "passed" if repeat_training_controls else "not_exercised_explicit_opt_out",
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
    if nomination is not None:
        result["intent_codebase_training_nomination"] = nomination
        result["qualification_outcomes"]["intent_training_join"] = "advisory_only_complete_metadata_reconstruction"
        result["qualification_outcomes"]["automatic_full_instruction_interpretation"] = "not_qualified"
    _write(output / "result.json", result)
    _progress("experiment_complete", result=str(output / "result.json"), elapsed=result["elapsed_seconds"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--instruction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--intent-matching-context", type=Path,
        help="Explicit native IntentIR and reviewed matching focus bound to the entire public instruction")
    parser.add_argument("--skip-repeat-training-controls", action="store_true",
        help="Retain one real repository fit and zero-weight inference; skip the three repeated fitting controls")
    args = parser.parse_args()
    existed_before = args.output.exists()
    try:
        run_experiment(source=args.source, instruction=args.instruction, output=args.output,
            intent_matching_context=args.intent_matching_context,
            repeat_training_controls=not args.skip_repeat_training_controls)
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
