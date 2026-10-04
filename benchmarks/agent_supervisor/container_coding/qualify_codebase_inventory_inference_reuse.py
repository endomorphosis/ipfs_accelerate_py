"""Paired native qualification of finite-operation CodebaseIR inference reuse.

The fixed setup fits a private two-epoch root and one-epoch same-head child.
Four balanced measured calls then use that exact child: optimized, reference,
reference, optimized.  Every call retains the complete admitted inventory and
has a 120-second deadline.  This is bounded CPU float64 structural evidence,
not a production throughput, CUDA, formal proof or source execution claim.
"""
from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time

from benchmarks.agent_supervisor.container_coding import qualify_codebase_inventory_scan as base

SCHEMA = "codebase-inventory-inference-reuse-qualification@1"
ORDER = (True, False, False, True)
NEW_MODULES = (
    "ipfs_datasets_py.logic.software_contracts.codebase_inventory_replay",
    "ipfs_datasets_py.logic.software_contracts.codebase_inventory_lineage",
)


def _producer_inputs(output):
    names = tuple(dict.fromkeys((*base.PRODUCER_MODULES, *NEW_MODULES)))
    selected = [(name, Path(importlib.import_module(name).__file__).resolve()) for name in names]
    datasets = selected[0][1].parents[3]
    selected.extend((("reuse_qualification_harness", Path(__file__).resolve()),
                     ("unchanged_fixture_helper", Path(base.__file__).resolve()),
                     *((name + "_unit_controls", datasets / ("tests/unit/logic/software_contracts/test_codebase_inventory_" + name + ".py"))
                       for name in ("scan", "replay", "lineage", "response"))))
    copies = output / "producers"
    copies.mkdir()
    rows = []
    for name, path in selected:
        raw = path.read_bytes()
        copy = copies / (name + ".py")
        with copy.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        rows.append({"name": name, "path": str(path), "copy": copy.relative_to(output).as_posix(),
                     "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    value = {"schema": "codebase-inventory-reuse-selected-inputs@1", "files": rows,
             "capture": "sequential_selected_source_copies", "execution_attestation": False,
             "scope": "listed_local_files_only_not_transitive_import_or_dependency_closure"}
    base._write(output / "generation-inputs.json", value)
    return value


def _summary(values):
    return {"samples": values, "count": len(values), "minimum": min(values),
            "maximum": max(values), "mean": statistics.mean(values), "median": statistics.median(values)}


def _performance(trials):
    result = {"balanced_order": ["optimized" if mode else "reference" for mode in ORDER],
              "samples_per_mode": 2, "scope": "same_fixed_source_model_complete_public_call",
              "statistical_significance_established": False,
              "generalized_production_speedup_claimed": False}
    for optimized in (True, False):
        samples = [row for row in trials if row["optimized"] is optimized]
        label = "optimized" if optimized else "reference"
        result[label] = {"public_call_seconds": _summary([row["public_call_seconds"] for row in samples]),
            "coordinator_stages": {}, "worker_stages": {},
            "actual_counters": [row["counters"] for row in samples]}
        for category, source in (("coordinator_stages", "coordinator"), ("worker_stages", "worker")):
            keys = sorted(set().union(*(row["stage_seconds"][source] for row in samples)))
            result[label][category] = {key: _summary([row["stage_seconds"][source][key] for row in samples])
                for key in keys if all(key in row["stage_seconds"][source] for row in samples)}
    result["observed_reference_over_optimized_median_public_call"] = (
        result["reference"]["public_call_seconds"]["median"] /
        result["optimized"]["public_call_seconds"]["median"])
    return result


def run(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    phases, training_attempts, trials, controls = [], [], [], []
    report = {"schema": SCHEMA, "qualified": False, "phases": phases,
              "setup_training_attempts": training_attempts, "measured_trials": trials, "controls": controls,
              "scope": "fixed_transductive_structural_cpu_float64_inventory",
              "proof_authority": False, "source_execution_attested": False,
              "production_default_activated": False, "cuda_qualified": False,
              "gradient_synchronization_qualified": False}
    scheduler = registry = connection = None

    def progress():
        report["elapsed_seconds_so_far"] = time.monotonic() - began
        base._progress(output / "progress.json", report)

    def phase(name, operation):
        row = {"name": name, "status": "running"}
        phases.append(row)
        progress()
        print("Inference reuse qualification: " + name, file=sys.stderr, flush=True)
        started = time.monotonic()
        try:
            result = operation()
            row["status"] = "completed"
            return result
        except BaseException as exc:
            row.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            row["elapsed_seconds"] = time.monotonic() - started
            progress()

    try:
        from ipfs_datasets_py.logic.software_contracts import codebase_inventory_scan as scanner
        from ipfs_datasets_py.logic.software_contracts import codebase_inventory_targets as transport
        from ipfs_datasets_py.logic.software_contracts import codebase_source_training as training
        from ipfs_datasets_py.logic.formalization.autoencoder.domain_targets import DomainTargetEnvelope
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_projection_features as features
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_runtime_registry as runtimes
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer import codebase_inventory_feature_worker as worker
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import collect_proof_host_resources
        from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import GlobalResourceScheduler, ResourceSchedulerConfig
        from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes

        selected_inputs = _producer_inputs(output)
        report["generation_inputs"] = "generation-inputs.json"
        report["genuine_host_resources_before"] = asdict(collect_proof_host_resources())
        scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
            state_path=output / "admission.json", lane_reservations={}, auto_renew_leases=True))
        report["scheduler_configuration"] = scheduler.config.persisted_dict()
        index, repository, head, registry, connection = phase("publish_complete_fixed_inventory",
            lambda: base._native_fixture(output, scheduler))
        selections = [training.CodebaseTrainingSelection(path, role) for path, role in
                      (("source00.py", "train"), ("tune.py", "tune"), ("canary.py", "canary"))]

        def fit(kind, epochs, parent=None):
            row = {"kind": kind, "requested_epochs": epochs, "completed": False,
                   "actual_completed_epochs": None, "unknown_actual_epochs_on_failure": True}
            training_attempts.append(row)
            progress()
            before = 0 if parent is None else runtimes._read_candidate(
                registry, registry.get_version(parent))["state"]["completed_epochs"]
            record = training.train_current_codebase_features(index, repository, expected_head=head,
                registry=registry, selections=selections, operation_id=kind, parent_version_id=parent,
                epochs=epochs, learning_rate=.002, seed=1729, scheduler=scheduler,
                admission_timeout_seconds=30, timeout_seconds=120, memory_mb=1024)
            saved = runtimes._read_candidate(registry, registry.get_version(record.to_dict()["version_id"]))
            actual = saved["state"]["completed_epochs"] - before
            if actual != epochs:
                raise AssertionError("setup actual epoch delta differs from its fixed request")
            row.update(completed=True, actual_completed_epochs=actual, unknown_actual_epochs_on_failure=False,
                       cumulative_model_epochs=saved["state"]["completed_epochs"], version_id=record.to_dict()["version_id"])
            base._write(output / (kind + ".json"), record.to_dict())
            progress()
            return record

        root = phase("fit_private_root_two_epochs", lambda: fit("root", 2))
        child = phase("fit_private_same_head_child_one_epoch", lambda: fit("child", 1, root.to_dict()["version_id"]))
        version = child.to_dict()["version_id"]
        report["setup_actual_completed_epochs"] = sum(row["actual_completed_epochs"] for row in training_attempts)
        report["scanned_version_id"] = version
        report["ancestry_version_ids"] = [version, root.to_dict()["version_id"]]
        baseline = base._owners(index, registry)
        base._write(output / "owners-before-scans.json", baseline)
        saved = runtimes._read_candidate(registry, registry.get_version(version))
        report["latent_width"] = saved["state"]["latent_width"]
        report["frozen_feature_columns"] = len(saved["feature_space"]["columns"])
        numerical_before = {"state_sha256": features.digest(saved["state"]),
                            "completed_epochs": saved["state"]["completed_epochs"],
                            "adam_steps": [row["step"] for row in saved["state"]["adam"]]}
        arguments = dict(expected_head=head, registry=registry, version_id=version, scheduler=scheduler,
                         admission_timeout_seconds=30, timeout_seconds=120, memory_mb=1024)

        def scan(**changes):
            return scanner.scan_current_codebase_features(index, repository, **{**arguments, **changes})

        def never_fit(*args, **kwargs):
            raise AssertionError("inference attempted fitting")

        with base._patch(features, "train_projection_features", never_fit), \
                base._patch(runtimes.SourceBoundCodebaseFeatureRuntime, "train", never_fit):
            records = []
            for number, optimized in enumerate(ORDER):
                label = f"trial-{number + 1:02d}-" + ("optimized" if optimized else "reference")
                host_before = asdict(collect_proof_host_resources())

                def measured_scan(optimized=optimized, number=number):
                    # Ledger writes and host sampling stay outside the public
                    # call measurement. First call exercises the public default.
                    started = time.monotonic()
                    record = scan() if number == 0 else scan(optimized=optimized)
                    return record, time.monotonic() - started

                record, elapsed = phase(label, measured_scan)
                value = record.to_dict()
                raw = base._wire(value)
                if record.artifact_cid != cid_for_bytes(raw):
                    raise AssertionError("scan record does not bind canonical raw native JSON")
                if base._owners(index, registry) != baseline:
                    raise AssertionError("measured inference changed its source/model owners")
                ending_pins = base._verify_producer_inputs(output, selected_inputs)
                base._write(output / (label + ".json"), value)
                trial = {"trial": number + 1, "optimized": optimized, "public_default_exercised": number == 0,
                         "record": label + ".json", "record_raw_cid": record.artifact_cid,
                         "record_sha256": hashlib.sha256(raw).hexdigest(), "public_call_seconds": elapsed,
                         "counters": value["counters"], "timings": value["timings"],
                         "stage_seconds": {"coordinator": value["timings"]["coordinator"],
                             "worker": value["timings"]["worker"]["stage_seconds"]},
                         "worker_receipt": value["worker_receipt"], "owner_state_preserved": True,
                         "source_copies_preserved": ending_pins, "resources_after": base._assert_clean(scheduler),
                         "genuine_host_resources_before": host_before,
                         "genuine_host_resources_after": asdict(collect_proof_host_resources())}
                trials.append(trial)
                records.append(value)
                progress()
            comparisons = [base._same_semantics(records[i], records[j]) for i, j in ((0, 1), (3, 2), (0, 2), (3, 1))]
            # Repeated calls in one mode must also preserve every entry, numerical row and model identity.
            for left, right in ((records[0], records[3]), (records[1], records[2])):
                if any(left[key] != right[key] for key in ("head", "model", "membership", "entries", "authority")):
                    raise AssertionError("repeated mode changed source/model/numerical identity")
            entries = records[0]["entries"]
            inferred = [entry for entry in entries if entry["disposition"] == "inferred"]
            if len(entries) != 28 or len(inferred) != 22 or [row["row_count"] for row in records[0]["shards"]] != [16, 6]:
                raise AssertionError("fixed fixture inventory or finite shard population changed")
            report["comparisons"] = comparisons
            report["performance"] = _performance(trials)
            report["positive_no_fit_owner_preservation"] = True
            progress()

            # Separate probes count actual parent-side calls without replacing
            # native functions. Their tracer overhead is excluded from the four
            # balanced performance samples, and never reaches the child process.
            from ipfs_datasets_py.logic.software_contracts import codebase_ir_targets as native_targets
            watched = {
                "native_target_validator": native_targets.validate_codebase_targets,
                "native_target_lowering": native_targets._prepare_bound,
                "native_source_binding_replay": native_targets.source_binding_from_target,
                "legacy_lineage_entry": training._lineage,
                "legacy_candidate_reader": training._read_candidate,
                "runtime_version_loader": runtimes.load_version,
                "runtime_candidate_reader": runtimes._read_candidate,
                "registry_version_reads": type(registry).get_version,
                "registry_artifact_verification": type(registry).verify_artifact,
                "manifest_reads": type(index).load,
                "captured_ast_reads": type(index).load_ast_artifact,
                "publication_receipt_reads": type(index.catalog)._read_receipt,
            }
            code_names = {function.__code__: name for name, function in watched.items()}
            profiled_records = {}
            for optimized in (False, True):
                counts = dict.fromkeys(watched, 0)
                previous = sys.getprofile()
                if previous is not None:
                    raise AssertionError("qualification parent already has a profiler")

                def profile_event(frame, event, argument):
                    if event == "call":
                        name = code_names.get(frame.f_code)
                        if name is not None:
                            counts[name] += 1

                def profiled_scan(optimized=optimized):
                    started = time.monotonic()
                    sys.setprofile(profile_event)
                    try:
                        return scan(optimized=optimized), time.monotonic() - started
                    finally:
                        sys.setprofile(previous)

                label = "profiled-" + ("optimized" if optimized else "reference")
                record, elapsed = phase(label, profiled_scan)
                value = record.to_dict()
                if base._owners(index, registry) != baseline:
                    raise AssertionError("profile probe changed source/model owners")
                base._write(output / (label + ".json"), value)
                profiled_records[optimized] = value
                report.setdefault("profile_probes", []).append({"optimized": optimized,
                    "record": label + ".json", "record_raw_cid": record.artifact_cid,
                    "actual_parent_function_calls": counts, "scope": "main_parent_thread_only_not_child_process",
                    "profile_probe_seconds_including_tracer": elapsed, "excluded_from_performance_samples": True,
                    "worker_counters": value["counters"]["worker"], "owner_state_preserved": True,
                    "resources_after": base._assert_clean(scheduler)})
                progress()
            report["profile_probe_parity"] = base._same_semantics(profiled_records[True], profiled_records[False])

            def refuse(name, operation, keywords, *, details=None, expected_owner_mutation=False):
                started = time.monotonic()
                before = base._owners(index, registry)
                try:
                    operation()
                except Exception as exc:
                    if not any(word in str(exc).lower() for word in keywords):
                        raise AssertionError(name + " failed for an unrelated reason: " + type(exc).__name__ + ": " + str(exc)) from exc
                    if not expected_owner_mutation and base._owners(index, registry) != before:
                        raise AssertionError(name + " changed source/model owners")
                    row = {"name": name, "refused": True, "error_type": type(exc).__name__, "error": str(exc),
                           "elapsed_seconds": time.monotonic() - started,
                           "resources_after": base._assert_clean(scheduler)}
                    if details is not None:
                        row["details"] = details
                    controls.append(row)
                    progress()
                    return
                raise AssertionError(name + " unexpectedly returned a current inference record")

            native_worker = scanner._worker

            def payload_control(name, mutate, keywords):
                evidence = {"genuine_isolated_worker_attempted": False, "payload_mutated_after_sender_preparation": True}
                native_runner = scanner.run_bounded_stdin_tool

                def capture_payload_lifecycle(*args, **kwargs):
                    observation = native_runner(*args, **kwargs)
                    evidence["native_process_observation"] = {"returncode": observation.returncode,
                        "workspace_cleaned": observation.workspace_cleaned, "cancelled": observation.cancelled,
                        "timed_out": observation.timed_out, "elapsed_ms": observation.elapsed_ms,
                        "termination_reason": observation.termination_reason}
                    return observation

                def changed(payload, **kwargs):
                    copied = json.loads(base._wire(payload))
                    mutate(copied)
                    evidence["genuine_isolated_worker_attempted"] = True
                    return native_worker(copied, **kwargs)

                with base._patch(scanner, "_worker", changed), \
                        base._patch(scanner, "run_bounded_stdin_tool", capture_payload_lifecycle):
                    phase(name, lambda: refuse(name, scan, keywords, details=evidence))
                observation = evidence.get("native_process_observation")
                if (not evidence["genuine_isolated_worker_attempted"] or observation is None
                        or observation["returncode"] == 0 or observation["workspace_cleaned"] is not True
                        or observation["cancelled"] or observation["timed_out"]):
                    raise AssertionError(name + " did not reach the actual receiving worker")

            def shared_rehashed(payload):
                shared = payload["shared_inventory"]
                shared["head"]["repository_id"] += ":tampered"
                shared["shared_sha256"] = transport._digest(
                    {key: value for key, value in shared.items() if key != "shared_sha256"},
                    16 * 1024 * 1024, "qualification shared inventory")
                for shard in payload["shards"]:
                    for compact in shard["targets"]:
                        compact["shared_sha256"] = shared["shared_sha256"]
                        compact["compact_sha256"] = transport._digest(
                            {key: value for key, value in compact.items() if key != "compact_sha256"},
                            4 * 1024 * 1024, "qualification compact target")

            payload_control("rehashed_shared_publication_binding", shared_rehashed, ("publication", "binding", "repository"))

            def rehash_target(payload, compact):
                restored = json.loads(base._wire(compact["detached_target"]))
                details = restored["validation"][0]["details"]
                details["manifest"] = payload["shared_inventory"]["manifest"]
                details["publication_receipt"] = payload["shared_inventory"]["publication_receipt"]
                compact["target_sha256"] = DomainTargetEnvelope.from_dict(restored).digest
                compact["compact_sha256"] = transport._digest(
                    {key: value for key, value in compact.items() if key != "compact_sha256"},
                    4 * 1024 * 1024, "qualification compact target")

            def projection_rehashed(payload):
                compact = payload["shards"][0]["targets"][0]
                compact["detached_target"]["projections"][0]["expression"]["schema"] = "qualification:tampered-projection"
                rehash_target(payload, compact)

            payload_control("rehashed_native_projection", projection_rehashed, ("native replay differs", "target replay differs"))

            def authority_rehashed(payload):
                compact = payload["shards"][0]["targets"][0]
                compact["detached_target"]["validation"][0]["details"]["proof_authority"] = True
                rehash_target(payload, compact)

            payload_control("rehashed_native_target_authority", authority_rehashed, ("native replay differs", "target replay differs"))

            def source_rehashed(payload):
                compact = payload["shards"][0]["targets"][0]
                details = compact["detached_target"]["validation"][0]["details"]
                details["source_bytes_hex"] += "0a"
                rehash_target(payload, compact)

            payload_control("rehashed_captured_source", source_rehashed, ("source bytes", "captured", "binding", "byte count"))

            manifest = index.load(head.manifest_cid)
            entry = next(item for item in manifest.snapshot.entries if item.path == "source00.py")
            unit = next(item for item in manifest.units if item.source_key == entry.source_key)
            source_path = index.artifacts.path_for(entry.source_cid, source=True)
            ast_path = index.artifacts.path_for(unit.ast_cid)
            for label, path in (("captured_source", source_path), ("captured_ast", ast_path)):
                with base._restore_bytes(path) as original:
                    path.write_bytes(original + b"\n")
                    phase("tampered_" + label, lambda label=label:
                        refuse("tampered_" + label, scan, ("cid", "canonical", "digest", "mismatch")))
            for label, version_id in (("selected", version), ("ancestor", root.to_dict()["version_id"])):
                path = registry.artifact_path(registry.get_version(version_id)["artifact"])
                with base._restore_bytes(path) as original:
                    path.write_bytes(original + b"\n")
                    phase("tampered_" + label + "_checkpoint", lambda label=label:
                        refuse("tampered_" + label + "_checkpoint", scan, ("artifact", "digest", "canonical", "checkpoint", "mismatch")))

            live_source = repository / "source00.py"
            with base._restore_bytes(live_source):
                source_fault = {"genuine_numerical_worker_returned": False}

                def edit_after_worker(*args, **kwargs):
                    result = native_worker(*args, **kwargs)
                    live_source.write_text("def step(n: int) -> int:\n    return n + 999\n")
                    source_fault["genuine_numerical_worker_returned"] = True
                    return result
                with base._patch(scanner, "_worker", edit_after_worker):
                    phase("source_change_after_native_worker", lambda:
                        refuse("source_change_after_native_worker", scan, ("repository", "source", "snapshot", "stale"),
                               details=source_fault))
                if not source_fault["genuine_numerical_worker_returned"]:
                    raise AssertionError("late source control did not complete a genuine worker")

            variant = child.to_dict()["variant_id"]
            mutated = False
            registry_fault = {"genuine_numerical_worker_returned": False,
                              "deliberate_owner_fault_restored_after_control": True}

            def mutate_registry_after_worker(*args, **kwargs):
                nonlocal mutated
                result = native_worker(*args, **kwargs)
                with registry._transaction() as control_connection:
                    control_connection.execute("INSERT INTO autoencoder_control.heads VALUES (?, 'reuse-qualification-control', ?, 1)",
                                               [variant, version])
                mutated = True
                registry_fault["genuine_numerical_worker_returned"] = True
                return result

            try:
                with base._patch(scanner, "_worker", mutate_registry_after_worker):
                    # This deliberate raw owner fault is restored outside the
                    # refusal helper; checking its exact final restoration is
                    # separate from checking normal inference preservation.
                    phase("registry_change_after_native_worker", lambda:
                        refuse("registry_change_after_native_worker", scan, ("registry", "control", "changed", "mutation"),
                               expected_owner_mutation=True,
                               details=registry_fault))
                if not registry_fault["genuine_numerical_worker_returned"]:
                    raise AssertionError("late registry control did not complete a genuine worker")
            finally:
                if mutated:
                    with registry._transaction() as control_connection:
                        control_connection.execute("DELETE FROM autoencoder_control.heads WHERE variant_id=? AND branch='reuse-qualification-control'", [variant])
            if base._owners(index, registry) != baseline:
                raise AssertionError("registry fault was not exactly restored")

            selected_checkpoint = registry.artifact_path(registry.get_version(version)["artifact"])
            with base._restore_bytes(selected_checkpoint) as original:
                artifact_fault = {"genuine_numerical_worker_returned": False,
                                  "deliberate_artifact_fault_restored_after_control": True}

                def mutate_artifact_after_worker(*args, **kwargs):
                    result = native_worker(*args, **kwargs)
                    selected_checkpoint.write_bytes(original + b"\n")
                    artifact_fault["genuine_numerical_worker_returned"] = True
                    return result

                with base._patch(scanner, "_worker", mutate_artifact_after_worker):
                    phase("selected_checkpoint_change_after_native_worker", lambda:
                        refuse("selected_checkpoint_change_after_native_worker", scan,
                            ("artifact", "checkpoint", "digest", "mismatch"),
                            expected_owner_mutation=True, details=artifact_fault))
                if not artifact_fault["genuine_numerical_worker_returned"]:
                    raise AssertionError("late checkpoint control did not complete a genuine worker")
            if base._owners(index, registry) != baseline:
                raise AssertionError("late artifact fault was not exactly restored")

            # The closing model check must still run after the last genuine
            # live-source observation, rather than relying on an earlier hash.
            with base._restore_bytes(selected_checkpoint) as original:
                closing_fault = {"successful_native_source_observations": 0,
                                 "artifact_changed_after_final_source_observation": False,
                                 "deliberate_artifact_fault_restored_after_control": True}
                native_observe = scanner._observe

                def mutate_artifact_after_final_source(*args, **kwargs):
                    observation = native_observe(*args, **kwargs)
                    closing_fault["successful_native_source_observations"] += 1
                    if closing_fault["successful_native_source_observations"] == 2:
                        selected_checkpoint.write_bytes(original + b"\n")
                        closing_fault["artifact_changed_after_final_source_observation"] = True
                    return observation

                with base._patch(scanner, "_observe", mutate_artifact_after_final_source):
                    phase("selected_checkpoint_change_after_final_source_observation", lambda:
                        refuse("selected_checkpoint_change_after_final_source_observation", scan,
                            ("artifact", "checkpoint", "digest", "mismatch"),
                            expected_owner_mutation=True, details=closing_fault))
                if (closing_fault["successful_native_source_observations"] != 2
                        or not closing_fault["artifact_changed_after_final_source_observation"]):
                    raise AssertionError("closing checkpoint control did not complete both genuine source observations")
            if base._owners(index, registry) != baseline:
                raise AssertionError("closing artifact fault was not exactly restored")

            head_fault = {"genuine_numerical_worker_returned": False,
                          "deliberate_catalog_fault_restored_after_control": True}

            def mutate_catalog_after_worker(*args, **kwargs):
                result = native_worker(*args, **kwargs)
                with index.catalog.store._lock:
                    with index.catalog.store._transaction():
                        index.catalog.store._connection.execute(
                            "UPDATE codebase_control.heads SET generation=generation+1 WHERE repository_id=?",
                            [head.repository_id])
                head_fault["genuine_numerical_worker_returned"] = True
                return result

            try:
                with base._patch(scanner, "_worker", mutate_catalog_after_worker):
                    phase("catalog_head_change_after_native_worker", lambda:
                        refuse("catalog_head_change_after_native_worker", scan,
                            ("head", "catalog", "receipt", "generation", "stale"),
                            expected_owner_mutation=True, details=head_fault))
                if not head_fault["genuine_numerical_worker_returned"]:
                    raise AssertionError("late catalog control did not complete a genuine worker")
            finally:
                if head_fault["genuine_numerical_worker_returned"]:
                    with index.catalog.store._lock:
                        with index.catalog.store._transaction():
                            index.catalog.store._connection.execute(
                                "UPDATE codebase_control.heads SET generation=? WHERE repository_id=?",
                                [head.generation, head.repository_id])
            if base._owners(index, registry) != baseline:
                raise AssertionError("late catalog fault was not exactly restored")

            event = threading.Event()
            event.set()
            phase("pre_cancelled_scan", lambda: refuse("pre_cancelled_scan", lambda: scan(cancel_event=event), ("cancel",)))
            event, stop = threading.Event(), threading.Event()
            evidence = {"actual_child_pids": [], "cancellation_triggered_after_child_observed": False}
            worker_path = str(Path(worker.__file__).resolve()).encode()

            def cancel_observed_child():
                deadline = time.monotonic() + 60
                while not stop.wait(.01) and time.monotonic() < deadline:
                    for pid in base._children(os.getpid()):
                        try:
                            command = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
                        except OSError:
                            continue
                        if worker_path in command:
                            evidence["actual_child_pids"].append(pid)
                            evidence["cancellation_triggered_after_child_observed"] = True
                            event.set()
                            return

            thread = threading.Thread(target=cancel_observed_child, name="cancel-observed-reuse-worker", daemon=True)
            thread.start()
            native_runner = scanner.run_bounded_stdin_tool

            def capture_lifecycle(*args, **kwargs):
                observation = native_runner(*args, **kwargs)
                evidence["native_process_observation"] = {"cancelled": observation.cancelled,
                    "timed_out": observation.timed_out, "workspace_cleaned": observation.workspace_cleaned,
                    "returncode": observation.returncode, "termination_reason": observation.termination_reason,
                    "elapsed_ms": observation.elapsed_ms}
                return observation

            try:
                with base._patch(scanner, "run_bounded_stdin_tool", capture_lifecycle):
                    phase("cancel_actual_running_child", lambda:
                        refuse("cancel_actual_running_child", lambda: scan(cancel_event=event), ("cancel",), details=evidence))
            finally:
                stop.set()
                thread.join(timeout=2)
            observation = evidence.get("native_process_observation")
            if (not evidence["cancellation_triggered_after_child_observed"] or observation is None
                    or observation["cancelled"] is not True or observation["workspace_cleaned"] is not True
                    or observation["timed_out"] is not False):
                raise AssertionError("running-child cancellation did not complete native lifecycle cleanup")
            if any((Path("/proc") / str(pid)).exists() for pid in evidence["actual_child_pids"]):
                raise AssertionError("cancelled numerical child survived resource release")

        ending = base._owners(index, registry)
        if ending != baseline:
            raise AssertionError("restored controls or inference changed persistent source/model owners")
        saved = runtimes._read_candidate(registry, registry.get_version(version))
        numerical_after = {"state_sha256": features.digest(saved["state"]),
                           "completed_epochs": saved["state"]["completed_epochs"],
                           "adam_steps": [row["step"] for row in saved["state"]["adam"]]}
        if numerical_after != numerical_before:
            raise AssertionError("inference changed frozen numerical state or Adam history")
        base._write(output / "owners-after-scans.json", ending)
        report.update(qualified=True, final_owner_preservation=True, numerical_before=numerical_before,
                      numerical_after=numerical_after, final_resources=base._assert_clean(scheduler),
                      genuine_host_resources_after=asdict(collect_proof_host_resources()),
                      no_model_head_selection=True, inference_training_executed=False,
                      selected_inputs_ending=base._verify_producer_inputs(output, selected_inputs))
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if registry is not None:
            registry.close()
        if connection is not None:
            connection.close()
        if scheduler is not None:
            report["final_resources"] = base._assert_clean(scheduler)
        report["elapsed_seconds"] = time.monotonic() - began
        base._write(output / "result.json", report)
        progress()
    return report


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: qualify_codebase_inventory_inference_reuse.py FRESH_OUTPUT_DIRECTORY")
    with redirect_stdout(sys.stderr):
        report = run(sys.argv[1])
    print(json.dumps({"qualified": report["qualified"], "elapsed_seconds": report["elapsed_seconds"],
        "setup_actual_completed_epochs": report["setup_actual_completed_epochs"],
        "measured_calls": len(report["measured_trials"]), "controls": len(report["controls"]),
        "output": str(Path(sys.argv[1]).resolve())}))


if __name__ == "__main__":
    main()
