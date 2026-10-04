"""Genuine bounded native qualification of finite CodebaseIR inventory scans.

Setup fits a private two-epoch root and one-epoch same-head child. Every scan
then uses that explicit child without fitting or selecting a model head. This
fixture is a structural CPU/float64 qualification, not a production benchmark,
proof, CUDA qualification, source execution attestation or parser guarantee.
"""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from dataclasses import asdict, replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

SCHEMA = "codebase-inventory-scan-native-qualification@1"
TABLES = ("meta", "operations", "variants", "versions", "heads", "runs", "events", "outbox")
PRODUCER_MODULES = (
    "ipfs_datasets_py.logic.software_contracts.codebase_inventory_scan",
    "ipfs_datasets_py.logic.software_contracts.codebase_inventory_targets",
    "ipfs_datasets_py.optimizers.logic_theorem_optimizer.codebase_inventory_feature_worker",
    "ipfs_datasets_py.logic.software_contracts.codebase_source_training",
    "ipfs_datasets_py.logic.software_contracts.codebase_ir_targets",
    "ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_runtime_registry",
    "ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_projection_features",
    "ipfs_datasets_py.optimizers.logic_theorem_optimizer.codebase_feature_worker",
    "ipfs_datasets_py.optimizers.logic_theorem_optimizer.modal_autoencoder_cuda",
    "ipfs_datasets_py.logic.software_contracts.codebase_ir",
    "ipfs_datasets_py.logic.software_contracts.cache",
    "ipfs_datasets_py.logic.software_contracts.content",
    "ipfs_datasets_py.logic.software_contracts.duckdb_ast_store",
    "ipfs_datasets_py.duckdb_control.codebase_catalog",
    "ipfs_datasets_py.duckdb_control.autoencoder_registry",
    "ipfs_datasets_py.logic.software_contracts.codebase_resources",
    "ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler",
    "ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety",
    "ipfs_datasets_py.logic.backends.process",
)


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_wire(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _progress(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".progress-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(_wire(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _files(root):
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AssertionError("fixture evidence directory contains a symlink")
        if path.is_file():
            raw = path.read_bytes()
            result.append({"path": path.relative_to(root).as_posix(), "bytes": len(raw),
                           "sha256": hashlib.sha256(raw).hexdigest()})
    return result


def _registry(registry):
    with registry._transaction() as connection:
        return {name: connection.execute("SELECT * FROM autoencoder_control." + name + " ORDER BY ALL").fetchall()
                for name in TABLES}


def _owners(index, registry):
    return {"registry": _registry(registry), "source_head": index.current("qualification:inventory").to_dict(),
            "source_artifacts": _files(index.artifacts.root),
            "model_artifacts": _files(registry.artifact_root)}


def _producer_inputs(output):
    selected = [(name, Path(importlib.import_module(name).__file__).resolve()) for name in PRODUCER_MODULES]
    datasets = selected[0][1].parents[3]
    selected.extend((("qualification_harness", Path(__file__).resolve()),
                     ("qualification_unit_controls", datasets / "tests/unit/logic/software_contracts/test_codebase_inventory_scan.py")))
    records = []
    copies = output / "producers"
    copies.mkdir()
    for name, path in selected:
        raw = path.read_bytes()
        copy_path = copies / (name + ".py")
        with copy_path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        records.append({"name": name, "path": str(path), "copy": copy_path.relative_to(output).as_posix(),
                        "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()})
    value = {"schema": "codebase-inventory-qualification-selected-inputs@1", "files": records,
             "capture": "sequential_selected_source_copies", "execution_attestation": False,
             "scope": "listed_local_files_only_not_transitive_import_or_dependency_closure"}
    _write(output / "generation-inputs.json", value)
    return value


def _verify_producer_inputs(output, selected):
    for row in selected["files"]:
        actual = Path(row["path"]).read_bytes()
        copy = (output / row["copy"]).read_bytes()
        if actual != copy or len(actual) != row["bytes"] or hashlib.sha256(actual).hexdigest() != row["sha256"]:
            raise AssertionError("selected fixture input changed: " + row["name"])
    return {"selected_file_count": len(selected["files"]), "current_and_retained_copies_unchanged": True,
            "execution_attestation": False}


def _assert_clean(scheduler):
    snapshot = scheduler.snapshot()
    if snapshot["active_lease_count"] or snapshot["waiting_request_count"]:
        raise AssertionError("native resource reservation was not fully released")
    return {"active_lease_count": 0, "waiting_request_count": 0}


@contextmanager
def _patch(module, name, replacement):
    original = getattr(module, name)
    setattr(module, name, replacement)
    try:
        yield original
    finally:
        setattr(module, name, original)


@contextmanager
def _restore_bytes(path):
    original = path.read_bytes()
    try:
        yield original
    finally:
        path.write_bytes(original)


def _native_fixture(root, scheduler):
    import duckdb
    from ipfs_datasets_py.duckdb_control.autoencoder_registry import AutoencoderRegistry
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog
    from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex, CodebaseScanLimits
    from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
    from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor

    repository = root / "repository"
    repository.mkdir()
    for index in range(20):
        (repository / f"source{index:02d}.py").write_text(
            f"def step(n: int) -> int:\n    return n + {index + 1}\n")
    for name, value in (("tune", 101), ("canary", 103)):
        (repository / (name + ".py")).write_text(f"def step(n: int) -> int:\n    return n + {value}\n")
    (repository / "unsupported.py").write_text("def step(n: int) -> int:\n    return external(n)\n")
    (repository / "malformed.py").write_text("def malformed(:\n")
    (repository / "non_utf8.py").write_bytes(b"\xff\xfe\x00invalid\n")
    (repository / "README.md").write_text("Bounded structural fixture; repository source is never executed.\n")
    (repository / "oversized.dat").write_bytes(b"z" * (64 * 1024 + 1))
    (repository / "source_alias.py").symlink_to("source00.py")
    (repository / "excluded").mkdir()
    (repository / "excluded" / "never_admitted.py").write_text("raise RuntimeError('excluded source')\n")
    environment = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
    for arguments in (["init", "--quiet"], ["config", "user.name", "Inventory qualification"],
                      ["config", "user.email", "inventory-qualification@example.invalid"],
                      ["add", "."], ["commit", "--quiet", "--no-verify", "-m", "Fixed native inventory"]):
        subprocess.run(["git", *arguments], cwd=repository, env=environment, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    connection = duckdb.connect(str(root / "source.duckdb"), config={"threads": 1, "memory_limit": "128MB"})
    store = DuckDBASTStore(connection=connection)
    artifacts = ImmutableCAS(root / "source-artifacts")
    index = RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store), artifacts=artifacts,
                                   catalog=CodebaseCatalog(store, artifacts))
    try:
        head = index.prepare_current(repository, repository_id="qualification:inventory", operation_id="initial",
            expected_head=None, limits=CodebaseScanLimits(64, 64 * 1024), exclusions=("excluded",),
            scheduler=scheduler, memory_mb=1024, admission_timeout_seconds=30, timeout_seconds=120).head
        registry = AutoencoderRegistry(root / "model.duckdb", root / "model-artifacts")
    except BaseException:
        connection.close()
        raise
    return index, repository, head, registry, connection


def _same_semantics(optimized, reference):
    for key in ("schema", "profile", "codec", "head", "membership", "authority"):
        if optimized[key] != reference[key]:
            raise AssertionError("optimized/reference differs: " + key)
    model_left, model_right = dict(optimized["model"]), dict(reference["model"])
    if model_left.pop("optimized") is not True or model_right.pop("optimized") is not False or model_left != model_right:
        raise AssertionError("optimized/reference exact model differs")
    if len(optimized["entries"]) != len(reference["entries"]):
        raise AssertionError("optimized/reference inventory length differs")
    maximum = 0.0
    for left, right in zip(optimized["entries"], reference["entries"]):
        before, after = dict(left), dict(right)
        numeric_left, numeric_right = before.pop("inference"), after.pop("inference")
        if before != after:
            raise AssertionError("optimized/reference source disposition or identity differs")
        if numeric_left is None or numeric_right is None:
            if numeric_left != numeric_right:
                raise AssertionError("optimized/reference numerical row presence differs")
            continue
        if set(numeric_left) != set(numeric_right) or numeric_left["source_digest"] != numeric_right["source_digest"]:
            raise AssertionError("optimized/reference numerical source differs")
        if set(numeric_left["reconstructed_projection_features"]) != set(numeric_right["reconstructed_projection_features"]):
            raise AssertionError("optimized/reference projection inventory differs")
        vectors_left = [numeric_left["latent"], *numeric_left["reconstructed_projection_features"].values()]
        vectors_right = [numeric_right["latent"], *numeric_right["reconstructed_projection_features"].values()]
        for values_left, values_right in zip(vectors_left, vectors_right):
            if len(values_left) != len(values_right):
                raise AssertionError("optimized/reference feature width differs")
            for value_left, value_right in zip(values_left, values_right):
                maximum = max(maximum, abs(value_left - value_right))
    if maximum > 1e-12:
        raise AssertionError("optimized/reference exceeds declared CPU float64 tolerance")
    return {"exact_source_and_structural_identity": True, "maximum_absolute_numerical_difference": maximum,
            "absolute_tolerance": 1e-12}


def _children(pid):
    try:
        values = (Path("/proc") / str(pid) / "task" / str(pid) / "children").read_text().split()
        return [int(value) for value in values]
    except (OSError, ValueError):
        return []


def run(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    phases, training_attempts, controls = [], [], []
    report = {"schema": SCHEMA, "qualified": False, "setup_training_attempts": training_attempts,
              "phases": phases, "controls": controls,
              "scope": "fixed_transductive_structural_cpu_float64_inventory",
              "proof_authority": False, "source_execution_attested": False,
              "production_default_activated": False, "cuda_qualified": False,
              "gradient_synchronization_qualified": False}
    scheduler = registry = connection = None

    def progress():
        report["elapsed_seconds_so_far"] = time.monotonic() - began
        _progress(output / "progress.json", report)

    def phase(name, operation):
        row = {"name": name, "status": "running"}
        phases.append(row)
        progress()
        print("Inventory qualification: " + name, file=sys.stderr, flush=True)
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
        from ipfs_datasets_py.logic.software_contracts import codebase_source_training as training
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
        fixture = phase("publish_complete_fixed_inventory", lambda: _native_fixture(output, scheduler))
        index, repository, head, registry, connection = fixture
        selections = [training.CodebaseTrainingSelection(path, role) for path, role in
                      (("source00.py", "train"), ("tune.py", "tune"), ("canary.py", "canary"))]

        def fit(kind, epochs, parent=None):
            row = {"kind": kind, "requested_epochs": epochs, "completed": False,
                   "actual_completed_epochs": None, "unknown_actual_epochs_on_failure": True}
            training_attempts.append(row)
            progress()
            record = training.train_current_codebase_features(index, repository, expected_head=head,
                registry=registry, selections=selections, operation_id=kind, parent_version_id=parent,
                epochs=epochs, learning_rate=.002, seed=1729, scheduler=scheduler,
                admission_timeout_seconds=30, timeout_seconds=120, memory_mb=1024)
            saved = runtimes._read_candidate(registry, registry.get_version(record.to_dict()["version_id"]))
            row.update(completed=True, actual_completed_epochs=epochs, unknown_actual_epochs_on_failure=False,
                       cumulative_model_epochs=saved["state"]["completed_epochs"], version_id=record.to_dict()["version_id"])
            _write(output / (kind + ".json"), record.to_dict())
            progress()
            return record

        root = phase("fit_private_root_two_epochs", lambda: fit("root", 2))
        child = phase("fit_private_same_head_child_one_epoch",
                      lambda: fit("child", 1, root.to_dict()["version_id"]))
        version = child.to_dict()["version_id"]
        report["setup_actual_completed_epochs"] = sum(row["actual_completed_epochs"] for row in training_attempts)
        report["scanned_version_id"] = version
        report["ancestry_version_ids"] = [version, root.to_dict()["version_id"]]
        baseline = _owners(index, registry)
        _write(output / "owners-before-scans.json", baseline)
        child_saved = runtimes._read_candidate(registry, registry.get_version(version))
        numerical_before = {"state_sha256": features.digest(child_saved["state"]),
                            "completed_epochs": child_saved["state"]["completed_epochs"],
                            "adam_steps": [row["step"] for row in child_saved["state"]["adam"]]}
        arguments = dict(expected_head=head, registry=registry, version_id=version, scheduler=scheduler,
                         admission_timeout_seconds=30, timeout_seconds=120, memory_mb=1024)

        def scan(**changes):
            supplied = {**arguments, **changes}
            return scanner.scan_current_codebase_features(index, repository, **supplied)

        def never_fit(*args, **kwargs):
            raise AssertionError("inference attempted fitting")

        with _patch(features, "train_projection_features", never_fit), \
                _patch(runtimes.SourceBoundCodebaseFeatureRuntime, "train", never_fit):
            optimized = phase("optimized_default_scan", lambda: scan())
            reference = phase("explicit_opt_out_reference_scan", lambda: scan(optimized=False))
            for label, record in (("optimized", optimized), ("reference", reference)):
                value = record.to_dict()
                raw = _wire(value)
                if record.artifact_cid != cid_for_bytes(raw):
                    raise AssertionError("scan record does not bind canonical raw native JSON")
                _write(output / (label + ".json"), value)
                report[label] = {"record_raw_cid": record.artifact_cid, "record_sha256": hashlib.sha256(raw).hexdigest(),
                                 "counters": value["counters"], "timings": value["timings"],
                                 "worker_receipt": value["worker_receipt"]}
            comparison = _same_semantics(optimized.to_dict(), reference.to_dict())
            entries = optimized.to_dict()["entries"]
            dispositions = {entry["path"]: entry["disposition"] for entry in entries}
            inferred = [entry for entry in entries if entry["disposition"] == "inferred"]
            if len(inferred) != 22 or len(optimized.to_dict()["shards"]) < 2:
                raise AssertionError("fixture did not infer more than 16 rows across multiple bounded shards")
            if not {"opaque", "unindexed", "parse_failed", "unsupported_target"} <= set(dispositions.values()):
                raise AssertionError("fixture lost an explicit non-inferred inventory disposition")
            if any(path.startswith("excluded/") for path in dispositions):
                raise AssertionError("excluded source widened the admitted inventory")
            comparison.update(inferred_rows=len(inferred), inventory_entries=len(entries), dispositions=dispositions,
                              shard_row_counts=[row["row_count"] for row in optimized.to_dict()["shards"]])
            report["comparison"] = comparison
            if _owners(index, registry) != baseline:
                raise AssertionError("positive inference changed its source/model owners")
            report["positive_no_fit_owner_preservation"] = True
            _assert_clean(scheduler)

            def budget_scan(label, limits, expected_inferred=None, *, require_partial=False):
                bounded = scan(limits=limits)
                value = bounded.to_dict()
                if value["membership"] != optimized.to_dict()["membership"] or len(value["entries"]) != len(entries):
                    raise AssertionError("bounded scan changed complete admitted membership")
                stable_fields = ("path", "source_key", "entry_cid", "source_cid", "ast_cid", "parse_status",
                                 "cohort_membership", "authored_contracts_origin", "target_sha256", "source_digest", "coverage")
                for left, right in zip(entries, value["entries"]):
                    if any(left[key] != right[key] for key in stable_fields):
                        raise AssertionError("bounded scan changed exact source/target/feature identity")
                    if left["disposition"] != "inferred" and left != right:
                        raise AssertionError("bounded scan changed non-inferred frontier")
                count = sum(row["disposition"] == "inferred" for row in value["entries"])
                deferred = [row for row in value["entries"] if row["disposition"] == "deferred_budget"]
                if expected_inferred is not None and count != expected_inferred:
                    raise AssertionError("bounded scan inferred an unexpected row count")
                if count + len(deferred) != len(inferred):
                    raise AssertionError("bounded scan lost compatible inventory members")
                if require_partial and not 0 < count < len(inferred):
                    raise AssertionError("cumulative input budget did not produce partial explicit inference")
                if count == 0:
                    if value["worker_receipt"] is not None or value["counters"]["numerical_process_starts"] != 0:
                        raise AssertionError("all-deferred scan launched a numerical worker")
                elif value["worker_receipt"]["input_bytes"] > limits.max_input_bytes:
                    raise AssertionError("bounded inference exceeded its actual serialized input limit")
                if _owners(index, registry) != baseline:
                    raise AssertionError("bounded inference changed source/model owner state")
                _assert_clean(scheduler)
                _write(output / (label + ".json"), value)
                row = {"name": label, "record_raw_cid": bounded.artifact_cid,
                       "inferred_rows": count, "deferred_rows": len(deferred), "inventory_entries": len(value["entries"]),
                       "limits": limits.to_dict(), "worker_receipt": value["worker_receipt"],
                       "counters": value["counters"], "timings": value["timings"],
                       "frontier_reasons": sorted({frontier.get("reason") for entry in deferred for frontier in entry["frontiers"]}),
                       "exact_membership_and_target_identity": True, "owner_state_preserved": True}
                report.setdefault("positive_budget_scans", []).append(row)
                progress()

            phase("bounded_four_row_positive_scan", lambda:
                  budget_scan("bounded-four-rows", replace(scanner.CodebaseInventoryScanLimits(), max_inferred_rows=4), 4))
            phase("bounded_one_byte_all_deferred_positive_scan", lambda:
                  budget_scan("bounded-one-byte", replace(scanner.CodebaseInventoryScanLimits(), max_input_bytes=1), 0))
            partial_bytes = optimized.to_dict()["worker_receipt"]["input_bytes"] - (
                optimized.to_dict()["counters"]["entry_seal"]["compact_target_transport_bytes"] // len(inferred))
            phase("bounded_cumulative_input_partial_positive_scan", lambda:
                  budget_scan("bounded-partial-input", replace(scanner.CodebaseInventoryScanLimits(), max_input_bytes=partial_bytes),
                              require_partial=True))

            def refuse(name, operation, keywords, *, expected_mutation=False, details=None):
                started = time.monotonic()
                before = _owners(index, registry)
                try:
                    operation()
                except Exception as exc:
                    text = str(exc).lower()
                    if not any(word in text for word in keywords):
                        raise AssertionError(name + " failed for an unrelated reason: " + type(exc).__name__ + ": " + str(exc)) from exc
                    row = {"name": name, "refused": True, "error_type": type(exc).__name__, "error": str(exc),
                           "elapsed_seconds": time.monotonic() - started,
                           "resources_after": _assert_clean(scheduler)}
                    if details is not None:
                        row["details"] = details
                    if not expected_mutation and _owners(index, registry) != before:
                        raise AssertionError(name + " changed source/model owners")
                    controls.append(row)
                    progress()
                    return
                raise AssertionError(name + " unexpectedly returned a current inference record")

            manifest = index.load(head.manifest_cid)
            entry = next(item for item in manifest.snapshot.entries if item.path == "source00.py")
            source_path = index.artifacts.path_for(entry.source_cid, source=True)
            unit = next(item for item in manifest.units if item.source_key == entry.source_key)
            ast_path = index.artifacts.path_for(unit.ast_cid)
            for name, path in (("tampered_source_cas", source_path), ("tampered_ast_cas", ast_path)):
                with _restore_bytes(path) as original:
                    path.write_bytes(original + b"\n")
                    phase(name, lambda name=name: refuse(name, scan, ("cid", "canonical", "digest", "mismatch")))
            quarantine = output / "quarantine"
            quarantine.mkdir()
            selected_checkpoint = registry.artifact_path(registry.get_version(version)["artifact"])
            for label, path in (("source_cas", source_path), ("ast_cas", ast_path),
                                ("selected_checkpoint", selected_checkpoint)):
                missing = quarantine / (label + "-" + path.name)
                os.replace(path, missing)
                try:
                    phase("missing_" + label, lambda label=label:
                          refuse("missing_" + label, scan, ("unavailable", "unreadable", "missing", "no such file")))
                finally:
                    os.replace(missing, path)
            for label, version_id in (("selected", version), ("ancestor", root.to_dict()["version_id"])):
                checkpoint = registry.artifact_path(registry.get_version(version_id)["artifact"])
                with _restore_bytes(checkpoint) as original:
                    checkpoint.write_bytes(original + b"\n")
                    phase("tampered_" + label + "_checkpoint", lambda label=label:
                          refuse("tampered_" + label + "_checkpoint", scan, ("artifact", "digest", "canonical", "checkpoint", "mismatch")))

            with registry._transaction() as connection_for_control:
                original_metadata = connection_for_control.execute(
                    "SELECT metadata FROM autoencoder_control.versions WHERE version_id=?", [version]).fetchone()[0]
                connection_for_control.execute("UPDATE autoencoder_control.versions SET metadata=? WHERE version_id=?",
                                               ['{"tampered":true}', version])
            try:
                phase("changed_registry_candidate_metadata", lambda:
                      refuse("changed_registry_candidate_metadata", scan,
                             ("metadata", "provenance", "source-bound", "candidate", "identity")))
            finally:
                with registry._transaction() as connection_for_control:
                    connection_for_control.execute("UPDATE autoencoder_control.versions SET metadata=? WHERE version_id=?",
                                                   [original_metadata, version])

            native_worker = scanner._worker
            live_source = repository / "source00.py"
            with _restore_bytes(live_source):
                def edit_after_worker(*args, **kwargs):
                    result = native_worker(*args, **kwargs)
                    live_source.write_text("def step(n: int) -> int:\n    return n + 999\n")
                    return result
                with _patch(scanner, "_worker", edit_after_worker):
                    phase("source_change_after_native_worker", lambda:
                          refuse("source_change_after_native_worker", scan, ("repository", "source", "snapshot", "stale"),
                                 expected_mutation=True, details={"genuine_numerical_worker_returned": True}))

            variant = child.to_dict()["variant_id"]
            mutated = False
            def mutate_after_worker(*args, **kwargs):
                nonlocal mutated
                result = native_worker(*args, **kwargs)
                with registry._transaction() as connection_for_control:
                    connection_for_control.execute("INSERT INTO autoencoder_control.heads VALUES (?, 'qualification-control', ?, 1)",
                                                   [variant, version])
                mutated = True
                return result
            try:
                with _patch(scanner, "_worker", mutate_after_worker):
                    phase("registry_control_change_after_native_worker", lambda:
                          refuse("registry_control_change_after_native_worker", scan, ("registry", "control", "changed", "mutation"),
                                 expected_mutation=True, details={"genuine_numerical_worker_returned": True}))
            finally:
                if mutated:
                    with registry._transaction() as connection_for_control:
                        connection_for_control.execute("DELETE FROM autoencoder_control.heads WHERE variant_id=? AND branch='qualification-control'", [variant])

            event = threading.Event()
            event.set()
            phase("pre_cancelled_scan", lambda: refuse("pre_cancelled_scan", lambda: scan(cancel_event=event), ("cancel",)))
            event = threading.Event()
            stop = threading.Event()
            child_evidence = {"actual_child_pids": [], "cancellation_triggered_after_child_observed": False}
            worker_path = str(Path(worker.__file__).resolve()).encode()
            def cancel_actual_child():
                deadline = time.monotonic() + 60
                while not stop.wait(.01) and time.monotonic() < deadline:
                    for pid in _children(os.getpid()):
                        try:
                            command = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
                        except OSError:
                            continue
                        if worker_path in command:
                            child_evidence["actual_child_pids"].append(pid)
                            child_evidence["cancellation_triggered_after_child_observed"] = True
                            event.set()
                            return
            thread = threading.Thread(target=cancel_actual_child, name="cancel-observed-inventory-worker", daemon=True)
            thread.start()
            native_runner = scanner.run_bounded_stdin_tool
            def capture_cancel_lifecycle(*args, **kwargs):
                observation = native_runner(*args, **kwargs)
                child_evidence["native_process_observation"] = {
                    "cancelled": observation.cancelled, "timed_out": observation.timed_out,
                    "workspace_cleaned": observation.workspace_cleaned,
                    "returncode": observation.returncode, "termination_reason": observation.termination_reason,
                    "elapsed_ms": observation.elapsed_ms}
                return observation
            try:
                with _patch(scanner, "run_bounded_stdin_tool", capture_cancel_lifecycle):
                    phase("cancel_actual_running_numerical_child", lambda:
                          refuse("cancel_actual_running_numerical_child", lambda: scan(cancel_event=event), ("cancel",), details=child_evidence))
            finally:
                stop.set()
                thread.join(timeout=2)
            if not child_evidence["cancellation_triggered_after_child_observed"]:
                raise AssertionError("running-child cancellation did not observe an actual numerical process")
            observation = child_evidence.get("native_process_observation")
            if (observation is None or observation["cancelled"] is not True
                    or observation["workspace_cleaned"] is not True or observation["timed_out"] is not False):
                raise AssertionError("cancelled child did not complete native cancellation and workspace cleanup")
            if any((Path("/proc") / str(pid)).exists() for pid in child_evidence["actual_child_pids"]):
                raise AssertionError("cancelled numerical child process survived resource release")
            limits = replace(scanner.CodebaseInventoryScanLimits(), max_entries=16)
            phase("bounded_inventory_refusal", lambda:
                  refuse("bounded_inventory_refusal", lambda: scan(limits=limits), ("bound", "entries", "inventory", "limit")))

        ending = _owners(index, registry)
        if ending != baseline:
            raise AssertionError("restored control fixtures or inference changed persistent owner state")
        child_saved = runtimes._read_candidate(registry, registry.get_version(version))
        numerical_after = {"state_sha256": features.digest(child_saved["state"]),
                           "completed_epochs": child_saved["state"]["completed_epochs"],
                           "adam_steps": [row["step"] for row in child_saved["state"]["adam"]]}
        if numerical_before != numerical_after:
            raise AssertionError("scans changed numerical state or Adam history")
        _write(output / "owners-after-scans.json", ending)
        report.update(qualified=True, final_owner_preservation=True, numerical_before=numerical_before,
                      numerical_after=numerical_after, final_resources=_assert_clean(scheduler),
                      genuine_host_resources_after=asdict(collect_proof_host_resources()),
                      no_model_head_selection=True, inference_training_executed=False,
                      selected_inputs_ending=_verify_producer_inputs(output, selected_inputs))
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if registry is not None:
            registry.close()
        if connection is not None:
            connection.close()
        if scheduler is not None:
            report["final_resources"] = _assert_clean(scheduler)
        report["elapsed_seconds"] = time.monotonic() - began
        _write(output / "result.json", report)
        progress()
    return report


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: qualify_codebase_inventory_scan.py FRESH_OUTPUT_DIRECTORY")
    with redirect_stdout(sys.stderr):
        report = run(sys.argv[1])
    print(json.dumps({"qualified": report["qualified"], "elapsed_seconds": report["elapsed_seconds"],
                      "setup_actual_completed_epochs": report["setup_actual_completed_epochs"],
                      "controls": len(report["controls"]), "output": str(Path(sys.argv[1]).resolve())}))


if __name__ == "__main__":
    main()
