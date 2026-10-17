"""Real native custody unwind/cancellation fixture; no external process effects.

Root reviews and executes this frozen source separately. It never starts a
process, Docker, provider, model or task. Actual owner records and original
private attempts, together with the exact zero-effect source closure, establish
the narrow known-unstarted stopped classification before consumer release.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys

HERE = Path(__file__).absolute().parent
RUN = Path("/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs/legal-ir-ordinary-custody-unwind-20261017-r01")
ROOT = Path("/home/barberb/lift_coding/external/ipfs_datasets")
ROOTS = (ROOT / "workspace/test-logs", ROOT / "workspace/todo-queues",
         ROOT / "docs/implementation/reports/evidence", Path("/tmp/pytest-of-barberb"))
LEDGER = ROOT / "workspace/test-logs/federal-corpus-audits/owned-daemon-resource-control/disk-reservations.json"
STATE = Path("/tmp/ipfs-datasets-resource-scheduler-1000.json")
POLICY = "f07cb14289fcb6275e4497dcc36be0dcf8bfc3dc87531ff7a9537eb85fdcc661"
DATASETS = Path("/home/barberb/lift_coding/artifacts/legal-conditions-composition-diagnostic-20261015-01/datasets")
NATIVE = "ipfs_datasets_py.optimizers.logic_theorem_optimizer."
NATIVE_NAMES = ("parented_phase_resources", "autoencoder_daemon_resources", "existing_resource_scheduler", "resource_scheduler")
CUSTODY_SOURCE = HERE.parent / "ordinary-custody-hold-v01/ordinary_custody_hold.py"
CUSTODY_MANIFEST = CUSTODY_SOURCE.with_name("SOURCE_MANIFEST.json")
CUSTODY_MANIFEST_SHA = "36742d137ef138fd720ffc9db0521e0107bb7616ef9c95494f785b21ef238ceb"
PROFILE = "legal-ir-ordinary-custody-unwind-r01"
MARKER = b"known-unstarted-no-Docker-provider-process-allocation\n"
NO_AUTHORITY = {name: False for name in ("acceptance_authority", "completion_authority", "dispatch_authority",
    "productive_authority_granted", "actual_provider_execution_qualified", "production_hook_deployed")}


def need(value, reason):
    if not value:
        raise RuntimeError(reason)


def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate_json_key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))


def witness(row):
    return (row.st_dev, row.st_ino, row.st_mode, row.st_uid, row.st_gid,
            row.st_nlink, row.st_size, row.st_mtime_ns, row.st_ctime_ns)


def read_stable(path, maximum=4 * 1024 * 1024):
    path = Path(path)
    need(path.is_absolute() and os.path.abspath(path) == str(path)
         and not any(item.is_symlink() for item in (path, *path.parents)), "canonical_unaliased_file_required")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_uid in {0, os.getuid()}
             and before.st_nlink == 1 and before.st_mode & 0o002 == 0
             and 0 < before.st_size <= maximum, "bounded_owned_regular_file_required")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    need(len(raw) == before.st_size and witness(before) == witness(after) == witness(path.lstat()),
         "file_changed_during_read")
    return raw


def verify_sources(request):
    manifest_raw = read_stable(HERE / "SOURCE_MANIFEST.json")
    need(hashlib.sha256(manifest_raw).hexdigest() == request["source_manifest_sha256"],
         "reviewed_unwind_source_manifest_changed")
    manifest = strict(manifest_raw)
    need(manifest.get("schema") == "legal-ir-ordinary-custody-unwind-source@1", "source_manifest_schema_differs")
    expected = {str(HERE / row["path"]): row["sha256"] for row in manifest["files"]}
    expected.update({row["path"]: row["sha256"] for row in manifest["input_pins"]})
    expected[str(HERE / "SOURCE_MANIFEST.json")] = request["source_manifest_sha256"]
    need(type(request.get("source_sha256")) is dict and request["source_sha256"] == expected,
         "complete_exact_unwind_source_closure_required")
    for spelling, digest in expected.items():
        need(type(digest) is str and re.fullmatch("[0-9a-f]{64}", digest), "exact_source_digest_required")
        raw = read_stable(Path(spelling), 256 * 1024 * 1024)
        need(hashlib.sha256(raw).hexdigest() == digest, "actual_unwind_source_pin_changed")
    need(hashlib.sha256(read_stable(CUSTODY_MANIFEST)).hexdigest() == CUSTODY_MANIFEST_SHA,
         "original_frozen_custody_manifest_changed")
    native_paths = {str(DATASETS / "ipfs_datasets_py/optimizers/logic_theorem_optimizer" / (name + ".py"))
                    for name in NATIVE_NAMES}
    transitive_paths = {str(DATASETS / relative) for relative in (
        "ipfs_datasets_py/__init__.py", "ipfs_datasets_py/router_deps.py", "ipfs_datasets_py/optimizers/__init__.py",
        "ipfs_datasets_py/optimizers/logic_theorem_optimizer/__init__.py",
        "ipfs_datasets_py/optimizers/logic_theorem_optimizer/proof_resource_safety.py",
        "ipfs_datasets_py/optimizers/logic_theorem_optimizer/runtime_telemetry.py")}
    need(native_paths | transitive_paths | {str(CUSTODY_SOURCE), str(CUSTODY_MANIFEST), "/usr/bin/python3.12"} <= set(expected),
         "complete_preimport_native_package_and_sampler_source_pins_required")
    subset = request.get("custody_source_sha256")
    need(type(subset) is dict and set(subset) == native_paths | {str(CUSTODY_SOURCE)}
         and all(expected.get(path) == digest for path, digest in subset.items()),
         "actual_frozen_custody_source_join_required")
    # Necessary source-scope check, not portable evidence or native authority.
    tree = ast.parse(read_stable(Path(__file__).absolute()))
    forbidden_imports = {"subprocess", "socket", "http", "urllib", "requests", "docker", "ctypes", "multiprocessing"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            need(not any(alias.name.split(".")[0] in forbidden_imports for alias in node.names), "zero_effect_import_scope_changed")
        if isinstance(node, ast.ImportFrom):
            need((node.module or "").split(".")[0] not in forbidden_imports, "zero_effect_import_scope_changed")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            need(node.func.attr not in {"Popen", "run", "system", "fork", "forkpty", "posix_spawn", "posix_spawnp",
                 "execv", "execve", "execl", "execlp", "spawnv", "spawnve", "spawnvp", "spawnvpe"},
                 "zero_external_process_effect_scope_changed")


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_marker(directory):
    descriptor = os.open(directory / "known-unstarted.txt", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(MARKER)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(directory)


def durable_closeout(directory, value):
    raw = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    need(len(raw) <= 1024 * 1024, "bounded_private_closeout_required")
    descriptor = os.open(directory / "stopped-closeout.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(directory)


def known_unstarted(owners, identities):
    """Observe original actual attempts and public child history, never JSON input."""
    rows = []
    for owner in owners:
        public = owner.to_dict()
        record = public.get("record")
        need(type(record) is dict and record.get("child") is None and record.get("prior_children", []) == []
             and record.get("external_charges") == {} and public.get("cleanup_error") is None,
             "actual_zero_child_history_and_external_allocation_required")
        directory = owner.directory
        actual = directory.lstat()
        need(stat.S_ISDIR(actual.st_mode) and (actual.st_dev, actual.st_ino) == identities[str(directory)]
             and actual.st_uid == os.getuid() and actual.st_mode & 0o077 == 0,
             "original_private_attempt_identity_changed")
        bound = record.get("attempt_directory")
        need(type(bound) is dict and bound.get("path") == str(directory)
             and (bound.get("device"), bound.get("inode")) == (actual.st_dev, actual.st_ino),
             "actual_native_bound_attempt_identity_differs")
        files = list(directory.iterdir())
        need({path.name for path in files} <= {"known-unstarted.txt"}, "unexpected_private_attempt_allocation")
        if files:
            marker = files[0]
            mode = marker.lstat().st_mode
            need(stat.S_ISREG(mode) and mode & 0o077 == 0 and read_stable(marker) == MARKER,
                 "original_private_marker_changed")
        rows.append({"reservation_id": public["reservation_id"], "status": public["status"],
            "attempt_directory": str(directory), "attempt_device": actual.st_dev, "attempt_inode": actual.st_ino,
            "child": None, "prior_children": [], "private_files": sorted(path.name for path in files)})
    return {"classification": "exact_source_known_unstarted_no_external_process_or_Docker_allocation",
        "actual_original_owner_records_and_attempts": rows,
        "independent_Docker_closeout_exercised": False, "independent_kernel_process_closeout_exercised": False}


def expect_refusal(label, function, kind, message, result):
    try:
        function()
    except kind as error:
        need(str(error) == message, label + ":wrong_refusal:" + str(error))
        result["refusals"].append({"control": label, "type": type(error).__name__, "reason": str(error)})
    else:
        raise RuntimeError(label + ":premature_operation_was_accepted")


def inventory(scheduler, identifiers):
    rows = {row["lease_id"]: row for row in scheduler.active_leases()}
    return {name: {"lease_id": value, "record_present": value in rows,
                  "cancelled": rows.get(value, {}).get("cancelled"),
                  "release_requested": rows.get(value, {}).get("release_requested", False)}
            for name, value in identifiers.items()}


class IntentionalOwnerContextUnwind(Exception):
    pass


def experiment(request, scheduler, phase, owner_type, daemon_error, C, result):
    owners, native_parents, consumers, identities = [], [], [], {}
    hold = None
    try:
        primary = owner_type(LEDGER, roots=ROOTS, attempt_directory=RUN / "primary", parent_lease=phase,
            storage_bytes=32 * 1024 * 1024, memory_mb=512, cpu_slots=1, child_process_slots=8,
            scheduler_state_path=STATE, expected_scheduler_policy_sha256=POLICY,
            timeout_seconds=request["duration_seconds"], ledger_lock_timeout_seconds=None)
        owners.append(primary)
        row = primary.directory.lstat()
        identities[str(primary.directory)] = (row.st_dev, row.st_ino)
        try:
            with primary:
                durable_marker(primary.directory)
                native_parents.append(primary.native_lease)
                watchdog = owner_type(LEDGER, roots=ROOTS, attempt_directory=RUN / "watchdog", parent_lease=phase,
                    storage_bytes=32 * 1024 * 1024, memory_mb=256, cpu_slots=1, child_process_slots=4,
                    scheduler_state_path=STATE, expected_scheduler_policy_sha256=POLICY,
                    timeout_seconds=request["duration_seconds"], ledger_lock_timeout_seconds=None)
                owners.append(watchdog)
                row = watchdog.directory.lstat()
                identities[str(watchdog.directory)] = (row.st_dev, row.st_ino)
                with watchdog:
                    durable_marker(watchdog.directory)
                    native_parents.append(watchdog.native_lease)
                    for native_parent in native_parents:
                        consumers.append(native_parent.acquire_child(cpu_slots=1, memory_mb=1,
                            child_process_slots=0, requires_gpu=False, timeout=30,
                            request_id=PROFILE + ":custody:" + str(len(consumers))))
                    hold = C.bind_ordinary_custody_hold(primary_owner=primary, watchdog_owner=watchdog,
                        primary_consumer=consumers[0], watchdog_consumer=consumers[1],
                        protected_source_sha256=request["custody_source_sha256"])
                    result["observations"]["before_unwind"] = C.observe_ordinary_custody_hold(hold)
                    expect_refusal("forget_before_context_unwind", lambda: C.forget_acknowledged_custody_hold(hold),
                        C.CustodyHoldRefusal, "unacknowledged_native_consumers_still_held", result)
                    for label, owner in zip(("primary", "watchdog"), owners):
                        expect_refusal(label + "_productive_finalize_before_context_unwind",
                            lambda owner=owner: owner.finalize(owner.directory, artifacts_durable=True), daemon_error,
                            "native consumers must finish and release before finalization", result)
                    result["observations"]["known_unstarted_before_unwind"] = known_unstarted(owners, identities)
                    # Deliberately exercise the original exception-driven with
                    # exits, not direct private __exit__ calls or fake owners.
                    raise IntentionalOwnerContextUnwind("expected_original_owner_with_unwind")
        except IntentionalOwnerContextUnwind as error:
            need(str(error) == "expected_original_owner_with_unwind", "unexpected_unwind_signal")
            result["actual_owner_contexts_unwound"] = True
        need(len(owners) == len(native_parents) == len(consumers) == 2 and hold is not None,
             "complete_actual_bound_owner_pair_required")
        public = [owner.to_dict() for owner in owners]
        need(all(row["status"] == "retained" for row in public), "original_owner_unwind_did_not_retain")
        result["observations"]["after_unwind"] = C.observe_ordinary_custody_hold(hold)
        result["capacity_retained_after_unwind"] = result["observations"]["after_unwind"]["capacity_records_retained"]
        need(result["capacity_retained_after_unwind"] and not any(parent.released for parent in native_parents),
             "actual_consumer_barrier_did_not_retain_parent_capacity")
        ids = {"owner_" + str(index): lease.lease_id for index, lease in enumerate(native_parents)}
        result["observations"]["actual_parent_inventory_after_unwind"] = inventory(scheduler, ids)
        need(all(row["record_present"] and not row["release_requested"]
                 for row in result["observations"]["actual_parent_inventory_after_unwind"].values()),
             "actual_parent_rows_not_retained_after_context_unwind")
        for label, owner in zip(("primary", "watchdog"), owners):
            expect_refusal(label + "_stopped_release_after_context_unwind",
                lambda owner=owner: owner.release(artifacts_durable=True), daemon_error,
                "native consumers must finish and release before finalization", result)
        result["actual_phase_release_requested"] = phase.release()
        need(phase.released and result["actual_phase_release_requested"] is True, "actual_phase_release_not_acknowledged")
        result["observations"]["after_phase_release_before_renew"] = C.observe_ordinary_custody_hold(hold)
        # Public renewal authenticates only these retained consumer records.
        # It cannot resurrect cancelled ancestors or authorize productive work.
        result["actual_consumer_renew_after_cancellation"] = [consumer.renew() for consumer in consumers]
        result["observations"]["after_cancelled_consumer_renew"] = C.observe_ordinary_custody_hold(hold)
        cancelled = result["observations"]["after_cancelled_consumer_renew"]
        result["capacity_retained_after_cancellation"] = cancelled["capacity_records_retained"]
        need(result["actual_consumer_renew_after_cancellation"] == [True, True]
             and cancelled["consumer_ancestry_cancelled"] == [True, True]
             and cancelled["productive_ancestry_retained"] is False and result["capacity_retained_after_cancellation"],
             "renewal_did_not_preserve_retained_cancelled_distinction")
        result["productive_authority_after_cancellation"] = False
        expect_refusal("forget_after_ancestry_cancellation", lambda: C.forget_acknowledged_custody_hold(hold),
            C.CustodyHoldRefusal, "unacknowledged_native_consumers_still_held", result)
        for label, owner in zip(("primary", "watchdog"), owners):
            expect_refusal(label + "_stopped_release_after_ancestry_cancellation",
                lambda owner=owner: owner.release(artifacts_durable=True), daemon_error,
                "native consumers must finish and release before finalization", result)
        result["passed"] = True
    except BaseException as error:
        result.update(passed=False, failure_type=type(error).__name__, failure_reason=str(error)[:2048])
    finally:
        # This fixture has no external allocation API. Still observe original
        # actual owner records and attempts before public acknowledgement.
        entered = [owner for owner in owners if owner.to_dict()["status"] != "not_entered"]
        for owner in entered:
            if str(owner.directory) not in identities:
                row = owner.directory.lstat()
                identities[str(owner.directory)] = (row.st_dev, row.st_ino)
        verify_sources(request)
        result["observations"]["known_unstarted_before_acknowledgement"] = known_unstarted(entered, identities)
        result["native_identifiers"] = {"phase": phase.lease_id,
            **{"owner_" + str(index): lease.lease_id for index, lease in enumerate(native_parents)},
            **{"consumer_" + str(index): lease.lease_id for index, lease in enumerate(consumers)}}
        result["observations"]["actual_inventory_before_acknowledgement"] = inventory(scheduler, result["native_identifiers"])
        if result["actual_phase_release_requested"]:
            phase_row = result["observations"]["actual_inventory_before_acknowledgement"]["phase"]
            need(phase_row["record_present"] and phase_row["release_requested"] and phase.cancelled,
                 "released_ancestor_capacity_record_not_retained_with_actual_consumers")
        if hold is not None:
            need(C.held_native_consumers(hold) == tuple(consumers), "actual_original_consumer_identity_changed")
        for consumer in reversed(consumers):
            consumer.release()
        actual_ids = {row["lease_id"] for row in scheduler.active_leases()}
        result["actual_consumer_rows_drained"] = all(consumer.released and consumer.lease_id not in actual_ids for consumer in consumers)
        need(result["actual_consumer_rows_drained"], "actual_consumer_rows_not_drained")
        if hold is not None:
            result["observations"]["actual_hold_forget_after_acknowledgement"] = C.forget_acknowledged_custody_hold(hold)
        statuses = {}
        # These original attempts remain charged while stopped-cleanup facts
        # are durably saved; this is not productive work or a new allocation.
        for owner in entered:
            durable_closeout(owner.directory, {"schema": "legal-ir-ordinary-custody-unwind-stopped-closeout@1",
                "fixture_only": True, "experiment_passed": result["passed"], "refusals": result["refusals"],
                "observations": result["observations"], "actual_consumer_rows_drained": result["actual_consumer_rows_drained"],
                "provider_calls_performed": False, "canonical_tasks_accepted": 0, **NO_AUTHORITY})
        for owner in reversed(entered):
            # This is the real public stopped-cleanup API, valid after ancestor
            # cancellation; productive finalize would be an incorrect choice.
            final = owner.release(artifacts_durable=True)
            native = final["resource_lease"]
            statuses[owner.reservation_id] = {"status": final["status"],
                "native_lease_id": native["lease_id"] if native is not None else None,
                "native_lease_released": native["released"] if native is not None else None,
                "fresh_finalization_profile": final["record"]["finalization_profile"]}
        result["actual_disk_owner_statuses"] = statuses
        need(all(row["status"] == "released" and row["native_lease_released"] in (True, None) for row in statuses.values()),
             "actual_disk_owner_stopped_finalization_incomplete")


def main():
    need(len(sys.argv) == 3 and re.fullmatch("[0-9a-f]{64}", sys.argv[2]), "reviewed_request_path_and_sha_required")
    request_raw = read_stable(Path(sys.argv[1]).absolute())
    need(hashlib.sha256(request_raw).hexdigest() == sys.argv[2], "reviewed_unwind_request_changed")
    request = strict(request_raw)
    need(request.get("schema") == "legal-ir-ordinary-custody-unwind-request@1"
         and request.get("run_directory") == str(RUN) and request.get("name") == PROFILE
         and request.get("duration_seconds") == 1800 and request.get("provider_calls_performed") is False
         and request.get("canonical_tasks_accepted") == 0, "exact_bounded_unwind_request_required")
    verify_sources(request)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", IPFS_DATASETS_PY_MINIMAL_IMPORTS="1", IPFS_ACCEL_SKIP_CORE="1",
        IPFS_AUTO_INSTALL="false", IPFS_DATASETS_AUTO_INSTALL="false", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    sys.dont_write_bytecode = True
    need(not any(name == "ipfs_datasets_py" or name.startswith("ipfs_datasets_py.") for name in sys.modules),
         "fresh_actual_native_module_import_required")
    sys.path.insert(0, str(DATASETS))
    modules = [importlib.import_module(NATIVE + name) for name in NATIVE_NAMES]
    need(all(Path(module.__file__).absolute() == DATASETS / "ipfs_datasets_py/optimizers/logic_theorem_optimizer" / (name + ".py")
             for name, module in zip(NATIVE_NAMES, modules)), "actual_held_native_modules_required")
    parented, daemon, existing, resource = modules
    for name in ("ipfs_datasets_py", "ipfs_datasets_py.router_deps", "ipfs_datasets_py.optimizers",
                 NATIVE[:-1], NATIVE + "proof_resource_safety", NATIVE + "runtime_telemetry"):
        module = sys.modules.get(name)
        relative = Path(*name.split("."))
        expected = DATASETS / (str(relative / "__init__.py") if name in
            {"ipfs_datasets_py", "ipfs_datasets_py.optimizers", NATIVE[:-1]} else str(relative) + ".py")
        need(module is not None and Path(module.__file__).absolute() == expected,
             "actual_transitive_native_module_origin_required")
    spec = importlib.util.spec_from_file_location("genuine_unwind_ordinary_custody_hold", CUSTODY_SOURCE)
    C = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = C
    spec.loader.exec_module(C)
    scheduler = existing.open_existing_scheduler(STATE, POLICY)
    # The exact existing CPU policy skips legacy full GPU telemetry. Refuse
    # instead of overriding policy or accidentally invoking nvidia-smi.
    need(all(getattr(scheduler.config, name) is None for name in
        ("max_memory_percent", "max_swap_percent", "max_gpu_memory_percent")),
        "existing_cpu_policy_must_skip_external_process_telemetry")
    need(not RUN.exists() and RUN.parent.is_dir(), "fresh_fixed_unwind_run_required")
    result = {"schema": "legal-ir-ordinary-custody-unwind-terminal@1", "fixture_only": True,
        "scope": "genuine_native_no_external_process_unwind_fixture_only", "source_request_sha256": sys.argv[2],
        "source_manifest_sha256": request["source_manifest_sha256"], "passed": False,
        "observations": {}, "refusals": [], "actual_owner_contexts_unwound": False,
        "actual_phase_release_requested": False, "capacity_retained_after_unwind": False,
        "capacity_retained_after_cancellation": False, "productive_authority_after_cancellation": False,
        "actual_consumer_rows_drained": False, "actual_disk_owner_statuses": {},
        "root_and_phase_rows_drained": False, "provider_calls_performed": False,
        "models_called": 0, "canonical_tasks_accepted": 0, "Docker_started": False,
        "OS_process_children_created": False, "host_process_death_survival_guaranteed": False, **NO_AUTHORITY}
    with scheduler.acquire(resource.ResourceLane.ORCHESTRATION, cpu_slots=2, memory_mb=1024, child_process_slots=12,
            timeout=1800, request_id=PROFILE) as root:
        with root.acquire_child(cpu_slots=2, memory_mb=1024, child_process_slots=12,
                timeout=1800, request_id=PROFILE + ":phase") as phase:
            RUN.mkdir(mode=0o700)
            for name in ("primary", "watchdog"):
                (RUN / name).mkdir(mode=0o700)
            fsync_directory(RUN)
            fsync_directory(RUN.parent)
            experiment(request, scheduler, phase, parented.ParentedPhaseResourceReservation,
                daemon.DaemonResourceError, C, result)
        result["native_identifiers"]["root"] = root.lease_id
    active = {row["lease_id"] for row in scheduler.active_leases()}
    result["root_and_phase_rows_drained"] = root.released and phase.released and all(
        identifier not in active for identifier in result["native_identifiers"].values())
    need(result["root_and_phase_rows_drained"], "actual_native_root_phase_or_descendant_rows_not_drained")
    # Terminal stdout is captured by root's direct finite native entry. Private
    # stopped-closeout outputs, if any, must precede actual owner release.
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
