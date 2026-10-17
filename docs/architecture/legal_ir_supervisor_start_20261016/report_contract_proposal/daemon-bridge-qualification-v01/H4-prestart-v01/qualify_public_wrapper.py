"""Genuine public H/C/G prestart boundary; deliberately no G._run dispatch.

Root executes only after independently finalized capsule preparation. No fake
native owner, monkeypatched runner, copied key, JSON capability or model call.
"""
from __future__ import annotations
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time

HERE = Path(__file__).absolute().parent
PHASE = HERE.parent
RUN = Path('/home/barberb/lift_coding/external/ipfs_datasets/workspace/test-logs/federal-corpus-audits/owned-supervisor-runs/legal-ir-native-host-h4-fixture-20261017-r01')
PREPARATION_RUN = RUN.with_name('legal-ir-native-host-h4-20261017-r01')
CAPSULE = PREPARATION_RUN / 'preparation/capsule'
NATIVE_RUN = RUN / 'public-wrapper-prestart'
EXTERNAL = NATIVE_RUN / 'external'
WORKSPACE, TEMP, HOME = (EXTERNAL / name for name in ('workspace', 'provider-temp', 'home'))
ROOT = Path('/home/barberb/lift_coding/external/ipfs_datasets')
ROOTS = (ROOT / 'workspace/test-logs', ROOT / 'workspace/todo-queues', ROOT / 'docs/implementation/reports/evidence', Path('/tmp/pytest-of-barberb'))
LEDGER = ROOT / 'workspace/test-logs/federal-corpus-audits/owned-daemon-resource-control/disk-reservations.json'
STATE = Path('/tmp/ipfs-datasets-resource-scheduler-1000.json')
POLICY = 'f07cb14289fcb6275e4497dcc36be0dcf8bfc3dc87531ff7a9537eb85fdcc661'
DATASETS = Path('/home/barberb/lift_coding/artifacts/legal-conditions-composition-diagnostic-20261015-01/datasets')
NATIVE = 'ipfs_datasets_py.optimizers.logic_theorem_optimizer.'
NATIVE_NAMES = ('parented_phase_resources','autoencoder_daemon_resources','existing_resource_scheduler','resource_scheduler')
RUNTIME = CAPSULE / 'ipfs_accelerate_py/agent_supervisor/runtime'
PREPARATION = PHASE / 'H4-capsule-preparation-root-observation-r01.json'
GROK = Path('/home/barberb/.grok/downloads/grok-1.0.60-linux-aarch64')
GROK_ENTRY = Path('/home/barberb/.grok/bin/grok')
RUNNER_MANIFEST = '1a2d28a952f0bac6a7af951b4754085dcb1110f86fb46ebf319014f2e66062e2'
HANDOFF_MANIFEST = 'b74455cd469cbf78892d6fda2b09c8234cd5922b7e72f741c845c2b0c2f8f0d8'
PREP_MANIFEST = 'bddccbde3f4b76131cec93ce71fd88e6f79264475bbb99ae9aade14dc170a0c7'
PREP_REQUEST = '3eb3c2eea54afb431851f8a84dc920984fdde523cc9cb020b7c455daadabe1b7'
PROFILE = 'legal-ir-H4-prestart-r01'
CASES = ('known_unstarted_abort','cancelled_unstarted')
NO_AUTHORITY = {name:False for name in ('acceptance_authority','completion_authority','dispatch_authority','actual_provider_execution_qualified','production_hook_deployed','actual_live_callable_transport_qualified')}

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


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def inventory(scheduler, identifiers):
    rows = {row["lease_id"]: row for row in scheduler.active_leases()}
    return {name: {"lease_id": value, "record_present": value in rows,
                  "cancelled": rows.get(value, {}).get("cancelled"),
                  "release_requested": rows.get(value, {}).get("release_requested", False)}
            for name, value in identifiers.items()}


def wire(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def save(path, value):
    raw = wire(value)
    need(len(raw) <= 2 * 1024 * 1024, "bounded_private_fixture_output_required")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)


def source_guard(request):
    pins = request.get("source_sha256")
    need(type(pins) is dict and 1 <= len(pins) <= 256, "bounded_complete_fixture_source_pins_required")
    required = {str(Path(__file__).absolute()), str(HERE / "SOURCE_MANIFEST.json"), str(PREPARATION),
        str(CAPSULE / "SOURCE_MANIFEST.json"), str(GROK), "/usr/bin/python3.12", "/usr/bin/docker"}
    required.update(str(RUNTIME / name) for name in ("grok_cli_runner.py", "ordinary_success_handoff.py", "ordinary_custody_hold.py"))
    required.update(str(DATASETS / relative) for relative in (
        "ipfs_datasets_py/__init__.py", "ipfs_datasets_py/router_deps.py", "ipfs_datasets_py/optimizers/__init__.py",
        "ipfs_datasets_py/optimizers/logic_theorem_optimizer/__init__.py"))
    required.update(str(DATASETS / "ipfs_datasets_py/optimizers/logic_theorem_optimizer" / (name + ".py")) for name in (
        *NATIVE_NAMES, "runtime_telemetry", "proof_resource_safety", "validation_run_resources", "autoencoder_resource_recovery"))
    need(required <= set(pins), "actual_preimport_native_callable_source_closure_required")
    need(request.get("trusted_grok_entrypoint") == str(GROK_ENTRY)
         and GROK_ENTRY.name == "grok" and GROK_ENTRY.resolve(strict=True) == GROK,
         "actual_existing_grok_entrypoint_and_resolved_binary_join_required")
    for spelling, digest in pins.items():
        need(type(digest) is str and re.fullmatch("[0-9a-f]{64}", digest), "exact_source_digest_required")
        path = Path(spelling)
        need(path.is_absolute() and os.path.abspath(path) == str(path) and ".." not in path.parts
             and not any(item.is_symlink() for item in (path, *path.parents)), "canonical_source_pin_required")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        actual, count = hashlib.sha256(), 0
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid in {0, os.getuid()}
                 and before.st_mode & 0o002 == 0 and 0 < before.st_size <= 256 * 1024 * 1024,
                 "bounded_owned_source_pin_required")
            while chunk := stream.read(1024 * 1024):
                count += len(chunk)
                need(count <= before.st_size, "source_pin_grew")
                actual.update(chunk)
            after = os.fstat(stream.fileno())
        need(count == before.st_size and witness(before) == witness(after) == witness(path.lstat())
             and actual.hexdigest() == digest, "actual_protected_source_changed")
    manifest_raw = read_stable(HERE / "SOURCE_MANIFEST.json")
    need(pins[str(HERE / "SOURCE_MANIFEST.json")] == request["source_manifest_sha256"]
         == hashlib.sha256(manifest_raw).hexdigest(), "exact_fixture_manifest_join_required")
    manifest = strict(manifest_raw)
    need(manifest.get("schema") == "legal-ir-callable-public-wrapper-prestart-source@2"
         and manifest.get("cases") == list(CASES) and manifest.get("controlled_cases") == 2
         and manifest.get("owner_lifetime_seconds") == 3600,
         "exact_reviewed_fixture_source_generation_required")
    for row in manifest["files"]:
        relative = Path(row["path"])
        need(not relative.is_absolute() and ".." not in relative.parts
             and relative.as_posix() == row["path"], "canonical_fixture_source_row_required")
        content = read_stable(HERE / relative)
        need(len(content) == row["bytes"] and hashlib.sha256(content).hexdigest()
             == row["sha256"] == pins.get(str(HERE / relative)), "complete_fixture_bundle_source_join_required")
    for row in manifest["input_pins"]:
        need(pins.get(row["path"]) == row["sha256"], "complete_reviewed_fixture_inputs_required")
    for field in ("handoff_source_sha256", "custody_source_sha256"):
        subset = request.get(field)
        need(type(subset) is dict and subset and all(pins.get(path) == digest for path, digest in subset.items()),
             "executed_subset_source_pin_join_required")


def ready_capsule(request):
    root_raw = read_stable(PREPARATION)
    need(hashlib.sha256(root_raw).hexdigest() == request["preparation_observation_sha256"], "actual_preparation_observation_changed")
    root = strict(root_raw)
    need(root.get("schema") == "legal-ir-callable-capsule-root-observation@1"
         and root.get("actual_exec_exit_code") == 0 and root.get("original_native_preparation_finalized") is True
         and root.get("source_manifest_sha256") == PREP_MANIFEST and root.get("source_request_sha256") == PREP_REQUEST,
         "actual_original_preparation_finalization_required")
    terminal = root.get("actual_native_terminal", {})
    need(terminal.get("status") == "completed" and terminal.get("native_finalized") is True
         and terminal.get("child_returncode") == 0 and terminal.get("failure_type") is None,
         "actual_preparation_owner_terminal_required")
    result_raw = read_stable(PREPARATION_RUN / "preparation/capsule-preparation.json")
    result = strict(result_raw)
    manifest_raw = read_stable(CAPSULE / "SOURCE_MANIFEST.json")
    digest = hashlib.sha256(manifest_raw).hexdigest()
    need(root.get("preparation_result_sha256") == hashlib.sha256(result_raw).hexdigest()
         and root.get("capsule_manifest_sha256") == result.get("capsule_manifest_sha256") == request["capsule_manifest_sha256"] == digest
         and result.get("schema") == "legal-ir-callable-capsule-preparation@1" and result.get("passed") is True,
         "actual_copied_capsule_result_join_required")
    manifest = strict(manifest_raw)
    need(manifest.get("schema") == "legal-ir-callable-handoff-capsule@1"
         and manifest.get("capsule_root") == str(CAPSULE)
         and manifest.get("base_capsule_manifest_sha256") == "71053aea3f7e5917cd218bcfbf6458fe351fb7fb36de64ed723789b0a44cee2d"
         and manifest.get("runner_source_manifest_sha256") == RUNNER_MANIFEST
         and manifest.get("handoff_source_manifest_sha256") == HANDOFF_MANIFEST,
         "actual_callable_capsule_provenance_required")
    rows = manifest.get("files")
    need(type(rows) is list and len(rows) == manifest.get("payload_file_count") == 2807
         and manifest.get("payload_bytes") == 106772927, "exact_callable_capsule_population_required")
    seen, total = set(), 0
    for row in rows:
        relative = Path(row["path"])
        need(not relative.is_absolute() and ".." not in relative.parts and relative.suffix == ".py"
             and row["path"] not in seen, "canonical_unique_capsule_source_required")
        path = CAPSULE / relative
        need(path.resolve(strict=True) == path and not any(item.is_symlink() for item in (path, *path.parents)),
             "actual_unaliased_capsule_source_required")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid == os.getuid()
                 and stat.S_IMODE(before.st_mode) == 0o444 and 0 <= before.st_size <= 16 * 1024 * 1024,
                 "actual_readonly_capsule_source_required")
            raw = stream.read(16 * 1024 * 1024 + 1)
            after = os.fstat(stream.fileno())
        need(witness(before) == witness(after) == witness(path.lstat()) and len(raw) == before.st_size == row["bytes"]
             and hashlib.sha256(raw).hexdigest() == row["sha256"], "actual_capsule_file_changed")
        total += len(raw)
        seen.add(row["path"])
    actual = {path.relative_to(CAPSULE).as_posix() for path in CAPSULE.rglob("*") if not path.is_dir()}
    need(actual == seen | {"SOURCE_MANIFEST.json"} and total == manifest["payload_bytes"],
         "complete_credential_free_capsule_inventory_required")


class KnownUnstartedFixtureAbort(RuntimeError):
    pass


def globals_witness():
    return (os.getcwd(), dict(os.environ), tempfile.tempdir, tuple(sys.argv), sys.stdin, sys.stdout, sys.stderr)


def original_exception_in_chain(error, original):
    observed, pending = set(), [error]
    while pending and len(observed) < 32:
        current = pending.pop()
        if current is original:
            return True
        if id(current) in observed:
            continue
        observed.add(id(current))
        for linked in (current.__context__, current.__cause__):
            if linked is not None:
                pending.append(linked)
    return False


def known_unstarted(owners, allowed_files):
    records = []
    for owner in owners:
        public = owner.to_dict()
        record = public["record"]
        actual = owner.directory.lstat()
        bound = record["attempt_directory"]
        need(record.get("child") is None and record.get("prior_children", []) == []
             and bound["path"] == str(owner.directory) and (bound["device"], bound["inode"]) == (actual.st_dev, actual.st_ino)
             and public["cleanup_error"] is None, "actual_original_zero_child_history_required")
        names = {path.name for path in owner.directory.iterdir()}
        need(names <= allowed_files.get(str(owner.directory), set()), "unexpected_private_fixture_allocation")
        for path in owner.directory.iterdir():
            meta = path.lstat()
            need(stat.S_ISREG(meta.st_mode) and meta.st_nlink == 1 and meta.st_uid == os.getuid()
                 and stat.S_IMODE(meta.st_mode) == 0o600 and meta.st_size == 0, "actual_output_sink_not_empty_private_regular")
        records.append({"reservation_id": public["reservation_id"], "attempt_directory": str(owner.directory),
            "child": None, "prior_children": [], "private_empty_sinks": sorted(names)})
    need({path.name for path in EXTERNAL.iterdir()} == {"workspace", "provider-temp", "home"},
         "unexpected_external_fixture_allocation")
    for path in (WORKSPACE, TEMP, HOME):
        meta = path.lstat()
        need(stat.S_ISDIR(meta.st_mode) and meta.st_uid == os.getuid() and stat.S_IMODE(meta.st_mode) == 0o700
             and not any(path.iterdir()), "actual_known_unstarted_external_directory_changed")
    return {"scope": "source_pinned_before_run_abort_or_earlier_refusal_with_original_empty_attempts",
        "actual_original_owner_records": records, "actual_external_directories_empty": True,
        "provider_or_Docker_process_started": False, "native_task_attempt_started": False}


def retain_unknown(error):
    # Caller frames and C's genuine registry retain the original objects. No
    # blind close, reconstructed authority, provider retry or consumer release.
    while True:
        print(json.dumps({"schema": "legal-ir-callable-prestart-custody-pending@1", "failure_type": type(error).__name__,
            "original_native_custody_retained": True, "provider_execution_qualified": False,
            "canonical_tasks_accepted": 0}), flush=True)
        try:
            time.sleep(30)
        except BaseException:
            pass


def run_case(G, H, C, scheduler, owners, native_parents, request, case, allowed_files):
    consumers, hold, descriptors = [], None, []
    before_count, callback_count = 0, 0
    stop = KnownUnstartedFixtureAbort("expected_source_pinned_before_run_abort")
    observed = {"fixture_case": case, "passed": False, "provider_calls_performed": False,
        "canonical_tasks_accepted": 0, "actual_public_wrapper_called": False, **NO_AUTHORITY}
    def on_success(_outcome):
        nonlocal callback_count
        callback_count += 1
        raise RuntimeError("nonmodel_prestart_fixture_must_not_receive_success")
    def before_run():
        nonlocal before_count
        before_count += 1
        rows = scheduler.active_leases()
        for parent, consumer in zip(native_parents, consumers):
            direct = [row for row in rows if row.get("parent_lease_id") == parent.lease_id]
            need(len(direct) == 1 and direct[0]["lease_id"] == consumer.lease_id
                 and direct[0]["owner_pid"] == os.getpid() and direct[0]["cpu_slots"] == 1
                 and direct[0]["memory_mb"] == 1 and direct[0]["child_process_slots"] == 0
                 and consumer.released is False and consumer.cancelled is False,
                 "actual_pre_run_original_consumers_not_retained")
        observed["actual_consumers_observed_inside_before_run"] = [consumer.lease_id for consumer in consumers]
        resolved = G._resolve_trusted_grok_bin(configured=str(GROK_ENTRY), workspace=WORKSPACE)
        need(resolved == str(GROK), "actual_original_trusted_binary_resolution_differs")
        H.ordinary_trusted_binary_resolved(resolved)
        need(H.ordinary_callable_binary_mount_allowed(Path(resolved)) is True,
             "actual_pinned_binary_mount_join_not_observed")
        try:
            H.ordinary_callable_binary_mount_allowed(Path("/usr/bin/python3.12"))
        except H.OrdinaryHandoffRefusal as error:
            need(str(error) == "actual_original_callable_binary_required",
                 "actual_binary_mismatch_refusal_differs")
        else:
            raise RuntimeError("different_binary_source_wrongly_admitted")
        observed["actual_binary_binding_checks"] = {
            "original_alias_resolved_to_pinned_binary": True,
            "actual_resolver_result_joined_to_original_context": True,
            "same_pinned_source_mount_join_allowed": True,
            "different_source_refused": "actual_original_callable_binary_required",
            "scope": "actual_preeffect_hooks_only_no_Docker_mount_or_provider_dispatch"}
        if case == "cancelled_unstarted":
            need(native_parents[0].cancel(), "original_primary_public_cancel_required")
            need(consumers[0].cancelled and not consumers[0].released,
                 "actual_cancelled_original_consumer_must_remain_charged")
            observed["actual_public_primary_cancel_performed"] = True
        raise stop
    try:
        for parent in native_parents:
            consumers.append(parent.acquire_child(cpu_slots=1, memory_mb=1, child_process_slots=0,
                requires_gpu=False, timeout=30, request_id=PROFILE + ":" + case + ":" + str(len(consumers))))
        hold = C.bind_ordinary_custody_hold(primary_owner=owners[0], watchdog_owner=owners[1],
            primary_consumer=consumers[0], watchdog_consumer=consumers[1], protected_source_sha256=request["custody_source_sha256"])
        for stream in ("stdout", "stderr"):
            path = owners[0].directory / (case + "." + stream)
            allowed_files[str(owners[0].directory)].add(path.name)
            descriptors.append(os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600))
        args = G.parse_runner_arguments(["--workspace", str(WORKSPACE), "--model", "grok-4.7", "--grok-bin", str(GROK_ENTRY)])
        invocation = G.OrdinaryCallableInvocation(prompt="Known-unstarted callable engineering fixture; no provider execution.",
            cwd=WORKSPACE, environment={"HOME": str(HOME), "PATH": "/usr/bin:/bin", "TMPDIR": str(TEMP), "TMP": str(TEMP),
                "TEMP": str(TEMP), "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            temporary_root=TEMP, stdout_path=owners[0].directory / (case + ".stdout"),
            stderr_path=owners[0].directory / (case + ".stderr"), stdout_fd=descriptors[0], stderr_fd=descriptors[1], output_limit_bytes=65536)
        pins = dict(request["handoff_source_sha256"])
        if case == "missing_runner_pin":
            del pins[str(RUNTIME / "grok_cli_runner.py")]
        expected = {"missing_callback": (H.OrdinaryHandoffRefusal, "explicit_callable_handoff_requires_success_callback"),
            "receipt_forbidden": (H.OrdinaryHandoffRefusal, "primary_only_handoff_forbids_supervised_receipt_dispatch"),
            "missing_runner_pin": (H.OrdinaryHandoffRefusal, "exact_handoff_source_pins_required"),
            "missing_bound_hold": (C.CustodyHoldRefusal, "genuine_bound_custody_hold_required")}
        need(H.ordinary_callable_binary_mount_allowed(GROK) is False,
             "binary_mount_permission_present_without_actual_context")
        prior = globals_witness()
        try:
            observed["actual_public_wrapper_called"] = True
            H.run_ordinary_with_success_handoff(args, descriptors[0] if case == "receipt_forbidden" else -1,
                owner=owners[0], watchdog_owner=owners[1], source_sha256=pins,
                declared_external_storage=request["declared_external_storage"], on_success=None if case == "missing_callback" else on_success,
                invocation=invocation, custody_hold=None if case == "missing_bound_hold" else hold,
                custody_source_sha256=request["custody_source_sha256"], before_run=before_run)
        except H.OrdinaryCustodyPending as error:
            need(case == "cancelled_unstarted" and before_count == 1 and callback_count == 0,
                 "only_actual_cancelled_unstarted_pending_expected")
            need(error.actual_owner is owners[0] and error.actual_watchdog_owner is owners[1]
                 and original_exception_in_chain(error, stop), "actual_original_pending_and_abort_identity_required")
            active = {row["lease_id"] for row in scheduler.active_leases()}
            need(all(not consumer.released and consumer.lease_id in active for consumer in consumers)
                 and consumers[0].cancelled, "cancelled_original_capacity_must_be_retained_before_ack")
            known_unstarted(owners, allowed_files)  # Independent actual zero-effect observation before ack.
            need(H.acknowledge_stopped_ordinary_custody(error) is True,
                 "actual_H4_stopped_cancel_acknowledgment_required")
            active = {row["lease_id"] for row in scheduler.active_leases()}
            need(all(consumer.released and consumer.lease_id not in active for consumer in consumers),
                 "cancelled_original_consumers_not_drained_after_ack")
            need(H.ordinary_callable_binary_mount_allowed(GROK) is False and globals_witness() == prior
                 and all(os.fstat(fd).st_size == 0 for fd in descriptors), "cancelled_scope_or_borrowed_outputs_changed")
            observed.update(passed=True,before_run_calls=before_count,success_callback_calls=callback_count,
                actual_pending_retained_before_ack=True,actual_stopped_ack_after_cancellation=True,
                actual_borrowed_FDs_empty=True,actual_ambient_globals_preserved=True)
        except BaseException as error:
            if case == "known_unstarted_abort":
                need(error is stop and before_count == 1, "original_before_run_exception_identity_not_preserved")
                active = {row["lease_id"] for row in scheduler.active_leases()}
                need(all(consumer.released and consumer.lease_id not in active for consumer in consumers),
                     "public_H_did_not_acknowledge_original_consumers_before_fixture_cleanup")
                try:
                    C.require_bound_ordinary_custody_hold(hold, primary_owner=owners[0], watchdog_owner=owners[1])
                except C.CustodyHoldRefusal as refusal:
                    need(str(refusal) == "genuine_bound_custody_hold_required", "actual_H_hold_forget_not_observed")
                else:
                    raise RuntimeError("public_H_did_not_forget_acknowledged_hold")
                observed["actual_H_consumers_drained_before_fixture_cleanup"] = True
                observed["same_before_run_exception_rethrown"] = True
            else:
                kind, message = expected[case]
                need(type(error) is kind and str(error) == message and before_count == 0,
                     "actual_early_wrapper_refusal_differs:" + type(error).__name__ + ":" + str(error))
                observed["actual_early_refusal"] = {"type": type(error).__name__, "reason": str(error)}
            need(H.ordinary_callable_binary_mount_allowed(GROK) is False,
                 "binary_mount_permission_survived_actual_context")
            need(callback_count == 0 and globals_witness() == prior, "callback_or_ambient_global_state_changed")
            need(all(os.fstat(fd).st_size == 0 and os.pread(fd, 1, 0) == b"" for fd in descriptors), "actual_callable_output_not_empty")
            observed.update(passed=True, before_run_calls=before_count, success_callback_calls=callback_count,
                actual_borrowed_FDs_empty=True, actual_ambient_globals_preserved=True)
        else:
            raise RuntimeError("known_unstarted_wrapper_returned_instead_of_refusing")
    except H.OrdinaryCustodyPending as error:
        retain_unknown(error)
    except BaseException as error:
        observed.update(passed=False, failure_type=type(error).__name__, failure_reason=str(error)[:2048])
    finally:
        try:
            observed["actual_known_unstarted_before_caller_cleanup"] = known_unstarted(owners, allowed_files)
        except BaseException as error:
            retain_unknown(error)
        # The positive assertion above precedes this fallback cleanup. This
        # only cleans actual caller-owned early-refusal/unstarted consumers.
        remaining = [consumer for consumer in consumers if not consumer.released]
        if remaining:
            for consumer in remaining:
                consumer.release()
            if hold is not None:
                C.forget_acknowledged_custody_hold(hold)
        active = {row["lease_id"] for row in scheduler.active_leases()}
        observed["actual_original_consumer_rows_drained"] = all(consumer.released and consumer.lease_id not in active for consumer in consumers)
        observed["actual_consumer_lease_ids"] = [consumer.lease_id for consumer in consumers]
        need(observed["actual_original_consumer_rows_drained"], "actual_original_consumers_not_drained")
        for fd in descriptors:
            os.fsync(fd)
            os.close(fd)
        fsync_directory(owners[0].directory)
    return observed


def main():
    need(len(sys.argv) == 3 and re.fullmatch("[0-9a-f]{64}", sys.argv[2]), "reviewed_fixture_request_and_sha_required")
    request_raw = read_stable(Path(sys.argv[1]).absolute())
    need(hashlib.sha256(request_raw).hexdigest() == sys.argv[2], "reviewed_fixture_request_changed")
    request = strict(request_raw)
    need(request.get("schema") == "legal-ir-callable-public-wrapper-prestart-request@1"
         and request.get("cases") == list(CASES) and request.get("owner_lifetime_seconds") == 3600
         and request.get("root_and_phase_envelope") == {"cpu_slots": 3, "memory_mb": 4096, "child_process_slots": 16},
         "exact_finite_five_case_fixture_request_required")
    source_guard(request)
    ready_capsule(request)
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", IPFS_DATASETS_PY_MINIMAL_IMPORTS="1", IPFS_ACCEL_SKIP_CORE="1",
        IPFS_AUTO_INSTALL="false", IPFS_DATASETS_AUTO_INSTALL="false", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    sys.dont_write_bytecode = True
    need(not any(name == "ipfs_datasets_py" or name.startswith("ipfs_datasets_py.") or name == "ipfs_accelerate_py"
                 or name.startswith("ipfs_accelerate_py.") for name in sys.modules), "fresh_actual_module_imports_required")
    sys.path.insert(0, str(DATASETS))
    modules = [importlib.import_module(NATIVE + name) for name in NATIVE_NAMES]
    parented, _daemon, existing, resource = modules
    need(all(Path(module.__file__).absolute() == DATASETS / "ipfs_datasets_py/optimizers/logic_theorem_optimizer" / (name + ".py")
             for name, module in zip(NATIVE_NAMES, modules)), "actual_held_native_module_origins_required")
    scheduler = existing.open_existing_scheduler(STATE, POLICY)
    need(all(getattr(scheduler.config, name) is None for name in (
        "max_memory_percent", "max_swap_percent", "max_gpu_memory_percent")),
        "existing_CPU_policy_must_skip_external_process_telemetry")
    need(not os.path.lexists(RUN) and RUN.parent.is_dir() and RUN.parent.resolve(strict=True) == RUN.parent
         and not any(path.is_symlink() for path in (RUN.parent, *RUN.parent.parents)),
         "fresh_canonical_public_wrapper_campaign_required")
    need(not os.path.lexists(NATIVE_RUN), "fresh_public_wrapper_fixture_attempt_required")
    with scheduler.acquire(resource.ResourceLane.ORCHESTRATION, cpu_slots=3, memory_mb=4096, child_process_slots=16,
            timeout=1800, request_id=PROFILE) as root:
        with root.acquire_child(cpu_slots=3, memory_mb=4096, child_process_slots=16, timeout=1800, request_id=PROFILE + ":phase") as phase:
            source_guard(request)
            ready_capsule(request)
            sys.path.insert(0, str(CAPSULE))
            G, H, C = [importlib.import_module("ipfs_accelerate_py.agent_supervisor.runtime." + name) for name in (
                "grok_cli_runner", "ordinary_success_handoff", "ordinary_custody_hold")]
            need(all(Path(module.__file__).absolute() == RUNTIME / name for module, name in zip((G, H, C), (
                "grok_cli_runner.py", "ordinary_success_handoff.py", "ordinary_custody_hold.py"))), "actual_readonly_callable_modules_required")
            RUN.mkdir(mode=0o700)
            fsync_directory(RUN.parent)
            NATIVE_RUN.mkdir(mode=0o700)
            primary_dir, watchdog_dir = NATIVE_RUN / "primary", NATIVE_RUN / "watchdog"
            for path in (primary_dir, watchdog_dir):
                path.mkdir(mode=0o700)
            fsync_directory(NATIVE_RUN)
            fsync_directory(RUN)
            with parented.ParentedPhaseResourceReservation(LEDGER, roots=ROOTS, attempt_directory=primary_dir, parent_lease=phase,
                    storage_bytes=512 * 1024 * 1024, memory_mb=1024, cpu_slots=1, child_process_slots=8,
                    scheduler_state_path=STATE, expected_scheduler_policy_sha256=POLICY, timeout_seconds=3600,
                    ledger_lock_timeout_seconds=None) as primary:
                with parented.ParentedPhaseResourceReservation(LEDGER, roots=ROOTS, attempt_directory=watchdog_dir, parent_lease=phase,
                        storage_bytes=128 * 1024 * 1024, memory_mb=512, cpu_slots=1, child_process_slots=4,
                        scheduler_state_path=STATE, expected_scheduler_policy_sha256=POLICY, timeout_seconds=3600,
                        ledger_lock_timeout_seconds=None) as watchdog:
                    declarations = [{"path": str(CAPSULE), "bytes": 128 * 1024 * 1024}, {"path": str(EXTERNAL), "bytes": 128 * 1024 * 1024}]
                    need(request.get("declared_external_storage") == declarations, "exact_capsule_and_external_promises_required")
                    for row in declarations:
                        primary.account_external_bytes(hashlib.sha256(row["path"].encode()).hexdigest(), row["bytes"])
                    EXTERNAL.mkdir(mode=0o700)
                    for path in (WORKSPACE, TEMP, HOME):
                        path.mkdir(mode=0o700)
                    fsync_directory(EXTERNAL)
                    fsync_directory(NATIVE_RUN)
                    owners = (primary, watchdog)
                    native_parents = (primary.native_lease, watchdog.native_lease)
                    allowed_files = {str(primary_dir): set(), str(watchdog_dir): set()}
                    results = [run_case(G, H, C, scheduler, owners, native_parents, request, case, allowed_files) for case in CASES]
                    source_guard(request)
                    ready_capsule(request)
                    result = {"schema": "legal-ir-callable-public-wrapper-prestart-terminal@1", "fixture_only": True,
                        "scope": "genuine_H4_public_before_run_abort_and_public_cancel_stopped_ack_only",
                        "passed": all(row["passed"] for row in results), "cases": results,
                        "source_manifest_sha256": request["source_manifest_sha256"], "source_request_sha256": sys.argv[2],
                        "capsule_manifest_sha256": request["capsule_manifest_sha256"], "controlled_cases": 2,
                        "actual_public_wrapper_called": any(row["actual_public_wrapper_called"] for row in results), "G_run_dispatch_performed": False,
                        "provider_calls_performed": False, "Docker_operations_performed": False,
                        "OS_process_children_created": False, "native_task_attempts_started": 0,
                        "native_engineering_consumer_pairs_acquired": sum(len(row["actual_consumer_lease_ids"]) == 2 for row in results),
                        "actual_public_wrapper_calls": sum(row["actual_public_wrapper_called"] for row in results),
                        "canonical_tasks_accepted": 0, **NO_AUTHORITY}
                    save(primary.directory / "public-wrapper-prestart.json", result)
                    fsync_directory(watchdog.directory)
                    result["actual_owner_finalization"] = {"watchdog": watchdog.release(artifacts_durable=True),
                        "primary": primary.release(artifacts_durable=True)}
                    need(all(row["status"] == "released" for row in result["actual_owner_finalization"].values()), "actual_owner_finalization_incomplete")
    ids = {root.lease_id, phase.lease_id, *(lease.lease_id for lease in native_parents)}
    ids.update(identifier for row in results for identifier in row["actual_consumer_lease_ids"])
    active = {row["lease_id"] for row in scheduler.active_leases()}
    result["root_and_phase_rows_drained"] = root.released and phase.released and not ids.intersection(active)
    need(result["root_and_phase_rows_drained"], "actual_native_fixture_rows_not_drained")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


