"""Finite real-worker metadata handoff qualification, without provider dispatch.

Execute only inside the root-provisioned pinned offline qualification container.
Native fixture transport is authored; worker birth and Unix peer identity must
come from actual Popen/proc/SO_PEERCRED observations. No target code is executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import select
import stat
import subprocess
import threading
import time

from .terminal_world_packet_replay_experiment import CLEANUP_SCOPE, MAX_FIXTURE_BYTES, _world_private_fixture
from .terminal_world_prelaunch_experiment import _bounded, _wire

SCHEMA = "terminal-world-worker-handoff-offline-experiment@1"
MAX_REPORT_BYTES = 128 * 1024
MAX_FRAME_BYTES = 64 * 1024
WORKER_ENTRY = Path("/usr/local/libexec/ipfs-world-handoff-observe-worker")
SUDO = Path("/usr/bin/sudo")
WORKER_ARGV = (str(SUDO), "-n", "-u", "benchmarkworker", "-g", "supervisor", "--", str(WORKER_ENTRY))
WORKER_PEER_ARGV = ("/opt/ipfs-supervisor/venv/bin/python", "-I", str(WORKER_ENTRY))
WORKER_ENTRY_SOURCE = '''#!/opt/ipfs-supervisor/venv/bin/python -I
import ctypes,os,sys
libc=ctypes.CDLL(None,use_errno=True)
if libc.prctl(38,1,0,0,0)!=0:raise OSError("no_new_privileges_unavailable")
os.environ.update({"IPFS_ACCEL_SKIP_CORE":"1","IPFS_KIT_DISABLE":"1","PYTHONDONTWRITEBYTECODE":"1","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","HF_DATASETS_OFFLINE":"1","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1","RAYON_NUM_THREADS":"1","TOKENIZERS_PARALLELISM":"false"})
sys.path[:0]=["/opt/ipfs-supervisor/source","/opt/ipfs-supervisor/datasets","/opt/ipfs-supervisor/kit"]
from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_worker_observer import worker_observer_client_main
raise SystemExit(worker_observer_client_main())
'''
_LOCK = threading.Lock()
FALSE = dict(provider_calls=0, model_calls=0, target_code_executed=False,
    training_executed=False, actual_quack_tcp_verified=False, compiled_context_verified=False,
    world_dispatch_ready=False, provider_launch=False, proof_authority=False,
    execution_authority=False, completion_authority=False, task_completed=False,
    raw_content_exported=False, database_access_granted=False, token_savings_established=False,
    container_image_independently_verified=False, full_container_containment_verified=False,
    whole_repository_observed=False, worker_workspace_verified=False)
CHECKS = frozenset({"trusted_fixture_selected_source_provisioning", "native_assignment_current_before_transition",
    "allocated_root_only_readonly_transition", "old_assignment_and_allocation_stay_stale",
    "real_literal_popen_registered", "real_worker_uid_gid_and_birth", "exact_handoff_metadata_received",
    "no_worker_broker_source_or_database_grant", "changed_native_body_refused_while_worker_alive",
    "owned_worker_reaped", "observer_stopped_with_complete_two_response_population", "native_contexts_closed"})
SOURCE_PATHS = (
    "benchmarks/agent_supervisor/container_coding/terminal_world_worker_handoff_experiment.py",
    "benchmarks/agent_supervisor/container_coding/terminal_world_packet_replay_experiment.py",
    "benchmarks/agent_supervisor/container_coding/terminal_world_prelaunch_experiment.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/repository_world_worker_handoff.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/repository_world_worker_observer.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/repository_world_prelaunch.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/current_source_task_admission.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/current_source_worktree_allocation.py",
    "test/api/test_current_source_task_admission.py",
    "test/api/test_current_source_worktree_allocation.py",
    "test/api/test_repository_world_prelaunch.py",
)


def _need(value, code):
    if not value:
        raise ValueError(code)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def report_sha256(report):
    return _sha(_wire(report))


def worker_client_main():
    """The fixed root-owned launcher exposes only the bounded observe client."""
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_worker_observer import worker_observer_client_main
    return worker_observer_client_main()


def _pins():
    root = Path(__file__).resolve().parents[3]
    files = {}
    for name in SOURCE_PATHS:
        path = root / name
        _need(path.resolve(strict=True) == path, "source_pin_alias")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            _need(stat.S_ISREG(before.st_mode) and before.st_size <= 2 * 1024**2, "source_pin_bound")
            raw = bytearray()
            while len(raw) < before.st_size:
                part = os.read(fd, min(65536, before.st_size - len(raw)))
                _need(bool(part), "source_pin_short_read")
                raw.extend(part)
            identity = lambda row: (row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns)
            _need(not os.read(fd, 1) and identity(before) == identity(os.fstat(fd)) == identity(path.lstat())
                and path.resolve(strict=True) == path, "source_pin_changed")
            files[name] = dict(bytes=len(raw), sha256=_sha(raw))
        finally:
            os.close(fd)
    return dict(scope="listed_source_files_only_not_transitive_provenance", files=files)


def _profile():
    """Require real preinstalled host identities/launcher, never simulate UID."""
    _need(os.getuid() == os.geteuid() == 1000 and os.getgid() == os.getegid() == 1000,
        "actual_owner_identity_required")
    worker = pwd.getpwnam("benchmarkworker")
    _need(worker.pw_uid == 1001 and worker.pw_gid == 1000, "preinstalled_worker_identity_required")
    for path in (SUDO, WORKER_ENTRY):
        _need(path.resolve(strict=True) == path, "trusted_worker_launcher_alias")
        for parent in (path, *path.parents):
            info = parent.lstat()
            _need(info.st_uid == 0 and not info.st_mode & 0o022, "trusted_worker_launcher_not_root_controlled")
        info = path.lstat()
        _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_mode & 0o111,
            "trusted_worker_launcher_not_regular")
    _need(WORKER_ENTRY.read_bytes() == WORKER_ENTRY_SOURCE.encode(), "trusted_worker_launcher_source_differs")
    return dict(owner_uid=os.getuid(), owner_gid=os.getgid(), worker_uid=worker.pw_uid,
        worker_gid=worker.pw_gid, launcher_sha256=_sha(WORKER_ENTRY_SOURCE.encode()),
        literal_popen_argv_sha256=_sha(_wire(list(WORKER_ARGV))),
        worker_peer_argv_sha256=_sha(_wire(list(WORKER_PEER_ARGV))),
        namespaces={name: os.readlink("/proc/self/ns/" + name) for name in ("pid", "mnt", "net")})


def _line(stream, *, timeout=5.0):
    deadline, raw = time.monotonic() + timeout, bytearray()
    while True:
        remaining = deadline - time.monotonic()
        _need(remaining > 0, "worker_observation_timeout")
        ready, _, _ = select.select([stream.fileno()], [], [], remaining)
        _need(bool(ready), "worker_observation_timeout")
        part = os.read(stream.fileno(), 1)
        _need(bool(part), "worker_observation_closed")
        raw.extend(part)
        _need(len(raw) <= MAX_FRAME_BYTES, "worker_observation_bound")
        if part == b"\n":
            return bytes(raw)


def _observe(process):
    process.stdin.write(b"observe\n"); process.stdin.flush()
    raw = _line(process.stdout)
    value = json.loads(raw)
    _need(type(value) is dict and _wire(value) + b"\n" == raw, "worker_observation_not_canonical")
    return value, _sha(raw)


def _refused(operation):
    try:
        operation()
    except (ValueError, RuntimeError):
        return True
    return False


def _cycle(fixture):
    import pytest
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_prelaunch import capture_repository_world_prelaunch_assignment
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_worker_handoff import capture_repository_world_worker_handoff
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_worker_observer import RepositoryWorldWorkerObserver, FALSE_FIELDS, RESPONSE_SCHEMA
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import FakeQuackTransport
    from test.api.test_current_source_task_admission import PROMPT
    from test.api.test_current_source_worktree_allocation import _allocation_case
    from test.api.test_repository_world_prelaunch import provision_authored_prelaunch_sources
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import canonical_json_bytes

    checks = {}
    process = observer = handoff = None
    with pytest.MonkeyPatch.context() as patch:
        with _allocation_case(fixture, patch, defer_source_binding=True) as case:
            native = case.native
            _need(type(native.server.transport) is FakeQuackTransport, "authored_native_transport_profile_differs")
            native.binding = case.guard.bind_current_source_lookup(**case.source_binding)
            provision = provision_authored_prelaunch_sources(case)
            checks["trusted_fixture_selected_source_provisioning"] = provision == {
                "selected_files": 2, "bytes_unchanged": True, "executable_bits_unchanged": True}
            assignment = capture_repository_world_prelaunch_assignment(task_admission=native.handle,
                worktree_admission=case.guard, prompt=PROMPT)
            initial = assignment.require_current()
            checks["native_assignment_current_before_transition"] = initial["world_dispatch_ready"] is False
            before = native.workspace.lstat()
            try:
                handoff = capture_repository_world_worker_handoff(assignment=assignment,
                    worker_uid=1001, worker_gid=1000, timeout_seconds=120)
                transitioned = handoff.apply_readonly_transition()
                after = native.workspace.lstat()
                checks["allocated_root_only_readonly_transition"] = (
                    stat.S_IMODE(before.st_mode) == 0o700 and stat.S_IMODE(after.st_mode) == 0o750
                    and (before.st_dev, before.st_ino, before.st_uid, before.st_gid)
                    == (after.st_dev, after.st_ino, after.st_uid, after.st_gid))
                checks["old_assignment_and_allocation_stay_stale"] = (
                    _refused(assignment.require_current) and _refused(case.guard.require_current))
                observer = RepositoryWorldWorkerObserver(handoff=handoff, prompt=PROMPT,
                    expected_launcher_argv=WORKER_ARGV, expected_worker_argv=WORKER_PEER_ARGV,
                    timeout_seconds=60.0)
                process = subprocess.Popen(list(WORKER_ARGV), cwd=native.workspace,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    start_new_session=True)
                observer.note_spawned(process)
                checks["real_literal_popen_registered"] = process.poll() is None
                initial_frame = dict(descriptor=observer.descriptor,
                    expected_descriptor_sha256=observer.descriptor_sha256)
                _need(len(_wire(initial_frame)) <= MAX_FRAME_BYTES, "worker_initial_frame_bound")
                process.stdin.write(_wire(initial_frame) + b"\n"); process.stdin.flush()
                positive, positive_sha = _observe(process)
                _need(positive.get("ok") is True and positive.get("worker_peer_authenticated") is True,
                    "actual_worker_observation_refused")
                peer = positive["peer"]
                checks["real_worker_uid_gid_and_birth"] = (peer["uid"] == 1001 and peer["gid"] == 1000
                    and type(peer["pid"]) is int and peer["pid"] > 0
                    and type(peer["start_time_ticks"]) is int and peer["start_time_ticks"] > 0
                    and peer["root_pid"] == process.pid and peer["root_start_time_ticks"] > 0)
                checks["exact_handoff_metadata_received"] = positive["assignment"] == handoff.require_current()
                checks["no_worker_broker_source_or_database_grant"] = all(positive[name] is False for name in FALSE_FIELDS)
                original_body = canonical_json_bytes(dict(native.native_task.body)).decode()
                changed = dict(native.native_task.body); changed["title"] = "Authored stale worker assignment"
                with native.server._lock:
                    native.server._connection.execute("UPDATE tasks SET body_json = ? WHERE task_cid = ?",
                        [canonical_json_bytes(changed).decode(), native.task_cid])
                try:
                    alive = process.poll() is None
                    negative, negative_sha = _observe(process)
                    checks["changed_native_body_refused_while_worker_alive"] = (alive
                        and negative.get("schema") == RESPONSE_SCHEMA
                        and negative.get("ok") is False and negative.get("assignment") is None
                        and negative.get("reason_code") == "observation_refused"
                        and negative.get("worker_peer_authenticated") is False
                        and _refused(handoff.require_current))
                finally:
                    with native.server._lock:
                        native.server._connection.execute("UPDATE tasks SET body_json = ? WHERE task_cid = ?",
                            [original_body, native.task_cid])
                process.stdin.close()
                code = process.wait(timeout=5)
                checks["owned_worker_reaped"] = code == 0
                stopped = observer.close(); observer = None
                checks["observer_stopped_with_complete_two_response_population"] = (
                    stopped["service_stopped"] is True and stopped["sent_responses"] == 2
                    and stopped["refused_responses"] == 1 and stopped["accepted_connections"] == 2)
                closed = handoff.close(); handoff = None
                measurement = dict(positive_response_sha256=positive_sha,
                    negative_response_sha256=negative_sha, handoff_observation_sha256=_sha(_wire(transitioned)),
                    observer_close_sha256=_sha(_wire(stopped)), handoff_close_sha256=_sha(_wire(closed)),
                    worker_peer=dict(peer), successful_metadata_observations=1, stale_metadata_refusals=1,
                    allocated_root_mode_before=0o700, allocated_root_mode_after=0o750,
                    actual_worker_birth_verified=True, actual_worker_uid_gid_verified=True,
                    observation_scope="authenticated_worker_assignment_metadata_only")
            finally:
                if process is not None and process.poll() is None:
                    # Terminate only the exact owned Popen monitor. sudo owns
                    # forwarding to its worker; root's Docker owner handles an
                    # uncertain isolated descendant, never a foreign host PID.
                    process.terminate()
                    process.wait(timeout=5)
                if process is not None:
                    for stream in (process.stdin, process.stdout, process.stderr):
                        if stream is not None and not stream.closed: stream.close()
                if observer is not None: observer.close()
                if handoff is not None: handoff.close()
    checks["native_contexts_closed"] = True
    _need(set(checks) == CHECKS and all(value is True for value in checks.values()), "worker_handoff_check_failed")
    return dict(checks=checks, measurement=measurement)


def run_experiment(*, scratch_root: Path):
    _need(_LOCK.acquire(blocking=False), "worker_handoff_experiment_already_running")
    previous = Path.cwd()
    try:
        before = _pins()
        profile = cycle = None
        failure = None
        with _world_private_fixture(scratch_root) as (fixture, cleanup):
            cleanup["private_fixture_native_contexts_closed"] = False
            try:
                profile = _profile()
                cycle = _cycle(fixture)
                cleanup["private_fixture_native_contexts_closed"] = True
            except Exception as error:
                text = str(error)
                reason = text if re.fullmatch(r"[a-z][a-z0-9_]{0,95}", text) else "unspecified_failure"
                failure = dict(stage="trusted_worker_handoff_fixture", error_type=type(error).__name__,
                    reason_code=reason)
        unchanged = before == _pins()
        qualified = cycle is not None and failure is None and unchanged and cleanup["private_fixture_cleanup_complete"]
        report = dict(schema=SCHEMA, qualified=qualified,
            scope="real_worker_readonly_handoff_and_authenticated_assignment_metadata_only",
            native_transport="authored_FakeQuackTransport_no_actual_TCP",
            inner_resource_accounting_scope="container_local_fixture_authority_only",
            source_pins=before, source_pins_unchanged=unchanged,
            profile=profile, cycle=cycle, failure=failure, **cleanup, **FALSE)
        report["report_sha256"] = _sha(_wire(report))
        _need(len(_wire(report)) <= MAX_REPORT_BYTES, "worker_handoff_report_bound")
        return report
    finally:
        os.chdir(previous); _LOCK.release()


def validate_report(report, *, expected_report_sha256):
    _bounded(report)
    _need(type(report) is dict and len(_wire(report)) <= MAX_REPORT_BYTES, "worker_handoff_report_bound")
    required = {"schema", "qualified", "scope", "native_transport", "inner_resource_accounting_scope",
        "source_pins", "source_pins_unchanged", "profile", "cycle", "failure", "report_sha256",
        "private_fixture_cleanup_complete", "private_fixture_files", "private_fixture_bytes",
        "private_fixture_storage_bound_bytes", "private_fixture_cleanup_directories_made_owner_writable",
        "private_fixture_cleanup_scope", "private_fixture_root_removed", "private_fixture_cleanup_removed_files",
        "private_fixture_cleanup_removed_directories", "private_fixture_native_contexts_closed", *FALSE}
    optional = {"private_fixture_observed_prefix_entries", "private_fixture_observed_prefix_files",
        "private_fixture_observed_prefix_bytes", "private_fixture_cleanup_reason"}
    _need(required <= set(report) <= required | optional, "worker_report_fields_differ")
    _need(type(expected_report_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", expected_report_sha256)
        and report_sha256(report) == expected_report_sha256, "independent_worker_report_pin_differs")
    body = dict(report); seal = body.pop("report_sha256", None)
    _need(seal == _sha(_wire(body)), "worker_report_self_seal_differs")
    _need(report.get("schema") == SCHEMA and report.get("scope") ==
        "real_worker_readonly_handoff_and_authenticated_assignment_metadata_only"
        and report.get("native_transport") == "authored_FakeQuackTransport_no_actual_TCP"
        and report.get("inner_resource_accounting_scope") == "container_local_fixture_authority_only",
        "worker_report_scope_differs")
    for name, expected in FALSE.items():
        _need(type(report.get(name)) is type(expected) and report[name] == expected, "worker_report_authority_claim")
    _need(report.get("private_fixture_cleanup_scope") == CLEANUP_SCOPE
        and report.get("private_fixture_root_removed") is False
        and report.get("private_fixture_storage_bound_bytes") == MAX_FIXTURE_BYTES, "worker_report_cleanup_scope_differs")
    for name in ("qualified", "source_pins_unchanged", "private_fixture_cleanup_complete",
            "private_fixture_native_contexts_closed"):
        _need(type(report[name]) is bool, "worker_report_boolean_invalid")
    pins = report["source_pins"]
    _need(type(pins) is dict and set(pins) == {"scope", "files"}
        and pins["scope"] == "listed_source_files_only_not_transitive_provenance"
        and type(pins["files"]) is dict and set(pins["files"]) == set(SOURCE_PATHS), "worker_report_source_population_differs")
    for pin in pins["files"].values():
        _need(type(pin) is dict and set(pin) == {"bytes", "sha256"}
            and type(pin["bytes"]) is int and 0 <= pin["bytes"] <= 2 * 1024**2
            and type(pin["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]), "worker_report_source_pin_invalid")
    failure = report["failure"]
    _need(failure is None or type(failure) is dict and set(failure) == {"stage", "error_type", "reason_code"}
        and failure["stage"] == "trusted_worker_handoff_fixture"
        and type(failure["error_type"]) is str and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", failure["error_type"])
        and type(failure["reason_code"]) is str and re.fullmatch(r"[a-z][a-z0-9_]{0,95}", failure["reason_code"]),
        "worker_report_failure_invalid")
    cycle = report.get("cycle")
    if cycle is not None:
        _validate_cycle(cycle)
    profile = report.get("profile")
    if profile is not None:
        _validate_profile(profile)
    _need(cycle is None or profile is not None, "worker_report_profile_missing")
    qualified = (cycle is not None and report.get("failure") is None
        and report.get("source_pins_unchanged") is True
        and report.get("private_fixture_cleanup_complete") is True
        and report.get("private_fixture_native_contexts_closed") is True)
    _need(type(report.get("qualified")) is bool and report["qualified"] is qualified, "worker_report_qualification_differs")
    return dict(valid=True, qualified=qualified, report_sha256=expected_report_sha256,
        native_operations_independently_replayed=False, **FALSE)


def _validate_cycle(cycle):
    _need(type(cycle) is dict and set(cycle) == {"checks", "measurement"}
        and type(cycle["checks"]) is dict and set(cycle["checks"]) == CHECKS
        and all(value is True for value in cycle["checks"].values()), "worker_report_checks_differ")
    measurement = cycle["measurement"]
    hashed = {"positive_response_sha256", "negative_response_sha256", "handoff_observation_sha256",
        "observer_close_sha256", "handoff_close_sha256"}
    _need(type(measurement) is dict and set(measurement) == hashed | {"worker_peer",
        "successful_metadata_observations", "stale_metadata_refusals", "allocated_root_mode_before",
        "allocated_root_mode_after", "actual_worker_birth_verified", "actual_worker_uid_gid_verified",
        "observation_scope"}, "worker_report_measurement_fields_differ")
    for name in hashed:
        _need(type(measurement[name]) is str and re.fullmatch(r"[0-9a-f]{64}", measurement[name]), "worker_report_hash_invalid")
    _need(type(measurement.get("successful_metadata_observations")) is int
        and measurement["successful_metadata_observations"] == 1
        and type(measurement.get("stale_metadata_refusals")) is int and measurement["stale_metadata_refusals"] == 1
        and measurement.get("actual_worker_birth_verified") is True
        and measurement.get("actual_worker_uid_gid_verified") is True
        and measurement.get("allocated_root_mode_before") == 0o700
        and measurement.get("allocated_root_mode_after") == 0o750
        and measurement.get("observation_scope") == "authenticated_worker_assignment_metadata_only",
        "worker_report_measurement_differs")
    peer = measurement.get("worker_peer")
    _need(type(peer) is dict and set(peer) == {"pid", "uid", "gid", "start_time_ticks", "root_pid",
        "root_start_time_ticks", "groups_sha256", "group_count"}
        and type(peer.get("uid")) is int and peer["uid"] == 1001
        and type(peer.get("gid")) is int and peer["gid"] == 1000
        and type(peer.get("pid")) is int and peer["pid"] > 0
        and type(peer.get("root_pid")) is int and peer["root_pid"] > 0
        and type(peer.get("start_time_ticks")) is int and peer["start_time_ticks"] > 0
        and type(peer.get("root_start_time_ticks")) is int and peer["root_start_time_ticks"] > 0,
        "worker_report_peer_differs")
    _need(type(peer["group_count"]) is int and 1 <= peer["group_count"] <= 64
        and type(peer["groups_sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", peer["groups_sha256"]),
        "worker_report_groups_invalid")


def _validate_profile(profile):
    _need(type(profile) is dict and set(profile) == {"owner_uid", "owner_gid", "worker_uid", "worker_gid",
        "launcher_sha256", "literal_popen_argv_sha256", "worker_peer_argv_sha256", "namespaces"},
        "worker_report_profile_fields_differ")
    for name, expected in (("owner_uid", 1000), ("owner_gid", 1000), ("worker_uid", 1001), ("worker_gid", 1000)):
        _need(type(profile[name]) is int and profile[name] == expected, "worker_report_profile_identity_differs")
    _need(profile["launcher_sha256"] == _sha(WORKER_ENTRY_SOURCE.encode())
        and profile["literal_popen_argv_sha256"] == _sha(_wire(list(WORKER_ARGV)))
        and profile["worker_peer_argv_sha256"] == _sha(_wire(list(WORKER_PEER_ARGV))),
        "worker_report_launcher_differs")
    namespaces = profile["namespaces"]
    _need(type(namespaces) is dict and set(namespaces) == {"pid", "mnt", "net"}
        and all(type(value) is str and re.fullmatch(name + r":\[[0-9]+\]", value)
            for name, value in namespaces.items()), "worker_report_namespace_observation_invalid")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    _need(args.output.parent.resolve(strict=True) == args.output.parent.absolute(), "canonical_output_parent_required")
    report = run_experiment(scratch_root=args.scratch_root)
    validate_report(report, expected_report_sha256=report_sha256(report))
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(_wire(report) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(schema=SCHEMA, qualified=report["qualified"],
        report_sha256=report_sha256(report), **FALSE), sort_keys=True))
    return 0 if report["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
