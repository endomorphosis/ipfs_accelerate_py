"""Bounded authored qualification of the owner-side source assignment gate.

Native claims, typed commands, Portal projection, Git allocation and lifecycle
records are real. The retained fixture uses FakeQuackTransport and authored
deployment metadata. Nothing launches a worker, provider, or target program.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
import time

from .terminal_world_packet_replay_experiment import (
    CLEANUP_SCOPE, MAX_FIXTURE_BYTES, _world_private_fixture,
)

SCHEMA = "terminal-world-prelaunch-offline-experiment@1"
MAX_REPORT_BYTES = 256 * 1024
MAX_PIN_FILE_BYTES = 2 * 1024 * 1024
_LOCK = threading.Lock()
FALSE = dict(provider_calls=0, model_calls=0, target_code_executed=False,
    training_executed=False, docker_started=False, live_terminal_bench_run=False,
    actual_quack_tcp_verified=False, container_boundary_verified=False,
    separate_worker_uid_verified=False, compiled_world_packet_verified=False,
    world_dispatch_ready=False, proof_authority=False, execution_authority=False,
    completion_authority=False, task_completed=False, raw_content_exported=False,
    token_savings_established=False)
CHECKS = frozenset({"exact_original_assignment_gate", "canonical_allocated_source_equal",
    "canonical_allocated_paths_distinct", "native_portal_task_ids_distinct",
    "initial_gate_current", "genuine_lifecycle_renewal_preserved",
    "stale_native_revision_refused", "changed_native_body_refused",
    "expired_allocation_record_refused", "allocation_mode_transition_refused",
    "authored_owner_uid_fault_refused", "released_native_claim_refused",
    "no_provider_invocation_recorded", "gate_consumed_before_source_proxy", "native_source_read_qualified",
    "trusted_fixture_selected_files_provisioned", "trusted_fixture_source_bytes_unchanged",
    "trusted_fixture_executable_bits_unchanged"})
SOURCE_PATHS = (
    "benchmarks/agent_supervisor/container_coding/terminal_world_prelaunch_experiment.py",
    "benchmarks/agent_supervisor/container_coding/terminal_world_packet_replay_experiment.py",
    "benchmarks/agent_supervisor/container_coding/terminal_capsule_columns_experiment.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/repository_world_prelaunch.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/current_source_task_admission.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/current_source_worktree_allocation.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/source_lookup_worker_broker.py",
    "ipfs_accelerate_py/agent_supervisor/merge/worktree_lifecycle.py",
    "test/api/test_current_source_task_admission.py",
    "test/api/test_current_source_worktree_allocation.py",
    "test/api/test_repository_world_prelaunch.py",
    "test/api/test_repository_world_prelaunch_broker.py",
)


def _need(value, code):
    if not value:
        raise ValueError(code)


def _wire(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")


def _bounded(value):
    pending, nodes, text_bytes = [(value, 0)], 0, 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        _need(nodes <= 4096 and depth <= 32, "prelaunch_report_structure_bound")
        if type(item) is str:
            text_bytes += len(item.encode("utf-8"))
            _need(text_bytes <= MAX_REPORT_BYTES, "prelaunch_report_exceeds_bound")
        elif type(item) is dict:
            _need(len(item) <= 4096 and all(type(k) is str for k in item), "prelaunch_report_structure_bound")
            pending.extend((v, depth + 1) for pair in item.items() for v in pair)
        elif type(item) is list:
            _need(len(item) <= 4096, "prelaunch_report_structure_bound")
            pending.extend((v, depth + 1) for v in item)
        else:
            _need(item is None or type(item) is bool or type(item) is int
                and -(1 << 63) < item < (1 << 63), "prelaunch_report_value_invalid")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def report_sha256(report):
    """FULL canonical report hash, including its internal seal."""
    return _sha(_wire(report))


def _pins():
    root = Path(__file__).resolve().parents[3]
    files = {}
    for name in SOURCE_PATHS:
        path = root / name
        _need(path.resolve(strict=True) == path, "source_pin_path_alias")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            _need(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= MAX_PIN_FILE_BYTES,
                "source_pin_file_bound")
            raw = bytearray()
            while len(raw) < before.st_size:
                block = os.read(fd, min(65536, before.st_size - len(raw)))
                _need(bool(block), "source_pin_short_read")
                raw.extend(block)
            _need(not os.read(fd, 1), "source_pin_file_changed")
            identity = lambda row: (row.st_dev, row.st_ino, row.st_size,
                row.st_mtime_ns, row.st_ctime_ns)
            _need(identity(before) == identity(os.fstat(fd)) == identity(path.lstat())
                and path.resolve(strict=True) == path, "source_pin_file_changed")
            files[name] = dict(bytes=len(raw), sha256=_sha(raw))
        finally:
            os.close(fd)
    return dict(scope="listed_source_files_only_not_transitive_provenance", files=files)


def _refused(operation):
    try:
        operation()
    except (ValueError, RuntimeError):
        return True
    return False


def _scenario(fixture):
    """Import retained authored fixtures only when root executes qualification."""
    import pytest
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_prelaunch import (
        capture_repository_world_prelaunch_assignment,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import FakeQuackTransport
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import canonical_json_bytes
    from ipfs_accelerate_py.agent_supervisor.runtime import source_lookup_worker_broker as broker
    from test.api.test_current_source_task_admission import PROMPT, SOURCE, _payload
    from test.api.test_current_source_worktree_allocation import _allocation_case, _record_change
    from test.api.test_repository_world_prelaunch import provision_authored_prelaunch_sources

    checks = {}
    with pytest.MonkeyPatch.context() as monkeypatch:
        with _allocation_case(fixture, monkeypatch, defer_source_binding=True) as case:
            native = case.native
            _need(type(native.server.transport) is FakeQuackTransport, "authored_transport_profile_differs")
            native.binding = case.guard.bind_current_source_lookup(**case.source_binding)
            # Trusted TEST fixture setup before capture; runtime/default owners
            # never alter source permissions. Only these two selected authored
            # files are provisioned, preserving bytes and executable bits.
            authored_paths = (native.canonical / "module.py", native.workspace / "module.py")
            authored_before = tuple(path.read_bytes() for path in authored_paths)
            executable_before = tuple(path.lstat().st_mode & 0o111 for path in authored_paths)
            provisioned = provision_authored_prelaunch_sources(case)
            _need(type(provisioned) is dict and set(provisioned) == {
                "selected_files", "bytes_unchanged", "executable_bits_unchanged"}
                and type(provisioned["selected_files"]) is int
                and provisioned["selected_files"] == len(authored_paths)
                and provisioned["bytes_unchanged"] is True
                and provisioned["executable_bits_unchanged"] is True,
                "trusted_fixture_source_provisioning_differs")
            checks["trusted_fixture_source_bytes_unchanged"] = authored_before == tuple(
                path.read_bytes() for path in authored_paths)
            checks["trusted_fixture_executable_bits_unchanged"] = executable_before == tuple(
                path.lstat().st_mode & 0o111 for path in authored_paths)
            checks["trusted_fixture_selected_files_provisioned"] = all(
                stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
                and path.lstat().st_uid == os.geteuid() and path.lstat().st_nlink == 1
                and not path.lstat().st_mode & 0o022 for path in authored_paths)
            gate = capture_repository_world_prelaunch_assignment(task_admission=native.handle,
                worktree_admission=case.guard, prompt=PROMPT)
            initial = gate.require_current()
            task_material = native.handle.require_current()
            allocation_material = case.guard.require_current()
            for key in ("proof_authority", "execution_authority", "completion_authority",
                    "world_dispatch_ready", "compiled_context_verified"):
                _need(initial[key] is False, "prelaunch_gate_authority_claim")
            consumer = dict(server=native.server, canonical_repository=native.canonical,
                workspace=native.workspace, prompt=PROMPT, task_admission=native.handle,
                worktree_admission=case.guard)
            checks["exact_original_assignment_gate"] = gate.require_for_broker(**consumer) == initial
            native.session = broker.start_source_lookup_worker_broker(**native.factory,
                world_prelaunch_assignment=gate)
            checks["gate_consumed_before_source_proxy"] = native.session._world_prelaunch_assignment is gate
            response = native.session._read_source(_payload(native))
            checks["native_source_read_qualified"] = (response["content"].encode() == SOURCE
                and response["read_only"] is True and response["completion_authority"] is False
                and native.session._native_response_bytes > 0)
            read_payload_bytes = native.session._native_response_bytes

            def rejected():
                return (_refused(gate.require_current)
                    and _refused(lambda: native.session._read_source(_payload(native)))
                    and native.session._native_response_bytes == read_payload_bytes)
            canonical = (native.canonical / "module.py").read_bytes()
            allocated = (native.workspace / "module.py").read_bytes()
            checks["canonical_allocated_source_equal"] = canonical == allocated == SOURCE
            checks["canonical_allocated_paths_distinct"] = native.canonical != native.workspace
            checks["native_portal_task_ids_distinct"] = native.task_cid != case.portal_identity["canonical_task_cid"]
            checks["initial_gate_current"] = gate.require_current() == initial

            record = case.store.load_workspace(native.workspace)
            renewed = case.store.renew_lease(native.workspace,
                lease_id=record.lease_id, expected_fence=record.fence)
            current = gate.require_current()
            advanced = renewed.fence == record.fence + 1
            checks["genuine_lifecycle_renewal_preserved"] = advanced and current == initial

            # Mutate only this fixture's existing native row; restore every
            # column before the next fault. Never serialize native authorities.
            original_body = canonical_json_bytes(dict(native.native_task.body)).decode()
            with native.server._lock:
                native.server._connection.execute("UPDATE tasks SET revision = revision + 1 WHERE task_cid = ?",
                    [native.task_cid])
            try:
                checks["stale_native_revision_refused"] = rejected()
            finally:
                with native.server._lock:
                    native.server._connection.execute("UPDATE tasks SET revision = ? WHERE task_cid = ?",
                        [native.native_task.revision, native.task_cid])
            changed = deepcopy(dict(native.native_task.body))
            changed["title"] = "Authored changed task body"
            with native.server._lock:
                native.server._connection.execute("UPDATE tasks SET body_json = ? WHERE task_cid = ?",
                    [canonical_json_bytes(changed).decode(), native.task_cid])
            try:
                checks["changed_native_body_refused"] = rejected()
            finally:
                with native.server._lock:
                    native.server._connection.execute("UPDATE tasks SET body_json = ? WHERE task_cid = ?",
                        [original_body, native.task_cid])
            with _record_change(case, lambda value: value.update(expires_at=time.time() - 1)):
                checks["expired_allocation_record_refused"] = rejected()
            mode = stat.S_IMODE(native.workspace.lstat().st_mode)
            native.workspace.chmod(mode | stat.S_IWGRP)
            try:
                checks["allocation_mode_transition_refused"] = rejected()
            finally:
                native.workspace.chmod(mode)
            original_uid = os.geteuid()
            with monkeypatch.context() as fault:
                fault.setattr(os, "geteuid", lambda: original_uid + 1)
                checks["authored_owner_uid_fault_refused"] = rejected()
            _need(gate.require_current() == initial, "restored_fixture_assignment_differs")
            checks["no_provider_invocation_recorded"] = native.daemon.provider_invocation_recorded(
                native.attempt.attempt_id, idempotency_key="provider:" + native.attempt.attempt_id) is None
            coordinator = native.daemon.coordinator
            claim = coordinator.get_task_claim(native.attempt.claim_id)
            lease = coordinator.protect_task_claim(claim)
            coordinator.release(lease, reason="authored prelaunch qualification release")
            checks["released_native_claim_refused"] = rejected()
            _need(set(checks) == CHECKS and all(value is True for value in checks.values()),
                "authored_prelaunch_check_failed")
            assignment = dict(gate_receipt_sha256=_sha(_wire(initial)),
                task_admission_sha256=_sha(_wire(task_material)),
                allocation_sha256=_sha(_wire(allocation_material)),
                canonical_repository_path_sha256=_sha(str(native.canonical).encode()),
                allocated_workspace_path_sha256=_sha(str(native.workspace).encode()),
                canonical_source_sha256=_sha(canonical), allocated_source_sha256=_sha(allocated),
                native_task_cid=native.task_cid,
                portal_task_cid=case.portal_identity["canonical_task_cid"],
                task_revision=native.native_task.revision)
            observations = dict(retained_positive_gate_snapshots=2, stale_refusals=6,
                read_only_native_payload_bytes=read_payload_bytes,
                authored_fixture_source_provisioned_files=provisioned["selected_files"],
                authored_fixture_source_bytes_unchanged=checks["trusted_fixture_source_bytes_unchanged"],
                authored_fixture_executable_bits_unchanged=checks["trusted_fixture_executable_bits_unchanged"],
                lifecycle_fence_advanced=advanced, native_claim_release_observed=True,
                kernel_uid_transition_performed=False, renewed_gate_receipt_sha256=_sha(_wire(current)))
    return dict(checks=checks, assignment=assignment, observations=observations,
        fault_scope="authored_metadata_faults_and_genuine_native_lifecycle_renewal_and_claim_release")


def run_experiment(*, scratch_root: Path):
    _need(_LOCK.acquire(blocking=False), "prelaunch_experiment_already_running")
    previous = Path.cwd()
    cleanup = {}
    scenario = None
    failure = None
    try:
        before = _pins()
        with _world_private_fixture(scratch_root) as (fixture, cleanup):
            cleanup["private_fixture_native_contexts_closed"] = False
            try:
                scenario = _scenario(fixture)
                cleanup["private_fixture_native_contexts_closed"] = True
            except Exception as error:
                # No private path, SQL message, owner token, or source is exported.
                failure = dict(stage="authored_prelaunch_scenario", error_type=type(error).__name__)
        unchanged = before == _pins()
        qualified = (scenario is not None and failure is None and unchanged
            and cleanup.get("private_fixture_cleanup_complete") is True)
        report = dict(schema=SCHEMA, qualified=qualified,
            scope="native_owner_prelaunch_assignment_only",
            transport_profile="authored_FakeQuackTransport_no_actual_TCP",
            deployment_profile="authored_host_metadata_no_actual_container_or_UID_transition",
            source_pins=before, source_pins_unchanged=unchanged,
            failure=failure, scenario=scenario, **cleanup, **FALSE)
        report["report_sha256"] = _sha(_wire(report))
        _need(len(_wire(report)) <= MAX_REPORT_BYTES, "prelaunch_report_exceeds_bound")
        return report
    finally:
        os.chdir(previous)
        _LOCK.release()


def validate_report(report, *, expected_report_sha256):
    """Authenticate external report custody and its finite observation contract."""
    _bounded(report)
    _need(type(report) is dict and len(_wire(report)) <= MAX_REPORT_BYTES,
        "prelaunch_report_exceeds_bound")
    required = {"schema", "qualified", "scope", "transport_profile", "deployment_profile",
        "source_pins", "source_pins_unchanged", "failure", "scenario", "report_sha256",
        "private_fixture_cleanup_complete", "private_fixture_files", "private_fixture_bytes",
        "private_fixture_storage_bound_bytes", "private_fixture_cleanup_directories_made_owner_writable",
        "private_fixture_cleanup_scope", "private_fixture_root_removed", "private_fixture_cleanup_removed_files",
        "private_fixture_cleanup_removed_directories", "private_fixture_native_contexts_closed", *FALSE}
    optional = {"private_fixture_observed_prefix_entries", "private_fixture_observed_prefix_files",
        "private_fixture_observed_prefix_bytes", "private_fixture_cleanup_reason"}
    _need(required <= set(report) <= required | optional, "prelaunch_report_fields_differ")
    _need(type(expected_report_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", expected_report_sha256)
        and report_sha256(report) == expected_report_sha256, "independent_report_hash_differs")
    body = dict(report)
    seal = body.pop("report_sha256", None)
    _need(seal == _sha(_wire(body)), "internal_report_hash_differs")
    _need(report.get("schema") == SCHEMA
        and report.get("scope") == "native_owner_prelaunch_assignment_only"
        and report.get("transport_profile") == "authored_FakeQuackTransport_no_actual_TCP"
        and report.get("deployment_profile") == "authored_host_metadata_no_actual_container_or_UID_transition",
        "prelaunch_report_scope_differs")
    for name, value in FALSE.items():
        _need(type(report.get(name)) is type(value) and report[name] == value,
            "prelaunch_authority_or_execution_claim")
    _need(type(report.get("source_pins_unchanged")) is bool
        and type(report.get("qualified")) is bool, "prelaunch_qualification_type")
    pins = report.get("source_pins")
    _need(type(pins) is dict and set(pins) == {"scope", "files"}
        and pins["scope"] == "listed_source_files_only_not_transitive_provenance"
        and type(pins["files"]) is dict and set(pins["files"]) == set(SOURCE_PATHS),
        "prelaunch_source_population_differs")
    for value in pins["files"].values():
        _need(type(value) is dict and set(value) == {"bytes", "sha256"}
            and type(value["bytes"]) is int and 0 <= value["bytes"] <= MAX_PIN_FILE_BYTES
            and type(value["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]),
            "prelaunch_source_pin_invalid")
    _need(report.get("private_fixture_cleanup_scope") == CLEANUP_SCOPE
        and report.get("private_fixture_root_removed") is False
        and report.get("private_fixture_storage_bound_bytes") == MAX_FIXTURE_BYTES
        and type(report.get("private_fixture_cleanup_complete")) is bool,
        "prelaunch_cleanup_scope_differs")
    for key in ("private_fixture_cleanup_directories_made_owner_writable", "private_fixture_cleanup_removed_files",
            "private_fixture_cleanup_removed_directories", "private_fixture_observed_prefix_entries",
            "private_fixture_observed_prefix_files", "private_fixture_observed_prefix_bytes"):
        if key in report:
            _need(type(report[key]) is int and 0 <= report[key] <= MAX_FIXTURE_BYTES,
                "prelaunch_cleanup_counter_invalid")
    for key in ("private_fixture_files", "private_fixture_bytes"):
        _need(report[key] is None or type(report[key]) is int and 0 <= report[key] <= MAX_FIXTURE_BYTES,
            "prelaunch_cleanup_counter_invalid")
    _need(type(report["private_fixture_native_contexts_closed"]) is bool,
        "prelaunch_context_close_invalid")
    if "private_fixture_cleanup_reason" in report:
        _need(report["private_fixture_cleanup_reason"] in {
            "private_fixture_identity_changed", "private_fixture_foreign_owner", "private_fixture_cross_device",
            "private_fixture_file_bound_exceeded", "private_fixture_byte_bound_exceeded",
            "unexpected_private_fixture_symlink", "unexpected_private_fixture_file_kind",
            "private_fixture_native_contexts_not_closed", "private_fixture_contents_changed",
            "owned_fixture_cleanup_io_unavailable", "owned_fixture_cleanup_unavailable"},
            "prelaunch_cleanup_reason_invalid")
    scenario = report.get("scenario")
    if scenario is not None:
        _validate_scenario(scenario)
    failure = report.get("failure")
    _need(failure is None or type(failure) is dict
        and set(failure) == {"stage", "error_type"}
        and failure["stage"] == "authored_prelaunch_scenario"
        and type(failure["error_type"]) is str and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", failure["error_type"]),
        "prelaunch_failure_projection_invalid")
    qualified = (scenario is not None and failure is None and report["source_pins_unchanged"]
        and report["private_fixture_cleanup_complete"]
        and report.get("private_fixture_native_contexts_closed") is True)
    _need(report["qualified"] is qualified, "prelaunch_qualification_claim_differs")
    return dict(schema=SCHEMA, valid=True, qualified=qualified,
        expected_report_sha256=expected_report_sha256, validation_scope="external_report_custody_and_closed_observation_contract",
        native_operations_independently_replayed=False, **FALSE)


def _validate_scenario(scenario):
    _need(type(scenario) is dict and set(scenario) == {"checks", "assignment", "observations", "fault_scope"},
        "prelaunch_scenario_fields_differ")
    _need(type(scenario["checks"]) is dict and set(scenario["checks"]) == CHECKS
        and all(value is True for value in scenario["checks"].values()), "prelaunch_checks_differ")
    assignment = scenario["assignment"]
    hashed = {"gate_receipt_sha256", "task_admission_sha256", "allocation_sha256",
        "canonical_repository_path_sha256", "allocated_workspace_path_sha256",
        "canonical_source_sha256", "allocated_source_sha256"}
    _need(type(assignment) is dict and set(assignment) == hashed | {
        "native_task_cid", "portal_task_cid", "task_revision"}, "prelaunch_assignment_fields_differ")
    for key in hashed:
        _need(type(assignment[key]) is str and re.fullmatch(r"[0-9a-f]{64}", assignment[key]),
            "prelaunch_assignment_hash_invalid")
    _need(assignment["canonical_repository_path_sha256"] != assignment["allocated_workspace_path_sha256"]
        and assignment["canonical_source_sha256"] == assignment["allocated_source_sha256"],
        "prelaunch_assignment_join_differs")
    _need(type(assignment["task_revision"]) is int and assignment["task_revision"] >= 0,
        "prelaunch_revision_invalid")
    for key in ("native_task_cid", "portal_task_cid"):
        _need(type(assignment[key]) is str and re.fullmatch(r"[A-Za-z0-9:]{1,256}", assignment[key]),
            "prelaunch_task_identity_invalid")
    _need(assignment["native_task_cid"] != assignment["portal_task_cid"], "prelaunch_task_namespace_confused")
    observation = scenario["observations"]
    _need(type(observation) is dict and set(observation) == {
        "retained_positive_gate_snapshots", "stale_refusals", "lifecycle_fence_advanced",
        "native_claim_release_observed", "kernel_uid_transition_performed",
        "renewed_gate_receipt_sha256", "read_only_native_payload_bytes",
        "authored_fixture_source_provisioned_files", "authored_fixture_source_bytes_unchanged",
        "authored_fixture_executable_bits_unchanged"}, "prelaunch_observation_fields_differ")
    _need(type(observation["retained_positive_gate_snapshots"]) is int and observation["retained_positive_gate_snapshots"] == 2
        and type(observation["stale_refusals"]) is int and observation["stale_refusals"] == 6
        and type(observation["read_only_native_payload_bytes"]) is int
        and 0 < observation["read_only_native_payload_bytes"] <= 8192
        and type(observation["authored_fixture_source_provisioned_files"]) is int
        and observation["authored_fixture_source_provisioned_files"] == 2
        and observation["authored_fixture_source_bytes_unchanged"] is True
        and observation["authored_fixture_executable_bits_unchanged"] is True
        and observation["lifecycle_fence_advanced"] is True
        and observation["native_claim_release_observed"] is True
        and observation["kernel_uid_transition_performed"] is False
        and observation["renewed_gate_receipt_sha256"] == assignment["gate_receipt_sha256"],
        "prelaunch_observation_join_differs")
    _need(scenario["fault_scope"] == "authored_metadata_faults_and_genuine_native_lifecycle_renewal_and_claim_release",
        "prelaunch_fault_scope_differs")


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
        stream.write(_wire(report) + b"\n")
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(schema=SCHEMA, qualified=report["qualified"],
        report_sha256=report_sha256(report), **FALSE), sort_keys=True))
    return 0 if report["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
