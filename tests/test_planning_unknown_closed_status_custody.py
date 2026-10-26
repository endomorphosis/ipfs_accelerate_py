"""Pure decision regressions; no native store, receipt or custody is qualified.

These use explicitly inert binding/receipt test data and narrowly simulated
/proc outcomes. The separate genuine status-owner gate must establish actual
read/close, handles, locks, bytes and resource custody before production audit.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import planning_unknown_abandonment as policy


def _pin(path):
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _json(tmp_path, name, value):
    path = tmp_path / name
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return _pin(path)


def _identity(inode):
    return {"device": 1, "inode": inode, "size": 0, "mtime_ns": 0}


@pytest.fixture
def owner_absence(monkeypatch):
    """Simulated /proc decision data, explicitly not an owner-stop witness."""
    current_boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    owner = {"schema": "legal-ir-native-orchestration-receipt@1", "owner_pid": 1073741700,
        "child_pid": 1073741701, "native_stop_verified": True, "owned_group_quiescent": True,
        "cleanup_pending": False, "completion_authority": False,
        "native_fence_proof": {"all_trees_fenced": True, "master_pid": 1073741702}}
    tag = {"pid": owner["owner_pid"], "observation": "pid_absent_unrecorded_birth", "observed_boot_id": current_boot}
    rows = [tag, {"pid": owner["child_pid"], "start_time_ticks": 101, "boot_id": "unit-test-other-boot"},
        {"pid": owner["native_fence_proof"]["master_pid"], "start_time_ticks": 102, "boot_id": "unit-test-other-boot"}]
    present = {"value": False}
    original = Path.lstat

    def lstat(path, *args, **kwargs):
        if str(path) == f"/proc/{owner['owner_pid']}":
            if present["value"]:
                return SimpleNamespace(st_mode=0o40555)
            raise FileNotFoundError(str(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "lstat", lstat)
    return owner, rows, present


def test_exact_tagged_original_owner_absence_is_a_distinct_decision(owner_absence):
    owner, rows, _ = owner_absence
    policy.stopped_processes_v2(rows, original_owner=owner)


def test_default_stopped_process_contract_still_requires_positive_birth(owner_absence):
    _, rows, _ = owner_absence
    with pytest.raises(policy.PlanningUnknownAbandonmentError):
        policy.stopped_processes(rows)


def test_positive_births_do_not_require_an_absence_tag(owner_absence):
    _, rows, _ = owner_absence
    policy.stopped_processes(rows[1:])


@pytest.mark.parametrize("case", [
    "owner_pid_present", "foreign_owner_tag", "different_observed_boot", "known_owner_birth",
    "tagged_child", "tagged_master", "duplicate_tag", "duplicate_positive_owner",
    "missing_child_master", "extra_tag_field", "bool_pid", "unsafe_receipt",
])
def test_unrecorded_owner_decision_refuses_ambiguous_identity(owner_absence, case):
    owner, rows, present = owner_absence
    owner, rows = deepcopy(owner), deepcopy(rows)
    if case == "owner_pid_present":
        # Presence alone refuses: no inferred reuse, zombie or birth comparison.
        present["value"] = True
    elif case == "foreign_owner_tag":
        rows[0]["pid"] += 20
    elif case == "different_observed_boot":
        rows[0]["observed_boot_id"] = "unit-test-other-boot"
    elif case == "known_owner_birth":
        owner["owner_birth_marker"] = "100"
    elif case == "tagged_child":
        owner["owner_pid"] = owner["child_pid"]
        rows[0]["pid"] = owner["owner_pid"]
    elif case == "tagged_master":
        owner["owner_pid"] = owner["native_fence_proof"]["master_pid"]
        rows[0]["pid"] = owner["owner_pid"]
    elif case == "duplicate_tag":
        rows.append(deepcopy(rows[0]))
    elif case == "duplicate_positive_owner":
        rows.append({"pid": owner["owner_pid"], "start_time_ticks": 100, "boot_id": "unit-test-other-boot"})
    elif case == "missing_child_master":
        rows = rows[:1]
    elif case == "extra_tag_field":
        rows[0]["start_time_ticks"] = None
    elif case == "bool_pid":
        rows[0]["pid"] = True
    elif case == "unsafe_receipt":
        owner["native_stop_verified"] = False
    with pytest.raises(policy.PlanningUnknownAbandonmentError):
        policy.stopped_processes_v2(rows, original_owner=owner)


@pytest.fixture
def closed_receipt(tmp_path):
    """Closed test records for validator decisions; never native admission."""
    database = tmp_path / "not-a-native-database.bin"
    database.write_bytes(b"inert unit-test bytes, never opened as a backend")
    database_pin = _pin(database)
    identity = _identity(11)
    identity["size"] = database_pin["bytes"]
    writer, lock = _identity(12), _identity(13)
    native_metadata = {"authority_mode": "quack", "execution_state_dir": str(tmp_path),
        "execution_state_prefix": "unit", "interface": "DatabaseImplementationDaemon@1",
        "logical_owner_session_id": "unit-owner", "process_instance_id": "process:unit",
        "schema": "ipfs_accelerate_py/agent-supervisor/database-implementation-daemon@1",
        "strict_task_sharding": "true", "task_shard_count": "3", "task_shard_index": "2",
        "state_schema_revision": "5", "control_schema_profile_id": "", "control_schema_fingerprint": ""}
    raw = {"schema": "legal-ir-attempt2-private-raw-native-observation/v1", "read_only": True, "recovery_authority": False,
        "schema_installation_attempted": False, "production_handle_write_connection_opened": False,
        "task_cid": "unit-task", "attempt_id": "unit-attempt",
        "database_pins": {"unit_execution": database_pin},
        "database_file_identities": {"unit_execution": identity}, "wal_pins": {}, "wal_file_identities": {},
        "execution": {"unit_execution": {"daemon_execution_metadata": [
            {"key": key, "value": value} for key, value in sorted(native_metadata.items())]}}}
    raw_pin = _json(tmp_path, "raw-observation.json", raw)
    snapshot_pin = _json(tmp_path, "snapshot.json", {"unit_test_decision_data": True})
    auth_pin = _json(tmp_path, "authorization.json", {"unit_test_decision_data": True})
    artifacts = {key: _json(tmp_path, key + ".json", {"unit_test_decision_data": key}) for key in (
        "launch_request", "native_owner_receipt", "master_terminal", "status_binding", "status_observation")}
    source_root = Path(policy.__file__).resolve().parents[1]
    source_pins = [_pin(source_root / relative) for relative in (
        "todo_daemon/native_execution_status_owner.py", "todo_daemon/implementation_daemon.py", "task_sources/duckdb_state.py")]
    adapter = tmp_path / "production_unknown_closed_status_owner_v01.py"
    adapter.write_text("# Inert unit-test source, never imported or executed.\n", encoding="utf-8")
    source_pins.append(_pin(adapter))
    owner = {"pid": 1073741600, "start_time_ticks": 99, "boot_id": "unit-test-other-boot",
        "process_birth_id": "birth:kernel:" + hashlib.sha256(b"1073741600:99").hexdigest()[:32]}
    context = {"task_cid": "unit-task", "stores": {"execution": str(database)}, "custody_artifacts": artifacts,
        "snapshot_pin": snapshot_pin, "closed_store_pins": {"execution": database_pin},
        "immutable_pins": [raw_pin, auth_pin, snapshot_pin], "source_pins": source_pins,
        "stopped_processes": [{key: owner[key] for key in ("pid", "start_time_ticks", "boot_id")}]}
    binding = SimpleNamespace(binding_cid="sha256:unit-binding", execution_file_identity=identity,
        writer_lock_identity=writer, expected_metadata=native_metadata)
    receipt = {"schema": policy.CLOSED_STATUS_RECEIPT_SCHEMA, "fixture_only": False, "production_access": True,
        "read_only": True, "status_binding_cid": binding.binding_cid,
        "status_binding_pin": artifacts["status_binding"], "status_observation_pin": artifacts["status_observation"],
        "raw_observation_pin": raw_pin, "authorization_pin": auth_pin, "snapshot_pin": snapshot_pin,
        "execution_path": str(database), "original_launch_request_pin": artifacts["launch_request"],
        "native_owner_receipt_pin": artifacts["native_owner_receipt"], "master_terminal_pin": artifacts["master_terminal"],
        "database_pin_before": database_pin, "database_pin_after": database_pin,
        "execution_file_identity_before": identity, "execution_file_identity_after": identity,
        "writer_lock_identity_before": writer, "writer_lock_identity_after": writer,
        "policy_lock_identity_before": lock, "policy_lock_identity_after": lock,
        "wal_absent_before": True, "wal_absent_after": True, "expected_native_rows_matched": True,
        "all_metadata_preserved": True, "provider_unknown_not_settled": True,
        "connection_close_verified": True, "original_locks_released": True,
        "physical_maintenance_performed": False, "logical_write_performed": False, "backup_created": False,
        "completion_authority": False, "semantic_authority": False, "automatic_retry_admitted": False,
        "callback_closed": False, "remote_cancellation_proved": False, "nonreachability_asserted": False,
        "source_pins": source_pins, "owner_process": owner, "root_lease_id": "unit-root",
        "phase_lease_id": "unit-phase", "disk_reservation_id": "unit-disk", "actual_parented_native_lease_id": "unit-native"}
    decoded = {"closed_store_status_receipt": receipt, "status_observation": {"native_metadata": native_metadata}}
    snapshot = {"attempt": {"attempt_id": "unit-attempt"}}
    return context, binding, receipt, decoded, raw, snapshot


def _seal(receipt):
    receipt["receipt_cid"] = policy.digest({key: value for key, value in receipt.items() if key != "receipt_cid"})


def test_closed_status_receipt_accepts_only_its_explicit_decision_shape(closed_receipt):
    context, binding, receipt, decoded, _, snapshot = closed_receipt
    _seal(receipt)
    policy._validate_closed_status_receipt_v2(context, snapshot, decoded, binding, check_current_file=False)


@pytest.mark.parametrize("case", [
    "maintenance_relabel", "missing_read_only", "integer_true", "logical_write_claim", "backup_claim",
    "different_snapshot", "database_bytes_drift", "execution_inode_drift", "bool_inode",
    "writer_lock_drift", "policy_lock_drift", "different_native_metadata", "omitted_owner_stop",
])
def test_closed_status_receipt_refuses_false_or_unbound_claims(closed_receipt, case):
    context, binding, receipt, decoded, _, snapshot = closed_receipt
    if case == "maintenance_relabel":
        receipt["schema"] = "native-execution-wal-maintenance-receipt@1"
    elif case == "missing_read_only":
        receipt.pop("read_only")
    elif case == "integer_true":
        receipt["wal_absent_after"] = 1
    elif case == "logical_write_claim":
        receipt["logical_write_performed"] = True
    elif case == "backup_claim":
        receipt["backup_created"] = True
    elif case == "different_snapshot":
        receipt["snapshot_pin"] = receipt["authorization_pin"]
    elif case == "database_bytes_drift":
        receipt["database_pin_after"] = {**receipt["database_pin_after"], "sha256": "0" * 64}
    elif case == "execution_inode_drift":
        receipt["execution_file_identity_after"] = {**receipt["execution_file_identity_after"], "inode": 44}
    elif case == "bool_inode":
        receipt["execution_file_identity_before"] = {**receipt["execution_file_identity_before"], "inode": True}
        receipt["execution_file_identity_after"] = receipt["execution_file_identity_before"]
        binding.execution_file_identity = receipt["execution_file_identity_before"]
    elif case == "writer_lock_drift":
        receipt["writer_lock_identity_after"] = {**receipt["writer_lock_identity_after"], "inode": 45}
    elif case == "policy_lock_drift":
        receipt["policy_lock_identity_after"] = {**receipt["policy_lock_identity_after"], "inode": 46}
    elif case == "different_native_metadata":
        decoded["status_observation"]["native_metadata"] = {**decoded["status_observation"]["native_metadata"], "logical_owner_session_id": "foreign-owner"}
    elif case == "omitted_owner_stop":
        context["stopped_processes"] = []
    _seal(receipt)
    with pytest.raises(policy.PlanningUnknownAbandonmentError):
        policy._validate_closed_status_receipt_v2(context, snapshot, decoded, binding, check_current_file=False)


def test_current_closed_status_refuses_a_new_wal(closed_receipt):
    context, binding, receipt, decoded, _, snapshot = closed_receipt
    _seal(receipt)
    Path(context["stores"]["execution"] + ".wal").write_bytes(b"unit-test marker, never replayed")
    with pytest.raises(policy.PlanningUnknownAbandonmentError):
        policy._validate_closed_status_receipt_v2(context, snapshot, decoded, binding, check_current_file=True)


def test_later_settlement_does_not_rehash_mutable_execution_bytes(closed_receipt):
    context, binding, receipt, decoded, _, snapshot = closed_receipt
    _seal(receipt)
    Path(context["stores"]["execution"]).write_bytes(b"later staged bytes, not native execution")
    policy._validate_closed_status_receipt_v2(context, snapshot, decoded, binding, check_current_file=False)


def test_v1_still_rejects_the_six_artifact_closed_status_population():
    context = {"schema": policy.AUTHORITY_SCHEMA, "custody_artifacts": {
        name: {} for name in ("launch_request", "native_owner_receipt", "master_terminal",
            "status_binding", "status_observation", "closed_store_status_receipt")}}
    with pytest.raises(policy.PlanningUnknownAbandonmentError, match="required named native custody"):
        policy.verify_custody(context, {}, check_current_file=False)


def test_v2_does_not_accept_the_seven_artifact_maintenance_population():
    context = {"schema": policy.CLOSED_AUTHORITY_SCHEMA, "custody_artifacts": {
        name: {} for name in ("launch_request", "native_owner_receipt", "master_terminal",
            "status_binding", "status_observation", "maintenance_binding", "maintenance_receipt")}}
    with pytest.raises(policy.PlanningUnknownAbandonmentError, match="required named native custody"):
        policy.verify_custody(context, {}, check_current_file=False)
