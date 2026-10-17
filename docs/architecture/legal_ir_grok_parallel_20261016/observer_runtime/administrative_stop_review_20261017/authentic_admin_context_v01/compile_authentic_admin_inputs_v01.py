"""Compile unbound authentic C/D fixture inputs from actual frozen A/B outputs.

Metadata-only: read source/artifact/database-file bytes, create ONE new private
JSON input; no product/backend imports, database connections, process launches,
native owners, administrative decisions or settlement. The output is explicitly
NOT an authority context: only a future actual ADMIN child may seal its live
gateway identity, kernel creator identity and resulting context CID.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

HERE = Path(__file__).resolve(strict=True).parent
AB = HERE.parent / "unknown-abandonment-authentic-fixture-v01"
AB_MANIFEST_SHA = "244016f11f23d14788da81456825dfb900dc2dc2cd1bb8461ab9ba2e7d28ad31"
MAX_JSON = 2 * 1024**2
MAX_FILE = 64 * 1024**2
SOURCE_RELATIVES = (
    "todo_daemon/planning_unknown_abandonment.py",
    "todo_daemon/planning_unknown_abandonment_settlement.py",
    "todo_daemon/implementation_daemon.py", "task_sources/typed_state_owner.py",
    "task_sources/database_task_source.py", "task_sources/intent_repository.py",
    "task_sources/control_plane_contracts.py", "task_sources/duckdb_state.py",
    "merge/database_coordination.py", "runtime/quack_state_server.py",
    "todo_daemon/native_execution_status_owner.py",
    "todo_daemon/native_execution_wal_maintenance.py",
)


def need(value, reason):
    if not value:
        raise ValueError(reason)


def read(path, maximum=MAX_JSON):
    path = Path(path)
    need(path.is_absolute() and path.resolve(strict=True) == path, "input path alias")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        need(stat.S_ISREG(before.st_mode) and before.st_uid in (0, os.geteuid())
             and before.st_nlink == 1 and not before.st_mode & 0o002
             and 0 <= before.st_size <= maximum, "input custody or bound differs")
        chunks, count = [], 0
        while block := os.read(descriptor, min(65536, maximum - count + 1)):
            count += len(block)
            need(count <= maximum, "input exceeds bound")
            chunks.append(block)
        fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink",
                  "st_size", "st_mtime_ns", "st_ctime_ns")
        witness = lambda row: tuple(getattr(row, field) for field in fields)
        need(count == before.st_size and witness(before) == witness(os.fstat(descriptor))
             == witness(path.lstat()), "input changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    json.dumps(value, allow_nan=False)
    return value


def pin(path, maximum=MAX_JSON):
    raw = read(path, maximum)
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def read_pin(value, maximum=MAX_JSON):
    need(type(value) is dict and set(value) == {"path", "bytes", "sha256"}
         and type(value["bytes"]) is int and 0 <= value["bytes"] <= maximum
         and type(value["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]),
         "pin is not closed")
    raw = read(value["path"], maximum)
    need(len(raw) == value["bytes"] and hashlib.sha256(raw).hexdigest() == value["sha256"], "pin differs")
    return raw


def identity(path):
    info = Path(path).lstat()
    need(stat.S_ISREG(info.st_mode), "store is not a regular file")
    return dict(device=info.st_dev, inode=info.st_ino, size=info.st_size, mtime_ns=info.st_mtime_ns)


def unique_pins(values):
    result = {}
    for value in values:
        prior = result.setdefault(value["path"], value)
        need(prior == value, "same immutable path has different pins")
    return [result[name] for name in sorted(result)]


def compile_input(preparation_path, maintenance_path, admin_worker, settlement_worker):
    manifest_pin = pin(AB / "manifest-v01.json")
    need(manifest_pin["sha256"] == AB_MANIFEST_SHA, "frozen A/B manifest differs")
    manifest = decode(read_pin(manifest_pin))
    for value in manifest["files"]:
        read_pin(value, MAX_FILE)
    preparation_pin, maintenance_pin = pin(preparation_path), pin(maintenance_path)
    preparation = decode(read_pin(preparation_pin))
    maintenance = decode(read_pin(maintenance_pin))
    for record, operation in ((preparation, "prepare_launch"), (maintenance, "maintenance")):
        need(record.get("schema") == "authentic-native-fixture-direct-owner-result@1"
             and record.get("fixture_only") is True and record.get("operation") == operation
             and record.get("original_v05_main_impersonated") is False,
             "genuine A/B direct owner result differs")
    prepared, maintained = preparation["result"], maintenance["result"]
    producer_request_pin = prepared["producer_request_pin"]
    producer_request = decode(read_pin(producer_request_pin))
    need(producer_request.get("profile") == maintained.get("profile") == "two_attempt"
         and maintained.get("actual_live_F07_guard") is True
         and maintained.get("fixture_custody_seam_used") is False
         and maintained.get("both_UNKNOWN_rows_retained") is True
         and maintained.get("administrative_decision_performed") is False,
         "authentic two-attempt physical maintenance is required")
    root = Path(producer_request["fixture_root"])
    need(root.is_absolute() and root.resolve(strict=True) == root
         and root.name.startswith("legal-ir-unknown-authentic-fixture-")
         and "legal-ir-grok-parallel-20261016-r06" not in str(root), "only authentic disposable fixture allowed")
    custody = maintained["custody_pins"]
    need(type(custody) is dict and set(custody) == {
        "launch_request", "native_owner_receipt", "producer_result", "master_terminal"}, "A custody slots differ")
    produced = decode(read_pin(custody["producer_result"]))
    need(custody["producer_result"]["path"] == str(root / "producer-result.json")
         and produced.get("schema") == "authentic-native-unknown-master-producer-result@1"
         and produced.get("fixture_only") is True and produced.get("profile") == "two_attempt"
         and produced.get("production_store_opened") is False
         and produced.get("request_pin") == producer_request_pin
         and produced.get("controller_head") == producer_request["controller_head"], "A producer crosswalk differs")
    snapshot_pin = produced["snapshot_pin"]
    snapshot = decode(read_pin(snapshot_pin, MAX_FILE))
    stores = produced["stores"]
    expected = {"canonical": str(root / "canonical.duckdb"), "coordination": str(root / "coordination.duckdb"),
                "execution": str(root / "current-lane/crash-copy.execution.shard-0001-of-0003.duckdb")}
    need(stores == expected and len(snapshot["historical_unknowns"]) == 1
         and snapshot["attempt"]["attempt_number"] == 2, "same native stores/complete history required")
    for value in produced["closed_store_pins"].values():
        read_pin(value, MAX_FILE)
        need(not Path(value["path"] + ".wal").exists(), "closed canonical/coordination WAL present")
    for history in snapshot["historical_unknowns"]:
        for value in history["store_pins"]:
            read_pin(value, MAX_FILE)
    artifacts = {key: custody[key] for key in ("launch_request", "native_owner_receipt", "master_terminal")}
    artifacts.update({"status_binding": maintained["status_binding_pin"],
        "status_observation": maintained["status_observation_pin"],
        "maintenance_binding": maintained["maintenance_binding_pin"],
        "maintenance_receipt": maintained["maintenance_receipt_pin"]})
    decoded = {key: decode(read_pin(value)) for key, value in artifacts.items()}
    binding, observation = decoded["status_binding"], decoded["status_observation"]
    need(binding["execution_path"] == stores["execution"]
         and binding["task_cid"] == snapshot["task"]["task_cid"]
         and binding["control_revision"] == snapshot["task"]["revision"]
         and binding["expected_control_status"] == "in_progress"
         and identity(stores["execution"]) == binding["execution_file_identity"]
         == observation["execution_file_identity"]
         and not Path(stores["execution"] + ".wal").exists(), "post-B current execution generation differs")
    product_sources = producer_request["product_sources"]
    package = Path(producer_request["repo_root"]) / "ipfs_accelerate_py/agent_supervisor"
    source_pins = [product_sources[str(package / relative)] for relative in SOURCE_RELATIVES]
    for value in source_pins:
        read_pin(value, 8 * 1024**2)
    workers = {"admin": pin(admin_worker, 8 * 1024**2), "settlement": pin(settlement_worker, 8 * 1024**2)}
    forbidden = HERE.parent / "unknown-abandonment-implementation-v01/integration-v01/native_unknown_admin_owner_fixture.py"
    need(all(value["path"] != str(forbidden) for value in workers.values())
         and workers["admin"]["path"] != workers["settlement"]["path"],
         "new authentic workers required; nonadmitting child cannot be invoked")
    immutable = unique_pins([manifest_pin, preparation_pin, maintenance_pin, producer_request_pin,
        custody["producer_result"], snapshot_pin, *artifacts.values(), *workers.values(), pin(Path(__file__)),
        producer_request["producer_pin"], producer_request["source_loader_pin"],
        producer_request["published_loader_pin"], producer_request["duckdb_installation_pin"],
        maintained["actual_A_proof_pin"]])
    stopped = list(maintained["original_owner_and_child_master_stopped"])
    maintenance_owner = {"pid": maintenance["owner_pid"],
        "start_time_ticks": maintenance["owner_start_time_ticks"], "boot_id": maintenance["boot_id"]}
    if maintenance_owner not in stopped:
        stopped.append(maintenance_owner)
    preimage = {"schema": "planning-unknown-abandonment-authority-context@1",
        "task_cid": snapshot["task"]["task_cid"], "namespace": produced["board_namespace"],
        "expected_task_revision": snapshot["task"]["revision"], "logical_store_id": binding["control_store_id"],
        "stores": stores, "snapshot_pin": snapshot_pin, "source_pins": source_pins, "immutable_pins": immutable,
        # A's execution_pin is deliberately not reused: B checkpointed the file.
        "closed_store_pins": {role: pin(Path(stores[role]), MAX_FILE) for role in ("execution", "coordination")},
        "stopped_processes": stopped,
        "operator_decision": {"decision": "administratively_abandon_local_execution_authority",
            "provider_outcome": "unknown", "possible_external_effects": "unverified", "callback_closed": False,
            "remote_cancellation_proved": False, "nonreachability_asserted": False,
            "native_execution_settlement_performed_by_this_operation": False,
            "automatic_retry_admitted": False, "completion_authority": False,
            "unknown_effect_risk_accepted": True,
            "reason": "DISPOSABLE AUTHENTIC FIXTURE ONLY: qualify local UNKNOWN administrative disposition; no production decision"},
        "custody_artifacts": artifacts}
    # Input completeness is checked here without pretending to execute policy.
    need(len(json.dumps(preimage, ensure_ascii=False, separators=(",", ":")).encode()) <= 60 * 1024,
         "unbound context leaves insufficient room under native64KiB bound")
    return {"schema": "authentic-native-unknown-admin-unbound-input@1", "fixture_only": True,
        "production_access": False, "authority_context_sealed": False, "policy_verified": False,
        "runtime_workers_qualified": False, "input_compiler_pin": pin(Path(__file__)), "AB_manifest_pin": manifest_pin,
        "repo_root": producer_request["repo_root"], "controller_head": producer_request["controller_head"],
        "fixture_root": str(root), "product_sources": product_sources,
        "producer_pin": producer_request["producer_pin"], "python_pin": producer_request["python_pin"],
        "source_loader_pin": producer_request["source_loader_pin"],
        "published_loader_pin": producer_request["published_loader_pin"],
        "duckdb_installation_pin": producer_request["duckdb_installation_pin"], "worker_pins": workers,
        "canonical_pre_admin_pin": produced["closed_store_pins"]["canonical"], "context_preimage": preimage,
        "missing_live_fields": ["registered_owner_identity", "registered_creator", "context_cid"],
        "mandatory_parent_gate": "Require actual A/B outer owner exit and genuine reservation finalization before C; these JSON records alone do not establish that gate."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparation-result", type=Path, required=True)
    parser.add_argument("--maintenance-result", type=Path, required=True)
    parser.add_argument("--admin-worker-source", type=Path, required=True)
    parser.add_argument("--settlement-worker-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = compile_input(args.preparation_result, args.maintenance_result,
                          args.admin_worker_source, args.settlement_worker_source)
    output = args.output
    need(output.is_absolute() and output.parent.resolve(strict=True) == output.parent and not output.exists(),
         "output must be NEW absolute private file")
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()
    need(len(raw) <= MAX_JSON, "unbound input exceeds bound")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            need(count > 0, "output short write")
            offset += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    print(json.dumps({"unbound_input_pin": pin(output), "metadata_only": True,
        "authority_context_sealed": False, "administrative_decision_performed": False}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"input_compilation_failed": True, "error_type": type(error).__name__}), file=sys.stderr)
        raise SystemExit(2)
