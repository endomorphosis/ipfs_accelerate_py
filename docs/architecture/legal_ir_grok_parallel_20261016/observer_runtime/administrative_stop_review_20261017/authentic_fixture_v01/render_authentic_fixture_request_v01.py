"""Metadata-only renderer for future parent-owned authentic fixture jobs.

Reads pinned source/JSON bytes, writes one NEW private request. It imports no
product/backend, opens no database, starts no process and acquires no native
owner. Root must validate/freeze each rendered request and serialize execution.
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
OWNER = Path("/home/barberb/lift_coding/artifacts/legal-ir-supervisor-start-20261016-01/run_native_owner_v05.py")
OWNER_SHA = "3e5357925d3be41630e73adc0b8a31b1fd25c35ad3f5bc4ad1b01dbaeffebb14"
DATASETS = Path("/home/barberb/lift_coding/artifacts/legal-conditions-composition-diagnostic-20261015-01/datasets")
LOADER = HERE.parent / "unknown-abandonment-implementation-v01/integration-v01/native_unknown_admin_owner_fixture.py"
WRAPPER = HERE / "authentic_fixture_owner_v01.py"
PRODUCER = HERE / "produce_authentic_unknown_master_v01.py"
RESOURCE_FILES = ("existing_resource_scheduler.py", "resource_scheduler.py",
                  "parented_phase_resources.py", "autoencoder_daemon_resources.py")


def need(value, reason):
    if not value:
        raise ValueError(reason)


def read(path, maximum=8 * 1024**2):
    path = Path(path)
    need(path.is_absolute() and path.resolve(strict=True) == path, "request source path alias")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        need(stat.S_ISREG(before.st_mode) and before.st_uid in (0, os.geteuid())
             and before.st_nlink == 1 and not before.st_mode & 0o002 and 0 <= before.st_size <= maximum,
             "request source custody/bound differs")
        chunks, count = [], 0
        while block := os.read(descriptor, min(65536, maximum - count + 1)):
            count += len(block)
            need(count <= maximum, "request source too large")
            chunks.append(block)
        fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
        witness = lambda row: tuple(getattr(row, name) for name in fields)
        need(count == before.st_size and witness(before) == witness(os.fstat(descriptor)) == witness(path.lstat()),
             "request source changed")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def pin(path, maximum=8 * 1024**2):
    raw = read(path, maximum)
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def read_pin(value, maximum=8 * 1024**2):
    need(type(value) is dict and set(value) == {"path", "bytes", "sha256"}
         and type(value["bytes"]) is int, "request pin is not closed")
    raw = read(value["path"], maximum)
    need(len(raw) == value["bytes"] and hashlib.sha256(raw).hexdigest() == value["sha256"], "request pin differs")
    return raw


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate request JSON key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs)
    json.dumps(value, allow_nan=False)
    return value


def load(path):
    return decode(read(path, 2 * 1024**2))


def base(repo, head, fixture, attempt, operation):
    need(type(head) is str and re.fullmatch(r"[0-9a-f]{40}", head), "qualified HEAD must be explicit")
    repo, fixture, attempt = map(Path, (repo, fixture, attempt))
    need(repo.is_absolute() and repo.resolve(strict=True) == repo and fixture.is_absolute()
         and fixture.parent.resolve(strict=True) == fixture.parent
         and fixture.name.startswith("legal-ir-unknown-authentic-fixture-")
         and attempt.is_absolute() and attempt.parent.resolve(strict=True) == attempt.parent and not attempt.exists(),
         "new fixture/attempt paths differ")
    owner_pin = pin(OWNER)
    need(owner_pin["sha256"] == OWNER_SHA, "original owner v05 changed")
    prefix = DATASETS / "ipfs_datasets_py/optimizers/logic_theorem_optimizer"
    return {"schema": "authentic-native-fixture-owner-request@1", "operation": operation,
        "fixture_only": True, "production_access": False, "owner_helper_pin": owner_pin,
        "wrapper_pin": pin(WRAPPER), "source_loader_pin": pin(LOADER),
        "resource_source_pins": [pin(prefix / name) for name in RESOURCE_FILES],
        "repo_root": str(repo), "controller_head": head, "fixture_root": str(fixture),
        "attempt_directory": str(attempt), "storage_bytes": 256 * 1024**2,
        "external_charges": [{"path": str(fixture), "bytes": 128 * 1024**2}],
        "duration_seconds": 120, "payload": {}}


def prepared(path):
    value = load(path)
    need(value.get("schema") == "authentic-native-fixture-direct-owner-result@1"
         and value.get("fixture_only") is True and value.get("operation") == "prepare_launch"
         and value.get("original_v05_main_impersonated") is False, "actual preparation result differs")
    result = value["result"]
    producer = decode(read_pin(result["producer_request_pin"], 2 * 1024**2))
    launch = decode(read_pin(result["original_v05_launch_request_pin"], 2 * 1024**2))
    need(producer["fixture_root"] + "/current-lane/master.pid.terminal.json" == launch["native_terminal_receipt"],
         "prepared exact producer/original launch crosswalk differs")
    return result, producer, launch


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="operation", required=True)
    pre = sub.add_parser("prepare")
    pre.add_argument("--repo-root", type=Path, required=True)
    pre.add_argument("--controller-head", required=True)
    pre.add_argument("--fixture-root", type=Path, required=True)
    pre.add_argument("--prepare-attempt", type=Path, required=True)
    pre.add_argument("--launch-attempt", type=Path, required=True)
    pre.add_argument("--profile", choices=("one_attempt", "two_attempt"), required=True)
    pre.add_argument("--board-namespace", required=True)
    pre.add_argument("--output", type=Path, required=True)
    for mode in ("launch", "maintenance"):
        command = sub.add_parser(mode)
        command.add_argument("--preparation-result", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if mode == "launch":
            command.add_argument("--preflight-owner-receipt", type=Path, required=True)
        else:
            command.add_argument("--launch-owner-receipt", type=Path, required=True)
            command.add_argument("--maintenance-attempt", type=Path, required=True)
            command.add_argument("--logical-store-id", required=True)
    args = parser.parse_args()
    if args.operation == "prepare":
        need(not args.fixture_root.exists() and not args.launch_attempt.exists(), "fixture/original launch outputs already exist")
        value = base(args.repo_root, args.controller_head, args.fixture_root, args.prepare_attempt, "prepare_launch")
        value["payload"] = {"producer_pin": pin(PRODUCER), "profile": args.profile,
            "board_namespace": args.board_namespace, "master_duration_seconds": 1,
            "launch_attempt_directory": str(args.launch_attempt)}
    else:
        result, producer, launch = prepared(args.preparation_result)
        if args.operation == "launch":
            value = base(producer["repo_root"], producer["controller_head"], producer["fixture_root"],
                         launch["attempt_directory"], "launch")
            preflight_request_pin = result["original_v05_producer_preflight_request_pin"]
            preflight_producer = decode(read_pin(result["producer_preflight_request_pin"], 2 * 1024**2))
            value["payload"] = {"original_launch_request_pin": result["original_v05_launch_request_pin"],
                "preflight_owner_request_pin": preflight_request_pin,
                "preflight_owner_receipt_pin": pin(args.preflight_owner_receipt, 2 * 1024**2),
                "preflight_producer_result_pin": pin(Path(preflight_producer["fixture_root"]) / "producer-result.json", 2 * 1024**2)}
        else:
            value = base(producer["repo_root"], producer["controller_head"], producer["fixture_root"],
                         args.maintenance_attempt, "maintenance")
            root = Path(producer["fixture_root"])
            a_proof = [item for item in result["actual_owner_A_source_pins"]
                       if item["path"].endswith("/proof/formal_verification_contracts.py")]
            need(len(a_proof) == 1, "actual A original owner proof source is ambiguous")
            value["payload"] = {"logical_store_id": args.logical_store_id,
                "product_sources": producer["product_sources"], "original_A_proof_pin": a_proof[0],
                "custody_pins": {"launch_request": result["original_v05_launch_request_pin"],
                    "native_owner_receipt": pin(args.launch_owner_receipt, 2 * 1024**2),
                    "producer_result": pin(root / "producer-result.json", 2 * 1024**2),
                    "master_terminal": pin(root / "current-lane/master.pid.terminal.json", 2 * 1024**2)}}
    output = args.output
    need(output.is_absolute() and output.parent.resolve(strict=True) == output.parent and not output.exists(),
         "request output must be new absolute private file")
    raw = (json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()
    need(len(raw) <= 2 * 1024**2, "request output exceeds native producer bound")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            need(count > 0, "request output short write")
            offset += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    print(json.dumps({"request_pin": pin(output, 2 * 1024**2), "metadata_only": True,
                      "native_execution_performed": False}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"request_render_failed": True, "error_type": type(error).__name__}), file=sys.stderr)
        raise SystemExit(2)
