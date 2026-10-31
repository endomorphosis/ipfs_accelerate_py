"""Maintained finite validation and deferred activation owners.

Only the public D resource CLI allocates validation resources. A distinct trusted
worker checks natural tests and owns source READ custody; its result is joined
with the original resource terminal, real command wait, and durable artifacts.
The adoption command stages an immutable release. A separate serialized service
may cut it over after the original repair job has genuinely closed. Verifiers
and resume never repeat an owner command or a service action.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import ExitStack
import fcntl
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

REPAIR_PROFILE = "legal-ir"

POLICY_SCHEMA = "agent-supervisor/repair-public-owner-policy@1"
WORK_SCHEMA = "agent-supervisor/repair-trusted-validation@1"
DEFERRED_SCHEMA = "agent-supervisor/repair-adoption-deferred@1"
ACTIVATION_SCHEMA = "agent-supervisor/repair-actual-activation@1"
VALIDATION_SCHEMA = "agent-supervisor/isolated-repair-candidate-validation@1"
ADOPTION_SCHEMA = "agent-supervisor/isolated-repair-candidate-adoption@1"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ORIGINAL_POPEN = subprocess.Popen


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def ordinary_bytes(path, maximum=8 * 1024**2):
    """Stdlib-only bootstrap read; no project package may precede the guard."""
    path = Path(path)
    need(path.is_absolute() and ".." not in path.parts
         and not any(p.is_symlink() for p in (path, *path.parents)), "unaliased absolute owner file required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid() and before.st_nlink == 1
             and 0 <= before.st_size <= maximum, "bounded ordinary owner file required")
        raw = bytearray()
        while len(raw) <= maximum:
            part = os.read(fd, min(65536, maximum + 1 - len(raw)))
            if not part:
                break
            raw.extend(part)
        stamp = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_nlink,
                           s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        need(len(raw) == before.st_size and len(raw) <= maximum
             and stamp(before) == stamp(os.fstat(fd)) == stamp(path.lstat()), "owner file changed while read")
        return bytes(raw)
    finally:
        os.close(fd)


def descriptor(path):
    path = Path(path); raw = ordinary_bytes(path)
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def checked_descriptor(pin):
    need(type(pin) is dict and set(pin) == {"path", "bytes", "sha256"}
         and type(pin["bytes"]) is int and 0 <= pin["bytes"] <= 8 * 1024**2
         and type(pin["sha256"]) is str and _SHA.fullmatch(pin["sha256"]), "closed typed owner descriptor required")
    raw = ordinary_bytes(Path(pin["path"]))
    need(len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["sha256"], "owner descriptor changed")
    return raw


def json_object(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate owner JSON key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite owner JSON")))
    need(type(value) is dict, "owner JSON object required")
    return value


def require_current_authorization(*args, **kwargs):
    # Deliberately lazy: validation-worker must first install its source guard
    # and normal candidate package search, before ANY project initialization.
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_authorization import require_current_authorization as require
    return require(*args, **kwargs)


def need(value, message):
    if not value:
        raise ValueError(message)


def new_json(path: Path, value) -> dict:
    raw = canonical(value) + b"\n"
    need(len(raw) <= 8 * 1024**2, "owner metadata bound exceeded")
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o400)
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return descriptor(path)


def proc_identity(pid):
    need(type(pid) is int and pid > 0, "positive process ID required")
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return None
    return {"pid": pid, "birth": fields[19]}


def checked_policy(path: Path, sha: str) -> dict:
    need(_SHA.fullmatch(sha) is not None, "owner policy hash required")
    raw = ordinary_bytes(path)
    need(hashlib.sha256(raw).hexdigest() == sha, "owner policy changed")
    value = json_object(raw)
    need(set(value) == {"schema", "datasets_root", "python", "source_descriptors",
        "validation", "adoption"} and value["schema"] == POLICY_SCHEMA,
        "closed maintained owner policy required")
    sources = value["source_descriptors"]
    need(type(sources) is list and 0 < len(sources) <= 128
         and len({r["path"] for r in sources}) == len(sources), "finite owner source population required")
    for row in sources:
        checked_descriptor(row)
    need(descriptor(Path(__file__).absolute()) in sources, "actual maintained owner source is not pinned")
    need(descriptor(Path(__file__).with_name("repair_validation_worker.py").absolute()) in sources, "trusted stdlib-only worker is not pinned")
    d = Path(value["datasets_root"])
    need(d.is_absolute() and not any(p.is_symlink() for p in (d, *d.parents)), "canonical D root required")
    required = {str(d / "scripts/ops/legal_ir/run_owned_validation.py"),
        str(d / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_source_lease.py"),
        str(d / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_resource_recovery.py"),
        str(d / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/validation_run_resources.py")}
    need(required <= {r["path"] for r in sources}, "public D owner, source lease and storage disposition inspector must be pinned")
    recovery_path = str(d / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_resource_recovery.py")
    recovery_pin = next(pin for pin in sources if pin["path"] == recovery_path)
    # Metadata-only interface preflight: deploying a profile without its public
    # negative-storage observer must fail before any coding attempt is spent.
    declared = {node.name for node in ast.parse(checked_descriptor(recovery_pin)).body if isinstance(node, ast.FunctionDef)}
    need({"inspect_validation_reservation", "inspect_validation_disposition"} <= declared,
         "qualified public negative-storage observation API is unavailable")
    need(Path(value["python"]).is_absolute(), "exact Python executable required")
    validation = value["validation"]
    need(type(validation) is dict and set(validation) == {"trusted_tests", "selected_checks", "pytest_module", "import_sources", "attempt_root",
        "cpu_slots", "memory_mb", "storage_bytes", "timeout_seconds", "child_process_slots",
        "scheduler_state_path", "expected_scheduler_policy_sha256", "roots", "resource_ledger"},
        "closed finite validation profile required")
    for name, maximum in (("cpu_slots", 1), ("memory_mb", 2048),
            ("storage_bytes", 67108864), ("timeout_seconds", 300), ("child_process_slots", 2)):
        need(type(validation[name]) is int and 0 < validation[name] <= maximum, "finite public validation envelope required")
    tests, checks = validation["trusted_tests"], validation["selected_checks"]
    checked_descriptor(validation["pytest_module"])
    need(type(validation["import_sources"]) is list and len(validation["import_sources"]) <= 4096,
         "finite complete normal import manifest required")
    for pin in validation["import_sources"]:
        checked_descriptor(pin)
    pytest_support = str(Path(validation["pytest_module"]["path"]).parent.parent / "_pytest/python.py")
    need(any(pin["path"] == pytest_support for pin in validation["import_sources"]),
         "original Pytest collection wrapper source must be pinned for attributed syntax failures")
    need(type(tests) is list and 0 < len(tests) <= 32, "finite maintained tests required")
    for pin in tests:
        checked_descriptor(pin)
    need(type(checks) is list and 0 < len(checks) <= 128 and len(set(checks)) == len(checks), "exact natural test nodes required")
    need(all(type(node) is str and "::" in node and node.split("::", 1)[0] in {r["path"] for r in tests}
             for node in checks), "only pinned maintained test nodes may execute")
    maintained = Path(__file__).resolve().parents[3]
    need(all(any((maintained / name) in Path(pin["path"]).parents for name in ("test", "tests")) for pin in tests),
         "trusted tests must be maintained repository tests, not historical controls")
    need(type(validation["roots"]) is list and 0 < len(validation["roots"]) <= 4
         and all(Path(r).is_absolute() for r in validation["roots"]), "original finite named roots required")
    attempt_root = Path(validation["attempt_root"])
    need(attempt_root.is_absolute() and ".." not in attempt_root.parts
         and not any(p.is_symlink() for p in (attempt_root, *attempt_root.parents))
         and any(attempt_root == Path(r) or Path(r) in attempt_root.parents for r in validation["roots"]),
         "validation attempt must remain below an existing original named root")
    return value


def frozen_candidate(path: Path, commitment: str, root: Path) -> dict:
    value = json_object(ordinary_bytes(path))
    need(value.get("stage") == "candidate_ready" and value.get("validation_passed") is False
         and value.get("runtime_adopted") is False, "original frozen candidate required")
    unsigned = {k: v for k, v in value.items() if k != "freeze_sha256"}
    need(value.get("freeze_sha256") == digest(unsigned), "candidate freeze commitment differs")
    need(value.get("source_commitment") == commitment and value["binding"]["candidate_root"] == str(root),
         "candidate source/root differs from the original binding")
    need(type(value["manifest"]) is dict and 0 < len(value["manifest"]) <= 4096, "finite supported source manifest required")
    for relative, row in value["manifest"].items():
        rel = Path(relative)
        need(not rel.is_absolute() and ".." not in rel.parts and rel.parts, "relative candidate file required")
        need(type(row) is dict and set(row) == {"bytes", "sha256", "executable"}, "typed frozen source row required")
        observed = descriptor(root / rel)
        need((observed["bytes"], observed["sha256"]) == (row["bytes"], row["sha256"]), "candidate source changed")
    return value


def validation_plan(policy, policy_pin, frozen, freeze_pin, root, invocation):
    binding = frozen["binding"]
    attempt = Path(policy["validation"]["attempt_root"]) / binding["namespace"] / binding["board_id"] / binding["attempt_id"]
    tests = policy["validation"]["trusted_tests"]
    # Tests are maintained inputs, never modified candidate tests accepted on
    # the strength of a provider-supplied passed flag.
    need(all(Path(r["path"]) != root and root not in Path(r["path"]).parents for r in tests),
         "trusted test generation must be outside the mutable candidate")
    pins = {}
    for pin in [*policy["source_descriptors"], *tests, policy["validation"]["pytest_module"], *policy["validation"]["import_sources"],
                *[{"path": str(root / rel), "sha256": row["sha256"]} for rel, row in frozen["manifest"].items()]]:
        need(pin["path"] not in pins or pins[pin["path"]] == pin["sha256"], "normal source manifest collision")
        pins[pin["path"]] = pin["sha256"]
    need(len(pins) <= 4096, "complete normal import manifest exceeds its original finite bound")
    controls = {r["path"]: r["sha256"] for r in policy["source_descriptors"] + tests}
    controls[policy_pin["path"]] = policy_pin["sha256"]
    controls[freeze_pin["path"]] = freeze_pin["sha256"]
    need(len(controls) <= 253, "original public request control cap reached; no cap increase")
    required = [str(root / rel) for rel in frozen["changed_paths"] if rel.endswith(".py")]
    need(required, "this finite validation profile requires a changed Python body")
    return {"schema": WORK_SCHEMA, "policy": policy_pin, "freeze": freeze_pin,
        "source_commitment": frozen["source_commitment"], "candidate_root": str(root),
        "attempt_directory": str(attempt), "source_pins": pins,
        "original_public_control_pins": controls,
        "selected_checks": policy["validation"]["selected_checks"],
        "expected_cases": len(policy["validation"]["selected_checks"]),
        "required_candidate_bodies": required}


def original_runner_wait(policy, argv, cwd, timeout):
    """Observe the actual Popen owned by the unchanged public engine helper."""
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import engine
    source = descriptor(Path(engine.__file__).absolute())
    need(source in policy["source_descriptors"], "original process owner source is not pinned")
    actual = engine.subprocess.Popen
    need(actual is _ORIGINAL_POPEN, "original process primitive was replaced before owner invocation")
    records = []
    def observed(command, *positional, **kwargs):
        frame = sys._getframe(1)
        need(frame.f_code is engine.run_command.__code__ and frame.f_globals is vars(engine)
             and frame.f_code.co_filename == source["path"] and list(command) == argv
             and kwargs.get("cwd") == str(cwd) and kwargs.get("start_new_session") is True,
             "only the exact original owned runner launch may be observed")
        process = actual(command, *positional, **kwargs)
        # Record uncertainty without raising after spawn: the original engine
        # must retain its actual process and perform its own stop/reap logic.
        records.append((process, proc_identity(process.pid)))
        return process
    try:
        engine.subprocess.Popen = observed
        completed = engine.run_command(argv, cwd=cwd, timeout_seconds=timeout)
    finally:
        engine.subprocess.Popen = actual
    need(len(records) == 1, "one genuine original runner process required")
    process, identity = records[0]
    need(identity is not None and process.returncode is not None
         and process.poll() == process.returncode == completed.returncode, "actual runner wait is unknown")
    need(descriptor(Path(engine.__file__).absolute()) == source, "original process owner source changed")
    return completed, {"runner_identity": identity, "actual_Popen_returned": True,
        "actual_process_waited": True, "actual_process_returncode": process.returncode, "engine_source": source}



CONTROL_PUBLICATION_SCHEMA = "agent-supervisor/public-validation-request-publication@1"
CLOSED_CONTROL_SCHEMA = "agent-supervisor/public-validation-closed-owner@1"


def immutable_bytes(path, raw):
    """Byte-exact append-only metadata; existing same bytes are read-only resume."""
    need(type(raw) is bytes and 0 < len(raw) <= 8 * 1024**2, "bounded immutable original bytes required")
    if path.exists():
        need(ordinary_bytes(path) == raw and not path.lstat().st_mode & 0o222,
             "existing immutable original copy differs; no replay")
        return descriptor(path)
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o400)
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return descriptor(path)


def validation_control_directory(policy, frozen):
    """The one approved namespace/board registry, sibling to every attempt."""
    binding = frozen["binding"]
    need(binding["namespace"] == "legal_ir_unattended_repair" and binding["board_id"] == REPAIR_PROFILE
         and type(binding["attempt_id"]) is str and re.fullmatch(r"[0-9a-f]{32}", binding["attempt_id"]),
         "finite original legal-ir control namespace/board/attempt required")
    directory = Path(policy["validation"]["attempt_root"]) / "control-requests" / binding["namespace"] / binding["board_id"] / binding["attempt_id"]
    need(not any(parent.is_symlink() for parent in (directory, *directory.parents)), "original control registry must not be aliased")
    return directory


def publish_validation_request(policy, frozen, invocation, local_request):
    """Publish the actual executed D request before invoking its original owner."""
    directory = validation_control_directory(policy, frozen)
    directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    count = 0
    with os.scandir(directory.parent) as entries:
        for entry in entries:
            count += 1
            need(count <= 64 and entry.is_dir(follow_symlinks=False)
                 and re.fullmatch(r"[0-9a-f]{32}", entry.name), "finite original control registry capacity/identity exceeded")
    need(count < 64 and not directory.exists(), "fresh original control generation required; do not replay")
    directory.mkdir(mode=0o700)
    request_pin = immutable_bytes(directory / "execution-request.json", checked_descriptor(local_request))
    plan_pin = descriptor(invocation / "validation-plan.json")
    plan = json_object(checked_descriptor(plan_pin))
    new_json(invocation / "execution-request-publication.json", {"schema": CONTROL_PUBLICATION_SCHEMA,
        "request": request_pin, "invocation_copy": local_request, "plan": plan_pin,
        "freeze": plan["freeze"], "owner_policy": plan["policy"], "source_commitment": frozen["source_commitment"]})
    return request_pin


def original_execution_request(invocation):
    """Resolve only the exact byte-identical original request publication."""
    local = descriptor(invocation / "execution-request.json")
    path = invocation / "execution-request-publication.json"
    if not path.exists():
        return local  # Existing closed local-only invocations remain readable.
    publication = json_object(ordinary_bytes(path))
    plan_pin = descriptor(invocation / "validation-plan.json"); plan = json_object(checked_descriptor(plan_pin))
    policy = json_object(checked_descriptor(plan["policy"])); frozen = json_object(checked_descriptor(plan["freeze"]))
    expected = validation_control_directory(policy, frozen) / "execution-request.json"
    need(set(publication) == {"schema", "request", "invocation_copy", "plan", "freeze", "owner_policy", "source_commitment"}
         and publication["schema"] == CONTROL_PUBLICATION_SCHEMA and publication["request"]["path"] == str(expected)
         and publication["invocation_copy"] == local and publication["plan"] == plan_pin
         and publication["freeze"] == plan["freeze"] and publication["owner_policy"] == plan["policy"]
         and publication["source_commitment"] == plan["source_commitment"]
         and checked_descriptor(publication["request"]) == checked_descriptor(local),
         "original executed request publication changed")
    return publication["request"]


def publish_closed_validation_control(policy, plan, frozen, request_pin, invocation, negative_pin):
    """Advertise closed failed resource custody, never semantic acceptance.

    The watchdog independently rechecks the fixed original D producer/resource
    version, exact current ledger row/native policy/PID/groups and typed failed
    inventory. The changing A publisher descriptor is historical metadata.
    """
    proof = json_object(checked_descriptor(negative_pin))
    if not all(proof[key] is True for key in ("source_closed", "native_closed", "processes_closed")):
        return None
    terminal = json_object(checked_descriptor(proof["original_failure_terminal"]))
    record = terminal["reservation"]["record"]
    request = json_object(checked_descriptor(request_pin))
    need(record["status"] == "retained" and record.get("retention_reason") == "context_failed" and record.get("external_charges") == {}
         and request["schema"] == "owned-validation-request/v1" and request["declared_external_writable_paths"] == []
         and proof["storage_closed"] is False and proof["original_record_sha256"]
             == hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
         "only the exact original nocharge context-failed retained claim may be advertised")
    wait = json_object(checked_descriptor(proof["original_runner_wait"]))
    need(wait["actual_Popen_returned"] is True and wait["actual_process_waited"] is True
         and wait["runner_identity"] == proof["original_owner_identity"]
         and wait["request"] == request_pin and json_object(wait["stdout"].encode()) == terminal,
         "genuine original wait/terminal bytes do not bind this failed claim")
    directory = validation_control_directory(policy, frozen)
    need(request_pin == descriptor(directory / "execution-request.json"), "original detached D request differs")
    wait_copy = immutable_bytes(directory / "runner-wait.json", checked_descriptor(proof["original_runner_wait"]))
    terminal_copy = immutable_bytes(directory / "original-failure-terminal.json", checked_descriptor(proof["original_failure_terminal"]))
    negative_copy = immutable_bytes(directory / "negative-validation.json", checked_descriptor(negative_pin))
    d = Path(policy["datasets_root"])
    producer = next(pin for pin in policy["source_descriptors"] if pin["path"] == str(d / "scripts/ops/legal_ir/run_owned_validation.py"))
    resources = next(pin for pin in policy["source_descriptors"] if pin["path"] == str(d / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/validation_run_resources.py"))
    checked_descriptor(producer); checked_descriptor(resources)
    value = {"schema": CLOSED_CONTROL_SCHEMA,
        "binding": {key: frozen["binding"][key] for key in ("namespace", "board_id", "attempt_id")},
        "producer": producer, "publisher": descriptor(Path(__file__).absolute()), "validation_resources": resources,
        "owner_policy": plan["policy"], "request": request_pin, "plan": descriptor(invocation / "validation-plan.json"),
        "freeze": plan["freeze"], "original_wait": wait_copy, "failure_terminal": terminal_copy,
        "negative_disposition": negative_copy, "invocation_copy": descriptor(invocation / "execution-request.json"),
        "owner": proof["original_owner_identity"], "reservation_id": proof["reservation_id"],
        "retained_record_sha256": proof["original_record_sha256"],
        "native_inspection": {"state_path": policy["validation"]["scheduler_state_path"],
            "policy_sha256": policy["validation"]["expected_scheduler_policy_sha256"],
            "request_id": "validation:" + proof["reservation_id"]},
        "genuine_wait_completed": True, "source_closed": True, "native_closed": True, "processes_closed": True,
        "storage_claim_released": False, "admitted": False, "Lean_admitted": False, "retry_authorized": False}
    return immutable_bytes(directory / "closed-owner.json", canonical(value) + b"\n")

def validate(args):
    policy = checked_policy(args.policy, args.policy_sha256)
    frozen = frozen_candidate(args.freeze_path, args.source_commitment, args.candidate_root)
    invocation = args.invocation_root
    need(args.receipt_path.parent == invocation and args.freeze_path.parent == invocation.parent,
         "original candidate owner invocation required")
    plan = validation_plan(policy, descriptor(args.policy), frozen, descriptor(args.freeze_path), args.candidate_root, invocation)
    need(len(canonical(plan)) + 1 <= 4 * 1024**2 and descriptor(args.freeze_path)["bytes"] <= 4 * 1024**2,
         "trusted immutable manifest exceeds the unchanged public source-file bound")
    plan_pin = new_json(invocation / "validation-plan.json", plan)
    attempt = Path(plan["attempt_directory"]); attempt.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    need(not any(p.is_symlink() for p in (attempt, *attempt.parents)), "attempt parent changed")
    attempt.mkdir(mode=0o700)
    v = policy["validation"]; module = str(Path(__file__).with_name("repair_validation_worker.py").absolute())
    request = {"schema": "owned-validation-request/v1", "argv": [policy["python"], "-I", "-B", module,
        "--plan", plan_pin["path"], "--plan-sha256", plan_pin["sha256"]],
        "cwd": str(args.candidate_root), "selected_checks": plan["selected_checks"],
        "source_sha256": {**plan["original_public_control_pins"], plan_pin["path"]: plan_pin["sha256"]},
        **{k: v[k] for k in ("timeout_seconds", "storage_bytes", "memory_mb", "cpu_slots", "child_process_slots",
                            "scheduler_state_path", "expected_scheduler_policy_sha256", "roots")},
        "attempt_directory": str(attempt), "outputs": {k: str(attempt / k) for k in ("pytest", "tmp", "cache", "db", "exports", "logs")},
        "declared_external_writable_paths": [], "process_scope": "sampled_birth_bound_descendants"}
    local_request = new_json(invocation / "execution-request.json", request)
    request_pin = publish_validation_request(policy, frozen, invocation, local_request)
    # The original engine owns stop/reap, and the public D CLI owns all native
    # and storage allocation/finalization. No historical operator code runs.
    argv = [policy["python"], "-I", "-B", str(Path(policy["datasets_root"]) / "scripts/ops/legal_ir/run_owned_validation.py"),
        "--request", request_pin["path"], "--expected-request-sha256", request_pin["sha256"],
        "--resource-ledger", v["resource_ledger"]]
    completed, witness = original_runner_wait(policy, argv, args.candidate_root, v["timeout_seconds"] + 60)
    new_json(invocation / "runner-wait.json", {"schema": WORK_SCHEMA, "request": request_pin, **witness,
        "argv": argv, "returncode": completed.returncode, "stdout": completed.stdout,
        "stderr": completed.stderr, "original_process_owner_returned": True})
    if type(completed.returncode) is int and completed.returncode != 0:
        capture_negative_validation(policy, plan, frozen, request_pin, invocation, completed)
        return completed.returncode if 0 < completed.returncode < 126 else 1
    need(type(completed.returncode) is int and completed.returncode == 0,
         "original validation owner failed or is unknown; retain without retry")
    terminal = json_object(completed.stdout.encode())
    original_pin = new_json(invocation / "original-terminal.json", terminal)
    summary = verify_validation_evidence(policy, plan, request_pin, terminal, invocation)
    new_json(args.receipt_path, {"schema": VALIDATION_SCHEMA, "freeze_sha256": frozen["freeze_sha256"],
        "source_commitment": args.source_commitment, "passed": True,
        "source_closed": True, "native_closed": True, "processes_closed": True, "storage_closed": True,
        "cleanup_errors": [], "original_receipt_reference": original_pin["path"],
        "original_receipt_descriptor": original_pin, "verified_checks": summary})


def verify_validation_evidence(policy, plan, request_pin, terminal, invocation):
    request = json_object(checked_descriptor(request_pin))
    wait = json_object(ordinary_bytes(invocation / "runner-wait.json"))
    need(wait["request"] == request_pin and wait["original_process_owner_returned"] is True
         and type(wait["returncode"]) is int and wait["returncode"] == 0
         and json_object(wait["stdout"].encode()) == terminal, "genuine original runner wait required")
    reservation = terminal["reservation"]; record = reservation["record"]; native = reservation["resource_lease"]
    need(reservation["status"] == record["status"] == "released" and record["artifacts_durable_asserted"] is True
         and native["released"] is True and native["lease_id"] == record["native_lease_id"]
         and record["validation_request_sha256"] == request_pin["sha256"] and reservation["cleanup_error"] is None,
         "original native/storage closure is not proven")
    need(wait["runner_identity"] == {"pid": record["owner_pid"], "birth": record["owner_birth"]}
         and native["owner_pid"] == record["owner_pid"]
         and wait["actual_Popen_returned"] is True and wait["actual_process_waited"] is True
         and wait["actual_process_returncode"] == 0, "original runner PID/birth/wait binding differs")
    checked_descriptor(wait["engine_source"])
    process = reservation["process_observation"]
    need(process["live_processes"] == 0 and process["out_of_group_live_identities"] == []
         and not process["tracking_error"], "original descendant custody remains unknown")
    attempt = Path(plan["attempt_directory"])
    owner = json_object(ordinary_bytes(Path(terminal["result"])))
    need(Path(terminal["result"]) == attempt / "logs/result.json" and owner["schema"] == "owned-validation-result/v1"
         and owner["completed"] is True and type(owner["returncode"]) is int and owner["returncode"] == 0
         and owner["request_sha256"] == request_pin["sha256"], "original result mismatch")
    report = json_object(ordinary_bytes(attempt / "exports/trusted-validation.json"))
    need(report["schema"] == WORK_SCHEMA and report["passed"] is True
         and report["source_READ_closed"] is True and report["guard_denials"] == []
         and report["plan_sha256"] == descriptor(invocation / "validation-plan.json")["sha256"],
         "trusted source/semantic result missing")
    need(report["full_source_manifest_sha256"] == digest(plan["source_pins"])
         and report["full_source_file_count"] == len(plan["source_pins"]), "complete trusted source guard was not applied")
    need(report["pytest_module_origin"] == policy["validation"]["pytest_module"]
         and report["startup"]["python"] == policy["python"]
         and report["startup"]["isolated"] == 1 and report["startup"]["no_bytecode"] is True,
         "trusted interpreter/Pytest origin differs")
    need(all(report["executed_source_buffers"].get(path) == plan["source_pins"][path]
             for path in plan["required_candidate_bodies"]), "edited candidate body was not actually tested")
    expected = Counter((node, phase) for node in plan["selected_checks"] for phase in ("setup", "call", "teardown"))
    need(all(row["outcome"] == "passed" for row in report["phases"])
         and Counter((r["nodeid"], r["when"]) for r in report["phases"]) == expected,
         "every natural phase must actually pass")
    suites = ET.fromstring(ordinary_bytes(attempt / "exports/test-results.xml")).findall(".//testsuite")
    need({k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
         == {"tests": plan["expected_cases"], "failures": 0, "errors": 0, "skipped": 0}, "JUnit result differs")
    durability = json_object(ordinary_bytes(attempt / "logs/validation-durability.json"))
    need(durability["schema"] == "owned-validation-durability/v1"
         and durability["request_sha256"] == request_pin["sha256"]
         and durability["reservation_id"] == record["reservation_id"]
         and durability["result_sha256"] == descriptor(Path(terminal["result"]))["sha256"], "original durability mismatch")
    seen = set()
    for row in durability["files"]:
        rel = Path(row["path"])
        need(not rel.is_absolute() and ".." not in rel.parts and row["path"] not in seen, "durability path mismatch")
        seen.add(row["path"]); path = attempt / rel; pin = descriptor(path); info = path.lstat()
        need((pin["bytes"], pin["sha256"], info.st_dev, info.st_ino) == (row["bytes"], row["sha256"], row["device"], row["inode"]), "durable artifact changed")
    for path, sha in request["source_sha256"].items():
        need(descriptor(Path(path))["sha256"] == sha, "validation source changed")
    return {"cases": plan["expected_cases"], "phases": sum(expected.values()), "reservation_id": record["reservation_id"],
            "native_lease_id": native["lease_id"], "natural_phase_counter_verified": True}


def verify_validation(args):
    policy = checked_policy(args.policy, args.policy_sha256)
    receipt = json_object(ordinary_bytes(args.receipt_path))
    plan = json_object(ordinary_bytes(args.invocation_root / "validation-plan.json"))
    frozen_candidate(args.freeze_path, args.source_commitment, args.candidate_root)
    need(receipt["schema"] == VALIDATION_SCHEMA and receipt["passed"] is True
         and receipt["source_commitment"] == args.source_commitment, "positive bound validation required")
    terminal = json_object(checked_descriptor(receipt["original_receipt_descriptor"]))
    request_pin = original_execution_request(args.invocation_root)
    verify_validation_evidence(policy, plan, request_pin, terminal, args.invocation_root)







NEGATIVE_SCHEMA = "agent-supervisor/isolated-repair-candidate-negative-validation@1"
NEGATIVE_CLOSURE_SCHEMA = "agent-supervisor/negative-validation-closure-observation@1"
NEGATIVE_VERIFIED_SCHEMA = "agent-supervisor/verified-negative-validation@1"


def verify_negative_semantic_report(policy, plan, report, invocation):
    """Recheck original failed-report attribution and raw compilation facts."""
    need(report["schema"] == WORK_SCHEMA and report["passed"] is False
         and report["source_READ_closed"] is True and report["source_snapshot_stable"] is True
         and report["cleanup_errors"] == [] and report["guard_denials"] == [] and report["error"] is None
         and report["junit_error"] is None
         and report["plan_sha256"] == descriptor(invocation / "validation-plan.json")["sha256"]
         and report["freeze"] == plan["freeze"] and report["source_commitment"] == plan["source_commitment"],
         "original negative source/report closure is unproven")
    need(report["full_source_manifest_sha256"] == digest(plan["source_pins"])
         and report["full_source_file_count"] == len(plan["source_pins"])
         and report["pytest_module_origin"] == policy["validation"]["pytest_module"]
         and report["startup"] == {"python": policy["python"], "isolated": 1, "no_bytecode": True}
         and report["original_compile_primitive"] == {"module": "builtins", "name": "compile", "builtin": True},
         "trusted negative interpreter/manifest/compiler identity differs")
    for path, sha in plan["source_pins"].items():
        need(descriptor(Path(path))["sha256"] == sha, "negative source identity changed")
    code = report["pytest_returncode"]
    need(type(code) is int and code in {1, 2}, "nonsemantic worker failure cannot authorize a coding attempt")
    suites = ET.fromstring(ordinary_bytes(Path(plan["attempt_directory"]) / "exports/test-results.xml")).findall(".//testsuite")
    junit = {key: sum(int(suite.get(key, 0)) for suite in suites) for key in ("tests", "failures", "errors", "skipped")}
    need(report["JUnit"] == junit, "original worker/JUnit counts differ")
    expected = Counter((node, phase) for node in plan["selected_checks"] for phase in ("setup", "call", "teardown"))
    phases = report["phases"]
    complete = (report["collected_cases"] == plan["selected_checks"]
        and Counter((row["nodeid"], row["when"]) for row in phases) == expected)
    bodies = all(report["executed_source_buffers"].get(path) == plan["source_pins"][path] for path in plan["required_candidate_bodies"])
    selected_files = {node.partition("::")[0] for node in plan["selected_checks"]}
    failures = report["failed_report_attributions"]
    need(type(failures) is list and bool(failures)
         and all(type(row) is dict and row["report_path"] in selected_files
             and row["when"] in {"call", "collect"} for row in failures),
         "original selected failed-report attribution is unavailable")
    failed_calls = [row["nodeid"] for row in phases if row["when"] == "call" and row["outcome"] == "failed"]
    call_shape = (complete and report["natural_phase_counter_complete"] is True and bool(failed_calls)
        and all(row["outcome"] == "passed" if row["when"] != "call" else row["outcome"] in {"passed", "failed"} for row in phases)
        and junit == {"tests": plan["expected_cases"], "failures": len(failed_calls), "errors": 0, "skipped": 0}
        and Counter(row["nodeid"] for row in failures if row["when"] == "call") == Counter(failed_calls)
        and all(row["when"] == "call" for row in failures))
    def attributed_call(row):
        if row["kind"] == "trusted_test_assertion":
            frame = row["trusted_test_frame"]
            return (row["exception_type"] == "AssertionError" and type(frame) is dict
                and frame["path"] in selected_files and frame["sha256"] == plan["source_pins"][frame["path"]]
                and report["raw_compiled_sources"].get(frame["path"]) == frame["sha256"])
        if row["kind"] == "changed_candidate_runtime_error":
            frames = row["candidate_traceback"]
            return (type(frames) is list and bool(frames) and all(frame["path"] in plan["required_candidate_bodies"]
                and frame["sha256"] == plan["source_pins"][frame["path"]]
                and report["executed_source_buffers"].get(frame["path"]) == frame["sha256"] for frame in frames))
        return False
    if code == 1 and call_shape and bodies and all(attributed_call(row) for row in failures):
        profile = "trusted_pytest_call_failure"
    else:
        valid_indices = {index for index, row in enumerate(report["syntax_failures"])
            if row["path"] in plan["required_candidate_bodies"] and row["filename"] == row["path"]
            and row["sha256"] == plan["source_pins"][row["path"]]
            and report["raw_compiled_sources"].get(row["path"]) == row["sha256"]
            and row["body_executed"] is False and row["path"] not in report["executed_source_buffers"]
            and row["exception_category"] == "SyntaxError" and row["exception_type"] in {"SyntaxError", "IndentationError", "TabError"}}
        wrapper_path = str(Path(policy["validation"]["pytest_module"]["path"]).parent.parent / "_pytest/python.py")
        def attributed_syntax(row):
            indices = row["syntax_witness_indices"]
            if not (type(indices) is list and bool(indices)
                    and all(type(index) is int and index in valid_indices for index in indices)):
                return False
            if row["kind"] == "original_source_syntax_error":
                return row["exception_type"] in {"SyntaxError", "IndentationError", "TabError"}
            if row["kind"] == "trusted_pytest_collect_wrapper":
                wrapper = row["trusted_wrapper_source"]
                return (row["exception_type"] == "CollectError" and type(wrapper) is dict
                    and wrapper["path"] == wrapper_path and descriptor(Path(wrapper_path)) == wrapper
                    and plan["source_pins"].get(wrapper_path) == wrapper["sha256"]
                    and report["executed_source_buffers"].get(wrapper_path) == wrapper["sha256"])
            return False
        collection_shape = (not phases and all(row["when"] == "collect" for row in failures)
            and junit["errors"] == len(failures) and junit["failures"] == 0 and junit["skipped"] == 0)
        need(all(attributed_syntax(row) for row in failures)
             and ((code == 1 and call_shape) or (code == 2 and collection_shape)),
             "original failed reports do not prove the changed-candidate compilation SyntaxError")
        profile = "changed_candidate_source_syntax_error"
    need(report["semantic_negative"] is True and report["negative_profile"] == profile,
         "trusted negative classification differs from its actual evidence")
    return profile


def negative_validation_material(policy, plan, frozen, request_pin, invocation, terminal, wait):
    """Join immutable original facts; retained storage remains explicitly false."""
    need(type(wait["returncode"]) is int and wait["returncode"] != 0 and wait["request"] == request_pin
         and wait["original_process_owner_returned"] is True and wait["actual_Popen_returned"] is True
         and wait["actual_process_waited"] is True and wait["actual_process_returncode"] == wait["returncode"]
         and json_object(wait["stdout"].encode()) == terminal, "original failed process wait is unknown")
    checked_descriptor(wait["engine_source"])
    need(wait["engine_source"] in policy["source_descriptors"], "original failed process source is not pinned")
    need(terminal["schema"] == "owned-validation-failure-terminal/v1" and terminal["completed"] is False
         and terminal["request_sha256"] == request_pin["sha256"]
         and terminal["process_cleanup_error"] is None and terminal["owner_exit_error"] is None,
         "original post-exit validation failure is incomplete")
    reservation = terminal["reservation"]; record = reservation["record"]; native = reservation["resource_lease"]
    identity = {"pid": record["owner_pid"], "birth": record["owner_birth"]}
    need(record["validation_request_sha256"] == request_pin["sha256"] and identity == wait["runner_identity"]
         and native["lease_id"] == record["native_lease_id"] and native["owner_pid"] == identity["pid"],
         "original failed request/native/PID/birth binding differs")
    lifecycle = terminal["source_custody_lifecycle"]
    need(lifecycle["request_sha256"] == request_pin["sha256"] and lifecycle["owner"] == identity
         and lifecycle["reservation_id"] == record["reservation_id"]
         and lifecycle["events"]["owner_exit_handling_completed"] is True
         and lifecycle["source_guards"]["errors"] == []
         and all(lifecycle["source_guards"][key] is True for key in ("request_unchanged", "source_pins_unchanged", "owner_identity_unchanged")),
         "original failed owner exit/source control flow is unknown")
    producer = lifecycle["producer"]
    need(type(producer) is dict and set(producer) == {"path", "sha256"}, "typed original failed producer required")
    producer_pin = next(pin for pin in policy["source_descriptors"] if pin["path"] == producer["path"])
    checked_descriptor(producer_pin)
    need(producer_pin["sha256"] == producer["sha256"], "original failed producer identity is unpinned")
    report_path = Path(plan["attempt_directory"]) / "exports/trusted-validation.json"
    need(not report_path.lstat().st_mode & 0o222, "trusted worker report is not immutable")
    report = json_object(ordinary_bytes(report_path))
    need(report["worker_identity"] == record["child"], "trusted failed worker is not the original attached PID and birth")
    need(report["schema"] == WORK_SCHEMA and report["plan_sha256"] == descriptor(invocation / "validation-plan.json")["sha256"]
         and report["freeze"] == plan["freeze"] and report["source_commitment"] == plan["source_commitment"]
         and report["full_source_manifest_sha256"] == digest(plan["source_pins"])
         and report["full_source_file_count"] == len(plan["source_pins"])
         and report["startup"] == {"python": policy["python"], "isolated": 1, "no_bytecode": True}
         and type(report["source_READ_closed"]) is bool and type(report["cleanup_errors"]) is list,
         "original failed worker/report provenance is unknown")
    # Resource custody and semantic diagnosis are separate. An unsupported
    # failed test may still close its own unused storage promise without any
    # task, retry, inference or validation-acceptance authority.
    source_closed = report["source_READ_closed"] is True and report["cleanup_errors"] == []
    profile = None; semantic_error = None
    try:
        profile = verify_negative_semantic_report(policy, plan, report, invocation)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ET.ParseError) as error:
        semantic_error = {"type": type(error).__name__, "message": str(error)[:1000]}
    process = reservation["process_observation"]
    process_closed = (process["live_processes"] == 0 and process["out_of_group_live_identities"] == [] and not process["tracking_error"])
    native_closed = native["released"] is True
    storage_closed = reservation["status"] == record["status"] == "released"
    need(reservation["cleanup_error"] is None, "original resource cleanup is unknown")
    return {"semantic_negative": profile is not None, "negative_profile": profile, "semantic_error": semantic_error,
        "source_closed": source_closed, "native_closed": native_closed,
        "processes_closed": process_closed, "storage_closed": storage_closed,
        "reservation_id": record["reservation_id"], "native_lease_id": native["lease_id"],
        "process_scope": record["process_scope"], "original_owner_identity": identity,
        "original_record_sha256": hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "trusted_semantic_report": descriptor(report_path),
        "original_JUnit_report": descriptor(Path(plan["attempt_directory"]) / "exports/test-results.xml")
            if (Path(plan["attempt_directory"]) / "exports/test-results.xml").exists() else None,
        "original_child_identity": record["child"]}


def capture_negative_validation(policy, plan, frozen, request_pin, invocation, completed):
    """Preserve original partial failure now; future storage release is separate."""
    wait_pin = descriptor(invocation / "runner-wait.json"); wait = json_object(checked_descriptor(wait_pin))
    value = {"schema": NEGATIVE_SCHEMA, "passed": False, "semantic_negative": False, "negative_profile": None, "semantic_error": None,
        "freeze_sha256": frozen["freeze_sha256"], "source_commitment": frozen["source_commitment"],
        "binding": frozen["binding"], "request": request_pin,
        "original_owner_returncode": completed.returncode, "original_owner_identity": wait["runner_identity"],
        "reservation_id": None, "native_lease_id": None, "process_scope": None, "original_record_sha256": None,
        "source_closed": False, "native_closed": False, "processes_closed": False, "storage_closed": False,
        "cleanup_errors": [], "original_failure_terminal": None, "original_runner_wait": wait_pin,
        "trusted_semantic_report": None, "original_JUnit_report": None, "original_child_identity": None,
        "observation_errors": [], "validation_replayed": False, "resource_recovery_performed": False}
    report_path = Path(plan["attempt_directory"]) / "exports/trusted-validation.json"
    if report_path.exists():
        # Retain the original file even when incomplete resource provenance
        # prevents treating its contents as accepted semantic/source evidence.
        value["trusted_semantic_report"] = descriptor(report_path)
    try:
        terminal = json_object(completed.stdout.encode())
        value["original_failure_terminal"] = immutable_bytes(invocation / "original-failure-terminal.json", completed.stdout.encode())
        value.update(negative_validation_material(policy, plan, frozen, request_pin, invocation, terminal, wait))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, StopIteration, ET.ParseError) as error:
        value["observation_errors"].append({"type": type(error).__name__, "message": str(error)[:1000]})
    proof_pin = new_json(invocation / "negative-validation.json", value)
    try:
        publish_closed_validation_control(policy, plan, frozen, request_pin, invocation, proof_pin)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, StopIteration) as error:
        new_json(invocation / "closed-owner-publication-failure.json", {"schema": CLOSED_CONTROL_SCHEMA,
            "negative_disposition": proof_pin, "publication_completed": False,
            "error": {"type": type(error).__name__, "message": str(error)[:1000]},
            "original_validation_replayed": False, "resource_recovery_performed": False})
    return proof_pin


def verify_negative_validation(args):
    """Read-only join of original failed work and later public storage history.

    No allocation, recovery, validation replay, source write or service action
    is authorized by this observation. Missing or ambiguous history stays held.
    """
    policy = checked_policy(args.policy, args.policy_sha256)
    frozen = frozen_candidate(args.freeze_path, args.source_commitment, args.candidate_root)
    invocation = args.invocation_root
    need(args.receipt_path.parent == invocation and args.freeze_path.parent == invocation.parent,
         "original negative invocation binding required")
    proof_path = invocation / "negative-validation.json"; proof_pin = descriptor(proof_path)
    need(not proof_path.lstat().st_mode & 0o222, "original negative proof must be immutable")
    proof = json_object(checked_descriptor(proof_pin))
    need(proof["schema"] == NEGATIVE_SCHEMA and proof["passed"] is False and proof["semantic_negative"] is True
         and proof["freeze_sha256"] == frozen["freeze_sha256"] and proof["source_commitment"] == frozen["source_commitment"]
         and proof["binding"] == frozen["binding"] and proof["cleanup_errors"] == [] and proof["observation_errors"] == [],
         "original trusted negative validation is unknown")
    plan = json_object(ordinary_bytes(invocation / "validation-plan.json"))
    need(plan["freeze"] == descriptor(args.freeze_path) and plan["source_commitment"] == frozen["source_commitment"],
         "negative original frozen plan differs")
    request_pin = original_execution_request(invocation)
    need(proof["request"] == request_pin and proof["original_runner_wait"] == descriptor(invocation / "runner-wait.json"),
         "negative original request/wait descriptors differ")
    terminal = json_object(checked_descriptor(proof["original_failure_terminal"]))
    wait = json_object(checked_descriptor(proof["original_runner_wait"]))
    observed = negative_validation_material(policy, plan, frozen, request_pin, invocation, terminal, wait)
    need(all(proof[key] == value for key, value in observed.items())
         and type(proof["original_owner_returncode"]) is int and proof["original_owner_returncode"] == wait["returncode"]
         and all(proof[key] is True for key in ("source_closed", "native_closed", "processes_closed")),
         "original failed source/native/process evidence does not establish closure")
    source = public_source_lease(policy)
    module_name = "ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_resource_recovery"
    expected_path = str(Path(policy["datasets_root"]) / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_resource_recovery.py")
    expected = next(pin for pin in policy["source_descriptors"] if pin["path"] == expected_path)
    with source(Path(policy["datasets_root"]), mode="read", timeout_seconds=policy["validation"]["timeout_seconds"]):
        checked_descriptor(expected)
        recovery = importlib.import_module(module_name)
        need(descriptor(Path(recovery.__file__).absolute()) == expected, "public disposition inspector origin differs")
        validation = policy["validation"]
        # This original selected-row inspector supplies an exact current CAS
        # preimage; status alone supplies no release authority.
        current = recovery.inspect_validation_reservation(validation["resource_ledger"], roots=validation["roots"],
            reservation_id=proof["reservation_id"], expected_request_sha256=request_pin["sha256"],
            attempt_directory=plan["attempt_directory"], process_scope=proof["process_scope"])
        need(current["present"] is True and current["binding_verified"] is True,
             "original selected validation custody is unavailable")
        disposition = recovery.inspect_validation_disposition(validation["resource_ledger"], roots=validation["roots"],
            reservation_id=proof["reservation_id"], expected_request_sha256=request_pin["sha256"],
            attempt_directory=plan["attempt_directory"], expected_record_sha256=current["record_sha256"],
            expected_previous_record_sha256=proof["original_record_sha256"], process_scope=proof["process_scope"])
    # Final key spelling must match the separately qualified public inspector.
    need(disposition["storage_release_established"] is True
         and disposition["reservation_id"] == proof["reservation_id"]
         and disposition["expected_request_sha256"] == request_pin["sha256"]
         and disposition["record_sha256"] == current["record_sha256"]
         and disposition["expected_previous_record_sha256"] == proof["original_record_sha256"]
         and disposition["schema"] == "validation-storage-disposition-inspection/v1"
         and disposition["present"] is True and disposition["binding_verified"] is True
         and disposition["status"] == "released" and disposition["native_release_established"] is False
         and disposition["state_written"] is False and disposition["recovery_performed"] is False and disposition["admitted"] is False
         and disposition["custody"]["native_lease_id"] == proof["native_lease_id"]
         and {"pid": disposition["custody"]["owner_pid"], "birth": disposition["custody"]["owner_birth"]} == proof["original_owner_identity"]
         and disposition["release_evidence"]["previous_record_sha256"] == proof["original_record_sha256"]
         and disposition["release_evidence"]["native_release_established"] is False
         and disposition["release_evidence"]["test_or_task_acceptance_asserted"] is False
         and disposition["release_evidence"]["historical_capacity_is_admission_authority"] is False,
         "genuine original selected-storage release/history is unproven")
    closure = {"schema": NEGATIVE_CLOSURE_SCHEMA, "negative_validation": proof_pin,
        **{key: proof[key] for key in ("freeze_sha256", "source_commitment", "binding", "negative_profile", "request", "original_owner_returncode",
            "original_owner_identity", "reservation_id", "native_lease_id")},
        "source_closed": True, "native_closed": True, "processes_closed": True, "storage_closed": True,
        "cleanup_errors": [], "storage_disposition": disposition, "observed_at": time.time()}
    print(canonical({"schema": NEGATIVE_VERIFIED_SCHEMA, "negative_validation": proof_pin,
        "closure_observation": closure}).decode(), flush=True)
    return 0

def public_source_lease(policy):
    d = Path(policy["datasets_root"])
    expected = str(d / "ipfs_datasets_py/optimizers/logic_theorem_optimizer/autoencoder_source_lease.py")
    pin = next(row for row in policy["source_descriptors"] if row["path"] == expected)
    checked_descriptor(pin)
    sys.path.insert(0, str(d))
    os.environ["IPFS_DATASETS_PY_MINIMAL_IMPORTS"] = "1"
    module = importlib.import_module("ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_source_lease")
    need(descriptor(Path(module.__file__).absolute()) == pin, "canonical public source owner origin differs")
    return module.source_lease


def git_command(root, argv, *, timeout, environment=None, stdin=None):
    """Fixed callers below provide all argv; provider strings are never commands."""
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.engine import run_command
    value = run_command(["git", *argv], cwd=Path(root), timeout_seconds=timeout,
                        environment=environment, stdin=stdin, preserve_output_newlines=True)
    need(type(value.returncode) is int and value.returncode == 0, "original Git command failed; no replay")
    return value.stdout


def stage_source_successor(args, policy, frozen, contract):
    """Write real immutable Git objects; leave active HEAD/index/files unchanged."""
    preparing = json_object(ordinary_bytes(args.freeze_path.parent / "preparing.json"))
    prior = preparing["policy"]; active = Path(prior["repository_root"])
    need(active == Path(frozen["binding"]["active_root"]), "source snapshot root differs")
    need(set(frozen["changed_paths"]) <= set(prior["allowed_paths"]), "source path authority widened")
    need(contract["source_successor"] is None, "operator contract cannot inject a source successor")
    timeout = min(prior["git_timeout_seconds"], policy["validation"]["timeout_seconds"])
    clean = {key: os.environ[key] for key in ("PATH", "HOME", "LANG") if key in os.environ}
    clean.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null", GIT_TERMINAL_PROMPT="0",
        GIT_OPTIONAL_LOCKS="0", GIT_INDEX_FILE=str(args.invocation_root / "source-snapshot.index"))
    author_time = str(int(time.time())) + " +0000"
    clean.update(GIT_AUTHOR_NAME="Fleet repair owner", GIT_AUTHOR_EMAIL="repair-owner@localhost",
        GIT_COMMITTER_NAME="Fleet repair owner", GIT_COMMITTER_EMAIL="repair-owner@localhost",
        GIT_AUTHOR_DATE=author_time, GIT_COMMITTER_DATE=author_time)
    commands = []
    def run(argv, text=None):
        output = git_command(active, argv, timeout=timeout, environment=clean, stdin=text)
        commands.append({"argv": ["git", *argv], "returncode": 0, "stdout": output})
        return output.strip()
    lease = public_source_lease(policy)
    with lease(active, mode="write", timeout_seconds=timeout):
        need(run(["rev-parse", "--verify", "HEAD"]) == prior["base_commit"], "active source HEAD changed")
        index_environment = {k: v for k, v in clean.items() if k != "GIT_INDEX_FILE"}
        index = Path(git_command(active, ["rev-parse", "--git-path", "index"], timeout=timeout, environment=index_environment).strip())
        if not index.is_absolute():
            index = active / index
        before_index = descriptor(index)
        head = Path(git_command(active, ["rev-parse", "--git-path", "HEAD"], timeout=timeout, environment=index_environment).strip())
        if not head.is_absolute():
            head = active / head
        common = Path(git_command(active, ["rev-parse", "--git-common-dir"], timeout=timeout, environment=index_environment).strip())
        if not common.is_absolute():
            common = (active / common).resolve(strict=True)
        need(head.resolve() != (common / "HEAD").resolve(), "repair publication requires the explicitly selected linked worktree")
        head_pin = descriptor(head); primary = [descriptor(common / "HEAD"), descriptor(common / "index")]
        head_text = ordinary_bytes(head).decode("ascii").strip()
        target = head_text.removeprefix("ref: ") if head_text.startswith("ref: ") else "HEAD"
        need(target == "HEAD" or target.startswith("refs/heads/") and ".." not in target,
             "owned active worktree ref required")
        need(not (head_text.startswith("ref: ") and ordinary_bytes(common / "HEAD").decode().strip() == head_text),
             "repair cannot update the primary worktree branch")
        before = {}
        for rel in prior["source_paths"]:
            pin = descriptor(active / rel); baseline = preparing["baseline"][rel]
            need((pin["bytes"], pin["sha256"]) == (baseline["bytes"], baseline["sha256"]), "active source beforeimage differs")
            before[rel] = pin
        run(["read-tree", prior["base_commit"]])
        for rel in frozen["changed_paths"]:
            raw = ordinary_bytes(args.candidate_root / rel, 4 * 1024**2)
            need(hashlib.sha256(raw).hexdigest() == frozen["manifest"][rel]["sha256"], "frozen source changed")
            oid = run(["hash-object", "-w", "--stdin"], raw.decode("utf-8"))
            need(re.fullmatch(r"[0-9a-f]{40}", oid), "actual source blob missing")
            mode = "100755" if frozen["manifest"][rel]["executable"] else "100644"
            run(["update-index", "--add", "--cacheinfo", mode + "," + oid + "," + rel])
        tree = run(["write-tree"])
        commit = run(["commit-tree", tree, "-p", prior["base_commit"]], "Validated repair " + frozen["freeze_sha256"] + "\n")
        header = run(["cat-file", "-p", commit])
        need(header.startswith("tree " + tree + "\nparent " + prior["base_commit"] + "\n"), "new commit parent/tree differs")
        need(descriptor(index) == before_index and descriptor(head) == head_pin
             and run(["rev-parse", "--verify", "HEAD"]) == prior["base_commit"], "staging changed the active index or HEAD")
        for pin in primary:
            checked_descriptor(pin)
    successor = dict(prior, base_commit=commit)
    source_refs = [{"path": str(active / rel), "bytes": row["bytes"], "sha256": row["sha256"]}
                   for rel, row in sorted(frozen["manifest"].items()) if rel in prior["source_paths"]]
    need(len(source_refs) == len(prior["source_paths"]), "successor selected source population differs")
    return {"schema": "agent-supervisor/repair-source-snapshot@1", "prior_policy": prior,
        "successor_policy": successor, "new_commit": commit, "new_tree": tree,
        "source_commitment": frozen["source_commitment"], "freeze": descriptor(args.freeze_path),
        "validation": new_json(args.invocation_root / "validated-owner-receipt.json", json_object(ordinary_bytes(args.validation_path))["validation"]),
        "candidate_root": str(args.candidate_root), "before": before, "active_index": before_index,
        "active_head": head_pin, "ref_target": target, "primary_Git_preserved": primary,
        "changed_paths": frozen["changed_paths"], "manifest": frozen["manifest"],
        "successor_source_descriptors": source_refs, "commands": commands,
        "public_owner_policy": descriptor(args.policy), "source_WRITE_closed": True,
        "attempt_number": frozen["binding"]["attempt_number"], "attempt_token": frozen["binding"]["attempt_id"],
        "board_id": frozen["binding"]["board_id"]}



def cutover_remaining(deadline):
    """One finite original monotonic budget; callers may only shorten it."""
    need(type(deadline) in (int, float) and 0 < deadline < float("inf"), "finite original cutover deadline required")
    left = deadline - time.monotonic()
    need(left > 0, "original cutover deadline expired; retain partial effects without replay")
    return left


def original_cutover_clock(unit, identity, runner):
    """Use the actual systemd execution start, within its existing120s limit."""
    observed = runner({"argv": ["systemctl", "--user", "show", unit,
        "--property=MainPID,InvocationID,ExecMainStartTimestampMonotonic"]}, cwd="/", timeout=5)
    need(type(observed["returncode"]) is int and observed["returncode"] == 0, "original cutover clock observation failed")
    props = dict(line.split("=", 1) for line in observed["stdout"].splitlines() if "=" in line)
    need(props.get("MainPID") == str(identity["pid"]) and proc_identity(identity["pid"]) == identity
         and re.fullmatch(r"[0-9a-f]{32}", props.get("InvocationID", ""))
         and props.get("ExecMainStartTimestampMonotonic", "").isdecimal(), "original external invocation clock differs")
    started = int(props["ExecMainStartTimestampMonotonic"]) / 1000000
    need(0 < started <= time.monotonic(), "original external start time is unknown")
    deadline = started + 120
    cutover_remaining(deadline)
    return {"owner": identity, "invocation_id": props["InvocationID"], "started_monotonic": started,
        "deadline_monotonic": deadline, "existing_kernel_runtime_seconds": 120}

def install_source_successor(snapshot, *, deadline=None):
    """Original checkout lease and source WRITE, exact finite files/HEAD CAS."""
    from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
        acquire_checkout_mutation_lease, checkout_lock_metadata, checkout_mutation_lock_path, release_checkout_mutation_lease)
    from scripts.ops.agent_supervisor.install_fleet_watchdog import atomic_write
    policy = json_object(checked_descriptor(snapshot["public_owner_policy"]))
    active = Path(snapshot["prior_policy"]["repository_root"])
    timeout = snapshot["prior_policy"]["git_timeout_seconds"]
    def budget():
        return min(timeout, cutover_remaining(deadline)) if deadline is not None else timeout
    metadata = checkout_lock_metadata(kind="fleet-repair-cutover", repo_root=active,
        task_id="infrastructure-repair:" + snapshot["board_id"], attempt=snapshot["attempt_number"],
        extra={"process_birth": proc_identity(os.getpid())["birth"], "repair_unit_token": snapshot["attempt_token"]})
    def alive(old):
        identity = proc_identity(int(old.get("pid") or 0)) if old.get("pid") else None
        return identity is not None and (not old.get("process_birth") or identity["birth"] == old["process_birth"])
    held, reason, _, _ = acquire_checkout_mutation_lease(checkout_mutation_lock_path(active), metadata,
        owner_active=alive, timeout_seconds=budget())
    need(held is not None, "original checkout owner unavailable: " + str(reason))
    try:
        with public_source_lease(policy)(active, mode="write", timeout_seconds=budget()):
            need(git_command(active, ["rev-parse", "--verify", "HEAD"], timeout=budget()).strip()
                 == snapshot["prior_policy"]["base_commit"], "source HEAD preimage differs")
            checked_descriptor(snapshot["active_index"])
            checked_descriptor(snapshot["active_head"])
            for pin in snapshot["primary_Git_preserved"]:
                checked_descriptor(pin)
            for pin in snapshot["before"].values():
                checked_descriptor(pin)
            # Exact owned branch update only; the primary worktree's HEAD and
            # index are untouched. No broad add/reset/checkout/merge occurs.
            for rel in snapshot["changed_paths"]:
                budget()
                source = Path(snapshot["candidate_root"]) / rel
                row = snapshot["manifest"][rel]; raw = ordinary_bytes(source, 4 * 1024**2)
                need(hashlib.sha256(raw).hexdigest() == row["sha256"], "qualified source origin changed")
                checked_descriptor(snapshot["before"][rel])
                atomic_write(active / rel, raw, mode=stat.S_IMODE((active / rel).lstat().st_mode))
            args = ["update-ref"] + (["--no-deref"] if snapshot["ref_target"] == "HEAD" else [])
            git_command(active, [*args, snapshot["ref_target"], snapshot["new_commit"], snapshot["prior_policy"]["base_commit"]], timeout=budget())
            checked_descriptor(snapshot["active_index"])
            for pin in snapshot["primary_Git_preserved"]:
                checked_descriptor(pin)
            for pin in snapshot["successor_source_descriptors"]:
                checked_descriptor(pin)
        with public_source_lease(policy)(active, mode="read", timeout_seconds=budget()):
            need(git_command(active, ["rev-parse", "--verify", "HEAD"], timeout=budget()).strip() == snapshot["new_commit"], "source HEAD postimage differs")
            for pin in snapshot["successor_source_descriptors"]:
                checked_descriptor(pin)
    finally:
        need(release_checkout_mutation_lease(held, require_exact_metadata=True), "original checkout release unknown")
    return {"completed": True, "source_WRITE_closed": True, "postwrite_source_READ_closed": True,
            "new_commit": snapshot["new_commit"], "new_tree": snapshot["new_tree"]}


def verify_original_job_closed(unit, runner, *, deadline=None):
    result = runner({"argv": ["systemctl", "--user", "show", unit,
        "--property=MainPID,ActiveState,ControlGroup"]}, cwd="/", timeout=min(10, cutover_remaining(deadline)) if deadline is not None else 10)
    need(result["returncode"] == 0, "original job state unavailable")
    props = dict(line.split("=", 1) for line in result["stdout"].splitlines() if "=" in line)
    need(props.get("MainPID") == "0" and props.get("ActiveState") in {"inactive", "failed"}, "original job is not closed")
    group = props.get("ControlGroup", "")
    if group:
        need(group.startswith("/") and ".." not in Path(group).parts, "original cgroup identity differs")
        path = Path("/sys/fs/cgroup") / group.lstrip("/")
        if path.exists():
            need(not (path / "cgroup.procs").read_text().split() and not any(p.is_dir() for p in path.iterdir()), "original cgroup is not stopped")
    return props


def verify_source_successor_activation(link, frozen, activation):
    """Historical actual activation + exact original Git parent/tree/source."""
    need(activation.get("schema") == ACTIVATION_SCHEMA and activation.get("completed") is True
         and activation.get("source_commitment") == frozen["source_commitment"]
         and activation.get("old_owner_job_closed") is True, "unknown or foreign source activation")
    stage = json_object(checked_descriptor(activation["stage"]))
    need(link["git_snapshot"] == activation["stage"], "source snapshot descriptor differs")
    snapshot = stage["source_successor"]
    need(snapshot["prior_policy"] == link["prior_policy"] and snapshot["successor_policy"] == link["successor_policy"]
         and snapshot["freeze"] == link["freeze"] and snapshot["attempt_number"] == link["attempt_number"],
         "source activation chain differs from its original snapshot")
    need(frozen["source_commitment"] == digest({"base_commit": link["prior_policy"]["base_commit"], "manifest": frozen["manifest"]})
         and frozen["freeze_sha256"] == digest({k: v for k, v in frozen.items() if k != "freeze_sha256"}),
         "original source freeze commitment differs")
    timeout = link["prior_policy"]["git_timeout_seconds"]
    header = git_command(Path(link["prior_policy"]["repository_root"]), ["cat-file", "-p", snapshot["new_commit"]], timeout=timeout)
    need(header.split("\n\n", 1)[0].splitlines()[:2] == ["tree " + snapshot["new_tree"], "parent " + link["prior_policy"]["base_commit"]]
         and len([line for line in header.split("\n\n", 1)[0].splitlines() if line.startswith("parent ")]) == 1,
         "genuine source successor commit parent/tree differs")
    for relative, row in frozen["manifest"].items():
        raw = git_command(Path(link["prior_policy"]["repository_root"]),
            ["show", snapshot["new_commit"] + ":" + relative], timeout=timeout).encode("utf-8")
        need(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"],
             "genuine source successor tree is not the validated source")
    installation = json_object(checked_descriptor(activation["installation"]))
    need(installation["completed"] is True and installation["source_WRITE_closed"] is True
         and installation["postwrite_source_READ_closed"] is True
         and all(type(c["returncode"]) is int and c["returncode"] == 0 for c in installation["commands"]),
         "original source/service cutover did not return known completion")
    verify_external_activation_closed(activation["external_owner"], link["external_terminal"], link["activation"])
    for suffix, pin in activation["loaded_runtime"].items():
        observed = json_object(checked_descriptor(pin))
        need(suffix in {"watchdog", "repair"} and observed["source_commitment"] == frozen["source_commitment"]
             and observed["loaded_module"] == stage["release_sources"]["fleet_" + suffix],
             "historical actual source-body attestation differs")


def validate_external_activation_terminal(owner, properties, observed_process):
    """Pure check of an original manager terminal and birth-bound identity.

    A recycled PID is evidence that the original process has gone. Its new
    owner never inherits this activation's identity or completion authority.
    """
    need(type(owner) is dict and set(owner) == {"unit", "owner", "invocation_id"},
         "closed original external owner identity required")
    identity = owner["owner"]
    need(type(identity) is dict and set(identity) == {"pid", "birth"}
         and type(identity["pid"]) is int and identity["pid"] > 0
         and type(identity["birth"]) is str and identity["birth"].isdecimal(),
         "original external PID and birth required")
    need(type(owner["unit"]) is str and re.fullmatch(r"ipfs-taskboard-" + re.escape(REPAIR_PROFILE)
         + r"-cutover-[0-9a-f]{32}\.service", owner["unit"])
         and type(owner["invocation_id"]) is str and re.fullmatch(r"[0-9a-f]{32}", owner["invocation_id"]),
         "finite original external unit/invocation required")
    keys = {"MainPID", "InvocationID", "Result", "ExecMainStatus", "SubState", "ControlGroup", "TasksCurrent", "KillMode"}
    need(type(properties) is dict and set(properties) == keys
         and all(type(value) is str for value in properties.values())
         and properties["MainPID"] == "0" and properties["InvocationID"] == owner["invocation_id"]
         and properties["Result"] == "success" and properties["ExecMainStatus"] == "0"
         and properties["SubState"] in {"exited", "dead"}
         and properties["TasksCurrent"] == "0" and properties["KillMode"] == "control-group",
         "external activation original manager terminal is unknown")
    if observed_process is not None:
        need(type(observed_process) is dict and set(observed_process) == {"pid", "birth"}
             and type(observed_process["pid"]) is int and observed_process["pid"] == identity["pid"]
             and type(observed_process["birth"]) is str and observed_process["birth"].isdecimal()
             and observed_process["birth"] != identity["birth"],
             "original external activation PID and birth remain live or unknown")
    return properties


def observe_external_activation_closed(owner):
    """Capture the original manager's genuine terminal once, before sealing."""
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog import command
    need(type(owner) is dict and type(owner.get("unit")) is str, "original external unit required")
    result = command({"argv": ["systemctl", "--user", "show", owner["unit"],
        "--property=MainPID,InvocationID,Result,ExecMainStatus,SubState,ControlGroup,TasksCurrent,KillMode"]}, cwd="/", timeout=10)
    need(type(result["returncode"]) is int and result["returncode"] == 0,
         "original external manager observation failed")
    props = dict(line.split("=", 1) for line in result["stdout"].splitlines() if "=" in line)
    process = proc_identity(owner["owner"]["pid"])
    validate_external_activation_terminal(owner, props, process)
    return {"terminal_properties": props, "process_observation": process}


def verify_external_activation_closed(owner, terminal_pin, activation_pin):
    """Authenticate immutable historical closure; never infer it from absence.

    The hash-bound observation is required even after transient manager state
    disappears. Missing, mutable or foreign history stays held for inspection.
    """
    need(terminal_pin is not None and activation_pin is not None,
         "immutable authenticated external terminal proof required")
    activation = json_object(checked_descriptor(activation_pin))
    need(activation.get("schema") == ACTIVATION_SCHEMA and activation.get("completed") is True
         and activation.get("external_owner") == owner, "external terminal activation binding differs")
    proof = json_object(checked_descriptor(terminal_pin))
    need(not Path(terminal_pin["path"]).lstat().st_mode & 0o222,
         "external terminal proof must be immutable")
    need(set(proof) == {"schema", "activation", "external_owner", "terminal_properties",
        "process_observation", "observed_at", "original_manager_terminal_verified"}
         and proof["schema"] == "agent-supervisor/repair-external-owner-terminal@1"
         and proof["activation"] == activation_pin and proof["external_owner"] == owner
         and proof["original_manager_terminal_verified"] is True
         and type(proof["observed_at"]) in (int, float) and proof["observed_at"] > 0,
         "external terminal proof is incomplete or belongs to another activation")
    return validate_external_activation_terminal(owner, proof["terminal_properties"], proof["process_observation"])


def seal_external_activation_terminal(activation_pin, invocation):
    """One original observation; existing proof is read-only and never replayed."""
    activation = json_object(checked_descriptor(activation_pin))
    need(activation.get("schema") == ACTIVATION_SCHEMA and activation.get("completed") is True
         and Path(activation_pin["path"]) == invocation / "actual-activation.json",
         "fixed original actual activation required")
    path = invocation / "external-owner-terminal.json"
    if path.exists():
        pin = descriptor(path)
    else:
        observed = observe_external_activation_closed(activation["external_owner"])
        pin = new_json(path, {"schema": "agent-supervisor/repair-external-owner-terminal@1",
            "activation": activation_pin, "external_owner": activation["external_owner"],
            **observed, "observed_at": time.time(), "original_manager_terminal_verified": True})
    verify_external_activation_closed(activation["external_owner"], pin, activation_pin)
    return pin



INITIAL_PROFILE_BOOTSTRAP_SCHEMA = "agent-supervisor/repair-initial-profile-bootstrap@1"
INITIAL_PROFILE_POLICY_HASH = "__CHECKED_OWNER_POLICY_SHA256__"


def _bootstrap_marker_paths(value, prefix=()):
    result = set()
    if type(value) is str:
        need(INITIAL_PROFILE_POLICY_HASH not in value or value == INITIAL_PROFILE_POLICY_HASH,
             "reserved bootstrap marker embedded in unrelated text")
        if value == INITIAL_PROFILE_POLICY_HASH:
            result.add(prefix)
    elif type(value) is dict:
        for key, item in value.items():
            need(type(key) is str and INITIAL_PROFILE_POLICY_HASH not in key,
                 "reserved bootstrap marker used as a key")
            result.update(_bootstrap_marker_paths(item, prefix + (key,)))
    elif type(value) is list:
        for index, item in enumerate(value):
            result.update(_bootstrap_marker_paths(item, prefix + (index,)))
    return result


def _bootstrap_config_projection(config, policy, checked_policy_reference, owner_policy_path):
    need(type(checked_policy_reference) is dict and
         checked_policy_reference.get("path") == owner_policy_path,
         "actual checked owner policy reference required")
    raw = checked_descriptor(checked_policy_reference)
    need(json_object(raw) == policy, "actual checked policy differs from the invoked policy")
    need(not _bootstrap_marker_paths(config), "live configuration contains a bootstrap marker")
    value = json_object(canonical(config))
    paths = set()
    runtime = value["repair_worker"]["candidate_runtime"]
    for role in ("validation_owner", "adoption_owner"):
        for command in ("argv", "verify_argv"):
            argv = runtime[role][command]
            need(type(argv) is list and all(type(arg) is str for arg in argv)
                 and argv.count("--policy") == argv.count("--policy-sha256") == 1
                 and not any(arg.startswith(("--policy=", "--policy-sha256=")) for arg in argv),
                 "four exact separate owner policy arguments required")
            policy_index = argv.index("--policy") + 1
            hash_index = argv.index("--policy-sha256") + 1
            need(policy_index < len(argv) and hash_index < len(argv)
                 and argv[policy_index] == owner_policy_path
                 and argv[hash_index] == checked_policy_reference["sha256"],
                 "live owner policy path/hash differs from the actual checked policy")
            argv[hash_index] = INITIAL_PROFILE_POLICY_HASH
            paths.add(("repair_worker", "candidate_runtime", role, command, hash_index))
    need(_bootstrap_marker_paths(value) == paths and len(paths) == 4,
         "only the four structural policy hash slots may be normalized")
    return value


def _bootstrap_current_profile_contract(policy, frozen, anchor, config, board, checked_policy_reference, config_raw):
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_authorization import digest as source_digest
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_candidate import parse_candidate_policy
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_candidate_runtime import parse_runtime_policy
    from scripts.ops.agent_supervisor.install_fleet_watchdog import observe_repair_upgrade
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog import command
    need(set(anchor) == {"schema", "owner_policy_path", "original_contract", "source_snapshot",
         "expected_config_projection", "expected_unit_postimages"}
         and anchor["schema"] == INITIAL_PROFILE_BOOTSTRAP_SCHEMA,
         "closed immutable initial profile bootstrap anchor required")
    anchor_path = Path(policy["adoption"]["path"])
    need(not anchor_path.lstat().st_mode & 0o222, "bootstrap anchor must be immutable")
    base = anchor["original_contract"]
    fields = {"schema", "profile", "config", "inventory", "units", "prior_job",
              "expected_loaded_before", "job_unit", "source_successor", "initial_release"}
    need(type(base) is dict and set(base) == fields
         and base["schema"] == "agent-supervisor/repair-profile-upgrade-contract@1"
         and base["profile"] == "legal-ir" and base["source_successor"] is None,
         "original existing-profile bootstrap grant differs")
    need(type(config_raw) is bytes and json_object(config_raw) == config,
         "bootstrap config object is not bound to its original bytes")
    expected = anchor["expected_config_projection"]
    projection = _bootstrap_config_projection(config, policy, checked_policy_reference,
                                              anchor["owner_policy_path"])
    need(type(expected) is dict and projection == expected
         and _bootstrap_marker_paths(expected) == _bootstrap_marker_paths(projection),
         "installed profile differs from the complete operator bootstrap projection")
    checked_descriptor(base["inventory"])
    checked_descriptor(base["prior_job"])
    snapshot = anchor["source_snapshot"]
    need(type(snapshot) is dict and set(snapshot) == {"prior_policy", "new_commit", "new_tree",
         "source_commitment", "manifest"}, "closed initial source snapshot required")
    worker = config["repair_worker"]
    selected = parse_candidate_policy(worker)
    need(selected is not None and snapshot["prior_policy"] == worker["candidate_policy"]
         and snapshot["new_commit"] == worker["candidate_policy"]["base_commit"]
         and type(snapshot["new_tree"]) is str and len(snapshot["new_tree"]) in (40, 64)
         and re.fullmatch(r"[0-9a-f]+", snapshot["new_tree"]) is not None,
         "bootstrap source policy/commit/tree differs")
    parse_runtime_policy(worker, selected)
    manifest = snapshot["manifest"]
    need(type(manifest) is dict and set(manifest) == set(selected.source_paths),
         "bootstrap selected source manifest differs")
    initial = json_object(checked_descriptor(base["initial_release"]))
    need(initial["schema"] == "agent-supervisor/repair-source-release@1"
         and initial["snapshot_commit"] == snapshot["new_commit"]
         and initial["snapshot_tree"] == snapshot["new_tree"]
         and initial["source_commitment"] == snapshot["source_commitment"]
         == source_digest({"base_commit": snapshot["new_commit"], "manifest": manifest}),
         "bootstrap release/source commitment differs")
    release = Path(config["runtime_release"])
    need(Path(base["initial_release"]["path"]).parent == release,
         "bootstrap release is not the explicit initial source grant")
    refs = {pin["path"]: pin for pin in worker["candidate_runtime"]["source_descriptors"]}
    for rel, row in manifest.items():
        need(type(row) is dict and set(row) == {"bytes", "sha256", "executable"}
             and type(row["executable"]) is bool, "closed bootstrap source row required")
        actual = refs[str(selected.repository_root / rel)]
        released = initial["selected_sources"][rel]
        need(actual["bytes"] == released["bytes"] == row["bytes"]
             and actual["sha256"] == released["sha256"] == row["sha256"]
             and released["path"] == str(release / rel)
             and bool((selected.repository_root / rel).lstat().st_mode & 0o111) == row["executable"],
             "bootstrap current/released source differs")
        checked_descriptor(released)
    posts = anchor["expected_unit_postimages"]
    need(type(posts) is dict and set(posts) == set(base["units"]) == {"watchdog", "repair"},
         "two exact bootstrap unit postimages required")
    current_units = {}
    unit_rows = []
    config_path = Path(base["config"]["path"])
    need(config_path.is_absolute() and len(config_path.parents) >= 4
         and config_path.parents[3] / ".config/ipfs-taskboard-watchdog/legal-ir/fleet.json" == config_path,
         "original canonical fleet configuration target differs")
    need(ordinary_bytes(config_path) == config_raw, "bootstrap configuration changed before descriptor")
    config_pin = descriptor(config_path)
    need(config_pin == {"path": str(config_path), "bytes": len(config_raw),
         "sha256": hashlib.sha256(config_raw).hexdigest()},
         "bootstrap configuration changed while its descriptor was authenticated")
    for suffix, post in posts.items():
        need(type(post) is dict and set(post) == {"path", "bytes", "sha256", "mode"}
             and post["path"] == base["units"][suffix]["path"]
             and Path(post["path"]).name == "ipfs-taskboard-legal-ir-" + suffix + ".service"
             and type(post["mode"]) is int and 0 <= post["mode"] <= 0o777,
             "bootstrap unit target/mode differs")
        pin = {key: post[key] for key in ("path", "bytes", "sha256")}
        raw = checked_descriptor(pin)
        need(stat.S_IMODE(Path(pin["path"]).lstat().st_mode) == post["mode"]
             and checked_policy_reference["sha256"].encode() not in raw
             and config_pin["sha256"].encode() not in raw,
             "bootstrap unit postimage differs or contains a self-referential hash")
        current_units[suffix] = pin
        unit_rows.append({"before": pin, "postimage_utf8": raw.decode(), "mode": post["mode"]})
    current = dict(base, config=config_pin, units=current_units)
    bindings = {"candidate_policy": worker["candidate_policy"],
        "candidate_runtime": worker["candidate_runtime"],
        "authorization_epoch": board["repair_authorization_epoch"], "source_snapshot": snapshot}
    seed = {"schema": "agent-supervisor/repair-profile-upgrade-stage@1",
        "release": str(release), "source_commitment": snapshot["source_commitment"],
        "state_dir": config["state_dir"], "contract": current,
        "release_sources": initial["sources"], "selected_release_sources": initial["selected_sources"],
        "release_commitment": base["initial_release"], "source_successor": None,
        "initial_bindings": bindings, "files": unit_rows + [
            {"before": config_pin, "postimage_utf8": config_raw.decode(),
             "mode": stat.S_IMODE(config_path.lstat().st_mode)}]}
    science = next((item for item in config["boards"] if item["id"] == "legal-ir-scientific"), None)
    if science is not None and "native_executor_binding" in science:
        seed["scientific_executor_binding"] = science["native_executor_binding"]
    if science is not None and "native_executor_git_binding" in science:
        need("scientific_executor_binding" in seed, "scientific Git binding requires its exact profile")
        seed["scientific_executor_git_binding"] = science["native_executor_git_binding"]
    need(ordinary_bytes(config_path) == config_raw and descriptor(config_path) == config_pin
         and _bootstrap_config_projection(json_object(config_raw), policy, checked_policy_reference,
             anchor["owner_policy_path"]) == expected,
         "bootstrap configuration/policy changed before readiness")
    observed = observe_repair_upgrade(seed, command)
    need(ordinary_bytes(config_path) == config_raw and descriptor(config_path) == config_pin,
         "bootstrap configuration changed during readiness")
    need(set(observed["loaded_runtime"]) == {"watchdog", "repair"},
         "both genuine imported-source observations required")
    return dict(current, expected_loaded_before={suffix: json_object(checked_descriptor(pin))["owner"]
                for suffix, pin in observed["loaded_runtime"].items()})


def current_profile_contract(policy, frozen, *, checked_policy_reference=None):
    """Advance only exact prior profile postimages from the anchored chain."""
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_authorization import repair_job_directory
    grant = json_object(checked_descriptor(policy["adoption"]))
    bootstrap = grant.get("schema") == INITIAL_PROFILE_BOOTSTRAP_SCHEMA
    if bootstrap:
        need(set(grant) == {"schema", "owner_policy_path", "original_contract", "source_snapshot",
             "expected_config_projection", "expected_unit_postimages"}
             and type(grant["original_contract"]) is dict,
             "closed initial profile bootstrap anchor required")
        base = grant["original_contract"]
    else:
        base = grant
    need(base["source_successor"] is None, "immutable adoption contract must not contain a supplied successor")
    config_raw = ordinary_bytes(Path(base["config"]["path"]))
    config = json_object(config_raw)
    board = next((b for b in config["boards"] if b["id"] == frozen["binding"]["board_id"]), None)
    need(board is not None, "original configured repair board required")
    path = repair_job_directory(Path(config["state_dir"]), board) / "job.json"
    job = json_object(ordinary_bytes(path))
    require_current_authorization(config, board, job=job)
    chain = job.get("repair_source_successors", [])
    if not chain:
        if bootstrap:
            return _bootstrap_current_profile_contract(policy, frozen, grant, config, board,
                                                       checked_policy_reference, config_raw)
        checked_descriptor(base["config"])
        initial = json_object(checked_descriptor(base["initial_release"]))
        need(initial["snapshot_commit"] == config["repair_worker"]["candidate_policy"]["base_commit"]
             and Path(base["initial_release"]["path"]).parent == Path(config["runtime_release"]),
             "first profile source is not the explicit initial source grant")
        from scripts.ops.agent_supervisor.install_fleet_watchdog import observe_repair_upgrade
        from ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog import command
        # A legitimate restart can change a PID while preserving the exact
        # anchored source/config. Refresh only through genuine imported-source
        # attestation and original manager observations, never a lookup PID.
        seed = {"schema": "agent-supervisor/repair-profile-upgrade-stage@1", "release": config["runtime_release"],
            "source_commitment": initial["source_commitment"], "state_dir": config["state_dir"], "contract": base,
            "release_sources": initial["sources"], "selected_release_sources": initial["selected_sources"],
            "release_commitment": base["initial_release"], "files": [
                {"before": pin, "postimage_utf8": ordinary_bytes(Path(pin["path"])).decode(),
                 "mode": stat.S_IMODE(Path(pin["path"]).lstat().st_mode)}
                for pin in [*base["units"].values(), base["config"]]]}
        science = next((item for item in config["boards"] if item["id"] == "legal-ir-scientific"), None)
        if science is not None and "native_executor_binding" in science:
            seed["scientific_executor_binding"] = science["native_executor_binding"]
        if science is not None and "native_executor_git_binding" in science:
            need("scientific_executor_binding" in seed, "scientific Git binding requires its exact profile")
            seed["scientific_executor_git_binding"] = science["native_executor_git_binding"]
        observed = observe_repair_upgrade(seed, command)
        return dict(base, expected_loaded_before={suffix: json_object(checked_descriptor(pin))["owner"]
                    for suffix, pin in observed["loaded_runtime"].items()})
    latest = json_object(checked_descriptor(chain[-1]))
    actual = json_object(checked_descriptor(latest["activation"]))
    previous = json_object(checked_descriptor(actual["stage"]))
    from scripts.ops.agent_supervisor.install_fleet_watchdog import verify_repair_upgrade, observe_repair_upgrade
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog import command
    verify_repair_upgrade(previous, installed=True)
    observed = observe_repair_upgrade(previous, command)
    need(previous["source_successor"]["successor_policy"] == config["repair_worker"]["candidate_policy"],
         "profile preimage is not the authenticated source-chain head")
    current = dict(base, config=descriptor(Path(base["config"]["path"])),
        units={suffix: descriptor(Path(pin["path"])) for suffix, pin in base["units"].items()},
        expected_loaded_before={suffix: json_object(checked_descriptor(pin))["owner"]
                               for suffix, pin in observed["loaded_runtime"].items()})
    # No new target, route, budget, epoch, inventory or legacy-job authority.
    need({k: v for k, v in current.items() if k not in {"config", "units", "expected_loaded_before"}}
         == {k: v for k, v in base.items() if k not in {"config", "units", "expected_loaded_before"}},
         "source successor widened the immutable profile contract")
    return current


def adopt(args):
    policy = checked_policy(args.policy, args.policy_sha256)
    frozen = frozen_candidate(args.freeze_path, args.source_commitment, args.candidate_root)
    pending = json_object(ordinary_bytes(args.validation_path))
    need(pending.get("stage") == "adoption_pending" and pending.get("freeze_sha256") == frozen["freeze_sha256"], "accepted original validation required")
    validation_receipt = pending["validation"]
    validation_invocation = Path(validation_receipt["original_receipt_reference"]).parent
    verify_validation_evidence(policy,
        json_object(ordinary_bytes(validation_invocation / "validation-plan.json")),
        original_execution_request(validation_invocation),
        json_object(checked_descriptor(validation_receipt["original_receipt_descriptor"])), validation_invocation)
    checked_reference = descriptor(args.policy)
    need(checked_reference["sha256"] == args.policy_sha256,
         "actual checked owner policy changed before adoption")
    contract = current_profile_contract(policy, frozen, checked_policy_reference=checked_reference)
    contract["source_successor"] = stage_source_successor(args, policy, frozen, contract)
    from scripts.ops.agent_supervisor.install_fleet_watchdog import prepare_repair_upgrade
    stage = prepare_repair_upgrade(args.candidate_root, contract, frozen["source_commitment"])
    stage_pin = new_json(args.invocation_root / "profile-upgrade-stage.json", stage)
    body = {"schema": DEFERRED_SCHEMA, "adopted": False, "known_deferred": True,
        "freeze_sha256": frozen["freeze_sha256"], "source_commitment": args.source_commitment,
        "validation_sha256": pending["validation_sha256"], "stage": stage_pin,
        "activation_path": str(args.invocation_root / "activation-receipt.json"),
        "policy": descriptor(args.policy), "original_receipt_reference": stage_pin["path"],
        "original_receipt_descriptor": stage_pin, "owner_command_replay_authorized": False}
    new_json(args.receipt_path, body)


def consume_deferred_adoption(receipt: dict, invocation: Path):
    need(receipt.get("schema") == DEFERRED_SCHEMA and receipt.get("known_deferred") is True
         and receipt.get("adopted") is False and receipt.get("owner_command_replay_authorized") is False,
         "known staged adoption required")
    checked_descriptor(receipt["stage"])
    path = Path(receipt["activation_path"])
    need(path == invocation / "activation-receipt.json", "fixed original activation receipt path required")
    if not path.exists():
        return None
    result = json_object(ordinary_bytes(path))
    need(result.get("schema") == ADOPTION_SCHEMA and result.get("adopted") is True
         and result.get("freeze_sha256") == receipt["freeze_sha256"]
         and result.get("validation_sha256") == receipt["validation_sha256"]
         and result.get("loaded_source_commitment") == receipt["source_commitment"], "actual activation belongs to another candidate")
    verify_actual_activation(json_object(checked_descriptor(result["original_receipt_descriptor"])),
        external_terminal=result.get("external_terminal"), activation_pin=result["original_receipt_descriptor"])
    return result, path


def verify_adoption(args):
    checked_policy(args.policy, args.policy_sha256)
    receipt = json_object(ordinary_bytes(args.receipt_path))
    need(receipt["freeze_sha256"] == json_object(ordinary_bytes(args.freeze_path))["freeze_sha256"]
         and receipt["source_commitment"] == args.source_commitment, "adoption binding differs")
    if receipt["schema"] == DEFERRED_SCHEMA:
        consume_deferred_adoption(receipt, args.invocation_root)
    else:
        verify_actual_activation(json_object(checked_descriptor(receipt["original_receipt_descriptor"])),
            external_terminal=receipt.get("external_terminal"), activation_pin=receipt["original_receipt_descriptor"])


def verify_actual_activation(value, *, external_terminal=None, activation_pin=None):
    need(value.get("schema") == ACTIVATION_SCHEMA and value.get("completed") is True
         and value.get("old_owner_job_closed") is True, "actual independently closed activation required")
    from scripts.ops.agent_supervisor.install_fleet_watchdog import verify_repair_upgrade, observe_repair_upgrade
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog import command
    stage = json_object(checked_descriptor(value["stage"]))
    verify_repair_upgrade(stage, installed=True)
    verify_external_activation_closed(value["external_owner"], external_terminal, activation_pin)
    # A normal subsequent service restart does not invalidate the historical
    # cutover. Require a fresh original observation of the SAME source/config,
    # with its own real current PID/birth/invocation attestation.
    observe_repair_upgrade(stage, command)
    return value


def request_deferred_activation(config, board, job_path, runner):
    """Outer dispatcher only, after its separate original job actually exited."""
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_candidate_runtime as runtime
    job = json_object(ordinary_bytes(job_path))
    bootstrap_pin = job.get("candidate_runtime", {}).get("bootstrap_result")
    if not bootstrap_pin:
        return False
    bootstrap = json_object(checked_descriptor(bootstrap_pin))
    pending_pin = bootstrap.get("deferred_adoption")
    if not pending_pin:
        return False
    need(bootstrap.get("original_checkout_lease_closed") is True
         and bootstrap.get("provider_known_closed") is True
         and bootstrap.get("stage") == "adoption_pending", "original owner/bootstrap has not closed")
    require_current_authorization(config, board, job=job)
    unit = runtime._json(runtime._wire(runner({"argv": ["systemctl", "--user", "show", config["repair_worker"]["job_unit"],
        "--property=MainPID,ActiveState"]}, cwd="/", timeout=10)))
    need(unit["returncode"] == 0 and "MainPID=0" in unit["stdout"].splitlines()
         and any(line in {"ActiveState=inactive", "ActiveState=failed"} for line in unit["stdout"].splitlines()), "original repair job is still live or unknown")
    pending = json_object(checked_descriptor(pending_pin)); invocation = Path(pending_pin["path"]).parent
    intent = invocation / "external-activation-request.json"
    if intent.exists():
        return False  # Unknown request outcomes are never replayed.
    policy = json_object(checked_descriptor(pending["policy"]))
    argv = ["systemd-run", "--user", "--unit=ipfs-taskboard-" + REPAIR_PROFILE + "-cutover-" + job["repair_unit_token"],
        "--property=RemainAfterExit=yes", "--property=KillMode=control-group", "--property=RuntimeMaxSec=120",
        "--property=MemoryMax=256M", "--property=TasksMax=32", "--property=CPUQuota=100%",
        "--", policy["python"], "-B", str(Path(__file__).absolute()), "activate", "--deferred", pending_pin["path"],
        "--deferred-sha256", pending_pin["sha256"], "--bootstrap", bootstrap_pin["path"], "--bootstrap-sha256", bootstrap_pin["sha256"]]
    new_json(intent, {"schema": DEFERRED_SCHEMA, "argv": argv, "bootstrap": bootstrap_pin, "deferred": pending_pin, "replay_authorized": False})
    result = runner({"argv": argv}, cwd="/", timeout=10)
    new_json(invocation / "external-activation-request-terminal.json", {"returncode": result["returncode"], "request": descriptor(intent)})
    return result["returncode"] == 0


def activate(args):
    pending_pin = {"path": str(args.deferred), "bytes": len(ordinary_bytes(args.deferred)), "sha256": args.deferred_sha256}
    bootstrap_pin = {"path": str(args.bootstrap), "bytes": len(ordinary_bytes(args.bootstrap)), "sha256": args.bootstrap_sha256}
    pending = json_object(checked_descriptor(pending_pin)); bootstrap = json_object(checked_descriptor(bootstrap_pin))
    need(bootstrap.get("deferred_adoption") == pending_pin and bootstrap.get("original_checkout_lease_closed") is True
         and bootstrap.get("provider_known_closed") is True and bootstrap.get("stage") == "adoption_pending", "genuine original owner closure required")
    invocation = args.deferred.parent; plan = json_object(checked_descriptor(pending["stage"]))
    need(Path(pending["activation_path"]) == invocation / "activation-receipt.json", "activation path differs")
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog import command
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_candidate_runtime import _unit_snapshot
    from scripts.ops.agent_supervisor.install_fleet_watchdog import apply_repair_upgrade, observe_repair_upgrade
    lock_path = Path(plan["state_dir"]) / "repair-profile-cutover.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        unit = "ipfs-taskboard-" + REPAIR_PROFILE + "-cutover-" + plan["source_successor"]["attempt_token"] + ".service"
        identity = proc_identity(os.getpid())
        clock = original_cutover_clock(unit, identity, command)
        deadline = clock["deadline_monotonic"]
        def bounded_runner(spec, *, cwd, timeout):
            return command(spec, cwd=cwd, timeout=min(timeout, cutover_remaining(deadline)))
        outer = _unit_snapshot(unit, {"memory_max_bytes": 256 * 1024**2,
            "cpu_quota_percent": 100, "tasks_max": 32, "runtime_max_seconds": 120}, bounded_runner)
        need(outer["invocation_id"] == clock["invocation_id"], "original cutover invocation changed")
        external_owner = {"unit": unit, "owner": identity, "invocation_id": outer["invocation_id"]}
        new_json(invocation / "activation-started.json", {"deferred": pending_pin, "bootstrap": bootstrap_pin, "replay_authorized": False})
        # The public installer rechecks the stopped original job and every CAS
        # immediately before its first write; this process is a separate unit.
        installation = apply_repair_upgrade(plan, bounded_runner, deadline=deadline)
        installation_pin = new_json(invocation / "profile-installation.json", installation)
        observed = observe_repair_upgrade(plan, bounded_runner, deadline=deadline)
        actual = {"schema": ACTIVATION_SCHEMA, "completed": True, "stage": pending["stage"],
            "installation": installation_pin, "source_commitment": pending["source_commitment"],
            "old_owner_job_closed": True, "bootstrap": bootstrap_pin, "deferred": pending_pin,
            "external_owner": external_owner, "external_owner_exit_not_yet_observed": True,
            "cutover_clock": clock, "completed_monotonic": time.monotonic(), **observed}
        # This immutable pre-exit fact is not an accepted adoption receipt.
        # The outer dispatcher observes the original manager's genuine terminal
        # before publishing any processes_closed/positive-adoption assertion.
        cutover_remaining(deadline)
        new_json(invocation / "actual-activation.json", actual)
    finally:
        os.close(fd)


def resume_source_publication(config, board, job_path):
    """Read-only existing proof + exact job CAS, never a cutover replay."""
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_watchdog import lock, write_json
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_authorization import parse_authorization_epoch
    epoch = parse_authorization_epoch(Path(config["state_dir"]), board)
    if epoch is None or not epoch.value["not_before_observed_at"] <= time.time() < epoch.value["expires_at"]:
        return False
    job_pin = descriptor(job_path); job = json_object(checked_descriptor(job_pin))
    if job.get("status") != "adoption_pending":
        return False
    bootstrap_pin = job.get("candidate_runtime", {}).get("bootstrap_result")
    if not bootstrap_pin:
        return False
    bootstrap = json_object(checked_descriptor(bootstrap_pin)); pending_pin = bootstrap.get("deferred_adoption")
    if not pending_pin:
        return False
    pending = json_object(checked_descriptor(pending_pin)); invocation = Path(pending_pin["path"]).parent
    actual_path = invocation / "actual-activation.json"
    if not actual_path.exists():
        return False  # Unknown partial effects remain held, never replayed.
    actual_pin = descriptor(actual_path); actual = json_object(checked_descriptor(actual_pin))
    external_terminal = seal_external_activation_terminal(actual_pin, invocation)
    verify_actual_activation(actual, external_terminal=external_terminal, activation_pin=actual_pin)
    need(bootstrap["original_checkout_lease_closed"] is True and bootstrap["provider_known_closed"] is True
         and bootstrap["attempt_token"] == job["repair_unit_token"]
         and type(job["attempts"]) is int and bootstrap["attempts"] == job["attempts"]
         and job.get("repair_authorization_sha256") == epoch.commitment, "original epoch/job/bootstrap differs")
    stage = json_object(checked_descriptor(actual["stage"])); snapshot = stage["source_successor"]
    receipt = {"schema": ADOPTION_SCHEMA, "adopted": True,
        "freeze_sha256": pending["freeze_sha256"], "validation_sha256": pending["validation_sha256"],
        "source_commitment": pending["source_commitment"], "loaded_source_commitment": pending["source_commitment"],
        "runtime_release": stage["release"], "source_closed": True, "native_closed": True,
        "processes_closed": True, "storage_closed": True, "cleanup_errors": [],
        "original_receipt_reference": actual_pin["path"], "original_receipt_descriptor": actual_pin,
        "external_terminal": external_terminal}
    receipt_path = invocation / "activation-receipt.json"
    if receipt_path.exists():
        need(json_object(ordinary_bytes(receipt_path)) == receipt, "existing final adoption receipt differs")
    else:
        new_json(receipt_path, receipt)
    chain = job.get("repair_source_successors", [])
    link_path = invocation / "source-successor-link.json"
    if link_path.exists() and chain and chain[-1] == descriptor(link_path):
        return False
    link = {"schema": "agent-supervisor/repair-source-successor@1", "epoch_sha256": epoch.commitment,
        "attempt_number": job["attempts"], "previous_link": chain[-1] if chain else None,
        "prior_policy": snapshot["prior_policy"], "successor_policy": snapshot["successor_policy"],
        "freeze": snapshot["freeze"], "validation": snapshot["validation"],
        "activation": actual_pin, "git_snapshot": actual["stage"], "external_terminal": external_terminal}
    link_path = invocation / "source-successor-link.json"
    if link_path.exists():
        need(json_object(ordinary_bytes(link_path)) == link, "existing source link differs; no fork/replay")
        link_pin = descriptor(link_path)
    else:
        link_pin = new_json(link_path, link)
    if chain and chain[-1] == link_pin:
        return False
    successor_job = {**job, "repair_source_successors": [*chain, link_pin]}
    # Authenticate the entire proposed chain before publishing the only job
    # field that changes. Attempts, token, epoch, history and expiry are exact.
    require_current_authorization(config, board, job=successor_job)
    with lock(job_path.parent / "queue.lock") as acquired:
        need(acquired and descriptor(job_path) == job_pin, "original job CAS changed before source-link publication")
        write_json(job_path, successor_job)
    return True


def serve(args):
    """Fixed two-service entrypoint with a genuine imported-source attestation."""
    config = json_object(ordinary_bytes(args.config)); release = Path(config["runtime_release"])
    metadata = json_object(ordinary_bytes(release / "repair-source-commitment.json"))
    module_name = "fleet_watchdog" if args.service == "watchdog" else "fleet_repair"
    module = importlib.import_module("ipfs_accelerate_py.agent_supervisor.rescue." + module_name)
    source = descriptor(Path(module.__file__).absolute())
    need(source == metadata["sources"][module_name] and release in Path(source["path"]).parents,
         "actual imported module is outside the immutable repair release")
    invocation = os.environ.get("INVOCATION_ID", "")
    need(re.fullmatch(r"[0-9a-f]{32}", invocation) is not None, "actual systemd invocation required")
    root = Path(config["state_dir"]) / "runtime-loaded"; root.mkdir(mode=0o700, exist_ok=True)
    new_json(root / (invocation + "-" + args.service + ".json"), {"schema": ACTIVATION_SCHEMA,
        "service": args.service, "owner": proc_identity(os.getpid()), "invocation_id": invocation,
        "source_commitment": metadata["source_commitment"], "loaded_module": source,
        "config": descriptor(args.config), "release": str(release)})
    return module.main((["--apply"] if args.service == "watchdog" else ["run"]) + ["--config", str(args.config)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "verify-validation", "verify-negative-validation", "adopt", "verify-adoption", "activate", "serve"))
    for name in ("policy", "candidate-root", "freeze-path", "receipt-path", "validation-path", "invocation-root", "plan", "deferred", "bootstrap", "config"):
        parser.add_argument("--" + name, type=Path)
    for name in ("policy-sha256", "source-commitment", "plan-sha256", "deferred-sha256", "bootstrap-sha256"):
        parser.add_argument("--" + name)
    parser.add_argument("--service", choices=("watchdog", "repair"))
    args = parser.parse_args(argv)
    required = {"activate": ("deferred", "deferred_sha256", "bootstrap", "bootstrap_sha256"), "serve": ("config", "service")}
    fields = required.get(args.action, ("policy", "policy_sha256", "candidate_root", "freeze_path", "receipt_path", "source_commitment", "invocation_root"))
    if any(getattr(args, name) is None for name in fields) or args.action == "adopt" and args.validation_path is None:
        parser.error("all fixed owner binding arguments are required")
    dispatch = {"validate": validate, "verify-validation": verify_validation, "verify-negative-validation": verify_negative_validation, "adopt": adopt, "verify-adoption": verify_adoption,
        "activate": activate, "serve": serve}
    result = dispatch[args.action](args)
    return 0 if result is None else result


if __name__ == "__main__":
    raise SystemExit(main())
