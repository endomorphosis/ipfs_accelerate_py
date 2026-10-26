"""Offline prepared, parent-run authentic fixture jobs; never R06 recovery.

prepare_launch and maintenance each enter ONE real existing F07 root/phase/disk
owner directly. Do not wrap those commands in another native owner. launch
acquires no resource root and calls the exact original v05 main unchanged; the
parent may instead execute that original v05 file directly. Only the original
owner/NativeFenceObserver emits its genuine orchestration custody receipt.

This wrapper creates distinct fixture records, never substitute native ACKs,
launch receipts, provider outcomes, administrative decisions or retry authority.
All imports/actions below occur only when a root-appointed native job executes.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import threading
import time
from types import ModuleType

ORIGINAL_OWNER = Path("/home/barberb/lift_coding/artifacts/legal-ir-supervisor-start-20261016-01/run_native_owner_v05.py")
ORIGINAL_OWNER_SHA = "3e5357925d3be41630e73adc0b8a31b1fd25c35ad3f5bc4ad1b01dbaeffebb14"
F07 = "f07cb14289fcb6275e4497dcc36be0dcf8bfc3dc87531ff7a9537eb85fdcc661"
SCHEMA = "authentic-native-fixture-owner-request@1"
MAX_SOURCE = 8 * 1024**2
MAX_JSON = 2 * 1024**2
MAX_FILE = 64 * 1024**2
FIXTURE_PREFIX = "legal-ir-unknown-authentic-fixture-"
COMMON_KEYS = {"schema", "operation", "fixture_only", "production_access", "owner_helper_pin",
    "wrapper_pin", "source_loader_pin", "resource_source_pins", "repo_root", "controller_head",
    "fixture_root", "attempt_directory", "storage_bytes", "external_charges", "duration_seconds", "payload"}
RESOURCE_FILES = ("existing_resource_scheduler.py", "resource_scheduler.py",
                  "parented_phase_resources.py", "autoencoder_daemon_resources.py")


def need(value, message):
    if not value:
        raise ValueError(message)


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate fixture request JSON key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite fixture JSON")))
    json.dumps(value, allow_nan=False)
    return value


def raw_file(path, maximum=MAX_SOURCE, *, private=False):
    path = Path(path)
    need(path.is_absolute() and path.resolve(strict=True) == path, "fixture path is aliased")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        need(stat.S_ISREG(before.st_mode) and before.st_uid in (0, os.geteuid())
             and before.st_nlink == 1 and not before.st_mode & 0o002
             and 0 <= before.st_size <= maximum
             and (not private or (before.st_uid == os.geteuid() and stat.S_IMODE(before.st_mode) == 0o600)),
             "fixture file custody/bound differs")
        chunks, count = [], 0
        while block := os.read(descriptor, min(65536, maximum - count + 1)):
            count += len(block)
            need(count <= maximum, "fixture input exceeds bound")
            chunks.append(block)
        fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
        witness = lambda row: tuple(getattr(row, name) for name in fields)
        need(count == before.st_size and witness(before) == witness(os.fstat(descriptor)) == witness(path.lstat()),
             "fixture input changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def pin(path, maximum=MAX_SOURCE):
    raw = raw_file(path, maximum)
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def read_pin(value, maximum=MAX_SOURCE):
    need(type(value) is dict and set(value) == {"path", "bytes", "sha256"}
         and type(value["bytes"]) is int and 0 <= value["bytes"] <= maximum
         and type(value["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]),
         "fixture pin is not a closed record")
    raw = raw_file(value["path"], maximum)
    need(len(raw) == value["bytes"] and hashlib.sha256(raw).hexdigest() == value["sha256"], "fixture pin differs")
    return raw


def new_json(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()
    need(len(raw) <= MAX_JSON, "fixture output exceeds JSON bound")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            need(written > 0, "fixture output short write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return pin(path, MAX_JSON)


def module_bytes(name, value):
    raw = read_pin(value)
    module = ModuleType(name)
    module.__file__ = value["path"]
    sys.modules[name] = module
    try:
        exec(compile(raw, value["path"], "exec", dont_inherit=True), module.__dict__)
        read_pin(value)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def ticks(pid):
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def boot():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def file_identity(path):
    info = Path(path).stat()
    return dict(device=info.st_dev, inode=info.st_ino, size=info.st_size, mtime_ns=info.st_mtime_ns)


def owner_module(value):
    need(value["path"] == str(ORIGINAL_OWNER) and value["sha256"] == ORIGINAL_OWNER_SHA,
         "only the actual qualified original v05 owner is allowed")
    owner = module_bytes("_exact_original_f07_fixture_owner", value)
    need(owner.POLICY == F07 and callable(owner.main), "original v05 owner policy/main differs")
    return owner


def allowed(path, owner, *, new=False):
    path = Path(path)
    need(path.is_absolute() and path.parent.resolve(strict=True) == path.parent
         and any(root == path or root in path.parents for root in owner.ROOTS)
         and not any(part.is_symlink() for part in (path, *path.parents)), "fixture path outside actual native owner roots")
    if new:
        need(not path.exists(), "fixture output must be new")
    return path


def request_data(path, expected_sha):
    raw = raw_file(path, MAX_JSON, private=True)
    need(hashlib.sha256(raw).hexdigest() == expected_sha, "owner wrapper request SHA differs")
    value = decode(raw)
    need(type(value) is dict and set(value) == COMMON_KEYS and value["schema"] == SCHEMA
         and value["fixture_only"] is True and value["production_access"] is False
         and value["operation"] in {"prepare_launch", "launch", "maintenance"}, "owner wrapper request fields differ")
    read_pin(value["wrapper_pin"])
    need(value["wrapper_pin"]["path"] == str(Path(__file__).resolve(strict=True)), "owner wrapper source path differs")
    owner = owner_module(value["owner_helper_pin"])
    root = allowed(value["fixture_root"], owner)
    need(root.name.startswith(FIXTURE_PREFIX) and "legal-ir-grok-parallel-20261016-r06" not in str(root),
         "only a distinctly named disposable authentic fixture is allowed")
    attempt = allowed(value["attempt_directory"], owner, new=True)
    need(attempt != root and not attempt.is_relative_to(root) and not root.is_relative_to(attempt),
         "fixture and direct owner attempt must be separate")
    need(type(value["storage_bytes"]) is int and 192 * 1024**2 <= value["storage_bytes"] <= 512 * 1024**2
         and type(value["duration_seconds"]) is int and 10 <= value["duration_seconds"] <= 120
         and type(value["external_charges"]) is list
         and value["external_charges"] == [{"path": str(root), "bytes": 128 * 1024**2}],
         "fixture native resource envelope differs")
    repo = Path(value["repo_root"])
    need(repo.is_absolute() and repo.resolve(strict=True) == repo
         and type(value["controller_head"]) is str and re.fullmatch(r"[0-9a-f]{40}", value["controller_head"]),
         "qualified product root/HEAD differs")
    loader = module_bytes("_authentic_fixture_pinned_source_loader", value["source_loader_pin"])
    resource_prefix = owner.DATASETS / "ipfs_datasets_py/optimizers/logic_theorem_optimizer"
    pins = value["resource_source_pins"]
    need(type(pins) is list and {item["path"] for item in pins} == {str(resource_prefix / name) for name in RESOURCE_FILES}
         and len(pins) == len(RESOURCE_FILES), "actual legacy native resource source closure differs")
    for item in pins:
        read_pin(item)

    def guard():
        need(raw_file(path, MAX_JSON, private=True) == raw, "owner wrapper request changed")
        for item in [value["wrapper_pin"], value["owner_helper_pin"], value["source_loader_pin"], *pins]:
            read_pin(item)
    guard()
    return value, owner, loader, guard


def safe_environment(attempt):
    # Direct fixture jobs receive no model/account/loader credentials. Original
    # v05 launch is a separate path and retains its unchanged environment rules.
    os.environ.clear()
    os.environ.update(PATH="/usr/bin:/bin", LANG="C.UTF-8", LC_ALL="C.UTF-8", TZ="UTC",
        PYTHONDONTWRITEBYTECODE="1", IPFS_DATASETS_PY_MINIMAL_IMPORTS="1",
        IPFS_AUTO_INSTALL="false", IPFS_DATASETS_AUTO_INSTALL="false", IPFS_ACCEL_SKIP_CORE="1",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
        CUDA_VISIBLE_DEVICES="", TMPDIR=str(attempt), TMP=str(attempt), TEMP=str(attempt),
        HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        XDG_CONFIG_HOME=str(attempt / "private-config"), XDG_CACHE_HOME=str(attempt / "private-cache"),
        HF_HOME=str(attempt / "private-hf"), HF_TOKEN_PATH=str(attempt / "absent-hf-token"),
        IPFS_ACCELERATE_LLM_ALLOCATION_DB=str(attempt / "fixture-allocation.duckdb"))
    os.environ["IPFS_ACCELERATE_AGENT_QUACK_PREFER"] = "0"
    sys.pycache_prefix = str(attempt / "unwritten-isolated-pycache")


def product_imports(request, loader, sources):
    repo = Path(request["repo_root"])
    need(type(sources) is dict and sources == loader._package_pins(repo), "qualified product source closure changed")
    sys.path.insert(0, str(repo))
    sys.meta_path.insert(0, loader._PinnedProductFinder(repo, sources))
    for name in tuple(sys.modules):
        if name == "ipfs_accelerate_py" or name.startswith("ipfs_accelerate_py."):
            origin = getattr(sys.modules[name], "__file__", None)
            need(origin is None or str(Path(origin)) in sources, "foreign product module was already loaded")


def external_census(path, maximum):
    """Bound the charged external tree without following filesystem aliases."""
    path = Path(path)
    try:
        root_info = path.lstat()
    except FileNotFoundError:
        return 0
    need(stat.S_ISDIR(root_info.st_mode) and not stat.S_ISLNK(root_info.st_mode)
         and root_info.st_uid == os.geteuid() and not root_info.st_mode & 0o002,
         "external fixture root custody differs")
    pending, entries, total = [path], 0, 0
    while pending:
        directory = pending.pop()
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            with os.scandir(descriptor) as children:
                for item in children:
                    entries += 1
                    need(entries <= 8192, "external fixture census entry bound exceeded")
                    info = item.stat(follow_symlinks=False)
                    need(info.st_uid == os.geteuid() and not info.st_mode & 0o002,
                         "external fixture entry custody differs")
                    if stat.S_ISDIR(info.st_mode):
                        pending.append(directory / item.name)
                    else:
                        need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                             "external fixture contains an alias or unsupported object")
                        total += info.st_size
                        need(total <= maximum, "charged external fixture storage bound exceeded")
        finally:
            os.close(descriptor)
    return total


def native_job(request, owner, guard_sources, action):
    """ONE genuine F07 root; caller must execute this adapter directly."""
    attempt = Path(request["attempt_directory"])
    attempt.mkdir(parents=True, mode=0o700, exist_ok=False)
    safe_environment(attempt)
    sys.path[:0] = [str(owner.DATASETS), request["repo_root"]]
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.existing_resource_scheduler import open_existing_scheduler
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceLane
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.parented_phase_resources import ParentedPhaseResourceReservation
    guard_sources()
    scheduler = open_existing_scheduler(owner.STATE, F07)
    cancelled = threading.Event()
    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, lambda _number, _frame: cancelled.set())
    result_pin = None
    with scheduler.acquire(ResourceLane.ORCHESTRATION, cpu_slots=1, memory_mb=2048, child_process_slots=4,
            timeout=300, cancel_event=cancelled, request_id="authentic-fixture:" + request["operation"] + ":" + attempt.name) as root:
        with root.acquire_child(lane=ResourceLane.ORCHESTRATION, cpu_slots=1, memory_mb=2048, child_process_slots=4,
                timeout=300, cancel_event=cancelled, request_id="authentic-fixture:" + attempt.name + ":phase") as phase:
            with ParentedPhaseResourceReservation(owner.LEDGER, roots=owner.ROOTS, attempt_directory=attempt,
                    parent_lease=phase, storage_bytes=request["storage_bytes"], memory_mb=2048, cpu_slots=1,
                    child_process_slots=4, scheduler_state_path=owner.STATE, expected_scheduler_policy_sha256=F07,
                    timeout_seconds=request["duration_seconds"] + 300, ledger_lock_timeout_seconds=60) as disk:
                deadline = time.monotonic() + request["duration_seconds"]
                for item in request["external_charges"]:
                    disk.account_external_bytes(hashlib.sha256(item["path"].encode()).hexdigest(), item["bytes"])

                def native_guard(*, force_ack=False, check_only=False):
                    need(type(force_ack) is bool and type(check_only) is bool
                         and not cancelled.is_set() and time.monotonic() < deadline, "actual native fixture owner expired/cancelled")
                    guard_sources()
                    # Public property performs the genuine live ancestry and
                    # exact existing policy checks; no JSON ACK is substituted.
                    actual_lease = disk.native_lease
                    need(actual_lease is not None and not phase.released and not root.released,
                         "actual native owner ancestry was released")
                    if force_ack:
                        disk.heartbeat()
                    disk.check_usage(attempt, full=check_only)
                    for charge in request["external_charges"]:
                        external_census(charge["path"], charge["bytes"])

                def retain_close_custody():
                    # Cleanup never acquires/replays a native operation. Keep
                    # the same live lexical owners even after cancellation,
                    # source drift or an expired operation deadline. A failed
                    # heartbeat is not reported as an ACK or a safe release.
                    try:
                        disk.heartbeat()
                        disk.check_usage(attempt)
                    except Exception:
                        pass

                native_guard.retain_close_custody = retain_close_custody
                native_guard(force_ack=True, check_only=True)
                result = action(attempt, native_guard, deadline)
                native_guard(force_ack=True, check_only=True)
                result_pin = new_json(attempt / "owner-result.json", {
                    "schema": "authentic-native-fixture-direct-owner-result@1", "fixture_only": True,
                    "operation": request["operation"], "original_v05_main_impersonated": False,
                    "owner_pid": os.getpid(), "owner_start_time_ticks": ticks(os.getpid()), "boot_id": boot(),
                    "root_lease_id": root.lease_id, "phase_lease_id": phase.lease_id,
                    "disk_reservation_id": disk.reservation_id, "actual_parented_native_lease_id": disk.native_lease.lease_id,
                    "scheduler_state": str(owner.STATE), "scheduler_policy_sha256": F07,
                    "result": result, "completion_authority": False, "automatic_retry_admitted": False,
                    "administrative_decision_performed": False, "finalization_pending_at_record_write": True})
                disk.finalize(attempt, artifacts_durable=True)
    # This statement follows actual disk finalization AND phase/root release.
    print(json.dumps({"owner_result_pin": result_pin, "native_finalized": True,
                      "fixture_only": True, "completion_authority": False}), flush=True)
    return 0



def close_native_custody(component, attempt, label, actual_guard):
    """Only actual close releases retained handles; never retry the operation."""
    retries = 0
    while True:
        try:
            component.close()
        except BaseException as error:
            if retries == 0:
                # Diagnostic failure cannot unwind retained native custody.
                try:
                    new_json(attempt / (label + "-cleanup-pending.json"), {
                        "fixture_only": True, "cleanup_pending": True,
                        "error_type": type(error).__name__, "same_component_and_F07_contexts_retained": True,
                        "native_operation_replayed": False, "completion_authority": False})
                    print(json.dumps({"fixture_only": True, "cleanup_pending": True,
                        "component": label, "same_native_custody_retained": True}), flush=True)
                except BaseException:
                    pass
            retries += 1
            actual_guard.retain_close_custody()
            try:
                time.sleep(1)
            except BaseException:
                pass
            continue
        if retries:
            new_json(attempt / (label + "-close-verified.json"), {
                "fixture_only": True, "actual_component_close_returned": True,
                "close_retry_count": retries, "native_operation_replayed": False,
                "completion_authority": False})
        return


def prepare_launch(request, owner, loader, guard_sources):
    payload = request["payload"]
    need(type(payload) is dict and set(payload) == {"producer_pin", "profile", "board_namespace",
         "master_duration_seconds", "launch_attempt_directory"}
         and payload["profile"] in {"one_attempt", "two_attempt"}
         and type(payload["board_namespace"]) is str and payload["board_namespace"].startswith("authentic-fixture-")
         and type(payload["master_duration_seconds"]) is int and 1 <= payload["master_duration_seconds"] <= 5,
         "prepare-launch payload differs")
    root = allowed(request["fixture_root"], owner, new=True)
    launch_attempt = allowed(payload["launch_attempt_directory"], owner, new=True)
    need(launch_attempt != Path(request["attempt_directory"]), "preparation and original launch owners must be distinct")
    read_pin(payload["producer_pin"])

    def action(attempt, guard, _deadline):
        # This backend import/binding creation occurs inside actual admission.
        import duckdb
        import _duckdb
        repo = Path(request["repo_root"])
        sources = loader._package_pins(repo)
        producer = module_bytes("_authentic_fixture_producer_contract", payload["producer_pin"])
        need(producer.local_git_head(repo) == request["controller_head"], "qualified C HEAD changed")
        published_path = repo / "test/api/native_execution_crash_wal_fixture.py"
        published_pin = pin(published_path)
        published = module_bytes("_authentic_fixture_published_duckdb_loader", published_pin)
        installation = published.installed_duckdb_binding()
        inputs = attempt / "seed-inputs"
        inputs.mkdir(mode=0o700)
        installation_pin = new_json(inputs / "duckdb-installation.json", installation)
        python_pin = pin(Path(sys.executable).resolve(strict=True), 128 * 1024**2)
        producer_request = {"schema": "authentic-native-unknown-master-producer-request@1",
            "fixture_root": str(root), "repo_root": str(repo), "controller_head": request["controller_head"],
            "board_namespace": payload["board_namespace"], "profile": payload["profile"],
            "master_duration_seconds": payload["master_duration_seconds"], "product_sources": sources,
            "producer_pin": payload["producer_pin"], "python_pin": python_pin,
            "source_loader_pin": request["source_loader_pin"], "published_loader_pin": published_pin,
            "duckdb_installation_pin": installation_pin}
        producer_request_pin = new_json(inputs / "producer-request.json", producer_request)
        # Original v05's own imports remain A; the isolated child imports C.
        a_script = owner.ACCELERATOR / "scripts/run_legal_autoformalization_supervisor.py"
        a_proof = owner.ACCELERATOR / "ipfs_accelerate_py/agent_supervisor/proof/formal_verification_contracts.py"
        a_init = owner.ACCELERATOR / "ipfs_accelerate_py/__init__.py"
        actual_a_pins = [pin(item) for item in (a_script, a_proof, a_init)]
        all_pins = [request["owner_helper_pin"], request["wrapper_pin"], payload["producer_pin"],
            request["source_loader_pin"], published_pin, producer_request_pin, installation_pin,
            *request["resource_source_pins"], *actual_a_pins, *sources.values()]
        launch = {"schema": "legal-ir-native-orchestration-request@1", "purpose": "supervisor_launch",
            "name": "authentic-fixture-" + launch_attempt.name, "duration_seconds": request["duration_seconds"],
            "attempt_directory": str(launch_attempt), "storage_bytes": request["storage_bytes"],
            "external_charges": request["external_charges"], "cwd": str(repo),
            "command": [python_pin["path"], "-I", "-S", "-B", payload["producer_pin"]["path"],
                "--request", producer_request_pin["path"], "--request-sha256", producer_request_pin["sha256"]],
            "source_sha256": {item["path"]: item["sha256"] for item in all_pins},
            "native_terminal_receipt": str(root / "current-lane/master.pid.terminal.json"),
            "board_namespace": payload["board_namespace"]}
        launch_pin = new_json(inputs / "original-v05-launch-request.json", launch)
        # The complete genuine producer must pass a separate bounded ordinary
        # child run before the original supervisor-launch fence is requested.
        # Original v05 retains custody when a supervisor exits without native
        # terminal proof. This preflight supplies NO replacement fence proof.
        preflight_root = root.with_name(root.name + "-preflight")
        allowed(preflight_root, owner, new=True)
        preflight_attempt = launch_attempt.with_name(launch_attempt.name + "-preflight")
        allowed(preflight_attempt, owner, new=True)
        preflight_producer = {**producer_request, "fixture_root": str(preflight_root)}
        preflight_producer_pin = new_json(inputs / "producer-preflight-request.json", preflight_producer)
        preflight_command = [*launch["command"]]
        preflight_command[-3] = preflight_producer_pin["path"]
        preflight_command[-1] = preflight_producer_pin["sha256"]
        preflight = {**launch, "purpose": "controller_preparation", "name": launch["name"] + "-preflight",
            "attempt_directory": str(preflight_attempt), "command": preflight_command,
            "external_charges": [{"path": str(preflight_root), "bytes": 128 * 1024**2}],
            "source_sha256": {**launch["source_sha256"], preflight_producer_pin["path"]: preflight_producer_pin["sha256"]}}
        # Keep metadata honest: the ordinary preflight has no native owner
        # fence contract, even though the producer attempts its real master.
        preflight.pop("native_terminal_receipt")
        preflight.pop("board_namespace")
        preflight_pin = new_json(inputs / "original-v05-producer-preflight-request.json", preflight)
        guard(force_ack=True, check_only=True)
        return {"producer_request_pin": producer_request_pin, "original_v05_launch_request_pin": launch_pin,
            "producer_preflight_request_pin": preflight_producer_pin,
            "original_v05_producer_preflight_request_pin": preflight_pin,
            "successful_full_producer_preflight_required_before_launch": True,
            "original_owner_pin": request["owner_helper_pin"], "actual_owner_A_source_pins": actual_a_pins,
            "child_C_source_count": len(sources), "child_C_controller_head": request["controller_head"],
            "fixture_root_still_absent": not root.exists(), "run_original_v05_directly": True,
            "production_access": False, "no_model_or_provider_constructed": True}
    return native_job(request, owner, guard_sources, action)


def launch(request, owner, _loader, guard_sources):
    payload = request["payload"]
    need(type(payload) is dict and set(payload) == {"original_launch_request_pin", "preflight_owner_request_pin",
         "preflight_owner_receipt_pin", "preflight_producer_result_pin"}, "launch payload differs")
    value = decode(read_pin(payload["original_launch_request_pin"], MAX_JSON))
    checked = decode(read_pin(payload["preflight_owner_request_pin"], MAX_JSON))
    preflight = decode(read_pin(payload["preflight_owner_receipt_pin"], MAX_JSON))
    result = decode(read_pin(payload["preflight_producer_result_pin"], MAX_JSON))
    need(value["schema"] == "legal-ir-native-orchestration-request@1" and value["purpose"] == "supervisor_launch"
         and value["attempt_directory"] == request["attempt_directory"]
         and value["native_terminal_receipt"] == str(Path(request["fixture_root"]) / "current-lane/master.pid.terminal.json")
         and value["source_sha256"].get(str(ORIGINAL_OWNER)) == ORIGINAL_OWNER_SHA,
         "original launch request/custody target differs")
    need(checked.get("purpose") == "controller_preparation"
         and preflight.get("schema") == "legal-ir-native-orchestration-receipt@1"
         and preflight.get("request_sha256") == payload["preflight_owner_request_pin"]["sha256"]
         and preflight.get("status") == "completed" and preflight.get("child_returncode") == 0
         and preflight.get("owned_group_quiescent") is True and preflight.get("completion_authority") is False
         and result.get("schema") == "authentic-native-unknown-master-producer-result@1"
         and result.get("fixture_only") is True and result.get("production_store_opened") is False
         and result.get("actual_native_callback_intent_path") is True
         and result.get("provider_row_sql_synthesis") is False and result.get("portal_or_model_constructed") is False
         and result.get("controller_head") == request["controller_head"]
         and result.get("pid") == preflight.get("child_pid")
         and result.get("parent_process", {}).get("pid") == preflight.get("owner_pid")
         and checked.get("command", [])[:5] == value.get("command", [])[:5]
         and all(value["source_sha256"].get(name) == sha for name, sha in checked["source_sha256"].items()
                 if name != checked["command"][-3]), "successful exact-source full producer preflight missing")
    # Pin both closed producer requests, not merely their executable/source
    # prefix: one_attempt preflight cannot qualify a two_attempt launch.
    launch_command, preflight_command = value["command"], checked["command"]
    need(len(launch_command) == len(preflight_command) == 9
         and launch_command[1:4] == preflight_command[1:4] == ["-I", "-S", "-B"]
         and launch_command[5] == preflight_command[5] == "--request"
         and launch_command[7] == preflight_command[7] == "--request-sha256",
         "exact isolated producer argv differs")
    request_pin = result.get("request_pin")
    need(type(request_pin) is dict and request_pin.get("path") == preflight_command[6]
         and request_pin.get("sha256") == preflight_command[8]
         and checked["source_sha256"].get(request_pin["path"]) == request_pin["sha256"],
         "actual preflight result does not name its original command request")
    checked_producer = decode(read_pin(request_pin, MAX_JSON))
    launch_raw = raw_file(launch_command[6], MAX_JSON, private=True)
    need(hashlib.sha256(launch_raw).hexdigest() == launch_command[8]
         and value["source_sha256"].get(launch_command[6]) == launch_command[8],
         "actual launch producer request differs")
    launch_producer = decode(launch_raw)
    need({key: item for key, item in checked_producer.items() if key != "fixture_root"}
              == {key: item for key, item in launch_producer.items() if key != "fixture_root"}
         and launch_producer["fixture_root"] == request["fixture_root"]
         and checked_producer["fixture_root"] == request["fixture_root"] + "-preflight"
         and payload["preflight_producer_result_pin"]["path"]
              == checked_producer["fixture_root"] + "/producer-result.json"
         and result["profile"] == checked_producer["profile"]
         and result["board_namespace"] == checked_producer["board_namespace"] == value["board_namespace"]
         and result["producer_pin"] == checked_producer["producer_pin"]
         and checked_producer["repo_root"] == request["repo_root"]
         and checked_producer["controller_head"] == request["controller_head"],
         "full exact producer preflight closure differs")
    # Ordinary preflight contributes no native fence authority. It must be a
    # fully exited generation before asking original v05 for real launch proof.
    for row in [result["parent_process"], {"pid": result["pid"], "start_time_ticks": result["start_time_ticks"],
                                          "boot_id": result["boot_id"]}]:
        if row["boot_id"] == boot():
            try:
                present = ticks(row["pid"]) == row["start_time_ticks"]
            except FileNotFoundError:
                present = False
            need(not present, "original producer preflight process is still present")
    allowed(request["fixture_root"], owner, new=True)
    need(not any(name == "ipfs_accelerate_py" or name.startswith("ipfs_accelerate_py.") for name in sys.modules),
         "original v05 must import its actual A proof module in a fresh interpreter")
    guard_sources()
    previous = list(sys.argv)
    try:
        sys.argv[:] = [str(ORIGINAL_OWNER), payload["original_launch_request_pin"]["path"]]
        code = owner.main()  # Original code, sys.path, Popen and fence unchanged.
    finally:
        sys.argv[:] = previous
    module = sys.modules.get("ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts")
    expected = owner.ACCELERATOR / "ipfs_accelerate_py/agent_supervisor/proof/formal_verification_contracts.py"
    need(module is not None and Path(module.__file__).resolve(strict=True) == expected
         and value["source_sha256"].get(str(expected)) == pin(expected)["sha256"],
         "actual original v05 content-identity module origin differs")
    print(json.dumps({"fixture_only": True, "original_owner_main_called_unchanged": True,
        "actual_owner_A_content_identity_source_pin": pin(expected), "child_C_root": request["repo_root"],
        "wrapper_acquired_no_native_root": True}), flush=True)
    return code


def maintenance(request, owner, loader, guard_sources):
    payload = request["payload"]
    need(type(payload) is dict and set(payload) == {"custody_pins", "product_sources", "logical_store_id", "original_A_proof_pin"}
         and type(payload["logical_store_id"]) is str and payload["logical_store_id"], "maintenance payload differs")
    root = allowed(request["fixture_root"], owner)
    need(root.is_dir(), "authentic fixture was not produced")
    pins = payload["custody_pins"]
    need(type(pins) is dict and set(pins) == {"launch_request", "native_owner_receipt", "producer_result", "master_terminal"},
         "original genuine custody artifact population differs")
    records = {name: decode(read_pin(value, MAX_JSON)) for name, value in pins.items()}
    launch_record, receipt, produced, terminal = (records[name] for name in (
        "launch_request", "native_owner_receipt", "producer_result", "master_terminal"))
    need(pins["producer_result"]["path"] == str(root / "producer-result.json")
         and pins["master_terminal"]["path"] == str(root / "current-lane/master.pid.terminal.json")
         and launch_record.get("schema") == "legal-ir-native-orchestration-request@1"
         and receipt.get("schema") == "legal-ir-native-orchestration-receipt@1"
         and receipt.get("request_sha256") == pins["launch_request"]["sha256"]
         and receipt.get("status") == "completed" and receipt.get("child_returncode") == 0
         and receipt.get("native_stop_verified") is True and receipt.get("owned_group_quiescent") is True
         and receipt.get("cleanup_pending") is False and receipt.get("completion_authority") is False
         and produced.get("schema") == "authentic-native-unknown-master-producer-result@1"
         and produced.get("fixture_only") is True and produced.get("production_store_opened") is False
         and produced.get("provider_row_sql_synthesis") is False and produced.get("actual_native_callback_intent_path") is True
         and produced.get("portal_or_model_constructed") is False
         and produced.get("controller_head") == request["controller_head"], "genuine original fixture launch/producer differs")
    expected_stores = {"canonical": str(root / "canonical.duckdb"), "coordination": str(root / "coordination.duckdb"),
        "execution": str(root / "current-lane/crash-copy.execution.shard-0001-of-0003.duckdb")}
    need(produced["stores"] == expected_stores, "only actual newly produced fixture stores are allowed")
    parent = produced["parent_process"]
    stopped = [parent, {"pid": produced["pid"], "start_time_ticks": produced["start_time_ticks"], "boot_id": produced["boot_id"]}]
    fence = receipt.get("native_fence_proof", {})
    need(parent["pid"] == receipt.get("owner_pid") and produced["pid"] == receipt.get("child_pid") == fence.get("master_pid")
         and produced["start_time_ticks"] == fence.get("master_start_time_ticks")
         and terminal.get("schema") == "ipfs_accelerate_py.agent_supervisor.multi-supervisor-terminal@1"
         and terminal.get("all_trees_fenced") is True and terminal.get("task_completion_authority") is False
         and all(terminal.get(key) == fence.get(key) for key in ("master_pid", "active_binding_cid",
             "run_started_at_epoch_nanoseconds", "terminal_receipt_cid"))
         and launch_record.get("board_namespace") == terminal.get("label") == produced["board_namespace"]
         and launch_record.get("native_terminal_receipt") == pins["master_terminal"]["path"],
         "actual original owner/child/master generation join differs")

    def action(attempt, actual_guard, deadline):
        product_imports(request, loader, payload["product_sources"])
        producer_pin = produced["producer_pin"]
        producer = module_bytes("_authentic_fixture_maintenance_source_contract", producer_pin)
        need(producer.local_git_head(Path(request["repo_root"])) == request["controller_head"],
             "qualified C HEAD changed before native maintenance")
        from ipfs_accelerate_py.agent_supervisor.todo_daemon import planning_unknown_abandonment as policy
        from ipfs_accelerate_py.agent_supervisor.todo_daemon import native_execution_status_owner as status
        from ipfs_accelerate_py.agent_supervisor.todo_daemon import native_execution_wal_maintenance as wal
        need(terminal["terminal_receipt_cid"] == policy.content_identity({key: value for key, value in terminal.items()
                                                                        if key != "terminal_receipt_cid"}),
             "actual C terminal CID differs")
        original_proof = module_bytes("_pinned_actual_A_owner_proof_contracts", payload["original_A_proof_pin"])
        need(payload["original_A_proof_pin"]["path"] == str(owner.ACCELERATOR /
             "ipfs_accelerate_py/agent_supervisor/proof/formal_verification_contracts.py")
             and launch_record["source_sha256"].get(payload["original_A_proof_pin"]["path"]) == payload["original_A_proof_pin"]["sha256"]
             and original_proof.content_identity({key: value for key, value in terminal.items() if key != "terminal_receipt_cid"})
                 == terminal["terminal_receipt_cid"], "actual A/C plain-JSON terminal identity protocols disagree")

        def guard(*, force_ack=False, check_only=False):
            actual_guard(force_ack=force_ack, check_only=check_only)
            for value in pins.values():
                read_pin(value, MAX_JSON)
            policy.stopped_processes(stopped)
            loader._loaded_product_sources(payload["product_sources"])
            read_pin(producer_pin)
            need(producer.local_git_head(Path(request["repo_root"])) == request["controller_head"],
                 "qualified C HEAD changed during native maintenance")
        guard(force_ack=True, check_only=True)
        snapshot = decode(read_pin(produced["snapshot_pin"], MAX_FILE))
        selected = snapshot["attempt"]
        path = Path(expected_stores["execution"])
        writer = path.with_name("." + path.name + ".writer.lock")
        policy_lock = path.with_name("." + path.name + ".lock")
        need(file_identity(path) == produced["execution_file_identity"]
             and file_identity(Path(str(path) + ".wal")) == produced["wal_file_identity"], "producer physical generation drifted")
        for value in produced["closed_store_pins"].values():
            read_pin(value, MAX_FILE)
            need(not Path(value["path"] + ".wal").exists(), "closed canonical/coordinator has WAL")
        metadata = produced["metadata"]
        known = {key: value for key, value in metadata.items() if key not in {
            "process_instance_id", "state_schema_revision", "control_schema_profile_id", "control_schema_fingerprint"}}
        package = Path(request["repo_root"]) / "ipfs_accelerate_py/agent_supervisor"
        reader_paths = [package / "todo_daemon/implementation_daemon.py",
                        package / "todo_daemon/native_execution_status_owner.py", package / "task_sources/duckdb_state.py"]
        sources = {str(path): payload["product_sources"][str(path)]["sha256"] for path in reader_paths}
        bound = status.NativeExecutionStatusBinding(backend="duckdb", execution_path=str(path),
            execution_file_identity=file_identity(path), writer_lock_identity=file_identity(writer), expected_metadata=known,
            source_bindings=sources, control_store_id=payload["logical_store_id"], control_schema_version=5,
            control_revision=snapshot["task"]["revision"], expected_control_status="in_progress",
            **{key: selected[key] for key in policy.IDENTITY_FIELDS})
        backups = root / "retained-original-execution"
        backups.mkdir(mode=0o700)
        binding = wal.NativeExecutionWalMaintenanceBinding.from_dict({"schema": wal.BINDING_SCHEMA,
            "status_binding": bound.to_dict(), "wal_file_identity": file_identity(Path(str(path) + ".wal")),
            "policy_lock_identity": file_identity(policy_lock), "backup_directory": str(backups),
            "maintenance_source_sha256": payload["product_sources"][str(Path(wal.__file__).resolve(strict=True))]["sha256"]})
        binding_pin = new_json(attempt / "maintenance-binding.json", binding.to_dict())
        component = wal.NativeExecutionWalMaintenanceOwner(binding, deadline_monotonic=min(deadline, time.monotonic() + 90),
                                                           custody_guard=guard)
        try:
            report = component.run()
        finally:
            close_native_custody(component, attempt, "wal-maintenance", actual_guard)
        need(report["connection_close_verified"] is True and report["original_locks_released"] is True
             and report["logical_rows_preserved"] is True and report["all_metadata_preserved"] is True
             and report["unknown_not_settled"] is True, "native physical maintenance did not close exactly")
        receipt_pin = new_json(attempt / "maintenance-receipt.json", report)
        refreshed = replace(bound, execution_file_identity=report["execution_file_identity"])
        status_binding_pin = new_json(attempt / "status-binding.json", refreshed.to_dict())
        reader = status.NativeExecutionStatusOwner(refreshed, deadline_monotonic=min(deadline, time.monotonic() + 30))
        try:
            reader.open()
            observation = reader.observe_attempt()
        finally:
            close_native_custody(reader, attempt, "native-status", actual_guard)
        expected_attempt = {key: value for key, value in selected.items() if key not in {"schema", "interface"}}
        expected_providers = [{**{key: value for key, value in row.items() if key != "result_json"},
                               "result": policy.decode(row["result_json"])} for row in snapshot["providers"]]
        need(observation["attempt"] == expected_attempt and observation["phase_history"] == snapshot["phases"]
             and observation["effect_records"] == [] and len(observation["provider_records"]) == 1
             and observation["provider_records"] == expected_providers,
             "fresh actual status differs from native producer snapshot")
        observation_pin = new_json(attempt / "status-observation.json", observation)
        for value in produced["closed_store_pins"].values():
            read_pin(value, MAX_FILE)
        for history in snapshot["historical_unknowns"]:
            for value in history["store_pins"]:
                read_pin(value, MAX_FILE)
        guard(force_ack=True, check_only=True)
        return {"custody_pins": pins, "maintenance_binding_pin": binding_pin, "maintenance_receipt_pin": receipt_pin,
            "status_binding_pin": status_binding_pin, "status_observation_pin": observation_pin,
            "original_owner_and_child_master_stopped": stopped,
            "actual_live_F07_guard": True, "fixture_custody_seam_used": False,
            "actual_A_proof_pin": payload["original_A_proof_pin"], "A_C_protocol_checked_against_actual_terminal": True,
            "profile": produced["profile"], "both_UNKNOWN_rows_retained": len(snapshot["historical_unknowns"]) == 1,
            "provider_outcome": "unknown", "callback_closed": False, "nonreachability_asserted": False,
            "automatic_retry_admitted": False, "administrative_decision_performed": False, "completion_authority": False}
    return native_job(request, owner, guard_sources, action)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    args = parser.parse_args()
    request, owner, loader, guard = request_data(args.request, args.request_sha256)
    return {"prepare_launch": prepare_launch, "launch": launch, "maintenance": maintenance}[request["operation"]](
        request, owner, loader, guard)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Only exception type is printed; native details stay in private output.
        print(json.dumps({"fixture_failed": True, "error_type": type(error).__name__,
                          "completion_authority": False}), file=sys.stderr, flush=True)
        raise SystemExit(2)
