"""Future parent-admitted authentic ADMIN child; not a resource owner.

The actual parent must hold/register this isolated child under genuine F07
custody. This child opens only the exact original A/B canonical file. It uses
the unchanged product custody verifier, actual typed gateway, real kernel peer
grant and actual native audit RPC. No model/provider, source rearm, execution
settlement, original launch receipt or native resource acknowledgement is made.

Importing this module performs no filesystem, process or database operations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import signal
import stat
import sys
import time
from types import ModuleType
import uuid

SCHEMA = "authentic-native-unknown-admin-worker-request@2"
READY_SCHEMA = "authentic-native-unknown-admin-ready@2"
STOP_SCHEMA = "authentic-native-unknown-admin-stop@2"
EXIT_SCHEMA = "authentic-native-unknown-admin-exited@2"
MAX_JSON = 2 * 1024**2
MAX_SOURCE = 8 * 1024**2
MAX_DATABASE = 64 * 1024**2
AB_MANIFEST_SHA = "244016f11f23d14788da81456825dfb900dc2dc2cd1bb8461ab9ba2e7d28ad31"
LOADER_SHA = "c385185d64b20f1d657bfc008686961f76cab8a769f106a206354cfaa547a446"
REQUEST_KEYS = {"schema", "fixture_only", "production_access", "unbound_input_pin",
    "worker_source_pin", "python_pin", "parent_kernel_identity", "control_scope",
    "control_scope_identity", "deadline_monotonic_ns", "nonce"}
UNBOUND_KEYS = {"schema", "fixture_only", "production_access", "authority_context_sealed",
    "policy_verified", "runtime_workers_qualified", "input_compiler_pin", "AB_manifest_pin",
    "repo_root", "controller_head", "fixture_root", "product_sources", "producer_pin", "python_pin",
    "source_loader_pin", "published_loader_pin", "duckdb_installation_pin", "worker_pins",
    "canonical_pre_admin_pin", "context_preimage", "missing_live_fields", "mandatory_parent_gate"}
LIVE_FIELDS = {"registered_owner_identity", "registered_creator", "context_cid"}


def need(value, reason):
    if not value:
        raise ValueError(reason)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate worker JSON key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite worker JSON")))
    encode(value)
    return value


def read(path, maximum=MAX_JSON, *, private=False):
    path = Path(path)
    need(path.is_absolute() and path.resolve(strict=True) == path, "worker input alias")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_uid in (0, os.geteuid())
             and before.st_nlink == 1 and not before.st_mode & 0o002
             and 0 <= before.st_size <= maximum, "worker input custody/bound differs")
        need(not private or (before.st_uid == os.geteuid() and stat.S_IMODE(before.st_mode) == 0o600),
             "worker private input mode differs")
        chunks, count = [], 0
        while block := os.read(fd, min(65536, maximum - count + 1)):
            count += len(block)
            need(count <= maximum, "worker input exceeds bound")
            chunks.append(block)
        fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
        witness = lambda info: tuple(getattr(info, key) for key in fields)
        need(count == before.st_size and witness(before) == witness(os.fstat(fd)) == witness(path.lstat()),
             "worker input changed during read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def read_pin(value, maximum=MAX_JSON, *, private=False):
    need(type(value) is dict and set(value) == {"path", "bytes", "sha256"}
         and type(value["bytes"]) is int and 0 <= value["bytes"] <= maximum
         and type(value["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]),
         "worker pin fields differ")
    raw = read(value["path"], maximum, private=private)
    need(len(raw) == value["bytes"] and hashlib.sha256(raw).hexdigest() == value["sha256"], "worker pin differs")
    return raw


def pin(path, maximum=MAX_JSON, *, private=False):
    raw = read(path, maximum, private=private)
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def write_new(path, value=None, *, raw=None):
    raw = encode(value) + b"\n" if raw is None else raw
    need(len(raw) <= MAX_JSON, "worker output exceeds bound")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        position = 0
        while position < len(raw):
            count = os.write(fd, raw[position:])
            need(count > 0, "worker output short write")
            position += count
        os.fsync(fd)
    finally:
        os.close(fd)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return pin(path, private=True)


def ticks(pid):
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def boot():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def scope_identity(path):
    info = Path(path).lstat()
    need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
         and stat.S_IMODE(info.st_mode) == 0o700, "worker control scope is not private")
    return {"device": info.st_dev, "inode": info.st_ino, "uid": info.st_uid, "mode": stat.S_IMODE(info.st_mode)}


def socket_identity(path):
    info = Path(path).lstat()
    need(stat.S_ISSOCK(info.st_mode) and info.st_uid == os.geteuid()
         and stat.S_IMODE(info.st_mode) == 0o600 and info.st_nlink == 1,
         "actual gateway socket identity differs")
    return {"device": info.st_dev, "inode": info.st_ino, "uid": info.st_uid, "mode": stat.S_IMODE(info.st_mode)}


def source_module(name, value):
    raw = read_pin(value, MAX_SOURCE)
    module = ModuleType(name)
    module.__file__ = value["path"]
    sys.modules[name] = module
    try:
        exec(compile(raw, value["path"], "exec", dont_inherit=True), module.__dict__)
        read_pin(value, MAX_SOURCE)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def forbid_external(event, args):
    if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.exec", "os.fork", "os.forkpty",
                 "socket.connect", "socket.getaddrinfo"}:
        raise PermissionError("authentic ADMIN child forbids external dispatch")
    if event == "socket.__new__" and len(args) > 1 and args[1] != 1:
        raise PermissionError("authentic ADMIN child permits local listener only")


def safe_environment(control):
    # The parent must also launch with a closed environment; this secondary
    # boundary prevents inherited credentials reaching backend/product imports.
    os.environ.clear()
    os.environ.update(PATH="/usr/bin:/bin", LANG="C.UTF-8", LC_ALL="C.UTF-8", TZ="UTC",
        PYTHONDONTWRITEBYTECODE="1", IPFS_DATASETS_PY_MINIMAL_IMPORTS="1", IPFS_ACCEL_SKIP_CORE="1",
        IPFS_AUTO_INSTALL="false", IPFS_DATASETS_AUTO_INSTALL="false", CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
        TMPDIR=str(control), TMP=str(control), TEMP=str(control), HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", IPFS_ACCELERATE_AGENT_QUACK_PREFER="0",
        XDG_CONFIG_HOME=str(control / "private-config"), XDG_CACHE_HOME=str(control / "private-cache"),
        HF_HOME=str(control / "private-hf"), HF_TOKEN_PATH=str(control / "absent-hf-token"),
        IPFS_ACCELERATE_LLM_ALLOCATION_DB=str(control / "fixture-allocation.duckdb"),
        IPFS_ACCELERATE_META_INDEX_DUCKDB=str(control / "fixture-meta-index.duckdb"))
    sys.pycache_prefix = str(control / "unwritten-isolated-pycache")


def run(request_path, expected_sha):
    need(sys.flags.isolated == 1 and sys.flags.no_site == 1 and sys.dont_write_bytecode is True,
         "authentic ADMIN child requires Python -I -S -B")
    # Native bind precedes the gateway's chmod. Keep that actual publication
    # window private without changing any existing A/B file permissions.
    os.umask(0o077)
    signal.setitimer(signal.ITIMER_REAL, 35)
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_DATABASE, MAX_DATABASE))
    resource.setrlimit(resource.RLIMIT_NOFILE, (192, 192))
    raw = read(request_path, private=True)
    need(type(expected_sha) is str and re.fullmatch(r"[0-9a-f]{64}", expected_sha)
         and hashlib.sha256(raw).hexdigest() == expected_sha, "worker request SHA differs")
    request = decode(raw)
    need(type(request) is dict and set(request) == REQUEST_KEYS and request["schema"] == SCHEMA
         and request["fixture_only"] is True and request["production_access"] is False
         and type(request["nonce"]) is str and re.fullmatch(r"[0-9a-f]{32}", request["nonce"]),
         "worker request closed scope differs")
    parent = request["parent_kernel_identity"]
    need(type(parent) is dict and set(parent) == {"pid", "start_time_ticks", "boot_id", "process_birth_id"}
         and type(parent["pid"]) is int and parent["pid"] > 1 and parent["pid"] == os.getppid()
         and type(parent["start_time_ticks"]) is int and parent["start_time_ticks"] > 0
         and parent["start_time_ticks"] == ticks(os.getppid()) and parent["boot_id"] == boot()
         and type(parent["process_birth_id"]) is str and parent["process_birth_id"], "actual native parent generation differs")
    need(os.getpgrp() == os.getpid() and os.getsid(0) == os.getpid(), "worker is not its admitted isolated group leader")
    deadline = request["deadline_monotonic_ns"]
    need(type(deadline) is int and 0 < deadline - time.monotonic_ns() <= 30_000_000_000,
         "worker finite deadline differs")
    signal.setitimer(signal.ITIMER_REAL, (deadline - time.monotonic_ns()) / 1e9 + 5)
    control = Path(request["control_scope"])
    need(control.is_absolute() and control.resolve(strict=True) == control
         and request_path == control / "admin-request.json" and len(os.fsencode(control / "admin.sock")) <= 107
         and encode(scope_identity(control)) == encode(request["control_scope_identity"])
         and {path.name for path in control.iterdir()} == {"admin-request.json"}, "exact private worker scope differs")
    need(request["worker_source_pin"]["path"] == str(Path(__file__).resolve(strict=True))
         and request["python_pin"]["path"] == str(Path(sys.executable).resolve(strict=True)), "exact worker/Python origins differ")
    read_pin(request["worker_source_pin"], MAX_SOURCE)
    read_pin(request["python_pin"], 128 * 1024**2)
    safe_environment(control)
    sys.addaudithook(forbid_external)
    unbound = decode(read_pin(request["unbound_input_pin"], private=True))
    need(type(unbound) is dict and set(unbound) == UNBOUND_KEYS
         and unbound["schema"] == "authentic-native-unknown-admin-unbound-input@1"
         and unbound["fixture_only"] is True and unbound["production_access"] is False
         and unbound["authority_context_sealed"] is False and unbound["policy_verified"] is False
         and unbound["runtime_workers_qualified"] is False
         and set(unbound["missing_live_fields"]) == LIVE_FIELDS
         and encode(unbound["worker_pins"]["admin"]) == encode(request["worker_source_pin"])
         and encode(unbound["python_pin"]) == encode(request["python_pin"]), "unbound compiler/worker input differs")
    read_pin(unbound["input_compiler_pin"], MAX_SOURCE)
    need(unbound["AB_manifest_pin"]["sha256"] == AB_MANIFEST_SHA, "frozen authentic A/B manifest differs")
    read_pin(unbound["AB_manifest_pin"])
    need(unbound["source_loader_pin"]["sha256"] == LOADER_SHA, "exact source-loader utilities differ")
    loader = source_module("_authentic_admin_exact_loader_utilities", unbound["source_loader_pin"])
    producer = source_module("_authentic_admin_exact_AB_producer_contract", unbound["producer_pin"])
    repo = Path(unbound["repo_root"])
    fixture = Path(unbound["fixture_root"])
    need(repo.is_absolute() and repo.resolve(strict=True) == repo
         and fixture.is_absolute() and fixture.resolve(strict=True) == fixture
         and fixture.name.startswith("legal-ir-unknown-authentic-fixture-")
         and "legal-ir-grok-parallel-20261016-r06" not in str(fixture)
         and stat.S_IMODE(fixture.stat().st_mode) == 0o700
         and not control.is_relative_to(fixture) and not fixture.is_relative_to(control), "authentic fixture/control roots differ")
    need(producer.local_git_head(repo) == unbound["controller_head"], "qualified product HEAD differs")
    sources = unbound["product_sources"]
    need(type(sources) is dict and sources == loader._package_pins(repo), "whole actual product source closure differs")
    preimage = unbound["context_preimage"]
    need(type(preimage) is dict and not LIVE_FIELDS.intersection(preimage)
         and preimage["stores"] == {"canonical": str(fixture / "canonical.duckdb"),
             "coordination": str(fixture / "coordination.duckdb"),
             "execution": str(fixture / "current-lane/crash-copy.execution.shard-0001-of-0003.duckdb")},
         "unbound context has live fields or another physical store")
    canonical = fixture / "canonical.duckdb"
    need(unbound["canonical_pre_admin_pin"]["path"] == str(canonical)
         and not Path(str(canonical) + ".wal").exists(), "canonical pre-admin generation differs")
    # Real A/B native files need not be mode600 within their private root. Do
    # not mutate their permissions or require the old copy-fixture assumption.
    read_pin(unbound["canonical_pre_admin_pin"], MAX_DATABASE)
    published_pin = unbound["published_loader_pin"]
    need(published_pin["path"] == str(repo / "test/api/native_execution_crash_wal_fixture.py"),
         "published native loader crosses product root")
    read_pin(published_pin, MAX_SOURCE)
    published = loader._published(Path(published_pin["path"]), published_pin["sha256"])
    installation_raw = read_pin(unbound["duckdb_installation_pin"], private=True)
    write_new(control / "duckdb-installation.json", raw=installation_raw)
    duckdb, installed = published._load_exact_installed_duckdb(control, hashlib.sha256(installation_raw).hexdigest())
    sys.path.insert(0, str(repo))
    sys.meta_path.insert(0, loader._PinnedProductFinder(repo, sources))
    from ipfs_accelerate_py.agent_supervisor.task_sources import typed_state_owner as typed
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import planning_unknown_abandonment as policy
    from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import content_identity
    need(set(preimage) == policy.CONTEXT_KEYS - LIVE_FIELDS, "exact native context preimage keys differ")
    custody_verifier, custody_code = policy.verify_custody, policy.verify_custody.__code__
    context_loader, context_code = policy.load_context, policy.load_context.__code__
    policy_path = str(repo / "ipfs_accelerate_py/agent_supervisor/todo_daemon/planning_unknown_abandonment.py")
    need(custody_code.co_filename == context_code.co_filename == policy_path
         and custody_verifier.__module__ == context_loader.__module__ == policy.__name__,
         "actual native policy function origin differs")
    child = {"pid": os.getpid(), "start_time_ticks": ticks(os.getpid()), "boot_id": boot(),
        "process_birth_id": typed.kernel_process_birth_id(os.getpid()),
        "process_group": os.getpgrp(), "session_id": os.getsid(0)}

    def guard(*, live_deadline=True):
        need(os.getppid() == parent["pid"] and ticks(parent["pid"]) == parent["start_time_ticks"]
             and boot() == parent["boot_id"] and typed.kernel_process_birth_id(parent["pid"]) == parent["process_birth_id"],
             "actual native parent disappeared or changed")
        need(not live_deadline or time.monotonic_ns() < deadline - 5_000_000_000, "ADMIN child graceful deadline reached")
        need(encode(scope_identity(control)) == encode(request["control_scope_identity"])
             and read(request_path, private=True) == raw, "ADMIN request/control scope changed")
        read_pin(request["worker_source_pin"], MAX_SOURCE)
        read_pin(request["unbound_input_pin"], private=True)
        need(producer.local_git_head(repo) == unbound["controller_head"], "qualified product HEAD changed")
        loader._loaded_product_sources(sources)
        need(policy.verify_custody is custody_verifier and policy.verify_custody.__code__ is custody_code
             and policy.load_context is context_loader and policy.load_context.__code__ is context_code,
             "native custody verifier was changed")

    connection, gateway = None, None
    graceful_stop = False
    failed_close = None
    context_pin = None
    try:
        guard()
        # No Intent schema installation, process-instance override, execution
        # or coordinator connection is performed by this ADMIN child.
        connection = duckdb.connect(str(canonical), config={"threads": "1", "memory_limit": "256MB",
            "autoinstall_known_extensions": "false", "autoload_known_extensions": "false",
            "enable_external_access": "false", "allow_unsigned_extensions": "false", "lock_configuration": "true"})
        host_identity = {"server_id": "authentic-admin-" + uuid.uuid4().hex,
                         "generation": 1, "store_id": preimage["logical_store_id"]}
        gateway = typed.TypedStateOwnerGateway(connection=connection, socket_path=control / "admin.sock",
            store_id=preimage["logical_store_id"], identity=host_identity)
        context = decode(encode(preimage))
        context["registered_owner_identity"] = dict(gateway.identity)
        context["registered_creator"] = {"pid": os.getpid(), "process_birth_id": child["process_birth_id"]}
        context["context_cid"] = content_identity(context)
        # This is the genuine unchanged verifier, including actual original
        # A/B launch/fence/status/maintenance and current closed sidecar joins.
        policy.load_context(context, check_closed_stores=True)
        guard()
        gateway.start()
        gateway.bind_planning_unknown_abandonment_service(expected_identity=dict(gateway.identity), authority_context=context)
        remaining = (deadline - time.monotonic_ns()) / 1e9 - 5
        need(remaining >= 1, "ADMIN startup left no native grant window")
        token, grant = gateway.issue_planning_unknown_abandonment_grant(peer_pid=parent["pid"], ttl_seconds=min(20.0, remaining))
        context_pin = write_new(control / "sealed-context.json", context)
        credential_pin = write_new(control / "admin-token.json", {
            "schema": "authentic-native-unknown-admin-credential@2", "request_sha256": expected_sha,
            "nonce": request["nonce"], "gateway_identity": dict(gateway.identity), "token": token})
        actual_socket = socket_identity(gateway.socket_path)
        guard()
        write_new(control / "admin-ready.json", {
            "schema": READY_SCHEMA, "fixture_only": True, "production_access": False,
            "request_sha256": expected_sha, "nonce": request["nonce"], "parent_kernel_identity": parent,
            "child_process": child, "gateway_identity": dict(gateway.identity),
            "socket_path": str(gateway.socket_path), "socket_identity": actual_socket,
            "sealed_context_pin": context_pin, "credential_pin": credential_pin,
            "grant": grant.public_dict(), "installed_backend": installed,
            "native_custody_verifier_preserved": True, "policy_context_verified": True,
            "native_resource_ack_created": False, "original_launch_receipt_created": False,
            "completion_authority": False})
        # Existence of the JSON file alone does not mean its writer finished.
        # A zero-byte marker is visible only after that exact ready record and
        # all referenced private records were fsynced/closed above.
        write_new(control / "admin-ready.commit", raw=b"")
        expected_stop = {"schema": STOP_SCHEMA, "request_sha256": expected_sha, "nonce": request["nonce"],
                         "parent_kernel_identity": parent, "child_process": child}
        next_guard = 0
        while time.monotonic_ns() < deadline - 5_000_000_000:
            if time.monotonic_ns() >= next_guard:
                guard()
                need(socket_identity(gateway.socket_path) == actual_socket, "actual ADMIN socket changed")
                next_guard = time.monotonic_ns() + 250_000_000
            stop = control / "admin-stop.json"
            stop_commit = control / "admin-stop.commit"
            if os.path.lexists(stop_commit):
                need(read(stop_commit, private=True) == b"", "exact ADMIN stop publication marker differs")
                need(encode(decode(read(stop, private=True))) == encode(expected_stop), "exact ADMIN stop marker differs")
                graceful_stop = True
                break
            time.sleep(0.025)
        need(graceful_stop, "ADMIN child stopped without exact parent marker")
    finally:
        # Keep this actual component until close returns. Parent retains its
        # real native resources, can observe failure and kill/reap this exact
        # child. No successful close witness is written on a failed close.
        while gateway is not None or connection is not None:
            try:
                if gateway is not None:
                    gateway.stop()
                    gateway = None
                if connection is not None:
                    connection.close()
                    connection = None
            except BaseException as error:
                if failed_close is None:
                    failed_close = type(error).__name__
                    try:
                        write_new(control / "admin-cleanup-pending.json", {
                            "schema": "authentic-native-unknown-admin-cleanup-pending@2", "fixture_only": True,
                            "request_sha256": expected_sha, "child_process": child, "error_type": failed_close,
                            "successful_close_asserted": False, "parent_native_custody_still_required": True})
                    except BaseException:
                        pass
                try:
                    time.sleep(0.1)
                except BaseException:
                    pass
    guard(live_deadline=False)
    need(graceful_stop and context_pin is not None and not os.path.lexists(control / "admin.sock"),
         "actual graceful gateway close differs")
    read_pin(context_pin, private=True)
    write_new(control / "admin-exited.json", {
        "schema": EXIT_SCHEMA, "fixture_only": True, "production_access": False,
        "request_sha256": expected_sha, "nonce": request["nonce"], "parent_kernel_identity": parent,
        "child_process": child, "sealed_context_pin": context_pin,
        "graceful_stop_requested": True, "gateway_stop_returned": True, "connection_close_returned": True,
        "close_failure_observed": failed_close is not None, "socket_absent_after_stop": True,
        "process_exit_observed_by_this_record": False, "actual_parent_reap_required": True,
        "native_resource_finalization_asserted": False, "completion_authority": False})
    os._exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    args = parser.parse_args()
    try:
        run(args.request, args.request_sha256)
    except Exception as error:
        # Never print context, token, request bytes or arbitrary native errors.
        sys.stderr.write(json.dumps({"authentic_ADMIN_worker_failed": True, "error_type": type(error).__name__,
                                     "completion_authority": False}) + "\n")
        sys.stderr.flush()
        os._exit(2)


if __name__ == "__main__":
    main()
