"""Draft retained real-F07 parent for future authentic C/D fixture gates.

Run directly, never inside another native resource owner. Runtime code below
opens ONE existing F07 root/phase/disk owner, registers each real isolated child
group, retains custody through actual child cleanup and prints completion only
after actual lexical owner release. --render reads metadata only. All gate
outputs are distinctly fixture-only observations, never original launch ACKs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time
from types import ModuleType
import uuid

HERE = Path(__file__).resolve(strict=True).parent
AB = HERE.parent / "unknown-abandonment-authentic-fixture-v01"
AB_MANIFEST_SHA = "244016f11f23d14788da81456825dfb900dc2dc2cd1bb8461ab9ba2e7d28ad31"
AB_WRAPPER_SHA = "a5263c6ab4a0db17cec22a9d857326a40d5909983fc5647930d68696e03efae1"
SCHEMA = "authentic-unknown-native-parent-request@2"
RESULT_SCHEMA = "authentic-unknown-admin-settlement-native-parent-result@2"
KEYS = {"schema", "fixture_only", "production_access", "operation", "parent_source_pin", "original_owner_pin",
        "resource_source_pins", "unbound_input_pin", "repo_root", "controller_head", "fixture_root",
        "attempt_directory", "control_scope", "storage_bytes", "duration_seconds", "payload"}
PRIOR = {"coordination": "failed", "blocked": "coordination", "terminal_event": "blocked",
         "complete": "terminal_event", "replay": "complete"}


def need(value, reason):
    if not value:
        raise ValueError(reason)


def load_ab():
    manifest_raw = (AB / "manifest-v01.json").read_bytes()
    need(hashlib.sha256(manifest_raw).hexdigest() == AB_MANIFEST_SHA, "frozen A/B manifest differs")
    manifest = json.loads(manifest_raw)
    value = next(item for item in manifest["files"] if Path(item["path"]).name == "authentic_fixture_owner_v01.py")
    raw = Path(value["path"]).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == value["sha256"] == AB_WRAPPER_SHA, "frozen utility source differs")
    module = ModuleType("_authentic_CD_AB_utilities")
    module.__file__ = value["path"]
    sys.modules[module.__name__] = module
    exec(compile(raw, value["path"], "exec", dont_inherit=True), module.__dict__)
    module.read_pin(value)
    return module


def owner_identity(ab):
    pid = os.getpid()
    start = ab.ticks(pid)
    return {"pid": pid, "start_time_ticks": start, "boot_id": ab.boot(),
            "process_birth_id": "birth:kernel:" + hashlib.sha256(f"{pid}:{start}".encode("ascii")).hexdigest()[:32]}


def stopped(ab, row):
    need(type(row["pid"]) is int and row["pid"] > 1 and type(row["start_time_ticks"]) is int,
         "stopped native parent/child identity differs")
    if row["boot_id"] == ab.boot():
        try:
            same = ab.ticks(row["pid"]) == row["start_time_ticks"]
        except FileNotFoundError:
            same = False
        need(not same, "previous actual native process generation still present")


def group_live(group):
    # Kernel observation of only this admitted isolated group, not a signal.
    for item in Path("/proc").iterdir():
        if not item.name.isdecimal():
            continue
        try:
            fields = (item / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == group and fields[0] != "Z":
            return True
    return False


def scope_identity(path):
    info = path.lstat()
    need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid() and stat.S_IMODE(info.st_mode) == 0o700,
         "actual private control scope differs")
    return {"device": info.st_dev, "inode": info.st_ino, "uid": info.st_uid, "mode": stat.S_IMODE(info.st_mode)}


class ActualChild:
    def __init__(self, ab, process, log_path):
        self.ab, self.process, self.log_path = ab, process, log_path
        self.pid, self.start, self.boot = process.pid, None, None
        self.reaped = False
        self.group_bound = False

    def bind(self):
        self.boot = self.ab.boot()
        self.start = self.ab.ticks(self.pid)
        need(os.getpgid(self.pid) == self.pid and os.getsid(self.pid) == self.pid,
             "actual child lacks isolated admitted group/session")
        self.group_bound = True

    def observe(self):
        need(self.log_path.stat().st_size <= 65536, "bounded worker diagnostic stream exceeded")
        code = self.process.poll()
        if code is not None:
            self.reaped = True
        return code

    def close(self):
        if self.process.poll() is None:
            if not self.group_bound:
                # Still the actual unreaped Popen child. No unattested group
                # signal or birth/stop witness is invented on bind failure.
                self.process.kill()
            else:
                need(self.ab.ticks(self.pid) == self.start and os.getpgid(self.pid) == self.pid,
                     "actual owned child identity changed before cleanup")
                os.killpg(self.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if not self.group_bound:
                    self.process.kill()
                else:
                    os.killpg(self.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        self.process.wait(timeout=5)
        self.reaped = True
        # A leader-dead group with surviving descendants is deliberately held
        # pending an authenticated native fence. No reused/foreign group is
        # signalled here, and uniformly bounded cleanup is not asserted.
        need(not group_live(self.pid), "actual isolated child group still live")

    def witness(self, schema, kind):
        need(self.group_bound and self.start is not None and self.reaped
             and self.process.returncode is not None and not group_live(self.pid),
             "actual child exit/reap/group absence not established")
        stopped(self.ab, {"pid": self.pid, "start_time_ticks": self.start, "boot_id": self.boot})
        return {"schema": schema, "fixture_only": True, "termination_kind": kind,
            "child_pid": self.pid, "child_start_time_ticks": self.start, "boot_id": self.boot,
            "process_group": self.pid, "child_returncode": self.process.returncode,
            "child_reaped": True, "owned_group_absent": True, "native_owner_receipt_created": False,
            "completion_authority": False}


class Runtime:
    def __init__(self, ab, request, unbound, attempt, control, disk, source_guard, cancel, deadline):
        self.ab, self.request, self.unbound = ab, request, unbound
        self.attempt, self.control, self.disk = attempt, control, disk
        self.source_guard, self.cancel, self.deadline = source_guard, cancel, deadline
        self.child = None
        self.socket_pin = None
        self.socket_ready_verified = False
        self.graceful_stop_published = False
        self.stop_pin = None
        self.control_identity = scope_identity(control)
        self.parent = owner_identity(ab)

    def control_census(self):
        need(scope_identity(self.control) == self.control_identity, "charged control root generation changed")
        total, count, pending = 0, 0, [self.control]
        while pending:
            directory = pending.pop()
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            try:
                with os.scandir(descriptor) as entries:
                    for item in entries:
                        count += 1
                        need(count <= 2048, "control census entry bound exceeded")
                        path = directory / item.name
                        try:
                            info = item.stat(follow_symlinks=False)
                        except FileNotFoundError:
                            need(path == self.control / "admin.sock" and self.socket_ready_verified
                                 and self.graceful_stop_published and self.stop_pin is not None
                                 and self.child is not None, "unexpected charged control entry disappeared")
                            self.ab.read_pin(self.stop_pin)
                            need(self.ab.raw_file(self.control / "admin-stop.commit", 0, private=True) == b"",
                                 "authenticated graceful stop publication changed")
                            code = self.child.observe()
                            same_generation = False
                            if code is None:
                                try:
                                    same_generation = self.ab.ticks(self.child.pid) == self.child.start
                                except FileNotFoundError:
                                    # The same genuine Popen may have exited
                                    # between poll and the kernel birth read.
                                    code = self.child.observe()
                            need((code is None and same_generation)
                                 or (code == 0 and self.child.reaped and not group_live(self.child.pid)),
                                 "socket disappearance lacks actual graceful child generation/reap")
                            continue
                        need(info.st_uid == os.geteuid() and not info.st_mode & 0o002,
                             "charged control entry custody differs")
                        if stat.S_ISDIR(info.st_mode):
                            pending.append(path)
                        elif stat.S_ISREG(info.st_mode):
                            need(info.st_nlink == 1, "charged control regular file alias")
                            total += info.st_size
                        else:
                            need(path == self.control / "admin.sock" and self.request["operation"] == "admin"
                                 and stat.S_ISSOCK(info.st_mode) and stat.S_IMODE(info.st_mode) in {0o600, 0o700}
                                 and info.st_nlink == 1 and self.child is not None,
                                 "unexpected special entry in charged control scope")
                            observed = {"device": info.st_dev, "inode": info.st_ino,
                                "uid": info.st_uid, "mode": stat.S_IMODE(info.st_mode)}
                            ready_published = (self.control / "admin-ready.commit").exists()
                            if observed["mode"] == 0o700 and ready_published:
                                # stat may precede the actual chmod+ready
                                # publication; refresh this same inode once.
                                refreshed = os.stat(item.name, dir_fd=descriptor, follow_symlinks=False)
                                need(stat.S_ISSOCK(refreshed.st_mode) and refreshed.st_nlink == 1
                                     and (refreshed.st_dev, refreshed.st_ino, refreshed.st_uid)
                                         == (info.st_dev, info.st_ino, info.st_uid)
                                     and stat.S_IMODE(refreshed.st_mode) == 0o600,
                                     "published ready socket is not the same chmodded0600 inode")
                                observed["mode"] = 0o600
                            if observed["mode"] == 0o700:
                                # Real bind under umask077 briefly creates0700
                                # before the genuine gateway chmods0600.
                                need(not self.socket_ready_verified and not ready_published
                                     and self.child.process.poll() is None
                                     and self.ab.ticks(self.child.pid) == self.child.start,
                                     "socket0700 is only a live owned pre-READY publication")
                            if self.socket_pin is None:
                                need(self.child.process.poll() is None
                                     and self.ab.ticks(self.child.pid) == self.child.start,
                                     "socket birth lacks actual live owned child")
                                # Temporary publication observation; actual
                                # ready/gateway/grant joins are required below.
                                self.socket_pin = observed
                            elif (self.socket_pin["mode"] == 0o700 and observed["mode"] == 0o600
                                  and all(self.socket_pin[key] == observed[key] for key in ("device", "inode", "uid"))
                                  and not self.socket_ready_verified):
                                self.socket_pin = observed
                            need(self.socket_pin == observed, "owned socket generation changed")
                            if self.child.process.poll() is not None:
                                need(self.child.reaped and self.socket_ready_verified and not group_live(self.child.pid),
                                     "stale socket lacks actual ready+reaped owner")
                            total += info.st_size
                        need(total <= 8 * 1024**2, "charged control scope exceeds8MiB")
            finally:
                os.close(descriptor)
        return total

    def guard(self, *, full=False):
        need(not self.cancel.is_set() and time.monotonic() < self.deadline, "native parent cancelled/expired")
        self.source_guard()
        need(self.disk.native_lease is not None, "actual F07 ancestry absent")
        self.disk.heartbeat()
        self.disk.check_usage(self.attempt, full=full)
        self.ab.external_census(self.request["fixture_root"], 128 * 1024**2)
        self.control_census()

    def retain_close_custody(self):
        try:
            self.disk.heartbeat()
            self.disk.check_usage(self.attempt)
        except Exception:
            pass

    def spawn(self, worker_request, role):
        need(self.child is None, "only one actual worker group admitted per native job")
        name = "admin-request.json" if role == "admin" else "settlement-request.json"
        value = self.ab.new_json(self.control / name, worker_request)
        log = self.attempt / "worker.sanitized.log"
        self.guard(full=True)
        need(str(Path(sys.executable).resolve(strict=True)) == self.unbound["python_pin"]["path"],
             "actual parent/child Python origin differs")
        self.ab.read_pin(self.unbound["python_pin"], 128 * 1024**2)
        descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        try:
            process = subprocess.Popen([str(Path(sys.executable).resolve(strict=True)), "-I", "-S", "-B",
                worker_request["worker_source_pin"]["path"], "--request", value["path"], "--request-sha256", value["sha256"]],
                cwd=self.request["repo_root"], env=dict(os.environ), stdin=subprocess.DEVNULL,
                stdout=descriptor, stderr=subprocess.STDOUT, start_new_session=True)
            # Retain the actual unreaped child before even log-FD cleanup can
            # raise. Constructor performs assignments only, no birth I/O.
            self.child = ActualChild(self.ab, process, log)
        finally:
            os.close(descriptor)
        # Keep real Popen even if kernel birth/registration fails afterward.
        self.child.bind()
        self.disk.check_usage(self.attempt, child_pid=process.pid, full=True)
        return value

    def wait_file(self, path):
        while not path.exists():
            self.guard()
            need(self.child.observe() is None, "actual worker exited before expected artifact")
            time.sleep(0.1)
        self.guard()
        self.ab.raw_file(path, 2 * 1024**2, private=True)
        return self.ab.pin(path, 2 * 1024**2)

    def wait_exit(self, expected):
        while self.child.observe() is None:
            self.guard()
            time.sleep(0.1)
        self.child.close()
        need(self.child.process.returncode == expected, "actual worker exit differs from requested gate")
        self.guard(full=True)

    def worker_base(self, role):
        return {"schema": "authentic-native-unknown-" + ("admin" if role == "admin" else "settlement") + "-worker-request@2",
            "fixture_only": True, "production_access": False, "unbound_input_pin": self.request["unbound_input_pin"],
            "worker_source_pin": self.unbound["worker_pins"][role], "python_pin": self.unbound["python_pin"],
            "parent_kernel_identity": self.parent, "control_scope": str(self.control),
            "control_scope_identity": self.control_identity,
            "deadline_monotonic_ns": min(int(self.deadline * 1e9), time.monotonic_ns() + 30_000_000_000),
            "nonce": uuid.uuid4().hex}


def publish_empty_commit(path):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def admin(runtime, loader):
    ab, request, unbound, control = runtime.ab, runtime.request, runtime.unbound, runtime.control
    payload = request["payload"]
    need(type(payload) is dict and set(payload) == {"admin_exit_mode"}
         and payload["admin_exit_mode"] in {"graceful", "crash"}, "explicit ADMIN exit gate differs")
    ab.product_imports(request, loader, unbound["product_sources"])
    from ipfs_accelerate_py.agent_supervisor.task_sources import typed_state_owner as typed
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import planning_unknown_abandonment as policy
    need(typed.kernel_process_birth_id() == runtime.parent["process_birth_id"], "actual native parent kernel birth differs")
    worker_request = runtime.worker_base("admin")
    request_pin = runtime.spawn(worker_request, "admin")
    marker = runtime.wait_file(control / "admin-ready.commit")
    need(ab.read_pin(marker) == b"", "actual ADMIN ready publication marker differs")
    ready_pin = ab.pin(control / "admin-ready.json", 2 * 1024**2)
    ready = ab.decode(ab.read_pin(ready_pin, 2 * 1024**2))
    child = ready["child_process"]
    need(ready.get("schema") == "authentic-native-unknown-admin-ready@2"
         and ready.get("fixture_only") is True and ready.get("production_access") is False
         and ready.get("request_sha256") == request_pin["sha256"] and ready.get("nonce") == worker_request["nonce"]
         and ready.get("parent_kernel_identity") == runtime.parent
         and set(child) == {"pid", "start_time_ticks", "boot_id", "process_birth_id", "process_group", "session_id"}
         and child["pid"] == runtime.child.pid and child["start_time_ticks"] == runtime.child.start
         and child["boot_id"] == runtime.child.boot
         and child["process_group"] == child["session_id"] == child["pid"]
         and child["process_birth_id"] == typed.kernel_process_birth_id(child["pid"], start_time_ticks=child["start_time_ticks"])
         and ready["socket_path"] == str(control / "admin.sock") and ready["socket_identity"] == runtime.socket_pin
         and ready.get("native_custody_verifier_preserved") is ready.get("policy_context_verified") is True
         and ready.get("native_resource_ack_created") is ready.get("original_launch_receipt_created") is False,
         "actual ADMIN ready/native child/socket/verification join differs")
    context = ab.decode(ab.read_pin(ready["sealed_context_pin"], 2 * 1024**2))
    need(ready["sealed_context_pin"]["path"] == str(control / "sealed-context.json")
         and policy.canonical_json_bytes({key: value for key, value in context.items()
             if key not in {"registered_owner_identity", "registered_creator", "context_cid"}})
             == policy.canonical_json_bytes(unbound["context_preimage"])
         and context["registered_owner_identity"] == ready["gateway_identity"]
         and context["registered_creator"] == {"pid": child["pid"], "process_birth_id": child["process_birth_id"]}
         and context["context_cid"] == policy.content_identity({key: value for key, value in context.items() if key != "context_cid"}),
         "actual sealed context was changed or has substituted live identity")
    grant = ready["grant"]
    snapshot = ab.decode(ab.read_pin(context["snapshot_pin"], 64 * 1024**2))
    expected_scope = {"task_cid": context["task_cid"], "abandonment_scope_cid": policy.content_identity({
        "context_cid": context["context_cid"], "attempt_id": snapshot["attempt"]["attempt_id"],
        "claim_id": snapshot["attempt"]["claim_id"]})}
    need(grant["client_id"] == policy.CLIENT_ID and grant["process_birth_id"] == runtime.parent["process_birth_id"]
         and grant["peer_pid"] == runtime.parent["pid"] and grant["peer_start_time_ticks"] == runtime.parent["start_time_ticks"]
         and grant["peer_uid"] == os.geteuid() and grant["allowed_operations"] == [policy.OPERATION]
         and grant["allowed_command_operations"] == grant["allowed_database_task_commands"] == []
         and grant["tenant_id"] == grant["federation_id"] == grant["authority_profile"] == ""
         and grant["entity_scopes"] == expected_scope, "real peer-bound singleton ADMIN grant differs")
    runtime.socket_ready_verified = True
    credential = ab.decode(ab.read_pin(ready["credential_pin"], 2 * 1024**2))
    ab.raw_file(ready["credential_pin"]["path"], 2 * 1024**2, private=True)
    need(ready["credential_pin"]["path"] == str(control / "admin-token.json")
         and set(credential) == {"schema", "request_sha256", "nonce", "gateway_identity", "token"}
         and credential["schema"] == "authentic-native-unknown-admin-credential@2"
         and credential["request_sha256"] == request_pin["sha256"] and credential["nonce"] == worker_request["nonce"]
         and credential["gateway_identity"] == ready["gateway_identity"], "exact private credential crosswalk differs")
    def client():
        runtime.guard()
        value = typed.TypedStateOwnerConnection(socket_path=control / "admin.sock", token=credential["token"],
            client_id=grant["client_id"], process_birth_id=grant["process_birth_id"],
            store_id=context["logical_store_id"], timeout_seconds=5)
        try:
            need(policy.canonical_json_bytes(dict(value.identity)) == policy.canonical_json_bytes(ready["gateway_identity"])
                 and policy.canonical_json_bytes(dict(value.grant)) == policy.canonical_json_bytes(grant),
                 "actual native transport handshake differs")
        except BaseException:
            value.close()
            raise
        return value
    first = client()
    genuine_request = first._request
    def discard_genuine_response(*args, **kwargs):
        genuine_request(*args, **kwargs)
        raise OSError("DISPOSABLE FIXTURE: actual committed ADMIN response intentionally discarded")
    first._request = discard_genuine_response
    lost = False
    try:
        try:
            first.record_planning_unknown_abandonment_decision(context["context_cid"], context["expected_task_revision"])
        except typed.TypedStateOwnerPlanningUnknownAbandonmentOutcomeUnknownError:
            lost = True
    finally:
        first.close()
    need(lost, "native post-response loss was not observed")
    replay_client = client()
    try:
        replay = policy.plain(replay_client.record_planning_unknown_abandonment_decision(
            context["context_cid"], context["expected_task_revision"]))
    finally:
        replay_client.close()
    need(replay.get("replayed") is True and replay.get("decision", {}).get("context_cid") == context["context_cid"],
         "same-live real ADMIN exact replay missing")
    response_pin = ab.new_json(control / "audit-response.json", replay)
    runtime.guard(full=True)
    exited_pin = None
    if payload["admin_exit_mode"] == "graceful":
        runtime.stop_pin = ab.new_json(control / "admin-stop.json", {"schema": "authentic-native-unknown-admin-stop@2",
            "request_sha256": request_pin["sha256"], "nonce": worker_request["nonce"],
            "parent_kernel_identity": runtime.parent, "child_process": child})
        publish_empty_commit(control / "admin-stop.commit")
        runtime.graceful_stop_published = True
        runtime.wait_exit(0)
        exited_pin = ab.pin(control / "admin-exited.json", 2 * 1024**2)
        exited = ab.decode(ab.read_pin(exited_pin, 2 * 1024**2))
        need(exited.get("schema") == "authentic-native-unknown-admin-exited@2"
             and exited.get("child_process") == child and exited.get("request_sha256") == request_pin["sha256"]
             and exited.get("nonce") == worker_request["nonce"] and exited.get("parent_kernel_identity") == runtime.parent
             and exited.get("gateway_stop_returned") is exited.get("connection_close_returned") is True
             and exited.get("process_exit_observed_by_this_record") is False
             and exited.get("native_resource_finalization_asserted") is False,
             "actual successful ADMIN component close record differs")
    else:
        need(runtime.child.process.poll() is None and ab.ticks(runtime.child.pid) == runtime.child.start,
             "actual owned ADMIN child changed before explicit crash gate")
        os.killpg(runtime.child.pid, signal.SIGKILL)
        runtime.wait_exit(-signal.SIGKILL)
        need(not (control / "admin-exited.json").exists(), "crash gate unexpectedly has graceful close record")
    witness = runtime.child.witness("authentic-unknown-admin-child-exit-witness@2", payload["admin_exit_mode"])
    witness.update({"request_pin": request_pin, "admin_ready_pin": ready_pin, "authority_context_pin": ready["sealed_context_pin"],
        "actual_parent_observer": runtime.parent, "actual_graceful_component_exit_pin": exited_pin,
        "canonical_wal_present_after_exit": Path(context["stores"]["canonical"] + ".wal").exists(),
        "gateway_stop_and_native_close_asserted": exited_pin is not None,
        "logical_post_response_discard_performed": True, "real_transport_drop_simulated": False})
    if payload["admin_exit_mode"] == "crash":
        need(witness["canonical_wal_present_after_exit"] is True, "actual committed ADMIN crash WAL absent")
    witness_pin = ab.new_json(control / "admin-exit-witness.json", witness)
    return {"authority_context_pin": ready["sealed_context_pin"], "audit_response_pin": response_pin,
        "admin_exit_witness_pin": witness_pin, "admin_ready_pin": ready_pin, "worker_request_pin": request_pin,
        "logical_post_response_discard_performed": True, "real_transport_drop_simulated": False,
        "same_live_gateway_exact_replay": True, "actual_child_creator_exited_and_reaped": True,
        "production_custody_verifier_replaced": False, "administrative_decision_scope": "DISPOSABLE_FIXTURE_ONLY",
        "provider_outcome": "unknown", "possible_external_effects": "unverified",
        "callback_closed": False, "nonreachability_asserted": False, "automatic_retry_admitted": False,
        "completion_authority": False}


def settle(runtime):
    ab, request, payload = runtime.ab, runtime.request, runtime.request["payload"]
    need(type(payload) is dict and set(payload) == {"stage", "admin_parent_result_pin", "prior_stage_parent_result_pin"}
         and payload["stage"] in {"failed", *PRIOR}, "explicit serial native settlement request differs")
    admin_parent = ab.decode(ab.read_pin(payload["admin_parent_result_pin"], 2 * 1024**2))
    need(admin_parent.get("schema") == RESULT_SCHEMA and admin_parent.get("operation") == "admin"
         and admin_parent.get("fixture_only") is True
         and admin_parent.get("unbound_input_pin") == request["unbound_input_pin"]
         and admin_parent.get("parent_source_pin") == request["parent_source_pin"], "original actual C parent source/input differs")
    stopped(ab, admin_parent["owner_process"])
    prior = None
    if payload["stage"] == "failed":
        need(payload["prior_stage_parent_result_pin"] is None, "first FAILED gate must follow actual C only")
    else:
        prior = ab.decode(ab.read_pin(payload["prior_stage_parent_result_pin"], 2 * 1024**2))
        need(prior.get("schema") == RESULT_SCHEMA and prior.get("operation") == "settle"
             and prior.get("stage") == PRIOR[payload["stage"]]
             and prior.get("unbound_input_pin") == request["unbound_input_pin"]
             and prior.get("parent_source_pin") == request["parent_source_pin"]
             and prior["result"]["admin_parent_result_pin"] == payload["admin_parent_result_pin"],
             "exact previous actual native stage parent differs")
        stopped(ab, prior["owner_process"])
        old_exit = ab.decode(ab.read_pin(prior["result"]["settlement_exit_witness_pin"], 2 * 1024**2))
        need(old_exit.get("child_reaped") is old_exit.get("owned_group_absent") is True
             and old_exit.get("native_owner_receipt_created") is False, "previous actual native child was not reaped")
        stopped(ab, {"pid": old_exit["child_pid"], "start_time_ticks": old_exit["child_start_time_ticks"], "boot_id": old_exit["boot_id"]})
    worker_request = runtime.worker_base("settlement")
    worker_request.update({"admin_parent_result_pin": payload["admin_parent_result_pin"], "stage": payload["stage"],
        **{key: admin_parent["result"][key] for key in ("authority_context_pin", "audit_response_pin", "admin_exit_witness_pin")}})
    request_pin = runtime.spawn(worker_request, "settlement")
    crash_stage = payload["stage"] in {"failed", "coordination", "blocked", "terminal_event"}
    runtime.wait_exit(73 if crash_stage else 0)
    output = runtime.control / ("lost-stage.json" if crash_stage else "settlement-result.json")
    output_pin = ab.pin(output, 2 * 1024**2)
    observed = ab.decode(ab.read_pin(output_pin, 2 * 1024**2))
    need(observed.get("schema") == ("authentic-unknown-native-committed-stage-loss@2" if crash_stage
        else "authentic-unknown-native-settlement-worker-result@2") and observed.get("fixture_only") is True
         and observed.get("stage") == payload["stage"] and observed.get("request_sha256") == request_pin["sha256"]
         and observed.get("nonce") == worker_request["nonce"] and observed.get("parent_kernel_identity") == runtime.parent
         and observed.get("child_pid") == runtime.child.pid and observed.get("child_start_time_ticks") == runtime.child.start
         and observed.get("boot_id") == runtime.child.boot and observed.get("process_group") == runtime.child.pid,
         "actual native worker stage/exit/output join differs")
    if crash_stage:
        need(observed.get("genuine_bound_method_returned") is True and observed.get("native_handle_close_claimed") is False,
             "actual committed crash gate lacks truthful native handle boundary")
    else:
        need(observed.get("actual_native_close_returned") is True and observed.get("production_custody_verifier_replaced") is False,
             "actual final native close/verifier observation differs")
        if payload["stage"] == "replay":
            completed = ab.decode(ab.read_pin(prior["result"]["settlement_worker_result_pin"], 2 * 1024**2))
            need(observed["result"] == completed["result"], "independently fresh D5/D6 terminal results differ")
    exit_pin = ab.new_json(runtime.control / "settlement-exit-witness.json",
        runtime.child.witness("authentic-unknown-settlement-child-exit-witness@2", "committed_stage_crash" if crash_stage else "native_close_exit"))
    return {"admin_parent_result_pin": payload["admin_parent_result_pin"],
        "prior_stage_parent_result_pin": payload["prior_stage_parent_result_pin"], "worker_request_pin": request_pin,
        "settlement_worker_result_pin": output_pin, "settlement_exit_witness_pin": exit_pin,
        "authority_context_pin": worker_request["authority_context_pin"], "stage": payload["stage"],
        "actual_child_exited_and_reaped": True, "actual_postcommit_stage_crash": crash_stage,
        "fresh_D5_D6_equal": payload["stage"] == "replay", "completion_authority": False}


def render(args, ab):
    unbound_pin = ab.pin(args.unbound_input, 2 * 1024**2)
    unbound = ab.decode(ab.read_pin(unbound_pin, 2 * 1024**2))
    original = ab.pin(ab.ORIGINAL_OWNER)
    owner = ab.owner_module(original)
    attempt, control = ab.allowed(args.attempt, owner, new=True), ab.allowed(args.control_scope, owner, new=True)
    fixture = Path(unbound["fixture_root"])
    need(not attempt.is_relative_to(fixture) and not control.is_relative_to(fixture)
         and not fixture.is_relative_to(attempt) and not fixture.is_relative_to(control)
         and not attempt.is_relative_to(control) and not control.is_relative_to(attempt)
         and len(os.fsencode(control / "admin.sock")) <= 107, "new native attempt/control/fixture scopes overlap or socket path too long")
    resource_prefix = owner.DATASETS / "ipfs_datasets_py/optimizers/logic_theorem_optimizer"
    payload = {"admin_exit_mode": args.admin_exit_mode} if args.operation == "admin" else {
        "stage": args.stage, "admin_parent_result_pin": ab.pin(args.admin_parent_result, 2 * 1024**2),
        "prior_stage_parent_result_pin": None if args.prior_stage_parent_result is None else
            ab.pin(args.prior_stage_parent_result, 2 * 1024**2)}
    if args.operation == "settle":
        need(args.stage in {"failed", *PRIOR}, "explicit native settlement stage required")
        need((args.stage == "failed") == (args.prior_stage_parent_result is None), "exact prior stage input differs")
    value = {"schema": SCHEMA, "fixture_only": True, "production_access": False, "operation": args.operation,
        "parent_source_pin": ab.pin(Path(__file__)), "original_owner_pin": original,
        "resource_source_pins": [ab.pin(resource_prefix / name) for name in ab.RESOURCE_FILES],
        "unbound_input_pin": unbound_pin, "repo_root": unbound["repo_root"], "controller_head": unbound["controller_head"],
        "fixture_root": str(fixture), "attempt_directory": str(attempt), "control_scope": str(control),
        "storage_bytes": 256 * 1024**2, "duration_seconds": 120, "payload": payload}
    result = ab.new_json(args.output, value)
    print(json.dumps({"request_pin": result, "metadata_only": True, "native_execution_performed": False}), flush=True)


def native(request_path, request_sha, ab):
    raw = ab.raw_file(request_path, 2 * 1024**2, private=True)
    need(hashlib.sha256(raw).hexdigest() == request_sha, "actual native parent request SHA differs")
    request = ab.decode(raw)
    need(type(request) is dict and set(request) == KEYS and request["schema"] == SCHEMA
         and request["fixture_only"] is True and request["production_access"] is False
         and request["operation"] in {"admin", "settle"}
         and type(request["duration_seconds"]) is int and request["duration_seconds"] == 120
         and type(request["storage_bytes"]) is int and request["storage_bytes"] == 256 * 1024**2,
         "closed actual native parent request differs")
    need(request["parent_source_pin"]["path"] == str(Path(__file__).resolve(strict=True)), "parent source origin differs")
    owner = ab.owner_module(request["original_owner_pin"])
    unbound = ab.decode(ab.read_pin(request["unbound_input_pin"], 2 * 1024**2))
    need(unbound.get("schema") == "authentic-native-unknown-admin-unbound-input@1"
         and unbound.get("fixture_only") is True and unbound.get("production_access") is False
         and unbound.get("authority_context_sealed") is False
         and all(request[key] == unbound[key] for key in ("repo_root", "controller_head", "fixture_root")),
         "original authentic unbound input differs")
    attempt, control = (ab.allowed(request[key], owner, new=True) for key in ("attempt_directory", "control_scope"))
    fixture = ab.allowed(request["fixture_root"], owner)
    need(fixture.name.startswith(ab.FIXTURE_PREFIX) and "legal-ir-grok-parallel-20261016-r06" not in str(fixture)
         and all(not left.is_relative_to(right) and not right.is_relative_to(left)
                 for left, right in ((attempt, control), (attempt, fixture), (control, fixture)))
         and len(os.fsencode(control / "admin.sock")) <= 107, "actual fixture/attempt/control scope differs")
    prefix = owner.DATASETS / "ipfs_datasets_py/optimizers/logic_theorem_optimizer"
    need(len(request["resource_source_pins"]) == 4 and {value["path"] for value in request["resource_source_pins"]}
         == {str(prefix / name) for name in ab.RESOURCE_FILES}, "actual legacy native resource closure differs")
    source_pins = [request["parent_source_pin"], request["original_owner_pin"], request["unbound_input_pin"],
        *request["resource_source_pins"], *unbound["worker_pins"].values(), unbound["producer_pin"], unbound["source_loader_pin"]]
    product_checker = None
    def source_guard():
        need(ab.raw_file(request_path, 2 * 1024**2, private=True) == raw, "native parent request changed")
        for value in source_pins:
            ab.read_pin(value, 8 * 1024**2)
        ab.read_pin(unbound["python_pin"], 128 * 1024**2)
        if product_checker is not None:
            product_checker()
    source_guard()
    producer = ab.module_bytes("_authentic_CD_pinned_producer_metadata", unbound["producer_pin"])
    loader = ab.module_bytes("_authentic_CD_pinned_loader_metadata", unbound["source_loader_pin"])
    need(producer.local_git_head(Path(request["repo_root"])) == request["controller_head"]
         and loader._package_pins(Path(request["repo_root"])) == unbound["product_sources"], "qualified C HEAD/source drifted")
    def product_checker():
        need(producer.local_git_head(Path(request["repo_root"])) == request["controller_head"],
             "trace-wide qualified C HEAD changed")
        loader._loaded_product_sources(unbound["product_sources"])
    attempt.mkdir(mode=0o700)
    ab.safe_environment(attempt)
    sys.path[:0] = [str(owner.DATASETS), request["repo_root"]]
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.existing_resource_scheduler import open_existing_scheduler
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceLane
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.parented_phase_resources import ParentedPhaseResourceReservation
    source_guard()
    cancel = threading.Event()
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, lambda _number, _frame: cancel.set())
    scheduler = open_existing_scheduler(owner.STATE, owner.POLICY)
    result_pin = None
    with scheduler.acquire(ResourceLane.ORCHESTRATION, cpu_slots=1, memory_mb=2048, child_process_slots=4,
            timeout=300, cancel_event=cancel, request_id="authentic-CD:" + attempt.name) as root:
        with root.acquire_child(lane=ResourceLane.ORCHESTRATION, cpu_slots=1, memory_mb=2048, child_process_slots=4,
                timeout=300, cancel_event=cancel, request_id="authentic-CD:" + attempt.name + ":phase") as phase:
            with ParentedPhaseResourceReservation(owner.LEDGER, roots=owner.ROOTS, attempt_directory=attempt,
                    parent_lease=phase, storage_bytes=request["storage_bytes"], memory_mb=2048, cpu_slots=1,
                    child_process_slots=4, scheduler_state_path=owner.STATE, expected_scheduler_policy_sha256=owner.POLICY,
                    timeout_seconds=420, ledger_lock_timeout_seconds=60) as disk:
                disk.account_external_bytes(hashlib.sha256(str(fixture).encode()).hexdigest(), 128 * 1024**2)
                disk.account_external_bytes(hashlib.sha256(str(control).encode()).hexdigest(), 8 * 1024**2)
                control.mkdir(mode=0o700)
                runtime = Runtime(ab, request, unbound, attempt, control, disk, source_guard, cancel, time.monotonic() + 120)
                runtime.guard(full=True)
                try:
                    result = admin(runtime, loader) if request["operation"] == "admin" else settle(runtime)
                finally:
                    if runtime.child is not None:
                        ab.close_native_custody(runtime.child, attempt, "actual-child", runtime)
                runtime.guard(full=True)
                result_pin = ab.new_json(attempt / "parent-result.json", {"schema": RESULT_SCHEMA,
                    "fixture_only": True, "production_access": False, "operation": request["operation"],
                    "stage": request["payload"].get("stage"), "parent_source_pin": request["parent_source_pin"],
                    "unbound_input_pin": request["unbound_input_pin"], "owner_process": runtime.parent,
                    "root_lease_id": root.lease_id, "phase_lease_id": phase.lease_id,
                    "disk_reservation_id": disk.reservation_id, "actual_parented_native_lease_id": disk.native_lease.lease_id,
                    "result": result, "native_finalization_pending_at_record_write": True,
                    "original_native_receipt_created": False, "completion_authority": False})
                disk.finalize(attempt, artifacts_durable=True)
    print(json.dumps({"fixture_only": True, "parent_result_pin": result_pin,
        "actual_lexical_native_owners_exited": True, "native_finalized": True, "completion_authority": False}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path)
    parser.add_argument("--request-sha256")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--unbound-input", type=Path)
    parser.add_argument("--operation", choices=("admin", "settle"))
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--control-scope", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--admin-exit-mode", choices=("graceful", "crash"), default="crash")
    parser.add_argument("--stage", choices=("failed", *PRIOR))
    parser.add_argument("--admin-parent-result", type=Path)
    parser.add_argument("--prior-stage-parent-result", type=Path)
    args = parser.parse_args()
    ab = load_ab()
    if args.render:
        need(all(getattr(args, key) is not None for key in ("unbound_input", "operation", "attempt", "control_scope", "output")),
             "explicit metadata render inputs required")
        render(args, ab)
        return 0
    need(args.request is not None and args.request_sha256 is not None, "exact native request and SHA required")
    return native(args.request, args.request_sha256, ab)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"fixture_failed": True, "error_type": type(error).__name__,
            "completion_authority": False}), file=sys.stderr, flush=True)
        raise SystemExit(2)
