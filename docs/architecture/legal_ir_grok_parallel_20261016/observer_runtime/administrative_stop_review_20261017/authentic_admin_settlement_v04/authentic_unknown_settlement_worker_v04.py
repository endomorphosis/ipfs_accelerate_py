"""Future parent-admitted authentic native settlement worker; never production.

No scheduler root is acquired here. An independently admitted retained F07
parent must register/reap this real isolated child before releasing custody.
All actual authority/store checks remain the unmodified product implementation.
Selected test losses occur only AFTER the real bound native method returns.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import stat
import sys
import time
from types import ModuleType

SCHEMA = "authentic-native-unknown-settlement-worker-request@2"
KEYS = {"schema", "fixture_only", "production_access", "unbound_input_pin", "worker_source_pin", "python_pin",
        "parent_kernel_identity", "control_scope", "control_scope_identity", "deadline_monotonic_ns", "nonce",
        "authority_context_pin", "audit_response_pin", "admin_exit_witness_pin", "admin_parent_result_pin", "stage"}
MAX_JSON = 2 * 1024**2
AB = Path(__file__).resolve(strict=True).parent.parent / "unknown-abandonment-authentic-fixture-v01"
AB_MANIFEST_SHA = "244016f11f23d14788da81456825dfb900dc2dc2cd1bb8461ab9ba2e7d28ad31"
AB_WRAPPER_SHA = "a5263c6ab4a0db17cec22a9d857326a40d5909983fc5647930d68696e03efae1"
HOOKS = {"failed": "_commit_portal_failure_phase", "coordination": "_fail_coordination_claim",
         "blocked": "_settle_portal_failure_control_task", "terminal_event": "_record_event"}


def need(value, reason):
    if not value:
        raise ValueError(reason)


def regular(path, maximum=MAX_JSON):
    path = Path(path)
    need(path.is_absolute() and path.resolve(strict=True) == path, "worker input path alias")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        need(stat.S_ISREG(before.st_mode) and before.st_uid in (0, os.geteuid()) and before.st_nlink == 1
             and not before.st_mode & 0o002 and 0 <= before.st_size <= maximum, "worker input custody/bound differs")
        chunks, size = [], 0
        while block := os.read(descriptor, min(65536, maximum - size + 1)):
            chunks.append(block)
            size += len(block)
            need(size <= maximum, "worker input exceeds bound")
        fields = ("st_dev", "st_ino", "st_size", "st_uid", "st_gid", "st_nlink", "st_mode", "st_mtime_ns", "st_ctime_ns")
        witness = lambda row: tuple(getattr(row, key) for key in fields)
        need(size == before.st_size and witness(before) == witness(os.fstat(descriptor)) == witness(path.lstat()),
             "worker input changed")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_ab():
    raw = regular(AB / "manifest-v01.json")
    need(hashlib.sha256(raw).hexdigest() == AB_MANIFEST_SHA, "frozen A/B manifest differs")
    manifest = json.loads(raw)
    value = next(item for item in manifest["files"] if Path(item["path"]).name == "authentic_fixture_owner_v01.py")
    raw = regular(Path(value["path"]), 8 * 1024**2)
    need(hashlib.sha256(raw).hexdigest() == value["sha256"] == AB_WRAPPER_SHA, "frozen A/B utility source differs")
    module = ModuleType("_authentic_settlement_AB_utilities")
    module.__file__ = value["path"]
    sys.modules[module.__name__] = module
    exec(compile(raw, value["path"], "exec", dont_inherit=True), module.__dict__)
    return module


def forbid_dispatch(event, args):
    if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.exec", "os.fork", "os.forkpty",
                 "socket.connect", "socket.getaddrinfo"}:
        raise PermissionError("authentic settlement worker forbids external dispatch")
    if event == "socket.__new__" and len(args) > 1 and args[1] != 1:
        raise PermissionError("authentic settlement worker forbids nonlocal sockets")


def run(request_path, request_sha):
    need(sys.flags.isolated == 1 and sys.flags.no_site == 1 and sys.dont_write_bytecode is True,
         "settlement worker requires exact -I -S -B")
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (192, 192))
    signal.setitimer(signal.ITIMER_REAL, 35)
    ab = load_ab()
    raw = regular(request_path)
    need(hashlib.sha256(raw).hexdigest() == request_sha, "settlement request SHA differs")
    request = ab.decode(raw)
    need(type(request) is dict and set(request) == KEYS and request["schema"] == SCHEMA
         and request["fixture_only"] is True and request["production_access"] is False
         and request["stage"] in {*HOOKS, "complete", "replay"}, "settlement request closed scope differs")
    control = Path(request["control_scope"])
    need(control.is_absolute() and control.resolve(strict=True) == control
         and request_path == control / "settlement-request.json", "actual settlement control scope differs")
    info = control.lstat()
    need(request["control_scope_identity"] == {"device": info.st_dev, "inode": info.st_ino,
        "uid": info.st_uid, "mode": stat.S_IMODE(info.st_mode)} and info.st_uid == os.geteuid()
         and stat.S_IMODE(info.st_mode) == 0o700, "settlement private control identity differs")
    need(type(request["deadline_monotonic_ns"]) is int
         and 0 < request["deadline_monotonic_ns"] - time.monotonic_ns() <= 30_000_000_000,
         "settlement finite child deadline differs")
    signal.setitimer(signal.ITIMER_REAL, (request["deadline_monotonic_ns"] - time.monotonic_ns()) / 1e9 + 5)
    parent = request["parent_kernel_identity"]
    need(type(parent) is dict and set(parent) == {"pid", "start_time_ticks", "boot_id", "process_birth_id"}
         and type(parent["pid"]) is int and parent["pid"] == os.getppid()
         and parent["start_time_ticks"] == ab.ticks(os.getppid()) and parent["boot_id"] == ab.boot()
         and os.getpgrp() == os.getpid(), "actual isolated child/native parent differs")
    need(request["worker_source_pin"]["path"] == str(Path(__file__).resolve(strict=True))
         and request["python_pin"]["path"] == str(Path(sys.executable).resolve(strict=True)),
         "settlement worker/Python origin differs")
    ab.read_pin(request["worker_source_pin"])
    ab.read_pin(request["python_pin"], 128 * 1024**2)
    unbound = ab.decode(ab.read_pin(request["unbound_input_pin"], MAX_JSON))
    need(unbound.get("schema") == "authentic-native-unknown-admin-unbound-input@1"
         and unbound.get("fixture_only") is True and unbound.get("production_access") is False
         and unbound.get("authority_context_sealed") is False
         and unbound["worker_pins"]["settlement"] == request["worker_source_pin"]
         and unbound["python_pin"] == request["python_pin"], "original unbound source input differs")
    sys.addaudithook(forbid_dispatch)
    producer = ab.module_bytes("_authentic_settlement_producer_loader", unbound["producer_pin"])
    need(producer.local_git_head(Path(unbound["repo_root"])) == unbound["controller_head"], "actual C HEAD drifted")
    loader, duckdb, installed = producer.prepare_imports(unbound, control)
    from ipfs_accelerate_py.agent_supervisor.task_sources import typed_state_owner as typed
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import planning_unknown_abandonment as policy
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import planning_unknown_abandonment_settlement as settlement
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon
    original_verify, original_load = policy.verify_custody, policy.load_context
    original_verify_code, original_load_code = original_verify.__code__, original_load.__code__
    need(typed.kernel_process_birth_id(os.getppid()) == parent["process_birth_id"], "native parent kernel birth differs")
    context = ab.decode(ab.read_pin(request["authority_context_pin"], MAX_JSON))
    response = ab.decode(ab.read_pin(request["audit_response_pin"], MAX_JSON))
    exited = ab.decode(ab.read_pin(request["admin_exit_witness_pin"], MAX_JSON))
    native_parent = ab.decode(ab.read_pin(request["admin_parent_result_pin"], MAX_JSON))
    need(native_parent.get("schema") == "authentic-unknown-admin-settlement-native-parent-result@2"
         and native_parent.get("operation") == "admin" and native_parent.get("fixture_only") is True
         and native_parent.get("unbound_input_pin") == request["unbound_input_pin"]
         and all(native_parent["result"][key] == request[key] for key in (
             "authority_context_pin", "audit_response_pin", "admin_exit_witness_pin")),
         "actual old C parent/result crosswalk differs")
    need(exited.get("schema") == "authentic-unknown-admin-child-exit-witness@2"
         and exited.get("fixture_only") is True and exited.get("child_reaped") is True
         and exited.get("owned_group_absent") is True and exited.get("native_owner_receipt_created") is False
         and exited["child_pid"] == context["registered_creator"]["pid"]
         and typed.kernel_process_birth_id(exited["child_pid"], start_time_ticks=exited["child_start_time_ticks"])
             == context["registered_creator"]["process_birth_id"], "actual ADMIN creator exit/birth crosswalk differs")
    policy.stopped_processes([{"pid": exited["child_pid"], "start_time_ticks": exited["child_start_time_ticks"],
        "boot_id": exited["boot_id"]}, {key: native_parent["owner_process"][key]
        for key in ("pid", "start_time_ticks", "boot_id")}])
    need(policy.canonical_json_bytes({key: value for key, value in context.items() if key not in {
        "registered_owner_identity", "registered_creator", "context_cid"}})
         == policy.canonical_json_bytes(unbound["context_preimage"]), "old sealed context preimage was replaced")
    snapshot = policy.load_context(context, check_closed_stores=False)

    def guard():
        need(time.monotonic_ns() < request["deadline_monotonic_ns"]
             and os.getppid() == parent["pid"] and ab.ticks(parent["pid"]) == parent["start_time_ticks"]
             and ab.boot() == parent["boot_id"], "actual finite native parent lost")
        need(policy.verify_custody is original_verify and policy.load_context is original_load
             and original_verify.__code__ is original_verify_code and original_load.__code__ is original_load_code
             and str(Path(original_verify.__code__.co_filename).resolve(strict=True)) == str(Path(policy.__file__).resolve(strict=True)),
             "actual production custody verifier was replaced")
        loader._loaded_product_sources(unbound["product_sources"])
        need(producer.local_git_head(Path(unbound["repo_root"])) == unbound["controller_head"], "C HEAD drifted")
        need(regular(request_path) == raw, "settlement request changed")
        for key in ("authority_context_pin", "audit_response_pin", "admin_exit_witness_pin", "admin_parent_result_pin",
                    "worker_source_pin", "unbound_input_pin"):
            ab.read_pin(request[key], MAX_JSON if key != "worker_source_pin" else 8 * 1024**2)

    stores = context["stores"]
    lane = Path(stores["execution"]).parent
    daemon = DatabaseImplementationDaemon(database_path=Path(stores["canonical"]),
        coordination_path=Path(stores["coordination"]), execution_path=Path(stores["execution"]),
        state_dir=lane, state_prefix="crash-copy", task_shard_count=3, task_shard_index=1, strict_task_sharding=True,
        authority_mode="embedded_exclusive", task_source_kind="duckdb", require_real_execution=True, lease_ms=3_600_000)
    daemon.open()

    def inspect():
        guard()
        actual = daemon.get_attempt(snapshot["attempt"]["attempt_id"])
        need(actual is not None and settlement.native_stores(daemon) == stores, "actual native attempt/stores disappeared")
        audit = policy.exact_audit(daemon.task_source, attempt=actual, binding=daemon._control_binding_for_attempt(actual))
        need(audit["context"] == context and audit["evidence_id"] == response["audit_receipt"]["subject_id"]
             and audit["event"]["event_id"] == response["audit_receipt"]["event_id"], "old actual audit differs")
        need(settlement.provider_rows(daemon, actual.attempt_id) == snapshot["providers"], "raw current UNKNOWN changed")
        need(daemon._require_connection().execute("SELECT COUNT(*) FROM effect_claims WHERE attempt_id=?",
            [actual.attempt_id]).fetchone()[0] == 0, "actual effect appeared")
        for history in snapshot["historical_unknowns"]:
            for value in history["store_pins"]:
                ab.read_pin(value, 64 * 1024**2)
        count = daemon._require_connection().execute("SELECT COUNT(*) FROM daemon_execution_events WHERE attempt_id=? AND event_type=?",
            [actual.attempt_id, "terminal_portal_failure_settled"]).fetchone()[0]
        task = daemon.task_source.get_task(actual.task_cid)
        claim, lease = daemon.coordinator.get_task_claim(actual.claim_id), daemon.coordinator.get_lease(actual.lease_id)
        coordinated = daemon.coordinator.get_task_attempt(actual.attempt_id)
        state = lambda value: str(getattr(value, "value", value))
        return {"attempt": policy.plain(actual.to_dict()), "control_status": task.status, "control_revision": task.revision,
            "claim_state": state(claim.state), "lease_state": state(lease.state),
            "coordination_attempt_status": state(coordinated.status), "terminal_count": count,
            "audit_evidence_id": audit["evidence_id"], "audit_event_id": audit["event"]["event_id"]}

    before = inspect()
    need(daemon.reconcile_terminal_portal_failures() == [] and inspect() == before,
         "ordinary reconciliation changed administrative stages")
    expected_start = {"failed": ("running", "in_progress", "accepted", 0),
        "coordination": ("failed", "in_progress", "accepted", 0), "blocked": ("failed", "in_progress", "released", 0),
        "terminal_event": ("failed", "blocked", "released", 0),
        "complete": ("failed", "blocked", "released", 1), "replay": ("failed", "blocked", "released", 1)}
    need((before["attempt"]["status"], before["control_status"], before["claim_state"], before["terminal_count"])
         == expected_start[request["stage"]], "actual native stage differs from explicit serial request")
    if request["stage"] in HOOKS:
        original = getattr(daemon, HOOKS[request["stage"]])
        def after_real_commit(*args, **kwargs):
            value = original(*args, **kwargs)
            if request["stage"] == "terminal_event" and args[0] != "terminal_portal_failure_settled":
                return value
            observed = inspect()
            expected_after = {"failed": ("failed", "in_progress", "accepted", 0),
                "coordination": ("failed", "in_progress", "released", 0), "blocked": ("failed", "blocked", "released", 0),
                "terminal_event": ("failed", "blocked", "released", 1)}[request["stage"]]
            need((observed["attempt"]["status"], observed["control_status"], observed["claim_state"], observed["terminal_count"])
                 == expected_after, "genuine method did not reach exact committed stage")
            ab.new_json(control / "lost-stage.json", {"schema": "authentic-unknown-native-committed-stage-loss@2",
                "fixture_only": True, "stage": request["stage"], "request_sha256": request_sha, "nonce": request["nonce"],
                "child_pid": os.getpid(), "child_start_time_ticks": ab.ticks(os.getpid()), "boot_id": ab.boot(),
                "process_group": os.getpgrp(), "parent_kernel_identity": parent, "native_observation": observed,
                "genuine_bound_method_returned": True, "native_handle_close_claimed": False,
                "authority_context_cid": context["context_cid"], "completion_authority": False})
            # Parent owns the real isolated group/resources and must reap us.
            os._exit(73)
        setattr(daemon, HOOKS[request["stage"]], after_real_commit)
    result = daemon.settle_planning_unknown_administrative_abandonment(authority_context=context)
    need(request["stage"] in {"complete", "replay"}, "selected committed-stage hook did not fire")
    observed = inspect()
    need(observed["attempt"]["status"] == "failed" and observed["control_status"] == "blocked"
         and observed["control_revision"] == context["expected_task_revision"] + 1
         and observed["claim_state"] == observed["lease_state"] == "released"
         and observed["coordination_attempt_status"] == "failed" and observed["terminal_count"] == 1,
         "actual final native stages differ")
    need(result["provider_outcome"] == "unknown" and result["possible_external_effects"] == "unverified"
         and result["raw_unknown_rows_preserved"] is True and all(result[key] is False for key in (
             "callback_closed", "remote_cancellation_proved", "nonreachability_asserted", "automatic_retry_admitted",
             "operator_rearm_performed", "successor_attempt_admitted", "completion_authority")), "truthful settlement limits differ")
    need(daemon.settle_planning_unknown_administrative_abandonment(authority_context=context) == result,
         "same-owner exact terminal replay differs")
    need(daemon.reconcile_terminal_portal_failures() == [] and inspect() == observed
         and context["task_cid"] in daemon._automatic_claim_exclusions(), "ordinary reconcile/retry exclusion differs")
    with daemon.task_source.intent._connection(write=False) as connection:
        need(connection.execute("SELECT COUNT(*) FROM evidence_nodes WHERE evidence_kind=?", [policy.EVIDENCE_KIND]).fetchone()[0] == 1
             and connection.execute("SELECT COUNT(*) FROM domain_events WHERE event_id=?", [observed["audit_event_id"]]).fetchone()[0] == 1,
             "actual native audit/evidence is duplicated")
    guard()
    daemon.close()  # Failure emits no successful close witness; parent reaps actual kernel exit.
    ab.new_json(control / "settlement-result.json", {"schema": "authentic-unknown-native-settlement-worker-result@2",
        "fixture_only": True, "stage": request["stage"], "request_sha256": request_sha, "nonce": request["nonce"],
        "child_pid": os.getpid(), "child_start_time_ticks": ab.ticks(os.getpid()), "boot_id": ab.boot(),
        "process_group": os.getpgrp(), "parent_kernel_identity": parent, "authority_context_cid": context["context_cid"],
        "actual_native_close_returned": True, "production_custody_verifier_replaced": False,
        "native_observation": observed, "result": policy.plain(result), "duckdb_installation": installed,
        "completion_authority": False})
    os._exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--request-sha256", required=True)
    args = parser.parse_args()
    try:
        run(args.request, args.request_sha256)
    except BaseException as error:
        sys.stderr.write(json.dumps({"fixture_failed": True, "error_type": type(error).__name__,
            "native_handle_close_claimed": False, "completion_authority": False}) + "\n")
        sys.stderr.flush()
        os._exit(2)


if __name__ == "__main__":
    main()
