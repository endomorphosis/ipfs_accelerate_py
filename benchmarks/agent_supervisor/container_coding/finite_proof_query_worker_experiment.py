"""Qualify one isolated worker publication with exact signed proof-query context.

The complete administrator population stays in the owner database. Public
checks complete the prerequisite; the native worker edits the residual task.
This authored, model-off fixture has a five-input observation domain.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import threading
import time

from .finite_repository_admission_experiment import (
    _git, _pin, _sources, _write,
)
from .terminal_codebase_finite_experiment import INTENT
from .terminal_codebase_finite_service_experiment import (
    _authority_materials, _catalog, _open, _request, _scheduler,
)

SCHEMA = "finite-proof-query-native-worker-qualification@1"
MODULE = "benchmarks.agent_supervisor.container_coding.finite_proof_query_worker_experiment"
SOURCES = (MODULE,
    "ipfs_accelerate_py.agent_supervisor.runtime.finite_proof_query_admission",
    "ipfs_accelerate_py.agent_supervisor.runtime.finite_proof_query_execution",
    "ipfs_accelerate_py.agent_supervisor.runtime.finite_proof_query_worker_context",
    "ipfs_accelerate_py.agent_supervisor.runtime.finite_proof_query_worker_dispatch",
    "ipfs_accelerate_py.agent_supervisor.runtime.finite_proof_query_worker_source_custody",
    "ipfs_accelerate_py.agent_supervisor.runtime.finite_proof_query_worktree_allocation",
    "ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle",
    "ipfs_accelerate_py.agent_supervisor.merge.database_coordination",
    "ipfs_accelerate_py.agent_supervisor.planning.finite_proof_query_join",
    "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon",
    "ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime",
    "benchmarks.agent_supervisor.container_coding.container_worker_deployment",
    "ipfs_datasets_py.duckdb_control.codebase_verification_catalog",
    "ipfs_datasets_py.duckdb_control.codebase_verification_queries",
    "ipfs_datasets_py.logic.software_contracts.codebase_verification",
    "ipfs_datasets_py.logic.software_contracts.codebase_applicability",
    "benchmarks.agent_supervisor.container_coding.finite_repository_admission_experiment",
    "ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_admission",
    "ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_execution",
    "ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_candidate_runner",
    "ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime",
    "ipfs_accelerate_py.agent_supervisor.runtime.candidate_execution",
    "ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge",
    "ipfs_accelerate_py.agent_supervisor.runtime.local_planning_admission",
    "ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner",
    "ipfs_accelerate_py.agent_supervisor.task_sources.database_task_source",
    "ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane",
    "ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge",
    "ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository")


from .finite_repository_worker_experiment import _native_graph



POST_BIRTH_CONTROLS = (
    "isolated_after_real_birth_callback_error",
    "isolated_real_birth_application_ACK_drop",
)
_NATIVE_ATTEMPT_FIELDS = ("task_cid", "claim_id", "attempt_id", "attempt_number",
    "lease_id", "owner_session_id", "fencing_token", "fence_epoch")


def _post_birth_kernel_identity(pid):
    """Independent bounded /proc read; no coordinator, peer or process census."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("literal kernel witness PID required")
    root = Path("/proc") / str(pid)
    def read(name, limit=65536):
        with (root / name).open("rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("kernel witness file exceeded bound")
        return raw
    raw_stat, raw_status, command = read("stat"), read("status"), read("cmdline")
    fields = raw_stat.decode().rsplit(")", 1)[1].strip().split()
    parent, start = int(fields[1]), int(fields[19])
    uids = next(line.split()[1:] for line in raw_status.decode().splitlines()
        if line.startswith("Uid:"))
    if len(uids) != 4 or not command:
        raise ValueError("kernel witness UID or live command absent")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if read("stat").decode().rsplit(")", 1)[1].strip().split()[19] != str(start):
        raise ValueError("kernel witness PID changed while reading")
    birth = "birth:" + hashlib.sha256(f"{pid}:{start}:{boot}:{parent}".encode()).hexdigest()[:32]
    return {"pid": pid, "uid": int(uids[1]), "uids": [int(uid) for uid in uids],
        "parent_pid": parent, "start_time_ticks": start, "boot_id": boot,
        "process_birth_id": birth, "process_state": fields[0],
        "command": [part.decode() for part in command.split(b"\x00") if part],
        "raw_stat": raw_stat.decode(), "raw_status": raw_status.decode(),
        "raw_cmdline_hex": command.hex(), "source": "independent_direct_proc_read"}


def _post_birth_gone(identity):
    """PID reuse/reparenting cannot turn a remaining original birth into gone."""
    try:
        actual = _post_birth_kernel_identity(identity["pid"])
    except FileNotFoundError:
        return {"original": identity, "original_birth_absent": True,
            "observation": "proc_PID_absent"}
    except (OSError, ValueError, StopIteration, IndexError, UnicodeError) as error:
        return {"original": identity, "original_birth_absent": None,
            "observation": "unknown", "error_type": type(error).__name__}
    same = actual["start_time_ticks"] == identity["start_time_ticks"] and actual["boot_id"] == identity["boot_id"]
    return {"original": identity, "current": actual, "original_birth_absent": not same,
        "observation": "original_PID_start_boot_still_present" if same else "PID_reused",
        "zombie_alone_is_not_birth_disappearance": True}


def _post_birth_native_snapshot(path, *, output, stage):
    """Retain every actual native schema/table/index first, then report errors."""
    import duckdb
    path, output = Path(path), Path(output)
    record = {"schema": "finite-proof-query-post-birth-native-database@1", "stage": stage,
        "database_path": str(path), "read_only": True, "tables": {}, "catalogs": {},
        "errors": [], "duckdb_version": duckdb.__version__, "authority_granted": False}
    paths = [path, Path(str(path) + ".wal")]
    before = {str(item): _pin(item) if item.is_file() and not item.is_symlink() else None for item in paths}
    record["file_pins_before"] = before
    connection = None
    try:
        if path.is_symlink() or not path.is_file():
            raise ValueError("exact existing native file required")
        connection = duckdb.connect(str(path), read_only=True, config={"threads": 1,
            "memory_limit": "256MB", "enable_external_access": False,
            "allow_unsigned_extensions": False, "autoinstall_known_extensions": False,
            "autoload_known_extensions": False})
        database = connection.execute("SELECT current_database()").fetchone()[0]
        connection.execute("BEGIN TRANSACTION")
        def rows(sql, parameters=()):
            cursor = connection.execute(sql, parameters)
            columns = [item[0] for item in cursor.description]
            values = cursor.fetchmany(2049)
            if len(values) > 2048:
                raise ValueError("native table/catalog exceeded finite fixture population")
            return {"columns": columns, "row_count": len(values),
                "rows": [dict(zip(columns, row)) for row in values]}
        for kind in ("schemas", "tables", "views", "indexes", "columns", "constraints"):
            record["catalogs"][kind] = rows("SELECT * FROM duckdb_" + kind + "() WHERE database_name = ?", [database])
        selected = record["catalogs"]["tables"]["rows"]
        if len(selected) > 64:
            raise ValueError("native authority table count exceeded fixture bound")
        quote = lambda value: '"' + value.replace('"', '""') + '"'
        for table in selected:
            schema, name = table["schema_name"], table["table_name"]
            key = schema + "." + name
            record["tables"][key] = {"schema_name": schema, "table_name": name,
                **rows("SELECT * FROM " + quote(schema) + "." + quote(name))}
        connection.rollback()
    except Exception as error:
        record["errors"].append({"type": type(error).__name__,
            "message_sha256": hashlib.sha256(str(error).encode()).hexdigest()})
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception as error:
                record["errors"].append({"type": type(error).__name__,
                    "message_sha256": hashlib.sha256(str(error).encode()).hexdigest()})
        record["file_pins_after"] = {str(item): _pin(item) if item.is_file() and not item.is_symlink() else None for item in paths}
        record["files_unchanged"] = record["file_pins_after"] == before
        record["status"] = "retained" if not record["errors"] and record["files_unchanged"] else "unknown_unqualified"
        # The native DuckDB files also remain in the fixture output. No rows or
        # committed effects are edited to match a smaller expected settlement.
        _write(output, record)
        output.chmod(0o600)
    return {"path": str(output), **_pin(output), "status": record["status"]}, record


def _post_birth_worktree_effects(*, allocations, output, stage):
    """Keep exact candidate files/diffs; do not remove a dirty allocation."""
    output = Path(output)
    output.mkdir(mode=0o700)
    record = {"schema": "finite-proof-query-post-birth-worktree-effects@1", "stage": stage,
        "allocations": [], "errors": [], "atomic_snapshot_claimed": False,
        "file_owner_is_not_write_origin_attestation": True}
    try:
        for allocation in allocations.values():
            path = Path(allocation["workspace_path"])
            item = {"allocation": allocation, "path": str(path), "exists": path.exists(), "files": []}
            record["allocations"].append(item)
            if not path.exists():
                continue
            item["head"] = _git(path, "rev-parse", "HEAD")
            for name, arguments in (("status", ["status", "--porcelain=v1", "-z", "--untracked-files=all"]),
                                    ("diff", ["diff", "--binary", "HEAD"])):
                raw = subprocess.check_output(["git", "-C", str(path), *arguments], timeout=10)
                if len(raw) > 1024 * 1024:
                    raise ValueError("candidate diff/status exceeds fixture bound")
                target = output / (str(len(record["allocations"])) + "-" + name + ".bin")
                target.write_bytes(raw); target.chmod(0o400)
                item[name] = _pin(target)
            count = 0
            for base, directories, names in os.walk(path, followlinks=False):
                for name in [*directories, *names]:
                    current = Path(base) / name
                    count += 1
                    if count > 128 or current.is_symlink():
                        raise ValueError("candidate contains unbounded entries or a symlink")
                    if not current.is_file():
                        continue
                    metadata = current.stat()
                    if metadata.st_size > 1024 * 1024 or metadata.st_nlink != 1:
                        raise ValueError("candidate file exceeds retention bound")
                    raw = current.read_bytes()
                    target = output / (str(len(record["allocations"])) + "-file-" + str(count) + ".bin")
                    target.write_bytes(raw); target.chmod(0o400)
                    after = current.stat()
                    item["files"].append({"relative": current.relative_to(path).as_posix(),
                        "source": str(current), "uid": metadata.st_uid, "gid": metadata.st_gid,
                        "mode": stat.S_IMODE(metadata.st_mode), "retained": _pin(target),
                        "unchanged_while_reading": (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
                            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)})
    except Exception as error:
        record["errors"].append({"type": type(error).__name__, "message_sha256": hashlib.sha256(str(error).encode()).hexdigest()})
    _write(output / "manifest.json", record)
    return {"path": str(output / "manifest.json"), **_pin(output / "manifest.json")}, record


class _PostBirthFixtureError(RuntimeError):
    """Deliberate fixture fault after a real acknowledged/received child birth."""


class _PostBirthControl:
    def __init__(self, *, broker, case, output):
        from ipfs_accelerate_py.agent_supervisor.runtime import finite_proof_query_worker_dispatch as dispatch
        self.broker, self.case, self.output, self.dispatch = broker, case, Path(output), dispatch
        self.output.mkdir(mode=0o700)
        self.original_fence = type(broker)._execute_with_attempt_fence
        self.original_receive = dispatch._receive
        self.prior_instance_fence = broker.__dict__.get("_execute_with_attempt_fence")
        self.record = {"schema": "finite-proof-query-post-birth-control@1", "case": case,
            "status": "armed_unqualified", "events": [], "injected": False,
            "actual_guard_supplied_lease": None, "raw_ACK": None,
            "effect_settlement": "UNKNOWN", "global_training_or_model_counts": None,
            "proof_authority": False, "completion_authority": False}
        self.before = None
        if case == POST_BIRTH_CONTROLS[0]:
            broker._execute_with_attempt_fence = self.fenced
        else:
            dispatch._receive = self.receive
        self.event("hook_installed")

    def event(self, name, **fields):
        self.record["events"].append({"sequence": len(self.record["events"]) + 1,
            "event": name, "monotonic_ns": time.monotonic_ns(), "wall_clock_ms": time.time_ns() // 1_000_000,
            **fields})
        _write(self.output / "post-birth-control.json", self.record)
        (self.output / "post-birth-control.json").chmod(0o600)

    def before_guard(self):
        broker = self.broker
        self.record["native_identity"] = {name: broker._wrapper["claim"][name] for name in _NATIVE_ATTEMPT_FIELDS}
        receipt, self.before = _post_birth_native_snapshot(broker._attempt_authority_path,
            output=self.output / "before-isolated-guard-coordination.json", stage="before-original-isolated-guard")
        self.record["before_guard_coordination"] = receipt
        self.event("native_cursor_retained_before_original_guard")
        if receipt["status"] != "retained":
            raise ValueError("post-birth control lacks a genuine stable pre-guard native cursor")

    def birth(self, fields):
        worker_pid = fields.get("worker_pid")
        sudo_pid = fields.get("sudo_pid", fields.get("pid"))
        worker = _post_birth_kernel_identity(worker_pid)
        sudo = _post_birth_kernel_identity(sudo_pid)
        wrapper = _post_birth_kernel_identity(self.broker._wrapper["pid"])
        daemon = _post_birth_kernel_identity(self.broker._wrapper["daemon_peer"][0])
        chain, current = [], worker
        for _ in range(8):
            chain.append(current)
            if current["pid"] == sudo_pid:
                break
            current = _post_birth_kernel_identity(current["parent_pid"])
        witness = {"worker": worker, "sudo": sudo, "wrapper": wrapper, "daemon": daemon,
            "parent_chain": chain, "owner_witness_pid": os.getpid(), "owner_witness_uid": os.geteuid(),
            "observed_at_ms": time.time_ns() // 1_000_000}
        self.record["kernel_birth_witness"] = witness
        self.event("actual_kernel_birth_witness_retained")
        expected_worker = fields.get("worker_process_birth_id")
        expected_sudo = fields.get("process_birth_id")
        if expected_sudo is None:
            native_sudo = [row["process_birth_id"] for row in fields.get("parent_chain", []) if row["pid"] == sudo_pid]
            expected_sudo = native_sudo[0] if len(native_sudo) == 1 else None
        if (worker["uid"] != 1001 or worker["process_state"] == "Z"
                or worker["process_birth_id"] != expected_worker
                or sudo["process_birth_id"] != expected_sudo
                or chain[-1]["pid"] != sudo_pid or sudo["parent_pid"] != wrapper["pid"]
                or wrapper["process_birth_id"] != self.broker._wrapper["process_birth_id"]
                or wrapper["parent_pid"] != daemon["pid"]
                or daemon["start_time_ticks"] != self.broker._wrapper["daemon_peer"][2]
                or daemon["process_birth_id"] != self.broker._wrapper["claim"]["process_birth_id"]):
            raise ValueError("real isolated ACK cannot be independently joined to live kernel births")
        self.record["kernel_witness_validated"] = True
        marker = self.output / "external-effect.marker"
        raw = json.dumps({"case": self.case, "writer_pid": os.getpid(), "writer_uid": os.geteuid(),
            "worker_pid": worker_pid, "worker_birth": worker["process_birth_id"]}, sort_keys=True).encode() + b"\n"
        with marker.open("xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        marker.chmod(0o600)
        self.record["external_marker"] = {**_pin(marker), "writer_uid": os.geteuid(),
            "origin": "owner_fixture_callback_effect_not_worker_write", "content_hex": raw.hex()}
        self.event("owner_external_marker_committed")

    def fenced(self, *, request, daemon_peer, callback):
        if self.broker._boundary != "isolated_worker" or self.record["injected"]:
            return self.original_fence(self.broker, request=request, daemon_peer=daemon_peer, callback=callback)
        self.before_guard()
        def controlled(claim, lease):
            result = callback(claim, lease)
            self.record["original_callback_result"] = result
            self.record["ACK_retention_scope"] = "original decoded validated callback result; raw wire ACK not observed by callback-only hook"
            self.record["actual_guard_supplied_lease"] = lease.to_dict()
            self.record["actual_guard_supplied_lease_scope"] = "genuine immutable argument inside original native transaction"
            self.record["native_identity"] = {name: claim[name] for name in _NATIVE_ATTEMPT_FIELDS}
            self.record["lease_tuple_matches_claim"] = all(getattr(lease, name) == claim[name] for name in _NATIVE_ATTEMPT_FIELDS)
            self.event("original_callback_returned_after_real_kernel_ACK")
            status = result[0]
            if status.get("status") != "spawned":
                raise ValueError("post-birth control did not receive actual original spawned ACK")
            self.birth(status)
            self.record["injected"] = True
            self.record["status"] = "deliberate_callback_error_after_real_birth"
            self.event("deliberate_exception_before_original_guard_postcheck", error_type="_PostBirthFixtureError")
            raise _PostBirthFixtureError("fixture callback error after genuine isolated child ACK")
        try:
            return self.original_fence(self.broker, request=request, daemon_peer=daemon_peer, callback=controlled)
        except BaseException as error:
            self.event("original_guard_raised", error_type=type(error).__name__)
            raise

    def receive(self, channel):
        broker = self.broker
        if threading.get_ident() != broker._thread.ident:
            return self.original_receive(channel)
        target = (not self.record["injected"] and broker._boundary == "isolated_worker"
            and broker._stage == "actual_isolated_birth_ack")
        if not target:
            value = self.original_receive(channel)
            if value.get("operation") == "prepare_isolated" and self.before is None:
                self.before_guard()
            return value
        chunks = []
        class Tee:
            def recv(self, *args, **kwargs):
                raw = channel.recv(*args, **kwargs)
                chunks.append(raw)
                return raw
            def __getattr__(self, name):
                return getattr(channel, name)
        value = self.original_receive(Tee())
        raw = b"".join(chunks)
        path = self.output / "raw-isolated-spawned-ACK.frame"
        path.write_bytes(raw); path.chmod(0o600)
        self.record["raw_ACK"] = {**_pin(path), "parsed": value,
            "includes_exact_four_byte_length_prefix": True,
            "capture": "original_receive_parser_actual_socket_recv_tee",
            "application_delivery_to_original_broker_callback": "withheld_after_actual_receive"}
        self.record["actual_guard_supplied_lease_scope"] = "not observed by receive-only hook; authentic before/closed SQL retained separately"
        self.event("original_receive_obtained_real_isolated_ACK")
        if (set(value) != {"schema", "operation", "handoff_id", "pid", "process_birth_id", "worker_pid", "worker_process_birth_id"}
                or value.get("schema") != self.dispatch.SCHEMA or value.get("operation") != "spawned"):
            raise ValueError("ACK-drop control did not receive the actual closed spawned frame")
        self.birth(value)
        self.record["injected"] = True
        self.record["status"] = "deliberate_application_drop_after_real_received_birth"
        self.event("deliberate_received_frame_application_drop", error_type="_PostBirthFixtureError")
        raise _PostBirthFixtureError("fixture application drop after genuine isolated spawned frame receipt")

    def restore(self):
        if self.case == POST_BIRTH_CONTROLS[0]:
            if self.prior_instance_fence is None:
                self.broker.__dict__.pop("_execute_with_attempt_fence", None)
            else:
                self.broker._execute_with_attempt_fence = self.prior_instance_fence
        else:
            self.dispatch._receive = self.original_receive
        self.event("fixture_hook_restored_after_native_close")


def _validate_post_birth_control(*, control, report, closed):
    """Validate observed control outcome only after all complete rows are retained."""
    record = control.record
    def table(snapshot, name):
        matches = [value["rows"] for value in snapshot.get("tables", {}).values()
            if value["table_name"] == name]
        if len(matches) != 1:
            raise ValueError("post-birth native table population is unavailable or ambiguous")
        return matches[0]
    coordinator, execution = closed["coordination"], closed["execution"]
    identity = record.get("native_identity", {})
    leases = table(coordinator, "fenced_leases")
    attempts = table(execution, "database_task_attempts")
    providers = table(execution, "provider_invocations")
    effects = table(execution, "effect_claims")
    completions = [row for row in table(coordinator, "task_completions") if row["task_cid"] == identity.get("task_cid")]
    task = report["task_after_stop"]
    portal_receipt = task.get("body", {}).get("completion_receipt")
    selected_attempts = [row for row in attempts if row["task_cid"] == identity.get("task_cid")]
    record["actual_native_settlement"] = {"task_status": task["status"], "task_revision": task["revision"],
        "execution_attempt_rows": selected_attempts, "provider_rows": providers, "effect_rows": effects,
        "selected_coordination_task_completion_rows": completions,
        "actual_control_Portal_body_completion_receipt": portal_receipt,
        "actual_control_Portal_body_completion_receipt_operation": (
            portal_receipt.get("operation") if type(portal_receipt) is dict else None),
        "native_failed_task_observed": task["status"] == "failed",
        "logical_completion_observed": task["status"] == "completed" or bool(completions)
            or (type(portal_receipt) is dict and portal_receipt.get("operation") == "database_complete")
            or any(row["status"] in ("succeeded", "completed") for row in selected_attempts),
        "external_effect_settlement": "UNKNOWN",
        "owner_marker_is_known_external_effect": "external_marker" in record,
        "worker_effect_freedom_proved": False}
    control.event("actual_native_settlement_retained_without_rewriting")
    expected_observed = control.case == POST_BIRTH_CONTROLS[0]
    observations = report["native_dispatch_observations"]
    wrappers = [row for row in observations if row.get("status") == "spawned"
        and row.get("boundary") == "native_daemon_to_owner_wrapper"]
    refusals = [row for row in observations if row.get("status") == "refused"
        and row.get("boundary") == "isolated_worker" and row.get("stage") == "actual_isolated_birth_ack"]
    disappeared = record.get("birth_disappearance_after_STOP", [])
    cleanup = report["stop"].get("data", {}).get("isolated_worker_cleanup", {})
    # Keep each predicate and the actual rows before reporting any failure.
    checks = {"actual_control_injected": record["injected"],
        "closed_authorities_retained": all(snapshot["status"] == "retained" for snapshot in closed.values()),
        "pre_guard_native_cursor_retained": control.before is not None and control.before["status"] == "retained",
        "first_actual_wrapper_birth": len(wrappers) == 1,
        "no_second_success_release": not any(row.get("status") == "spawned" and row.get("boundary") == "isolated_worker" for row in observations),
        "actual_refusal_birth_diagnostics": any(row.get("worker_birth_observed") is expected_observed
            and row.get("unacknowledged_child_birth_possible") is True for row in refusals),
        "full_native_lease_tuple_join": len(identity) == 8 and len([row for row in leases
            if all(type(row.get(name)) is type(value) and row.get(name) == value for name, value in identity.items())]) == 1,
        "native_STOP_and_isolated_cleanup": report["stop"]["status"] == "succeeded"
            and report["stop"].get("data", {}).get("old_tree_fenced") is True
            and cleanup.get("returncode") == 0 and cleanup.get("single_worker") is True,
        "actual_witnessed_worker_and_sudo_births_gone": len(disappeared) == 2
            and all(row.get("original_birth_absent") is True for row in disappeared),
        "original_socket_removed": report["broker_closed"] and report["broker_socket_removed"],
        "actual_native_process_tree_empty": report["remaining_processes"] == 0,
        "noncompletion_observed": not record["actual_native_settlement"]["logical_completion_observed"],
        "selected_coordinator_completion_absent": not completions,
        # An admitted claim legitimately occupies the same body receipt field.
        # This literal operation check is not validity of an unknown receipt;
        # the independent cold readback joins the full canonical Portal rows.
        "actual_Portal_database_complete_receipt_absent": portal_receipt is None
            or (type(portal_receipt) is dict and type(portal_receipt.get("operation")) is str
                and portal_receipt["operation"] != "database_complete"),
        "owner_marker_retained": "external_marker" in record
            and _pin(record["external_marker"]["path"])["sha256"] == record["external_marker"]["sha256"],
        "before_after_worktree_effects_retained": not record["worktree_effects_before_STOP_errors"]
            and not record["worktree_effects_after_STOP_errors"],
    }
    if expected_observed:
        checks["actual_callback_lease_observed"] = record.get("lease_tuple_matches_claim") is True
        checks["original_callback_then_deliberate_guard_error"] = any(event["event"] == "original_guard_raised"
            and event["error_type"] == "_PostBirthFixtureError" for event in record["events"])
    else:
        checks["receive_only_no_synthetic_lease"] = record["actual_guard_supplied_lease"] is None
        frame = Path(record["raw_ACK"]["path"]).read_bytes() if record.get("raw_ACK") else b""
        checks["actual_original_wire_frame_matches_parsed_fields"] = (len(frame) >= 6
            and int.from_bytes(frame[:4], "big") == len(frame) - 4
            and json.loads(frame[4:]) == record["raw_ACK"]["parsed"])
    record["qualification_checks"] = checks
    record["status"] = "qualified_control_outcome" if all(checks.values()) else "unqualified_control_outcome"
    control.event("control_outcome_validation_finished")
    if not all(checks.values()):
        raise ValueError("post-birth fixture refused qualification; complete native rows and actual effects retained")
    return record



def _bounded_dispatch_refusals(observations):
    """Retain only the owner's bounded diagnostics, never private payloads."""
    result = []
    for row in observations:
        if type(row) is not dict or row.get("status") != "refused":
            continue
        item = {"schema": "finite-proof-query-native-refusal-diagnostic@1",
            "source": "actual_owner_broker_observation", "settlement_authority": False}
        for name in ("boundary", "stage", "error_type"):
            value = row.get(name)
            item[name] = value if type(value) is str and len(value) <= 128 else None
        value = row.get("message_sha256")
        item["message_sha256"] = (value if type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value) else None)
        for name in ("worker_birth_observed", "unacknowledged_child_birth_possible"):
            value = row.get(name)
            item[name] = value if type(value) is bool else None
        result.append(item)
        if len(result) == 16:
            break
    return result


def _retain_unknown_callback_ledger(*, execution_path, task, selected, output):
    """Read the closed native cursor; an unknown callback grants no settlement."""
    import duckdb
    from ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner import (
        TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA, _validated_database_claim_identity,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        DATABASE_CONTROL_CLAIM_BINDING_SCHEMA, DATABASE_PORTAL_BINDING_BASIS_SCHEMA,
        canonical_json, content_identity,
    )

    execution_path = Path(execution_path)
    if execution_path.is_symlink() or not execution_path.is_file():
        raise ValueError("native callback ledger is not an exact closed file")
    paths = [execution_path]
    wal = Path(str(execution_path) + ".wal")
    if wal.exists():
        if wal.is_symlink() or not wal.is_file():
            raise ValueError("native callback WAL is not an exact closed file")
        paths.append(wal)
    before = [_pin(path) for path in paths]
    connection = duckdb.connect(str(execution_path), read_only=True,
        config={"enable_external_access": False, "allow_unsigned_extensions": False})
    tables = {}
    try:
        for name in ("database_task_attempts", "attempt_phases", "provider_invocations",
                     "effect_claims", "daemon_execution_events"):
            count = connection.execute("SELECT count(*) FROM " + name).fetchone()[0]
            if type(count) is not int or not 0 <= count <= 128:
                raise ValueError("native callback ledger population exceeds its bound")
            cursor = connection.execute("SELECT * FROM " + name)
            columns = [item[0] for item in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            if len(rows) != count:
                raise ValueError("native callback ledger lost full table population")
            tables[name] = {"columns": columns, "row_count": count, "rows": rows}
    finally:
        connection.close()
    if [_pin(path) for path in paths] != before or wal.exists() != (wal in paths):
        raise ValueError("closed native callback ledger changed during readback")
    record = {"schema": "finite-proof-query-native-callback-ledger@1",
        "stage": "after-native-STOP-before-owner-close", "execution_path": str(execution_path),
        "file_pins": before, "tables": tables, "native_task": task,
        "selected_task_cid": selected["task_cid"], "duckdb_version": duckdb.__version__,
        "read_only": True, "freshness_asserted": False, "execution_authority": False,
        "proof_authority": False, "completion_authority": False,
        "native_settlement": "unknown-retained", "native_failed_claim": False}
    # Retain the real complete rows even if their shape refuses qualification.
    _write(output, record)
    receipt = task.get("body", {}).get("completion_receipt", {})
    identity = _validated_database_claim_identity(receipt)
    if (task.get("status") != "in_progress" or task.get("revision") != selected["revision"] + 2
            or receipt.get("operation") != "database_attempt_admitted"
            or receipt.get("claim_phase_schema") != TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA
            or type(receipt.get("claimed_from_revision")) is not int
            or receipt["claimed_from_revision"] != selected["revision"]
            or type(receipt.get("admitted_from_revision")) is not int
            or receipt["admitted_from_revision"] != task["revision"] - 1
            or receipt.get("attempt_execution_phase") != "claimed"
            or type(receipt.get("attempt_execution_revision")) is not int
            or receipt["attempt_execution_revision"] != 1):
        raise ValueError("refused child lost its genuine native admitted claim cursor")
    attempts = tables["database_task_attempts"]["rows"]
    if len(attempts) != 1:
        raise ValueError("refused child has an ambiguous native attempt population")
    attempt = attempts[0]
    if (attempt["task_cid"] != selected["task_cid"] or attempt["task_alias"] != selected["task_alias"]
            or any(type(attempt[name]) is not type(value) or attempt[name] != value
                   for name, value in identity.items())
            or attempt["status"] != "running" or attempt["committed_phase"] != "context"
            or attempt["revision"] != 2 or attempt["finished_at_ms"] is not None):
        raise ValueError("refused child native execution cursor is not exact and unresolved")
    binding = json.loads(attempt["body_json"])["control_binding"]
    unsigned = {name: value for name, value in binding.items() if name != "binding_id"}
    basis = binding["database_portal_binding_basis"]
    if (binding["schema"] != DATABASE_CONTROL_CLAIM_BINDING_SCHEMA
            or binding["task_cid"] != selected["task_cid"]
            or any(type(binding[name]) is not type(value) or binding[name] != value
                   for name, value in identity.items())
            or binding["control_expected_status"] != task["status"]
            or binding["control_expected_revision"] != task["revision"]
            or binding["binding_id"] != content_identity(unsigned)
            or basis["schema"] != DATABASE_PORTAL_BINDING_BASIS_SCHEMA
            or basis["task_alias"] != selected["task_alias"]
            or basis["task_revision"] != task["revision"]
            or basis["control_task_projection_cid"] != binding["control_task_projection_cid"]
            or basis["task_body_digest"] != "sha256:" + hashlib.sha256(
                canonical_json(task["body"]).encode("utf-8")).hexdigest()
            or binding["database_portal_binding_basis_cid"] != content_identity(basis)):
        raise ValueError("unknown callback lost its native claim and Portal basis binding")
    phases = sorted(tables["attempt_phases"]["rows"], key=lambda row: row["revision"])
    if ([(row["phase"], row["revision"]) for row in phases] != [("claimed", 1), ("context", 2)]
            or any(row["attempt_id"] != identity["attempt_id"]
                   or row["fencing_token"] != identity["fencing_token"]
                   or row["fence_epoch"] != identity["fence_epoch"] for row in phases)):
        raise ValueError("refused child committed an unexpected native execution phase")
    claimed, context = (json.loads(row["body_json"]) for row in phases)
    if claimed != {} or set(context) != {"resumed"} or type(context["resumed"]) is not bool:
        raise ValueError("refused child changed native claimed/context phase material")
    callbacks = tables["provider_invocations"]["rows"]
    if len(callbacks) != 1 or tables["effect_claims"]["rows"]:
        raise ValueError("refused child has accepted provider or effect evidence")
    callback = callbacks[0]
    expected = {"schema": "database-portal-callback-intent@1",
        "callback_state": "started_outcome_unknown", "provider_effect_state": "unknown_may_have_started",
        "task_cid": selected["task_cid"], **{name: identity[name] for name in
            ("attempt_id", "claim_id", "lease_id", "fencing_token", "fence_epoch")}}
    payload = json.loads(callback["result_json"])
    if (callback["task_cid"] != selected["task_cid"] or callback["attempt_id"] != identity["attempt_id"]
            or callback["owner_session_id"] != identity["owner_session_id"]
            or callback["idempotency_key"] != "provider:" + identity["attempt_id"]
            or json.dumps(payload, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False)):
        raise ValueError("refused child has no exact genuine unknown callback ledger")
    return {"path": str(output), **_pin(output), "native_settlement": "unknown-retained",
        "execution_authority": False, "proof_authority": False, "completion_authority": False,
        "native_failed_claim": False, "accepted_provider_receipt": False,
        "committed_effect_rows": 0, "full_ledger_rows_retained": True}


def _retain_broker_custody(*, broker, output, stage):
    """Copy real detached material and bounded bytes without granting freshness."""
    from ipfs_accelerate_py.agent_supervisor.runtime.finite_proof_query_worktree_allocation import (
        FrozenFiniteProofQueryWorktreeAllocation,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.finite_proof_query_worker_source_custody import (
        FrozenFiniteProofQueryWorkerSourceBaseline, FrozenFiniteProofQueryWorkerSourceCustody,
    )
    output = Path(output)
    output.mkdir(parents=True, mode=0o700)
    raw_root = output / "raw-files"
    raw_root.mkdir(mode=0o700)
    record = {"schema": "finite-proof-query-harness-custody-retention@1", "stage": stage,
        "materials": {}, "current_files": [], "original_git_files": [],
        "freshness_asserted": False, "proof_authority": False, "completion_authority": False,
        "process_origin_attested": False, "current_files_may_reflect_authorized_edit_or_merge": True}
    paths = {}

    def retain_material(name, value):
        path = output / (name + ".json")
        _write(path, value)
        path.chmod(0o400)
        record["materials"][name] = {"path": str(path), **_pin(path)}

    def add_path(role, path):
        if len(paths) >= 128:
            raise ValueError("bounded harness custody path population exceeded")
        paths.setdefault(str(Path(path).absolute()), []).append(role)

    baseline = broker._worker_source_baseline
    if type(baseline) is FrozenFiniteProofQueryWorkerSourceBaseline:
        retain_material("source-baseline", baseline.material_binding)
        original = dict(baseline._git_raw)
        for name, witness in baseline._git_files:
            raw = original[name]
            if len(raw) > 262144:
                raise ValueError("bounded original Git control retention exceeded")
            path = raw_root / ("original-git-" + str(len(record["original_git_files"])) + ".bin")
            path.write_bytes(raw)
            path.chmod(0o400)
            record["original_git_files"].append({"name": name, "witness": witness.material(),
                "retained": str(path), **_pin(path)})
            add_path("canonical-git:" + name, witness.path)
        for witness in baseline._scope._custody._files:
            if witness.role.startswith("working:"):
                add_path("canonical-" + witness.role, witness.path)
    else:
        record["source_baseline_available"] = False
    allocation = broker._allocation
    if type(allocation) is FrozenFiniteProofQueryWorktreeAllocation:
        retain_material("allocation", allocation.material_binding)
        retain_material("original-portal-binding", json.loads(allocation._binding_bytes))
        retain_material("original-portal-identity", json.loads(allocation._portal_identity_bytes))
        retain_material("original-lifecycle", allocation._frozen_lifecycle)
        add_path("portal-binding", allocation._paths.binding)
        add_path("portal-projection", allocation._paths.task_projection)
        add_path("lifecycle-workspace", allocation._store.workspace_path_for(allocation._worktree))
        lifecycle = allocation._frozen_lifecycle
        add_path("lifecycle-task-index", allocation._store.task_index_path_for(
            canonical_task_cid=lifecycle["canonical_task_cid"], task_id=lifecycle["task_id"],
            attempt=lifecycle["attempt"]))
        add_path("allocated-git-marker", allocation._worktree / ".git")
    else:
        record["allocation_available"] = False
    custody = broker._worker_source_custody
    if type(custody) is FrozenFiniteProofQueryWorkerSourceCustody:
        retain_material("source-custody", custody.material_binding)
        for name, witness in custody._files:
            add_path("linked-control:" + name, witness.path)
    else:
        record["source_custody_available"] = False
    value = json.loads(broker._closure._value)
    closure = value["indexed_plan"]["proof_query_closure"]
    for name in ("projection", "verification", "applicability"):
        add_path("proof-cas:" + name, broker._catalog.index.artifacts.path_for(closure[name + "_cid"]))
    for source, roles in sorted(paths.items()):
        item = {"path": source, "roles": sorted(set(roles)), "current_freshness_asserted": False}
        bound = 16 * 1024**2 if any(role.startswith("proof-cas:") for role in roles) else 262144
        try:
            fd = os.open(source, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
            try:
                before = os.fstat(fd)
                if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                        or not 0 <= before.st_size <= bound):
                    raise ValueError("bounded regular custody retention file required")
                with os.fdopen(fd, "rb", closefd=False) as stream:
                    raw = stream.read(bound + 1)
                after = os.fstat(fd)
                if (len(raw) > bound or any(getattr(before, key) != getattr(after, key)
                        for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))):
                    raise ValueError("custody retention file changed during bounded read")
            finally:
                os.close(fd)
            path = raw_root / ("current-" + str(len(record["current_files"])) + ".bin")
            path.write_bytes(raw)
            path.chmod(0o400)
            item.update(status="observed", retained=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                bytes=len(raw), source_stat={key: getattr(before, key) for key in
                    ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink", "st_mtime_ns", "st_ctime_ns")})
        except FileNotFoundError:
            item.update(status="absent", absence_is_not_freshness_or_cleanup_authority=True)
        record["current_files"].append(item)
    _write(output / "manifest.json", record)
    (output / "manifest.json").chmod(0o400)
    return {"path": str(output / "manifest.json"), **_pin(output / "manifest.json"),
        "stage": stage, "freshness_asserted": False}



def _parent_late_control(*, owner, admission, candidate, worker_context, verification_catalog,
        native, command, scheduler, worktree_root, output, mutation):
    """Mutate after genuine parent receiving checks, before literal Popen."""
    from .finite_advisory_spawn_experiment import _native_evidence_snapshot
    from ipfs_accelerate_py.agent_supervisor.entrypoints import admitted_benchmark_runtime as entry
    from ipfs_accelerate_py.agent_supervisor.runtime.finite_proof_query_execution import reserve_finite_proof_query_execution
    output.mkdir(mode=0o700)
    def tasks():
        page = native.source.list_tasks(limit=17)
        if page.next_cursor:
            raise ValueError("complete two-task native page required")
        return [dict(task.to_dict()) for task in page.tasks]
    original_tasks = tasks()
    initial_sql = _native_evidence_snapshot(native.server)["tables"]
    effects = ("task_claims", "task_attempts", "merge_attempts", "completion_receipts")
    initial_head = _git(owner.repository, "rev-parse", "HEAD")
    proof = admission["indexed_plan"]["proof_query_closure"]
    path = (owner.repository / "calc.py" if mutation == "source"
        else owner.index.artifacts.path_for(proof["verification_cid"]))
    original = path.read_bytes()
    mode = path.stat().st_mode & 0o777
    record = {"schema": "finite-proof-query-parent-late-callback-refusal@1", "case": mutation,
        "status": "incomplete", "artifact": str(path), "original_sha256": hashlib.sha256(original).hexdigest(),
        "manager_popen_count": 0, "after_actual_receiving_callback": False, "injected": False,
        "task_rows_before": original_tasks, "publication_head_before": initial_head,
        "effect_rows_before": {name: initial_sql[name] for name in effects},
        "popen_scope": "actual admitted manager child; no manager implies no delegated coding child",
        "proof_authority": False, "completion_authority": False, "training_steps": 0}
    try:
        with reserve_finite_proof_query_execution(owner=owner, admission=admission,
                verification_catalog=verification_catalog, candidate=candidate, worker_context=worker_context,
                server=native.server, source=native.source, output=output / "execution-evidence",
                policy_observer=lambda bound: bound.roots, admission_timeout_seconds=90) as scope:
            runtime = entry.AdmittedBenchmarkRuntime.create(output / "launch",
                admission=admission["finite_admission"]["local_admission"], server=native.server,
                source=native.source, implement=True, implementation_command=command,
                candidate_runner_argv=("/opt/ipfs-supervisor/bin/validation-worker",), finite_execution_scope=scope,
                max_task_attempts=1, lifetime_seconds=300, worker_worktree_root=Path(worktree_root), timeout_ms=30_000)
            real_popen, real_require, real_subprocess = runtime.process._popen, scope.require_runtime, entry.subprocess
            def counted(*args, **kwargs):
                record["manager_popen_count"] += 1
                return real_subprocess.Popen(*args, **kwargs)
            class CountActualPopen:
                Popen = staticmethod(counted)
                def __getattr__(self, name):
                    return getattr(real_subprocess, name)
            def after_actual(runtime_argument, *, before_spawn=False, stopping=False):
                result = real_require(runtime_argument, before_spawn=before_spawn, stopping=stopping)
                if before_spawn and not record["injected"]:
                    record["after_actual_receiving_callback"] = True
                    path.chmod(0o600)
                    path.write_bytes(original + b"\n")
                    path.chmod(mode)
                    record["injected"] = True
                    record["changed_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                return result
            def armed(*args, **kwargs):
                scope.require_runtime, entry.subprocess = after_actual, CountActualPopen()
                try:
                    return real_popen(*args, **kwargs)
                finally:
                    scope.require_runtime, entry.subprocess = real_require, real_subprocess
            runtime.process._popen = armed
            try:
                try:
                    record["start"] = runtime.start().to_dict()
                    record["refused"] = record["start"]["status"] != "succeeded"
                except Exception as error:
                    record["refused"] = True
                    record["exception"] = {"type": type(error).__name__, "message": str(error)[:4096]}
                record["children_count"] = len(runtime._children)
                record["processes_before_stop"] = len(runtime.process.snapshot(runtime.profile).members)
                record["task_rows_after"] = tasks()
                after_sql = _native_evidence_snapshot(native.server)["tables"]
                record["effect_rows_after"] = {name: after_sql[name] for name in effects}
                record["publication_head_after"] = _git(owner.repository, "rev-parse", "HEAD")
                if (not all(record[name] is True for name in ("after_actual_receiving_callback", "injected", "refused"))
                        or record["manager_popen_count"] or record["children_count"] or record["processes_before_stop"]
                        or record["task_rows_after"] != original_tasks
                        or record["effect_rows_after"] != record["effect_rows_before"]
                        or record["publication_head_after"] != initial_head):
                    raise ValueError("late proof-query parent callback escaped the actual pre-Popen fence")
            finally:
                runtime.process._popen = real_popen
                record["stop"] = runtime.stop().to_dict()
                record["remaining_processes"] = len(runtime.process.snapshot(runtime.profile).members)
                record["bootstrap_errors"] = list(runtime.bootstrap_errors)
                runtime.close()
            if record["stop"]["status"] != "succeeded" or record["remaining_processes"] or record["bootstrap_errors"]:
                raise ValueError("stale proof-query parent refusal prevented STOP or isolated cleanup")
        record["lease_released"] = scope.parent_lease.released
        record["resources_after"] = scheduler.snapshot()
        if (not record["lease_released"] or record["resources_after"]["active_lease_count"]
                or record["resources_after"]["waiting_request_count"]):
            raise ValueError("parent refusal retained a lease or waiter")
        record["status"] = "completed"
        return record
    finally:
        if record["injected"]:
            path.chmod(0o600)
            path.write_bytes(original)
            path.chmod(mode)
        record["restored_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        record["restored_bytes_equal"] = path.read_bytes() == original
        _write(output / "result.json", record)



def _cleanup_allocations(*, repository, worktree_root, allocations, published, output):
    """Explicit fixture cleanup after STOP; native recovery is not claimed."""
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        WorktreeLifecycleStore, WorkspaceLifecycleRecord, read_process_birth,
    )
    cleanup_records = []
    store = WorktreeLifecycleStore(repo_root=repository)
    linked = _git(repository, "worktree", "list", "--porcelain")
    for observed in allocations.values():
        allocation = WorkspaceLifecycleRecord.from_dict(observed)
        path = Path(allocation.workspace_path)
        if not path.exists():
            continue
        if (allocation.record_id not in allocations or path.resolve(strict=True) != path
                or not path.is_relative_to(Path(worktree_root)) or path == Path(worktree_root)
                or "worktree " + str(path) not in linked.splitlines()
                or _git(path, "status", "--porcelain")):
            raise ValueError("published allocation identity/cleanliness cannot authorize fixture cleanup")
        if read_process_birth(allocation.owner.pid) == allocation.owner:
            raise ValueError("observed allocation still has its live owner birth")
        current_allocation = store.load_workspace(path)
        if current_allocation is not None and any(
                getattr(current_allocation, field) != getattr(allocation, field)
                for field in ("record_id", "task_id", "canonical_task_cid", "owner", "lease_id",
                              "workspace_path", "branch", "state_dir")):
            raise ValueError("published allocation has a different current lifecycle owner")
        subprocess.run(["git", "-C", str(repository), "merge-base", "--is-ancestor",
            _git(path, "rev-parse", "HEAD"), published], check=True, capture_output=True, timeout=10)
        decision = store.authorize_cleanup(workspace_path=path, branch=allocation.branch,
            expected_state_dir=allocation.state_dir)
        cleanup_records.append({"allocation": allocation.to_dict(), "decision": decision.to_dict(),
            "native_lifecycle_record_after_stop": None if current_allocation is None else current_allocation.to_dict(),
            "scope": "explicit owner fixture cleanup after native STOP; not native completion recovery"})
        if not decision.allowed:
            raise ValueError("native lifecycle owner refused published fixture worktree cleanup")
        subprocess.run(["git", "-C", str(repository), "worktree", "remove", str(path)],
            check=True, capture_output=True, timeout=30)
    _write(output / "owner-fixture-worktree-cleanup.json", cleanup_records)
    if (repository / ".git/worktrees").exists():
        if any((repository / ".git/worktrees").iterdir()):
            raise ValueError("unaccounted linked worktree prevents successor custody")
        (repository / ".git/worktrees").rmdir()
    return cleanup_records


def replay(output):
    """Historical signed/public replay after publication, without current claims."""
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_proof_query_admission as proof
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_candidate_runner as base
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_proof_query_worker_context as public
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
    admission = json.loads((output / "before-admission.json").read_bytes())
    value, historical = proof._received(admission)
    candidate = json.loads((output / "candidate-descriptor.json").read_bytes())
    descriptor = json.loads((output / "worker-context-descriptor.json").read_bytes())
    body = base.load_finite_repository_candidate(artifact=Path(candidate["artifact"]), expected_sha256=candidate["sha256"])
    public.validate_finite_proof_query_worker_descriptor(descriptor=descriptor, candidate=body)
    materialized = json.loads((output / "materialized.json").read_bytes())
    with IntentRepository(output / "private/intent.duckdb") as intent:
        tasks = intent.list_tasks()
        statuses = {row["task_alias"]: row["status"] for row in tasks}
        if (statuses != {"FINITE-TYPE": "completed", "FINITE-OFFSET": "completed"}
                or sorted(row["task_cid"] for row in tasks) != sorted(materialized["task_cids"])):
            raise ValueError("historical replay lost the complete original native population")
        plan = intent.get_plan(materialized["plan_id"])
        for field, expected in (("finite_repository_admission_ref", value["finite_admission"]),
                                ("finite_proof_query_admission_ref", value)):
            ref = plan["body"][field]
            raw = Path(ref["path"]).read_bytes()
            if (json.loads(raw) != expected or len(raw) != ref["bytes"]
                    or hashlib.sha256(raw).hexdigest() != ref["sha256"]
                    or cid_for_structured(expected) != ref["admission_cid"]):
                raise ValueError("native historical signed reference differs")
    return {"schema": "finite-proof-query-worker-historical-replay@1", "historical_integrity_verified": True,
        "public_worker_context_verified": True, "task_statuses": statuses, "current_freshness_claimed": False,
        "observed_current": historical["observed_current"], "training_steps": 0}


def run(output, *, python_executable, lean_executable, handoff_root, worktree_root, child_control=None, post_birth_control=None):
    from .native_quack_qualification import open_existing_native_owner
    from .terminal_container_supervisor import _native_diagnostics
    from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
    from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
    from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
        WorktreeLifecycleStore, WorkspaceLifecycleRecord, read_process_birth,
    )
    from ipfs_accelerate_py.agent_supervisor.planning.finite_integer_codebase import build_finite_integer_intent
    from ipfs_accelerate_py.agent_supervisor.planning.repository_plan_preview import RepositoryPlanPreviewOwner
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_admission as boundary
    from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
    from ipfs_accelerate_py.agent_supervisor.runtime.finite_repository_candidate_runner import author_finite_repository_candidate
    from ipfs_accelerate_py.agent_supervisor.runtime.finite_proof_query_execution import reserve_finite_proof_query_execution
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_proof_query_admission as proof_boundary
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_proof_query_worker_context as public
    from ipfs_accelerate_py.agent_supervisor.planning.finite_proof_query_join import derive_finite_proof_spec
    from ipfs_datasets_py.duckdb_control.codebase_verification_catalog import CodebaseVerificationCatalog
    from ipfs_datasets_py.logic.software_contracts import codebase_verification, codebase_applicability
    from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import run_owner_local_task_validations
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon
    from ipfs_datasets_py.logic.software_contracts.codebase_finite_integer_observation import seal_finite_integer_tools
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import (
        IntegerOffsetContract, UnsupportedIntegerProfile, compile_integer_offset,
    )

    output = Path(output).absolute()
    if child_control not in (None, "proof", "epoch"):
        raise ValueError("exact native isolated-child control required")
    if post_birth_control not in (None, *POST_BIRTH_CONTROLS) or (child_control and post_birth_control):
        raise ValueError("one exact pre-birth or post-birth native control required")
    if output.exists() or output.parent.resolve() != output.parent:
        raise ValueError("fresh exact qualification output required")
    output.mkdir(mode=0o755)
    private = output / "private"
    private.mkdir(mode=0o700)
    started = time.monotonic()
    pins = [_pin(importlib.import_module(name).__file__) for name in SOURCES]
    copies = output / "selected-source-snapshot"
    copies.mkdir()
    for name, pin in zip(SOURCES, pins):
        path = copies / (name + ".py")
        path.write_bytes(Path(pin["path"]).read_bytes())
        if _pin(path)["sha256"] != pin["sha256"]:
            raise ValueError("selected producer changed during sequential copy")
    _write(output / "selected-source-pins.json", {"schema": "finite-proof-query-native-source-pins@1",
        "source_modules": {name: pin for name, pin in zip(SOURCES, pins)},
        "source_module_count": len(SOURCES),
        "scope": "selected native producers and boundary dependencies; not a complete import census"})
    repository = output / "repository"
    inventory = _sources(repository)
    original_commit = _git(repository, "rev-parse", "HEAD")
    profile, lifecycle = private / "profile", private / "lifecycle"
    Supervisor.init_local(repository=repository, consent=True, profile_dir=profile, lifecycle_dir=lifecycle)
    document, catalog = build_finite_integer_intent(INTENT), _catalog()
    tools = seal_finite_integer_tools(python_executable=Path(python_executable).resolve(strict=True),
        lean_executable=Path(lean_executable).resolve(strict=True))
    scheduler = _scheduler(private / "resource-admission.json")
    connection, index = _open(output)
    report = {"schema": SCHEMA, "status": "incomplete", "worker_launched": False,
        "training_steps": 0, "provider_calls": 0, "production_activated": False,
        "task_omission_authority": False, "universal_python_semantics_proved": False,
        "inventory_paths": sorted(inventory), "original_commit": original_commit,
        "new_ducklake_hydration": False, "stage_timings": []}
    def observed_stage(stage):
        report["stage_timings"].append({"stage": stage, "seconds": time.monotonic() - started})
        raw = json.dumps(report["stage_timings"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        if len(raw) > 65536:
            raise ValueError("bounded fixture stage progress exceeded")
        (output / "stage-timings.json").write_bytes(raw)
        with (output / "stage-timings.jsonl").open("ab") as stream:
            stream.write(json.dumps(report["stage_timings"][-1], sort_keys=True, allow_nan=False).encode() + b"\n")
    observed_stage("native_fixture_prepared")
    try:
        first = index.prepare_current(repository, repository_id="repository:finite-worker-qualification",
            operation_id="initial", expected_head=None, scheduler=scheduler,
            admission_timeout_seconds=90).head
        observed_stage("initial_source_capture_returned")
        captured = index.load(first.manifest_cid)
        entries = {entry.path: entry for entry in captured.snapshot.entries}
        if set(entries) != set(inventory):
            raise ValueError("captured inventory lost a repository unit")
        ast_units = {}
        for name in ("calc.py", "decoy.py", "unsupported.py"):
            unit = next(item for item in captured.units if item.source_key == entries[name].source_key)
            if unit.ast_cid is None or index.load_ast_artifact(captured, name) is None:
                raise ValueError("captured AST inventory lost " + name)
            ast_units[name] = unit.ast_cid
        try:
            compile_integer_offset((repository / "unsupported.py").read_bytes(),
                IntegerOffsetContract(path="unsupported.py", function_name="dynamic", parameter="value", offset=2),
                revision="snapshot:" + first.snapshot_cid)
        except UnsupportedIntegerProfile as error:
            unsupported = {"rejected": True, "error": str(error)}
        else:
            raise ValueError("unsupported AST gained an integer-model fact")
        _write(output / "captured-inventory.json", {"paths": sorted(entries), "ast_units": ast_units,
            "unsupported_integer_profile": unsupported})
        owner = RepositoryPlanPreviewOwner(index=index, repository=repository, expected_head=first,
            scheduler=scheduler, timeout_seconds=90, memory_mb=1024)
        request = _request(index, repository, first, document, catalog, _authority_materials())
        graph, manifest, bindings = _native_graph(repository=repository, request=request,
            profile=profile, lifecycle=lifecycle)
        tasks = {task.task_key: task for task in graph.tasks}
        declaration = boundary.author_finite_repository_declaration(owner=owner, manifest=manifest,
            request=request, intent_document=document, source_text=INTENT, operation_catalog=catalog,
            tool_policy=tools, task_bindings=bindings)
        proof_contract, proof_domain = derive_finite_proof_spec(intent_document=document, source_text=INTENT)
        controls = dict(expected_head=first, scheduler=scheduler)
        verification = codebase_verification.verify_current_codebase_unit(index, repository,
            path="calc.py", contracts=[proof_contract], **controls)
        applicability = codebase_applicability.verify_current_codebase_applicability(index, repository,
            verification_cid=verification.artifact_cid, domains=[proof_domain], **controls)
        verification_catalog = CodebaseVerificationCatalog(index)
        projection = verification_catalog.publish(repository, verification_cid=verification.artifact_cid,
            applicability_cid=applicability.artifact_cid, operation_id="native-worker-proof-query", **controls)
        observed_stage("native_proof_publication_returned")
        _write(output / "native-verification.json", verification.to_dict())
        _write(output / "native-applicability.json", applicability.to_dict())
        _write(output / "native-proof-projection.json", projection.to_dict())
        proof_admission = proof_boundary.admit_finite_proof_query_plan(owner=owner, declaration=declaration,
            graph=graph, verification_catalog=verification_catalog, output=output / "before-preview",
            policy_observer=lambda bound: bound.roots)
        observed_stage("native_proof_admission_returned")
        admission = proof_admission["finite_admission"]
        selected_source = boundary.verify_finite_repository_admission(admission=admission)["semantic_context"]["source_cid"]
        if selected_source != entries["calc.py"].source_cid or selected_source == entries["decoy.py"].source_cid:
            raise ValueError("same-name decoy changed the selected source identity")
        _write(output / "before-admission.json", proof_admission)
        with IntentRepository(private / "intent.duckdb") as intent:
            materialized = proof_boundary.materialize_finite_proof_query_plan(owner=owner, admission=proof_admission,
                verification_catalog=verification_catalog, intent=intent, output=output / "materialization-preview",
                policy_observer=lambda bound: bound.roots)
            _write(output / "materialized.json", materialized)
        observed_stage("native_materialization_returned")
        (private / "intent.duckdb").chmod(0o600)
        with open_existing_native_owner(database=private / "intent.duckdb", checkout=repository,
                state_dir=private / "owner", repository_id=manifest["payload"]["repository_cid"],
                execution_routes={task.task_key: GROK_CODEX_EXECUTION_MODE for task in graph.tasks}) as native:
            driver = DatabaseImplementationDaemon(database_path=native.database,
                coordination_path=private / "prerequisite-coordination.duckdb",
                execution_path=private / "prerequisite-execution.duckdb", authority_mode="quack",
                task_source_kind="duckdb", owner_session_id="session:finite-worker-prerequisite",
                process_instance_id=native.identity.process_birth_id, quack_uri=native.identity.listen_uri,
                task_source=native.source, close_task_source=False,
                state_owner_bootstrap_credentials=native.credentials, strict_task_sharding=True,
                max_task_attempts=1, lease_ms=120_000, require_real_execution=True).open()
            try:
                attempt = driver.claim_next()
                if attempt is None or attempt.task_cid != tasks["FINITE-TYPE"].task_cid:
                    raise ValueError("actual native prerequisite claim selected a different task")
                prerequisite = native.source.get_task(attempt.task_cid)
                checked = run_owner_local_task_validations(server=native.server, task_cid=attempt.task_cid,
                    attempt_id=attempt.attempt_id, expected_revision=prerequisite.revision)
                if checked["passed"] is not True:
                    raise ValueError("actual native public prerequisite did not pass")
                claimed = dict(prerequisite.body["completion_receipt"])
                digest = checked["results"][0]["evidence_digest"]
                native.source.compare_and_set_status(prerequisite.task_cid, prerequisite.revision, "completed",
                    receipt={"operation": "database_complete", "evidence_digest": digest,
                        **{key: claimed[key] for key in ("attempt_id", "claim_id", "lease_id",
                            "owner_session_id", "fencing_token", "fence_epoch")}},
                    expected_control_receipt=claimed, evidence_digests=[digest])
                if native.source.get_task(attempt.task_cid).status != "completed":
                    raise ValueError("native prerequisite completion was not committed")
                _write(output / "prerequisite-native-claim.json", claimed)
                _write(output / "prerequisite-validation.json", checked)
            finally:
                driver.close()
            observed_stage("native_prerequisite_completion_returned")
            residual_task = native.source.get_task(tasks["FINITE-OFFSET"].task_cid)
            if residual_task.status != "ready":
                raise ValueError("residual must remain ready for the genuine native claim")
            with native.server._lock:
                with IntentRepository(bound_connection=native.server._connection, install_schema=False) as intent:
                    residual = intent.get_task(residual_task.task_cid)
                    candidate = author_finite_repository_candidate(admission=admission, intent=intent,
                        task_cid=residual_task.task_cid, after_bytes=b"def increment(n: int) -> int:\n    return n + 2\n",
                        output=Path(handoff_root) / "candidate.json")
            _write(output / "candidate-descriptor.json", candidate)
            worker_context = public.author_finite_proof_query_worker_context(owner=owner,
                admission=proof_admission, candidate_descriptor=candidate,
                output=Path(handoff_root) / "proof-query-context.json")
            observed_stage("public_context_authoring_returned")
            _write(output / "worker-context-descriptor.json", worker_context)
            evidence = {str(path): _pin(path) for root in (output / "cas", output / "before-preview",
                output / "materialization-preview") for path in root.rglob("*") if path.is_file()}
            for path in (output / "before-admission.json", output / "materialized.json",
                         Path(materialized["finite_admission_ref"]["path"]),
                         Path(materialized["finite_proof_query_admission_ref"]["path"]),
                         Path(candidate["artifact"]), Path(worker_context["artifact"])):
                evidence[str(path)] = _pin(path)
            _write(output / "historical-parent-artifact-pins.json", list(evidence.values()))
            command = shlex.join(["/opt/ipfs-supervisor/bin/owner-worker",
                "--finite-repository-artifact", candidate["artifact"],
                "--finite-repository-sha256", candidate["sha256"],
                "--finite-repository-task-cid", residual["task_cid"],
                "--finite-proof-query-context", worker_context["artifact"],
                "--finite-proof-query-sha256", worker_context["sha256"],
                "--finite-proof-query-context-cid", worker_context["context_cid"]])
            report["parent_late_controls"] = []
            for mutation in (() if child_control or post_birth_control else ("source", "proof")):
                report["parent_late_controls"].append(_parent_late_control(owner=owner,
                    admission=proof_admission, candidate=candidate, worker_context=worker_context,
                    verification_catalog=verification_catalog, native=native, command=command,
                    scheduler=scheduler, worktree_root=worktree_root,
                    output=private / ("late-parent-" + mutation), mutation=mutation))
            with reserve_finite_proof_query_execution(owner=owner, admission=proof_admission,
                    verification_catalog=verification_catalog, candidate=candidate, worker_context=worker_context,
                    server=native.server, source=native.source, output=private / "launch-evidence",
                    policy_observer=lambda bound: bound.roots, admission_timeout_seconds=90) as scope:
                observed_stage("native_execution_reservation_returned")
                _write(output / "execution-scope.json", scope.to_dict())
                runtime = AdmittedBenchmarkRuntime.create(private / "launch", admission=admission["local_admission"],
                    server=native.server, source=native.source, implement=True, implementation_command=command,
                    candidate_runner_argv=("/opt/ipfs-supervisor/bin/validation-worker",),
                    finite_execution_scope=scope, max_task_attempts=1, lifetime_seconds=300,
                    worker_worktree_root=Path(worktree_root), timeout_ms=30_000)
                observed_stage("native_runtime_creation_returned")
                broker = runtime.proof_query_dispatch_broker
                proof_capability = scope._proof_query_closure
                actual_dispatch = proof_capability.require_worker_dispatch_current
                report["broker_custody_snapshots"] = []
                control = {"schema": "finite-proof-query-isolated-child-late-control@1", "case": child_control,
                    "real_dispatch_calls": 0, "genuine_dispatch_returned": False, "injected": False,
                    "boundary": "registered owner wrapper before sudo and UID1001 worker Popen"}
                changed_path, changed_raw, changed_mode = None, None, None
                def after_native_dispatch(*args, **kwargs):
                    nonlocal changed_path, changed_raw, changed_mode
                    result = actual_dispatch(*args, **kwargs)
                    control["real_dispatch_calls"] += 1
                    tag = "gate-" + str(control["real_dispatch_calls"]).zfill(2) + "-before-control"
                    report["broker_custody_snapshots"].append(_retain_broker_custody(broker=broker,
                        output=private / "launch/broker-custody-retention" / tag, stage=tag))
                    # Retention precedes the broker's detached final fence.
                    # Its bytes and original return confer no dispatch authority.
                    if child_control and control["real_dispatch_calls"] == 2 and not control["injected"]:
                        control["genuine_dispatch_returned"] = True
                        closure = proof_admission["indexed_plan"]["proof_query_closure"]
                        page = closure["exact_query"]["page"]
                        control["inventory_before"] = {"epoch": page["epoch"], "inventory_cid": page["inventory_cid"]}
                        if child_control == "epoch":
                            rebuilt = verification_catalog.rebuild_current(repository, expected_head=first,
                                parent_lease=scope.parent_lease, timeout_seconds=60,
                                admission_timeout_seconds=30, memory_mb=512)
                            control["inventory_after"] = {"epoch": rebuilt.epoch, "inventory_cid": rebuilt.inventory_cid}
                        else:
                            changed_path = index.artifacts.path_for(closure["verification_cid"])
                            changed_raw, changed_mode = changed_path.read_bytes(), changed_path.stat().st_mode & 0o777
                            changed_path.chmod(0o600)
                            changed_path.write_bytes(changed_raw + b"\n")
                            changed_path.chmod(changed_mode)
                            control["proof_before_sha256"] = hashlib.sha256(changed_raw).hexdigest()
                            control["proof_after_sha256"] = hashlib.sha256(changed_path.read_bytes()).hexdigest()
                        control["injected"] = True
                        tag = "gate-02-after-control"
                        report["broker_custody_snapshots"].append(_retain_broker_custody(broker=broker,
                            output=private / "launch/broker-custody-retention" / tag, stage=tag))
                    return result
                proof_capability.require_worker_dispatch_current = after_native_dispatch
                post_control = (_PostBirthControl(broker=broker, case=post_birth_control,
                    output=private / "post-birth-control") if post_birth_control else None)
                allocations = {}
                try:
                    observed_stage("native_START_entered")
                    report["start"] = runtime.start().to_dict()
                    observed_stage("native_START_returned")
                    if report["start"]["status"] != "succeeded":
                        raise ValueError("native START failed")
                    report["manager_launched"] = True
                    observations, previous = [], None
                    allocation_store = WorktreeLifecycleStore(repo_root=repository)
                    deadline = time.monotonic() + 120
                    refusal_deadline = None
                    while time.monotonic() < deadline:
                        task = native.source.get_task(residual["task_cid"])
                        for allocation in allocation_store.iter_records():
                            if (allocation.canonical_task_cid == residual["task_cid"]
                                    or allocation.task_id == residual["task_alias"]):
                                allocations[allocation.record_id] = allocation.to_dict()
                        current = (task.status, task.revision)
                        if current != previous:
                            observations.append({"status": task.status, "revision": task.revision,
                                "seconds": time.monotonic() - started})
                            previous = current
                        diagnostics = _bounded_dispatch_refusals(broker.observations)
                        if diagnostics:
                            report["native_dispatch_refusal_diagnostics"] = diagnostics
                            if refusal_deadline is None:
                                refusal_deadline = min(deadline, time.monotonic() + 10)
                                observed_stage("actual_native_dispatch_refusal_observed")
                            if time.monotonic() >= refusal_deadline:
                                observed_stage("native_refusal_settlement_wait_expired")
                                break
                        if task.status in {"completed", "failed", "blocked", "cancelled"}:
                            break
                        if not runtime.process.snapshot(runtime.profile).members:
                            raise ValueError("native supervisor exited before residual completion")
                        time.sleep(0.25)
                    observed_stage("native_task_poll_finished")
                    report["task_observations"] = observations
                    report["task"] = {"status": task.status, "revision": task.revision,
                        "body": dict(task.body)}
                    report["observed_worker_allocations"] = list(allocations.values())
                    report["resource_before_stop"] = scheduler.snapshot()
                    report["native_dispatch_observations"] = list(broker.observations)
                    report["worker_launched"] = any(row.get("status") == "spawned"
                        and row.get("boundary") == "isolated_worker" and row.get("worker_uid") == 1001
                        for row in broker.observations)
                    if post_control:
                        report["broker_successful_isolated_release_observed"] = report["worker_launched"]
                        report["worker_launched"] = True if post_control.record.get("kernel_witness_validated") else None
                        report["worker_birth_observation_scope"] = "independent retained live kernel witness; not broker release success"
                        report["post_birth_control"] = post_control.record
                finally:
                    proof_capability.require_worker_dispatch_current = actual_dispatch
                    try:
                        report["broker_custody_snapshots"].append(_retain_broker_custody(broker=broker,
                            output=private / "launch/broker-custody-retention/before-STOP", stage="before-STOP"))
                    except Exception as error:
                        report["broker_custody_retention_error"] = {"type": type(error).__name__,
                            "message_sha256": hashlib.sha256(str(error).encode()).hexdigest()}
                    if post_control:
                        try:
                            receipt, effects_before = _post_birth_worktree_effects(allocations=allocations,
                                output=post_control.output / "worktree-effects-before-STOP", stage="before-native-STOP")
                            post_control.record["worktree_effects_before_STOP"] = receipt
                            post_control.record["worktree_effects_before_STOP_errors"] = effects_before["errors"]
                            post_control.event("actual_candidate_effects_retained_before_STOP")
                        except Exception as error:
                            # Retention failure cannot bypass mandatory native STOP.
                            post_control.record["worktree_effects_before_STOP_errors"] = [{"type": type(error).__name__,
                                "message_sha256": hashlib.sha256(str(error).encode()).hexdigest()}]
                            report["post_birth_retention_failure_before_STOP"] = post_control.record["worktree_effects_before_STOP_errors"]
                    observed_stage("native_STOP_entered")
                    try:
                        report["stop"] = runtime.stop().to_dict()
                        observed_stage("native_STOP_returned")
                        report["remaining_processes"] = len(runtime.process.snapshot(runtime.profile).members)
                        report["bootstrap_errors"] = runtime.bootstrap_errors
                        report["native_diagnostics"] = _native_diagnostics(runtime.state)
                        _write(output / "native-lifecycle.json", report)
                    finally:
                        try:
                            runtime.close()
                        finally:
                            if post_control:
                                post_control.restore()
                    report["broker_closed"] = broker._closed
                    report["broker_socket_removed"] = not broker.socket_path.exists()
                    if post_control:
                        final_task = native.source.get_task(residual["task_cid"])
                        report["task_after_stop"] = {"status": final_task.status, "revision": final_task.revision,
                            "body": dict(final_task.body)}
                        from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
                            _checkout_sidecar_duckdb, _database_daemon_quack_sidecar_paths,
                        )
                        database = Path(native.server.config.database_path)
                        coordination_path, execution_path = _database_daemon_quack_sidecar_paths(database,
                            coordination_path=broker._attempt_authority_path,
                            execution_path=_checkout_sidecar_duckdb(database.with_name(database.stem + ".execution.duckdb")))
                        closed, receipts = {}, {}
                        for name, path in (("coordination", coordination_path), ("execution", execution_path)):
                            receipts[name], closed[name] = _post_birth_native_snapshot(path,
                                output=post_control.output / ("closed-" + name + ".json"), stage="after-native-STOP-and-runtime-close")
                        post_control.record["closed_native_authorities"] = receipts
                        _write(post_control.output / "closed-native-authorities.json", {"schema": "finite-proof-query-post-birth-closed-authorities@1",
                            "stage": "after-native-STOP-and-runtime-close", "authorities": closed,
                            "retained_before_control_assertions": True,
                            "all_actual_tables_and_rows_retained": all(row["status"] == "retained" for row in closed.values())})
                        post_control.record["closed_native_authorities_artifact"] = _pin(post_control.output / "closed-native-authorities.json")
                        receipt, effects_after = _post_birth_worktree_effects(allocations=allocations,
                            output=post_control.output / "worktree-effects-after-STOP", stage="after-native-STOP-before-any-worktree-removal")
                        post_control.record["worktree_effects_after_STOP"] = receipt
                        post_control.record["worktree_effects_after_STOP_errors"] = effects_after["errors"]
                        witness = post_control.record.get("kernel_birth_witness", {})
                        post_control.record["birth_disappearance_after_STOP"] = [_post_birth_gone(witness[name])
                            for name in ("worker", "sudo") if name in witness]
                        post_control.event("closed_native_rows_and_birth_disappearance_retained")
                        _write(output / "native-lifecycle.json", report)
                    if changed_path is not None:
                        changed_path.chmod(0o600)
                        changed_path.write_bytes(changed_raw)
                        changed_path.chmod(changed_mode)
                        control["proof_restored_bytes_equal"] = changed_path.read_bytes() == changed_raw
                    if child_control:
                        control["observations"] = list(broker.observations)
                        _write(output / "isolated-child-late-control.json", control)
                if report.get("broker_custody_retention_error"):
                    raise ValueError("native lifecycle stopped but bounded custody retention was incomplete")
                expected_status = {"in_progress"} if child_control else {"completed"}
                if ((not post_control and report["task"]["status"] not in expected_status) or report["stop"]["status"] != "succeeded"
                        or report["remaining_processes"] or report["bootstrap_errors"]):
                    raise ValueError("isolated native residual worker lifecycle is incomplete; retained stage timings and bounded broker refusal diagnostics identify the observed boundary")
                if not report["broker_closed"] or not report["broker_socket_removed"]:
                    raise ValueError("native dispatch broker was not retired after STOP")
                spawned = [row for row in broker.observations if row["status"] == "spawned"]
                if post_control:
                    _validate_post_birth_control(control=post_control, report=report, closed=closed)
                elif child_control:
                    if (not control["genuine_dispatch_returned"] or not control["injected"]
                            or not any(row.get("boundary") == "native_daemon_to_owner_wrapper" for row in spawned)
                            or any(row.get("boundary") == "isolated_worker" for row in spawned)
                            or not any(row["status"] == "refused"
                                and row.get("boundary") == "isolated_worker"
                                and row.get("stage") == "detached_proof_close"
                                and row.get("worker_birth_observed") is False
                                and row.get("unacknowledged_child_birth_possible") is False
                                for row in broker.observations)):
                        raise ValueError("late native isolated-child mutation escaped the final broker fence")
                    if child_control == "epoch" and control["inventory_after"]["epoch"] <= control["inventory_before"]["epoch"]:
                        raise ValueError("isolated-child epoch control did not genuinely rebuild native inventory")
                    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
                        _checkout_sidecar_duckdb, _database_daemon_quack_sidecar_paths,
                    )
                    database = Path(native.server.config.database_path)
                    _, execution_path = _database_daemon_quack_sidecar_paths(database,
                        execution_path=_checkout_sidecar_duckdb(database.with_name(
                            database.stem + ".execution.duckdb")))
                    report["native_callback_ledger"] = _retain_unknown_callback_ledger(
                        execution_path=execution_path, task=report["task"], selected=residual,
                        output=output / "native-provider-unknown-ledger.json")
                    report.update(native_settlement="unknown-retained", native_failed_claim=False,
                        execution_authority=False, proof_authority=False, completion_authority=False)
                elif (not any(row.get("boundary") == "native_daemon_to_owner_wrapper" for row in spawned)
                        or not any(row.get("boundary") == "isolated_worker" and row.get("worker_uid") == 1001
                            and row.get("isolated_uid_observed") is True for row in spawned)):
                    raise ValueError("both actual native coding dispatch boundaries were not observed")
                report["execution_scope_after_stop"] = scope.to_dict()
        observed_stage("native_owner_close_returned")
        report["resource_after_owner_close"] = scheduler.snapshot()
        published = _git(repository, "rev-parse", "HEAD")
        if post_birth_control:
            publication_unchanged = published == original_commit
            canonical_unchanged = (repository / "calc.py").read_bytes() == inventory["calc.py"].encode()
            report.update(status="completed" if publication_unchanged and canonical_unchanged else "unqualified",
                post_birth_control=post_control.record, publication_unchanged=publication_unchanged,
                canonical_source_unchanged=canonical_unchanged, native_task_completion_claimed=False,
                native_settlement=post_control.record["actual_native_settlement"], effect_settlement="UNKNOWN",
                active_leases=report["resource_after_owner_close"]["active_lease_count"],
                waiting_requests=report["resource_after_owner_close"]["waiting_request_count"],
                global_training_or_model_counts=None,
                literal_fixture_counter_scope="authored MODEL_OFF only; not native provider table or global census",
                worktree_cleanup={"status": "retained_for_independent_readback", "automatic_recovery_qualified": False,
                    "allocation_paths": [row["workspace_path"] for row in allocations.values()],
                    "dirty_or_unpublished_worktrees_not_removed": True},
                execution_authority=False, proof_authority=False, completion_authority=False,
                elapsed_seconds=time.monotonic() - started,
                qualification_scope="genuine isolated kernel birth followed by callback failure or received-frame application ACK drop; no effect-freedom or general rollback theorem")
            _write(output / "result.json", report)
            if not publication_unchanged or not canonical_unchanged or report["active_leases"] or report["waiting_requests"]:
                raise ValueError("post-birth control publication or resource cleanup refused qualification; actual effects retained")
            return report
        if child_control:
            if published != original_commit or (repository / "calc.py").read_bytes() != inventory["calc.py"].encode():
                raise ValueError("isolated child refusal changed canonical source or publication")
            _cleanup_allocations(repository=repository, worktree_root=worktree_root,
                allocations=allocations, published=published, output=output)
            report.update(status="completed", child_control=control, publication_unchanged=True,
                canonical_source_unchanged=True, active_leases=report["resource_after_owner_close"]["active_lease_count"],
                waiting_requests=report["resource_after_owner_close"]["waiting_request_count"],
                qualification_scope="genuine claimed owner wrapper; stale proof or epoch refused before isolated coding child")
            if report["active_leases"] or report["waiting_requests"]:
                raise ValueError("isolated child refusal retained native resources")
            _write(output / "result.json", report)
            return report
        parents = _git(repository, "rev-list", "--parents", "-n", "1", published).split()
        if len(parents) != 3 or parents[1] != original_commit:
            raise ValueError("native publication is not one exact baseline two-parent merge")
        changed = _git(repository, "diff", "--name-only", original_commit, published).splitlines()
        if changed != ["calc.py"]:
            raise ValueError("published edit exceeded the original output permission")
        for name in ("check_type.py", "check_offset.py"):
            subprocess.run([sys.executable, "-B", name], cwd=repository, check=True,
                capture_output=True, timeout=10)
        # Canonical database workers retain their allocated Git worktrees.
        # This fixture explicitly cleans only its published, fenced allocation;
        # native crash recovery and automatic completion cleanup remain separate.
        cleanup_records = _cleanup_allocations(repository=repository, worktree_root=worktree_root,
            allocations=allocations, published=published, output=output)
        fresh = subprocess.run([sys.executable, "-m", MODULE, "replay", str(output)],
            check=True, capture_output=True, text=True, timeout=60)
        _write(output / "fresh-process-replay.json", json.loads(fresh.stdout))
        if any(_pin(path) != pin for path, pin in evidence.items()):
            raise ValueError("worker altered historical signed/CAS evidence")
        if any(_pin(pin["path"]) != pin for pin in pins):
            raise ValueError("selected producer changed during native qualification")
        resources = scheduler.snapshot()
        if resources["active_lease_count"] or resources["waiting_request_count"]:
            raise ValueError("native resources remain after isolated worker STOP")
        report.update(status="completed", elapsed_seconds=time.monotonic() - started,
            published_commit=published, published_commit_parents=parents[1:], changed_paths=changed,
            complete_task_population_retained=True, actual_public_checks_passed=True,
            historical_artifacts_unchanged=True, fresh_process_historical_replay=json.loads(fresh.stdout),
            public_context_cid=worker_context["context_cid"],
            full_proof_closure_cid=proof_admission["indexed_plan"]["proof_query_closure_cid"],
            proof_query_key_count=len(proof_admission["indexed_plan"]["proof_query_closure"]["canonical_key_membership"]),
            execution_sources=pins, active_leases=0, waiting_requests=0,
            qualification_scope="one authored two-task five-input model-off native isolated worker; no convergence claim")
        _write(output / "result.json", report)
        return report
    finally:
        connection.close()


def run_all(output, *, python_executable, lean_executable, handoff_root, worktree_root):
    """Keep successful publication and each genuine stale-child fixture separate."""
    output = Path(output).absolute()
    if output.exists() or output.parent.resolve() != output.parent:
        raise ValueError("fresh proof-query fixture collection required")
    output.mkdir(mode=0o755)
    rows = []
    for name, control in (("positive", None), ("child-late-proof", "proof"), ("child-late-epoch", "epoch")):
        handoffs = Path(handoff_root) / name
        handoffs.mkdir(mode=0o755)
        result = run(output / name, python_executable=python_executable, lean_executable=lean_executable,
            handoff_root=handoffs, worktree_root=worktree_root, child_control=control)
        rows.append({"case": name, "status": result["status"], "result": str(output / name / "result.json"),
            "active_leases": result["active_leases"], "waiting_requests": result["waiting_requests"]})
    summary = {"schema": "finite-proof-query-worker-fixture-collection@1", "status": "completed", "cases": rows,
        "training_steps": 0, "convergence_proved": False,
        "scope": "one authored native publication; two parent and two isolated-child stale-context controls"}
    _write(output / "result.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "replay", "run-all"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--lean", type=Path, default=Path("/toolchains/lean/bin/lean"))
    parser.add_argument("--handoff-root", type=Path, default=Path("/opt/ipfs-supervisor/finite-handoffs"))
    parser.add_argument("--worktree-root", type=Path, default=Path("/opt/ipfs-supervisor/worktrees"))
    parser.add_argument("--child-control", choices=("proof", "epoch"))
    parser.add_argument("--post-birth-control", choices=POST_BIRTH_CONTROLS)
    args = parser.parse_args()
    if args.action == "replay":
        result = replay(args.output)
    elif args.action == "run-all":
        if args.child_control or args.post_birth_control:
            parser.error("run-all preserves separate fixed child controls")
        result = run_all(args.output, python_executable=args.python, lean_executable=args.lean,
            handoff_root=args.handoff_root, worktree_root=args.worktree_root)
    else:
        result = run(args.output, python_executable=args.python, lean_executable=args.lean,
            handoff_root=args.handoff_root, worktree_root=args.worktree_root, child_control=args.child_control,
            post_birth_control=args.post_birth_control)
    printed = result if args.action == "replay" else {"schema": result["schema"], "status": result["status"]}
    print(json.dumps(printed, sort_keys=True))


if __name__ == "__main__":
    main()
