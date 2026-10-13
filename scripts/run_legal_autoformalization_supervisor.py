#!/usr/bin/env python3
"""Hold existing shared host admission while one bounded supervisor runs.

This bootstrap launcher accounts for a CPU inventory envelope. It does not
enforce kernel quotas or authorize semantic completion, training, or publishing.
The native datasets scheduler remains the sole shared host resource authority.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid


SCHEMA = "legal-autoformalization-supervisor-resource-launch/v1"
MAX_LINE_BYTES = 65536
THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false", "CUDA_VISIBLE_DEVICES": "",
    "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
    "IPFS_AUTO_INSTALL": "false", "IPFS_DATASETS_AUTO_INSTALL": "false",
    "IPFS_DATASETS_PY_MINIMAL_IMPORTS": "1",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def redacted(text):
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = re.sub(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9+/_.=:-]+",
                  r"\1 [REDACTED]", text)
    text = re.sub(
        r'''(?i)((?:[A-Za-z0-9_]*)(?:api[_-]?key|token|password|secret|authorization|lease[_-]?key)(?:[A-Za-z0-9_]*)[\"']?\s*[:=]\s*)(?:[\"'][^\"'\r\n]*[\"']|[^\s,;}]+)''',
        r"\1[REDACTED]", text)
    text = re.sub(r"(?i)([?&](?:token|key|secret|password)=)[^\s&#]+",
                  r"\1[REDACTED]", text)
    text = re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@",
                  r"\1[REDACTED]@", text)
    text = re.sub(r"\b(?:sk-|xai-|hf_)[A-Za-z0-9_-]{16,}\b", "[REDACTED]", text)
    return "".join(character for character in text
                   if character in "\t\n\r" or ord(character) >= 32)


def stream_output(pipe, log_path, state):
    """Never write raw child output; bound partial lines before redaction."""
    try:
        with log_path.open("x", encoding="utf-8") as output:
            os.chmod(log_path, 0o600)
            pending = bytearray()
            oversized = False
            while True:
                chunk = os.read(pipe.fileno(), 4096)
                if not chunk:
                    if pending and not oversized:
                        output.write(redacted(pending.decode("utf-8", errors="replace")))
                    break
                for fragment in chunk.splitlines(keepends=True):
                    pending.extend(fragment)
                    if len(pending) > MAX_LINE_BYTES:
                        pending.clear()
                        oversized = True
                    if fragment.endswith((b"\n", b"\r")):
                        if oversized:
                            output.write("[oversized child output line omitted]\n")
                        else:
                            output.write(redacted(pending.decode("utf-8", errors="replace")))
                        output.flush()
                        pending.clear()
                        oversized = False
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        state["output_capture_failed"] = True
    finally:
        pipe.close()


def live_group(pgid):
    """Ignore zombies, which cannot execute and may await a foreign reaper."""
    found = []
    for candidate in Path("/proc").iterdir():
        if not candidate.name.isdigit():
            continue
        try:
            fields = (candidate / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] not in ("Z", "X"):
                found.append(int(candidate.name))
        except (OSError, ValueError, IndexError):
            continue
    return found


def process_birth(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[1]), int(fields[19])


class NativeFenceObserver:
    """Accept only native STOP evidence bound to this exact launched master.

    Managed tracks deliberately create new sessions. Their native coordinator
    owns marker-based cleanup and publishes all_trees_fenced only after those
    trees reach its empty fixed point. The outer process group is insufficient.
    This observer never converts a receipt PID into signaling authority.
    """
    ACTIVE = "ipfs_accelerate_py.agent_supervisor.multi-supervisor-active-binding@1"
    TERMINAL = "ipfs_accelerate_py.agent_supervisor.multi-supervisor-terminal@1"

    def __init__(self, path, label, child, launched_at_ns, identity):
        self.path, self.label, self.child = path, label, child
        self.launched_at_ns, self.identity = launched_at_ns, identity
        self.child_birth = process_birth(child.pid)[1]
        self.generation = None
        self.master_birth = None
        self.active_cid = None
        self.reason = "current_native_terminal_missing"
        self.proof = None

    def owned_master(self, pid):
        if pid == self.child.pid:
            # The Popen child remains exact authority after reaping; its
            # generation is additionally bound to the pre-Popen wall clock.
            return True
        seen = set()
        while pid > 1 and pid not in seen:
            seen.add(pid)
            parent, started = process_birth(pid)
            if parent == self.child.pid:
                return process_birth(parent)[1] == self.child_birth
            pid = parent
        return False

    def read(self):
        descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size > 1024 * 1024:
                raise ValueError("native run artifact is not bounded regular JSON")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                payload = json.loads(stream.read(1024 * 1024 + 1))
            if not isinstance(payload, dict):
                raise ValueError("native run artifact must be a JSON object")
            after = os.fstat(descriptor)
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                    after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError("native run artifact changed during observation")
            return payload
        finally:
            os.close(descriptor)

    def observe(self):
        try:
            payload = self.read()
            schema = payload.get("schema")
            field = "active_binding_cid" if schema == self.ACTIVE else "terminal_receipt_cid"
            if schema not in (self.ACTIVE, self.TERMINAL):
                raise ValueError("unsupported native run artifact")
            body = dict(payload)
            claimed = body.pop(field, None)
            if not claimed or self.identity(body) != claimed:
                raise ValueError("native run artifact CID differs")
            pid, started = payload.get("master_pid"), payload.get("run_started_at_epoch_nanoseconds")
            if (type(pid) is not int or type(started) is not int
                    or payload.get("label") != self.label
                    or not self.launched_at_ns <= started <= time.time_ns()):
                raise ValueError("native run artifact is not this launch generation")
            generation = (pid, started)
            if self.generation is None:
                if not self.owned_master(pid):
                    raise ValueError("native master is not the exact child or observed descendant")
                self.generation = generation
                self.master_birth = self.child_birth if pid == self.child.pid else process_birth(pid)[1]
            if self.generation != generation:
                raise ValueError("native master generation changed")
            if schema == self.ACTIVE:
                if payload.get("active") is not True:
                    raise ValueError("native active binding is inactive")
                if self.active_cid is not None and self.active_cid != claimed:
                    raise ValueError("native active binding changed")
                self.active_cid = claimed
                self.reason = "current_native_master_active"
                return False
            # Fast direct children may finish before ACTIVE is observed.
            # Their exact Popen PID+fresh start+terminal CID bind the run;
            # when ACTIVE was observed its CID must also match exactly.
            if not isinstance(payload.get("active_binding_cid"), str) or not payload["active_binding_cid"]:
                raise ValueError("native terminal has no active-generation binding")
            if self.active_cid is not None and payload.get("active_binding_cid") != self.active_cid:
                raise ValueError("native terminal belongs to a different active binding")
            if payload.get("all_trees_fenced") is not True:
                self.reason = "native_stop_did_not_fence_all_trees"
                return False
            self.proof = {"master_pid": pid, "run_started_at_epoch_nanoseconds": started,
                "master_start_time_ticks": self.master_birth,
                "active_binding_cid": payload["active_binding_cid"],
                "terminal_receipt_cid": claimed, "all_trees_fenced": True}
            self.reason = "native_managed_trees_fenced"
            return True
        except (OSError, ValueError, TypeError, KeyError):
            self.reason = "native_run_artifact_missing_invalid_or_unbound"
            return False


def terminate_group(child, grace_seconds, receipt, persist, native_fence,
                    native_stop_seconds):
    """Keep admission through native detached-tree STOP and outer group exit."""
    receipt["native_stop_verified"] = native_fence.observe()
    if live_group(child.pid):
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + max(grace_seconds, native_stop_seconds)
    last_receipt = 0.0
    while live_group(child.pid) or not receipt["native_stop_verified"]:
        child.poll()
        receipt["native_stop_verified"] = native_fence.observe()
        receipt["native_fence_reason"] = native_fence.reason
        if time.monotonic() >= deadline and live_group(child.pid):
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if time.monotonic() - last_receipt >= 15:
            receipt["status_before_fence_monitor"] = receipt.get(
                "status_before_fence_monitor", receipt["status"])
            receipt["status"] = "resource_held_pending_native_fence"
            receipt["cleanup_pending"] = True
            persist()
            last_receipt = time.monotonic()
        # Missing native evidence deliberately retains the heartbeating lease
        # even after outer group exit. A foreign/unbound PID is never signaled.
        time.sleep(0.25)
    child.wait(timeout=5)
    receipt["owned_group_quiescent"] = True
    receipt["native_fence_proof"] = native_fence.proof
    receipt["cleanup_pending"] = False
    if receipt["status"] == "resource_held_pending_native_fence":
        receipt["status"] = receipt["status_before_fence_monitor"]


def read_policy(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("existing regular shared scheduler state required")
    state = json.loads(path.read_text())
    if not isinstance(state.get("config"), dict):
        raise ValueError("shared scheduler policy is missing")
    return state


def pinned_scheduler(datasets_root, state_path, expected_digest):
    os.environ.update({key: value for key, value in THREAD_ENV.items()
                       if key.startswith("IPFS_")})
    sys.path.insert(0, str(datasets_root))
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import resource_scheduler as native
    if not Path(native.__file__).resolve().is_relative_to(datasets_root):
        raise ValueError("datasets scheduler import escaped pinned source root")
    initial = read_policy(state_path)
    original = deepcopy(initial["config"])
    if expected_digest and digest(original) != expected_digest:
        raise ValueError("shared scheduler policy does not match expected digest")
    config = native.ResourceSchedulerConfig(state_path=state_path, **original)
    config.validate()
    if not config.proof_safety_enabled:
        raise ValueError("shared scheduler proof safety must remain enabled")
    if config.reservations().get("validation", native.LaneReservation()).cpu_slots < 1:
        raise ValueError("shared scheduler validation reserve is missing")

    class PinnedPolicyScheduler(native.GlobalResourceScheduler):
        # The native constructor allows reconfiguration and legacy field
        # normalization. This adapter disables both; admission remains native.
        def _new_state(self):
            raise native.SchedulerStateError("existing shared state disappeared")

        def _validate_state_configuration(self, state, *, allow_reconfigure=False):
            stored = deepcopy(state.get("config"))
            if stored != original:
                raise native.ResourceConfigurationError("shared scheduler policy changed")
            super()._validate_state_configuration(state, allow_reconfigure=False)
            state["config"] = stored

    scheduler = PinnedPolicyScheduler(config)
    if read_policy(state_path)["config"] != original:
        raise ValueError("native initialization changed shared scheduler policy")
    return native, scheduler, original


def positive(value):
    result = float(value)
    if not 0 < result <= 86400:
        raise argparse.ArgumentTypeError("must be in (0, 86400]")
    return result


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--scheduler-state", required=True, type=Path)
    parser.add_argument("--expected-policy-sha256", required=True)
    parser.add_argument("--duration-seconds", required=True, type=positive)
    parser.add_argument("--queue-timeout-seconds", type=positive, default=300)
    parser.add_argument("--acquire-timeout-seconds", type=positive, default=5)
    parser.add_argument("--stop-grace-seconds", type=positive, default=10)
    parser.add_argument("--native-stop-timeout-seconds", type=positive, default=180)
    parser.add_argument("--native-terminal-receipt", required=True, type=Path)
    parser.add_argument("--native-label", required=True)
    parser.add_argument("--cpu-slots", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--memory-mib", type=int, choices=(1024, 2048, 3072, 4096), default=3072)
    parser.add_argument("--process-slots", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("supervisor command is required after --")
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_policy_sha256):
        parser.error("expected-policy-sha256 must be 64 lowercase hexadecimal characters")
    for name in ("datasets_root", "repo_root", "runtime_root", "scheduler_state", "native_terminal_receipt"):
        path = getattr(args, name)
        if not path.is_absolute():
            parser.error(name.replace("_", "-") + " must be absolute")
        setattr(args, name, path.resolve())
    if args.runtime_root.is_relative_to(args.repo_root):
        parser.error("runtime-root must be outside the clean controller repository")
    return args


def main():
    args = arguments()
    args.runtime_root.mkdir(parents=True, exist_ok=True)
    attempt = args.runtime_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                   + "-" + uuid.uuid4().hex[:12])
    attempt.mkdir(mode=0o700)
    cancelled = threading.Event()
    signal_seen = []
    receipt = {
        "schema": SCHEMA, "status": "preparing", "wrapper_pid": os.getpid(),
        "attempt_root": str(attempt), "controller_root": str(args.repo_root),
        "scheduler_state": str(args.scheduler_state),
        "policy_sha256": args.expected_policy_sha256,
        "datasets_root": str(args.datasets_root),
        "command_sha256": digest(args.command),
        "command_executable": Path(args.command[0]).name,
        "demand": {"cpu_slots": args.cpu_slots, "memory_mib": args.memory_mib,
                   "child_process_slots": args.process_slots},
        "duration_seconds": args.duration_seconds,
        "log_path": str(attempt / "supervisor.sanitized.log"),
        "owned_group_quiescent": True, "hard_enforcement": False,
        "accounting_scope": "whole bounded CPU inventory supervisor; thread environment is advisory",
        "process_accounting": "declared native admission units; does not count or cap every supervisor, provider, collector, or validation descendant",
        "parallelism_limit": "at most three one-thread inventory jobs; provider concurrency remains separately gated by native supervisor",
        "descendant_scope": "outer owned process group plus current-generation native all_trees_fenced receipt for detached managed trees",
        "native_terminal_receipt": str(args.native_terminal_receipt),
        "native_label": args.native_label,
        "native_stop_verified": False,
        "completion_authority": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    def persist():
        receipt["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(attempt / "receipt.json", receipt)

    def on_signal(number, _frame):
        signal_seen.append(number)
        cancelled.set()

    for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, on_signal)
    persist()
    atomic_json(attempt / "pid.json", {"wrapper_pid": os.getpid(),
                                       "receipt": str(attempt / "receipt.json")})
    print(json.dumps({"runtime_attempt": str(attempt), "wrapper_pid": os.getpid()}), flush=True)
    lease = None
    child = None
    output_thread = None
    native_fence = None
    result = 70
    try:
        native, scheduler, original = pinned_scheduler(
            args.datasets_root, args.scheduler_state, args.expected_policy_sha256)
        sys.path.insert(0, str(args.repo_root))
        from ipfs_accelerate_py.agent_supervisor.proof import formal_verification_contracts as contracts
        if not Path(contracts.__file__).resolve().is_relative_to(args.repo_root):
            raise ValueError("native fence checker import escaped controller source root")
        receipt["scheduler_source_sha256"] = hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()
        receipt["validation_reserve"] = scheduler.config.reservations()["validation"].to_dict()
        queue_deadline = time.monotonic() + args.queue_timeout_seconds
        receipt["status"] = "waiting_for_shared_resources"
        persist()
        while lease is None:
            remaining = queue_deadline - time.monotonic()
            if remaining <= 0:
                receipt["status"] = "resource_queue_timeout"
                result = 75
                break
            if read_policy(args.scheduler_state)["config"] != original:
                raise ValueError("shared policy changed before admission")
            try:
                lease = scheduler.acquire(native.ResourceLane.ORCHESTRATION,
                    cpu_slots=args.cpu_slots, memory_mb=args.memory_mib,
                    child_process_slots=args.process_slots,
                    timeout=min(args.acquire_timeout_seconds, remaining),
                    cancel_event=cancelled, request_id="lac-bootstrap:" + attempt.name)
            except native.LeaseTimeoutError:
                receipt["queue_retries"] = receipt.get("queue_retries", 0) + 1
                persist()
        if lease is not None:
            if cancelled.is_set():
                raise native.LeaseCancelledError("launcher cancelled before child start")
            if not lease.renew():
                raise ValueError("shared admission was lost before child start")
            receipt.update(status="admitted", lease_id=lease.lease_id,
                           lease_wait_seconds=lease.wait_seconds)
            persist()
            child_environment = dict(os.environ)
            child_environment.update(THREAD_ENV)
            # The launch envelope stays owned here. No lease capability/key is
            # exported into child environment, argv, logs, or launch receipts.
            child_environment["IPFS_DATASETS_RESOURCE_SCHEDULER_PATH"] = str(args.scheduler_state)
            deadline = time.monotonic() + args.duration_seconds
            launched_at_ns = time.time_ns()
            child = subprocess.Popen(args.command, cwd=args.repo_root,
                env=child_environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
            native_fence = NativeFenceObserver(args.native_terminal_receipt,
                args.native_label, child, launched_at_ns, contracts.content_identity)
            receipt.update(status="running", child_pid=child.pid,
                           child_pgid=child.pid, owned_group_quiescent=False)
            atomic_json(attempt / "pid.json", {"wrapper_pid": os.getpid(),
                "child_pid": child.pid, "child_pgid": child.pid,
                "receipt": str(attempt / "receipt.json")})
            output_thread = threading.Thread(target=stream_output,
                args=(child.stdout, Path(receipt["log_path"]), receipt), daemon=True)
            output_thread.start()
            persist()
            last_receipt = time.monotonic()
            while child.poll() is None:
                native_fence.observe()
                if cancelled.is_set():
                    receipt["status"] = "stopping_for_signal"
                    result = 128 + signal_seen[-1]
                    break
                if time.monotonic() >= deadline:
                    receipt["status"] = "stopping_for_duration_limit"
                    result = 124
                    break
                if read_policy(args.scheduler_state)["config"] != original or lease.cancelled:
                    receipt["status"] = "stopping_for_lost_shared_admission"
                    result = 70
                    break
                if receipt.get("output_capture_failed"):
                    receipt["status"] = "stopping_for_log_capture_failure"
                    result = 74
                    break
                if time.monotonic() - last_receipt >= 15:
                    if not lease.renew():
                        raise ValueError("shared admission heartbeat failed")
                    persist()
                    last_receipt = time.monotonic()
                cancelled.wait(0.25)
            else:
                result = child.returncode
                receipt["status"] = "child_exited"
            receipt["child_returncode"] = child.poll()
    except Exception as error:
        # Exception messages, reprs and tracebacks can contain provider argv or
        # environment values. Retain only the exception type at this boundary.
        receipt["error_type"] = type(error).__name__
        receipt["status"] = "launch_failed"
        if cancelled.is_set():
            receipt["status"] = "cancelled"
            result = 128 + (signal_seen[-1] if signal_seen else signal.SIGTERM)
    finally:
        if child is not None:
            # A failed observer construction still cannot make group absence
            # evidence for detached descendants: retain admission and monitor.
            while native_fence is None:
                if live_group(child.pid):
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                receipt["status"] = "resource_held_native_observer_unavailable"
                persist()
                time.sleep(15)
            terminate_group(child, args.stop_grace_seconds, receipt, persist,
                            native_fence, args.native_stop_timeout_seconds)
            receipt["child_returncode"] = child.returncode
        if output_thread is not None:
            output_thread.join(timeout=5)
            if output_thread.is_alive():
                receipt["output_capture_incomplete"] = True
                result = 74
        if lease is not None:
            try:
                receipt["lease_released"] = lease.release()
            except Exception as error:
                receipt["lease_release_error_type"] = type(error).__name__
                result = 70
        try:
            receipt["policy_unchanged"] = digest(read_policy(args.scheduler_state)["config"]) == args.expected_policy_sha256
        except Exception:
            receipt["policy_unchanged"] = False
        if not receipt["policy_unchanged"]:
            result = 70
        receipt.update(returncode=result, ended_at=datetime.now(timezone.utc).isoformat())
        persist()
    print(json.dumps({"receipt": str(attempt / "receipt.json"), "returncode": result}), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
