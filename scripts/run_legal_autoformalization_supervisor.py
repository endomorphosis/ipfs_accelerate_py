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


def terminate_group(child, grace_seconds, receipt, persist):
    """Retain admission until every executing member of the owned group stops."""
    for sig, wait_seconds in ((signal.SIGTERM, grace_seconds), (signal.SIGKILL, 5)):
        if not live_group(child.pid):
            break
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline and live_group(child.pid):
            child.poll()
            time.sleep(0.1)
    # An uninterruptible kernel task cannot safely be declared stopped. Keep
    # the native lease heartbeating while acting only as its recovery monitor.
    while live_group(child.pid):
        receipt["status"] = "owned_group_reaper_pending"
        receipt["cleanup_pending"] = True
        persist()
        time.sleep(1)
    child.wait(timeout=5)
    receipt["owned_group_quiescent"] = True


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
    for name in ("datasets_root", "repo_root", "runtime_root", "scheduler_state"):
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
        "descendant_scope": "owned session process group; daemonizing children are forbidden",
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
    result = 70
    try:
        native, scheduler, original = pinned_scheduler(
            args.datasets_root, args.scheduler_state, args.expected_policy_sha256)
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
            child = subprocess.Popen(args.command, cwd=args.repo_root,
                env=child_environment, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
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
            terminate_group(child, args.stop_grace_seconds, receipt, persist)
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
