"""Observe one existing watchdog; optionally replace a genuinely absent owner.

This stdlib-only outer guardian never signals processes, repairs application
code, changes boards, or grants completion. ``--once`` is always observational.
The manifest directory contains its flock, atomic telemetry, and stop markers.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

WATCHDOG_MODULE = "ipfs_accelerate_py.agent_supervisor.rescue.supervisor_watchdog"
STOP_MARKERS = ("HOLD", "OPERATOR_STOP", "watchdog.disabled")


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write(path, json.dumps(payload, indent=2) + "\n")


def acquire_lock(path: Path):
    """The open descriptor owns the lock; never unlink a possibly live lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def _stat_identity(directory: Path) -> tuple[str, str]:
    fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
    return fields[0], fields[19]  # state and kernel start time fence PID reuse


def _option(argv: list[str], name: str) -> str | None:
    values = []
    for index, value in enumerate(argv):
        if value == name:
            values.append(argv[index + 1] if index + 1 < len(argv) else "")
        elif value.startswith(name + "="):
            values.append(value[len(name) + 1 :])
    return values[0] if len(values) == 1 and values[0] else None


def _python_entrypoint(argv: list[str]) -> tuple[str, str, int]:
    """Read Python's entrypoint before its script/module application arguments.

    In particular, a monitor's evidence path is data, even when its value is
    the watchdog source file. Options that consume Python configuration must
    not turn their operands into script candidates either.
    """

    index = 1
    while index < len(argv):
        argument = argv[index]
        if argument == "--":
            return ("script", argv[index + 1], index + 1) if index + 1 < len(argv) else ("", "", -1)
        if argument == "-":
            return "stdin", "", index
        if not argument.startswith("-"):
            return "script", argument, index
        if argument == "--check-hash-based-pycs":
            index += 2
            continue
        if argument.startswith("--"):
            index += 1
            continue
        # Python permits grouped short flags, including -um module and
        # -uW warning-filter. Only m/c select an entrypoint; W/X consume their
        # remaining group text or the next operand as interpreter settings.
        for offset, flag in enumerate(argument[1:], 2):
            if flag in {"m", "c"}:
                value = argument[offset:] or (argv[index + 1] if index + 1 < len(argv) else "")
                return ("module" if flag == "m" else "command"), value, index
            if flag in {"W", "X"}:
                index += 1 if argument[offset:] else 2
                break
        else:
            index += 1
    return "", "", -1


def discover_watchdogs(
    manifest: Path, repo: Path, *, proc_root: Path = Path("/proc")
) -> tuple[list[dict[str, Any]], list[str]]:
    """Match process ownership and invocation, never just a PID file or substring."""
    owners, errors = [], []
    try:
        directories = list(proc_root.iterdir())
    except OSError as exc:
        return [], [f"proc_unavailable:{exc.errno}"]
    for directory in directories:
        if not directory.name.isdecimal():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            before = _stat_identity(directory)
            if before[0] in {"Z", "X"}:
                continue
            argv = (directory / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
            python_command = re.fullmatch(r"(?:python|pypy)(?:\d+(?:\.\d+)*)?", Path(argv[0]).name)
            mode, entrypoint, module_index = _python_entrypoint(argv) if python_command else ("script", argv[0], 0)
            if mode != "module" or entrypoint != WATCHDOG_MODULE:
                # Direct-file invocation is not an adoptable module identity;
                # its presence still prevents launching a duplicate owner.
                if mode == "script" and Path(entrypoint).name == "supervisor_watchdog.py":
                    cwd = (directory / "cwd").resolve(strict=True)
                    suffix = Path(*WATCHDOG_MODULE.split(".")).with_suffix(".py")
                    expected = {(repo / suffix).resolve(), (repo / "external/ipfs_accelerate" / suffix).resolve()}
                    if (cwd / entrypoint).resolve() in expected:
                        errors.append(f"invocation_unverified:{directory.name}")
                continue
            uids = next(line.split()[1:] for line in (directory / "status").read_text().splitlines() if line.startswith("Uid:"))
            if not uids or any(int(uid) != os.getuid() for uid in uids):
                continue
            cwd = (directory / "cwd").resolve(strict=True)
            manifest_arg, repo_arg = _option(argv, "--manifest-path"), _option(argv, "--repo-root")
            repo_option_present = any(value == "--repo-root" or value.startswith("--repo-root=") for value in argv)
            if repo_arg is None and not repo_option_present:
                repo_arg = str(cwd)  # supervisor_watchdog defaults --repo-root to cwd
            if not manifest_arg or not repo_arg:
                errors.append(f"identity_options_invalid:{directory.name}")
                continue
            if (cwd / manifest_arg).resolve() != manifest or (cwd / repo_arg).resolve() != repo:
                continue
            if (not python_command or argv[module_index] != "-m"
                or any(flag not in {"-u", "-B", "-E", "-s", "-S", "-I", "-O", "-OO"}
                       for flag in argv[1:module_index])):
                errors.append(f"invocation_unverified:{directory.name}")
                continue
            after = _stat_identity(directory)
            if before[1] != after[1] or after[0] in {"Z", "X"}:
                errors.append(f"identity_changed:{directory.name}")
                continue
            owners.append({"pid": int(directory.name), "start_ticks": after[1], "state": after[0]})
        except FileNotFoundError:  # normal exit during the snapshot
            continue
        except (OSError, UnicodeError, ValueError, IndexError, StopIteration) as exc:
            errors.append(f"identity_unreadable:{directory.name}:{type(exc).__name__}")
    return sorted(owners, key=lambda item: item["pid"]), errors


class WatchdogGuardian:
    def __init__(self, manifest: Path, repo: Path, *, python: str = sys.executable,
                 apply: bool = False, poll_seconds: float = 30,
                 restart_cooldown: float = 120, proc_root: Path = Path("/proc")):
        self.repo, self.manifest = repo.resolve(), manifest.resolve()
        self.python = str(Path(python).expanduser().absolute())  # retain venv symlink
        self.apply, self.poll_seconds = apply, max(1.0, poll_seconds)
        self.cooldown, self.proc_root = max(1.0, restart_cooldown), proc_root
        self.status_path = self.manifest.parent / "watchdog-guardian-status.json"
        self.lock_path = self.manifest.parent / "watchdog-guardian.lock"
        previous = read_json(self.status_path)
        try:
            self.next_launch_after = float(previous.get("next_launch_after") or 0)
            self.launch_attempts = max(0, int(previous.get("launch_attempts_since_healthy") or 0))
        except (ValueError, TypeError):
            self.next_launch_after, self.launch_attempts = 0.0, 0
        self.children: list[subprocess.Popen] = []

    def _path(self, value: str) -> Path:
        return (self.repo / value).resolve()

    def _lanes(self, manifest: dict[str, Any], now: float) -> list[dict[str, Any]]:
        lanes = []
        for lane in manifest.get("lanes", []):
            if not isinstance(lane, dict):
                continue
            directory = self._path(lane.get("state_dir") or ".")
            status = read_json(directory / f"{lane.get('state_prefix', '')}_status.json")
            log = self._path(lane["log_path"]) if lane.get("log_path") else directory / "supervisor.log"
            item = {key: status.get(key) for key in ("pid", "state", "phase", "active_task_id", "last_task_transition_at_ms")}
            item["bundle_key"] = lane.get("bundle_key")
            stamp = status.get("heartbeat_at_ms")
            try:
                item["heartbeat_age_seconds"] = max(0, now - float(stamp) / 1000) if stamp else None
            except (ValueError, TypeError):
                item["heartbeat_age_seconds"] = None
            try:
                metadata = log.stat()
                item.update(log_age_seconds=max(0, now - metadata.st_mtime), log_size_bytes=metadata.st_size)
                with log.open("rb") as handle:
                    handle.seek(max(0, metadata.st_size - 8192))
                    rows = re.findall(rb"live-click (\d+) ", handle.read(8192))
                item["last_logged_control_index"] = int(rows[-1]) if rows else None
            except OSError:
                item.update(log_age_seconds=None, log_size_bytes=None, last_logged_control_index=None)
            lanes.append(item)
        return lanes

    def inspect(self, *, now: float | None = None, once: bool = False) -> dict[str, Any]:
        now = time.time() if now is None else now
        self.children = [child for child in self.children if child.poll() is None]
        manifest = read_json(self.manifest)
        valid_manifest = bool(manifest) and isinstance(manifest.get("lanes"), list) and all(
            isinstance(lane, dict) and all(isinstance(lane.get(key, ""), str)
                                          for key in ("state_dir", "state_prefix", "log_path"))
            for lane in manifest.get("lanes", [])) and isinstance(manifest.get("report_path", ""), str)
        if not valid_manifest:
            manifest = {}
        owners, errors = discover_watchdogs(self.manifest, self.repo, proc_root=self.proc_root)
        markers = sorted(str(directory / name) for directory in {self.repo, self.manifest.parent}
                         for name in STOP_MARKERS if (directory / name).exists())
        report_path = self._path(manifest.get("report_path") or str(self.manifest.parent / "watchdog-report.json"))
        report = read_json(report_path)
        report_status = report.get("status") if isinstance(report.get("status"), dict) else {}
        try:
            report_age = max(0, now - report_path.stat().st_mtime)
        except OSError:
            report_age = None
        owner = owners[0] if len(owners) == 1 else None
        health = "missing" if not report else "stale" if report_age is None or report_age > 600 else "fresh"
        if owner and report and report_status.get("pid") != owner["pid"]:
            health = "different_owner"
        if owner and health == "fresh":
            self.launch_attempts = 0
        if owner and self.apply and not once and not markers:
            pid_path = self.manifest.parent / "watchdog.pid"
            try:
                previous_pid = pid_path.read_text().strip()
            except OSError:
                previous_pid = ""
            if previous_pid != str(owner["pid"]):
                atomic_write(pid_path, f"{owner['pid']}\n")
        action = "held" if markers else "duplicates_detected" if len(owners) > 1 else "owner_present" if owner else "owner_absent"
        if owner and not markers and health != "fresh":
            action = "owner_unresponsive"
        if not valid_manifest:
            action = "configuration_error"
        elif errors and not owners and not markers:
            action = "discovery_uncertain"
        elif action == "owner_absent":
            if once or not self.apply:
                action = "would_launch"
            elif now < self.next_launch_after:
                action = "restart_cooldown"
            else:
                # Repeat ownership and hold checks immediately before the only mutation.
                owners, errors = discover_watchdogs(self.manifest, self.repo, proc_root=self.proc_root)
                if owners or errors or any((directory / name).exists() for directory in {self.repo, self.manifest.parent} for name in STOP_MARKERS):
                    action = "launch_deferred"
                else:
                    self.launch_attempts += 1
                    self.next_launch_after = now + min(3600, self.cooldown * 2 ** min(self.launch_attempts - 1, 6))
                    # Persist the cooldown before spawning, including across guardian restarts.
                    atomic_json(self.status_path, {"next_launch_after": self.next_launch_after,
                                "launch_attempts_since_healthy": self.launch_attempts,
                                "work_complete": False, "completion_authority": False})
                    try:
                        self._launch()
                        action = "launched"
                    except OSError as exc:
                        action = "launch_failed"
                        errors.append(f"spawn_failed:{type(exc).__name__}:{exc.errno}")
        result = {"schema": "ipfs_accelerate_py.agent_supervisor.watchdog-guardian@1",
                  "guardian_pid": os.getpid(), "observed_at_ms": int(now * 1000),
                  "manifest_path": str(self.manifest), "repo_root": str(self.repo),
                  "apply_enabled": self.apply and not once, "action": action,
                  "owner_pid": owners[0]["pid"] if len(owners) == 1 else None,
                  "owner_alive": bool(owners), "owners": owners, "errors": errors,
                  "hold_markers": markers, "report_age_seconds": report_age,
                  "report_health": health, "reported_state": report_status.get("state"),
                  "lanes": self._lanes(manifest, now) if isinstance(manifest.get("lanes"), list) else [],
                  "next_launch_after": self.next_launch_after,
                  "launch_attempts_since_healthy": self.launch_attempts,
                  "work_complete": False, "completion_authority": False}
        atomic_json(self.status_path, result)
        return result

    def _launch(self) -> None:
        source = self.repo / "external" / "ipfs_accelerate"
        if not (source / "ipfs_accelerate_py").is_dir():
            source = self.repo
        if not (source / Path(*WATCHDOG_MODULE.split("."))).with_suffix(".py").is_file():
            raise FileNotFoundError("watchdog module is missing from the requested checkout")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(source) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        with (self.manifest.parent / "guardian-watchdog.log").open("ab") as handle:
            self.children.append(subprocess.Popen(
                [self.python, "-m", WATCHDOG_MODULE, "--manifest-path", str(self.manifest),
                 "--repo-root", str(self.repo), "--check-interval", "120", "--lane-timeout", "600",
                 "--max-consecutive-restarts", "3"], cwd=self.repo, env=env,
                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--restart-cooldown", type=float, default=120)
    parser.add_argument("--apply", action="store_true", help="Allow launching an absent watchdog")
    parser.add_argument("--once", action="store_true", help="Observe once; never launch")
    args = parser.parse_args(argv)
    guardian = WatchdogGuardian(args.manifest_path, args.repo_root, python=args.python,
                                apply=args.apply, poll_seconds=args.poll_seconds,
                                restart_cooldown=args.restart_cooldown)
    lock = acquire_lock(guardian.lock_path)
    if lock is None:
        print(json.dumps({"action": "guardian_already_running", "work_complete": False,
                          "completion_authority": False}))
        return 0
    with lock:
        while True:
            result = guardian.inspect(once=args.once)
            print(json.dumps(result), flush=True)
            if args.once:
                return 0
            time.sleep(guardian.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
