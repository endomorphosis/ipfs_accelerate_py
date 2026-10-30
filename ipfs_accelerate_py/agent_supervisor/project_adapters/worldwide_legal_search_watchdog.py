"""File-protocol bridge to the single-owner worldwide legal-search runtime.

This module deliberately never opens a task database. The native owner emits
atomic status/readiness projections and consumes immutable, idempotent control
requests. Missing mutation bindings remain false in readiness; observing a
healthy process cannot create repair or completion authority.

WATCHDOG.md observe, dispatch and verify run over the existing native
authorities: ``lifecycle_restart`` through ``restart_lane``, rescue
planner/orchestrator request factories, current-tree completion proof,
process-birth fencing and a durable incident claim.

A published recovery also seals the loop's charged attempts, cooldown and
stop/hold markers. Restarting the loop and the supervisor reloads that seal
and refuses a refund. The optional user-systemd guardian may start only the
registered watchdog loop, and only when the hold markers are absent and the
restart bound still has room.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping

MAX_BYTES = 1_048_576
ACTION_ID = re.compile(r"[0-9a-f]{64}")
INCIDENT_ID = re.compile(r"[0-9A-Za-z][0-9A-Za-z:._-]{0,127}")
COMPLETION_PROOF_FIELDS = (
    "schema", "run_id", "accepted_tree", "required_task_ids", "verified_task_ids",
    "runnable_count", "task_id", "effects_unknown", "hold",
)
HOLD_MARKERS = ("HOLD", "STOP", "OPERATOR_STOP")
LOOP_UNIT = "worldwide-legal-search-watchdog.service"
SUPERVISOR_UNIT = "worldwide-legal-search-supervisor.service"
LOOP_MODULE = "ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_watchdog"
SUPERVISOR_MODULE = "ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_runtime"
ATTEMPT_KEYS = ("reconcile", "recover", "diagnose", "repair")
GUARDIAN_RESTART_BOUND = 3
GUARDIAN_WINDOW_SECONDS = 1800
GUARDIAN_RESTART_SECONDS = 30
CANONICAL_PYTHON = "/usr/bin/python3.12"


class WatchdogBridgeError(ValueError):
    """Projection or request does not match this admitted campaign."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def identity(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def native_identity(value: Any) -> str:
    """Match the native runtime's UTF-8 receipt encoding."""
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def resource_observation(*, repository: Path, state_root: Path, framework_root: Path | None = None,
                         config: Mapping[str, Any] | None = None, statvfs=None) -> dict[str, Any]:
    """Read actual storage admission evidence without modifying any file."""
    bounds = (config or {}).get("resource_pressure", {})
    if not isinstance(bounds, Mapping):
        raise WatchdogBridgeError("resource pressure thresholds must be an object")
    state_min = bounds.get("minimum_state_free_bytes", 64 * 1024**2)
    repository_min = bounds.get("minimum_repository_free_bytes", 256 * 1024**2)
    if any(type(value) is not int or not 0 < value <= 1024**4 for value in (state_min, repository_min)):
        raise WatchdogBridgeError("resource pressure thresholds must be finite positive byte counts")
    probe = statvfs or os.statvfs
    roots = [("state", state_root, state_min), ("site", repository, repository_min)]
    if framework_root is not None:
        roots.append(("framework", framework_root, repository_min))
    evidence = []
    for role, path, minimum in roots:
        item = {"role": role, "path": str(path), "minimum_free_bytes": minimum}
        try:
            usage = probe(path)
            free = max(0, int(usage.f_bavail)) * int(usage.f_frsize)
            item.update(free_bytes=free, reason="sufficient" if free >= minimum else "insufficient_free_space")
        except OSError as error:
            item.update(free_bytes=None, reason="statvfs_unavailable", error_errno=error.errno)
        evidence.append(item)
    return {"schema": "wwl.resource-observation@1", "blocked": any(item["reason"] != "sufficient" for item in evidence), "paths": evidence}


def read_owned_json(path: Path) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or before.st_nlink != 1 or before.st_size > MAX_BYTES:
            raise WatchdogBridgeError("projection must be a bounded owned regular file")
        raw = source.read(MAX_BYTES + 1)
        after = os.fstat(source.fileno())
        if len(raw) > MAX_BYTES or (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise WatchdogBridgeError("projection changed while reading")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise WatchdogBridgeError("projection must be an object")
    return value


def publish_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a complete immutable request, refusing identity reuse."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.resolve() != path.parent or path.parent.stat().st_uid != os.getuid():
        raise WatchdogBridgeError("control directory ownership or path changed")
    try:
        current = read_owned_json(path)
    except FileNotFoundError:
        current = None
    if current is not None:
        if canonical(current) != canonical(payload):
            raise WatchdogBridgeError("action id reused with a different request")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".control-request-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(canonical(payload) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        try:
            # link is an atomic publish without replacing another submission.
            os.link(temporary, path)
        except FileExistsError:
            if canonical(read_owned_json(path)) != canonical(payload):
                raise WatchdogBridgeError("concurrent action id conflict")
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def hold_markers(state_root: Path) -> list[str]:
    """Filesystem pause markers. A stale projection cannot hide one."""
    return [name for name in HOLD_MARKERS if os.path.lexists(state_root / name)]


def _systemd_token(value: str) -> str:
    if not isinstance(value, str) or not value or any(character.isspace() or character in "\"'\\$" for character in value):
        raise WatchdogBridgeError("registered loop path must be one systemd argument")
    return value


def _systemd_exec_arg(value: str) -> str:
    """Quote one ExecStart argument. Plain paths stay bare; JSON uses single quotes."""
    if not isinstance(value, str) or value == "" or "\n" in value or "\r" in value or "\x00" in value:
        raise WatchdogBridgeError("registered loop path must be one systemd argument")
    if not any(character.isspace() or character in "\"'\\$" for character in value):
        return value
    if "'" in value or "\\" in value or "$" in value:
        raise WatchdogBridgeError("registered loop path must be one systemd argument")
    return "'" + value + "'"


def empty_restart_journal(run_id: str) -> dict[str, Any]:
    return {
        "schema": "wwl.watchdog.restart-journal@1", "run_id": run_id, "actions": {},
        "sealed_hold_markers": [], "required_task_ids": [], "loop_policy_digest": "",
        "guardian_restarts": [], "completion_authority": False,
    }


def _journal_file(state_root: Path) -> Path:
    return state_root / "watchdog" / "restart-journal.json"


def atomic_replace(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace one owned JSON object. Callers hold the journal lock."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.parent.stat().st_uid != os.getuid():
        raise WatchdogBridgeError("restart journal directory is not owned")
    raw = canonical(payload) + b"\n"
    if len(raw) > MAX_BYTES:
        raise WatchdogBridgeError("restart journal exceeds its bound")
    descriptor, temporary = tempfile.mkstemp(prefix=".restart-journal-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def journal_lock(state_root: Path):
    directory = state_root / "watchdog"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(directory / "restart-journal.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        if os.fstat(descriptor).st_uid != os.getuid():
            raise WatchdogBridgeError("restart journal lock is not owned")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _finite_number(value: Any, *, minimum: float = 0) -> float:
    if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise WatchdogBridgeError("restart accounting must be a finite number")
    return float(value)


def _attempt_map(value: Mapping[str, Any]) -> dict[str, int]:
    attempts = {}
    for key in ATTEMPT_KEYS:
        raw = value.get(key, 0)
        if isinstance(raw, bool) or type(raw) is not int or raw < 0:
            raise WatchdogBridgeError("invalid durable incident accounting")
        attempts[key] = raw
    return attempts


def load_restart_journal(state_root: Path, run_id: str) -> dict[str, Any]:
    path = _journal_file(state_root)
    try:
        value = read_owned_json(path)
    except FileNotFoundError:
        return empty_restart_journal(run_id)
    if (value.get("schema") != "wwl.watchdog.restart-journal@1" or value.get("run_id") != run_id
            or value.get("completion_authority") is not False or not isinstance(value.get("actions"), dict)
            or not isinstance(value.get("guardian_restarts"), list)
            or not isinstance(value.get("sealed_hold_markers"), list)
            or not isinstance(value.get("required_task_ids"), list)):
        raise WatchdogBridgeError("restart journal run differs")
    for stamp in value["guardian_restarts"]:
        _finite_number(stamp)
    for name in value["sealed_hold_markers"]:
        if name not in HOLD_MARKERS:
            raise WatchdogBridgeError("restart journal names an unknown hold marker")
    return value


def update_restart_journal(state_root: Path, run_id: str, mutator: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    with journal_lock(state_root):
        journal = load_restart_journal(state_root, run_id)
        mutator(journal)
        if journal.get("schema") != "wwl.watchdog.restart-journal@1" or journal.get("run_id") != run_id or journal.get("completion_authority") is not False:
            raise WatchdogBridgeError("restart journal lost its run scope")
        atomic_replace(_journal_file(state_root), journal)
        stored = read_owned_json(_journal_file(state_root))
        if canonical(stored) != canonical(journal):
            raise WatchdogBridgeError("restart journal changed while it was sealed")
        return stored


def _action_places(loop_state: Mapping[str, Any], action_id: str) -> list[Mapping[str, Any]]:
    places: list[Mapping[str, Any]] = []
    outbox = loop_state.get("outbox")
    if isinstance(outbox, Mapping):
        places.append(outbox)
    for bucket in ("inflight", "uncertain_actions"):
        found = loop_state.get(bucket, {})
        if isinstance(found, Mapping):
            item = found.get(action_id)
            if isinstance(item, Mapping):
                places.append(item)
    return places


def _covers_action(stored: Mapping[str, Any], action: Mapping[str, Any]) -> bool:
    return all(stored.get(key) == value for key, value in action.items())


def _require_loop_state(loop_state: Mapping[str, Any], action: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    if loop_state.get("schema") != "wwl.watchdog.state@1" or loop_state.get("run_id") != run_id:
        raise WatchdogBridgeError("restarted loop state run differs")
    if not isinstance(loop_state.get("policy_digest"), str) or not loop_state["policy_digest"]:
        raise WatchdogBridgeError("restarted loop policy digest differs")
    if not isinstance(loop_state.get("incidents"), dict) or not isinstance(loop_state.get("required_task_ids"), list):
        raise WatchdogBridgeError("restarted loop state is missing its incident ledger")
    for key in ("actions", "repairs"):
        if isinstance(loop_state.get(key), bool) or type(loop_state.get(key)) is not int or loop_state[key] < 0:
            raise WatchdogBridgeError("invalid durable budget counter")
    incident_id = action.get("incident_id")
    incident = loop_state["incidents"].get(incident_id) if isinstance(incident_id, str) else None
    if not isinstance(incident, Mapping):
        raise WatchdogBridgeError("restarted loop lost the in-flight incident")
    _attempt_map(incident)
    _finite_number(incident.get("next_action_at"))
    if not any(_covers_action(item, action) for item in _action_places(loop_state, action["action_id"])):
        raise WatchdogBridgeError("restarted loop lost the in-flight recovery")
    return dict(incident)


def _accounting_from_loop(loop_state: Mapping[str, Any], incident: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "attempts": _attempt_map(incident),
        "cooldown_until": _finite_number(incident.get("next_action_at")),
        "campaign_actions": loop_state["actions"],
        "campaign_repairs": loop_state["repairs"],
    }


def _require_accounting(loop_state: Mapping[str, Any], incident: Mapping[str, Any], record: Mapping[str, Any]) -> None:
    """A restarted ledger may charge more later. It may not refund or shorten a wait."""
    if record.get("attempts") is None:
        return
    current = _attempt_map(incident)
    sealed = _attempt_map(record["attempts"])
    if any(current[key] < sealed[key] for key in ATTEMPT_KEYS):
        raise WatchdogBridgeError("attempt accounting regressed")
    if loop_state["actions"] < record["campaign_actions"] or loop_state["repairs"] < record["campaign_repairs"]:
        raise WatchdogBridgeError("attempt accounting regressed")
    if _finite_number(incident.get("next_action_at")) < _finite_number(record["cooldown_until"]):
        raise WatchdogBridgeError("cooldown moved earlier")


class OptionalSystemdLoopGuardian:
    """Revive only the registered watchdog loop. This does not install the unit.

    Start is refused while a hold marker exists, after the bounded restart
    window is spent, or when the unit fragment is any other command.
    """

    def __init__(self, *, repository: Path, state_root: Path, run_id: str, loop_script: Path, command: Callable[..., str] | None = None, clock=time.time, show: Callable[[], Mapping[str, str]] | None = None, fragment_path: Path | None = None, python: str = CANONICAL_PYTHON, max_restarts: int = GUARDIAN_RESTART_BOUND, window_seconds: float = GUARDIAN_WINDOW_SECONDS):
        self.repository = repository.expanduser().resolve(strict=True)
        self.state_root = state_root.expanduser().resolve(strict=True)
        self.loop_script = Path(loop_script)
        if not run_id or self.loop_script.name != "watchdog_loop.py":
            raise WatchdogBridgeError("guardian may revive only the registered watchdog loop")
        if isinstance(max_restarts, bool) or type(max_restarts) is not int or not 1 <= max_restarts <= 8:
            raise WatchdogBridgeError("guardian restart bound is outside the campaign limit")
        self.window_seconds = _finite_number(window_seconds, minimum=1)
        self.max_restarts = max_restarts
        self.run_id = run_id
        self.clock = clock
        self.command = command
        self.show = show
        self.python = _systemd_token(python)
        self.state_path = self.state_root / "watchdog" / "state.json"
        self.fragment_path = fragment_path or (Path.home() / ".config" / "systemd" / "user" / LOOP_UNIT)
        self.adapter_argv = [
            self.python, "-m", LOOP_MODULE, "--repository", _systemd_token(str(self.repository)),
            "--state-root", _systemd_token(str(self.state_root)), "--run-id", _systemd_token(run_id),
        ]
        self.loop_argv = [
            self.python, _systemd_token(str(self.loop_script)),
            "--adapter-argv-json", json.dumps(self.adapter_argv, separators=(",", ":")),
            "--state", _systemd_token(str(self.state_path)), "--execute",
        ]

    def render_unit(self) -> str:
        pre = " ".join(_systemd_exec_arg(part) for part in [*self.adapter_argv, "guardian-start-check"])
        start = " ".join(_systemd_exec_arg(part) for part in self.loop_argv)
        if SUPERVISOR_MODULE in start or SUPERVISOR_UNIT in start or start.count("watchdog_loop.py") != 1:
            raise WatchdogBridgeError("guardian unit would start something other than this loop")
        lines = [
            "[Unit]",
            "Description=Optional worldwide legal-search watchdog loop",
            "Documentation=docs/worldwide/host-operation.md",
            "StartLimitIntervalSec=" + str(int(self.window_seconds)),
            "StartLimitBurst=" + str(self.max_restarts),
            "",
            "[Service]",
            "Type=simple",
            "Restart=on-failure",
            "RestartSec=" + str(GUARDIAN_RESTART_SECONDS),
            "KillMode=control-group",
            "ExecStartPre=" + pre,
            "ExecStart=" + start,
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
        return "\n".join(lines)

    def _run(self, argv: list[str], timeout: float) -> str:
        if self.command is not None:
            produced = self.command(argv, timeout)
            return produced if isinstance(produced, str) else ""
        try:
            result = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=max(0.1, timeout), check=False)
        except subprocess.TimeoutExpired as error:
            raise WatchdogBridgeError("registered watchdog guardian command exceeded its bound") from error
        if result.returncode:
            raise WatchdogBridgeError("registered watchdog guardian command failed: " + str(result.returncode))
        return result.stdout

    def _show_unit(self) -> dict[str, str]:
        if self.show is not None:
            shown = self.show()
            if not isinstance(shown, Mapping):
                raise WatchdogBridgeError("systemd show must be an object")
            return {str(key): str(value) for key, value in shown.items()}
        raw = self._run(["systemctl", "--user", "show", LOOP_UNIT, "--property=Id,FragmentPath,ActiveState,MainPID"], 5)
        return dict(line.split("=", 1) for line in raw.splitlines() if "=" in line)

    def _restarts_in_window(self, journal: Mapping[str, Any] | None = None) -> list[float]:
        source = journal if journal is not None else load_restart_journal(self.state_root, self.run_id)
        now = float(self.clock())
        return [float(stamp) for stamp in source.get("guardian_restarts", []) if now - self.window_seconds < float(stamp) <= now]

    def start_check(self) -> dict[str, Any]:
        """Hold-aware admission for this unit only. This method starts nothing."""
        markers = hold_markers(self.state_root)
        report = {
            "schema": "wwl.watchdog.guardian-start@1", "unit": LOOP_UNIT, "run_id": self.run_id,
            "hold_markers": markers, "started_units": [], "completion_authority": False,
            "restart_bound": self.max_restarts, "window_seconds": self.window_seconds,
        }
        if markers:
            return {**report, "may_start": False, "reason": "hold_blocks_start"}
        try:
            used = self._restarts_in_window()
        except (OSError, ValueError) as error:
            return {**report, "may_start": False, "reason": "restart_journal_unreadable", "error": str(error)}
        if len(used) >= self.max_restarts:
            return {**report, "may_start": False, "reason": "restart_bound_exhausted", "restarts_in_window": len(used)}
        try:
            shown = self._show_unit()
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            return {**report, "may_start": False, "reason": "systemd_show_failed", "error": str(error), "restarts_in_window": len(used)}
        if shown.get("Id") != LOOP_UNIT:
            return {**report, "may_start": False, "reason": "unregistered_unit", "restarts_in_window": len(used)}
        fragment = Path(shown.get("FragmentPath") or "")
        expected = self.render_unit()
        starts = [line for line in expected.splitlines() if line.startswith("ExecStart=")]
        if (fragment != Path(self.fragment_path) or not fragment.is_file() or fragment.is_symlink()
                or fragment.stat().st_uid != os.getuid()):
            return {**report, "may_start": False, "reason": "fragment_not_registered", "restarts_in_window": len(used)}
        try:
            text = fragment.read_text(encoding="utf-8")
        except OSError as error:
            return {**report, "may_start": False, "reason": "fragment_not_registered", "error": str(error), "restarts_in_window": len(used)}
        observed_starts = [line for line in text.splitlines() if line.startswith("ExecStart=")]
        if text != expected or len(starts) != 1 or observed_starts != starts or SUPERVISOR_MODULE in text or SUPERVISOR_UNIT in text:
            return {**report, "may_start": False, "reason": "fragment_does_not_match_registered_loop", "restarts_in_window": len(used)}
        if shown.get("ActiveState") == "active" and shown.get("MainPID") not in {None, "", "0"}:
            return {**report, "may_start": False, "reason": "registered_loop_already_running", "restarts_in_window": len(used)}
        return {**report, "may_start": True, "reason": "registered_loop_start_allowed", "restarts_in_window": len(used)}

    def revive(self) -> dict[str, Any]:
        """Start ``worldwide-legal-search-watchdog.service`` when the start check allows it."""
        check = self.start_check()
        if check.get("may_start") is not True:
            return {**check, "revived": False}
        charged = False

        def mutate(journal: dict[str, Any]) -> None:
            nonlocal charged
            now = float(self.clock())
            used = [float(stamp) for stamp in journal["guardian_restarts"] if now - self.window_seconds < float(stamp) <= now]
            if len(used) >= self.max_restarts:
                raise WatchdogBridgeError("restart_bound_exhausted")
            journal["guardian_restarts"].append(now)
            charged = True

        try:
            update_restart_journal(self.state_root, self.run_id, mutate)
        except WatchdogBridgeError as error:
            return {**check, "may_start": False, "revived": False, "reason": str(error)}
        try:
            self._run(["systemctl", "--user", "start", LOOP_UNIT], 5)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            # The charge stays. A failed start is not a refund and is not another unit.
            return {**check, "revived": False, "reason": "registered_loop_start_failed", "restart_charged": charged, "error": str(error), "started_units": []}
        return {**check, "revived": True, "reason": "registered_loop_revived", "restart_charged": charged, "started_units": [LOOP_UNIT]}


def _mapping_result(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        dumped = to_dict()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    raise WatchdogBridgeError("native authority returned a non-object")


def _diagnosis_parts(diagnosis: Any) -> tuple[Any, str, list[Any], dict[str, Any], tuple[Any, ...]]:
    if isinstance(diagnosis, Mapping):
        kind = diagnosis.get("incident_kind", "")
        reasons = diagnosis.get("reason_codes", ())
        health = diagnosis.get("health", {})
        evidence = diagnosis.get("evidence_reference_cids", ())
        incident = diagnosis.get("incident")
    else:
        kind_value = getattr(diagnosis, "kind", "")
        kind = getattr(kind_value, "value", kind_value)
        reasons = getattr(diagnosis, "reason_codes", ())
        health = getattr(diagnosis, "health", {})
        incident = getattr(diagnosis, "incident", None)
        evidence = getattr(incident, "evidence_cids", ()) if incident is not None else ()
    if not isinstance(health, Mapping):
        raise WatchdogBridgeError("rescue diagnosis health must be an object")
    return incident, str(kind or ""), list(reasons or ()), dict(health), tuple(evidence or ())


def default_rescue_request_factory(diagnosis: Any, exhaustion: Any, roots: Mapping[str, str]):
    """Build the native planning request. This factory never calls a provider."""
    from ..rescue.rescue_planner import RescuePlanningRequest

    if not isinstance(roots, Mapping):
        raise WatchdogBridgeError("rescue request roots must be an object")
    for key in ("repository_root_cid", "run_cid", "policy_root"):
        if not isinstance(roots.get(key), str) or not roots[key]:
            raise WatchdogBridgeError("rescue request is missing " + key)
    incident, kind, reasons, health, evidence = _diagnosis_parts(diagnosis)
    if incident is None:
        raise WatchdogBridgeError("rescue request is missing its incident")
    return RescuePlanningRequest(
        incident=incident,
        exhaustion_receipt=exhaustion,
        diagnostics={"incident_kind": kind, "reason_codes": reasons, "health": health},
        evidence_redacted=True,
        current_repository_root_cid=roots["repository_root_cid"],
        current_run_cid=roots["run_cid"],
        current_policy_root=roots["policy_root"],
        evidence_reference_cids=evidence,
        now_ms=int(time.time() * 1000),
    )


def default_rescue_execution_request_factory(diagnosis: Any, exhaustion: Any, plan: Any, roots: Mapping[str, Any]):
    """Build a native execution request when its types are already bound.

    Callers that have not crossed into rescue-orchestrator types receive a
    content-addressed campaign request. Neither form executes effects.
    """
    from ..rescue.rescue_orchestrator import RescueExecutionRequest, RescuePlan, RescueRootBinding

    incident, _, _, _, _ = _diagnosis_parts(diagnosis)
    if not isinstance(plan, RescuePlan):
        roots = roots.to_dict() if isinstance(roots, RescueRootBinding) else roots
    elif isinstance(roots, RescueRootBinding) and incident is not None:
        return RescueExecutionRequest(
            plan=plan,
            incident=incident,
            exhaustion_receipt=exhaustion,
            roots=roots,
            lease_id=str(getattr(diagnosis, "lease_id", "") or "wwl-rescue"),
            fencing_epoch=int(getattr(diagnosis, "fencing_epoch", 0) or 0),
            idempotency_scope=str(getattr(incident, "incident_cid", "") or identity(roots.to_dict())),
            caller="worldwide-legal-search-watchdog",
        )
    if not isinstance(roots, Mapping):
        raise WatchdogBridgeError("rescue execution roots must be an object")
    root_fields = (
        "incident_cid", "exhaustion_receipt_cid", "request_root", "program_root", "repository_root_cid",
        "tree_id", "run_cid", "intent_ir_root", "legal_ir_root", "security_ir_root", "policy_root", "catalog_root",
    )
    if isinstance(plan, RescuePlan) and incident is not None and all(isinstance(roots.get(key), str) and roots[key] for key in root_fields):
        return RescueExecutionRequest(
            plan=plan,
            incident=incident,
            exhaustion_receipt=exhaustion,
            roots=RescueRootBinding(**{key: roots[key] for key in root_fields}),
            lease_id=str(roots.get("lease_id") or "wwl-rescue"),
            fencing_epoch=int(roots.get("fencing_epoch") or 0),
            idempotency_scope=str(getattr(incident, "incident_cid", "") or roots["incident_cid"]),
            caller="worldwide-legal-search-watchdog",
        )
    if not isinstance(plan, (str, bytes)):
        try:
            plan_body: Any = _mapping_result(plan)
        except WatchdogBridgeError:
            plan_body = {"plan_cid": str(getattr(plan, "plan_cid", "") or "")}
    else:
        plan_body = {"plan_cid": str(plan)}
    body = {
        "schema": "wwl.watchdog.rescue-execution-request@1",
        "incident_cid": str(getattr(incident, "incident_cid", "") or ""),
        "plan": plan_body,
        "roots": {key: roots[key] for key in sorted(roots) if isinstance(roots[key], (str, int, bool)) or roots[key] is None},
        "completion_authority": False,
    }
    body["request_identity"] = identity({key: value for key, value in body.items() if key != "request_identity"})
    return body


class WorldwideWatchdogBridge:
    def __init__(self, *, repository: Path, state_root: Path, run_id: str, freshness_seconds: float = 120, clock=time.time, sleep=time.sleep, host_lifecycle=None, statvfs=None, lifecycle_restart=None, rescue_planner=None, rescue_request_factory=None, rescue_orchestrator=None, rescue_execution_request_factory=None, completion_verifier=None, process_identity=None):
        self.repository = repository.expanduser().resolve(strict=True)
        self.state_root = state_root.expanduser().resolve(strict=True)
        if not self.repository.is_dir() or not self.state_root.is_dir() or self.state_root.stat().st_uid != os.getuid() or not run_id:
            raise WatchdogBridgeError("owned repository/state root and exact run id are required")
        if self.state_root == self.repository or self.repository in self.state_root.parents:
            raise WatchdogBridgeError("supervisor state must stay outside the website repository")
        if not math.isfinite(freshness_seconds) or freshness_seconds <= 0:
            raise WatchdogBridgeError("freshness bound must be positive")
        self.run_id, self.freshness_seconds, self.clock, self.sleep = run_id, freshness_seconds, clock, sleep
        self.host_lifecycle = host_lifecycle
        self.statvfs = statvfs
        config = self.repository / "config/worldwide_supervisor_host.json"
        self.host_config = read_owned_json(config) if config.is_file() else {}
        self.framework_root = Path(self.host_config["framework_root"]).resolve(strict=True) if self.host_config.get("framework_root") else None
        # Validate configured bounds at construction, before any action can run.
        self.resources()
        if self.host_lifecycle is None and config.is_file():
            from .worldwide_legal_search_host_lifecycle import RegisteredSystemdLifecycle
            self.host_lifecycle = RegisteredSystemdLifecycle(repository=self.repository, state_root=self.state_root, run_id=self.run_id)
        if lifecycle_restart is not None and not callable(lifecycle_restart):
            raise WatchdogBridgeError("lifecycle_restart must be callable")
        if rescue_planner is not None and not callable(getattr(rescue_planner, "plan", None)):
            raise WatchdogBridgeError("rescue planner must provide plan")
        if rescue_orchestrator is not None and not callable(getattr(rescue_orchestrator, "execute", None)):
            raise WatchdogBridgeError("rescue orchestrator must provide execute")
        for name, factory in (("rescue_request_factory", rescue_request_factory), ("rescue_execution_request_factory", rescue_execution_request_factory), ("completion_verifier", completion_verifier), ("process_identity", process_identity)):
            if factory is not None and not callable(factory):
                raise WatchdogBridgeError(name + " must be callable")
        self._lifecycle_restart_injected = lifecycle_restart is not None
        self.lifecycle_restart = lifecycle_restart if lifecycle_restart is not None else (self._host_lifecycle_transition if self.host_lifecycle is not None else None)
        self.rescue_planner = rescue_planner
        self.rescue_request_factory = rescue_request_factory or default_rescue_request_factory
        self.rescue_orchestrator = rescue_orchestrator
        self.rescue_execution_request_factory = rescue_execution_request_factory or default_rescue_execution_request_factory
        self.completion_verifier = completion_verifier
        self.process_identity = process_identity

    def resources(self) -> dict[str, Any]:
        return resource_observation(repository=self.repository, state_root=self.state_root, framework_root=self.framework_root, config=self.host_config, statvfs=self.statvfs)

    def _host_probe(self) -> dict[str, Any] | None:
        if self.host_lifecycle is None:
            return None
        try:
            return self.host_lifecycle.probe()
        except (OSError, ValueError):
            return None

    def _readiness(self) -> dict[str, Any]:
        value = read_owned_json(self.state_root / "campaign-readiness.json")
        if value.get("schema") != "wwl.watchdog.ready@1" or value.get("run_id") != self.run_id or value.get("repository") != str(self.repository):
            raise WatchdogBridgeError("native readiness scope differs from this campaign")
        for name in ("policy_id", "native_revision", "admitted_plan_revision"):
            if not isinstance(value.get(name), str) or not value[name]:
                raise WatchdogBridgeError("native readiness is missing " + name)
        if not isinstance(value.get("bindings"), dict) or any(type(v) is not bool for v in value["bindings"].values()):
            raise WatchdogBridgeError("native readiness bindings must be explicit booleans")
        required = value.get("required_task_ids")
        if not isinstance(required, list) or not required or not all(isinstance(v, str) and v for v in required) or len(set(required)) != len(required):
            raise WatchdogBridgeError("native readiness has an invalid required task set")
        # Actual capability flags are passed through. In particular, this
        # bridge never turns a missing native repair handler into true.
        host = self._host_probe()
        if host is not None:
            value = {**value, "bindings": {**value["bindings"], "lifecycle_restart": True}, "registered_host_lifecycle": host}
        return self._apply_authority_overlay(value, host is not None)

    def ready(self, request: Mapping[str, Any]) -> dict[str, Any]:
        value = self._readiness()
        if request.get("policy_id") != value["policy_id"]:
            raise WatchdogBridgeError("requested policy differs from native readiness")
        requested = request.get("required_task_ids")
        if not isinstance(requested, list) or not set(requested) <= set(value["required_task_ids"]):
            raise WatchdogBridgeError("native readiness omitted requested work")
        resources = self.resources()
        return {**value, "resource_pressure": resources} if resources["blocked"] else value

    def observe(self, request: Mapping[str, Any]) -> dict[str, Any]:
        ready = self._readiness()
        if request.get("run_id") != self.run_id or request.get("policy_id") != ready["policy_id"]:
            raise WatchdogBridgeError("observation request identity differs")
        snapshot = read_owned_json(self.state_root / "campaign-status.json")
        if snapshot.get("schema") != "wwl.watchdog.snapshot@1" or snapshot.get("run_id") != self.run_id:
            raise WatchdogBridgeError("native status identity differs")
        observed = snapshot.get("observed_at")
        if type(observed) not in (int, float) or not math.isfinite(observed) or observed > self.clock() + 30:
            raise WatchdogBridgeError("native owner projection is stale or from the future")
        stale = self.clock() - observed > self.freshness_seconds
        host = self._host_probe()
        if host is None and stale:
            raise WatchdogBridgeError("native owner projection is stale and no registered host lifecycle is available")
        if host is not None:
            snapshot = {**snapshot, "projection_observed_at": observed, "source_projection_stale": stale,
                        "registered_host_lifecycle": host, "observed_at": self.clock(),
                        "owner_alive": host["owner_alive"],
                        "owner_identity_verified": bool(host["owner_identity_verified"] and (not host["owner_alive"] or snapshot.get("owner_pid") == host["main_pid"]))}
            if stale or snapshot.get("owner_pid") != host["main_pid"]:
                # A current host observation is not a current task-state proof.
                snapshot.update(completion_verified=False, completion_receipt="")
        resources = self.resources()
        if resources["blocked"]:
            # Disk pressure is an external admission blocker, not a software
            # defect. In particular, do not publish settlement files on a full
            # filesystem while observing it.
            return self._project_completion({**snapshot, "blocked_kind": "resource_quota", "resource_pressure": resources})
        from ..rescue.maintenance_hold import binding_released, review_hold
        holds = [self.state_root / name for name in ("HOLD", "STOP", "OPERATOR_STOP")]
        reviews = [review_hold(path, apply=request.get("execution_requested") is True, now=self.clock())
                   for path in holds if os.path.lexists(path)]
        # A native pause can be independent of filesystem markers. Clear its
        # stale projection only when every explicitly bound maintenance hold
        # has an exact durable release archive; preserve unbound native holds.
        bindings = snapshot.get("maintenance_hold_bindings")
        released = (type(bindings) is list and bool(bindings)
                    and all(type(item) is dict and item.get("path") in {str(p) for p in holds}
                            and binding_released(item) for item in bindings))
        snapshot = {**snapshot, "native_diagnostic_available": ready["bindings"].get("diagnose") is True, "hold": any(os.path.lexists(path) for path in holds)
                    or (bool(snapshot.get("hold")) and not released)}
        if reviews:
            snapshot["hold_reviews"] = reviews
        if snapshot["hold"]:
            self._record_stop_hold(hold_markers(self.state_root))
        # Canonical heartbeat, progress token and source revision are retained.
        observations = {item["action_id"]: item for item in snapshot.get("action_observations", [])}
        for action_id in request.get("action_ids", []):
            if not isinstance(action_id, str) or ACTION_ID.fullmatch(action_id) is None:
                raise WatchdogBridgeError("requested action id is malformed")
            name = action_id + ".json"
            try:
                item = read_owned_json(self.state_root / "control/host-terminal-observations" / name)
            except FileNotFoundError:
                try:
                    item = read_owned_json(self.state_root / "control/host-observations" / name)
                except FileNotFoundError:
                    continue
            if item.get("status") == "unknown" and host is not None:
                item = self._settle_superseded_host_recovery(action_id, item, host) or item
            if item.get("status") == "unknown" and item.get("fence_verified") is True and host is not None and host.get("fresh_owner_projection"):
                try:
                    before = read_owned_json(self.state_root / "control/host-before" / name)
                except FileNotFoundError:
                    before = None
                if before is not None and (before["main_pid"], before["start_ticks"]) != (host["main_pid"], host["start_ticks"]):
                    settlement = {"schema": "wwl.watchdog.host-settlement@1", "action_id": action_id, "before": before, "after": host, "prior_native_receipt": item["receipt"], "effects": ["old_lane_fenced", "lane_restarted"]}
                    publish_once(self.state_root / "control/host-settlement-receipts" / name, settlement)
                    item = {"schema": "wwl.watchdog.action-observation@1", "action_id": action_id, "status": "succeeded", "receipt": identity(settlement), "effects_verified": True}
                    publish_once(self.state_root / "control/host-terminal-observations" / name, item)
            observations[action_id] = item
        # Promotion is not incident recovery. A separate, admitted host
        # operation must fence/reload before a fresh native pass can settle it.
        for action_id in set(request.get("action_ids", [])) | set(observations):
            terminal = self._repair_settlement(action_id, snapshot, host)
            if terminal is not None:
                observations[action_id] = terminal
        snapshot["action_observations"] = list(observations.values())
        return self._project_completion(snapshot)

    def _settle_superseded_host_recovery(self, action_id: str, prior: Mapping[str, Any], host: dict[str, Any]) -> dict[str, Any] | None:
        """A later native fence can settle an earlier timed-out host action.

        This certifies known fencing of the old process tree. It never claims
        current owner health, provider effects, task completion, or success of
        the earlier incomplete recovery.
        """
        if self.host_lifecycle is None or not hasattr(self.host_lifecycle, "completed_recovery_receipt"):
            return None
        try:
            request = read_owned_json(self.state_root / "control/host-requests" / (action_id + ".json"))
            before = read_owned_json(self.state_root / "control/host-before" / (action_id + ".json"))
        except FileNotFoundError:
            return None
        if request.get("run_id") != self.run_id or request.get("action_id") != action_id or request.get("kind") != "recover":
            return None
        if type(request.get("issued_at")) not in (int, float) or not math.isfinite(request["issued_at"]):
            return None
        paths = list((self.state_root / "control/host-requests").glob("*.json"))
        if len(paths) > 128:
            raise WatchdogBridgeError("host recovery settlement exceeds the campaign bound")
        for path in sorted(paths):
            if path.stem == action_id:
                continue
            later = read_owned_json(path)
            if type(later.get("issued_at")) not in (int, float) or not math.isfinite(later["issued_at"]):
                continue
            if (later.get("action_id") != path.stem or later.get("run_id") != self.run_id or later.get("policy_id") != request.get("policy_id")
                    or later.get("kind") != "recover" or later.get("issued_at", 0) <= request.get("issued_at", 0)
                    or (action_id not in later.get("unsettled_action_ids", []) and later.get("previous_unacknowledged_action") != action_id)):
                continue
            try:
                completed = read_owned_json(self.state_root / "control/host-observations" / path.name)
                admission = read_owned_json(self.state_root / "host-recovery/admissions" / path.name)
                if completed.get("status") != "succeeded" or completed.get("effects_verified") is not True or canonical(admission.get("action")) != canonical(later):
                    continue
                native = self.host_lifecycle.completed_recovery_receipt(path.stem, completed["receipt"])
                prior_native = self.host_lifecycle.recorded_recovery_receipt(action_id, prior["receipt"])
                fenced = self.host_lifecycle.prior_owner_fenced(before, host)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError):
                continue
            if fenced is None or native.get("started_at_ms", 0) < int(request["issued_at"] * 1000):
                continue
            settlement = {"schema": "wwl.watchdog.host-supersession@1", "run_id": self.run_id, "action_id": action_id,
                "prior_request_identity": identity(request), "prior_unknown_receipt": prior["receipt"], "before": before,
                "prior_native_receipt": prior_native,
                "later_action_id": path.stem, "later_native_fence_receipt": native, "old_owner_fence": fenced,
                "effects": ["old_lane_fenced"], "recovery_verified": False}
            publish_once(self.state_root / "control/host-supersession-receipts" / (action_id + ".json"), settlement)
            terminal = {"schema": "wwl.watchdog.action-observation@1", "action_id": action_id, "status": "failed",
                "receipt": identity(settlement), "effects_verified": True, "fence_verified": True, "recovery_verified": False,
                "reason": "incomplete_recovery_superseded_by_later_verified_native_fence"}
            publish_once(self.state_root / "control/host-terminal-observations" / (action_id + ".json"), terminal)
            return terminal
        return None

    def verify_host_supersession(self, action_id: str, *, expected_task_id: str | None = None,
                                 expected_stage: str | None = None, require_owner_stopped: bool = False) -> dict[str, Any]:
        """Recheck the complete read-only proof chain for native maintenance.

        The returned observation is not task rearm or completion authority.
        The native owner must separately authorize its exact task-status CAS.
        """
        if not isinstance(action_id, str) or ACTION_ID.fullmatch(action_id) is None or self.host_lifecycle is None:
            raise WatchdogBridgeError("host supersession needs an exact registered action")
        name = action_id + ".json"
        proof_path = self.state_root / "control/host-supersession-receipts" / name
        proof = read_owned_json(proof_path)
        terminal = read_owned_json(self.state_root / "control/host-terminal-observations" / name)
        original = read_owned_json(self.state_root / "control/host-requests" / name)
        before = read_owned_json(self.state_root / "control/host-before" / name)
        admission = read_owned_json(self.state_root / "host-recovery/admissions" / name)
        unknown = read_owned_json(self.state_root / "control/host-observations" / name)
        if (proof.get("schema") != "wwl.watchdog.host-supersession@1" or proof.get("run_id") != self.run_id or proof.get("action_id") != action_id
                or original.get("schema") != "wwl.watchdog.action@1" or original.get("action_id") != action_id or original.get("run_id") != self.run_id or original.get("kind") != "recover"
                or proof.get("prior_request_identity") != identity(original) or canonical(proof.get("before")) != canonical(before)
                or canonical(admission.get("action")) != canonical(original) or unknown.get("action_id") != action_id or unknown.get("status") != "unknown"
                or unknown.get("receipt") != proof.get("prior_unknown_receipt") or proof.get("recovery_verified") is not False):
            raise WatchdogBridgeError("superseded native action proof bindings differ")
        if expected_task_id is not None and original.get("task_id") != expected_task_id:
            raise WatchdogBridgeError("superseded host action belongs to another task")
        if expected_stage is not None and original.get("stage") != expected_stage:
            raise WatchdogBridgeError("superseded host action belongs to another native phase")
        if (terminal.get("schema") != "wwl.watchdog.action-observation@1" or terminal.get("action_id") != action_id or terminal.get("status") != "failed"
                or terminal.get("receipt") != identity(proof) or terminal.get("effects_verified") is not True or terminal.get("fence_verified") is not True or terminal.get("recovery_verified") is not False):
            raise WatchdogBridgeError("supersession terminal receipt is not exact failed/fenced evidence")
        later_id = proof.get("later_action_id")
        if not isinstance(later_id, str) or ACTION_ID.fullmatch(later_id) is None or later_id == action_id:
            raise WatchdogBridgeError("supersession later action identity is invalid")
        later = read_owned_json(self.state_root / "control/host-requests" / (later_id + ".json"))
        later_admission = read_owned_json(self.state_root / "host-recovery/admissions" / (later_id + ".json"))
        if (later.get("run_id") != self.run_id or later.get("policy_id") != original.get("policy_id") or later.get("action_id") != later_id or later.get("kind") != "recover"
                or canonical(later_admission.get("action")) != canonical(later)
                or (action_id not in later.get("unsettled_action_ids", []) and later.get("previous_unacknowledged_action") != action_id)
                or type(later.get("issued_at")) not in (int, float) or type(original.get("issued_at")) not in (int, float) or later["issued_at"] <= original["issued_at"]):
            raise WatchdogBridgeError("supersession later request does not fence the original action")
        native = self.host_lifecycle.completed_recovery_receipt(later_id, proof["later_native_fence_receipt"]["receipt_cid"])
        prior_native = self.host_lifecycle.recorded_recovery_receipt(action_id, unknown["receipt"])
        if canonical(native) != canonical(proof["later_native_fence_receipt"]) or canonical(prior_native) != canonical(proof["prior_native_receipt"]):
            raise WatchdogBridgeError("supersession native content-addressed receipts changed")
        current = self.host_lifecycle.probe()
        fenced = self.host_lifecycle.prior_owner_fenced(before, current)
        if fenced is None or (require_owner_stopped and (current.get("owner_alive") or current.get("main_pid") != 0)):
            raise WatchdogBridgeError("superseded native owner is not currently fenced")
        for key in ("main_pid", "start_ticks", "control_group", "unit"):
            if admission.get("before", {}).get(key) != before.get(key):
                raise WatchdogBridgeError("supersession original PID birth/cgroup differs from native admission")
        return {"schema": "wwl.watchdog.verified-host-supersession@1", "action": original, "supersession_path": str(proof_path),
                "supersession_receipt": identity(proof), "native_prior_receipt": prior_native, "native_later_fence_receipt": native,
                "current_owner_fence": fenced, "current_host": current, "completion_authority": False, "task_retry_authority": False}

    def _reload_binding(self, action: Mapping[str, Any]) -> dict[str, Any]:
        repair_id = action.get("reload_after_action_id")
        expected = action.get("expected_framework_revision")
        failure_id = action.get("original_failure_receipt")
        if action.get("kind") != "recover" or not isinstance(repair_id, str) or ACTION_ID.fullmatch(repair_id) is None or repair_id == action["action_id"]:
            raise WatchdogBridgeError("reload must reference a distinct native repair action")
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", expected) is None or not isinstance(failure_id, str) or not failure_id:
            raise WatchdogBridgeError("reload requires the exact promoted source and original failure")
        ledger = read_owned_json(self.state_root / "control/action-ledger.json")
        entry = ledger.get(repair_id, {})
        request, result, observation = entry.get("request", {}), entry.get("result", {}), entry.get("observation", {})
        if request.get("action_id") != repair_id or request.get("kind") != "repair" or request.get("run_id") != self.run_id or request.get("policy_id") != action["policy_id"]:
            raise WatchdogBridgeError("reload does not reference this campaign's admitted repair")
        if result.get("status") != "accepted" or result.get("action_id") != repair_id or type(result.get("deadline_at")) not in (int, float) or not math.isfinite(result["deadline_at"]):
            raise WatchdogBridgeError("referenced repair has no bounded native admission")
        if observation.get("status") != "running" or observation.get("candidate_promotion_verified") is not True or observation.get("effects_verified") is not True or observation.get("requires_fenced_reload") is not True or observation.get("recovery_verified") is not False:
            raise WatchdogBridgeError("referenced repair has no validated pending promotion")
        if observation.get("expected_framework_revision") != expected or observation.get("original_failure_receipt") != failure_id or request.get("reproducer_receipt") != failure_id:
            raise WatchdogBridgeError("reload source or failure differs from the native promotion")
        evidence = read_owned_json(self.state_root / "repairs" / repair_id / "native-repair-result.json")
        if evidence.get("schema") != "wwl.native-repair-result@1" or evidence.get("action_id") != repair_id or evidence.get("validated") is not True or not evidence.get("native_completion_receipt") or not evidence.get("native_projection"):
            raise WatchdogBridgeError("reload is missing native validation and promotion evidence")
        if evidence.get("action_request_identity") != native_identity(request) or evidence.get("framework_head") != expected or evidence.get("original_failure_receipt") != failure_id:
            raise WatchdogBridgeError("reload native evidence belongs to another request or source")
        return {"schema": "wwl.watchdog.repair-reload@1", "run_id": self.run_id, "repair_action_id": repair_id,
                "recover_action_id": action["action_id"], "recover_request_identity": identity(action),
                "expected_framework_revision": expected, "original_failure_receipt": failure_id,
                "original_task_id": request.get("task_id", ""),
                "original_repair_deadline_at": result["deadline_at"], "native_promotion_receipt": observation["receipt"],
                "native_completion_receipt": evidence["native_completion_receipt"], "issued_at": action["issued_at"]}

    def _repair_settlement(self, repair_id: str, snapshot: Mapping[str, Any], host: Mapping[str, Any] | None) -> dict[str, Any] | None:
        terminal_path = self.state_root / "control/repair-terminal-observations" / (repair_id + ".json")
        try:
            return read_owned_json(terminal_path)
        except FileNotFoundError:
            pass
        if host is None or not host.get("fresh_owner_projection") or snapshot.get("source_projection_stale") or snapshot.get("owner_pid") != host.get("main_pid") or snapshot.get("effects_unknown") or snapshot.get("native_error"):
            return None
        successful_pass = snapshot.get("last_successful_native_pass", {})
        if not isinstance(successful_pass, dict) or successful_pass.get("owner_pid") != host["main_pid"] or successful_pass.get("framework_revision") != snapshot.get("loaded_framework_revision"):
            return None
        if not isinstance(successful_pass.get("result"), dict) or not successful_pass["result"] or successful_pass.get("receipt_id") != native_identity(successful_pass["result"]) or not successful_pass.get("progress_token"):
            return None
        completed_at = successful_pass.get("completed_at")
        if type(completed_at) not in (int, float) or not math.isfinite(completed_at) or completed_at > self.clock() + 30:
            return None
        reloads = self.state_root / "control/repair-reloads" / repair_id
        paths = list(reloads.glob("*.json"))
        if len(paths) > 128:
            raise WatchdogBridgeError("repair reload receipt lookup exceeds the campaign bound")
        for path in sorted(paths):
            binding = read_owned_json(path)
            if binding.get("repair_action_id") != repair_id or binding.get("run_id") != self.run_id or binding.get("recover_action_id") != path.stem:
                raise WatchdogBridgeError("repair reload receipt has mismatched scope")
            if completed_at < binding["issued_at"] or snapshot.get("loaded_framework_revision") != binding["expected_framework_revision"]:
                continue
            name = path.name
            try:
                before = read_owned_json(self.state_root / "control/host-before" / name)
                try:
                    recovery = read_owned_json(self.state_root / "control/host-terminal-observations" / name)
                except FileNotFoundError:
                    recovery = read_owned_json(self.state_root / "control/host-observations" / name)
            except FileNotFoundError:
                continue
            if recovery.get("status") != "succeeded" or recovery.get("effects_verified") is not True or (before["main_pid"], before["start_ticks"]) == (host["main_pid"], host["start_ticks"]):
                continue
            # Resolve the original failure content; a cleared bootstrap bool
            # alone cannot substitute for a new native pass of the same task.
            failures = list((self.state_root / "failures").glob("*.json"))
            if len(failures) > 256:
                raise WatchdogBridgeError("original repair failure lookup exceeds its bound")
            matches = [value for item in failures if (value := read_owned_json(item)).get("receipt_id") == binding["original_failure_receipt"]]
            if len(matches) != 1:
                continue
            failure = matches[0]
            if failure.get("run_id") != self.run_id or native_identity({k: v for k, v in failure.items() if k != "receipt_id"}) != failure["receipt_id"]:
                raise WatchdogBridgeError("original repair failure receipt is invalid")
            task_id = failure.get("task_id") or binding.get("original_task_id") or successful_pass["result"].get("active_task_id")
            completion = snapshot.get("native_completion_receipts_by_alias", {}).get(task_id)
            # A nonthrowing daemon pass can still return a blocked or failed
            # attempt. Only the original task's canonical validated completion
            # proves recovery; a cleared error flag or return value cannot.
            if (completed_at <= failure.get("observed_at", 0) or not task_id
                    or successful_pass["result"].get("active_task_id") != task_id
                    or task_id not in snapshot.get("verified_task_ids", [])
                    or snapshot.get("task_statuses", {}).get(task_id) not in {"completed", "complete", "done"}
                    or not isinstance(completion, str) or not completion
                    or not snapshot.get("native_completion_projection")):
                continue
            on_time = completed_at <= binding["original_repair_deadline_at"]
            settlement = {"schema": "wwl.watchdog.repair-settlement@1", "run_id": self.run_id, "repair_action_id": repair_id,
                "reload_binding": binding, "host_recovery_receipt": recovery["receipt"], "host_owner": dict(host),
                "loaded_framework_revision": snapshot["loaded_framework_revision"], "native_pass": successful_pass,
                "recovered_task_id": task_id, "native_task_completion_receipt": completion,
                "native_completion_projection": snapshot["native_completion_projection"],
                "original_failure_receipt": failure["receipt_id"], "original_fault_cleared": True, "within_original_deadline": on_time}
            publish_once(self.state_root / "control/repair-settlement-receipts" / (repair_id + ".json"), settlement)
            terminal = {"schema": "wwl.watchdog.action-observation@1", "action_id": repair_id, "status": "succeeded" if on_time else "failed",
                "receipt": identity(settlement), "effects_verified": True, "recovery_verified": True,
                "reason": "fenced_native_owner_proved_recovery" if on_time else "recovery_verified_after_original_repair_deadline"}
            publish_once(terminal_path, terminal)
            return terminal
        return None

    def _refuse(self, action: Mapping[str, Any], reason: str) -> dict[str, Any]:
        receipt = {"schema": "wwl.watchdog.control-refusal@1", "action_id": action["action_id"], "request_identity": identity(action), "run_id": self.run_id, "reason": reason, "effects_applied": False}
        publish_once(self.state_root / "control" / "refusals" / (action["action_id"] + ".json"), receipt)
        return {"schema": "wwl.watchdog.action-result@1", "action_id": action["action_id"], "status": "blocked", "receipt": identity(receipt)}

    def dispatch(self, action: Mapping[str, Any], *, timeout_seconds: float = 50) -> dict[str, Any]:
        if not isinstance(action, dict) or action.get("schema") != "wwl.watchdog.action@1" or not isinstance(action.get("action_id"), str) or ACTION_ID.fullmatch(action["action_id"]) is None:
            raise WatchdogBridgeError("malformed bounded native action request")
        if action.get("run_id") != self.run_id or action.get("kind") not in {"reconcile", "recover", "diagnose", "repair"}:
            raise WatchdogBridgeError("action run or kind differs")
        if type(action.get("issued_at")) not in (int, float) or not math.isfinite(action["issued_at"]) or action["issued_at"] > self.clock() + 30:
            raise WatchdogBridgeError("action must have a finite original issue time")
        ready = self._readiness()
        if action.get("policy_id") != ready["policy_id"]:
            raise WatchdogBridgeError("action policy differs")
        if not math.isfinite(timeout_seconds) or not 0 <= timeout_seconds <= 50:
            raise WatchdogBridgeError("dispatch wait must be within 50 seconds")
        path = self.state_root / "control" / "requests" / (action["action_id"] + ".json")
        host_path = self.state_root / "control" / "host-requests" / path.name
        refusal_path = self.state_root / "control" / "refusals" / path.name
        try:
            refusal = read_owned_json(refusal_path)
        except FileNotFoundError:
            refusal = None
        if refusal is not None:
            if refusal.get("request_identity") != identity(action):
                raise WatchdogBridgeError("action id reused after refusal")
            return {"schema": "wwl.watchdog.action-result@1", "action_id": action["action_id"], "status": "blocked", "receipt": identity(refusal)}
        try:
            submitted = read_owned_json(path)
        except FileNotFoundError:
            try:
                submitted = read_owned_json(host_path)
            except FileNotFoundError:
                submitted = None
        if submitted is not None:
            if canonical(submitted) != canonical(action):
                raise WatchdogBridgeError("action id reused with a different request")
            # Existing submission may already have effects. A later scope
            # transition cannot turn its outcome into a no-effect refusal.
            return self._wait_result(action, path, timeout_seconds)
        resources = self.resources()
        if resources["blocked"]:
            # No durable no-effect receipt can be promised on a full volume.
            # Preserve the request/outbox as unknown and perform no mutation.
            return {"schema": "wwl.watchdog.action-result@1", "action_id": action["action_id"], "status": "unknown",
                    "receipt": identity({"not_dispatched_for_resource_pressure": identity(action), "evidence": resources}), "resource_pressure": resources}
        capability = {"reconcile": "reconcile", "recover": "lifecycle_restart", "diagnose": "diagnose", "repair": "repair_dispatch"}[action["kind"]]
        if ready["bindings"].get(capability) is not True:
            return self._refuse(action, "native_" + capability + "_not_bound")
        snapshot = self.observe({"run_id": self.run_id, "policy_id": ready["policy_id"]})
        # A stopped owner cannot refresh its cached hold projection. Respect
        # current operator controls before publishing any new operation.
        if snapshot.get("hold") or any((self.state_root / name).exists() for name in ("HOLD", "STOP", "OPERATOR_STOP")):
            return self._refuse(action, "native_pause_or_hold")
        if action.get("authority_revision") != snapshot.get("authority_revision") or action.get("accepted_tree") != snapshot.get("accepted_tree"):
            return self._refuse(action, "native_observation_revision_changed")
        if action["kind"] == "diagnose":
            if (not action.get("task_id") or not action.get("active_attempt_id")
                    or action.get("task_id") != snapshot.get("task_id")
                    or action.get("active_attempt_id") != snapshot.get("active_attempt_id")):
                return self._refuse(action, "native_diagnosis_attempt_scope_changed")
        host = snapshot.get("registered_host_lifecycle")
        duplicate = self._claim_or_duplicate(action)
        if duplicate is not None:
            return duplicate
        if action.get("reload_after_action_id") is not None:
            if host is None:
                return self._refuse(action, "registered_host_lifecycle_required_for_repair_reload")
            try:
                binding = self._reload_binding(action)
            except (OSError, ValueError, KeyError) as error:
                return self._refuse(action, "native_promotion_reload_binding_rejected: " + str(error))
            publish_once(self.state_root / "control/repair-reloads" / binding["repair_action_id"] / (action["action_id"] + ".json"), binding)
        if host is not None and (action["kind"] == "recover" or (action["kind"] == "reconcile" and (not host["owner_alive"] or snapshot.get("source_projection_stale")))):
            return self._host_recover(action, host_path)
        publish_once(path, action)
        self._seal_published_action(action, status="inflight")
        return self._wait_result(action, path, timeout_seconds)

    def _host_recover(self, action: Mapping[str, Any], path: Path) -> dict[str, Any]:
        publish_once(self.state_root / "control/host-before" / path.name, self.host_lifecycle.probe())
        publish_once(path, action)
        self._seal_published_action(action, status="inflight")
        result = self.host_lifecycle.recover(dict(action))
        receipt = result["receipt"]
        observation = {"schema": "wwl.watchdog.action-observation@1", "action_id": action["action_id"],
                       "status": "succeeded" if result["recovered"] else "unknown", "receipt": receipt,
                       "effects_verified": bool(result["recovered"]), "fence_verified": result.get("old_owner_fenced") is True}
        publish_once(self.state_root / "control/host-observations" / path.name, observation)
        fresh_admission = result["recovered"] and float(action["issued_at"]) + 120 > self.clock()
        response = {"schema": "wwl.watchdog.action-result@1", "action_id": action["action_id"],
                    "status": "accepted" if fresh_admission else "unknown", "receipt": receipt}
        if fresh_admission:
            response["deadline_at"] = min(float(action["issued_at"]) + 120, self.clock() + 60)
        publish_once(self.state_root / "control/results" / path.name, response)
        return response

    def _wait_result(self, action: Mapping[str, Any], path: Path, timeout_seconds: float) -> dict[str, Any]:
        result_path = self.state_root / "control" / "results" / path.name
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                result = read_owned_json(result_path)
            except FileNotFoundError:
                result = None
            if result is not None:
                if result.get("schema") != "wwl.watchdog.action-result@1" or result.get("action_id") != action["action_id"] or result.get("status") not in {"accepted", "blocked", "unknown"} or not isinstance(result.get("receipt"), str) or not result["receipt"]:
                    raise WatchdogBridgeError("native action result is malformed or mismatched")
                return result
            if time.monotonic() >= deadline:
                # Durable submission is not an admission or success receipt.
                return {"schema": "wwl.watchdog.action-result@1", "action_id": action["action_id"], "status": "unknown", "receipt": identity({"submitted_request": identity(action)})}
            self.sleep(min(0.1, max(0, deadline - time.monotonic())))

    def _apply_authority_overlay(self, value: dict[str, Any], host_ready: bool) -> dict[str, Any]:
        bindings = dict(value["bindings"])
        overlay: dict[str, bool] = {}
        if self._lifecycle_restart_injected or host_ready:
            overlay["lifecycle_restart"] = True
        if self.rescue_planner is not None and callable(self.rescue_request_factory):
            overlay["rescue_planner"] = True
        if self.rescue_orchestrator is not None and callable(self.rescue_execution_request_factory):
            overlay["rescue_orchestrator"] = True
        # The ledger is implemented. Surface that only when the native projection
        # already names the capability, so an omitted flag is not invented.
        if "incident_deduplication" in bindings:
            overlay["incident_deduplication"] = True
        if not overlay:
            return value
        return {**value, "bindings": {**bindings, **overlay}}

    def _host_lifecycle_transition(self, lane_info: Mapping[str, Any]) -> dict[str, Any]:
        """Adapt registered host recovery to restart_lane's lifecycle callback."""
        if self.host_lifecycle is None:
            raise WatchdogBridgeError("registered host lifecycle is not bound")
        action_id = lane_info.get("action_id")
        if not isinstance(action_id, str) or ACTION_ID.fullmatch(action_id) is None or not isinstance(lane_info.get("policy_id"), str) or not lane_info["policy_id"]:
            raise WatchdogBridgeError("fenced lifecycle restart requires an exact action and policy")
        issued_at = lane_info.get("issued_at", self.clock())
        if type(issued_at) not in (int, float) or isinstance(issued_at, bool) or not math.isfinite(issued_at):
            raise WatchdogBridgeError("fenced lifecycle restart requires a finite issue time")
        action = {"schema": "wwl.watchdog.action@1", "action_id": action_id, "run_id": self.run_id, "policy_id": lane_info["policy_id"], "kind": "recover", "issued_at": issued_at}
        result = self.host_lifecycle.recover(action)
        current = self.host_lifecycle.probe()
        new_pid = current.get("main_pid")
        restarted = bool(result.get("recovered") and result.get("old_owner_fenced") and isinstance(new_pid, int) and new_pid > 0)
        return {"restarted": restarted, "succeeded": restarted, "new_pid": new_pid if restarted else None, "receipt_id": result.get("receipt"),
                "old_owner_fenced": result.get("old_owner_fenced") is True, "deduplicated": result.get("deduplicated") is True,
                "reason": "registered_host_lifecycle_fenced" if restarted else "registered_host_lifecycle_not_fenced"}

    def _birth(self, lane_info: Mapping[str, Any]) -> dict[str, Any] | None:
        pid = lane_info.get("pid", lane_info.get("main_pid"))
        if isinstance(pid, bool) or type(pid) is not int or pid <= 0:
            return None
        if self.process_identity is not None:
            probed = self.process_identity(pid)
        else:
            from .worldwide_legal_search_host_lifecycle import process_identity
            probed = process_identity(pid)
        if not isinstance(probed, Mapping):
            raise WatchdogBridgeError("process identity probe returned a non-object")
        ticks = probed.get("start_ticks", lane_info.get("start_ticks", 0))
        if isinstance(ticks, bool) or type(ticks) is not int:
            ticks = 0
        return {"pid": pid, "alive": bool(probed.get("alive")), "start_ticks": ticks}

    def fenced_lifecycle_restart(self, lane_info: Mapping[str, Any]) -> dict[str, Any]:
        """Restart through the native lifecycle callback and fence the old birth.

        A returned restart that still names the previous pid and start ticks is
        not a fence. Successful fences are stored once and replay without
        calling the lifecycle callback again.
        """
        if self.lifecycle_restart is None or not isinstance(lane_info, Mapping):
            raise WatchdogBridgeError("fenced lifecycle restart is not bound")
        incident = lane_info.get("incident_id") or lane_info.get("action_id")
        if not isinstance(incident, str) or INCIDENT_ID.fullmatch(incident) is None:
            raise WatchdogBridgeError("fenced lifecycle restart requires an incident identity")
        before = self._birth(lane_info)
        key_body = {"run_id": self.run_id, "incident": incident, "before": before}
        result_path = self.state_root / "control/process-fences" / (hashlib.sha256(canonical(key_body)).hexdigest() + ".json")
        try:
            stored = read_owned_json(result_path)
        except FileNotFoundError:
            stored = None
        if stored is not None:
            return stored
        calls = {"count": 0}

        def transition(info: Mapping[str, Any]) -> dict[str, Any]:
            calls["count"] += 1
            raw = self.lifecycle_restart(dict(info))
            if not isinstance(raw, Mapping):
                raise WatchdogBridgeError("lifecycle_restart returned a non-object")
            result = dict(raw)
            restarted = bool(result.get("restarted") or result.get("succeeded") or result.get("status") == "succeeded")
            new_pid = result.get("new_pid")
            if restarted:
                after = self._birth({"pid": new_pid, "start_ticks": result.get("start_ticks")})
                if before is not None and before.get("alive") and after == before:
                    return {"restarted": False, "succeeded": False, "reason": "previous_process_not_fenced", "old_owner_fenced": False, "new_pid": new_pid}
                if isinstance(new_pid, bool) or type(new_pid) is not int:
                    return {"restarted": False, "succeeded": False, "reason": "lifecycle_receipt_missing_new_process_identity", "old_owner_fenced": False}
                result["restarted"] = True
                result["old_owner_fenced"] = True
            return result

        from ..rescue.supervisor_watchdog import restart_lane
        restarted = restart_lane(dict(lane_info), repo_root=self.repository, lifecycle_transition=transition)
        restarted = dict(restarted)
        restarted["lifecycle_calls"] = calls["count"]
        if not restarted.get("restarted"):
            return restarted
        kept = {key: restarted.get(key) for key in ("restarted", "new_pid", "reason", "receipt_id", "old_owner_fenced")}
        receipt = {"schema": "wwl.watchdog.process-fence@1", "run_id": self.run_id, "incident": incident, "before": before, "result": kept, "completion_authority": False}
        receipt["receipt"] = identity({key: value for key, value in receipt.items() if key != "receipt"})
        publish_once(result_path, receipt)
        return read_owned_json(result_path)

    def bind_supervisor_watchdog(self, manifest_path: Path, **kwargs: Any):
        """Construct SupervisorWatchdog with this campaign's bound authorities."""
        from ..rescue.supervisor_watchdog import SupervisorWatchdog

        return SupervisorWatchdog(
            manifest_path=Path(manifest_path),
            repo_root=self.repository,
            lifecycle_restart=self.lifecycle_restart,
            rescue_planner=self.rescue_planner,
            rescue_request_factory=self.rescue_request_factory,
            rescue_orchestrator=self.rescue_orchestrator,
            rescue_execution_request_factory=self.rescue_execution_request_factory,
            **kwargs,
        )

    def _incident_path(self, action: Mapping[str, Any]) -> Path | None:
        incident = action.get("incident_id")
        if not isinstance(incident, str) or INCIDENT_ID.fullmatch(incident) is None:
            return None
        digest = hashlib.sha256(incident.encode("utf-8")).hexdigest()
        return self.state_root / "control/incidents" / (digest + ".json")

    def _claim_or_duplicate(self, action: Mapping[str, Any]) -> dict[str, Any] | None:
        """Persist the first action for an incident and refuse a second writer."""
        path = self._incident_path(action)
        if path is None:
            return None
        claim = {"schema": "wwl.watchdog.incident-claim@1", "run_id": self.run_id, "incident_id": action["incident_id"],
                 "action_id": action["action_id"], "kind": action["kind"], "request_identity": identity(action)}
        try:
            current = read_owned_json(path)
        except FileNotFoundError:
            current = None
        if current is None:
            publish_once(path, claim)
            return None
        if current.get("schema") != claim["schema"] or current.get("run_id") != self.run_id or current.get("incident_id") != action["incident_id"]:
            raise WatchdogBridgeError("durable incident claim scope differs")
        if current.get("action_id") == action["action_id"] and current.get("request_identity") == identity(action):
            return None
        return self._refuse(action, "durable_incident_already_claimed")

    def _authority_receipt(self, directory: str, incident_id: str, producer) -> dict[str, Any]:
        if not isinstance(incident_id, str) or INCIDENT_ID.fullmatch(incident_id) is None:
            raise WatchdogBridgeError("rescue incident identity is invalid")
        digest = hashlib.sha256((self.run_id + ":" + incident_id).encode("utf-8")).hexdigest()
        result_path = self.state_root / "control" / directory / (digest + ".json")
        claim_path = self.state_root / "control" / (directory + "-claims") / (digest + ".json")
        try:
            return read_owned_json(result_path)
        except FileNotFoundError:
            pass
        try:
            claim = read_owned_json(claim_path)
        except FileNotFoundError:
            claim = None
        if claim is not None:
            return {"schema": "wwl.watchdog.authority-unknown@1", "run_id": self.run_id, "incident_id": incident_id, "status": "unknown",
                    "effects_unknown": True, "completion_authority": False, "receipt": identity(claim)}
        claim = {"schema": "wwl.watchdog.authority-claim@1", "run_id": self.run_id, "incident_id": incident_id, "directory": directory}
        publish_once(claim_path, claim)
        produced = _mapping_result(producer())
        if produced.get("completion_authority") is True:
            raise WatchdogBridgeError("native authority tried to grant completion")
        receipt = {"schema": "wwl.watchdog.authority-receipt@1", "run_id": self.run_id, "incident_id": incident_id, "directory": directory,
                   "result": produced, "completion_authority": False}
        receipt["receipt"] = identity({key: value for key, value in receipt.items() if key != "receipt"})
        publish_once(result_path, receipt)
        return receipt

    def plan_rescue(self, diagnosis: Any, exhaustion: Any, roots: Mapping[str, str], *, incident_id: str) -> dict[str, Any]:
        if self.rescue_planner is None or not callable(self.rescue_request_factory):
            raise WatchdogBridgeError("rescue planner is not bound")

        def produce():
            request = self.rescue_request_factory(diagnosis, exhaustion, roots)
            return self.rescue_planner.plan(request)

        return self._authority_receipt("rescue-plans", incident_id, produce)

    def execute_rescue(self, diagnosis: Any, exhaustion: Any, plan: Any, roots: Mapping[str, Any], *, incident_id: str) -> dict[str, Any]:
        if self.rescue_orchestrator is None or not callable(self.rescue_execution_request_factory):
            raise WatchdogBridgeError("rescue orchestrator is not bound")

        def produce():
            request = self.rescue_execution_request_factory(diagnosis, exhaustion, plan, roots)
            return self.rescue_orchestrator.execute(request)

        return self._authority_receipt("rescue-executions", incident_id, produce)

    def _proof_rejection(self, snapshot: Mapping[str, Any], proof: Mapping[str, Any]) -> str:
        if proof.get("schema") != "wwl.native-current-tree-completion@1" or proof.get("run_id") != self.run_id:
            return "current_tree_completion_scope"
        required = proof.get("required_task_ids")
        if not isinstance(required, list) or not required or not all(isinstance(item, str) and item for item in required) or len(set(required)) != len(required):
            return "current_tree_required_tasks"
        if snapshot.get("required_task_ids") != required or proof.get("verified_task_ids") != required or snapshot.get("verified_task_ids") != required:
            return "current_tree_verified_tasks"
        tree = proof.get("accepted_tree")
        if not isinstance(tree, str) or not tree or tree != snapshot.get("accepted_tree"):
            return "current_tree_mismatch"
        # Missing frontier fields are not an empty frontier. Counts alone cannot
        # satisfy the current-tree proof.
        if (snapshot.get("task_id") != "" or snapshot.get("runnable_count") != 0 or snapshot.get("effects_unknown") is not False
                or snapshot.get("hold") is not False or proof.get("task_id") != "" or proof.get("runnable_count") != 0
                or proof.get("effects_unknown") is not False or proof.get("hold") is not False):
            return "current_tree_work_remains"
        body = {key: proof.get(key) for key in COMPLETION_PROOF_FIELDS}
        receipt = proof.get("completion_receipt")
        if not isinstance(receipt, str) or receipt != native_identity(body):
            return "current_tree_receipt"
        observed_receipt = snapshot.get("completion_receipt")
        if observed_receipt not in (None, "", receipt):
            return "current_tree_receipt"
        return ""

    def _completion_verification(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        verdict = {"schema": "wwl.watchdog.completion-verification@1", "run_id": self.run_id, "accepted_tree": snapshot.get("accepted_tree", ""),
                   "completion_verified": False, "completion_receipt": "", "completion_authority": False, "task_retry_authority": False, "reason": ""}
        if self.completion_verifier is not None:
            produced = self.completion_verifier(dict(snapshot))
            if not isinstance(produced, Mapping) or produced.get("completion_authority") is True or produced.get("task_retry_authority") is True:
                return {**verdict, "reason": "completion_verifier_refused_authority"}
            verified = produced.get("completion_verified") is True
            receipt = produced.get("completion_receipt")
            if verified and (not isinstance(receipt, str) or not receipt or produced.get("accepted_tree") != snapshot.get("accepted_tree")):
                return {**verdict, "reason": "completion_verifier_tree_unbound"}
            if not verified:
                return {**verdict, "reason": str(produced.get("reason") or "current_tree_completion_rejected")}
            return {**verdict, "completion_verified": True, "completion_receipt": receipt, "reason": "native_current_tree_completion_verified"}
        path = self.state_root / "control/current-tree-completion.json"
        try:
            proof = read_owned_json(path)
        except FileNotFoundError:
            return {**verdict, "reason": "current_tree_completion_proof_missing"}
        except (OSError, ValueError) as error:
            return {**verdict, "reason": "current_tree_completion_proof_unreadable: " + str(error)}
        reason = self._proof_rejection(snapshot, proof)
        if reason:
            return {**verdict, "reason": reason}
        return {**verdict, "completion_verified": True, "completion_receipt": proof["completion_receipt"], "reason": "native_current_tree_completion_verified"}

    def _project_completion(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        claimed = bool(snapshot.get("completion_verified")) or bool(snapshot.get("completion_receipt"))
        proof_exists = (self.state_root / "control/current-tree-completion.json").is_file() or self.completion_verifier is not None
        if not claimed and not proof_exists:
            return snapshot
        verdict = self._completion_verification(snapshot)
        if verdict["completion_verified"]:
            return {**snapshot, "completion_verified": True, "completion_receipt": verdict["completion_receipt"], "current_tree_completion": verdict}
        if not claimed:
            return {**snapshot, "current_tree_completion": verdict}
        return {**snapshot, "completion_verified": False, "completion_receipt": "", "current_tree_completion": verdict}

    def verify(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Verify either one settled action or current-tree completion.

        The result is evidence for the watchdog loop. It does not rewrite task
        status, rearm work, or grant completion authority.
        """
        if not isinstance(request, Mapping):
            raise WatchdogBridgeError("verification request must be an object")
        action_id = request.get("action_id")
        if action_id is not None:
            if not isinstance(action_id, str) or ACTION_ID.fullmatch(action_id) is None:
                raise WatchdogBridgeError("verification action id is malformed")
            if (self.state_root / "control/host-supersession-receipts" / (action_id + ".json")).is_file():
                return self.verify_host_supersession(
                    action_id,
                    expected_task_id=request.get("expected_task_id"),
                    expected_stage=request.get("expected_stage"),
                    require_owner_stopped=request.get("require_owner_stopped") is True,
                )
            name = action_id + ".json"
            observation = None
            for directory in ("control/host-terminal-observations", "control/repair-terminal-observations", "control/host-observations"):
                try:
                    observation = read_owned_json(self.state_root / directory / name)
                    break
                except FileNotFoundError:
                    continue
            if observation is None or observation.get("action_id") != action_id:
                raise WatchdogBridgeError("verification has no native action observation")
            if observation.get("effects_verified") is not True or observation.get("status") not in {"succeeded", "failed", "blocked"} or not observation.get("receipt"):
                raise WatchdogBridgeError("action effects are not terminally verified")
            return {"schema": "wwl.watchdog.verified-action@1", "action_id": action_id, "status": observation["status"], "receipt": observation["receipt"],
                    "effects_verified": True, "completion_authority": False, "task_retry_authority": False}
        if request.get("run_id") != self.run_id:
            raise WatchdogBridgeError("verification request identity differs")
        snapshot = self.observe(request)
        verdict = snapshot.get("current_tree_completion")
        if not isinstance(verdict, Mapping):
            verdict = self._completion_verification(snapshot)
        return {**dict(verdict), "snapshot_accepted_tree": snapshot.get("accepted_tree", "")}

    def _seal_published_action(self, action: Mapping[str, Any], *, status: str) -> None:
        """Bind the published recovery to the loop ledger before the effect returns.

        Missing loop state still records the action identity. A present loop
        state must already contain that action and its charged counters.
        """
        state_path = self.state_root / "watchdog" / "state.json"
        loop_state = read_owned_json(state_path) if state_path.is_file() else None
        incident = _require_loop_state(loop_state, action, self.run_id) if loop_state is not None else None
        accounting = _accounting_from_loop(loop_state, incident) if incident is not None and loop_state is not None else None
        markers = hold_markers(self.state_root)

        def mutate(journal: dict[str, Any]) -> None:
            if accounting is not None:
                if journal["loop_policy_digest"] and journal["loop_policy_digest"] != loop_state["policy_digest"]:
                    raise WatchdogBridgeError("restarted loop policy digest differs")
                journal["loop_policy_digest"] = loop_state["policy_digest"]
                journal["required_task_ids"] = list(dict.fromkeys([*journal["required_task_ids"], *loop_state["required_task_ids"]]))
            journal["sealed_hold_markers"] = list(dict.fromkeys([*journal["sealed_hold_markers"], *markers]))
            prior = journal["actions"].get(action["action_id"])
            record = {
                "action": dict(action), "status": status, "incident_id": action.get("incident_id", ""),
                "request_identity": identity(action), "hold_markers": markers,
                "attempts": None if accounting is None else accounting["attempts"],
                "cooldown_until": None if accounting is None else accounting["cooldown_until"],
                "campaign_actions": None if accounting is None else accounting["campaign_actions"],
                "campaign_repairs": None if accounting is None else accounting["campaign_repairs"],
            }
            if prior is not None:
                if prior.get("request_identity") != record["request_identity"]:
                    raise WatchdogBridgeError("action id reused with a different request")
                if accounting is not None and prior.get("attempts") is not None:
                    _require_accounting(loop_state, incident, prior)
                    record["attempts"] = {key: max(prior["attempts"][key], accounting["attempts"][key]) for key in ATTEMPT_KEYS}
                    record["cooldown_until"] = max(float(prior["cooldown_until"]), accounting["cooldown_until"])
                    record["campaign_actions"] = max(prior["campaign_actions"], accounting["campaign_actions"])
                    record["campaign_repairs"] = max(prior["campaign_repairs"], accounting["campaign_repairs"])
            journal["actions"][action["action_id"]] = record

        update_restart_journal(self.state_root, self.run_id, mutate)

    def _record_stop_hold(self, markers: list[str]) -> None:
        """Remember a pause that arrived while a recovery journal already exists.

        Observation does not create a journal and does not delete the marker.
        """
        if not markers or not _journal_file(self.state_root).is_file():
            return

        def mutate(journal: dict[str, Any]) -> None:
            journal["sealed_hold_markers"] = list(dict.fromkeys([*journal["sealed_hold_markers"], *markers]))

        update_restart_journal(self.state_root, self.run_id, mutate)

    def resume_after_restart(self) -> dict[str, Any]:
        """Reload the sealed loop and the supervisor journal without charging again.

        A new bridge is the restarted supervisor. The loop state file is the
        restarted loop. Deduplication, cooldown, attempt counts and stop/hold
        markers must still match the seal.
        """
        journal = load_restart_journal(self.state_root, self.run_id)
        pending = [record for record in journal["actions"].values() if record.get("status") == "inflight" and isinstance(record.get("action"), dict)]
        if not pending:
            raise WatchdogBridgeError("restart resume has no in-flight recovery")
        try:
            loop_state = read_owned_json(self.state_root / "watchdog" / "state.json")
        except FileNotFoundError:
            raise WatchdogBridgeError("restarted loop state is missing") from None
        if journal.get("loop_policy_digest") and loop_state.get("policy_digest") != journal["loop_policy_digest"]:
            raise WatchdogBridgeError("restarted loop policy digest differs")
        required = journal.get("required_task_ids") or []
        if required and not set(required) <= set(loop_state.get("required_task_ids") or []):
            raise WatchdogBridgeError("restart pruned admitted tasks")
        current_markers = hold_markers(self.state_root)
        missing = [name for name in journal.get("sealed_hold_markers", []) if name not in current_markers]
        if missing:
            raise WatchdogBridgeError("stop/hold did not survive restart")
        resumed = []
        for record in pending:
            action = record["action"]
            incident = _require_loop_state(loop_state, action, self.run_id)
            _require_accounting(loop_state, incident, record)
            claim_path = self._incident_path(action)
            if claim_path is None:
                raise WatchdogBridgeError("durable incident claim differs after restart")
            claim = read_owned_json(claim_path)
            if claim.get("run_id") != self.run_id or claim.get("incident_id") != action.get("incident_id") or claim.get("action_id") != action["action_id"] or claim.get("request_identity") != identity(action):
                raise WatchdogBridgeError("durable incident claim differs after restart")
            resumed.append({
                "action_id": action["action_id"], "incident_id": action.get("incident_id", ""), "kind": action.get("kind"),
                "attempt_accounting": {**_attempt_map(incident), "actions": loop_state["actions"], "repairs": loop_state["repairs"]},
                "cooldown_until": _finite_number(incident.get("next_action_at")),
            })
        return {
            "schema": "wwl.watchdog.restart-resume@1", "run_id": self.run_id, "inflight": resumed,
            "hold_markers": current_markers, "deduplicated": True, "attempts_refunded": False,
            "cooldown_reset": False, "hold_cleared": False, "loop_restarted": True,
            "supervisor_restarted": True, "completion_authority": False,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--loop-script", type=Path)
    parser.add_argument("--dispatch-timeout", type=float, default=50)
    parser.add_argument("operation", choices=("ready", "observe", "dispatch", "verify", "guardian-start-check"))
    args = parser.parse_args(argv)
    try:
        if args.operation == "guardian-start-check":
            if args.loop_script is None:
                raise WatchdogBridgeError("guardian start check requires the registered loop script")
            guardian = OptionalSystemdLoopGuardian(repository=args.repository, state_root=args.state_root, run_id=args.run_id, loop_script=args.loop_script)
            result = guardian.start_check()
            print(json.dumps(result, sort_keys=True, allow_nan=False))
            return 0 if result.get("may_start") is True else 2
        raw = sys.stdin.buffer.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise WatchdogBridgeError("request exceeds its bound")
        request = json.loads(raw)
        if not isinstance(request, dict):
            raise WatchdogBridgeError("request must be a JSON object")
        bridge = WorldwideWatchdogBridge(repository=args.repository, state_root=args.state_root, run_id=args.run_id)
        if args.operation == "dispatch":
            result = bridge.dispatch(request, timeout_seconds=args.dispatch_timeout)
        else:
            result = getattr(bridge, args.operation)(request)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"schema": "wwl.watchdog.bridge-error@1", "error": str(exc), "completion_authority": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
