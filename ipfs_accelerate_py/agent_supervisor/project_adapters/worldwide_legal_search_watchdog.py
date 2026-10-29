"""File-protocol bridge to the single-owner worldwide legal-search runtime.

This module deliberately never opens a task database. The native owner emits
atomic status/readiness projections and consumes immutable, idempotent control
requests. Missing mutation bindings remain false in readiness; observing a
healthy process cannot create repair or completion authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time
from typing import Any, Mapping

MAX_BYTES = 1_048_576
ACTION_ID = re.compile(r"[0-9a-f]{64}")


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


class WorldwideWatchdogBridge:
    def __init__(self, *, repository: Path, state_root: Path, run_id: str, freshness_seconds: float = 120, clock=time.time, sleep=time.sleep, host_lifecycle=None, statvfs=None):
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
        return value

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
            return {**snapshot, "blocked_kind": "resource_quota", "resource_pressure": resources}
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
        snapshot = {**snapshot, "hold": any(os.path.lexists(path) for path in holds)
                    or (bool(snapshot.get("hold")) and not released)}
        if reviews:
            snapshot["hold_reviews"] = reviews
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
        return snapshot

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
        if action.get("run_id") != self.run_id or action.get("kind") not in {"reconcile", "recover", "repair"}:
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
        capability = {"reconcile": "reconcile", "recover": "lifecycle_restart", "repair": "repair_dispatch"}[action["kind"]]
        if ready["bindings"].get(capability) is not True:
            return self._refuse(action, "native_" + capability + "_not_bound")
        snapshot = self.observe({"run_id": self.run_id, "policy_id": ready["policy_id"]})
        # A stopped owner cannot refresh its cached hold projection. Respect
        # current operator controls before publishing any new operation.
        if snapshot.get("hold") or any((self.state_root / name).exists() for name in ("HOLD", "STOP", "OPERATOR_STOP")):
            return self._refuse(action, "native_pause_or_hold")
        if action.get("authority_revision") != snapshot.get("authority_revision") or action.get("accepted_tree") != snapshot.get("accepted_tree"):
            return self._refuse(action, "native_observation_revision_changed")
        host = snapshot.get("registered_host_lifecycle")
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
        return self._wait_result(action, path, timeout_seconds)

    def _host_recover(self, action: Mapping[str, Any], path: Path) -> dict[str, Any]:
        publish_once(self.state_root / "control/host-before" / path.name, self.host_lifecycle.probe())
        publish_once(path, action)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dispatch-timeout", type=float, default=50)
    parser.add_argument("operation", choices=("ready", "observe", "dispatch"))
    args = parser.parse_args(argv)
    try:
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
