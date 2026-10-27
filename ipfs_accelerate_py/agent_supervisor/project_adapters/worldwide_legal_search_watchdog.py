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
    def __init__(self, *, repository: Path, state_root: Path, run_id: str, freshness_seconds: float = 120, clock=time.time, sleep=time.sleep, host_lifecycle=None):
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
        config = self.repository / "config/worldwide_supervisor_host.json"
        if self.host_lifecycle is None and config.is_file():
            from .worldwide_legal_search_host_lifecycle import RegisteredSystemdLifecycle
            self.host_lifecycle = RegisteredSystemdLifecycle(repository=self.repository, state_root=self.state_root, run_id=self.run_id)

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
        return value

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
        snapshot["action_observations"] = list(observations.values())
        return snapshot

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
