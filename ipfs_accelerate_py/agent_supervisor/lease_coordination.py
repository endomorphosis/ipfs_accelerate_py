"""Lease-safe Profile G adapters and coordination for accelerator daemon lanes.

The coordinator deliberately uses SQLite rather than process-local locks: bundle
supervisors are separate processes and an accepted lease must be visible to all
of them.  Every mutating operation checks the accepted claim and fencing token
inside an immediate transaction, so an expired worker cannot publish progress
or a terminal receipt after a takeover.
"""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .task_identity import canonical_bundle_identity

MIN_LEASE_MS = 5_000
MAX_LEASE_MS = 300_000
PROVIDER_VERSION = "3.2.0"


class LeaseError(RuntimeError):
    """Base error for lease protocol failures."""

    code = "G_CLAIM_CONFLICT"


class LeaseConflictError(LeaseError):
    """Raised when another non-expired claim owns the task."""


class LeaseExpiredError(LeaseError):
    """Raised when a caller's accepted lease is no longer current."""

    code = "G_LEASE_EXPIRED"


class StaleFencingTokenError(LeaseError):
    """Raised when execution uses a superseded fencing token."""

    code = "G_CLAIM_CONFLICT"


def canonical_profile_g_bytes(value: Any) -> bytes:
    """Encode canonical DAG-JSON-compatible bytes and reject floats."""

    def check(item: Any) -> None:
        if item is None or isinstance(item, str | bool | int):
            return
        if isinstance(item, float):
            raise ValueError("Profile G artifacts cannot contain floats")
        if isinstance(item, list):
            for child in item:
                check(child)
            return
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ValueError("Profile G object keys must be strings")
            for child in item.values():
                check(child)
            return
        raise ValueError(f"unsupported Profile G value: {type(item).__name__}")

    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def profile_g_cid(value: Any) -> str:
    """Return CIDv1 DAG-JSON/sha2-256 without requiring an optional codec package."""

    digest = hashlib.sha256(canonical_profile_g_bytes(value)).digest()
    # CIDv1 + dag-json (0x0129 varint) + sha2-256 multihash.
    raw = b"\x01\xa9\x02\x12\x20" + digest
    return "b" + base64.b32encode(raw).decode("ascii").rstrip("=").lower()


def _link(value: Any) -> str:
    """Create a content link for adapter inputs that are not already artifacts."""

    return profile_g_cid(value)


def adapt_goal_bundle(bundle: Mapping[str, Any], *, created_at_ms: int | None = None) -> dict[str, Any]:
    """Adapt one objective bundle payload into a canonical Goal/Subgoal/TaskSpec chain."""

    now = int(time.time() * 1000) if created_at_ms is None else int(created_at_ms)
    bundle_key = str(bundle.get("bundle_key") or "objective/general")
    correlation = str(bundle.get("correlation_id") or bundle_key)[:128]
    owner_did = str(bundle.get("owner_did") or "did:web:ipfs-accelerate.local")
    canonical_identity = canonical_bundle_identity(bundle)
    objective = {
        "bundle_key": bundle_key,
        "source_todo": str(bundle.get("source_todo") or ""),
        "task_ids": sorted(
            str(item.get("task_id"))
            for item in bundle.get("tasks", [])
            if isinstance(item, Mapping) and item.get("task_id")
        ),
    }
    objective_cid = _link(objective)
    policy_cid = str(bundle.get("policy_cid") or _link({"policy": "accelerator-daemon-lane-v1"}))
    goal = {
        "schema": "mcp++/profile-g/goal@1",
        "created_at_ms": now,
        "parents": [],
        "correlation_id": correlation,
        "owner_did": owner_did,
        "objective_cid": objective_cid,
        "policy_cid": policy_cid,
        "parent_goal_cids": [],
        "labels": sorted({"accelerator", "daemon-lane"}),
    }
    goal_cid = profile_g_cid(goal)
    subgoal = {
        "schema": "mcp++/profile-g/subgoal@1",
        "created_at_ms": now,
        "parents": [goal_cid],
        "correlation_id": correlation,
        "goal_cid": goal_cid,
        "parent_subgoal_cid": None,
        "objective_cid": objective_cid,
        "decomposition_method": "objective-bundle-v1",
        "decomposer_cid": _link({"adapter": "ipfs_accelerate_py.agent_supervisor"}),
        "selection_cid": None,
    }
    subgoal_cid = profile_g_cid(subgoal)
    template_cid = _link({"tool": "codex.todo_bundle", "bundle_key": bundle_key})
    plan = {
        "schema": "mcp++/profile-g/plan-branch@1",
        "created_at_ms": now,
        "parents": [subgoal_cid],
        "correlation_id": correlation,
        "subgoal_cid": subgoal_cid,
        "candidate_input_cids": [_link(dict(bundle))],
        "task_template_cids": [template_cid],
        "evaluator_cid": _link({"evaluator": "objective-bundle-priority-v1"}),
        "score_millionths": 1_000_000,
        "explanation_cid": _link({"reason": "selected generated objective bundle"}),
    }
    plan_cid = profile_g_cid(plan)
    selection = {
        "schema": "mcp++/profile-g/plan-selection@1",
        "created_at_ms": now,
        "parents": [plan_cid],
        "correlation_id": correlation,
        "subgoal_cid": subgoal_cid,
        "plan_branch_cid": plan_cid,
        "selector_did": owner_did,
        "proof_cid": str(bundle.get("proof_cid") or _link({"proof": bundle_key})),
        "policy_decision_cid": str(bundle.get("policy_decision_cid") or _link({"decision": "allow"})),
        "reason_cid": _link({"reason": "bundle emitted by accepted objective graph"}),
    }
    selection_cid = profile_g_cid(selection)
    # Subgoal.selection_cid remains null because making both immutable objects
    # point at one another would create an impossible content-addressed cycle.
    # PlanSelection is the authoritative selected-branch record.
    task = {
        "schema": "mcp++/profile-g/task@1",
        "created_at_ms": now,
        "parents": [selection_cid],
        "correlation_id": correlation,
        "subgoal_cid": subgoal_cid,
        "plan_branch_cid": plan_cid,
        "selection_cid": selection_cid,
        "interface_cid": _link({"interface": "codex.todo_bundle@1"}),
        "input_cid": _link(dict(bundle)),
        "tool": "codex.todo_bundle",
        "dependency_task_cids": [],
        "idempotency_key": canonical_identity.semantic_fingerprint[:32],
        "canonical_task_key": canonical_identity.canonical_task_key,
        "canonical_task_cid": canonical_identity.canonical_task_cid,
        "resource_class": str(bundle.get("resource_class") or "cpu-small"),
        "deadline_ms": int(bundle.get("deadline_ms") or now + 86_400_000),
        "expected_value_millionths": int(bundle.get("expected_value_millionths") or 500_000),
        "max_attempts": int(bundle.get("max_attempts") or 3),
        "execution_mode": "idempotent",
    }
    task_spec_cid = profile_g_cid(task)
    artifacts = {profile_g_cid(item): item for item in (goal, subgoal, plan, selection, task)}
    return {
        "goal": goal,
        "goal_cid": goal_cid,
        "subgoal": subgoal,
        "subgoal_cid": subgoal_cid,
        "plan_branch": plan,
        "plan_branch_cid": plan_cid,
        "selection": selection,
        "selection_cid": selection_cid,
        "task": task,
        "task_cid": task_spec_cid,
        "task_spec_cid": task_spec_cid,
        "canonical_task_key": canonical_identity.canonical_task_key,
        "canonical_task_cid": canonical_identity.canonical_task_cid,
        "artifacts": artifacts,
    }


@dataclass(frozen=True)
class LeaseGrant:
    task_cid: str
    goal_cid: str
    subgoal_cid: str
    claim_cid: str
    resolution_cid: str
    claimant_did: str
    logical_epoch: int
    fencing_token: int
    lease_expires_at_ms: int
    attempt: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LeaseCoordinator:
    """Durable accepted-lease registry for independent daemon processes."""

    def __init__(self, path: str | Path, *, clock_ms: Callable[[], int] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1000))
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                  cid TEXT PRIMARY KEY, kind TEXT NOT NULL, payload_json TEXT NOT NULL,
                  created_at_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                  task_cid TEXT PRIMARY KEY, goal_cid TEXT NOT NULL, subgoal_cid TEXT NOT NULL,
                  task_id TEXT NOT NULL, bundle_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS leases (
                  task_cid TEXT PRIMARY KEY, claim_cid TEXT NOT NULL, resolution_cid TEXT NOT NULL,
                  claimant_did TEXT NOT NULL, logical_epoch INTEGER NOT NULL,
                  fencing_token INTEGER NOT NULL, expires_at_ms INTEGER NOT NULL,
                  attempt INTEGER NOT NULL, state TEXT NOT NULL, started_at_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS token_history (
                  task_cid TEXT NOT NULL, fencing_token INTEGER NOT NULL,
                  PRIMARY KEY(task_cid, fencing_token)
                );
                CREATE TABLE IF NOT EXISTS heartbeats (
                  heartbeat_cid TEXT PRIMARY KEY, task_cid TEXT NOT NULL, claimant_did TEXT NOT NULL,
                  fencing_token INTEGER NOT NULL, observed_at_ms INTEGER NOT NULL,
                  expires_at_ms INTEGER NOT NULL, capacity_millionths INTEGER NOT NULL,
                  payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS receipts (
                  receipt_cid TEXT PRIMARY KEY, task_cid TEXT NOT NULL, goal_cid TEXT NOT NULL,
                  subgoal_cid TEXT NOT NULL, claim_cid TEXT NOT NULL, fencing_token INTEGER NOT NULL,
                  payload_json TEXT NOT NULL
                );
                """
            )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> LeaseCoordinator:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _put_artifact(self, connection: sqlite3.Connection, kind: str, payload: Mapping[str, Any]) -> str:
        body = dict(payload)
        cid = profile_g_cid(body)
        connection.execute(
            "INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?)",
            (cid, kind, canonical_profile_g_bytes(body).decode("utf-8"), int(body.get("created_at_ms") or self._clock_ms())),
        )
        return cid

    def register_bundle(self, bundle: Mapping[str, Any], *, created_at_ms: int | None = None) -> dict[str, Any]:
        embedded = bundle.get("profile_g")
        adapted = dict(embedded) if isinstance(embedded, Mapping) and embedded.get("task_cid") else adapt_goal_bundle(bundle, created_at_ms=created_at_ms)
        canonical_identity = canonical_bundle_identity(bundle)
        canonical_task_cid = str(adapted.get("canonical_task_cid") or canonical_identity.canonical_task_cid)
        task_spec_cid = str(adapted.get("task_spec_cid") or adapted.get("task_cid") or "")
        adapted["canonical_task_key"] = str(
            adapted.get("canonical_task_key") or canonical_identity.canonical_task_key
        )
        adapted["canonical_task_cid"] = canonical_task_cid
        adapted["task_spec_cid"] = task_spec_cid
        # Coordination is keyed by semantic execution identity. The immutable
        # Profile-G TaskSpec CID remains available separately for provenance.
        adapted["task_cid"] = canonical_task_cid
        with self._lock, self._connection:
            for cid, artifact in adapted["artifacts"].items():
                self._connection.execute(
                    "INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?)",
                    (cid, str(artifact["schema"]), canonical_profile_g_bytes(artifact).decode("utf-8"), artifact["created_at_ms"]),
                )
            task_id = ",".join(
                str(item.get("task_id")) for item in bundle.get("tasks", []) if isinstance(item, Mapping)
            ) or str(bundle.get("bundle_key") or adapted["task_cid"])
            self._connection.execute(
                "INSERT OR IGNORE INTO tasks VALUES(?,?,?,?,?)",
                (canonical_task_cid, adapted["goal_cid"], adapted["subgoal_cid"], task_id, json.dumps(dict(bundle), sort_keys=True)),
            )
        return adapted

    def _expire(self, connection: sqlite3.Connection, task_cid: str, now: int) -> None:
        row = connection.execute(
            "SELECT * FROM leases WHERE task_cid=? AND state='accepted' AND expires_at_ms<=?",
            (task_cid, now),
        ).fetchone()
        if row is None:
            return
        connection.execute("UPDATE leases SET state='expired' WHERE task_cid=?", (task_cid,))
        resolution = self._resolution_payload(row, outcome="expired", now=now)
        self._put_artifact(connection, "ClaimResolution", resolution)

    def claim(
        self,
        task_cid: str,
        claimant_did: str,
        *,
        requested_lease_ms: int = 60_000,
        now_ms: int | None = None,
    ) -> LeaseGrant:
        duration = int(requested_lease_ms)
        if not MIN_LEASE_MS <= duration <= MAX_LEASE_MS:
            raise ValueError(f"lease duration must be in [{MIN_LEASE_MS}, {MAX_LEASE_MS}]")
        now = self._clock_ms() if now_ms is None else int(now_ms)
        with self._lock:
            conn = self._connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                task = conn.execute("SELECT * FROM tasks WHERE task_cid=?", (task_cid,)).fetchone()
                if task is None:
                    raise KeyError(f"unknown task CID: {task_cid}")
                self._expire(conn, task_cid, now)
                active = conn.execute(
                    "SELECT * FROM leases WHERE task_cid=? AND state='accepted' AND expires_at_ms>?",
                    (task_cid, now),
                ).fetchone()
                if active is not None:
                    if active["claimant_did"] == claimant_did:
                        grant = self._grant(active, task)
                        conn.commit()
                        return grant
                    raise LeaseConflictError(f"task is leased by {active['claimant_did']}")
                terminal = conn.execute("SELECT state FROM leases WHERE task_cid=?", (task_cid,)).fetchone()
                if terminal is not None and terminal["state"] == "completed":
                    raise LeaseConflictError("task already has a successful terminal receipt")
                prior = conn.execute(
                    "SELECT COALESCE(MAX(fencing_token),0), COALESCE(MAX(logical_epoch),0), COALESCE(MAX(attempt),0) FROM leases WHERE task_cid=?",
                    (task_cid,),
                ).fetchone()
                token, epoch, attempt = int(prior[0]) + 1, int(prior[1]) + 1, int(prior[2]) + 1
                claim = self._claim_payload(task, claimant_did, epoch, attempt, duration, now)
                claim_cid = self._put_artifact(conn, "TaskClaim", claim)
                expires = now + duration
                resolution = {
                    "schema": "mcp++/profile-g/claim-resolution@1",
                    "created_at_ms": now,
                    "parents": [claim_cid],
                    "correlation_id": claim["correlation_id"],
                    "task_cid": task_cid,
                    "logical_epoch": epoch,
                    "considered_claim_cids": [claim_cid],
                    "accepted_claim_cid": claim_cid,
                    "outcome": "accepted",
                    "fencing_token": token,
                    "lease_expires_at_ms": expires,
                    "attestation_cids": [],
                    "quorum_policy_cid": _link({"policy": "local-atomic-claim-v1"}),
                    "policy_decision_cid": claim["policy_decision_cid"],
                    "coordination_receipt_cid": None,
                    "retry_not_before_ms": 0,
                    "resolver_did": "did:web:ipfs-accelerate.local",
                }
                resolution_cid = self._put_artifact(conn, "ClaimResolution", resolution)
                conn.execute(
                    "INSERT OR REPLACE INTO leases VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (task_cid, claim_cid, resolution_cid, claimant_did, epoch, token, expires, attempt, "accepted", now),
                )
                conn.execute("INSERT INTO token_history VALUES(?,?)", (task_cid, token))
                conn.commit()
                return LeaseGrant(task_cid, task["goal_cid"], task["subgoal_cid"], claim_cid, resolution_cid, claimant_did, epoch, token, expires, attempt)
            except Exception:
                conn.rollback()
                raise

    def _claim_payload(self, task: sqlite3.Row, claimant: str, epoch: int, attempt: int, duration: int, now: int) -> dict[str, Any]:
        bundle = json.loads(task["bundle_json"])
        correlation = str(bundle.get("correlation_id") or bundle.get("bundle_key") or task["task_id"])[:128]
        return {
            "schema": "mcp++/profile-g/task-claim@1", "created_at_ms": now, "parents": [],
            "correlation_id": correlation, "task_cid": task["task_cid"],
            "proposal_cid": _link({"proposal": task["task_cid"], "epoch": epoch}),
            "claimant_did": claimant, "record_cid": _link({"peer": claimant}), "logical_epoch": epoch,
            "requested_lease_ms": duration, "risk_bucket": 0, "capability_fit_millionths": 1_000_000,
            "expected_finish_ms": now + duration, "proof_cid": str(bundle.get("proof_cid") or _link({"proof": claimant})),
            "policy_decision_cid": str(bundle.get("policy_decision_cid") or _link({"decision": "allow"})), "attempt": attempt,
        }

    def _grant(self, lease: sqlite3.Row, task: sqlite3.Row | None = None) -> LeaseGrant:
        task = task or self._connection.execute("SELECT * FROM tasks WHERE task_cid=?", (lease["task_cid"],)).fetchone()
        assert task is not None
        return LeaseGrant(lease["task_cid"], task["goal_cid"], task["subgoal_cid"], lease["claim_cid"], lease["resolution_cid"], lease["claimant_did"], lease["logical_epoch"], lease["fencing_token"], lease["expires_at_ms"], lease["attempt"])

    def _current(self, connection: sqlite3.Connection, grant: LeaseGrant, now: int) -> sqlite3.Row:
        self._expire(connection, grant.task_cid, now)
        row = connection.execute("SELECT * FROM leases WHERE task_cid=?", (grant.task_cid,)).fetchone()
        if row is None or row["state"] != "accepted" or row["expires_at_ms"] <= now:
            raise LeaseExpiredError("lease has expired or was released")
        if row["claim_cid"] != grant.claim_cid or row["claimant_did"] != grant.claimant_did:
            raise StaleFencingTokenError("claim is no longer accepted")
        if row["fencing_token"] != grant.fencing_token:
            raise StaleFencingTokenError("fencing token is stale")
        return row

    def validate(self, grant: LeaseGrant, *, now_ms: int | None = None) -> LeaseGrant:
        now = self._clock_ms() if now_ms is None else int(now_ms)
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._current(self._connection, grant, now)
                result = self._grant(row)
                self._connection.commit()
                return result
            except Exception:
                self._connection.rollback()
                raise

    def renew(self, grant: LeaseGrant, *, requested_lease_ms: int = 60_000, now_ms: int | None = None) -> LeaseGrant:
        duration = int(requested_lease_ms)
        if not MIN_LEASE_MS <= duration <= MAX_LEASE_MS:
            raise ValueError(f"lease duration must be in [{MIN_LEASE_MS}, {MAX_LEASE_MS}]")
        now = self._clock_ms() if now_ms is None else int(now_ms)
        with self._lock:
            conn = self._connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._current(conn, grant, now)
                expires = now + duration
                claim = json.loads(conn.execute("SELECT payload_json FROM artifacts WHERE cid=?", (grant.claim_cid,)).fetchone()[0])
                event = {
                    "schema": "mcp++/profile-g/claim-resolution@1", "created_at_ms": now,
                    "parents": [row["resolution_cid"]], "correlation_id": claim["correlation_id"],
                    "task_cid": grant.task_cid, "logical_epoch": grant.logical_epoch,
                    "considered_claim_cids": [grant.claim_cid], "accepted_claim_cid": grant.claim_cid,
                    "outcome": "accepted", "fencing_token": grant.fencing_token,
                    "lease_expires_at_ms": expires, "attestation_cids": [],
                    "quorum_policy_cid": _link({"policy": "local-atomic-claim-v1"}),
                    "policy_decision_cid": claim["policy_decision_cid"], "coordination_receipt_cid": None,
                    "retry_not_before_ms": 0, "resolver_did": "did:web:ipfs-accelerate.local",
                }
                renewal_cid = self._put_artifact(conn, "ClaimResolution", event)
                conn.execute("UPDATE leases SET expires_at_ms=?, resolution_cid=? WHERE task_cid=?", (expires, renewal_cid, grant.task_cid))
                conn.commit()
                return LeaseGrant(grant.task_cid, grant.goal_cid, grant.subgoal_cid, grant.claim_cid, renewal_cid, grant.claimant_did, grant.logical_epoch, grant.fencing_token, expires, grant.attempt)
            except Exception:
                conn.rollback()
                raise

    def _resolution_payload(self, row: sqlite3.Row, *, outcome: str, now: int) -> dict[str, Any]:
        claim = json.loads(self._connection.execute("SELECT payload_json FROM artifacts WHERE cid=?", (row["claim_cid"],)).fetchone()[0])
        return {
            "schema": "mcp++/profile-g/claim-resolution@1", "created_at_ms": now, "parents": [row["resolution_cid"]],
            "correlation_id": claim["correlation_id"], "task_cid": row["task_cid"], "logical_epoch": row["logical_epoch"],
            "considered_claim_cids": [row["claim_cid"]], "accepted_claim_cid": None, "outcome": outcome,
            "fencing_token": row["fencing_token"], "lease_expires_at_ms": None, "attestation_cids": [],
            "quorum_policy_cid": _link({"policy": "local-atomic-claim-v1"}), "policy_decision_cid": claim["policy_decision_cid"],
            "coordination_receipt_cid": None, "retry_not_before_ms": 0, "resolver_did": "did:web:ipfs-accelerate.local",
        }

    def release(self, grant: LeaseGrant, *, now_ms: int | None = None) -> str:
        now = self._clock_ms() if now_ms is None else int(now_ms)
        with self._lock:
            conn = self._connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._current(conn, grant, now)
                cid = self._put_artifact(conn, "ClaimResolution", self._resolution_payload(row, outcome="released", now=now))
                conn.execute("UPDATE leases SET state='released', resolution_cid=? WHERE task_cid=?", (cid, grant.task_cid))
                conn.commit()
                return cid
            except Exception:
                conn.rollback()
                raise

    def heartbeat(self, grant: LeaseGrant, *, capacity_millionths: int, ttl_ms: int = 15_000, now_ms: int | None = None) -> dict[str, Any]:
        now = self._clock_ms() if now_ms is None else int(now_ms)
        capacity = int(capacity_millionths)
        if not 0 <= capacity <= 1_000_000:
            raise ValueError("capacity_millionths must be in [0, 1000000]")
        with self._lock:
            conn = self._connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._current(conn, grant, now)
                payload = {"schema": "ipfs_accelerate_py/daemon-heartbeat@1", "created_at_ms": now,
                           "task_cid": grant.task_cid, "goal_cid": grant.goal_cid, "subgoal_cid": grant.subgoal_cid,
                           "claim_cid": grant.claim_cid, "claimant_did": grant.claimant_did,
                           "fencing_token": grant.fencing_token, "capacity_millionths": capacity,
                           "expires_at_ms": min(grant.lease_expires_at_ms, now + int(ttl_ms))}
                cid = self._put_artifact(conn, "DaemonHeartbeat", payload)
                conn.execute("INSERT OR REPLACE INTO heartbeats VALUES(?,?,?,?,?,?,?,?)",
                             (cid, grant.task_cid, grant.claimant_did, grant.fencing_token, now, payload["expires_at_ms"], capacity, json.dumps(payload, sort_keys=True)))
                conn.commit()
                return {**payload, "heartbeat_cid": cid}
            except Exception:
                conn.rollback()
                raise

    def receipt(
        self, grant: LeaseGrant, *, status: str, output: Mapping[str, Any] | None = None,
        failure_class: str = "none", started_at_ms: int | None = None, now_ms: int | None = None,
    ) -> dict[str, Any]:
        now = self._clock_ms() if now_ms is None else int(now_ms)
        status = str(status)
        if status not in {"succeeded", "failed", "cancelled", "compensated"}:
            raise ValueError("invalid receipt status")
        with self._lock:
            conn = self._connection
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._current(conn, grant, now)
                output_cid = _link(dict(output or {})) if output is not None else None
                payload = {
                    "schema": "mcp++/profile-g/task-receipt@1", "created_at_ms": now, "parents": [row["resolution_cid"]],
                    "correlation_id": json.loads(conn.execute("SELECT payload_json FROM artifacts WHERE cid=?", (grant.claim_cid,)).fetchone()[0])["correlation_id"],
                    "task_cid": grant.task_cid, "claim_cid": grant.claim_cid, "resolution_cid": row["resolution_cid"],
                    "fencing_token": grant.fencing_token, "profile_b_receipt_cid": _link({"task": grant.task_cid, "token": grant.fencing_token, "finished": now}),
                    "output_cid": output_cid, "status": status, "failure_class": failure_class, "attempt": grant.attempt,
                    "started_at_ms": int(started_at_ms if started_at_ms is not None else row["started_at_ms"]), "finished_at_ms": now,
                    "resource_use_cid": _link({"heartbeats": self._heartbeat_count(conn, grant)}), "provider": "ipfs_accelerate_py",
                    "provider_version": PROVIDER_VERSION, "next_state": "complete" if status == "succeeded" else "ready",
                }
                if status == "succeeded" and output is None:
                    raise ValueError("successful receipt requires output")
                cid = self._put_artifact(conn, "TaskReceipt", payload)
                conn.execute("INSERT INTO receipts VALUES(?,?,?,?,?,?,?)", (cid, grant.task_cid, grant.goal_cid, grant.subgoal_cid, grant.claim_cid, grant.fencing_token, json.dumps(payload, sort_keys=True)))
                terminal = "completed" if status == "succeeded" else "released"
                conn.execute("UPDATE leases SET state=? WHERE task_cid=?", (terminal, grant.task_cid))
                conn.commit()
                return {"receipt_cid": cid, "goal_cid": grant.goal_cid, "subgoal_cid": grant.subgoal_cid, "receipt": payload}
            except Exception:
                conn.rollback()
                raise

    @staticmethod
    def _heartbeat_count(connection: sqlite3.Connection, grant: LeaseGrant) -> int:
        return int(connection.execute("SELECT COUNT(*) FROM heartbeats WHERE task_cid=? AND fencing_token=?", (grant.task_cid, grant.fencing_token)).fetchone()[0])

    def active_lease(self, task_cid: str, *, now_ms: int | None = None) -> LeaseGrant | None:
        now = self._clock_ms() if now_ms is None else int(now_ms)
        with self._lock, self._connection:
            self._expire(self._connection, task_cid, now)
            row = self._connection.execute("SELECT * FROM leases WHERE task_cid=? AND state='accepted' AND expires_at_ms>?", (task_cid, now)).fetchone()
            return self._grant(row) if row is not None else None

    def list_receipts(self, task_cid: str) -> list[dict[str, Any]]:
        rows = self._connection.execute("SELECT * FROM receipts WHERE task_cid=? ORDER BY rowid", (task_cid,)).fetchall()
        return [{"receipt_cid": row["receipt_cid"], "goal_cid": row["goal_cid"], "subgoal_cid": row["subgoal_cid"], "receipt": json.loads(row["payload_json"])} for row in rows]

    def get_artifact(self, cid: str) -> dict[str, Any] | None:
        """Return a stored coordination artifact by CID."""

        row = self._connection.execute("SELECT payload_json FROM artifacts WHERE cid=?", (cid,)).fetchone()
        return json.loads(row[0]) if row is not None else None


@dataclass(frozen=True)
class LeasedQueuedTask:
    """A queue task paired with the only grant that authorizes its execution."""

    task: Any
    grant: LeaseGrant


class LeaseQueueBridge:
    """Bind the legacy DuckDB queue claim lifecycle to Profile G leases.

    Queue ownership alone is never execution authority.  A bridge claim is
    returned only after the embedded canonical TaskSpec has an accepted lease;
    a conflicting queue claim is immediately returned to the queue.
    """

    def __init__(
        self,
        queue: Any,
        coordinator: LeaseCoordinator,
        *,
        worker_id: str,
        claimant_did: str,
        lease_ms: int = 60_000,
    ) -> None:
        self.queue = queue
        self.coordinator = coordinator
        self.worker_id = worker_id
        self.claimant_did = claimant_did
        self.lease_ms = lease_ms

    def claim_next(self, *, supported_task_types: list[str] | None = None) -> LeasedQueuedTask | None:
        task = self.queue.claim_next(worker_id=self.worker_id, supported_task_types=supported_task_types)
        if task is None:
            return None
        payload = task.payload if isinstance(task.payload, Mapping) else {}
        try:
            adapted = self.coordinator.register_bundle(payload)
            grant = self.coordinator.claim(
                adapted["task_cid"],
                self.claimant_did,
                requested_lease_ms=self.lease_ms,
            )
        except Exception:
            self.queue.release(task_id=task.task_id, worker_id=self.worker_id, reason="profile-g lease not accepted")
            raise
        return LeasedQueuedTask(task=task, grant=grant)

    def renew(self, leased: LeasedQueuedTask) -> LeasedQueuedTask:
        return LeasedQueuedTask(
            task=leased.task,
            grant=self.coordinator.renew(leased.grant, requested_lease_ms=self.lease_ms),
        )

    def release(self, leased: LeasedQueuedTask, *, reason: str = "released") -> bool:
        self.coordinator.release(leased.grant)
        return bool(self.queue.release(task_id=leased.task.task_id, worker_id=self.worker_id, reason=reason))

    def complete(
        self,
        leased: LeasedQueuedTask,
        *,
        status: str,
        output: Mapping[str, Any] | None = None,
        failure_class: str = "none",
    ) -> dict[str, Any]:
        receipt = self.coordinator.receipt(
            leased.grant,
            status=status,
            output=output,
            failure_class=failure_class,
        )
        queue_status = "completed" if status == "succeeded" else "failed"
        self.queue.complete(
            task_id=leased.task.task_id,
            status=queue_status,
            result={"profile_g": receipt},
            error=None if status == "succeeded" else failure_class,
        )
        return receipt


__all__ = [
    "LeaseConflictError", "LeaseCoordinator", "LeaseError", "LeaseExpiredError", "LeaseGrant",
    "LeaseQueueBridge", "LeasedQueuedTask",
    "MAX_LEASE_MS", "MIN_LEASE_MS", "StaleFencingTokenError", "adapt_goal_bundle",
    "canonical_profile_g_bytes", "profile_g_cid",
]
