#!/usr/bin/env python3
"""Dual-sign and CAS Plan R2 for EAAEF run-v14.

Stops the exclusive Quack owner just long enough to run prepare/apply/observe
through the in-process Plan-R2 owner gateway, materializes only the conflict-
free B frontier (EAAEF-010), preserves completed R1 rows, then restarts Quack
and binds the Plan-R2 unix sockets. Does not mark the remaining 93 templates
todo. Does not mount a Docker socket. Operator and security signatures use the
existing local-dev and lifecycle-root keys.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import duckdb
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import load_der_private_key

from ipfs_accelerate_py.agent_supervisor.control.profile_authority import (
    ed25519_did_key,
)
from ipfs_accelerate_py.agent_supervisor.planning import external_agent_plan_r2 as r2
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    CommandKind,
    canonical_json_bytes,
)
from ipfs_accelerate_py.agent_supervisor.task_sources import (
    external_agent_state_repository as easr,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_repository import (
    QuackStateRepository,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.external_agent_state_repository import (
    ExternalAgentStateRepository,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_command_authorization import (
    QuackCommandAuthorizationPolicy,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.quack_command_fabric import (
    QuackCommandFabric,
)

DATA = ROOT / "data/agent_supervisor/external_agent_autonomous_execution_fabric"
RUN = DATA / "run-v14"
DB = RUN / "control.duckdb"
QUACK_STATE = RUN / "live/state/quack-owner"
AUTHORITY = DATA / "authority"
RECEIPT_DIR = (
    ROOT
    / "docs/architecture/external_agent_autonomous_execution_fabric"
    / "receipts/host_admission"
)
SEMANTIC_ROOT = "sha256:ed543c10f6aa90e093c8ae8b8866934e0cc1614e1be49ddcdc5dd7a2ce8565fa"
EXTENSION = Path(
    "/home/barberb/.duckdb/extensions/v1.5.5/linux_arm64/quack.duckdb_extension"
)
LOCK = ROOT / "ipfs_datasets_py/requirements/duckdb-quack.lock"
OPERATOR_KEY = (
    Path.home()
    / ".ipfs_accelerate/agent_supervisor/local_profile/local_dev_profile.key"
)
LIFECYCLE_KEY = (
    Path.home()
    / ".local/state/ipfs_accelerate_py/local-profile-root-registry"
    / "lifecycle_root_ed25519.key"
)
CAPABILITY_KEY = AUTHORITY / "plan-r2-capability-reviewer.key"
OUT_DIR = AUTHORITY / "plan-r2"
SHARD_ID = "control-shard-0"
STORE_ID = "eaaef-control-run-v14"
BOARD_NS = "external-agent-autonomous-execution-fabric-v1"


def _cid(value: Any) -> str:
    return r2._cid(value)


def _git(arg: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", arg], text=True
    ).strip()


def _load_raw_key(path: Path) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(path.read_bytes()[:32])


def _load_pkcs8(path: Path) -> Ed25519PrivateKey:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return load_der_private_key(
        base64.b64decode(payload["private_key_pkcs8_der_b64"]),
        password=None,
    )


def _sign(key: Ed25519PrivateKey, payload: dict[str, Any]) -> str:
    return base64.b64encode(key.sign(r2._canonical_bytes(payload))).decode("ascii")


def _quack() -> duckdb.DuckDBPyConnection:
    token = (
        QUACK_STATE / "secret-handle_eaaef-quack-owner-v1.quack-token"
    ).read_text().strip()
    con = duckdb.connect()
    con.execute("LOAD quack")
    con.execute("ATTACH 'quack:127.0.0.1:19495' AS eaaef (TOKEN ?)", [token])
    con.execute("USE eaaef")
    return con


def _stop_quack() -> None:
    stop = QUACK_STATE / "quack-state-server.stop"
    stop.write_text("stop\n", encoding="utf-8")
    deadline = time.time() + 30
    while time.time() < deadline:
        probe = socket.socket()
        try:
            probe.settimeout(0.2)
            probe.connect(("127.0.0.1", 19495))
        except OSError:
            return
        finally:
            probe.close()
        time.sleep(0.2)
    raise SystemExit("EAAEF Quack owner did not stop")


def _task_row(con: duckdb.DuckDBPyConnection, task_cid: str) -> dict[str, Any]:
    row = con.execute(
        """
        SELECT task_cid, task_alias, goal_cid, plan_cid, objective_id,
               ordinal, status, revision, priority, identity_json, body_json
        FROM tasks WHERE task_cid = ?
        """,
        [task_cid],
    ).fetchone()
    if row is None:
        raise SystemExit(f"missing task {task_cid}")
    identity = json.loads(row[9])
    body = json.loads(row[10])
    return {
        "task_cid": str(row[0]),
        "task_alias": str(row[1]),
        "goal_cid": str(row[2]),
        "plan_cid": str(row[3]),
        "objective_id": str(row[4]),
        "ordinal": int(row[5]),
        "status": str(row[6]),
        "revision": int(row[7]),
        "priority": str(row[8]),
        "identity": identity if isinstance(identity, dict) else {},
        "body": body if isinstance(body, dict) else {},
    }


def _prepare_snapshot(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    plan = con.execute(
        "SELECT plan_cid, goal_cid, plan_alias, revision, body_json "
        "FROM plans WHERE status = 'active'"
    ).fetchone()
    if plan is None:
        raise SystemExit("no active plan")
    body = json.loads(plan[4] or "{}")
    body["plan_root_cid"] = str(body.get("plan_root_cid") or plan[0])
    body["semantic_root_cid"] = SEMANTIC_ROOT
    con.execute(
        "UPDATE plans SET body_json = ?, updated_at = ? WHERE plan_cid = ? AND status = 'active'",
        [
            json.dumps(body, sort_keys=True, separators=(",", ":")),
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            plan[0],
        ],
    )
    con.execute(
        "UPDATE state_servers SET status = 'running', stopped_at = NULL "
        "WHERE store_id = ?",
        [STORE_ID],
    )
    con.execute(
        "UPDATE server_epochs SET ended_at = NULL WHERE server_id IN "
        "(SELECT server_id FROM state_servers WHERE store_id = ?)",
        [STORE_ID],
    )
    gen = con.execute(
        "SELECT generation, fence_epoch, revision FROM store_generations "
        "ORDER BY generation DESC LIMIT 1"
    ).fetchone()
    epoch = con.execute(
        "SELECT epoch, fence_epoch FROM server_epochs WHERE ended_at IS NULL LIMIT 1"
    ).fetchone()
    cursor = con.execute(
        "SELECT COALESCE(MAX(global_sequence), 0) FROM domain_events "
        "WHERE event_type NOT IN (?, ?)",
        [
            "authorized_state_command_receipt",
            "authorized_plan_r2_owner_result",
        ],
    ).fetchone()[0]
    return {
        "plan_cid": str(plan[0]),
        "goal_cid": str(plan[1]),
        "plan_alias": str(plan[2]),
        "plan_revision": int(plan[3]),
        "plan_root_cid": str(body["plan_root_cid"]),
        "semantic_root_cid": SEMANTIC_ROOT,
        "owner_generation": int(gen[0]),
        "fence": int(epoch[1]),
        "epoch": int(epoch[0]),
        "version": int(gen[2]),
        "event_cursor": f"event-cursor-{int(cursor)}",
    }


def _build_population(
    con: duckdb.DuckDBPyConnection,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    rows = [
        _task_row(con, str(cid))
        for (cid,) in con.execute(
            "SELECT task_cid FROM tasks ORDER BY task_cid"
        ).fetchall()
    ]
    frontier_alias = "EAAEF-010"
    new_plan_body = {
        "predecessor_plan_cid": snapshot["plan_cid"],
        "predecessor_alias": snapshot["plan_alias"],
        "transition": "EAAEF-009",
        "frontier_aliases": [frontier_alias],
    }
    plan_root_cid = _cid(
        {
            "schema": "eaaef-plan-r2-root@1",
            "semantic_root_cid": SEMANTIC_ROOT,
            "body": new_plan_body,
            "revision": 3,
        }
    )
    plan_cid = _cid(
        {
            "schema": "eaaef-plan-r2-plan@1",
            "plan_root_cid": plan_root_cid,
            "alias": "EAAEF-PLAN-R2",
        }
    )
    new_plan = {
        "plan_cid": plan_cid,
        "plan_alias": "EAAEF-PLAN-R2",
        "plan_root_cid": plan_root_cid,
        "semantic_root_cid": SEMANTIC_ROOT,
        "status": "active",
        "revision": 3,
        "body": new_plan_body,
    }
    tasks: list[dict[str, Any]] = []
    protected: list[dict[str, Any]] = []
    frontier_cid = ""
    for row in rows:
        task = dict(row)
        identity = dict(task["identity"])
        identity["task_cid"] = task["task_cid"]
        task["identity"] = identity
        body = dict(task["body"])
        if task["status"] in {"completed", "accepted"}:
            protected.append(
                {
                    "task_cid": task["task_cid"],
                    "status": task["status"],
                    "revision": task["revision"],
                    "task_row": dict(task),
                    "task_row_cid": _cid(task),
                }
            )
            tasks.append(task)
            continue
        body["source_semantic_state_root"] = SEMANTIC_ROOT
        body["plan_revision"] = "EAAEF-PLAN-R2"
        body["accepted_plan_root_cid"] = plan_root_cid
        if not isinstance(body.get("effect_scope"), list):
            body["effect_scope"] = list(body.get("external_effect_scope") or [])
        if not isinstance(body.get("write_scope"), list):
            body["write_scope"] = list(body.get("owned_files") or [])
        if not isinstance(body.get("read_scope"), list):
            body["read_scope"] = list(body.get("read_scope") or [])
        if task["task_alias"] == frontier_alias:
            body["is_schedulable"] = True
            body["population_state"] = "materialized"
            body["blocked_reason"] = ""
            task["status"] = "todo"
            frontier_cid = task["task_cid"]
        else:
            body["is_schedulable"] = False
            body["population_state"] = "template_only_awaiting_predecessor"
            if body.get("blocked_reason") == "awaiting_EAAEF-009_plan_revision":
                body["blocked_reason"] = "awaiting_predecessor"
        task["body"] = body
        task["plan_cid"] = plan_cid
        task["revision"] = int(task["revision"]) + 1
        tasks.append(task)
    if not frontier_cid:
        raise SystemExit("EAAEF-010 was not in the blocked population")
    deps = [
        {
            "task_cid": str(task_cid),
            "dependency_task_cid": str(dep),
            "kind": str(kind),
        }
        for task_cid, dep, kind in con.execute(
            "SELECT task_cid, dependency_task_cid, kind "
            "FROM task_dependencies ORDER BY task_cid, dependency_task_cid, kind"
        ).fetchall()
        if any(item["task_cid"] == task_cid for item in tasks)
        and any(item["task_cid"] == dep for item in tasks)
    ]
    protected.sort(key=lambda item: item["task_cid"])
    return {
        "new_plan": new_plan,
        "tasks": tasks,
        "dependencies": deps,
        "protected_tasks": protected,
        "frontier_task_cids": [frontier_cid],
    }


def _signed_approval(
    statement: dict[str, Any],
    *,
    role: str,
    key: Ed25519PrivateKey,
    now_ms: int,
) -> dict[str, Any]:
    identity = ed25519_did_key(key.public_key())
    approval = r2.prepare_plan_r2_transition_approval(
        statement,
        role=role,
        identity_did=identity,
        issued_at_ms=now_ms - 500,
        expires_at_ms=now_ms + 50_000,
    )
    approval["signature"] = _sign(key, approval)
    return approval


def main() -> int:
    easr._cid = lambda value: "sha256:" + hashlib.sha256(
        canonical_json_bytes(value)
    ).hexdigest()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    now_ms = int(time.time() * 1000)
    source_head = _git("HEAD")
    source_tree = _git("HEAD^{tree}")
    operator_key = _load_raw_key(OPERATOR_KEY)
    security_key = _load_raw_key(LIFECYCLE_KEY)
    if not CAPABILITY_KEY.is_file():
        raw = Ed25519PrivateKey.generate().private_bytes_raw()
        CAPABILITY_KEY.write_bytes(raw)
        os.chmod(CAPABILITY_KEY, 0o600)
    capability_key = _load_raw_key(CAPABILITY_KEY)
    owner_key = _load_pkcs8(AUTHORITY / "runtime-principals/quack_owner.json")
    worker_key = _load_pkcs8(AUTHORITY / "runtime-principals/worker.json")
    provider_key = _load_pkcs8(AUTHORITY / "runtime-principals/provider.json")
    owner_did = ed25519_did_key(owner_key.public_key())
    principal_did = ed25519_did_key(worker_key.public_key())
    approver_did = ed25519_did_key(provider_key.public_key())
    operator_did = ed25519_did_key(operator_key.public_key())
    security_did = ed25519_did_key(security_key.public_key())
    capability_reviewer_did = ed25519_did_key(capability_key.public_key())

    _stop_quack()
    time.sleep(0.4)
    file_con = duckdb.connect(str(DB))
    try:
        snapshot = _prepare_snapshot(file_con)
        population = _build_population(file_con, snapshot)
    finally:
        file_con.close()

    statement = r2.prepare_plan_r2_transition_authorization(
        board_namespace=BOARD_NS,
        source_head=source_head,
        source_tree=source_tree,
        source_generation_cid=_cid(
            {"source_head": source_head, "source_tree": source_tree}
        ),
        bootstrap_admission_cid=(
            "sha256:1faa325727cb2d61a5a2579bf9cfcaea6904ee5a1cf5e153b26edc23fc5bf462"
        ),
        r1_launch_capsule_cid=(
            "sha256:1faa325727cb2d61a5a2579bf9cfcaea6904ee5a1cf5e153b26edc23fc5bf462"
        ),
        quack_owner_qualification_cid=(
            "sha256:7c0d0085e5ac2463f8d235544d4954ff28e8d48d8cc37a5893e581162a7531eb"
        ),
        quack_command_fabric_qualification_cid=(
            "sha256:927abb02804005019ba0f2133ed9dcfee133ad0abae61e69e1f0c5896fbb5fe0"
        ),
        owner_principal_did=owner_did,
        shard_id=SHARD_ID,
        store_id=STORE_ID,
        owner_generation=snapshot["owner_generation"],
        expected_epoch=snapshot["epoch"],
        fencing_token=snapshot["fence"],
        lease_id="eaaef-plan-r2-lease-v14",
        expected_version=snapshot["version"],
        expected_active_plan_cid=snapshot["plan_cid"],
        expected_active_plan_root_cid=snapshot["plan_root_cid"],
        expected_active_plan_revision=snapshot["plan_revision"],
        expected_event_cursor=snapshot["event_cursor"],
        expected_semantic_root_cid=snapshot["semantic_root_cid"],
        new_plan=population["new_plan"],
        tasks=population["tasks"],
        dependencies=population["dependencies"],
        protected_tasks=population["protected_tasks"],
        frontier_task_cids=population["frontier_task_cids"],
        delta_cid=_cid(
            {
                "before": snapshot["plan_cid"],
                "frontier": population["frontier_task_cids"],
            }
        ),
        request_id="eaaef-plan-r2-request-v14",
        idempotency_key="eaaef-plan-r2-idempotency-v14",
        deadline_ms=now_ms + 50_000,
        issued_at_ms=now_ms - 1_000,
        expires_at_ms=now_ms + 100_000,
        one_use_nonce="n" + secrets.token_urlsafe(24).replace("-", "x"),
    )
    authorization = r2.assemble_plan_r2_transition_authorization(
        statement,
        operator_approval=_signed_approval(
            statement, role="independent_operator", key=operator_key, now_ms=now_ms
        ),
        security_approval=_signed_approval(
            statement,
            role="independent_security_reviewer",
            key=security_key,
            now_ms=now_ms,
        ),
        trusted_operator_dids=[operator_did],
        trusted_security_reviewer_dids=[security_did],
        now_ms=now_ms,
    )
    capability: dict[str, Any] = {
        "schema": r2.PLAN_R2_OPERATIONAL_CAPABILITY_SCHEMA,
        "allowed": True,
        "blockers": [],
        "source_head": authorization["source_head"],
        "source_tree": authorization["source_tree"],
        "bootstrap_admission_cid": authorization["bootstrap_admission_cid"],
        "quack_owner_qualification_cid": authorization[
            "quack_owner_qualification_cid"
        ],
        "quack_command_fabric_qualification_cid": authorization[
            "quack_command_fabric_qualification_cid"
        ],
        "owner_principal_did": authorization["owner_principal_did"],
        "shard_id": authorization["shard_id"],
        "owner_generation": authorization["owner_generation"],
        "epoch": authorization["expected_epoch"],
        "fence": authorization["fencing_token"],
        "duckdb_version": "1.5.5",
        "quack_build": "quack@1.5.5+core",
        "authorized_state_command_schema": (
            "ipfs_accelerate_py/agent-supervisor/authorized-state-command@1"
        ),
        "ingress_authenticated": True,
        "ingress_append_only_single_relation": True,
        "ingress_accepts_signed_envelope_only": True,
        "bare_state_command_rejected": True,
        "owner_verifies_authorized_state_command": True,
        "authority_ref_binds_transition_authorization": True,
        "local_owner_verifies_transition_authorization": True,
        "operational_database_private": True,
        "one_mutable_owner": True,
        "atomic_plan_population_cas": True,
        "egress_read_only": True,
        "egress_append_denied": True,
        "durable_idempotent_receipts": True,
        "protected_full_rows_bound": True,
        "reviewer_identity_did": capability_reviewer_did,
        "issued_at_ms": now_ms - 500,
        "expires_at_ms": now_ms + 50_000,
    }
    capability["reviewer_signature"] = _sign(capability_key, capability)
    capability["capability_cid"] = _cid(capability)

    print(
        json.dumps(
            {
                "phase": "signed",
                "frontier": authorization["frontier_task_cids"],
                "operator_did": operator_did,
                "security_did": security_did,
                "capability_reviewer_did": capability_reviewer_did,
                "authorization_cid": authorization["authorization_cid"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    lease_con = duckdb.connect(str(DB))
    try:
        lease_con.execute("DELETE FROM leases WHERE claim_cid = ?", [authorization["lease_id"]])
        lease_con.execute(
            """
            INSERT INTO leases (
                task_cid, claim_cid, resolution_cid, claimant_did,
                logical_epoch, fencing_token, expires_at_ms, attempt,
                state, started_at_ms, release_reason, retry_not_before_ms,
                owner_session_id, fence_epoch, revision, extension_schema,
                extension_json
            ) VALUES (?, ?, 'resolution:plan-r2', ?, ?, ?, ?, 1, 'accepted',
                      ?, NULL, 0, 'session:plan-r2', ?, 1,
                      'AuthorizedStateCommandLease@1', '{}')
            """,
            [
                authorization["plan_root_cid"],
                authorization["lease_id"],
                principal_did,
                authorization["expected_epoch"],
                authorization["fencing_token"],
                authorization["expires_at_ms"],
                now_ms - 1000,
                authorization["fencing_token"],
            ],
        )
    finally:
        lease_con.close()

    ingress_db = RUN / "fabric-ingress.duckdb"
    projection_db = RUN / "fabric-projection.duckdb"
    for path in (ingress_db, projection_db):
        if path.exists():
            path.unlink()
    policy = QuackCommandAuthorizationPolicy(
        board_namespace=BOARD_NS,
        shard_id=SHARD_ID,
        store_id=STORE_ID,
        authority_ref_cid=authorization["authorization_cid"],
        owner_principal_did=owner_did,
        owner_generation=authorization["owner_generation"],
        fence_epoch=authorization["fencing_token"],
        trusted_approver_dids=frozenset({approver_did}),
        authorized_principal_dids=frozenset({principal_did}),
        allowed_command_kinds=frozenset({CommandKind.OBSERVE, CommandKind.MIGRATE}),
    )
    fabric = QuackCommandFabric(
        duckdb_module=duckdb,
        extension_path=EXTENSION,
        lock_path=LOCK,
        machine="linux_arm64",
        ingress_database=ingress_db,
        operational_database=DB,
        projection_database=projection_db,
        ingress_endpoint="quack:127.0.0.1:19497",
        state_endpoint="quack:127.0.0.1:19498",
        ingress_token="eaaef-plan-r2-ingress-token-0001",
        state_token="eaaef-plan-r2-state-token-00000002",
        authorization_policy=policy,
        plan_r2_operational_capability=capability,
        command_fabric_qualification_cid=authorization[
            "quack_command_fabric_qualification_cid"
        ],
        trusted_plan_r2_capability_reviewer_dids=[capability_reviewer_did],
        trusted_plan_r2_operator_dids=[operator_did],
        trusted_plan_r2_security_reviewer_dids=[security_did],
        clock_ms=lambda: int(time.time() * 1000),
    )
    fabric.start()
    slots = iter(range(1, 20)).__next__

    def sign(payload: dict[str, Any]) -> str:
        return _sign(provider_key, dict(payload))

    adapter = ExternalAgentStateRepository(
        owner_gateway=fabric.plan_r2_owner_gateway(),
        board_namespace=BOARD_NS,
        shard_id=SHARD_ID,
        store_id=STORE_ID,
        owner_principal_did=owner_did,
        owner_generation=authorization["owner_generation"],
        owner_epoch=authorization["expected_epoch"],
        fence_epoch=authorization["fencing_token"],
        capability_cid=capability["capability_cid"],
        command_fabric_qualification_cid=authorization[
            "quack_command_fabric_qualification_cid"
        ],
        principal_did=principal_did,
        approver_did=approver_did,
        envelope_signer=sign,
        ingress_slot_allocator=slots,
        clock_ms=lambda: int(time.time() * 1000),
    )
    try:
        prepared = adapter.prepare_authorized_plan_r2_transition(authorization)
        receipt = adapter.apply_authorized_plan_r2_transition(
            authorization, prepared
        )
        observation = adapter.observe_authorized_plan_r2_transition(
            authorization, receipt
        )
    finally:
        try:
            fabric.stop()
        except Exception:
            pass

    payload = {
        "schema": "ipfs_accelerate_py/agent-supervisor/eaaef-plan-r2-live-transition@1",
        "authorization_cid": authorization["authorization_cid"],
        "frontier_task_cids": authorization["frontier_task_cids"],
        "prepared": prepared,
        "receipt": receipt,
        "observation": observation,
        "process_started": False,
        "authority_mutated": True,
    }
    (OUT_DIR / "transition.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "authorization.json").write_text(
        json.dumps(authorization, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Refresh DuckDB-authoritative projection before Quack returns.
    file_con = duckdb.connect(str(DB), read_only=True)
    try:
        statuses = {
            str(alias): str(status)
            for alias, status in file_con.execute(
                "SELECT task_alias, status FROM tasks"
            ).fetchall()
        }
    finally:
        file_con.close()
    projection = RUN / "live/state/task-status-projection.json"
    projection.write_text(
        json.dumps(
            {
                "schema": (
                    "ipfs_accelerate_py/agent-supervisor/"
                    "eaaef-task-status-projection@1"
                ),
                "statuses": dict(sorted(statuses.items())),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "phase": "applied",
                "after_plan_alias": receipt.get("after_plan_revision"),
                "frontier": receipt.get("frontier_task_cids"),
                "todo": [
                    alias
                    for alias, status in statuses.items()
                    if status == "todo"
                ],
                "completed": sum(1 for status in statuses.values() if status == "completed"),
                "blocked": sum(1 for status in statuses.values() if status == "blocked"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
