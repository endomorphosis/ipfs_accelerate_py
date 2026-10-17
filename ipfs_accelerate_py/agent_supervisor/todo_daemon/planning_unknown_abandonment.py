"""Explicit administrative disposition of a retained UNKNOWN planning attempt.

The separately registered host administrator supplies an exact native preimage
under stopped-store custody. This operation records that decision; it does not
prove model nonreachability, callback closure, cancellation or missing effects.
Only the later genuine daemon can validate recovered execution/coordinator
state and settle local authority. No provider row or retry policy is changed.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

from ..task_sources.control_plane_contracts import canonical_json_bytes, content_identity

OPERATION = "planning_unknown_abandonment_decide/v1"
CLIENT_ID = "planning-unknown-abandonment-administrator"
EVIDENCE_KIND = "planning_unknown_abandonment_decision_v1"
AUTHORITY_SCHEMA = "planning-unknown-abandonment-authority-context@1"
SNAPSHOT_SCHEMA = "planning-unknown-abandonment-native-preimage@1"
AUDIT_SCHEMA = "planning-unknown-administrative-abandonment-decision@1"
FAILURE_KIND = "administrative_operator_stop_outcome_unknown"
MAX_BYTES = 64 * 1024 * 1024
IDENTITY_FIELDS = ("task_cid", "attempt_id", "attempt_number", "claim_id", "lease_id",
                   "owner_session_id", "fencing_token", "fence_epoch")
DECISION = {
    "decision": "administratively_abandon_local_execution_authority",
    "provider_outcome": "unknown", "possible_external_effects": "unverified",
    "callback_closed": False, "remote_cancellation_proved": False,
    "nonreachability_asserted": False,
    "native_execution_settlement_performed_by_this_operation": False,
    "automatic_retry_admitted": False, "completion_authority": False,
}
CONTEXT_KEYS = {"schema", "context_cid", "task_cid", "namespace", "expected_task_revision",
    "logical_store_id", "stores", "snapshot_pin", "source_pins", "immutable_pins",
    "closed_store_pins", "stopped_processes", "operator_decision", "registered_owner_identity",
    "registered_creator", "custody_artifacts"}
SNAPSHOT_KEYS = {"schema", "task", "intent_task", "canonical_population", "attempt", "phases",
                 "providers", "effects", "coordination", "historical_unknowns"}


class PlanningUnknownAbandonmentError(ValueError):
    """An administrative decision or its exact native preimage differs."""


def need(value: Any, reason: str) -> None:
    if not value:
        raise PlanningUnknownAbandonmentError(reason)


def plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    need(value is None or type(value) in (str, int, bool, float), "non-JSON native scalar")
    return value


def decode(raw: str | bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate registered JSON key")
            result[key] = value
        return result
    need(len(raw) <= MAX_BYTES, "registered JSON exceeds bound")
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _value: (_ for _ in ()).throw(PlanningUnknownAbandonmentError("non-finite JSON")))


def digest(value: Any) -> str:
    # This hashes an opaque, full native JSON preimage, whose internal body can
    # contain finite numeric observations. Typed command/context/envelope IDs
    # still use the unchanged control-plane canonical_json_bytes contract.
    raw = json.dumps(plain(value), sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def read_pin(pin: Mapping[str, Any]) -> bytes:
    need(type(pin) is dict and set(pin) == {"path", "bytes", "sha256"}, "pin fields differ")
    need(type(pin["bytes"]) is int and 0 <= pin["bytes"] <= MAX_BYTES
         and type(pin["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]), "pin bounds differ")
    path = Path(pin["path"])
    need(path.is_absolute() and path.resolve(strict=True) == path, "pinned path is aliased")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        need(stat.S_ISREG(before.st_mode) and before.st_uid in (0, os.geteuid())
             and before.st_nlink == 1 and not before.st_mode & 0o002
             and before.st_size == pin["bytes"], "pinned file custody differs")
        chunks = []
        remaining = pin["bytes"] + 1
        while remaining:
            block = os.read(descriptor, min(remaining, 65536))
            if not block: break
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
        witness = lambda row: tuple(getattr(row, field) for field in fields)
        need(len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["sha256"]
             and witness(before) == witness(os.fstat(descriptor)) == witness(path.lstat()), "pinned input changed")
        return raw
    finally:
        os.close(descriptor)


def stopped_processes(rows: Any) -> None:
    need(type(rows) is list and 1 <= len(rows) <= 128, "stopped owner population differs")
    current_boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    for row in rows:
        need(type(row) is dict and set(row) == {"pid", "start_time_ticks", "boot_id"}
             and type(row["pid"]) is int and row["pid"] > 1
             and type(row["start_time_ticks"]) is int and row["start_time_ticks"] > 0
             and type(row["boot_id"]) is str and bool(row["boot_id"]), "stopped process identity differs")
        if row["boot_id"] != current_boot: continue
        try:
            process = Path(f"/proc/{row['pid']}/stat").read_text()
        except FileNotFoundError:
            continue
        need(int(process.rsplit(")", 1)[1].split()[19]) != row["start_time_ticks"],
             "registered original local process is still present")


def unknown_row(row: Mapping[str, Any], *, task_cid: str, identity: Mapping[str, Any] | None = None) -> None:
    need(type(row) is dict and set(row) == {"invocation_id", "attempt_id", "task_cid", "idempotency_key",
         "owner_session_id", "recorded_at_ms", "result_json"}, "raw UNKNOWN row fields differ")
    need(row["task_cid"] == task_cid and type(row["recorded_at_ms"]) is int
         and type(row["result_json"]) is str, "raw UNKNOWN identity differs")
    outcome = decode(row["result_json"])
    native_identity_fields = {"attempt_id", "task_cid", "claim_id", "lease_id", "fencing_token", "fence_epoch"}
    need(type(outcome) is dict and set(outcome) == native_identity_fields | {"schema", "callback_state", "provider_effect_state"}
         and outcome.get("schema") == "database-portal-callback-intent@1"
         and outcome.get("callback_state") == "started_outcome_unknown"
         and outcome.get("provider_effect_state") == "unknown_may_have_started",
         "registered provider row is not an unresolved native callback intent")
    need(row["idempotency_key"] == "provider:" + row["attempt_id"]
         and outcome["attempt_id"] == row["attempt_id"] and outcome["task_cid"] == row["task_cid"],
         "raw callback idempotency/identity crosswalk differs")
    if identity is not None:
        need(row["owner_session_id"] == identity["owner_session_id"]
             and all(type(outcome[k]) is type(identity[k]) and outcome[k] == identity[k] for k in native_identity_fields),
             "native UNKNOWN callback owner/claim/lease/fence differs")


def verify_custody(context: Mapping[str, Any], snapshot: Mapping[str, Any], *, check_current_file: bool) -> None:
    """Bind named native launch/terminal/status/maintenance artifacts.

    These immutable files remain host-administrator attestations under the
    registered custody assumption; they are not remote provider cancellation
    proofs. The published maintenance/status components keep their own limits.
    """
    artifacts = context["custody_artifacts"]
    required = {"launch_request", "native_owner_receipt", "master_terminal", "status_binding",
                "status_observation", "maintenance_binding", "maintenance_receipt"}
    need(type(artifacts) is dict and set(artifacts) == required, "required named native custody crosswalk absent")
    decoded = {name: decode(read_pin(pin)) for name, pin in artifacts.items()}
    request, owner, terminal = (decoded[name] for name in ("launch_request", "native_owner_receipt", "master_terminal"))
    need(request.get("schema") == "legal-ir-native-orchestration-request@1"
         and owner.get("schema") == "legal-ir-native-orchestration-receipt@1"
         and owner.get("request_sha256") == artifacts["launch_request"]["sha256"]
         and owner.get("native_stop_verified") is True and owner.get("owned_group_quiescent") is True
         and owner.get("cleanup_pending") is False and owner.get("completion_authority") is False,
         "original finite native owner is not durably stopped")
    fence = owner.get("native_fence_proof", {})
    need(terminal.get("schema") == "ipfs_accelerate_py.agent_supervisor.multi-supervisor-terminal@1"
         and terminal.get("all_trees_fenced") is True and terminal.get("task_completion_authority") is False
         and fence.get("all_trees_fenced") is True
         and terminal.get("terminal_receipt_cid") == content_identity({k: v for k, v in terminal.items() if k != "terminal_receipt_cid"})
         and all(terminal.get(key) == fence.get(key) for key in ("master_pid", "active_binding_cid",
             "run_started_at_epoch_nanoseconds", "terminal_receipt_cid"))
         and request.get("native_terminal_receipt") == artifacts["master_terminal"]["path"]
         and request.get("board_namespace") == terminal.get("label") == context["namespace"]
         and Path(context["stores"]["execution"]).is_relative_to(Path(artifacts["master_terminal"]["path"]).parent),
         "original native launch/terminal/store crosswalk differs")
    processes = context["stopped_processes"]
    need(any(row["pid"] == owner.get("owner_pid") for row in processes)
         and any(row["pid"] == owner.get("child_pid") and str(row["start_time_ticks"]) == owner.get("child_birth_marker") for row in processes)
         and any(row["pid"] == fence.get("master_pid") and row["start_time_ticks"] == fence.get("master_start_time_ticks") for row in processes),
         "original owner/child/master process birth crosswalk differs")
    from .native_execution_status_owner import NativeExecutionStatusBinding, OBSERVATION_SCHEMA
    from .native_execution_wal_maintenance import NativeExecutionWalMaintenanceBinding, RECEIPT_SCHEMA
    status_binding = NativeExecutionStatusBinding.from_dict(decoded["status_binding"])
    observation = decoded["status_observation"]
    expected_attempt = {key: value for key, value in snapshot["attempt"].items() if key not in {"schema", "interface"}}
    expected_provider = [{**{key: value for key, value in row.items() if key != "result_json"},
                          "result": decode(row["result_json"])} for row in snapshot["providers"]]
    need(status_binding.execution_path == context["stores"]["execution"]
         and status_binding.control_store_id == context["logical_store_id"]
         and status_binding.task_cid == context["task_cid"]
         and all(getattr(status_binding, key) == snapshot["attempt"][key] for key in IDENTITY_FIELDS)
         and status_binding.control_revision == context["expected_task_revision"]
         and status_binding.expected_control_status == "in_progress"
         and observation.get("schema") == OBSERVATION_SCHEMA
         and observation.get("binding_cid") == status_binding.binding_cid
         and observation.get("backend_read_only") is True and observation.get("original_process_impersonated") is False
         and observation.get("attempt") == expected_attempt
         and observation.get("phase_history") == snapshot["phases"]
         and observation.get("provider_records") == expected_provider and observation.get("effect_records") == []
         and observation.get("phase_history_complete") is True
         and observation.get("provider_records_complete") is True and observation.get("effect_records_complete") is True,
         "published native status does not corroborate registered execution preimage")
    maintenance_binding = NativeExecutionWalMaintenanceBinding.from_dict(decoded["maintenance_binding"])
    maintenance = decoded["maintenance_receipt"]
    need(str(maintenance_binding.execution_path) == context["stores"]["execution"]
         and all(getattr(maintenance_binding.status_binding, key) == snapshot["attempt"][key] for key in IDENTITY_FIELDS)
         and maintenance.get("schema") == RECEIPT_SCHEMA
         and maintenance.get("binding_cid") == maintenance_binding.binding_cid
         and maintenance.get("receipt_cid") == digest({k: v for k, v in maintenance.items() if k != "receipt_cid"})
         and maintenance.get("physical_maintenance_only") is True
         and maintenance.get("logical_rows_preserved") is True and maintenance.get("all_metadata_preserved") is True
         and maintenance.get("connection_close_verified") is True and maintenance.get("original_locks_released") is True
         and maintenance.get("wal_after") == "absent" and maintenance.get("unknown_not_settled") is True
         and maintenance.get("completion_authority") is False and maintenance.get("automatic_retry_admitted") is False
         and maintenance.get("execution_file_identity") == observation.get("execution_file_identity")
         == dict(status_binding.execution_file_identity), "published physical maintenance/status generation crosswalk differs")
    backups = maintenance.get("retained_backups")
    need(type(backups) is dict and set(backups) == {"database", "wal"}, "retained original DB/WAL backups missing")
    for value in backups.values():
        path = Path(maintenance["backup_directory"]) / value["name"]
        read_pin({"path": str(path), "bytes": value["identity"]["size"], "sha256": value["sha256"]})
    if check_current_file:
        info = Path(context["stores"]["execution"]).stat()
        need(dict(device=info.st_dev, inode=info.st_ino, size=info.st_size, mtime_ns=info.st_mtime_ns)
             == dict(status_binding.execution_file_identity), "current execution generation differs from published native status")


def load_context(context: Mapping[str, Any], *, check_closed_stores: bool) -> dict[str, Any]:
    need(type(context) is dict and set(context) == CONTEXT_KEYS and context["schema"] == AUTHORITY_SCHEMA,
         "administrative context is not a closed record")
    need(len(canonical_json_bytes(context)) <= 64 * 1024
         and context["context_cid"] == content_identity({key: value for key, value in context.items() if key != "context_cid"}),
         "administrative context identity differs")
    need(type(context["expected_task_revision"]) is int and context["expected_task_revision"] >= 1
         and all(type(context[key]) is str and context[key] for key in ("task_cid", "namespace", "logical_store_id")),
         "registered canonical scope differs")
    need(type(context["registered_owner_identity"]) is dict and context["registered_owner_identity"].get("server_id")
         and context["registered_owner_identity"].get("store_id") == context["logical_store_id"]
         and type(context["registered_creator"]) is dict
         and set(context["registered_creator"]) == {"pid", "process_birth_id"}
         and type(context["registered_creator"]["pid"]) is int
         and type(context["registered_creator"]["process_birth_id"]) is str
         and context["registered_creator"]["process_birth_id"], "registered canonical owner/creator identity absent")
    decision = context["operator_decision"]
    need(type(decision) is dict and set(decision) == set(DECISION) | {"reason", "unknown_effect_risk_accepted"}
         and all(type(decision[key]) is type(value) and decision[key] == value for key, value in DECISION.items())
         and decision["unknown_effect_risk_accepted"] is True
         and type(decision["reason"]) is str and 1 <= len(decision["reason"]) <= 4096,
         "explicit UNKNOWN disposition/risk acknowledgement differs")
    stores = context["stores"]
    need(type(stores) is dict and set(stores) == {"canonical", "execution", "coordination"}
         and len(set(stores.values())) == 3, "three distinct physical stores required")
    for value in stores.values():
        path = Path(value)
        need(path.is_absolute() and path.resolve(strict=True) == path, "registered physical store is aliased")
    need(type(context["source_pins"]) is list and 1 <= len(context["source_pins"]) <= 128
         and type(context["immutable_pins"]) is list and 1 <= len(context["immutable_pins"]) <= 256,
         "source/custody input population differs")
    for pin in context["source_pins"] + context["immutable_pins"]:
        need(pin["path"] not in stores.values() and pin["path"] not in {s + ".wal" for s in stores.values()},
             "mutable native store is incorrectly registered as immutable")
        read_pin(pin)
    prefix = Path(__file__).resolve().parents[1]
    required_sources = {str(prefix / relative) for relative in (
        "todo_daemon/planning_unknown_abandonment.py", "todo_daemon/planning_unknown_abandonment_settlement.py",
        "todo_daemon/implementation_daemon.py", "task_sources/typed_state_owner.py",
        "task_sources/database_task_source.py", "task_sources/intent_repository.py",
        "task_sources/control_plane_contracts.py", "task_sources/duckdb_state.py",
        "merge/database_coordination.py", "runtime/quack_state_server.py")}
    required_sources |= {str(prefix / relative) for relative in (
        "todo_daemon/native_execution_status_owner.py", "todo_daemon/native_execution_wal_maintenance.py")}
    need(required_sources <= {pin["path"] for pin in context["source_pins"]},
         "administrative source closure omits an actual native implementation")
    snapshot = decode(read_pin(context["snapshot_pin"]))
    need(type(snapshot) is dict and set(snapshot) == SNAPSHOT_KEYS and snapshot["schema"] == SNAPSHOT_SCHEMA,
         "registered native preimage fields differ")
    task = snapshot["task"]
    need(task["task_cid"] == context["task_cid"] and task["status"] == "in_progress"
         and task["revision"] == context["expected_task_revision"]
         and task["body"].get("board_namespace") == context["namespace"], "registered native control preimage differs")
    need(snapshot["intent_task"]["task_cid"] == task["task_cid"]
         and snapshot["intent_task"]["created_at"] and snapshot["intent_task"]["identity"],
         "full native Intent identity/creation preimage is required")
    attempt = snapshot["attempt"]
    need(attempt["task_cid"] == task["task_cid"] and attempt["status"] == "running"
         and attempt["committed_phase"] == "context" and attempt["revision"] == 2
         and attempt["finished_at_ms"] is None and type(attempt["attempt_number"]) is int
         and attempt["attempt_number"] == 2 and snapshot["effects"] == [],
         "original running context-phase execution preimage differs")
    need(type(snapshot["providers"]) is list and len(snapshot["providers"]) == 1,
         "one exact latest-attempt UNKNOWN row required")
    unknown_row(snapshot["providers"][0], task_cid=task["task_cid"], identity=attempt)
    need(snapshot["providers"][0]["attempt_id"] == attempt["attempt_id"], "UNKNOWN belongs to another attempt")
    need(type(snapshot["historical_unknowns"]) is list and len(snapshot["historical_unknowns"]) == 1,
         "historical UNKNOWN population differs")
    for history in snapshot["historical_unknowns"]:
        need(type(history) is dict and set(history) == {"row", "attempt", "store_pins"}
             and history["row"]["attempt_id"] != attempt["attempt_id"], "historical UNKNOWN binding differs")
        need(history["attempt"]["attempt_number"] == 1 and history["attempt"]["status"] == "failed",
             "historical UNKNOWN requires the previously settled original attempt")
        unknown_row(history["row"], task_cid=task["task_cid"], identity=history["attempt"])
        need(type(history["store_pins"]) is list and 1 <= len(history["store_pins"]) <= 2,
             "historical DB/WAL custody required")
        for pin in history["store_pins"]:
            need(pin["path"] not in stores.values(), "historical store overlaps current mutable authority")
            read_pin(pin)
    coordination = snapshot["coordination"]
    need(type(coordination) is dict and set(coordination) == {"claim", "lease", "attempt", "previous_attempts"}
         and type(coordination["previous_attempts"]) is list and len(coordination["previous_attempts"]) == 1,
         "complete native predecessor coordination scope required")
    predecessor = coordination["previous_attempts"][0]
    need(predecessor["attempt_number"] == 1 and predecessor["task_cid"] == task["task_cid"]
         and predecessor["attempt_id"] == snapshot["historical_unknowns"][0]["row"]["attempt_id"],
         "historical ordinal/UNKNOWN crosswalk differs")
    need(all(predecessor[key] == snapshot["historical_unknowns"][0]["attempt"][key]
             for key in ("task_cid", "attempt_id", "attempt_number", "owner_session_id", "fencing_token", "fence_epoch")),
         "historical execution/coordination owner/fence crosswalk differs")
    need(type(snapshot["phases"]) is list and len(snapshot["phases"]) == 2
         and [row["phase"] for row in snapshot["phases"]] == ["claimed", "context"], "original phase history differs")
    stopped_processes(context["stopped_processes"])
    verify_custody(context, snapshot, check_current_file=check_closed_stores)
    need(type(context["closed_store_pins"]) is dict and set(context["closed_store_pins"]) == {"execution", "coordination"},
         "registered stopped sidecar pins differ")
    for role, pin in context["closed_store_pins"].items():
        need(pin["path"] == stores[role], "stopped sidecar pin has another physical path")
        if check_closed_stores:
            read_pin(pin)
            need(not Path(stores[role] + ".wal").exists(),
                 "WAL requires separately qualified published physical maintenance before audit")
    return snapshot


def scope(context: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = load_context(context, check_closed_stores=False)
    return {"task_cid": context["task_cid"], "context_cid": context["context_cid"],
            "expected_task_revision": context["expected_task_revision"],
            "canonical_store_id": context["stores"]["canonical"], "logical_store_id": context["logical_store_id"],
            "abandonment_scope_cid": content_identity({"context_cid": context["context_cid"],
                "attempt_id": snapshot["attempt"]["attempt_id"], "claim_id": snapshot["attempt"]["claim_id"]})}


def evidence_id(*, task_cid: str, attempt_id: str, claim_id: str, control_revision: int) -> str:
    return content_identity({"namespace": EVIDENCE_KIND, "task_cid": task_cid,
        "attempt_id": attempt_id, "claim_id": claim_id, "control_expected_revision": control_revision})


def audit_body(context: Mapping[str, Any], snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": AUDIT_SCHEMA, **DECISION, "context_cid": context["context_cid"],
        "authority_context": dict(context), "task_cid": context["task_cid"],
        "attempt_id": snapshot["attempt"]["attempt_id"], "claim_id": snapshot["attempt"]["claim_id"],
        "original_control_revision": context["expected_task_revision"],
        "registered_preimage_digest": digest(snapshot),
        "observation_attestation": "exact_native_preimage_registered_by_host_administrator",
        "recovered_execution_verified_by_this_audit": False,
        "unknown_provider_rows_digest": digest(snapshot["providers"]),
        "historical_unknown_rows_digest": digest(snapshot["historical_unknowns"])}


def canonical_population(source: Any) -> dict[str, Any]:
    result = {}
    # Fixed native relations only; no relation/SQL comes from the caller.
    statements = {"tasks": "SELECT * FROM tasks ORDER BY task_cid LIMIT 1025",
        "goals": "SELECT * FROM goals ORDER BY goal_cid LIMIT 1025",
        "goal_edges": "SELECT * FROM goal_edges ORDER BY parent_goal_cid,child_goal_cid,edge_kind LIMIT 4097",
        "plans": "SELECT * FROM plans ORDER BY plan_cid LIMIT 1025",
        "objectives": "SELECT * FROM objectives ORDER BY objective_id LIMIT 1025",
        "task_dependencies": "SELECT * FROM task_dependencies ORDER BY task_cid,dependency_task_cid LIMIT 4097",
        "task_outputs": "SELECT * FROM task_outputs ORDER BY task_cid,ordinal LIMIT 4097",
        "task_acceptance": "SELECT * FROM task_acceptance ORDER BY task_cid,ordinal LIMIT 4097",
        "task_validations": "SELECT * FROM task_validations ORDER BY task_cid,ordinal LIMIT 4097"}
    with source.intent._connection(write=False) as connection:
        for name, sql in statements.items():
            cursor = connection.execute(sql)
            rows = cursor.fetchall()
            need(len(rows) < (4097 if name in {"goal_edges", "task_dependencies", "task_outputs", "task_acceptance", "task_validations"}
                              else 1025), "canonical population exceeds bound")
            columns = [item[0] for item in getattr(cursor, "description", ())]
            need(not rows or isinstance(rows[0], Mapping) or columns, "native row metadata unavailable")
            result[name] = [plain(dict(row) if isinstance(row, Mapping) else dict(zip(columns, row))) for row in rows]
    return result


def verify_request(payload: Mapping[str, Any], *, authority_context: Mapping[str, Any],
                   current_observation: Mapping[str, Any]) -> dict[str, Any]:
    need(type(payload) is dict and set(payload) == {"operation", "context_cid", "expected_task_revision"}
         and payload["operation"] == OPERATION and payload["context_cid"] == authority_context["context_cid"]
         and type(payload["expected_task_revision"]) is int
         and payload["expected_task_revision"] == authority_context["expected_task_revision"], "ADMIN request scope differs")
    snapshot = load_context(authority_context, check_closed_stores=True)
    need(current_observation["task"] == snapshot["task"]
         and current_observation["intent_task"] == snapshot["intent_task"]
         and current_observation["canonical_population"] == snapshot["canonical_population"],
         "current canonical source/preimage differs from registered decision")
    return audit_body(authority_context, snapshot)


def exact_audit(source: Any, *, attempt: Any, binding: Mapping[str, Any]) -> dict[str, Any]:
    identity = evidence_id(task_cid=attempt.task_cid, attempt_id=attempt.attempt_id,
        claim_id=attempt.claim_id, control_revision=binding["control_expected_revision"])
    columns = ("evidence_id", "parent_evidence_id", "task_cid", "evidence_kind", "digest", "created_at", "body_json")
    with source.intent._connection(write=False) as connection:
        cursor = connection.execute("SELECT evidence_id,parent_evidence_id,task_cid,evidence_kind,digest,created_at,body_json "
            "FROM evidence_nodes WHERE evidence_id=? AND task_cid=? AND evidence_kind=? LIMIT 2",
            [identity, attempt.task_cid, EVIDENCE_KIND])
        rows = cursor.fetchall()
        need(len(rows) == 1, "exact durable UNKNOWN abandonment audit missing or ambiguous")
        audit = dict(rows[0]) if isinstance(rows[0], Mapping) else dict(zip(columns, rows[0]))
        body = decode(audit["body_json"])
        context = body["authority_context"]
        snapshot = load_context(context, check_closed_stores=False)
        need(body == audit_body(context, snapshot) and audit["parent_evidence_id"] == ""
             and audit["body_json"] == canonical_json_bytes(body).decode()
             and audit["digest"] == content_identity(body)
             and context["task_cid"] == attempt.task_cid
             and context["expected_task_revision"] == binding["control_expected_revision"]
             and all(snapshot["attempt"][key] == getattr(attempt, key) for key in IDENTITY_FIELDS),
             "durable audit/context/attempt identity differs")
        cursor = connection.execute("SELECT * FROM domain_events WHERE task_cid=? AND event_type=? ORDER BY global_sequence LIMIT 1025",
            [attempt.task_cid, "intent.evidence_recorded"])
        native = cursor.fetchall()
        need(len(native) <= 1024, "native ADMIN event population exceeds bound")
        names = [item[0] for item in getattr(cursor, "description", ())]
        need(not native or isinstance(native[0], Mapping) or names, "native event metadata unavailable")
        events = [dict(row) if isinstance(row, Mapping) else dict(zip(names, row)) for row in native]
    matches = [(event, decode(event["body_json"])) for event in events
               if decode(event["body_json"]).get("subject_id") == identity]
    need(len(matches) == 1, "durable native ADMIN event missing or ambiguous")
    event, envelope = matches[0]
    from ..task_sources.intent_repository import INTENT_EVENT_SCHEMA
    need(event["attempt_id"] == "" and envelope == {"schema": INTENT_EVENT_SCHEMA,
        "event_type": "intent.evidence_recorded", "subject_id": identity,
        "recorded_at": event["recorded_at"], "owner_id": context["registered_owner_identity"]["server_id"],
        "body": {"evidence_id": identity, "parent_evidence_id": "", "task_cid": attempt.task_cid,
                 "evidence_kind": EVIDENCE_KIND, "digest": audit["digest"], "body": body,
                 "created_at": str(audit["created_at"]), "revision": 0}}
        and event["event_id"] == content_identity({key: (envelope if key == "body" else event[key])
            for key in ("stream_id", "sequence", "global_sequence", "event_type", "body")}),
        "durable canonical ADMIN event/content identity differs")
    return {**audit, "body": body, "event": event, "context": context, "snapshot": snapshot}


def failure_payload(audit: Mapping[str, Any], binding: Mapping[str, Any]) -> str:
    return digest({"failure_kind": FAILURE_KIND, "audit_evidence_id": audit["evidence_id"],
        "audit_body_digest": audit["digest"], "context_cid": audit["context"]["context_cid"],
        "original_control_binding": dict(binding),
        "unknown_provider_rows_digest": audit["body"]["unknown_provider_rows_digest"],
        "historical_unknown_rows_digest": audit["body"]["historical_unknown_rows_digest"]})
