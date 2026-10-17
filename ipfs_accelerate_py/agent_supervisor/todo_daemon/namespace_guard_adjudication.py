"""Explicit administrator adjudication of one retained namespace-guard attempt.

This is an intentionally singleton historical source profile. It is not a
generic UNKNOWN-callback recovery rule. The typed owner must register the
historical custody context; transport callers supply only its content identity
and a task revision. Diagnostic frames and missing artifacts confer no authority.
The existing UNKNOWN invocation, spent ordinal and every earlier receipt remain
unchanged. The audit authorizes a separate native failure settlement, never a
provider replay, task completion or successor attempt.
"""
from __future__ import annotations

import ast
import base64
from collections.abc import Mapping
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any

OPERATION = "namespace_guard_nonreachability_adjudicate/v1"
EVIDENCE_KIND = "namespace_guard_nonreachability_adjudicated_v1"
AUTHORITY_SCHEMA = "namespace-guard-adjudication-authority-context@1"
RECEIPT_SCHEMA = "namespace-guard-nonreachability-administrative-adjudication@1"
SOURCE_PROFILE = "ordinary-db-portal-namespace-guard-20261016-run05@1"
TASK_CID = "baguqeeraqk2ln7d3plwfonnyycsxkr5tjrpo7e5uq4p7rhe6vi6btpxsaeyq"
ATTEMPT_ID = "attempt:c7b7da33596041afa5b8658918568ac4"
CLAIM_ID = "claim:1974757172de4c5e9307b0e5eb382280"
TASK_NAMESPACE = "legal-ir-span-graph-lean-refinement-20261016"
EXECUTION_NAMESPACE = "legal-ir-grok-parallel-20261016"
ORIGINAL_HEAD = "d1c515a8a6130fef4a5477dd1ecdbd64a0329e61"
ORIGINAL_TREE = "7e156601940f31e33557f6669bedae8f9cc8c589"
ORIGINAL_ROOT = "/home/barberb/lift_coding/external/ipfs_datasets/workspace/todo-queues/legal-ir-supervisor-start-20261016-01/controller"
ORIGINAL_RUNTIME = ORIGINAL_ROOT + "/workspace/legal-ir-grok-parallel-20261016-r05"
ORIGINAL_BOARD = ORIGINAL_ROOT + "/docs/architecture/legal_ir_grok_parallel_20261016/tasks.todo.md"
ORIGINAL_BOOT_ID = "82e446b4-5572-47d2-a5ea-4259f31bb2d9"
GUARD_REASON = "database task board namespace conflicts with execution board"
MAX_BYTES = 64 * 1024 * 1024
_BASE = "ipfs_accelerate_py/agent_supervisor/"
ORIGINAL_SOURCE_PINS = {
    _BASE + "todo_daemon/database_portal_bridge.py": "afc5d1843b3477b1ba3d46ed4f08ad427351dfae788fc36de720f6d139911ebb",
    _BASE + "todo_daemon/implementation_daemon.py": "d8bff8e58c3168082ed2b175710dc077333d99b41312b40425a38b41cc064d54",
    _BASE + "todo_daemon/implementation_daemon_runner.py": "045907b45ae3ec96f5de3a1d5ab09440fc72d5f25eb38024d9887dd1022c0ee9",
    _BASE + "todo_daemon/implementation_supervisor.py": "bce7c7c3c4d206f286e5538861c410b4e8ff381b60f63f7d5c738609f444eef9",
    _BASE + "runtime/multi_supervisor_runner.py": "52be1001b70ec5538e088fa6e45288afa38e05d49749346dfebd5271f4e38092",
    _BASE + "runtime/configured_board_scheduler.py": "9a3db2b03f792f977ddd84e76a6d5fb522ac9b94678c6d13122c264461c31580",
}
_CONTEXT_KEYS = {
    "schema", "snapshot_pin", "snapshot_native_receipt_pin",
    "snapshot_native_request_pin", "original_request_pin",
    "source_custody_registration", "proof_id",
}
_OBSERVATION_KEYS = {"task", "task_revision_history", "plan_projection", "domain_events"}
_IDENTITY_FIELDS = ("task_cid", "attempt_id", "claim_id", "lease_id", "owner_session_id",
                    "attempt_number", "fencing_token", "fence_epoch")


class NamespaceGuardAdjudicationError(ValueError):
    """The exact historical administrative proof or fresh CAS differs."""


def _need(condition: Any, reason: str) -> None:
    if not condition:
        raise NamespaceGuardAdjudicationError(reason)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _cid(value: Any) -> str:
    digest = hashlib.sha256(_canonical(value)).digest()
    return "b" + base64.b32encode(b"\x01\xa9\x02\x12\x20" + digest).decode().lower().rstrip("=")


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _need(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _decode(raw: str | bytes) -> Any:
    _need(len(raw) <= MAX_BYTES, "JSON exceeds proof bound")
    return json.loads(raw, object_pairs_hook=_object_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(
                          NamespaceGuardAdjudicationError("non-finite JSON")))


def _witness(info: os.stat_result) -> tuple[int, ...]:
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read(pin: Mapping[str, Any]) -> bytes:
    _need(type(pin) is dict and set(pin) == {"path", "bytes", "sha256"}, "file pin is not closed")
    _need(type(pin["bytes"]) is int and 0 <= pin["bytes"] <= MAX_BYTES
          and type(pin["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", pin["sha256"]),
          "file pin type/bound differs")
    path = Path(pin["path"])
    _need(path.is_absolute() and path.resolve(strict=True) == path, "proof path is aliased")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        _need(stat.S_ISREG(before.st_mode) and before.st_uid in (0, os.geteuid())
              and before.st_nlink == 1 and not before.st_mode & 0o002
              and before.st_size == pin["bytes"], "proof file custody differs")
        chunks: list[bytes] = []
        remaining = pin["bytes"] + 1
        while remaining:
            block = os.read(fd, min(65536, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        _need(len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["sha256"]
              and _witness(before) == _witness(os.fstat(fd)) == _witness(path.lstat()),
              "pinned proof input changed")
        return raw
    finally:
        os.close(fd)


def _one(rows: Any, *, noun: str) -> dict[str, Any]:
    _need(type(rows) is list and len(rows) == 1 and type(rows[0]) is dict,
          noun + " is missing or ambiguous")
    return rows[0]


def _same_identity(record: Mapping[str, Any], identity: Mapping[str, Any], fields=_IDENTITY_FIELDS) -> None:
    _need(all(type(record.get(key)) is type(identity[key]) and record.get(key) == identity[key]
              for key in fields), "native identity/fence differs")


def _verify_source_guard(source: bytes) -> None:
    """Check the reviewed exact-source profile's dominating guard ordering."""
    tree = ast.parse(source)
    bridge = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                  and n.name == "DatabasePortalExecutionBridge")
    functions = {n.name: n for n in bridge.body if isinstance(n, ast.FunctionDef)}
    ensure = functions["_ensure_attempt_projection"]
    calls = [(n.lineno, getattr(n.func, "attr", "")) for n in ast.walk(ensure)
             if isinstance(n, ast.Call)]
    render = [line for line, name in calls if name == "_render_projection"]
    seal = [line for line, name in calls if name == "_seal_attempt_directory"]
    _need(len(render) == 1 and seal and render[0] < min(seal), "guard no longer dominates projection creation")
    provider = functions["run_provider"]
    calls = [(n.lineno, getattr(n.func, "attr", "")) for n in ast.walk(provider)
             if isinstance(n, ast.Call)]
    ensure_lines = [line for line, name in calls if name == "_ensure_attempt_projection"]
    factory_lines = [line for line, name in calls if name == "portal_factory"]
    _need(len(ensure_lines) == 1 and factory_lines and ensure_lines[0] < min(factory_lines),
          "guard no longer dominates Portal construction")
    seed = functions["_render_projection_seed"]
    raises = [n for n in ast.walk(seed) if isinstance(n, ast.Raise) and isinstance(n.exc, ast.Call)
              and any(isinstance(a, ast.Constant) and a.value == GUARD_REASON for a in n.exc.args)]
    _need(len(raises) == 1, "reviewed namespace guard literal differs")


def _verify_native_state(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    task = snapshot["task"]
    _need(task.get("task_cid") == TASK_CID and task.get("task_alias") == "LIR-9001"
          and task.get("status") == "in_progress" and task.get("revision") == 7,
          "retained task is outside the singleton admitted scope")
    history = snapshot["task_revision_history"]
    _need(history["task_cid"] == TASK_CID and history["projection_cid"] == _cid(
        {key: value for key, value in history.items() if key != "projection_cid"}), "history identity differs")
    revisions = history["revisions"]
    _need([r["revision"] for r in revisions] == list(range(1, 8)), "complete task history differs")
    _need(all(r["body"].get("board_namespace") == TASK_NAMESPACE for r in revisions[5:])
          and revisions[5]["status"] == "todo" and revisions[6]["status"] == "in_progress"
          and revisions[6]["body"] == task["body"], "callback-interval namespace/body differs")
    lanes = snapshot["execution"]
    _need(type(lanes) is dict and set(lanes) == {"0", "1", "2"}, "three native execution shards required")
    _need(not any(lanes[lane][name] for lane in ("0", "1") for name in
                 ("database_task_attempts", "provider_invocations", "effect_claims", "attempt_phases")),
          "target attempt escaped selected shard")
    active = lanes["2"]
    attempt = _one(active["database_task_attempts"], noun="native execution attempt")
    _need(attempt["task_cid"] == TASK_CID and attempt["attempt_id"] == ATTEMPT_ID
          and attempt["claim_id"] == CLAIM_ID and attempt["attempt_number"] == 1
          and attempt["fencing_token"] == 1 and attempt["fence_epoch"] == 1
          and attempt["status"] == "running" and attempt["committed_phase"] == "context"
          and attempt["revision"] == 2 and attempt["finished_at_ms"] is None,
          "native attempt/ordinal/phase differs")
    identity = {key: attempt[key] for key in _IDENTITY_FIELDS}
    binding = _decode(attempt["body_json"])["control_binding"]
    _same_identity(binding, identity)
    _need(binding["binding_id"] == _cid({k: v for k, v in binding.items() if k != "binding_id"})
          and binding["control_expected_status"] == "in_progress"
          and binding["control_expected_revision"] == 7
          and binding["control_task_projection_cid"] == _cid(task), "claimed native control preimage differs")
    basis = binding["database_portal_binding_basis"]
    _need(basis["task_body_digest"] == _sha(task["body"]) and basis["task_revision"] == 7
          and basis["control_task_projection_cid"] == _cid(task)
          and binding["database_portal_binding_basis_cid"] == _cid(basis), "historical namespace preimage differs")
    phases = active["attempt_phases"]
    _need([(r["phase"], r["revision"]) for r in phases] == [("claimed", 1), ("context", 2)],
          "later/native phase appeared")
    for row in phases:
        _same_identity(row, identity, ("attempt_id", "fencing_token", "fence_epoch"))
    invocation = _one(active["provider_invocations"], noun="preserved UNKNOWN invocation")
    _same_identity(invocation, identity, ("task_cid", "attempt_id", "owner_session_id"))
    _need(invocation["idempotency_key"] == "provider:" + ATTEMPT_ID, "provider key differs")
    unknown = _decode(invocation["result_json"])
    _need(set(unknown) == {"schema", "callback_state", "provider_effect_state",
                           *[k for k in _IDENTITY_FIELDS if k not in ("owner_session_id", "attempt_number")]}
          and unknown["schema"] == "database-portal-callback-intent@1"
          and unknown["callback_state"] == "started_outcome_unknown"
          and unknown["provider_effect_state"] == "unknown_may_have_started", "UNKNOWN history was rewritten")
    _same_identity(unknown, identity, tuple(k for k in _IDENTITY_FIELDS
                                          if k not in ("owner_session_id", "attempt_number")))
    _need(active["effect_claims"] == [], "candidate effect claims exist")
    coordination = snapshot["coordination"]
    claim = _one(coordination["task_claims"], noun="coordination claim")
    lease = _one(coordination["fenced_leases"], noun="coordination lease")
    coord_attempt = _one(coordination["task_attempts"], noun="coordination attempt")
    for record, fields in ((claim, _IDENTITY_FIELDS), (lease, _IDENTITY_FIELDS),
                           (coord_attempt, tuple(k for k in _IDENTITY_FIELDS if k not in ("lease_id", "claim_id")))):
        _same_identity(record, identity, fields)
    _need(claim["state"] == "accepted" and lease["state"] == "accepted"
          and coord_attempt["status"] == "running" and coordination["task_completions"] == [],
          "coordination already settled or superseded")
    events = snapshot["domain_events"]
    _need(type(events) is list and len(events) == 7
          and [e["global_sequence"] for e in events] == sorted({e["global_sequence"] for e in events}),
          "full ordered native task events required")
    native = [_decode(e["body_json"]) for e in events]
    _need(all(e["task_cid"] == TASK_CID and n["subject_id"] == TASK_CID
              and n["event_type"] == e["event_type"] for e, n in zip(events, native)),
          "native event subject differs")
    for revision, envelope in zip(revisions[:-1], native[:-1]):
        body = envelope["body"]
        _need(envelope["event_type"] == "intent.task_upserted"
              and body["revision"] == revision["revision"] and body["body"] == revision["body"]
              and body["status"] == revision["status"], "history/domain event preimage differs")
    transition = native[-1]["body"]
    _need(native[-1]["event_type"] == "intent.task_status_changed"
          and transition["revision"] == 7 and transition["previous_status"] == "todo"
          and transition["status"] == "in_progress"
          and transition["receipt"]["operation"] == "database_claim"
          and transition["receipt"]["claimed_from_revision"] == 6, "claim transition differs")
    _need(transition["task_cid"] == TASK_CID, "native claim transition task differs")
    _same_identity(transition["receipt"], identity, tuple(k for k in _IDENTITY_FIELDS if k != "task_cid"))
    expected_body = dict(revisions[5]["body"])
    expected_body["completion_receipt"] = transition["receipt"]
    _need(expected_body == revisions[6]["body"], "claim changed source contract or namespace")
    _need(snapshot["intent_snapshot"]["task_count"] == 64
          and snapshot["intent_snapshot"]["goal_count"] == 34
          and snapshot["intent_snapshot"]["dependency_count"] == 115,
          "retained population changed")
    return identity


def _load_authority(context: Mapping[str, Any], *, check_live_stores: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    _need(type(context) is dict and set(context) == _CONTEXT_KEYS
          and context["schema"] == AUTHORITY_SCHEMA, "authority context is not closed")
    _need(context["proof_id"] == _sha({k: v for k, v in context.items() if k != "proof_id"}),
          "registered proof content identity differs")
    custody = context["source_custody_registration"]
    _need(type(custody) is dict and set(custody) == {"kind", "registration_reason"}
          and custody["kind"] == "explicit_administrator_historical_source_registration"
          and type(custody["registration_reason"]) is str and 16 <= len(custody["registration_reason"]) <= 2048,
          "explicit historical administrator registration required")
    snapshot = _decode(_read(context["snapshot_pin"]))
    receipt = _decode(_read(context["snapshot_native_receipt_pin"]))
    request_raw = _read(context["snapshot_native_request_pin"])
    request = _decode(request_raw)
    _need(receipt["status"] == "completed" and receipt["child_returncode"] == 0
          and receipt["owned_group_quiescent"] is True and receipt["purpose"] == "supervisor_preflight"
          and receipt["request_sha256"] == hashlib.sha256(request_raw).hexdigest()
          and Path(context["snapshot_pin"]["path"]).parent == Path(request["attempt_directory"])
          and request["source_sha256"].get(snapshot["descriptor_pin"]["path"]) == snapshot["descriptor_pin"]["sha256"],
          "independent original native snapshot job binding differs")
    _read(snapshot["descriptor_pin"])
    _need(snapshot["schema"] == "legal-ir-original-namespace-denial-snapshot/v1"
          and snapshot["task_cid"] == TASK_CID and snapshot["attempt_id"] == ATTEMPT_ID
          and snapshot["original_head"] == ORIGINAL_HEAD and snapshot["original_tree"] == ORIGINAL_TREE
          and snapshot["read_only"] is True
          and all(snapshot[key] is False for key in ("database_mutations_attempted", "recovery_attempted",
                       "completion_authority", "retry_authority", "settlement_authority")), "read-only subject snapshot differs")
    run = {role: _decode(_read(item["snapshot_pin"])) for role, item in snapshot["original_run"].items()}
    original_raw = _read(context["original_request_pin"])
    _need(snapshot["original_run"]["original_request"]["original_pin"] == context["original_request_pin"]
          and original_raw == _read(snapshot["original_run"]["original_request"]["snapshot_pin"]),
          "original guarded launch request differs")
    original_request = run["original_request"]
    outer, terminal, result = run["outer_receipt"], run["native_terminal"], run["native_run_result"]
    _need(outer["status"] == "completed" and outer["child_returncode"] == 0
          and outer["request_sha256"] == hashlib.sha256(original_raw).hexdigest()
          and outer["native_stop_verified"] is True and outer["owned_group_quiescent"] is True
          and outer["cleanup_pending"] is False and terminal["all_trees_fenced"] is True
          and outer["native_fence_proof"]["terminal_receipt_cid"] == terminal["terminal_receipt_cid"]
          and result["native_fence_proof"]["terminal_receipt_cid"] == terminal["terminal_receipt_cid"]
          and result["status"] == "stopped_and_original_owner_closed"
          and result["standard_database_portal_bridge"] is True and result["custom_execution_callbacks"] is False
          and result["public_database_program"]["store_id"] == "legal-ir-catalog02.duckdb"
          and result["public_database_program"]["authority_mode"] == "quack",
          "original native run is not wholly fenced")
    commits = [r["body"] for r in run["launch_operations"]
               if r["operation"] == "actual_owner_control_commit_and_preflight"]
    commit = _one(commits, noun="original native source boundary")
    _need(commit["controller_head"] == ORIGINAL_HEAD and commit["controller_tree"] == ORIGINAL_TREE,
          "historical source boundary differs")
    originals: dict[str, bytes] = {}
    guarded_paths: set[str] = set()
    for row in snapshot["original_sources"]:
        raw = _read(row["archive"])
        if row["original_request_pin_verified"]:
            _need(row["original_path"] not in guarded_paths
                  and original_request["source_sha256"].get(row["original_path"]) == row["archive"]["sha256"],
                  "historical guarded source differs")
            guarded_paths.add(row["original_path"])
        if "git_path" in row:
            _need(row["git_head"] == ORIGINAL_HEAD and row["git_tree"] == ORIGINAL_TREE,
                  "historical Git source origin differs")
            _need(row["git_path"] not in originals, "duplicate historical source")
            originals[row["git_path"]] = raw
    _need(guarded_paths == set(original_request["source_sha256"]), "original guarded source closure incomplete")
    for name, digest in ORIGINAL_SOURCE_PINS.items():
        _need(name in originals and hashlib.sha256(originals[name]).hexdigest() == digest,
              "source outside reviewed historical nonreachability profile")
    _verify_source_guard(originals[_BASE + "todo_daemon/database_portal_bridge.py"])
    supervisor_ast = ast.parse(originals[_BASE + "todo_daemon/implementation_supervisor.py"])
    assignments = [n for n in supervisor_ast.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "ORDINARY_IMPLEMENTATION_DAEMON_BOOTSTRAP"
                           for t in n.targets)]
    _need(len(assignments) == 1, "reviewed native bootstrap is ambiguous")
    expression = assignments[0].value
    _need(isinstance(expression, ast.Call) and isinstance(expression.func, ast.Attribute)
          and expression.func.attr == "join" and isinstance(expression.func.value, ast.Constant)
          and expression.func.value.value == "\n" and len(expression.args) == 1
          and isinstance(expression.args[0], ast.Tuple)
          and all(isinstance(n, ast.Constant) and type(n.value) is str for n in expression.args[0].elts),
          "reviewed native bootstrap literal differs")
    bootstrap = "\n".join(n.value for n in expression.args[0].elts)
    seen_lanes: set[int] = set()
    births: set[tuple[int, int]] = set()
    _need(type(snapshot["daemon_identities"]) is list and len(snapshot["daemon_identities"]) == 19,
          "full original daemon identity interval required")
    for item in snapshot["daemon_identities"]:
        observed = _decode(_read(item["snapshot_pin"]))
        _need(observed == item["identity"], "original daemon identity bytes differ")
        argv = observed["command"]
        def flag(name: str) -> str:
            _need(argv.count(name) == 1, "original daemon option is ambiguous")
            return argv[argv.index(name) + 1]
        lane = int(flag("--task-shard-index"))
        sentinel = "ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon"
        _need(argv[:5] == ["/usr/bin/python3", "-P", "-c", bootstrap, sentinel]
              and lane in (0, 1, 2) and flag("--task-shard-count") == "3"
              and flag("--board-namespace") == EXECUTION_NAMESPACE
              and flag("--authority-mode") == "quack" and flag("--task-source-kind") == "duckdb"
              and "--implement" in argv and "--strict-task-sharding" in argv
              and observed["command_id"] == _sha({"command": argv})
              and observed["record_id"] == _sha({k: v for k, v in observed.items() if k != "record_id"}),
              "native daemon command/source route differs")
        state_dir = ORIGINAL_RUNTIME + "/state/lane-" + str(lane)
        state_prefix = "lir_lane_" + str(lane)
        _need(observed["owner_scope"] == {"daemon_entrypoint": sentinel, "repo_root": ORIGINAL_ROOT,
                  "state_dir": state_dir, "state_prefix": state_prefix, "todo_path": ORIGINAL_BOARD}
              and flag("--state-prefix") == state_prefix and flag("--state-dir") == state_dir
              and flag("--todo-path") == ORIGINAL_BOARD
              and flag("--merge-target-branch") == "agent/" + EXECUTION_NAMESPACE,
              "native daemon lane/root/board binding differs")
        process = observed["process_birth"]
        _need(process["boot_id"] == ORIGINAL_BOOT_ID
              and all(type(process[key]) is int and process[key] > 1
                      for key in ("pid", "parent_pid", "start_time_ticks"))
              and result["master_started_at_epoch_nanoseconds"] / 1_000_000_000
                  <= datetime.fromisoformat(observed["created_at"]).timestamp()
                  <= outer["ended_at_epoch"], "native child birth interval differs")
        birth = (observed["process_birth"]["pid"], observed["process_birth"]["start_time_ticks"])
        _need(birth not in births, "duplicate daemon process identity")
        births.add(birth)
        seen_lanes.add(lane)
    _need(seen_lanes == {0, 1, 2}, "an original native lane is unobserved")
    identity = _verify_native_state(snapshot)
    # Closed execution/coordination stores must still be the exact witnessed
    # files. Canonical store raw bytes may change on a genuine owner startup;
    # its four source projections are compared under the owner's transaction.
    if check_live_stores:
        for role, pin in snapshot["database_input_pins"].items():
            if role != "canonical":
                _read(pin)
    return snapshot, identity


def namespace_guard_adjudication_scope(authority_context: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the protected registration before issuing its singleton grant."""
    snapshot, identity = _load_authority(authority_context)
    return {"task_cid": identity["task_cid"], "proof_id": authority_context["proof_id"],
            "canonical_store_id": snapshot["database_input_pins"]["canonical"]["path"],
            "canonical_path": snapshot["database_input_pins"]["canonical"]["path"],
            "logical_store_id": _decode(_read(snapshot["original_run"]["native_run_result"]["snapshot_pin"]))[
                "public_database_program"]["store_id"],
            "expected_task_revision": snapshot["task"]["revision"],
            "recovery_scope_cid": _cid({"operation": OPERATION, "proof_id": authority_context["proof_id"],
                                       **identity})}


def verify_namespace_guard_adjudication(
    payload: Mapping[str, Any], *, authority_context: Mapping[str, Any],
    current_observation: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify exact historical conditions plus a fresh owner-transaction CAS."""
    _need(type(payload) is dict and set(payload) == {"operation", "proof_id", "expected_task_revision"}
          and payload["operation"] == OPERATION and payload["proof_id"] == authority_context.get("proof_id")
          and type(payload["expected_task_revision"]) is int, "ADMIN command is not closed")
    snapshot, identity = _load_authority(authority_context)
    _need(type(current_observation) is dict and set(current_observation) == _OBSERVATION_KEYS,
          "fresh native observation is not closed")
    _need(payload["expected_task_revision"] == snapshot["task"]["revision"]
          and all(current_observation[key] == snapshot[key] for key in _OBSERVATION_KEYS),
          "native task/source history changed before administrative CAS")
    return _adjudication_receipt(snapshot, identity, authority_context)


def _adjudication_receipt(snapshot: Mapping[str, Any], identity: Mapping[str, Any],
                         authority_context: Mapping[str, Any]) -> dict[str, Any]:
    """Build the deterministic receipt after either native authority check."""
    receipt = {
        "schema": RECEIPT_SCHEMA, "operation": OPERATION, "proof_id": authority_context["proof_id"],
        "source_profile": SOURCE_PROFILE, "original_head": ORIGINAL_HEAD, "original_tree": ORIGINAL_TREE,
        "task_alias": "LIR-9001", **identity, "control_expected_status": "in_progress",
        "control_expected_revision": snapshot["task"]["revision"],
        "snapshot_sha256": authority_context["snapshot_pin"]["sha256"],
        "original_request_sha256": authority_context["original_request_pin"]["sha256"],
        "source_custody_registration_sha256": _sha(authority_context["source_custody_registration"]),
        "authority_kind": "explicit_typed_owner_administrative_adjudication",
        "reason": GUARD_REASON, "task_namespace": TASK_NAMESPACE, "execution_namespace": EXECUTION_NAMESPACE,
        "nonreachability_scope": "model_provider_dispatch_and_candidate_effect_for_exact_attempt",
        "provider_invocation_count": 1, "effect_claim_count": 0,
        "unknown_preserved": True, "spent_attempt_ordinal_preserved": True,
        "provider_result_inferred": False, "provider_exit_inferred": False,
        "history_rewritten": False, "completion_authority": False,
        "automatic_retry_admitted": False, "successor_attempt_admitted": False,
        "native_failure_settlement_authorized": True,
    }
    receipt["adjudication_id"] = _cid(receipt)
    return receipt
