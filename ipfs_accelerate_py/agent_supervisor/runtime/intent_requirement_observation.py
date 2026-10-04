"""Read source-bound public-check progress without changing native intent.

The interpretation and reviewed task bindings stay immutable.  This projection
measures only persisted owner-signed public validations and output presence;
neither of those observations proves the meaning of the original instruction.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

from ..proof.formal_verification_contracts import content_identity
from ..task_sources.intent_repository import (
    IntentRepository, completion_evidence_projection_on_connection, _SUCCESSFUL_TASK_STATUSES,
)
from . import local_planning_admission as local

SCHEMA = "intent-requirement-observation@1"
MAX_OBSERVATIONS = 4096
MAX_BYTES = 8_000_000
_AUTHORITY = {
    "semantic_alignment_verified": False, "source_semantics_verified": False,
    "proof_authority": False, "execution_authority": False,
    "completion_authority": False, "canonical_state_mutated": False,
}


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _bounded(value):
    raw = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(raw.encode("utf-8")) > MAX_BYTES:
        raise local.LocalPlanningError("requirement observation exceeds byte bound")
    return json.loads(raw)


def _rows(connection, task_cids):
    placeholders = ",".join("?" for _ in task_cids)
    population = connection.execute(
        "SELECT COUNT(*),COALESCE(SUM(octet_length(CAST(r.body_json AS BLOB))"
        "+COALESCE(octet_length(CAST(e.body_json AS BLOB)),0)"
        "+COALESCE(octet_length(CAST(v.body_json AS BLOB)),0)),0) "
        "FROM validation_results r LEFT JOIN validation_runs v ON v.run_id=r.run_id AND v.task_cid=r.task_cid "
        "LEFT JOIN domain_events e ON e.task_cid=r.task_cid "
        "AND e.event_type='intent.validation_recorded' "
        "AND json_extract_string(TRY_CAST(e.body_json AS JSON),'$.subject_id')=r.result_id "
        f"WHERE r.task_cid IN ({placeholders})", list(task_cids),
    ).fetchone()
    if population[0] > MAX_OBSERVATIONS or population[1] > MAX_BYTES:
        raise local.LocalPlanningError("requirement validation population exceeds bound")
    rows = connection.execute(
        "SELECT r.task_cid,r.result_id,r.run_id,r.outcome,r.evidence_digest,r.body_json AS result_body_json,e.global_sequence,"
        "v.attempt_id AS run_attempt_id,v.status AS run_status,v.command_digest,e.attempt_id AS event_attempt_id,"
        "e.body_json AS event_body_json,v.body_json AS run_body_json,"
        "e.event_id,e.stream_id,e.sequence "
        "FROM validation_results r LEFT JOIN validation_runs v ON v.run_id=r.run_id AND v.task_cid=r.task_cid "
        "LEFT JOIN domain_events e ON e.task_cid=r.task_cid "
        "AND e.event_type='intent.validation_recorded' "
        "AND json_extract_string(TRY_CAST(e.body_json AS JSON),'$.subject_id')=r.result_id "
        f"WHERE r.task_cid IN ({placeholders}) ORDER BY e.global_sequence DESC LIMIT ?",
        [*task_cids, MAX_OBSERVATIONS + 1],
    ).fetchall()
    if len(rows) > MAX_OBSERVATIONS:
        raise local.LocalPlanningError("requirement validation population exceeds bound")
    # Native DuckDBRow iterates column names; numeric indexing yields values.
    rows = [tuple(row[index] for index in range(16)) for row in rows]
    rows.extend(_orphan_rows(connection, task_cids))
    if len(rows) > MAX_OBSERVATIONS:
        raise local.LocalPlanningError("requirement validation population exceeds bound")
    counts = {}
    for row in rows:
        counts[row[1]] = counts.get(row[1], 0) + 1
    return [(*row, counts[row[1]]) for row in rows]


def _orphan_rows(connection, task_cids):
    """Retain intact signed copies when another native projection is missing."""
    placeholders = ",".join("?" for _ in task_cids)
    queries = [(
        "FROM validation_runs v WHERE v.task_cid IN (" + placeholders + ") "
        "AND NOT EXISTS (SELECT 1 FROM validation_results r WHERE r.run_id=v.run_id AND r.task_cid=v.task_cid)",
        "v.body_json",
        "v.task_cid,NULL,v.run_id,NULL,NULL,NULL,NULL,v.attempt_id,v.status,v.command_digest,"
        "NULL,NULL,v.body_json,NULL,NULL,NULL",
    ), (
        "FROM domain_events e WHERE e.task_cid IN (" + placeholders + ") "
        "AND e.event_type='intent.validation_recorded' AND NOT EXISTS "
        "(SELECT 1 FROM validation_results r WHERE r.task_cid=e.task_cid "
        "AND r.result_id=json_extract_string(TRY_CAST(e.body_json AS JSON),'$.subject_id'))",
        "e.body_json",
        "e.task_cid,json_extract_string(TRY_CAST(e.body_json AS JSON),'$.subject_id'),"
        "json_extract_string(TRY_CAST(e.body_json AS JSON),'$.body.run_id'),NULL,NULL,NULL,"
        "e.global_sequence,NULL,NULL,NULL,e.attempt_id,e.body_json,NULL,e.event_id,e.stream_id,e.sequence",
    )]
    result = []
    total_bytes = 0
    for clause, body, fields in queries:
        population = connection.execute(
            "SELECT COUNT(*),COALESCE(SUM(octet_length(CAST(" + body + " AS BLOB))),0) " + clause,
            list(task_cids),
        ).fetchone()
        total_bytes += population[1]
        if population[0] + len(result) > MAX_OBSERVATIONS or total_bytes > MAX_BYTES:
            raise local.LocalPlanningError("orphan requirement evidence exceeds bound")
        rows = connection.execute("SELECT " + fields + " " + clause + " LIMIT ?",
            [*task_cids, MAX_OBSERVATIONS + 1]).fetchall()
        result.extend(tuple(row[index] for index in range(16)) for row in rows)
    return result


def _signed_observation(row, *, tasks, contracts, profile, manifest_cid):
    """Recognize signed check identity before checking native evidence links.

    Unrelated evidence is ignored.  Corruption of an authentic admitted check
    makes this projection unavailable instead of resurrecting an older pass.
    """
    def decoded(raw):
        try:
            value = json.loads(raw)
            return value if isinstance(value, dict) else {}
        except (ValueError, TypeError):
            return {}
    result_body, run_body, event_wrapper = (decoded(row[index]) for index in (5, 12, 11))
    event = event_wrapper.get("body", {})
    event_body = event.get("body", {}) if isinstance(event, dict) else {}
    task, contract = tasks[row[0]], contracts[row[0]]
    recognized = None
    # Any intact persisted copy can authenticate an original local check.
    # A damaged result copy cannot hide its genuine run/event envelope.
    for body in (result_body, run_body, event_body):
        if not isinstance(body, dict):
            continue
        envelope = body.get("local_observed_validation")
        if not envelope:
            continue
        try:
            observed = local._verify_signature(envelope, profile)
            check = observed.get("validation", {})
            if (
                observed.get("schema") == local.RESULT_SCHEMA
                and observed.get("task_cid") == row[0]
                and observed.get("manifest_cid") == manifest_cid
                and observed.get("contract_cid") == content_identity(task["body"][local.CONTRACT_KEY])
                and observed.get("pending_cid") == contract["pending_cid"]
                and observed.get("intent_owner_id") == contract["intent_owner_id"]
                and check in contract["task_spec"]["validations"]
            ):
                recognized = observed, check, envelope
                break
        except (ValueError, TypeError, KeyError, OSError):
            continue
    if recognized is None:
        return None
    observed, check, envelope = recognized
    try:
        fields = {"schema", "task_cid", "task_revision", "attempt_id", "contract_cid",
            "intent_owner_id", "manifest_cid", "pending_cid", "source_tree_id", "validation",
            "exit_code", "stdout_sha256", "stderr_sha256", "outcome"}
        if (not fields <= set(observed) or set(observed) - fields - {"source_transition", "candidate_runner"}
            or content_identity(envelope) != row[4]
            or not isinstance(observed.get("attempt_id"), str) or not observed["attempt_id"]
            or type(observed.get("task_revision")) is not int
            or (observed.get("exit_code") is not None and type(observed["exit_code"]) is not int)
            or observed.get("outcome") != row[3] or row[3] not in {"passed", "failed"}
            or (row[3] == "passed" and observed.get("exit_code") != 0)
            or (row[3] == "failed" and observed.get("exit_code") == 0)
            or row[16] != 1 or row[6] is None
            or row[7] != observed["attempt_id"] or row[10] != observed["attempt_id"]
            or row[8] != row[3] or row[9] != content_identity({"argv": check.get("argv")})
            or run_body != {"argv": check["argv"], "local_observed_validation": envelope}
            or result_body != {"local_observed_validation": envelope}
            or event.get("run_id") != row[2] or event.get("result_id") != row[1]
            or event.get("outcome") != row[3] or event.get("evidence_digest") != row[4]
            or event.get("argv") != check.get("argv")
            or event.get("body", {}).get("local_observed_validation") != envelope
            or event_wrapper.get("owner_id") != contract["intent_owner_id"]
            or event_wrapper.get("subject_id") != row[1]
            or event_wrapper.get("event_type") != "intent.validation_recorded"
            or row[1] != content_identity({"run_id": row[2], "outcome": row[3], "evidence_digest": row[4]})
            or row[13] != content_identity({"stream_id": row[14], "sequence": row[15],
                "global_sequence": row[6], "event_type": "intent.validation_recorded", "body": event_wrapper})
        ):
            raise ValueError("native signed local check links differ")
        return observed
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise local.LocalPlanningError("signed public check has inconsistent native evidence links") from exc


def _native_tasks(reader, graph, verified, admission):
    tasks, contracts = {}, {}
    head = reader.get_plan_head(graph.root_goal.goal_cid)
    plan = reader.get_plan(verified["receipt"]["plan_id"])
    if (head is None or head.plan_cid != verified["receipt"]["plan_id"] or plan is None
            or plan["goal_cid"] != graph.root_goal.goal_cid or plan["status"] != "active"
            or plan["body"].get("local_planning_receipt_ref") != local._receipt_reference(
                admission["receipt"], local._receipt_bytes(admission["receipt"]),
            )):
        raise local.LocalPlanningError("native requirement plan head differs from immutable admission")
    # A selected subset cannot hide another native task in this local plan.
    population = reader.list_tasks(limit=17)
    if len(population) > 16 or {row["task_cid"] for row in population} != {
            task.task_cid for task in graph.tasks}:
        raise local.LocalPlanningError("native requirement task population differs from admission")
    for task in graph.tasks:
        native = reader.get_task(task.task_cid)
        envelope = native["body"].get(local.CONTRACT_KEY, {})
        contract = local._verify_signature(envelope, verified["profile"])
        owner_id = contract.get("intent_owner_id")
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise local.LocalPlanningError("exact admitted intent owner required")
        with reader._connection(write=False) as connection:
            original = connection.execute(
                "SELECT json_extract_string(body_json,'$.owner_id'),"
                "json_extract_string(body_json,'$.body.identity.local_contract_cid') "
                "FROM domain_events WHERE task_cid=? AND event_type='intent.task_upserted' "
                "ORDER BY global_sequence ASC LIMIT 1", [task.task_cid],
            ).fetchone()
        if (original is None or original[0] != owner_id
                or original[1] != content_identity(envelope)):
            raise local.LocalPlanningError("native requirement task differs from original materialization owner")
        expected = local._pending_contract_payload(
            admission=admission, verified=verified, task=task, intent_owner_id=owner_id,
        )
        spec = expected["task_spec"]
        outputs = [{"ordinal": index, "path": item["path"], "effect": item}
            for index, item in enumerate(spec["outputs"])]
        checks = [{"ordinal": index, "argv": item["argv"],
            "policy": {key: value for key, value in item.items() if key != "argv"}}
            for index, item in enumerate(spec["validations"])]
        acceptance = [{"ordinal": index, "criterion": item["criterion"], "evidence_policy": item}
            for index, item in enumerate(spec["acceptance"])]
        if (contract != expected
                or native["identity"].get("local_contract_cid") != content_identity(envelope)
                or native["identity"].get("repository_tree_id") != verified["receipt"]["source_tree_id"]
                or native["task_alias"] != task.task_key
                or native["goal_cid"] != task.goal_cid
                or native["plan_cid"] != expected["plan_id"]
                or list(native["dependencies"]) != expected["dependencies"]
                or local._plain(native["outputs"]) != outputs
                or local._plain(native["validations"]) != checks
                or local._plain(native["acceptance"]) != acceptance):
            raise local.LocalPlanningError("native requirement task differs from immutable admitted contract")
        tasks[task.task_cid], contracts[task.task_cid] = native, contract
    return tasks, contracts


def _verify_admission(admission):
    """Verify immutable proposal bytes before choosing a current source owner."""
    local._require_admission_fields(admission)
    manifest = admission["manifest"]["payload"]
    if manifest.get("schema") != local.INTENT_MANIFEST_SCHEMA:
        raise local.LocalPlanningError("source-bound intent admission required")
    profile = local.load_local_profile(
        repository_cid=manifest["repository_cid"], profile_dir=Path(manifest["profile_dir"]),
        lifecycle_dir=Path(manifest["lifecycle_dir"]),
    )
    manifest = local._verify_signature(admission["manifest"], profile)
    graph = local.PromptGoalGraph.from_dict(admission["graph"])
    receipt = local._verify_signature(admission["receipt"], profile)
    expected = local._planning_payload(
        graph, admission["manifest"], manifest, profile, manifest["sources"],
        admission["requirement_bindings"],
        source_applicability_nomination=local._header_nomination(receipt),
    )
    if receipt != expected:
        raise local.LocalPlanningError("requirement observation admission differs from replay")
    return {"manifest": manifest, "profile": profile, "receipt": receipt, "graph": graph}


def _current_sources(admission, verified, tasks, contracts, rows, source_transition):
    if source_transition is not None:
        return (*local._manifest(admission["manifest"], source_transition=source_transition), source_transition)
    try:
        return (*local._manifest(admission["manifest"]), None)
    except local.LocalPlanningError as baseline_error:
        # Discover publication custody only from actual native signed observations.
        for row in rows:
            observed = _signed_observation(row, tasks=tasks, contracts=contracts,
                profile=verified["profile"], manifest_cid=verified["receipt"]["manifest_cid"])
            if observed is None or not observed.get("source_transition"):
                continue
            transition = observed["source_transition"]
            native = tasks[row[0]]
            claim = native["body"].get("completion_receipt", {})
            if (observed["task_revision"] != native["revision"] - (native["status"] in _SUCCESSFUL_TASK_STATUSES)
                    or (claim.get("attempt_id") and observed["attempt_id"] != claim["attempt_id"])):
                continue
            try:
                _, manifest, profile, current = local._contract(
                    native["body"], row[0], source_transition=transition,
                )
                if observed["source_tree_id"] != local._tree(current):
                    continue
                return manifest, profile, current, transition
            except (ValueError, KeyError, TypeError, OSError):
                continue
        raise local.LocalPlanningError("current requirement source has no exact owner observation") from baseline_error


def _validation(row, observed, *, native, current_tree, completion, published_source):
    stale = []
    expected_revision = native["revision"] - (native["status"] in _SUCCESSFUL_TASK_STATUSES)
    claim = native["body"].get("completion_receipt", {})
    if observed["task_revision"] != expected_revision:
        stale.append("task_revision_changed")
    if claim.get("attempt_id") and observed["attempt_id"] != claim["attempt_id"]:
        stale.append("attempt_changed")
    if native["status"] in _SUCCESSFUL_TASK_STATUSES:
        receipt = completion.get(native["task_cid"])
        if (not receipt or not claim.get("attempt_id")
                or (receipt.get("attempt_id") and receipt["attempt_id"] != observed["attempt_id"])
                or receipt["body"]["receipt"].get("attempt_id") != observed["attempt_id"]
                or receipt["body"]["receipt"] != claim):
            stale.append("retained_completion_attempt_unavailable")
    if observed.get("source_tree_id") != current_tree:
        stale.append("source_tree_changed")
    transition = observed.get("source_transition")
    if published_source and transition is None:
        stale.append("publication_binding_unavailable")
    if transition is not None:
        try:
            local._contract(native["body"], native["task_cid"], source_transition=transition)
            if (transition["payload"]["task_revision"] != observed["task_revision"]
                    or transition["payload"]["attempt_id"] != observed["attempt_id"]):
                stale.append("publication_attempt_changed")
        except (ValueError, TypeError, KeyError, OSError):
            stale.append("publication_source_changed")
    return {**observed["validation"], "status": "stale" if stale else observed["outcome"],
        "evidence_digest": row[4], "result_id": row[1], "run_id": row[2],
        "event_sequence": row[6], "observation_source_tree_id": observed.get("source_tree_id"),
        "observation_task_revision": observed["task_revision"], "attempt_id": observed["attempt_id"],
        "outcome": observed["outcome"], "exit_code": observed.get("exit_code"), "stale_reasons": sorted(set(stale))}


def _requirement_rows(contract, coverage, tasks):
    bindings = {row["requirement_id"]: row for row in coverage["bindings"]}
    specs = {row["requirement_id"]: row for row in contract["requirements"]}
    result = []
    for source in contract["ledger"]["requirements"]:
        key = source["requirement_id"]
        binding, spec = bindings.get(key), specs.get(key)
        task_cids = list(binding["task_cids"]) if binding else []
        refs, missing, failed, stale, unobserved = [], [], [], [], []
        if binding:
            for output in spec["outputs"]:
                observed = [item for cid in task_cids for item in tasks[cid]["outputs"]
                    if all(item[name] == output[name] for name in ("path", "effect", "media_type"))]
                if not observed or not all(item["present"] for item in observed):
                    missing.append(output["path"])
            for cid in task_cids:
                for check in tasks[cid]["validations"]:
                    if check["validation_key"] not in spec["validation_keys"]:
                        continue
                    ref = {"task_cid": cid, "validation_key": check["validation_key"]}
                    refs.append(ref)
                    if check["status"] == "failed":
                        failed.append(ref)
                    elif check["status"] == "stale":
                        stale.append(ref)
                    elif check["status"] == "unobserved":
                        unobserved.append(ref)
        measured = binding is not None
        passed = measured and bool(refs) and not (failed or stale or unobserved)
        status = ("not_measured" if not measured else "failed" if failed else
            "stale" if stale else "unobserved" if unobserved else
            "missing_outputs" if missing else "not_measured" if not refs else "public_checks_passed")
        result.append({"requirement_id": key, "source_unit_id": source["source_unit_id"],
            "kind": source["kind"], "modality": source["modality"], "measurement_status": status,
            "task_cids": task_cids, "output_paths": sorted(item["path"] for item in spec["outputs"]) if spec else [],
            "validation_refs": refs, "public_checks_passed": passed,
            "missing_output_paths": sorted(set(missing)), "failed_validation_refs": failed,
            "stale_validation_refs": stale, "unobserved_validation_refs": unobserved,
            "measurement_reason": ("requirement_scope_not_measured" if not measured else
                "bound_public_check_projection" if refs else "no_bound_public_validation"),
            "source_semantic_status": "unresolved", "source_semantics_verified": False})
    return result


def _observe_on_connection(*, admission, intent, connection, source_transition,
                          expected_intent_owner_id=None, native_owner_binding=None):
    verified = _verify_admission(admission)
    graph, receipt = verified["graph"], verified["receipt"]
    watermark = intent.event_watermark()
    native, contracts = _native_tasks(intent, graph, verified, admission)
    owners = {contract["intent_owner_id"] for contract in contracts.values()}
    if len(owners) != 1 or (expected_intent_owner_id is not None and owners != {expected_intent_owner_id}):
        raise local.LocalPlanningError("requirement intent owner differs from native materialization")
    _bounded(native)
    native_identity = content_identity(native)
    rows = _rows(connection, sorted(native))
    manifest, profile, current, transition = _current_sources(
        admission, verified, native, contracts, rows, source_transition,
    )
    if manifest != verified["manifest"] or profile != verified["profile"]:
        raise local.LocalPlanningError("requirement source owner differs from admission")
    current_tree = local._tree(current)
    completion = completion_evidence_projection_on_connection(
        connection, task_cids=sorted(native), transaction_owned_by_caller=True,
    )
    completions = {row["task_cid"]: row for row in completion["completion_receipts"]}
    latest = {}
    for row in rows:
        observed = _signed_observation(row, tasks=native, contracts=contracts,
            profile=profile, manifest_cid=receipt["manifest_cid"])
        if observed is None:
            continue
        key = row[0], observed["validation"]["validation_key"]
        if key not in latest:
            latest[key] = _validation(row, observed, native=native[row[0]],
                current_tree=current_tree, completion=completions, published_source=transition is not None)
    projected = {}
    for cid, task in sorted(native.items()):
        spec = contracts[cid]["task_spec"]
        checks = [latest.get((cid, check["validation_key"]), {**check, "status": "unobserved",
            "evidence_digest": None, "result_id": None, "run_id": None, "event_sequence": None,
            "observation_source_tree_id": None, "observation_task_revision": None,
            "attempt_id": None, "outcome": None, "exit_code": None, "stale_reasons": []})
            for check in spec["validations"]]
        outputs = [{**output, "present": output["path"] in current,
            "source_sha256": current.get(output["path"], {}).get("sha256"),
            "size_bytes": (Path(manifest["repository"]) / output["path"]).stat().st_size
                if output["path"] in current else None}
            for output in spec["outputs"]]
        projected[cid] = {"task_cid": cid, "task_key": task["task_alias"],
            "goal_cid": task["goal_cid"], "status": task["status"], "revision": task["revision"],
            "contract_cid": content_identity(task["body"][local.CONTRACT_KEY]),
            "dependency_task_cids": list(task["dependencies"]), "outputs": outputs,
            "validations": checks, "acceptance": spec["acceptance"],
            "current_completion_receipt_cid": completions.get(cid, {}).get("receipt_cid")}
    # This transaction binds native relations; repeat filesystem checks because
    # MVCC alone says nothing about live repository bytes.
    _, _, after = local._manifest(admission["manifest"], source_transition=transition)
    after_native, _ = _native_tasks(intent, graph, verified, admission)
    if (after != current or intent.event_watermark() != watermark
            or content_identity(after_native) != native_identity
            or completion["event_watermark"] != watermark):
        raise local.LocalPlanningError("requirement sources or native relations changed during observation")
    contract = local.decode_intent_requirement_contract(manifest)
    requirements = _requirement_rows(contract, receipt["requirement_coverage"], projected)
    revision = {"manifest_cid": receipt["manifest_cid"],
        "planning_receipt_cid": content_identity(admission["receipt"]), "graph_cid": graph.content_id,
        "contract_cid": manifest["intent_requirements"]["contract_cid"],
        "ledger_sha256": contract["ledger"]["ledger_sha256"]}
    value = _bounded({"schema": SCHEMA, "measurement_scope": "public_validation_and_output_presence",
        **revision, "intent_revision_cid": content_identity(revision), "plan_id": receipt["plan_id"],
        "repository_id": manifest["repository_cid"], "policy_id": content_identity(manifest["policy"]),
        "intent_owner_id": next(iter(owners)),
        "native_owner_binding": native_owner_binding,
        "source_path": contract["source_path"], "source_sha256": contract["ledger"]["source"]["sha256"],
        "current_source_tree_id": current_tree, "native_event_watermark": watermark,
        "freshness_scope": "signed_source_inventory_and_native_revision",
        "live_validation_rerun_performed": False,
        "native_population_complete": True, "tasks": list(projected.values()), "requirements": requirements,
        "residual_requirement_ids": sorted(row["requirement_id"] for row in requirements
            if row["measurement_status"] not in {"public_checks_passed", "not_measured"}),
        "unmeasured_requirement_ids": sorted(row["requirement_id"] for row in requirements
            if row["measurement_status"] == "not_measured"),
        "official_reward": None, "provider_calls": 0, **_AUTHORITY})
    return {**value, "observation_cid": content_identity(value)}


def observe_local_intent_requirements(*, admission: Mapping, intent: IntentRepository,
                                    source_transition: Mapping | None = None) -> dict:
    """Observe an independently owned native intent through a read transaction."""
    if not isinstance(intent, IntentRepository) or intent.uses_bound_connection:
        raise local.LocalPlanningError("independently owned intent repository required")
    import duckdb
    with intent._connection(write=False) as connection:
        connection.execute("BEGIN TRANSACTION")
        try:
            reader = IntentRepository(bound_connection=connection, install_schema=False, owner_id=intent.owner_id)
            value = _observe_on_connection(admission=admission, intent=reader,
                connection=connection, source_transition=source_transition,
                expected_intent_owner_id=intent.owner_id)
            connection.execute("COMMIT")
        except BaseException as exc:
            connection.execute("ROLLBACK")
            if isinstance(exc, duckdb.Error):
                raise local.LocalPlanningError("native requirement observation query is unavailable") from exc
            raise
    if intent.event_watermark() != value["native_event_watermark"]:
        raise local.LocalPlanningError("native intent changed after requirement capture")
    return value


def observe_owner_intent_requirements(*, server, admission: Mapping,
                                    source_transition: Mapping | None = None) -> dict:
    """Read through the actual ready owner; never reopen its database file."""
    from .quack_state_server import QuackStateServer, QuackStateServerReadyError
    import duckdb

    if type(server) is not QuackStateServer:
        raise local.LocalPlanningError("actual native owner required for requirement observation")
    with server._lock:
        try:
            ready = server.identity is not None and server.ready().get("ready") is True
        except QuackStateServerReadyError as exc:
            raise local.LocalPlanningError("native requirement owner is not ready") from exc
        if not ready:
            raise local.LocalPlanningError("native requirement owner is not ready")
        identity = server.identity
        if identity.repository_id != admission.get("manifest", {}).get("payload", {}).get("repository_cid"):
            raise local.LocalPlanningError("native requirement owner belongs to another repository")
        binding = {"server_id": identity.server_id, "database_uuid": identity.database_uuid,
            "generation": identity.generation, "process_birth_id": identity.process_birth_id,
            "repository_id": identity.repository_id, "ready_checked": True}
        connection = server._connection
        connection.execute("BEGIN TRANSACTION")
        try:
            reader = IntentRepository(bound_connection=connection, install_schema=False,
                owner_id=server.identity.server_id, session_id=server.identity.process_birth_id)
            value = _observe_on_connection(admission=admission, intent=reader,
                connection=connection, source_transition=source_transition, native_owner_binding=binding)
            connection.execute("COMMIT")
        except BaseException as exc:
            connection.execute("ROLLBACK")
            if isinstance(exc, duckdb.Error):
                raise local.LocalPlanningError("native owner requirement observation query is unavailable") from exc
            raise
        if reader.event_watermark() != value["native_event_watermark"] or server.identity != identity:
            raise local.LocalPlanningError("native intent changed after owner requirement capture")
        return value
