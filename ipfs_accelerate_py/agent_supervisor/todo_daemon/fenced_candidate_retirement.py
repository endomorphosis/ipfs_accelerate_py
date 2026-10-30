"""Audited retirement of fenced rejected candidates; UNKNOWN stays UNKNOWN.

Only an exact campaign candidate request plus the registered stopped service
fence permits this administrative closeout. It grants no provider replay,
completion, cancellation or remote-effect proof. A separately admitted Native
candidate continuation may subsequently revalidate the retained source.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

FAILURE_KIND = "administrative_fenced_validation_outcome_unknown"
AUDIT_EVENT = "fenced_validation_candidate_retirement_admitted"
SCHEMA = "database-fenced-validation-candidate-retirement@1"
_IDENTITY = ("attempt_id", "task_cid", "task_alias", "claim_id", "lease_id", "owner_session_id", "fencing_token", "fence_epoch", "attempt_number")


def need(value, reason):
    if not value:
        raise ValueError(reason)


def digest(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _audit(daemon, attempt):
    rows = daemon._require_connection().execute(
        "SELECT body_json FROM daemon_execution_events WHERE attempt_id=? AND event_type=? LIMIT 2",
        [attempt.attempt_id, AUDIT_EVENT]).fetchall()
    need(len(rows) == 1, "fenced candidate retirement audit missing or ambiguous")
    body = json.loads(rows[0][0])
    need(body.get("schema") == SCHEMA and body.get("audit_id") == digest(
        {key: value for key, value in body.items() if key != "audit_id"}), "fenced candidate audit identity differs")
    return body


def _verify_preimage(daemon, attempt, audit, receipt):
    from .planning_unknown_abandonment_settlement import native_stores, provider_rows
    need(native_stores(daemon) == audit["stores"], "fenced candidate physical store scope differs")
    before = audit["attempt"]
    actual = attempt.to_dict()
    mutable = {"status", "committed_phase", "revision", "finished_at_ms", "updated_at_ms"}
    need({key:value for key,value in actual.items() if key not in mutable} ==
         {key:value for key,value in before.items() if key not in mutable}, "fenced candidate immutable attempt changed")
    need(provider_rows(daemon, attempt.attempt_id) == audit["providers"], "fenced candidate raw UNKNOWN provider row changed")
    need(daemon.effect_claim_recorded(attempt.attempt_id,
         idempotency_key="effect:" + attempt.attempt_id) is None, "fenced candidate has accepted execution effects")
    phases = daemon.phase_history(attempt.attempt_id)
    if attempt.status == "running":
        need(actual == before and phases == audit["phases"], "fenced candidate running preimage changed")
    else:
        need(attempt.status == "failed" and phases[:-1] == audit["phases"]
             and len(phases) == len(audit["phases"]) + 1 and phases[-1]["phase"] == "failed"
             and phases[-1]["body"] == dict(receipt), "fenced candidate failed crash stage differs")
    claim = daemon.coordinator.get_task_claim(attempt.claim_id)
    coordinated = daemon.coordinator.get_task_attempt(attempt.attempt_id)
    need(claim is not None and coordinated is not None, "fenced candidate coordination missing")
    for key in ("attempt_id", "task_cid", "lease_id", "owner_session_id", "fencing_token", "fence_epoch", "attempt_number"):
        need(type(getattr(claim, key)) is type(before[key]) and getattr(claim, key) == before[key],
             "fenced candidate claim identity differs")
    states = (str(getattr(claim.state, "value", claim.state)),
              str(getattr(coordinated.status, "value", coordinated.status)))
    need(states in {("accepted", "running"), ("released", "failed")}, "fenced candidate coordination crash stage differs")
    task = daemon.task_source.get_task(attempt.task_cid)
    original_task = audit["task"]
    historical = daemon.coordinator.observe_rearmed_failed_task_claim(claim, failure_receipt=dict(receipt))
    if historical is not None:
        need(attempt.status == "failed" and states == ("released", "failed")
             and historical.get("failure_settlement_id") == receipt["settlement_id"]
             and historical.get("control_revision") == original_task["revision"] + 2
             and task.revision >= historical["control_revision"], "fenced candidate historical rearm binding differs")
        return
    observed = task.to_dict()
    if task.status == "retrying":
        from ..merge.database_coordination import TASK_CLAIM_FAILURE_REARM_OPERATION
        compact = {"operation":TASK_CLAIM_FAILURE_REARM_OPERATION,"settlement_id":receipt["settlement_id"]}
        body = dict(original_task["body"]); body["completion_receipt"] = compact
        need(attempt.status == "failed" and states == ("released", "failed")
             and task.revision == original_task["revision"] + 2 and task.body == body
             and all(observed[key] == original_task[key] for key in set(observed) - {"status","revision","updated_at","body"}),
             "fenced candidate retrying crash stage differs")
        return
    if task.status == "in_progress":
        need(observed == original_task, "fenced candidate canonical running task changed")
    else:
        need(attempt.status == "failed" and states == ("released", "failed")
             and task.status == "blocked" and task.revision == original_task["revision"] + 1
             and task.body.get("completion_receipt") == dict(receipt), "fenced candidate canonical blocked crash stage differs")
        immutable = set(observed) - {"status", "revision", "updated_at", "body"}
        need(all(observed[key] == original_task[key] for key in immutable), "fenced candidate canonical identity changed")
        original_body = dict(original_task["body"])
        original_body["completion_receipt"] = dict(receipt)
        need(task.body == original_body, "fenced candidate canonical body changed")


def validate_failure(daemon, attempt, receipt):
    audit = _audit(daemon, attempt)
    need(receipt.get("failure_kind") == FAILURE_KIND
         and receipt.get("failure_payload_digest") == audit["audit_id"], "fenced candidate failure lacks exact audit")
    _verify_preimage(daemon, attempt, audit, receipt)


def retire(daemon, *, config, candidate_request_path, apply=False):
    """Retire only this fenced failed-validation candidate, preserving UNKNOWN."""
    from .implementation_daemon import DatabaseImplementationDaemon, DATABASE_PORTAL_FAILURE_SETTLEMENT_SCHEMA, content_identity
    from .database_portal_bridge import DatabasePortalExecutionBridge, DatabasePortalAttemptPaths, _accepted_source_events
    from .planning_unknown_abandonment_settlement import native_stores, provider_rows
    from ..runtime.event_log import _event_identity
    from ..project_adapters.worldwide_legal_search_runtime import CampaignConfig
    from ..project_adapters.worldwide_legal_search_migration import _registered_stopped_fence
    from ..project_adapters.worldwide_legal_search_continuation import candidate_manifest, task_contract_id, _git, _FIELDS, SCHEMA as REQUEST_SCHEMA
    from ..project_adapters.worldwide_legal_search_watchdog import read_owned_json
    from .implementation_daemon import parse_task_text

    need(type(daemon) is DatabaseImplementationDaemon and type(config) is CampaignConfig
         and daemon.owner_session_id == config.run_id and daemon.require_real_execution is True,
         "fenced candidate requires the genuine campaign execution owner")
    stores = native_stores(daemon)
    need(stores == {"canonical":str(config.database), "execution":str(config.state_root / "execution.duckdb"),
                   "coordination":str(config.state_root / "coordination.duckdb")}, "fenced candidate actual store crosswalk differs")
    fence = _registered_stopped_fence(config)
    request_path = Path(candidate_request_path)
    need(request_path.is_absolute() and request_path.resolve(strict=True) == request_path
         and request_path.is_relative_to(config.state_root), "fenced candidate request outside exact owned state")
    request = read_owned_json(request_path)
    need(set(request) == _FIELDS and request.get("schema") == REQUEST_SCHEMA
         and request.get("receipt_id") == digest({k:v for k,v in request.items() if k != "receipt_id"})
         and request.get("run_id") == config.run_id, "fenced candidate request identity differs")
    attempt = daemon.get_attempt(request["predecessor_attempt_id"])
    need(attempt is not None and attempt.task_alias == request["task_id"]
         and attempt.task_cid == request["database_task_cid"]
         and attempt.attempt_number == request["predecessor_attempt_number"]
         and request["next_attempt_number"] == attempt.attempt_number + 1 <= config.max_task_attempts,
         "fenced candidate native attempt or remaining budget differs")
    if attempt.status == "failed":
        need(_audit(daemon,attempt)["candidate_request"] == request, "fenced candidate replay request changed")
        failure = daemon._portal_failure_phase_receipt(attempt)
        validate_failure(daemon, attempt, failure)
        if apply:
            claim = daemon.coordinator.get_task_claim(attempt.claim_id)
            daemon._fail_coordination_claim(claim,failure)
            daemon._settle_portal_failure_control_task(attempt,failure)
            validate_failure(daemon,attempt,failure)
        return {"schema": SCHEMA, "applied":bool(apply), "replayed":True, "failure_receipt":failure,
                "provider_outcome":"unknown", "raw_unknown_rows_preserved":True, "automatic_retry_admitted":False,
                "completion_authority":False}
    need(attempt.status == "running" and attempt.committed_phase in {"claimed", "context"},
         "fenced candidate requires an unaccepted running callback")
    need([row.attempt_id for row in daemon.list_running_attempts()] == [attempt.attempt_id],
         "fenced candidate requires the sole exact running native attempt")
    providers = provider_rows(daemon, attempt.attempt_id)
    need(len(providers) == 1 and providers[0]["owner_session_id"] == attempt.owner_session_id
         and providers[0]["task_cid"] == attempt.task_cid
         and providers[0]["idempotency_key"] == "provider:" + attempt.attempt_id,
         "fenced candidate callback inventory differs")
    callback = daemon.provider_invocation_recorded(attempt.attempt_id, idempotency_key="provider:" + attempt.attempt_id)
    expected = {"schema":"database-portal-callback-intent@1", "callback_state":"started_outcome_unknown",
                "provider_effect_state":"unknown_may_have_started", **{key:getattr(attempt,key) for key in
                ("attempt_id", "task_cid", "claim_id", "lease_id", "fencing_token", "fence_epoch")}}
    need(callback == expected and daemon.effect_claim_recorded(attempt.attempt_id,
         idempotency_key="effect:" + attempt.attempt_id) is None
         and daemon._attempt_execution_evidence_counts(attempt.attempt_id)["effect_claim_count"] == 0,
         "fenced candidate unknown callback or effects differ")
    root = config.state_root / "native/wwl_database_portal_attempts" / hashlib.sha256(attempt.attempt_id.encode()).hexdigest()[:24]
    paths = DatabasePortalAttemptPaths(root=root,task_projection=root / "task-projection.runtime.todo.md",
        binding=root / "database-attempt-binding.json",state=root / "portal-task-state.json",
        strategy=root / "portal-strategy.json",events=root / "portal-events.jsonl",implementation_logs=root / "implementation-logs")
    binding = DatabasePortalExecutionBridge._strict_binding(paths.binding)
    need(binding["attempt_id"] == attempt.attempt_id and binding["task_cid"] == attempt.task_cid
         and binding["binding_id"] == request["predecessor_binding_id"]
         and binding["projection_immutable_digest"] == request["predecessor_projection_immutable_digest"],
         "fenced candidate sealed attempt binding differs")
    projection = DatabasePortalExecutionBridge._verify_projection(paths, binding)
    tasks = parse_task_text(projection,path=paths.task_projection,task_header_prefix="## " + attempt.task_alias)
    need(len(tasks) == 1 and task_contract_id(tasks[0]) == request["task_contract_id"], "fenced candidate task contract differs")
    events, events_digest = _accepted_source_events(paths.events)
    previous = ""
    for sequence,event in enumerate(events,1):
        need(event.get("sequence") == sequence and event.get("previous_event_id") == previous
             and event.get("event_id") == _event_identity(event), "fenced candidate native event chain differs")
        previous = event["event_id"]
    finished = [event for event in events if event.get("type") == "implementation_finished"
                and event.get("provider_dispatched") is True and event.get("attempt_consumed") is True]
    need(len(finished) == 1 and not any(event.get("type") in {"merge_started","merge_finished","task_completed"}
         for event in events), "fenced candidate native terminal inventory differs")
    terminal = finished[0]
    need(terminal.get("task_id") == attempt.task_alias and terminal.get("validation_result",{}).get("attempted") is True
         and terminal.get("validation_result",{}).get("passed") is False
         and terminal.get("merge_result",{}).get("merged") is False and terminal.get("merge_result",{}).get("queued",False) is False
         and terminal.get("board_completion",{}).get("complete") is False
         and terminal.get("failed_preservation_result",{}).get("preserved") is True
         and terminal.get("lifecycle_finalize",{}).get("finalized") is True,
         "fenced candidate requires exact retained rejected Native validation")
    workspace = Path(request["worktree_path"])
    repo = Path(request["repository_root"])
    admitted = next(task for task in json.loads(config.plan_path.read_text())["tasks"] if task["id"] == attempt.task_alias)
    need(repo == (config.site_root if admitted["repository"] == "site" else config.framework_root)
         and workspace.resolve(strict=True) == workspace and workspace.is_relative_to(config.state_root / "worktrees")
         and _git(workspace,"symbolic-ref","--short","HEAD") == request["branch_name"]
         and _git(workspace,"rev-parse","HEAD^{tree}") == request["candidate_tree"]
         and _git(workspace,"rev-parse","--path-format=absolute","--git-common-dir") ==
             _git(repo,"rev-parse","--path-format=absolute","--git-common-dir")
         and candidate_manifest(workspace, request["baseline_commit"],request["candidate_commit"]) == request["changed_files"]
         and set(request["changed_files"]) <= set(tasks[0].outputs), "fenced candidate retained bytes or custody differ")
    need("worktree " + str(workspace) in _git(repo,"worktree","list","--porcelain").splitlines(),
         "fenced candidate is not a listed repository worktree")
    # Require exact source ancestry from the native retained commit, even when
    # its three declared outputs are replayed onto a repaired framework base.
    import subprocess
    retained = terminal["failed_preservation_result"]["preserved_commit"]
    import re
    need(type(retained) is str and re.fullmatch(r"[0-9a-f]{40}",retained), "fenced candidate retained commit invalid")
    for path in request["changed_files"]:
        original = subprocess.run(["git","show",retained + ":" + path],cwd=repo,capture_output=True,check=True,timeout=30).stdout
        need(hashlib.sha256(original).hexdigest() == request["changed_files"][path]["sha256"],
             "fenced candidate differs from the actual retained Native output")
    task = daemon.task_source.get_task(attempt.task_cid)
    audit = {"schema":SCHEMA, "stores":stores, "attempt":attempt.to_dict(), "phases":daemon.phase_history(attempt.attempt_id),
             "task":task.to_dict(), "providers":providers, "candidate_request":request,
             "native_finished_event_id":terminal["event_id"], "native_events_digest":events_digest, "registered_stopped_fence":fence,
             "provider_outcome":"unknown", "remote_cancellation_proved":False, "callback_closed":False,
             "automatic_retry_admitted":False,"completion_authority":False}
    audit["audit_id"] = digest(audit)
    report = {"schema":SCHEMA,"applied":False,"audit_id":audit["audit_id"],"task_id":attempt.task_alias,
              "attempt_id":attempt.attempt_id,"attempt_number_preserved":attempt.attempt_number,
              "provider_outcome":"unknown","raw_unknown_rows_preserved":True,"automatic_retry_admitted":False,
              "completion_authority":False}
    if not apply:
        return report
    rechecked_fence = _registered_stopped_fence(config)
    need(_accepted_source_events(paths.events)[1] == events_digest
         and {key:value for key,value in rechecked_fence.items() if key != "observed_at"} ==
             {key:value for key,value in fence.items() if key != "observed_at"},
         "fenced candidate changed before native retirement")
    existing = daemon._require_connection().execute(
        "SELECT body_json FROM daemon_execution_events WHERE attempt_id=? AND event_type=? LIMIT 2",
        [attempt.attempt_id,AUDIT_EVENT]).fetchall()
    if existing:
        saved = _audit(daemon,attempt)
        stable = lambda value: {key:item for key,item in value.items() if key not in {"audit_id", "registered_stopped_fence"}}
        need(stable(saved) == stable(audit), "fenced candidate retirement preimage changed after audit")
        audit = saved
    else:
        daemon._record_event(AUDIT_EVENT,attempt_id=attempt.attempt_id,task_cid=attempt.task_cid,body=audit)
    failure = daemon._build_portal_failure_settlement_receipt(attempt,failure=RuntimeError(FAILURE_KIND))
    failure.update(failure_kind=FAILURE_KIND,failure_payload_digest=audit["audit_id"])
    failure.pop("settlement_id")
    failure["settlement_id"] = content_identity(failure)
    daemon._validate_portal_failure_settlement_receipt(failure,attempt=attempt)
    claim = daemon.coordinator.get_task_claim(attempt.claim_id)
    attempt = daemon._commit_portal_failure_phase(attempt,failure)
    daemon._fail_coordination_claim(claim,failure)
    control = daemon._settle_portal_failure_control_task(attempt,failure)
    validate_failure(daemon,attempt,failure)
    need(provider_rows(daemon,attempt.attempt_id) == providers, "fenced candidate UNKNOWN row changed during retirement")
    return {**report,"applied":True,"failure_receipt":failure,"control":control}


def rearm(daemon, *, config, candidate_request_path, apply=False):
    """Admit one separately fenced no-model successor, never refund history."""
    from ..project_adapters.worldwide_legal_search_migration import _registered_stopped_fence
    from ..project_adapters.worldwide_legal_search_watchdog import read_owned_json
    from ..merge.database_coordination import TASK_CLAIM_FAILURE_REARM_OPERATION
    _registered_stopped_fence(config)
    path = Path(candidate_request_path)
    need(path.is_absolute() and path.resolve(strict=True) == path and path.is_relative_to(config.state_root),
         "candidate rearm request outside exact owned state")
    request = read_owned_json(path)
    attempt = daemon.get_attempt(request["predecessor_attempt_id"])
    need(attempt is not None and attempt.status == "failed", "candidate rearm requires its audited failed predecessor")
    audit = _audit(daemon,attempt)
    need(audit["candidate_request"] == request and config.run_id == daemon.owner_session_id
         and audit["stores"] == {"canonical":str(config.database),"execution":str(config.state_root / "execution.duckdb"),
                                "coordination":str(config.state_root / "coordination.duckdb")},
         "candidate rearm request or actual campaign stores differ from retirement audit")
    failure = daemon._portal_failure_phase_receipt(attempt)
    task = daemon.task_source.get_task(attempt.task_cid)
    compact = {"operation":TASK_CLAIM_FAILURE_REARM_OPERATION,"settlement_id":failure["settlement_id"]}
    prior = daemon.coordinator.observe_rearmed_failed_task_claim(
        daemon.coordinator.get_task_claim(attempt.claim_id),failure_receipt=failure)
    blocked = (task.status == "blocked" and task.revision == audit["task"]["revision"] + 1
               and task.body.get("completion_receipt") == failure)
    retrying = (task.status == "retrying" and task.revision == audit["task"]["revision"] + 2
                and task.body.get("completion_receipt") == compact)
    need(blocked or retrying, "candidate rearm exact blocked or retrying crash stage changed")
    before_phases = daemon.phase_history(attempt.attempt_id)
    report = {"schema":SCHEMA,"applied":False,"audit_id":audit["audit_id"],"task_id":attempt.task_alias,
        "attempt_id":attempt.attempt_id,"attempt_number_preserved":attempt.attempt_number,
        "next_attempt_number":request["next_attempt_number"],"provider_outcome":"unknown","replayed":prior is not None,
        "raw_unknown_rows_preserved":True,"paid_attempt_budget_reset":False,"completion_authority":False,
        "successor_protocol":"native_candidate_validation_without_provider"}
    if not apply:
        return report
    # Publish the no-model route before opening canonical claimability. A
    # crash after either CAS must reopen this exact candidate protocol rather
    # than ordinary provider dispatch against an UNKNOWN predecessor.
    from ..project_adapters.worldwide_legal_search_runtime import atomic_json
    pointer = {"schema":"wwl.pending-candidate-continuation@1","run_id":config.run_id,
        "request_path":str(path),"request_sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    pointer_path = config.state_root / "control/pending-candidate-continuation.json"
    pointer_path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    need(pointer_path.parent.resolve(strict=True) == pointer_path.parent, "candidate rearm control directory aliased")
    if pointer_path.exists():
        need(read_owned_json(pointer_path) == pointer, "candidate rearm pending continuation differs")
    else:
        atomic_json(pointer_path,pointer)
    changed = daemon.task_source.rearm_blocked_task(task.task_cid,receipt=compact) if blocked else task
    coordinated = daemon.coordinator.rearm_failed_task(failure_receipt=failure,control_task_observation=changed.to_dict())
    need(daemon.phase_history(attempt.attempt_id) == before_phases, "candidate rearm changed failed predecessor history")
    return {**report,"applied":True,"control_revision":changed.revision,"coordination_receipt":coordinated}


def permits_candidate_claim(daemon, task_cid, prior_attempt_id):
    """Bypass one UNKNOWN exclusion only for the bound no-provider protocol."""
    from .database_portal_bridge import DatabasePortalExecutionBridge
    from ..project_adapters.worldwide_legal_search_runtime import CampaignRuntime
    try:
        previous = daemon.get_attempt(prior_attempt_id)
        if previous is None or previous.task_cid != task_cid or previous.status != "failed":
            return False
        official = daemon._resolve_native_portal_provider_callback(daemon._provider_fn,previous)
        bridge = getattr(official,"__self__",None)
        if (type(bridge) is not DatabasePortalExecutionBridge
                or getattr(official,"__func__",None) is not DatabasePortalExecutionBridge.run_provider):
            return False
        loader = bridge.candidate_reconciliation_loader
        runtime = getattr(loader,"__self__",None)
        if (type(runtime) is not CampaignRuntime or getattr(loader,"__func__",None) is not CampaignRuntime._load_candidate_admission
                or runtime._daemon is not daemon or runtime._candidate_request_path is None):
            return False
        request = runtime._read_candidate_request()
        audit = _audit(daemon,previous)
        if (audit["candidate_request"] != request or request["predecessor_attempt_id"] != prior_attempt_id
                or request["database_task_cid"] != task_cid
                or request["next_attempt_number"] != previous.attempt_number + 1):
            return False
        failure = daemon._portal_failure_phase_receipt(previous)
        validate_failure(daemon,previous,failure)
        claim = daemon.coordinator.get_task_claim(previous.claim_id)
        rearm = daemon.coordinator.observe_rearmed_failed_task_claim(claim,failure_receipt=failure)
        task = daemon.task_source.get_task(task_cid)
        return bool(rearm is not None and task.status == "retrying"
            and task.revision == audit["task"]["revision"] + 2)
    except (ValueError,KeyError,TypeError,OSError):
        return False
