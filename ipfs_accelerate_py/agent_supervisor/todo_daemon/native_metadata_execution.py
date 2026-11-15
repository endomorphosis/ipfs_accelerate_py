"""One explicitly selected, signed metadata execution; completion stays manual.

This route does not enter Portal or a provider phase. The original intent,
packet authority, fenced coordinator, typed local policy and validation
scheduler retain their respective authority. An original enclosing resource
owner must admit the whole operation before invoking the daemon.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import time

from ..task_sources.control_plane_contracts import content_identity
from ..merge.database_coordination import LeaseKind, exclusive_scope_key
from ..task_sources.database_task_source import DatabaseTaskSource
from ..task_sources.execution_packet_admission import (
    EVIDENCE_KIND, NativeExecutionPacketAuthority, PURPOSES,
)
from ..task_sources.intent_repository import IntentRepository
from ..runtime.resource_scheduler import ResourceLeaseBudget
from ..validation.validation_commands import build_validation_commands
from ..validation.validation_scheduler import ValidationScheduler, PreparedValidationDisposition
from .native_metadata_parent_resources import make_prepared_validation_admission
from .task_execution_policy import (
    ExecutionMode, ExecutionStatus, LocalOperationType, TaskContextMetadata,
    TaskExecutionPolicy, TaskExecutionRequest, TypedLocalOperation,
    MAX_TASK_CONTEXT_BYTES, MAX_TASK_CONTEXT_TOKENS,
)

EXECUTION_POLICY_SCHEMA = "native-metadata-validation-execution/v1"
EXECUTION_POLICY_FIELD = "execution_packet_execution_policy"
RESULT_SCHEMA = "native-metadata-validation-execution-result/v1"
_POLICY_FIELDS = frozenset({
    "schema", "task_cid", "purpose", "execution_mode", "completion_mode",
    "max_attempts", "provider_authority", "completion_authority",
})
_EXECUTION_EVIDENCE = "native_metadata_execution"


class NativeMetadataExecutionError(ValueError):
    """The selected native task has no current execution-only authorization."""


def _need(value, reason):
    if not value:
        raise NativeMetadataExecutionError(reason)


def _plain(value):
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _policy(task):
    body = _plain(task.body)
    policy = body.get(EXECUTION_POLICY_FIELD)
    _need(type(policy) is dict and set(policy) == _POLICY_FIELDS,
          "metadata_execution_policy_shape")
    _need(policy["schema"] == EXECUTION_POLICY_SCHEMA
          and policy["task_cid"] == task.task_cid
          and policy["purpose"] in PURPOSES
          and policy["execution_mode"] == "deterministic-only"
          and policy["completion_mode"] == "manual"
          and type(policy["max_attempts"]) is int and policy["max_attempts"] == 1
          and policy["provider_authority"] is False
          and policy["completion_authority"] is False,
          "metadata_execution_policy_not_execution_only")
    _need(body.get("completion") == "manual"
          and body.get("provider_role") == "deterministic-only"
          and body.get("context_budget_tokens") == "0"
          and body.get("execution_packet_required") is True
          and body.get("code_edit_allowlist", []) == []
          and body.get("review_only", False) is False
          and body.get("is_schedulable", True) is True,
          "metadata_execution_task_mode")
    return policy


def _current(source, authority, cid, *, phase="dispatch", revision=None):
    task = source.get(cid)
    _need(task is not None and task.task_cid == cid, "metadata_execution_task_missing")
    if revision is not None:
        _need(type(revision) is int and task.revision == revision,
              "metadata_execution_revision_changed")
    policy = _policy(task)
    # This original check verifies the native signature, independent observed
    # validation, whole authority spec, dependency receipts, profiles and files.
    accepted = authority.require(cid, phase=phase)
    _need(type(accepted) is dict and accepted.get("purpose") == policy["purpose"],
          "metadata_execution_packet_purpose_changed")
    matching = [row for row in source.intent.current_evidence_for_task(cid)
                if row["evidence_kind"] == EVIDENCE_KIND
                and row["digest"] == accepted["contract_digest"]]
    _need(len(matching) == 1, "metadata_execution_contract_not_unique")
    envelope = _plain(matching[0]["body"]).get("execution_packet_contract")
    _need(type(envelope) is dict and content_identity(envelope) == accepted["contract_digest"],
          "metadata_execution_contract_changed")
    packet = envelope["payload"]["packet"]
    rows = [_plain(row) for row in task.validations]
    declared = [row.get("argv") for row in rows]
    _need(len(rows) == 2 and declared[0] == packet["argv"]
          and declared[0] != declared[1]
          and all(type(argv) is list and bool(argv)
                  and all(type(part) is str and bool(part) and "\x00" not in part for part in argv)
                  for argv in declared), "metadata_execution_declared_commands_changed")
    # packet.validation_argv is the already-observed PRE-ADMISSION check.
    # The post-output verifier is separately bound by the native authority
    # spec's second declared validation; do not replace it with that precheck.
    paths = {pin["path"] for pin in packet["source_descriptors"]}
    _need(all(any(path in argv for path in paths) for argv in declared),
          "metadata_execution_declared_command_source_unbound")
    for row, role in zip(rows, ("producer", "independent_verifier")):
        metadata = row.get("policy", {})
        _need(metadata.get("purpose") == policy["purpose"]
              and metadata.get("role") == role
              and metadata.get("provider_allowed") is False,
              "metadata_execution_validation_roles_changed")
    limits = packet["budgets"]
    _need(limits["max_attempts"] == 1 and limits["max_tokens"] == 0
          and 0 < limits["deadline_seconds"] <= 300,
          "metadata_execution_budget_changed")
    return task, accepted, packet, declared



def _scheduler_context(task, accepted, packet):
    """Use the already-admitted metadata contract, not code-edit discovery."""
    from ..control.profile_authority import load_local_profile
    reference = _plain(task.body)["execution_packet_policy"]["owner"]
    profile = load_local_profile(repository_cid=reference["repository_cid"],
        profile_dir=Path(reference["profile_dir"]),
        lifecycle_dir=Path(reference["lifecycle_dir"]), mirror=False)
    # NativeExecutionPacketAuthority already checked this exact signed profile
    # against the current repository HEAD. Recheck its immutable native IDs;
    # never invent a commit or reinterpret the pre-admission CHECK as execution.
    _need(profile.profile_id == reference["profile_id"]
          and profile.content_id == reference["content_id"]
          and profile.identity_did == reference["identity_did"]
          and profile.repository_cid == reference["repository_cid"]
          and profile.allows("read") and profile.allows("test"),
          "metadata_execution_scheduler_profile_changed")
    head = profile.baseline_commit
    _need(type(head) is str and len(head) in (40, 64)
          and all(char in "0123456789abcdef" for char in head),
          "metadata_execution_scheduler_profile_commit_invalid")
    # The authenticated contract digest includes completed dependency receipts;
    # source/input descriptors remain explicit report context. Code edits are
    # forbidden and both declared commands always require full validation.
    dependencies = {"schema": "native-metadata-scheduler-context/v1",
        "contract_digest": accepted["contract_digest"],
        "authority_spec_cid": accepted["authority_spec_cid"],
        "source_descriptors": _plain(packet["source_descriptors"]),
        "inputs": _plain(packet["inputs"])}
    return {"target_commit": head, "changed_files": (),
            "dependency_state": dependencies}


def _durable_json(path, value):
    raw = (json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _existing_journal(path):
    """Hash a claim journal the in-progress attempt already published."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
              and info.st_uid == os.geteuid() and 0 < info.st_size <= 65536,
              "metadata_execution_resume_journal_changed")
        raw = os.read(fd, 65537)
        _need(len(raw) == info.st_size, "metadata_execution_resume_journal_changed")
    finally:
        os.close(fd)
    _need(not any(candidate.is_symlink() for candidate in (path, *path.parents)),
          "metadata_execution_resume_journal_changed")
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


class _ResumeClaim:
    """Claim identity already stored on the control task. It has no live lease."""

    def __init__(self, receipt, task_cid):
        self.claim_id = receipt["claim_id"]
        self.task_cid = task_cid
        self.owner_session_id = receipt["owner_session_id"]
        self.fencing_token = receipt["fencing_token"]
        self.fence_epoch = receipt["fence_epoch"]
        self.attempt_id = receipt["attempt_id"]
        self.attempt_number = receipt["attempt_number"]
        self.lease_id = receipt["lease_id"]
        self._receipt = receipt

    def to_dict(self):
        return dict(self._receipt)


def _outputs(packet):
    root = Path(packet["cwd"])
    rows = []
    total = 0
    for relative in packet["outputs"]:
        path = root / relative
        _need(not Path(relative).is_absolute() and ".." not in Path(relative).parts
              and path != root and not any(p.is_symlink() for p in (path, *path.parents)),
              "metadata_execution_output_alias_or_scope")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            before = os.fstat(fd)
            _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                  and before.st_uid == os.geteuid(), "metadata_execution_output_not_owned_regular")
            total += before.st_size
            _need(total <= packet["budgets"]["storage_bytes"], "metadata_execution_output_budget")
            digest = hashlib.sha256()
            while block := os.read(fd, 1024 * 1024):
                digest.update(block)
            os.fsync(fd)
            def identity(value):
                return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
                        value.st_uid, value.st_gid, value.st_size,
                        value.st_mtime_ns, value.st_ctime_ns)
            _need(identity(os.fstat(fd)) == identity(before)
                  and identity(path.stat()) == identity(before),
                  "metadata_execution_output_changed")
        finally:
            os.close(fd)
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        rows.append({"path": str(path), "bytes": before.st_size, "sha256": digest.hexdigest()})
    return rows


def _stopped_observation(observed, command):
    rows = observed.get("results", ())
    _need(type(rows) is list and len(rows) == 1, "metadata_execution_stop_witness_missing")
    row = rows[0]
    custody = row.get("validation_process_custody", {})
    _need(row.get("command") == command and row.get("cache_hit") is False
          and custody.get("schema") == "native-validation-process-custody/v1"
          and custody.get("supported") is True
          and custody.get("children_reaped") is True
          and custody.get("termination_signal") == 0
          and type(custody.get("returncode")) is int
          and custody["returncode"] == row.get("returncode")
          and custody.get("timed_out") is False
          and custody.get("output_limit_exceeded") is False,
          "metadata_execution_stop_witness_missing")
    return {"returncode": custody["returncode"], "children_reaped": True,
            "termination_signal": 0, "timed_out": False,
            "output_limit_exceeded": False}


def _claim_intent_root(database):
    path = Path(database).with_name("." + Path(database).name + ".native-metadata-execution")
    try:
        path.mkdir(mode=0o700)
    except FileExistsError:
        pass
    st = path.lstat()
    _need(path.is_absolute() and stat.S_ISDIR(st.st_mode)
          and st.st_uid == os.geteuid() and not st.st_mode & 0o077
          and not any(p.is_symlink() for p in (path, *path.parents)),
          "metadata_execution_claim_journal_not_private")
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return path


def _invocation(authority_mode, max_task_attempts, execution_slice_task_cids, task_cid):
    _need(authority_mode in {"embedded", "embedded_exclusive"}
          and max_task_attempts == 1
          and execution_slice_task_cids == frozenset({task_cid}),
          "metadata_execution_exact_single_owner_slice_required")


def execute_native_metadata_validation(daemon, *, task_cid, expected_revision):
    """Execute a single manual task's signed producer/verifier and release it.

    Successful execution is a candidate awaiting independent native acceptance.
    A raised/unknown command outcome retains custody; this function never
    guesses that a child is stopped, retries a task or marks one completed.
    """
    from .implementation_daemon import DatabaseImplementationDaemon
    _need(type(daemon) is DatabaseImplementationDaemon,
          "metadata_execution_original_daemon_required")
    _invocation(daemon.authority_mode, daemon.max_task_attempts,
                daemon.execution_slice_task_cids, task_cid)
    source = daemon.task_source
    _need(type(source) is DatabaseTaskSource and type(source.intent) is IntentRepository,
          "metadata_execution_original_intent_owner_required")
    authority = NativeExecutionPacketAuthority(source.intent)
    existing = source.get(task_cid)
    stored_receipt = _plain(existing.body).get("completion_receipt") if existing is not None else None
    resuming = (
        existing is not None
        and existing.status == "in_progress"
        and existing.revision == expected_revision + 1
        and type(stored_receipt) is dict
        and stored_receipt.get("operation") == "database_claim"
        and stored_receipt.get("owner_session_id") == daemon.owner_session_id
        and stored_receipt.get("attempt_number") == 1
        and type(stored_receipt.get("attempt_id")) is str
        and stored_receipt["attempt_id"]
    )
    if resuming:
        task, accepted, packet, declared = _current(
            source, authority, task_cid, phase="dispatch", revision=expected_revision + 1)
    else:
        task, accepted, packet, declared = _current(
            source, authority, task_cid, phase="claim", revision=expected_revision)
        _need(task.status == "ready", "metadata_execution_task_not_ready")
    _need(not daemon.execution_slice_task_ids
          or daemon.execution_slice_task_ids == frozenset({task.task_alias}),
          "metadata_execution_alias_slice_changed")
    scheduler_context = _scheduler_context(task, accepted, packet)
    prior = source.intent.current_evidence_for_task(task_cid)
    execution_rows = [row for row in prior if row["evidence_kind"] == _EXECUTION_EVIDENCE]
    if resuming:
        _need(execution_rows, "metadata_execution_resume_evidence_missing")
        for row in execution_rows:
            body = _plain(row["body"])
            execution = body.get("execution")
            _need(body.get("state") in {"claim_intended", "started"},
                  "metadata_execution_resume_evidence_changed")
            if execution is not None:
                _need(execution.get("passed") is False and execution.get("commands") == []
                      and execution.get("outputs") == [] and execution.get("claim_released") is False,
                      "metadata_execution_resume_command_already_recorded")
        _need(all(not (Path(packet["cwd"]) / relative).exists() for relative in packet["outputs"]),
              "metadata_execution_resume_output_exists")
    else:
        _need(not execution_rows, "metadata_execution_prior_attempt_requires_operator_acceptance")
    dispatch_control = getattr(daemon, "_native_dispatch_control", None)
    if dispatch_control is not None:
        _need(dispatch_control.before_claim().get("new_dispatch_permitted") is True,
              "metadata_execution_native_dispatch_closed")
    root = Path(daemon.execution_state_dir)
    _need(root.is_absolute() and root.is_dir()
          and not any(p.is_symlink() for p in (root, *root.parents))
          and root.stat().st_uid == os.geteuid() and not root.stat().st_mode & 0o077,
          "metadata_execution_private_state_root_required")
    claim_scope = exclusive_scope_key(lease_kind=LeaseKind.TASK,
                                      scope=task_cid, task_cid=task_cid)
    _need(not daemon.coordinator.lease_events(scope_key=claim_scope, limit=1),
          "metadata_execution_prior_native_claim_forbids_retry")
    # The stable journal is beside the exact canonical database, independent
    # of a caller's fresh state/coordination directory. O_EXCL remains binding
    # after a lost claim response or failed native-evidence write; no second
    # claim is attempted for this task on a later invocation, even after a
    # new contract is signed.
    journal_key = content_identity({"task_cid": task_cid})
    journal_root = Path(daemon.database_path).with_name("." + Path(daemon.database_path).name + ".native-metadata-execution")
    journal = journal_root / (
        hashlib.sha256(journal_key.encode()).hexdigest() + ".claim-intent.json")
    intent_value = {"schema": "native-metadata-claim-intent/v1", "task_cid": task_cid,
                    "contract": accepted, "expected_revision": expected_revision,
                    "owner_session_id": daemon.owner_session_id,
                    "claim_call_attempted": False, "execution_started": False,
                    "completion_authority": False}
    deadline = time.monotonic() + packet["budgets"]["deadline_seconds"]
    milliseconds = packet["budgets"]["deadline_seconds"] * 1000
    fixed_budget = ResourceLeaseBudget(max_parallel=1,
        max_processes=packet["budgets"]["max_children"], wall_time_ms=milliseconds,
        cpu_time_ms=milliseconds * packet["budgets"]["cpu_threads"],
        memory_bytes=packet["budgets"]["memory_mib"] * 1024 * 1024,
        disk_bytes=packet["budgets"]["storage_bytes"])
    database = Path(daemon.database_path)
    paths = (database, database.with_name(database.name + ".wal"),
        database.with_name("." + database.name + ".intent.lock"),
        database.with_name("." + database.name + ".lock"),
        Path(daemon.coordination_path), root,
        journal, journal.with_suffix(".claim-call.json"),
        journal.with_suffix(".claim-outcome-unknown.json"),
        *(Path(packet["cwd"]) / relative for relative in packet["outputs"]))
    prepared = make_prepared_validation_admission(task_cid=task_cid,
        contract_digest=accepted["contract_digest"], workspace_path=Path(packet["cwd"]),
        commands=tuple(shlex.join(argv) for argv in declared), budget=fixed_budget,
        deadline_monotonic=deadline,
        demand={"cpu_slots": packet["budgets"]["cpu_threads"],
            "memory_mb": packet["budgets"]["memory_mib"],
            "child_process_slots": packet["budgets"]["max_children"],
            "storage_bytes": packet["budgets"]["storage_bytes"]},
        writable_paths=paths)
    prepared.prepare()
    disposition = PreparedValidationDisposition.NOT_STARTED
    try:
        prepared.assert_binding(task_cid=task_cid, contract_digest=accepted["contract_digest"])
        deadline = prepared.deadline_monotonic
        lease = None
        if resuming:
            # The control receipt and journals are the claim. A new attempt
            # database cannot see the old lease, so do not claim or fence again.
            call_journal = journal.with_suffix(".claim-call.json")
            unknown_journal = journal.with_suffix(".claim-outcome-unknown.json")
            _need(journal.is_file() and call_journal.is_file() and not unknown_journal.exists(),
                  "metadata_execution_resume_journal_changed")
            _need(set(stored_receipt) == {"operation", "claim_id", "attempt_id", "lease_id",
                  "attempt_number", "owner_session_id", "fencing_token", "fence_epoch"}
                  and type(stored_receipt["claim_id"]) is str and stored_receipt["claim_id"]
                  and type(stored_receipt["lease_id"]) is str and stored_receipt["lease_id"]
                  and type(stored_receipt["fencing_token"]) is int and stored_receipt["fencing_token"] > 0
                  and type(stored_receipt["fence_epoch"]) is int and stored_receipt["fence_epoch"] > 0,
                  "metadata_execution_resume_receipt_changed")
            intent_ref = _existing_journal(journal)
            call_ref = _existing_journal(call_journal)
            claim = _ResumeClaim(stored_receipt, task_cid)
        else:
            _claim_intent_root(daemon.database_path)
            intent_ref = _durable_json(journal, intent_value)
            source.intent.record_evidence(task_cid=task_cid, evidence_kind=_EXECUTION_EVIDENCE,
                digest=content_identity(intent_value), body={"state": "claim_intended",
                    "contract_digest": accepted["contract_digest"], "claim_intent": intent_ref})
            # The existing native packet check authenticated all completed dependency
            # receipts. Preserve the coordinator's established projection convention.
            daemon.coordinator.register_task(task_cid=task_cid, task_id=task.task_alias,
                dependency_task_cids=(), body={"task_alias": task.task_alias,
                                             "execution_packet_required": True})
            task, current, packet, declared = _current(
                source, authority, task_cid, phase="claim", revision=expected_revision)
            _need(current == accepted, "metadata_execution_acceptance_changed_before_claim")
            _need(not daemon.coordinator.lease_events(scope_key=claim_scope, limit=1),
                  "metadata_execution_prior_native_claim_forbids_retry")
            call_ref = _durable_json(journal.with_suffix(".claim-call.json"),
                {**intent_value, "claim_call_attempted": True})
            disposition = PreparedValidationDisposition.UNKNOWN
            try:
                claim = daemon.coordinator.claim_task(task_cid=task_cid,
                    owner_session_id=daemon.owner_session_id,
                    lease_ms=packet["budgets"]["deadline_seconds"] * 1000,
                    execution_packet_authority=authority)
            except BaseException as error:
                # The call may have committed before a lost response. Do not claim
                # again or guess a lease identity; the exclusive call intent survives
                # even if writing this optional diagnostic also fails.
                _durable_json(journal.with_suffix(".claim-outcome-unknown.json"),
                    {**intent_value, "claim_call_attempted": True,
                     "claim_outcome_unknown": True, "error_type": type(error).__name__})
                raise
            lease = daemon.coordinator.get_lease(claim.lease_id)
            _need(lease is not None, "metadata_execution_claim_lease_missing")
        # No provider/effect phase or execution-sidecar attempt is fabricated.
        result = {"schema": RESULT_SCHEMA, "task_cid": task_cid,
            "claim": claim.to_dict(), "contract": accepted, "claim_intent": intent_ref,
            "claim_call": call_ref, "provider_dispatched": False,
            "completion_authority": False, "automatic_retry_admitted": False,
            "passed": False, "commands": [], "outputs": [], "claim_released": False}
        in_flight = False
        control_uncertain = False
        claimed_task = None
        receipt_path = root / ("native-metadata-" + hashlib.sha256(
            claim.attempt_id.encode()).hexdigest() + ".json")
        try:
            _need(claim.attempt_number == 1, "metadata_execution_retry_forbidden")
            if resuming:
                receipt = stored_receipt
                claimed_task = task
                _need(claimed_task is not None and claimed_task.status == "in_progress"
                      and claimed_task.revision == expected_revision + 1
                      and _plain(claimed_task.body).get("completion_receipt") == receipt,
                      "metadata_execution_claim_response_not_proven")
            else:
                receipt = daemon._database_claim_receipt(claim)
                control_uncertain = True
                try:
                    source.compare_and_set_status(task_cid, expected_revision, "in_progress", receipt=receipt)
                except Exception:
                    # A CAS may have committed before losing its response. Never issue
                    # a second CAS: require one fresh exact receipt/revision readback.
                    pass
                claimed_task = source.get(task_cid)
                _need(claimed_task is not None and claimed_task.status == "in_progress"
                      and claimed_task.revision == expected_revision + 1
                      and _plain(claimed_task.body).get("completion_receipt") == receipt,
                      "metadata_execution_claim_response_not_proven")
                control_uncertain = False
                disposition = PreparedValidationDisposition.NOT_STARTED
                source.intent.record_evidence(task_cid=task_cid, evidence_kind=_EXECUTION_EVIDENCE,
                    digest=content_identity(result), body={"state": "started", "execution": result})
            plan = {"task_cid": task_cid, "contract": accepted, "commands": declared}
            plan_cid = content_identity(plan)

            def authorize(arguments, context):
                _need(dict(arguments) == {"validation_plan_cid": plan_cid}
                      and _plain(context) == plan, "metadata_execution_typed_plan_changed")
                return {"authorized": True, "validation_plan_cid": plan_cid}

            authorization = TaskExecutionPolicy(local_operation_handlers={
                LocalOperationType.DECLARED_VALIDATION_PLAN: authorize}).execute(
                    TaskExecutionRequest(task_id=task.task_alias,
                        mode=ExecutionMode.DETERMINISTIC_ONLY, context=plan,
                        context_metadata=TaskContextMetadata(max_bytes=MAX_TASK_CONTEXT_BYTES,
                                                            max_tokens=MAX_TASK_CONTEXT_TOKENS),
                        local_operations=(TypedLocalOperation(LocalOperationType.DECLARED_VALIDATION_PLAN,
                                                            {"validation_plan_cid": plan_cid}),)))
            _need(authorization.status is ExecutionStatus.SUCCEEDED,
                  "metadata_execution_typed_plan_not_authorized")
            result["typed_authorization"] = _durable_json(
                receipt_path.with_suffix(".authorization.json"), authorization.to_dict())
            for argv in declared:
                live, fresh, live_packet, live_declared = _current(
                    source, authority, task_cid, revision=claimed_task.revision)
                _need(live.status == "in_progress" and fresh == accepted
                      and live_packet == packet and live_declared == declared
                      and _plain(live.body).get("completion_receipt") == receipt,
                      "metadata_execution_scope_changed")
                if lease is not None:
                    daemon.coordinator.protect_task_claim(claim)
                remaining = deadline - time.monotonic()
                _need(remaining > 0, "metadata_execution_deadline_expired")
                commands = tuple(replace(c, cacheable=False, timeout_seconds=remaining)
                                 for c in build_validation_commands([shlex.join(argv)]))
                _need(len(commands) == 1, "metadata_execution_command_not_atomic")
                scheduler = ValidationScheduler(max_workers=1, resource_budget=1,
                    default_timeout_seconds=remaining,
                    resource_lease_budget=prepared.budget,
                    prepared_admission=prepared, prepared_task_cid=task_cid,
                    prepared_contract_digest=accepted["contract_digest"])
                in_flight = True
                disposition = PreparedValidationDisposition.UNKNOWN
                from .implementation_daemon import PortalImplementationDaemon
                observed = scheduler.run(commands, workspace_path=Path(packet["cwd"]),
                    require_full_validation=True, scope="native_metadata",
                    runner=PortalImplementationDaemon._validation_command_runner,
                    **scheduler_context)
                # Even a returned/rejected scheduler result can carry unknown child
                # custody. Only this original runner's verified subreaper receipt
                # permits a release. Keep in_flight=True on every other path.
                observed_ref = _durable_json(receipt_path.with_suffix(
                    ".command-" + str(len(result["commands"])) + ".json"), observed)
                stopped = _stopped_observation(observed, commands[0].command)
                prepared.acknowledge_stopped(command=commands[0].command)
                in_flight = False
                disposition = PreparedValidationDisposition.STOPPED
                summary = {"schema": "native-metadata-command-observation/v1",
                    "ordinal": len(result["commands"]), "argv": argv,
                    "passed": observed.get("passed") is True,
                    "returncode": stopped["returncode"], "stopped": stopped,
                    "raw_observation": observed_ref}
                result["commands"].append(summary)
                # The original native codec forbids floats. Raw scheduler timings
                # stay in the fsynced artifact and native evidence binds its exact
                # descriptor plus this deliberately float-free summary.
                source.intent.record_validation_result(task_cid=task_cid,
                    outcome="passed" if summary["passed"] else "failed",
                    evidence_digest=content_identity(summary), argv=argv,
                    body={"native_metadata_execution": {"contract": accepted,
                        "claim": claim.to_dict(), "observation": summary,
                        "completion_authority": False}})
                _need(observed.get("attempted") is True and summary["passed"],
                      "metadata_execution_declared_validation_failed")
            _current(source, authority, task_cid, revision=claimed_task.revision)
            if lease is not None:
                daemon.coordinator.protect_task_claim(claim)
            result["outputs"] = _outputs(packet)
            result["passed"] = True
        except Exception as error:
            result.update(error_type=type(error).__name__, error=str(error)[:2000],
                          outcome_unknown=in_flight or control_uncertain)
            if in_flight or control_uncertain:
                # No stop/reap witness was returned by the original scheduler.
                # Keep native custody and the started evidence; never auto-rearm.
                raise
        # A returned original scheduler result proves the command boundary. Persist
        # the candidate/failure before changing control or releasing its live fence.
        result["state"] = "candidate_awaiting_manual_acceptance" if result["passed"] else "execution_failed"
        result_ref = _durable_json(receipt_path, result)
        source.intent.record_evidence(task_cid=task_cid, evidence_kind=_EXECUTION_EVIDENCE,
            digest=content_identity(result), body={"state": result["state"], "execution": result,
                                                 "durable_result": result_ref})
        if claimed_task is not None:
            source.compare_and_set_status(task_cid, claimed_task.revision, "blocked",
                receipt={"operation": "native_metadata_execution_awaiting_manual_acceptance",
                         "claim": claim.to_dict(), "passed": result["passed"],
                         "durable_result": result_ref, "completion_authority": False},
                expected_control_receipt=receipt)
        if lease is not None:
            daemon.coordinator.release(lease, reason=result["state"],
                expected_fencing_token=claim.fencing_token, expected_fence_epoch=claim.fence_epoch)
            result["claim_released"] = True
        result["durable_result"] = result_ref
        result["release_closeout"] = _durable_json(receipt_path.with_suffix(".closeout.json"), result)
        return result
    finally:
        # UNKNOWN retains real capacity. Returning a scheduler ticket never
        # substitutes for the unchanged original stop/reap witness above.
        prepared.close(disposition=disposition)
