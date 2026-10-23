"""Resume only an ordinary, already integrated completion after Git contention.

An immutable local observation and the original fenced queue row retain the
candidate. Neither a file's presence nor a pending verdict grants completion.
Every continuation rechecks the actual task, validation and Git bindings.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import stat
import time

from ..merge.git_checkout_transaction import CheckoutGitMutationBlocked, checkout_git_mutation
from ..merge.merge_queue import MergeQueue
from ..proof.formal_verification_contracts import content_identity

SCHEMA = "ipfs_accelerate_py/agent-supervisor/completion-git-lock-continuation@1"
MAX_RECORD_BYTES = 64 * 1024
RETRY_SECONDS = 0.6
MAX_PUBLICATION_ATTEMPTS = 3
_REASONS = {"checkout_git_foreign_lock_present", "checkout_git_mutation_busy"}


def _path(daemon, request):
    name = content_identity({"request_id": str(request.request_id)})
    return daemon.state_path.parent / "completion-git-locks" / (name + ".json")


def present(daemon, request):
    return os.path.lexists(_path(daemon, request))


def queue_supported(daemon):
    # Database/custody adapters own different settlement authority. No
    # duck-typed lookalike may inherit this ordinary queue continuation.
    return type(daemon.merge_queue) is MergeQueue


def pending_observation(daemon, *, include_active=True):
    """Observe bounded current custody, including newly created records."""
    if not queue_supported(daemon):
        return _pending("completion_continuation_queue_authority_changed")
    snapshots = [("completion_continuation_terminal_hold",
        daemon.merge_queue.quarantined_requests(limit=256, include_cancelled=True))]
    if include_active:
        snapshots += [("completion_continuation_cooldown_or_consumer_owned",
            daemon.merge_queue.pending_requests(limit=256, include_deferred=True)),
            ("completion_continuation_cooldown_or_consumer_owned",
            daemon.merge_queue.processing_requests(limit=256))]
    for reason, requests in snapshots:
        held = next((request for request in requests if present(daemon, request)), None)
        if held is not None:
            return _pending(reason, request_id=held.request_id, task_id=held.task_id,
                retry_not_before=held.retry_not_before)
        if len(requests) >= 256:
            return _pending("completion_continuation_inventory_bound")
    return None


def exact_deferral(result):
    return bool(isinstance(result, Mapping)
        and result.get("reason") in _REASONS
        and result.get("operation") == "_mark_tasks_completed_in_todo"
        and result.get("infrastructure_failure") is True
        and result.get("blocked") is True
        and all(result.get(key) is False for key in
            ("attempted", "updated", "committed", "durable", "merged", "provider_dispatched")))


def _private_directory(path):
    # Resolve no symlinks, including ancestors, before publishing a receipt.
    if path.parent.absolute() != path.parent.resolve(strict=True):
        raise ValueError("completion continuation parent is not canonical")
    path.mkdir(mode=0o700, exist_ok=True)
    observed = path.lstat()
    if (not stat.S_ISDIR(observed.st_mode) or observed.st_uid != os.getuid()
            or stat.S_IMODE(observed.st_mode) != 0o700):
        raise ValueError("completion continuation directory is not private")


def _read(path):
    _private_directory(path.parent)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid()
                or before.st_nlink != 1 or stat.S_IMODE(before.st_mode) != 0o600
                or before.st_size > MAX_RECORD_BYTES):
            raise ValueError("completion continuation file is not private and bounded")
        raw = stream.read(MAX_RECORD_BYTES + 1)
        after, named = os.fstat(stream.fileno()), path.lstat()
        identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                   value.st_mtime_ns, value.st_ctime_ns)
        if len(raw) > MAX_RECORD_BYTES or identity(before) != identity(after) or identity(after) != identity(named):
            raise ValueError("completion continuation changed while reading")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate completion continuation key")
            result[key] = value
        return result
    def constant(_value):
        raise ValueError("non-finite completion continuation value")
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError("completion continuation is not an object")
    body = dict(value)
    record_id = body.pop("record_id", None)
    if body.get("schema") != SCHEMA or record_id != content_identity(body):
        raise ValueError("completion continuation identity differs")
    return value


def _write(path, body):
    _private_directory(path.parent)
    value = {**body, "record_id": content_identity(body)}
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if len(raw) > MAX_RECORD_BYTES:
        raise ValueError("completion continuation exceeds its bound")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    except FileExistsError:
        if _read(path) != value:
            raise ValueError("completion continuation binding was replaced")
    else:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return value


def _ordinary(daemon, metadata):
    return bool(queue_supported(daemon)
        and metadata.get("schema") == "ipfs_accelerate_py/agent-supervisor/merge-candidate@3"
        and daemon.task_source is None and daemon.production_provider_policy is None
        and daemon.production_owner_handoff_factory is None and getattr(daemon, "_database_attempt_authority", None) is None
        and not daemon.manual_completion_authority_task_ids
        and not daemon.manual_completion_authority_required_task_ids
        and not metadata.get("production_provider_policy_id")
        and not metadata.get("manual_completion_authority_task_ids")
        and not metadata.get("manual_completion_authority_required_task_ids")
        and not metadata.get("bundle_work_order") and not metadata.get("quarantine")
        and not metadata.get("revivals") and not metadata.get("false_completion_revivals"))


def _binding(daemon, request):
    metadata = request.metadata
    if not isinstance(metadata, dict) or not _ordinary(daemon, metadata):
        raise ValueError("completion continuation is outside ordinary schema-v3 scope")
    task = daemon._portal_task_from_merge_request(request)
    identity = daemon._identity_for_task(task)
    proof = metadata.get("validation_proof")
    if (not isinstance(proof, dict) or proof.get("attempted") is not True
            or proof.get("passed") is not True or proof.get("returncode") != 0
            or not isinstance(proof.get("selection"), Mapping)
            or proof["selection"].get("scope") != "pre_merge"
            or daemon._scope_adjudication_merge_binding_error(proof)):
        raise ValueError("completion continuation lacks validated candidate evidence")
    if (request.canonical_task_id != identity.canonical_task_cid
            or request.canonical_task_key != identity.canonical_task_key
            or request.target_repository_id != daemon.merge_target_repository_id
            or request.target_branch != daemon.resolved_merge_target_branch
            or metadata.get("completion_task_cids") != {task.task_id: identity.canonical_task_cid}
            or Path(str(metadata.get("todo_path"))).resolve() != daemon.todo_path.resolve()
            or Path(str(metadata.get("state_path"))).resolve() != daemon.state_path.resolve()):
        raise ValueError("completion continuation queue/task/source identity differs")
    # Queue bookkeeping (claim, cooldown, deferrals) is mutable; the sealed
    # candidate metadata, exact source and validation contract are not.
    candidate_metadata = {key: value for key, value in metadata.items()
        if key not in {"deferrals", "completion", "quarantine", "revivals"}}
    return task, {
        "schema": SCHEMA, "request_id": request.request_id,
        "repo_root": str(daemon.repo_root.resolve()),
        "target_repository_id": daemon.merge_target_repository_id,
        "target_branch": daemon.resolved_merge_target_branch,
        "todo_path": str(daemon.todo_path.resolve()), "state_path": str(daemon.state_path.resolve()),
        "canonical_task_cid": identity.canonical_task_cid,
        "canonical_task_key": identity.canonical_task_key,
        "candidate_commit": request.commit_sha,
        "candidate_tree": daemon._candidate_repository_tree(request.commit_sha),
        "candidate_metadata_id": content_identity(candidate_metadata),
        "validation_proof_id": content_identity(proof),
        "completion_task_cids": {task.task_id: identity.canonical_task_cid},
    }


def _pending(reason, record=None, **details):
    return {"attempted": False, "merged": False, "already_merged": False,
        "returncode": 2, "completion_only_pending": True,
        "integration_occurred": bool(record), "integrated": bool(record),
        "accepted": False, "acceptance_pending": True,
        "completion_authoritative": False, "provider_dispatched": False,
        "reason": reason, "completion_continuation_id": (record or {}).get("record_id", ""),
        "merge_commit": (record or {}).get("integration_commit", ""), **details}


def begin(daemon, request, integration, todo_result, *, completion_intent):
    """Return None for every route that retains its original failure policy."""
    if not exact_deferral(todo_result) or not _ordinary(daemon, request.metadata):
        return None
    task, body = _binding(daemon, request)
    proof = integration.get("integration_commit_proof")
    if not isinstance(proof, dict) or proof.get("passed") is not True:
        return None
    commit = str(proof.get("integration_commit") or "")
    if not commit or integration.get("post_merge_declared_output_invariant", {}).get("passed") is not True:
        return None
    record = _write(_path(daemon, request), {**body,
        "integration_commit": commit,
        "integration_tree": daemon._candidate_repository_tree(commit),
        "integration_commit_proof": proof,
        "completion_intent": dict(completion_intent),
    })
    daemon._record_event("merge_completion_git_lock_deferred", {
        "task_id": task.task_id, "request_id": request.request_id,
        "completion_continuation_id": record["record_id"], "merge_commit": commit,
        "reason": todo_result["reason"], "accepted": False, "provider_dispatched": False})
    return resume(daemon, request)


def _authority(daemon, request, record):
    from .implementation_daemon import PortalTaskState
    if _read(_path(daemon, request)) != record:
        raise ValueError("completion continuation observation was replaced")
    stored = daemon.merge_queue.get(request.request_id)
    if (stored is None or stored.status != "processing"
            or not daemon.merge_queue.owns_claim(request)):
        raise ValueError("completion continuation lost its exact queue claim")
    task, expected = _binding(daemon, stored)
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("completion continuation candidate binding differs")
    protected = daemon._candidate_protected_path_check(
        baseline_ref=str(stored.metadata.get("baseline_ref") or ""),
        implementation_commit=stored.commit_sha,
        protected_paths=stored.metadata.get("implementation_protected_paths"))
    if (daemon._implementation_cancel_requested()
            or not daemon._ordinary_task_unheld(task, PortalTaskState.load(daemon.state_path), completion_queue_owned=True)
            or os.path.lexists(daemon._implementation_protected_incident_path())
            or os.path.lexists(daemon._implementation_protected_active_snapshot_path())
            or protected.get("checked") is not True or protected.get("protected_paths_changed")):
        return None, None, "completion_continuation_held"
    target = daemon._run_git(["rev-parse", "--verify",
        f"refs/heads/{daemon.resolved_merge_target_branch}^{{commit}}"], cwd=daemon.repo_root)
    if target.returncode or target.stdout.strip() != record["integration_commit"]:
        return None, None, "completion_continuation_target_changed"
    proof = daemon._immutable_integration_commit({"merge_commit": record["integration_commit"]},
        implementation_commit=stored.commit_sha, target_branch=daemon.resolved_merge_target_branch)
    if (proof.get("passed") is not True
            or daemon._candidate_repository_tree(record["integration_commit"]) != record["integration_tree"]
            or daemon._completion_task_revision_binding_error(stored.metadata, require_pending=True)
            or daemon._declared_output_tracking_invariant([task], repository_ref=record["integration_commit"]).get("passed") is not True):
        return None, None, "completion_continuation_evidence_changed"
    current = [item for item in daemon._load_tasks() if item.task_id == task.task_id]
    if len(current) != 1:
        raise ValueError("completion continuation current task is ambiguous")
    intent = record.get("completion_intent")
    if not isinstance(intent, dict):
        raise ValueError("completion continuation publication intent missing")
    intent_body = dict(intent)
    intent_id = intent_body.pop("intent_id", None)
    payload = intent.get("completion_payload", {})
    if (intent_id != content_identity(intent_body)
            or payload.get("task_id") != task.task_id
            or payload.get("task_cid") != record["canonical_task_cid"]
            or payload.get("merged_tree_id") != record["integration_commit"]
            or intent.get("queue_task_cid") != record["canonical_task_cid"]):
        raise ValueError("completion continuation publication binding differs")
    daemon._validated_completion_publication_sink(intent)
    return current[0], proof, ""


def resume(daemon, request):
    """Run only the normal completion publisher under fresh queue/Git fences."""
    record = None
    reason = "completion_continuation_invalid"
    try:
        record = _read(_path(daemon, request))
        deadline = time.monotonic() + RETRY_SECONDS
        for ordinal in range(MAX_PUBLICATION_ATTEMPTS):
            task, proof, reason = _authority(daemon, request, record)
            if reason:
                return _pending(reason, record)
            reason = "merge_completion_git_lock_deferred"
            daemon._record_event("merge_completion_git_lock_retry", {
                "task_id": task.task_id, "request_id": request.request_id,
                "completion_continuation_id": record["record_id"], "publication_attempt": ordinal + 1,
                "accepted": False, "provider_dispatched": False})
            try:
                with checkout_git_mutation(daemon.repo_root, timeout_seconds=min(.15, max(0, deadline-time.monotonic()))):
                    # A hold or queue/target change while waiting for the Git
                    # guard cannot be admitted using the earlier observation.
                    task, proof, reason = _authority(daemon, request, record)
                    if reason:
                        return _pending(reason, record)
                    reason = "completion_continuation_publication_unproven"
                    # This remains the authoritative publisher and its own
                    # runtime/board/source/checkout authority gates all run.
                    result = daemon._mark_tasks_completed_in_todo([task.task_id],
                        primary_task_id=task.task_id, completion_reason="merge_completion_git_lock_continuation",
                        expected_task_cids=record["completion_task_cids"],
                        expected_task_statuses={task.task_id: task.status},
                        expected_target_commit=record["integration_commit"],
                        completion_intent=(record["completion_intent"]
                            if daemon._todo_board_is_implementation_protected() else None))
            except CheckoutGitMutationBlocked as exc:
                if exc.reason not in _REASONS:
                    return _pending("completion_continuation_checkout_blocked", record)
                result = None
            if result is not None and not exact_deferral(result):
                persisted = daemon._reconciled_completion_persisted(result, record["completion_task_cids"])
                if (persisted.get("passed") is True
                        and not daemon._completed_task_binding_error(request.metadata)):
                    published = result.get("completion_publication", {}).get("published") is True
                    if not published:
                        daemon._decision_runtime_completion(task, merged_tree_id=record["integration_commit"],
                            evidence=record["completion_intent"]["completion_payload"]["post_merge_evidence"])
                        daemon._record_task_queue_outcome(task, 0)
                    daemon._record_event("merge_completion_git_lock_published", {
                        "task_id": task.task_id, "request_id": request.request_id,
                        "completion_continuation_id": record["record_id"],
                        "merge_commit": record["integration_commit"], "provider_dispatched": False})
                    return {"attempted": False, "merged": True, "already_merged": True, "returncode": 0,
                        "reason": "merge_completion_git_lock_published", "integration_occurred": True,
                        "merge_commit": record["integration_commit"], "target_commit": record["integration_commit"],
                        "integration_commit_proof": proof, "todo_update_result": result,
                        "completion_continuation_id": record["record_id"], "provider_dispatched": False}
                return _pending("completion_continuation_publication_unproven", record)
            if ordinal + 1 >= MAX_PUBLICATION_ATTEMPTS or time.monotonic() >= deadline:
                break
            time.sleep(min(.05 * (ordinal + 1), max(0, deadline-time.monotonic())))
        return _pending(reason, record)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        daemon._record_event("merge_completion_git_lock_blocked", {
            "request_id": str(request.request_id), "reason": reason,
            "error_type": type(exc).__name__, "accepted": False, "provider_dispatched": False})
        return _pending(reason, record)
