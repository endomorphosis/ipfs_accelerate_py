"""Trusted local owner validation; workers never receive profile signing keys.

The closed native RPC accepts only task/attempt/revision identities. Actual
owner checks produce signed observations; the normal typed claim/fence/evidence
CAS still decides whether a task can complete.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import re
import subprocess
from dataclasses import replace
from typing import Mapping

from . import local_planning_admission as local
from .quack_state_server import QuackStateServer
from ..task_sources.intent_repository import IntentRepository
from ..task_sources.typed_state_owner import TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA


SOURCE_TRANSITION_SCHEMA = "supervisor-local-published-source-transition@1"
_COMPLETION_SERVICE_SEAL = object()


class OwnerLocalCompletionService(dict):
    """A fixed owner callback with opt-in, exact-runtime retirement custody."""

    def __init__(self, seal, *, server, gateway, handler, binding):
        if seal is not _COMPLETION_SERVICE_SEAL:
            raise local.LocalPlanningError("completion retirement needs its native binding")
        super().__init__(schema="supervisor-local-owner-validation-service@1", bound=True,
                         operation="local.task.validation.run", completion_authority=False)
        self._server, self._gateway = server, gateway
        self._handler, self._binding, self._retired = handler, binding, False

    @property
    def retired(self):
        return self._retired

    def close(self, runtime):
        from ..entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
        from .finite_repository_execution import ADVISORY_PROFILE, FrozenFiniteRepositoryExecutionScope
        from .finite_proof_query_execution import PROFILE as PROOF_QUERY_PROFILE
        scope = getattr(runtime, "finite_execution_scope", None)
        from .codebase_inventory_execution import PROFILE as INVENTORY_PROFILE, FrozenInventoryExecutionScope
        inventory_scope = getattr(runtime, "inventory_execution_scope", None)
        qualified_scope = (
            type(scope) is FrozenFiniteRepositoryExecutionScope
            and inventory_scope is None and scope.to_dict()["payload"]["profile"] in {ADVISORY_PROFILE, PROOF_QUERY_PROFILE}
        ) or (
            type(inventory_scope) is FrozenInventoryExecutionScope
            and scope is None and inventory_scope.to_dict()["payload"]["profile"] == INVENTORY_PROFILE
        )
        if (type(runtime) is not AdmittedBenchmarkRuntime
                or getattr(runtime, "completion_service", None) is not self
                or runtime.server is not self._server
                or not qualified_scope
                or self._server._command_gateway is not self._gateway
                or not runtime._context_refresh_stopped()):
            raise local.LocalPlanningError("completion retirement requires exact native STOP and isolated UID cleanup")
        if inventory_scope is not None:
            inventory_scope.require_close(runtime)
        if not self._server._lock.acquire(blocking=False):
            raise local.LocalPlanningError("completion retirement cannot interrupt owner callback custody")
        try:
            if not runtime._context_refresh_stopped():
                raise local.LocalPlanningError("completion retirement lost native STOP cleanup")
            if not self._retired:
                if not self._gateway.unbind_local_task_validation_handler(self._handler, self._binding):
                    raise local.LocalPlanningError("completion retirement cannot detach a foreign handler")
                self._retired = True
        finally:
            self._server._lock.release()
        return {**self, "retired": True, "detached_exact_handler": True}


class _OwnerCompletionBindingCustody:
    """Retain one failed service construction's exact installed callback."""

    def __init__(self, seal, *, server, gateway, handler, binding):
        if seal is not _COMPLETION_SERVICE_SEAL:
            raise local.LocalPlanningError("native completion binding custody required")
        self._seal, self._server, self._gateway = seal, server, gateway
        self._handler, self._binding = handler, binding
        self._service = None

    def close(self, runtime):
        # Reuse the ordinary native STOP/UID-cleanup retirement contract. A
        # failed constructor never grants a replacement callback or token.
        if (self._seal is not _COMPLETION_SERVICE_SEAL
                or runtime.server is not self._server
                or self._server._command_gateway is not self._gateway
                or not runtime._context_refresh_stopped()):
            raise local.LocalPlanningError("pending completion binding requires exact native STOP cleanup")
        current = getattr(runtime, "completion_service", None)
        if self._service is None:
            if current is not None:
                raise local.LocalPlanningError("pending completion binding cannot replace a foreign service")
            self._service = OwnerLocalCompletionService(_COMPLETION_SERVICE_SEAL,
                server=self._server, gateway=self._gateway, handler=self._handler, binding=self._binding)
            runtime.completion_service = self._service
        elif current is not self._service:
            raise local.LocalPlanningError("pending completion binding lost its original service")
        return self._service.close(runtime)


class _CompletionBindingCleanupError(local.LocalPlanningError):
    """An exact handler remains installed after constructor rollback failed."""

    def __init__(self, seal, custody):
        if seal is not _COMPLETION_SERVICE_SEAL or type(custody) is not _OwnerCompletionBindingCustody:
            raise local.LocalPlanningError("native completion rollback custody required")
        super().__init__("completion service construction failed; exact callback binding custody retained")
        self._custody = custody


def verify_owner_local_benchmark_observation(*, server: QuackStateServer, admission: Mapping) -> dict:
    """Verify baseline or owner-observed publication for observation and stop.

    This grants no start, dispatch or completion authority. A changed HEAD
    must have an actual signed validation observation in the current native
    owner's database, bound to this admission and the retained task claim.
    """
    if type(server) is not QuackStateServer or server.identity is None:
        raise local.LocalPlanningError("live native owner required for local observation")
    local._require_admission_fields(admission)
    try:
        return local.verify_local_benchmark_admission(admission, initial=False)
    except local.LocalPlanningError:
        pass
    with server._lock:
        intent = IntentRepository(bound_connection=server._connection, install_schema=False)
        graph = local.PromptGoalGraph.from_dict(admission["graph"])
        for task_spec in graph.tasks:
            task = intent.get_task(task_spec.task_cid)
            if task is None or task["status"] not in {"in_progress", "completed"}:
                continue
            rows = server._connection.execute(
                "SELECT outcome, evidence_digest, body_json FROM validation_results WHERE task_cid = ?",
                [task_spec.task_cid],
            ).fetchall()
            for row in rows:
                outcome, digest, body_json = row[0], row[1], row[2]
                try:
                    envelope = json.loads(body_json).get("local_observed_validation")
                    if not envelope or local.content_identity(envelope) != digest:
                        continue
                    transition = envelope["payload"]["source_transition"]
                    contract, manifest, profile, current = local._contract(
                        task["body"], task["task_cid"], source_transition=transition,
                    )
                    observed = local._verify_signature(envelope, profile)
                    receipt = local._verify_signature(admission["receipt"], profile)
                    expected = local._planning_payload(
                        graph, admission["manifest"], manifest, profile, manifest["sources"],
                        admission.get("requirement_bindings"),
                    )
                    claim = task["body"].get("completion_receipt", {})
                    revision = transition["payload"]["task_revision"]
                    if (
                        contract["manifest"] != admission["manifest"] or receipt != expected
                        or contract["graph_cid"] != graph.content_id
                        or contract["planning_receipt_cid"] != local.content_identity(admission["receipt"])
                        or observed.get("schema") != local.RESULT_SCHEMA
                        or observed.get("task_cid") != task["task_cid"]
                        or observed.get("task_revision") != revision
                        or task["revision"] != revision + (task["status"] == "completed")
                        or observed.get("attempt_id") != transition["payload"]["attempt_id"]
                        or observed.get("attempt_id") != claim.get("attempt_id")
                        or observed.get("intent_owner_id") != contract["intent_owner_id"]
                        or observed.get("contract_cid") != local.content_identity(task["body"][local.CONTRACT_KEY])
                        or observed.get("manifest_cid") != contract["manifest_cid"]
                        or observed.get("pending_cid") != contract["pending_cid"]
                        or observed.get("source_tree_id") != local._tree(current)
                        or observed.get("validation") not in contract["task_spec"]["validations"]
                        or observed.get("outcome") != outcome
                        or outcome not in {"passed", "failed"}
                    ):
                        continue
                    return {"manifest": manifest, "profile": profile, "receipt": receipt,
                            "graph": graph, "current_source_tree_id": local._tree(current)}
                except (ValueError, KeyError, TypeError, OSError):
                    continue
    raise local.LocalPlanningError("current source has no exact native owner observation")


def _git(root: Path, *args: str) -> bytes:
    process = subprocess.run(
        ["/usr/bin/git", "--no-replace-objects", "-c", "core.hooksPath=/dev/null",
         "-c", "core.fsmonitor=false", "-C", str(root), *args],
        env={"PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1",
             "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_NO_REPLACE_OBJECTS": "1",
             "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"},
        capture_output=True, timeout=10, check=False,
    )
    if process.returncode != 0:
        raise local.LocalPlanningError("exact published Git source verification failed")
    return process.stdout


def verify_local_source_transition(
    envelope: Mapping, *, manifest_envelope: Mapping, profile,
    current_head: str, current_repository_cid: str,
) -> dict:
    """Recheck the signed publication against current Git and immutable inputs."""
    from ..core.multiformats_identity import cid_for_dag_json
    from ..todo_daemon.database_portal_bridge import DATABASE_PORTAL_ACCEPTED_SOURCE_TRANSITION_SCHEMA

    payload = local._verify_signature(envelope, profile)
    fields = {
        "schema", "manifest_cid", "profile_content_id", "repository_cid",
        "repository", "task_cid", "task_revision", "attempt_id", "contract_cid",
        "baseline_commit", "published_commit", "published_tree", "published_repository_cid",
        "target_ref", "changed_paths", "sources", "task_spec", "native_transition",
        "completion_authority",
    }
    manifest = manifest_envelope["payload"]
    if set(payload) != fields or payload.get("schema") != SOURCE_TRANSITION_SCHEMA:
        raise local.LocalPlanningError("exact local source transition contract required")
    if (
        payload["manifest_cid"] != local.content_identity(manifest_envelope)
        or payload["profile_content_id"] != profile.content_id
        or payload["repository_cid"] != manifest["repository_cid"]
        or payload["repository_cid"] != profile.repository_cid
        or payload["repository"] != manifest["repository"]
        or payload["baseline_commit"] != manifest["baseline_commit"]
        or payload["baseline_commit"] != profile.baseline_commit
        or payload["task_spec"] not in manifest["tasks"]
        or payload["completion_authority"] is not False
        or type(payload["task_revision"]) is not int or payload["task_revision"] < 1
        or not all(type(payload[name]) is str and payload[name] for name in (
            "task_cid", "attempt_id", "contract_cid",
        ))
    ):
        raise local.LocalPlanningError("source transition differs from independent local authority")
    native = payload["native_transition"]
    if not isinstance(native, dict):
        raise local.LocalPlanningError("native Portal source transition required")
    normalized = dict(native)
    transition_cid = normalized.pop("transition_cid", "")
    from ..todo_daemon.database_portal_bridge import _canonical_transition_json

    if (
        native.get("schema") != DATABASE_PORTAL_ACCEPTED_SOURCE_TRANSITION_SCHEMA
        or native.get("database_task_cid") != payload["task_cid"]
        or native.get("attempt_id") != payload["attempt_id"]
        or native.get("baseline_ref") != payload["baseline_commit"]
        or native.get("merge_commit") != payload["published_commit"]
        or native.get("merge_tree") != payload["published_tree"]
        or native.get("worker_self_approval") is not False
        or native.get("task_completion_authority") is not False
        or transition_cid != "sha256:" + hashlib.sha256(_canonical_transition_json(normalized)).hexdigest()
    ):
        raise local.LocalPlanningError("native Portal publication identity differs")
    root = Path(manifest["repository"])
    baseline, published = payload["baseline_commit"], payload["published_commit"]
    implementation = native.get("implementation_commit", "")
    if any(re.fullmatch(r"[0-9a-f]{40}", value) is None for value in (baseline, published, implementation)):
        raise local.LocalPlanningError("exact Git commit identities required")
    if _git(root, "rev-list", "--parents", "-n", "1", published).decode().split() != [published, baseline, implementation]:
        raise local.LocalPlanningError("local transition requires the exact baseline/two-parent merge")
    branch = _git(root, "symbolic-ref", "HEAD").decode().strip()
    tree = _git(root, "rev-parse", published + "^{tree}").decode().strip()
    repository_cid = cid_for_dag_json({
        "schema": "ipfs_accelerate_py.agent_supervisor.observed-repository-root@1",
        "root": str(root), "head_tree": tree,
    })
    if (
        current_head != published
        or _git(root, "rev-parse", "HEAD").decode().strip() != published
        or branch != payload["target_ref"]
        or branch != "refs/heads/" + native.get("target_branch", "")
        or _git(root, "rev-parse", branch).decode().strip() != published
        or tree != payload["published_tree"]
        or repository_cid != payload["published_repository_cid"]
        or current_repository_cid != repository_cid
    ):
        raise local.LocalPlanningError("current published ref/tree differs from source transition")
    _git(root, "diff", "--quiet", published, "--")
    paths = sorted(filter(None, _git(root, "diff", "--name-only", "-z", baseline, published, "--").decode().split("\0")))
    permitted = {row["path"] for row in payload["task_spec"]["outputs"]}
    if not paths or paths != payload["changed_paths"] or not set(paths) <= permitted:
        raise local.LocalPlanningError("published changes exceed exact task outputs")
    current = (
        local.observe_local_manifest_sources(root, manifest)
        if local.supports_created_outputs(manifest)
        else local._sources(root, sorted(manifest["sources"]))
    )
    if current != payload["sources"] or any(
        current[name] != original for name, original in manifest["sources"].items()
        if name not in permitted
    ):
        raise local.LocalPlanningError("published source or public check bytes differ")
    return payload


def authorize_owner_portal_source_transition(
    *, server: QuackStateServer, bridge, merge_queue, attempt, provider_result: Mapping,
) -> dict:
    """Owner-sign a publication recreated from the native Portal/Git/queue path.

    This version supports one exact baseline merge. Concurrent target advance
    and queued reconciliation remain outside this closed local policy.
    """
    from ..control.profile_authority import load_local_profile
    from ..merge.merge_queue import MergeQueue
    from ..todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge

    if type(server) is not QuackStateServer or type(bridge) is not DatabasePortalExecutionBridge or type(merge_queue) is not MergeQueue:
        raise local.LocalPlanningError("actual native owner, Portal bridge and merge queue required")
    bridge._require_accepted_provider(attempt, provider_result)
    record = bridge._record_for_attempt(bridge.task_source, attempt)
    paths = bridge._paths(attempt)
    bridge._seal_attempt_directory(paths, attempt_id=attempt.attempt_id, create=False)
    binding = bridge._strict_binding(paths.binding)
    if binding != bridge._binding(attempt, record, bridge._render_projection(attempt, record), schema=binding["schema"]):
        raise local.LocalPlanningError("Portal binding no longer matches the native attempt")
    transition = bridge._accepted_source_transition(
        attempt=attempt, paths=paths, binding=binding, task_alias=record.task_alias,
        task_cid=record.task_cid, merge_request_loader=merge_queue.get,
    )
    if transition is None or transition != provider_result.get("accepted_source_transition"):
        raise local.LocalPlanningError("native Portal publication could not be independently reconstructed")
    with server._lock:
        identity = server.identity
        if identity is None or server.ready().get("ready") is not True:
            raise local.LocalPlanningError("native owner is not ready")
        intent = IntentRepository(bound_connection=server._connection, install_schema=False)
        task = intent.get_task(attempt.task_cid)
        if task is None or task["revision"] != record.revision or task["status"] != "in_progress":
            raise local.LocalPlanningError("native publication claim revision is stale")
        claimed = task["body"].get("completion_receipt", {})
        if claimed.get("operation") != "database_attempt_admitted" or any(
            claimed.get(key) != getattr(attempt, key) for key in (
                "attempt_id", "claim_id", "lease_id", "owner_session_id", "fencing_token", "fence_epoch",
            )
        ):
            raise local.LocalPlanningError("native publication differs from admitted claim")
        contract_envelope = task["body"][local.CONTRACT_KEY]
        manifest_envelope = contract_envelope["payload"]["manifest"]
        manifest = manifest_envelope["payload"]
        profile = load_local_profile(
            repository_cid=manifest["repository_cid"], profile_dir=Path(manifest["profile_dir"]),
            lifecycle_dir=Path(manifest["lifecycle_dir"]),
        )
        local._verify_signature(manifest_envelope, profile)
        contract = local._verify_signature(contract_envelope, profile)
        root, head, repository_cid = local._repository(Path(manifest["repository"]))
        if bridge.repo_root != root:
            raise local.LocalPlanningError("Portal repository differs from independent manifest")
        payload = {
            "schema": SOURCE_TRANSITION_SCHEMA,
            "manifest_cid": local.content_identity(manifest_envelope),
            "profile_content_id": profile.content_id,
            "repository_cid": manifest["repository_cid"], "repository": str(root),
            "task_cid": task["task_cid"], "task_revision": task["revision"],
            "attempt_id": attempt.attempt_id, "contract_cid": local.content_identity(contract_envelope),
            "baseline_commit": manifest["baseline_commit"], "published_commit": head,
            "published_tree": _git(root, "rev-parse", "HEAD^{tree}").decode().strip(),
            "published_repository_cid": repository_cid,
            "target_ref": _git(root, "symbolic-ref", "HEAD").decode().strip(),
            "changed_paths": sorted(filter(None, _git(root, "diff", "--name-only", "-z", manifest["baseline_commit"], head, "--").decode().split("\0"))),
            "sources": (
                local.observe_local_manifest_sources(root, manifest)
                if local.supports_created_outputs(manifest)
                else local._sources(root, sorted(manifest["sources"]))
            ),
            "task_spec": contract["task_spec"], "native_transition": transition,
            "completion_authority": False,
        }
        envelope = local._signed(payload, manifest)
        local._contract(task["body"], task["task_cid"], source_transition=envelope)
        return envelope


def bind_owner_local_completion_service(
    *, server: QuackStateServer, portal_attempt_root: Path, repo_root: Path,
    merge_queue_dir: Path, board_namespace: str, target_branch: str,
    candidate_runner=None, retirable=False,
) -> dict:
    """Bind a closed grant-authenticated service to owner-chosen native paths.

    The request contains only task/attempt/revision identities. The owner
    derives the retained claim and native control projection from its own
    database and independently reads fixed Portal/queue/Git publication paths.
    No worker-selected source, command, signing key or validation result enters
    the service's authority decision.
    """
    from ..merge.checkout_lock import checkout_repository_id
    from ..merge.merge_queue import MergeQueue
    from ..task_sources.database_task_source import DatabaseTaskSource
    from ..task_sources.typed_state_owner import _require_database_claim_process_attestation
    from ..todo_daemon.database_portal_bridge import DatabasePortalExecutionBridge
    from ..todo_daemon.implementation_daemon import DatabaseImplementationDaemon, DatabaseTaskAttempt

    if type(server) is not QuackStateServer or server.identity is None:
        raise local.LocalPlanningError("live native owner required for local completion service")
    if type(retirable) is not bool:
        raise local.LocalPlanningError("completion retirement opt-in must be an exact boolean")
    repository = Path(repo_root).resolve(strict=True)
    attempts = Path(portal_attempt_root).absolute()
    queue_dir = Path(merge_queue_dir).absolute()
    if not board_namespace or not target_branch or not (repository / ".git").exists():
        raise local.LocalPlanningError("exact owner repository/board/branch configuration required")
    if any(path.resolve(strict=False) != path for path in (attempts, queue_dir)):
        raise local.LocalPlanningError("owner completion paths must not contain symlinks")
    if candidate_runner is not None:
        from .candidate_execution import verify_candidate_runner
        verify_candidate_runner(candidate_runner)

    def no_provider(*_args):
        raise local.LocalPlanningError("owner completion service cannot dispatch a provider")

    def handler(task_cid, attempt_id, expected_revision, grant):
        intent = IntentRepository(bound_connection=server._connection, install_schema=False)
        source = DatabaseTaskSource(intent=intent, install_schema=False)
        record = source.get_task(task_cid)
        if record is None or record.status != "in_progress" or record.revision != expected_revision:
            raise local.LocalPlanningError("local completion request task revision is stale")
        receipt = record.body.get("completion_receipt", {})
        if (
            receipt.get("operation") != "database_attempt_admitted"
            or receipt.get("claim_phase_schema") != TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA
            or receipt.get("attempt_id") != attempt_id
        ):
            raise local.LocalPlanningError("local completion request has no exact admitted attempt")
        _require_database_claim_process_attestation(receipt, grant=grant)
        contract = record.body.get(local.CONTRACT_KEY)
        if not isinstance(contract, Mapping) or contract.get("payload", {}).get("manifest", {}).get("payload", {}).get("repository") != str(repository):
            raise local.LocalPlanningError("local completion request is outside owner-bound repository")
        # This is a deterministic projection of the current native admission,
        # not a new attempt row or an assertion of provider execution. Portal
        # publication is independently reverified below.
        attempt = DatabaseTaskAttempt(
            task_cid=record.task_cid, task_alias=record.task_alias,
            **{key: receipt[key] for key in (
                "attempt_id", "claim_id", "attempt_number", "owner_session_id",
                "fencing_token", "fence_epoch", "lease_id",
            )},
            committed_phase=receipt["attempt_execution_phase"], status="running",
            started_at_ms=0, revision=receipt["attempt_execution_revision"], body={},
        )
        attempt = replace(attempt, body={
            "control_binding": DatabaseImplementationDaemon._control_claim_binding(attempt, record),
        })
        bridge = DatabasePortalExecutionBridge(
            task_source=source, attempt_root=attempts, portal_factory=no_provider,
            repo_root=repository, board_namespace=board_namespace, merge_target_branch=target_branch,
            task_header_prefix="## " + record.task_alias,
        )
        paths = bridge._paths(attempt)
        bridge._seal_attempt_directory(paths, attempt_id=attempt_id, create=False)
        binding = bridge._strict_binding(paths.binding)
        if binding != bridge._binding(attempt, record, bridge._render_projection(attempt, record), schema=binding["schema"]):
            raise local.LocalPlanningError("owner-derived claim differs from sealed Portal binding")
        if not (queue_dir / "merge_queue.duckdb").is_file():
            raise local.LocalPlanningError("owner-bound native merge queue is absent")
        queue = MergeQueue(queue_dir, target_repository_id=checkout_repository_id(repository),
                           target_branch=target_branch, require_target_binding=True)
        accepted = bridge._acceptance_receipt(
            attempt=attempt, paths=paths, binding=binding, summaries=[], merge_request_loader=queue.get,
        )
        transition = authorize_owner_portal_source_transition(
            server=server, bridge=bridge, merge_queue=queue, attempt=attempt, provider_result=accepted,
        )
        observed = run_owner_local_task_validations(
            server=server, task_cid=task_cid, attempt_id=attempt_id,
            expected_revision=expected_revision, source_transition=transition,
            candidate_runner=candidate_runner,
        )
        return {**observed, "attempt_id": attempt_id, "task_revision": expected_revision}

    gateway = server._command_gateway
    if gateway is None:
        raise local.LocalPlanningError("native typed owner gateway is unavailable")
    if retirable:
        # Retirement needs explicit custody support. Reject an older gateway
        # before installing a handler that this runtime could not detach.
        try:
            inspect.signature(gateway.bind_local_task_validation_handler).bind(handler, retirable=True)
            inspect.signature(gateway.unbind_local_task_validation_handler).bind(handler, object())
        except (AttributeError, TypeError, ValueError) as error:
            raise local.LocalPlanningError("native typed owner gateway does not support completion retirement") from error
        binding = gateway.bind_local_task_validation_handler(handler, retirable=True)
    else:
        # The ordinary, owner-lifetime service keeps its original one-argument
        # binding contract, including gateways without optional retirement.
        binding = gateway.bind_local_task_validation_handler(handler)
    if retirable:
        try:
            return OwnerLocalCompletionService(_COMPLETION_SERVICE_SEAL, server=server,
                gateway=gateway, handler=handler, binding=binding)
        except BaseException as construction_error:
            custody = _OwnerCompletionBindingCustody(_COMPLETION_SERVICE_SEAL,
                server=server, gateway=gateway, handler=handler, binding=binding)
            try:
                if not gateway.unbind_local_task_validation_handler(handler, binding):
                    raise local.LocalPlanningError("completion rollback cannot detach a foreign handler")
            except BaseException as cleanup_error:
                retained = _CompletionBindingCleanupError(_COMPLETION_SERVICE_SEAL, custody)
                retained.construction_error = construction_error
                raise retained from cleanup_error
            raise
    return {"schema": "supervisor-local-owner-validation-service@1", "bound": True,
            "operation": "local.task.validation.run", "completion_authority": False}


def run_owner_local_task_validations(
    *, server: QuackStateServer, task_cid: str, attempt_id: str,
    expected_revision: int, timeout: float = 60,
    source_transition: Mapping | None = None, candidate_runner=None,
) -> dict:
    """Run real local checks against an exact native admitted attempt.

    The caller must hold the actual owner object. This function supplies no
    provider credential and performs no task-status transition. The exclusive
    owner lock keeps claim custody stable throughout the bounded checks.
    """
    if type(server) is not QuackStateServer or type(expected_revision) is not int:
        raise local.LocalPlanningError("actual native owner and exact revision required")
    if expected_revision < 1 or not isinstance(attempt_id, str) or not attempt_id:
        raise local.LocalPlanningError("exact admitted attempt required")
    with server._lock:
        identity = server.identity
        if identity is None or server.ready().get("ready") is not True:
            raise local.LocalPlanningError("native owner is not ready")
        # Read through the owner's existing connection, never reopen its file
        # and never give a worker a profile key or an arbitrary command runner.
        observer = IntentRepository(
            bound_connection=server._connection, install_schema=False,
            owner_id=identity.server_id, session_id=identity.process_birth_id,
        )
        task = observer.get_task(task_cid)
        if task is None or task["revision"] != expected_revision or task["status"] != "in_progress":
            raise local.LocalPlanningError("local validation task revision/status is stale")
        receipt = task["body"].get("completion_receipt", {})
        if (
            receipt.get("operation") != "database_attempt_admitted"
            or receipt.get("claim_phase_schema") != TYPED_DATABASE_ATTEMPT_ADMISSION_SCHEMA
            or receipt.get("attempt_id") != attempt_id
        ):
            raise local.LocalPlanningError("local validation requires its exact admitted attempt")
        contract, _, _, _ = local._contract(
            task["body"], task_cid, source_transition=source_transition,
        )
        if source_transition is not None and (
            source_transition["payload"].get("task_revision") != expected_revision
            or source_transition["payload"].get("attempt_id") != attempt_id
        ):
            raise local.LocalPlanningError("source transition belongs to another admitted attempt")
        intent = IntentRepository(
            bound_connection=server._connection, install_schema=False,
            owner_id=contract["intent_owner_id"], session_id=identity.process_birth_id,
        )
        return local.run_local_task_validations(
            intent=intent, task_cid=task_cid, attempt_id=attempt_id,
            timeout=timeout, source_transition=source_transition,
            candidate_runner=candidate_runner,
        )
