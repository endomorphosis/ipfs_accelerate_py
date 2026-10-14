"""Current native intent and public-check evidence for source task packets.

This bridge reads native task data through the existing signed observation
reader and world snapshot owner. Native readiness may maintain the owner's
private attach-token handoff file. It does not manufacture accepted repository/world roots, run
task validation commands, or persist another catalog. Source and intent namespaces remain
distinct. Freshness is cooperative and must be checked at each dispatch.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import threading
import time
import weakref

from ..context.context_contracts import ContextReference, ContextTier
from ..proof.formal_verification_contracts import content_identity
from ..prompt.prompt_workflow import PromptGoalGraph
from ..semantic_state.intent_world_snapshot import capture_intent_world_snapshot
from ..task_sources.intent_repository import IntentRepository
from .intent_requirement_observation import (
    observe_local_intent_requirements, observe_owner_intent_requirements,
)
from .repository_source_session import RepositorySourceSession
from .supervisor_meta_index import public_replay_without_metadata


SCHEMA = "repository-intent-world-observation@1"
RECEIPT_SCHEMA = "repository-intent-world-observation-receipt@1"
KIND = "repository-intent-world"
MAX_BYTES = 65536
_ISSUED = weakref.WeakKeyDictionary()
_ISSUANCE_LOCK = threading.RLock()
_AUTHORITY = dict(proof_authority=False, execution_authority=False,
                  completion_authority=False, canonical_state_mutated=False,
                  semantic_alignment_verified=False, source_semantics_verified=False)


class RepositoryIntentWorldError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _require(value, code):
    if not value:
        raise RepositoryIntentWorldError(code)


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _json(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _owner_identity(intent):
    _require(type(intent) is IntentRepository and not intent.uses_bound_connection
             and not intent._quack_transport, "independent_intent_owner_required")
    path = intent.database_path
    _require(isinstance(path, Path) and path.is_absolute() and not path.is_symlink()
             and path.resolve(strict=True) == path and path.is_file(),
             "intent_owner_path_changed")
    stat = path.stat()
    return (str(path), stat.st_dev, stat.st_ino, intent.owner_id, intent.session_id,
            str(intent._open_target))


def _require_native_server(server):
    from .quack_state_server import InProcessQuackTransport, QuackStateServer

    _require(type(server) is QuackStateServer, "native_intent_owner_required")
    _require(type(server.transport) is InProcessQuackTransport,
             "native_intent_owner_transport_required")


@contextmanager
def _native_owner_lock(server, deadline):
    _require_native_server(server)
    remaining = deadline - time.monotonic()
    _require(remaining > 0 and server._lock.acquire(timeout=max(0.0, remaining)),
             "intent_world_capture_timeout")
    try:
        yield
    finally:
        server._lock.release()


def _native_owner_identity(server, deadline):
    """Check the live owner without exporting endpoints, paths or credentials."""
    from .quack_state_server import QuackStateServerReadyError

    with _native_owner_lock(server, deadline):
        try:
            ready = server.ready()
        except QuackStateServerReadyError as exc:
            raise RepositoryIntentWorldError("native_intent_owner_not_ready") from exc
        identity = server.identity
        _require(ready.get("ready") is True and identity is not None
                 and server._connection is not None, "native_intent_owner_not_ready")
        binding = {key: getattr(identity, key) for key in (
            "server_id", "store_id", "database_uuid", "generation", "process_birth_id",
            "repository_id", "schema_revision", "schema_fingerprint",
        )}
        binding["ready_checked"] = True
        # Retain actual objects as well as their public identities. An in-place
        # rebind, restart or replaced transport cannot revive an issued packet.
        custody = (server._connection, server.transport, _json(binding),
                   identity.fence_epoch, identity.startup_epoch)
        _require(time.monotonic() < deadline, "intent_world_capture_timeout")
        return custody, binding


def _native_world_capture(*, server, owner, repository_id, task_id, deadline):
    """Borrow the exclusive handle under a transaction owned by this helper."""
    with _native_owner_lock(server, deadline):
        current, _ = _native_owner_identity(server, deadline)
        _require(current == owner, "intent_world_owner_or_admission_changed")
        connection = server._connection
        identity = server.identity
        reader = IntentRepository(bound_connection=connection, install_schema=False,
            owner_id=identity.server_id, session_id=identity.process_birth_id)
        # BEGIN stays outside the rollback handler: failure to start must never
        # roll back a transaction already owned by another caller.
        connection.execute("BEGIN TRANSACTION")
        try:
            world = capture_intent_world_snapshot(reader, repository_id=repository_id,
                task_cids=(task_id,), transaction_owned_by_caller=True)
            connection.execute("COMMIT")
        except BaseException:
            try:
                connection.execute("ROLLBACK")
            except Exception:
                pass
            raise
        _require(_native_owner_identity(server, deadline)[0] == owner,
                 "intent_world_owner_or_admission_changed")
        return world


def _native_owner_current(*, server, owner, watermark, deadline):
    with _native_owner_lock(server, deadline):
        _require(_native_owner_identity(server, deadline)[0] == owner,
                 "intent_world_owner_or_admission_changed")
        reader = IntentRepository(bound_connection=server._connection, install_schema=False)
        _require(reader.event_watermark() == watermark, "intent_world_capture_changed")
        _require(time.monotonic() < deadline, "intent_world_capture_timeout")


def _source_binding(session, generation):
    _require(type(session) is RepositorySourceSession, "native_source_session_required")
    _require(type(generation) is int and generation >= 1, "invalid_source_generation")
    status = session.status_summary()
    _require(status["source_current"] is True and status["generation"] == generation,
             "intent_world_source_not_current")
    return {key: status[key] for key in (
        "session_id", "generation", "package_manifest_sha256", "source_tree_sha256",
        "datasets_head", "semantic_state_cid",
    )}


def _admission_serialized(admission):
    _require(isinstance(admission, Mapping), "intent_admission_required")
    encoded = _json(admission)
    _require(len(encoded.encode("utf-8")) <= 8_000_000, "intent_admission_byte_limit")
    return encoded


def _admission_identity(admission):
    return _sha(_admission_serialized(admission))


def _selected(values, task_id):
    selected = [item for item in values if item.get("task_cid") == task_id]
    _require(len(selected) == 1, "exact_native_task_required")
    return selected[0]


def _resources(session, deadline):
    from ipfs_datasets_py.logic.software_contracts.codebase_resources import acquire_codebase_resources

    _require(time.monotonic() < deadline, "intent_world_capture_timeout")
    return acquire_codebase_resources(
        scheduler=getattr(session._index, "_scheduler", None),
        timeout_seconds=min(10.0, max(0.001, deadline - time.monotonic())), memory_mb=256,
    )


def _capture(*, session, intent, admission, task_id, generation, max_bytes,
             server=None, expected_owner=None):
    _require(type(task_id) is str and task_id and task_id.strip() == task_id
             and len(task_id.encode("utf-8")) <= 512, "invalid_intent_task_id")
    _require(type(max_bytes) is int and 1024 <= max_bytes <= MAX_BYTES,
             "invalid_intent_world_limit")
    _require(type(session) is RepositorySourceSession, "native_source_session_required")
    if server is None:
        owner = _owner_identity(intent)
    else:
        _require(intent is None, "ambiguous_intent_owner")
        _require_native_server(server)
        owner = None
    native_owner_binding = None
    # Every native read and the eventual public projection consume the same
    # captured bytes. A caller's mutable mapping never becomes a second input
    # after its signature and digest checks have completed.
    original_admission = admission
    admission_json = _admission_serialized(original_admission)
    admission_sha = _sha(admission_json)
    admission = json.loads(admission_json)
    try:
        declared_root = Path(admission["manifest"]["payload"]["repository"])
    except (KeyError, TypeError) as exc:
        raise RepositoryIntentWorldError("intent_workspace_binding_missing") from exc
    _require(declared_root.is_absolute() and declared_root.resolve(strict=True) == session.workspace,
             "intent_workspace_mismatch")
    before = _source_binding(session, generation)
    deadline = time.monotonic() + 30.0
    with public_replay_without_metadata():
        # Admission denial/cancellation must precede all native intent work.
        # Release this gate before the strict reader's own applicability work
        # can request the same scheduler; never recursively reserve its slot.
        with _resources(session, deadline) as lease:
            _require(not lease.cancelled, "intent_world_capture_cancelled")
        if server is None:
            def observe():
                return observe_local_intent_requirements(admission=admission, intent=intent)
        else:
            # No readiness query or native owner read precedes the admission
            # gate. The strict reader owns its transaction and may request
            # applicability resources; never wrap it in a root lease.
            owner, native_owner_binding = _native_owner_identity(server, deadline)
            _require(expected_owner is None or owner == expected_owner,
                     "intent_world_owner_or_admission_changed")
            _require(native_owner_binding["repository_id"] == admission["manifest"]["payload"].get("repository_cid"),
                     "intent_world_owner_mismatch")
            def observe():
                # Bound contention for the same RLock the reader already holds
                # throughout observation. Its reentrant acquisition adds no
                # new lock lifetime, and no resource lease is held here.
                with _native_owner_lock(server, deadline):
                    return observe_owner_intent_requirements(server=server, admission=admission)
        observed = observe()
        observed_task = _selected(observed["tasks"], task_id)
        if server is None:
            _require(observed["intent_owner_id"] == intent.owner_id,
                     "intent_world_owner_mismatch")
        else:
            binding = observed.get("native_owner_binding")
            _require(isinstance(binding, Mapping) and binding == {
                key: native_owner_binding[key] for key in (
                    "server_id", "database_uuid", "generation", "process_birth_id",
                    "repository_id", "ready_checked",
                )}, "intent_world_owner_mismatch")
        _require(time.monotonic() < deadline, "intent_world_capture_timeout")
        # The strict reader may invoke its own native applicability admission;
        # do not hold another root lease around that reader. The pure MVCC
        # world capture gets a separate bounded shared-resource reservation.
        with _resources(session, deadline) as lease:
            _require(not lease.cancelled, "intent_world_capture_cancelled")
            if server is None:
                world = capture_intent_world_snapshot(intent,
                    repository_id=observed["repository_id"], task_cids=(task_id,))
            else:
                # Acquire resources before waiting for the exclusive owner.
                world = _native_world_capture(server=server, owner=owner,
                    repository_id=observed["repository_id"], task_id=task_id, deadline=deadline)
            _require(not lease.cancelled, "intent_world_capture_cancelled")
        selected = _selected(world["plan_projection"]["tasks"], task_id)
        _require(world["planning_context"]["event_watermark"] == observed["native_event_watermark"]
                 and selected["revision"] == observed_task["revision"],
                 "intent_world_capture_changed")
        # The separate native transactions must agree in both directions.
        repeated = observe()
        _require(repeated == observed, "intent_world_capture_changed")
        if server is None:
            _require(intent.event_watermark() == observed["native_event_watermark"],
                     "intent_world_capture_changed")
        else:
            _native_owner_current(server=server, owner=owner,
                watermark=observed["native_event_watermark"], deadline=deadline)
        _require(time.monotonic() < deadline, "intent_world_capture_timeout")
    after = _source_binding(session, generation)
    _require(before == after, "intent_world_source_changed")
    def owner_current():
        if server is None:
            return _owner_identity(intent) == owner
        _native_owner_current(server=server, owner=owner,
            watermark=observed["native_event_watermark"], deadline=deadline)
        return True
    _require(owner_current() and _admission_identity(original_admission) == admission_sha,
             "intent_world_owner_or_admission_changed")

    # Signed contracts and arbitrary task/plan bodies stay behind their owner.
    # The reviewed graph supplies the complete worker-facing task declaration.
    graph = PromptGoalGraph.from_dict(admission["graph"])
    graph_tasks = [task for task in graph.tasks if task.task_cid == task_id]
    _require(len(graph_tasks) == 1, "exact_native_task_required")
    graph_task = graph_tasks[0].to_dict()
    task_spec = {key: value for key, value in graph_task.items()
                 if key not in {"provenance", "created_at_ms", "updated_at_ms"}}
    task_spec["provenance_cid"] = content_identity(graph_task.get("provenance", {}))
    task_spec.update(task_cid=task_id, native_revision=selected["revision"], native_status=selected["status"],
        objective_id=selected["objective_id"], native_spec_cid=selected["spec_cid"],
        native_body_cid=content_identity(selected["body"]))
    requirements = [row for row in observed["requirements"] if task_id in row["task_cids"]]
    intent_binding = {key: observed[key] for key in (
        "repository_id", "current_source_tree_id", "native_event_watermark", "intent_owner_id",
        "intent_revision_cid", "plan_id", "graph_cid", "policy_id", "contract_cid",
        "manifest_cid", "planning_receipt_cid",
    )}
    material = dict(schema=SCHEMA, task_id=task_id, source_binding=before,
        intent_binding=intent_binding, task_spec=task_spec, observed_task=observed_task,
        requirements=requirements, requirement_observation_cid=observed["observation_cid"],
        intent_world=dict(snapshot=world["snapshot"], capture_cid=world["capture_cid"],
            plan_projection_cid=world["plan_projection"]["projection_cid"],
            component_status=world["planning_context"]["component_status"],
            unavailable_components=world["planning_context"]["unavailable_components"],
            schedulable=world["schedulable"], repository_world_projection="unavailable",
            semantic_root_cid=world["planning_context"]["semantic_root_cid"]),
        identity_join="same_exact_workspace_distinct_owner_namespaces",
        measurement_scope=observed["measurement_scope"],
        freshness_scope="source_session_and_live_native_intent",
        missing_authority="accepted_world_tree_policy_contract_and_other_unavailable_components",
        **_AUTHORITY)
    if native_owner_binding is not None:
        material.update(native_owner_binding=native_owner_binding,
                        native_owner_access="in_process_exclusive_connection")
    summary = _json(material)
    observation_cid = content_identity(material)
    reference = ContextReference(reference_id="repository-intent-world:" + observation_cid,
        kind=KIND, tier=ContextTier.INVARIANT, referenced_content_id=observation_cid,
        repository_id=before["datasets_head"]["repository_id"],
        tree_id="sha256:" + before["source_tree_sha256"], summary=summary,
        byte_count=len(summary.encode("utf-8")),
        metadata=dict(required=True, coverage_ids=("repository-intent-world:" + task_id,),
                      authority="observed_native_public_checks", **_AUTHORITY))
    _require(len(reference.canonical_bytes()) <= max_bytes, "intent_world_reference_byte_limit")
    receipt = dict(schema=RECEIPT_SCHEMA, observation_cid=observation_cid,
        reference_id=reference.reference_id, reference_content_id=reference.content_id,
        summary_sha256=_sha(summary), summary_bytes=len(summary.encode("utf-8")),
        reference_bytes=len(reference.canonical_bytes()), task_id=task_id,
        task_revision=selected["revision"], intent_owner_id=observed["intent_owner_id"],
        native_event_watermark=observed["native_event_watermark"],
        intent_repository_id=observed["repository_id"],
        intent_source_tree_id=observed["current_source_tree_id"],
        source_repository_id=before["datasets_head"]["repository_id"],
        source_tree_sha256=before["source_tree_sha256"], session_id=before["session_id"],
        generation=generation, world_snapshot_cid=world["snapshot"]["snapshot_cid"],
        world_capture_cid=world["capture_cid"],
        requirement_observation_cid=observed["observation_cid"],
        requirement_count=len(requirements), validation_count=len(observed_task["validations"]),
        world_schedulable=world["schedulable"], native_intent_freshness_checked=True,
        source_freshness_checked=True, resource_limits=dict(admission_memory_mb=256,
            cooperative_deadline_seconds=30, parser_hard_rss_limit=False), **_AUTHORITY)
    if native_owner_binding is not None:
        receipt.update(native_owner_binding=native_owner_binding,
                       native_owner_access="in_process_exclusive_connection")
    _require(owner_current() and _admission_identity(original_admission) == admission_sha,
             "intent_world_owner_or_admission_changed")
    return reference, _json(receipt), owner, admission_sha


@dataclass(frozen=True, eq=False)
class RepositoryIntentWorldObservation:
    """Issued evidence with live owner references; not a portable authority."""

    reference: ContextReference
    _receipt_json: str = field(repr=False)
    _session: RepositorySourceSession = field(repr=False)
    _intent: IntentRepository | None = field(repr=False)
    _admission: Mapping = field(repr=False)
    _owner: tuple = field(repr=False)
    _admission_sha: str = field(repr=False)
    _task_id: str = field(repr=False)
    _generation: int = field(repr=False)
    _max_bytes: int = field(repr=False)
    _server: object | None = field(default=None, repr=False)

    def receipt(self) -> dict:
        return json.loads(self._receipt_json)

    def _seal(self):
        return (self.reference.canonical_bytes(), self._receipt_json, self._session, self._intent,
                id(self._admission), self._owner, self._admission_sha, self._task_id,
                self._generation, self._max_bytes, id(self._server))

    def validate_current(self, *, session, generation, task_id) -> dict:
        with _ISSUANCE_LOCK:
            issued = _ISSUED.get(self)
        _require(issued is not None and issued == self._seal(), "intent_world_not_issued")
        _require(session is self._session and generation == self._generation
                 and type(generation) is int and task_id == self._task_id,
                 "intent_world_consumer_binding_mismatch")
        # Native owner checks run inside _capture, after resource admission.
        _require((self._server is not None or _owner_identity(self._intent) == self._owner)
                 and _admission_identity(self._admission) == self._admission_sha,
                 "intent_world_owner_or_admission_changed")
        reference, receipt, owner, admission_sha = _capture(session=session, intent=self._intent,
            admission=self._admission, task_id=task_id, generation=generation, max_bytes=self._max_bytes,
            server=self._server, expected_owner=self._owner)
        _require(reference.canonical_bytes() == self.reference.canonical_bytes()
                 and receipt == self._receipt_json and owner == self._owner
                 and admission_sha == self._admission_sha, "intent_world_observation_changed")
        return self.receipt()


def capture_repository_intent_world(*, session, intent, admission, task_id,
                                    generation, max_bytes=32768) -> RepositoryIntentWorldObservation:
    """Capture all selected native task evidence, or refuse without truncating."""
    reference, receipt, owner, admission_sha = _capture(session=session, intent=intent,
        admission=admission, task_id=task_id, generation=generation, max_bytes=max_bytes)
    result = RepositoryIntentWorldObservation(reference, receipt, session, intent, admission,
        owner, admission_sha, task_id, generation, max_bytes)
    with _ISSUANCE_LOCK:
        _ISSUED[result] = result._seal()
    return result


def capture_repository_owner_intent_world(*, session, server, admission, task_id,
                                          generation, max_bytes=32768) -> RepositoryIntentWorldObservation:
    """Capture through a live in-process native Quack owner, without reopening it.

    The exact server and connection remain private custody references. This is
    not a remote-client API or a portable receipt: each dispatch must reobserve
    that live owner. Signed materialization and current server identities are
    retained separately because custody may have transferred between them.
    """
    _require_native_server(server)
    reference, receipt, owner, admission_sha = _capture(session=session, intent=None,
        server=server, admission=admission, task_id=task_id, generation=generation, max_bytes=max_bytes)
    result = RepositoryIntentWorldObservation(reference, receipt, session, None, admission,
        owner, admission_sha, task_id, generation, max_bytes, server)
    with _ISSUANCE_LOCK:
        _ISSUED[result] = result._seal()
    return result
