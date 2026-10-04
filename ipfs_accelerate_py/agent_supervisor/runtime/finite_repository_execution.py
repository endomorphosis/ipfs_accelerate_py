"""One ready finite task in a complete, genuinely admitted native population.

This private launch control holds native shared admission until native STOP and
isolated worker cleanup finish. Its signed material describes prelaunch rows;
the existing typed owner still issues the actual later claim and fencing token.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shlex
import stat
import threading
import time
import uuid

from ipfs_datasets_py.logic.software_contracts.content import (
    canonical_dag_json_bytes, cid_for_structured,
)
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceLane, ResourceLease,
)

from ..planning.repository_plan_preview import RepositoryPlanPreviewOwner
from ..planning import finite_integer_source_custody as source_custody
from ..task_sources.intent_repository import IntentRepository
from ..task_sources.typed_database_task_source import TypedDatabaseTaskSource
from .quack_state_server import QuackStateServer, ServerLifecycle
from . import finite_repository_admission as finite
from . import local_planning_admission as local

PROFILE = "finite-repository-one-ready-native-worker@1"
SCHEMA = "supervisor-finite-repository-execution-scope@1"
ADVISORY_PROFILE = "finite-repository-one-ready-advisory-bound-native-worker@1"
ADVISORY_SCHEMA = "supervisor-finite-repository-advisory-execution-scope@1"
_SEAL = object()
_ACTIVE = {}
_RETAINED_UNSAFE_SCOPES = {}
_LOCK = threading.RLock()


class FiniteRepositoryExecutionError(ValueError):
    """The exact finite launch scope or its safe release was refused."""


def _need(condition, message):
    if not condition:
        raise FiniteRepositoryExecutionError(message)


def _same(left, right):
    return canonical_dag_json_bytes(left) == canonical_dag_json_bytes(right)


def _pins(advisory=False, proof_query=False):
    from ..entrypoints import admitted_benchmark_runtime
    from . import finite_repository_candidate_runner
    modules = (finite, local, admitted_benchmark_runtime, finite_repository_candidate_runner,
               source_custody)
    if advisory:
        from . import finite_advisory_artifact_closure, local_completion_bridge
        from ..task_sources import typed_state_owner
        from ..control import before_popen_refusal, control_plane, lifecycle_orchestrator
        modules += (finite_advisory_artifact_closure, before_popen_refusal,
                    control_plane, lifecycle_orchestrator, local_completion_bridge, typed_state_owner)
    if proof_query:
        from . import finite_proof_query_execution, finite_proof_query_worker_context
        from . import finite_proof_query_worker_dispatch, local_completion_bridge
        from . import finite_proof_query_worker_source_custody, finite_proof_query_worktree_allocation
        from ..merge import worktree_lifecycle, database_coordination
        from ..task_sources import typed_state_owner, intent_repository, database_task_source, board_control_plane
        from ..todo_daemon import database_portal_bridge
        from ..control import before_popen_refusal
        from ..todo_daemon import implementation_daemon, supervisor_runtime
        modules += (finite_proof_query_execution, finite_proof_query_worker_context,
                    finite_proof_query_worker_dispatch, local_completion_bridge,
                    finite_proof_query_worker_source_custody, finite_proof_query_worktree_allocation,
                    worktree_lifecycle, database_coordination,
                    typed_state_owner, intent_repository, database_task_source, board_control_plane,
                    database_portal_bridge,
                    before_popen_refusal, implementation_daemon, supervisor_runtime)
    return {module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in modules} | {
                __name__: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _candidate(descriptor, semantic, population, admission_cid, proof_query_closure=None):
    from .finite_repository_candidate_runner import load_finite_repository_candidate
    fields = {"artifact", "sha256", "candidate_cid", "finite_admission_cid", "semantic_context_cid",
              "task_cid", "task_id", "task_revision", "before_sha256", "after_sha256"}
    _need(type(descriptor) is dict and set(descriptor) == fields, "exact authored finite candidate descriptor required")
    candidate = load_finite_repository_candidate(artifact=Path(descriptor["artifact"]),
                                                expected_sha256=descriptor["sha256"])
    selected = population["selected_task_cids"][0]
    task = next(row for row in population["tasks"] if row["task_cid"] == selected)
    expected = {"artifact": descriptor["artifact"], "sha256": descriptor["sha256"],
        "candidate_cid": candidate["candidate_cid"], "finite_admission_cid": admission_cid,
        "semantic_context_cid": cid_for_structured(semantic), "task_cid": selected,
        "task_id": task["task_alias"], "task_revision": task["revision"],
        "before_sha256": candidate["edit"]["before_sha256"], "after_sha256": candidate["edit"]["after_sha256"]}
    _need(_same(descriptor, expected)
          and candidate["finite_admission_cid"] == admission_cid
          and candidate["semantic_context_cid"] == expected["semantic_context_cid"]
          and candidate["task_cid"] == selected and candidate["task_id"] == task["task_alias"]
          and type(candidate["task_revision"]) is int and candidate["task_revision"] == task["revision"],
          "finite candidate context, task or ready revision differs from native launch scope")
    argv = ["/opt/ipfs-supervisor/bin/owner-worker", "--finite-repository-artifact", descriptor["artifact"],
            "--finite-repository-sha256", descriptor["sha256"], "--finite-repository-task-cid", selected]
    binding = {"descriptor": descriptor, "argv": argv, "implementation_command": shlex.join(argv)}
    if proof_query_closure is not None:
        return proof_query_closure.extend_candidate_binding(binding)
    return binding


def _physical_native(connection):
    """Detached owner SQL projection; no signer, source or policy callbacks."""
    fields = ("task_cid", "task_alias", "goal_cid", "plan_cid", "objective_id", "ordinal", "status",
              "revision", "priority", "created_at", "updated_at", "identity", "body")
    rows = connection.execute("SELECT task_cid, task_alias, goal_cid, plan_cid, objective_id, ordinal, "
        "status, revision, priority, created_at, updated_at, identity_json, body_json FROM tasks "
        "ORDER BY task_cid LIMIT 17").fetchall()
    _need(len(rows) <= 16, "native full population exceeds bounded execution profile")
    tasks, receipts = [], {}
    for values in rows:
        row = {name: values[position] for position, name in enumerate(fields)}
        for name in fields:
            if name in {"identity", "body"}:
                row[name] = json.loads(row[name])
            elif name in {"ordinal", "revision"}:
                row[name] = int(row[name])
            else:
                row[name] = str(row[name] or "")
        cid = row["task_cid"]
        row["dependencies"] = [str(value[0]) for value in connection.execute(
            "SELECT dependency_task_cid FROM task_dependencies WHERE task_cid = ? ORDER BY dependency_task_cid",
            [cid]).fetchall()]
        for name, query, keys in (
                ("outputs", "SELECT ordinal,path,effect_json FROM task_outputs WHERE task_cid = ? ORDER BY ordinal",
                 ("ordinal", "path", "effect")),
                ("acceptance", "SELECT ordinal,criterion,evidence_policy_json FROM task_acceptance WHERE task_cid = ? ORDER BY ordinal",
                 ("ordinal", "criterion", "evidence_policy")),
                ("validations", "SELECT ordinal,argv_json,policy_json FROM task_validations WHERE task_cid = ? ORDER BY ordinal",
                 ("ordinal", "argv", "policy"))):
            row[name] = [{keys[0]: int(value[0]), keys[1]: json.loads(value[1]) if name == "validations"
                          else str(value[1]), keys[2]: json.loads(value[2])}
                         for value in connection.execute(query, [cid]).fetchall()]
        if row["status"] == "completed":
            receipts[cid] = [[value[position] for position in range(10)] for value in connection.execute(
                "SELECT receipt_cid, task_cid, goal_cid, attempt_id, claim_cid, fencing_token, "
                "completed_at, validation_run_id, evidence_digest, body_json "
                "FROM completion_receipts WHERE task_cid = ? ORDER BY completed_at, receipt_cid LIMIT 129",
                [cid]).fetchall()]
        tasks.append(row)
    return {"tasks": tasks, "completion_rows": receipts}


def _candidate_bytes(descriptor):
    path = Path(descriptor["artifact"])
    raw = finite._read(path, finite.MAX_BYTES)
    info = path.stat(follow_symlinks=False)
    _need(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444
          and info.st_uid == os.geteuid() and info.st_nlink == 1
          and hashlib.sha256(raw).hexdigest() == descriptor["sha256"],
          "immutable finite candidate descriptor bytes or readonly ownership changed")


def _native_population(server, source, admission, semantic):
    """Replay full native contracts and actual completed-prerequisite receipts."""
    _need(type(server) is QuackStateServer and server.lifecycle is ServerLifecycle.READY
          and type(source) is TypedDatabaseTaskSource and source.execution_route_policy is not None,
          "live native owner and route-sealed task source required")
    verified = local.verify_local_benchmark_admission(admission, initial=True)
    _need(server.identity.repository_id == verified["manifest"]["repository_cid"],
          "native owner repository differs from signed local admission")
    expected = {task.task_cid: task for task in verified["graph"].tasks}
    residual = semantic["residual_requirement_ids"]
    _need(len(residual) == 1, "first finite execution profile requires exactly one residual requirement")
    selected = semantic["native_task_bindings"][residual[0]]["task_cid"]
    _need(selected in expected and sorted(expected) == semantic["administrator_task_cids"],
          "full original task population and exact residual binding required")
    page = source.list_tasks(limit=17)
    _need(not page.next_cursor and len(page.tasks) == len(expected)
          and {row.task_cid for row in page.tasks} == set(expected),
          "native owner task population differs from full signed admission")
    ready_page = source.ready_tasks(limit=17)
    _need(not ready_page.next_cursor, "bounded native readiness population required")
    ready = {row.task_cid for row in ready_page.tasks}
    _need(ready == {selected}, "exactly the finite residual task must be native-ready")
    population, completion, completion_rows = [], {}, {}
    with server._lock:
        intent = IntentRepository(bound_connection=server._connection, install_schema=False)
        for projected in sorted(page.tasks, key=lambda row: row.task_cid):
            native = intent.get_task(projected.task_cid)
            _need(native is not None and type(projected.revision) is int and projected.revision > 0,
                  "native task revision required")
            row = local._plain(dict(native))
            _need(row["revision"] == projected.revision and row["status"] == projected.status
                  and _same(row["body"], local._plain(dict(projected.body)))
                  and row["task_alias"] == projected.task_alias
                  and row["goal_cid"] == projected.goal_cid and row["plan_cid"] == projected.plan_cid
                  and row["dependencies"] == list(projected.dependencies),
                  "typed task projection differs from actual native owner rows")
            envelope = row["body"].get(local.CONTRACT_KEY, {})
            contract = local._verify_signature(envelope, verified["profile"])
            owner_id = contract.get("intent_owner_id")
            _need(type(owner_id) is str and owner_id, "signed task owner required")
            wanted = local._pending_contract_payload(admission=admission, verified=verified,
                task=expected[row["task_cid"]], intent_owner_id=owner_id)
            _need(_same(contract, wanted)
                  and row["identity"].get("local_contract_cid") == local.content_identity(envelope)
                  and row["task_alias"] == wanted["task_key"]
                  and row["dependencies"] == wanted["dependencies"]
                  and row["goal_cid"] == expected[row["task_cid"]].goal_cid
                  and row["plan_cid"] == wanted["plan_id"],
                  "native task differs from complete signed pending contract")
            plan = intent.get_plan(row["plan_cid"])
            reference = plan["body"].get("finite_repository_admission_ref", {}) if plan else {}
            _need(reference.get("schema") == finite.REFERENCE_SCHEMA
                  and reference.get("admission_cid") == semantic["finite_admission_cid"],
                  "native plan lacks this immutable full finite admission reference")
            raw = finite._read(reference["path"], finite.MAX_BYTES)
            _need(len(raw) == reference["bytes"]
                  and hashlib.sha256(raw).hexdigest() == reference["sha256"]
                  and cid_for_structured(json.loads(raw)) == reference["admission_cid"],
                  "retained full finite admission differs from native plan reference")
            if row["task_cid"] == selected:
                _need(row["status"] == "ready", "selected residual task is not ready")
            else:
                _need(row["status"] == "completed" and row["revision"] >= 2,
                      "all other original tasks require actual native completion")
                receipts = server._connection.execute(
                    "SELECT receipt_cid, task_cid, goal_cid, attempt_id, claim_cid, fencing_token, "
                    "completed_at, validation_run_id, evidence_digest, body_json "
                    "FROM completion_receipts WHERE task_cid = ? ORDER BY completed_at, receipt_cid LIMIT 129",
                    [row["task_cid"]]).fetchall()
                _need(len(receipts) <= 128, "bounded completion receipt population required")
                completion_rows[row["task_cid"]] = [[value[position] for position in range(10)]
                                                     for value in receipts]
                binding, reasons = IntentRepository._current_task_completion_binding(row, receipts)
                _need(binding is not None and not reasons
                      and not local.local_completion_missing(server._connection, row["task_cid"],
                                                            row["body"], row["revision"] - 1),
                      "completed prerequisite lacks current native public-check evidence")
                completion[row["task_cid"]] = binding
            population.append(row)
    return {"tasks": population, "completed_prerequisites": completion, "completion_rows": completion_rows,
            "selected_task_cids": [selected], "owner_identity": server.identity.to_dict(),
            "execution_route_policy": source.execution_route_policy.to_dict()}


class FrozenFiniteRepositoryExecutionScope:
    """Private live lease control; a serialized record cannot grant launch."""

    def __init__(self, seal, *, owner, admission, server, source, lease, output, observer,
                 advisory_closure=None, proof_query_closure=None):
        _need(seal is _SEAL, "execution scope must be created by native reservation")
        self._seal, self._owner, self._admission = seal, owner, admission
        self._server, self._source, self._lease = server, source, lease
        self._output, self._observer = output, observer
        self._advisory_closure = advisory_closure
        self._proof_query_closure = proof_query_closure
        self._runtime, self._spawned, self._cleaned, self._released = None, False, False, False
        self._renew_stop, self._renew_failed = threading.Event(), threading.Event()
        # An execution envelope must survive even schedulers whose preview
        # leases disable automatic renewal. Keep renewing until safe release.
        self._renew_thread = threading.Thread(target=self._renew, daemon=True,
                                              name="finite-execution-lease-" + lease.lease_id[:8])
        self._renew_thread.start()

    def _renew(self):
        interval = max(0.01, min(20.0, self._lease._scheduler.config.lease_ttl_seconds / 3))
        while not self._renew_stop.wait(interval):
            try:
                if not self._lease.renew():
                    self._renew_failed.set()
                    return
            except Exception:
                # No release on a transient failure. The next native read
                # detects expiry; STOP and cleanup remain available.
                self._renew_failed.set()

    @property
    def parent_lease(self):
        return self._lease

    @property
    def selected_task_cids(self):
        return tuple(json.loads(self._material)["payload"]["native_population"]["selected_task_cids"])

    @property
    def material_binding(self):
        return self.to_dict()

    def to_dict(self):
        return json.loads(self._material)

    def _active(self, *, allow_cancelled=False):
        with _LOCK:
            _need(self._seal is _SEAL and _ACTIVE.get(self._lease.lease_id) is self
                  and not self._released and not self._lease.released,
                  "an exact active native finite execution scope is required")
        if not allow_cancelled:
            _need(not self._renew_failed.is_set() and not self._lease.cancelled
                  and (self._owner.cancel_event is None or not self._owner.cancel_event.is_set()),
                  "native execution reservation cancelled or expired")

    def _checkpoint(self):
        self._active()
        return self._owner.timeout_seconds

    def require_prelaunch_current(self):
        """Fresh native source observation, then detached source/artifact closure."""
        self._active()
        _need(not self._spawned, "finite execution scope permits one native process launch")
        bound = self.to_dict()["payload"]
        pins = _pins(advisory=self._advisory_closure is not None,
                     proof_query=self._proof_query_closure is not None)
        _need(pins == bound["implementation"], "selected finite launch producer bytes changed")
        verified = finite.verify_finite_repository_admission(admission=self._admission)
        _need(_same(verified["semantic_context"], self._semantic), "finite context changed")
        self._owner.index.observe_current(self._owner.repository, expected_head=self._owner.expected_head,
            scheduler=None, parent_lease=self._lease, cancel_event=self._owner.cancel_event,
            timeout_seconds=self._owner.timeout_seconds, memory_mb=self._owner.memory_mb)
        payload = self._admission["declaration"]["payload"]
        from ..planning.plan_revision_contracts import PlanAuthorityRoots, PlanCreateRequest
        request = PlanCreateRequest.from_dict(payload["request"])
        roots = self._observer(request)
        _need(type(roots) is PlanAuthorityRoots, "exact independently observed native policy roots required")
        roots.require_current(request.roots)
        population = _native_population(self._server, self._source, self._admission["local_admission"],
                                       {**self._semantic, "finite_admission_cid": bound["finite_admission_cid"]})
        _need(_same(population, bound["native_population"]), "ready revision or full native task population changed")
        candidate = _candidate(bound["candidate"]["descriptor"], self._semantic, population,
                               bound["finite_admission_cid"], self._proof_query_closure)
        _need(_same(candidate, bound["candidate"]), "closed finite worker invocation changed")
        profile = finite._profile(payload["manifest"])
        signed = local._verify_signature(self.to_dict(), profile)
        _need(_same(signed, bound), "finite execution scope signature or closed material changed")
        self._custody.require_current(self._checkpoint)
        self._fence()
        self._detached_fence()
        self._require_advisory_current()
        self._require_proof_query_current()

    def _require_advisory_current(self):
        """Detached advisory bytes after callbacks; never a shutdown condition."""
        closure = self._advisory_closure
        bound = self.to_dict()["payload"]
        if closure is None:
            if self._proof_query_closure is None:
                schema, profile = SCHEMA, PROFILE
            else:
                from .finite_proof_query_execution import EXECUTION_SCHEMA as schema, PROFILE as profile
            _need("advisory_closure" not in bound
                  and bound["schema"] == schema and bound["profile"] == profile,
                  "signed finite advisory closure cannot be removed from its live scope")
            return
        from .finite_advisory_artifact_closure import FrozenFiniteAdvisoryArtifactClosure
        _need(type(closure) is FrozenFiniteAdvisoryArtifactClosure
              and bound["schema"] == ADVISORY_SCHEMA and bound["profile"] == ADVISORY_PROFILE
              and _same(bound.get("advisory_closure"), closure.material_binding),
              "signed finite advisory closure differs from its exact live native object")
        closure.require_detached(head=self._owner.expected_head.to_dict(),
            finite_admission_cid=bound["finite_admission_cid"],
            semantic_context_cid=bound["semantic_context_cid"],
            candidate=bound["candidate"]["descriptor"],
            administrator_task_cids=self._semantic["administrator_task_cids"])

    def _require_proof_query_current(self):
        """Close the opt-in indexed context; shutdown never calls this method."""
        closure = self._proof_query_closure
        bound = self.to_dict()["payload"]
        if closure is None:
            _need("proof_query_closure" not in bound,
                  "signed proof-query context cannot lose its live receiving control")
            return
        from .finite_proof_query_execution import (
            FrozenFiniteProofQueryExecutionClosure, EXECUTION_SCHEMA as schema, PROFILE as profile,
        )
        _need(type(closure) is FrozenFiniteProofQueryExecutionClosure
              and self._advisory_closure is None
              and bound["schema"] == schema and bound["profile"] == profile
              and _same(bound.get("proof_query_closure"), closure.material_binding),
              "signed proof-query context differs from its exact live native control")
        closure.require_detached(scope=self)

    def _uses_paired_receiving(self):
        return self._proof_query_closure is not None

    def _receiving_operation(self, *, purpose, runtime=None):
        _need(self._proof_query_closure is not None,
              "paired finite receiving requires the new exact proof-query control")
        return self._proof_query_closure._receiving_operation(purpose=purpose, runtime=runtime)

    def _finish_creation_receiving(self, runtime):
        self._proof_query_closure._finish_creation_receiving(runtime)

    @contextmanager
    def _proof_query_spawn_guard(self, runtime):
        """Keep the current proof owner locked across the actual supervisor birth."""
        if self._proof_query_closure is None:
            yield
        else:
            with self._proof_query_closure.require_spawn_fence(scope=self, runtime=runtime):
                yield

    def _detached_fence(self):
        bound = self.to_dict()["payload"]
        with self._server._lock:
            actual = _physical_native(self._server._connection)
        _need(_same(actual, {key: bound["native_population"][key] for key in ("tasks", "completion_rows")}),
              "full native rows or ready revision changed after prelaunch callbacks")
        _candidate_bytes(bound["candidate"]["descriptor"])
        if hasattr(self, "_launcher"):
            from .candidate_execution import _root_file
            _need(_root_file(Path(self._launcher["path"])) == self._launcher["sha256"],
                  "finite worker launcher bytes changed after prelaunch callbacks")

    def require_spawn_fence(self, runtime):
        """Run under the native owner lock immediately before actual Popen."""
        self._active()
        _need(runtime is self._runtime and not self._spawned, "exact unlaunched finite runtime required")
        self._physical_source()
        self._fence()
        self._detached_fence()
        self._require_advisory_current()
        self._require_proof_query_current()

    def _physical_source(self):
        """Repeat only custody's detached file/Git closure at process birth."""
        custody = self._custody
        working = source_custody._working(self._owner, custody._exclusions, self._checkpoint)
        git = source_custody._git(self._owner, self._checkpoint)
        _need(working == custody._working_inventory and git == custody._git_inventory,
              "finite source or Git inventory changed at process launch")
        for expected in custody._files:
            limit = source_custody.LIMITS["native_object_bytes"] if expected.role.startswith("cas:") \
                else source_custody.LIMITS["working_file_bytes"]
            actual, raw = source_custody._read(expected.path, role=expected.role, bound=limit,
                                               checkpoint=self._checkpoint)
            if expected.role == "git:index":
                _need(actual.path == expected.path and actual.role == expected.role
                      and actual.witness[2] == expected.witness[2]
                      and canonical_dag_json_bytes(source_custody._index_identity(
                          raw, custody._manifest.snapshot.git_commit)) == custody._git_index,
                      "finite staged Git identity changed at process launch")
            else:
                _need((actual.path, actual.role, actual.size, actual.sha, actual.witness[2]) ==
                      (expected.path, expected.role, expected.size, expected.sha, expected.witness[2]),
                      "finite source or captured evidence bytes changed at process launch")
        head = self._owner.expected_head
        rows = self._owner.index.catalog._cx.execute(
            "SELECT repository_id,generation,manifest_cid,snapshot_cid,ast_revision_id,receipt_cid "
            "FROM codebase_control.heads WHERE repository_id=? LIMIT 2", [head.repository_id]).fetchall()
        _need(rows == [(head.repository_id, head.generation, head.manifest_cid, head.snapshot_cid,
                        head.ast_revision_id, head.receipt_cid)], "selected native source head changed at process launch")
        _need(source_custody._working(self._owner, custody._exclusions, self._checkpoint) == working
              and source_custody._git(self._owner, self._checkpoint) == git,
              "finite source inventory changed during process launch closure")

    def require_launch(self, *, admission, server, source, implement, candidate_runner,
                       context_bundle, refresh_context_on_completion, implementation_command):
        self._active()
        _need(server is self._server and source is self._source
              and _same(admission, self._admission["local_admission"])
              and implement is True and candidate_runner is not None
              and context_bundle is None and refresh_context_on_completion is False,
              "finite worker launch requires its exact native owner, full admission and isolated runner")
        bound = self.to_dict()["payload"]
        _need(type(implementation_command) is str
              and implementation_command == bound["candidate"]["implementation_command"],
              "finite scope requires the exact immutable candidate worker command")
        from .candidate_execution import _root_file, verify_candidate_runner
        verify_candidate_runner(candidate_runner)
        path = Path(bound["candidate"]["argv"][0])
        self._launcher = {"path": str(path), "sha256": _root_file(path)}
        self.require_prelaunch_current()

    @property
    def worker_launcher_binding(self):
        return dict(self._launcher)

    def bind_runtime(self, runtime):
        self._active()
        from ..entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
        _need(type(runtime) is AdmittedBenchmarkRuntime and runtime.finite_execution_scope is self
              and self._runtime is None
              and runtime.server is self._server and runtime.source is self._source
              and _same(runtime.admission, self._admission["local_admission"]),
              "scope can bind exactly one matching native runtime")
        self._runtime = runtime
        if self._proof_query_closure is not None:
            self._proof_query_closure.bind_runtime(runtime)

    def require_runtime(self, runtime, *, before_spawn=False, stopping=False):
        self._active(allow_cancelled=stopping)
        _need(runtime is self._runtime
              and _same(runtime.manifest.get("finite_execution_scope"), self.to_dict()),
              "signed launch is not bound to this exact active finite scope")
        from .candidate_execution import _root_file
        _need(_same(runtime.manifest.get("finite_worker_launcher"), self._launcher)
              and _root_file(Path(self._launcher["path"])) == self._launcher["sha256"],
              "finite worker launcher differs from the signed native launch")
        if self._proof_query_closure is not None:
            self._proof_query_closure.require_runtime(scope=self, runtime=runtime, stopping=stopping)
        if before_spawn:
            self.require_prelaunch_current()

    def note_spawned(self, runtime):
        _need(runtime is self._runtime and not self._spawned, "finite execution scope already launched")
        # Popen has already succeeded. Record that cleanup is mandatory before
        # any cancellation/material check can throw in this parent process.
        self._spawned = True
        if self._proof_query_closure is not None:
            self._proof_query_closure.note_spawned(runtime)

    def finish_runtime(self, runtime):
        """No source, revision or cancellation gate may prevent safe shutdown."""
        self.require_runtime(runtime, stopping=True)
        _need(runtime._context_refresh_stopped(), "finite release requires native STOP and isolated UID cleanup")
        self._cleaned = True

    def require_close(self, runtime):
        _need(runtime is self._runtime, "finite close requires its exact runtime")
        if self._spawned:
            _need(self._cleaned and runtime._context_refresh_stopped(),
                  "finite runtime close requires successful STOP and isolated UID cleanup")

    def _finish(self):
        _need(not getattr(self._runtime, "_construction_cleanup_failed", False),
              "failed native construction cleanup retains its resource envelope")
        if (self._runtime is not None and hasattr(self._runtime, "process")
                and self._runtime.process.snapshot(self._runtime.profile).members):
            self._spawned = True
        if self._runtime is not None and self._spawned and not self._cleaned:
            self._runtime.stop()
            self.finish_runtime(self._runtime)
        if self._runtime is not None and hasattr(self._runtime, "process"):
            _need(not self._runtime.process.snapshot(self._runtime.profile).members,
                  "finite execution envelope still owns a live native process")
            if self._spawned:
                _need(self._cleaned and self._runtime._context_refresh_stopped(),
                      "finite execution envelope lacks successful isolated worker cleanup")
        self._renew_stop.set()
        self._lease.release()
        self._released = True
        with _LOCK:
            _ACTIVE.pop(self._lease.lease_id, None)
            _RETAINED_UNSAFE_SCOPES.pop(self._lease.lease_id, None)


@contextmanager
def reserve_finite_repository_execution(*, owner, admission, candidate, server, source, output, policy_observer,
        cpu_slots=4, memory_mb=4096, child_process_slots=8, admission_timeout_seconds=30,
        advisory_closure=None, proof_query_closure=None):
    """Hold a real orchestration envelope through STOP and isolated cleanup.

The resource budget is native admission/accounting, not an OS limits claim.
Failure to demonstrate cleanup retains the live lease and heartbeat for retry.
"""
    _need(type(owner) is RepositoryPlanPreviewOwner and callable(policy_observer),
          "exact native codebase owner and policy observer required")
    if advisory_closure is not None:
        from .finite_advisory_artifact_closure import FrozenFiniteAdvisoryArtifactClosure
        _need(type(advisory_closure) is FrozenFiniteAdvisoryArtifactClosure,
              "exact live native finite advisory artifact closure required")
    if proof_query_closure is not None:
        from .finite_proof_query_execution import FrozenFiniteProofQueryExecutionClosure
        _need(type(proof_query_closure) is FrozenFiniteProofQueryExecutionClosure
              and advisory_closure is None,
              "exact proof-query execution control requires its separate finite profile")
    _need(type(cpu_slots) is int and 4 <= cpu_slots <= 16
          and type(memory_mb) is int and 4096 <= memory_mb <= 16384
          and type(child_process_slots) is int and 8 <= child_process_slots <= 64
          and type(admission_timeout_seconds) in {int, float} and 0 < admission_timeout_seconds <= 90,
          "bounded full-process native execution envelope required")
    scheduler = owner.scheduler
    if scheduler is None and type(owner.parent_lease) is ResourceLease:
        scheduler = owner.parent_lease._scheduler
    _need(type(scheduler) is GlobalResourceScheduler and scheduler.config.proof_safety_enabled,
          "native shared scheduler with host proof safety enabled required")
    frozen = finite.verify_finite_repository_admission(admission=admission)
    admission = frozen["admission"]
    candidate = finite._plain(candidate)
    _need(admission["local_admission"] is not None and frozen["receipt"]["planning_permitted"] is True,
          "no-work finite preview is review-only and cannot launch a worker")
    if advisory_closure is not None:
        advisory_closure.require_detached(head=owner.expected_head.to_dict(),
            finite_admission_cid=cid_for_structured(admission),
            semantic_context_cid=cid_for_structured(frozen["semantic_context"]),
            candidate=candidate,
            administrator_task_cids=frozen["semantic_context"]["administrator_task_cids"])
    output = Path(output).absolute()
    _need(not output.exists() and output.resolve() == output and not output.is_relative_to(owner.repository),
          "fresh external execution evidence directory required")
    lease = scheduler.acquire(ResourceLane.ORCHESTRATION, cpu_slots=cpu_slots, memory_mb=memory_mb,
        child_process_slots=child_process_slots, parent_lease=owner.parent_lease,
        timeout=admission_timeout_seconds, cancel_event=owner.cancel_event,
        request_id="finite-native-execution-" + uuid.uuid4().hex)
    owned = replace(owner, scheduler=None, parent_lease=lease,
                    cancel_event=lease.combined_cancellation_signal(owner.cancel_event))
    scope = FrozenFiniteRepositoryExecutionScope(_SEAL, owner=owned, admission=admission,
        server=server, source=source, lease=lease, output=output, observer=policy_observer,
        advisory_closure=advisory_closure, proof_query_closure=proof_query_closure)
    with _LOCK:
        _ACTIVE[lease.lease_id] = scope
    try:
        output.mkdir(mode=0o700)
        evidence, semantic, custody, checkpoint, fence = finite._fresh(owned, admission["declaration"],
            admission["graph"], output / "current-preview", policy_observer)
        _need(_same(semantic, frozen["semantic_context"]), "fresh finite source/context partition differs")
        population = _native_population(server, source, admission["local_admission"],
            {**semantic, "finite_admission_cid": cid_for_structured(admission)})
        candidate_binding = _candidate(candidate, semantic, population, cid_for_structured(admission),
                                       proof_query_closure)
        if proof_query_closure is not None:
            from .finite_proof_query_execution import EXECUTION_SCHEMA as selected_schema, PROFILE as selected_profile
        else:
            selected_schema = ADVISORY_SCHEMA if advisory_closure is not None else SCHEMA
            selected_profile = ADVISORY_PROFILE if advisory_closure is not None else PROFILE
        payload = {"schema": selected_schema, "profile": selected_profile,
            "finite_admission_cid": cid_for_structured(admission),
            "semantic_context_cid": cid_for_structured(semantic),
            "fresh_evidence_cid": cid_for_structured(evidence), "native_population": population,
            "candidate": candidate_binding,
            "head": owned.expected_head.to_dict(), "source_custody": custody.material_binding,
            "lease": {"lease_id": lease.lease_id, "lane": lease.lane, "cpu_slots": lease.cpu_slots,
                      "memory_mb": lease.memory_mb, "child_process_slots": lease.child_process_slots,
                      "owner_pid": lease.owner_pid, "parent_lease_id": lease.parent_lease_id},
            "implementation": _pins(advisory=advisory_closure is not None,
                                    proof_query=proof_query_closure is not None),
            "task_population_preserved": True,
            "finite_facts_are_context_only": True, "future_claim_and_fence": "existing_native_typed_owner",
            "resource_scope": "native_admission_accounting_until_STOP_and_isolated_cleanup",
            "task_omission_authority": False, "completion_authority": False,
            "proof_authority": False, "publication_authority": False, "production_activation": False}
        if advisory_closure is not None:
            payload["advisory_closure"] = advisory_closure.material_binding
        if proof_query_closure is not None:
            payload["proof_query_closure"] = proof_query_closure.material_binding
        envelope = local._signed(payload, admission["declaration"]["payload"]["manifest"]["payload"])
        scope._material = canonical_dag_json_bytes(envelope)
        scope._semantic, scope._custody = semantic, custody
        scope._fence = finite._artifact_fence(owned, evidence, scope._checkpoint)
        if proof_query_closure is not None:
            proof_query_closure._bind_scope(scope)
        custody.require_current(checkpoint)
        fence()
        scope.require_prelaunch_current()
        (output / "execution-scope.json").write_bytes(scope._material)
        scope._require_advisory_current()
        scope._require_proof_query_current()
        yield scope
    finally:
        try:
            scope._finish()
        except BaseException as error:
            with _LOCK:
                _RETAINED_UNSAFE_SCOPES[lease.lease_id] = scope
            raise FiniteRepositoryExecutionError(
                "native execution cleanup unproven; resource lease retained for safe STOP/cleanup retry") from error


__all__ = ["PROFILE", "SCHEMA", "ADVISORY_PROFILE", "ADVISORY_SCHEMA", "FiniteRepositoryExecutionError",
           "FrozenFiniteRepositoryExecutionScope", "reserve_finite_repository_execution"]
