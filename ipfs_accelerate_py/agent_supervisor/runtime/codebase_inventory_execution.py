"""Owner-held launch custody for one ready inventory-bound native worker.

The signed local task contracts grant the existing execution scope. Inventory
features remain advisory, with no facts or omitted tasks. Source, model, scan,
query-owner and task observations are sequential freshness fences, not one
atomic snapshot. The public worker independently checks its signed input and
allocated worktree; it cannot requery the private registry or evidence owner.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import stat
import threading
import uuid

from ipfs_datasets_py.logic.software_contracts.content import canonical_dag_json_bytes
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceLane, ResourceLease, collect_proof_host_resources,
)

from ..proof.formal_verification_contracts import content_identity
from ..task_sources.intent_repository import IntentRepository
from ..task_sources.typed_database_task_source import TypedDatabaseTaskSource
from .quack_state_server import QuackStateServer, ServerLifecycle
from . import codebase_inventory_evidence_admission as inventory
from . import local_planning_admission as local

PROFILE = "codebase-inventory-one-ready-native-worker@1"
SCHEMA = "supervisor-codebase-inventory-execution-scope@1"
MAX_TASKS = 16
_SEAL = object()
_ACTIVE = {}
_RETAINED_UNSAFE_SCOPES = {}
_LOCK = threading.RLock()
_OPERATION_SEAL = object()


class InventoryExecutionError(ValueError):
    """The exact inventory launch custody or safe release was refused."""


def _need(condition, message):
    if not condition:
        raise InventoryExecutionError(message)


def _same(left, right):
    # Canonical bytes preserve the difference between integers and booleans.
    return canonical_dag_json_bytes(left) == canonical_dag_json_bytes(right)


def _native_admission_projection(admission):
    """Project the two reviewed native requirement enums without type aliases.

    Signed JSON uses the enum values. The private guard additionally retains
    the exact enum paths/classes/members, so a string or authored dictionary
    cannot replace an enum while keeping the same operation identity.
    """
    from ..planning.formal_planning_contracts import EvidenceRequirementKind
    from ..proof.formal_verification_contracts import AssuranceLevel
    ledger = []

    def project(value, path):
        kind = type(value)
        if kind is EvidenceRequirementKind or kind is AssuranceLevel:
            field = ("kind" if kind is EvidenceRequirementKind else "minimum_code_assurance")
            _need(len(path) == 5 and path[:3] == ("receipt", "payload", "pending_requirements")
                  and type(path[3]) is int and path[4] == field
                  and type(value.value) is str and kind.__members__.get(value.name) is value
                  and len(ledger) < 4096,
                  "exact reviewed native admission enum field and member required")
            ledger.append({"path": list(path), "class": kind.__module__ + "." + kind.__qualname__,
                           "name": value.name, "value": value.value})
            return value.value
        if value is None or kind is str or kind is bool or kind is int:
            return value
        if kind is dict:
            _need(all(type(key) is str for key in value), "exact native admission object keys required")
            return {key: project(value[key], path + (key,)) for key in sorted(value)}
        if kind is list:
            return [project(item, path + (number,)) for number, item in enumerate(value)]
        raise InventoryExecutionError("unsupported native admission identity type: " + kind.__name__)

    _need(type(admission) is dict, "exact native admission object required")
    projected = project(admission, ())
    _need(len(canonical_dag_json_bytes(ledger)) <= 1024 * 1024,
          "native admission enum ledger exceeds bounded private guard")
    return projected, canonical_dag_json_bytes({"admission": projected, "enum_ledger": ledger})


def _pins():
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_receiving
    from ..entrypoints import admitted_benchmark_runtime
    from ..task_sources import typed_state_owner
    from . import candidate_execution, local_completion_bridge, router_public_instruction
    from . import codebase_inventory_evidence_worker_context
    from . import codebase_successor_dispatch_admission, codebase_successor_dispatch_context
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor_receiving
    modules = (inventory, local, admitted_benchmark_runtime, typed_state_owner,
               candidate_execution, local_completion_bridge, router_public_instruction,
               codebase_inventory_evidence_worker_context, codebase_inventory_receiving,
               codebase_successor_dispatch_admission, codebase_successor_dispatch_context,
               codebase_inventory_successor_receiving)
    return {module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in modules} | {__name__: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _physical_native(connection):
    """Detached full owner SQL projection without signer or source callbacks."""
    fields = ("task_cid", "task_alias", "goal_cid", "plan_cid", "objective_id", "ordinal", "status",
              "revision", "priority", "created_at", "updated_at", "identity", "body")
    rows = connection.execute("SELECT task_cid, task_alias, goal_cid, plan_cid, objective_id, ordinal, "
        "status, revision, priority, created_at, updated_at, identity_json, body_json FROM tasks "
        "ORDER BY task_cid LIMIT 17").fetchall()
    _need(len(rows) <= MAX_TASKS, "native full population exceeds bounded inventory execution profile")
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
            values = connection.execute(
                "SELECT receipt_cid, task_cid, goal_cid, attempt_id, claim_cid, fencing_token, "
                "completed_at, validation_run_id, evidence_digest, body_json "
                "FROM completion_receipts WHERE task_cid = ? ORDER BY completed_at, receipt_cid LIMIT 129",
                [cid]).fetchall()
            _need(len(values) <= 128, "bounded actual completion receipt population required")
            receipts[cid] = [[item[position] for position in range(10)] for item in values]
        tasks.append(row)
    return {"tasks": tasks, "completion_rows": receipts,
            "authority_rows": _physical_native_authority(connection)}


def _physical_native_authority(connection):
    """Exact bounded SQL inputs authenticated by the unlocked native replay."""
    result = {}
    for name, columns, source, order, maximum, body_columns, byte_limit in (
            ("plans", "plan_cid,goal_cid,plan_alias,status,created_at,updated_at,revision,body_json",
             "plans", "plan_cid", MAX_TASKS, ("body_json",), 4 * 1024 * 1024),
            ("goals", "goal_cid,objective_id,goal_alias,status,created_at,updated_at,revision,body_json",
             "goals", "goal_cid", MAX_TASKS, ("body_json",), 4 * 1024 * 1024),
            ("validation_evidence", "r.result_id,r.run_id,r.task_cid,r.ordinal,r.outcome,r.evidence_digest,"
             "r.body_json,e.event_id,e.global_sequence,e.event_type,e.body_json",
             "validation_results r JOIN domain_events e ON e.task_cid=r.task_cid "
             "AND e.event_type='intent.validation_recorded' "
             "AND json_extract_string(e.body_json,'$.subject_id')=r.result_id",
             "r.task_cid,e.global_sequence,r.result_id,e.event_id", 128 * MAX_TASKS,
             ("r.body_json", "e.body_json"), 8 * 1024 * 1024),
            ("store_generation", "generation,schema_revision,fence_epoch,revision,database_uuid,birth_id",
             "(SELECT * FROM store_generations ORDER BY generation DESC LIMIT 1)",
             "generation", 1, (), 0),
            ("metadata", "key,value", "control_plane_metadata", "key", 256,
             ("key", "value"), 64 * 1024)):
        measured = " + ".join("length(hex(COALESCE(" + column + ",'')))/2"
                              for column in body_columns) or "0"
        measured_row = connection.execute(
            "SELECT COUNT(*),CAST(COALESCE(SUM(" + measured + "),0) AS BIGINT) FROM " + source).fetchone()
        count, size = measured_row[0], measured_row[1]
        _need(type(count) is int and 0 <= count <= maximum
              and type(size) is int and 0 <= size <= byte_limit,
              "native detached authority rows exceed bounded inventory profile: " + name)
        rows = connection.execute("SELECT " + columns + " FROM " + source
                                  + " ORDER BY " + order + " LIMIT " + str(maximum + 1)).fetchall()
        _need(len(rows) == count, "native detached authority row count changed")
        # These are fixed-schema SQL columns. Preserve raw JSON and NULL, with
        # a stable textual representation of the other native column values.
        result[name] = [[None if row[position] is None else str(row[position])
                         for position in range(len(row))] for row in rows]
    return result


def _planning_receipt_bytes(reference, manifest):
    """Final guarded byte identity only; no signer/compiler/typed callbacks."""
    _need(type(reference) is dict and type(reference.get("bytes")) is int
          and 0 < reference["bytes"] <= local.MAX_PLANNING_RECEIPT_BYTES,
          "bounded retained native planning receipt required")
    path = local._receipt_artifact_path(manifest, reference.get("sha256"))
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_uid == os.geteuid()
              and stat.S_IMODE(before.st_mode) == 0o600 and before.st_size == reference["bytes"],
              "retained native planning receipt owner storage changed")
        raw = stream.read(reference["bytes"] + 1)
        after = os.fstat(stream.fileno())
    _need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
          == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
          and len(raw) == reference["bytes"] and hashlib.sha256(raw).hexdigest() == reference["sha256"],
          "retained native planning receipt bytes changed")
    return raw


def _relations(spec):
    return {
        "outputs": [{"ordinal": number, "path": row["path"], "effect": row}
                    for number, row in enumerate(spec["outputs"])],
        "acceptance": [{"ordinal": number, "criterion": row["criterion"], "evidence_policy": row}
                       for number, row in enumerate(spec["acceptance"])],
        "validations": [{"ordinal": number, "argv": row["argv"],
                         "policy": {key: value for key, value in row.items()
                                    if key not in {"argv", "validation_commands", "command"}}}
                        for number, row in enumerate(spec["validations"])],
    }


def _native_population(server, source, admission, selected):
    """Verify every signed native contract and genuine completed prerequisite."""
    _need(type(server) is QuackStateServer and server.lifecycle is ServerLifecycle.READY
          and type(source) is TypedDatabaseTaskSource and source.execution_route_policy is not None,
          "live native owner and route-sealed task source required")
    verified = local.verify_local_benchmark_admission(admission, initial=True)
    _need(verified["manifest"]["schema"] in local.INVENTORY_MANIFEST_SCHEMAS
          and server.identity.repository_id == verified["manifest"]["repository_cid"],
          "native owner differs from signed inventory admission")
    expected = {task.task_cid: task for task in verified["graph"].tasks}
    _need(1 <= len(expected) <= MAX_TASKS and selected in expected,
          "bounded full original task population and exact selected task required")
    page = source.list_tasks(limit=17)
    _need(not page.next_cursor and len(page.tasks) == len(expected)
          and {row.task_cid for row in page.tasks} == set(expected),
          "native owner task population differs from full signed inventory admission")
    ready = source.ready_tasks(limit=17)
    _need(not ready.next_cursor and {row.task_cid for row in ready.tasks} == {selected},
          "exactly the selected inventory-bound task must be native-ready")
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
                  and row["identity"].get("local_contract_cid") == content_identity(envelope)
                  and row["task_alias"] == wanted["task_key"]
                  and row["dependencies"] == wanted["dependencies"]
                  and row["goal_cid"] == expected[row["task_cid"]].goal_cid
                  and row["plan_cid"] == wanted["plan_id"],
                  "native task differs from complete signed pending contract")
            _need(_same({key: row[key] for key in ("outputs", "acceptance", "validations")},
                        _relations(wanted["task_spec"])),
                  "native outputs, acceptance or validations differ from signed task scope")
            plan = intent.get_plan(row["plan_cid"])
            reference = plan["body"].get("local_planning_receipt_ref", {}) if plan else {}
            retained = local.load_local_planning_receipt(reference, manifest=admission["manifest"])
            _need(_same(retained, admission["receipt"]),
                  "native plan lacks the complete immutable inventory planning receipt")
            if row["task_cid"] == selected:
                _need(row["status"] == "ready", "selected inventory-bound task is not ready")
            else:
                _need(row["status"] == "completed" and row["revision"] >= 2,
                      "all other original tasks require actual native completion")
                receipts = server._connection.execute(
                    "SELECT receipt_cid, task_cid, goal_cid, attempt_id, claim_cid, fencing_token, "
                    "completed_at, validation_run_id, evidence_digest, body_json "
                    "FROM completion_receipts WHERE task_cid = ? ORDER BY completed_at, receipt_cid LIMIT 129",
                    [row["task_cid"]]).fetchall()
                _need(len(receipts) <= 128, "bounded completion receipt population required")
                completion_rows[row["task_cid"]] = [[value[position] for position in range(10)] for value in receipts]
                binding, reasons = IntentRepository._current_task_completion_binding(row, receipts)
                _need(binding is not None and not reasons
                      and not local.local_completion_missing(server._connection, row["task_cid"],
                                                            row["body"], row["revision"] - 1),
                      "completed prerequisite lacks current native public-check evidence")
                completion[row["task_cid"]] = binding
            population.append(row)
        authority_rows = _physical_native_authority(server._connection)
    return {"tasks": population, "completed_prerequisites": completion, "completion_rows": completion_rows,
            "authority_rows": authority_rows,
            "selected_task_cids": [selected], "owner_identity": server.identity.to_dict(),
            "execution_route_policy": source.execution_route_policy.to_dict()}


def _public_artifact_bytes(descriptor, repository):
    """Guarded bytes only: no signing, planning, source or model callbacks."""
    from . import router_public_instruction as public
    from .doctor_candidate_runner import _directory
    fields = {"artifact", "sha256", "task_cid", "context_cid", "manifest_cid", "source_path",
              "source_sha256", "source_bytes", "completion_authority", "scope_expansion_authority",
              "inventory_context_cid", "codebase_inventory_context_cid"}
    successor_fields = fields | {"codebase_successor_context_cid", "successor_selection_cid", "source_delta_cid"}
    _need(type(descriptor) is dict and set(descriptor) in (fields, successor_fields),
          "exact prepared inventory public instruction descriptor required")
    maximum = public.MAX_SUCCESSOR_BYTES if set(descriptor) == successor_fields else public.MAX_INVENTORY_BYTES
    path = Path(descriptor["artifact"])
    repository = Path(repository)
    _need(path.is_absolute() and path.parent == repository / public.DIRECTORY,
          "public instruction belongs to another repository")
    parent = _directory(path.parent)
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=parent)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                  and before.st_uid == os.geteuid() and stat.S_IMODE(before.st_mode) == 0o444
                  and 0 < before.st_size <= maximum,
                  "public instruction must be immutable owner-authored worker-readable storage")
            raw = stream.read(maximum + 1)
            after = os.fstat(stream.fileno())
    finally:
        os.close(parent)
    _need(len(raw) == before.st_size and hashlib.sha256(raw).hexdigest() == descriptor["sha256"]
          and all(getattr(before, key) == getattr(after, key)
                  for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")),
          "public instruction bytes changed")
    return raw


def _public_artifact(descriptor, admission, population):
    from . import router_public_instruction as public
    from .doctor_candidate_runner import _unique
    repository = Path(admission["manifest"]["payload"]["repository"])
    raw = _public_artifact_bytes(descriptor, repository)
    path = Path(descriptor["artifact"])
    payload = json.loads(raw, object_pairs_hook=_unique)
    successor = admission["manifest"]["payload"]["schema"] == local.SUCCESSOR_MANIFEST_SCHEMA
    _need(type(payload) is dict and set(payload) == (public.SUCCESSOR_FIELDS if successor else public.INVENTORY_FIELDS)
          and payload["schema"] == (public.SUCCESSOR_SCHEMA if successor else public.INVENTORY_SCHEMA)
          and _same(payload["manifest"], admission["manifest"])
          and _same(payload["inventory_plan_admission"], {"graph": admission["graph"], "receipt": admission["receipt"]})
          and content_identity({key: value for key, value in payload.items() if key != "context_cid"}) == payload["context_cid"],
          "public instruction differs from the full signed inventory task admission")
    manifest = public._selection(payload)
    context = public._inventory_context(payload)
    expected = {key: payload[key] for key in (
        "task_cid", "context_cid", "manifest_cid", "source_path", "source_sha256", "source_bytes",
        "completion_authority", "scope_expansion_authority")}
    expected.update(artifact=str(path), sha256=hashlib.sha256(raw).hexdigest(),
        inventory_context_cid=context["context_cid"],
        codebase_inventory_context_cid=context["codebase_inventory_context_cid"])
    if successor:
        expected.update(codebase_successor_context_cid=context["codebase_successor_context_cid"],
            successor_selection_cid=context["codebase_successor"]["selection_cid"],
            source_delta_cid=context["codebase_successor"]["source_delta_cid"])
    selected = population["selected_task_cids"][0]
    task = next(row for row in population["tasks"] if row["task_cid"] == selected)
    _need(_same(descriptor, expected) and payload["task_cid"] == selected
          and payload["task_id"] == task["task_alias"]
          and all(payload[key] is False for key in
                  ("completion_authority", "publication_authority", "scope_expansion_authority")),
          "public instruction selection or authority differs from the selected native task")
    public._current_sources(repository, manifest["sources"])
    return expected


def _candidate(candidate, admission, population):
    _need(type(candidate) is dict and set(candidate) == {"public_instruction", "argv", "task_revision"},
          "exact owner-selected inventory worker candidate required")
    descriptor = _public_artifact(candidate["public_instruction"], admission, population)
    selected = population["selected_task_cids"][0]
    task = next(row for row in population["tasks"] if row["task_cid"] == selected)
    argv = candidate["argv"]
    _need(type(candidate["task_revision"]) is int and candidate["task_revision"] == task["revision"]
          and type(argv) is list and 7 <= len(argv) <= 128
          and all(type(arg) is str and arg and len(arg) <= 8192 and "\0" not in arg for arg in argv)
          and argv[0] == "/opt/ipfs-supervisor/bin/owner-worker",
          "exact ready revision and fixed isolated inventory worker invocation required")
    for flag, value in (("--public-instruction-artifact", descriptor["artifact"]),
                        ("--public-instruction-sha256", descriptor["sha256"]),
                        ("--public-instruction-task-cid", selected)):
        _need(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv)
              and argv[argv.index(flag) + 1] == value,
              "inventory worker command differs from its exact public instruction binding")
    _need(not any(arg.startswith(("--finite-repository", "--doctor-", "--preflight")) for arg in argv)
          and ("--purpose" not in argv or (argv.count("--purpose") == 1
               and argv.index("--purpose") + 1 < len(argv) and argv[argv.index("--purpose") + 1] == "coding")),
          "inventory worker requires its explicit coding profile")
    allowed = {"--model", "--reasoning-effort", "--timeout", "--max-output-tokens", "--purpose",
               "--semantic-repository", "--public-instruction-artifact", "--public-instruction-sha256",
               "--public-instruction-task-cid"}
    flags = argv[1::2]
    _need(len(argv) % 2 == 1 and len(flags) == len(set(flags)) and set(flags) <= allowed,
          "closed literal inventory worker options required")
    options = dict(zip(flags, argv[2::2]))
    for flag, maximum in (("--timeout", 300), ("--max-output-tokens", 16384)):
        if flag in options:
            value = options[flag]
            _need(value.isascii() and value.isdecimal() and str(int(value)) == value
                  and 1 <= int(value) <= maximum, "bounded exact worker invocation setting required")
    _need(options.get("--reasoning-effort", "high") in {"low", "medium", "high", "xhigh", "max"}
          and options.get("--semantic-repository", admission["manifest"]["payload"]["repository"])
              == admission["manifest"]["payload"]["repository"],
          "inventory worker route or repository binding differs")
    return {"public_instruction": descriptor, "argv": argv, "task_revision": task["revision"],
            "implementation_command": shlex.join(argv)}


@dataclass(frozen=True)
class _Owner:
    root: object
    completion: object
    index: object
    repository: Path
    registry: object
    evidence_record: object
    verification_catalog: object
    cancel_event: object
    timeout_seconds: float
    memory_mb: int
    successor_selection: object = None


@dataclass
class _ReceivingOperation:
    seal: object
    purpose: str
    runtime: object
    owner: object
    owner_fields: tuple
    native_fields: tuple
    pid: int
    thread_id: int
    admission_bytes: bytes
    candidate_bytes: bytes
    material_bytes: bytes
    observed: dict
    close_current: object
    closed: bool = False
    exited: bool = False
    native_admission: object = None
    native_admission_bytes: bytes | None = None


@dataclass
class _SpawnPreparation:
    seal: object
    runtime: object
    operation: object
    owner: object
    owner_fields: tuple
    native_fields: tuple
    pid: int
    thread_id: int
    admission_bytes: bytes
    candidate_bytes: bytes
    material_bytes: bytes
    launcher_bytes: bytes
    output: object
    output_bytes: bytes
    native_admission: object
    native_admission_bytes: bytes | None
    observed: object = None
    observed_bytes: bytes | None = None
    consumed: bool = False


class _InventoryStartRefused(Exception):
    """Abort a private receiving walk while preserving a failed control result."""

    def __init__(self, result):
        self.result = result


@dataclass(frozen=True)
class _ConstructorCleanupCustody:
    runtime: object
    server: object
    source: object
    gateway: object
    initial_handler: object
    initial_binding: object
    resources: tuple
    owner_pid: int


class FrozenInventoryExecutionScope:
    """Private live owner lease; a serialized envelope cannot grant launch."""

    def __init__(self, seal, *, owner, admission, candidate, server, source, lease, output):
        _need(seal is _SEAL, "inventory execution scope must be created by native reservation")
        self._seal, self._owner, self._admission, self._candidate_input = seal, owner, admission, candidate
        self._server, self._source, self._lease, self._output = server, source, lease, output
        self._runtime, self._spawned, self._cleaned, self._released = None, False, False, False
        self._receiving = None
        self._receiving_entering = False
        self._reservation_receiving_used = False
        self._reservation_preparation_used = False
        self._spawn_preparation = None
        self._constructor_custody = None
        self._constructor_cleanup_pending = False
        self._constructor_released_run_lease = None
        self._native_admission, self._native_admission_bytes = None, None
        self._native_admission_ref = None
        self._renew_stop, self._renew_failed = threading.Event(), threading.Event()
        self._renew_thread = threading.Thread(target=self._renew, daemon=True,
                                              name="inventory-execution-lease-" + lease.lease_id[:8])
        self._renew_thread.start()

    def _renew(self):
        interval = max(0.01, min(20.0, self._lease._scheduler.config.lease_ttl_seconds / 3))
        while not self._renew_stop.wait(interval):
            try:
                if not self._lease.renew():
                    self._renew_failed.set()
                    return
            except Exception:
                self._renew_failed.set()

    @property
    def parent_lease(self):
        return self._lease

    @property
    def selected_task_cids(self):
        return tuple(self.to_dict()["payload"]["native_population"]["selected_task_cids"])

    @property
    def material_binding(self):
        return self.to_dict()

    @property
    def worker_launcher_binding(self):
        return dict(self._launcher)

    def to_dict(self):
        return json.loads(self._material)

    def _active(self, *, allow_cancelled=False):
        with _LOCK:
            _need(self._seal is _SEAL and _ACTIVE.get(self._lease.lease_id) is self
                  and not self._released and not self._lease.released,
                  "an exact active native inventory execution scope is required")
        if not allow_cancelled:
            _need(not self._renew_failed.is_set() and not self._lease.cancelled
                  and (self._owner.cancel_event is None or not self._owner.cancel_event.is_set()),
                  "native inventory execution reservation cancelled or expired")

    def _current_inventory(self):
        owner = self._owner
        if owner.successor_selection is not None:
            from .codebase_successor_dispatch_admission import verify_current_successor_admission
            return verify_current_successor_admission(selection=owner.successor_selection,
                root=owner.root, completion=owner.completion, index=owner.index, repository=owner.repository,
                registry=owner.registry, admission=self._admission, evidence_record=owner.evidence_record,
                verification_catalog=owner.verification_catalog, parent_lease=self._lease,
                cancel_event=owner.cancel_event, timeout_seconds=owner.timeout_seconds, memory_mb=owner.memory_mb)
        return inventory.verify_current_inventory_admission(root=owner.root, completion=owner.completion,
            index=owner.index, repository=owner.repository, registry=owner.registry, admission=self._admission,
            evidence_record=owner.evidence_record, verification_catalog=owner.verification_catalog,
            parent_lease=self._lease, cancel_event=owner.cancel_event,
            timeout_seconds=owner.timeout_seconds, memory_mb=owner.memory_mb)

    def _uses_paired_receiving(self):
        from ipfs_datasets_py.logic.software_contracts.codebase_inventory_resume import CodebaseScanResumeRoot
        return type(self._owner.root) is CodebaseScanResumeRoot and self._owner.root.to_dict()["optimized"] is True

    def _owner_fields(self):
        owner = self._owner
        timeout, memory = owner.timeout_seconds, owner.memory_mb
        _need(type(timeout) in {int, float} and 0 < timeout <= 600 and math.isfinite(timeout)
              and type(memory) is int and 1024 <= memory <= 4096,
              "bounded exact inventory receiving owner settings required")
        # The canonical identity wire is float-free. Keep the exact timeout
        # type and binary float value so 120 and 120.0 cannot alias a guard.
        timeout_identity = ({"type": "float", "hex": timeout.hex()} if type(timeout) is float
                            else {"type": "int", "value": timeout})
        settings = [timeout_identity, {"type": "int", "value": memory}]
        if owner.successor_selection is not None:
            from ipfs_datasets_py.logic.software_contracts.codebase_inventory_successor_model import CodebaseSuccessorScanRecord
            from ipfs_datasets_py.logic.software_contracts.codebase_inventory_receiving import _record_guard
            _need(type(owner.successor_selection) is CodebaseSuccessorScanRecord,
                  "exact selected successor owner record required")
            settings.append({"successor_selection_object": id(owner.successor_selection),
                "record_guard": list(_record_guard(owner.successor_selection, 256 * 1024))})
        return (owner.root, owner.completion, owner.index, owner.repository, owner.registry,
                owner.evidence_record, owner.verification_catalog, owner.cancel_event,
                canonical_dag_json_bytes(settings))

    def _native_owner_fields(self):
        identity, route = self._server.identity, self._source.execution_route_policy
        return (self._server, self._source, self._server._connection, self._server._lock,
                self._lease, identity, route,
                canonical_dag_json_bytes([identity.to_dict(), route.to_dict()]))

    def _operation_guard(self, operation, *, closing=False):
        self._active()
        self._native_admission_unchanged()
        _need(type(operation) is _ReceivingOperation and operation.seal is _OPERATION_SEAL
              and self._receiving is operation and not operation.exited
              and operation.pid == os.getpid() and operation.thread_id == threading.get_ident()
              and operation.owner is self._owner,
              "an exact private one-operation receiving scope is required")
        fields = self._owner_fields()
        native = self._native_owner_fields()
        _need(all(left is right for left, right in zip(fields[:8], operation.owner_fields[:8]))
              and fields[8] == operation.owner_fields[8]
              and all(left is right for left, right in zip(native[:7], operation.native_fields[:7]))
              and native[7] == operation.native_fields[7]
              and canonical_dag_json_bytes(self._admission) == operation.admission_bytes
              and canonical_dag_json_bytes(self._candidate_input) == operation.candidate_bytes
              and self._material == operation.material_bytes,
              "inventory receiving inputs changed during owner callbacks")
        if operation.native_admission is not None:
            _need(operation.native_admission is self._native_admission
                  and operation.native_admission_bytes == self._native_admission_bytes
                  and _native_admission_projection(operation.native_admission)[1]
                  == operation.native_admission_bytes,
                  "native typed admission changed during owner callbacks")
        _need(not operation.closed or closing, "inventory receiving operation already closed")
        if operation.purpose == "reserve":
            _need(operation.runtime is None and self._runtime is None
                  and self._native_admission is None and self._spawn_preparation is None,
                  "inventory reservation cannot bind or prepare a runtime")
        elif operation.purpose == "create":
            _need(operation.runtime is self._runtime, "inventory receiving constructor identity changed")
        else:
            _need(operation.purpose == "start" and operation.runtime is self._runtime,
                  "inventory receiving launch identity changed")

    @contextmanager
    def _receiving_operation(self, *, purpose, runtime=None):
        """Full entry before callbacks, one native close before commit or Popen."""
        self._active()
        _need((purpose in {"reserve", "create"} and runtime is None and self._runtime is None)
              or (purpose == "start" and runtime is self._runtime and runtime is not None),
              "exact unlaunched inventory runtime required")
        with _LOCK:
            _need(purpose in {"reserve", "create", "start"} and self._receiving is None
                  and not self._receiving_entering and not self._spawned,
                  "inventory receiving operations cannot be nested or reused")
            if purpose == "reserve":
                _need(not self._reservation_receiving_used,
                      "inventory reservation receiving operation already used")
                self._reservation_receiving_used = True
            self._receiving_entering = True
        try:
            owner = self._owner
            fields = self._owner_fields()
            native = self._native_owner_fields()
            admission_bytes = canonical_dag_json_bytes(self._admission)
            candidate_bytes = canonical_dag_json_bytes(self._candidate_input)
            material_bytes = self._material
            native_admission, native_admission_bytes = self._native_admission, self._native_admission_bytes
            operation_scope = inventory._current_inventory_admission_operation
            successor_options = {}
            if owner.successor_selection is not None:
                from .codebase_successor_dispatch_admission import _current_successor_admission_operation
                operation_scope = _current_successor_admission_operation
                successor_options = {"selection": owner.successor_selection}
            with operation_scope(root=owner.root, completion=owner.completion,
                    index=owner.index, repository=owner.repository, registry=owner.registry,
                    admission=self._admission, evidence_record=owner.evidence_record,
                    verification_catalog=owner.verification_catalog, parent_lease=self._lease,
                    cancel_event=owner.cancel_event, timeout_seconds=owner.timeout_seconds,
                    memory_mb=owner.memory_mb, **successor_options) as (observed, close):
                operation = _ReceivingOperation(_OPERATION_SEAL, purpose, runtime, owner, fields, native,
                    os.getpid(), threading.get_ident(), admission_bytes, candidate_bytes, material_bytes,
                    observed, close,
                    native_admission=native_admission if purpose == "start" else None,
                    native_admission_bytes=native_admission_bytes if purpose == "start" else None)
                self._receiving = operation
                try:
                    self._operation_guard(operation)
                    _need(_same(observed, self.to_dict()["payload"]["current_inventory"]),
                          "current scan, model or inventory admission changed")
                    yield operation
                    self._operation_guard(operation, closing=True)
                    _need(operation.closed, "inventory receiving operation has no successful closing fence")
                finally:
                    if (self._spawn_preparation is not None
                            and self._spawn_preparation.operation is operation):
                        self._spawn_preparation.consumed = True
                        self._spawn_preparation = None
                    operation.exited = True
                    self._receiving = None
        finally:
            self._receiving_entering = False

    def _close_receiving_operation(self, operation):
        self._operation_guard(operation)
        _need(_same(operation.close_current(), operation.observed),
              "inventory state changed during owner callbacks")
        self._operation_guard(operation)
        operation.closed = True

    def _reservation_artifact_current(self, artifact):
        """Direct byte custody after callbacks; no signing or owner hooks."""
        _need(artifact == self._output / "execution-scope.json",
              "inventory reservation artifact belongs to another output")
        parent = os.open(self._output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fd = os.open(artifact.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=parent)
            with os.fdopen(fd, "rb") as stream:
                before = os.fstat(stream.fileno())
                _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                      and before.st_uid == os.geteuid() and stat.S_IMODE(before.st_mode) == 0o400
                      and before.st_size == len(self._material),
                      "inventory reservation artifact storage changed")
                digest, amount = hashlib.sha256(), 0
                while True:
                    self._active()
                    chunk = stream.read(min(65536, len(self._material) + 1 - amount))
                    if not chunk:
                        break
                    amount += len(chunk)
                    _need(amount <= len(self._material), "inventory reservation artifact grew")
                    digest.update(chunk)
                after = os.fstat(stream.fileno())
                path_after = os.stat(artifact.name, dir_fd=parent, follow_symlinks=False)
                fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode", "st_nlink")
                _need(amount == len(self._material) and digest.digest() == hashlib.sha256(self._material).digest()
                      and all(getattr(before, name) == getattr(after, name) == getattr(path_after, name)
                              for name in fields), "inventory reservation artifact bytes changed")
        finally:
            os.close(parent)

    def _prepare_reservation_artifact(self, artifact):
        """Keep one fresh receiver across both callbacks and the scope write.

        The initial public verification remains unchanged. Optimized roots
        replace four later full receiving pairs with one fresh entry and a
        mandatory close after every reservation callback and the fsync. No
        receiver or freshness result survives into construction or START.
        """
        self._active()
        _need(not self._reservation_preparation_used and self._runtime is None and not self._spawned
              and artifact == self._output / "execution-scope.json",
              "inventory reservation artifact preparation cannot be reused")
        self._reservation_preparation_used = True

        def write_artifact():
            with artifact.open("xb") as stream:
                stream.write(self._material)
                stream.flush()
                os.fchmod(stream.fileno(), 0o400)
                os.fsync(stream.fileno())

        if not self._uses_paired_receiving():
            self.require_prelaunch_current()
            write_artifact()
            self.require_prelaunch_current()
            return

        output, output_bytes = self._output, canonical_dag_json_bytes(str(self._output))
        with self._receiving_operation(purpose="reserve") as operation:
            self.require_prelaunch_current()
            write_artifact()
            self.require_prelaunch_current()
            with self._server._lock:
                self._close_receiving_operation(operation)
                _need(_pins() == self.to_dict()["payload"]["implementation"],
                      "selected inventory launch producer bytes changed")
                self._detached_fence()
                self._reservation_artifact_current(artifact)
                self._operation_guard(operation, closing=True)
        # Resource closure cannot replace the bound inputs or output. These
        # direct observations invoke no model/source or signed graph callback.
        self._active()
        fields, native = self._owner_fields(), self._native_owner_fields()
        _need(operation.seal is _OPERATION_SEAL and operation.purpose == "reserve"
              and operation.pid == os.getpid() and operation.thread_id == threading.get_ident()
              and operation.owner is self._owner and operation.runtime is None
              and operation.closed and operation.exited and self._receiving is None
              and not self._receiving_entering and self._reservation_receiving_used
              and self._reservation_preparation_used
              and self._runtime is None and self._spawn_preparation is None and not self._spawned
              and operation.native_admission is None and operation.native_admission_bytes is None
              and self._native_admission is None and self._native_admission_bytes is None
              and self._native_admission_ref is None
              and all(left is right for left, right in zip(fields[:8], operation.owner_fields[:8]))
              and fields[8] == operation.owner_fields[8]
              and all(left is right for left, right in zip(native[:7], operation.native_fields[:7]))
              and native[7] == operation.native_fields[7]
              and canonical_dag_json_bytes(self._admission) == operation.admission_bytes
              and canonical_dag_json_bytes(self._candidate_input) == operation.candidate_bytes
              and self._material == operation.material_bytes and self._output is output
              and canonical_dag_json_bytes(str(self._output)) == output_bytes,
              "inventory reservation inputs changed during resource closure")
        _need(_pins() == self.to_dict()["payload"]["implementation"],
              "selected inventory launch producer bytes changed")
        self._detached_fence()
        self._reservation_artifact_current(artifact)
        self._active()

    def _finish_creation_receiving(self, runtime):
        operation = self._receiving
        _need(operation is not None and operation.purpose == "create" and runtime is self._runtime,
              "exact private inventory construction operation required")
        # Typed task queries use a separate gateway thread which needs this
        # same owner lock. Finish every callback before acquiring it.
        self.require_prelaunch_current()
        with self._server._lock:
            self._close_receiving_operation(operation)
            self._detached_fence()
            self._active()

    def _detached_fence(self):
        bound = self.to_dict()["payload"]
        self._native_admission_unchanged()
        if self._receiving is not None:
            self._operation_guard(self._receiving, closing=self._receiving.closed)
        with self._server._lock:
            if self._receiving is not None:
                _need(_same(self._server.identity.to_dict(), bound["native_population"]["owner_identity"])
                      and _same(self._source.execution_route_policy.to_dict(),
                                bound["native_population"]["execution_route_policy"]),
                      "native inventory owner or execution route changed after callbacks")
            actual = _physical_native(self._server._connection)
        _need(_same(actual, {key: bound["native_population"][key]
                            for key in ("tasks", "completion_rows", "authority_rows")}),
              "full native rows or ready revision changed after inventory callbacks")
        referenced_plans = {row["plan_cid"] for row in bound["native_population"]["tasks"]}
        for row in bound["native_population"]["authority_rows"]["plans"]:
            if row[0] in referenced_plans:
                reference = json.loads(row[7]).get("local_planning_receipt_ref", {})
                _planning_receipt_bytes(reference, self._admission["manifest"]["payload"])
        _public_artifact_bytes(bound["candidate"]["public_instruction"], self._owner.repository)
        if hasattr(self, "_launcher"):
            from .candidate_execution import _root_file
            _need(_root_file(Path(self._launcher["path"])) == self._launcher["sha256"],
                  "inventory worker launcher bytes changed")

    def require_prelaunch_current(self):
        self._active()
        self._native_admission_unchanged()
        _need(not self._spawned, "inventory execution scope permits one native process launch")
        bound = self.to_dict()["payload"]
        operation = self._receiving
        if operation is None:
            observed = self._current_inventory()
        else:
            self._operation_guard(operation)
            observed = operation.observed
        _need(_same(observed, bound["current_inventory"]), "current scan, model or inventory admission changed")
        self._prelaunch_callbacks(bound)
        # A signer/public-artifact/owner callback can change private scan state.
        # Receiving verification after these calls observes it again.
        if operation is None:
            _need(_same(self._current_inventory(), observed), "inventory state changed during prelaunch callbacks")
        else:
            self._operation_guard(operation)
        self._native_admission_unchanged()
        self._detached_fence()
        self._active()

    def _prelaunch_callbacks(self, bound):
        _need(not self._server._lock._is_owned(),
              "inventory prelaunch callbacks must run before acquiring the native owner lock")
        _need(_pins() == bound["implementation"], "selected inventory launch producer bytes changed")
        population = _native_population(self._server, self._source, self._admission, self.selected_task_cids[0])
        _need(_same(population, bound["native_population"]), "ready revision or full native task population changed")
        candidate = _candidate(self._candidate_input, self._admission, population)
        _need(_same(candidate, bound["candidate"]), "closed inventory worker invocation changed")
        profile = local.verify_local_benchmark_admission(self._admission, initial=True)["profile"]
        _need(_same(local._verify_signature(self.to_dict(), profile), bound),
              "inventory execution scope signature or material changed")

    def _spawn_preparation_guard(self, prepared, *, closing=False):
        self._active()
        self._native_admission_unchanged()
        _need(type(prepared) is _SpawnPreparation and prepared.seal is _OPERATION_SEAL
              and self._spawn_preparation is prepared and not prepared.consumed
              and prepared.pid == os.getpid() and prepared.thread_id == threading.get_ident()
              and prepared.runtime is self._runtime and prepared.owner is self._owner
              and prepared.operation is self._receiving and not self._spawned,
              "exact private one-use inventory spawn preparation required")
        fields, native = self._owner_fields(), self._native_owner_fields()
        _need(all(left is right for left, right in zip(fields[:8], prepared.owner_fields[:8]))
              and fields[8] == prepared.owner_fields[8]
              and all(left is right for left, right in zip(native[:7], prepared.native_fields[:7]))
              and native[7] == prepared.native_fields[7]
              and canonical_dag_json_bytes(self._admission) == prepared.admission_bytes
              and canonical_dag_json_bytes(self._candidate_input) == prepared.candidate_bytes
              and self._material == prepared.material_bytes
              and canonical_dag_json_bytes(self._launcher) == prepared.launcher_bytes
              and self._output is prepared.output
              and canonical_dag_json_bytes(str(self._output)) == prepared.output_bytes
              and self._native_admission is prepared.native_admission
              and self._native_admission_bytes == prepared.native_admission_bytes
              and (prepared.observed is None or canonical_dag_json_bytes(prepared.observed)
                   == prepared.observed_bytes),
              "inventory spawn inputs changed during unlocked callbacks")
        _need(prepared.runtime.server is self._server and prepared.runtime.source is self._source
              and _same(prepared.runtime.admission, self._admission)
              and _same(prepared.runtime.manifest.get("inventory_execution_scope"), self.to_dict())
              and _same(prepared.runtime.manifest.get("inventory_worker_launcher"), self._launcher),
              "inventory runtime binding changed during unlocked callbacks")
        if prepared.operation is not None:
            self._operation_guard(prepared.operation, closing=closing)
            _need(prepared.operation.purpose == "start", "inventory constructor cannot authorize Popen")

    def prepare_spawn_fence(self, runtime):
        """Replay callbacks unlocked; retain no source validity across a launch."""
        self._active()
        _need(runtime is self._runtime and not self._spawned and self._spawn_preparation is None,
              "exact unlaunched inventory runtime and unused preparation required")
        _need(not self._server._lock._is_owned(),
              "inventory spawn preparation must precede the native owner lock")
        operation = self._receiving
        prepared = _SpawnPreparation(_OPERATION_SEAL, runtime, operation, self._owner,
            self._owner_fields(), self._native_owner_fields(), os.getpid(), threading.get_ident(),
            canonical_dag_json_bytes(self._admission), canonical_dag_json_bytes(self._candidate_input),
            self._material, canonical_dag_json_bytes(self._launcher), self._output,
            canonical_dag_json_bytes(str(self._output)), self._native_admission, self._native_admission_bytes)
        self._spawn_preparation = prepared
        try:
            self._spawn_preparation_guard(prepared)
            bound = self.to_dict()["payload"]
            prepared.observed = self._current_inventory() if operation is None else operation.observed
            prepared.observed_bytes = canonical_dag_json_bytes(prepared.observed)
            self._spawn_preparation_guard(prepared)
            _need(_same(prepared.observed, bound["current_inventory"]),
                  "current scan, model or inventory admission changed")
            self._prelaunch_callbacks(bound)
            self._spawn_preparation_guard(prepared)
            self._detached_fence()
            self._spawn_preparation_guard(prepared)
        except BaseException:
            prepared.consumed = True
            self._spawn_preparation = None
            raise

    def require_spawn_fence(self, runtime):
        """Native receiving gate under the task-owner lock before parent Popen."""
        self._active()
        _need(runtime is self._runtime and not self._spawned, "exact unlaunched inventory runtime required")
        prepared = self._spawn_preparation
        try:
            self._spawn_preparation_guard(prepared)
            _need(self._server._lock._is_owned(), "inventory final spawn fence requires the native owner lock")
            if self._receiving is not None:
                self._close_receiving_operation(self._receiving)
                # Closing the native receiving scope intentionally consumes it.
                _need(prepared.operation is self._receiving, "inventory receiving operation changed")
            else:
                _need(_same(self._current_inventory(), prepared.observed),
                      "inventory state changed during prelaunch callbacks")
            self._spawn_preparation_guard(prepared, closing=True)
            self._native_admission_unchanged()
            _need(_pins() == self.to_dict()["payload"]["implementation"],
                  "selected inventory launch producer bytes changed")
            self._detached_fence()
            self._spawn_preparation_guard(prepared, closing=True)
        finally:
            if type(prepared) is _SpawnPreparation:
                prepared.consumed = True
            if self._spawn_preparation is prepared:
                self._spawn_preparation = None

    def require_launch(self, *, admission, server, source, implement, candidate_runner,
                       context_bundle, refresh_context_on_completion, implementation_command):
        self._active()
        projected, typed_bytes = _native_admission_projection(admission)
        _need(server is self._server and source is self._source and _same(projected, self._admission)
              and implement is True and candidate_runner is not None and context_bundle is None
              and refresh_context_on_completion is False,
              "inventory launch requires its exact owner, full admission and isolated runner")
        _need(self._native_admission is None, "native typed launch admission can bind only once")
        self._native_admission, self._native_admission_bytes = admission, typed_bytes
        self._native_admission_ref = admission
        if self._receiving is not None:
            operation = self._receiving
            self._operation_guard(operation)
            _need(operation.purpose == "create" and operation.native_admission is None,
                  "native typed admission can bind only once during construction")
            operation.native_admission, operation.native_admission_bytes = admission, typed_bytes
            self._operation_guard(operation)
        candidate = self.to_dict()["payload"]["candidate"]
        _need(type(implementation_command) is str and implementation_command == candidate["implementation_command"],
              "inventory scope requires the exact immutable worker command")
        from .candidate_execution import _root_file, verify_candidate_runner
        verify_candidate_runner(candidate_runner)
        path = Path(candidate["argv"][0])
        self._launcher = {"path": str(path), "sha256": _root_file(path)}
        self.require_prelaunch_current()

    def _native_admission_unchanged(self):
        if self._native_admission_ref is None:
            _need(self._native_admission is None and self._native_admission_bytes is None,
                  "native typed launch admission binding changed")
        else:
            _need(self._native_admission is self._native_admission_ref
                  and _native_admission_projection(self._native_admission)[1] == self._native_admission_bytes,
                  "native typed launch admission changed during owner callbacks")

    def bind_runtime(self, runtime):
        self._active()
        from ..entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
        _need(type(runtime) is AdmittedBenchmarkRuntime and runtime.inventory_execution_scope is self
              and runtime.finite_execution_scope is None and self._runtime is None
              and runtime.server is self._server and runtime.source is self._source
              and _same(runtime.admission, self._admission),
              "inventory scope can bind exactly one matching native runtime")
        self._runtime = runtime
        if self._receiving is not None:
            _need(self._receiving.purpose == "create" and self._receiving.runtime is None,
                  "inventory receiving runtime can bind only once")
            self._receiving.runtime = runtime

    def require_runtime(self, runtime, *, before_spawn=False, stopping=False):
        self._active(allow_cancelled=stopping)
        _need(runtime is self._runtime and _same(runtime.manifest.get("inventory_execution_scope"), self.to_dict()),
              "signed launch is not bound to this exact active inventory scope")
        _need(_same(runtime.manifest.get("inventory_worker_launcher"), self._launcher),
              "inventory launcher differs from signed native launch")
        if not stopping:
            from .candidate_execution import _root_file
            _need(_root_file(Path(self._launcher["path"])) == self._launcher["sha256"],
                  "inventory worker launcher bytes changed")
        if before_spawn:
            self.require_prelaunch_current()

    def note_spawned(self, runtime):
        _need(runtime is self._runtime and not self._spawned, "inventory execution scope already launched")
        # Record cleanup duty immediately after Popen, before a cancelled
        # receiving operation or lease can raise during the post-birth check.
        self._spawned = True
        if self._receiving is not None:
            self._operation_guard(self._receiving, closing=True)
            _need(self._receiving.purpose == "start" and self._receiving.closed,
                  "native Popen requires its consumed closing receiving fence")

    def finish_runtime(self, runtime):
        """Changed source, task state or cancellation must not prevent STOP."""
        self.require_runtime(runtime, stopping=True)
        _need(runtime._context_refresh_stopped(), "inventory release requires native STOP and isolated UID cleanup")
        self._cleaned = True

    def require_close(self, runtime):
        _need(self._seal is _SEAL and runtime is self._runtime, "inventory close requires its exact native runtime")
        if self._spawned:
            _need(self._cleaned and runtime._context_refresh_stopped(),
                      "inventory runtime close requires successful STOP and isolated UID cleanup")

    def _remember_constructor_cleanup(self, runtime):
        """Capture newly allocated resources before constructor callbacks."""
        from ..entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
        _need(type(runtime) is AdmittedBenchmarkRuntime
              and runtime.inventory_execution_scope is self and runtime.finite_execution_scope is None
              and runtime.server is self._server and runtime.source is self._source,
              "inventory constructor custody requires its original native runtime")
        custody = self._constructor_custody
        gateway = self._server._command_gateway
        if custody is None:
            with self._server._lock:
                with gateway._grants_lock:
                    custody = _ConstructorCleanupCustody(runtime, self._server, self._source, gateway,
                        gateway._local_task_validation_handler, gateway._local_task_validation_binding, (), os.getpid())
        _need(custody.runtime is runtime and custody.server is self._server
              and custody.source is self._source and custody.gateway is gateway
              and custody.owner_pid == os.getpid(), "inventory constructor custody changed")
        resources = dict(custody.resources)
        for name in ("_listener", "_bootstrap_stop", "coordinator", "lease", "profile",
                     "process", "orchestrator", "service", "_bootstrap_thread"):
            if hasattr(runtime, name):
                value = getattr(runtime, name)
                _need(name not in resources or resources[name] is value,
                      "inventory constructor replaced an owned resource")
                resources[name] = value
        self._constructor_custody = _ConstructorCleanupCustody(runtime, custody.server, custody.source,
            custody.gateway, custody.initial_handler, custody.initial_binding,
            tuple(resources.items()), custody.owner_pid)

    def _retain_constructor_cleanup(self, runtime):
        custody = self._constructor_custody
        if custody is None:
            _need(not any(hasattr(runtime, name) for name in
                          ("_listener", "coordinator", "lease", "_bootstrap_thread")),
                  "failed inventory constructor allocated unrecorded cleanup custody")
            return
        _need(custody is not None and custody.runtime is runtime,
              "failed inventory constructor lacks original cleanup custody")
        self._constructor_cleanup_pending = True

    def _close_constructor_run_coordinator(self, runtime):
        """Dispose the original run fence and retain its native release result."""
        from ..merge.database_coordination import FencedLease, LeaseState
        custody = self._constructor_custody
        _need(custody is not None and custody.runtime is runtime,
              "inventory run lease needs its original constructor custody")
        resources = dict(custody.resources)
        coordinator, lease = resources.get("coordinator"), resources.get("lease")
        _need(coordinator is not None and runtime.coordinator is coordinator
              and getattr(runtime, "lease", None) is lease,
              "inventory constructor replaced its native run coordinator or fence")
        if lease is not None:
            if self._constructor_released_run_lease is None:
                released = coordinator.release(lease, expected_fencing_token=lease.fencing_token,
                                               expected_fence_epoch=lease.fence_epoch)
                _need(type(lease) is FencedLease and type(released) is FencedLease
                      and released == replace(lease, state=LeaseState.RELEASED, revision=lease.revision + 1),
                      "inventory constructor lacks its exact native run-lease release")
                self._constructor_released_run_lease = released
        coordinator.close()

    def _require_constructor_cleanup(self):
        custody = self._constructor_custody
        _need(custody is not None and custody.owner_pid == os.getpid()
              and custody.server is self._server and custody.source is self._source
              and (self._runtime is None or custody.runtime is self._runtime)
              and custody.runtime.server is custody.server
              and custody.runtime.source is custody.source
              and custody.server._command_gateway is custody.gateway,
              "failed inventory constructor lost original cleanup custody")
        runtime, resources = custody.runtime, dict(custody.resources)
        _need(all(getattr(runtime, name, None) is value for name, value in custody.resources),
              "failed inventory constructor replaced an owned cleanup resource")
        listener = resources.get("_listener")
        thread = resources.get("_bootstrap_thread")
        coordinator = resources.get("coordinator")
        _need((listener is None or listener.fileno() == -1)
              and (thread is None or (resources["_bootstrap_stop"].is_set() and not thread.is_alive()))
              and (coordinator is None or not coordinator.is_open)
              and not getattr(runtime, "_children", ()),
              "failed inventory constructor still owns transport, thread or run-lease custody")
        if "lease" in resources:
            from ..merge.database_coordination import FencedLease, LeaseState
            lease, released = resources["lease"], self._constructor_released_run_lease
            _need(type(lease) is FencedLease and type(released) is FencedLease
                  and released == replace(lease, state=LeaseState.RELEASED, revision=lease.revision + 1),
                  "failed inventory constructor lacks its original native run-lease release")
        with custody.server._lock:
            with custody.gateway._grants_lock:
                _need(custody.gateway._local_task_validation_handler is custody.initial_handler
                      and custody.gateway._local_task_validation_binding is custody.initial_binding,
                      "failed inventory constructor still owns an exact completion binding")
        if "service" in resources:
            _need(self._cleaned and runtime._context_refresh_stopped(),
                  "failed inventory constructor lacks genuine native STOP cleanup")

    def _finish(self):
        if self._constructor_cleanup_pending:
            # The runtime's diagnostic marker cannot authorize resource release.
            # Original native objects must actually be stopped and disposed.
            self._require_constructor_cleanup()
        if (self._runtime is not None and hasattr(self._runtime, "process")
                and self._runtime.process.snapshot(self._runtime.profile).members):
            self._spawned = True
        if self._runtime is not None and self._spawned and not self._cleaned:
            self._runtime.stop()
            self.finish_runtime(self._runtime)
        if self._runtime is not None and hasattr(self._runtime, "process"):
            _need(not self._runtime.process.snapshot(self._runtime.profile).members,
                  "inventory execution envelope still owns a live native process")
            if self._spawned:
                _need(self._cleaned and self._runtime._context_refresh_stopped(),
                      "inventory execution envelope lacks isolated worker cleanup")
        self._renew_stop.set()
        self._lease.release()
        self._released = True
        with _LOCK:
            _ACTIVE.pop(self._lease.lease_id, None)
            _RETAINED_UNSAFE_SCOPES.pop(self._lease.lease_id, None)


@contextmanager
def reserve_inventory_execution(*, root, completion, index, repository, registry, admission, candidate,
        server, source, output, scheduler=None, parent_lease=None, evidence_record=None,
        verification_catalog=None, cancel_event=None, timeout_seconds=120, memory_mb=1024,
        cpu_slots=4, execution_memory_mb=4096, child_process_slots=8, admission_timeout_seconds=30,
        successor_selection=None):
    """Hold genuine native admission from preparation through STOP/UID cleanup.

    This reservation is accounting, not a kernel resource-limit claim. Failed
    cleanup retains the live lease and heartbeat for a safe shutdown retry.
    """
    _need(type(cpu_slots) is int and 4 <= cpu_slots <= 16
          and type(execution_memory_mb) is int and 4096 <= execution_memory_mb <= 16384
          and type(child_process_slots) is int and 8 <= child_process_slots <= 64
          and type(admission_timeout_seconds) in {int, float} and math.isfinite(admission_timeout_seconds)
          and 0 < admission_timeout_seconds <= 90
          and type(timeout_seconds) in {int, float} and math.isfinite(timeout_seconds)
          and 0 < timeout_seconds <= 600 and type(memory_mb) is int and 1024 <= memory_mb <= 4096,
          "bounded full-process native inventory execution envelope required")
    if scheduler is None and type(parent_lease) is ResourceLease:
        scheduler = parent_lease._scheduler
    _need(type(scheduler) is GlobalResourceScheduler and scheduler.config.proof_safety_enabled
          and scheduler.config.proof_resource_sampler is collect_proof_host_resources,
          "native shared scheduler with genuine host proof safety required")
    repository = Path(repository).absolute()
    _need(repository.resolve(strict=True) == repository and repository.is_dir(), "exact native repository required")
    output = Path(output).absolute()
    _need(not output.exists() and output.resolve() == output and not output.is_relative_to(repository),
          "fresh external inventory execution evidence directory required")
    admission, candidate = local._plain(admission), local._plain(candidate)
    is_successor = admission.get("manifest", {}).get("payload", {}).get("schema") == local.SUCCESSOR_MANIFEST_SCHEMA
    _need(is_successor == (successor_selection is not None),
          "signed successor execution requires its exact selected successor owner")
    _need(type(candidate) is dict and type(candidate.get("public_instruction")) is dict,
          "owner-prepared inventory worker descriptor required")
    selected = candidate["public_instruction"].get("task_cid")
    lease = scheduler.acquire(ResourceLane.ORCHESTRATION, cpu_slots=cpu_slots, memory_mb=execution_memory_mb,
        child_process_slots=child_process_slots, parent_lease=parent_lease,
        timeout=admission_timeout_seconds, cancel_event=cancel_event,
        request_id="inventory-native-execution-" + uuid.uuid4().hex)
    owner = _Owner(root, completion, index, repository, registry, evidence_record, verification_catalog,
                   lease.combined_cancellation_signal(cancel_event), timeout_seconds, memory_mb, successor_selection)
    scope = FrozenInventoryExecutionScope(_SEAL, owner=owner, admission=admission, candidate=candidate,
                                          server=server, source=source, lease=lease, output=output)
    with _LOCK:
        _ACTIVE[lease.lease_id] = scope
    try:
        output.mkdir(mode=0o700)
        current = scope._current_inventory()
        population = _native_population(server, source, admission, selected)
        binding = _candidate(candidate, admission, population)
        payload = {"schema": SCHEMA, "profile": PROFILE, "admission_cid": content_identity(admission),
            "current_inventory": current, "native_population": population, "candidate": binding,
            "head": current["head"], "lease": {
                "lease_id": lease.lease_id, "lane": lease.lane, "cpu_slots": lease.cpu_slots,
                "memory_mb": lease.memory_mb, "child_process_slots": lease.child_process_slots,
                "owner_pid": lease.owner_pid, "parent_lease_id": lease.parent_lease_id},
            "implementation": _pins(), "task_population_preserved": True,
            "current_facts": [], "removed_task_cids": [], "inventory_features_are_advisory": True,
            "future_claim_and_fence": "existing_native_typed_owner",
            "resource_scope": "native_admission_accounting_until_STOP_and_isolated_cleanup",
            "freshness_scope": "sequential_owner_scan_model_source_and_native_task_observations",
            "worker_freshness_scope": "signed_public_inputs_and_allocated_worktree_without_private_registry",
            "task_omission_authority": False, "completion_authority": False, "proof_authority": False,
            "publication_authority": False, "production_activation": False}
        envelope = local._signed(payload, admission["manifest"]["payload"])
        scope._material = canonical_dag_json_bytes(envelope)
        artifact = output / "execution-scope.json"
        scope._prepare_reservation_artifact(artifact)
        yield scope
    finally:
        try:
            scope._finish()
        except BaseException as error:
            with _LOCK:
                _RETAINED_UNSAFE_SCOPES[lease.lease_id] = scope
            raise InventoryExecutionError(
                "inventory execution cleanup unproven; resource lease retained for safe STOP/cleanup retry") from error


__all__ = ["PROFILE", "SCHEMA", "InventoryExecutionError", "FrozenInventoryExecutionScope",
           "reserve_inventory_execution"]
