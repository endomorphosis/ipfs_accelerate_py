"""Private operation/launch controls, with explicit receiving protocol doubles.

The resource lease and canonical control contracts are real. Scan receivers,
signer callbacks and native population adapters below are controlled protocol
fixtures; these cases claim no native source/model freshness or process launch.
"""
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from enum import Enum
import base64
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as resume
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceLane, ResourceSchedulerConfig,
)
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import Operation, get_operation_catalog
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.entrypoints.isolated_benchmark_runtime import IsolatedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.runtime import codebase_inventory_execution as execution
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from test.api.test_codebase_inventory_evidence_worker_context import native_refs


@pytest.fixture(scope="module")
def retained_typed_admission():
    """Actual closed10 signed bytes, rebuilt by the native typed contracts.

    This host qualification fixture reads retained files only. It opens no
    source/model owner and verifies public signatures without metadata writes.
    """
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "closed10-admission.json"
    raw = fixture.read_bytes()
    assert len(raw) <= 4 * 1024 * 1024
    assert hashlib.sha256(raw).hexdigest() == "f6f641aea1b89fdbbe8fcf2ce0f025140b6a0236d3ac071828494839a48d1176"
    native = json.loads(raw)
    typed = deepcopy(native)
    from ipfs_accelerate_py.agent_supervisor.planning.formal_planning_contracts import EvidenceRequirement
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import PromptGoalGraph
    from ipfs_accelerate_py.agent_supervisor.control import profile_authority
    typed["graph"] = PromptGoalGraph.from_dict(native["graph"]).to_dict()
    for row in typed["receipt"]["payload"]["pending_requirements"]:
        row.update(EvidenceRequirement.from_dict(row).to_dict())
    assert local._plain(typed) == native
    for name in ("manifest", "receipt"):
        envelope = typed[name]
        # Use the native signed-byte encoding and DID decoder directly: the
        # retained verifier predates the optional metadata-mirror switch.
        signature = base64.b64decode(envelope["binding"]["signature"].encode("ascii"), validate=True)
        profile_authority.ed25519_public_key_from_did(envelope["binding"]["identity"]).verify(
            signature, profile_authority._canonical(envelope["payload"]))
    return typed


@pytest.fixture
def operation_case(native_refs, tmp_path, monkeypatch):
    scan = native_refs["scan"]
    root = resume.CodebaseScanResumeRoot.from_dict(scan["root_cid"], scan["root_record"])
    completion = resume.CodebaseScanResumeCompletion.from_dict(scan["completion_cid"], scan["completion_record"])
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path / "resources.json", lane_reservations={}, auto_renew_leases=False))
    cancel = threading.Event()
    lease = scheduler.acquire(ResourceLane.ORCHESTRATION, cpu_slots=1, memory_mb=64,
                              child_process_slots=1, timeout=3)
    owner = execution._Owner(root, completion, object(), tmp_path, object(), None, None, cancel, 120, 1024)
    identity, route = {"owner": "original"}, {"route": "original"}
    server = SimpleNamespace(_lock=threading.RLock(), _connection=object(),
                             identity=SimpleNamespace(to_dict=lambda: dict(identity)))
    source = SimpleNamespace(execution_route_policy=SimpleNamespace(to_dict=lambda: dict(route)))
    admission, candidate_input = {"fixture": "signed admission"}, {"fixture": "worker input"}
    scope = execution.FrozenInventoryExecutionScope(execution._SEAL, owner=owner, admission=admission,
        candidate=candidate_input, server=server, source=source, lease=lease, output=tmp_path)
    observed = {"root_cid": root.artifact_cid, "completion_cid": completion.artifact_cid}
    population = {"selected_task_cids": ["selected"], "tasks": [], "completion_rows": {},
                  "owner_identity": identity, "execution_route_policy": route}
    candidate = {"public_instruction": {}, "argv": [], "task_revision": 3}
    payload = {"implementation": {}, "native_population": population,
               "current_inventory": observed, "candidate": candidate}
    scope._material = json.dumps({"payload": payload}).encode()
    scope._launcher = {"path": "/opt/ipfs-supervisor/bin/owner-worker", "sha256": "ab" * 32}
    events = []
    state = {"source": "original", "entry_callback": None, "closing_error": None}
    @contextmanager
    def paired(**arguments):
        assert arguments["parent_lease"] is lease and not lease.released
        events.append("entry")
        if state["entry_callback"]:
            state["entry_callback"]()
        def close():
            assert scope._server._lock._is_owned(), "closing must run under the native owner lock"
            events.append("close")
            assert state["source"] == "original"
            if state["closing_error"]:
                raise state["closing_error"]
            return observed
        try:
            yield observed, close
        except BaseException:
            events.append("abort")
            raise
        else:
            events.append("exit")
    monkeypatch.setattr(execution.inventory, "_current_inventory_admission_operation", paired)
    monkeypatch.setattr(scope, "_current_inventory", lambda: events.append("reference") or observed)
    monkeypatch.setattr(scope, "_detached_fence", lambda: events.append("detached"))
    monkeypatch.setattr(execution, "_pins", lambda: {})
    monkeypatch.setattr(execution, "_native_population", lambda *args: events.append("population") or population)
    monkeypatch.setattr(execution, "_candidate", lambda *args: events.append("candidate") or candidate)
    monkeypatch.setattr(local, "verify_local_benchmark_admission",
                        lambda *args, **kwargs: {"profile": object()})
    from ipfs_accelerate_py.agent_supervisor.entrypoints import admitted_benchmark_runtime
    monkeypatch.setattr(admitted_benchmark_runtime, "verify_local_benchmark_admission",
                        lambda *args, **kwargs: {"profile": object()})
    monkeypatch.setattr(local, "_verify_signature",
                        lambda envelope, profile: events.append("signature") or envelope["payload"])
    from ipfs_accelerate_py.agent_supervisor.runtime import candidate_execution
    monkeypatch.setattr(candidate_execution, "_root_file", lambda path: "ab" * 32)
    with execution._LOCK:
        execution._ACTIVE[lease.lease_id] = scope
    case = SimpleNamespace(scope=scope, owner=owner, scheduler=scheduler, cancel=cancel,
                           observed=observed, population=population, events=events, state=state)
    try:
        yield case
    finally:
        scope._runtime = None
        scope._finish()
        scope._renew_thread.join(timeout=2)
        assert scheduler.snapshot()["active_lease_count"] == 0


def runtime_case(case):
    runtime = AdmittedBenchmarkRuntime()
    runtime.inventory_execution_scope, runtime.finite_execution_scope = case.scope, None
    runtime.server, runtime.source, runtime.admission = case.scope._server, case.scope._source, case.scope._admission
    runtime.manifest = {"inventory_execution_scope": case.scope.to_dict(),
                        "inventory_worker_launcher": case.scope.worker_launcher_binding}
    return runtime


def test_one_start_entry_preserves_callbacks_and_closes_before_popen(operation_case, monkeypatch):
    case, events = operation_case, operation_case.events
    runtime = runtime_case(case)
    case.scope._runtime = runtime
    result = SimpleNamespace(succeeded=True)
    def backend(self):
        events.append("dispatch")
        case.scope.require_runtime(runtime, before_spawn=True)
        case.scope.prepare_spawn_fence(runtime)
        with case.scope._server._lock:
            case.scope.require_spawn_fence(runtime)
            events.append("popen")
            case.scope.note_spawned(runtime)
        # A real worker can now edit source; normal manager exit must not
        # rerun source validation after that independently authorized effect.
        case.state["source"] = "worker edit"
        return result
    monkeypatch.setattr(IsolatedBenchmarkRuntime, "start", backend)
    assert runtime.start() is result
    assert events.count("entry") == events.count("close") == 1 and "reference" not in events
    assert events.index("entry") < events.index("dispatch") < events.index("close") < events.index("popen")
    assert events[events.index("close") + 1:events.index("popen")] == ["detached"]
    assert events.count("population") == events.count("candidate") == events.count("signature") == 3
    assert case.scope._receiving is None and case.scope._spawned


def test_failed_control_result_abandons_entry_and_retains_its_original_result(operation_case, monkeypatch):
    case = operation_case
    runtime = runtime_case(case)
    case.scope._runtime = runtime
    result = SimpleNamespace(succeeded=False)
    monkeypatch.setattr(IsolatedBenchmarkRuntime, "start", lambda self: result)
    assert runtime.start() is result
    assert case.events.count("entry") == case.events.count("abort") == 1
    assert "close" not in case.events and not case.scope._spawned
    assert case.scope._receiving is None and not case.scope._receiving_entering


def test_start_cannot_claim_success_without_consuming_the_closing_gate(operation_case, monkeypatch):
    case = operation_case
    runtime = runtime_case(case)
    case.scope._runtime = runtime
    monkeypatch.setattr(IsolatedBenchmarkRuntime, "start", lambda self: SimpleNamespace(succeeded=True))
    with pytest.raises(execution.InventoryExecutionError, match="without its closing"):
        runtime.start()
    assert not case.scope._spawned and case.scope._receiving is None


def test_create_uses_one_entry_and_closes_after_all_constructor_callbacks(operation_case, monkeypatch, tmp_path):
    case = operation_case
    runtime = runtime_case(case)
    def construct(cls, directory, **arguments):
        case.events.append("construct")
        case.scope.require_prelaunch_current()
        case.scope.bind_runtime(runtime)
        case.scope.require_runtime(runtime, before_spawn=True)
        case.events.append("constructor-finished")
        return runtime
    monkeypatch.setattr(AdmittedBenchmarkRuntime, "_create", classmethod(construct))
    actual = AdmittedBenchmarkRuntime.create(tmp_path / "launch", admission=runtime.admission,
        server=runtime.server, source=runtime.source, inventory_execution_scope=case.scope)
    assert actual is runtime
    assert case.events.count("entry") == case.events.count("close") == 1 and "reference" not in case.events
    assert case.events.index("entry") < case.events.index("construct")
    assert case.events.index("constructor-finished") < case.events.index("close")
    assert case.events.count("population") == case.events.count("signature") == 3
    assert case.scope._receiving is None


def test_native_float_timeout_survives_create_then_start(operation_case, monkeypatch, tmp_path):
    case = operation_case
    object.__setattr__(case.owner, "timeout_seconds", 120.0)
    identity = json.loads(case.scope._owner_fields()[8])
    assert identity == [{"type": "float", "hex": float(120).hex()},
                        {"type": "int", "value": 1024}]
    runtime = runtime_case(case)
    def construct(cls, directory, **arguments):
        case.scope.require_prelaunch_current()
        case.scope.bind_runtime(runtime)
        case.scope.require_runtime(runtime, before_spawn=True)
        return runtime
    monkeypatch.setattr(AdmittedBenchmarkRuntime, "_create", classmethod(construct))
    assert AdmittedBenchmarkRuntime.create(tmp_path / "launch", admission=runtime.admission,
        server=runtime.server, source=runtime.source, inventory_execution_scope=case.scope) is runtime
    assert case.events.count("entry") == case.events.count("close") == 1
    result = SimpleNamespace(succeeded=True)
    def backend(self):
        case.events.append("dispatch")
        case.scope.require_runtime(runtime, before_spawn=True)
        case.scope.prepare_spawn_fence(runtime)
        with case.scope._server._lock:
            case.scope.require_spawn_fence(runtime)
            case.events.append("popen")
            case.scope.note_spawned(runtime)
        case.state["source"] = "worker edit"
        return result
    monkeypatch.setattr(IsolatedBenchmarkRuntime, "start", backend)
    assert runtime.start() is result
    assert case.events.count("entry") == case.events.count("close") == 2
    assert case.events.count("population") == case.events.count("signature") == 6
    assert "reference" not in case.events and case.scope._spawned
    assert case.scope._receiving is None and type(case.owner.timeout_seconds) is float


@pytest.mark.parametrize("setting,value", [
    ("timeout_seconds", 120), ("timeout_seconds", True), ("timeout_seconds", float("nan")),
    ("timeout_seconds", float("inf")), ("timeout_seconds", -float("inf")),
    ("timeout_seconds", 120.00000000000001), ("memory_mb", True), ("memory_mb", 1024.0),
])
def test_float_owner_settings_cannot_change_type_value_or_finiteness_before_close(
        operation_case, setting, value):
    case = operation_case
    object.__setattr__(case.owner, "timeout_seconds", 120.0)
    runtime = runtime_case(case)
    case.scope._runtime = runtime
    with pytest.raises(execution.InventoryExecutionError, match="inputs changed|owner settings"):
        with case.scope._receiving_operation(purpose="start", runtime=runtime) as operation:
            object.__setattr__(case.owner, setting, value)
            with case.scope._server._lock:
                case.scope._close_receiving_operation(operation)
    assert "close" not in case.events and not case.scope._spawned
    assert case.scope._receiving is None and not case.scope._receiving_entering


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -float("inf"), 0.0, 601.0])
def test_invalid_float_owner_budget_refuses_before_receiving_entry(operation_case, value):
    case = operation_case
    object.__setattr__(case.owner, "timeout_seconds", value)
    runtime = runtime_case(case)
    case.scope._runtime = runtime
    with pytest.raises(execution.InventoryExecutionError, match="owner settings"):
        with case.scope._receiving_operation(purpose="start", runtime=runtime):
            pytest.fail("invalid owner budget admitted")
    assert "entry" not in case.events and not case.scope._spawned
    assert case.scope._receiving is None and not case.scope._receiving_entering


def _typed_launch_case(case, admission, monkeypatch, *, optimized):
    body = case.owner.root.to_dict()
    body["optimized"] = optimized
    object.__setattr__(case.owner, "root", resume.CodebaseScanResumeRoot.from_dict(cid_for_structured(body), body))
    object.__setattr__(case.owner, "timeout_seconds", 120.0)
    case.scope._admission = local._plain(admission)
    payload = case.scope.to_dict()["payload"]
    payload["candidate"]["implementation_command"] = "reviewed typed admission fixture"
    payload["candidate"]["argv"] = ["/opt/ipfs-supervisor/bin/owner-worker"]
    case.scope._material = json.dumps({"payload": payload}).encode()
    monkeypatch.setattr(execution, "_candidate", lambda *args: payload["candidate"])
    from ipfs_accelerate_py.agent_supervisor.runtime import candidate_execution
    monkeypatch.setattr(candidate_execution, "verify_candidate_runner", lambda _: {})
    runtime = runtime_case(case)
    def construct(cls, directory, **arguments):
        case.scope.require_launch(admission=arguments["admission"], server=runtime.server,
            source=runtime.source, implement=True, candidate_runner={"authored": "fixture"},
            context_bundle=None, refresh_context_on_completion=False,
            implementation_command="reviewed typed admission fixture")
        case.scope.bind_runtime(runtime)
        case.scope.require_runtime(runtime, before_spawn=True)
        return runtime
    monkeypatch.setattr(AdmittedBenchmarkRuntime, "_create", classmethod(construct))
    return runtime


def test_complete_retained_native_admission_projects_only_eight_exact_enum_leaves(retained_typed_admission):
    projected, guarded = execution._native_admission_projection(retained_typed_admission)
    identity = json.loads(guarded)
    assert projected == local._plain(retained_typed_admission) == identity["admission"]
    ledger = identity["enum_ledger"]
    assert len(ledger) == 8
    assert {row["path"][-1] for row in ledger} == {"kind", "minimum_code_assurance"}
    assert len({tuple(row["path"]) for row in ledger}) == 8
    _, plain_guarded = execution._native_admission_projection(projected)
    assert guarded != plain_guarded and json.loads(plain_guarded)["enum_ledger"] == []
    with pytest.raises(ValueError, match="EvidenceRequirementKind"):
        execution.canonical_dag_json_bytes(retained_typed_admission)


@pytest.mark.parametrize("optimized", [True, False])
def test_retained_native_typed_admission_binds_create_then_start(
        operation_case, retained_typed_admission, optimized, monkeypatch, tmp_path):
    case, admission = operation_case, deepcopy(retained_typed_admission)
    runtime = _typed_launch_case(case, admission, monkeypatch, optimized=optimized)
    assert AdmittedBenchmarkRuntime.create(tmp_path / "launch", admission=admission,
        server=runtime.server, source=runtime.source, inventory_execution_scope=case.scope) is runtime
    assert case.scope._native_admission_ref is case.scope._native_admission is admission
    result = SimpleNamespace(succeeded=True)
    def backend(self):
        case.scope.require_runtime(runtime, before_spawn=True)
        case.scope.prepare_spawn_fence(runtime)
        with case.scope._server._lock:
            case.scope.require_spawn_fence(runtime)
            case.scope.note_spawned(runtime)
        return result
    monkeypatch.setattr(IsolatedBenchmarkRuntime, "start", backend)
    assert runtime.start() is result and case.scope._spawned
    assert case.events.count("entry") == (2 if optimized else 0)
    assert case.events.count("reference") == (0 if optimized else 10)


@pytest.mark.parametrize("optimized", [True, False])
@pytest.mark.parametrize("change", ["string", "dictionary", "member", "foreign_enum", "wrong_class",
    "string_subclass", "tuple", "boolean_alias", "foreign_reference", "cleared_binding", "forged_member",
    "hostile_type"])
def test_native_typed_admission_mutation_during_launcher_callback_refuses_before_close(
        operation_case, retained_typed_admission, optimized, change, monkeypatch, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.planning.formal_planning_contracts import EvidenceRequirementKind
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import AssuranceLevel
    from ipfs_accelerate_py.agent_supervisor.runtime import candidate_execution
    class ForeignKind(str, Enum):
        TEST = "test"
    class StringSubclass(str):
        pass
    case, admission = operation_case, deepcopy(retained_typed_admission)
    runtime = _typed_launch_case(case, admission, monkeypatch, optimized=optimized)
    def mutate(_):
        row = admission["receipt"]["payload"]["pending_requirements"][0]
        if change == "string": row["kind"] = row["kind"].value
        elif change == "dictionary":
            row["kind"] = {"class": type(row["kind"]).__module__ + "." + type(row["kind"]).__qualname__,
                           "name": row["kind"].name, "value": row["kind"].value}
        elif change == "member": row["kind"] = EvidenceRequirementKind.REVIEW
        elif change == "foreign_enum": row["kind"] = ForeignKind.TEST
        elif change == "wrong_class": row["kind"] = AssuranceLevel.CANDIDATE
        elif change == "string_subclass": row["kind"] = StringSubclass(row["kind"].value)
        elif change == "tuple": row["subject_ids"] = tuple(row["subject_ids"])
        elif change == "boolean_alias": row["required"] = 1
        elif change == "foreign_reference": case.scope._native_admission = deepcopy(admission)
        elif change == "cleared_binding": case.scope._native_admission = None
        elif change == "hostile_type":
            class EqualityMimic(type):
                def __hash__(cls): return hash(AssuranceLevel)
                def __eq__(cls, other): return other is AssuranceLevel
                @property
                def __members__(cls): return {"CANDIDATE": mimic}
            class AuthoredHostObject(metaclass=EqualityMimic):
                value, name = "candidate", "CANDIDATE"
            mimic = AuthoredHostObject()
            assert type(mimic) == AssuranceLevel and type(mimic) is not AssuranceLevel
            row["minimum_code_assurance"] = mimic
        else:
            forged = str.__new__(EvidenceRequirementKind, "test")
            forged._name_, forged._value_ = "TEST", "test"
            row["kind"] = forged
        return {}
    monkeypatch.setattr(candidate_execution, "verify_candidate_runner", mutate)
    with pytest.raises(execution.InventoryExecutionError, match="native.*admission|native.*enum"):
        AdmittedBenchmarkRuntime.create(tmp_path / "launch", admission=admission,
            server=runtime.server, source=runtime.source, inventory_execution_scope=case.scope)
    assert "close" not in case.events and not case.scope._spawned
    assert case.scope._receiving is None and not case.scope._receiving_entering


def test_start_entry_cannot_substitute_equivalent_typed_admission_reference(
        operation_case, retained_typed_admission, monkeypatch, tmp_path):
    case, admission = operation_case, deepcopy(retained_typed_admission)
    runtime = _typed_launch_case(case, admission, monkeypatch, optimized=True)
    AdmittedBenchmarkRuntime.create(tmp_path / "launch", admission=admission,
        server=runtime.server, source=runtime.source, inventory_execution_scope=case.scope)
    original_closes = case.events.count("close")
    case.state["entry_callback"] = lambda: setattr(case.scope, "_native_admission", deepcopy(admission))
    monkeypatch.setattr(IsolatedBenchmarkRuntime, "start", lambda self: pytest.fail("substitution reached dispatch"))
    with pytest.raises(execution.InventoryExecutionError, match="typed launch admission changed"):
        runtime.start()
    assert case.events.count("close") == original_closes and not case.scope._spawned
    assert case.scope._receiving is None and not case.scope._receiving_entering


def test_constructor_closing_refusal_uses_native_stop_before_completion_close(operation_case, monkeypatch, tmp_path):
    case = operation_case
    runtime = runtime_case(case)
    def construct(cls, directory, **arguments):
        case.scope.bind_runtime(runtime)
        return runtime
    monkeypatch.setattr(AdmittedBenchmarkRuntime, "_create", classmethod(construct))
    def stopped():
        case.events.append("stop")
        return SimpleNamespace(succeeded=True)
    monkeypatch.setattr(runtime, "stop", stopped)
    monkeypatch.setattr(runtime, "_context_refresh_stopped", lambda: True)
    monkeypatch.setattr(runtime, "close", lambda: case.events.append("retire"))
    case.state["closing_error"] = ValueError("fresh model changed")
    with pytest.raises(ValueError, match="fresh model"):
        AdmittedBenchmarkRuntime.create(tmp_path / "launch", admission=runtime.admission,
            server=runtime.server, source=runtime.source, inventory_execution_scope=case.scope)
    assert case.events[-3:] == ["stop", "retire", "abort"]
    assert not case.scope._spawned and case.scope._receiving is None


@pytest.mark.parametrize("field", ["owner", "root", "timeout", "admission", "candidate", "material",
    "server", "source", "connection", "lock", "lease", "native_identity", "native_route"])
def test_mutated_operation_inputs_refuse_before_native_closing(operation_case, field):
    case = operation_case
    runtime = runtime_case(case)
    case.scope._runtime = runtime
    original_lease, replacement = case.scope.parent_lease, None
    try:
        with pytest.raises(execution.InventoryExecutionError, match="inputs changed|one-operation|exact active"):
            with case.scope._receiving_operation(purpose="start", runtime=runtime) as operation:
                if field == "owner": case.scope._owner = execution._Owner(**vars(case.owner))
                elif field == "root": object.__setattr__(case.owner, "root", object())
                elif field == "timeout": object.__setattr__(case.owner, "timeout_seconds", 121)
                elif field == "admission": case.scope._admission["new"] = True
                elif field == "candidate": case.scope._candidate_input["new"] = True
                elif field == "material": case.scope._material += b" "
                elif field == "server": case.scope._server = SimpleNamespace(**vars(case.scope._server))
                elif field == "source": case.scope._source = SimpleNamespace(**vars(case.scope._source))
                elif field == "connection": case.scope._server._connection = object()
                elif field == "lock": case.scope._server._lock = threading.RLock()
                elif field == "lease":
                    replacement = case.scheduler.acquire(ResourceLane.ORCHESTRATION, cpu_slots=1,
                        memory_mb=64, child_process_slots=1, timeout=3)
                    case.scope._lease = replacement
                elif field == "native_identity":
                    case.scope._server.identity = SimpleNamespace(to_dict=lambda: {"owner": "original"})
                else:
                    case.scope._source.execution_route_policy = SimpleNamespace(to_dict=lambda: {"route": "original"})
                case.scope._close_receiving_operation(operation)
    finally:
        case.scope._lease = original_lease
        if replacement is not None:
            replacement.release()
    assert "close" not in case.events and case.scope._receiving is None


def test_operation_rejects_nesting_during_native_entry(operation_case):
    case = operation_case
    runtime = runtime_case(case)
    case.scope._runtime = runtime
    def nested():
        with pytest.raises(execution.InventoryExecutionError, match="nested"):
            with case.scope._receiving_operation(purpose="start", runtime=runtime):
                pytest.fail("nested native entry admitted")
    case.state["entry_callback"] = nested
    with case.scope._receiving_operation(purpose="start", runtime=runtime) as operation:
        with case.scope._server._lock:
            case.scope._close_receiving_operation(operation)
    assert case.events.count("entry") == case.events.count("close") == 1


def test_private_closer_cannot_cross_threads_or_be_reused(operation_case):
    case = operation_case
    runtime = runtime_case(case)
    case.scope._runtime = runtime
    errors = []
    with case.scope._receiving_operation(purpose="start", runtime=runtime) as operation:
        def cross_thread():
            try: case.scope._close_receiving_operation(operation)
            except execution.InventoryExecutionError as error: errors.append(str(error))
        child = threading.Thread(target=cross_thread)
        child.start(); child.join(timeout=3)
        assert not child.is_alive() and errors and "one-operation" in errors[0]
        with case.scope._server._lock:
            case.scope._close_receiving_operation(operation)
        with pytest.raises(execution.InventoryExecutionError, match="already closed"):
            case.scope._close_receiving_operation(operation)
    with pytest.raises(execution.InventoryExecutionError, match="one-operation"):
        case.scope._close_receiving_operation(operation)


def test_cancellation_after_popen_cannot_erase_native_cleanup_duty(operation_case):
    case = operation_case
    runtime = runtime_case(case)
    case.scope._runtime = runtime
    with pytest.raises(execution.InventoryExecutionError, match="cancelled"):
        with case.scope._receiving_operation(purpose="start", runtime=runtime) as operation:
            with case.scope._server._lock:
                case.scope._close_receiving_operation(operation)
            case.cancel.set()
            case.scope.note_spawned(runtime)
    assert case.scope._spawned and not case.scope.parent_lease.released
    assert case.scope._receiving is None


def test_native_population_callbacks_remain_mandatory_inside_operation(operation_case, monkeypatch):
    case = operation_case
    runtime = runtime_case(case)
    case.scope._runtime = runtime
    with pytest.raises(execution.InventoryExecutionError, match="full native task population"):
        with case.scope._receiving_operation(purpose="start", runtime=runtime):
            monkeypatch.setattr(execution, "_native_population", lambda *args: {"selected_task_cids": ["foreign"]})
            case.scope.require_prelaunch_current()
    assert "close" not in case.events and not case.scope._spawned


def test_opt_out_retains_all_original_start_receiving_repeats(operation_case, monkeypatch):
    case = operation_case
    body = case.owner.root.to_dict(); body["optimized"] = False
    object.__setattr__(case.owner, "root", resume.CodebaseScanResumeRoot.from_dict(cid_for_structured(body), body))
    runtime = runtime_case(case)
    case.scope._runtime = runtime
    result = SimpleNamespace(succeeded=True)
    def backend(self):
        case.scope.require_runtime(runtime, before_spawn=True)
        case.scope.prepare_spawn_fence(runtime)
        with case.scope._server._lock:
            case.scope.require_spawn_fence(runtime)
        return result
    monkeypatch.setattr(IsolatedBenchmarkRuntime, "start", backend)
    assert runtime.start() is result
    assert case.events.count("reference") == 6 and "entry" not in case.events


def test_canonical_start_stop_limits_are_unchanged_and_cannot_be_overridden(tmp_path):
    catalog = get_operation_catalog()
    assert catalog.operation(Operation.START).bounds.timeout_ms == 30_000
    assert catalog.operation(Operation.STOP).bounds.timeout_ms == 30_000
    with pytest.raises(ValueError, match="30000"):
        AdmittedBenchmarkRuntime.create(tmp_path / "launch", admission={}, server=None, source=None,
                                        timeout_ms=300_000)
    assert not (tmp_path / "launch").exists()


@pytest.mark.parametrize("optimized", [True, False])
def test_prepared_spawn_rejects_equal_foreign_native_owner_and_clears_failed_marker(operation_case, optimized):
    case, scope = operation_case, operation_case.scope
    body = case.owner.root.to_dict(); body["optimized"] = optimized
    object.__setattr__(case.owner, "root", resume.CodebaseScanResumeRoot.from_dict(cid_for_structured(body), body))
    runtime = runtime_case(case); scope._runtime = runtime
    original = scope._server
    try:
        with pytest.raises(execution.InventoryExecutionError, match="inputs changed"):
            with (scope._receiving_operation(purpose="start", runtime=runtime) if optimized else nullcontext()):
                scope.prepare_spawn_fence(runtime)
                scope._server = SimpleNamespace(_lock=original._lock, _connection=original._connection,
                                                identity=original.identity)
                with original._lock:
                    scope.require_spawn_fence(runtime)
    finally:
        scope._server = original
    assert scope._spawn_preparation is None and scope._receiving is None and not scope._spawned
    assert "close" not in case.events and not scope.parent_lease.released


def test_spawn_preparation_requires_unlocked_callbacks_same_thread_and_one_use(operation_case):
    case, scope = operation_case, operation_case.scope
    runtime = runtime_case(case); scope._runtime = runtime
    failures = []
    with scope._receiving_operation(purpose="start", runtime=runtime):
        with scope._server._lock:
            with pytest.raises(execution.InventoryExecutionError, match="must precede"):
                scope.prepare_spawn_fence(runtime)
            with pytest.raises(execution.InventoryExecutionError, match="one-use"):
                scope.require_spawn_fence(runtime)
        assert "population" not in case.events
        scope.prepare_spawn_fence(runtime)
        def foreign_thread():
            with scope._server._lock:
                try:
                    scope.require_spawn_fence(runtime)
                except execution.InventoryExecutionError as error:
                    failures.append(str(error))
        child = threading.Thread(target=foreign_thread)
        child.start(); child.join(timeout=2)
        assert not child.is_alive() and failures and "one-use" in failures[0]
        assert scope._spawn_preparation is None and "close" not in case.events
        scope.prepare_spawn_fence(runtime)
        with scope._server._lock:
            scope.require_spawn_fence(runtime)
            with pytest.raises(execution.InventoryExecutionError, match="one-use"):
                scope.require_spawn_fence(runtime)
    assert case.events.count("close") == 1 and case.events.count("population") == 2
    assert scope._spawn_preparation is None and scope._receiving is None


def test_unconsumed_spawn_preparation_cannot_survive_a_start_operation(operation_case):
    case, scope = operation_case, operation_case.scope
    runtime = runtime_case(case); scope._runtime = runtime
    with pytest.raises(execution.InventoryExecutionError, match="no successful closing"):
        with scope._receiving_operation(purpose="start", runtime=runtime):
            scope.prepare_spawn_fence(runtime)
    assert scope._spawn_preparation is None and scope._receiving is None
    with scope._receiving_operation(purpose="start", runtime=runtime) as operation:
        with scope._server._lock:
            with pytest.raises(execution.InventoryExecutionError, match="one-use"):
                scope.require_spawn_fence(runtime)
            scope._close_receiving_operation(operation)
    assert case.events.count("close") == 1 and not scope._spawned


@pytest.mark.parametrize("optimized", [True, False])
def test_genuine_threaded_quack_callbacks_and_locked_authority_mutation_refusals(
        operation_case, optimized, monkeypatch, tmp_path):
    """Real native owner/RPC/SQL; scan and signature adapters remain controlled.

    No worker, proof, or model freshness is claimed. Every callback performs
    authenticated typed RPC against the actual threaded native Quack owner.
    """
    from benchmarks.agent_supervisor.container_coding.native_quack_qualification import (
        _prepare, open_existing_native_owner,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server import InProcessQuackTransport
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
    case, scope = operation_case, operation_case.scope
    body = case.owner.root.to_dict(); body["optimized"] = optimized
    object.__setattr__(case.owner, "root", resume.CodebaseScanResumeRoot.from_dict(cid_for_structured(body), body))
    output = tmp_path / "native-quack"
    checkout, database, task_cid, _tree, _baseline = _prepare(output)
    lifecycle = output / "lifecycle"
    receipts = lifecycle / "local-planning-receipts"
    receipts.mkdir(parents=True, mode=0o700)
    receipt_raw = b'{"fixture":"independently retained signed-byte adapter"}'
    receipt_sha = hashlib.sha256(receipt_raw).hexdigest()
    receipt_path = receipts / (receipt_sha + ".json")
    receipt_path.write_bytes(receipt_raw); receipt_path.chmod(0o600)
    reference = {"sha256": receipt_sha, "bytes": len(receipt_raw)}
    with IntentRepository(database) as intent:
        task = intent.get_task(task_cid)
        plan_cid = cid_for_structured({"fixture": "native owner plan"})
        intent.upsert_plan(plan_cid=plan_cid, goal_cid=task["goal_cid"], plan_alias="LOCK-PLAN",
                           body={"local_planning_receipt_ref": reference})
        with intent._connection(write=True) as connection:
            connection.execute("UPDATE tasks SET plan_cid=? WHERE task_cid=?", [plan_cid, task_cid])
        intent.record_validation_result(task_cid=task_cid, outcome="failed",
            evidence_digest=cid_for_structured({"fixture": "retained failed validation"}),
            body={"fixture": "no completion authority"})
    with open_existing_native_owner(database=database, checkout=checkout, state_dir=output / "owner",
            repository_id=cid_for_structured({"fixture": "native owner"}),
            execution_routes={"NQQ-T001": GROK_CODEX_EXECUTION_MODE}) as session:
        server, source = session.server, session.source
        assert type(server.transport) is InProcessQuackTransport
        scope._server, scope._source = server, source
        scope._admission = {"manifest": {"payload": {"lifecycle_dir": str(lifecycle)}}}
        with server._lock:
            physical = execution._physical_native(server._connection)
        assert len(physical["authority_rows"]["plans"]) == 2
        assert len(physical["authority_rows"]["validation_evidence"]) == 1
        population = {**physical, "selected_task_cids": [task_cid], "completed_prerequisites": {},
            "owner_identity": server.identity.to_dict(),
            "execution_route_policy": source.execution_route_policy.to_dict()}
        payload = scope.to_dict()["payload"]
        payload["native_population"] = population
        scope._material = json.dumps({"payload": payload}).encode()
        rpc_calls = []
        def native_callbacks(owner, typed_source, admission, selected):
            assert owner is server and typed_source is source and selected == task_cid
            assert not server._lock._is_owned(), "typed RPC under the owner lock would deadlock"
            page, ready = source.list_tasks(limit=17), source.ready_tasks(limit=17)
            assert {row.task_cid for row in page.tasks} == {task_cid}
            assert {row.task_cid for row in ready.tasks} == {task_cid}
            rpc_calls.append((threading.get_ident(), page.revision, ready.revision))
            with server._lock:
                current = execution._physical_native(server._connection)
            return {**population, **current}
        monkeypatch.setattr(execution, "_native_population", native_callbacks)
        monkeypatch.setattr(execution, "_public_artifact_bytes", lambda *_args: b"controlled public-byte adapter")
        monkeypatch.setattr(scope, "_detached_fence",
            lambda: execution.FrozenInventoryExecutionScope._detached_fence(scope))
        runtime = runtime_case(case)
        def construct(cls, directory, **arguments):
            scope.require_prelaunch_current()
            scope.bind_runtime(runtime)
            scope.require_runtime(runtime, before_spawn=True)
            return runtime
        monkeypatch.setattr(AdmittedBenchmarkRuntime, "_create", classmethod(construct))
        assert AdmittedBenchmarkRuntime.create(tmp_path / "launch", admission=runtime.admission,
            server=server, source=source, inventory_execution_scope=scope) is runtime
        assert len(rpc_calls) == (3 if optimized else 2)
        state = {"fault": None}
        refusals = []
        probe_threads = []
        result = SimpleNamespace(succeeded=True)
        def backend(self):
            scope.require_runtime(runtime, before_spawn=True)
            scope.prepare_spawn_fence(runtime)
            with server._lock:
                assert server._lock._is_owned()
                fault = state["fault"]
                if fault:
                    server._connection.execute("BEGIN TRANSACTION")
                    try:
                        if fault == "task":
                            server._connection.execute("UPDATE tasks SET revision=revision+1 WHERE task_cid=?", [task_cid])
                        elif fault == "plan":
                            server._connection.execute("UPDATE plans SET body_json='{}' WHERE plan_cid=?", [plan_cid])
                        elif fault == "goal":
                            server._connection.execute("UPDATE goals SET body_json='{}'")
                        elif fault == "validation":
                            server._connection.execute("UPDATE validation_results SET evidence_digest='foreign'")
                        elif fault == "validation_event":
                            server._connection.execute("UPDATE domain_events SET body_json='{}' "
                                                       "WHERE event_type='intent.validation_recorded'")
                        elif fault == "generation":
                            server._connection.execute("UPDATE store_generations SET fence_epoch=fence_epoch+1")
                        elif fault == "metadata":
                            server._connection.execute("UPDATE control_plane_metadata SET value='foreign' WHERE key='schema_fingerprint'")
                        elif fault == "receipt":
                            receipt_path.write_bytes(b"changed retained receipt")
                        scope.require_spawn_fence(runtime)
                        pytest.fail("native authority mutation reached Popen")
                    finally:
                        server._connection.execute("ROLLBACK")
                        receipt_path.write_bytes(receipt_raw)
                    return result
                attempting, acquired = threading.Event(), threading.Event()
                def competitor():
                    attempting.set()
                    with server._lock:
                        acquired.set()
                child = threading.Thread(target=competitor)
                probe_threads.append(child)
                child.start()
                assert attempting.wait(2) and not acquired.is_set()
                scope.require_spawn_fence(runtime)
                assert not acquired.is_set(), "owner lock released before the literal launch point"
                case.events.append("popen")
                scope.note_spawned(runtime)
            child.join(timeout=2)
            assert not child.is_alive() and acquired.is_set()
            return result
        monkeypatch.setattr(IsolatedBenchmarkRuntime, "start", backend)
        for fault in ("task", "plan", "goal", "validation", "validation_event", "generation", "metadata", "receipt"):
            state["fault"] = fault
            before = len(rpc_calls)
            with pytest.raises(execution.InventoryExecutionError, match="native rows|planning receipt"):
                runtime.start()
            assert len(rpc_calls) - before == 3
            assert scope._spawn_preparation is None and not scope._spawned and "popen" not in case.events
            refusals.append(fault)
        state["fault"] = None
        before, reference_before = len(rpc_calls), case.events.count("reference")
        assert runtime.start() is result
        assert len(rpc_calls) - before == 3 and scope._spawned and scope._spawn_preparation is None
        assert case.events.count("reference") - reference_before == (0 if optimized else 6)
        assert len(refusals) == 8 and len(probe_threads) == 1
