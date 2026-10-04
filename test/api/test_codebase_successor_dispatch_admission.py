"""Real local signing/Git/Intent around authored successor receiving controls.

Historical source/model records are genuine typed authored metadata. Current
receiving is explicitly doubled; these tests cover admission/rollback/public
contracts, not native source/model freshness or worker qualification. No fits,
forward inference, private source/model owners or worker jobs are run.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.control import profile_authority as authority
from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    PromptAcceptanceRecord, PromptGoalGraph, PromptGoalRecord, PromptOutputRecord,
    PromptTaskRecord, PromptValidationRecord,
)
from ipfs_accelerate_py.agent_supervisor.runtime import codebase_inventory_execution as execution
from ipfs_accelerate_py.agent_supervisor.runtime import codebase_successor_dispatch_admission as joined
from ipfs_accelerate_py.agent_supervisor.runtime import codebase_successor_dispatch_context as worker
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import router_public_instruction as public
from ipfs_accelerate_py.agent_supervisor.runtime import supervisor_meta_index as meta
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_datasets_py.logic.software_contracts import codebase_inventory_resume as resume
from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor_model as successor
from test.api.test_codebase_successor_dispatch_context import authored_successor_context


def _git(root, *arguments):
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


@pytest.fixture(autouse=True)
def prohibit_private_source_model_and_numerical_work(monkeypatch, tmp_path):
    from ipfs_datasets_py.duckdb_control.autoencoder_registry import AutoencoderRegistry
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex
    from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore

    def forbidden(*_, **__):
        pytest.fail("admission protocol control opened a private source/model owner or performed numerical work")

    for owner, name in ((AutoencoderRegistry, "__init__"), (CodebaseCatalog, "__init__"),
            (RepositoryCodebaseIndex, "__init__"), (DuckDBASTStore, "__init__"),
            (resume, "_worker"), (resume.features, "train_projection_features"),
            (resume.features, "infer_projection_features"), (resume.training, "train_current_codebase_features"),
            (resume.training, "_worker")):
        monkeypatch.setattr(owner, name, forbidden)
    # Genuine local lifecycle/signatures remain inside this fixture's private
    # temporary account registry, rather than changing the user's registry.
    monkeypatch.setattr(authority, "_LIFECYCLE_REGISTRY_ROOT_OVERRIDE", tmp_path / "lifecycle-root-registry")
    monkeypatch.delenv(meta.ENV_DUCKDB, raising=False)
    monkeypatch.delenv(meta.ENV_DUCKLAKE, raising=False)


@pytest.fixture
def successor_case(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "input.py").write_bytes(b"VALUE = 2\n")
    (repository / "keep.py").write_bytes(b"VALUE = 1\n")
    _git(repository, "init", "-q")
    _git(repository, "add", ".")
    _git(repository, "-c", "user.name=Successor admission controls",
        "-c", "user.email=successor-controls@example.invalid", "commit", "-qm", "independent fixture")
    profile, lifecycle = tmp_path / "profile", tmp_path / "lifecycle"
    Supervisor.init_local(repository=repository, consent=True, profile_dir=profile, lifecycle_dir=lifecycle)
    policy = content_identity(local.LOCAL_POLICY)
    check = PromptValidationRecord(validation_key="public-value-check", argv=(sys.executable, "keep.py"),
        policy_cid=policy)
    acceptance = PromptAcceptanceRecord(criterion_key="declared-check-pass",
        criterion="The independently declared public runtime check must pass", validation_keys=(check.validation_key,))
    goal = PromptGoalRecord(goal_key="SUCCESSOR-ALL", parent_goal_cid="", dependency_goal_cids=(),
        title="Retain both independent tasks", objective="Modify the admitted value and keep both runtime checks",
        rationale="Authored administrator task population", scope_paths=("input.py", "keep.py"),
        acceptance=(acceptance,))
    tasks = []
    for key in ("SUCCESSOR-FIRST", "SUCCESSOR-SECOND"):
        tasks.append(PromptTaskRecord(task_key=key, goal_cid=goal.goal_cid,
            dependency_task_cids=() if not tasks else (tasks[0].task_cid,),
            objective="Modify the explicitly admitted value and satisfy the public check",
            rationale="Independent signed work remains pending", scope_paths=("input.py", "keep.py"),
            outputs=(PromptOutputRecord(path="input.py", effect="modify", media_type="text/x-python"),),
            validations=(check,), acceptance=(acceptance,), evidence_cids=(), policy_roots=(policy,),
            predicted_files=("input.py",)))
    roots = {name: content_identity({"authored-successor-root": name})
        for name in ("request_cid", "scan_cid", "program_root")}
    graph = PromptGoalGraph(**roots, policy_roots=(policy,), goals=(goal,), tasks=tuple(tasks), evidence=())
    specs = [{"task_key": task.task_key, "scope_paths": list(task.scope_paths),
        "outputs": [{name: getattr(item, name) for name in ("path", "effect", "media_type")}
            for item in task.outputs],
        "validations": [{name: local._plain(getattr(item, name)) for name in
            ("validation_key", "argv", "cwd", "expected_exit_codes", "policy_cid")}
            for item in task.validations],
        "acceptance": [{name: local._plain(getattr(item, name)) for name in
            ("criterion_key", "criterion", "evidence_cids", "validation_keys")}
            for item in task.acceptance],
        "dependencies": [] if position == 0 else [tasks[0].task_key]}
        for position, task in enumerate(tasks)]
    context = authored_successor_context()
    selection = successor.CodebaseSuccessorScanRecord.from_dict(
        context["selection"]["artifact_cid"], context["selection"]["value"])
    refs = context["inventory"]["scan"]
    root = resume.CodebaseScanResumeRoot.from_dict(refs["root_cid"], refs["root_record"])
    completion = resume.CodebaseScanResumeCompletion.from_dict(refs["completion_cid"], refs["completion_record"])
    declaration = worker.successor_declaration(context)
    observations = []
    native_scope = joined._current_scope
    case = {"repository": repository, "profile": profile, "lifecycle": lifecycle, "context": context,
        "selection": selection, "root": root, "completion": completion, "index": object(), "registry": object(),
        "tasks": tasks, "graph": graph, "specs": specs, "roots": roots, "observations": observations,
        "native_scope": native_scope, "close_error": None, "close_callback": None, "exit_callback": None}

    @contextmanager
    def controlled_current_scope(**arguments):
        # Explicit receiving-protocol double. This does not qualify native
        # source/model currentness even when a product receipt is returned.
        assert arguments["selection"] is selection
        assert arguments["root"] is root and arguments["completion"] is completion
        assert arguments["index"] is case["index"] and arguments["registry"] is case["registry"]
        observations.append("entry")

        def close():
            observations.append("close")
            if case["close_error"] is not None:
                raise ValueError(case["close_error"])
            if case["close_callback"] is not None:
                case["close_callback"]()

        yield deepcopy(context), deepcopy(declaration), close
        observations.append("exit")
        if case["exit_callback"] is not None:
            case["exit_callback"]()

    monkeypatch.setattr(joined, "_current_scope", controlled_current_scope)
    manifest = joined.author_current_successor_manifest(selection, root, completion, case["index"], repository,
        registry=case["registry"], profile_dir=profile, lifecycle_dir=lifecycle, task_specs=specs, planning_roots=roots)
    admission = joined.admit_current_successor_plan(**_current_arguments(case), manifest=manifest, graph=graph)
    case.update(manifest=manifest, admission=admission)
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        case["intent"] = intent
        yield case


def _current_arguments(case):
    return {name: case[name] for name in ("selection", "root", "completion", "index", "repository", "registry")}


def _prepare(case, *, position=0):
    return joined.prepare_current_successor_worker_context(**_current_arguments(case),
        admission=case["admission"], task_cid=case["tasks"][position].task_cid, source_path="keep.py",
        expected_source_sha256=hashlib.sha256((case["repository"] / "keep.py").read_bytes()).hexdigest())


def test_real_signatures_use_versioned_successor_profiles_and_all_pending_tasks(successor_case):
    case = successor_case
    verified = local.verify_local_benchmark_admission(case["admission"])
    manifest, receipt = verified["manifest"], verified["receipt"]
    assert manifest["schema"] == local.SUCCESSOR_MANIFEST_SCHEMA == "supervisor-local-benchmark-manifest@6"
    assert receipt["schema"] == local.SUCCESSOR_PLANNING_RECEIPT_SCHEMA == "supervisor-local-planning-receipt@4"
    assert manifest["tasks"] == case["specs"]
    assert verified["graph"].to_dict() == case["graph"].to_dict()
    assert receipt["administrator_task_cids"] == sorted(task.task_cid for task in case["tasks"])
    assert receipt["current_facts"] == receipt["removed_task_cids"] == []
    assert receipt["runtime_requirements_preserved"] is True
    assert receipt["pending_requirements"]
    assert all(row["phase"] == "post_execution" and row["required"] is True
        for row in receipt["pending_requirements"])
    assert receipt["codebase_successor_context_cid"] == content_identity(case["context"])
    assert receipt["successor_selection_cid"] == case["selection"].artifact_cid
    assert case["observations"] == ["entry", "close", "exit"] * 2


def test_real_signed_successor_runtime_requires_native_inventory_scope(
        successor_case, monkeypatch, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.entrypoints import admitted_benchmark_runtime as runtime

    case, observations = successor_case, []
    native_verify = runtime.verify_local_benchmark_admission

    def verified(admission, **options):
        result = native_verify(admission, **options)
        observations.append(result["manifest"]["schema"])
        return result

    monkeypatch.setattr(runtime, "verify_local_benchmark_admission", verified)
    destination = tmp_path / "refused-runtime"
    # Genuine signed native admission is replayed. The unused owner arguments
    # are inert: refusal must precede any owner, lease, listener or Popen work.
    with pytest.raises(ValueError, match="exact native inventory execution scope"):
        runtime.AdmittedBenchmarkRuntime._create(destination,
            admission=case["admission"], server=object(), source=case["intent"])
    assert observations == [local.SUCCESSOR_MANIFEST_SCHEMA]
    assert not destination.exists()


def test_native_materialization_retains_both_contracts_and_successor_reference(successor_case):
    case = successor_case
    result = joined.materialize_current_successor_plan(**_current_arguments(case),
        admission=case["admission"], intent=case["intent"])
    assert sorted(result["task_cids"]) == sorted(task.task_cid for task in case["tasks"])
    assert result["schema"] == "supervisor-current-successor-native-materialization@1"
    assert result["administrator_task_population_preserved"] is result["runtime_requirements_preserved"] is True
    assert result["current_facts"] == result["removed_task_cids"] == []
    plan = case["intent"].get_plan(result["plan_id"])
    reference = plan["body"]["local_planning_receipt_ref"]
    assert reference["schema"] == local.SUCCESSOR_PLANNING_RECEIPT_REFERENCE_SCHEMA
    assert reference["codebase_successor_context_cid"] == content_identity(case["context"])
    assert reference["successor_selection_cid"] == case["selection"].artifact_cid
    assert local.load_local_planning_receipt(reference, manifest=case["manifest"]) == case["admission"]["receipt"]
    for task in case["tasks"]:
        installed = case["intent"].get_task(task.task_cid)
        contract, manifest, _, _ = local._contract(installed["body"], task.task_cid)
        assert contract["task_spec"] == next(spec for spec in case["specs"] if spec["task_key"] == task.task_key)
        assert manifest["codebase_successor_context"] == worker.successor_declaration(case["context"])
        assert contract["planning_receipt_cid"] == reference["receipt_cid"]
        assert contract["pending_cid"] == reference["pending_cid"]
        assert contract["pending_requirements"] == reference["pending_requirements"]


def test_closing_receiving_fault_rolls_back_entire_native_intent_population(successor_case):
    case = successor_case
    case["close_error"] = "controlled closing successor receiving fault"
    with pytest.raises(ValueError, match="controlled closing"):
        joined.materialize_current_successor_plan(**_current_arguments(case),
            admission=case["admission"], intent=case["intent"])
    assert all(case["intent"].get_task(task.task_cid) is None for task in case["tasks"])
    assert case["intent"].get_plan(case["admission"]["receipt"]["payload"]["plan_id"]) is None


def test_owner_verification_preserves_full_task_and_pending_reference(successor_case):
    case = successor_case
    receipt = joined.verify_current_successor_admission(**_current_arguments(case), admission=case["admission"])
    assert receipt["schema"] == "supervisor-current-successor-admission-verification@1"
    assert receipt["successor_selection_cid"] == case["selection"].artifact_cid
    assert receipt["source_delta_cid"] == case["context"]["source_delta"]["artifact_cid"]
    assert receipt["administrator_task_cids"] == sorted(task.task_cid for task in case["tasks"])
    assert receipt["pending_cid"] == case["admission"]["receipt"]["payload"]["pending_cid"]
    assert receipt["native_persistence_verified_here"] is False
    assert all(flag is False for flag in receipt["authority"].values())


@pytest.mark.parametrize("change", ["selection", "source_delta", "context", "tasks", "pending", "runtime_flag"])
def test_resigned_admission_cannot_change_exact_successor_or_full_task_contract(successor_case, change):
    case = successor_case
    admission = deepcopy(case["admission"])
    receipt = admission["receipt"]["payload"]
    if change == "tasks":
        receipt["administrator_task_cids"].pop()
    elif change == "pending":
        receipt["pending_requirements"][0]["required"] = False
    elif change == "runtime_flag":
        receipt["runtime_requirements_preserved"] = 1
    else:
        field = {"selection": "successor_selection_cid", "source_delta": "source_delta_cid",
            "context": "codebase_successor_context_cid"}[change]
        receipt[field] = content_identity({"foreign-receipt": change})
    admission["receipt"] = local._signed(receipt, admission["manifest"]["payload"])
    with pytest.raises(ValueError):
        joined.verify_current_successor_admission(**_current_arguments(case), admission=admission)


@pytest.mark.parametrize("position", [0, 1])
def test_public_successor_artifact_replays_without_private_or_metadata_owners(
        successor_case, monkeypatch, tmp_path, position):
    case = successor_case
    descriptor = _prepare(case, position=position)
    artifact = Path(descriptor["artifact"])
    payload = json.loads(artifact.read_bytes())
    assert payload["schema"] == public.SUCCESSOR_SCHEMA == "supervisor-public-instruction@4"
    assert set(payload) == public.SUCCESSOR_FIELDS
    assert descriptor["successor_selection_cid"] == case["selection"].artifact_cid
    workspace = tmp_path / "worker"
    _git(case["repository"], "worktree", "add", "--detach", "-q", str(workspace))

    def forbidden(*_, **__):
        pytest.fail("public successor replay accessed a private or metadata owner")

    monkeypatch.setattr(local, "load_local_profile", forbidden)
    monkeypatch.setattr(local, "_signed", forbidden)
    monkeypatch.setattr(joined, "_current_scope", forbidden)
    monkeypatch.setenv(meta.ENV_DUCKDB, str(tmp_path / "unopened-meta.duckdb"))
    monkeypatch.setattr(meta.SupervisorMetaIndex, "from_env", classmethod(forbidden))
    monkeypatch.setattr(meta, "_connect", forbidden)
    task = case["tasks"][position]
    block, receipt = public.load_public_instruction(artifact=artifact, expected_sha256=descriptor["sha256"],
        task_cid=task.task_cid, prompt=json.dumps({"objective_id": task.task_key}), workspace=workspace)
    assert "VALUE = 1" in block and "CODEBASE INVENTORY ADVISORY" in block
    assert receipt["schema"] == "supervisor-public-instruction-inclusion@4"
    assert receipt["extra_provider_calls"] == 0
    assert receipt["codebase_successor"]["selection_cid"] == case["selection"].artifact_cid
    assert receipt["codebase_successor"]["native_successor_current_verified_here"] is False
    assert receipt["codebase_inventory"]["administrator_task_cids"] == sorted(item.task_cid for item in case["tasks"])
    assert receipt["codebase_inventory"]["pending_cid"] == case["admission"]["receipt"]["payload"]["pending_cid"]
    assert receipt["codebase_inventory"]["current_facts"] == receipt["codebase_inventory"]["removed_task_cids"] == []
    assert receipt["codebase_inventory"]["runtime_requirements_preserved"] is True
    assert all(flag is False for flag in receipt["codebase_successor"]["authority"].values())
    assert not (tmp_path / "unopened-meta.duckdb").exists()


@pytest.mark.parametrize("location", ["canonical", "worktree"])
@pytest.mark.parametrize("change", ["undeclared", "signed_bytes"])
def test_successor_public_worktree_rejects_extra_and_stale_sources(successor_case, tmp_path, location, change):
    case = successor_case
    descriptor = _prepare(case)
    workspace = tmp_path / "worker"
    _git(case["repository"], "worktree", "add", "--detach", "-q", str(workspace))
    target = case["repository"] if location == "canonical" else workspace
    if change == "undeclared":
        (target / "undeclared.py").write_text("VALUE = 3\n")
    else:
        (target / "keep.py").write_text("VALUE = 3\n")
    with pytest.raises(ValueError):
        public.load_public_instruction(artifact=Path(descriptor["artifact"]), expected_sha256=descriptor["sha256"],
            task_cid=case["tasks"][0].task_cid, prompt=json.dumps({"objective_id": case["tasks"][0].task_key}),
            workspace=workspace)


def test_real_resource_scope_shares_one_lease_with_controlled_paired_receiver(successor_case, monkeypatch, tmp_path):
    from ipfs_datasets_py.logic.software_contracts import codebase_inventory_successor_receiving as paired
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
        GlobalResourceScheduler, ResourceSchedulerConfig,
    )

    case = successor_case
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig(state_path=tmp_path / "resources.json",
        total_cpu_slots=2, total_memory_mb=2048, total_child_process_slots=2,
        lane_reservations={}, auto_renew_leases=False))
    calls = []
    monkeypatch.setattr(joined, "_full_context", lambda *_: deepcopy(case["context"]))

    @contextmanager
    def controlled_receiver(selection, completion, index, repository, **arguments):
        # The resource scope is genuine; source/model receiving is a labelled
        # protocol double and establishes no native owner freshness.
        assert selection is case["selection"] and completion is case["completion"]
        assert arguments["root"] is case["root"] and arguments["registry"] is case["registry"]
        lease = arguments["parent_lease"]
        assert not lease.released
        calls.append(lease)

        def close():
            assert not lease.released
            calls.append(lease)
            return completion

        yield close

    monkeypatch.setattr(paired, "_paired_current_codebase_successor_completion", controlled_receiver)
    with case["native_scope"](**_current_arguments(case), scheduler=scheduler, timeout_seconds=5) as (
            context, declaration, close):
        assert context == case["context"]
        assert declaration == worker.successor_declaration(case["context"])
        close()
    assert len(calls) == 2 and calls[0] is calls[1] and calls[0].released
    assert scheduler.snapshot()["active_lease_count"] == scheduler.snapshot()["waiting_request_count"] == 0


def test_execution_currentness_routes_to_exact_successor_owner_selection(successor_case, monkeypatch):
    case = successor_case
    owner = execution._Owner(case["root"], case["completion"], case["index"], case["repository"],
        case["registry"], None, None, None, 120, 1024, case["selection"])
    scope = object.__new__(execution.FrozenInventoryExecutionScope)
    scope._owner, scope._admission, scope._lease = owner, case["admission"], object()
    seen = []

    def receive(**arguments):
        seen.append(arguments)
        return {"protocol-double": "successor-currentness"}

    monkeypatch.setattr(joined, "verify_current_successor_admission", receive)
    assert scope._current_inventory() == {"protocol-double": "successor-currentness"}
    assert len(seen) == 1 and seen[0]["selection"] is case["selection"]
    assert seen[0]["root"] is case["root"] and seen[0]["completion"] is case["completion"]
    assert seen[0]["parent_lease"] is scope._lease
    original = scope._owner_fields()
    different = deepcopy(case["context"]["selection"]["value"])
    different["training_record_cid"] = content_identity({"foreign-training-record": 1})
    from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured

    scope._owner = replace(owner, successor_selection=successor.CodebaseSuccessorScanRecord.from_dict(
        cid_for_structured(different), different))
    assert scope._owner_fields() != original
    scope._owner = replace(owner, successor_selection={"artifact_cid": case["selection"].artifact_cid})
    with pytest.raises(execution.InventoryExecutionError, match="exact selected"):
        scope._owner_fields()


def test_successor_reservation_wrapper_preserves_exact_selection_and_configuration(successor_case, monkeypatch):
    case = successor_case
    seen = []

    @contextmanager
    def reserve(**arguments):
        # Boundary-forwarding protocol only; this is not a native lease or
        # launch authorization and never starts a worker.
        seen.append(arguments)
        yield "controlled-reservation"

    monkeypatch.setattr(execution, "reserve_inventory_execution", reserve)
    options = {**_current_arguments(case), "admission": case["admission"], "candidate": {"authored": 1},
        "timeout_seconds": 17, "memory_mb": 1024, "server": object(), "source": object(), "output": Path("authored-output")}
    selection = options.pop("selection")
    with joined.reserve_current_successor_execution(selection=selection, **options) as scope:
        assert scope == "controlled-reservation"
    assert seen == [{"successor_selection": selection, **options}]


def _mutation_at(case, phase, mutation):
    # A controlled native receiving boundary mutates a captured product
    # object after genuine signing/verification; no work or owner is added.
    case["close_callback" if phase == "closing" else "exit_callback"] = mutation


@pytest.mark.parametrize("phase", ["closing", "resource_exit"])
def test_signed_manifest_output_cannot_change_after_native_verification(successor_case, monkeypatch, phase):
    case = successor_case
    captured = []
    original = local.author_local_benchmark_manifest

    def observe(**arguments):
        result = original(**arguments)
        captured.append(result)
        return result

    monkeypatch.setattr(local, "author_local_benchmark_manifest", observe)

    def mutate():
        captured[-1]["payload"]["codebase_successor_context"]["selection_cid"] = content_identity(
            {"foreign-manifest-after-verification": phase})

    _mutation_at(case, phase, mutate)
    with pytest.raises(ValueError, match="manifest changed"):
        joined.author_current_successor_manifest(case["selection"], case["root"], case["completion"],
            case["index"], case["repository"], registry=case["registry"], profile_dir=case["profile"],
            lifecycle_dir=case["lifecycle"], task_specs=case["specs"], planning_roots=case["roots"])
    assert len(captured) == 1


@pytest.mark.parametrize("phase", ["closing", "resource_exit"])
@pytest.mark.parametrize("target", ["returned_admission", "detached_typed_graph"])
def test_admission_and_typed_graph_cannot_change_after_native_verification(
        successor_case, monkeypatch, phase, target):
    case = successor_case
    original_graph = case["graph"].to_dict()
    captured = []
    original = local.admit_local_benchmark_plan

    def observe(**arguments):
        result = original(**arguments)
        captured.append((result, arguments["graph"]))
        return result

    monkeypatch.setattr(local, "admit_local_benchmark_plan", observe)

    def mutate():
        admission, graph = captured[-1]
        if target == "returned_admission":
            admission["receipt"]["payload"]["runtime_requirements_preserved"] = False
        else:
            # Explicit forced mutation of an otherwise frozen typed object;
            # ordinary detachment does not replace the closing integrity fence.
            object.__setattr__(graph, "tasks", graph.tasks[:1])

    _mutation_at(case, phase, mutate)
    with pytest.raises(ValueError, match="graph or admission changed"):
        joined.admit_current_successor_plan(**_current_arguments(case),
            manifest=case["manifest"], graph=case["graph"])
    assert len(captured) == 1
    assert case["graph"].to_dict() == original_graph


@pytest.mark.parametrize("phase", ["closing", "resource_exit"])
def test_public_descriptor_cannot_change_after_real_artifact_preparation(successor_case, monkeypatch, phase):
    case = successor_case
    captured = []
    original = public.prepare_public_instruction_context

    def observe(**arguments):
        result = original(**arguments)
        captured.append(result)
        return result

    monkeypatch.setattr(public, "prepare_public_instruction_context", observe)

    def mutate():
        captured[-1]["successor_selection_cid"] = content_identity({"foreign-public-descriptor": phase})

    _mutation_at(case, phase, mutate)
    with pytest.raises(ValueError, match="descriptor changed"):
        _prepare(case)
    assert len(captured) == 1


@pytest.mark.parametrize("phase", ["closing", "resource_exit"])
@pytest.mark.parametrize("target", ["returned_result", "detached_admission"])
def test_materialization_mutation_before_commit_rolls_back_both_native_tasks(
        successor_case, monkeypatch, phase, target):
    case = successor_case
    original_admission = deepcopy(case["admission"])
    captured = []
    original = local._materialize_local_benchmark_plan

    def observe(**arguments):
        result = original(**arguments)
        captured.append((result, arguments["admission"]))
        return result

    monkeypatch.setattr(local, "_materialize_local_benchmark_plan", observe)

    def mutate():
        result, admission = captured[-1]
        if target == "returned_result":
            result["task_cids"].pop()
        else:
            admission["receipt"]["payload"]["runtime_requirements_preserved"] = False

    _mutation_at(case, phase, mutate)
    with pytest.raises(ValueError, match="materialization changed"):
        joined.materialize_current_successor_plan(**_current_arguments(case),
            admission=case["admission"], intent=case["intent"])
    assert len(captured) == 1
    assert case["admission"] == original_admission
    assert all(case["intent"].get_task(task.task_cid) is None for task in case["tasks"])
    assert case["intent"].get_plan(case["admission"]["receipt"]["payload"]["plan_id"]) is None
