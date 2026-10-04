"""Native opt-in preview preserves the signed administrative terminal route."""
import json
import threading

import duckdb
import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import original
from benchmarks.agent_supervisor.container_coding.test_terminal_intent_requirement_planning import _requirements
from ipfs_accelerate_py.agent_supervisor.planning import intent_symbolic_planning
from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import PlanCreateService
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex, CodebaseScanLimits
from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import GlobalResourceScheduler, ResourceSchedulerConfig


@pytest.fixture
def terminal_case(original, tmp_path):
    repository, instruction, state = original
    contract = _requirements(instruction, symbolic=True)
    contract_path = tmp_path / "reviewed-operation-contract.json"
    contract_path.write_text(json.dumps(contract))
    prepared = prep.prepare(repository=repository, instruction=instruction, state=state,
        intent_requirement_contract=contract_path, disable_intent_autoencoder=True)
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=tmp_path / "resources.json", proof_resource_sampler=lambda: ProofHostResources(8, 8192, 8192),
        lane_reservations={}, auto_renew_leases=False, proof_backoff_seconds=0, poll_interval_seconds=0.005))
    connection = duckdb.connect(str(tmp_path / "native-codebase.duckdb"), config={"threads": 1, "memory_limit": "128MB"})
    store = DuckDBASTStore(connection=connection)
    artifacts = ImmutableCAS(tmp_path / "native-cas")
    index = RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store), artifacts=artifacts,
        catalog=CodebaseCatalog(store, artifacts))
    try:
        yield {"repository": repository, "state": state, "instruction": instruction,
            "prepared": prepared, "contract": contract, "index": index, "scheduler": scheduler}
    finally:
        connection.close()


def owner(case, **kwargs):
    return prep.prepare_repository_preview(state=case["state"], index=case["index"],
        repository_id="terminal:repository-preview", operation_id="capture:initial",
        scheduler=case["scheduler"], **kwargs)


def assert_released(case):
    resources = case["scheduler"].snapshot()
    assert resources["active_lease_count"] == resources["waiting_request_count"] == 0


def test_opt_in_real_service_consumes_native_context_without_changing_signed_selection(terminal_case):
    case = terminal_case
    selected_owner = owner(case)
    expected = intent_symbolic_planning.build_intent_symbolic_plan(case["contract"], manifest=case["prepared"]["manifest"])
    result = prep.plan(case["state"], repository_preview=selected_owner)
    assert result["qualified"] is True, result
    assert result["provider_calls"] == 0
    observed = result["repository_preview"]
    stages = {row["stage"]: row for row in observed["preview"]["stage_results"]}
    assert stages["obligation"]["passed"] and stages["candidate"]["passed"], stages
    assert observed["input_snapshot"]["snapshot_cid"] == observed["preview"]["input_snapshot_cid"]
    assert observed["structural_context"]["head"]["snapshot_cid"] == selected_owner.expected_head.snapshot_cid
    assert observed["source_root_correspondence"]["signed_source_count"] == len(case["prepared"]["manifest"]["payload"]["sources"])
    assert observed["selection_authority"] == "existing_administrative_manifest"
    assert observed["observed_facts_supplied"] == observed["model_calls"] == 0
    assert observed["production_admitted"] is observed["execution_authority"] is observed["completion_authority"] is False
    assert result["symbolic_planning"] == expected["receipt"]
    admission = json.loads((case["state"] / "admission.json").read_bytes())
    local.verify_local_benchmark_admission(admission)
    assert admission["graph"] == expected["graph"].to_dict()
    assert admission["receipt"]["payload"]["intent_symbolic_planning"] == expected["receipt"]
    assert json.loads((case["state"] / "repository-planning-preview.json").read_bytes()) == observed
    assert len(result["task_cids"]) == len(expected["graph"].tasks) == 1
    assert not (case["repository"] / "report.jsonl").exists()
    assert_released(case)


def test_default_symbolic_path_does_not_invoke_repository_preview(terminal_case, monkeypatch):
    from benchmarks.agent_supervisor.container_coding import terminal_repository_preview as adapter
    def forbidden(**kwargs):
        pytest.fail("unselected preview route executed")
    monkeypatch.setattr(adapter, "preview_prepared_repository_plan", forbidden)
    result = prep.plan(terminal_case["state"])
    assert result["qualified"] is True, result
    assert "repository_preview" not in result
    assert not (terminal_case["state"] / "repository-planning-preview.json").exists()
    assert terminal_case["index"].current("terminal:repository-preview") is None
    assert_released(terminal_case)


@pytest.mark.parametrize("drift", [None, "world", "descriptor"])
def test_repository_preview_preserves_selected_initial_context(terminal_case, monkeypatch, drift):
    """The optional native preview must retain the indexed selection gate."""
    from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
    case = terminal_case
    prep.initial_context(state=case["state"])
    loaded = initial.load_initial_context(state=case["state"], prepared=case["prepared"], require_empty_owner=True)
    selected = owner(case)
    if drift is not None:
        relative = (loaded["descriptor"]["world"]["artifact"] if drift == "world"
                    else loaded["receipt"]["descriptor"]["artifact"])
        path = case["repository"] / relative
        build = intent_symbolic_planning.build_intent_symbolic_plan
        def change(*args, **kwargs):
            planned = build(*args, **kwargs)
            path.write_bytes(path.read_bytes() + b" ")
            return planned
        monkeypatch.setattr(intent_symbolic_planning, "build_intent_symbolic_plan", change)
    result = prep.plan(case["state"], repository_preview=selected)
    assert result["qualified"] is (drift is None), result
    assert result["provider_calls"] == 0
    assert result["repository_preview"]["execution_authority"] is False
    assert (case["state"] / "admission.json").exists() is (drift is None)
    if drift is not None:
        assert result["failure"]["type"] == "ValueError"
        from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
        with IntentRepository(case["state"] / "intent.duckdb", install_schema=False) as intent:
            assert intent.plan_projection()["tasks"] == []
            assert intent.event_watermark() == 0
    assert_released(case)


def test_superseded_same_byte_head_refuses_before_admission(terminal_case):
    case = terminal_case
    selected = owner(case)
    successor = case["index"].prepare_current(case["repository"], repository_id=selected.expected_head.repository_id,
        operation_id="capture:successor", expected_head=selected.expected_head, scheduler=case["scheduler"], memory_mb=1024,
        limits=CodebaseScanLimits(max_entries=256, max_file_bytes=256 * 1024))
    assert successor.head.generation > selected.expected_head.generation
    result = prep.plan(case["state"], repository_preview=selected)
    assert result["qualified"] is False, result
    assert not (case["state"] / "admission.json").exists()
    assert not (case["state"] / "repository-planning-preview.json").exists()
    assert_released(case)


def test_source_change_during_actual_service_candidate_work_prevents_admission(terminal_case, monkeypatch):
    case = terminal_case
    selected = owner(case)
    candidate = PlanCreateService._stage_candidate
    def edit(self, *args, **kwargs):
        result = candidate(self, *args, **kwargs)
        assert result[0].passed
        (case["repository"] / "bottle.py").write_text("def application():\n    return 'changed after selection'\n")
        return result
    monkeypatch.setattr(PlanCreateService, "_stage_candidate", edit)
    result = prep.plan(case["state"], repository_preview=selected)
    assert result["qualified"] is False
    assert not (case["state"] / "admission.json").exists()
    assert not (case["state"] / "repository-planning-preview.json").exists()
    assert_released(case)


def test_head_change_after_preview_is_checked_again_before_administration(terminal_case, monkeypatch):
    case = terminal_case
    selected = owner(case)
    build = intent_symbolic_planning.build_intent_symbolic_plan
    def publish(*args, **kwargs):
        proposed = build(*args, **kwargs)
        case["index"].prepare_current(case["repository"], repository_id=selected.expected_head.repository_id,
            operation_id="capture:late-successor", expected_head=selected.expected_head, scheduler=case["scheduler"], memory_mb=1024,
            limits=CodebaseScanLimits(max_entries=256, max_file_bytes=256 * 1024))
        return proposed
    monkeypatch.setattr(intent_symbolic_planning, "build_intent_symbolic_plan", publish)
    result = prep.plan(case["state"], repository_preview=selected)
    assert result["qualified"] is False
    assert (case["state"] / "repository-planning-preview.json").exists()
    assert not (case["state"] / "admission.json").exists()
    assert_released(case)


def test_explicit_capture_retry_and_preview_never_reprepare(terminal_case, monkeypatch):
    case = terminal_case
    selected = owner(case)
    assert owner(case).expected_head == selected.expected_head
    def forbidden(*args, **kwargs):
        pytest.fail("preview or replay started a source capture")
    monkeypatch.setattr(case["index"], "prepare_current", forbidden)
    result = prep.plan(case["state"], repository_preview=selected)
    assert result["qualified"] is True, result
    assert_released(case)


@pytest.mark.parametrize("replay_number", [2, 3], ids=["admission", "materialization"])
@pytest.mark.parametrize("fault", ["head_successor", "cancellation"])
def test_administrative_replay_drift_cannot_publish_or_commit_tasks(terminal_case, monkeypatch, replay_number, fault):
    from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
    case = terminal_case
    cancelled = threading.Event()
    selected = owner(case, cancel_event=cancelled)
    build = intent_symbolic_planning.build_intent_symbolic_plan
    expected = build(case["contract"], manifest=case["prepared"]["manifest"])
    calls = 0
    def drift(*args, **kwargs):
        nonlocal calls
        proposed = build(*args, **kwargs)
        calls += 1
        if calls == replay_number:
            if fault == "cancellation":
                cancelled.set()
            else:
                case["index"].prepare_current(case["repository"], repository_id=selected.expected_head.repository_id,
                    operation_id="capture:replay-successor", expected_head=selected.expected_head,
                    scheduler=case["scheduler"], memory_mb=1024,
                    limits=CodebaseScanLimits(max_entries=256, max_file_bytes=256 * 1024))
        return proposed
    monkeypatch.setattr(intent_symbolic_planning, "build_intent_symbolic_plan", drift)
    result = prep.plan(case["state"], repository_preview=selected)
    assert calls >= replay_number
    assert result["qualified"] is False, result
    assert not (case["state"] / "admission.json").exists()
    if (case["state"] / "intent.duckdb").exists():
        with IntentRepository(case["state"] / "intent.duckdb") as intent:
            assert all(intent.get_task(task.task_cid) is None for task in expected["graph"].tasks)
    assert_released(case)


def test_unselected_direct_preparation_cannot_enable_preview_or_dispatch_provider(original, tmp_path):
    root, instruction, state = original
    prep.prepare(repository=root, instruction=instruction, state=state, disable_intent_autoencoder=True)
    calls = []
    with pytest.raises(ValueError, match="symbolic operation"):
        prep.prepare_repository_preview(state=state, index=None, repository_id="terminal:wrong", operation_id="forbidden")
    with pytest.raises(ValueError, match="symbolic operation"):
        prep.plan(state, repository_preview=object(), provider_callable=lambda *args, **kwargs: calls.append(args))
    assert calls == []
    assert not (state / "planner-invoked.json").exists()
