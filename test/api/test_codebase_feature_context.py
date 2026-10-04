"""Actual source-bound numerical lineage and inert planning identity controls."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time

import duckdb
import pytest

from ipfs_accelerate_py.agent_supervisor.planning import codebase_feature_context as module
from ipfs_accelerate_py.agent_supervisor.planning.repository_plan_preview import RepositoryPlanPreviewOwner
from ipfs_datasets_py.duckdb_control.autoencoder_registry import AutoencoderRegistry
from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog
from ipfs_datasets_py.logic.software_contracts import codebase_source_training as training
from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex, StaleCodebaseError
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
from ipfs_datasets_py.logic.software_verification.pipeline import ContractSpec
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.proof_resource_safety import ProofHostResources
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceSchedulerConfig, LeaseCancelledError, LeaseTimeoutError,
    ResourceLane, ResourceUnavailableError,
)


def source(offset):
    return f"def increment(n: int) -> int:\n    return n + {offset}\n".encode()


def no_float(value):
    if type(value) is dict:
        return all(no_float(child) for child in value.values())
    if type(value) is list:
        return all(no_float(child) for child in value)
    return type(value) in {str, int, bool, type(None)}


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    root = tmp_path_factory.mktemp("feature-context")
    repository = root / "repository"
    repository.mkdir()
    for name, offset in (("calc.py", 1), ("tune.py", 3), ("canary.py", 5)):
        (repository / name).write_bytes(source(offset))
    (repository / "scope.txt").write_text("Authored four-file source scope.\n")
    for args in (("init", "-q"), ("config", "user.name", "Feature Fixture"),
                 ("config", "user.email", "feature@example.invalid"), ("add", "."), ("commit", "-qm", "authored")):
        subprocess.run(["git", "-C", str(repository), *args], check=True, capture_output=True)
    connection = duckdb.connect(str(root / "codebase.duckdb"), config={"threads": 1, "memory_limit": "64MB"})
    store = DuckDBASTStore(connection=connection)
    artifacts = ImmutableCAS(root / "source-artifacts")
    index = RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store), artifacts=artifacts,
        catalog=CodebaseCatalog(store, artifacts))
    scheduler = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=root / "admission.json", proof_resource_sampler=lambda: ProofHostResources(8, 16384, 16384),
        lane_reservations={}, auto_renew_leases=False, poll_interval_seconds=.005))
    head = index.prepare_current(repository, repository_id="fixture:feature-context",
        operation_id="initial", expected_head=None, scheduler=scheduler).head
    registry = AutoencoderRegistry(root / "model.duckdb", root / "model-artifacts")
    selections = tuple(training.CodebaseTrainingSelection(name, role,
        (ContractSpec("increment", postconditions=(f"result == n + {offset}",), contract_id="authored:" + name),))
        for name, role, offset in (("calc.py", "train", 2), ("tune.py", "tune", 3), ("canary.py", "canary", 5)))
    value = dict(root=root, repository=repository, connection=connection, registry=registry,
        scheduler=scheduler, selections=selections, index=index,
        owner=RepositoryPlanPreviewOwner(index=index, repository=repository, expected_head=head,
            scheduler=scheduler, timeout_seconds=90, memory_mb=1024))
    yield value
    state = scheduler.snapshot()
    assert state["active_lease_count"] == state["waiting_request_count"] == 0
    value["registry"].close()
    value["connection"].close()


def prepare(native, output, **kwargs):
    return module.prepare_codebase_feature_context(owner=native["owner"], registry=native["registry"],
        output=output, **kwargs)


@pytest.fixture(scope="module")
def learned(native):
    parent_owner = native["owner"]
    parent = prepare(native, native["root"] / "root-context", mode="train",
        selections=native["selections"], operation_id="context-root", epochs=1, learning_rate=.002)
    before = parent.retained_artifacts
    commit = subprocess.check_output(["git", "-C", str(native["repository"]), "rev-parse", "HEAD"])
    (native["repository"] / "calc.py").write_bytes(source(2))
    head = native["index"].prepare_current(native["repository"], repository_id=parent_owner.expected_head.repository_id,
        expected_head=parent_owner.expected_head, operation_id="successor", scheduler=native["scheduler"]).head
    assert subprocess.check_output(["git", "-C", str(native["repository"]), "rev-parse", "HEAD"]) == commit
    native["owner"] = replace(parent_owner, expected_head=head)
    child = prepare(native, native["root"] / "child-context", mode="train", selections=native["selections"],
        operation_id="context-child", parent_version_id=parent.version_id, epochs=1, learning_rate=.002)
    return dict(parent=parent, child=child, before=before, parent_owner=parent_owner)


def test_model_off_never_loads_fits_or_infers(native, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("model-off entered numerical/history path")
    for name in ("load_codebase_feature_training", "train_current_codebase_features", "infer_current_codebase_features"):
        monkeypatch.setattr(training, name, forbidden)
    context = prepare(native, tmp_path / "off")
    assert context.mode == "model_off" and context.version_id is None and context.actual_training_delta == 0
    assert context.material_binding["artifacts"].keys() == {"invocation"}
    assert module.verify_current_context(native["owner"], native["registry"], context) == context.material_binding
    assert no_float(context.material_binding)
    assert all(value is False for value in context.material_binding["authority"].values())
    assert context.material_binding["model_record_cid"] is None


def test_actual_root_child_retains_exact_basis_optimizer_and_current_sources(native, learned):
    parent, child = learned["parent"], learned["child"]
    root, successor = learned["before"], child.retained_artifacts
    assert parent.actual_training_delta == child.actual_training_delta == 1
    assert root["checkpoint"]["feature_space"] == successor["checkpoint"]["feature_space"]
    assert root["checkpoint"]["contract"] == successor["checkpoint"]["contract"]
    assert root["checkpoint"]["state"]["completed_epochs"] == 1
    assert successor["checkpoint"]["state"]["completed_epochs"] == 2
    assert root["checkpoint"]["state"]["parameters"] != successor["checkpoint"]["state"]["parameters"]
    assert successor["record"]["parent_version_id"] == parent.version_id
    assert len(successor["lineage"]) == 2
    assert successor["lineage"][1]["checkpoint"] == root["checkpoint"]
    assert sum(row["unknown_atoms"] for row in successor["metrics"]["train_coverage"]) > 0
    assert all(row["step"] == 2 for row in successor["checkpoint"]["state"]["adam"])
    assert parent.retained_artifacts == root
    assert len(native["index"].load(child.head.manifest_cid).snapshot.entries) == 4
    assert no_float(child.material_binding)
    assert child.material_binding["candidate_checkpoint_raw_cid"] == successor["record"]["checkpoint_raw_cid"]
    assert cid_for_structured({k: v for k, v in child.material_binding.items() if k != "context_cid"}) == child.cid
    assert native["registry"].resolve_head(successor["record"]["variant_id"], "main") is None


def test_frozen_and_verification_actually_infer_without_fitting(native, learned, tmp_path, monkeypatch):
    monkeypatch.setattr(training, "train_current_codebase_features", lambda *a, **k: pytest.fail("frozen mode fitted"))
    real, calls = training.infer_current_codebase_features, []
    def infer(*args, **kwargs):
        calls.append(kwargs["version_id"])
        return real(*args, **kwargs)
    monkeypatch.setattr(training, "infer_current_codebase_features", infer)
    context = prepare(native, tmp_path / "frozen", mode="frozen", version_id=learned["child"].version_id)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in context.output.iterdir()}
    binding = module.verify_current_context(native["owner"], native["registry"], context)
    assert calls == [context.version_id, context.version_id]
    assert context.actual_training_delta == 0 and binding == context.material_binding
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in context.output.iterdir()}
    assert context.retained_artifacts["inference"]["training_executed"] is False


@pytest.mark.parametrize("mode,changes", [
    ("unknown", {}), (True, {}), ("model_off", {"version_id": "invented"}),
    ("frozen", {}), ("model_off", {"epochs": 2}), ("frozen", {"operation_id": "fit"}),
    ("train", {"epochs": 48}), ("train", {"epochs": True}),
])
def test_mode_and_budget_controls_refuse_before_any_fit(native, tmp_path, monkeypatch, mode, changes):
    monkeypatch.setattr(training, "train_current_codebase_features", lambda *a, **k: pytest.fail("invalid mode fitted"))
    with pytest.raises(module.CodebaseFeatureContextError):
        prepare(native, tmp_path / "invalid", mode=mode, **changes)
    assert not (tmp_path / "invalid").exists()


def test_unknown_and_old_source_model_are_refused(native, learned, tmp_path):
    with pytest.raises((module.CodebaseFeatureContextError, ValueError), match="current|head"):
        prepare(native, tmp_path / "old", mode="frozen", version_id=learned["parent"].version_id)
    with pytest.raises(ValueError):
        prepare(native, tmp_path / "foreign", mode="frozen", version_id="sha256:" + "0" * 64)


def test_old_context_and_same_head_dirty_source_are_refused(native, learned, tmp_path):
    with pytest.raises(module.CodebaseFeatureContextError, match="another selected"):
        module.verify_current_context(native["owner"], native["registry"], learned["parent"])
    path = native["repository"] / "calc.py"
    original = path.read_bytes()
    path.write_bytes(source(9))
    try:
        with pytest.raises(StaleCodebaseError):
            prepare(native, tmp_path / "dirty")
        with pytest.raises(StaleCodebaseError):
            module.verify_current_context(native["owner"], native["registry"], learned["child"])
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize("role", ["checkpoint", "inference", "record", "metrics", "lineage", "invocation"])
def test_full_retained_artifact_tampering_is_refused(native, learned, tmp_path, role):
    target = tmp_path / "copy"
    shutil.copytree(learned["child"].output, target)
    context = module.FrozenCodebaseFeatureContext(target, learned["child"]._material_bytes)
    with (target / (role + ".json")).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(module.CodebaseFeatureContextError, match="identity"):
        module.verify_current_context(native["owner"], native["registry"], context)


def test_context_material_is_detached_and_dataclass_frozen(learned):
    context = learned["child"]
    value = context.material_binding
    value["authority"]["proof_authority"] = True
    value["head"]["generation"] = 100
    assert context.material_binding["authority"]["proof_authority"] is False
    assert context.head.generation == 2
    with pytest.raises(FrozenInstanceError):
        context.output = Path("/tmp/changed")


def test_extra_file_and_role_path_escape_are_refused(native, learned, tmp_path):
    target = tmp_path / "copy"
    shutil.copytree(learned["child"].output, target)
    context = module.FrozenCodebaseFeatureContext(target, learned["child"]._material_bytes)
    (target / "extra.json").write_text("{}")
    with pytest.raises(module.CodebaseFeatureContextError, match="inventory"):
        module.verify_current_context(native["owner"], native["registry"], context)
    value = deepcopy(context.material_binding)
    value["artifacts"]["record"]["relative_path"] = "../record.json"
    value["context_cid"] = cid_for_structured({k: v for k, v in value.items() if k != "context_cid"})
    raw = module._wire(value)
    (target / "context.json").write_bytes(raw)
    forged = module.FrozenCodebaseFeatureContext(target, raw)
    with pytest.raises(module.CodebaseFeatureContextError, match="role-specific"):
        module.verify_current_context(native["owner"], native["registry"], forged)


def test_reused_training_operation_and_output_refuse_without_fitting(native, learned, tmp_path, monkeypatch):
    monkeypatch.setattr(training, "train_current_codebase_features", lambda *a, **k: pytest.fail("replay fitted"))
    with pytest.raises(module.CodebaseFeatureContextError, match="already exists"):
        prepare(native, tmp_path / "reused-operation", mode="train", selections=native["selections"],
            parent_version_id=learned["parent"].version_id, operation_id="context-child", epochs=1, learning_rate=.002)
    with pytest.raises(module.CodebaseFeatureContextError, match="must be new"):
        prepare(native, learned["child"].output)


def test_output_inside_repository_and_symlink_are_refused(native, tmp_path):
    with pytest.raises(module.CodebaseFeatureContextError, match="outside"):
        prepare(native, native["repository"] / "context")
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(module.CodebaseFeatureContextError, match="canonical"):
        prepare(native, tmp_path / "link" / "context")


def test_pre_cancel_and_deadline_refuse_without_artifacts(native, tmp_path):
    event = threading.Event()
    event.set()
    owner = replace(native["owner"], cancel_event=event)
    with pytest.raises(LeaseCancelledError):
        module.prepare_codebase_feature_context(owner=owner, registry=native["registry"], output=tmp_path / "cancelled")
    assert not (tmp_path / "cancelled").exists()
    owner = replace(native["owner"], timeout_seconds=1e-9)
    with pytest.raises(LeaseTimeoutError):
        module.prepare_codebase_feature_context(owner=owner, registry=native["registry"], output=tmp_path / "expired")
    assert not (tmp_path / "expired").exists()


def test_released_parent_cannot_supply_current_context(native, tmp_path):
    lease = native["scheduler"].acquire(lane=ResourceLane.SNAPSHOT_EVALUATION,
        cpu_slots=1, memory_mb=1024, child_process_slots=1, timeout=1)
    lease.release()
    owner = replace(native["owner"], scheduler=None, parent_lease=lease)
    with pytest.raises(LeaseCancelledError):
        module.prepare_codebase_feature_context(owner=owner, registry=native["registry"], output=tmp_path / "released")
    assert not (tmp_path / "released").exists()


@pytest.mark.parametrize("role", ["metrics", "invocation", "inference", "receipt"])
def test_resealed_payloads_cannot_replace_native_numerics_or_controls(native, learned, tmp_path, role):
    target = tmp_path / "copy"
    shutil.copytree(learned["child"].output, target)
    artifact_role = "inference" if role == "receipt" else role
    payload = json.loads((target / (artifact_role + ".json")).read_text())
    if role == "metrics":
        payload["after"]["objective"] = 0.
    elif role == "invocation":
        payload["configuration"]["epochs"] = 32
    elif role == "inference":
        payload["inference"]["rows"][0]["latent"][0] += 1.
    else:
        payload["worker_receipt"]["executable_sha256"] = "0" * 64
    raw = module._wire(payload)
    (target / (artifact_role + ".json")).write_bytes(raw)
    binding = deepcopy(learned["child"].material_binding)
    binding["artifacts"][artifact_role] = module._artifact(artifact_role, raw)
    binding["context_cid"] = cid_for_structured({k: v for k, v in binding.items() if k != "context_cid"})
    material = module._wire(binding)
    (target / "context.json").write_bytes(material)
    forged = module.FrozenCodebaseFeatureContext(target, material)
    with pytest.raises(module.CodebaseFeatureContextError, match="metrics|controls|inference receipt|inference differs"):
        module.verify_current_context(native["owner"], native["registry"], forged)


def test_source_edit_after_actual_inference_withholds_context(native, learned, tmp_path, monkeypatch):
    real = training.infer_current_codebase_features
    path = native["repository"] / "calc.py"
    original = path.read_bytes()
    def infer(*args, **kwargs):
        result = real(*args, **kwargs)
        path.write_bytes(source(8))
        return result
    monkeypatch.setattr(training, "infer_current_codebase_features", infer)
    try:
        with pytest.raises(StaleCodebaseError):
            prepare(native, tmp_path / "late-edit", mode="frozen", version_id=learned["child"].version_id)
    finally:
        path.write_bytes(original)
    assert native["scheduler"].snapshot()["active_lease_count"] == 0


def test_artifact_mutation_on_final_source_observation_is_caught(native, tmp_path, monkeypatch):
    output = tmp_path / "late-artifact"
    real = native["index"].observe_current
    def observe(*args, **kwargs):
        result = real(*args, **kwargs)
        if (output / "context.json").exists():
            with (output / "invocation.json").open("ab") as stream:
                stream.write(b" ")
        return result
    monkeypatch.setattr(native["index"], "observe_current", observe)
    with pytest.raises(module.CodebaseFeatureContextError, match="identity"):
        prepare(native, output)
    assert native["scheduler"].snapshot()["active_lease_count"] == 0


def test_actual_frozen_worker_cancellation_releases_all_resources(native, learned, tmp_path, monkeypatch):
    event = threading.Event()
    owner = replace(native["owner"], cancel_event=event)
    real = training.run_bounded_stdin_tool
    calls = []
    def execute(*args, **kwargs):
        calls.append(args[0])
        timer = threading.Timer(.05, event.set)
        timer.start()
        try:
            return real(*args, **kwargs)
        finally:
            timer.cancel()
    monkeypatch.setattr(training, "run_bounded_stdin_tool", execute)
    with pytest.raises(LeaseCancelledError):
        module.prepare_codebase_feature_context(owner=owner, registry=native["registry"],
            output=tmp_path / "cancel-worker", mode="frozen", version_id=learned["child"].version_id)
    assert len(calls) == 1 and "-I" in calls[0]
    snapshot = native["scheduler"].snapshot()
    assert snapshot["active_lease_count"] == snapshot["waiting_request_count"] == 0


def test_exact_small_parent_budget_supports_real_nested_frozen_inference(native, learned, tmp_path):
    with native["scheduler"].acquire(lane=ResourceLane.SNAPSHOT_EVALUATION,
            cpu_slots=1, memory_mb=1024, child_process_slots=1, timeout=1) as lease:
        owner = replace(native["owner"], scheduler=None, parent_lease=lease, timeout_seconds=30)
        context = module.prepare_codebase_feature_context(owner=owner, registry=native["registry"],
            output=tmp_path / "nested", mode="frozen", version_id=learned["child"].version_id)
        assert context.actual_training_delta == 0
        assert not lease.released
        assert native["scheduler"].snapshot()["active_lease_count"] == 1
    assert native["scheduler"].snapshot()["active_lease_count"] == 0


def test_insufficient_parent_reservation_refuses_instead_of_deadlocking(native, tmp_path):
    with native["scheduler"].acquire(lane=ResourceLane.SNAPSHOT_EVALUATION,
            cpu_slots=1, memory_mb=64, child_process_slots=1, timeout=1) as lease:
        owner = replace(native["owner"], scheduler=None, parent_lease=lease, timeout_seconds=1)
        started = time.monotonic()
        with pytest.raises(ResourceUnavailableError, match="exceeds parent lease capacity"):
            module.prepare_codebase_feature_context(owner=owner, registry=native["registry"], output=tmp_path / "too-small")
        assert time.monotonic() - started < 3
        assert not (tmp_path / "too-small").exists()


def test_concurrent_same_operation_is_not_labeled_as_a_second_fit(native, learned, tmp_path, monkeypatch):
    real = training.train_current_codebase_features
    entered, release = threading.Event(), threading.Event()
    calls, completed, errors = [], [], []
    def train(*args, **kwargs):
        calls.append(kwargs["operation_id"])
        entered.set()
        assert release.wait(10)
        return real(*args, **kwargs)
    monkeypatch.setattr(training, "train_current_codebase_features", train)
    def first():
        try:
            completed.append(prepare(native, tmp_path / "first", mode="train", selections=native["selections"],
                operation_id="concurrent-context", parent_version_id=learned["child"].version_id, epochs=1, learning_rate=.002))
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=first)
    thread.start()
    assert entered.wait(10)
    try:
        with pytest.raises(module.CodebaseFeatureContextError, match="already active"):
            prepare(native, tmp_path / "second", mode="train", selections=native["selections"],
                operation_id="concurrent-context", parent_version_id=learned["child"].version_id, epochs=1, learning_rate=.002)
    finally:
        release.set()
        thread.join(30)
    assert not thread.is_alive() and errors == []
    assert len(completed) == 1 and completed[0].actual_training_delta == 1
    assert calls == ["concurrent-context"] and not (tmp_path / "second").exists()
    assert native["scheduler"].snapshot()["active_lease_count"] == 0


def test_cold_native_owners_replay_context_without_training(native, learned, monkeypatch):
    native["registry"].close()
    native["connection"].close()
    connection = duckdb.connect(str(native["root"] / "codebase.duckdb"), config={"threads": 1, "memory_limit": "64MB"})
    store = DuckDBASTStore(connection=connection)
    artifacts = ImmutableCAS(native["root"] / "source-artifacts")
    index = RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store), artifacts=artifacts,
        catalog=CodebaseCatalog(store, artifacts))
    registry = AutoencoderRegistry(native["root"] / "model.duckdb", native["root"] / "model-artifacts")
    native.update(connection=connection, index=index, registry=registry,
        owner=replace(native["owner"], index=index))
    monkeypatch.setattr(training, "train_current_codebase_features", lambda *a, **k: pytest.fail("cold replay fitted"))
    monkeypatch.setattr(artifacts, "put", lambda *a, **k: pytest.fail("cold replay wrote CAS"))
    assert module.verify_current_context(native["owner"], registry, learned["child"]) == learned["child"].material_binding
