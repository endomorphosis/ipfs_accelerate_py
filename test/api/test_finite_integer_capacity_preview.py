"""Joined source, model and real host capacity qualification; no tool shims."""
from dataclasses import replace
from pathlib import Path
import threading

import pytest

from ipfs_accelerate_py.agent_supervisor.planning import finite_integer_capacity_preview as adapter
from ipfs_accelerate_py.agent_supervisor.planning import finite_integer_capacity as capacity_module
from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import PlanCreatePreviewReceipt
from ipfs_datasets_py.logic.software_contracts.codebase_ir import StaleCodebaseError
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import (
    GlobalResourceScheduler, ResourceSchedulerConfig, LeaseCancelledError,
    ResourceLane,
)
from test.api.test_finite_integer_codebase import (
    finite_prepared, finite_tools, finite_source, finite_git,
)
from test.api.test_finite_integer_plan_preview import preview_arguments, assert_released, TASK_OFFSET


@pytest.fixture
def prepared(finite_prepared):
    # Independently collect actual host resources. The prior suite's injected
    # source-only sampler cannot supply this new capacity profile.
    finite_prepared["scheduler"] = GlobalResourceScheduler(ResourceSchedulerConfig.for_proof_host(
        state_path=finite_prepared["output"].parent / "real-admission.json",
        lane_reservations={}, auto_renew_leases=False))
    return finite_prepared


def run(prepared, tools, **changes):
    args = preview_arguments(prepared, tools)
    args.update(changes)
    return adapter.preview_capacity_bound_finite_integer_plan(**args)


def test_actual_reservation_forecast_and_universal_model_refutation(prepared, finite_tools):
    original = (prepared["repository"] / "calc.py").read_bytes()
    result = run(prepared, finite_tools)
    receipt = PlanCreatePreviewReceipt.from_dict(result["preview"])
    stages = {row.stage.value: row for row in receipt.stage_results}
    assert stages["parallel_plan"].passed and result["execution_plan"]["admitted"] is True
    assert not stages["admission"].passed and not receipt.admitted
    assert receipt.read_only and receipt.wrote_effects == ()
    assert result["selected_task_ids"] == [TASK_OFFSET] and result["current_facts_count"] == 1
    assert result["operational_model"]["status"] == "model_refuted"
    assert result["operational_model"]["source_identity_proved"] is True
    assert result["operational_model"]["requested_model_theorem_proved"] is False
    assert result["operational_model"]["kernel_checked_model"] is True
    assert result["capacity_observation"]["cpu_slots"] >= 1
    assert result["capacity_observation"]["process_slots"] >= 1
    assert result["capacity_observation"]["memory_bytes"] >= 1024**3
    assert result["capacity_binding"]["selected_head"] == prepared["expected_head"].to_dict()
    assert result["reservation_released_on_return"] is True
    assert result["training_steps_during_preview"] == result["planning_model_calls"] == 0
    for name in adapter._FALSE:
        assert result[name] is False
    assert (prepared["repository"] / "calc.py").read_bytes() == original
    assert_released(prepared)


def test_same_git_head_successor_proves_model_and_has_explicit_no_work(prepared, finite_tools):
    old = prepared["expected_head"]
    git_head = finite_git(prepared["repository"], "rev-parse", "HEAD")
    (prepared["repository"] / "calc.py").write_bytes(finite_source(2))
    assert finite_git(prepared["repository"], "rev-parse", "HEAD") == git_head
    with pytest.raises(StaleCodebaseError):
        run(prepared, finite_tools)
    successor = prepared["index"].prepare_current(prepared["repository"],
        repository_id=old.repository_id, operation_id="capacity-preview-successor",
        expected_head=old, scheduler=prepared["scheduler"]).head
    prepared["expected_head"] = successor
    result = run(prepared, finite_tools, output=prepared["output"].parent / "new-preview")
    assert result["current_facts_count"] == 2 and result["selected_task_ids"] == []
    assert result["operational_model"]["status"] == "model_proved"
    assert result["operational_model"]["requested_model_theorem_proved"] is True
    assert result["execution_plan"]["status"] == "no_execution_requested"
    assert result["execution_plan"]["admitted"] is False
    assert result["capacity_compilation_request"] is None
    assert_released(prepared)


@pytest.mark.parametrize("field", ["prompt_source_cid", "repository_root", "scope_paths", "budget", "feature_context"])
def test_exact_request_and_native_feature_selection_required(prepared, finite_tools, field):
    args = preview_arguments(prepared, finite_tools)
    request = args["request"]
    if field == "prompt_source_cid":
        request = replace(request, prompt_source_cid=request.roots.configuration_root)
    elif field == "repository_root":
        request = replace(request, repository_root=str(prepared["repository"].parent))
    elif field == "scope_paths":
        request = replace(request, scope_paths=("different.py",))
    elif field == "budget":
        request = replace(request, budget=replace(request.budget, max_model_calls=1))
    else:
        args["feature_context"] = {"mode": "frozen", "version_id": "caller-model"}
        args["model_registry"] = object()
    args["request"] = request
    with pytest.raises(adapter.FiniteIntegerPlanPreviewError):
        adapter.preview_capacity_bound_finite_integer_plan(**args)
    assert not prepared["output"].exists()
    assert_released(prepared)


def test_cancellation_before_owned_output_or_training(prepared, finite_tools):
    args = preview_arguments(prepared, finite_tools)
    cancellation = threading.Event()
    cancellation.set()
    args["owner"] = replace(args["owner"], cancel_event=cancellation)
    with pytest.raises(LeaseCancelledError):
        adapter.preview_capacity_bound_finite_integer_plan(**args)
    assert_released(prepared)


def test_policy_callback_cannot_change_model_olean_before_return(prepared, finite_tools):
    args = preview_arguments(prepared, finite_tools)
    changed = []
    def observe(request):
        path = prepared["output"] / "operational-model" / "IntegerModel.olean"
        if path.exists() and not changed:
            path.write_bytes(path.read_bytes() + b"tamper")
            changed.append(True)
        return request.roots
    args["policy_observer"] = observe
    with pytest.raises(ValueError, match="artifact|model|bytes"):
        adapter.preview_capacity_bound_finite_integer_plan(**args)
    assert changed
    assert_released(prepared)


def test_after_release_model_observation_cannot_modify_previously_checked_finite_artifact(prepared, finite_tools, monkeypatch):
    released, changed = [], []
    original_observe = prepared["index"].observe_current
    original_reserve = capacity_module.reserve_finite_integer_capacity
    from contextlib import contextmanager
    @contextmanager
    def reserve(**kwargs):
        with original_reserve(**kwargs) as held:
            yield held
        released.append(True)
    def observe(*args, **kwargs):
        result = original_observe(*args, **kwargs)
        path = prepared["output"] / "finite-observation" / "FiniteInteger.olean"
        if released and not changed:
            path.write_bytes(path.read_bytes() + b"final-tamper")
            changed.append(True)
        return result
    monkeypatch.setattr(capacity_module, "reserve_finite_integer_capacity", reserve)
    monkeypatch.setattr(prepared["index"], "observe_current", observe)
    with pytest.raises(ValueError, match="artifact|bytes"):
        run(prepared, finite_tools)
    assert released and changed
    assert_released(prepared)


def test_real_parent_budget_keeps_capacity_and_source_as_siblings(prepared, finite_tools):
    args = preview_arguments(prepared, finite_tools)
    with prepared["scheduler"].acquire(ResourceLane.ORCHESTRATION, cpu_slots=2,
            memory_mb=2048, child_process_slots=2, timeout=1) as parent:
        args["owner"] = replace(args["owner"], scheduler=None, parent_lease=parent)
        result = adapter.preview_capacity_bound_finite_integer_plan(**args)
        assert result["capacity_binding"]["reservation"]["parent_lease_id"] == parent.lease_id
        assert not parent.released
        assert prepared["scheduler"].snapshot()["active_lease_count"] == 1
    assert_released(prepared)


def test_selected_off_context_never_loads_fits_or_infers(prepared, finite_tools, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.planning import codebase_feature_context as features
    from ipfs_datasets_py.duckdb_control.autoencoder_registry import AutoencoderRegistry
    from ipfs_datasets_py.logic.software_contracts import codebase_source_training as training
    args = preview_arguments(prepared, finite_tools)
    registry = AutoencoderRegistry(prepared["output"].parent / "model.duckdb",
        prepared["output"].parent / "model-artifacts")
    try:
        def forbidden(*args, **kwargs):
            pytest.fail("model-off preview entered a model load/fit/inference path")
        for name in ("load_codebase_feature_training", "train_current_codebase_features", "infer_current_codebase_features"):
            monkeypatch.setattr(training, name, forbidden)
        context = features.prepare_codebase_feature_context(owner=args["owner"], registry=registry,
            output=prepared["output"].parent / "off-context")
        args.update(feature_context=context, model_registry=registry)
        result = adapter.preview_capacity_bound_finite_integer_plan(**args)
        assert result["feature_context"] == context.material_binding
        assert result["feature_context"]["actual_training_delta"] == 0
    finally:
        registry.close()
    assert_released(prepared)


def test_last_feature_validation_cannot_tamper_already_checked_model(prepared, finite_tools, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.planning import codebase_feature_context as features
    from ipfs_datasets_py.duckdb_control.autoencoder_registry import AutoencoderRegistry
    args = preview_arguments(prepared, finite_tools)
    registry = AutoencoderRegistry(prepared["output"].parent / "model.duckdb",
        prepared["output"].parent / "model-artifacts")
    released, changed = [], []
    from contextlib import contextmanager
    original_reserve = capacity_module.reserve_finite_integer_capacity
    original_verify = features.verify_current_context
    @contextmanager
    def reserve(**kwargs):
        with original_reserve(**kwargs) as held:
            yield held
        released.append(True)
    def verify(*args, **kwargs):
        result = original_verify(*args, **kwargs)
        if released and not changed:
            path = prepared["output"] / "operational-model" / "IntegerModel.olean"
            path.write_bytes(path.read_bytes() + b"last-feature-tamper")
            changed.append(True)
        return result
    try:
        context = features.prepare_codebase_feature_context(owner=args["owner"], registry=registry,
            output=prepared["output"].parent / "off-context")
        args.update(feature_context=context, model_registry=registry)
        monkeypatch.setattr(capacity_module, "reserve_finite_integer_capacity", reserve)
        monkeypatch.setattr(features, "verify_current_context", verify)
        with pytest.raises(ValueError, match="artifact|bytes"):
            adapter.preview_capacity_bound_finite_integer_plan(**args)
        assert released and changed
    finally:
        registry.close()
    assert_released(prepared)


def test_final_open_descriptor_cannot_hide_atomic_path_replacement(prepared, finite_tools, monkeypatch):
    released, changed = [], []
    from contextlib import contextmanager
    original_reserve = capacity_module.reserve_finite_integer_capacity
    original_open = adapter.os.open
    target = prepared["output"] / "finite-observation" / "FiniteInteger.olean"
    @contextmanager
    def reserve(**kwargs):
        with original_reserve(**kwargs) as held:
            yield held
        released.append(True)
    def opened(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        if released and not changed and Path(path) == target:
            replacement = target.with_suffix(".replacement")
            replacement.write_bytes(target.read_bytes() + b"replaced-after-open")
            replacement.replace(target)
            changed.append(True)
        return descriptor
    monkeypatch.setattr(capacity_module, "reserve_finite_integer_capacity", reserve)
    monkeypatch.setattr(adapter.os, "open", opened)
    with pytest.raises(ValueError, match="artifact|bytes"):
        run(prepared, finite_tools)
    assert released and changed
    assert_released(prepared)


@pytest.mark.parametrize("mutation", ["source", "manifest", "ast", "addition", "git_head"])
def test_last_genuine_observation_cannot_hide_source_or_native_cas_edit(prepared, finite_tools, monkeypatch, mutation):
    released, changed, final_observations = [], [], []
    from contextlib import contextmanager
    original_reserve = capacity_module.reserve_finite_integer_capacity
    original_observe = prepared["index"].observe_current
    head = prepared["expected_head"]
    manifest = prepared["index"].load(head.manifest_cid)
    @contextmanager
    def reserve(**kwargs):
        with original_reserve(**kwargs) as held:
            yield held
        released.append(True)
    def observe(*args, **kwargs):
        result = original_observe(*args, **kwargs)
        if released:
            final_observations.append(True)
        # The real observer has finished its source/head checks. Mutate only
        # after the model validator's final exit observation, then return its
        # genuine old result. Another observer callback is not a closure.
        if len(final_observations) == 2 and not changed:
            if mutation == "source":
                (prepared["repository"] / "calc.py").write_bytes(finite_source(9))
            elif mutation in {"manifest", "ast"}:
                cid = head.manifest_cid if mutation == "manifest" else manifest.units[0].ast_cid
                path = prepared["index"].artifacts.path_for(cid)
                path.write_bytes(path.read_bytes() + b"final-cas-tamper")
            elif mutation == "addition":
                (prepared["repository"] / "added.py").write_bytes(b"unexpected = 1\n")
            else:
                (prepared["repository"] / ".git" / "HEAD").write_bytes(b"ref: refs/heads/changed\n")
            changed.append(True)
        return result
    monkeypatch.setattr(capacity_module, "reserve_finite_integer_capacity", reserve)
    monkeypatch.setattr(prepared["index"], "observe_current", observe)
    with pytest.raises(ValueError, match="source|custody|inventory|CAS|Git|artifact|bytes|identity"):
        run(prepared, finite_tools)
    assert changed and len(final_observations) == 2
    assert_released(prepared)


def test_last_frozen_verification_cannot_corrupt_original_native_checkpoint(prepared, finite_tools, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.planning import codebase_feature_context as features
    from ipfs_datasets_py.duckdb_control.autoencoder_registry import AutoencoderRegistry
    from ipfs_datasets_py.logic.software_contracts.codebase_source_training import CodebaseTrainingSelection
    repository, old = prepared["repository"], prepared["expected_head"]
    for name, prefix, offset in (("known.py", "known variant", 2), ("tune.py", "fixed tuning", 1),
                                ("canary.py", "fixed diagnostic", 1)):
        (repository / name).write_bytes(("# " + prefix + "\n").encode() + finite_source(offset))
    prepared["expected_head"] = prepared["index"].prepare_current(repository,
        repository_id=old.repository_id, operation_id="native-feature-custody-corpus",
        expected_head=old, scheduler=prepared["scheduler"]).head
    args = preview_arguments(prepared, finite_tools)
    registry = AutoencoderRegistry(prepared["output"].parent / "model.duckdb",
        prepared["output"].parent / "model-artifacts")
    released, changed = [], []
    from contextlib import contextmanager
    original_reserve = capacity_module.reserve_finite_integer_capacity
    original_verify = features.verify_current_context
    @contextmanager
    def reserve(**kwargs):
        with original_reserve(**kwargs) as held:
            yield held
        released.append(True)
    def verify(*args, **kwargs):
        result = original_verify(*args, **kwargs)
        if released and not changed:
            version = kwargs["registry"].get_version(kwargs["context"].version_id)
            path = kwargs["registry"].artifact_path(version["artifact"])
            path.write_bytes(path.read_bytes() + b"native-checkpoint-tamper")
            changed.append(True)
        return result
    try:
        selections = tuple(CodebaseTrainingSelection(name, role, contracts=()) for name, role in
            (("calc.py", "train"), ("known.py", "train"), ("tune.py", "tune"), ("canary.py", "canary")))
        trained = features.prepare_codebase_feature_context(owner=args["owner"], registry=registry,
            mode="train", output=prepared["output"].parent / "training-context",
            selections=selections, operation_id="original-checkpoint-custody", epochs=1, learning_rate=.002)
        frozen = features.prepare_codebase_feature_context(owner=args["owner"], registry=registry,
            mode="frozen", output=prepared["output"].parent / "frozen-context", version_id=trained.version_id)
        args.update(feature_context=frozen, model_registry=registry)
        monkeypatch.setattr(capacity_module, "reserve_finite_integer_capacity", reserve)
        monkeypatch.setattr(features, "verify_current_context", verify)
        with pytest.raises(ValueError, match="artifact|bytes|checkpoint"):
            adapter.preview_capacity_bound_finite_integer_plan(**args)
        assert released and changed
    finally:
        registry.close()
    assert_released(prepared)


def test_frozen_native_features_verify_at_entry_and_exit_with_same_tasks(prepared, finite_tools, monkeypatch):
    from hashlib import sha256
    from ipfs_accelerate_py.agent_supervisor.planning import codebase_feature_context as features
    from ipfs_datasets_py.duckdb_control.autoencoder_registry import AutoencoderRegistry
    from ipfs_datasets_py.logic.software_contracts.codebase_source_training import CodebaseTrainingSelection
    repository, old = prepared["repository"], prepared["expected_head"]
    for name, prefix, offset in (("known.py", "known variant", 2), ("tune.py", "fixed tuning", 1),
                                ("canary.py", "fixed diagnostic", 1)):
        (repository / name).write_bytes(("# " + prefix + "\n").encode() + finite_source(offset))
    prepared["expected_head"] = prepared["index"].prepare_current(repository,
        repository_id=old.repository_id, operation_id="native-feature-two-verifications",
        expected_head=old, scheduler=prepared["scheduler"]).head
    args = preview_arguments(prepared, finite_tools)
    baseline = adapter.preview_capacity_bound_finite_integer_plan(**args)
    args["output"] = prepared["output"].parent / "frozen-preview"
    registry = AutoencoderRegistry(prepared["output"].parent / "model.duckdb",
        prepared["output"].parent / "model-artifacts")
    verified, committed_policies = [], []
    original_verify = features.verify_current_context
    original_freeze = adapter.freeze_plan_create_input_snapshot

    def inventory():
        with registry._transaction() as cx:
            names = cx.execute("SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='autoencoder_control' ORDER BY table_name").fetchall()
            tables = {name: sorted(cx.execute('SELECT * FROM autoencoder_control."' +
                name.replace('"', '""') + '"').fetchall(), key=repr) for (name,) in names}
        artifacts = {str(path): sha256(path.read_bytes()).hexdigest()
            for path in registry.artifact_root.rglob("*") if path.is_file()}
        return tables, artifacts

    def verify(*values, **kwargs):
        result = original_verify(*values, **kwargs)
        verified.append((kwargs["owner"].expected_head, kwargs["context"].cid))
        return result

    def freeze(*values, **kwargs):
        snapshot = original_freeze(*values, **kwargs)
        committed_policies.append(kwargs["materials"].extra["feature_validation_policy"])
        return snapshot

    try:
        selections = tuple(CodebaseTrainingSelection(name, role, contracts=()) for name, role in
            (("calc.py", "train"), ("known.py", "train"), ("tune.py", "tune"), ("canary.py", "canary")))
        trained = features.prepare_codebase_feature_context(owner=args["owner"], registry=registry,
            mode="train", output=prepared["output"].parent / "training-context",
            selections=selections, operation_id="two-verifications", epochs=1, learning_rate=.002)
        frozen = features.prepare_codebase_feature_context(owner=args["owner"], registry=registry,
            mode="frozen", output=prepared["output"].parent / "frozen-context", version_id=trained.version_id)
        before = inventory()
        args.update(feature_context=frozen, model_registry=registry)
        monkeypatch.setattr(features, "verify_current_context", verify)
        monkeypatch.setattr(adapter, "freeze_plan_create_input_snapshot", freeze)
        result = adapter.preview_capacity_bound_finite_integer_plan(**args)
        assert verified == [(prepared["expected_head"], frozen.cid)] * 2
        assert inventory() == before
        assert result["current_facts_count"] == baseline["current_facts_count"] == 1
        assert result["selected_task_ids"] == baseline["selected_task_ids"] == [TASK_OFFSET]
        assert result["declared_task_requirement_ids"] == baseline["declared_task_requirement_ids"]
        policy = result["feature_validation_policy"]
        assert policy["numerical_verification_points"] == ["entry", "after_reservation_release"]
        assert committed_policies == [policy]
        assert result["input_snapshot"]["material_binding"]["reuse_supported"]
        assert "extra" in result["input_snapshot"]["material_binding"]["field_digests"]
        assert result["execution_plan"]["admitted"] is True and not result["production_admitted"]
        assert result["training_steps_during_preview"] == 0
    finally:
        registry.close()
    assert_released(prepared)
