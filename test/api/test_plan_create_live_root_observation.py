"""Opt-in service observations fence every preview and cached return."""

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import (
    LIVE_ROOT_OBSERVATION_PROFILE,
    ROOT_OBSERVATION_PROFILE_FIELD,
    PlanCreateService,
    PlanCreateServiceError,
    PlanCreateStaleRootError,
    PlanCreateVerdict,
    create_default_plan_create_service,
    freeze_plan_create_input_snapshot,
)
from test.api.test_plan_create_semantic_input_identity import _cid, _materials, _request


def _strict(observer, store=None):
    return PlanCreateService(root_observer=observer,
        require_live_root_observation=True, receipt_store=store)


def _stale(request):
    return replace(request.roots, dirty_worktree_root=_cid("edited-live-source"))


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_strict_policy_requires_exact_boolean(value):
    with pytest.raises(PlanCreateServiceError, match="exact bool"):
        PlanCreateService(require_live_root_observation=value)


@pytest.mark.parametrize("observer", [None, object()])
def test_static_roots_cannot_replace_a_missing_callable_live_observer(observer):
    request, materials, store = _request(), _materials(), {}
    materials.current_roots = request.roots
    service = _strict(observer, store)
    with pytest.raises(PlanCreateStaleRootError, match="callable live observer"):
        service.preview_create(request, materials=materials)
    assert not store and not service._preview_by_key


@pytest.mark.parametrize("incomplete", ["none", "empty", "partial", "missing-configuration"])
def test_strict_profile_rejects_incomplete_live_authority_observation(incomplete):
    request, store = _request(), {}
    observed = {"none": None, "empty": {},
                "partial": {"policy_root": request.roots.policy_root}}.get(incomplete)
    if incomplete == "missing-configuration":
        observed = request.roots.to_dict()
        observed.pop("configuration_root")
    service = _strict(lambda _: observed, store)
    with pytest.raises(PlanCreateStaleRootError, match="complete authority roots"):
        service.preview_create(request, materials=_materials())
    assert not store and not service._preview_by_key


def test_matching_static_roots_cannot_shadow_stale_live_roots():
    request, materials, store, calls = _request(), _materials(), {}, []
    materials.current_roots = request.roots
    def observe(bound):
        calls.append(bound.request_cid)
        return _stale(bound)
    service = _strict(observe, store)
    with pytest.raises(PlanCreateStaleRootError, match="stale root/policy"):
        service.preview_create(request, materials=materials)
    assert calls == [request.request_cid]
    assert not store and not service._preview_by_key


def test_stale_static_binding_is_checked_in_addition_to_live_observation():
    request, materials, store, calls = _request(), _materials(), {}, []
    materials.current_roots = _stale(request)
    def observe(bound):
        calls.append(bound.request_cid)
        return bound.roots
    service = _strict(observe, store)
    with pytest.raises(PlanCreateStaleRootError, match="stale root/policy"):
        service.preview_create(request, materials=materials)
    assert calls == [request.request_cid]
    assert not store and not service._preview_by_key


@pytest.mark.parametrize("drift_call", [2, 3], ids=["before-admission", "before-publication"])
def test_drift_during_preview_cannot_publish_or_cache_a_receipt(monkeypatch, drift_call):
    request, materials, store, calls, admissions = _request(), _materials(), {}, [], []
    materials.current_roots = request.roots
    def observe(bound):
        calls.append(bound.request_cid)
        return _stale(bound) if len(calls) == drift_call else bound.roots
    service = _strict(observe, store)
    original = service._stage_admission
    def admission(*args, **kwargs):
        admissions.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(service, "_stage_admission", admission)
    with pytest.raises(PlanCreateStaleRootError, match="stale root/policy"):
        service.preview_create(request, materials=materials)
    assert len(calls) == drift_call
    assert len(admissions) == (1 if drift_call == 3 else 0)
    assert not store and not service._preview_by_key


@pytest.mark.parametrize("cache_drift_call", [4, 5], ids=["cache-entry", "cache-exit"])
def test_cached_return_requires_fresh_entry_and_exit_observations(cache_drift_call):
    request, materials, store, calls = _request(), _materials(), {}, []
    def observe(bound):
        calls.append(bound.request_cid)
        return _stale(bound) if len(calls) == cache_drift_call else bound.roots
    service = _strict(observe, store)
    initial = service.preview_create(request, materials=materials)
    retained_store, retained_cache = dict(store), dict(service._preview_by_key)
    with pytest.raises(PlanCreateStaleRootError, match="stale root/policy"):
        service.preview_create(request, materials=materials)
    assert len(calls) == cache_drift_call
    assert store == retained_store and service._preview_by_key == retained_cache
    assert len(store) == 1 and initial.receipt_cid in store


def test_strict_service_binds_policy_without_mutating_caller_and_fences_cache():
    request, materials, store, calls = _request(), _materials(), {}, []
    materials.extra = {"owner_context": "source-bound"}
    original_extra = dict(materials.extra)
    def observe(bound):
        calls.append(bound.request_cid)
        return bound.roots.to_dict()
    service = _strict(observe, store)
    preview = service.preview_create(request, materials=materials)
    assert preview.verdict is PlanCreateVerdict.REVIEW_ONLY
    assert len(calls) == 3
    assert preview.read_only and not preview.wrote_effects
    strict_materials = replace(materials, extra={**materials.extra,
        ROOT_OBSERVATION_PROFILE_FIELD: LIVE_ROOT_OBSERVATION_PROFILE})
    strict_snapshot = freeze_plan_create_input_snapshot(request, materials=strict_materials)
    assert preview.input_snapshot_cid == strict_snapshot.snapshot_cid
    assert preview.input_snapshot_cid != freeze_plan_create_input_snapshot(request,
        materials=materials).snapshot_cid
    assert materials.extra == original_extra
    assert service.preview_create(request, materials=materials) is preview
    assert len(calls) == 5 and len(store) == 1


def test_legacy_service_rejects_strict_material_profile_instead_of_downgrading():
    request, materials, store = _request(), _materials(), {}
    materials.extra = {ROOT_OBSERVATION_PROFILE_FIELD: LIVE_ROOT_OBSERVATION_PROFILE}
    service = PlanCreateService(root_observer=lambda bound: bound.roots, receipt_store=store)
    with pytest.raises(PlanCreateServiceError, match="strict service policy"):
        service.preview_create(request, materials=materials)
    assert not store and not service._preview_by_key


def test_strict_service_rejects_a_conflicting_reserved_profile():
    materials, store = _materials(), {}
    materials.extra = {ROOT_OBSERVATION_PROFILE_FIELD: "unrecognized-profile"}
    service = _strict(lambda bound: bound.roots, store)
    with pytest.raises(PlanCreateServiceError, match="unsupported repository root"):
        service.preview_create(_request(), materials=materials)
    assert not store and not service._preview_by_key


def test_legacy_profile_retains_static_root_precedence_and_snapshot_identity():
    request, materials, calls = _request(), _materials(), []
    materials.current_roots = request.roots
    def observe(bound):
        calls.append(bound.request_cid)
        return _stale(bound)
    preview = PlanCreateService(root_observer=observe).preview_create(request, materials=materials)
    assert calls == []
    assert preview.input_snapshot_cid == freeze_plan_create_input_snapshot(request,
        materials=materials).snapshot_cid
    assert ROOT_OBSERVATION_PROFILE_FIELD not in materials.extra


def test_default_factory_forwards_the_explicit_strict_policy():
    service = create_default_plan_create_service(build_analysis_factory=False,
        root_observer=lambda bound: bound.roots, require_live_root_observation=True)
    assert service.require_live_root_observation is True
    assert service.preview_create(_request(), materials=_materials()).read_only
