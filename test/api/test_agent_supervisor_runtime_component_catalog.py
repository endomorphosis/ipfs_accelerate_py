"""Contract tests for CID-bound SwissKnife runtime component discovery."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.runtime_component_catalog import (
    DuplicateRuntimeComponentError,
    DuplicateRuntimeRouteError,
    ImplementationAuthorityKind,
    MissingRuntimeRouteError,
    RuntimeAuthorityError,
    RuntimeCIDError,
    RuntimeComponentKind,
    RuntimeRouteKind,
    build_runtime_component_catalog,
    load_runtime_component_catalog,
    validate_runtime_sources,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CATALOG_PATH = REPOSITORY_ROOT / "config/swissknife_runtime_contract_scope.json"
SWISSKNIFE_ROOT = REPOSITORY_ROOT / "swissknife"


def _checked_in_payload() -> dict[str, object]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def _unmaterialized_payload() -> dict[str, object]:
    payload = _checked_in_payload()
    payload.pop("catalogCid", None)
    payload.pop("routes", None)
    for component in payload["components"]:
        component.pop("rootCid", None)
        component["authority"].pop("authorityCid", None)
    for profile in payload["routeProfiles"]:
        profile.pop("profileCid", None)
    return payload


def test_checked_in_catalog_is_complete_cid_bound_and_resolves_sources() -> None:
    catalog = load_runtime_component_catalog(CATALOG_PATH)

    validate_runtime_sources(catalog, SWISSKNIFE_ROOT)
    primary_kinds = {
        component.kind
        for component in catalog.components
        if component.authority.kind is ImplementationAuthorityKind.PRIMARY
    }
    assert primary_kinds == set(RuntimeComponentKind)
    assert len(catalog.routes) == len(catalog.components) * len(RuntimeRouteKind)
    assert catalog.catalog_cid.startswith("b")

    for component in catalog.components:
        assert component.root_cid.startswith("b")
        assert component.authority.authority_cid.startswith("b")
        for kind in RuntimeRouteKind:
            route = catalog.route(component.component_id, kind)
            assert route.component_id == component.component_id
            assert route.component_root_cid == component.root_cid
            assert route.kind is kind
            assert route.route_cid.startswith("b")


def test_alternate_implementation_has_typed_canonical_authority() -> None:
    catalog = load_runtime_component_catalog(CATALOG_PATH)
    alternate = catalog.component("model-server-patched")
    canonical = catalog.component(alternate.authority.canonical_component_id)

    assert alternate.authority.kind is ImplementationAuthorityKind.ALTERNATE
    assert canonical.authority.kind is ImplementationAuthorityKind.PRIMARY
    assert alternate.kind is canonical.kind is RuntimeComponentKind.MODEL_SERVER
    assert alternate.root_cid != canonical.root_cid


def test_route_normalization_never_joins_on_display_name() -> None:
    payload = _unmaterialized_payload()
    for component in payload["components"]:
        component["displayName"] = "intentionally duplicated descriptive name"

    catalog = build_runtime_component_catalog(payload)

    assert len(catalog.routes) == len(catalog.components) * len(RuntimeRouteKind)
    assert len({route.route_cid for route in catalog.routes}) == len(catalog.routes)
    for component in catalog.components:
        routes = [
            route
            for route in catalog.routes
            if route.component_id == component.component_id
            and route.component_root_cid == component.root_cid
        ]
        assert {route.kind for route in routes} == set(RuntimeRouteKind)


def test_missing_and_duplicate_profile_routes_fail_closed() -> None:
    missing = _unmaterialized_payload()
    missing["routeProfiles"][0]["routes"].pop()
    with pytest.raises(MissingRuntimeRouteError):
        build_runtime_component_catalog(missing)

    duplicate = _unmaterialized_payload()
    duplicate["routeProfiles"][0]["routes"].append(
        copy.deepcopy(duplicate["routeProfiles"][0]["routes"][0])
    )
    with pytest.raises(DuplicateRuntimeRouteError):
        build_runtime_component_catalog(duplicate)


def test_missing_profile_and_duplicate_component_fail_closed() -> None:
    missing = _unmaterialized_payload()
    missing["components"][0]["routeProfileId"] = "undeclared-profile"
    with pytest.raises(MissingRuntimeRouteError):
        build_runtime_component_catalog(missing)

    duplicate = _unmaterialized_payload()
    duplicate["components"].append(copy.deepcopy(duplicate["components"][0]))
    with pytest.raises(DuplicateRuntimeComponentError):
        build_runtime_component_catalog(duplicate)


def test_alternate_authority_cannot_target_a_different_runtime_kind() -> None:
    payload = _unmaterialized_payload()
    alternate = next(
        component
        for component in payload["components"]
        if component["componentId"] == "model-server-patched"
    )
    alternate["authority"]["canonicalComponentId"] = "scheduler"

    with pytest.raises(RuntimeAuthorityError):
        build_runtime_component_catalog(payload)


def test_stored_cids_reject_tampering() -> None:
    payload = _checked_in_payload()
    payload["components"][0]["rootCid"] = "bafystale"

    with pytest.raises(RuntimeCIDError):
        build_runtime_component_catalog(payload, require_stored_cids=True)
