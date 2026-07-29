"""CID-bound catalog of the SwissKnife runtime component and route roots.

Interface: ``RuntimeComponentCatalog@1``

Inventories canonical and alternate entrypoints for the model server,
orchestrator, scheduler, and supervisor.  The catalog deliberately joins
routes to a component id *and* component-root CID.  Display names are
descriptive only and can never become authority for a runtime join.

Normative rules:

* Every primary component root is complete and independently CID-bound.
* Alternate implementations carry typed authority that points at a primary
  of the same runtime kind.
* Connector / launcher / health / list / call routes normalize without
  name-only joins.
* Missing or duplicate components, profiles, or routes fail closed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Final, Mapping, Sequence

from .content_identity_bridge import identify_strict_artifact


RUNTIME_COMPONENT_CATALOG_INTERFACE: Final = "RuntimeComponentCatalog@1"
CATALOG_VERSION: Final = "1"

AUTHORITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-implementation-authority@1"
)
COMPONENT_ROOT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-component-root@1"
)
ROUTE_PROFILE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-route-profile@1"
)
NORMALIZED_ROUTE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/normalized-runtime-route@1"
)
RUNTIME_CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/runtime-component-catalog@1"
)


class RuntimeComponentCatalogError(ValueError):
    """Base class for fail-closed catalog validation errors."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "runtime_component_catalog_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = dict(details or {})


class MissingRuntimeComponentError(RuntimeComponentCatalogError):
    """A required runtime component root is absent."""


class DuplicateRuntimeComponentError(RuntimeComponentCatalogError):
    """A component identity or primary authority is duplicated."""


class MissingRuntimeRouteError(RuntimeComponentCatalogError):
    """A required route or referenced route profile is absent."""


class DuplicateRuntimeRouteError(RuntimeComponentCatalogError):
    """A route identity or route kind is duplicated."""


class RuntimeAuthorityError(RuntimeComponentCatalogError):
    """Implementation authority is invalid or unresolved."""


class RuntimeCIDError(RuntimeComponentCatalogError):
    """A stored CID is absent or does not match its canonical preimage."""


class RuntimeSourceError(RuntimeComponentCatalogError):
    """A cataloged source file or symbol cannot be found."""


class RuntimeComponentKind(str, Enum):
    """Required SwissKnife runtime roles."""

    MODEL_SERVER = "model_server"
    ORCHESTRATOR = "orchestrator"
    SCHEDULER = "scheduler"
    SUPERVISOR = "supervisor"


class RuntimeRouteKind(str, Enum):
    """Routes every component root must normalize."""

    CONNECTOR = "connector"
    LAUNCHER = "launcher"
    HEALTH = "health"
    LIST = "list"
    CALL = "call"


class ImplementationAuthorityKind(str, Enum):
    """Whether an implementation is the primary root or a typed alternative."""

    PRIMARY = "primary"
    ALTERNATE = "alternate"


def _cid(payload: Mapping[str, Any]) -> str:
    return identify_strict_artifact(payload).cid


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeComponentCatalogError(
            f"{field_name} must be an object",
            reason_code="invalid_catalog_field",
            details={"field": field_name},
        )
    return value


def _sequence(value: object, field_name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RuntimeComponentCatalogError(
            f"{field_name} must be an array",
            reason_code="invalid_catalog_field",
            details={"field": field_name},
        )
    return value


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeComponentCatalogError(
            f"{field_name} must be a nonempty string",
            reason_code="invalid_catalog_field",
            details={"field": field_name},
        )
    return value


def _source_path(value: object, field_name: str) -> str:
    source = _text(value, field_name)
    parsed = PurePosixPath(source)
    if parsed.is_absolute() or ".." in parsed.parts or source != parsed.as_posix():
        raise RuntimeComponentCatalogError(
            f"{field_name} must be a normalized relative POSIX path",
            reason_code="invalid_source_path",
            details={"field": field_name, "value": source},
        )
    return source


def _enum(enum_type: type[Enum], value: object, field_name: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeComponentCatalogError(
            f"{field_name} has an unsupported value",
            reason_code="invalid_catalog_enum",
            details={"field": field_name, "value": value},
        ) from exc


def _verified_cid(
    data: Mapping[str, Any],
    field_name: str,
    preimage: Mapping[str, Any],
    *,
    require_stored_cids: bool,
) -> str:
    expected = _cid(preimage)
    stored = data.get(field_name)
    if stored is None and not require_stored_cids:
        return expected
    if not isinstance(stored, str) or not stored:
        raise RuntimeCIDError(
            f"{field_name} is required",
            reason_code="runtime_cid_missing",
            details={"field": field_name},
        )
    if stored != expected:
        raise RuntimeCIDError(
            f"{field_name} does not match its canonical preimage",
            reason_code="runtime_cid_mismatch",
            details={"field": field_name, "stored": stored, "expected": expected},
        )
    return stored


@dataclass(frozen=True)
class ImplementationAuthority:
    kind: ImplementationAuthorityKind
    canonical_component_id: str
    decision: str
    source_path: str
    authority_cid: str

    def preimage(self) -> dict[str, Any]:
        return {
            "schema": AUTHORITY_SCHEMA,
            "kind": self.kind.value,
            "canonicalComponentId": self.canonical_component_id,
            "decision": self.decision,
            "sourcePath": self.source_path,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.preimage(), "authorityCid": self.authority_cid}


@dataclass(frozen=True)
class RuntimeComponentRoot:
    component_id: str
    display_name: str
    kind: RuntimeComponentKind
    implementation_symbol: str
    source_path: str
    route_profile_id: str
    authority: ImplementationAuthority
    root_cid: str

    def preimage(self) -> dict[str, Any]:
        return {
            "schema": COMPONENT_ROOT_SCHEMA,
            "componentId": self.component_id,
            "displayName": self.display_name,
            "kind": self.kind.value,
            "implementationSymbol": self.implementation_symbol,
            "sourcePath": self.source_path,
            "routeProfileId": self.route_profile_id,
            "authority": self.authority.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.preimage(), "rootCid": self.root_cid}


@dataclass(frozen=True)
class RuntimeRouteSpec:
    kind: RuntimeRouteKind
    transport: str
    selector: str
    source_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "transport": self.transport,
            "selector": self.selector,
            "sourcePath": self.source_path,
        }


@dataclass(frozen=True)
class RuntimeRouteProfile:
    profile_id: str
    routes: tuple[RuntimeRouteSpec, ...]
    profile_cid: str

    def preimage(self) -> dict[str, Any]:
        return {
            "schema": ROUTE_PROFILE_SCHEMA,
            "profileId": self.profile_id,
            "routes": [route.to_dict() for route in self.routes],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.preimage(), "profileCid": self.profile_cid}


@dataclass(frozen=True)
class NormalizedRuntimeRoute:
    route_id: str
    component_id: str
    component_root_cid: str
    route_profile_cid: str
    kind: RuntimeRouteKind
    transport: str
    selector: str
    source_path: str
    route_cid: str

    def preimage(self) -> dict[str, Any]:
        return {
            "schema": NORMALIZED_ROUTE_SCHEMA,
            "routeId": self.route_id,
            "componentId": self.component_id,
            "componentRootCid": self.component_root_cid,
            "routeProfileCid": self.route_profile_cid,
            "kind": self.kind.value,
            "transport": self.transport,
            "selector": self.selector,
            "sourcePath": self.source_path,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.preimage(), "routeCid": self.route_cid}


@dataclass(frozen=True)
class RuntimeComponentCatalog:
    components: tuple[RuntimeComponentRoot, ...]
    route_profiles: tuple[RuntimeRouteProfile, ...]
    routes: tuple[NormalizedRuntimeRoute, ...]
    catalog_cid: str

    def preimage(self) -> dict[str, Any]:
        return {
            "schema": RUNTIME_CATALOG_SCHEMA,
            "components": [component.to_dict() for component in self.components],
            "routeProfiles": [
                profile.to_dict() for profile in self.route_profiles
            ],
            "routes": [route.to_dict() for route in self.routes],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.preimage(), "catalogCid": self.catalog_cid}

    def component(self, component_id: str) -> RuntimeComponentRoot:
        matches = [
            component
            for component in self.components
            if component.component_id == component_id
        ]
        if len(matches) != 1:
            raise MissingRuntimeComponentError(
                f"component id does not resolve uniquely: {component_id}",
                reason_code="component_lookup_failed",
                details={"componentId": component_id, "matches": len(matches)},
            )
        return matches[0]

    def route(
        self,
        component_id: str,
        kind: RuntimeRouteKind | str,
    ) -> NormalizedRuntimeRoute:
        route_kind = _enum(RuntimeRouteKind, kind, "kind")
        component = self.component(component_id)
        matches = [
            route
            for route in self.routes
            if route.component_id == component.component_id
            and route.component_root_cid == component.root_cid
            and route.kind is route_kind
        ]
        if len(matches) != 1:
            raise MissingRuntimeRouteError(
                "component route does not resolve uniquely",
                reason_code="route_lookup_failed",
                details={
                    "componentId": component_id,
                    "componentRootCid": component.root_cid,
                    "kind": route_kind.value,
                    "matches": len(matches),
                },
            )
        return matches[0]


def _parse_authority(
    raw: Mapping[str, Any],
    *,
    require_stored_cids: bool,
) -> ImplementationAuthority:
    kind = _enum(ImplementationAuthorityKind, raw.get("kind"), "authority.kind")
    provisional = ImplementationAuthority(
        kind=kind,
        canonical_component_id=_text(
            raw.get("canonicalComponentId"),
            "authority.canonicalComponentId",
        ),
        decision=_text(raw.get("decision"), "authority.decision"),
        source_path=_source_path(raw.get("sourcePath"), "authority.sourcePath"),
        authority_cid="",
    )
    return ImplementationAuthority(
        **{
            **provisional.__dict__,
            "authority_cid": _verified_cid(
                raw,
                "authorityCid",
                provisional.preimage(),
                require_stored_cids=require_stored_cids,
            ),
        }
    )


def _parse_component(
    raw: Mapping[str, Any],
    *,
    require_stored_cids: bool,
) -> RuntimeComponentRoot:
    provisional = RuntimeComponentRoot(
        component_id=_text(raw.get("componentId"), "componentId"),
        display_name=_text(raw.get("displayName"), "displayName"),
        kind=_enum(RuntimeComponentKind, raw.get("kind"), "kind"),
        implementation_symbol=_text(
            raw.get("implementationSymbol"),
            "implementationSymbol",
        ),
        source_path=_source_path(raw.get("sourcePath"), "sourcePath"),
        route_profile_id=_text(raw.get("routeProfileId"), "routeProfileId"),
        authority=_parse_authority(
            _mapping(raw.get("authority"), "authority"),
            require_stored_cids=require_stored_cids,
        ),
        root_cid="",
    )
    return RuntimeComponentRoot(
        **{
            **provisional.__dict__,
            "root_cid": _verified_cid(
                raw,
                "rootCid",
                provisional.preimage(),
                require_stored_cids=require_stored_cids,
            ),
        }
    )


def _parse_profile(
    raw: Mapping[str, Any],
    *,
    require_stored_cids: bool,
) -> RuntimeRouteProfile:
    profile_id = _text(raw.get("profileId"), "routeProfile.profileId")
    routes: list[RuntimeRouteSpec] = []
    seen: set[RuntimeRouteKind] = set()
    for item in _sequence(raw.get("routes"), f"routeProfiles[{profile_id}].routes"):
        route = _mapping(item, f"routeProfiles[{profile_id}].routes[]")
        kind = _enum(RuntimeRouteKind, route.get("kind"), "route.kind")
        if kind in seen:
            raise DuplicateRuntimeRouteError(
                f"duplicate {kind.value} route in profile {profile_id}",
                reason_code="duplicate_route_kind",
                details={"profileId": profile_id, "kind": kind.value},
            )
        seen.add(kind)
        routes.append(
            RuntimeRouteSpec(
                kind=kind,
                transport=_text(route.get("transport"), "route.transport"),
                selector=_text(route.get("selector"), "route.selector"),
                source_path=_source_path(
                    route.get("sourcePath"),
                    "route.sourcePath",
                ),
            )
        )
    missing = set(RuntimeRouteKind) - seen
    if missing:
        raise MissingRuntimeRouteError(
            f"route profile {profile_id} is incomplete",
            reason_code="missing_route_kind",
            details={
                "profileId": profile_id,
                "missing": sorted(kind.value for kind in missing),
            },
        )
    provisional = RuntimeRouteProfile(
        profile_id=profile_id,
        routes=tuple(routes),
        profile_cid="",
    )
    return RuntimeRouteProfile(
        profile_id=profile_id,
        routes=tuple(routes),
        profile_cid=_verified_cid(
            raw,
            "profileCid",
            provisional.preimage(),
            require_stored_cids=require_stored_cids,
        ),
    )


def build_runtime_component_catalog(
    payload: Mapping[str, Any],
    *,
    require_stored_cids: bool = False,
) -> RuntimeComponentCatalog:
    """Validate and normalize a runtime catalog mapping.

    Unmaterialized fixture mappings may omit CIDs.  Production loading uses
    ``require_stored_cids=True`` so every checked-in root is independently
    revalidated against its canonical preimage.
    """

    if payload.get("schema") not in (None, RUNTIME_CATALOG_SCHEMA):
        raise RuntimeComponentCatalogError(
            "unsupported runtime catalog schema",
            reason_code="unsupported_catalog_schema",
            details={"schema": payload.get("schema")},
        )

    profiles = tuple(
        _parse_profile(
            _mapping(item, "routeProfiles[]"),
            require_stored_cids=require_stored_cids,
        )
        for item in _sequence(payload.get("routeProfiles"), "routeProfiles")
    )
    profile_by_id: dict[str, RuntimeRouteProfile] = {}
    for profile in profiles:
        if profile.profile_id in profile_by_id:
            raise DuplicateRuntimeRouteError(
                f"duplicate route profile id: {profile.profile_id}",
                reason_code="duplicate_route_profile",
                details={"profileId": profile.profile_id},
            )
        profile_by_id[profile.profile_id] = profile

    components = tuple(
        _parse_component(
            _mapping(item, "components[]"),
            require_stored_cids=require_stored_cids,
        )
        for item in _sequence(payload.get("components"), "components")
    )
    by_id: dict[str, RuntimeComponentRoot] = {}
    by_root: dict[str, RuntimeComponentRoot] = {}
    primaries: dict[RuntimeComponentKind, RuntimeComponentRoot] = {}
    for component in components:
        if component.component_id in by_id:
            raise DuplicateRuntimeComponentError(
                f"duplicate component id: {component.component_id}",
                reason_code="duplicate_component_id",
                details={"componentId": component.component_id},
            )
        if component.root_cid in by_root:
            raise DuplicateRuntimeComponentError(
                "duplicate component root CID",
                reason_code="duplicate_component_root",
                details={"rootCid": component.root_cid},
            )
        by_id[component.component_id] = component
        by_root[component.root_cid] = component
        if component.authority.kind is ImplementationAuthorityKind.PRIMARY:
            if component.kind in primaries:
                raise DuplicateRuntimeComponentError(
                    f"duplicate primary {component.kind.value} root",
                    reason_code="duplicate_primary_component",
                    details={"kind": component.kind.value},
                )
            primaries[component.kind] = component

    missing_components = set(RuntimeComponentKind) - set(primaries)
    if missing_components:
        raise MissingRuntimeComponentError(
            "required primary component roots are missing",
            reason_code="missing_primary_component",
            details={
                "missing": sorted(kind.value for kind in missing_components),
            },
        )

    for component in components:
        authority = component.authority
        canonical = by_id.get(authority.canonical_component_id)
        if authority.kind is ImplementationAuthorityKind.PRIMARY:
            valid = canonical is component
        else:
            valid = (
                canonical is not None
                and canonical.authority.kind is ImplementationAuthorityKind.PRIMARY
                and canonical.kind is component.kind
                and canonical is not component
            )
        if not valid:
            raise RuntimeAuthorityError(
                f"invalid authority for component {component.component_id}",
                reason_code="invalid_implementation_authority",
                details={
                    "componentId": component.component_id,
                    "authorityKind": authority.kind.value,
                    "canonicalComponentId": authority.canonical_component_id,
                },
            )
        if authority.source_path != component.source_path:
            raise RuntimeAuthorityError(
                "authority source must bind the component source",
                reason_code="authority_source_mismatch",
                details={"componentId": component.component_id},
            )

    normalized: list[NormalizedRuntimeRoute] = []
    route_keys: set[tuple[str, str, RuntimeRouteKind]] = set()
    for component in components:
        profile = profile_by_id.get(component.route_profile_id)
        if profile is None:
            raise MissingRuntimeRouteError(
                f"unknown route profile: {component.route_profile_id}",
                reason_code="route_profile_missing",
                details={
                    "componentId": component.component_id,
                    "profileId": component.route_profile_id,
                },
            )
        for spec in profile.routes:
            key = (component.component_id, component.root_cid, spec.kind)
            if key in route_keys:
                raise DuplicateRuntimeRouteError(
                    "duplicate normalized route",
                    reason_code="duplicate_normalized_route",
                    details={
                        "componentId": component.component_id,
                        "componentRootCid": component.root_cid,
                        "kind": spec.kind.value,
                    },
                )
            route_keys.add(key)
            route_id = f"{component.component_id}:{spec.kind.value}"
            provisional = NormalizedRuntimeRoute(
                route_id=route_id,
                component_id=component.component_id,
                component_root_cid=component.root_cid,
                route_profile_cid=profile.profile_cid,
                kind=spec.kind,
                transport=spec.transport,
                selector=spec.selector,
                source_path=spec.source_path,
                route_cid="",
            )
            normalized.append(
                NormalizedRuntimeRoute(
                    **{
                        **provisional.__dict__,
                        "route_cid": _cid(provisional.preimage()),
                    }
                )
            )

    provisional_catalog = RuntimeComponentCatalog(
        components=components,
        route_profiles=profiles,
        routes=tuple(normalized),
        catalog_cid="",
    )
    catalog_cid = _verified_cid(
        payload,
        "catalogCid",
        provisional_catalog.preimage(),
        require_stored_cids=require_stored_cids,
    )

    stored_routes = payload.get("routes")
    if require_stored_cids:
        supplied = _sequence(stored_routes, "routes")
        expected = [route.to_dict() for route in normalized]
        if list(supplied) != expected:
            raise RuntimeCIDError(
                "stored normalized routes do not match their derived roots",
                reason_code="normalized_routes_mismatch",
            )

    return RuntimeComponentCatalog(
        components=components,
        route_profiles=profiles,
        routes=tuple(normalized),
        catalog_cid=catalog_cid,
    )


def materialize_runtime_component_catalog(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a fully CID-bound serializable form of an unmaterialized catalog."""

    return build_runtime_component_catalog(payload).to_dict()


def load_runtime_component_catalog(path: str | Path) -> RuntimeComponentCatalog:
    """Load a fully materialized catalog, rejecting missing or stale CIDs."""

    catalog_path = Path(path)
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeComponentCatalogError(
            f"unable to load runtime component catalog: {catalog_path}",
            reason_code="catalog_load_failed",
            details={"path": str(catalog_path), "cause": repr(exc)},
        ) from exc
    return build_runtime_component_catalog(
        _mapping(payload, "catalog"),
        require_stored_cids=True,
    )


def validate_runtime_sources(
    catalog: RuntimeComponentCatalog,
    swissknife_root: str | Path,
) -> None:
    """Prove that every declared source and implementation symbol exists."""

    root = Path(swissknife_root)
    declarations: set[tuple[str, str | None]] = {
        (component.source_path, component.implementation_symbol)
        for component in catalog.components
    }
    declarations.update(
        (route.source_path, route.selector)
        for route in catalog.routes
    )
    for source_path, symbol in declarations:
        candidate = root / source_path
        if not candidate.is_file():
            raise RuntimeSourceError(
                f"runtime source does not exist: {source_path}",
                reason_code="runtime_source_missing",
                details={"sourcePath": source_path},
            )
        if symbol is not None and symbol not in candidate.read_text(encoding="utf-8"):
            raise RuntimeSourceError(
                f"runtime symbol does not exist: {symbol}",
                reason_code="runtime_symbol_missing",
                details={"sourcePath": source_path, "symbol": symbol},
            )


__all__ = [
    "RUNTIME_COMPONENT_CATALOG_INTERFACE",
    "CATALOG_VERSION",
    "AUTHORITY_SCHEMA",
    "COMPONENT_ROOT_SCHEMA",
    "ROUTE_PROFILE_SCHEMA",
    "NORMALIZED_ROUTE_SCHEMA",
    "RUNTIME_CATALOG_SCHEMA",
    "RuntimeComponentCatalogError",
    "MissingRuntimeComponentError",
    "DuplicateRuntimeComponentError",
    "MissingRuntimeRouteError",
    "DuplicateRuntimeRouteError",
    "RuntimeAuthorityError",
    "RuntimeCIDError",
    "RuntimeSourceError",
    "RuntimeComponentKind",
    "RuntimeRouteKind",
    "ImplementationAuthorityKind",
    "ImplementationAuthority",
    "RuntimeComponentRoot",
    "RuntimeRouteSpec",
    "RuntimeRouteProfile",
    "NormalizedRuntimeRoute",
    "RuntimeComponentCatalog",
    "build_runtime_component_catalog",
    "materialize_runtime_component_catalog",
    "load_runtime_component_catalog",
    "validate_runtime_sources",
]
