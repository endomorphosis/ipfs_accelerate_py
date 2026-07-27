"""Side-effect-free projections of router-owned catalog discovery records.

Routers remain the authority for invocation.  This adapter only calls their
discovery snapshot (or, for older compatible routers, their list methods) and
never resolves a provider, constructs a client, probes a network, or invokes a
model.
"""

from __future__ import annotations

import re
from typing import Any, Optional, Tuple

from ..schema import (
    MAX_SNAPSHOT_RECORDS,
    CatalogSnapshot,
    ModelDescriptor,
    ProviderDescriptor,
    Provenance,
    RouterBinding,
)
from ..snapshot import snapshot_records
from .static import CatalogSourceResult, SourceMetadata

DEFAULT_ROUTER_PRECEDENCE = 30
_SOURCE_BAD = re.compile(r"[^a-z0-9._/-]+")


def _canonical_name(value: Any, field_name: str, maximum: int = 128) -> str:
    if not isinstance(value, str):
        raise ValueError("%s must be a string" % field_name)
    normalized = _SOURCE_BAD.sub("-", value.strip().casefold()).strip("-._/")
    normalized = re.sub(r"/+", "/", normalized)
    normalized = re.sub(r"\.{2,}", ".", normalized)
    if (
        not normalized
        or len(normalized.encode("utf-8")) > maximum
        or "//" in normalized
        or ".." in normalized
    ):
        raise ValueError("%s must be a bounded canonical name" % field_name)
    return normalized


def _router_name(router: Any) -> str:
    name = getattr(router, "__name__", None)
    if isinstance(name, str) and name:
        return _canonical_name(name.rsplit(".", 1)[-1], "router", 64)
    return _canonical_name(type(router).__name__, "router", 64)


def _router_source(router: Any) -> str:
    return _canonical_name("routers/%s" % _router_name(router), "source")


def _coerce_snapshot(value: Any) -> CatalogSnapshot:
    if isinstance(value, CatalogSnapshot):
        return value
    if isinstance(value, dict):
        return CatalogSnapshot.from_dict(value)
    raise TypeError("router discovery must return a CatalogSnapshot")


def _compatible_snapshot(router: Any, source: str) -> CatalogSnapshot:
    """Project the four shared list methods for a compatible older router."""

    list_providers = getattr(router, "list_providers", None)
    list_models = getattr(router, "list_models", None)
    if not callable(list_providers) or not callable(list_models):
        raise TypeError(
            "router must expose get_catalog_snapshot/catalog_snapshot or "
            "list_providers and list_models"
        )
    providers = tuple(list_providers())
    models = tuple(list_models())
    if any(not isinstance(item, ProviderDescriptor) for item in providers):
        raise TypeError("router list_providers returned a non-provider record")
    if any(not isinstance(item, ModelDescriptor) for item in models):
        raise TypeError("router list_models returned a non-model record")

    router_name = _router_name(router)
    bindings = []
    for index, model in enumerate(models):
        operations = tuple(
            sorted(
                {
                    operation
                    for capability in model.capabilities
                    for operation in capability.operations
                },
                key=lambda item: item.value,
            )
        )
        invokable = tuple(
            item for item in operations if item.value not in {"batch", "stream"}
        )
        if not invokable:
            continue
        bindings.append(
            RouterBinding(
                router=router_name,
                provider_id=model.provider_id,
                model_id=model.model_id,
                operations=operations,
                priority=index,
                state=model.state,
                provenance=(Provenance(source=source),),
            )
        )
    return CatalogSnapshot(
        providers=providers,
        models=models,
        bindings=tuple(bindings),
    )


class RouterCatalogSource:
    """Adapt one injected router's discovery surface into a catalog source."""

    side_effecting = False

    def __init__(
        self,
        router: Any,
        *,
        source: Optional[str] = None,
        precedence: int = DEFAULT_ROUTER_PRECEDENCE,
        max_records: int = MAX_SNAPSHOT_RECORDS,
    ) -> None:
        if router is None:
            raise ValueError("router is required")
        if (
            isinstance(precedence, bool)
            or not isinstance(precedence, int)
            or not -1_000_000 <= precedence <= 1_000_000
        ):
            raise ValueError("precedence must be between -1000000 and 1000000")
        if (
            isinstance(max_records, bool)
            or not isinstance(max_records, int)
            or not 0 <= max_records <= MAX_SNAPSHOT_RECORDS
        ):
            raise ValueError(
                "max_records must be between 0 and %d" % MAX_SNAPSHOT_RECORDS
            )
        self.router = router
        self.source = _canonical_name(
            source if source is not None else _router_source(router), "source"
        )
        self.precedence = precedence
        self.max_records = max_records

    def load(self) -> CatalogSourceResult:
        """Read immutable router metadata without invoking normal resolution."""

        loader = getattr(self.router, "get_catalog_snapshot", None)
        if not callable(loader):
            loader = getattr(self.router, "catalog_snapshot", None)
        snapshot = (
            _coerce_snapshot(loader())
            if callable(loader)
            else _compatible_snapshot(self.router, self.source)
        )
        count = len(snapshot_records(snapshot))
        if count > self.max_records:
            raise ValueError(
                "router source exceeds maximum record count (%d > %d)"
                % (count, self.max_records)
            )
        return CatalogSourceResult(
            snapshot=snapshot,
            metadata=SourceMetadata(
                source=self.source,
                precedence=self.precedence,
                revision=snapshot.revision,
                created_at=snapshot.created_at,
                updated_at=None,
            ),
        )

    snapshot = load
    read = load


RouterSourceAdapter = RouterCatalogSource
RouterSourceResult = CatalogSourceResult


def adapt_router_source(router: Any, **kwargs: Any) -> CatalogSourceResult:
    """Adapt one injected router without importing or initializing routers."""

    return RouterCatalogSource(router, **kwargs).load()


def build_router_sources(
    *routers: Any,
    precedence: int = DEFAULT_ROUTER_PRECEDENCE,
    max_records: int = MAX_SNAPSHOT_RECORDS,
) -> Tuple[RouterCatalogSource, ...]:
    """Return deterministic adapters for an explicitly supplied router set."""

    result = tuple(
        RouterCatalogSource(
            router,
            precedence=precedence,
            max_records=max_records,
        )
        for router in routers
    )
    names = [item.source for item in result]
    if len(names) != len(set(names)):
        raise ValueError("router sources must have unique canonical names")
    return tuple(sorted(result, key=lambda item: item.source))


load_router_catalog = adapt_router_source


__all__ = [
    "DEFAULT_ROUTER_PRECEDENCE",
    "RouterCatalogSource",
    "RouterSourceAdapter",
    "RouterSourceResult",
    "adapt_router_source",
    "build_router_sources",
    "load_router_catalog",
]
