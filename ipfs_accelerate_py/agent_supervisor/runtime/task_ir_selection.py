"""Task context nominations of exact persisted IR catalog declarations.

This reads the same explicit metadata stores as the ModelManager catalog. It
does not instantiate ModelManager, authenticate checkpoint or Hugging Face
payloads, load an encoder/decoder, or qualify a runtime. Native nullable fields
and declaration flags remain unchanged; a declaration is not an observation.
Repeated reads are cooperative catalog generation checks, not an atomic store
snapshot. Context consumers must separately validate their actual artifacts.
"""
from __future__ import annotations

from pathlib import Path

from ...model_catalog.sources.ir_persistent import IRPersistentCatalogSource, _selectors

MAX_SELECTIONS = 16
_NAMESPACE_FIELDS = (
    "ir_family_id", "dimension", "dimension_role", "schema_version", "task_id",
    "profile_id", "format_id", "role",
)


def resolve_task_ir_selections(*, catalog_path: Path, selections: list[dict]) -> list[dict]:
    """Resolve 1–16 exact metadata nominations in one unchanged catalog generation.

    Returns the native ``ir-persisted-binding-resolution/v1`` records, including
    their false authority observations, unchanged. Requests use all ten native
    selector fields. Distinct heads may share a family/dimension, but two assets
    cannot silently compete for one complete namespace. No unknown schema,
    task, profile, format, token/span limit or inventory identity is inferred.
    """
    if (not isinstance(catalog_path, Path) or not catalog_path.is_absolute()
            or ".." in catalog_path.parts or catalog_path.resolve() != catalog_path):
        raise ValueError("task IR selection requires an exact absolute catalog path")
    if type(selections) is not list or not 1 <= len(selections) <= MAX_SELECTIONS:
        raise ValueError("task IR selection requires 1 to 16 exact catalog selectors")
    # Reuse the native closed parser before touching a store; keep the public
    # resolver as the owner of actual persisted selection and readiness scope.
    requests = [_selectors(item) for item in selections]
    namespaces = [tuple(item[name] for name in _NAMESPACE_FIELDS) for item in requests]
    if len(set(namespaces)) != len(namespaces):
        raise ValueError("task IR selections cannot compete for one complete namespace")
    source = IRPersistentCatalogSource(path=catalog_path)
    results = [source.resolve_ir_binding(item) for item in requests]
    closing = source.load()
    if any(result["binding_snapshot_revision"] != closing.binding_snapshot_revision
           or result["catalog_revision"] != closing.revision for result in results):
        raise ValueError("task IR catalog generation changed during selection")
    return results
