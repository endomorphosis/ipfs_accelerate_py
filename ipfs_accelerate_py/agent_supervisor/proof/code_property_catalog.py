"""Reviewed, content-addressed catalog of codebase proof properties (CBP-020).

Interface: ``CodePropertyCatalog@1``

The catalog is a closed registry: properties may only bind to reviewed
obligation template identifiers.  Natural-language theorem invent is not
supported.  ``semantic_authority`` defaults to ``False`` — proofs never
replace domain semantic metrics by themselves.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

from .formal_verification_contracts import (
    AssuranceLevel,
    content_identity,
)
from .proof_obligation_templates import (
    DEFAULT_TEMPLATE_REGISTRY,
    ProofObligationTemplateRegistry,
    ReviewedCodeShape,
)


CODE_PROPERTY_CATALOG_INTERFACE: Final = "CodePropertyCatalog@1"
CODE_PROPERTY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-property@1"
)
CODE_PROPERTY_CATALOG_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/code-property-catalog@1"
)
CATALOG_VERSION: Final = "1"

# Structural constraint tags shared with semantic-roundtrip StructuralAdmission.
SRT_STRUCTURAL_TAGS: Final[tuple[str, ...]] = (
    "non_vacuous_candidate",
    "rule_cardinality_preserved",
    "untriggered_projection_preserved",
)


class CodePropertyCatalogError(ValueError):
    """Catalog input is malformed or violates the closed-registration policy."""


class UnknownCodePropertyError(LookupError):
    """Raised when a property id is not present in the catalog."""


def _norm_id(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CodePropertyCatalogError(f"{field_name} must be a non-empty string")
    return value.strip()


def _sorted_unique_strings(
    values: Iterable[Any], *, field_name: str, required: bool = False
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        values = (values,)
    result = tuple(
        sorted({str(v).strip() for v in values if str(v).strip()})
    )
    if required and not result:
        raise CodePropertyCatalogError(f"{field_name} must not be empty")
    return result


@dataclass(frozen=True)
class CodeProperty:
    """One reviewed codebase property entry."""

    property_id: str
    template_id: str
    template_version: str
    template_semantic_hash: str
    code_shape: str
    sorts: tuple[str, ...]
    required_assurance: AssuranceLevel
    query_tags: tuple[str, ...]
    semantic_authority: bool = False
    invariant_class: str = ""
    title: str = ""
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "property_id", _norm_id(self.property_id, field_name="property_id")
        )
        object.__setattr__(
            self, "template_id", _norm_id(self.template_id, field_name="template_id")
        )
        object.__setattr__(
            self,
            "template_version",
            _norm_id(self.template_version, field_name="template_version"),
        )
        object.__setattr__(
            self,
            "template_semantic_hash",
            _norm_id(
                self.template_semantic_hash, field_name="template_semantic_hash"
            ),
        )
        object.__setattr__(
            self, "code_shape", _norm_id(self.code_shape, field_name="code_shape")
        )
        object.__setattr__(
            self,
            "sorts",
            _sorted_unique_strings(self.sorts, field_name="sorts", required=True),
        )
        assurance = self.required_assurance
        if not isinstance(assurance, AssuranceLevel):
            assurance = AssuranceLevel(str(assurance))
        object.__setattr__(self, "required_assurance", assurance)
        object.__setattr__(
            self,
            "query_tags",
            _sorted_unique_strings(self.query_tags, field_name="query_tags"),
        )
        if not isinstance(self.semantic_authority, bool):
            raise CodePropertyCatalogError("semantic_authority must be a boolean")
        object.__setattr__(
            self, "invariant_class", str(self.invariant_class or "").strip()
        )
        object.__setattr__(self, "title", str(self.title or "").strip())
        if not isinstance(self.metadata, Mapping):
            raise CodePropertyCatalogError("metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CODE_PROPERTY_SCHEMA,
            "property_id": self.property_id,
            "template_id": self.template_id,
            "template_version": self.template_version,
            "template_semantic_hash": self.template_semantic_hash,
            "code_shape": self.code_shape,
            "sorts": list(self.sorts),
            "required_assurance": self.required_assurance.value,
            "query_tags": list(self.query_tags),
            "semantic_authority": self.semantic_authority,
            "invariant_class": self.invariant_class,
            "title": self.title,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodeProperty":
        if not isinstance(payload, Mapping):
            raise CodePropertyCatalogError("code property must be an object")
        schema = payload.get("schema")
        if schema not in (None, CODE_PROPERTY_SCHEMA):
            raise CodePropertyCatalogError("unsupported code-property schema")
        return cls(
            property_id=str(payload.get("property_id") or ""),
            template_id=str(payload.get("template_id") or ""),
            template_version=str(payload.get("template_version") or ""),
            template_semantic_hash=str(payload.get("template_semantic_hash") or ""),
            code_shape=str(payload.get("code_shape") or ""),
            sorts=tuple(payload.get("sorts") or ()),
            required_assurance=AssuranceLevel(
                str(payload.get("required_assurance") or AssuranceLevel.KERNEL_VERIFIED.value)
            ),
            query_tags=tuple(payload.get("query_tags") or ()),
            semantic_authority=bool(payload.get("semantic_authority", False)),
            invariant_class=str(payload.get("invariant_class") or ""),
            title=str(payload.get("title") or ""),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class CodePropertyCatalog:
    """Immutable, content-addressed property catalog."""

    properties: tuple[CodeProperty, ...]
    catalog_version: str = CATALOG_VERSION
    declared_tags: tuple[str, ...] = SRT_STRUCTURAL_TAGS

    def __post_init__(self) -> None:
        if not isinstance(self.properties, tuple):
            object.__setattr__(self, "properties", tuple(self.properties))
        seen: set[str] = set()
        ordered = tuple(
            sorted(self.properties, key=lambda item: item.property_id)
        )
        for prop in ordered:
            if not isinstance(prop, CodeProperty):
                raise CodePropertyCatalogError(
                    "properties must be CodeProperty instances"
                )
            if prop.property_id in seen:
                raise CodePropertyCatalogError(
                    f"duplicate property_id: {prop.property_id}"
                )
            seen.add(prop.property_id)
        object.__setattr__(self, "properties", ordered)
        object.__setattr__(
            self,
            "catalog_version",
            _norm_id(self.catalog_version, field_name="catalog_version"),
        )
        object.__setattr__(
            self,
            "declared_tags",
            _sorted_unique_strings(
                self.declared_tags, field_name="declared_tags", required=True
            ),
        )
        index = {prop.property_id: prop for prop in ordered}
        object.__setattr__(self, "_index", MappingProxyType(index))

    @property
    def catalog_id(self) -> str:
        return content_identity(
            {
                "schema": CODE_PROPERTY_CATALOG_SCHEMA,
                "catalog_version": self.catalog_version,
                "declared_tags": list(self.declared_tags),
                "properties": [prop.to_dict() for prop in self.properties],
            }
        )

    def get(self, property_id: str) -> CodeProperty | None:
        return getattr(self, "_index").get(str(property_id).strip())

    def require(self, property_id: str) -> CodeProperty:
        prop = self.get(property_id)
        if prop is None:
            raise UnknownCodePropertyError(
                f"unknown code property id: {property_id!r}"
            )
        return prop

    def property_ids(self) -> tuple[str, ...]:
        return tuple(prop.property_id for prop in self.properties)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CODE_PROPERTY_CATALOG_SCHEMA,
            "interface": CODE_PROPERTY_CATALOG_INTERFACE,
            "catalog_version": self.catalog_version,
            "catalog_id": self.catalog_id,
            "declared_tags": list(self.declared_tags),
            "properties": [prop.to_dict() for prop in self.properties],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CodePropertyCatalog":
        if not isinstance(payload, Mapping):
            raise CodePropertyCatalogError("catalog must be an object")
        schema = payload.get("schema")
        if schema not in (None, CODE_PROPERTY_CATALOG_SCHEMA):
            raise CodePropertyCatalogError("unsupported catalog schema")
        raw_props = payload.get("properties") or ()
        properties = tuple(CodeProperty.from_dict(item) for item in raw_props)
        catalog = cls(
            properties=properties,
            catalog_version=str(payload.get("catalog_version") or CATALOG_VERSION),
            declared_tags=tuple(payload.get("declared_tags") or SRT_STRUCTURAL_TAGS),
        )
        claimed = payload.get("catalog_id")
        if claimed is not None and str(claimed) != catalog.catalog_id:
            raise CodePropertyCatalogError("catalog_id does not match content")
        return catalog


def _property_id_for_template(template_id: str) -> str:
    return f"property:{template_id}"


def build_seed_code_properties(
    registry: ProofObligationTemplateRegistry = DEFAULT_TEMPLATE_REGISTRY,
    *,
    required_assurance: AssuranceLevel = AssuranceLevel.KERNEL_VERIFIED,
    extra_tags: Sequence[str] = SRT_STRUCTURAL_TAGS,
) -> tuple[CodeProperty, ...]:
    """Build one property per reviewed template / code shape."""

    known_shapes = {shape.value for shape in ReviewedCodeShape}
    properties: list[CodeProperty] = []
    for template in registry._templates:
        shapes = tuple(template.supported_code_shapes or ())
        if not shapes:
            continue
        shape = shapes[0]
        if shape not in known_shapes:
            raise CodePropertyCatalogError(
                f"template {template.template_id!r} binds unknown shape {shape!r}"
            )
        tags = _sorted_unique_strings(
            (
                shape,
                template.template_id,
                template.invariant_class,
                *extra_tags,
            ),
            field_name="query_tags",
        )
        properties.append(
            CodeProperty(
                property_id=_property_id_for_template(template.template_id),
                template_id=template.template_id,
                template_version=str(template.version),
                template_semantic_hash=str(template.semantic_hash),
                code_shape=shape,
                sorts=("code", "protocol"),
                required_assurance=required_assurance,
                query_tags=tags,
                semantic_authority=False,
                invariant_class=str(template.invariant_class or ""),
                title=template.template_id.replace("-", " "),
                metadata={
                    "supported_code_shapes": list(shapes),
                    "registry_version": getattr(registry, "version", "1"),
                },
            )
        )
    if len(properties) != len(ReviewedCodeShape):
        # Allow fewer only if registry intentionally omits a shape; still seed
        # fail-closed unsupported if missing.
        have = {p.code_shape for p in properties}
        missing = known_shapes - have
        if missing and ReviewedCodeShape.UNSUPPORTED_PROOF_FAIL_CLOSED.value not in have:
            raise CodePropertyCatalogError(
                f"seed catalog missing shapes: {sorted(missing)}"
            )
    return tuple(sorted(properties, key=lambda item: item.property_id))


def build_default_code_property_catalog(
    registry: ProofObligationTemplateRegistry = DEFAULT_TEMPLATE_REGISTRY,
) -> CodePropertyCatalog:
    """Return the sealed default catalog used by CBP queries and packets."""

    return CodePropertyCatalog(
        properties=build_seed_code_properties(registry),
        catalog_version=CATALOG_VERSION,
        declared_tags=SRT_STRUCTURAL_TAGS,
    )


def register_code_property(
    catalog: CodePropertyCatalog,
    property_: CodeProperty,
    *,
    registry: ProofObligationTemplateRegistry = DEFAULT_TEMPLATE_REGISTRY,
) -> CodePropertyCatalog:
    """Return a new catalog with ``property_`` if the template is reviewed.

    Registration is closed: the template_id must exist in the reviewed
    registry and the code_shape must be among that template's supported shapes.
    """

    if not isinstance(catalog, CodePropertyCatalog):
        raise CodePropertyCatalogError("catalog must be a CodePropertyCatalog")
    if not isinstance(property_, CodeProperty):
        raise CodePropertyCatalogError("property must be a CodeProperty")
    if property_.semantic_authority:
        raise CodePropertyCatalogError(
            "semantic_authority=true is not allowed for registered properties"
        )
    known_templates = {
        template.template_id: template for template in registry._templates
    }
    template = known_templates.get(property_.template_id)
    if template is None:
        raise CodePropertyCatalogError(
            f"unknown reviewed template_id: {property_.template_id!r}"
        )
    shapes = set(template.supported_code_shapes or ())
    if property_.code_shape not in shapes:
        raise CodePropertyCatalogError(
            f"code_shape {property_.code_shape!r} is not supported by "
            f"template {property_.template_id!r}"
        )
    if catalog.get(property_.property_id) is not None:
        raise CodePropertyCatalogError(
            f"property_id already registered: {property_.property_id}"
        )
    return CodePropertyCatalog(
        properties=catalog.properties + (property_,),
        catalog_version=catalog.catalog_version,
        declared_tags=catalog.declared_tags,
    )


DEFAULT_CODE_PROPERTY_CATALOG = build_default_code_property_catalog()


__all__ = [
    "CODE_PROPERTY_CATALOG_INTERFACE",
    "CODE_PROPERTY_SCHEMA",
    "CODE_PROPERTY_CATALOG_SCHEMA",
    "CATALOG_VERSION",
    "SRT_STRUCTURAL_TAGS",
    "CodePropertyCatalogError",
    "UnknownCodePropertyError",
    "CodeProperty",
    "CodePropertyCatalog",
    "build_seed_code_properties",
    "build_default_code_property_catalog",
    "register_code_property",
    "DEFAULT_CODE_PROPERTY_CATALOG",
]
