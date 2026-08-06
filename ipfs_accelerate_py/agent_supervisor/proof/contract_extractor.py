"""Deterministic multi-source program-contract extractor (VFS-015 / VFS-G050).

Consumes compact, artifact-referenced source units and emits versioned
:class:`~ipfs_accelerate_py.agent_supervisor.program_contracts.ExpectedProgramContract`
and
:class:`~ipfs_accelerate_py.agent_supervisor.program_contracts.ObservedProgramContract`
records under the closed expectation precedence:

1. reviewed MCP++/MCP IDL, JSON Schema, typed interfaces, protocol specs
2. public signatures, type annotations, stable exports
3. executable contract / conformance tests
4. normative documentation
5. compatibility manifests and generated SDKs

Implementation observations may only populate observed contracts.  Examples,
mocks, fixtures, deprecated variants, and generated copies are classified and
never silently promoted to authoritative expectations.  Conflicting
equal-precedence clauses are reported rather than resolved.  Large source
bodies remain outside this module; units carry compact facts, spans, CIDs, and
extraction-rule identifiers only.

Extraction is deliberately independent from satisfaction checking
(:mod:`contract_checker`).  This module never imports the checker; observed
implementation behavior cannot define its own expectation (no circular
oracles).  Objective validation repair for VFS-G050 anchors the synthetic
discovery term ``objective validation repair`` so scans re-find the
validation gate after domain evidence (``vfs/contract-ir@1``,
``vfs/contract-source-precedence@1``) is present — without granting that term
contract identity or completion authority.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Final, Iterator

from .program_contracts import (
    CONTRACT_IR_EVIDENCE,
    CONTRACT_SOURCE_PRECEDENCE_EVIDENCE,
    CONTRACT_VERSION,
    MAX_CLAUSE_BYTES,
    MAX_COLLECTION_ITEMS,
    MAX_CONFLICTS,
    MAX_UNSUPPORTED,
    OBJECTIVE_GOAL_ID,
    OBJECTIVE_VALIDATION_REPAIR_EVIDENCE,
    SOURCE_PRECEDENCE,
    Applicability,
    Assumption,
    AtomicityMode,
    AtomicitySpec,
    AuthorizationMode,
    AuthorizationSpec,
    CapabilityMode,
    CapabilitySpec,
    CircularExpectationError,
    ConfidenceClass,
    ConflictKind,
    ConsistencyMode,
    ConsistencySpec,
    ContractBoundsError,
    ContractConflict,
    ContractSourceKind,
    DegradationMode,
    EffectKind,
    EffectPolarity,
    ErrorSpec,
    ExpectedProgramContract,
    FallbackSpec,
    ForgedSourceError,
    IdempotenceMode,
    IdempotenceSpec,
    InterfaceIdentity,
    ObservedProgramContract,
    Optionality,
    OrderingMode,
    OrderingSpec,
    ParameterKind,
    ParameterSpec,
    ProgramContractBundle,
    ProgramContractError,
    ProgramContractRole,
    ResourceBounds,
    ReturnSpec,
    SemanticAspect,
    SideEffectSpec,
    SourceReference,
    SupportStatus,
    SymbolIdentity,
    SyncAsyncSpec,
    SyncMode,
    TypeConstructor,
    TypeShape,
    UnsupportedSemantics,
    all_program_contract_evidence_terms,
    canonical_program_contract_json_bytes,
    may_define_expectation,
    objective_validation_repair_evidence_terms,
    program_contract_content_identity,
    program_contract_evidence_terms,
    source_precedence_rank,
)
from .formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    content_identity,
)


# ---------------------------------------------------------------------------
# Versioning and bounds
# ---------------------------------------------------------------------------

CONTRACT_EXTRACTOR_VERSION: Final[int] = 1
SCHEMA_VERSION: Final[int] = CONTRACT_EXTRACTOR_VERSION

# Re-export VFS-G050 evidence terms so scanners and callers discover them on
# both the IR module and this independent extractor surface.  Extraction
# never imports contract_checker; satisfaction checking never feeds back
# into expected-contract compilation (no circular oracles).
# Domain envelope evidence stays IR/precedence-only; the synthetic objective
# validation repair term is discoverable via objective_validation_repair_evidence_terms
# / all_covered_evidence_terms and never enters content-addressed identity.
CONTRACT_EXTRACTOR_EVIDENCE: Final[tuple[str, ...]] = (
    CONTRACT_IR_EVIDENCE,
    CONTRACT_SOURCE_PRECEDENCE_EVIDENCE,
)
assert OBJECTIVE_VALIDATION_REPAIR_EVIDENCE == "objective validation repair"
assert OBJECTIVE_GOAL_ID == "VFS-G050"

CONTRACT_SOURCE_UNIT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-source-unit@1"
)
PARTIAL_CONTRACT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-partial@1"
)
EXTRACTION_RESULT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-extraction-result@1"
)
SKIPPED_SOURCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/contract-skipped-source@1"
)

MAX_SOURCE_UNITS: Final[int] = 512
MAX_SCHEMA_DEPTH: Final[int] = 16
MAX_OVERLOADS: Final[int] = 32
MAX_SKIPPED: Final[int] = 256
MAX_TEXT: Final[int] = 4_096
MAX_RESULT_BYTES: Final[int] = 1_048_576

# Default policy revision when callers omit one.
DEFAULT_POLICY_REVISION: Final[str] = "policy:vfs-assurance/contract-extractor@1"

_TYPE_ALIASES: Final[dict[str, TypeConstructor]] = {
    "any": TypeConstructor.ANY,
    "unknown": TypeConstructor.UNKNOWN,
    "never": TypeConstructor.NEVER,
    "null": TypeConstructor.NULL,
    "none": TypeConstructor.NULL,
    "nil": TypeConstructor.NULL,
    "bool": TypeConstructor.BOOL,
    "boolean": TypeConstructor.BOOL,
    "int": TypeConstructor.INT,
    "integer": TypeConstructor.INT,
    "number": TypeConstructor.INT,
    "float": TypeConstructor.INT,
    "bytes": TypeConstructor.BYTES,
    "bytearray": TypeConstructor.BYTES,
    "binary": TypeConstructor.BYTES,
    "str": TypeConstructor.STRING,
    "string": TypeConstructor.STRING,
    "text": TypeConstructor.STRING,
    "list": TypeConstructor.ARRAY,
    "array": TypeConstructor.ARRAY,
    "sequence": TypeConstructor.ARRAY,
    "dict": TypeConstructor.OBJECT,
    "object": TypeConstructor.OBJECT,
    "map": TypeConstructor.OBJECT,
    "mapping": TypeConstructor.OBJECT,
    "enum": TypeConstructor.ENUM,
    "union": TypeConstructor.UNION,
    "optional": TypeConstructor.UNION,
}

_JSON_TYPE_MAP: Final[dict[str, TypeConstructor]] = {
    "string": TypeConstructor.STRING,
    "integer": TypeConstructor.INT,
    "number": TypeConstructor.INT,
    "boolean": TypeConstructor.BOOL,
    "null": TypeConstructor.NULL,
    "array": TypeConstructor.ARRAY,
    "object": TypeConstructor.OBJECT,
}


# ---------------------------------------------------------------------------
# Errors and closed vocabularies
# ---------------------------------------------------------------------------


class ContractExtractorError(ProgramContractError):
    """Base error for contract extraction failures."""


class ContractExtractorBoundsError(ContractExtractorError, ContractBoundsError):
    """Extraction exceeded a declared bound."""


class MissingReferenceError(ContractExtractorError):
    """A schema or type reference could not be resolved."""


class UnsupportedExtractionError(ContractExtractorError):
    """An extraction rule cannot represent a clause."""


class SourceArtifactClass(str, Enum):
    """How a source unit relates to normative contract authority.

    Only :attr:`NORMATIVE` units (and carefully demoted :attr:`DEPRECATED` /
    :attr:`GENERATED` units) may contribute expectation clauses.  Examples,
    mocks, fixtures, and pure observations never define expectations.
    """

    NORMATIVE = "normative"
    EXAMPLE = "example"
    MOCK = "mock"
    FIXTURE = "fixture"
    DEPRECATED = "deprecated"
    GENERATED = "generated"
    OBSERVATION = "observation"

    @property
    def may_define_expectation(self) -> bool:
        return self in {
            SourceArtifactClass.NORMATIVE,
            SourceArtifactClass.DEPRECATED,
            SourceArtifactClass.GENERATED,
        }

    @property
    def is_non_authoritative(self) -> bool:
        return self in {
            SourceArtifactClass.EXAMPLE,
            SourceArtifactClass.MOCK,
            SourceArtifactClass.FIXTURE,
        }


class ExtractionRule(str, Enum):
    """Closed vocabulary of extractor rules (stable for CIDs and spans)."""

    MCP_IDL_V1 = "mcp_idl_v1"
    MCP_PLUS_PLUS_IDL_V1 = "mcp_plusplus_idl_v1"
    JSON_SCHEMA_V1 = "json_schema_v1"
    TYPED_INTERFACE_V1 = "typed_interface_v1"
    PUBLIC_SIGNATURE_V1 = "public_signature_v1"
    CONTRACT_TEST_V1 = "contract_test_v1"
    NORMATIVE_DOC_V1 = "normative_doc_v1"
    COMPAT_MANIFEST_V1 = "compat_manifest_v1"
    GENERATED_SDK_V1 = "generated_sdk_v1"
    IMPLEMENTATION_OBS_V1 = "implementation_obs_v1"
    OVERLOAD_MERGE_V1 = "overload_merge_v1"
    PRECEDENCE_MERGE_V1 = "precedence_merge_v1"
    CONFLICT_DETECT_V1 = "conflict_detect_v1"
    SELF_EXPECTATION_GUARD_V1 = "self_expectation_guard_v1"
    ARTIFACT_CLASSIFY_V1 = "artifact_classify_v1"


class SkipReason(str, Enum):
    """Why a source unit did not contribute an expectation."""

    EXAMPLE = "example"
    MOCK = "mock"
    FIXTURE = "fixture"
    OBSERVATION_ONLY = "observation_only"
    NON_AUTHORITATIVE = "non_authoritative"
    EMPTY_PAYLOAD = "empty_payload"
    DUPLICATE = "duplicate"
    OUT_OF_SCOPE = "out_of_scope"
    DEPRECATED_SHADOWED = "deprecated_shadowed"
    GENERATED_SHADOWED = "generated_shadowed"


class ContentKind(str, Enum):
    """Structured payload kind inside a source unit."""

    MCP_TOOL = "mcp_tool"
    MCP_IDL = "mcp_idl"
    MCP_PLUS_PLUS = "mcp_plusplus"
    JSON_SCHEMA = "json_schema"
    TYPED_INTERFACE = "typed_interface"
    PUBLIC_SIGNATURE = "public_signature"
    CONTRACT_TEST = "contract_test"
    NORMATIVE_DOC = "normative_doc"
    COMPAT_MANIFEST = "compat_manifest"
    GENERATED_SDK = "generated_sdk"
    OBSERVATION = "observation"
    OVERLOAD_SET = "overload_set"
    UNKNOWN = "unknown"


# Default extraction rule for each expectation source kind.
_DEFAULT_RULE_FOR_KIND: Final[dict[ContractSourceKind, ExtractionRule]] = {
    ContractSourceKind.REVIEWED_INTERFACE: ExtractionRule.MCP_IDL_V1,
    ContractSourceKind.PUBLIC_SIGNATURE: ExtractionRule.PUBLIC_SIGNATURE_V1,
    ContractSourceKind.CONTRACT_TEST: ExtractionRule.CONTRACT_TEST_V1,
    ContractSourceKind.NORMATIVE_DOCUMENTATION: ExtractionRule.NORMATIVE_DOC_V1,
    ContractSourceKind.COMPATIBILITY_MANIFEST: ExtractionRule.COMPAT_MANIFEST_V1,
    ContractSourceKind.IMPLEMENTATION_OBSERVATION: ExtractionRule.IMPLEMENTATION_OBS_V1,
}

_DEFAULT_CONFIDENCE: Final[dict[ContractSourceKind, ConfidenceClass]] = {
    ContractSourceKind.REVIEWED_INTERFACE: ConfidenceClass.HIGH,
    ContractSourceKind.PUBLIC_SIGNATURE: ConfidenceClass.HIGH,
    ContractSourceKind.CONTRACT_TEST: ConfidenceClass.HIGH,
    ContractSourceKind.NORMATIVE_DOCUMENTATION: ConfidenceClass.MEDIUM,
    ContractSourceKind.COMPATIBILITY_MANIFEST: ConfidenceClass.MEDIUM,
    ContractSourceKind.IMPLEMENTATION_OBSERVATION: ConfidenceClass.MEDIUM,
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _text(value: Any, *, field_name: str, required: bool = True) -> str:
    if value is None:
        if required:
            raise ContractExtractorError(f"{field_name} is required")
        return ""
    if not isinstance(value, str):
        raise ContractExtractorError(f"{field_name} must be a string")
    text = value.strip()
    if required and not text:
        raise ContractExtractorError(f"{field_name} must be non-empty")
    if len(text.encode("utf-8")) > MAX_TEXT:
        raise ContractExtractorBoundsError(
            f"{field_name} exceeds {MAX_TEXT} bytes"
        )
    return text


def _optional_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractExtractorError(f"{field_name} must be an integer")
    if value < 0:
        raise ContractExtractorError(f"{field_name} must be non-negative")
    return value


def _boolean(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ContractExtractorError(f"{field_name} must be a boolean")
    return value


def _enum(value: Any, enum_type: type[Enum], *, field_name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except ValueError as exc:
            raise ContractExtractorError(
                f"{field_name} has unknown value {value!r}"
            ) from exc
    raise ContractExtractorError(f"{field_name} must be a {enum_type.__name__}")


def _mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractExtractorError(f"{field_name} must be a mapping")
    return value


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _content_sha256(payload: Any) -> str:
    digest = _sha256_hex(canonical_program_contract_json_bytes(payload))
    return f"sha256:{digest}"


def _infer_content_kind(payload: Mapping[str, Any]) -> ContentKind:
    raw = payload.get("kind") or payload.get("content_kind") or ""
    if isinstance(raw, ContentKind):
        return raw
    if isinstance(raw, str) and raw:
        try:
            return ContentKind(raw)
        except ValueError:
            pass
    if "inputSchema" in payload or "input_schema" in payload:
        return ContentKind.MCP_TOOL
    if "tools" in payload and isinstance(payload.get("tools"), Sequence):
        return ContentKind.MCP_IDL
    if payload.get("$schema") or payload.get("type") or payload.get("properties"):
        if "parameters" not in payload and "returns" not in payload:
            return ContentKind.JSON_SCHEMA
    if "overloads" in payload:
        return ContentKind.OVERLOAD_SET
    if "asserts" in payload or payload.get("kind") == "contract_test":
        return ContentKind.CONTRACT_TEST
    if "observed" in payload or "repository_observation_id" in payload:
        return ContentKind.OBSERVATION
    if "parameters" in payload or "returns" in payload or "signature" in payload:
        return ContentKind.PUBLIC_SIGNATURE
    if "clauses" in payload:
        return ContentKind.NORMATIVE_DOC
    if "manifest" in payload or "generated" in payload:
        return ContentKind.COMPAT_MANIFEST
    return ContentKind.UNKNOWN


def classify_artifact_path(
    path: str,
    *,
    explicit: SourceArtifactClass | str | None = None,
) -> SourceArtifactClass:
    """Classify a path/locator as example, mock, fixture, generated, etc."""

    if explicit is not None:
        return _enum(explicit, SourceArtifactClass, field_name="artifact_class")
    lowered = (path or "").replace("\\", "/").lower()
    base = lowered.rsplit("/", 1)[-1]
    markers = (
        (SourceArtifactClass.MOCK, ("mock", "fake", "stub", "double")),
        (SourceArtifactClass.FIXTURE, ("fixture", "fixtures", "testdata", "goldens")),
        (SourceArtifactClass.EXAMPLE, ("example", "examples", "sample", "demo")),
        (SourceArtifactClass.GENERATED, ("generated", ".gen.", "_pb2", ".sdk.")),
        (
            SourceArtifactClass.DEPRECATED,
            (".deprecated", ".old", ".legacy", ".broken", ".fixed"),
        ),
    )
    for klass, tokens in markers:
        for token in tokens:
            if token in lowered or token in base:
                return klass
    if "/test/" in f"/{lowered}" or base.startswith("test_"):
        # Test helpers are fixtures unless explicitly normative contract tests.
        if "contract" in lowered or "conformance" in lowered:
            return SourceArtifactClass.NORMATIVE
        return SourceArtifactClass.FIXTURE
    return SourceArtifactClass.NORMATIVE


def confidence_for(
    source_kind: ContractSourceKind,
    artifact_class: SourceArtifactClass,
) -> ConfidenceClass:
    """Bounded confidence for a source kind / artifact class pair."""

    if artifact_class is SourceArtifactClass.EXAMPLE:
        return ConfidenceClass.SPECULATIVE
    if artifact_class is SourceArtifactClass.MOCK:
        return ConfidenceClass.SPECULATIVE
    if artifact_class is SourceArtifactClass.FIXTURE:
        return ConfidenceClass.LOW
    if artifact_class is SourceArtifactClass.DEPRECATED:
        return ConfidenceClass.LOW
    if artifact_class is SourceArtifactClass.GENERATED:
        if source_kind is ContractSourceKind.COMPATIBILITY_MANIFEST:
            return ConfidenceClass.MEDIUM
        return ConfidenceClass.LOW
    return _DEFAULT_CONFIDENCE.get(source_kind, ConfidenceClass.MEDIUM)


def extraction_rule_for(
    source_kind: ContractSourceKind,
    content_kind: ContentKind,
    *,
    artifact_class: SourceArtifactClass = SourceArtifactClass.NORMATIVE,
) -> ExtractionRule:
    """Select the stable extraction rule identifier for a unit."""

    if content_kind is ContentKind.MCP_PLUS_PLUS:
        return ExtractionRule.MCP_PLUS_PLUS_IDL_V1
    if content_kind in {ContentKind.MCP_TOOL, ContentKind.MCP_IDL}:
        return ExtractionRule.MCP_IDL_V1
    if content_kind is ContentKind.JSON_SCHEMA:
        return ExtractionRule.JSON_SCHEMA_V1
    if content_kind is ContentKind.TYPED_INTERFACE:
        return ExtractionRule.TYPED_INTERFACE_V1
    if content_kind is ContentKind.PUBLIC_SIGNATURE:
        return ExtractionRule.PUBLIC_SIGNATURE_V1
    if content_kind is ContentKind.CONTRACT_TEST:
        return ExtractionRule.CONTRACT_TEST_V1
    if content_kind is ContentKind.NORMATIVE_DOC:
        return ExtractionRule.NORMATIVE_DOC_V1
    if content_kind is ContentKind.GENERATED_SDK:
        return ExtractionRule.GENERATED_SDK_V1
    if content_kind is ContentKind.COMPAT_MANIFEST:
        return ExtractionRule.COMPAT_MANIFEST_V1
    if content_kind is ContentKind.OBSERVATION:
        return ExtractionRule.IMPLEMENTATION_OBS_V1
    if content_kind is ContentKind.OVERLOAD_SET:
        return ExtractionRule.OVERLOAD_MERGE_V1
    if artifact_class is SourceArtifactClass.GENERATED:
        return ExtractionRule.GENERATED_SDK_V1
    return _DEFAULT_RULE_FOR_KIND.get(
        source_kind, ExtractionRule.NORMATIVE_DOC_V1
    )


# ---------------------------------------------------------------------------
# TypeShape conversion (JSON Schema + compact type DSL)
# ---------------------------------------------------------------------------


def type_shape_from_name(name: Any, *, nullable: bool = False) -> TypeShape:
    """Map a compact type name (``str``, ``list[str]``, ``A|B``) to TypeShape."""

    if name is None:
        return TypeShape(
            constructor=TypeConstructor.UNKNOWN,
            support=SupportStatus.UNKNOWN,
        )
    if isinstance(name, TypeShape):
        return name
    if isinstance(name, Mapping):
        return type_shape_from_json_schema(name)
    if not isinstance(name, str):
        raise ContractExtractorError("type name must be a string or mapping")
    text = name.strip()
    if not text:
        return TypeShape(
            constructor=TypeConstructor.UNKNOWN,
            support=SupportStatus.UNKNOWN,
        )
    if text.endswith(" | None") or text.endswith("|None"):
        base = text.rsplit("|", 1)[0].strip()
        return type_shape_from_name(base, nullable=True)
    if text.startswith("Optional[") and text.endswith("]"):
        return type_shape_from_name(text[len("Optional[") : -1], nullable=True)
    if "|" in text and not text.startswith("list[") and "->" not in text:
        parts = [part.strip() for part in text.split("|") if part.strip()]
        if len(parts) >= 2:
            return TypeShape(
                constructor=TypeConstructor.UNION,
                alternatives=tuple(type_shape_from_name(part) for part in parts),
                nullable=nullable,
            )
    list_match = re.fullmatch(r"(?:list|List|Sequence|Array)\[(.+)\]", text)
    if list_match:
        return TypeShape(
            constructor=TypeConstructor.ARRAY,
            item=type_shape_from_name(list_match.group(1)),
            nullable=nullable,
        )
    dict_match = re.fullmatch(
        r"(?:dict|Dict|Mapping|Object)\[(.+),\s*(.+)\]", text
    )
    if dict_match:
        return TypeShape(
            constructor=TypeConstructor.OBJECT,
            name=text,
            nullable=nullable,
            constraints=(f"value:{dict_match.group(2).strip()}",),
        )
    key = text.lower().replace(" ", "")
    constructor = _TYPE_ALIASES.get(key)
    if constructor is not None:
        return TypeShape(
            constructor=constructor,
            name=text,
            nullable=nullable,
        )
    # Treat unknown identifiers as named references.
    return TypeShape(
        constructor=TypeConstructor.REFERENCE,
        name=text,
        reference=text,
        nullable=nullable,
        support=SupportStatus.ASSUMED,
    )


def type_shape_from_json_schema(
    schema: Any,
    *,
    definitions: Mapping[str, Any] | None = None,
    depth: int = 0,
    ref_stack: tuple[str, ...] = (),
    missing_refs: list[str] | None = None,
) -> TypeShape:
    """Convert a JSON Schema fragment into a bounded TypeShape.

    Missing ``$ref`` targets are recorded (when ``missing_refs`` is provided)
    and emitted as unsupported reference shapes rather than invented types.
    Circular ``$ref`` chains yield an unsupported residual.
    """

    if missing_refs is None:
        missing_refs = []
    if depth > MAX_SCHEMA_DEPTH:
        return TypeShape(
            constructor=TypeConstructor.UNSUPPORTED,
            support=SupportStatus.UNSUPPORTED,
            constraints=("max_schema_depth",),
        )
    if schema is None:
        return TypeShape(
            constructor=TypeConstructor.UNKNOWN,
            support=SupportStatus.UNKNOWN,
        )
    if not isinstance(schema, Mapping):
        if isinstance(schema, str):
            return type_shape_from_name(schema)
        return TypeShape(
            constructor=TypeConstructor.UNSUPPORTED,
            support=SupportStatus.UNSUPPORTED,
            constraints=("non_object_schema",),
        )

    defs: dict[str, Any] = {}
    if definitions:
        defs.update(dict(definitions))
    for key in ("$defs", "definitions"):
        block = schema.get(key)
        if isinstance(block, Mapping):
            defs.update(dict(block))

    ref = schema.get("$ref")
    if isinstance(ref, str) and ref:
        if ref in ref_stack:
            return TypeShape(
                constructor=TypeConstructor.UNSUPPORTED,
                reference=ref,
                support=SupportStatus.UNSUPPORTED,
                constraints=("circular_ref",),
            )
        target = _resolve_json_ref(ref, defs, schema)
        if target is None:
            missing_refs.append(ref)
            return TypeShape(
                constructor=TypeConstructor.REFERENCE,
                reference=ref,
                support=SupportStatus.UNSUPPORTED,
                constraints=("missing_ref",),
            )
        return type_shape_from_json_schema(
            target,
            definitions=defs,
            depth=depth + 1,
            ref_stack=ref_stack + (ref,),
            missing_refs=missing_refs,
        )

    if "oneOf" in schema or "anyOf" in schema:
        key = "oneOf" if "oneOf" in schema else "anyOf"
        alts_raw = schema.get(key) or ()
        alternatives: list[TypeShape] = []
        if isinstance(alts_raw, Sequence) and not isinstance(
            alts_raw, (str, bytes, bytearray)
        ):
            for item in alts_raw[:32]:
                alternatives.append(
                    type_shape_from_json_schema(
                        item,
                        definitions=defs,
                        depth=depth + 1,
                        ref_stack=ref_stack,
                        missing_refs=missing_refs,
                    )
                )
        return TypeShape(
            constructor=TypeConstructor.UNION,
            alternatives=tuple(alternatives),
            name=_text(schema.get("title") or "", field_name="title", required=False),
            nullable=bool(schema.get("nullable", False)),
        )

    if "allOf" in schema:
        parts = schema.get("allOf") or ()
        alternatives = []
        if isinstance(parts, Sequence) and not isinstance(
            parts, (str, bytes, bytearray)
        ):
            for item in parts[:32]:
                alternatives.append(
                    type_shape_from_json_schema(
                        item,
                        definitions=defs,
                        depth=depth + 1,
                        ref_stack=ref_stack,
                        missing_refs=missing_refs,
                    )
                )
        return TypeShape(
            constructor=TypeConstructor.INTERSECTION,
            alternatives=tuple(alternatives),
            name=_text(schema.get("title") or "", field_name="title", required=False),
        )

    if "enum" in schema and isinstance(schema.get("enum"), Sequence):
        values = tuple(
            str(item)
            for item in schema.get("enum") or ()
            if item is not None
        )[:256]
        return TypeShape(
            constructor=TypeConstructor.ENUM,
            enum_values=values,
            name=_text(schema.get("title") or "", field_name="title", required=False),
        )

    type_field = schema.get("type")
    nullable = bool(schema.get("nullable", False))
    if isinstance(type_field, list):
        # JSON Schema type unions, often including "null".
        non_null = [item for item in type_field if item != "null"]
        if "null" in type_field:
            nullable = True
        if len(non_null) == 1:
            type_field = non_null[0]
        elif non_null:
            return TypeShape(
                constructor=TypeConstructor.UNION,
                alternatives=tuple(
                    type_shape_from_json_schema(
                        {**schema, "type": item},
                        definitions=defs,
                        depth=depth + 1,
                        ref_stack=ref_stack,
                        missing_refs=missing_refs,
                    )
                    for item in non_null[:32]
                ),
                nullable=nullable,
            )

    if type_field == "array" or (
        isinstance(type_field, str) and type_field.lower() == "array"
    ):
        items = schema.get("items")
        return TypeShape(
            constructor=TypeConstructor.ARRAY,
            item=type_shape_from_json_schema(
                items if items is not None else {"type": "any"},
                definitions=defs,
                depth=depth + 1,
                ref_stack=ref_stack,
                missing_refs=missing_refs,
            ),
            nullable=nullable,
            name=_text(schema.get("title") or "", field_name="title", required=False),
        )

    if type_field == "object" or (
        isinstance(type_field, str) and type_field.lower() == "object"
    ) or (
        type_field is None and isinstance(schema.get("properties"), Mapping)
    ):
        properties = schema.get("properties") or {}
        fields: list[tuple[str, TypeShape]] = []
        if isinstance(properties, Mapping):
            for name, prop in list(properties.items())[:128]:
                fields.append(
                    (
                        str(name),
                        type_shape_from_json_schema(
                            prop,
                            definitions=defs,
                            depth=depth + 1,
                            ref_stack=ref_stack,
                            missing_refs=missing_refs,
                        ),
                    )
                )
        constraints: list[str] = []
        required = schema.get("required") or ()
        if isinstance(required, Sequence) and not isinstance(
            required, (str, bytes, bytearray)
        ):
            for name in required[:64]:
                constraints.append(f"required:{name}")
        return TypeShape(
            constructor=TypeConstructor.OBJECT,
            fields=tuple(fields),
            nullable=nullable,
            name=_text(schema.get("title") or "", field_name="title", required=False),
            constraints=tuple(constraints),
        )

    if isinstance(type_field, str):
        if type_field == "any":
            return TypeShape(constructor=TypeConstructor.ANY, nullable=nullable)
        constructor = _JSON_TYPE_MAP.get(type_field.lower())
        if constructor is not None:
            return TypeShape(
                constructor=constructor,
                name=type_field,
                nullable=nullable,
            )

    if schema.get("const") is not None:
        return TypeShape(
            constructor=TypeConstructor.ENUM,
            enum_values=(str(schema.get("const")),),
            nullable=nullable,
        )

    return TypeShape(
        constructor=TypeConstructor.UNKNOWN,
        support=SupportStatus.UNKNOWN,
        nullable=nullable,
        name=_text(schema.get("title") or "", field_name="title", required=False),
    )


def _resolve_json_ref(
    ref: str,
    definitions: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Resolve local JSON Schema ``$ref`` forms only (``#/$defs/X`` etc.)."""

    if not ref.startswith("#"):
        return None
    path = ref[1:]
    if path.startswith("/"):
        path = path[1:]
    if not path:
        return dict(schema)
    parts = path.split("/")
    # Common: $defs/Name or definitions/Name
    if len(parts) >= 2 and parts[0] in {"$defs", "definitions", "defs"}:
        name = parts[1]
        target = definitions.get(name)
        if isinstance(target, Mapping):
            return target
        # Also check embedded schema blocks.
        for key in ("$defs", "definitions"):
            block = schema.get(key)
            if isinstance(block, Mapping) and isinstance(block.get(name), Mapping):
                return block[name]
        return None
    cursor: Any = schema
    for part in parts:
        if not isinstance(cursor, Mapping) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor if isinstance(cursor, Mapping) else None


# ---------------------------------------------------------------------------
# Source unit and intermediate partial contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContractSourceUnit(CanonicalContract):
    """Compact, artifact-referenced unit of contract source material.

    Bodies are structured facts only.  Large IDL/schema/test corpora stay in
    content-addressed storage; this record holds the extraction-facing view.
    """

    SCHEMA: ClassVar[str] = CONTRACT_SOURCE_UNIT_SCHEMA

    artifact_id: str
    source_kind: ContractSourceKind
    payload: Mapping[str, Any]
    repository_id: str = ""
    tree_id: str = ""
    module_path: str = ""
    symbol_name: str = ""
    interface_name: str = ""
    surface: str = ""
    method: str = ""
    protocol: str = ""
    version: str = ""
    path_or_uri: str = ""
    locator: str = ""
    language: str = ""
    media_type: str = ""
    blob_cid: str = ""
    sha256: str = ""
    span_start: int | None = None
    span_end: int | None = None
    artifact_class: SourceArtifactClass = SourceArtifactClass.NORMATIVE
    extraction_rule: ExtractionRule | None = None
    policy_revision: str = ""
    repository_observation_id: str = ""
    producer_id: str = ""
    producer_version: str = ""
    definitions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "artifact_id",
            _text(self.artifact_id, field_name="artifact_id"),
        )
        object.__setattr__(
            self,
            "source_kind",
            _enum(self.source_kind, ContractSourceKind, field_name="source_kind"),
        )
        payload = _mapping(self.payload, field_name="payload")
        # Freeze payload as a plain dict of JSON-compatible values only.
        object.__setattr__(self, "payload", dict(payload))
        for name in (
            "repository_id",
            "tree_id",
            "module_path",
            "symbol_name",
            "interface_name",
            "surface",
            "method",
            "protocol",
            "version",
            "path_or_uri",
            "locator",
            "language",
            "media_type",
            "blob_cid",
            "policy_revision",
            "repository_observation_id",
            "producer_id",
            "producer_version",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=False),
            )
        object.__setattr__(
            self,
            "artifact_class",
            _enum(
                self.artifact_class,
                SourceArtifactClass,
                field_name="artifact_class",
            ),
        )
        if self.extraction_rule is not None:
            object.__setattr__(
                self,
                "extraction_rule",
                _enum(
                    self.extraction_rule,
                    ExtractionRule,
                    field_name="extraction_rule",
                ),
            )
        object.__setattr__(
            self,
            "span_start",
            _optional_int(self.span_start, field_name="span_start"),
        )
        object.__setattr__(
            self,
            "span_end",
            _optional_int(self.span_end, field_name="span_end"),
        )
        if (
            self.span_start is not None
            and self.span_end is not None
            and self.span_end < self.span_start
        ):
            raise ContractExtractorError("span_end must be >= span_start")
        sha = self.sha256
        if not sha:
            sha = _content_sha256(
                {
                    "artifact_id": self.artifact_id,
                    "source_kind": self.source_kind.value,
                    "payload": self.payload,
                    "locator": self.locator,
                }
            )
        elif not str(sha).startswith("sha256:"):
            if re.fullmatch(r"[0-9a-f]{64}", str(sha)):
                sha = f"sha256:{sha}"
            else:
                raise ContractExtractorError("sha256 must be sha256:<hex>")
        object.__setattr__(self, "sha256", sha)
        defs = self.definitions or {}
        if not isinstance(defs, Mapping):
            raise ContractExtractorError("definitions must be a mapping")
        object.__setattr__(self, "definitions", dict(defs))
        # Observation units must use observation source kind / class.
        if self.source_kind is ContractSourceKind.IMPLEMENTATION_OBSERVATION:
            if self.artifact_class is SourceArtifactClass.NORMATIVE:
                object.__setattr__(
                    self, "artifact_class", SourceArtifactClass.OBSERVATION
                )
        # Fail closed: observation kind cannot be an expectation-class unit.
        if (
            self.source_kind is ContractSourceKind.IMPLEMENTATION_OBSERVATION
            and self.artifact_class.may_define_expectation
            and self.artifact_class is not SourceArtifactClass.OBSERVATION
        ):
            raise CircularExpectationError(
                "implementation observations cannot be classified as "
                "expectation-authoritative artifacts"
            )

    @property
    def unit_id(self) -> str:
        return self.content_id

    @property
    def content_kind(self) -> ContentKind:
        return _infer_content_kind(self.payload)

    @property
    def resolved_extraction_rule(self) -> ExtractionRule:
        if self.extraction_rule is not None:
            return self.extraction_rule
        return extraction_rule_for(
            self.source_kind,
            self.content_kind,
            artifact_class=self.artifact_class,
        )

    @property
    def confidence(self) -> ConfidenceClass:
        return confidence_for(self.source_kind, self.artifact_class)

    def subject_key(self) -> str:
        """Stable subject key for grouping units that describe one surface."""

        interface = self.interface_name or self.symbol_name or self.method
        symbol = self.symbol_name or self.method or interface
        surface = self.surface or self.protocol or "default"
        return "|".join(
            (
                self.repository_id or "",
                self.tree_id or "",
                surface,
                interface or "",
                symbol or "",
                self.method or "",
            )
        )

    def to_source_reference(
        self,
        *,
        role: ProgramContractRole,
    ) -> SourceReference:
        if (
            role is ProgramContractRole.EXPECTED
            and not self.source_kind.may_define_expectation
        ):
            raise CircularExpectationError(
                "implementation observations cannot define expectations"
            )
        return SourceReference(
            source_kind=self.source_kind,
            role=role,
            artifact_id=self.artifact_id,
            locator=self.locator
            or self.path_or_uri
            or self.module_path
            or self.artifact_id,
            extractor_rule=self.resolved_extraction_rule.value,
            confidence=self.confidence,
            sha256=self.sha256,
            span_start=self.span_start,
            span_end=self.span_end,
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_EXTRACTOR_VERSION,
            "artifact_id": self.artifact_id,
            "source_kind": self.source_kind.value,
            "payload": self.payload,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "module_path": self.module_path,
            "symbol_name": self.symbol_name,
            "interface_name": self.interface_name,
            "surface": self.surface,
            "method": self.method,
            "protocol": self.protocol,
            "version": self.version,
            "path_or_uri": self.path_or_uri,
            "locator": self.locator,
            "language": self.language,
            "media_type": self.media_type,
            "blob_cid": self.blob_cid,
            "sha256": self.sha256,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "artifact_class": self.artifact_class.value,
            "extraction_rule": (
                None
                if self.extraction_rule is None
                else self.extraction_rule.value
            ),
            "policy_revision": self.policy_revision,
            "repository_observation_id": self.repository_observation_id,
            "producer_id": self.producer_id,
            "producer_version": self.producer_version,
            "definitions": dict(self.definitions),
        }


@dataclass(frozen=True)
class SkippedSource:
    """A source unit deliberately excluded from expectation formation."""

    SCHEMA: ClassVar[str] = SKIPPED_SOURCE_SCHEMA

    artifact_id: str
    reason: SkipReason
    artifact_class: SourceArtifactClass
    source_kind: ContractSourceKind
    extraction_rule: ExtractionRule
    locator: str = ""
    summary: str = ""
    sha256: str = ""
    span_start: int | None = None
    span_end: int | None = None
    blob_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "artifact_id", _text(self.artifact_id, field_name="artifact_id")
        )
        object.__setattr__(
            self, "reason", _enum(self.reason, SkipReason, field_name="reason")
        )
        object.__setattr__(
            self,
            "artifact_class",
            _enum(
                self.artifact_class,
                SourceArtifactClass,
                field_name="artifact_class",
            ),
        )
        object.__setattr__(
            self,
            "source_kind",
            _enum(self.source_kind, ContractSourceKind, field_name="source_kind"),
        )
        object.__setattr__(
            self,
            "extraction_rule",
            _enum(
                self.extraction_rule,
                ExtractionRule,
                field_name="extraction_rule",
            ),
        )
        for name in ("locator", "summary", "sha256", "blob_cid"):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), field_name=name, required=False),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "artifact_id": self.artifact_id,
            "reason": self.reason.value,
            "artifact_class": self.artifact_class.value,
            "source_kind": self.source_kind.value,
            "extraction_rule": self.extraction_rule.value,
            "locator": self.locator,
            "summary": self.summary,
            "sha256": self.sha256,
            "span_start": self.span_start,
            "span_end": self.span_end,
            "blob_cid": self.blob_cid,
        }


@dataclass
class _PartialAspects:
    """Mutable accumulator for one subject's aspect clauses before merge."""

    sources: list[SourceReference] = field(default_factory=list)
    units: list[ContractSourceUnit] = field(default_factory=list)
    inputs_by_source: list[tuple[SourceReference, tuple[ParameterSpec, ...]]] = (
        field(default_factory=list)
    )
    returns_by_source: list[tuple[SourceReference, ReturnSpec]] = field(
        default_factory=list
    )
    errors_by_source: list[tuple[SourceReference, tuple[ErrorSpec, ...]]] = field(
        default_factory=list
    )
    sync_by_source: list[tuple[SourceReference, SyncAsyncSpec]] = field(
        default_factory=list
    )
    effects_by_source: list[
        tuple[SourceReference, tuple[SideEffectSpec, ...]]
    ] = field(default_factory=list)
    capabilities_by_source: list[
        tuple[SourceReference, tuple[CapabilitySpec, ...]]
    ] = field(default_factory=list)
    authorization_by_source: list[tuple[SourceReference, AuthorizationSpec]] = (
        field(default_factory=list)
    )
    idempotence_by_source: list[tuple[SourceReference, IdempotenceSpec]] = field(
        default_factory=list
    )
    ordering_by_source: list[tuple[SourceReference, OrderingSpec]] = field(
        default_factory=list
    )
    atomicity_by_source: list[tuple[SourceReference, AtomicitySpec]] = field(
        default_factory=list
    )
    consistency_by_source: list[tuple[SourceReference, ConsistencySpec]] = field(
        default_factory=list
    )
    bounds_by_source: list[tuple[SourceReference, ResourceBounds]] = field(
        default_factory=list
    )
    fallback_by_source: list[tuple[SourceReference, FallbackSpec]] = field(
        default_factory=list
    )
    applicability_by_source: list[tuple[SourceReference, Applicability]] = field(
        default_factory=list
    )
    assumptions: list[Assumption] = field(default_factory=list)
    unsupported: list[UnsupportedSemantics] = field(default_factory=list)
    conflicts: list[ContractConflict] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)
    symbol: SymbolIdentity | None = None
    interface: InterfaceIdentity | None = None
    policy_revision: str = ""
    # Observation-only accumulators
    observation_units: list[ContractSourceUnit] = field(default_factory=list)
    observed_partials: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ContractExtractionResult(CanonicalContract):
    """Result of a complete multi-source extraction pass."""

    SCHEMA: ClassVar[str] = EXTRACTION_RESULT_SCHEMA

    repository_id: str
    tree_id: str
    policy_revision: str
    expected: tuple[ExpectedProgramContract, ...]
    observed: tuple[ObservedProgramContract, ...]
    conflicts: tuple[ContractConflict, ...]
    unsupported: tuple[UnsupportedSemantics, ...]
    skipped: tuple[SkippedSource, ...]
    sources: tuple[SourceReference, ...]
    summary: str = ""
    extractor_version: int = CONTRACT_EXTRACTOR_VERSION

    def __post_init__(self) -> None:
        for name in ("repository_id", "tree_id", "policy_revision"):
            object.__setattr__(
                self, name, _text(getattr(self, name), field_name=name)
            )
        object.__setattr__(self, "expected", tuple(self.expected or ()))
        object.__setattr__(self, "observed", tuple(self.observed or ()))
        object.__setattr__(self, "conflicts", tuple(self.conflicts or ()))
        object.__setattr__(self, "unsupported", tuple(self.unsupported or ()))
        object.__setattr__(self, "skipped", tuple(self.skipped or ()))
        object.__setattr__(self, "sources", tuple(self.sources or ()))
        object.__setattr__(
            self,
            "summary",
            _text(self.summary, field_name="summary", required=False),
        )
        if len(self.expected) > MAX_COLLECTION_ITEMS:
            raise ContractExtractorBoundsError("expected exceeds collection bound")
        if len(self.observed) > MAX_COLLECTION_ITEMS:
            raise ContractExtractorBoundsError("observed exceeds collection bound")
        if len(self.skipped) > MAX_SKIPPED:
            raise ContractExtractorBoundsError("skipped exceeds bound")

    @property
    def extraction_id(self) -> str:
        return self.content_id

    @property
    def has_conflicts(self) -> bool:
        return bool(self.conflicts) or any(
            contract.has_conflicts for contract in self.expected
        )

    def to_bundle(self) -> ProgramContractBundle:
        """Materialize a :class:`ProgramContractBundle` from this result."""

        bundle_conflicts = list(self.conflicts)
        for contract in self.expected:
            bundle_conflicts.extend(contract.conflicts)
        # Deduplicate by conflict_id while preserving order.
        seen: set[str] = set()
        unique: list[ContractConflict] = []
        for item in bundle_conflicts:
            if item.conflict_id in seen:
                continue
            seen.add(item.conflict_id)
            unique.append(item)
            if len(unique) >= MAX_CONFLICTS:
                break
        return ProgramContractBundle(
            repository_id=self.repository_id,
            tree_id=self.tree_id,
            policy_revision=self.policy_revision,
            expected=self.expected,
            observed=self.observed,
            refinements=(),
            conflicts=tuple(unique),
            summary=self.summary
            or (
                f"extracted {len(self.expected)} expected, "
                f"{len(self.observed)} observed contracts"
            ),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_EXTRACTOR_VERSION,
            "extractor_version": self.extractor_version,
            "repository_id": self.repository_id,
            "tree_id": self.tree_id,
            "policy_revision": self.policy_revision,
            "expected": [item.to_dict() for item in self.expected],
            "observed": [item.to_dict() for item in self.observed],
            "conflicts": [item.to_dict() for item in self.conflicts],
            "unsupported": [item.to_dict() for item in self.unsupported],
            "skipped": [item.to_dict() for item in self.skipped],
            "sources": [item.to_dict() for item in self.sources],
            "summary": self.summary,
        }

    def to_record(self) -> dict[str, Any]:
        return {
            **self.to_dict(),
            "extraction_id": self.extraction_id,
            "has_conflicts": self.has_conflicts,
            "bundle_id": self.to_bundle().bundle_id,
            "evidence": list(CONTRACT_EXTRACTOR_EVIDENCE),
            "evidence_contract_ir": CONTRACT_IR_EVIDENCE,
            "evidence_source_precedence": CONTRACT_SOURCE_PRECEDENCE_EVIDENCE,
        }


# ---------------------------------------------------------------------------
# Per-unit extraction
# ---------------------------------------------------------------------------


def _symbol_from_unit(unit: ContractSourceUnit) -> SymbolIdentity:
    symbol_name = (
        unit.symbol_name
        or unit.method
        or unit.interface_name
        or unit.payload.get("name")
        or unit.payload.get("symbol")
        or "unknown"
    )
    if not isinstance(symbol_name, str):
        symbol_name = str(symbol_name)
    module_path = unit.module_path or unit.locator or unit.artifact_id
    return SymbolIdentity(
        repository_id=unit.repository_id or "repository:unknown",
        tree_id=unit.tree_id or "tree:unknown",
        module_path=module_path,
        symbol_name=symbol_name,
        language=unit.language,
        span_start=unit.span_start,
        span_end=unit.span_end,
        blob_cid=unit.blob_cid,
    )


def _interface_from_unit(unit: ContractSourceUnit) -> InterfaceIdentity:
    payload = unit.payload
    name = (
        unit.interface_name
        or payload.get("name")
        or payload.get("interface")
        or unit.symbol_name
        or unit.method
        or unit.artifact_id
    )
    surface = unit.surface or payload.get("surface") or "api"
    method = unit.method or payload.get("method") or ""
    protocol = unit.protocol or payload.get("protocol") or ""
    version = unit.version or payload.get("version") or ""
    path_or_uri = unit.path_or_uri or payload.get("path_or_uri") or unit.locator
    if unit.content_kind in {
        ContentKind.MCP_TOOL,
        ContentKind.MCP_IDL,
        ContentKind.MCP_PLUS_PLUS,
    }:
        surface = unit.surface or payload.get("surface") or "mcp++"
        protocol = protocol or "mcp"
    return InterfaceIdentity(
        interface_name=str(name),
        surface=str(surface),
        version=str(version) if version is not None else "",
        method=str(method) if method is not None else "",
        protocol=str(protocol) if protocol is not None else "",
        media_type=unit.media_type,
        path_or_uri=str(path_or_uri) if path_or_uri is not None else "",
    )


def _parse_parameters(
    raw: Any,
    *,
    definitions: Mapping[str, Any] | None = None,
    missing_refs: list[str] | None = None,
) -> tuple[ParameterSpec, ...]:
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        # JSON Schema object properties form.
        if "properties" in raw or raw.get("type") == "object":
            props = raw.get("properties") or {}
            required = set(raw.get("required") or ())
            params: list[ParameterSpec] = []
            if isinstance(props, Mapping):
                for index, (name, schema) in enumerate(props.items()):
                    params.append(
                        ParameterSpec(
                            name=str(name),
                            type_shape=type_shape_from_json_schema(
                                schema,
                                definitions=definitions,
                                missing_refs=missing_refs,
                            ),
                            kind=ParameterKind.KEYWORD,
                            optionality=(
                                Optionality.REQUIRED
                                if name in required
                                else Optionality.OPTIONAL
                            ),
                            position=index,
                            description=_text(
                                (schema or {}).get("description")
                                if isinstance(schema, Mapping)
                                else "",
                                field_name="description",
                                required=False,
                            )
                            if isinstance(schema, Mapping)
                            else "",
                        )
                    )
            return tuple(params)
        # Single named parameter map.
        if all(isinstance(v, (str, Mapping)) for v in raw.values()):
            params = []
            for index, (name, typ) in enumerate(raw.items()):
                params.append(
                    ParameterSpec(
                        name=str(name),
                        type_shape=type_shape_from_json_schema(
                            typ,
                            definitions=definitions,
                            missing_refs=missing_refs,
                        )
                        if isinstance(typ, Mapping)
                        else type_shape_from_name(typ),
                        kind=ParameterKind.KEYWORD,
                        optionality=Optionality.REQUIRED,
                        position=index,
                    )
                )
            return tuple(params)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ContractExtractorError("parameters must be a sequence or mapping")
    params = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            params.append(
                ParameterSpec(
                    name=item,
                    type_shape=TypeShape(constructor=TypeConstructor.ANY),
                    kind=ParameterKind.POSITIONAL,
                    optionality=Optionality.REQUIRED,
                    position=index,
                )
            )
            continue
        if not isinstance(item, Mapping):
            raise ContractExtractorError(
                f"parameters[{index}] must be a mapping"
            )
        name = item.get("name") or item.get("param") or f"arg{index}"
        typ = (
            item.get("type_shape")
            or item.get("type")
            or item.get("schema")
            or item.get("typeShape")
        )
        if isinstance(typ, Mapping):
            shape = type_shape_from_json_schema(
                typ, definitions=definitions, missing_refs=missing_refs
            )
        else:
            shape = type_shape_from_name(typ)
        optional = item.get("optional")
        optionality_raw = item.get("optionality")
        if optionality_raw is not None:
            optionality = _enum(
                optionality_raw, Optionality, field_name="optionality"
            )
        elif optional is True:
            optionality = Optionality.OPTIONAL
        elif optional is False:
            optionality = Optionality.REQUIRED
        else:
            optionality = Optionality.REQUIRED
        kind_raw = item.get("kind") or item.get("parameter_kind") or "positional"
        try:
            kind = ParameterKind(str(kind_raw))
        except ValueError:
            kind = ParameterKind.OTHER
        params.append(
            ParameterSpec(
                name=str(name),
                type_shape=shape,
                kind=kind,
                optionality=optionality,
                position=item.get("position", index),
                default_summary=_text(
                    item.get("default") or item.get("default_summary") or "",
                    field_name="default_summary",
                    required=False,
                ),
                description=_text(
                    item.get("description") or "",
                    field_name="description",
                    required=False,
                ),
            )
        )
    return tuple(params)


def _parse_returns(
    raw: Any,
    *,
    definitions: Mapping[str, Any] | None = None,
    missing_refs: list[str] | None = None,
) -> ReturnSpec | None:
    if raw is None:
        return None
    if isinstance(raw, ReturnSpec):
        return raw
    if isinstance(raw, str):
        return ReturnSpec(type_shape=type_shape_from_name(raw))
    if isinstance(raw, Mapping):
        if (
            "type" in raw
            or "properties" in raw
            or "$ref" in raw
            or "oneOf" in raw
            or "anyOf" in raw
        ) and "type_shape" not in raw and "constructor" not in raw:
            shape = type_shape_from_json_schema(
                raw, definitions=definitions, missing_refs=missing_refs
            )
            return ReturnSpec(
                type_shape=shape,
                description=_text(
                    raw.get("description") or "",
                    field_name="description",
                    required=False,
                ),
            )
        typ = raw.get("type_shape") or raw.get("type") or raw.get("schema")
        if isinstance(typ, Mapping) and "constructor" in typ:
            shape = TypeShape.from_dict(typ) if "schema" in typ else type_shape_from_json_schema(
                typ, definitions=definitions, missing_refs=missing_refs
            )
            # Prefer TypeShape fields if present.
            if "constructor" in typ and (
                "schema" in typ
                or typ.get("constructor")
                in {c.value for c in TypeConstructor}
            ):
                try:
                    if "contract_version" in typ or "schema" in typ:
                        shape = TypeShape.from_dict(typ)
                    else:
                        shape = TypeShape(
                            constructor=typ.get("constructor"),
                            name=typ.get("name", ""),
                            nullable=bool(typ.get("nullable", False)),
                            item=typ.get("item"),
                            fields=tuple(
                                (f["name"], f["type"])
                                if isinstance(f, Mapping)
                                else f
                                for f in (typ.get("fields") or ())
                            ),
                            alternatives=tuple(typ.get("alternatives") or ()),
                            enum_values=tuple(typ.get("enum_values") or ()),
                            reference=typ.get("reference", ""),
                            constraints=tuple(typ.get("constraints") or ()),
                            support=typ.get("support", SupportStatus.SUPPORTED),
                        )
                except (ProgramContractError, ContractValidationError, TypeError):
                    shape = type_shape_from_json_schema(
                        typ, definitions=definitions, missing_refs=missing_refs
                    )
        elif isinstance(typ, Mapping):
            shape = type_shape_from_json_schema(
                typ, definitions=definitions, missing_refs=missing_refs
            )
        else:
            shape = type_shape_from_name(typ)
        optionality = raw.get("optionality", Optionality.REQUIRED)
        return ReturnSpec(
            type_shape=shape,
            optionality=_enum(optionality, Optionality, field_name="optionality")
            if not isinstance(optionality, Optionality)
            else optionality,
            description=_text(
                raw.get("description") or "",
                field_name="description",
                required=False,
            ),
            multi_value=bool(raw.get("multi_value", False)),
        )
    return ReturnSpec(type_shape=type_shape_from_name(str(raw)))


def _parse_errors(raw: Any) -> tuple[ErrorSpec, ...]:
    if not raw:
        return ()
    if isinstance(raw, Mapping):
        raw = [
            {"error_name": key, **(value if isinstance(value, Mapping) else {"code": str(value)})}
            for key, value in raw.items()
        ]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ContractExtractorError("errors must be a sequence or mapping")
    errors: list[ErrorSpec] = []
    for item in raw:
        if isinstance(item, str):
            errors.append(ErrorSpec(error_name=item))
            continue
        if not isinstance(item, Mapping):
            continue
        name = (
            item.get("error_name")
            or item.get("name")
            or item.get("error")
            or item.get("code")
            or "Error"
        )
        code = item.get("code") or ""
        retriable = bool(item.get("retriable", False))
        conditions = item.get("conditions") or ()
        if isinstance(conditions, str):
            conditions = (conditions,)
        err_type = item.get("error_type") or item.get("type")
        type_shape = None
        if err_type is not None:
            type_shape = (
                type_shape_from_json_schema(err_type)
                if isinstance(err_type, Mapping)
                else type_shape_from_name(err_type)
            )
        errors.append(
            ErrorSpec(
                error_name=str(name),
                error_type=type_shape,
                code=str(code) if code is not None else "",
                retriable=retriable,
                conditions=tuple(str(c) for c in conditions),
            )
        )
    return tuple(errors)


def _parse_sync(raw: Any) -> SyncAsyncSpec | None:
    if raw is None:
        return None
    if isinstance(raw, SyncAsyncSpec):
        return raw
    if isinstance(raw, bool):
        return SyncAsyncSpec(
            mode=SyncMode.ASYNC if raw else SyncMode.SYNC,
            awaitable=raw,
        )
    if isinstance(raw, str):
        lowered = raw.lower()
        if lowered in {"async", "asynchronous", "awaitable"}:
            return SyncAsyncSpec(mode=SyncMode.ASYNC, awaitable=True)
        if lowered in {"sync", "synchronous", "blocking"}:
            return SyncAsyncSpec(mode=SyncMode.SYNC)
        if lowered in {"dual", "both"}:
            return SyncAsyncSpec(mode=SyncMode.DUAL, awaitable=True)
        return SyncAsyncSpec(mode=SyncMode.UNKNOWN)
    if isinstance(raw, Mapping):
        mode_raw = raw.get("mode")
        if mode_raw is None:
            if raw.get("async") is True:
                mode_raw = SyncMode.ASYNC
            elif raw.get("async") is False:
                mode_raw = SyncMode.SYNC
            else:
                mode_raw = SyncMode.UNKNOWN
        return SyncAsyncSpec(
            mode=_enum(mode_raw, SyncMode, field_name="mode")
            if not isinstance(mode_raw, SyncMode)
            else mode_raw,
            awaitable=bool(raw.get("awaitable", mode_raw in {SyncMode.ASYNC, "async"})),
            callback_style=bool(raw.get("callback_style", False)),
            description=_text(
                raw.get("description") or "",
                field_name="description",
                required=False,
            ),
        )
    return None


def _parse_effects(raw: Any) -> tuple[SideEffectSpec, ...]:
    if not raw:
        return ()
    if isinstance(raw, Mapping):
        raw = [raw]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ContractExtractorError("side_effects must be a sequence")
    effects: list[SideEffectSpec] = []
    for item in raw:
        if isinstance(item, str):
            try:
                kind = EffectKind(item)
            except ValueError:
                kind = EffectKind.UNKNOWN
            effects.append(
                SideEffectSpec(
                    effect_kind=kind,
                    polarity=EffectPolarity.ALLOWED,
                )
            )
            continue
        if not isinstance(item, Mapping):
            continue
        kind_raw = item.get("effect_kind") or item.get("kind") or item.get("effect")
        polarity_raw = (
            item.get("polarity") or item.get("mode") or EffectPolarity.ALLOWED
        )
        try:
            kind = (
                kind_raw
                if isinstance(kind_raw, EffectKind)
                else EffectKind(str(kind_raw))
            )
        except ValueError:
            kind = EffectKind.UNKNOWN
        try:
            polarity = (
                polarity_raw
                if isinstance(polarity_raw, EffectPolarity)
                else EffectPolarity(str(polarity_raw))
            )
        except ValueError:
            polarity = EffectPolarity.UNKNOWN
        effects.append(
            SideEffectSpec(
                effect_kind=kind,
                polarity=polarity,
                target=_text(
                    item.get("target") or "", field_name="target", required=False
                ),
                description=_text(
                    item.get("description") or "",
                    field_name="description",
                    required=False,
                ),
            )
        )
    return tuple(effects)


def _parse_capabilities(raw: Any) -> tuple[CapabilitySpec, ...]:
    if not raw:
        return ()
    if isinstance(raw, Mapping):
        # {name: mode} form
        if all(not isinstance(v, Mapping) for v in raw.values()):
            raw = [
                {"capability_name": k, "mode": v} for k, v in raw.items()
            ]
        else:
            raw = [
                {"capability_name": k, **(v if isinstance(v, Mapping) else {})}
                for k, v in raw.items()
            ]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ContractExtractorError("capabilities must be a sequence or mapping")
    caps: list[CapabilitySpec] = []
    for item in raw:
        if isinstance(item, str):
            caps.append(
                CapabilitySpec(
                    capability_name=item,
                    mode=CapabilityMode.REQUIRED,
                )
            )
            continue
        if not isinstance(item, Mapping):
            continue
        name = (
            item.get("capability_name")
            or item.get("name")
            or item.get("capability")
            or "capability"
        )
        mode_raw = item.get("mode") or item.get("capability_mode") or "required"
        if item.get("optional") is True:
            mode_raw = CapabilityMode.OPTIONAL
        if item.get("negotiated") is True:
            mode_raw = CapabilityMode.NEGOTIATED
        try:
            mode = (
                mode_raw
                if isinstance(mode_raw, CapabilityMode)
                else CapabilityMode(str(mode_raw))
            )
        except ValueError:
            mode = CapabilityMode.UNKNOWN
        caps.append(
            CapabilitySpec(
                capability_name=str(name),
                mode=mode,
                version=str(item.get("version") or ""),
                description=_text(
                    item.get("description") or "",
                    field_name="description",
                    required=False,
                ),
            )
        )
    return tuple(caps)


def _parse_authorization(raw: Any) -> AuthorizationSpec | None:
    if raw is None:
        return None
    if isinstance(raw, AuthorizationSpec):
        return raw
    if isinstance(raw, str):
        try:
            mode = AuthorizationMode(raw)
        except ValueError:
            mode = AuthorizationMode.UNKNOWN
        return AuthorizationSpec(mode=mode)
    if not isinstance(raw, Mapping):
        return None
    mode_raw = raw.get("mode") or AuthorizationMode.UNKNOWN
    try:
        mode = (
            mode_raw
            if isinstance(mode_raw, AuthorizationMode)
            else AuthorizationMode(str(mode_raw))
        )
    except ValueError:
        mode = AuthorizationMode.UNKNOWN
    scopes = raw.get("scopes") or ()
    principals = raw.get("principals") or ()
    policies = raw.get("policies") or ()
    if isinstance(scopes, str):
        scopes = (scopes,)
    if isinstance(principals, str):
        principals = (principals,)
    if isinstance(policies, str):
        policies = (policies,)
    return AuthorizationSpec(
        mode=mode,
        scopes=tuple(str(s) for s in scopes),
        principals=tuple(str(p) for p in principals),
        policies=tuple(str(p) for p in policies),
        description=_text(
            raw.get("description") or "",
            field_name="description",
            required=False,
        ),
    )


def _parse_simple_mode_spec(
    raw: Any,
    *,
    enum_type: type[Enum],
    factory: Any,
    field_name: str = "mode",
) -> Any | None:
    if raw is None:
        return None
    if isinstance(raw, enum_type):
        return factory(mode=raw)
    if isinstance(raw, str):
        try:
            return factory(mode=enum_type(raw))
        except ValueError:
            return factory(mode=enum_type("unknown") if "unknown" in {e.value for e in enum_type} else list(enum_type)[0])
    if isinstance(raw, Mapping):
        mode_raw = raw.get("mode") or raw.get(field_name)
        if mode_raw is None:
            return None
        try:
            mode = (
                mode_raw
                if isinstance(mode_raw, enum_type)
                else enum_type(str(mode_raw))
            )
        except ValueError:
            return None
        kwargs = {"mode": mode}
        if "description" in raw:
            kwargs["description"] = raw.get("description") or ""
        try:
            return factory(**kwargs)
        except TypeError:
            return factory(mode=mode)
    return None


def _parse_resource_bounds(raw: Any) -> ResourceBounds | None:
    if raw is None:
        return None
    if isinstance(raw, ResourceBounds):
        return raw
    if not isinstance(raw, Mapping):
        return None
    kwargs: dict[str, Any] = {}
    aliases = {
        "max_wall_time_ms": "max_wall_time_ms",
        "max_cpu_time_ms": "max_cpu_time_ms",
        "max_memory_bytes": "max_memory_bytes",
        "max_output_bytes": "max_output_bytes",
        "max_payload_bytes": "max_payload_bytes",
        "max_input_bytes": "max_payload_bytes",
        "max_calls": "max_calls",
        "max_concurrency": "max_concurrency",
    }
    for src, dest in aliases.items():
        if src in raw and raw[src] is not None and dest not in kwargs:
            kwargs[dest] = raw[src]
    if not kwargs:
        return None
    return ResourceBounds(**kwargs)


def _parse_fallback(raw: Any) -> FallbackSpec | None:
    if raw is None:
        return None
    if isinstance(raw, FallbackSpec):
        return raw
    if isinstance(raw, str):
        try:
            return FallbackSpec(mode=DegradationMode(raw))
        except ValueError:
            return FallbackSpec(
                mode=DegradationMode.UNKNOWN, description=raw
            )
    if isinstance(raw, Mapping):
        mode_raw = raw.get("mode") or DegradationMode.UNKNOWN
        try:
            mode = (
                mode_raw
                if isinstance(mode_raw, DegradationMode)
                else DegradationMode(str(mode_raw))
            )
        except ValueError:
            mode = DegradationMode.UNKNOWN
        return FallbackSpec(
            mode=mode,
            description=_text(
                raw.get("description") or "",
                field_name="description",
                required=False,
            ),
            fallback_interface=_text(
                raw.get("fallback_interface") or raw.get("fallback") or "",
                field_name="fallback_interface",
                required=False,
            ),
        )
    return None


def _parse_applicability(raw: Any) -> Applicability | None:
    if raw is None:
        return None
    if isinstance(raw, Applicability):
        return raw
    if isinstance(raw, Mapping):
        versions = raw.get("versions") or raw.get("version_range") or ()
        if isinstance(versions, str):
            versions = (versions,)
        surfaces = raw.get("surfaces") or ()
        if isinstance(surfaces, str):
            surfaces = (surfaces,)
        environments = raw.get("environments") or ()
        if isinstance(environments, str):
            environments = (environments,)
        conditions = raw.get("conditions") or ()
        if isinstance(conditions, str):
            conditions = (conditions,)
        return Applicability(
            conditions=tuple(str(c) for c in conditions),
            surfaces=tuple(str(s) for s in surfaces),
            environments=tuple(str(e) for e in environments),
            versions=tuple(str(v) for v in versions),
            always=bool(raw.get("always", not (versions or conditions))),
            description=_text(
                raw.get("description") or "",
                field_name="description",
                required=False,
            ),
        )
    return None


def _extract_clause_dict(
    unit: ContractSourceUnit,
) -> tuple[dict[str, Any], list[UnsupportedSemantics], list[str]]:
    """Extract aspect fields from one unit into a plain clause dict."""

    payload = unit.payload
    content_kind = unit.content_kind
    missing_refs: list[str] = []
    unsupported: list[UnsupportedSemantics] = []
    definitions = dict(unit.definitions)
    for key in ("$defs", "definitions"):
        block = payload.get(key)
        if isinstance(block, Mapping):
            definitions.update(dict(block))

    # Normalize MCP-style keys.
    body = dict(payload)
    if content_kind in {
        ContentKind.MCP_TOOL,
        ContentKind.MCP_IDL,
        ContentKind.MCP_PLUS_PLUS,
    }:
        if "inputSchema" in body and "inputs" not in body:
            body["inputs"] = body["inputSchema"]
        if "input_schema" in body and "inputs" not in body:
            body["inputs"] = body["input_schema"]
        if "outputSchema" in body and "returns" not in body:
            body["returns"] = body["outputSchema"]
        if "output_schema" in body and "returns" not in body:
            body["returns"] = body["output_schema"]
        if body.get("async") is not None and "sync_async" not in body:
            body["sync_async"] = body["async"]

    if content_kind is ContentKind.JSON_SCHEMA:
        role = str(body.get("role") or body.get("schema_role") or "input").lower()
        schema_body = body.get("schema") or body
        if role in {"output", "returns", "return", "result"}:
            body = {**body, "returns": schema_body}
        else:
            body = {**body, "inputs": schema_body}

    if content_kind is ContentKind.CONTRACT_TEST:
        asserts = body.get("asserts") or body.get("expectations") or body
        if isinstance(asserts, Mapping):
            body = {**body, **dict(asserts)}

    if content_kind is ContentKind.NORMATIVE_DOC:
        clauses = body.get("clauses") or {}
        if isinstance(clauses, Mapping):
            body = {**body, **dict(clauses)}

    if content_kind is ContentKind.OBSERVATION:
        observed = body.get("observed") or body.get("observation") or body
        if isinstance(observed, Mapping):
            body = {**body, **dict(observed)}

    if content_kind is ContentKind.OVERLOAD_SET:
        overloads = body.get("overloads") or ()
        if isinstance(overloads, Sequence) and overloads:
            # Represent overload set as a union of parameter arities under
            # unsupported overload residual when >1 distinct shapes appear.
            first = overloads[0]
            if isinstance(first, Mapping):
                body = {**body, **dict(first)}
            if len(overloads) > 1:
                unsupported.append(
                    UnsupportedSemantics(
                        aspect=SemanticAspect.INPUTS,
                        reason="overload_set_requires_disambiguation",
                        residual=f"{len(overloads)}_overloads",
                    )
                )

    if content_kind in {
        ContentKind.COMPAT_MANIFEST,
        ContentKind.GENERATED_SDK,
    }:
        tool = body.get("tool") or body.get("manifest") or body
        if isinstance(tool, Mapping):
            body = {**body, **dict(tool)}
            if "inputSchema" in tool and "inputs" not in body:
                body["inputs"] = tool["inputSchema"]
            if "outputSchema" in tool and "returns" not in body:
                body["returns"] = tool["outputSchema"]

    inputs = _parse_parameters(
        body.get("inputs") or body.get("parameters") or body.get("args"),
        definitions=definitions,
        missing_refs=missing_refs,
    )
    returns = _parse_returns(
        body.get("returns")
        or body.get("return")
        or body.get("output")
        or body.get("result"),
        definitions=definitions,
        missing_refs=missing_refs,
    )
    errors = _parse_errors(
        body.get("errors") or body.get("raises") or body.get("error_map")
    )
    sync_async = _parse_sync(
        body.get("sync_async")
        if "sync_async" in body
        else body.get("async")
        if "async" in body
        else body.get("sync")
    )
    effects = _parse_effects(
        body.get("side_effects") or body.get("effects")
    )
    capabilities = _parse_capabilities(body.get("capabilities"))
    # Version negotiation often appears as negotiated capabilities.
    if body.get("version_negotiation") or body.get("negotiate_version"):
        vn = body.get("version_negotiation") or body.get("negotiate_version")
        if isinstance(vn, Mapping):
            cap_name = str(vn.get("capability") or vn.get("name") or "version")
            capabilities = capabilities + (
                CapabilitySpec(
                    capability_name=cap_name,
                    mode=CapabilityMode.NEGOTIATED,
                    version=str(vn.get("version") or vn.get("range") or ""),
                    description="version negotiation",
                ),
            )
        elif isinstance(vn, str):
            capabilities = capabilities + (
                CapabilitySpec(
                    capability_name=vn,
                    mode=CapabilityMode.NEGOTIATED,
                    description="version negotiation",
                ),
            )
        elif vn is True:
            capabilities = capabilities + (
                CapabilitySpec(
                    capability_name="protocol.version",
                    mode=CapabilityMode.NEGOTIATED,
                    description="version negotiation",
                ),
            )
    authorization = _parse_authorization(body.get("authorization") or body.get("auth"))
    idempotence = _parse_simple_mode_spec(
        body.get("idempotence"),
        enum_type=IdempotenceMode,
        factory=IdempotenceSpec,
    )
    ordering = _parse_simple_mode_spec(
        body.get("ordering"),
        enum_type=OrderingMode,
        factory=OrderingSpec,
    )
    atomicity = _parse_simple_mode_spec(
        body.get("atomicity"),
        enum_type=AtomicityMode,
        factory=AtomicitySpec,
    )
    consistency = _parse_simple_mode_spec(
        body.get("consistency"),
        enum_type=ConsistencyMode,
        factory=ConsistencySpec,
    )
    bounds = _parse_resource_bounds(
        body.get("resource_bounds") or body.get("bounds")
    )
    fallback = _parse_fallback(
        body.get("fallback") or body.get("degradation")
    )
    applicability = _parse_applicability(
        body.get("applicability") or body.get("applies_to")
    )

    for ref in missing_refs:
        unsupported.append(
            UnsupportedSemantics(
                aspect=SemanticAspect.INPUTS
                if returns is None
                else SemanticAspect.OUTPUTS,
                reason="missing_schema_ref",
                residual=ref[:MAX_CLAUSE_BYTES],
            )
        )

    # Unsupported residual keywords (explicit opt-in for tests/docs).
    for residual in body.get("unsupported") or ():
        if isinstance(residual, Mapping):
            aspect_raw = residual.get("aspect") or SemanticAspect.IDENTITY
            try:
                aspect = (
                    aspect_raw
                    if isinstance(aspect_raw, SemanticAspect)
                    else SemanticAspect(str(aspect_raw))
                )
            except ValueError:
                aspect = SemanticAspect.IDENTITY
            unsupported.append(
                UnsupportedSemantics(
                    aspect=aspect,
                    reason=str(residual.get("reason") or "unsupported_clause"),
                    residual=str(residual.get("residual") or ""),
                )
            )
        elif isinstance(residual, str):
            unsupported.append(
                UnsupportedSemantics(
                    aspect=SemanticAspect.IDENTITY,
                    reason=residual,
                )
            )

    summary = _text(
        body.get("summary")
        or body.get("description")
        or payload.get("description")
        or "",
        field_name="summary",
        required=False,
    )

    clause = {
        "inputs": inputs,
        "returns": returns,
        "errors": errors,
        "sync_async": sync_async,
        "side_effects": effects,
        "capabilities": capabilities,
        "authorization": authorization,
        "idempotence": idempotence,
        "ordering": ordering,
        "atomicity": atomicity,
        "consistency": consistency,
        "resource_bounds": bounds,
        "fallback": fallback,
        "applicability": applicability,
        "summary": summary,
    }
    return clause, unsupported, missing_refs


# ---------------------------------------------------------------------------
# Merge / conflict detection
# ---------------------------------------------------------------------------


def _aspect_fingerprint(value: Any) -> str:
    """Stable fingerprint for conflict comparison (not caller-facing identity)."""

    if value is None:
        return "null"
    if hasattr(value, "to_dict"):
        return program_contract_content_identity(value.to_dict())
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return program_contract_content_identity(
            [
                item.to_dict() if hasattr(item, "to_dict") else item
                for item in value
            ]
        )
    return program_contract_content_identity(value)


def _select_by_precedence(
    entries: Sequence[tuple[SourceReference, Any]],
) -> tuple[Any | None, list[ContractConflict], list[SourceReference]]:
    """Pick the dominant value; emit conflicts for equal-rank disagreement."""

    if not entries:
        return None, [], []
    # Group by precedence rank (lower is stronger).
    best_rank = min(source.precedence_rank for source, _ in entries)
    dominant = [(source, value) for source, value in entries if source.precedence_rank == best_rank]
    # Weaker sources are ignored for the value but remain in provenance.
    all_sources = [source for source, _ in entries]
    if len(dominant) == 1:
        return dominant[0][1], [], all_sources

    # Multiple equal-rank sources: agree or conflict.
    fingerprints = {
        _aspect_fingerprint(value): (source, value) for source, value in dominant
    }
    if len(fingerprints) == 1:
        return dominant[0][1], [], all_sources

    conflicts: list[ContractConflict] = []
    # Pairwise conflicts among disagreeing dominant sources.
    items = list(fingerprints.values())
    for index, (left_source, left_value) in enumerate(items):
        for right_source, right_value in items[index + 1 :]:
            conflicts.append(
                ContractConflict(
                    kind=ConflictKind.PRECEDENCE_COLLISION,
                    aspect=SemanticAspect.SOURCE_PRECEDENCE,
                    left_source_id=left_source.source_id,
                    right_source_id=right_source.source_id,
                    summary="equal-precedence sources disagree",
                    left_summary=_aspect_fingerprint(left_value)[:64],
                    right_summary=_aspect_fingerprint(right_value)[:64],
                    resolved=False,
                )
            )
    # Prefer the first dominant entry by artifact_id for a deterministic pick
    # while still reporting the conflict.
    dominant_sorted = sorted(dominant, key=lambda pair: pair[0].artifact_id)
    return dominant_sorted[0][1], conflicts, all_sources


def _select_aspect(
    entries: Sequence[tuple[SourceReference, Any]],
    *,
    aspect: SemanticAspect,
    kind_on_mismatch: ConflictKind = ConflictKind.SOURCE_DISAGREEMENT,
) -> tuple[Any | None, list[ContractConflict]]:
    if not entries:
        return None, []
    best_rank = min(source.precedence_rank for source, _ in entries)
    dominant = [
        (source, value)
        for source, value in entries
        if source.precedence_rank == best_rank
    ]
    fingerprints = {}
    for source, value in dominant:
        fingerprints.setdefault(_aspect_fingerprint(value), []).append(
            (source, value)
        )
    if len(fingerprints) == 1:
        return dominant[0][1], []

    conflicts: list[ContractConflict] = []
    groups = list(fingerprints.values())
    for index, left_group in enumerate(groups):
        for right_group in groups[index + 1 :]:
            left_source, left_value = left_group[0]
            right_source, right_value = right_group[0]
            conflicts.append(
                ContractConflict(
                    kind=kind_on_mismatch,
                    aspect=aspect,
                    left_source_id=left_source.source_id,
                    right_source_id=right_source.source_id,
                    summary=f"sources disagree on {aspect.value}",
                    left_summary=str(_aspect_fingerprint(left_value))[:64],
                    right_summary=str(_aspect_fingerprint(right_value))[:64],
                    resolved=False,
                )
            )
    # Deterministic selection: strongest kind already filtered; break ties by
    # artifact_id so results are stable without silent resolution of the
    # conflict record itself.
    dominant_sorted = sorted(dominant, key=lambda pair: pair[0].artifact_id)
    return dominant_sorted[0][1], conflicts


def _cross_rank_type_conflicts(
    entries: Sequence[tuple[SourceReference, Any]],
    *,
    aspect: SemanticAspect,
) -> list[ContractConflict]:
    """Detect type disagreements between different precedence ranks.

    Higher-authority values win, but an explicit type mismatch against a
    weaker source is still recorded so reviewers can see documentation/test
    drift without elevating those sources.
    """

    if len(entries) < 2:
        return []
    by_rank: dict[int, list[tuple[SourceReference, Any]]] = defaultdict(list)
    for source, value in entries:
        by_rank[source.precedence_rank].append((source, value))
    ranks = sorted(by_rank)
    if len(ranks) < 2:
        return []
    conflicts: list[ContractConflict] = []
    strongest_group = sorted(
        by_rank[ranks[0]],
        key=lambda pair: (pair[0].artifact_id, pair[0].source_id),
    )
    strongest = strongest_group[0]
    strong_fp = _aspect_fingerprint(strongest[1])
    for rank in ranks[1:]:
        for source, value in sorted(
            by_rank[rank],
            key=lambda pair: (pair[0].artifact_id, pair[0].source_id),
        ):
            if _aspect_fingerprint(value) != strong_fp:
                conflicts.append(
                    ContractConflict(
                        kind=ConflictKind.TYPE_MISMATCH
                        if aspect
                        in {SemanticAspect.INPUTS, SemanticAspect.OUTPUTS}
                        else ConflictKind.SOURCE_DISAGREEMENT,
                        aspect=aspect,
                        left_source_id=strongest[0].source_id,
                        right_source_id=source.source_id,
                        summary=(
                            f"weaker source disagrees with dominant on "
                            f"{aspect.value}"
                        ),
                        left_summary=str(strong_fp)[:64],
                        right_summary=str(_aspect_fingerprint(value))[:64],
                        resolved=False,
                    )
                )
    return conflicts


def _merge_partial(
    partial: _PartialAspects,
    *,
    repository_id: str,
    tree_id: str,
    policy_revision: str,
) -> tuple[
    ExpectedProgramContract | None,
    list[ObservedProgramContract],
    list[ContractConflict],
    list[UnsupportedSemantics],
]:
    conflicts: list[ContractConflict] = list(partial.conflicts)
    unsupported: list[UnsupportedSemantics] = list(partial.unsupported)

    # Self-expectation: observation units must not be the sole expectation path.
    expectation_units = [
        unit
        for unit in partial.units
        if unit.source_kind.may_define_expectation
        and unit.artifact_class.may_define_expectation
    ]

    expected_contract: ExpectedProgramContract | None = None
    if expectation_units:
        sources = list(partial.sources)
        # Ensure every expectation unit contributes a SourceReference.
        existing_ids = {s.artifact_id for s in sources}
        for unit in expectation_units:
            if unit.artifact_id not in existing_ids:
                sources.append(
                    unit.to_source_reference(role=ProgramContractRole.EXPECTED)
                )
        # Deterministic provenance order (rank, then artifact id).
        sources = sorted(
            sources,
            key=lambda item: (item.precedence_rank, item.artifact_id, item.source_id),
        )

        inputs, c1 = _select_aspect(
            partial.inputs_by_source,
            aspect=SemanticAspect.INPUTS,
            kind_on_mismatch=ConflictKind.TYPE_MISMATCH,
        )
        returns, c2 = _select_aspect(
            partial.returns_by_source,
            aspect=SemanticAspect.OUTPUTS,
            kind_on_mismatch=ConflictKind.TYPE_MISMATCH,
        )
        errors, c3 = _select_aspect(
            partial.errors_by_source, aspect=SemanticAspect.ERRORS
        )
        sync_async, c4 = _select_aspect(
            partial.sync_by_source, aspect=SemanticAspect.SYNC_ASYNC
        )
        effects, c5 = _select_aspect(
            partial.effects_by_source,
            aspect=SemanticAspect.SIDE_EFFECTS,
            kind_on_mismatch=ConflictKind.EFFECT_MISMATCH,
        )
        capabilities, c6 = _select_aspect(
            partial.capabilities_by_source, aspect=SemanticAspect.CAPABILITIES
        )
        authorization, c7 = _select_aspect(
            partial.authorization_by_source, aspect=SemanticAspect.AUTHORIZATION
        )
        idempotence, c8 = _select_aspect(
            partial.idempotence_by_source, aspect=SemanticAspect.IDEMPOTENCE
        )
        ordering, c9 = _select_aspect(
            partial.ordering_by_source, aspect=SemanticAspect.ORDERING
        )
        atomicity, c10 = _select_aspect(
            partial.atomicity_by_source, aspect=SemanticAspect.ATOMICITY
        )
        consistency, c11 = _select_aspect(
            partial.consistency_by_source, aspect=SemanticAspect.CONSISTENCY
        )
        bounds, c12 = _select_aspect(
            partial.bounds_by_source,
            aspect=SemanticAspect.RESOURCE_BOUNDS,
            kind_on_mismatch=ConflictKind.BOUND_MISMATCH,
        )
        fallback, c13 = _select_aspect(
            partial.fallback_by_source,
            aspect=SemanticAspect.FALLBACK_DEGRADATION,
        )
        applicability, c14 = _select_aspect(
            partial.applicability_by_source, aspect=SemanticAspect.IDENTITY
        )
        for batch in (
            c1,
            c2,
            c3,
            c4,
            c5,
            c6,
            c7,
            c8,
            c9,
            c10,
            c11,
            c12,
            c13,
            c14,
        ):
            conflicts.extend(batch)

        # Cross-rank drift reports (docs/tests vs IDL).
        conflicts.extend(
            _cross_rank_type_conflicts(
                partial.returns_by_source, aspect=SemanticAspect.OUTPUTS
            )
        )
        conflicts.extend(
            _cross_rank_type_conflicts(
                partial.inputs_by_source, aspect=SemanticAspect.INPUTS
            )
        )
        conflicts.extend(
            _cross_rank_type_conflicts(
                partial.sync_by_source, aspect=SemanticAspect.SYNC_ASYNC
            )
        )
        conflicts.extend(
            _cross_rank_type_conflicts(
                partial.errors_by_source, aspect=SemanticAspect.ERRORS
            )
        )

        symbol = partial.symbol or _symbol_from_unit(expectation_units[0])
        interface = partial.interface or _interface_from_unit(
            expectation_units[0]
        )
        # Normalize repository/tree on symbol if missing.
        if symbol.repository_id in {"", "repository:unknown"} and repository_id:
            symbol = SymbolIdentity(
                repository_id=repository_id,
                tree_id=tree_id or symbol.tree_id,
                module_path=symbol.module_path,
                symbol_name=symbol.symbol_name,
                qualified_name=symbol.qualified_name,
                language=symbol.language,
                span_start=symbol.span_start,
                span_end=symbol.span_end,
                blob_cid=symbol.blob_cid,
            )
        summary = next(
            (text for text in partial.summaries if text),
            f"extracted expectation for {interface.interface_name}",
        )
        # Cap conflicts/unsupported with stable ordering.
        conflicts = sorted(
            conflicts,
            key=lambda item: (item.kind.value, item.aspect.value, item.conflict_id),
        )[:MAX_CONFLICTS]
        unsupported = sorted(
            unsupported,
            key=lambda item: (item.aspect.value, item.reason, item.unsupported_id),
        )[:MAX_UNSUPPORTED]
        assumptions = list(partial.assumptions)
        for unit in expectation_units:
            if unit.artifact_class is SourceArtifactClass.DEPRECATED:
                assumptions.append(
                    Assumption(
                        statement=(
                            f"source {unit.artifact_id} is deprecated and "
                            "must not outrank a non-deprecated peer"
                        ),
                        aspect=SemanticAspect.SOURCE_PRECEDENCE,
                        confidence=ConfidenceClass.LOW,
                    )
                )
            if unit.artifact_class is SourceArtifactClass.GENERATED:
                assumptions.append(
                    Assumption(
                        statement=(
                            f"source {unit.artifact_id} is generated; "
                            "prefer reviewed IDL when present"
                        ),
                        aspect=SemanticAspect.SOURCE_PRECEDENCE,
                        confidence=ConfidenceClass.MEDIUM,
                    )
                )

        expected_contract = ExpectedProgramContract(
            symbol=symbol,
            interface=interface,
            policy_revision=policy_revision,
            sources=tuple(sources),
            inputs=tuple(inputs or ()),
            returns=returns,
            errors=tuple(errors or ()),
            sync_async=sync_async,
            side_effects=tuple(effects or ()),
            capabilities=tuple(capabilities or ()),
            authorization=authorization,
            idempotence=idempotence,
            ordering=ordering,
            atomicity=atomicity,
            consistency=consistency,
            resource_bounds=bounds,
            fallback=fallback,
            applicability=applicability,
            assumptions=tuple(assumptions[:64]),
            unsupported=tuple(unsupported),
            conflicts=tuple(conflicts),
            summary=summary,
        )

    observed_contracts: list[ObservedProgramContract] = []
    for unit, clause in zip(partial.observation_units, partial.observed_partials):
        source_ref = unit.to_source_reference(role=ProgramContractRole.OBSERVED)
        symbol = _symbol_from_unit(unit)
        if symbol.repository_id in {"", "repository:unknown"} and repository_id:
            symbol = SymbolIdentity(
                repository_id=repository_id,
                tree_id=tree_id or symbol.tree_id,
                module_path=symbol.module_path,
                symbol_name=symbol.symbol_name,
                qualified_name=symbol.qualified_name,
                language=symbol.language,
                span_start=symbol.span_start,
                span_end=symbol.span_end,
                blob_cid=symbol.blob_cid,
            )
        interface = _interface_from_unit(unit)
        obs_id = (
            unit.repository_observation_id
            or unit.payload.get("repository_observation_id")
            or f"observation:{unit.artifact_id}"
        )
        observed_contracts.append(
            ObservedProgramContract(
                symbol=symbol,
                interface=interface,
                policy_revision=policy_revision,
                repository_observation_id=str(obs_id),
                sources=(source_ref,),
                inputs=tuple(clause.get("inputs") or ()),
                returns=clause.get("returns"),
                errors=tuple(clause.get("errors") or ()),
                sync_async=clause.get("sync_async"),
                side_effects=tuple(clause.get("side_effects") or ()),
                capabilities=tuple(clause.get("capabilities") or ()),
                authorization=clause.get("authorization"),
                idempotence=clause.get("idempotence"),
                ordering=clause.get("ordering"),
                atomicity=clause.get("atomicity"),
                consistency=clause.get("consistency"),
                resource_bounds=clause.get("resource_bounds"),
                fallback=clause.get("fallback"),
                applicability=clause.get("applicability"),
                unsupported=tuple(clause.get("unsupported") or ()),
                summary=str(clause.get("summary") or ""),
                producer_id=unit.producer_id or "contract_extractor",
                producer_version=unit.producer_version
                or str(CONTRACT_EXTRACTOR_VERSION),
            )
        )

    return expected_contract, observed_contracts, conflicts, unsupported


def _ingest_unit(
    unit: ContractSourceUnit,
    partial: _PartialAspects,
) -> None:
    clause, unsupported, _missing = _extract_clause_dict(unit)
    partial.unsupported.extend(unsupported)
    if clause.get("summary"):
        partial.summaries.append(str(clause["summary"]))

    if unit.source_kind is ContractSourceKind.IMPLEMENTATION_OBSERVATION:
        partial.observation_units.append(unit)
        clause_copy = dict(clause)
        clause_copy["unsupported"] = unsupported
        partial.observed_partials.append(clause_copy)
        return

    source_ref = unit.to_source_reference(role=ProgramContractRole.EXPECTED)
    partial.sources.append(source_ref)
    partial.units.append(unit)
    if partial.symbol is None:
        partial.symbol = _symbol_from_unit(unit)
    if partial.interface is None:
        partial.interface = _interface_from_unit(unit)

    if clause.get("inputs") is not None and clause["inputs"] != ():
        partial.inputs_by_source.append((source_ref, clause["inputs"]))
    if clause.get("returns") is not None:
        partial.returns_by_source.append((source_ref, clause["returns"]))
    if clause.get("errors"):
        partial.errors_by_source.append((source_ref, clause["errors"]))
    if clause.get("sync_async") is not None:
        partial.sync_by_source.append((source_ref, clause["sync_async"]))
    if clause.get("side_effects"):
        partial.effects_by_source.append((source_ref, clause["side_effects"]))
    if clause.get("capabilities"):
        partial.capabilities_by_source.append(
            (source_ref, clause["capabilities"])
        )
    if clause.get("authorization") is not None:
        partial.authorization_by_source.append(
            (source_ref, clause["authorization"])
        )
    if clause.get("idempotence") is not None:
        partial.idempotence_by_source.append(
            (source_ref, clause["idempotence"])
        )
    if clause.get("ordering") is not None:
        partial.ordering_by_source.append((source_ref, clause["ordering"]))
    if clause.get("atomicity") is not None:
        partial.atomicity_by_source.append((source_ref, clause["atomicity"]))
    if clause.get("consistency") is not None:
        partial.consistency_by_source.append(
            (source_ref, clause["consistency"])
        )
    if clause.get("resource_bounds") is not None:
        partial.bounds_by_source.append(
            (source_ref, clause["resource_bounds"])
        )
    if clause.get("fallback") is not None:
        partial.fallback_by_source.append((source_ref, clause["fallback"]))
    if clause.get("applicability") is not None:
        partial.applicability_by_source.append(
            (source_ref, clause["applicability"])
        )


def _skip_for_unit(unit: ContractSourceUnit) -> SkippedSource | None:
    if unit.artifact_class is SourceArtifactClass.EXAMPLE:
        return SkippedSource(
            artifact_id=unit.artifact_id,
            reason=SkipReason.EXAMPLE,
            artifact_class=unit.artifact_class,
            source_kind=unit.source_kind,
            extraction_rule=unit.resolved_extraction_rule,
            locator=unit.locator,
            summary="example sources do not define expectations",
            sha256=unit.sha256,
            span_start=unit.span_start,
            span_end=unit.span_end,
            blob_cid=unit.blob_cid,
        )
    if unit.artifact_class is SourceArtifactClass.MOCK:
        return SkippedSource(
            artifact_id=unit.artifact_id,
            reason=SkipReason.MOCK,
            artifact_class=unit.artifact_class,
            source_kind=unit.source_kind,
            extraction_rule=unit.resolved_extraction_rule,
            locator=unit.locator,
            summary="mock sources do not define expectations",
            sha256=unit.sha256,
            span_start=unit.span_start,
            span_end=unit.span_end,
            blob_cid=unit.blob_cid,
        )
    if unit.artifact_class is SourceArtifactClass.FIXTURE:
        # Contract tests may be under fixture paths but still normative when
        # source_kind is CONTRACT_TEST and explicitly classified NORMATIVE.
        # Fixture class always skips for expectation.
        return SkippedSource(
            artifact_id=unit.artifact_id,
            reason=SkipReason.FIXTURE,
            artifact_class=unit.artifact_class,
            source_kind=unit.source_kind,
            extraction_rule=unit.resolved_extraction_rule,
            locator=unit.locator,
            summary="fixture sources do not define expectations",
            sha256=unit.sha256,
            span_start=unit.span_start,
            span_end=unit.span_end,
            blob_cid=unit.blob_cid,
        )
    if unit.source_kind is ContractSourceKind.IMPLEMENTATION_OBSERVATION:
        return None  # handled as observation, not skip
    if not unit.source_kind.may_define_expectation:
        return SkippedSource(
            artifact_id=unit.artifact_id,
            reason=SkipReason.OBSERVATION_ONLY,
            artifact_class=unit.artifact_class,
            source_kind=unit.source_kind,
            extraction_rule=unit.resolved_extraction_rule,
            locator=unit.locator,
            summary="source kind cannot define expectations",
            sha256=unit.sha256,
            span_start=unit.span_start,
            span_end=unit.span_end,
            blob_cid=unit.blob_cid,
        )
    if not unit.payload:
        return SkippedSource(
            artifact_id=unit.artifact_id,
            reason=SkipReason.EMPTY_PAYLOAD,
            artifact_class=unit.artifact_class,
            source_kind=unit.source_kind,
            extraction_rule=unit.resolved_extraction_rule,
            locator=unit.locator,
            summary="empty payload",
            sha256=unit.sha256,
            span_start=unit.span_start,
            span_end=unit.span_end,
            blob_cid=unit.blob_cid,
        )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_contracts(
    units: Sequence[ContractSourceUnit | Mapping[str, Any]],
    *,
    repository_id: str,
    tree_id: str,
    policy_revision: str = DEFAULT_POLICY_REVISION,
) -> ContractExtractionResult:
    """Extract expected and observed contracts from source units.

    This function is deliberately independent from satisfaction checking
    (:mod:`contract_checker`).  It compiles reviewed expectations and
    separate observations under closed source precedence, emits conflicts
    rather than resolving them, and never imports or invokes a checker.
    Domain evidence terms covered: ``vfs/contract-ir@1`` and
    ``vfs/contract-source-precedence@1``.  The synthetic objective validation
    repair discovery key is available via
    :func:`objective_validation_repair_evidence_terms` and never enters
    extraction result identity.

    Parameters
    ----------
    units:
        Compact source units (IDL, schema, signatures, tests, docs, manifests,
        observations).  Mappings are coerced to :class:`ContractSourceUnit`.
    repository_id / tree_id:
        Bundle-level repository binding; applied to symbols when units omit
        them.
    policy_revision:
        Policy identity stamped on every emitted contract.
    """

    if not isinstance(units, Sequence) or isinstance(
        units, (str, bytes, bytearray)
    ):
        raise ContractExtractorError("units must be a sequence")
    if len(units) > MAX_SOURCE_UNITS:
        raise ContractExtractorBoundsError(
            f"units exceeds {MAX_SOURCE_UNITS}"
        )
    repository_id = _text(repository_id, field_name="repository_id")
    tree_id = _text(tree_id, field_name="tree_id")
    policy_revision = _text(policy_revision, field_name="policy_revision")

    normalized: list[ContractSourceUnit] = []
    for index, item in enumerate(units):
        if isinstance(item, ContractSourceUnit):
            unit = item
        elif isinstance(item, Mapping):
            unit = contract_source_unit_from_mapping(item)
        else:
            raise ContractExtractorError(
                f"units[{index}] must be ContractSourceUnit or mapping"
            )
        # Fill repository/tree defaults without rewriting caller-supplied values.
        if not unit.repository_id or not unit.tree_id or not unit.policy_revision:
            unit = ContractSourceUnit(
                artifact_id=unit.artifact_id,
                source_kind=unit.source_kind,
                payload=unit.payload,
                repository_id=unit.repository_id or repository_id,
                tree_id=unit.tree_id or tree_id,
                module_path=unit.module_path,
                symbol_name=unit.symbol_name,
                interface_name=unit.interface_name,
                surface=unit.surface,
                method=unit.method,
                protocol=unit.protocol,
                version=unit.version,
                path_or_uri=unit.path_or_uri,
                locator=unit.locator,
                language=unit.language,
                media_type=unit.media_type,
                blob_cid=unit.blob_cid,
                sha256=unit.sha256,
                span_start=unit.span_start,
                span_end=unit.span_end,
                artifact_class=unit.artifact_class,
                extraction_rule=unit.extraction_rule,
                policy_revision=unit.policy_revision or policy_revision,
                repository_observation_id=unit.repository_observation_id,
                producer_id=unit.producer_id,
                producer_version=unit.producer_version,
                definitions=unit.definitions,
            )
        normalized.append(unit)

    skipped: list[SkippedSource] = []
    partials: dict[str, _PartialAspects] = {}
    global_conflicts: list[ContractConflict] = []
    all_sources: list[SourceReference] = []

    # Detect circular self-expectation attempts up front: an observation unit
    # that claims an expectation-capable artifact class / source kind.
    for unit in normalized:
        if (
            unit.source_kind is ContractSourceKind.IMPLEMENTATION_OBSERVATION
            and unit.artifact_class.may_define_expectation
            and unit.artifact_class
            not in {
                SourceArtifactClass.OBSERVATION,
                SourceArtifactClass.EXAMPLE,
                SourceArtifactClass.MOCK,
                SourceArtifactClass.FIXTURE,
            }
        ):
            # Already rejected in ContractSourceUnit; defensive path.
            raise CircularExpectationError(
                "observation unit cannot define expectations"
            )

    # Stable processing order so equal inputs always yield the same identity.
    normalized.sort(
        key=lambda item: (
            item.subject_key(),
            item.source_kind.rank,
            item.artifact_id,
            item.unit_id,
        )
    )

    for unit in normalized:
        # Attempting to treat observation kind as expected role is fail-closed.
        if unit.source_kind is ContractSourceKind.IMPLEMENTATION_OBSERVATION:
            try:
                unit.to_source_reference(role=ProgramContractRole.EXPECTED)
            except CircularExpectationError:
                # Record as conflict and continue as observation.
                obs_ref = unit.to_source_reference(
                    role=ProgramContractRole.OBSERVED
                )
                global_conflicts.append(
                    ContractConflict(
                        kind=ConflictKind.SELF_EXPECTATION,
                        aspect=SemanticAspect.SOURCE_PRECEDENCE,
                        left_source_id=obs_ref.source_id,
                        right_source_id=obs_ref.source_id,
                        summary=(
                            "implementation observation cannot define its "
                            "own expectation"
                        ),
                        left_summary=unit.artifact_id,
                        right_summary="expectation_role_rejected",
                        resolved=False,
                    )
                )

        skip = _skip_for_unit(unit)
        if skip is not None:
            skipped.append(skip)
            continue

        key = unit.subject_key()
        partial = partials.setdefault(key, _PartialAspects())
        partial.policy_revision = policy_revision
        _ingest_unit(unit, partial)

    # Shadow deprecated/generated when a stronger normative peer exists for
    # the same subject.
    for key, partial in list(partials.items()):
        normative_present = any(
            unit.artifact_class is SourceArtifactClass.NORMATIVE
            and unit.source_kind.may_define_expectation
            for unit in partial.units
        )
        if not normative_present:
            continue
        retained_units: list[ContractSourceUnit] = []
        for unit in partial.units:
            if unit.artifact_class is SourceArtifactClass.DEPRECATED:
                skipped.append(
                    SkippedSource(
                        artifact_id=unit.artifact_id,
                        reason=SkipReason.DEPRECATED_SHADOWED,
                        artifact_class=unit.artifact_class,
                        source_kind=unit.source_kind,
                        extraction_rule=unit.resolved_extraction_rule,
                        locator=unit.locator,
                        summary=(
                            "deprecated variant shadowed by normative peer"
                        ),
                        sha256=unit.sha256,
                        span_start=unit.span_start,
                        span_end=unit.span_end,
                        blob_cid=unit.blob_cid,
                    )
                )
                # Remove this unit's contributions by rebuilding later.
                continue
            if (
                unit.artifact_class is SourceArtifactClass.GENERATED
                and any(
                    u.source_kind is ContractSourceKind.REVIEWED_INTERFACE
                    and u.artifact_class is SourceArtifactClass.NORMATIVE
                    for u in partial.units
                )
            ):
                skipped.append(
                    SkippedSource(
                        artifact_id=unit.artifact_id,
                        reason=SkipReason.GENERATED_SHADOWED,
                        artifact_class=unit.artifact_class,
                        source_kind=unit.source_kind,
                        extraction_rule=unit.resolved_extraction_rule,
                        locator=unit.locator,
                        summary=(
                            "generated copy shadowed by reviewed IDL peer"
                        ),
                        sha256=unit.sha256,
                        span_start=unit.span_start,
                        span_end=unit.span_end,
                        blob_cid=unit.blob_cid,
                    )
                )
                continue
            retained_units.append(unit)
        if len(retained_units) != len(partial.units):
            # Rebuild partial from retained units + observations.
            rebuilt = _PartialAspects()
            rebuilt.policy_revision = policy_revision
            rebuilt.observation_units = list(partial.observation_units)
            rebuilt.observed_partials = list(partial.observed_partials)
            for unit in retained_units:
                _ingest_unit(unit, rebuilt)
            partials[key] = rebuilt

    expected: list[ExpectedProgramContract] = []
    observed: list[ObservedProgramContract] = []
    all_unsupported: list[UnsupportedSemantics] = []
    all_conflicts: list[ContractConflict] = list(global_conflicts)

    for key in sorted(partials):
        partial = partials[key]
        exp, obs_list, conflicts, unsupported = _merge_partial(
            partial,
            repository_id=repository_id,
            tree_id=tree_id,
            policy_revision=policy_revision,
        )
        if exp is not None:
            expected.append(exp)
            all_sources.extend(exp.sources)
            all_conflicts.extend(exp.conflicts)
        for obs in obs_list:
            observed.append(obs)
            all_sources.extend(obs.sources)
        all_unsupported.extend(unsupported)
        all_conflicts.extend(conflicts)

    # Deduplicate conflicts by id with stable sort.
    seen_c: set[str] = set()
    unique_conflicts: list[ContractConflict] = []
    for item in sorted(
        all_conflicts,
        key=lambda c: (c.kind.value, c.aspect.value, c.conflict_id),
    ):
        if item.conflict_id in seen_c:
            continue
        seen_c.add(item.conflict_id)
        unique_conflicts.append(item)
        if len(unique_conflicts) >= MAX_CONFLICTS:
            break

    seen_u: set[str] = set()
    unique_unsupported: list[UnsupportedSemantics] = []
    for item in sorted(
        all_unsupported,
        key=lambda u: (u.aspect.value, u.reason, u.unsupported_id),
    ):
        if item.unsupported_id in seen_u:
            continue
        seen_u.add(item.unsupported_id)
        unique_unsupported.append(item)
        if len(unique_unsupported) >= MAX_UNSUPPORTED:
            break

    skipped.sort(
        key=lambda item: (item.reason.value, item.artifact_id, item.sha256)
    )

    # Stable ordering.
    expected.sort(
        key=lambda item: (
            item.interface.interface_name,
            item.symbol.symbol_name,
            item.expected_contract_id,
        )
    )
    observed.sort(
        key=lambda item: (
            item.interface.interface_name,
            item.repository_observation_id,
            item.observed_contract_id
            if hasattr(item, "observed_contract_id")
            else item.content_id,
        )
    )
    all_sources_sorted = sorted(
        all_sources,
        key=lambda item: (item.precedence_rank, item.artifact_id, item.source_id),
    )
    # Deduplicate sources by source_id.
    seen_s: set[str] = set()
    unique_sources: list[SourceReference] = []
    for item in all_sources_sorted:
        if item.source_id in seen_s:
            continue
        seen_s.add(item.source_id)
        unique_sources.append(item)
    all_sources = unique_sources

    summary = (
        f"contract extraction v{CONTRACT_EXTRACTOR_VERSION}: "
        f"{len(expected)} expected, {len(observed)} observed, "
        f"{len(unique_conflicts)} conflicts, {len(skipped)} skipped"
    )
    return ContractExtractionResult(
        repository_id=repository_id,
        tree_id=tree_id,
        policy_revision=policy_revision,
        expected=tuple(expected),
        observed=tuple(observed),
        conflicts=tuple(unique_conflicts),
        unsupported=tuple(unique_unsupported),
        skipped=tuple(skipped),
        sources=tuple(all_sources),
        summary=summary,
    )


def contract_source_unit_from_mapping(
    payload: Mapping[str, Any],
) -> ContractSourceUnit:
    """Coerce a mapping into a validated :class:`ContractSourceUnit`."""

    if not isinstance(payload, Mapping):
        raise ContractExtractorError("payload must be a mapping")
    source_kind = payload.get("source_kind") or payload.get("kind_source")
    if source_kind is None:
        # Infer from content.
        content_kind = _infer_content_kind(payload.get("payload") or payload)
        if content_kind is ContentKind.OBSERVATION:
            source_kind = ContractSourceKind.IMPLEMENTATION_OBSERVATION
        elif content_kind in {
            ContentKind.MCP_TOOL,
            ContentKind.MCP_IDL,
            ContentKind.MCP_PLUS_PLUS,
            ContentKind.JSON_SCHEMA,
            ContentKind.TYPED_INTERFACE,
        }:
            source_kind = ContractSourceKind.REVIEWED_INTERFACE
        elif content_kind is ContentKind.PUBLIC_SIGNATURE:
            source_kind = ContractSourceKind.PUBLIC_SIGNATURE
        elif content_kind is ContentKind.CONTRACT_TEST:
            source_kind = ContractSourceKind.CONTRACT_TEST
        elif content_kind is ContentKind.NORMATIVE_DOC:
            source_kind = ContractSourceKind.NORMATIVE_DOCUMENTATION
        elif content_kind in {
            ContentKind.COMPAT_MANIFEST,
            ContentKind.GENERATED_SDK,
        }:
            source_kind = ContractSourceKind.COMPATIBILITY_MANIFEST
        else:
            source_kind = ContractSourceKind.NORMATIVE_DOCUMENTATION

    body = payload.get("payload")
    if body is None:
        # Allow flattened units where the mapping *is* the payload plus meta.
        reserved = {
            "artifact_id",
            "source_kind",
            "kind_source",
            "repository_id",
            "tree_id",
            "module_path",
            "symbol_name",
            "interface_name",
            "surface",
            "method",
            "protocol",
            "version",
            "path_or_uri",
            "locator",
            "language",
            "media_type",
            "blob_cid",
            "sha256",
            "span_start",
            "span_end",
            "artifact_class",
            "extraction_rule",
            "policy_revision",
            "repository_observation_id",
            "producer_id",
            "producer_version",
            "definitions",
            "schema",
            "unit_id",
            "content_id",
        }
        body = {
            key: value
            for key, value in payload.items()
            if key not in reserved
        }
        if not body and "payload" not in payload:
            body = dict(payload)

    locator = payload.get("locator") or payload.get("path") or ""
    artifact_class = payload.get("artifact_class")
    if artifact_class is None:
        artifact_class = classify_artifact_path(str(locator))

    return ContractSourceUnit(
        artifact_id=str(
            payload.get("artifact_id")
            or payload.get("id")
            or f"artifact:{locator or 'unit'}"
        ),
        source_kind=source_kind,
        payload=_mapping(body, field_name="payload"),
        repository_id=str(payload.get("repository_id") or ""),
        tree_id=str(payload.get("tree_id") or ""),
        module_path=str(payload.get("module_path") or ""),
        symbol_name=str(payload.get("symbol_name") or ""),
        interface_name=str(payload.get("interface_name") or ""),
        surface=str(payload.get("surface") or ""),
        method=str(payload.get("method") or ""),
        protocol=str(payload.get("protocol") or ""),
        version=str(payload.get("version") or ""),
        path_or_uri=str(payload.get("path_or_uri") or ""),
        locator=str(locator),
        language=str(payload.get("language") or ""),
        media_type=str(payload.get("media_type") or ""),
        blob_cid=str(payload.get("blob_cid") or ""),
        sha256=str(payload.get("sha256") or ""),
        span_start=payload.get("span_start"),
        span_end=payload.get("span_end"),
        artifact_class=artifact_class,
        extraction_rule=payload.get("extraction_rule"),
        policy_revision=str(payload.get("policy_revision") or ""),
        repository_observation_id=str(
            payload.get("repository_observation_id") or ""
        ),
        producer_id=str(payload.get("producer_id") or ""),
        producer_version=str(payload.get("producer_version") or ""),
        definitions=dict(payload.get("definitions") or {}),
    )


def make_mcp_tool_unit(
    *,
    artifact_id: str,
    name: str,
    input_schema: Mapping[str, Any] | None = None,
    output_schema: Mapping[str, Any] | None = None,
    errors: Sequence[Any] | None = None,
    async_mode: bool | str | None = None,
    capabilities: Sequence[Any] | None = None,
    description: str = "",
    source_kind: ContractSourceKind = ContractSourceKind.REVIEWED_INTERFACE,
    artifact_class: SourceArtifactClass = SourceArtifactClass.NORMATIVE,
    repository_id: str = "",
    tree_id: str = "",
    surface: str = "mcp++",
    version: str = "1.0",
    locator: str = "",
    span_start: int | None = None,
    span_end: int | None = None,
    blob_cid: str = "",
    **extra: Any,
) -> ContractSourceUnit:
    """Convenience constructor for MCP / MCP++ tool IDL units."""

    payload: dict[str, Any] = {
        "kind": "mcp_tool",
        "name": name,
        "description": description,
    }
    if input_schema is not None:
        payload["inputSchema"] = dict(input_schema)
    if output_schema is not None:
        payload["outputSchema"] = dict(output_schema)
    if errors is not None:
        payload["errors"] = list(errors)
    if async_mode is not None:
        payload["async"] = async_mode
    if capabilities is not None:
        payload["capabilities"] = list(capabilities)
    payload.update(extra)
    return ContractSourceUnit(
        artifact_id=artifact_id,
        source_kind=source_kind,
        payload=payload,
        repository_id=repository_id,
        tree_id=tree_id,
        symbol_name=name.split(".")[-1],
        interface_name=name,
        surface=surface,
        method=name.split(".")[-1],
        protocol="mcp",
        version=version,
        locator=locator or f"tools/list#{name}",
        path_or_uri=f"mcp://{name}",
        artifact_class=artifact_class,
        extraction_rule=ExtractionRule.MCP_PLUS_PLUS_IDL_V1
        if surface in {"mcp++", "mcplusplus", "mcp_plus_plus"}
        else ExtractionRule.MCP_IDL_V1,
        span_start=span_start,
        span_end=span_end,
        blob_cid=blob_cid,
        language="json",
        media_type="application/schema+json",
    )


def make_signature_unit(
    *,
    artifact_id: str,
    symbol_name: str,
    parameters: Sequence[Any] | None = None,
    returns: Any = None,
    raises: Sequence[Any] | None = None,
    async_mode: bool | None = None,
    module_path: str = "",
    repository_id: str = "",
    tree_id: str = "",
    artifact_class: SourceArtifactClass = SourceArtifactClass.NORMATIVE,
    source_kind: ContractSourceKind = ContractSourceKind.PUBLIC_SIGNATURE,
    locator: str = "",
    span_start: int | None = None,
    span_end: int | None = None,
    blob_cid: str = "",
    language: str = "python",
    **extra: Any,
) -> ContractSourceUnit:
    """Convenience constructor for public signature / typed interface units."""

    payload: dict[str, Any] = {
        "kind": "public_signature",
        "name": symbol_name,
    }
    if parameters is not None:
        payload["parameters"] = list(parameters)
    if returns is not None:
        payload["returns"] = returns
    if raises is not None:
        payload["raises"] = list(raises)
    if async_mode is not None:
        payload["async"] = async_mode
    payload.update(extra)
    return ContractSourceUnit(
        artifact_id=artifact_id,
        source_kind=source_kind,
        payload=payload,
        repository_id=repository_id,
        tree_id=tree_id,
        module_path=module_path,
        symbol_name=symbol_name,
        interface_name=symbol_name,
        surface="python",
        method=symbol_name,
        locator=locator or f"{module_path}:{symbol_name}",
        artifact_class=artifact_class,
        extraction_rule=ExtractionRule.PUBLIC_SIGNATURE_V1,
        span_start=span_start,
        span_end=span_end,
        blob_cid=blob_cid,
        language=language,
    )


def make_observation_unit(
    *,
    artifact_id: str,
    symbol_name: str,
    repository_observation_id: str,
    observed: Mapping[str, Any],
    repository_id: str = "",
    tree_id: str = "",
    module_path: str = "",
    locator: str = "",
    producer_id: str = "static-observer",
    producer_version: str = "1",
    span_start: int | None = None,
    span_end: int | None = None,
    blob_cid: str = "",
) -> ContractSourceUnit:
    """Convenience constructor for implementation observation units."""

    return ContractSourceUnit(
        artifact_id=artifact_id,
        source_kind=ContractSourceKind.IMPLEMENTATION_OBSERVATION,
        payload={
            "kind": "observation",
            "name": symbol_name,
            "observed": dict(observed),
            "repository_observation_id": repository_observation_id,
        },
        repository_id=repository_id,
        tree_id=tree_id,
        module_path=module_path,
        symbol_name=symbol_name,
        interface_name=symbol_name,
        surface="python",
        method=symbol_name,
        locator=locator or f"observation:{symbol_name}",
        artifact_class=SourceArtifactClass.OBSERVATION,
        extraction_rule=ExtractionRule.IMPLEMENTATION_OBS_V1,
        repository_observation_id=repository_observation_id,
        producer_id=producer_id,
        producer_version=producer_version,
        span_start=span_start,
        span_end=span_end,
        blob_cid=blob_cid,
    )


def expectation_source_kinds() -> tuple[ContractSourceKind, ...]:
    """Return the closed expectation precedence tuple."""

    return SOURCE_PRECEDENCE


def all_extraction_rules() -> tuple[ExtractionRule, ...]:
    return tuple(ExtractionRule)


def all_artifact_classes() -> tuple[SourceArtifactClass, ...]:
    return tuple(SourceArtifactClass)


def covered_evidence_terms() -> tuple[str, ...]:
    """Return the domain objective evidence terms this extractor proves.

    Mirrors :func:`program_contract_evidence_terms` so both IR and extractor
    surfaces name ``vfs/contract-ir@1`` and
    ``vfs/contract-source-precedence@1`` for discovery scans.

    The synthetic ``objective validation repair`` term is intentionally
    omitted here so extraction envelope ``evidence`` stays domain-only; use
    :func:`objective_validation_repair_evidence_terms` (or
    :func:`all_covered_evidence_terms`) for the validation gate.
    """

    return program_contract_evidence_terms()


def all_covered_evidence_terms() -> tuple[str, ...]:
    """Return domain VFS-G050 terms plus the objective validation repair gate.

    Domain IR/precedence terms come first; the synthetic objective validation
    repair discovery key is appended last.  Ranking and identity payloads
    continue to use :func:`covered_evidence_terms` only.
    """

    return all_program_contract_evidence_terms()


def reject_observation_as_expectation_source(
    unit: ContractSourceUnit,
) -> None:
    """Fail closed if a caller promotes an observation into an expectation."""

    if unit.source_kind is ContractSourceKind.IMPLEMENTATION_OBSERVATION:
        raise CircularExpectationError(
            "implementation observations cannot define expectations"
        )
    if not unit.source_kind.may_define_expectation:
        raise ForgedSourceError(
            f"{unit.source_kind.value} cannot define expectations"
        )
    unit.to_source_reference(role=ProgramContractRole.EXPECTED)


__all__ = [
    "CONTRACT_EXTRACTOR_VERSION",
    "SCHEMA_VERSION",
    "DEFAULT_POLICY_REVISION",
    "MAX_SOURCE_UNITS",
    "CONTRACT_EXTRACTOR_EVIDENCE",
    "CONTRACT_IR_EVIDENCE",
    "CONTRACT_SOURCE_PRECEDENCE_EVIDENCE",
    "OBJECTIVE_VALIDATION_REPAIR_EVIDENCE",
    "OBJECTIVE_GOAL_ID",
    "ContractExtractorError",
    "ContractExtractorBoundsError",
    "MissingReferenceError",
    "UnsupportedExtractionError",
    "SourceArtifactClass",
    "ExtractionRule",
    "SkipReason",
    "ContentKind",
    "ContractSourceUnit",
    "SkippedSource",
    "ContractExtractionResult",
    "classify_artifact_path",
    "confidence_for",
    "extraction_rule_for",
    "type_shape_from_name",
    "type_shape_from_json_schema",
    "extract_contracts",
    "contract_source_unit_from_mapping",
    "make_mcp_tool_unit",
    "make_signature_unit",
    "make_observation_unit",
    "expectation_source_kinds",
    "all_extraction_rules",
    "all_artifact_classes",
    "covered_evidence_terms",
    "all_covered_evidence_terms",
    "objective_validation_repair_evidence_terms",
    "all_program_contract_evidence_terms",
    "reject_observation_as_expectation_source",
    "source_precedence_rank",
    "may_define_expectation",
    "program_contract_content_identity",
    "program_contract_evidence_terms",
    "content_identity",
]
