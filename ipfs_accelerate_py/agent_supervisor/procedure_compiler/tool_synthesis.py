"""Bounded deterministic tool synthesis from a reviewed transformation DSL.

Repeated pure transformations may become interpreted tools.  This module owns
the closed grammar, reviewed template library, compiler, and translation
validator.  It never executes arbitrary Python or shell, never grants
authority, and never promotes optimized Python until exact differential
validation and a tool certificate are both present.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, Final, NoReturn

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    canonical_json_bytes,
    content_identity,
)
from .contracts import (
    ARTIFACT_TYPES_BY_SCHEMA,
    MAX_ITEMS,
    MAX_MAPPING_ITEMS,
    MAX_NESTING,
    MAX_SCOPE_PATHS,
    MAX_TEXT_BYTES,
    PROCEDURE_CONTRACT_VERSION,
    ArtifactBindings,
    ArtifactState,
    EffectClass,
    ProcedureBoundsError,
    ProcedureContractError,
    ProcedureSafetyError,
    _bounded,
    _decode_fields,
    _enum,
    _freeze,
    _identifier,
    _nested,
    _nonnegative_int,
    _positive_int,
    _relative_path,
    _schema_name,
    _strings,
    _text,
    _unsafe_key,
    _verify_identity,
)


DSL_REVISION: Final[str] = "TransformationDsl@1"
DETERMINISTIC_TOOL_DSL_REVISION: Final[str] = "DeterministicToolDsl@1"
COMPILER_REVISION: Final[str] = "GeneratedToolCompiler@1"
VALIDATOR_REVISION: Final[str] = "TranslationValidator@1"
GRAMMAR_REVISION: Final[str] = "transformation-grammar@1"
TEMPLATE_LIBRARY_REVISION: Final[str] = "transformation-template-library@1"
MAX_TOOL_STEPS: Final[int] = 16
MAX_TOOL_ITEMS: Final[int] = MAX_ITEMS
MAX_TOOL_OUTPUT_BYTES: Final[int] = 65_536
MAX_PATH_DEPTH: Final[int] = 12
MAX_TEMPLATE_PARAMS: Final[int] = MAX_MAPPING_ITEMS
MAX_FIXTURES: Final[int] = 32
MAX_TRANSLATION_BYTES: Final[int] = MAX_TEXT_BYTES
ALLOWED_EFFECT_CLASSES: Final[frozenset[EffectClass]] = frozenset(
    {EffectClass.OBSERVE, EffectClass.VALIDATION}
)
APPROVED_REPAIR_TEMPLATE_IDS: Final[tuple[str, ...]] = (
    "template.import-purity",
    "template.path-scope",
    "template.bounded-rename",
)

_GENERIC_ENVELOPE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "bindings",
        "artifact_version",
        "state",
        "subject_cid",
        "reference_cids",
        "labels",
        "facts",
        "created_at_ms",
    }
)
_FORBIDDEN_OPERATION_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "arbitrary_python",
        "arbitrary_shell",
        "eval",
        "exec",
        "import",
        "network",
        "os_system",
        "popen",
        "shell",
        "subprocess",
        "system",
        "__import__",
    }
)
_FORBIDDEN_AST_TYPES: Final[frozenset[str]] = frozenset(
    {
        "AsyncFor",
        "AsyncFunctionDef",
        "AsyncWith",
        "Await",
        "ClassDef",
        "Delete",
        "ExceptHandler",
        "Global",
        "Import",
        "ImportFrom",
        "Lambda",
        "Match",
        "NamedExpr",
        "Nonlocal",
        "Raise",
        "Try",
        "TryStar",
        "With",
        "Yield",
        "YieldFrom",
    }
)
_ALLOWED_AST_TYPES: Final[frozenset[str]] = frozenset(
    {
        "Assign",
        "Call",
        "Constant",
        "Dict",
        "Expr",
        "FunctionDef",
        "List",
        "Load",
        "Module",
        "Name",
        "Pass",
        "Return",
        "Store",
        "Tuple",
        "arg",
        "arguments",
    }
)
_ALLOWED_HELPERS: Final[frozenset[str]] = frozenset(
    {
        "op_const",
        "op_filter_equals",
        "op_filter_in",
        "op_hash_canonical",
        "op_identity",
        "op_join_identifiers",
        "op_limit",
        "op_map_items",
        "op_normalize_path_field",
        "op_prefix_normalize",
        "op_prefix_path_field",
        "op_project",
        "op_relative_path_field",
        "op_rename_fields",
        "op_select_fields",
        "op_select_rename",
        "op_sort_by",
        "op_sort_limit",
    }
)


class ToolSynthesisError(ProcedureContractError):
    """A generated-tool declaration, translation, or promotion is unsafe."""

    def __init__(self, message: str, reason_code: "ToolReason" | None = None) -> None:
        super().__init__(message)
        self.reason_code = reason_code or ToolReason.GRAMMAR_VIOLATION


class ToolGrammarError(ToolSynthesisError):
    """The transformation program is outside the reviewed grammar."""


class ToolSafetyError(ToolSynthesisError):
    """The candidate would escape scope, effects, or the closed operation set."""


class ToolTranslationError(ToolSynthesisError):
    """DSL and optimized translations are not exactly equivalent."""


class ToolPromotionError(ToolSynthesisError):
    """Optimized Python was not eligible for promotion."""


class ToolReason(str, Enum):
    ACCEPTED = "accepted"
    UNKNOWN_OPERATION = "unknown-operation"
    UNKNOWN_TEMPLATE = "unknown-template"
    GRAMMAR_VIOLATION = "grammar-violation"
    SCHEMA_MISSING = "schema-missing"
    SCHEMA_MISMATCH = "schema-mismatch"
    EFFECT_FORBIDDEN = "effect-forbidden"
    PATH_ESCAPE = "path-escape"
    RESOURCE_EXCEEDED = "resource-exceeded"
    ARBITRARY_CODE = "arbitrary-code"
    ARBITRARY_SHELL = "arbitrary-shell"
    UNBOUNDED = "unbounded"
    TRANSLATION_MISMATCH = "translation-mismatch"
    MISSING_CERTIFICATE = "missing-certificate"
    CERTIFICATE_REJECTED = "certificate-rejected"
    PROMOTION_FORBIDDEN = "promotion-forbidden"
    CANDIDATE_TIER_REQUIRED = "candidate-tier-required"
    ADVERSARIAL_FAILURE = "adversarial-failure"
    MISSING_TESTS = "missing-tests"
    BINDING_MISMATCH = "binding-mismatch"
    INJECTION_REJECTED = "injection-rejected"
    AUTHORITY_REJECTED = "authority-rejected"
    MISSING_SCOPE = "missing-scope"


class TransformationOp(str, Enum):
    IDENTITY = "identity"
    SELECT_FIELDS = "select-fields"
    RENAME_FIELDS = "rename-fields"
    FILTER_EQUALS = "filter-equals"
    FILTER_IN = "filter-in"
    MAP_ITEMS = "map-items"
    SORT_BY = "sort-by"
    LIMIT = "limit"
    PROJECT = "project"
    NORMALIZE_PATH = "normalize-path"
    PREFIX_PATH = "prefix-path"
    RELATIVE_PATH = "relative-path"
    JOIN_IDENTIFIERS = "join-identifiers"
    HASH_CANONICAL = "hash-canonical"
    CONST = "const"
    SELECT_RENAME = "select-rename"
    SORT_LIMIT = "sort-limit"
    PREFIX_NORMALIZE = "prefix-normalize"


class ToolAction(str, Enum):
    CANDIDATE = "candidate"
    CERTIFY = "certify"
    PROMOTE_OPTIMIZED = "promote-optimized"
    INVOKE = "invoke"
    REFUSE = "refuse"


class TranslationStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class FixtureKind(str, Enum):
    HAPPY = "happy"
    ADVERSARIAL = "adversarial"
    BOUNDARY = "boundary"


FORBIDDEN_TRANSFORMATION_OPS: Final[frozenset[str]] = frozenset(
    {
        "ARBITRARY_PYTHON",
        "ARBITRARY_SHELL",
        "EVAL",
        "EXEC",
        "IMPORT",
        "NETWORK_REQUEST",
        "SUBPROCESS",
        "SYSTEM",
        "UNBOUNDED_LOOP",
    }
)
FUSED_OPERATIONS: Final[frozenset[TransformationOp]] = frozenset(
    {
        TransformationOp.SELECT_RENAME,
        TransformationOp.SORT_LIMIT,
        TransformationOp.PREFIX_NORMALIZE,
    }
)


def _bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise ToolSynthesisError(f"{field_name} must be a boolean", ToolReason.GRAMMAR_VIOLATION)
    return value


def _bindings(value: Any) -> ArtifactBindings:
    return _nested(value, ArtifactBindings, "bindings")


def _refuse(reason: ToolReason, message: str) -> NoReturn:
    if reason in {ToolReason.ARBITRARY_CODE, ToolReason.ARBITRARY_SHELL, ToolReason.PATH_ESCAPE}:
        raise ToolSafetyError(message, reason)
    if reason in {ToolReason.TRANSLATION_MISMATCH, ToolReason.ADVERSARIAL_FAILURE}:
        raise ToolTranslationError(message, reason)
    if reason in {
        ToolReason.PROMOTION_FORBIDDEN,
        ToolReason.MISSING_CERTIFICATE,
        ToolReason.CERTIFICATE_REJECTED,
        ToolReason.CANDIDATE_TIER_REQUIRED,
    }:
        raise ToolPromotionError(message, reason)
    if reason in {
        ToolReason.UNKNOWN_OPERATION,
        ToolReason.UNKNOWN_TEMPLATE,
        ToolReason.GRAMMAR_VIOLATION,
        ToolReason.UNBOUNDED,
    }:
        raise ToolGrammarError(message, reason)
    raise ToolSynthesisError(message, reason)


def _normalized_marker(value: str) -> str:
    return value.lower().replace("-", "_")


def _forbidden_operation_token(value: str) -> ToolReason | None:
    normalized = _normalized_marker(value)
    if "shell" in normalized or normalized in {"system", "popen", "subprocess"}:
        return ToolReason.ARBITRARY_SHELL
    if any(marker in normalized for marker in _FORBIDDEN_OPERATION_MARKERS):
        return ToolReason.ARBITRARY_CODE
    if value in FORBIDDEN_TRANSFORMATION_OPS or normalized in {
        item.lower() for item in FORBIDDEN_TRANSFORMATION_OPS
    }:
        if "SHELL" in value or "shell" in normalized:
            return ToolReason.ARBITRARY_SHELL
        return ToolReason.ARBITRARY_CODE
    return None


def _scan_injection(value: Any, field_name: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                _refuse(ToolReason.INJECTION_REJECTED, f"{field_name} keys must be strings")
            if _unsafe_key(raw_key):
                _refuse(
                    ToolReason.INJECTION_REJECTED,
                    f"{field_name} contains a forbidden secret or executable field",
                )
            reason = _forbidden_operation_token(raw_key)
            if reason is not None:
                _refuse(reason, f"{field_name} contains a forbidden operation field")
            _scan_injection(item, field_name)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        for item in value:
            _scan_injection(item, field_name)
        return
    if isinstance(value, str):
        reason = _forbidden_operation_token(value)
        if reason is not None:
            _refuse(reason, f"{field_name} contains a forbidden operation token")


def _path_is_within(path: str, prefixes: Sequence[str]) -> bool:
    candidate = PurePosixPath(path)
    if not prefixes:
        return False
    for prefix in prefixes:
        if prefix == ".":
            return True
        root = PurePosixPath(prefix)
        if candidate == root or root in candidate.parents:
            return True
    return False


def _path_depth(path: str) -> int:
    if path == ".":
        return 0
    return len(PurePosixPath(path).parts)


def _literal(value: Any, field_name: str) -> Any:
    frozen = _freeze(value, field_name)
    _scan_injection(frozen, field_name)
    return frozen


def _fixture_payload(value: Any, field_name: str, *, depth: int = 0) -> Any:
    """Bound adversarial fixture bodies without executing or dropping the attack."""

    if depth > MAX_NESTING:
        _refuse(ToolReason.UNBOUNDED, f"{field_name} exceeds its nesting bound")
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        return _nonnegative_int(value, field_name)
    if type(value) is str:
        if "\x00" in value:
            raise ProcedureSafetyError(f"{field_name} contains a NUL byte")
        if len(value.encode("utf-8")) > MAX_TEXT_BYTES:
            _refuse(ToolReason.UNBOUNDED, f"{field_name} exceeds its byte bound")
        return value
    if isinstance(value, float):
        raise ToolGrammarError(f"{field_name} cannot contain floating point values", ToolReason.GRAMMAR_VIOLATION)
    if isinstance(value, Mapping):
        if len(value) > MAX_MAPPING_ITEMS:
            _refuse(ToolReason.UNBOUNDED, f"{field_name} exceeds its mapping bound")
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise ToolGrammarError(f"{field_name} keys must be strings", ToolReason.GRAMMAR_VIOLATION)
            result[raw_key] = _fixture_payload(item, field_name, depth=depth + 1)
        return MappingProxyType(result)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        if len(value) > MAX_ITEMS + 1:
            _refuse(ToolReason.UNBOUNDED, f"{field_name} exceeds the adversarial item bound")
        return tuple(_fixture_payload(item, field_name, depth=depth + 1) for item in value)
    raise ToolSafetyError(
        f"{field_name} contains unsupported value type {type(value).__name__}",
        ToolReason.ARBITRARY_CODE,
    )


def _string_tuple(value: Any, field_name: str, *, required: bool = True) -> tuple[str, ...]:
    return _strings(value, field_name, identifiers=True, required=required)


def _mapping_of_identifiers(value: Any, field_name: str) -> Mapping[str, str]:
    frozen = _freeze(value if value is not None else {}, field_name)
    if not isinstance(frozen, Mapping):
        raise ToolGrammarError(f"{field_name} must be a mapping", ToolReason.GRAMMAR_VIOLATION)
    result: dict[str, str] = {}
    for raw_key, item in frozen.items():
        key = _identifier(raw_key, field_name)
        result[key] = _identifier(item, f"{field_name}.{key}")
    return MappingProxyType(result)


def _unwrap_generic_envelope(payload: Mapping[str, Any], schema: str) -> Mapping[str, Any]:
    body = dict(payload)
    keys = set(body).difference({"schema", "contract_version", "content_id", "cid"})
    if keys and keys <= _GENERIC_ENVELOPE_FIELDS and "facts" in body:
        facts = body.get("facts")
        if not isinstance(facts, Mapping):
            raise ToolSynthesisError("generic tool facts must be a mapping")
        merged = {
            "schema": schema,
            "contract_version": body.get("contract_version", PROCEDURE_CONTRACT_VERSION),
            "bindings": body.get("bindings"),
            "state": body.get("state", ArtifactState.CANDIDATE.value),
            **dict(facts),
        }
        if "tool_id" not in merged and body.get("subject_cid"):
            merged["tool_id"] = body.get("subject_cid")
        return merged
    return body


@dataclass(frozen=True)
class ToolResourceEnvelope:
    """Integer resource ceiling for one generated tool interpretation."""

    max_steps: int = MAX_TOOL_STEPS
    max_items: int = MAX_TOOL_ITEMS
    max_output_bytes: int = MAX_TOOL_OUTPUT_BYTES
    max_path_depth: int = MAX_PATH_DEPTH
    max_duration_ms: int = 1_000

    def __post_init__(self) -> None:
        try:
            object.__setattr__(
                self,
                "max_steps",
                _positive_int(self.max_steps, "max_steps", maximum=MAX_TOOL_STEPS),
            )
            object.__setattr__(
                self,
                "max_items",
                _positive_int(self.max_items, "max_items", maximum=MAX_TOOL_ITEMS),
            )
            object.__setattr__(
                self,
                "max_output_bytes",
                _positive_int(
                    self.max_output_bytes, "max_output_bytes", maximum=MAX_TOOL_OUTPUT_BYTES
                ),
            )
            object.__setattr__(
                self,
                "max_path_depth",
                _positive_int(self.max_path_depth, "max_path_depth", maximum=MAX_PATH_DEPTH),
            )
            object.__setattr__(
                self,
                "max_duration_ms",
                _positive_int(self.max_duration_ms, "max_duration_ms", maximum=60_000),
            )
        except ProcedureBoundsError as exc:
            raise ToolGrammarError(str(exc), ToolReason.UNBOUNDED) from exc

    def to_record(self) -> dict[str, int]:
        return {
            "max_steps": self.max_steps,
            "max_items": self.max_items,
            "max_output_bytes": self.max_output_bytes,
            "max_path_depth": self.max_path_depth,
            "max_duration_ms": self.max_duration_ms,
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any] | ToolResourceEnvelope | None) -> ToolResourceEnvelope:
        if isinstance(payload, ToolResourceEnvelope):
            return payload
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ToolGrammarError("resource envelope must be a mapping", ToolReason.GRAMMAR_VIOLATION)
        return cls(
            max_steps=payload.get("max_steps", MAX_TOOL_STEPS),
            max_items=payload.get("max_items", MAX_TOOL_ITEMS),
            max_output_bytes=payload.get("max_output_bytes", MAX_TOOL_OUTPUT_BYTES),
            max_path_depth=payload.get("max_path_depth", MAX_PATH_DEPTH),
            max_duration_ms=payload.get("max_duration_ms", 1_000),
        )


@dataclass(frozen=True)
class TransformationStep:
    """One closed, pure transformation.  Parameters are data, never code."""

    operation: TransformationOp
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.operation, str):
            forbidden = _forbidden_operation_token(self.operation)
            if forbidden is not None:
                _refuse(forbidden, "transformation operation is arbitrary code or shell")
        try:
            operation = _enum(self.operation, TransformationOp, "operation")
        except ProcedureContractError as exc:
            raise ToolGrammarError(str(exc), ToolReason.UNKNOWN_OPERATION) from exc
        object.__setattr__(self, "operation", operation)
        params = _literal(self.parameters if self.parameters is not None else {}, "parameters")
        if not isinstance(params, Mapping):
            raise ToolGrammarError("parameters must be a mapping", ToolReason.GRAMMAR_VIOLATION)
        object.__setattr__(self, "parameters", params)
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        params = self.parameters
        op = self.operation
        if op is TransformationOp.IDENTITY:
            if params:
                _refuse(ToolReason.GRAMMAR_VIOLATION, "identity takes no parameters")
            return
        if op is TransformationOp.SELECT_FIELDS:
            _string_tuple(params.get("fields"), "fields")
            return
        if op is TransformationOp.RENAME_FIELDS:
            mapping = _mapping_of_identifiers(params.get("mapping"), "mapping")
            if not mapping:
                _refuse(ToolReason.GRAMMAR_VIOLATION, "rename-fields mapping must not be empty")
            return
        if op is TransformationOp.FILTER_EQUALS:
            _identifier(params.get("field"), "field")
            _identifier(params.get("equals"), "equals")
            return
        if op is TransformationOp.FILTER_IN:
            _identifier(params.get("field"), "field")
            _string_tuple(params.get("allowed"), "allowed")
            return
        if op is TransformationOp.MAP_ITEMS:
            nested = params.get("program")
            if not isinstance(nested, TransformationProgram):
                TransformationProgram.from_record(nested)
            return
        if op is TransformationOp.SORT_BY:
            _identifier(params.get("key"), "key")
            return
        if op is TransformationOp.LIMIT:
            _positive_int(params.get("limit"), "limit", maximum=MAX_TOOL_ITEMS)
            return
        if op is TransformationOp.PROJECT:
            _identifier(params.get("field"), "field")
            return
        if op in {
            TransformationOp.NORMALIZE_PATH,
            TransformationOp.RELATIVE_PATH,
        }:
            _identifier(params.get("field"), "field")
            return
        if op is TransformationOp.PREFIX_PATH:
            _identifier(params.get("field"), "field")
            _relative_path(params.get("prefix"), "prefix")
            return
        if op is TransformationOp.JOIN_IDENTIFIERS:
            _text(params.get("separator", "."), "separator", required=False)
            return
        if op is TransformationOp.HASH_CANONICAL:
            return
        if op is TransformationOp.CONST:
            if "value" not in params:
                _refuse(ToolReason.GRAMMAR_VIOLATION, "const requires a value")
            _literal(params.get("value"), "const.value")
            return
        if op is TransformationOp.SELECT_RENAME:
            _string_tuple(params.get("fields"), "fields")
            _mapping_of_identifiers(params.get("mapping"), "mapping")
            return
        if op is TransformationOp.SORT_LIMIT:
            _identifier(params.get("key"), "key")
            _positive_int(params.get("limit"), "limit", maximum=MAX_TOOL_ITEMS)
            return
        if op is TransformationOp.PREFIX_NORMALIZE:
            _identifier(params.get("field"), "field")
            _relative_path(params.get("prefix"), "prefix")
            return
        _refuse(ToolReason.UNKNOWN_OPERATION, "transformation operation is not in the reviewed grammar")

    def to_record(self) -> dict[str, Any]:
        parameters = dict(self.parameters)
        nested = parameters.get("program")
        if isinstance(nested, TransformationProgram):
            parameters = {**parameters, "program": nested.to_record()}
        return {"operation": self.operation.value, "parameters": parameters}

    @classmethod
    def from_record(cls, payload: Mapping[str, Any] | TransformationStep) -> TransformationStep:
        if isinstance(payload, TransformationStep):
            return payload
        if not isinstance(payload, Mapping):
            raise ToolGrammarError("transformation step must be a mapping", ToolReason.GRAMMAR_VIOLATION)
        operation = payload.get("operation", "")
        forbidden = _forbidden_operation_token(str(operation))
        if forbidden is not None:
            _refuse(forbidden, "transformation operation is arbitrary code or shell")
        parameters = dict(payload.get("parameters") or {})
        nested = parameters.get("program")
        if nested is not None and not isinstance(nested, TransformationProgram):
            parameters["program"] = TransformationProgram.from_record(nested)
        return cls(operation=operation, parameters=parameters)


def _steps(values: Any, *, required: bool = True) -> tuple[TransformationStep, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray, memoryview)):
        raw = values
    else:
        raise ToolGrammarError("steps must be a sequence", ToolReason.GRAMMAR_VIOLATION)
    if len(raw) > MAX_TOOL_STEPS:
        _refuse(ToolReason.UNBOUNDED, "transformation exceeds the step bound")
    result = tuple(TransformationStep.from_record(item) for item in raw)
    if required and not result:
        _refuse(ToolReason.GRAMMAR_VIOLATION, "transformation program must not be empty")
    return result


@dataclass(frozen=True)
class TransformationProgram:
    """Closed sequence of reviewed transformation operations."""

    steps: tuple[TransformationStep, ...]
    grammar_revision: str = GRAMMAR_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", _steps(self.steps))
        object.__setattr__(
            self,
            "grammar_revision",
            _identifier(self.grammar_revision, "grammar_revision"),
        )
        if self.grammar_revision != GRAMMAR_REVISION:
            _refuse(ToolReason.GRAMMAR_VIOLATION, "transformation grammar revision is not current")
        if any(step.operation is TransformationOp.MAP_ITEMS for step in self.steps):
            depth = _map_depth(self)
            if depth > 2:
                _refuse(ToolReason.UNBOUNDED, "map-items nesting exceeds its bound")

    @property
    def operations(self) -> tuple[TransformationOp, ...]:
        return tuple(step.operation for step in self.steps)

    def to_record(self) -> dict[str, Any]:
        return {
            "grammar_revision": self.grammar_revision,
            "steps": tuple(step.to_record() for step in self.steps),
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any] | TransformationProgram | Sequence[Any]) -> TransformationProgram:
        if isinstance(payload, TransformationProgram):
            return payload
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, Mapping)):
            return cls(steps=_steps(payload))
        if not isinstance(payload, Mapping):
            raise ToolGrammarError("transformation program must be a mapping", ToolReason.GRAMMAR_VIOLATION)
        return cls(
            steps=payload.get("steps", ()),
            grammar_revision=payload.get("grammar_revision", GRAMMAR_REVISION),
        )


def _map_depth(program: TransformationProgram) -> int:
    depth = 0
    for step in program.steps:
        if step.operation is not TransformationOp.MAP_ITEMS:
            continue
        nested = step.parameters.get("program")
        nested_program = (
            nested if isinstance(nested, TransformationProgram) else TransformationProgram.from_record(nested)
        )
        depth = max(depth, 1 + _map_depth(nested_program))
    return depth


@dataclass(frozen=True)
class TransformationTemplate:
    """Reviewed composition of closed operations.  Parameters fill data only."""

    template_id: str
    operations: tuple[TransformationOp, ...]
    required_parameters: tuple[str, ...]
    requires_scope: bool = False
    effect_class: EffectClass = EffectClass.OBSERVE
    repair_template_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "template_id", _identifier(self.template_id, "template_id")
        )
        operations = tuple(
            _enum(item, TransformationOp, "operations") for item in self.operations
        )
        if not operations:
            _refuse(ToolReason.GRAMMAR_VIOLATION, "template must declare operations")
        object.__setattr__(self, "operations", operations)
        object.__setattr__(
            self,
            "required_parameters",
            _string_tuple(self.required_parameters, "required_parameters", required=False),
        )
        object.__setattr__(self, "requires_scope", _bool(self.requires_scope, "requires_scope"))
        object.__setattr__(
            self, "effect_class", _enum(self.effect_class, EffectClass, "effect_class")
        )
        if self.effect_class not in ALLOWED_EFFECT_CLASSES:
            _refuse(ToolReason.EFFECT_FORBIDDEN, "template effect class is not observe or validation")
        object.__setattr__(
            self,
            "repair_template_ids",
            _strings(self.repair_template_ids, "repair_template_ids", identifiers=True),
        )
        unknown = [item for item in self.repair_template_ids if item not in APPROVED_REPAIR_TEMPLATE_IDS]
        if unknown:
            _refuse(ToolReason.UNKNOWN_TEMPLATE, "template referenced an unapproved repair template")

    def instantiate(self, parameters: Mapping[str, Any]) -> TransformationProgram:
        params = _literal(parameters, "template_parameters")
        if not isinstance(params, Mapping):
            raise ToolGrammarError("template parameters must be a mapping", ToolReason.GRAMMAR_VIOLATION)
        if len(params) > MAX_TEMPLATE_PARAMS:
            _refuse(ToolReason.UNBOUNDED, "template parameters exceed the mapping bound")
        missing = [name for name in self.required_parameters if name not in params]
        if missing:
            _refuse(ToolReason.GRAMMAR_VIOLATION, "template is missing a required parameter")
        steps: list[TransformationStep] = []
        for operation in self.operations:
            step_params: dict[str, Any] = {}
            if operation is TransformationOp.SELECT_FIELDS:
                step_params["fields"] = params["fields"]
            elif operation is TransformationOp.RENAME_FIELDS:
                step_params["mapping"] = params["mapping"]
            elif operation is TransformationOp.FILTER_EQUALS:
                step_params["field"] = params["field"]
                step_params["equals"] = params["equals"]
            elif operation is TransformationOp.FILTER_IN:
                step_params["field"] = params["field"]
                step_params["allowed"] = params["allowed"]
            elif operation is TransformationOp.SORT_BY:
                step_params["key"] = params["key"]
            elif operation is TransformationOp.LIMIT:
                step_params["limit"] = params["limit"]
            elif operation is TransformationOp.PROJECT:
                step_params["field"] = params["field"]
            elif operation in {
                TransformationOp.NORMALIZE_PATH,
                TransformationOp.RELATIVE_PATH,
            }:
                step_params["field"] = params["field"]
            elif operation is TransformationOp.PREFIX_PATH:
                step_params["field"] = params["field"]
                step_params["prefix"] = params["prefix"]
            elif operation is TransformationOp.HASH_CANONICAL:
                if "fields" in params:
                    steps.append(
                        TransformationStep(
                            TransformationOp.SELECT_FIELDS, {"fields": params["fields"]}
                        )
                    )
                steps.append(TransformationStep(operation, {}))
                continue
            elif operation is TransformationOp.JOIN_IDENTIFIERS:
                if "separator" in params:
                    step_params["separator"] = params["separator"]
            elif operation is TransformationOp.CONST:
                step_params["value"] = params["value"]
            steps.append(TransformationStep(operation, step_params))
        return TransformationProgram(steps=tuple(steps))

    def to_record(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "operations": tuple(item.value for item in self.operations),
            "required_parameters": self.required_parameters,
            "requires_scope": self.requires_scope,
            "effect_class": self.effect_class.value,
            "repair_template_ids": self.repair_template_ids,
        }


def _default_templates() -> tuple[TransformationTemplate, ...]:
    return (
        TransformationTemplate(
            template_id="select-and-rename",
            operations=(TransformationOp.SELECT_FIELDS, TransformationOp.RENAME_FIELDS),
            required_parameters=("fields", "mapping"),
        ),
        TransformationTemplate(
            template_id="scoped-path-normalize",
            operations=(TransformationOp.NORMALIZE_PATH,),
            required_parameters=("field",),
            requires_scope=True,
        ),
        TransformationTemplate(
            template_id="bounded-sort-limit",
            operations=(TransformationOp.SORT_BY, TransformationOp.LIMIT),
            required_parameters=("key", "limit"),
        ),
        TransformationTemplate(
            template_id="canonical-digest",
            operations=(TransformationOp.HASH_CANONICAL,),
            required_parameters=("fields",),
        ),
        TransformationTemplate(
            template_id="filter-identifier-set",
            operations=(TransformationOp.FILTER_IN,),
            required_parameters=("field", "allowed"),
        ),
        TransformationTemplate(
            template_id="prefix-scoped-path",
            operations=(TransformationOp.PREFIX_PATH, TransformationOp.NORMALIZE_PATH),
            required_parameters=("field", "prefix"),
            requires_scope=True,
        ),
        TransformationTemplate(
            template_id="select-filter-limit",
            operations=(
                TransformationOp.SELECT_FIELDS,
                TransformationOp.FILTER_EQUALS,
                TransformationOp.LIMIT,
            ),
            required_parameters=("fields", "field", "equals", "limit"),
        ),
        TransformationTemplate(
            template_id="repair-template-path-scope",
            operations=(
                TransformationOp.PREFIX_PATH,
                TransformationOp.NORMALIZE_PATH,
                TransformationOp.SELECT_FIELDS,
            ),
            required_parameters=("field", "prefix", "fields"),
            requires_scope=True,
            repair_template_ids=("template.path-scope", "template.import-purity"),
        ),
    )


class TemplateLibrary:
    """Reviewed grammar/template library.  Unknown templates fail closed."""

    revision: ClassVar[str] = TEMPLATE_LIBRARY_REVISION

    def __init__(self, templates: Sequence[TransformationTemplate] | None = None) -> None:
        items = tuple(templates) if templates is not None else _default_templates()
        index: dict[str, TransformationTemplate] = {}
        for item in items:
            if not isinstance(item, TransformationTemplate):
                raise ToolGrammarError("template library requires TransformationTemplate records")
            if item.template_id in index:
                _refuse(ToolReason.GRAMMAR_VIOLATION, "template library contains a duplicate template_id")
            index[item.template_id] = item
        self._templates = index

    def get(self, template_id: str) -> TransformationTemplate:
        identity = _identifier(template_id, "template_id")
        template = self._templates.get(identity)
        if template is None:
            _refuse(ToolReason.UNKNOWN_TEMPLATE, "template is not in the reviewed library")
        return template

    def contains(self, template_id: str) -> bool:
        return template_id in self._templates

    @property
    def template_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._templates))

    def instantiate(self, template_id: str, parameters: Mapping[str, Any]) -> TransformationProgram:
        return self.get(template_id).instantiate(parameters)


def _optimize_steps(steps: Sequence[TransformationStep]) -> tuple[TransformationStep, ...]:
    optimized: list[TransformationStep] = []
    index = 0
    while index < len(steps):
        current = steps[index]
        nxt = steps[index + 1] if index + 1 < len(steps) else None
        if (
            current.operation is TransformationOp.SELECT_FIELDS
            and nxt is not None
            and nxt.operation is TransformationOp.RENAME_FIELDS
        ):
            optimized.append(
                TransformationStep(
                    TransformationOp.SELECT_RENAME,
                    {
                        "fields": current.parameters["fields"],
                        "mapping": nxt.parameters["mapping"],
                    },
                )
            )
            index += 2
            continue
        if (
            current.operation is TransformationOp.SORT_BY
            and nxt is not None
            and nxt.operation is TransformationOp.LIMIT
        ):
            optimized.append(
                TransformationStep(
                    TransformationOp.SORT_LIMIT,
                    {"key": current.parameters["key"], "limit": nxt.parameters["limit"]},
                )
            )
            index += 2
            continue
        if (
            current.operation is TransformationOp.PREFIX_PATH
            and nxt is not None
            and nxt.operation is TransformationOp.NORMALIZE_PATH
            and current.parameters.get("field") == nxt.parameters.get("field")
        ):
            optimized.append(
                TransformationStep(
                    TransformationOp.PREFIX_NORMALIZE,
                    {
                        "field": current.parameters["field"],
                        "prefix": current.parameters["prefix"],
                    },
                )
            )
            index += 2
            continue
        optimized.append(current)
        index += 1
    return tuple(optimized)


class _Runtime:
    """Mutable remaining budget for one interpretation.  Never escapes the module."""

    def __init__(self, resources: ToolResourceEnvelope, scope_paths: Sequence[str]) -> None:
        self.remaining_steps = resources.max_steps
        self.max_items = resources.max_items
        self.max_output_bytes = resources.max_output_bytes
        self.max_path_depth = resources.max_path_depth
        self.scope_paths = tuple(scope_paths)
        self.resources = resources

    def consume_step(self) -> None:
        if self.remaining_steps < 1:
            _refuse(ToolReason.RESOURCE_EXCEEDED, "transformation exceeded its step bound")
        self.remaining_steps -= 1

    def check_sequence(self, value: Sequence[Any], field_name: str) -> None:
        if len(value) > self.max_items:
            _refuse(ToolReason.RESOURCE_EXCEEDED, f"{field_name} exceeds the item bound")

    def check_output(self, value: Any) -> None:
        encoded = canonical_json_bytes(_jsonable(value))
        if len(encoded) > self.max_output_bytes:
            _refuse(ToolReason.RESOURCE_EXCEEDED, "transformation exceeded its output-byte bound")

    def check_path(self, path: str, field_name: str) -> str:
        try:
            normalized = _relative_path(path, field_name)
        except ProcedureSafetyError as exc:
            raise ToolSafetyError(str(exc), ToolReason.PATH_ESCAPE) from exc
        if _path_depth(normalized) > self.max_path_depth:
            _refuse(ToolReason.RESOURCE_EXCEEDED, f"{field_name} exceeds the path-depth bound")
        if not _path_is_within(normalized, self.scope_paths):
            _refuse(ToolReason.PATH_ESCAPE, f"{field_name} escapes the declared path scope")
        return normalized


def _jsonable(value: Any) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        return [_jsonable(item) for item in value]
    raise ToolSynthesisError("interpretation produced an unsupported value", ToolReason.SCHEMA_MISMATCH)


def _require_mapping(value: Any, operation: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _refuse(ToolReason.SCHEMA_MISMATCH, f"{operation} requires a mapping input")
    return value


def _require_sequence(value: Any, operation: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray, memoryview)) or not isinstance(value, Sequence):
        _refuse(ToolReason.SCHEMA_MISMATCH, f"{operation} requires a sequence input")
    return value


def _select_fields(value: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    return {name: value[name] for name in fields if name in value}


def _rename_fields(value: Mapping[str, Any], mapping: Mapping[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        result[mapping.get(key, key)] = item
    return result


def _apply_step(step: TransformationStep, value: Any, runtime: _Runtime) -> Any:
    runtime.consume_step()
    op = step.operation
    params = step.parameters
    if op is TransformationOp.IDENTITY:
        return value
    if op is TransformationOp.SELECT_FIELDS:
        fields = _string_tuple(params.get("fields"), "fields")
        return _select_fields(_require_mapping(value, op.value), fields)
    if op is TransformationOp.RENAME_FIELDS:
        mapping = _mapping_of_identifiers(params.get("mapping"), "mapping")
        return _rename_fields(_require_mapping(value, op.value), mapping)
    if op is TransformationOp.SELECT_RENAME:
        fields = _string_tuple(params.get("fields"), "fields")
        mapping = _mapping_of_identifiers(params.get("mapping"), "mapping")
        selected = _select_fields(_require_mapping(value, op.value), fields)
        return _rename_fields(selected, mapping)
    if op is TransformationOp.FILTER_EQUALS:
        field_name = _identifier(params.get("field"), "field")
        expected = _identifier(params.get("equals"), "equals")
        items = _require_sequence(value, op.value)
        runtime.check_sequence(items, "filter-equals")
        return tuple(
            item
            for item in items
            if isinstance(item, Mapping) and item.get(field_name) == expected
        )
    if op is TransformationOp.FILTER_IN:
        field_name = _identifier(params.get("field"), "field")
        allowed = set(_string_tuple(params.get("allowed"), "allowed"))
        items = _require_sequence(value, op.value)
        runtime.check_sequence(items, "filter-in")
        return tuple(
            item
            for item in items
            if isinstance(item, Mapping) and item.get(field_name) in allowed
        )
    if op is TransformationOp.MAP_ITEMS:
        nested = params.get("program")
        program = nested if isinstance(nested, TransformationProgram) else TransformationProgram.from_record(nested)
        items = _require_sequence(value, op.value)
        runtime.check_sequence(items, "map-items")
        return tuple(_interpret_program(program, item, runtime) for item in items)
    if op is TransformationOp.SORT_BY:
        key = _identifier(params.get("key"), "key")
        items = _require_sequence(value, op.value)
        runtime.check_sequence(items, "sort-by")
        decorated: list[tuple[str, int, Any]] = []
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                _refuse(ToolReason.SCHEMA_MISMATCH, "sort-by requires mapping items")
            sort_key = item.get(key, "")
            if type(sort_key) is not str:
                sort_key = "" if sort_key is None else str(sort_key)
            decorated.append((sort_key, index, item))
        decorated.sort(key=lambda row: (row[0], row[1]))
        return tuple(row[2] for row in decorated)
    if op is TransformationOp.LIMIT:
        limit = _positive_int(params.get("limit"), "limit", maximum=runtime.max_items)
        items = _require_sequence(value, op.value)
        runtime.check_sequence(items, "limit")
        return tuple(items[:limit])
    if op is TransformationOp.SORT_LIMIT:
        key = _identifier(params.get("key"), "key")
        limit = _positive_int(params.get("limit"), "limit", maximum=runtime.max_items)
        items = _require_sequence(value, op.value)
        runtime.check_sequence(items, "sort-limit")
        decorated: list[tuple[str, int, Any]] = []
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                _refuse(ToolReason.SCHEMA_MISMATCH, "sort-limit requires mapping items")
            sort_key = item.get(key, "")
            if type(sort_key) is not str:
                sort_key = "" if sort_key is None else str(sort_key)
            decorated.append((sort_key, index, item))
        decorated.sort(key=lambda row: (row[0], row[1]))
        return tuple(row[2] for row in decorated[:limit])
    if op is TransformationOp.PROJECT:
        field_name = _identifier(params.get("field"), "field")
        mapping = _require_mapping(value, op.value)
        if field_name not in mapping:
            _refuse(ToolReason.SCHEMA_MISMATCH, "project field is absent from the mapping")
        return mapping[field_name]
    if op is TransformationOp.NORMALIZE_PATH:
        return _mutate_path_field(value, _identifier(params.get("field"), "field"), runtime, prefix="")
    if op is TransformationOp.PREFIX_PATH:
        return _mutate_path_field(
            value,
            _identifier(params.get("field"), "field"),
            runtime,
            prefix=_relative_path(params.get("prefix"), "prefix"),
            normalize=False,
        )
    if op is TransformationOp.PREFIX_NORMALIZE:
        return _mutate_path_field(
            value,
            _identifier(params.get("field"), "field"),
            runtime,
            prefix=_relative_path(params.get("prefix"), "prefix"),
        )
    if op is TransformationOp.RELATIVE_PATH:
        return _mutate_path_field(
            value,
            _identifier(params.get("field"), "field"),
            runtime,
            prefix="",
            relative_to=True,
        )
    if op is TransformationOp.JOIN_IDENTIFIERS:
        separator = _text(params.get("separator", "."), "separator", required=False) or "."
        items = _require_sequence(value, op.value)
        runtime.check_sequence(items, "join-identifiers")
        parts = tuple(_identifier(item, "join-identifiers") for item in items)
        joined = separator.join(parts)
        if len(joined.encode("utf-8")) > MAX_TEXT_BYTES:
            _refuse(ToolReason.RESOURCE_EXCEEDED, "joined identifier exceeds its byte bound")
        return joined
    if op is TransformationOp.HASH_CANONICAL:
        return content_identity(_jsonable(value))
    if op is TransformationOp.CONST:
        return _literal(params.get("value"), "const.value")
    _refuse(ToolReason.UNKNOWN_OPERATION, "transformation operation is not in the reviewed grammar")


def _mutate_path_field(
    value: Any,
    field_name: str,
    runtime: _Runtime,
    *,
    prefix: str = "",
    normalize: bool = True,
    relative_to: bool = False,
) -> Any:
    if isinstance(value, str):
        path = value
        mapping = None
    else:
        mapping = dict(_require_mapping(value, "path-op"))
        if field_name not in mapping or type(mapping[field_name]) is not str:
            _refuse(ToolReason.SCHEMA_MISMATCH, "path operation requires a string field")
        path = mapping[field_name]
    if relative_to and runtime.scope_paths:
        root = runtime.scope_paths[0]
        if path == root:
            path = "."
        elif path.startswith(root + "/"):
            path = path[len(root) + 1 :]
            if not path:
                path = "."
    if prefix:
        path = prefix if path in {"", "."} else f"{prefix}/{path}"
    if normalize or prefix:
        path = runtime.check_path(path, field_name)
    else:
        path = runtime.check_path(path, field_name)
    if mapping is None:
        return path
    mapping[field_name] = path
    return mapping


def _interpret_program(program: TransformationProgram, value: Any, runtime: _Runtime) -> Any:
    current = _literal(value, "input")
    for step in program.steps:
        current = _apply_step(step, current, runtime)
    runtime.check_output(current)
    return current


def _py_literal(value: Any) -> str:
    if value is None or type(value) in {bool, int, str}:
        return repr(value)
    if isinstance(value, Mapping):
        items = ", ".join(f"{_py_literal(key)}: {_py_literal(item)}" for key, item in value.items())
        return "{" + items + "}"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        inner = ", ".join(_py_literal(item) for item in value)
        return "(" + inner + ("," if len(value) == 1 else "") + ")"
    raise ToolGrammarError("cannot emit a closed Python literal", ToolReason.ARBITRARY_CODE)


def _emit_python(program: TransformationProgram) -> str:
    lines = ["def transform(value, env):", "    current = value"]
    for step in program.steps:
        op = step.operation
        params = step.parameters
        if op is TransformationOp.IDENTITY:
            lines.append("    current = op_identity(current)")
        elif op is TransformationOp.SELECT_FIELDS:
            lines.append(f"    current = op_select_fields(current, {_py_literal(tuple(params['fields']))})")
        elif op is TransformationOp.RENAME_FIELDS:
            lines.append(
                f"    current = op_rename_fields(current, {_py_literal(dict(params['mapping']))})"
            )
        elif op is TransformationOp.SELECT_RENAME:
            lines.append(
                "    current = op_select_rename(current, "
                f"{_py_literal(tuple(params['fields']))}, {_py_literal(dict(params['mapping']))})"
            )
        elif op is TransformationOp.FILTER_EQUALS:
            lines.append(
                "    current = op_filter_equals(current, "
                f"{_py_literal(params['field'])}, {_py_literal(params['equals'])})"
            )
        elif op is TransformationOp.FILTER_IN:
            lines.append(
                "    current = op_filter_in(current, "
                f"{_py_literal(params['field'])}, {_py_literal(tuple(params['allowed']))})"
            )
        elif op is TransformationOp.MAP_ITEMS:
            nested = params["program"]
            nested_program = (
                nested if isinstance(nested, TransformationProgram) else TransformationProgram.from_record(nested)
            )
            lines.append(
                f"    current = op_map_items(current, {_py_literal(nested_program.to_record())}, env)"
            )
        elif op is TransformationOp.SORT_BY:
            lines.append(f"    current = op_sort_by(current, {_py_literal(params['key'])})")
        elif op is TransformationOp.LIMIT:
            lines.append(f"    current = op_limit(current, {_py_literal(params['limit'])})")
        elif op is TransformationOp.SORT_LIMIT:
            lines.append(
                "    current = op_sort_limit(current, "
                f"{_py_literal(params['key'])}, {_py_literal(params['limit'])})"
            )
        elif op is TransformationOp.PROJECT:
            lines.append(f"    current = op_project(current, {_py_literal(params['field'])})")
        elif op is TransformationOp.NORMALIZE_PATH:
            lines.append(
                f"    current = op_normalize_path_field(current, {_py_literal(params['field'])}, env)"
            )
        elif op is TransformationOp.PREFIX_PATH:
            lines.append(
                "    current = op_prefix_path_field(current, "
                f"{_py_literal(params['field'])}, {_py_literal(params['prefix'])}, env)"
            )
        elif op is TransformationOp.PREFIX_NORMALIZE:
            lines.append(
                "    current = op_prefix_normalize(current, "
                f"{_py_literal(params['field'])}, {_py_literal(params['prefix'])}, env)"
            )
        elif op is TransformationOp.RELATIVE_PATH:
            lines.append(
                f"    current = op_relative_path_field(current, {_py_literal(params['field'])}, env)"
            )
        elif op is TransformationOp.JOIN_IDENTIFIERS:
            separator = params.get("separator", ".")
            lines.append(f"    current = op_join_identifiers(current, {_py_literal(separator)})")
        elif op is TransformationOp.HASH_CANONICAL:
            lines.append("    current = op_hash_canonical(current)")
        elif op is TransformationOp.CONST:
            lines.append(f"    current = op_const(current, {_py_literal(params['value'])})")
        else:
            _refuse(ToolReason.UNKNOWN_OPERATION, "cannot translate an unknown operation")
    lines.append("    return current")
    source = "\n".join(lines) + "\n"
    if len(source.encode("utf-8")) > MAX_TRANSLATION_BYTES:
        _refuse(ToolReason.UNBOUNDED, "optimized translation exceeds its byte bound")
    return source


def _assert_restricted_ast(source: str) -> ast.Module:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ToolSafetyError("optimized translation is not valid restricted Python", ToolReason.ARBITRARY_CODE) from exc
    for node in ast.walk(tree):
        name = type(node).__name__
        if name in _FORBIDDEN_AST_TYPES or name not in _ALLOWED_AST_TYPES:
            _refuse(ToolReason.ARBITRARY_CODE, "optimized translation contains a forbidden AST node")
        if isinstance(node, ast.Attribute):
            _refuse(ToolReason.ARBITRARY_CODE, "optimized translation cannot use attribute access")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_HELPERS:
                _refuse(ToolReason.ARBITRARY_CODE, "optimized translation called an unallowlisted helper")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            _refuse(ToolReason.ARBITRARY_CODE, "optimized translation cannot use dunder names")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        _refuse(ToolReason.ARBITRARY_CODE, "optimized translation must be a single transform function")
    func = tree.body[0]
    if (
        func.name != "transform"
        or func.args.vararg
        or func.args.kwarg
        or func.decorator_list
        or func.args.defaults
        or func.args.kwonlyargs
        or func.args.posonlyargs
        or getattr(func, "type_params", ())
    ):
        _refuse(ToolReason.ARBITRARY_CODE, "optimized translation function shape is not closed")
    arg_names = [item.arg for item in func.args.args]
    if arg_names != ["value", "env"]:
        _refuse(ToolReason.ARBITRARY_CODE, "optimized translation must take value and env")
    return tree


class _RestrictedPython:
    """Interpret template-generated Python by walking a closed AST.  Never exec."""

    def __init__(self, helpers: Mapping[str, Any]) -> None:
        self._helpers = helpers

    def run(self, source: str, value: Any, env: _Runtime) -> Any:
        tree = _assert_restricted_ast(source)
        func = tree.body[0]
        assert isinstance(func, ast.FunctionDef)
        locals_: dict[str, Any] = {"value": value, "env": env}
        result: Any = None
        for stmt in func.body:
            produced = self._exec_stmt(stmt, locals_)
            if isinstance(stmt, ast.Return):
                return produced
            result = produced
        return result

    def _exec_stmt(self, stmt: ast.stmt, locals_: dict[str, Any]) -> Any:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                _refuse(ToolReason.ARBITRARY_CODE, "optimized translation assignment is not closed")
            value = self._eval(stmt.value, locals_)
            locals_[stmt.targets[0].id] = value
            return value
        if isinstance(stmt, ast.Return):
            if stmt.value is None:
                return None
            return self._eval(stmt.value, locals_)
        if isinstance(stmt, ast.Pass):
            return None
        _refuse(ToolReason.ARBITRARY_CODE, "optimized translation statement is not closed")

    def _eval(self, expr: ast.expr, locals_: dict[str, Any]) -> Any:
        if isinstance(expr, ast.Constant):
            if type(expr.value) not in {str, int, bool, type(None)}:
                _refuse(ToolReason.ARBITRARY_CODE, "optimized translation constant is not closed")
            return expr.value
        if isinstance(expr, ast.Name):
            if expr.id in locals_:
                return locals_[expr.id]
            if expr.id in self._helpers:
                return self._helpers[expr.id]
            _refuse(ToolReason.ARBITRARY_CODE, "optimized translation referenced an unknown name")
        if isinstance(expr, ast.Tuple):
            return tuple(self._eval(item, locals_) for item in expr.elts)
        if isinstance(expr, ast.List):
            return [self._eval(item, locals_) for item in expr.elts]
        if isinstance(expr, ast.Dict):
            result: dict[Any, Any] = {}
            for key_node, value_node in zip(expr.keys, expr.values):
                if key_node is None:
                    _refuse(ToolReason.ARBITRARY_CODE, "optimized translation cannot splat dict keys")
                key = self._eval(key_node, locals_)
                if type(key) is not str:
                    _refuse(ToolReason.ARBITRARY_CODE, "optimized translation dict keys must be strings")
                result[key] = self._eval(value_node, locals_)
            return result
        if isinstance(expr, ast.Call):
            if not isinstance(expr.func, ast.Name):
                _refuse(ToolReason.ARBITRARY_CODE, "optimized translation call target is not closed")
            helper = self._helpers.get(expr.func.id)
            if helper is None:
                _refuse(ToolReason.ARBITRARY_CODE, "optimized translation called an unknown helper")
            args = [self._eval(item, locals_) for item in expr.args]
            if expr.keywords:
                _refuse(ToolReason.ARBITRARY_CODE, "optimized translation cannot use keyword calls")
            return helper(*args)
        _refuse(ToolReason.ARBITRARY_CODE, "optimized translation expression is not closed")


def _helpers_for(runtime: _Runtime) -> dict[str, Any]:
    dsl = TransformationDsl()

    def _with_runtime(program: TransformationProgram, value: Any) -> Any:
        return _interpret_program(program, value, runtime)

    return {
        "op_identity": lambda current: _apply_step(
            TransformationStep(TransformationOp.IDENTITY, {}), current, runtime
        ),
        "op_select_fields": lambda current, fields: _apply_step(
            TransformationStep(TransformationOp.SELECT_FIELDS, {"fields": fields}),
            current,
            runtime,
        ),
        "op_rename_fields": lambda current, mapping: _apply_step(
            TransformationStep(TransformationOp.RENAME_FIELDS, {"mapping": mapping}),
            current,
            runtime,
        ),
        "op_select_rename": lambda current, fields, mapping: _apply_step(
            TransformationStep(
                TransformationOp.SELECT_RENAME, {"fields": fields, "mapping": mapping}
            ),
            current,
            runtime,
        ),
        "op_filter_equals": lambda current, field_name, equals: _apply_step(
            TransformationStep(
                TransformationOp.FILTER_EQUALS, {"field": field_name, "equals": equals}
            ),
            current,
            runtime,
        ),
        "op_filter_in": lambda current, field_name, allowed: _apply_step(
            TransformationStep(
                TransformationOp.FILTER_IN, {"field": field_name, "allowed": allowed}
            ),
            current,
            runtime,
        ),
        "op_map_items": lambda current, nested, env: _apply_step(
            TransformationStep(
                TransformationOp.MAP_ITEMS,
                {"program": TransformationProgram.from_record(nested)},
            ),
            current,
            runtime,
        ),
        "op_sort_by": lambda current, key: _apply_step(
            TransformationStep(TransformationOp.SORT_BY, {"key": key}), current, runtime
        ),
        "op_limit": lambda current, limit: _apply_step(
            TransformationStep(TransformationOp.LIMIT, {"limit": limit}), current, runtime
        ),
        "op_sort_limit": lambda current, key, limit: _apply_step(
            TransformationStep(TransformationOp.SORT_LIMIT, {"key": key, "limit": limit}),
            current,
            runtime,
        ),
        "op_project": lambda current, field_name: _apply_step(
            TransformationStep(TransformationOp.PROJECT, {"field": field_name}),
            current,
            runtime,
        ),
        "op_normalize_path_field": lambda current, field_name, env: _apply_step(
            TransformationStep(TransformationOp.NORMALIZE_PATH, {"field": field_name}),
            current,
            runtime,
        ),
        "op_prefix_path_field": lambda current, field_name, prefix, env: _apply_step(
            TransformationStep(
                TransformationOp.PREFIX_PATH, {"field": field_name, "prefix": prefix}
            ),
            current,
            runtime,
        ),
        "op_prefix_normalize": lambda current, field_name, prefix, env: _apply_step(
            TransformationStep(
                TransformationOp.PREFIX_NORMALIZE, {"field": field_name, "prefix": prefix}
            ),
            current,
            runtime,
        ),
        "op_relative_path_field": lambda current, field_name, env: _apply_step(
            TransformationStep(TransformationOp.RELATIVE_PATH, {"field": field_name}),
            current,
            runtime,
        ),
        "op_join_identifiers": lambda current, separator: _apply_step(
            TransformationStep(TransformationOp.JOIN_IDENTIFIERS, {"separator": separator}),
            current,
            runtime,
        ),
        "op_hash_canonical": lambda current: _apply_step(
            TransformationStep(TransformationOp.HASH_CANONICAL, {}), current, runtime
        ),
        "op_const": lambda current, literal: _apply_step(
            TransformationStep(TransformationOp.CONST, {"value": literal}), current, runtime
        ),
        "dsl": dsl,
        "with_runtime": _with_runtime,
    }


class TransformationDsl:
    """Reviewed grammar, template library, interpreter, and optimizer."""

    revision: ClassVar[str] = DSL_REVISION

    def __init__(self, library: TemplateLibrary | None = None) -> None:
        self.library = library if library is not None else TemplateLibrary()

    def parse_program(self, payload: Mapping[str, Any] | Sequence[Any] | TransformationProgram) -> TransformationProgram:
        if isinstance(payload, TransformationProgram):
            program = payload
        else:
            program = TransformationProgram.from_record(payload)
        for step in program.steps:
            forbidden = _forbidden_operation_token(step.operation.value)
            if forbidden is not None:
                _refuse(forbidden, "program contains a forbidden operation")
        return program

    def instantiate(self, template_id: str, parameters: Mapping[str, Any]) -> TransformationProgram:
        return self.library.instantiate(template_id, parameters)

    def optimize(self, program: TransformationProgram) -> TransformationProgram:
        parsed = self.parse_program(program)
        return TransformationProgram(steps=_optimize_steps(parsed.steps))

    def interpret(
        self,
        program: TransformationProgram,
        value: Any,
        *,
        scope_paths: Sequence[str],
        resources: ToolResourceEnvelope | None = None,
    ) -> Any:
        parsed = self.parse_program(program)
        envelope = resources if resources is not None else ToolResourceEnvelope()
        scopes = _strings(scope_paths, "scope_paths", paths=True, required=True, limit=MAX_SCOPE_PATHS)
        runtime = _Runtime(envelope, scopes)
        return _interpret_program(parsed, value, runtime)

    def emit_optimized_python(self, program: TransformationProgram) -> str:
        optimized = self.optimize(program)
        source = _emit_python(optimized)
        _assert_restricted_ast(source)
        return source

    def interpret_python(
        self,
        source: str,
        value: Any,
        *,
        scope_paths: Sequence[str],
        resources: ToolResourceEnvelope | None = None,
    ) -> Any:
        envelope = resources if resources is not None else ToolResourceEnvelope()
        scopes = _strings(scope_paths, "scope_paths", paths=True, required=True, limit=MAX_SCOPE_PATHS)
        runtime = _Runtime(envelope, scopes)
        interpreter = _RestrictedPython(_helpers_for(runtime))
        result = interpreter.run(source, _literal(value, "input"), runtime)
        runtime.check_output(result)
        return result


DeterministicToolDsl = TransformationDsl


@dataclass(frozen=True)
class ToolFixture:
    """Compact evaluation or adversarial input.  Large bodies stay out of band."""

    fixture_id: str
    kind: FixtureKind
    payload: Any
    expected_refusal: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "fixture_id", _identifier(self.fixture_id, "fixture_id"))
        object.__setattr__(self, "kind", _enum(self.kind, FixtureKind, "kind"))
        object.__setattr__(self, "payload", _fixture_payload(self.payload, "payload"))
        object.__setattr__(
            self,
            "expected_refusal",
            _identifier(self.expected_refusal, "expected_refusal", required=False),
        )

    @property
    def content_id(self) -> str:
        return content_identity(
            {
                "fixture_id": self.fixture_id,
                "kind": self.kind.value,
                "payload": _jsonable(self.payload),
                "expected_refusal": self.expected_refusal,
            }
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "kind": self.kind.value,
            "payload": self.payload,
            "expected_refusal": self.expected_refusal,
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any] | ToolFixture) -> ToolFixture:
        if isinstance(payload, ToolFixture):
            return payload
        if not isinstance(payload, Mapping):
            raise ToolSynthesisError("fixture must be a mapping")
        return cls(
            fixture_id=payload.get("fixture_id", ""),
            kind=payload.get("kind", FixtureKind.HAPPY),
            payload=payload.get("payload"),
            expected_refusal=payload.get("expected_refusal", ""),
        )


def _fixtures(values: Any) -> tuple[ToolFixture, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray, memoryview)):
        raw = values
    else:
        raise ToolSynthesisError("fixtures must be a sequence")
    if len(raw) > MAX_FIXTURES:
        _refuse(ToolReason.UNBOUNDED, "fixtures exceed the fixture bound")
    result: list[ToolFixture] = []
    seen: set[str] = set()
    for item in raw:
        record = ToolFixture.from_record(item)
        if record.fixture_id in seen:
            raise ToolSynthesisError("fixtures contain a duplicate fixture_id")
        seen.add(record.fixture_id)
        result.append(record)
    return tuple(result)


def default_adversarial_fixtures(*, scope_paths: Sequence[str]) -> tuple[ToolFixture, ...]:
    """Compact adversarial recipes that fail at the input boundary for every tool."""

    del scope_paths
    overflow = tuple({"id": f"item-{index}", "kind": "keep"} for index in range(MAX_TOOL_ITEMS + 1))
    return (
        ToolFixture(
            fixture_id="adv.path-escape",
            kind=FixtureKind.ADVERSARIAL,
            payload={"path": "../secrets", "kind": "escape"},
            expected_refusal=ToolReason.PATH_ESCAPE.value,
        ),
        ToolFixture(
            fixture_id="adv.absolute-path",
            kind=FixtureKind.ADVERSARIAL,
            payload={"path": "/etc/passwd", "kind": "absolute"},
            expected_refusal=ToolReason.PATH_ESCAPE.value,
        ),
        ToolFixture(
            fixture_id="adv.shell-token",
            kind=FixtureKind.ADVERSARIAL,
            payload={"kind": "arbitrary-shell"},
            expected_refusal=ToolReason.ARBITRARY_SHELL.value,
        ),
        ToolFixture(
            fixture_id="adv.unbounded-items",
            kind=FixtureKind.ADVERSARIAL,
            payload=overflow,
            expected_refusal=ToolReason.RESOURCE_EXCEEDED.value,
        ),
        ToolFixture(
            fixture_id="adv.executable-field",
            kind=FixtureKind.ADVERSARIAL,
            payload={"callback": "not-code"},
            expected_refusal=ToolReason.INJECTION_REJECTED.value,
        ),
    )


@dataclass(frozen=True)
class ToolSynthesisRequest:
    """Declarative synthesis request.  Source is a template or closed program."""

    bindings: ArtifactBindings
    tool_id: str
    input_schema_ref: str
    output_schema_ref: str
    scope_paths: tuple[str, ...]
    template_id: str = ""
    template_parameters: Mapping[str, Any] = field(default_factory=dict)
    program: TransformationProgram | None = None
    effect_class: EffectClass = EffectClass.OBSERVE
    resources: ToolResourceEnvelope = field(default_factory=ToolResourceEnvelope)
    fixtures: tuple[ToolFixture, ...] = ()
    repair_template_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(self, "tool_id", _identifier(self.tool_id, "tool_id"))
        if not isinstance(self.input_schema_ref, str) or not str(self.input_schema_ref).strip():
            _refuse(ToolReason.SCHEMA_MISSING, "generated tools require closed input and output schemas")
        if not isinstance(self.output_schema_ref, str) or not str(self.output_schema_ref).strip():
            _refuse(ToolReason.SCHEMA_MISSING, "generated tools require closed input and output schemas")
        object.__setattr__(
            self, "input_schema_ref", _identifier(self.input_schema_ref, "input_schema_ref")
        )
        object.__setattr__(
            self, "output_schema_ref", _identifier(self.output_schema_ref, "output_schema_ref")
        )
        object.__setattr__(
            self,
            "scope_paths",
            _strings(self.scope_paths, "scope_paths", paths=True, required=True, limit=MAX_SCOPE_PATHS),
        )
        object.__setattr__(
            self, "template_id", _identifier(self.template_id, "template_id", required=False)
        )
        object.__setattr__(
            self,
            "template_parameters",
            _literal(self.template_parameters, "template_parameters")
            if self.template_parameters is not None
            else MappingProxyType({}),
        )
        program = self.program
        if program is not None and not isinstance(program, TransformationProgram):
            program = TransformationProgram.from_record(program)
        object.__setattr__(self, "program", program)
        object.__setattr__(
            self, "effect_class", _enum(self.effect_class, EffectClass, "effect_class")
        )
        if self.effect_class not in ALLOWED_EFFECT_CLASSES:
            _refuse(ToolReason.EFFECT_FORBIDDEN, "generated tools may only observe or validate")
        object.__setattr__(
            self,
            "resources",
            ToolResourceEnvelope.from_record(self.resources),
        )
        object.__setattr__(self, "fixtures", _fixtures(self.fixtures))
        object.__setattr__(
            self,
            "repair_template_ids",
            _strings(self.repair_template_ids, "repair_template_ids", identifiers=True),
        )
        unknown = [item for item in self.repair_template_ids if item not in APPROVED_REPAIR_TEMPLATE_IDS]
        if unknown:
            _refuse(ToolReason.UNKNOWN_TEMPLATE, "request referenced an unapproved repair template")
        if not self.input_schema_ref or not self.output_schema_ref:
            _refuse(ToolReason.SCHEMA_MISSING, "generated tools require closed input and output schemas")


def _artifact_fields(cls: type[Any]) -> tuple[str, ...]:
    return tuple(
        name
        for name in cls.__dataclass_fields__  # type: ignore[attr-defined]
        if name != "SCHEMA"
    )


@dataclass(frozen=True)
class GeneratedToolSpec(CanonicalContract):
    """Closed generated-tool schema: operations, effects, paths, and resources."""

    SCHEMA: ClassVar[str] = _schema_name("GeneratedToolSpec")

    bindings: ArtifactBindings
    tool_id: str
    input_schema_ref: str
    output_schema_ref: str
    program: TransformationProgram
    optimized_program: TransformationProgram
    template_id: str = ""
    effect_class: EffectClass = EffectClass.OBSERVE
    scope_paths: tuple[str, ...] = ()
    resources: ToolResourceEnvelope = field(default_factory=ToolResourceEnvelope)
    grammar_revision: str = GRAMMAR_REVISION
    compiler_revision: str = COMPILER_REVISION
    dsl_revision: str = DSL_REVISION
    repair_template_ids: tuple[str, ...] = ()
    state: ArtifactState = ArtifactState.CANDIDATE
    can_authorize: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        for name in ("tool_id", "input_schema_ref", "output_schema_ref"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self, "program", TransformationProgram.from_record(self.program)
        )
        object.__setattr__(
            self,
            "optimized_program",
            TransformationProgram.from_record(self.optimized_program),
        )
        object.__setattr__(
            self, "template_id", _identifier(self.template_id, "template_id", required=False)
        )
        object.__setattr__(
            self, "effect_class", _enum(self.effect_class, EffectClass, "effect_class")
        )
        if self.effect_class not in ALLOWED_EFFECT_CLASSES:
            _refuse(ToolReason.EFFECT_FORBIDDEN, "generated tool effect class is forbidden")
        object.__setattr__(
            self,
            "scope_paths",
            _strings(self.scope_paths, "scope_paths", paths=True, required=True, limit=MAX_SCOPE_PATHS),
        )
        object.__setattr__(self, "resources", ToolResourceEnvelope.from_record(self.resources))
        for name, expected in (
            ("grammar_revision", GRAMMAR_REVISION),
            ("compiler_revision", COMPILER_REVISION),
            ("dsl_revision", DSL_REVISION),
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
            if getattr(self, name) != expected:
                raise ToolSynthesisError(f"{name} is not current")
        object.__setattr__(
            self,
            "repair_template_ids",
            _strings(self.repair_template_ids, "repair_template_ids", identifiers=True),
        )
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if self.state is ArtifactState.PROMOTED:
            _refuse(ToolReason.PROMOTION_FORBIDDEN, "tool specs remain candidate until certified invocation")
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        if self.can_authorize:
            _refuse(ToolReason.AUTHORITY_REJECTED, "generated tool specs cannot authorize")
        if not self.input_schema_ref or not self.output_schema_ref:
            _refuse(ToolReason.SCHEMA_MISSING, "generated tools require closed schemas")
        _bounded(self, "GeneratedToolSpec")

    @property
    def can_grant_authority(self) -> bool:
        return False

    @property
    def can_promote(self) -> bool:
        return False

    @property
    def can_skip_validation(self) -> bool:
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "tool_id": self.tool_id,
            "input_schema_ref": self.input_schema_ref,
            "output_schema_ref": self.output_schema_ref,
            "program": self.program.to_record(),
            "optimized_program": self.optimized_program.to_record(),
            "template_id": self.template_id,
            "effect_class": self.effect_class.value,
            "scope_paths": self.scope_paths,
            "resources": self.resources.to_record(),
            "grammar_revision": self.grammar_revision,
            "compiler_revision": self.compiler_revision,
            "dsl_revision": self.dsl_revision,
            "repair_template_ids": self.repair_template_ids,
            "state": self.state.value,
            "can_authorize": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GeneratedToolSpec:
        if not isinstance(payload, Mapping):
            raise ToolSynthesisError("GeneratedToolSpec payload must be a mapping")
        body = _unwrap_generic_envelope(payload, cls.SCHEMA)
        record = cls(**_decode_fields(body, cls.SCHEMA, _artifact_fields(cls), cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class GeneratedToolCandidate(CanonicalContract):
    """Interpreted or optimized tool.  Candidate until certified promotion."""

    SCHEMA: ClassVar[str] = _schema_name("GeneratedToolCandidate")

    bindings: ArtifactBindings
    tool_id: str
    spec_cid: str
    program: TransformationProgram
    representation: str = "dsl"
    optimized_translation: str = ""
    translation_digest: str = ""
    fixture_cids: tuple[str, ...] = ()
    state: ArtifactState = ArtifactState.CANDIDATE
    validated: bool = False
    certified: bool = False
    can_authorize: bool = False
    compiler_revision: str = COMPILER_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(self, "tool_id", _identifier(self.tool_id, "tool_id"))
        object.__setattr__(self, "spec_cid", _identifier(self.spec_cid, "spec_cid"))
        object.__setattr__(self, "program", TransformationProgram.from_record(self.program))
        representation = _identifier(self.representation, "representation")
        if representation not in {"dsl", "optimized-python"}:
            _refuse(ToolReason.GRAMMAR_VIOLATION, "candidate representation is outside the closed set")
        object.__setattr__(self, "representation", representation)
        translation = _text(self.optimized_translation, "optimized_translation", required=False)
        if translation:
            _assert_restricted_ast(translation)
        object.__setattr__(self, "optimized_translation", translation)
        object.__setattr__(
            self,
            "translation_digest",
            _identifier(self.translation_digest, "translation_digest", required=False),
        )
        if translation and self.translation_digest:
            expected = content_identity({"optimized_translation": translation})
            if self.translation_digest != expected:
                _refuse(ToolReason.TRANSLATION_MISMATCH, "optimized translation digest does not match")
        object.__setattr__(
            self, "fixture_cids", _strings(self.fixture_cids, "fixture_cids", identifiers=True)
        )
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        object.__setattr__(self, "validated", _bool(self.validated, "validated"))
        object.__setattr__(self, "certified", _bool(self.certified, "certified"))
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        if self.can_authorize:
            _refuse(ToolReason.AUTHORITY_REJECTED, "generated tool candidates cannot authorize")
        if self.state is ArtifactState.PROMOTED:
            if self.representation != "optimized-python":
                _refuse(ToolReason.PROMOTION_FORBIDDEN, "only optimized Python may be promoted")
            if not self.validated or not self.certified:
                _refuse(
                    ToolReason.PROMOTION_FORBIDDEN,
                    "optimized Python is promoted only after differential validation and certificate",
                )
        elif self.state is not ArtifactState.CANDIDATE and self.state is not ArtifactState.VERIFIED:
            _refuse(ToolReason.CANDIDATE_TIER_REQUIRED, "generated tools remain candidate-tier")
        if self.representation == "dsl" and self.state is ArtifactState.PROMOTED:
            _refuse(ToolReason.PROMOTION_FORBIDDEN, "DSL tools remain candidates")
        object.__setattr__(
            self, "compiler_revision", _identifier(self.compiler_revision, "compiler_revision")
        )
        if self.compiler_revision != COMPILER_REVISION:
            raise ToolSynthesisError("compiler revision is not current")
        _bounded(self, "GeneratedToolCandidate")

    @property
    def can_grant_authority(self) -> bool:
        return False

    @property
    def can_promote(self) -> bool:
        return False

    @property
    def can_skip_validation(self) -> bool:
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "tool_id": self.tool_id,
            "spec_cid": self.spec_cid,
            "program": self.program.to_record(),
            "representation": self.representation,
            "optimized_translation": self.optimized_translation,
            "translation_digest": self.translation_digest,
            "fixture_cids": self.fixture_cids,
            "state": self.state.value,
            "validated": self.validated,
            "certified": self.certified,
            "can_authorize": False,
            "compiler_revision": self.compiler_revision,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GeneratedToolCandidate:
        if not isinstance(payload, Mapping):
            raise ToolSynthesisError("GeneratedToolCandidate payload must be a mapping")
        body = _unwrap_generic_envelope(payload, cls.SCHEMA)
        record = cls(**_decode_fields(body, cls.SCHEMA, _artifact_fields(cls), cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class TranslationReceipt(CanonicalContract):
    """Exact DSL-versus-optimized differential result.  Never a promotion grant."""

    SCHEMA: ClassVar[str] = _schema_name("TranslationReceipt")

    bindings: ArtifactBindings
    spec_cid: str
    candidate_cid: str
    optimized_cid: str
    status: TranslationStatus
    reason_code: ToolReason
    matched_fixture_ids: tuple[str, ...] = ()
    mismatched_fixture_ids: tuple[str, ...] = ()
    adversarial_fixture_ids: tuple[str, ...] = ()
    validator_revision: str = VALIDATOR_REVISION
    exact: bool = False
    state: ArtifactState = ArtifactState.CANDIDATE
    can_authorize: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        for name in ("spec_cid", "candidate_cid", "optimized_cid"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "status", _enum(self.status, TranslationStatus, "status"))
        object.__setattr__(self, "reason_code", _enum(self.reason_code, ToolReason, "reason_code"))
        object.__setattr__(
            self,
            "matched_fixture_ids",
            _strings(self.matched_fixture_ids, "matched_fixture_ids", identifiers=True),
        )
        object.__setattr__(
            self,
            "mismatched_fixture_ids",
            _strings(self.mismatched_fixture_ids, "mismatched_fixture_ids", identifiers=True),
        )
        object.__setattr__(
            self,
            "adversarial_fixture_ids",
            _strings(self.adversarial_fixture_ids, "adversarial_fixture_ids", identifiers=True),
        )
        object.__setattr__(
            self, "validator_revision", _identifier(self.validator_revision, "validator_revision")
        )
        if self.validator_revision != VALIDATOR_REVISION:
            raise ToolSynthesisError("translation validator revision is not current")
        object.__setattr__(self, "exact", _bool(self.exact, "exact"))
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        if self.can_authorize:
            _refuse(ToolReason.AUTHORITY_REJECTED, "translation receipts cannot authorize")
        if self.status is TranslationStatus.ACCEPTED:
            if not self.exact or self.mismatched_fixture_ids or self.reason_code is not ToolReason.ACCEPTED:
                _refuse(ToolReason.TRANSLATION_MISMATCH, "accepted translation must be exact")
            if self.state is not ArtifactState.VERIFIED:
                raise ToolSynthesisError("accepted translation receipts are verified, not promoted")
        else:
            if self.exact:
                raise ToolSynthesisError("rejected translation cannot claim exact equivalence")
            if self.state is ArtifactState.PROMOTED:
                _refuse(ToolReason.PROMOTION_FORBIDDEN, "rejected translations cannot be promoted")
        _bounded(self, "TranslationReceipt")

    @property
    def accepted(self) -> bool:
        return self.status is TranslationStatus.ACCEPTED and self.exact

    @property
    def can_promote(self) -> bool:
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "spec_cid": self.spec_cid,
            "candidate_cid": self.candidate_cid,
            "optimized_cid": self.optimized_cid,
            "status": self.status.value,
            "reason_code": self.reason_code.value,
            "matched_fixture_ids": self.matched_fixture_ids,
            "mismatched_fixture_ids": self.mismatched_fixture_ids,
            "adversarial_fixture_ids": self.adversarial_fixture_ids,
            "validator_revision": self.validator_revision,
            "exact": self.exact,
            "state": self.state.value,
            "can_authorize": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TranslationReceipt:
        if not isinstance(payload, Mapping):
            raise ToolSynthesisError("TranslationReceipt payload must be a mapping")
        record = cls(**_decode_fields(payload, cls.SCHEMA, _artifact_fields(cls), cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class GeneratedToolCertificate(CanonicalContract):
    """Independent tool certificate.  Identity is not authority or promotion."""

    SCHEMA: ClassVar[str] = _schema_name("GeneratedToolCertificate")

    bindings: ArtifactBindings
    tool_id: str
    spec_cid: str
    candidate_cid: str
    optimized_cid: str
    translation_cid: str
    template_id: str
    grammar_revision: str
    compiler_revision: str
    validator_revision: str
    input_schema_ref: str
    output_schema_ref: str
    effect_class: EffectClass
    scope_paths: tuple[str, ...]
    test_fixture_cids: tuple[str, ...]
    adversarial_fixture_cids: tuple[str, ...]
    translation_digest: str
    resource_max_steps: int
    issued_at_ms: int = 0
    state: ArtifactState = ArtifactState.VERIFIED
    can_authorize: bool = False
    accepted: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        for name in (
            "tool_id",
            "spec_cid",
            "candidate_cid",
            "optimized_cid",
            "translation_cid",
            "grammar_revision",
            "compiler_revision",
            "validator_revision",
            "input_schema_ref",
            "output_schema_ref",
            "translation_digest",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self, "template_id", _identifier(self.template_id, "template_id", required=False)
        )
        object.__setattr__(
            self, "effect_class", _enum(self.effect_class, EffectClass, "effect_class")
        )
        object.__setattr__(
            self,
            "scope_paths",
            _strings(self.scope_paths, "scope_paths", paths=True, required=True, limit=MAX_SCOPE_PATHS),
        )
        object.__setattr__(
            self,
            "test_fixture_cids",
            _strings(self.test_fixture_cids, "test_fixture_cids", identifiers=True, required=True),
        )
        object.__setattr__(
            self,
            "adversarial_fixture_cids",
            _strings(
                self.adversarial_fixture_cids,
                "adversarial_fixture_cids",
                identifiers=True,
                required=True,
            ),
        )
        object.__setattr__(
            self,
            "resource_max_steps",
            _positive_int(self.resource_max_steps, "resource_max_steps", maximum=MAX_TOOL_STEPS),
        )
        object.__setattr__(self, "issued_at_ms", _nonnegative_int(self.issued_at_ms, "issued_at_ms"))
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        object.__setattr__(self, "accepted", _bool(self.accepted, "accepted"))
        if self.can_authorize:
            _refuse(ToolReason.AUTHORITY_REJECTED, "tool certificates cannot authorize")
        if self.state is ArtifactState.PROMOTED:
            _refuse(ToolReason.PROMOTION_FORBIDDEN, "certificates never promote")
        if self.accepted and self.state is not ArtifactState.VERIFIED:
            raise ToolSynthesisError("accepted tool certificates are verified, not promoted")
        if not self.accepted:
            object.__setattr__(self, "state", ArtifactState.REJECTED)
        if self.grammar_revision != GRAMMAR_REVISION:
            raise ToolSynthesisError("certificate grammar revision is not current")
        if self.compiler_revision != COMPILER_REVISION:
            raise ToolSynthesisError("certificate compiler revision is not current")
        if self.validator_revision != VALIDATOR_REVISION:
            raise ToolSynthesisError("certificate validator revision is not current")
        _bounded(self, "GeneratedToolCertificate")

    @property
    def grants_authority(self) -> bool:
        return False

    @property
    def grants_promotion(self) -> bool:
        return False

    @property
    def can_promote(self) -> bool:
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "tool_id": self.tool_id,
            "spec_cid": self.spec_cid,
            "candidate_cid": self.candidate_cid,
            "optimized_cid": self.optimized_cid,
            "translation_cid": self.translation_cid,
            "template_id": self.template_id,
            "grammar_revision": self.grammar_revision,
            "compiler_revision": self.compiler_revision,
            "validator_revision": self.validator_revision,
            "input_schema_ref": self.input_schema_ref,
            "output_schema_ref": self.output_schema_ref,
            "effect_class": self.effect_class.value,
            "scope_paths": self.scope_paths,
            "test_fixture_cids": self.test_fixture_cids,
            "adversarial_fixture_cids": self.adversarial_fixture_cids,
            "translation_digest": self.translation_digest,
            "resource_max_steps": self.resource_max_steps,
            "issued_at_ms": self.issued_at_ms,
            "state": self.state.value,
            "can_authorize": False,
            "accepted": self.accepted,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GeneratedToolCertificate:
        if not isinstance(payload, Mapping):
            raise ToolSynthesisError("GeneratedToolCertificate payload must be a mapping")
        body = _unwrap_generic_envelope(payload, cls.SCHEMA)
        record = cls(**_decode_fields(body, cls.SCHEMA, _artifact_fields(cls), cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class GeneratedToolInvocationReceipt(CanonicalContract):
    """Candidate-tier invocation observation.  Cannot authorize or complete."""

    SCHEMA: ClassVar[str] = _schema_name("GeneratedToolInvocationReceipt")

    bindings: ArtifactBindings
    tool_id: str
    spec_cid: str
    candidate_cid: str
    certificate_cid: str
    representation: str
    output_digest: str
    fixture_id: str = ""
    state: ArtifactState = ArtifactState.CANDIDATE
    can_authorize: bool = False
    can_establish_proof: bool = False
    can_establish_postcondition: bool = False
    can_establish_completion: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        for name in ("tool_id", "spec_cid", "candidate_cid", "certificate_cid", "output_digest"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self, "representation", _identifier(self.representation, "representation")
        )
        object.__setattr__(
            self, "fixture_id", _identifier(self.fixture_id, "fixture_id", required=False)
        )
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        for name in (
            "can_authorize",
            "can_establish_proof",
            "can_establish_postcondition",
            "can_establish_completion",
        ):
            object.__setattr__(self, name, _bool(getattr(self, name), name))
            if getattr(self, name):
                _refuse(ToolReason.AUTHORITY_REJECTED, "invocation receipts cannot discharge authority")
        if self.state is ArtifactState.PROMOTED:
            _refuse(ToolReason.PROMOTION_FORBIDDEN, "invocation receipts remain observations")
        _bounded(self, "GeneratedToolInvocationReceipt")

    @property
    def can_grant_authority(self) -> bool:
        return False

    @property
    def can_promote(self) -> bool:
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "tool_id": self.tool_id,
            "spec_cid": self.spec_cid,
            "candidate_cid": self.candidate_cid,
            "certificate_cid": self.certificate_cid,
            "representation": self.representation,
            "output_digest": self.output_digest,
            "fixture_id": self.fixture_id,
            "state": ArtifactState.CANDIDATE.value,
            "can_authorize": False,
            "can_establish_proof": False,
            "can_establish_postcondition": False,
            "can_establish_completion": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GeneratedToolInvocationReceipt:
        if not isinstance(payload, Mapping):
            raise ToolSynthesisError("GeneratedToolInvocationReceipt payload must be a mapping")
        body = _unwrap_generic_envelope(payload, cls.SCHEMA)
        record = cls(**_decode_fields(body, cls.SCHEMA, _artifact_fields(cls), cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class SynthesisResult:
    """Candidate tools produced from one bounded synthesis.  Never promoted."""

    spec: GeneratedToolSpec
    candidate: GeneratedToolCandidate
    optimized: GeneratedToolCandidate
    fixtures: tuple[ToolFixture, ...]


@dataclass(frozen=True)
class _EvalOutcome:
    ok: bool
    value: Any = None
    reason: ToolReason | None = None


def _run_closed(
    runner: Any,
    value: Any,
) -> _EvalOutcome:
    try:
        result = runner(value)
    except ToolSynthesisError as exc:
        return _EvalOutcome(False, reason=exc.reason_code)
    except ProcedureSafetyError as exc:
        message = str(exc).lower()
        if "path" in message or "filesystem" in message:
            return _EvalOutcome(False, reason=ToolReason.PATH_ESCAPE)
        return _EvalOutcome(False, reason=ToolReason.INJECTION_REJECTED)
    except ProcedureBoundsError:
        return _EvalOutcome(False, reason=ToolReason.RESOURCE_EXCEEDED)
    except ProcedureContractError:
        return _EvalOutcome(False, reason=ToolReason.SCHEMA_MISMATCH)
    return _EvalOutcome(True, value=result)


def _same_outcome(left: _EvalOutcome, right: _EvalOutcome) -> bool:
    if left.ok != right.ok:
        return False
    if not left.ok:
        return left.reason is right.reason
    return canonical_json_bytes(_jsonable(left.value)) == canonical_json_bytes(_jsonable(right.value))


class TranslationValidator:
    """Require exact DSL/optimized/Python equivalence on happy and adversarial fixtures."""

    revision: ClassVar[str] = VALIDATOR_REVISION

    def __init__(self, dsl: TransformationDsl | None = None) -> None:
        self.dsl = dsl if dsl is not None else TransformationDsl()

    def validate(
        self,
        spec: GeneratedToolSpec,
        candidate: GeneratedToolCandidate,
        optimized: GeneratedToolCandidate,
        fixtures: Sequence[ToolFixture],
    ) -> TranslationReceipt:
        if spec.bindings != candidate.bindings or spec.bindings != optimized.bindings:
            _refuse(ToolReason.BINDING_MISMATCH, "translation artifacts do not share exact bindings")
        if candidate.spec_cid != spec.content_id or optimized.spec_cid != spec.content_id:
            _refuse(ToolReason.BINDING_MISMATCH, "candidates are not bound to the generated spec")
        if candidate.representation != "dsl" or optimized.representation != "optimized-python":
            _refuse(ToolReason.GRAMMAR_VIOLATION, "translation requires a DSL candidate and optimized Python")
        if not optimized.optimized_translation:
            _refuse(ToolReason.TRANSLATION_MISMATCH, "optimized candidate omitted its translation")
        records = _fixtures(fixtures)
        if not any(item.kind is FixtureKind.HAPPY for item in records):
            _refuse(ToolReason.MISSING_TESTS, "translation validation requires a happy-path fixture")
        adversarial = tuple(item.fixture_id for item in records if item.kind is FixtureKind.ADVERSARIAL)
        if not adversarial:
            _refuse(ToolReason.ADVERSARIAL_FAILURE, "translation validation requires adversarial fixtures")

        matched: list[str] = []
        mismatched: list[str] = []

        def dsl_run(value: Any) -> Any:
            return self.dsl.interpret(
                spec.program,
                value,
                scope_paths=spec.scope_paths,
                resources=spec.resources,
            )

        def fused_run(value: Any) -> Any:
            return self.dsl.interpret(
                spec.optimized_program,
                value,
                scope_paths=spec.scope_paths,
                resources=spec.resources,
            )

        def python_run(value: Any) -> Any:
            return self.dsl.interpret_python(
                optimized.optimized_translation,
                value,
                scope_paths=spec.scope_paths,
                resources=spec.resources,
            )

        for fixture in records:
            dsl_outcome = _run_closed(dsl_run, fixture.payload)
            fused_outcome = _run_closed(fused_run, fixture.payload)
            python_outcome = _run_closed(python_run, fixture.payload)
            if not (_same_outcome(dsl_outcome, fused_outcome) and _same_outcome(dsl_outcome, python_outcome)):
                mismatched.append(fixture.fixture_id)
                continue
            if fixture.expected_refusal:
                expected = _enum(fixture.expected_refusal, ToolReason, "expected_refusal")
                if dsl_outcome.ok or dsl_outcome.reason is not expected:
                    mismatched.append(fixture.fixture_id)
                    continue
            matched.append(fixture.fixture_id)

        exact = not mismatched
        status = TranslationStatus.ACCEPTED if exact else TranslationStatus.REJECTED
        reason = ToolReason.ACCEPTED if exact else ToolReason.TRANSLATION_MISMATCH
        return TranslationReceipt(
            bindings=spec.bindings,
            spec_cid=spec.content_id,
            candidate_cid=candidate.content_id,
            optimized_cid=optimized.content_id,
            status=status,
            reason_code=reason,
            matched_fixture_ids=tuple(matched),
            mismatched_fixture_ids=tuple(mismatched),
            adversarial_fixture_ids=adversarial,
            exact=exact,
            state=ArtifactState.VERIFIED if exact else ArtifactState.REJECTED,
        )


class GeneratedToolCompiler:
    """Compile reviewed DSL programs into candidate tools.  Promotion is gated."""

    revision: ClassVar[str] = COMPILER_REVISION

    def __init__(
        self,
        dsl: TransformationDsl | None = None,
        validator: TranslationValidator | None = None,
    ) -> None:
        self.dsl = dsl if dsl is not None else TransformationDsl()
        self.validator = validator if validator is not None else TranslationValidator(self.dsl)

    def compile_program(self, request: ToolSynthesisRequest) -> TransformationProgram:
        if request.template_id:
            template = self.dsl.library.get(request.template_id)
            if template.requires_scope and not request.scope_paths:
                _refuse(ToolReason.MISSING_SCOPE, "template requires declared path scope")
            if request.repair_template_ids:
                allowed = set(template.repair_template_ids)
                if any(item not in allowed for item in request.repair_template_ids):
                    _refuse(ToolReason.UNKNOWN_TEMPLATE, "repair template is not bound to this transformation")
            return template.instantiate(request.template_parameters)
        if request.program is None:
            _refuse(ToolReason.GRAMMAR_VIOLATION, "synthesis requires a template or closed program")
        return self.dsl.parse_program(request.program)

    def synthesize(self, request: ToolSynthesisRequest) -> SynthesisResult:
        if not isinstance(request, ToolSynthesisRequest):
            raise ToolGrammarError("compiler requires a ToolSynthesisRequest", ToolReason.GRAMMAR_VIOLATION)
        program = self.compile_program(request)
        if len(program.steps) > request.resources.max_steps:
            _refuse(ToolReason.RESOURCE_EXCEEDED, "program exceeds the declared step bound")
        optimized_program = self.dsl.optimize(program)
        translation = self.dsl.emit_optimized_python(program)
        digest = content_identity({"optimized_translation": translation})
        happy = [item for item in request.fixtures if item.kind is FixtureKind.HAPPY]
        if not happy:
            first_op = program.steps[0].operation
            mapping_payload = {
                "path": f"{request.scope_paths[0]}/module.py",
                "kind": "keep",
                "id": "symbol-a",
            }
            if first_op in {
                TransformationOp.FILTER_EQUALS,
                TransformationOp.FILTER_IN,
                TransformationOp.MAP_ITEMS,
                TransformationOp.SORT_BY,
                TransformationOp.LIMIT,
                TransformationOp.SORT_LIMIT,
                TransformationOp.JOIN_IDENTIFIERS,
            }:
                payload: Any = (mapping_payload,)
            else:
                payload = mapping_payload
            happy = [
                ToolFixture(
                    fixture_id="happy.default",
                    kind=FixtureKind.HAPPY,
                    payload=payload,
                )
            ]
        fixtures = tuple(happy) + tuple(
            item for item in request.fixtures if item.kind is not FixtureKind.HAPPY
        )
        adversarial = [item for item in fixtures if item.kind is FixtureKind.ADVERSARIAL]
        if not adversarial:
            fixtures = fixtures + default_adversarial_fixtures(scope_paths=request.scope_paths)
        if not any(item.kind is FixtureKind.HAPPY for item in fixtures):
            _refuse(ToolReason.MISSING_TESTS, "generated tools require tests")
        spec = GeneratedToolSpec(
            bindings=request.bindings,
            tool_id=request.tool_id,
            input_schema_ref=request.input_schema_ref,
            output_schema_ref=request.output_schema_ref,
            program=program,
            optimized_program=optimized_program,
            template_id=request.template_id,
            effect_class=request.effect_class,
            scope_paths=request.scope_paths,
            resources=request.resources,
            repair_template_ids=request.repair_template_ids,
        )
        fixture_cids = tuple(item.content_id for item in fixtures)
        candidate = GeneratedToolCandidate(
            bindings=request.bindings,
            tool_id=request.tool_id,
            spec_cid=spec.content_id,
            program=program,
            representation="dsl",
            fixture_cids=fixture_cids,
        )
        optimized = GeneratedToolCandidate(
            bindings=request.bindings,
            tool_id=request.tool_id,
            spec_cid=spec.content_id,
            program=optimized_program,
            representation="optimized-python",
            optimized_translation=translation,
            translation_digest=digest,
            fixture_cids=fixture_cids,
        )
        if candidate.state is not ArtifactState.CANDIDATE or optimized.state is not ArtifactState.CANDIDATE:
            _refuse(ToolReason.CANDIDATE_TIER_REQUIRED, "synthesis may yield candidate tools only")
        return SynthesisResult(spec=spec, candidate=candidate, optimized=optimized, fixtures=fixtures)

    def certify(
        self,
        result: SynthesisResult,
        receipt: TranslationReceipt,
        *,
        issued_at_ms: int = 0,
    ) -> GeneratedToolCertificate:
        if not receipt.accepted:
            _refuse(ToolReason.CERTIFICATE_REJECTED, "certificate requires exact differential validation")
        if receipt.spec_cid != result.spec.content_id:
            _refuse(ToolReason.BINDING_MISMATCH, "translation receipt is not bound to the spec")
        if receipt.candidate_cid != result.candidate.content_id:
            _refuse(ToolReason.BINDING_MISMATCH, "translation receipt is not bound to the DSL candidate")
        if receipt.optimized_cid != result.optimized.content_id:
            _refuse(ToolReason.BINDING_MISMATCH, "translation receipt is not bound to the optimized candidate")
        tests = tuple(item.content_id for item in result.fixtures if item.kind is FixtureKind.HAPPY)
        adversarial = tuple(
            item.content_id for item in result.fixtures if item.kind is FixtureKind.ADVERSARIAL
        )
        if not tests or not adversarial:
            _refuse(ToolReason.MISSING_TESTS, "certificate requires tests and adversarial fixtures")
        return GeneratedToolCertificate(
            bindings=result.spec.bindings,
            tool_id=result.spec.tool_id,
            spec_cid=result.spec.content_id,
            candidate_cid=result.candidate.content_id,
            optimized_cid=result.optimized.content_id,
            translation_cid=receipt.content_id,
            template_id=result.spec.template_id,
            grammar_revision=GRAMMAR_REVISION,
            compiler_revision=COMPILER_REVISION,
            validator_revision=VALIDATOR_REVISION,
            input_schema_ref=result.spec.input_schema_ref,
            output_schema_ref=result.spec.output_schema_ref,
            effect_class=result.spec.effect_class,
            scope_paths=result.spec.scope_paths,
            test_fixture_cids=tests,
            adversarial_fixture_cids=adversarial,
            translation_digest=result.optimized.translation_digest,
            resource_max_steps=result.spec.resources.max_steps,
            issued_at_ms=issued_at_ms,
        )

    def promote_optimized(
        self,
        optimized: GeneratedToolCandidate,
        receipt: TranslationReceipt,
        certificate: GeneratedToolCertificate,
    ) -> GeneratedToolCandidate:
        if not isinstance(certificate, GeneratedToolCertificate):
            _refuse(ToolReason.MISSING_CERTIFICATE, "optimized Python is promoted only after certificate")
        if not isinstance(receipt, TranslationReceipt):
            _refuse(ToolReason.TRANSLATION_MISMATCH, "optimized Python is promoted only after exact differential validation")
        if optimized.representation != "optimized-python":
            _refuse(ToolReason.PROMOTION_FORBIDDEN, "only optimized Python may be promoted")
        if not receipt.accepted:
            _refuse(
                ToolReason.TRANSLATION_MISMATCH,
                "optimized Python is promoted only after exact differential validation",
            )
        if not certificate.accepted or certificate.state is not ArtifactState.VERIFIED:
            _refuse(ToolReason.CERTIFICATE_REJECTED, "optimized Python is promoted only after certificate")
        if certificate.optimized_cid != optimized.content_id:
            _refuse(ToolReason.BINDING_MISMATCH, "certificate is not bound to the optimized candidate")
        if certificate.translation_cid != receipt.content_id:
            _refuse(ToolReason.BINDING_MISMATCH, "certificate is not bound to the translation receipt")
        if certificate.translation_digest != optimized.translation_digest:
            _refuse(ToolReason.BINDING_MISMATCH, "certificate is not bound to the optimized translation")
        if optimized.state is ArtifactState.PROMOTED:
            return optimized
        return GeneratedToolCandidate(
            bindings=optimized.bindings,
            tool_id=optimized.tool_id,
            spec_cid=optimized.spec_cid,
            program=optimized.program,
            representation=optimized.representation,
            optimized_translation=optimized.optimized_translation,
            translation_digest=optimized.translation_digest,
            fixture_cids=optimized.fixture_cids,
            state=ArtifactState.PROMOTED,
            validated=True,
            certified=True,
        )

    def invoke(
        self,
        spec: GeneratedToolSpec,
        candidate: GeneratedToolCandidate,
        certificate: GeneratedToolCertificate,
        payload: Any,
        *,
        fixture_id: str = "",
        use_optimized: bool = False,
    ) -> GeneratedToolInvocationReceipt:
        if not isinstance(certificate, GeneratedToolCertificate):
            _refuse(ToolReason.MISSING_CERTIFICATE, "invocation requires a bound tool certificate")
        if certificate.spec_cid != spec.content_id or certificate.candidate_cid == "":
            _refuse(ToolReason.MISSING_CERTIFICATE, "invocation requires a bound tool certificate")
        if not certificate.accepted:
            _refuse(ToolReason.CERTIFICATE_REJECTED, "invocation requires an accepted certificate")
        if spec.bindings != candidate.bindings or spec.bindings != certificate.bindings:
            _refuse(ToolReason.BINDING_MISMATCH, "invocation artifacts do not share exact bindings")
        if use_optimized:
            if candidate.state is not ArtifactState.PROMOTED or candidate.representation != "optimized-python":
                _refuse(
                    ToolReason.PROMOTION_FORBIDDEN,
                    "optimized Python may be invoked only after promotion",
                )
            if candidate.spec_cid != certificate.spec_cid:
                _refuse(ToolReason.BINDING_MISMATCH, "certificate is not bound to the promoted translation")
            if candidate.translation_digest != certificate.translation_digest:
                _refuse(ToolReason.BINDING_MISMATCH, "certificate is not bound to the promoted translation")
            output = self.dsl.interpret_python(
                candidate.optimized_translation,
                payload,
                scope_paths=spec.scope_paths,
                resources=spec.resources,
            )
            representation = "optimized-python"
        else:
            if certificate.candidate_cid != candidate.content_id:
                _refuse(ToolReason.BINDING_MISMATCH, "certificate is not bound to the DSL candidate")
            output = self.dsl.interpret(
                spec.program,
                payload,
                scope_paths=spec.scope_paths,
                resources=spec.resources,
            )
            representation = "dsl"
        digest = content_identity(_jsonable(output))
        return GeneratedToolInvocationReceipt(
            bindings=spec.bindings,
            tool_id=spec.tool_id,
            spec_cid=spec.content_id,
            candidate_cid=candidate.content_id,
            certificate_cid=certificate.content_id,
            representation=representation,
            output_digest=digest,
            fixture_id=fixture_id,
        )


for _artifact_type in (
    GeneratedToolSpec,
    GeneratedToolCandidate,
    GeneratedToolCertificate,
    GeneratedToolInvocationReceipt,
    TranslationReceipt,
):
    ARTIFACT_TYPES_BY_SCHEMA[_artifact_type.SCHEMA] = _artifact_type


__all__ = [
    "ALLOWED_EFFECT_CLASSES",
    "APPROVED_REPAIR_TEMPLATE_IDS",
    "COMPILER_REVISION",
    "DETERMINISTIC_TOOL_DSL_REVISION",
    "DSL_REVISION",
    "FORBIDDEN_TRANSFORMATION_OPS",
    "GRAMMAR_REVISION",
    "MAX_TOOL_ITEMS",
    "MAX_TOOL_STEPS",
    "TEMPLATE_LIBRARY_REVISION",
    "VALIDATOR_REVISION",
    "DeterministicToolDsl",
    "FixtureKind",
    "GeneratedToolCandidate",
    "GeneratedToolCertificate",
    "GeneratedToolCompiler",
    "GeneratedToolInvocationReceipt",
    "GeneratedToolSpec",
    "SynthesisResult",
    "TemplateLibrary",
    "ToolAction",
    "ToolFixture",
    "ToolGrammarError",
    "ToolPromotionError",
    "ToolReason",
    "ToolResourceEnvelope",
    "ToolSafetyError",
    "ToolSynthesisError",
    "ToolSynthesisRequest",
    "ToolTranslationError",
    "TransformationDsl",
    "TransformationOp",
    "TransformationProgram",
    "TransformationStep",
    "TransformationTemplate",
    "TranslationReceipt",
    "TranslationStatus",
    "TranslationValidator",
    "default_adversarial_fixtures",
]
