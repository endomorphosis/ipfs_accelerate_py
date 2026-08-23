"""Bounded deterministic tool synthesis from a reviewed transformation DSL.

Repeated pure/bounded transformations may become interpreted DSL tools drawn
from a reviewed grammar and template library.  This module owns synthesis, not
authority: generated tools remain candidates.  Optimized Python is a fused
translation of the same closed opcodes and stays candidate-tier until exact
differential validation and an independently issued certificate.  Arbitrary
scripts, shell, network, and filesystem-path synthesis are refused.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field as data_field
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Final, NoReturn

from ..proof.formal_verification_contracts import CanonicalContract, canonical_json_bytes
from .contracts import (
    ARTIFACT_TYPES_BY_SCHEMA,
    FORBIDDEN_STEP_OPERATIONS,
    MAX_ITEMS,
    MAX_MAPPING_ITEMS,
    MAX_NESTING,
    MAX_RECORD_BYTES,
    PROCEDURE_CONTRACT_VERSION,
    ArtifactBindings,
    ArtifactState,
    EffectClass,
    ProcedureContractError,
    ProcedureSafetyError,
    _bounded,
    _decode_fields,
    _enum,
    _enums,
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
COMPILER_REVISION: Final[str] = "GeneratedToolCompiler@1"
VALIDATOR_REVISION: Final[str] = "TranslationValidator@1"
GRAMMAR_ID: Final[str] = "deterministic-transformation-dsl@1"
CERTIFICATE_ISSUER: Final[str] = "translation-validator@1"
MAX_TOOL_STEPS: Final[int] = 16
MAX_TOOL_OUTPUT_BYTES: Final[int] = 65_536
MAX_PATH_DEPTH: Final[int] = 8
MAX_TEMPLATE_EXPANSION: Final[int] = 32
MAX_TRACES: Final[int] = 32
MAX_FIXTURES: Final[int] = 32

ALLOWED_EFFECT_CLASSES: Final[frozenset[EffectClass]] = frozenset({EffectClass.OBSERVE})
ALLOWED_SEPARATORS: Final[frozenset[str]] = frozenset({".", ":", "/", "-", "_", "@", "+"})
APPROVED_REPAIR_TEMPLATE_IDS: Final[tuple[str, ...]] = (
    "repair.replace-import-path",
    "repair.normalize-identifier",
    "repair.add-missing-test-name",
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
_FORBIDDEN_OPCODE_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "import",
        "shell",
        "subprocess",
        "system",
        "popen",
        "network",
        "http",
        "callback",
        "arbitrary-shell",
        "arbitrary-python",
        "arbitrary-network",
        "arbitrary-filesystem",
        "arbitrary_shell",
        "arbitrary_python",
        "arbitrary_network_request",
        "arbitrary_filesystem_path",
        "python",
        "bash",
        "sh",
    }
) | frozenset(item.lower().replace("_", "-") for item in FORBIDDEN_STEP_OPERATIONS)
_CODE_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "python_source",
        "source_code",
        "code_body",
        "shell_command",
        "executable",
        "callback",
        "callable",
        "policy_code",
        "command",
    }
)
_SHELL_MARKERS: Final[frozenset[str]] = frozenset(
    {
        "shell",
        "bash",
        "sh",
        "powershell",
        "cmd",
        "subprocess",
        "popen",
        "system",
    }
)
_REQUIRED_OPCODE_PARAMS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "project": ("fields",),
        "drop": ("fields",),
        "rename": ("mapping",),
        "copy": (),
        "require": ("fields",),
        "map-enum": ("mapping",),
        "set-literal": ("value",),
        "lowercase": (),
        "join": ("fields", "separator", "target"),
        "split": ("separator", "targets"),
        "sort-keys": (),
        "relative-path-prefix": ("prefix",),
        "coalesce": ("fields", "target"),
        "integer-clamp": ("minimum", "maximum"),
        "filter-equals": ("key", "value"),
        "apply-template": ("template_id",),
        "select-approved-repair-template": ("allowed_template_ids",),
    }
)


class ToolSynthesisError(ProcedureContractError):
    """A generated-tool grammar, bound, translation, or promotion is unsafe."""

    def __init__(self, message: str, reason_code: "ToolSynthesisReason" | None = None) -> None:
        super().__init__(message)
        self.reason_code = reason_code or ToolSynthesisReason.GRAMMAR_BOUND


class ToolGrammarError(ToolSynthesisError):
    """The program is outside the reviewed grammar or template library."""


class ToolBoundError(ToolSynthesisError):
    """A path, resource, schema, or output bound was exceeded."""


class ToolSafetyError(ToolSynthesisError):
    """Arbitrary code, shell, network, or filesystem synthesis was attempted."""


class ToolTranslationError(ToolSynthesisError):
    """DSL and optimized translations are not exactly equivalent."""


class ToolPromotionError(ToolSynthesisError):
    """Optimized Python cannot be promoted without exact validation and certificate."""


class ToolSynthesisReason(str, Enum):
    CANDIDATE_SYNTHESIZED = "candidate-synthesized"
    UNKNOWN_OPCODE = "unknown-opcode"
    UNKNOWN_TEMPLATE = "unknown-template"
    ARBITRARY_CODE = "arbitrary-code"
    ARBITRARY_SHELL = "arbitrary-shell"
    PATH_ESCAPE = "path-escape"
    EFFECT_ESCALATION = "effect-escalation"
    SCHEMA_MISMATCH = "schema-mismatch"
    RESOURCE_EXCEEDED = "resource-exceeded"
    TRANSLATION_MISMATCH = "translation-mismatch"
    CERTIFICATE_REQUIRED = "certificate-required"
    CERTIFICATE_MISMATCH = "certificate-mismatch"
    DSL_PROMOTION_FORBIDDEN = "dsl-promotion-forbidden"
    ENUM_CLOSED = "enum-closed"
    MISSING_FIELD = "missing-field"
    NO_TEMPLATE_MATCH = "no-template-match"
    ADVERSARIAL_REJECTED = "adversarial-rejected"
    GRAMMAR_BOUND = "grammar-bound"
    NETWORK_FORBIDDEN = "network-forbidden"
    PROMOTION_FORBIDDEN = "promotion-forbidden"
    EQUIVALENT = "equivalent"
    FIXTURE_FAILED = "fixture-failed"


class TransformationOpcode(str, Enum):
    PROJECT = "project"
    DROP = "drop"
    RENAME = "rename"
    COPY = "copy"
    REQUIRE = "require"
    MAP_ENUM = "map-enum"
    SET_LITERAL = "set-literal"
    LOWERCASE = "lowercase"
    JOIN = "join"
    SPLIT = "split"
    SORT_KEYS = "sort-keys"
    RELATIVE_PATH_PREFIX = "relative-path-prefix"
    COALESCE = "coalesce"
    INTEGER_CLAMP = "integer-clamp"
    FILTER_EQUALS = "filter-equals"
    APPLY_TEMPLATE = "apply-template"
    SELECT_APPROVED_REPAIR_TEMPLATE = "select-approved-repair-template"


class ToolRepresentation(str, Enum):
    INTERPRETED_DSL = "interpreted-dsl"
    OPTIMIZED_PYTHON = "optimized-python"


class FixtureKind(str, Enum):
    TEST = "test"
    ADVERSARIAL = "adversarial"


ALLOWED_OPCODES: Final[frozenset[str]] = frozenset(item.value for item in TransformationOpcode)


def _bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise ToolSynthesisError(f"{field_name} must be a boolean")
    return value


def _bindings(value: Any) -> ArtifactBindings:
    return _nested(value, ArtifactBindings, "bindings")


def _refuse(reason: ToolSynthesisReason, message: str, error: type[ToolSynthesisError] = ToolSynthesisError) -> NoReturn:
    raise error(message, reason)


def _normalized_marker(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")


def _marker_hit(value: str, markers: frozenset[str]) -> bool:
    normalized = _normalized_marker(value)
    return normalized in markers or any(marker in normalized for marker in markers)


def _scan_forbidden_code(value: Any) -> ToolSynthesisReason | None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                return ToolSynthesisReason.ARBITRARY_CODE
            if _unsafe_key(raw_key) or _marker_hit(raw_key, _CODE_MARKERS):
                if _marker_hit(raw_key, _SHELL_MARKERS):
                    return ToolSynthesisReason.ARBITRARY_SHELL
                return ToolSynthesisReason.ARBITRARY_CODE
            if _marker_hit(raw_key, _SHELL_MARKERS):
                return ToolSynthesisReason.ARBITRARY_SHELL
            nested = _scan_forbidden_code(item)
            if nested is not None:
                return nested
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        for item in value:
            nested = _scan_forbidden_code(item)
            if nested is not None:
                return nested
        return None
    if isinstance(value, str):
        lowered = value.lower()
        if any(marker in lowered for marker in ("import os", "subprocess", "os.system", "/bin/sh")):
            return ToolSynthesisReason.ARBITRARY_SHELL
        if any(marker in lowered for marker in ("eval(", "exec(", "__import__")):
            return ToolSynthesisReason.ARBITRARY_CODE
    return None


def _reject_code(value: Any, field_name: str) -> None:
    reason = _scan_forbidden_code(value)
    if reason is ToolSynthesisReason.ARBITRARY_SHELL:
        _refuse(reason, f"{field_name} contains arbitrary shell", ToolSafetyError)
    if reason is ToolSynthesisReason.ARBITRARY_CODE:
        _refuse(reason, f"{field_name} contains arbitrary code", ToolSafetyError)


def _payload_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    raw = value if value is not None else {}
    _reject_code(raw, field_name)
    try:
        frozen = _freeze(raw, field_name)
    except ProcedureSafetyError as exc:
        message = str(exc)
        if "path" in message.lower() or "filesystem" in message.lower():
            _refuse(ToolSynthesisReason.PATH_ESCAPE, message, ToolSafetyError)
        _refuse(ToolSynthesisReason.ARBITRARY_CODE, message, ToolSafetyError)
    if not isinstance(frozen, Mapping):
        raise ToolSynthesisError(f"{field_name} must be a mapping")
    return frozen


def _digest(value: Mapping[str, Any] | Sequence[Any]) -> str:
    if isinstance(value, Mapping):
        payload: Any = dict(value)
    else:
        payload = list(value)
    digest = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return f"sha256:{digest}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_jsonable(item) for item in value)
    return value


def _path_depth(path: str) -> int:
    if path in {"", "."}:
        return 0
    return len(path.split("/"))


def _path_in_scope(path: str, prefixes: Sequence[str]) -> bool:
    for prefix in prefixes:
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return False


def _unwrap_generic_envelope(payload: Mapping[str, Any], schema: str) -> Mapping[str, Any]:
    body = dict(payload)
    keys = set(body).difference({"schema", "contract_version", "content_id", "cid"})
    if keys and keys <= _GENERIC_ENVELOPE_FIELDS and "facts" in body:
        facts = body.get("facts")
        if not isinstance(facts, Mapping):
            raise ToolSynthesisError("generic generated-tool facts must be a mapping")
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


def _nested_record(value: Any, cls: type[Any], field_name: str) -> Any:
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        return cls.from_record(value)
    raise ToolSynthesisError(f"{field_name} must be {cls.__name__}")


@dataclass(frozen=True)
class ToolResourceEnvelope:
    """Closed integer resource ceiling for one generated tool."""

    max_steps: int = MAX_TOOL_STEPS
    max_output_bytes: int = MAX_TOOL_OUTPUT_BYTES
    max_items: int = MAX_ITEMS
    max_nesting: int = MAX_NESTING
    max_path_depth: int = MAX_PATH_DEPTH
    subprocess_limit: int = 0
    network_request_limit: int = 0
    model_call_limit: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_steps",
            _positive_int(self.max_steps, "max_steps", maximum=MAX_TOOL_STEPS),
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
            "max_items",
            _positive_int(self.max_items, "max_items", maximum=MAX_ITEMS),
        )
        object.__setattr__(
            self,
            "max_nesting",
            _positive_int(self.max_nesting, "max_nesting", maximum=MAX_NESTING),
        )
        object.__setattr__(
            self,
            "max_path_depth",
            _positive_int(self.max_path_depth, "max_path_depth", maximum=MAX_PATH_DEPTH),
        )
        for name in ("subprocess_limit", "network_request_limit", "model_call_limit"):
            object.__setattr__(
                self, name, _nonnegative_int(getattr(self, name), name, maximum=0)
            )

    def to_record(self) -> dict[str, int]:
        return {
            "max_steps": self.max_steps,
            "max_output_bytes": self.max_output_bytes,
            "max_items": self.max_items,
            "max_nesting": self.max_nesting,
            "max_path_depth": self.max_path_depth,
            "subprocess_limit": self.subprocess_limit,
            "network_request_limit": self.network_request_limit,
            "model_call_limit": self.model_call_limit,
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any] | None) -> ToolResourceEnvelope:
        if payload is None:
            return cls()
        if not isinstance(payload, Mapping):
            raise ToolSynthesisError("resources must be a mapping")
        return cls(
            max_steps=payload.get("max_steps", MAX_TOOL_STEPS),
            max_output_bytes=payload.get("max_output_bytes", MAX_TOOL_OUTPUT_BYTES),
            max_items=payload.get("max_items", MAX_ITEMS),
            max_nesting=payload.get("max_nesting", MAX_NESTING),
            max_path_depth=payload.get("max_path_depth", MAX_PATH_DEPTH),
            subprocess_limit=payload.get("subprocess_limit", 0),
            network_request_limit=payload.get("network_request_limit", 0),
            model_call_limit=payload.get("model_call_limit", 0),
        )


@dataclass(frozen=True)
class TransformationStep:
    """One closed, non-executable transformation from the reviewed grammar."""

    opcode: TransformationOpcode
    field: str = ""
    target: str = ""
    parameters: Mapping[str, Any] = data_field(default_factory=dict)

    def __post_init__(self) -> None:
        opcode_value = self.opcode.value if isinstance(self.opcode, TransformationOpcode) else self.opcode
        if not isinstance(opcode_value, str):
            _refuse(ToolSynthesisReason.UNKNOWN_OPCODE, "opcode must be a closed grammar token", ToolGrammarError)
        normalized = opcode_value.strip().lower().replace("_", "-")
        if normalized in _FORBIDDEN_OPCODE_MARKERS or normalized in {
            item.lower().replace("_", "-") for item in FORBIDDEN_STEP_OPERATIONS
        }:
            reason = (
                ToolSynthesisReason.ARBITRARY_SHELL
                if _marker_hit(normalized, _SHELL_MARKERS)
                else ToolSynthesisReason.ARBITRARY_CODE
            )
            _refuse(reason, "transformation opcode is arbitrary code or shell", ToolSafetyError)
        try:
            opcode = _enum(normalized, TransformationOpcode, "opcode")
        except ProcedureContractError:
            _refuse(
                ToolSynthesisReason.UNKNOWN_OPCODE,
                "transformation opcode is outside the reviewed grammar",
                ToolGrammarError,
            )
        object.__setattr__(self, "opcode", opcode)
        object.__setattr__(self, "field", _identifier(self.field, "field", required=False))
        object.__setattr__(self, "target", _identifier(self.target, "target", required=False))
        parameters = _payload_mapping(self.parameters, "parameters")
        required = _REQUIRED_OPCODE_PARAMS[opcode.value]
        missing = tuple(name for name in required if name not in parameters)
        if missing:
            _refuse(
                ToolSynthesisReason.GRAMMAR_BOUND,
                "transformation is missing required grammar parameters",
                ToolGrammarError,
            )
        if opcode is TransformationOpcode.COPY and not (self.field and (self.target or parameters.get("target"))):
            _refuse(
                ToolSynthesisReason.GRAMMAR_BOUND,
                "copy requires a source field and target",
                ToolGrammarError,
            )
        if opcode in {
            TransformationOpcode.LOWERCASE,
            TransformationOpcode.RELATIVE_PATH_PREFIX,
            TransformationOpcode.INTEGER_CLAMP,
            TransformationOpcode.FILTER_EQUALS,
            TransformationOpcode.SPLIT,
            TransformationOpcode.MAP_ENUM,
            TransformationOpcode.SET_LITERAL,
            TransformationOpcode.SELECT_APPROVED_REPAIR_TEMPLATE,
        } and not self.field:
            _refuse(
                ToolSynthesisReason.GRAMMAR_BOUND,
                f"{opcode.value} requires a field",
                ToolGrammarError,
            )
        object.__setattr__(self, "parameters", parameters)

    def to_record(self) -> dict[str, Any]:
        return {
            "opcode": self.opcode.value,
            "field": self.field,
            "target": self.target,
            "parameters": _jsonable(self.parameters),
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any] | TransformationStep) -> TransformationStep:
        if isinstance(payload, TransformationStep):
            return payload
        if not isinstance(payload, Mapping):
            raise ToolGrammarError("transformation step must be a mapping")
        return cls(
            opcode=payload.get("opcode", ""),
            field=payload.get("field", ""),
            target=payload.get("target", ""),
            parameters=payload.get("parameters", {}),
        )


def _steps(values: Any, *, limit: int = MAX_TOOL_STEPS) -> tuple[TransformationStep, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray, memoryview)):
        raw = values
    else:
        raise ToolGrammarError("transformations must be a sequence")
    if len(raw) > limit:
        _refuse(ToolSynthesisReason.RESOURCE_EXCEEDED, "transformations exceed the step bound", ToolBoundError)
    result = tuple(TransformationStep.from_record(item) for item in raw)
    if not result:
        _refuse(ToolSynthesisReason.GRAMMAR_BOUND, "a generated tool requires at least one transformation", ToolGrammarError)
    return result


@dataclass(frozen=True)
class ToolFixture:
    """Closed test or adversarial fixture.  Bodies stay compact and non-executable."""

    fixture_id: str
    kind: FixtureKind
    payload: Mapping[str, Any]
    expected: Mapping[str, Any] | None = None
    must_refuse: bool = False
    refusal_reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "fixture_id", _identifier(self.fixture_id, "fixture_id"))
        object.__setattr__(self, "kind", _enum(self.kind, FixtureKind, "kind"))
        if self.kind is FixtureKind.ADVERSARIAL:
            object.__setattr__(self, "must_refuse", True)
        else:
            object.__setattr__(self, "must_refuse", _bool(self.must_refuse, "must_refuse"))
        if self.must_refuse:
            object.__setattr__(self, "expected", None)
            object.__setattr__(
                self,
                "refusal_reason",
                _identifier(self.refusal_reason, "refusal_reason", required=False),
            )
            try:
                object.__setattr__(self, "payload", _payload_mapping(self.payload, "payload"))
            except (ProcedureContractError, ToolSynthesisError):
                frozen = _freeze_adversarial_payload(self.payload)
                object.__setattr__(self, "payload", frozen)
        else:
            object.__setattr__(self, "payload", _payload_mapping(self.payload, "payload"))
            if self.expected is None:
                _refuse(ToolSynthesisReason.FIXTURE_FAILED, "test fixtures require an expected output")
            object.__setattr__(self, "expected", _payload_mapping(self.expected, "expected"))
            object.__setattr__(self, "refusal_reason", "")

    def to_record(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "kind": self.kind.value,
            "payload": _jsonable(self.payload),
            "expected": None if self.expected is None else _jsonable(self.expected),
            "must_refuse": self.must_refuse,
            "refusal_reason": self.refusal_reason,
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any] | ToolFixture) -> ToolFixture:
        if isinstance(payload, ToolFixture):
            return payload
        if not isinstance(payload, Mapping):
            raise ToolSynthesisError("fixture must be a mapping")
        return cls(
            fixture_id=payload.get("fixture_id", ""),
            kind=payload.get("kind", FixtureKind.TEST),
            payload=payload.get("payload", {}),
            expected=payload.get("expected"),
            must_refuse=payload.get("must_refuse", False),
            refusal_reason=payload.get("refusal_reason", ""),
        )


def _freeze_adversarial_payload(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return MappingProxyType({"adversarial": True})
    compact: dict[str, Any] = {}
    for raw_key, item in list(value.items())[:MAX_MAPPING_ITEMS]:
        if not isinstance(raw_key, str):
            continue
        key = raw_key.strip()[:128] or "adversarial"
        if type(item) is bool or type(item) is int:
            compact[key] = item
        elif type(item) is str:
            compact[key] = item[:256]
        else:
            compact[key] = True
    if not compact:
        compact["adversarial"] = True
    return MappingProxyType(compact)


def _fixtures(values: Any) -> tuple[ToolFixture, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray, memoryview)):
        raw = values
    else:
        raise ToolSynthesisError("fixtures must be a sequence")
    if len(raw) > MAX_FIXTURES:
        _refuse(ToolSynthesisReason.RESOURCE_EXCEEDED, "fixtures exceed the fixture bound", ToolBoundError)
    result: list[ToolFixture] = []
    seen: set[str] = set()
    for item in raw:
        record = ToolFixture.from_record(item)
        if record.fixture_id in seen:
            raise ToolSynthesisError("fixtures contain a duplicate fixture_id")
        seen.add(record.fixture_id)
        result.append(record)
    return tuple(result)


@dataclass(frozen=True)
class ApprovedTemplate:
    """Reviewed grammar template.  Templates never contain executable source."""

    template_id: str
    input_schema_ref: str
    output_schema_ref: str
    transformations: tuple[TransformationStep, ...]
    path_prefixes: tuple[str, ...]
    effect_classes: tuple[EffectClass, ...] = (EffectClass.OBSERVE,)
    resources: ToolResourceEnvelope = data_field(default_factory=ToolResourceEnvelope)
    fixtures: tuple[ToolFixture, ...] = ()
    repair_template: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "template_id", _identifier(self.template_id, "template_id"))
        object.__setattr__(
            self, "input_schema_ref", _identifier(self.input_schema_ref, "input_schema_ref")
        )
        object.__setattr__(
            self, "output_schema_ref", _identifier(self.output_schema_ref, "output_schema_ref")
        )
        steps = _steps(self.transformations)
        if any(step.opcode is TransformationOpcode.APPLY_TEMPLATE for step in steps):
            _refuse(
                ToolSynthesisReason.GRAMMAR_BOUND,
                "approved templates cannot recursively apply templates",
                ToolGrammarError,
            )
        object.__setattr__(self, "transformations", steps)
        object.__setattr__(
            self,
            "path_prefixes",
            _strings(self.path_prefixes, "path_prefixes", paths=True, required=True),
        )
        effects = _enums(
            self.effect_classes,
            EffectClass,
            "effect_classes",
            limit=len(EffectClass),
            required=True,
        )
        if set(effects) - ALLOWED_EFFECT_CLASSES:
            _refuse(
                ToolSynthesisReason.EFFECT_ESCALATION,
                "generated tools may only declare observe effects",
                ToolSafetyError,
            )
        object.__setattr__(self, "effect_classes", effects)
        object.__setattr__(
            self, "resources", _nested_record(self.resources, ToolResourceEnvelope, "resources")
        )
        object.__setattr__(self, "fixtures", _fixtures(self.fixtures))
        object.__setattr__(self, "repair_template", _bool(self.repair_template, "repair_template"))


def _step(
    opcode: TransformationOpcode,
    *,
    field: str = "",
    target: str = "",
    **parameters: Any,
) -> TransformationStep:
    return TransformationStep(opcode=opcode, field=field, target=target, parameters=parameters)


def _test_fixture(fixture_id: str, payload: Mapping[str, Any], expected: Mapping[str, Any]) -> ToolFixture:
    return ToolFixture(
        fixture_id=fixture_id,
        kind=FixtureKind.TEST,
        payload=payload,
        expected=expected,
    )


def _adversarial_fixture(
    fixture_id: str,
    payload: Mapping[str, Any],
    refusal_reason: str,
) -> ToolFixture:
    return ToolFixture(
        fixture_id=fixture_id,
        kind=FixtureKind.ADVERSARIAL,
        payload=payload,
        must_refuse=True,
        refusal_reason=refusal_reason,
    )


def _approved_templates() -> tuple[ApprovedTemplate, ...]:
    scope = "ipfs_accelerate_py/agent_supervisor"
    status_map = {"pending": "candidate", "done": "accepted", "failed": "rejected"}
    repair_map = {item: item for item in APPROVED_REPAIR_TEMPLATE_IDS}
    return (
        ApprovedTemplate(
            template_id="normalize-identifier-record",
            input_schema_ref="schema.identifier-record.in",
            output_schema_ref="schema.identifier-record.out",
            path_prefixes=(scope,),
            transformations=(
                _step(TransformationOpcode.REQUIRE, fields=("record_id", "status")),
                _step(TransformationOpcode.LOWERCASE, field="record_id"),
                _step(TransformationOpcode.MAP_ENUM, field="status", mapping=status_map),
                _step(TransformationOpcode.SORT_KEYS),
            ),
            fixtures=(
                _test_fixture(
                    "normalize-identifier-record.ok",
                    {"record_id": "Receipt.One", "status": "pending", "extra": "drop-me"},
                    {"extra": "drop-me", "record_id": "receipt.one", "status": "candidate"},
                ),
                _adversarial_fixture(
                    "normalize-identifier-record.unknown-status",
                    {"record_id": "receipt.one", "status": "invented"},
                    ToolSynthesisReason.ENUM_CLOSED.value,
                ),
            ),
        ),
        ApprovedTemplate(
            template_id="project-receipt-fields",
            input_schema_ref="schema.receipt.in",
            output_schema_ref="schema.receipt.out",
            path_prefixes=(scope,),
            transformations=(
                _step(
                    TransformationOpcode.PROJECT,
                    fields=("receipt_id", "tree_id", "status", "producer"),
                ),
                _step(
                    TransformationOpcode.REQUIRE,
                    fields=("receipt_id", "tree_id", "status", "producer"),
                ),
                _step(TransformationOpcode.SORT_KEYS),
            ),
            fixtures=(
                _test_fixture(
                    "project-receipt-fields.ok",
                    {
                        "receipt_id": "receipt-1",
                        "tree_id": "tree-1",
                        "status": "accepted",
                        "producer": "tests@1",
                        "note": "omit",
                    },
                    {
                        "producer": "tests@1",
                        "receipt_id": "receipt-1",
                        "status": "accepted",
                        "tree_id": "tree-1",
                    },
                ),
                _adversarial_fixture(
                    "project-receipt-fields.missing",
                    {"receipt_id": "receipt-1", "status": "accepted"},
                    ToolSynthesisReason.MISSING_FIELD.value,
                ),
            ),
        ),
        ApprovedTemplate(
            template_id="map-closed-status",
            input_schema_ref="schema.status.in",
            output_schema_ref="schema.status.out",
            path_prefixes=(scope,),
            transformations=(
                _step(TransformationOpcode.REQUIRE, fields=("status",)),
                _step(TransformationOpcode.MAP_ENUM, field="status", mapping=status_map),
            ),
            fixtures=(
                _test_fixture(
                    "map-closed-status.ok",
                    {"status": "done", "other": "keep"},
                    {"other": "keep", "status": "accepted"},
                ),
                _adversarial_fixture(
                    "map-closed-status.unknown",
                    {"status": "promoted-by-model"},
                    ToolSynthesisReason.ENUM_CLOSED.value,
                ),
            ),
        ),
        ApprovedTemplate(
            template_id="scope-relative-path",
            input_schema_ref="schema.path.in",
            output_schema_ref="schema.path.out",
            path_prefixes=(scope,),
            transformations=(
                _step(TransformationOpcode.REQUIRE, fields=("path",)),
                _step(
                    TransformationOpcode.RELATIVE_PATH_PREFIX,
                    field="path",
                    prefix=scope,
                ),
            ),
            fixtures=(
                _test_fixture(
                    "scope-relative-path.ok",
                    {"path": "procedure_compiler/tool_synthesis.py"},
                    {"path": f"{scope}/procedure_compiler/tool_synthesis.py"},
                ),
                _adversarial_fixture(
                    "scope-relative-path.escape",
                    {"path": "../secrets"},
                    ToolSynthesisReason.PATH_ESCAPE.value,
                ),
            ),
        ),
        ApprovedTemplate(
            template_id="select-approved-repair-template",
            input_schema_ref="schema.repair-template.in",
            output_schema_ref="schema.repair-template.out",
            path_prefixes=(scope,),
            repair_template=True,
            transformations=(
                _step(TransformationOpcode.REQUIRE, fields=("template_id", "target_path")),
                _step(
                    TransformationOpcode.SELECT_APPROVED_REPAIR_TEMPLATE,
                    field="template_id",
                    allowed_template_ids=APPROVED_REPAIR_TEMPLATE_IDS,
                ),
                _step(
                    TransformationOpcode.RELATIVE_PATH_PREFIX,
                    field="target_path",
                    prefix=scope,
                ),
                _step(
                    TransformationOpcode.PROJECT,
                    fields=("template_id", "target_path"),
                ),
                _step(TransformationOpcode.SORT_KEYS),
            ),
            fixtures=(
                _test_fixture(
                    "select-approved-repair-template.ok",
                    {
                        "template_id": "repair.replace-import-path",
                        "target_path": "procedure_compiler/contracts.py",
                        "note": "omit",
                    },
                    {
                        "target_path": f"{scope}/procedure_compiler/contracts.py",
                        "template_id": "repair.replace-import-path",
                    },
                ),
                _adversarial_fixture(
                    "select-approved-repair-template.unknown",
                    {
                        "template_id": "repair.invented-shell",
                        "target_path": "procedure_compiler/contracts.py",
                    },
                    ToolSynthesisReason.UNKNOWN_TEMPLATE.value,
                ),
            ),
        ),
    )


APPROVED_TEMPLATE_LIBRARY: Final[Mapping[str, ApprovedTemplate]] = MappingProxyType(
    {item.template_id: item for item in _approved_templates()}
)


def _library(templates: Mapping[str, ApprovedTemplate] | None) -> Mapping[str, ApprovedTemplate]:
    if templates is None:
        return APPROVED_TEMPLATE_LIBRARY
    if not isinstance(templates, Mapping) or not templates:
        _refuse(ToolSynthesisReason.UNKNOWN_TEMPLATE, "template library must be a nonempty mapping", ToolGrammarError)
    frozen: dict[str, ApprovedTemplate] = {}
    for raw_key, item in templates.items():
        key = _identifier(raw_key, "template_id")
        if not isinstance(item, ApprovedTemplate):
            raise ToolGrammarError("template library values must be ApprovedTemplate")
        if item.template_id != key:
            raise ToolGrammarError("template library key must match template_id")
        frozen[key] = item
    return MappingProxyType(frozen)


class TransformationDsl:
    """Reviewed grammar, template expansion, and pure interpretation."""

    revision: ClassVar[str] = DSL_REVISION
    grammar_id: ClassVar[str] = GRAMMAR_ID

    def __init__(self, templates: Mapping[str, ApprovedTemplate] | None = None) -> None:
        self._templates = _library(templates)

    @property
    def templates(self) -> Mapping[str, ApprovedTemplate]:
        return self._templates

    def parse(self, transformations: Sequence[TransformationStep | Mapping[str, Any]]) -> tuple[TransformationStep, ...]:
        return _steps(transformations)

    def expand(self, transformations: Sequence[TransformationStep | Mapping[str, Any]]) -> tuple[TransformationStep, ...]:
        steps = self.parse(transformations)
        expanded: list[TransformationStep] = []
        for step in steps:
            if step.opcode is TransformationOpcode.APPLY_TEMPLATE:
                template_id = _identifier(step.parameters.get("template_id", ""), "template_id")
                template = self._templates.get(template_id)
                if template is None:
                    _refuse(
                        ToolSynthesisReason.UNKNOWN_TEMPLATE,
                        "apply-template referenced an unreviewed template",
                        ToolGrammarError,
                    )
                if len(expanded) + len(template.transformations) > MAX_TEMPLATE_EXPANSION:
                    _refuse(
                        ToolSynthesisReason.RESOURCE_EXCEEDED,
                        "template expansion exceeds the grammar bound",
                        ToolBoundError,
                    )
                expanded.extend(template.transformations)
                continue
            expanded.append(step)
        if any(item.opcode is TransformationOpcode.APPLY_TEMPLATE for item in expanded):
            _refuse(
                ToolSynthesisReason.GRAMMAR_BOUND,
                "template expansion left a nested apply-template",
                ToolGrammarError,
            )
        if len(expanded) > MAX_TEMPLATE_EXPANSION:
            _refuse(
                ToolSynthesisReason.RESOURCE_EXCEEDED,
                "expanded transformations exceed the grammar bound",
                ToolBoundError,
            )
        return tuple(expanded)

    def optimize(self, transformations: Sequence[TransformationStep | Mapping[str, Any]]) -> tuple[TransformationStep, ...]:
        expanded = list(self.expand(transformations))
        fused: list[TransformationStep] = []
        for step in expanded:
            if fused and fused[-1].opcode is step.opcode:
                previous = fused[-1]
                if step.opcode is TransformationOpcode.SORT_KEYS:
                    continue
                if step.opcode is TransformationOpcode.REQUIRE:
                    fused[-1] = TransformationStep(
                        opcode=step.opcode,
                        parameters={
                            "fields": tuple(
                                dict.fromkeys(
                                    (
                                        *tuple(previous.parameters.get("fields", ())),
                                        *tuple(step.parameters.get("fields", ())),
                                    )
                                )
                            )
                        },
                    )
                    continue
                if step.opcode is TransformationOpcode.DROP:
                    fused[-1] = TransformationStep(
                        opcode=step.opcode,
                        parameters={
                            "fields": tuple(
                                dict.fromkeys(
                                    (
                                        *tuple(previous.parameters.get("fields", ())),
                                        *tuple(step.parameters.get("fields", ())),
                                    )
                                )
                            )
                        },
                    )
                    continue
                if step.opcode is TransformationOpcode.PROJECT:
                    previous_fields = tuple(previous.parameters.get("fields", ()))
                    next_fields = tuple(
                        name for name in step.parameters.get("fields", ()) if name in previous_fields
                    )
                    fused[-1] = TransformationStep(
                        opcode=step.opcode,
                        parameters={"fields": next_fields},
                    )
                    continue
            fused.append(step)
        if not fused:
            _refuse(ToolSynthesisReason.GRAMMAR_BOUND, "optimized program is empty", ToolGrammarError)
        if len(fused) > MAX_TOOL_STEPS:
            _refuse(
                ToolSynthesisReason.RESOURCE_EXCEEDED,
                "optimized program exceeds the step bound",
                ToolBoundError,
            )
        return tuple(fused)

    def interpret(
        self,
        transformations: Sequence[TransformationStep | Mapping[str, Any]],
        payload: Mapping[str, Any],
        *,
        path_prefixes: Sequence[str],
        resources: ToolResourceEnvelope | Mapping[str, Any] | None = None,
        optimized: bool = False,
    ) -> Mapping[str, Any]:
        envelope = _nested_record(resources or ToolResourceEnvelope(), ToolResourceEnvelope, "resources")
        prefixes = _strings(path_prefixes, "path_prefixes", paths=True, required=True)
        program = self.optimize(transformations) if optimized else self.expand(transformations)
        if len(program) > envelope.max_steps:
            _refuse(
                ToolSynthesisReason.RESOURCE_EXCEEDED,
                "program exceeds the declared step bound",
                ToolBoundError,
            )
        try:
            current = _payload_mapping(payload, "payload")
        except ProcedureSafetyError as exc:
            message = str(exc)
            if "path" in message.lower() or "filesystem" in message.lower():
                _refuse(ToolSynthesisReason.PATH_ESCAPE, message, ToolSafetyError)
            if "secret" in message.lower() or "executable" in message.lower():
                reason = (
                    ToolSynthesisReason.ARBITRARY_SHELL
                    if any(marker in message.lower() for marker in _SHELL_MARKERS)
                    else ToolSynthesisReason.ARBITRARY_CODE
                )
                _refuse(reason, message, ToolSafetyError)
            raise ToolSafetyError(message, ToolSynthesisReason.ADVERSARIAL_REJECTED) from exc
        for step in program:
            current = self._apply_step(step, current, path_prefixes=prefixes, resources=envelope)
            self._enforce_output_bounds(current, envelope)
        return current

    def _enforce_output_bounds(self, payload: Mapping[str, Any], resources: ToolResourceEnvelope) -> None:
        if len(payload) > resources.max_items:
            _refuse(
                ToolSynthesisReason.RESOURCE_EXCEEDED,
                "tool output exceeds its item bound",
                ToolBoundError,
            )
        encoded = canonical_json_bytes(_jsonable(payload))
        if len(encoded) > resources.max_output_bytes or len(encoded) > MAX_RECORD_BYTES:
            _refuse(
                ToolSynthesisReason.RESOURCE_EXCEEDED,
                "tool output exceeds its byte bound",
                ToolBoundError,
            )

    def _apply_step(
        self,
        step: TransformationStep,
        payload: Mapping[str, Any],
        *,
        path_prefixes: Sequence[str],
        resources: ToolResourceEnvelope,
    ) -> Mapping[str, Any]:
        data = dict(payload)
        opcode = step.opcode
        params = step.parameters
        if opcode is TransformationOpcode.PROJECT:
            fields = _strings(params.get("fields"), "fields", identifiers=True, required=True)
            return MappingProxyType({key: data[key] for key in fields if key in data})
        if opcode is TransformationOpcode.DROP:
            fields = set(_strings(params.get("fields"), "fields", identifiers=True, required=True))
            return MappingProxyType({key: value for key, value in data.items() if key not in fields})
        if opcode is TransformationOpcode.RENAME:
            mapping = _payload_mapping(params.get("mapping"), "mapping")
            renamed: dict[str, Any] = {}
            consumed: set[str] = set()
            for source, destination in mapping.items():
                src = _identifier(source, "rename.source")
                dst = _identifier(destination, "rename.target")
                if src not in data:
                    _refuse(ToolSynthesisReason.MISSING_FIELD, "rename source field is missing")
                renamed[dst] = data[src]
                consumed.add(src)
            for key, value in data.items():
                if key not in consumed and key not in renamed:
                    renamed[key] = value
            return MappingProxyType(renamed)
        if opcode is TransformationOpcode.COPY:
            target = step.target or _identifier(params.get("target", ""), "target")
            source = step.field or _identifier(params.get("field", ""), "field")
            if source not in data:
                _refuse(ToolSynthesisReason.MISSING_FIELD, "copy source field is missing")
            copied = dict(data)
            copied[target] = data[source]
            return MappingProxyType(copied)
        if opcode is TransformationOpcode.REQUIRE:
            fields = _strings(params.get("fields"), "fields", identifiers=True, required=True)
            missing = tuple(name for name in fields if name not in data)
            if missing:
                _refuse(ToolSynthesisReason.MISSING_FIELD, "required field is missing")
            return payload
        if opcode is TransformationOpcode.MAP_ENUM:
            mapping = _payload_mapping(params.get("mapping"), "mapping")
            closed = {
                _identifier(source, "enum.source"): _identifier(destination, "enum.target")
                for source, destination in mapping.items()
            }
            current = data.get(step.field)
            if type(current) is not str or current not in closed:
                _refuse(ToolSynthesisReason.ENUM_CLOSED, "value is outside the closed enum map")
            updated = dict(data)
            updated[step.field] = closed[current]
            return MappingProxyType(updated)
        if opcode is TransformationOpcode.SET_LITERAL:
            value = params.get("value")
            if type(value) is bool:
                literal: Any = value
            elif type(value) is int:
                literal = _nonnegative_int(value, "value")
            else:
                literal = _identifier(value, "value")
            updated = dict(data)
            updated[step.field] = literal
            return MappingProxyType(updated)
        if opcode is TransformationOpcode.LOWERCASE:
            current = data.get(step.field)
            if type(current) is not str:
                _refuse(ToolSynthesisReason.SCHEMA_MISMATCH, "lowercase requires a string field")
            updated = dict(data)
            updated[step.field] = _identifier(current.lower(), step.field)
            return MappingProxyType(updated)
        if opcode is TransformationOpcode.JOIN:
            fields = _strings(params.get("fields"), "fields", identifiers=True, required=True)
            separator = _text(params.get("separator", ""), "separator")
            if separator not in ALLOWED_SEPARATORS:
                _refuse(ToolSynthesisReason.GRAMMAR_BOUND, "join separator is outside the closed set")
            parts: list[str] = []
            for name in fields:
                item = data.get(name)
                if type(item) is not str:
                    _refuse(ToolSynthesisReason.MISSING_FIELD, "join field is missing")
                parts.append(_identifier(item, name))
            target = _identifier(params.get("target", ""), "target")
            updated = dict(data)
            updated[target] = _identifier(separator.join(parts), target)
            return MappingProxyType(updated)
        if opcode is TransformationOpcode.SPLIT:
            separator = _text(params.get("separator", ""), "separator")
            if separator not in ALLOWED_SEPARATORS:
                _refuse(ToolSynthesisReason.GRAMMAR_BOUND, "split separator is outside the closed set")
            targets = _strings(params.get("targets"), "targets", identifiers=True, required=True)
            current = data.get(step.field)
            if type(current) is not str:
                _refuse(ToolSynthesisReason.SCHEMA_MISMATCH, "split requires a string field")
            parts = current.split(separator)
            if len(parts) != len(targets):
                _refuse(ToolSynthesisReason.SCHEMA_MISMATCH, "split arity does not match targets")
            updated = dict(data)
            for name, item in zip(targets, parts):
                updated[name] = _identifier(item, name)
            return MappingProxyType(updated)
        if opcode is TransformationOpcode.SORT_KEYS:
            return MappingProxyType({key: data[key] for key in sorted(data)})
        if opcode is TransformationOpcode.RELATIVE_PATH_PREFIX:
            prefix = _relative_path(params.get("prefix", ""), "prefix")
            current = data.get(step.field)
            if type(current) is not str:
                _refuse(ToolSynthesisReason.SCHEMA_MISMATCH, "relative-path-prefix requires a path field")
            try:
                relative = _relative_path(current, step.field)
            except ProcedureSafetyError:
                _refuse(ToolSynthesisReason.PATH_ESCAPE, "path is not repository-relative", ToolSafetyError)
            if relative == prefix or relative.startswith(prefix + "/"):
                joined = relative
            else:
                joined = _relative_path(f"{prefix}/{relative}", step.field)
            if _path_depth(joined) > resources.max_path_depth:
                _refuse(ToolSynthesisReason.RESOURCE_EXCEEDED, "path exceeds its depth bound", ToolBoundError)
            if not _path_in_scope(joined, path_prefixes):
                _refuse(ToolSynthesisReason.PATH_ESCAPE, "path escapes the declared tool scope", ToolSafetyError)
            updated = dict(data)
            updated[step.field] = joined
            return MappingProxyType(updated)
        if opcode is TransformationOpcode.COALESCE:
            fields = _strings(params.get("fields"), "fields", identifiers=True, required=True)
            target = _identifier(params.get("target", ""), "target")
            selected: Any = None
            found = False
            for name in fields:
                if name in data and data[name] not in (None, ""):
                    selected = data[name]
                    found = True
                    break
            if not found:
                _refuse(ToolSynthesisReason.MISSING_FIELD, "coalesce found no present field")
            updated = dict(data)
            updated[target] = selected
            return MappingProxyType(updated)
        if opcode is TransformationOpcode.INTEGER_CLAMP:
            minimum = _nonnegative_int(params.get("minimum", 0), "minimum")
            maximum = _nonnegative_int(params.get("maximum", 0), "maximum")
            if maximum < minimum:
                _refuse(ToolSynthesisReason.GRAMMAR_BOUND, "integer-clamp maximum is below minimum")
            current = data.get(step.field)
            if type(current) is not int or isinstance(current, bool):
                _refuse(ToolSynthesisReason.SCHEMA_MISMATCH, "integer-clamp requires an integer field")
            clamped = minimum if current < minimum else maximum if current > maximum else current
            updated = dict(data)
            updated[step.field] = clamped
            return MappingProxyType(updated)
        if opcode is TransformationOpcode.FILTER_EQUALS:
            key = _identifier(params.get("key", ""), "key")
            expected = params.get("value")
            current = data.get(step.field)
            if not isinstance(current, Sequence) or isinstance(current, (str, bytes, bytearray, memoryview)):
                _refuse(ToolSynthesisReason.SCHEMA_MISMATCH, "filter-equals requires a record sequence")
            if len(current) > resources.max_items:
                _refuse(ToolSynthesisReason.RESOURCE_EXCEEDED, "filter sequence exceeds its item bound", ToolBoundError)
            kept: list[Any] = []
            for item in current:
                if not isinstance(item, Mapping):
                    _refuse(ToolSynthesisReason.SCHEMA_MISMATCH, "filter-equals records must be mappings")
                if item.get(key) == expected:
                    kept.append(_payload_mapping(item, step.field))
            updated = dict(data)
            updated[step.field] = tuple(kept)
            return MappingProxyType(updated)
        if opcode is TransformationOpcode.SELECT_APPROVED_REPAIR_TEMPLATE:
            allowed = _strings(
                params.get("allowed_template_ids", APPROVED_REPAIR_TEMPLATE_IDS),
                "allowed_template_ids",
                identifiers=True,
                required=True,
            )
            current = data.get(step.field)
            if type(current) is not str or current not in allowed:
                _refuse(
                    ToolSynthesisReason.UNKNOWN_TEMPLATE,
                    "repair template is outside the approved library",
                    ToolGrammarError,
                )
            if current not in APPROVED_REPAIR_TEMPLATE_IDS:
                _refuse(
                    ToolSynthesisReason.UNKNOWN_TEMPLATE,
                    "repair template is not an approved repair template",
                    ToolGrammarError,
                )
            return payload
        _refuse(ToolSynthesisReason.UNKNOWN_OPCODE, "opcode is not interpretable", ToolGrammarError)


def _effects(values: Any) -> tuple[EffectClass, ...]:
    effects = _enums(values, EffectClass, "effect_classes", limit=len(EffectClass), required=True)
    extra = tuple(item for item in effects if item not in ALLOWED_EFFECT_CLASSES)
    if extra:
        _refuse(
            ToolSynthesisReason.EFFECT_ESCALATION,
            "generated tools cannot escalate effects",
            ToolSafetyError,
        )
    return effects


@dataclass(frozen=True)
class GeneratedToolSpec(CanonicalContract):
    """Declarative generated-tool program.  Synthesis leaves it candidate-tier."""

    SCHEMA: ClassVar[str] = _schema_name("GeneratedToolSpec")

    bindings: ArtifactBindings
    tool_id: str
    input_schema_ref: str
    output_schema_ref: str
    transformations: tuple[TransformationStep, ...]
    path_prefixes: tuple[str, ...]
    effect_classes: tuple[EffectClass, ...] = (EffectClass.OBSERVE,)
    template_id: str = ""
    grammar_id: str = GRAMMAR_ID
    resources: ToolResourceEnvelope = data_field(default_factory=ToolResourceEnvelope)
    test_fixture_ids: tuple[str, ...] = ()
    adversarial_fixture_ids: tuple[str, ...] = ()
    fixtures: tuple[ToolFixture, ...] = ()
    state: ArtifactState = ArtifactState.CANDIDATE
    dsl_revision: str = DSL_REVISION
    can_authorize: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(self, "tool_id", _identifier(self.tool_id, "tool_id"))
        object.__setattr__(
            self, "input_schema_ref", _identifier(self.input_schema_ref, "input_schema_ref")
        )
        object.__setattr__(
            self, "output_schema_ref", _identifier(self.output_schema_ref, "output_schema_ref")
        )
        object.__setattr__(self, "transformations", _steps(self.transformations))
        object.__setattr__(
            self,
            "path_prefixes",
            _strings(self.path_prefixes, "path_prefixes", paths=True, required=True),
        )
        object.__setattr__(self, "effect_classes", _effects(self.effect_classes))
        object.__setattr__(
            self, "template_id", _identifier(self.template_id, "template_id", required=False)
        )
        object.__setattr__(self, "grammar_id", _identifier(self.grammar_id, "grammar_id"))
        if self.grammar_id != GRAMMAR_ID:
            _refuse(ToolSynthesisReason.GRAMMAR_BOUND, "generated tool grammar is not current", ToolGrammarError)
        object.__setattr__(
            self, "resources", _nested_record(self.resources, ToolResourceEnvelope, "resources")
        )
        object.__setattr__(self, "fixtures", _fixtures(self.fixtures))
        derived_tests = tuple(item.fixture_id for item in self.fixtures if item.kind is FixtureKind.TEST)
        derived_adv = tuple(item.fixture_id for item in self.fixtures if item.kind is FixtureKind.ADVERSARIAL)
        object.__setattr__(
            self,
            "test_fixture_ids",
            _strings(self.test_fixture_ids or derived_tests, "test_fixture_ids", identifiers=True),
        )
        object.__setattr__(
            self,
            "adversarial_fixture_ids",
            _strings(
                self.adversarial_fixture_ids or derived_adv,
                "adversarial_fixture_ids",
                identifiers=True,
            ),
        )
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if self.state is not ArtifactState.CANDIDATE:
            _refuse(
                ToolSynthesisReason.PROMOTION_FORBIDDEN,
                "generated tool specs remain candidate-tier",
                ToolPromotionError,
            )
        object.__setattr__(self, "dsl_revision", _identifier(self.dsl_revision, "dsl_revision"))
        if self.dsl_revision != DSL_REVISION:
            raise ToolSynthesisError("transformation DSL revision is not current")
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        if self.can_authorize:
            _refuse(
                ToolSynthesisReason.PROMOTION_FORBIDDEN,
                "generated tool specs cannot authorize",
                ToolSafetyError,
            )
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
            "transformations": tuple(item.to_record() for item in self.transformations),
            "path_prefixes": self.path_prefixes,
            "effect_classes": tuple(item.value for item in self.effect_classes),
            "template_id": self.template_id,
            "grammar_id": self.grammar_id,
            "resources": self.resources.to_record(),
            "test_fixture_ids": self.test_fixture_ids,
            "adversarial_fixture_ids": self.adversarial_fixture_ids,
            "fixtures": tuple(item.to_record() for item in self.fixtures),
            "state": ArtifactState.CANDIDATE.value,
            "dsl_revision": DSL_REVISION,
            "can_authorize": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GeneratedToolSpec:
        if not isinstance(payload, Mapping):
            raise ToolSynthesisError("GeneratedToolSpec payload must be a mapping")
        body = _unwrap_generic_envelope(payload, cls.SCHEMA)
        fields = (
            "bindings",
            "tool_id",
            "input_schema_ref",
            "output_schema_ref",
            "transformations",
            "path_prefixes",
            "effect_classes",
            "template_id",
            "grammar_id",
            "resources",
            "test_fixture_ids",
            "adversarial_fixture_ids",
            "fixtures",
            "state",
            "dsl_revision",
            "can_authorize",
        )
        record = cls(**_decode_fields(body, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class GeneratedToolCandidate(CanonicalContract):
    """Compiled tool.  Interpreted DSL never leaves candidate tier."""

    SCHEMA: ClassVar[str] = _schema_name("GeneratedToolCandidate")

    bindings: ArtifactBindings
    tool_id: str
    spec_cid: str
    representation: ToolRepresentation
    transformations: tuple[TransformationStep, ...]
    input_schema_ref: str
    output_schema_ref: str
    path_prefixes: tuple[str, ...]
    effect_classes: tuple[EffectClass, ...] = (EffectClass.OBSERVE,)
    template_id: str = ""
    grammar_id: str = GRAMMAR_ID
    resources: ToolResourceEnvelope = data_field(default_factory=ToolResourceEnvelope)
    test_fixture_ids: tuple[str, ...] = ()
    adversarial_fixture_ids: tuple[str, ...] = ()
    fixtures: tuple[ToolFixture, ...] = ()
    state: ArtifactState = ArtifactState.CANDIDATE
    compiler_revision: str = COMPILER_REVISION
    predecessor_cid: str = ""
    certificate_cid: str = ""
    translation_receipt_cid: str = ""
    can_authorize: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(self, "tool_id", _identifier(self.tool_id, "tool_id"))
        object.__setattr__(self, "spec_cid", _identifier(self.spec_cid, "spec_cid"))
        object.__setattr__(
            self,
            "representation",
            _enum(self.representation, ToolRepresentation, "representation"),
        )
        object.__setattr__(self, "transformations", _steps(self.transformations))
        object.__setattr__(
            self, "input_schema_ref", _identifier(self.input_schema_ref, "input_schema_ref")
        )
        object.__setattr__(
            self, "output_schema_ref", _identifier(self.output_schema_ref, "output_schema_ref")
        )
        object.__setattr__(
            self,
            "path_prefixes",
            _strings(self.path_prefixes, "path_prefixes", paths=True, required=True),
        )
        object.__setattr__(self, "effect_classes", _effects(self.effect_classes))
        object.__setattr__(
            self, "template_id", _identifier(self.template_id, "template_id", required=False)
        )
        object.__setattr__(self, "grammar_id", _identifier(self.grammar_id, "grammar_id"))
        if self.grammar_id != GRAMMAR_ID:
            _refuse(ToolSynthesisReason.GRAMMAR_BOUND, "candidate grammar is not current", ToolGrammarError)
        object.__setattr__(
            self, "resources", _nested_record(self.resources, ToolResourceEnvelope, "resources")
        )
        object.__setattr__(self, "fixtures", _fixtures(self.fixtures))
        object.__setattr__(
            self,
            "test_fixture_ids",
            _strings(self.test_fixture_ids, "test_fixture_ids", identifiers=True),
        )
        object.__setattr__(
            self,
            "adversarial_fixture_ids",
            _strings(self.adversarial_fixture_ids, "adversarial_fixture_ids", identifiers=True),
        )
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        object.__setattr__(
            self, "compiler_revision", _identifier(self.compiler_revision, "compiler_revision")
        )
        if self.compiler_revision != COMPILER_REVISION:
            raise ToolSynthesisError("generated-tool compiler revision is not current")
        object.__setattr__(
            self,
            "predecessor_cid",
            _identifier(self.predecessor_cid, "predecessor_cid", required=False),
        )
        object.__setattr__(
            self,
            "certificate_cid",
            _identifier(self.certificate_cid, "certificate_cid", required=False),
        )
        object.__setattr__(
            self,
            "translation_receipt_cid",
            _identifier(self.translation_receipt_cid, "translation_receipt_cid", required=False),
        )
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        if self.can_authorize:
            _refuse(
                ToolSynthesisReason.PROMOTION_FORBIDDEN,
                "generated tool candidates cannot authorize",
                ToolSafetyError,
            )
        if self.representation is ToolRepresentation.INTERPRETED_DSL and self.state is not ArtifactState.CANDIDATE:
            _refuse(
                ToolSynthesisReason.DSL_PROMOTION_FORBIDDEN,
                "interpreted DSL tools remain candidate-tier",
                ToolPromotionError,
            )
        if self.state is ArtifactState.PROMOTED:
            if self.representation is not ToolRepresentation.OPTIMIZED_PYTHON:
                _refuse(
                    ToolSynthesisReason.DSL_PROMOTION_FORBIDDEN,
                    "only optimized Python may be promoted",
                    ToolPromotionError,
                )
            if not self.certificate_cid or not self.translation_receipt_cid:
                _refuse(
                    ToolSynthesisReason.CERTIFICATE_REQUIRED,
                    "promoted optimized Python requires a certificate and translation receipt",
                    ToolPromotionError,
                )
        elif self.state is not ArtifactState.CANDIDATE:
            _refuse(
                ToolSynthesisReason.PROMOTION_FORBIDDEN,
                "unverified generated tools remain candidate-tier",
                ToolPromotionError,
            )
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
            "representation": self.representation.value,
            "transformations": tuple(item.to_record() for item in self.transformations),
            "input_schema_ref": self.input_schema_ref,
            "output_schema_ref": self.output_schema_ref,
            "path_prefixes": self.path_prefixes,
            "effect_classes": tuple(item.value for item in self.effect_classes),
            "template_id": self.template_id,
            "grammar_id": self.grammar_id,
            "resources": self.resources.to_record(),
            "test_fixture_ids": self.test_fixture_ids,
            "adversarial_fixture_ids": self.adversarial_fixture_ids,
            "fixtures": tuple(item.to_record() for item in self.fixtures),
            "state": self.state.value,
            "compiler_revision": COMPILER_REVISION,
            "predecessor_cid": self.predecessor_cid,
            "certificate_cid": self.certificate_cid,
            "translation_receipt_cid": self.translation_receipt_cid,
            "can_authorize": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GeneratedToolCandidate:
        if not isinstance(payload, Mapping):
            raise ToolSynthesisError("GeneratedToolCandidate payload must be a mapping")
        body = _unwrap_generic_envelope(payload, cls.SCHEMA)
        fields = (
            "bindings",
            "tool_id",
            "spec_cid",
            "representation",
            "transformations",
            "input_schema_ref",
            "output_schema_ref",
            "path_prefixes",
            "effect_classes",
            "template_id",
            "grammar_id",
            "resources",
            "test_fixture_ids",
            "adversarial_fixture_ids",
            "fixtures",
            "state",
            "compiler_revision",
            "predecessor_cid",
            "certificate_cid",
            "translation_receipt_cid",
            "can_authorize",
        )
        record = cls(**_decode_fields(body, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class TranslationReceipt(CanonicalContract):
    """Exact differential comparison of interpreted DSL and optimized Python."""

    SCHEMA: ClassVar[str] = _schema_name("TranslationReceipt")

    bindings: ArtifactBindings
    spec_cid: str
    dsl_candidate_cid: str
    optimized_candidate_cid: str
    equivalent: bool
    passed_fixture_ids: tuple[str, ...]
    failed_fixture_ids: tuple[str, ...]
    adversarial_rejected_ids: tuple[str, ...]
    reason_code: ToolSynthesisReason = ToolSynthesisReason.EQUIVALENT
    validator_revision: str = VALIDATOR_REVISION
    state: ArtifactState = ArtifactState.CANDIDATE
    can_authorize: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        for name in ("spec_cid", "dsl_candidate_cid", "optimized_candidate_cid"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "equivalent", _bool(self.equivalent, "equivalent"))
        object.__setattr__(
            self,
            "passed_fixture_ids",
            _strings(self.passed_fixture_ids, "passed_fixture_ids", identifiers=True),
        )
        object.__setattr__(
            self,
            "failed_fixture_ids",
            _strings(self.failed_fixture_ids, "failed_fixture_ids", identifiers=True),
        )
        object.__setattr__(
            self,
            "adversarial_rejected_ids",
            _strings(self.adversarial_rejected_ids, "adversarial_rejected_ids", identifiers=True),
        )
        object.__setattr__(
            self, "reason_code", _enum(self.reason_code, ToolSynthesisReason, "reason_code")
        )
        object.__setattr__(
            self, "validator_revision", _identifier(self.validator_revision, "validator_revision")
        )
        if self.validator_revision != VALIDATOR_REVISION:
            raise ToolSynthesisError("translation validator revision is not current")
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if self.state is not ArtifactState.CANDIDATE:
            _refuse(
                ToolSynthesisReason.PROMOTION_FORBIDDEN,
                "translation receipts remain candidate-tier",
                ToolPromotionError,
            )
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        if self.can_authorize:
            _refuse(
                ToolSynthesisReason.PROMOTION_FORBIDDEN,
                "translation receipts cannot authorize",
                ToolSafetyError,
            )
        if self.equivalent:
            if self.failed_fixture_ids:
                _refuse(
                    ToolSynthesisReason.TRANSLATION_MISMATCH,
                    "equivalent translation cannot record failed fixtures",
                    ToolTranslationError,
                )
            if self.reason_code is not ToolSynthesisReason.EQUIVALENT:
                raise ToolSynthesisError("equivalent translation must use the equivalent reason")
        elif self.reason_code is ToolSynthesisReason.EQUIVALENT:
            raise ToolSynthesisError("inequivalent translation cannot claim equivalence")
        _bounded(self, "TranslationReceipt")

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
            "spec_cid": self.spec_cid,
            "dsl_candidate_cid": self.dsl_candidate_cid,
            "optimized_candidate_cid": self.optimized_candidate_cid,
            "equivalent": self.equivalent,
            "passed_fixture_ids": self.passed_fixture_ids,
            "failed_fixture_ids": self.failed_fixture_ids,
            "adversarial_rejected_ids": self.adversarial_rejected_ids,
            "reason_code": self.reason_code.value,
            "validator_revision": VALIDATOR_REVISION,
            "state": ArtifactState.CANDIDATE.value,
            "can_authorize": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TranslationReceipt:
        fields = (
            "bindings",
            "spec_cid",
            "dsl_candidate_cid",
            "optimized_candidate_cid",
            "equivalent",
            "passed_fixture_ids",
            "failed_fixture_ids",
            "adversarial_rejected_ids",
            "reason_code",
            "validator_revision",
            "state",
            "can_authorize",
        )
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class GeneratedToolCertificate(CanonicalContract):
    """Independent evidence that optimized Python matched the DSL exactly."""

    SCHEMA: ClassVar[str] = _schema_name("GeneratedToolCertificate")

    bindings: ArtifactBindings
    tool_id: str
    spec_cid: str
    candidate_cid: str
    translation_receipt_cid: str
    test_fixture_ids: tuple[str, ...]
    adversarial_fixture_ids: tuple[str, ...]
    equivalent: bool = True
    grammar_id: str = GRAMMAR_ID
    representation: ToolRepresentation = ToolRepresentation.OPTIMIZED_PYTHON
    issuer: str = CERTIFICATE_ISSUER
    validator_revision: str = VALIDATOR_REVISION
    state: ArtifactState = ArtifactState.VERIFIED
    can_authorize: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        for name in ("tool_id", "spec_cid", "candidate_cid", "translation_receipt_cid", "issuer"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "test_fixture_ids",
            _strings(self.test_fixture_ids, "test_fixture_ids", identifiers=True, required=True),
        )
        object.__setattr__(
            self,
            "adversarial_fixture_ids",
            _strings(
                self.adversarial_fixture_ids,
                "adversarial_fixture_ids",
                identifiers=True,
                required=True,
            ),
        )
        object.__setattr__(self, "equivalent", _bool(self.equivalent, "equivalent"))
        object.__setattr__(self, "grammar_id", _identifier(self.grammar_id, "grammar_id"))
        object.__setattr__(
            self,
            "representation",
            _enum(self.representation, ToolRepresentation, "representation"),
        )
        object.__setattr__(
            self, "validator_revision", _identifier(self.validator_revision, "validator_revision")
        )
        if self.validator_revision != VALIDATOR_REVISION:
            raise ToolSynthesisError("generated-tool certificate validator revision is not current")
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        if self.can_authorize:
            _refuse(
                ToolSynthesisReason.PROMOTION_FORBIDDEN,
                "generated tool certificates cannot authorize",
                ToolSafetyError,
            )
        if self.state is ArtifactState.PROMOTED:
            _refuse(
                ToolSynthesisReason.PROMOTION_FORBIDDEN,
                "certificates cannot self-promote",
                ToolPromotionError,
            )
        if self.equivalent:
            if self.state is not ArtifactState.VERIFIED:
                raise ToolSynthesisError("equivalent certificates must be verified")
            if self.representation is not ToolRepresentation.OPTIMIZED_PYTHON:
                _refuse(
                    ToolSynthesisReason.DSL_PROMOTION_FORBIDDEN,
                    "certificates bind optimized Python only",
                    ToolPromotionError,
                )
            if self.issuer != CERTIFICATE_ISSUER:
                raise ToolSynthesisError("generated-tool certificate issuer is not the translation validator")
        elif self.state not in {ArtifactState.REJECTED, ArtifactState.REVOKED, ArtifactState.STALE}:
            raise ToolSynthesisError("inequivalent certificates cannot be verified")
        _bounded(self, "GeneratedToolCertificate")

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
            "candidate_cid": self.candidate_cid,
            "translation_receipt_cid": self.translation_receipt_cid,
            "test_fixture_ids": self.test_fixture_ids,
            "adversarial_fixture_ids": self.adversarial_fixture_ids,
            "equivalent": self.equivalent,
            "grammar_id": self.grammar_id,
            "representation": self.representation.value,
            "issuer": self.issuer,
            "validator_revision": VALIDATOR_REVISION,
            "state": self.state.value,
            "can_authorize": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GeneratedToolCertificate:
        if not isinstance(payload, Mapping):
            raise ToolSynthesisError("GeneratedToolCertificate payload must be a mapping")
        body = _unwrap_generic_envelope(payload, cls.SCHEMA)
        fields = (
            "bindings",
            "tool_id",
            "spec_cid",
            "candidate_cid",
            "translation_receipt_cid",
            "test_fixture_ids",
            "adversarial_fixture_ids",
            "equivalent",
            "grammar_id",
            "representation",
            "issuer",
            "validator_revision",
            "state",
            "can_authorize",
        )
        record = cls(**_decode_fields(body, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class GeneratedToolInvocationReceipt(CanonicalContract):
    """Observation of one bounded invocation.  Never grants authority."""

    SCHEMA: ClassVar[str] = _schema_name("GeneratedToolInvocationReceipt")

    bindings: ArtifactBindings
    tool_id: str
    candidate_cid: str
    representation: ToolRepresentation
    input_digest: str
    output_digest: str
    accepted: bool
    reason_code: ToolSynthesisReason
    state: ArtifactState = ArtifactState.CANDIDATE
    compiler_revision: str = COMPILER_REVISION
    can_authorize: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        for name in ("tool_id", "candidate_cid", "input_digest", "output_digest"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(
            self,
            "representation",
            _enum(self.representation, ToolRepresentation, "representation"),
        )
        object.__setattr__(self, "accepted", _bool(self.accepted, "accepted"))
        object.__setattr__(
            self, "reason_code", _enum(self.reason_code, ToolSynthesisReason, "reason_code")
        )
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if self.state is not ArtifactState.CANDIDATE:
            _refuse(
                ToolSynthesisReason.PROMOTION_FORBIDDEN,
                "invocation receipts remain candidate-tier",
                ToolPromotionError,
            )
        object.__setattr__(
            self, "compiler_revision", _identifier(self.compiler_revision, "compiler_revision")
        )
        if self.compiler_revision != COMPILER_REVISION:
            raise ToolSynthesisError("invocation compiler revision is not current")
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        if self.can_authorize:
            _refuse(
                ToolSynthesisReason.PROMOTION_FORBIDDEN,
                "invocation receipts cannot authorize",
                ToolSafetyError,
            )
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
            "candidate_cid": self.candidate_cid,
            "representation": self.representation.value,
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
            "accepted": self.accepted,
            "reason_code": self.reason_code.value,
            "state": ArtifactState.CANDIDATE.value,
            "compiler_revision": COMPILER_REVISION,
            "can_authorize": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GeneratedToolInvocationReceipt:
        if not isinstance(payload, Mapping):
            raise ToolSynthesisError("GeneratedToolInvocationReceipt payload must be a mapping")
        body = _unwrap_generic_envelope(payload, cls.SCHEMA)
        fields = (
            "bindings",
            "tool_id",
            "candidate_cid",
            "representation",
            "input_digest",
            "output_digest",
            "accepted",
            "reason_code",
            "state",
            "compiler_revision",
            "can_authorize",
        )
        record = cls(**_decode_fields(body, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class CompiledTool:
    """Compiler output: candidate plus the declarative spec it was compiled from."""

    spec: GeneratedToolSpec
    candidate: GeneratedToolCandidate


@dataclass(frozen=True)
class ToolInvocationResult:
    """One interpreted invocation and its candidate-tier receipt."""

    output: Mapping[str, Any]
    receipt: GeneratedToolInvocationReceipt


@dataclass(frozen=True)
class TranslationValidation:
    """Validator output.  A certificate exists only after exact equivalence."""

    receipt: TranslationReceipt
    certificate: GeneratedToolCertificate | None


class GeneratedToolCompiler:
    """Compile reviewed DSL programs into candidate tools only."""

    revision: ClassVar[str] = COMPILER_REVISION

    def __init__(self, templates: Mapping[str, ApprovedTemplate] | None = None) -> None:
        self._dsl = TransformationDsl(templates)

    @property
    def dsl(self) -> TransformationDsl:
        return self._dsl

    def compile(
        self,
        *,
        bindings: ArtifactBindings,
        tool_id: str,
        template_id: str = "",
        transformations: Sequence[TransformationStep | Mapping[str, Any]] | None = None,
        input_schema_ref: str = "",
        output_schema_ref: str = "",
        path_prefixes: Sequence[str] | None = None,
        effect_classes: Sequence[EffectClass | str] = (EffectClass.OBSERVE,),
        resources: ToolResourceEnvelope | Mapping[str, Any] | None = None,
        fixtures: Sequence[ToolFixture | Mapping[str, Any]] | None = None,
    ) -> CompiledTool:
        template: ApprovedTemplate | None = None
        if template_id:
            template = self._dsl.templates.get(_identifier(template_id, "template_id"))
            if template is None:
                _refuse(
                    ToolSynthesisReason.UNKNOWN_TEMPLATE,
                    "template is outside the reviewed library",
                    ToolGrammarError,
                )
        if transformations is None:
            if template is None:
                _refuse(
                    ToolSynthesisReason.GRAMMAR_BOUND,
                    "compile requires a reviewed template or transformations",
                    ToolGrammarError,
                )
            program = template.transformations
        else:
            program = self._dsl.parse(transformations)
        if template is not None:
            prefixes = tuple(path_prefixes) if path_prefixes is not None else template.path_prefixes
            schema_in = input_schema_ref or template.input_schema_ref
            schema_out = output_schema_ref or template.output_schema_ref
            tool_fixtures = fixtures if fixtures is not None else template.fixtures
            assigned_template = template.template_id
            if path_prefixes is not None:
                requested = _strings(path_prefixes, "path_prefixes", paths=True, required=True)
                for item in requested:
                    if not _path_in_scope(item, template.path_prefixes):
                        _refuse(
                            ToolSynthesisReason.PATH_ESCAPE,
                            "compiled path prefixes escape the template scope",
                            ToolSafetyError,
                        )
        else:
            if not path_prefixes:
                _refuse(ToolSynthesisReason.PATH_ESCAPE, "compiled tools require path prefixes", ToolSafetyError)
            prefixes = _strings(path_prefixes, "path_prefixes", paths=True, required=True)
            schema_in = input_schema_ref or "schema.generated-tool.in"
            schema_out = output_schema_ref or "schema.generated-tool.out"
            tool_fixtures = fixtures or ()
            assigned_template = ""
        envelope = _nested_record(resources or ToolResourceEnvelope(), ToolResourceEnvelope, "resources")
        effects = _effects(effect_classes)
        expanded = self._dsl.expand(program)
        if len(expanded) > envelope.max_steps:
            _refuse(
                ToolSynthesisReason.RESOURCE_EXCEEDED,
                "compiled program exceeds the step bound",
                ToolBoundError,
            )
        spec = GeneratedToolSpec(
            bindings=bindings,
            tool_id=tool_id,
            input_schema_ref=schema_in,
            output_schema_ref=schema_out,
            transformations=program,
            path_prefixes=prefixes,
            effect_classes=effects,
            template_id=assigned_template,
            resources=envelope,
            fixtures=tool_fixtures,
        )
        candidate = GeneratedToolCandidate(
            bindings=bindings,
            tool_id=tool_id,
            spec_cid=spec.content_id,
            representation=ToolRepresentation.INTERPRETED_DSL,
            transformations=expanded,
            input_schema_ref=schema_in,
            output_schema_ref=schema_out,
            path_prefixes=prefixes,
            effect_classes=effects,
            template_id=assigned_template,
            resources=envelope,
            test_fixture_ids=spec.test_fixture_ids,
            adversarial_fixture_ids=spec.adversarial_fixture_ids,
            fixtures=spec.fixtures,
        )
        return CompiledTool(spec=spec, candidate=candidate)

    def synthesize(
        self,
        traces: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
        *,
        bindings: ArtifactBindings,
        tool_id: str = "",
    ) -> CompiledTool:
        if not isinstance(traces, Sequence) or isinstance(traces, (str, bytes, bytearray, memoryview)):
            raise ToolSynthesisError("traces must be a sequence")
        if not traces:
            _refuse(ToolSynthesisReason.NO_TEMPLATE_MATCH, "synthesis requires repeated traces")
        if len(traces) > MAX_TRACES:
            _refuse(ToolSynthesisReason.RESOURCE_EXCEEDED, "trace set exceeds its bound", ToolBoundError)
        matches: list[ApprovedTemplate] = []
        for template in self._dsl.templates.values():
            if self._covers(template, traces):
                matches.append(template)
        if not matches:
            _refuse(
                ToolSynthesisReason.NO_TEMPLATE_MATCH,
                "repeated traces do not match a reviewed template",
                ToolGrammarError,
            )
        matches.sort(key=lambda item: (len(item.transformations), item.template_id))
        chosen = matches[0]
        assigned = tool_id or f"tool.{chosen.template_id}"
        return self.compile(bindings=bindings, tool_id=assigned, template_id=chosen.template_id)

    def optimize(self, compiled: CompiledTool | GeneratedToolCandidate) -> GeneratedToolCandidate:
        candidate = compiled.candidate if isinstance(compiled, CompiledTool) else compiled
        if not isinstance(candidate, GeneratedToolCandidate):
            raise ToolSynthesisError("optimize requires a generated tool candidate")
        fused = self._dsl.optimize(candidate.transformations)
        return GeneratedToolCandidate(
            bindings=candidate.bindings,
            tool_id=candidate.tool_id,
            spec_cid=candidate.spec_cid,
            representation=ToolRepresentation.OPTIMIZED_PYTHON,
            transformations=fused,
            input_schema_ref=candidate.input_schema_ref,
            output_schema_ref=candidate.output_schema_ref,
            path_prefixes=candidate.path_prefixes,
            effect_classes=candidate.effect_classes,
            template_id=candidate.template_id,
            resources=candidate.resources,
            test_fixture_ids=candidate.test_fixture_ids,
            adversarial_fixture_ids=candidate.adversarial_fixture_ids,
            fixtures=candidate.fixtures,
            predecessor_cid=candidate.content_id,
        )

    def invoke(
        self,
        candidate: GeneratedToolCandidate,
        payload: Mapping[str, Any],
    ) -> ToolInvocationResult:
        if not isinstance(candidate, GeneratedToolCandidate):
            raise ToolSynthesisError("invoke requires a generated tool candidate")
        input_digest = _digest(_payload_mapping(payload, "payload"))
        output = self._dsl.interpret(
            candidate.transformations,
            payload,
            path_prefixes=candidate.path_prefixes,
            resources=candidate.resources,
            optimized=candidate.representation is ToolRepresentation.OPTIMIZED_PYTHON,
        )
        receipt = GeneratedToolInvocationReceipt(
            bindings=candidate.bindings,
            tool_id=candidate.tool_id,
            candidate_cid=candidate.content_id,
            representation=candidate.representation,
            input_digest=input_digest,
            output_digest=_digest(output),
            accepted=True,
            reason_code=ToolSynthesisReason.CANDIDATE_SYNTHESIZED,
        )
        return ToolInvocationResult(output=output, receipt=receipt)

    def _covers(
        self,
        template: ApprovedTemplate,
        traces: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    ) -> bool:
        for raw_input, raw_output in traces:
            try:
                actual = self._dsl.interpret(
                    template.transformations,
                    raw_input,
                    path_prefixes=template.path_prefixes,
                    resources=template.resources,
                )
            except (ProcedureContractError, ToolSynthesisError):
                return False
            expected = _payload_mapping(raw_output, "trace.output")
            if dict(actual) != dict(expected):
                return False
        return True


class TranslationValidator:
    """Exact DSL-to-optimized differential validation and certificate issuance."""

    revision: ClassVar[str] = VALIDATOR_REVISION

    def __init__(self, templates: Mapping[str, ApprovedTemplate] | None = None) -> None:
        self._dsl = TransformationDsl(templates)

    def validate(
        self,
        dsl_candidate: GeneratedToolCandidate,
        optimized_candidate: GeneratedToolCandidate,
        fixtures: Sequence[ToolFixture | Mapping[str, Any]] | None = None,
    ) -> TranslationReceipt:
        if dsl_candidate.representation is not ToolRepresentation.INTERPRETED_DSL:
            _refuse(
                ToolSynthesisReason.TRANSLATION_MISMATCH,
                "translation requires an interpreted DSL candidate",
                ToolTranslationError,
            )
        if optimized_candidate.representation is not ToolRepresentation.OPTIMIZED_PYTHON:
            _refuse(
                ToolSynthesisReason.TRANSLATION_MISMATCH,
                "translation requires an optimized Python candidate",
                ToolTranslationError,
            )
        if dsl_candidate.spec_cid != optimized_candidate.spec_cid:
            _refuse(
                ToolSynthesisReason.CERTIFICATE_MISMATCH,
                "translated candidates must bind the same spec",
                ToolTranslationError,
            )
        if dsl_candidate.bindings != optimized_candidate.bindings:
            _refuse(
                ToolSynthesisReason.CERTIFICATE_MISMATCH,
                "translated candidates must share exact bindings",
                ToolTranslationError,
            )
        records = _fixtures(fixtures if fixtures is not None else dsl_candidate.fixtures)
        if not records:
            _refuse(
                ToolSynthesisReason.FIXTURE_FAILED,
                "translation validation requires fixtures",
                ToolTranslationError,
            )
        passed: list[str] = []
        failed: list[str] = []
        adversarial: list[str] = []
        equivalent = True
        for fixture in records:
            dsl_outcome = self._run(dsl_candidate, fixture, optimized=False)
            opt_outcome = self._run(optimized_candidate, fixture, optimized=True)
            if fixture.must_refuse:
                if (
                    dsl_outcome[0] is None
                    and opt_outcome[0] is None
                    and dsl_outcome[1] == opt_outcome[1]
                ):
                    adversarial.append(fixture.fixture_id)
                    continue
                failed.append(fixture.fixture_id)
                equivalent = False
                continue
            if (
                dsl_outcome[0] is not None
                and opt_outcome[0] is not None
                and dict(dsl_outcome[0]) == dict(opt_outcome[0])
                and (fixture.expected is None or dict(dsl_outcome[0]) == dict(fixture.expected))
            ):
                passed.append(fixture.fixture_id)
                continue
            failed.append(fixture.fixture_id)
            equivalent = False
        reason = (
            ToolSynthesisReason.EQUIVALENT
            if equivalent
            else ToolSynthesisReason.TRANSLATION_MISMATCH
        )
        return TranslationReceipt(
            bindings=dsl_candidate.bindings,
            spec_cid=dsl_candidate.spec_cid,
            dsl_candidate_cid=dsl_candidate.content_id,
            optimized_candidate_cid=optimized_candidate.content_id,
            equivalent=equivalent,
            passed_fixture_ids=tuple(passed),
            failed_fixture_ids=tuple(failed),
            adversarial_rejected_ids=tuple(adversarial),
            reason_code=reason,
        )

    def certify(self, receipt: TranslationReceipt, optimized_candidate: GeneratedToolCandidate) -> GeneratedToolCertificate:
        if not receipt.equivalent:
            _refuse(
                ToolSynthesisReason.TRANSLATION_MISMATCH,
                "certificate requires exact differential equivalence",
                ToolTranslationError,
            )
        if receipt.optimized_candidate_cid != optimized_candidate.content_id:
            _refuse(
                ToolSynthesisReason.CERTIFICATE_MISMATCH,
                "certificate candidate does not match the translation receipt",
                ToolTranslationError,
            )
        if optimized_candidate.representation is not ToolRepresentation.OPTIMIZED_PYTHON:
            _refuse(
                ToolSynthesisReason.DSL_PROMOTION_FORBIDDEN,
                "certificates are issued only for optimized Python",
                ToolPromotionError,
            )
        if not receipt.passed_fixture_ids or not receipt.adversarial_rejected_ids:
            _refuse(
                ToolSynthesisReason.FIXTURE_FAILED,
                "certificate requires passing tests and rejected adversarial fixtures",
                ToolTranslationError,
            )
        return GeneratedToolCertificate(
            bindings=optimized_candidate.bindings,
            tool_id=optimized_candidate.tool_id,
            spec_cid=optimized_candidate.spec_cid,
            candidate_cid=optimized_candidate.content_id,
            translation_receipt_cid=receipt.content_id,
            test_fixture_ids=receipt.passed_fixture_ids,
            adversarial_fixture_ids=receipt.adversarial_rejected_ids,
        )

    def validate_and_certify(
        self,
        dsl_candidate: GeneratedToolCandidate,
        optimized_candidate: GeneratedToolCandidate,
        fixtures: Sequence[ToolFixture | Mapping[str, Any]] | None = None,
    ) -> TranslationValidation:
        receipt = self.validate(dsl_candidate, optimized_candidate, fixtures)
        certificate = None
        if receipt.equivalent:
            certificate = self.certify(receipt, optimized_candidate)
        return TranslationValidation(receipt=receipt, certificate=certificate)

    def promote(
        self,
        optimized_candidate: GeneratedToolCandidate,
        certificate: GeneratedToolCertificate,
        receipt: TranslationReceipt,
    ) -> GeneratedToolCandidate:
        if optimized_candidate.representation is not ToolRepresentation.OPTIMIZED_PYTHON:
            _refuse(
                ToolSynthesisReason.DSL_PROMOTION_FORBIDDEN,
                "interpreted DSL tools cannot be promoted",
                ToolPromotionError,
            )
        if optimized_candidate.state is ArtifactState.PROMOTED:
            _refuse(
                ToolSynthesisReason.PROMOTION_FORBIDDEN,
                "generated tools cannot self-promote",
                ToolPromotionError,
            )
        if not receipt.equivalent or not certificate.equivalent:
            _refuse(
                ToolSynthesisReason.TRANSLATION_MISMATCH,
                "promotion requires exact differential equivalence",
                ToolPromotionError,
            )
        if certificate.state is not ArtifactState.VERIFIED:
            _refuse(
                ToolSynthesisReason.CERTIFICATE_REQUIRED,
                "promotion requires a verified certificate",
                ToolPromotionError,
            )
        if certificate.candidate_cid != optimized_candidate.content_id:
            _refuse(
                ToolSynthesisReason.CERTIFICATE_MISMATCH,
                "certificate does not bind this optimized candidate",
                ToolPromotionError,
            )
        if certificate.spec_cid != optimized_candidate.spec_cid:
            _refuse(
                ToolSynthesisReason.CERTIFICATE_MISMATCH,
                "certificate does not bind this tool spec",
                ToolPromotionError,
            )
        if certificate.translation_receipt_cid != receipt.content_id:
            _refuse(
                ToolSynthesisReason.CERTIFICATE_MISMATCH,
                "certificate does not bind this translation receipt",
                ToolPromotionError,
            )
        if receipt.optimized_candidate_cid != optimized_candidate.content_id:
            _refuse(
                ToolSynthesisReason.CERTIFICATE_MISMATCH,
                "translation receipt does not bind this optimized candidate",
                ToolPromotionError,
            )
        if certificate.bindings != optimized_candidate.bindings or receipt.bindings != optimized_candidate.bindings:
            _refuse(
                ToolSynthesisReason.CERTIFICATE_MISMATCH,
                "promotion bindings are not exact",
                ToolPromotionError,
            )
        return GeneratedToolCandidate(
            bindings=optimized_candidate.bindings,
            tool_id=optimized_candidate.tool_id,
            spec_cid=optimized_candidate.spec_cid,
            representation=ToolRepresentation.OPTIMIZED_PYTHON,
            transformations=optimized_candidate.transformations,
            input_schema_ref=optimized_candidate.input_schema_ref,
            output_schema_ref=optimized_candidate.output_schema_ref,
            path_prefixes=optimized_candidate.path_prefixes,
            effect_classes=optimized_candidate.effect_classes,
            template_id=optimized_candidate.template_id,
            resources=optimized_candidate.resources,
            test_fixture_ids=optimized_candidate.test_fixture_ids,
            adversarial_fixture_ids=optimized_candidate.adversarial_fixture_ids,
            fixtures=optimized_candidate.fixtures,
            state=ArtifactState.PROMOTED,
            predecessor_cid=optimized_candidate.content_id,
            certificate_cid=certificate.content_id,
            translation_receipt_cid=receipt.content_id,
        )

    def _run(
        self,
        candidate: GeneratedToolCandidate,
        fixture: ToolFixture,
        *,
        optimized: bool,
    ) -> tuple[Mapping[str, Any] | None, ToolSynthesisReason | None]:
        try:
            output = self._dsl.interpret(
                candidate.transformations,
                fixture.payload,
                path_prefixes=candidate.path_prefixes,
                resources=candidate.resources,
                optimized=optimized,
            )
        except (ProcedureContractError, ToolSynthesisError) as exc:
            reason = getattr(exc, "reason_code", None)
            if not isinstance(reason, ToolSynthesisReason):
                if fixture.must_refuse:
                    reason = ToolSynthesisReason.ADVERSARIAL_REJECTED
                else:
                    reason = ToolSynthesisReason.FIXTURE_FAILED
            return None, reason
        return output, None


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
    "ALLOWED_OPCODES",
    "ALLOWED_SEPARATORS",
    "APPROVED_REPAIR_TEMPLATE_IDS",
    "APPROVED_TEMPLATE_LIBRARY",
    "CERTIFICATE_ISSUER",
    "COMPILER_REVISION",
    "DSL_REVISION",
    "GRAMMAR_ID",
    "MAX_TOOL_OUTPUT_BYTES",
    "MAX_TOOL_STEPS",
    "VALIDATOR_REVISION",
    "ApprovedTemplate",
    "CompiledTool",
    "FixtureKind",
    "GeneratedToolCandidate",
    "GeneratedToolCertificate",
    "GeneratedToolCompiler",
    "GeneratedToolInvocationReceipt",
    "GeneratedToolSpec",
    "ToolBoundError",
    "ToolFixture",
    "ToolGrammarError",
    "ToolInvocationResult",
    "ToolPromotionError",
    "ToolRepresentation",
    "ToolResourceEnvelope",
    "ToolSafetyError",
    "ToolSynthesisError",
    "ToolSynthesisReason",
    "ToolTranslationError",
    "TransformationDsl",
    "TransformationOpcode",
    "TransformationStep",
    "TranslationReceipt",
    "TranslationValidation",
    "TranslationValidator",
]
