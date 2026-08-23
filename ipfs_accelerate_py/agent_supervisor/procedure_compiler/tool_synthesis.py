"""Bounded deterministic tool synthesis from a reviewed transformation DSL.

Repeated pure/bounded transformations may become interpreted DSL tools from a
reviewed grammar/template library.  This module owns synthesis, not authority:
generated tools remain candidates.  Optimized Python is a reviewed translation
of the same template, never an arbitrary script, and is promoted only after
exact differential validation and an independent certificate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    FORBIDDEN_STEP_OPERATIONS,
    MAX_ITEMS,
    PROCEDURE_CONTRACT_VERSION,
    ArtifactBindings,
    ArtifactState,
    EffectClass,
    ProcedureContractError,
    ProcedureSafetyError,
    RiskClass,
    _bounded,
    _decode_fields,
    _enum,
    _enums,
    _freeze,
    _identifier,
    _nested,
    _nonnegative_int,
    _positive_int,
    _schema_name,
    _strings,
    _text,
    _unsafe_key,
    _verify_identity,
)
from .verifier import FORBIDDEN_SELF_PRODUCERS


DSL_REVISION: Final[str] = "DeterministicToolDsl@1"
COMPILER_REVISION: Final[str] = "GeneratedToolCompiler@1"
VALIDATOR_REVISION: Final[str] = "TranslationValidator@1"
GRAMMAR_REVISION: Final[str] = "transformation-dsl-grammar@1"
DEFAULT_CERTIFICATE_ISSUER: Final[str] = "translation-validator@1"
MIN_REPETITIONS: Final[int] = 2
MAX_OPERATIONS: Final[int] = 16
MAX_OCCURRENCES: Final[int] = 128
MAX_FIXTURES: Final[int] = 32
MAX_OUTPUT_BYTES: Final[int] = 65_536
MAX_PATH_READS: Final[int] = 128
MAX_DURATION_MS: Final[int] = 60_000
RISK_CEILING: Final[RiskClass] = RiskClass.OBSERVATION_ONLY
ALLOWED_EFFECTS: Final[frozenset[EffectClass]] = frozenset({EffectClass.OBSERVE})
JOIN_SEPARATORS: Final[frozenset[str]] = frozenset({"", ".", ":", "/", "_", "-"})

REQUIRED_TOOL_DECLARATION_FIELDS: Final[tuple[str, ...]] = (
    "schema",
    "effects",
    "path_limits",
    "resources",
    "tests",
    "adversarial_fixtures",
)

APPROVED_REPAIR_TEMPLATE_IDS: Final[tuple[str, ...]] = (
    "repair.qualify-symbol",
    "repair.normalize-import",
    "repair.sort-declared-reads",
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
_FORBIDDEN_SELF_ISSUERS: Final[frozenset[str]] = FORBIDDEN_SELF_PRODUCERS | frozenset(
    {
        COMPILER_REVISION.lower(),
        "generated-tool-compiler",
        "generated-tool-compiler@1",
        "self-promoted",
        "tool-self",
    }
)
_FORBIDDEN_VALUE_MARKERS: Final[tuple[str, ...]] = (
    "eval(",
    "exec(",
    "compile(",
    "__import__",
    "os.system",
    "os.popen",
    "subprocess",
    "socket.socket",
    "/bin/sh",
    "/bin/bash",
    "cmd.exe",
    "powershell",
    "arbitrary_shell",
    "arbitrary_python",
    "network_request",
)
_PATH_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    {"path", "paths", "target", "targets", "file", "files", "scope_path", "scope_paths"}
)
_SCOPE = "ipfs_accelerate_py/agent_supervisor/procedure_compiler"


class ToolSynthesisError(ProcedureContractError):
    """A transformation program, generated tool, or translation is unsafe."""


class ToolGrammarError(ToolSynthesisError):
    """The DSL program is outside the reviewed grammar bound."""


class ToolTranslationError(ToolSynthesisError):
    """DSL and optimized Python translations are not differentially exact."""


class ToolCertificateError(ToolSynthesisError):
    """Generated-tool certificate issuance or verification failed closed."""


class ToolPromotionError(ToolSynthesisError):
    """Optimized Python cannot be promoted from the supplied evidence."""


class ToolSynthesisRefusal(ToolSynthesisError):
    """A synthesis, validation, or promotion request was refused fail-closed."""

    def __init__(self, message: str, reason_code: "ToolReason") -> None:
        super().__init__(message)
        self.reason_code = reason_code


class ToolReason(str, Enum):
    ADMITTED = "admitted"
    INSUFFICIENT_REPETITION = "insufficient-repetition"
    UNKNOWN_TEMPLATE = "unknown-template"
    TEMPLATE_MISMATCH = "template-mismatch"
    GRAMMAR_UNBOUNDED = "grammar-unbounded"
    FORBIDDEN_OPCODE = "forbidden-opcode"
    ARBITRARY_CODE = "arbitrary-code"
    PATH_ESCAPE = "path-escape"
    EFFECT_ESCALATION = "effect-escalation"
    MISSING_SCHEMA = "missing-schema"
    MISSING_TESTS = "missing-tests"
    MISSING_ADVERSARIAL = "missing-adversarial"
    RESOURCE_BOUND = "resource-bound"
    TRANSLATION_MISMATCH = "translation-mismatch"
    MISSING_CERTIFICATE = "missing-certificate"
    MISSING_VALIDATION = "missing-validation"
    SELF_ISSUED = "self-issued"
    FORGED_CERTIFICATE = "forged-certificate"
    STALE_CERTIFICATE = "stale-certificate"
    CANDIDATE_TIER_REQUIRED = "candidate-tier-required"
    AUTHORITY_REJECTED = "authority-rejected"
    PROMOTION_FORBIDDEN = "promotion-forbidden"
    OPTIMIZED_NOT_PROMOTED = "optimized-not-promoted"
    BINDING_MISMATCH = "binding-mismatch"
    SCHEMA_MISMATCH = "schema-mismatch"
    UNAPPROVED_TEMPLATE = "unapproved-template"
    RISK_CEILING = "risk-ceiling"


class TransformationOpcode(str, Enum):
    PROJECT = "project"
    RENAME = "rename"
    FILTER_EQUALS = "filter-equals"
    FILTER_IN = "filter-in"
    SORT = "sort"
    DEDUPLICATE = "deduplicate"
    JOIN_IDENTIFIERS = "join-identifiers"
    CLAMP_INT = "clamp-int"
    SELECT_SCOPED_PATHS = "select-scoped-paths"
    MAP_APPROVED_TEMPLATE = "map-approved-template"


class TranslationKind(str, Enum):
    INTERPRETED_DSL = "interpreted-dsl"
    OPTIMIZED_PYTHON = "optimized-python"


class TranslationStatus(str, Enum):
    CANDIDATE = "candidate"
    VERIFIED = "verified"
    PROMOTED = "promoted"
    REJECTED = "rejected"


class ToolPromotionAction(str, Enum):
    PROMOTE_OPTIMIZED = "promote-optimized"
    REFUSE = "refuse"


class FixtureKind(str, Enum):
    TEST = "test"
    ADVERSARIAL = "adversarial"


def _bool(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise ToolSynthesisError(f"{field_name} must be a boolean")
    return value


def _bindings(value: Any) -> ArtifactBindings:
    return _nested(value, ArtifactBindings, "bindings")


def _refuse(reason: ToolReason, message: str) -> NoReturn:
    raise ToolSynthesisRefusal(message, reason)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_thaw(item) for item in value)
    return value


def _canonical_equal(left: Any, right: Any) -> bool:
    return canonical_json_bytes(_thaw(left)) == canonical_json_bytes(_thaw(right))


def _output_bytes(value: Any) -> int:
    return len(canonical_json_bytes(_thaw(value)))


def _normalized_marker(value: str) -> str:
    return value.lower().replace("-", "_")


def _contains_forbidden_code(value: str) -> bool:
    normalized = _normalized_marker(value)
    return any(marker in normalized for marker in _FORBIDDEN_VALUE_MARKERS)


def _scan_forbidden(value: Any) -> ToolReason | None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                return ToolReason.ARBITRARY_CODE
            if _unsafe_key(raw_key) or _contains_forbidden_code(raw_key):
                return ToolReason.ARBITRARY_CODE
            nested = _scan_forbidden(item)
            if nested is not None:
                return nested
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        for item in value:
            nested = _scan_forbidden(item)
            if nested is not None:
                return nested
        return None
    if isinstance(value, str) and _contains_forbidden_code(value):
        return ToolReason.ARBITRARY_CODE
    return None


def _reject_forbidden(value: Any, field_name: str) -> None:
    reason = _scan_forbidden(value)
    if reason is not None:
        _refuse(reason, f"{field_name} contains arbitrary code, shell, or executable material")


def _freeze_input(value: Any, field_name: str) -> Any:
    try:
        _reject_forbidden(value, field_name)
        return _freeze(value, field_name)
    except ToolSynthesisRefusal:
        raise
    except ProcedureSafetyError as exc:
        message = str(exc).lower()
        if "path" in message:
            _refuse(ToolReason.PATH_ESCAPE, str(exc))
        _refuse(ToolReason.ARBITRARY_CODE, str(exc))
    except ProcedureContractError as exc:
        _refuse(ToolReason.GRAMMAR_UNBOUNDED, str(exc))


def _path_is_within(path: str, prefixes: Sequence[str]) -> bool:
    candidate = PurePosixPath(path)
    for prefix in prefixes:
        root = PurePosixPath(prefix)
        if prefix == ".":
            return True
        if candidate == root or root in candidate.parents:
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


def _item_count(value: Any) -> int:
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview)):
        return len(value)
    return 1


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and not isinstance(value, (str, bytes, bytearray, memoryview))


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview))


def _require_records(value: Any, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if _is_mapping(value):
        return (value,)
    if not _is_sequence(value):
        _refuse(ToolReason.SCHEMA_MISMATCH, f"{field_name} must be a mapping or a sequence of mappings")
    records: list[Mapping[str, Any]] = []
    for item in value:
        if not _is_mapping(item):
            _refuse(ToolReason.SCHEMA_MISMATCH, f"{field_name} must contain mappings")
        records.append(item)
    return tuple(records)


def _restore_shape(original: Any, records: Sequence[Mapping[str, Any]]) -> Any:
    if _is_mapping(original):
        if len(records) != 1:
            _refuse(ToolReason.SCHEMA_MISMATCH, "mapping transformations must yield one record")
        return records[0]
    return tuple(records)


def _record_paths(record: Mapping[str, Any]) -> tuple[str, ...]:
    found: list[str] = []
    for key, item in record.items():
        if key not in _PATH_FIELD_NAMES:
            continue
        if isinstance(item, str):
            found.append(item)
        elif _is_sequence(item):
            for entry in item:
                if isinstance(entry, str):
                    found.append(entry)
    return tuple(found)


def _enforce_scope(paths: Sequence[str], scope_paths: Sequence[str]) -> None:
    if not scope_paths:
        _refuse(ToolReason.PATH_ESCAPE, "generated tools require declared path limits")
    for path in paths:
        if not _path_is_within(path, scope_paths):
            _refuse(ToolReason.PATH_ESCAPE, f"path {path} escapes the declared tool scope")


def _check_value_scope(value: Any, scope_paths: Sequence[str]) -> tuple[str, ...]:
    seen: list[str] = []
    if _is_mapping(value):
        seen.extend(_record_paths(value))
    elif _is_sequence(value):
        for item in value:
            if isinstance(item, str):
                seen.append(item)
            elif _is_mapping(item):
                seen.extend(_record_paths(item))
    _enforce_scope(seen, scope_paths)
    return tuple(dict.fromkeys(seen))


def _join_separator(value: Any) -> str:
    separator = _text(value, "operand", required=False)
    if separator not in JOIN_SEPARATORS:
        _refuse(ToolReason.ARBITRARY_CODE, "join separator is outside the closed vocabulary")
    return separator


@dataclass(frozen=True)
class ToolResourceBound:
    """Fixed integer resource envelope.  Exceeding it is fail-closed."""

    max_operations: int = MAX_OPERATIONS
    max_output_bytes: int = MAX_OUTPUT_BYTES
    max_items: int = MAX_ITEMS
    max_duration_ms: int = 1_000
    max_path_reads: int = MAX_PATH_READS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_operations",
            _positive_int(self.max_operations, "max_operations", maximum=MAX_OPERATIONS),
        )
        object.__setattr__(
            self,
            "max_output_bytes",
            _positive_int(self.max_output_bytes, "max_output_bytes", maximum=MAX_OUTPUT_BYTES),
        )
        object.__setattr__(
            self,
            "max_items",
            _positive_int(self.max_items, "max_items", maximum=MAX_ITEMS),
        )
        object.__setattr__(
            self,
            "max_duration_ms",
            _positive_int(self.max_duration_ms, "max_duration_ms", maximum=MAX_DURATION_MS),
        )
        object.__setattr__(
            self,
            "max_path_reads",
            _positive_int(self.max_path_reads, "max_path_reads", maximum=MAX_PATH_READS),
        )

    def admits(self, *, operations: int, output_bytes: int, items: int, path_reads: int) -> bool:
        return (
            operations <= self.max_operations
            and output_bytes <= self.max_output_bytes
            and items <= self.max_items
            and path_reads <= self.max_path_reads
        )

    def to_record(self) -> dict[str, int]:
        return {
            "max_operations": self.max_operations,
            "max_output_bytes": self.max_output_bytes,
            "max_items": self.max_items,
            "max_duration_ms": self.max_duration_ms,
            "max_path_reads": self.max_path_reads,
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any] | ToolResourceBound | None) -> ToolResourceBound:
        if isinstance(payload, ToolResourceBound):
            return payload
        if payload is None or not isinstance(payload, Mapping):
            raise ToolGrammarError("resource bound must be a mapping")
        return cls(
            max_operations=payload.get("max_operations", MAX_OPERATIONS),
            max_output_bytes=payload.get("max_output_bytes", MAX_OUTPUT_BYTES),
            max_items=payload.get("max_items", MAX_ITEMS),
            max_duration_ms=payload.get("max_duration_ms", 1_000),
            max_path_reads=payload.get("max_path_reads", MAX_PATH_READS),
        )


@dataclass(frozen=True)
class TransformationOp:
    """One closed, non-recursive transformation step."""

    opcode: TransformationOpcode
    field: str = ""
    operand: Any = None

    def __post_init__(self) -> None:
        raw_opcode = self.opcode
        if isinstance(raw_opcode, str) and (
            raw_opcode in FORBIDDEN_STEP_OPERATIONS or _contains_forbidden_code(raw_opcode)
        ):
            _refuse(
                ToolReason.FORBIDDEN_OPCODE,
                "arbitrary shell, Python, or network opcodes are forbidden",
            )
        try:
            object.__setattr__(
                self, "opcode", _enum(raw_opcode, TransformationOpcode, "opcode")
            )
        except ProcedureContractError as exc:
            _refuse(ToolReason.FORBIDDEN_OPCODE, str(exc))
        object.__setattr__(
            self, "field", _identifier(self.field, "field", required=False)
        )
        frozen_operand = _freeze_input(
            () if self.operand is None else self.operand, "operand"
        )
        object.__setattr__(self, "operand", frozen_operand)
        if self.opcode is TransformationOpcode.JOIN_IDENTIFIERS:
            _join_separator(self.operand)
        if self.opcode is TransformationOpcode.MAP_APPROVED_TEMPLATE:
            template_id = _identifier(self.operand, "operand")
            if template_id not in APPROVED_REPAIR_TEMPLATE_IDS:
                _refuse(ToolReason.UNAPPROVED_TEMPLATE, "repair template is not in the reviewed library")
            object.__setattr__(self, "operand", template_id)

    def to_record(self) -> dict[str, Any]:
        return {
            "opcode": self.opcode.value,
            "field": self.field,
            "operand": _thaw(self.operand),
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any] | TransformationOp) -> TransformationOp:
        if isinstance(payload, TransformationOp):
            return payload
        if not isinstance(payload, Mapping):
            raise ToolGrammarError("transformation operation must be a mapping")
        return cls(
            opcode=payload.get("opcode", ""),
            field=payload.get("field", ""),
            operand=payload.get("operand"),
        )


def _operations(values: Any) -> tuple[TransformationOp, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif _is_sequence(values):
        raw = values
    else:
        raise ToolGrammarError("operations must be a sequence")
    if not raw:
        _refuse(ToolReason.GRAMMAR_UNBOUNDED, "a transformation program must contain operations")
    if len(raw) > MAX_OPERATIONS:
        _refuse(ToolReason.GRAMMAR_UNBOUNDED, "transformation program exceeds the operation bound")
    return tuple(TransformationOp.from_record(item) for item in raw)


@dataclass(frozen=True)
class TransformationProgram:
    """Flat bounded program.  Loops, recursion, and callbacks are forbidden."""

    operations: tuple[TransformationOp, ...]
    input_schema_ref: str
    output_schema_ref: str
    scope_paths: tuple[str, ...]
    effect_classes: tuple[EffectClass, ...] = (EffectClass.OBSERVE,)
    grammar_revision: str = GRAMMAR_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "operations", _operations(self.operations))
        object.__setattr__(
            self, "input_schema_ref", _identifier(self.input_schema_ref, "input_schema_ref")
        )
        object.__setattr__(
            self, "output_schema_ref", _identifier(self.output_schema_ref, "output_schema_ref")
        )
        object.__setattr__(
            self,
            "scope_paths",
            _strings(self.scope_paths, "scope_paths", paths=True, required=True),
        )
        effects = _enums(
            self.effect_classes,
            EffectClass,
            "effect_classes",
            limit=len(EffectClass),
            required=True,
        )
        if set(effects) - ALLOWED_EFFECTS:
            _refuse(ToolReason.EFFECT_ESCALATION, "generated tools may only declare observe effects")
        object.__setattr__(self, "effect_classes", effects)
        object.__setattr__(
            self, "grammar_revision", _identifier(self.grammar_revision, "grammar_revision")
        )
        if self.grammar_revision != GRAMMAR_REVISION:
            raise ToolGrammarError("transformation grammar revision is not current")

    @property
    def fingerprint(self) -> str:
        return content_identity(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/procedure-compiler/transformation-program-fingerprint@1",
                "program": self.to_record(),
            }
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "operations": tuple(item.to_record() for item in self.operations),
            "input_schema_ref": self.input_schema_ref,
            "output_schema_ref": self.output_schema_ref,
            "scope_paths": self.scope_paths,
            "effect_classes": tuple(item.value for item in self.effect_classes),
            "grammar_revision": self.grammar_revision,
        }

    @classmethod
    def from_record(
        cls, payload: Mapping[str, Any] | TransformationProgram
    ) -> TransformationProgram:
        if isinstance(payload, TransformationProgram):
            return payload
        if not isinstance(payload, Mapping):
            raise ToolGrammarError("transformation program must be a mapping")
        return cls(
            operations=payload.get("operations", ()),
            input_schema_ref=payload.get("input_schema_ref", ""),
            output_schema_ref=payload.get("output_schema_ref", ""),
            scope_paths=payload.get("scope_paths", ()),
            effect_classes=payload.get("effect_classes", (EffectClass.OBSERVE,)),
            grammar_revision=payload.get("grammar_revision", GRAMMAR_REVISION),
        )


@dataclass(frozen=True)
class ToolFixture:
    """Closed test or adversarial vector.  Large bodies are forbidden."""

    fixture_id: str
    input_value: Any
    expected_output: Any = ()
    expect_reject: bool = False
    reason_code: str = ""
    kind: FixtureKind = FixtureKind.TEST

    def __post_init__(self) -> None:
        object.__setattr__(self, "fixture_id", _identifier(self.fixture_id, "fixture_id"))
        object.__setattr__(self, "kind", _enum(self.kind, FixtureKind, "kind"))
        object.__setattr__(self, "expect_reject", _bool(self.expect_reject, "expect_reject"))
        object.__setattr__(
            self, "reason_code", _identifier(self.reason_code, "reason_code", required=False)
        )
        if self.expect_reject:
            object.__setattr__(self, "input_value", _freeze_input(self.input_value, "input_value"))
            object.__setattr__(self, "expected_output", ())
            if not self.reason_code:
                _refuse(ToolReason.MISSING_ADVERSARIAL, "rejecting fixtures must declare a reason")
        else:
            object.__setattr__(self, "input_value", _freeze_input(self.input_value, "input_value"))
            object.__setattr__(
                self, "expected_output", _freeze_input(self.expected_output, "expected_output")
            )
            if self.kind is FixtureKind.ADVERSARIAL:
                _refuse(
                    ToolReason.MISSING_ADVERSARIAL,
                    "adversarial fixtures must expect a typed refusal",
                )

    def to_record(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "input_value": _thaw(self.input_value),
            "expected_output": _thaw(self.expected_output),
            "expect_reject": self.expect_reject,
            "reason_code": self.reason_code,
            "kind": self.kind.value,
        }

    @classmethod
    def from_record(cls, payload: Mapping[str, Any] | ToolFixture) -> ToolFixture:
        if isinstance(payload, ToolFixture):
            return payload
        if not isinstance(payload, Mapping):
            raise ToolGrammarError("tool fixture must be a mapping")
        return cls(
            fixture_id=payload.get("fixture_id", ""),
            input_value=payload.get("input_value"),
            expected_output=payload.get("expected_output", ()),
            expect_reject=payload.get("expect_reject", False),
            reason_code=payload.get("reason_code", ""),
            kind=payload.get("kind", FixtureKind.TEST),
        )


def _fixtures(values: Any, field_name: str) -> tuple[ToolFixture, ...]:
    if values is None:
        raw: Sequence[Any] = ()
    elif _is_sequence(values):
        raw = values
    else:
        raise ToolGrammarError(f"{field_name} must be a sequence")
    if len(raw) > MAX_FIXTURES:
        _refuse(ToolReason.GRAMMAR_UNBOUNDED, f"{field_name} exceeds its fixture bound")
    result: list[ToolFixture] = []
    seen: set[str] = set()
    for item in raw:
        record = ToolFixture.from_record(item)
        if record.fixture_id in seen:
            raise ToolGrammarError(f"{field_name} contains a duplicate fixture_id")
        seen.add(record.fixture_id)
        result.append(record)
    return tuple(result)


@dataclass(frozen=True)
class TransformationTemplate:
    """Reviewed grammar/template entry.  Arbitrary generators cannot add entries."""

    template_id: str
    program: TransformationProgram
    optimized_translation_id: str
    resource_bound: ToolResourceBound
    tests: tuple[ToolFixture, ...]
    adversarial_fixtures: tuple[ToolFixture, ...]
    approved_repair_template_ids: tuple[str, ...] = ()
    risk_class: RiskClass = RiskClass.OBSERVATION_ONLY

    def __post_init__(self) -> None:
        object.__setattr__(self, "template_id", _identifier(self.template_id, "template_id"))
        object.__setattr__(
            self, "program", TransformationProgram.from_record(self.program)
        )
        object.__setattr__(
            self,
            "optimized_translation_id",
            _identifier(self.optimized_translation_id, "optimized_translation_id"),
        )
        object.__setattr__(
            self,
            "resource_bound",
            ToolResourceBound.from_record(self.resource_bound),
        )
        tests = _fixtures(self.tests, "tests")
        adversarial = _fixtures(self.adversarial_fixtures, "adversarial_fixtures")
        if not tests or any(item.expect_reject or item.kind is FixtureKind.ADVERSARIAL for item in tests):
            _refuse(ToolReason.MISSING_TESTS, "reviewed templates require non-rejecting test fixtures")
        if not adversarial or any(
            not item.expect_reject or item.kind is not FixtureKind.ADVERSARIAL for item in adversarial
        ):
            _refuse(
                ToolReason.MISSING_ADVERSARIAL,
                "reviewed templates require rejecting adversarial fixtures",
            )
        object.__setattr__(self, "tests", tests)
        object.__setattr__(self, "adversarial_fixtures", adversarial)
        object.__setattr__(
            self,
            "approved_repair_template_ids",
            _strings(
                self.approved_repair_template_ids,
                "approved_repair_template_ids",
                identifiers=True,
            ),
        )
        unknown = set(self.approved_repair_template_ids).difference(APPROVED_REPAIR_TEMPLATE_IDS)
        if unknown:
            _refuse(ToolReason.UNAPPROVED_TEMPLATE, "template references an unapproved repair template")
        used = {
            item.operand
            for item in self.program.operations
            if item.opcode is TransformationOpcode.MAP_APPROVED_TEMPLATE
        }
        if used and not set(used).issubset(set(self.approved_repair_template_ids)):
            _refuse(ToolReason.UNAPPROVED_TEMPLATE, "program uses an undeclared repair template")
        object.__setattr__(
            self, "risk_class", _enum(self.risk_class, RiskClass, "risk_class")
        )
        if self.risk_class is not RISK_CEILING:
            _refuse(ToolReason.RISK_CEILING, "generated tools cannot exceed observation-only risk")

    @property
    def fingerprint(self) -> str:
        return self.program.fingerprint

    def to_record(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "program": self.program.to_record(),
            "optimized_translation_id": self.optimized_translation_id,
            "resource_bound": self.resource_bound.to_record(),
            "tests": tuple(item.to_record() for item in self.tests),
            "adversarial_fixtures": tuple(item.to_record() for item in self.adversarial_fixtures),
            "approved_repair_template_ids": self.approved_repair_template_ids,
            "risk_class": self.risk_class.value,
        }


@dataclass(frozen=True)
class Interpretation:
    """Deterministic interpretation result.  Not an authority or proof."""

    value: Any
    paths_read: tuple[str, ...]
    operations_executed: int
    output_bytes: int
    item_count: int
    effects: tuple[EffectClass, ...] = (EffectClass.OBSERVE,)
    translation_kind: TranslationKind = TranslationKind.INTERPRETED_DSL

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _freeze_input(self.value, "value"))
        object.__setattr__(
            self, "paths_read", _strings(self.paths_read, "paths_read", paths=True)
        )
        object.__setattr__(
            self,
            "operations_executed",
            _nonnegative_int(self.operations_executed, "operations_executed", maximum=MAX_OPERATIONS),
        )
        object.__setattr__(
            self,
            "output_bytes",
            _nonnegative_int(self.output_bytes, "output_bytes", maximum=MAX_OUTPUT_BYTES),
        )
        object.__setattr__(
            self, "item_count", _nonnegative_int(self.item_count, "item_count", maximum=MAX_ITEMS)
        )
        object.__setattr__(
            self,
            "effects",
            _enums(self.effects, EffectClass, "effects", limit=len(EffectClass), required=True),
        )
        object.__setattr__(
            self,
            "translation_kind",
            _enum(self.translation_kind, TranslationKind, "translation_kind"),
        )


def _apply_project(value: Any, operand: Any) -> Any:
    fields = _strings(operand, "operand", identifiers=True, required=True)
    records = []
    for record in _require_records(value, "input"):
        records.append({key: record[key] for key in fields if key in record})
    return _restore_shape(value, records)


def _apply_rename(value: Any, operand: Any) -> Any:
    if not _is_mapping(operand) or not operand:
        _refuse(ToolReason.SCHEMA_MISMATCH, "rename operand must be a nonempty field mapping")
    mapping = { _identifier(key, "operand"): _identifier(item, "operand") for key, item in operand.items() }
    records = []
    for record in _require_records(value, "input"):
        renamed: dict[str, Any] = {}
        for key, item in record.items():
            renamed[mapping.get(key, key)] = item
        records.append(renamed)
    return _restore_shape(value, records)


def _apply_filter_equals(value: Any, field_name: str, operand: Any) -> Any:
    field_name = _identifier(field_name, "field")
    records = [record for record in _require_records(value, "input") if record.get(field_name) == operand]
    return tuple(records)


def _apply_filter_in(value: Any, field_name: str, operand: Any) -> Any:
    field_name = _identifier(field_name, "field")
    allowed = operand if _is_sequence(operand) else (operand,)
    records = [record for record in _require_records(value, "input") if record.get(field_name) in allowed]
    return tuple(records)


def _sort_key(record: Mapping[str, Any], fields: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(record.get(name, "")) for name in fields)


def _apply_sort(value: Any, operand: Any) -> Any:
    if _is_sequence(value) and value and all(isinstance(item, str) for item in value):
        return tuple(sorted(value))
    fields = operand if _is_sequence(operand) else (operand,)
    names = _strings(fields, "operand", identifiers=True, required=True)
    records = list(_require_records(value, "input"))
    records.sort(key=lambda record: _sort_key(record, names))
    return tuple(records)


def _apply_deduplicate(value: Any, field_name: str) -> Any:
    if _is_sequence(value) and value and all(isinstance(item, str) for item in value):
        return tuple(dict.fromkeys(value))
    key_name = _identifier(field_name, "field") if field_name else ""
    seen: set[Any] = set()
    records: list[Mapping[str, Any]] = []
    for record in _require_records(value, "input"):
        identity = record.get(key_name) if key_name else canonical_json_bytes(_thaw(record))
        if identity in seen:
            continue
        seen.add(identity)
        records.append(record)
    return tuple(records)


def _apply_join(value: Any, field_name: str, operand: Any) -> Any:
    separator = _join_separator(operand)
    if _is_sequence(value) and all(isinstance(item, str) for item in value):
        parts = _strings(value, "input", identifiers=True, required=True)
        return {"joined": separator.join(parts)}
    key_name = _identifier(field_name, "field")
    parts = []
    for record in _require_records(value, "input"):
        item = record.get(key_name, "")
        if item == "" or item is None:
            continue
        parts.append(_identifier(item, key_name))
    return {"joined": separator.join(parts)}


def _apply_clamp(value: Any, field_name: str, operand: Any) -> Any:
    field_name = _identifier(field_name, "field")
    if not _is_mapping(operand):
        _refuse(ToolReason.SCHEMA_MISMATCH, "clamp operand must declare min and max")
    minimum = _nonnegative_int(operand.get("min", 0), "min")
    maximum = _nonnegative_int(operand.get("max", 0), "max")
    if maximum < minimum:
        _refuse(ToolReason.GRAMMAR_UNBOUNDED, "clamp max must be at least min")

    def _clamp_record(record: Mapping[str, Any]) -> dict[str, Any]:
        current = record.get(field_name, minimum)
        number = _nonnegative_int(current, field_name)
        clamped = minimum if number < minimum else maximum if number > maximum else number
        updated = dict(record)
        updated[field_name] = clamped
        return updated

    records = [_clamp_record(record) for record in _require_records(value, "input")]
    return _restore_shape(value, records)


def _apply_select_scoped_paths(value: Any, field_name: str, scope_paths: Sequence[str]) -> Any:
    if _is_sequence(value) and value and all(isinstance(item, str) for item in value):
        _enforce_scope(tuple(value), scope_paths)
        kept = tuple(item for item in value if _path_is_within(item, scope_paths))
        return kept
    key_name = _identifier(field_name or "path", "field")
    records = []
    for record in _require_records(value, "input"):
        path = record.get(key_name)
        if not isinstance(path, str):
            _refuse(ToolReason.SCHEMA_MISMATCH, "scoped path records must declare a path field")
        _enforce_scope((path,), scope_paths)
        records.append(record)
    return _restore_shape(value, records)


def _apply_repair_template(template_id: str, value: Any, scope_paths: Sequence[str]) -> Any:
    if template_id == "repair.sort-declared-reads":
        if _is_sequence(value) and all(isinstance(item, str) for item in value):
            _enforce_scope(tuple(value), scope_paths)
            return tuple(sorted(dict.fromkeys(value)))
        records = []
        for record in _require_records(value, "input"):
            paths = record.get("paths", ())
            if not _is_sequence(paths):
                _refuse(ToolReason.SCHEMA_MISMATCH, "sort-declared-reads requires a paths sequence")
            path_values = _strings(paths, "paths", paths=True, required=True)
            _enforce_scope(path_values, scope_paths)
            updated = dict(record)
            updated["paths"] = tuple(sorted(dict.fromkeys(path_values)))
            records.append(updated)
        return _restore_shape(value, records)

    records = []
    for record in _require_records(value, "input"):
        updated = dict(record)
        module = _identifier(record.get("module", ""), "module")
        symbol = _identifier(record.get("symbol", record.get("name", "")), "symbol")
        if template_id == "repair.qualify-symbol":
            updated["qualified"] = f"{module}:{symbol}"
        elif template_id == "repair.normalize-import":
            updated["import_kind"] = "from-import"
            updated["module"] = module
            updated["name"] = symbol
        else:
            _refuse(ToolReason.UNAPPROVED_TEMPLATE, "repair template is not in the reviewed library")
        records.append(updated)
    return _restore_shape(value, records)


def _apply_op(
    operation: TransformationOp,
    value: Any,
    scope_paths: Sequence[str],
) -> Any:
    opcode = operation.opcode
    if opcode is TransformationOpcode.PROJECT:
        return _apply_project(value, operation.operand)
    if opcode is TransformationOpcode.RENAME:
        return _apply_rename(value, operation.operand)
    if opcode is TransformationOpcode.FILTER_EQUALS:
        return _apply_filter_equals(value, operation.field, operation.operand)
    if opcode is TransformationOpcode.FILTER_IN:
        return _apply_filter_in(value, operation.field, operation.operand)
    if opcode is TransformationOpcode.SORT:
        return _apply_sort(value, operation.operand)
    if opcode is TransformationOpcode.DEDUPLICATE:
        return _apply_deduplicate(value, operation.field)
    if opcode is TransformationOpcode.JOIN_IDENTIFIERS:
        return _apply_join(value, operation.field, operation.operand)
    if opcode is TransformationOpcode.CLAMP_INT:
        return _apply_clamp(value, operation.field, operation.operand)
    if opcode is TransformationOpcode.SELECT_SCOPED_PATHS:
        return _apply_select_scoped_paths(value, operation.field, scope_paths)
    if opcode is TransformationOpcode.MAP_APPROVED_TEMPLATE:
        return _apply_repair_template(str(operation.operand), value, scope_paths)
    _refuse(ToolReason.FORBIDDEN_OPCODE, "opcode is outside the closed transformation grammar")


def _finish_interpretation(
    value: Any,
    *,
    paths_read: Sequence[str],
    operations_executed: int,
    bound: ToolResourceBound,
    translation_kind: TranslationKind,
) -> Interpretation:
    frozen = _freeze_input(value, "output")
    output_bytes = _output_bytes(frozen)
    items = _item_count(frozen)
    path_reads = len(tuple(dict.fromkeys(paths_read)))
    if not bound.admits(
        operations=operations_executed,
        output_bytes=output_bytes,
        items=items,
        path_reads=path_reads,
    ):
        _refuse(ToolReason.RESOURCE_BOUND, "interpretation exceeded the declared resource bound")
    return Interpretation(
        value=frozen,
        paths_read=tuple(dict.fromkeys(paths_read)),
        operations_executed=operations_executed,
        output_bytes=output_bytes,
        item_count=items,
        translation_kind=translation_kind,
    )


def _opt_project_rename_sort(
    value: Any, program: TransformationProgram, bound: ToolResourceBound
) -> Interpretation:
    records = []
    paths: list[str] = []
    for record in _require_records(value, "input"):
        paths.extend(_record_paths(record))
        _enforce_scope(_record_paths(record), program.scope_paths)
        item: dict[str, Any] = {}
        if "module" in record:
            item["module"] = record["module"]
        if "name" in record:
            item["symbol"] = record["name"]
        elif "symbol" in record:
            item["symbol"] = record["symbol"]
        if "path" in record:
            item["path"] = record["path"]
        records.append(item)
    records.sort(key=lambda record: str(record.get("symbol", "")))
    return _finish_interpretation(
        tuple(records),
        paths_read=paths,
        operations_executed=len(program.operations),
        bound=bound,
        translation_kind=TranslationKind.OPTIMIZED_PYTHON,
    )


def _opt_filter_dedupe_join(
    value: Any, program: TransformationProgram, bound: ToolResourceBound
) -> Interpretation:
    allowed: Any = None
    field_name = "language"
    for operation in program.operations:
        if operation.opcode is TransformationOpcode.FILTER_IN:
            allowed = operation.operand
            field_name = operation.field
            break
    else:
        _refuse(ToolReason.TEMPLATE_MISMATCH, "optimized filter-join translation is missing filter-in")
    seen: set[Any] = set()
    parts: list[str] = []
    for record in _require_records(value, "input"):
        if record.get(field_name) not in (allowed if _is_sequence(allowed) else (allowed,)):
            continue
        module = record.get("module", "")
        if module in seen:
            continue
        seen.add(module)
        parts.append(_identifier(module, "module"))
    return _finish_interpretation(
        {"joined": ":".join(parts)},
        paths_read=_check_value_scope(value, program.scope_paths),
        operations_executed=len(program.operations),
        bound=bound,
        translation_kind=TranslationKind.OPTIMIZED_PYTHON,
    )


def _opt_qualify_in_scope(
    value: Any, program: TransformationProgram, bound: ToolResourceBound
) -> Interpretation:
    records = []
    paths: list[str] = []
    for record in _require_records(value, "input"):
        path = record.get("path")
        if not isinstance(path, str):
            _refuse(ToolReason.SCHEMA_MISMATCH, "qualify-in-scope records must declare path")
        _enforce_scope((path,), program.scope_paths)
        paths.append(path)
        module = _identifier(record.get("module", ""), "module")
        symbol = _identifier(record.get("symbol", record.get("name", "")), "symbol")
        updated = dict(record)
        updated["qualified"] = f"{module}:{symbol}"
        records.append(updated)
    return _finish_interpretation(
        _restore_shape(value, records),
        paths_read=paths,
        operations_executed=len(program.operations),
        bound=bound,
        translation_kind=TranslationKind.OPTIMIZED_PYTHON,
    )


_OPTIMIZED_EVALUATORS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "opt.project-rename-sort": _opt_project_rename_sort,
        "opt.filter-dedupe-join": _opt_filter_dedupe_join,
        "opt.qualify-in-scope": _opt_qualify_in_scope,
    }
)


class TransformationDsl:
    """Parser and interpreter for the reviewed bounded transformation grammar."""

    revision: ClassVar[str] = DSL_REVISION
    grammar_revision: ClassVar[str] = GRAMMAR_REVISION

    def parse(self, program: TransformationProgram | Mapping[str, Any]) -> TransformationProgram:
        parsed = TransformationProgram.from_record(program)
        if len(parsed.operations) > MAX_OPERATIONS:
            _refuse(ToolReason.GRAMMAR_UNBOUNDED, "program exceeds the grammar operation bound")
        return parsed

    def fingerprint(self, program: TransformationProgram | Mapping[str, Any]) -> str:
        return self.parse(program).fingerprint

    def interpret(
        self,
        program: TransformationProgram | Mapping[str, Any],
        input_value: Any,
        *,
        resource_bound: ToolResourceBound | Mapping[str, Any] | None = None,
    ) -> Interpretation:
        parsed = self.parse(program)
        bound = ToolResourceBound.from_record(resource_bound or ToolResourceBound())
        current = _freeze_input(input_value, "input")
        paths = list(_check_value_scope(current, parsed.scope_paths))
        if not bound.admits(
            operations=0,
            output_bytes=_output_bytes(current),
            items=_item_count(current),
            path_reads=len(paths),
        ):
            _refuse(ToolReason.RESOURCE_BOUND, "input exceeds the declared resource bound")
        executed = 0
        for operation in parsed.operations:
            executed += 1
            if executed > bound.max_operations:
                _refuse(ToolReason.RESOURCE_BOUND, "interpretation exceeded the operation bound")
            try:
                current = _apply_op(operation, current, parsed.scope_paths)
            except ToolSynthesisRefusal:
                raise
            except ProcedureContractError as exc:
                _refuse(ToolReason.SCHEMA_MISMATCH, str(exc))
            paths.extend(_check_value_scope(current, parsed.scope_paths))
        return _finish_interpretation(
            current,
            paths_read=paths,
            operations_executed=executed,
            bound=bound,
            translation_kind=TranslationKind.INTERPRETED_DSL,
        )


DeterministicToolDsl = TransformationDsl


def _project_rename_sort_template() -> TransformationTemplate:
    program = TransformationProgram(
        operations=(
            TransformationOp(
                opcode=TransformationOpcode.PROJECT,
                operand=("module", "name", "path"),
            ),
            TransformationOp(
                opcode=TransformationOpcode.RENAME,
                operand={"name": "symbol"},
            ),
            TransformationOp(opcode=TransformationOpcode.SORT, operand="symbol"),
        ),
        input_schema_ref="schema.symbol-records.in",
        output_schema_ref="schema.symbol-records.out",
        scope_paths=(_SCOPE,),
    )
    in_scope_a = f"{_SCOPE}/contracts.py"
    in_scope_b = f"{_SCOPE}/tool_synthesis.py"
    return TransformationTemplate(
        template_id="tool.project-rename-sort",
        program=program,
        optimized_translation_id="opt.project-rename-sort",
        resource_bound=ToolResourceBound(max_items=8, max_operations=8),
        tests=(
            ToolFixture(
                fixture_id="test.project-rename-sort.basic",
                input_value=(
                    {
                        "module": "pkg.mod",
                        "name": "beta",
                        "path": in_scope_b,
                        "noise": "drop-me",
                    },
                    {
                        "module": "pkg.mod",
                        "name": "alpha",
                        "path": in_scope_a,
                    },
                ),
                expected_output=(
                    {"module": "pkg.mod", "path": in_scope_a, "symbol": "alpha"},
                    {"module": "pkg.mod", "path": in_scope_b, "symbol": "beta"},
                ),
            ),
        ),
        adversarial_fixtures=(
            ToolFixture(
                fixture_id="adv.project-rename-sort.path-escape",
                input_value=(
                    {
                        "module": "pkg.mod",
                        "name": "alpha",
                        "path": "tmp/escape.py",
                    },
                ),
                expect_reject=True,
                reason_code=ToolReason.PATH_ESCAPE.value,
                kind=FixtureKind.ADVERSARIAL,
            ),
        ),
    )


def _filter_dedupe_join_template() -> TransformationTemplate:
    program = TransformationProgram(
        operations=(
            TransformationOp(
                opcode=TransformationOpcode.FILTER_IN,
                field="language",
                operand=("python",),
            ),
            TransformationOp(opcode=TransformationOpcode.DEDUPLICATE, field="module"),
            TransformationOp(
                opcode=TransformationOpcode.JOIN_IDENTIFIERS,
                field="module",
                operand=":",
            ),
        ),
        input_schema_ref="schema.module-records.in",
        output_schema_ref="schema.joined-modules.out",
        scope_paths=(_SCOPE,),
    )
    return TransformationTemplate(
        template_id="tool.filter-dedupe-join",
        program=program,
        optimized_translation_id="opt.filter-dedupe-join",
        resource_bound=ToolResourceBound(max_items=8, max_operations=8),
        tests=(
            ToolFixture(
                fixture_id="test.filter-dedupe-join.basic",
                input_value=(
                    {"module": "pkg.a", "language": "python", "path": f"{_SCOPE}/a.py"},
                    {"module": "pkg.a", "language": "python", "path": f"{_SCOPE}/a.py"},
                    {"module": "pkg.b", "language": "python", "path": f"{_SCOPE}/b.py"},
                    {"module": "pkg.c", "language": "javascript", "path": f"{_SCOPE}/c.py"},
                ),
                expected_output={"joined": "pkg.a:pkg.b"},
            ),
        ),
        adversarial_fixtures=(
            ToolFixture(
                fixture_id="adv.filter-dedupe-join.non-record",
                input_value="not-a-record",
                expect_reject=True,
                reason_code=ToolReason.SCHEMA_MISMATCH.value,
                kind=FixtureKind.ADVERSARIAL,
            ),
        ),
    )


def _qualify_in_scope_template() -> TransformationTemplate:
    program = TransformationProgram(
        operations=(
            TransformationOp(opcode=TransformationOpcode.SELECT_SCOPED_PATHS, field="path"),
            TransformationOp(
                opcode=TransformationOpcode.MAP_APPROVED_TEMPLATE,
                operand="repair.qualify-symbol",
            ),
        ),
        input_schema_ref="schema.qualify.in",
        output_schema_ref="schema.qualify.out",
        scope_paths=(_SCOPE,),
    )
    path = f"{_SCOPE}/hole_resolution.py"
    return TransformationTemplate(
        template_id="tool.qualify-in-scope",
        program=program,
        optimized_translation_id="opt.qualify-in-scope",
        resource_bound=ToolResourceBound(max_items=8, max_operations=8),
        approved_repair_template_ids=("repair.qualify-symbol",),
        tests=(
            ToolFixture(
                fixture_id="test.qualify-in-scope.basic",
                input_value=(
                    {
                        "module": "pkg.mod",
                        "name": "HoleRequest",
                        "path": path,
                    },
                ),
                expected_output=(
                    {
                        "module": "pkg.mod",
                        "name": "HoleRequest",
                        "path": path,
                        "qualified": "pkg.mod:HoleRequest",
                    },
                ),
            ),
        ),
        adversarial_fixtures=(
            ToolFixture(
                fixture_id="adv.qualify-in-scope.path-escape",
                input_value=(
                    {
                        "module": "pkg.mod",
                        "name": "secret",
                        "path": "tmp/secret.py",
                    },
                ),
                expect_reject=True,
                reason_code=ToolReason.PATH_ESCAPE.value,
                kind=FixtureKind.ADVERSARIAL,
            ),
        ),
    )


def reviewed_template_library() -> tuple[TransformationTemplate, ...]:
    """Closed reviewed grammar/template library.  Callers cannot register new entries."""

    return (
        _project_rename_sort_template(),
        _filter_dedupe_join_template(),
        _qualify_in_scope_template(),
    )


def _lookup_template(
    *,
    fingerprint: str,
    template_id: str = "",
    library: Sequence[TransformationTemplate] | None = None,
) -> TransformationTemplate:
    templates = tuple(library) if library is not None else reviewed_template_library()
    if template_id:
        matches = [item for item in templates if item.template_id == template_id]
        if not matches:
            _refuse(ToolReason.UNKNOWN_TEMPLATE, "template_id is not in the reviewed library")
        template = matches[0]
        if template.fingerprint != fingerprint:
            _refuse(ToolReason.TEMPLATE_MISMATCH, "program does not match the reviewed template")
        return template
    matches = [item for item in templates if item.fingerprint == fingerprint]
    if not matches:
        _refuse(
            ToolReason.UNKNOWN_TEMPLATE,
            "program is not a reviewed grammar/template; arbitrary tools cannot be generated",
        )
    return matches[0]


@dataclass(frozen=True)
class GeneratedToolSpec(CanonicalContract):
    """Closed generated-tool specification.  Synthesis cannot promote it."""

    SCHEMA: ClassVar[str] = _schema_name("GeneratedToolSpec")

    bindings: ArtifactBindings
    tool_id: str
    template_id: str
    program: TransformationProgram
    optimized_translation_id: str
    resource_bound: ToolResourceBound
    test_ids: tuple[str, ...]
    adversarial_fixture_ids: tuple[str, ...]
    occurrence_count: int
    grammar_revision: str = GRAMMAR_REVISION
    dsl_revision: str = DSL_REVISION
    compiler_revision: str = COMPILER_REVISION
    state: ArtifactState = ArtifactState.CANDIDATE
    can_authorize: bool = False
    risk_class: RiskClass = RiskClass.OBSERVATION_ONLY

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        for name in ("tool_id", "template_id", "optimized_translation_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "program", TransformationProgram.from_record(self.program))
        object.__setattr__(
            self, "resource_bound", ToolResourceBound.from_record(self.resource_bound)
        )
        object.__setattr__(
            self, "test_ids", _strings(self.test_ids, "test_ids", identifiers=True, required=True)
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
        object.__setattr__(
            self,
            "occurrence_count",
            _positive_int(self.occurrence_count, "occurrence_count", maximum=MAX_OCCURRENCES),
        )
        if self.occurrence_count < MIN_REPETITIONS:
            _refuse(ToolReason.INSUFFICIENT_REPETITION, "tools require repeated transformations")
        for name, expected in (
            ("grammar_revision", GRAMMAR_REVISION),
            ("dsl_revision", DSL_REVISION),
            ("compiler_revision", COMPILER_REVISION),
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
            if getattr(self, name) != expected:
                raise ToolSynthesisError(f"{name} is not current")
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if self.state is not ArtifactState.CANDIDATE:
            _refuse(ToolReason.CANDIDATE_TIER_REQUIRED, "generated tool specs remain candidate-tier")
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        if self.can_authorize:
            _refuse(ToolReason.AUTHORITY_REJECTED, "generated tools cannot authorize")
        object.__setattr__(self, "risk_class", _enum(self.risk_class, RiskClass, "risk_class"))
        if self.risk_class is not RISK_CEILING:
            _refuse(ToolReason.RISK_CEILING, "generated tools cannot exceed observation-only risk")
        if self.program.input_schema_ref == "" or self.program.output_schema_ref == "":
            _refuse(ToolReason.MISSING_SCHEMA, "generated tools require closed input and output schemas")
        _bounded(self, "GeneratedToolSpec")

    @property
    def can_grant_authority(self) -> bool:
        return False

    @property
    def can_promote(self) -> bool:
        return False

    def missing_declaration_fields(self) -> tuple[str, ...]:
        present = {
            "schema": self.program.input_schema_ref and self.program.output_schema_ref,
            "effects": self.program.effect_classes,
            "path_limits": self.program.scope_paths,
            "resources": self.resource_bound.to_record(),
            "tests": self.test_ids,
            "adversarial_fixtures": self.adversarial_fixture_ids,
        }
        return tuple(name for name in REQUIRED_TOOL_DECLARATION_FIELDS if not present[name])

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "tool_id": self.tool_id,
            "template_id": self.template_id,
            "program": self.program.to_record(),
            "optimized_translation_id": self.optimized_translation_id,
            "resource_bound": self.resource_bound.to_record(),
            "test_ids": self.test_ids,
            "adversarial_fixture_ids": self.adversarial_fixture_ids,
            "occurrence_count": self.occurrence_count,
            "grammar_revision": GRAMMAR_REVISION,
            "dsl_revision": DSL_REVISION,
            "compiler_revision": COMPILER_REVISION,
            "state": ArtifactState.CANDIDATE.value,
            "can_authorize": False,
            "risk_class": RISK_CEILING.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GeneratedToolSpec:
        if not isinstance(payload, Mapping):
            raise ToolSynthesisError("GeneratedToolSpec payload must be a mapping")
        body = _unwrap_generic_envelope(payload, cls.SCHEMA)
        fields = (
            "bindings",
            "tool_id",
            "template_id",
            "program",
            "optimized_translation_id",
            "resource_bound",
            "test_ids",
            "adversarial_fixture_ids",
            "occurrence_count",
            "grammar_revision",
            "dsl_revision",
            "compiler_revision",
            "state",
            "can_authorize",
            "risk_class",
        )
        record = cls(**_decode_fields(body, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class GeneratedToolCandidate(CanonicalContract):
    """Synthesized tool.  Always a candidate until independent promotion authority."""

    SCHEMA: ClassVar[str] = _schema_name("GeneratedToolCandidate")

    bindings: ArtifactBindings
    tool_id: str
    spec_cid: str
    template_id: str
    program: TransformationProgram
    optimized_translation_id: str
    resource_bound: ToolResourceBound
    occurrence_count: int
    test_ids: tuple[str, ...]
    adversarial_fixture_ids: tuple[str, ...]
    optimized_status: TranslationStatus = TranslationStatus.CANDIDATE
    state: ArtifactState = ArtifactState.CANDIDATE
    can_authorize: bool = False
    compiler_revision: str = COMPILER_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        for name in ("tool_id", "spec_cid", "template_id", "optimized_translation_id"):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        object.__setattr__(self, "program", TransformationProgram.from_record(self.program))
        object.__setattr__(
            self, "resource_bound", ToolResourceBound.from_record(self.resource_bound)
        )
        object.__setattr__(
            self,
            "occurrence_count",
            _positive_int(self.occurrence_count, "occurrence_count", maximum=MAX_OCCURRENCES),
        )
        object.__setattr__(
            self, "test_ids", _strings(self.test_ids, "test_ids", identifiers=True, required=True)
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
        object.__setattr__(
            self,
            "optimized_status",
            _enum(self.optimized_status, TranslationStatus, "optimized_status"),
        )
        if self.optimized_status is TranslationStatus.PROMOTED:
            _refuse(
                ToolReason.PROMOTION_FORBIDDEN,
                "generated tool candidates cannot carry a promoted optimized translation",
            )
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if self.state is not ArtifactState.CANDIDATE:
            _refuse(ToolReason.CANDIDATE_TIER_REQUIRED, "generated tools remain candidate-tier")
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        if self.can_authorize:
            _refuse(ToolReason.AUTHORITY_REJECTED, "generated tool candidates cannot authorize")
        object.__setattr__(
            self, "compiler_revision", _identifier(self.compiler_revision, "compiler_revision")
        )
        if self.compiler_revision != COMPILER_REVISION:
            raise ToolSynthesisError("generated-tool compiler revision is not current")
        _bounded(self, "GeneratedToolCandidate")

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
            "template_id": self.template_id,
            "program": self.program.to_record(),
            "optimized_translation_id": self.optimized_translation_id,
            "resource_bound": self.resource_bound.to_record(),
            "occurrence_count": self.occurrence_count,
            "test_ids": self.test_ids,
            "adversarial_fixture_ids": self.adversarial_fixture_ids,
            "optimized_status": TranslationStatus.CANDIDATE.value,
            "state": ArtifactState.CANDIDATE.value,
            "can_authorize": False,
            "compiler_revision": COMPILER_REVISION,
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
            "template_id",
            "program",
            "optimized_translation_id",
            "resource_bound",
            "occurrence_count",
            "test_ids",
            "adversarial_fixture_ids",
            "optimized_status",
            "state",
            "can_authorize",
            "compiler_revision",
        )
        record = cls(**_decode_fields(body, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class TranslationValidationReceipt(CanonicalContract):
    """Independent DSL-versus-optimized differential result."""

    SCHEMA: ClassVar[str] = _schema_name("TranslationValidationReceipt")

    bindings: ArtifactBindings
    candidate_cid: str
    template_id: str
    equivalent: bool
    tests_passed: tuple[str, ...]
    adversarial_passed: tuple[str, ...]
    dsl_output_digest: str
    optimized_output_digest: str
    reason_code: ToolReason = ToolReason.ADMITTED
    validator_revision: str = VALIDATOR_REVISION
    state: ArtifactState = ArtifactState.CANDIDATE
    can_authorize: bool = False
    can_promote: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(self, "candidate_cid", _identifier(self.candidate_cid, "candidate_cid"))
        object.__setattr__(self, "template_id", _identifier(self.template_id, "template_id"))
        object.__setattr__(self, "equivalent", _bool(self.equivalent, "equivalent"))
        object.__setattr__(
            self,
            "tests_passed",
            _strings(self.tests_passed, "tests_passed", identifiers=True, required=True),
        )
        object.__setattr__(
            self,
            "adversarial_passed",
            _strings(
                self.adversarial_passed, "adversarial_passed", identifiers=True, required=True
            ),
        )
        object.__setattr__(
            self, "dsl_output_digest", _identifier(self.dsl_output_digest, "dsl_output_digest")
        )
        object.__setattr__(
            self,
            "optimized_output_digest",
            _identifier(self.optimized_output_digest, "optimized_output_digest"),
        )
        object.__setattr__(self, "reason_code", _enum(self.reason_code, ToolReason, "reason_code"))
        object.__setattr__(
            self, "validator_revision", _identifier(self.validator_revision, "validator_revision")
        )
        if self.validator_revision != VALIDATOR_REVISION:
            raise ToolTranslationError("translation validator revision is not current")
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if self.state is not ArtifactState.CANDIDATE:
            _refuse(ToolReason.CANDIDATE_TIER_REQUIRED, "validation receipts remain candidate-tier")
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        object.__setattr__(self, "can_promote", _bool(self.can_promote, "can_promote"))
        if self.can_authorize or self.can_promote:
            _refuse(ToolReason.AUTHORITY_REJECTED, "translation validation cannot authorize or promote")
        if self.equivalent:
            if self.reason_code is not ToolReason.ADMITTED:
                raise ToolTranslationError("equivalent translations must use the admitted reason")
            if self.dsl_output_digest != self.optimized_output_digest:
                _refuse(ToolReason.TRANSLATION_MISMATCH, "equivalent translations must share an output digest")
        elif self.reason_code is ToolReason.ADMITTED:
            raise ToolTranslationError("non-equivalent translations cannot be admitted")
        _bounded(self, "TranslationValidationReceipt")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "candidate_cid": self.candidate_cid,
            "template_id": self.template_id,
            "equivalent": self.equivalent,
            "tests_passed": self.tests_passed,
            "adversarial_passed": self.adversarial_passed,
            "dsl_output_digest": self.dsl_output_digest,
            "optimized_output_digest": self.optimized_output_digest,
            "reason_code": self.reason_code.value,
            "validator_revision": VALIDATOR_REVISION,
            "state": ArtifactState.CANDIDATE.value,
            "can_authorize": False,
            "can_promote": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TranslationValidationReceipt:
        if not isinstance(payload, Mapping):
            raise ToolTranslationError("TranslationValidationReceipt payload must be a mapping")
        fields = (
            "bindings",
            "candidate_cid",
            "template_id",
            "equivalent",
            "tests_passed",
            "adversarial_passed",
            "dsl_output_digest",
            "optimized_output_digest",
            "reason_code",
            "validator_revision",
            "state",
            "can_authorize",
            "can_promote",
        )
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


def _certificate_statement(
    *,
    bindings: ArtifactBindings,
    tool_id: str,
    spec_cid: str,
    candidate_cid: str,
    template_id: str,
    validation_cid: str,
    test_ids: Sequence[str],
    adversarial_fixture_ids: Sequence[str],
    translation_equivalent: bool,
    issuer: str,
    issued_at_ms: int,
    expires_at_ms: int,
) -> dict[str, Any]:
    return {
        "schema": _schema_name("GeneratedToolCertificate"),
        "contract_version": PROCEDURE_CONTRACT_VERSION,
        "bindings": bindings.to_dict(),
        "tool_id": tool_id,
        "spec_cid": spec_cid,
        "candidate_cid": candidate_cid,
        "template_id": template_id,
        "validation_cid": validation_cid,
        "test_ids": tuple(test_ids),
        "adversarial_fixture_ids": tuple(adversarial_fixture_ids),
        "translation_equivalent": translation_equivalent,
        "issuer": issuer,
        "issued_at_ms": issued_at_ms,
        "expires_at_ms": expires_at_ms,
        "state": ArtifactState.VERIFIED.value,
        "can_authorize": False,
        "can_promote": False,
        "validator_revision": VALIDATOR_REVISION,
    }


@dataclass(frozen=True)
class GeneratedToolCertificate(CanonicalContract):
    """Independent certificate.  Never grants tool promotion or authority."""

    SCHEMA: ClassVar[str] = _schema_name("GeneratedToolCertificate")

    bindings: ArtifactBindings
    tool_id: str
    spec_cid: str
    candidate_cid: str
    template_id: str
    validation_cid: str
    test_ids: tuple[str, ...]
    adversarial_fixture_ids: tuple[str, ...]
    translation_equivalent: bool
    issuer: str
    issued_at_ms: int
    expires_at_ms: int
    statement_digest: str = ""
    state: ArtifactState = ArtifactState.VERIFIED
    can_authorize: bool = False
    can_promote: bool = False
    validator_revision: str = VALIDATOR_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        for name in (
            "tool_id",
            "spec_cid",
            "candidate_cid",
            "template_id",
            "validation_cid",
            "issuer",
        ):
            object.__setattr__(self, name, _identifier(getattr(self, name), name))
        if self.issuer.lower() in _FORBIDDEN_SELF_ISSUERS:
            _refuse(ToolReason.SELF_ISSUED, "generated-tool certificates cannot be self-issued")
        object.__setattr__(
            self, "test_ids", _strings(self.test_ids, "test_ids", identifiers=True, required=True)
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
        object.__setattr__(
            self,
            "translation_equivalent",
            _bool(self.translation_equivalent, "translation_equivalent"),
        )
        if not self.translation_equivalent:
            _refuse(ToolReason.TRANSLATION_MISMATCH, "certificates require exact translation equivalence")
        object.__setattr__(
            self, "issued_at_ms", _nonnegative_int(self.issued_at_ms, "issued_at_ms")
        )
        object.__setattr__(
            self, "expires_at_ms", _positive_int(self.expires_at_ms, "expires_at_ms")
        )
        if self.expires_at_ms <= self.issued_at_ms:
            raise ToolCertificateError("certificate expiry must follow issuance")
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if self.state is ArtifactState.PROMOTED:
            _refuse(ToolReason.PROMOTION_FORBIDDEN, "certificates cannot promote generated tools")
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        object.__setattr__(self, "can_promote", _bool(self.can_promote, "can_promote"))
        if self.can_authorize or self.can_promote:
            _refuse(ToolReason.AUTHORITY_REJECTED, "certificates cannot authorize or promote")
        object.__setattr__(
            self, "validator_revision", _identifier(self.validator_revision, "validator_revision")
        )
        if self.validator_revision != VALIDATOR_REVISION:
            raise ToolCertificateError("certificate validator revision is not current")
        expected = content_identity(
            _certificate_statement(
                bindings=self.bindings,
                tool_id=self.tool_id,
                spec_cid=self.spec_cid,
                candidate_cid=self.candidate_cid,
                template_id=self.template_id,
                validation_cid=self.validation_cid,
                test_ids=self.test_ids,
                adversarial_fixture_ids=self.adversarial_fixture_ids,
                translation_equivalent=self.translation_equivalent,
                issuer=self.issuer,
                issued_at_ms=self.issued_at_ms,
                expires_at_ms=self.expires_at_ms,
            )
        )
        digest = self.statement_digest or expected
        object.__setattr__(self, "statement_digest", _identifier(digest, "statement_digest"))
        if self.statement_digest != expected:
            _refuse(ToolReason.FORGED_CERTIFICATE, "certificate statement digest does not match canonical content")
        _bounded(self, "GeneratedToolCertificate")

    @property
    def can_grant_authority(self) -> bool:
        return False

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "tool_id": self.tool_id,
            "spec_cid": self.spec_cid,
            "candidate_cid": self.candidate_cid,
            "template_id": self.template_id,
            "validation_cid": self.validation_cid,
            "test_ids": self.test_ids,
            "adversarial_fixture_ids": self.adversarial_fixture_ids,
            "translation_equivalent": True,
            "issuer": self.issuer,
            "issued_at_ms": self.issued_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "statement_digest": self.statement_digest,
            "state": self.state.value,
            "can_authorize": False,
            "can_promote": False,
            "validator_revision": VALIDATOR_REVISION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GeneratedToolCertificate:
        if not isinstance(payload, Mapping):
            raise ToolCertificateError("GeneratedToolCertificate payload must be a mapping")
        body = _unwrap_generic_envelope(payload, cls.SCHEMA)
        fields = (
            "bindings",
            "tool_id",
            "spec_cid",
            "candidate_cid",
            "template_id",
            "validation_cid",
            "test_ids",
            "adversarial_fixture_ids",
            "translation_equivalent",
            "issuer",
            "issued_at_ms",
            "expires_at_ms",
            "statement_digest",
            "state",
            "can_authorize",
            "can_promote",
            "validator_revision",
        )
        record = cls(**_decode_fields(body, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class ToolPromotionDecision(CanonicalContract):
    """Optimized-Python promotion decision.  The tool artifact stays a candidate."""

    SCHEMA: ClassVar[str] = _schema_name("ToolPromotionDecision")

    bindings: ArtifactBindings
    candidate_cid: str
    certificate_cid: str
    validation_cid: str
    action: ToolPromotionAction
    reason_code: ToolReason
    optimized_status: TranslationStatus
    tool_state: ArtifactState = ArtifactState.CANDIDATE
    can_authorize: bool = False
    can_promote_tool: bool = False
    compiler_revision: str = COMPILER_REVISION

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        for name in ("candidate_cid", "certificate_cid", "validation_cid"):
            object.__setattr__(
                self,
                name,
                _identifier(
                    getattr(self, name),
                    name,
                    required=self.action is ToolPromotionAction.PROMOTE_OPTIMIZED,
                ),
            )
        object.__setattr__(self, "action", _enum(self.action, ToolPromotionAction, "action"))
        object.__setattr__(self, "reason_code", _enum(self.reason_code, ToolReason, "reason_code"))
        object.__setattr__(
            self,
            "optimized_status",
            _enum(self.optimized_status, TranslationStatus, "optimized_status"),
        )
        object.__setattr__(self, "tool_state", _enum(self.tool_state, ArtifactState, "tool_state"))
        if self.tool_state is ArtifactState.PROMOTED:
            _refuse(ToolReason.PROMOTION_FORBIDDEN, "tool synthesis cannot promote the tool artifact")
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        object.__setattr__(
            self, "can_promote_tool", _bool(self.can_promote_tool, "can_promote_tool")
        )
        if self.can_authorize or self.can_promote_tool:
            _refuse(ToolReason.AUTHORITY_REJECTED, "promotion decisions cannot authorize or promote the tool")
        object.__setattr__(
            self, "compiler_revision", _identifier(self.compiler_revision, "compiler_revision")
        )
        if self.action is ToolPromotionAction.PROMOTE_OPTIMIZED:
            if self.optimized_status is not TranslationStatus.PROMOTED:
                raise ToolPromotionError("successful optimized promotion must mark optimized Python promoted")
            if self.reason_code is not ToolReason.ADMITTED:
                raise ToolPromotionError("successful optimized promotion must use the admitted reason")
        elif self.optimized_status is TranslationStatus.PROMOTED:
            _refuse(ToolReason.PROMOTION_FORBIDDEN, "refused decisions cannot mark optimized Python promoted")
        _bounded(self, "ToolPromotionDecision")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "candidate_cid": self.candidate_cid,
            "certificate_cid": self.certificate_cid,
            "validation_cid": self.validation_cid,
            "action": self.action.value,
            "reason_code": self.reason_code.value,
            "optimized_status": self.optimized_status.value,
            "tool_state": ArtifactState.CANDIDATE.value,
            "can_authorize": False,
            "can_promote_tool": False,
            "compiler_revision": COMPILER_REVISION,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ToolPromotionDecision:
        if not isinstance(payload, Mapping):
            raise ToolPromotionError("ToolPromotionDecision payload must be a mapping")
        fields = (
            "bindings",
            "candidate_cid",
            "certificate_cid",
            "validation_cid",
            "action",
            "reason_code",
            "optimized_status",
            "tool_state",
            "can_authorize",
            "can_promote_tool",
            "compiler_revision",
        )
        record = cls(**_decode_fields(payload, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


@dataclass(frozen=True)
class GeneratedToolInvocationReceipt(CanonicalContract):
    """One bounded invocation.  Results remain non-authoritative observations."""

    SCHEMA: ClassVar[str] = _schema_name("GeneratedToolInvocationReceipt")

    bindings: ArtifactBindings
    candidate_cid: str
    translation_kind: TranslationKind
    input_digest: str
    output_digest: str
    paths_read: tuple[str, ...]
    operations_executed: int
    refused: bool = False
    reason_code: str = ""
    state: ArtifactState = ArtifactState.CANDIDATE
    can_authorize: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "bindings", _bindings(self.bindings))
        object.__setattr__(self, "candidate_cid", _identifier(self.candidate_cid, "candidate_cid"))
        object.__setattr__(
            self,
            "translation_kind",
            _enum(self.translation_kind, TranslationKind, "translation_kind"),
        )
        object.__setattr__(self, "input_digest", _identifier(self.input_digest, "input_digest"))
        object.__setattr__(
            self, "output_digest", _identifier(self.output_digest, "output_digest", required=False)
        )
        object.__setattr__(
            self, "paths_read", _strings(self.paths_read, "paths_read", paths=True)
        )
        object.__setattr__(
            self,
            "operations_executed",
            _nonnegative_int(self.operations_executed, "operations_executed", maximum=MAX_OPERATIONS),
        )
        object.__setattr__(self, "refused", _bool(self.refused, "refused"))
        object.__setattr__(
            self, "reason_code", _identifier(self.reason_code, "reason_code", required=False)
        )
        object.__setattr__(self, "state", _enum(self.state, ArtifactState, "state"))
        if self.state is not ArtifactState.CANDIDATE:
            _refuse(ToolReason.CANDIDATE_TIER_REQUIRED, "invocation receipts remain candidate-tier")
        object.__setattr__(self, "can_authorize", _bool(self.can_authorize, "can_authorize"))
        if self.can_authorize:
            _refuse(ToolReason.AUTHORITY_REJECTED, "invocations cannot authorize")
        _bounded(self, "GeneratedToolInvocationReceipt")

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": PROCEDURE_CONTRACT_VERSION,
            "bindings": self.bindings,
            "candidate_cid": self.candidate_cid,
            "translation_kind": self.translation_kind.value,
            "input_digest": self.input_digest,
            "output_digest": self.output_digest,
            "paths_read": self.paths_read,
            "operations_executed": self.operations_executed,
            "refused": self.refused,
            "reason_code": self.reason_code,
            "state": ArtifactState.CANDIDATE.value,
            "can_authorize": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GeneratedToolInvocationReceipt:
        if not isinstance(payload, Mapping):
            raise ToolSynthesisError("GeneratedToolInvocationReceipt payload must be a mapping")
        body = _unwrap_generic_envelope(payload, cls.SCHEMA)
        fields = (
            "bindings",
            "candidate_cid",
            "translation_kind",
            "input_digest",
            "output_digest",
            "paths_read",
            "operations_executed",
            "refused",
            "reason_code",
            "state",
            "can_authorize",
        )
        record = cls(**_decode_fields(body, cls.SCHEMA, fields, cls.__name__))
        _verify_identity(payload, record)
        return record


class TranslationValidator:
    """Independent DSL-to-optimized differential validator and certificate issuer."""

    revision: ClassVar[str] = VALIDATOR_REVISION

    def __init__(
        self,
        *,
        dsl: TransformationDsl | None = None,
        library: Sequence[TransformationTemplate] | None = None,
        issuer: str = DEFAULT_CERTIFICATE_ISSUER,
    ) -> None:
        self._dsl = dsl or TransformationDsl()
        self._library = tuple(library) if library is not None else reviewed_template_library()
        issuer_id = _identifier(issuer, "issuer")
        if issuer_id.lower() in _FORBIDDEN_SELF_ISSUERS:
            _refuse(ToolReason.SELF_ISSUED, "translation validator issuer is not independent")
        self._issuer = issuer_id

    @property
    def issuer(self) -> str:
        return self._issuer

    def evaluate_optimized(
        self,
        candidate: GeneratedToolCandidate,
        input_value: Any,
        *,
        template: TransformationTemplate | None = None,
    ) -> Interpretation:
        if not isinstance(candidate, GeneratedToolCandidate):
            raise ToolTranslationError("optimized evaluation requires a GeneratedToolCandidate")
        resolved = template or _lookup_template(
            fingerprint=candidate.program.fingerprint,
            template_id=candidate.template_id,
            library=self._library,
        )
        if candidate.optimized_translation_id != resolved.optimized_translation_id:
            _refuse(
                ToolReason.TRANSLATION_MISMATCH,
                "candidate optimized translation is not the reviewed template translation",
            )
        evaluator = _OPTIMIZED_EVALUATORS.get(candidate.optimized_translation_id)
        if evaluator is None:
            _refuse(ToolReason.UNKNOWN_TEMPLATE, "optimized translation is not in the reviewed library")
        frozen = _freeze_input(input_value, "input")
        try:
            return evaluator(frozen, candidate.program, candidate.resource_bound)
        except ToolSynthesisRefusal:
            raise
        except ProcedureContractError as exc:
            _refuse(ToolReason.SCHEMA_MISMATCH, str(exc))

    def validate(
        self,
        candidate: GeneratedToolCandidate,
        *,
        template: TransformationTemplate | None = None,
    ) -> TranslationValidationReceipt:
        if not isinstance(candidate, GeneratedToolCandidate):
            raise ToolTranslationError("validation requires a GeneratedToolCandidate")
        resolved = template or _lookup_template(
            fingerprint=candidate.program.fingerprint,
            template_id=candidate.template_id,
            library=self._library,
        )
        if resolved.template_id != candidate.template_id:
            _refuse(ToolReason.TEMPLATE_MISMATCH, "validation template does not match the candidate")
        tests_passed: list[str] = []
        adversarial_passed: list[str] = []
        dsl_digest = ""
        optimized_digest = ""
        for fixture in resolved.tests:
            dsl_result = self._dsl.interpret(
                candidate.program, fixture.input_value, resource_bound=candidate.resource_bound
            )
            optimized_result = self.evaluate_optimized(
                candidate, fixture.input_value, template=resolved
            )
            if not _canonical_equal(dsl_result.value, optimized_result.value):
                _refuse(ToolReason.TRANSLATION_MISMATCH, "optimized Python is not differentially exact")
            if not _canonical_equal(dsl_result.value, fixture.expected_output):
                _refuse(ToolReason.TRANSLATION_MISMATCH, "translation does not match the reviewed test fixture")
            if set(dsl_result.effects) != set(optimized_result.effects):
                _refuse(ToolReason.EFFECT_ESCALATION, "optimized translation changed observable effects")
            tests_passed.append(fixture.fixture_id)
            dsl_digest = content_identity({"output": _thaw(dsl_result.value)})
            optimized_digest = content_identity({"output": _thaw(optimized_result.value)})
        for fixture in resolved.adversarial_fixtures:
            dsl_rejected = False
            optimized_rejected = False
            dsl_reason = ""
            optimized_reason = ""
            try:
                self._dsl.interpret(
                    candidate.program, fixture.input_value, resource_bound=candidate.resource_bound
                )
            except ToolSynthesisRefusal as exc:
                dsl_rejected = True
                dsl_reason = exc.reason_code.value
            try:
                self.evaluate_optimized(candidate, fixture.input_value, template=resolved)
            except ToolSynthesisRefusal as exc:
                optimized_rejected = True
                optimized_reason = exc.reason_code.value
            if not dsl_rejected or not optimized_rejected:
                _refuse(
                    ToolReason.MISSING_ADVERSARIAL,
                    "adversarial fixture was not refused by both translations",
                )
            expected_reason = fixture.reason_code
            if expected_reason and (
                dsl_reason != expected_reason or optimized_reason != expected_reason
            ):
                _refuse(
                    ToolReason.TRANSLATION_MISMATCH,
                    "adversarial refusal reasons are not differentially exact",
                )
            adversarial_passed.append(fixture.fixture_id)
        if not tests_passed:
            _refuse(ToolReason.MISSING_TESTS, "translation validation requires passing tests")
        if not adversarial_passed:
            _refuse(ToolReason.MISSING_ADVERSARIAL, "translation validation requires adversarial fixtures")
        if dsl_digest != optimized_digest:
            _refuse(ToolReason.TRANSLATION_MISMATCH, "translation output digests differ")
        return TranslationValidationReceipt(
            bindings=candidate.bindings,
            candidate_cid=candidate.content_id,
            template_id=candidate.template_id,
            equivalent=True,
            tests_passed=tuple(tests_passed),
            adversarial_passed=tuple(adversarial_passed),
            dsl_output_digest=dsl_digest,
            optimized_output_digest=optimized_digest,
        )

    def issue_certificate(
        self,
        candidate: GeneratedToolCandidate,
        validation: TranslationValidationReceipt,
        *,
        now_ms: int,
        expires_at_ms: int,
        issuer: str = "",
    ) -> GeneratedToolCertificate:
        if not isinstance(validation, TranslationValidationReceipt):
            _refuse(ToolReason.MISSING_VALIDATION, "certificate issuance requires independent validation")
        if validation.candidate_cid != candidate.content_id:
            _refuse(ToolReason.BINDING_MISMATCH, "validation receipt is not bound to the candidate")
        if validation.bindings != candidate.bindings:
            _refuse(ToolReason.BINDING_MISMATCH, "validation receipt bindings do not match the candidate")
        if not validation.equivalent:
            _refuse(ToolReason.TRANSLATION_MISMATCH, "certificates require exact differential equivalence")
        issuer_id = _identifier(issuer or self._issuer, "issuer")
        if issuer_id.lower() in _FORBIDDEN_SELF_ISSUERS:
            _refuse(ToolReason.SELF_ISSUED, "certificate issuer is not independent of the compiler")
        return GeneratedToolCertificate(
            bindings=candidate.bindings,
            tool_id=candidate.tool_id,
            spec_cid=candidate.spec_cid,
            candidate_cid=candidate.content_id,
            template_id=candidate.template_id,
            validation_cid=validation.content_id,
            test_ids=validation.tests_passed,
            adversarial_fixture_ids=validation.adversarial_passed,
            translation_equivalent=True,
            issuer=issuer_id,
            issued_at_ms=now_ms,
            expires_at_ms=expires_at_ms,
        )


class GeneratedToolCompiler:
    """Compiles repeated reviewed transformations into candidate tools only."""

    revision: ClassVar[str] = COMPILER_REVISION

    def __init__(
        self,
        *,
        dsl: TransformationDsl | None = None,
        validator: TranslationValidator | None = None,
        library: Sequence[TransformationTemplate] | None = None,
    ) -> None:
        self._dsl = dsl or TransformationDsl()
        self._library = tuple(library) if library is not None else reviewed_template_library()
        self._validator = validator or TranslationValidator(dsl=self._dsl, library=self._library)

    @property
    def dsl(self) -> TransformationDsl:
        return self._dsl

    @property
    def validator(self) -> TranslationValidator:
        return self._validator

    def compile(
        self,
        *,
        bindings: ArtifactBindings,
        program: TransformationProgram | Mapping[str, Any],
        occurrence_count: int | None = None,
        observed_programs: Sequence[TransformationProgram | Mapping[str, Any]] = (),
        template_id: str = "",
        tool_id: str = "",
    ) -> GeneratedToolCandidate:
        parsed = self._dsl.parse(program)
        if observed_programs:
            fingerprints = tuple(self._dsl.fingerprint(item) for item in observed_programs)
            if any(item != parsed.fingerprint for item in fingerprints):
                _refuse(ToolReason.TEMPLATE_MISMATCH, "observed programs are not identical")
            count = len(fingerprints)
        elif occurrence_count is None or type(occurrence_count) is not int:
            _refuse(
                ToolReason.INSUFFICIENT_REPETITION,
                "repeated transformations are required before a candidate tool may be synthesized",
            )
        else:
            count = occurrence_count
        if count < MIN_REPETITIONS:
            _refuse(
                ToolReason.INSUFFICIENT_REPETITION,
                "repeated transformations are required before a candidate tool may be synthesized",
            )
        count = _positive_int(count, "occurrence_count", maximum=MAX_OCCURRENCES)
        template = _lookup_template(
            fingerprint=parsed.fingerprint,
            template_id=template_id,
            library=self._library,
        )
        missing = []
        if not parsed.input_schema_ref or not parsed.output_schema_ref:
            missing.append("schema")
        if not parsed.effect_classes:
            missing.append("effects")
        if not parsed.scope_paths:
            missing.append("path_limits")
        if not template.tests:
            missing.append("tests")
        if not template.adversarial_fixtures:
            missing.append("adversarial_fixtures")
        if missing:
            _refuse(ToolReason.MISSING_SCHEMA, "generated tools require closed schemas, effects, paths, tests, and fixtures")
        assigned_id = tool_id or f"tool.{template.template_id}.{count}"
        spec = GeneratedToolSpec(
            bindings=bindings,
            tool_id=_identifier(assigned_id, "tool_id"),
            template_id=template.template_id,
            program=parsed,
            optimized_translation_id=template.optimized_translation_id,
            resource_bound=template.resource_bound,
            test_ids=tuple(item.fixture_id for item in template.tests),
            adversarial_fixture_ids=tuple(item.fixture_id for item in template.adversarial_fixtures),
            occurrence_count=count,
        )
        if spec.missing_declaration_fields():
            _refuse(
                ToolReason.MISSING_SCHEMA,
                "generated tool omitted a required declaration field",
            )
        return GeneratedToolCandidate(
            bindings=bindings,
            tool_id=spec.tool_id,
            spec_cid=spec.content_id,
            template_id=spec.template_id,
            program=parsed,
            optimized_translation_id=spec.optimized_translation_id,
            resource_bound=spec.resource_bound,
            occurrence_count=count,
            test_ids=spec.test_ids,
            adversarial_fixture_ids=spec.adversarial_fixture_ids,
        )

    def promote_optimized(
        self,
        candidate: GeneratedToolCandidate,
        validation: TranslationValidationReceipt | None,
        certificate: GeneratedToolCertificate | None,
        *,
        now_ms: int = 0,
    ) -> ToolPromotionDecision:
        def _refused(reason: ToolReason) -> ToolPromotionDecision:
            return ToolPromotionDecision(
                bindings=candidate.bindings,
                candidate_cid=candidate.content_id,
                certificate_cid=certificate.content_id if isinstance(certificate, GeneratedToolCertificate) else "",
                validation_cid=validation.content_id if isinstance(validation, TranslationValidationReceipt) else "",
                action=ToolPromotionAction.REFUSE,
                reason_code=reason,
                optimized_status=TranslationStatus.REJECTED
                if reason is not ToolReason.CANDIDATE_TIER_REQUIRED
                else TranslationStatus.CANDIDATE,
            )

        if not isinstance(candidate, GeneratedToolCandidate):
            raise ToolPromotionError("optimized promotion requires a GeneratedToolCandidate")
        if candidate.state is not ArtifactState.CANDIDATE:
            return _refused(ToolReason.CANDIDATE_TIER_REQUIRED)
        if not isinstance(validation, TranslationValidationReceipt):
            return _refused(ToolReason.MISSING_VALIDATION)
        if not isinstance(certificate, GeneratedToolCertificate):
            return _refused(ToolReason.MISSING_CERTIFICATE)
        if validation.candidate_cid != candidate.content_id or certificate.candidate_cid != candidate.content_id:
            return _refused(ToolReason.BINDING_MISMATCH)
        if validation.bindings != candidate.bindings or certificate.bindings != candidate.bindings:
            return _refused(ToolReason.BINDING_MISMATCH)
        if certificate.validation_cid != validation.content_id:
            return _refused(ToolReason.BINDING_MISMATCH)
        if not validation.equivalent or not certificate.translation_equivalent:
            return _refused(ToolReason.TRANSLATION_MISMATCH)
        if certificate.can_promote or certificate.can_authorize:
            return _refused(ToolReason.PROMOTION_FORBIDDEN)
        if now_ms and certificate.expires_at_ms <= now_ms:
            return _refused(ToolReason.STALE_CERTIFICATE)
        if certificate.issuer.lower() in _FORBIDDEN_SELF_ISSUERS:
            return _refused(ToolReason.SELF_ISSUED)
        return ToolPromotionDecision(
            bindings=candidate.bindings,
            candidate_cid=candidate.content_id,
            certificate_cid=certificate.content_id,
            validation_cid=validation.content_id,
            action=ToolPromotionAction.PROMOTE_OPTIMIZED,
            reason_code=ToolReason.ADMITTED,
            optimized_status=TranslationStatus.PROMOTED,
        )

    def invoke(
        self,
        candidate: GeneratedToolCandidate,
        input_value: Any,
        *,
        translation: TranslationKind | str = TranslationKind.INTERPRETED_DSL,
        promotion: ToolPromotionDecision | None = None,
        certificate: GeneratedToolCertificate | None = None,
    ) -> GeneratedToolInvocationReceipt:
        if not isinstance(candidate, GeneratedToolCandidate):
            raise ToolSynthesisError("invocation requires a GeneratedToolCandidate")
        kind = _enum(translation, TranslationKind, "translation")
        frozen = _freeze_input(input_value, "input")
        input_digest = content_identity({"input": _thaw(frozen)})
        try:
            if kind is TranslationKind.OPTIMIZED_PYTHON:
                if not isinstance(promotion, ToolPromotionDecision):
                    _refuse(
                        ToolReason.OPTIMIZED_NOT_PROMOTED,
                        "optimized Python cannot run before exact validation and certificate",
                    )
                if promotion.action is not ToolPromotionAction.PROMOTE_OPTIMIZED:
                    _refuse(promotion.reason_code, "optimized Python promotion was refused")
                if promotion.candidate_cid != candidate.content_id:
                    _refuse(ToolReason.BINDING_MISMATCH, "promotion decision is not bound to the candidate")
                if not isinstance(certificate, GeneratedToolCertificate):
                    _refuse(ToolReason.MISSING_CERTIFICATE, "optimized invocation requires the bound certificate")
                if certificate.content_id != promotion.certificate_cid:
                    _refuse(ToolReason.BINDING_MISMATCH, "promotion certificate does not match the decision")
                result = self._validator.evaluate_optimized(candidate, frozen)
            else:
                result = self._dsl.interpret(
                    candidate.program, frozen, resource_bound=candidate.resource_bound
                )
        except ToolSynthesisRefusal as exc:
            return GeneratedToolInvocationReceipt(
                bindings=candidate.bindings,
                candidate_cid=candidate.content_id,
                translation_kind=kind,
                input_digest=input_digest,
                output_digest="",
                paths_read=(),
                operations_executed=0,
                refused=True,
                reason_code=exc.reason_code.value,
            )
        return GeneratedToolInvocationReceipt(
            bindings=candidate.bindings,
            candidate_cid=candidate.content_id,
            translation_kind=kind,
            input_digest=input_digest,
            output_digest=content_identity({"output": _thaw(result.value)}),
            paths_read=result.paths_read,
            operations_executed=result.operations_executed,
        )


for _artifact_type in (
    GeneratedToolSpec,
    GeneratedToolCandidate,
    GeneratedToolCertificate,
    GeneratedToolInvocationReceipt,
    TranslationValidationReceipt,
    ToolPromotionDecision,
):
    ARTIFACT_TYPES_BY_SCHEMA[_artifact_type.SCHEMA] = _artifact_type


__all__ = [
    "ALLOWED_EFFECTS",
    "APPROVED_REPAIR_TEMPLATE_IDS",
    "COMPILER_REVISION",
    "DEFAULT_CERTIFICATE_ISSUER",
    "DSL_REVISION",
    "GRAMMAR_REVISION",
    "MAX_OPERATIONS",
    "MIN_REPETITIONS",
    "REQUIRED_TOOL_DECLARATION_FIELDS",
    "RISK_CEILING",
    "VALIDATOR_REVISION",
    "DeterministicToolDsl",
    "FixtureKind",
    "GeneratedToolCandidate",
    "GeneratedToolCertificate",
    "GeneratedToolCompiler",
    "GeneratedToolInvocationReceipt",
    "GeneratedToolSpec",
    "Interpretation",
    "ToolCertificateError",
    "ToolFixture",
    "ToolGrammarError",
    "ToolPromotionAction",
    "ToolPromotionDecision",
    "ToolPromotionError",
    "ToolReason",
    "ToolResourceBound",
    "ToolSynthesisError",
    "ToolSynthesisRefusal",
    "ToolTranslationError",
    "TransformationDsl",
    "TransformationOp",
    "TransformationOpcode",
    "TransformationProgram",
    "TransformationTemplate",
    "TranslationKind",
    "TranslationStatus",
    "TranslationValidationReceipt",
    "TranslationValidator",
    "reviewed_template_library",
]
