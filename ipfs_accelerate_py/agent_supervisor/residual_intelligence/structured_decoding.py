"""Grammar-constrained parsing for compact residual expert outputs."""

# Python 3.8 support requires ``str, Enum`` rather than ``enum.StrEnum``.
# ruff: noqa: UP042

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Final

from .contracts import (
    ResidualIntelligenceError,
    ResidualTaskFamily,
    bounded_int,
    bounded_json_mapping,
    canonical_id,
    required_text,
    text_tuple,
)
from .residual_ir import ResidualTaskOutput

EXPERT_GRAMMAR_SCHEMA: Final = "ipfs_accelerate_py/agent-supervisor/residual-expert-grammar@1"
STRUCTURED_DECODE_RESULT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-structured-decode-result@1"
)
MAX_STRUCTURED_OUTPUT_BYTES: Final = 32_768
MAX_PAYLOAD_TOKEN_BYTES: Final = 512
MAX_PATCH_PATH_BYTES: Final = 1_024
MAX_PATCH_CHANGED_LINES: Final = 10_000
MAX_SCORE_PPM: Final = 1_000_000
PAYLOAD_FIELD_CONTRACT_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/residual-payload-field-contract@1"
)
_PAYLOAD_TOKEN_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+#=~-]*$")


class DecodeStatus(str, Enum):
    VALID = "valid"
    INVALID_OUTPUT = "invalid_output"


class PayloadFieldKind(str, Enum):
    """Closed JSON value kinds permitted inside a specialist payload."""

    TOKEN = "token"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    TOKEN_LIST = "token_list"
    INTEGER_LIST = "integer_list"
    RELATIVE_PATH_LIST = "relative_path_list"


@dataclass(frozen=True)
class PayloadFieldContract:
    """Exact type and bounds for one named payload field."""

    kind: PayloadFieldKind
    maximum_text_bytes: int = MAX_PAYLOAD_TOKEN_BYTES
    maximum_items: int = 1
    minimum_integer: int = 0
    maximum_integer: int = MAX_SCORE_PPM
    allow_empty: bool = False
    unique_items: bool = True
    allowed_values: tuple[str, ...] = ()
    schema: str = PAYLOAD_FIELD_CONTRACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != PAYLOAD_FIELD_CONTRACT_SCHEMA:
            raise ResidualIntelligenceError("unsupported payload field contract schema")
        object.__setattr__(self, "kind", PayloadFieldKind(self.kind))
        object.__setattr__(
            self,
            "maximum_text_bytes",
            bounded_int(
                self.maximum_text_bytes,
                "maximum_text_bytes",
                minimum=1,
                maximum=MAX_STRUCTURED_OUTPUT_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "maximum_items",
            bounded_int(self.maximum_items, "maximum_items", minimum=1, maximum=1024),
        )
        if type(self.minimum_integer) is not int or type(self.maximum_integer) is not int:
            raise ResidualIntelligenceError("payload integer bounds must be integers")
        if self.minimum_integer > self.maximum_integer:
            raise ResidualIntelligenceError("payload integer bounds are inverted")
        if type(self.allow_empty) is not bool or type(self.unique_items) is not bool:
            raise ResidualIntelligenceError("payload field flags must be boolean")
        choices = text_tuple(self.allowed_values, "allowed_values", max_items=256)
        if choices and self.kind is not PayloadFieldKind.TOKEN:
            raise ResidualIntelligenceError("allowed_values apply only to token fields")
        object.__setattr__(self, "allowed_values", choices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind.value,
            "maximum_text_bytes": self.maximum_text_bytes,
            "maximum_items": self.maximum_items,
            "minimum_integer": self.minimum_integer,
            "maximum_integer": self.maximum_integer,
            "allow_empty": self.allow_empty,
            "unique_items": self.unique_items,
            "allowed_values": list(self.allowed_values),
        }


@dataclass(frozen=True)
class ExpertGrammar:
    """Closed outer and payload grammar for one residual task family."""

    task_family: ResidualTaskFamily
    output_classes: tuple[str, ...]
    payload_fields: tuple[str, ...]
    required_payload_fields: tuple[str, ...]
    field_contracts: Mapping[str, PayloadFieldContract]
    aligned_list_fields: tuple[tuple[str, str], ...] = ()
    descending_integer_list_fields: tuple[str, ...] = ()
    maximum_output_bytes: int = 4096
    maximum_list_items: int = 64
    abstention_output_class: str = "ABSTAIN"
    schema: str = EXPERT_GRAMMAR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EXPERT_GRAMMAR_SCHEMA:
            raise ResidualIntelligenceError("unsupported expert grammar schema")
        object.__setattr__(self, "task_family", ResidualTaskFamily(self.task_family))
        object.__setattr__(
            self,
            "output_classes",
            text_tuple(self.output_classes, "output_classes", allow_empty=False),
        )
        object.__setattr__(
            self, "payload_fields", text_tuple(self.payload_fields, "payload_fields")
        )
        object.__setattr__(
            self,
            "required_payload_fields",
            text_tuple(self.required_payload_fields, "required_payload_fields"),
        )
        if not set(self.required_payload_fields).issubset(self.payload_fields):
            raise ResidualIntelligenceError("required payload fields are not declared")
        if not isinstance(self.field_contracts, Mapping):
            raise ResidualIntelligenceError("field_contracts must be an object")
        contracts: dict[str, PayloadFieldContract] = {}
        for key, contract in self.field_contracts.items():
            name = required_text(key, "field contract", max_bytes=256)
            if not isinstance(contract, PayloadFieldContract):
                raise ResidualIntelligenceError(f"field contract {name} is not typed")
            contracts[name] = contract
        if set(contracts) != set(self.payload_fields):
            raise ResidualIntelligenceError(
                "every payload field must have exactly one field contract"
            )
        object.__setattr__(self, "field_contracts", contracts)
        aligned: list[tuple[str, str]] = []
        for pair in self.aligned_list_fields:
            if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
                raise ResidualIntelligenceError("aligned_list_fields entries must be field pairs")
            left, right = (required_text(item, "aligned field") for item in pair)
            if left not in contracts or right not in contracts or left == right:
                raise ResidualIntelligenceError("aligned fields must name distinct payload fields")
            list_kinds = {
                PayloadFieldKind.TOKEN_LIST,
                PayloadFieldKind.INTEGER_LIST,
                PayloadFieldKind.RELATIVE_PATH_LIST,
            }
            if contracts[left].kind not in list_kinds or contracts[right].kind not in list_kinds:
                raise ResidualIntelligenceError("aligned fields must both be lists")
            aligned.append((left, right))
        object.__setattr__(self, "aligned_list_fields", tuple(aligned))
        descending = text_tuple(
            self.descending_integer_list_fields,
            "descending_integer_list_fields",
        )
        for name in descending:
            if name not in contracts or contracts[name].kind is not PayloadFieldKind.INTEGER_LIST:
                raise ResidualIntelligenceError(
                    "descending field must name an integer-list payload field"
                )
        object.__setattr__(self, "descending_integer_list_fields", descending)
        object.__setattr__(
            self,
            "maximum_output_bytes",
            bounded_int(
                self.maximum_output_bytes,
                "maximum_output_bytes",
                minimum=128,
                maximum=MAX_STRUCTURED_OUTPUT_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "maximum_list_items",
            bounded_int(
                self.maximum_list_items,
                "maximum_list_items",
                minimum=1,
                maximum=1024,
            ),
        )
        list_kinds = {
            PayloadFieldKind.TOKEN_LIST,
            PayloadFieldKind.INTEGER_LIST,
            PayloadFieldKind.RELATIVE_PATH_LIST,
        }
        if any(
            contract.kind in list_kinds and contract.maximum_items > self.maximum_list_items
            for contract in self.field_contracts.values()
        ):
            raise ResidualIntelligenceError(
                "payload field item bound exceeds the grammar list bound"
            )
        object.__setattr__(
            self,
            "abstention_output_class",
            required_text(self.abstention_output_class, "abstention_output_class"),
        )
        if self.abstention_output_class not in self.output_classes:
            raise ResidualIntelligenceError("abstention output class is not allowed")

    @property
    def grammar_id(self) -> str:
        return canonical_id(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task_family": self.task_family.value,
            "output_classes": list(self.output_classes),
            "payload_fields": list(self.payload_fields),
            "required_payload_fields": list(self.required_payload_fields),
            "field_contracts": {
                key: value.to_dict() for key, value in sorted(self.field_contracts.items())
            },
            "aligned_list_fields": [list(pair) for pair in self.aligned_list_fields],
            "descending_integer_list_fields": list(self.descending_integer_list_fields),
            "maximum_output_bytes": self.maximum_output_bytes,
            "maximum_list_items": self.maximum_list_items,
            "abstention_output_class": self.abstention_output_class,
        }


@dataclass(frozen=True)
class StructuredDecodeResult:
    status: DecodeStatus
    grammar_id: str
    output: ResidualTaskOutput | None
    reason_codes: tuple[str, ...]
    schema: str = STRUCTURED_DECODE_RESULT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != STRUCTURED_DECODE_RESULT_SCHEMA:
            raise ResidualIntelligenceError("unsupported structured decode result schema")
        object.__setattr__(self, "status", DecodeStatus(self.status))
        object.__setattr__(self, "grammar_id", required_text(self.grammar_id, "grammar_id"))
        object.__setattr__(
            self, "reason_codes", text_tuple(self.reason_codes, "reason_codes", max_items=16)
        )
        if self.status is DecodeStatus.VALID and not isinstance(self.output, ResidualTaskOutput):
            raise ResidualIntelligenceError("valid decode requires a typed output")
        if self.status is DecodeStatus.INVALID_OUTPUT and self.output is not None:
            raise ResidualIntelligenceError("invalid_output cannot carry a parsed candidate")
        if self.status is DecodeStatus.INVALID_OUTPUT and not self.reason_codes:
            raise ResidualIntelligenceError("invalid_output requires reason codes")

    @property
    def result_id(self) -> str:
        return canonical_id(self.to_dict(include_id=False))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": self.schema,
            "status": self.status.value,
            "grammar_id": self.grammar_id,
            "output": None if self.output is None else self.output.to_dict(),
            "reason_codes": list(self.reason_codes),
        }
        if include_id:
            result["result_id"] = self.result_id
        return result


def _closed_json_loads(raw: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ResidualIntelligenceError(f"duplicate JSON field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=reject_duplicates)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ResidualIntelligenceError("output is not one strict JSON object") from exc
    if not isinstance(value, Mapping):
        raise ResidualIntelligenceError("output root must be an object")
    return value


def _bounded_payload_token(value: Any, *, name: str, maximum_bytes: int) -> str:
    token = required_text(value, name, max_bytes=maximum_bytes)
    if "\n" in token or _PAYLOAD_TOKEN_RE.fullmatch(token) is None:
        raise ResidualIntelligenceError(f"{name} must be one bounded identifier token")
    return token


def _validate_field_value(
    value: Any,
    *,
    name: str,
    contract: PayloadFieldContract,
) -> Any:
    kind = contract.kind
    if kind is PayloadFieldKind.TOKEN:
        token = _bounded_payload_token(
            value,
            name=name,
            maximum_bytes=contract.maximum_text_bytes,
        )
        if contract.allowed_values and token not in contract.allowed_values:
            raise ResidualIntelligenceError(f"{name} is outside its closed enumeration")
        return token
    if kind is PayloadFieldKind.BOOLEAN:
        if type(value) is not bool:
            raise ResidualIntelligenceError(f"{name} must be boolean")
        return value
    if kind is PayloadFieldKind.INTEGER:
        return bounded_int(
            value,
            name,
            minimum=contract.minimum_integer,
            maximum=contract.maximum_integer,
        )
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ResidualIntelligenceError(f"{name} must be an array")
    if len(value) > contract.maximum_items:
        raise ResidualIntelligenceError(f"{name} exceeds its item bound")
    if not value and not contract.allow_empty:
        raise ResidualIntelligenceError(f"{name} must not be empty")
    if kind in {PayloadFieldKind.TOKEN_LIST, PayloadFieldKind.RELATIVE_PATH_LIST}:
        items = [
            _bounded_payload_token(
                item,
                name=f"{name}[{index}]",
                maximum_bytes=contract.maximum_text_bytes,
            )
            for index, item in enumerate(value)
        ]
        if kind is PayloadFieldKind.RELATIVE_PATH_LIST:
            for item in items:
                path = PurePosixPath(item)
                if path.is_absolute() or item in {".", ".."} or ".." in path.parts:
                    raise ResidualIntelligenceError(
                        f"{name} must contain safe repository-relative paths"
                    )
        if contract.unique_items and len(items) != len(set(items)):
            raise ResidualIntelligenceError(f"{name} contains duplicate items")
        return items
    if kind is PayloadFieldKind.INTEGER_LIST:
        items = [
            bounded_int(
                item,
                f"{name}[{index}]",
                minimum=contract.minimum_integer,
                maximum=contract.maximum_integer,
            )
            for index, item in enumerate(value)
        ]
        if contract.unique_items and len(items) != len(set(items)):
            raise ResidualIntelligenceError(f"{name} contains duplicate items")
        return items
    raise ResidualIntelligenceError(f"{name} has an unsupported field kind")


def _validate_payload(grammar: ExpertGrammar, payload: Any, *, abstaining: bool) -> dict[str, Any]:
    result = bounded_json_mapping(payload, "structured_payload")
    unknown = sorted(set(result) - set(grammar.payload_fields))
    if unknown:
        raise ResidualIntelligenceError(
            "structured payload contains arbitrary fields: " + ", ".join(unknown)
        )
    if abstaining:
        if result:
            raise ResidualIntelligenceError("abstention structured_payload must be empty")
        return {}
    required = set(grammar.required_payload_fields)
    missing = sorted(required - set(result))
    if missing:
        raise ResidualIntelligenceError(
            "structured payload is missing required fields: " + ", ".join(missing)
        )
    normalized = {
        key: _validate_field_value(
            value,
            name=f"structured_payload.{key}",
            contract=grammar.field_contracts[key],
        )
        for key, value in result.items()
    }
    for left, right in grammar.aligned_list_fields:
        if (left in normalized) != (right in normalized):
            raise ResidualIntelligenceError(
                f"aligned payload fields {left} and {right} must appear together"
            )
        if left in normalized and len(normalized[left]) != len(normalized[right]):
            raise ResidualIntelligenceError(
                f"aligned payload fields {left} and {right} differ in length"
            )
    for name in grammar.descending_integer_list_fields:
        values = normalized.get(name)
        if values is not None and any(
            values[index] < values[index + 1] for index in range(len(values) - 1)
        ):
            raise ResidualIntelligenceError(f"payload ranking field {name} is not descending")
    return normalized


def decode_structured_output(raw: str | bytes, grammar: ExpertGrammar) -> StructuredDecodeResult:
    """Strict parse; every failure is ``invalid_output`` with no prose recovery."""

    if not isinstance(grammar, ExpertGrammar):
        raise ResidualIntelligenceError("grammar must be ExpertGrammar")
    try:
        if isinstance(raw, bytes):
            encoded = bytes(raw)
            text = encoded.decode("utf-8")
        elif isinstance(raw, str):
            text = raw
            encoded = raw.encode("utf-8")
        else:
            raise ResidualIntelligenceError("decoder input must be UTF-8 text or bytes")
        if len(encoded) > grammar.maximum_output_bytes:
            raise ResidualIntelligenceError("structured output exceeds grammar byte bound")
        value = _closed_json_loads(text)
        outer_fields = {
            "output_class",
            "structured_payload",
            "confidence_or_score",
            "calibration_group",
            "abstained",
            "reason_codes",
            "evidence_references",
            "candidate_only",
        }
        unknown = sorted(set(value) - outer_fields)
        missing = sorted(outer_fields - set(value))
        if unknown:
            raise ResidualIntelligenceError(
                "structured output contains arbitrary fields: " + ", ".join(unknown)
            )
        if missing:
            raise ResidualIntelligenceError(
                "structured output is missing fields: " + ", ".join(missing)
            )
        output_class = value.get("output_class")
        if not isinstance(output_class, str):
            raise ResidualIntelligenceError("output_class must be a string")
        if output_class not in grammar.output_classes:
            raise ResidualIntelligenceError("output_class is outside grammar")
        payload = _validate_payload(
            grammar,
            value.get("structured_payload"),
            abstaining=output_class == grammar.abstention_output_class,
        )
        abstained = value.get("abstained")
        if abstained is True and output_class != grammar.abstention_output_class:
            raise ResidualIntelligenceError("abstention must use the grammar abstention class")
        if abstained is False and output_class == grammar.abstention_output_class:
            raise ResidualIntelligenceError("abstention class requires abstained=true")
        calibration_group = value.get("calibration_group")
        if not isinstance(calibration_group, str):
            raise ResidualIntelligenceError("calibration_group must be a string")
        output = ResidualTaskOutput(
            output_class=output_class,
            structured_payload=payload,
            confidence_or_score=value.get("confidence_or_score"),
            calibration_group=calibration_group,
            abstained=abstained,
            reason_codes=value.get("reason_codes"),
            evidence_references=value.get("evidence_references"),
            candidate_only=value.get("candidate_only"),
        )
        return StructuredDecodeResult(
            status=DecodeStatus.VALID,
            grammar_id=grammar.grammar_id,
            output=output,
            reason_codes=(),
        )
    except (ResidualIntelligenceError, TypeError, ValueError):
        return StructuredDecodeResult(
            status=DecodeStatus.INVALID_OUTPUT,
            grammar_id=grammar.grammar_id,
            output=None,
            reason_codes=("invalid_output",),
        )


def _grammar(
    family: ResidualTaskFamily,
    output_class: str,
    contracts: Mapping[str, PayloadFieldContract],
    *,
    required: tuple[str, ...] = (),
    aligned: tuple[tuple[str, str], ...] = (),
    descending: tuple[str, ...] = (),
    maximum_output_bytes: int = 4096,
) -> ExpertGrammar:
    return ExpertGrammar(
        task_family=family,
        output_classes=("ABSTAIN",) if output_class == "ABSTAIN" else (output_class, "ABSTAIN"),
        payload_fields=tuple(contracts),
        required_payload_fields=required,
        field_contracts=contracts,
        aligned_list_fields=aligned,
        descending_integer_list_fields=descending,
        maximum_output_bytes=maximum_output_bytes,
    )


def _token(
    *allowed_values: str, maximum_text_bytes: int = MAX_PAYLOAD_TOKEN_BYTES
) -> PayloadFieldContract:
    return PayloadFieldContract(
        kind=PayloadFieldKind.TOKEN,
        maximum_text_bytes=maximum_text_bytes,
        allowed_values=tuple(allowed_values),
    )


def _boolean() -> PayloadFieldContract:
    return PayloadFieldContract(kind=PayloadFieldKind.BOOLEAN)


def _integer(*, minimum: int, maximum: int) -> PayloadFieldContract:
    return PayloadFieldContract(
        kind=PayloadFieldKind.INTEGER,
        minimum_integer=minimum,
        maximum_integer=maximum,
    )


def _tokens(*, allow_empty: bool = True, maximum_items: int = 64) -> PayloadFieldContract:
    return PayloadFieldContract(
        kind=PayloadFieldKind.TOKEN_LIST,
        maximum_items=maximum_items,
        allow_empty=allow_empty,
    )


def _scores(*, maximum_items: int = 64) -> PayloadFieldContract:
    return PayloadFieldContract(
        kind=PayloadFieldKind.INTEGER_LIST,
        maximum_items=maximum_items,
        minimum_integer=0,
        maximum_integer=MAX_SCORE_PPM,
        allow_empty=False,
        unique_items=False,
    )


def _paths(*, maximum_items: int = 64) -> PayloadFieldContract:
    return PayloadFieldContract(
        kind=PayloadFieldKind.RELATIVE_PATH_LIST,
        maximum_text_bytes=MAX_PATCH_PATH_BYTES,
        maximum_items=maximum_items,
        allow_empty=False,
    )


def _classification_contracts(*, labels: tuple[str, ...] = ()) -> dict[str, PayloadFieldContract]:
    return {
        "label": _token(*labels),
        "reference_ids": _tokens(),
    }


def _ranking_grammar(family: ResidualTaskFamily, output_class: str) -> ExpertGrammar:
    return _grammar(
        family,
        output_class,
        {
            "ranked_reference_ids": _tokens(allow_empty=False),
            "scores_ppm": _scores(),
        },
        required=("ranked_reference_ids", "scores_ppm"),
        aligned=(("ranked_reference_ids", "scores_ppm"),),
        descending=("scores_ppm",),
    )


DEFAULT_GRAMMARS: Final[Mapping[ResidualTaskFamily, ExpertGrammar]] = {
    ResidualTaskFamily.TASK_CLASSIFICATION: _grammar(
        ResidualTaskFamily.TASK_CLASSIFICATION,
        "TASK_CLASSIFICATION",
        _classification_contracts(),
        required=("label",),
    ),
    ResidualTaskFamily.RISK_CLASSIFICATION: _grammar(
        ResidualTaskFamily.RISK_CLASSIFICATION,
        "RISK_CLASSIFICATION",
        _classification_contracts(labels=("R0", "R1", "R2", "R3", "R4", "R5")),
        required=("label",),
    ),
    ResidualTaskFamily.EFFECT_CLASSIFICATION: _grammar(
        ResidualTaskFamily.EFFECT_CLASSIFICATION,
        "EFFECT_CLASSIFICATION",
        {
            "effect_classes": _tokens(allow_empty=False),
            "reference_ids": _tokens(),
        },
        required=("effect_classes",),
    ),
    ResidualTaskFamily.AUTHORITY_REQUIREMENT_CLASSIFICATION: _grammar(
        ResidualTaskFamily.AUTHORITY_REQUIREMENT_CLASSIFICATION,
        "AUTHORITY_REQUIREMENT_CLASSIFICATION",
        _classification_contracts(),
        required=("label",),
    ),
    ResidualTaskFamily.CONTEXT_SUFFICIENCY: _grammar(
        ResidualTaskFamily.CONTEXT_SUFFICIENCY,
        "CONTEXT_SUFFICIENCY",
        {
            "sufficient": _boolean(),
            "missing_reference_ids": _tokens(),
            "reason_code": _token(),
        },
        required=("sufficient",),
    ),
    ResidualTaskFamily.EVIDENCE_RANKING: _ranking_grammar(
        ResidualTaskFamily.EVIDENCE_RANKING,
        "EVIDENCE_RANKING",
    ),
    ResidualTaskFamily.PROCEDURE_MATCHING: _grammar(
        ResidualTaskFamily.PROCEDURE_MATCHING,
        "PROCEDURE_MATCHING",
        {
            "procedure_id": _token(),
            "precondition_reference_ids": _tokens(),
            "match_class": _token(),
        },
        required=("procedure_id", "match_class"),
    ),
    ResidualTaskFamily.PLAN_BRANCH_RANKING: _ranking_grammar(
        ResidualTaskFamily.PLAN_BRANCH_RANKING,
        "PLAN_BRANCH_RANKING",
    ),
    ResidualTaskFamily.TEST_SELECTION: _grammar(
        ResidualTaskFamily.TEST_SELECTION,
        "TEST_SELECTION",
        {
            "test_ids": _tokens(allow_empty=False),
            "coverage_reference_ids": _tokens(),
        },
        required=("test_ids",),
    ),
    ResidualTaskFamily.PROOF_SELECTION: _grammar(
        ResidualTaskFamily.PROOF_SELECTION,
        "PROOF_SELECTION",
        {
            "proof_ids": _tokens(allow_empty=False),
            "obligation_reference_ids": _tokens(),
        },
        required=("proof_ids",),
    ),
    ResidualTaskFamily.FAILURE_ATTRIBUTION: _grammar(
        ResidualTaskFamily.FAILURE_ATTRIBUTION,
        "FAILURE_ATTRIBUTION",
        {
            "failure_class": _token(),
            "recommended_action": _token(),
            "reference_ids": _tokens(),
        },
        required=("failure_class", "recommended_action"),
    ),
    ResidualTaskFamily.RETRY_OR_ESCALATE: _grammar(
        ResidualTaskFamily.RETRY_OR_ESCALATE,
        "RETRY_OR_ESCALATE",
        {
            "decision": _token("retry", "escalate", "stop"),
            "reason_code": _token(),
            "reference_ids": _tokens(),
        },
        required=("decision",),
    ),
    ResidualTaskFamily.CACHE_REUSE_CLASSIFICATION: _grammar(
        ResidualTaskFamily.CACHE_REUSE_CLASSIFICATION,
        "CACHE_REUSE_CLASSIFICATION",
        {
            "reuse": _boolean(),
            "dependency_reference_ids": _tokens(),
            "reason_code": _token(),
        },
        required=("reuse",),
    ),
    ResidualTaskFamily.MERGE_CONFLICT_CLASSIFICATION: _grammar(
        ResidualTaskFamily.MERGE_CONFLICT_CLASSIFICATION,
        "MERGE_CONFLICT_CLASSIFICATION",
        {
            "conflict_class": _token(),
            "symbol_ids": _tokens(),
            "reference_ids": _tokens(),
        },
        required=("conflict_class",),
    ),
    ResidualTaskFamily.PATCH_TEMPLATE_SELECTION: _grammar(
        ResidualTaskFamily.PATCH_TEMPLATE_SELECTION,
        "PATCH_TEMPLATE_SELECTION",
        {
            "template_id": _token(),
            "symbol_ids": _tokens(),
            "reference_ids": _tokens(),
        },
        required=("template_id",),
    ),
    ResidualTaskFamily.PROCEDURE_HOLE_FILLING: _grammar(
        ResidualTaskFamily.PROCEDURE_HOLE_FILLING,
        "PROCEDURE_HOLE_RESOLUTION",
        {
            "hole_id": _token(),
            "operator_id": _token(),
            "argument_reference_ids": _tokens(),
            "precondition_reference_ids": _tokens(),
        },
        required=("hole_id", "operator_id"),
        maximum_output_bytes=8192,
    ),
    ResidualTaskFamily.PATCH_SKETCH_GENERATION: _grammar(
        ResidualTaskFamily.PATCH_SKETCH_GENERATION,
        "PATCH_SKETCH",
        {
            "files": _paths(),
            "symbol_ids": _tokens(allow_empty=False),
            "operations": _tokens(allow_empty=False),
            "maximum_changed_lines": _integer(
                minimum=1,
                maximum=MAX_PATCH_CHANGED_LINES,
            ),
            "validation_ids": _tokens(),
        },
        required=("files", "symbol_ids", "operations", "maximum_changed_lines"),
        maximum_output_bytes=16_384,
    ),
    ResidualTaskFamily.LEMMA_SUGGESTION: _grammar(
        ResidualTaskFamily.LEMMA_SUGGESTION,
        "LEMMA_SUGGESTION",
        {
            "lemma_ids": _tokens(allow_empty=False),
            "premise_ids": _tokens(),
            "obligation_id": _token(),
        },
        required=("lemma_ids", "obligation_id"),
    ),
    ResidualTaskFamily.TACTIC_SUGGESTION: _grammar(
        ResidualTaskFamily.TACTIC_SUGGESTION,
        "TACTIC_SUGGESTION",
        {
            "tactic_ids": _tokens(allow_empty=False),
            "premise_ids": _tokens(),
            "obligation_id": _token(),
        },
        required=("tactic_ids", "obligation_id"),
    ),
    ResidualTaskFamily.COUNTEREXAMPLE_EXPLANATION: _grammar(
        ResidualTaskFamily.COUNTEREXAMPLE_EXPLANATION,
        "COUNTEREXAMPLE_EXPLANATION",
        {
            "failure_class": _token(),
            "violated_invariant_ids": _tokens(),
            "counterexample_reference_ids": _tokens(allow_empty=False),
        },
        required=("failure_class", "counterexample_reference_ids"),
    ),
    ResidualTaskFamily.GOAL_REFINEMENT_CANDIDATE: _grammar(
        ResidualTaskFamily.GOAL_REFINEMENT_CANDIDATE,
        "GOAL_REFINEMENT_CANDIDATE",
        {
            "parent_goal_id": _token(),
            "candidate_goal_kinds": _tokens(allow_empty=False),
            "acceptance_reference_ids": _tokens(),
        },
        required=("parent_goal_id", "candidate_goal_kinds"),
    ),
    ResidualTaskFamily.DOCUMENTATION_CLAIM_CLASSIFICATION: _grammar(
        ResidualTaskFamily.DOCUMENTATION_CLAIM_CLASSIFICATION,
        "DOCUMENTATION_CLAIM_CLASSIFICATION",
        {
            "claim_class": _token(),
            "evidence_reference_ids": _tokens(),
            "rewrite_required": _boolean(),
        },
        required=("claim_class",),
    ),
    ResidualTaskFamily.HUMAN_ESCALATION_CLASSIFICATION: _grammar(
        ResidualTaskFamily.HUMAN_ESCALATION_CLASSIFICATION,
        "HUMAN_ESCALATION_CLASSIFICATION",
        {
            "escalate": _boolean(),
            "reason_code": _token(),
            "evidence_reference_ids": _tokens(),
        },
        required=("escalate", "reason_code"),
    ),
    ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING: _grammar(
        ResidualTaskFamily.NOVEL_UNBOUNDED_REASONING,
        "ABSTAIN",
        {"reason_code": _token()},
        required=("reason_code",),
        maximum_output_bytes=1024,
    ),
}


def grammar_for(task_family: ResidualTaskFamily | str) -> ExpertGrammar:
    return DEFAULT_GRAMMARS[ResidualTaskFamily(task_family)]


__all__ = (
    "DEFAULT_GRAMMARS",
    "DecodeStatus",
    "EXPERT_GRAMMAR_SCHEMA",
    "ExpertGrammar",
    "MAX_PATCH_CHANGED_LINES",
    "MAX_PAYLOAD_TOKEN_BYTES",
    "MAX_SCORE_PPM",
    "MAX_STRUCTURED_OUTPUT_BYTES",
    "PAYLOAD_FIELD_CONTRACT_SCHEMA",
    "PayloadFieldContract",
    "PayloadFieldKind",
    "STRUCTURED_DECODE_RESULT_SCHEMA",
    "StructuredDecodeResult",
    "decode_structured_output",
    "grammar_for",
)
