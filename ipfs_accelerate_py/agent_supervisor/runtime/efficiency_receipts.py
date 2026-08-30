"""Closed ASEH task-efficiency receipt and paired-benchmark manifest contracts.

These schemas extend canonical telemetry with identity, model, compute, cost,
work, quality, evidence-state, and terminal fields. Missing values remain
``unavailable`` and are never numeric zero. Construction admits a closed
document; it does not attest measurements or task completion.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final, Iterable, TypeVar

from ..proof.formal_verification_contracts import (
    CanonicalContract,
    ContractValidationError,
    canonical_json_bytes,
    content_identity,
)


CONTRACT_VERSION: Final[int] = 1
TASK_EFFICIENCY_RECEIPT_SCHEMA: Final[str] = "aseh/task-efficiency-receipt@1"
PAIRED_BENCHMARK_MANIFEST_SCHEMA: Final[str] = "aseh/paired-benchmark-manifest@1"
PROGRAM_ID: Final[str] = "agent-supervisor-efficiency-and-state-hardening-v1"
GOAL_ID: Final[str] = "ASEH-G020"

SCHEMA_DIRECTORY: Final[Path] = Path(__file__).resolve().parent / "schemas"
TASK_EFFICIENCY_RECEIPT_SCHEMA_PATH: Final[Path] = (
    SCHEMA_DIRECTORY / "task_efficiency_receipt.schema.json"
)
PAIRED_BENCHMARK_MANIFEST_SCHEMA_PATH: Final[Path] = (
    SCHEMA_DIRECTORY / "paired_benchmark_manifest.schema.json"
)

MAX_INTEGER: Final[int] = 10**18
MAX_TEXT_BYTES: Final[int] = 256
MAX_ID_BYTES: Final[int] = 96
MAX_REQUEST_IDS: Final[int] = 32
MAX_INTERFACE_SCHEMAS: Final[int] = 64
MAX_FIELD_PATHS: Final[int] = 256
MAX_SERIALIZED_BYTES: Final[int] = 262_144

UNIT_TOKENS: Final[str] = "tokens"
UNIT_COUNT: Final[str] = "count"
UNIT_BYTES: Final[str] = "bytes"
UNIT_SECONDS_MILLIONTHS: Final[str] = "seconds_millionths"
UNIT_MICROUSD: Final[str] = "microusd"
UNIT_RATIO_MILLIONTHS: Final[str] = "ratio_millionths"
UNIT_COMPUTE_UNITS: Final[str] = "compute_units"
UNITS: Final[tuple[str, ...]] = (
    UNIT_TOKENS,
    UNIT_COUNT,
    UNIT_BYTES,
    UNIT_SECONDS_MILLIONTHS,
    UNIT_MICROUSD,
    UNIT_RATIO_MILLIONTHS,
    UNIT_COMPUTE_UNITS,
)

MODEL_CLASSES: Final[tuple[str, ...]] = (
    "deterministic",
    "local_small_model",
    "local_medium_model",
    "remote_standard_model",
    "remote_frontier_model",
    "human",
)
TRUTH_STATES: Final[tuple[str, ...]] = (
    "measured",
    "estimated",
    "unavailable",
    "attempted",
    "observed",
    "verified",
    "simulated",
)
DISTINGUISHED_TRUTH_STATES: Final[tuple[str, ...]] = (
    "measured",
    "estimated",
    "unavailable",
    "attempted",
    "observed",
    "verified",
)
UNAVAILABLE_REASONS: Final[tuple[str, ...]] = (
    "sensor-absent",
    "permission-denied",
    "hardware-absent",
    "provider-omitted",
    "collection-failed",
    "not-reported",
    "not-applicable",
    "not-sealed",
    "deadline-elapsed",
    "identity-missing",
)
HISTORICAL_REQUIRED_OUTCOMES: Final[tuple[str, ...]] = (
    "successful",
    "failed",
    "retried",
    "rescued",
    "conflicted",
    "human_escalated",
)
PAIRED_STATISTICS: Final[tuple[str, ...]] = (
    "median_difference",
    "mean_difference",
    "per_task_ratios",
    "bootstrap_confidence_intervals",
    "distribution_by_task_class",
    "outlier_analysis",
    "quality_adjusted_cost",
    "accepted_patch_rate",
    "time_to_terminal_outcome",
)
DIRECT_ARM_CONSTRAINTS: Final[tuple[str, ...]] = (
    "same_safe_isolation_and_acceptance_tests",
    "no_candidate_context_pack_optimization",
    "no_candidate_proof_or_test_reuse_optimization",
    "same_available_model_and_provider_set",
    "no_unsafe_direct_mutation",
)
SEALED_CURRENT_ARM_CONSTRAINTS: Final[tuple[str, ...]] = (
    "exact_pre_campaign_policy",
    "exact_pre_campaign_routing_behavior",
)
CANDIDATE_ARM_CONSTRAINTS: Final[tuple[str, ...]] = (
    "campaign_changes_only",
    "shadow_before_mutation",
)
EQUAL_CONTROL_IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "repository_revision",
    "objective",
    "task_inputs",
    "acceptance_tests",
    "available_providers_and_models",
    "price_accounting",
    "resource_limits",
    "human_intervention_policy",
)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,95}$")
_TEXT_SAFE_RE = re.compile(r"^[^\x00]{1,256}$")
_SENSOR_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_CID_RE = re.compile(r"^b[a-z2-7]{20,}$")
_GIT_RE = re.compile(r"^[0-9a-f]{40}$")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_INTERFACE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-@]{0,255}$")
_FIELD_PATH_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")
_SECRET_MARKERS = (
    "access_token",
    "api_key",
    "authorization",
    "password",
    "private_key",
    "secret",
    "session_token",
)
_TContract = TypeVar("_TContract", bound=CanonicalContract)


class EfficiencyReceiptError(ValueError):
    """Closed efficiency receipt or paired-manifest contract was rejected."""


class EvidenceState(str, Enum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"
    ATTEMPTED = "attempted"
    OBSERVED = "observed"
    VERIFIED = "verified"
    SIMULATED = "simulated"


class ArmId(str, Enum):
    DIRECT = "direct_minimal_orchestration_baseline"
    SEALED_CURRENT = "sealed_current_supervisor_baseline"
    CANDIDATE = "candidate_optimized_supervisor"


class FinalTaskOutcome(str, Enum):
    SUCCESSFUL = "successful"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CONFLICTED = "conflicted"
    QUARANTINED = "quarantined"
    COMPENSATED = "compensated"
    UNAVAILABLE = "unavailable"


class ValidationResult(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    NOT_REQUIRED = "not_required"
    UNAVAILABLE = "unavailable"


class PatchDisposition(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"
    REVERTED = "reverted"
    UNAVAILABLE = "unavailable"


class PopulationKind(str, Enum):
    HERMETIC = "hermetic"
    HISTORICAL_REPLAY = "historical_replay"
    LIVE = "live"
    UNAVAILABLE = "unavailable"


_QUANTITATIVE_FIELDS: dict[EvidenceState, tuple[str, ...]] = {
    EvidenceState.MEASURED: ("status", "value", "unit", "sensor_id"),
    EvidenceState.ESTIMATED: (
        "status",
        "value",
        "unit",
        "estimate_method",
        "price_snapshot_identity",
    ),
    EvidenceState.UNAVAILABLE: ("status", "reason_code", "sensor_id"),
    EvidenceState.ATTEMPTED: ("status", "operation", "attempt_id"),
    EvidenceState.OBSERVED: ("status", "value", "unit", "sensor_id"),
    EvidenceState.VERIFIED: (
        "status",
        "value",
        "unit",
        "verifier_identity",
        "verifier_receipt_cid",
    ),
    EvidenceState.SIMULATED: ("status", "value", "unit", "fixture_identity"),
}
_QUANTITATIVE_REQUIRED: dict[EvidenceState, tuple[str, ...]] = {
    EvidenceState.MEASURED: ("status", "value", "unit", "sensor_id"),
    EvidenceState.ESTIMATED: ("status", "value", "unit", "estimate_method"),
    EvidenceState.UNAVAILABLE: ("status", "reason_code", "sensor_id"),
    EvidenceState.ATTEMPTED: ("status", "operation", "attempt_id"),
    EvidenceState.OBSERVED: ("status", "value", "unit", "sensor_id"),
    EvidenceState.VERIFIED: (
        "status",
        "value",
        "unit",
        "verifier_identity",
        "verifier_receipt_cid",
    ),
    EvidenceState.SIMULATED: ("status", "value", "unit", "fixture_identity"),
}
_IDENTITY_FIELDS: dict[EvidenceState, tuple[str, ...]] = {
    EvidenceState.OBSERVED: ("status", "identity", "sensor_id"),
    EvidenceState.VERIFIED: (
        "status",
        "identity",
        "verifier_identity",
        "verifier_receipt_cid",
    ),
    EvidenceState.UNAVAILABLE: ("status", "reason_code", "sensor_id"),
    EvidenceState.ATTEMPTED: ("status", "operation", "attempt_id"),
}
_REQUEST_FIELDS: dict[EvidenceState, tuple[str, ...]] = {
    EvidenceState.OBSERVED: ("status", "request_ids", "sensor_id"),
    EvidenceState.VERIFIED: (
        "status",
        "request_ids",
        "verifier_identity",
        "verifier_receipt_cid",
    ),
    EvidenceState.UNAVAILABLE: ("status", "reason_code", "sensor_id"),
    EvidenceState.ATTEMPTED: ("status", "operation", "attempt_id"),
}


def sensor_id_for(*parts: str) -> str:
    body = json.dumps(list(parts), separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return f"sensor:sha256:{digest}"


def fixture_cid(label: str) -> str:
    return content_identity({"aseh": "efficiency-receipts", "label": label})


def _enum(value: Any, enum_type: type[Enum], name: str) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise EfficiencyReceiptError(f"{name} must be one of: {allowed}") from exc


def _integer(value: Any, name: str, *, maximum: int = MAX_INTEGER, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EfficiencyReceiptError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise EfficiencyReceiptError(f"{name} must be between {minimum} and {maximum}")
    return value


def _text(value: Any, name: str, *, maximum: int = MAX_TEXT_BYTES, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str):
        raise EfficiencyReceiptError(f"{name} must be text")
    if "\x00" in value:
        raise EfficiencyReceiptError(f"{name} must not contain NUL bytes")
    encoded = value.encode("utf-8")
    if not encoded:
        raise EfficiencyReceiptError(f"{name} must not be empty")
    if len(encoded) > maximum:
        raise EfficiencyReceiptError(f"{name} exceeds the {maximum}-byte bound")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise EfficiencyReceiptError(f"{name} is not a closed {name} value")
    return value


def _bounded_id(value: Any, name: str) -> str:
    return _text(value, name, maximum=MAX_ID_BYTES, pattern=_ID_RE)


def _bounded_text(value: Any, name: str) -> str:
    return _text(value, name, maximum=MAX_TEXT_BYTES, pattern=_TEXT_SAFE_RE)


def _sensor(value: Any, name: str = "sensor_id") -> str:
    return _text(value, name, maximum=MAX_TEXT_BYTES, pattern=_SENSOR_RE)


def _cid(value: Any, name: str) -> str:
    return _text(value, name, maximum=128, pattern=_CID_RE)


def _git_id(value: Any, name: str) -> str:
    return _text(value, name, maximum=40, pattern=_GIT_RE)


def _unit(value: Any, name: str = "unit") -> str:
    result = _text(value, name, maximum=32)
    if result not in UNITS:
        raise EfficiencyReceiptError(f"{name} is not a supported unit")
    return result


def _reason(value: Any, name: str = "reason_code") -> str:
    result = _text(value, name, maximum=64)
    if result not in UNAVAILABLE_REASONS:
        raise EfficiencyReceiptError(f"{name} is not a supported unavailable reason")
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EfficiencyReceiptError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise EfficiencyReceiptError(f"{name} keys must be text")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: Iterable[str], name: str) -> None:
    unknown = set(payload) - set(allowed)
    if unknown:
        raise EfficiencyReceiptError(
            f"{name} contains unknown fields: {sorted(unknown)}"
        )


def _require(payload: Mapping[str, Any], required: Iterable[str], name: str) -> None:
    missing = [key for key in required if key not in payload]
    if missing:
        raise EfficiencyReceiptError(f"{name} missing required fields: {missing}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in pairs:
        if key in result:
            raise EfficiencyReceiptError("canonical payload JSON contains duplicate object keys")
        result[key] = item
    return result


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "array"
    return type(value).__name__


def _resolve_schema(node: Any, document: Mapping[str, Any]) -> Any:
    if not isinstance(node, Mapping) or "$ref" not in node:
        return node
    ref = node["$ref"]
    if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
        raise EfficiencyReceiptError("unsupported schema $ref")
    name = ref[len("#/$defs/") :]
    defs = document.get("$defs")
    if "/" in name or not name or not isinstance(defs, Mapping) or name not in defs:
        raise EfficiencyReceiptError(f"unresolved schema $ref: {ref}")
    return defs[name]


def _assert_closed_schema_node(document: Mapping[str, Any], node: Any, path: str) -> None:
    if isinstance(node, list):
        for index, item in enumerate(node):
            _assert_closed_schema_node(document, item, f"{path}[{index}]")
        return
    if not isinstance(node, Mapping):
        return
    if "$ref" in node:
        return
    if node.get("type") == "object" or "properties" in node or "required" in node:
        if node.get("additionalProperties") is not False:
            raise EfficiencyReceiptError(f"{path} must be a closed object schema")
    for index, option in enumerate(node.get("oneOf") or ()):
        _assert_closed_schema_node(document, option, f"{path}.oneOf[{index}]")
    properties = node.get("properties")
    if isinstance(properties, Mapping):
        for name, child in properties.items():
            _assert_closed_schema_node(document, child, f"{path}.properties.{name}")
    defs = node.get("$defs")
    if isinstance(defs, Mapping):
        for name, child in defs.items():
            _assert_closed_schema_node(document, child, f"{path}.$defs.{name}")
    items = node.get("items")
    if isinstance(items, Mapping):
        _assert_closed_schema_node(document, items, f"{path}.items")
    prefix = node.get("prefixItems")
    if isinstance(prefix, list):
        for index, child in enumerate(prefix):
            _assert_closed_schema_node(
                document, child, f"{path}.prefixItems[{index}]"
            )


def _check_type(instance: Any, expected: Any, path: str) -> None:
    types = (expected,) if isinstance(expected, str) else tuple(expected)
    actual = _json_type(instance)
    if actual == "integer" and "number" in types:
        return
    if actual not in types:
        raise EfficiencyReceiptError(f"{path} must be {' or '.join(types)}")


def validate_closed_schema(
    instance: Any,
    schema: Any,
    *,
    document: Mapping[str, Any] | None = None,
    path: str = "$",
) -> None:
    """Validate ``instance`` against a closed JSON Schema subset."""

    document = schema if document is None and isinstance(schema, Mapping) else document
    if document is None:
        raise EfficiencyReceiptError("schema document is required")
    node = _resolve_schema(schema, document)
    if node is True:
        return
    if node is False:
        raise EfficiencyReceiptError(f"{path} is not permitted")
    if not isinstance(node, Mapping):
        raise EfficiencyReceiptError(f"{path} schema must be an object")
    if "oneOf" in node:
        matches = 0
        for option in node["oneOf"]:
            try:
                validate_closed_schema(
                    instance, option, document=document, path=path
                )
            except EfficiencyReceiptError:
                continue
            matches += 1
        if matches != 1:
            raise EfficiencyReceiptError(
                f"{path} does not match exactly one closed alternative"
            )
        return
    if "type" in node:
        _check_type(instance, node["type"], path)
    if "const" in node and instance != node["const"]:
        raise EfficiencyReceiptError(f"{path} must equal the closed const")
    if "enum" in node and instance not in node["enum"]:
        raise EfficiencyReceiptError(f"{path} is not a supported enum value")
    if isinstance(instance, str):
        if "minLength" in node and len(instance) < int(node["minLength"]):
            raise EfficiencyReceiptError(f"{path} is shorter than minLength")
        if "maxLength" in node and len(instance) > int(node["maxLength"]):
            raise EfficiencyReceiptError(f"{path} exceeds maxLength")
        pattern = node.get("pattern")
        if isinstance(pattern, str) and re.fullmatch(pattern, instance) is None:
            raise EfficiencyReceiptError(f"{path} does not match the closed pattern")
    if isinstance(instance, int) and not isinstance(instance, bool):
        if "minimum" in node and instance < int(node["minimum"]):
            raise EfficiencyReceiptError(f"{path} is below minimum")
        if "maximum" in node and instance > int(node["maximum"]):
            raise EfficiencyReceiptError(f"{path} exceeds maximum")
    if isinstance(instance, Mapping):
        properties = node.get("properties") or {}
        additional = node.get("additionalProperties", True)
        unknown = set(instance) - set(properties)
        if additional is False and unknown:
            raise EfficiencyReceiptError(
                f"{path} contains unknown fields: {sorted(unknown)}"
            )
        required = node.get("required") or []
        missing = [key for key in required if key not in instance]
        if missing:
            raise EfficiencyReceiptError(f"{path} missing required fields: {missing}")
        for key, child in properties.items():
            if key in instance:
                validate_closed_schema(
                    instance[key], child, document=document, path=f"{path}.{key}"
                )
        return
    if isinstance(instance, Sequence) and not isinstance(instance, (str, bytes, bytearray)):
        minimum = int(node.get("minItems", 0))
        maximum = node.get("maxItems")
        if len(instance) < minimum:
            raise EfficiencyReceiptError(f"{path} has fewer than minItems")
        if maximum is not None and len(instance) > int(maximum):
            raise EfficiencyReceiptError(f"{path} exceeds maxItems")
        prefix = node.get("prefixItems") or []
        items = node.get("items", True)
        for index, option in enumerate(prefix):
            if index < len(instance):
                validate_closed_schema(
                    instance[index],
                    option,
                    document=document,
                    path=f"{path}[{index}]",
                )
        rest_start = len(prefix)
        if items is False and len(instance) > rest_start:
            raise EfficiencyReceiptError(f"{path} contains extra items")
        if items is not False:
            for index in range(rest_start, len(instance)):
                validate_closed_schema(
                    instance[index],
                    items,
                    document=document,
                    path=f"{path}[{index}]",
                )
        if node.get("uniqueItems"):
            encoded = [canonical_json_bytes(item) for item in instance]
            if len(set(encoded)) != len(encoded):
                raise EfficiencyReceiptError(f"{path} items must be unique")


@lru_cache(maxsize=4)
def load_schema_document(path: str) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EfficiencyReceiptError("schema document is malformed JSON") from exc
    if not isinstance(document, Mapping):
        raise EfficiencyReceiptError("schema document must be an object")
    payload = dict(document)
    _assert_closed_schema_node(payload, payload, "$")
    return payload


def load_task_efficiency_receipt_schema() -> dict[str, Any]:
    return load_schema_document(str(TASK_EFFICIENCY_RECEIPT_SCHEMA_PATH))


def load_paired_benchmark_manifest_schema() -> dict[str, Any]:
    return load_schema_document(str(PAIRED_BENCHMARK_MANIFEST_SCHEMA_PATH))


def _empty_evidence_state() -> dict[str, list[str]]:
    return {state: [] for state in TRUTH_STATES}


def _record_evidence(
    index: dict[str, list[str]], path: str, status: EvidenceState
) -> None:
    if _FIELD_PATH_RE.fullmatch(path) is None:
        raise EfficiencyReceiptError(f"{path} is not a closed evidence field path")
    index[status.value].append(path)


def _freeze_evidence_state(index: Mapping[str, list[str]]) -> dict[str, list[str]]:
    payload: dict[str, list[str]] = {}
    seen: set[str] = set()
    for state in TRUTH_STATES:
        paths = tuple(sorted(index.get(state, ())))
        if len(set(paths)) != len(paths):
            raise EfficiencyReceiptError("evidence_state paths must be unique")
        overlap = seen.intersection(paths)
        if overlap:
            raise EfficiencyReceiptError(
                f"evidence_state conflates truth states for {sorted(overlap)}"
            )
        seen.update(paths)
        payload[state] = list(paths)
    return payload


def _claim_content_id(payload: Mapping[str, Any], actual: str) -> None:
    claimed = payload.get("content_id")
    if claimed not in (None, "") and claimed != actual:
        raise EfficiencyReceiptError("content identity does not match the canonical payload")


def _from_canonical_bytes(cls: type[_TContract], payload: bytes) -> _TContract:
    if not isinstance(payload, (bytes, bytearray)):
        raise EfficiencyReceiptError("canonical payload must be bytes")
    raw = bytes(payload)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EfficiencyReceiptError("canonical payload must be UTF-8") from exc
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as exc:
        raise EfficiencyReceiptError("canonical payload is malformed JSON") from exc
    if not isinstance(value, Mapping):
        raise EfficiencyReceiptError("canonical payload must be an object")
    try:
        canonical = canonical_json_bytes(value)
    except ContractValidationError as exc:
        raise EfficiencyReceiptError("payload is not canonical bytes") from exc
    if canonical != raw:
        raise EfficiencyReceiptError("payload is not canonical bytes")
    result = cls.from_dict(value)
    if result.canonical_bytes() != raw:
        raise EfficiencyReceiptError("payload does not round-trip canonical bytes")
    return result


def _expect_unit(evidence: "QuantitativeEvidence", unit: str, name: str) -> None:
    if evidence.status in {
        EvidenceState.UNAVAILABLE,
        EvidenceState.ATTEMPTED,
    }:
        return
    if evidence.unit != unit:
        raise EfficiencyReceiptError(f"{name} unit must be {unit}")


def _forbid_status(
    evidence: Any, name: str, forbidden: set[EvidenceState]
) -> None:
    if evidence.status in forbidden:
        raise EfficiencyReceiptError(
            f"{name} cannot use conflated truth state {evidence.status.value}"
        )


@dataclass(frozen=True)
class QuantitativeEvidence:
    """Integer metric tagged with exactly one evidence state."""

    status: EvidenceState
    value: int = 0
    unit: str = ""
    sensor_id: str = ""
    reason_code: str = ""
    estimate_method: str = ""
    price_snapshot_identity: str = ""
    operation: str = ""
    attempt_id: str = ""
    verifier_identity: str = ""
    verifier_receipt_cid: str = ""
    fixture_identity: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "status", _enum(self.status, EvidenceState, "status")
        )
        if self.status in {
            EvidenceState.MEASURED,
            EvidenceState.ESTIMATED,
            EvidenceState.OBSERVED,
            EvidenceState.VERIFIED,
            EvidenceState.SIMULATED,
        }:
            object.__setattr__(self, "value", _integer(self.value, "value"))
            object.__setattr__(self, "unit", _unit(self.unit))
        else:
            if isinstance(self.value, bool) or not isinstance(self.value, int):
                raise EfficiencyReceiptError("value must be an integer")
            if self.value != 0:
                raise EfficiencyReceiptError(
                    f"{self.status.value} evidence must not encode a numeric value"
                )
        if self.status is EvidenceState.MEASURED:
            object.__setattr__(self, "sensor_id", _sensor(self.sensor_id))
            if self.reason_code or self.estimate_method or self.verifier_identity:
                raise EfficiencyReceiptError("measured evidence has foreign fields")
        elif self.status is EvidenceState.ESTIMATED:
            object.__setattr__(
                self, "estimate_method", _bounded_id(self.estimate_method, "estimate_method")
            )
            if self.price_snapshot_identity:
                object.__setattr__(
                    self,
                    "price_snapshot_identity",
                    _bounded_text(self.price_snapshot_identity, "price_snapshot_identity"),
                )
            if self.sensor_id or self.reason_code or self.verifier_identity:
                raise EfficiencyReceiptError("estimated evidence has foreign fields")
        elif self.status is EvidenceState.UNAVAILABLE:
            object.__setattr__(self, "reason_code", _reason(self.reason_code))
            object.__setattr__(self, "sensor_id", _sensor(self.sensor_id))
            if self.unit or self.estimate_method or self.verifier_identity:
                raise EfficiencyReceiptError(
                    "unavailable evidence must not encode a numeric value or unit"
                )
        elif self.status is EvidenceState.ATTEMPTED:
            object.__setattr__(self, "operation", _bounded_id(self.operation, "operation"))
            object.__setattr__(self, "attempt_id", _bounded_text(self.attempt_id, "attempt_id"))
            if self.unit or self.sensor_id or self.verifier_identity:
                raise EfficiencyReceiptError("attempted evidence has foreign fields")
        elif self.status is EvidenceState.OBSERVED:
            object.__setattr__(self, "sensor_id", _sensor(self.sensor_id))
            if self.verifier_identity or self.reason_code or self.estimate_method:
                raise EfficiencyReceiptError("observed evidence has foreign fields")
        elif self.status is EvidenceState.VERIFIED:
            object.__setattr__(
                self,
                "verifier_identity",
                _bounded_id(self.verifier_identity, "verifier_identity"),
            )
            object.__setattr__(
                self,
                "verifier_receipt_cid",
                _cid(self.verifier_receipt_cid, "verifier_receipt_cid"),
            )
            if self.sensor_id or self.reason_code or self.estimate_method:
                raise EfficiencyReceiptError("verified evidence has foreign fields")
        else:
            object.__setattr__(
                self,
                "fixture_identity",
                _bounded_text(self.fixture_identity, "fixture_identity"),
            )
            if self.sensor_id or self.verifier_identity or self.reason_code:
                raise EfficiencyReceiptError("simulated evidence has foreign fields")

    @classmethod
    def measured(cls, value: int, *, unit: str, sensor_id: str) -> "QuantitativeEvidence":
        return cls(
            status=EvidenceState.MEASURED,
            value=value,
            unit=unit,
            sensor_id=sensor_id,
        )

    @classmethod
    def estimated(
        cls,
        value: int,
        *,
        unit: str,
        estimate_method: str,
        price_snapshot_identity: str = "",
    ) -> "QuantitativeEvidence":
        return cls(
            status=EvidenceState.ESTIMATED,
            value=value,
            unit=unit,
            estimate_method=estimate_method,
            price_snapshot_identity=price_snapshot_identity,
        )

    @classmethod
    def unavailable(cls, reason: str, *, sensor_id: str) -> "QuantitativeEvidence":
        return cls(
            status=EvidenceState.UNAVAILABLE,
            reason_code=reason,
            sensor_id=sensor_id,
        )

    @classmethod
    def attempted(cls, operation: str, *, attempt_id: str) -> "QuantitativeEvidence":
        return cls(
            status=EvidenceState.ATTEMPTED,
            operation=operation,
            attempt_id=attempt_id,
        )

    @classmethod
    def observed(cls, value: int, *, unit: str, sensor_id: str) -> "QuantitativeEvidence":
        return cls(
            status=EvidenceState.OBSERVED,
            value=value,
            unit=unit,
            sensor_id=sensor_id,
        )

    @classmethod
    def verified(
        cls,
        value: int,
        *,
        unit: str,
        verifier_identity: str,
        verifier_receipt_cid: str,
    ) -> "QuantitativeEvidence":
        return cls(
            status=EvidenceState.VERIFIED,
            value=value,
            unit=unit,
            verifier_identity=verifier_identity,
            verifier_receipt_cid=verifier_receipt_cid,
        )

    @classmethod
    def simulated(
        cls, value: int, *, unit: str, fixture_identity: str
    ) -> "QuantitativeEvidence":
        return cls(
            status=EvidenceState.SIMULATED,
            value=value,
            unit=unit,
            fixture_identity=fixture_identity,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": self.status.value}
        if self.status is EvidenceState.MEASURED:
            payload.update(value=self.value, unit=self.unit, sensor_id=self.sensor_id)
        elif self.status is EvidenceState.ESTIMATED:
            payload.update(
                value=self.value,
                unit=self.unit,
                estimate_method=self.estimate_method,
            )
            if self.price_snapshot_identity:
                payload["price_snapshot_identity"] = self.price_snapshot_identity
        elif self.status is EvidenceState.UNAVAILABLE:
            payload.update(reason_code=self.reason_code, sensor_id=self.sensor_id)
        elif self.status is EvidenceState.ATTEMPTED:
            payload.update(operation=self.operation, attempt_id=self.attempt_id)
        elif self.status is EvidenceState.OBSERVED:
            payload.update(value=self.value, unit=self.unit, sensor_id=self.sensor_id)
        elif self.status is EvidenceState.VERIFIED:
            payload.update(
                value=self.value,
                unit=self.unit,
                verifier_identity=self.verifier_identity,
                verifier_receipt_cid=self.verifier_receipt_cid,
            )
        else:
            payload.update(
                value=self.value,
                unit=self.unit,
                fixture_identity=self.fixture_identity,
            )
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QuantitativeEvidence":
        mapping = _mapping(payload, "quantitative evidence")
        status = _enum(mapping.get("status", ""), EvidenceState, "status")
        allowed = _QUANTITATIVE_FIELDS[status]
        _reject_unknown(mapping, allowed, "quantitative evidence")
        _require(mapping, _QUANTITATIVE_REQUIRED[status], "quantitative evidence")
        return cls(
            status=status,
            value=mapping.get("value", 0),
            unit=mapping.get("unit", ""),
            sensor_id=mapping.get("sensor_id", ""),
            reason_code=mapping.get("reason_code", ""),
            estimate_method=mapping.get("estimate_method", ""),
            price_snapshot_identity=mapping.get("price_snapshot_identity", ""),
            operation=mapping.get("operation", ""),
            attempt_id=mapping.get("attempt_id", ""),
            verifier_identity=mapping.get("verifier_identity", ""),
            verifier_receipt_cid=mapping.get("verifier_receipt_cid", ""),
            fixture_identity=mapping.get("fixture_identity", ""),
        )


@dataclass(frozen=True)
class IdentityEvidence:
    """Optional identity tagged with observed, verified, unavailable, or attempted."""

    status: EvidenceState
    identity: str = ""
    sensor_id: str = ""
    reason_code: str = ""
    operation: str = ""
    attempt_id: str = ""
    verifier_identity: str = ""
    verifier_receipt_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "status", _enum(self.status, EvidenceState, "status")
        )
        if self.status not in _IDENTITY_FIELDS:
            raise EfficiencyReceiptError(
                f"identity evidence cannot use truth state {self.status.value}"
            )
        if self.status in {EvidenceState.OBSERVED, EvidenceState.VERIFIED}:
            object.__setattr__(self, "identity", _bounded_text(self.identity, "identity"))
        elif self.identity:
            raise EfficiencyReceiptError(
                f"{self.status.value} identity evidence must not encode an identity"
            )
        if self.status is EvidenceState.OBSERVED:
            object.__setattr__(self, "sensor_id", _sensor(self.sensor_id))
        elif self.status is EvidenceState.VERIFIED:
            object.__setattr__(
                self,
                "verifier_identity",
                _bounded_id(self.verifier_identity, "verifier_identity"),
            )
            object.__setattr__(
                self,
                "verifier_receipt_cid",
                _cid(self.verifier_receipt_cid, "verifier_receipt_cid"),
            )
        elif self.status is EvidenceState.UNAVAILABLE:
            object.__setattr__(self, "reason_code", _reason(self.reason_code))
            object.__setattr__(self, "sensor_id", _sensor(self.sensor_id))
        else:
            object.__setattr__(self, "operation", _bounded_id(self.operation, "operation"))
            object.__setattr__(self, "attempt_id", _bounded_text(self.attempt_id, "attempt_id"))

    @classmethod
    def observed(cls, identity: str, *, sensor_id: str) -> "IdentityEvidence":
        return cls(status=EvidenceState.OBSERVED, identity=identity, sensor_id=sensor_id)

    @classmethod
    def verified(
        cls,
        identity: str,
        *,
        verifier_identity: str,
        verifier_receipt_cid: str,
    ) -> "IdentityEvidence":
        return cls(
            status=EvidenceState.VERIFIED,
            identity=identity,
            verifier_identity=verifier_identity,
            verifier_receipt_cid=verifier_receipt_cid,
        )

    @classmethod
    def unavailable(cls, reason: str, *, sensor_id: str) -> "IdentityEvidence":
        return cls(
            status=EvidenceState.UNAVAILABLE,
            reason_code=reason,
            sensor_id=sensor_id,
        )

    @classmethod
    def attempted(cls, operation: str, *, attempt_id: str) -> "IdentityEvidence":
        return cls(
            status=EvidenceState.ATTEMPTED,
            operation=operation,
            attempt_id=attempt_id,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": self.status.value}
        if self.status is EvidenceState.OBSERVED:
            payload.update(identity=self.identity, sensor_id=self.sensor_id)
        elif self.status is EvidenceState.VERIFIED:
            payload.update(
                identity=self.identity,
                verifier_identity=self.verifier_identity,
                verifier_receipt_cid=self.verifier_receipt_cid,
            )
        elif self.status is EvidenceState.UNAVAILABLE:
            payload.update(reason_code=self.reason_code, sensor_id=self.sensor_id)
        else:
            payload.update(operation=self.operation, attempt_id=self.attempt_id)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "IdentityEvidence":
        mapping = _mapping(payload, "identity evidence")
        status = _enum(mapping.get("status", ""), EvidenceState, "status")
        if status not in _IDENTITY_FIELDS:
            raise EfficiencyReceiptError(
                f"identity evidence cannot use truth state {status.value}"
            )
        allowed = _IDENTITY_FIELDS[status]
        _reject_unknown(mapping, allowed, "identity evidence")
        _require(mapping, allowed, "identity evidence")
        return cls(
            status=status,
            identity=mapping.get("identity", ""),
            sensor_id=mapping.get("sensor_id", ""),
            reason_code=mapping.get("reason_code", ""),
            operation=mapping.get("operation", ""),
            attempt_id=mapping.get("attempt_id", ""),
            verifier_identity=mapping.get("verifier_identity", ""),
            verifier_receipt_cid=mapping.get("verifier_receipt_cid", ""),
        )


def _request_ids(values: Any) -> tuple[str, ...]:
    if values is None:
        source: Sequence[Any] = ()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        source = values
    else:
        raise EfficiencyReceiptError("request_ids must be a sequence")
    if len(source) > MAX_REQUEST_IDS:
        raise EfficiencyReceiptError("request_ids exceeds the 32-item bound")
    result: list[str] = []
    for index, item in enumerate(source):
        text = _text(item, f"request_ids[{index}]", maximum=128, pattern=_REQUEST_ID_RE)
        lowered = text.lower()
        if any(marker in lowered for marker in _SECRET_MARKERS):
            raise EfficiencyReceiptError("request_ids must not contain credentials")
        if text not in result:
            result.append(text)
    return tuple(result)


@dataclass(frozen=True)
class RequestIdEvidence:
    """Safe provider request identifiers, or an explicit unavailable/attempted state."""

    status: EvidenceState
    request_ids: tuple[str, ...] = ()
    sensor_id: str = ""
    reason_code: str = ""
    operation: str = ""
    attempt_id: str = ""
    verifier_identity: str = ""
    verifier_receipt_cid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "status", _enum(self.status, EvidenceState, "status")
        )
        if self.status not in _REQUEST_FIELDS:
            raise EfficiencyReceiptError(
                f"request-id evidence cannot use truth state {self.status.value}"
            )
        object.__setattr__(self, "request_ids", _request_ids(self.request_ids))
        if self.status in {EvidenceState.UNAVAILABLE, EvidenceState.ATTEMPTED} and self.request_ids:
            raise EfficiencyReceiptError(
                f"{self.status.value} request-id evidence must not encode identifiers"
            )
        if self.status is EvidenceState.OBSERVED:
            object.__setattr__(self, "sensor_id", _sensor(self.sensor_id))
        elif self.status is EvidenceState.VERIFIED:
            object.__setattr__(
                self,
                "verifier_identity",
                _bounded_id(self.verifier_identity, "verifier_identity"),
            )
            object.__setattr__(
                self,
                "verifier_receipt_cid",
                _cid(self.verifier_receipt_cid, "verifier_receipt_cid"),
            )
        elif self.status is EvidenceState.UNAVAILABLE:
            object.__setattr__(self, "reason_code", _reason(self.reason_code))
            object.__setattr__(self, "sensor_id", _sensor(self.sensor_id))
        else:
            object.__setattr__(self, "operation", _bounded_id(self.operation, "operation"))
            object.__setattr__(self, "attempt_id", _bounded_text(self.attempt_id, "attempt_id"))

    @classmethod
    def observed(cls, request_ids: Sequence[str], *, sensor_id: str) -> "RequestIdEvidence":
        return cls(
            status=EvidenceState.OBSERVED,
            request_ids=tuple(request_ids),
            sensor_id=sensor_id,
        )

    @classmethod
    def unavailable(cls, reason: str, *, sensor_id: str) -> "RequestIdEvidence":
        return cls(
            status=EvidenceState.UNAVAILABLE,
            reason_code=reason,
            sensor_id=sensor_id,
        )

    @classmethod
    def attempted(cls, operation: str, *, attempt_id: str) -> "RequestIdEvidence":
        return cls(
            status=EvidenceState.ATTEMPTED,
            operation=operation,
            attempt_id=attempt_id,
        )

    @classmethod
    def verified(
        cls,
        request_ids: Sequence[str],
        *,
        verifier_identity: str,
        verifier_receipt_cid: str,
    ) -> "RequestIdEvidence":
        return cls(
            status=EvidenceState.VERIFIED,
            request_ids=tuple(request_ids),
            verifier_identity=verifier_identity,
            verifier_receipt_cid=verifier_receipt_cid,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": self.status.value}
        if self.status is EvidenceState.OBSERVED:
            payload.update(request_ids=list(self.request_ids), sensor_id=self.sensor_id)
        elif self.status is EvidenceState.VERIFIED:
            payload.update(
                request_ids=list(self.request_ids),
                verifier_identity=self.verifier_identity,
                verifier_receipt_cid=self.verifier_receipt_cid,
            )
        elif self.status is EvidenceState.UNAVAILABLE:
            payload.update(reason_code=self.reason_code, sensor_id=self.sensor_id)
        else:
            payload.update(operation=self.operation, attempt_id=self.attempt_id)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RequestIdEvidence":
        mapping = _mapping(payload, "request-id evidence")
        status = _enum(mapping.get("status", ""), EvidenceState, "status")
        if status not in _REQUEST_FIELDS:
            raise EfficiencyReceiptError(
                f"request-id evidence cannot use truth state {status.value}"
            )
        allowed = _REQUEST_FIELDS[status]
        _reject_unknown(mapping, allowed, "request-id evidence")
        _require(mapping, allowed, "request-id evidence")
        return cls(
            status=status,
            request_ids=tuple(mapping.get("request_ids") or ()),
            sensor_id=mapping.get("sensor_id", ""),
            reason_code=mapping.get("reason_code", ""),
            operation=mapping.get("operation", ""),
            attempt_id=mapping.get("attempt_id", ""),
            verifier_identity=mapping.get("verifier_identity", ""),
            verifier_receipt_cid=mapping.get("verifier_receipt_cid", ""),
        )


def unavailable_quantitative(path: str, *, reason: str = "not-reported") -> QuantitativeEvidence:
    return QuantitativeEvidence.unavailable(reason, sensor_id=sensor_id_for(path, reason))


def unavailable_identity(path: str, *, reason: str = "identity-missing") -> IdentityEvidence:
    return IdentityEvidence.unavailable(reason, sensor_id=sensor_id_for(path, reason))


def unavailable_request_ids(path: str, *, reason: str = "provider-omitted") -> RequestIdEvidence:
    return RequestIdEvidence.unavailable(reason, sensor_id=sensor_id_for(path, reason))


def _interface_identities(values: Any) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise EfficiencyReceiptError("interface_schema_identities must be a sequence")
    if not values:
        raise EfficiencyReceiptError("interface_schema_identities must not be empty")
    if len(values) > MAX_INTERFACE_SCHEMAS:
        raise EfficiencyReceiptError("interface_schema_identities exceeds the 64-item bound")
    result: list[str] = []
    for index, item in enumerate(values):
        text = _text(
            item,
            f"interface_schema_identities[{index}]",
            maximum=MAX_TEXT_BYTES,
            pattern=_INTERFACE_RE,
        )
        if text not in result:
            result.append(text)
    return tuple(result)


def _coerce_quantitative(value: Any, name: str) -> QuantitativeEvidence:
    if isinstance(value, QuantitativeEvidence):
        return value
    if isinstance(value, Mapping):
        return QuantitativeEvidence.from_dict(value)
    raise EfficiencyReceiptError(f"{name} must be quantitative evidence")


def _coerce_identity(value: Any, name: str) -> IdentityEvidence:
    if isinstance(value, IdentityEvidence):
        return value
    if isinstance(value, Mapping):
        return IdentityEvidence.from_dict(value)
    raise EfficiencyReceiptError(f"{name} must be identity evidence")


def _coerce_request_ids(value: Any, name: str) -> RequestIdEvidence:
    if isinstance(value, RequestIdEvidence):
        return value
    if isinstance(value, Mapping):
        return RequestIdEvidence.from_dict(value)
    raise EfficiencyReceiptError(f"{name} must be request-id evidence")


@dataclass(frozen=True)
class ReceiptIdentity:
    task_id: str
    task_cid: str
    objective_id: str
    objective_revision: str
    repository_commit: str
    repository_tree: str
    policy_identity: str
    context_pack_cid: IdentityEvidence
    interface_schema_identities: tuple[str, ...]
    provider_id: str
    model_id: str
    model_revision: IdentityEvidence

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _bounded_id(self.task_id, "task_id"))
        object.__setattr__(self, "task_cid", _cid(self.task_cid, "task_cid"))
        object.__setattr__(self, "objective_id", _bounded_id(self.objective_id, "objective_id"))
        object.__setattr__(
            self,
            "objective_revision",
            _bounded_text(self.objective_revision, "objective_revision"),
        )
        object.__setattr__(
            self, "repository_commit", _git_id(self.repository_commit, "repository_commit")
        )
        object.__setattr__(
            self, "repository_tree", _git_id(self.repository_tree, "repository_tree")
        )
        object.__setattr__(
            self, "policy_identity", _bounded_text(self.policy_identity, "policy_identity")
        )
        object.__setattr__(
            self,
            "context_pack_cid",
            _coerce_identity(self.context_pack_cid, "context_pack_cid"),
        )
        object.__setattr__(
            self,
            "interface_schema_identities",
            _interface_identities(self.interface_schema_identities),
        )
        object.__setattr__(self, "provider_id", _bounded_id(self.provider_id, "provider_id"))
        object.__setattr__(self, "model_id", _bounded_id(self.model_id, "model_id"))
        object.__setattr__(
            self,
            "model_revision",
            _coerce_identity(self.model_revision, "model_revision"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_cid": self.task_cid,
            "objective_id": self.objective_id,
            "objective_revision": self.objective_revision,
            "repository_commit": self.repository_commit,
            "repository_tree": self.repository_tree,
            "policy_identity": self.policy_identity,
            "context_pack_cid": self.context_pack_cid.to_dict(),
            "interface_schema_identities": list(self.interface_schema_identities),
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "model_revision": self.model_revision.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReceiptIdentity":
        mapping = _mapping(payload, "identity")
        _reject_unknown(
            mapping,
            {
                "task_id",
                "task_cid",
                "objective_id",
                "objective_revision",
                "repository_commit",
                "repository_tree",
                "policy_identity",
                "context_pack_cid",
                "interface_schema_identities",
                "provider_id",
                "model_id",
                "model_revision",
            },
            "identity",
        )
        _require(
            mapping,
            {
                "task_id",
                "task_cid",
                "objective_id",
                "objective_revision",
                "repository_commit",
                "repository_tree",
                "policy_identity",
                "context_pack_cid",
                "interface_schema_identities",
                "provider_id",
                "model_id",
                "model_revision",
            },
            "identity",
        )
        return cls(
            task_id=mapping["task_id"],
            task_cid=mapping["task_cid"],
            objective_id=mapping["objective_id"],
            objective_revision=mapping["objective_revision"],
            repository_commit=mapping["repository_commit"],
            repository_tree=mapping["repository_tree"],
            policy_identity=mapping["policy_identity"],
            context_pack_cid=mapping["context_pack_cid"],
            interface_schema_identities=tuple(mapping["interface_schema_identities"]),
            provider_id=mapping["provider_id"],
            model_id=mapping["model_id"],
            model_revision=mapping["model_revision"],
        )

    def collect_evidence(self, index: dict[str, list[str]]) -> None:
        _record_evidence(index, "identity.context_pack_cid", self.context_pack_cid.status)
        _record_evidence(index, "identity.model_revision", self.model_revision.status)


@dataclass(frozen=True)
class ModelUse:
    input_tokens: QuantitativeEvidence
    output_tokens: QuantitativeEvidence
    cached_input_tokens: QuantitativeEvidence
    reasoning_tokens: QuantitativeEvidence
    number_of_calls: QuantitativeEvidence
    calls_by_model_class: Mapping[str, QuantitativeEvidence]
    safe_provider_request_ids: RequestIdEvidence
    provider_reported_usage: QuantitativeEvidence

    def __post_init__(self) -> None:
        token_fields = (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
        )
        for name in token_fields:
            evidence = _coerce_quantitative(getattr(self, name), name)
            _expect_unit(evidence, UNIT_TOKENS, name)
            object.__setattr__(self, name, evidence)
        calls = _coerce_quantitative(self.number_of_calls, "number_of_calls")
        _expect_unit(calls, UNIT_COUNT, "number_of_calls")
        object.__setattr__(self, "number_of_calls", calls)
        raw_classes = self.calls_by_model_class
        if not isinstance(raw_classes, Mapping):
            raise EfficiencyReceiptError("calls_by_model_class must be an object")
        _reject_unknown(raw_classes, MODEL_CLASSES, "calls_by_model_class")
        _require(raw_classes, MODEL_CLASSES, "calls_by_model_class")
        classes: dict[str, QuantitativeEvidence] = {}
        for name in MODEL_CLASSES:
            evidence = _coerce_quantitative(raw_classes[name], f"calls_by_model_class.{name}")
            _expect_unit(evidence, UNIT_COUNT, f"calls_by_model_class.{name}")
            classes[name] = evidence
        object.__setattr__(self, "calls_by_model_class", MappingProxyType(classes))
        object.__setattr__(
            self,
            "safe_provider_request_ids",
            _coerce_request_ids(self.safe_provider_request_ids, "safe_provider_request_ids"),
        )
        usage = _coerce_quantitative(self.provider_reported_usage, "provider_reported_usage")
        _expect_unit(usage, UNIT_COUNT, "provider_reported_usage")
        object.__setattr__(self, "provider_reported_usage", usage)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens.to_dict(),
            "output_tokens": self.output_tokens.to_dict(),
            "cached_input_tokens": self.cached_input_tokens.to_dict(),
            "reasoning_tokens": self.reasoning_tokens.to_dict(),
            "number_of_calls": self.number_of_calls.to_dict(),
            "calls_by_model_class": {
                name: self.calls_by_model_class[name].to_dict() for name in MODEL_CLASSES
            },
            "safe_provider_request_ids": self.safe_provider_request_ids.to_dict(),
            "provider_reported_usage": self.provider_reported_usage.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelUse":
        mapping = _mapping(payload, "model")
        required = (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
            "number_of_calls",
            "calls_by_model_class",
            "safe_provider_request_ids",
            "provider_reported_usage",
        )
        _reject_unknown(mapping, required, "model")
        _require(mapping, required, "model")
        return cls(**{name: mapping[name] for name in required})

    def collect_evidence(self, index: dict[str, list[str]]) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
            "number_of_calls",
            "provider_reported_usage",
        ):
            _record_evidence(index, f"model.{name}", getattr(self, name).status)
        for name in MODEL_CLASSES:
            _record_evidence(
                index,
                f"model.calls_by_model_class.{name}",
                self.calls_by_model_class[name].status,
            )
        _record_evidence(
            index,
            "model.safe_provider_request_ids",
            self.safe_provider_request_ids.status,
        )


_COMPUTE_UNITS = {
    "cpu_seconds": UNIT_SECONDS_MILLIONTHS,
    "gpu_seconds": UNIT_SECONDS_MILLIONTHS,
    "peak_memory": UNIT_BYTES,
    "wall_clock_duration": UNIT_SECONDS_MILLIONTHS,
    "test_execution_time": UNIT_SECONDS_MILLIONTHS,
    "prover_execution_time": UNIT_SECONDS_MILLIONTHS,
    "static_analysis_time": UNIT_SECONDS_MILLIONTHS,
    "indexing_retrieval_time": UNIT_SECONDS_MILLIONTHS,
    "bytes_read": UNIT_BYTES,
    "bytes_written": UNIT_BYTES,
    "process_count": UNIT_COUNT,
    "concurrency": UNIT_COUNT,
}


@dataclass(frozen=True)
class ComputeUse:
    cpu_seconds: QuantitativeEvidence
    gpu_seconds: QuantitativeEvidence
    peak_memory: QuantitativeEvidence
    wall_clock_duration: QuantitativeEvidence
    test_execution_time: QuantitativeEvidence
    prover_execution_time: QuantitativeEvidence
    static_analysis_time: QuantitativeEvidence
    indexing_retrieval_time: QuantitativeEvidence
    bytes_read: QuantitativeEvidence
    bytes_written: QuantitativeEvidence
    process_count: QuantitativeEvidence
    concurrency: QuantitativeEvidence

    def __post_init__(self) -> None:
        for name, unit in _COMPUTE_UNITS.items():
            evidence = _coerce_quantitative(getattr(self, name), name)
            _expect_unit(evidence, unit, name)
            object.__setattr__(self, name, evidence)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name).to_dict() for name in _COMPUTE_UNITS}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ComputeUse":
        mapping = _mapping(payload, "compute")
        _reject_unknown(mapping, _COMPUTE_UNITS, "compute")
        _require(mapping, _COMPUTE_UNITS, "compute")
        return cls(**{name: mapping[name] for name in _COMPUTE_UNITS})

    def collect_evidence(self, index: dict[str, list[str]]) -> None:
        for name in _COMPUTE_UNITS:
            _record_evidence(index, f"compute.{name}", getattr(self, name).status)


_COST_UNITS = {
    "provider_reported_charge": UNIT_MICROUSD,
    "token_based_estimated_charge": UNIT_MICROUSD,
    "local_compute_units": UNIT_COMPUTE_UNITS,
    "audit_and_verification_overhead": UNIT_MICROUSD,
    "cost_per_accepted_patch": UNIT_MICROUSD,
    "cost_per_completed_task": UNIT_MICROUSD,
}


@dataclass(frozen=True)
class CostUse:
    provider_reported_charge: QuantitativeEvidence
    token_based_estimated_charge: QuantitativeEvidence
    price_snapshot_identity: IdentityEvidence
    price_snapshot_timestamp: IdentityEvidence
    local_compute_units: QuantitativeEvidence
    audit_and_verification_overhead: QuantitativeEvidence
    cost_per_accepted_patch: QuantitativeEvidence
    cost_per_completed_task: QuantitativeEvidence

    def __post_init__(self) -> None:
        for name, unit in _COST_UNITS.items():
            evidence = _coerce_quantitative(getattr(self, name), name)
            _expect_unit(evidence, unit, name)
            object.__setattr__(self, name, evidence)
        object.__setattr__(
            self,
            "price_snapshot_identity",
            _coerce_identity(self.price_snapshot_identity, "price_snapshot_identity"),
        )
        object.__setattr__(
            self,
            "price_snapshot_timestamp",
            _coerce_identity(self.price_snapshot_timestamp, "price_snapshot_timestamp"),
        )
        _forbid_status(
            self.provider_reported_charge,
            "provider_reported_charge",
            {EvidenceState.ESTIMATED},
        )
        _forbid_status(
            self.token_based_estimated_charge,
            "token_based_estimated_charge",
            {
                EvidenceState.MEASURED,
                EvidenceState.OBSERVED,
                EvidenceState.VERIFIED,
            },
        )
        if self.token_based_estimated_charge.status is EvidenceState.ESTIMATED:
            if not self.token_based_estimated_charge.price_snapshot_identity:
                raise EfficiencyReceiptError(
                    "token_based_estimated_charge requires price_snapshot_identity"
                )
            if self.price_snapshot_identity.status is EvidenceState.UNAVAILABLE:
                raise EfficiencyReceiptError(
                    "estimated charge requires an identified price snapshot"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_reported_charge": self.provider_reported_charge.to_dict(),
            "token_based_estimated_charge": self.token_based_estimated_charge.to_dict(),
            "price_snapshot_identity": self.price_snapshot_identity.to_dict(),
            "price_snapshot_timestamp": self.price_snapshot_timestamp.to_dict(),
            "local_compute_units": self.local_compute_units.to_dict(),
            "audit_and_verification_overhead": self.audit_and_verification_overhead.to_dict(),
            "cost_per_accepted_patch": self.cost_per_accepted_patch.to_dict(),
            "cost_per_completed_task": self.cost_per_completed_task.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CostUse":
        mapping = _mapping(payload, "cost")
        required = (
            "provider_reported_charge",
            "token_based_estimated_charge",
            "price_snapshot_identity",
            "price_snapshot_timestamp",
            "local_compute_units",
            "audit_and_verification_overhead",
            "cost_per_accepted_patch",
            "cost_per_completed_task",
        )
        _reject_unknown(mapping, required, "cost")
        _require(mapping, required, "cost")
        return cls(**{name: mapping[name] for name in required})

    def collect_evidence(self, index: dict[str, list[str]]) -> None:
        for name in _COST_UNITS:
            _record_evidence(index, f"cost.{name}", getattr(self, name).status)
        _record_evidence(
            index, "cost.price_snapshot_identity", self.price_snapshot_identity.status
        )
        _record_evidence(
            index, "cost.price_snapshot_timestamp", self.price_snapshot_timestamp.status
        )


_WORK_FIELDS = (
    "tests_selected",
    "tests_executed",
    "full_suite_tests",
    "type_static_schema_checks",
    "proof_obligations_selected",
    "proof_obligations_executed",
    "proof_receipts_reused",
    "retries",
    "rescue_attempts",
    "merge_conflicts",
    "manual_recovery",
    "human_interventions",
)


@dataclass(frozen=True)
class WorkUse:
    tests_selected: QuantitativeEvidence
    tests_executed: QuantitativeEvidence
    full_suite_tests: QuantitativeEvidence
    type_static_schema_checks: QuantitativeEvidence
    proof_obligations_selected: QuantitativeEvidence
    proof_obligations_executed: QuantitativeEvidence
    proof_receipts_reused: QuantitativeEvidence
    retries: QuantitativeEvidence
    rescue_attempts: QuantitativeEvidence
    merge_conflicts: QuantitativeEvidence
    manual_recovery: QuantitativeEvidence
    human_interventions: QuantitativeEvidence

    def __post_init__(self) -> None:
        for name in _WORK_FIELDS:
            evidence = _coerce_quantitative(getattr(self, name), name)
            _expect_unit(evidence, UNIT_COUNT, name)
            object.__setattr__(self, name, evidence)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name).to_dict() for name in _WORK_FIELDS}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WorkUse":
        mapping = _mapping(payload, "work")
        _reject_unknown(mapping, _WORK_FIELDS, "work")
        _require(mapping, _WORK_FIELDS, "work")
        return cls(**{name: mapping[name] for name in _WORK_FIELDS})

    def collect_evidence(self, index: dict[str, list[str]]) -> None:
        for name in _WORK_FIELDS:
            _record_evidence(index, f"work.{name}", getattr(self, name).status)


_QUALITY_UNITS = {
    "accepted_patch_quality": UNIT_RATIO_MILLIONTHS,
    "quality_adjusted_cost": UNIT_MICROUSD,
    "false_positives": UNIT_COUNT,
    "false_negatives": UNIT_COUNT,
    "selected_test_false_negatives": UNIT_COUNT,
    "controlled_critical_omissions": UNIT_COUNT,
    "escaped_critical_seeded_defects": UNIT_COUNT,
}


@dataclass(frozen=True)
class QualityUse:
    accepted_patch_quality: QuantitativeEvidence
    quality_adjusted_cost: QuantitativeEvidence
    false_positives: QuantitativeEvidence
    false_negatives: QuantitativeEvidence
    selected_test_false_negatives: QuantitativeEvidence
    controlled_critical_omissions: QuantitativeEvidence
    escaped_critical_seeded_defects: QuantitativeEvidence

    def __post_init__(self) -> None:
        for name, unit in _QUALITY_UNITS.items():
            evidence = _coerce_quantitative(getattr(self, name), name)
            _expect_unit(evidence, unit, name)
            object.__setattr__(self, name, evidence)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name).to_dict() for name in _QUALITY_UNITS}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QualityUse":
        mapping = _mapping(payload, "quality")
        _reject_unknown(mapping, _QUALITY_UNITS, "quality")
        _require(mapping, _QUALITY_UNITS, "quality")
        return cls(**{name: mapping[name] for name in _QUALITY_UNITS})

    def collect_evidence(self, index: dict[str, list[str]]) -> None:
        for name in _QUALITY_UNITS:
            _record_evidence(index, f"quality.{name}", getattr(self, name).status)


@dataclass(frozen=True)
class TerminalRecord:
    final_task_outcome: FinalTaskOutcome
    validation_result: ValidationResult
    patch_disposition: PatchDisposition
    population_kind: PopulationKind

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "final_task_outcome",
            _enum(self.final_task_outcome, FinalTaskOutcome, "final_task_outcome"),
        )
        object.__setattr__(
            self,
            "validation_result",
            _enum(self.validation_result, ValidationResult, "validation_result"),
        )
        object.__setattr__(
            self,
            "patch_disposition",
            _enum(self.patch_disposition, PatchDisposition, "patch_disposition"),
        )
        object.__setattr__(
            self,
            "population_kind",
            _enum(self.population_kind, PopulationKind, "population_kind"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_task_outcome": self.final_task_outcome.value,
            "validation_result": self.validation_result.value,
            "patch_disposition": self.patch_disposition.value,
            "population_kind": self.population_kind.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TerminalRecord":
        mapping = _mapping(payload, "terminal")
        required = (
            "final_task_outcome",
            "validation_result",
            "patch_disposition",
            "population_kind",
        )
        _reject_unknown(mapping, required, "terminal")
        _require(mapping, required, "terminal")
        return cls(**{name: mapping[name] for name in required})


def _authority_flags(payload: Mapping[str, Any], name: str) -> None:
    for field_name in ("authority", "attests_measurements", "attests_task_completion"):
        if field_name in payload and payload[field_name] is not False:
            raise EfficiencyReceiptError(
                f"{name} has schema authority only and cannot attest {field_name}"
            )
    version = payload.get("contract_version")
    if version is not None and version != CONTRACT_VERSION:
        raise EfficiencyReceiptError(f"unsupported {name} contract version")


class _ClosedContract(CanonicalContract):
    @property
    def schema_version(self) -> int:
        return CONTRACT_VERSION

    @classmethod
    def from_canonical_bytes(cls, payload: bytes) -> Any:
        return _from_canonical_bytes(cls, payload)


@dataclass(frozen=True, kw_only=True)
class TaskEfficiencyReceipt(_ClosedContract):
    """Closed task telemetry receipt. Admission is not measurement attestation."""

    SCHEMA: ClassVar[str] = TASK_EFFICIENCY_RECEIPT_SCHEMA

    # CanonicalContract.identity is a content-id property; field() keeps this required.
    identity: ReceiptIdentity = field()
    model: ModelUse
    compute: ComputeUse
    cost: CostUse
    work: WorkUse
    quality: QualityUse
    terminal: TerminalRecord

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ReceiptIdentity):
            object.__setattr__(self, "identity", ReceiptIdentity.from_dict(self.identity))
        if not isinstance(self.model, ModelUse):
            object.__setattr__(self, "model", ModelUse.from_dict(self.model))
        if not isinstance(self.compute, ComputeUse):
            object.__setattr__(self, "compute", ComputeUse.from_dict(self.compute))
        if not isinstance(self.cost, CostUse):
            object.__setattr__(self, "cost", CostUse.from_dict(self.cost))
        if not isinstance(self.work, WorkUse):
            object.__setattr__(self, "work", WorkUse.from_dict(self.work))
        if not isinstance(self.quality, QualityUse):
            object.__setattr__(self, "quality", QualityUse.from_dict(self.quality))
        if not isinstance(self.terminal, TerminalRecord):
            object.__setattr__(self, "terminal", TerminalRecord.from_dict(self.terminal))
        state = self.evidence_state()
        if (
            self.terminal.population_kind is PopulationKind.LIVE
            and state["simulated"]
        ):
            raise EfficiencyReceiptError("simulated fields cannot be admitted as live")
        encoded = self.canonical_bytes()
        if len(encoded) > MAX_SERIALIZED_BYTES:
            raise EfficiencyReceiptError("task efficiency receipt exceeds the serialized bound")
        validate_closed_schema(self.to_dict(), load_task_efficiency_receipt_schema())

    def evidence_state(self) -> dict[str, list[str]]:
        index = _empty_evidence_state()
        self.identity.collect_evidence(index)
        self.model.collect_evidence(index)
        self.compute.collect_evidence(index)
        self.cost.collect_evidence(index)
        self.work.collect_evidence(index)
        self.quality.collect_evidence(index)
        return _freeze_evidence_state(index)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "authority": False,
            "attests_measurements": False,
            "attests_task_completion": False,
            "identity": self.identity.to_dict(),
            "model": self.model.to_dict(),
            "compute": self.compute.to_dict(),
            "cost": self.cost.to_dict(),
            "work": self.work.to_dict(),
            "quality": self.quality.to_dict(),
            "terminal": self.terminal.to_dict(),
            "evidence_state": self.evidence_state(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskEfficiencyReceipt":
        mapping = _mapping(payload, "task efficiency receipt")
        if mapping.get("schema") not in (None, cls.SCHEMA):
            raise EfficiencyReceiptError("task efficiency receipt has foreign schema")
        _authority_flags(mapping, "task efficiency receipt")
        validate_closed_schema(mapping, load_task_efficiency_receipt_schema())
        allowed = {
            "schema",
            "contract_version",
            "authority",
            "attests_measurements",
            "attests_task_completion",
            "content_id",
            "identity",
            "model",
            "compute",
            "cost",
            "work",
            "quality",
            "terminal",
            "evidence_state",
        }
        _reject_unknown(mapping, allowed, "task efficiency receipt")
        _require(
            mapping,
            {"identity", "model", "compute", "cost", "work", "quality", "terminal"},
            "task efficiency receipt",
        )
        result = cls(
            identity=mapping["identity"],
            model=mapping["model"],
            compute=mapping["compute"],
            cost=mapping["cost"],
            work=mapping["work"],
            quality=mapping["quality"],
            terminal=mapping["terminal"],
        )
        claimed = mapping.get("evidence_state")
        if claimed is not None and claimed != result.evidence_state():
            raise EfficiencyReceiptError("evidence_state does not match field truth states")
        _claim_content_id(mapping, result.content_id)
        return result


@dataclass(frozen=True)
class PairedArm:
    arm_id: ArmId
    constraints: tuple[str, ...]
    shadow_before_mutation: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm_id", _enum(self.arm_id, ArmId, "arm_id"))
        if self.arm_id is ArmId.DIRECT:
            expected = DIRECT_ARM_CONSTRAINTS
        elif self.arm_id is ArmId.SEALED_CURRENT:
            expected = SEALED_CURRENT_ARM_CONSTRAINTS
        else:
            expected = CANDIDATE_ARM_CONSTRAINTS
        if tuple(self.constraints) != expected:
            raise EfficiencyReceiptError(f"{self.arm_id.value} constraints are not closed")
        object.__setattr__(self, "constraints", expected)
        if self.arm_id is ArmId.CANDIDATE:
            if self.shadow_before_mutation is not True:
                raise EfficiencyReceiptError("candidate arm must shadow before mutation")
        elif self.shadow_before_mutation is not None:
            raise EfficiencyReceiptError(f"{self.arm_id.value} cannot set shadow_before_mutation")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "arm_id": self.arm_id.value,
            "constraints": list(self.constraints),
        }
        if self.arm_id is ArmId.CANDIDATE:
            payload["shadow_before_mutation"] = True
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PairedArm":
        mapping = _mapping(payload, "arm")
        arm_id = _enum(mapping.get("arm_id", ""), ArmId, "arm_id")
        allowed = {"arm_id", "constraints"}
        if arm_id is ArmId.CANDIDATE:
            allowed = {"arm_id", "constraints", "shadow_before_mutation"}
        _reject_unknown(mapping, allowed, "arm")
        return cls(
            arm_id=arm_id,
            constraints=tuple(mapping.get("constraints") or ()),
            shadow_before_mutation=mapping.get("shadow_before_mutation"),
        )


def canonical_paired_arms() -> tuple[PairedArm, PairedArm, PairedArm]:
    return (
        PairedArm(ArmId.DIRECT, DIRECT_ARM_CONSTRAINTS),
        PairedArm(ArmId.SEALED_CURRENT, SEALED_CURRENT_ARM_CONSTRAINTS),
        PairedArm(ArmId.CANDIDATE, CANDIDATE_ARM_CONSTRAINTS, True),
    )


@dataclass(frozen=True)
class EqualControls:
    repository_revision: IdentityEvidence
    objective: IdentityEvidence
    task_inputs: IdentityEvidence
    acceptance_tests: IdentityEvidence
    available_providers_and_models: IdentityEvidence
    price_accounting: IdentityEvidence
    resource_limits: IdentityEvidence
    maximum_retries: QuantitativeEvidence
    human_intervention_policy: IdentityEvidence

    def __post_init__(self) -> None:
        for name in EQUAL_CONTROL_IDENTITY_FIELDS:
            object.__setattr__(self, name, _coerce_identity(getattr(self, name), name))
        retries = _coerce_quantitative(self.maximum_retries, "maximum_retries")
        _expect_unit(retries, UNIT_COUNT, "maximum_retries")
        object.__setattr__(self, "maximum_retries", retries)

    def to_dict(self) -> dict[str, Any]:
        payload = {name: getattr(self, name).to_dict() for name in EQUAL_CONTROL_IDENTITY_FIELDS}
        payload["maximum_retries"] = self.maximum_retries.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EqualControls":
        mapping = _mapping(payload, "equal_controls")
        required = EQUAL_CONTROL_IDENTITY_FIELDS + ("maximum_retries",)
        _reject_unknown(mapping, required, "equal_controls")
        _require(mapping, required, "equal_controls")
        return cls(**{name: mapping[name] for name in required})

    def collect_evidence(self, index: dict[str, list[str]]) -> None:
        for name in EQUAL_CONTROL_IDENTITY_FIELDS:
            _record_evidence(
                index, f"equal_controls.{name}", getattr(self, name).status
            )
        _record_evidence(
            index, "equal_controls.maximum_retries", self.maximum_retries.status
        )


@dataclass(frozen=True)
class Populations:
    hermetic_sealed_count: QuantitativeEvidence
    replay_sealed_count: QuantitativeEvidence
    live_enrolled_count: QuantitativeEvidence
    enrollment_deadline_maximum_days: int = 30

    def __post_init__(self) -> None:
        hermetic = _coerce_quantitative(self.hermetic_sealed_count, "hermetic sealed_count")
        replay = _coerce_quantitative(self.replay_sealed_count, "replay sealed_count")
        live = _coerce_quantitative(self.live_enrolled_count, "live enrolled_count")
        _expect_unit(hermetic, UNIT_COUNT, "hermetic sealed_count")
        _expect_unit(replay, UNIT_COUNT, "replay sealed_count")
        _expect_unit(live, UNIT_COUNT, "live enrolled_count")
        object.__setattr__(self, "hermetic_sealed_count", hermetic)
        object.__setattr__(self, "replay_sealed_count", replay)
        object.__setattr__(self, "live_enrolled_count", live)
        object.__setattr__(
            self,
            "enrollment_deadline_maximum_days",
            _integer(
                self.enrollment_deadline_maximum_days,
                "enrollment_deadline_maximum_days",
                minimum=1,
                maximum=30,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hermetic_development": {
                "minimum": 60,
                "sufficient_for_production_promotion": False,
                "sealed_count": self.hermetic_sealed_count.to_dict(),
            },
            "historical_exact_tree_replay": {
                "minimum": 20,
                "sealed_count": self.replay_sealed_count.to_dict(),
            },
            "new_live_shadow_canary": {
                "minimum": 10,
                "enrollment_deadline_maximum_days": self.enrollment_deadline_maximum_days,
                "enrolled_count": self.live_enrolled_count.to_dict(),
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Populations":
        mapping = _mapping(payload, "populations")
        _reject_unknown(
            mapping,
            {
                "hermetic_development",
                "historical_exact_tree_replay",
                "new_live_shadow_canary",
            },
            "populations",
        )
        _require(
            mapping,
            {
                "hermetic_development",
                "historical_exact_tree_replay",
                "new_live_shadow_canary",
            },
            "populations",
        )
        hermetic = _mapping(mapping["hermetic_development"], "hermetic_development")
        replay = _mapping(
            mapping["historical_exact_tree_replay"], "historical_exact_tree_replay"
        )
        live = _mapping(mapping["new_live_shadow_canary"], "new_live_shadow_canary")
        _reject_unknown(
            hermetic,
            {"minimum", "sufficient_for_production_promotion", "sealed_count"},
            "hermetic_development",
        )
        _reject_unknown(replay, {"minimum", "sealed_count"}, "historical_exact_tree_replay")
        _reject_unknown(
            live,
            {"minimum", "enrollment_deadline_maximum_days", "enrolled_count"},
            "new_live_shadow_canary",
        )
        if hermetic.get("minimum") != 60:
            raise EfficiencyReceiptError("hermetic_development.minimum must be 60")
        if hermetic.get("sufficient_for_production_promotion") is not False:
            raise EfficiencyReceiptError(
                "hermetic evidence is never sufficient for production promotion"
            )
        if replay.get("minimum") != 20:
            raise EfficiencyReceiptError("historical_exact_tree_replay.minimum must be 20")
        if live.get("minimum") != 10:
            raise EfficiencyReceiptError("new_live_shadow_canary.minimum must be 10")
        return cls(
            hermetic_sealed_count=hermetic["sealed_count"],
            replay_sealed_count=replay["sealed_count"],
            live_enrolled_count=live["enrolled_count"],
            enrollment_deadline_maximum_days=live.get(
                "enrollment_deadline_maximum_days", 30
            ),
        )

    def collect_evidence(self, index: dict[str, list[str]]) -> None:
        _record_evidence(
            index,
            "populations.hermetic_development.sealed_count",
            self.hermetic_sealed_count.status,
        )
        _record_evidence(
            index,
            "populations.historical_exact_tree_replay.sealed_count",
            self.replay_sealed_count.status,
        )
        _record_evidence(
            index,
            "populations.new_live_shadow_canary.enrolled_count",
            self.live_enrolled_count.status,
        )


@dataclass(frozen=True)
class PairedBenchmarkManifest(_ClosedContract):
    """Closed pairing contract. Admission is not measurement attestation."""

    SCHEMA: ClassVar[str] = PAIRED_BENCHMARK_MANIFEST_SCHEMA

    objective_id: str
    objective_revision: str
    repository_commit: str
    repository_tree: str
    policy_identity: str
    sealed_input_manifest_identity: IdentityEvidence
    price_snapshot_identity: IdentityEvidence
    environment_identity: IdentityEvidence
    provider_model_config_identity: IdentityEvidence
    equal_controls: EqualControls
    populations: Populations
    arms: tuple[PairedArm, ...] = canonical_paired_arms()

    def __post_init__(self) -> None:
        object.__setattr__(self, "objective_id", _bounded_id(self.objective_id, "objective_id"))
        object.__setattr__(
            self,
            "objective_revision",
            _bounded_text(self.objective_revision, "objective_revision"),
        )
        object.__setattr__(
            self, "repository_commit", _git_id(self.repository_commit, "repository_commit")
        )
        object.__setattr__(
            self, "repository_tree", _git_id(self.repository_tree, "repository_tree")
        )
        object.__setattr__(
            self, "policy_identity", _bounded_text(self.policy_identity, "policy_identity")
        )
        for name in (
            "sealed_input_manifest_identity",
            "price_snapshot_identity",
            "environment_identity",
            "provider_model_config_identity",
        ):
            object.__setattr__(self, name, _coerce_identity(getattr(self, name), name))
        if not isinstance(self.equal_controls, EqualControls):
            object.__setattr__(
                self, "equal_controls", EqualControls.from_dict(self.equal_controls)
            )
        if not isinstance(self.populations, Populations):
            object.__setattr__(self, "populations", Populations.from_dict(self.populations))
        arms = tuple(
            item if isinstance(item, PairedArm) else PairedArm.from_dict(item)
            for item in self.arms
        )
        expected = canonical_paired_arms()
        if tuple(item.arm_id for item in arms) != tuple(item.arm_id for item in expected):
            raise EfficiencyReceiptError("paired arms must be the closed three-arm set")
        object.__setattr__(self, "arms", expected)
        if self.populations.live_enrolled_count.status is EvidenceState.SIMULATED:
            raise EfficiencyReceiptError("simulated live enrollment cannot be admitted")
        encoded = self.canonical_bytes()
        if len(encoded) > MAX_SERIALIZED_BYTES:
            raise EfficiencyReceiptError("paired benchmark manifest exceeds the serialized bound")
        validate_closed_schema(self.to_dict(), load_paired_benchmark_manifest_schema())

    def evidence_state(self) -> dict[str, list[str]]:
        index = _empty_evidence_state()
        for name in (
            "sealed_input_manifest_identity",
            "price_snapshot_identity",
            "environment_identity",
            "provider_model_config_identity",
        ):
            _record_evidence(index, name, getattr(self, name).status)
        self.equal_controls.collect_evidence(index)
        self.populations.collect_evidence(index)
        return _freeze_evidence_state(index)

    def _payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "authority": False,
            "attests_measurements": False,
            "attests_task_completion": False,
            "program_id": PROGRAM_ID,
            "goal_id": GOAL_ID,
            "objective_id": self.objective_id,
            "objective_revision": self.objective_revision,
            "repository_commit": self.repository_commit,
            "repository_tree": self.repository_tree,
            "policy_identity": self.policy_identity,
            "sealed_input_manifest_identity": self.sealed_input_manifest_identity.to_dict(),
            "price_snapshot_identity": self.price_snapshot_identity.to_dict(),
            "environment_identity": self.environment_identity.to_dict(),
            "provider_model_config_identity": self.provider_model_config_identity.to_dict(),
            "arms": [item.to_dict() for item in self.arms],
            "equal_controls": self.equal_controls.to_dict(),
            "populations": self.populations.to_dict(),
            "historical_required_outcomes": list(HISTORICAL_REQUIRED_OUTCOMES),
            "statistics": list(PAIRED_STATISTICS),
            "promotion_without_paired_campaign": False,
            "pairing_rejects_unequal_controls": True,
            "evidence_state": self.evidence_state(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PairedBenchmarkManifest":
        mapping = _mapping(payload, "paired benchmark manifest")
        if mapping.get("schema") not in (None, cls.SCHEMA):
            raise EfficiencyReceiptError("paired benchmark manifest has foreign schema")
        _authority_flags(mapping, "paired benchmark manifest")
        validate_closed_schema(mapping, load_paired_benchmark_manifest_schema())
        allowed = {
            "schema",
            "contract_version",
            "authority",
            "attests_measurements",
            "attests_task_completion",
            "content_id",
            "program_id",
            "goal_id",
            "objective_id",
            "objective_revision",
            "repository_commit",
            "repository_tree",
            "policy_identity",
            "sealed_input_manifest_identity",
            "price_snapshot_identity",
            "environment_identity",
            "provider_model_config_identity",
            "arms",
            "equal_controls",
            "populations",
            "historical_required_outcomes",
            "statistics",
            "promotion_without_paired_campaign",
            "pairing_rejects_unequal_controls",
            "evidence_state",
        }
        _reject_unknown(mapping, allowed, "paired benchmark manifest")
        required = {
            "objective_id",
            "objective_revision",
            "repository_commit",
            "repository_tree",
            "policy_identity",
            "sealed_input_manifest_identity",
            "price_snapshot_identity",
            "environment_identity",
            "provider_model_config_identity",
            "equal_controls",
            "populations",
        }
        _require(mapping, required, "paired benchmark manifest")
        if mapping.get("program_id") not in (None, PROGRAM_ID):
            raise EfficiencyReceiptError("program_id is not the closed ASEH program")
        if mapping.get("goal_id") not in (None, GOAL_ID):
            raise EfficiencyReceiptError("goal_id is not ASEH-G020")
        outcomes = mapping.get("historical_required_outcomes")
        if outcomes is not None and tuple(outcomes) != HISTORICAL_REQUIRED_OUTCOMES:
            raise EfficiencyReceiptError("historical_required_outcomes are not closed")
        statistics = mapping.get("statistics")
        if statistics is not None and tuple(statistics) != PAIRED_STATISTICS:
            raise EfficiencyReceiptError("paired statistics are not closed")
        if mapping.get("promotion_without_paired_campaign") not in (None, False):
            raise EfficiencyReceiptError("promotion without a paired campaign is forbidden")
        if mapping.get("pairing_rejects_unequal_controls") not in (None, True):
            raise EfficiencyReceiptError("pairing must reject unequal controls")
        result = cls(
            objective_id=mapping["objective_id"],
            objective_revision=mapping["objective_revision"],
            repository_commit=mapping["repository_commit"],
            repository_tree=mapping["repository_tree"],
            policy_identity=mapping["policy_identity"],
            sealed_input_manifest_identity=mapping["sealed_input_manifest_identity"],
            price_snapshot_identity=mapping["price_snapshot_identity"],
            environment_identity=mapping["environment_identity"],
            provider_model_config_identity=mapping["provider_model_config_identity"],
            equal_controls=mapping["equal_controls"],
            populations=mapping["populations"],
            arms=tuple(mapping.get("arms") or canonical_paired_arms()),
        )
        claimed = mapping.get("evidence_state")
        if claimed is not None and claimed != result.evidence_state():
            raise EfficiencyReceiptError("evidence_state does not match field truth states")
        _claim_content_id(mapping, result.content_id)
        return result


def _unavailable_model() -> ModelUse:
    classes = {
        name: unavailable_quantitative(f"model.calls_by_model_class.{name}")
        for name in MODEL_CLASSES
    }
    return ModelUse(
        input_tokens=unavailable_quantitative("model.input_tokens"),
        output_tokens=unavailable_quantitative("model.output_tokens"),
        cached_input_tokens=unavailable_quantitative("model.cached_input_tokens"),
        reasoning_tokens=unavailable_quantitative("model.reasoning_tokens"),
        number_of_calls=unavailable_quantitative("model.number_of_calls"),
        calls_by_model_class=classes,
        safe_provider_request_ids=unavailable_request_ids("model.safe_provider_request_ids"),
        provider_reported_usage=unavailable_quantitative(
            "model.provider_reported_usage", reason="provider-omitted"
        ),
    )


def _unavailable_compute() -> ComputeUse:
    return ComputeUse(
        **{name: unavailable_quantitative(f"compute.{name}") for name in _COMPUTE_UNITS}
    )


def _unavailable_cost() -> CostUse:
    return CostUse(
        provider_reported_charge=unavailable_quantitative(
            "cost.provider_reported_charge", reason="provider-omitted"
        ),
        token_based_estimated_charge=unavailable_quantitative(
            "cost.token_based_estimated_charge", reason="not-reported"
        ),
        price_snapshot_identity=unavailable_identity("cost.price_snapshot_identity"),
        price_snapshot_timestamp=unavailable_identity("cost.price_snapshot_timestamp"),
        local_compute_units=unavailable_quantitative("cost.local_compute_units"),
        audit_and_verification_overhead=unavailable_quantitative(
            "cost.audit_and_verification_overhead"
        ),
        cost_per_accepted_patch=unavailable_quantitative("cost.cost_per_accepted_patch"),
        cost_per_completed_task=unavailable_quantitative("cost.cost_per_completed_task"),
    )


def _unavailable_work() -> WorkUse:
    return WorkUse(**{name: unavailable_quantitative(f"work.{name}") for name in _WORK_FIELDS})


def _unavailable_quality() -> QualityUse:
    return QualityUse(
        **{name: unavailable_quantitative(f"quality.{name}") for name in _QUALITY_UNITS}
    )


def unavailable_task_efficiency_receipt(
    *,
    identity: ReceiptIdentity,
    terminal: TerminalRecord | None = None,
    model: ModelUse | None = None,
    compute: ComputeUse | None = None,
    cost: CostUse | None = None,
    work: WorkUse | None = None,
    quality: QualityUse | None = None,
) -> TaskEfficiencyReceipt:
    """Build a closed receipt whose unspecified metrics remain unavailable."""

    return TaskEfficiencyReceipt(
        identity=identity,
        model=model or _unavailable_model(),
        compute=compute or _unavailable_compute(),
        cost=cost or _unavailable_cost(),
        work=work or _unavailable_work(),
        quality=quality or _unavailable_quality(),
        terminal=terminal
        or TerminalRecord(
            final_task_outcome=FinalTaskOutcome.UNAVAILABLE,
            validation_result=ValidationResult.UNAVAILABLE,
            patch_disposition=PatchDisposition.UNAVAILABLE,
            population_kind=PopulationKind.UNAVAILABLE,
        ),
    )


def _admit_canonical(cls: type[_TContract], value: Mapping[str, Any]) -> _TContract:
    try:
        encoded = canonical_json_bytes(value)
    except ContractValidationError as exc:
        raise EfficiencyReceiptError("payload is not canonical bytes") from exc
    return cls.from_canonical_bytes(encoded)


def admit_task_efficiency_receipt(
    value: Mapping[str, Any] | bytes | bytearray | TaskEfficiencyReceipt,
) -> TaskEfficiencyReceipt:
    if isinstance(value, TaskEfficiencyReceipt):
        return TaskEfficiencyReceipt.from_canonical_bytes(value.canonical_bytes())
    if isinstance(value, (bytes, bytearray)):
        return TaskEfficiencyReceipt.from_canonical_bytes(bytes(value))
    return _admit_canonical(TaskEfficiencyReceipt, _mapping(value, "task efficiency receipt"))


def admit_paired_benchmark_manifest(
    value: Mapping[str, Any] | bytes | bytearray | PairedBenchmarkManifest,
) -> PairedBenchmarkManifest:
    if isinstance(value, PairedBenchmarkManifest):
        return PairedBenchmarkManifest.from_canonical_bytes(value.canonical_bytes())
    if isinstance(value, (bytes, bytearray)):
        return PairedBenchmarkManifest.from_canonical_bytes(bytes(value))
    return _admit_canonical(
        PairedBenchmarkManifest, _mapping(value, "paired benchmark manifest")
    )


__all__ = (
    "ArmId",
    "CONTRACT_VERSION",
    "CostUse",
    "ComputeUse",
    "DISTINGUISHED_TRUTH_STATES",
    "EfficiencyReceiptError",
    "EqualControls",
    "EvidenceState",
    "FinalTaskOutcome",
    "IdentityEvidence",
    "MODEL_CLASSES",
    "ModelUse",
    "PAIRED_BENCHMARK_MANIFEST_SCHEMA",
    "PAIRED_BENCHMARK_MANIFEST_SCHEMA_PATH",
    "PairedArm",
    "PairedBenchmarkManifest",
    "PatchDisposition",
    "PopulationKind",
    "Populations",
    "QualityUse",
    "QuantitativeEvidence",
    "ReceiptIdentity",
    "RequestIdEvidence",
    "TASK_EFFICIENCY_RECEIPT_SCHEMA",
    "TASK_EFFICIENCY_RECEIPT_SCHEMA_PATH",
    "TRUTH_STATES",
    "TaskEfficiencyReceipt",
    "TerminalRecord",
    "UNIT_COUNT",
    "UNIT_MICROUSD",
    "UNIT_TOKENS",
    "UNAVAILABLE_REASONS",
    "ValidationResult",
    "WorkUse",
    "admit_paired_benchmark_manifest",
    "admit_task_efficiency_receipt",
    "canonical_paired_arms",
    "fixture_cid",
    "load_paired_benchmark_manifest_schema",
    "load_task_efficiency_receipt_schema",
    "sensor_id_for",
    "unavailable_identity",
    "unavailable_quantitative",
    "unavailable_request_ids",
    "unavailable_task_efficiency_receipt",
    "validate_closed_schema",
)
