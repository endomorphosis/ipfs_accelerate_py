"""Closed ASEH task-efficiency receipts and paired-benchmark manifests.

The wire contracts live beside this module as JSON Schema documents.  Admission
is fail-closed: unknown fields, floats, boolean-as-integer, out-of-bound
values, and conflated truth states are rejected.  Missing telemetry remains
``unavailable`` and is never encoded as numeric zero.

Measured, estimated, unavailable, attempted, observed, verified, and simulated
are distinct tagged states.  Schema authority here cannot attest a measurement
or complete a task.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Final, Iterator

from ..proof.formal_verification_contracts import (
    ContractValidationError,
    canonical_json_bytes as _contract_canonical_json_bytes,
    content_identity as _contract_content_identity,
)


EFFICIENCY_RECEIPT_CONTRACT_VERSION: Final[int] = 1
SCHEMA_VERSION: Final[int] = EFFICIENCY_RECEIPT_CONTRACT_VERSION

TASK_EFFICIENCY_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/aseh-task-efficiency-receipt@1"
)
PAIRED_BENCHMARK_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/aseh-paired-benchmark-manifest@1"
)
TASK_EFFICIENCY_RECEIPT_INTERFACE: Final[str] = "TaskEfficiencyReceipt@1"
PAIRED_BENCHMARK_MANIFEST_INTERFACE: Final[str] = "PairedBenchmarkManifest@1"

SCHEMA_DIR: Final[Path] = Path(__file__).resolve().parent / "schemas"
TASK_EFFICIENCY_RECEIPT_SCHEMA_PATH: Final[Path] = (
    SCHEMA_DIR / "task_efficiency_receipt.schema.json"
)
PAIRED_BENCHMARK_MANIFEST_SCHEMA_PATH: Final[Path] = (
    SCHEMA_DIR / "paired_benchmark_manifest.schema.json"
)

MAX_INTEGER: Final[int] = 10**18
MAX_TEXT_BYTES: Final[int] = 512
MAX_ARRAY_ITEMS: Final[int] = 256
MAX_RETRIES: Final[int] = 32
MAX_SERIALIZED_BYTES: Final[int] = 262_144

PROGRAM_ID: Final[str] = "agent-supervisor-efficiency-and-state-hardening-v1"
OBJECTIVE_ID: Final[str] = "ASEH-G020"
HERMETIC_MINIMUM: Final[int] = 60
HISTORICAL_MINIMUM: Final[int] = 20
LIVE_MINIMUM: Final[int] = 10
LIVE_ENROLLMENT_DEADLINE_MAXIMUM_DAYS: Final[int] = 30

CID_RE: Final[re.Pattern[str]] = re.compile(r"^b[a-z2-7]{20,}$")
GIT_OID_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")
DIGEST_RE: Final[re.Pattern[str]] = re.compile(r"^sha256:[0-9a-f]{64}$")
UTC_RE: Final[re.Pattern[str]] = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)
FIELD_PATH_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_.]*$")
IDENTITY_REF_RE: Final[re.Pattern[str]] = re.compile(
    r"^(unavailable|b[a-z2-7]{20,}|sha256:[0-9a-f]{64})$"
)
_SCHEMA_CACHE: dict[str, dict[str, Any]] = {}

MODEL_CLASSES: Final[tuple[str, ...]] = (
    "deterministic",
    "local_small_model",
    "local_medium_model",
    "remote_standard_model",
    "remote_frontier_model",
    "human",
)
PATCH_DISPOSITIONS: Final[tuple[str, ...]] = (
    "accepted",
    "rejected",
    "quarantined",
    "reverted",
)
TASK_OUTCOMES: Final[tuple[str, ...]] = (
    "succeeded",
    "failed",
    "retried",
    "rescued",
    "conflicted",
    "human_escalated",
    "quarantined",
    "compensated",
)
HISTORICAL_OUTCOMES: Final[tuple[str, ...]] = (
    "successful",
    "failed",
    "retried",
    "rescued",
    "conflicted",
    "human_escalated",
)
POPULATION_KINDS: Final[tuple[str, ...]] = (
    "hermetic_development",
    "historical_exact_tree_replay",
    "new_live_shadow_canary",
)
PAIRED_ARMS: Final[tuple[str, ...]] = (
    "direct_minimal_orchestration_baseline",
    "sealed_current_supervisor_baseline",
    "candidate_optimized_supervisor",
)
UNITS: Final[tuple[str, ...]] = (
    "count",
    "tokens",
    "bytes",
    "seconds_millionths",
    "microusd",
    "compute_units",
    "ratio_millionths",
)
REASON_CODES: Final[tuple[str, ...]] = (
    "not_reported",
    "sensor_absent",
    "provider_omitted",
    "collection_failed",
    "permission_denied",
    "hardware_absent",
    "not_applicable",
    "not_yet_measured",
    "fixture_only",
    "deadline_elapsed",
    "not_sealed",
    "not_admitted",
)
ESTIMATOR_METHODS: Final[tuple[str, ...]] = (
    "token_price_snapshot",
    "provider_quote",
    "local_compute_model",
    "manual_audit",
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
QUANTITY_TRUTH_STATES: Final[frozenset[str]] = frozenset(
    {"measured", "estimated", "unavailable", "simulated"}
)
OPERATION_TRUTH_STATES: Final[frozenset[str]] = frozenset(
    {"attempted", "observed", "verified", "unavailable", "simulated"}
)
IDENTITY_TRUTH_STATES: Final[frozenset[str]] = frozenset(
    {"observed", "verified", "attempted", "unavailable", "simulated"}
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
EQUAL_CONTROL_FIELDS: Final[tuple[str, ...]] = (
    "repository_revision",
    "objective",
    "task_inputs",
    "acceptance_tests",
    "available_providers_and_models",
    "price_accounting",
    "resource_limits",
    "maximum_retries",
    "human_intervention_policy",
)
IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "task_id",
    "task_cid",
    "objective_id",
    "objective_revision",
    "repository_commit",
    "repository_tree",
    "policy_identity",
    "context_pack_cid",
    "interface_schema_identities",
    "provider_model",
)
MODEL_USE_FIELDS: Final[tuple[str, ...]] = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
    "number_of_calls",
    "calls_by_model_class",
    "safe_provider_request_ids",
    "provider_reported_usage",
)
COMPUTE_FIELDS: Final[tuple[str, ...]] = (
    "cpu_seconds",
    "gpu_seconds",
    "peak_memory",
    "wall_clock_duration",
    "test_execution_time",
    "prover_execution_time",
    "static_analysis_time",
    "indexing_retrieval_time",
    "bytes_read",
    "bytes_written",
    "process_count",
    "concurrency",
)
COST_FIELDS: Final[tuple[str, ...]] = (
    "provider_reported_charge",
    "token_based_estimated_charge",
    "price_snapshot_identity",
    "price_snapshot_timestamp",
    "local_compute_units",
    "audit_and_verification_overhead",
    "cost_per_accepted_patch",
    "cost_per_completed_task",
)
WORK_FIELDS: Final[tuple[str, ...]] = (
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
    "final_task_outcome",
    "validation_result",
    "patch_disposition",
)
QUALITY_FIELDS: Final[tuple[str, ...]] = (
    "accepted_patch_quality",
    "false_positives",
    "false_negatives",
    "selected_test_false_negatives",
    "quality_adjusted_cost",
    "accepted_patch_rate",
)
TERMINAL_FIELDS: Final[tuple[str, ...]] = (
    "final_task_outcome",
    "validation_result",
    "patch_disposition",
    "time_to_terminal_outcome",
    "terminalized",
    "single_terminalization",
)
EVIDENCE_STATE_FIELDS: Final[tuple[str, ...]] = (
    "population_kind",
    "live",
    "simulated_as_live",
    "estimated_as_measured",
    "attempted_as_observed",
    "observed_as_verified",
    "unavailable_as_zero",
)
STATISTIC_FIELDS: Final[tuple[str, ...]] = (
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

_JSON_TYPES: Final[dict[str, type | tuple[type, ...]]] = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
}


class EfficiencyReceiptError(ValueError):
    """Closed efficiency receipt or paired-manifest contract violation."""


class TruthState(str, Enum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"
    ATTEMPTED = "attempted"
    OBSERVED = "observed"
    VERIFIED = "verified"
    SIMULATED = "simulated"


class ModelClass(str, Enum):
    DETERMINISTIC = "deterministic"
    LOCAL_SMALL_MODEL = "local_small_model"
    LOCAL_MEDIUM_MODEL = "local_medium_model"
    REMOTE_STANDARD_MODEL = "remote_standard_model"
    REMOTE_FRONTIER_MODEL = "remote_frontier_model"
    HUMAN = "human"


class PatchDisposition(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"
    REVERTED = "reverted"


class TaskOutcome(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RETRIED = "retried"
    RESCUED = "rescued"
    CONFLICTED = "conflicted"
    HUMAN_ESCALATED = "human_escalated"
    QUARANTINED = "quarantined"
    COMPENSATED = "compensated"


class PopulationKind(str, Enum):
    HERMETIC_DEVELOPMENT = "hermetic_development"
    HISTORICAL_EXACT_TREE_REPLAY = "historical_exact_tree_replay"
    NEW_LIVE_SHADOW_CANARY = "new_live_shadow_canary"


class ReasonCode(str, Enum):
    NOT_REPORTED = "not_reported"
    SENSOR_ABSENT = "sensor_absent"
    PROVIDER_OMITTED = "provider_omitted"
    COLLECTION_FAILED = "collection_failed"
    PERMISSION_DENIED = "permission_denied"
    HARDWARE_ABSENT = "hardware_absent"
    NOT_APPLICABLE = "not_applicable"
    NOT_YET_MEASURED = "not_yet_measured"
    FIXTURE_ONLY = "fixture_only"
    DEADLINE_ELAPSED = "deadline_elapsed"
    NOT_SEALED = "not_sealed"
    NOT_ADMITTED = "not_admitted"


# ---------------------------------------------------------------------------
# Canonical bytes
# ---------------------------------------------------------------------------


def _reject_floats(value: Any, *, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) is int:
        return
    if isinstance(value, float):
        raise EfficiencyReceiptError(f"{path}: canonical contracts cannot contain floats")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise EfficiencyReceiptError(f"{path}: object keys must be strings")
            _reject_floats(child, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_floats(child, path=f"{path}[{index}]")
        return
    raise EfficiencyReceiptError(
        f"{path}: unsupported canonical value {type(value).__name__}"
    )


def canonical_bytes(value: Any) -> bytes:
    """Encode DAG-JSON-compatible UTF-8 bytes; reject floats."""

    _reject_floats(value)
    try:
        encoded = _contract_canonical_json_bytes(value)
    except ContractValidationError as exc:
        raise EfficiencyReceiptError("canonical encoding failed") from exc
    if len(encoded) > MAX_SERIALIZED_BYTES:
        raise EfficiencyReceiptError("canonical payload exceeds the serialized-byte bound")
    return encoded


def canonical_json(value: Any) -> str:
    return canonical_bytes(value).decode("utf-8")


def content_identity(value: Any) -> str:
    """Return a CIDv1 DAG-JSON/sha2-256 identity."""

    _reject_floats(value)
    try:
        return _contract_content_identity(value)
    except ContractValidationError as exc:
        raise EfficiencyReceiptError("content identity encoding failed") from exc


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EfficiencyReceiptError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise EfficiencyReceiptError(f"{name} keys must be strings")
    return {str(key): item for key, item in value.items()}


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EfficiencyReceiptError(f"schema document is unreadable: {path.name}") from exc
    if not isinstance(payload, dict):
        raise EfficiencyReceiptError(f"{path.name} must contain a JSON object")
    return payload


def _assert_schema_document_is_closed(schema: Mapping[str, Any], *, name: str) -> None:
    """Reject any object schema that does not fail closed on unknown fields."""

    def walk(node: Any, *, path: str) -> None:
        if isinstance(node, Mapping):
            if node.get("type") == "object" or "properties" in node:
                if node.get("additionalProperties") is not False:
                    raise EfficiencyReceiptError(
                        f"{name} {path} is not a closed object schema"
                    )
            for key, child in node.items():
                walk(child, path=f"{path}.{key}")
            return
        if isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, path=f"{path}[{index}]")

    walk(schema, path="$")


def load_task_efficiency_receipt_schema() -> dict[str, Any]:
    cached = _SCHEMA_CACHE.get("receipt")
    if cached is not None:
        return cached
    schema = _load_json_object(TASK_EFFICIENCY_RECEIPT_SCHEMA_PATH)
    _assert_schema_document_is_closed(schema, name="task efficiency receipt schema")
    if schema.get("$id") != TASK_EFFICIENCY_RECEIPT_SCHEMA:
        raise EfficiencyReceiptError("task efficiency receipt schema $id mismatch")
    _SCHEMA_CACHE["receipt"] = schema
    return schema


def load_paired_benchmark_manifest_schema() -> dict[str, Any]:
    cached = _SCHEMA_CACHE.get("manifest")
    if cached is not None:
        return cached
    schema = _load_json_object(PAIRED_BENCHMARK_MANIFEST_SCHEMA_PATH)
    _assert_schema_document_is_closed(schema, name="paired benchmark manifest schema")
    if schema.get("$id") != PAIRED_BENCHMARK_MANIFEST_SCHEMA:
        raise EfficiencyReceiptError("paired benchmark manifest schema $id mismatch")
    _SCHEMA_CACHE["manifest"] = schema
    return schema


# ---------------------------------------------------------------------------
# Closed JSON Schema subset validator
# ---------------------------------------------------------------------------


def _resolve_ref(schema: Mapping[str, Any], defs: Mapping[str, Any]) -> Mapping[str, Any]:
    ref = schema.get("$ref")
    if not isinstance(ref, str):
        return schema
    prefix = "#/$defs/"
    if not ref.startswith(prefix) or "/" in ref[len(prefix):]:
        raise EfficiencyReceiptError("unsupported schema $ref")
    name = ref[len(prefix):]
    target = defs.get(name)
    if not isinstance(target, Mapping):
        raise EfficiencyReceiptError(f"unknown schema $defs entry {name}")
    return target


def _type_matches(instance: Any, expected: str) -> bool:
    if expected == "integer":
        return type(instance) is int
    python_type = _JSON_TYPES.get(expected)
    if python_type is None:
        return False
    return isinstance(instance, python_type)


def _canonical_item(value: Any) -> bytes:
    return canonical_bytes(value)


def validate_against_schema(
    instance: Any,
    schema: Mapping[str, Any],
    *,
    defs: Mapping[str, Any] | None = None,
    path: str = "$",
) -> None:
    """Validate ``instance`` against a closed JSON Schema subset."""

    resolved = _resolve_ref(schema, defs or {})
    local_defs = resolved.get("$defs") if isinstance(resolved.get("$defs"), Mapping) else None
    active_defs: Mapping[str, Any] = local_defs if local_defs is not None else (defs or {})
    if local_defs is None and "$defs" in schema and isinstance(schema["$defs"], Mapping):
        active_defs = schema["$defs"]

    if "oneOf" in resolved:
        variants = resolved["oneOf"]
        if not isinstance(variants, list) or not variants:
            raise EfficiencyReceiptError(f"{path}: oneOf must be a nonempty array")
        matches = 0
        first_error = None
        for variant in variants:
            if not isinstance(variant, Mapping):
                raise EfficiencyReceiptError(f"{path}: oneOf entries must be objects")
            try:
                validate_against_schema(instance, variant, defs=active_defs, path=path)
            except EfficiencyReceiptError as exc:
                if first_error is None:
                    first_error = exc
                continue
            matches += 1
        if matches != 1:
            detail = f": {first_error}" if first_error is not None and matches == 0 else ""
            raise EfficiencyReceiptError(
                f"{path}: does not uniquely match a closed variant{detail}"
            )
        return

    if "const" in resolved and instance != resolved["const"]:
        raise EfficiencyReceiptError(f"{path}: must equal the closed const")
    if "enum" in resolved:
        allowed = resolved["enum"]
        if not isinstance(allowed, list) or instance not in allowed:
            raise EfficiencyReceiptError(f"{path}: is not an allowed enum value")

    expected_type = resolved.get("type")
    if expected_type is not None:
        if not isinstance(expected_type, str) or not _type_matches(instance, expected_type):
            raise EfficiencyReceiptError(f"{path}: has the wrong JSON type")

    if expected_type == "string" or isinstance(instance, str):
        if isinstance(instance, str):
            if "minLength" in resolved and len(instance) < int(resolved["minLength"]):
                raise EfficiencyReceiptError(f"{path}: is shorter than minLength")
            if "maxLength" in resolved and len(instance) > int(resolved["maxLength"]):
                raise EfficiencyReceiptError(f"{path}: exceeds maxLength")
            pattern = resolved.get("pattern")
            if isinstance(pattern, str) and re.search(pattern, instance) is None:
                raise EfficiencyReceiptError(f"{path}: does not match the closed pattern")
            encoded = instance.encode("utf-8")
            if b"\x00" in encoded or len(encoded) > MAX_TEXT_BYTES:
                raise EfficiencyReceiptError(f"{path}: is unsafe or too large")

    if expected_type == "integer" or type(instance) is int:
        if type(instance) is int:
            if "minimum" in resolved and instance < int(resolved["minimum"]):
                raise EfficiencyReceiptError(f"{path}: is below the minimum bound")
            if "maximum" in resolved and instance > int(resolved["maximum"]):
                raise EfficiencyReceiptError(f"{path}: exceeds the maximum bound")
            if instance < 0 or instance > MAX_INTEGER:
                raise EfficiencyReceiptError(f"{path}: is outside the integer bound")

    if expected_type == "object" or isinstance(instance, Mapping):
        if not isinstance(instance, Mapping):
            if "properties" in resolved or expected_type == "object":
                raise EfficiencyReceiptError(f"{path}: must be an object")
        else:
            properties = resolved.get("properties")
            if not isinstance(properties, Mapping):
                properties = {}
            if resolved.get("additionalProperties") is False:
                unknown = sorted(set(instance) - set(properties))
                if unknown:
                    raise EfficiencyReceiptError(
                        f"{path}: contains unknown fields: {unknown}"
                    )
            required = resolved.get("required")
            if isinstance(required, list):
                missing = [name for name in required if name not in instance]
                if missing:
                    raise EfficiencyReceiptError(
                        f"{path}: missing required fields: {missing}"
                    )
            for key, child_schema in properties.items():
                if key in instance:
                    if not isinstance(child_schema, Mapping):
                        raise EfficiencyReceiptError(f"{path}.{key}: invalid nested schema")
                    validate_against_schema(
                        instance[key],
                        child_schema,
                        defs=active_defs,
                        path=f"{path}.{key}",
                    )

    if expected_type == "array" or isinstance(instance, list):
        if isinstance(instance, list):
            if "minItems" in resolved and len(instance) < int(resolved["minItems"]):
                raise EfficiencyReceiptError(f"{path}: has too few items")
            if "maxItems" in resolved and len(instance) > int(resolved["maxItems"]):
                raise EfficiencyReceiptError(f"{path}: exceeds maxItems")
            if len(instance) > MAX_ARRAY_ITEMS:
                raise EfficiencyReceiptError(f"{path}: exceeds the array bound")
            items_schema = resolved.get("items")
            if isinstance(items_schema, Mapping):
                for index, item in enumerate(instance):
                    validate_against_schema(
                        item,
                        items_schema,
                        defs=active_defs,
                        path=f"{path}[{index}]",
                    )
            if resolved.get("uniqueItems") is True:
                encoded_items = [_canonical_item(item) for item in instance]
                if len(set(encoded_items)) != len(encoded_items):
                    raise EfficiencyReceiptError(f"{path}: items must be unique")


# ---------------------------------------------------------------------------
# Evidence constructors
# ---------------------------------------------------------------------------


def _reason(code: ReasonCode | str) -> str:
    raw = code.value if isinstance(code, ReasonCode) else str(code)
    if raw not in REASON_CODES:
        raise EfficiencyReceiptError("reason_code is not a closed unavailable reason")
    return raw


def _unit(unit: str) -> str:
    if unit not in UNITS:
        raise EfficiencyReceiptError("unit is not a closed quantity unit")
    return unit


def _method(method: str) -> str:
    if method not in ESTIMATOR_METHODS:
        raise EfficiencyReceiptError("estimator method is not a closed value")
    return method


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise EfficiencyReceiptError(f"{name} must be text")
    result = value.strip()
    if not result:
        raise EfficiencyReceiptError(f"{name} must not be empty")
    encoded = result.encode("utf-8")
    if b"\x00" in encoded or len(encoded) > MAX_TEXT_BYTES:
        raise EfficiencyReceiptError(f"{name} is unsafe or too large")
    return result


def _int(value: Any, *, name: str, minimum: int = 0, maximum: int = MAX_INTEGER) -> int:
    if type(value) is not int:
        raise EfficiencyReceiptError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise EfficiencyReceiptError(f"{name} must be between {minimum} and {maximum}")
    return value


def unavailable(reason: ReasonCode | str = ReasonCode.NOT_YET_MEASURED) -> dict[str, str]:
    return {"truth_state": TruthState.UNAVAILABLE.value, "reason_code": _reason(reason)}


def simulated(
    fixture_id: str,
    reason: ReasonCode | str = ReasonCode.FIXTURE_ONLY,
) -> dict[str, str]:
    return {
        "truth_state": TruthState.SIMULATED.value,
        "fixture_id": _text(fixture_id, name="fixture_id"),
        "reason_code": _reason(reason),
    }


def measured_quantity(value: int, *, unit: str, sensor_id: str) -> dict[str, Any]:
    return {
        "truth_state": TruthState.MEASURED.value,
        "value": _int(value, name="value"),
        "unit": _unit(unit),
        "sensor_id": _text(sensor_id, name="sensor_id"),
    }


def estimated_quantity(
    value: int,
    *,
    unit: str,
    estimator_id: str,
    method: str,
    price_snapshot_identity: str = "unavailable",
) -> dict[str, Any]:
    identity = _text(price_snapshot_identity, name="price_snapshot_identity")
    if IDENTITY_REF_RE.fullmatch(identity) is None:
        raise EfficiencyReceiptError("price_snapshot_identity is not a closed identity ref")
    return {
        "truth_state": TruthState.ESTIMATED.value,
        "value": _int(value, name="value"),
        "unit": _unit(unit),
        "estimator_id": _text(estimator_id, name="estimator_id"),
        "method": _method(method),
        "price_snapshot_identity": identity,
    }


def attempted_operation(operation_id: str, *, attempt_count: int = 1) -> dict[str, Any]:
    return {
        "truth_state": TruthState.ATTEMPTED.value,
        "operation_id": _text(operation_id, name="operation_id"),
        "attempt_count": _int(attempt_count, name="attempt_count", minimum=1),
    }


def observed_operation(
    operation_id: str,
    *,
    observer_id: str,
    count: int,
) -> dict[str, Any]:
    return {
        "truth_state": TruthState.OBSERVED.value,
        "operation_id": _text(operation_id, name="operation_id"),
        "observer_id": _text(observer_id, name="observer_id"),
        "count": _int(count, name="count"),
    }


def verified_operation(
    operation_id: str,
    *,
    observer_id: str,
    count: int,
    verifier_id: str,
    verifier_receipt_cid: str,
) -> dict[str, Any]:
    cid = _text(verifier_receipt_cid, name="verifier_receipt_cid")
    if CID_RE.fullmatch(cid) is None:
        raise EfficiencyReceiptError("verifier_receipt_cid must be a CIDv1")
    return {
        "truth_state": TruthState.VERIFIED.value,
        "operation_id": _text(operation_id, name="operation_id"),
        "observer_id": _text(observer_id, name="observer_id"),
        "count": _int(count, name="count"),
        "verifier_id": _text(verifier_id, name="verifier_id"),
        "verifier_receipt_cid": cid,
    }


def observed_identity(value: str) -> dict[str, str]:
    return {
        "truth_state": TruthState.OBSERVED.value,
        "value": _text(value, name="value"),
    }


def verified_identity(
    value: str,
    *,
    verifier_id: str,
    verifier_receipt_cid: str,
) -> dict[str, str]:
    cid = _text(verifier_receipt_cid, name="verifier_receipt_cid")
    if CID_RE.fullmatch(cid) is None:
        raise EfficiencyReceiptError("verifier_receipt_cid must be a CIDv1")
    return {
        "truth_state": TruthState.VERIFIED.value,
        "value": _text(value, name="value"),
        "verifier_id": _text(verifier_id, name="verifier_id"),
        "verifier_receipt_cid": cid,
    }


def attempted_identity(operation_id: str) -> dict[str, str]:
    return {
        "truth_state": TruthState.ATTEMPTED.value,
        "operation_id": _text(operation_id, name="operation_id"),
    }


def observed_request_ids(values: Sequence[str]) -> dict[str, Any]:
    items = tuple(dict.fromkeys(_text(item, name="request_id") for item in values))
    if len(items) > MAX_ARRAY_ITEMS:
        raise EfficiencyReceiptError("safe_provider_request_ids exceeds the array bound")
    return {"truth_state": TruthState.OBSERVED.value, "values": list(items)}


def verified_request_ids(
    values: Sequence[str],
    *,
    verifier_id: str,
    verifier_receipt_cid: str,
) -> dict[str, Any]:
    payload = observed_request_ids(values)
    cid = _text(verifier_receipt_cid, name="verifier_receipt_cid")
    if CID_RE.fullmatch(cid) is None:
        raise EfficiencyReceiptError("verifier_receipt_cid must be a CIDv1")
    payload["truth_state"] = TruthState.VERIFIED.value
    payload["verifier_id"] = _text(verifier_id, name="verifier_id")
    payload["verifier_receipt_cid"] = cid
    return payload


def unavailable_calls_by_model_class(
    reason: ReasonCode | str = ReasonCode.NOT_YET_MEASURED,
) -> dict[str, Any]:
    blank = unavailable(reason)
    return {name: dict(blank) for name in MODEL_CLASSES}


def unavailable_provider_reported_usage(
    reason: ReasonCode | str = ReasonCode.PROVIDER_OMITTED,
) -> dict[str, Any]:
    return unavailable(reason)


# ---------------------------------------------------------------------------
# Tree walkers and semantic rules
# ---------------------------------------------------------------------------


def _truth_state_of(value: Any) -> str | None:
    if isinstance(value, Mapping) and "truth_state" in value:
        state = value.get("truth_state")
        if not isinstance(state, str) or state not in TRUTH_STATES:
            raise EfficiencyReceiptError("truth_state is not a closed evidence state")
        return state
    return None


def iter_evidence_nodes(
    value: Any,
    *,
    path: str = "",
) -> Iterator[tuple[str, Mapping[str, Any], str]]:
    """Yield ``(path, node, truth_state)`` for every tagged evidence object."""

    if isinstance(value, Mapping):
        state = _truth_state_of(value)
        if state is not None:
            yield path, value, state
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from iter_evidence_nodes(child, path=child_path)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            yield from iter_evidence_nodes(child, path=child_path)


def collect_unavailable_fields(value: Any) -> tuple[str, ...]:
    paths = [
        path
        for path, _node, state in iter_evidence_nodes(value)
        if state == TruthState.UNAVAILABLE.value and path
    ]
    return tuple(sorted(dict.fromkeys(paths)))


def _assert_truth_state_payload(node: Mapping[str, Any], state: str, *, path: str) -> None:
    if state == TruthState.MEASURED.value:
        if "estimator_id" in node or "method" in node:
            raise EfficiencyReceiptError(f"{path}: measured values cannot carry an estimator")
        if "reason_code" in node:
            raise EfficiencyReceiptError(f"{path}: measured values cannot carry an unavailable reason")
        if "verifier_receipt_cid" in node:
            raise EfficiencyReceiptError(f"{path}: measured values are not verified operations")
    elif state == TruthState.ESTIMATED.value:
        if "sensor_id" in node:
            raise EfficiencyReceiptError(f"{path}: estimated values cannot be labeled measured")
        if node.get("reason_code") == "measured":
            raise EfficiencyReceiptError(f"{path}: estimated values cannot claim measurement")
    elif state == TruthState.UNAVAILABLE.value:
        for forbidden in ("value", "count", "unit", "sensor_id", "estimator_id"):
            if forbidden in node:
                raise EfficiencyReceiptError(
                    f"{path}: unavailable evidence cannot encode {forbidden}"
                )
        if node.get("value", "unavailable") == 0 or node.get("count", "unavailable") == 0:
            raise EfficiencyReceiptError(f"{path}: unavailable evidence cannot encode numeric zero")
    elif state == TruthState.ATTEMPTED.value:
        if any(key in node for key in ("observer_id", "verifier_id", "verifier_receipt_cid", "count")):
            raise EfficiencyReceiptError(
                f"{path}: attempted operations cannot be represented as observed or verified"
            )
    elif state == TruthState.OBSERVED.value:
        if "verifier_id" in node or "verifier_receipt_cid" in node:
            raise EfficiencyReceiptError(
                f"{path}: observed operations cannot be represented as verified"
            )
    elif state == TruthState.VERIFIED.value:
        if "verifier_receipt_cid" not in node or "verifier_id" not in node:
            raise EfficiencyReceiptError(
                f"{path}: verified evidence requires an admitted verifier receipt"
            )
    elif state == TruthState.SIMULATED.value:
        if any(key in node for key in ("sensor_id", "verifier_receipt_cid")):
            raise EfficiencyReceiptError(f"{path}: simulated evidence cannot claim live verification")


def _assert_semantic_invariants(payload: Mapping[str, Any], *, live: bool) -> None:
    for path, node, state in iter_evidence_nodes(payload):
        _assert_truth_state_payload(node, state, path=path or "$")
        if live and state == TruthState.SIMULATED.value:
            raise EfficiencyReceiptError(
                f"{path or '$'}: simulated evidence cannot be represented as live"
            )
        if state == TruthState.UNAVAILABLE.value:
            for key, child in node.items():
                if type(child) is int and child == 0:
                    raise EfficiencyReceiptError(
                        f"{path}.{key}: unavailable evidence cannot encode numeric zero"
                    )


def _require_matching_unavailable_index(
    payload: Mapping[str, Any],
    claimed: Any,
    *,
    cid_field: str,
) -> tuple[str, ...]:
    body = {key: value for key, value in payload.items() if key != cid_field}
    actual = collect_unavailable_fields(body)
    if not isinstance(claimed, list) or any(not isinstance(item, str) for item in claimed):
        raise EfficiencyReceiptError("explicit_unavailable_fields must be a string array")
    claimed_tuple = tuple(claimed)
    if claimed_tuple != actual:
        raise EfficiencyReceiptError(
            "explicit_unavailable_fields must list every unavailable evidence path"
        )
    return actual


def _bind_cid(payload: dict[str, Any], *, field: str) -> str:
    body = {key: value for key, value in payload.items() if key != field}
    cid = content_identity(body)
    claimed = payload.get(field)
    if claimed not in (None, "", cid):
        raise EfficiencyReceiptError(f"{field} does not match canonical content identity")
    payload[field] = cid
    return cid


def _assert_evidence_state_flags(payload: Mapping[str, Any]) -> None:
    evidence = payload.get("evidence_state")
    if not isinstance(evidence, Mapping):
        raise EfficiencyReceiptError("evidence_state is required")
    live = evidence.get("live")
    if type(live) is not bool:
        raise EfficiencyReceiptError("evidence_state.live must be a boolean")
    population = evidence.get("population_kind")
    if population not in POPULATION_KINDS:
        raise EfficiencyReceiptError("population_kind is not a closed value")
    if population == PopulationKind.HERMETIC_DEVELOPMENT.value and live:
        raise EfficiencyReceiptError("hermetic evidence cannot be represented as live")
    if live and population != PopulationKind.NEW_LIVE_SHADOW_CANARY.value:
        raise EfficiencyReceiptError("only the live shadow/canary population may be marked live")
    for flag in (
        "simulated_as_live",
        "estimated_as_measured",
        "attempted_as_observed",
        "observed_as_verified",
        "unavailable_as_zero",
    ):
        if evidence.get(flag) is not False:
            raise EfficiencyReceiptError(f"evidence_state.{flag} must remain false")
    _assert_semantic_invariants(payload, live=live)


def _assert_terminal_alignment(payload: Mapping[str, Any]) -> None:
    work = payload.get("work")
    terminal = payload.get("terminal")
    if not isinstance(work, Mapping) or not isinstance(terminal, Mapping):
        raise EfficiencyReceiptError("work and terminal sections are required")
    if work.get("final_task_outcome") != terminal.get("final_task_outcome"):
        raise EfficiencyReceiptError("terminal outcome must match work.final_task_outcome")
    if work.get("patch_disposition") != terminal.get("patch_disposition"):
        raise EfficiencyReceiptError("terminal patch_disposition must match work")
    if work.get("validation_result") != terminal.get("validation_result"):
        raise EfficiencyReceiptError("terminal validation_result must match work")
    if terminal.get("terminalized") is not True or terminal.get("single_terminalization") is not True:
        raise EfficiencyReceiptError("a receipt requires single terminalization")


# ---------------------------------------------------------------------------
# Admitted records
# ---------------------------------------------------------------------------


def _decode_json(payload: Any, *, name: str) -> dict[str, Any]:
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = bytes(payload).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EfficiencyReceiptError(f"{name} is not UTF-8") from exc
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise EfficiencyReceiptError(f"{name} JSON is malformed") from exc
    return _mapping(payload, name=name)


@dataclass(frozen=True)
class _AdmittedRecord:
    SCHEMA: ClassVar[str] = ""
    CID_FIELD: ClassVar[str] = ""

    _payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        encoded = json.loads(canonical_bytes(self._payload).decode("utf-8"))
        if not isinstance(encoded, dict):
            raise EfficiencyReceiptError("canonical payload must remain an object")
        return encoded

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def canonical_json(self) -> str:
        return self.canonical_bytes().decode("utf-8")

    def to_json(self) -> str:
        return self.canonical_json()

    @property
    def content_id(self) -> str:
        return str(self._payload[self.CID_FIELD])

    @property
    def cid(self) -> str:
        return self.content_id

    @property
    def schema(self) -> str:
        return self.SCHEMA

    @property
    def schema_version(self) -> int:
        return SCHEMA_VERSION

    def round_trip(self) -> bytes:
        encoded = self.canonical_bytes()
        again = type(self).from_bytes(encoded)
        replayed = again.canonical_bytes()
        if replayed != encoded:
            raise EfficiencyReceiptError("canonical bytes did not round-trip")
        return encoded

    @classmethod
    def from_bytes(cls, payload: bytes | str) -> "_AdmittedRecord":
        return cls.from_dict(_decode_json(payload, name=cls.__name__))

    from_json = from_bytes

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "_AdmittedRecord":
        raise NotImplementedError


@dataclass(frozen=True)
class TaskEfficiencyReceipt(_AdmittedRecord):
    """One closed task-efficiency receipt."""

    SCHEMA: ClassVar[str] = TASK_EFFICIENCY_RECEIPT_SCHEMA
    CID_FIELD: ClassVar[str] = "receipt_cid"

    @property
    def receipt_cid(self) -> str:
        return self.content_id

    @property
    def identity(self) -> dict[str, Any]:
        return dict(self._payload["identity"])

    @property
    def model_use(self) -> dict[str, Any]:
        return dict(self._payload["model_use"])

    @property
    def compute(self) -> dict[str, Any]:
        return dict(self._payload["compute"])

    @property
    def cost(self) -> dict[str, Any]:
        return dict(self._payload["cost"])

    @property
    def work(self) -> dict[str, Any]:
        return dict(self._payload["work"])

    @property
    def quality(self) -> dict[str, Any]:
        return dict(self._payload["quality"])

    @property
    def evidence_state(self) -> dict[str, Any]:
        return dict(self._payload["evidence_state"])

    @property
    def terminal(self) -> dict[str, Any]:
        return dict(self._payload["terminal"])

    @property
    def explicit_unavailable_fields(self) -> tuple[str, ...]:
        return tuple(self._payload["explicit_unavailable_fields"])

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskEfficiencyReceipt":
        return admit_task_efficiency_receipt(payload)


@dataclass(frozen=True)
class PairedBenchmarkManifest(_AdmittedRecord):
    """Closed three-arm paired benchmark manifest."""

    SCHEMA: ClassVar[str] = PAIRED_BENCHMARK_MANIFEST_SCHEMA
    CID_FIELD: ClassVar[str] = "manifest_cid"

    @property
    def manifest_cid(self) -> str:
        return self.content_id

    @property
    def identity(self) -> dict[str, Any]:
        return dict(self._payload["identity"])

    @property
    def arms(self) -> dict[str, Any]:
        return dict(self._payload["arms"])

    @property
    def equal_controls(self) -> dict[str, Any]:
        return dict(self._payload["equal_controls"])

    @property
    def populations(self) -> dict[str, Any]:
        return dict(self._payload["populations"])

    @property
    def statistics(self) -> dict[str, Any]:
        return dict(self._payload["statistics"])

    @property
    def explicit_unavailable_fields(self) -> tuple[str, ...]:
        return tuple(self._payload["explicit_unavailable_fields"])

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PairedBenchmarkManifest":
        return admit_paired_benchmark_manifest(payload)


def admit_task_efficiency_receipt(
    payload: Mapping[str, Any] | str | bytes,
) -> TaskEfficiencyReceipt:
    data = _decode_json(payload, name="task efficiency receipt")
    schema = load_task_efficiency_receipt_schema()
    validate_against_schema(data, schema, path="$")
    _assert_evidence_state_flags(data)
    _assert_terminal_alignment(data)
    _require_matching_unavailable_index(
        data,
        data.get("explicit_unavailable_fields"),
        cid_field="receipt_cid",
    )
    admitted = json.loads(canonical_bytes(data).decode("utf-8"))
    _bind_cid(admitted, field="receipt_cid")
    validate_against_schema(admitted, schema, path="$")
    return TaskEfficiencyReceipt(_payload=admitted)


def admit_paired_benchmark_manifest(
    payload: Mapping[str, Any] | str | bytes,
) -> PairedBenchmarkManifest:
    data = _decode_json(payload, name="paired benchmark manifest")
    schema = load_paired_benchmark_manifest_schema()
    validate_against_schema(data, schema, path="$")
    _assert_semantic_invariants(data, live=False)
    if data.get("promotion_without_paired_campaign") is not False:
        raise EfficiencyReceiptError("promotion_without_paired_campaign must remain false")
    if data.get("hermetic_sufficient_for_production_promotion") is not False:
        raise EfficiencyReceiptError(
            "hermetic evidence cannot satisfy the live promotion gate"
        )
    _require_matching_unavailable_index(
        data,
        data.get("explicit_unavailable_fields"),
        cid_field="manifest_cid",
    )
    admitted = json.loads(canonical_bytes(data).decode("utf-8"))
    _bind_cid(admitted, field="manifest_cid")
    validate_against_schema(admitted, schema, path="$")
    return PairedBenchmarkManifest(_payload=admitted)


# ---------------------------------------------------------------------------
# Compact constructors for honest unavailable envelopes
# ---------------------------------------------------------------------------


def _provider_model(
    *,
    provider_id: str,
    model_id: str,
    model_class: str,
    model_revision: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if model_class not in MODEL_CLASSES:
        raise EfficiencyReceiptError("model_class is not a closed value")
    return {
        "provider_id": _text(provider_id, name="provider_id"),
        "model_id": _text(model_id, name="model_id"),
        "model_revision": dict(model_revision or unavailable(ReasonCode.NOT_REPORTED)),
        "model_class": model_class,
    }


def build_task_efficiency_receipt(
    *,
    task_id: str,
    task_cid: str,
    objective_id: str,
    objective_revision: str,
    repository_commit: str,
    repository_tree: str,
    policy_identity: str,
    provider_id: str,
    model_id: str,
    model_class: str,
    final_task_outcome: str,
    patch_disposition: str,
    population_kind: str,
    live: bool,
    context_pack_cid: Mapping[str, Any] | None = None,
    interface_schema_identities: Sequence[str] | None = None,
    model_revision: Mapping[str, Any] | None = None,
    model_use: Mapping[str, Any] | None = None,
    compute: Mapping[str, Any] | None = None,
    cost: Mapping[str, Any] | None = None,
    work: Mapping[str, Any] | None = None,
    quality: Mapping[str, Any] | None = None,
    terminal: Mapping[str, Any] | None = None,
    reason: ReasonCode | str = ReasonCode.NOT_YET_MEASURED,
) -> TaskEfficiencyReceipt:
    """Build a closed receipt; omitted measurements stay unavailable."""

    if final_task_outcome not in TASK_OUTCOMES:
        raise EfficiencyReceiptError("final_task_outcome is not a closed value")
    if patch_disposition not in PATCH_DISPOSITIONS:
        raise EfficiencyReceiptError("patch_disposition is not a closed value")
    if population_kind not in POPULATION_KINDS:
        raise EfficiencyReceiptError("population_kind is not a closed value")
    if GIT_OID_RE.fullmatch(repository_commit) is None:
        raise EfficiencyReceiptError("repository_commit must be a git object id")
    if GIT_OID_RE.fullmatch(repository_tree) is None:
        raise EfficiencyReceiptError("repository_tree must be a git object id")
    if CID_RE.fullmatch(task_cid) is None:
        raise EfficiencyReceiptError("task_cid must be a CIDv1")

    blank_quantity = unavailable(reason)
    blank_operation = unavailable(reason)
    blank_identity = unavailable(reason)
    default_validation = dict(blank_operation)
    default_model_use = {
        "input_tokens": dict(blank_quantity),
        "output_tokens": dict(blank_quantity),
        "cached_input_tokens": dict(blank_quantity),
        "reasoning_tokens": dict(blank_quantity),
        "number_of_calls": dict(blank_quantity),
        "calls_by_model_class": unavailable_calls_by_model_class(reason),
        "safe_provider_request_ids": dict(blank_identity),
        "provider_reported_usage": unavailable_provider_reported_usage(reason),
    }
    default_compute = {name: dict(blank_quantity) for name in COMPUTE_FIELDS}
    default_cost = {
        "provider_reported_charge": dict(blank_quantity),
        "token_based_estimated_charge": dict(blank_quantity),
        "price_snapshot_identity": dict(blank_identity),
        "price_snapshot_timestamp": dict(blank_identity),
        "local_compute_units": dict(blank_quantity),
        "audit_and_verification_overhead": dict(blank_quantity),
        "cost_per_accepted_patch": dict(blank_quantity),
        "cost_per_completed_task": dict(blank_quantity),
    }
    default_work = {
        **{name: dict(blank_operation) for name in WORK_FIELDS if name not in {
            "final_task_outcome",
            "patch_disposition",
            "validation_result",
        }},
        "final_task_outcome": final_task_outcome,
        "validation_result": default_validation,
        "patch_disposition": patch_disposition,
    }
    default_quality = {
        "accepted_patch_quality": dict(blank_quantity),
        "false_positives": dict(blank_operation),
        "false_negatives": dict(blank_operation),
        "selected_test_false_negatives": dict(blank_operation),
        "quality_adjusted_cost": dict(blank_quantity),
        "accepted_patch_rate": dict(blank_quantity),
    }
    default_terminal = {
        "final_task_outcome": final_task_outcome,
        "validation_result": dict(default_validation),
        "patch_disposition": patch_disposition,
        "time_to_terminal_outcome": dict(blank_quantity),
        "terminalized": True,
        "single_terminalization": True,
    }

    def merge(base: dict[str, Any], overlay: Mapping[str, Any] | None) -> dict[str, Any]:
        if overlay is None:
            return base
        merged = dict(base)
        merged.update(dict(overlay))
        return merged

    interfaces = list(
        interface_schema_identities
        or (TASK_EFFICIENCY_RECEIPT_SCHEMA,)
    )
    work_payload = merge(default_work, work)
    terminal_payload = merge(default_terminal, terminal)
    if terminal is None:
        terminal_payload["final_task_outcome"] = work_payload["final_task_outcome"]
        terminal_payload["validation_result"] = dict(work_payload["validation_result"])
        terminal_payload["patch_disposition"] = work_payload["patch_disposition"]
    payload: dict[str, Any] = {
        "schema": TASK_EFFICIENCY_RECEIPT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "task_id": _text(task_id, name="task_id"),
            "task_cid": task_cid,
            "objective_id": _text(objective_id, name="objective_id"),
            "objective_revision": _text(objective_revision, name="objective_revision"),
            "repository_commit": repository_commit,
            "repository_tree": repository_tree,
            "policy_identity": _text(policy_identity, name="policy_identity"),
            "context_pack_cid": dict(context_pack_cid or blank_identity),
            "interface_schema_identities": [
                _text(item, name="interface_schema_identity") for item in interfaces
            ],
            "provider_model": _provider_model(
                provider_id=provider_id,
                model_id=model_id,
                model_class=model_class,
                model_revision=model_revision,
            ),
        },
        "model_use": merge(default_model_use, model_use),
        "compute": merge(default_compute, compute),
        "cost": merge(default_cost, cost),
        "work": work_payload,
        "quality": merge(default_quality, quality),
        "evidence_state": {
            "population_kind": population_kind,
            "live": live,
            "simulated_as_live": False,
            "estimated_as_measured": False,
            "attempted_as_observed": False,
            "observed_as_verified": False,
            "unavailable_as_zero": False,
        },
        "terminal": terminal_payload,
    }
    payload["explicit_unavailable_fields"] = list(collect_unavailable_fields(payload))
    _bind_cid(payload, field="receipt_cid")
    return admit_task_efficiency_receipt(payload)


def build_paired_benchmark_manifest(
    *,
    objective_revision: str,
    repository_commit: str,
    repository_tree: str,
    policy_identity: str,
    task_inputs: str,
    acceptance_tests: str,
    available_providers_and_models: str,
    price_accounting: str,
    resource_limits: str,
    human_intervention_policy: str,
    maximum_retries: int = 3,
    price_snapshot_identity: Mapping[str, Any] | None = None,
    environment_identity: Mapping[str, Any] | None = None,
    direct_policy_identity: Mapping[str, Any] | None = None,
    sealed_policy_identity: Mapping[str, Any] | None = None,
    candidate_policy_identity: Mapping[str, Any] | None = None,
    hermetic_count: Mapping[str, Any] | None = None,
    historical_count: Mapping[str, Any] | None = None,
    live_count: Mapping[str, Any] | None = None,
    enrollment_deadline: Mapping[str, Any] | None = None,
    statistics: Mapping[str, Any] | None = None,
    hermetic_status: str = "unavailable",
    historical_status: str = "unavailable",
    live_status: str = "unavailable",
    reason: ReasonCode | str = ReasonCode.NOT_YET_MEASURED,
) -> PairedBenchmarkManifest:
    """Build a closed paired manifest; omitted statistics stay unavailable."""

    if GIT_OID_RE.fullmatch(repository_commit) is None:
        raise EfficiencyReceiptError("repository_commit must be a git object id")
    if GIT_OID_RE.fullmatch(repository_tree) is None:
        raise EfficiencyReceiptError("repository_tree must be a git object id")
    for name, value in (
        ("task_inputs", task_inputs),
        ("acceptance_tests", acceptance_tests),
        ("available_providers_and_models", available_providers_and_models),
        ("price_accounting", price_accounting),
        ("resource_limits", resource_limits),
        ("human_intervention_policy", human_intervention_policy),
    ):
        if CID_RE.fullmatch(value) is None:
            raise EfficiencyReceiptError(f"{name} must be a CIDv1")
    _int(maximum_retries, name="maximum_retries", maximum=MAX_RETRIES)
    blank = unavailable(reason)
    blank_identity = unavailable(reason)
    default_statistics = {name: dict(blank) for name in STATISTIC_FIELDS}
    payload: dict[str, Any] = {
        "schema": PAIRED_BENCHMARK_MANIFEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "program_id": PROGRAM_ID,
            "objective_id": OBJECTIVE_ID,
            "objective_revision": _text(objective_revision, name="objective_revision"),
            "repository_commit": repository_commit,
            "repository_tree": repository_tree,
            "policy_identity": _text(policy_identity, name="policy_identity"),
            "price_snapshot_identity": dict(price_snapshot_identity or blank_identity),
            "environment_identity": dict(environment_identity or blank_identity),
        },
        "arms": {
            "direct_minimal_orchestration_baseline": {
                "arm_id": "direct_minimal_orchestration_baseline",
                "constraints": list(DIRECT_ARM_CONSTRAINTS),
                "policy_identity": dict(direct_policy_identity or blank_identity),
                "shadow_before_mutation": False,
            },
            "sealed_current_supervisor_baseline": {
                "arm_id": "sealed_current_supervisor_baseline",
                "constraints": list(SEALED_CURRENT_ARM_CONSTRAINTS),
                "policy_identity": dict(sealed_policy_identity or blank_identity),
                "shadow_before_mutation": False,
            },
            "candidate_optimized_supervisor": {
                "arm_id": "candidate_optimized_supervisor",
                "constraints": list(CANDIDATE_ARM_CONSTRAINTS),
                "policy_identity": dict(candidate_policy_identity or blank_identity),
                "shadow_before_mutation": True,
            },
        },
        "equal_controls": {
            "repository_revision": repository_tree,
            "objective": OBJECTIVE_ID,
            "task_inputs": task_inputs,
            "acceptance_tests": acceptance_tests,
            "available_providers_and_models": available_providers_and_models,
            "price_accounting": price_accounting,
            "resource_limits": resource_limits,
            "maximum_retries": maximum_retries,
            "human_intervention_policy": human_intervention_policy,
        },
        "populations": {
            "hermetic_development": {
                "minimum": HERMETIC_MINIMUM,
                "count": dict(hermetic_count or blank),
                "status": hermetic_status,
            },
            "historical_exact_tree_replay": {
                "minimum": HISTORICAL_MINIMUM,
                "count": dict(historical_count or blank),
                "status": historical_status,
                "required_outcomes": list(HISTORICAL_OUTCOMES),
            },
            "new_live_shadow_canary": {
                "minimum": LIVE_MINIMUM,
                "count": dict(live_count or blank),
                "status": live_status,
                "enrollment_deadline": dict(enrollment_deadline or blank_identity),
                "enrollment_deadline_maximum_days": LIVE_ENROLLMENT_DEADLINE_MAXIMUM_DAYS,
            },
        },
        "statistics": {**default_statistics, **dict(statistics or {})},
        "promotion_without_paired_campaign": False,
        "hermetic_sufficient_for_production_promotion": False,
    }
    payload["explicit_unavailable_fields"] = list(collect_unavailable_fields(payload))
    _bind_cid(payload, field="manifest_cid")
    return admit_paired_benchmark_manifest(payload)


def round_trip_task_efficiency_receipt(
    payload: Mapping[str, Any] | str | bytes,
) -> bytes:
    return admit_task_efficiency_receipt(payload).round_trip()


def round_trip_paired_benchmark_manifest(
    payload: Mapping[str, Any] | str | bytes,
) -> bytes:
    return admit_paired_benchmark_manifest(payload).round_trip()


__all__ = (
    "CANDIDATE_ARM_CONSTRAINTS",
    "COMPUTE_FIELDS",
    "COST_FIELDS",
    "DIRECT_ARM_CONSTRAINTS",
    "EQUAL_CONTROL_FIELDS",
    "EVIDENCE_STATE_FIELDS",
    "EfficiencyReceiptError",
    "HISTORICAL_OUTCOMES",
    "IDENTITY_FIELDS",
    "MODEL_CLASSES",
    "MODEL_USE_FIELDS",
    "ModelClass",
    "PAIRED_ARMS",
    "PAIRED_BENCHMARK_MANIFEST_INTERFACE",
    "PAIRED_BENCHMARK_MANIFEST_SCHEMA",
    "PAIRED_BENCHMARK_MANIFEST_SCHEMA_PATH",
    "PATCH_DISPOSITIONS",
    "PatchDisposition",
    "PairedBenchmarkManifest",
    "PopulationKind",
    "QUALITY_FIELDS",
    "REASON_CODES",
    "ReasonCode",
    "SEALED_CURRENT_ARM_CONSTRAINTS",
    "STATISTIC_FIELDS",
    "TASK_EFFICIENCY_RECEIPT_INTERFACE",
    "TASK_EFFICIENCY_RECEIPT_SCHEMA",
    "TASK_EFFICIENCY_RECEIPT_SCHEMA_PATH",
    "TASK_OUTCOMES",
    "TERMINAL_FIELDS",
    "TRUTH_STATES",
    "TaskEfficiencyReceipt",
    "TaskOutcome",
    "TruthState",
    "UNITS",
    "WORK_FIELDS",
    "admit_paired_benchmark_manifest",
    "admit_task_efficiency_receipt",
    "attempted_identity",
    "attempted_operation",
    "build_paired_benchmark_manifest",
    "build_task_efficiency_receipt",
    "canonical_bytes",
    "canonical_json",
    "collect_unavailable_fields",
    "content_identity",
    "estimated_quantity",
    "iter_evidence_nodes",
    "load_paired_benchmark_manifest_schema",
    "load_task_efficiency_receipt_schema",
    "measured_quantity",
    "observed_identity",
    "observed_operation",
    "observed_request_ids",
    "round_trip_paired_benchmark_manifest",
    "round_trip_task_efficiency_receipt",
    "simulated",
    "unavailable",
    "unavailable_calls_by_model_class",
    "validate_against_schema",
    "verified_identity",
    "verified_operation",
    "verified_request_ids",
)
