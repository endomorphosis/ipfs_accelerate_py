"""Closed, canonical telemetry and paired-benchmark receipt contracts.

These records are measurement contracts, not authority to complete a task or
promote a candidate.  Every observation carries its own truth state so that a
missing value cannot silently turn into zero, an estimate cannot become a
measurement, and an observed action cannot become verified evidence.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Final


TASK_EFFICIENCY_RECEIPT_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/task-efficiency-receipt@1"
)
TASK_EFFICIENCY_RECEIPT_INTERFACE: Final[str] = "TaskEfficiencyReceipt@1"
PAIRED_BENCHMARK_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/paired-benchmark-manifest@1"
)
PAIRED_BENCHMARK_MANIFEST_INTERFACE: Final[str] = "PairedBenchmarkManifest@1"
MAX_CANONICAL_BYTES: Final[int] = 1_048_576
MAX_TEXT_BYTES: Final[int] = 4_096
MAX_COLLECTION_ITEMS: Final[int] = 1_024


class EfficiencyReceiptError(ValueError):
    """Raised when a closed efficiency contract is malformed or unsafe."""


class TruthState(str, Enum):
    """How strongly a value is supported; these states are never interchangeable."""

    MEASURED = "measured"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"
    ATTEMPTED = "attempted"
    OBSERVED = "observed"
    VERIFIED = "verified"
    SIMULATED = "simulated"


TRUTH_STATES: Final[frozenset[str]] = frozenset(item.value for item in TruthState)
MODEL_CLASSES: Final[frozenset[str]] = frozenset(
    {
        "deterministic",
        "local_small_model",
        "local_medium_model",
        "remote_standard_model",
        "remote_frontier_model",
        "human",
    }
)
PATCH_DISPOSITIONS: Final[frozenset[str]] = frozenset(
    {"accepted", "rejected", "quarantined", "reverted"}
)
BENCHMARK_ARMS: Final[tuple[str, ...]] = (
    "direct_minimal_orchestration_baseline",
    "sealed_current_supervisor_baseline",
    "candidate_optimized_supervisor",
)
_ARM_CONSTRAINTS: Final[dict[str, frozenset[str]]] = {
    "direct_minimal_orchestration_baseline": frozenset({
        "same_safe_isolation_and_acceptance_tests", "no_candidate_context_pack_optimization",
        "no_candidate_proof_or_test_reuse_optimization", "same_available_model_and_provider_set",
        "no_unsafe_direct_mutation",
    }),
    "sealed_current_supervisor_baseline": frozenset({"exact_pre_campaign_policy", "exact_pre_campaign_routing_behavior"}),
    "candidate_optimized_supervisor": frozenset({"campaign_changes_only", "shadow_before_mutation"}),
}
_EQUAL_CONTROLS: Final[frozenset[str]] = frozenset({
    "repository_revision", "objective", "task_inputs", "acceptance_tests",
    "available_providers_and_models", "price_accounting", "resource_limits",
    "maximum_retries", "human_intervention_policy",
})
_HISTORICAL_REQUIRED_OUTCOMES: Final[frozenset[str]] = frozenset({
    "successful", "failed", "retried", "rescued", "conflicted", "human_escalated",
})
TERMINAL_OUTCOMES: Final[frozenset[str]] = frozenset(
    {"completed", "failed", "cancelled", "blocked", "incomplete"}
)
VALIDATION_RESULTS: Final[frozenset[str]] = frozenset(
    {"passed", "failed", "unavailable", "not_run", "inconclusive"}
)

_IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "task_id",
    "task_cid",
    "objective_id",
    "objective_revision",
    "repository_commit",
    "repository_tree",
    "policy_identity",
    "context_pack_cid",
    "interface_schema_identities",
    "provider_model_exact_identity_and_revision",
)
_MODEL_VALUE_FIELDS: Final[tuple[str, ...]] = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens_where_reported",
    "reasoning_tokens_where_reported",
    "number_of_calls",
    "safe_provider_request_ids",
    "provider_reported_usage_where_available",
)
_COMPUTE_VALUE_FIELDS: Final[tuple[str, ...]] = (
    "cpu_seconds",
    "gpu_seconds",
    "peak_memory_bytes",
    "wall_clock_duration_ms",
    "test_execution_time_ms",
    "prover_execution_time_ms",
    "static_analysis_time_ms",
    "indexing_retrieval_time_ms",
    "bytes_read",
    "bytes_written",
    "process_count",
    "concurrency",
)
_COST_VALUE_FIELDS: Final[tuple[str, ...]] = (
    "provider_reported_charge_microunits",
    "token_based_estimated_charge_microunits",
    "price_snapshot_identity_and_timestamp",
    "local_compute_units",
    "audit_and_verification_overhead_microunits",
    "cost_per_accepted_patch_microunits",
    "cost_per_completed_task_microunits",
)
_WORK_VALUE_FIELDS: Final[tuple[str, ...]] = (
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
_QUALITY_VALUE_FIELDS: Final[tuple[str, ...]] = (
    "acceptance_tests_passed",
    "quality_gate_result",
    "accepted_patch_rate_basis",
)
_EVIDENCE_VALUE_FIELDS: Final[tuple[str, ...]] = (
    "attempted_operations",
    "observed_effects",
    "verified_evidence",
)


def schema_path(name: str) -> Path:
    """Return the bundled schema path for a declared receipt schema file."""

    allowed = {
        "task_efficiency_receipt.schema.json",
        "paired_benchmark_manifest.schema.json",
    }
    if name not in allowed:
        raise EfficiencyReceiptError("unknown efficiency receipt schema")
    return Path(__file__).with_name("schemas") / name


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON after rejecting non-canonical values."""

    _validate_json_value(value, "payload")
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:  # defensive for unusual Mapping values
        raise EfficiencyReceiptError("payload is not canonical JSON") from exc
    if len(encoded) > MAX_CANONICAL_BYTES:
        raise EfficiencyReceiptError("canonical receipt exceeds maximum size")
    return encoded


def _validate_json_value(value: Any, path: str, *, depth: int = 0) -> None:
    if depth > 24:
        raise EfficiencyReceiptError(f"{path} exceeds maximum nesting")
    if value is None or type(value) in {bool, int}:
        if type(value) is int and value.bit_length() > 256:
            raise EfficiencyReceiptError(f"{path} integer exceeds maximum size")
        return
    if isinstance(value, float):
        raise EfficiencyReceiptError(f"{path} must use integer units, not float")
    if isinstance(value, str):
        if not value or len(value.encode("utf-8")) > MAX_TEXT_BYTES:
            raise EfficiencyReceiptError(f"{path} has an invalid text bound")
        return
    if isinstance(value, Mapping):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise EfficiencyReceiptError(f"{path} has too many fields")
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise EfficiencyReceiptError(f"{path} has a non-string key")
            _validate_json_value(item, f"{path}.{key}", depth=depth + 1)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise EfficiencyReceiptError(f"{path} has too many items")
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]", depth=depth + 1)
        return
    raise EfficiencyReceiptError(f"{path} is not JSON compatible")


def _closed(value: Any, fields: Sequence[str], path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EfficiencyReceiptError(f"{path} must be an object")
    actual, expected = set(value), set(fields)
    missing, unknown = expected - actual, actual - expected
    if missing:
        raise EfficiencyReceiptError(f"{path} is missing required fields: {sorted(missing)}")
    if unknown:
        raise EfficiencyReceiptError(f"{path} has unknown fields: {sorted(unknown)}")
    return dict(value)


def _text(value: Any, path: str, *, choices: frozenset[str] | None = None) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > MAX_TEXT_BYTES:
        raise EfficiencyReceiptError(f"{path} must be non-empty bounded text")
    if choices is not None and value not in choices:
        raise EfficiencyReceiptError(f"{path} must be one of {sorted(choices)}")
    return value


def _nonnegative_int(value: Any, path: str) -> int:
    if type(value) is not int or value < 0 or value > 2**63 - 1:
        raise EfficiencyReceiptError(f"{path} must be a bounded non-negative integer")
    return value


def _strings(value: Any, path: str, *, unique: bool = True) -> list[str]:
    if not isinstance(value, list) or len(value) > MAX_COLLECTION_ITEMS:
        raise EfficiencyReceiptError(f"{path} must be a bounded array")
    result = [_text(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if unique and len(set(result)) != len(result):
        raise EfficiencyReceiptError(f"{path} must not contain duplicates")
    return result


def _evidence(value: Any, path: str) -> dict[str, Any]:
    item = _closed(value, ("state", "value", "source", "reference_ids"), path)
    state = _text(item["state"], f"{path}.state", choices=TRUTH_STATES)
    source = _text(item["source"], f"{path}.source")
    references = _strings(item["reference_ids"], f"{path}.reference_ids")
    _validate_json_value(item["value"], f"{path}.value")
    if state == TruthState.UNAVAILABLE.value:
        if item["value"] is not None or source != TruthState.UNAVAILABLE.value or references:
            raise EfficiencyReceiptError(
                f"{path} unavailable evidence must have null value, unavailable source, and no references"
            )
    elif item["value"] is None:
        raise EfficiencyReceiptError(f"{path} {state} evidence requires a value")
    if state == TruthState.VERIFIED.value and not references:
        raise EfficiencyReceiptError(f"{path} verified evidence requires reference_ids")
    # Parsing our own canonical encoding normalizes tuples and mapping subclasses
    # into the exact JSON representation retained by the durable receipt.
    normalized_value = json.loads(canonical_json_bytes(item["value"]))
    return {"state": state, "value": normalized_value, "source": source, "reference_ids": references}


def _evidence_block(value: Any, fields: Sequence[str], path: str) -> dict[str, dict[str, Any]]:
    block = _closed(value, fields, path)
    return {name: _evidence(block[name], f"{path}.{name}") for name in fields}


def _validate_value_kinds(block: Mapping[str, Mapping[str, Any]], fields: Sequence[str], path: str) -> None:
    """Keep units numeric where they are contractually quantities, never floats."""

    textual_or_list = {
        "safe_provider_request_ids", "provider_reported_usage_where_available",
        "price_snapshot_identity_and_timestamp", "tests_selected", "tests_executed",
        "full_suite_tests", "type_static_schema_checks", "proof_obligations_selected",
        "proof_obligations_executed", "proof_receipts_reused", "manual_recovery",
        "human_interventions", "quality_gate_result", "accepted_patch_rate_basis",
        "attempted_operations", "observed_effects", "verified_evidence",
    }
    for name in fields:
        evidence = block[name]
        item = evidence["value"]
        if evidence["state"] == TruthState.UNAVAILABLE.value:
            continue
        if name in textual_or_list:
            if not isinstance(item, (str, list, Mapping)):
                raise EfficiencyReceiptError(f"{path}.{name}.value must be text, list, or object")
        elif type(item) is not int or item < 0:
            raise EfficiencyReceiptError(f"{path}.{name}.value must be a non-negative integer")


def _identity(value: Any) -> dict[str, Any]:
    identity = _closed(value, _IDENTITY_FIELDS, "identity")
    result: dict[str, Any] = {}
    for name in _IDENTITY_FIELDS[:-2]:
        result[name] = _text(identity[name], f"identity.{name}")
    result["interface_schema_identities"] = _strings(
        identity["interface_schema_identities"], "identity.interface_schema_identities"
    )
    providers = identity["provider_model_exact_identity_and_revision"]
    if not isinstance(providers, list) or not providers or len(providers) > MAX_COLLECTION_ITEMS:
        raise EfficiencyReceiptError("identity.provider_model_exact_identity_and_revision must be non-empty")
    normalized_providers: list[dict[str, str]] = []
    for index, provider in enumerate(providers):
        item = _closed(provider, ("provider", "model", "revision"), f"identity.provider_models[{index}]")
        normalized_providers.append(
            {key: _text(item[key], f"identity.provider_models[{index}].{key}") for key in item}
        )
    if len({(item["provider"], item["model"], item["revision"]) for item in normalized_providers}) != len(normalized_providers):
        raise EfficiencyReceiptError("identity.provider_model_exact_identity_and_revision has duplicates")
    result["provider_model_exact_identity_and_revision"] = sorted(
        normalized_providers, key=lambda item: (item["provider"], item["model"], item["revision"])
    )
    return result


def validate_task_efficiency_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a task receipt into its canonical closed form."""

    root = _closed(
        payload,
        ("schema", "interface", "identity", "model_use", "compute", "cost", "work", "quality", "evidence", "terminal"),
        "task_efficiency_receipt",
    )
    if root["schema"] != TASK_EFFICIENCY_RECEIPT_SCHEMA or root["interface"] != TASK_EFFICIENCY_RECEIPT_INTERFACE:
        raise EfficiencyReceiptError("task receipt schema or interface is not supported")
    model = _evidence_block(root["model_use"], (*_MODEL_VALUE_FIELDS, "calls_by_model_class", "explicit_unavailable_fields"), "model_use")
    calls = model["calls_by_model_class"]
    if calls["state"] != TruthState.UNAVAILABLE.value:
        raw_calls = calls["value"]
        if not isinstance(raw_calls, list) or not raw_calls:
            raise EfficiencyReceiptError("model_use.calls_by_model_class.value must be a non-empty list")
        normalized_calls: list[dict[str, Any]] = []
        for index, item in enumerate(raw_calls):
            entry = _closed(item, ("model_class", "count"), f"model_use.calls_by_model_class.value[{index}]")
            normalized_calls.append({
                "model_class": _text(entry["model_class"], "model class", choices=MODEL_CLASSES),
                "count": _nonnegative_int(entry["count"], "model class count"),
            })
        if len({item["model_class"] for item in normalized_calls}) != len(normalized_calls):
            raise EfficiencyReceiptError("model classes must be unique")
        calls["value"] = sorted(normalized_calls, key=lambda item: item["model_class"])
    unavailable = model["explicit_unavailable_fields"]
    if unavailable["state"] != TruthState.UNAVAILABLE.value:
        if not isinstance(unavailable["value"], list):
            raise EfficiencyReceiptError("explicit_unavailable_fields.value must be a list")
        names = _strings(unavailable["value"], "explicit_unavailable_fields.value")
        permitted = set(_MODEL_VALUE_FIELDS) | set(_COMPUTE_VALUE_FIELDS) | set(_COST_VALUE_FIELDS) | set(_WORK_VALUE_FIELDS) | set(_QUALITY_VALUE_FIELDS) | set(_EVIDENCE_VALUE_FIELDS)
        if not set(names) <= permitted:
            raise EfficiencyReceiptError("explicit_unavailable_fields names an unknown telemetry field")
        model["explicit_unavailable_fields"]["value"] = sorted(names)
    compute = _evidence_block(root["compute"], _COMPUTE_VALUE_FIELDS, "compute")
    cost = _evidence_block(root["cost"], _COST_VALUE_FIELDS, "cost")
    work = _evidence_block(root["work"], (*_WORK_VALUE_FIELDS, "final_task_outcome", "validation_result", "patch_disposition"), "work")
    quality = _evidence_block(root["quality"], _QUALITY_VALUE_FIELDS, "quality")
    evidence = _evidence_block(root["evidence"], _EVIDENCE_VALUE_FIELDS, "evidence")
    for name, required_state in (
        ("attempted_operations", TruthState.ATTEMPTED.value),
        ("observed_effects", TruthState.OBSERVED.value),
        ("verified_evidence", TruthState.VERIFIED.value),
    ):
        state = evidence[name]["state"]
        if state not in {required_state, TruthState.UNAVAILABLE.value}:
            raise EfficiencyReceiptError(
                f"evidence.{name} must be {required_state} or unavailable, never {state}"
            )
    _validate_value_kinds(model, _MODEL_VALUE_FIELDS, "model_use")
    _validate_value_kinds(compute, _COMPUTE_VALUE_FIELDS, "compute")
    _validate_value_kinds(cost, _COST_VALUE_FIELDS, "cost")
    _validate_value_kinds(work, _WORK_VALUE_FIELDS, "work")
    _validate_value_kinds(quality, _QUALITY_VALUE_FIELDS, "quality")
    _validate_value_kinds(evidence, _EVIDENCE_VALUE_FIELDS, "evidence")
    for name, choices in (("final_task_outcome", TERMINAL_OUTCOMES), ("validation_result", VALIDATION_RESULTS), ("patch_disposition", PATCH_DISPOSITIONS)):
        item = work[name]
        if item["state"] != TruthState.UNAVAILABLE.value:
            item["value"] = _text(item["value"], f"work.{name}.value", choices=choices)
    terminal = _closed(root["terminal"], ("is_terminal", "outcome", "evidence_state", "terminal_reason"), "terminal")
    if type(terminal["is_terminal"]) is not bool:
        raise EfficiencyReceiptError("terminal.is_terminal must be boolean")
    terminal_state = _text(terminal["evidence_state"], "terminal.evidence_state", choices=TRUTH_STATES)
    outcome = _text(terminal["outcome"], "terminal.outcome", choices=TERMINAL_OUTCOMES)
    reason = _text(terminal["terminal_reason"], "terminal.terminal_reason")
    if terminal["is_terminal"] is False and outcome != "incomplete":
        raise EfficiencyReceiptError("non-terminal receipts must have incomplete outcome")
    if terminal_state == TruthState.UNAVAILABLE.value:
        raise EfficiencyReceiptError("terminal evidence state cannot be unavailable")
    normalized = {
        "schema": TASK_EFFICIENCY_RECEIPT_SCHEMA,
        "interface": TASK_EFFICIENCY_RECEIPT_INTERFACE,
        "identity": _identity(root["identity"]),
        "model_use": model,
        "compute": compute,
        "cost": cost,
        "work": work,
        "quality": quality,
        "evidence": evidence,
        "terminal": {"is_terminal": terminal["is_terminal"], "outcome": outcome, "evidence_state": terminal_state, "terminal_reason": reason},
    }
    _validate_unavailable_declarations(normalized)
    canonical_json_bytes(normalized)
    return normalized


def _validate_unavailable_declarations(payload: Mapping[str, Any]) -> None:
    declared = payload["model_use"]["explicit_unavailable_fields"]
    if declared["state"] == TruthState.UNAVAILABLE.value:
        return
    values = set(declared["value"])
    locations = {
        **{name: payload["model_use"][name] for name in _MODEL_VALUE_FIELDS},
        **{name: payload["compute"][name] for name in _COMPUTE_VALUE_FIELDS},
        **{name: payload["cost"][name] for name in _COST_VALUE_FIELDS},
        **{name: payload["work"][name] for name in _WORK_VALUE_FIELDS},
        **{name: payload["quality"][name] for name in _QUALITY_VALUE_FIELDS},
        **{name: payload["evidence"][name] for name in _EVIDENCE_VALUE_FIELDS},
    }
    actual = {name for name, item in locations.items() if item["state"] == TruthState.UNAVAILABLE.value}
    if values != actual:
        raise EfficiencyReceiptError("explicit_unavailable_fields must exactly name unavailable telemetry")


def validate_paired_benchmark_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a closed manifest for the three paired arms."""

    root = _closed(payload, ("schema", "interface", "manifest_id", "baseline", "arms", "equal_controls", "populations", "historical_required_outcomes", "statistics", "promotion_without_paired_campaign"), "paired_benchmark_manifest")
    if root["schema"] != PAIRED_BENCHMARK_MANIFEST_SCHEMA or root["interface"] != PAIRED_BENCHMARK_MANIFEST_INTERFACE:
        raise EfficiencyReceiptError("paired benchmark schema or interface is not supported")
    _text(root["manifest_id"], "manifest_id")
    baseline = _closed(root["baseline"], ("repository_revision", "repository_tree", "objective_id", "objective_revision", "task_inputs_cid", "acceptance_tests_cid", "available_provider_models", "price_accounting", "resource_limits", "maximum_retries", "human_intervention_policy"), "baseline")
    normalized_baseline: dict[str, Any] = {}
    for name in ("repository_revision", "repository_tree", "objective_id", "objective_revision", "task_inputs_cid", "acceptance_tests_cid", "price_accounting", "human_intervention_policy"):
        normalized_baseline[name] = _text(baseline[name], f"baseline.{name}")
    normalized_baseline["available_provider_models"] = _strings(baseline["available_provider_models"], "baseline.available_provider_models")
    normalized_baseline["resource_limits"] = _evidence(baseline["resource_limits"], "baseline.resource_limits")
    normalized_baseline["maximum_retries"] = _nonnegative_int(baseline["maximum_retries"], "baseline.maximum_retries")
    arms = root["arms"]
    if not isinstance(arms, list) or len(arms) != len(BENCHMARK_ARMS):
        raise EfficiencyReceiptError("arms must contain exactly the three required benchmark arms")
    normalized_arms: list[dict[str, Any]] = []
    for index, arm in enumerate(arms):
        item = _closed(arm, ("arm", "constraints", "receipt_ids", "evidence_state"), f"arms[{index}]")
        constraints = _strings(item["constraints"], f"arms[{index}].constraints")
        arm_name = _text(item["arm"], f"arms[{index}].arm", choices=frozenset(BENCHMARK_ARMS))
        if set(constraints) != _ARM_CONSTRAINTS[arm_name]:
            raise EfficiencyReceiptError(f"arms[{index}].constraints must bind the required arm controls")
        normalized_arms.append({
            "arm": arm_name,
            "constraints": sorted(constraints),
            "receipt_ids": _strings(item["receipt_ids"], f"arms[{index}].receipt_ids"),
            "evidence_state": _text(item["evidence_state"], f"arms[{index}].evidence_state", choices=TRUTH_STATES),
        })
    if {item["arm"] for item in normalized_arms} != set(BENCHMARK_ARMS):
        raise EfficiencyReceiptError("arms must contain each required arm exactly once")
    equal_controls = _strings(root["equal_controls"], "equal_controls")
    if set(equal_controls) != _EQUAL_CONTROLS:
        raise EfficiencyReceiptError("equal_controls must bind every paired control")
    populations = _closed(root["populations"], ("hermetic_development_count", "historical_exact_tree_replay_count", "new_live_shadow_canary_count", "live_enrollment_deadline_days"), "populations")
    normalized_populations = {name: _nonnegative_int(populations[name], f"populations.{name}") for name in populations}
    if normalized_populations["live_enrollment_deadline_days"] > 30:
        raise EfficiencyReceiptError("live enrollment deadline cannot exceed 30 days")
    historical_outcomes = _strings(root["historical_required_outcomes"], "historical_required_outcomes")
    if set(historical_outcomes) != _HISTORICAL_REQUIRED_OUTCOMES:
        raise EfficiencyReceiptError("historical_required_outcomes must bind each required outcome")
    statistics = _strings(root["statistics"], "statistics")
    if type(root["promotion_without_paired_campaign"]) is not bool or root["promotion_without_paired_campaign"]:
        raise EfficiencyReceiptError("promotion_without_paired_campaign must be false")
    normalized = {
        "schema": PAIRED_BENCHMARK_MANIFEST_SCHEMA,
        "interface": PAIRED_BENCHMARK_MANIFEST_INTERFACE,
        "manifest_id": root["manifest_id"],
        "baseline": normalized_baseline,
        "arms": sorted(normalized_arms, key=lambda item: BENCHMARK_ARMS.index(item["arm"])),
        "equal_controls": sorted(equal_controls),
        "populations": normalized_populations,
        "historical_required_outcomes": sorted(historical_outcomes),
        "statistics": sorted(statistics),
        "promotion_without_paired_campaign": False,
    }
    canonical_json_bytes(normalized)
    return normalized


class _CanonicalContract:
    _validator: Any

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._payload = self._validator(payload)

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._payload)

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self._payload)

    def canonical_json(self) -> str:
        return self.canonical_bytes().decode("utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]):
        return cls(payload)

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes | bytearray | memoryview):
        if not isinstance(encoded, (bytes, bytearray, memoryview)):
            raise EfficiencyReceiptError("canonical receipt must be bytes")
        raw = bytes(encoded)
        if not raw or len(raw) > MAX_CANONICAL_BYTES:
            raise EfficiencyReceiptError("canonical receipt has an invalid size")
        try:
            def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
                result: dict[str, Any] = {}
                for key, value in pairs:
                    if key in result:
                        raise EfficiencyReceiptError("canonical receipt has duplicate object keys")
                    result[key] = value
                return result

            payload = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EfficiencyReceiptError("canonical receipt is not UTF-8 JSON") from exc
        contract = cls(payload)
        if contract.canonical_bytes() != raw:
            raise EfficiencyReceiptError("receipt bytes are not canonical")
        return contract

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self._payload == other._payload  # type: ignore[attr-defined]


class TaskEfficiencyReceipt(_CanonicalContract):
    """Canonical task telemetry receipt; this object has no promotion authority."""

    _validator = staticmethod(validate_task_efficiency_receipt)


class PairedBenchmarkManifest(_CanonicalContract):
    """Canonical three-arm benchmark manifest; it records no promotion decision."""

    _validator = staticmethod(validate_paired_benchmark_manifest)
