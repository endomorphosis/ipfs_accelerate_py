"""Closed ASEH route receipts and highest-level machine output.

The routing-decision wire contract lives beside this module as a JSON Schema
document.  Admission is fail-closed: unknown fields, floats, stage mismatch,
unbound usage, absent decisive evidence, division by missing data, and claimed
question closure without evidence are rejected.

Requested rates are always produced.  A missing or zero denominator stays
``unavailable`` and is never imputed as numeric zero.  Receipts report routing
observations only; they cannot validate patches or promote policy.

``emit_decision`` remains the LGSWF typed envelope helper.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Final

from . import efficiency_receipts as _er


def emit_decision(record):
    return MappingProxyType(
        {
            "schema": "lgswf/decision-receipt@1",
            "decision": record.get("decision"),
            "metrics": record.get("metrics") or {},
        }
    )


ROUTE_RECEIPT_CONTRACT_VERSION: Final[int] = 1
SCHEMA_VERSION: Final[int] = ROUTE_RECEIPT_CONTRACT_VERSION
MILLIONTHS: Final[int] = 1_000_000

ROUTING_DECISION_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/aseh-routing-decision@1"
)
ROUTE_RECEIPT_SCHEMA: Final[str] = ROUTING_DECISION_SCHEMA
ROUTING_DECISION_INTERFACE: Final[str] = "RoutingDecision@1"
ROUTE_RECEIPT_INTERFACE: Final[str] = "RouteReceipt@1"

SCHEMA_DIR: Final[Path] = Path(__file__).resolve().parent / "schemas"
ROUTING_DECISION_SCHEMA_PATH: Final[Path] = SCHEMA_DIR / "routing_decision.schema.json"

PROGRAM_ID: Final[str] = "agent-supervisor-efficiency-and-state-hardening-v1"
OBJECTIVE_ID: Final[str] = "ASEH-G030"
RATE_SENSOR_ID: Final[str] = "aseh-024-route-receipt-rates"

ORDERED_STAGES: Final[tuple[str, ...]] = (
    "exact_current_authoritative_cached_receipt",
    "ast_symbol_dependency_and_impact_analysis",
    "schema_type_static_lint_and_contract_checks",
    "selected_tests",
    "incremental_smt_or_theorem_prover",
    "local_small_specialist_model",
    "local_or_remote_medium_model",
    "remote_strong_or_frontier_model",
    "human_decision",
)
DETERMINISTIC_STAGES: Final[frozenset[str]] = frozenset(ORDERED_STAGES[:5])
MODEL_STAGES: Final[frozenset[str]] = frozenset(ORDERED_STAGES[5:8])
HUMAN_STAGE: Final[str] = "human_decision"
SMALL_STAGE: Final[str] = "local_small_specialist_model"
MEDIUM_STAGE: Final[str] = "local_or_remote_medium_model"
FRONTIER_STAGE: Final[str] = "remote_strong_or_frontier_model"

STAGE_RUN_REASONS: Final[tuple[str, ...]] = (
    "eligible_deterministic_evidence",
    "unresolved_deterministic_evidence",
    "smallest_adequate_executor",
    "human_decision_required",
)
RESOLVING_RUN_REASONS: Final[frozenset[str]] = frozenset(
    {
        "eligible_deterministic_evidence",
        "smallest_adequate_executor",
        "human_decision_required",
    }
)
STAGE_SKIP_REASONS: Final[tuple[str, ...]] = (
    "resolved_by_prior_stage",
    "evidence_unavailable",
    "evidence_ineligible",
    "model_not_required",
    "not_smallest_adequate_executor",
    "required_tier_unavailable",
    "human_gate_preempts_model",
)
ESCALATION_REASONS: Final[tuple[str, ...]] = (
    "not_applicable",
    "unresolved_deterministic_evidence",
    "smallest_adequate_executor",
    "human_decision_required",
    "required_tier_unavailable",
)
EXECUTOR_CLASSES: Final[tuple[str, ...]] = (
    "deterministic",
    "local_small_model",
    "local_medium_model",
    "remote_frontier_model",
    "human",
)
MODEL_ROUTES: Final[tuple[str, ...]] = (
    "deterministic_only",
    "small_local_model",
    "medium_model",
    "frontier_model",
    "human_review_required",
)
EVIDENCE_KINDS: Final[tuple[str, ...]] = (
    "cached_receipt",
    "impact_analysis",
    "static_contract",
    "selected_tests",
    "incremental_proof",
    "model_answer",
    "human_decision",
)
STAGE_EVIDENCE_KIND: Final[dict[str, str]] = {
    "exact_current_authoritative_cached_receipt": "cached_receipt",
    "ast_symbol_dependency_and_impact_analysis": "impact_analysis",
    "schema_type_static_lint_and_contract_checks": "static_contract",
    "selected_tests": "selected_tests",
    "incremental_smt_or_theorem_prover": "incremental_proof",
    "local_small_specialist_model": "model_answer",
    "local_or_remote_medium_model": "model_answer",
    "remote_strong_or_frontier_model": "model_answer",
    "human_decision": "human_decision",
}
STAGE_TO_EXECUTOR_CLASS: Final[dict[str, str]] = {
    **{stage: "deterministic" for stage in DETERMINISTIC_STAGES},
    SMALL_STAGE: "local_small_model",
    MEDIUM_STAGE: "local_medium_model",
    FRONTIER_STAGE: "remote_frontier_model",
    HUMAN_STAGE: "human",
}
STAGE_TO_ROUTE: Final[dict[str, str]] = {
    **{stage: "deterministic_only" for stage in DETERMINISTIC_STAGES},
    SMALL_STAGE: "small_local_model",
    MEDIUM_STAGE: "medium_model",
    FRONTIER_STAGE: "frontier_model",
    HUMAN_STAGE: "human_review_required",
}
ROUTE_RECEIPT_FIELDS: Final[tuple[str, ...]] = (
    "stages_run",
    "stages_skipped_and_reason",
    "decisive_evidence",
    "escalation_reason",
    "selected_executor",
    "actual_resource_use",
    "final_outcome",
)
REQUESTED_RATES: Final[tuple[str, ...]] = (
    "deterministic_resolution_rate",
    "small_model_resolution_rate",
    "frontier_model_rate",
    "human_rate",
    "escalation_precision",
    "unnecessary_escalation_rate",
    "unresolved_question_closure_rate",
)
IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "task_id",
    "task_cid",
    "objective_id",
    "objective_revision",
    "repository_commit",
    "repository_tree",
    "policy_identity",
)
RESOURCE_QUANTITY_FIELDS: Final[tuple[str, ...]] = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
    "number_of_calls",
    "provider_reported_charge",
    "cpu_seconds",
    "gpu_seconds",
    "peak_memory",
    "wall_clock_duration",
    "audit_and_verification_overhead",
)
PROVIDER_RESOURCE_FIELDS: Final[tuple[str, ...]] = (
    "input_tokens",
    "output_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
    "number_of_calls",
    "provider_reported_charge",
)
WORK_RESOURCE_FIELDS: Final[tuple[str, ...]] = (
    "cpu_seconds",
    "gpu_seconds",
    "peak_memory",
    "wall_clock_duration",
    "audit_and_verification_overhead",
)
RESOURCE_IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "provider_usage_record_id",
    "work_telemetry_record_id",
    "span_id",
)
AUTHORITY: Final[dict[str, bool]] = {
    "observation_only": True,
    "validates_patches": False,
    "promotes_policy": False,
    "schema_attests_measurement": False,
}
QUESTION_CLOSURE_STATUSES: Final[tuple[str, ...]] = (
    "not_applicable",
    "open",
    "closed",
)
TASK_OUTCOMES: Final[tuple[str, ...]] = _er.TASK_OUTCOMES

_SCHEMA_CACHE: dict[str, dict[str, Any]] = {}


class DecisionReceiptError(ValueError):
    """Closed routing-decision or route-receipt contract violation."""


class ExecutorClass(str, Enum):
    DETERMINISTIC = "deterministic"
    LOCAL_SMALL_MODEL = "local_small_model"
    LOCAL_MEDIUM_MODEL = "local_medium_model"
    REMOTE_FRONTIER_MODEL = "remote_frontier_model"
    HUMAN = "human"


class QuestionClosureStatus(str, Enum):
    NOT_APPLICABLE = "not_applicable"
    OPEN = "open"
    CLOSED = "closed"


def _raise(message: str) -> None:
    raise DecisionReceiptError(message)


def _wrap_efficiency(exc: _er.EfficiencyReceiptError) -> DecisionReceiptError:
    return DecisionReceiptError(str(exc))


def canonical_bytes(value: Any) -> bytes:
    try:
        return _er.canonical_bytes(value)
    except _er.EfficiencyReceiptError as exc:
        raise _wrap_efficiency(exc) from exc


def canonical_json(value: Any) -> str:
    return canonical_bytes(value).decode("utf-8")


def content_identity(value: Any) -> str:
    try:
        return _er.content_identity(value)
    except _er.EfficiencyReceiptError as exc:
        raise _wrap_efficiency(exc) from exc


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _raise(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        _raise(f"{name} keys must be strings")
    return {str(key): item for key, item in value.items()}


def _decode_json(payload: Any, *, name: str) -> dict[str, Any]:
    if isinstance(payload, (bytes, bytearray)):
        try:
            payload = bytes(payload).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DecisionReceiptError(f"{name} is not UTF-8") from exc
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise DecisionReceiptError(f"{name} JSON is malformed") from exc
    return _mapping(payload, name=name)


def _text(value: Any, *, name: str) -> str:
    try:
        return _er._text(value, name=name)
    except _er.EfficiencyReceiptError as exc:
        raise _wrap_efficiency(exc) from exc


def _int(
    value: Any, *, name: str, minimum: int = 0, maximum: int = _er.MAX_INTEGER
) -> int:
    try:
        return _er._int(value, name=name, minimum=minimum, maximum=maximum)
    except _er.EfficiencyReceiptError as exc:
        raise _wrap_efficiency(exc) from exc


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DecisionReceiptError(f"schema document is unreadable: {path.name}") from exc
    if not isinstance(payload, dict):
        _raise(f"{path.name} must contain a JSON object")
    return payload


def _assert_schema_document_is_closed(schema: Mapping[str, Any], *, name: str) -> None:
    def walk(node: Any, *, path: str) -> None:
        if isinstance(node, Mapping):
            if node.get("type") == "object" or "properties" in node:
                if node.get("additionalProperties") is not False:
                    _raise(f"{name} {path} is not a closed object schema")
            for key, child in node.items():
                walk(child, path=f"{path}.{key}")
            return
        if isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, path=f"{path}[{index}]")

    walk(schema, path="$")


def load_routing_decision_schema() -> dict[str, Any]:
    cached = _SCHEMA_CACHE.get("routing_decision")
    if cached is not None:
        return cached
    schema = _load_json_object(ROUTING_DECISION_SCHEMA_PATH)
    _assert_schema_document_is_closed(schema, name="routing decision schema")
    if schema.get("$id") != ROUTING_DECISION_SCHEMA:
        _raise("routing decision schema $id mismatch")
    _SCHEMA_CACHE["routing_decision"] = schema
    return schema


def validate_against_schema(
    instance: Any,
    schema: Mapping[str, Any],
    *,
    defs: Mapping[str, Any] | None = None,
    path: str = "$",
) -> None:
    try:
        _er.validate_against_schema(instance, schema, defs=defs, path=path)
    except _er.EfficiencyReceiptError as exc:
        raise _wrap_efficiency(exc) from exc


def unavailable(reason: str | _er.ReasonCode = _er.ReasonCode.NOT_YET_MEASURED) -> dict[str, str]:
    try:
        return _er.unavailable(reason)
    except _er.EfficiencyReceiptError as exc:
        raise _wrap_efficiency(exc) from exc


def measured_quantity(value: int, *, unit: str, sensor_id: str) -> dict[str, Any]:
    try:
        return _er.measured_quantity(value, unit=unit, sensor_id=sensor_id)
    except _er.EfficiencyReceiptError as exc:
        raise _wrap_efficiency(exc) from exc


def observed_identity(value: str) -> dict[str, str]:
    try:
        return _er.observed_identity(value)
    except _er.EfficiencyReceiptError as exc:
        raise _wrap_efficiency(exc) from exc


def collect_unavailable_fields(value: Any) -> tuple[str, ...]:
    return _er.collect_unavailable_fields(value)


def iter_evidence_nodes(value: Any, *, path: str = ""):
    return _er.iter_evidence_nodes(value, path=path)


def _bind_cid(payload: dict[str, Any], *, field: str) -> str:
    body = {key: item for key, item in payload.items() if key != field}
    cid = content_identity(body)
    claimed = payload.get(field)
    if claimed not in (None, "", cid):
        _raise(f"{field} does not match canonical content identity")
    payload[field] = cid
    return cid


def _canonical_copy(value: Any) -> Any:
    encoded = canonical_bytes(value)
    return json.loads(encoded.decode("utf-8"))


def _same_canonical(left: Any, right: Any) -> bool:
    return canonical_bytes(left) == canonical_bytes(right)


def measured_rate(
    numerator: int,
    denominator: int,
    *,
    sensor_id: str = RATE_SENSOR_ID,
) -> dict[str, Any]:
    """Encode a measured ratio.  A missing denominator is never imputed."""

    num = _int(numerator, name="numerator")
    den = _int(denominator, name="denominator", minimum=0)
    if den < 1:
        _raise("division by missing data")
    if num > den:
        _raise("rate numerator exceeds denominator")
    return {
        "truth_state": "measured",
        "value": (num * MILLIONTHS) // den,
        "unit": "ratio_millionths",
        "numerator": num,
        "denominator": den,
        "sensor_id": _text(sensor_id, name="sensor_id"),
    }


def unavailable_rate(
    reason: str | _er.ReasonCode = _er.ReasonCode.NOT_APPLICABLE,
) -> dict[str, str]:
    return unavailable(reason)


def ratio_millionths(numerator: int, denominator: int | None) -> dict[str, Any]:
    """Return a rate observation without imputing an unavailable denominator."""

    if denominator is None:
        return unavailable_rate("not_yet_measured")
    if type(denominator) is not int:
        _raise("denominator must be an integer")
    if denominator < 1:
        return unavailable_rate("not_applicable")
    return measured_rate(numerator, denominator)


def executor_class_for_stage(stage: str) -> str:
    try:
        return STAGE_TO_EXECUTOR_CLASS[stage]
    except KeyError as exc:
        raise DecisionReceiptError(f"stage mismatch: unsupported stage {stage!r}") from exc


def route_for_stage(stage: str) -> str:
    try:
        return STAGE_TO_ROUTE[stage]
    except KeyError as exc:
        raise DecisionReceiptError(f"stage mismatch: unsupported stage {stage!r}") from exc


def selected_executor_for_stage(stage: str) -> dict[str, str]:
    return {
        "stage": stage,
        "executor_class": executor_class_for_stage(stage),
        "route": route_for_stage(stage),
    }


def observed_unnecessary_escalation(
    unnecessary: bool, *, observer_id: str = "aseh-024-route-observer"
) -> dict[str, Any]:
    if type(unnecessary) is not bool:
        _raise("unnecessary must be a boolean")
    return {
        "truth_state": "observed",
        "unnecessary": unnecessary,
        "observer_id": _text(observer_id, name="observer_id"),
    }


def _attr(record: Any, name: str, default: Any = None) -> Any:
    if record is None:
        return default
    if isinstance(record, Mapping):
        return record.get(name, default)
    value = getattr(record, name, default)
    return value() if callable(value) else value


def _require_admitted_binding(
    record: Any,
    *,
    kind: str,
    expected_task_id: str,
) -> Any:
    if record is None:
        return None
    admitted = _attr(record, "admitted")
    task_id = _attr(record, "task_id")
    span_id = _attr(record, "span_id")
    if admitted is not True or not task_id or not span_id:
        _raise(f"unbound usage: {kind} telemetry is not admitted to a causal task span")
    if task_id != expected_task_id:
        _raise(f"unbound usage: {kind} telemetry task_id does not match the route")
    return record


def _quantity_from_sample(sample: Any) -> dict[str, Any]:
    method = getattr(sample, "to_quantity_envelope", None)
    if callable(method):
        envelope = method()
        return _mapping(envelope, name="quantity envelope")
    if isinstance(sample, Mapping) and "truth_state" in sample:
        return dict(sample)
    _raise("unbound usage: telemetry sample is not a quantity envelope")
    raise AssertionError("unreachable")


def _project_provider_resources(record: Any) -> dict[str, Any]:
    method = getattr(record, "to_model_use_fields", None)
    if callable(method):
        projected = _mapping(method(), name="provider usage projection")
        charge = _attr(record, "provider_reported_charge")
        payload = {name: dict(projected[name]) for name in PROVIDER_RESOURCE_FIELDS if name != "provider_reported_charge"}
        if charge is not None:
            payload["provider_reported_charge"] = _quantity_from_sample(charge)
        elif "provider_reported_charge" in projected:
            payload["provider_reported_charge"] = dict(projected["provider_reported_charge"])
        else:
            payload["provider_reported_charge"] = unavailable("provider_omitted")
        return payload
    if not isinstance(record, Mapping):
        _raise("unbound usage: provider telemetry cannot be projected")
    payload: dict[str, Any] = {}
    for name in PROVIDER_RESOURCE_FIELDS:
        if name not in record:
            _raise(f"unbound usage: provider telemetry omitted {name}")
        payload[name] = dict(record[name]) if isinstance(record[name], Mapping) else record[name]
    return payload


def _project_work_resources(record: Any) -> dict[str, Any]:
    compute_method = getattr(record, "to_compute_fields", None)
    if callable(compute_method):
        compute = _mapping(compute_method(), name="work compute projection")
        overhead = _attr(record, "audit_and_verification_overhead")
        payload = {
            "cpu_seconds": dict(compute["cpu_seconds"]),
            "gpu_seconds": dict(compute["gpu_seconds"]),
            "peak_memory": dict(compute["peak_memory"]),
            "wall_clock_duration": dict(compute["wall_clock_duration"]),
            "audit_and_verification_overhead": (
                _quantity_from_sample(overhead)
                if overhead is not None
                else unavailable("not_reported")
            ),
        }
        return payload
    if not isinstance(record, Mapping):
        _raise("unbound usage: work telemetry cannot be projected")
    payload = {}
    for name in WORK_RESOURCE_FIELDS:
        if name not in record:
            _raise(f"unbound usage: work telemetry omitted {name}")
        payload[name] = dict(record[name]) if isinstance(record[name], Mapping) else record[name]
    return payload


def _identity_from_record(record: Any, *, field: str) -> dict[str, Any]:
    value = _attr(record, field)
    if not value:
        return unavailable("not_admitted")
    return observed_identity(str(value))


def reconcile_actual_resource_use(
    *,
    task_id: str,
    actual_resource_use: Mapping[str, Any] | None,
    provider_usage: Any = None,
    work_telemetry: Any = None,
    reason: str | _er.ReasonCode = _er.ReasonCode.NOT_YET_MEASURED,
) -> dict[str, Any]:
    """Bind claimed resources to admitted provider/work telemetry."""

    provider = _require_admitted_binding(
        provider_usage, kind="provider", expected_task_id=task_id
    )
    work = _require_admitted_binding(
        work_telemetry, kind="work", expected_task_id=task_id
    )
    blank = unavailable(reason)
    resources: dict[str, Any] = {
        "provider_usage_record_id": dict(blank),
        "work_telemetry_record_id": dict(blank),
        "span_id": dict(blank),
        **{name: dict(blank) for name in RESOURCE_QUANTITY_FIELDS},
    }
    span_ids: list[str] = []
    if provider is not None:
        resources["provider_usage_record_id"] = _identity_from_record(
            provider, field="record_id"
        )
        projected = _project_provider_resources(provider)
        resources.update(projected)
        span = _attr(provider, "span_id")
        if span:
            span_ids.append(str(span))
    if work is not None:
        resources["work_telemetry_record_id"] = _identity_from_record(
            work, field="record_id"
        )
        resources.update(_project_work_resources(work))
        span = _attr(work, "span_id")
        if span:
            span_ids.append(str(span))
        outcome = _attr(work, "final_task_outcome")
        if outcome:
            resources["_work_final_outcome"] = str(outcome)
    if span_ids:
        unique = list(dict.fromkeys(span_ids))
        if len(unique) != 1:
            _raise("unbound usage: provider and work telemetry spans disagree")
        resources["span_id"] = observed_identity(unique[0])
    if actual_resource_use is not None:
        claimed = _mapping(actual_resource_use, name="actual_resource_use")
        for key, value in claimed.items():
            if key not in resources:
                _raise(f"actual_resource_use contains unknown field {key}")
            expected = resources[key]
            if not _same_canonical(expected, value):
                _raise(
                    "unbound usage: actual_resource_use does not reconcile to "
                    f"provider/work telemetry at {key}"
                )
    resources.pop("_work_final_outcome", None)
    return resources


def _ordered_stage_records(
    stages_run: Sequence[Mapping[str, Any]],
    stages_skipped: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_name: dict[str, dict[str, Any]] = {}
    if not isinstance(stages_run, Sequence) or isinstance(stages_run, (str, bytes)):
        _raise("stage mismatch: stages_run must be an array")
    if not isinstance(stages_skipped, Sequence) or isinstance(stages_skipped, (str, bytes)):
        _raise("stage mismatch: stages_skipped_and_reason must be an array")
    for item in stages_run:
        record = _mapping(item, name="stages_run item")
        stage = record.get("stage")
        if stage not in ORDERED_STAGES:
            _raise(f"stage mismatch: unknown run stage {stage!r}")
        if stage in by_name:
            _raise(f"stage mismatch: stage {stage} appears more than once")
        reason = record.get("reason")
        if reason not in STAGE_RUN_REASONS:
            _raise(f"stage mismatch: invalid run reason for {stage}")
        resolved = record.get("resolved")
        if type(resolved) is not bool:
            _raise(f"stage mismatch: resolved must be a boolean for {stage}")
        by_name[stage] = {
            "stage": stage,
            "action": "run",
            "reason": reason,
            "resolved": resolved,
        }
    for item in stages_skipped:
        record = _mapping(item, name="stages_skipped_and_reason item")
        stage = record.get("stage")
        if stage not in ORDERED_STAGES:
            _raise(f"stage mismatch: unknown skipped stage {stage!r}")
        if stage in by_name:
            _raise(f"stage mismatch: stage {stage} appears more than once")
        reason = record.get("reason")
        if reason not in STAGE_SKIP_REASONS:
            _raise(f"stage mismatch: invalid skip reason for {stage}")
        by_name[stage] = {
            "stage": stage,
            "action": "skip",
            "reason": reason,
            "resolved": False,
        }
    missing = [stage for stage in ORDERED_STAGES if stage not in by_name]
    extra = sorted(set(by_name) - set(ORDERED_STAGES))
    if missing or extra:
        _raise(
            "stage mismatch: route receipts must record every ladder stage "
            f"exactly once; missing={missing} extra={extra}"
        )
    return [by_name[stage] for stage in ORDERED_STAGES]


def _assert_ladder_invariants(
    ordered: Sequence[Mapping[str, Any]],
    *,
    selected_executor: Mapping[str, Any],
    escalation_reason: str,
) -> dict[str, Any]:
    resolved = [item for item in ordered if item["resolved"]]
    if len(resolved) != 1:
        _raise("stage mismatch: exactly one stage must resolve")
    winner = resolved[0]
    if winner["action"] != "run":
        _raise("stage mismatch: a skipped stage cannot resolve")
    if winner["reason"] not in RESOLVING_RUN_REASONS:
        _raise("stage mismatch: resolving reason cannot mark a stage resolved")
    seen_resolution = False
    for item in ordered:
        stage = str(item["stage"])
        if item["action"] == "run":
            if stage in DETERMINISTIC_STAGES:
                if item["resolved"]:
                    if item["reason"] != "eligible_deterministic_evidence":
                        _raise(
                            "stage mismatch: deterministic resolution requires "
                            "eligible_deterministic_evidence"
                        )
                elif item["reason"] != "unresolved_deterministic_evidence":
                    _raise(
                        "stage mismatch: a non-resolving deterministic run requires "
                        "unresolved_deterministic_evidence"
                    )
            elif stage in MODEL_STAGES:
                if not item["resolved"] or item["reason"] != "smallest_adequate_executor":
                    _raise(
                        "stage mismatch: model stages cannot be run without resolving "
                        "as the smallest adequate executor"
                    )
            elif stage == HUMAN_STAGE:
                if not item["resolved"] or item["reason"] != "human_decision_required":
                    _raise(
                        "stage mismatch: human_decision cannot be run without resolving"
                    )
        if item["resolved"]:
            seen_resolution = True
            continue
        if seen_resolution and (
            item["action"] != "skip" or item["reason"] != "resolved_by_prior_stage"
        ):
            _raise(
                "stage mismatch: stages after the resolver must skip with "
                "resolved_by_prior_stage"
            )
        if not seen_resolution and item["reason"] == "resolved_by_prior_stage":
            _raise("stage mismatch: resolved_by_prior_stage cannot precede the resolver")
    executor = _mapping(selected_executor, name="selected_executor")
    stage = executor.get("stage")
    if stage != winner["stage"]:
        _raise("stage mismatch: selected_executor.stage must match the resolving stage")
    expected_class = executor_class_for_stage(winner["stage"])
    expected_route = route_for_stage(winner["stage"])
    if executor.get("executor_class") != expected_class:
        _raise("stage mismatch: selected_executor.executor_class does not match the stage")
    if executor.get("route") != expected_route:
        _raise("stage mismatch: selected_executor.route does not match the stage")
    if expected_class == "deterministic":
        if escalation_reason != "not_applicable":
            _raise("stage mismatch: deterministic routes cannot carry an escalation reason")
        if winner["reason"] != "eligible_deterministic_evidence":
            _raise("stage mismatch: deterministic resolution requires eligible evidence")
    elif expected_class == "human":
        if escalation_reason not in {
            "human_decision_required",
            "required_tier_unavailable",
        }:
            _raise("stage mismatch: human routes require a human escalation reason")
        if winner["reason"] != "human_decision_required":
            _raise("stage mismatch: human resolution requires human_decision_required")
    else:
        if escalation_reason not in {
            "smallest_adequate_executor",
            "unresolved_deterministic_evidence",
        }:
            _raise("stage mismatch: model routes require a model escalation reason")
        if winner["reason"] != "smallest_adequate_executor":
            _raise("stage mismatch: model resolution requires smallest_adequate_executor")
    return dict(winner)


def _assert_decisive_evidence(
    evidence: Mapping[str, Any], *, resolving_stage: str
) -> None:
    payload = _mapping(evidence, name="decisive_evidence")
    if payload.get("stage") != resolving_stage:
        _raise("absent decisive evidence: evidence stage must match the resolver")
    kind = payload.get("evidence_kind")
    expected_kind = STAGE_EVIDENCE_KIND[resolving_stage]
    if kind != expected_kind:
        _raise("absent decisive evidence: evidence_kind does not match the resolver")
    evidence_id = payload.get("evidence_id")
    if not isinstance(evidence_id, str) or not evidence_id.strip():
        _raise("absent decisive evidence")
    state = payload.get("truth_state")
    if state not in {"observed", "verified"}:
        _raise("absent decisive evidence")
    if state == "verified":
        if "verifier_id" not in payload or "verifier_receipt_cid" not in payload:
            _raise("absent decisive evidence: verified evidence requires a verifier")
    elif "verifier_id" in payload or "verifier_receipt_cid" in payload:
        _raise("absent decisive evidence: observed evidence cannot claim verification")


def _question_value(node: Mapping[str, Any]) -> str | None:
    state = node.get("truth_state")
    if state == "unavailable":
        return None
    value = node.get("value")
    return value if isinstance(value, str) and value else None


def _assert_question_closure(
    closure: Mapping[str, Any],
    *,
    executor_class: str,
) -> None:
    payload = _mapping(closure, name="question_closure")
    status = payload.get("status")
    if status not in QUESTION_CLOSURE_STATUSES:
        _raise("question_closure.status is not a closed value")
    question = _mapping(payload.get("question_id"), name="question_closure.question_id")
    evidence = _mapping(
        payload.get("closure_evidence"), name="question_closure.closure_evidence"
    )
    question_value = _question_value(question)
    evidence_value = _question_value(evidence)
    if executor_class == "deterministic":
        if status != "not_applicable":
            _raise("deterministic routes cannot open or close a model question")
        if question.get("truth_state") != "unavailable":
            _raise("deterministic question_id must remain unavailable")
        if evidence.get("truth_state") != "unavailable":
            _raise("deterministic closure evidence must remain unavailable")
        return
    if status == "not_applicable":
        _raise("model and human routes require a typed unresolved question")
    if question_value is None or not _er.DIGEST_RE.fullmatch(question_value):
        _raise("model and human routes require a typed unresolved question")
    if status == "closed":
        if evidence_value is None or not _er.CID_RE.fullmatch(evidence_value):
            _raise("claimed closure without evidence")
        if evidence.get("truth_state") not in {"observed", "verified"}:
            _raise("claimed closure without evidence")
        return
    if evidence.get("truth_state") != "unavailable":
        _raise("open questions cannot carry closure evidence")


def _assert_unnecessary_escalation(
    observation: Mapping[str, Any],
    *,
    executor_class: str,
) -> bool | None:
    payload = _mapping(observation, name="unnecessary_escalation")
    state = payload.get("truth_state")
    if state == "unavailable":
        if executor_class == "deterministic" and payload.get("reason_code") != "not_applicable":
            _raise("deterministic unnecessary_escalation must be not_applicable when unavailable")
        return None
    if state != "observed":
        _raise("unnecessary_escalation truth_state is not a closed value")
    flag = payload.get("unnecessary")
    if type(flag) is not bool:
        _raise("unnecessary_escalation.unnecessary must be a boolean")
    if executor_class == "deterministic" and flag is True:
        _raise("stage mismatch: deterministic routes cannot be unnecessary escalations")
    return flag


def _assert_unavailable_never_zero(payload: Mapping[str, Any]) -> None:
    for path, node, state in iter_evidence_nodes(payload):
        try:
            _er._assert_truth_state_payload(node, state, path=path or "$")
        except _er.EfficiencyReceiptError as exc:
            raise _wrap_efficiency(exc) from exc
        if state == "unavailable":
            for key, child in node.items():
                if type(child) is int and child == 0:
                    _raise(
                        f"{path}.{key}: unavailable evidence cannot encode numeric zero"
                    )


def _assert_matching_unavailable_index(payload: Mapping[str, Any]) -> tuple[str, ...]:
    body = {key: value for key, value in payload.items() if key != "receipt_cid"}
    actual = collect_unavailable_fields(body)
    claimed = payload.get("explicit_unavailable_fields")
    if not isinstance(claimed, list) or any(not isinstance(item, str) for item in claimed):
        _raise("explicit_unavailable_fields must be a string array")
    if tuple(claimed) != actual:
        _raise("explicit_unavailable_fields must list every unavailable evidence path")
    return actual


def _escalated(executor_class: str) -> bool:
    return executor_class != "deterministic"


def _question_opened(status: str) -> bool:
    return status in {"open", "closed"}


def _rate_inputs_from_receipt(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    executor = _mapping(payload["selected_executor"], name="selected_executor")
    executor_class = str(executor["executor_class"])
    unnecessary = _assert_unnecessary_escalation(
        _mapping(payload["unnecessary_escalation"], name="unnecessary_escalation"),
        executor_class=executor_class,
    )
    closure = _mapping(payload["question_closure"], name="question_closure")
    return {
        "executor_class": executor_class,
        "escalated": _escalated(executor_class),
        "unnecessary": unnecessary,
        "question_status": str(closure["status"]),
    }


def _compute_rates_from_inputs(
    rows: Sequence[Mapping[str, Any]],
    *,
    sensor_id: str = RATE_SENSOR_ID,
) -> dict[str, Any]:
    if not rows:
        blank = unavailable_rate("not_yet_measured")
        return {name: dict(blank) for name in REQUESTED_RATES}

    total = len(rows)
    class_counts = {name: 0 for name in EXECUTOR_CLASSES}
    for row in rows:
        class_counts[str(row["executor_class"])] += 1
    rates: dict[str, Any] = {
        "deterministic_resolution_rate": measured_rate(
            class_counts["deterministic"], total, sensor_id=sensor_id
        ),
        "small_model_resolution_rate": measured_rate(
            class_counts["local_small_model"], total, sensor_id=sensor_id
        ),
        "frontier_model_rate": measured_rate(
            class_counts["remote_frontier_model"], total, sensor_id=sensor_id
        ),
        "human_rate": measured_rate(class_counts["human"], total, sensor_id=sensor_id),
    }

    escalated_rows = [row for row in rows if row["escalated"]]
    if not escalated_rows:
        unavailable_escalation = unavailable_rate("not_applicable")
        rates["escalation_precision"] = dict(unavailable_escalation)
        rates["unnecessary_escalation_rate"] = dict(unavailable_escalation)
    elif any(row["unnecessary"] is None for row in escalated_rows):
        missing = unavailable_rate("not_yet_measured")
        rates["escalation_precision"] = dict(missing)
        rates["unnecessary_escalation_rate"] = dict(missing)
    else:
        unnecessary_count = sum(1 for row in escalated_rows if row["unnecessary"] is True)
        necessary_count = len(escalated_rows) - unnecessary_count
        denom = len(escalated_rows)
        rates["escalation_precision"] = measured_rate(
            necessary_count, denom, sensor_id=sensor_id
        )
        rates["unnecessary_escalation_rate"] = measured_rate(
            unnecessary_count, denom, sensor_id=sensor_id
        )

    question_rows = [row for row in rows if _question_opened(str(row["question_status"]))]
    unavailable_questions = [
        row for row in rows if str(row["question_status"]) not in QUESTION_CLOSURE_STATUSES
    ]
    if unavailable_questions:
        rates["unresolved_question_closure_rate"] = unavailable_rate("not_yet_measured")
    elif not question_rows:
        rates["unresolved_question_closure_rate"] = unavailable_rate("not_applicable")
    else:
        closed = sum(1 for row in question_rows if row["question_status"] == "closed")
        rates["unresolved_question_closure_rate"] = measured_rate(
            closed, len(question_rows), sensor_id=sensor_id
        )
    return rates


def compute_requested_rates(
    receipts: Sequence[Any],
    *,
    sensor_id: str = RATE_SENSOR_ID,
) -> dict[str, Any]:
    """Compute every requested rate without imputing unavailable denominators."""

    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes, bytearray)):
        _raise("receipts must be a sequence of closed route receipts")
    rows: list[dict[str, Any]] = []
    for item in receipts:
        if isinstance(item, RouteReceipt):
            payload = item.to_dict()
        else:
            payload = admit_route_receipt(item).to_dict()
        rows.append(_rate_inputs_from_receipt(payload))
    return _compute_rates_from_inputs(rows, sensor_id=sensor_id)


def _single_receipt_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _compute_rates_from_inputs([_rate_inputs_from_receipt(payload)])


def _assert_metrics_match(payload: Mapping[str, Any]) -> None:
    expected = _single_receipt_metrics(payload)
    claimed = payload.get("metrics")
    if not isinstance(claimed, Mapping):
        _raise("metrics must be an object")
    if set(claimed) != set(REQUESTED_RATES):
        _raise("metrics must contain every requested rate")
    for name in REQUESTED_RATES:
        if not _same_canonical(claimed[name], expected[name]):
            node = claimed[name]
            if (
                isinstance(node, Mapping)
                and node.get("truth_state") == "measured"
                and (
                    not isinstance(node.get("denominator"), int)
                    or int(node.get("denominator") or 0) < 1
                )
            ):
                _raise("division by missing data")
            if (
                isinstance(node, Mapping)
                and node.get("truth_state") == "measured"
                and expected[name].get("truth_state") == "unavailable"
            ):
                _raise("division by missing data")
            _raise(f"metrics.{name} does not match observed route facts")


@dataclass(frozen=True)
class RouteReceipt:
    """One closed routing-decision / route receipt."""

    SCHEMA: ClassVar[str] = ROUTING_DECISION_SCHEMA
    CID_FIELD: ClassVar[str] = "receipt_cid"

    _payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        encoded = json.loads(canonical_bytes(self._payload).decode("utf-8"))
        if not isinstance(encoded, dict):
            _raise("canonical payload must remain an object")
        return encoded

    def canonical_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    def canonical_json(self) -> str:
        return self.canonical_bytes().decode("utf-8")

    def to_json(self) -> str:
        return self.canonical_json()

    @property
    def receipt_cid(self) -> str:
        return str(self._payload["receipt_cid"])

    @property
    def schema(self) -> str:
        return self.SCHEMA

    @property
    def schema_version(self) -> int:
        return SCHEMA_VERSION

    @property
    def identity(self) -> dict[str, Any]:
        return dict(self._payload["identity"])

    @property
    def stages_run(self) -> list[dict[str, Any]]:
        return list(self._payload["stages_run"])

    @property
    def stages_skipped_and_reason(self) -> list[dict[str, Any]]:
        return list(self._payload["stages_skipped_and_reason"])

    @property
    def decisive_evidence(self) -> dict[str, Any]:
        return dict(self._payload["decisive_evidence"])

    @property
    def escalation_reason(self) -> str:
        return str(self._payload["escalation_reason"])

    @property
    def selected_executor(self) -> dict[str, Any]:
        return dict(self._payload["selected_executor"])

    @property
    def actual_resource_use(self) -> dict[str, Any]:
        return dict(self._payload["actual_resource_use"])

    @property
    def final_outcome(self) -> str:
        return str(self._payload["final_outcome"])

    @property
    def unnecessary_escalation(self) -> dict[str, Any]:
        return dict(self._payload["unnecessary_escalation"])

    @property
    def question_closure(self) -> dict[str, Any]:
        return dict(self._payload["question_closure"])

    @property
    def metrics(self) -> dict[str, Any]:
        return dict(self._payload["metrics"])

    @property
    def authority(self) -> dict[str, Any]:
        return dict(self._payload["authority"])

    @property
    def observation_only(self) -> bool:
        return True

    @property
    def validates_patches(self) -> bool:
        return False

    @property
    def promotes_policy(self) -> bool:
        return False

    @property
    def explicit_unavailable_fields(self) -> tuple[str, ...]:
        return tuple(self._payload["explicit_unavailable_fields"])

    def requested_rates(self) -> dict[str, Any]:
        return dict(self.metrics)

    def round_trip(self) -> bytes:
        encoded = self.canonical_bytes()
        replayed = admit_route_receipt(encoded).canonical_bytes()
        if replayed != encoded:
            _raise("canonical bytes did not round-trip")
        return encoded

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RouteReceipt":
        return admit_route_receipt(payload)

    @classmethod
    def from_bytes(cls, payload: bytes | str) -> "RouteReceipt":
        return admit_route_receipt(payload)


def admit_route_receipt(payload: Mapping[str, Any] | str | bytes) -> RouteReceipt:
    data = _decode_json(payload, name="route receipt")
    schema = load_routing_decision_schema()
    validate_against_schema(data, schema, path="$")
    if data.get("schema") != ROUTING_DECISION_SCHEMA:
        _raise("schema identity mismatch")
    if data.get("authority") != AUTHORITY:
        _raise("receipts report routing observations only and cannot validate patches or promote policy")
    identity = _mapping(data.get("identity"), name="identity")
    for field in IDENTITY_FIELDS:
        if field not in identity:
            _raise(f"identity missing {field}")
    if _er.GIT_OID_RE.fullmatch(str(identity["repository_commit"])) is None:
        _raise("repository_commit must be a git object id")
    if _er.GIT_OID_RE.fullmatch(str(identity["repository_tree"])) is None:
        _raise("repository_tree must be a git object id")
    if _er.CID_RE.fullmatch(str(identity["task_cid"])) is None:
        _raise("task_cid must be a CIDv1")
    if data.get("final_outcome") not in TASK_OUTCOMES:
        _raise("final_outcome is not a closed value")
    ordered = _ordered_stage_records(
        data.get("stages_run") or (),
        data.get("stages_skipped_and_reason") or (),
    )
    winner = _assert_ladder_invariants(
        ordered,
        selected_executor=_mapping(data.get("selected_executor"), name="selected_executor"),
        escalation_reason=str(data.get("escalation_reason")),
    )
    _assert_decisive_evidence(
        _mapping(data.get("decisive_evidence"), name="decisive_evidence"),
        resolving_stage=str(winner["stage"]),
    )
    executor_class = str(data["selected_executor"]["executor_class"])
    _assert_question_closure(
        _mapping(data.get("question_closure"), name="question_closure"),
        executor_class=executor_class,
    )
    _assert_unnecessary_escalation(
        _mapping(data.get("unnecessary_escalation"), name="unnecessary_escalation"),
        executor_class=executor_class,
    )
    _assert_unavailable_never_zero(data)
    _assert_matching_unavailable_index(data)
    _assert_metrics_match(data)
    admitted = _canonical_copy(data)
    _bind_cid(admitted, field="receipt_cid")
    validate_against_schema(admitted, schema, path="$")
    return RouteReceipt(_payload=admitted)


def admit_routing_decision(payload: Mapping[str, Any] | str | bytes) -> RouteReceipt:
    return admit_route_receipt(payload)


def _default_question_closure(executor_class: str) -> dict[str, Any]:
    if executor_class == "deterministic":
        return {
            "status": "not_applicable",
            "question_id": unavailable("not_applicable"),
            "closure_evidence": unavailable("not_applicable"),
        }
    _raise("model and human routes require a typed unresolved question")
    raise AssertionError("unreachable")


def build_route_receipt(
    *,
    task_id: str,
    task_cid: str,
    objective_id: str,
    objective_revision: str,
    repository_commit: str,
    repository_tree: str,
    policy_identity: str,
    stages_run: Sequence[Mapping[str, Any]],
    stages_skipped_and_reason: Sequence[Mapping[str, Any]],
    decisive_evidence: Mapping[str, Any],
    escalation_reason: str,
    selected_executor: Mapping[str, Any] | None = None,
    actual_resource_use: Mapping[str, Any] | None = None,
    final_outcome: str,
    provider_usage: Any = None,
    work_telemetry: Any = None,
    unnecessary_escalation: Mapping[str, Any] | bool | None = None,
    question_closure: Mapping[str, Any] | None = None,
    reason: str | _er.ReasonCode = _er.ReasonCode.NOT_YET_MEASURED,
) -> RouteReceipt:
    """Build a closed route receipt; omitted telemetry stays unavailable."""

    if final_outcome not in TASK_OUTCOMES:
        _raise("final_outcome is not a closed value")
    if escalation_reason not in ESCALATION_REASONS:
        _raise("escalation_reason is not a closed value")
    ordered = _ordered_stage_records(stages_run, stages_skipped_and_reason)
    resolved = [item for item in ordered if item["resolved"]]
    if len(resolved) != 1:
        _raise("stage mismatch: exactly one stage must resolve")
    executor = (
        dict(selected_executor)
        if selected_executor is not None
        else selected_executor_for_stage(str(resolved[0]["stage"]))
    )
    _assert_ladder_invariants(
        ordered,
        selected_executor=executor,
        escalation_reason=escalation_reason,
    )
    executor_class = executor["executor_class"]
    if work_telemetry is not None:
        work_outcome = _attr(work_telemetry, "final_task_outcome")
        if work_outcome and work_outcome != final_outcome:
            _raise("final_outcome does not reconcile to work telemetry")
    resources = reconcile_actual_resource_use(
        task_id=_text(task_id, name="task_id"),
        actual_resource_use=actual_resource_use,
        provider_usage=provider_usage,
        work_telemetry=work_telemetry,
        reason=reason,
    )
    if isinstance(unnecessary_escalation, bool):
        unnecessary_payload = observed_unnecessary_escalation(unnecessary_escalation)
    elif unnecessary_escalation is None:
        if executor_class == "deterministic":
            unnecessary_payload = observed_unnecessary_escalation(False)
        else:
            unnecessary_payload = unavailable("not_yet_measured")
    else:
        unnecessary_payload = dict(unnecessary_escalation)
    if question_closure is None:
        closure_payload = _default_question_closure(executor_class)
    else:
        closure_payload = dict(question_closure)

    payload: dict[str, Any] = {
        "schema": ROUTING_DECISION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "task_id": _text(task_id, name="task_id"),
            "task_cid": task_cid,
            "objective_id": _text(objective_id, name="objective_id"),
            "objective_revision": _text(objective_revision, name="objective_revision"),
            "repository_commit": repository_commit,
            "repository_tree": repository_tree,
            "policy_identity": _text(policy_identity, name="policy_identity"),
        },
        "stages_run": [dict(item) for item in stages_run],
        "stages_skipped_and_reason": [dict(item) for item in stages_skipped_and_reason],
        "decisive_evidence": dict(decisive_evidence),
        "escalation_reason": escalation_reason,
        "selected_executor": executor,
        "actual_resource_use": resources,
        "final_outcome": final_outcome,
        "unnecessary_escalation": unnecessary_payload,
        "question_closure": closure_payload,
        "authority": dict(AUTHORITY),
    }
    payload["metrics"] = _single_receipt_metrics(payload)
    payload["explicit_unavailable_fields"] = list(collect_unavailable_fields(payload))
    _bind_cid(payload, field="receipt_cid")
    return admit_route_receipt(payload)


def build_routing_decision(**kwargs: Any) -> RouteReceipt:
    return build_route_receipt(**kwargs)


def round_trip_route_receipt(payload: Mapping[str, Any] | str | bytes) -> bytes:
    return admit_route_receipt(payload).round_trip()


def round_trip_routing_decision(payload: Mapping[str, Any] | str | bytes) -> bytes:
    return round_trip_route_receipt(payload)


__all__ = (
    "AUTHORITY",
    "DETERMINISTIC_STAGES",
    "DecisionReceiptError",
    "EVIDENCE_KINDS",
    "EXECUTOR_CLASSES",
    "ExecutorClass",
    "IDENTITY_FIELDS",
    "MODEL_ROUTES",
    "ORDERED_STAGES",
    "OBJECTIVE_ID",
    "QUESTION_CLOSURE_STATUSES",
    "QuestionClosureStatus",
    "RATE_SENSOR_ID",
    "REQUESTED_RATES",
    "RESOURCE_QUANTITY_FIELDS",
    "ROUTE_RECEIPT_FIELDS",
    "ROUTE_RECEIPT_INTERFACE",
    "ROUTE_RECEIPT_SCHEMA",
    "ROUTING_DECISION_INTERFACE",
    "ROUTING_DECISION_SCHEMA",
    "ROUTING_DECISION_SCHEMA_PATH",
    "RouteReceipt",
    "STAGE_RUN_REASONS",
    "STAGE_SKIP_REASONS",
    "admit_route_receipt",
    "admit_routing_decision",
    "build_route_receipt",
    "build_routing_decision",
    "canonical_bytes",
    "canonical_json",
    "collect_unavailable_fields",
    "compute_requested_rates",
    "content_identity",
    "emit_decision",
    "executor_class_for_stage",
    "load_routing_decision_schema",
    "measured_quantity",
    "measured_rate",
    "observed_identity",
    "observed_unnecessary_escalation",
    "ratio_millionths",
    "reconcile_actual_resource_use",
    "round_trip_route_receipt",
    "round_trip_routing_decision",
    "route_for_stage",
    "selected_executor_for_stage",
    "unavailable",
    "unavailable_rate",
    "validate_against_schema",
)
