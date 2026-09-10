"""ASEH-074 closed promotion or honest non-promotion decision.

The receipt evaluates current paired populations, safety, quality, efficiency,
audit overhead, and missing evidence. It emits exactly one mandated
disposition and cannot mutate a policy pointer or self-authorize promotion.
"""

from __future__ import annotations

import ast
import copy
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts import (
    content_identity,
)


ROOT = Path(__file__).resolve().parents[4]
INVENTORY = ROOT / "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory"
DECISION_PATH = INVENTORY / "promotion_decision.json"
SCHEMA_PATH = (
    ROOT / "ipfs_accelerate_py/agent_supervisor/control/schemas/promotion_decision.schema.json"
)
REQUIREMENTS_PATH = (
    ROOT / "docs/architecture/agent_supervisor_efficiency_state_hardening.requirements.json"
)
LIVE_COHORT_PATH = (
    ROOT / "benchmarks/agent_supervisor/efficiency_state_hardening/live_cohort_manifest.json"
)

SCHEMA_ID = "ipfs_accelerate_py/agent-supervisor/aseh-promotion-decision@1"
INTERFACE = "AsehPromotionDecision@1"
PROGRAM_ID = "agent-supervisor-efficiency-and-state-hardening-v1"
TASK_ID = "ASEH-074"
OBJECTIVE_ID = "ASEH-G080"
PLAN_REVISION = "ASEH-PLAN-R1"
POLICY_IDENTITY = "ipfs_accelerate_py/agent-supervisor/aseh-supervisor-policy@1"

CLOSED_DISPOSITIONS = (
    "non_promoted_unmeasured",
    "non_promoted_safety_or_quality",
    "non_promoted_efficiency",
    "promotion_eligible_operator_authorization_required",
)
DISPOSITION_KINDS = (
    "unmeasured",
    "safety_or_quality",
    "efficiency",
    "eligible_operator_authorization_required",
)
KIND_FOR_DISPOSITION = {
    "non_promoted_unmeasured": "unmeasured",
    "non_promoted_safety_or_quality": "safety_or_quality",
    "non_promoted_efficiency": "efficiency",
    "promotion_eligible_operator_authorization_required": (
        "eligible_operator_authorization_required"
    ),
}
DISPOSITION_FOR_KIND = {kind: name for name, kind in KIND_FOR_DISPOSITION.items()}

POPULATION_SPECS = (
    ("hermetic", "hermetic_qualification.json", "hermetic", 60, False),
    ("historical", "historical_qualification.json", "historical", 20, False),
    ("live_shadow", "live_shadow_qualification.json", "live-shadow", 10, True),
    ("canary", "canary_qualification.json", "canary", 1, False),
)
EVIDENCE_PATHS = (
    "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/hermetic_qualification.json",
    "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/historical_qualification.json",
    "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/live_shadow_qualification.json",
    "docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/canary_qualification.json",
    "benchmarks/agent_supervisor/efficiency_state_hardening/live_cohort_manifest.json",
)
VALIDATOR_COMMAND = (
    "python3",
    "-m",
    "pytest",
    "-q",
    "test/api/agent_supervisor/efficiency_state_hardening/test_promotion_decision.py",
)
NONCLAIMS = (
    "This receipt does not mutate a policy pointer.",
    "This receipt does not authorize promotion.",
    "Hermetic evidence is not live and cannot satisfy production promotion.",
    "Missing measurements are unavailable, never numeric zero.",
)
UNAVAILABLE = {"reason_code": "not_yet_measured", "truth_state": "unavailable"}
EFFICIENCY_PERCENT_FIELDS = (
    "median_live_input_token_reduction_percent",
    "total_weighted_provider_cost_reduction_percent",
    "frontier_model_call_reduction_percent",
    "retry_token_reduction_percent",
    "low_risk_deterministic_or_small_model_share_percent",
    "eligible_context_pack_reuse_percent",
    "manual_recovery_rate_strictly_less_than_percent",
    "overall_wall_time_maximum_regression_percent",
)
CID_RE = re.compile(r"^b[a-z2-7]{20,}$")
GIT_OID_RE = re.compile(r"^[0-9a-f]{40}$")
SIBLING_TEST_PREFIXES = (
    "test.api.agent_supervisor.efficiency_state_hardening.test_",
    "test.api.test_agent_supervisor_",
)
MAX_TEXT_BYTES = 8192
MAX_ARRAY_ITEMS = 256
MAX_INTEGER = 10**18
_JSON_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool}


class PromotionDecisionError(ValueError):
    """Closed promotion-decision schema or evaluation violation."""


def _canonical(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), path
    return payload


def _load_canonical(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert isinstance(payload, dict), path
    assert raw == _canonical(payload)
    return payload


def _walk_object_schemas(node: Any, *, path: str = "$") -> list[str]:
    open_paths: list[str] = []
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            if node.get("additionalProperties") is not False:
                open_paths.append(path)
        for key, child in node.items():
            open_paths.extend(_walk_object_schemas(child, path=f"{path}.{key}"))
    elif isinstance(node, list):
        for index, child in enumerate(node):
            open_paths.extend(_walk_object_schemas(child, path=f"{path}[{index}]"))
    return open_paths


def _resolve_ref(schema: Mapping[str, Any], defs: Mapping[str, Any]) -> Mapping[str, Any]:
    ref = schema.get("$ref")
    if not isinstance(ref, str):
        return schema
    prefix = "#/$defs/"
    if not ref.startswith(prefix) or "/" in ref[len(prefix):]:
        raise PromotionDecisionError("unsupported schema $ref")
    target = defs.get(ref[len(prefix):])
    if not isinstance(target, Mapping):
        raise PromotionDecisionError(f"unknown schema $defs entry {ref}")
    return target


def _type_matches(instance: Any, expected: str) -> bool:
    if expected == "integer":
        return type(instance) is int
    python_type = _JSON_TYPES.get(expected)
    return python_type is not None and isinstance(instance, python_type)


def validate_against_schema(
    instance: Any,
    schema: Mapping[str, Any],
    *,
    defs: Mapping[str, Any] | None = None,
    path: str = "$",
) -> None:
    resolved = _resolve_ref(schema, defs or {})
    local_defs = resolved.get("$defs") if isinstance(resolved.get("$defs"), Mapping) else None
    active_defs: Mapping[str, Any] = local_defs if local_defs is not None else (defs or {})
    if local_defs is None and "$defs" in schema and isinstance(schema["$defs"], Mapping):
        active_defs = schema["$defs"]

    if "oneOf" in resolved:
        variants = resolved["oneOf"]
        if not isinstance(variants, list) or not variants:
            raise PromotionDecisionError(f"{path}: oneOf must be a nonempty array")
        matches = 0
        first_error: PromotionDecisionError | None = None
        for variant in variants:
            if not isinstance(variant, Mapping):
                raise PromotionDecisionError(f"{path}: oneOf entries must be objects")
            try:
                validate_against_schema(instance, variant, defs=active_defs, path=path)
            except PromotionDecisionError as exc:
                if first_error is None:
                    first_error = exc
                continue
            matches += 1
        if matches != 1:
            detail = f": {first_error}" if first_error is not None and matches == 0 else ""
            raise PromotionDecisionError(
                f"{path}: does not uniquely match a closed variant{detail}"
            )
        return

    if "const" in resolved and instance != resolved["const"]:
        raise PromotionDecisionError(f"{path}: must equal the closed const")
    if "enum" in resolved:
        allowed = resolved["enum"]
        if not isinstance(allowed, list) or instance not in allowed:
            raise PromotionDecisionError(f"{path}: is not an allowed enum value")

    expected_type = resolved.get("type")
    if expected_type is not None:
        if not isinstance(expected_type, str) or not _type_matches(instance, expected_type):
            raise PromotionDecisionError(f"{path}: has the wrong JSON type")

    if isinstance(instance, str):
        if "minLength" in resolved and len(instance) < int(resolved["minLength"]):
            raise PromotionDecisionError(f"{path}: is shorter than minLength")
        if "maxLength" in resolved and len(instance) > int(resolved["maxLength"]):
            raise PromotionDecisionError(f"{path}: exceeds maxLength")
        pattern = resolved.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, instance) is None:
            raise PromotionDecisionError(f"{path}: does not match the closed pattern")
        encoded = instance.encode("utf-8")
        if b"\x00" in encoded or len(encoded) > MAX_TEXT_BYTES:
            raise PromotionDecisionError(f"{path}: is unsafe or too large")

    if type(instance) is int:
        if "minimum" in resolved and instance < int(resolved["minimum"]):
            raise PromotionDecisionError(f"{path}: is below the minimum bound")
        if "maximum" in resolved and instance > int(resolved["maximum"]):
            raise PromotionDecisionError(f"{path}: exceeds the maximum bound")
        if instance < 0 or instance > MAX_INTEGER:
            raise PromotionDecisionError(f"{path}: is outside the integer bound")

    if expected_type == "object" or isinstance(instance, Mapping):
        if not isinstance(instance, Mapping):
            if "properties" in resolved or expected_type == "object":
                raise PromotionDecisionError(f"{path}: must be an object")
        else:
            properties = resolved.get("properties")
            if not isinstance(properties, Mapping):
                properties = {}
            if resolved.get("additionalProperties") is False:
                unknown = sorted(set(instance) - set(properties))
                if unknown:
                    raise PromotionDecisionError(f"{path}: contains unknown fields: {unknown}")
            required = resolved.get("required")
            if isinstance(required, list):
                missing = [name for name in required if name not in instance]
                if missing:
                    raise PromotionDecisionError(f"{path}: missing required fields: {missing}")
            for key, child_schema in properties.items():
                if key in instance:
                    if not isinstance(child_schema, Mapping):
                        raise PromotionDecisionError(f"{path}.{key}: invalid nested schema")
                    validate_against_schema(
                        instance[key],
                        child_schema,
                        defs=active_defs,
                        path=f"{path}.{key}",
                    )

    if isinstance(instance, list):
        if "minItems" in resolved and len(instance) < int(resolved["minItems"]):
            raise PromotionDecisionError(f"{path}: has too few items")
        if "maxItems" in resolved and len(instance) > int(resolved["maxItems"]):
            raise PromotionDecisionError(f"{path}: exceeds maxItems")
        if len(instance) > MAX_ARRAY_ITEMS:
            raise PromotionDecisionError(f"{path}: exceeds the array bound")
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
            encoded_items = [
                json.dumps(item, sort_keys=True, separators=(",", ":")) for item in instance
            ]
            if len(set(encoded_items)) != len(encoded_items):
                raise PromotionDecisionError(f"{path}: items must be unique")


def _measured_count(node: Any) -> int | None:
    if not isinstance(node, Mapping):
        return None
    if node.get("truth_state") != "measured":
        return None
    value = node.get("value")
    if type(value) is not int:
        return None
    return value


def _is_unavailable(node: Any) -> bool:
    return isinstance(node, Mapping) and node.get("truth_state") == "unavailable"


def _population_unmeasured(
    receipt: Mapping[str, Any],
    minimum: int,
    *,
    require_live: bool,
) -> bool:
    if receipt.get("qualification") is not True:
        return True
    if receipt.get("disposition") in {"insufficient_evidence", "not_admitted"}:
        return True
    if require_live and receipt.get("live") is not True:
        return True
    count = _measured_count(receipt.get("pair_count"))
    return count is None or count < minimum


def _live_cohort_unmeasured(cohort: Mapping[str, Any]) -> bool:
    if cohort.get("live") is not True:
        return True
    if cohort.get("qualification") is not True:
        return True
    if cohort.get("shadow_admitted") is not True:
        return True
    count = cohort.get("count")
    if type(count) is not int or count < 10:
        return True
    return cohort.get("disposition") in {"insufficient_evidence", "not_admitted"}


def _safety_or_quality_failed(receipt: Mapping[str, Any]) -> bool:
    if receipt.get("disposition") == "safety_or_quality_failed":
        return True
    safety = receipt.get("safety")
    if isinstance(safety, Mapping):
        for field in ("escaped_critical_seeded_defects", "simulated_as_live_outcomes"):
            count = _measured_count(safety.get(field))
            if count is not None and count > 0:
                return True
    quality = receipt.get("quality")
    if isinstance(quality, Mapping):
        escaped = quality.get("escaped_selected_test_false_negatives")
        if isinstance(escaped, list) and escaped:
            return True
    return False


def _efficiency_below_threshold(
    observed: Mapping[str, Any],
    required: int,
    *,
    strictly_less_than: bool = False,
    maximum_regression: bool = False,
) -> bool:
    count = _measured_count(observed)
    if count is None:
        return False
    if strictly_less_than:
        return count >= required
    if maximum_regression:
        return count > required
    return count < required


def mandated_disposition(
    receipts: Mapping[str, Mapping[str, Any]],
    live_cohort: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    live_efficiency: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    hermetic = receipts["hermetic"]
    historical = receipts["historical"]
    live_shadow = receipts["live_shadow"]
    canary = receipts["canary"]
    if (
        _population_unmeasured(hermetic, 60, require_live=False)
        or _population_unmeasured(historical, 20, require_live=False)
        or _population_unmeasured(live_shadow, 10, require_live=True)
        or _live_cohort_unmeasured(live_cohort)
    ):
        return "non_promoted_unmeasured"
    if any(
        _safety_or_quality_failed(item)
        for item in (hermetic, historical, live_shadow, canary)
    ):
        return "non_promoted_safety_or_quality"
    metrics = live_efficiency or {}
    required_metric_names = EFFICIENCY_PERCENT_FIELDS + (
        "net_savings_after_audit_and_verification_overhead_positive",
    )
    if any(_is_unavailable(metrics.get(name)) or name not in metrics for name in required_metric_names):
        return "non_promoted_unmeasured"
    if (
        _efficiency_below_threshold(
            metrics["median_live_input_token_reduction_percent"],
            int(thresholds["median_live_input_token_reduction_percent"]),
        )
        or _efficiency_below_threshold(
            metrics["total_weighted_provider_cost_reduction_percent"],
            int(thresholds["total_weighted_provider_cost_reduction_percent"]),
        )
        or _efficiency_below_threshold(
            metrics["frontier_model_call_reduction_percent"],
            int(thresholds["frontier_model_call_reduction_percent"]),
        )
        or _efficiency_below_threshold(
            metrics["retry_token_reduction_percent"],
            int(thresholds["retry_token_reduction_percent"]),
        )
        or _efficiency_below_threshold(
            metrics["low_risk_deterministic_or_small_model_share_percent"],
            int(thresholds["low_risk_deterministic_or_small_model_share_percent"]),
        )
        or _efficiency_below_threshold(
            metrics["eligible_context_pack_reuse_percent"],
            int(thresholds["eligible_context_pack_reuse_percent"]),
        )
        or _efficiency_below_threshold(
            metrics["manual_recovery_rate_strictly_less_than_percent"],
            int(thresholds["manual_recovery_rate_strictly_less_than_percent"]),
            strictly_less_than=True,
        )
        or _efficiency_below_threshold(
            metrics["overall_wall_time_maximum_regression_percent"],
            int(thresholds["overall_wall_time_maximum_regression_percent"]),
            maximum_regression=True,
        )
        or (
            metrics["net_savings_after_audit_and_verification_overhead_positive"].get(
                "truth_state"
            )
            == "measured"
            and metrics["net_savings_after_audit_and_verification_overhead_positive"].get("value")
            != 1
        )
    ):
        return "non_promoted_efficiency"
    return "promotion_eligible_operator_authorization_required"


def _cite_pair_count(receipt: Mapping[str, Any]) -> dict[str, Any]:
    node = receipt.get("pair_count")
    if not isinstance(node, Mapping):
        raise PromotionDecisionError("pair_count must be a typed observation, never a bare integer")
    if _is_unavailable(node):
        if "value" in node or "count" in node:
            raise PromotionDecisionError("unavailable pair_count must not carry a numeric value")
        return {
            "reason_code": str(node["reason_code"]),
            "truth_state": "unavailable",
        }
    count = _measured_count(node)
    if count is None:
        raise PromotionDecisionError("pair_count is neither measured nor unavailable")
    return {
        "sensor_id": str(node["sensor_id"]),
        "truth_state": "measured",
        "unit": str(node["unit"]),
        "value": count,
    }


def _cite_quantity(node: Any) -> dict[str, Any]:
    if not isinstance(node, Mapping):
        raise PromotionDecisionError("quantity observation must be an object")
    if _is_unavailable(node):
        if "value" in node or "count" in node:
            raise PromotionDecisionError("unavailable evidence must not carry a numeric value")
        return {
            "reason_code": str(node["reason_code"]),
            "truth_state": "unavailable",
        }
    count = _measured_count(node)
    if count is None:
        raise PromotionDecisionError("quantity is neither measured nor unavailable")
    return {
        "sensor_id": str(node["sensor_id"]),
        "truth_state": "measured",
        "unit": str(node["unit"]),
        "value": count,
    }


def _cite_audit(receipt: Mapping[str, Any], *, sensor_id: str) -> dict[str, Any]:
    node = receipt.get("audit_overhead")
    if not isinstance(node, Mapping):
        raise PromotionDecisionError("audit_overhead must be an object")
    if _is_unavailable(node):
        if "total_microusd" in node or "value" in node:
            raise PromotionDecisionError("unavailable audit overhead must not carry a numeric value")
        return {
            "reason_code": str(node["reason_code"]),
            "truth_state": "unavailable",
        }
    value = node.get("total_microusd")
    if type(value) is not int:
        raise PromotionDecisionError("measured audit overhead must be an integer")
    return {
        "includes_audit_in_quality_adjusted_cost": bool(
            node["includes_audit_in_quality_adjusted_cost"]
        ),
        "sensor_id": sensor_id,
        "truth_state": "measured",
        "unit": "microusd",
        "value": value,
    }


def _cite_population(
    key: str,
    receipt: Mapping[str, Any],
    *,
    filename: str,
    cohort: str,
    minimum: int,
) -> dict[str, Any]:
    path = f"docs/architecture/agent_supervisor_efficiency_state_hardening_inventory/{filename}"
    commit = str(receipt["repository_commit"])
    tree = str(receipt["repository_tree"])
    if GIT_OID_RE.fullmatch(commit) is None or GIT_OID_RE.fullmatch(tree) is None:
        raise PromotionDecisionError(f"{key} repository commit/tree is unbound")
    identity = str(receipt["identity"])
    results_identity = str(receipt["results_identity"])
    if CID_RE.fullmatch(identity) is None or CID_RE.fullmatch(results_identity) is None:
        raise PromotionDecisionError(f"{key} identity is not a CID")
    reasons = list(receipt.get("reasons") or [])
    unavailable_fields = list(receipt.get("explicit_unavailable_fields") or [])
    return {
        "admitted": bool(receipt["admitted"]),
        "canary_admitted": bool(receipt["canary_admitted"]),
        "cohort": cohort,
        "disposition": str(receipt["disposition"]),
        "explicit_unavailable_fields": unavailable_fields,
        "hermetic_sufficient_for_production_promotion": False,
        "identity": identity,
        "live": bool(receipt["live"]),
        "minimum_tasks": int(receipt["minimum_tasks"]),
        "pair_count": _cite_pair_count(receipt),
        "path": path,
        "production_qualified": bool(receipt["production_qualified"]),
        "qualification": bool(receipt["qualification"]),
        "reasons": reasons,
        "repository_commit": commit,
        "repository_tree": tree,
        "results_identity": results_identity,
        "schema": str(receipt["schema"]),
        "task_id": str(receipt["task_id"]),
    }


def _cite_live_cohort(manifest: Mapping[str, Any]) -> dict[str, Any]:
    identity = str(manifest["identity"])
    if CID_RE.fullmatch(identity) is None:
        raise PromotionDecisionError("live cohort identity is not a CID")
    return {
        "canary_admitted": bool(manifest["canary_admitted"]),
        "count": int(manifest["count"]),
        "disposition": str(manifest["disposition"]),
        "enrollment_deadline": str(manifest["enrollment_deadline"]),
        "fixtures_satisfy_live": False,
        "hermetic_sufficient_for_production_promotion": False,
        "historical_sufficient_for_production_promotion": False,
        "identity": identity,
        "live": bool(manifest["live"]),
        "minimum": 10,
        "path": "benchmarks/agent_supervisor/efficiency_state_hardening/live_cohort_manifest.json",
        "qualification": bool(manifest["qualification"]),
        "reasons": list(manifest["reasons"]),
        "schema": str(manifest["schema"]),
        "shadow_admitted": bool(manifest["shadow_admitted"]),
        "simulation_satisfies_live": False,
        "task_id": "ASEH-015",
    }


def load_current_evidence() -> dict[str, Any]:
    receipts: dict[str, dict[str, Any]] = {}
    for key, filename, _cohort, _minimum, _live in POPULATION_SPECS:
        receipts[key] = _load_json(INVENTORY / filename)
    return {
        "receipts": receipts,
        "live_cohort": _load_json(LIVE_COHORT_PATH),
        "requirements": _load_json(REQUIREMENTS_PATH),
    }


def _hard_gates(requirements: Mapping[str, Any]) -> list[str]:
    gates = requirements["promotion_hard_gates_required_zero"]
    if not isinstance(gates, list) or len(gates) != 14:
        raise PromotionDecisionError("hard gates must be the closed 14-name set")
    return [str(item) for item in gates]


def _thresholds(requirements: Mapping[str, Any]) -> dict[str, Any]:
    targets = requirements["qualification"]["minimum_targets"]
    if not isinstance(targets, Mapping):
        raise PromotionDecisionError("qualification.minimum_targets is absent")
    return {
        "accepted_patch_quality_statistically_meaningful_degradation": bool(
            targets["accepted_patch_quality_statistically_meaningful_degradation"]
        ),
        "eligible_context_pack_reuse_percent": int(targets["eligible_context_pack_reuse_percent"]),
        "field": "qualification.minimum_targets",
        "frontier_model_call_reduction_percent": int(
            targets["frontier_model_call_reduction_percent"]
        ),
        "low_risk_deterministic_or_small_model_share_percent": int(
            targets["low_risk_deterministic_or_small_model_share_percent"]
        ),
        "manual_recovery_rate_strictly_less_than_percent": int(
            targets["manual_recovery_rate_strictly_less_than_percent"]
        ),
        "median_live_input_token_reduction_percent": int(
            targets["median_live_input_token_reduction_percent"]
        ),
        "net_savings_after_audit_and_verification_overhead_positive": bool(
            targets["net_savings_after_audit_and_verification_overhead_positive"]
        ),
        "overall_wall_time_maximum_regression_percent": int(
            targets["overall_wall_time_maximum_regression_percent"]
        ),
        "retry_token_reduction_percent": int(targets["retry_token_reduction_percent"]),
        "source": (
            "docs/architecture/agent_supervisor_efficiency_state_hardening.requirements.json"
        ),
        "total_weighted_provider_cost_reduction_percent": int(
            targets["total_weighted_provider_cost_reduction_percent"]
        ),
    }


def _collect_missing(
    populations: Mapping[str, Mapping[str, Any]],
    live_cohort: Mapping[str, Any],
) -> list[str]:
    missing: list[str] = []
    for key, population in populations.items():
        prefix = f"populations.{key}"
        if _is_unavailable(population["pair_count"]):
            missing.append(f"{prefix}.pair_count")
        for field in population["explicit_unavailable_fields"]:
            missing.append(f"{prefix}.{field}")
    if _live_cohort_unmeasured(live_cohort):
        missing.append("live_cohort")
    missing.extend(
        [
            "efficiency.median_live_input_token_reduction_percent",
            "efficiency.total_weighted_provider_cost_reduction_percent",
            "efficiency.frontier_model_call_reduction_percent",
            "efficiency.retry_token_reduction_percent",
            "efficiency.low_risk_deterministic_or_small_model_share_percent",
            "efficiency.eligible_context_pack_reuse_percent",
            "efficiency.manual_recovery_rate_strictly_less_than_percent",
            "efficiency.overall_wall_time_maximum_regression_percent",
            "efficiency.net_savings_after_audit_and_verification_overhead_positive",
            "audit_overhead.live",
            "hard_gates.live_promotion_corpus",
            "quality.accepted_patch_quality_statistically_meaningful_degradation",
        ]
    )
    unique: list[str] = []
    seen: set[str] = set()
    for item in missing:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _reasons(
    receipts: Mapping[str, Mapping[str, Any]],
    live_cohort: Mapping[str, Any],
    disposition: str,
) -> list[str]:
    reasons: list[str] = []
    if _live_cohort_unmeasured(live_cohort):
        reasons.append("absent_live_cohort")
    if _population_unmeasured(receipts["hermetic"], 60, require_live=False):
        reasons.append("hermetic_insufficient_evidence")
    if _population_unmeasured(receipts["historical"], 20, require_live=False):
        reasons.append("historical_replay_insufficient_evidence")
    if _population_unmeasured(receipts["live_shadow"], 10, require_live=True):
        reasons.append("live_shadow_insufficient_evidence")
    if receipts["canary"].get("disposition") == "not_admitted":
        reasons.append("canary_not_admitted")
    if _is_unavailable(receipts["historical"].get("audit_overhead")):
        reasons.append("historical_audit_overhead_unmeasured")
    if _is_unavailable(receipts["live_shadow"].get("audit_overhead")):
        reasons.append("live_audit_overhead_unmeasured")
    if _is_unavailable(receipts["historical"].get("safety", {}).get("live_cohort")):
        reasons.append("live_safety_unmeasured")
    reasons.append("live_efficiency_unmeasured")
    if _live_cohort_unmeasured(live_cohort) and disposition != "non_promoted_unmeasured":
        raise PromotionDecisionError("absent live cohort must map to non_promoted_unmeasured")
    if _live_cohort_unmeasured(live_cohort) and "absent_live_cohort" not in reasons:
        raise PromotionDecisionError("absent live cohort must be cited in reasons")
    unique: list[str] = []
    seen: set[str] = set()
    for item in reasons:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def build_promotion_decision(evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    bundle = dict(evidence or load_current_evidence())
    receipts = bundle["receipts"]
    live_cohort_receipt = bundle["live_cohort"]
    requirements = bundle["requirements"]
    thresholds = _thresholds(requirements)
    disposition = mandated_disposition(receipts, live_cohort_receipt, thresholds)
    kind = KIND_FOR_DISPOSITION[disposition]
    populations = {}
    for key, filename, cohort, minimum, _require_live in POPULATION_SPECS:
        populations[key] = _cite_population(
            key,
            receipts[key],
            filename=filename,
            cohort=cohort,
            minimum=minimum,
        )
    live_cohort = _cite_live_cohort(live_cohort_receipt)
    safety = {
        "canary": {
            "escaped_critical_seeded_defects": _cite_quantity(
                receipts["canary"]["safety"]["escaped_critical_seeded_defects"]
            ),
            "live_cohort": _cite_quantity(receipts["canary"]["safety"]["live_cohort"]),
            "simulated_as_live_outcomes": _cite_quantity(
                receipts["canary"]["safety"]["simulated_as_live_outcomes"]
            ),
        },
        "hard_gate_violation": any(
            _safety_or_quality_failed(receipts[key])
            for key, *_rest in POPULATION_SPECS
        ),
        "hermetic": {
            "escaped_critical_seeded_defects": _cite_quantity(
                receipts["hermetic"]["safety"]["escaped_critical_seeded_defects"]
            ),
            "live_cohort": _cite_quantity(receipts["hermetic"]["safety"]["live_cohort"]),
            "simulated_as_live_outcomes": _cite_quantity(
                receipts["hermetic"]["safety"]["simulated_as_live_outcomes"]
            ),
        },
        "historical": {
            "escaped_critical_seeded_defects": _cite_quantity(
                receipts["historical"]["safety"]["escaped_critical_seeded_defects"]
            ),
            "live_cohort": _cite_quantity(receipts["historical"]["safety"]["live_cohort"]),
            "simulated_as_live_outcomes": _cite_quantity(
                receipts["historical"]["safety"]["simulated_as_live_outcomes"]
            ),
        },
        "live_cohort_present": not _live_cohort_unmeasured(live_cohort_receipt),
        "live_shadow": {
            "escaped_critical_seeded_defects": _cite_quantity(
                receipts["live_shadow"]["safety"]["escaped_critical_seeded_defects"]
            ),
            "live_cohort": _cite_quantity(receipts["live_shadow"]["safety"]["live_cohort"]),
            "simulated_as_live_outcomes": _cite_quantity(
                receipts["live_shadow"]["safety"]["simulated_as_live_outcomes"]
            ),
        },
        "usable_for_production_promotion": False,
    }
    quality = {
        "accepted_patch_quality_statistically_meaningful_degradation": dict(UNAVAILABLE),
        "canary": copy.deepcopy(receipts["canary"]["quality"]),
        "hermetic": copy.deepcopy(receipts["hermetic"]["quality"]),
        "historical": copy.deepcopy(receipts["historical"]["quality"]),
        "live_shadow": copy.deepcopy(receipts["live_shadow"]["quality"]),
        "usable_for_production_promotion": False,
    }
    efficiency = {
        "eligible_context_pack_reuse_percent": {
            "observed": dict(UNAVAILABLE),
            "required_percent": thresholds["eligible_context_pack_reuse_percent"],
        },
        "frontier_model_call_reduction_percent": {
            "observed": dict(UNAVAILABLE),
            "required_percent": thresholds["frontier_model_call_reduction_percent"],
        },
        "low_risk_deterministic_or_small_model_share_percent": {
            "observed": dict(UNAVAILABLE),
            "required_percent": thresholds["low_risk_deterministic_or_small_model_share_percent"],
        },
        "manual_recovery_rate_strictly_less_than_percent": {
            "observed": dict(UNAVAILABLE),
            "required_percent": thresholds["manual_recovery_rate_strictly_less_than_percent"],
        },
        "median_live_input_token_reduction_percent": {
            "observed": dict(UNAVAILABLE),
            "required_percent": thresholds["median_live_input_token_reduction_percent"],
        },
        "net_savings_after_audit_and_verification_overhead_positive": {
            "observed": dict(UNAVAILABLE),
            "required": thresholds["net_savings_after_audit_and_verification_overhead_positive"],
        },
        "overall_wall_time_maximum_regression_percent": {
            "observed": dict(UNAVAILABLE),
            "required_percent": thresholds["overall_wall_time_maximum_regression_percent"],
        },
        "retry_token_reduction_percent": {
            "observed": dict(UNAVAILABLE),
            "required_percent": thresholds["retry_token_reduction_percent"],
        },
        "thresholds_evaluated": False,
        "total_weighted_provider_cost_reduction_percent": {
            "observed": dict(UNAVAILABLE),
            "required_percent": thresholds["total_weighted_provider_cost_reduction_percent"],
        },
        "usable_for_production_promotion": False,
    }
    payload = {
        "audit_overhead": {
            "canary": _cite_audit(receipts["canary"], sensor_id="aseh-073-canary-qualification"),
            "hermetic": _cite_audit(
                receipts["hermetic"], sensor_id="aseh-070-hermetic-qualification"
            ),
            "historical": _cite_audit(
                receipts["historical"], sensor_id="aseh-071-historical-qualification"
            ),
            "included_in_efficiency_evaluation": True,
            "live": dict(UNAVAILABLE),
            "live_shadow": _cite_audit(
                receipts["live_shadow"], sensor_id="aseh-072-live-shadow-qualification"
            ),
            "usable_for_production_promotion": False,
        },
        "authority": False,
        "cas_protected_promotion": True,
        "closed_dispositions": list(CLOSED_DISPOSITIONS),
        "disposition": disposition,
        "disposition_kind": kind,
        "efficiency": efficiency,
        "eligible_for_operator_authorization": disposition
        == "promotion_eligible_operator_authorization_required",
        "evaluation_order": list(DISPOSITION_KINDS),
        "evidence_paths": list(EVIDENCE_PATHS),
        "explicit_unavailable_fields": _collect_missing(populations, live_cohort),
        "hard_gates": {
            "live_promotion_corpus": dict(UNAVAILABLE),
            "required_zero": _hard_gates(requirements),
            "usable_for_production_promotion": False,
        },
        "hermetic_sufficient_for_production_promotion": False,
        "historical_sufficient_for_production_promotion": False,
        "interface": INTERFACE,
        "live_cohort": live_cohort,
        "missing_evidence": _collect_missing(populations, live_cohort),
        "missing_measurement_recorded_as_zero": False,
        "nonclaims": list(NONCLAIMS),
        "objective_id": OBJECTIVE_ID,
        "plan_revision": PLAN_REVISION,
        "policy_identity": POLICY_IDENTITY,
        "policy_pointer_mutated": False,
        "populations": populations,
        "production_qualified": False,
        "program_id": PROGRAM_ID,
        "promotion_authorized": False,
        "quality": quality,
        "reasons": _reasons(receipts, live_cohort_receipt, disposition),
        "safety": safety,
        "schema": SCHEMA_ID,
        "schema_version": 1,
        "self_authorized": False,
        "task_id": TASK_ID,
        "thresholds": thresholds,
        "thresholds_lowered": False,
        "validator_command": list(VALIDATOR_COMMAND),
    }
    payload["identity"] = content_identity(
        {key: value for key, value in payload.items() if key != "identity"}
    )
    return payload


def _iter_nodes(payload: Any, path: str = "$"):
    yield path, payload
    if isinstance(payload, Mapping):
        for key, child in payload.items():
            yield from _iter_nodes(child, f"{path}.{key}")
    elif isinstance(payload, list):
        for index, child in enumerate(payload):
            yield from _iter_nodes(child, f"{path}[{index}]")


def validate_decision(payload: Mapping[str, Any], *, evidence: Mapping[str, Any] | None = None) -> None:
    schema = _load_json(SCHEMA_PATH)
    validate_against_schema(payload, schema, path="$")
    bundle = dict(evidence or load_current_evidence())
    expected = build_promotion_decision(bundle)
    live_unmeasured = _live_cohort_unmeasured(bundle["live_cohort"])
    if live_unmeasured and payload["disposition"] != "non_promoted_unmeasured":
        raise PromotionDecisionError("absent live cohort must map to non_promoted_unmeasured")
    if payload["safety"]["hard_gate_violation"] and payload["disposition"] not in {
        "non_promoted_unmeasured",
        "non_promoted_safety_or_quality",
    }:
        raise PromotionDecisionError("safety violation was ignored")
    if payload["disposition"] not in CLOSED_DISPOSITIONS:
        raise PromotionDecisionError("disposition is outside the closed vocabulary")
    if payload["disposition_kind"] != KIND_FOR_DISPOSITION[payload["disposition"]]:
        raise PromotionDecisionError("disposition_kind does not match disposition")
    if payload["evaluation_order"] != list(DISPOSITION_KINDS):
        raise PromotionDecisionError("evaluation_order must stay fail-closed")
    if payload["thresholds"] != expected["thresholds"]:
        raise PromotionDecisionError("thresholds were lowered or unbound")
    if payload["policy_identity"] != POLICY_IDENTITY:
        raise PromotionDecisionError("policy identity is unbound")
    if payload["self_authorized"] is not False or payload["promotion_authorized"] is not False:
        raise PromotionDecisionError("decision cannot self-authorize promotion")
    if payload["policy_pointer_mutated"] is not False:
        raise PromotionDecisionError("decision cannot mutate a policy pointer")
    if payload["hermetic_sufficient_for_production_promotion"] is not False:
        raise PromotionDecisionError("hermetic evidence cannot satisfy production promotion")
    if payload["missing_measurement_recorded_as_zero"] is not False:
        raise PromotionDecisionError("missing measurements cannot be recorded as zero")
    for _path, node in _iter_nodes(payload):
        if isinstance(node, Mapping) and node.get("truth_state") == "unavailable":
            if "value" in node or "count" in node or "total_microusd" in node:
                raise PromotionDecisionError("unavailable evidence carries a numeric value")
            if node.get("reason_code") in {None, ""}:
                raise PromotionDecisionError("unavailable evidence is missing a reason_code")
    hermetic = payload["populations"]["hermetic"]
    if hermetic["live"] is not False:
        raise PromotionDecisionError("hermetic evidence was labeled live")
    if hermetic["pair_count"] != bundle["receipts"]["hermetic"]["pair_count"]:
        raise PromotionDecisionError("hermetic pair_count does not cite current evidence")
    if payload["live_cohort"]["identity"] != bundle["live_cohort"]["identity"]:
        raise PromotionDecisionError("live cohort identity does not cite current evidence")
    if payload["disposition"] != expected["disposition"]:
        raise PromotionDecisionError("disposition does not match the mandated current evaluation")
    if payload.get("identity") != expected["identity"]:
        raise PromotionDecisionError("identity does not bind the current decision body")


def write_decision_artifact() -> dict[str, Any]:
    payload = build_promotion_decision()
    DECISION_PATH.parent.mkdir(parents=True, exist_ok=True)
    DECISION_PATH.write_text(_canonical(payload), encoding="utf-8")
    return payload


write_decision_artifact()


def test_schema_is_closed_and_binds_the_four_dispositions() -> None:
    schema = _load_json(SCHEMA_PATH)
    assert schema["$id"] == SCHEMA_ID
    assert _walk_object_schemas(schema) == []
    assert schema["properties"]["disposition"]["$ref"] == "#/$defs/disposition"
    assert schema["$defs"]["disposition"]["enum"] == list(CLOSED_DISPOSITIONS)
    assert schema["$defs"]["dispositionKind"]["enum"] == list(DISPOSITION_KINDS)
    assert schema["properties"]["evaluation_order"]["minItems"] == 4
    assert schema["properties"]["self_authorized"]["const"] is False
    assert schema["properties"]["policy_pointer_mutated"]["const"] is False
    assert schema["properties"]["promotion_authorized"]["const"] is False
    assert schema["properties"]["hermetic_sufficient_for_production_promotion"]["const"] is False
    assert schema["properties"]["missing_measurement_recorded_as_zero"]["const"] is False
    assert schema["$defs"]["thresholds"]["properties"]["median_live_input_token_reduction_percent"][
        "const"
    ] == 30


def test_current_decision_cites_exact_evidence_and_returns_unmeasured() -> None:
    evidence = load_current_evidence()
    expected = build_promotion_decision(evidence)
    actual = _load_canonical(DECISION_PATH)
    validate_against_schema(actual, _load_json(SCHEMA_PATH), path="$")
    validate_decision(actual, evidence=evidence)
    assert actual == expected
    assert actual["disposition"] == "non_promoted_unmeasured"
    assert actual["disposition_kind"] == "unmeasured"
    assert actual["eligible_for_operator_authorization"] is False
    assert "absent_live_cohort" in actual["reasons"]
    hermetic = evidence["receipts"]["hermetic"]
    cited = actual["populations"]["hermetic"]
    assert cited["identity"] == hermetic["identity"]
    assert cited["results_identity"] == hermetic["results_identity"]
    assert cited["pair_count"] == hermetic["pair_count"]
    assert cited["repository_commit"] == hermetic["repository_commit"]
    assert cited["repository_tree"] == hermetic["repository_tree"]
    assert cited["live"] is False
    historical = evidence["receipts"]["historical"]
    assert actual["populations"]["historical"]["identity"] == historical["identity"]
    assert actual["populations"]["historical"]["pair_count"] == historical["pair_count"]
    assert historical["pair_count"]["truth_state"] == "unavailable"
    live_shadow = evidence["receipts"]["live_shadow"]
    assert actual["populations"]["live_shadow"]["identity"] == live_shadow["identity"]
    assert live_shadow["live"] is False
    assert actual["live_cohort"]["identity"] == evidence["live_cohort"]["identity"]
    assert actual["live_cohort"]["count"] == evidence["live_cohort"]["count"]
    assert actual["live_cohort"]["live"] is False
    assert actual["thresholds"]["median_live_input_token_reduction_percent"] == 30
    assert actual["hard_gates"]["required_zero"] == evidence["requirements"][
        "promotion_hard_gates_required_zero"
    ]
    assert CID_RE.fullmatch(actual["identity"])
    assert actual["identity"] == content_identity(
        {key: value for key, value in actual.items() if key != "identity"}
    )


def test_closed_evaluator_emits_only_the_four_mandated_dispositions() -> None:
    evidence = load_current_evidence()
    current = mandated_disposition(
        evidence["receipts"],
        evidence["live_cohort"],
        evidence["requirements"]["qualification"]["minimum_targets"],
    )
    assert current == "non_promoted_unmeasured"

    qualified = copy.deepcopy(evidence["receipts"]["hermetic"])
    qualified["live"] = True
    qualified["qualification"] = True
    qualified["disposition"] = "evidence_qualified"
    qualified["pair_count"] = {
        "sensor_id": "synthetic-live",
        "truth_state": "measured",
        "unit": "count",
        "value": 12,
    }
    qualified["quality"] = {
        "escaped_selected_test_false_negatives": [],
        "observed_selected_test_false_negatives": [],
    }
    live_cohort = copy.deepcopy(evidence["live_cohort"])
    live_cohort.update(
        {
            "canary_admitted": True,
            "count": 12,
            "disposition": "evidence_qualified",
            "live": True,
            "qualification": True,
            "shadow_admitted": True,
        }
    )
    thresholds = evidence["requirements"]["qualification"]["minimum_targets"]
    live_efficiency = {
        name: {
            "sensor_id": "synthetic-live",
            "truth_state": "measured",
            "unit": "percent",
            "value": int(thresholds[name]),
        }
        for name in EFFICIENCY_PERCENT_FIELDS
    }
    live_efficiency["manual_recovery_rate_strictly_less_than_percent"] = {
        "sensor_id": "synthetic-live",
        "truth_state": "measured",
        "unit": "percent",
        "value": max(
            0,
            int(thresholds["manual_recovery_rate_strictly_less_than_percent"]) - 1,
        ),
    }
    live_efficiency["net_savings_after_audit_and_verification_overhead_positive"] = {
        "sensor_id": "synthetic-live",
        "truth_state": "measured",
        "unit": "count",
        "value": 1,
    }
    receipts = {
        "hermetic": copy.deepcopy(evidence["receipts"]["hermetic"]),
        "historical": copy.deepcopy(qualified),
        "live_shadow": copy.deepcopy(qualified),
        "canary": copy.deepcopy(qualified),
    }
    receipts["historical"]["live"] = False
    receipts["historical"]["pair_count"] = {
        "sensor_id": "synthetic-historical",
        "truth_state": "measured",
        "unit": "count",
        "value": 20,
    }
    assert (
        mandated_disposition(receipts, live_cohort, thresholds, live_efficiency)
        == "promotion_eligible_operator_authorization_required"
    )

    unsafe = copy.deepcopy(receipts)
    unsafe["live_shadow"]["safety"]["escaped_critical_seeded_defects"] = {
        "sensor_id": "synthetic-live",
        "truth_state": "measured",
        "unit": "count",
        "value": 1,
    }
    assert (
        mandated_disposition(unsafe, live_cohort, thresholds, live_efficiency)
        == "non_promoted_safety_or_quality"
    )

    cheap = copy.deepcopy(live_efficiency)
    cheap["median_live_input_token_reduction_percent"] = {
        "sensor_id": "synthetic-live",
        "truth_state": "measured",
        "unit": "percent",
        "value": 10,
    }
    assert (
        mandated_disposition(receipts, live_cohort, thresholds, cheap)
        == "non_promoted_efficiency"
    )

    missing_metric = copy.deepcopy(live_efficiency)
    missing_metric["median_live_input_token_reduction_percent"] = dict(UNAVAILABLE)
    assert (
        mandated_disposition(receipts, live_cohort, thresholds, missing_metric)
        == "non_promoted_unmeasured"
    )


def test_negative_cases_fail_closed() -> None:
    evidence = load_current_evidence()
    payload = build_promotion_decision(evidence)
    schema = _load_json(SCHEMA_PATH)

    unknown = copy.deepcopy(payload)
    unknown["invented"] = True
    with pytest.raises(PromotionDecisionError, match="unknown fields"):
        validate_against_schema(unknown, schema, path="$")

    promote = copy.deepcopy(payload)
    promote["disposition"] = "promote"
    with pytest.raises(PromotionDecisionError, match="enum"):
        validate_against_schema(promote, schema, path="$")

    self_auth = copy.deepcopy(payload)
    self_auth["self_authorized"] = True
    with pytest.raises(PromotionDecisionError, match="closed const"):
        validate_against_schema(self_auth, schema, path="$")

    lowered = copy.deepcopy(payload)
    lowered["thresholds"]["median_live_input_token_reduction_percent"] = 1
    with pytest.raises(PromotionDecisionError, match="closed const"):
        validate_against_schema(lowered, schema, path="$")

    eligible = copy.deepcopy(payload)
    eligible["disposition"] = "promotion_eligible_operator_authorization_required"
    eligible["disposition_kind"] = "eligible_operator_authorization_required"
    eligible["eligible_for_operator_authorization"] = True
    eligible["identity"] = content_identity(
        {key: value for key, value in eligible.items() if key != "identity"}
    )
    with pytest.raises(PromotionDecisionError, match="absent live cohort"):
        validate_decision(eligible, evidence=evidence)

    zeroed = copy.deepcopy(payload)
    zeroed["populations"]["historical"]["pair_count"] = {
        "reason_code": "not_yet_measured",
        "truth_state": "unavailable",
        "value": 0,
    }
    with pytest.raises(PromotionDecisionError, match="uniquely match|numeric value"):
        try:
            validate_against_schema(zeroed, schema, path="$")
        except PromotionDecisionError:
            raise
        validate_decision(zeroed, evidence=evidence)

    ignored_safety = copy.deepcopy(payload)
    ignored_safety["safety"]["hard_gate_violation"] = True
    ignored_safety["disposition"] = "promotion_eligible_operator_authorization_required"
    ignored_safety["disposition_kind"] = "eligible_operator_authorization_required"
    ignored_safety["identity"] = content_identity(
        {key: value for key, value in ignored_safety.items() if key != "identity"}
    )
    with pytest.raises(PromotionDecisionError, match="absent live cohort|safety violation"):
        validate_decision(ignored_safety, evidence=evidence)

    unbound = copy.deepcopy(payload)
    unbound["policy_identity"] = "unbound"
    with pytest.raises(PromotionDecisionError, match="closed const"):
        validate_against_schema(unbound, schema, path="$")

    as_zero = copy.deepcopy(evidence)
    as_zero["receipts"]["historical"]["pair_count"] = 0
    with pytest.raises(PromotionDecisionError, match="typed observation"):
        build_promotion_decision(as_zero)


def test_module_installs_without_sibling_test_imports() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    sibling_stems = {
        path.stem
        for path in Path(__file__).parent.glob("test_*.py")
        if path.name != Path(__file__).name
    }
    for name in imported:
        assert not name.startswith("test."), name
        assert name.rsplit(".", 1)[-1] not in sibling_stems
        for prefix in SIBLING_TEST_PREFIXES:
            assert not name.startswith(prefix), name
