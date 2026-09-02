#!/usr/bin/env python3
"""ASEH-013 paired benchmark harness for sealed hermetic fixtures.

The harness expands compact recipe vectors into the three required arms,
rejects unequal controls, and computes integer paired statistics. Hermetic
evidence cannot be labeled live and cannot satisfy production promotion.
Audit and verification overhead is included in quality-adjusted cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ipfs_accelerate_py.agent_supervisor.runtime.efficiency_receipts import (  # noqa: E402
    EQUAL_CONTROL_FIELDS,
    PAIRED_ARMS,
    PROGRAM_ID,
    STATISTIC_FIELDS,
    admit_paired_benchmark_manifest,
    build_paired_benchmark_manifest,
    canonical_bytes,
    content_identity,
    estimated_quantity,
    measured_quantity,
    observed_identity,
    unavailable,
)

PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent
HERMETIC_VECTORS_PATH: Final[Path] = PACKAGE_DIR / "hermetic_vectors.jsonl"
HERMETIC_MANIFEST_PATH: Final[Path] = PACKAGE_DIR / "hermetic_manifest.json"
PAIRED_MANIFEST_PATH: Final[Path] = PACKAGE_DIR / "manifest.json"
PROVIDER_PATH: Final[Path] = PACKAGE_DIR / "provider_model_config.json"
PRICE_PATH: Final[Path] = PACKAGE_DIR / "price_snapshot.json"
ENVIRONMENT_PATH: Final[Path] = PACKAGE_DIR / "environment_identity.json"

HERMETIC_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/aseh-hermetic-fixture-manifest@1"
)
HUMAN_POLICY_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/aseh-human-intervention-policy@1"
)
ACCEPTANCE_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/aseh-hermetic-acceptance@1"
)
RESOURCE_LIMITS_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/aseh-equal-control-resource-limits@1"
)
OBJECTIVE_ID: Final[str] = "ASEH-G020"
TASK_ID: Final[str] = "ASEH-013"
POLICY_IDENTITY: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/aseh-supervisor-policy@1"
)
REPOSITORY_COMMIT: Final[str] = "755f45475cc2d13dacd8b330036c1d597afeddde"
REPOSITORY_TREE: Final[str] = "729da9f8293ecfa046a0136381a3d3808f9ed140"
SENSOR_ID: Final[str] = "aseh-013-paired-harness"
ESTIMATOR_ID: Final[str] = "aseh-013-paired-harness"
HERMETIC_MINIMUM: Final[int] = 60
HERMETIC_COUNT: Final[int] = 64
SEALED_SEED: Final[int] = 20260823
BOOTSTRAP_RESAMPLES: Final[int] = 512
MILLIONTHS: Final[int] = 1_000_000
MAX_INTEGER: Final[int] = 10**18
MAXIMUM_RETRIES: Final[int] = 4

DIRECT_ARM: Final[str] = "direct_minimal_orchestration_baseline"
SEALED_ARM: Final[str] = "sealed_current_supervisor_baseline"
CANDIDATE_ARM: Final[str] = "candidate_optimized_supervisor"

TASK_CLASSES: Final[tuple[str, ...]] = (
    "routing",
    "context_pack",
    "state_machine",
    "planning",
    "synthesis",
    "selected_test",
    "safety_seed",
    "retry_rescue",
)

OUTCOME_TABLE: Final[tuple[tuple[str, str], ...]] = (
    ("succeeded", "accepted"),
    ("succeeded", "accepted"),
    ("succeeded", "accepted"),
    ("succeeded", "accepted"),
    ("succeeded", "accepted"),
    ("failed", "rejected"),
    ("retried", "rejected"),
    ("rescued", "accepted"),
    ("conflicted", "quarantined"),
    ("human_escalated", "quarantined"),
    ("quarantined", "quarantined"),
    ("succeeded", "reverted"),
    ("succeeded", "accepted"),
    ("failed", "rejected"),
    ("succeeded", "accepted"),
    ("succeeded", "accepted"),
)

ARM_COST_SCALE: Final[dict[str, int]] = {
    DIRECT_ARM: 100,
    SEALED_ARM: 84,
    CANDIDATE_ARM: 57,
}
ARM_AUDIT_SCALE: Final[dict[str, int]] = {
    DIRECT_ARM: 36,
    SEALED_ARM: 48,
    CANDIDATE_ARM: 64,
}
ARM_TIME_SCALE: Final[dict[str, int]] = {
    DIRECT_ARM: 100,
    SEALED_ARM: 88,
    CANDIDATE_ARM: 61,
}

RESOURCE_LIMITS: Final[dict[str, Any]] = {
    "schema": RESOURCE_LIMITS_SCHEMA,
    "implementation_log_stall_seconds": 1200,
    "implementation_max_timeout_seconds": 21600,
    "implementation_retry_budget": 4,
    "implementation_timeout_seconds": 14400,
    "max_concurrency": 4,
    "max_lanes": 4,
    "max_restarts": 8,
    "max_task_attempts": 4,
    "merge_retry_budget": 4,
    "poll_interval_seconds": 5,
    "stale_seconds": 1800,
    "validation_retry_budget": 4,
    "maximum_retries": MAXIMUM_RETRIES,
}

HUMAN_INTERVENTION_POLICY: Final[dict[str, Any]] = {
    "schema": HUMAN_POLICY_SCHEMA,
    "routine_choice_approval": "not_required",
    "escalation_only_for": [
        "irreducible_value_laden_decision",
        "credential_bearing_decision",
        "destructive_decision",
        "promotion_authority_decision",
    ],
    "human_may_mutate_policy_pointer": False,
    "maximum_retries_before_human": MAXIMUM_RETRIES,
}

ACCEPTANCE_CONTRACT: Final[dict[str, Any]] = {
    "schema": ACCEPTANCE_SCHEMA,
    "command": [
        "python3",
        "-m",
        "pytest",
        "-q",
        "test/api/agent_supervisor/efficiency_state_hardening/test_paired_harness.py",
    ],
    "same_safe_isolation_and_acceptance_tests": True,
    "no_unsafe_direct_mutation": True,
}

OBJECTIVE_REVISION: Final[str] = content_identity(
    {
        "objective_id": OBJECTIVE_ID,
        "plan_revision": "ASEH-PLAN-R1",
        "task_id": TASK_ID,
    }
)


class PairedHarnessError(ValueError):
    """Fail-closed paired-harness contract violation."""


class UnequalControlsError(PairedHarnessError):
    """Raised when paired arms do not share identical controls."""


def _pretty(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _compact(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PairedHarnessError(f"{path.name} is unreadable") from exc
    if not isinstance(payload, dict):
        raise PairedHarnessError(f"{path.name} must contain a JSON object")
    return payload


def _companion_identity(path: Path) -> str:
    payload = _load_json(path)
    claimed = payload.get("identity")
    if isinstance(claimed, str) and claimed.startswith("b") and len(claimed) >= 21:
        return claimed
    body = {key: value for key, value in payload.items() if key != "identity"}
    return content_identity(body)


def _bounded_int(value: int, *, name: str) -> int:
    if type(value) is not int:
        raise PairedHarnessError(f"{name} must be an integer")
    if value < 0 or value > MAX_INTEGER:
        raise PairedHarnessError(f"{name} is outside the integer bound")
    return value


def _median(values: Sequence[int]) -> int:
    if not values:
        raise PairedHarnessError("median is undefined for an empty sample")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) // 2


def _mean(values: Sequence[int]) -> int:
    if not values:
        raise PairedHarnessError("mean is undefined for an empty sample")
    return sum(values) // len(values)


def _ratio_millionths(numerator: int, denominator: int) -> int | None:
    if denominator == 0:
        return None
    return _bounded_int((numerator * MILLIONTHS) // denominator, name="ratio")


def _percentile_nearest(ordered: Sequence[int], millionths: int) -> int:
    if not ordered:
        raise PairedHarnessError("percentile is undefined for an empty sample")
    if millionths < 0 or millionths > MILLIONTHS:
        raise PairedHarnessError("percentile millionths is out of range")
    index = (millionths * (len(ordered) - 1)) // MILLIONTHS
    return ordered[index]


def _rng(seed: int, *parts: str) -> random.Random:
    material = canonical_bytes({"parts": list(parts), "seed": seed})
    digest = hashlib.sha256(material).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _mix(seed: int, arm_id: str, salt: str) -> int:
    material = f"{seed}:{arm_id}:{salt}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def bootstrap_interval(
    values: Sequence[int],
    *,
    seed: int,
    domain: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, int]:
    if not values:
        raise PairedHarnessError("bootstrap is undefined for an empty sample")
    rng = _rng(seed, "bootstrap", domain)
    n = len(values)
    stats: list[int] = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(_median(sample))
    stats.sort()
    low = _percentile_nearest(stats, 25_000)
    high = _percentile_nearest(stats, 975_000)
    if high < low:
        raise PairedHarnessError("bootstrap interval inverted")
    return {
        "low": low,
        "high": high,
        "width": high - low,
        "median": _median(stats),
        "resamples": resamples,
    }


def tukey_outliers(values: Sequence[int], *, labels: Sequence[str]) -> dict[str, Any]:
    if len(values) != len(labels):
        raise PairedHarnessError("outlier labels must align with values")
    if len(values) < 4:
        return {"count": 0, "fixture_ids": [], "fence_low": 0, "fence_high": 0}
    ordered = sorted(values)
    q1 = _percentile_nearest(ordered, 250_000)
    q3 = _percentile_nearest(ordered, 750_000)
    iqr = q3 - q1
    fence = (3 * iqr) // 2
    low = q1 - fence
    high = q3 + fence
    flagged = [
        labels[index]
        for index, value in enumerate(values)
        if value < low or value > high
    ]
    return {
        "count": len(flagged),
        "fixture_ids": flagged,
        "fence_low": low,
        "fence_high": high,
        "q1": q1,
        "q3": q3,
    }


def generate_hermetic_recipes(count: int = HERMETIC_COUNT) -> list[dict[str, Any]]:
    if count < HERMETIC_MINIMUM:
        raise PairedHarnessError("hermetic corpus is below the sealed minimum")
    recipes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index in range(count):
        fixture_id = f"ASEH-H-{index + 1:03d}"
        if fixture_id in seen:
            raise PairedHarnessError("hermetic fixture identities must be unique")
        seen.add(fixture_id)
        outcome, disposition = OUTCOME_TABLE[index % len(OUTCOME_TABLE)]
        recipes.append(
            {
                "bounded": True,
                "fixture_id": fixture_id,
                "live": False,
                "outcome": outcome,
                "patch_disposition": disposition,
                "seed": SEALED_SEED * 1000 + index + 1,
                "task_class": TASK_CLASSES[index % len(TASK_CLASSES)],
            }
        )
    return recipes


def serialize_vectors(recipes: Sequence[Mapping[str, Any]]) -> str:
    lines = [_compact(recipe) for recipe in recipes]
    return "\n".join(lines) + "\n"


def load_hermetic_recipes(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or HERMETIC_VECTORS_PATH
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise PairedHarnessError("hermetic vectors are unreadable") from exc
    recipes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PairedHarnessError(
                f"hermetic vector line {line_number} is malformed"
            ) from exc
        if not isinstance(payload, dict):
            raise PairedHarnessError(f"hermetic vector line {line_number} must be an object")
        fixture_id = payload.get("fixture_id")
        if not isinstance(fixture_id, str) or not fixture_id:
            raise PairedHarnessError(f"hermetic vector line {line_number} lacks fixture_id")
        if fixture_id in seen:
            raise PairedHarnessError(f"duplicate hermetic fixture_id {fixture_id}")
        seen.add(fixture_id)
        recipes.append(payload)
    if len(recipes) < HERMETIC_MINIMUM:
        raise PairedHarnessError("hermetic vectors are below the 60-fixture minimum")
    return recipes


def vectors_identity(recipes: Sequence[Mapping[str, Any]]) -> str:
    return content_identity({"recipes": [dict(item) for item in recipes], "seed": SEALED_SEED})


def sealed_equal_controls(*, task_inputs: str) -> dict[str, Any]:
    controls = {
        "repository_revision": REPOSITORY_TREE,
        "objective": OBJECTIVE_ID,
        "task_inputs": task_inputs,
        "acceptance_tests": content_identity(ACCEPTANCE_CONTRACT),
        "available_providers_and_models": _companion_identity(PROVIDER_PATH),
        "price_accounting": _companion_identity(PRICE_PATH),
        "resource_limits": content_identity(RESOURCE_LIMITS),
        "maximum_retries": MAXIMUM_RETRIES,
        "human_intervention_policy": content_identity(HUMAN_INTERVENTION_POLICY),
    }
    if tuple(controls) != EQUAL_CONTROL_FIELDS:
        raise PairedHarnessError("equal control fields drifted from the closed contract")
    return controls


def controls_identity(controls: Mapping[str, Any]) -> str:
    return content_identity(dict(controls))


def _quality_for(disposition: str) -> int:
    if disposition == "accepted":
        return MILLIONTHS
    if disposition == "reverted":
        return 400_000
    if disposition == "quarantined":
        return 250_000
    return 0


def _arm_scales(arm_id: str, *, index: int) -> tuple[int, int, int]:
    cost = ARM_COST_SCALE[arm_id]
    audit = ARM_AUDIT_SCALE[arm_id]
    time_scale = ARM_TIME_SCALE[arm_id]
    if arm_id == CANDIDATE_ARM and index % 16 == 15:
        cost = 148
        audit = 92
        time_scale = 133
    return cost, audit, time_scale


def expand_recipe(
    recipe: Mapping[str, Any],
    arm_id: str,
    *,
    controls: Mapping[str, Any],
    index: int,
) -> dict[str, Any]:
    if arm_id not in PAIRED_ARMS:
        raise PairedHarnessError(f"unknown paired arm {arm_id}")
    if recipe.get("live") is True:
        raise PairedHarnessError("pairing rejects fabricated live labels")
    if recipe.get("bounded") is not True:
        raise PairedHarnessError("hermetic fixtures must be bounded")
    seed = recipe.get("seed")
    if type(seed) is not int:
        raise PairedHarnessError("pairing rejects unseeded nondeterminism")
    fixture_id = recipe.get("fixture_id")
    task_class = recipe.get("task_class")
    outcome = recipe.get("outcome")
    disposition = recipe.get("patch_disposition")
    if not isinstance(fixture_id, str) or not isinstance(task_class, str):
        raise PairedHarnessError("hermetic recipe is missing identity fields")
    if not isinstance(outcome, str) or not isinstance(disposition, str):
        raise PairedHarnessError("hermetic recipe is missing outcome fields")
    if task_class not in TASK_CLASSES:
        raise PairedHarnessError(f"unknown hermetic task class {task_class}")

    cost_scale, audit_scale, time_scale = _arm_scales(arm_id, index=index)
    base_tokens = 90 + (seed % 50)
    input_tokens = _bounded_int(
        (base_tokens * cost_scale) // 100 + (_mix(seed, arm_id, "in") % 9),
        name="input_tokens",
    )
    output_tokens = _bounded_int(
        ((40 + (seed % 20)) * cost_scale) // 100 + (_mix(seed, arm_id, "out") % 5),
        name="output_tokens",
    )
    cached_input_tokens = _bounded_int(
        ((8 + (seed % 12)) * cost_scale) // 100,
        name="cached_input_tokens",
    )
    reasoning_tokens = _bounded_int(
        ((seed % 7) * cost_scale) // 100,
        name="reasoning_tokens",
    )
    number_of_calls = 1 if disposition != "quarantined" else 2
    wall_clock = _bounded_int(
        ((12_000 + (seed % 8_000)) * time_scale) // 100,
        name="wall_clock_duration",
    )
    token_charge = _bounded_int(
        (input_tokens * 3) + (output_tokens * 9) + (reasoning_tokens * 12),
        name="token_based_estimated_charge",
    )
    audit = _bounded_int(
        ((180 + (seed % 90)) * audit_scale) // 100,
        name="audit_and_verification_overhead",
    )
    quality = _quality_for(disposition)
    gross = token_charge + audit
    if quality <= 0:
        quality_adjusted: int | None = None
    else:
        quality_adjusted = _bounded_int(
            (gross * MILLIONTHS) // quality,
            name="quality_adjusted_cost",
        )
    accepted = disposition == "accepted"
    return {
        "arm_id": arm_id,
        "audit_and_verification_overhead": audit,
        "accepted": accepted,
        "accepted_patch_quality": quality,
        "accepted_patch_rate": MILLIONTHS if accepted else 0,
        "bounded": True,
        "cached_input_tokens": cached_input_tokens,
        "controls": dict(controls),
        "controls_identity": controls_identity(controls),
        "fixture_id": fixture_id,
        "input_tokens": input_tokens,
        "live": False,
        "number_of_calls": number_of_calls,
        "outcome": outcome,
        "output_tokens": output_tokens,
        "patch_disposition": disposition,
        "quality_adjusted_cost": quality_adjusted,
        "reasoning_tokens": reasoning_tokens,
        "seed": seed,
        "task_class": task_class,
        "time_to_terminal_outcome": wall_clock,
        "token_based_estimated_charge": token_charge,
        "wall_clock_duration": wall_clock,
    }


def _require_audit(observation: Mapping[str, Any]) -> int:
    value = observation.get("audit_and_verification_overhead")
    if type(value) is not int:
        raise PairedHarnessError("pairing rejects incomplete audit cost")
    return _bounded_int(value, name="audit_and_verification_overhead")


def assert_equal_controls(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(observations) != 3:
        raise PairedHarnessError("pairing requires exactly three arm observations")
    identities = []
    controls_seen: list[dict[str, Any]] = []
    arms: list[str] = []
    fixtures: list[str] = []
    for item in observations:
        if item.get("live") is True:
            raise PairedHarnessError("pairing rejects fabricated live labels")
        if "seed" not in item or type(item.get("seed")) is not int:
            raise PairedHarnessError("pairing rejects unseeded nondeterminism")
        _require_audit(item)
        controls = item.get("controls")
        if not isinstance(controls, Mapping):
            raise PairedHarnessError("arm observation is missing equal controls")
        missing = [name for name in EQUAL_CONTROL_FIELDS if name not in controls]
        if missing:
            raise UnequalControlsError(f"pairing rejects missing controls: {missing}")
        identities.append(item.get("controls_identity") or controls_identity(controls))
        controls_seen.append(dict(controls))
        arms.append(str(item.get("arm_id")))
        fixtures.append(str(item.get("fixture_id")))
    if len(set(fixtures)) != 1:
        raise PairedHarnessError("pairing rejects missing paired inputs")
    if tuple(sorted(arms)) != tuple(sorted(PAIRED_ARMS)):
        raise PairedHarnessError("pairing rejects missing paired inputs")
    if len(set(identities)) != 1:
        raise UnequalControlsError("pairing rejects unequal controls")
    reference = controls_seen[0]
    for candidate in controls_seen[1:]:
        if candidate != reference:
            raise UnequalControlsError("pairing rejects unequal controls")
    return reference


def expand_all_arms(
    recipe: Mapping[str, Any],
    *,
    controls: Mapping[str, Any],
    index: int,
) -> dict[str, dict[str, Any]]:
    observations = {
        arm_id: expand_recipe(recipe, arm_id, controls=controls, index=index)
        for arm_id in PAIRED_ARMS
    }
    assert_equal_controls(tuple(observations[arm_id] for arm_id in PAIRED_ARMS))
    return observations


def _available(values: Iterable[int | None]) -> list[int]:
    available = [value for value in values if type(value) is int]
    return available


def _arm_metric(observations: Sequence[Mapping[str, Any]], field: str) -> list[int]:
    return _available(item.get(field) for item in observations)


def compute_paired_statistics(
    pairs: Sequence[Mapping[str, Mapping[str, Any]]],
    *,
    baseline_arm: str = SEALED_ARM,
    candidate_arm: str = CANDIDATE_ARM,
    seed: int = SEALED_SEED,
) -> dict[str, Any]:
    if not pairs:
        raise PairedHarnessError("pairing rejects missing paired inputs")
    differences: list[int] = []
    ratios: list[int] = []
    labels: list[str] = []
    classes: list[str] = []
    candidate_qac: list[int] = []
    baseline_qac: list[int] = []
    candidate_accept: list[int] = []
    baseline_accept: list[int] = []
    candidate_time: list[int] = []
    baseline_time: list[int] = []
    candidate_audit: list[int] = []
    baseline_audit: list[int] = []
    direct_qac: list[int] = []

    for pair in pairs:
        missing = [arm for arm in PAIRED_ARMS if arm not in pair]
        if missing:
            raise PairedHarnessError("pairing rejects missing paired inputs")
        assert_equal_controls(tuple(pair[arm] for arm in PAIRED_ARMS))
        baseline = pair[baseline_arm]
        candidate = pair[candidate_arm]
        direct = pair[DIRECT_ARM]
        fixture_id = str(candidate["fixture_id"])
        classes.append(str(candidate["task_class"]))
        candidate_accept.append(int(candidate["accepted_patch_rate"]))
        baseline_accept.append(int(baseline["accepted_patch_rate"]))
        candidate_time.append(int(candidate["time_to_terminal_outcome"]))
        baseline_time.append(int(baseline["time_to_terminal_outcome"]))
        candidate_audit.append(_require_audit(candidate))
        baseline_audit.append(_require_audit(baseline))
        cand_cost = candidate.get("quality_adjusted_cost")
        base_cost = baseline.get("quality_adjusted_cost")
        direct_cost = direct.get("quality_adjusted_cost")
        if type(cand_cost) is int:
            candidate_qac.append(cand_cost)
        if type(base_cost) is int:
            baseline_qac.append(base_cost)
        if type(direct_cost) is int:
            direct_qac.append(direct_cost)
        if type(cand_cost) is int and type(base_cost) is int:
            labels.append(fixture_id)
            differences.append(base_cost - cand_cost)
            ratio = _ratio_millionths(cand_cost, base_cost)
            if ratio is not None:
                ratios.append(ratio)

    if not differences or not ratios:
        raise PairedHarnessError("quality-adjusted cost sample is empty")

    outliers = tukey_outliers(differences, labels=labels)
    class_distribution = {
        name: count for name, count in sorted(Counter(classes).items())
    }
    bootstrap = bootstrap_interval(differences, seed=seed, domain="qac-difference")
    ratio_bootstrap = bootstrap_interval(ratios, seed=seed, domain="qac-ratio")
    statistics = {
        "accepted_patch_rate": {
            "baseline": _mean(baseline_accept),
            "candidate": _mean(candidate_accept),
            "difference": _mean(baseline_accept) - _mean(candidate_accept),
        },
        "audit_and_verification_overhead": {
            "baseline": _median(baseline_audit),
            "candidate": _median(candidate_audit),
            "direct": _median(
                _arm_metric(
                    [pair[DIRECT_ARM] for pair in pairs],
                    "audit_and_verification_overhead",
                )
            ),
        },
        "bootstrap_confidence_intervals": {
            "difference": bootstrap,
            "ratio": ratio_bootstrap,
        },
        "distribution_by_task_class": class_distribution,
        "mean_difference": _mean(differences),
        "median_difference": _median(differences),
        "outlier_analysis": outliers,
        "pair_count": len(pairs),
        "per_task_ratios": ratios,
        "quality_adjusted_cost": {
            "baseline": _median(baseline_qac),
            "candidate": _median(candidate_qac),
            "direct": _median(direct_qac) if direct_qac else None,
        },
        "ratio_median": _median(ratios),
        "ratio_mean": _mean(ratios),
        "time_to_terminal_outcome": {
            "baseline": _median(baseline_time),
            "candidate": _median(candidate_time),
            "difference": _median(baseline_time) - _median(candidate_time),
        },
    }
    return statistics


def _magnitude(value: int) -> int:
    return value if value >= 0 else -value


def schema_statistics(
    statistics: Mapping[str, Any],
    *,
    price_snapshot_identity: str,
) -> dict[str, Any]:
    def estimate(value: int, *, unit: str, name: str) -> dict[str, Any]:
        return estimated_quantity(
            _bounded_int(value, name=name),
            unit=unit,
            estimator_id=f"{ESTIMATOR_ID}:{name}",
            method="local_compute_model",
            price_snapshot_identity=price_snapshot_identity,
        )

    class_count = len(statistics["distribution_by_task_class"])
    outlier_count = int(statistics["outlier_analysis"]["count"])
    projected = {
        "median_difference": estimate(
            _magnitude(int(statistics["median_difference"])),
            unit="microusd",
            name="median_difference_magnitude",
        ),
        "mean_difference": estimate(
            _magnitude(int(statistics["mean_difference"])),
            unit="microusd",
            name="mean_difference_magnitude",
        ),
        "per_task_ratios": estimate(
            int(statistics["ratio_median"]),
            unit="ratio_millionths",
            name="per_task_ratios",
        ),
        "bootstrap_confidence_intervals": estimate(
            int(statistics["bootstrap_confidence_intervals"]["difference"]["width"]),
            unit="microusd",
            name="bootstrap_confidence_intervals",
        ),
        "distribution_by_task_class": measured_quantity(
            class_count,
            unit="count",
            sensor_id=f"{SENSOR_ID}:distribution_by_task_class",
        ),
        "outlier_analysis": measured_quantity(
            outlier_count,
            unit="count",
            sensor_id=f"{SENSOR_ID}:outlier_analysis",
        ),
        "quality_adjusted_cost": estimate(
            int(statistics["quality_adjusted_cost"]["candidate"]),
            unit="microusd",
            name="quality_adjusted_cost",
        ),
        "accepted_patch_rate": measured_quantity(
            int(statistics["accepted_patch_rate"]["candidate"]),
            unit="ratio_millionths",
            sensor_id=f"{SENSOR_ID}:accepted_patch_rate",
        ),
        "time_to_terminal_outcome": estimate(
            int(statistics["time_to_terminal_outcome"]["candidate"]),
            unit="seconds_millionths",
            name="time_to_terminal_outcome",
        ),
    }
    if tuple(projected) != STATISTIC_FIELDS:
        raise PairedHarnessError("schema statistic projection drifted from the closed contract")
    return projected


def run_hermetic_pairing(
    recipes: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    supplied = recipes is not None
    loaded = list(recipes) if supplied else load_hermetic_recipes()
    if not loaded:
        raise PairedHarnessError("pairing rejects missing paired inputs")
    # Caller-supplied diagnostic subsets may be smaller than the sealed corpus.
    if not supplied and len(loaded) < HERMETIC_MINIMUM:
        raise PairedHarnessError("hermetic corpus is below the sealed minimum")
    sealed_corpus = len(loaded) >= HERMETIC_MINIMUM
    task_inputs = vectors_identity(loaded)
    controls = sealed_equal_controls(task_inputs=task_inputs)
    pairs: list[dict[str, dict[str, Any]]] = []
    for index, recipe in enumerate(loaded):
        pairs.append(expand_all_arms(recipe, controls=controls, index=index))
    statistics = compute_paired_statistics(pairs)
    price_identity = _companion_identity(PRICE_PATH)
    environment_identity = _companion_identity(ENVIRONMENT_PATH)
    schema_stats = schema_statistics(statistics, price_snapshot_identity=price_identity)
    if tuple(schema_stats) != STATISTIC_FIELDS:
        missing = [name for name in STATISTIC_FIELDS if name not in schema_stats]
        raise PairedHarnessError(f"paired statistics omitted required fields: {missing}")
    manifest = build_paired_benchmark_manifest(
        objective_revision=OBJECTIVE_REVISION,
        repository_commit=REPOSITORY_COMMIT,
        repository_tree=REPOSITORY_TREE,
        policy_identity=POLICY_IDENTITY,
        task_inputs=controls["task_inputs"],
        acceptance_tests=controls["acceptance_tests"],
        available_providers_and_models=controls["available_providers_and_models"],
        price_accounting=controls["price_accounting"],
        resource_limits=controls["resource_limits"],
        human_intervention_policy=controls["human_intervention_policy"],
        maximum_retries=MAXIMUM_RETRIES,
        price_snapshot_identity=observed_identity(price_identity),
        environment_identity=observed_identity(environment_identity),
        direct_policy_identity=observed_identity(POLICY_IDENTITY),
        sealed_policy_identity=observed_identity(POLICY_IDENTITY),
        candidate_policy_identity=observed_identity(POLICY_IDENTITY),
        hermetic_count=measured_quantity(
            len(loaded),
            unit="count",
            sensor_id=f"{SENSOR_ID}:hermetic_count",
        ),
        historical_count=unavailable("not_sealed"),
        live_count=unavailable("not_sealed"),
        hermetic_status="sealed" if sealed_corpus else "insufficient",
        historical_status="unavailable",
        live_status="unavailable",
        statistics=schema_stats,
        reason="not_sealed",
    )
    admitted = manifest.to_dict()
    if admitted["hermetic_sufficient_for_production_promotion"] is not False:
        raise PairedHarnessError("hermetic evidence cannot satisfy live promotion")
    if admitted["promotion_without_paired_campaign"] is not False:
        raise PairedHarnessError("promotion without a paired campaign is forbidden")
    return {
        "controls": controls,
        "controls_identity": controls_identity(controls),
        "live": False,
        "manifest": admitted,
        "pair_count": len(pairs),
        "pairs": pairs,
        "population_kind": "hermetic_development",
        "recipes": [dict(item) for item in loaded],
        "statistics": statistics,
        "vectors_identity": task_inputs,
    }


def build_hermetic_manifest(result: Mapping[str, Any]) -> dict[str, Any]:
    recipes = result["recipes"]
    fixture_ids = [str(item["fixture_id"]) for item in recipes]
    payload = {
        "bounded": True,
        "count": len(recipes),
        "equal_controls": dict(result["controls"]),
        "fixture_ids": fixture_ids,
        "hermetic_sufficient_for_production_promotion": False,
        "live": False,
        "minimum": HERMETIC_MINIMUM,
        "objective_id": OBJECTIVE_ID,
        "objective_revision": OBJECTIVE_REVISION,
        "outcomes": sorted({str(item["outcome"]) for item in recipes}),
        "population_kind": "hermetic_development",
        "program_id": PROGRAM_ID,
        "repository_commit": REPOSITORY_COMMIT,
        "repository_tree": REPOSITORY_TREE,
        "schema": HERMETIC_MANIFEST_SCHEMA,
        "schema_version": 1,
        "seed": SEALED_SEED,
        "status": "sealed",
        "task_classes": list(TASK_CLASSES),
        "task_id": TASK_ID,
        "vectors_identity": result["vectors_identity"],
        "vectors_path": "benchmarks/agent_supervisor/efficiency_state_hardening/hermetic_vectors.jsonl",
    }
    payload["identity"] = content_identity(payload)
    return payload


def materialize_sealed_artifacts(
    *,
    directory: Path | None = None,
) -> dict[str, Any]:
    recipes = generate_hermetic_recipes()
    result = run_hermetic_pairing(recipes)
    target_dir = directory or PACKAGE_DIR
    vectors_path = target_dir / HERMETIC_VECTORS_PATH.name
    hermetic_path = target_dir / HERMETIC_MANIFEST_PATH.name
    paired_path = target_dir / PAIRED_MANIFEST_PATH.name
    vectors_path.write_text(serialize_vectors(recipes), encoding="utf-8")
    hermetic_payload = build_hermetic_manifest(result)
    hermetic_path.write_text(_pretty(hermetic_payload), encoding="utf-8")
    paired_path.write_text(_pretty(result["manifest"]), encoding="utf-8")
    replayed = admit_paired_benchmark_manifest(result["manifest"])
    if replayed.canonical_bytes() != canonical_bytes(result["manifest"]):
        raise PairedHarnessError("paired manifest did not round-trip")
    return {
        "hermetic_manifest": hermetic_payload,
        "paired_manifest": result["manifest"],
        "pair_count": result["pair_count"],
        "statistics": result["statistics"],
        "vectors_identity": result["vectors_identity"],
    }


def load_sealed_pairing() -> dict[str, Any]:
    recipes = load_hermetic_recipes()
    generated = generate_hermetic_recipes(len(recipes))
    if serialize_vectors(generated) != HERMETIC_VECTORS_PATH.read_text(encoding="utf-8"):
        raise PairedHarnessError("sealed hermetic vectors drifted from the recipe generator")
    result = run_hermetic_pairing(recipes)
    sealed_hermetic = _load_json(HERMETIC_MANIFEST_PATH)
    expected_hermetic = build_hermetic_manifest(result)
    if sealed_hermetic != expected_hermetic:
        raise PairedHarnessError("sealed hermetic manifest drifted from pairing")
    sealed_paired = _load_json(PAIRED_MANIFEST_PATH)
    if sealed_paired != result["manifest"]:
        raise PairedHarnessError("sealed paired manifest drifted from pairing")
    admit_paired_benchmark_manifest(sealed_paired)
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_pretty(payload), encoding="utf-8")


def run_cohort(cohort: str, *, minimum_tasks: int, allow_honest_nonpromotion: bool) -> dict[str, Any]:
    if cohort == "hermetic":
        result = load_sealed_pairing()
        if result["pair_count"] < minimum_tasks:
            if not allow_honest_nonpromotion:
                raise PairedHarnessError("hermetic pair count is below the requested minimum")
            status = "insufficient_evidence"
        else:
            status = "sealed_hermetic_pairs"
        return {
            "cohort": cohort,
            "hermetic_sufficient_for_production_promotion": False,
            "live": False,
            "pair_count": result["pair_count"],
            "population_kind": "hermetic_development",
            "statistics": result["statistics"],
            "status": status,
            "vectors_identity": result["vectors_identity"],
        }
    if cohort in {"historical", "live-shadow", "canary"}:
        if not allow_honest_nonpromotion:
            raise PairedHarnessError(f"{cohort} population is not sealed in ASEH-013")
        reason = "not_sealed" if cohort != "canary" else "not_admitted"
        return {
            "cohort": cohort,
            "hermetic_sufficient_for_production_promotion": False,
            "live": False,
            "pair_count": 0,
            "reason_code": reason,
            "status": "unavailable",
            "truth_state": "unavailable",
        }
    raise PairedHarnessError(f"unknown cohort {cohort}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASEH paired benchmark harness")
    parser.add_argument(
        "--cohort",
        choices=("hermetic", "historical", "live-shadow", "canary"),
        default="hermetic",
    )
    parser.add_argument("--minimum-tasks", type=int, default=HERMETIC_MINIMUM)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qualification-output", type=Path)
    parser.add_argument("--allow-honest-nonpromotion", action="store_true")
    parser.add_argument("--allow-not-admitted", action="store_true")
    parser.add_argument("--require-shadow-receipt", type=Path)
    parser.add_argument("--materialize", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.materialize:
        materialize_sealed_artifacts()
        return 0
    allow = bool(args.allow_honest_nonpromotion or args.allow_not_admitted)
    if args.cohort == "canary":
        if args.require_shadow_receipt is None or not args.require_shadow_receipt.is_file():
            payload = {
                "cohort": "canary",
                "live": False,
                "reason_code": "not_admitted",
                "status": "not_admitted",
                "truth_state": "unavailable",
            }
            if not allow:
                raise PairedHarnessError("canary requires admitted shadow evidence")
        else:
            payload = run_cohort("canary", minimum_tasks=args.minimum_tasks, allow_honest_nonpromotion=True)
    else:
        payload = run_cohort(
            args.cohort,
            minimum_tasks=args.minimum_tasks,
            allow_honest_nonpromotion=allow,
        )
    if args.output is not None:
        _write_json(args.output, payload)
    if args.qualification_output is not None:
        _write_json(args.qualification_output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
