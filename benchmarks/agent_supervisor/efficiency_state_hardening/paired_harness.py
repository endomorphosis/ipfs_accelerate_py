#!/usr/bin/env python3
"""ASEH-013 paired hermetic benchmark harness.

Interface: ``AsehPairedHarness@1``

Seals a compact 60+ fixture recipe corpus, runs the three required arms under
identical controls, and computes paired statistics plus audit overhead.
Hermetic evidence cannot grant live or promotion status. Missing paired
inputs, unequal controls, and unseeded bootstrap resampling fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.runtime.efficiency_receipts import (
    CANDIDATE_ARM_CONSTRAINTS,
    DIRECT_ARM_CONSTRAINTS,
    EQUAL_CONTROL_FIELDS,
    HERMETIC_MINIMUM,
    OBJECTIVE_ID,
    PAIRED_ARMS,
    PROGRAM_ID,
    SEALED_CURRENT_ARM_CONSTRAINTS,
    STATISTIC_FIELDS,
    admit_paired_benchmark_manifest,
    build_paired_benchmark_manifest,
    content_identity,
    measured_quantity,
    observed_identity,
    unavailable,
)


PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent
REPO_ROOT: Final[Path] = PACKAGE_DIR.parents[3]
HERMETIC_MANIFEST_PATH: Final[Path] = PACKAGE_DIR / "hermetic_manifest.json"
HERMETIC_VECTORS_PATH: Final[Path] = PACKAGE_DIR / "hermetic_vectors.jsonl"
CAMPAIGN_MANIFEST_PATH: Final[Path] = PACKAGE_DIR / "manifest.json"
PROVIDER_CONFIG_PATH: Final[Path] = PACKAGE_DIR / "provider_model_config.json"
PRICE_SNAPSHOT_PATH: Final[Path] = PACKAGE_DIR / "price_snapshot.json"
ENVIRONMENT_IDENTITY_PATH: Final[Path] = PACKAGE_DIR / "environment_identity.json"

HARNESS_INTERFACE: Final[str] = "AsehPairedHarness@1"
HERMETIC_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/aseh-hermetic-fixture-manifest@1"
)
CAMPAIGN_MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/aseh-paired-benchmark-manifest@1"
)
SENSOR_ID: Final[str] = "aseh-013-paired-harness"
TASK_ID: Final[str] = "ASEH-013"
POLICY_IDENTITY: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/aseh-supervisor-policy@1"
)
OBJECTIVE_REVISION: Final[str] = (
    "baguqeeray6iu7h6kiajjow44w3l6gexkiihui2423wrhtpou22thcm6sqhbq"
)
REPOSITORY_COMMIT: Final[str] = "755f45475cc2d13dacd8b330036c1d597afeddde"
REPOSITORY_TREE: Final[str] = "729da9f8293ecfa046a0136381a3d3808f9ed140"
MAXIMUM_RETRIES: Final[int] = 3
MILLIONTHS: Final[int] = 1_000_000
BASIS_POINTS: Final[int] = 10_000
BOOTSTRAP_SEED: Final[int] = 13
BOOTSTRAP_RESAMPLES: Final[int] = 199
BOOTSTRAP_LOWER_MILLIONTHS: Final[int] = 25_000
BOOTSTRAP_UPPER_MILLIONTHS: Final[int] = 975_000

MAX_TOKENS: Final[int] = 100_000
MAX_RETRIES: Final[int] = 32
MAX_DURATION_US: Final[int] = 3_600_000_000
MAX_BYTES: Final[int] = 1_048_576
MAX_QUALITY_BPS: Final[int] = 10_000
MAX_SERIALIZED_RECIPE_BYTES: Final[int] = 4_096
MAX_FIXTURES: Final[int] = 256

TASK_CLASSES: Final[tuple[str, ...]] = (
    "schema_change",
    "bug_repair",
    "feature_addition",
    "proof_obligation",
    "test_selection",
    "recovery_replay",
    "state_migration",
    "context_pack_build",
    "routing_policy",
    "merge_conflict",
    "human_escalation",
    "retry_rescue",
)
OUTCOMES: Final[tuple[str, ...]] = (
    "succeeded",
    "failed",
    "retried",
    "rescued",
    "conflicted",
    "human_escalated",
)
ACCEPTED_OUTCOMES: Final[frozenset[str]] = frozenset(
    {"succeeded", "retried", "rescued"}
)
ARM_MULTIPLIERS: Final[dict[str, dict[str, int]]] = {
    "direct_minimal_orchestration_baseline": {
        "tokens": 10_000,
        "cost": 10_000,
        "time": 10_000,
        "audit": 8_000,
        "quality": 9_800,
        "local": 10_000,
    },
    "sealed_current_supervisor_baseline": {
        "tokens": 8_800,
        "cost": 9_000,
        "time": 9_200,
        "audit": 10_000,
        "quality": 9_900,
        "local": 9_400,
    },
    "candidate_optimized_supervisor": {
        "tokens": 6_200,
        "cost": 7_000,
        "time": 7_400,
        "audit": 11_800,
        "quality": 10_000,
        "local": 8_100,
    },
}

RESOURCE_LIMITS: Final[dict[str, int]] = {
    "max_tokens": MAX_TOKENS,
    "max_retries": MAXIMUM_RETRIES,
    "max_duration_us": MAX_DURATION_US,
    "max_bytes": MAX_BYTES,
    "max_concurrency": 4,
}
HUMAN_INTERVENTION_POLICY: Final[dict[str, Any]] = {
    "schema": "ipfs_accelerate_py/agent-supervisor/aseh-human-intervention-policy@1",
    "maximum_human_interventions": 1,
    "escalation_required_for": ["human_escalated"],
    "shadow_candidate_cannot_page_human": True,
}
ACCEPTANCE_TESTS: Final[dict[str, str]] = {
    "schema": "ipfs_accelerate_py/agent-supervisor/aseh-acceptance-tests@1",
    "validator": (
        "test/api/agent_supervisor/efficiency_state_hardening/test_paired_harness.py"
    ),
    "command": (
        "python3 -m pytest -q "
        "test/api/agent_supervisor/efficiency_state_hardening/test_paired_harness.py"
    ),
}


class PairedHarnessError(ValueError):
    """Closed paired-harness contract violation."""


class UnequalControlError(PairedHarnessError):
    """Arms do not share identical required controls."""


def _text(value: Any, *, name: str, maximum: int = 512) -> str:
    if not isinstance(value, str):
        raise PairedHarnessError(f"{name} must be text")
    result = value.strip()
    if not result:
        raise PairedHarnessError(f"{name} must not be empty")
    encoded = result.encode("utf-8")
    if b"\x00" in encoded or len(encoded) > maximum:
        raise PairedHarnessError(f"{name} is unsafe or too large")
    return result


def _int(
    value: Any,
    *,
    name: str,
    minimum: int = 0,
    maximum: int = 10**18,
) -> int:
    if type(value) is not int:
        raise PairedHarnessError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise PairedHarnessError(f"{name} must be between {minimum} and {maximum}")
    return value


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PairedHarnessError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise PairedHarnessError(f"{name} keys must be strings")
    return {str(key): item for key, item in value.items()}


def _reject_floats(value: Any, *, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) is int:
        return
    if isinstance(value, float):
        raise PairedHarnessError(f"{path}: canonical contracts cannot contain floats")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_floats(child, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_floats(child, path=f"{path}[{index}]")
        return
    raise PairedHarnessError(f"{path}: unsupported value {type(value).__name__}")


def canonical_object(value: Mapping[str, Any]) -> dict[str, Any]:
    _reject_floats(value)
    return json.loads(json.dumps(dict(value), sort_keys=True, separators=(",", ":")))


def pretty_json(value: Mapping[str, Any]) -> str:
    _reject_floats(value)
    return json.dumps(dict(value), indent=2, sort_keys=True) + "\n"


def compact_json(value: Mapping[str, Any]) -> str:
    _reject_floats(value)
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _scale(value: int, multiplier_bps: int, *, name: str) -> int:
    scaled = (value * multiplier_bps) // BASIS_POINTS
    return _int(scaled, name=name, maximum=10**18)


def _median_int(values: Sequence[int]) -> int:
    if not values:
        raise PairedHarnessError("median is unavailable for an empty population")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2


def _mean_int(values: Sequence[int]) -> int:
    if not values:
        raise PairedHarnessError("mean is unavailable for an empty population")
    return sum(values) // len(values)


def _percentile_int(ordered: Sequence[int], percentile_millionths: int) -> int:
    if not ordered:
        raise PairedHarnessError("percentile is unavailable for an empty population")
    bounded = _int(
        percentile_millionths,
        name="percentile_millionths",
        maximum=MILLIONTHS,
    )
    index = (bounded * (len(ordered) - 1)) // MILLIONTHS
    return ordered[index]


def _ratio_millionths(numerator: int, denominator: int, *, name: str) -> int | None:
    if denominator <= 0:
        return None
    return _int(
        (numerator * MILLIONTHS) // denominator,
        name=name,
        maximum=10**18,
    )


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PairedHarnessError(f"{path.name} is unreadable") from exc
    return _mapping(payload, name=path.name)


def load_identity(path: Path, *, fallback_name: str) -> str:
    if path.is_file():
        payload = load_json_object(path)
        identity = payload.get("identity")
        if isinstance(identity, str) and identity.startswith("b"):
            return identity
        body = {key: value for key, value in payload.items() if key != "identity"}
        return content_identity(body)
    return content_identity({"name": fallback_name, "status": "companion_absent"})


def generate_fixture_recipes() -> tuple[dict[str, Any], ...]:
    """Compact combinatorial recipes; one unique bounded fixture per class/outcome."""

    recipes: list[dict[str, Any]] = []
    index = 0
    for task_class in TASK_CLASSES:
        for outcome in OUTCOMES:
            index += 1
            outlier = task_class == "retry_rescue" and outcome == "failed"
            token_budget = 48_000 if outlier else 400 + index * 17
            quality = 4_200 if outcome == "failed" else 7_200 + (index * 37) % 2_400
            recipe = {
                "accepted": outcome in ACCEPTED_OUTCOMES,
                "bytes_bound": 4_096 + index * 128,
                "duration_us": 80_000 + index * 1_250,
                "fixture_id": f"aseh-h{index:02d}",
                "live": False,
                "outcome": outcome,
                "population_kind": "hermetic_development",
                "quality_bps": quality,
                "retry_budget": index % 4,
                "seed": 13_000 + index,
                "task_class": task_class,
                "token_budget": token_budget,
            }
            recipes.append(admit_recipe(recipe))
    if len(recipes) < HERMETIC_MINIMUM:
        raise PairedHarnessError("recipe catalogue is below the hermetic minimum")
    return tuple(recipes)


def admit_recipe(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = _mapping(payload, name="recipe")
    fixture_id = _text(data.get("fixture_id"), name="fixture_id", maximum=64)
    task_class = _text(data.get("task_class"), name="task_class", maximum=64)
    outcome = _text(data.get("outcome"), name="outcome", maximum=64)
    if task_class not in TASK_CLASSES:
        raise PairedHarnessError(f"{fixture_id}: task_class is not a closed value")
    if outcome not in OUTCOMES:
        raise PairedHarnessError(f"{fixture_id}: outcome is not a closed value")
    if data.get("live") is not False:
        raise PairedHarnessError(f"{fixture_id}: hermetic fixtures cannot be live")
    if data.get("population_kind") != "hermetic_development":
        raise PairedHarnessError(f"{fixture_id}: population_kind must be hermetic")
    accepted = data.get("accepted")
    if type(accepted) is not bool:
        raise PairedHarnessError(f"{fixture_id}: accepted must be a boolean")
    if accepted != (outcome in ACCEPTED_OUTCOMES):
        raise PairedHarnessError(f"{fixture_id}: accepted flag does not match outcome")
    recipe = {
        "accepted": accepted,
        "bytes_bound": _int(data.get("bytes_bound"), name="bytes_bound", minimum=1, maximum=MAX_BYTES),
        "duration_us": _int(
            data.get("duration_us"),
            name="duration_us",
            minimum=1,
            maximum=MAX_DURATION_US,
        ),
        "fixture_id": fixture_id,
        "live": False,
        "outcome": outcome,
        "population_kind": "hermetic_development",
        "quality_bps": _int(
            data.get("quality_bps"),
            name="quality_bps",
            minimum=1,
            maximum=MAX_QUALITY_BPS,
        ),
        "retry_budget": _int(
            data.get("retry_budget"),
            name="retry_budget",
            maximum=MAX_RETRIES,
        ),
        "seed": _int(data.get("seed"), name="seed", minimum=1),
        "task_class": task_class,
        "token_budget": _int(
            data.get("token_budget"),
            name="token_budget",
            minimum=1,
            maximum=MAX_TOKENS,
        ),
    }
    encoded = compact_json(recipe)
    if len(encoded.encode("utf-8")) > MAX_SERIALIZED_RECIPE_BYTES:
        raise PairedHarnessError(f"{fixture_id}: recipe exceeds the serialized bound")
    return canonical_object(recipe)


def recipe_identity(recipe: Mapping[str, Any]) -> str:
    return content_identity(admit_recipe(recipe))


def bound_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    admitted = admit_recipe(recipe)
    if admitted["retry_budget"] > MAXIMUM_RETRIES:
        raise PairedHarnessError(f"{admitted['fixture_id']}: retry_budget exceeds campaign limit")
    return admitted


def shared_equal_controls(
    *,
    task_inputs: str,
    acceptance_tests: str | None = None,
    available_providers_and_models: str | None = None,
    price_accounting: str | None = None,
    resource_limits: str | None = None,
    human_intervention_policy: str | None = None,
    maximum_retries: int = MAXIMUM_RETRIES,
    repository_revision: str = REPOSITORY_TREE,
    objective: str = OBJECTIVE_ID,
) -> dict[str, Any]:
    controls = {
        "acceptance_tests": acceptance_tests or content_identity(ACCEPTANCE_TESTS),
        "available_providers_and_models": available_providers_and_models
        or load_identity(PROVIDER_CONFIG_PATH, fallback_name="providers"),
        "human_intervention_policy": human_intervention_policy
        or content_identity(HUMAN_INTERVENTION_POLICY),
        "maximum_retries": _int(maximum_retries, name="maximum_retries", maximum=MAX_RETRIES),
        "objective": _text(objective, name="objective", maximum=64),
        "price_accounting": price_accounting
        or load_identity(PRICE_SNAPSHOT_PATH, fallback_name="price"),
        "repository_revision": _text(repository_revision, name="repository_revision", maximum=40),
        "resource_limits": resource_limits or content_identity(RESOURCE_LIMITS),
        "task_inputs": _text(task_inputs, name="task_inputs", maximum=128),
    }
    if set(controls) != set(EQUAL_CONTROL_FIELDS):
        raise PairedHarnessError("equal controls drifted from the closed field set")
    return controls


def require_equal_controls(
    arm_controls: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Reject missing arms or any control mismatch across the three paired arms."""

    controls = _mapping(arm_controls, name="arm_controls")
    missing_arms = [arm for arm in PAIRED_ARMS if arm not in controls]
    extra_arms = sorted(set(controls) - set(PAIRED_ARMS))
    if missing_arms or extra_arms:
        raise UnequalControlError(
            "pairing requires the three closed arms; "
            f"missing={missing_arms or 'none'} extra={extra_arms or 'none'}"
        )
    admitted: dict[str, dict[str, Any]] = {}
    for arm_id in PAIRED_ARMS:
        bundle = _mapping(controls[arm_id], name=f"{arm_id}.controls")
        unknown = sorted(set(bundle) - set(EQUAL_CONTROL_FIELDS))
        missing = [field for field in EQUAL_CONTROL_FIELDS if field not in bundle]
        if unknown or missing:
            raise UnequalControlError(
                f"{arm_id} controls are unequal to the closed set; "
                f"missing={missing or 'none'} unknown={unknown or 'none'}"
            )
        admitted[arm_id] = {field: bundle[field] for field in EQUAL_CONTROL_FIELDS}
    reference = admitted[PAIRED_ARMS[0]]
    mismatches: list[str] = []
    for arm_id in PAIRED_ARMS[1:]:
        for field in EQUAL_CONTROL_FIELDS:
            if admitted[arm_id][field] != reference[field]:
                mismatches.append(f"{arm_id}.{field}")
    if mismatches:
        raise UnequalControlError(
            "pairing rejects unequal controls: " + ", ".join(mismatches)
        )
    return dict(reference)


def controls_for_arms(controls: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    shared = {field: controls[field] for field in EQUAL_CONTROL_FIELDS}
    return {arm_id: dict(shared) for arm_id in PAIRED_ARMS}


def simulate_arm(recipe: Mapping[str, Any], *, arm_id: str) -> dict[str, Any]:
    admitted = bound_recipe(recipe)
    if arm_id not in ARM_MULTIPLIERS:
        raise PairedHarnessError(f"{arm_id} is not a closed paired arm")
    multipliers = ARM_MULTIPLIERS[arm_id]
    provider_cost = _scale(admitted["token_budget"] * 3, multipliers["cost"], name="provider_cost")
    local_cost = _scale(2 + admitted["retry_budget"], multipliers["local"], name="local_cost")
    audit = _scale(40 + admitted["retry_budget"] * 5, multipliers["audit"], name="audit_overhead")
    quality = _scale(admitted["quality_bps"], multipliers["quality"], name="quality_bps")
    quality = max(1, min(MAX_QUALITY_BPS, quality))
    tokens = max(1, _scale(admitted["token_budget"], multipliers["tokens"], name="tokens"))
    duration = max(1, _scale(admitted["duration_us"], multipliers["time"], name="duration_us"))
    gross = provider_cost + local_cost
    net = gross + audit
    quality_adjusted = (net * BASIS_POINTS) // quality
    observation = {
        "accepted": admitted["accepted"],
        "arm_id": arm_id,
        "audit_overhead_microusd": audit,
        "fixture_id": admitted["fixture_id"],
        "gross_cost_microusd": gross,
        "input_tokens": tokens,
        "live": False,
        "net_cost_microusd": net,
        "outcome": admitted["outcome"],
        "population_kind": "hermetic_development",
        "provider_cost_microusd": provider_cost,
        "quality_adjusted_cost_microusd": quality_adjusted,
        "quality_bps": quality,
        "simulated": True,
        "task_class": admitted["task_class"],
        "terminal_time_us": duration,
        "truth_state": "simulated",
    }
    return canonical_object(observation)


def _require_seed(seed: int | None) -> int:
    if seed is None:
        raise PairedHarnessError("bootstrap resampling requires an explicit seed")
    return _int(seed, name="bootstrap_seed", minimum=0)


def bootstrap_interval(
    values: Sequence[int],
    *,
    seed: int | None,
    resamples: int = BOOTSTRAP_RESAMPLES,
    lower_millionths: int = BOOTSTRAP_LOWER_MILLIONTHS,
    upper_millionths: int = BOOTSTRAP_UPPER_MILLIONTHS,
    statistic: str = "median",
) -> dict[str, int]:
    bound_seed = _require_seed(seed)
    if statistic not in {"median", "mean"}:
        raise PairedHarnessError("bootstrap statistic must be median or mean")
    sample = [_int(item, name="bootstrap_value") for item in values]
    if not sample:
        raise PairedHarnessError("bootstrap interval is unavailable for an empty population")
    count = _int(resamples, name="bootstrap_resamples", minimum=1, maximum=10_000)
    rng = random.Random(bound_seed)
    stats: list[int] = []
    selector = _median_int if statistic == "median" else _mean_int
    population = len(sample)
    for _ in range(count):
        draw = [sample[rng.randrange(population)] for _ in range(population)]
        stats.append(selector(draw))
    ordered = sorted(stats)
    lower = _percentile_int(ordered, lower_millionths)
    upper = _percentile_int(ordered, upper_millionths)
    width = upper - lower if upper >= lower else lower - upper
    return {
        "lower": lower,
        "resamples": count,
        "seed": bound_seed,
        "upper": upper,
        "width": width,
    }


def _outlier_ids(pairs: Sequence[Mapping[str, Any]], values: Sequence[int]) -> tuple[str, ...]:
    if len(values) < 4:
        return ()
    ordered = sorted(values)
    q1 = _percentile_int(ordered, 250_000)
    q3 = _percentile_int(ordered, 750_000)
    iqr = q3 - q1
    low = q1 - (3 * iqr) // 2
    high = q3 + (3 * iqr) // 2
    flagged: list[str] = []
    for pair, value in zip(pairs, values):
        if value < low or value > high:
            flagged.append(str(pair["fixture_id"]))
    return tuple(flagged)


def _pair_metric(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    field: str,
) -> tuple[int, int | None]:
    difference = _int(left[field], name=field) - _int(right[field], name=field)
    ratio = _ratio_millionths(
        _int(right[field], name=field),
        _int(left[field], name=field),
        name=f"{field}_ratio",
    )
    return difference, ratio


def compute_paired_statistics(
    pairs: Sequence[Mapping[str, Any]],
    *,
    baseline_arm: str,
    candidate_arm: str,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    if not pairs:
        raise PairedHarnessError("paired statistics are unavailable for an empty population")
    _require_seed(seed)
    cost_diffs: list[int] = []
    token_diffs: list[int] = []
    time_diffs: list[int] = []
    qac_diffs: list[int] = []
    cost_ratios: list[int] = []
    qac_values: list[int] = []
    terminal_times: list[int] = []
    accepted = 0
    classes: list[str] = []
    for pair in pairs:
        mapping = _mapping(pair, name="pair")
        baseline = _mapping(mapping["baseline"], name="baseline")
        candidate = _mapping(mapping["candidate"], name="candidate")
        if baseline.get("arm_id") != baseline_arm or candidate.get("arm_id") != candidate_arm:
            raise PairedHarnessError("pair arms do not match the requested comparison")
        if not (
            mapping.get("fixture_id")
            == baseline.get("fixture_id")
            == candidate.get("fixture_id")
        ):
            raise PairedHarnessError("paired inputs must share fixture_id")
        cost_diff, cost_ratio = _pair_metric(baseline, candidate, "net_cost_microusd")
        token_diff, _token_ratio = _pair_metric(baseline, candidate, "input_tokens")
        time_diff, _time_ratio = _pair_metric(baseline, candidate, "terminal_time_us")
        qac_diff, _qac_ratio = _pair_metric(
            baseline, candidate, "quality_adjusted_cost_microusd"
        )
        cost_diffs.append(cost_diff)
        token_diffs.append(token_diff)
        time_diffs.append(time_diff)
        qac_diffs.append(qac_diff)
        if cost_ratio is not None:
            cost_ratios.append(cost_ratio)
        qac_values.append(_int(candidate["quality_adjusted_cost_microusd"], name="qac"))
        terminal_times.append(_int(candidate["terminal_time_us"], name="terminal_time_us"))
        if candidate.get("accepted") is True:
            accepted += 1
        classes.append(_text(candidate.get("task_class"), name="task_class", maximum=64))
    if not cost_ratios:
        raise PairedHarnessError("per-task ratios are unavailable because every baseline was zero")
    class_counts = {name: classes.count(name) for name in TASK_CLASSES if name in classes}
    outliers = _outlier_ids(pairs, cost_diffs)
    median_cost = _median_int(cost_diffs)
    mean_cost = _mean_int(cost_diffs)
    interval = bootstrap_interval(
        cost_diffs,
        seed=seed,
        resamples=resamples,
        statistic="median",
    )
    acceptance_rate = _ratio_millionths(accepted, len(pairs), name="accepted_patch_rate")
    if acceptance_rate is None:
        raise PairedHarnessError("accepted patch rate is unavailable")
    unsigned_median = median_cost if median_cost >= 0 else -median_cost
    unsigned_mean = mean_cost if mean_cost >= 0 else -mean_cost
    return {
        "accepted_patch_rate": {
            "accepted": accepted,
            "total": len(pairs),
            "value": acceptance_rate,
            "unit": "ratio_millionths",
        },
        "baseline_arm": baseline_arm,
        "bootstrap_confidence_intervals": interval,
        "candidate_arm": candidate_arm,
        "distribution_by_task_class": class_counts,
        "mean_difference": {
            "cost_microusd": mean_cost,
            "input_tokens": _mean_int(token_diffs),
            "terminal_time_us": _mean_int(time_diffs),
            "unsigned_cost_microusd": unsigned_mean,
        },
        "median_difference": {
            "cost_microusd": median_cost,
            "input_tokens": _median_int(token_diffs),
            "quality_adjusted_cost_microusd": _median_int(qac_diffs),
            "terminal_time_us": _median_int(time_diffs),
            "unsigned_cost_microusd": unsigned_median,
        },
        "outlier_analysis": {
            "count": len(outliers),
            "fixture_ids": list(outliers),
            "method": "tukey_iqr_150",
        },
        "pair_count": len(pairs),
        "per_task_ratios": {
            "median_ratio_millionths": _median_int(cost_ratios),
            "mean_ratio_millionths": _mean_int(cost_ratios),
            "values": cost_ratios,
        },
        "quality_adjusted_cost": {
            "median_candidate_microusd": _median_int(qac_values),
            "includes_audit_overhead": True,
        },
        "time_to_terminal_outcome": {
            "median_us": _median_int(terminal_times),
            "mean_us": _mean_int(terminal_times),
            "unit": "seconds_millionths",
        },
    }


def pair_fixture_observations(
    observations: Mapping[str, Mapping[str, Any]],
    *,
    controls: Mapping[str, Any],
) -> dict[str, Any]:
    arms = _mapping(observations, name="observations")
    missing = [arm for arm in PAIRED_ARMS if arm not in arms]
    if missing:
        raise PairedHarnessError(f"missing paired inputs for arms: {missing}")
    for arm_id in PAIRED_ARMS:
        _mapping(arms[arm_id], name=f"{arm_id}.observation")
    fixture_ids = {arms[arm]["fixture_id"] for arm in PAIRED_ARMS}
    if len(fixture_ids) != 1:
        raise PairedHarnessError("paired inputs must share one fixture_id")
    live_flags = [arms[arm].get("live") for arm in PAIRED_ARMS]
    if any(flag is True for flag in live_flags):
        raise PairedHarnessError("hermetic pairing cannot mark an arm live")
    return {
        "controls": {field: controls[field] for field in EQUAL_CONTROL_FIELDS},
        "fixture_id": next(iter(fixture_ids)),
        "live": False,
        "observations": {arm: canonical_object(arms[arm]) for arm in PAIRED_ARMS},
        "task_class": arms[PAIRED_ARMS[0]]["task_class"],
    }


def run_fixture_arms(
    recipe: Mapping[str, Any],
    *,
    controls: Mapping[str, Any],
) -> dict[str, Any]:
    observations = {
        arm_id: simulate_arm(recipe, arm_id=arm_id) for arm_id in PAIRED_ARMS
    }
    return pair_fixture_observations(observations, controls=controls)


def _comparison_pairs(
    paired_fixtures: Sequence[Mapping[str, Any]],
    *,
    baseline_arm: str,
    candidate_arm: str,
) -> tuple[dict[str, Any], ...]:
    pairs: list[dict[str, Any]] = []
    for item in paired_fixtures:
        mapping = _mapping(item, name="paired_fixture")
        observations = _mapping(mapping["observations"], name="observations")
        pairs.append(
            {
                "baseline": observations[baseline_arm],
                "candidate": observations[candidate_arm],
                "fixture_id": mapping["fixture_id"],
            }
        )
    return tuple(pairs)


def run_paired_campaign(
    recipes: Sequence[Mapping[str, Any]] | None = None,
    *,
    controls: Mapping[str, Any] | None = None,
    arm_controls: Mapping[str, Mapping[str, Any]] | None = None,
    bootstrap_seed: int | None = BOOTSTRAP_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    seed = _require_seed(bootstrap_seed)
    catalogue = tuple(admit_recipe(item) for item in (recipes or generate_fixture_recipes()))
    if len(catalogue) < HERMETIC_MINIMUM:
        raise PairedHarnessError("campaign is below the hermetic minimum of 60 unique fixtures")
    identities = [recipe_identity(item) for item in catalogue]
    if len(set(identities)) != len(identities):
        raise PairedHarnessError("hermetic fixtures must be unique by content identity")
    if len({item["fixture_id"] for item in catalogue}) != len(catalogue):
        raise PairedHarnessError("hermetic fixture_id values must be unique")
    vectors_identity = content_identity({"recipes": list(catalogue)})
    shared = dict(controls) if controls is not None else shared_equal_controls(
        task_inputs=vectors_identity
    )
    if arm_controls is None:
        require_equal_controls(controls_for_arms(shared))
    else:
        shared = require_equal_controls(arm_controls)
    paired = [run_fixture_arms(recipe, controls=shared) for recipe in catalogue]
    comparisons = {
        "candidate_vs_direct": compute_paired_statistics(
            _comparison_pairs(
                paired,
                baseline_arm="direct_minimal_orchestration_baseline",
                candidate_arm="candidate_optimized_supervisor",
            ),
            baseline_arm="direct_minimal_orchestration_baseline",
            candidate_arm="candidate_optimized_supervisor",
            seed=seed,
            resamples=bootstrap_resamples,
        ),
        "candidate_vs_sealed_current": compute_paired_statistics(
            _comparison_pairs(
                paired,
                baseline_arm="sealed_current_supervisor_baseline",
                candidate_arm="candidate_optimized_supervisor",
            ),
            baseline_arm="sealed_current_supervisor_baseline",
            candidate_arm="candidate_optimized_supervisor",
            seed=seed,
            resamples=bootstrap_resamples,
        ),
        "sealed_current_vs_direct": compute_paired_statistics(
            _comparison_pairs(
                paired,
                baseline_arm="direct_minimal_orchestration_baseline",
                candidate_arm="sealed_current_supervisor_baseline",
            ),
            baseline_arm="direct_minimal_orchestration_baseline",
            candidate_arm="sealed_current_supervisor_baseline",
            seed=seed,
            resamples=bootstrap_resamples,
        ),
    }
    primary = comparisons["candidate_vs_sealed_current"]
    audit_total = 0
    for item in paired:
        for arm_id in PAIRED_ARMS:
            audit_total += item["observations"][arm_id]["audit_overhead_microusd"]
    result = {
        "arms": list(PAIRED_ARMS),
        "audit_overhead": {
            "includes_audit_in_quality_adjusted_cost": True,
            "total_microusd": audit_total,
            "unit": "microusd",
        },
        "authority": False,
        "bootstrap_seed": seed,
        "comparisons": comparisons,
        "controls": shared,
        "fixture_count": len(catalogue),
        "hermetic_sufficient_for_production_promotion": False,
        "live": False,
        "paired_fixtures": paired,
        "population_kind": "hermetic_development",
        "primary_comparison": "candidate_vs_sealed_current",
        "promotion_without_paired_campaign": False,
        "recipes": list(catalogue),
        "required_statistics": list(STATISTIC_FIELDS),
        "statistics": primary,
        "vectors_identity": vectors_identity,
    }
    _reject_floats(result)
    return result


def build_hermetic_manifest(
    recipes: Sequence[Mapping[str, Any]],
    *,
    controls: Mapping[str, Any],
    vectors_sha256: str,
    vectors_identity: str,
) -> dict[str, Any]:
    fixtures = []
    for recipe in recipes:
        admitted = admit_recipe(recipe)
        fixtures.append(
            {
                "bounded": True,
                "bytes_bound": admitted["bytes_bound"],
                "fixture_id": admitted["fixture_id"],
                "identity": recipe_identity(admitted),
                "live": False,
                "outcome": admitted["outcome"],
                "task_class": admitted["task_class"],
                "token_budget": admitted["token_budget"],
            }
        )
    payload = {
        "arm_constraints": {
            "candidate_optimized_supervisor": list(CANDIDATE_ARM_CONSTRAINTS),
            "direct_minimal_orchestration_baseline": list(DIRECT_ARM_CONSTRAINTS),
            "sealed_current_supervisor_baseline": list(SEALED_CURRENT_ARM_CONSTRAINTS),
        },
        "arms": list(PAIRED_ARMS),
        "authority": False,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bounds": {
            "max_bytes": MAX_BYTES,
            "max_duration_us": MAX_DURATION_US,
            "max_quality_bps": MAX_QUALITY_BPS,
            "max_retries": MAX_RETRIES,
            "max_tokens": MAX_TOKENS,
        },
        "controls": {field: controls[field] for field in EQUAL_CONTROL_FIELDS},
        "count": len(fixtures),
        "equal_control_fields": list(EQUAL_CONTROL_FIELDS),
        "fixtures": fixtures,
        "hermetic_sufficient_for_production_promotion": False,
        "interface": HARNESS_INTERFACE,
        "live": False,
        "minimum": HERMETIC_MINIMUM,
        "objective_id": OBJECTIVE_ID,
        "objective_revision": OBJECTIVE_REVISION,
        "population_kind": "hermetic_development",
        "program_id": PROGRAM_ID,
        "recipe_identity": content_identity({"recipes": [admit_recipe(item) for item in recipes]}),
        "schema": HERMETIC_MANIFEST_SCHEMA,
        "schema_version": 1,
        "status": "sealed",
        "task_classes": list(TASK_CLASSES),
        "task_id": TASK_ID,
        "vectors_identity": vectors_identity,
        "vectors_path": "benchmarks/agent_supervisor/efficiency_state_hardening/hermetic_vectors.jsonl",
        "vectors_sha256": vectors_sha256,
    }
    body = {key: value for key, value in payload.items() if key != "identity"}
    payload["identity"] = content_identity(body)
    return canonical_object(payload)


def _quantity_from_statistic(value: int, *, unit: str) -> dict[str, Any]:
    return measured_quantity(value if value >= 0 else -value, unit=unit, sensor_id=SENSOR_ID)


def build_campaign_manifest(result: Mapping[str, Any]) -> dict[str, Any]:
    stats = _mapping(result["statistics"], name="statistics")
    controls = _mapping(result["controls"], name="controls")
    environment = load_identity(ENVIRONMENT_IDENTITY_PATH, fallback_name="environment")
    price = controls["price_accounting"]
    policy = observed_identity(POLICY_IDENTITY)
    unavailable_historical = unavailable("not_sealed")
    hermetic_count = measured_quantity(
        _int(result["fixture_count"], name="fixture_count", minimum=HERMETIC_MINIMUM),
        unit="count",
        sensor_id=SENSOR_ID,
    )
    statistics = {
        "accepted_patch_rate": _quantity_from_statistic(
            stats["accepted_patch_rate"]["value"],
            unit="ratio_millionths",
        ),
        "bootstrap_confidence_intervals": _quantity_from_statistic(
            stats["bootstrap_confidence_intervals"]["width"],
            unit="microusd",
        ),
        "distribution_by_task_class": _quantity_from_statistic(
            len(stats["distribution_by_task_class"]),
            unit="count",
        ),
        "mean_difference": _quantity_from_statistic(
            stats["mean_difference"]["unsigned_cost_microusd"],
            unit="microusd",
        ),
        "median_difference": _quantity_from_statistic(
            stats["median_difference"]["unsigned_cost_microusd"],
            unit="microusd",
        ),
        "outlier_analysis": _quantity_from_statistic(
            stats["outlier_analysis"]["count"],
            unit="count",
        ),
        "per_task_ratios": _quantity_from_statistic(
            stats["per_task_ratios"]["median_ratio_millionths"],
            unit="ratio_millionths",
        ),
        "quality_adjusted_cost": _quantity_from_statistic(
            stats["quality_adjusted_cost"]["median_candidate_microusd"],
            unit="microusd",
        ),
        "time_to_terminal_outcome": _quantity_from_statistic(
            stats["time_to_terminal_outcome"]["median_us"],
            unit="seconds_millionths",
        ),
    }
    manifest = build_paired_benchmark_manifest(
        objective_revision=OBJECTIVE_REVISION,
        repository_commit=REPOSITORY_COMMIT,
        repository_tree=REPOSITORY_TREE,
        policy_identity=POLICY_IDENTITY,
        task_inputs=str(controls["task_inputs"]),
        acceptance_tests=str(controls["acceptance_tests"]),
        available_providers_and_models=str(controls["available_providers_and_models"]),
        price_accounting=str(controls["price_accounting"]),
        resource_limits=str(controls["resource_limits"]),
        human_intervention_policy=str(controls["human_intervention_policy"]),
        maximum_retries=_int(controls["maximum_retries"], name="maximum_retries"),
        price_snapshot_identity=observed_identity(price),
        environment_identity=observed_identity(environment),
        direct_policy_identity=policy,
        sealed_policy_identity=policy,
        candidate_policy_identity=policy,
        hermetic_count=hermetic_count,
        historical_count=unavailable_historical,
        live_count=unavailable("not_sealed"),
        enrollment_deadline=unavailable("not_sealed"),
        statistics=statistics,
        hermetic_status="sealed",
        historical_status="unavailable",
        live_status="unavailable",
    )
    payload = manifest.to_dict()
    if payload["hermetic_sufficient_for_production_promotion"] is not False:
        raise PairedHarnessError("hermetic evidence cannot satisfy production promotion")
    if payload["promotion_without_paired_campaign"] is not False:
        raise PairedHarnessError("promotion without a paired campaign is forbidden")
    return payload


def render_vectors(recipes: Sequence[Mapping[str, Any]]) -> str:
    lines = [compact_json(admit_recipe(item)) for item in recipes]
    return "\n".join(lines) + "\n"


def load_vectors(path: Path = HERMETIC_VECTORS_PATH) -> tuple[dict[str, Any], ...]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PairedHarnessError(f"{path.name} is unreadable") from exc
    recipes: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PairedHarnessError(f"{path.name}:{line_number} is not JSON") from exc
        recipes.append(admit_recipe(payload))
    if len(recipes) > MAX_FIXTURES:
        raise PairedHarnessError("hermetic vector corpus exceeds the fixture bound")
    return tuple(recipes)


def seal_artifacts(
    recipes: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    catalogue = tuple(admit_recipe(item) for item in (recipes or generate_fixture_recipes()))
    vectors_text = render_vectors(catalogue)
    vectors_identity = content_identity({"recipes": list(catalogue)})
    controls = shared_equal_controls(task_inputs=vectors_identity)
    campaign = run_paired_campaign(catalogue, controls=controls)
    hermetic_manifest = build_hermetic_manifest(
        catalogue,
        controls=controls,
        vectors_sha256=sha256_bytes(vectors_text.encode("utf-8")),
        vectors_identity=vectors_identity,
    )
    campaign_manifest = build_campaign_manifest(campaign)
    return {
        "campaign": campaign,
        "campaign_manifest": campaign_manifest,
        "hermetic_manifest": hermetic_manifest,
        "recipes": list(catalogue),
        "vectors_text": vectors_text,
    }


def write_sealed_artifacts(directory: Path | None = None) -> dict[str, Path]:
    target = directory or PACKAGE_DIR
    sealed = seal_artifacts()
    paths = {
        "hermetic_manifest": target / "hermetic_manifest.json",
        "hermetic_vectors": target / "hermetic_vectors.jsonl",
        "manifest": target / "manifest.json",
    }
    paths["hermetic_manifest"].write_text(
        pretty_json(sealed["hermetic_manifest"]), encoding="utf-8"
    )
    paths["hermetic_vectors"].write_text(sealed["vectors_text"], encoding="utf-8")
    paths["manifest"].write_text(pretty_json(sealed["campaign_manifest"]), encoding="utf-8")
    return paths


def verify_sealed_artifacts(directory: Path | None = None) -> dict[str, Any]:
    target = directory or PACKAGE_DIR
    sealed = seal_artifacts()
    observed_manifest = (target / "hermetic_manifest.json").read_text(encoding="utf-8")
    observed_vectors = (target / "hermetic_vectors.jsonl").read_text(encoding="utf-8")
    observed_campaign = (target / "manifest.json").read_text(encoding="utf-8")
    expected_manifest = pretty_json(sealed["hermetic_manifest"])
    expected_campaign = pretty_json(sealed["campaign_manifest"])
    if observed_vectors != sealed["vectors_text"]:
        raise PairedHarnessError("hermetic_vectors.jsonl drifted from the sealed recipe catalogue")
    if observed_manifest != expected_manifest:
        raise PairedHarnessError("hermetic_manifest.json drifted from the sealed corpus")
    if observed_campaign != expected_campaign:
        raise PairedHarnessError("manifest.json drifted from the paired campaign")
    loaded = load_vectors(target / "hermetic_vectors.jsonl")
    identities = {recipe_identity(item) for item in loaded}
    if len(identities) < HERMETIC_MINIMUM:
        raise PairedHarnessError("sealed corpus is below the unique hermetic minimum")
    admit_paired_benchmark_manifest(json.loads(observed_campaign))
    return {
        "fixture_count": len(loaded),
        "unique_identities": len(identities),
        "vectors_identity": sealed["campaign"]["vectors_identity"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASEH-013 paired hermetic harness")
    parser.add_argument("--write", action="store_true", help="seal corpus files in place")
    parser.add_argument("--check", action="store_true", help="verify sealed corpus files")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.write:
        write_sealed_artifacts()
        if not args.check:
            return 0
    if args.check:
        verify_sealed_artifacts()
        return 0
    campaign = run_paired_campaign()
    sys.stdout.write(
        pretty_json(
            {
                "fixture_count": campaign["fixture_count"],
                "live": campaign["live"],
                "primary_comparison": campaign["primary_comparison"],
                "statistics": {
                    key: campaign["statistics"][key]
                    for key in (
                        "median_difference",
                        "mean_difference",
                        "accepted_patch_rate",
                        "quality_adjusted_cost",
                    )
                },
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
