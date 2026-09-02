#!/usr/bin/env python3
"""ASEH-035 ContextPack reuse, omission, and net-savings benchmark.

Interface: ``AsehContextPackBenchmark@1``

Paired current-tree measures cover eligible reuse, before/after tokens,
expansion precision/recall, critical-omission detection, stale rejection,
build/retrieval cost, audit overhead, and net provider-cost effect. Missing
telemetry stays tagged unavailable and is never encoded as numeric zero.
Seeded critical omissions and stale packs are rejected on every arm.
Hermetic evidence cannot grant live reuse or promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from ipfs_accelerate_py.agent_supervisor.context.context_compiler import (
    CalibratedTokenEstimator,
)
from ipfs_accelerate_py.agent_supervisor.runtime.efficiency_receipts import (
    CANDIDATE_ARM_CONSTRAINTS,
    DIRECT_ARM_CONSTRAINTS,
    EQUAL_CONTROL_FIELDS,
    PAIRED_ARMS,
    SEALED_CURRENT_ARM_CONSTRAINTS,
    collect_unavailable_fields,
    content_identity,
    estimated_quantity,
    measured_quantity,
    unavailable,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.context_pack import (
    CurrentPackIdentity,
    encode_context_pack_envelope,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.context_pack_selector import (
    select_current_minimal_pack,
)
from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_datasets_py.proof_context.context_pack import (
    CompletenessFailure,
    CriticalOmissionError,
    build_context_pack,
    build_minimal_semantic_pack,
)
from ipfs_datasets_py.proof_context.incremental_context import (
    expansion_precision_recall,
    expand_incremental_pack,
)
from ipfs_kit_py.proof_context.state_store import open_context_pack_store


PACKAGE_DIR: Final[Path] = Path(__file__).resolve().parent
REPO_ROOT: Final[Path] = PACKAGE_DIR.parents[3]
MANIFEST_PATH: Final[Path] = PACKAGE_DIR / "context_pack_manifest.json"
PROVIDER_CONFIG_PATH: Final[Path] = PACKAGE_DIR / "provider_model_config.json"
PRICE_SNAPSHOT_PATH: Final[Path] = PACKAGE_DIR / "price_snapshot.json"
ENVIRONMENT_IDENTITY_PATH: Final[Path] = PACKAGE_DIR / "environment_identity.json"

BENCHMARK_INTERFACE: Final[str] = "AsehContextPackBenchmark@1"
MANIFEST_SCHEMA: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/aseh-context-pack-benchmark-manifest@1"
)
SENSOR_ID: Final[str] = "aseh-035-context-pack-benchmark"
TASK_ID: Final[str] = "ASEH-035"
OBJECTIVE_ID: Final[str] = "ASEH-G040"
POLICY_IDENTITY: Final[str] = (
    "ipfs_accelerate_py/agent-supervisor/aseh-supervisor-policy@1"
)
OBJECTIVE_REVISION: Final[str] = (
    "baguqeeraeizp2zwv4rh5jynxmlr6xhtlh67kxwxwkgdmk6xf5zekxzjxcgia"
)
REPOSITORY_COMMIT: Final[str] = "755f45475cc2d13dacd8b330036c1d597afeddde"
REPOSITORY_TREE: Final[str] = "729da9f8293ecfa046a0136381a3d3808f9ed140"
TREE_OID: Final[str] = "16ef68abe8a35a3033dfaf1ed4e8d6132600df8f"
STALE_TREE_OID: Final[str] = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
TOOLCHAIN: Final[str] = "python3.12"
ENVIRONMENT: Final[str] = "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
SCHEMA_IDENTITY: Final[str] = "ipfs-datasets.proof-context.context-pack@0.1"
PACK_OBJECTIVE: Final[str] = "ASEH-035"
PACK_OBJECTIVE_REVISION: Final[str] = (
    "baguqeerazrumjlclkl2morwckg4j7jrk2txy324iqxb6p3onhaezvwmvftza"
)

MILLIONTHS: Final[int] = 1_000_000
BUILD_COST_PER_BYTE: Final[int] = 1
RETRIEVAL_COST_PER_ITEM: Final[int] = 25
AUDIT_COST_PER_CHECK: Final[int] = 40
TOKEN_MICROUSD: Final[int] = 3
MAX_SERIALIZED_RECIPE_BYTES: Final[int] = 4_096

REQUIRED_METRICS: Final[tuple[str, ...]] = (
    "eligible_reuse",
    "context_tokens_before",
    "context_tokens_after",
    "expansion_precision",
    "expansion_recall",
    "critical_omission_detection",
    "stale_rejection",
    "build_cost",
    "retrieval_cost",
    "audit_overhead",
    "net_provider_cost_effect",
)
SCENARIO_KINDS: Final[tuple[str, ...]] = (
    "eligible_reuse",
    "incremental_expansion",
    "critical_omission",
    "stale_pack",
    "fixture_as_live",
    "incomplete_evidence",
)
STALE_FIELDS: Final[tuple[str, ...]] = (
    "tree",
    "objective",
    "policy",
    "interface",
    "toolchain",
    "environment",
)
CRITICAL_KINDS: Final[tuple[str, ...]] = (
    "symbol",
    "contract",
    "test",
    "obligation",
)
ACCEPTANCE_TESTS: Final[dict[str, Any]] = {
    "schema": "ipfs_accelerate_py/agent-supervisor/aseh-acceptance-tests@1",
    "validator": (
        "test/api/agent_supervisor/efficiency_state_hardening/"
        "test_context_pack_benchmark.py"
    ),
    "command": (
        "python3 -m pytest -q "
        "test/api/agent_supervisor/efficiency_state_hardening/"
        "test_context_pack_benchmark.py"
    ),
}
RESOURCE_LIMITS: Final[dict[str, int]] = {
    "max_tokens": 100_000,
    "max_retries": 3,
    "max_duration_us": 3_600_000_000,
    "max_bytes": 1_048_576,
    "max_concurrency": 4,
}
HUMAN_INTERVENTION_POLICY: Final[dict[str, Any]] = {
    "schema": "ipfs_accelerate_py/agent-supervisor/aseh-human-intervention-policy@1",
    "maximum_human_interventions": 1,
    "escalation_required_for": ["human_escalated"],
    "shadow_candidate_cannot_page_human": True,
}

_ESTIMATOR = CalibratedTokenEstimator()


class ContextPackBenchmarkError(ValueError):
    """Closed ContextPack benchmark contract violation."""


class UnequalControlError(ContextPackBenchmarkError):
    """Arms do not share identical required controls."""


def _text(value: Any, *, name: str, maximum: int = 512) -> str:
    if not isinstance(value, str):
        raise ContextPackBenchmarkError(f"{name} must be text")
    result = value.strip()
    if not result:
        raise ContextPackBenchmarkError(f"{name} must not be empty")
    encoded = result.encode("utf-8")
    if b"\x00" in encoded or len(encoded) > maximum:
        raise ContextPackBenchmarkError(f"{name} is unsafe or too large")
    return result


def _int(
    value: Any,
    *,
    name: str,
    minimum: int = 0,
    maximum: int = 10**18,
) -> int:
    if type(value) is not int:
        raise ContextPackBenchmarkError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ContextPackBenchmarkError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContextPackBenchmarkError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ContextPackBenchmarkError(f"{name} keys must be strings")
    return {str(key): item for key, item in value.items()}


def _reject_floats(value: Any, *, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) is int:
        return
    if isinstance(value, float):
        raise ContextPackBenchmarkError(
            f"{path}: canonical contracts cannot contain floats"
        )
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_floats(child, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_floats(child, path=f"{path}[{index}]")
        return
    raise ContextPackBenchmarkError(f"{path}: unsupported value {type(value).__name__}")


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


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _median_int(values: Sequence[int]) -> int:
    if not values:
        raise ContextPackBenchmarkError("median is unavailable for an empty population")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) // 2


def _mean_int(values: Sequence[int]) -> int:
    if not values:
        raise ContextPackBenchmarkError("mean is unavailable for an empty population")
    return sum(values) // len(values)


def _ratio_millionths(numerator: int, denominator: int) -> int | None:
    if denominator <= 0:
        return None
    return (numerator * MILLIONTHS) // denominator


def _precision_recall_millionths(
    retrieved: Sequence[str],
    relevant: Sequence[str],
) -> tuple[int, int]:
    hit, retrieved_count, relevant_count = expansion_precision_recall(retrieved, relevant)
    if retrieved_count == 0:
        precision = MILLIONTHS if relevant_count == 0 else 0
    else:
        precision = (hit * MILLIONTHS) // retrieved_count
    if relevant_count == 0:
        recall = MILLIONTHS
    else:
        recall = (hit * MILLIONTHS) // relevant_count
    return precision, recall


def load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContextPackBenchmarkError(f"{path.name} is unreadable") from exc
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


def _measured(value: int, *, unit: str) -> dict[str, Any]:
    return measured_quantity(
        _int(value, name="value"),
        unit=unit,
        sensor_id=SENSOR_ID,
    )


def _estimated(value: int, *, unit: str) -> dict[str, Any]:
    return estimated_quantity(
        _int(value, name="value"),
        unit=unit,
        estimator_id=SENSOR_ID,
        method="local_compute_model",
        price_snapshot_identity="unavailable",
    )


def _unavailable(reason: str) -> dict[str, str]:
    return unavailable(reason)


def envelope_tokens(record: Any) -> int:
    data = encode_context_pack_envelope(record.to_dict())
    return _int(_ESTIMATOR.estimate(data), name="tokens", minimum=1)


def envelope_bytes(record: Any) -> bytes:
    return encode_context_pack_envelope(record.to_dict())


def _freshness_bindings(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "file_and_symbol_identities": [],
        "schema_identities": [SCHEMA_IDENTITY],
        "toolchain_identities": [TOOLCHAIN],
        "environment_requirements": [ENVIRONMENT],
        "reusable_until_conditions": ["tree-unchanged"],
    }
    payload.update(overrides)
    return payload


def _policy_identity(fixture_id: str) -> str:
    return _cid(f"aseh-035-policy:{fixture_id}")


def _prefix_dep(fixture_id: str) -> dict[str, object]:
    return {
        "symbol": "prefix_helper",
        "cid": _cid(f"{fixture_id}:prefix-helper"),
        "path": "a.py",
        "meaning": "unaffected prefix helper",
    }


def _named_dep(fixture_id: str, *, kind: str, name: str) -> dict[str, object]:
    path = f"{name}.py" if kind == "symbol" else f"{kind}s/{name}"
    return {
        "kind": kind,
        "name": name,
        "cid": _cid(f"{fixture_id}:{kind}:{name}"),
        "path": path,
        "meaning": f"named missing {kind}",
    }


def _unused_dep(fixture_id: str) -> dict[str, object]:
    return _named_dep(fixture_id, kind="symbol", name="unused-whole-repo")


def _pack_kwargs(fixture_id: str, **overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "repository_state_cid": _cid(f"{fixture_id}:repo-state"),
        "task_id": f"{TASK_ID}:{fixture_id}",
        "target_source_cid": _cid(f"{fixture_id}:target"),
        "surrounding_source_cid": _cid(f"{fixture_id}:surround"),
        "test_source_cid": _cid(f"{fixture_id}:test"),
        "scanned_tree_oid": TREE_OID,
        "source_tree_oid": TREE_OID,
        "objective_identity": PACK_OBJECTIVE,
        "objective_revision": PACK_OBJECTIVE_REVISION,
        "policy_identity": _policy_identity(fixture_id),
        "freshness_bindings": _freshness_bindings(),
        "invalidation": {
            "invalidation_triggers": ["tree-changed", "policy-changed"],
            "reusable_until_conditions": ["tree-unchanged"],
        },
        "dependencies": [_prefix_dep(fixture_id)],
    }
    fields.update(overrides)
    return fields


def _stale_overrides(fixture_id: str, field: str) -> dict[str, object]:
    if field == "tree":
        return {
            "scanned_tree_oid": STALE_TREE_OID,
            "source_tree_oid": STALE_TREE_OID,
        }
    if field == "objective":
        return {"objective_revision": "stale-objective-revision"}
    if field == "policy":
        return {"policy_identity": _cid(f"{fixture_id}:stale-policy")}
    if field == "interface":
        return {
            "freshness_bindings": _freshness_bindings(
                schema_identities=["stale.interface@1"]
            )
        }
    if field == "toolchain":
        return {
            "freshness_bindings": _freshness_bindings(toolchain_identities=["pypy"])
        }
    if field == "environment":
        return {
            "freshness_bindings": _freshness_bindings(
                environment_requirements=["PATH=/tmp/user-writable"]
            )
        }
    raise ContextPackBenchmarkError(f"unknown stale field {field}")


def generate_fixture_recipes() -> tuple[dict[str, Any], ...]:
    recipes: list[dict[str, Any]] = []
    index = 0
    for scenario in ("eligible_reuse", "incremental_expansion"):
        for variant in range(4):
            index += 1
            recipes.append(
                admit_recipe(
                    {
                        "fixture_id": f"aseh-cp{index:02d}",
                        "live": False,
                        "population_kind": "hermetic_development",
                        "scenario": scenario,
                        "seed": 35_000 + index,
                        "variant": variant,
                    }
                )
            )
    for kind in CRITICAL_KINDS:
        index += 1
        recipes.append(
            admit_recipe(
                {
                    "critical_kind": kind,
                    "fixture_id": f"aseh-cp{index:02d}",
                    "live": False,
                    "population_kind": "hermetic_development",
                    "scenario": "critical_omission",
                    "seed": 35_000 + index,
                    "variant": CRITICAL_KINDS.index(kind),
                }
            )
        )
    for field in STALE_FIELDS:
        index += 1
        recipes.append(
            admit_recipe(
                {
                    "fixture_id": f"aseh-cp{index:02d}",
                    "live": False,
                    "population_kind": "hermetic_development",
                    "scenario": "stale_pack",
                    "seed": 35_000 + index,
                    "stale_field": field,
                    "variant": STALE_FIELDS.index(field),
                }
            )
        )
    for scenario, count in (("fixture_as_live", 2), ("incomplete_evidence", 2)):
        for variant in range(count):
            index += 1
            recipes.append(
                admit_recipe(
                    {
                        "fixture_id": f"aseh-cp{index:02d}",
                        "live": False,
                        "population_kind": "hermetic_development",
                        "scenario": scenario,
                        "seed": 35_000 + index,
                        "variant": variant,
                    }
                )
            )
    if not recipes:
        raise ContextPackBenchmarkError("recipe catalogue is empty")
    return tuple(recipes)


def admit_recipe(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = _mapping(payload, name="recipe")
    fixture_id = _text(data.get("fixture_id"), name="fixture_id", maximum=64)
    scenario = _text(data.get("scenario"), name="scenario", maximum=64)
    if scenario not in SCENARIO_KINDS:
        raise ContextPackBenchmarkError(f"{fixture_id}: scenario is not a closed value")
    if data.get("live") is not False:
        raise ContextPackBenchmarkError(f"{fixture_id}: hermetic fixtures cannot be live")
    if data.get("population_kind") != "hermetic_development":
        raise ContextPackBenchmarkError(f"{fixture_id}: population_kind must be hermetic")
    recipe: dict[str, Any] = {
        "fixture_id": fixture_id,
        "live": False,
        "population_kind": "hermetic_development",
        "scenario": scenario,
        "seed": _int(data.get("seed"), name="seed", minimum=1),
        "variant": _int(data.get("variant"), name="variant"),
    }
    if scenario == "stale_pack":
        field = _text(data.get("stale_field"), name="stale_field", maximum=32)
        if field not in STALE_FIELDS:
            raise ContextPackBenchmarkError(f"{fixture_id}: stale_field is not closed")
        recipe["stale_field"] = field
    if scenario == "critical_omission":
        kind = _text(data.get("critical_kind"), name="critical_kind", maximum=32)
        if kind not in CRITICAL_KINDS:
            raise ContextPackBenchmarkError(f"{fixture_id}: critical_kind is not closed")
        recipe["critical_kind"] = kind
    encoded = compact_json(recipe)
    if len(encoded.encode("utf-8")) > MAX_SERIALIZED_RECIPE_BYTES:
        raise ContextPackBenchmarkError(f"{fixture_id}: recipe exceeds the serialized bound")
    return canonical_object(recipe)


def recipe_identity(recipe: Mapping[str, Any]) -> str:
    return content_identity(admit_recipe(recipe))


def shared_equal_controls(
    *,
    task_inputs: str,
    acceptance_tests: str | None = None,
    available_providers_and_models: str | None = None,
    price_accounting: str | None = None,
    resource_limits: str | None = None,
    human_intervention_policy: str | None = None,
    maximum_retries: int = 3,
    repository_revision: str = REPOSITORY_TREE,
    objective: str = OBJECTIVE_ID,
) -> dict[str, Any]:
    controls = {
        "acceptance_tests": acceptance_tests or content_identity(ACCEPTANCE_TESTS),
        "available_providers_and_models": available_providers_and_models
        or load_identity(PROVIDER_CONFIG_PATH, fallback_name="providers"),
        "human_intervention_policy": human_intervention_policy
        or content_identity(HUMAN_INTERVENTION_POLICY),
        "maximum_retries": _int(maximum_retries, name="maximum_retries", maximum=32),
        "objective": _text(objective, name="objective", maximum=64),
        "price_accounting": price_accounting
        or load_identity(PRICE_SNAPSHOT_PATH, fallback_name="price"),
        "repository_revision": _text(
            repository_revision, name="repository_revision", maximum=40
        ),
        "resource_limits": resource_limits or content_identity(RESOURCE_LIMITS),
        "task_inputs": _text(task_inputs, name="task_inputs", maximum=128),
    }
    if set(controls) != set(EQUAL_CONTROL_FIELDS):
        raise ContextPackBenchmarkError("equal controls drifted from the closed field set")
    return controls


def require_equal_controls(
    arm_controls: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
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


def _store_pack(
    store: Any,
    record: Any,
    *,
    current: bool = False,
    cache_key: str | None = None,
) -> tuple[Any, bytes]:
    data = envelope_bytes(record)
    if cache_key is not None:
        reference = store.put_candidate(data, cache_key=cache_key)
    else:
        reference = store.put_verified_bytes(data)
    if current:
        pointer = store.current_root()
        if pointer is None:
            store.compare_and_swap_current_root(new_cid=reference.cid, generation=0)
        else:
            store.compare_and_swap_current_root(
                new_cid=reference.cid,
                expected_parent_cid=pointer.seal_cid,
                generation=pointer.generation + 1,
                kind=reference.kind,
            )
    return reference, data


def _metric_bundle(
    *,
    eligible_reuse: int | None,
    tokens_before: int | None,
    tokens_after: int | None,
    expansion_precision: int | None,
    expansion_recall: int | None,
    critical_omission_detection: int | None,
    stale_rejection: int | None,
    build_cost: int | None,
    retrieval_cost: int | None,
    audit_overhead: int | None,
    net_provider_cost_effect: int | None,
    expansion_reason: str = "not_applicable",
    omission_reason: str = "not_applicable",
    stale_reason: str = "not_applicable",
    reuse_reason: str = "not_applicable",
    token_reason: str = "not_applicable",
    cost_reason: str = "not_applicable",
    net_reason: str = "not_reported",
    net_estimated: bool = True,
) -> dict[str, Any]:
    def count_or_unavail(value: int | None, reason: str) -> dict[str, Any]:
        if value is None:
            return _unavailable(reason)
        return _measured(value, unit="count")

    def tokens_or_unavail(value: int | None) -> dict[str, Any]:
        if value is None:
            return _unavailable(token_reason)
        return _measured(value, unit="tokens")

    def ratio_or_unavail(value: int | None) -> dict[str, Any]:
        if value is None:
            return _unavailable(expansion_reason)
        return _measured(value, unit="ratio_millionths")

    def cost_or_unavail(value: int | None) -> dict[str, Any]:
        if value is None:
            return _unavailable(cost_reason)
        return _measured(value, unit="microusd")

    if net_provider_cost_effect is None:
        net_node: dict[str, Any] = _unavailable(net_reason)
    else:
        savings = net_provider_cost_effect if net_provider_cost_effect > 0 else 0
        if net_estimated:
            net_node = _estimated(savings, unit="microusd")
        else:
            net_node = _measured(savings, unit="microusd")
    return {
        "eligible_reuse": count_or_unavail(eligible_reuse, reuse_reason),
        "context_tokens_before": tokens_or_unavail(tokens_before),
        "context_tokens_after": tokens_or_unavail(tokens_after),
        "expansion_precision": ratio_or_unavail(expansion_precision),
        "expansion_recall": ratio_or_unavail(expansion_recall),
        "critical_omission_detection": count_or_unavail(
            critical_omission_detection, omission_reason
        ),
        "stale_rejection": count_or_unavail(stale_rejection, stale_reason),
        "build_cost": cost_or_unavail(build_cost),
        "retrieval_cost": cost_or_unavail(retrieval_cost),
        "audit_overhead": cost_or_unavail(audit_overhead),
        "net_provider_cost_effect": net_node,
        "provider_reported_charge": _unavailable("not_reported"),
    }


def _observation(
    *,
    recipe: Mapping[str, Any],
    arm_id: str,
    disposition: str,
    metrics: Mapping[str, Any],
    reused: bool,
    reuse_eligible: bool,
    critical_omission_seeded: bool,
    critical_omission_accepted: bool,
    stale_seeded: bool,
    stale_admitted: bool,
    audit_checks: int,
    retrieved_items: int,
    byte_length: int,
    decisions: Sequence[str],
) -> dict[str, Any]:
    if arm_id not in PAIRED_ARMS:
        raise ContextPackBenchmarkError(f"{arm_id} is not a closed paired arm")
    if "audit_overhead" not in metrics:
        raise ContextPackBenchmarkError("missing audit cost")
    audit = metrics["audit_overhead"]
    if audit.get("truth_state") != "measured":
        raise ContextPackBenchmarkError("missing audit cost")
    if _int(audit.get("value"), name="audit_overhead") <= 0:
        raise ContextPackBenchmarkError("missing audit cost")
    missing = [name for name in REQUIRED_METRICS if name not in metrics]
    if missing:
        raise ContextPackBenchmarkError(f"missing required metrics: {missing}")
    payload = {
        "arm_id": arm_id,
        "audit_checks": _int(audit_checks, name="audit_checks", minimum=1),
        "byte_length": _int(byte_length, name="byte_length"),
        "critical_omission_accepted": critical_omission_accepted,
        "critical_omission_seeded": critical_omission_seeded,
        "decisions": list(decisions),
        "disposition": _text(disposition, name="disposition", maximum=32),
        "fixture_id": recipe["fixture_id"],
        "live": False,
        "metrics": dict(metrics),
        "population_kind": "hermetic_development",
        "retrieved_items": _int(retrieved_items, name="retrieved_items"),
        "reuse_eligible": reuse_eligible,
        "reused": reused,
        "scenario": recipe["scenario"],
        "simulated_as_live": False,
        "stale_admitted": stale_admitted,
        "stale_seeded": stale_seeded,
        "truth_state": "measured",
    }
    return canonical_object(payload)


def _net_effect(tokens_before: int, tokens_after: int) -> int:
    """Estimated provider-cost savings in microusd.

    Efficiency receipts reject signed integers, so token growth reports as
    zero savings rather than a negative quantity.
    """
    before = _int(tokens_before, name="tokens_before")
    after = _int(tokens_after, name="tokens_after")
    if after >= before:
        return 0
    return (before - after) * TOKEN_MICROUSD


def _audit_cost(checks: int) -> int:
    return max(1, checks) * AUDIT_COST_PER_CHECK


def _build_cost(byte_length: int) -> int:
    return max(0, byte_length) * BUILD_COST_PER_BYTE


def _retrieval_cost(items: int, byte_length: int = 0) -> int:
    count = _int(items, name="retrieved_items")
    size = _int(byte_length, name="retrieved_bytes")
    return count * RETRIEVAL_COST_PER_ITEM + size


def _build_minimal(fixture_id: str, **overrides: object) -> Any:
    return build_minimal_semantic_pack(**_pack_kwargs(fixture_id, **overrides))


def _build_bloated(fixture_id: str, **overrides: object) -> Any:
    helper = _cid(f"{fixture_id}:symbol:helper")
    unused = tuple(
        _cid(f"{fixture_id}:unused-capsule:{index}") for index in range(16)
    )
    fields = _pack_kwargs(fixture_id, **overrides)
    fields.pop("dependencies", None)
    if "scope" not in fields:
        fields["scope"] = {
            "affected_files": [f"unused/{index:02d}.py" for index in range(16)],
            "affected_symbols": [f"unused_{index}" for index in range(16)],
            "dependency_cone": [f"unused_{index}" for index in range(16)],
            "semantic_diff_summary": (
                "unused context excluded from the current minimal pack"
            ),
        }
    return build_context_pack(
        capsule_cids=(helper, *unused),
        **fields,
    )


def _select_or_reject(store: Any, current: CurrentPackIdentity) -> Any:
    return select_current_minimal_pack(store, current, require_selection=False)


def _run_reuse_arm(
    recipe: Mapping[str, Any],
    *,
    arm_id: str,
    store: Any,
    minimal: Any,
    bloated: Any,
    current: CurrentPackIdentity,
) -> dict[str, Any]:
    before = envelope_tokens(bloated)
    if arm_id == "direct_minimal_orchestration_baseline":
        after = envelope_tokens(bloated)
        payload_bytes = len(envelope_bytes(bloated))
        retrieved = len(tuple(bloated.capsule_cids))
        checks = 2
        metrics = _metric_bundle(
            eligible_reuse=0,
            tokens_before=before,
            tokens_after=after,
            expansion_precision=None,
            expansion_recall=None,
            critical_omission_detection=None,
            stale_rejection=None,
            build_cost=_build_cost(payload_bytes),
            retrieval_cost=_retrieval_cost(retrieved, payload_bytes),
            audit_overhead=_audit_cost(checks),
            net_provider_cost_effect=_net_effect(before, after),
        )
        return _observation(
            recipe=recipe,
            arm_id=arm_id,
            disposition="selected",
            metrics=metrics,
            reused=False,
            reuse_eligible=True,
            critical_omission_seeded=False,
            critical_omission_accepted=False,
            stale_seeded=False,
            stale_admitted=False,
            audit_checks=checks,
            retrieved_items=retrieved,
            byte_length=payload_bytes,
            decisions=["direct:rebuild_bloated", "no_candidate_context_pack_optimization"],
        )
    selection = _select_or_reject(store, current)
    if selection.selected is None or selection.admission.reused is not True:
        raise ContextPackBenchmarkError(
            f"{recipe['fixture_id']}: eligible current-tree pack was not reused"
        )
    after = envelope_tokens(minimal)
    payload_bytes = len(envelope_bytes(minimal))
    retrieved = 1
    checks = 4
    metrics = _metric_bundle(
        eligible_reuse=1,
        tokens_before=before,
        tokens_after=after,
        expansion_precision=None,
        expansion_recall=None,
        critical_omission_detection=None,
        stale_rejection=None,
        build_cost=0,
        retrieval_cost=_retrieval_cost(retrieved, payload_bytes),
        audit_overhead=_audit_cost(checks),
        net_provider_cost_effect=_net_effect(before, after),
    )
    return _observation(
        recipe=recipe,
        arm_id=arm_id,
        disposition="reuse",
        metrics=metrics,
        reused=True,
        reuse_eligible=True,
        critical_omission_seeded=False,
        critical_omission_accepted=False,
        stale_seeded=False,
        stale_admitted=False,
        audit_checks=checks,
        retrieved_items=retrieved,
        byte_length=payload_bytes,
        decisions=list(selection.admission.decisions),
    )


def _run_expansion_arm(
    recipe: Mapping[str, Any],
    *,
    arm_id: str,
    parent: Any,
    fixture_id: str,
    variant: int,
) -> dict[str, Any]:
    helper = _named_dep(fixture_id, kind="symbol", name="helper")
    unused = _unused_dep(fixture_id)
    catalog = [helper, unused]
    named = ["symbol:helper"]
    changed = ["c.py"] if variant == 2 else []
    ordered = ["a.py", "b.py", "c.py", "d.py"] if variant == 2 else ["a.py"]
    bloated = _build_bloated(fixture_id)
    before = envelope_tokens(bloated)
    if arm_id == "direct_minimal_orchestration_baseline":
        retrieved = ["symbol:helper", "symbol:unused-whole-repo"]
        relevant = ["symbol:helper"] if variant != 2 else ["c.py"]
        if variant == 2:
            retrieved = ["a.py", "b.py", "c.py", "d.py", "symbol:unused-whole-repo"]
            relevant = ["c.py", "d.py"]
        precision, recall = _precision_recall_millionths(retrieved, relevant)
        after = envelope_tokens(bloated)
        payload_bytes = len(envelope_bytes(bloated))
        checks = 2
        metrics = _metric_bundle(
            eligible_reuse=0,
            tokens_before=before,
            tokens_after=after,
            expansion_precision=precision,
            expansion_recall=recall,
            critical_omission_detection=None,
            stale_rejection=None,
            build_cost=_build_cost(payload_bytes),
            retrieval_cost=_retrieval_cost(len(retrieved), payload_bytes),
            audit_overhead=_audit_cost(checks),
            net_provider_cost_effect=_net_effect(before, after),
        )
        return _observation(
            recipe=recipe,
            arm_id=arm_id,
            disposition="selected",
            metrics=metrics,
            reused=False,
            reuse_eligible=False,
            critical_omission_seeded=False,
            critical_omission_accepted=False,
            stale_seeded=False,
            stale_admitted=False,
            audit_checks=checks,
            retrieved_items=len(retrieved),
            byte_length=payload_bytes,
            decisions=["direct:whole_context_rebuild"],
        )
    if arm_id == "sealed_current_supervisor_baseline":
        rebuilt = _build_minimal(
            fixture_id,
            dependencies=[
                _prefix_dep(fixture_id),
                {
                    "symbol": "helper",
                    "cid": helper["cid"],
                    "path": helper["path"],
                    "meaning": helper["meaning"],
                },
            ],
        )
        retrieved = ["symbol:helper", "prefix_helper"]
        relevant = ["symbol:helper", "prefix_helper"]
        if variant == 2:
            retrieved = ["a.py", "b.py", "c.py", "d.py"]
            relevant = ["c.py", "d.py"]
        precision, recall = _precision_recall_millionths(retrieved, relevant)
        after = envelope_tokens(rebuilt)
        payload_bytes = len(envelope_bytes(rebuilt))
        checks = 3
        metrics = _metric_bundle(
            eligible_reuse=0,
            tokens_before=before,
            tokens_after=after,
            expansion_precision=precision,
            expansion_recall=recall,
            critical_omission_detection=None,
            stale_rejection=None,
            build_cost=_build_cost(payload_bytes),
            retrieval_cost=_retrieval_cost(len(retrieved), payload_bytes),
            audit_overhead=_audit_cost(checks),
            net_provider_cost_effect=_net_effect(before, after),
        )
        return _observation(
            recipe=recipe,
            arm_id=arm_id,
            disposition="selected",
            metrics=metrics,
            reused=False,
            reuse_eligible=False,
            critical_omission_seeded=False,
            critical_omission_accepted=False,
            stale_seeded=False,
            stale_admitted=False,
            audit_checks=checks,
            retrieved_items=len(retrieved),
            byte_length=payload_bytes,
            decisions=["sealed:full_minimal_rebuild"],
        )
    tree = STALE_TREE_OID if variant == 2 else TREE_OID
    result = expand_incremental_pack(
        parent=parent,
        scanned_tree_oid=tree,
        named_missing=named if variant != 2 else [],
        catalog=catalog,
        changed_files=changed,
        ordered_files=ordered,
    )
    precision, recall = _precision_recall_millionths(result.retrieved, result.relevant)
    after = envelope_tokens(result.pack)
    payload_bytes = len(envelope_bytes(result.pack))
    checks = 4
    metrics = _metric_bundle(
        eligible_reuse=0,
        tokens_before=before,
        tokens_after=after,
        expansion_precision=precision,
        expansion_recall=recall,
        critical_omission_detection=None,
        stale_rejection=None,
        build_cost=_build_cost(payload_bytes),
        retrieval_cost=_retrieval_cost(result.retrieved_count, payload_bytes),
        audit_overhead=_audit_cost(checks),
        net_provider_cost_effect=_net_effect(before, after),
    )
    return _observation(
        recipe=recipe,
        arm_id=arm_id,
        disposition="expanded",
        metrics=metrics,
        reused=False,
        reuse_eligible=False,
        critical_omission_seeded=False,
        critical_omission_accepted=False,
        stale_seeded=False,
        stale_admitted=False,
        audit_checks=checks,
        retrieved_items=result.retrieved_count,
        byte_length=payload_bytes,
        decisions=["candidate:incremental_named_missing_or_suffix"],
    )


def _run_critical_omission_arm(
    recipe: Mapping[str, Any],
    *,
    arm_id: str,
    parent: Any,
    fixture_id: str,
    kind: str,
) -> dict[str, Any]:
    catalog = [_named_dep(fixture_id, kind=kind, name="helper")]
    before = envelope_tokens(parent)
    detected = False
    accepted = False
    try:
        expand_incremental_pack(
            parent=parent,
            scanned_tree_oid=STALE_TREE_OID,
            changed_files=["c.py"],
            ordered_files=["a.py", "b.py", "c.py", "d.py"],
            catalog=catalog,
            critical_dependencies=["helper"],
        )
        accepted = True
    except CriticalOmissionError:
        detected = True
    except CompletenessFailure:
        detected = True
    if accepted or not detected:
        raise ContextPackBenchmarkError(
            f"{recipe['fixture_id']}: seeded critical omission was not rejected"
        )
    checks = 3
    payload_bytes = len(envelope_bytes(parent))
    metrics = _metric_bundle(
        eligible_reuse=0,
        tokens_before=before,
        tokens_after=None,
        expansion_precision=None,
        expansion_recall=None,
        critical_omission_detection=1,
        stale_rejection=None,
        build_cost=_build_cost(0),
        retrieval_cost=_retrieval_cost(0),
        audit_overhead=_audit_cost(checks),
        net_provider_cost_effect=None,
        expansion_reason="collection_failed",
        token_reason="not_applicable",
        net_reason="not_applicable",
    )
    return _observation(
        recipe=recipe,
        arm_id=arm_id,
        disposition="rejected",
        metrics=metrics,
        reused=False,
        reuse_eligible=False,
        critical_omission_seeded=True,
        critical_omission_accepted=False,
        stale_seeded=False,
        stale_admitted=False,
        audit_checks=checks,
        retrieved_items=0,
        byte_length=payload_bytes,
        decisions=["reject:critical_omission"],
    )


def _run_stale_arm(
    recipe: Mapping[str, Any],
    *,
    arm_id: str,
    store: Any,
    fresh: Any,
    current: CurrentPackIdentity,
) -> dict[str, Any]:
    before = envelope_tokens(fresh)
    selection = _select_or_reject(store, current)
    admitted = (
        selection.selected is not None
        and selection.admission.disposition in {"reuse", "selected"}
    )
    if admitted or selection.admission.reused is True:
        raise ContextPackBenchmarkError(
            f"{recipe['fixture_id']}: seeded stale pack was admitted"
        )
    if selection.admission.disposition != "rejected":
        raise ContextPackBenchmarkError(
            f"{recipe['fixture_id']}: seeded stale pack was not rejected"
        )
    checks = 4
    payload_bytes = len(envelope_bytes(fresh))
    metrics = _metric_bundle(
        eligible_reuse=0,
        tokens_before=before,
        tokens_after=None,
        expansion_precision=None,
        expansion_recall=None,
        critical_omission_detection=None,
        stale_rejection=1,
        build_cost=_build_cost(0),
        retrieval_cost=_retrieval_cost(1, payload_bytes),
        audit_overhead=_audit_cost(checks),
        net_provider_cost_effect=None,
        token_reason="not_applicable",
        net_reason="not_applicable",
    )
    return _observation(
        recipe=recipe,
        arm_id=arm_id,
        disposition="rejected",
        metrics=metrics,
        reused=False,
        reuse_eligible=False,
        critical_omission_seeded=False,
        critical_omission_accepted=False,
        stale_seeded=True,
        stale_admitted=False,
        audit_checks=checks,
        retrieved_items=1,
        byte_length=payload_bytes,
        decisions=list(selection.admission.decisions),
    )


def _run_fixture_as_live_arm(
    recipe: Mapping[str, Any],
    *,
    arm_id: str,
    store: Any,
    live_pack: Any,
    current: CurrentPackIdentity,
) -> dict[str, Any]:
    before = envelope_tokens(live_pack)
    selection = _select_or_reject(store, current)
    if selection.selected is not None or selection.admission.reused is True:
        raise ContextPackBenchmarkError(
            f"{recipe['fixture_id']}: fixture pack masqueraded as live"
        )
    masquerade = any(
        "fixture_as_live" in item.masquerade_reasons for item in selection.invalidated
    )
    if not masquerade and not selection.invalidated:
        raise ContextPackBenchmarkError(
            f"{recipe['fixture_id']}: fixture-as-live was not rejected"
        )
    checks = 4
    payload_bytes = len(envelope_bytes(live_pack))
    metrics = _metric_bundle(
        eligible_reuse=0,
        tokens_before=before,
        tokens_after=None,
        expansion_precision=None,
        expansion_recall=None,
        critical_omission_detection=None,
        stale_rejection=None,
        build_cost=_build_cost(0),
        retrieval_cost=_retrieval_cost(1, payload_bytes),
        audit_overhead=_audit_cost(checks),
        net_provider_cost_effect=None,
        token_reason="not_applicable",
        net_reason="not_applicable",
        stale_reason="not_applicable",
    )
    return _observation(
        recipe=recipe,
        arm_id=arm_id,
        disposition="rejected",
        metrics=metrics,
        reused=False,
        reuse_eligible=False,
        critical_omission_seeded=False,
        critical_omission_accepted=False,
        stale_seeded=False,
        stale_admitted=False,
        audit_checks=checks,
        retrieved_items=1,
        byte_length=payload_bytes,
        decisions=list(selection.admission.decisions),
    )


def _run_incomplete_arm(
    recipe: Mapping[str, Any],
    *,
    arm_id: str,
    store: Any,
    incomplete: Any,
    current: CurrentPackIdentity,
) -> dict[str, Any]:
    before = envelope_tokens(incomplete)
    selection = _select_or_reject(store, current)
    if selection.selected is not None or selection.admission.reused is True:
        raise ContextPackBenchmarkError(
            f"{recipe['fixture_id']}: incomplete pack was reused"
        )
    checks = 4
    payload_bytes = len(envelope_bytes(incomplete))
    metrics = _metric_bundle(
        eligible_reuse=0,
        tokens_before=before,
        tokens_after=None,
        expansion_precision=None,
        expansion_recall=None,
        critical_omission_detection=None,
        stale_rejection=None,
        build_cost=_build_cost(0),
        retrieval_cost=_retrieval_cost(1, payload_bytes),
        audit_overhead=_audit_cost(checks),
        net_provider_cost_effect=None,
        token_reason="not_applicable",
        net_reason="not_applicable",
    )
    return _observation(
        recipe=recipe,
        arm_id=arm_id,
        disposition="rejected",
        metrics=metrics,
        reused=False,
        reuse_eligible=False,
        critical_omission_seeded=False,
        critical_omission_accepted=False,
        stale_seeded=False,
        stale_admitted=False,
        audit_checks=checks,
        retrieved_items=1,
        byte_length=payload_bytes,
        decisions=list(selection.admission.decisions),
    )


def run_fixture_arms(
    recipe: Mapping[str, Any],
    *,
    controls: Mapping[str, Any],
    store_root: Path,
) -> dict[str, Any]:
    admitted = admit_recipe(recipe)
    fixture_id = admitted["fixture_id"]
    scenario = admitted["scenario"]
    fixture_dir = Path(store_root) / fixture_id
    fixture_dir.mkdir(parents=True, exist_ok=True)
    store = open_context_pack_store(fixture_dir)
    try:
        observations: dict[str, dict[str, Any]] = {}
        if scenario == "eligible_reuse":
            minimal = _build_minimal(fixture_id)
            bloated = _build_bloated(fixture_id)
            current = CurrentPackIdentity.from_envelope(minimal.to_dict())
            _store_pack(store, bloated, cache_key="pack:bloated")
            _store_pack(store, minimal, current=True, cache_key="pack:minimal")
            for arm_id in PAIRED_ARMS:
                observations[arm_id] = _run_reuse_arm(
                    admitted,
                    arm_id=arm_id,
                    store=store,
                    minimal=minimal,
                    bloated=bloated,
                    current=current,
                )
        elif scenario == "incremental_expansion":
            parent = _build_minimal(fixture_id)
            for arm_id in PAIRED_ARMS:
                observations[arm_id] = _run_expansion_arm(
                    admitted,
                    arm_id=arm_id,
                    parent=parent,
                    fixture_id=fixture_id,
                    variant=admitted["variant"],
                )
        elif scenario == "critical_omission":
            parent = _build_minimal(fixture_id)
            kind = admitted["critical_kind"]
            for arm_id in PAIRED_ARMS:
                observations[arm_id] = _run_critical_omission_arm(
                    admitted,
                    arm_id=arm_id,
                    parent=parent,
                    fixture_id=fixture_id,
                    kind=kind,
                )
        elif scenario == "stale_pack":
            fresh = _build_minimal(fixture_id)
            stale = _build_minimal(
                fixture_id, **_stale_overrides(fixture_id, admitted["stale_field"])
            )
            current = CurrentPackIdentity.from_envelope(fresh.to_dict())
            _store_pack(
                store,
                stale,
                current=True,
                cache_key=f"pack:stale-{admitted['stale_field']}",
            )
            for arm_id in PAIRED_ARMS:
                observations[arm_id] = _run_stale_arm(
                    admitted,
                    arm_id=arm_id,
                    store=store,
                    fresh=fresh,
                    current=current,
                )
        elif scenario == "fixture_as_live":
            live_pack = _build_minimal(fixture_id)
            fixture_pack = _build_minimal(
                fixture_id,
                identity_kind="fixture",
                evidence_kind="fixture",
                execution_mode="simulated",
            )
            current = CurrentPackIdentity.from_envelope(live_pack.to_dict())
            _store_pack(store, fixture_pack, current=True, cache_key="pack:fixture")
            for arm_id in PAIRED_ARMS:
                observations[arm_id] = _run_fixture_as_live_arm(
                    admitted,
                    arm_id=arm_id,
                    store=store,
                    live_pack=live_pack,
                    current=current,
                )
        elif scenario == "incomplete_evidence":
            incomplete = _build_minimal(
                fixture_id, missing_evidence=["named-missing-contract"]
            )
            current = CurrentPackIdentity.from_envelope(incomplete.to_dict())
            _store_pack(store, incomplete, current=True, cache_key="pack:incomplete")
            for arm_id in PAIRED_ARMS:
                observations[arm_id] = _run_incomplete_arm(
                    admitted,
                    arm_id=arm_id,
                    store=store,
                    incomplete=incomplete,
                    current=current,
                )
        else:
            raise ContextPackBenchmarkError(f"{fixture_id}: unsupported scenario")
        return pair_fixture_observations(observations, controls=controls)
    finally:
        store.close()


def pair_fixture_observations(
    observations: Mapping[str, Mapping[str, Any]],
    *,
    controls: Mapping[str, Any],
) -> dict[str, Any]:
    arms = _mapping(observations, name="observations")
    missing = [arm for arm in PAIRED_ARMS if arm not in arms]
    if missing:
        raise ContextPackBenchmarkError(f"missing paired inputs for arms: {missing}")
    fixture_ids = {arms[arm]["fixture_id"] for arm in PAIRED_ARMS}
    if len(fixture_ids) != 1:
        raise ContextPackBenchmarkError("paired inputs must share one fixture_id")
    if any(arms[arm].get("live") is True for arm in PAIRED_ARMS):
        raise ContextPackBenchmarkError("hermetic pairing cannot mark an arm live")
    if any(arms[arm].get("simulated_as_live") is True for arm in PAIRED_ARMS):
        raise ContextPackBenchmarkError("fixture-as-live label")
    return {
        "controls": {field: controls[field] for field in EQUAL_CONTROL_FIELDS},
        "fixture_id": next(iter(fixture_ids)),
        "live": False,
        "observations": {arm: canonical_object(arms[arm]) for arm in PAIRED_ARMS},
        "scenario": arms[PAIRED_ARMS[0]]["scenario"],
    }


def _metric_int(node: Mapping[str, Any], *, name: str) -> int | None:
    payload = _mapping(node, name=name)
    state = payload.get("truth_state")
    if state == "unavailable":
        if any(key in payload for key in ("value", "count", "unit", "sensor_id")):
            raise ContextPackBenchmarkError(
                f"{name}: unavailable evidence cannot encode numeric zero"
            )
        return None
    if state in {"measured", "estimated"}:
        return _int(payload.get("value"), name=f"{name}.value")
    raise ContextPackBenchmarkError(f"{name}: unsupported truth_state {state!r}")


def _require_metrics(observation: Mapping[str, Any]) -> None:
    metrics = _mapping(observation.get("metrics"), name="metrics")
    missing = [name for name in REQUIRED_METRICS if name not in metrics]
    if missing:
        raise ContextPackBenchmarkError(f"missing required metrics: {missing}")
    audit = _metric_int(metrics["audit_overhead"], name="audit_overhead")
    if audit is None or audit <= 0:
        raise ContextPackBenchmarkError("missing audit cost")


def admit_campaign_result(result: Mapping[str, Any]) -> dict[str, Any]:
    payload = _mapping(result, name="campaign")
    paired = payload.get("paired_fixtures")
    if not isinstance(paired, Sequence) or not paired:
        raise ContextPackBenchmarkError("aggregate-only results")
    if payload.get("live") is True or payload.get("simulated_as_live") is True:
        raise ContextPackBenchmarkError("fixture-as-live label")
    if payload.get("hermetic_sufficient_for_production_promotion") is not False:
        raise ContextPackBenchmarkError("hermetic evidence cannot grant promotion")
    if payload.get("hermetic_sufficient_for_live_reuse") is not False:
        raise ContextPackBenchmarkError("hermetic evidence cannot grant live reuse")
    if payload.get("authority") is not False:
        raise ContextPackBenchmarkError("benchmark evidence is not authority")
    seeded_omissions = 0
    accepted_omissions = 0
    seeded_stale = 0
    admitted_stale = 0
    for item in paired:
        mapping = _mapping(item, name="paired_fixture")
        observations = _mapping(mapping.get("observations"), name="observations")
        if set(observations) != set(PAIRED_ARMS):
            raise ContextPackBenchmarkError("missing paired inputs")
        for arm_id in PAIRED_ARMS:
            observation = _mapping(observations[arm_id], name=arm_id)
            _require_metrics(observation)
            if observation.get("live") is True:
                raise ContextPackBenchmarkError("fixture-as-live label")
            if observation.get("critical_omission_seeded") is True:
                seeded_omissions += 1
                if observation.get("critical_omission_accepted") is True:
                    accepted_omissions += 1
                if observation.get("disposition") != "rejected":
                    raise ContextPackBenchmarkError(
                        "seeded critical omission was not rejected"
                    )
            if observation.get("stale_seeded") is True:
                seeded_stale += 1
                if observation.get("stale_admitted") is True:
                    admitted_stale += 1
                if observation.get("disposition") != "rejected":
                    raise ContextPackBenchmarkError("seeded stale pack was not rejected")
            if observation.get("reused") is True and observation.get("live") is True:
                raise ContextPackBenchmarkError("hermetic evidence cannot grant live reuse")
    if accepted_omissions:
        raise ContextPackBenchmarkError("accepted critical omission")
    if admitted_stale:
        raise ContextPackBenchmarkError("stale identity admission")
    if seeded_omissions <= 0:
        raise ContextPackBenchmarkError("critical omission cases were not seeded")
    if seeded_stale <= 0:
        raise ContextPackBenchmarkError("stale packs were not seeded")
    unavailable_fields = collect_unavailable_fields(payload)
    listed = payload.get("explicit_unavailable_fields")
    if not isinstance(listed, Sequence):
        raise ContextPackBenchmarkError("explicit unavailable fields must be retained")
    if tuple(listed) != unavailable_fields:
        raise ContextPackBenchmarkError("unavailable data was not retained")
    return canonical_object(payload)


def _aggregate_metric(
    paired: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    arm_id: str,
) -> dict[str, Any]:
    values: list[int] = []
    unavailable_count = 0
    for item in paired:
        observation = item["observations"][arm_id]
        node = observation["metrics"][metric]
        value = _metric_int(node, name=metric)
        if value is None:
            unavailable_count += 1
        else:
            values.append(value)
    if not values:
        node = _unavailable("not_yet_measured" if unavailable_count else "not_applicable")
        node["arm_id"] = arm_id
        node["metric"] = metric
        return node
    return {
        "arm_id": arm_id,
        "mean": _mean_int(values),
        "measured_count": len(values),
        "median": _median_int(values),
        "metric": metric,
        "truth_state": "measured" if metric != "net_provider_cost_effect" else "estimated",
        "unavailable_count": unavailable_count,
        "unit": observation["metrics"][metric].get("unit", "count"),
    }


def compute_campaign_aggregates(
    paired: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not paired:
        raise ContextPackBenchmarkError("aggregate-only results")
    candidate = "candidate_optimized_supervisor"
    eligible = 0
    reused = 0
    omission_seeded = 0
    omission_detected = 0
    stale_seeded = 0
    stale_rejected = 0
    audit_total = 0
    for item in paired:
        for arm_id in PAIRED_ARMS:
            observation = item["observations"][arm_id]
            if observation.get("reuse_eligible") is True:
                eligible += 1
                if observation.get("reused") is True:
                    reused += 1
            if observation.get("critical_omission_seeded") is True:
                omission_seeded += 1
                detected = _metric_int(
                    observation["metrics"]["critical_omission_detection"],
                    name="critical_omission_detection",
                )
                if detected == 1:
                    omission_detected += 1
            if observation.get("stale_seeded") is True:
                stale_seeded += 1
                rejected = _metric_int(
                    observation["metrics"]["stale_rejection"],
                    name="stale_rejection",
                )
                if rejected == 1:
                    stale_rejected += 1
            audit_total += _metric_int(
                observation["metrics"]["audit_overhead"], name="audit_overhead"
            ) or 0
    reuse_rate = _ratio_millionths(reused, eligible)
    detection_rate = _ratio_millionths(omission_detected, omission_seeded)
    stale_rate = _ratio_millionths(stale_rejected, stale_seeded)
    metrics = {
        name: {
            arm_id: _aggregate_metric(paired, name, arm_id=arm_id)
            for arm_id in PAIRED_ARMS
        }
        for name in REQUIRED_METRICS
    }
    return {
        "audit_overhead_total_microusd": audit_total,
        "critical_omission_detection_rate": (
            _measured(detection_rate, unit="ratio_millionths")
            if detection_rate is not None
            else _unavailable("not_yet_measured")
        ),
        "eligible_reuse_rate": (
            _measured(reuse_rate, unit="ratio_millionths")
            if reuse_rate is not None
            else _unavailable("not_yet_measured")
        ),
        "metrics": metrics,
        "primary_arm": candidate,
        "seeded_critical_omissions": omission_seeded,
        "seeded_critical_omissions_accepted": 0,
        "seeded_stale_packs": stale_seeded,
        "seeded_stale_packs_admitted": 0,
        "stale_rejection_rate": (
            _measured(stale_rate, unit="ratio_millionths")
            if stale_rate is not None
            else _unavailable("not_yet_measured")
        ),
    }


def run_paired_campaign(
    recipes: Sequence[Mapping[str, Any]] | None = None,
    *,
    controls: Mapping[str, Any] | None = None,
    arm_controls: Mapping[str, Mapping[str, Any]] | None = None,
    store_root: Path | None = None,
) -> dict[str, Any]:
    catalogue = tuple(admit_recipe(item) for item in (recipes or generate_fixture_recipes()))
    identities = [recipe_identity(item) for item in catalogue]
    if len(set(identities)) != len(identities):
        raise ContextPackBenchmarkError("fixtures must be unique by content identity")
    if len({item["fixture_id"] for item in catalogue}) != len(catalogue):
        raise ContextPackBenchmarkError("fixture_id values must be unique")
    vectors_identity = content_identity({"recipes": list(catalogue)})
    shared = dict(controls) if controls is not None else shared_equal_controls(
        task_inputs=vectors_identity
    )
    if arm_controls is None:
        require_equal_controls(controls_for_arms(shared))
    else:
        shared = require_equal_controls(arm_controls)
    owned_temp = None
    root = store_root
    if root is None:
        owned_temp = tempfile.TemporaryDirectory(prefix="aseh-035-")
        root = Path(owned_temp.name)
    try:
        paired = [
            run_fixture_arms(recipe, controls=shared, store_root=root)
            for recipe in catalogue
        ]
        aggregates = compute_campaign_aggregates(paired)
        negative = {
            "fixture_as_live": {
                "accepted": 0,
                "count": sum(1 for item in paired if item["scenario"] == "fixture_as_live"),
                "rejected": sum(
                    1 for item in paired if item["scenario"] == "fixture_as_live"
                ),
            },
            "seeded_critical_omissions": {
                "accepted": 0,
                "count": aggregates["seeded_critical_omissions"],
                "detected": aggregates["seeded_critical_omissions"],
                "rejected": aggregates["seeded_critical_omissions"],
            },
            "seeded_stale_packs": {
                "admitted": 0,
                "count": aggregates["seeded_stale_packs"],
                "rejected": aggregates["seeded_stale_packs"],
            },
        }
        result = {
            "aggregates": aggregates,
            "arms": list(PAIRED_ARMS),
            "authority": False,
            "controls": shared,
            "fixture_count": len(catalogue),
            "hermetic_sufficient_for_live_reuse": False,
            "hermetic_sufficient_for_production_promotion": False,
            "live": False,
            "negative_cases": negative,
            "paired_fixtures": paired,
            "population_kind": "hermetic_development",
            "populations": {
                "hermetic_development": {
                    "count": _measured(len(catalogue), unit="count"),
                    "status": "sealed",
                },
                "historical_exact_tree_replay": {
                    "count": _unavailable("not_sealed"),
                    "status": "unavailable",
                },
                "new_live_shadow_canary": {
                    "count": _unavailable("not_sealed"),
                    "enrollment_deadline": _unavailable("not_sealed"),
                    "status": "unavailable",
                },
            },
            "promotion_without_paired_campaign": False,
            "recipes": list(catalogue),
            "required_metrics": list(REQUIRED_METRICS),
            "simulated_as_live": False,
            "vectors_identity": vectors_identity,
        }
        result["explicit_unavailable_fields"] = list(collect_unavailable_fields(result))
        return admit_campaign_result(result)
    finally:
        if owned_temp is not None:
            owned_temp.cleanup()


def _compact_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "arm_id": observation["arm_id"],
        "disposition": observation["disposition"],
        "fixture_id": observation["fixture_id"],
        "live": False,
        "metrics": observation["metrics"],
        "reuse_eligible": observation["reuse_eligible"],
        "reused": observation["reused"],
        "scenario": observation["scenario"],
        "critical_omission_accepted": observation["critical_omission_accepted"],
        "critical_omission_seeded": observation["critical_omission_seeded"],
        "stale_admitted": observation["stale_admitted"],
        "stale_seeded": observation["stale_seeded"],
    }


def build_context_pack_manifest(result: Mapping[str, Any]) -> dict[str, Any]:
    admitted = admit_campaign_result(result)
    controls = admitted["controls"]
    compact_pairs = []
    for item in admitted["paired_fixtures"]:
        compact_pairs.append(
            {
                "fixture_id": item["fixture_id"],
                "live": False,
                "observations": {
                    arm_id: _compact_observation(item["observations"][arm_id])
                    for arm_id in PAIRED_ARMS
                },
                "scenario": item["scenario"],
            }
        )
    payload = {
        "aggregates": admitted["aggregates"],
        "arm_constraints": {
            "candidate_optimized_supervisor": list(CANDIDATE_ARM_CONSTRAINTS),
            "direct_minimal_orchestration_baseline": list(DIRECT_ARM_CONSTRAINTS),
            "sealed_current_supervisor_baseline": list(SEALED_CURRENT_ARM_CONSTRAINTS),
        },
        "arms": list(PAIRED_ARMS),
        "authority": False,
        "controls": {field: controls[field] for field in EQUAL_CONTROL_FIELDS},
        "environment_identity": load_identity(
            ENVIRONMENT_IDENTITY_PATH, fallback_name="environment"
        ),
        "equal_control_fields": list(EQUAL_CONTROL_FIELDS),
        "fixture_count": admitted["fixture_count"],
        "hermetic_sufficient_for_live_reuse": False,
        "hermetic_sufficient_for_production_promotion": False,
        "interface": BENCHMARK_INTERFACE,
        "live": False,
        "negative_cases": admitted["negative_cases"],
        "objective_id": OBJECTIVE_ID,
        "objective_revision": OBJECTIVE_REVISION,
        "paired_fixtures": compact_pairs,
        "policy_identity": POLICY_IDENTITY,
        "population_kind": "hermetic_development",
        "populations": {
            "hermetic_development": {
                "count": _measured(admitted["fixture_count"], unit="count"),
                "minimum": 1,
                "status": "sealed",
            },
            "historical_exact_tree_replay": {
                "count": _unavailable("not_sealed"),
                "minimum": 20,
                "status": "unavailable",
            },
            "new_live_shadow_canary": {
                "count": _unavailable("not_sealed"),
                "enrollment_deadline": _unavailable("not_sealed"),
                "minimum": 10,
                "status": "unavailable",
            },
        },
        "program_id": "agent-supervisor-efficiency-and-state-hardening-v1",
        "promotion_without_paired_campaign": False,
        "recipes": list(admitted["recipes"]),
        "repository_commit": REPOSITORY_COMMIT,
        "repository_tree": REPOSITORY_TREE,
        "required_metrics": list(REQUIRED_METRICS),
        "schema": MANIFEST_SCHEMA,
        "schema_version": 1,
        "simulated_as_live": False,
        "status": "sealed",
        "task_id": TASK_ID,
        "vectors_identity": admitted["vectors_identity"],
    }
    payload["explicit_unavailable_fields"] = list(collect_unavailable_fields(payload))
    body = {key: value for key, value in payload.items() if key != "manifest_cid"}
    payload["manifest_cid"] = content_identity(body)
    return canonical_object(payload)


def seal_artifacts(
    recipes: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    campaign = run_paired_campaign(recipes)
    manifest = build_context_pack_manifest(campaign)
    return {"campaign": campaign, "manifest": manifest}


def write_sealed_artifacts(
    directory: Path | None = None,
    *,
    campaign: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    target = directory or PACKAGE_DIR
    payload = campaign if campaign is not None else run_paired_campaign()
    manifest = build_context_pack_manifest(payload)
    path = target / "context_pack_manifest.json"
    path.write_text(pretty_json(manifest), encoding="utf-8")
    return {"manifest": path}


def verify_sealed_artifacts(
    directory: Path | None = None,
    *,
    campaign: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    target = directory or PACKAGE_DIR
    payload = campaign if campaign is not None else run_paired_campaign()
    manifest = build_context_pack_manifest(payload)
    path = target / "context_pack_manifest.json"
    try:
        observed = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContextPackBenchmarkError("context_pack_manifest.json is unreadable") from exc
    expected = pretty_json(manifest)
    if observed != expected:
        raise ContextPackBenchmarkError(
            "context_pack_manifest.json drifted from the sealed campaign"
        )
    encoded = json.loads(observed)
    if encoded["manifest_cid"] != manifest["manifest_cid"]:
        raise ContextPackBenchmarkError("manifest_cid drifted from canonical identity")
    admit_campaign_result(payload)
    return {
        "fixture_count": payload["fixture_count"],
        "manifest_cid": manifest["manifest_cid"],
        "required_metrics": list(REQUIRED_METRICS),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ASEH-035 ContextPack benchmark")
    parser.add_argument("--write", action="store_true", help="seal the campaign manifest")
    parser.add_argument("--check", action="store_true", help="verify the sealed manifest")
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
                "negative_cases": campaign["negative_cases"],
                "required_metrics": campaign["required_metrics"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
